import argparse, glob, json, os, random, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------- Dataset -------------------------
class NPZWindowsDataset(Dataset):
    """
    读取我们生成的 .npz：
      - amp: (N, win, 30, C)  ; phase: 同形状
      - label: int（手势类别）
      - meta: JSON 字符串（含 user/gesture/face 等）
    将 (amp, phase) 在通道维拼接 => (N, win, 30, 2C)
    并 reshape 成 TCN 需要的 (N, channels, time) = (N, 30*2C, win)
    """
    def __init__(self, file_items):
        # file_items: list of (path, take_indices or None)
        self.items = []
        for p, idx_list in file_items:
            with np.load(p, allow_pickle=True) as z:
                n = z["amp"].shape[0]
            if idx_list is None:
                idx_list = list(range(n))
            self.items += [(p, i) for i in idx_list]
        # 简单缓存，避免频繁 IO
        self.cache = {}

    def _load_file(self, p):
        if p in self.cache:
            return self.cache[p]
        z = np.load(p, allow_pickle=True)
        amp, phase = z["amp"], z["phase"]
        label = int(z["label"])
        meta = json.loads(str(z["meta"]))
        self.cache = {p: (amp, phase, label, meta)}  # 简易单文件缓存
        return amp, phase, label, meta

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        p, i = self.items[idx]
        amp, phase, label, meta = self._load_file(p)
        a, ph = amp[i], phase[i]                  # (win, 30, C)
        x = np.concatenate([a, ph], axis=-1)      # (win, 30, 2C)
        # (time, features)->(channels, time): features=30*2C
        win, sc, ch2 = x.shape
        x = x.reshape(win, sc*ch2).T              # (sc*ch2, win)
        x = torch.from_numpy(x).float()
        y = torch.tensor(label, dtype=torch.long)
        # 额外返回 meta 里 user 便于分组评估
        return x, y, meta

# ------------------------- Model (TCN baseline) -------------------------
class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, d=1, p=0.1):
        super().__init__()
        pad = (k-1)*d
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=pad, dilation=d),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=d),
            nn.ReLU(inplace=True),
        )
        self.short = nn.Conv1d(c_in, c_out, 1) if c_in!=c_out else nn.Identity()
    def forward(self, x):
        return self.net(x) + self.short(x)

class HAR_TCN(nn.Module):
    def __init__(self, c_in, n_classes, p=0.1):
        super().__init__()
        self.b1 = TCNBlock(c_in, 256, d=1, p=p)
        self.b2 = TCNBlock(256, 256, d=2, p=p)
        self.b3 = TCNBlock(256, 256, d=4, p=p)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_classes)
        )
    def forward(self, x, return_feat=False):  # x: (B, C, T)
        h = self.b3(self.b2(self.b1(x)))
        feat = torch.mean(h, dim=-1)  # (B, 256)
        logits = self.head[2:](h)     # same as pooling->MLP
        if return_feat: return logits, feat
        return logits

# --------------------- Optional: Graph/Laplacian reg ---------------------
def laplacian_smoothness_loss(feat, y=None, k=8, tau=0.2):
    """
    简易谱平滑正则：构图 A = relu(cos_sim - tau)，L = D - A，min Tr(F^T L F)
    feat: (B, D)
    """
    with torch.no_grad():
        f = torch.nn.functional.normalize(feat, dim=1)
        S = torch.einsum("bd,cd->bc", f, f)               # cosine sim
        S.fill_diagonal_(0.0)
        # 取 top-k
        vals, idx = torch.topk(S, k=k, dim=1)
        A = torch.zeros_like(S)
        A.scatter_(1, idx, torch.relu(vals - tau))
        A = torch.maximum(A, A.t())                       # symmetrize
        D = torch.diag(A.sum(1))
        L = D - A
    # Tr(F^T L F) = sum_i,j L_ij <f_i, f_j>
    return torch.einsum("bi,ij,bj->", feat, L, feat) / feat.size(0)

# ------------------------- Utilities -------------------------
def list_npz(globs):
    paths = []
    for g in globs:
        paths += [str(p) for p in glob.glob(g)]
    paths = sorted(set(paths))
    if not paths: raise ValueError(f"No files matched: {globs}")
    return paths

def split_source(paths, val_ratio=0.2, group_by_user=True, seed=0):
    rng = random.Random(seed)
    if not group_by_user:
        rng.shuffle(paths)
        n_val = max(1, int(len(paths)*val_ratio))
        return paths[n_val:], paths[:n_val]
    # group by meta['user']
    user2files = {}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        u = int(meta["user"])
        user2files.setdefault(u, []).append(p)
    users = sorted(user2files.keys())
    rng.shuffle(users)
    n_val_u = max(1, int(len(users)*val_ratio))
    val_users = set(users[:n_val_u])
    train, val = [], []
    for u, fls in user2files.items():
        (val if u in val_users else train).extend(fls)
    return sorted(set(train)), sorted(set(val))

def build_items(paths):
    # 每个文件的所有窗口都用（也可在此进行下采样/采样策略）
    return [(p, None) for p in paths]

# --------------------------- Train & Eval ---------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    src_paths = list_npz(args.source_glob)
    if args.val_from_source:
        train_paths, val_paths = split_source(src_paths, val_ratio=args.val_ratio, seed=args.seed)
    else:
        train_paths, val_paths = src_paths, []
    tgt_paths = list_npz(args.target_glob) if args.target_glob else []

    # peek one file to get input channels & num_classes
    z0 = np.load(src_paths[0], allow_pickle=True)
    amp0, phase0 = z0["amp"], z0["phase"]
    n_win, win, sc, C = amp0.shape
    c_in = sc * (2*C)     # 30 * (幅相*通道数)
    n_classes = int(z0["label"].item()) + 1  # 若你的 label 从 0 开始，改成固定常数
    # 更稳妥：从源域所有文件收集出现过的 label
    labels = set()
    for p in src_paths:
        labels.add(int(np.load(p, allow_pickle=True)["label"]))
    n_classes = max(labels)+1

    # dataloaders
    train_ds = NPZWindowsDataset(build_items(train_paths))
    val_ds   = NPZWindowsDataset(build_items(val_paths)) if val_paths else None
    test_ds  = NPZWindowsDataset(build_items(tgt_paths)) if tgt_paths else None
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0) if val_ds else None
    test_loader  = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=0) if test_ds else None

    # model / opt
    model = HAR_TCN(c_in=c_in, n_classes=n_classes, p=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ce = nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None
    patience = args.patience
    for epoch in range(1, args.epochs+1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)         # x: (B, C, T)
            logits, feat = model(x, return_feat=True)
            loss = ce(logits, y)
            if args.lambda_spec > 0:
                loss = loss + args.lambda_spec * laplacian_smoothness_loss(feat)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.item()) * x.size(0)
            total += x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        train_acc = correct/total; train_loss = loss_sum/total

        # 源域内验证（用于早停）
        val_acc = None
        if val_loader is not None:
            model.eval(); v_tot=v_cor=0; v_loss=0.0
            with torch.no_grad():
                for x,y,_ in val_loader:
                    x,y = x.to(device), y.to(device)
                    logits = model(x)
                    v_loss += float(ce(logits, y).item())*x.size(0)
                    v_tot  += x.size(0)
                    v_cor  += (logits.argmax(1)==y).sum().item()
            val_loss = v_loss/v_tot; val_acc = v_cor/v_tot

            # 早停逻辑：按 val_acc
            if val_acc > best_val:
                best_val = val_acc; best_state = {k:v.cpu() for k,v in model.state_dict().items()}
                patience = args.patience
            else:
                patience -= 1
                if patience <= 0:
                    print(f"[Early Stop] epoch={epoch}")
                    break

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} acc={train_acc:.3f}"
              + (f" | val_acc={val_acc:.3f}" if val_loader else ""))

    # 恢复最佳模型（源域验证）
    if best_state is not None:
        model.load_state_dict(best_state)

    # 目标域测试（零目标域样本）
    if test_loader is not None:
        model.eval(); tot=cor=0
        all_y, all_p = [], []
        with torch.no_grad():
            for x,y,_ in test_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(1)
                tot += x.size(0); cor += (pred==y).sum().item()
                all_y.append(y.cpu().numpy()); all_p.append(pred.cpu().numpy())
        test_acc = cor/tot
        y_true = np.concatenate(all_y); y_pred = np.concatenate(all_p)
        # macro-F1
        from sklearn.metrics import f1_score, classification_report
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"[TARGET TEST] acc={test_acc:.3f} macroF1={macro_f1:.3f}")
        print(classification_report(y_true, y_pred, digits=3))
    else:
        print("No target set; training finished.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_glob", nargs="+", required=True, help="如 data/R1/*.npz 或多段通配符")
    ap.add_argument("--target_glob", nargs="+", default=None, help="目标域仅测试（零目标域）")
    ap.add_argument("--val_from_source", action="store_true", help="源域划分一部分做验证（早停）")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lambda_spec", type=float, default=0.0, help="谱/图平滑正则权重（0 关闭）")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    run(args)
