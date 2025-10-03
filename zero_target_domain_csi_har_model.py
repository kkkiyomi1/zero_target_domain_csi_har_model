"""
Zero-Target-Domain CSI-HAR: Training code skeleton (PyTorch)

Implements the following, aligned with the user's theory doc:
- Differentiable preprocessing: amplitude |X| and detrended phase \hat{\Phi}
- Shared TCN backbone producing embeddings z (Eq. 3–4 overview)
- Symmetric contrastive loss (InfoNCE) between (z_A, z_\Phi) (Eq. 5 style)
- Group-consistency regularizer L_G for structured shifts (local subcarrier permutations,
  amplitude scaling, low-frequency perturbations)
- Within-class sliding-graph alignment L_loc with EMA prototypes (Eq. 8–10)
- Spectral–Cheeger discriminant L_chg (normalized-cut surrogate) (Eq. 11–16)
- Overall objective L = L_CE + λ_NCE L_NCE + λ_G L_G + λ_loc L_loc + λ_chg L_chg (Eq. 14)
- Target-free early stopping via Cheeger–consistency monitor M_t (Sec. 4.4)

Notes
-----
* Keep batch sizes moderate (e.g., 64–128) so graph terms are tractable.
* This is a solid, readable starting point; you can refine models/augmentations as needed.
* Dataset loader supports either a directory of .npz shards or a single monolithic .npz.
  Expected keys in .npz: either
    - {'X': complex64 array (N,T,K), 'y': int64 (N,)}   OR
    - {'A': float32 (N,T,K), 'Phi_hat': float32 (N,T,K), 'y': int64 (N,)}
  Optional: 'domain' (N,) for multi-source DG bookkeeping.

Author: you + ChatGPT (2025)
"""
from __future__ import annotations
import math
import os
import glob
import random
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Utilities
# ---------------------------

def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.copy())


def unwrap_phase(phi: np.ndarray) -> np.ndarray:
    """Naive 1D unwrap along time for each subcarrier.
    phi: (T, K) or (N,T,K)
    """
    return np.unwrap(phi, axis=-2)


def detrend_linear(phi: np.ndarray) -> np.ndarray:
    """Per-subcarrier linear detrend along time."""
    if phi.ndim == 3:
        N, T, K = phi.shape
        t = np.arange(T, dtype=np.float32)
        t = (t - t.mean()) / (t.std() + 1e-6)
        Xt = np.stack([t, np.ones_like(t)], axis=1)  # (T,2)
        # Closed-form (a,b) via least squares per (N,K)
        # Solve [a,b] = (Xt^T Xt)^{-1} Xt^T y
        XtTXt_inv = np.linalg.inv(Xt.T @ Xt)  # (2,2)
        H = XtTXt_inv @ Xt.T                   # (2,T)
        # Apply per (N,K)
        a = np.zeros((N, K), dtype=np.float32)
        b = np.zeros((N, K), dtype=np.float32)
        for n in range(N):
            Y = phi[n]  # (T,K)
            # (2,T) @ (T,K) -> (2,K)
            ab = H @ Y
            a[n] = ab[0]
            b[n] = ab[1]
        trend = (t[:, None] * a[:, None, :]) + b[:, None, :]
        return phi - trend
    elif phi.ndim == 2:
        T, K = phi.shape
        t = np.arange(T, dtype=np.float32)
        t = (t - t.mean()) / (t.std() + 1e-6)
        Xt = np.stack([t, np.ones_like(t)], axis=1)
        XtTXt_inv = np.linalg.inv(Xt.T @ Xt)
        H = XtTXt_inv @ Xt.T
        ab = H @ phi  # (2,K)
        trend = (t[:, None] * ab[0][None, :]) + ab[1][None, :]
        return phi - trend
    else:
        raise ValueError("phi must be 2D or 3D array")


# ---------------------------
# Dataset
# ---------------------------
class CSINpzDataset(Dataset):
    """Loads CSI from .npz (directory of shards or single file).

    Two formats are supported:
      1) {'X': complex64 (N,T,K), 'y': int64 (N,)}  # raw complex CSI
      2) {'A': float32 (N,T,K), 'Phi_hat': float32 (N,T,K), 'y': int64 (N,)}

    Also tolerated (auto-detected):
      - 'Phi' raw phase (we unwrap+detrend)
      - label keys in {'y','label','labels','Y'}
      - amplitude keys in {'A','amp','amplitude'}
      - phase-hat keys in {'Phi_hat','phi_hat','phase_hat','phase_detrended'}
    """
    def __init__(self, path: str, split: Optional[str] = None, split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42):
        super().__init__()
        self.files: List[str] = []

        def first_present_key(d, candidates: List[str]) -> Optional[str]:
            for k in candidates:
                if k in d:
                    return k
            return None

        def get_label_array(d) -> Optional[np.ndarray]:
            k = first_present_key(d, ['y', 'label', 'labels', 'Y'])
            return None if k is None else d[k]

        if os.path.isdir(path):
            self.files = sorted(glob.glob(os.path.join(path, "*.npz")))
            if not self.files:
                raise FileNotFoundError(f"No .npz files found in {path}")

            A_list: List[np.ndarray] = []
            P_list: List[np.ndarray] = []
            y_list: List[np.ndarray] = []
            bad_files: List[Tuple[str, List[str]]] = []

            for f in self.files:
                d = np.load(f, allow_pickle=True)
                keys_here = list(d.files)
                try:
                    if 'X' in d.files:
                        X = d['X']
                        y = get_label_array(d)
                        if y is None:
                            raise KeyError("missing label (y/label/labels/Y)")
                        A = np.abs(X).astype(np.float32)
                        Phi = np.angle(X).astype(np.float32)
                        Phi = unwrap_phase(Phi)
                        Phi_hat = detrend_linear(Phi).astype(np.float32)
                    else:
                        kA = first_present_key(d, ['A', 'amp', 'amplitude'])
                        kPhat = first_present_key(d, ['Phi_hat', 'phi_hat', 'phase_hat', 'phase_detrended'])
                        kPraw = first_present_key(d, ['Phi', 'phi', 'phase'])
                        y = get_label_array(d)
                        if kA is None or (kPhat is None and kPraw is None) or y is None:
                            raise KeyError("need A and (Phi_hat or Phi) and y")
                        A = d[kA].astype(np.float32)
                        if kPhat is not None:
                            Phi_hat = d[kPhat].astype(np.float32)
                        else:
                            Praw = d[kPraw].astype(np.float32)
                            Praw = unwrap_phase(Praw)
                            Phi_hat = detrend_linear(Praw).astype(np.float32)
                    # ---- shape & dtype sanitation ----
                    if A.dtype == np.object_ or Phi_hat.dtype == np.object_:
                        raise ValueError("object-dtype arrays not supported; please save dense numeric arrays")
                    # (T,K) -> (1,T,K)
                    if A.ndim == 2 and Phi_hat.ndim == 2:
                        A = A[None, ...]
                        Phi_hat = Phi_hat[None, ...]
                    if A.ndim != 3 or Phi_hat.ndim != 3:
                        raise ValueError(f"expected (N,T,K) per shard; got A{A.shape} / P{Phi_hat.shape}")
                    # labels -> (N,)
                    y = np.asarray(y)
                    if y.ndim == 0:
                        y = np.repeat(y[None], A.shape[0], axis=0)
                    elif y.ndim == 1 and y.shape[0] != A.shape[0]:
                        if y.shape[0] == 1:
                            y = np.repeat(y, A.shape[0], axis=0)
                        else:
                            raise ValueError(f"label length {y.shape[0]} != N {A.shape[0]}")
                    if A.shape != Phi_hat.shape:
                        raise ValueError(f"shape mismatch A{A.shape} vs Phi_hat{Phi_hat.shape}")
                    A_list.append(A)
                    P_list.append(Phi_hat)
                    y_list.append(y.astype(np.int64))
                except Exception:
                    bad_files.append((f, keys_here))
                    continue

            if not A_list:
                hint_lines = [f"- {bf}: keys={keys}" for bf, keys in bad_files[:10]]
                hint = "\n".join(hint_lines)
                raise RuntimeError(
                    "No usable shards found. Expected keys like X/y or A/(Phi_hat|Phi)+y.\n"
                    "Here are some examples of shards I could not parse:\n" + hint
                )
            # concatenate across shards
            A = np.concatenate(A_list, axis=0)
            Phi_hat = np.concatenate(P_list, axis=0)
            y = np.concatenate(y_list, axis=0)
            print(f"[CSINpzDataset] Loaded {len(A_list)} shard(s); total N={len(y)}")
        else:
            data = np.load(path, allow_pickle=True)
            if 'X' in data.files:
                X = data['X']
                y = data['y'] if 'y' in data.files else (
                    data['label'] if 'label' in data.files else (
                        data['labels'] if 'labels' in data.files else (
                            data['Y'] if 'Y' in data.files else None)))
                if y is None:
                    raise KeyError("missing label (y/label/labels/Y)")
                A = np.abs(X).astype(np.float32)
                Phi = np.angle(X).astype(np.float32)
                Phi = unwrap_phase(Phi)
                Phi_hat = detrend_linear(Phi)
            else:
                # amplitude + phase
                def get_first(d, ks):
                    for k in ks:
                        if k in d.files:
                            return k
                    return None
                kA = get_first(data, ['A', 'amp', 'amplitude'])
                if kA is None:
                    raise KeyError("No amplitude key found (A/amp/amplitude)")
                A = data[kA].astype(np.float32)
                kPhat = get_first(data, ['Phi_hat', 'phi_hat', 'phase_hat', 'phase_detrended'])
                if kPhat is not None:
                    Phi_hat = data[kPhat].astype(np.float32)
                else:
                    kPraw = get_first(data, ['Phi', 'phi', 'phase'])
                    if kPraw is None:
                        raise KeyError("No phase key found (Phi_hat or Phi)")
                    Phi = data[kPraw].astype(np.float32)
                    Phi = unwrap_phase(Phi)
                    Phi_hat = detrend_linear(Phi)
                y = data['y'] if 'y' in data.files else (
                    data['label'] if 'label' in data.files else (
                        data['labels'] if 'labels' in data.files else (
                            data['Y'] if 'Y' in data.files else None)))
                if y is None:
                    raise KeyError("missing label (y/label/labels/Y)")
        # ---- single-file sanitation ----
        if A.ndim == 2 and Phi_hat.ndim == 2:
            A = A[None, ...]
            Phi_hat = Phi_hat[None, ...]
        if A.ndim != 3 or Phi_hat.ndim != 3:
            raise ValueError(f"expected (N,T,K) got A{A.shape} / P{Phi_hat.shape}")
        y = np.asarray(y)
        if y.ndim == 0:
            y = np.repeat(y[None], A.shape[0], axis=0)
        elif y.ndim == 1 and y.shape[0] != A.shape[0]:
            if y.shape[0] == 1:
                y = np.repeat(y, A.shape[0], axis=0)
            else:
                raise ValueError(f"label length {y.shape[0]} != N {A.shape[0]}")
        assert A.shape == Phi_hat.shape, "A and Phi_hat must have the same shape"
        self.A = A  # (N,T,K)
        self.Phi_hat = Phi_hat
        self.y = y.astype(np.int64)

        # optional train/val/test split indices
        N = len(self.y)
        idx = np.arange(N)
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
        n_tr = int(N * split_ratio[0])
        n_va = int(N * split_ratio[1])
        tr_idx = idx[:n_tr]
        va_idx = idx[n_tr:n_tr+n_va]
        te_idx = idx[n_tr+n_va:]
        if split == 'train':
            self.sel = tr_idx
        elif split == 'val':
            self.sel = va_idx
        elif split == 'test' or split is None:
            self.sel = te_idx if split == 'test' else idx
        else:
            raise ValueError("split must be one of {None,'train','val','test'}")

    def __len__(self):
        return len(self.sel)

    def __getitem__(self, i):
        j = self.sel[i]
        A = self.A[j]      # (T,K)
        P = self.Phi_hat[j]
        y = self.y[j]
        # convert to torch (C=2 stacked along channel-of-subcarriers)
        # We'll pack as (2, T, K)
        A = _to_tensor(A).float()
        P = _to_tensor(P).float()
        return {'A': A, 'P': P, 'y': int(y)}


# ---------------------------
# Group transformations G
# ---------------------------
@dataclass
class GParams:
    permute_groups: int = 4      # number of local groups along K to shuffle
    amp_jitter_std: float = 0.05 # log-normal-like scaling via exp(N(0,std))
    lpf_kernel: int = 51         # low-pass filter length for phase drift
    lpf_sigma: float = 10.0      # std for Gaussian kernel


def gaussian_kernel1d(L: int, sigma: float) -> torch.Tensor:
    x = torch.arange(L, dtype=torch.float32)
    x = x - x.mean()
    k = torch.exp(-0.5 * (x / (sigma + 1e-6)) ** 2)
    k = k / (k.sum() + 1e-6)
    return k


def apply_group_transform(A: torch.Tensor, P: torch.Tensor, g: GParams) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a random transform gπ,s,δ on (A,P).
    A,P: (T,K)
    Returns transformed (A', P').
    """
    T, K = A.shape
    # 1) local subcarrier permutation by groups
    Gg = max(1, g.permute_groups)
    group_size = K // Gg
    idx = torch.arange(K)
    if group_size > 1:
        groups = [idx[i*group_size:(i+1)*group_size] for i in range(Gg-1)]
        groups.append(idx[(Gg-1)*group_size:])
        random.shuffle(groups)
        perm = torch.cat(groups, dim=0)
        A = A[:, perm]
        P = P[:, perm]
    # 2) amplitude scaling
    s = math.exp(random.gauss(0.0, g.amp_jitter_std))
    A = A * s
    # 3) low-frequency phase drift (additive, smoothed noise along time)
    ker = gaussian_kernel1d(g.lpf_kernel, g.lpf_sigma).to(A.device)
    noise = torch.randn(T)
    drift = F.conv1d(noise.view(1,1,-1), ker.view(1,1,-1), padding=ker.numel()//2).view(-1)
    drift = drift / (drift.std() + 1e-6)
    P = P + drift[:, None]
    return A, P


# ---------------------------
# Model: TCN backbone + heads
# ---------------------------
class TCNBlock(nn.Module):
    def __init__(self, ch: int, hid: int, ksz: int = 3, dil: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (ksz - 1) * dil
        self.conv1 = nn.Conv1d(ch, hid, ksz, padding=pad, dilation=dil)
        self.bn1 = nn.BatchNorm1d(hid)
        self.conv2 = nn.Conv1d(hid, hid, ksz, padding=pad, dilation=dil)
        self.bn2 = nn.BatchNorm1d(hid)
        self.proj = nn.Conv1d(ch, hid, 1) if ch != hid else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.dropout(y)
        y = self.bn2(self.conv2(y))
        out = F.relu(y + self.proj(x))
        return out


class EncoderTCN(nn.Module):
    def __init__(self, K: int, d_embed: int = 128, base: int = 64, dilations: Tuple[int,...] = (1,2,4,8), dropout: float = 0.1):
        super().__init__()
        # We treat (A,P) as two modalities across K subcarriers -> channels = 2*K
        ch_in = 2 * K
        layers: List[nn.Module] = []
        ch = ch_in
        for d in dilations:
            layers.append(TCNBlock(ch, base, ksz=3, dil=d, dropout=dropout))
            ch = base
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # global avg over time
        )
        self.proj = nn.Linear(ch, d_embed)

    def forward(self, A: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        # A,P: (B,T,K)
        B, T, K = A.shape
        x = torch.stack([A, P], dim=1)               # (B, 2, T, K)
        x = x.permute(0, 3, 1, 2).contiguous()       # (B, K, 2, T)
        x = x.view(B, 2*K, T)                        # (B, 2K, T)
        y = self.tcn(x)                              # (B, C, T)
        y = self.head(y).squeeze(-1)                 # (B, C)
        z = F.normalize(self.proj(y), dim=-1)        # (B, d)
        return z


class Classifier(nn.Module):
    def __init__(self, d_embed: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_embed, n_classes)
    def forward(self, z):
        return self.fc(z)


# ---------------------------
# Losses
# ---------------------------
class InfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.t = temperature
    def forward(self, zA: torch.Tensor, zP: torch.Tensor) -> torch.Tensor:
        # symmetric NT-Xent between two views
        zA = F.normalize(zA, dim=-1)
        zP = F.normalize(zP, dim=-1)
        B, D = zA.shape
        logits = zA @ zP.T / self.t  # (B,B)
        labels = torch.arange(B, device=zA.device)
        loss1 = F.cross_entropy(logits, labels)
        loss2 = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss1 + loss2)


def group_consistency_loss(encoder: EncoderTCN, A: torch.Tensor, P: torch.Tensor, gcfg: GParams, n_samples: int = 1) -> torch.Tensor:
    """L_G = E_g || h(g(X)) - h(X) ||^2 over encoder output z (h∘f).
    """
    with torch.no_grad():
        z_ref = encoder(A, P).detach()
    B = A.size(0)
    losses = []
    for _ in range(n_samples):
        A2 = []
        P2 = []
        for b in range(B):
            a, p = apply_group_transform(A[b], P[b], gcfg)
            A2.append(a.unsqueeze(0))
            P2.append(p.unsqueeze(0))
        A2 = torch.cat(A2, dim=0)
        P2 = torch.cat(P2, dim=0)
        z_aug = encoder(A2, P2)
        losses.append(((z_aug - z_ref) ** 2).sum(dim=1).mean())
    return sum(losses) / len(losses)


def _pairwise_sqdist(x: torch.Tensor) -> torch.Tensor:
    # x: (N,d) -> (N,N) squared Euclidean
    x2 = (x**2).sum(dim=1, keepdim=True)
    d = x2 + x2.T - 2 * (x @ x.T)
    return d.clamp_min_(0.0)


def _knn_mask(d2: torch.Tensor, k: int) -> torch.Tensor:
    # returns mask (N,N) indicating knn neighbors (excluding self)
    N = d2.size(0)
    vals, idx = torch.topk(-d2 + torch.eye(N, device=d2.device)*(-1e9), k=k+1, dim=1)
    # exclude self (closest by construction)
    idx = idx[:, 1:]
    mask = torch.zeros((N, N), device=d2.device, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    # make symmetric
    mask = mask | mask.T
    return mask


def build_affinity(z: torch.Tensor, k: int = 10, sigma: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (W,D,L) where W is symmetric kNN Gaussian affinity.
    z: (N,d)
    """
    with torch.no_grad():
        d2 = _pairwise_sqdist(z)
        if sigma is None:
            # median heuristic on within-batch distances
            tri = d2[torch.triu(torch.ones_like(d2, dtype=torch.bool), diagonal=1)]
            sigma = torch.sqrt(torch.median(tri) / 2.0 + 1e-6).item()
        W = torch.exp(-d2 / (2.0 * (sigma**2 + 1e-8)))
        mask = _knn_mask(d2, k)
        W = W * mask.float()
        # symmetrize + remove self-loops
        W = 0.5 * (W + W.T)
        W.fill_diagonal_(0.0)
        d = W.sum(dim=1)
        D = torch.diag(d)
        L = D - W
    return W, D, L


def cheeger_loss(z: torch.Tensor, y: torch.Tensor, k: int = 10, sigma: Optional[float] = None) -> torch.Tensor:
    """L_chg = sum_c y_c^T L y_c / y_c^T D y_c (normalized cut per class)
    Computed on the batch graph.
    """
    W, D, L = build_affinity(z.detach(), k=k, sigma=sigma)  # stop gradients through topology
    num = 0.0
    den = 0.0
    classes = torch.unique(y)
    loss = 0.0
    for c in classes:
        yc = (y == c).float()
        num = yc @ (L @ yc)
        den = yc @ (D @ yc) + 1e-8
        loss = loss + num / den
    return loss


def within_class_alignment(z: torch.Tensor, y: torch.Tensor, k: int = 10, sigma: Optional[float] = None, mu: float = 0.0,
                            prototypes: Optional[torch.Tensor] = None) -> torch.Tensor:
    """L_loc = sum_c tr(Z_c^T L_c Z_c) + (mu/2) ||Z_c - 1 mu_c^T||_F^2 (on batch graph)
    We approximate tr(Z^T L Z) by 0.5 * sum_ij W_ij ||z_i - z_j||^2 using the same W mask
    inside each class.
    """
    d2 = _pairwise_sqdist(z.detach())  # use detached z for graph weights
    if sigma is None:
        tri = d2[torch.triu(torch.ones_like(d2, dtype=torch.bool), diagonal=1)]
        sigma = math.sqrt(tri.median().item() / 2.0 + 1e-6)
    W = torch.exp(-d2 / (2.0 * (sigma**2 + 1e-8)))
    mask = _knn_mask(d2, k)
    W = W * mask.float()
    W = 0.5 * (W + W.T)
    W.fill_diagonal_(0.0)
    loss = torch.zeros((), device=z.device)
    classes = torch.unique(y)
    for c in classes:
        idx = (y == c).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() <= 1:
            continue
        Wc = W[idx][:, idx]
        # Laplacian smoothness surrogate: 0.5 * sum_{i,j in c} W_ij ||z_i - z_j||^2
        Zc = z[idx]
        diff = _pairwise_sqdist(Zc)
        loc = 0.5 * (Wc * diff).sum() / (idx.numel() + 1e-6)
        if mu > 0.0 and prototypes is not None:
            mu_c = prototypes[c]
            loc = loc + 0.5 * mu * ((Zc - mu_c) ** 2).sum(dim=1).mean()
        loss = loss + loc
    return loss


# ---------------------------
# Training harness
# ---------------------------
@dataclass
class TrainConfig:
    data_path: str
    n_classes: int
    batch_size: int = 128
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    lambda_nce: float = 0.1
    lambda_g: float = 0.1
    lambda_loc: float = 0.1
    lambda_chg: float = 0.1
    k_graph: int = 10
    d_embed: int = 128
    ema_momentum: float = 0.99
    temperature: float = 0.2
    # Early stopping monitor
    es_beta: float = 0.2  # EMA smoothing factor β
    es_alpha: float = 1.0 # weight for δ_t in B_t = L_chg_val + α δ_t
    es_patience: int = 3  # consecutive upticks to stop
    es_eta: float = 0.0   # minimal uptick to count


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        ds_tr = CSINpzDataset(cfg.data_path, split='train')
        ds_va = CSINpzDataset(cfg.data_path, split='val')
        self.train_loader = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)
        self.val_loader = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0)
        # infer K and T
        sample = next(iter(self.train_loader))
        T, K = sample['A'].shape[-2:]
        self.encoder = EncoderTCN(K, d_embed=cfg.d_embed).to(self.device)
        self.cls = Classifier(cfg.d_embed, cfg.n_classes).to(self.device)
        self.nce = InfoNCE(cfg.temperature)
        self.opt = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.cls.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.gcfg = GParams()
        # prototypes
        self.prototypes = torch.zeros((cfg.n_classes, cfg.d_embed), device=self.device)
        self.proto_counts = torch.zeros(cfg.n_classes, device=self.device)
        self.best_state = None

    def _update_prototypes(self, z: torch.Tensor, y: torch.Tensor):
        # EMA per class
        with torch.no_grad():
            for c in torch.unique(y):
                idx = (y == c).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue
                zc = z[idx].mean(dim=0)
                m = self.cfg.ema_momentum
                self.prototypes[c] = m * self.prototypes[c] + (1 - m) * zc
                self.proto_counts[c] = self.proto_counts[c] + 1

    def _delta_t(self, A: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        # RMS group-consistency deviation δ_t ≈ sqrt(E ||h(g(X)) - h(X)||^2)
        with torch.no_grad():
            z_ref = self.encoder(A, P)
            # one sample of g per item
            z_shift = []
            for b in range(A.size(0)):
                a2, p2 = apply_group_transform(A[b], P[b], self.gcfg)
                z_shift.append(self.encoder(a2.unsqueeze(0), p2.unsqueeze(0)))
            z_shift = torch.cat(z_shift, dim=0)
            delta = ((z_shift - z_ref) ** 2).sum(dim=1).mean().sqrt()
        return delta

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.encoder.train(); self.cls.train()
        running = {'loss':0.0, 'LCE':0.0, 'LNCE':0.0, 'LG':0.0, 'Lloc':0.0, 'Lchg':0.0}
        for batch in self.train_loader:
            A = batch['A'].to(self.device)  # (B,T,K)
            P = batch['P'].to(self.device)
            y = torch.tensor(batch['y'], device=self.device)

            # forward
            z = self.encoder(A, P)
            logits = self.cls(z)
            LCE = F.cross_entropy(logits, y)

            # symmetric InfoNCE between two stochastic views of (A,P)
            # Use two independent augmentations of inputs (simulate view agreement)
            A1, P1 = [], []
            A2, P2 = [], []
            for b in range(A.size(0)):
                a1, p1 = apply_group_transform(A[b], P[b], self.gcfg)
                a2, p2 = apply_group_transform(A[b], P[b], self.gcfg)
                A1.append(a1.unsqueeze(0)); P1.append(p1.unsqueeze(0))
                A2.append(a2.unsqueeze(0)); P2.append(p2.unsqueeze(0))
            A1 = torch.cat(A1, dim=0); P1 = torch.cat(P1, dim=0)
            A2 = torch.cat(A2, dim=0); P2 = torch.cat(P2, dim=0)
            zA = self.encoder(A1, P)   # treat (A,P) as two correlated views
            zP = self.encoder(A, P2)
            LNCE = self.nce(zA, zP)

            # group consistency LG
            LG = group_consistency_loss(self.encoder, A, P, self.gcfg, n_samples=1)

            # graph terms built on current z
            Lloc = within_class_alignment(z, y, k=self.cfg.k_graph, sigma=None, mu=0.0, prototypes=self.prototypes)
            Lchg = cheeger_loss(z, y, k=self.cfg.k_graph, sigma=None)

            loss = LCE + self.cfg.lambda_nce*LNCE + self.cfg.lambda_g*LG + self.cfg.lambda_loc*Lloc + self.cfg.lambda_chg*Lchg

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.cls.parameters()), 5.0)
            self.opt.step()

            self._update_prototypes(z.detach(), y)

            running['loss'] += loss.item()
            running['LCE']  += LCE.item()
            running['LNCE'] += LNCE.item()
            running['LG']   += LG.item()
            running['Lloc'] += Lloc.item()
            running['Lchg'] += Lchg.item()
        n = len(self.train_loader)
        return {k: v/max(n,1) for k,v in running.items()}

    @torch.no_grad()
    def eval_val(self) -> Dict[str, float]:
        self.encoder.eval(); self.cls.eval()
        all_logits = []
        all_y = []
        Lchg_vals = []
        deltas = []
        for batch in self.val_loader:
            A = batch['A'].to(self.device)
            P = batch['P'].to(self.device)
            y = torch.tensor(batch['y'], device=self.device)
            z = self.encoder(A, P)
            logits = self.cls(z)
            all_logits.append(logits)
            all_y.append(y)
            Lchg_vals.append(cheeger_loss(z, y, k=self.cfg.k_graph).item())
            deltas.append(self._delta_t(A, P).item())
        logits = torch.cat(all_logits, dim=0)
        y = torch.cat(all_y, dim=0)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        # macro-F1
        C = self.cfg.n_classes
        f1s = []
        for c in range(C):
            tp = ((pred==c) & (y==c)).sum().item()
            fp = ((pred==c) & (y!=c)).sum().item()
            fn = ((pred!=c) & (y==c)).sum().item()
            prec = tp / (tp+fp+1e-8)
            rec  = tp / (tp+fn+1e-8)
            f1 = 2*prec*rec/(prec+rec+1e-8)
            f1s.append(f1)
        macro_f1 = float(np.mean(f1s))
        Lchg_mean = float(np.mean(Lchg_vals))
        delta_mean = float(np.mean(deltas))
        return {'val_acc':acc, 'val_macro_f1':macro_f1, 'val_Lchg':Lchg_mean, 'val_delta':delta_mean}

    def train(self):
        cfg = self.cfg
        best_M = float('inf')
        M = None
        upticks = 0
        history: List[Dict[str, float]] = []
        for ep in range(1, cfg.epochs+1):
            tr = self.train_epoch(ep)
            va = self.eval_val()
            B_t = va['val_Lchg'] + cfg.es_alpha * va['val_delta']
            M = B_t if M is None else (1-cfg.es_beta)*M + cfg.es_beta*B_t
            # uptick condition
            deltaM = M - (history[-1]['M'] if history else M)
            if ep > 1 and deltaM > cfg.es_eta:
                upticks += 1
            else:
                upticks = 0
            # save best (before the streak)
            state = {
                'encoder': self.encoder.state_dict(),
                'cls': self.cls.state_dict(),
                'opt': self.opt.state_dict(),
                'epoch': ep,
            }
            if M < best_M:
                best_M = M
                self.best_state = state
            # log
            rec = {
                'epoch': ep,
                **{f'train_{k}': v for k,v in tr.items()},
                **va,
                'B_t': B_t,
                'M': M,
                'upticks': upticks,
            }
            history.append(rec)
            print(f"[E{ep:03d}] loss={tr['loss']:.4f} acc={va['val_acc']:.3f} F1={va['val_macro_f1']:.3f} Lchg={va['val_Lchg']:.4f} δ={va['val_delta']:.4f} M={M:.4f} upticks={upticks}")
            # early stopping rule: stop at first time upticks reach patience
            if upticks >= cfg.es_patience:
                print(f"Early stopping triggered at epoch {ep}. Returning best checkpoint.")
                break
        return history, self.best_state


# ---------------------------
# Convenience entry point
# ---------------------------

def train_main(data_path: str, n_classes: int, **kwargs):
    cfg = TrainConfig(data_path=data_path, n_classes=n_classes, **kwargs)
    trainer = Trainer(cfg)
    history, state = trainer.train()
    return history, state


if __name__ == '__main__':
    # Example quick-start (edit these paths/values as needed):
    # Place your .npz (monolithic) or folder of .npz shards at DATA_PATH.
    DATA_PATH = "C:/Users/ZY/Downloads/code/code1/out_npz"  # or './your_npz_folder/'
    N_CLASSES = 6
    history, state = train_main(DATA_PATH, N_CLASSES)
    # You can torch.save(state, 'checkpoint_best.pt') if desired.
