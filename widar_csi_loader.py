# widar_csi_loader.py
# Minimal CSI parser (Widar/Intel5300-like, 215-byte records) + windowed dataset builder
import numpy as np, re, json
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

RECORD_SIZE = 215
CSI_IQ_SIZE = 30*3*2             # 30 subcarriers * 3 rx-ant * 2 (I,Q) = 180 bytes
CSI_OFFSET   = RECORD_SIZE - CSI_IQ_SIZE

@dataclass
class CSIFileInfo:
    path: Path; user: int; gesture: int; torso: int; face: int; repeat: int; rx_id: int; n_frames: int

def parse_name(p: Path) -> CSIFileInfo:
    m = re.match(r"user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat$", p.name)
    if not m: raise ValueError(f"Bad filename: {p.name}")
    user,a,b,c,d,rx = map(int, m.groups())
    return CSIFileInfo(p, user,a,b,c,d,rx, p.stat().st_size // RECORD_SIZE)

def load_csi_file(path: Path, max_frames=None) -> np.ndarray:
    """Return complex CSI (T, 30, 3) for one r?.dat file."""
    data = path.read_bytes()
    n = len(data)//RECORD_SIZE
    if max_frames: n = min(n, max_frames)
    buf = np.frombuffer(data[:n*RECORD_SIZE], dtype=np.uint8).reshape(n, RECORD_SIZE)
    iq  = buf[:, CSI_OFFSET:].astype(np.int8).reshape(n, 30, 3, 2)  # signed int8 I/Q
    return iq[...,0] + 1j*iq[...,1]

def load_prefix_group(paths: List[Path], max_frames=None) -> Tuple[np.ndarray, Dict]:
    infos = [parse_name(p) for p in paths]
    key = (infos[0].user, infos[0].gesture, infos[0].torso, infos[0].face, infos[0].repeat)
    assert all((i.user,i.gesture,i.torso,i.face,i.repeat)==key for i in infos), "prefix mismatch"
    arrays, counts = [], []
    for i in sorted(infos, key=lambda x: x.rx_id):
        csi = load_csi_file(i.path, max_frames)   # (T, 30, 3)
        arrays.append(csi); counts.append(csi.shape[0])
    T = min(counts)
    csi = np.concatenate([a[:T] for a in arrays], axis=2)  # (T, 30, 3*nr)
    meta = {'user': key[0], 'gesture': key[1], 'torso': key[2], 'face': key[3], 'repeat': key[4],
            'rx_ids': [parse_name(p).rx_id for p in paths], 'T': int(T), 'nr': len(paths), 'rx_ant_per_receiver': 3}
    return csi, meta

def csi_to_amp_phase(csi: np.ndarray):
    amp   = np.abs(csi).astype(np.float32)
    phase = np.unwrap(np.angle(csi).astype(np.float32), axis=0)  # unwrap over time
    # z-score per (subcarrier, channel)
    amp   = (amp   - amp.mean(axis=0, keepdims=True))   / (amp.std(axis=0, keepdims=True)   + 1e-6)
    phase = (phase - phase.mean(axis=0, keepdims=True)) / (phase.std(axis=0, keepdims=True) + 1e-6)
    return amp, phase

def window_stack(x: np.ndarray, win: int=256, hop: int=64):
    T = x.shape[0]
    if T < win: return np.empty((0, win, *x.shape[1:]), dtype=x.dtype)
    n = 1 + (T - win)//hop
    out = np.empty((n, win, *x.shape[1:]), dtype=x.dtype)
    for i in range(n):
        s = i*hop; out[i] = x[s:s+win]
    return out

def save_npz(out_path: Path, amp_w, phase_w, meta, label: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, amp=amp_w, phase=phase_w, meta=json.dumps(meta), label=int(label))
    return out_path

# --------- quick CLI ----------
if __name__ == "__main__":
    ROOT = Path("C:/Users/ZY/Downloads/CSI_20181117/20181117/user4")  # 改成你放 .dat 的根目录 "C:\Users\ZY\Downloads\CSI_20181117\20181117\user4"
    out_dir = Path("./out_npz")
    # 收集所有同前缀的 r?.dat 组
    groups: Dict[str, List[Path]] = {}
    for p in ROOT.rglob("user*-r*.dat"):
        prefix = re.sub(r"-r\d+\.dat$", "", p.name)  # user4-1-1-1-1
        groups.setdefault(prefix, []).append(p)
    print(f"Found {len(groups)} prefixes.")
    for prefix, paths in groups.items():
        if len(paths) < 3: 
            print(f"[skip] {prefix}: only {len(paths)} receivers")
            continue
        csi, meta = load_prefix_group(paths)
        amp, phase = csi_to_amp_phase(csi)
        amp_w  = window_stack(amp,  win=256, hop=64)
        phase_w= window_stack(phase,win=256, hop=64)
        if amp_w.shape[0] == 0: 
            print(f"[skip] {prefix}: T={meta['T']} < win")
            continue
        label = meta["gesture"]
        out_path = out_dir / f"{prefix}_rx{len(paths)}.npz"
        save_npz(out_path, amp_w, phase_w, meta, label)
        print(f"[ok] {prefix}: windows={amp_w.shape[0]}, out={out_path}")
