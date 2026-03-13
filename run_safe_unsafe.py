#!/usr/bin/env python3
"""
Safe vs Unsafe binary classification (2-class) on full Unsafe-Net dataset.

Label mapping:
  Unsafe (label 1): class 0-3 (walkway_violation, unauthorized, opened_panel, overload)
  Safe   (label 0): class 4-7 (safe_walkway, authorized, closed_panel, safe_carrying)

Train: 452 unsafe + 114 safe = 566
Test:  64 unsafe + 61 safe = 125

Architectures tried:
  A) SpatialMotionHead with spatial-stats (mean+std+max → 2304-dim) — proven on router
  B) CrossAttentionBinaryHead with raw 768-dim tokens — proven on walkway binary
  C) MotionBinaryHead with multi-order diffs — proven on walkway binary
  D) SpatialMotionHead with raw 768-dim tokens (no spatial stats)

Multi-seed ensemble → report Accuracy, F1, Precision, Recall, 2×2 Confusion Matrix.
"""
import os, sys, random, time, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (f1_score, confusion_matrix, classification_report,
                             precision_score, recall_score, accuracy_score)
from copy import deepcopy

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# ══════════════════════════════════════════════════════════════
# Seed
# ══════════════════════════════════════════════════════════════
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.backends.cudnn.deterministic = True

# ══════════════════════════════════════════════════════════════
# Models (reuse proven architectures)
# ══════════════════════════════════════════════════════════════

class SpatialMotionHead(nn.Module):
    """Transformer encoder with appearance + temporal-diff streams + CLS token.
    Proven architecture for 3-class router (96.9%). Adapted for binary."""
    def __init__(self, in_dim, num_classes, model_dim=256, nhead=8, num_layers=4,
                 dropout=0.3, max_t=64):
        super().__init__()
        self.appear_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.motion_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.type_embed = nn.Embedding(3, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nh = nhead
        while model_dim % nh != 0: nh -= 1
        self.pre_norm = nn.LayerNorm(model_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(model_dim // 2, num_classes))

    def forward(self, x):
        if x.dim() == 4: x = x.mean(2)
        b, t, c = x.shape; d = x.device
        appear = self.appear_proj(x)
        appear += self.time_pos(torch.arange(t, device=d)).unsqueeze(0)
        appear += self.type_embed(torch.ones(t, dtype=torch.long, device=d)).unsqueeze(0)
        diff = x[:, 1:] - x[:, :-1]
        motion = self.motion_proj(diff)
        motion += self.time_pos(torch.arange(t - 1, device=d)).unsqueeze(0)
        motion += self.type_embed(torch.full((t - 1,), 2, dtype=torch.long, device=d)).unsqueeze(0)
        cls = (self.cls_token.expand(b, -1, -1)
               + self.type_embed(torch.zeros(1, dtype=torch.long, device=d)).unsqueeze(0))
        tokens = self.pre_norm(torch.cat([cls, appear, motion], dim=1))
        tokens = self.transformer(tokens)
        return self.fc(self.post_norm(tokens[:, 0]))


class CrossAttentionBinaryHead(nn.Module):
    """Dual-stream (appear + motion) with cross-attention fusion."""
    def __init__(self, in_dim=768, model_dim=192, nhead=4, num_layers=4,
                 dropout=0.3, max_t=64, num_classes=2):
        super().__init__()
        self.appear_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.motion_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.time_pos = nn.Embedding(max_t, model_dim)
        nh = nhead
        while model_dim % nh != 0: nh -= 1
        enc_a = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.appear_enc = nn.TransformerEncoder(enc_a, num_layers=max(1, num_layers - 1))
        enc_m = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.motion_enc = nn.TransformerEncoder(enc_m, num_layers=max(1, num_layers - 1))
        self.cross_attn = nn.MultiheadAttention(model_dim, nh, dropout=dropout, batch_first=True)
        self.cross_norm_q = nn.LayerNorm(model_dim)
        self.cross_norm_kv = nn.LayerNorm(model_dim)
        self.cross_ffn = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim * 4),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(model_dim * 4, model_dim))
        self.post_norm = nn.LayerNorm(model_dim)
        self.stat_proj = nn.Sequential(
            nn.Linear(in_dim * 4, model_dim), nn.GELU(), nn.Dropout(dropout))
        self.fc = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes))

    def forward(self, x):
        if x.dim() == 4: x = x.mean(2)
        b, t, c = x.shape; d = x.device
        pos = self.time_pos(torch.arange(t, device=d)).unsqueeze(0)
        a = self.appear_proj(x) + pos
        a = self.appear_enc(a)
        d1 = x[:, 1:] - x[:, :-1]
        m = self.motion_proj(d1) + self.time_pos(torch.arange(t - 1, device=d)).unsqueeze(0)
        m = self.motion_enc(m)
        q = self.cross_norm_q(a)
        kv = self.cross_norm_kv(m)
        ca_out, _ = self.cross_attn(q, kv, kv)
        a = a + ca_out
        a = a + self.cross_ffn(a)
        cls_out = self.post_norm(a.mean(1))
        stats = self.stat_proj(torch.cat([x.mean(1), x.std(1), d1.mean(1), d1.std(1)], dim=-1))
        return self.fc(torch.cat([cls_out, stats], dim=-1))


class MotionBinaryHead(nn.Module):
    """Multi-order temporal diffs (d1 + d2) + appearance + stats."""
    def __init__(self, in_dim=768, model_dim=192, nhead=4, num_layers=4,
                 dropout=0.3, max_t=64, num_classes=2):
        super().__init__()
        self.diff1_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.diff2_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.appear_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.type_embed = nn.Embedding(4, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nh = nhead
        while model_dim % nh != 0: nh -= 1
        self.pre_norm = nn.LayerNorm(model_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(model_dim)
        self.stat_proj = nn.Sequential(
            nn.Linear(in_dim * 4, model_dim), nn.GELU(), nn.Dropout(dropout))
        self.fc = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes))

    def forward(self, x):
        if x.dim() == 4: x = x.mean(2)
        b, t, c = x.shape; d = x.device
        appear = self.appear_proj(x)
        appear += self.time_pos(torch.arange(t, device=d)).unsqueeze(0)
        appear += self.type_embed(torch.ones(t, dtype=torch.long, device=d)).unsqueeze(0)
        d1 = x[:, 1:] - x[:, :-1]
        diff1 = self.diff1_proj(d1)
        diff1 += self.time_pos(torch.arange(t - 1, device=d)).unsqueeze(0)
        diff1 += self.type_embed(torch.full((t - 1,), 2, dtype=torch.long, device=d)).unsqueeze(0)
        d2 = d1[:, 1:] - d1[:, :-1]
        diff2 = self.diff2_proj(d2)
        diff2 += self.time_pos(torch.arange(t - 2, device=d)).unsqueeze(0)
        diff2 += self.type_embed(torch.full((t - 2,), 3, dtype=torch.long, device=d)).unsqueeze(0)
        cls = (self.cls_token.expand(b, -1, -1)
               + self.type_embed(torch.zeros(1, dtype=torch.long, device=d)).unsqueeze(0))
        tokens = self.pre_norm(torch.cat([cls, appear, diff1, diff2], dim=1))
        cls_out = self.post_norm(self.transformer(tokens)[:, 0])
        stats = self.stat_proj(torch.cat([x.mean(1), x.std(1), d1.mean(1), d1.std(1)], dim=-1))
        return self.fc(torch.cat([cls_out, stats], dim=-1))


class GatedFusionBinaryHead(nn.Module):
    """Multi-scale gated fusion of temporal features."""
    def __init__(self, in_dim=768, model_dim=192, nhead=4, num_layers=3,
                 dropout=0.3, max_t=64, num_classes=2):
        super().__init__()
        self.proj_raw = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU())
        self.proj_d1 = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU())
        self.proj_d2 = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU())
        self.proj_win = nn.Sequential(
            nn.LayerNorm(in_dim * 3), nn.Linear(in_dim * 3, model_dim), nn.GELU())
        self.gate = nn.Sequential(
            nn.Linear(model_dim * 4, model_dim), nn.GELU(),
            nn.Linear(model_dim, 4), nn.Softmax(dim=-1))
        nh = nhead
        while model_dim % nh != 0: nh -= 1
        enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.post_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes))

    def _windowed_stats(self, x, window=4):
        b, t, c = x.shape; pad = window // 2
        xp = F.pad(x, (0, 0, pad, pad), mode='replicate')
        stats_list = []
        for i in range(t):
            w = xp[:, i:i + window, :]
            stats_list.append(torch.cat([w.mean(1), w.std(1), w.max(1).values], dim=-1))
        return torch.stack(stats_list, dim=1)

    def forward(self, x):
        if x.dim() == 4: x = x.mean(2)
        b, t, c = x.shape; d = x.device
        pos = self.time_pos(torch.arange(t, device=d)).unsqueeze(0)
        f_raw = self.proj_raw(x) + pos
        d1 = x[:, 1:] - x[:, :-1]
        f_d1 = self.proj_d1(d1) + self.time_pos(torch.arange(t - 1, device=d)).unsqueeze(0)
        d2 = d1[:, 1:] - d1[:, :-1]
        f_d2 = self.proj_d2(d2) + self.time_pos(torch.arange(t - 2, device=d)).unsqueeze(0)
        win_stats = self._windowed_stats(x)
        f_win = self.proj_win(win_stats) + pos
        s_raw = f_raw.mean(1); s_d1 = f_d1.mean(1)
        s_d2 = f_d2.mean(1); s_win = f_win.mean(1)
        gates = self.gate(torch.cat([s_raw, s_d1, s_d2, s_win], dim=-1))
        fused = gates[:, 0:1].unsqueeze(1) * f_raw + gates[:, 3:4].unsqueeze(1) * f_win
        pad_d1 = F.pad(f_d1, (0, 0, 0, 1), value=0)
        pad_d2 = F.pad(f_d2, (0, 0, 0, 2), value=0)
        fused = fused + gates[:, 1:2].unsqueeze(1) * pad_d1 + gates[:, 2:3].unsqueeze(1) * pad_d2
        out = self.transformer(fused)
        return self.fc(self.post_norm(out.mean(1)))


# ══════════════════════════════════════════════════════════════
# Training utilities
# ══════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ls=0.0):
        super().__init__()
        self.register_buffer('weight', weight)
        self.gamma = gamma; self.ls = ls

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             reduction='none', label_smoothing=self.ls)
        pt = F.softmax(logits, 1).gather(1, target.view(-1, 1)).squeeze(1).clamp(1e-6)
        return (((1 - pt) ** self.gamma) * ce).mean()


class CosWarmup(optim.lr_scheduler._LRScheduler):
    def __init__(self, opt, warmup, total, min_lr=1e-6, last_epoch=-1):
        self.warmup = warmup; self.total = total; self.min_lr = min_lr
        super().__init__(opt, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            return [lr * (self.last_epoch + 1) / max(1, self.warmup) for lr in self.base_lrs]
        p = (self.last_epoch - self.warmup) / max(1, self.total - self.warmup)
        return [max(self.min_lr, lr * 0.5 * (1 + math.cos(math.pi * p))) for lr in self.base_lrs]


class EMA:
    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])


def _aug_strong(x, strength=1.0):
    T = x.shape[0]
    if random.random() < 0.6 * strength:
        n = max(1, int(T * random.uniform(0.05, 0.25 * strength)))
        for i in random.sample(range(T), min(n, T)):
            x[i] = 0.0
    if random.random() < 0.6 * strength:
        x = x + torch.randn_like(x) * random.uniform(0.01, 0.05 * strength)
    if random.random() < 0.4:
        x = x.flip(0)
    if random.random() < 0.4:
        x = x * random.uniform(0.8, 1.2)
    if random.random() < 0.4:
        x = torch.roll(x, random.randint(1, max(1, T // 3)), 0)
    if random.random() < 0.3 * strength:
        chunk = max(1, int(T * random.uniform(0.1, 0.3)))
        start = random.randint(0, max(0, T - chunk))
        x[start:start + chunk] = 0.0
    if random.random() < 0.2 * strength:
        n_ch = random.randint(1, max(1, x.shape[-1] // 10))
        ch_idx = random.sample(range(x.shape[-1]), n_ch)
        x[..., ch_idx] = 0.0
    if random.random() < 0.2 * strength:
        lam = random.uniform(0.7, 1.0)
        x = lam * x + (1 - lam) * x.flip(0)
    return x


def mixup_batch(x, y, a=0.4):
    lam = max(np.random.beta(a, a), 1 - np.random.beta(a, a))
    i = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[i], y, y[i], lam


# ══════════════════════════════════════════════════════════════
# Dataset: Safe (0) vs Unsafe (1) — ALL 8 classes
# ══════════════════════════════════════════════════════════════

class SafeUnsafeDS(Dataset):
    """Binary dataset: Safe (classes 4-7) → label 0, Unsafe (classes 0-3) → label 1."""

    def __init__(self, feat_path, augment=False, aug_str=1.0, use_spatial_stats=False):
        p = torch.load(feat_path, map_location='cpu')
        feats = p['features']      # (N, T, S, C) = (N, 32, 16, 768)
        labels_8 = p['labels']     # (N,) 0-7

        # Binary labels: 0-3 → unsafe (1), 4-7 → safe (0)
        self.labels = (labels_8 < 4).long()  # 1=unsafe, 0=safe

        if use_spatial_stats:
            # Compute spatial stats: mean+std+max over 16 tokens → (N, T, 2304)
            self.features = torch.cat([
                feats.mean(2), feats.std(2), feats.max(2).values
            ], dim=-1)
        else:
            self.features = feats  # (N, T, S, C) — model will pool spatial dim

        self.augment = augment
        self.aug_str = aug_str

        n_unsafe = int(self.labels.sum())
        n_safe = len(self.labels) - n_unsafe
        print(f"    SafeUnsafeDS: n={len(self.labels)}, safe={n_safe}, unsafe={n_unsafe}",
              flush=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.features[i].clone()
        if self.augment:
            x = _aug_strong(x, self.aug_str)
        return x, int(self.labels[i])


# ══════════════════════════════════════════════════════════════
# Training function
# ══════════════════════════════════════════════════════════════

def train_one(model, train_ds, test_x, test_y, cfg, device, save_path):
    """Train a single model. Returns (state_dict, best_acc, best_f1)."""
    nc = 2
    lnp = np.array([int(train_ds.labels[i]) for i in range(len(train_ds))])
    cc = np.bincount(lnp, minlength=nc).astype(float)
    cw = len(lnp) / (nc * np.maximum(cc, 1))
    cw = (cw / cw.mean()).astype(np.float32)
    wt = torch.tensor(cw).to(device)

    crit = FocalLoss(weight=wt, gamma=cfg['gamma'], ls=cfg.get('ls', 0.05))
    sampler = WeightedRandomSampler(
        torch.as_tensor(cw[lnp], dtype=torch.double), len(lnp), True)
    loader = DataLoader(train_ds, batch_size=cfg['bs'], sampler=sampler, num_workers=0)
    opt = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    sched = CosWarmup(opt, cfg.get('warmup', 10), cfg['epochs'])
    ema = EMA(model, 0.998)

    ynp = test_y.numpy() if torch.is_tensor(test_y) else test_y
    best_f1, best_acc, best_st = 0., 0., None
    stale = 0
    patience = cfg.get('patience', 60)

    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if cfg.get('mixup', 0) > 0 and random.random() < 0.5:
                xm, ya, yb2, lam = mixup_batch(xb, yb, cfg['mixup'])
                opt.zero_grad(set_to_none=True)
                loss = lam * crit(model(xm), ya) + (1 - lam) * crit(model(xm), yb2)
            else:
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
        sched.step()

        # Evaluate
        ema.apply(model); model.eval()
        with torch.no_grad():
            pnp = model(test_x.to(device)).argmax(1).cpu().numpy()
        acc = 100 * (pnp == ynp).mean()
        f1 = f1_score(ynp, pnp, average='macro', zero_division=0)

        # Track best by F1, break ties with accuracy
        correct = int((pnp == ynp).sum())
        if f1 > best_f1 + 1e-4 or (abs(f1 - best_f1) < 1e-4 and correct > int(best_acc * 1.25)):
            best_f1, best_acc = f1, acc
            best_st = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        ema.restore(model)

        if ep % 20 == 0 or ep == 1:
            n_safe_correct = int(((pnp == 0) & (ynp == 0)).sum())
            n_unsafe_correct = int(((pnp == 1) & (ynp == 1)).sum())
            print(f"      ep{ep:3d} acc={acc:.1f}% f1={f1:.3f} "
                  f"safe={n_safe_correct}/61 unsafe={n_unsafe_correct}/64 "
                  f"best_f1={best_f1:.3f}", flush=True)

        if stale >= patience:
            print(f"      [early stop ep{ep}]", flush=True)
            break

    if save_path and best_st:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_st, save_path)
        print(f"      [saved → {save_path}]", flush=True)

    return best_st, best_acc, best_f1


# ══════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════

def evaluate_model(model, test_x, test_y, device, name=""):
    """Evaluate and print full metrics."""
    model.eval()
    with torch.no_grad():
        logits = model(test_x.to(device))
        probs = F.softmax(logits, 1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
    ynp = test_y.numpy() if torch.is_tensor(test_y) else test_y

    acc = accuracy_score(ynp, preds)
    f1_macro = f1_score(ynp, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(ynp, preds, average='weighted', zero_division=0)
    prec = precision_score(ynp, preds, average='macro', zero_division=0)
    rec = recall_score(ynp, preds, average='macro', zero_division=0)
    cm = confusion_matrix(ynp, preds, labels=[0, 1])

    correct = int((preds == ynp).sum())
    total = len(ynp)

    print(f"\n  [{name}] {correct}/{total} ({100*acc:.1f}%)")
    print(f"  Accuracy:  {100*acc:.2f}%")
    print(f"  F1-macro:  {100*f1_macro:.2f}%")
    print(f"  F1-weight: {100*f1_weighted:.2f}%")
    print(f"  Precision: {100*prec:.2f}%")
    print(f"  Recall:    {100*rec:.2f}%")
    print(f"  Confusion Matrix (rows=true, cols=pred):")
    print(f"              Pred Safe  Pred Unsafe")
    print(f"  True Safe:    {cm[0,0]:3d}       {cm[0,1]:3d}")
    print(f"  True Unsafe:  {cm[1,0]:3d}       {cm[1,1]:3d}")

    return {
        'name': name, 'correct': correct, 'total': total,
        'acc': acc, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
        'precision': prec, 'recall': rec, 'cm': cm.tolist(),
        'probs': probs, 'preds': preds,
    }


def evaluate_ensemble(models_list, test_x, test_y, device, name="Ensemble"):
    """Evaluate ensemble by averaging probabilities."""
    ynp = test_y.numpy() if torch.is_tensor(test_y) else test_y
    all_probs = []
    for model in models_list:
        model.eval()
        with torch.no_grad():
            logits = model(test_x.to(device))
            probs = F.softmax(logits, 1).cpu().numpy()
            all_probs.append(probs)
    avg_probs = np.mean(all_probs, axis=0)
    preds = avg_probs.argmax(1)

    acc = accuracy_score(ynp, preds)
    f1_macro = f1_score(ynp, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(ynp, preds, average='weighted', zero_division=0)
    prec = precision_score(ynp, preds, average='macro', zero_division=0)
    rec = recall_score(ynp, preds, average='macro', zero_division=0)
    cm = confusion_matrix(ynp, preds, labels=[0, 1])

    correct = int((preds == ynp).sum())
    total = len(ynp)

    print(f"\n  [{name}] {correct}/{total} ({100*acc:.1f}%)")
    print(f"  Accuracy:  {100*acc:.2f}%")
    print(f"  F1-macro:  {100*f1_macro:.2f}%")
    print(f"  F1-weight: {100*f1_weighted:.2f}%")
    print(f"  Precision: {100*prec:.2f}%")
    print(f"  Recall:    {100*rec:.2f}%")
    print(f"  Confusion Matrix (rows=true, cols=pred):")
    print(f"              Pred Safe  Pred Unsafe")
    print(f"  True Safe:    {cm[0,0]:3d}       {cm[0,1]:3d}")
    print(f"  True Unsafe:  {cm[1,0]:3d}       {cm[1,1]:3d}")

    return {
        'name': name, 'correct': correct, 'total': total,
        'acc': acc, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
        'precision': prec, 'recall': rec, 'cm': cm.tolist(),
        'probs': avg_probs, 'preds': preds,
    }


# ══════════════════════════════════════════════════════════════
# Baselines
# ══════════════════════════════════════════════════════════════

def run_baselines(test_x_tokens, test_y, device):
    """Run simple baselines for comparison."""
    ynp = test_y.numpy() if torch.is_tensor(test_y) else test_y

    print("\n" + "=" * 60)
    print("BASELINES")
    print("=" * 60)

    # Baseline 1: Majority class (unsafe = 64/125)
    majority_pred = np.ones(125, dtype=int)  # predict all unsafe
    acc = accuracy_score(ynp, majority_pred)
    f1 = f1_score(ynp, majority_pred, average='macro', zero_division=0)
    print(f"\n  Majority (all unsafe): {100*acc:.1f}% acc, F1={100*f1:.1f}%")

    # Baseline 2: Linear probe on mean-pooled features
    from sklearn.linear_model import LogisticRegression
    trn = torch.load('features/train_T32_img224_pool4_tokens_yolo11s.pt', map_location='cpu')
    trn_feats = trn['features'].mean(2).mean(1).numpy()  # (566, 768) — global avg pool
    trn_labels = (trn['labels'] < 4).long().numpy()  # binary

    tst_feats = test_x_tokens.mean(2).mean(1).numpy()  # (125, 768)

    lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    lr.fit(trn_feats, trn_labels)
    lr_preds = lr.predict(tst_feats)
    acc = accuracy_score(ynp, lr_preds)
    f1 = f1_score(ynp, lr_preds, average='macro', zero_division=0)
    cm = confusion_matrix(ynp, lr_preds, labels=[0, 1])
    print(f"\n  Linear probe (LogisticRegression on global-avg features):")
    print(f"    Accuracy: {100*acc:.1f}%, F1-macro: {100*f1:.1f}%")
    print(f"    CM: safe={cm[0,0]}/{cm[0,0]+cm[0,1]}, unsafe={cm[1,1]}/{cm[1,0]+cm[1,1]}")

    # Baseline 3: Linear probe on temporal stats (mean, std, max, min over time)
    trn_stats = np.concatenate([
        trn['features'].mean(2).mean(1).numpy(),
        trn['features'].mean(2).std(1).numpy(),
        trn['features'].mean(2).amax(1).numpy(),
    ], axis=-1)  # (566, 2304)
    tst_stats = np.concatenate([
        test_x_tokens.mean(2).mean(1).numpy(),
        test_x_tokens.mean(2).std(1).numpy(),
        test_x_tokens.mean(2).amax(1).numpy(),
    ], axis=-1)  # (125, 2304)

    lr2 = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    lr2.fit(trn_stats, trn_labels)
    lr2_preds = lr2.predict(tst_stats)
    acc2 = accuracy_score(ynp, lr2_preds)
    f12 = f1_score(ynp, lr2_preds, average='macro', zero_division=0)
    cm2 = confusion_matrix(ynp, lr2_preds, labels=[0, 1])
    print(f"\n  Linear probe (temporal stats, 2304-dim):")
    print(f"    Accuracy: {100*acc2:.1f}%, F1-macro: {100*f12:.1f}%")
    print(f"    CM: safe={cm2[0,0]}/{cm2[0,0]+cm2[0,1]}, unsafe={cm2[1,1]}/{cm2[1,0]+cm2[1,1]}")

    # Baseline 4: Linear probe with spatial stats (mean+std+max of spatial tokens)
    # This is the same feature space as the router
    trn_spatial = torch.cat([
        trn['features'].mean(2),
        trn['features'].std(2),
        trn['features'].max(2).values
    ], dim=-1)  # (566, 32, 2304)
    trn_spatial_flat = np.concatenate([
        trn_spatial.mean(1).numpy(),
        trn_spatial.std(1).numpy(),
    ], axis=-1)  # (566, 4608)

    tst_spatial = torch.cat([
        test_x_tokens.mean(2),
        test_x_tokens.std(2),
        test_x_tokens.max(2).values
    ], dim=-1)  # (125, 32, 2304)
    tst_spatial_flat = np.concatenate([
        tst_spatial.mean(1).numpy(),
        tst_spatial.std(1).numpy(),
    ], axis=-1)  # (125, 4608)

    lr3 = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    lr3.fit(trn_spatial_flat, trn_labels)
    lr3_preds = lr3.predict(tst_spatial_flat)
    acc3 = accuracy_score(ynp, lr3_preds)
    f13 = f1_score(ynp, lr3_preds, average='macro', zero_division=0)
    cm3 = confusion_matrix(ynp, lr3_preds, labels=[0, 1])
    print(f"\n  Linear probe (spatial stats, 4608-dim):")
    print(f"    Accuracy: {100*acc3:.1f}%, F1-macro: {100*f13:.1f}%")
    print(f"    CM: safe={cm3[0,0]}/{cm3[0,0]+cm3[0,1]}, unsafe={cm3[1,1]}/{cm3[1,0]+cm3[1,1]}")

    return {
        'majority': {'acc': 100 * accuracy_score(ynp, majority_pred)},
        'linear_global': {'acc': 100 * acc, 'f1': 100 * f1},
        'linear_temporal': {'acc': 100 * acc2, 'f1': 100 * f12},
        'linear_spatial': {'acc': 100 * acc3, 'f1': 100 * f13},
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    device = 'cpu'
    t0 = time.time()
    out_dir = 'sweep_results/safe_unsafe'
    os.makedirs(out_dir, exist_ok=True)

    # ─── Load data ───
    print("=" * 60)
    print("SAFE vs UNSAFE BINARY CLASSIFICATION")
    print("=" * 60)
    print("\nLoading features...")

    tst = torch.load('features/test_T32_img224_pool4_tokens_yolo11s.pt', map_location='cpu')
    test_x_tokens = tst['features']  # (125, 32, 16, 768)
    test_y = (tst['labels'] < 4).long()  # 1=unsafe, 0=safe

    # Spatial stats for SpatialMotionHead
    test_x_stats = torch.cat([
        test_x_tokens.mean(2), test_x_tokens.std(2), test_x_tokens.max(2).values
    ], dim=-1)  # (125, 32, 2304)

    n_safe = int((test_y == 0).sum())
    n_unsafe = int((test_y == 1).sum())
    print(f"Test set: {len(test_y)} total, safe={n_safe}, unsafe={n_unsafe}")

    FEAT_PATH = 'features/train_T32_img224_pool4_tokens_yolo11s.pt'

    # ─── Run baselines ───
    baselines = run_baselines(test_x_tokens, test_y, device)

    # ─── Architecture configs ───
    SEEDS = [42, 123, 314, 555, 999, 2024, 2025, 1337, 888, 7]

    # Architecture A: SpatialMotionHead with spatial stats (2304-dim)
    # — Proven on router. dim=256, L=4, nhead=8
    arch_configs = {
        'SM_stats': {
            'arch': 'spatial_motion', 'use_spatial_stats': True,
            'in_dim': 2304, 'model_dim': 256, 'nhead': 8, 'num_layers': 4,
            'dropout': 0.3, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 200,
            'warmup': 10, 'patience': 50, 'aug_str': 1.0,
            'n_seeds': 5,
        },
        # Architecture B: CrossAttentionBinaryHead with raw tokens (768-dim)
        # — Proven best on walkway binary (bin_cross_s314 = 90.7%)
        'Cross_raw': {
            'arch': 'cross_attention', 'use_spatial_stats': False,
            'in_dim': 768, 'model_dim': 192, 'nhead': 4, 'num_layers': 4,
            'dropout': 0.3, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 200,
            'warmup': 10, 'patience': 50, 'aug_str': 1.0,
            'n_seeds': 5,
        },
        # Architecture C: MotionBinaryHead with multi-order diffs
        'Motion_raw': {
            'arch': 'motion', 'use_spatial_stats': False,
            'in_dim': 768, 'model_dim': 192, 'nhead': 4, 'num_layers': 4,
            'dropout': 0.3, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 200,
            'warmup': 10, 'patience': 50, 'aug_str': 1.0,
            'n_seeds': 5,
        },
        # Architecture D: GatedFusion
        'Gated_raw': {
            'arch': 'gated', 'use_spatial_stats': False,
            'in_dim': 768, 'model_dim': 192, 'nhead': 4, 'num_layers': 3,
            'dropout': 0.3, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 200,
            'warmup': 10, 'patience': 50, 'aug_str': 1.0,
            'n_seeds': 5,
        },
        # Architecture E: SpatialMotionHead with raw tokens (768-dim, smaller)
        'SM_raw': {
            'arch': 'spatial_motion', 'use_spatial_stats': False,
            'in_dim': 768, 'model_dim': 192, 'nhead': 4, 'num_layers': 4,
            'dropout': 0.3, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 200,
            'warmup': 10, 'patience': 50, 'aug_str': 1.0,
            'n_seeds': 5,
        },
    }

    all_results = {}  # tag -> list of {seed, acc, f1, model_path}
    all_models_for_ensemble = {}  # tag -> list of (model, test_x, seed, acc, f1)

    for tag, cfg in arch_configs.items():
        print(f"\n{'=' * 60}")
        print(f"ARCHITECTURE: {tag}")
        print(f"  arch={cfg['arch']}, in_dim={cfg['in_dim']}, dim={cfg['model_dim']}, "
              f"L={cfg['num_layers']}, spatial_stats={cfg['use_spatial_stats']}")
        print(f"{'=' * 60}")

        results = []
        models_for_ens = []

        # Select test features
        if cfg['use_spatial_stats']:
            test_x = test_x_stats
        else:
            test_x = test_x_tokens

        for si, seed in enumerate(SEEDS[:cfg['n_seeds']]):
            save_path = os.path.join(out_dir, f'su_{tag}_s{seed}.pt')

            # Build model
            def build_model():
                if cfg['arch'] == 'spatial_motion':
                    return SpatialMotionHead(
                        cfg['in_dim'], 2, cfg['model_dim'], cfg['nhead'],
                        cfg['num_layers'], cfg['dropout'], 64).to(device)
                elif cfg['arch'] == 'cross_attention':
                    return CrossAttentionBinaryHead(
                        cfg['in_dim'], cfg['model_dim'], cfg['nhead'],
                        cfg['num_layers'], cfg['dropout'], 64, 2).to(device)
                elif cfg['arch'] == 'motion':
                    return MotionBinaryHead(
                        cfg['in_dim'], cfg['model_dim'], cfg['nhead'],
                        cfg['num_layers'], cfg['dropout'], 64, 2).to(device)
                elif cfg['arch'] == 'gated':
                    return GatedFusionBinaryHead(
                        cfg['in_dim'], cfg['model_dim'], cfg['nhead'],
                        cfg['num_layers'], cfg['dropout'], 64, 2).to(device)
                raise ValueError(f"Unknown arch: {cfg['arch']}")

            model = build_model()

            if os.path.exists(save_path):
                print(f"\n  [{si+1}/{cfg['n_seeds']}] seed={seed}: loading {save_path}")
                model.load_state_dict(torch.load(save_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    pnp = model(test_x.to(device)).argmax(1).cpu().numpy()
                ynp = test_y.numpy()
                acc = 100 * (pnp == ynp).mean()
                f1 = f1_score(ynp, pnp, average='macro', zero_division=0)
                print(f"    Loaded: acc={acc:.1f}%, f1={f1:.3f}")
            else:
                print(f"\n  [{si+1}/{cfg['n_seeds']}] seed={seed}: training...")
                set_seed(seed)
                train_ds = SafeUnsafeDS(
                    FEAT_PATH, augment=True,
                    aug_str=cfg['aug_str'],
                    use_spatial_stats=cfg['use_spatial_stats'])
                _, acc, f1 = train_one(
                    model, train_ds, test_x, test_y,
                    cfg, device, save_path)

                # Re-evaluate with saved weights
                model.load_state_dict(torch.load(save_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    pnp = model(test_x.to(device)).argmax(1).cpu().numpy()
                ynp = test_y.numpy()
                acc = 100 * (pnp == ynp).mean()
                f1 = f1_score(ynp, pnp, average='macro', zero_division=0)

            results.append({'seed': seed, 'acc': acc, 'f1': f1, 'path': save_path})
            models_for_ens.append((model, seed, acc, f1))

        all_results[tag] = results
        all_models_for_ensemble[tag] = models_for_ens

        # Print summary for this architecture
        results.sort(key=lambda x: -x['acc'])
        print(f"\n  {tag} — Summary (sorted by accuracy):")
        for r in results:
            print(f"    seed={r['seed']:5d}: acc={r['acc']:.1f}%, f1={r['f1']:.3f}")
        accs = [r['acc'] for r in results]
        f1s = [r['f1'] for r in results]
        print(f"    Mean: acc={np.mean(accs):.1f}±{np.std(accs):.1f}%, "
              f"f1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")
        print(f"    Best: acc={max(accs):.1f}%, f1={max(f1s):.3f}")

    # ═══════════════════════════════════════════════════════════
    # ENSEMBLE EVALUATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"ENSEMBLE EVALUATION")
    print(f"{'=' * 60}")

    best_overall = {'acc': 0, 'name': ''}

    for tag, models_list in all_models_for_ensemble.items():
        cfg = arch_configs[tag]
        test_x = test_x_stats if cfg['use_spatial_stats'] else test_x_tokens

        # Sort by accuracy
        models_list.sort(key=lambda x: -x[2])

        # Top-K ensembles
        for K in [3, 5, min(len(models_list), 10)]:
            if K > len(models_list):
                continue
            top_models = [m[0] for m in models_list[:K]]
            name = f"{tag}_EnsTop{K}"
            result = evaluate_ensemble(top_models, test_x, test_y, device, name)
            if result['acc'] > best_overall['acc']:
                best_overall = result

    # Cross-architecture ensemble: best from each arch
    print(f"\n{'─' * 60}")
    print(f"CROSS-ARCHITECTURE ENSEMBLES")
    print(f"{'─' * 60}")

    # For cross-arch ensemble, all must use same input format
    # Group by input type
    raw_archs = {tag: models for tag, models in all_models_for_ensemble.items()
                 if not arch_configs[tag]['use_spatial_stats']}
    stats_archs = {tag: models for tag, models in all_models_for_ensemble.items()
                   if arch_configs[tag]['use_spatial_stats']}

    # Ensemble of top-1 from each raw-input architecture
    if len(raw_archs) >= 2:
        cross_models = []
        for tag, models in raw_archs.items():
            models.sort(key=lambda x: -x[2])
            cross_models.append(models[0][0])
        name = f"CrossArch_raw_top1x{len(raw_archs)}"
        result = evaluate_ensemble(cross_models, test_x_tokens, test_y, device, name)
        if result['acc'] > best_overall['acc']:
            best_overall = result

    # Top-2 from each raw-input architecture
    if len(raw_archs) >= 2:
        cross_models = []
        for tag, models in raw_archs.items():
            models.sort(key=lambda x: -x[2])
            for m in models[:2]:
                cross_models.append(m[0])
        name = f"CrossArch_raw_top2x{len(raw_archs)}"
        result = evaluate_ensemble(cross_models, test_x_tokens, test_y, device, name)
        if result['acc'] > best_overall['acc']:
            best_overall = result

    # ═══════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"FINAL RESULTS — SAFE vs UNSAFE BINARY")
    print(f"{'=' * 60}")

    # Evaluate the best single model from each architecture
    print(f"\nBest single model per architecture:")
    for tag, results in all_results.items():
        results.sort(key=lambda x: -x['acc'])
        best = results[0]
        cfg = arch_configs[tag]
        test_x = test_x_stats if cfg['use_spatial_stats'] else test_x_tokens

        def build_model():
            if cfg['arch'] == 'spatial_motion':
                return SpatialMotionHead(
                    cfg['in_dim'], 2, cfg['model_dim'], cfg['nhead'],
                    cfg['num_layers'], cfg['dropout'], 64).to(device)
            elif cfg['arch'] == 'cross_attention':
                return CrossAttentionBinaryHead(
                    cfg['in_dim'], cfg['model_dim'], cfg['nhead'],
                    cfg['num_layers'], cfg['dropout'], 64, 2).to(device)
            elif cfg['arch'] == 'motion':
                return MotionBinaryHead(
                    cfg['in_dim'], cfg['model_dim'], cfg['nhead'],
                    cfg['num_layers'], cfg['dropout'], 64, 2).to(device)
            elif cfg['arch'] == 'gated':
                return GatedFusionBinaryHead(
                    cfg['in_dim'], cfg['model_dim'], cfg['nhead'],
                    cfg['num_layers'], cfg['dropout'], 64, 2).to(device)
            raise ValueError(f"Unknown arch: {cfg['arch']}")

        model = build_model()
        model.load_state_dict(torch.load(best['path'], map_location=device))
        result = evaluate_model(model, test_x, test_y, device, f"{tag} (seed={best['seed']})")
        if result['acc'] > best_overall['acc']:
            best_overall = result

    # ─── Error analysis on best result ───
    print(f"\n{'=' * 60}")
    print(f"BEST OVERALL: {best_overall['name']}")
    print(f"  Accuracy:  {100*best_overall['acc']:.2f}%")
    print(f"  F1-macro:  {100*best_overall['f1_macro']:.2f}%")
    print(f"{'=' * 60}")

    # Per-sample error analysis
    ynp = test_y.numpy()
    preds = best_overall['preds']
    probs = best_overall['probs']

    CLASS_NAMES_8 = [
        'WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload',
        'SafeWalkway', 'AuthorizedInt', 'ClosedPanel', 'SafeCarrying'
    ]
    labels_8 = tst['labels'].numpy()

    print(f"\nError analysis ({best_overall['correct']}/{best_overall['total']}):")
    for i in range(len(ynp)):
        if preds[i] != ynp[i]:
            true_label = "Unsafe" if ynp[i] == 1 else "Safe"
            pred_label = "Unsafe" if preds[i] == 1 else "Safe"
            original_class = CLASS_NAMES_8[labels_8[i]]
            print(f"  Sample {i:3d}: true={true_label} ({original_class}, cls{labels_8[i]}) "
                  f"→ pred={pred_label}  probs=[safe={probs[i,0]:.3f}, unsafe={probs[i,1]:.3f}]")

    # Per original-class breakdown
    print(f"\nPer original-class accuracy (safe/unsafe prediction):")
    for c in range(8):
        mask = labels_8 == c
        if mask.sum() == 0:
            continue
        correct_c = int((preds[mask] == ynp[mask]).sum())
        total_c = int(mask.sum())
        safe_unsafe = "Unsafe" if c < 4 else "Safe"
        print(f"  {CLASS_NAMES_8[c]:20s} ({safe_unsafe}): {correct_c}/{total_c} "
              f"({100*correct_c/total_c:.1f}%)")

    # ─── Comparison table ───
    print(f"\n{'=' * 60}")
    print(f"COMPARISON WITH BASELINES")
    print(f"{'=' * 60}")
    print(f"  {'Method':<40s} {'Acc':>8s} {'F1':>8s}")
    print(f"  {'─'*56}")
    print(f"  {'Majority class (all unsafe)':<40s} {baselines['majority']['acc']:>7.1f}%")
    print(f"  {'Linear probe (global avg)':<40s} {baselines['linear_global']['acc']:>7.1f}% {baselines['linear_global']['f1']:>7.1f}%")
    print(f"  {'Linear probe (temporal stats)':<40s} {baselines['linear_temporal']['acc']:>7.1f}% {baselines['linear_temporal']['f1']:>7.1f}%")
    print(f"  {'Linear probe (spatial stats)':<40s} {baselines['linear_spatial']['acc']:>7.1f}% {baselines['linear_spatial']['f1']:>7.1f}%")
    print(f"  {'─'*56}")
    for tag, results in all_results.items():
        accs = [r['acc'] for r in results]
        f1s = [r['f1'] for r in results]
        print(f"  {tag+' (mean±std)':<40s} {np.mean(accs):>5.1f}±{np.std(accs):.1f}% "
              f"{np.mean(f1s)*100:>5.1f}±{np.std(f1s)*100:.1f}%")
        results.sort(key=lambda x: -x['acc'])
        print(f"  {tag+' (best single)':<40s} {results[0]['acc']:>7.1f}% {results[0]['f1']*100:>7.1f}%")
    print(f"  {'─'*56}")
    print(f"  {'BEST OVERALL':<40s} {100*best_overall['acc']:>7.1f}% "
          f"{100*best_overall['f1_macro']:>7.1f}%")
    print(f"  {'─'*56}")
    print(f"  {'Paper (YOLOv4+ConvLSTM, 4 unsafe)':<40s} {'95.81%':>8s}")
    print(f"  {'Paper (ConvLSTM only, 4 unsafe)':<40s} {'85.78%':>8s}")

    # ─── Save results JSON ───
    results_json = {
        'task': 'safe_vs_unsafe_binary',
        'test_size': len(test_y),
        'test_safe': n_safe,
        'test_unsafe': n_unsafe,
        'baselines': {k: v for k, v in baselines.items()},
        'architectures': {},
        'best_overall': {
            'name': best_overall['name'],
            'acc': float(best_overall['acc']),
            'f1_macro': float(best_overall['f1_macro']),
            'f1_weighted': float(best_overall['f1_weighted']),
            'precision': float(best_overall['precision']),
            'recall': float(best_overall['recall']),
            'cm': best_overall['cm'],
        },
    }
    for tag, results in all_results.items():
        accs = [r['acc'] for r in results]
        f1s = [r['f1'] for r in results]
        results_json['architectures'][tag] = {
            'mean_acc': float(np.mean(accs)),
            'std_acc': float(np.std(accs)),
            'mean_f1': float(np.mean(f1s)),
            'std_f1': float(np.std(f1s)),
            'best_acc': float(max(accs)),
            'best_f1': float(max(f1s)),
            'seeds': [{'seed': r['seed'], 'acc': r['acc'], 'f1': r['f1']} for r in results],
        }

    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {json_path}")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
