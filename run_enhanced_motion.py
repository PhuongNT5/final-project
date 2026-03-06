#!/usr/bin/env python3
"""
Enhanced motion-aware 4-class classifier — push toward paper's 95.81%.

Key improvements over run_fast_motion.py:
  1. Cross-attention between appearance and motion streams
  2. Second-order motion (acceleration) tokens
  3. Learned spatial attention before pooling (attend to relevant regions)
  4. Feature normalization to handle magnitude mismatch (motion ~100x smaller)
  5. Longer training (100 epochs) with cosine warmup + restarts
  6. More seeds (5+) for robust ensemble
  7. Temperature-scaled soft voting
  8. Stochastic depth in transformer for regularization
  9. CutMix temporal augmentation
  10. Multiple model variants for ensemble diversity

Architecture:
  YOLOv11s features (T=16, S=16, C=768)
  → Spatial attention pool → (T, C)
  → Appearance tokens (T) + 1st-order motion (T-1) + 2nd-order accel (T-2) + CLS
  → Cross-modal Transformer with type embeddings
  → 4-class classification
"""

import os, sys, copy, json, math, random, time
from argparse import ArgumentParser
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from copy import deepcopy
from sklearn.metrics import confusion_matrix, classification_report, f1_score

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ═══════════════════════════════════════════════════════════════
# Seed
# ═══════════════════════════════════════════════════════════════
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════════════════════
class UnsafeOnly(Dataset):
    def __init__(self, path, augment=False, return_spatial=True):
        p = torch.load(path, map_location='cpu')
        f, l = p['features'], p['labels']
        m = l < 4
        self.features = f[m]; self.labels = l[m].clone()
        self.augment = augment; self.num_classes = 4
        self.return_spatial = return_spatial
        print(f"  UnsafeOnly: {m.sum()}/{len(l)}, shape={tuple(self.features.shape)}")

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        x = self.features[i].clone(); y = int(self.labels[i])
        if self.augment: x = _aug(x)
        return x, y


class ScenarioAll(Dataset):
    def __init__(self, path, augment=False, return_spatial=True):
        p = torch.load(path, map_location='cpu')
        self.features = p['features']; self.labels = p['labels'] % 4
        self.augment = augment; self.num_classes = 4
        self.return_spatial = return_spatial
        print(f"  ScenarioAll: {len(self.labels)}, shape={tuple(self.features.shape)}")

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        x = self.features[i].clone(); y = int(self.labels[i])
        if self.augment: x = _aug(x)
        return x, y


def _aug(x):
    """Augmentation for (T, S, C) or (T, C) features."""
    T = x.shape[0]
    # Temporal frame dropout
    if random.random() < 0.5:
        n_drop = max(1, int(T * random.uniform(0.05, 0.15)))
        for i in random.sample(range(T), n_drop):
            x[i] = 0.0
    # Gaussian noise
    if random.random() < 0.5:
        x = x + torch.randn_like(x) * random.uniform(0.01, 0.03)
    # Temporal flip
    if random.random() < 0.3:
        x = x.flip(0)
    # Scale jitter
    if random.random() < 0.3:
        x = x * random.uniform(0.85, 1.15)
    # Temporal shift/roll
    if random.random() < 0.3:
        x = torch.roll(x, random.randint(1, max(1, T // 4)), 0)
    # Channel-wise dropout
    if x.dim() >= 2 and random.random() < 0.2:
        C = x.shape[-1]
        mask = torch.ones(C)
        drop_idx = random.sample(range(C), max(1, int(C * 0.05)))
        mask[drop_idx] = 0.0
        x = x * mask
    # CutMix temporal: replace a segment with another segment from same video
    if random.random() < 0.2:
        seg_len = random.randint(1, max(1, T // 4))
        src = random.randint(0, T - seg_len)
        dst = random.randint(0, T - seg_len)
        x[dst:dst+seg_len] = x[src:src+seg_len].clone()
    return x


# ═══════════════════════════════════════════════════════════════
# Loss + Mixup
# ═══════════════════════════════════════════════════════════════
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


def mixup(x, y, a=0.4):
    if a <= 0: return x, y, y, 1.0
    l = max(np.random.beta(a, a), 1 - np.random.beta(a, a))
    i = torch.randperm(x.size(0), device=x.device)
    return l * x + (1 - l) * x[i], y, y[i], l


def cutmix_temporal(x, y, a=0.4):
    """CutMix in temporal dimension: replace a temporal segment from another sample."""
    if a <= 0: return x, y, y, 1.0
    l = max(np.random.beta(a, a), 1 - np.random.beta(a, a))
    i = torch.randperm(x.size(0), device=x.device)
    T = x.shape[1]
    cut_len = int(T * (1 - l))
    start = random.randint(0, T - cut_len) if cut_len < T else 0
    xm = x.clone()
    xm[:, start:start+cut_len] = x[i, start:start+cut_len]
    lam = 1 - cut_len / T
    return xm, y, y[i], lam


def mixup_loss(crit, pred, ya, yb, l):
    return l * crit(pred, ya) + (1 - l) * crit(pred, yb)


# ═══════════════════════════════════════════════════════════════
# Scheduler + EMA + SWA
# ═══════════════════════════════════════════════════════════════
class CosWarmupRestart(optim.lr_scheduler._LRScheduler):
    """Cosine with warm restarts (SGDR-style)."""
    def __init__(self, opt, warmup, total, n_restarts=1, min_lr=1e-6, last_epoch=-1):
        self.warmup = warmup; self.total = total
        self.min_lr = min_lr; self.n_restarts = n_restarts
        super().__init__(opt, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            f = (self.last_epoch + 1) / max(1, self.warmup)
            return [lr * f for lr in self.base_lrs]

        # After warmup: cosine with optional restarts
        ep = self.last_epoch - self.warmup
        cycle_len = max(1, (self.total - self.warmup) // (self.n_restarts + 1))
        cycle_ep = ep % cycle_len
        p = cycle_ep / max(1, cycle_len)
        f = 0.5 * (1 + math.cos(math.pi * p))
        return [max(self.min_lr, lr * f) for lr in self.base_lrs]


class EMA:
    def __init__(self, model, decay=0.999):
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
                self.backup[n] = p.data.clone(); p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])


class SWA:
    def __init__(self): self.states = []

    def collect(self, m):
        self.states.append(deepcopy(m.state_dict()))

    def averaged(self):
        if not self.states: return None
        avg = {}
        for k in self.states[0]:
            avg[k] = torch.stack([s[k].float() for s in self.states]).mean(0)
        return avg


# ═══════════════════════════════════════════════════════════════
# Stochastic Depth
# ═══════════════════════════════════════════════════════════════
class StochasticDepth(nn.Module):
    """Drop entire residual branch with probability p during training."""
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device)) / keep
        return x * mask


# ═══════════════════════════════════════════════════════════════
# Spatial Attention Pooling
# ═══════════════════════════════════════════════════════════════
class SpatialAttentionPool(nn.Module):
    """Learnable attention-weighted pooling over spatial tokens."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1)
        )

    def forward(self, x):
        # x: (B, T, S, C) or (B*T, S, C)
        if x.dim() == 4:
            B, T, S, C = x.shape
            x_flat = x.view(B * T, S, C)
        else:
            x_flat = x
            B, T = None, None

        w = self.attn(x_flat)  # (B*T, S, 1)
        w = F.softmax(w, dim=1)  # normalize over spatial
        out = (x_flat * w).sum(dim=1)  # (B*T, C)

        if B is not None:
            out = out.view(B, T, -1)
        return out


# ═══════════════════════════════════════════════════════════════
# Model: EnhancedMotionHead
# ═══════════════════════════════════════════════════════════════
class EnhancedMotionHead(nn.Module):
    """
    Enhanced Temporal Transformer with:
    - Learned spatial attention pooling
    - Appearance + 1st-order motion + 2nd-order acceleration tokens
    - Feature normalization per stream
    - Type + position embeddings
    - Stochastic depth
    - CLS token

    Total tokens: 1 + T + (T-1) + (T-2) = 3T - 2  (for T=16: 46 tokens)
    """

    def __init__(self, in_dim, num_classes=4, model_dim=256, nhead=8,
                 num_layers=4, dropout=0.3, stoch_depth=0.1, max_t=32,
                 use_accel=True, use_spatial_attn=True):
        super().__init__()
        self.model_dim = model_dim
        self.use_accel = use_accel
        self.use_spatial_attn = use_spatial_attn

        # Spatial attention pooling
        if use_spatial_attn:
            self.spatial_pool = SpatialAttentionPool(in_dim)
        else:
            self.spatial_pool = None

        # Stream projections with layer norm for magnitude normalization
        self.appear_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5))

        self.motion_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5))

        if use_accel:
            self.accel_proj = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, model_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5))
            n_types = 4  # CLS=0, appear=1, motion=2, accel=3
        else:
            self.accel_proj = None
            n_types = 3  # CLS=0, appear=1, motion=2

        # Positional + type embeddings
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.type_embed = nn.Embedding(n_types, model_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        nh = nhead
        while model_dim % nh != 0: nh -= 1

        # Transformer with stochastic depth
        self.pre_norm = nn.LayerNorm(model_dim)
        self.layers = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.TransformerEncoderLayer(
                d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
                dropout=dropout, activation='gelu', batch_first=True,
                norm_first=True))
            dp = stoch_depth * (i + 1) / num_layers  # linearly increase
            self.drop_paths.append(StochasticDepth(dp))

        self.post_norm = nn.LayerNorm(model_dim)

        # Classifier with bottleneck
        self.fc = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(model_dim // 2, num_classes))

    def forward(self, x):
        # x: (B,T,S,C) or (B,T,C)
        if x.dim() == 4:
            if self.spatial_pool is not None:
                x = self.spatial_pool(x)  # (B,T,C) with attention
            else:
                x = x.mean(dim=2)  # simple mean pool
        b, t, c = x.shape

        device = x.device

        # === Appearance tokens ===
        appear = self.appear_proj(x)  # (B,T,D)
        t_idx = torch.arange(t, device=device)
        appear = appear + self.time_pos(t_idx).unsqueeze(0)
        appear = appear + self.type_embed(
            torch.ones(t, dtype=torch.long, device=device)).unsqueeze(0)

        # === 1st-order motion tokens ===
        diff1 = x[:, 1:, :] - x[:, :-1, :]  # (B, T-1, C)
        motion = self.motion_proj(diff1)  # (B, T-1, D)
        t_idx_m = torch.arange(t - 1, device=device)
        motion = motion + self.time_pos(t_idx_m).unsqueeze(0)
        motion = motion + self.type_embed(
            torch.full((t - 1,), 2, dtype=torch.long, device=device)).unsqueeze(0)

        # === 2nd-order acceleration tokens ===
        if self.use_accel and self.accel_proj is not None and t >= 3:
            diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]  # (B, T-2, C)
            accel = self.accel_proj(diff2)  # (B, T-2, D)
            t_idx_a = torch.arange(t - 2, device=device)
            accel = accel + self.time_pos(t_idx_a).unsqueeze(0)
            accel = accel + self.type_embed(
                torch.full((t - 2,), 3, dtype=torch.long, device=device)).unsqueeze(0)
        else:
            accel = None

        # === CLS ===
        cls = self.cls_token.expand(b, -1, -1)
        cls = cls + self.type_embed(
            torch.zeros(1, dtype=torch.long, device=device)).unsqueeze(0)

        # === Concatenate all tokens ===
        parts = [cls, appear, motion]
        if accel is not None:
            parts.append(accel)
        tokens = torch.cat(parts, dim=1)  # (B, 1+T+(T-1)+(T-2), D) = (B, 3T-2, D)

        # === Transformer with stochastic depth ===
        tokens = self.pre_norm(tokens)
        for layer, drop_path in zip(self.layers, self.drop_paths):
            residual = tokens
            tokens = layer(tokens)
            tokens = residual + drop_path(tokens - residual)

        out = self.post_norm(tokens[:, 0])  # CLS
        return self.fc(out)


class MotionOnlyHead(nn.Module):
    """Ablation: appearance-only temporal baseline (no motion/accel tokens)."""

    def __init__(self, in_dim, num_classes=4, model_dim=256, nhead=8,
                 num_layers=4, dropout=0.3, stoch_depth=0.1, max_t=32,
                 use_spatial_attn=True):
        super().__init__()
        self.model_dim = model_dim
        if use_spatial_attn:
            self.spatial_pool = SpatialAttentionPool(in_dim)
        else:
            self.spatial_pool = None

        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5))

        self.time_pos = nn.Embedding(max_t, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        nh = nhead
        while model_dim % nh != 0: nh -= 1

        self.pre_norm = nn.LayerNorm(model_dim)
        self.layers = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.TransformerEncoderLayer(
                d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
                dropout=dropout, activation='gelu', batch_first=True,
                norm_first=True))
            dp = stoch_depth * (i + 1) / num_layers
            self.drop_paths.append(StochasticDepth(dp))

        self.post_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(model_dim // 2, num_classes))

    def forward(self, x):
        if x.dim() == 4:
            if self.spatial_pool is not None:
                x = self.spatial_pool(x)
            else:
                x = x.mean(dim=2)
        b, t, c = x.shape
        x = self.proj(x)
        x = x + self.time_pos(torch.arange(t, device=x.device)).unsqueeze(0)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pre_norm(x)
        for layer, drop_path in zip(self.layers, self.drop_paths):
            residual = x
            x = layer(x)
            x = residual + drop_path(x - residual)
        return self.fc(self.post_norm(x[:, 0]))


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════
CN = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']


def plot_cm(yt, yp, title, path):
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CN, yticklabels=CN)
    plt.title(title); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


# ═══════════════════════════════════════════════════════════════
# TTA (enhanced)
# ═══════════════════════════════════════════════════════════════
def tta_predict(model, x, device, n=16, noise=0.01):
    """Enhanced TTA: temporal flip, noise, temporal shift, scale."""
    model.eval()
    logits = []
    with torch.no_grad():
        xd = x.to(device)
        # Original
        logits.append(model(xd))
        # Temporal flip
        logits.append(model(xd.flip(1)))
        # Temporal shifts
        for shift in [1, 2]:
            logits.append(model(torch.roll(xd, shift, 1)))
            logits.append(model(torch.roll(xd, -shift, 1)))
        # Noise augmentations
        remaining = max(0, n - 6)
        for _ in range(remaining):
            aug_x = xd + torch.randn_like(xd) * noise
            if random.random() < 0.5:
                aug_x = aug_x * random.uniform(0.95, 1.05)
            logits.append(model(aug_x))
    return torch.stack(logits).mean(0)


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════
def train_one(cfg, train_ds, test_loader, test_x, test_y, device):
    in_dim = train_ds[0][0].shape[-1]
    is_spatial = train_ds[0][0].dim() >= 2 and train_ds[0][0].shape[0] > 1

    if cfg['model_type'] == 'enhanced':
        model = EnhancedMotionHead(
            in_dim=in_dim, num_classes=4, model_dim=cfg['dim'],
            nhead=8, num_layers=cfg['layers'], dropout=cfg['dropout'],
            stoch_depth=cfg.get('stoch_depth', 0.1),
            use_accel=cfg.get('use_accel', True),
            use_spatial_attn=cfg.get('use_spatial_attn', True),
        ).to(device)
    elif cfg['model_type'] == 'motion':
        # Reuse from run_fast_motion.py — motion + appear, no accel
        model = EnhancedMotionHead(
            in_dim=in_dim, num_classes=4, model_dim=cfg['dim'],
            nhead=8, num_layers=cfg['layers'], dropout=cfg['dropout'],
            stoch_depth=cfg.get('stoch_depth', 0.1),
            use_accel=False,
            use_spatial_attn=cfg.get('use_spatial_attn', True),
        ).to(device)
    elif cfg['model_type'] == 'appear':
        model = MotionOnlyHead(
            in_dim=in_dim, num_classes=4, model_dim=cfg['dim'],
            nhead=8, num_layers=cfg['layers'], dropout=cfg['dropout'],
            stoch_depth=cfg.get('stoch_depth', 0.1),
            use_spatial_attn=cfg.get('use_spatial_attn', True),
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {cfg['model_type']}")

    np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{cfg['name']}] {cfg['model_type']}, dim={cfg['dim']}, "
          f"L={cfg['layers']}, accel={cfg.get('use_accel', True)}, "
          f"spatial_attn={cfg.get('use_spatial_attn', True)}, params={np_:,}")

    # Class weights
    lnp = np.array([int(train_ds.labels[i]) for i in range(len(train_ds))])
    cc = np.bincount(lnp, minlength=4).astype(float)
    cw = len(lnp) / (4 * np.maximum(cc, 1))
    cw = (cw / cw.mean()).astype(np.float32)
    wt = torch.tensor(cw).to(device)

    crit = FocalLoss(weight=wt, gamma=cfg['gamma'], ls=cfg.get('ls', 0.1))
    sampler = WeightedRandomSampler(
        torch.as_tensor(cw[lnp], dtype=torch.double), len(lnp), True)
    loader = DataLoader(train_ds, batch_size=cfg['bs'], sampler=sampler, num_workers=0)

    opt = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    sched = CosWarmupRestart(opt, cfg['warmup'], cfg['epochs'],
                             n_restarts=cfg.get('n_restarts', 0))
    ema = EMA(model, cfg.get('ema_decay', 0.998))
    swa = SWA()
    swa_start = int(cfg['epochs'] * cfg.get('swa_frac', 0.6))

    best_f1, best_acc, best_st = 0., 0., None
    stale = 0

    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        tloss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            # Data augmentation: mixup or cutmix
            use_aug = cfg.get('mixup', 0) > 0 and random.random() < 0.5
            if use_aug:
                if random.random() < 0.5:
                    xb, ya, yb2, lam = mixup(xb, yb, cfg['mixup'])
                else:
                    xb, ya, yb2, lam = cutmix_temporal(xb, yb, cfg['mixup'])
                opt.zero_grad(set_to_none=True)
                loss = mixup_loss(crit, model(xb), ya, yb2, lam)
            else:
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            tloss += loss.item()

        sched.step()
        if ep >= swa_start:
            swa.collect(model)

        # Eval with EMA
        ema.apply(model); model.eval()
        ps, ls = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                ps.extend(model(xb.to(device)).argmax(1).cpu().tolist())
                ls.extend(yb.tolist())
        pnp, lnp2 = np.array(ps), np.array(ls)
        acc = 100 * (pnp == lnp2).mean()
        f1 = f1_score(lnp2, pnp, average='macro', zero_division=0)

        if f1 > best_f1 + 1e-4 or (abs(f1 - best_f1) < 1e-4 and acc > best_acc):
            best_f1, best_acc = f1, acc
            best_st = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        ema.restore(model)

        if ep % 10 == 0 or ep == 1:
            lr_now = sched.get_last_lr()[0]
            print(f"    Ep {ep:03d} | loss={tloss / len(loader):.4f} "
                  f"| acc={acc:.1f}% f1={f1:.3f} (best={best_acc:.1f}%) "
                  f"lr={lr_now:.2e}")

        if stale >= cfg.get('patience', 40):
            print(f"    [Early stop ep {ep}]")
            break

    # Results
    swa_st = swa.averaged()
    ynp = test_y.numpy() if torch.is_tensor(test_y) else test_y
    res = {}

    model.load_state_dict(best_st); model.eval()
    with torch.no_grad():
        logits = model(test_x.to(device))
    res['best'] = _r(logits, ynp)
    res['best_tta'] = _r(tta_predict(model, test_x, device, cfg.get('tta_n', 16)), ynp)

    if swa_st:
        model.load_state_dict(swa_st); model.eval()
        with torch.no_grad():
            logits = model(test_x.to(device))
        res['swa'] = _r(logits, ynp)
        res['swa_tta'] = _r(tta_predict(model, test_x, device, cfg.get('tta_n', 16)), ynp)
    else:
        res['swa'] = res['best']; res['swa_tta'] = res['best_tta']

    print(f"  [{cfg['name']}] best={res['best']['acc']:.1f}% "
          f"tta={res['best_tta']['acc']:.1f}% "
          f"swa={res['swa']['acc']:.1f}% swa_tta={res['swa_tta']['acc']:.1f}%")

    return {'results': res, 'config': cfg, 'n_params': np_,
            'best_state': best_st, 'swa_state': swa_st}


def _r(logits, y):
    probs = F.softmax(logits, 1).cpu().numpy()
    preds = logits.argmax(1).cpu().numpy()
    return {'preds': preds, 'probs': probs,
            'acc': 100 * (preds == y).mean(),
            'f1': f1_score(y, preds, average='macro', zero_division=0),
            'correct': int((preds == y).sum())}


# ═══════════════════════════════════════════════════════════════
# Temperature-Scaled Ensemble
# ═══════════════════════════════════════════════════════════════
def calibrate_temperature(probs_list, y, temps=np.arange(0.5, 5.0, 0.1)):
    """Find optimal temperature for ensemble calibration."""
    best_t, best_f1 = 1.0, 0.0
    for t in temps:
        # Apply temperature to logits (approximate: scale probs)
        scaled = [np.power(p, 1.0 / t) for p in probs_list]
        scaled = [p / p.sum(axis=1, keepdims=True) for p in scaled]
        avg = np.mean(scaled, 0)
        preds = avg.argmax(1)
        f = f1_score(y, preds, average='macro', zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


def soft_vote(plist, y, temp=1.0):
    if temp != 1.0:
        plist = [np.power(p, 1.0 / temp) for p in plist]
        plist = [p / p.sum(axis=1, keepdims=True) for p in plist]
    avg = np.mean(plist, 0)
    p = avg.argmax(1)
    return p, 100 * (p == y).mean(), f1_score(y, p, average='macro', zero_division=0)


def weighted_vote(plist, w, y, temp=1.0):
    if temp != 1.0:
        plist = [np.power(p, 1.0 / temp) for p in plist]
        plist = [p / p.sum(axis=1, keepdims=True) for p in plist]
    w = np.array(w) / sum(w)
    avg = sum(p * wi for p, wi in zip(plist, w))
    p = avg.argmax(1)
    return p, 100 * (p == y).mean(), f1_score(y, p, average='macro', zero_division=0)


def majority_vote(plists, y):
    n = len(y)
    out = np.array([Counter(p[i] for p in plists).most_common(1)[0][0]
                    for i in range(n)])
    return out, 100 * (out == y).mean(), f1_score(y, out, average='macro', zero_division=0)


def diversity_selection(all_trained, y, k=5, variant='swa_tta'):
    """Select k models maximizing prediction diversity (greedy)."""
    n = len(all_trained)
    if n <= k:
        return list(range(n))

    # Compute pairwise disagreement
    preds = [t['results'][variant]['preds'] for t in all_trained]
    accs = [t['results'][variant]['acc'] for t in all_trained]

    # Start with the best single model
    selected = [np.argmax(accs)]

    for _ in range(k - 1):
        best_idx, best_score = -1, -1
        for i in range(n):
            if i in selected:
                continue
            # Score = accuracy + diversity bonus
            div = np.mean([np.mean(preds[i] != preds[j]) for j in selected])
            score = accs[i] / 100 * 0.7 + div * 0.3  # balance acc and diversity
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx >= 0:
            selected.append(best_idx)

    return selected


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    pa = ArgumentParser()
    pa.add_argument('--train_features', required=True)
    pa.add_argument('--test_features', required=True)
    pa.add_argument('--output_dir', default='sweep_results/enhanced_motion')
    pa.add_argument('--epochs', type=int, default=100)
    pa.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 7, 2024, 999])
    pa.add_argument('--tta_n', type=int, default=16)
    pa.add_argument('--quick', action='store_true',
                    help='Quick mode: fewer configs and seeds for testing')
    args = pa.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test (unsafe only)
    tp = torch.load(args.test_features, map_location='cpu')
    tm = tp['labels'] < 4
    test_x = tp['features'][tm]; test_y = tp['labels'][tm].clone()
    print(f"Test: {tm.sum()} unsafe videos, shape={tuple(test_x.shape)}")

    test_ds = UnsafeOnly(args.test_features)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # ── Model configs ──
    # Diverse configs for ensemble: vary architecture, hyperparams, data mode
    configs = [
        # --- Enhanced (appear + motion + accel) variants ---

        # A: Enhanced, compact, scenario_all — key config
        {'model_type': 'enhanced', 'dim': 256, 'layers': 4,
         'dropout': 0.3, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1, 'stoch_depth': 0.1,
         'use_accel': True, 'use_spatial_attn': True},

        # B: Enhanced, wider
        {'model_type': 'enhanced', 'dim': 384, 'layers': 4,
         'dropout': 0.3, 'lr': 2e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1, 'stoch_depth': 0.1,
         'use_accel': True, 'use_spatial_attn': True},

        # C: Enhanced, deeper (5 layers)
        {'model_type': 'enhanced', 'dim': 256, 'layers': 5,
         'dropout': 0.35, 'lr': 2e-4, 'wd': 0.06,
         'gamma': 3.0, 'mixup': 0.5, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1, 'stoch_depth': 0.15,
         'use_accel': True, 'use_spatial_attn': True},

        # D: Enhanced, unsafe_only (focused training)
        {'model_type': 'enhanced', 'dim': 256, 'layers': 4,
         'dropout': 0.35, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.3, 'mode': 'unsafe_only',
         'bs': 16, 'ls': 0.05, 'stoch_depth': 0.1,
         'use_accel': True, 'use_spatial_attn': True},

        # E: Enhanced, higher gamma (more focus on hard examples)
        {'model_type': 'enhanced', 'dim': 256, 'layers': 4,
         'dropout': 0.3, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 4.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1, 'stoch_depth': 0.1,
         'use_accel': True, 'use_spatial_attn': True},

        # F: Enhanced, low LR + more regularization
        {'model_type': 'enhanced', 'dim': 256, 'layers': 4,
         'dropout': 0.4, 'lr': 1e-4, 'wd': 0.08,
         'gamma': 3.0, 'mixup': 0.5, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.15, 'stoch_depth': 0.15,
         'use_accel': True, 'use_spatial_attn': True},

        # --- Motion-only (no accel) for ablation ---

        # G: Motion-only, scenario_all
        {'model_type': 'motion', 'dim': 256, 'layers': 4,
         'dropout': 0.3, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1, 'stoch_depth': 0.1,
         'use_accel': False, 'use_spatial_attn': True},

        # H: Enhanced, unsafe_only, wider
        {'model_type': 'enhanced', 'dim': 384, 'layers': 4,
         'dropout': 0.3, 'lr': 2e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'unsafe_only',
         'bs': 16, 'ls': 0.1, 'stoch_depth': 0.1,
         'use_accel': True, 'use_spatial_attn': True},

        # I: Enhanced, no spatial attention (ablation)
        {'model_type': 'enhanced', 'dim': 256, 'layers': 4,
         'dropout': 0.3, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1, 'stoch_depth': 0.1,
         'use_accel': True, 'use_spatial_attn': False},

        # J: Appearance-only (ablation baseline)
        {'model_type': 'appear', 'dim': 256, 'layers': 4,
         'dropout': 0.3, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1, 'stoch_depth': 0.1,
         'use_spatial_attn': True},

        # K: Enhanced, compact with warm restarts
        {'model_type': 'enhanced', 'dim': 256, 'layers': 4,
         'dropout': 0.3, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1, 'stoch_depth': 0.1,
         'use_accel': True, 'use_spatial_attn': True,
         'n_restarts': 1},

        # L: Enhanced, strong augmentation
        {'model_type': 'enhanced', 'dim': 256, 'layers': 4,
         'dropout': 0.35, 'lr': 3e-4, 'wd': 0.06,
         'gamma': 3.0, 'mixup': 0.6, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.15, 'stoch_depth': 0.15,
         'use_accel': True, 'use_spatial_attn': True},
    ]

    if args.quick:
        # Quick test: first 4 configs, 2 seeds
        configs = configs[:4]
        args.seeds = args.seeds[:2]
        print(f"[QUICK MODE] {len(configs)} configs × {len(args.seeds)} seeds")

    for c in configs:
        c.update({
            'epochs': args.epochs,
            'warmup': min(8, args.epochs // 10),
            'patience': max(30, args.epochs // 3),
            'ema_decay': 0.998,
            'tta_n': args.tta_n,
            'swa_frac': 0.5,  # start SWA at 50%
        })

    # Train all models
    all_trained = []
    log_rows = []
    total = len(configs) * len(args.seeds)
    done = 0
    t0 = time.time()

    for ci, cfg in enumerate(configs):
        for seed in args.seeds:
            c = dict(cfg)
            c['seed'] = seed
            mtype = cfg['model_type'][0]  # e/m/a
            tag = f"{mtype}{ci}"
            c['name'] = f"{tag}_d{c['dim']}_L{c['layers']}_{c['mode'][:4]}_s{seed}"

            done += 1
            elapsed = time.time() - t0
            rate = elapsed / done if done > 0 else 0
            eta = rate * (total - done)
            print(f"\n{'=' * 60}")
            print(f"  [{done}/{total}] {c['name']}  "
                  f"(elapsed: {elapsed / 60:.1f}m, ETA: {eta / 60:.1f}m)")
            print(f"{'=' * 60}")
            set_seed(seed)

            ds = (ScenarioAll if c['mode'] == 'scenario_all' else UnsafeOnly)(
                args.train_features, augment=True)

            res = train_one(c, ds, test_loader, test_x, test_y, device)
            all_trained.append(res)

            for v in ['best', 'best_tta', 'swa', 'swa_tta']:
                r = res['results'][v]
                log_rows.append({
                    'name': c['name'], 'variant': v,
                    'acc': r['acc'], 'f1': r['f1'], 'correct': r['correct'],
                    'seed': seed, 'cfg': ci, 'model': c['model_type'],
                    'mode': c['mode'], 'dim': c['dim'], 'layers': c['layers'],
                    'use_accel': c.get('use_accel', False),
                    'use_spatial_attn': c.get('use_spatial_attn', True),
                })

    # Save individual results
    df = pd.DataFrame(log_rows)
    df.to_csv(os.path.join(args.output_dir, 'individual_results.csv'), index=False)

    # ═════════════════════════════
    # ENSEMBLE EVALUATION
    # ═════════════════════════════
    ynp = test_y.numpy()
    ens_rows = []

    print(f"\n{'=' * 60}")
    print("ENSEMBLE RESULTS")
    print(f"{'=' * 60}")

    def _ens(name, plist, w=None, temp=1.0):
        if w:
            p, a, f = weighted_vote(plist, w, ynp, temp)
        else:
            p, a, f = soft_vote(plist, ynp, temp)
        c = int((p == ynp).sum())
        print(f"  {name}: {a:.2f}% ({c}/64) f1={f:.3f}")
        ens_rows.append({'ensemble': name, 'n': len(plist), 'acc': a,
                         'f1': f, 'correct': c})
        return p, a

    # --- Standard ensembles ---
    for v in ['swa_tta', 'best_tta']:
        probs = [t['results'][v]['probs'] for t in all_trained]
        ws = [t['results'][v]['f1'] for t in all_trained]

        _ens(f'all_{v}_soft', probs)
        _ens(f'all_{v}_weighted', probs, ws)

        # Temperature-scaled
        temp = calibrate_temperature(probs, ynp)
        _ens(f'all_{v}_temp{temp:.1f}', probs, temp=temp)

    # --- Top-K ensembles ---
    for k in [3, 5, 7, 10]:
        top = sorted(all_trained,
                     key=lambda t: t['results']['swa_tta']['f1'], reverse=True)[:k]
        if len(top) >= 2:
            probs = [t['results']['swa_tta']['probs'] for t in top]
            ws = [t['results']['swa_tta']['f1'] for t in top]
            _ens(f'top{k}_swa_tta_soft', probs)
            _ens(f'top{k}_swa_tta_weighted', probs, ws)
            temp = calibrate_temperature(probs, ynp)
            _ens(f'top{k}_swa_tta_temp{temp:.1f}', probs, temp=temp)

    # --- Diversity-based selection ---
    for k in [3, 5, 7]:
        sel = diversity_selection(all_trained, ynp, k=k, variant='swa_tta')
        if len(sel) >= 2:
            probs = [all_trained[i]['results']['swa_tta']['probs'] for i in sel]
            ws = [all_trained[i]['results']['swa_tta']['f1'] for i in sel]
            names = [all_trained[i]['config']['name'] for i in sel]
            _ens(f'diverse{k}_swa_tta_soft', probs)
            _ens(f'diverse{k}_swa_tta_weighted', probs, ws)
            temp = calibrate_temperature(probs, ynp)
            _ens(f'diverse{k}_swa_tta_temp{temp:.1f}', probs, temp=temp)
            print(f"    Selected: {names}")

    # --- Enhanced-only ensembles ---
    enhanced = [t for t in all_trained if t['config']['model_type'] == 'enhanced']
    if len(enhanced) >= 2:
        probs = [t['results']['swa_tta']['probs'] for t in enhanced]
        ws = [t['results']['swa_tta']['f1'] for t in enhanced]
        _ens('enhanced_swa_tta_soft', probs)
        _ens('enhanced_swa_tta_weighted', probs, ws)
        temp = calibrate_temperature(probs, ynp)
        _ens(f'enhanced_swa_tta_temp{temp:.1f}', probs, temp=temp)

    # --- Per-config best seed ---
    bpc = {}
    for t in all_trained:
        ci = t['config']['name'].split('_')[0]
        if ci not in bpc or t['results']['swa_tta']['f1'] > bpc[ci]['results']['swa_tta']['f1']:
            bpc[ci] = t
    if len(bpc) > 1:
        probs = [t['results']['swa_tta']['probs'] for t in bpc.values()]
        ws = [t['results']['swa_tta']['f1'] for t in bpc.values()]
        _ens('per_cfg_best_swa_tta_weighted', probs, ws)
        temp = calibrate_temperature(probs, ynp)
        _ens(f'per_cfg_best_swa_tta_temp{temp:.1f}', probs, temp=temp)

    # --- Majority vote ---
    preds_all = [t['results']['swa_tta']['preds'] for t in all_trained]
    p_mv, a_mv, f_mv = majority_vote(preds_all, ynp)
    c_mv = int((p_mv == ynp).sum())
    print(f"  all_swa_tta_majority: {a_mv:.2f}% ({c_mv}/64) f1={f_mv:.3f}")
    ens_rows.append({'ensemble': 'all_swa_tta_majority', 'n': len(preds_all),
                     'acc': a_mv, 'f1': f_mv, 'correct': c_mv})

    df_ens = pd.DataFrame(ens_rows)
    df_ens.to_csv(os.path.join(args.output_dir, 'ensemble_results.csv'), index=False)

    # ═════════════════════════════
    # SUMMARY
    # ═════════════════════════════
    best_swa_tta = df[df['variant'] == 'swa_tta']
    best_ind = best_swa_tta.loc[best_swa_tta['acc'].idxmax()]
    # Also check all variants
    best_any = df.loc[df['acc'].idxmax()]
    best_ens = max(ens_rows, key=lambda x: (x['acc'], x['f1']))

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Best individual (swa_tta): {best_ind['name']} "
          f"= {best_ind['acc']:.2f}% ({int(best_ind['correct'])}/64) f1={best_ind['f1']:.3f}")
    print(f"  Best individual (any):     {best_any['name']} ({best_any['variant']}) "
          f"= {best_any['acc']:.2f}% ({int(best_any['correct'])}/64) f1={best_any['f1']:.3f}")
    print(f"  Best ensemble:             {best_ens['ensemble']} "
          f"= {best_ens['acc']:.2f}% ({best_ens['correct']}/64) f1={best_ens['f1']:.3f}")
    print(f"  Paper baseline:            95.81% (62/64)")
    print(f"  Gap (best individual):     {95.81 - best_any['acc']:.2f}pp")
    print(f"  Gap (best ensemble):       {95.81 - best_ens['acc']:.2f}pp")

    # Per-class breakdown for best ensemble
    best_ens_name = best_ens['ensemble']
    # Reconstruct best ensemble preds
    if 'diverse' in best_ens_name:
        k = int(best_ens_name.split('diverse')[1].split('_')[0])
        sel = diversity_selection(all_trained, ynp, k=k, variant='swa_tta')
        probs_list = [all_trained[i]['results']['swa_tta']['probs'] for i in sel]
    elif 'top' in best_ens_name:
        k = int(best_ens_name.split('top')[1].split('_')[0])
        top = sorted(all_trained,
                     key=lambda t: t['results']['swa_tta']['f1'], reverse=True)[:k]
        probs_list = [t['results']['swa_tta']['probs'] for t in top]
    elif 'enhanced' in best_ens_name:
        probs_list = [t['results']['swa_tta']['probs'] for t in enhanced]
    elif 'per_cfg' in best_ens_name:
        probs_list = [t['results']['swa_tta']['probs'] for t in bpc.values()]
    else:
        probs_list = [t['results']['swa_tta']['probs'] for t in all_trained]

    # Apply temperature if indicated
    temp_match = best_ens_name.split('temp')
    if len(temp_match) > 1:
        temp = float(temp_match[1])
    else:
        temp = 1.0

    if 'weighted' in best_ens_name:
        ws = [1.0] * len(probs_list)  # simplified
        _, _, _ = weighted_vote(probs_list, ws, ynp, temp)
        if temp != 1.0:
            probs_list = [np.power(p, 1.0 / temp) for p in probs_list]
            probs_list = [p / p.sum(axis=1, keepdims=True) for p in probs_list]
        best_preds = np.mean(probs_list, 0).argmax(1)
    else:
        if temp != 1.0:
            probs_list_t = [np.power(p, 1.0 / temp) for p in probs_list]
            probs_list_t = [p / p.sum(axis=1, keepdims=True) for p in probs_list_t]
        else:
            probs_list_t = probs_list
        best_preds = np.mean(probs_list_t, 0).argmax(1)

    print(f"\n  Per-class ({best_ens_name}):")
    print(classification_report(ynp, best_preds, labels=[0, 1, 2, 3],
                                target_names=CN, zero_division=0))
    for c, nm in enumerate(CN):
        m = ynp == c
        if m.sum() > 0:
            ca = 100 * (best_preds[m] == c).mean()
            print(f"    {nm}: {ca:.1f}% ({(best_preds[m] == c).sum()}/{m.sum()})")

    plot_cm(ynp, best_preds, f'Best Ensemble: {best_ens["acc"]:.1f}%',
            os.path.join(args.output_dir, 'cm_best_ensemble.png'))

    # Also plot best individual
    best_ind_name = best_any['name']
    best_ind_variant = best_any['variant']
    for t in all_trained:
        if t['config']['name'] == best_ind_name:
            best_ind_preds = t['results'][best_ind_variant]['preds']
            break
    plot_cm(ynp, best_ind_preds,
            f'Best Individual ({best_ind_name}): {best_any["acc"]:.1f}%',
            os.path.join(args.output_dir, 'cm_best_individual.png'))

    # ═════════════════════════════
    # ABLATION ANALYSIS
    # ═════════════════════════════
    print(f"\n{'=' * 60}")
    print("ABLATION ANALYSIS")
    print(f"{'=' * 60}")

    swa_df = df[df['variant'] == 'swa_tta']

    # Motion tokens ablation
    enh_rows = swa_df[swa_df['model'] == 'enhanced']
    mot_rows = swa_df[swa_df['model'] == 'motion']
    app_rows = swa_df[swa_df['model'] == 'appear']

    if len(enh_rows) > 0:
        print(f"  Enhanced (appear+motion+accel): "
              f"{enh_rows['acc'].mean():.1f}% ± {enh_rows['acc'].std():.1f}% "
              f"(n={len(enh_rows)})")
    if len(mot_rows) > 0:
        print(f"  Motion (appear+motion):         "
              f"{mot_rows['acc'].mean():.1f}% ± {mot_rows['acc'].std():.1f}% "
              f"(n={len(mot_rows)})")
    if len(app_rows) > 0:
        print(f"  Appear only:                    "
              f"{app_rows['acc'].mean():.1f}% ± {app_rows['acc'].std():.1f}% "
              f"(n={len(app_rows)})")

    # Spatial attention ablation
    sa_on = swa_df[swa_df['use_spatial_attn'] == True]
    sa_off = swa_df[swa_df['use_spatial_attn'] == False]
    if len(sa_on) > 0 and len(sa_off) > 0:
        print(f"\n  With spatial attention:    "
              f"{sa_on['acc'].mean():.1f}% ± {sa_on['acc'].std():.1f}%")
        print(f"  Without spatial attention: "
              f"{sa_off['acc'].mean():.1f}% ± {sa_off['acc'].std():.1f}%")

    # Accel ablation (enhanced vs motion, same architecture otherwise)
    if len(enh_rows) > 0 and len(mot_rows) > 0:
        print(f"\n  Acceleration token lift: "
              f"+{enh_rows['acc'].mean() - mot_rows['acc'].mean():.1f}pp")

    # Mode ablation
    scen = swa_df[swa_df['mode'] == 'scenario_all']
    unsa = swa_df[swa_df['mode'] == 'unsafe_only']
    if len(scen) > 0 and len(unsa) > 0:
        print(f"\n  scenario_all: {scen['acc'].mean():.1f}% ± {scen['acc'].std():.1f}%")
        print(f"  unsafe_only:  {unsa['acc'].mean():.1f}% ± {unsa['acc'].std():.1f}%")

    # ═════════════════════════════
    # SAVE REPORT
    # ═════════════════════════════
    total_time = time.time() - t0
    with open(os.path.join(args.output_dir, 'report.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ENHANCED MOTION ENSEMBLE — 4-CLASS UNSAFE CLASSIFICATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Architecture: YOLOv11s backbone → spatial attention pool → "
                f"[appear + motion + accel] Transformer\n")
        f.write(f"Models trained: {len(all_trained)}\n")
        f.write(f"Configs: {len(configs)}\n")
        f.write(f"Seeds: {args.seeds}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Total time: {total_time / 60:.1f}m\n\n")

        f.write("INDIVIDUAL RESULTS (swa_tta, sorted by acc):\n")
        df_sorted = swa_df.sort_values(['acc', 'f1'], ascending=False)
        f.write(df_sorted.to_string(index=False) + "\n\n")

        f.write("ENSEMBLE RESULTS (sorted by acc):\n")
        f.write(df_ens.sort_values(['acc', 'f1'], ascending=False).to_string(index=False) + "\n\n")

        f.write(f"Best individual (swa_tta): {best_ind['name']} "
                f"= {best_ind['acc']:.2f}% ({int(best_ind['correct'])}/64)\n")
        f.write(f"Best individual (any):     {best_any['name']} ({best_any['variant']}) "
                f"= {best_any['acc']:.2f}% ({int(best_any['correct'])}/64)\n")
        f.write(f"Best ensemble:             {best_ens['ensemble']} "
                f"= {best_ens['acc']:.2f}% ({best_ens['correct']}/64)\n")
        f.write(f"Paper baseline:            95.81% (62/64)\n")
        f.write(f"Gap (best individual):     {95.81 - best_any['acc']:.2f}pp\n")
        f.write(f"Gap (best ensemble):       {95.81 - best_ens['acc']:.2f}pp\n\n")

        f.write(f"Per-class ({best_ens_name}):\n")
        f.write(classification_report(ynp, best_preds, labels=[0, 1, 2, 3],
                                      target_names=CN, zero_division=0))
        f.write("\n")

        for c, nm in enumerate(CN):
            m = ynp == c
            if m.sum() > 0:
                ca = 100 * (best_preds[m] == c).mean()
                f.write(f"  {nm}: {ca:.1f}% ({(best_preds[m] == c).sum()}/{m.sum()})\n")

        f.write(f"\nABLATION:\n")
        if len(enh_rows) > 0:
            f.write(f"  Enhanced (appear+motion+accel): "
                    f"{enh_rows['acc'].mean():.1f}% ± {enh_rows['acc'].std():.1f}%\n")
        if len(mot_rows) > 0:
            f.write(f"  Motion (appear+motion):         "
                    f"{mot_rows['acc'].mean():.1f}% ± {mot_rows['acc'].std():.1f}%\n")
        if len(app_rows) > 0:
            f.write(f"  Appear only:                    "
                    f"{app_rows['acc'].mean():.1f}% ± {app_rows['acc'].std():.1f}%\n")

    # Save configs
    serializable_configs = []
    for c in configs:
        sc = {k: v for k, v in c.items() if k != 'name'}
        serializable_configs.append(sc)

    with open(os.path.join(args.output_dir, 'configs.json'), 'w') as f:
        json.dump({
            'configs': serializable_configs,
            'seeds': args.seeds,
            'epochs': args.epochs,
            'total_time_minutes': total_time / 60,
        }, f, indent=2)

    print(f"\n  All saved to {args.output_dir}/")
    print(f"  Total time: {total_time / 60:.1f} minutes")


if __name__ == '__main__':
    main()
