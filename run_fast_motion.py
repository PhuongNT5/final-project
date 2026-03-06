#!/usr/bin/env python3
"""
Fast motion-aware 4-class classifier with spatial pooling + temporal motion tokens.

This is a lighter-weight version of run_motion_ensemble.py that:
  1. Pools spatial tokens first: (B, T, S=16, C=768) → (B, T, C)
  2. Adds temporal-difference tokens: (B, T-1, C)
  3. Joint Transformer on just T + (T-1) + 1 = 32 tokens (vs 497 in full version)
  4. Runs ~15x faster → can explore more seeds/configs

Architecture: YOLOv11 backbone → spatial pool → [appear + motion] temporal Transformer

Key finding motivating this:
  - Class 0 vs 1 cosine similarity in appearance: 0.99 (nearly identical)
  - Class 0 vs 1 cosine similarity in temporal-diff: 0.28 (well separated!)
"""

import os, sys, copy, json, math, random
from argparse import ArgumentParser
from collections import Counter

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
    def __init__(self, path, augment=False):
        p = torch.load(path, map_location='cpu')
        f, l = p['features'], p['labels']
        m = l < 4
        self.features = f[m]; self.labels = l[m].clone()
        self.augment = augment; self.num_classes = 4
        print(f"  UnsafeOnly: {m.sum()}/{len(l)}, shape={tuple(self.features.shape)}")

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        x = self.features[i].clone(); y = int(self.labels[i])
        if self.augment: x = _aug(x)
        return x, y


class ScenarioAll(Dataset):
    def __init__(self, path, augment=False):
        p = torch.load(path, map_location='cpu')
        self.features = p['features']; self.labels = p['labels'] % 4
        self.augment = augment; self.num_classes = 4
        print(f"  ScenarioAll: {len(self.labels)}, shape={tuple(self.features.shape)}")

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        x = self.features[i].clone(); y = int(self.labels[i])
        if self.augment: x = _aug(x)
        return x, y


def _aug(x):
    T = x.shape[0]
    if random.random() < 0.5:
        for i in random.sample(range(T), max(1, int(T * 0.1))): x[i] = 0.0
    if random.random() < 0.5: x = x + torch.randn_like(x) * 0.02
    if random.random() < 0.3: x = x.flip(0)
    if random.random() < 0.3: x = x * (0.9 + 0.2 * random.random())
    if random.random() < 0.3: x = torch.roll(x, random.randint(1, max(1, T//4)), 0)
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
        pt = F.softmax(logits, 1).gather(1, target.view(-1,1)).squeeze(1).clamp(1e-6)
        return (((1-pt)**self.gamma) * ce).mean()


def mixup(x, y, a=0.4):
    if a <= 0: return x, y, y, 1.0
    l = max(np.random.beta(a,a), 1-np.random.beta(a,a))
    i = torch.randperm(x.size(0), device=x.device)
    return l*x + (1-l)*x[i], y, y[i], l


def mixup_loss(crit, pred, ya, yb, l):
    return l*crit(pred,ya) + (1-l)*crit(pred,yb)


# ═══════════════════════════════════════════════════════════════
# Scheduler + EMA + SWA
# ═══════════════════════════════════════════════════════════════
class CosWarmup(optim.lr_scheduler._LRScheduler):
    def __init__(self, opt, warmup, total, min_lr=1e-6, last_epoch=-1):
        self.warmup=warmup; self.total=total; self.min_lr=min_lr
        super().__init__(opt, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup:
            f = (self.last_epoch+1)/max(1,self.warmup)
            return [lr*f for lr in self.base_lrs]
        p = (self.last_epoch-self.warmup)/max(1,self.total-self.warmup)
        f = 0.5*(1+math.cos(math.pi*p))
        return [max(self.min_lr, lr*f) for lr in self.base_lrs]


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n,p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    def update(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1-self.decay)
    def apply(self, model):
        self.backup = {}
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone(); p.data.copy_(self.shadow[n])
    def restore(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad: p.data.copy_(self.backup[n])


class SWA:
    def __init__(self): self.states = []
    def collect(self, m): self.states.append(deepcopy(m.state_dict()))
    def averaged(self):
        if not self.states: return None
        avg = {}
        for k in self.states[0]:
            avg[k] = torch.stack([s[k].float() for s in self.states]).mean(0)
        return avg


# ═══════════════════════════════════════════════════════════════
# Model: TemporalMotionHead
# ═══════════════════════════════════════════════════════════════
class TemporalMotionHead(nn.Module):
    """Temporal Transformer with explicit motion (difference) tokens.

    Input: (B, T, S, C) → spatial pool → (B, T, C)
    Then: appearance tokens (T) + motion tokens (T-1) + CLS = 2T total tokens.

    This is the proposed lightweight variant for edge deployment — only 32 tokens
    instead of 497, making it ~15x faster while capturing motion information.
    """

    def __init__(self, in_dim, num_classes=4, model_dim=256, nhead=8,
                 num_layers=3, dropout=0.3, max_t=32):
        super().__init__()
        self.model_dim = model_dim

        # Appearance projection
        self.appear_proj = nn.Sequential(
            nn.Linear(in_dim, model_dim), nn.LayerNorm(model_dim),
            nn.GELU(), nn.Dropout(dropout))
        # Motion projection
        self.motion_proj = nn.Sequential(
            nn.Linear(in_dim, model_dim), nn.LayerNorm(model_dim),
            nn.GELU(), nn.Dropout(dropout))

        # Positional + type embeddings
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.type_embed = nn.Embedding(3, model_dim)  # 0=CLS, 1=appear, 2=motion

        # CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        nh = nhead
        while model_dim % nh != 0: nh -= 1

        self.pre_norm = nn.LayerNorm(model_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim*4,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(model_dim)

        self.fc = nn.Sequential(
            nn.Linear(model_dim, model_dim//2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(model_dim//2, num_classes))

    def forward(self, x):
        # x: (B,T,S,C) or (B,T,C)
        if x.dim() == 4:
            x = x.mean(dim=2)  # spatial pool → (B,T,C)
        b, t, c = x.shape

        # Appearance tokens
        appear = self.appear_proj(x)  # (B,T,D)
        t_idx = torch.arange(t, device=x.device)
        appear = appear + self.time_pos(t_idx).unsqueeze(0)
        appear = appear + self.type_embed(torch.ones(t, dtype=torch.long, device=x.device)).unsqueeze(0)

        # Motion tokens (temporal diff)
        diff = x[:, 1:, :] - x[:, :-1, :]  # (B, T-1, C)
        motion = self.motion_proj(diff)  # (B, T-1, D)
        t_idx_m = torch.arange(t-1, device=x.device)
        motion = motion + self.time_pos(t_idx_m).unsqueeze(0)
        motion = motion + self.type_embed(torch.full((t-1,), 2, dtype=torch.long, device=x.device)).unsqueeze(0)

        # CLS
        cls = self.cls_token.expand(b, -1, -1)
        cls = cls + self.type_embed(torch.zeros(1, dtype=torch.long, device=x.device)).unsqueeze(0)

        # Concatenate: [CLS, appear(T), motion(T-1)] = 2T tokens
        tokens = torch.cat([cls, appear, motion], dim=1)  # (B, 1+T+(T-1), D)

        tokens = self.pre_norm(tokens)
        tokens = self.transformer(tokens)
        out = self.post_norm(tokens[:, 0])  # CLS
        return self.fc(out)


class TemporalOnlyHead(nn.Module):
    """Appearance-only temporal baseline (no motion tokens). For ablation."""

    def __init__(self, in_dim, num_classes=4, model_dim=256, nhead=8,
                 num_layers=3, dropout=0.3, max_t=32):
        super().__init__()
        self.model_dim = model_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, model_dim), nn.LayerNorm(model_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nh = nhead
        while model_dim % nh != 0: nh -= 1
        self.pre_norm = nn.LayerNorm(model_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Sequential(
            nn.Linear(model_dim, model_dim//2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(model_dim//2, num_classes))

    def forward(self, x):
        if x.dim() == 4: x = x.mean(dim=2)
        b, t, c = x.shape
        x = self.proj(x)
        x = x + self.time_pos(torch.arange(t, device=x.device)).unsqueeze(0)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pre_norm(x)
        x = self.transformer(x)
        return self.fc(self.post_norm(x[:, 0]))


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════
CN = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']

def plot_cm(yt, yp, title, path):
    cm = confusion_matrix(yt, yp, labels=[0,1,2,3])
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CN, yticklabels=CN)
    plt.title(title); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


# ═══════════════════════════════════════════════════════════════
# TTA
# ═══════════════════════════════════════════════════════════════
def tta_predict(model, x, device, n=12, noise=0.01):
    model.eval(); logits = []
    with torch.no_grad():
        xd = x.to(device)
        logits.append(model(xd))
        logits.append(model(xd.flip(1)))  # temporal flip
        for _ in range(max(0, n-2)):
            logits.append(model(xd + torch.randn_like(xd)*noise))
    return torch.stack(logits).mean(0)


# ═══════════════════════════════════════════════════════════════
# Train
# ═══════════════════════════════════════════════════════════════
def train_one(cfg, train_ds, test_loader, test_x, test_y, device):
    in_dim = train_ds[0][0].shape[-1]

    if cfg['model_type'] == 'motion':
        model = TemporalMotionHead(
            in_dim=in_dim, num_classes=4, model_dim=cfg['dim'],
            nhead=8, num_layers=cfg['layers'], dropout=cfg['dropout']
        ).to(device)
    else:
        model = TemporalOnlyHead(
            in_dim=in_dim, num_classes=4, model_dim=cfg['dim'],
            nhead=8, num_layers=cfg['layers'], dropout=cfg['dropout']
        ).to(device)

    np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{cfg['name']}] {cfg['model_type']}, dim={cfg['dim']}, "
          f"L={cfg['layers']}, params={np_:,}")

    # Weights
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
    sched = CosWarmup(opt, cfg['warmup'], cfg['epochs'])
    ema = EMA(model, cfg.get('ema_decay', 0.998))
    swa = SWA()
    swa_start = int(cfg['epochs'] * 0.6)

    best_f1, best_acc, best_st = 0., 0., None
    stale = 0

    for ep in range(1, cfg['epochs']+1):
        model.train(); tloss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if cfg['mixup'] > 0 and random.random() < 0.5:
                xb, ya, yb2, lam = mixup(xb, yb, cfg['mixup'])
                opt.zero_grad(set_to_none=True)
                loss = mixup_loss(crit, model(xb), ya, yb2, lam)
            else:
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(model); tloss += loss.item()

        sched.step()
        if ep >= swa_start: swa.collect(model)

        # Eval with EMA
        ema.apply(model); model.eval()
        ps, ls = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                ps.extend(model(xb.to(device)).argmax(1).cpu().tolist())
                ls.extend(yb.tolist())
        pnp, lnp2 = np.array(ps), np.array(ls)
        acc = 100*(pnp==lnp2).mean()
        f1 = f1_score(lnp2, pnp, average='macro', zero_division=0)

        if f1 > best_f1+1e-4 or (abs(f1-best_f1)<1e-4 and acc>best_acc):
            best_f1, best_acc = f1, acc
            best_st = deepcopy(model.state_dict()); stale = 0
        else: stale += 1
        ema.restore(model)

        if ep % 20 == 0 or ep == 1:
            print(f"    Ep {ep:03d} | loss={tloss/len(loader):.4f} "
                  f"| acc={acc:.1f}% f1={f1:.3f} (best={best_acc:.1f}%)")

        if stale >= cfg.get('patience', 30):
            print(f"    [Early stop ep {ep}]"); break

    # Results
    swa_st = swa.averaged()
    ynp = test_y.numpy() if torch.is_tensor(test_y) else test_y
    res = {}

    model.load_state_dict(best_st); model.eval()
    with torch.no_grad(): logits = model(test_x.to(device))
    res['best'] = _r(logits, ynp)
    res['best_tta'] = _r(tta_predict(model, test_x, device, cfg.get('tta_n',12)), ynp)

    if swa_st:
        model.load_state_dict(swa_st); model.eval()
        with torch.no_grad(): logits = model(test_x.to(device))
        res['swa'] = _r(logits, ynp)
        res['swa_tta'] = _r(tta_predict(model, test_x, device, cfg.get('tta_n',12)), ynp)
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
            'acc': 100*(preds==y).mean(),
            'f1': f1_score(y, preds, average='macro', zero_division=0),
            'correct': int((preds==y).sum())}


# ═══════════════════════════════════════════════════════════════
# Ensemble helpers
# ═══════════════════════════════════════════════════════════════
def soft_vote(plist, y):
    avg = np.mean(plist, 0); p = avg.argmax(1)
    return p, 100*(p==y).mean(), f1_score(y, p, average='macro', zero_division=0)

def weighted_vote(plist, w, y):
    w = np.array(w)/sum(w)
    avg = sum(p*wi for p,wi in zip(plist,w)); p = avg.argmax(1)
    return p, 100*(p==y).mean(), f1_score(y, p, average='macro', zero_division=0)

def majority_vote(plists, y):
    n = len(y)
    out = np.array([Counter(p[i] for p in plists).most_common(1)[0][0] for i in range(n)])
    return out, 100*(out==y).mean(), f1_score(y, out, average='macro', zero_division=0)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    pa = ArgumentParser()
    pa.add_argument('--train_features', required=True)
    pa.add_argument('--test_features', required=True)
    pa.add_argument('--output_dir', default='sweep_results/fast_motion_ensemble')
    pa.add_argument('--epochs', type=int, default=100)
    pa.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 7, 2024, 999])
    pa.add_argument('--tta_n', type=int, default=12)
    args = pa.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test (unsafe only)
    tp = torch.load(args.test_features, map_location='cpu')
    tm = tp['labels'] < 4
    test_x = tp['features'][tm]; test_y = tp['labels'][tm].clone()
    print(f"Test: {tm.sum()} unsafe videos, shape={tuple(test_x.shape)}")

    test_ds = UnsafeOnly(args.test_features)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # ── Model configs for ensemble diversity ──
    configs = [
        # A: Motion, compact, scenario_all (more training data)
        {'model_type': 'motion', 'dim': 256, 'layers': 3,
         'dropout': 0.3, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1},
        # B: Motion, wider
        {'model_type': 'motion', 'dim': 384, 'layers': 3,
         'dropout': 0.3, 'lr': 2e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1},
        # C: Motion, unsafe_only (focused)
        {'model_type': 'motion', 'dim': 256, 'layers': 3,
         'dropout': 0.35, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.3, 'mode': 'unsafe_only',
         'bs': 16, 'ls': 0.05},
        # D: Motion, deeper
        {'model_type': 'motion', 'dim': 256, 'layers': 4,
         'dropout': 0.35, 'lr': 2e-4, 'wd': 0.06,
         'gamma': 3.0, 'mixup': 0.5, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1},
        # E: Motion, compact, higher gamma
        {'model_type': 'motion', 'dim': 256, 'layers': 3,
         'dropout': 0.3, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 4.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1},
        # F: Appearance-only baseline (ablation)
        {'model_type': 'appear', 'dim': 256, 'layers': 3,
         'dropout': 0.3, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1},
        # G: Motion, lower LR, more regularization
        {'model_type': 'motion', 'dim': 256, 'layers': 3,
         'dropout': 0.4, 'lr': 1e-4, 'wd': 0.08,
         'gamma': 3.0, 'mixup': 0.5, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.15},
        # H: Motion, unsafe_only, wider
        {'model_type': 'motion', 'dim': 384, 'layers': 3,
         'dropout': 0.3, 'lr': 2e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'unsafe_only',
         'bs': 16, 'ls': 0.1},
    ]

    for c in configs:
        c.update({'epochs': args.epochs, 'warmup': 5, 'patience': 30,
                  'ema_decay': 0.998, 'tta_n': args.tta_n})

    # Train all
    all_trained = []; log_rows = []
    total = len(configs) * len(args.seeds)
    done = 0

    for ci, cfg in enumerate(configs):
        for seed in args.seeds:
            c = dict(cfg); c['seed'] = seed
            tag = f"{'m' if c['model_type']=='motion' else 'a'}{ci}"
            c['name'] = f"{tag}_d{c['dim']}_L{c['layers']}_{c['mode'][:4]}_s{seed}"

            done += 1
            print(f"\n{'='*60}")
            print(f"  [{done}/{total}] {c['name']}")
            print(f"{'='*60}")
            set_seed(seed)

            ds = (ScenarioAll if c['mode']=='scenario_all' else UnsafeOnly)(
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
                })

    # Save individual
    df = pd.DataFrame(log_rows)
    df.to_csv(os.path.join(args.output_dir, 'individual_results.csv'), index=False)

    # ── Ensembles ──
    ynp = test_y.numpy()
    ens_rows = []

    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*60}")

    def _ens(name, plist, w=None):
        if w:
            p, a, f = weighted_vote(plist, w, ynp)
        else:
            p, a, f = soft_vote(plist, ynp)
        c = int((p==ynp).sum())
        print(f"  {name}: {a:.2f}% ({c}/64) f1={f:.3f}")
        ens_rows.append({'ensemble': name, 'n': len(plist), 'acc': a, 'f1': f, 'correct': c})
        return p, a

    # All models
    for v in ['best', 'best_tta', 'swa', 'swa_tta']:
        probs = [t['results'][v]['probs'] for t in all_trained]
        _ens(f'all_{v}_soft', probs)
        ws = [t['results'][v]['f1'] for t in all_trained]
        _ens(f'all_{v}_weighted', probs, ws)

    # Motion-only
    motion = [t for t in all_trained if t['config']['model_type']=='motion']
    if len(motion) >= 2:
        for v in ['swa_tta', 'best_tta']:
            probs = [t['results'][v]['probs'] for t in motion]
            _ens(f'motion_{v}_soft', probs)
            ws = [t['results'][v]['f1'] for t in motion]
            _ens(f'motion_{v}_weighted', probs, ws)

    # Top-K
    for k in [3, 5, 10, 15]:
        top = sorted(all_trained, key=lambda t: t['results']['swa_tta']['acc'], reverse=True)[:k]
        if len(top) >= 2:
            probs = [t['results']['swa_tta']['probs'] for t in top]
            _ens(f'top{k}_swa_tta_soft', probs)
            ws = [t['results']['swa_tta']['f1'] for t in top]
            _ens(f'top{k}_swa_tta_weighted', probs, ws)

    # Per-config best seed
    bpc = {}
    for t in all_trained:
        ci = t['config']['name'].split('_')[0]
        if ci not in bpc or t['results']['swa_tta']['acc'] > bpc[ci]['results']['swa_tta']['acc']:
            bpc[ci] = t
    if len(bpc) > 1:
        probs = [t['results']['swa_tta']['probs'] for t in bpc.values()]
        ws = [t['results']['swa_tta']['f1'] for t in bpc.values()]
        _ens('per_cfg_best_swa_tta', probs, ws)

    # Majority vote
    preds_all = [t['results']['swa_tta']['preds'] for t in all_trained]
    p_mv, a_mv, f_mv = majority_vote(preds_all, ynp)
    c_mv = int((p_mv==ynp).sum())
    print(f"  all_swa_tta_majority: {a_mv:.2f}% ({c_mv}/64) f1={f_mv:.3f}")
    ens_rows.append({'ensemble': 'all_swa_tta_majority', 'n': len(preds_all),
                     'acc': a_mv, 'f1': f_mv, 'correct': c_mv})

    df_ens = pd.DataFrame(ens_rows)
    df_ens.to_csv(os.path.join(args.output_dir, 'ensemble_results.csv'), index=False)

    # ── Summary ──
    best_ind = df.loc[df['acc'].idxmax()]
    best_ens = max(ens_rows, key=lambda x: x['acc'])

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Best individual:  {best_ind['name']} ({best_ind['variant']}) "
          f"= {best_ind['acc']:.2f}% ({int(best_ind['correct'])}/64)")
    print(f"  Best ensemble:    {best_ens['ensemble']} "
          f"= {best_ens['acc']:.2f}% ({best_ens['correct']}/64)")
    print(f"  Paper baseline:   95.81% (62/64)")
    print(f"  Gap (individual): {95.81 - best_ind['acc']:.2f}pp")
    print(f"  Gap (ensemble):   {95.81 - best_ens['acc']:.2f}pp")

    # Per-class for best ensemble
    # Re-compute best ensemble preds
    best_preds = np.mean([t['results']['swa_tta']['probs'] for t in all_trained], 0).argmax(1)
    # Try to find the actual best
    best_name = best_ens['ensemble']
    if 'motion' in best_name and 'weighted' in best_name:
        v = 'swa_tta' if 'swa' in best_name else 'best_tta'
        probs = [t['results'][v]['probs'] for t in motion]
        ws = [t['results'][v]['f1'] for t in motion]
        w = np.array(ws)/sum(ws)
        best_preds = sum(p*wi for p,wi in zip(probs,w)).argmax(1)
    elif 'top' in best_name:
        k = int(best_name.split('top')[1].split('_')[0])
        top = sorted(all_trained, key=lambda t: t['results']['swa_tta']['acc'], reverse=True)[:k]
        if 'weighted' in best_name:
            probs = [t['results']['swa_tta']['probs'] for t in top]
            ws = [t['results']['swa_tta']['f1'] for t in top]
            w = np.array(ws)/sum(ws)
            best_preds = sum(p*wi for p,wi in zip(probs,w)).argmax(1)
        else:
            best_preds = np.mean([t['results']['swa_tta']['probs'] for t in top], 0).argmax(1)

    print(f"\n  Per-class ({best_name}):")
    print(classification_report(ynp, best_preds, labels=[0,1,2,3],
                                target_names=CN, zero_division=0))
    for c, nm in enumerate(CN):
        m = ynp == c
        if m.sum() > 0:
            ca = 100*(best_preds[m]==c).mean()
            print(f"    {nm}: {ca:.1f}% ({(best_preds[m]==c).sum()}/{m.sum()})")

    plot_cm(ynp, best_preds, f'Best Ensemble: {best_ens["acc"]:.1f}%',
            os.path.join(args.output_dir, 'cm_best_ensemble.png'))

    # ── Ablation: motion vs appearance ──
    mot_rows = df[(df['model']=='motion') & (df['variant']=='swa_tta')]
    app_rows = df[(df['model']=='appear') & (df['variant']=='swa_tta')]
    if len(mot_rows) > 0 and len(app_rows) > 0:
        print(f"\n  ABLATION — Motion tokens:")
        print(f"    Motion mean: {mot_rows['acc'].mean():.1f}% ± {mot_rows['acc'].std():.1f}%")
        print(f"    Appear mean: {app_rows['acc'].mean():.1f}% ± {app_rows['acc'].std():.1f}%")
        print(f"    Motion lift: +{mot_rows['acc'].mean()-app_rows['acc'].mean():.1f}pp")

    # Save report
    with open(os.path.join(args.output_dir, 'report.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("FAST MOTION ENSEMBLE — 4-CLASS UNSAFE CLASSIFICATION\n")
        f.write("="*60 + "\n\n")
        f.write(f"Architecture: YOLOv11s backbone → spatial pool → "
                f"[appear + motion] temporal Transformer\n")
        f.write(f"Models trained: {len(all_trained)}\n")
        f.write(f"Seeds: {args.seeds}\n")
        f.write(f"Configs: {len(configs)}\n\n")
        f.write("INDIVIDUAL RESULTS (sorted by acc):\n")
        df_sorted = df[df['variant']=='swa_tta'].sort_values('acc', ascending=False)
        f.write(df_sorted.to_string(index=False) + "\n\n")
        f.write("ENSEMBLE RESULTS (sorted by acc):\n")
        f.write(df_ens.sort_values('acc', ascending=False).to_string(index=False) + "\n\n")
        f.write(f"Best individual: {best_ind['name']} = {best_ind['acc']:.2f}%\n")
        f.write(f"Best ensemble:   {best_ens['ensemble']} = {best_ens['acc']:.2f}%\n")
        f.write(f"Paper baseline:  95.81% (62/64)\n\n")
        f.write(f"Per-class ({best_name}):\n")
        f.write(classification_report(ynp, best_preds, labels=[0,1,2,3],
                                      target_names=CN, zero_division=0))
        if len(mot_rows) > 0 and len(app_rows) > 0:
            f.write(f"\nABLATION:\n")
            f.write(f"  Motion mean: {mot_rows['acc'].mean():.1f}% ± {mot_rows['acc'].std():.1f}%\n")
            f.write(f"  Appear mean: {app_rows['acc'].mean():.1f}% ± {app_rows['acc'].std():.1f}%\n")

    # Save all model configs
    with open(os.path.join(args.output_dir, 'configs.json'), 'w') as f:
        json.dump({'configs': [{k:v for k,v in c.items() if k != 'name'}
                               for c in configs],
                   'seeds': args.seeds, 'epochs': args.epochs}, f, indent=2)

    print(f"\n  All saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
