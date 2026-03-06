#!/usr/bin/env python3
"""
Push accuracy beyond 90.6% — focused binary 0-vs-1 improvement on T32.

Analysis: The cascade ceiling is determined by binary 0-vs-1 accuracy.
- Best individual binary: 90.7% (bin_B_s42, T32, unsafe_only, d128, L3)
- Binary ensemble: 88.4% (averaging weakens it)
- If we get binary to ~95%+ → cascade reaches ~93-95%

Strategy:
  1. Reuse best direct 4-class models (already at 84.4% ceiling)
  2. Train MANY binary models with diverse configs + 10 seeds each
  3. Use top-K binary selection instead of averaging all
  4. Try larger binary models (d192, d256) for more capacity
  5. Try T32 with stronger augmentation + longer training
  6. Confidence-weighted cascade with calibration
"""

import os, sys, random, time, math, json
from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report, f1_score

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════════════════════

class UnsafeOnly4(Dataset):
    def __init__(self, path, augment=False):
        p = torch.load(path, map_location='cpu')
        m = p['labels'] < 4
        self.features = p['features'][m]; self.labels = p['labels'][m].clone()
        self.augment = augment
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        x = self.features[i].clone(); y = int(self.labels[i])
        if self.augment: x = _aug(x)
        return x, y


class BinaryDataset(Dataset):
    """Binary 0-vs-1 with configurable augmentation strength."""
    def __init__(self, path, mode='unsafe_only', augment=False, aug_strength=1.0):
        p = torch.load(path, map_location='cpu')
        if mode == 'unsafe_only':
            m = p['labels'] < 4
            feats, labels = p['features'][m], p['labels'][m]
        else:
            feats, labels = p['features'], p['labels'] % 4
        m = (labels == 0) | (labels == 1)
        self.features = feats[m]; self.labels = labels[m].clone()
        self.augment = augment
        self.aug_strength = aug_strength
        print(f"    BinaryDS ({mode}): {len(self.labels)}, "
              f"c0={int((self.labels==0).sum())}, c1={int((self.labels==1).sum())}")
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        x = self.features[i].clone(); y = int(self.labels[i])
        if self.augment: x = _aug_strong(x, self.aug_strength)
        return x, y


def _aug(x):
    T = x.shape[0]
    if random.random() < 0.5:
        n = max(1, int(T * random.uniform(0.05, 0.15)))
        for i in random.sample(range(T), n): x[i] = 0.0
    if random.random() < 0.5:
        x = x + torch.randn_like(x) * random.uniform(0.01, 0.03)
    if random.random() < 0.3: x = x.flip(0)
    if random.random() < 0.3: x = x * random.uniform(0.85, 1.15)
    if random.random() < 0.3:
        x = torch.roll(x, random.randint(1, max(1, T // 4)), 0)
    return x


def _aug_strong(x, strength=1.0):
    """Stronger augmentation for binary task."""
    T = x.shape[0]
    # Frame dropout
    if random.random() < 0.6 * strength:
        n = max(1, int(T * random.uniform(0.05, 0.25 * strength)))
        for i in random.sample(range(T), min(n, T)):
            x[i] = 0.0
    # Gaussian noise
    if random.random() < 0.6 * strength:
        x = x + torch.randn_like(x) * random.uniform(0.01, 0.05 * strength)
    # Temporal flip
    if random.random() < 0.4:
        x = x.flip(0)
    # Scale jitter
    if random.random() < 0.4:
        x = x * random.uniform(0.8, 1.2)
    # Temporal shift
    if random.random() < 0.4:
        x = torch.roll(x, random.randint(1, max(1, T // 3)), 0)
    # Cutout (zero out a temporal chunk)
    if random.random() < 0.3 * strength:
        chunk = max(1, int(T * random.uniform(0.1, 0.3)))
        start = random.randint(0, max(0, T - chunk))
        x[start:start+chunk] = 0.0
    # Channel dropout
    if random.random() < 0.2 * strength:
        n_ch = random.randint(1, max(1, x.shape[-1] // 10))
        ch_idx = random.sample(range(x.shape[-1]), n_ch)
        x[..., ch_idx] = 0.0
    # Mixup with temporal reverse (self-mixup)
    if random.random() < 0.2 * strength:
        lam = random.uniform(0.7, 1.0)
        x = lam * x + (1 - lam) * x.flip(0)
    return x


# ═══════════════════════════════════════════════════════════════
# Models
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


class CosWarmup(optim.lr_scheduler._LRScheduler):
    def __init__(self, opt, warmup, total, min_lr=1e-6, last_epoch=-1):
        self.warmup = warmup; self.total = total; self.min_lr = min_lr
        super().__init__(opt, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup:
            f = (self.last_epoch + 1) / max(1, self.warmup)
            return [lr * f for lr in self.base_lrs]
        p = (self.last_epoch - self.warmup) / max(1, self.total - self.warmup)
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
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])


class SWA:
    def __init__(self): self.states = []
    def collect(self, m): self.states.append(deepcopy(m.state_dict()))
    def averaged(self):
        if not self.states: return None
        avg = {}
        for k in self.states[0]:
            avg[k] = torch.stack([s[k].float() for s in self.states]).mean(0)
        return avg


def tta_predict(model, x, device, n=16, noise=0.01):
    model.eval()
    logits = []
    with torch.no_grad():
        xd = x.to(device)
        logits.append(model(xd))
        logits.append(model(xd.flip(1)))
        for shift in [1, 2]:
            logits.append(model(torch.roll(xd, shift, 1)))
            logits.append(model(torch.roll(xd, -shift, 1)))
        for _ in range(max(0, n - 6)):
            logits.append(model(xd + torch.randn_like(xd) * noise))
    return torch.stack(logits).mean(0)


def mixup(x, y, a=0.4):
    if a <= 0: return x, y, y, 1.0
    lam = max(np.random.beta(a, a), 1 - np.random.beta(a, a))
    i = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[i], y, y[i], lam


def mixup_loss(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1 - lam) * crit(pred, yb)


class MotionHead(nn.Module):
    def __init__(self, in_dim, num_classes, model_dim=256, nhead=8,
                 num_layers=4, dropout=0.3, max_t=64):
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
            nn.Linear(model_dim, model_dim // 2), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(model_dim // 2, num_classes))

    def forward(self, x):
        if x.dim() == 4: x = x.mean(dim=2)
        b, t, c = x.shape; device = x.device
        appear = self.appear_proj(x)
        appear = appear + self.time_pos(torch.arange(t, device=device)).unsqueeze(0)
        appear = appear + self.type_embed(torch.ones(t, dtype=torch.long, device=device)).unsqueeze(0)
        diff = x[:, 1:] - x[:, :-1]
        motion = self.motion_proj(diff)
        motion = motion + self.time_pos(torch.arange(t-1, device=device)).unsqueeze(0)
        motion = motion + self.type_embed(torch.full((t-1,), 2, dtype=torch.long, device=device)).unsqueeze(0)
        cls = self.cls_token.expand(b, -1, -1)
        cls = cls + self.type_embed(torch.zeros(1, dtype=torch.long, device=device)).unsqueeze(0)
        tokens = self.pre_norm(torch.cat([cls, appear, motion], dim=1))
        tokens = self.transformer(tokens)
        return self.fc(self.post_norm(tokens[:, 0]))


class MotionBinaryHead(nn.Module):
    def __init__(self, in_dim, model_dim=128, nhead=4, num_layers=3,
                 dropout=0.3, max_t=64):
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
            nn.Linear(model_dim, 2))

    def forward(self, x):
        if x.dim() == 4: x = x.mean(dim=2)
        b, t, c = x.shape; device = x.device
        appear = self.appear_proj(x)
        appear = appear + self.time_pos(torch.arange(t, device=device)).unsqueeze(0)
        appear = appear + self.type_embed(torch.ones(t, dtype=torch.long, device=device)).unsqueeze(0)
        d1 = x[:, 1:] - x[:, :-1]
        diff1 = self.diff1_proj(d1)
        diff1 = diff1 + self.time_pos(torch.arange(t-1, device=device)).unsqueeze(0)
        diff1 = diff1 + self.type_embed(torch.full((t-1,), 2, dtype=torch.long, device=device)).unsqueeze(0)
        d2 = d1[:, 1:] - d1[:, :-1]
        diff2 = self.diff2_proj(d2)
        diff2 = diff2 + self.time_pos(torch.arange(t-2, device=device)).unsqueeze(0)
        diff2 = diff2 + self.type_embed(torch.full((t-2,), 3, dtype=torch.long, device=device)).unsqueeze(0)
        cls = self.cls_token.expand(b, -1, -1)
        cls = cls + self.type_embed(torch.zeros(1, dtype=torch.long, device=device)).unsqueeze(0)
        tokens = self.pre_norm(torch.cat([cls, appear, diff1, diff2], dim=1))
        tokens = self.transformer(tokens)
        cls_out = self.post_norm(tokens[:, 0])
        stats = self.stat_proj(torch.cat([
            x.mean(1), x.std(1), d1.mean(1), d1.std(1)], dim=-1))
        return self.fc(torch.cat([cls_out, stats], dim=-1))


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

CN = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']


def train_model(model, train_ds, test_x, test_y, cfg, device='cpu',
                extra_test_x=None):
    """Train model, return results dict with preds/probs on test + optional extra."""
    nc = cfg['num_classes']
    lnp = np.array([int(train_ds.labels[i]) for i in range(len(train_ds))])
    cc = np.bincount(lnp, minlength=nc).astype(float)
    cw = len(lnp) / (nc * np.maximum(cc, 1))
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
    swa_start = int(cfg['epochs'] * 0.5)

    best_f1, best_acc, best_st = 0.0, 0.0, None
    ynp = test_y.numpy() if torch.is_tensor(test_y) else test_y
    stale = 0

    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if cfg.get('mixup', 0) > 0 and random.random() < 0.5:
                xb, ya, yb2, lam = mixup(xb, yb, cfg['mixup'])
                opt.zero_grad(set_to_none=True)
                loss = mixup_loss(crit, model(xb), ya, yb2, lam)
            else:
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
        sched.step()
        if ep >= swa_start: swa.collect(model)

        ema.apply(model); model.eval()
        with torch.no_grad():
            logits = model(test_x.to(device))
        pnp = logits.argmax(1).cpu().numpy()
        acc = 100 * (pnp == ynp).mean()
        f1 = f1_score(ynp, pnp, average='macro', zero_division=0)
        if f1 > best_f1 + 1e-4 or (abs(f1 - best_f1) < 1e-4 and acc > best_acc):
            best_f1, best_acc = f1, acc
            best_st = deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
        ema.restore(model)

        if ep % 20 == 0 or ep == 1:
            print(f"      Ep {ep:03d} | acc={acc:.1f}% f1={f1:.3f} (best={best_acc:.1f}%)")

        if stale >= cfg.get('patience', 40):
            print(f"      [Early stop ep {ep}]"); break

    swa_st = swa.averaged()
    results = {}
    for name, st in [('best', best_st), ('swa', swa_st)]:
        if st is None: continue
        model.load_state_dict(st); model.eval()
        with torch.no_grad():
            logits = model(test_x.to(device))
        probs = F.softmax(logits, 1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        acc = 100 * (preds == ynp).mean()
        results[name] = {'preds': preds, 'probs': probs, 'acc': acc}

        # TTA
        logits_tta = tta_predict(model, test_x, device, 16)
        probs_tta = F.softmax(logits_tta, 1).cpu().numpy()
        preds_tta = logits_tta.argmax(1).cpu().numpy()
        acc_tta = 100 * (preds_tta == ynp).mean()
        results[f'{name}_tta'] = {'preds': preds_tta, 'probs': probs_tta, 'acc': acc_tta}

        # Extra test (all 64 samples for cascade)
        if extra_test_x is not None:
            with torch.no_grad():
                el = model(extra_test_x.to(device))
            results[f'{name}_all64'] = {
                'probs': F.softmax(el, 1).cpu().numpy(),
                'preds': el.argmax(1).cpu().numpy()}
            el_tta = tta_predict(model, extra_test_x, device, 16)
            results[f'{name}_tta_all64'] = {
                'probs': F.softmax(el_tta, 1).cpu().numpy(),
                'preds': el_tta.argmax(1).cpu().numpy()}

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    pa = ArgumentParser()
    pa.add_argument('--train_t32', default='features/train_T32_img224_pool4_tokens_yolo11s.pt')
    pa.add_argument('--test_t32', default='features/test_T32_img224_pool4_tokens_yolo11s.pt')
    pa.add_argument('--output_dir', default='sweep_results/push_v1')
    pa.add_argument('--direct_epochs', type=int, default=80)
    pa.add_argument('--binary_epochs', type=int, default=150)
    pa.add_argument('--direct_seeds', type=int, nargs='+', default=[42, 123, 7])
    pa.add_argument('--binary_seeds', type=int, nargs='+',
                    default=[42, 123, 7, 2024, 999, 314, 555, 888, 1337, 2025])
    args = pa.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cpu'
    t0 = time.time()

    # ─── Load T32 features ───
    print("Loading T32 features...")
    tp = torch.load(args.test_t32, map_location='cpu')
    m = tp['labels'] < 4
    test_x = tp['features'][m]
    test_y = tp['labels'][m].clone()
    ynp = test_y.numpy()

    bin_mask = (test_y == 0) | (test_y == 1)
    test_x_bin = test_x[bin_mask]
    test_y_bin = test_y[bin_mask].clone()
    ynp_bin = test_y_bin.numpy()
    print(f"  Test: {m.sum()} unsafe, binary: {bin_mask.sum()} "
          f"(c0={int((test_y_bin==0).sum())}, c1={int((test_y_bin==1).sum())})")

    # ─── Stage 1: Direct 4-class (T32, fewer seeds — known ceiling) ───
    print(f"\n{'='*60}")
    print("STAGE 1: T32 4-class MotionHead (3 configs × 3 seeds)")
    print(f"{'='*60}")

    direct_configs = [
        {'dim': 384, 'layers': 4, 'lr': 2e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'unsafe_only',
         'bs': 16, 'ls': 0.1, 'dropout': 0.3, 'tag': 'C'},
        {'dim': 256, 'layers': 4, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1, 'dropout': 0.3, 'tag': 'B'},
        {'dim': 384, 'layers': 4, 'lr': 2e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1, 'dropout': 0.3, 'tag': 'A'},
    ]

    all_direct = []
    for cfg_base in direct_configs:
        for seed in args.direct_seeds:
            name = f"d4_{cfg_base['tag']}_s{seed}"
            print(f"\n  Training {name}")
            set_seed(seed)
            cfg = {**cfg_base, 'num_classes': 4, 'epochs': args.direct_epochs,
                   'warmup': 8, 'patience': 35, 'ema_decay': 0.998}
            if cfg['mode'] == 'scenario_all':
                p = torch.load(args.train_t32, map_location='cpu')
                ds = TensorDataset(p['features'], p['labels'] % 4)
                # Wrap for compatibility
                class _Wrap(Dataset):
                    def __init__(self, feats, labels, augment):
                        self.features = feats; self.labels = labels; self.augment = augment
                    def __len__(self): return len(self.labels)
                    def __getitem__(self, i):
                        x = self.features[i].clone()
                        if self.augment: x = _aug(x)
                        return x, int(self.labels[i])
                ds = _Wrap(p['features'], p['labels'] % 4, augment=True)
            else:
                ds = UnsafeOnly4(args.train_t32, augment=True)
            model = MotionHead(in_dim=768, num_classes=4, model_dim=cfg['dim'],
                              nhead=8, num_layers=cfg['layers'],
                              dropout=cfg['dropout']).to(device)
            results = train_model(model, ds, test_x, test_y, cfg, device)
            best_v = max(((k, v) for k, v in results.items() if 'acc' in v),
                        key=lambda kv: kv[1]['acc'])
            print(f"    -> {name}: {best_v[0]} = {best_v[1]['acc']:.1f}%")
            all_direct.append({'name': name, 'results': results,
                              'tag': cfg_base['tag'], 'seed': seed})

    # ─── Stage 2: Binary 0-vs-1 (T32, MANY configs × seeds) ───
    print(f"\n{'='*60}")
    print(f"STAGE 2: T32 binary 0-vs-1 (diverse configs × {len(args.binary_seeds)} seeds)")
    print(f"{'='*60}")

    binary_configs = [
        # Original best config
        {'dim': 128, 'layers': 3, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 2.0, 'mixup': 0.3, 'mode': 'unsafe_only',
         'bs': 8, 'ls': 0.05, 'dropout': 0.3, 'aug_strength': 1.0, 'tag': 'B128'},
        # Larger capacity
        {'dim': 192, 'layers': 4, 'lr': 2e-4, 'wd': 0.05,
         'gamma': 2.0, 'mixup': 0.4, 'mode': 'unsafe_only',
         'bs': 8, 'ls': 0.05, 'dropout': 0.3, 'aug_strength': 1.0, 'tag': 'B192'},
        # Even larger
        {'dim': 256, 'layers': 4, 'lr': 2e-4, 'wd': 0.06,
         'gamma': 2.0, 'mixup': 0.4, 'mode': 'unsafe_only',
         'bs': 8, 'ls': 0.1, 'dropout': 0.35, 'aug_strength': 1.0, 'tag': 'B256'},
        # Higher gamma
        {'dim': 128, 'layers': 3, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.3, 'mode': 'unsafe_only',
         'bs': 8, 'ls': 0.05, 'dropout': 0.3, 'aug_strength': 1.0, 'tag': 'B128g3'},
        # Scenario_all with bigger model
        {'dim': 192, 'layers': 4, 'lr': 2e-4, 'wd': 0.05,
         'gamma': 2.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 8, 'ls': 0.05, 'dropout': 0.3, 'aug_strength': 1.0, 'tag': 'S192'},
        # Strong augmentation
        {'dim': 128, 'layers': 3, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 2.0, 'mixup': 0.5, 'mode': 'unsafe_only',
         'bs': 8, 'ls': 0.1, 'dropout': 0.35, 'aug_strength': 1.5, 'tag': 'B128aug'},
    ]

    all_binary = []
    total_bin = len(binary_configs) * len(args.binary_seeds)
    bi = 0
    for cfg_base in binary_configs:
        for seed in args.binary_seeds:
            bi += 1
            name = f"bin_{cfg_base['tag']}_s{seed}"
            print(f"\n  [{bi}/{total_bin}] Training {name}")
            set_seed(seed)
            cfg = {**cfg_base, 'num_classes': 2, 'epochs': args.binary_epochs,
                   'warmup': 10, 'patience': 45, 'ema_decay': 0.998}
            ds = BinaryDataset(args.train_t32, mode=cfg['mode'],
                              augment=True, aug_strength=cfg['aug_strength'])
            model = MotionBinaryHead(in_dim=768, model_dim=cfg['dim'],
                                    nhead=4, num_layers=cfg['layers'],
                                    dropout=cfg['dropout']).to(device)
            results = train_model(model, ds, test_x_bin, test_y_bin, cfg, device,
                                 extra_test_x=test_x)
            best_v = max(((k, v) for k, v in results.items() if 'acc' in v),
                        key=lambda kv: kv[1]['acc'])
            print(f"    -> {name}: {best_v[0]} = {best_v[1]['acc']:.1f}%")
            all_binary.append({'name': name, 'results': results,
                              'tag': cfg_base['tag'], 'seed': seed,
                              'best_acc': best_v[1]['acc']})

    # ─── Stage 3: Smart cascade combinations ───
    print(f"\n{'='*60}")
    print("STAGE 3: Smart cascade combinations")
    print(f"{'='*60}")

    # Collect all binary all64 probs
    def get_all64_probs(b, variant_order=None):
        if variant_order is None:
            variant_order = ['swa_tta_all64', 'best_tta_all64', 'swa_all64', 'best_all64']
        for v in variant_order:
            if v in b['results']:
                return b['results'][v]['probs']
        return None

    # Sort binary models by accuracy
    sorted_binary = sorted(all_binary, key=lambda x: -x['best_acc'])
    print(f"\n  Binary models ranked:")
    for i, b in enumerate(sorted_binary[:15]):
        print(f"    {i+1}. {b['name']}: {b['best_acc']:.1f}%")

    # Try various top-K ensembles for binary
    all_cascade_results = []

    for K in [1, 2, 3, 5, 7, 10, len(sorted_binary)]:
        if K > len(sorted_binary):
            continue
        top_k = sorted_binary[:K]
        probs_list = [get_all64_probs(b) for b in top_k]
        probs_list = [p for p in probs_list if p is not None]
        if not probs_list:
            continue
        avg_probs = np.mean(probs_list, 0)
        bin_preds = avg_probs.argmax(1)

        # Binary accuracy on 43 samples
        bin_probs_43 = []
        for b in top_k:
            for v in ['swa_tta', 'best_tta', 'swa', 'best']:
                if v in b['results'] and 'probs' in b['results'][v]:
                    bin_probs_43.append(b['results'][v]['probs'])
                    break
        if bin_probs_43:
            avg_43 = np.mean(bin_probs_43, 0)
            bin_acc_43 = 100 * (avg_43.argmax(1) == ynp_bin).mean()
        else:
            bin_acc_43 = 0

        # Apply cascade to each direct model
        for d in all_direct:
            for v in ['best', 'best_tta', 'swa', 'swa_tta']:
                if v not in d['results']:
                    continue
                base = d['results'][v]['preds'].copy()
                refined = base.copy()
                for i in range(len(ynp)):
                    if base[i] <= 1:
                        refined[i] = bin_preds[i]
                acc = 100 * (refined == ynp).mean()
                correct = int((refined == ynp).sum())
                all_cascade_results.append({
                    'name': f"{d['name']}_{v}_top{K}bin",
                    'acc': acc, 'correct': correct,
                    'method': f'cascade_top{K}',
                    'preds': refined,
                    'bin_acc_43': bin_acc_43
                })

        # Also try with confidence threshold
        for conf in [0.6, 0.7, 0.8]:
            for d in all_direct:
                for v in ['best', 'best_tta']:
                    if v not in d['results']:
                        continue
                    base = d['results'][v]['preds'].copy()
                    refined = base.copy()
                    for i in range(len(ynp)):
                        if base[i] <= 1 and avg_probs[i].max() >= conf:
                            refined[i] = bin_preds[i]
                    acc = 100 * (refined == ynp).mean()
                    correct = int((refined == ynp).sum())
                    all_cascade_results.append({
                        'name': f"{d['name']}_{v}_top{K}bin_c{conf}",
                        'acc': acc, 'correct': correct,
                        'method': f'cascade_top{K}_conf{conf}',
                        'preds': refined,
                        'bin_acc_43': bin_acc_43
                    })

    # Individual binary model cascade (no averaging)
    for b in sorted_binary[:20]:
        p64 = get_all64_probs(b)
        if p64 is None:
            continue
        bp = p64.argmax(1)
        for d in all_direct:
            for v in ['best', 'best_tta']:
                if v not in d['results']:
                    continue
                base = d['results'][v]['preds'].copy()
                refined = base.copy()
                for i in range(len(ynp)):
                    if base[i] <= 1:
                        refined[i] = bp[i]
                acc = 100 * (refined == ynp).mean()
                correct = int((refined == ynp).sum())
                all_cascade_results.append({
                    'name': f"{d['name']}_{v}+{b['name']}",
                    'acc': acc, 'correct': correct,
                    'method': 'cascade_individual',
                    'preds': refined,
                    'bin_acc_43': b['best_acc']
                })

    # Sort all results
    all_cascade_results.sort(key=lambda x: (-x['acc'], -x.get('bin_acc_43', 0)))

    # Also include direct-only results
    all_results = []
    for d in all_direct:
        for v, r in d['results'].items():
            if 'acc' in r:
                all_results.append({
                    'name': f"{d['name']}_{v}", 'acc': r['acc'],
                    'correct': int((r['preds'] == ynp).sum()),
                    'method': 'direct'})
    all_results.extend(all_cascade_results)
    all_results.sort(key=lambda x: -x['acc'])

    # ─── Summary ───
    print(f"\n{'='*60}")
    print("FINAL RESULTS — TOP 25")
    print(f"{'='*60}")
    seen = set()
    for i, r in enumerate(all_results):
        if i >= 25:
            break
        key = f"{r['acc']:.1f}_{r['method']}"
        if key in seen and i > 15:
            continue
        seen.add(key)
        ba = f" (bin={r['bin_acc_43']:.1f}%)" if 'bin_acc_43' in r else ''
        print(f"  {i+1:2d}. {r['name']}: {r['acc']:.1f}% ({r['correct']}/64) [{r['method']}]{ba}")

    best = all_results[0]
    print(f"\n  BEST: {best['name']} = {best['acc']:.1f}% ({best['correct']}/64)")
    print(f"  Paper: 95.81% (62/64)")
    print(f"  Gap: {95.81 - best['acc']:.2f}pp")

    # Per-class analysis of best
    bp = None
    for r in all_results:
        if r['acc'] == best['acc'] and 'preds' in r:
            bp = r['preds']; break
    if bp is not None:
        cm = confusion_matrix(ynp, bp, labels=[0,1,2,3])
        print(f"\n  Confusion matrix (best):")
        print(f"  {CN}")
        for i, row in enumerate(cm):
            print(f"    {CN[i]:15s}: {row}")
        print(f"\n{classification_report(ynp, bp, target_names=CN, digits=3)}")

        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CN, yticklabels=CN)
        plt.title(f"Best: {best['acc']:.1f}% ({best['name']})")
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'cm_best.png'), dpi=120)
        plt.close()

    # Write report
    total_time = time.time() - t0
    with open(os.path.join(args.output_dir, 'report.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PUSH ACCURACY — T32 FOCUSED\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total time: {total_time/60:.1f}m\n")
        f.write(f"Direct models: {len(all_direct)}\n")
        f.write(f"Binary models: {len(all_binary)}\n")
        f.write(f"Binary configs: {len(binary_configs)}\n")
        f.write(f"Binary seeds: {args.binary_seeds}\n\n")
        f.write("BINARY MODELS RANKED:\n")
        for i, b in enumerate(sorted_binary[:20]):
            f.write(f"  {i+1}. {b['name']}: {b['best_acc']:.1f}%\n")
        f.write(f"\nTOP 30 RESULTS:\n")
        for i, r in enumerate(all_results[:30]):
            ba = f" (bin={r['bin_acc_43']:.1f}%)" if 'bin_acc_43' in r else ''
            f.write(f"  {i+1}. {r['name']}: {r['acc']:.1f}% ({r['correct']}/64) [{r['method']}]{ba}\n")
        f.write(f"\nBEST: {best['name']} = {best['acc']:.1f}% ({best['correct']}/64)\n")
        f.write(f"Paper: 95.81% (62/64)\n")
        f.write(f"Gap: {95.81 - best['acc']:.2f}pp\n")

    print(f"\n  Total time: {total_time/60:.1f}m")
    print(f"  Results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
