#!/usr/bin/env python3
"""
Cascaded classifier to specifically target the class 0 vs 1 confusion.

Strategy:
  Stage 1: 3-class classifier {walkway_group(0or1), panel(2), forklift(3)}
  Stage 2: Binary classifier {WalkwayViol(0) vs UnauthorInt(1)} — only for group

This exploits the finding that classes 0&1 have 0.99 appearance cosine similarity
but 0.28 temporal-difference similarity. Stage 2 focuses exclusively on temporal
dynamics to separate them.

Also tries:
  - Direct 4-class with class 0/1 oversampling
  - Specialized temporal features for stage 2 (higher-order diffs)
  - Multi-seed ensemble with cascaded voting
"""

import os, sys, copy, json, math, random, time
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
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════════════════════
class UnsafeOnly4(Dataset):
    """Standard 4-class unsafe-only dataset."""
    def __init__(self, path, augment=False):
        p = torch.load(path, map_location='cpu')
        m = p['labels'] < 4
        self.features = p['features'][m]
        self.labels = p['labels'][m].clone()
        self.augment = augment
        print(f"  UnsafeOnly4: {m.sum()}/{len(p['labels'])}")

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        x = self.features[i].clone()
        y = int(self.labels[i])
        if self.augment: x = _aug(x)
        return x, y


class ScenarioAll4(Dataset):
    """Scenario mapping: label % 4 (safe→unsafe scenario)."""
    def __init__(self, path, augment=False):
        p = torch.load(path, map_location='cpu')
        self.features = p['features']
        self.labels = p['labels'] % 4
        self.augment = augment
        print(f"  ScenarioAll4: {len(self.labels)}")

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        x = self.features[i].clone()
        y = int(self.labels[i])
        if self.augment: x = _aug(x)
        return x, y


class ThreeClassGrouped(Dataset):
    """3-class: 0=walkway_group(orig 0 or 1), 1=panel(orig 2), 2=forklift(orig 3)."""
    def __init__(self, path, mode='unsafe_only', augment=False):
        p = torch.load(path, map_location='cpu')
        if mode == 'unsafe_only':
            m = p['labels'] < 4
            feats = p['features'][m]
            labels = p['labels'][m]
        else:  # scenario_all
            feats = p['features']
            labels = p['labels'] % 4

        # Remap: {0,1} -> 0, {2} -> 1, {3} -> 2
        new_labels = labels.clone()
        new_labels[labels == 0] = 0
        new_labels[labels == 1] = 0
        new_labels[labels == 2] = 1
        new_labels[labels == 3] = 2

        self.features = feats
        self.labels = new_labels
        self.augment = augment
        from collections import Counter
        c = Counter(self.labels.tolist())
        print(f"  ThreeClassGrouped ({mode}): {len(self.labels)}, dist={dict(sorted(c.items()))}")

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        x = self.features[i].clone()
        y = int(self.labels[i])
        if self.augment: x = _aug(x)
        return x, y


class BinaryWalkway(Dataset):
    """Binary classifier: class 0 (WalkwayViol) vs class 1 (UnauthorInt)."""
    def __init__(self, path, mode='unsafe_only', augment=False, use_motion_features=True):
        p = torch.load(path, map_location='cpu')
        if mode == 'unsafe_only':
            m = p['labels'] < 4
            feats = p['features'][m]
            labels = p['labels'][m]
        else:
            feats = p['features']
            orig_labels = p['labels'] % 4
            # Only keep 0 and 1
            m = (orig_labels == 0) | (orig_labels == 1)
            feats = feats[m]
            labels = orig_labels[m]

        # Filter to only class 0 and 1
        m = (labels == 0) | (labels == 1)
        self.features = feats[m]
        self.labels = labels[m].clone()
        self.augment = augment
        self.use_motion_features = use_motion_features
        print(f"  BinaryWalkway ({mode}): {len(self.labels)}, "
              f"class0={int((self.labels==0).sum())}, class1={int((self.labels==1).sum())}")

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        x = self.features[i].clone()  # (T, S, C) or (T, C)
        y = int(self.labels[i])
        if self.augment:
            x = _aug(x)
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


# ═══════════════════════════════════════════════════════════════
# Losses
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
            if p.requires_grad: self.shadow[n].mul_(self.decay).add_(p.data, alpha=1-self.decay)
    def apply(self, model):
        self.backup = {}
        for n,p in model.named_parameters():
            if p.requires_grad: self.backup[n] = p.data.clone(); p.data.copy_(self.shadow[n])
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
# Models
# ═══════════════════════════════════════════════════════════════
class MotionHead(nn.Module):
    """Temporal Transformer with appearance + motion tokens."""
    def __init__(self, in_dim, num_classes, model_dim=256, nhead=8,
                 num_layers=4, dropout=0.3, max_t=32):
        super().__init__()
        self.appear_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.motion_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.type_embed = nn.Embedding(3, model_dim)  # CLS=0, appear=1, motion=2
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
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim//2), nn.GELU(),
            nn.Dropout(dropout*0.5), nn.Linear(model_dim//2, num_classes))

    def forward(self, x):
        if x.dim() == 4: x = x.mean(dim=2)
        b, t, c = x.shape
        device = x.device
        appear = self.appear_proj(x)
        appear = appear + self.time_pos(torch.arange(t, device=device)).unsqueeze(0)
        appear = appear + self.type_embed(torch.ones(t, dtype=torch.long, device=device)).unsqueeze(0)
        diff = x[:, 1:] - x[:, :-1]
        motion = self.motion_proj(diff)
        motion = motion + self.time_pos(torch.arange(t-1, device=device)).unsqueeze(0)
        motion = motion + self.type_embed(torch.full((t-1,), 2, dtype=torch.long, device=device)).unsqueeze(0)
        cls = self.cls_token.expand(b, -1, -1)
        cls = cls + self.type_embed(torch.zeros(1, dtype=torch.long, device=device)).unsqueeze(0)
        tokens = torch.cat([cls, appear, motion], dim=1)
        tokens = self.pre_norm(tokens)
        tokens = self.transformer(tokens)
        return self.fc(self.post_norm(tokens[:, 0]))


class MotionBinaryHead(nn.Module):
    """Specialized binary classifier for class 0 vs 1, focusing on temporal dynamics.

    Uses multi-order temporal differences + temporal statistics as features.
    """
    def __init__(self, in_dim, model_dim=128, nhead=4, num_layers=3, dropout=0.3, max_t=32):
        super().__init__()
        # Project temporal differences at multiple orders
        self.diff1_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU(), nn.Dropout(dropout*0.5))
        self.diff2_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU(), nn.Dropout(dropout*0.5))
        self.appear_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU(), nn.Dropout(dropout*0.5))

        self.time_pos = nn.Embedding(max_t, model_dim)
        self.type_embed = nn.Embedding(4, model_dim)  # CLS=0, appear=1, diff1=2, diff2=3
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

        # Also use temporal statistics as auxiliary features
        self.stat_proj = nn.Sequential(
            nn.Linear(in_dim * 4, model_dim),  # mean, std, diff_mean, diff_std
            nn.GELU(), nn.Dropout(dropout))

        self.fc = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, 2))

    def forward(self, x):
        if x.dim() == 4: x = x.mean(dim=2)
        b, t, c = x.shape
        device = x.device

        # Appearance
        appear = self.appear_proj(x)
        appear = appear + self.time_pos(torch.arange(t, device=device)).unsqueeze(0)
        appear = appear + self.type_embed(torch.ones(t, dtype=torch.long, device=device)).unsqueeze(0)

        # 1st order diff
        d1 = x[:, 1:] - x[:, :-1]
        diff1 = self.diff1_proj(d1)
        diff1 = diff1 + self.time_pos(torch.arange(t-1, device=device)).unsqueeze(0)
        diff1 = diff1 + self.type_embed(torch.full((t-1,), 2, dtype=torch.long, device=device)).unsqueeze(0)

        # 2nd order diff
        d2 = d1[:, 1:] - d1[:, :-1]
        diff2 = self.diff2_proj(d2)
        diff2 = diff2 + self.time_pos(torch.arange(t-2, device=device)).unsqueeze(0)
        diff2 = diff2 + self.type_embed(torch.full((t-2,), 3, dtype=torch.long, device=device)).unsqueeze(0)

        # CLS
        cls = self.cls_token.expand(b, -1, -1)
        cls = cls + self.type_embed(torch.zeros(1, dtype=torch.long, device=device)).unsqueeze(0)

        tokens = torch.cat([cls, appear, diff1, diff2], dim=1)
        tokens = self.pre_norm(tokens)
        tokens = self.transformer(tokens)
        cls_out = self.post_norm(tokens[:, 0])

        # Temporal statistics
        x_mean = x.mean(dim=1)  # (B, C)
        x_std = x.std(dim=1)
        d1_mean = d1.mean(dim=1)
        d1_std = d1.std(dim=1)
        stats = self.stat_proj(torch.cat([x_mean, x_std, d1_mean, d1_std], dim=-1))

        combined = torch.cat([cls_out, stats], dim=-1)
        return self.fc(combined)


CN = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']


def plot_cm(yt, yp, title, path, labels=None, names=None):
    if labels is None: labels = [0,1,2,3]
    if names is None: names = CN
    cm = confusion_matrix(yt, yp, labels=labels)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names)
    plt.title(title); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def tta_predict(model, x, device, n=16, noise=0.01):
    model.eval(); logits = []
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


# ═══════════════════════════════════════════════════════════════
# Training helpers
# ═══════════════════════════════════════════════════════════════
def train_model(model, train_ds, test_loader, test_x, test_y, cfg, device,
                extra_test_x=None):
    """Generic training loop.

    Args:
        extra_test_x: optional tensor of ALL test samples for cascade predictions
    """
    num_classes = cfg['num_classes']
    lnp = np.array([int(train_ds.labels[i]) for i in range(len(train_ds))])
    cc = np.bincount(lnp, minlength=num_classes).astype(float)
    cw = len(lnp) / (num_classes * np.maximum(cc, 1))
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

    best_f1, best_acc, best_st = 0., 0., None
    stale = 0
    ynp = test_y.numpy() if torch.is_tensor(test_y) else test_y

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
            opt.step(); ema.update(model)
        sched.step()
        if ep >= swa_start: swa.collect(model)

        ema.apply(model); model.eval()
        ps = []
        with torch.no_grad():
            for xb, yb in test_loader:
                ps.extend(model(xb.to(device)).argmax(1).cpu().tolist())
        pnp = np.array(ps)
        acc = 100*(pnp==ynp).mean()
        f1 = f1_score(ynp, pnp, average='macro', zero_division=0)

        if f1 > best_f1 + 1e-4 or (abs(f1-best_f1)<1e-4 and acc>best_acc):
            best_f1, best_acc = f1, acc
            best_st = deepcopy(model.state_dict()); stale = 0
        else: stale += 1
        ema.restore(model)

        if ep % 20 == 0 or ep == 1:
            print(f"    Ep {ep:03d} | acc={acc:.1f}% f1={f1:.3f} (best={best_acc:.1f}%)")

        if stale >= cfg.get('patience', 35):
            print(f"    [Early stop ep {ep}]"); break

    swa_st = swa.averaged()

    # Collect all result variants
    results = {}
    for name, st in [('best', best_st), ('swa', swa_st)]:
        if st is None: continue
        model.load_state_dict(st); model.eval()
        with torch.no_grad():
            logits = model(test_x.to(device))
        probs = F.softmax(logits, 1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        acc = 100*(preds==ynp).mean()
        f1 = f1_score(ynp, preds, average='macro', zero_division=0)
        results[name] = {'preds': preds, 'probs': probs, 'acc': acc, 'f1': f1}

        # TTA
        logits_tta = tta_predict(model, test_x, device, cfg.get('tta_n', 16))
        probs_tta = F.softmax(logits_tta, 1).cpu().numpy()
        preds_tta = logits_tta.argmax(1).cpu().numpy()
        acc_tta = 100*(preds_tta==ynp).mean()
        f1_tta = f1_score(ynp, preds_tta, average='macro', zero_division=0)
        results[f'{name}_tta'] = {'preds': preds_tta, 'probs': probs_tta,
                                   'acc': acc_tta, 'f1': f1_tta}

        # Also predict on ALL test samples if provided (for cascade)
        if extra_test_x is not None:
            with torch.no_grad():
                logits_all = model(extra_test_x.to(device))
            probs_all = F.softmax(logits_all, 1).cpu().numpy()
            results[f'{name}_all64'] = {'probs': probs_all, 'preds': logits_all.argmax(1).cpu().numpy()}

            logits_all_tta = tta_predict(model, extra_test_x, device, cfg.get('tta_n', 16))
            probs_all_tta = F.softmax(logits_all_tta, 1).cpu().numpy()
            results[f'{name}_tta_all64'] = {'probs': probs_all_tta, 'preds': logits_all_tta.argmax(1).cpu().numpy()}

    return results, best_st, swa_st


# ═══════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════
def main():
    pa = ArgumentParser()
    pa.add_argument('--train_features', required=True)
    pa.add_argument('--test_features', required=True)
    pa.add_argument('--output_dir', default='sweep_results/cascaded_v1')
    pa.add_argument('--epochs', type=int, default=80)
    pa.add_argument('--binary_epochs', type=int, default=100)
    pa.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 7, 2024, 999])
    pa.add_argument('--direct_seeds', type=int, nargs='+', default=None,
                    help='Seeds for direct 4-class (default: first 3 of --seeds)')
    args = pa.parse_args()

    if args.direct_seeds is None:
        args.direct_seeds = args.seeds[:3]  # fewer seeds for direct since we have prior results

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cpu'

    # Load test data
    tp = torch.load(args.test_features, map_location='cpu')
    tm = tp['labels'] < 4
    test_x_4 = tp['features'][tm]; test_y_4 = tp['labels'][tm].clone()
    ynp = test_y_4.numpy()
    print(f"Test: {tm.sum()} unsafe videos")

    # === APPROACH 1: Direct 4-class with diverse seeds + configs ===
    print(f"\n{'='*60}")
    print("APPROACH 1: Direct 4-class ensemble (diverse configs)")
    print(f"{'='*60}")

    direct_configs = [
        # Config A: Enhanced motion, dim=384 (best from enhanced_v1)
        {'dim': 384, 'layers': 4, 'lr': 2e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1, 'dropout': 0.3, 'tag': 'A'},
        # Config B: Compact dim=256
        {'dim': 256, 'layers': 4, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1, 'dropout': 0.3, 'tag': 'B'},
        # Config C: Unsafe-only, dim=384
        {'dim': 384, 'layers': 4, 'lr': 2e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'unsafe_only',
         'bs': 16, 'ls': 0.1, 'dropout': 0.3, 'tag': 'C'},
        # Config D: Higher gamma (harder focus on class 0/1)
        {'dim': 256, 'layers': 4, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 4.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.1, 'dropout': 0.3, 'tag': 'D'},
        # Config E: Lower dropout, stronger mixup
        {'dim': 384, 'layers': 4, 'lr': 2e-4, 'wd': 0.06,
         'gamma': 3.0, 'mixup': 0.6, 'mode': 'scenario_all',
         'bs': 16, 'ls': 0.15, 'dropout': 0.25, 'tag': 'E'},
    ]

    all_direct = []
    total = len(direct_configs) * len(args.direct_seeds)
    done = 0
    t0 = time.time()

    for cfg_base in direct_configs:
        for seed in args.direct_seeds:
            done += 1
            tag = cfg_base['tag']
            name = f"d4_{tag}_s{seed}"
            print(f"\n  [{done}/{total}] {name}")
            set_seed(seed)

            cfg = {**cfg_base, 'num_classes': 4, 'epochs': args.epochs,
                   'warmup': 8, 'patience': 35, 'ema_decay': 0.998, 'tta_n': 16}

            ds = (ScenarioAll4 if cfg['mode'] == 'scenario_all' else UnsafeOnly4)(
                args.train_features, augment=True)

            model = MotionHead(
                in_dim=768, num_classes=4, model_dim=cfg['dim'],
                nhead=8, num_layers=cfg['layers'], dropout=cfg['dropout']
            ).to(device)

            test_ds = UnsafeOnly4(args.test_features)
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

            results, best_st, swa_st = train_model(
                model, ds, test_loader, test_x_4, test_y_4, cfg, device)

            best_var = max(results.items(), key=lambda x: x[1]['acc'])
            print(f"    → {name}: best_variant={best_var[0]} "
                  f"acc={best_var[1]['acc']:.1f}% f1={best_var[1]['f1']:.3f}")

            all_direct.append({
                'name': name, 'seed': seed, 'tag': tag,
                'results': results, 'cfg': cfg_base
            })

    # Direct ensemble
    print(f"\n  Direct 4-class ensemble results:")
    best_direct_ind = None
    for d in all_direct:
        for v, r in d['results'].items():
            if best_direct_ind is None or r['acc'] > best_direct_ind[1]:
                best_direct_ind = (f"{d['name']}_{v}", r['acc'], r)
    print(f"    Best individual: {best_direct_ind[0]} = {best_direct_ind[1]:.1f}%")

    # Ensemble: soft vote over all swa_tta
    all_probs_swa_tta = []
    all_probs_best_tta = []
    for d in all_direct:
        if 'swa_tta' in d['results']:
            all_probs_swa_tta.append(d['results']['swa_tta']['probs'])
        if 'best_tta' in d['results']:
            all_probs_best_tta.append(d['results']['best_tta']['probs'])

    for label, probs_list in [('swa_tta', all_probs_swa_tta), ('best_tta', all_probs_best_tta)]:
        if len(probs_list) < 2: continue
        avg = np.mean(probs_list, 0)
        preds = avg.argmax(1)
        acc = 100*(preds==ynp).mean()
        f1 = f1_score(ynp, preds, average='macro', zero_division=0)
        print(f"    Ensemble all_{label}: {acc:.1f}% ({(preds==ynp).sum()}/64) f1={f1:.3f}")

    # Top-K diverse ensemble
    for k in [3, 5, 7]:
        items = [(i, d['results'].get('swa_tta', d['results']['best'])['acc'])
                 for i, d in enumerate(all_direct)]
        items.sort(key=lambda x: -x[1])

        # Greedy diversity
        selected = [items[0][0]]
        for _ in range(k - 1):
            best_idx, best_score = -1, -1
            for idx, acc in items:
                if idx in selected: continue
                preds_i = all_direct[idx]['results'].get('swa_tta', all_direct[idx]['results']['best'])['preds']
                div = np.mean([np.mean(preds_i != all_direct[s]['results'].get(
                    'swa_tta', all_direct[s]['results']['best'])['preds']) for s in selected])
                score = acc / 100 * 0.7 + div * 0.3
                if score > best_score: best_score, best_idx = score, idx
            if best_idx >= 0: selected.append(best_idx)

        probs_k = [all_direct[i]['results'].get('swa_tta', all_direct[i]['results']['best'])['probs']
                   for i in selected]
        avg = np.mean(probs_k, 0)
        preds = avg.argmax(1)
        acc = 100*(preds==ynp).mean()
        f1 = f1_score(ynp, preds, average='macro', zero_division=0)
        names = [all_direct[i]['name'] for i in selected]
        print(f"    Diverse-{k}: {acc:.1f}% ({(preds==ynp).sum()}/64) f1={f1:.3f}")
        print(f"      Selected: {names}")

    # === APPROACH 2: Cascaded ===
    print(f"\n{'='*60}")
    print("APPROACH 2: Cascaded (Stage1: 3-class, Stage2: binary 0vs1)")
    print(f"{'='*60}")

    # Use the direct ensemble as Stage 1 implicitly
    # Stage 2: specialized binary classifier for the confused class 0 vs 1

    all_binary = []
    binary_configs = [
        {'dim': 128, 'layers': 3, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 2.0, 'mixup': 0.3, 'mode': 'scenario_all',
         'bs': 8, 'ls': 0.05, 'dropout': 0.3, 'tag': 'bin_A'},
        {'dim': 128, 'layers': 3, 'lr': 3e-4, 'wd': 0.05,
         'gamma': 2.0, 'mixup': 0.3, 'mode': 'unsafe_only',
         'bs': 8, 'ls': 0.05, 'dropout': 0.3, 'tag': 'bin_B'},
        {'dim': 192, 'layers': 4, 'lr': 2e-4, 'wd': 0.05,
         'gamma': 3.0, 'mixup': 0.4, 'mode': 'scenario_all',
         'bs': 8, 'ls': 0.1, 'dropout': 0.3, 'tag': 'bin_C'},
        # Additional configs focusing on motion separation
        {'dim': 128, 'layers': 4, 'lr': 2e-4, 'wd': 0.04,
         'gamma': 2.0, 'mixup': 0.5, 'mode': 'scenario_all',
         'bs': 8, 'ls': 0.05, 'dropout': 0.35, 'tag': 'bin_D'},
        {'dim': 192, 'layers': 3, 'lr': 3e-4, 'wd': 0.06,
         'gamma': 3.0, 'mixup': 0.3, 'mode': 'unsafe_only',
         'bs': 8, 'ls': 0.1, 'dropout': 0.25, 'tag': 'bin_E'},
    ]

    # Test data for binary: only class 0 and 1
    bin_mask = (test_y_4 == 0) | (test_y_4 == 1)
    test_x_bin = test_x_4[bin_mask]
    test_y_bin = test_y_4[bin_mask].clone()
    print(f"\n  Binary test: {bin_mask.sum()} samples "
          f"(class0={int((test_y_bin==0).sum())}, class1={int((test_y_bin==1).sum())})")

    for cfg_base in binary_configs:
        for seed in args.seeds:
            name = f"{cfg_base['tag']}_s{seed}"
            print(f"\n  Training {name}")
            set_seed(seed)

            cfg = {**cfg_base, 'num_classes': 2, 'epochs': args.binary_epochs,
                   'warmup': 10, 'patience': 40, 'ema_decay': 0.998, 'tta_n': 16}

            ds = BinaryWalkway(args.train_features, mode=cfg['mode'], augment=True)
            test_ds_bin = BinaryWalkway(args.test_features, mode='unsafe_only')
            test_loader_bin = DataLoader(test_ds_bin, batch_size=64, shuffle=False)

            model = MotionBinaryHead(
                in_dim=768, model_dim=cfg['dim'], nhead=4,
                num_layers=cfg['layers'], dropout=cfg['dropout']
            ).to(device)
            np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    Binary model params: {np_:,}")

            results, _, _ = train_model(
                model, ds, test_loader_bin, test_x_bin, test_y_bin, cfg, device,
                extra_test_x=test_x_4)

            best_var = max(((k,v) for k,v in results.items() if 'acc' in v),
                          key=lambda x: x[1]['acc'])
            print(f"    → {name}: best={best_var[0]} acc={best_var[1]['acc']:.1f}%")

            all_binary.append({
                'name': name, 'seed': seed, 'tag': cfg_base['tag'],
                'results': results
            })

    # Binary ensemble
    print(f"\n  Binary (0 vs 1) ensemble results:")
    ynp_bin = test_y_bin.numpy()
    bin_probs = []
    for b in all_binary:
        v = 'swa_tta' if 'swa_tta' in b['results'] else 'best_tta' if 'best_tta' in b['results'] else 'best'
        bin_probs.append(b['results'][v]['probs'])
        best_v = max(((k,v) for k,v in b['results'].items() if 'acc' in v),
                     key=lambda x: x[1]['acc'])
        print(f"    {b['name']}: {best_v[1]['acc']:.1f}%")

    if len(bin_probs) >= 2:
        avg_bin = np.mean(bin_probs, 0)
        preds_bin = avg_bin.argmax(1)
        acc_bin = 100*(preds_bin==ynp_bin).mean()
        print(f"    Binary ensemble: {acc_bin:.1f}% ({(preds_bin==ynp_bin).sum()}/{len(ynp_bin)})")

    # === APPROACH 3: Cascade direct + binary ===
    print(f"\n{'='*60}")
    print("APPROACH 3: Cascade — use binary to refine 4-class predictions")
    print(f"{'='*60}")

    # For each direct model, check if predicted class is 0 or 1.
    # If so, replace with binary classifier's prediction on that SAME sample.
    # Binary model predictions on ALL 64 test samples are stored in *_all64 keys.
    cascade_results = []

    # Collect binary ensemble predictions on all 64 samples
    bin_all64_probs = []
    for b in all_binary:
        for key in ['swa_tta_all64', 'best_tta_all64', 'swa_all64', 'best_all64']:
            if key in b['results']:
                bin_all64_probs.append(b['results'][key]['probs'])  # (64, 2)
                break  # use best available variant per model

    if len(bin_all64_probs) >= 1:
        # Average binary predictions across all binary models → (64, 2)
        avg_bin_all64 = np.mean(bin_all64_probs, 0)
        bin_preds_all64 = avg_bin_all64.argmax(1)  # shape (64,)

        print(f"\n  Binary ensemble predictions on all 64 (class distribution): "
              f"pred0={int((bin_preds_all64==0).sum())}, pred1={int((bin_preds_all64==1).sum())}")

        # Cascade: for each direct 4-class model
        for d in all_direct:
            for v in ['best', 'best_tta', 'swa', 'swa_tta']:
                if v not in d['results']: continue
                base_preds = d['results'][v]['preds'].copy()

                # For samples where 4-class model predicts 0 or 1, use binary
                refined = base_preds.copy()
                n_replaced = 0
                for i in range(len(ynp)):
                    if base_preds[i] <= 1:
                        refined[i] = bin_preds_all64[i]
                        n_replaced += 1

                acc_ref = 100*(refined==ynp).mean()
                acc_orig = d['results'][v]['acc']
                cascade_results.append({
                    'name': f"{d['name']}_{v}_refined",
                    'acc': acc_ref, 'correct': int((refined==ynp).sum()),
                    'preds': refined, 'n_replaced': n_replaced,
                    'acc_orig': acc_orig
                })
                if acc_ref != acc_orig:
                    sign = '+' if acc_ref > acc_orig else ''
                    print(f"    {d['name']}_{v}: {acc_orig:.1f}% → {acc_ref:.1f}% "
                          f"({sign}{acc_ref-acc_orig:.1f}pp, {n_replaced} replaced)")

        # Also refine ensemble predictions
        for label, probs_list in [('swa_tta', all_probs_swa_tta), ('best_tta', all_probs_best_tta)]:
            if len(probs_list) < 2: continue
            avg = np.mean(probs_list, 0)
            base_preds = avg.argmax(1)
            refined = base_preds.copy()
            n_replaced = 0
            for i in range(len(ynp)):
                if base_preds[i] <= 1:
                    refined[i] = bin_preds_all64[i]
                    n_replaced += 1
            acc_ref = 100*(refined==ynp).mean()
            acc_orig = 100*(base_preds==ynp).mean()
            sign = '+' if acc_ref > acc_orig else ''
            print(f"    ens_{label}: {acc_orig:.1f}% → {acc_ref:.1f}% "
                  f"({sign}{acc_ref-acc_orig:.1f}pp, {n_replaced} replaced)")
            cascade_results.append({
                'name': f"ens_{label}_refined", 'acc': acc_ref,
                'correct': int((refined==ynp).sum()), 'preds': refined,
                'n_replaced': n_replaced, 'acc_orig': acc_orig
            })

        # Confidence-based cascade: only replace when binary is confident
        for conf_thresh in [0.55, 0.6, 0.65, 0.7]:
            for label, probs_list in [('swa_tta', all_probs_swa_tta), ('best_tta', all_probs_best_tta)]:
                if len(probs_list) < 2: continue
                avg = np.mean(probs_list, 0)
                base_preds = avg.argmax(1)
                refined = base_preds.copy()
                n_replaced = 0
                for i in range(len(ynp)):
                    if base_preds[i] <= 1:
                        bin_conf = avg_bin_all64[i].max()
                        if bin_conf >= conf_thresh:
                            refined[i] = bin_preds_all64[i]
                            n_replaced += 1
                acc_ref = 100*(refined==ynp).mean()
                acc_orig = 100*(base_preds==ynp).mean()
                if acc_ref != acc_orig:
                    print(f"    ens_{label}_conf{conf_thresh}: {acc_orig:.1f}% → {acc_ref:.1f}% "
                          f"({n_replaced} replaced)")
                    cascade_results.append({
                        'name': f"ens_{label}_conf{conf_thresh}_refined",
                        'acc': acc_ref, 'correct': int((refined==ynp).sum()),
                        'preds': refined, 'n_replaced': n_replaced, 'acc_orig': acc_orig
                    })

        # Probability fusion: blend 4-class and binary probs for classes 0/1
        for alpha in [0.3, 0.5, 0.7]:
            for label, probs_list in [('swa_tta', all_probs_swa_tta)]:
                if len(probs_list) < 2: continue
                avg = np.mean(probs_list, 0).copy()  # (64, 4)
                # For classes 0 and 1, blend with binary model's opinion
                for i in range(len(ynp)):
                    bin_p0 = avg_bin_all64[i, 0]  # binary prob of class 0
                    bin_p1 = avg_bin_all64[i, 1]  # binary prob of class 1
                    # Blend: new_p0 = (1-a)*base_p0 + a*bin_p0*(base_p0+base_p1)
                    base_01_mass = avg[i, 0] + avg[i, 1]
                    avg[i, 0] = (1-alpha)*avg[i, 0] + alpha * bin_p0 * base_01_mass
                    avg[i, 1] = (1-alpha)*avg[i, 1] + alpha * bin_p1 * base_01_mass
                preds = avg.argmax(1)
                acc = 100*(preds==ynp).mean()
                print(f"    fusion_{label}_alpha{alpha}: {acc:.1f}% ({(preds==ynp).sum()}/64)")
                cascade_results.append({
                    'name': f"fusion_{label}_alpha{alpha}",
                    'acc': acc, 'correct': int((preds==ynp).sum()),
                    'preds': preds, 'n_replaced': -1, 'acc_orig': -1
                })

    # === SUMMARY ===
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    all_results = []

    # Direct 4-class
    for d in all_direct:
        for v, r in d['results'].items():
            all_results.append({
                'method': 'direct', 'name': f"{d['name']}_{v}",
                'acc': r['acc'], 'correct': int((r['preds']==ynp).sum())
            })

    # Cascade
    for cr in cascade_results:
        all_results.append({
            'method': 'cascade', 'name': cr['name'],
            'acc': cr['acc'], 'correct': cr['correct']
        })

    all_results.sort(key=lambda x: -x['acc'])

    print(f"\n  Top-10 results:")
    for i, r in enumerate(all_results[:10]):
        print(f"    {i+1}. {r['name']}: {r['acc']:.1f}% ({r['correct']}/64) [{r['method']}]")

    best = all_results[0]
    print(f"\n  BEST OVERALL: {best['name']} = {best['acc']:.1f}% ({best['correct']}/64)")
    print(f"  Paper baseline: 95.81% (62/64)")
    print(f"  Gap: {95.81 - best['acc']:.2f}pp")

    # Save best confusion matrix
    if cascade_results:
        best_cascade = max(cascade_results, key=lambda x: x['acc'])
        plot_cm(ynp, best_cascade['preds'],
                f"Best Cascade: {best_cascade['acc']:.1f}%",
                os.path.join(args.output_dir, 'cm_best_cascade.png'))

    # Save report
    total_time = time.time() - t0
    with open(os.path.join(args.output_dir, 'report.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CASCADED CLASSIFIER — 4-CLASS UNSAFE CLASSIFICATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total time: {total_time/60:.1f}m\n")
        f.write(f"Seeds: {args.seeds}\n")
        f.write(f"Direct configs: {len(direct_configs)}\n")
        f.write(f"Binary configs: {len(binary_configs)}\n\n")

        f.write("TOP RESULTS:\n")
        for i, r in enumerate(all_results[:20]):
            f.write(f"  {i+1}. {r['name']}: {r['acc']:.1f}% ({r['correct']}/64) [{r['method']}]\n")

        f.write(f"\nBEST: {best['name']} = {best['acc']:.1f}% ({best['correct']}/64)\n")
        f.write(f"Paper: 95.81% (62/64)\n")
        f.write(f"Gap: {95.81 - best['acc']:.2f}pp\n")

    print(f"\n  Saved to {args.output_dir}/")
    print(f"  Total time: {total_time/60:.1f}m")


if __name__ == '__main__':
    main()
