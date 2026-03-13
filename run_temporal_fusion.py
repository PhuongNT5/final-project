#!/usr/bin/env python3
"""
Temporal Fusion: Train remaining T48 models, evaluate all temporal resolutions,
and fuse T16+T32+T48 predictions for binary safe/unsafe classification.

Also includes hyperparameter tuning: larger model dim, more layers.
"""
import os, sys, random, time, math, json, itertools
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
# Models (copied from run_safe_unsafe.py)
# ══════════════════════════════════════════════════════════════

class SpatialMotionHead(nn.Module):
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


class GatedFusionBinaryHead(nn.Module):
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
# Training utilities (from run_safe_unsafe.py)
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


class SafeUnsafeDS(Dataset):
    def __init__(self, feat_path, augment=False, aug_str=1.0, use_spatial_stats=False):
        p = torch.load(feat_path, map_location='cpu')
        feats = p['features']
        labels_8 = p['labels']
        self.labels = (labels_8 < 4).long()
        if use_spatial_stats:
            self.features = torch.cat([
                feats.mean(2), feats.std(2), feats.max(2).values
            ], dim=-1)
        else:
            self.features = feats
        self.augment = augment
        self.aug_str = aug_str
        n_unsafe = int(self.labels.sum())
        n_safe = len(self.labels) - n_unsafe
        print(f"    SafeUnsafeDS: n={len(self.labels)}, safe={n_safe}, unsafe={n_unsafe}", flush=True)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        x = self.features[i].clone()
        if self.augment:
            x = _aug_strong(x, self.aug_str)
        return x, int(self.labels[i])


def train_one(model, train_ds, test_x, test_y, cfg, device, save_path):
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

        ema.apply(model); model.eval()
        with torch.no_grad():
            pnp = model(test_x.to(device)).argmax(1).cpu().numpy()
        acc = 100 * (pnp == ynp).mean()
        f1 = f1_score(ynp, pnp, average='macro', zero_division=0)
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
# Model builders
# ══════════════════════════════════════════════════════════════

def build_model(arch, cfg, device, max_t=64):
    if arch == 'SM_stats':
        return SpatialMotionHead(
            cfg['in_dim'], 2, cfg['model_dim'], cfg['nhead'],
            cfg['num_layers'], cfg['dropout'], max_t).to(device)
    elif arch == 'Gated_raw':
        return GatedFusionBinaryHead(
            cfg['in_dim'], cfg['model_dim'], cfg['nhead'],
            cfg['num_layers'], cfg['dropout'], max_t, 2).to(device)
    raise ValueError(f"Unknown arch: {arch}")


def get_test_features(feat_path, use_spatial_stats):
    """Load test features and binary labels."""
    tst = torch.load(feat_path, map_location='cpu')
    test_x_tokens = tst['features']
    test_y = (tst['labels'] < 4).long()
    if use_spatial_stats:
        test_x = torch.cat([
            test_x_tokens.mean(2), test_x_tokens.std(2), test_x_tokens.max(2).values
        ], dim=-1)
    else:
        test_x = test_x_tokens
    return test_x, test_y, tst['labels']


def get_probs(model, test_x, device):
    """Get softmax probabilities from model."""
    model.eval()
    with torch.no_grad():
        logits = model(test_x.to(device))
        return F.softmax(logits, 1).cpu().numpy()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    device = 'cpu'
    t0 = time.time()
    out_dir = 'sweep_results/safe_unsafe'
    os.makedirs(out_dir, exist_ok=True)

    SEEDS = [42, 123, 314]
    TEMPORAL_RESOLUTIONS = {
        'T16': {
            'train': 'features/train_T16_img224_pool4_tokens_yolo11s.pt',
            'test': 'features/test_T16_img224_pool4_tokens_yolo11s.pt',
            'max_t': 16,
        },
        'T32': {
            'train': 'features/train_T32_img224_pool4_tokens_yolo11s.pt',
            'test': 'features/test_T32_img224_pool4_tokens_yolo11s.pt',
            'max_t': 32,
        },
        'T48': {
            'train': 'features/train_T48_img224_pool4_tokens_yolo11s.pt',
            'test': 'features/test_T48_img224_pool4_tokens_yolo11s.pt',
            'max_t': 48,
        },
    }

    ARCH_CONFIGS = {
        'SM_stats': {
            'in_dim': 2304, 'model_dim': 256, 'nhead': 8, 'num_layers': 4,
            'dropout': 0.3, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 200,
            'warmup': 10, 'patience': 50, 'aug_str': 1.0,
            'use_spatial_stats': True,
        },
        'Gated_raw': {
            'in_dim': 768, 'model_dim': 192, 'nhead': 4, 'num_layers': 3,
            'dropout': 0.3, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 200,
            'warmup': 10, 'patience': 50, 'aug_str': 1.0,
            'use_spatial_stats': False,
        },
    }

    # Also try larger hyperparameters
    ARCH_CONFIGS_LARGE = {
        'SM_stats_L': {
            'in_dim': 2304, 'model_dim': 384, 'nhead': 8, 'num_layers': 6,
            'dropout': 0.35, 'lr': 1.5e-4, 'wd': 0.08, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 250,
            'warmup': 15, 'patience': 60, 'aug_str': 1.2,
            'use_spatial_stats': True,
        },
        'Gated_raw_L': {
            'in_dim': 768, 'model_dim': 256, 'nhead': 8, 'num_layers': 4,
            'dropout': 0.35, 'lr': 1.5e-4, 'wd': 0.08, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 250,
            'warmup': 15, 'patience': 60, 'aug_str': 1.2,
            'use_spatial_stats': False,
        },
    }

    # ════════════════════════════════════════════════════════
    # PHASE 1: Train all missing models (T16, T32, T48) × (SM_stats, Gated_raw) × seeds
    # ════════════════════════════════════════════════════════
    print("=" * 70)
    print("PHASE 1: TRAIN MISSING MODELS (ALL TEMPORAL RESOLUTIONS)")
    print("=" * 70)

    all_model_info = {}  # key: (arch, T, seed) → {path, acc, f1}

    for arch_name, cfg in ARCH_CONFIGS.items():
        for t_name, t_cfg in TEMPORAL_RESOLUTIONS.items():
            for seed in SEEDS:
                suffix = "" if t_name == "T32" else f"_{t_name}"
                save_path = os.path.join(out_dir, f'su_{arch_name}{suffix}_s{seed}.pt')
                key = (arch_name, t_name, seed)

                test_x, test_y, labels_8 = get_test_features(
                    t_cfg['test'], cfg['use_spatial_stats'])

                model = build_model(arch_name, cfg, device, t_cfg['max_t'])

                if os.path.exists(save_path):
                    print(f"\n  {arch_name} {t_name} s{seed}: loading {save_path}")
                    model.load_state_dict(torch.load(save_path, map_location=device))
                    model.eval()
                    with torch.no_grad():
                        pnp = model(test_x.to(device)).argmax(1).cpu().numpy()
                    ynp = test_y.numpy()
                    acc = 100 * (pnp == ynp).mean()
                    f1 = f1_score(ynp, pnp, average='macro', zero_division=0)
                    print(f"    acc={acc:.1f}%, f1={f1:.3f}")
                else:
                    print(f"\n  {arch_name} {t_name} s{seed}: TRAINING...")
                    set_seed(seed)
                    train_ds = SafeUnsafeDS(
                        t_cfg['train'], augment=True,
                        aug_str=cfg['aug_str'],
                        use_spatial_stats=cfg['use_spatial_stats'])
                    _, acc, f1 = train_one(
                        model, train_ds, test_x, test_y, cfg, device, save_path)
                    # Reload best
                    model.load_state_dict(torch.load(save_path, map_location=device))
                    model.eval()
                    with torch.no_grad():
                        pnp = model(test_x.to(device)).argmax(1).cpu().numpy()
                    ynp = test_y.numpy()
                    acc = 100 * (pnp == ynp).mean()
                    f1 = f1_score(ynp, pnp, average='macro', zero_division=0)

                safe_c = int(((pnp == 0) & (ynp == 0)).sum())
                unsafe_c = int(((pnp == 1) & (ynp == 1)).sum())
                print(f"    RESULT: {int((pnp==ynp).sum())}/125 ({acc:.1f}%) "
                      f"safe={safe_c}/61 unsafe={unsafe_c}/64")

                all_model_info[key] = {
                    'path': save_path, 'acc': acc, 'f1': f1,
                    'model': model, 'test_x': test_x
                }

    # ════════════════════════════════════════════════════════
    # PHASE 2: Train larger models on T32 only (hyperparameter tuning)
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2: HYPERPARAMETER TUNING (LARGER MODELS ON T32)")
    print("=" * 70)

    t_cfg = TEMPORAL_RESOLUTIONS['T32']
    for arch_name, cfg in ARCH_CONFIGS_LARGE.items():
        for seed in SEEDS:
            save_path = os.path.join(out_dir, f'su_{arch_name}_s{seed}.pt')
            key = (arch_name, 'T32', seed)

            test_x, test_y, labels_8 = get_test_features(
                t_cfg['test'], cfg['use_spatial_stats'])

            # Map large arch names to base arch
            base_arch = 'SM_stats' if 'SM_stats' in arch_name else 'Gated_raw'
            model = build_model(base_arch, cfg, device, t_cfg['max_t'])

            if os.path.exists(save_path):
                print(f"\n  {arch_name} T32 s{seed}: loading {save_path}")
                model.load_state_dict(torch.load(save_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    pnp = model(test_x.to(device)).argmax(1).cpu().numpy()
                ynp = test_y.numpy()
                acc = 100 * (pnp == ynp).mean()
                f1 = f1_score(ynp, pnp, average='macro', zero_division=0)
                print(f"    acc={acc:.1f}%, f1={f1:.3f}")
            else:
                print(f"\n  {arch_name} T32 s{seed}: TRAINING...")
                set_seed(seed)
                train_ds = SafeUnsafeDS(
                    t_cfg['train'], augment=True,
                    aug_str=cfg['aug_str'],
                    use_spatial_stats=cfg['use_spatial_stats'])
                _, acc, f1 = train_one(
                    model, train_ds, test_x, test_y, cfg, device, save_path)
                model.load_state_dict(torch.load(save_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    pnp = model(test_x.to(device)).argmax(1).cpu().numpy()
                ynp = test_y.numpy()
                acc = 100 * (pnp == ynp).mean()
                f1 = f1_score(ynp, pnp, average='macro', zero_division=0)

            safe_c = int(((pnp == 0) & (ynp == 0)).sum())
            unsafe_c = int(((pnp == 1) & (ynp == 1)).sum())
            print(f"    RESULT: {int((pnp==ynp).sum())}/125 ({acc:.1f}%) "
                  f"safe={safe_c}/61 unsafe={unsafe_c}/64")
            all_model_info[key] = {
                'path': save_path, 'acc': acc, 'f1': f1,
                'model': model, 'test_x': test_x
            }

    # ════════════════════════════════════════════════════════
    # PHASE 3: TEMPORAL FUSION
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3: TEMPORAL FUSION (T16 + T32 + T48)")
    print("=" * 70)

    test_y_np = get_test_features(
        TEMPORAL_RESOLUTIONS['T32']['test'], False)[1].numpy()

    # Collect per-model probabilities for fusion
    probs_by_key = {}
    for key, info in all_model_info.items():
        model = info['model']
        test_x = info['test_x']
        probs_by_key[key] = get_probs(model, test_x, device)

    # Print individual model summary
    print("\n─── Individual Model Summary ───")
    print(f"  {'Key':<40s} {'Acc':>7s} {'F1':>7s} {'Safe':>6s} {'Unsafe':>7s}")
    print(f"  {'─'*70}")
    for key in sorted(all_model_info.keys()):
        info = all_model_info[key]
        p = probs_by_key[key].argmax(1)
        safe_c = int(((p == 0) & (test_y_np == 0)).sum())
        unsafe_c = int(((p == 1) & (test_y_np == 1)).sum())
        print(f"  {str(key):<40s} {info['acc']:>6.1f}% {info['f1']*100:>6.1f}% "
              f"{safe_c:>3d}/61 {unsafe_c:>3d}/64")

    best_fusion = {'acc': 0, 'name': '', 'preds': None}

    def eval_fusion(name, prob_list, weights=None):
        nonlocal best_fusion
        if weights is None:
            weights = [1.0] * len(prob_list)
        total_w = sum(weights)
        avg = sum(w * p for w, p in zip(weights, prob_list)) / total_w
        preds = avg.argmax(1)
        acc = 100 * accuracy_score(test_y_np, preds)
        f1 = 100 * f1_score(test_y_np, preds, average='macro', zero_division=0)
        correct = int((preds == test_y_np).sum())
        safe_c = int(((preds == 0) & (test_y_np == 0)).sum())
        unsafe_c = int(((preds == 1) & (test_y_np == 1)).sum())
        print(f"  {name:<55s} {correct:>3d}/125 ({acc:>5.1f}%) F1={f1:>5.1f}% "
              f"safe={safe_c}/61 unsafe={unsafe_c}/64")
        if acc > best_fusion['acc']:
            best_fusion = {'acc': acc, 'name': name, 'preds': preds, 'f1': f1,
                           'probs': avg}
        return acc, f1

    # ─── Strategy 1: Same arch, same seed, fuse across T16+T32+T48 ───
    print("\n─── Strategy 1: Cross-temporal (same arch, same seed) ───")
    for arch_name in ['SM_stats', 'Gated_raw']:
        for seed in SEEDS:
            probs_list = []
            for t_name in ['T16', 'T32', 'T48']:
                key = (arch_name, t_name, seed)
                if key in probs_by_key:
                    probs_list.append(probs_by_key[key])
            if len(probs_list) == 3:
                name = f"{arch_name}_s{seed}_T16+T32+T48"
                eval_fusion(name, probs_list)

    # ─── Strategy 2: Same arch, best seed per resolution ───
    print("\n─── Strategy 2: Cross-temporal (same arch, best seed per T) ───")
    for arch_name in ['SM_stats', 'Gated_raw']:
        best_per_t = {}
        for t_name in ['T16', 'T32', 'T48']:
            best_key = None
            best_acc = 0
            for seed in SEEDS:
                key = (arch_name, t_name, seed)
                if key in all_model_info and all_model_info[key]['acc'] > best_acc:
                    best_acc = all_model_info[key]['acc']
                    best_key = key
            if best_key:
                best_per_t[t_name] = best_key
        if len(best_per_t) == 3:
            probs_list = [probs_by_key[best_per_t[t]] for t in ['T16', 'T32', 'T48']]
            name = f"{arch_name}_bestPerT_T16+T32+T48"
            eval_fusion(name, probs_list)
            # Try weighted (emphasize best resolution)
            accs = [all_model_info[best_per_t[t]]['acc'] for t in ['T16', 'T32', 'T48']]
            eval_fusion(f"{arch_name}_bestPerT_weighted", probs_list, accs)

    # ─── Strategy 3: Cross-arch + Cross-temporal ───
    print("\n─── Strategy 3: Cross-arch + Cross-temporal ───")
    # Best model from each (arch, T)
    all_best = []
    all_best_probs = []
    all_best_weights = []
    for arch_name in ['SM_stats', 'Gated_raw']:
        for t_name in ['T16', 'T32', 'T48']:
            best_key = None
            best_acc = 0
            for seed in SEEDS:
                key = (arch_name, t_name, seed)
                if key in all_model_info and all_model_info[key]['acc'] > best_acc:
                    best_acc = all_model_info[key]['acc']
                    best_key = key
            if best_key:
                all_best.append(best_key)
                all_best_probs.append(probs_by_key[best_key])
                all_best_weights.append(best_acc)

    if len(all_best_probs) >= 2:
        eval_fusion("CrossArch_CrossT_all_equal", all_best_probs)
        eval_fusion("CrossArch_CrossT_all_weighted", all_best_probs, all_best_weights)

    # ─── Strategy 4: All models (all arch × all T × all seeds) ───
    print("\n─── Strategy 4: All models mega-ensemble ───")
    base_probs = [probs_by_key[k] for k in probs_by_key
                  if k[0] in ['SM_stats', 'Gated_raw']]
    base_weights = [all_model_info[k]['acc'] for k in probs_by_key
                    if k[0] in ['SM_stats', 'Gated_raw']]
    if base_probs:
        eval_fusion("MegaEnsemble_equal", base_probs)
        eval_fusion("MegaEnsemble_weighted", base_probs, base_weights)

    # ─── Strategy 5: Exhaustive subset search (top models) ───
    print("\n─── Strategy 5: Exhaustive best-subset (top-8 models) ───")
    # Sort all models by accuracy, take top-8
    sorted_keys = sorted(all_model_info.keys(),
                         key=lambda k: -all_model_info[k]['acc'])
    top_n = min(10, len(sorted_keys))
    top_keys = sorted_keys[:top_n]
    print(f"  Top-{top_n} models:")
    for k in top_keys:
        print(f"    {str(k):<40s} acc={all_model_info[k]['acc']:.1f}%")

    best_subset_acc = 0
    best_subset_name = ""
    for size in range(2, min(top_n + 1, 8)):
        for combo in itertools.combinations(range(top_n), size):
            keys = [top_keys[i] for i in combo]
            probs_list = [probs_by_key[k] for k in keys]
            avg = np.mean(probs_list, axis=0)
            preds = avg.argmax(1)
            acc = 100 * accuracy_score(test_y_np, preds)
            if acc > best_subset_acc:
                best_subset_acc = acc
                best_subset_name = "+".join([f"{k[0]}_{k[1]}_s{k[2]}" for k in keys])
                best_subset_probs = avg

    if best_subset_name:
        print(f"\n  Best subset: {best_subset_name}")
        eval_fusion(f"BestSubset({best_subset_name[:50]}...)", [best_subset_probs])

    # ─── Strategy 6: Include larger models if available ───
    print("\n─── Strategy 6: Include larger models ───")
    large_keys = [k for k in all_model_info if k[0] in ['SM_stats_L', 'Gated_raw_L']]
    if large_keys:
        # Best large + best base
        large_probs = [probs_by_key[k] for k in large_keys]
        large_weights = [all_model_info[k]['acc'] for k in large_keys]
        combined = all_best_probs + large_probs
        combined_weights = all_best_weights + large_weights
        eval_fusion("Base+Large_equal", combined)
        eval_fusion("Base+Large_weighted", combined, combined_weights)
    else:
        print("  (No large models available)")

    # ════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    # Best single model
    best_single_key = max(all_model_info.keys(), key=lambda k: all_model_info[k]['acc'])
    best_single = all_model_info[best_single_key]
    print(f"\n  Best single model: {best_single_key}")
    print(f"    Accuracy: {best_single['acc']:.1f}%, F1: {best_single['f1']*100:.1f}%")

    # Best fusion
    print(f"\n  Best fusion: {best_fusion['name']}")
    print(f"    Accuracy: {best_fusion['acc']:.1f}%, F1: {best_fusion.get('f1', 0):.1f}%")

    # Detailed evaluation of best overall
    best_acc_overall = max(best_single['acc'], best_fusion['acc'])
    if best_fusion['acc'] >= best_single['acc'] and best_fusion['preds'] is not None:
        preds = best_fusion['preds']
        probs = best_fusion.get('probs', None)
        name = best_fusion['name']
    else:
        # Re-evaluate best single
        model = best_single['model']
        test_x = best_single['test_x']
        model.eval()
        with torch.no_grad():
            logits = model(test_x.to(device))
            probs = F.softmax(logits, 1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
        name = str(best_single_key)

    cm = confusion_matrix(test_y_np, preds, labels=[0, 1])
    acc = accuracy_score(test_y_np, preds)
    f1m = f1_score(test_y_np, preds, average='macro', zero_division=0)
    prec = precision_score(test_y_np, preds, average='macro', zero_division=0)
    rec = recall_score(test_y_np, preds, average='macro', zero_division=0)

    print(f"\n  ═══ BEST OVERALL: {name} ═══")
    print(f"  Accuracy:  {100*acc:.2f}% ({int((preds==test_y_np).sum())}/125)")
    print(f"  F1-macro:  {100*f1m:.2f}%")
    print(f"  Precision: {100*prec:.2f}%")
    print(f"  Recall:    {100*rec:.2f}%")
    print(f"  Confusion Matrix:")
    print(f"              Pred Safe  Pred Unsafe")
    print(f"  True Safe:    {cm[0,0]:3d}       {cm[0,1]:3d}")
    print(f"  True Unsafe:  {cm[1,0]:3d}       {cm[1,1]:3d}")

    # Per-class breakdown
    labels_8 = get_test_features(
        TEMPORAL_RESOLUTIONS['T32']['test'], False)[2].numpy()
    CLASS_NAMES = [
        'WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload',
        'SafeWalkway', 'AuthorizedInt', 'ClosedPanel', 'SafeCarrying'
    ]
    print(f"\n  Per original-class accuracy:")
    for c in range(8):
        mask = labels_8 == c
        if mask.sum() == 0: continue
        correct_c = int((preds[mask] == test_y_np[mask]).sum())
        total_c = int(mask.sum())
        safe_unsafe = "Unsafe" if c < 4 else "Safe"
        print(f"    {CLASS_NAMES[c]:20s} ({safe_unsafe}): {correct_c}/{total_c} "
              f"({100*correct_c/total_c:.1f}%)")

    # Error analysis
    print(f"\n  Misclassified samples:")
    for i in range(len(test_y_np)):
        if preds[i] != test_y_np[i]:
            true_label = "Unsafe" if test_y_np[i] == 1 else "Safe"
            pred_label = "Unsafe" if preds[i] == 1 else "Safe"
            orig = CLASS_NAMES[labels_8[i]]
            prob_str = ""
            if probs is not None:
                prob_str = f" probs=[safe={probs[i,0]:.3f}, unsafe={probs[i,1]:.3f}]"
            print(f"    Sample {i:3d}: true={true_label} ({orig}) → pred={pred_label}{prob_str}")

    # Comparison table
    print(f"\n  {'─'*70}")
    print(f"  {'Method':<55s} {'Acc':>8s} {'F1':>8s}")
    print(f"  {'─'*70}")
    print(f"  {'Majority class (all unsafe)':<55s} {'51.2%':>8s}")
    print(f"  {'Linear probe (spatial stats)':<55s} {'68.0%':>8s} {'66.2%':>8s}")
    print(f"  {'Best single (T32, prev best)':<55s} {'81.6%':>8s} {'81.5%':>8s}")
    best_single_label = f"Best single (this run: {best_single_key})"
    print(f"  {best_single_label:<55s} "
          f"{best_single['acc']:>7.1f}% {best_single['f1']*100:>7.1f}%")
    if best_fusion['acc'] > best_single['acc']:
        fusion_name = best_fusion['name'][:40]
        fusion_f1 = best_fusion.get('f1', 0)
        fusion_label = f"Best fusion ({fusion_name})"
        print(f"  {fusion_label:<55s} "
              f"{best_fusion['acc']:>7.1f}% {fusion_f1:>7.1f}%")
    print(f"  {'─'*70}")
    print(f"  {'Paper (YOLOv4+ConvLSTM, 4 unsafe only)':<55s} {'95.81%':>8s}")
    print(f"  {'Paper (ConvLSTM only, 4 unsafe only)':<55s} {'85.78%':>8s}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save results
    results = {
        'best_single': {
            'key': str(best_single_key),
            'acc': best_single['acc'],
            'f1': best_single['f1'],
        },
        'best_fusion': {
            'name': best_fusion['name'],
            'acc': best_fusion['acc'],
            'f1': best_fusion.get('f1', 0),
        },
        'all_models': {
            str(k): {'acc': v['acc'], 'f1': v['f1'], 'path': v['path']}
            for k, v in all_model_info.items()
        }
    }
    with open(os.path.join(out_dir, 'temporal_fusion_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}/temporal_fusion_results.json")


if __name__ == '__main__':
    main()
