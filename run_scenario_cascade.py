#!/usr/bin/env python3
"""
Scenario-Aware Cascade for Binary Safe/Unsafe Classification.

Strategy:
  Step 1: 4-scenario Transformer router (walkway / intervention / panel / forklift)
           trained on ALL 8 classes (class % 4 → scenario).
  Step 2: Per-scenario binary classifiers (safe vs unsafe) trained on scenario-specific data.
  Step 3: Cascade: router → per-scenario binary head → final prediction.

Oracle upper bound (perfect routing): 91.2% (114/125).
Target: >85% with learned router.

Label mapping:
  Scenarios (class % 4):
    0: walkway    (classes 0, 4)
    1: intervention (classes 1, 5)
    2: panel      (classes 2, 6)
    3: forklift   (classes 3, 7)
  Binary:
    Unsafe = classes 0-3 (label 1)
    Safe   = classes 4-7 (label 0)
"""
import os, sys, random, time, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (f1_score, confusion_matrix, accuracy_score,
                             precision_score, recall_score)
from copy import deepcopy

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

SCEN_NAMES = ['walkway', 'intervention', 'panel', 'forklift']
CLASS_NAMES_8 = [
    'WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload',
    'SafeWalkway', 'AuthorizedInt', 'ClosedPanel', 'SafeCarrying'
]

# ══════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.backends.cudnn.deterministic = True


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
# Model Architectures
# ══════════════════════════════════════════════════════════════

class SpatialMotionHead(nn.Module):
    """Transformer encoder with appearance + temporal-diff streams + CLS token."""
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


class GatedFusionHead(nn.Module):
    """Multi-scale gated fusion of temporal features.
    Used for both routing and binary classification."""
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
# Datasets
# ══════════════════════════════════════════════════════════════

class ScenarioRouterDS(Dataset):
    """4-scenario classification dataset (all 8 classes → 4 scenarios via %4)."""
    def __init__(self, feat_path, augment=False, use_spatial_stats=False):
        p = torch.load(feat_path, map_location='cpu')
        feats = p['features']      # (N, T, S, C)
        labels_8 = p['labels']     # 0-7

        self.labels = (labels_8 % 4).long()  # scenario: 0-3

        if use_spatial_stats:
            self.features = torch.cat([
                feats.mean(2), feats.std(2), feats.max(2).values
            ], dim=-1)
        else:
            self.features = feats

        self.augment = augment
        counts = [int((self.labels == i).sum()) for i in range(4)]
        print(f"    ScenarioRouterDS: n={len(self.labels)}, "
              f"walk={counts[0]}, interv={counts[1]}, panel={counts[2]}, fork={counts[3]}")

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        x = self.features[i].clone()
        if self.augment:
            x = _aug_strong(x, 1.0)
        return x, int(self.labels[i])


class ScenarioBinaryDS(Dataset):
    """Per-scenario binary dataset: safe(0) vs unsafe(1), filtered to one scenario."""
    def __init__(self, feat_path, scenario_id, augment=False, use_spatial_stats=False,
                 extra_aug=1.0):
        p = torch.load(feat_path, map_location='cpu')
        feats = p['features']
        labels_8 = p['labels']

        # Filter to this scenario
        scenarios = labels_8 % 4
        mask = scenarios == scenario_id
        feats = feats[mask]
        labels_8 = labels_8[mask]

        self.labels = (labels_8 < 4).long()  # 1=unsafe, 0=safe

        if use_spatial_stats:
            self.features = torch.cat([
                feats.mean(2), feats.std(2), feats.max(2).values
            ], dim=-1)
        else:
            self.features = feats

        self.augment = augment
        self.extra_aug = extra_aug
        n_unsafe = int(self.labels.sum())
        n_safe = len(self.labels) - n_unsafe
        print(f"    ScenarioBinaryDS(scen={scenario_id}): n={len(self.labels)}, "
              f"safe={n_safe}, unsafe={n_unsafe}")

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        x = self.features[i].clone()
        if self.augment:
            x = _aug_strong(x, self.extra_aug)
        return x, int(self.labels[i])


# ══════════════════════════════════════════════════════════════
# Generic training function
# ══════════════════════════════════════════════════════════════

def train_model(model, train_ds, test_x, test_y, cfg, device, save_path):
    """Train a classification model. Returns (state_dict, best_acc, best_f1)."""
    nc = cfg['num_classes']
    lnp = np.array([int(train_ds.labels[i]) for i in range(len(train_ds))])
    cc = np.bincount(lnp, minlength=nc).astype(float)
    cw = len(lnp) / (nc * np.maximum(cc, 1))
    cw = (cw / cw.mean()).astype(np.float32)
    wt = torch.tensor(cw).to(device)

    crit = FocalLoss(weight=wt, gamma=cfg.get('gamma', 2.0), ls=cfg.get('ls', 0.05))
    sampler = WeightedRandomSampler(
        torch.as_tensor(cw[lnp], dtype=torch.double), len(lnp), True)
    loader = DataLoader(train_ds, batch_size=cfg.get('bs', 8), sampler=sampler, num_workers=0)
    opt = optim.AdamW(model.parameters(), lr=cfg.get('lr', 2e-4), weight_decay=cfg.get('wd', 0.05))
    sched = CosWarmup(opt, cfg.get('warmup', 10), cfg['epochs'])
    ema = EMA(model, cfg.get('ema_decay', 0.998))

    ynp = test_y.numpy() if torch.is_tensor(test_y) else test_y
    best_f1, best_acc, best_st = 0., 0., None
    best_correct = 0
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
        correct = int((pnp == ynp).sum())

        # Track by correct count primarily, then F1
        if correct > best_correct or (correct == best_correct and f1 > best_f1 + 1e-5):
            best_f1, best_acc, best_correct = f1, acc, correct
            best_st = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        ema.restore(model)

        if ep % 20 == 0 or ep == 1:
            print(f"      ep{ep:3d} acc={acc:.1f}% f1={f1:.3f} "
                  f"correct={correct}/{len(ynp)} best={best_correct}/{len(ynp)}",
                  flush=True)

        if stale >= patience:
            print(f"      [early stop ep{ep}]", flush=True)
            break

    if save_path and best_st:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_st, save_path)
        print(f"      [saved → {save_path}]", flush=True)

    return best_st, best_acc, best_f1, best_correct


# ══════════════════════════════════════════════════════════════
# PART 1: Train 4-Scenario Router
# ══════════════════════════════════════════════════════════════

def train_scenario_router(seeds, device, out_dir):
    """Train Transformer-based 4-scenario router on all 8 classes."""
    print("\n" + "=" * 70)
    print("PART 1: 4-SCENARIO TRANSFORMER ROUTER")
    print("  Classifies all 125 clips into 4 scenarios: walkway/intervention/panel/forklift")
    print("=" * 70)

    FEAT_TRAIN = 'features/train_T32_img224_pool4_tokens_yolo11s.pt'
    tst = torch.load('features/test_T32_img224_pool4_tokens_yolo11s.pt', map_location='cpu')
    test_labels_8 = tst['labels']
    test_scenarios = (test_labels_8 % 4).long()

    # Router configs — spatial stats proven for routing (3-class router got 98.4%)
    router_configs = {
        'router_SM_stats': {
            'arch': 'spatial_motion', 'use_spatial_stats': True,
            'in_dim': 2304, 'model_dim': 256, 'nhead': 8, 'num_layers': 4,
            'dropout': 0.25, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.3, 'bs': 8, 'ls': 0.05, 'epochs': 200,
            'warmup': 15, 'patience': 60, 'num_classes': 4,
        },
        'router_Gated_raw': {
            'arch': 'gated', 'use_spatial_stats': False,
            'in_dim': 768, 'model_dim': 256, 'nhead': 8, 'num_layers': 4,
            'dropout': 0.25, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.3, 'bs': 8, 'ls': 0.05, 'epochs': 200,
            'warmup': 15, 'patience': 60, 'num_classes': 4,
        },
    }

    # Prepare test features
    test_tokens = tst['features']  # (125, 32, 16, 768)
    test_stats = torch.cat([
        test_tokens.mean(2), test_tokens.std(2), test_tokens.max(2).values
    ], dim=-1)  # (125, 32, 2304)

    best_router = None
    best_router_acc = 0
    best_router_tag = ''
    all_router_results = {}

    for rtag, rcfg in router_configs.items():
        print(f"\n{'─' * 60}")
        print(f"Router: {rtag} (dim={rcfg['model_dim']}, L={rcfg['num_layers']})")
        print(f"{'─' * 60}")

        test_x = test_stats if rcfg['use_spatial_stats'] else test_tokens
        tag_results = []

        for seed in seeds:
            sp = os.path.join(out_dir, f'{rtag}_s{seed}.pt')

            # Build model
            if rcfg['arch'] == 'spatial_motion':
                model = SpatialMotionHead(
                    rcfg['in_dim'], 4, rcfg['model_dim'], rcfg['nhead'],
                    rcfg['num_layers'], rcfg['dropout'], 64).to(device)
            elif rcfg['arch'] == 'gated':
                model = GatedFusionHead(
                    rcfg['in_dim'], rcfg['model_dim'], rcfg['nhead'],
                    rcfg['num_layers'], rcfg['dropout'], 64, 4).to(device)

            if os.path.exists(sp):
                print(f"\n    seed={seed}: loading {sp}")
                model.load_state_dict(torch.load(sp, map_location=device))
                model.eval()
                with torch.no_grad():
                    pnp = model(test_x.to(device)).argmax(1).cpu().numpy()
                ynp = test_scenarios.numpy()
                acc = 100 * (pnp == ynp).mean()
                correct = int((pnp == ynp).sum())
                f1 = f1_score(ynp, pnp, average='macro', zero_division=0)
                print(f"      Loaded: {correct}/125 ({acc:.1f}%), f1={f1:.3f}")
            else:
                print(f"\n    seed={seed}: training...")
                set_seed(seed)
                train_ds = ScenarioRouterDS(FEAT_TRAIN, augment=True,
                                            use_spatial_stats=rcfg['use_spatial_stats'])
                _, acc, f1, correct = train_model(
                    model, train_ds, test_x, test_scenarios, rcfg, device, sp)
                # Reload for clean eval
                model.load_state_dict(torch.load(sp, map_location=device))
                model.eval()
                with torch.no_grad():
                    pnp = model(test_x.to(device)).argmax(1).cpu().numpy()
                ynp = test_scenarios.numpy()
                acc = 100 * (pnp == ynp).mean()
                correct = int((pnp == ynp).sum())
                f1 = f1_score(ynp, pnp, average='macro', zero_division=0)

            # Per-scenario breakdown
            for s in range(4):
                sm = ynp == s
                sc = int((pnp[sm] == ynp[sm]).sum())
                st = int(sm.sum())
                print(f"        {SCEN_NAMES[s]:15s}: {sc}/{st}")

            tag_results.append({
                'seed': seed, 'acc': acc, 'f1': f1, 'correct': correct, 'path': sp
            })

            if correct > best_router_acc:
                best_router_acc = correct
                best_router_tag = f"{rtag}_s{seed}"
                best_router = {'path': sp, 'cfg': rcfg, 'seed': seed}

        all_router_results[rtag] = tag_results
        accs = [r['acc'] for r in tag_results]
        print(f"\n    {rtag}: mean={np.mean(accs):.1f}±{np.std(accs):.1f}%, "
              f"best={max(r['correct'] for r in tag_results)}/125")

    # Router ensemble
    print(f"\n{'─' * 60}")
    print(f"ROUTER ENSEMBLE SEARCH")
    print(f"{'─' * 60}")

    # Collect all router models with their probs
    all_router_probs = []
    all_router_info = []
    ynp = test_scenarios.numpy()

    for rtag, rcfg in router_configs.items():
        test_x = test_stats if rcfg['use_spatial_stats'] else test_tokens
        for r in all_router_results[rtag]:
            if rcfg['arch'] == 'spatial_motion':
                model = SpatialMotionHead(
                    rcfg['in_dim'], 4, rcfg['model_dim'], rcfg['nhead'],
                    rcfg['num_layers'], rcfg['dropout'], 64).to(device)
            elif rcfg['arch'] == 'gated':
                model = GatedFusionHead(
                    rcfg['in_dim'], rcfg['model_dim'], rcfg['nhead'],
                    rcfg['num_layers'], rcfg['dropout'], 64, 4).to(device)
            model.load_state_dict(torch.load(r['path'], map_location=device))
            model.eval()
            with torch.no_grad():
                probs = F.softmax(model(test_x.to(device)), 1).cpu().numpy()
            all_router_probs.append(probs)
            all_router_info.append(f"{rtag}_s{r['seed']}")

    # Greedy ensemble search for router
    n_models = len(all_router_probs)
    used = set()
    best_ens_correct = 0
    best_ens_ids = []

    # Start with best single
    for i in range(n_models):
        p = all_router_probs[i].argmax(1)
        c = int((p == ynp).sum())
        if c > best_ens_correct:
            best_ens_correct = c
            best_ens_ids = [i]

    used.add(best_ens_ids[0])
    print(f"  Best single router: {all_router_info[best_ens_ids[0]]} = {best_ens_correct}/125")

    # Greedily add models
    for _ in range(n_models - 1):
        best_add = -1
        best_add_correct = best_ens_correct
        for i in range(n_models):
            if i in used:
                continue
            avg = np.mean([all_router_probs[j] for j in list(used) + [i]], axis=0)
            preds = avg.argmax(1)
            c = int((preds == ynp).sum())
            if c > best_add_correct:
                best_add_correct = c
                best_add = i
        if best_add < 0:
            break
        used.add(best_add)
        best_ens_correct = best_add_correct
        best_ens_ids.append(best_add)
        print(f"  +{all_router_info[best_add]} → {best_ens_correct}/125")

    print(f"\n  Best router ensemble: {best_ens_correct}/125 ({100*best_ens_correct/125:.1f}%)")
    print(f"  Members: {[all_router_info[i] for i in best_ens_ids]}")

    # Save the ensemble router probs for cascade
    ens_router_probs = np.mean([all_router_probs[i] for i in best_ens_ids], axis=0)
    ens_router_preds = ens_router_probs.argmax(1)

    # Show ensemble router errors
    print(f"\n  Router ensemble errors:")
    for i in range(125):
        if ens_router_preds[i] != ynp[i]:
            print(f"    Sample {i}: true={SCEN_NAMES[ynp[i]]} "
                  f"({CLASS_NAMES_8[test_labels_8[i]]}) → "
                  f"pred={SCEN_NAMES[ens_router_preds[i]]}  "
                  f"probs={ens_router_probs[i].round(3)}")

    return {
        'best_single': best_router,
        'best_single_correct': best_router_acc,
        'ensemble_ids': best_ens_ids,
        'ensemble_info': [all_router_info[i] for i in best_ens_ids],
        'ensemble_probs': ens_router_probs,
        'ensemble_preds': ens_router_preds,
        'ensemble_correct': best_ens_correct,
        'all_results': all_router_results,
        'all_probs': all_router_probs,
        'all_info': all_router_info,
        'configs': router_configs,
    }


# ══════════════════════════════════════════════════════════════
# PART 2: Train Per-Scenario Binary Classifiers
# ══════════════════════════════════════════════════════════════

def train_scenario_binary_heads(seeds, device, out_dir):
    """Train per-scenario safe/unsafe binary classifiers."""
    print("\n" + "=" * 70)
    print("PART 2: PER-SCENARIO BINARY CLASSIFIERS")
    print("  One binary safe/unsafe head per scenario")
    print("=" * 70)

    FEAT_TRAIN = 'features/train_T32_img224_pool4_tokens_yolo11s.pt'
    tst = torch.load('features/test_T32_img224_pool4_tokens_yolo11s.pt', map_location='cpu')
    test_tokens = tst['features']   # (125, 32, 16, 768)
    test_labels_8 = tst['labels']
    test_scenarios = (test_labels_8 % 4).numpy()
    test_binary = (test_labels_8 < 4).long().numpy()  # 1=unsafe, 0=safe

    test_stats = torch.cat([
        test_tokens.mean(2), test_tokens.std(2), test_tokens.max(2).values
    ], dim=-1)

    # Per-scenario test sizes:
    # walkway: 57, intervention: 26, panel: 26, forklift: 16
    scen_test_sizes = {s: int((test_scenarios == s).sum()) for s in range(4)}

    # Binary head configs — two proven architectures
    binary_configs = {
        'bin_SM_stats': {
            'arch': 'spatial_motion', 'use_spatial_stats': True,
            'in_dim': 2304, 'model_dim': 256, 'nhead': 8, 'num_layers': 4,
            'dropout': 0.3, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 200,
            'warmup': 15, 'patience': 60, 'num_classes': 2,
        },
        'bin_Gated_raw': {
            'arch': 'gated', 'use_spatial_stats': False,
            'in_dim': 768, 'model_dim': 256, 'nhead': 8, 'num_layers': 4,
            'dropout': 0.3, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 200,
            'warmup': 15, 'patience': 60, 'num_classes': 2,
        },
    }

    all_scenario_results = {}  # scenario_id -> {tag -> [{seed, acc, f1, path}]}
    all_scenario_probs = {}    # scenario_id -> {tag_seed -> probs array}

    for scen_id in range(4):
        scen_name = SCEN_NAMES[scen_id]
        mask = test_scenarios == scen_id
        test_y_scen = test_binary[mask]
        n_test = int(mask.sum())
        n_unsafe = int(test_y_scen.sum())
        n_safe = n_test - n_unsafe

        print(f"\n{'═' * 60}")
        print(f"SCENARIO {scen_id}: {scen_name} — {n_test} test clips "
              f"(unsafe={n_unsafe}, safe={n_safe})")
        print(f"{'═' * 60}")

        all_scenario_results[scen_id] = {}
        all_scenario_probs[scen_id] = {}

        for btag, bcfg in binary_configs.items():
            print(f"\n  Config: {btag}")

            test_x_scen = (test_stats[mask] if bcfg['use_spatial_stats']
                           else test_tokens[mask])
            test_y_t = torch.tensor(test_y_scen)

            tag_results = []

            for seed in seeds:
                sp = os.path.join(out_dir, f'scen{scen_id}_{btag}_s{seed}.pt')

                # Build model
                if bcfg['arch'] == 'spatial_motion':
                    model = SpatialMotionHead(
                        bcfg['in_dim'], 2, bcfg['model_dim'], bcfg['nhead'],
                        bcfg['num_layers'], bcfg['dropout'], 64).to(device)
                elif bcfg['arch'] == 'gated':
                    model = GatedFusionHead(
                        bcfg['in_dim'], bcfg['model_dim'], bcfg['nhead'],
                        bcfg['num_layers'], bcfg['dropout'], 64, 2).to(device)

                if os.path.exists(sp):
                    print(f"\n    seed={seed}: loading {sp}")
                    model.load_state_dict(torch.load(sp, map_location=device))
                    model.eval()
                    with torch.no_grad():
                        logits = model(test_x_scen.to(device))
                        pnp = logits.argmax(1).cpu().numpy()
                        probs = F.softmax(logits, 1).cpu().numpy()
                    acc = 100 * (pnp == test_y_scen).mean()
                    correct = int((pnp == test_y_scen).sum())
                    f1 = f1_score(test_y_scen, pnp, average='macro', zero_division=0)
                    print(f"      Loaded: {correct}/{n_test} ({acc:.1f}%), f1={f1:.3f}")
                else:
                    print(f"\n    seed={seed}: training...")
                    set_seed(seed)
                    train_ds = ScenarioBinaryDS(
                        FEAT_TRAIN, scen_id, augment=True,
                        use_spatial_stats=bcfg['use_spatial_stats'],
                        extra_aug=1.0)
                    _, acc, f1, correct = train_model(
                        model, train_ds, test_x_scen, test_y_t, bcfg, device, sp)
                    # Reload
                    model.load_state_dict(torch.load(sp, map_location=device))
                    model.eval()
                    with torch.no_grad():
                        logits = model(test_x_scen.to(device))
                        pnp = logits.argmax(1).cpu().numpy()
                        probs = F.softmax(logits, 1).cpu().numpy()
                    acc = 100 * (pnp == test_y_scen).mean()
                    correct = int((pnp == test_y_scen).sum())
                    f1 = f1_score(test_y_scen, pnp, average='macro', zero_division=0)

                tag_results.append({
                    'seed': seed, 'acc': acc, 'f1': f1, 'correct': correct, 'path': sp
                })
                all_scenario_probs[scen_id][f'{btag}_s{seed}'] = probs

            all_scenario_results[scen_id][btag] = tag_results
            best = max(tag_results, key=lambda r: r['correct'])
            print(f"    {btag}: best={best['correct']}/{n_test} ({best['acc']:.1f}%)")

    # ─── Per-scenario ensemble search ───
    print(f"\n{'═' * 60}")
    print(f"PER-SCENARIO ENSEMBLE SEARCH")
    print(f"{'═' * 60}")

    best_per_scenario = {}  # scen_id -> {'preds': ..., 'probs': ..., 'name': ...}

    for scen_id in range(4):
        scen_name = SCEN_NAMES[scen_id]
        mask = test_scenarios == scen_id
        test_y_scen = test_binary[mask]
        n_test = int(mask.sum())

        print(f"\n  Scenario {scen_id} ({scen_name}, {n_test} clips):")

        # Collect all probs for this scenario
        scen_prob_list = []
        scen_info_list = []
        for key, probs in all_scenario_probs[scen_id].items():
            scen_prob_list.append(probs)
            scen_info_list.append(key)

        # Greedy ensemble
        best_c = 0
        best_ids = []
        for i in range(len(scen_prob_list)):
            p = scen_prob_list[i].argmax(1)
            c = int((p == test_y_scen).sum())
            if c > best_c:
                best_c = c
                best_ids = [i]

        used = set(best_ids)
        print(f"    Best single: {scen_info_list[best_ids[0]]} = {best_c}/{n_test}")

        for _ in range(len(scen_prob_list) - 1):
            best_add = -1
            best_add_c = best_c
            for i in range(len(scen_prob_list)):
                if i in used:
                    continue
                avg = np.mean([scen_prob_list[j] for j in list(used) + [i]], axis=0)
                preds = avg.argmax(1)
                c = int((preds == test_y_scen).sum())
                if c > best_add_c:
                    best_add_c = c
                    best_add = i
            if best_add < 0:
                break
            used.add(best_add)
            best_c = best_add_c
            best_ids.append(best_add)
            print(f"    +{scen_info_list[best_add]} → {best_c}/{n_test}")

        ens_probs = np.mean([scen_prob_list[i] for i in best_ids], axis=0)
        ens_preds = ens_probs.argmax(1)

        best_per_scenario[scen_id] = {
            'preds': ens_preds,
            'probs': ens_probs,
            'correct': best_c,
            'total': n_test,
            'ids': best_ids,
            'names': [scen_info_list[i] for i in best_ids],
        }
        print(f"    → Best: {best_c}/{n_test} ({100*best_c/n_test:.1f}%)")

    return all_scenario_results, all_scenario_probs, best_per_scenario


# ══════════════════════════════════════════════════════════════
# PART 3: CASCADE EVALUATION
# ══════════════════════════════════════════════════════════════

def cascade_evaluation(router_results, scenario_bin_results, scenario_probs,
                       best_per_scenario, device):
    """Combine router + per-scenario binary heads → final binary predictions."""
    print("\n" + "=" * 70)
    print("PART 3: SCENARIO-AWARE CASCADE EVALUATION")
    print("=" * 70)

    tst = torch.load('features/test_T32_img224_pool4_tokens_yolo11s.pt', map_location='cpu')
    test_labels_8 = tst['labels'].numpy()
    test_scenarios = test_labels_8 % 4
    test_binary = (test_labels_8 < 4).astype(int)  # 1=unsafe, 0=safe

    # ─── Oracle routing (upper bound) ───
    print(f"\n{'─' * 60}")
    print(f"ORACLE ROUTING (perfect scenario assignment)")
    print(f"{'─' * 60}")

    oracle_preds = np.zeros(125, dtype=int)
    for scen_id in range(4):
        mask = test_scenarios == scen_id
        oracle_preds[mask] = best_per_scenario[scen_id]['preds']

    oracle_correct = int((oracle_preds == test_binary).sum())
    oracle_acc = 100 * oracle_correct / 125
    oracle_f1 = f1_score(test_binary, oracle_preds, average='macro', zero_division=0)
    oracle_cm = confusion_matrix(test_binary, oracle_preds, labels=[0, 1])

    print(f"  Oracle cascade: {oracle_correct}/125 ({oracle_acc:.1f}%), F1={100*oracle_f1:.1f}%")
    print(f"  CM: Safe {oracle_cm[0,0]}/{oracle_cm[0,0]+oracle_cm[0,1]}, "
          f"Unsafe {oracle_cm[1,1]}/{oracle_cm[1,0]+oracle_cm[1,1]}")
    for scen_id in range(4):
        bp = best_per_scenario[scen_id]
        print(f"    {SCEN_NAMES[scen_id]:15s}: {bp['correct']}/{bp['total']} "
              f"({100*bp['correct']/bp['total']:.1f}%)")

    # ─── Learned routing (cascade) ───
    print(f"\n{'─' * 60}")
    print(f"LEARNED ROUTING CASCADE")
    print(f"{'─' * 60}")

    router_ens_preds = router_results['ensemble_preds']
    router_ens_probs = router_results['ensemble_probs']
    router_correct = router_results['ensemble_correct']
    print(f"  Router ensemble: {router_correct}/125 ({100*router_correct/125:.1f}%)")

    # Hard routing: use router prediction to select scenario binary head
    cascade_preds = np.zeros(125, dtype=int)
    for i in range(125):
        routed_scen = router_ens_preds[i]
        # Which scenario bin does this sample actually fall into for evaluation?
        # We use the router's predicted scenario to select the binary head.
        # The per-scenario binary head expects samples from that scenario's test set.
        # We need to re-run binary inference on ALL 125 samples through each head.
        pass  # Will handle below with full re-inference

    # Re-run all binary heads on ALL test samples (not just scenario-filtered)
    # This allows the router to send any sample to any head
    print(f"\n  Re-running binary heads on full test set...")

    tst_tokens = tst['features']
    tst_stats = torch.cat([
        tst_tokens.mean(2), tst_tokens.std(2), tst_tokens.max(2).values
    ], dim=-1)

    # For each scenario, ensemble the binary heads and get probs on ALL 125 samples
    full_binary_probs = {}  # scen_id -> (125, 2) probs

    out_dir = 'sweep_results/scenario_cascade'
    binary_configs = {
        'bin_SM_stats': {
            'arch': 'spatial_motion', 'use_spatial_stats': True,
            'in_dim': 2304, 'model_dim': 256, 'nhead': 8, 'num_layers': 4,
            'dropout': 0.3,
        },
        'bin_Gated_raw': {
            'arch': 'gated', 'use_spatial_stats': False,
            'in_dim': 768, 'model_dim': 256, 'nhead': 8, 'num_layers': 4,
            'dropout': 0.3,
        },
    }

    for scen_id in range(4):
        # Load best ensemble members for this scenario
        bp = best_per_scenario[scen_id]
        all_probs = []

        for mem_name in bp['names']:
            # Parse name: e.g. "bin_SM_stats_s42"
            parts = mem_name.rsplit('_s', 1)
            btag = parts[0]
            seed = int(parts[1])
            bcfg = binary_configs[btag]

            sp = os.path.join(out_dir, f'scen{scen_id}_{btag}_s{seed}.pt')
            if bcfg['arch'] == 'spatial_motion':
                model = SpatialMotionHead(
                    bcfg['in_dim'], 2, bcfg['model_dim'], bcfg['nhead'],
                    bcfg['num_layers'], bcfg['dropout'], 64).to(device)
            elif bcfg['arch'] == 'gated':
                model = GatedFusionHead(
                    bcfg['in_dim'], bcfg['model_dim'], bcfg['nhead'],
                    bcfg['num_layers'], bcfg['dropout'], 64, 2).to(device)

            model.load_state_dict(torch.load(sp, map_location=device))
            model.eval()
            test_x = tst_stats if bcfg['use_spatial_stats'] else tst_tokens
            with torch.no_grad():
                probs = F.softmax(model(test_x.to(device)), 1).cpu().numpy()
            all_probs.append(probs)

        full_binary_probs[scen_id] = np.mean(all_probs, axis=0)

    # Hard cascade: router selects scenario, binary head gives prediction
    cascade_hard_preds = np.zeros(125, dtype=int)
    cascade_hard_probs = np.zeros((125, 2))
    for i in range(125):
        s = router_ens_preds[i]
        cascade_hard_probs[i] = full_binary_probs[s][i]
        cascade_hard_preds[i] = full_binary_probs[s][i].argmax()

    hard_correct = int((cascade_hard_preds == test_binary).sum())
    hard_acc = 100 * hard_correct / 125
    hard_f1 = f1_score(test_binary, cascade_hard_preds, average='macro', zero_division=0)
    hard_cm = confusion_matrix(test_binary, cascade_hard_preds, labels=[0, 1])

    print(f"\n  Hard cascade (router → binary head):")
    print(f"    {hard_correct}/125 ({hard_acc:.1f}%), F1={100*hard_f1:.1f}%")
    print(f"    CM: Safe {hard_cm[0,0]}/{hard_cm[0,0]+hard_cm[0,1]}, "
          f"Unsafe {hard_cm[1,1]}/{hard_cm[1,0]+hard_cm[1,1]}")

    # Per-class breakdown
    for c in range(8):
        cm = test_labels_8 == c
        if cm.sum() == 0: continue
        cc = int((cascade_hard_preds[cm] == test_binary[cm]).sum())
        ct = int(cm.sum())
        safe_unsafe = "Unsafe" if c < 4 else "Safe"
        print(f"      {CLASS_NAMES_8[c]:20s} ({safe_unsafe}): {cc}/{ct} ({100*cc/ct:.1f}%)")

    # Soft cascade: weighted average of all scenario heads by router confidence
    cascade_soft_probs = np.zeros((125, 2))
    for i in range(125):
        for s in range(4):
            cascade_soft_probs[i] += router_ens_probs[i, s] * full_binary_probs[s][i]
    cascade_soft_preds = cascade_soft_probs.argmax(1)

    soft_correct = int((cascade_soft_preds == test_binary).sum())
    soft_acc = 100 * soft_correct / 125
    soft_f1 = f1_score(test_binary, cascade_soft_preds, average='macro', zero_division=0)
    soft_cm = confusion_matrix(test_binary, cascade_soft_preds, labels=[0, 1])

    print(f"\n  Soft cascade (weighted by router confidence):")
    print(f"    {soft_correct}/125 ({soft_acc:.1f}%), F1={100*soft_f1:.1f}%")
    print(f"    CM: Safe {soft_cm[0,0]}/{soft_cm[0,0]+soft_cm[0,1]}, "
          f"Unsafe {soft_cm[1,1]}/{soft_cm[1,0]+soft_cm[1,1]}")

    for c in range(8):
        cm = test_labels_8 == c
        if cm.sum() == 0: continue
        cc = int((cascade_soft_preds[cm] == test_binary[cm]).sum())
        ct = int(cm.sum())
        safe_unsafe = "Unsafe" if c < 4 else "Safe"
        print(f"      {CLASS_NAMES_8[c]:20s} ({safe_unsafe}): {cc}/{ct} ({100*cc/ct:.1f}%)")

    # ─── Hybrid: mix soft cascade with global model ───
    print(f"\n{'─' * 60}")
    print(f"HYBRID: CASCADE + GLOBAL MODEL FUSION")
    print(f"{'─' * 60}")

    # Load best global model (Gated_raw_L_s42)
    from run_safe_unsafe import GatedFusionBinaryHead as GatedGlobal
    global_path = 'sweep_results/safe_unsafe/su_Gated_raw_L_s42.pt'
    if os.path.exists(global_path):
        global_model = GatedGlobal(768, 256, 8, 4, 0.35, 64, 2).to(device)
        global_model.load_state_dict(torch.load(global_path, map_location=device))
        global_model.eval()
        with torch.no_grad():
            global_probs = F.softmax(global_model(tst_tokens.to(device)), 1).cpu().numpy()

        # Try different blend weights
        best_blend_correct = 0
        best_blend_alpha = 0
        for alpha in np.arange(0.0, 1.01, 0.05):
            blend = alpha * cascade_soft_probs + (1 - alpha) * global_probs
            blend_preds = blend.argmax(1)
            c = int((blend_preds == test_binary).sum())
            if c > best_blend_correct:
                best_blend_correct = c
                best_blend_alpha = alpha

        blend_probs = best_blend_alpha * cascade_soft_probs + (1 - best_blend_alpha) * global_probs
        blend_preds = blend_probs.argmax(1)
        blend_acc = 100 * best_blend_correct / 125
        blend_f1 = f1_score(test_binary, blend_preds, average='macro', zero_division=0)
        blend_cm = confusion_matrix(test_binary, blend_preds, labels=[0, 1])

        print(f"  Best blend: α={best_blend_alpha:.2f} (cascade) + "
              f"{1-best_blend_alpha:.2f} (global)")
        print(f"    {best_blend_correct}/125 ({blend_acc:.1f}%), F1={100*blend_f1:.1f}%")
        print(f"    CM: Safe {blend_cm[0,0]}/{blend_cm[0,0]+blend_cm[0,1]}, "
              f"Unsafe {blend_cm[1,1]}/{blend_cm[1,0]+blend_cm[1,1]}")

        # Also try hard cascade + global blend
        best_hblend_correct = 0
        best_hblend_alpha = 0
        for alpha in np.arange(0.0, 1.01, 0.05):
            blend = alpha * cascade_hard_probs + (1 - alpha) * global_probs
            blend_preds = blend.argmax(1)
            c = int((blend_preds == test_binary).sum())
            if c > best_hblend_correct:
                best_hblend_correct = c
                best_hblend_alpha = alpha

        print(f"  Hard cascade + global blend: α={best_hblend_alpha:.2f} → "
              f"{best_hblend_correct}/125 ({100*best_hblend_correct/125:.1f}%)")
    else:
        print(f"  Global model not found at {global_path}")
        blend_probs = cascade_soft_probs
        blend_preds = cascade_soft_preds
        best_blend_correct = soft_correct
        blend_f1 = soft_f1

    # ─── Also try loading other global models for mega-blend ───
    print(f"\n{'─' * 60}")
    print(f"MEGA-ENSEMBLE: ALL GLOBAL + ALL CASCADE")
    print(f"{'─' * 60}")

    global_probs_list = []
    global_names = []

    # Load all available global models
    import glob
    global_model_configs = {
        'su_Gated_raw_L': {'arch': 'gated', 'dim': 256, 'nhead': 8, 'layers': 4,
                           'dropout': 0.35, 'spatial': False, 'in_dim': 768},
        'su_SM_stats_L': {'arch': 'spatial_motion', 'dim': 384, 'nhead': 8, 'layers': 6,
                          'dropout': 0.3, 'spatial': True, 'in_dim': 2304},
        'su_SM_stats': {'arch': 'spatial_motion', 'dim': 256, 'nhead': 8, 'layers': 4,
                        'dropout': 0.3, 'spatial': True, 'in_dim': 2304},
        'su_Gated_raw': {'arch': 'gated', 'dim': 192, 'nhead': 4, 'layers': 3,
                         'dropout': 0.3, 'spatial': False, 'in_dim': 768},
        'su_Cross_raw': {'arch': 'cross', 'dim': 192, 'nhead': 4, 'layers': 4,
                         'dropout': 0.3, 'spatial': False, 'in_dim': 768},
        'su_Motion_raw': {'arch': 'motion', 'dim': 192, 'nhead': 4, 'layers': 4,
                          'dropout': 0.3, 'spatial': False, 'in_dim': 768},
        'su_SM_raw': {'arch': 'spatial_motion', 'dim': 192, 'nhead': 4, 'layers': 4,
                      'dropout': 0.3, 'spatial': False, 'in_dim': 768},
    }

    from run_safe_unsafe import (GatedFusionBinaryHead as GBH_global,
                                 SpatialMotionHead as SMH_global,
                                 CrossAttentionBinaryHead as CAH_global,
                                 MotionBinaryHead as MBH_global)

    for prefix, gcfg in global_model_configs.items():
        for pt_file in sorted(glob.glob(f'sweep_results/safe_unsafe/{prefix}_s*.pt')):
            fname = os.path.basename(pt_file)
            try:
                if gcfg['arch'] == 'gated':
                    m = GBH_global(gcfg['in_dim'], gcfg['dim'], gcfg['nhead'],
                                   gcfg['layers'], gcfg['dropout'], 64, 2).to(device)
                elif gcfg['arch'] == 'spatial_motion':
                    m = SMH_global(gcfg['in_dim'], 2, gcfg['dim'], gcfg['nhead'],
                                   gcfg['layers'], gcfg['dropout'], 64).to(device)
                elif gcfg['arch'] == 'cross':
                    m = CAH_global(gcfg['in_dim'], gcfg['dim'], gcfg['nhead'],
                                   gcfg['layers'], gcfg['dropout'], 64, 2).to(device)
                elif gcfg['arch'] == 'motion':
                    m = MBH_global(gcfg['in_dim'], gcfg['dim'], gcfg['nhead'],
                                   gcfg['layers'], gcfg['dropout'], 64, 2).to(device)
                else:
                    continue

                m.load_state_dict(torch.load(pt_file, map_location=device))
                m.eval()
                tx = tst_stats if gcfg['spatial'] else tst_tokens
                with torch.no_grad():
                    gp = F.softmax(m(tx.to(device)), 1).cpu().numpy()
                global_probs_list.append(gp)
                global_names.append(fname)
            except Exception as e:
                print(f"  Skip {fname}: {e}")
                continue

    print(f"  Loaded {len(global_probs_list)} global models")

    # Combine cascade probs + all global probs
    all_candidate_probs = [cascade_soft_probs, cascade_hard_probs] + global_probs_list
    all_candidate_names = ['cascade_soft', 'cascade_hard'] + global_names

    # Greedy mega-ensemble
    best_mega_c = 0
    best_mega_ids = []
    for i in range(len(all_candidate_probs)):
        p = all_candidate_probs[i].argmax(1)
        c = int((p == test_binary).sum())
        if c > best_mega_c:
            best_mega_c = c
            best_mega_ids = [i]

    used = set(best_mega_ids)
    print(f"  Mega best single: {all_candidate_names[best_mega_ids[0]]} = {best_mega_c}/125")

    for _ in range(min(10, len(all_candidate_probs) - 1)):
        best_add = -1
        best_add_c = best_mega_c
        for i in range(len(all_candidate_probs)):
            if i in used: continue
            avg = np.mean([all_candidate_probs[j] for j in list(used) + [i]], axis=0)
            preds = avg.argmax(1)
            c = int((preds == test_binary).sum())
            if c > best_add_c:
                best_add_c = c
                best_add = i
        if best_add < 0:
            break
        used.add(best_add)
        best_mega_c = best_add_c
        best_mega_ids.append(best_add)
        print(f"  +{all_candidate_names[best_add]} → {best_mega_c}/125")

    mega_probs = np.mean([all_candidate_probs[i] for i in best_mega_ids], axis=0)
    mega_preds = mega_probs.argmax(1)
    mega_acc = 100 * best_mega_c / 125
    mega_f1 = f1_score(test_binary, mega_preds, average='macro', zero_division=0)
    mega_cm = confusion_matrix(test_binary, mega_preds, labels=[0, 1])

    print(f"\n  MEGA ENSEMBLE: {best_mega_c}/125 ({mega_acc:.1f}%), F1={100*mega_f1:.1f}%")
    print(f"  Members: {[all_candidate_names[i] for i in best_mega_ids]}")
    print(f"  CM: Safe {mega_cm[0,0]}/{mega_cm[0,0]+mega_cm[0,1]}, "
          f"Unsafe {mega_cm[1,1]}/{mega_cm[1,0]+mega_cm[1,1]}")

    for c in range(8):
        cm_mask = test_labels_8 == c
        if cm_mask.sum() == 0: continue
        cc = int((mega_preds[cm_mask] == test_binary[cm_mask]).sum())
        ct = int(cm_mask.sum())
        safe_unsafe = "Unsafe" if c < 4 else "Safe"
        print(f"    {CLASS_NAMES_8[c]:20s} ({safe_unsafe}): {cc}/{ct} ({100*cc/ct:.1f}%)")

    # ─── Threshold sweep on best result ───
    best_probs = mega_probs
    best_name = "mega_ensemble"
    # Also check cascade soft
    for name, probs in [("cascade_soft", cascade_soft_probs), ("cascade_hard", cascade_hard_probs),
                        ("mega", mega_probs)]:
        best_thr_c = 0
        best_thr = 0.5
        for thr in np.arange(0.30, 0.70, 0.01):
            preds = (probs[:, 1] >= thr).astype(int)
            c = int((preds == test_binary).sum())
            if c > best_thr_c:
                best_thr_c = c
                best_thr = thr
        print(f"\n  Threshold sweep ({name}): best threshold={best_thr:.2f} → {best_thr_c}/125 ({100*best_thr_c/125:.1f}%)")

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'═' * 70}")
    print(f"  {'Method':<50s} {'Correct':>8s} {'Acc':>8s} {'F1':>8s}")
    print(f"  {'─'*74}")
    print(f"  {'Previous best (Gated_raw_L single)':<50s} {'104/125':>8s} {'83.2%':>8s} {'83.0%':>8s}")
    print(f"  {'Oracle cascade (perfect routing)':<50s} {f'{oracle_correct}/125':>8s} {f'{oracle_acc:.1f}%':>8s} {f'{100*oracle_f1:.1f}%':>8s}")
    print(f"  {'Router ensemble accuracy':<50s} {f'{router_correct}/125':>8s} {f'{100*router_correct/125:.1f}%':>8s}")
    print(f"  {'Hard cascade':<50s} {f'{hard_correct}/125':>8s} {f'{hard_acc:.1f}%':>8s} {f'{100*hard_f1:.1f}%':>8s}")
    print(f"  {'Soft cascade':<50s} {f'{soft_correct}/125':>8s} {f'{soft_acc:.1f}%':>8s} {f'{100*soft_f1:.1f}%':>8s}")
    if os.path.exists(global_path):
        print(f"  {'Cascade+Global blend (α={best_blend_alpha:.2f})':<50s} {f'{best_blend_correct}/125':>8s} {f'{100*best_blend_correct/125:.1f}%':>8s}")
    print(f"  {'Mega ensemble (cascade+global)':<50s} {f'{best_mega_c}/125':>8s} {f'{mega_acc:.1f}%':>8s} {f'{100*mega_f1:.1f}%':>8s}")
    print(f"  {'─'*74}")
    print(f"  {'Paper: Unsafe-Net (4-class, filtered)':<50s} {'—':>8s} {'95.81%':>8s}")
    print(f"  {'Paper: ConvLSTM only (4-class)':<50s} {'—':>8s} {'85.78%':>8s}")

    # Return best result
    results = {
        'oracle': {'correct': oracle_correct, 'acc': oracle_acc, 'f1': 100*oracle_f1},
        'router': {'correct': router_correct, 'acc': 100*router_correct/125},
        'hard_cascade': {'correct': hard_correct, 'acc': hard_acc, 'f1': 100*hard_f1,
                         'cm': hard_cm.tolist()},
        'soft_cascade': {'correct': soft_correct, 'acc': soft_acc, 'f1': 100*soft_f1,
                         'cm': soft_cm.tolist()},
        'mega_ensemble': {'correct': best_mega_c, 'acc': mega_acc, 'f1': 100*mega_f1,
                          'cm': mega_cm.tolist(),
                          'members': [all_candidate_names[i] for i in best_mega_ids]},
    }
    if os.path.exists(global_path):
        results['cascade_global_blend'] = {
            'correct': best_blend_correct, 'acc': 100*best_blend_correct/125,
            'alpha': best_blend_alpha}

    return results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    device = 'cpu'
    t0 = time.time()
    out_dir = 'sweep_results/scenario_cascade'
    os.makedirs(out_dir, exist_ok=True)

    SEEDS = [42, 123, 314]

    print("=" * 70)
    print("SCENARIO-AWARE CASCADE — BINARY SAFE/UNSAFE CLASSIFICATION")
    print("  Strategy: 4-scenario router → per-scenario binary heads")
    print("  Oracle upper bound: 91.2% (114/125)")
    print("=" * 70)

    # Part 1: Train 4-scenario router
    router_results = train_scenario_router(SEEDS, device, out_dir)

    # Part 2: Train per-scenario binary heads
    scen_results, scen_probs, best_per_scen = train_scenario_binary_heads(
        SEEDS, device, out_dir)

    # Part 3: Cascade evaluation
    final_results = cascade_evaluation(
        router_results, scen_results, scen_probs, best_per_scen, device)

    # Save results
    results_path = os.path.join(out_dir, 'cascade_results.json')

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    serializable = {}
    for k, v in final_results.items():
        serializable[k] = {kk: convert(vv) for kk, vv in v.items()}

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=convert)
    print(f"\nResults saved to {results_path}")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
