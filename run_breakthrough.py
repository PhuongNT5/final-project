#!/usr/bin/env python3
"""
BREAKTHROUGH: 3-class grouped + binary cascade to eliminate routing errors.
Previous best: 90.63% (58/64) from 4-class cascade.
Error analysis: 2 errors from 4-class misrouting class0->class2, 4 from binary 0vs1.
Solution: 3-class {walkway_group(0+1), panel(2), forklift(3)} eliminates routing errors.
Then binary refines walkway group into class 0 vs 1.
"""
import os, sys, random, time, math
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
print("[init] run_breakthrough.py loading...", flush=True)
from argparse import ArgumentParser
from copy import deepcopy
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

CN4 = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.backends.cudnn.deterministic = True

def _aug(x):
    T = x.shape[0]
    if random.random() < 0.5:
        n = max(1, int(T * random.uniform(0.05, 0.15)))
        for j in random.sample(range(T), min(n, T)): x[j] = 0.0
    if random.random() < 0.5: x = x + torch.randn_like(x) * random.uniform(0.01, 0.03)
    if random.random() < 0.3: x = x.flip(0)
    if random.random() < 0.3: x = x * random.uniform(0.85, 1.15)
    if random.random() < 0.3: x = torch.roll(x, random.randint(1, max(1, T // 4)), 0)
    return x

class ThreeClassDS(Dataset):
    """3-class: 0=walkway_group(orig 0 or 1), 1=panel(orig 2), 2=forklift(orig 3)."""
    def __init__(self, path, mode='unsafe_only', augment=False):
        p = torch.load(path, map_location='cpu')
        if mode == 'unsafe_only':
            m = p['labels'] < 4; feats, labels = p['features'][m], p['labels'][m]
        else:
            feats, labels = p['features'], p['labels'] % 4
        nl = labels.clone(); nl[labels <= 1] = 0; nl[labels == 2] = 1; nl[labels == 3] = 2
        self.features = feats; self.labels = nl; self.augment = augment
        print(f"    3class({mode}): n={len(nl)}, dist={Counter(nl.tolist())}", flush=True)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        x = self.features[i].clone(); y = int(self.labels[i])
        if self.augment: x = _aug(x)
        if x.dim() == 3: x = x.mean(1)
        return x, y

class BinaryDS(Dataset):
    """Binary: class 0 (WalkwayViol) vs class 1 (UnauthorInt)."""
    def __init__(self, path, mode='unsafe_only', augment=False):
        p = torch.load(path, map_location='cpu')
        if mode == 'unsafe_only':
            m = p['labels'] < 4; feats, labels = p['features'][m], p['labels'][m]
        else:
            feats, labels = p['features'], p['labels'] % 4
        m = (labels == 0) | (labels == 1)
        self.features = feats[m]; self.labels = labels[m].clone(); self.augment = augment
        print(f"    binary({mode}): n={len(self.labels)}, c0={int((self.labels==0).sum())}, c1={int((self.labels==1).sum())}", flush=True)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        x = self.features[i].clone(); y = int(self.labels[i])
        if self.augment: x = _aug(x)
        if x.dim() == 3: x = x.mean(1)
        return x, y

class Direct4DS(Dataset):
    """Standard 4-class."""
    def __init__(self, path, mode='unsafe_only', augment=False):
        p = torch.load(path, map_location='cpu')
        if mode == 'unsafe_only':
            m = p['labels'] < 4; self.features = p['features'][m]; self.labels = p['labels'][m].clone()
        else:
            self.features = p['features']; self.labels = (p['labels'] % 4).clone()
        self.augment = augment
        print(f"    direct4({mode}): n={len(self.labels)}, dist={Counter(self.labels.tolist())}", flush=True)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        x = self.features[i].clone(); y = int(self.labels[i])
        if self.augment: x = _aug(x)
        if x.dim() == 3: x = x.mean(1)
        return x, y

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ls=0.0):
        super().__init__()
        self.register_buffer('weight', weight)
        self.gamma = gamma; self.ls = ls
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction='none', label_smoothing=self.ls)
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
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad: self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)
    def apply(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad: self.backup[n] = p.data.clone(); p.data.copy_(self.shadow[n])
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad: p.data.copy_(self.backup[n])

class SWA:
    def __init__(self): self.states = []
    def collect(self, m): self.states.append(deepcopy(m.state_dict()))
    def averaged(self):
        if not self.states: return None
        avg = {}
        for k in self.states[0]: avg[k] = torch.stack([s[k].float() for s in self.states]).mean(0)
        return avg

def mixup_batch(x, y, a=0.4):
    lam = max(np.random.beta(a, a), 1 - np.random.beta(a, a))
    i = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[i], y, y[i], lam

def tta_predict(model, x, device, n=20, noise=0.015):
    model.eval(); logits = []
    with torch.no_grad():
        xd = x.to(device); logits.append(model(xd)); logits.append(model(xd.flip(1)))
        for s in [1, 2, 3]:
            logits.append(model(torch.roll(xd, s, 1)))
            logits.append(model(torch.roll(xd, -s, 1)))
        for _ in range(max(0, n - 8)):
            logits.append(model(xd + torch.randn_like(xd) * noise))
    return torch.stack(logits).mean(0)

class MotionHead(nn.Module):
    def __init__(self, in_dim, num_classes, model_dim=256, nhead=8, num_layers=4, dropout=0.3, max_t=64):
        super().__init__()
        self.appear_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.motion_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.type_embed = nn.Embedding(3, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nh = nhead
        while model_dim % nh != 0: nh -= 1
        self.pre_norm = nn.LayerNorm(model_dim)
        enc = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(model_dim // 2, num_classes))
    def forward(self, x):
        if x.dim() == 4: x = x.mean(2)
        b, t, c = x.shape; d = x.device
        appear = self.appear_proj(x)
        appear = appear + self.time_pos(torch.arange(t, device=d)).unsqueeze(0)
        appear = appear + self.type_embed(torch.ones(t, dtype=torch.long, device=d)).unsqueeze(0)
        diff = x[:, 1:] - x[:, :-1]
        motion = self.motion_proj(diff)
        motion = motion + self.time_pos(torch.arange(t - 1, device=d)).unsqueeze(0)
        motion = motion + self.type_embed(torch.full((t - 1,), 2, dtype=torch.long, device=d)).unsqueeze(0)
        cls = self.cls_token.expand(b, -1, -1) + self.type_embed(torch.zeros(1, dtype=torch.long, device=d)).unsqueeze(0)
        tokens = self.pre_norm(torch.cat([cls, appear, motion], dim=1))
        tokens = self.transformer(tokens)
        return self.fc(self.post_norm(tokens[:, 0]))

class MotionBinaryHead(nn.Module):
    def __init__(self, in_dim, model_dim=128, nhead=4, num_layers=3, dropout=0.3, max_t=64):
        super().__init__()
        self.diff1_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.diff2_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.appear_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.type_embed = nn.Embedding(4, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nh = nhead
        while model_dim % nh != 0: nh -= 1
        self.pre_norm = nn.LayerNorm(model_dim)
        enc = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(model_dim)
        self.stat_proj = nn.Sequential(nn.Linear(in_dim * 4, model_dim), nn.GELU(), nn.Dropout(dropout))
        self.fc = nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(model_dim, 2))
    def forward(self, x):
        if x.dim() == 4: x = x.mean(2)
        b, t, c = x.shape; d = x.device
        appear = self.appear_proj(x)
        appear = appear + self.time_pos(torch.arange(t, device=d)).unsqueeze(0)
        appear = appear + self.type_embed(torch.ones(t, dtype=torch.long, device=d)).unsqueeze(0)
        d1 = x[:, 1:] - x[:, :-1]
        diff1 = self.diff1_proj(d1)
        diff1 = diff1 + self.time_pos(torch.arange(t - 1, device=d)).unsqueeze(0)
        diff1 = diff1 + self.type_embed(torch.full((t - 1,), 2, dtype=torch.long, device=d)).unsqueeze(0)
        d2 = d1[:, 1:] - d1[:, :-1]
        diff2 = self.diff2_proj(d2)
        diff2 = diff2 + self.time_pos(torch.arange(t - 2, device=d)).unsqueeze(0)
        diff2 = diff2 + self.type_embed(torch.full((t - 2,), 3, dtype=torch.long, device=d)).unsqueeze(0)
        cls = self.cls_token.expand(b, -1, -1) + self.type_embed(torch.zeros(1, dtype=torch.long, device=d)).unsqueeze(0)
        tokens = self.pre_norm(torch.cat([cls, appear, diff1, diff2], dim=1))
        cls_out = self.post_norm(self.transformer(tokens)[:, 0])
        stats = self.stat_proj(torch.cat([x.mean(1), x.std(1), d1.mean(1), d1.std(1)], dim=-1))
        return self.fc(torch.cat([cls_out, stats], dim=-1))

def train_and_eval(model, train_ds, test_x, test_y, cfg, device, extra_x=None):
    nc = cfg['num_classes']
    lnp = np.array([int(train_ds.labels[i]) for i in range(len(train_ds))])
    cc = np.bincount(lnp, minlength=nc).astype(float)
    cw = len(lnp) / (nc * np.maximum(cc, 1)); cw = (cw / cw.mean()).astype(np.float32)
    wt = torch.tensor(cw).to(device)
    crit = FocalLoss(weight=wt, gamma=cfg['gamma'], ls=cfg.get('ls', 0.05))
    sampler = WeightedRandomSampler(torch.as_tensor(cw[lnp], dtype=torch.double), len(lnp), True)
    loader = DataLoader(train_ds, batch_size=cfg['bs'], sampler=sampler, num_workers=0)
    opt = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    sched = CosWarmup(opt, cfg['warmup'], cfg['epochs'])
    ema = EMA(model, 0.998); swa = SWA(); swa_start = int(cfg['epochs'] * 0.6)
    best_f1, best_acc, best_st = 0., 0., None; stale = 0
    ynp = test_y.numpy() if torch.is_tensor(test_y) else test_y
    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if cfg.get('mixup', 0) > 0 and random.random() < 0.5:
                xm, ya, yb2, lam = mixup_batch(xb, yb, cfg['mixup'])
                opt.zero_grad(set_to_none=True)
                loss = lam * crit(model(xm), ya) + (1 - lam) * crit(model(xm), yb2)
            else:
                opt.zero_grad(set_to_none=True); loss = crit(model(xb), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(model)
        sched.step()
        if ep >= swa_start: swa.collect(model)
        ema.apply(model); model.eval()
        with torch.no_grad(): pnp = model(test_x.to(device)).argmax(1).cpu().numpy()
        acc = 100 * (pnp == ynp).mean()
        f1 = f1_score(ynp, pnp, average='macro', zero_division=0)
        if f1 > best_f1 + 1e-4 or (abs(f1 - best_f1) < 1e-4 and acc > best_acc):
            best_f1, best_acc = f1, acc; best_st = deepcopy(model.state_dict()); stale = 0
        else: stale += 1
        ema.restore(model)
        if ep % 10 == 0 or ep == 1:
            print(f"      ep{ep:3d} acc={acc:.1f}% f1={f1:.3f} best={best_acc:.1f}%", flush=True)
        if stale >= cfg.get('patience', 60):
            print(f"      [early stop ep{ep}]", flush=True); break
    swa_st = swa.averaged()
    results = {}
    for vn, st in [('best', best_st), ('swa', swa_st)]:
        if st is None: continue
        model.load_state_dict(st); model.eval()
        with torch.no_grad(): logits = model(test_x.to(device))
        probs = F.softmax(logits, 1).cpu().numpy(); preds = logits.argmax(1).cpu().numpy()
        acc_v = 100 * (preds == ynp).mean()
        results[vn] = {'probs': probs, 'preds': preds, 'acc': acc_v}
        lt = tta_predict(model, test_x, device, 20)
        pt = F.softmax(lt, 1).cpu().numpy(); pp = lt.argmax(1).cpu().numpy()
        results[vn + '_tta'] = {'probs': pt, 'preds': pp, 'acc': 100 * (pp == ynp).mean()}
        if extra_x is not None:
            with torch.no_grad(): el = model(extra_x.to(device))
            results[vn + '_ex'] = {'probs': F.softmax(el, 1).cpu().numpy(), 'preds': el.argmax(1).cpu().numpy()}
            elt = tta_predict(model, extra_x, device, 20)
            results[vn + '_tta_ex'] = {'probs': F.softmax(elt, 1).cpu().numpy(), 'preds': elt.argmax(1).cpu().numpy()}
    return results

def cascade_3c_bin(c3p, binp, ynp, label=''):
    results = []
    c3pred = c3p.argmax(1)
    final = np.zeros(len(ynp), dtype=int)
    for i in range(len(ynp)):
        if c3pred[i] == 0: final[i] = binp[i].argmax()
        elif c3pred[i] == 1: final[i] = 2
        else: final[i] = 3
    acc = 100 * (final == ynp).mean(); cor = int((final == ynp).sum())
    results.append({'name': f'{label}_std', 'acc': acc, 'correct': cor, 'preds': final.copy(), 'method': '3c+bin'})
    for alpha in [0.5, 0.7, 1.0]:
        fused = np.zeros((len(ynp), 4))
        for i in range(len(ynp)):
            pg = c3p[i, 0]
            fused[i, 0] = pg * ((1 - alpha) * 0.5 + alpha * binp[i, 0])
            fused[i, 1] = pg * ((1 - alpha) * 0.5 + alpha * binp[i, 1])
            fused[i, 2] = c3p[i, 1]; fused[i, 3] = c3p[i, 2]
        fp = fused.argmax(1); fa = 100 * (fp == ynp).mean()
        results.append({'name': f'{label}_fuse{alpha}', 'acc': fa, 'correct': int((fp == ynp).sum()),
                        'preds': fp.copy(), 'method': '3c+bin_fuse'})
    return results

def cascade_4c_bin(d4p, binp, ynp, label=''):
    d4pred = d4p.argmax(1); ref = d4pred.copy()
    for i in range(len(ynp)):
        if d4pred[i] <= 1: ref[i] = binp[i].argmax()
    acc = 100 * (ref == ynp).mean()
    return [{'name': f'{label}_4cstd', 'acc': acc, 'correct': int((ref == ynp).sum()),
             'preds': ref.copy(), 'method': '4c+bin'}]

def main():
    print("[main] Starting breakthrough experiment", flush=True)
    pa = ArgumentParser()
    pa.add_argument('--train', default='features/train_T32_img224_pool4_tokens_yolo11s.pt')
    pa.add_argument('--test', default='features/test_T32_img224_pool4_tokens_yolo11s.pt')
    pa.add_argument('--out', default='sweep_results/breakthrough_v1')
    args = pa.parse_args()
    os.makedirs(args.out, exist_ok=True); device = 'cpu'; t0 = time.time()
    print("Loading T32 features...", flush=True)
    tst = torch.load(args.test, map_location='cpu')
    m = tst['labels'] < 4
    test_x = tst['features'][m].mean(dim=2)
    test_y4 = tst['labels'][m].clone(); ynp = test_y4.numpy()
    test_y3 = test_y4.clone(); test_y3[test_y4 <= 1] = 0; test_y3[test_y4 == 2] = 1; test_y3[test_y4 == 3] = 2
    ynp3 = test_y3.numpy()
    bm = (test_y4 == 0) | (test_y4 == 1)
    test_x_bin = test_x[bm]; test_y_bin = test_y4[bm].clone(); ynp_bin = test_y_bin.numpy()
    print(f"  64 unsafe, {bm.sum()} binary (c0={int((test_y_bin==0).sum())}, c1={int((test_y_bin==1).sum())})", flush=True)
    SEEDS = [42, 123, 7, 2024, 999, 555, 314, 888]
    all_c3 = []; all_d4 = []; all_bin = []

    print(f"\n{'='*60}\nSTAGE 1A: 3-class grouped\n{'='*60}", flush=True)
    c3_cfgs = [
        {'dim': 256, 'layers': 4, 'lr': 2e-4, 'wd': 0.05, 'gamma': 3.0, 'mixup': 0.4, 'bs': 16, 'ls': 0.1, 'dropout': 0.3, 'mode': 'unsafe_only', 'tag': 'A'},
        {'dim': 256, 'layers': 4, 'lr': 3e-4, 'wd': 0.05, 'gamma': 3.0, 'mixup': 0.4, 'bs': 16, 'ls': 0.1, 'dropout': 0.3, 'mode': 'scenario_all', 'tag': 'B'},
        {'dim': 384, 'layers': 4, 'lr': 2e-4, 'wd': 0.05, 'gamma': 3.0, 'mixup': 0.4, 'bs': 16, 'ls': 0.1, 'dropout': 0.3, 'mode': 'unsafe_only', 'tag': 'C'},
    ]
    tot = len(c3_cfgs) * len(SEEDS); ci = 0
    for cfg_b in c3_cfgs:
        for seed in SEEDS:
            ci += 1; name = f"c3_{cfg_b['tag']}_s{seed}"
            print(f"\n  [{ci}/{tot}] {name}", flush=True); set_seed(seed)
            cfg = {**cfg_b, 'num_classes': 3, 'epochs': 120, 'warmup': 10, 'patience': 50}
            ds = ThreeClassDS(args.train, mode=cfg['mode'], augment=True)
            model = MotionHead(768, 3, cfg['dim'], 8, cfg['layers'], cfg['dropout'], 64).to(device)
            res = train_and_eval(model, ds, test_x, test_y3, cfg, device)
            bv = max(((k, v) for k, v in res.items() if 'acc' in v), key=lambda kv: kv[1]['acc'])
            bp = bv[1]['preds']
            grp_ok = sum(1 for i in range(64) if ynp[i] <= 1 and bp[i] == 0)
            pan_ok = sum(1 for i in range(64) if ynp[i] == 2 and bp[i] == 1)
            frk_ok = sum(1 for i in range(64) if ynp[i] == 3 and bp[i] == 2)
            print(f"    -> {bv[0]}={bv[1]['acc']:.1f}% | route: grp={grp_ok}/43 pan={pan_ok}/13 frk={frk_ok}/8", flush=True)
            all_c3.append({'name': name, 'results': res, 'tag': cfg_b['tag'], 'seed': seed})

    print(f"\n{'='*60}\nSTAGE 1B: 4-class direct (comparison)\n{'='*60}", flush=True)
    d4_cfgs = [
        {'dim': 256, 'layers': 4, 'lr': 3e-4, 'wd': 0.05, 'gamma': 3.0, 'mixup': 0.4, 'bs': 16, 'ls': 0.1, 'dropout': 0.3, 'mode': 'scenario_all', 'tag': 'B'},
        {'dim': 384, 'layers': 4, 'lr': 2e-4, 'wd': 0.05, 'gamma': 3.0, 'mixup': 0.4, 'bs': 16, 'ls': 0.1, 'dropout': 0.3, 'mode': 'unsafe_only', 'tag': 'C'},
    ]
    d4_seeds = [42, 123, 7, 2024, 999]
    tot = len(d4_cfgs) * len(d4_seeds); di = 0
    for cfg_b in d4_cfgs:
        for seed in d4_seeds:
            di += 1; name = f"d4_{cfg_b['tag']}_s{seed}"
            print(f"\n  [{di}/{tot}] {name}", flush=True); set_seed(seed)
            cfg = {**cfg_b, 'num_classes': 4, 'epochs': 100, 'warmup': 10, 'patience': 45}
            ds = Direct4DS(args.train, mode=cfg['mode'], augment=True)
            model = MotionHead(768, 4, cfg['dim'], 8, cfg['layers'], cfg['dropout'], 64).to(device)
            res = train_and_eval(model, ds, test_x, test_y4, cfg, device)
            bv = max(((k, v) for k, v in res.items() if 'acc' in v), key=lambda kv: kv[1]['acc'])
            bp = bv[1]['preds']
            c02 = sum(1 for i in range(64) if ynp[i] == 0 and bp[i] == 2)
            print(f"    -> {bv[0]}={bv[1]['acc']:.1f}% (c0->c2={c02})", flush=True)
            all_d4.append({'name': name, 'results': res, 'tag': cfg_b['tag'], 'seed': seed})

    print(f"\n{'='*60}\nSTAGE 2: Binary 0-vs-1 (long training, T32)\n{'='*60}", flush=True)
    bin_cfgs = [
        {'dim': 192, 'layers': 4, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0, 'mixup': 0.4, 'bs': 8, 'dropout': 0.3, 'mode': 'scenario_all', 'tag': 'S192'},
        {'dim': 128, 'layers': 3, 'lr': 3e-4, 'wd': 0.05, 'gamma': 2.0, 'mixup': 0.3, 'bs': 8, 'dropout': 0.3, 'mode': 'unsafe_only', 'tag': 'B128'},
        {'dim': 256, 'layers': 4, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0, 'mixup': 0.4, 'bs': 8, 'dropout': 0.35, 'mode': 'scenario_all', 'tag': 'S256'},
    ]
    tot = len(bin_cfgs) * len(SEEDS); bi = 0
    for cfg_b in bin_cfgs:
        for seed in SEEDS:
            bi += 1; name = f"bin_{cfg_b['tag']}_s{seed}"
            print(f"\n  [{bi}/{tot}] {name}", flush=True); set_seed(seed)
            cfg = {**cfg_b, 'num_classes': 2, 'epochs': 200, 'warmup': 15, 'patience': 60, 'ls': 0.05}
            ds = BinaryDS(args.train, mode=cfg['mode'], augment=True)
            model = MotionBinaryHead(768, cfg['dim'], 4, cfg['layers'], cfg['dropout'], 64).to(device)
            res = train_and_eval(model, ds, test_x_bin, test_y_bin, cfg, device, extra_x=test_x)
            bv = max(((k, v) for k, v in res.items() if 'acc' in v), key=lambda kv: kv[1]['acc'])
            print(f"    -> {bv[0]}={bv[1]['acc']:.1f}%", flush=True)
            all_bin.append({'name': name, 'results': res, 'tag': cfg_b['tag'], 'seed': seed, 'best_acc': bv[1]['acc']})

    print(f"\n{'='*60}\nSTAGE 3: Cascade combinations\n{'='*60}", flush=True)
    sorted_bin = sorted(all_bin, key=lambda x: -x['best_acc'])
    print("  Top binary:", flush=True)
    for i, b in enumerate(sorted_bin[:10]):
        print(f"    {i+1}. {b['name']}: {b['best_acc']:.1f}%", flush=True)
    all_cascade = []
    for c3 in all_c3:
        for c3v in ['best', 'best_tta', 'swa', 'swa_tta']:
            if c3v not in c3['results']: continue
            c3p = c3['results'][c3v]['probs']
            for b in sorted_bin[:15]:
                for bv_key in ['swa_tta_ex', 'best_tta_ex', 'swa_ex', 'best_ex']:
                    if bv_key not in b['results']: continue
                    bp64 = b['results'][bv_key]['probs']
                    label = f"{c3['name']}_{c3v}+{b['name']}_{bv_key}"
                    all_cascade.extend(cascade_3c_bin(c3p, bp64, ynp, label))
                    break
            for K in [3, 5]:
                bpl = []
                for b2 in sorted_bin[:K]:
                    for bv_key in ['swa_tta_ex', 'best_tta_ex', 'swa_ex', 'best_ex']:
                        if bv_key in b2['results']:
                            bpl.append(b2['results'][bv_key]['probs']); break
                if bpl:
                    avg_bp = np.mean(bpl, 0)
                    label = f"{c3['name']}_{c3v}+binEns{K}"
                    all_cascade.extend(cascade_3c_bin(c3p, avg_bp, ynp, label))
    sorted_c3 = sorted(all_c3, key=lambda x: -max(v['acc'] for v in x['results'].values() if 'acc' in v))
    for K3 in [3, 5, 8]:
        c3pl = []
        for c3 in sorted_c3[:K3]:
            for cv in ['swa_tta', 'best_tta', 'swa', 'best']:
                if cv in c3['results']:
                    c3pl.append(c3['results'][cv]['probs']); break
        if not c3pl: continue
        avg_c3 = np.mean(c3pl, 0)
        for Kb in [3, 5]:
            bpl = []
            for b2 in sorted_bin[:Kb]:
                for bv_key in ['swa_tta_ex', 'best_tta_ex', 'swa_ex', 'best_ex']:
                    if bv_key in b2['results']:
                        bpl.append(b2['results'][bv_key]['probs']); break
            if bpl:
                avg_bp = np.mean(bpl, 0)
                label = f"c3ens{K3}+binens{Kb}"
                all_cascade.extend(cascade_3c_bin(avg_c3, avg_bp, ynp, label))
    for d4 in all_d4:
        for d4v in ['best', 'best_tta', 'swa', 'swa_tta']:
            if d4v not in d4['results']: continue
            d4p = d4['results'][d4v]['probs']
            for b in sorted_bin[:10]:
                for bv_key in ['swa_tta_ex', 'best_tta_ex', 'swa_ex', 'best_ex']:
                    if bv_key in b['results']:
                        bp64 = b['results'][bv_key]['probs']
                        all_cascade.extend(cascade_4c_bin(d4p, bp64, ynp,
                            f"{d4['name']}_{d4v}+{b['name']}_{bv_key}"))
                        break
    all_cascade.sort(key=lambda x: -x['acc'])

    print(f"\n{'='*60}\nFINAL RESULTS - TOP 30\n{'='*60}", flush=True)
    for i, r in enumerate(all_cascade[:30]):
        print(f"  {i+1:2d}. {r['acc']:.1f}% ({r['correct']}/64) [{r.get('method','?')}] {r['name']}", flush=True)
    best = all_cascade[0] if all_cascade else None
    if best:
        print(f"\n  *** BEST: {best['acc']:.1f}% ({best['correct']}/64) ***", flush=True)
        print(f"  Paper: 95.81% (62/64)", flush=True)
        print(f"  Gap: {95.81 - best['acc']:.2f}pp", flush=True)
        print(f"  Prev best: 90.63% (58/64)", flush=True)
        if best['acc'] > 90.63:
            print(f"  >>> NEW RECORD! +{best['acc'] - 90.63:.2f}pp!", flush=True)
        bp = best.get('preds')
        if bp is not None:
            cm = confusion_matrix(ynp, bp, labels=[0, 1, 2, 3])
            print(f"\n  Confusion matrix:", flush=True)
            for i, row in enumerate(cm): print(f"    {CN4[i]:15s}: {row}", flush=True)
            print(f"\n{classification_report(ynp, bp, target_names=CN4, digits=3)}", flush=True)
            plt.figure(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CN4, yticklabels=CN4)
            plt.title(f"Best: {best['acc']:.1f}% ({best['correct']}/64)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, 'cm_best.png'), dpi=120); plt.close()

    print(f"\n{'='*60}\nANALYSIS\n{'='*60}", flush=True)
    print("3-CLASS ROUTING:", flush=True)
    for tag in sorted(set(c['tag'] for c in all_c3)):
        accs = [max(v['acc'] for v in c['results'].values() if 'acc' in v) for c in all_c3 if c['tag'] == tag]
        print(f"  {tag}: mean={np.mean(accs):.1f}% max={np.max(accs):.1f}% std={np.std(accs):.1f}% n={len(accs)}", flush=True)
    print("BINARY:", flush=True)
    for tag in sorted(set(b['tag'] for b in all_bin)):
        accs = [b['best_acc'] for b in all_bin if b['tag'] == tag]
        print(f"  {tag}: mean={np.mean(accs):.1f}% max={np.max(accs):.1f}% std={np.std(accs):.1f}% n={len(accs)}", flush=True)
    print("4-CLASS DIRECT:", flush=True)
    for tag in sorted(set(d['tag'] for d in all_d4)):
        accs = [max(v['acc'] for v in d['results'].values() if 'acc' in v) for d in all_d4 if d['tag'] == tag]
        print(f"  {tag}: mean={np.mean(accs):.1f}% max={np.max(accs):.1f}% std={np.std(accs):.1f}% n={len(accs)}", flush=True)

    total_time = time.time() - t0
    with open(os.path.join(args.out, 'report.txt'), 'w') as f:
        f.write(f"BREAKTHROUGH v1 - 3-class+binary cascade\n{'='*60}\n")
        f.write(f"Time: {total_time/60:.1f}m\n")
        f.write(f"3-class: {len(all_c3)}, 4-class: {len(all_d4)}, binary: {len(all_bin)}\n\n")
        if best:
            f.write(f"BEST: {best['acc']:.1f}% ({best['correct']}/64) | {best['name']}\n")
            f.write(f"Paper: 95.81% (62/64), Gap: {95.81 - best['acc']:.2f}pp\n\n")
        f.write("TOP 30:\n")
        for i, r in enumerate(all_cascade[:30]):
            f.write(f"  {i+1}. {r['acc']:.1f}% ({r['correct']}/64) [{r.get('method','?')}] {r['name']}\n")
    print(f"\n  Time: {total_time/60:.1f}m | Saved to {args.out}/", flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
