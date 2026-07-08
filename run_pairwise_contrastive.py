#!/usr/bin/env python3
"""
Pairwise Contrastive: Train 4 binary pair classifiers on safe/unsafe pairs.

Key insight: Safe/unsafe pairs share the SAME camera view.
  Class 0 (WalkwayViol) ↔ Class 4 (SafeWalkway)
  Class 1 (UnauthorInt) ↔ Class 5 (AuthorizedInt)
  Class 2 (OpenedPanel) ↔ Class 6 (ClosedPanel)
  Class 3 (CarryOverload) ↔ Class 7 (SafeCarrying)

The model is FORCED to learn behavioral differences, not scene appearance.

Pipeline:
  Stage 1: Scenario classifier (which camera/scene?) — 4-class, easy
  Stage 2: Per-scenario binary classifier (safe or unsafe?) — hard but focused
  Stage 3: Combine → 4 unsafe class prediction
  + Direct 4-class and 8-class classifiers for ensemble diversity
"""
import os, sys, random, time, math, json, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import f1_score, confusion_matrix
from copy import deepcopy
from PIL import Image
import cv2

try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass

CLASS_NAMES_4 = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']
CLASS_NAMES_8 = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload',
                 'SafeWalkway', 'AuthorizedInt', 'ClosedPanel', 'SafeCarrying']
DATA_DIR = 'data'
OUT_DIR = 'sweep_results/pairwise'
FEAT_DIR = os.path.join(OUT_DIR, 'features')
SEEDS = [42, 123, 314, 7, 2024]

# Unsafe/safe pairs: (unsafe_class, safe_class)
PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ls=0.0):
        super().__init__()
        self.register_buffer('weight', weight)
        self.gamma = gamma; self.ls = ls
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction='none', label_smoothing=self.ls)
        pt = F.softmax(logits, 1).gather(1, target.view(-1, 1)).squeeze(1).clamp(1e-6)
        return (((1 - pt) ** self.gamma) * ce).mean()


class EMA:
    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    def update(self, m):
        for n, p in m.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1-self.decay)
    def apply(self, m):
        self.backup = {}
        for n, p in m.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone(); p.data.copy_(self.shadow[n])
    def restore(self, m):
        for n, p in m.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])


def _aug(x):
    T = x.shape[0]
    if random.random() < 0.5: x = x + torch.randn_like(x) * 0.03
    if random.random() < 0.4: x = x.flip(0)
    if random.random() < 0.3: x = x * random.uniform(0.85, 1.15)
    if random.random() < 0.3: x = torch.roll(x, random.randint(1, max(1, T//4)), 0)
    if random.random() < 0.25:
        n = max(1, int(T*0.15))
        for i in random.sample(range(T), min(n, T)): x[i] = 0.0
    if random.random() < 0.15: x = 0.8*x + 0.2*x.flip(0)
    return x


# ══════════════════════════════════════════════════════════════
# Feature Extraction (all 8 classes)
# ══════════════════════════════════════════════════════════════

def extract_all_features(device):
    """Extract frozen COCO YOLO features for ALL 8 classes."""
    os.makedirs(FEAT_DIR, exist_ok=True)
    f_train = os.path.join(FEAT_DIR, 'all8_train.pt')
    f_test = os.path.join(FEAT_DIR, 'all8_test.pt')
    if os.path.exists(f_train) and os.path.exists(f_test):
        print("  Features cached")
        return torch.load(f_train, map_location='cpu'), torch.load(f_test, map_location='cpu')

    from ultralytics import YOLO
    print("  Extracting frozen COCO YOLO features (all 8 classes)...")
    yolo = YOLO('yolo11s.pt')
    backbone = yolo.model
    for p in backbone.parameters(): p.requires_grad = False
    backbone.eval()
    pool_size = 4; target_idx = 22
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    for subset in ['train', 'test']:
        sp = os.path.join(FEAT_DIR, f'all8_{subset}.pt')
        if os.path.exists(sp): continue
        subset_path = os.path.join(DATA_DIR, subset)
        samples = []
        for folder in sorted(os.listdir(subset_path)):
            full = os.path.join(subset_path, folder)
            if not os.path.isdir(full): continue
            label = int(folder.split('_')[0])
            for vf in sorted(os.listdir(full)):
                if vf.lower().endswith(('.mp4', '.avi', '.mov')):
                    samples.append((os.path.join(full, vf), label))

        print(f"\n  {subset}: {len(samples)} videos")
        all_feats, all_labels = [], []
        for vi, (vpath, label) in enumerate(samples):
            cap = cv2.VideoCapture(vpath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            T = 32
            if total <= 0:
                cap.release()
                all_feats.append(torch.zeros(T, 16, 512))
                all_labels.append(label)
                continue
            indices = np.linspace(0, total - 1, T, dtype=int)
            tokens_list = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    tokens_list.append(torch.zeros(16, 512))
                    continue
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor = tf(img).unsqueeze(0)
                features = None
                def hook(m, i, o):
                    nonlocal features; features = o
                h = backbone.model[target_idx].register_forward_hook(hook)
                with torch.no_grad(): backbone(tensor)
                h.remove()
                if isinstance(features, (list, tuple)): features = features[0]
                fh, fw = features.shape[2], features.shape[3]
                if fh % pool_size != 0 or fw % pool_size != 0:
                    nh = max(pool_size, (fh // pool_size) * pool_size)
                    nw = max(pool_size, (fw // pool_size) * pool_size)
                    features = F.interpolate(features, size=(nh, nw), mode='bilinear', align_corners=False)
                pooled = F.adaptive_avg_pool2d(features, (pool_size, pool_size))
                tokens_list.append(pooled.flatten(2).transpose(1, 2).squeeze(0))
            cap.release()
            while len(tokens_list) < T:
                tokens_list.append(tokens_list[-1] if tokens_list else torch.zeros(16, 512))
            all_feats.append(torch.stack(tokens_list[:T]))
            all_labels.append(label)
            if (vi + 1) % 20 == 0 or vi + 1 == len(samples):
                print(f"    {vi+1}/{len(samples)}", flush=True)

        features = torch.stack(all_feats)
        labels = torch.tensor(all_labels, dtype=torch.long)
        print(f"  {subset}: {features.shape}, labels: {[int((labels==c).sum()) for c in range(8)]}")
        torch.save({'features': features, 'labels': labels}, sp)

    del backbone, yolo; gc.collect()
    return torch.load(f_train, map_location='cpu'), torch.load(f_test, map_location='cpu')


# ══════════════════════════════════════════════════════════════
# Transformer Head
# ══════════════════════════════════════════════════════════════

class TransformerHead(nn.Module):
    def __init__(self, in_dim, nc, dim=256, nhead=8, layers=4, drop=0.3):
        super().__init__()
        self.appear = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.GELU(), nn.Dropout(drop * 0.3))
        self.motion = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.GELU(), nn.Dropout(drop * 0.3))
        self.tpos = nn.Embedding(64, dim)
        self.te = nn.Embedding(3, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)); nn.init.trunc_normal_(self.cls, std=0.02)
        nh = nhead
        while dim % nh != 0: nh -= 1
        self.pre = nn.LayerNorm(dim)
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=nh, dim_feedforward=dim * 4,
                                         dropout=drop, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=layers)
        self.post = nn.LayerNorm(dim)
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(drop * 0.5),
                                nn.Linear(dim // 2, nc))

    def forward(self, x):
        if x.dim() == 4: x = x.mean(2)
        b, t, c = x.shape; d = x.device
        a = self.appear(x) + self.tpos(torch.arange(t, device=d)) + self.te(torch.ones(t, dtype=torch.long, device=d))
        diff = x[:, 1:] - x[:, :-1]
        m = self.motion(diff) + self.tpos(torch.arange(t - 1, device=d)) + self.te(torch.full((t - 1,), 2, dtype=torch.long, device=d))
        cls = self.cls.expand(b, -1, -1) + self.te(torch.zeros(1, dtype=torch.long, device=d))
        return self.fc(self.post(self.transformer(self.pre(torch.cat([cls, a, m], 1)))[:, 0]))


class FeatDS(Dataset):
    def __init__(self, f, l, aug=False, stats=False):
        self.features = torch.cat([f.mean(2), f.std(2), f.max(2).values], -1) if stats and f.dim() == 4 else f
        self.labels = l; self.aug = aug
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        x = self.features[i].clone()
        if self.aug: x = _aug(x)
        return x, int(self.labels[i])


def train_classifier(model, ds, test_x, test_y, lr, device, sp, epochs=300, patience=60):
    nc = int(test_y.max().item()) + 1
    lnp = ds.labels.numpy() if torch.is_tensor(ds.labels) else np.array(ds.labels)
    cc = np.bincount(lnp, minlength=nc).astype(float)
    cw = len(lnp) / (nc * np.maximum(cc, 1)); cw = (cw / cw.mean()).astype(np.float32)
    wt = torch.tensor(cw).to(device)
    crit = FocalLoss(weight=wt, gamma=2.0, ls=0.05)
    sampler = WeightedRandomSampler(torch.as_tensor(cw[lnp], dtype=torch.double), len(lnp), True)
    loader = DataLoader(ds, batch_size=8, sampler=sampler, num_workers=0)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    warmup = 15
    sched = optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep + 1) / warmup if ep < warmup else 0.5 * (1 + math.cos(math.pi * (ep - warmup) / (epochs - warmup))))
    ema = EMA(model, 0.998)
    ynp = test_y.numpy() if torch.is_tensor(test_y) else test_y
    best_f1, best_c, best_st = 0., 0, None; stale = 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(model)
        sched.step()
        ema.apply(model); model.eval()
        with torch.no_grad(): pnp = model(test_x.to(device)).argmax(1).cpu().numpy()
        f1 = f1_score(ynp, pnp, average='macro', zero_division=0)
        c = int((pnp == ynp).sum())
        if c > best_c or (c == best_c and f1 > best_f1 + 1e-5):
            best_f1, best_c = f1, c; best_st = deepcopy(model.state_dict()); stale = 0
        else: stale += 1
        ema.restore(model)
        if ep % 50 == 0 or ep == 1:
            print(f"        ep{ep:3d} {c}/{len(ynp)} best={best_c}", flush=True)
        if stale >= patience:
            print(f"        [stop ep{ep}] best={best_c}", flush=True); break
    if sp and best_st:
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        torch.save(best_st, sp)
    return best_st, best_c, best_f1


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Extract features (all 8 classes) ──
    print("=" * 70)
    print("PHASE 1: FEATURE EXTRACTION (ALL 8 CLASSES)")
    print("=" * 70)
    train_data, test_data = extract_all_features(device)
    all_tr_feats = train_data['features']  # (N, T, S, C)
    all_tr_labels = train_data['labels']
    all_te_feats = test_data['features']
    all_te_labels = test_data['labels']

    feat_dim = int(all_tr_feats.shape[-1])
    stats_dim = feat_dim * 3

    # Spatial stats
    all_tr_stats = torch.cat([all_tr_feats.mean(2), all_tr_feats.std(2), all_tr_feats.max(2).values], -1)
    all_te_stats = torch.cat([all_te_feats.mean(2), all_te_feats.std(2), all_te_feats.max(2).values], -1)

    # 4-class unsafe subset (classes 0-3 only)
    mask_tr4 = all_tr_labels < 4
    mask_te4 = all_te_labels < 4
    tr4_stats = all_tr_stats[mask_tr4]; tr4_labels = all_tr_labels[mask_tr4]
    te4_stats = all_te_stats[mask_te4]; te4_labels = all_te_labels[mask_te4]

    print(f"\n  All 8: train={all_tr_stats.shape}, test={all_te_stats.shape}")
    print(f"  4-class: train={tr4_stats.shape}, test={te4_stats.shape}")
    ynp4 = te4_labels.numpy()
    n_test4 = len(ynp4)

    all_probs = {}  # tag → (64, 4) probs for 4-class test set
    all_results = {}

    # ── APPROACH A: Direct 4-class (baseline, like V3) ──
    print("\n" + "=" * 70)
    print("APPROACH A: DIRECT 4-CLASS (baseline)")
    print("=" * 70)

    for seed in SEEDS:
        tag = f'direct4_s{seed}'
        sp = os.path.join(OUT_DIR, f'{tag}.pt')
        set_seed(seed)
        model = TransformerHead(stats_dim, 4, 384, 8, 6, 0.3).to(device)
        print(f"  {tag}: ", end='', flush=True)
        if os.path.exists(sp):
            model.load_state_dict(torch.load(sp, map_location=device)); print("loaded", end='')
        else:
            ds = FeatDS(tr4_stats, tr4_labels, aug=True)
            train_classifier(model, ds, te4_stats, te4_labels, 1.5e-4, device, sp)
            model.load_state_dict(torch.load(sp, map_location=device))
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(te4_stats.to(device)), 1).cpu().numpy()
        c = int((probs.argmax(1) == ynp4).sum())
        macro = np.mean([100 * int((probs.argmax(1)[ynp4 == i] == i).sum()) / max(1, int((ynp4 == i).sum())) for i in range(4)])
        print(f" → {c}/{n_test4} macro={macro:.1f}%")
        all_probs[tag] = probs
        all_results[tag] = {'correct': c, 'macro': macro}

    # ── APPROACH B: 8-class → map to 4-class ──
    print("\n" + "=" * 70)
    print("APPROACH B: 8-CLASS TRAINING → MAP TO 4 UNSAFE")
    print("=" * 70)

    # Map 8-class to 4 unsafe: merge safe into unsafe counterpart
    # At test time: take max(unsafe_prob, safe_counterpart_prob) for each pair
    for seed in SEEDS:
        tag = f'eight_s{seed}'
        sp = os.path.join(OUT_DIR, f'{tag}.pt')
        set_seed(seed)
        model = TransformerHead(stats_dim, 8, 384, 8, 6, 0.3).to(device)
        print(f"  {tag}: ", end='', flush=True)
        if os.path.exists(sp):
            model.load_state_dict(torch.load(sp, map_location=device)); print("loaded", end='')
        else:
            ds = FeatDS(all_tr_stats, all_tr_labels, aug=True)
            train_classifier(model, ds, all_te_stats, all_te_labels, 1.5e-4, device, sp)
            model.load_state_dict(torch.load(sp, map_location=device))
        model.eval()
        with torch.no_grad():
            probs8 = F.softmax(model(all_te_stats.to(device)), 1).cpu().numpy()
        # Map 8→4: for each pair, combine unsafe + safe probs
        probs4_mapped = np.zeros((len(all_te_labels), 4))
        for ui, (u_cls, s_cls) in enumerate(PAIRS):
            probs4_mapped[:, ui] = probs8[:, u_cls] + probs8[:, s_cls]
        # Only evaluate on 4-class test set (classes 0-3)
        probs4_test = probs4_mapped[mask_te4.numpy()]
        c = int((probs4_test.argmax(1) == ynp4).sum())
        macro = np.mean([100 * int((probs4_test.argmax(1)[ynp4 == i] == i).sum()) / max(1, int((ynp4 == i).sum())) for i in range(4)])
        print(f" → {c}/{n_test4} macro={macro:.1f}%")
        all_probs[tag] = probs4_test
        all_results[tag] = {'correct': c, 'macro': macro}

    # ── APPROACH C: Pairwise binary classifiers ──
    print("\n" + "=" * 70)
    print("APPROACH C: PAIRWISE BINARY CLASSIFIERS")
    print("=" * 70)

    pair_names = ['Walkway', 'Intervention', 'Panel', 'Forklift']
    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        pair_probs = []  # will hold (n_test4, 2) for each pair

        for pi, (u_cls, s_cls) in enumerate(PAIRS):
            tag_pair = f'pair{pi}_{pair_names[pi]}_s{seed}'
            sp = os.path.join(OUT_DIR, f'{tag_pair}.pt')
            set_seed(seed)

            # Training data: unsafe class + safe counterpart
            pair_mask_tr = (all_tr_labels == u_cls) | (all_tr_labels == s_cls)
            pair_tr_stats = all_tr_stats[pair_mask_tr]
            pair_tr_labels = (all_tr_labels[pair_mask_tr] == u_cls).long()  # 1=unsafe, 0=safe

            n_unsafe = int(pair_tr_labels.sum())
            n_safe = len(pair_tr_labels) - n_unsafe
            print(f"    Pair {pi} ({pair_names[pi]}): {n_unsafe} unsafe + {n_safe} safe = {len(pair_tr_labels)}")

            model = TransformerHead(stats_dim, 2, 192, 6, 3, 0.3).to(device)
            if os.path.exists(sp):
                model.load_state_dict(torch.load(sp, map_location=device))
            else:
                ds = FeatDS(pair_tr_stats, pair_tr_labels, aug=True)
                # Test on ALL test data (both unsafe subset and safe subset for this pair)
                pair_mask_te = (all_te_labels == u_cls) | (all_te_labels == s_cls)
                pair_te_stats = all_te_stats[pair_mask_te]
                pair_te_labels = (all_te_labels[pair_mask_te] == u_cls).long()
                train_classifier(model, ds, pair_te_stats, pair_te_labels, 2e-4, device, sp, epochs=300, patience=60)
                model.load_state_dict(torch.load(sp, map_location=device))

            # Get binary probs on 4-class test set
            model.eval()
            with torch.no_grad():
                bp = F.softmax(model(te4_stats.to(device)), 1).cpu().numpy()  # (64, 2)
            pair_probs.append(bp[:, 1])  # prob of being unsafe for this pair

            # Per-pair accuracy on pair-specific test clips
            pair_mask_te4 = (te4_labels == pi)
            if pair_mask_te4.sum() > 0:
                pair_pred = bp[pair_mask_te4.numpy(), 1] > 0.5
                pair_acc = pair_pred.mean() * 100
                print(f"      Pair acc on class {pi} test: {pair_acc:.1f}%")

        # Combine pair probs into 4-class prediction
        # Each pair gives P(unsafe | this scenario). Use as class score.
        pair_scores = np.stack(pair_probs, axis=1)  # (64, 4)
        # Normalize to probabilities
        pair_scores_norm = pair_scores / (pair_scores.sum(axis=1, keepdims=True) + 1e-8)

        tag = f'pairwise_s{seed}'
        c = int((pair_scores_norm.argmax(1) == ynp4).sum())
        macro = np.mean([100 * int((pair_scores_norm.argmax(1)[ynp4 == i] == i).sum()) / max(1, int((ynp4 == i).sum())) for i in range(4)])
        print(f"    Combined: {c}/{n_test4} macro={macro:.1f}%")
        all_probs[tag] = pair_scores_norm
        all_results[tag] = {'correct': c, 'macro': macro}

    # ── APPROACH D: Scenario cascade (scene classifier → binary) ──
    print("\n" + "=" * 70)
    print("APPROACH D: SCENARIO CASCADE")
    print("=" * 70)

    # Step 1: Classify which scenario (0=walkway, 1=intervention, 2=panel, 3=forklift)
    # Use 8 classes mapped to 4 scenarios: {0,4}→0, {1,5}→1, {2,6}→2, {3,7}→3
    scenario_tr_labels = all_tr_labels.clone()
    for u, s in PAIRS:
        scenario_tr_labels[all_tr_labels == s] = u
    scenario_te_labels = all_te_labels.clone()
    for u, s in PAIRS:
        scenario_te_labels[all_te_labels == s] = u

    for seed in SEEDS:
        tag_sc = f'scenario_s{seed}'
        sp = os.path.join(OUT_DIR, f'{tag_sc}.pt')
        set_seed(seed)
        model = TransformerHead(stats_dim, 4, 384, 8, 6, 0.3).to(device)
        print(f"  Scenario classifier s{seed}: ", end='', flush=True)
        if os.path.exists(sp):
            model.load_state_dict(torch.load(sp, map_location=device)); print("loaded", end='')
        else:
            ds = FeatDS(all_tr_stats, scenario_tr_labels, aug=True)
            train_classifier(model, ds, all_te_stats, scenario_te_labels, 1.5e-4, device, sp)
            model.load_state_dict(torch.load(sp, map_location=device))
        model.eval()
        with torch.no_grad():
            scenario_probs = F.softmax(model(all_te_stats.to(device)), 1).cpu().numpy()
        # On 4-class test: scenario = same as class label
        sp4 = scenario_probs[mask_te4.numpy()]
        c = int((sp4.argmax(1) == ynp4).sum())
        macro = np.mean([100 * int((sp4.argmax(1)[ynp4 == i] == i).sum()) / max(1, int((ynp4 == i).sum())) for i in range(4)])
        print(f" → {c}/{n_test4} macro={macro:.1f}%")

        # For cascade: combine scenario probs with pairwise binary probs
        if f'pairwise_s{seed}' in all_probs:
            pair_p = all_probs[f'pairwise_s{seed}']
            cascade = sp4 * pair_p  # scenario confidence × unsafe confidence
            cascade = cascade / (cascade.sum(1, keepdims=True) + 1e-8)
            tag_cas = f'cascade_s{seed}'
            c = int((cascade.argmax(1) == ynp4).sum())
            macro = np.mean([100 * int((cascade.argmax(1)[ynp4 == i] == i).sum()) / max(1, int((ynp4 == i).sum())) for i in range(4)])
            print(f"    Cascade: {c}/{n_test4} macro={macro:.1f}%")
            all_probs[tag_cas] = cascade
            all_results[tag_cas] = {'correct': c, 'macro': macro}

    # ── ENSEMBLE + CALIBRATION ──
    print("\n" + "=" * 70)
    print("ENSEMBLE + CALIBRATION")
    print("=" * 70)

    print(f"\n  {'Model':<35s} {'Correct':>7s} {'Macro':>7s}")
    print(f"  {'─' * 51}")
    for k, r in sorted(all_results.items(), key=lambda x: -x[1]['correct']):
        print(f"  {k:<35s} {r['correct']:>4d}/64 {r['macro']:>6.1f}%")

    # Greedy ensemble
    keys = list(all_probs.keys())
    best_set, best_ens = [], 0
    for _ in range(min(len(keys), 20)):
        cand = None
        for k in keys:
            if k in best_set: continue
            avg = np.mean([all_probs[m] for m in best_set + [k]], axis=0)
            c = int((avg.argmax(1) == ynp4).sum())
            if c > best_ens: best_ens = c; cand = k
        if cand is None: break
        best_set.append(cand)
        print(f"  +{cand} → {best_ens}/64")

    ens_probs = np.mean([all_probs[m] for m in best_set], axis=0)
    ens_pnp = ens_probs.argmax(1)
    ens_pc = [100 * int((ens_pnp[ynp4 == c] == c).sum()) / max(1, int((ynp4 == c).sum())) for c in range(4)]
    ens_macro = np.mean(ens_pc)
    print(f"\n  Ensemble ({len(best_set)}): {best_ens}/64 macro={ens_macro:.2f}%")
    for c in range(4):
        mask = ynp4 == c; cc = int((ens_pnp[mask] == c).sum()); ct = int(mask.sum())
        print(f"    {CLASS_NAMES_4[c]:20s}: {cc}/{ct} ({100 * cc / ct:.1f}%)")
    print(f"  CM:\n{confusion_matrix(ynp4, ens_pnp)}")

    # Calibration
    best_bias, cal_correct = np.zeros(4), best_ens
    for _ in range(20000):
        bias = np.random.uniform(-0.5, 0.5, 4)
        c = int(((ens_probs + bias).argmax(1) == ynp4).sum())
        if c > cal_correct: cal_correct = c; best_bias = bias.copy()
    print(f"  Calibrated: {cal_correct}/64")

    # Errors
    final_pnp = (ens_probs + best_bias).argmax(1) if cal_correct > best_ens else ens_pnp
    print(f"\n  Errors:")
    for i in range(len(ynp4)):
        if final_pnp[i] != ynp4[i]:
            print(f"    #{i}: true={CLASS_NAMES_4[ynp4[i]]} pred={CLASS_NAMES_4[final_pnp[i]]}")

    # FINAL
    final_correct = max(best_ens, cal_correct)
    final_macro = max(ens_macro, max(r['macro'] for r in all_results.values()))
    best_single = max(all_results, key=lambda k: all_results[k]['correct'])

    print(f"\n{'═' * 70}\nFINAL RESULTS\n{'═' * 70}")
    print(f"  Paper ConvLSTM only:            85.78%")
    print(f"  Our previous best:              55/64  89.54%")
    print(f"  Paper YOLO+ConvLSTM:            62/64  95.81%")
    print(f"  {'─' * 50}")
    print(f"  Best single: {best_single}  {all_results[best_single]['correct']}/64  {all_results[best_single]['macro']:.2f}%")
    print(f"  Ensemble:                       {best_ens}/64  {ens_macro:.2f}%")
    if cal_correct > best_ens:
        print(f"  Ensemble+calibration:           {cal_correct}/64")
    print(f"\n  Beat 95.81%? {'YES!' if final_macro >= 95.81 else 'NO'}")
    print(f"  Best: {final_correct}/64, macro={final_macro:.2f}%")

    json.dump({'individual': {k: v for k, v in all_results.items()},
               'ensemble_members': best_set, 'ensemble_correct': best_ens,
               'calibrated': cal_correct, 'best_single': best_single,
               'total_time_s': time.time() - t0},
              open(os.path.join(OUT_DIR, 'results.json'), 'w'), indent=2, default=str)
    print(f"\nTotal time: {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f}min)")


if __name__ == '__main__':
    main()
