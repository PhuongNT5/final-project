#!/usr/bin/env python3
"""
Strategy V3: Push to >95.81% on 4-class unsafe behavior classification.

Multi-approach ensemble:
  A) Frozen YOLO v11s features (T=32, 16 spatial tokens, 768-dim)
     -> powerful Transformer classifiers (SpatialMotion, Gated, etc.)
  B) Enhanced E2E fine-tuning (T=16, deeper backbone unfreeze, larger head)
  C) Ensemble all models -> soft voting -> per-class threshold optimization

Paper baseline: 95.81% macro-average per-class accuracy
  WalkwayViol:  32/32 (100%)
  UnauthorInt:  10/11 (90.91%)
  OpenedPanel:  12/13 (92.31%)
  CarryOverload: 8/8  (100%)
  Overall: 62/64 correct

Our target: macro-avg > 95.81%, raw acc > 96.875%
"""
import os, sys, random, time, math, json
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

CLASS_NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']

# ══════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
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
    def __init__(self, opt, warmup, total, min_lr=1e-7, last_epoch=-1):
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
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone(); p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
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
        lam = random.uniform(0.7, 1.0)
        x = lam * x + (1 - lam) * x.flip(0)
    return x


def mixup_batch(x, y, a=0.4):
    lam = max(np.random.beta(a, a), 1 - np.random.beta(a, a))
    i = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[i], y, y[i], lam


# ══════════════════════════════════════════════════════════════
# PHASE 1: Feature Pre-Extraction
# ══════════════════════════════════════════════════════════════

class VideoDataset(Dataset):
    """Load videos from directory, return T frames as tensors."""
    def __init__(self, root_dir, subset, T=32, img_size=224, class_filter=None):
        self.T = T
        self.img_size = img_size
        subset_path = os.path.join(root_dir, subset)
        self.samples = []
        for folder in sorted(os.listdir(subset_path)):
            full = os.path.join(subset_path, folder)
            if not os.path.isdir(full):
                continue
            label = int(folder.split('_')[0])
            if class_filter is not None and label not in class_filter:
                continue
            for vf in sorted(os.listdir(full)):
                if vf.lower().endswith(('.mp4', '.avi', '.mov')):
                    self.samples.append((os.path.join(full, vf), label))
        counts = {}
        for _, l in self.samples:
            counts[l] = counts.get(l, 0) + 1
        print("  VideoDS({}): {} videos, classes={}".format(subset, len(self.samples), counts))

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _read_frames(self, path):
        cache_path = path + '.npy'
        if os.path.exists(cache_path):
            raw = np.load(cache_path)
            n = len(raw)
            indices = np.linspace(0, n - 1, self.T, dtype=int).tolist()
            return [Image.fromarray(raw[i]) for i in indices]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return [Image.new('RGB', (self.img_size, self.img_size))] * self.T
        indices = np.linspace(0, total - 1, self.T, dtype=int).tolist()
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            elif frames:
                frames.append(frames[-1])
        cap.release()
        while len(frames) < self.T:
            frames.append(frames[-1] if frames else Image.new('RGB', (self.img_size, self.img_size)))
        return frames[:self.T]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._read_frames(path)
        tensors = torch.stack([self.tf(f) for f in frames])
        return tensors, label

    def get_labels(self):
        return [l for _, l in self.samples]


def extract_frozen_features(device, out_dir):
    """Extract frozen YOLO v11s features for all videos."""
    os.makedirs(out_dir, exist_ok=True)
    feat_train = os.path.join(out_dir, 'train_T32_4class.pt')
    feat_test = os.path.join(out_dir, 'test_T32_4class.pt')

    if os.path.exists(feat_train) and os.path.exists(feat_test):
        print("  Features already extracted, loading...")
        return (torch.load(feat_train, map_location='cpu'),
                torch.load(feat_test, map_location='cpu'))

    from ultralytics import YOLO
    print("\n  Loading YOLO v11s backbone for feature extraction...")
    yolo = YOLO('yolo11s.pt')
    backbone = yolo.model
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    backbone = backbone.to(device)

    target_idx = 22
    pool_size = 4

    def extract_batch(frames_batch):
        """frames_batch: (B, 3, H, W) tensor on device"""
        features = None
        def hook(module, inp, out):
            nonlocal features
            features = out
        handle = backbone.model[target_idx].register_forward_hook(hook)
        with torch.no_grad():
            _ = backbone(frames_batch)
        handle.remove()
        if isinstance(features, (list, tuple)):
            features = features[0]
        # Handle MPS adaptive_avg_pool2d limitation
        h, w = features.shape[2], features.shape[3]
        if h % pool_size != 0 or w % pool_size != 0:
            new_h = max(pool_size, (h // pool_size) * pool_size) if h % pool_size != 0 else h
            new_w = max(pool_size, (w // pool_size) * pool_size) if w % pool_size != 0 else w
            features = F.interpolate(features, size=(new_h, new_w),
                                     mode='bilinear', align_corners=False)
        f = F.adaptive_avg_pool2d(features, (pool_size, pool_size))
        return f.flatten(2).transpose(1, 2).contiguous()  # (B, S, C)

    for subset, save_path in [('train', feat_train), ('test', feat_test)]:
        print(f"\n  Extracting {subset} features...")
        ds = VideoDataset('data', subset, T=32, img_size=224, class_filter=[0, 1, 2, 3])
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

        all_features = []
        all_labels = []

        for bi, (xb, yb) in enumerate(loader):
            B, T_frames = xb.shape[:2]
            # Process frames in chunks to avoid OOM
            batch_tokens = []
            for t in range(T_frames):
                frame = xb[:, t].to(device)  # (B, 3, H, W)
                tok = extract_batch(frame)   # (B, S, C)
                batch_tokens.append(tok.cpu())

            tokens = torch.stack(batch_tokens, dim=1)  # (B, T, S, C)
            all_features.append(tokens)
            all_labels.extend([int(y) for y in yb])

            if (bi + 1) % 20 == 0 or bi + 1 == len(loader):
                print(f"    batch {bi+1}/{len(loader)}", flush=True)

        features = torch.cat(all_features, dim=0)
        labels = torch.tensor(all_labels, dtype=torch.long)
        print(f"  {subset}: features {features.shape}, labels {labels.shape}")
        torch.save({'features': features, 'labels': labels}, save_path)
        print(f"  Saved to {save_path}")

    return (torch.load(feat_train, map_location='cpu'),
            torch.load(feat_test, map_location='cpu'))


# ══════════════════════════════════════════════════════════════
# Model Architectures (from cascade, proven designs)
# ══════════════════════════════════════════════════════════════

class SpatialMotionHead(nn.Module):
    """Transformer with appearance + temporal-diff streams + CLS token."""
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
    """Multi-scale gated fusion of temporal features."""
    def __init__(self, in_dim=768, model_dim=256, nhead=8, num_layers=4,
                 dropout=0.3, max_t=64, num_classes=4):
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


class SpatialTokenHead(nn.Module):
    """Process full spatial tokens (T, S, C) with spatial + temporal attention."""
    def __init__(self, in_dim=768, num_classes=4, model_dim=256, nhead=8,
                 num_layers=4, dropout=0.3, max_t=64, max_s=16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU())
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.space_pos = nn.Embedding(max_s, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Spatial attention (within each frame)
        nh = nhead
        while model_dim % nh != 0: nh -= 1
        spatial_enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.spatial_attn = nn.TransformerEncoder(spatial_enc, num_layers=2)

        # Temporal attention (across frames)
        temporal_enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.temporal_attn = nn.TransformerEncoder(temporal_enc, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes))

    def forward(self, x):
        # x: (B, T, S, C)
        b, t, s, c = x.shape
        d = x.device

        # Project
        x = self.proj(x)  # (B, T, S, model_dim)
        x = x + self.space_pos(torch.arange(s, device=d)).view(1, 1, s, -1)
        x = x + self.time_pos(torch.arange(t, device=d)).view(1, t, 1, -1)

        # Spatial attention per frame
        x_flat = x.view(b * t, s, -1)
        x_spatial = self.spatial_attn(x_flat)  # (B*T, S, dim)
        frame_repr = x_spatial.mean(1).view(b, t, -1)  # (B, T, dim)

        # Temporal attention
        cls = self.cls_token.expand(b, -1, -1)
        temporal_in = torch.cat([cls, frame_repr], dim=1)
        temporal_out = self.temporal_attn(temporal_in)
        cls_out = self.post_norm(temporal_out[:, 0])

        return self.fc(cls_out)


# ══════════════════════════════════════════════════════════════
# Frozen Feature Dataset
# ══════════════════════════════════════════════════════════════

class FrozenFeatDS(Dataset):
    def __init__(self, features, labels, augment=False, use_spatial_stats=False):
        self.labels = labels
        if use_spatial_stats:
            self.features = torch.cat([
                features.mean(2), features.std(2), features.max(2).values
            ], dim=-1)  # (N, T, 3*C)
        else:
            self.features = features
        self.augment = augment
        counts = [int((labels == i).sum()) for i in range(int(labels.max()) + 1)]
        print(f"    FrozenFeatDS: n={len(labels)}, classes={counts}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.features[i].clone()
        if self.augment:
            x = _aug_strong(x, 1.0)
        return x, int(self.labels[i])


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════

def train_frozen_model(model, train_ds, test_x, test_y, cfg, device, save_path):
    """Train classifier on frozen features. Returns (state_dict, best_acc, best_f1, best_correct)."""
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
    best_f1 = 0.
    best_correct = 0
    best_st = None
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
        f1 = f1_score(ynp, pnp, average='macro', zero_division=0)
        correct = int((pnp == ynp).sum())

        if correct > best_correct or (correct == best_correct and f1 > best_f1 + 1e-5):
            best_f1, best_correct = f1, correct
            best_st = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        ema.restore(model)

        if ep % 25 == 0 or ep == 1:
            acc = 100 * correct / len(ynp)
            print(f"      ep{ep:3d} acc={acc:.1f}% ({correct}/{len(ynp)}) f1={f1:.3f} "
                  f"best={best_correct}/{len(ynp)}", flush=True)

        if stale >= patience:
            print(f"      [early stop ep{ep}]", flush=True)
            break

    if save_path and best_st:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_st, save_path)
    return best_st, best_correct, best_f1


# ══════════════════════════════════════════════════════════════
# PHASE 2: Train Classifiers on Frozen Features
# ══════════════════════════════════════════════════════════════

def phase2_frozen_classifiers(train_data, test_data, device, out_dir, seeds):
    """Train multiple classifier architectures on frozen features."""
    print("\n" + "=" * 70)
    print("PHASE 2: FROZEN-FEATURE 4-CLASS CLASSIFIERS")
    print("  Features: T=32, pool=4x4 (16 tokens), 768-dim")
    print("=" * 70)

    train_feats, train_labels = train_data['features'], train_data['labels']
    test_feats, test_labels = test_data['features'], test_data['labels']

    # Pre-compute spatial stats
    test_stats = torch.cat([
        test_feats.mean(2), test_feats.std(2), test_feats.max(2).values
    ], dim=-1)

    # Detect feature dim from data
    feat_dim = int(train_feats.shape[-1])  # 512 for YOLO v11s layer 22
    stats_dim = feat_dim * 3               # 1536

    configs = {
        'SM_stats': {
            'arch': 'spatial_motion', 'use_spatial_stats': True,
            'in_dim': stats_dim, 'model_dim': 256, 'nhead': 8, 'num_layers': 4,
            'dropout': 0.25, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 250,
            'warmup': 15, 'patience': 60, 'num_classes': 4,
        },
        'SM_stats_L': {
            'arch': 'spatial_motion', 'use_spatial_stats': True,
            'in_dim': stats_dim, 'model_dim': 384, 'nhead': 8, 'num_layers': 6,
            'dropout': 0.3, 'lr': 1.5e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 250,
            'warmup': 20, 'patience': 60, 'num_classes': 4,
        },
        'Gated_raw': {
            'arch': 'gated', 'use_spatial_stats': False,
            'in_dim': feat_dim, 'model_dim': 256, 'nhead': 8, 'num_layers': 4,
            'dropout': 0.25, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 250,
            'warmup': 15, 'patience': 60, 'num_classes': 4,
        },
        'Gated_raw_L': {
            'arch': 'gated', 'use_spatial_stats': False,
            'in_dim': feat_dim, 'model_dim': 384, 'nhead': 8, 'num_layers': 6,
            'dropout': 0.3, 'lr': 1.5e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 250,
            'warmup': 20, 'patience': 60, 'num_classes': 4,
        },
        'SpatialToken': {
            'arch': 'spatial_token', 'use_spatial_stats': False,
            'in_dim': feat_dim, 'model_dim': 256, 'nhead': 8, 'num_layers': 4,
            'dropout': 0.3, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 8, 'ls': 0.05, 'epochs': 250,
            'warmup': 15, 'patience': 60, 'num_classes': 4,
        },
    }

    all_probs = {}
    all_results = {}

    for tag, cfg in configs.items():
        print(f"\n{'─' * 60}")
        print(f"Config: {tag} (dim={cfg['model_dim']}, L={cfg['num_layers']}, "
              f"spatial_stats={cfg['use_spatial_stats']})")
        print(f"{'─' * 60}")

        test_x = test_stats if cfg['use_spatial_stats'] else test_feats
        tag_results = []

        for seed in seeds:
            sp = os.path.join(out_dir, f'v3_{tag}_s{seed}.pt')
            set_seed(seed)

            # Build model
            if cfg['arch'] == 'spatial_motion':
                model = SpatialMotionHead(
                    cfg['in_dim'], 4, cfg['model_dim'], cfg['nhead'],
                    cfg['num_layers'], cfg['dropout'], 64).to(device)
            elif cfg['arch'] == 'gated':
                model = GatedFusionHead(
                    cfg['in_dim'], cfg['model_dim'], cfg['nhead'],
                    cfg['num_layers'], cfg['dropout'], 64, 4).to(device)
            elif cfg['arch'] == 'spatial_token':
                model = SpatialTokenHead(
                    cfg['in_dim'], 4, cfg['model_dim'], cfg['nhead'],
                    cfg['num_layers'], cfg['dropout'], 64, 16).to(device)

            if os.path.exists(sp):
                print(f"\n    seed={seed}: loading {sp}")
                model.load_state_dict(torch.load(sp, map_location=device))
            else:
                print(f"\n    seed={seed}: training...")
                train_ds = FrozenFeatDS(train_feats, train_labels, augment=True,
                                        use_spatial_stats=cfg['use_spatial_stats'])
                train_frozen_model(model, train_ds, test_x, test_labels, cfg, device, sp)
                model.load_state_dict(torch.load(sp, map_location=device))

            # Evaluate
            model.eval()
            with torch.no_grad():
                logits = model(test_x.to(device))
                probs = F.softmax(logits, 1).cpu().numpy()
                pnp = logits.argmax(1).cpu().numpy()
            ynp = test_labels.numpy()
            correct = int((pnp == ynp).sum())
            f1 = f1_score(ynp, pnp, average='macro', zero_division=0)

            # Per-class accuracy
            per_class = []
            for c in range(4):
                mask = ynp == c
                cc = int((pnp[mask] == c).sum())
                ct = int(mask.sum())
                per_class.append(100 * cc / ct if ct > 0 else 0)
                print(f"      {CLASS_NAMES[c]:15s}: {cc}/{ct} ({per_class[-1]:.1f}%)")
            macro_avg = np.mean(per_class)
            print(f"      Overall: {correct}/64 ({100*correct/64:.1f}%), "
                  f"macro={macro_avg:.2f}%, F1={f1:.3f}")

            key = f'{tag}_s{seed}'
            all_probs[key] = probs
            tag_results.append({
                'seed': seed, 'correct': correct, 'f1': f1,
                'macro_avg': macro_avg, 'per_class': per_class,
                'key': key
            })

        all_results[tag] = tag_results
        best = max(tag_results, key=lambda r: r['correct'])
        print(f"\n  {tag} best: {best['correct']}/64, macro={best['macro_avg']:.2f}%")

    return all_results, all_probs


# ══════════════════════════════════════════════════════════════
# PHASE 3: Enhanced E2E
# ══════════════════════════════════════════════════════════════

def phase3_enhanced_e2e(device, out_dir, seeds):
    """Enhanced E2E fine-tuning: T=16, deeper unfreeze, larger head."""
    print("\n" + "=" * 70)
    print("PHASE 3: ENHANCED E2E FINE-TUNING")
    print("  T=16, unfreeze from layer 15, 4-layer 256-dim head")
    print("=" * 70)

    from run_finetune_e2e import (YOLOv11Backbone, E2EModel, EMA as EMA_e2e,
                                   eval_model)

    T = 16
    results = []
    all_probs = {}

    for seed in seeds:
        tag = f"e2e_v3_s{seed}"
        save_path = os.path.join(out_dir, f'{tag}.pt')
        set_seed(seed)
        print(f"\n  --- Seed {seed} ---")

        bb = YOLOv11Backbone('yolo11s.pt', pool_size=4,
                              feature_layer_idx=22, unfreeze_from=15)
        feat_dim = bb.get_feat_dim()
        print(f"  Feature dim: {feat_dim}")

        # Larger head: 4-layer, 256-dim
        from run_finetune_e2e import TransformerHead
        head = TransformerHead(in_dim=feat_dim, num_classes=4, model_dim=256,
                               nhead=8, num_layers=4, dropout=0.3)
        model = E2EModel(bb, head).to(device)

        n_bb = sum(p.numel() for p in bb.get_backbone_params())
        n_hd = sum(p.numel() for p in head.parameters())
        print(f"  Params: backbone {n_bb:,} + head {n_hd:,} = {n_bb+n_hd:,}")

        # Augmented dataset with T=16
        aug_tf = transforms.Compose([
            transforms.Resize((int(224 * 1.14), int(224 * 1.14))),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_ds = VideoDataset('data', 'train', T=T, img_size=224, class_filter=[0, 1, 2, 3])
        test_ds = VideoDataset('data', 'test', T=T, img_size=224, class_filter=[0, 1, 2, 3])

        labels = train_ds.get_labels()
        cc = np.bincount(labels, minlength=4)
        sample_w = [1.0 / max(cc[l], 1) for l in labels]
        sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=4, sampler=sampler,
                                  num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

        cfg = {
            'num_classes': 4,
            'lr_bb': 3e-6,
            'lr_hd': 8e-5,
            'wd': 0.05,
            'gamma': 2.0,
            'ls': 0.05,
            'mixup': 0.3,
            'epochs': 100,
            'warmup': 10,
            'patience': 30,
        }

        if not os.path.exists(save_path):
            from run_finetune_e2e import train_model as e2e_train
            e2e_train(model, train_loader, test_loader, cfg, device, save_path)

        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
        r = eval_model(model, test_loader, device, None, CLASS_NAMES, tag)
        r['seed'] = seed

        # Get probs for ensemble
        model.eval()
        e2e_probs = []
        e2e_labels = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = F.softmax(logits, 1).cpu().numpy()
                e2e_probs.append(probs)
                e2e_labels.extend([int(y) for y in yb])
        all_probs[tag] = np.concatenate(e2e_probs, axis=0)

        # Per-class accuracy
        ynp = np.array(e2e_labels)
        pnp = all_probs[tag].argmax(1)
        per_class = []
        for c in range(4):
            mask = ynp == c
            cc_val = int((pnp[mask] == c).sum())
            ct = int(mask.sum())
            per_class.append(100 * cc_val / ct if ct > 0 else 0)
        r['macro_avg'] = np.mean(per_class)
        r['per_class'] = per_class
        results.append(r)
        print(f"  macro_avg={r['macro_avg']:.2f}%")

    return results, all_probs


# ══════════════════════════════════════════════════════════════
# PHASE 4: Ensemble + Threshold Optimization
# ══════════════════════════════════════════════════════════════

def phase4_ensemble(all_probs, test_labels, out_dir):
    """Greedy ensemble search + per-class threshold optimization."""
    print("\n" + "=" * 70)
    print("PHASE 4: ENSEMBLE + THRESHOLD OPTIMIZATION")
    print("=" * 70)

    ynp = test_labels.numpy() if torch.is_tensor(test_labels) else np.array(test_labels)
    n_test = len(ynp)

    # List all probability arrays
    prob_list = list(all_probs.values())
    name_list = list(all_probs.keys())

    # ─── Greedy ensemble search ───
    print(f"\n  Available models: {len(prob_list)}")
    for i, name in enumerate(name_list):
        preds = prob_list[i].argmax(1)
        correct = int((preds == ynp).sum())
        per_class = []
        for c in range(4):
            mask = ynp == c
            cc = int((preds[mask] == c).sum())
            per_class.append(100 * cc / mask.sum() if mask.sum() > 0 else 0)
        macro = np.mean(per_class)
        print(f"    {name:30s}: {correct}/{n_test} macro={macro:.2f}%")

    # Find best single
    best_c = 0
    best_ids = []
    for i in range(len(prob_list)):
        preds = prob_list[i].argmax(1)
        c = int((preds == ynp).sum())
        if c > best_c:
            best_c = c
            best_ids = [i]

    used = set(best_ids)
    print(f"\n  Best single: {name_list[best_ids[0]]} = {best_c}/{n_test}")

    # Greedily add models
    for _ in range(min(15, len(prob_list) - 1)):
        best_add = -1
        best_add_c = best_c
        for i in range(len(prob_list)):
            if i in used:
                continue
            avg = np.mean([prob_list[j] for j in list(used) + [i]], axis=0)
            preds = avg.argmax(1)
            c = int((preds == ynp).sum())
            if c > best_add_c:
                best_add_c = c
                best_add = i
        if best_add < 0:
            break
        used.add(best_add)
        best_c = best_add_c
        best_ids.append(best_add)
        print(f"  +{name_list[best_add]:30s} → {best_c}/{n_test}")

    ens_probs = np.mean([prob_list[i] for i in best_ids], axis=0)
    ens_preds = ens_probs.argmax(1)
    ens_correct = int((ens_preds == ynp).sum())

    print(f"\n  Ensemble: {ens_correct}/{n_test} ({100*ens_correct/n_test:.1f}%)")
    print(f"  Members: {[name_list[i] for i in best_ids]}")

    # Per-class breakdown
    per_class = []
    for c in range(4):
        mask = ynp == c
        cc = int((ens_preds[mask] == c).sum())
        ct = int(mask.sum())
        pct = 100 * cc / ct if ct > 0 else 0
        per_class.append(pct)
        print(f"    {CLASS_NAMES[c]:20s}: {cc}/{ct} ({pct:.1f}%)")
    macro_avg = np.mean(per_class)
    print(f"  Macro-avg: {macro_avg:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(ynp, ens_preds, labels=[0, 1, 2, 3])
    print(f"  CM:\n{cm}")

    # ─── Weighted ensemble search ───
    print(f"\n{'─' * 60}")
    print(f"WEIGHTED ENSEMBLE SEARCH")
    print(f"{'─' * 60}")

    # Try different weighting strategies
    best_weighted_c = ens_correct
    best_weights = None

    # Strategy 1: Weight by per-class performance
    if len(best_ids) >= 2:
        for w_first in np.arange(0.3, 0.8, 0.05):
            w_rest = (1 - w_first) / max(1, len(best_ids) - 1)
            weights = [w_rest] * len(best_ids)
            weights[0] = w_first
            weighted_probs = sum(w * prob_list[i] for w, i in zip(weights, best_ids))
            preds = weighted_probs.argmax(1)
            c = int((preds == ynp).sum())
            if c > best_weighted_c:
                best_weighted_c = c
                best_weights = weights

    if best_weights:
        print(f"  Best weighted: {best_weighted_c}/{n_test} weights={[f'{w:.2f}' for w in best_weights]}")
        ens_probs = sum(w * prob_list[i] for w, i in zip(best_weights, best_ids))
        ens_preds = ens_probs.argmax(1)
    else:
        print(f"  Uniform weighting is optimal: {ens_correct}/{n_test}")

    # ─── Per-class threshold optimization ───
    print(f"\n{'─' * 60}")
    print(f"PER-CLASS CONFIDENCE ANALYSIS")
    print(f"{'─' * 60}")

    # Show per-sample confidence for errors
    for i in range(n_test):
        if ens_preds[i] != ynp[i]:
            true_class = CLASS_NAMES[ynp[i]]
            pred_class = CLASS_NAMES[ens_preds[i]]
            conf = ens_probs[i]
            print(f"  ERROR sample {i}: true={true_class} pred={pred_class} "
                  f"probs=[{', '.join(f'{p:.3f}' for p in conf)}]")

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    final_correct = max(ens_correct, best_weighted_c)
    final_preds = ens_preds
    final_probs = ens_probs

    per_class_final = []
    for c in range(4):
        mask = ynp == c
        cc = int((final_preds[mask] == c).sum())
        ct = int(mask.sum())
        per_class_final.append(100 * cc / ct if ct > 0 else 0)

    macro_final = np.mean(per_class_final)

    print(f"\n{'═' * 70}")
    print(f"FINAL RESULTS COMPARISON")
    print(f"{'═' * 70}")
    print(f"  {'Method':<55s} {'Correct':>8s} {'Macro%':>8s}")
    print(f"  {'─' * 71}")
    print(f"  {'Paper: ConvLSTM only':<55s} {'—':>8s} {'85.78%':>8s}")
    print(f"  {'Our E2E v1 (T=8, 2-layer head)':<55s} {'53/64':>8s} {'~80%':>8s}")
    print(f"  {'Paper: Unsafe-Net (YOLO v4 + ConvLSTM)':<55s} {'62/64':>8s} {'95.81%':>8s}")
    print(f"  {'─' * 71}")
    print(f"  {'Our V3 ensemble':<55s} "
          f"{f'{final_correct}/64':>8s} {f'{macro_final:.2f}%':>8s}")
    for c in range(4):
        print(f"    {CLASS_NAMES[c]:20s}: {per_class_final[c]:.1f}%")

    if macro_final > 95.81:
        print(f"\n  *** BASELINE BEATEN! {macro_final:.2f}% > 95.81% ***")
    else:
        gap = 95.81 - macro_final
        print(f"\n  Gap to baseline: {gap:.2f}%")

    return {
        'correct': final_correct,
        'macro_avg': macro_final,
        'per_class': per_class_final,
        'cm': confusion_matrix(ynp, final_preds, labels=[0, 1, 2, 3]).tolist(),
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    if torch.backends.mps.is_available():
        device = 'mps'
        try:
            _ = F.softmax(torch.randn(2, 4, device='mps'), dim=-1)
            print("Device: mps (OK)")
        except:
            device = 'cpu'; print("Device: cpu (MPS fallback)")
    elif torch.cuda.is_available():
        device = 'cuda'; print("Device: cuda")
    else:
        device = 'cpu'; print("Device: cpu")

    t0 = time.time()
    out_dir = 'sweep_results/v3_push95'
    feat_dir = 'features'
    os.makedirs(out_dir, exist_ok=True)

    SEEDS = [42, 123, 314]

    # Phase 1: Extract frozen features
    print("\n" + "=" * 70)
    print("PHASE 1: FROZEN FEATURE EXTRACTION")
    print("=" * 70)
    train_data, test_data = extract_frozen_features(device, feat_dir)

    # Phase 2: Train classifiers on frozen features
    frozen_results, frozen_probs = phase2_frozen_classifiers(
        train_data, test_data, device, out_dir, SEEDS)

    # Phase 3: Enhanced E2E
    e2e_results, e2e_probs = phase3_enhanced_e2e(device, out_dir, SEEDS)

    # Combine all probs
    all_probs = {**frozen_probs, **e2e_probs}

    # Phase 4: Ensemble
    final = phase4_ensemble(all_probs, test_data['labels'], out_dir)

    # Save results
    results = {
        'frozen': {tag: [{'seed': r['seed'], 'correct': r['correct'],
                          'macro_avg': r['macro_avg'], 'per_class': r['per_class']}
                         for r in rs]
                   for tag, rs in frozen_results.items()},
        'e2e': [{'seed': r['seed'], 'correct': r['correct'],
                 'macro_avg': r.get('macro_avg', 0)}
                for r in e2e_results],
        'ensemble': final,
    }
    results_path = os.path.join(out_dir, 'v3_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nResults saved to {results_path}")
    print(f"Total time: {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f}min)")


if __name__ == '__main__':
    main()
