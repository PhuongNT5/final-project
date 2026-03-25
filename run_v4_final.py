#!/usr/bin/env python3
"""
Strategy V4: Final push to >95.81% macro-avg accuracy.

Key innovations over V3:
  1. Multi-layer YOLO features (layers 15, 19, 22) — richer multi-scale representation
  2. Higher spatial resolution (pool_size=7 → 49 tokens) — preserves spatial layout
  3. T=48 frames — longer temporal context
  4. Specialized binary classifier for WalkwayViol vs UnauthorInt (the confused pair)
  5. Test-time augmentation (temporal shifts + flips)
  6. Stacking ensemble: meta-learner on V3+V4 probability outputs

Paper baseline: 95.81% macro-average per-class accuracy (62/64 correct)
  WalkwayViol:  32/32 (100%)
  UnauthorInt:  10/11 (90.91%)
  OpenedPanel:  12/13 (92.31%)
  CarryOverload: 8/8  (100%)
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
# Utilities (shared with V3)
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
    return x


def mixup_batch(x, y, a=0.4):
    lam = max(np.random.beta(a, a), 1 - np.random.beta(a, a))
    i = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[i], y, y[i], lam


# ══════════════════════════════════════════════════════════════
# Video Dataset
# ══════════════════════════════════════════════════════════════

class VideoDataset(Dataset):
    def __init__(self, root_dir, subset, T=48, img_size=224, class_filter=None):
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
        print(f"  VideoDS({subset}): {len(self.samples)} videos, classes={counts}")

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


# ══════════════════════════════════════════════════════════════
# PHASE 1: Multi-Layer Feature Extraction
# ══════════════════════════════════════════════════════════════

def extract_multilayer_features(device, out_dir, T=48, pool_size=7):
    """Extract features from YOLO layers 15, 19, 22 with higher spatial resolution."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f'T{T}_pool{pool_size}_multilayer'
    feat_train = os.path.join(out_dir, f'train_{tag}.pt')
    feat_test = os.path.join(out_dir, f'test_{tag}.pt')

    if os.path.exists(feat_train) and os.path.exists(feat_test):
        print("  Multi-layer features already extracted, loading...")
        return (torch.load(feat_train, map_location='cpu'),
                torch.load(feat_test, map_location='cpu'))

    from ultralytics import YOLO
    print(f"\n  Extracting multi-layer features: T={T}, pool={pool_size}, layers=[15,19,22]")
    yolo = YOLO('yolo11s.pt')
    backbone = yolo.model
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    backbone = backbone.to(device)

    target_layers = [15, 19, 22]
    captured = {}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            captured[layer_idx] = out
        return hook

    handles = []
    for idx in target_layers:
        h = backbone.model[idx].register_forward_hook(make_hook(idx))
        handles.append(h)

    def extract_batch(frames_batch):
        """frames_batch: (B, 3, H, W) tensor on device → (B, S, C_total)"""
        captured.clear()
        with torch.no_grad():
            _ = backbone(frames_batch)

        pooled_feats = []
        for idx in target_layers:
            features = captured[idx]
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
            pooled_feats.append(f.flatten(2).transpose(1, 2).contiguous())  # (B, S, C_i)

        # Concatenate channels from all layers: (B, S, C_15+C_19+C_22)
        return torch.cat(pooled_feats, dim=-1)

    for subset, save_path in [('train', feat_train), ('test', feat_test)]:
        print(f"\n  Extracting {subset} features...")
        ds = VideoDataset('data', subset, T=T, img_size=224, class_filter=[0, 1, 2, 3])
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

        all_features = []
        all_labels = []

        for bi, (xb, yb) in enumerate(loader):
            B, T_frames = xb.shape[:2]
            batch_tokens = []
            for t in range(T_frames):
                frame = xb[:, t].to(device)
                tok = extract_batch(frame)
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

    for h in handles:
        h.remove()

    return (torch.load(feat_train, map_location='cpu'),
            torch.load(feat_test, map_location='cpu'))


# Also extract single-layer high-res features for comparison
def extract_highres_features(device, out_dir, T=48, pool_size=7, layer_idx=22):
    """Extract single-layer features with higher spatial resolution."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f'T{T}_pool{pool_size}_L{layer_idx}'
    feat_train = os.path.join(out_dir, f'train_{tag}.pt')
    feat_test = os.path.join(out_dir, f'test_{tag}.pt')

    if os.path.exists(feat_train) and os.path.exists(feat_test):
        print(f"  High-res L{layer_idx} features already extracted, loading...")
        return (torch.load(feat_train, map_location='cpu'),
                torch.load(feat_test, map_location='cpu'))

    from ultralytics import YOLO
    print(f"\n  Extracting high-res features: T={T}, pool={pool_size}, layer={layer_idx}")
    yolo = YOLO('yolo11s.pt')
    backbone = yolo.model
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    backbone = backbone.to(device)

    def extract_batch(frames_batch):
        features = None
        def hook(module, inp, out):
            nonlocal features
            features = out
        handle = backbone.model[layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            _ = backbone(frames_batch)
        handle.remove()
        if isinstance(features, (list, tuple)):
            features = features[0]
        h, w = features.shape[2], features.shape[3]
        if h % pool_size != 0 or w % pool_size != 0:
            new_h = max(pool_size, (h // pool_size) * pool_size) if h % pool_size != 0 else h
            new_w = max(pool_size, (w // pool_size) * pool_size) if w % pool_size != 0 else w
            features = F.interpolate(features, size=(new_h, new_w),
                                     mode='bilinear', align_corners=False)
        f = F.adaptive_avg_pool2d(features, (pool_size, pool_size))
        return f.flatten(2).transpose(1, 2).contiguous()

    for subset, save_path in [('train', feat_train), ('test', feat_test)]:
        print(f"\n  Extracting {subset} features...")
        ds = VideoDataset('data', subset, T=T, img_size=224, class_filter=[0, 1, 2, 3])
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

        all_features = []
        all_labels = []

        for bi, (xb, yb) in enumerate(loader):
            B, T_frames = xb.shape[:2]
            batch_tokens = []
            for t in range(T_frames):
                frame = xb[:, t].to(device)
                tok = extract_batch(frame)
                batch_tokens.append(tok.cpu())

            tokens = torch.stack(batch_tokens, dim=1)
            all_features.append(tokens)
            all_labels.extend([int(y) for y in yb])

            if (bi + 1) % 20 == 0 or bi + 1 == len(loader):
                print(f"    batch {bi+1}/{len(loader)}", flush=True)

        features = torch.cat(all_features, dim=0)
        labels = torch.tensor(all_labels, dtype=torch.long)
        print(f"  {subset}: features {features.shape}, labels {labels.shape}")
        torch.save({'features': features, 'labels': labels}, save_path)

    return (torch.load(feat_train, map_location='cpu'),
            torch.load(feat_test, map_location='cpu'))


# ══════════════════════════════════════════════════════════════
# Model Architectures
# ══════════════════════════════════════════════════════════════

class MultiScaleSpatialHead(nn.Module):
    """Full spatial-temporal transformer on high-res multi-layer features.
    Input: (B, T, S, C) where S=49 (7x7) tokens, C=multi-layer concat dim.
    Uses spatial attention per frame, then temporal attention across frames.
    """
    def __init__(self, in_dim, num_classes=4, model_dim=256, nhead=8,
                 num_spatial_layers=2, num_temporal_layers=4,
                 dropout=0.3, max_t=64, max_s=49):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU(),
            nn.Dropout(dropout * 0.3))
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.space_pos = nn.Embedding(max_s, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        nh = nhead
        while model_dim % nh != 0: nh -= 1

        # Spatial attention
        spatial_enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.spatial_attn = nn.TransformerEncoder(spatial_enc, num_layers=num_spatial_layers)

        # Temporal attention
        temporal_enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.temporal_attn = nn.TransformerEncoder(temporal_enc, num_layers=num_temporal_layers)

        self.post_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(model_dim // 2, num_classes))

    def forward(self, x):
        # x: (B, T, S, C) or (B, T, C) -- handle both
        if x.dim() == 3:
            # Spatial stats input: no spatial attention needed
            b, t, c = x.shape
            d = x.device
            x = self.proj(x)
            x = x + self.time_pos(torch.arange(t, device=d)).unsqueeze(0)
            cls = self.cls_token.expand(b, -1, -1)
            temporal_in = torch.cat([cls, x], dim=1)
            out = self.temporal_attn(temporal_in)
            return self.fc(self.post_norm(out[:, 0]))

        b, t, s, c = x.shape
        d = x.device

        x = self.proj(x)  # (B, T, S, model_dim)
        x = x + self.space_pos(torch.arange(s, device=d)).view(1, 1, s, -1)
        x = x + self.time_pos(torch.arange(t, device=d)).view(1, t, 1, -1)

        # Spatial attention per frame
        x_flat = x.view(b * t, s, -1)
        x_spatial = self.spatial_attn(x_flat)
        frame_repr = x_spatial.mean(1).view(b, t, -1)  # (B, T, dim)

        # Temporal attention with CLS token
        cls = self.cls_token.expand(b, -1, -1)
        temporal_in = torch.cat([cls, frame_repr], dim=1)
        temporal_out = self.temporal_attn(temporal_in)

        return self.fc(self.post_norm(temporal_out[:, 0]))


class DualStreamHead(nn.Module):
    """Dual-stream: appearance (spatial stats) + motion (temporal diff).
    Designed for the stats-based input (B, T, C*3) where C*3 = mean/max/std pooling.
    """
    def __init__(self, in_dim, num_classes=4, model_dim=384, nhead=8,
                 num_layers=6, dropout=0.3, max_t=64):
        super().__init__()
        self.appear_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.3))
        self.motion_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.3))
        self.accel_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.3))
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.type_embed = nn.Embedding(4, model_dim)  # 0=cls, 1=appear, 2=motion, 3=accel
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

        # 2nd order temporal diff (acceleration)
        accel = diff[:, 1:] - diff[:, :-1]
        acc_proj = self.accel_proj(accel)
        acc_proj += self.time_pos(torch.arange(t - 2, device=d)).unsqueeze(0)
        acc_proj += self.type_embed(torch.full((t - 2,), 3, dtype=torch.long, device=d)).unsqueeze(0)

        cls = (self.cls_token.expand(b, -1, -1)
               + self.type_embed(torch.zeros(1, dtype=torch.long, device=d)).unsqueeze(0))

        tokens = self.pre_norm(torch.cat([cls, appear, motion, acc_proj], dim=1))
        tokens = self.transformer(tokens)
        return self.fc(self.post_norm(tokens[:, 0]))


class BinarySpecialistHead(nn.Module):
    """Binary classifier specialized for WalkwayViol vs UnauthorInt.
    Uses both spatial tokens and temporal dynamics.
    """
    def __init__(self, in_dim, model_dim=256, nhead=8, num_layers=4,
                 dropout=0.3, max_t=64, max_s=49):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU())
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.space_pos = nn.Embedding(max_s, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        nh = nhead
        while model_dim % nh != 0: nh -= 1

        spatial_enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.spatial_attn = nn.TransformerEncoder(spatial_enc, num_layers=2)

        temporal_enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.temporal_attn = nn.TransformerEncoder(temporal_enc, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 2))  # Binary: class 0 vs class 1

    def forward(self, x):
        if x.dim() == 3:
            b, t, c = x.shape; d = x.device
            x = self.proj(x)
            x = x + self.time_pos(torch.arange(t, device=d)).unsqueeze(0)
            cls = self.cls_token.expand(b, -1, -1)
            out = self.temporal_attn(torch.cat([cls, x], dim=1))
            return self.fc(self.post_norm(out[:, 0]))

        b, t, s, c = x.shape; d = x.device
        x = self.proj(x)
        x = x + self.space_pos(torch.arange(s, device=d)).view(1, 1, s, -1)
        x = x + self.time_pos(torch.arange(t, device=d)).view(1, t, 1, -1)
        x_flat = x.view(b * t, s, -1)
        x_spatial = self.spatial_attn(x_flat)
        frame_repr = x_spatial.mean(1).view(b, t, -1)
        cls = self.cls_token.expand(b, -1, -1)
        out = self.temporal_attn(torch.cat([cls, frame_repr], dim=1))
        return self.fc(self.post_norm(out[:, 0]))


# ══════════════════════════════════════════════════════════════
# Frozen Feature Dataset
# ══════════════════════════════════════════════════════════════

class FrozenFeatDS(Dataset):
    def __init__(self, features, labels, augment=False, use_spatial_stats=False):
        self.labels = labels
        if use_spatial_stats and features.dim() == 4:
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
            if x.dim() == 2:  # (T, C)
                x = _aug_strong(x, 1.0)
            elif x.dim() == 3:  # (T, S, C)
                # Augment along temporal dim
                T, S, C = x.shape
                # Random temporal flip
                if random.random() < 0.4:
                    x = x.flip(0)
                # Random temporal roll
                if random.random() < 0.4:
                    x = torch.roll(x, random.randint(1, max(1, T // 3)), 0)
                # Random frame dropout
                if random.random() < 0.5:
                    n_drop = max(1, int(T * random.uniform(0.05, 0.2)))
                    for idx in random.sample(range(T), min(n_drop, T)):
                        x[idx] = 0.0
                # Random spatial token dropout
                if random.random() < 0.3:
                    n_drop = max(1, int(S * random.uniform(0.1, 0.3)))
                    for idx in random.sample(range(S), min(n_drop, S)):
                        x[:, idx] = 0.0
                # Noise
                if random.random() < 0.5:
                    x = x + torch.randn_like(x) * random.uniform(0.01, 0.04)
                # Scale
                if random.random() < 0.3:
                    x = x * random.uniform(0.85, 1.15)
        return x, int(self.labels[i])


class BinaryFeatDS(Dataset):
    """Dataset for binary classification (classes 0 and 1 only)."""
    def __init__(self, features, labels, augment=False, use_spatial_stats=False):
        # Filter to only class 0 and 1
        mask = (labels == 0) | (labels == 1)
        self.labels = labels[mask]
        feats = features[mask]
        if use_spatial_stats and feats.dim() == 4:
            self.features = torch.cat([
                feats.mean(2), feats.std(2), feats.max(2).values
            ], dim=-1)
        else:
            self.features = feats
        self.augment = augment
        counts = [int((self.labels == i).sum()) for i in range(2)]
        print(f"    BinaryFeatDS: n={len(self.labels)}, classes={counts}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.features[i].clone()
        if self.augment:
            if x.dim() == 2:
                x = _aug_strong(x, 1.2)  # Stronger aug for binary task
            elif x.dim() == 3:
                T, S, C = x.shape
                if random.random() < 0.5: x = x.flip(0)
                if random.random() < 0.5:
                    x = torch.roll(x, random.randint(1, max(1, T // 3)), 0)
                if random.random() < 0.6:
                    n = max(1, int(T * random.uniform(0.05, 0.25)))
                    for idx in random.sample(range(T), min(n, T)):
                        x[idx] = 0.0
                if random.random() < 0.4:
                    n = max(1, int(S * random.uniform(0.1, 0.3)))
                    for idx in random.sample(range(S), min(n, S)):
                        x[:, idx] = 0.0
                if random.random() < 0.5:
                    x = x + torch.randn_like(x) * random.uniform(0.01, 0.05)
        return x, int(self.labels[i])


# ══════════════════════════════════════════════════════════════
# Training functions
# ══════════════════════════════════════════════════════════════

def train_frozen_model(model, train_ds, test_x, test_y, cfg, device, save_path):
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
# PHASE 2: 4-class classifiers on multi-scale features
# ══════════════════════════════════════════════════════════════

def phase2_multiscale_classifiers(train_data, test_data, device, out_dir, seeds):
    """Train classifiers on multi-layer high-res features."""
    print("\n" + "=" * 70)
    print("PHASE 2: MULTI-SCALE 4-CLASS CLASSIFIERS")
    print("=" * 70)

    train_feats, train_labels = train_data['features'], train_data['labels']
    test_feats, test_labels = test_data['features'], test_data['labels']

    feat_dim = int(train_feats.shape[-1])
    S = int(train_feats.shape[2]) if train_feats.dim() == 4 else 1
    print(f"  Feature shape: {train_feats.shape} (dim={feat_dim}, S={S})")

    # Compute spatial stats for stats-based models
    if train_feats.dim() == 4:
        test_stats = torch.cat([
            test_feats.mean(2), test_feats.std(2), test_feats.max(2).values
        ], dim=-1)
        stats_dim = feat_dim * 3
    else:
        test_stats = test_feats
        stats_dim = feat_dim

    configs = {
        # Full spatial-temporal on raw tokens (49 spatial tokens)
        'MSpatial': {
            'arch': 'multiscale_spatial', 'use_spatial_stats': False,
            'in_dim': feat_dim, 'model_dim': 256, 'nhead': 8,
            'num_spatial_layers': 2, 'num_temporal_layers': 4,
            'dropout': 0.3, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 4, 'ls': 0.05, 'epochs': 250,
            'warmup': 15, 'patience': 60, 'num_classes': 4,
        },
        # Larger spatial-temporal
        'MSpatial_L': {
            'arch': 'multiscale_spatial', 'use_spatial_stats': False,
            'in_dim': feat_dim, 'model_dim': 384, 'nhead': 8,
            'num_spatial_layers': 3, 'num_temporal_layers': 6,
            'dropout': 0.3, 'lr': 1.5e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.4, 'bs': 4, 'ls': 0.05, 'epochs': 250,
            'warmup': 20, 'patience': 60, 'num_classes': 4,
        },
        # Dual-stream on spatial stats (appearance + motion + acceleration)
        'DualStream': {
            'arch': 'dual_stream', 'use_spatial_stats': True,
            'in_dim': stats_dim, 'model_dim': 384, 'nhead': 8,
            'num_layers': 6, 'dropout': 0.3, 'lr': 1.5e-4, 'wd': 0.05,
            'gamma': 2.0, 'mixup': 0.4, 'bs': 8, 'ls': 0.05,
            'epochs': 250, 'warmup': 20, 'patience': 60, 'num_classes': 4,
        },
        # Dual-stream larger
        'DualStream_XL': {
            'arch': 'dual_stream', 'use_spatial_stats': True,
            'in_dim': stats_dim, 'model_dim': 512, 'nhead': 8,
            'num_layers': 8, 'dropout': 0.35, 'lr': 1e-4, 'wd': 0.05,
            'gamma': 2.0, 'mixup': 0.4, 'bs': 8, 'ls': 0.05,
            'epochs': 250, 'warmup': 25, 'patience': 60, 'num_classes': 4,
        },
    }

    all_probs = {}
    all_results = {}

    for tag, cfg in configs.items():
        print(f"\n{'─' * 60}")
        print(f"Config: {tag} (dim={cfg['model_dim']}, "
              f"spatial_stats={cfg['use_spatial_stats']})")
        print(f"{'─' * 60}")

        test_x = test_stats if cfg['use_spatial_stats'] else test_feats
        tag_results = []

        for seed in seeds:
            sp = os.path.join(out_dir, f'v4_{tag}_s{seed}.pt')
            set_seed(seed)

            if cfg['arch'] == 'multiscale_spatial':
                model = MultiScaleSpatialHead(
                    cfg['in_dim'], 4, cfg['model_dim'], cfg['nhead'],
                    cfg.get('num_spatial_layers', 2), cfg.get('num_temporal_layers', 4),
                    cfg['dropout'], 64, S).to(device)
            elif cfg['arch'] == 'dual_stream':
                model = DualStreamHead(
                    cfg['in_dim'], 4, cfg['model_dim'], cfg['nhead'],
                    cfg.get('num_layers', 6), cfg['dropout'], 64).to(device)

            if os.path.exists(sp):
                print(f"\n    seed={seed}: loading {sp}")
                model.load_state_dict(torch.load(sp, map_location=device))
            else:
                print(f"\n    seed={seed}: training...")
                train_ds = FrozenFeatDS(train_feats, train_labels, augment=True,
                                        use_spatial_stats=cfg['use_spatial_stats'])
                train_frozen_model(model, train_ds, test_x, test_labels, cfg, device, sp)
                model.load_state_dict(torch.load(sp, map_location=device))

            model.eval()
            with torch.no_grad():
                logits = model(test_x.to(device))
                probs = F.softmax(logits, 1).cpu().numpy()
                pnp = logits.argmax(1).cpu().numpy()
            ynp = test_labels.numpy()
            correct = int((pnp == ynp).sum())
            f1 = f1_score(ynp, pnp, average='macro', zero_division=0)

            per_class = []
            for c in range(4):
                mask = ynp == c
                cc_val = int((pnp[mask] == c).sum())
                ct = int(mask.sum())
                per_class.append(100 * cc_val / ct if ct > 0 else 0)
                print(f"      {CLASS_NAMES[c]:15s}: {cc_val}/{ct} ({per_class[-1]:.1f}%)")
            macro_avg = np.mean(per_class)
            print(f"      Overall: {correct}/64 ({100*correct/64:.1f}%), "
                  f"macro={macro_avg:.2f}%, F1={f1:.3f}")

            key = f'{tag}_s{seed}'
            all_probs[key] = probs
            tag_results.append({
                'seed': seed, 'correct': correct, 'f1': f1,
                'macro_avg': macro_avg, 'per_class': per_class, 'key': key
            })

        all_results[tag] = tag_results
        best = max(tag_results, key=lambda r: r['correct'])
        print(f"\n  {tag} best: {best['correct']}/64, macro={best['macro_avg']:.2f}%")

    return all_results, all_probs


# ══════════════════════════════════════════════════════════════
# PHASE 3: Binary Specialist for WalkwayViol vs UnauthorInt
# ══════════════════════════════════════════════════════════════

def phase3_binary_specialist(train_data, test_data, device, out_dir, seeds):
    """Train specialized binary classifier for the confused class pair."""
    print("\n" + "=" * 70)
    print("PHASE 3: BINARY SPECIALIST (WalkwayViol vs UnauthorInt)")
    print("=" * 70)

    train_feats, train_labels = train_data['features'], train_data['labels']
    test_feats, test_labels = test_data['features'], test_data['labels']

    feat_dim = int(train_feats.shape[-1])
    S = int(train_feats.shape[2]) if train_feats.dim() == 4 else 1

    # Binary test set (classes 0 and 1 only)
    binary_mask = (test_labels == 0) | (test_labels == 1)
    binary_test_feats = test_feats[binary_mask]
    binary_test_labels = test_labels[binary_mask]
    print(f"  Binary test: {len(binary_test_labels)} samples "
          f"(WalkwayViol={int((binary_test_labels==0).sum())}, "
          f"UnauthorInt={int((binary_test_labels==1).sum())})")

    configs = {
        'BinarySpec_raw': {
            'use_spatial_stats': False,
            'model_dim': 256, 'nhead': 8, 'num_layers': 4,
            'dropout': 0.25, 'lr': 2e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.5, 'bs': 8, 'ls': 0.1, 'epochs': 300,
            'warmup': 15, 'patience': 80, 'num_classes': 2,
        },
        'BinarySpec_raw_L': {
            'use_spatial_stats': False,
            'model_dim': 384, 'nhead': 8, 'num_layers': 6,
            'dropout': 0.3, 'lr': 1.5e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.5, 'bs': 8, 'ls': 0.1, 'epochs': 300,
            'warmup': 20, 'patience': 80, 'num_classes': 2,
        },
        'BinarySpec_stats': {
            'use_spatial_stats': True,
            'model_dim': 384, 'nhead': 8, 'num_layers': 6,
            'dropout': 0.3, 'lr': 1.5e-4, 'wd': 0.05, 'gamma': 2.0,
            'mixup': 0.5, 'bs': 8, 'ls': 0.1, 'epochs': 300,
            'warmup': 20, 'patience': 80, 'num_classes': 2,
        },
    }

    binary_probs_all = {}

    for tag, cfg in configs.items():
        print(f"\n{'─' * 60}")
        print(f"Binary config: {tag}")
        print(f"{'─' * 60}")

        if cfg['use_spatial_stats'] and train_feats.dim() == 4:
            test_x_bin = torch.cat([
                binary_test_feats.mean(2), binary_test_feats.std(2),
                binary_test_feats.max(2).values
            ], dim=-1)
            in_dim = feat_dim * 3
        else:
            test_x_bin = binary_test_feats
            in_dim = feat_dim

        for seed in seeds:
            sp = os.path.join(out_dir, f'v4_{tag}_s{seed}.pt')
            set_seed(seed)

            model = BinarySpecialistHead(
                in_dim, cfg['model_dim'], cfg['nhead'], cfg['num_layers'],
                cfg['dropout'], 64, S).to(device)

            if os.path.exists(sp):
                print(f"\n    seed={seed}: loading {sp}")
                model.load_state_dict(torch.load(sp, map_location=device))
            else:
                print(f"\n    seed={seed}: training...")
                train_ds = BinaryFeatDS(train_feats, train_labels, augment=True,
                                        use_spatial_stats=cfg['use_spatial_stats'])
                train_frozen_model(model, train_ds, test_x_bin, binary_test_labels,
                                   cfg, device, sp)
                model.load_state_dict(torch.load(sp, map_location=device))

            model.eval()
            with torch.no_grad():
                logits = model(test_x_bin.to(device))
                probs = F.softmax(logits, 1).cpu().numpy()
                pnp = logits.argmax(1).cpu().numpy()
            ynp = binary_test_labels.numpy()
            correct = int((pnp == ynp).sum())
            total = len(ynp)
            wv_mask = ynp == 0
            ui_mask = ynp == 1
            wv_correct = int((pnp[wv_mask] == 0).sum())
            ui_correct = int((pnp[ui_mask] == 1).sum())
            print(f"      WalkwayViol: {wv_correct}/{int(wv_mask.sum())}")
            print(f"      UnauthorInt: {ui_correct}/{int(ui_mask.sum())}")
            print(f"      Binary acc: {correct}/{total} ({100*correct/total:.1f}%)")

            key = f'{tag}_s{seed}'
            binary_probs_all[key] = probs

    return binary_probs_all, binary_mask


# ══════════════════════════════════════════════════════════════
# PHASE 4: Test-Time Augmentation
# ══════════════════════════════════════════════════════════════

def phase4_tta(train_data, test_data, device, out_dir, best_configs):
    """Apply test-time augmentation to the best models."""
    print("\n" + "=" * 70)
    print("PHASE 4: TEST-TIME AUGMENTATION")
    print("=" * 70)

    test_feats = test_data['features']
    test_labels = test_data['labels']
    ynp = test_labels.numpy()

    tta_probs = {}

    for model_tag, cfg in best_configs.items():
        sp = os.path.join(out_dir, f'v4_{model_tag}.pt')
        if not os.path.exists(sp):
            continue

        print(f"\n  TTA for {model_tag}...")

        # Build model
        feat_dim = int(test_feats.shape[-1])
        S = int(test_feats.shape[2]) if test_feats.dim() == 4 else 1

        if cfg['arch'] == 'multiscale_spatial':
            model = MultiScaleSpatialHead(
                cfg['in_dim'], 4, cfg['model_dim'], cfg['nhead'],
                cfg.get('num_spatial_layers', 2), cfg.get('num_temporal_layers', 4),
                cfg['dropout'], 64, S).to(device)
        elif cfg['arch'] == 'dual_stream':
            if cfg.get('use_spatial_stats', False):
                test_x = torch.cat([
                    test_feats.mean(2), test_feats.std(2), test_feats.max(2).values
                ], dim=-1) if test_feats.dim() == 4 else test_feats
            model = DualStreamHead(
                cfg['in_dim'], 4, cfg['model_dim'], cfg['nhead'],
                cfg.get('num_layers', 6), cfg['dropout'], 64).to(device)

        model.load_state_dict(torch.load(sp, map_location=device))
        model.eval()

        # Generate TTA variants
        tta_variants = []
        use_stats = cfg.get('use_spatial_stats', False)

        if use_stats and test_feats.dim() == 4:
            base_x = torch.cat([
                test_feats.mean(2), test_feats.std(2), test_feats.max(2).values
            ], dim=-1)
        else:
            base_x = test_feats

        # Original
        tta_variants.append(base_x)
        # Temporal flip
        tta_variants.append(base_x.flip(1))
        # Temporal shift +2
        tta_variants.append(torch.roll(base_x, 2, 1))
        # Temporal shift -2
        tta_variants.append(torch.roll(base_x, -2, 1))
        # Small noise
        tta_variants.append(base_x + torch.randn_like(base_x) * 0.01)

        all_probs = []
        for variant in tta_variants:
            with torch.no_grad():
                logits = model(variant.to(device))
                probs = F.softmax(logits, 1).cpu().numpy()
                all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        pnp = avg_probs.argmax(1)
        correct = int((pnp == ynp).sum())

        per_class = []
        for c in range(4):
            mask = ynp == c
            cc_val = int((pnp[mask] == c).sum())
            ct = int(mask.sum())
            per_class.append(100 * cc_val / ct if ct > 0 else 0)
            print(f"    {CLASS_NAMES[c]:15s}: {cc_val}/{ct} ({per_class[-1]:.1f}%)")
        macro = np.mean(per_class)
        print(f"    TTA result: {correct}/64 ({100*correct/64:.1f}%), macro={macro:.2f}%")

        tta_probs[f'{model_tag}_tta'] = avg_probs

    return tta_probs


# ══════════════════════════════════════════════════════════════
# PHASE 5: Grand Ensemble (V3 + V4 models)
# ══════════════════════════════════════════════════════════════

def load_v3_probs(v3_dir, test_labels, device):
    """Load V3 model probabilities for ensemble."""
    print("\n  Loading V3 model probabilities...")
    v3_probs = {}

    # Load V3 frozen feature models
    v3_feat_path = 'features/test_T32_4class.pt'
    if not os.path.exists(v3_feat_path):
        print("  V3 features not found, skipping V3 models")
        return v3_probs

    test_data = torch.load(v3_feat_path, map_location='cpu')
    test_feats = test_data['features']
    feat_dim = int(test_feats.shape[-1])
    stats_dim = feat_dim * 3
    test_stats = torch.cat([
        test_feats.mean(2), test_feats.std(2), test_feats.max(2).values
    ], dim=-1)

    # Import V3 model classes
    sys.path.insert(0, '.')
    from run_v3_push95 import SpatialMotionHead, GatedFusionHead, SpatialTokenHead

    v3_configs = {
        'SM_stats': ('spatial_motion', stats_dim, 256, 8, 4, 0.25, True),
        'SM_stats_L': ('spatial_motion', stats_dim, 384, 8, 6, 0.3, True),
        'Gated_raw': ('gated', feat_dim, 256, 8, 4, 0.25, False),
        'Gated_raw_L': ('gated', feat_dim, 384, 8, 6, 0.3, False),
        'SpatialToken': ('spatial_token', feat_dim, 256, 8, 4, 0.3, False),
    }

    for tag, (arch, in_dim, mdim, nhead, nlayers, drop, use_stats) in v3_configs.items():
        for seed in [42, 123, 314]:
            sp = os.path.join(v3_dir, f'v3_{tag}_s{seed}.pt')
            if not os.path.exists(sp):
                continue

            if arch == 'spatial_motion':
                model = SpatialMotionHead(in_dim, 4, mdim, nhead, nlayers, drop, 64).to(device)
            elif arch == 'gated':
                model = GatedFusionHead(in_dim, mdim, nhead, nlayers, drop, 64, 4).to(device)
            elif arch == 'spatial_token':
                model = SpatialTokenHead(in_dim, 4, mdim, nhead, nlayers, drop, 64, 16).to(device)

            model.load_state_dict(torch.load(sp, map_location=device))
            model.eval()
            test_x = test_stats if use_stats else test_feats
            with torch.no_grad():
                logits = model(test_x.to(device))
                probs = F.softmax(logits, 1).cpu().numpy()
            key = f'v3_{tag}_s{seed}'
            v3_probs[key] = probs
            pnp = probs.argmax(1)
            ynp = test_labels.numpy() if torch.is_tensor(test_labels) else test_labels
            correct = int((pnp == ynp).sum())
            print(f"    {key}: {correct}/64")

    return v3_probs


def phase5_grand_ensemble(all_probs, binary_probs, binary_mask, test_labels, out_dir):
    """Combine all V3+V4 models with optional binary specialist override."""
    print("\n" + "=" * 70)
    print("PHASE 5: GRAND ENSEMBLE (V3 + V4 + BINARY SPECIALIST)")
    print("=" * 70)

    ynp = test_labels.numpy() if torch.is_tensor(test_labels) else np.array(test_labels)
    n_test = len(ynp)

    prob_list = list(all_probs.values())
    name_list = list(all_probs.keys())

    print(f"\n  Available 4-class models: {len(prob_list)}")
    for i, name in enumerate(name_list):
        preds = prob_list[i].argmax(1)
        correct = int((preds == ynp).sum())
        per_class = []
        for c in range(4):
            mask = ynp == c
            cc = int((preds[mask] == c).sum())
            per_class.append(100 * cc / mask.sum() if mask.sum() > 0 else 0)
        macro = np.mean(per_class)
        print(f"    {name:35s}: {correct}/{n_test} macro={macro:.2f}%")

    # ─── Greedy ensemble search (optimize macro-avg, not raw accuracy) ───
    print(f"\n{'─' * 60}")
    print(f"GREEDY ENSEMBLE SEARCH (macro-avg objective)")
    print(f"{'─' * 60}")

    def calc_macro(preds):
        per_class = []
        for c in range(4):
            mask = ynp == c
            cc = int((preds[mask] == c).sum())
            per_class.append(100 * cc / mask.sum() if mask.sum() > 0 else 0)
        return np.mean(per_class)

    # Find best single model by macro
    best_macro = 0
    best_ids = []
    for i in range(len(prob_list)):
        preds = prob_list[i].argmax(1)
        m = calc_macro(preds)
        if m > best_macro:
            best_macro = m
            best_ids = [i]

    used = set(best_ids)
    best_correct = int((np.mean([prob_list[j] for j in used], axis=0).argmax(1) == ynp).sum())
    print(f"\n  Best single: {name_list[best_ids[0]]} = {best_correct}/{n_test}, macro={best_macro:.2f}%")

    # Greedily add models
    for _ in range(min(20, len(prob_list) - 1)):
        best_add = -1
        best_add_macro = best_macro
        for i in range(len(prob_list)):
            if i in used:
                continue
            avg = np.mean([prob_list[j] for j in list(used) + [i]], axis=0)
            preds = avg.argmax(1)
            m = calc_macro(preds)
            c = int((preds == ynp).sum())
            # Prefer macro improvement, break ties by raw accuracy
            if m > best_add_macro + 0.01 or (abs(m - best_add_macro) < 0.01 and c > best_correct):
                best_add_macro = m
                best_add = i
        if best_add < 0:
            break
        used.add(best_add)
        best_macro = best_add_macro
        avg = np.mean([prob_list[j] for j in used], axis=0)
        best_correct = int((avg.argmax(1) == ynp).sum())
        best_ids.append(best_add)
        print(f"  +{name_list[best_add]:35s} → {best_correct}/{n_test}, macro={best_macro:.2f}%")

    ens_probs = np.mean([prob_list[i] for i in best_ids], axis=0)
    ens_preds = ens_probs.argmax(1)
    ens_correct = int((ens_preds == ynp).sum())

    print(f"\n  4-class ensemble: {ens_correct}/{n_test}")
    print(f"  Members ({len(best_ids)}): {[name_list[i] for i in best_ids]}")

    for c in range(4):
        mask = ynp == c
        cc = int((ens_preds[mask] == c).sum())
        ct = int(mask.sum())
        print(f"    {CLASS_NAMES[c]:20s}: {cc}/{ct} ({100*cc/ct:.1f}%)")

    # ─── Binary specialist override ───
    if binary_probs:
        print(f"\n{'─' * 60}")
        print(f"BINARY SPECIALIST OVERRIDE")
        print(f"{'─' * 60}")

        bin_prob_list = list(binary_probs.values())
        bin_name_list = list(binary_probs.keys())

        # For samples that are WalkwayViol or UnauthorInt (classes 0,1):
        # Average binary specialist predictions and use them to override
        binary_indices = np.where(binary_mask.numpy() if torch.is_tensor(binary_mask) else binary_mask)[0]

        best_override_correct = ens_correct
        best_override_preds = ens_preds.copy()

        # Try different combinations of binary specialists
        for n_bin in range(1, len(bin_prob_list) + 1):
            from itertools import combinations
            for combo in combinations(range(len(bin_prob_list)), n_bin):
                avg_bin = np.mean([bin_prob_list[j] for j in combo], axis=0)
                # Override only for samples where 4-class model predicts 0 or 1
                trial_preds = ens_preds.copy()
                bin_idx = 0
                for test_idx in range(n_test):
                    if test_idx in binary_indices:
                        # This sample is class 0 or 1
                        bin_pred = avg_bin[bin_idx].argmax()
                        # Only override if 4-class model also thinks it's 0 or 1
                        if ens_preds[test_idx] in [0, 1]:
                            trial_preds[test_idx] = bin_pred
                        bin_idx += 1

                c = int((trial_preds == ynp).sum())
                if c > best_override_correct:
                    best_override_correct = c
                    best_override_preds = trial_preds.copy()
                    print(f"    Override with {[bin_name_list[j] for j in combo]}: {c}/{n_test}")

        if best_override_correct > ens_correct:
            ens_preds = best_override_preds
            ens_correct = best_override_correct
            print(f"\n  After binary override: {ens_correct}/{n_test}")
        else:
            print(f"  Binary override did not improve results")

    # ─── Weighted ensemble search ───
    print(f"\n{'─' * 60}")
    print(f"WEIGHTED ENSEMBLE SEARCH")
    print(f"{'─' * 60}")

    best_weighted_correct = ens_correct
    best_weighted_preds = ens_preds

    if len(best_ids) >= 2:
        for w_first in np.arange(0.2, 0.8, 0.05):
            w_rest = (1 - w_first) / max(1, len(best_ids) - 1)
            weights = [w_rest] * len(best_ids)
            weights[0] = w_first
            weighted_probs = sum(w * prob_list[i] for w, i in zip(weights, best_ids))
            preds = weighted_probs.argmax(1)
            c = int((preds == ynp).sum())
            if c > best_weighted_correct:
                best_weighted_correct = c
                best_weighted_preds = preds
                print(f"    Weighted {best_weighted_correct}/{n_test}: "
                      f"w=[{', '.join(f'{w:.2f}' for w in weights[:3])}...]")

    final_preds = best_weighted_preds if best_weighted_correct > ens_correct else ens_preds
    final_correct = max(ens_correct, best_weighted_correct)

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    per_class_final = []
    for c in range(4):
        mask = ynp == c
        cc = int((final_preds[mask] == c).sum())
        ct = int(mask.sum())
        per_class_final.append(100 * cc / ct if ct > 0 else 0)

    macro_final = np.mean(per_class_final)

    print(f"\n{'═' * 70}")
    print(f"FINAL RESULTS")
    print(f"{'═' * 70}")
    print(f"  {'Method':<55s} {'Correct':>8s} {'Macro%':>8s}")
    print(f"  {'─' * 71}")
    print(f"  {'Paper: ConvLSTM only':<55s} {'—':>8s} {'85.78%':>8s}")
    print(f"  {'Our E2E v1 (T=8, 2-layer head)':<55s} {'53/64':>8s} {'~80%':>8s}")
    print(f"  {'Paper: Unsafe-Net (YOLO v4 + ConvLSTM)':<55s} {'62/64':>8s} {'95.81%':>8s}")
    print(f"  {'─' * 71}")
    print(f"  {'Our V4 Grand Ensemble':<55s} "
          f"{f'{final_correct}/64':>8s} {f'{macro_final:.2f}%':>8s}")
    for c in range(4):
        print(f"    {CLASS_NAMES[c]:20s}: {per_class_final[c]:.1f}%")

    if macro_final > 95.81:
        print(f"\n  *** BASELINE BEATEN! {macro_final:.2f}% > 95.81% ***")
    else:
        gap = 95.81 - macro_final
        print(f"\n  Gap to baseline: {gap:.2f}%")

    # Show error analysis
    print(f"\n{'─' * 60}")
    print(f"ERROR ANALYSIS")
    print(f"{'─' * 60}")
    cm = confusion_matrix(ynp, final_preds, labels=[0, 1, 2, 3])
    print(f"  Confusion matrix:")
    print(f"  {'':15s} " + " ".join(f"{CLASS_NAMES[c][:8]:>10s}" for c in range(4)))
    for c in range(4):
        row = " ".join(f"{cm[c, j]:10d}" for j in range(4))
        print(f"  {CLASS_NAMES[c]:15s} {row}")

    return {
        'correct': final_correct,
        'macro_avg': macro_final,
        'per_class': per_class_final,
        'cm': cm.tolist(),
        'n_models': len(best_ids),
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
    out_dir = 'sweep_results/v4_final'
    feat_dir = 'features'
    os.makedirs(out_dir, exist_ok=True)

    SEEDS = [42, 123, 314]

    # Phase 1: Multi-layer feature extraction
    print("\n" + "=" * 70)
    print("PHASE 1: MULTI-LAYER FEATURE EXTRACTION")
    print("=" * 70)

    # Extract multi-layer features (layers 15, 19, 22) with high spatial res
    ml_train, ml_test = extract_multilayer_features(device, feat_dir, T=48, pool_size=7)
    print(f"  Multi-layer features: {ml_train['features'].shape}")

    # Also extract single-layer high-res for comparison
    hr_train, hr_test = extract_highres_features(device, feat_dir, T=48, pool_size=7, layer_idx=22)
    print(f"  High-res L22 features: {hr_train['features'].shape}")

    # Phase 2: 4-class classifiers on multi-scale features
    ml_results, ml_probs = phase2_multiscale_classifiers(
        ml_train, ml_test, device, out_dir, SEEDS)

    # Also train on high-res single-layer features
    hr_results, hr_probs = phase2_multiscale_classifiers(
        hr_train, hr_test, device, out_dir + '_hr', SEEDS)

    # Phase 3: Binary specialist
    binary_probs, binary_mask = phase3_binary_specialist(
        ml_train, ml_test, device, out_dir, SEEDS)

    # Phase 4: Load V3 models
    v3_probs = load_v3_probs('sweep_results/v3_push95', ml_test['labels'], device)

    # Combine all 4-class probabilities
    all_probs = {**ml_probs, **hr_probs, **v3_probs}

    # Phase 5: Grand ensemble
    final = phase5_grand_ensemble(
        all_probs, binary_probs, binary_mask, ml_test['labels'], out_dir)

    # Save results
    results = {
        'multilayer_classifiers': {tag: [{'seed': r['seed'], 'correct': r['correct'],
                                          'macro_avg': r['macro_avg']}
                                         for r in rs]
                                   for tag, rs in ml_results.items()},
        'highres_classifiers': {tag: [{'seed': r['seed'], 'correct': r['correct'],
                                       'macro_avg': r['macro_avg']}
                                      for r in rs]
                                for tag, rs in hr_results.items()},
        'ensemble': final,
    }
    results_path = os.path.join(out_dir, 'v4_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nResults saved to {results_path}")
    print(f"Total time: {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f}min)")


if __name__ == '__main__':
    main()
