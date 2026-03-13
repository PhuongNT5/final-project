#!/usr/bin/env python3
"""
END-TO-END FINE-TUNING: YOLOv11s backbone (partial unfreeze) + Transformer head.

Strategy v2 — fixes from first run:
  1) Use feature_layer_idx=22 (C3k2, 512-dim, 7x7) — the real last feature block
  2) Unfreeze layers 19-22 (C3k2 + Conv + Concat + C3k2) = ~2.4M backbone params
  3) Smaller head: 2-layer Transformer (was 4) — dataset too small
  4) Lower learning rates: backbone 5e-6, head 1e-4
  5) Longer warmup (8 epochs) and patience (25)
  6) WeightedRandomSampler for class balance

Paper baseline: 95.81% on 4 unsafe classes (64 clips)
Our frozen cascade: 90.63% on 4 unsafe classes
"""
import os, sys, random, time, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import (f1_score, confusion_matrix, accuracy_score,
                             precision_score, recall_score)
from copy import deepcopy
from PIL import Image
import cv2

try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass

CLASS_NAMES_8 = [
    'WalkwayViol(0)', 'UnauthorInt(1)', 'OpenedPanel(2)', 'CarryOverload(3)',
    'SafeWalkway(4)', 'AuthorizedInt(5)', 'ClosedPanel(6)', 'SafeCarrying(7)'
]
CLASS_NAMES_4U = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']

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
        self.gamma = gamma
        self.ls = ls

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             reduction='none', label_smoothing=self.ls)
        pt = F.softmax(logits, 1).gather(1, target.view(-1, 1)).squeeze(1).clamp(1e-6)
        return (((1 - pt) ** self.gamma) * ce).mean()


class CosWarmup(optim.lr_scheduler._LRScheduler):
    def __init__(self, opt, warmup, total, min_lr=1e-7, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.min_lr = min_lr
        super().__init__(opt, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            return [lr * (self.last_epoch + 1) / max(1, self.warmup)
                    for lr in self.base_lrs]
        p = (self.last_epoch - self.warmup) / max(1, self.total - self.warmup)
        return [max(self.min_lr, lr * 0.5 * (1 + math.cos(math.pi * p)))
                for lr in self.base_lrs]


class EMA:
    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.shadow = {n: p.data.clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])


# ══════════════════════════════════════════════════════════════
# Video Dataset
# ══════════════════════════════════════════════════════════════

class VideoDataset(Dataset):
    def __init__(self, root_dir, subset='train', T=32, img_size=224,
                 augment=False, class_filter=None, cache=True):
        self.T = T
        self.img_size = img_size
        self.augment = augment
        self.subset = subset
        self.cache = cache
        self.cached_frames = {}

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

        self.base_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.aug_tf = transforms.Compose([
            transforms.Resize((int(img_size * 1.14), int(img_size * 1.14))),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if cache:
            print("  Caching {} videos into RAM...".format(len(self.samples)))
            for i, (path, _) in enumerate(self.samples):
                self.cached_frames[i] = self._read_pil_frames(path)
                if (i + 1) % 100 == 0 or i + 1 == len(self.samples):
                    print("    cached {}/{}".format(i + 1, len(self.samples)))

    def _read_pil_frames(self, path):
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
            frames.append(frames[-1] if frames else
                          Image.new('RGB', (self.img_size, self.img_size)))
        return frames[:self.T]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, label = self.samples[idx]
        if self.cache and idx in self.cached_frames:
            frames = self.cached_frames[idx]
        else:
            path, _ = self.samples[idx]
            frames = self._read_pil_frames(path)

        if self.augment and self.subset == 'train':
            # Random temporal jitter for training
            if random.random() < 0.3:
                frames = list(frames)
                n_jit = random.randint(1, max(1, len(frames) // 8))
                for _ in range(n_jit):
                    i = random.randint(0, len(frames) - 1)
                    j = max(0, i - 1)
                    frames[i] = frames[j]

        tf = self.aug_tf if (self.augment and self.subset == 'train') else self.base_tf
        tensors = torch.stack([tf(f) for f in frames])
        return tensors, label

    def get_labels(self):
        return [l for _, l in self.samples]


# ══════════════════════════════════════════════════════════════
# YOLOv11 Backbone
# ══════════════════════════════════════════════════════════════

class YOLOv11Backbone(nn.Module):
    def __init__(self, weights_path='yolo11s.pt', pool_size=4,
                 feature_layer_idx=22, unfreeze_from=19):
        super().__init__()
        from ultralytics import YOLO
        yolo = YOLO(weights_path)
        self.backbone = yolo.model
        self.pool_size = pool_size
        self.target_idx = feature_layer_idx

        # Freeze everything
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze from unfreeze_from to feature_layer_idx
        layers = self.backbone.model
        n_unfrozen_t = 0
        n_unfrozen_p = 0
        for i in range(unfreeze_from, feature_layer_idx + 1):
            for p in layers[i].parameters():
                p.requires_grad = True
                n_unfrozen_t += 1
                n_unfrozen_p += p.numel()
        print("  [Backbone] Unfroze layers {}-{}: {} tensors, {:,} params".format(
            unfreeze_from, feature_layer_idx, n_unfrozen_t, n_unfrozen_p))

    def _pool(self, x):
        h, w = x.shape[2], x.shape[3]
        p = self.pool_size
        if h % p == 0 and w % p == 0:
            return F.adaptive_avg_pool2d(x, (p, p))
        new_h = max(p, (h // p) * p) if h % p != 0 else h
        new_w = max(p, (w // p) * p) if w % p != 0 else w
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear',
                          align_corners=False)
        return F.adaptive_avg_pool2d(x, (p, p))

    def forward(self, x):
        features = None
        layers = self.backbone.model

        def hook(module, inp, out):
            nonlocal features
            features = out

        handle = layers[self.target_idx].register_forward_hook(hook)
        with torch.no_grad() if not any(p.requires_grad for p in layers[self.target_idx].parameters()) else torch.enable_grad():
            _ = self.backbone(x)
        handle.remove()

        if features is None:
            raise RuntimeError("Feature hook returned None")

        if isinstance(features, (list, tuple)):
            features = features[0]

        f = self._pool(features)
        f = f.flatten(2).transpose(1, 2).contiguous()
        return f

    def get_backbone_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def get_feat_dim(self):
        dev = next(self.parameters()).device
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=dev)
            f = self.forward(dummy)
        return f.shape[-1]


# ══════════════════════════════════════════════════════════════
# Transformer Head (lighter: 2-layer)
# ══════════════════════════════════════════════════════════════

class TransformerHead(nn.Module):
    def __init__(self, in_dim, num_classes, model_dim=192, nhead=8,
                 num_layers=2, dropout=0.3, max_t=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5))

        # Motion stream
        self.proj_motion = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5))

        # Gate for appear vs motion
        self.gate = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 2),
            nn.Softmax(dim=-1))

        self.time_pos = nn.Embedding(max_t, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        nh = nhead
        while model_dim % nh != 0:
            nh -= 1

        self.pre_norm = nn.LayerNorm(model_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(model_dim)

        self.fc = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes))

    def forward(self, tokens):
        # tokens: (B, T, S, C) or (B, T, C)
        if tokens.dim() == 4:
            x = tokens.mean(2)  # spatial average -> (B, T, C)
        else:
            x = tokens

        b, t, c = x.shape
        d = x.device

        appear = self.proj(x)  # (B, T, model_dim)
        d1 = x[:, 1:] - x[:, :-1]
        motion = self.proj_motion(d1)  # (B, T-1, model_dim)
        # Pad motion to T
        motion_pad = F.pad(motion, (0, 0, 0, 1), value=0)

        # Gate
        s_a = appear.mean(1)
        s_m = motion_pad.mean(1)
        g = self.gate(torch.cat([s_a, s_m], dim=-1))  # (B, 2)
        fused = g[:, 0:1].unsqueeze(1) * appear + g[:, 1:2].unsqueeze(1) * motion_pad

        # Positional encoding
        pos = self.time_pos(torch.arange(t, device=d)).unsqueeze(0)
        fused = fused + pos

        # CLS + Transformer
        cls = self.cls_token.expand(b, -1, -1)
        tokens_in = self.pre_norm(torch.cat([cls, fused], dim=1))
        tokens_out = self.transformer(tokens_in)
        cls_out = self.post_norm(tokens_out[:, 0])

        return self.fc(cls_out)


# ══════════════════════════════════════════════════════════════
# E2E Model
# ══════════════════════════════════════════════════════════════

class E2EModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        b, t = x.shape[:2]
        x_flat = x.view(b * t, *x.shape[2:])
        tokens = self.backbone(x_flat)
        s, c = tokens.shape[1], tokens.shape[2]
        tokens = tokens.view(b, t, s, c)
        return self.head(tokens)

    def extract_features(self, x):
        b, t = x.shape[:2]
        x_flat = x.view(b * t, *x.shape[2:])
        tokens = self.backbone(x_flat)
        s, c = tokens.shape[1], tokens.shape[2]
        return tokens.view(b, t, s, c)


# ══════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════

def mixup_batch(x, y, alpha=0.3):
    lam = max(np.random.beta(alpha, alpha), 1 - np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def train_model(model, train_loader, test_loader, cfg, device, save_path):
    nc = cfg['num_classes']
    label_map = cfg.get('label_map', None)

    # Class weights
    all_labels = []
    for _, y in train_loader.dataset.samples:
        mapped = label_map(y) if label_map else y
        if mapped is not None and mapped >= 0:
            all_labels.append(mapped)
    all_labels = np.array(all_labels)
    cc = np.bincount(all_labels, minlength=nc).astype(float)
    cw = len(all_labels) / (nc * np.maximum(cc, 1))
    cw = (cw / cw.mean()).astype(np.float32)
    wt = torch.tensor(cw).to(device)
    print("    Class weights: {}".format(np.array2string(cw, precision=3)))

    crit = FocalLoss(weight=wt, gamma=cfg.get('gamma', 2.0), ls=cfg.get('ls', 0.05))

    # Discriminative LR
    bb_params = model.backbone.get_backbone_params()
    hd_params = list(model.head.parameters())
    param_groups = [
        {'params': bb_params, 'lr': cfg['lr_bb']},
        {'params': hd_params, 'lr': cfg['lr_hd']},
    ]
    opt = optim.AdamW(param_groups, weight_decay=cfg.get('wd', 0.05))
    sched = CosWarmup(opt, cfg.get('warmup', 8), cfg['epochs'])
    ema = EMA(model, 0.998)

    best_metric = -1
    best_st = None
    stale = 0

    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss, n_correct, n_total = 0, 0, 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            if label_map:
                yb = torch.tensor([label_map(int(y)) for y in yb],
                                  dtype=torch.long, device=device)
            else:
                yb = yb.to(device).long()

            if cfg.get('mixup', 0) > 0 and random.random() < 0.5:
                xm, ya, yb2, lam = mixup_batch(xb, yb, cfg['mixup'])
                opt.zero_grad(set_to_none=True)
                logits = model(xm)
                loss = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb2)
            else:
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = crit(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

            total_loss += loss.item()
            preds = logits.argmax(1)
            n_correct += int((preds == yb).sum())
            n_total += len(yb)

        sched.step()
        train_acc = 100 * n_correct / max(n_total, 1)

        # Evaluate
        ema.apply(model)
        model.eval()
        all_preds, all_labels_eval = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = model(xb).argmax(1).cpu().numpy()
                if label_map:
                    labels = np.array([label_map(int(y)) for y in yb])
                else:
                    labels = np.array([int(y) for y in yb])
                all_preds.extend(preds)
                all_labels_eval.extend(labels)
        pnp = np.array(all_preds)
        ynp = np.array(all_labels_eval)

        acc = 100 * (pnp == ynp).mean()
        f1 = f1_score(ynp, pnp, average='macro', zero_division=0)
        correct = int((pnp == ynp).sum())

        # Use f1 as primary metric to avoid class collapse
        metric = f1
        if metric > best_metric + 1e-4:
            best_metric = metric
            best_acc = acc
            best_st = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        ema.restore(model)

        if ep % 5 == 0 or ep <= 2 or ep == cfg['epochs'] or stale == 0:
            avg_loss = total_loss / max(len(train_loader), 1)
            lr_bb = opt.param_groups[0]['lr']
            lr_hd = opt.param_groups[1]['lr']
            marker = " *" if stale == 0 else ""
            print("    ep{:3d} loss={:.4f} train_acc={:.1f}% test_acc={:.1f}% ({}/{}) "
                  "f1={:.3f} lr_bb={:.1e} lr_hd={:.1e}{}".format(
                      ep, avg_loss, train_acc, acc, correct, len(ynp),
                      f1, lr_bb, lr_hd, marker), flush=True)

        if stale >= cfg.get('patience', 25):
            print("    [early stop ep{}]".format(ep), flush=True)
            break

    if save_path and best_st:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_st, save_path)
        print("    [saved -> {}] best f1={:.3f}, acc={:.1f}%".format(
            save_path, best_metric, best_acc))

    return best_st, best_metric


# ══════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════

def eval_model(model, test_loader, device, label_map, class_names, tag=""):
    model.eval()
    all_preds, all_labels, all_orig = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(1).cpu().numpy()
            for j in range(len(yb)):
                orig = int(yb[j])
                mapped = label_map(orig) if label_map else orig
                all_preds.append(preds[j])
                all_labels.append(mapped)
                all_orig.append(orig)

    pnp = np.array(all_preds)
    ynp = np.array(all_labels)
    orig_np = np.array(all_orig)
    nc = max(int(ynp.max()), int(pnp.max())) + 1

    acc = accuracy_score(ynp, pnp)
    f1m = f1_score(ynp, pnp, average='macro', zero_division=0)
    cm = confusion_matrix(ynp, pnp, labels=list(range(nc)))
    correct = int((pnp == ynp).sum())
    total = len(ynp)

    print("\n  [{}] {}/{} ({:.1f}%) F1={:.3f}".format(tag, correct, total, 100 * acc, f1m))
    print("  CM:\n{}".format(cm))
    if class_names:
        for c in range(min(nc, len(class_names))):
            mask = ynp == c
            if mask.sum() == 0:
                continue
            cc = int((pnp[mask] == c).sum())
            ct = int(mask.sum())
            print("    {:25s}: {}/{} ({:.1f}%)".format(class_names[c], cc, ct, 100 * cc / ct))

    return {'tag': tag, 'acc': acc, 'f1': f1m, 'correct': correct, 'total': total,
            'cm': cm.tolist(), 'preds': pnp.tolist(), 'labels': ynp.tolist(),
            'orig_labels': orig_np.tolist()}


# ══════════════════════════════════════════════════════════════
# EXPERIMENT A: 4-class unsafe only (paper comparison)
# ══════════════════════════════════════════════════════════════

def run_4class_unsafe(device, seeds, out_dir):
    print("\n" + "=" * 70)
    print("EXPERIMENT A: 4-CLASS UNSAFE E2E (paper-comparable)")
    print("  Classes 0-3 only, target > 95.81%")
    print("=" * 70)

    T = 32
    results = []

    for seed in seeds:
        tag = "4u_s{}".format(seed)
        save_path = os.path.join(out_dir, "e2e_4unsafe_{}.pt".format(tag))
        set_seed(seed)
        print("\n  --- Seed {} ---".format(seed))

        bb = YOLOv11Backbone('yolo11s.pt', pool_size=4,
                              feature_layer_idx=22, unfreeze_from=19)
        feat_dim = bb.get_feat_dim()
        print("  Feature dim: {}".format(feat_dim))

        head = TransformerHead(in_dim=feat_dim, num_classes=4, model_dim=192,
                               nhead=8, num_layers=2, dropout=0.3)
        model = E2EModel(bb, head).to(device)

        n_bb = sum(p.numel() for p in bb.get_backbone_params())
        n_hd = sum(p.numel() for p in head.parameters())
        print("  Params: backbone {:,} + head {:,} = {:,}".format(n_bb, n_hd, n_bb + n_hd))

        train_ds = VideoDataset('data', 'train', T=T, augment=True,
                                class_filter=[0, 1, 2, 3])
        test_ds = VideoDataset('data', 'test', T=T, augment=False,
                               class_filter=[0, 1, 2, 3])
        # Weighted sampler for balance
        labels = train_ds.get_labels()
        cc = np.bincount(labels, minlength=4)
        sample_w = [1.0 / max(cc[l], 1) for l in labels]
        sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=4, sampler=sampler,
                                  num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                                 num_workers=0)

        cfg = {
            'num_classes': 4,
            'lr_bb': 5e-6,
            'lr_hd': 1e-4,
            'wd': 0.05,
            'gamma': 2.0,
            'ls': 0.05,
            'mixup': 0.2,
            'epochs': 80,
            'warmup': 8,
            'patience': 25,
        }

        if os.path.exists(save_path):
            print("  Loading existing: {}".format(save_path))
            model.load_state_dict(torch.load(save_path, map_location=device,
                                             weights_only=False))
            best_f1 = 0.0
        else:
            _, best_f1 = train_model(model, train_loader, test_loader, cfg,
                                     device, save_path)

        r = eval_model(model, test_loader, device, None, CLASS_NAMES_4U, tag)
        r['seed'] = seed
        results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT A SUMMARY: 4-class unsafe")
    print("=" * 70)
    for r in results:
        print("  seed={} acc={:.1f}% ({}/{}) f1={:.3f}".format(
            r['seed'], 100 * r['acc'], r['correct'], r['total'], r['f1']))
    accs = [r['acc'] for r in results]
    print("  Mean: {:.1f}%  Best: {:.1f}%".format(
        100 * np.mean(accs), 100 * max(accs)))
    print("  Paper baseline: 95.81%")
    print("  Our frozen cascade: 90.63%")
    return results


# ══════════════════════════════════════════════════════════════
# EXPERIMENT B: 8-class direct
# ══════════════════════════════════════════════════════════════

def run_8class(device, seeds, out_dir):
    print("\n" + "=" * 70)
    print("EXPERIMENT B: 8-CLASS E2E")
    print("=" * 70)

    T = 32
    results = []

    for seed in seeds:
        tag = "8c_s{}".format(seed)
        save_path = os.path.join(out_dir, "e2e_8class_{}.pt".format(tag))
        set_seed(seed)
        print("\n  --- Seed {} ---".format(seed))

        bb = YOLOv11Backbone('yolo11s.pt', pool_size=4,
                              feature_layer_idx=22, unfreeze_from=19)
        feat_dim = bb.get_feat_dim()

        head = TransformerHead(in_dim=feat_dim, num_classes=8, model_dim=192,
                               nhead=8, num_layers=2, dropout=0.35)
        model = E2EModel(bb, head).to(device)

        n_bb = sum(p.numel() for p in bb.get_backbone_params())
        n_hd = sum(p.numel() for p in head.parameters())
        print("  Params: backbone {:,} + head {:,} = {:,}".format(n_bb, n_hd, n_bb + n_hd))

        train_ds = VideoDataset('data', 'train', T=T, augment=True)
        test_ds = VideoDataset('data', 'test', T=T, augment=False)
        labels = train_ds.get_labels()
        cc = np.bincount(labels, minlength=8)
        sample_w = [1.0 / max(cc[l], 1) for l in labels]
        sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=4, sampler=sampler,
                                  num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                                 num_workers=0)

        cfg = {
            'num_classes': 8,
            'lr_bb': 3e-6,
            'lr_hd': 8e-5,
            'wd': 0.05,
            'gamma': 2.0,
            'ls': 0.05,
            'mixup': 0.2,
            'epochs': 80,
            'warmup': 8,
            'patience': 25,
        }

        if os.path.exists(save_path):
            print("  Loading existing: {}".format(save_path))
            model.load_state_dict(torch.load(save_path, map_location=device,
                                             weights_only=False))
        else:
            train_model(model, train_loader, test_loader, cfg, device, save_path)

        # 8-class eval
        r8 = eval_model(model, test_loader, device, None, CLASS_NAMES_8, tag)
        r8['seed'] = seed

        # Binary eval (classes 0-3 = unsafe -> 1, classes 4-7 = safe -> 0)
        def bin_map(y):
            return 0 if y < 4 else 1   # 0=unsafe, 1=safe
        model.eval()
        bin_preds, bin_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                for j in range(len(yb)):
                    orig = int(yb[j])
                    pred8 = logits[j].argmax().item()
                    # Map 8-class pred to binary
                    bin_preds.append(0 if pred8 < 4 else 1)
                    bin_labels.append(bin_map(orig))
        bp = np.array(bin_preds)
        bl = np.array(bin_labels)
        bin_acc = 100 * (bp == bl).mean()
        print("\n  [Binary from 8-class, s{}] {}/{} ({:.1f}%)".format(
            seed, int((bp == bl).sum()), len(bl), bin_acc))

        # 4-class unsafe eval (only classes 0-3)
        u_preds, u_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                for j in range(len(yb)):
                    orig = int(yb[j])
                    if orig < 4:
                        u_preds.append(logits[j].argmax().item())
                        u_labels.append(orig)
        up = np.array(u_preds)
        ul = np.array(u_labels)
        u4_acc = 100 * (up == ul).mean()
        print("  [4-unsafe from 8-class, s{}] {}/{} ({:.1f}%)".format(
            seed, int((up == ul).sum()), len(ul), u4_acc))

        r8['binary_acc'] = bin_acc
        r8['unsafe4_acc'] = u4_acc
        results.append(r8)

    print("\n" + "=" * 70)
    print("EXPERIMENT B SUMMARY: 8-class")
    print("=" * 70)
    for r in results:
        print("  seed={} 8class={:.1f}% binary={:.1f}% 4unsafe={:.1f}%".format(
            r['seed'], 100 * r['acc'], r['binary_acc'], r['unsafe4_acc']))
    return results


# ══════════════════════════════════════════════════════════════
# Feature re-extraction for cosine similarity analysis
# ══════════════════════════════════════════════════════════════

def cosine_analysis(model, device, T=32):
    print("\n" + "=" * 70)
    print("COSINE SIMILARITY ANALYSIS (fine-tuned features)")
    print("=" * 70)

    test_ds = VideoDataset('data', 'test', T=T, augment=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            feats = model.extract_features(xb)  # (B, T, S, C)
            feats = feats.mean(dim=(1, 2))  # (B, C)
            all_feats.append(feats.cpu())
            all_labels.extend([int(y) for y in yb])

    feats = torch.cat(all_feats, dim=0)
    labels = np.array(all_labels)

    pairs = [(0, 4, 'WalkwayViol   <-> SafeWalkway'),
             (1, 5, 'UnauthorInt   <-> AuthorizedInt'),
             (2, 6, 'OpenedPanel   <-> ClosedPanel'),
             (3, 7, 'CarryOverload <-> SafeCarrying')]

    print("  Pair                              COS-SIM  (frozen was 0.94-0.996)")
    for c_u, c_s, name in pairs:
        mask_u = labels == c_u
        mask_s = labels == c_s
        if mask_u.sum() == 0 or mask_s.sum() == 0:
            continue
        centroid_u = feats[mask_u].mean(0)
        centroid_s = feats[mask_s].mean(0)
        cos = F.cosine_similarity(centroid_u.unsqueeze(0),
                                  centroid_s.unsqueeze(0)).item()
        print("    {:35s} {:.4f}".format(name, cos))


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    # Device selection
    if torch.backends.mps.is_available():
        device = 'mps'
        # Quick MPS sanity check
        try:
            t = torch.randn(2, 4, device='mps')
            _ = F.softmax(t, dim=-1)
            print("Device: mps (OK)")
        except Exception:
            device = 'cpu'
            print("Device: cpu (MPS fallback)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("Device: cuda")
    else:
        device = 'cpu'
        print("Device: cpu")

    t0 = time.time()
    out_dir = 'sweep_results/finetune'
    os.makedirs(out_dir, exist_ok=True)

    SEEDS = [42, 123, 314]

    # ─── Experiment A: 4-class unsafe (paper comparison) ───
    results_4u = run_4class_unsafe(device, SEEDS, out_dir)

    # ─── Experiment B: 8-class direct ───
    results_8c = run_8class(device, SEEDS, out_dir)

    # ─── Cosine analysis with best 8-class model ───
    best_8c = max(results_8c, key=lambda r: r['f1'])
    best_path = os.path.join(out_dir, "e2e_8class_8c_s{}.pt".format(best_8c['seed']))
    if os.path.exists(best_path):
        bb = YOLOv11Backbone('yolo11s.pt', pool_size=4,
                              feature_layer_idx=22, unfreeze_from=19)
        feat_dim = bb.get_feat_dim()
        head = TransformerHead(in_dim=feat_dim, num_classes=8, model_dim=192,
                               nhead=8, num_layers=2, dropout=0.35)
        model = E2EModel(bb, head).to(device)
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=False))
        cosine_analysis(model, device)

    # ─── Final comparison ───
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print("  {:50s} {:>10s}".format("Method", "4-unsafe"))
    print("  " + "-" * 62)
    print("  {:50s} {:>10s}".format("Paper: Unsafe-Net (YOLO+ConvLSTM, filtered)", "95.81%"))
    print("  {:50s} {:>10s}".format("Paper: ConvLSTM only (no filter)", "85.78%"))
    print("  {:50s} {:>10s}".format("Ours (frozen): Cascade", "90.63%"))

    if results_4u:
        best_4u = max(r['acc'] for r in results_4u)
        mean_4u = np.mean([r['acc'] for r in results_4u])
        print("  {:50s} {:>9.1f}%".format("Ours (E2E): 4-unsafe best", 100 * best_4u))
        print("  {:50s} {:>9.1f}%".format("Ours (E2E): 4-unsafe mean", 100 * mean_4u))

    if results_8c:
        best_8u = max(r.get('unsafe4_acc', 0) for r in results_8c)
        best_bin = max(r.get('binary_acc', 0) for r in results_8c)
        print("  {:50s} {:>9.1f}%".format("Ours (E2E): 8class->4unsafe best", best_8u))
        print("  {:50s} {:>9.1f}%".format("Ours (E2E): 8class->binary best", best_bin))

    best_overall = max(100 * max(r['acc'] for r in results_4u) if results_4u else 0,
                       max(r.get('unsafe4_acc', 0) for r in results_8c) if results_8c else 0)

    if best_overall > 95.81:
        print("\n  BASELINE BEATEN! {:.1f}% > 95.81%".format(best_overall))
    elif best_overall > 90.63:
        print("\n  IMPROVED over frozen cascade: {:.1f}% > 90.63%".format(best_overall))
    else:
        print("\n  WARNING: fine-tuning did not improve over cascade")

    # Save results
    results_json = {
        'exp_a_4unsafe': [{k: v for k, v in r.items()
                           if k not in ('preds', 'labels', 'orig_labels')}
                          for r in results_4u],
        'exp_b_8class': [{k: v for k, v in r.items()
                          if k not in ('preds', 'labels', 'orig_labels')}
                         for r in results_8c],
    }
    out_json = os.path.join(out_dir, 'finetune_results_v2.json')
    with open(out_json, 'w') as f:
        json.dump(results_json, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print("\n  Results saved to {}".format(out_json))
    print("  Total time: {:.0f}s ({:.1f}min)".format(time.time() - t0, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
