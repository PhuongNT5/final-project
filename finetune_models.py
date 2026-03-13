#!/usr/bin/env python3
"""
Core model architectures for end-to-end fine-tuning.
- YOLOv11Backbone: partial unfreeze of YOLO backbone
- TransformerClassHead: multi-stream gated temporal fusion
- E2EModel: end-to-end backbone + head
"""
import os, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy


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
    def __init__(self, opt, warmup, total, min_lr=1e-6, last_epoch=-1):
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


class YOLOv11Backbone(nn.Module):
    """YOLOv11s backbone with hook-based feature extraction and partial unfreezing."""

    def __init__(self, weights_path='yolo11s.pt', pool_size=4,
                 feature_layer_idx=-3, unfreeze_last_n=3):
        super().__init__()
        from ultralytics import YOLO
        yolo = YOLO(weights_path)
        self.backbone = yolo.model
        self.feature_layer_idx = feature_layer_idx
        self.pool_size = pool_size

        layers = self.backbone.model
        self.target_idx = (feature_layer_idx if feature_layer_idx >= 0
                           else len(layers) + feature_layer_idx)

        for p in self.backbone.parameters():
            p.requires_grad = False

        if unfreeze_last_n > 0:
            start = max(0, self.target_idx - unfreeze_last_n + 1)
            n_unfrozen = 0
            for i in range(start, self.target_idx + 1):
                for p in layers[i].parameters():
                    p.requires_grad = True
                    n_unfrozen += 1
            print(f"  [Backbone] Unfroze layers {start}-{self.target_idx} ({n_unfrozen} params)")


    def _pool(self, x):
        """MPS-safe adaptive average pooling."""
        h, w = x.shape[2], x.shape[3]
        p = self.pool_size
        if h % p == 0 and w % p == 0:
            return F.adaptive_avg_pool2d(x, (p, p))
        new_h = max(p, (h // p) * p)
        new_w = max(p, (w // p) * p)
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
        _ = self.backbone(x)
        handle.remove()
        if features is None:
            raise RuntimeError("Feature hook returned None")
        f = self._pool(features)
        f = f.flatten(2).transpose(1, 2).contiguous()
        return f

    def get_backbone_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def get_feat_dim(self):
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            f = self.forward(dummy)
        return f.shape[-1]


class TransformerClassHead(nn.Module):
    """Multi-stream gated temporal fusion + Transformer encoder."""

    def __init__(self, in_dim=768, num_classes=8, model_dim=256,
                 nhead=8, num_layers=4, dropout=0.3, max_t=64):
        super().__init__()
        self.in_dim = in_dim
        self.proj_appear = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.proj_motion = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.proj_accel = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.proj_stats = nn.Sequential(
            nn.LayerNorm(in_dim * 3), nn.Linear(in_dim * 3, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.gate = nn.Sequential(
            nn.Linear(model_dim * 4, model_dim), nn.GELU(),
            nn.Linear(model_dim, 4), nn.Softmax(dim=-1))
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.type_embed = nn.Embedding(5, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nh = nhead
        while model_dim % nh != 0:
            nh -= 1
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

    def _windowed_stats(self, x, window=4):
        b, t, c = x.shape
        pad = window // 2
        xp = F.pad(x, (0, 0, pad, pad), mode='replicate')
        stats = []
        for i in range(t):
            w = xp[:, i:i + window, :]
            stats.append(torch.cat([w.mean(1), w.std(1), w.max(1).values], dim=-1))
        return torch.stack(stats, dim=1)

    def forward(self, tokens):
        if tokens.dim() == 4:
            x = tokens.mean(2)
        else:
            x = tokens
        b, t, c = x.shape
        d = x.device
        appear = self.proj_appear(x)
        d1 = x[:, 1:] - x[:, :-1]
        motion = self.proj_motion(d1)
        d2 = d1[:, 1:] - d1[:, :-1]
        accel = self.proj_accel(d2)
        win = self._windowed_stats(x)
        stats = self.proj_stats(win)
        pos = self.time_pos(torch.arange(t, device=d)).unsqueeze(0)
        appear = appear + pos + self.type_embed(torch.ones(t, dtype=torch.long, device=d)).unsqueeze(0)
        motion = motion + self.time_pos(torch.arange(t-1, device=d)).unsqueeze(0) + self.type_embed(torch.full((t-1,), 2, dtype=torch.long, device=d)).unsqueeze(0)
        accel = accel + self.time_pos(torch.arange(t-2, device=d)).unsqueeze(0) + self.type_embed(torch.full((t-2,), 3, dtype=torch.long, device=d)).unsqueeze(0)
        stats = stats + pos + self.type_embed(torch.full((t,), 4, dtype=torch.long, device=d)).unsqueeze(0)
        s_a, s_m, s_c, s_s = appear.mean(1), motion.mean(1), accel.mean(1), stats.mean(1)
        gates = self.gate(torch.cat([s_a, s_m, s_c, s_s], dim=-1))
        pad_m = F.pad(motion, (0, 0, 0, 1), value=0)
        pad_c = F.pad(accel, (0, 0, 0, 2), value=0)
        fused = (gates[:,0:1].unsqueeze(1) * appear + gates[:,1:2].unsqueeze(1) * pad_m +
                 gates[:,2:3].unsqueeze(1) * pad_c + gates[:,3:4].unsqueeze(1) * stats)
        cls = self.cls_token.expand(b, -1, -1) + self.type_embed(torch.zeros(1, dtype=torch.long, device=d)).unsqueeze(0)
        tokens_in = self.pre_norm(torch.cat([cls, fused], dim=1))
        tokens_out = self.transformer(tokens_in)
        return self.fc(self.post_norm(tokens_out[:, 0]))


class E2EModel(nn.Module):
    """End-to-end: YOLOv11 backbone + Transformer head."""
    def __init__(self, backbone, head, T=32):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.T = T

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
