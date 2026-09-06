"""Stage B - per-frame behaviour recogniser (the paper's YOLO-v4 role, done as a classifier).

The paper spends its 2,262-frame annotation budget on *behaviour at the frame level*; this module
reproduces that supervision placement by propagating each clip's label to its sampled frames and
fine-tuning an ImageNet backbone.  Training uses all 8 source classes (the safe counterparts are the
contrast that defines each unsafe class); inference marginalises the 8 logits onto the 4 unsafe ones.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision as tv

from . import config as C

BACKBONES = {
    'resnet34': (tv.models.resnet34, tv.models.ResNet34_Weights.IMAGENET1K_V1, 512),
    'resnet50': (tv.models.resnet50, tv.models.ResNet50_Weights.IMAGENET1K_V2, 2048),
    'effb0': (tv.models.efficientnet_b0, tv.models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
}


def _trunk(name: str):
    fn, weights, dim = BACKBONES[name]
    net = fn(weights=weights)
    if name.startswith('resnet'):
        body = nn.Sequential(*list(net.children())[:-1])       # -> (B, dim, 1, 1)
    else:
        body = nn.Sequential(net.features, net.avgpool)
    return body, dim


class FrameNet(nn.Module):
    """Backbone -> pooled embedding -> 8-class logits (+ 4-class view via marginalisation)."""

    def __init__(self, backbone: str = 'resnet34', n_cls: int = 8, p_drop: float = 0.3):
        super().__init__()
        self.body, self.dim = _trunk(backbone)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(self.dim, n_cls)
        self.n_cls = n_cls

    def features(self, x):
        return torch.flatten(self.body(x), 1)

    def forward(self, x, return_feat: bool = False):
        f = self.features(x)
        logits = self.fc(self.drop(f))
        return (logits, f) if return_feat else logits


def to4(logits8: torch.Tensor) -> torch.Tensor:
    """Marginalise 8-class logits onto the 4 unsafe classes.

    log p(unsafe_k | one of the 4) = logsumexp over the unsafe logit only.  The safe logits act as
    a learned rejection class during training; at test time only unsafe clips are scored, so the
    4-class posterior is the softmax restricted to columns 0..3.
    """
    return logits8[..., :4] if logits8.shape[-1] >= 4 else logits8


def pair_contrast(logits8: torch.Tensor) -> torch.Tensor:
    """Unsafe-vs-safe contrast score per behaviour family: logit(unsafe_k) - logit(safe_k).

    Encodes 'panel open rather than closed', 'overload rather than safe load', 'off-walkway rather
    than on-walkway' - the discriminative axis the 8-class head is trained on.
    """
    safe = torch.stack([logits8[..., C.SAFE_OF[k]] for k in range(4)], dim=-1)
    return logits8[..., :4] - safe


def build(backbone: str = 'resnet34', n_cls: int = 8, device: str = 'mps') -> FrameNet:
    return FrameNet(backbone, n_cls).to(device)
