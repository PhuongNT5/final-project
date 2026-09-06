"""Stage C - temporal classifiers over per-frame evidence.

`mean` is the TSN-style average of per-frame posteriors (what the paper's ConvLSTM is compared
against); `lstm` is the paper-faithful recurrent head (a BiLSTM over per-frame CNN features is the
sequence-modelling role ConvLSTM plays once the CNN is separated out); `tfm` is the thesis's
Transformer head.  All three consume the same multi-stream frame vector so the comparison is clean.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, x, mask=None):
        a = self.w(x).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, -1e4)
        a = torch.softmax(a, -1).unsqueeze(-1)
        return (x * a).sum(1)


class TemporalNet(nn.Module):
    """(B,T,d_in) frame vectors + (B,d_clip) clip vector -> n_cls logits."""

    def __init__(self, d_in: int, d_clip: int = 0, kind: str = 'lstm', d: int = 128,
                 n_cls: int = 4, p: float = 0.3, layers: int = 1, heads: int = 4):
        super().__init__()
        self.kind = kind
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Dropout(p))
        if kind == 'lstm':
            self.rnn = nn.LSTM(d, d // 2, num_layers=layers, batch_first=True, bidirectional=True)
        elif kind == 'tfm':
            layer = nn.TransformerEncoderLayer(d, heads, d * 2, p, batch_first=True, norm_first=True)
            self.rnn = nn.TransformerEncoder(layer, layers)
            self.pos = nn.Parameter(torch.zeros(1, 512, d)); nn.init.trunc_normal_(self.pos, std=0.02)
        else:
            self.rnn = None
        self.pool = AttnPool(d)
        self.clip_proj = nn.Sequential(nn.LayerNorm(d_clip), nn.Linear(d_clip, d), nn.GELU()) if d_clip else None
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(p), nn.Linear(d, n_cls))
        self.frame_head = nn.Linear(d, n_cls)          # per-frame auxiliary supervision

    def forward(self, x, clip_feat=None, return_frames: bool = False):
        h = self.inp(x)
        if self.kind == 'lstm':
            h, _ = self.rnn(h)
        elif self.kind == 'tfm':
            h = self.rnn(h + self.pos[:, :h.shape[1]])
        z = self.pool(h)
        if self.clip_proj is not None and clip_feat is not None:
            z = z + self.clip_proj(clip_feat)
        out = self.head(z)
        return (out, self.frame_head(h)) if return_frames else out
