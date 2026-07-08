"""
TransformerHead — copied verbatim from run_pairwise_contrastive.py.
DO NOT modify. Used by export-transformer-heads.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
