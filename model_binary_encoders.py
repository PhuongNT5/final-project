"""Binary Safe/Unsafe encoder family E1/E2/E3 — see
plans/2026-09-06-thesis-2week-final/reports/design-2026-09-16-binary-encoder-redesign.md §2.

E0 = the existing RelationTransformer (model_relation_head.py), used unmodified via `n_cls=2`.
E2 subclasses it (only the positional table changes). E1/E3 are standalone modules with a
forward() contract matching E0's: `(cls_logits[, sub_logits][, frame_logits, frame_valid][, cls_repr])`,
so run_relation_transformer.py / run_kfold_relation.py can train all four archs identically.

All heads/loss are IDENTICAL across archs (design §3.2): 2-logit main + 8-way sub-class aux
(λ=0.3) + per-frame binary aux (λ=0.5), label smoothing 0.05 on all three.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from model_relation_head import RelationTransformer


def _pack(cls_logits, sub_logits=None, frame_logits=None, frame_valid=None, cls_repr=None):
    out = [cls_logits]
    if sub_logits is not None:
        out.append(sub_logits)
    if frame_logits is not None:
        out += [frame_logits, frame_valid]
    if cls_repr is not None:
        out.append(cls_repr)
    return tuple(out) if len(out) > 1 else out[0]


class E2FrameOnlyPos(RelationTransformer):
    """Surgical control: identical to E0 except the 129-slot positional table (frame index x
    confidence-rank entangled) is replaced by a 17-slot frame-only table, broadcast across the
    N_MAX slots of each frame. Isolates "was the defect the positional prior?" (design §2, E2)."""

    def __init__(self, n_cls=2, n_entity=8, d=128, layers=2, heads=4, ffn=256, dropout=0.3,
                 t=16, n_max=8, cls_embed_dim=32, n_num=73, n_sub=8):
        super().__init__(n_cls=n_cls, n_entity=n_entity, d=d, layers=layers, heads=heads, ffn=ffn,
                          dropout=dropout, t=t, n_max=n_max, cls_embed_dim=cls_embed_dim,
                          n_num=n_num, n_sub=n_sub)
        self.t, self.n_max = t, n_max
        self.pos = nn.Parameter(torch.zeros(1, t + 1, d))   # frame-only (was t*n_max+1)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def _positional(self, x):
        cls_pos = self.pos[:, :1]
        frame_pos = self.pos[:, 1:].repeat_interleave(self.n_max, dim=1)   # broadcast to N slots/frame
        return torch.cat([cls_pos, frame_pos], dim=1)[:, :x.size(1)]


class E1FSTE(nn.Module):
    """Factorized set-then-time encoder (design §2, E1 — the recommended proposal).

    Intra-frame stage: permutation-invariant (no positional embedding — box slots are an
    unordered set, not a sequence) TransformerEncoderLayer over the <=N_MAX entity tokens of
    each frame, masked mean||max pooled to one frame embedding. Temporal stage: [CLS] + 16 frame
    embeddings + the model's only positional table (17 slots), one more encoder layer, pooled
    CLS||mean||max -> heads.
    """

    def __init__(self, n_cls=2, n_entity=8, d=64, heads=4, dropout=0.2, t=16, n_max=8,
                 cls_embed_dim=16, n_num=73, n_sub=8):
        super().__init__()
        self.n_num, self.t, self.n_max = n_num, t, n_max
        self.cls_embed = nn.Embedding(n_entity, cls_embed_dim)
        self.num_proj = nn.Linear(n_num, d - cls_embed_dim)
        self.token_ln = nn.LayerNorm(d)
        intra = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, activation='gelu')
        self.intra_enc = nn.TransformerEncoder(intra, 1)
        self.frame_proj = nn.Linear(2 * d, d)
        self.empty_frame = nn.Parameter(torch.zeros(d))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.temporal_pos = nn.Parameter(torch.zeros(1, t + 1, d))
        temp = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, activation='gelu')
        self.temporal_enc = nn.TransformerEncoder(temp, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(3 * d, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, n_cls))
        self.sub_head = nn.Linear(3 * d, n_sub) if n_sub else None
        self.frame_head = nn.Linear(d, n_cls)
        for p in (self.cls_token, self.temporal_pos):
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, tok, mask, return_frame=False, return_repr=False, return_sub=False):
        B, T, N, _ = tok.shape
        cid = tok[..., 0].long().clamp(0, self.cls_embed.num_embeddings - 1)
        num = tok[..., 1:1 + self.n_num]
        x = self.token_ln(torch.cat([self.cls_embed(cid), self.num_proj(num)], dim=-1))   # (B,T,N,d)
        x = x.view(B * T, N, -1)
        m = mask.view(B * T, N)
        all_masked = (m == 0).all(dim=1, keepdim=True)               # empty frames -> no valid key
        kpm = (m == 0) & ~all_masked                                  # unmask everything to dodge NaN;
        x = self.intra_enc(x, src_key_padding_mask=kpm)               # output overwritten by empty_frame below
        mv = m.unsqueeze(-1)                                          # (B*T,N,1)
        cnt = mv.sum(1).clamp(min=1)
        mean_p = (x * mv).sum(1) / cnt
        max_p = (x.masked_fill(mv == 0, float('-inf'))).amax(1)
        max_p = torch.where(torch.isfinite(max_p), max_p, torch.zeros_like(max_p))
        f = self.frame_proj(torch.cat([mean_p, max_p], dim=-1)).view(B, T, -1)   # (B,T,d)
        frame_valid = mask.sum(-1) > 0                                # (B,T)
        f = torch.where(frame_valid.unsqueeze(-1), f, self.empty_frame.expand(B, T, -1))

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, f], dim=1) + self.temporal_pos[:, :T + 1]   # (B,T+1,d)
        seq = self.temporal_enc(self.drop(seq))                          # no masking: every frame slot valid
        cls_repr3 = torch.cat([seq[:, 0], seq[:, 1:].mean(1), seq[:, 1:].amax(1)], dim=-1)   # (B,3d)
        cls_logits = self.head(cls_repr3)
        sub_logits = self.sub_head(cls_repr3) if return_sub else None
        frame_logits = self.frame_head(seq[:, 1:]) if return_frame else None   # (B,T,n_cls)
        return _pack(cls_logits, sub_logits, frame_logits,
                     frame_valid if return_frame else None, cls_repr3 if return_repr else None)


class E3DeepSets(nn.Module):
    """Anti-overfitting floor (design §2, E3): no attention at all. Masked-mean box pooling per
    frame (DeepSets), a per-frame MLP, mean||max pooled over frames -> heads."""

    def __init__(self, n_cls=2, n_entity=8, d=64, t=16, n_max=8, cls_embed_dim=8, n_num=73, n_sub=8):
        super().__init__()
        self.n_num = n_num
        self.cls_embed = nn.Embedding(n_entity, cls_embed_dim)
        self.num_proj = nn.Linear(n_num, d - cls_embed_dim)
        self.token_ln = nn.LayerNorm(d)
        self.frame_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * d, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, n_cls))
        self.sub_head = nn.Linear(2 * d, n_sub) if n_sub else None
        self.frame_head = nn.Linear(d, n_cls)

    def forward(self, tok, mask, return_frame=False, return_repr=False, return_sub=False):
        B, T, N, _ = tok.shape
        cid = tok[..., 0].long().clamp(0, self.cls_embed.num_embeddings - 1)
        num = tok[..., 1:1 + self.n_num]
        x = self.token_ln(torch.cat([self.cls_embed(cid), self.num_proj(num)], dim=-1))   # (B,T,N,d)
        mv = mask.unsqueeze(-1)                                       # (B,T,N,1)
        pooled = (x * mv).sum(2) / mv.sum(2).clamp(min=1)              # (B,T,d) masked mean; empty frame -> 0
        f = self.frame_mlp(pooled)                                     # (B,T,d)
        cls_repr2 = torch.cat([f.mean(1), f.amax(1)], dim=-1)          # (B,2d)
        cls_logits = self.head(cls_repr2)
        sub_logits = self.sub_head(cls_repr2) if return_sub else None
        frame_valid = mask.sum(-1) > 0
        frame_logits = self.frame_head(f) if return_frame else None
        return _pack(cls_logits, sub_logits, frame_logits,
                     frame_valid if return_frame else None, cls_repr2 if return_repr else None)


ARCHES = {'E0': RelationTransformer, 'E2': E2FrameOnlyPos, 'E1': E1FSTE, 'E3': E3DeepSets}


def build_encoder(arch, n_num, n_cls=2, n_sub=8, d=None):
    """Factory used by run_relation_transformer.py / run_kfold_relation.py --arch selection.
    `d` overrides E1's width (64 primary, 96 capacity check); ignored by E0/E2/E3."""
    if arch not in ARCHES:
        raise ValueError(f"unknown arch {arch!r}, expected one of {sorted(ARCHES)}")
    cls = ARCHES[arch]
    if arch == 'E1':
        return cls(n_cls=n_cls, n_num=n_num, n_sub=n_sub, d=d or 64)
    return cls(n_cls=n_cls, n_num=n_num, n_sub=n_sub)
