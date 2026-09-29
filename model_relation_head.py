"""Box-token spatial-relation Transformer (Stage B) — classifies the 4 unsafe behaviours
from per-clip entity-token sequences produced by build_relation_tokens.py.

Input per clip: tokens (T, N_MAX, 1 + n_num) + mask (T, N_MAX).
  feat = [class_id, xc, yc, w, h, conf, iou_walkway, iou_restricted, foot_in_walkway, norm_dist]
n_num is set from the token file, not fixed: the geometry-only tokens give n_num=9 (feat 10), while
the headline tag `yolo26app_` appends 64 appearance dims -> n_num=73 (feat 74), and `yolo26L1_`
drops xc,yc -> n_num=71 (feat 72).
The entity class is embedded; the 9 numeric features (coords + conf + relative geometry) are projected;
[CLS] + all box tokens go through a Transformer encoder; the CLS embedding → 4-class logits.

Self-attention learns the relation the appearance features can't: e.g. Person_Unauthorized × high
iou_restricted → Unauthorized Intervention; person with foot_in_walkway=0 → Walkway Violation.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class RelationTransformer(nn.Module):
    def __init__(self, n_cls=4, n_entity=8, d=128, layers=2, heads=4, ffn=256,
                 dropout=0.3, t=16, n_max=8, cls_embed_dim=32, n_num=9, n_sub=None):
        super().__init__()
        self.n_num = n_num                                   # numeric features per token (= feat_dim - 1)
        self.n_sub = n_sub                                    # E0/E2 8-way sub-class aux head (binary recipe only)
        self.cls_embed = nn.Embedding(n_entity, cls_embed_dim)
        self.num_proj = nn.Linear(n_num, d - cls_embed_dim)  # coords,conf,geometry[,relations][,appearance]
        self.token_ln = nn.LayerNorm(d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, t * n_max + 1, d))   # learned positional
        enc = nn.TransformerEncoderLayer(d, heads, ffn, dropout, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc, layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, n_cls))
        self.frame_head = nn.Linear(d, n_cls)   # S1a: per-frame behaviour head (auxiliary supervision)
        if n_sub:
            self.sub_head = nn.Linear(d, n_sub)               # 8-way sub-class aux (design §3.2)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def _positional(self, x):
        """Hook so E2 (frame-only positional table) can subclass without duplicating forward()."""
        return self.pos[:, :x.size(1)]

    def forward(self, tok, mask, return_frame=False, return_repr=False, return_sub=False):
        # tok (B,T,N,10)  mask (B,T,N) 1=valid
        # return_repr: also emit the CLS embedding (pre-head, d-dim) — used by the DANN/GRL domain head
        #   (run_lco_grl.py) to make the representation camera-invariant. Off by default → the champion
        #   path returns exactly what it did before.
        # return_sub: also emit the 8-way sub-class aux logits (binary recipe only; requires n_sub set).
        B, T, N, _ = tok.shape
        cid = tok[..., 0].long().clamp(0, self.cls_embed.num_embeddings - 1)
        num = tok[..., 1:1 + self.n_num]                     # (B,T,N,n_num)
        x = torch.cat([self.cls_embed(cid), self.num_proj(num)], dim=-1)   # (B,T,N,d)
        x = self.token_ln(x).view(B, T * N, -1)
        m = mask.view(B, T * N)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                        # (B, 1+T*N, d)
        x = self.drop(x + self._positional(x))
        # key_padding_mask: True = ignore; CLS (col 0) always attended
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), (m == 0)], dim=1)
        x = self.enc(x, src_key_padding_mask=kpm)
        cls_repr = x[:, 0]                                    # (B,d) pooled clip representation
        cls_logits = self.head(cls_repr)                     # CLS → (B, n_cls)
        out = [cls_logits]
        if return_sub:
            out.append(self.sub_head(cls_repr))
        if not return_frame:
            if return_repr:
                out.append(cls_repr)
            return tuple(out) if len(out) > 1 else out[0]
        tok_out = x[:, 1:].view(B, T, N, -1)                  # per-token encoder outputs
        mfr = mask.view(B, T, N, 1)
        frame_emb = (tok_out * mfr).sum(2) / mfr.sum(2).clamp(min=1)   # (B,T,d) masked-mean per frame
        frame_logits = self.frame_head(frame_emb)            # (B,T,n_cls)
        frame_valid = mask.sum(-1) > 0                        # (B,T) frames with >=1 detection
        out += [frame_logits, frame_valid]
        if return_repr:
            out.append(cls_repr)
        return tuple(out)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
