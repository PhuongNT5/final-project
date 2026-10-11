"""R48 architectures — factorised space-time attention over YOLOv11 entity tokens (G14).

Three modules, deliberately sharing one temporal block:

  `TemporalTransformer`   attends over T frame states -> clip state. Used by BOTH arms.
  `EntitySpatioTemporal`  ESTA: stage 1 attends over entities within a frame -> frame state,
                          stage 2 is the shared temporal block. The architecture G14 tests.
  `WholeFrameTemporal`    B1: one CLIP embedding per frame straight into the SAME temporal block.

The sharing is the point. B1 is not a weaker re-implementation of the baseline — it is ESTA with the
entity axis deleted and nothing else changed: same backbone, same frames, same temporal block, same
width, same depth, same regularisation. A G14 win therefore isolates the entity decomposition and
cannot be explained by backbone, capacity or training budget.

Capacity is held near E0's 298,118 params on purpose, so "the new model is simply bigger" is not an
available explanation either.
"""
import torch
import torch.nn as nn

CLIP_DIM = 512
N_ENTITY = 8          # 8 entity rows; cls_emb is relearned per run, so the row meaning
                      # follows the token build: coarse COCO groups
                      # (build_isafetybench_tokens.GROUP_NAMES), or the 8 fine-tuned
                      # factory classes when built with --class_map identity
N_BOX = 5             # xc, yc, w, h, conf


def _encoder_layer(d, heads, ff, dropout):
    return nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff,
                                      dropout=dropout, batch_first=True, norm_first=True,
                                      activation='gelu')


class TemporalTransformer(nn.Module):
    """Attend over T frame states with a learned CLS and learned temporal positions.

    `valid` marks frames that carry information; an all-invalid clip falls back to attending over
    every frame rather than emitting a NaN (softmax over a fully-masked row is undefined).
    """

    def __init__(self, d=128, heads=4, ff=256, layers=1, dropout=0.1, t_max=16):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, t_max, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.enc = nn.TransformerEncoder(_encoder_layer(d, heads, ff, dropout), layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, frames, valid):
        b, t, _ = frames.shape
        x = frames + self.pos[:, :t]
        x = torch.cat([self.cls.expand(b, -1, -1), x], 1)
        dead = valid.sum(1) == 0
        valid = torch.where(dead.unsqueeze(1), torch.ones_like(valid), valid)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), valid < 0.5], 1)
        return self.norm(self.enc(x, src_key_padding_mask=pad)[:, 0])


class _Tokeniser(nn.Module):
    """CLIP embedding + entity class + box geometry -> one d-dim token."""

    def __init__(self, d, dropout):
        super().__init__()
        self.app = nn.Linear(CLIP_DIM, d)
        self.cls_emb = nn.Embedding(N_ENTITY, d)
        self.box = nn.Sequential(nn.Linear(N_BOX, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, app, geom):
        cid = geom[..., 0].long().clamp(0, N_ENTITY - 1)
        return self.drop(self.norm(self.app(app) + self.cls_emb(cid) + self.box(geom[..., 1:])))


class EntitySpatioTemporal(nn.Module):
    """ESTA — stage 1 over entities within each frame, stage 2 over frames (shared block)."""

    def __init__(self, d=128, heads=4, ff=256, s_layers=1, t_layers=1, dropout=0.1,
                 n_cls=2, t_max=16):
        super().__init__()
        self.tok = _Tokeniser(d, dropout)
        self.frame_cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.frame_cls, std=0.02)
        self.spatial = nn.TransformerEncoder(_encoder_layer(d, heads, ff, dropout), s_layers)
        self.snorm = nn.LayerNorm(d)
        self.temporal = TemporalTransformer(d, heads, ff, t_layers, dropout, t_max)
        self.head = nn.Linear(d, n_cls)

    def forward(self, app, geom, mask):
        return self.head(self.encode(app, geom, mask))

    def encode(self, app, geom, mask):
        """-> the clip state, before the classifier. Split out so TwoStreamFusion reuses this arm
        exactly as G14 evaluated it, rather than a copy that could drift from it."""
        b, t, n, _ = app.shape
        x = self.tok(app, geom).reshape(b * t, n, -1)
        x = torch.cat([self.frame_cls.expand(b * t, -1, -1), x], 1)
        m = mask.reshape(b * t, n)
        # A frame with zero detections would mask every entity; keep its CLS alive and let the
        # temporal stage drop the frame instead (an empty frame is a fact about the clip, not a bug).
        empty = m.sum(1) == 0
        m = torch.where(empty.unsqueeze(1), torch.ones_like(m), m)
        pad = torch.cat([torch.zeros(b * t, 1, dtype=torch.bool, device=x.device), m < 0.5], 1)
        frames = self.snorm(self.spatial(x, src_key_padding_mask=pad)[:, 0]).reshape(b, t, -1)
        return self.temporal(frames, (mask.sum(-1) > 0).float())


class WholeFrameTemporal(nn.Module):
    """B1 — ESTA with the entity axis deleted. Same backbone, frames, temporal block and width."""

    def __init__(self, d=128, heads=4, ff=256, t_layers=1, dropout=0.1, n_cls=2, t_max=16,
                 in_dim=CLIP_DIM):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d), nn.Dropout(dropout))
        self.temporal = TemporalTransformer(d, heads, ff, t_layers, dropout, t_max)
        self.head = nn.Linear(d, n_cls)

    def forward(self, frames, valid):
        return self.head(self.encode(frames, valid))

    def encode(self, frames, valid):
        """-> the clip state, before the classifier. See EntitySpatioTemporal.encode."""
        return self.temporal(self.proj(frames), valid)


class TwoStreamFusion(nn.Module):
    """TSF (G18) — the entity stream and the whole-frame stream, fused by attention.

    G14 settled that the entity decomposition does not *replace* a whole-frame representation. What
    it left open is whether it *complements* one, and the measured answer was yes: on the LSO-1100
    OOF vectors the two arms' correctness vectors correlate at phi=0.49, against 0.86 for two
    whole-frame variants of the same family, with 91.2% oracle against 84.4% best-single. They fail
    on different clips.

    So both arms are kept whole -- `EntitySpatioTemporal` and `WholeFrameTemporal` exactly as G14
    evaluated them -- and their clip states become a 2-token sequence that a CLS attends over. A
    learned stream embedding tells the fusion block which token is which; the fusion is learned
    jointly with both arms, not a post-hoc average of two frozen models (that shortcut is what the
    banned complementarity sweep did, and its numbers are selection statistics, not measurements).

    Auxiliary heads on each arm carry lambda=esta_training.AUX_W of the loss, the same weight the binary relation
    heads in this repo already use. Without them the fusion block can starve one arm and the model
    quietly collapses to a single stream, which would make a PASS unreadable.
    """

    def __init__(self, d=128, heads=4, ff=256, dropout=0.1, n_cls=2, t_max=16, f_layers=1):
        super().__init__()
        self.entity = EntitySpatioTemporal(d, heads, ff, 1, 1, dropout, n_cls, t_max)
        self.frame = WholeFrameTemporal(d, heads, ff, 1, dropout, n_cls, t_max)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.stream = nn.Parameter(torch.zeros(1, 2, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.stream, std=0.02)
        self.fuse = nn.TransformerEncoder(_encoder_layer(d, heads, ff, dropout), f_layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_cls)

    def forward(self, app, geom, mask, frames, valid):
        """-> (fused logits, entity-arm logits, frame-arm logits). The aux logits are returned
        rather than scored internally so the loss stays visible in the training loop."""
        e = self.entity.encode(app, geom, mask)
        f = self.frame.encode(frames, valid)
        x = torch.stack([e, f], 1) + self.stream
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1)
        z = self.norm(self.fuse(x)[:, 0])
        return self.head(z), self.entity.head(e), self.frame.head(f)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
