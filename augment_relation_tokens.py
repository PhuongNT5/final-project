#!/usr/bin/env python3
"""v2a — enrich cached relation tokens with person<->entity proximity features (no detector re-run).

For every person token (class 0/1) in a frame, append distances (normalized coord space) to the
nearest Panel, nearest Opened-Panel, and nearest Forklift among the same frame's detections:
  "person next to an opened panel" -> UnauthorizedIntervention cue;
  "person off-walkway, far from machines" -> WalkwayViolation cue.
feat 10 -> 13. Non-person tokens get neutral 1.5 ('far'). Output: tokens_v2_{train,test}.pt

Usage: .venv/bin/python augment_relation_tokens.py
"""
import os
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(ROOT, 'sweep_results', 'relation_tokens')
FAR = 1.5                          # sentinel for "no such entity in frame"
PANEL = {4, 5}; OPENED = {5}; FORKLIFT = {6, 7}


def nearest(px, py, cxs, cys):
    if len(cxs) == 0:
        return FAR
    d = ((cxs - px) ** 2 + (cys - py) ** 2) ** 0.5
    return float(d.min())


def augment(sp):
    d = torch.load(os.path.join(TOK, f'tokens_{sp}.pt'), weights_only=False)
    tok, mask = d['tokens'], d['mask']                # (Nc,T,N,10), (Nc,T,N)
    Nc, T, N, _ = tok.shape
    extra = torch.full((Nc, T, N, 3), FAR)
    for c in range(Nc):
        for t in range(T):
            m = mask[c, t] > 0
            if m.sum() == 0:
                continue
            cls = tok[c, t, :, 0]; xc = tok[c, t, :, 1]; yc = tok[c, t, :, 2]
            def cents(ids):
                sel = m & torch.tensor([int(v) in ids for v in cls])
                return xc[sel], yc[sel]
            pxs, pys = cents(PANEL); oxs, oys = cents(OPENED); fxs, fys = cents(FORKLIFT)
            for n in range(N):
                if not m[n] or int(cls[n]) not in (0, 1):
                    continue
                px, py = float(xc[n]), float(yc[n])
                extra[c, t, n, 0] = nearest(px, py, pxs, pys)
                extra[c, t, n, 1] = nearest(px, py, oxs, oys)
                extra[c, t, n, 2] = nearest(px, py, fxs, fys)
    d['tokens'] = torch.cat([tok, extra], dim=-1)     # (Nc,T,N,13)
    d['feat_layout'] = d['feat_layout'] + ['d_panel', 'd_opened_panel', 'd_forklift']
    torch.save(d, os.path.join(TOK, f'tokens_v2_{sp}.pt'))
    print(f"[{sp}] {Nc} clips -> tokens_v2_{sp}.pt  feat={d['tokens'].shape[-1]}  layout={d['feat_layout']}")


if __name__ == '__main__':
    for sp in ('train', 'test'):
        augment(sp)
