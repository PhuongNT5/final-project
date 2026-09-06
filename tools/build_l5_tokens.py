#!/usr/bin/env python3
"""Phase 1 · L5 — view-agnostic relational-context features (only entities the detector actually emits).

Structural finding (D2 + token audit): the detector emits boxes for 5/8 classes only — Person_Authorized,
Person_Unauthorized, Panel_Closed, Panel_Opened, Forklift_Overload. Safe_Walkway, Restricted_Zone,
Forklift_Safe are NEVER detected, so proximity-to-zone-box is identically zero. The WalkwayViol↔UnauthorInt
pair (both Person_Unauthorized, no zone boxes) has no distinguishing detected entity; the one remaining
view-agnostic signal is spatial context among detected boxes: person↔person density (walkway = shared path
with others; intrusion = more isolated) and person↔nearest-Panel/Forklift offset.

Per person box (class 0/1), within the same frame (all translation-invariant, no absolute coords):
  [n_persons, nn_person_dist, mean_person_dist, n_within_0p2,
   panel_present, panel_dist, fork_present, fork_dist]  -> 8 features.

Reads yolo26app_ (retains xc,yc to compute offsets); --drop_xcyc strips absolute centre (compound on L1).
Usage: build_l5_tokens.py --src yolo26app_ --dst yolo26L15_ --drop_xcyc
"""
import argparse, os
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = os.path.join(ROOT, 'sweep_results', 'relation_tokens')
PERSON = (0, 1)
PANEL = (4, 5)          # Panel_Closed, Panel_Opened
FORK = (7,)             # Forklift_Overload (only forklift class detected)
NEW_NAMES = ['n_persons', 'nn_person_dist', 'mean_person_dist', 'n_within_0p2',
             'panel_present', 'panel_dist', 'fork_present', 'fork_dist']
RADIUS = 0.2


def nearest_dist(px, py, boxes):
    if not boxes:
        return 0.0, 0.0
    ds = [np.sqrt((px - bx) ** 2 + (py - by) ** 2) for (bx, by) in boxes]
    return 1.0, float(min(ds))


def convert(src, dst, split, drop_xcyc):
    d = torch.load(os.path.join(TOK, f'tokens_{src}{split}.pt'), weights_only=False)
    tok = d['tokens'].numpy()
    mask = d['mask'].numpy()
    N, T, B, F = tok.shape
    newf = np.zeros((N, T, B, 8), dtype=np.float32)
    for i in range(N):
        for t in range(T):
            valid = [b for b in range(B) if mask[i, t, b] >= 0.5]
            cxy = {b: (float(tok[i, t, b, 1]), float(tok[i, t, b, 2])) for b in valid}
            cls = {b: int(round(tok[i, t, b, 0])) for b in valid}
            persons = [b for b in valid if cls[b] in PERSON]
            panels = [cxy[b] for b in valid if cls[b] in PANEL]
            forks = [cxy[b] for b in valid if cls[b] in FORK]
            for b in persons:
                px, py = cxy[b]
                others = [cxy[o] for o in persons if o != b]
                if others:
                    ds = [np.sqrt((px - ox) ** 2 + (py - oy) ** 2) for (ox, oy) in others]
                    nn, meand = float(min(ds)), float(np.mean(ds))
                    nwithin = float(sum(1 for x in ds if x <= RADIUS))
                else:
                    nn = meand = nwithin = 0.0
                pp, pd = nearest_dist(px, py, panels)
                fp, fd = nearest_dist(px, py, forks)
                newf[i, t, b] = [float(len(persons)), nn, meand, nwithin, pp, pd, fp, fd]
    out = np.concatenate([tok, newf], axis=-1)
    layout = list(d.get('feat_layout', [f'f{k}' for k in range(F)])) + NEW_NAMES
    if drop_xcyc:
        col_keep = [j for j in range(out.shape[-1]) if j not in (1, 2)]
        out = out[..., col_keep]; layout = [layout[j] for j in col_keep]
    d['tokens'] = torch.from_numpy(out.astype(np.float32)).contiguous()
    d['feat_layout'] = layout
    outp = os.path.join(TOK, f'tokens_{dst}{split}.pt')
    torch.save(d, outp)
    print(f"  {split}: {tok.shape} -> {tuple(d['tokens'].shape)}  [saved] {outp}")
    print(f"    layout[:14]: {layout[:14]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='yolo26app_')
    ap.add_argument('--dst', default='yolo26L15_')
    ap.add_argument('--drop_xcyc', action='store_true', help='strip absolute xc,yc (compound on L1)')
    args = ap.parse_args()
    for sp in ('train', 'test'):
        convert(args.src, args.dst, sp, args.drop_xcyc)


if __name__ == '__main__':
    main()
