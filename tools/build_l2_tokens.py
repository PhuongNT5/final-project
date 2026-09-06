#!/usr/bin/env python3
"""Phase 1 · L2 — homography-registered zone geometry (the promoted top lever, per D2).

The current tokens' 4 geometry cols [iou_walkway, iou_restricted, foot_in_walkway, norm_dist] use ONE
canonical polygon per view with no per-clip alignment -> foot_in_walkway ~0.12 for BOTH confused classes
(the polygons don't line up). This replaces them with 5 homography-registered signed-distance features
[in_walkway, sdist_walkway, in_restricted, sdist_restricted, green_under_foot] from unsafenet/geometry.py:
each clip's static background is ORB+RANSAC-registered to its view's reference background, polygons warped
through that homography, then a signed distance (continuous, >0 inside) of each person's foot-point.

Reads yolo26app_ tokens (which retain xc,yc,w,h needed to recompute geometry); YOLO detections and
appearance are reused untouched. --drop_xcyc also strips absolute centre (produces the L1+L2 combo).

Usage: build_l2_tokens.py --src yolo26app_ --dst yolo26L2_ [--drop_xcyc]
"""
import argparse, os, sys
import numpy as np
import torch
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TOK = os.path.join(ROOT, 'sweep_results', 'relation_tokens')

from unsafenet import data_index as DI, geometry as G, frame_cache as FC  # noqa: E402

OLD_GEOM = [6, 7, 8, 9]  # iou_walkway, iou_restricted, foot_in_walkway, norm_dist  (replaced)
NEW_NAMES = ['in_walkway', 'sdist_walkway', 'in_restricted', 'sdist_restricted', 'green_under_foot']
PERSON = (0, 1)


def clip_cache(clips):
    """basename -> (warped polygons, green mask) computed once per clip."""
    cache = {}
    for c in clips:
        try:
            polys = G.clip_polygons(c, c.view)
        except Exception:
            polys = {'walkway': [], 'restricted': []}
        bgp = FC.bg_path(c)
        bg = cv2.imread(bgp) if os.path.exists(bgp) else None
        gmask = G.green_walkway_mask(bg) if bg is not None else None
        cache[c.name] = (polys, gmask)
    return cache


def convert(src, dst, split, cache, drop_xcyc, keep_old=False):
    d = torch.load(os.path.join(TOK, f'tokens_{src}{split}.pt'), weights_only=False)
    tok = d['tokens'].numpy()                    # (N,T,B,feat)
    mask = d['mask'].numpy()
    cids = [str(c) for c in d['clip_ids']]
    N, T, B, F = tok.shape
    # keep_old (L2b, augment): retain the canonical 4 geom too (append new geom); else replace them (L2)
    drop = [] if keep_old else OLD_GEOM
    keep = [i for i in range(F) if i not in drop]
    base = tok[..., keep]                                # (N,T,B, F) or (N,T,B, F-4)
    newgeom = np.zeros((N, T, B, 5), dtype=np.float32)
    n_miss = 0
    for i, cid in enumerate(cids):
        polys, gmask = cache.get(cid, (None, None))
        if polys is None:
            n_miss += 1; continue
        for t in range(T):
            for b in range(B):
                if mask[i, t, b] < 0.5:
                    continue
                cls = int(round(tok[i, t, b, 0]))
                if cls not in PERSON:
                    continue
                xc, yc, w, h = tok[i, t, b, 1:5].tolist()
                newgeom[i, t, b] = G.person_zone_feats([xc, yc, w, h], polys, gmask)
    # reassemble: [class,xc,yc,w,h,conf, <5 new geom>, <rest: (old geom if keep_old) + appearance>]
    head = base[..., :6]          # class,xc,yc,w,h,conf
    rest = base[..., 6:]          # (old geom if keep_old) + appearance
    out = np.concatenate([head, newgeom, rest], axis=-1)
    old_names = ['iou_walkway', 'iou_restricted', 'foot_in_walkway', 'norm_dist'] if keep_old else []
    rest_names = old_names + [f'app_{k}' for k in range(rest.shape[-1] - len(old_names))]
    layout = ['class', 'xc', 'yc', 'w', 'h', 'conf'] + NEW_NAMES + rest_names
    if drop_xcyc:
        col_keep = [j for j in range(out.shape[-1]) if j not in (1, 2)]
        out = out[..., col_keep]; layout = [layout[j] for j in col_keep]
    d['tokens'] = torch.from_numpy(out.astype(np.float32)).contiguous()
    d['feat_layout'] = layout
    outp = os.path.join(TOK, f'tokens_{dst}{split}.pt')
    torch.save(d, outp)
    # quick separation diagnostic: mean sdist_walkway by person class among valid person tokens
    print(f"  {split}: {tok.shape} -> {tuple(d['tokens'].shape)}  miss={n_miss}  [saved] {outp}")
    print(f"    layout[:12]: {layout[:12]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='yolo26app_')
    ap.add_argument('--dst', default='yolo26L2_')
    ap.add_argument('--drop_xcyc', action='store_true', help='also strip absolute xc,yc (L1+L2 combo)')
    ap.add_argument('--keep_old', action='store_true', help='L2b: augment (retain canonical 4 geom, add 5 new)')
    args = ap.parse_args()
    clips = DI.load_clips(labels=[0, 1, 2, 3])
    print(f"clips indexed: {len(clips)} | building per-clip homography + zone cache ...")
    cache = clip_cache(clips)
    ok = sum(1 for v in cache.values() if v[0] is not None and (v[0]['walkway'] or v[0]['restricted']))
    print(f"  clips with non-empty warped polygons: {ok}/{len(cache)}")
    for sp in ('train', 'test'):
        convert(args.src, args.dst, sp, cache, args.drop_xcyc, args.keep_old)


if __name__ == '__main__':
    main()
