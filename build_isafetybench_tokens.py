#!/usr/bin/env python3
"""Stage B input builder for the iSafetyBench corpus (external dataset, binary hazard/normal).

Reuses `build_relation_tokens.process_clip` unchanged so the token layout is byte-compatible with the
Unsafe-Net tokens (T=16, N_MAX=8, FEAT=10 + APP_DIM=64 = 74) and every existing encoder (E0-E3) runs
on it without a model change.

Two deliberate differences from the Unsafe-Net builder, both forced by the dataset:
  * zones=None for every clip -> the 4 geometry channels are all zero. iSafetyBench ships no zone
    polygons. (Geometry was already proven dead for the walkway sub-task; see
    finding-2026-09-19-walkway-geometry-is-unsalvageable.md.)
  * the detector is COCO-pretrained, so its 80 classes are folded into 8 coarse entity groups
    (COCO_GROUPS below) to keep `nn.Embedding(n_entity=8, ...)` valid.

Grouping for cross-validation comes free: the filename encodes the source YouTube video id
(`0kog0iuGxOc_trim_34.mp4`), stored in the 'views' field so the existing leave-group-out driver works
unchanged. 0/150 sources carry both classes, so grouped CV is mandatory — see R41.

Usage:
  .venv/bin/python build_isafetybench_tokens.py --appearance --imgsz 640
"""
import argparse
import glob
import os
import re
import numpy as np
import torch
from ultralytics import YOLO

import build_relation_tokens as B

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data', 'isafetybench')
OUT = B.OUT
SRC_RE = re.compile(r'([A-Za-z0-9_-]{11})_')

# COCO 80 -> 8 coarse entity groups. Index 0 stays `person` so the geometry branch in process_clip
# (which keys on class in (0,1)) still refers to people; with zones=None it emits zeros regardless.
COCO_GROUPS = {
    0: [0],                                                        # person
    1: [1, 2, 3, 4, 5, 6, 7, 8],                                   # vehicle
    2: [24, 26, 28, 39, 40, 41, 27, 25],                           # carried / container
    3: [56, 57, 59, 60, 61, 13, 58],                               # furniture / fixture
    4: [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],               # electronics / appliance
    5: [32, 34, 35, 38, 43, 42, 44, 29, 30, 31, 33, 36, 37],       # tool / sports implement
    6: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],                   # animal
}
GROUP_NAMES = ['person', 'vehicle', 'carried', 'furniture', 'electronics', 'tool', 'animal', 'other']


def coco_to_group():
    """80-entry lookup: COCO class id -> coarse group id (7 = other)."""
    lut = np.full(80, 7, dtype=np.int64)
    for g, ids in COCO_GROUPS.items():
        for c in ids:
            if 0 <= c < 80:
                lut[c] = g
    return lut


def clips():
    """(path, binary label, source-video id) for every clip. hazard -> 1 (Unsafe), normal -> 0 (Safe)."""
    out = []
    for folder, label in (('hazard', 1), ('normal', 0)):
        for p in sorted(glob.glob(os.path.join(DATA, folder, '*.mp4'))):
            m = SRC_RE.search(os.path.basename(p))
            out.append((p, label, m.group(1) if m else os.path.basename(p)))
    return out


def build(det, device, app, lut, tag, limit=None):
    items = clips()
    if limit:
        items = items[:limit]
    all_tok, all_mask, labels, cids, srcs = [], [], [], [], []
    for k, (path, label, src) in enumerate(items):
        tok, msk = B.process_clip(path, det, None, device, app=app)   # zones=None -> geometry zeros
        cls = tok[:, :, 0].astype(np.int64)
        tok[:, :, 0] = lut[np.clip(cls, 0, 79)] * (msk > 0)           # remap COCO id -> group id
        all_tok.append(tok); all_mask.append(msk)
        labels.append(label); cids.append(os.path.basename(path)); srcs.append(src)
        if (k + 1) % 100 == 0:
            print(f"  [isb] {k+1}/{len(items)}", flush=True)
    layout = ['class', 'xc', 'yc', 'w', 'h', 'conf', 'iou_walkway', 'iou_restricted',
              'foot_in_walkway', 'norm_dist'] + ([f'app_{i}' for i in range(B.APP_DIM)] if app else [])
    tokens = torch.tensor(np.stack(all_tok))
    mask = torch.tensor(np.stack(all_mask))
    out = {'tokens': tokens, 'mask': mask, 'labels': torch.tensor(labels),
           'clip_ids': cids, 'views': srcs, 'feat_layout': layout,
           'entity_names': GROUP_NAMES, 'geometry_is_zero': True, 'dataset': 'isafetybench'}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'tokens_{tag}all.pt')
    torch.save(out, path)
    n_haz = int(sum(labels))
    print(f"[isb] {len(items)} clips -> {os.path.basename(path)} | feat={tokens.shape[-1]} "
          f"| detections={int(mask.sum())} | hazard={n_haz} normal={len(items)-n_haz} "
          f"| distinct sources={len(set(srcs))}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='cap clips (smoke test)')
    ap.add_argument('--device', default=None, help='mps/cpu/0 (inference — MPS is fine)')
    ap.add_argument('--detector', default=os.path.join(ROOT, 'yolo11s.pt'),
                    help='COCO-pretrained detector weights (the Unsafe-Net 8-class detector is '
                         'meaningless on this corpus)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--appearance', action='store_true', help='add per-box appearance feature')
    ap.add_argument('--appearance_backbone', default='mobilenet', choices=['mobilenet', 'yolo'])
    ap.add_argument('--out_tag', default='isb_')
    args = ap.parse_args()

    B.IMGSZ, B.CONF, B.CORRUPT = args.imgsz, args.conf, None
    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    app = B.get_appearance(device, args.appearance_backbone) if args.appearance else None
    print(f"detector={os.path.relpath(args.detector, ROOT)} device={device} imgsz={args.imgsz} "
          f"appearance={args.appearance} T={B.T} N_MAX={B.N_MAX}")
    build(YOLO(args.detector), device, app, coco_to_group(), args.out_tag, args.limit)


if __name__ == '__main__':
    main()
