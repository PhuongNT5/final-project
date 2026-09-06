#!/usr/bin/env python3
"""Phase-01 frame sampler for the multi-entity annotation ontology.

Samples ~3,000 frames from the TRAIN clips only (the 64-clip test set is never
touched), viewpoint-balanced and oversampling the WalkwayViol↔UnauthorInt
confusion pair. Produces a manifest by default; `--extract` writes native-res
JPEGs ready for CVAT import. Train/val split is at the CLIP level so no clip's
frames cross the split (fixes the old `val: images/train` leakage).

The sampler does NOT assign entity labels — humans do that in CVAT against
annotation_guide/ontology-multi-entity.md. It only selects which frames to label.

Usage (from repo root):
  .venv/bin/python tools/sample_frames_for_annotation.py --dry-run        # manifest only
  .venv/bin/python tools/sample_frames_for_annotation.py --extract        # + write JPEGs
  .venv/bin/python tools/sample_frames_for_annotation.py --total 3000     # scale target
"""
import argparse
import csv
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRAIN = os.path.join(ROOT, 'data', 'train')
VIEW_MAP = os.path.join(ROOT, 'sweep_results', 'zone_prior', 'clip_view_map.json')
OUT_DIR = os.path.join(ROOT, 'annotation_guide')
MANIFEST = os.path.join(OUT_DIR, 'frame_manifest.csv')
# YOLO-ready layout: images/{train,val} written here; CVAT label export -> labels/{train,val}
DATASET_DIR = os.path.join(OUT_DIR, 'dataset')
FRAMES_DIR = os.path.join(DATASET_DIR, 'images')
T_NPY = 32  # frames per clip in the .npy linspace sampling

SRC_CLASS_NAMES = {
    0: 'safe_walkway_violation', 1: 'unauthorized_intervention',
    2: 'opened_panel_cover', 3: 'carrying_overload_with_forklift',
    4: 'safe_walkway', 5: 'authorized_intervention',
    6: 'closed_panel_cover', 7: 'safe_carrying',
}
# Per-source-class frame targets (sum = 3000). Confusion pair (0,1) is oversampled;
# their safe counterparts (4 in-walkway, 5 authorized-in-zone) are kept as contrastive negatives.
# Target realized total = 2,262 annotated frames — matches the paper's YOLO training-set size
# (Önal & Dandıl: 2,262 labelled images). Scarce safe classes (5,6,7) cap at 184/152/176
# (available clips × 8-frame cap) = 512; the remaining 1,750 is distributed over classes 0–4,
# prioritising the confusion pair (0,1).
# Realized = 550+510+290+200+200 + 184+152+176 = 2,262.
DEFAULT_TARGETS = {0: 550, 1: 510, 2: 290, 3: 200, 4: 200, 5: 350, 6: 250, 7: 250}
VAL_FRACTION = 0.15
SEED = 20260711


def class_folder(cls_id):
    for d in sorted(os.listdir(DATA_TRAIN)):
        if d.startswith(f'{cls_id}_') and os.path.isdir(os.path.join(DATA_TRAIN, d)):
            return d
    return None


def load_view_map():
    """basename -> view id (only the 4 unsafe classes are mapped; else 'na')."""
    if not os.path.exists(VIEW_MAP):
        return {}
    m = json.load(open(VIEW_MAP))
    return {r['basename']: r['view'] for r in m if r.get('subset') == 'train'}


def interleave_by_view(clips, view_of):
    """Round-robin clips across their views so a per-class budget spreads over all
    camera scenes instead of piling onto one view."""
    buckets = {}
    for c in clips:
        buckets.setdefault(view_of.get(c, 'na'), []).append(c)
    for v in buckets:
        buckets[v].sort()
    order, keys = [], sorted(buckets, key=lambda k: str(k))
    while any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                order.append(buckets[k].pop(0))
    return order


def frames_for_clip(k):
    """k spread frame indices in [0, T_NPY-1]."""
    if k <= 0:
        return []
    if k == 1:
        return [T_NPY // 2]
    return sorted(set(int(round(x)) for x in np.linspace(0, T_NPY - 1, k)))


def plan_class(cls_id, target, view_of, rng):
    folder = class_folder(cls_id)
    if folder is None:
        return []
    fdir = os.path.join(DATA_TRAIN, folder)
    clips = sorted(f for f in os.listdir(fdir) if f.endswith('.mp4'))
    if not clips:
        return []
    order = interleave_by_view(clips, view_of)

    # clip-level split: last VAL_FRACTION of a shuffled clip list -> val
    shuffled = order.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * VAL_FRACTION)))
    val_clips = set(shuffled[:n_val])

    # distribute target frames across clips (view-balanced order), cap 8/clip
    n = len(order)
    base, rem = divmod(target, n)
    rows = []
    for i, clip in enumerate(order):
        k = min(8, base + (1 if i < rem else 0))
        if k == 0 and len(rows) >= target:
            break
        if k == 0:
            k = 1  # ensure coverage until target met
        split = 'val' if clip in val_clips else 'train'
        view = view_of.get(clip, 'na')
        reason = 'confusion_pair' if cls_id in (0, 1) else (
            'contrastive_negative' if cls_id in (4, 5) else 'entity_coverage')
        for idx in frames_for_clip(k):
            rows.append({
                'source_class_id': cls_id, 'source_class': SRC_CLASS_NAMES[cls_id],
                'clip': clip, 'clip_path': os.path.join('data', 'train', folder, clip),
                'view': view, 'frame_idx': idx, 'split': split, 'reason': reason,
            })
    # trim to target (keep view-balanced prefix)
    return rows[:target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--total', type=int, default=None, help='override total (scales targets)')
    ap.add_argument('--extract', action='store_true', help='decode + write native-res JPEGs')
    ap.add_argument('--dry-run', action='store_true', help='manifest only (default)')
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    view_of = load_view_map()
    targets = dict(DEFAULT_TARGETS)
    if args.total:
        s = sum(targets.values())
        targets = {c: max(1, round(t * args.total / s)) for c, t in targets.items()}

    all_rows = []
    for cls_id in range(8):
        rows = plan_class(cls_id, targets[cls_id], view_of, rng)
        all_rows.extend(rows)
        n_tr = sum(r['split'] == 'train' for r in rows)
        print(f"  class {cls_id} {SRC_CLASS_NAMES[cls_id]:32s}: "
              f"{len(rows):4d} frames  (train {n_tr}, val {len(rows)-n_tr})")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MANIFEST, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['source_class_id', 'source_class', 'clip',
                                          'clip_path', 'view', 'frame_idx', 'split', 'reason'])
        w.writeheader()
        w.writerows(all_rows)

    # summaries
    from collections import Counter
    views = Counter(str(r['view']) for r in all_rows)
    splits = Counter(r['split'] for r in all_rows)
    reasons = Counter(r['reason'] for r in all_rows)
    print(f"\nTOTAL frames: {len(all_rows)}")
    print(f"  split:   {dict(splits)}")
    print(f"  view:    {dict(sorted(views.items()))}")
    print(f"  reason:  {dict(reasons)}")
    print(f"  manifest: {os.path.relpath(MANIFEST, ROOT)}")

    if args.extract:
        extract_frames(all_rows)
    else:
        print("\n(dry-run) no JPEGs written. Re-run with --extract to decode frames.")


def extract_frames(rows):
    """Open each source clip once; write one JPEG per manifest row. Idempotent
    (skips already-written files) with neighbour-frame fallback so a corrupt
    target frame still yields an image → the realized count matches the manifest."""
    import cv2
    from collections import defaultdict
    by_clip = defaultdict(list)
    for r in rows:
        by_clip[r['clip_path']].append(r)
    for split in ('train', 'val'):
        os.makedirs(os.path.join(FRAMES_DIR, split), exist_ok=True)
    written, skipped, missing = 0, 0, 0
    for clip_path, rs in by_clip.items():
        pending = []
        for r in rs:
            stem = f"{r['clip'].replace('.mp4', '')}_{int(r['frame_idx']):02d}.jpg"
            out = os.path.join(FRAMES_DIR, r['split'], stem)
            (pending.append((r, out)) if not os.path.exists(out) else None)
            skipped += os.path.exists(out)
        if not pending:
            continue
        cap = cv2.VideoCapture(os.path.join(ROOT, clip_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or T_NPY
        for r, out in pending:
            base = int(np.linspace(0, total - 1, T_NPY)[int(r['frame_idx'])])
            frame = None
            for off in (0, -1, -2, 1, -3, 2, -4, 3, -5):   # fallback around the target
                pos = min(max(0, base + off), max(0, total - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, f = cap.read()
                if ok:
                    frame = f
                    break
            if frame is None:
                missing += 1
                continue
            cv2.imwrite(out, frame)
            written += 1
        cap.release()
    print(f"\n[extract] wrote {written}, skipped {skipped} existing, {missing} unreadable "
          f"→ {written + skipped} total JPEGs under {os.path.relpath(FRAMES_DIR, ROOT)}/(train|val)")


if __name__ == '__main__':
    main()
