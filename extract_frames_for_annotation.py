#!/usr/bin/env python3
"""
Stage 0 of phase-02 (MT1 bbox scale): extract frames for annotation, scaling the V10
400-frame set toward 1000 (pilot), fixing V10's "easy-frame / class-imbalance" flaw.

Frame identity convention (matches V10 labels): `{cls}_tr{N}_{idx}` where idx ∈ 0..31 is the
position in the 32-frame linspace sampling used to build `{video}.mp4.npy`. The actual video frame
= int(linspace(0, total-1, 32)[idx]); frames are re-extracted at NATIVE resolution for cleaner
annotation. Existing 401 V10 labels are recovered (their frames re-extracted) and reused; the rest
are sampled NEW, stratified, oversampling the confusion pair (cls0 WalkwayViol ↔ cls1 UnauthorInt).

Outputs:
  annotation_guide/frames_to_annotate/class_{c}/{cls}_tr{N}_{idx:02d}.jpg
  annotation_guide/frame_manifest.csv   (class, clip, idx, status[existing|new], path)

Usage:
  .venv/bin/python3 extract_frames_for_annotation.py            # pilot targets → ~1000
  .venv/bin/python3 extract_frames_for_annotation.py --dry-run  # plan only, no extraction
"""
import os, sys, csv, argparse, re
import numpy as np
import cv2

DATA_TRAIN = 'data/train'
ANNOT_DIR = 'annotation_guide/annotations'           # existing V10 labels: class_{c}/*.txt
FRAMES_DIR = 'annotation_guide/frames_to_annotate'   # output frames: class_{c}/*.jpg
MANIFEST = 'annotation_guide/frame_manifest.csv'
T_NPY = 32  # frames per clip in the .npy linspace sampling

# Pilot targets per class. ~1200 unsafe (oversample confusion pair cls0/cls1) + 400 safe negatives.
# Detector trains on the 4 unsafe classes; safe frames are background. Source: 566 train clips × 32 fr.
PILOT_TARGETS = {0: 350, 1: 350, 2: 250, 3: 250, 4: 100, 5: 100, 6: 100, 7: 100}

LABEL_RE = re.compile(r'^(\d)_tr(\d+)_(\d+)\.txt$')


def class_folder(cls_id):
    for d in sorted(os.listdir(DATA_TRAIN)):
        if d.startswith(f'{cls_id}_') and os.path.isdir(os.path.join(DATA_TRAIN, d)):
            return d
    return None


def existing_labels(cls_id):
    """Return set of (clip_name, idx) already annotated in V10 for this class."""
    d = os.path.join(ANNOT_DIR, f'class_{cls_id}')
    out = []
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        m = LABEL_RE.match(f)
        if m and int(m.group(1)) == cls_id:
            out.append((f'{cls_id}_tr{m.group(2)}', int(m.group(3))))
    return out


def list_clips(folder):
    """clip_name -> video path, for `{cls}_tr{N}.mp4` files (skip .npy)."""
    base = os.path.join(DATA_TRAIN, folder)
    clips = {}
    for f in sorted(os.listdir(base)):
        if f.lower().endswith(('.mp4', '.avi', '.mov')) and not f.endswith('.npy'):
            clips[os.path.splitext(f)[0]] = os.path.join(base, f)
    return clips


def frame_at(video_path, idx, rng_total=None):
    """Extract native-res frame at linspace position idx (0..T_NPY-1)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return None
    real = int(np.linspace(0, total - 1, T_NPY)[idx])
    cap.set(cv2.CAP_PROP_POS_FRAMES, real)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def sample_new(cls_id, clips, existing, n_new, seed=0):
    """Round-robin clips, spread idx across 0..31, skip existing → n_new (clip, idx) pairs."""
    rng = np.random.RandomState(seed + cls_id)
    have = set(existing)
    names = sorted(clips.keys())
    rng.shuffle(names)
    # spread idx order: middle-out so we don't always grab frame 0 (V10's "easy" bias)
    idx_order = [16, 8, 24, 4, 12, 20, 28, 0, 2, 6, 10, 14, 18, 22, 26, 30,
                 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    picked, ci, depth = [], 0, 0
    while len(picked) < n_new and depth < T_NPY:
        name = names[ci % len(names)]
        idx = idx_order[depth]
        key = (name, idx)
        if key not in have:
            have.add(key); picked.append(key)
        ci += 1
        if ci % len(names) == 0:
            depth += 1
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(FRAMES_DIR, exist_ok=True)
    rows, totals = [], {}
    for cls_id in range(8):
        folder = class_folder(cls_id)
        if folder is None:
            print(f"  ! no folder for class {cls_id}"); continue
        clips = list_clips(folder)
        exist = existing_labels(cls_id)
        target = PILOT_TARGETS[cls_id]
        n_new = max(0, target - len(exist))
        new = sample_new(cls_id, clips, exist, n_new, args.seed)
        totals[cls_id] = {'existing': len(exist), 'new': len(new), 'target': target,
                          'clips': len(clips)}
        print(f"  class {cls_id} ({folder}): {len(clips)} clips | existing {len(exist)} "
              f"+ new {len(new)} → {len(exist)+len(new)} (target {target})")
        for (clip, idx), status in [(p, 'existing') for p in exist] + [(p, 'new') for p in new]:
            rows.append((cls_id, clip, idx, status))

    print(f"\n  TOTAL: existing {sum(t['existing'] for t in totals.values())} "
          f"+ new {sum(t['new'] for t in totals.values())} = "
          f"{sum(t['existing']+t['new'] for t in totals.values())} frames")

    if args.dry_run:
        print("\n  [dry-run] no extraction. Targets above; re-run without --dry-run to extract.")
        return

    # extract
    n_ok, n_fail = 0, 0
    man = []
    for cls_id in range(8):
        folder = class_folder(cls_id)
        if folder is None:
            continue
        clips = list_clips(folder)
        out_dir = os.path.join(FRAMES_DIR, f'class_{cls_id}')
        os.makedirs(out_dir, exist_ok=True)
        cls_rows = [r for r in rows if r[0] == cls_id]
        for k, (_, clip, idx, status) in enumerate(cls_rows):
            vp = clips.get(clip)
            out = os.path.join(out_dir, f'{clip}_{idx:02d}.jpg')
            if os.path.exists(out):
                man.append((cls_id, clip, idx, status, out)); n_ok += 1; continue
            if vp is None:
                n_fail += 1; continue
            fr = frame_at(vp, idx)
            if fr is None:
                n_fail += 1; continue
            cv2.imwrite(out, fr)
            man.append((cls_id, clip, idx, status, out)); n_ok += 1
            if (k + 1) % 25 == 0:
                print(f"    class {cls_id}: {k+1}/{len(cls_rows)}", flush=True)

    with open(MANIFEST, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['class', 'clip', 'idx', 'status', 'path']); w.writerows(man)
    print(f"\n  Extracted {n_ok} frames ({n_fail} failed) → {FRAMES_DIR}/")
    print(f"  Manifest → {MANIFEST}")
    print(f"\n  Next: run auto_annotate.py on the NEW frames, then review/correct in CVAT.")


if __name__ == '__main__':
    main()
