#!/usr/bin/env python3
"""
κ Stage 1 — prepare a blind inter-annotator subset (addresses Phản biện 2d).

Samples ~100 frames stratified across the 4 unsafe classes + background, copies them to a subset
folder with ANONYMOUS shuffled names (so annotator-2 cannot infer the class from the folder/name),
and writes:
  - kappa_groundtruth.csv   (anon_id, orig_path, annotator1_label)   ← your labels, hidden
  - kappa_annotator2.csv    (anon_id, label)                          ← BLANK, annotator-2 fills

annotator-1 label = primary class of the boxes in the frame (most frequent box class), or 'bg' if the
frame is empty/background. annotator-2 fills one of {0,1,2,3,bg} per anonymous image.

Usage:  .venv/bin/python3 run_kappa_prepare.py [--n 100] [--seed 0]
Then: annotator-2 opens annotation_guide/kappa_subset/*.jpg, fills the label column in
kappa_annotator2.csv, and you run run_kappa_compute.py.
"""
import os, csv, glob, shutil, random, argparse
from collections import Counter

FRAMES = 'annotation_guide/frames_to_annotate'
LABELS = 'annotation_guide/annotations'
OUT = 'annotation_guide/kappa_subset'


def a1_label(cls, name):
    """Annotator-1 label for a frame: primary box class (0-3) or 'bg'."""
    txt = os.path.join(LABELS, f'class_{cls}', name.replace('.jpg', '.txt'))
    if not os.path.exists(txt):
        return 'bg'
    ids = [l.split()[0] for l in open(txt).read().split('\n') if l.strip()]
    if not ids:
        return 'bg'
    return Counter(ids).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # collect (path, name, a1) for all unsafe-class frames
    pool = {'0': [], '1': [], '2': [], '3': [], 'bg': []}
    for cls in range(4):
        for jpg in sorted(glob.glob(f'{FRAMES}/class_{cls}/*.jpg')):
            name = os.path.basename(jpg)
            pool[a1_label(cls, name)].append((jpg, name, a1_label(cls, name)))

    # stratified sample: n/5 per bucket (0,1,2,3,bg)
    per = max(1, args.n // 5)
    picked = []
    for k in ['0', '1', '2', '3', 'bg']:
        items = pool[k][:]
        rng.shuffle(items)
        picked += items[:per]
    rng.shuffle(picked)

    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(f'{OUT}/*'):
        os.remove(f)
    gt, a2 = [], []
    for i, (path, name, a1) in enumerate(picked, 1):
        anon = f'k{i:03d}.jpg'
        shutil.copy2(path, os.path.join(OUT, anon))
        gt.append((anon, name, a1))
        a2.append((anon, ''))   # blank for annotator-2

    with open('annotation_guide/kappa_groundtruth.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['anon_id', 'orig_name', 'annotator1_label']); w.writerows(gt)
    with open('annotation_guide/kappa_annotator2.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['anon_id', 'label']); w.writerows(a2)

    dist = Counter(a1 for _, _, a1 in picked)
    print(f"  Sampled {len(picked)} frames → {OUT}/ (anonymous k001..)")
    print(f"  annotator-1 label distribution: {dict(sorted(dist.items()))}")
    print(f"  Labels: 0=walkway_violation 1=unauthorized_intervention 2=opened_panel "
          f"3=carrying_overload bg=background(no unsafe object)")
    print(f"\n  NEXT: annotator-2 views {OUT}/*.jpg and fills the 'label' column in")
    print(f"        annotation_guide/kappa_annotator2.csv (one of 0/1/2/3/bg per row),")
    print(f"        then run: .venv/bin/python3 run_kappa_compute.py")


if __name__ == '__main__':
    main()
