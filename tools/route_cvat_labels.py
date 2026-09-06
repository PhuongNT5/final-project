#!/usr/bin/env python3
"""Route CVAT-exported YOLO labels into dataset/labels/{train,val} by the manifest split.

CVAT "Ultralytics YOLO 1.0" export gives a flat folder of <stem>.txt (mixed train/val,
because tasks are grouped by view). This copies each label to the correct split folder so
the detector sees a clean train != val split, then reports coverage + class distribution.

Usage (repo root):
  .venv/bin/python tools/route_cvat_labels.py --src /path/to/unzipped/cvat_export/labels
"""
import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(ROOT, 'annotation_guide', 'frame_manifest.csv')
LAB = os.path.join(ROOT, 'annotation_guide', 'dataset', 'labels')
CLASSES = ['Person_Authorized', 'Person_Unauthorized', 'Safe_Walkway', 'Restricted_Zone',
           'Panel_Closed', 'Panel_Opened', 'Forklift_Safe_Load', 'Forklift_Overload']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='folder of exported <stem>.txt label files')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(MAN)))
    split_of = {f"{r['clip'].replace('.mp4', '')}_{int(r['frame_idx']):02d}": r['split'] for r in rows}
    for sp in ('train', 'val'):
        os.makedirs(os.path.join(LAB, sp), exist_ok=True)

    routed, unknown = Counter(), []
    cls_counts = Counter()
    labeled_stems = set()
    META = {'train.txt', 'val.txt', 'test.txt'}   # CVAT image-list files, not labels
    txt_files = []                                  # recurse: handles labels/train/ + many jobs
    for dp, _, fns in os.walk(args.src):
        for fn in fns:
            if fn.endswith('.txt') and fn not in META:
                txt_files.append((fn[:-4], os.path.join(dp, fn)))
    for stem, path in txt_files:
        sp = split_of.get(stem)
        if sp is None:
            unknown.append(stem)
            continue
        shutil.copy(path, os.path.join(LAB, sp, stem + '.txt'))
        routed[sp] += 1
        labeled_stems.add(stem)
        with open(path) as f:
            for line in f:
                if line.strip():
                    cls_counts[int(line.split()[0])] += 1

    # coverage: which manifest frames still have no label file (pure-background frames are OK)
    missing = defaultdict(int)
    for stem, sp in split_of.items():
        if stem not in labeled_stems:
            missing[sp] += 1

    print(f"Routed labels: train {routed['train']}, val {routed['val']} "
          f"(of {len(split_of)} frames)")
    if unknown:
        print(f"  WARN {len(unknown)} label files not in manifest (skipped), e.g. {unknown[:3]}")
    print(f"  frames with NO label file (background or not-yet-labelled): "
          f"train {missing['train']}, val {missing['val']}")
    print("  box class distribution:")
    for c in range(8):
        print(f"    {c} {CLASSES[c]:20s}: {cls_counts.get(c, 0)}")
    print(f"\nLabels in {os.path.relpath(LAB, ROOT)}/(train|val). Ready for Phase 02 when coverage looks right.")


if __name__ == '__main__':
    main()
