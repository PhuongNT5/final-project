#!/usr/bin/env python3
"""Group the 2,262 annotation frames into per-viewpoint ZIPs for CVAT task creation.

Each CVAT task = one camera view, so the fixed-camera zones (Safe_Walkway,
Restricted_Zone) can be drawn ONCE per task and interpolated. Each zip mixes
train + val frames on purpose — you annotate everything once; tools/route_cvat_labels.py
sorts the exported labels back into train/val afterwards.

Reads:  annotation_guide/frame_manifest.csv  +  annotation_guide/dataset/images/{train,val}/
Writes: annotation_guide/cvat_tasks/view_<id>.zip   (real JPEG content, flat)

Usage (repo root):  .venv/bin/python tools/organize_frames_by_view.py
"""
import csv
import os
import zipfile
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(ROOT, 'annotation_guide', 'frame_manifest.csv')
IMG = os.path.join(ROOT, 'annotation_guide', 'dataset', 'images')
OUT = os.path.join(ROOT, 'annotation_guide', 'cvat_tasks')


def stem_of(r):
    return f"{r['clip'].replace('.mp4', '')}_{int(r['frame_idx']):02d}.jpg"


def main():
    rows = list(csv.DictReader(open(MAN)))
    by_view = defaultdict(list)
    for r in rows:
        by_view[str(r['view'])].append(r)

    os.makedirs(OUT, exist_ok=True)
    print(f"{len(rows)} frames → {len(by_view)} viewpoint tasks\n")
    grand = 0
    for v in sorted(by_view, key=lambda k: (k == 'na', k)):
        items = by_view[v]
        zp = os.path.join(OUT, f"view_{v}.zip")
        written, missing = 0, 0
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_STORED) as z:
            for r in items:
                stem = stem_of(r)
                src = os.path.join(IMG, r['split'], stem)
                if os.path.exists(src):
                    z.write(src, arcname=stem)   # real content, flat archive
                    written += 1
                else:
                    missing += 1
        splits = Counter(r['split'] for r in items)
        grand += written
        print(f"  view_{v:<3s}: {written:4d} frames  (train {splits['train']}, val {splits['val']})"
              f"{'  MISSING ' + str(missing) if missing else ''}  → {os.path.relpath(zp, ROOT)}")
    print(f"\nTotal zipped: {grand}. Upload each view_<id>.zip as a separate CVAT task (SOP step 3).")


if __name__ == '__main__':
    main()
