#!/usr/bin/env python3
"""Extract per-view zone polygons (Safe_Walkway, Restricted_Zone) from CVAT COCO exports.

Detection export drops polygons, so zones come from the COCO 1.0 export. Zones are fixed per
camera view (drawn once + propagated), so we store one canonical polygon set per view →
annotation_guide/zones/view_<v>.json. These feed the foot-in-polygon / IoU geometry features
(Phase 02/03) and can be injected as constants at inference.

Input:  annotation_guide/annotated/_extracted/coco/**/annotations/instances_default.json
        + annotation_guide/frame_manifest.csv  (frame stem → view)
Output: annotation_guide/zones/view_<v>.json

Usage:  .venv/bin/python tools/extract_zones_from_coco.py
"""
import csv
import glob
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO_GLOB = os.path.join(ROOT, 'annotation_guide', 'annotated', '_extracted', 'coco',
                         '*', 'annotations', 'instances_default.json')
MAN = os.path.join(ROOT, 'annotation_guide', 'frame_manifest.csv')
OUT = os.path.join(ROOT, 'annotation_guide', 'zones')
# COCO category id → our zone key (COCO is 1-indexed = YOLO index + 1)
ZONE_CATS = {3: 'safe_walkway', 4: 'restricted_zone'}


def poly_flat_to_pts(seg):
    """COCO segmentation [x1,y1,x2,y2,...] → [[x,y],...] (rounded to int px)."""
    return [[round(seg[i]), round(seg[i + 1])] for i in range(0, len(seg) - 1, 2)]


def main():
    view_of = {}
    for r in csv.DictReader(open(MAN)):
        stem = f"{r['clip'].replace('.mp4', '')}_{int(r['frame_idx']):02d}"
        view_of[stem] = str(r['view'])

    # per view → per frame stem → {zone_key: [polygons]}
    per_view = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    size_of = {}
    jsons = sorted(glob.glob(COCO_GLOB))
    for jp in jsons:
        d = json.load(open(jp))
        img = {im['id']: im for im in d['images']}
        for a in d['annotations']:
            key = ZONE_CATS.get(a['category_id'])
            if not key or not a.get('segmentation'):
                continue
            im = img[a['image_id']]
            stem = os.path.splitext(im['file_name'])[0]
            v = view_of.get(stem)
            if v is None:
                continue
            size_of[v] = [im['width'], im['height']]
            for seg in a['segmentation']:
                if isinstance(seg, list) and len(seg) >= 6:
                    per_view[v][stem][key].append(poly_flat_to_pts(seg))

    os.makedirs(OUT, exist_ok=True)
    print(f"Parsed {len(jsons)} COCO files → zones for {len(per_view)} views\n")
    for v in sorted(per_view, key=lambda k: (k == 'na', k)):
        frames = per_view[v]
        # representative frame = the one with the most zone polygons (captures all zones consistently)
        rep = max(frames, key=lambda s: sum(len(p) for p in frames[s].values()))
        walkway = frames[rep].get('safe_walkway', [])
        restricted = frames[rep].get('restricted_zone', [])
        out = {
            'view': v, 'image_size': size_of.get(v, [1920, 1080]),
            'source_frame': rep, 'n_frames_in_view': len(frames),
            'safe_walkway': walkway, 'restricted_zone': restricted,
        }
        json.dump(out, open(os.path.join(OUT, f'view_{v}.json'), 'w'), indent=1)
        print(f"  view_{v:<3s}: walkway={len(walkway)} poly, restricted={len(restricted)} poly "
              f"(from {rep}, {len(frames)} frames) → zones/view_{v}.json")

    covered = set(per_view)
    all_views = set(view_of.values())
    missing = sorted(all_views - covered)
    if missing:
        print(f"\n  NOTE: views with NO zone export: {missing} "
              f"(their COCO job wasn't exported, or no zones drawn)")


if __name__ == '__main__':
    main()
