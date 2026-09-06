#!/usr/bin/env python3
"""Zone-arm data prep — generate detectable zone-box labels (classes 2,3) from the static polygons.

Zones (Safe_Walkway=2, Restricted_Zone=3) were never annotated as boxes (0 instances); they live as
per-view polygons in `zones/view_*.json` and were injected as inference-time constants. To let the
detector EMIT zone boxes, this converts each view's polygons to axis-aligned YOLO boxes and appends them
to the label file of every image belonging to that view.

Image -> view resolution uses clip_view_map.json (image basename `{cls}_tr{N}_{frame}` -> clip `{cls}_tr{N}`
-> view). Images whose view cannot be resolved, or whose view has no zone file (views 2,7), are EXCLUDED
from the zone dataset so no image is a false-negative for zones. Builds a NEW dataset dir (images symlinked,
labels rewritten) — non-destructive; the movable-entity dataset is untouched.

Caveat: an axis-aligned box around a diagonal walkway polygon is loose (may span most of the frame). This
is inherent to YOLO boxes; the downstream relation head still gets a real, per-view zone token.

Usage (Kaggle): python build-zone-labels.py \
    --src_root /kaggle/input/<ds>/dataset --zones_dir /kaggle/input/<ds>/zones \
    --view_map /kaggle/input/<ds>/clip_view_map.json --dst_root /kaggle/working/dataset_zones
"""
import argparse, json, os, re, glob

CLS = {'safe_walkway': 2, 'restricted_zone': 3}


def poly_to_yolo(poly, W, H, cls):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    xc = (xmin + xmax) / 2 / W; yc = (ymin + ymax) / 2 / H
    w = (xmax - xmin) / W; h = (ymax - ymin) / H
    xc, yc = min(max(xc, 0), 1), min(max(yc, 0), 1)
    w, h = min(w, 1), min(h, 1)
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def load_zone_lines(zones_dir):
    """view (str) -> list of YOLO label lines for its zone boxes."""
    out = {}
    for f in glob.glob(os.path.join(zones_dir, 'view_*.json')):
        z = json.load(open(f)); W, H = z['image_size']; lines = []
        for key, cls in CLS.items():
            for poly in z.get(key, []):
                if len(poly) >= 3:
                    lines.append(poly_to_yolo(poly, W, H, cls))
        out[str(z['view'])] = lines
    return out


def clip_to_view(view_map):
    m = json.load(open(view_map))
    return {os.path.splitext(e['basename'])[0]: str(e['view']) for e in m}


def img_clip(name):
    s = os.path.splitext(os.path.basename(name))[0]
    m = re.match(r'^(\d+_tr\d+)', s)
    return m.group(1) if m else None


def build(src_root, zones_dir, view_map, dst_root):
    zone_lines = load_zone_lines(zones_dir)
    c2v = clip_to_view(view_map)
    have = set(zone_lines)                       # views with a zone file
    stats = {'total': 0, 'no_view': 0, 'no_zonefile': 0, 'written': 0, 'by_view': {}}
    for split in ('train', 'val'):
        img_dir = os.path.join(src_root, 'images', split)
        lbl_dir = os.path.join(src_root, 'labels', split)
        di = os.path.join(dst_root, 'images', split); dl = os.path.join(dst_root, 'labels', split)
        os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
        for ip in sorted(glob.glob(os.path.join(img_dir, '*'))):
            stats['total'] += 1
            clip = img_clip(ip); view = c2v.get(clip)
            if view is None:
                stats['no_view'] += 1; continue
            if view not in have:
                stats['no_zonefile'] += 1; continue
            base = os.path.splitext(os.path.basename(ip))[0]
            # original entity labels (may be empty) + appended zone lines
            orig = os.path.join(lbl_dir, base + '.txt')
            body = open(orig).read().rstrip('\n').splitlines() if os.path.exists(orig) else []
            body += zone_lines[view]
            open(os.path.join(dl, base + '.txt'), 'w').write('\n'.join(body) + '\n')
            link = os.path.join(di, os.path.basename(ip))
            if not os.path.exists(link):
                try:
                    os.symlink(os.path.abspath(ip), link)
                except OSError:
                    import shutil; shutil.copy2(ip, link)
            stats['written'] += 1
            stats['by_view'][view] = stats['by_view'].get(view, 0) + 1
    # data yaml
    names = ['Person_Authorized', 'Person_Unauthorized', 'Safe_Walkway', 'Restricted_Zone',
             'Panel_Closed', 'Panel_Opened', 'Forklift_Safe_Load', 'Forklift_Overload']
    yaml = (f"path: {os.path.abspath(dst_root)}\ntrain: images/train\nval: images/val\nnc: 8\nnames:\n"
            + ''.join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    open(os.path.join(dst_root, 'data.yaml'), 'w').write(yaml)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_root', default='annotation_guide/dataset')
    ap.add_argument('--zones_dir', default='annotation_guide/zones')
    ap.add_argument('--view_map', default='sweep_results/zone_prior/clip_view_map.json')
    ap.add_argument('--dst_root', default='annotation_guide/dataset_zones')
    args = ap.parse_args()
    s = build(args.src_root, args.zones_dir, args.view_map, args.dst_root)
    print(f"[zone-labels] total images scanned : {s['total']}")
    print(f"  excluded (view unresolved)       : {s['no_view']}")
    print(f"  excluded (view has no zone file) : {s['no_zonefile']}")
    print(f"  written with zone boxes          : {s['written']}")
    print(f"  by view: {dict(sorted(s['by_view'].items()))}")
    print(f"[done] zone dataset -> {os.path.abspath(args.dst_root)}/data.yaml")


if __name__ == '__main__':
    main()
