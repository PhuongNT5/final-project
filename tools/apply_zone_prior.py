#!/usr/bin/env python3
"""
GZT stage 1-2 — per-clip GEOMETRIC ZONE features from person foot-points vs the camera's
WALKWAY polygon and (optionally) a MACHINE / DANGER-zone polygon.

The go/no-go probe showed "off-walkway" (z1) cannot separate WalkwayViolation from
UnauthorizedIntervention (both leave the painted walkway); the discriminative signal is
PROXIMITY TO MACHINES (Unauth workers stand deeper in the equipment area). So we also compute
features against a hand-drawn danger zone.

Per clip, for T uniform frames, detect persons (COCO yolo11s.pt -> unbiased foot-points; or
--detector best), take each foot-point (bbox bottom-centre) and emit up to 5 features:
  WALKWAY (needs 'walkway' polygon):
    z1 = frac of detected frames with >=1 person OUTSIDE walkway
    z2 = mean normalised signed dist of the most-OUTSIDE person (neg = outside)
    z3 = walkway boundary crossings (sign changes)
  DANGER  (needs 'danger' polygon, drawn via tools/draw_zone.py):
    z4 = frac of detected frames with >=1 person INSIDE the machine/danger zone
    z5 = mean normalised signed dist of the most-INSIDE person (pos = inside danger)

Polygons are merged from (later overrides earlier, per view per zone-type):
  zones_auto.json  (auto green walkway)  ->  zones.json  ->  zones_manual.json (your danger draw)

Outputs (sweep_results/zone_prior/): zone_features_{train,test}.pt + zone_features.csv

Usage:
  .venv/bin/python3 tools/apply_zone_prior.py --classes 0 1 --frames 16
"""
import os, sys, json, argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_viewpoint_frames import clip_paths  # DRY: same clip ordering as feature extraction

FEAT = 'sweep_results/pairwise/features'
ZONE_DIR = 'annotation_guide/zones'
OUT_DIR = 'sweep_results/zone_prior'
CLASS_NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']
FEATURE_NAMES = ['z1_offwalk_frac', 'z2_walk_signed', 'z3_crossings', 'z4_in_danger_frac', 'z5_danger_signed']
COCO = 'yolo11s.pt'
FT = 'runs/detect/sweep_results/finetune_bbox/yolo_finetuned/weights/best.pt'


def build_clip_view_map():
    """Replay extract_viewpoint_frames KMeans -> [{path,basename,label,view,subset}] (516 clips)."""
    cache = os.path.join(OUT_DIR, 'clip_view_map.json')
    if os.path.exists(cache):
        return json.load(open(cache))
    from sklearn.cluster import KMeans
    tr = torch.load(os.path.join(FEAT, 'all8_train.pt'), map_location='cpu')
    te = torch.load(os.path.join(FEAT, 'all8_test.pt'), map_location='cpu')
    feats = torch.cat([tr['features'], te['features']], 0)
    labels = torch.cat([tr['labels'], te['labels']], 0)
    paths, _ = clip_paths()
    assert len(paths) == len(labels), f"path/feature mismatch {len(paths)} vs {len(labels)}"
    m4 = (labels < 4).numpy()
    rep = feats.mean(1).reshape(len(labels), -1).numpy()[m4]
    y4 = labels.numpy()[m4]
    paths4 = [p for p, k in zip(paths, m4) if k]
    views = KMeans(8, random_state=0, n_init=10).fit(rep).labels_
    rec = [{'path': p, 'basename': os.path.basename(p), 'label': int(y), 'view': int(v),
            'subset': ('train' if f'{os.sep}train{os.sep}' in p else 'test')}
           for p, y, v in zip(paths4, y4, views)]
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(rec, open(cache, 'w'), indent=2)
    print(f"[map] built clip->view for {len(rec)} clips -> {cache}")
    return rec


def load_zones():
    """Merge walkway/danger polygons per view; later files override per zone-type."""
    merged = {}
    for fn in ['zones_auto.json', 'zones.json', 'zones_manual.json']:
        p = os.path.join(ZONE_DIR, fn)
        if not os.path.exists(p):
            continue
        for vid, d in json.load(open(p)).items():
            merged.setdefault(str(vid), {}).update({k: v for k, v in d.items() if v})
    return merged


def sample_frames(path, T):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n <= 0:
        cap.release(); return []
    out = []
    for i in np.linspace(0, n - 1, min(T, n)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            out.append(fr)
    cap.release()
    return out


def detect_feet(frames, model, conf, device):
    """Per frame -> list of (x,y) foot-points (bbox bottom-centre) for detected persons."""
    res = model.predict(frames, conf=conf, classes=[0], verbose=False, device=device)
    feet = []
    for r in res:
        fs = [(float((b[0] + b[2]) / 2), float(b[3])) for b in r.boxes.xyxy.cpu().numpy()]
        feet.append(fs)
    return feet


def walkway_feats(feet, poly, H):
    P = np.array(poly, dtype=np.int32)
    seq = [min(cv2.pointPolygonTest(P, (x, y), True) for x, y in fs) for fs in feet if fs]
    if not seq:
        return float('nan'), float('nan'), float('nan')
    d = np.array(seq)
    s = np.sign(d); s[s == 0] = 1
    return float((d < 0).mean()), float(np.clip(d / H, -0.3, 0.3).mean()), int((np.diff(s) != 0).sum())


def danger_feats(feet, poly, H):
    P = np.array(poly, dtype=np.int32)
    seq = [max(cv2.pointPolygonTest(P, (x, y), True) for x, y in fs) for fs in feet if fs]
    if not seq:
        return float('nan'), float('nan')
    d = np.array(seq)
    return float((d >= 0).mean()), float(np.clip(d / H, -0.3, 0.3).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', type=int, nargs='*', default=None)
    ap.add_argument('--classes', type=int, nargs='*', default=[0, 1, 2, 3])
    ap.add_argument('--frames', type=int, default=16)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--detector', choices=['coco', 'best'], default='coco')
    ap.add_argument('--device', default=None)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO
    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = YOLO(COCO if args.detector == 'coco' else FT)

    zones = load_zones()
    have_walk = {v for v, d in zones.items() if d.get('walkway')}
    have_danger = {v for v, d in zones.items() if d.get('danger')}
    print(f"[zones] walkway views={sorted(have_walk)}  danger views={sorted(have_danger) or '(none yet)'}  "
          f"detector={args.detector}({device})")

    cmap = build_clip_view_map()
    todo = [c for c in cmap if str(c['view']) in zones and c['label'] in args.classes
            and (args.views is None or c['view'] in args.views)]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[run] {len(todo)} clips across views {sorted({c['view'] for c in todo})}\n")

    rows, per_subset = [], {'train': [], 'test': []}
    for i, c in enumerate(todo):
        zd = zones[str(c['view'])]
        frames = sample_frames(c['path'], args.frames)
        H = frames[0].shape[0] if frames else 1080
        feet = detect_feet(frames, model, args.conf, device) if frames else []
        cov = (sum(1 for fs in feet if fs) / len(feet)) if feet else 0.0
        z1, z2, z3 = walkway_feats(feet, zd['walkway'], H) if zd.get('walkway') and feet else (np.nan,) * 3
        z4, z5 = danger_feats(feet, zd['danger'], H) if zd.get('danger') and feet else (np.nan, np.nan)
        vec = [z1, z2, z3, z4, z5]
        rows.append((c['subset'], c['basename'], c['label'], CLASS_NAMES[c['label']], c['view'], cov, *vec))
        per_subset[c['subset']].append((c, vec, cov))
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            dtxt = f"z4={z4:.2f}" if zd.get('danger') else "z4=--"
            print(f"  {i+1}/{len(todo)}  {c['basename']:>12} v{c['view']} cov={cov:.2f} "
                  f"z1={z1:.2f} z2={z2:+.3f} {dtxt}")

    os.makedirs(OUT_DIR, exist_ok=True)
    for subset, items in per_subset.items():
        if not items:
            continue
        torch.save({
            'features': torch.tensor([vec for _, vec, _ in items], dtype=torch.float),
            'labels': torch.tensor([c['label'] for c, *_ in items], dtype=torch.long),
            'views': torch.tensor([c['view'] for c, *_ in items], dtype=torch.long),
            'coverage': torch.tensor([cov for *_, cov in items], dtype=torch.float),
            'clips': [c['basename'] for c, *_ in items],
            'feature_names': FEATURE_NAMES, 'detector': args.detector,
        }, os.path.join(OUT_DIR, f'zone_features_{subset}.pt'))
    with open(os.path.join(OUT_DIR, 'zone_features.csv'), 'w') as f:
        f.write('subset,clip,label,classname,view,coverage,' + ','.join(FEATURE_NAMES) + '\n')
        for r in rows:
            f.write('{},{},{},{},{},{:.3f},{:.4f},{:.4f},{},{:.4f},{:.4f}\n'.format(*r))
    cov_all = np.array([r[5] for r in rows])
    print(f"\n[SAVED] {OUT_DIR}/zone_features_*.pt + zone_features.csv  ({len(rows)} clips)")
    print(f"[detect-coverage] mean person coverage = {cov_all.mean():.2%} (0-detection clips: {(cov_all == 0).sum()})")
    if not have_danger:
        print("[next] draw machine/danger zones:  .venv/bin/python3 tools/draw_zone.py --views 1 3 5 --type danger")


if __name__ == '__main__':
    main()
