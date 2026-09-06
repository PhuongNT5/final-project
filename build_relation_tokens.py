#!/usr/bin/env python3
"""Stage B input builder — fuse trained-detector detections with per-view zone geometry
into box-token sequences for the spatial-relation Transformer.

For each clip: sample T frames, run the entity detector, and for every detection emit a token
  feat[10] = [class_id, xc, yc, w, h, conf, iou_walkway, iou_restricted, foot_in_walkway, norm_dist]
Geometry (last 4) is computed for people (class 0/1) against the clip's per-view zone polygons;
zeros for non-person entities. Output: sweep_results/relation_tokens/tokens_{train,test}.pt

Runs LOCALLY — inference on MPS/CPU is fine; only TRAINING hits the MPS bug.
Usage:  .venv/bin/python build_relation_tokens.py [--limit N] [--device mps|cpu]
"""
import argparse
import glob
import json
import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from shapely.geometry import Polygon, Point, box as shp_box

ROOT = os.path.dirname(os.path.abspath(__file__))
DET = os.path.join(ROOT, 'sweep_results', 'entity_detector', 'detector_best.pt')
VIEW_MAP = os.path.join(ROOT, 'sweep_results', 'zone_prior', 'clip_view_map.json')
ZONES_DIR = os.path.join(ROOT, 'annotation_guide', 'zones')
OUT = os.path.join(ROOT, 'sweep_results', 'relation_tokens')
T, N_MAX, CONF, IMGSZ = 16, 8, 0.25, 800
UNSAFE = [0, 1, 2, 3]            # source behaviour-class folders (clip labels)
FEAT = 10
APP_DIM = 64                     # v2b: per-box appearance feature (context crop -> MobileNet -> random-projected)
_APP = {}


def get_appearance(device, backbone='mobilenet'):
    """Lazy-load a per-box appearance extractor + fixed random projection to APP_DIM.
    backbone='mobilenet' (ImageNet) or 'yolo' (the fine-tuned detector's own backbone, entity-conditioned)."""
    if _APP:
        return _APP
    if backbone == 'yolo':
        from ultralytics import YOLO
        det = YOLO(DET)
        layers = det.model.model
        net = torch.nn.Sequential(*[layers[i] for i in range(11)]).to(device).eval()  # backbone 0..10
        cdim = 512

        def preprocess(crop_bgr):
            im = cv2.resize(crop_bgr[:, :, ::-1], (128, 128)).astype('float32') / 255.0   # RGB, 0-1 (YOLO norm)
            return torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)
    else:
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        from PIL import Image
        w = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        m = mobilenet_v3_small(weights=w)
        net = torch.nn.Sequential(m.features, m.avgpool, torch.nn.Flatten()).to(device).eval()
        cdim = 576
        tf = w.transforms()

        def preprocess(crop_bgr):
            return tf(Image.fromarray(crop_bgr[:, :, ::-1]))
    rng = np.random.default_rng(0)
    proj = torch.tensor((rng.standard_normal((cdim, APP_DIM)) / np.sqrt(APP_DIM)).astype('float32'), device=device)
    _APP.update(feat=net, proj=proj, preprocess=preprocess, device=device, kind=backbone)
    return _APP


@torch.no_grad()
def embed_boxes(frame, boxes_xywhn, W, H, app):
    """Crop each box (30% context padding), embed via the chosen backbone, project to APP_DIM."""
    ims = []
    for cx, cy, w, h in boxes_xywhn:
        x1 = max(0, int((cx - w / 2 * 1.3) * W)); y1 = max(0, int((cy - h / 2 * 1.3) * H))
        x2 = min(W, int((cx + w / 2 * 1.3) * W)); y2 = min(H, int((cy + h / 2 * 1.3) * H))
        crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame
        ims.append(app['preprocess'](crop))
    if not ims:
        return np.zeros((0, APP_DIM), dtype=np.float32)
    x = torch.stack(ims).to(app['device'])
    f = app['feat'](x)
    if f.ndim == 4:                                          # yolo backbone → (K,C,h,w)
        f = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
    return (f @ app['proj']).cpu().numpy()


def load_zones():
    z = {}
    for f in glob.glob(os.path.join(ZONES_DIR, 'view_*.json')):
        d = json.load(open(f))
        mk = lambda ps: [Polygon(p).buffer(0) for p in ps if len(p) >= 3]
        z[str(d['view'])] = {'walkway': mk(d.get('safe_walkway', [])),
                             'restricted': mk(d.get('restricted_zone', []))}
    return z


def view_map():
    return {r['basename']: str(r['view']) for r in json.load(open(VIEW_MAP))}


def geometry(cx, cy, w, h, W, H, zones):
    """Relative geometry of a person box vs the view's zone polygons (pixel space)."""
    x1, y1, x2, y2 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
    pbox = shp_box(x1, y1, x2, y2)
    foot = Point((x1 + x2) / 2, y2)          # foot-point = decisive in/out-of-walkway signal
    def max_iou(polys):
        best = 0.0
        for pg in polys:
            uni = pbox.union(pg).area
            if uni > 0:
                best = max(best, pbox.intersection(pg).area / uni)
        return best
    iou_w = max_iou(zones['walkway']) if zones else 0.0
    iou_r = max_iou(zones['restricted']) if zones else 0.0
    foot_in = 1.0 if (zones and any(pg.contains(foot) for pg in zones['walkway'])) else 0.0
    nd = 0.0
    if zones and zones['restricted']:
        pc = Point((x1 + x2) / 2, (y1 + y2) / 2)
        nd = min(pc.distance(pg.centroid) for pg in zones['restricted']) / max(W, 1)
    return [iou_w, iou_r, foot_in, nd]


CORRUPT = None   # test-time corruption/aug applied to each decoded frame (set in main)


def corrupt_frame(fr, rng):
    """Apply a corruption/aug to a BGR frame (for TTA / robustness). Returns (frame, x_flipped?)."""
    if CORRUPT == 'flip':
        return fr[:, ::-1].copy(), True
    if CORRUPT == 'noise':
        return np.clip(fr.astype('float32') + rng.normal(0, 25, fr.shape), 0, 255).astype('uint8'), False
    if CORRUPT == 'blur':
        return cv2.GaussianBlur(fr, (9, 9), 0), False
    if CORRUPT == 'jpeg':
        ok, enc = cv2.imencode('.jpg', fr, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
        return (cv2.imdecode(enc, 1) if ok else fr), False
    if CORRUPT == 'dark':
        return np.clip(fr.astype('float32') * 0.4, 0, 255).astype('uint8'), False
    return fr, False


def process_clip(path, det, zones, device, app=None):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or T
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    idxs = [int(x) for x in np.linspace(0, max(0, total - 1), T)]
    rng = np.random.default_rng(0)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i, max(0, total - 2)))
        ok, fr = cap.read()
        if ok and CORRUPT:
            fr, _ = corrupt_frame(fr, rng)      # test-time corruption / flip
        frames.append(fr if ok else None)
    cap.release()

    fd = FEAT + (APP_DIM if app else 0)
    tokens = np.zeros((T, N_MAX, fd), dtype=np.float32)
    mask = np.zeros((T, N_MAX), dtype=np.float32)
    valid = [(t, fr) for t, fr in enumerate(frames) if fr is not None]
    if not valid:
        return tokens, mask
    res = det.predict([fr for _, fr in valid], imgsz=IMGSZ, conf=CONF, verbose=False, device=device)
    for (t, fr), r in zip(valid, res):
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xywhn = b.xywhn.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        conf = b.conf.cpu().numpy()
        order = np.argsort(-conf)[:N_MAX]
        app_feats = embed_boxes(fr, xywhn[order], W, H, app) if app else None
        for j, oi in enumerate(order):
            cx, cy, w, h = xywhn[oi]
            if CORRUPT == 'flip':
                cx = 1.0 - cx                    # un-flip coord for geometry vs original zones (crop stays mirrored)
            c = int(cls[oi])
            g = geometry(cx, cy, w, h, W, H, zones) if c in (0, 1) else [0.0, 0.0, 0.0, 0.0]
            row = [c, cx, cy, w, h, float(conf[oi])] + g
            if app:
                row = row + app_feats[j].tolist()
            tokens[t, j] = row
            mask[t, j] = 1.0
    return tokens, mask


def build_subset(subset, det, zones, vmap, device, limit=None, app=None, tag=''):
    clips = []
    for c in UNSAFE:
        clips += sorted(glob.glob(os.path.join(ROOT, 'data', subset, f'{c}_*', '*.mp4')))
    if limit:
        clips = clips[:limit]
    all_tok, all_mask, labels, cids, views = [], [], [], [], []
    for k, path in enumerate(clips):
        base = os.path.basename(path)
        label = int(base.split('_')[0])            # source folder class = behaviour label
        v = vmap.get(base)
        tok, msk = process_clip(path, det, zones.get(v), device, app=app)
        all_tok.append(tok); all_mask.append(msk)
        labels.append(label); cids.append(base); views.append(v)
        if (k + 1) % 50 == 0:
            print(f"  [{subset}] {k+1}/{len(clips)}", flush=True)
    layout = ['class', 'xc', 'yc', 'w', 'h', 'conf', 'iou_walkway', 'iou_restricted',
              'foot_in_walkway', 'norm_dist'] + ([f'app_{i}' for i in range(APP_DIM)] if app else [])
    out = {'tokens': torch.tensor(np.stack(all_tok)), 'mask': torch.tensor(np.stack(all_mask)),
           'labels': torch.tensor(labels), 'clip_ids': cids, 'views': views, 'feat_layout': layout}
    os.makedirs(OUT, exist_ok=True)
    torch.save(out, os.path.join(OUT, f'tokens_{tag}{subset}.pt'))
    print(f"[{subset}] {len(clips)} clips → tokens_{tag}{subset}.pt | feat={out['tokens'].shape[-1]} "
          f"| total detections={int(np.stack(all_mask).sum())}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='cap clips per subset (smoke test)')
    ap.add_argument('--device', default=None, help="mps/cpu/0 (inference — MPS is fine)")
    ap.add_argument('--subsets', default='train,test')
    ap.add_argument('--appearance', action='store_true', help='v2b: add per-box appearance feature')
    ap.add_argument('--appearance_backbone', default='mobilenet', choices=['mobilenet', 'yolo'],
                    help="appearance backbone: mobilenet (ImageNet) or yolo (fine-tuned detector, entity-conditioned)")
    ap.add_argument('--corrupt', default=None, choices=['flip', 'noise', 'blur', 'jpeg', 'dark'],
                    help='test-time corruption/aug applied to frames (F/G robustness)')
    ap.add_argument('--detector', default=None, help='override detector weights (MT3: YOLO26 swap)')
    ap.add_argument('--out_tag', default=None, help='override output tag prefix (avoid clobbering baseline tokens)')
    args = ap.parse_args()
    global CORRUPT, DET
    CORRUPT = args.corrupt
    if args.detector:
        DET = os.path.abspath(args.detector)   # swap detector (+ its ROI backbone) for YOLO26 comparison
    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"detector={os.path.relpath(DET, ROOT)} device={device} appearance={args.appearance} "
          f"backbone={args.appearance_backbone} corrupt={args.corrupt}")
    det = YOLO(DET)
    zones, vmap = load_zones(), view_map()
    app = get_appearance(device, args.appearance_backbone) if args.appearance else None
    tag = args.out_tag if args.out_tag else \
        (('yoloapp_' if args.appearance_backbone == 'yolo' else 'app_') if args.appearance else '')
    if args.corrupt:
        tag = f'{tag}{args.corrupt}_'
    print(f"zones for views: {sorted(zones)} | view-map entries: {len(vmap)}")
    for sp in args.subsets.split(','):
        build_subset(sp, det, zones, vmap, device, args.limit, app=app, tag=tag)


if __name__ == '__main__':
    main()
