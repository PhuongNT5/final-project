#!/usr/bin/env python3
"""jetson-p3-demo-video.py — render the P3 PERCEPTION front-end on a real clip, on-device.

Scope, stated plainly: this draws what the TensorRT detector actually returns (confidence-
filtered, top-N_MAX selected, exactly as `benchmark-p3-utils.postprocess_frame` selects for the
token builder) plus the view's zone polygons. It does **not** render a Safe/Unsafe decision —
that needs the 74-d token assembly (6 box fields + 4 geometry + 64 appearance) whose parity with
the training-time pipeline has not been verified on-device. Drawing an unverified label would be
worse than drawing none.

It is also the validation step for that later work: if the boxes land on people, the engine's
coordinate convention and the letterbox mapping here are right.

Usage:
    python3 deployment/scripts/jetson-p3-demo-video.py \
        --engine-dir deployment/export/jetson-p3 --clip data/test/0_safe_walkway_violation/0_te10.mp4 \
        --out deployment/artifacts/demo-p3-perception.mp4
"""
import argparse
import json
import os
import sys
from importlib.util import module_from_spec, spec_from_file_location

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))


def _load(name, fname):
    spec = spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LAT = _load('jetson_measure_latency', 'jetson-measure-latency.py')
TRU = _load('trt_runner_utils', 'trt-runner-utils.py')

IMGSZ = 800
PAD = 114                      # ultralytics letterbox fill
BOX_COLOR = (60, 220, 60)
ZONE_COLORS = {'walkway': (255, 190, 60), 'restricted': (60, 60, 255)}


def letterbox(frame, size):
    """Scale-preserving resize + centre pad, the convention the detector was trained under.
    Returns the padded image and the (scale, dx, dy) needed to map boxes back."""
    h, w = frame.shape[:2]
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    dx, dy = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), PAD, dtype=np.uint8)
    out[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, s, dx, dy


def preprocess(frame, size, letterboxed=True):
    img, s, dx, dy = letterbox(frame, size) if letterboxed else (
        cv2.resize(frame, (size, size)), None, 0, 0)
    x = img[:, :, ::-1].astype(np.float32) / 255.0          # BGR->RGB, 0-1
    x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]     # HWC -> NCHW
    return x, s, dx, dy


def unletterbox(box, s, dx, dy, w, h):
    x1, y1, x2, y2 = box
    if s is None:                                            # plain resize path
        sx, sy = w / IMGSZ, h / IMGSZ
        return x1 * sx, y1 * sy, x2 * sx, y2 * sy
    return ((x1 - dx) / s, (y1 - dy) / s, (x2 - dx) / s, (y2 - dy) / s)


def load_zones_for_clip(clip_path, view_arg):
    """Zone polygons for this clip's camera view. Without a view the demo still runs — it just
    draws no zones, and says so on the overlay rather than drawing the wrong ones."""
    zones_dir = os.path.join(ROOT, 'annotation_guide', 'zones')
    view = view_arg
    if view is None:
        # clip_view_map_691.json is the map the bin8 tokens were actually built with (all 691
        # clips). The older clip_view_map.json covers only the 516 unsafe clips — using it here
        # would silently draw no zones on every Safe clip.
        zp = os.path.join(ROOT, 'sweep_results', 'zone_prior')
        for cand in ('clip_view_map_691.json', 'clip_view_map.json'):
            vmap = os.path.join(zp, cand)
            if os.path.exists(vmap):
                base = os.path.basename(clip_path)
                view = {r['basename']: str(r['view']) for r in json.load(open(vmap))}.get(base)
                break
    if view is None:
        return None, None
    path = os.path.join(zones_dir, f'view_{view}.json')
    if not os.path.exists(path):
        return None, view
    d = json.load(open(path))
    return {'walkway': d.get('safe_walkway', []), 'restricted': d.get('restricted_zone', [])}, view


def draw_zones(frame, zones):
    overlay = frame.copy()
    for kind, polys in zones.items():
        for p in polys:
            if len(p) < 3:
                continue
            pts = np.array(p, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], ZONE_COLORS[kind])
            cv2.polylines(frame, [pts], True, ZONE_COLORS[kind], 2)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)


def draw_hud(frame, lines):
    y = 34
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (16, y + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
        cv2.putText(frame, t, (16, y + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def select_boxes(raw, conf_thr, n_max):
    """Same selection the token builder applies: confidence filter then top-N_MAX by score."""
    raw = np.asarray(raw).reshape(-1, raw.shape[-1])
    conf = raw[:, 4]
    keep = np.where(conf >= conf_thr)[0]
    if len(keep) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return raw[keep[np.argsort(-conf[keep])][:n_max]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--engine-dir', default=os.path.join(ROOT, 'deployment', 'export', 'jetson'))
    ap.add_argument('--mode', default='fp16', choices=['fp32', 'fp16', 'int8'])
    ap.add_argument('--clip', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-frames', type=int, default=0, help='0 = whole clip')
    ap.add_argument('--conf', type=float, default=None, help='default: build_relation_tokens.CONF')
    ap.add_argument('--n-max', type=int, default=None, help='default: build_relation_tokens.N_MAX')
    ap.add_argument('--view', default=None, help='camera view id if clip_view_map.json is absent')
    ap.add_argument('--no-letterbox', action='store_true', help='plain square resize instead')
    args = ap.parse_args()

    conf_thr, n_max = args.conf, args.n_max
    if conf_thr is None or n_max is None:
        sys.path.insert(0, ROOT)
        import build_relation_tokens as BRT
        conf_thr = BRT.CONF if conf_thr is None else conf_thr
        n_max = BRT.N_MAX if n_max is None else n_max

    trt, cuda = LAT._import_trt()
    if trt is None:
        sys.exit(f'tensorrt/pycuda unavailable: {cuda}')
    stages = LAT.discover_engines(args.engine_dir)
    if 'p3-detector' not in stages or args.mode not in stages['p3-detector']:
        sys.exit(f'no p3-detector {args.mode} engine in {args.engine_dir}')
    det = TRU.TRTRunner(trt, cuda, stages['p3-detector'][args.mode])
    in_name = det.input_order[0]
    size = det.inputs[in_name]['shape'][-1]

    zones, view = load_zones_for_clip(args.clip, args.view)
    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        sys.exit(f'cannot open {args.clip}')
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    n, total_boxes = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        x, s, dx, dy = preprocess(frame, size, not args.no_letterbox)
        raw = det.run(**{in_name: x.astype(det.inputs[in_name]['host'].dtype)})
        if isinstance(raw, dict):
            raw = next(iter(raw.values()))
        boxes = select_boxes(np.asarray(raw), conf_thr, n_max)
        if zones:
            draw_zones(frame, zones)
        for b in boxes:
            x1, y1, x2, y2 = unletterbox(b[:4], s, dx, dy, W, H)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BOX_COLOR, 2)
            cv2.putText(frame, f'{b[4]:.2f}', (int(x1), max(14, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, BOX_COLOR, 2)
        total_boxes += len(boxes)
        draw_hud(frame, [
            f'P3 perception front-end - TensorRT {args.mode} - Jetson Orin Nano Super',
            f'detector {size}px | conf>={conf_thr} | top-{n_max} | entities: {len(boxes)}',
            (f'zones: view {view}' if zones else 'zones: none (view unknown - none drawn)'),
            'decision head NOT shown - see report section 5e',
        ])
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()
    print(f'[SAVED] {args.out}  frames={n}  mean_entities={total_boxes / max(1, n):.2f}  view={view}')


if __name__ == '__main__':
    main()
