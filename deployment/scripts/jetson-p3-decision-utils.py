"""jetson-p3-decision-utils.py — the P3 per-frame decision front-end on TensorRT: detector ->
select + zone geometry -> context crops -> appearance -> one (N_MAX, 74) token block.

Split out of jetson-p3-multi-stream.py to keep both files inside the 200-line guideline, and
because the assembly is not harness-specific: jetson-p3-demo-video.py renders the perception
front-end only (it states that the 74-d assembly is unverified on-device and draws no label), and
whatever closes that gap needs exactly this code rather than a second copy of it.

No tensorrt/pycuda import here — runners are passed in, so this module loads on Mac or Jetson.
"""
import os
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LAT = _load('jetson_measure_latency', 'jetson-measure-latency.py')   # discover_engines
TRU = _load('trt_runner_utils', 'trt-runner-utils.py')
# letterbox/unletterbox/select_boxes and the clip->view lookup live in the demo script; reusing
# them keeps ONE copy of the coordinate convention (a second copy could drift and silently
# mis-place every box). extract_crops is the already-benchmarked crop path.
DEMO = _load('jetson_p3_demo_video', 'jetson-p3-demo-video.py')
B = _load('benchmark_p3_utils', 'benchmark-p3-utils.py')
BRT = B.BRT
_ZONES_BY_VIEW = None

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def zones_for_clip(clip_path, view=None):
    """(zones, view) for a clip, where zones are SHAPELY polygons.

    jetson-p3-demo-video.load_zones_for_clip returns raw coordinate lists — it only draws them with
    cv2.fillPoly — and handing those to geometry() raises AttributeError on .area. So its clip->view
    lookup is reused (one copy of the 691-clip map logic) while the polygons come from
    BRT.load_zones(), i.e. the same objects the token builder used. Views 2 and 7 have no polygon
    file: .get returns None and geometry() then yields zeros, exactly as in the token build.

    Returning the one shared dict per view also means geometry()'s id()-keyed per-view precompute
    is built once, not once per stream.
    """
    global _ZONES_BY_VIEW
    view_id = DEMO.load_zones_for_clip(clip_path, view)[1]
    if view_id is None:
        return None, None
    if _ZONES_BY_VIEW is None:
        _ZONES_BY_VIEW = BRT.load_zones()
    return _ZONES_BY_VIEW.get(str(view_id)), view_id


def feed(runner, arr_by_name):
    """Casts each array to the dtype the engine's own host buffer was allocated with."""
    return {k: v.astype(runner.inputs[k]['host'].dtype) for k, v in arr_by_name.items()}


def _out(x):
    return np.asarray(next(iter(x.values())) if isinstance(x, dict) else x)


def preprocess(frame, size):
    """Returns the detector input AND the original frame: P3 crops come from full-resolution
    pixels, so the letterboxed 640/800 px tensor cannot be the only thing passed downstream."""
    x, s, dx, dy = DEMO.preprocess(frame, size, True)
    return x, s, dx, dy, frame


def appearance_embed(apps, crops):
    """ImageNet-normalised (K,3,224,224) through the bucketed appearance engine. Plain resize to
    224 (what extract_crops does), not the training transform's resize-256 + centre-crop — a
    pixel-level difference that moves the embedding values, not their cost.

    Padding up to the next bucket lets one runner per bucket be built once instead of one per
    distinct crop count; it adds work, never removes it, so the timing stays pessimistic."""
    k = len(crops)
    bs = min(b for b in sorted(apps) if b >= k)
    batch = crops + [crops[-1]] * (bs - k)
    arr = np.stack([c[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0 for c in batch])
    name = apps[bs].input_order[0]
    return _out(apps[bs].run(**feed(apps[bs], {name: (arr - _MEAN) / _STD})))[:k]


def frame_tokens(det, apps, zones, packed, conf_thr, n_max, crop_px):
    """One frame -> (N_MAX, FEAT+APP_DIM) token rows + mask row. Layout mirrors
    build_relation_tokens.process_clip: [class, xc, yc, w, h, conf, iou_walkway, iou_restricted,
    foot_in_walkway, norm_dist] then APP_DIM appearance.

    Appearance runs per FRAME (<= N_MAX crops) rather than once per decision over all T*N_MAX
    crops: that is what a streaming pipeline does, and many small batches cost more per crop than
    one large one — again the pessimistic direction.
    """
    x, s, dx, dy, frame = packed
    H, W = frame.shape[:2]
    sel = DEMO.select_boxes(_out(det.run(**feed(det, {det.input_order[0]: x}))), conf_thr, n_max)
    rows = np.zeros((n_max, BRT.FEAT + BRT.APP_DIM), dtype=np.float32)
    mask = np.zeros((n_max,), dtype=np.float32)
    if len(sel) == 0:
        return rows, mask
    boxes, meta = [], []
    for b in sel:
        x1, y1, x2, y2 = DEMO.unletterbox(b[:4], s, dx, dy, W, H)
        cx, cy = (x1 + x2) / (2 * W), (y1 + y2) / (2 * H)
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        geo = BRT.geometry(cx, cy, bw, bh, W, H, zones) if int(b[5]) in (0, 1) else [0.0] * 4
        boxes.append({'cx': cx, 'cy': cy, 'w': bw, 'h': bh})
        meta.append([float(b[5]), cx, cy, bw, bh, float(b[4])] + list(geo))
    app = appearance_embed(apps, B.extract_crops(frame, boxes, crop_px))
    for i, (m_, a_) in enumerate(zip(meta, app)):
        rows[i, :BRT.FEAT] = m_
        rows[i, BRT.FEAT:] = a_
        mask[i] = 1.0
    return rows, mask


def build_runners(trt, cuda, engine_dir, mode, detector_stage, head_stage, n_max):
    """Engines + the cfg the harness runs under, every field read from the engines themselves.
    Nothing about resolution, T, N_MAX or token width is assumed: a shape mismatch raises here
    instead of producing a plausible-looking throughput number for the wrong pipeline."""
    stages = LAT.discover_engines(engine_dir)
    for want in (detector_stage, 'p3-appearance', head_stage):
        if mode not in stages.get(want, {}):
            raise SystemExit(f'missing {mode} engine for {want} in {engine_dir}; '
                             f'found {sorted(stages)}')
    det = TRU.TRTRunner(trt, cuda, stages[detector_stage][mode])
    head = TRU.TRTRunner(trt, cuda, stages[head_stage][mode])
    buckets = sorted({b for b in (1, 2, 4, n_max) if b <= n_max})
    apps = {b: TRU.TRTRunner(trt, cuda, stages['p3-appearance'][mode], batch=b) for b in buckets}
    a0 = apps[buckets[0]]
    tok_in = [k for k, v in head.inputs.items() if len(v['shape']) == 4]
    msk_in = [k for k, v in head.inputs.items() if len(v['shape']) == 3]
    if not tok_in or not msk_in:
        raise SystemExit(f'head {head_stage} inputs are not (B,T,N,F)+(B,T,N): {list(head.inputs)}')
    _, t_w, n_w, f_w = (int(d) for d in head.inputs[tok_in[0]]['shape'])
    if (n_w, f_w) != (n_max, BRT.FEAT + BRT.APP_DIM):
        raise SystemExit(f'head expects tokens (T={t_w},N={n_w},F={f_w}) but this build assembles '
                         f'N={n_max},F={BRT.FEAT + BRT.APP_DIM} — the run would be meaningless')
    cfg = {'imgsz': int(det.inputs[det.input_order[0]]['shape'][-1]),
           'crop_px': int(a0.inputs[a0.input_order[0]]['shape'][-1]),
           't_window': t_w, 'n_max': n_max, 'conf': BRT.CONF, 'app_buckets': buckets,
           'token_input': tok_in[0], 'mask_input': msk_in[0],
           'detector_stage': detector_stage, 'head_stage': head_stage}
    return det, apps, head, cfg
