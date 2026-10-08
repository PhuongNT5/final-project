"""benchmark-p3-utils.py — shared helpers for benchmark-p3-onnx.py, split out to keep each
file under the ~200-line guideline. Real geometry/crop/percentile logic only, no mocks.
"""
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import build_relation_tokens as BRT  # noqa: E402  (read-only: CONF/N_MAX/geometry() reuse)

_TOK_DIR = os.path.join(ROOT, 'sweep_results', 'relation_tokens')
# P3_TOKEN_TAG selects WHICH token build the crop-count percentiles come from, because the
# crop-count distribution is a property of the detector's operating point, not a constant: the
# 640 px arm must be costed against its own tokens (tokens_bin8i640_*), not against the 800 px
# ones. Default keeps the historical 'yolo26app' set so every previously recorded number
# reproduces unchanged.
_TAG = os.environ.get('P3_TOKEN_TAG', 'yolo26app')
TOKENS = os.path.join(_TOK_DIR, f'tokens_{_TAG}_test.pt')
TOKENS_TRAIN = os.path.join(_TOK_DIR, f'tokens_{_TAG}_train.pt')
CLIP_W, CLIP_H = 1920, 1080     # Unsafe-Net source resolution (build_relation_tokens fallback)
# Last-measured fallback (all8, 516 clips) if the token files aren't present in this checkout
# (e.g. a fresh clone before tokens are rebuilt) — crop_count_percentiles() recomputes these
# from the real stored mask tensors whenever they exist, so this is only a safety net.
FALLBACK_CROP_PCTS = {'p50': 54, 'p95': 106, 'p99': 119, 'cap': 128, 'source': 'fallback'}


def crop_count_percentiles():
    """Real per-clip context-crop counts (p50/p95/p99/max), from the actual stored mask
    tensors (train+test) — not fabricated. Using only the mean (58) understates tail cost;
    the previous version of this file benchmarked the mean crop count only, even though its
    own comment already recorded p95=106 next to it."""
    counts = []
    for p in (TOKENS_TRAIN, TOKENS):
        if os.path.exists(p):
            d = torch.load(p, weights_only=False)
            counts.append(d['mask'].sum(dim=(1, 2)).numpy())
    if not counts:
        print('  [crop-counts] token files missing — using last-measured fallback '
              f'{FALLBACK_CROP_PCTS}')
        return dict(FALLBACK_CROP_PCTS)
    counts = np.concatenate(counts)
    return {'p50': int(round(np.percentile(counts, 50))),
            'p95': int(round(np.percentile(counts, 95))),
            'p99': int(round(np.percentile(counts, 99))),
            'cap': int(counts.max()), 'n_clips': int(len(counts)), 'source': 'measured',
            'token_tag': _TAG}


def postprocess_frame(raw, zones):
    """Confidence-filter + top-N_MAX select + per-box geometry — the detector's raw (300,6)
    [x1,y1,x2,y2,conf,cls] end2end output is NMS-free but still needs this selection step
    before crops/tokens can be built. Mirrors build_relation_tokens.process_clip's own
    postprocessing exactly (same BRT.CONF threshold, BRT.N_MAX cap, geometry() call) so the
    timed cost matches the real decision path, including real Shapely IoU-vs-zone math."""
    conf = raw[:, 4]
    keep = np.where(conf >= BRT.CONF)[0]
    if len(keep) == 0:
        return []
    order = keep[np.argsort(-conf[keep])][:BRT.N_MAX]
    boxes = []
    for oi in order:
        x1, y1, x2, y2, c, cls = raw[oi]
        cx, cy = (x1 + x2) / (2 * CLIP_W), (y1 + y2) / (2 * CLIP_H)
        bw, bh = (x2 - x1) / CLIP_W, (y2 - y1) / CLIP_H
        if int(cls) in (0, 1):
            BRT.geometry(cx, cy, bw, bh, CLIP_W, CLIP_H, zones)  # timed for real cost; unused
        boxes.append({'cx': cx, 'cy': cy, 'w': bw, 'h': bh})
    return boxes


def extract_crops(frame, boxes, crop_size):
    """Real cv2 crop + resize, matching build_relation_tokens.embed_boxes' 30% context pad."""
    out = []
    for b in boxes:
        cx, cy, w, h = b['cx'], b['cy'], b['w'], b['h']
        x1 = max(0, int((cx - w / 2 * 1.3) * CLIP_W)); y1 = max(0, int((cy - h / 2 * 1.3) * CLIP_H))
        x2 = min(CLIP_W, int((cx + w / 2 * 1.3) * CLIP_W)); y2 = min(CLIP_H, int((cy + h / 2 * 1.3) * CLIP_H))
        crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame
        out.append(cv2.resize(crop, (crop_size, crop_size)))
    return out


def sessions(ep, export_dir, graphs, ep_providers):
    import onnxruntime as ort
    out = {}
    for k, fn in graphs.items():
        p = os.path.join(export_dir, fn)
        if not os.path.exists(p):
            sys.exit(f'missing graph: {p}\n  run: python deployment/scripts/export-p3-graphs.py')
        out[k] = ort.InferenceSession(p, providers=ep_providers[ep])
    return out


def synthetic_raw_frames(t_frames, seed=0):
    """Fixed (not-timed) synthetic detector-style raw output, one (300,6) [x1,y1,x2,y2,conf,cls]
    array per frame, in original-clip pixel space (CLIP_W x CLIP_H) — matches what
    postprocess_frame expects (process_clip: ultralytics rescales xywhn to the source frame,
    not the 800x800 detector input). Generated ONCE outside any timed lambda."""
    rng = np.random.default_rng(seed)
    dims = np.array([CLIP_W, CLIP_H])
    frames = []
    for _ in range(t_frames):
        x1y1 = rng.uniform(0, dims * 0.7, size=(300, 2))
        wh = rng.uniform(10, dims * 0.3, size=(300, 2))
        x2y2 = np.clip(x1y1 + wh, [0, 0], dims)
        conf = rng.uniform(0, 1, size=300).astype(np.float32)
        cls = rng.integers(0, 4, size=300).astype(np.float32)
        frames.append(np.concatenate([x1y1, x2y2, conf[:, None], cls[:, None]], axis=1).astype(np.float32))
    return frames


def timed(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return {str(p): round(float(np.percentile(a, p)), 2) for p in (50, 95, 99)}
