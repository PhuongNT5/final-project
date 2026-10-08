"""Measure what the P3 pipeline loses when the detector input shrinks below DET_IMGSZ.

Motivation: on Jetson the detector is 62.3 % of the decision budget (Table 5.5H), and
detector latency scales with input pixels, so a smaller input is the strongest cadence
lever available. The cost is detection quality. This script quantifies that cost at the
only place it matters for P3 -- the entity token set the relation head actually consumes
(top-N_BOX boxes by confidence, as built in unsafenet/entities.py).

TRAIN CLIPS ONLY. Never reads data/test: picking an operating point is model selection,
and selection on the blind split is forbidden (standing rule 2).

Detector/conf come from build_relation_tokens.py, NOT from unsafenet/config.py. The two
disagree: config.py names detector_best.pt (YOLO11) at conf 0.20 and feeds the entity cache
lane, while the P3 relation-token pipeline -- and the exported p3-detector-fp32.onnx this
experiment is costing -- uses yolo26_best.pt at conf 0.25. Measuring the config pair here
would cost the wrong model.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unsafenet.config as C  # noqa: E402

FRAME_ROOT = 'cache_r2/frames/train'
SIZES = [800, 640, 512, 416]
PIPE_DETECTOR = 'sweep_results/entity_detector/yolo26_best.pt'   # build_relation_tokens.DET override
PIPE_CONF = 0.25                                                # build_relation_tokens.CONF


def _iou_matrix(a, b):
    """a: (N,4) b: (M,4), both xyxy. Returns (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return (inter / np.clip(area_a + area_b - inter, 1e-9, None)).astype(np.float32)


def _greedy_match(ref, cand, iou_thr):
    """Greedy highest-IoU matching, class-aware. ref/cand: dicts with xyxy + cls."""
    iou = _iou_matrix(ref['xyxy'], cand['xyxy'])
    if iou.size:
        same = ref['cls'][:, None] == cand['cls'][None, :]
        iou = np.where(same, iou, 0.0)
    matched, used_r, used_c = 0, set(), set()
    order = np.argsort(-iou.flatten())
    for flat in order:
        if iou.size == 0:
            break
        r, c = divmod(int(flat), iou.shape[1])
        if iou[r, c] < iou_thr:
            break
        if r in used_r or c in used_c:
            continue
        used_r.add(r), used_c.add(c)
        matched += 1
    return matched


def _predict(model, paths, imgsz, conf, device, batch=8):
    """Run detector in chunks, return per-frame top-N_BOX dict in pixel xyxy.

    Chunked because ultralytics batches a whole path list into one tensor, which
    overflows MPS at a few hundred frames.

    No max_det: build_relation_tokens.process_clip takes every detection and then keeps
    argsort(-conf)[:N_MAX]. Capping inside predict() is a different selection rule, so the
    cap is applied here the same way the pipeline applies it.
    """
    out = []
    res = []
    for i in range(0, len(paths), batch):
        res += model.predict(paths[i:i + batch], imgsz=imgsz, conf=conf, device=device,
                             verbose=False)
    for r in res:
        b = r.boxes
        if b is None or len(b) == 0:
            out.append({'xyxy': np.zeros((0, 4), np.float32),
                        'cls': np.zeros((0,), np.int32), 'conf': np.zeros((0,), np.float32)})
            continue
        cf = b.conf.cpu().numpy()
        keep = np.argsort(-cf)[:C.N_BOX]
        out.append({'xyxy': b.xyxy.cpu().numpy()[keep].astype(np.float32),
                    'cls': b.cls.cpu().numpy()[keep].astype(np.int32),
                    'conf': cf[keep].astype(np.float32)})
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--clips', type=int, default=24, help='train clips to sample')
    a.add_argument('--device', default='mps')
    a.add_argument('--iou', type=float, default=0.5)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--detector', default=PIPE_DETECTOR)
    a.add_argument('--conf', type=float, default=PIPE_CONF)
    a.add_argument('--out', default='deployment/artifacts/detector-imgsz-tradeoff.json')
    v = a.parse_args()

    from ultralytics import YOLO
    model = YOLO(v.detector)

    clip_dirs = sorted(glob.glob(os.path.join(FRAME_ROOT, '*')))
    rng = np.random.default_rng(v.seed)
    pick = [clip_dirs[i] for i in rng.choice(len(clip_dirs), size=min(v.clips, len(clip_dirs)),
                                             replace=False)]
    frames = []
    for d in pick:
        frames += sorted(f for f in glob.glob(os.path.join(d, 'f*.jpg')))
    print(f'[tradeoff] {len(pick)} train clips, {len(frames)} frames, '
          f'detector={os.path.basename(v.detector)}, conf={v.conf}, '
          f'N_BOX={C.N_BOX}, iou_thr={v.iou}', flush=True)

    per_size, timing = {}, {}
    for s in SIZES:
        t0 = time.perf_counter()
        per_size[s] = _predict(model, frames, s, v.conf, v.device)
        timing[s] = round((time.perf_counter() - t0) / max(len(frames), 1) * 1000, 2)
        n = sum(len(f['cls']) for f in per_size[s])
        print(f'  imgsz={s:4d}  {n:6d} boxes  {timing[s]:7.2f} ms/frame (host, not Jetson)',
              flush=True)

    ref = per_size[SIZES[0]]
    n_ref = sum(len(f['cls']) for f in ref)
    results = {}
    for s in SIZES:
        cand = per_size[s]
        n_cand = sum(len(f['cls']) for f in cand)
        matched = sum(_greedy_match(ref[i], cand[i], v.iou) for i in range(len(frames)))
        same_count = sum(len(ref[i]['cls']) == len(cand[i]['cls']) for i in range(len(frames)))
        empty_new = sum(1 for i in range(len(frames))
                        if len(ref[i]['cls']) > 0 and len(cand[i]['cls']) == 0)
        results[s] = {
            'boxes': n_cand,
            'recall_vs_800': round(matched / n_ref, 4) if n_ref else None,
            'precision_vs_800': round(matched / n_cand, 4) if n_cand else None,
            'frames_identical_count': round(same_count / len(frames), 4),
            'frames_emptied': empty_new,
            'host_ms_per_frame': timing[s],
        }
        r = results[s]
        print(f'  imgsz={s:4d}  recall={r["recall_vs_800"]}  prec={r["precision_vs_800"]}  '
              f'same-count frames={r["frames_identical_count"]}  emptied={empty_new}')

    import platform
    import socket
    payload = {
        'note': 'Entity-set agreement vs the imgsz=800 reference, TRAIN clips only. Detector and '
                'conf are the P3 relation-token pipeline pair (yolo26_best.pt @ 0.25), NOT the '
                'unsafenet/config.py pair (detector_best.pt @ 0.20) which serves the entity-cache '
                'lane. host_ms_per_frame is this host, NOT a Jetson number and not quotable as one.',
        'host': socket.gethostname(), 'platform': platform.platform(),
        'detector': v.detector, 'det_conf': v.conf, 'n_box': C.N_BOX,
        'iou_thr': v.iou, 'n_clips': len(pick), 'n_frames': len(frames), 'seed': v.seed,
        'clips': [os.path.basename(p) for p in pick],
        'results': {str(k): val for k, val in results.items()},
    }
    os.makedirs(os.path.dirname(v.out), exist_ok=True)
    with open(v.out, 'w') as f:
        json.dump(payload, f, indent=1)
    print(f'[SAVED] {v.out}')


if __name__ == '__main__':
    main()
