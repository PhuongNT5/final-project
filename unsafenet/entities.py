"""Stage A - entity detections + greedy IoU tracks, cached per clip.

Reuses the already-trained YOLOv11s entity detector (mAP@50 = 0.969 on a clip-level held-out
split).  Output per clip: a (T, N_BOX, 7) array [cls, conf, xc, yc, w, h, track_id] in normalised
coordinates, plus the raw per-frame detection lists for geometry.
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import torch

from . import config as C
from .data_index import load_clips
from .frame_cache import frame_paths

CACHE = os.path.join(C.CACHE_DIR, 'entities')
os.makedirs(CACHE, exist_ok=True)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def link_tracks(per_frame: list[np.ndarray], thr: float = 0.25) -> list[np.ndarray]:
    """Greedy IoU association across consecutive frames; appends a track_id column."""
    out, prev, next_id = [], [], 0
    for dets in per_frame:
        cur = np.zeros((len(dets), 7), dtype=np.float32)
        cur[:, :6] = dets
        used = set()
        for i, d in enumerate(dets):
            best, bj = thr, -1
            for j, (pd, pid) in enumerate(prev):
                if j in used or int(pd[0]) != int(d[0]):
                    continue
                v = _iou(d[2:6], pd[2:6])
                if v > best:
                    best, bj = v, j
            if bj >= 0:
                cur[i, 6] = prev[bj][1]; used.add(bj)
            else:
                cur[i, 6] = next_id; next_id += 1
        out.append(cur)
        prev = [(cur[i, :6], cur[i, 6]) for i in range(len(cur))]
    return out


def path_for(clip) -> str:
    return os.path.join(CACHE, f'{clip.subset}__{clip.name}.npz')


def load(clip):
    d = np.load(path_for(clip), allow_pickle=True)
    return list(d['frames'])


def build(subsets=('train', 'test'), device='mps', conf=C.DET_CONF, force=False):
    from ultralytics import YOLO
    model = YOLO(C.DETECTOR)
    clips = load_clips(subsets)
    todo = [c for c in clips if force or not os.path.exists(path_for(c))]
    print(f'[entities] {len(todo)}/{len(clips)} clips', flush=True)
    for k, c in enumerate(todo):
        paths = frame_paths(c)
        res = model.predict(paths, imgsz=C.DET_IMGSZ, conf=conf, device=device,
                            verbose=False, max_det=C.N_BOX)
        per_frame = []
        for r in res:
            b = r.boxes
            if b is None or len(b) == 0:
                per_frame.append(np.zeros((0, 6), dtype=np.float32)); continue
            xywhn = b.xywhn.cpu().numpy()
            arr = np.concatenate([b.cls.cpu().numpy()[:, None], b.conf.cpu().numpy()[:, None], xywhn], 1)
            per_frame.append(arr.astype(np.float32)[np.argsort(-arr[:, 1])][:C.N_BOX])
        np.savez_compressed(path_for(c), frames=np.array(link_tracks(per_frame), dtype=object))
        if (k + 1) % 50 == 0:
            print(f'  [{k+1}/{len(todo)}] {c.key}', flush=True)
    print('[entities] done', flush=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--device', default='mps'); a.add_argument('--force', action='store_true')
    v = a.parse_args()
    build(device=v.device, force=v.force)
