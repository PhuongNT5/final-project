"""Per-clip interpretable stream: entity evidence + zone geometry + person motion.

Produces (T, D_FRAME) per-frame vectors and a (D_CLIP,) clip-level motion summary.  Motion is the
cue the earlier token pipeline never encoded, yet it is what physically separates the two confused
classes: a walkway violation is someone *walking across open floor*, an unauthorized intervention
is someone *stationary at a machine*.
"""
from __future__ import annotations
import argparse
import os

import cv2
import numpy as np
import torch

from . import config as C
from . import entities as E
from . import geometry as G
from .data_index import load_clips
from .frame_cache import bg_path

D_FRAME = 8 + 12          # per-class max conf + person/zone summary
D_CLIP = 10
OUTP = os.path.join(C.OUT_DIR, 'streams_{}.pt')


def _frame_vec(dets: np.ndarray, polys: dict, mask) -> np.ndarray:
    v = np.zeros(D_FRAME, dtype=np.float32)
    for cid in range(8):
        m = dets[:, 0] == cid
        v[cid] = dets[m, 1].max() if m.any() else 0.0
    persons = dets[np.isin(dets[:, 0], C.PERSON_IDS)]
    if len(persons):
        feats = np.array([G.person_zone_feats(p[2:6], polys, mask) for p in persons], np.float32)
        areas = persons[:, 4] * persons[:, 5]
        j = int(np.argmax(areas))                        # most prominent person
        v[8] = len(persons)
        v[9] = areas.max()
        v[10] = feats[j, 0]                              # foot inside walkway polygon
        v[11] = feats[j, 1]                              # signed distance to walkway
        v[12] = feats[:, 1].max()
        v[13] = float((feats[:, 0] == 0).mean())         # fraction of people off the walkway
        v[14] = feats[j, 2]                              # inside restricted zone
        v[15] = feats[j, 3]
        v[16] = feats[j, 4]                              # green paint under the foot point
        v[17] = persons[:, 1][persons[:, 0] == 1].max() if (persons[:, 0] == 1).any() else 0.0
    v[18] = v[C.PANEL_IDS[1]] - v[C.PANEL_IDS[0]]        # opened - closed
    v[19] = v[C.FORKLIFT_IDS[1]] - v[C.FORKLIFT_IDS[0]]  # overload - safe load
    return v


def _motion_vec(frames: list[np.ndarray]) -> np.ndarray:
    """Trajectory statistics of the longest-lived person track."""
    tracks: dict[int, list[tuple[int, np.ndarray]]] = {}
    for t, d in enumerate(frames):
        for row in d:
            if int(row[0]) in C.PERSON_IDS:
                tracks.setdefault(int(row[6]), []).append((t, row[2:6]))
    v = np.zeros(D_CLIP, dtype=np.float32)
    if not tracks:
        return v
    tid = max(tracks, key=lambda k: len(tracks[k]))
    pts = np.array([p for _, p in tracks[tid]], np.float32)
    steps = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1) if len(pts) > 1 else np.zeros(1)
    v[0] = len(tracks)
    v[1] = len(pts) / max(1, len(frames))                # track persistence
    v[2] = np.linalg.norm(pts[-1, :2] - pts[0, :2])      # net displacement
    v[3] = steps.sum()                                   # path length
    v[4] = steps.mean(); v[5] = steps.std(); v[6] = steps.max()
    v[7] = float((steps < 0.004).mean())                 # dwell fraction (near-stationary frames)
    v[8] = pts[:, 0].max() - pts[:, 0].min()
    v[9] = pts[:, 1].max() - pts[:, 1].min()
    return v


def build(subsets=('train', 'test')) -> None:
    for sp in subsets:
        clips = load_clips([sp])
        F = np.zeros((len(clips), C.T, D_FRAME), np.float32)
        M = np.zeros((len(clips), D_CLIP), np.float32)
        for i, c in enumerate(clips):
            frames = E.load(c)
            polys = G.clip_polygons(c, c.view)
            bg = cv2.imread(bg_path(c))
            mask = G.green_walkway_mask(bg) if bg is not None else None
            for t, d in enumerate(frames[:C.T]):
                F[i, t] = _frame_vec(d, polys, mask)
            M[i] = _motion_vec(frames)
            if (i + 1) % 100 == 0:
                print(f'  [{sp}] {i+1}/{len(clips)}', flush=True)
        torch.save({'names': [c.name for c in clips], 'labels': [c.label for c in clips],
                    'views': [c.view for c in clips], 'frame': torch.from_numpy(F),
                    'motion': torch.from_numpy(M)}, OUTP.format(sp))
        print(f'[streams] {sp}: frame{tuple(F.shape)} motion{tuple(M.shape)}', flush=True)


def load(sp: str):
    return torch.load(OUTP.format(sp), weights_only=False)


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args()
    build()
