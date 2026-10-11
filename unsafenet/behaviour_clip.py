"""Paper-faithful clip decision from the behaviour detector (Unsafe-Net's YOLO v4 -> ConvLSTM path).

The detector already emits a behaviour class per box, so a clip decision is an aggregation over its
frames.  Members produced here mirror the paper's own inference: `det-mean`/`det-max` aggregate the
per-frame unsafe evidence, `det-pair` scores each unsafe behaviour against its safe counterpart, and
`det-gateQ` first applies the frame-reduction step (keep only frames carrying unsafe evidence).
Written in the same member format as the other stages so fusion/eval consume them unchanged.
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import torch
from ultralytics import YOLO

from . import config as C
from .data_index import load_clips, blind_split
from .frame_cache import frame_paths

CACHE = os.path.join(C.CACHE_DIR, 'behaviour_conf_{}{}.npy')
G22_FLOOR, G22_GATE = 0.85, 0.90   # pre-registered in `kaggle-detector-retrain/...-gpu.py`


def detector_gate_status(weights: str):
    """Peak mAP50 of the detector behind `weights`, and its standing against gate G22.

    Stamped into every member artifact so a downstream table cannot silently present a clip number
    built on a below-floor detector as a gated result.  A comment in this file is not enforcement:
    nothing can read it.  The figure is read from the run's own `results.csv`, never hand-entered.
    """
    csv = os.path.join(os.path.dirname(os.path.dirname(weights)), 'results.csv')
    if not os.path.exists(csv):
        return {'detector_map50_peak': None, 'gate_status': 'UNKNOWN - no results.csv found'}
    rows = [l.split(',') for l in open(csv).read().strip().splitlines()]
    col = rows[0].index('metrics/mAP50(B)')
    peak = max(float(r[col]) for r in rows[1:])
    status = ('PASSES G22' if peak >= G22_GATE else
              'BELOW GATE, ABOVE FLOOR - label "detector under gate"' if peak >= G22_FLOOR else
              'BELOW FLOOR - EXPLORATORY ONLY, never cite as a gated result')
    return {'detector_map50_peak': peak, 'gate_status': status}


def frame_scores(weights: str, subsets=('train', 'test'), imgsz=416, conf=0.05, device='mps', tag=''):
    """(N, T, 8) per-frame max confidence for each behaviour class."""
    model = YOLO(weights)
    for sp in subsets:
        out = CACHE.format(tag, sp)
        if os.path.exists(out):
            continue
        clips = load_clips([sp])
        S = np.zeros((len(clips), C.T, 8), np.float32)
        for i, c in enumerate(clips):
            paths = frame_paths(c)
            res = model.predict(paths, imgsz=imgsz, conf=conf, verbose=False, device=device)
            for t, r in enumerate(res):
                if r.boxes is None or not len(r.boxes):
                    continue
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                for k in range(8):
                    m = cls == k
                    if m.any():
                        S[i, t, k] = cf[m].max()
            if (i + 1) % 100 == 0:
                print(f'  [{sp}] {i+1}/{len(clips)}', flush=True)
        np.save(out, S)
        print(f'[behaviour_clip] {sp}: {S.shape}', flush=True)


def _macro(y, p):
    return float(np.mean([np.mean(p[y == k] == k) for k in range(4) if (y == k).any()]))


def _score(S: np.ndarray, mode: str, q: float = 1.0) -> torch.Tensor:
    U, Sf = S[..., :4], S[..., [C.SAFE_OF[k] for k in range(4)]]
    if mode == 'pair':
        X = U - Sf
    else:
        X = U
    if mode == 'max':
        v = X.max(1)
    elif mode == 'gate':
        keep = U.sum(-1)                                   # per-frame unsafe evidence
        k = max(1, int(round(q * S.shape[1])))
        idx = np.argsort(-keep, axis=1)[:, :k]
        v = np.stack([X[i, idx[i]].mean(0) for i in range(len(X))])
    else:
        v = X.mean(1)
    return torch.softmax(torch.from_numpy(v).float() * 8.0, -1)


def run(split_seed=0, gates=(0.25, 0.5, 0.75), tag='', weights=''):
    Str, Ste = np.load(CACHE.format(tag, 'train')), np.load(CACHE.format(tag, 'test'))
    tr, te = load_clips(['train']), load_clips(['test'])
    ytr, yte = np.array([c.label for c in tr]), np.array([c.label for c in te])
    tr_u, te_u = np.where(ytr < 4)[0], np.where(yte < 4)[0]
    gate = detector_gate_status(weights) if weights else {}
    _, val_i = blind_split([c for c in tr if c.is_unsafe], seed=split_seed)
    g_val = tr_u[val_i]
    made = []
    for mode in ('mean', 'max', 'pair', 'gate'):
        for q in (gates if mode == 'gate' else (1.0,)):
            name = f'det{tag}-{mode}' + (f'{int(q*100)}' if mode == 'gate' else '')
            Pv, Pt = _score(Str[g_val], mode, q), _score(Ste[te_u], mode, q)
            yv, yt = ytr[g_val], yte[te_u]
            res = {'name': name, 'kind': 'det', 'use': [mode], 'seeds': 1,
                   'val_macro': _macro(yv, Pv.argmax(1).numpy()),
                   'val_top1': float((Pv.argmax(1).numpy() == yv).mean()),
                   'Pv': Pv, 'yv': yv, 'Pt': Pt, 'yt': yt,
                   'test_names': [te[i].name for i in te_u],
                   'val_names': [tr[i].name for i in g_val], 'per_seed': [],
                   'detector_weights': weights, **gate}
            torch.save(res, os.path.join(C.OUT_DIR, f'temporal_{name}.pt'))
            print(f'[{name}] val macro={res["val_macro"]*100:.1f} top1={res["val_top1"]*100:.1f}', flush=True)
            made.append(name)
    return made


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    # Default is the best VALID behaviour detector: beh_ctx, mAP50 0.806 at imgsz 416 on the
    # clip-level val split (`runs/detect/sweep_results/r2/beh_ctx/results.csv`, epoch 15).  The
    # old default pointed at `sweep_results/r2/behaviour_yolo/weights/best.pt`, which does not
    # exist; the run of that name (`runs/detect/.../behaviour_yolo`) is an Ultralytics-on-MPS
    # convergence failure that never exceeded mAP50 0.015 and must never be used as a detector.
    # 0.806 is BELOW gate G22's 0.85 floor, so any clip number built on it is exploratory and
    # must be labelled as such.
    a.add_argument('--weights', default=os.path.join(
        C.ROOT, 'runs/detect/sweep_results/r2/beh_ctx/weights/best.pt'))
    a.add_argument('--device', default='mps'); a.add_argument('--tag', default='')
    v = a.parse_args()
    frame_scores(v.weights, device=v.device, tag=v.tag)
    run(tag=v.tag, weights=v.weights)
