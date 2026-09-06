"""Final scoring against the paper - run once per finished configuration."""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch

from . import config as C

N = C.NAMES4


def bootstrap_ci(y, pred, fn, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def macro(y, p):
    r = [np.mean(p[y == k] == k) for k in range(4) if (y == k).any()]
    return float(np.mean(r)) if r else 0.0


def top1(y, p):
    return float((y == p).mean())


def report(P: torch.Tensor, y: np.ndarray, title: str) -> dict:
    pred = P.argmax(1).numpy()
    rec = {k: float(np.mean(pred[y == k] == k)) for k in range(4)}
    cm = np.zeros((4, 4), int)
    for t, q in zip(y, pred):
        cm[t, q] += 1
    t1, mc = top1(y, pred), macro(y, pred)
    ci1, cim = bootstrap_ci(y, pred, top1), bootstrap_ci(y, pred, macro)
    print(f'\n=== {title} (N={len(y)}) ===')
    print(f'  top-1 = {int((pred==y).sum())}/{len(y)} = {t1*100:.2f}%   95% CI [{ci1[0]*100:.1f}, {ci1[1]*100:.1f}]')
    print(f'  macro = {mc*100:.2f}%                 95% CI [{cim[0]*100:.1f}, {cim[1]*100:.1f}]')
    print('  per-class recall: ' + '  '.join(f'{N[k]} {rec[k]*100:.1f}% ({int(cm[k,k])}/{int(cm[k].sum())})' for k in range(4)))
    print('  confusion (rows=true):')
    for k in range(4):
        print(f'    {N[k]:14s} ' + ' '.join(f'{v:3d}' for v in cm[k]))
    for key, lab in (('full', 'Unsafe-Net full (YOLOv4+ConvLSTM, bbox)'), ('nobbox', 'ConvLSTM-only (no bbox)')):
        p = C.PAPER[key]
        print(f'  vs {lab}: top-1 {t1*100:.2f} vs {p["top1"]*100:.2f} ({t1-p["top1"]:+.2%}) | '
              f'macro {mc*100:.2f} vs {p["macro"]*100:.2f} ({mc-p["macro"]:+.2%})')
    return {'title': title, 'n': int(len(y)), 'correct': int((pred == y).sum()), 'top1': t1,
            'macro': mc, 'top1_ci': ci1, 'macro_ci': cim, 'per_class_recall': rec,
            'confusion': cm.tolist()}


def run(path='fusion', save=True):
    d = torch.load(os.path.join(C.OUT_DIR, f'{path}.pt'), weights_only=False)
    out = {'val': report(d['Pv'], d['yv'], f'{path} - VAL (selection slice)'),
           'test': report(d['Pt'], d['yt'], f'{path} - BLIND TEST (64 clips)')}
    if 'picked' in d:
        out['picked'] = d['picked']
    if save:
        p = os.path.join(C.OUT_DIR, f'report_{path}.json')
        json.dump(out, open(p, 'w'), indent=1)
        print(f'\nsaved {p}')
    return out


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--path', default='fusion')
    run(a.parse_args().path)
