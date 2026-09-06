"""Stage D - blind fusion of stream posteriors.

Members are chosen by greedy forward selection *with replacement* maximising val macro accuracy
(Caruana ensemble selection).  Every decision - which members, how many times each - is made on the
val slice; the test posteriors are combined once with the resulting weights.
"""
from __future__ import annotations
import argparse
import glob
import os

import numpy as np
import torch

from . import config as C


def _macro(y, p, n=4):
    return float(np.mean([np.mean(p[y == k] == k) for k in range(n) if (y == k).any()]))


def load_members(pattern: str = '*') -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(C.OUT_DIR, f'temporal_{pattern}.pt'))):
        out.append(torch.load(f, weights_only=False))
    return out


def greedy_select(members: list[dict], rounds: int = 12, score=('macro', 'top1')):
    """Return (counts, val_macro, val_top1); ties on macro broken by val top-1."""
    yv = members[0]['yv']
    for m in members:
        assert np.array_equal(m['yv'], yv), 'members disagree on the val slice'
    counts = np.zeros(len(members))
    acc = torch.zeros_like(members[0]['Pv'])
    best_hist = []
    for _ in range(rounds):
        cand = []
        for i, m in enumerate(members):
            p = ((acc + m['Pv']) / (counts.sum() + 1)).argmax(1).numpy()
            cand.append((_macro(yv, p), float((p == yv).mean()), i))
        cand.sort(reverse=True)
        s_macro, s_top1, i = cand[0]
        counts[i] += 1
        acc = acc + members[i]['Pv']
        best_hist.append((s_macro, s_top1, counts.copy()))
    k = int(np.argmax([h[0] + 0.001 * h[1] for h in best_hist]))
    return best_hist[k][2], best_hist[k][0], best_hist[k][1]



def bagged_select(members, rounds=12, bags=40, frac=0.8, seed=0):
    """Bagged Caruana selection: greedy on many val subsamples, counts averaged.

    A single greedy pass over 83 val clips latches onto one member; bagging over subsamples keeps
    the members that win repeatedly and is still a val-only decision.
    """
    rng = np.random.RandomState(seed)
    yv = members[0]['yv']
    n = len(yv)
    total = np.zeros(len(members))
    for b in range(bags):
        idx = rng.choice(n, int(round(frac * n)), replace=False)
        sub = [{'yv': yv[idx], 'Pv': m['Pv'][idx]} for m in members]
        c, _, _ = greedy_select(sub, rounds)
        total += c / max(1.0, c.sum())
    counts = total / total.sum()
    P = sum(float(w) * m['Pv'] for w, m in zip(counts, members))
    pred = P.argmax(1).numpy()
    return counts, _macro(yv, pred), float((pred == yv).mean())


def apply_weights(members, counts, key='Pt'):
    w = counts / counts.sum()
    P = sum(float(wi) * m[key] for wi, m in zip(w, members) if wi > 0)
    return P


def run(pattern='*', rounds=12, save='fusion', bagged=True):
    members = load_members(pattern)
    assert members, 'no temporal_*.pt members found'
    counts, vm, vt = bagged_select(members, rounds) if bagged else greedy_select(members, rounds)
    Pt = apply_weights(members, counts, 'Pt')
    Pv = apply_weights(members, counts, 'Pv')
    picked = {m['name']: round(float(c), 3) for m, c in zip(members, counts) if c > 1e-6}
    res = {'members': [m['name'] for m in members], 'counts': counts.tolist(), 'picked': picked,
           'val_macro': vm, 'val_top1': vt, 'Pv': Pv, 'yv': members[0]['yv'],
           'Pt': Pt, 'yt': members[0]['yt'], 'test_names': members[0]['test_names']}
    torch.save(res, os.path.join(C.OUT_DIR, f'{save}.pt'))
    print('[fusion] picked (val-selected):', picked)
    print(f'[fusion] val macro={vm*100:.1f} top1={vt*100:.1f}')
    return res


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--pattern', default='*'); a.add_argument('--rounds', type=int, default=12)
    a.add_argument('--save', default='fusion')
    v = a.parse_args()
    run(v.pattern, v.rounds, v.save)
