"""Training-free clip decisions from the per-frame recogniser - the paper's aggregation step.

Members produced here mirror Unsafe-Net's own inference logic: `mean` is plain temporal averaging
(the paper's no-reduction baseline), `gate` keeps only the frames the recogniser is confident carry
an unsafe behaviour (the YOLO-v4 frame-reduction), and `pair` scores each behaviour against its safe
counterpart before averaging.  All are written in the same member format the fusion stage consumes.
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import torch

from . import config as C
from . import frame_model as FM
from .data_index import load_clips, blind_split


def _load(tag, sp):
    return torch.load(os.path.join(C.OUT_DIR, f'frameout_{tag}_{sp}.pt'), weights_only=False)


def _macro(y, p):
    return float(np.mean([np.mean(p[y == k] == k) for k in range(4) if (y == k).any()]))


def _scores(L8: torch.Tensor, mode: str, q: float):
    """(N,T,8) logits -> (N,4) posterior."""
    if mode == 'pair':
        s = FM.pair_contrast(L8)                      # unsafe_k - safe_k
        return torch.softmax(s, -1).mean(1)
    p = torch.softmax(L8[..., :4], -1)
    if mode == 'mean':
        return p.mean(1)
    if mode == 'max':
        return p.max(1).values / p.max(1).values.sum(-1, keepdim=True)
    if mode == 'gate':
        # frame-reduction: keep the top-q fraction of frames by unsafe-vs-safe evidence
        conf = torch.softmax(L8, -1)[..., :4].sum(-1)          # P(any unsafe behaviour | frame)
        k = max(1, int(round(q * L8.shape[1])))
        idx = conf.topk(k, dim=1).indices
        sel = torch.stack([p[i, idx[i]] for i in range(len(p))])
        return sel.mean(1)
    raise ValueError(mode)


def run(tag='b0', split_seed=0, gates=(0.25, 0.5, 0.75)):
    tr, te = _load(tag, 'train'), _load(tag, 'test')
    ytr, yte = np.array(tr['labels']), np.array(te['labels'])
    tr_u, te_u = np.where(ytr < 4)[0], np.where(yte < 4)[0]
    clips_u = [c for c in load_clips(['train']) if c.is_unsafe]
    _, val_i = blind_split(clips_u, seed=split_seed)
    g_val = tr_u[val_i]
    Ltr, Lte = tr['logits'].float(), te['logits'].float()
    made = []
    for mode in ('mean', 'max', 'pair', 'gate'):
        for q in (gates if mode == 'gate' else (1.0,)):
            name = f'agg{tag}-{mode}' + (f'{int(q*100)}' if mode == 'gate' else '')
            Pv, Pt = _scores(Ltr[g_val], mode, q), _scores(Lte[te_u], mode, q)
            yv, yt = ytr[g_val], yte[te_u]
            res = {'name': name, 'tag': tag, 'kind': 'agg', 'use': [mode], 'seeds': 1,
                   'val_macro': _macro(yv, Pv.argmax(1).numpy()),
                   'val_top1': float((Pv.argmax(1).numpy() == yv).mean()),
                   'Pv': Pv, 'yv': yv, 'Pt': Pt, 'yt': yt,
                   'test_names': [te['names'][i] for i in te_u],
                   'val_names': [tr['names'][i] for i in g_val], 'per_seed': []}
            torch.save(res, os.path.join(C.OUT_DIR, f'temporal_{name}.pt'))
            print(f'[{name}] val macro={res["val_macro"]*100:.1f} top1={res["val_top1"]*100:.1f}', flush=True)
            made.append(name)
    return made


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--tag', default='b0')
    run(a.parse_args().tag)
