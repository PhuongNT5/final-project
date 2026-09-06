"""Stage C driver - train temporal heads on the cached multi-stream frame evidence.

Blind protocol throughout: fit/val are clip-level slices of the TRAIN clips, the epoch and the
configuration are chosen on val macro accuracy, and the 64-clip test posteriors are only *stored*
here (never used for any decision).  Fusion weights are likewise chosen on val, in fusion.py.
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import streams as S
from .data_index import load_clips, blind_split
from .temporal import TemporalNet

STREAMS = ('emb', 'logit', 'geom')


def load_features(tag: str, sp: str, use=STREAMS):
    """`tag` may be several backbone tags joined by '+' - their streams are concatenated."""
    tags = tag.split('+')
    fos = [torch.load(os.path.join(C.OUT_DIR, f'frameout_{t}_{sp}.pt'), weights_only=False) for t in tags]
    fo = fos[0]
    st = S.load(sp)
    assert fo['names'] == st['names'], 'stream/frame caches disagree on clip order'
    assert all(f['names'] == fo['names'] for f in fos), 'backbone caches disagree on clip order'
    parts = []
    if 'emb' in use:
        parts += [f['feats'].float() for f in fos]
    if 'logit' in use:
        parts += [f['logits'].float() for f in fos]
    if 'geom' in use:
        parts.append(st['frame'].float())
    X = torch.cat(parts, -1)
    return X, st['motion'].float(), np.array(fo['labels']), fo['names'], np.array(fo['views'], dtype=object)


def _macro(y, p, n=4):
    return float(np.mean([np.mean(p[y == k] == k) for k in range(n) if (y == k).any()]))


def _fit_one(Xf, Cf, yf, Xv, Cv, yv, kind, seed, epochs, lr, d, p, layers, aux, device, n_cls):
    torch.manual_seed(seed); np.random.seed(seed)
    m = TemporalNet(Xf.shape[-1], Cf.shape[-1], kind, d, n_cls, p, layers).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    yf_t = torch.from_numpy(yf).to(device)
    best = {'macro': -1.0}
    bs = 32
    for ep in range(epochs):
        m.train()
        idx = torch.randperm(len(Xf))
        for i in range(0, len(Xf), bs):
            b = idx[i:i + bs]
            logit, fr = m(Xf[b].to(device), Cf[b].to(device), return_frames=True)
            yb = yf_t[b]
            loss = nn.functional.cross_entropy(logit, yb, label_smoothing=0.05)
            if aux > 0:
                t = fr.shape[1]
                loss = loss + aux * nn.functional.cross_entropy(
                    fr.reshape(-1, n_cls), yb.view(-1, 1).expand(-1, t).reshape(-1), label_smoothing=0.05)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            pv = torch.softmax(m(Xv.to(device), Cv.to(device))[:, :4], -1).cpu()
        mm = _macro(yv, pv.argmax(1).numpy())
        if mm > best['macro']:
            best = {'macro': mm, 'ep': ep, 'state': {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}}
    m.load_state_dict(best['state'])
    return m, best


@torch.no_grad()
def _post(m, X, Cc, device, bs=64):
    m.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.softmax(m(X[i:i + bs].to(device), Cc[i:i + bs].to(device))[:, :4], -1).cpu())
    return torch.cat(out)


def run(tag='b0', kind='lstm', use=STREAMS, seeds=5, epochs=40, lr=1e-3, d=128, p=0.3,
        layers=1, aux=0.3, device='mps', name=None, n_cls=4, split_seed=0):
    Xtr, Mtr, ytr, ntr, vtr = load_features(tag, 'train', use)
    Xte, Mte, yte, nte, vte = load_features(tag, 'test', use)
    tr_u = np.where(ytr < 4)[0]; te_u = np.where(yte < 4)[0]
    clips_u = [c for c in load_clips(['train']) if c.is_unsafe]
    fit_i, val_i = blind_split(clips_u, seed=split_seed)
    g_fit, g_val = tr_u[fit_i], tr_u[val_i]

    mu, sd = Xtr[g_fit].reshape(-1, Xtr.shape[-1]).mean(0), Xtr[g_fit].reshape(-1, Xtr.shape[-1]).std(0) + 1e-5
    mm, ms = Mtr[g_fit].mean(0), Mtr[g_fit].std(0) + 1e-5
    nz = lambda X, M: ((X - mu) / sd, (M - mm) / ms)
    Xf, Cf = nz(Xtr[g_fit], Mtr[g_fit]); Xv, Cv = nz(Xtr[g_val], Mtr[g_val]); Xt, Ct = nz(Xte[te_u], Mte[te_u])
    yf, yv, yt = ytr[g_fit], ytr[g_val], yte[te_u]

    Pv = torch.zeros(len(yv), 4); Pt = torch.zeros(len(yt), 4); per_seed = []
    for s in range(seeds):
        m, b = _fit_one(Xf, Cf, yf, Xv, Cv, yv, kind, s, epochs, lr, d, p, layers, aux, device, n_cls)
        pv, pt = _post(m, Xv, Cv, device), _post(m, Xt, Ct, device)
        Pv += pv; Pt += pt
        per_seed.append({'seed': s, 'val_macro': b['macro'], 'ep': b['ep'],
                         'test_top1': float((pt.argmax(1).numpy() == yt).mean())})
        print(f'  seed{s} val_macro={b["macro"]*100:.1f} (ep{b["ep"]})', flush=True)
    Pv /= seeds; Pt /= seeds
    vm = _macro(yv, Pv.argmax(1).numpy())
    name = name or f'{kind}_{"-".join(use)}'
    res = {'name': name, 'tag': tag, 'kind': kind, 'use': list(use), 'seeds': seeds,
           'val_macro': vm, 'val_top1': float((Pv.argmax(1).numpy() == yv).mean()),
           'Pv': Pv, 'yv': yv, 'Pt': Pt, 'yt': yt, 'test_names': [nte[i] for i in te_u],
           'val_names': [ntr[i] for i in g_val], 'per_seed': per_seed}
    torch.save(res, os.path.join(C.OUT_DIR, f'temporal_{name}.pt'))
    print(f'[{name}] ENSEMBLE val_macro={vm*100:.1f} val_top1={res["val_top1"]*100:.1f} '
          f'(test held for fusion)', flush=True)
    return res


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--tag', default='b0'); a.add_argument('--kind', default='lstm')
    a.add_argument('--use', default=','.join(STREAMS)); a.add_argument('--seeds', type=int, default=5)
    a.add_argument('--epochs', type=int, default=40); a.add_argument('--lr', type=float, default=1e-3)
    a.add_argument('--d', type=int, default=128); a.add_argument('--p', type=float, default=0.3)
    a.add_argument('--layers', type=int, default=1); a.add_argument('--aux', type=float, default=0.3)
    a.add_argument('--device', default='mps'); a.add_argument('--name', default=None)
    v = a.parse_args()
    run(tag=v.tag, kind=v.kind, use=tuple(v.use.split(',')), seeds=v.seeds, epochs=v.epochs,
        lr=v.lr, d=v.d, p=v.p, layers=v.layers, aux=v.aux, device=v.device, name=v.name)
