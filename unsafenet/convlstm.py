"""The paper's actual temporal model: a ConvLSTM over raw frames.

Every temporal head we built so far (mean / BiLSTM / Transformer over entity-geometry and CNN
streams) departs from Onal & Dandil, who classify raw frame sequences with a ConvLSTM.  Without this
module we cannot claim to have reproduced their *baseline* (Table 5: 57/64 = 89.06 % top-1, 85.78 %
macro without frame reduction), which is the anchor every other comparison hangs on.

Members are written in the same format as the rest of the pipeline so fusion/eval consume them
unchanged.
"""
from __future__ import annotations
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .data_index import load_clips, blind_split
from .frame_cache import frame_paths

CACHE = os.path.join(C.CACHE_DIR, 'raw_{}_{}px_t{}.npy')


def clip_tensor(subsets=('train', 'test'), px=112, t=16):
    """Cache uint8 (N,T,px,px,3) raw frames - the ConvLSTM's input."""
    for sp in subsets:
        out = CACHE.format(sp, px, t)
        if os.path.exists(out):
            continue
        clips = load_clips([sp])
        X = np.zeros((len(clips), t, px, px, 3), np.uint8)
        idx = np.linspace(0, C.T - 1, t).round().astype(int)
        for i, c in enumerate(clips):
            paths = frame_paths(c)
            for j, k in enumerate(idx):
                im = cv2.imread(paths[k])
                X[i, j] = cv2.resize(im, (px, px), interpolation=cv2.INTER_AREA)[:, :, ::-1]
            if (i + 1) % 150 == 0:
                print(f'  [{sp}] {i+1}/{len(clips)}', flush=True)
        np.save(out, X)
        print(f'[convlstm] cached {sp}: {X.shape}', flush=True)


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, ch, k=3):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(cin + ch, 4 * ch, k, padding=k // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class ConvLSTM(nn.Module):
    """Paper-shaped: strided conv stem -> stacked ConvLSTM -> global pool -> linear."""

    def __init__(self, n_cls=4, chans=(32, 64), width=64, drop=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width // 2, 5, 2, 2), nn.BatchNorm2d(width // 2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width // 2, width, 3, 1, 1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.MaxPool2d(2))
        cells, cin = [], width
        for ch in chans:
            cells.append(ConvLSTMCell(cin, ch)); cin = ch
        self.cells = nn.ModuleList(cells)
        self.norm = nn.LayerNorm(cin)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(cin, n_cls))

    def forward(self, x):                                  # x (B,T,3,H,W)
        B, T = x.shape[:2]
        f = self.stem(x.flatten(0, 1))
        f = f.view(B, T, *f.shape[1:])
        for cell in self.cells:
            h = torch.zeros(B, cell.ch, *f.shape[-2:], device=f.device)
            c = torch.zeros_like(h)
            outs = []
            for t in range(T):
                h, c = cell(f[:, t], (h, c))
                outs.append(h)
            f = torch.stack(outs, 1)
        v = f.mean((-2, -1)).mean(1)                       # global spatial then temporal pool
        return self.head(self.norm(v))


def _macro(y, p, n=4):
    return float(np.mean([np.mean(p[y == k] == k) for k in range(n) if (y == k).any()]))


def _batches(X, y, bs, train, gen=None):
    idx = torch.randperm(len(X), generator=gen) if train else torch.arange(len(X))
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        if train and len(j) < 2:                          # BatchNorm needs >1 sample
            continue
        xb = X[j].float().div_(255.)
        if train and torch.rand(1, generator=gen).item() < 0.5:     # temporal jitter only;
            xb = xb + torch.randn(xb.shape, generator=gen) * 0.02   # no flip - walkway side matters
        yield xb.permute(0, 1, 4, 2, 3), y[j.numpy()]   # .numpy(): a 1-elem tensor indexes as a scalar


def _run_seed(Xtr, ytr, Xva, yva, Xte, n_cls, seed, epochs, bs, lr, device):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = ConvLSTM(n_cls=n_cls).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * max(1, len(Xtr) // bs + 1))
    w = torch.tensor([1.0 / max(1, (ytr == k).sum()) for k in range(n_cls)], dtype=torch.float)
    lossf = nn.CrossEntropyLoss(weight=(w / w.mean()).to(device), label_smoothing=0.05)
    best, best_state = -1.0, None
    for ep in range(epochs):
        net.train()
        for xb, yb in _batches(Xtr, ytr, bs, True, gen):
            opt.zero_grad()
            loss = lossf(net(xb.to(device)), torch.as_tensor(yb).to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(xb.to(device)).cpu() for xb, _ in _batches(Xva, yva, bs, False)])
        m = _macro(yva, pv.argmax(1).numpy(), n_cls)
        print(f'    seed{seed} ep{ep+1:02d} val_macro={m*100:.1f}', flush=True)
        if m > best:
            best, best_state = m, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        Pv = torch.cat([net(xb.to(device)).cpu() for xb, _ in _batches(Xva, yva, bs, False)])
        Pt = torch.cat([net(xb.to(device)).cpu() for xb, _ in _batches(Xte, np.zeros(len(Xte), int), bs, False)])
    return best, torch.softmax(Pv, -1), torch.softmax(Pt, -1)


def run(px=112, t=16, seeds=5, epochs=18, bs=8, lr=3e-4, device='mps', n_cls=4, name='convlstm'):
    clip_tensor(px=px, t=t)
    Xtr_all = torch.from_numpy(np.load(CACHE.format('train', px, t)))
    Xte_all = torch.from_numpy(np.load(CACHE.format('test', px, t)))
    tr, te = load_clips(['train']), load_clips(['test'])
    ytr_all = np.array([c.label for c in tr]); yte_all = np.array([c.label for c in te])
    keep_te = np.where(yte_all < 4)[0]
    unsafe = np.where(ytr_all < 4)[0]
    fit_i, val_i = blind_split([c for c in tr if c.is_unsafe], seed=0)
    g_val = unsafe[val_i]                                 # val is always unsafe-only (the task)
    g_fit = (np.array(sorted(set(range(len(tr))) - set(g_val.tolist())))
             if n_cls == 8 else unsafe[fit_i])            # safe clips only ever join the fit split
    Xf, yf = Xtr_all[g_fit], ytr_all[g_fit]
    Xv, yv = Xtr_all[g_val], ytr_all[g_val]
    Xt, yt = Xte_all[keep_te], yte_all[keep_te]
    print(f'[convlstm] fit={len(Xf)} val={len(Xv)} test={len(Xt)} n_cls={n_cls}', flush=True)
    Pvs, Pts, per = [], [], []
    for s in range(seeds):
        m, Pv, Pt = _run_seed(Xf, yf, Xv, yv, Xt, n_cls, s, epochs, bs, lr, device)
        per.append(m); Pvs.append(Pv); Pts.append(Pt)
        print(f'  [{name}] seed{s} val_macro={m*100:.1f}', flush=True)
    Pv, Pt = torch.stack(Pvs).mean(0)[:, :4], torch.stack(Pts).mean(0)[:, :4]
    res = {'name': name, 'kind': 'convlstm', 'use': ['raw'], 'seeds': seeds,
           'val_macro': _macro(yv, Pv.argmax(1).numpy()), 'val_top1': float((Pv.argmax(1).numpy() == yv).mean()),
           'Pv': Pv, 'yv': yv, 'Pt': Pt, 'yt': yt,
           'test_names': [te[i].name for i in keep_te], 'val_names': [tr[i].name for i in g_val],
           'per_seed': per}
    torch.save(res, os.path.join(C.OUT_DIR, f'temporal_{name}.pt'))
    print(f'[{name}] val macro={res["val_macro"]*100:.1f} top1={res["val_top1"]*100:.1f} '
          f'seed_macros={[round(x*100,1) for x in per]}', flush=True)
    return res


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--px', type=int, default=112); a.add_argument('--t', type=int, default=16)
    a.add_argument('--seeds', type=int, default=5); a.add_argument('--epochs', type=int, default=18)
    a.add_argument('--bs', type=int, default=8); a.add_argument('--lr', type=float, default=3e-4)
    a.add_argument('--device', default='mps'); a.add_argument('--n-cls', type=int, default=4)
    a.add_argument('--name', default='convlstm')
    v = a.parse_args()
    run(px=v.px, t=v.t, seeds=v.seeds, epochs=v.epochs, bs=v.bs, lr=v.lr,
        device=v.device, n_cls=v.n_cls, name=v.name)
