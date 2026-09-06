"""Train the Stage-B per-frame behaviour recogniser under the blind protocol.

Selection signal = clip-level 4-class macro accuracy on a val slice carved from TRAIN clips
(mean-softmax over the clip's frames).  The 64-clip test set is never touched here; the script
only exports per-frame logits/embeddings for every clip, which later stages consume.
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as TT

from . import config as C
from . import frame_model as FM
from .data_index import load_clips, blind_split
from .frame_cache import frame_paths

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SIZE = (224, 400)          # H, W  - keeps the 16:9 aspect, ~1.8x the cost of 224x224


def transforms(train: bool, hflip: bool = False):
    ops = [TT.Resize((252, 448), antialias=True)]
    if train:
        ops += [TT.RandomCrop(SIZE), TT.ColorJitter(0.25, 0.25, 0.15, 0.0)]   # hue=0: keep vest colour
        if hflip:
            ops.append(TT.RandomHorizontalFlip(0.5))
    else:
        ops.append(TT.CenterCrop(SIZE))
    ops += [TT.ToImage(), TT.ToDtype(torch.float32, scale=True), TT.Normalize(MEAN, STD)]
    if train:
        ops.append(TT.RandomErasing(0.25, scale=(0.02, 0.12)))
    return TT.Compose(ops)


class FrameDS(Dataset):
    """(frame image, clip label) pairs - the clip label is propagated to each sampled frame."""

    def __init__(self, clips, stride: int, train: bool, hflip: bool = False):
        self.items = [(p, c.label) for c in clips for p in frame_paths(c)[::stride]]
        self.tf = transforms(train, hflip)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        p, y = self.items[i]
        return self.tf(Image.open(p).convert('RGB')), y


class ClipFrameDS(Dataset):
    """Every frame of every clip, in order - for clip-level inference/export."""

    def __init__(self, clips, stride: int = 1):
        self.groups = [frame_paths(c)[::stride] for c in clips]
        self.items = [(gi, p) for gi, g in enumerate(self.groups) for p in g]
        self.tf = transforms(False)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        gi, p = self.items[i]
        return self.tf(Image.open(p).convert('RGB')), gi


@torch.no_grad()
def clip_logits(model, clips, device, stride=1, bs=64, workers=4):
    """-> (n_clips, n_frames, n_cls) logits and (n_clips, n_frames, dim) embeddings."""
    ds = ClipFrameDS(clips, stride)
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, pin_memory=False)
    n_f = len(ds.groups[0])
    L = torch.zeros(len(clips), n_f, model.n_cls)
    F = torch.zeros(len(clips), n_f, model.dim)
    fill = np.zeros(len(clips), dtype=int)
    model.eval()
    for x, gi in dl:
        lo, ft = model(x.to(device), return_feat=True)
        for j, g in enumerate(gi.tolist()):
            L[g, fill[g]] = lo[j].float().cpu()
            F[g, fill[g]] = ft[j].float().cpu()
            fill[g] += 1
    return L, F


def macro4(y, pred):
    return float(np.mean([np.mean(pred[y == k] == k) for k in range(4) if (y == k).any()]))


def run(backbone='resnet34', epochs=14, lr=3e-4, bs=32, stride=2, seed=0,
        device='mps', workers=4, hflip=False, tag='b0'):
    torch.manual_seed(seed); np.random.seed(seed)
    train_all = load_clips(['train'])
    unsafe_tr = [c for c in train_all if c.is_unsafe]
    fit_i, val_i = blind_split(unsafe_tr, seed=seed)
    val_clips = [unsafe_tr[i] for i in val_i]
    val_names = {c.name for c in val_clips}
    fit_clips = [c for c in train_all if c.name not in val_names]     # incl. all 4 safe classes

    model = FM.build(backbone, 8, device)
    dl = DataLoader(FrameDS(fit_clips, stride, True, hflip), batch_size=bs, shuffle=True,
                    num_workers=workers, drop_last=True, persistent_workers=workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * len(dl), pct_start=0.25)
    yv = np.array([c.label for c in val_clips])
    best = {'macro': -1.0}
    print(f'[frame] {backbone} fit={len(fit_clips)}clips/{len(dl.dataset)}frames val={len(val_clips)}clips', flush=True)
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in dl:
            loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss)
        L, _ = clip_logits(model, val_clips, device, stride=4, workers=workers)
        pred = torch.softmax(L[..., :4], -1).mean(1).argmax(1).numpy()
        m = macro4(yv, pred); acc = float((pred == yv).mean())
        print(f'  ep{ep:02d} loss={tot/len(dl):.3f} val_top1={acc*100:.1f} val_macro={m*100:.1f} ({time.time()-t0:.0f}s)', flush=True)
        if m > best['macro']:
            best = {'macro': m, 'top1': acc, 'ep': ep,
                    'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best['state'])
    ck = os.path.join(C.OUT_DIR, f'frame_{tag}.pt')
    torch.save({'state': best['state'], 'backbone': backbone, 'val_macro': best['macro'],
                'val_top1': best['top1'], 'epoch': best['ep'], 'seed': seed, 'val_names': sorted(val_names)}, ck)
    print(f'[frame] best ep{best["ep"]} val_macro={best["macro"]*100:.1f} -> {ck}', flush=True)
    export(model, device, tag, workers)
    return ck


@torch.no_grad()
def export(model, device, tag, workers=4):
    """Cache per-frame logits + embeddings for every clip (train & test)."""
    for sp in ('train', 'test'):
        clips = load_clips([sp])
        L, F = clip_logits(model, clips, device, stride=1, workers=workers)
        torch.save({'names': [c.name for c in clips], 'labels': [c.label for c in clips],
                    'views': [c.view for c in clips], 'logits': L.half(), 'feats': F.half()},
                   os.path.join(C.OUT_DIR, f'frameout_{tag}_{sp}.pt'))
        print(f'  exported {sp}: {tuple(L.shape)}', flush=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--backbone', default='resnet34'); a.add_argument('--epochs', type=int, default=14)
    a.add_argument('--lr', type=float, default=3e-4); a.add_argument('--bs', type=int, default=32)
    a.add_argument('--stride', type=int, default=2); a.add_argument('--seed', type=int, default=0)
    a.add_argument('--device', default='mps'); a.add_argument('--workers', type=int, default=4)
    a.add_argument('--hflip', action='store_true'); a.add_argument('--tag', default='b0')
    v = a.parse_args()
    run(**vars(v))
