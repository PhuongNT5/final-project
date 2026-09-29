"""Training loop for the ESTA family (R48 / gates G14-G16).

Deliberately the SAME recipe as run_relation_transformer.train_one: AdamW lr 1e-3, weight decay
0.01, cosine schedule over the run, inverse-frequency class weights, label smoothing 0.05, and a
best-val-macro checkpoint. Optimisation is therefore not a free variable between ESTA, its
whole-frame baselines and the 64-d relation encoder they are compared against.

Training runs on CPU: the macOS 14 MPS addcmul_ bug corrupts Adam updates (see reports/).
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from eval_utils.metrics import macro_accuracy
from model_entity_spatiotemporal import EntitySpatioTemporal, WholeFrameTemporal, count_params
from run_relation_transformer import class_weights, set_seed

DEVICE = 'cpu'
BS = 32
EPOCHS = 60


def build(name, in_dim=None):
    """ESTA and its baselines. B1w is B1 with a second temporal layer: capacity-matched to ESTA so
    a win cannot be read as 'you simply gave the entity model more parameters' (pre-registered
    amendment 2026-09-29, before any of these models had been trained)."""
    if name == 'ESTA':
        return EntitySpatioTemporal()
    kw = {'in_dim': in_dim} if in_dim else {}
    if name == 'B1':
        return WholeFrameTemporal(t_layers=1, **kw)
    if name == 'B1w':
        return WholeFrameTemporal(t_layers=2, **kw)
    if name == 'B2':
        # G16's honest strong opponent: same head, video-pretrained backbone (VideoMAE-base, 768-d
        # per tubelet state). If this beats ESTA the entity pipeline is not justified on accuracy.
        return WholeFrameTemporal(t_layers=1, in_dim=in_dim or 768)
    raise ValueError(f'unknown model {name}')


def parts_for(name, feats, idx):
    """Model inputs for a set of clip rows. ESTA sees entities; the baselines see frame vectors."""
    if name == 'ESTA':
        return (feats['entity'][idx], feats['geom'][idx], feats['mask'][idx])
    fr = feats['wholeframe'][idx]
    # B1/B1w reuse ESTA's own frame-validity so the two differ only by the entity axis. A backbone
    # with no detector pass (B2) has no such mask: every state it emits is valid by construction.
    valid = ((feats['mask'][idx].sum(-1) > 0).float() if 'mask' in feats
             else torch.ones(fr.shape[:2]))
    return (fr, valid)


def _forward(model, parts):
    return model(*[p.to(DEVICE) for p in parts])


def _epoch(model, parts, y, opt, lossfn, train):
    model.train(train)
    n = len(y)
    order = torch.randperm(n) if train else torch.arange(n)
    out = torch.zeros(n, 2)
    for i in range(0, n, BS):
        sel = order[i:i + BS]
        logits = _forward(model, [p[sel] for p in parts])
        if train:
            loss = lossfn(logits, y[sel].to(DEVICE))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        out[sel] = logits.detach().cpu()
    return out


def train_one(name, feats, fit, val, y, seed, epochs=EPOCHS, in_dim=None):
    """Train one model on `fit`, select the epoch by macro accuracy on `val`. Returns (model, best)."""
    set_seed(seed)
    model = build(name, in_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossfn = nn.CrossEntropyLoss(weight=class_weights(y[fit], k=2), label_smoothing=0.05)
    pf, pv = parts_for(name, feats, fit), parts_for(name, feats, val)
    yf, yv = y[fit], y[val]
    best = {'val_macro': -1}
    for ep in range(epochs):
        _epoch(model, pf, yf, opt, lossfn, train=True)
        sched.step()
        with torch.no_grad():
            logits = _epoch(model, pv, yv, opt, lossfn, train=False)
        vm = macro_accuracy(yv.numpy(), logits.argmax(1).numpy()) * 100
        if vm > best['val_macro']:
            best = {'val_macro': vm, 'epoch': ep, 'state': copy.deepcopy(model.state_dict())}
    model.load_state_dict(best['state'])
    return model, best


@torch.no_grad()
def predict_softmax(model, name, feats, idx):
    model.eval()
    parts = parts_for(name, feats, idx)
    out = [_forward(model, [p[i:i + BS] for p in parts]) for i in range(0, len(idx), BS)]
    return torch.softmax(torch.cat(out).cpu(), 1)


def describe(name, in_dim=None):
    return f'{name} params={count_params(build(name, in_dim)):,}'
