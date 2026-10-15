#!/usr/bin/env python3
"""POST-HOC sensitivity sweep around B1, the deployed whole-frame graph. NOT a pre-registered gate.

Why this exists
---------------
The submitted thesis varies architecture (B1/B1w/B1xl/B2/ESTA/TSF, 201k-697k params) but never
varies a TRAINING hyperparameter: lr, weight decay, dropout, label smoothing, grad-clip threshold
and epoch budget each hold exactly one value, which is why chapter 3 labels so many of them
`[mac dinh cua ma]`. This script measures, one factor at a time, how much the headline number
actually depends on those un-swept values.

Rules this script obeys, so it cannot contaminate the frozen record
-------------------------------------------------------------------
  * Writes ONLY under sweep_results/posthoc_sensitivity/. No published artifact path is touched.
  * Reuses run-esta-lso's own fold construction and val split (StratifiedGroupKFold(10, seed=0)
    over 151 sources; test_size=0.15 random_state=0), so every row is paired with the published
    B1 row on the same clips and the same folds.
  * Re-trains from the SAME cached frozen-CLIP features the published run used. No re-extraction,
    so the input side of the experiment is bit-identical.
  * Row `B1_published_config` is the control: it re-runs the published recipe unchanged. It does
    NOT reproduce the published 84.2% BA -- it gives 83.25% at OMP_NUM_THREADS=4 -- and that gap
    is itself a measured result, explained in thesis section 4.5.6: checkpoint selection by a
    saturated val_macro with first-epoch-wins tie-breaking makes the selected epoch sensitive to
    ~1e-7 float-reduction differences from CPU thread count (band 83.25% / 84.17% / 84.71% for
    4 / published / 1 thread). The consequence for this sweep is a comparison rule, not a
    discard rule: every row must run at OMP_NUM_THREADS=4 and be compared to the 83.25% control,
    never to 84.2%. A row compared to 84.2% is comparing two different run environments.
  * NO result here may re-adjudicate a pre-registered gate, and none may replace a chapter-6 row.

Usage: .venv/bin/python run_b1_sensitivity_sweep.py [--only lr_3e-4] [--seeds 3] [--list]
"""
import argparse
import copy
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import esta_training as T
import isafetybench_dataset as ISB
from eval_utils.metrics import macro_accuracy, top1
from model_entity_spatiotemporal import WholeFrameTemporal, count_params
from run_relation_transformer import class_weights, set_seed

OUT = 'sweep_results/posthoc_sensitivity'
FEATS = 'sweep_results/relation_tokens/feats_isbclip_all.pt'

# The published B1 recipe, read off model_entity_spatiotemporal.WholeFrameTemporal defaults and
# esta_training.train_one. Every sweep row below is this dict with ONE key changed.
BASE = dict(d=128, heads=4, ff=256, t_layers=1, dropout=0.1,
            lr=1e-3, wd=0.01, smoothing=0.05, clip=1.0, epochs=60, bs=32)

# One factor at a time. `ff` tracks `d` only where the point is to hold the FFN ratio at 2x.
GRID = {
    'B1_published_config': {},
    'depth_2layer':        dict(t_layers=2),
    'depth_3layer':        dict(t_layers=3),
    'width_d64':           dict(d=64, ff=128),
    'width_d192':          dict(d=192, ff=384),
    'ffn_ratio_1x':        dict(ff=128),
    'ffn_ratio_4x':        dict(ff=512),
    'heads_2':             dict(heads=2),
    'heads_8':             dict(heads=8),
    'dropout_0.0':         dict(dropout=0.0),
    'dropout_0.2':         dict(dropout=0.2),
    'dropout_0.3':         dict(dropout=0.3),
    'lr_3e-4':             dict(lr=3e-4),
    'lr_3e-3':             dict(lr=3e-3),
    'wd_0.0':              dict(wd=0.0),
    'wd_0.1':              dict(wd=0.1),
    'smoothing_0.0':       dict(smoothing=0.0),
    'smoothing_0.1':       dict(smoothing=0.1),
    'gradclip_5.0':        dict(clip=5.0),   # the value the relation-encoder lane uses
    'epochs_30':           dict(epochs=30),
    'epochs_120':          dict(epochs=120),
}


def _load_aligned(path):
    """run-esta-lso.load_aligned, imported by path because the module name carries a hyphen."""
    spec = importlib.util.spec_from_file_location('run_esta_lso', 'run-esta-lso.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_aligned(path)


def _epoch(model, fr, valid, y, opt, lossfn, cfg, train):
    model.train(train)
    n = len(y)
    order = torch.randperm(n) if train else torch.arange(n)
    out = torch.zeros(n, 2)
    for i in range(0, n, cfg['bs']):
        sel = order[i:i + cfg['bs']]
        logits = model(fr[sel], valid[sel])
        if train:
            loss = lossfn(logits, y[sel])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'])
            opt.step()
        out[sel] = logits.detach()
    return out


def train_one(cfg, fr, valid, y, fit, val, seed, in_dim):
    """esta_training.train_one with every constant it hard-codes exposed as a cfg key."""
    set_seed(seed)
    model = WholeFrameTemporal(d=cfg['d'], heads=cfg['heads'], ff=cfg['ff'],
                               t_layers=cfg['t_layers'], dropout=cfg['dropout'], in_dim=in_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    lossfn = nn.CrossEntropyLoss(weight=class_weights(y[fit], k=2),
                                 label_smoothing=cfg['smoothing'])
    best = {'val_macro': -1}
    for _ in range(cfg['epochs']):
        _epoch(model, fr[fit], valid[fit], y[fit], opt, lossfn, cfg, train=True)
        sched.step()
        with torch.no_grad():
            lg = _epoch(model, fr[val], valid[val], y[val], opt, lossfn, cfg, train=False)
        vm = macro_accuracy(y[val].numpy(), lg.argmax(1).numpy()) * 100
        if vm > best['val_macro']:
            best = {'val_macro': vm, 'state': copy.deepcopy(model.state_dict())}
    model.load_state_dict(best['state'])
    return model


def run_config(name, over, feats, y, groups, clip_ids, seeds, folds):
    cfg = dict(BASE, **over)
    fr = feats['wholeframe']
    valid = ((feats['mask'].sum(-1) > 0).float() if 'mask' in feats
             else torch.ones(fr.shape[:2]))
    yn = y.numpy()
    in_dim = fr.shape[-1]
    n_par = count_params(WholeFrameTemporal(d=cfg['d'], heads=cfg['heads'], ff=cfg['ff'],
                                            t_layers=cfg['t_layers'], dropout=cfg['dropout'],
                                            in_dim=in_dim))
    soft = np.zeros((len(yn), 2), dtype=np.float32)
    t0 = time.time()
    for gi, (_, te) in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(yn)), te)
        fit, val = train_test_split(tr, test_size=0.15, stratify=yn[tr], random_state=0)
        acc = torch.zeros(len(te), 2)
        for s in range(seeds):
            m = train_one(cfg, fr, valid, y, fit, val, s, in_dim)
            m.eval()
            with torch.no_grad():
                acc += torch.softmax(m(fr[te], valid[te]), 1)
        soft[te] = (acc / seeds).numpy()
        print(f'    fold {gi + 1:2d}/{len(folds)} n={len(te):3d} '
              f'acc={100 * (soft[te].argmax(1) == yn[te]).mean():5.1f}%', flush=True)
    pred = soft.argmax(1)
    res = {'config': name, 'overrides': over, 'full_config': cfg, 'params': int(n_par),
           'seeds': seeds, 'protocol': 'lso-1100', 'n': int(len(yn)),
           'ba': round(100 * macro_accuracy(yn, pred), 2),
           'top1': round(100 * top1(yn, pred), 2),
           'minutes': round((time.time() - t0) / 60, 1),
           'rows': [{'clip_id': str(clip_ids[i]), 'y_true': int(yn[i]),
                     'prob_unsafe': float(soft[i, 1]), 'y_pred': int(pred[i])}
                    for i in range(len(yn))]}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f'{OUT}/{name}_oof.json', 'w'))
    print(f'  [{name}] params={n_par:,} BA={res["ba"]:.2f}% top1={res["top1"]:.2f}% '
          f'({res["minutes"]} min)', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', help='run just these config names')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--feats', default=FEATS)
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    if args.list:
        for k, v in GRID.items():
            print(f'{k:24s} {v}')
        return

    feats, y, groups, clip_ids = _load_aligned(args.feats)
    folds = [(f'f{i + 1}', te) for i, (_, te)
             in enumerate(ISB.folds(y, groups, n_splits=10, seed=0))]
    names = args.only or list(GRID)
    print(f'[posthoc sensitivity] {len(names)} configs, {len(folds)} LSO folds, '
          f'{args.seeds} seeds, {len(y)} clips. NOT a pre-registered gate.', flush=True)
    for i, name in enumerate(names, 1):
        print(f'\n=== [{i}/{len(names)}] {name}  {GRID[name]}', flush=True)
        run_config(name, GRID[name], feats, y, groups, clip_ids, args.seeds, folds)


if __name__ == '__main__':
    main()
