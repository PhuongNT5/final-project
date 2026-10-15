#!/usr/bin/env python3
"""POST-HOC round 2: alternative temporal models, checkpoint-selection rules, ensemble size.

NOT a pre-registered gate. Nothing here may replace a chapter-6 row or re-adjudicate G10-G18.

Why this exists
---------------
Round 1 (run_b1_sensitivity_sweep.py) varies one TRAINING hyperparameter at a time inside the
published B1 recipe. It cannot answer three questions the thesis's own weakness sections raise:

  (1) Section 4.5.6 names the checkpoint rule -- first epoch to reach a SATURATED val_macro -- as a
      measured defect worth 1.46 pp of run-to-run drift, states the three textbook fixes, and
      declines to apply them because that would create a second number set after the freeze. It
      never measures whether any fix actually helps. Families `sel_*` measure exactly that.
  (2) The thesis varies architecture only inside one family: a learned-CLS Transformer over T frame
      states (B1/B1w/B1xl/B2/ESTA/TSF). So "a temporal encoder beats a counting rule" (H1) is
      tested against a counting rule, never against a DIFFERENT temporal encoder, and never
      against no temporal model at all. Families `arch_*` add a masked-mean floor (no temporal
      model), a GRU, a BiGRU and a dilated TCN on the same frozen features.
  (3) Published rows average 3 seeds. How much of the number is the ensemble, not the model, is
      unmeasured. Families `seeds_*` vary only the ensemble size.

Rules this script obeys
-----------------------
  * Writes ONLY under sweep_results/posthoc_sensitivity/round2/.
  * Same cached frozen-CLIP features, same StratifiedGroupKFold(10, seed=0) over 151 sources, same
    within-fold val split (test_size=0.15, random_state=0) as the published run. Every row is
    paired with the published B1 row on the same clips and folds.
  * MUST run at OMP_NUM_THREADS=4 and be read against the 83.25% control of round 1, never against
    the published 84.2% -- see section 4.5.6 for why those are two different run environments.
  * Alternative models are defined HERE, not in model_entity_spatiotemporal.py, so the module that
    produced the frozen artifacts stays byte-identical.
  * Saves the within-fold VALIDATION probabilities as well as the test-fold ones, so a decision
    threshold can be chosen on validation only and never on the held-out fold.

Usage: OMP_NUM_THREADS=4 .venv/bin/python run_b1_round2_experiments.py [--only sel_valloss] [--list]
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import isafetybench_dataset as ISB
from eval_utils.metrics import macro_accuracy, top1
from model_entity_spatiotemporal import WholeFrameTemporal, count_params
from run_b1_sensitivity_sweep import BASE, _epoch, _load_aligned
from run_relation_transformer import class_weights, set_seed

OUT = 'sweep_results/posthoc_sensitivity/round2'
TAG = ''   # set by --tag; prefixes every output name so a tower swap cannot overwrite a head sweep


def _masked_mean(x, valid):
    w = valid.unsqueeze(-1)
    return (x * w).sum(1) / w.sum(1).clamp(min=1.0)


class MeanPoolMLP(nn.Module):
    """No temporal model at all: masked mean over frames, then an MLP. The honest floor for H1."""

    def __init__(self, d=128, ff=256, dropout=0.1, n_cls=2, in_dim=512, **_):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d), nn.Dropout(dropout))
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ff, d), nn.LayerNorm(d))
        self.head = nn.Linear(d, n_cls)

    def forward(self, frames, valid):
        return self.head(self.mlp(_masked_mean(self.proj(frames), valid)))


class GRUTemporal(nn.Module):
    """Recurrent temporal model. Order-sensitive like attention, but no global receptive field."""

    def __init__(self, d=128, dropout=0.1, n_cls=2, in_dim=512, bidir=False, **_):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d), nn.Dropout(dropout))
        self.rnn = nn.GRU(d, d, batch_first=True, bidirectional=bidir)
        self.norm = nn.LayerNorm(d * (2 if bidir else 1))
        self.head = nn.Linear(d * (2 if bidir else 1), n_cls)

    def forward(self, frames, valid):
        h, _ = self.rnn(self.proj(frames))
        return self.head(self.norm(_masked_mean(h, valid)))


class TCNTemporal(nn.Module):
    """Dilated temporal convolution: local-then-wider receptive field instead of all-pairs."""

    def __init__(self, d=128, dropout=0.1, n_cls=2, in_dim=512, **_):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d), nn.Dropout(dropout))
        self.c1 = nn.Conv1d(d, d, 3, padding=1, dilation=1)
        self.c2 = nn.Conv1d(d, d, 3, padding=2, dilation=2)
        self.act, self.drop, self.norm = nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d)
        self.head = nn.Linear(d, n_cls)

    def forward(self, frames, valid):
        x = self.proj(frames).transpose(1, 2)
        x = self.drop(self.act(self.c1(x)))
        x = self.drop(self.act(self.c2(x))).transpose(1, 2)
        return self.head(self.norm(_masked_mean(x, valid)))


ARCH = {'transformer': WholeFrameTemporal, 'meanpool': MeanPoolMLP,
        'gru': GRUTemporal, 'bigru': lambda **k: GRUTemporal(bidir=True, **k), 'tcn': TCNTemporal}

# One factor at a time, against BASE (the published B1 recipe) with arch=transformer, sel=first_max.
#   sel: first_max  = published rule, first epoch reaching max val_macro  (`>`)
#        last_max   = same metric, ties broken toward the LAST epoch      (`>=`)
#        val_loss   = lowest validation loss, a metric that does not saturate
#        swa10      = no selection; average the weights of the final 10 epochs
#        top3_ens   = average the softmax of the 3 best-val_macro epochs
GRID = {
    'control_round2':  {},
    'sel_last_max':    dict(sel='last_max'),
    'sel_val_loss':    dict(sel='val_loss'),
    'sel_swa10':       dict(sel='swa10'),
    'sel_top3_ens':    dict(sel='top3_ens'),
    'arch_meanpool':   dict(arch='meanpool'),
    'arch_gru':        dict(arch='gru'),
    'arch_bigru':      dict(arch='bigru'),
    'arch_tcn':        dict(arch='tcn'),
    'seeds_1':         dict(seeds=1),
    'seeds_5':         dict(seeds=5),
    'seeds_10':        dict(seeds=10),
}


def _build(cfg, in_dim):
    kw = dict(d=cfg['d'], dropout=cfg['dropout'], in_dim=in_dim)
    if cfg['arch'] == 'transformer':
        kw.update(heads=cfg['heads'], ff=cfg['ff'], t_layers=cfg['t_layers'])
    elif cfg['arch'] == 'meanpool':
        kw.update(ff=cfg['ff'])
    return ARCH[cfg['arch']](**kw)


def _predict(states, cfg, in_dim, fr, valid):
    """Average softmax over one or more checkpoints of the same recipe."""
    acc = torch.zeros(len(fr), 2)
    for st in states:
        m = _build(cfg, in_dim)
        m.load_state_dict(st)
        m.eval()
        with torch.no_grad():
            acc += torch.softmax(m(fr, valid), 1)
    return acc / len(states)


def train_one(cfg, fr, valid, y, fit, val, seed, in_dim):
    """Returns the checkpoint set the configured selection rule asks for."""
    set_seed(seed)
    model = _build(cfg, in_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    lossfn = nn.CrossEntropyLoss(weight=class_weights(y[fit], k=2),
                                 label_smoothing=cfg['smoothing'])
    hist, tail = [], []
    for ep in range(cfg['epochs']):
        _epoch(model, fr[fit], valid[fit], y[fit], opt, lossfn, cfg, train=True)
        sched.step()
        with torch.no_grad():
            lg = _epoch(model, fr[val], valid[val], y[val], opt, lossfn, cfg, train=False)
            vloss = float(lossfn(lg, y[val]))
        vm = macro_accuracy(y[val].numpy(), lg.argmax(1).numpy()) * 100
        state = copy.deepcopy(model.state_dict())
        hist.append({'epoch': ep, 'val_macro': vm, 'val_loss': vloss, 'state': state})
        tail.append(state)
        if len(tail) > 10:
            tail.pop(0)
    if cfg['sel'] == 'swa10':
        avg = {k: torch.stack([s[k].float() for s in tail]).mean(0) for k in tail[0]}
        return [avg], [h['epoch'] for h in hist[-10:]]
    if cfg['sel'] == 'val_loss':
        best = min(hist, key=lambda h: h['val_loss'])
        return [best['state']], [best['epoch']]
    if cfg['sel'] == 'top3_ens':
        top = sorted(hist, key=lambda h: (-h['val_macro'], h['epoch']))[:3]
        return [h['state'] for h in top], [h['epoch'] for h in top]
    ties_to_last = cfg['sel'] == 'last_max'
    best = hist[0]
    for h in hist[1:]:        # the only difference from the published rule is `>` versus `>=`
        if (h['val_macro'] >= best['val_macro'] if ties_to_last
                else h['val_macro'] > best['val_macro']):
            best = h
    return [best['state']], [best['epoch']]


def run_config(name, over, feats, y, clip_ids, folds, default_seeds, feats_path=''):
    # `dict(BASE, arch=..., **over)` raises TypeError the moment `over` itself sets `arch` or
    # `sel`, which is every arch_* and sel_* row. Set the defaults first, then let `over` win.
    cfg = dict(BASE, arch='transformer', sel='first_max')
    cfg.update(over)
    seeds = cfg.pop('seeds', default_seeds)
    fr = feats['wholeframe']
    valid = ((feats['mask'].sum(-1) > 0).float() if 'mask' in feats
             else torch.ones(fr.shape[:2]))
    yn, in_dim = y.numpy(), fr.shape[-1]
    n_par = count_params(_build(cfg, in_dim))
    soft = np.zeros((len(yn), 2), dtype=np.float32)
    val_rows, epochs_used, t0 = [], [], time.time()
    for gi, (_, te) in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(yn)), te)
        fit, val = train_test_split(tr, test_size=0.15, stratify=yn[tr], random_state=0)
        acc, vacc = torch.zeros(len(te), 2), torch.zeros(len(val), 2)
        for s in range(seeds):
            states, eps = train_one(cfg, fr, valid, y, fit, val, s, in_dim)
            epochs_used.append(eps)
            acc += _predict(states, cfg, in_dim, fr[te], valid[te])
            vacc += _predict(states, cfg, in_dim, fr[val], valid[val])
        soft[te] = (acc / seeds).numpy()
        vp = (vacc / seeds).numpy()
        val_rows += [{'fold': gi + 1, 'y_true': int(yn[val[j]]),
                      'prob_unsafe': float(vp[j, 1])} for j in range(len(val))]
        print(f'    fold {gi + 1:2d}/{len(folds)} n={len(te):3d} '
              f'acc={100 * (soft[te].argmax(1) == yn[te]).mean():5.1f}%', flush=True)
    pred = soft.argmax(1)
    res = {'config': TAG + name, 'overrides': over, 'full_config': cfg, 'params': int(n_par),
           'feats': feats_path, 'in_dim': int(in_dim), 'frames_T': int(fr.shape[1]),
           'seeds': seeds, 'protocol': 'lso-1100', 'n': int(len(yn)),
           'ba': round(100 * macro_accuracy(yn, pred), 2),
           'top1': round(100 * top1(yn, pred), 2),
           'minutes': round((time.time() - t0) / 60, 1),
           'selected_epochs': epochs_used,
           'rows': [{'clip_id': str(clip_ids[i]), 'y_true': int(yn[i]),
                     'prob_unsafe': float(soft[i, 1]), 'y_pred': int(pred[i])}
                    for i in range(len(yn))],
           'val_rows': val_rows}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f'{OUT}/{TAG}{name}_oof.json', 'w'))
    print(f'  [{TAG}{name}] in_dim={in_dim} T={fr.shape[1]} arch={cfg["arch"]} '
          f'sel={cfg["sel"]} seeds={seeds} params={n_par:,} '
          f'BA={res["ba"]:.2f}% top1={res["top1"]:.2f}% ({res["minutes"]} min)', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--feats', default='sweep_results/relation_tokens/feats_isbclip_all.pt')
    ap.add_argument('--tag', default='', help='output-name prefix, e.g. towerL_')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    global TAG
    TAG = args.tag
    if args.list:
        for k, v in GRID.items():
            print(f'{k:20s} {v}')
        return
    if os.environ.get('OMP_NUM_THREADS') != '4':
        raise SystemExit('refusing to run: set OMP_NUM_THREADS=4 (thesis section 4.5.6)')
    feats, y, groups, clip_ids = _load_aligned(args.feats)
    folds = [(f'f{i + 1}', te) for i, (_, te)
             in enumerate(ISB.folds(y, groups, n_splits=10, seed=0))]
    names = args.only or list(GRID)
    print(f'[posthoc round 2] {len(names)} configs, {len(folds)} LSO folds, {len(y)} clips. '
          f'NOT a pre-registered gate.', flush=True)
    for i, name in enumerate(names, 1):
        print(f'\n=== [{i}/{len(names)}] {name}  {GRID[name]}', flush=True)
        run_config(name, GRID[name], feats, y, clip_ids, folds, args.seeds, args.feats)


if __name__ == '__main__':
    main()
