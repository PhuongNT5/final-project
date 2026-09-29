#!/usr/bin/env python3
"""Whole-frame falsification baseline on iSafetyBench — no detector, no entity tokens.

If classifying frozen whole-frame features reaches the entity model's accuracy, the detector +
relation head buy nothing and the thesis cannot claim a relational result. This is the strongest
adversarial baseline available for this corpus, so it is run BEFORE any claim is written, on the
identical clips and the identical leave-source-out folds.

Three frozen-feature probes are reported and the BEST is the baseline the encoder must beat --
choosing the weakest one would be grading our own homework.

Saves per-clip OOF in run_kfold_relation.py's row schema so tools/compare-oof-runs.py can pair it
against any encoder run.

Usage: .venv/bin/python run_wholeframe_baseline.py
"""
import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import isafetybench_dataset as ISB

TOK = ISB.TOK


def load(tag='isb_'):
    d = torch.load(f'{TOK}/tokens_wf_{tag}all.pt', weights_only=False)
    F = d['feats'].float().numpy()                       # (n, T, 64)
    y = d['labels'].long()
    groups = np.array([str(v) for v in d['views']])
    clip_ids = np.array([str(c) for c in d['clip_ids']])
    return F, y, groups, clip_ids


def pool(F, kind):
    """Clip-level descriptor from the per-frame features. 'meanstd' keeps a crude motion/variability
    signal that a plain temporal mean throws away, so the baseline is not handicapped on the one
    axis where a video model would otherwise be trivially better."""
    if kind == 'mean':
        return F.mean(1)
    if kind == 'meanstd':
        return np.concatenate([F.mean(1), F.std(1)], axis=1)
    raise ValueError(kind)


def probe(kind):
    if kind == 'linear':
        return lambda: LogisticRegression(max_iter=4000, class_weight='balanced')
    if kind == 'mlp':
        return lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=800, random_state=0,
                                     early_stopping=True)
    raise ValueError(kind)


def oof(Z, y, splits, make_clf):
    yn = y.numpy() if torch.is_tensor(y) else np.asarray(y)
    pred = np.full(len(yn), -1)
    prob = np.zeros(len(yn), dtype=np.float32)
    for tr, te in splits:
        sc = StandardScaler().fit(Z[tr])
        clf = make_clf().fit(sc.transform(Z[tr]), yn[tr])
        p = clf.predict_proba(sc.transform(Z[te]))[:, 1]
        prob[te] = p
        pred[te] = (p >= 0.5).astype(int)
    return pred, prob


def report(name, yn, pred, prob):
    ba = float(np.mean([(pred[yn == c] == c).mean() for c in (0, 1)])) * 100
    mf1 = f1_score(yn, pred, average='macro') * 100
    auc = roc_auc_score(yn, prob)
    print(f"  {name:22s} balanced_acc={ba:5.1f}%  macro_f1={mf1:5.1f}%  ROC-AUC={auc:.3f}")
    return ba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='isb_')
    ap.add_argument('--folds', type=int, default=10)
    args = ap.parse_args()

    F, y, groups, clip_ids = load(args.tag)
    yn = y.numpy()
    splits = ISB.folds(y, groups, n_splits=args.folds, seed=0)
    print(f"[wholeframe baseline] {len(yn)} clips / {len(set(groups))} sources / "
          f"{args.folds} leave-source-out folds | features {F.shape[1]}x{F.shape[2]} "
          f"(whole frame, no detector)")

    best = None
    for pk in ('mean', 'meanstd'):
        Z = pool(F, pk)
        for ck in ('linear', 'mlp'):
            if pk == 'mean' and ck == 'mlp':
                continue                                   # meanstd strictly dominates as MLP input
            pred, prob = oof(Z, y, splits, probe(ck))
            ba = report(f'{ck}-{pk}', yn, pred, prob)
            if best is None or ba > best[0]:
                best = (ba, f'{ck}-{pk}', pred, prob)

    ba, name, pred, prob = best
    print(f"\n  BEST whole-frame probe: {name} at {ba:.1f}% balanced accuracy")
    rows = [{'clip_id': str(clip_ids[i]), 'y_true': int(yn[i]), 'prob_unsafe': float(prob[i]),
             'y_pred': int(pred[i]), 'fold_id': str(groups[i])} for i in range(len(yn))]
    path = f'{TOK}/wholeframe_{args.tag}lso-{len(yn)}_oof.json'
    json.dump({'model': f'wholeframe-{name}', 'protocol': f'lso-{len(yn)}', 'dedupe_policy': 'none',
               'tag': args.tag, 'n': len(rows), 'rows': rows}, open(path, 'w'))
    print(f"  [saved OOF] {path}")


if __name__ == '__main__':
    main()
