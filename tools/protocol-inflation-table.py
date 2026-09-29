"""G12/G13 — recompute the 2x2 protocol x capacity table from saved OOF tensors.

Two corpora (iSafetyBench, Unsafe-Net) x two fold-construction protocols (identity-respecting vs
identity-leaking) x a capacity ladder. Reads only the per-clip out-of-fold softmax tensors that the
training runs already wrote, so this file re-derives published numbers and never re-trains.

Balanced accuracy is recomputed here from `labels` + `oof_pred` rather than parsed from the logs, so
a mismatch against a log line is a real disagreement, not a formatting one.
"""
import sys
from pathlib import Path

import numpy as np
import torch

TOK = Path('sweep_results/relation_tokens')

# (row label, params, respecting-protocol tensor, leaking-protocol tensor)
ISB = [
    ('E3  DeepSets',          17_270, 'lco_oof_softmax_isb_binary_E3_none.pt',
                                      'lco_oof_softmax_isb_binary_E3_rand-1100_none.pt'),
    ('E1  d64',               93_222, 'lco_oof_softmax_isb_binary_E1_d64_none.pt',
                                      'lco_oof_softmax_isb_binary_E1_d64_rand-1100_none.pt'),
    ('E1  d96',              195_558, 'lco_oof_softmax_isb_binary_E1_d96_none.pt',
                                      'lco_oof_softmax_isb_binary_E1_d96_rand-1100_none.pt'),
    ('E0  RelationTransformer', 298_118, 'lco_oof_softmax_isb_binary_E0_none.pt',
                                      'lco_oof_softmax_isb_binary_E0_rand-1100_none.pt'),
]
UNSAFENET = [
    ('E3  DeepSets',           18_044, 'lco_oof_softmax_bin8_binary_E3_drop_conflict.pt',
                                       'kfold_oof_softmax_bin8_binary_E3_drop_conflict.pt'),
    ('E1  d64',                94_380, 'lco_oof_softmax_bin8_binary_E1_d64_drop_conflict.pt',
                                       'kfold_oof_softmax_bin8_binary_E1_d64_drop_conflict.pt'),
    ('E1  d96',               197_292, 'lco_oof_softmax_bin8_binary_E1_d96_drop_conflict.pt',
                                       'kfold_oof_softmax_bin8_binary_E1_d96_drop_conflict.pt'),
    ('E2  frame-positional',  284_556, 'lco_oof_softmax_bin8_binary_E2_drop_conflict.pt',
                                       'kfold_oof_softmax_bin8_binary_E2_drop_conflict.pt'),
    ('E0  RelationTransformer', 298_892, 'lco_oof_softmax_bin8_binary_E0_drop_conflict.pt',
                                       'kfold_oof_softmax_bin8_binary_E0_drop_conflict.pt'),
]


def balanced_acc(path):
    """Balanced accuracy of one saved OOF tensor, recomputed from labels and predictions."""
    d = torch.load(TOK / path, weights_only=False)
    y = np.asarray(d['labels'])
    p = np.asarray(d['oof_pred'])
    if y.shape != p.shape:
        raise ValueError(f'{path}: labels {y.shape} vs preds {p.shape}')
    recalls = [float((p[y == c] == c).mean()) for c in np.unique(y)]
    return 100.0 * float(np.mean(recalls)), len(y)


def table(title, rows, respect_name, leak_name):
    print(f'\n=== {title} ===')
    print(f'{"encoder":<26}{"params":>10}{respect_name:>14}{leak_name:>14}{"inflation":>12}')
    respecting, leaking, n_seen = [], [], set()
    for label, params, a, b in rows:
        try:
            ba_r, n_r = balanced_acc(a)
            ba_l, n_l = balanced_acc(b)
        except FileNotFoundError as e:
            print(f'{label:<26}{params:>10,}   MISSING: {Path(e.filename).name}')
            continue
        n_seen.update({n_r, n_l})
        respecting.append(ba_r); leaking.append(ba_l)
        print(f'{label:<26}{params:>10,}{ba_r:>13.1f}%{ba_l:>13.1f}%{ba_l - ba_r:>+11.1f} pp')
    if not respecting:
        return
    sr, sl = max(respecting) - min(respecting), max(leaking) - min(leaking)
    print(f'{"spread across the ladder":<26}{"":>10}{sr:>12.1f} pp{sl:>12.1f} pp')
    print(f'  n clips scored: {sorted(n_seen)}   ladder span: '
          f'{max(r[1] for r in rows) / min(r[1] for r in rows):.0f}x params')
    print(f'  -> the identity-respecting protocol separates the ladder by {sr:.1f} pp; '
          f'the leaking one by {sl:.1f} pp.')


if __name__ == '__main__':
    if not TOK.is_dir():
        sys.exit(f'run from the project root: {TOK} not found')
    table('iSafetyBench (1,100 clips, 151 YouTube sources)', ISB,
          'LSO-1100', 'RAND-1100')
    table('Unsafe-Net (687 clips after dedupe, 8 scene groups)', UNSAFENET,
          'LCO-691', 'k-fold-691')
