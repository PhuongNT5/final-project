"""Paired comparison of two pooled-OOF runs scored on the SAME clips.

Every own-vs-own comparison in this project must be paired (standing rule), and on a grouped corpus
the resampling unit is the GROUP, not the clip — so this prints the cluster bootstrap CI as the
primary number and the clip-level CI only as a disclosure of what the wrong unit would have said.

Usage:
  .venv/bin/python tools/compare-oof-runs.py A_oof.json B_oof.json [--label_a X --label_b Y]

Both files are the {rows:[{clip_id,y_true,y_pred,fold_id}]} schema written by run_kfold_relation.py.
`fold_id` carries the grouping key (source video id / scene group), so the cluster unit comes from
the artifacts themselves and cannot drift from what the run actually held out.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, '.')
from eval_utils.cluster_bootstrap import balanced_accuracy, paired_metric_cluster_bootstrap
from eval_utils.paired_tests import mcnemar_exact


def load(path, ids_from=None):
    """Per-clip OOF records keyed by clip id.

    Two on-disk shapes exist: the `*_oof.json` the iSafetyBench driver writes (rows already carry
    the clustering key in `fold_id` -- the YouTube source id), and the `*_oof_softmax_*.pt` tensors
    the Unsafe-Net runs wrote, which keep the clustering key in a separate `groups` array. Both are
    normalised to the same record here so the Unsafe-Net ladder can be paired-tested without
    re-training it.
    """
    if path.endswith('.pt'):
        import torch
        d = torch.load(path, weights_only=False)
        if 'clip_ids' not in d or 'groups' not in d:
            # The random k-fold Unsafe-Net runs saved predictions without identity columns. They are
            # row-aligned with the leave-camera-out tensors of the same corpus, so --ids_from supplies
            # them; the caller asserts label equality row by row before trusting that alignment.
            return None if ids_from is None else _attach_ids(d, load(ids_from))
        return {str(c): {'clip_id': str(c), 'y_true': int(y), 'y_pred': int(p_),
                         'fold_id': str(g), '_row': i}
                for i, (c, y, p_, g) in enumerate(
                    zip(d['clip_ids'], d['labels'], d['oof_pred'], d['groups']))}
    rows = json.load(open(path))['rows']
    return {r['clip_id']: r for r in rows}


def _attach_ids(d, donor):
    """Row-align an identity-less tensor onto a donor run's clip ids and clustering key.

    Refuses on any label mismatch: identical labels in identical row order is the only evidence
    that the two runs scored the same clips in the same order.
    """
    if any('_row' not in r for r in donor.values()):
        raise ValueError('--ids_from must be a .pt donor carrying clip_ids + groups (e.g. an '
                         'lco_oof_softmax_*.pt). A *_oof.json donor has no row index, so its rows '
                         'cannot be ordered against the identity-less tensor.')
    rows = sorted(donor.values(), key=lambda r: r['_row'])
    labels, preds = d['labels'], d['oof_pred']
    if len(rows) != len(labels):
        raise ValueError(f'row count differs: donor {len(rows)} vs tensor {len(labels)}')
    for i, r in enumerate(rows):
        if r['y_true'] != int(labels[i]):
            raise ValueError(f'label mismatch at row {i}: donor {r["y_true"]} vs tensor '
                             f'{int(labels[i])} -- the two runs are NOT row-aligned, refusing')
    return {r['clip_id']: {**r, 'y_pred': int(preds[i])} for i, r in enumerate(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a'); ap.add_argument('b')
    ap.add_argument('--label_a', default=None); ap.add_argument('--label_b', default=None)
    ap.add_argument('--n_boot', type=int, default=5000)
    ap.add_argument('--ids_from', default=None,
                    help='tensor supplying clip ids + clustering key for a run that saved neither '
                         '(row-aligned; label equality is asserted before use)')
    args = ap.parse_args()
    la = args.label_a or args.a.split('/')[-1]
    lb = args.label_b or args.b.split('/')[-1]

    A, B = load(args.a, args.ids_from), load(args.b, args.ids_from)
    ids = sorted(set(A) & set(B))
    if len(ids) != len(A) or len(ids) != len(B):
        print(f"[warn] clip sets differ: |A|={len(A)} |B|={len(B)} |A n B|={len(ids)} -- "
              f"comparing the intersection only")
    y = np.array([A[i]['y_true'] for i in ids])
    pa = np.array([A[i]['y_pred'] for i in ids])
    pb = np.array([B[i]['y_pred'] for i in ids])
    assert (y == np.array([B[i]['y_true'] for i in ids])).all(), 'label mismatch between the two runs'
    g = np.array([str(A[i]['fold_id']) for i in ids])

    ba_a, ba_b = balanced_accuracy(y, pa) * 100, balanced_accuracy(y, pb) * 100
    print(f"n={len(ids)} clips, {len(set(g))} groups")
    print(f"  {la}: balanced_acc={ba_a:.1f}%")
    print(f"  {lb}: balanced_acc={ba_b:.1f}%")
    print(f"  A - B = {ba_a - ba_b:+.1f} pp")

    d, lo, hi, nv = paired_metric_cluster_bootstrap(y, pa, pb, g, n_boot=args.n_boot)
    print(f"  [cluster bootstrap over {len(set(g))} groups, {nv}/{args.n_boot} valid draws] "
          f"diff={d:+.1f} pp  95% CI [{lo:+.1f}, {hi:+.1f}]  "
          f"{'EXCLUDES 0' if lo > 0 or hi < 0 else 'includes 0'}")
    dc, loc, hic, nvc = paired_metric_cluster_bootstrap(y, pa, pb, np.arange(len(ids)).astype(str),
                                                        n_boot=args.n_boot)
    print(f"  [clip-level bootstrap -- WRONG UNIT, disclosure only] 95% CI [{loc:+.1f}, {hic:+.1f}]")

    mc = mcnemar_exact(pa == y, pb == y)
    print(f"  [McNemar exact] b={mc['b']} c={mc['c']} n_discordant={mc['n_discordant']} "
          f"p={mc['p_value']:.3g} | {mc['note']}")


if __name__ == '__main__':
    main()
