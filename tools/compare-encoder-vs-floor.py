"""Paired encoder-vs-R42-count-rule comparison on iSafetyBench, with the cluster bootstrap CI.

The floor is recomputed here (not read from a file) on the SAME clips, SAME folds and SAME ablation
as the encoder run it is being compared against — that identity is the whole content of the G10
pre-registration, so it is enforced in code rather than trusted to a filename.

Usage:
  .venv/bin/python tools/compare-encoder-vs-floor.py <encoder_oof.json> [--ablate no_conf]
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, '.')
import feature_ablation as ABL
import isafetybench_dataset as ISB
from eval_utils.cluster_bootstrap import balanced_accuracy, paired_metric_cluster_bootstrap
from eval_utils.paired_tests import mcnemar_exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('encoder_oof')
    ap.add_argument('--ablate', default='none', choices=list(ABL.MODES))
    ap.add_argument('--tag', default='isb_')
    ap.add_argument('--folds', type=int, default=10)
    ap.add_argument('--n_boot', type=int, default=5000)
    args = ap.parse_args()

    X, M, y, groups, clip_ids = ISB.load(args.tag)
    X = ABL.apply(X, M, args.ablate, ISB.feat_layout(args.tag))
    splits = ISB.folds(y, groups, n_splits=args.folds, seed=0)
    fpred, _ = ISB.count_rule_oof(X, M, y, splits)

    rows = {r['clip_id']: r for r in json.load(open(args.encoder_oof))['rows']}
    order = [rows[str(c)] for c in clip_ids]            # align the encoder OOF to the loader's order
    epred = np.array([r['y_pred'] for r in order])
    yn = y.numpy()
    assert (np.array([r['y_true'] for r in order]) == yn).all(), 'label mismatch encoder vs loader'

    eba, fba = balanced_accuracy(yn, epred) * 100, balanced_accuracy(yn, fpred) * 100
    print(f"ablation={args.ablate}  n={len(yn)} clips / {len(set(groups))} sources")
    print(f"  encoder   balanced_acc = {eba:.1f}%   ({args.encoder_oof.split('/')[-1]})")
    print(f"  R42 floor balanced_acc = {fba:.1f}%   (recomputed, identical folds + identical ablation)")
    print(f"  margin = {eba - fba:+.1f} pp")

    d, lo, hi, nv = paired_metric_cluster_bootstrap(yn, epred, fpred, groups, n_boot=args.n_boot)
    print(f"  [cluster bootstrap over sources, {nv}/{args.n_boot} valid draws] "
          f"{d:+.1f} pp  95% CI [{lo:+.1f}, {hi:+.1f}]  "
          f"{'EXCLUDES 0' if lo > 0 or hi < 0 else 'INCLUDES 0'}")
    mc = mcnemar_exact(epred == yn, fpred == yn)
    print(f"  [McNemar exact] b={mc['b']} c={mc['c']} n_disc={mc['n_discordant']} p={mc['p_value']:.3g}")


if __name__ == '__main__':
    main()
