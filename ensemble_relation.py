#!/usr/bin/env python3
"""Hybrid ensemble of the two best relation-Transformer configs (v1 geometry-no-weights ⊕ v2b
appearance-no-weights). Averages their seed-averaged test softmax (a 14-model ensemble) → prediction.
Equal 50/50 weight (no test-tuning). Reports blind-comparable top-1/macro/confusion/CI vs baselines.

Usage: .venv/bin/python ensemble_relation.py
"""
import os
import numpy as np
import torch
from eval_utils.metrics import top1, macro_accuracy, per_class_recall, per_class_fnr, confusion
from eval_utils.bootstrap_ci import bootstrap_ci
from eval_utils.paired_tests import paired_vs_aggregate_baseline, mcnemar_exact

TOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_results', 'relation_tokens')
NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']


def main():
    a = torch.load(os.path.join(TOK, 'logits_v1nocw.pt'), weights_only=False)
    b = torch.load(os.path.join(TOK, 'logits_v2bnocw.pt'), weights_only=False)
    y = a['labels'].numpy()
    ens = (a['ens_softmax'] + b['ens_softmax']) / 2.0
    pred = ens.argmax(1).numpy()
    c = int((pred == y).sum())

    print("=== HYBRID ENSEMBLE  v1(geom) ⊕ v2b(appearance), no-weights, 7+7 seeds ===")
    print(f"top-1 = {c}/64 = {100*c/64:.1f}%  |  macro = {macro_accuracy(y, pred)*100:.1f}%")
    lo, hi = bootstrap_ci(y, pred, lambda p, t: top1(p, t)*100, n_boot=5000, seed=0)[1:3]
    mlo, mhi = bootstrap_ci(y, pred, lambda p, t: macro_accuracy(p, t)*100, n_boot=5000, seed=0)[1:3]
    print(f"top-1 BCa 95% CI = [{lo:.1f}, {hi:.1f}]  |  macro CI = [{mlo:.1f}, {mhi:.1f}]")
    pcr = per_class_recall(y, pred); fnr = per_class_fnr(y, pred)
    for k in range(4):
        n = int((y == k).sum())
        print(f"  {NAMES[k]:14s} n={n:2d}  recall={pcr[k]*100:5.1f}%  FNR={fnr[k]*100:5.1f}%")
    print("confusion (rows=true):\n", confusion(y, pred, 4))

    print("\n--- vs individual configs (blind seed) ---")
    print(f"  v1 geom no-weights : {a['blind_top1']}/64")
    print(f"  v2b appearance nw  : {b['blind_top1']}/64")
    # own-vs-own, same 64 clips, per-clip preds already loaded -> exact McNemar (not a bound).
    v1_pred = a['blind_pred'].numpy() if hasattr(a['blind_pred'], 'numpy') else np.asarray(a['blind_pred'])
    v2b_pred = b['blind_pred'].numpy() if hasattr(b['blind_pred'], 'numpy') else np.asarray(b['blind_pred'])
    for cfg_name, cfg_pred in [('v1 geom no-weights', v1_pred), ('v2b appearance nw', v2b_pred)]:
        m = mcnemar_exact(pred == y, cfg_pred == y)
        print(f"  McNemar hybrid vs {cfg_name}: b={m['b']} c={m['c']} p={m['p_value']:.4f} "
              f"diff={m['diff_accuracy_pp']:+.2f}pp CI95=[{m['diff_accuracy_ci95_pp'][0]:+.1f},"
              f"{m['diff_accuracy_ci95_pp'][1]:+.1f}]pp")
    print("\n--- vs paper ---")
    # Paper's per-clip predictions are unpublished -> paired (b,c) table can't be reconstructed ->
    # McNemar BOUND is the valid stat (not two_proportion_z, which wrongly assumes 2 independent
    # n=64 samples on what are actually the SAME 64 clips). See
    # review-2026-09-13-eval-statistics.md finding #1.
    for name, k in [('no-bbox baseline', 57), ('full-bbox', 62)]:
        r = paired_vs_aggregate_baseline(64, c, k, label=name)
        print(f"  ensemble {c}/64 vs {name} {k}/64: Δ={r['delta_pp']:+.1f}pp "
              f"McNemar bound p∈[{r['min_p_over_splits']:.4f},{r['max_p_over_splits']:.4f}] "
              f"(diagnostic-only two_proportion_z p={r['diagnostic_two_proportion_z']['p_value']:.3f}, invalid independence)")


if __name__ == '__main__':
    main()
