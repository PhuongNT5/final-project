#!/usr/bin/env python3
"""Tier-1 GATES for the binary Safe/Unsafe reframe — S1 detector-rule, S2 scene-appearance.

Must run BEFORE any encoder (E0/E1/...) training (phase-06 §Implementation Steps 2). S1 is a
3-feature-per-class rule over the entity detector's own confidences; if it is near ceiling, the
encoder has no headroom and the thesis claim needs restructuring. S2 is background-only (no crop,
no detector) — the primary control for scene-fitting (design-2026-09-16-binary-encoder-redesign.md
§4.2). Both are cheap sklearn baselines (LogReg / depth<=2 tree), no NN training.

Protocols: k-fold-691 (pooled train+test, matches the existing 4-class run_kfold_relation.py /
run_lco_eval.py precedent — pooling here is 0 test touches per phase-06 and the 2026-09-17
pre-registration in reports/test-touch-log.md) and LCO-691 (leave-one-scene-cluster-out via
sweep_results/zone_prior/clip_scene_map_691.json, pooled OOF only — those folds are severely
class-skewed, never report per-fold accuracy).

Usage: .venv/bin/python run_binary_baseline_gates.py [--k 5] [--repeats 5]
"""
import argparse
import numpy as np

from eval_utils import evaluate_suite
from eval_utils.paired_tests import nadeau_bengio, mcnemar_exact
from binary_gate_lib import load_pool, kfold_oof, lco_oof, save_oof, NAMES, SUB_NAMES

SYSTEMS = ['s1_logreg', 's1_tree', 's2_logreg']
HARD_CEIL = {'none': (124, 125), 'drop_conflict': (124, 125)}
PRACTICAL_CEIL = {'none': (121, 125), 'drop_conflict': (124, 125)}


def per_subclass_recall(sub, y, pred):
    return {SUB_NAMES[k]: (round(100 * (pred[sub == k] == y[sub == k]).mean(), 1)
                           if (sub == k).any() else float('nan')) for k in range(8)}


def print_headline(tag, res, sub, y, pred):
    p = res['per_class']['Unsafe']
    # balanced_accuracy is a literal alias of macro_accuracy in unsafenet/metrics.py (balanced_accuracy
    # = macro_accuracy), so macro_accuracy's bootstrap CI95 IS balanced_accuracy's CI95.
    ba_lo, ba_hi = res['macro_accuracy']['ci95']
    print(f"\n  [{tag}] n={res['n']}")
    print(f"    balanced_acc={100*res['balanced_accuracy']['value']:.1f}% CI95=[{100*ba_lo:.1f},"
          f"{100*ba_hi:.1f}]  "
          f"macro_f1={100*res['macro_f1']['value']:.1f}% CI95=[{100*res['macro_f1']['ci95'][0]:.1f},"
          f"{100*res['macro_f1']['ci95'][1]:.1f}]  "
          f"unsafe_recall={100*p['recall']:.1f}%  unsafe_prec={100*p['precision']:.1f}%  "
          f"safe_recall={100*res['per_class']['Safe']['recall']:.1f}%  "
          f"PR-AUC={res['pr_auc_ovr']['macro']:.3f}  ROC-AUC={res['roc_auc_ovr']['macro']:.3f}")
    psc = per_subclass_recall(sub, y, pred)
    print(f"    per-subclass recall: {psc}")
    print(f"    PANEL PAIR (2 OpenedPanel vs 6 ClosedPanel, 0.12 co-cluster overlap): "
          f"OpenedPanel={psc['OpenedPanel']}%  ClosedPanel={psc['ClosedPanel']}%")


def run_protocol(name, pool, dedupe_tag, k, repeats):
    print(f"\n=== {name} | dedupe={dedupe_tag} | n={pool['n']} (dropped {pool['dropped']}) ===")
    preds, nb_fold_acc = {}, {}
    for sysname in SYSTEMS:
        X = pool['s1'] if sysname.startswith('s1') else pool['s2']
        if name == 'kfold-691':
            prob, facc, ntr, nte, fold_id = kfold_oof(X, pool['y'], sysname, k=k, repeats=repeats)
            nb_fold_acc[sysname] = (facc, ntr, nte)
        else:
            prob, _ = lco_oof(X, pool['y'], pool['group'], sysname)
            fold_id = pool['group']
        pred = prob.argmax(1)
        preds[sysname] = pred
        res = evaluate_suite(pool['y'], prob, class_names=NAMES, protocol_tag=f'{name}_{dedupe_tag}_{sysname}',
                             unsafe_classes=[1], n_seeds=1)
        print_headline(f'{name}/{dedupe_tag}/{sysname}', res, pool['sub'], pool['y'], pred)
        oof_path = save_oof(sysname, name, dedupe_tag, pool['clip_ids'], pool['y'], prob, pred, fold_id)
        print(f"    OOF saved: {oof_path}")

    # majority-class floor (S0), pooled, for ceiling context
    maj = int(np.bincount(pool['y']).argmax())
    maj_pred = np.full(pool['n'], maj)
    print(f"  [S0 majority] acc={100*(maj_pred==pool['y']).mean():.1f}%  "
          f"(constant predicts {NAMES[maj]} for every clip)")

    # own-vs-own exact McNemar: S2 vs S1-logreg (both per-clip predictions available)
    mc = mcnemar_exact(preds['s2_logreg'] == pool['y'], preds['s1_logreg'] == pool['y'])
    print(f"  McNemar S2-logreg vs S1-logreg: b={mc['b']} c={mc['c']} p={mc['p_value']:.4f} "
          f"diff={mc['diff_accuracy_pp']:+.2f}pp CI95={mc['diff_accuracy_ci95_pp']}")

    if name == 'kfold-691':
        facc, ntr, nte = nb_fold_acc['s2_logreg']
        facc1 = nb_fold_acc['s1_logreg'][0]
        diffs = [a - b for a, b in zip(facc, facc1)]
        nb = nadeau_bengio(diffs, ntr, nte)
        print(f"  NB-corrected (S2-logreg vs S1-logreg, {len(diffs)} fold-diffs): "
              f"mean_diff={nb['mean_diff']:+.2f}pp t={nb['t']:.2f} p={nb['p_value']:.4f}")
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=5)
    args = ap.parse_args()

    for policy, tag in (('none', 'none'), ('drop_conflict', 'headline-drop_conflict')):
        pool = load_pool(policy)
        h, p = HARD_CEIL[policy], PRACTICAL_CEIL[policy]
        print(f"\n########## DEDUPE POLICY = {tag} ({policy}) — hard ceiling {h[0]}/{h[1]}="
              f"{100*h[0]/h[1]:.1f}%  practical ceiling {p[0]}/{p[1]}={100*p[0]/p[1]:.1f}% ##########")
        run_protocol('kfold-691', pool, tag, args.k, args.repeats)
        run_protocol('lco-691', pool, tag, args.k, args.repeats)

    print("\n=== GATE VERDICT ===")
    print("See report for the full pre-committed reading of the numbers above (phase-06 §Risk R1/R6).")


if __name__ == '__main__':
    main()
