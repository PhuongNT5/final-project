"""Mechanical adjudication of the pre-registered R48 gates G14/G15/G16.

The thresholds below are copied from risks.md, which was written before any of these models existed.
They are constants in this file so the verdict is computed, not decided after seeing the numbers: if
a gate fails, this script says FAIL and there is nothing to renegotiate.

Usage: .venv/bin/python tools/adjudicate-r48-gates.py
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_utils.cluster_bootstrap import balanced_accuracy, paired_metric_cluster_bootstrap
from eval_utils.paired_tests import mcnemar_exact

TOK = 'sweep_results/relation_tokens'
MARGIN = 3.0          # pp, pre-registered for G14 and G15
ALPHA = 0.05
RUNS = {'ESTA': f'{TOK}/enc_isbclip_ESTA_lso-1100_headline-none_oof.json',
        'B1': f'{TOK}/enc_isbclip_B1_lso-1100_headline-none_oof.json',
        'B1w': f'{TOK}/enc_isbclip_B1w_lso-1100_headline-none_oof.json',
        'B2': f'{TOK}/enc_isbclip_B2_lso-1100_headline-none_oof.json',
        'E0@64d': f'{TOK}/enc_isb_E0_lso-1100_headline-none_oof.json'}


def _loader():
    """Reuse compare-oof-runs' loader rather than reimplementing it; its filename is not importable."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compare-oof-runs.py')
    spec = importlib.util.spec_from_file_location('compare_oof_runs', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load


def paired(a, b, runs):
    """-> dict of the paired statistics for run `a` minus run `b`, on their common clips."""
    A, B = runs[a], runs[b]
    ids = sorted(set(A) & set(B))
    y = np.array([A[i]['y_true'] for i in ids])
    pa = np.array([A[i]['y_pred'] for i in ids])
    pb = np.array([B[i]['y_pred'] for i in ids])
    g = np.array([str(A[i]['fold_id']) for i in ids])
    d, lo, hi, nv = paired_metric_cluster_bootstrap(y, pa, pb, g, n_boot=5000)
    mc = mcnemar_exact(pa == y, pb == y)
    # McNemar's discordant counts are named mc_b/mc_c: plain 'b' is the baseline run's name, and
    # letting the count reuse that key silently replaced the name in the printed table.
    return {'a': a, 'b': b, 'n': len(ids), 'groups': len(set(g)),
            'ba_a': balanced_accuracy(y, pa) * 100, 'ba_b': balanced_accuracy(y, pb) * 100,
            'diff': d, 'lo': lo, 'hi': hi, 'draws': nv,
            'p': mc['p_value'], 'mc_b': mc['b'], 'mc_c': mc['c']}


def show(r):
    excl = r['lo'] > 0 or r['hi'] < 0
    print(f"  {r['a']} {r['ba_a']:.1f}%  -  {r['b']} {r['ba_b']:.1f}%  =  {r['diff']:+.1f} pp")
    print(f"    cluster bootstrap over {r['groups']} sources ({r['draws']}/5000 valid): "
          f"95% CI [{r['lo']:+.1f}, {r['hi']:+.1f}] {'EXCLUDES 0' if excl else 'includes 0'}")
    print(f"    McNemar exact: b={r['mc_b']} c={r['mc_c']} p={r['p']:.3g}")
    return excl


def main():
    load = _loader()
    runs, missing = {}, []
    for k, p in RUNS.items():
        (runs.__setitem__(k, load(p)) if os.path.exists(p) else missing.append(f'{k} ({p})'))
    if missing:
        print(f"[missing] {'; '.join(missing)}\n         adjudicating only the gates whose runs exist.")

    if 'ESTA' in runs and {'B1', 'B1w'} & set(runs):
        print("\n=== G14 (THE GATE): ESTA vs the best whole-frame baseline at matched backbone ===")
        cands = [k for k in ('B1', 'B1w') if k in runs]
        best = max(cands, key=lambda k: balanced_accuracy(
            np.array([runs[k][i]['y_true'] for i in sorted(runs[k])]),
            np.array([runs[k][i]['y_pred'] for i in sorted(runs[k])])))
        print(f"  baseline = {best} (higher of {', '.join(cands)}; pre-registered best-of-two)")
        r = paired('ESTA', best, runs)
        excl = show(r)
        ok = r['diff'] >= MARGIN and excl and r['p'] < ALPHA
        print(f"  VERDICT: {'PASS' if ok else 'FAIL'}  (needs >= {MARGIN} pp, CI excluding 0, p < {ALPHA})")
        if not ok:
            print("  -> pre-registered consequence: the entity decomposition adds nothing over a "
                  "whole-frame representation at matched backbone, and that is final.")

    if {'ESTA', 'E0@64d'} <= set(runs):
        print("\n=== G15 (component ablation, NOT the contribution): representation bottleneck ===")
        r = paired('ESTA', 'E0@64d', runs)
        excl = show(r)
        print(f"  VERDICT: {'PASS' if r['diff'] >= MARGIN and excl else 'FAIL'}")

    if {'ESTA', 'B2'} <= set(runs):
        print("\n=== G16 (honest strong opponent): ESTA vs frozen Kinetics VideoMAE ===")
        r = paired('ESTA', 'B2', runs)
        excl = show(r)
        if r['diff'] < 0 and excl:
            print("  -> B2 BEATS ESTA with a CI excluding 0: the entity pipeline is not justified "
                  "on accuracy grounds, even if G14 passed. This must be reported.")
        else:
            print(f"  -> ESTA {'ahead' if r['diff'] > 0 else 'behind'}; "
                  f"{'CI excludes 0' if excl else 'CI includes 0 (no separation)'}")


if __name__ == '__main__':
    main()
