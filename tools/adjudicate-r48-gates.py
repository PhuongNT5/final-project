"""Mechanical adjudication of the pre-registered R48 gates G14/G15/G16 and G18.

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
MARGIN = 3.0          # pp, pre-registered for G14, G15 and G18a
G18B_MARGIN = 2.0     # pp, pre-registered second tier for G18 (risks.md, before the run)
ALPHA = 0.05
RUNS = {'ESTA': f'{TOK}/enc_isbclip_ESTA_lso-1100_headline-none_oof.json',
        'B1': f'{TOK}/enc_isbclip_B1_lso-1100_headline-none_oof.json',
        'B1w': f'{TOK}/enc_isbclip_B1w_lso-1100_headline-none_oof.json',
        'B2': f'{TOK}/enc_isbclip_B2_lso-1100_headline-none_oof.json',
        'E0@64d': f'{TOK}/enc_isb_E0_lso-1100_headline-none_oof.json',
        'TSF': f'{TOK}/enc_isbclip_TSF_lso-1100_headline-none_oof.json',
        'B1xl': f'{TOK}/enc_isbclip_B1xl_lso-1100_headline-none_oof.json'}


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



def adjudicate_g18(runs):
    """G18: the two-stream fusion against the incumbent best single stream.

    Two tiers, both written into risks.md before the model existed. The weak tier exists because
    R41 (151 sources, 0 carrying both classes) makes a cluster CI excluding 0 unreachable for any
    margin this lane can produce -- but a gate that can only fail is not a gate, so the weak tier
    states in advance exactly which weaker claim a significant paired test would license.
    """
    print("\n=== G18: two-stream fusion (TSF) vs the incumbent best single stream ===")
    base = 'B1w'          # pre-registered baseline, not best-of-N chosen after the fact
    print(f'  baseline = {base} (pre-registered as the incumbent best single stream, 84.4%)')
    r = paired('TSF', base, runs)
    excl = show(r)
    strong = r['diff'] >= MARGIN and excl and r['p'] < ALPHA
    weak = r['diff'] >= G18B_MARGIN and r['p'] < ALPHA
    if strong:
        print(f'  VERDICT: G18a PASS (strong) -- may claim the two-stream architecture '
              f'outperforms the whole-frame baseline.')
    elif weak:
        print(f'  VERDICT: G18b PASS (weak) -- may claim ONLY that the streams are complementary '
              f'and fusion improves accuracy on this corpus.')
        print('    Mandatory caveat: at 151 sources with 0 carrying both classes (R41), '
              'source-level generalisation of a margin this size is not certifiable.')
        print('    Forbidden: any unqualified superiority claim; the clip-level CI as headline.')
    else:
        print(f'  VERDICT: FAIL (needs >= {G18B_MARGIN} pp and p < {ALPHA} for the weak tier)')
        print('  -> pre-registered consequence: the entity stream adds nothing even as a '
              'complementary view. Thesis headlines B1/B1w; YOLO is localisation + edge.')
    # The aux arms are not the contribution, but a fusion that lost to its own entity arm would
    # mean the fusion block is harming the model, and that must surface rather than hide.
    if 'ESTA' in runs:
        print('\n  [control] TSF vs its own entity arm, to catch a fusion block that hurts:')
        show(paired('TSF', 'ESTA', runs))

    # Pre-registered capacity condition: a PASS must also survive a LARGER whole-frame model.
    if 'B1xl' in runs:
        print('\n  [capacity control] TSF (686,726 params) vs B1xl (696,962 -- 1.5% LARGER):')
        rc = paired('TSF', 'B1xl', runs)
        exc = show(rc)
        tier = ('G18a' if rc['diff'] >= MARGIN and exc and rc['p'] < ALPHA
                else 'G18b' if rc['diff'] >= G18B_MARGIN and rc['p'] < ALPHA else 'FAIL')
        print(f'    capacity condition reaches: {tier}')
        if strong and tier != 'G18a':
            print('    -> G18a is NOT sustained: the strong claim required the same tier against '
                  'B1xl. Report at the tier the capacity control supports, per the registration.')
        elif (strong or weak) and tier == 'FAIL':
            print('    -> CLAIM WITHDRAWN: TSF does not beat a larger whole-frame model, so the '
                  'fusion result is a capacity effect and the entity stream gets no credit.')
        elif strong or weak:
            print('    -> capacity condition SUSTAINED; "it is simply bigger" is not available.')


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

    if {'TSF', 'B1w'} <= set(runs):
        adjudicate_g18(runs)

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
