#!/usr/bin/env python3
"""Round-3 frozen-tower family: cluster-bootstrap p-values and the pre-registered BH correction.

Why a separate script
---------------------
`analyze_posthoc_paired_vs_control.py` compares every post-hoc row to the 83.25% four-thread
control and prints a margin plus a source-level CI, but it applies NO multiple-comparison
correction and says so in its own output. Round 3 was pre-registered with one
(`plans/.../chien-luoc-thuc-nghiem-vong-3.md`, section 3): Benjamini-Hochberg at q = 0.05 over the
frozen-tower family, with the denominator fixed BEFORE any tower was run. This script applies it.

Three disclosures the numbers cannot carry on their own
-------------------------------------------------------
1. DENOMINATOR. The pre-registration names ten tower rows, two of them controls, so the family is
   EIGHT non-control towers -- the document's own sentence says "7", which was a miscount, and the
   larger, more conservative figure is the one used here. `eva02B16` never loaded, so only seven
   p-values exist; the missing one enters BH as p = 1. Both choices make the test harder to pass,
   never easier, so neither can be a post-hoc convenience.
2. ONE ROW PER TOWER. Each tower was trained with two heads. Using both would silently double the
   family after the fact. The primary row is the TRANSFORMER head, because that is the deployed B1
   graph of the thesis and the only choice that holds the architecture identical to the published
   model; it is not the better head for every tower (meanpool wins on four of seven), so the rule
   is not a selection on the outcome. The meanpool rows are corrected separately as a secondary
   family and reported as such.
3. P-VALUE DEFINITION. The percentile bootstrap gives an interval, not a p-value. The two-sided
   value here is the achieved significance level of the SAME source-resampled draws:
   p = 2 * min(P(d* <= 0), P(d* >= 0)), floored at 1/B. The draws are generated with the identical
   generator, seed and discard rule as the published CI (default_rng(0), 5000 draws, drop any draw
   that loses a class), so the p-value and the interval describe one bootstrap, not two.

Writes only sweep_results/posthoc_sensitivity/round3_tower_bh.json. Re-adjudicates no gate.
"""
import json
import os

import numpy as np

from eval_utils.cluster_bootstrap import balanced_accuracy

POSTHOC = 'sweep_results/posthoc_sensitivity'
CONTROL = f'{POSTHOC}/B1_published_config_oof.json'
PUBLISHED = 'sweep_results/relation_tokens/enc_isbclip_B1_lso-1100_headline-none_oof.json'
OUT = f'{POSTHOC}/round3_tower_bh.json'
DRIFT_BAND_PP = 1.46
Q = 0.05

# Fixed before any tower ran: eight non-control towers. `eva02B16` has no row (it never loaded) and
# is carried as p = 1. tag -> (pre-registered name, role)
FAMILY = {
    'towerB16':   ('clipB16', 'same training data, patch 16 instead of 32'),
    'towerDC1B':  ('openclipB16_dc1b', 'DataComp-1B'),
    'towerDFN2B': ('dfn2bB16', 'DFN-2B'),
    'towerSIG2':  ('siglip2B16', 'SigLIP 2, sigmoid loss, 768-d'),
    'towerMETA':  ('metaclipB16', 'MetaCLIP 400M'),
    'towerL2B':   ('openclipB16_l2b', 'LAION-2B'),
    'towerMOBS1': ('mobileclip_s1', 'SMALLER than the deployed B/32'),
    'towerEVA02': ('eva02B16', 'NOT MEASURED: open_clip could not load it'),
}


def rows_of(path):
    rec = json.load(open(path))
    rows = rec['rows']
    return (rec, np.array([r['y_true'] for r in rows]),
            np.array([r['y_pred'] for r in rows]), np.array([r['clip_id'] for r in rows]))


def paired_draws(y, pred_a, pred_b, groups, n_boot=5000, seed=0):
    """The draw set behind the published CI, returned instead of summarised."""
    uniq = np.unique(groups)
    idx_by_group = [np.where(groups == g)[0] for g in uniq]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_group[p] for p in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append((balanced_accuracy(y[idx], pred_a[idx])
                      - balanced_accuracy(y[idx], pred_b[idx])) * 100.0)
    return np.array(boots)


def asl(boots):
    """Two-sided achieved significance level of the bootstrap distribution, floored at 1/B."""
    b = len(boots)
    if b == 0:
        return float('nan')
    p = 2 * min((boots <= 0).sum(), (boots >= 0).sum()) / b
    return float(max(min(p, 1.0), 1.0 / b))


def benjamini_hochberg(pairs, m, q=Q):
    """pairs = [(key, p)]. m is the pre-registered denominator, >= len(pairs). -> set of rejected."""
    ordered = sorted(pairs, key=lambda kv: kv[1])
    rejected, k_max = set(), 0
    for i, (_, p) in enumerate(ordered, 1):
        if p <= q * i / m:
            k_max = i
    for key, _ in ordered[:k_max]:
        rejected.add(key)
    return rejected, ordered


def evaluate(suffix, label, y, pred_c, groups, control_ba):
    rows, pairs = {}, []
    for tag, (name, role) in FAMILY.items():
        path = f'{POSTHOC}/round2/{tag}_{suffix}_oof.json'
        if not os.path.exists(path):
            rows[tag] = {'tower': name, 'role': role, 'measured': False, 'p_bootstrap': 1.0}
            pairs.append((tag, 1.0))
            continue
        _, ya, pred_a, _ = rows_of(path)
        boots = paired_draws(y, pred_a, pred_c, groups)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        diff = (balanced_accuracy(y, pred_a) - balanced_accuracy(y, pred_c)) * 100.0
        p = asl(boots)
        rows[tag] = {'tower': name, 'role': role, 'measured': True,
                     'ba': round(100 * balanced_accuracy(y, pred_a), 2),
                     'margin_pp_vs_control': round(diff, 2),
                     'cluster_ci_pp': [round(lo, 2), round(hi, 2)],
                     'p_bootstrap': round(p, 4), 'draws': int(len(boots)),
                     'clears_drift_band': bool(diff > DRIFT_BAND_PP),
                     'ci_excludes_zero': bool(lo > 0 or hi < 0)}
        pairs.append((tag, p))
    m = len(FAMILY)
    rejected, ordered = benjamini_hochberg(pairs, m)
    for tag in rows:
        rows[tag]['bh_rejected'] = tag in rejected
        rows[tag]['survives_preregistered_rule'] = bool(
            rows[tag].get('clears_drift_band') and rows[tag].get('ci_excludes_zero')
            and tag in rejected)
    print(f'\n=== {label}  (control {control_ba:.2f}%, BH m={m}, q={Q}) ===')
    for tag, p in ordered:
        r = rows[tag]
        if not r['measured']:
            print(f"{r['tower']:20s} NOT MEASURED, carried as p=1")
            continue
        verdict = ('SURVIVES' if r['survives_preregistered_rule']
                   else 'band+CI, BH rejects' if r['ci_excludes_zero'] and r['clears_drift_band']
                   else 'band only' if r['clears_drift_band'] else 'inside drift band')
        print(f"{r['tower']:20s} BA={r['ba']:6.2f}  margin={r['margin_pp_vs_control']:+6.2f} pp  "
              f"CI=[{r['cluster_ci_pp'][0]:+6.2f},{r['cluster_ci_pp'][1]:+6.2f}]  "
              f"p={p:.4f}  {verdict}")
    return rows


def main():
    pub, _, _, _ = rows_of(PUBLISHED)
    group_of = {x['clip_id']: x['fold_id'] for x in pub['rows']}
    _, y, pred_c, ids_c = rows_of(CONTROL)
    groups = np.array([group_of[c] for c in ids_c])
    ba_c = 100 * balanced_accuracy(y, pred_c)
    primary = evaluate('control_round2', 'PRIMARY: transformer head (the deployed B1 graph)',
                       y, pred_c, groups, ba_c)
    second = evaluate('arch_meanpool', 'SECONDARY family: mean-pool head', y, pred_c, groups, ba_c)
    json.dump({'note': 'POST-HOC. Not a gate. Margins are against the same-environment 4-thread '
                       'control (83.25%), never against the published 84.17%.',
               'control': {'artifact': CONTROL, 'ba': round(ba_c, 2)},
               'drift_band_pp': DRIFT_BAND_PP,
               'bootstrap': {'unit': 'source video', 'B': 5000, 'seed': 0, 'alpha': 0.05,
                             'kind': 'percentile', 'paired': True,
                             'p_value': 'two-sided achieved significance level of the same draws'},
               'multiple_comparisons': {'method': 'Benjamini-Hochberg', 'q': Q,
                                        'm': len(FAMILY),
                                        'denominator_fixed': 'before any tower was run',
                                        'note': 'eva02B16 never loaded; carried as p=1'},
               'primary_transformer_head': primary, 'secondary_meanpool_head': second},
              open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
