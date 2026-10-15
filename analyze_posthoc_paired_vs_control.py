#!/usr/bin/env python3
"""POST-HOC: is any sweep row actually better than the control, or is it inside the drift band?

NOT a pre-registered gate. Re-trains nothing; reads prediction files only.

The problem this solves
-----------------------
Section 4.5.6 measures a 1.46 pp band of run-to-run drift (83.25 / 84.17 / 84.71 for 4 / published
/ 1 CPU thread) caused by checkpoint selection on a saturated val_macro. Any sweep row whose margin
over the 4-thread control is smaller than that band is indistinguishable from the drift and must not
be read as an improvement. A margin LARGER than the band is still not evidence on its own, because
the independence unit is the source video, not the clip: 1,100 clips come from 151 sources and the
top 5 sources hold ~24% of the corpus.

So each row gets, against the same-environment control:
  * the margin in pp,
  * a PAIRED cluster bootstrap CI resampling the 151 SOURCES (B = 5000, seed = 0, alpha = 0.05,
    percentile) -- the unit and settings of section 3.7.4,
  * exact McNemar on the discordant clips, reported for disclosure and labelled WRONG UNIT,
    because McNemar assumes independent clips and these are not,
  * whether the margin clears the 1.46 pp drift band.

A row counts as a candidate effect only when the margin clears the band AND the CI excludes 0.
Both conditions, as a conjunction, exactly as the E2 backbone pre-registration ruled.

Usage: .venv/bin/python analyze_posthoc_paired_vs_control.py
"""
import glob
import json
import os

import numpy as np

from eval_utils.cluster_bootstrap import balanced_accuracy, paired_metric_cluster_bootstrap
from eval_utils.paired_tests import mcnemar_exact

PUBLISHED = 'sweep_results/relation_tokens/enc_isbclip_B1_lso-1100_headline-none_oof.json'
CONTROL = 'sweep_results/posthoc_sensitivity/B1_published_config_oof.json'
DRIFT_BAND_PP = 1.46          # section 4.5.6: 84.71 - 83.25
OUT = 'sweep_results/posthoc_sensitivity/paired_vs_control.json'


def rows_of(path):
    rec = json.load(open(path))
    r = rec['rows']
    return (rec, np.array([x['y_true'] for x in r]), np.array([x['y_pred'] for x in r]),
            np.array([x['clip_id'] for x in r]))


def main():
    pub, _, _, _ = rows_of(PUBLISHED)
    group_of = {x['clip_id']: x['fold_id'] for x in pub['rows']}
    _, y, pred_c, ids_c = rows_of(CONTROL)
    groups = np.array([group_of[c] for c in ids_c])
    ba_c = 100 * balanced_accuracy(y, pred_c)
    print(f'control = {CONTROL}  BA={ba_c:.2f}%  '
          f'{len(np.unique(groups))} sources  drift band {DRIFT_BAND_PP} pp\n')
    paths = [p for p in sorted(glob.glob('sweep_results/posthoc_sensitivity/*_oof.json'))
             + sorted(glob.glob('sweep_results/posthoc_sensitivity/round2/*_oof.json'))
             if os.path.abspath(p) != os.path.abspath(CONTROL)]
    out = []
    for path in paths:
        rec, ya, pred_a, ids_a = rows_of(path)
        if not np.array_equal(ids_a, ids_c) or not np.array_equal(ya, y):
            print(f'SKIP {path}: different clip order or labels')
            continue
        diff, lo, hi, nb = paired_metric_cluster_bootstrap(
            y, pred_a, pred_c, groups, metric=balanced_accuracy, n_boot=5000, seed=0)
        mc = mcnemar_exact(pred_a == y, pred_c == y)
        p_mc = {'p_value': mc['p_value'], 'n_discordant': mc['n_discordant'],
                'underpowered': mc['underpowered']}
        clears = diff > DRIFT_BAND_PP
        ci_excl = (lo > 0) or (hi < 0)
        row = {'config': os.path.basename(path).replace('_oof.json', ''),
               'ba': round(100 * balanced_accuracy(y, pred_a), 2),
               'margin_pp_vs_control': round(diff, 2),
               'cluster_ci_pp': [round(lo, 2), round(hi, 2)], 'bootstrap_draws': nb,
               'mcnemar_WRONG_UNIT': p_mc,
               'clears_drift_band': bool(clears), 'ci_excludes_zero': bool(ci_excl),
               'candidate_effect': bool(clears and ci_excl)}
        out.append(row)
        verdict = ('CANDIDATE' if row['candidate_effect']
                   else 'band only' if clears else 'inside drift band')
        print(f"{row['config']:34s} BA={row['ba']:6.2f}  margin={diff:+6.2f} pp  "
              f"CI=[{lo:+6.2f},{hi:+6.2f}]  n_draws={nb}  {verdict}")
    out.sort(key=lambda r: -r['margin_pp_vs_control'])
    json.dump({'note': 'POST-HOC. Not a gate. Margins are against the same-environment 4-thread '
                        'control, never against the published 84.17%.',
               'control': {'artifact': CONTROL, 'ba': round(ba_c, 2)},
               'drift_band_pp': DRIFT_BAND_PP,
               'bootstrap': {'unit': 'source video', 'B': 5000, 'seed': 0, 'alpha': 0.05,
                             'kind': 'percentile', 'paired': True},
               'multiple_comparisons': f'{len(out)} rows compared to one control; no correction '
                                        'applied, so a single CANDIDATE row is not a finding.',
               'rows': out}, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
