#!/usr/bin/env python3
"""Full disclosure table for sweep_results/posthoc_sensitivity/. POST-FREEZE, POST-HOC, NOT a gate.

run_b1_sensitivity_sweep.py varies one B1 hyperparameter at a time (*_oof.json per config). Some
configs score ABOVE the same-environment control (e.g. lr_3e-4 ~85.5 vs control ~83.25); none of
this sweep is in the thesis or any plan report. Builds one table: recomputed metrics, the exact
knob changed, margins vs BOTH the correct control and the published headline (kept distinct), a
paired cluster-bootstrap CI + Benjamini-Hochberg over the whole post-hoc family, and a drift-band
flag per row. Also checks a falsifiability claim: two chapters say no grad-clip 1.0-vs-5.0
comparison exists on the same data; gradclip_5.0_oof.json is exactly that. Recorded, not fixed --
chapters are NOT edited here. Scope: the 20 configs of GRID, iSafetyBench 1,100-clip BINARY lane,
LSO-1100. round2/ (a different, already-analysed pre-registered family) is untouched.

Usage: cd <repo root> && PYTHONPATH=. python3 tools/compute-posthoc-sweep-table.py
"""
import ast
import datetime
import glob
import json
import os

import numpy as np
from sklearn.metrics import f1_score

from eval_utils.cluster_bootstrap import balanced_accuracy, paired_metric_cluster_bootstrap

SWEEP_DIR = 'sweep_results/posthoc_sensitivity'
CONTROL_PATH = f'{SWEEP_DIR}/B1_published_config_threads4_oof.json'
ENV_REFS = {'B1_published_config': f'{SWEEP_DIR}/B1_published_config_oof.json',
            'B1_published_config_threads1': f'{SWEEP_DIR}/B1_published_config_threads1_oof.json'}
PUBLISHED_PATH = 'sweep_results/relation_tokens/enc_isbclip_B1_lso-1100_headline-none_oof.json'
MANIFEST_PATH = 'kaggle/isb-clip-manifest.json'
SWEEP_SCRIPT = 'run_b1_sensitivity_sweep.py'
SCRIPT_PATH = 'tools/compute-posthoc-sweep-table.py'
OUT_PATH = 'sweep_results/review_261101/posthoc-sensitivity-full-table.json'
GRADCLIP_FILE = f'{SWEEP_DIR}/gradclip_5.0_oof.json'
CH3, CH4 = 'report/vn/chuong-03-phuong-phap.txt', 'report/vn/chuong-04-chi-tiet-thuc-nghiem.txt'
DRIFT_BAND_PP = 1.46
N_BOOT, SEED, ALPHA, Q_BH = 5000, 0, 0.05, 0.05
LANE = ('iSafetyBench, 1,100 clips / 151 sources, binary safe(0)/unsafe(1), leave-source-out '
        '(LSO-1100) -- the whole-frame/B1 lane. Cross-lane comparison is forbidden (thesis Sec. '
        '6.1.1): never set these numbers beside the 64-/516-clip 4-class Unsafe-Net lane.')

def load_by_clip(path):
    rec = json.load(open(path))
    return rec, {r['clip_id']: r for r in rec['rows']}

def aligned_arrays(rows_a, rows_b, clip_order):
    y = np.array([rows_a[c]['y_true'] for c in clip_order])
    assert np.array_equal(y, [rows_b[c]['y_true'] for c in clip_order]), 'label mismatch'
    return y, np.array([rows_a[c]['y_pred'] for c in clip_order]), np.array([rows_b[c]['y_pred'] for c in clip_order])

def confusion_counts(y, pred):
    return {'tp': int(((y == 1) & (pred == 1)).sum()), 'tn': int(((y == 0) & (pred == 0)).sum()),
            'fp': int(((y == 0) & (pred == 1)).sum()), 'fn': int(((y == 1) & (pred == 0)).sum())}

def paired_draws(y, pred_a, pred_b, groups, n_boot=N_BOOT, seed=SEED):
    """Same algorithm/seed as eval_utils.cluster_bootstrap.paired_metric_cluster_bootstrap, but
    surfaces the raw per-draw differences too, so a p-value reads off the SAME draws as the CI
    (precedent: analyze_round3_tower_bh.py's local paired_draws/asl)."""
    uniq = np.unique(groups)
    idx_by_group = [np.where(groups == g)[0] for g in uniq]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_group[p] for p in rng.integers(0, len(uniq), len(uniq))])
        if len(np.unique(y[idx])) < 2:
            continue
        out.append((balanced_accuracy(y[idx], pred_a[idx]) - balanced_accuracy(y[idx], pred_b[idx])) * 100.0)
    return np.array(out)

def asl_pvalue(draws):
    b = len(draws)
    if b == 0:
        return float('nan')
    return float(max(min(2 * min((draws <= 0).sum(), (draws >= 0).sum()) / b, 1.0), 1.0 / b))

def quote_lines(path, start, end):
    """1-indexed inclusive line range, verbatim (incl. Vietnamese diacritics), joined by space."""
    lines = open(path, encoding='utf-8').read().splitlines()
    return ' '.join(l.strip() for l in lines[start - 1:end])

def benjamini_hochberg(pairs, m, q=Q_BH):
    ordered = sorted(pairs, key=lambda kv: kv[1])
    k_max = 0
    for i, (_, p) in enumerate(ordered, 1):
        if p <= q * i / m:
            k_max = i
    return {k for k, _ in ordered[:k_max]}

def build_config_row(path, ctrl_rec, ctrl_rows, clip_order, groups, ba_pub):
    rec, rows = load_by_clip(path)
    y, pred_a, pred_c = aligned_arrays(rows, ctrl_rows, clip_order)
    ba, mf1 = 100 * balanced_accuracy(y, pred_a), 100 * f1_score(y, pred_a, average='macro')
    diff, lo, hi, n_valid = paired_metric_cluster_bootstrap(
        y, pred_a, pred_c, groups, metric=balanced_accuracy, n_boot=N_BOOT, seed=SEED, alpha=ALPHA)
    p_val = asl_pvalue(paired_draws(y, pred_a, pred_c, groups))
    name = rec['config']
    return rec, {'config': name, 'overrides': rec['overrides'],
                 'base_value': {k: ctrl_rec['full_config'][k] for k in rec['overrides']},
                 'n': int(len(y)), 'ba': ba, 'macro_f1': mf1, 'confusion': confusion_counts(y, pred_a),
                 'display': f'{name}: BA={ba:.2f}%, macro-F1={mf1:.2f}%',
                 'margin_pp_vs_control': diff, 'margin_pp_vs_published_84_17': ba - ba_pub,
                 'cluster_bootstrap_ci95_pp_vs_control': [lo, hi], 'bootstrap_valid_draws': n_valid,
                 'p_value_two_sided_asl': p_val, 'ci_excludes_zero': bool(lo > 0 or hi < 0),
                 'outside_drift_band_1_46pp': bool(abs(diff) > DRIFT_BAND_PP),
                 'beats_published_headline': bool(ba > ba_pub)}

def main():
    manifest = json.load(open(MANIFEST_PATH))
    cid_to_source = dict(zip(manifest['clip_ids'], manifest['views']))
    assert len(set(manifest['views'])) == 151, 'canonical grouping must be 151 sources'
    ctrl_rec, ctrl_rows = load_by_clip(CONTROL_PATH)
    clip_order = [r['clip_id'] for r in ctrl_rec['rows']]
    groups = np.array([cid_to_source[c] for c in clip_order])
    y_ctrl = np.array([ctrl_rows[c]['y_true'] for c in clip_order])
    pred_ctrl = np.array([ctrl_rows[c]['y_pred'] for c in clip_order])
    ba_ctrl = 100 * balanced_accuracy(y_ctrl, pred_ctrl)
    _, pub_rows = load_by_clip(PUBLISHED_PATH)
    y_pub, _, pred_pub = aligned_arrays(ctrl_rows, pub_rows, clip_order)
    ba_pub = 100 * balanced_accuracy(y_pub, pred_pub)

    config_paths = sorted(p for p in glob.glob(f'{SWEEP_DIR}/*_oof.json')
                           if 'B1_published_config' not in os.path.basename(p))
    configs, pairs = [], []
    source_artifacts = [CONTROL_PATH, PUBLISHED_PATH, MANIFEST_PATH, SWEEP_SCRIPT, CH3, CH4]
    for path in config_paths:
        source_artifacts.append(path)
        _, row = build_config_row(path, ctrl_rec, ctrl_rows, clip_order, groups, ba_pub)
        configs.append(row); pairs.append((row['config'], row['p_value_two_sided_asl']))
    m = len(pairs)
    rejected = benjamini_hochberg(pairs, m)
    for row in configs:
        row['bh_q05_rejected'] = row['config'] in rejected
        row['candidate_effect'] = bool(row['outside_drift_band_1_46pp'] and row['ci_excludes_zero']
                                        and row['bh_q05_rejected'])
    configs.sort(key=lambda r: -r['margin_pp_vs_control'])

    env_refs = {}
    for tag, path in ENV_REFS.items():
        source_artifacts.append(path)
        _, rows = load_by_clip(path)
        y, pred_a, _ = aligned_arrays(rows, ctrl_rows, clip_order)
        env_refs[tag] = {'ba': 100 * balanced_accuracy(y, pred_a), 'artifact': path}

    doc = ast.get_docstring(ast.parse(open(SWEEP_SCRIPT).read()))
    i0 = doc.find('Row `B1_published_config`')
    i1 = doc.find('comparing two different run environments.') + len('comparing two different run environments.')
    sweep_docstring_quote = ' '.join(doc[i0:i1].split())

    gradclip_rec = json.load(open(GRADCLIP_FILE))
    gradclip_check = {
        'file': GRADCLIP_FILE, 'exists': True,
        'mtime_iso': datetime.datetime.fromtimestamp(os.path.getmtime(GRADCLIP_FILE)).isoformat(),
        'overrides': gradclip_rec['overrides'], 'ba': gradclip_rec['ba'],
        'margin_pp_vs_control': next(r['margin_pp_vs_control'] for r in configs if r['config'] == 'gradclip_5.0'),
        'claim_checked': 'two chapters state no clip=1.0-vs-5.0 comparison exists on the same data',
        'falsified_chapter_lines': [
            {'file': CH3, 'lines': '725-729', 'quote': quote_lines(CH3, 725, 729)},
            {'file': CH4, 'lines': '588-590', 'quote': quote_lines(CH4, 588, 590)}],
        'note': 'CONTRADICTED by this artifact: gradclip_5.0_oof.json is a clip=1.0 (control) vs clip=5.0 run on the SAME features/folds/recipe otherwise -- exactly the comparison both chapters say does not exist. The margin is INSIDE the drift band, so no gate verdict changes -- but the "no comparison exists" sentence in both chapters is false on the repo as it stands. Chapters NOT edited by this script; see existing_files_that_look_wrong in the agent report.'}

    beats_headline = sorted([r for r in configs if r['beats_published_headline']],
                             key=lambda r: -r['margin_pp_vs_published_84_17'])
    out = {
        'note': 'POST-FREEZE (computed 2026-11-01, after the 2026-10-05 number freeze) and POST-HOC / NOT A PRE-REGISTERED GATE: may be used to disclose that undisclosed post-hoc configs exist, some above and some below the published headline, and whether each margin clears measurement noise; may NOT be used to re-adjudicate any pre-registered gate, replace any Chapter 6 row, or claim a hyperparameter search occurred before the architecture freeze.',
        'lane': LANE, 'resampling_unit': 'source_video', 'n_boot': N_BOOT, 'seed': SEED, 'alpha': ALPHA,
        'source_artifacts': sorted(set(source_artifacts)), 'script': SCRIPT_PATH,
        'results': {
            'control': {'artifact': CONTROL_PATH, 'ba': ba_ctrl,
                        'why_control_not_published': 'Thesis Sec. 4.5.6 measures a 1.46 pp environment-drift band (1-thread 84.71 / published 84.17 / 4-thread 83.25) from CPU-thread-dependent checkpoint-epoch selection on a saturated val_macro. Every row of this sweep ran at OMP_NUM_THREADS=4, so the only same-environment comparator is the 4-thread control; comparing a 4-thread row to the published 84.17 mixes two run environments and misreads drift as a hyperparameter effect.'},
            'published_reference': {'artifact': PUBLISHED_PATH, 'ba': ba_pub,
                                     'used_for': 'disclosure margin only, never as the comparator'},
            'environment_drift_band_pp': DRIFT_BAND_PP, 'environment_reference_runs': env_refs,
            'multiple_comparisons': {'method': 'Benjamini-Hochberg', 'q': Q_BH, 'm': m, 'pre_registered': False,
                                      'note': f'POST-HOC family: m={m} is the count of sweep config files found on disk at analysis time, not a denominator fixed before the sweep ran. A single surviving row is exploratory, not a finding.'},
            'sweep_script': SWEEP_SCRIPT, 'sweep_script_docstring_quote': sweep_docstring_quote,
            'configs': configs,
            'grouped_by_drift_band': {
                'outside_band_1_46pp': [r['config'] for r in configs if r['outside_drift_band_1_46pp']],
                'inside_band_uninterpretable_as_signal': [r['config'] for r in configs if not r['outside_drift_band_1_46pp']]},
            'beats_published_headline_84_17': [
                {'config': r['config'], 'ba': r['ba'], 'margin_pp': r['margin_pp_vs_published_84_17'],
                 'display': f"{r['config']} beats published 84.17% by {r['margin_pp_vs_published_84_17']:.2f} pp"}
                for r in beats_headline],
            'gradclip_1v5_contradiction_check': gradclip_check,
        },
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump(out, open(OUT_PATH, 'w'), indent=1)
    print(f'control (4-thread) BA={ba_ctrl:.4f}%  published BA={ba_pub:.4f}%  m={m}')
    for r in configs:
        print(f"{r['config']:16s} BA={r['ba']:6.2f}  margin_ctrl={r['margin_pp_vs_control']:+6.2f}pp  "
              f"p={r['p_value_two_sided_asl']:.4f}  band={'OUT' if r['outside_drift_band_1_46pp'] else 'in'}  "
              f"BH={'rej' if r['bh_q05_rejected'] else 'no'}")
    print(f'wrote {OUT_PATH}')

if __name__ == '__main__':
    main()
