#!/usr/bin/env python3
"""Cluster-bootstrap CI for every iSafetyBench binary-lane headline row (Bang 6.1 / "Table 3").

GAP closed: the thesis prints "84.2%" and its six table-mates as bare one-decimal constants with
no interval. Recomputes each row from its saved out-of-fold predictions; reports a cluster
bootstrap CI resampled by SOURCE VIDEO (correct unit -- clips from one source share a scene, so
clip-level resampling treats near-duplicates as independent evidence and is too narrow) alongside
a clip-level "WRONG_UNIT" bootstrap for disclosure.
Lane: iSafetyBench, binary (unsafe=1/normal=0), 1,100 clips, 151 sources; never touches the
64-/516-clip Unsafe-Net lane (thesis Sec. 6.1.1 cross-lane ban).
Canonical clip -> source grouping = kaggle/isb-clip-manifest.json 'views' joined by clip_id (NOT a
filename regex, which gives 152 and is wrong); every oof file's fold_id is cross-checked against
this, mismatch count reported per row.
Run: cd <repo root> && PYTHONPATH=. python3 tools/compute-headline-cluster-ci.py
Deterministic (seed=0, n_boot=5000) -> byte-identical artifact on re-run.
"""
import json
import os
import numpy as np
from eval_utils.cluster_bootstrap import balanced_accuracy, metric_cluster_bootstrap

REL = 'sweep_results/relation_tokens'
MANIFEST = 'kaggle/isb-clip-manifest.json'
OUT = 'sweep_results/review_261101/headline-cluster-ci.json'
SCRIPT = 'tools/compute-headline-cluster-ci.py'
N_BOOT, SEED, ALPHA, POSITIVE, TOL_PP = 5000, 0, 0.05, 1, 0.05

# (label, filename, printed Bang 6.1 anchor, thesis source line); anchors hardcoded from a one-time
# manual read of chuong-06, every number still independently recomputed below and asserted to agree.
ROWS = [
    ('B1', 'enc_isbclip_B1_lso-1100_headline-none_oof.json',
     dict(ba=84.2, macro_f1=83.5, recall_unsafe=84.5, precision_unsafe=76.3), 'chuong-06-danh-gia.txt:300'),
    ('B1w', 'enc_isbclip_B1w_lso-1100_headline-none_oof.json',
     dict(ba=84.4, macro_f1=83.6, recall_unsafe=84.8, precision_unsafe=76.6), 'chuong-06-danh-gia.txt:299'),
    ('B1xl', 'enc_isbclip_B1xl_lso-1100_headline-none_oof.json',
     dict(ba=82.7, macro_f1=82.0, recall_unsafe=82.6, precision_unsafe=74.8), 'chuong-06-danh-gia.txt:303'),
    ('B2', 'enc_isbclip_B2_lso-1100_headline-none_oof.json',
     dict(ba=81.5, macro_f1=80.7, recall_unsafe=82.1, precision_unsafe=72.6), 'chuong-06-danh-gia.txt:305'),
    ('ESTA', 'enc_isbclip_ESTA_lso-1100_headline-none_oof.json',
     dict(ba=83.8, macro_f1=83.1, recall_unsafe=84.3, precision_unsafe=75.8), 'chuong-06-danh-gia.txt:302'),
    ('TSF', 'enc_isbclip_TSF_lso-1100_headline-none_oof.json',
     dict(ba=84.1, macro_f1=83.1, recall_unsafe=86.0, precision_unsafe=74.9), 'chuong-06-danh-gia.txt:301'),
    # Several similarly-named files exist (_y26_/_26ft_/_v11ft_/_smoke_); identity settled by BA match.
    ('ESTA_at_YOLO26s', 'enc_isbclip_ESTA_y26_lso-1100_headline-none_oof.json',
     dict(ba=81.6, macro_f1=80.3, recall_unsafe=85.0, precision_unsafe=70.7), 'chuong-06-danh-gia.txt:304'),
]
# LEAKING random-split row the thesis declined to publish as a result (Sec 6.2.7); only BA is printed.
RANDOM_ROW = ('B1_random_split_LEAKING', 'enc_isbclip_B1_rand-1100_headline-none_oof.json',
              dict(ba=97.7), 'chuong-06-danh-gia.txt:50,347-348')
# Candidates rejected for the ESTA@YOLO26s slot: recomputed BA matches no Bang 6.1 row.
REJECTED_DETECTOR_CANDIDATES = ['enc_isbclip_ESTA_26ft_lso-1100_headline-none_oof.json',
                                 'enc_isbclip_ESTA_v11ft_lso-1100_headline-none_oof.json',
                                 'enc_isbclip_ESTA_smoke_lso-1100_headline-none_oof.json']

def macro_f1(y_true, y_pred):
    """Mean per-class F1 over classes present in y_true (mirrors balanced_accuracy's convention)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = []
    for c in np.unique(y_true):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else float('nan')

def ba_pct(y, p): return balanced_accuracy(y, p) * 100.0
def f1_pct(y, p): return macro_f1(y, p) * 100.0

def confusion(y, p, positive=POSITIVE):
    tp = int(((p == positive) & (y == positive)).sum())
    fp = int(((p == positive) & (y != positive)).sum())
    fn = int(((p != positive) & (y == positive)).sum())
    tn = int(((p != positive) & (y != positive)).sum())
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

def load_group_map():
    man = json.load(open(MANIFEST))
    assert len(man['clip_ids']) == len(man['views']) == 1100
    return dict(zip(man['clip_ids'], man['views'])), len(set(man['views']))

def load_row(fname, group_map):
    d = json.load(open(os.path.join(REL, fname)))
    rows = d['rows']
    cids = [r['clip_id'] for r in rows]
    missing = [c for c in cids if c not in group_map]
    if missing:
        raise ValueError(f'{fname}: {len(missing)} clip_ids absent from canonical manifest grouping')
    y = np.array([r['y_true'] for r in rows], dtype=int)
    p = np.array([r['y_pred'] for r in rows], dtype=int)
    groups = np.array([group_map[c] for c in cids])
    bad_fold = sum(1 for r in rows if group_map[r['clip_id']] != str(r['fold_id']))
    return y, p, groups, bad_fold, d.get('model'), d.get('protocol'), len(rows)

def ci_block(y, p, groups, metric):
    pt_c, lo_c, hi_c, nv_c = metric_cluster_bootstrap(y, p, groups, metric, N_BOOT, SEED, ALPHA)
    pt_w, lo_w, hi_w, nv_w = metric_cluster_bootstrap(y, p, np.arange(len(y)).astype(str), metric, N_BOOT, SEED, ALPHA)
    width_c, width_w = hi_c - lo_c, hi_w - lo_w
    return {
        'point_pp': pt_c,
        'cluster_bootstrap_source_video': {
            'unit': 'source_video', 'lo_pp': lo_c, 'hi_pp': hi_c, 'width_pp': width_c,
            'n_valid_draws': nv_c, 'n_boot': N_BOOT,
            'display': f'{pt_c:.1f}% [{lo_c:.1f}, {hi_c:.1f}] (width {width_c:.1f} pp)'},
        'clip_level_bootstrap_WRONG_UNIT': {
            'unit': 'clip (singleton groups) -- WRONG UNIT, disclosure only',
            'lo_pp': lo_w, 'hi_pp': hi_w, 'width_pp': width_w, 'n_valid_draws': nv_w, 'n_boot': N_BOOT,
            'display': f'{pt_w:.1f}% [{lo_w:.1f}, {hi_w:.1f}] (width {width_w:.1f} pp)'},
        'width_ratio_correct_over_wrong_unit': (width_c / width_w) if width_w else None,
    }

def build_row(label, fname, anchor, anchor_src, group_map):
    y, p, groups, bad_fold, model, protocol, n = load_row(fname, group_map)
    ba, f1 = ba_pct(y, p), f1_pct(y, p)
    conf = confusion(y, p)
    recall_u = conf['tp'] / (conf['tp'] + conf['fn']) * 100.0
    precision_u = conf['tp'] / (conf['tp'] + conf['fp']) * 100.0
    computed = {'ba': ba, 'macro_f1': f1, 'recall_unsafe': recall_u, 'precision_unsafe': precision_u}
    anchor_check = {k: abs(computed[k] - v) <= TOL_PP for k, v in anchor.items()}
    if not all(anchor_check.values()):
        raise ValueError(f'{label} ({fname}): recomputed value disagrees with thesis-printed anchor '
                         f'{anchor} -> computed {computed}, checks {anchor_check}')
    row = {
        'label': label, 'source_file': fname, 'source_model_field': model,
        'source_protocol_field': protocol, 'n_clips': n, 'n_groups': int(len(set(groups))),
        'n_fold_id_mismatches_vs_manifest': bad_fold,
        'balanced_accuracy_pct': ba, 'macro_f1_pct': f1,
        'recall_unsafe_pct': recall_u, 'precision_unsafe_pct': precision_u,
        'confusion_unsafe_positive': conf,
        'thesis_printed_anchor': {**anchor, 'source': anchor_src}, 'anchor_match': anchor_check,
        'ci_balanced_accuracy': ci_block(y, p, groups, ba_pct),
        'ci_macro_f1': ci_block(y, p, groups, f1_pct),
    }
    return row, ba

def main():
    group_map, n_sources = load_group_map()
    assert n_sources == 151, f'expected 151 canonical sources, got {n_sources}'
    rows_out, ba_by_label = {}, {}
    for label, fname, anchor, src in ROWS + [RANDOM_ROW]:
        rows_out[label], ba_by_label[label] = build_row(label, fname, anchor, src, group_map)
    rfname = RANDOM_ROW[1]
    rejected = []
    for fname in REJECTED_DETECTOR_CANDIDATES:
        y, p, _, _, _, _, _ = load_row(fname, group_map)
        rejected.append({'file': fname, 'recomputed_ba_pct': ba_pct(y, p),
                         'reason': 'recomputed BA matches no printed Bang 6.1 row (incl. 81.6% for ESTA@YOLO26s); excluded as an unrelated/ambiguous detector run'})
    half_b1 = rows_out['B1']['ci_balanced_accuracy']['cluster_bootstrap_source_video']['width_pp'] / 2.0
    effects = {
        'B1w_minus_B1_pp': ba_by_label['B1w'] - ba_by_label['B1'],
        'ESTA_minus_B1w_pp': ba_by_label['ESTA'] - ba_by_label['B1w'],
        'TSF_minus_B1_pp': ba_by_label['TSF'] - ba_by_label['B1'],
        'B1xl_minus_B1_pp': ba_by_label['B1xl'] - ba_by_label['B1'],
        'detector_swap_delta_pp_ESTAv11s_minus_ESTAy26': ba_by_label['ESTA'] - ba_by_label['ESTA_at_YOLO26s'],
    }
    swallowed = {k: {'effect_pp': v, 'abs_effect_pp': abs(v), 'half_width_B1_cluster_ci_pp': half_b1,
                     'swallowed_by_B1_noise': abs(v) <= half_b1,
                     'lane': 'iSafetyBench binary, 1,100 clips, LSO-1100 (same table, computed here)'}
                for k, v in effects.items()}
    swallowed['largest_encoder_sweep_margin_pp_B1w_minus_E0'] = {
        'effect_pp': 4.00, 'abs_effect_pp': 4.00, 'half_width_B1_cluster_ci_pp': half_b1,
        'swallowed_by_B1_noise': 4.00 <= half_b1, 'measured_here': False,
        'lane': 'iSafetyBench binary, 1,100 clips -- encoder-representation-sweep sub-track (E0-E3), NOT the Bang 6.1 headline family; value cited verbatim from report/vn/chuong-06-danh-gia.txt:458 as a magnitude reference, not recomputed here'}

    artifact = {
        'note': 'POST-FREEZE (computed 2026-11-01, after the 2026-10-05 number freeze), POST-HOC, NOT A PRE-REGISTERED GATE: adds the missing cluster-bootstrap CI around every Bang 6.1 / "Table 3" row. May be used to show 84.2% and its table-mates carry no reported interval and that the correct source-video interval is far wider than a naive clip-level one; may NOT be used to assert a new pass/fail verdict on any pre-registered gate, revise the G14-G18 adjudications, or compare across the iSafetyBench/Unsafe-Net lane boundary.',
        'lane': 'iSafetyBench, binary (unsafe=1/normal=0), 1,100 clips, 151 source videos, leave-source-out (LSO-1100) except the explicitly-labelled random-split row',
        'resampling_unit': 'source_video (kaggle/isb-clip-manifest.json "views", joined by clip_id -- not a filename regex); clip-level (WRONG_UNIT) bootstrap also reported per row, disclosure only, singleton per-clip groups, same B/seed/alpha',
        'n_boot': N_BOOT, 'seed': SEED, 'alpha': ALPHA,
        'source_artifacts': ([MANIFEST] + [os.path.join(REL, f) for _, f, _, _ in ROWS]
                             + [os.path.join(REL, rfname)]
                             + [os.path.join(REL, f) for f in REJECTED_DETECTOR_CANDIDATES]),
        'script': SCRIPT,
        'results': {
            'headline_table_rows': rows_out,
            'detector_variant_identification': {
                'kept': 'enc_isbclip_ESTA_y26_lso-1100_headline-none_oof.json',
                'kept_recomputed_ba_pct': ba_by_label['ESTA_at_YOLO26s'],
                'printed_anchor_pct': 81.6, 'rejected_candidates': rejected},
            'swallowed_effects': swallowed,
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(artifact, f, indent=2, sort_keys=False)
        f.write('\n')
    print(f'[ok] wrote {OUT}')

if __name__ == '__main__':
    main()
