#!/usr/bin/env python3
"""Score the paper-protocol ConvLSTM baseline: leg-1 4-class (G21) and binary Safe/Unsafe (G25).

Pre-registration (written before any of these numbers existed):
  plans/261011-1502-baseline-giao-thuc-bai-bao-goc-unsafe-net/tien-dang-ky-g21-g25.md

Reads only artifacts already written by `python -m unsafenet.convlstm`; trains nothing. Two tasks
are scored and must never be merged:
  G21  4-class unsafe subset, 64 test clips (all unsafe)  -> comparable to [17] Table 5/6
  G25  binary Safe/Unsafe, frozen 125-clip test set       -> the only split where BA is defined
`NP-64` is a SLICE of the binary run, not a binary measurement: on 64 unsafe-only clips the sole
defined quantity is unsafe recall.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch

from unsafenet import config as C, metrics as M, metrics_stats as S
from unsafenet.data_index import load_clips

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENE_MAP = os.path.join(ROOT, 'sweep_results', 'zone_prior', 'clip_scene_map_691.json')
AUDIT_BINARY = os.path.join(ROOT, 'sweep_results', 'duplicate_audit_binary.json')
PLAN = os.path.join(ROOT, 'plans', '261011-1502-baseline-giao-thuc-bai-bao-goc-unsafe-net')


def load(name):
    p = os.path.join(C.OUT_DIR, f'temporal_{name}.pt')
    if not os.path.exists(p):
        return None
    d = torch.load(p, map_location='cpu', weights_only=False)
    d['_path'] = os.path.relpath(p, ROOT)
    return d


def paper_macro(y, pred, n_cls):
    """[17]'s 'Overall (%)' = mean per-class sensitivity. NOT sklearn macro-F1."""
    rec = [float(np.mean(pred[y == k] == k)) for k in range(n_cls) if (y == k).any()]
    return float(np.mean(rec)), rec


def mcnemar_interval(n_total, correct_a, correct_b):
    """[min_p, max_p] over every feasible discordant split - used when the opponent's per-clip
    predictions do not exist (the paper never published theirs)."""
    import scipy.stats as st
    ps = [2 * st.binom.cdf(min(b, c), b + c, 0.5)
          for b, c in S.feasible_discordant_splits(n_total, correct_a, correct_b) if b + c]
    return [float(min(min(ps), 1.0)), float(min(max(ps), 1.0))] if ps else [1.0, 1.0]


def scene_controls(test_names, y_bin, policy):
    """Scene-only controls on the SAME clips the model was scored on.

    Grouping is `clip_scene_map_691.json` (KMeans on each clip's static background), NOT
    `clip_view_map*.json` - the latter clusters the same appearance features used for
    classification and run_lco_eval.py:132 flags it as circular.
    """
    scene = json.load(open(SCENE_MAP))
    tr = load_clips(['train'])
    from unsafenet import dedupe
    keep = dedupe.keep_mask([c.name for c in tr], 'train',
                            'none' if policy == 'none' else 'drop_conflict', task='binary')
    fit = [c for c, k in zip(tr, keep) if k]
    by_scene: dict = {}
    for c in fit:
        by_scene.setdefault(scene.get(c.name, -1), []).append(1 if c.is_unsafe else 0)
    train_major = {s: int(round(float(np.mean(v)))) for s, v in by_scene.items()}
    glob_major = int(round(float(np.mean([1 if c.is_unsafe else 0 for c in fit]))))
    sid = np.array([scene.get(n, -1) for n in test_names])
    honest = np.array([train_major.get(s, glob_major) for s in sid])
    oracle = np.zeros(len(y_bin), int)                      # per-scene majority read off TEST
    for s in set(sid.tolist()):
        m = sid == s
        oracle[m] = int(round(float(np.mean(y_bin[m]))))
    const = np.full(len(y_bin), int(np.mean(y_bin) >= 0.5))
    return {'scene_majority_train_fit': honest, 'scene_majority_oracle': oracle,
            'constant_majority_class': const}, sid


def np_bounds(names, y):
    """Majority floor and the hard duplicate ceiling for whatever test split is actually present.

    Both were hardcoded to the NP-125 values (64/125 and 124/125).  They are only correct for the
    frozen 125-clip set; `dedupe.drop_conflict_test` is an anticipated sensitivity split (120
    clips, 63/57) whose true floor is 63/120 and whose true ceiling is 1.0, because dropping the
    test members of the opposite-label groups removes the very pair that caps NP-125.  Derive, do
    not assume: the ceiling costs one clip per surplus member of every byte-identical group that
    lands wholly inside this split with disagreeing binary labels, since a deterministic
    classifier emits one label for identical input.
    """
    present = set(names)
    n = len(names)
    floor = max(int(y.sum()), int((1 - y).sum())) / n
    unattainable = 0
    for g in json.load(open(AUDIT_BINARY))['groups']:
        if g['kind'] != 'opposite':
            continue
        mem = [m for m in g['members'] if m['name'] in present]
        if len(mem) < 2:
            continue
        lab = [m['binary'] for m in mem]
        unattainable += len(mem) - max(lab.count('safe'), lab.count('unsafe'))
    return floor, (n - unattainable) / n


def score_binary(d, out):
    y = np.asarray(d['yt']); P = np.asarray(d['Pt']); pred = P.argmax(1)
    names = list(d['test_names'])
    rep = M.report(y, P, 'convlstm_binary', names=None, n_cls=2)
    ctrl, sid = scene_controls(names, y, d.get('policy', 'none'))
    floor, ceiling = np_bounds(names, y)
    out['binary_np125'] = {
        'artifact': d['_path'], 'device': d.get('device'), 'policy': d.get('policy'),
        'n': int(len(y)), 'n_unsafe': int(y.sum()), 'n_safe': int((1 - y).sum()),
        'correct': rep['correct'], 'top1': rep['top1'],
        'top1_ci_wilson': rep['top1_ci_wilson'],
        'balanced_accuracy': rep['balanced_accuracy'],
        'ba_ci_bootstrap_CLIP_UNIT_not_group': rep['macro_accuracy_ci_bootstrap'],
        'recall_safe_unsafe': rep['recall'], 'precision_safe_unsafe': rep['precision'],
        'macro_f1': rep['macro_f1'], 'mcc': rep['mcc'], 'kappa': rep['kappa'],
        'roc_auc': rep['roc_auc_macro'], 'pr_auc_unsafe': rep['pr_auc_per_class'][1],
        'ece': rep['ece'], 'brier': rep['brier'], 'confusion_rows_true_safe_unsafe': rep['confusion'],
        'floor_majority_class_top1': floor, 'ceiling_duplicate_conflict': ceiling,
        'val_macro': d['val_macro'], 'per_seed_val_macro': d['per_seed'],
    }
    out['binary_np125']['controls'] = {
        k: {'top1': float(np.mean(p == y)), 'correct': int((p == y).sum()),
            'balanced_accuracy': M.balanced_accuracy(y, p, 2),
            'mcnemar_vs_convlstm': S.mcnemar_exact(y, pred, p)}
        for k, p in ctrl.items()}
    # NP-64: the paper's 64 unsafe-only clips. Only unsafe recall is defined here.
    is64 = np.array([n.split('_')[0] in ('0', '1', '2', '3') for n in names])
    out['binary_np64_slice'] = {
        'n': int(is64.sum()), 'all_unsafe': bool(y[is64].all()),
        'unsafe_recall': float(np.mean(pred[is64] == 1)),
        'correct': int((pred[is64] == 1).sum()),
        'note': 'NOT a binary measurement: 0 safe clips, so specificity/BA/PR-AUC are undefined.'}
    return out


def score_unsafe4(d, tag, out):
    y = np.asarray(d['yt']); P = np.asarray(d['Pt']); pred = P.argmax(1)
    correct = int((pred == y).sum())
    macro, rec = paper_macro(y, pred, 4)
    rep = M.report(y, P, tag, names=list(d['test_names']), n_cls=4)
    out[tag] = {
        'artifact': d['_path'], 'device': d.get('device', 'mps (pre-2026-10-11 default)'),
        'n': int(len(y)), 'correct': correct, 'top1': correct / len(y),
        'paper_macro_mean_per_class_sensitivity': macro, 'per_class_sensitivity': rec,
        'per_class_n': np.bincount(y, minlength=4).tolist(),
        'macro_f1': rep['macro_f1'], 'top1_ci_wilson': rep['top1_ci_wilson'],
        'val_macro': d['val_macro'], 'per_seed_val_macro': d['per_seed'],
        'vs_paper_nobbox_57': {'their_correct': 57, 'their_macro': C.PAPER['nobbox']['macro'],
                               'delta_correct': correct - 57,
                               'mcnemar_feasible_interval': mcnemar_interval(64, correct, 57)},
        'vs_paper_full_62': {'their_correct': 62, 'their_macro': C.PAPER['full']['macro'],
                             'delta_correct': correct - 62,
                             'mcnemar_feasible_interval': mcnemar_interval(64, correct, 62)},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(PLAN, 'ket-qua-g21-g25.json'))
    v = ap.parse_args()
    out: dict = {'preregistration': 'tien-dang-ky-g21-g25.md', 'lane': 'C (Unsafe-Net) only'}
    for name, tag in [('convlstm', 'unsafe4_mps_2026_09_06'), ('convlstm_cpu', 'unsafe4_cpu_2026_10_11')]:
        d = load(name)
        if d is not None:
            score_unsafe4(d, tag, out)
    b = load('convlstm_binary')
    if b is not None:
        score_binary(b, out)
    a, c = out.get('unsafe4_mps_2026_09_06'), out.get('unsafe4_cpu_2026_10_11')
    if a and c:
        out['G21_device_control'] = {
            'delta_correct_cpu_minus_mps': c['correct'] - a['correct'],
            'delta_paper_macro_pp': 100 * (c['paper_macro_mean_per_class_sensitivity']
                                           - a['paper_macro_mean_per_class_sensitivity']),
            'c1_unauthorint_sensitivity_mps': a['per_class_sensitivity'][1],
            'c1_unauthorint_sensitivity_cpu': c['per_class_sensitivity'][1],
            'prereg_rule': 'reproduced iff top1 >= 54/64 AND mcnemar interval vs 57/64 includes '
                           'p >= 0.05; refuted iff top1 <= 47/64 or macro within +-3pp of 75.64%',
        }
    os.makedirs(os.path.dirname(v.out), exist_ok=True)
    json.dump(out, open(v.out, 'w'), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print(f'\n[written] {os.path.relpath(v.out, ROOT)}', flush=True)


if __name__ == '__main__':
    main()
