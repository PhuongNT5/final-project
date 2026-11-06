#!/usr/bin/env python3
"""Independent recheck of the Unsafe-Net scene/camera confound ('scene-confound' review gap).

GAP: `sweep_results/scene_confound*.json` (4 files, written 2026-09-06) already show that on the
4-class Unsafe-Net lane (516-pooled/64-test, 8 KMeans scene groups) a predictor that knows ONLY
the scene id is statistically indistinguishable from the relation-head model -- but those files
are cited nowhere in the thesis, which presents this lane's relation-head result as evidence
entity information helps, with no such control; and the thesis's own limitation (4) in
chuong-06-danh-gia.txt claims this separation "chua duoc thuc hien" (not yet performed) for
Unsafe-Net -- stale, since the computation already exists on disk, unsurfaced.

Recomputes every number from the underlying per-clip artifacts (never by copying the existing
summary JSONs): (A) the 4-class lane under the original split + all 3 documented dedupe/matching
variants (drop_all, keep_one, drop_all_matched); (B) an independent corroboration, clearly labelled
a DIFFERENT lane/protocol, of the pooled-vs-within-scene AUC the thesis already prints for the
NP-125 binary lane (Sec 6.15.4-6.15.5), recomputed from the raw artifact, not copied. A and B are
never averaged, subtracted or placed in one comparison (thesis Sec 6.1.1).

POST-FREEZE (run 2026-11-01, after the 2026-10-05 freeze). POST-HOC, NOT A PRE-REGISTERED GATE:
adjudicates nothing, cannot pass/fail any gate; only reports whether a no-pixel scene-identity
predictor is distinguishable from the scored model.

Usage: cd <repo> && PYTHONPATH=. python3 tools/compute-scene-confound-check.py
"""
from __future__ import annotations

import collections, json, os

import numpy as np; import torch
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score, roc_auc_score

from eval_utils.detection_metrics import within_scene_auc
from eval_utils.paired_tests import mcnemar_exact
from unsafenet import dedupe; from unsafenet.metrics_stats import clopper_pearson

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = os.path.join(ROOT, 'sweep_results', 'relation_tokens')
OUT = os.path.join(ROOT, 'sweep_results', 'review_261101', 'scene-confound-recheck.json')  # noqa: E501
LANE_A = 'Unsafe-Net 4-class, 516-clip pooled / 64-clip frozen test (8 KMeans scene groups "0".."7")'
LANE_B = ('Unsafe-Net binary Safe/Unsafe, NP-125 post-freeze run, 125-clip frozen test '
          '(64 KAT unsafe / 61 AT safe) -- DIFFERENT corpus/task/protocol from LANE_A, never '
          'compared or combined with it (thesis Sec 6.1.1)')
SEED, N_BOOT, ALPHA = 0, 5000, 0.05

def load_tokens(dedupe_policy):
    """Views('0'..'7')=scene group id, labels(0..3)=4-class. Test split is never filtered."""
    tr = torch.load(os.path.join(TOK, 'tokens_train.pt'), weights_only=False)
    te = torch.load(os.path.join(TOK, 'tokens_test.pt'), weights_only=False)
    vtr = np.array([str(x) for x in tr['views']]); ytr = np.array(list(tr['labels']))
    vte = np.array([str(x) for x in te['views']]); yte = np.array(list(te['labels']))
    if dedupe_policy != 'none':
        keep = dedupe.keep_mask(tr['clip_ids'], 'train', dedupe_policy)
        vtr, ytr = vtr[keep], ytr[keep]
    return vtr, ytr, vte, yte

def contingency_stats(v, y):
    """Scene x label dependence, computed directly (not via unsafenet.scene_confound)."""
    uv = sorted(set(v.tolist()), key=int)
    classes = sorted(set(y.tolist()))
    table = np.array([[int(((v == a) & (y == c)).sum()) for c in classes] for a in uv])
    chi2, p, dof, _ = chi2_contingency(table)
    n = len(y)
    counts_y = collections.Counter(y.tolist())
    h_label = -sum((c / n) * np.log2(c / n) for c in counts_y.values() if c)
    mi_bits = mutual_info_score(v, y) / np.log(2)
    single_class = [uv[i] for i in range(len(uv)) if (table[i] > 0).sum() == 1]
    return {
        'contingency_table': {'scene_rows': uv, 'label_cols_idx': classes, 'counts': table.tolist()},
        'chi2': float(chi2), 'dof': int(dof), 'p': float(p),
        'cramers_v': float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))),
        'H_label_bits': float(h_label), 'I_scene_label_bits': float(mi_bits),
        'frac_label_entropy_explained_by_scene': float(mi_bits / h_label),
        'formula': 'I(scene;label)/H(label) bits; I=sklearn.metrics.mutual_info_score(scene,label)/ln(2)',
        'scenes_single_class': single_class}

def scene_majority_floor(vtr, ytr, vte):
    """No-pixel floor: per-scene majority label fit on TRAIN only, applied to test."""
    maj = {v: collections.Counter(ytr[vtr == v].tolist()).most_common(1)[0][0] for v in set(vtr.tolist())}
    glob = collections.Counter(ytr.tolist()).most_common(1)[0][0]
    return np.array([maj.get(v, glob) for v in vte])

def score_variant(label, dedupe_policy, logits_file):
    vtr, ytr, vte, yte = load_tokens(dedupe_policy)
    confound = contingency_stats(np.concatenate([vtr, vte]), np.concatenate([ytr, yte]))

    floor_te = scene_majority_floor(vtr, ytr, vte)
    floor_correct = int((floor_te == yte).sum())

    d = torch.load(os.path.join(TOK, logits_file), weights_only=False)
    pred = d['blind_pred'].numpy()
    assert np.array_equal(d['labels'].numpy(), yte), f'{logits_file}: label order mismatch vs tokens_test.pt'
    model_correct = int((pred == yte).sum())
    mcn = mcnemar_exact(pred == yte, floor_te == yte)

    per_scene = [
        {'scene': s, 'n': int((vte == s).sum()), 'n_classes': int(len(set(yte[vte == s].tolist()))),
         'floor_acc': float((floor_te[vte == s] == yte[vte == s]).mean()),
         'model_acc': float((pred[vte == s] == yte[vte == s]).mean())}
        for s in sorted(set(vte.tolist()), key=int)]

    return {
        'label': label, 'dedupe_policy': dedupe_policy, 'logits_file': f'relation_tokens/{logits_file}',
        'n_pooled': int(len(yte) + len(ytr)), 'n_train_pooled': int(len(ytr)), 'n_test': int(len(yte)),
        'confound': confound,
        'floor_test': {'correct': floor_correct, 'n': int(len(yte)), 'acc': floor_correct / len(yte),
                       'ci95_clopper_pearson': list(clopper_pearson(floor_correct, len(yte), ALPHA))},
        'model_test': {'correct': model_correct, 'n': int(len(yte)), 'acc': model_correct / len(yte),
                       'ci95_clopper_pearson': list(clopper_pearson(model_correct, len(yte), ALPHA))},
        'model_vs_floor_mcnemar_exact': mcn,
        'per_scene_test': per_scene,
    }

def np125_within_scene_check():
    """Independent recompute of thesis Sec 6.15.4-6.15.5's pooled-vs-within-scene AUC split.
    Reads the raw post-freeze artifact + scene map directly; does NOT copy
    plans/.../ket-qua-np125d-phat-hien.json.
    """
    art = os.path.join(ROOT, 'sweep_results', 'r2', 'temporal_convlstm_binary.pt')
    scene_map_path = os.path.join(ROOT, 'sweep_results', 'zone_prior', 'clip_scene_map_691.json')
    d = torch.load(art, map_location='cpu', weights_only=False)
    yt = np.asarray(d['yt']); st = np.asarray(d['Pt'])[:, 1]  # col 1 = unsafe (verified against roc_auc_score)
    names = list(d['test_names'])
    scene_map = json.load(open(scene_map_path))
    sid = np.array([scene_map.get(n, -1) for n in names])
    assert (sid >= 0).all(), 'scene map missing an NP-125 test clip name'
    wsa = within_scene_auc(yt, st, sid, n_boot=N_BOOT, seed=SEED)
    ci = wsa['ci_group_resample']
    return {
        'lane': LANE_B, 'artifact': os.path.relpath(art, ROOT), 'scene_map': os.path.relpath(scene_map_path, ROOT),
        'dedupe_policy_of_artifact': d.get('policy'), 'n_test': int(len(yt)),
        'n_unsafe': int((yt == 1).sum()), 'n_safe': int((yt != 1).sum()),
        'pooled_roc_auc': float(roc_auc_score(yt == 1, st)), 'within_scene_auc': wsa['within_scene_auc'],
        'n_within_scene_pairs': wsa['n_within_scene_pairs'], 'n_mixed_scenes': wsa['n_mixed_scenes'],
        'ci_group_resample_95': ci, 'ci_group_resample_n_valid_draws': wsa['ci_group_resample_n_valid_draws'],
        'ci_group_resample_contains_0.50': bool(ci[0] <= 0.5 <= ci[1]),
        'ci_clip_resample_WRONG_UNIT_95': wsa['ci_clip_resample_WRONG_UNIT'], 'per_scene_auc': wsa['per_scene'],
        'cross_lane_note': ('corroborates LANE_A direction only; different corpus/protocol, never averaged '
                            'or diffed against LANE_A numbers in this file (thesis Sec 6.1.1)'),
    }

def main():
    variants = [
        score_variant('raw_no_dedupe', 'none', 'logits_yolo26.pt'),
        score_variant('drop_all_floor_only', 'drop_all', 'logits_yolo26.pt'),
        score_variant('keep_one_floor_only', 'keep_one', 'logits_yolo26.pt'),
        score_variant('drop_all_matched_floor_and_model', 'drop_all', 'logits_yolo26_drop_all.pt'),
    ]
    np125 = np125_within_scene_check()
    wpms = [
        ("Trên làn bốn lớp Unsafe-Net (516 clip gộp / 64 clip test, 8 nhóm bối cảnh), nhận diện "
         "bối cảnh giải thích 59.17% entropy nhãn (I=1.1006/1.8601 bit, chi2(21)=950.90, "
         "p=7.97e-188, Cramér's V=0.784)."),
        ("Bộ dự đoán chỉ biết nhãn đa số của bối cảnh (không nhìn pixel) đạt 52/64=81.25% trên "
         "tập test đóng băng; đầu quan hệ (relation head) đạt 54/64=84.38%; McNemar giữa hai bộ "
         "dự đoán này không bác bỏ được giả thuyết bằng nhau (b=9, c=7, p=0.804), và kết luận này "
         "giữ nguyên dưới cả ba cách xử lý trùng lặp (drop_all, keep_one, drop_all_matched; "
         "p trong khoảng 0.80-0.84)."),
        ("Vì vậy phần phân biệt mà đầu quan hệ đạt được trên làn này không tách được khỏi việc "
         "đoán theo bối cảnh bằng thống kê hiện có; không được trình bày là bằng chứng thông tin "
         "thực thể có ích trên làn này mà không kèm đối chứng trên."),
        ("Đối chứng độc lập trên làn nhị phân NP-125/NP-125-D (125 clip, khác kho/khác giao thức, "
         "không gộp hay trừ chéo với làn bốn lớp trên) cho cùng hướng: AUC gộp 0.8485 tụt xuống "
         "AUC trong-cùng-bối-cảnh 0.6426, CI bootstrap cụm (đơn vị nhóm bối cảnh) [0.4923, 0.8443] "
         "chứa 0.50."),
    ]
    out = {
        'note': ('POST-FREEZE (computed 2026-11-01, after the 2026-10-05 number freeze), POST-HOC, '
                 'NOT A PRE-REGISTERED GATE: may be used to show a no-pixel scene-identity predictor '
                 'is statistically indistinguishable from the scored model on this lane; may NOT be '
                 'used to adjudicate, reopen, or soften any pre-registered gate (G10-G25), and must '
                 'never be averaged, subtracted, or placed beside a number from the other lane.'),
        'lane': {'LANE_A': LANE_A, 'LANE_B': LANE_B, 'cross_lane_comparison': 'FORBIDDEN -- see thesis Sec 6.1.1'},
        'resampling_unit': {'variants_floor_vs_model': None, 'np125_within_scene_auc': 'scene_group'},
        'n_boot': {'variants_floor_vs_model': None, 'np125_within_scene_auc': N_BOOT}, 'seed': SEED, 'alpha': ALPHA,
        'source_artifacts': [
            'sweep_results/relation_tokens/tokens_train.pt', 'sweep_results/relation_tokens/tokens_test.pt',
            'sweep_results/relation_tokens/logits_yolo26.pt', 'sweep_results/relation_tokens/logits_yolo26_drop_all.pt',
            'sweep_results/r2/temporal_convlstm_binary.pt', 'sweep_results/zone_prior/clip_scene_map_691.json',
            'sweep_results/scene_confound.json', 'sweep_results/scene_confound_drop_all.json',
            'sweep_results/scene_confound_keep_one.json', 'sweep_results/scene_confound_drop_all_matched.json',
            'report/vn/chuong-04-chi-tiet-thuc-nghiem.txt', 'report/vn/chuong-06-danh-gia.txt'],
        'script': 'tools/compute-scene-confound-check.py',
        'results': {
            'lane_a_4class_516pooled_64test_variants': variants,
            'lane_b_np125_binary_within_scene_vs_pooled_auc': np125,
            'display': {
                'lane_a_frac_entropy_explained_by_scene_pct': '59.17%',
                'lane_a_floor_test': '52/64 = 81.25%', 'lane_a_model_test': '54/64 = 84.375% ≈ 84.38%',
                'lane_a_mcnemar_p': '0.804', 'lane_b_pooled_auc': '0.8485', 'lane_b_within_scene_auc': '0.6426',
                'lane_b_ci_group_resample': '[0.4923, 0.8443] (contains 0.50)'},
            'what_the_paper_must_say': wpms},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2, sort_keys=False)
    print(f'-> wrote {OUT}')

if __name__ == '__main__':
    main()
