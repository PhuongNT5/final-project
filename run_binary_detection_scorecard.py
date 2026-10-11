#!/usr/bin/env python3
"""Re-express the NP-125 binary run as a DETECTION measurement: positive class = unsafe.

Pre-registration (written before any number here existed):
  plans/261011-1502-baseline-giao-thuc-bai-bao-goc-unsafe-net/tien-dang-ky-np125d-phat-hien-nhi-phan.md

G25 scored balanced accuracy, which prices one missed unsafe clip exactly like one false alarm on a
safe clip.  The stated objective is detecting the unsafe, so this script prints the asymmetric
quantities instead: recall, false-alarm rate, precision and F1 with unsafe as the positive class.

Trains nothing and runs no new inference: it re-reads the SAME `Pt` the single spent test touch
produced.  Every threshold is fixed on the 103-clip val slice before test is read at it (§2-3 of the
pre-registration); no threshold is ever chosen on test.
"""
from __future__ import annotations
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from eval_utils.cluster_bootstrap import metric_cluster_bootstrap
from eval_utils.detection_metrics import (counts, duplicate_floor, fpr_fn, per_unsafe_class_recall,
                                          pick_thresholds, POS, recall_fn, scene_alarm_profile,
                                          within_scene_auc)
from run_convlstm_paper_baseline import PLAN, SCENE_MAP, scene_controls
from unsafenet import config as C, metrics_stats as S

ART = os.path.join(C.OUT_DIR, 'temporal_convlstm_binary.pt')
AUDIT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'sweep_results', 'duplicate_audit_binary.json')
def main():
    d = torch.load(ART, map_location='cpu', weights_only=False)
    yt = np.asarray(d['yt']); st = np.asarray(d['Pt'])[:, POS]; names = list(d['test_names'])
    yv = np.asarray(d['yv']); sv = np.asarray(d['Pv'])[:, POS]; vnames = list(d['val_names'])
    policy = d.get('policy', 'none')

    ctrl_v, _ = scene_controls(vnames, yv, policy)
    ctrl_t, sid = scene_controls(names, yt, policy)
    scene_v = ctrl_v['scene_majority_train_fit']
    scene_t = ctrl_t['scene_majority_train_fit']

    taus = pick_thresholds(sv, yv, counts(yv, scene_v)['false_alarm_rate'])
    ops = {}
    for k, t in taus.items():
        dec = (st >= t).astype(int)
        c = counts(yt, dec)
        r = metric_cluster_bootstrap(yt, dec, sid, recall_fn)
        f = metric_cluster_bootstrap(yt, dec, sid, fpr_fn)
        ops[k] = {'tau_from_val': t, 'val': counts(yv, (sv >= t).astype(int)), 'test': c,
                  'per_unsafe_class_recall': per_unsafe_class_recall(names, yt, dec),
                  'label_induced_false_alarms': duplicate_floor(names, yt, dec, AUDIT),
                  'recall_ci_cluster_8_scene_groups': [r[1], r[2]],
                  'recall_ci_n_valid_draws': r[3],
                  'false_alarm_ci_cluster_8_scene_groups': [f[1], f[2]],
                  'false_alarm_ci_n_valid_draws': f[3],
                  'recall_ci_CLIP_UNIT_wrong_unit': list(
                      S.wilson_interval(c['tp'], c['tp'] + c['fn'])),
                  'mcnemar_unsafe_decision_vs_scene_only': S.mcnemar_exact(
                      yt, dec, scene_t)}

    prim = ops['OP3_val_alarm_matched_to_scene']
    dec_prim = (st >= taus['OP3_val_alarm_matched_to_scene']).astype(int)
    margin = (prim['test']['recall_unsafe'] - counts(yt, scene_t)['recall_unsafe']) * 100
    p3 = prim['mcnemar_unsafe_decision_vs_scene_only']['p']
    out = {
        'preregistration': 'tien-dang-ky-np125d-phat-hien-nhi-phan.md',
        'relation_to_G25': 'RE-EXPRESSION, NON-ADJUDICATING. Cannot overturn or soften G25.',
        'new_test_touches': 0, 'positive_class': 'unsafe', 'artifact': os.path.relpath(ART),
        'n': int(len(yt)), 'n_unsafe': int((yt == POS).sum()), 'n_safe': int((yt != POS).sum()),
        'scene_groups_covering_test': int(len(np.unique(sid))),
        'threshold_free': {'pr_auc_unsafe': float(average_precision_score(yt == POS, st)),
                           'roc_auc': float(roc_auc_score(yt == POS, st))},
        'within_scene_discrimination': within_scene_auc(yt, st, sid),
        'scene_alarm_profile_at_OP0': scene_alarm_profile(yt, (st >= 0.5).astype(int), sid),
        'operating_points': ops,
        'controls_in_detection_terms': {
            k: counts(yt, p) for k, p in ctrl_t.items()},
        'np64_slice': {'note': 'unsafe-only: only recall is defined, nothing else is printed',
                       'recall_unsafe_at_OP0': float(np.mean(
                           st[np.array([n.split('_')[0] in ('0', '1', '2', '3') for n in names])] >= 0.5))},
        'prereg_rule': 'has content iff at OP3 recall margin over scene-only >= 5 pp AND '
                       'McNemar on the unsafe decision p < 0.05',
        'verdict': {'recall_margin_pp_at_OP3': margin, 'mcnemar_p_at_OP3': p3,
                    'has_content': bool(margin >= 5.0 and p3 < 0.05)},
    }
    p = os.path.join(PLAN, 'ket-qua-np125d-phat-hien.json')
    json.dump(out, open(p, 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print('->', p)


if __name__ == '__main__':
    main()
