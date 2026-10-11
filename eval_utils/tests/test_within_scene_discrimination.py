"""Tests for the two context-conditioned diagnostics behind the NP-125-D finding.

`within_scene_auc` must return ~0.5 when the whole ranking is explained by the background context,
even where the pooled AUC is near-perfect: that gap is the reported result, so it is constructed
here from a case where the base rate alone does all the work.

Its interval, and `cluster_bootstrap.metric_cluster_bootstrap`, must resample whole SCENES. A
clip-level interval on grouped data is too narrow and is published only under its WRONG_UNIT label.

Usage: PYTHONPATH=. .venv/bin/python eval_utils/tests/test_within_scene_discrimination.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval_utils.cluster_bootstrap import metric_cluster_bootstrap
from eval_utils.detection_metrics import fpr_fn, recall_fn, within_scene_auc


def _pure_context_case():
    """Two scenes whose BASE RATE, not whose content, explains the ranking.

    Scene 0 is 6/7 unsafe and everything in it scores high; scene 1 is 6/7 safe and everything in
    it scores low.  Inside each scene the one minority clip sits in the middle of the majority, so
    the within-scene ordering is a coin flip while the pooled ordering looks strong.
    """
    y = np.array([1, 1, 1, 1, 1, 1, 0] + [0, 0, 0, 0, 0, 0, 1])
    sid = np.array([0] * 7 + [1] * 7)
    score = np.array([0.95, 0.92, 0.90, 0.80, 0.78, 0.75, 0.88]
                     + [0.30, 0.28, 0.25, 0.20, 0.12, 0.10, 0.22])
    return y, score, sid


def test_within_scene_auc_strips_the_context_channel_the_pooled_auc_rewards():
    from sklearn.metrics import roc_auc_score
    y, score, sid = _pure_context_case()
    pooled = roc_auc_score(y, score)
    out = within_scene_auc(y, score, sid, n_boot=200, seed=0)
    assert pooled > 0.85                                       # pooled: looks strong
    assert abs(out['within_scene_auc'] - 0.5) < 1e-12          # conditioned: exactly chance
    assert pooled - out['within_scene_auc'] > 0.30
    assert out['n_mixed_scenes'] == 2 and out['n_within_scene_pairs'] == 12


def test_within_scene_auc_skips_scenes_that_carry_one_label_only():
    y = np.array([1, 1, 0, 0, 1, 1])
    sid = np.array([0, 0, 0, 0, 1, 1])                         # scene 1 is all unsafe
    score = np.array([0.9, 0.8, 0.2, 0.1, 0.5, 0.4])
    out = within_scene_auc(y, score, sid, n_boot=200, seed=0)
    assert out['n_mixed_scenes'] == 1 and out['n_within_scene_pairs'] == 4


def test_group_resampled_interval_is_wider_than_the_clip_resampled_one():
    """The published interval must carry the scene as the unit; the clip one is kept only labelled.

    Half the scenes are ranked perfectly and half are ranked backwards, so which scenes a draw
    happens to contain dominates the uncertainty -- exactly the variance clip resampling hides.
    """
    sid = np.repeat(np.arange(8), 12)
    y = np.tile(np.array([1] * 6 + [0] * 6), 8)
    score = np.where(sid < 4, np.where(y == 1, 0.8, 0.2), np.where(y == 1, 0.2, 0.8))
    out = within_scene_auc(y, score, sid, n_boot=1000, seed=0)
    g_lo, g_hi = out['ci_group_resample']
    c_lo, c_hi = out['ci_clip_resample_WRONG_UNIT']
    assert abs(out['within_scene_auc'] - 0.5) < 1e-12
    assert g_hi - g_lo > 3 * (c_hi - c_lo)
    assert out['ci_group_resample_n_valid_draws'] > 0


def test_metric_cluster_bootstrap_resamples_groups_not_clips():
    """With one group per label block, group resampling can lose a class; clip resampling cannot."""
    y = np.array([1] * 20 + [0] * 20)
    dec = np.array([1] * 16 + [0] * 4 + [1] * 6 + [0] * 14)
    groups = np.repeat(np.arange(4), 10)
    point, lo, hi, n = metric_cluster_bootstrap(y, dec, groups, recall_fn, n_boot=500, seed=0)
    assert abs(point - 0.8) < 1e-12
    assert lo <= point <= hi and 0 < n <= 500
    _, flo, fhi, _ = metric_cluster_bootstrap(y, dec, groups, fpr_fn, n_boot=500, seed=0)
    assert flo <= 0.30 <= fhi


def test_metric_cluster_bootstrap_discards_draws_where_the_metric_is_undefined():
    y = np.array([1, 1, 0, 0])
    dec = np.array([1, 0, 0, 1])
    groups = np.array([0, 0, 1, 1])             # a draw of {0,0} has no negative -> fpr undefined
    _, _, _, n = metric_cluster_bootstrap(y, dec, groups, fpr_fn, n_boot=400, seed=0)
    assert 0 < n < 400


if __name__ == '__main__':
    import traceback
    failures = 0
    mod = sys.modules[__name__]
    for name in sorted(dir(mod)):
        if not name.startswith('test_'):
            continue
        try:
            getattr(mod, name)()
            print(f"  PASS  {name}")
        except Exception:
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
    sys.exit(1 if failures else 0)
