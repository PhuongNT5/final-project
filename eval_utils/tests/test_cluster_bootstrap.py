"""Tests for eval_utils/cluster_bootstrap.py — the grouped-resampling paired CI used for G10.

The property that matters: on grouped data the cluster interval must be WIDER than the clip-level
interval, because clip-level resampling counts near-duplicates from the same source as independent
evidence. A regression that silently reverts to clip-level resampling would make every
leave-source-out claim look more significant than it is, so it is asserted explicitly.

Usage: PYTHONPATH=. .venv/bin/python eval_utils/tests/test_cluster_bootstrap.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval_utils.cluster_bootstrap import balanced_accuracy, paired_metric_cluster_bootstrap


def test_balanced_accuracy_matches_hand_computation_on_imbalanced_input():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0, 0, 0, 1, 1, 0])
    # class 0 recall 3/4 = .75, class 1 recall 1/2 = .5 -> .625 (plain accuracy would be 4/6 = .667)
    assert abs(balanced_accuracy(y, p) - 0.625) < 1e-12


def test_balanced_accuracy_ignores_absent_class():
    y = np.array([1, 1, 1]); p = np.array([1, 1, 0])
    assert abs(balanced_accuracy(y, p) - 2 / 3) < 1e-12


def test_identical_predictions_give_zero_diff_and_zero_width_ci():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200); p = rng.integers(0, 2, 200)
    g = np.repeat(np.arange(20), 10)
    diff, lo, hi, n = paired_metric_cluster_bootstrap(y, p, p, g, n_boot=200)
    assert diff == 0.0 and abs(lo) < 1e-9 and abs(hi) < 1e-9 and n > 0


def test_cluster_ci_is_wider_than_clip_ci_when_clips_are_grouped():
    """Construct genuine within-group dependence: each group is one repeated clip, so a group is
    really a single independent observation. The clip-level CI must under-report that."""
    rng = np.random.default_rng(1)
    n_groups, per = 40, 10
    y_g = rng.integers(0, 2, n_groups)
    a_g = (rng.random(n_groups) < 0.80).astype(int)      # model A right on 80% of GROUPS
    b_g = (rng.random(n_groups) < 0.55).astype(int)
    y = np.repeat(y_g, per)
    pred_a = np.where(np.repeat(a_g, per) == 1, y, 1 - y)
    pred_b = np.where(np.repeat(b_g, per) == 1, y, 1 - y)
    groups = np.repeat(np.arange(n_groups), per)
    _, lo_c, hi_c, _ = paired_metric_cluster_bootstrap(y, pred_a, pred_b, groups, n_boot=2000, seed=0)
    _, lo_i, hi_i, _ = paired_metric_cluster_bootstrap(y, pred_a, pred_b, np.arange(len(y)),
                                                       n_boot=2000, seed=0)
    assert (hi_c - lo_c) > (hi_i - lo_i), ((lo_c, hi_c), (lo_i, hi_i))


def test_pairing_is_preserved_diff_is_antisymmetric():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    a = np.where(rng.random(300) < 0.7, y, 1 - y)
    b = np.where(rng.random(300) < 0.6, y, 1 - y)
    g = np.repeat(np.arange(30), 10)
    d1, lo1, hi1, _ = paired_metric_cluster_bootstrap(y, a, b, g, n_boot=500, seed=3)
    d2, lo2, hi2, _ = paired_metric_cluster_bootstrap(y, b, a, g, n_boot=500, seed=3)
    assert abs(d1 + d2) < 1e-9
    assert abs(lo1 + hi2) < 1e-9 and abs(hi1 + lo2) < 1e-9


def test_single_class_resamples_are_discarded_not_crashed():
    y = np.array([0] * 18 + [1, 1])                       # class 1 lives in exactly one group
    g = np.repeat(np.arange(10), 2)
    p = y.copy(); q = 1 - y
    diff, lo, hi, n = paired_metric_cluster_bootstrap(y, p, q, g, n_boot=300, seed=0)
    assert n < 300 and n > 0                              # some draws lost class 1 and were dropped
    assert np.isfinite(lo) and np.isfinite(hi)


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
