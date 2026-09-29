"""Unit tests for the paired-significance fix (McNemar exact / bound) —
plans/2026-09-06-thesis-2week-final/reports/review-2026-09-13-eval-statistics.md finding #1.

No pytest in the project venv (see eval_utils/tests/run_all.py precedent) — plain assert
functions named test_*, runnable standalone via `if __name__ == '__main__'` below.

Usage: .venv/bin/python tests/test_paired_significance.py
"""
import os
import sys

import numpy as np
from scipy import stats as sst

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_utils.paired_tests import (mcnemar_exact, mcnemar_discordance_bound,
                                      paired_vs_aggregate_baseline, paired_diff_bootstrap_ci)


def test_worked_example_hand_computed_exact_p():
    """n=6 clips, discordant b=1 (a right/b wrong), c=5 (a wrong/b right). Hand computation:
    X ~ Binom(6, 0.5); P(X<=1) = [C(6,0)+C(6,1)] / 2^6 = (1+6)/64 = 7/64 = 0.109375;
    two-sided exact p = 2 * 0.109375 = 0.21875 (matches scipy.stats.binomtest independently)."""
    # 1 clip where a correct & b wrong, 5 clips where a wrong & b correct, rest concordant-correct.
    correct_a = np.array([True, False, False, False, False, False, True, True])
    correct_b = np.array([False, True, True, True, True, True, True, True])
    out = mcnemar_exact(correct_a, correct_b)
    assert out['b'] == 1 and out['c'] == 5, out
    assert abs(out['p_value'] - 0.21875) < 1e-12, out['p_value']
    ref = sst.binomtest(1, 6, 0.5, alternative='two-sided')
    assert abs(out['p_value'] - ref.pvalue) < 1e-9


def test_b_plus_c_zero_edge_case():
    """Identical predictions on every clip -> zero discordant pairs -> p=1.0 (no evidence of a
    difference), diff_accuracy_pp=0, and the paired bootstrap CI collapses to a point at 0 because
    every resample re-applies the SAME index to both (identical) arrays."""
    rng = np.random.default_rng(0)
    a = rng.random(40) < 0.7
    out = mcnemar_exact(a, a.copy())
    assert out['n_discordant'] == 0
    assert out['p_value'] == 1.0
    assert out['diff_accuracy_pp'] == 0.0
    lo, hi = out['diff_accuracy_ci95_pp']
    assert lo == 0.0 and hi == 0.0, (lo, hi)


def test_perfectly_discordant_case():
    """a always correct, b always wrong -> every clip discordant, all favouring a. Hand
    computation at n=10: p = 2 * P(X<=0) = 2 * (0.5)^10 = 2/1024 = 0.001953125 (max possible
    significance at this N). diff_accuracy_pp must be exactly +100 (a always right, b always
    wrong) and its bootstrap CI collapses to a point at 100 (every resample keeps that gap)."""
    n = 10
    correct_a = np.ones(n, dtype=bool)
    correct_b = np.zeros(n, dtype=bool)
    out = mcnemar_exact(correct_a, correct_b)
    assert out['b'] == n and out['c'] == 0, out
    assert abs(out['p_value'] - (2 * (0.5 ** n))) < 1e-12, out['p_value']
    assert out['diff_accuracy_pp'] == 100.0
    lo, hi = out['diff_accuracy_ci95_pp']
    assert lo == 100.0 and hi == 100.0, (lo, hi)
    # sanity: this is the smallest possible p at this n (fully discordant is the max-evidence case)
    assert out['p_value'] < 0.01


def test_bound_path_when_only_aggregate_counts_known():
    """paired_vs_aggregate_baseline (the McNemar BOUND path) for our_correct=56 vs an aggregate-only
    paper baseline of 62/64 must reproduce the previously hand-verified audit-table lower bound
    min_p=0.03125 (plans/2026-09-05-thesis-final-push/reports/t1-objectives-audit.md S3b, also
    cross-checked in eval_utils/tests/test_statistics.py::test_mcnemar_discordance_bound_reproduces_audit_table).
    Also checks the NEW max_p (least-significant bound) is present, >= min_p, and that the
    diagnostic two_proportion_z is clearly labelled invalid rather than silently substituted."""
    out = paired_vs_aggregate_baseline(64, 56, 62, label='paper full-bbox (62/64)')
    assert abs(out['min_p_over_splits'] - 0.03125) < 1e-9, out
    assert out['max_p_over_splits'] >= out['min_p_over_splits'] - 1e-12, out
    assert out['p_value_bound'] == [out['min_p_over_splits'], out['max_p_over_splits']]
    assert out['primary_test'].startswith('mcnemar_discordance_bound')
    assert 'INVALID' in out['diagnostic_warning']
    assert abs(out['delta_pp'] - (56 / 64 - 62 / 64) * 100) < 1e-9
    # favours_b: diff = our_correct(56) - other_correct(62) = -6 < 0 -> bound favours "b" (the
    # baseline), i.e. the discordance is structurally forced toward the baseline leading.
    assert out['favours_b'] is True


def test_bound_max_p_never_below_min_p_over_random_cases():
    """Sanity property: for any (n, correct_a, correct_b), the least-significant bound can never
    be smaller than the most-significant one (they bracket the true unknown p)."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(10, 130))
        ca = int(rng.integers(0, n + 1))
        cb = int(rng.integers(0, n + 1))
        b = mcnemar_discordance_bound(n, ca, cb)
        assert b['max_p_over_splits'] >= b['min_p_over_splits'] - 1e-9, (n, ca, cb, b)


def test_paired_diff_bootstrap_ci_matches_point_estimate():
    """The bootstrap CI must be centred near the observed point-estimate difference and the point
    estimate itself must equal the closed-form accuracy difference."""
    rng = np.random.default_rng(1)
    a = rng.random(64) < 0.85
    b = rng.random(64) < 0.70
    diff, lo, hi = paired_diff_bootstrap_ci(a, b, n_boot=2000, seed=1)
    assert abs(diff - (a.mean() - b.mean()) * 100) < 1e-9
    assert lo <= diff <= hi, (lo, diff, hi)


if __name__ == '__main__':
    import traceback
    failures = 0
    mod = sys.modules[__name__]
    for name in sorted(dir(mod)):
        if not name.startswith('test_'):
            continue
        fn = getattr(mod, name)
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
    sys.exit(1 if failures else 0)
