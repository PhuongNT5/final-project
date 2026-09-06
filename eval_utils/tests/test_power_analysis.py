"""Power/required-n: cross-check against the independently-published r5 table
(plans/2026-09-06-thesis-2week-final/research/r5-alternative-evaluation-frameworks.md
B1) and Fisher-z against a hand-solved reference."""
import numpy as np
from scipy import stats as sst

from eval_utils.power_analysis import (
    required_n_noninferiority, achievable_margin, required_n_table, fisher_z_ci,
)


def test_required_n_matches_r5_table_p90():
    # r5 B1 table, p~=0.90: delta=.05->445, .10->111, .15->50, .20->28
    assert abs(required_n_noninferiority(0.90, 0.05) - 445) <= 2
    assert abs(required_n_noninferiority(0.90, 0.10) - 111) <= 2
    assert abs(required_n_noninferiority(0.90, 0.15) - 50) <= 2
    assert abs(required_n_noninferiority(0.90, 0.20) - 28) <= 2


def test_required_n_matches_r5_table_p958():
    # r5 B1 table, p~=0.9581: delta=.05->199, .10->50
    assert abs(required_n_noninferiority(0.9581, 0.05) - 199) <= 2
    assert abs(required_n_noninferiority(0.9581, 0.10) - 50) <= 2


def test_required_n_matches_hand_formula():
    p, delta, alpha, power = 0.85, 0.08, 0.05, 0.80
    z_a = sst.norm.ppf(1 - alpha)
    z_b = sst.norm.ppf(power)
    expected = np.ceil((z_a + z_b) ** 2 * (2 * p * (1 - p)) / delta ** 2)
    assert required_n_noninferiority(p, delta, alpha=alpha, power=power) == expected


def test_achievable_margin_matches_r5_at_n64():
    # r5: achievable margin at n=64 is ~13.2pp (p~.90) to ~8.8pp (p~.958)
    assert abs(achievable_margin(64, 0.90) - 0.132) < 0.003
    assert abs(achievable_margin(64, 0.9581) - 0.088) < 0.003


def test_achievable_margin_is_inverse_of_required_n():
    p, delta = 0.90, 0.10
    n = required_n_noninferiority(p, delta)
    back = achievable_margin(n, p)
    assert abs(back - delta) < 0.002  # ceil(n) rounds up -> margin slightly <= delta


def test_required_n_table_shape():
    tbl = required_n_table(deltas=(0.05, 0.10), ps=(0.90, 0.9581))
    assert set(tbl.keys()) == {(0.05, 0.90), (0.05, 0.9581), (0.10, 0.90), (0.10, 0.9581)}
    assert all(v > 0 for v in tbl.values())


def test_fisher_z_ci_matches_hand_solved_reference():
    r, n = -0.247, 46
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    zc = sst.norm.ppf(0.975)
    lo, hi = np.tanh(z - zc * se), np.tanh(z + zc * se)
    out = fisher_z_ci(r, n)
    assert abs(out['ci_lo'] - lo) < 1e-9
    assert abs(out['ci_hi'] - hi) < 1e-9
    assert out['spans_zero'] is True  # matches audit's "CI spans zero, non-significant" claim


def test_fisher_z_ci_rejects_tiny_n():
    try:
        fisher_z_ci(0.5, 3)
        assert False, 'expected ValueError'
    except ValueError:
        pass


def test_fisher_z_ci_perfect_correlation_does_not_crash():
    out = fisher_z_ci(1.0, 10)
    assert out['ci_hi'] <= 1.0000001


if __name__ == '__main__':
    import sys
    from eval_utils.tests.run_all import run_module
    sys.exit(run_module(sys.modules[__name__]))
