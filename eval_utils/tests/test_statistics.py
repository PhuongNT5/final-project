"""4f statistics: verify against scipy references where one exists (no statsmodels
in this venv); otherwise verify internal consistency / hand-derived values."""
import numpy as np
from scipy import stats as sst

from eval_utils.paired_tests import (wilson_ci, mcnemar_exact, mcnemar_discordance_bound,
                                     nadeau_bengio, two_proportion_z)
from eval_utils.bootstrap_ci import bootstrap_ci


def test_wilson_ci_matches_published_reference():
    """Newcombe (1998) worked example: 81/263 -> Wilson 95% CI [0.255, 0.365]."""
    lo, hi = wilson_ci(81, 263)
    assert abs(lo - 0.255) < 0.002, lo
    assert abs(hi - 0.365) < 0.002, hi


def test_wilson_ci_matches_scipy_proportion_ci_wilson():
    """scipy 1.17 has no proportion_confint, but its formula is standard: cross-check
    against a hand-solved quadratic for k=8, n=10, alpha=0.05."""
    k, n = 8, 10
    z = sst.norm.ppf(0.975)
    phat = k / n
    denom = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    expected = (centre - half, centre + half)
    got = wilson_ci(k, n)
    assert abs(got[0] - expected[0]) < 1e-9 and abs(got[1] - expected[1]) < 1e-9


def test_bca_bootstrap_matches_scipy_stats_bootstrap():
    """Cross-check eval_utils' BCa against scipy.stats.bootstrap(method='BCa') on the
    same data + statistic (mean). Intervals should closely agree (same algorithm,
    independent implementation)."""
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=2.0, size=40)
    y_true = x  # dummy "labels" arg for our (y_true, y_pred, metric) signature
    y_pred = np.zeros_like(x)  # unused by the metric below

    def mean_metric(a, b):
        return float(a.mean())

    _, lo, hi, _ = bootstrap_ci(y_true, y_pred, mean_metric, n_boot=3000, seed=1, method='bca')
    ref = sst.bootstrap((x,), np.mean, method='BCa', n_resamples=3000, random_state=1)
    assert abs(lo - ref.confidence_interval.low) < 0.5, (lo, ref.confidence_interval.low)
    assert abs(hi - ref.confidence_interval.high) < 0.5, (hi, ref.confidence_interval.high)


def test_mcnemar_exact_matches_scipy_binomtest():
    """McNemar exact = a two-sided exact binomial test on the discordant pairs —
    cross-check directly against scipy.stats.binomtest."""
    rng = np.random.default_rng(2)
    correct_a = rng.random(64) < 0.85
    correct_b = rng.random(64) < 0.70
    out = mcnemar_exact(correct_a, correct_b)
    b01, b10 = out['b01'], out['b10']
    n_disc = b01 + b10
    ref = sst.binomtest(min(b01, b10), n_disc, 0.5, alternative='two-sided')
    assert abs(out['p_value'] - ref.pvalue) < 1e-9, (out['p_value'], ref.pvalue)


def test_mcnemar_exact_zero_discordant():
    a = np.array([True, True, False, False])
    out = mcnemar_exact(a, a)  # identical -> zero discordant pairs
    assert out['n_discordant'] == 0 and out['p_value'] == 1.0


def test_mcnemar_discordance_bound_reproduces_audit_table():
    """Reproduces plans/2026-09-05-thesis-final-push/reports/t1-objectives-audit.md
    S3b table exactly (paper=62/64 fixed, our_correct varying)."""
    expected = {55: 0.015625, 56: 0.03125, 57: 0.0625, 58: 0.125, 60: 0.5, 62: 1.0, 64: 0.5}
    for our_correct, exp_p in expected.items():
        out = mcnemar_discordance_bound(64, 62, our_correct)
        assert abs(out['min_p_over_splits'] - exp_p) < 1e-9, (our_correct, out)
    # our_correct > paper_correct=62 => diff<0 => favours the SECOND arg (paper in
    # this call ordering is arg 'correct_a'=62; second arg is our_correct) — check
    # the flag direction is internally consistent with the sign of diff.
    out64 = mcnemar_discordance_bound(64, 62, 64)
    assert out64['diff'] == -2 and out64['favours_b'] is True


def test_nadeau_bengio_zero_variance_gives_tie():
    diffs = [2.0, 2.0, 2.0, 2.0, 2.0]  # identical diffs -> var=0
    out = nadeau_bengio(diffs, n_train=400, n_test=100)
    assert out['t'] == 0.0 and out['p_value'] == 1.0


def test_nadeau_bengio_matches_hand_formula():
    diffs = np.array([1.0, 2.0, -1.0, 3.0, 0.5])
    n_train, n_test = 400, 116
    d, n = diffs, len(diffs)
    mean, var = d.mean(), d.var(ddof=1)
    corr = 1.0 / n + n_test / n_train
    expected_t = mean / (var * corr) ** 0.5
    out = nadeau_bengio(diffs, n_train, n_test)
    assert abs(out['t'] - expected_t) < 1e-9


def test_two_proportion_z_symmetry():
    out1 = two_proportion_z(56, 64, 62, 64)
    out2 = two_proportion_z(62, 64, 56, 64)
    assert abs(out1['z'] + out2['z']) < 1e-9  # antisymmetric in the compared groups
    assert abs(out1['p_value'] - out2['p_value']) < 1e-9  # two-sided p is symmetric


if __name__ == '__main__':
    import sys
    from eval_utils.tests.run_all import run_module
    sys.exit(run_module(sys.modules[__name__]))
