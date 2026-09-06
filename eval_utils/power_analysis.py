"""Power / required-n for two-proportion non-inferiority (Chow, Shao & Wang
2003, ch.4, normal-approximation form), plus a Fisher-z CI on a correlation
(RQ8: does val macro predict test macro across the 46-config sweep, reported
in plans/2026-09-05-thesis-final-push/reports/t1-objectives-audit.md S1b).

No statsmodels dependency -- both are closed-form with scipy.stats.norm only.
"""
from __future__ import annotations
import numpy as np
from scipy import stats as _sst


def required_n_noninferiority(p1, delta, p2=None, alpha=0.05, power=0.80):
    """Per-arm sample size for a one-sided two-independent-proportion
    non-inferiority test.

    H0: p1 - p2 <= -delta  vs  H1: p1 - p2 > -delta
    n = (z_alpha + z_beta)^2 * [p1(1-p1) + p2(1-p2)] / delta^2

    alpha is ONE-SIDED (non-inferiority is inherently a one-sided test).
    p2 defaults to p1 (the common "true accuracy ~p on both arms" assumption
    used when only one system's rate is precisely known). Returns the ceiling
    (protects nominal power against rounding down)."""
    if p2 is None:
        p2 = p1
    z_a = _sst.norm.ppf(1 - alpha)
    z_b = _sst.norm.ppf(power)
    var_sum = p1 * (1 - p1) + p2 * (1 - p2)
    n = (z_a + z_b) ** 2 * var_sum / (delta ** 2)
    return float(np.ceil(n))


def achievable_margin(n, p1, p2=None, alpha=0.05, power=0.80):
    """Invert required_n_noninferiority: the smallest margin delta resolvable
    at a given per-arm n, alpha, power. This is the THEORETICAL best-case
    margin (uses an assumed p, not any single observed k) -- it will generally
    be narrower than a margin derived from one actual observed k/n, because
    that empirical version also carries the sampling noise of the specific
    trial (see plans/.../research/r5-alternative-evaluation-frameworks.md B1)."""
    if p2 is None:
        p2 = p1
    z_a = _sst.norm.ppf(1 - alpha)
    z_b = _sst.norm.ppf(power)
    var_sum = p1 * (1 - p1) + p2 * (1 - p2)
    return float(np.sqrt((z_a + z_b) ** 2 * var_sum / n))


def required_n_table(deltas=(0.05, 0.10, 0.15, 0.20), ps=(0.90, 0.9581), **kwargs):
    """Grid of required per-arm n over margins x assumed baseline accuracies.
    Returns {(delta, p): n}."""
    return {(d, p): required_n_noninferiority(p, d, **kwargs) for d in deltas for p in ps}


def fisher_z_ci(r, n, alpha=0.05):
    """Fisher-z confidence interval on a Pearson correlation r estimated from
    n pairs. z = arctanh(r); SE = 1/sqrt(n-3); CI = tanh(z +/- z_crit * SE)
    (Fisher 1921). Converts "this benchmark's val/test correlation is small
    and non-significant" into a numbered interval instead of a bare p-value."""
    if n < 4:
        raise ValueError(f'Fisher-z requires n>=4 (got n={n})')
    if not -1.0 <= r <= 1.0:
        raise ValueError(f'r must be in [-1, 1] (got {r})')
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    z_crit = _sst.norm.ppf(1 - alpha / 2)
    lo, hi = z - z_crit * se, z + z_crit * se
    return {'r': float(r), 'n': int(n),
            'ci_lo': float(np.tanh(lo)), 'ci_hi': float(np.tanh(hi)),
            'spans_zero': bool(np.tanh(lo) <= 0 <= np.tanh(hi))}
