"""Statistical tests + confidence intervals for metrics.py - split out to respect the 200-line
project cap. Everything here is seeded / deterministic. No sklearn; scipy.stats supplies the
standard distributions (beta, normal, binomial) used by the exact/asymptotic intervals.
"""
from __future__ import annotations
import numpy as np
from scipy import stats as _st


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact binomial CI for a proportion k/n (inverts the binomial CDF via the beta dist)."""
    if n == 0:
        return 0.0, 1.0
    lo = float(_st.beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(_st.beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def wilson_interval(k: int, n: int, alpha: float = 0.05):
    """Wilson score interval - better small-n coverage than the naive normal approximation."""
    if n == 0:
        return 0.0, 1.0
    z = float(_st.norm.ppf(1 - alpha / 2))
    p = k / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    adj = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return float((centre - adj) / denom), float((centre + adj) / denom)


def bootstrap_ci(y, pred, fn, n: int = 5000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for any scalar metric fn(y, pred), resampling clips with
    replacement. `fn` must tolerate the resampled label set missing a class (report NaN-safely).
    """
    rng = np.random.default_rng(seed)
    N = len(y)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)
        vals[i] = fn(y[idx], pred[idx])
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mcnemar_exact(y, pred_a, pred_b):
    """Exact McNemar (sign-binomial) test between two models scored on the SAME clips."""
    ca, cb = pred_a == y, pred_b == y
    b = int(np.sum(ca & ~cb))   # a right, b wrong
    c = int(np.sum(~ca & cb))   # b right, a wrong
    n = b + c
    p = float(min(2 * _st.binom.cdf(min(b, c), n, 0.5), 1.0)) if n else 1.0
    return {'b': b, 'c': c, 'n_discordant': n, 'p': p}


def mcnemar_bound(n_total: int, correct_a: int, correct_b: int):
    """Bounded McNemar when only total-correct counts are known for the opponent (we lack the
    paper's per-clip predictions). Enumerates every discordant split (b, c) consistent with
    correct_a - correct_b = b - c, 0 <= b, c, b + c <= n_total, and returns the SMALLEST exact
    p across all such splits - i.e. the most favourable split for the "difference is real" claim.
    Report this as a lower bound on p (p >= this value); it can never be tightened without the
    per-clip predictions.
    """
    diff = correct_a - correct_b
    best_p = 1.0
    for c in range(0, n_total + 1):
        b = diff + c
        if b < 0 or b > n_total or b + c > n_total:
            continue
        n = b + c
        if n == 0:
            continue
        p = 2 * _st.binom.cdf(min(b, c), n, 0.5)
        best_p = min(best_p, p)
    return {'diff': int(diff), 'min_p_over_splits': float(min(best_p, 1.0))}


def permutation_test(y, pred_a, pred_b, metric_fn, n: int = 10000, seed: int = 0):
    """Paired permutation test for the difference metric_fn(y,pred_a) - metric_fn(y,pred_b).
    Under H0 the two models' per-clip predictions are exchangeable, so each clip's (a, b)
    assignment is swapped with prob 0.5 and the metric gap is recomputed on the shuffled labels.
    Works for any scalar metric_fn(y, pred), not just accuracy.
    """
    obs = metric_fn(y, pred_a) - metric_fn(y, pred_b)
    rng = np.random.default_rng(seed)
    N = len(y)
    stacked = np.stack([np.asarray(pred_a), np.asarray(pred_b)])
    diffs = np.empty(n)
    for i in range(n):
        swap = rng.integers(0, 2, N)
        a = stacked[swap, np.arange(N)]
        b = stacked[1 - swap, np.arange(N)]
        diffs[i] = metric_fn(y, a) - metric_fn(y, b)
    p = float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12))
    return {'observed_diff': float(obs), 'p': p}
