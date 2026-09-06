"""Paired / proportion significance tests for small-N (N=64) comparisons.

- mcnemar_exact : exact binomial McNemar for two paired classifiers (needs per-clip
  correctness of both models on the SAME clips).
- two_proportion_z : unpaired z-test (used against the paper baselines, whose
  per-clip predictions are not published).
- wilson_ci : Wilson score interval for a single proportion.
- nadeau_bengio : corrected resampled paired t-test for k-fold differences.

All flag the N=64 underpowering where relevant; CIs remain the primary evidence.
"""
from __future__ import annotations
import numpy as np
from scipy import stats as _sst


def mcnemar_exact(correct_a, correct_b):
    """Exact two-sided McNemar on paired boolean-correct arrays.

    On N=64 the discordant count is typically ~6-8 -> underpowered; the returned
    'underpowered' flag makes that explicit.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    b01 = int((~a & b).sum())   # a wrong, b right
    b10 = int((a & ~b).sum())   # a right, b wrong
    n_disc = b01 + b10
    if n_disc == 0:
        p = 1.0
    else:
        k = min(b01, b10)
        p = float(min(1.0, 2.0 * _sst.binom.cdf(k, n_disc, 0.5)))
    return {
        'b01': b01, 'b10': b10, 'n_discordant': n_disc, 'p_value': p,
        'underpowered': n_disc < 10,
        'note': 'N=64: few discordant cells -> use bootstrap CIs as primary evidence',
    }


def two_proportion_z(k1, n1, k2, n2):
    """Unpaired two-proportion z-test. Returns z, p (two-sided), delta_pp."""
    p1, p2 = k1 / n1, k2 / n2
    ph = (k1 + k2) / (n1 + n2)
    se = (ph * (1 - ph) * (1 / n1 + 1 / n2)) ** 0.5
    z = (p1 - p2) / se if se > 0 else 0.0
    return {'z': float(z), 'p_value': float(2 * (1 - _sst.norm.cdf(abs(z)))),
            'delta_pp': float((p1 - p2) * 100)}


def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for a single proportion (returns lo, hi in 0..1)."""
    if n == 0:
        return (0.0, 0.0)
    z = _sst.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (float(center - half), float(center + half))


def mcnemar_discordance_bound(n_total, correct_a, correct_b):
    """McNemar bound when only total-correct counts are known for one side (the
    opponent's per-clip predictions are unavailable) — see
    plans/2026-09-05-thesis-final-push/reports/t1-objectives-audit.md S3b for the
    required semantics. Thin wrapper around unsafenet.metrics_stats.mcnemar_bound
    (reused, not reimplemented) plus the "could this favour b?" flag that section
    3b's table reports as "best split that favors us".

    Enumerates every discordant split (b01, b10) consistent with
    correct_a - correct_b = b01 - b10 and returns the smallest exact two-sided
    McNemar p across all such splits: a LOWER BOUND on p (p >= this value), which
    can only be tightened with the real per-clip predictions.

    diff = correct_a - correct_b. If diff > 0, b01 - b10 = diff > 0 for every valid
    split, i.e. discordance is structurally forced to favour a, not b — no split
    can show b ahead (`favours_b=False`). If diff < 0 the reverse holds. diff == 0
    is a tie at the aggregate level.
    """
    from unsafenet.metrics_stats import mcnemar_bound
    base = mcnemar_bound(n_total, correct_a, correct_b)
    diff = base['diff']
    return {
        **base,
        'favours_b': diff < 0,
        'note': ('bound on how significantly a leads' if diff > 0 else
                 'bound on how significantly b leads' if diff < 0 else
                 'aggregate tie: no discordance-favoured direction'),
    }


def nadeau_bengio(diffs, n_train, n_test):
    """Corrected resampled paired t-test (Nadeau & Bengio, 2003) for k-fold diffs.

    Corrects the naive t-test variance for the train/test overlap across folds.
    """
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean, var = d.mean(), d.var(ddof=1)
    if var == 0:
        return {'t': 0.0, 'p_value': 1.0, 'mean_diff': float(mean)}
    corr = 1.0 / n + n_test / max(1, n_train)
    t = mean / ((var * corr) ** 0.5)
    p = 2 * (1 - _sst.t.cdf(abs(t), df=n - 1))
    return {'t': float(t), 'p_value': float(p), 'mean_diff': float(mean)}
