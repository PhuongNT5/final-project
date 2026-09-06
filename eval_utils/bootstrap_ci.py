"""Bootstrap confidence intervals for any metric(y_true_sub, y_pred_sub).

Default 5000 resamples. BCa (bias-corrected and accelerated; Efron & Tibshirani
1993) is preferred at small N because it corrects for bias and skew that the
plain percentile interval ignores.
"""
from __future__ import annotations
import numpy as np
from scipy import stats as _sst


def bootstrap_ci(y_true, y_pred, metric, n_boot=5000, alpha=0.05, seed=0, method='bca'):
    """Return (point, lo, hi, std).

    metric: callable(y_true_sub, y_pred_sub) -> scalar.
    method: 'percentile' or 'bca'.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    point = float(metric(y_true, y_pred))

    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = metric(y_true[idx], y_pred[idx])

    if method == 'percentile':
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return point, float(lo), float(hi), float(boot.std())

    # --- BCa ---
    prop_less = float((boot < point).mean())
    z0 = _sst.norm.ppf(prop_less) if 0.0 < prop_less < 1.0 else 0.0

    # jackknife estimate of acceleration
    jack = np.empty(n)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        keep[i] = False
        jack[i] = metric(y_true[keep], y_pred[keep])
        keep[i] = True
    jbar = jack.mean()
    num = ((jbar - jack) ** 3).sum()
    den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
    a = float(num / den) if den != 0 else 0.0

    zl, zu = _sst.norm.ppf(alpha / 2), _sst.norm.ppf(1 - alpha / 2)

    def _adj(z):
        return _sst.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lo = np.percentile(boot, 100 * _adj(zl))
    hi = np.percentile(boot, 100 * _adj(zu))
    return point, float(lo), float(hi), float(boot.std())
