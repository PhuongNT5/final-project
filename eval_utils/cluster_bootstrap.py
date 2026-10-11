"""Cluster (grouped) bootstrap for paired metric differences.

`paired_tests.paired_diff_bootstrap_ci` resamples CLIPS, which assumes clips are independent. On a
grouped corpus that assumption is false: iSafetyBench's 1,100 clips come from 151 source videos, all
clips of a source share a scene, and the top 5 sources hold ~24% of the corpus. Clip-level
resampling therefore reports an interval that is too narrow — it counts near-duplicates as
independent evidence.

This module resamples SOURCES with replacement instead, so the interval carries the real unit of
independence. It also takes an arbitrary metric (balanced accuracy, not plain accuracy) since a
class-imbalanced corpus makes plain accuracy the wrong headline.
"""
import numpy as np


def balanced_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    recalls = [(y_pred[y_true == c] == c).mean() for c in np.unique(y_true)
               if (y_true == c).any()]
    return float(np.mean(recalls)) if recalls else float('nan')


def paired_metric_cluster_bootstrap(y_true, pred_a, pred_b, groups, metric=balanced_accuracy,
                                    n_boot=5000, seed=0, alpha=0.05):
    """Percentile CI for metric(a) - metric(b) in percentage points, resampling whole groups.

    Both predictions are evaluated on the SAME resampled clip set every draw, so the pairing is
    preserved exactly as in the clip-level version. Draws where the resampled set lost a class are
    discarded (they make balanced accuracy undefined) and the count is returned for disclosure.

    -> (diff_pp, lo_pp, hi_pp, n_valid_draws)
    """
    y_true = np.asarray(y_true); pred_a = np.asarray(pred_a); pred_b = np.asarray(pred_b)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = [np.where(groups == g)[0] for g in uniq]
    diff = (metric(y_true, pred_a) - metric(y_true, pred_b)) * 100.0
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_group[p] for p in pick])
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append((metric(y_true[idx], pred_a[idx]) - metric(y_true[idx], pred_b[idx])) * 100.0)
    if not boots:
        return float(diff), float('nan'), float('nan'), 0
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diff), float(lo), float(hi), len(boots)


def metric_cluster_bootstrap(y_true, pred, groups, metric, n_boot=5000, seed=0, alpha=0.05):
    """Percentile CI for a SINGLE metric, resampling whole groups (same unit as the paired form).

    `metric(y_true, pred) -> float` in native units; the interval is returned in the same units.
    Draws where the metric is undefined (nan, e.g. a resample that lost the positive class) are
    discarded and the surviving count is returned for disclosure.

    -> (point, lo, hi, n_valid_draws)
    """
    y_true = np.asarray(y_true); pred = np.asarray(pred); groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = [np.where(groups == g)[0] for g in uniq]
    point = float(metric(y_true, pred))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_group[p] for p in pick])
        v = float(metric(y_true[idx], pred[idx]))
        if not np.isnan(v):
            boots.append(v)
    if not boots:
        return point, float('nan'), float('nan'), 0
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi), len(boots)
