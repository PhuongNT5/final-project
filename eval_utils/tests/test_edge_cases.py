"""Cross-cutting edge cases required by the task: zero-support class, a bootstrap
resample that collapses to a single class, undefined AUC — none should crash."""
import numpy as np

from unsafenet.metrics import macro_accuracy, roc_pr_auc_ovr
from eval_utils.bootstrap_ci import bootstrap_ci
from eval_utils.safety_metrics import roc_pr_auc, threshold_for_recall


def test_bootstrap_survives_single_class_resample():
    """With a tiny, imbalanced y (1 sample of class 1), some bootstrap resamples will
    be all-class-0. macro_accuracy must not crash / return NaN for the whole CI."""
    y = np.array([0, 0, 0, 0, 1])
    pred = np.array([0, 0, 0, 1, 1])
    point, lo, hi, sd = bootstrap_ci(y, pred, lambda a, b: macro_accuracy(a, b, 2),
                                     n_boot=500, seed=0, method='percentile')
    assert not np.isnan(point)
    assert not np.isnan(lo) and not np.isnan(hi)


def test_bca_survives_single_class_resample():
    y = np.array([0, 0, 0, 0, 1])
    pred = np.array([0, 0, 0, 1, 1])
    point, lo, hi, sd = bootstrap_ci(y, pred, lambda a, b: macro_accuracy(a, b, 2),
                                     n_boot=500, seed=0, method='bca')
    assert not np.isnan(point) and not np.isnan(lo) and not np.isnan(hi)


def test_roc_pr_auc_binary_all_one_class_returns_nan_not_crash():
    """Degenerate all-positive label set: ROC-AUC is mathematically undefined (no
    negatives to rank against) -> NaN, not a crash and not a fabricated number.
    PR-AUC (average precision) IS well-defined here (all positive -> AP=1.0)."""
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    y_binary = np.array([1, 1, 1, 1])  # degenerate: no negatives
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = roc_pr_auc(scores, y_binary)
    assert np.isnan(out['roc_auc'])
    assert out['pr_auc'] == 1.0


def test_multiclass_ovr_auc_undefined_when_single_class_present():
    """Only class 0 ever occurs -> ROC-AUC needs both a positive and a negative for
    EVERY class, so both per-class ROC-AUCs are undefined (NaN) and macro ROC-AUC is
    NaN (correctly propagated, not silently zeroed). PR-AUC only needs positives for
    the class being scored, so class 0's AP is still defined (=1.0, all-positive);
    only class 1's AP (no positives) is undefined -> macro PR-AUC = nanmean = 1.0.
    Neither path crashes."""
    y = np.array([0, 0, 0, 0])
    probs = np.array([[0.6, 0.4], [0.7, 0.3], [0.55, 0.45], [0.9, 0.1]])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        roc, pr, roc_macro, pr_macro = roc_pr_auc_ovr(y, probs, 2)
    assert np.isnan(roc[0]) and np.isnan(roc[1]) and np.isnan(roc_macro)
    assert pr[0] == 1.0 and np.isnan(pr[1]) and pr_macro == 1.0


def test_threshold_for_recall_unreachable_target_flagged():
    scores = np.array([0.1, 0.2, 0.3])
    y = np.array([1, 0, 1])
    out = threshold_for_recall(scores, y, target_recall=1.5)  # impossible target
    assert out['threshold'] is None
    assert 'unreachable' in out['note']


if __name__ == '__main__':
    import sys
    from eval_utils.tests.run_all import run_module
    sys.exit(run_module(sys.modules[__name__]))
