"""Known-answer tests for 4b classification metrics. No pytest in the project venv —
each test_* function raises AssertionError on failure; run via run_all.py."""
import numpy as np
from unsafenet.metrics import (top1, macro_accuracy, balanced_accuracy, precision_recall_f1,
                               f1_variants, matthews_corrcoef, cohen_kappa, confusion_matrix,
                               roc_pr_auc_ovr)
from eval_utils.metrics import per_class_fnr


def test_macro_equals_mean_per_class_recall():
    """Brief S4.4a: macro accuracy MUST equal mean per-class recall, not macro-F1."""
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    pred = np.array([0, 0, 1, 1, 0, 2, 2, 1, 2])
    # per-class recall: class0=2/3, class1=1/2, class2=3/4
    expected = np.mean([2 / 3, 1 / 2, 3 / 4])
    got = macro_accuracy(y, pred, 3)
    assert abs(got - expected) < 1e-9, (got, expected)
    assert balanced_accuracy(y, pred, 3) == got  # documented alias


def test_top1_hand_computed():
    y = np.array([0, 1, 2, 0, 1])
    pred = np.array([0, 1, 1, 0, 0])
    assert abs(top1(y, pred) - 0.6) < 1e-9


def test_confusion_matrix_hand_computed():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    cm = confusion_matrix(y, pred, 2)
    assert cm.tolist() == [[1, 1], [0, 2]], cm.tolist()


def test_precision_recall_f1_hand_computed():
    # class 1: tp=1 (idx2), fp=1 (idx0 pred=1 true=0), fn=0 -> P=0.5, R=1.0, F1=2/3
    y = np.array([0, 0, 1])
    pred = np.array([1, 0, 1])
    prec, rec, f1 = precision_recall_f1(y, pred, 2)
    assert abs(prec[1] - 0.5) < 1e-9 and abs(rec[1] - 1.0) < 1e-9
    assert abs(f1[1] - (2 / 3)) < 1e-9


def test_cohen_kappa_perfect_and_chance():
    y = np.array([0, 1, 0, 1, 0, 1])
    assert abs(cohen_kappa(y, y, 2) - 1.0) < 1e-9  # perfect agreement -> kappa=1
    pred_swap = 1 - y
    assert cohen_kappa(y, pred_swap, 2) < 0  # systematic disagreement -> kappa<0


def test_mcc_perfect_prediction():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert abs(matthews_corrcoef(y, y, 3) - 1.0) < 1e-9


def test_per_class_fnr_is_one_minus_recall():
    y = np.array([0, 0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1, 1])  # class0 recall=1/3 -> fnr=2/3; class1 recall=1 -> fnr=0
    fnr = per_class_fnr(y, pred, classes=range(2))
    assert abs(fnr[0] - 2 / 3) < 1e-9
    assert abs(fnr[1] - 0.0) < 1e-9


def test_zero_support_class_is_nan_safe():
    """Class 2 never appears in y_true -> its recall/FNR must be NaN, and macro_accuracy
    must exclude it from the mean rather than crashing or silently treating it as 0."""
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    fnr = per_class_fnr(y, pred, classes=range(3))
    assert np.isnan(fnr[2])
    macro = macro_accuracy(y, pred, 3)  # only classes 0,1 contribute
    expected = np.mean([0.5, 1.0])
    assert abs(macro - expected) < 1e-9, (macro, expected)


def test_undefined_auc_excluded_from_macro():
    """A class with no positive examples has undefined ROC/PR-AUC (NaN); the macro
    average must ignore it (nanmean), not treat it as 0 or crash."""
    y = np.array([0, 0, 1, 1])  # class 2 never occurs
    probs = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1],
                      [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]])
    roc, pr, roc_macro, pr_macro = roc_pr_auc_ovr(y, probs, 3)
    assert np.isnan(roc[2]) and np.isnan(pr[2])
    assert not np.isnan(roc_macro) and not np.isnan(pr_macro)


if __name__ == '__main__':
    import sys
    from eval_utils.tests.run_all import run_module
    sys.exit(run_module(sys.modules[__name__]))
