"""Core classification metrics.

macro_accuracy = mean per-class recall (the thesis 'macro' metric), NOT macro-F1.
This is the one that the k-fold bootstrap in run_kfold_eval.py should have used
(see phase-00 report): macro-F1 != macro-recall unless precision == recall.
"""
from __future__ import annotations
import numpy as np


def _arr(y_true, y_pred):
    return np.asarray(y_true), np.asarray(y_pred)


def top1(y_true, y_pred):
    y_true, y_pred = _arr(y_true, y_pred)
    return float((y_pred == y_true).mean())


def per_class_recall(y_true, y_pred, classes=None):
    """Return {class_id: recall} over the given classes (default: those present)."""
    y_true, y_pred = _arr(y_true, y_pred)
    if classes is None:
        classes = np.unique(y_true)
    out = {}
    for c in classes:
        m = y_true == c
        n = int(m.sum())
        out[int(c)] = float((y_pred[m] == c).mean()) if n else float('nan')
    return out


def macro_accuracy(y_true, y_pred, classes=None):
    """Mean per-class recall over classes present in y_true (returns 0..1)."""
    rec = per_class_recall(y_true, y_pred, classes)
    vals = [v for v in rec.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0


def per_class_fnr(y_true, y_pred, classes=None):
    """Per-class false-negative (miss) rate = 1 - recall."""
    return {c: (1.0 - r if not np.isnan(r) else float('nan'))
            for c, r in per_class_recall(y_true, y_pred, classes).items()}


def confusion(y_true, y_pred, n_classes=None):
    y_true, y_pred = _arr(y_true, y_pred)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm
