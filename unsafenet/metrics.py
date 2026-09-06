"""Evaluation metrics for unsafe-behaviour clip classification - single reusable source of truth
so every track reports the same numbers the same way (thesis SS4.7.7 protocol).

Every metric takes (y, probs) with probs (N,C) row-stochastic and y (N,) int labels in [0,C).
`report()` computes the full set once; `markdown_table()` renders several reports side by side.
No sklearn - real implementations from numpy/scipy only. Statistical tests + CIs live in
`metrics_stats.py` to keep this file under the 200-line project cap.
"""
from __future__ import annotations
import numpy as np

from . import metrics_stats as S


def confusion_matrix(y, pred, n_cls: int):
    cm = np.zeros((n_cls, n_cls), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm


def top1(y, pred):
    return float((y == pred).mean())


def per_class_recall(y, pred, n_cls: int):
    return np.array([np.mean(pred[y == k] == k) if (y == k).any() else np.nan for k in range(n_cls)])


def macro_accuracy(y, pred, n_cls: int):
    return float(np.nanmean(per_class_recall(y, pred, n_cls)))


# Balanced accuracy IS the macro-average recall for single-label multiclass - same number, kept
# as a separate name because both terms appear in the literature and reviewers ask for both.
balanced_accuracy = macro_accuracy


def precision_recall_f1(y, pred, n_cls: int):
    prec, rec, f1 = np.zeros(n_cls), np.zeros(n_cls), np.zeros(n_cls)
    for k in range(n_cls):
        tp = int(np.sum((pred == k) & (y == k)))
        fp = int(np.sum((pred == k) & (y != k)))
        fn = int(np.sum((pred != k) & (y == k)))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        prec[k], rec[k] = p, r
        f1[k] = 2 * p * r / (p + r) if (p + r) else 0.0
    return prec, rec, f1


def f1_variants(y, pred, n_cls: int):
    prec, rec, f1 = precision_recall_f1(y, pred, n_cls)
    support = np.array([(y == k).sum() for k in range(n_cls)])
    macro_f1 = float(f1.mean())
    weighted_f1 = float((f1 * support).sum() / support.sum()) if support.sum() else 0.0
    micro_f1 = top1(y, pred)  # micro-F1 == accuracy for single-label multiclass
    return macro_f1, weighted_f1, micro_f1, prec, rec, f1


def matthews_corrcoef(y, pred, n_cls: int):
    """Multiclass MCC (Gorodkin 2004), from the confusion matrix directly."""
    cm = confusion_matrix(y, pred, n_cls).astype(float)
    t, p, n, c = cm.sum(1), cm.sum(0), cm.sum(), np.trace(cm)
    denom = np.sqrt((n * n - t @ t) * (n * n - p @ p))
    return float((n * c - t @ p) / denom) if denom > 0 else 0.0


def cohen_kappa(y, pred, n_cls: int):
    cm = confusion_matrix(y, pred, n_cls).astype(float)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(1) * cm.sum(0)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def _roc_auc_binary(y_bin, score):
    order = np.argsort(-score)
    y_bin = y_bin[order]
    P, N = y_bin.sum(), len(y_bin) - y_bin.sum()
    if P == 0 or N == 0:
        return float('nan')
    tpr = np.concatenate([[0], np.cumsum(y_bin) / P, [1]])
    fpr = np.concatenate([[0], np.cumsum(1 - y_bin) / N, [1]])
    return float(np.trapezoid(tpr, fpr))


def _pr_auc_binary(y_bin, score):
    """Average precision: step-interpolated area under the precision-recall curve."""
    order = np.argsort(-score)
    y_bin = y_bin[order]
    P = y_bin.sum()
    if P == 0:
        return float('nan')
    tp, fp = np.cumsum(y_bin), np.cumsum(1 - y_bin)
    precision = np.concatenate([[1], tp / (tp + fp)])
    recall = np.concatenate([[0], tp / P])
    return float(np.sum(np.diff(recall) * precision[1:]))


def roc_pr_auc_ovr(y, probs, n_cls: int):
    """One-vs-rest ROC-AUC and PR-AUC, per class and macro-averaged (NaN-safe: classes absent
    from the given labels are excluded from the macro mean, not treated as zero)."""
    roc = np.array([_roc_auc_binary((y == k).astype(int), probs[:, k]) for k in range(n_cls)])
    pr = np.array([_pr_auc_binary((y == k).astype(int), probs[:, k]) for k in range(n_cls)])
    return roc, pr, float(np.nanmean(roc)), float(np.nanmean(pr))


def ece_mce(y, probs, n_bins: int = 15):
    """Expected / maximum calibration error over equal-width confidence bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = mce = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        gap = abs(correct[m].mean() - conf[m].mean())
        ece += m.mean() * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def brier_score(y, probs, n_cls: int):
    onehot = np.eye(n_cls)[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, 1)))


def nll(y, probs, eps: float = 1e-12):
    p = np.clip(probs[np.arange(len(y)), y], eps, 1)
    return float(-np.mean(np.log(p)))


def per_class_error(y, pred, n_cls: int):
    r = per_class_recall(y, pred, n_cls)
    return {int(k): (float(1 - r[k]) if not np.isnan(r[k]) else None) for k in range(n_cls)}


def per_view_error(y, pred, names):
    """Top-1 error by camera view via data_index's view map (names = Clip.name basenames)."""
    try:
        from .data_index import load_clips
    except Exception:
        return {}
    vmap = {c.name: c.view for c in load_clips()}
    views = [vmap.get(nm, 'na') for nm in names]
    out = {}
    for v in sorted(set(views)):
        idx = [i for i, vv in enumerate(views) if vv == v]
        yi, pi = y[idx], pred[idx]
        out[v] = {'n': len(idx), 'error': float((yi != pi).mean())}
    return out


def report(y, probs, name: str, names=None, n_cls: int | None = None) -> dict:
    """Every metric this module knows, for one model on one split. `names` (optional, clip
    basenames aligned with y) enables the per-camera-view breakdown."""
    y = np.asarray(y)
    n_cls = n_cls or probs.shape[1]
    pred = probs.argmax(1)
    macro_f1, weighted_f1, micro_f1, prec, rec, f1 = f1_variants(y, pred, n_cls)
    roc, pr, roc_macro, pr_macro = roc_pr_auc_ovr(y, probs, n_cls)
    ece, mce = ece_mce(y, probs)
    cm = confusion_matrix(y, pred, n_cls)
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    correct = int((pred == y).sum())
    return {
        'name': name, 'n': int(len(y)),
        'top1': top1(y, pred), 'correct': correct,
        'top1_ci_clopper_pearson': S.clopper_pearson(correct, len(y)),
        'top1_ci_wilson': S.wilson_interval(correct, len(y)),
        'macro_accuracy': macro_accuracy(y, pred, n_cls),
        'macro_accuracy_ci_bootstrap': S.bootstrap_ci(y, pred, lambda yy, pp: macro_accuracy(yy, pp, n_cls)),
        'balanced_accuracy': balanced_accuracy(y, pred, n_cls),
        'precision': prec.tolist(), 'recall': rec.tolist(), 'f1': f1.tolist(),
        'macro_f1': macro_f1,
        'macro_f1_ci_bootstrap': S.bootstrap_ci(y, pred, lambda yy, pp: f1_variants(yy, pp, n_cls)[0]),
        'weighted_f1': weighted_f1, 'micro_f1': micro_f1,
        'mcc': matthews_corrcoef(y, pred, n_cls), 'kappa': cohen_kappa(y, pred, n_cls),
        'confusion': cm.tolist(), 'confusion_normalized': cmn.tolist(),
        'roc_auc_per_class': roc.tolist(), 'roc_auc_macro': roc_macro,
        'pr_auc_per_class': pr.tolist(), 'pr_auc_macro': pr_macro,
        'ece': ece, 'mce': mce, 'brier': brier_score(y, probs, n_cls), 'nll': nll(y, probs),
        'per_class_error': per_class_error(y, pred, n_cls),
        'per_view_error': per_view_error(y, pred, names) if names is not None else {},
    }


def markdown_table(reports: list[dict]) -> str:
    """Side-by-side comparison table for `report()` dicts - headline + secondary scalars only."""
    cols = ['name', 'n', 'top1', 'macro_accuracy', 'macro_f1', 'weighted_f1', 'mcc', 'kappa',
            'roc_auc_macro', 'pr_auc_macro', 'ece', 'brier', 'nll']
    hdr = '| ' + ' | '.join(cols) + ' |'
    sep = '|' + '---|' * len(cols)
    rows = []
    for r in reports:
        vals = [f'{r[c]:.4f}' if isinstance(r.get(c), float) else str(r.get(c)) for c in cols]
        rows.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join([hdr, sep] + rows)
