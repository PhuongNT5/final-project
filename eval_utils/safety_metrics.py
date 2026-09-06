"""Safety-oriented metrics for the unsafe-behaviour deployment setting.

- per_class_miss_rate : per-class FNR (missing an unsafe event is the costly error).
- cost_sensitive_score : total cost with missed-unsafe penalised over false-alarm.
- roc_pr_auc : binary unsafe-vs-safe discrimination (needs a P(unsafe) score).
- threshold_for_recall : operating point achieving a target safety recall, and
  the false-alarm rate it costs.
"""
from __future__ import annotations
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False


def per_class_miss_rate(y_true, y_pred, n_classes=4):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    for c in range(n_classes):
        m = y_true == c
        n = int(m.sum())
        out[c] = float(1 - (y_pred[m] == c).mean()) if n else float('nan')
    return out


def cost_sensitive_score(y_true, y_pred, miss_cost=10.0, false_alarm_cost=1.0,
                         unsafe_classes=(0, 1, 2, 3)):
    """Total misclassification cost (lower is better). Errors on a true-unsafe
    clip cost miss_cost; errors on a true-safe clip cost false_alarm_cost."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cost = 0.0
    for t, p in zip(y_true, y_pred):
        if t == p:
            continue
        cost += miss_cost if int(t) in unsafe_classes else false_alarm_cost
    return float(cost)


def roc_pr_auc(scores_unsafe, y_binary):
    """scores_unsafe: P(unsafe) in [0,1]; y_binary: 1=unsafe, 0=safe."""
    if not _HAS_SK:
        return {'roc_auc': None, 'pr_auc': None, 'note': 'sklearn unavailable'}
    return {'roc_auc': float(roc_auc_score(y_binary, scores_unsafe)),
            'pr_auc': float(average_precision_score(y_binary, scores_unsafe))}


def threshold_for_recall(scores_unsafe, y_binary, target_recall=0.95):
    """Highest threshold that still achieves recall >= target on the unsafe class
    (i.e. the lowest false-alarm operating point meeting the safety requirement)."""
    scores = np.asarray(scores_unsafe, dtype=float)
    y = np.asarray(y_binary)
    best = None
    for tau in np.unique(np.concatenate([[0.0], scores, [1.0]])):
        pred = (scores >= tau).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        recall = tp / max(1, tp + fn)
        far = fp / max(1, fp + tn)
        if recall >= target_recall and (best is None or tau > best['threshold']):
            best = {'threshold': float(tau), 'recall': float(recall),
                    'false_alarm_rate': float(far)}
    return best or {'threshold': None,
                    'note': f'target recall {target_recall} unreachable'}
