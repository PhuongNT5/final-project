"""Declared cost matrix for safety-oriented scoring (brief S4.4c).

The cost matrix is an INPUT, declared before any result is computed, never inferred
from the confusion matrix or from which errors happen to occur. `default_cost_matrix`
gives the project's documented default; callers that want a different matrix build
their own C x C array and pass it straight to `total_cost`.

Default rationale (matches the miss_cost/false_alarm_cost split already used by
`eval_utils.safety_metrics.cost_sensitive_score`, generalised to a full matrix):
  - true UNSAFE clip misclassified as anything else  -> MISS_COST   (10.0)
    missing a real unsafe event is the costly error in a safety system.
  - true SAFE clip misclassified as anything else     -> FALSE_ALARM_COST (1.0)
    a false alarm wastes operator attention but nobody is at risk.
  - correct predictions                               -> 0.0
`unsafe_classes` defaults to ALL classes, matching the current 4-class test protocol
(WalkwayViol/UnauthorInt/OpenedPanel/CarryOverload — every one of the 64 test clips is
an unsafe event; there is no safe class in this split). Pass unsafe_classes=range(4)
with n_classes=8 for the full 8-class ontology (4 safe + 4 unsafe) once a safe-class
test split exists.
"""
from __future__ import annotations
import numpy as np

MISS_COST = 10.0
FALSE_ALARM_COST = 1.0


def default_cost_matrix(n_classes, unsafe_classes=None, miss_cost=MISS_COST,
                        false_alarm_cost=FALSE_ALARM_COST):
    """Return a documented C x C cost matrix (rows=true, cols=predicted), diag=0.

    unsafe_classes: iterable of true-class indices treated as "must not miss".
    Defaults to every class in [0, n_classes) — the current all-unsafe test protocol.
    """
    if unsafe_classes is None:
        unsafe_classes = range(n_classes)
    unsafe_classes = set(int(c) for c in unsafe_classes)
    m = np.zeros((n_classes, n_classes), dtype=float)
    for i in range(n_classes):
        cost = miss_cost if i in unsafe_classes else false_alarm_cost
        for j in range(n_classes):
            if i != j:
                m[i, j] = cost
    return m


def total_cost(y_true, y_pred, cost_matrix):
    """Sum of cost_matrix[true, pred] over all samples. Lower is better."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    return float(cost_matrix[y_true, y_pred].sum())


def mean_cost(y_true, y_pred, cost_matrix):
    """total_cost normalised by N — comparable across differently sized test sets."""
    n = len(np.asarray(y_true))
    return total_cost(y_true, y_pred, cost_matrix) / n if n else 0.0


def cost_report(y_true, y_pred, cost_matrix=None, n_classes=None, unsafe_classes=None):
    """Convenience wrapper: builds the default matrix if none given, returns both the
    declared matrix and the resulting total/mean cost so the matrix is always reported
    alongside the number it produced (brief requirement: declared, not inferred)."""
    y_true = np.asarray(y_true)
    if n_classes is None:
        n_classes = int(max(y_true.max(), np.asarray(y_pred).max())) + 1
    if cost_matrix is None:
        cost_matrix = default_cost_matrix(n_classes, unsafe_classes)
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    return {
        'cost_matrix': cost_matrix.tolist(),
        'unsafe_classes': sorted(unsafe_classes) if unsafe_classes is not None
                          else list(range(n_classes)),
        'total_cost': total_cost(y_true, y_pred, cost_matrix),
        'mean_cost': mean_cost(y_true, y_pred, cost_matrix),
    }
