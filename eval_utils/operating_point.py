"""Operating-point selection by SAFETY RECALL, not accuracy (brief S4.4d).

Generalises the risk-coverage sweep in `run_operating_point.py` into a reusable
function. The decision rule: fire an alert when the model's confidence
(max-softmax) >= tau, else abstain (defer to a human / escalate). Sweeping tau
trades coverage (= fraction of clips the system commits to, i.e. "firing recall"
on this all-unsafe test) against accuracy-on-fired. The safety operating point is
the highest tau that still keeps recall >= a declared target (default 0.95) — it is
NEVER chosen by which tau maximises accuracy.

For a genuine binary safe/unsafe label, use `safety_metrics.threshold_for_recall`
(true-positive-rate framing) instead — re-exported here for a single import site.
"""
from __future__ import annotations
import numpy as np

from eval_utils.safety_metrics import threshold_for_recall  # re-export, binary framing

__all__ = ['threshold_for_recall', 'confidence_threshold_sweep',
           'select_operating_point_by_safety_recall']


def confidence_threshold_sweep(y_true, y_prob, thresholds=None, n_classes=None):
    """Sweep max-softmax confidence thresholds. Returns a list of per-tau dicts:
    tau, coverage (firing recall), accuracy_on_fired, n_fired, n_abstained,
    per_class_recall_on_fired (safety recall per class among fired clips)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    pred = y_prob.argmax(1)
    conf = y_prob.max(1)
    n_classes = n_classes or y_prob.shape[1]
    if thresholds is None:
        thresholds = np.unique(np.concatenate([[0.0], conf, [1.0]]))

    rows = []
    for tau in thresholds:
        fired = conf >= tau
        n_fired = int(fired.sum())
        coverage = float(fired.mean())
        acc = float((pred[fired] == y_true[fired]).mean()) if n_fired else 0.0
        per_class = {}
        for c in range(n_classes):
            m = fired & (y_true == c)
            n_c = int((y_true == c).sum())
            per_class[c] = float(m.sum() / n_c) if n_c else float('nan')
        rows.append({
            'tau': float(tau), 'coverage': coverage, 'accuracy_on_fired': acc,
            'n_fired': n_fired, 'n_abstained': int((~fired).sum()),
            'per_class_recall_on_fired': per_class,
        })
    return rows


def select_operating_point_by_safety_recall(y_true, y_prob, target_recall=0.95,
                                            selection_y_true=None, selection_y_prob=None,
                                            n_classes=None):
    """Pick the highest confidence threshold whose COVERAGE (safety recall proxy on
    this all-unsafe protocol) >= target_recall, then apply it untouched to
    (y_true, y_prob). If a leakage-free selection set is given, tau is chosen there
    and only applied (not re-selected) on the main set — the honest protocol. Without
    it, tau is selected on the same data it's evaluated on and the result is flagged
    `test_selected=True` (diagnostic only, not evidence-grade — mirrors
    run_operating_point.py's existing warning)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    n_classes = n_classes or y_prob.shape[1]

    if selection_y_true is not None and selection_y_prob is not None:
        sel_rows = confidence_threshold_sweep(selection_y_true, selection_y_prob, n_classes=n_classes)
        test_selected = False
    else:
        sel_rows = confidence_threshold_sweep(y_true, y_prob, n_classes=n_classes)
        test_selected = True

    ok = [r for r in sel_rows if r['coverage'] >= target_recall]
    tau_star = max(ok, key=lambda r: r['tau'])['tau'] if ok else min(
        sel_rows, key=lambda r: r['tau'])['tau']

    conf = y_prob.max(1)
    pred = y_prob.argmax(1)
    fired = conf >= tau_star
    n_fired = int(fired.sum())
    applied = {
        'tau': float(tau_star), 'target_recall': float(target_recall),
        'coverage': float(fired.mean()),
        'accuracy_on_fired': float((pred[fired] == y_true[fired]).mean()) if n_fired else 0.0,
        'n_fired': n_fired, 'n_total': int(len(y_true)),
        'test_selected': test_selected,
        'note': ('tau chosen on the SAME data it is scored on — diagnostic only'
                 if test_selected else 'tau chosen on an independent selection set'),
    }
    return applied
