"""Single reusable entry point for the thesis evaluation-metric contract (brief S4).

`evaluate_suite(y_true, y_prob, class_names, protocol_tag, n_seeds, ...)` returns one
canonical, JSON-serialisable dict covering S4b (multi-lens classification), S4c
(safety/cost), S4d (operating point), S4e (calibration) and the bootstrap/Wilson leg
of S4f. Built on the project's EXISTING metric code (DRY, no reimplementation):
`unsafenet.metrics.report()` supplies the core classification/AUC/calibration
battery; `eval_utils.*` supplies bootstrap CIs, paired tests, the declared cost
matrix, and safety-recall operating-point selection.

4a (timing anchors) lives in `timing.py`, 4f's McNemar/Nadeau-Bengio in
`paired_tests.py`, 4g (Jetson) is out of scope — `deployment/` owns those numbers.
"""
from __future__ import annotations
import numpy as np

from unsafenet.metrics import report as _classification_report
from unsafenet.metrics import macro_accuracy as _macro_accuracy
from unsafenet.metrics import f1_variants as _f1_variants
from eval_utils.metrics import per_class_fnr as _per_class_fnr
from eval_utils.bootstrap_ci import bootstrap_ci
from eval_utils.paired_tests import (wilson_ci, mcnemar_exact, mcnemar_discordance_bound,
                                     two_proportion_z)
from eval_utils.calibration import reliability_bins
from eval_utils.cost_matrix import cost_report
from eval_utils.operating_point import confidence_threshold_sweep, select_operating_point_by_safety_recall
from eval_utils.report_schema import make_record, to_markdown_table


def evaluate_suite(y_true, y_prob, class_names, protocol_tag, n_seeds=1, y_pred=None,
                   cost_matrix=None, unsafe_classes=None, target_safety_recall=0.95,
                   n_boot=5000, seed=0, compare_to=None, selection_y_true=None,
                   selection_y_prob=None, n_bins=15):
    """y_true: (N,) int labels. y_prob: (N,C) row-stochastic scores. class_names:
    list[str] len C. protocol_tag: e.g. 'same-dist-64' | 'kfold-516' | 'lco-516',
    stamped on every record. n_seeds: seeds behind this result (brief S4.4h.5 wants
    >= 5 for any quoted number — recorded, not enforced here). compare_to: optional
    {'name', 'correct', 'n', 'pred'(optional per-clip array on the SAME clips)} for
    McNemar exact (per-clip available) / discordance bound (aggregate-only) /
    two-proportion z (different N, e.g. vs a paper baseline)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    n_cls = len(class_names)
    pred = np.asarray(y_pred) if y_pred is not None else y_prob.argmax(1)
    n = int(len(y_true))
    correct = int((pred == y_true).sum())

    base = _classification_report(y_true, y_prob, name=protocol_tag, n_cls=n_cls)
    fnr = _per_class_fnr(y_true, pred, classes=range(n_cls))
    cost = cost_report(y_true, pred, cost_matrix=cost_matrix, n_classes=n_cls,
                        unsafe_classes=unsafe_classes)
    sweep = confidence_threshold_sweep(y_true, y_prob, n_classes=n_cls)
    op_point = select_operating_point_by_safety_recall(
        y_true, y_prob, target_recall=target_safety_recall,
        selection_y_true=selection_y_true, selection_y_prob=selection_y_prob, n_classes=n_cls)
    rb = reliability_bins(y_prob, y_true, n_bins=n_bins)

    _, t_lo, t_hi, _ = bootstrap_ci(y_true, pred, lambda a, b: float((a == b).mean()),
                                    n_boot=n_boot, seed=seed, method='bca')
    _, m_lo, m_hi, _ = bootstrap_ci(y_true, pred, lambda a, b: _macro_accuracy(a, b, n_cls),
                                    n_boot=n_boot, seed=seed, method='bca')
    _, f_lo, f_hi, _ = bootstrap_ci(y_true, pred, lambda a, b: _f1_variants(a, b, n_cls)[0],
                                    n_boot=n_boot, seed=seed, method='bca')

    def rec(value, ci=None, **extra):
        return make_record(value, ci=ci, n=n, seed_count=n_seeds, protocol_tag=protocol_tag, **extra)

    result = {
        'protocol_tag': protocol_tag, 'n': n, 'seed_count': n_seeds,
        'class_names': list(class_names),
        'top1': rec(base['top1'], ci=(t_lo, t_hi), correct=correct,
                    ci_method='bootstrap_bca_5000',
                    wilson_ci95=list(wilson_ci(correct, n))),
        'macro_accuracy': rec(base['macro_accuracy'], ci=(m_lo, m_hi), ci_method='bootstrap_bca_5000',
                              definition='mean per-class recall'),
        'macro_f1': rec(base['macro_f1'], ci=(f_lo, f_hi), ci_method='bootstrap_bca_5000'),
        'balanced_accuracy': rec(base['balanced_accuracy']),
        'cohen_kappa': rec(base['kappa']),
        'matthews_mcc': rec(base['mcc']),
        'per_class': {class_names[c]: {
            'precision': base['precision'][c], 'recall': base['recall'][c], 'f1': base['f1'][c],
            'fnr_miss_rate': fnr[c], 'n': int((y_true == c).sum()),
        } for c in range(n_cls)},
        'confusion_matrix': base['confusion'], 'confusion_matrix_normalized': base['confusion_normalized'],
        'safety_cost': {**cost, 'protocol_tag': protocol_tag, 'n': n, 'seed_count': n_seeds},
        'roc_auc_ovr': {'per_class': base['roc_auc_per_class'], 'macro': base['roc_auc_macro']},
        'pr_auc_ovr': {'per_class': base['pr_auc_per_class'], 'macro': base['pr_auc_macro']},
        'threshold_sweep': sweep,
        'operating_point_safety_recall': op_point,
        'calibration': {'ece': base['ece'], 'mce': base['mce'], 'n_bins': n_bins,
                        'binning_scheme': 'equal-width confidence bins on max-softmax, [0,1]',
                        'reliability_bins': rb},
        'brier': base['brier'], 'nll': base['nll'],
    }
    if compare_to is not None:
        result['comparison'] = _compare(y_true, pred, correct, n, compare_to)
    return result


def _compare(y_true, pred, correct, n, compare_to):
    out = {'name': compare_to.get('name', 'baseline')}
    other_pred = compare_to.get('pred')
    if other_pred is not None:
        other_pred = np.asarray(other_pred)
        out['mcnemar_exact'] = mcnemar_exact(pred == y_true, other_pred == y_true)
    other_n = compare_to.get('n', n)
    other_correct = compare_to['correct']
    if other_n == n:
        # a = this result (`protocol_tag`), b = compare_to['name'] — 'favours_b' means
        # favours compare_to, NOT this result; disambiguated explicitly below.
        bound = mcnemar_discordance_bound(n, correct, other_correct)
        out['mcnemar_discordance_bound'] = {
            **bound, 'a_is': 'this_result', 'b_is': compare_to.get('name', 'baseline'),
        }
    out['two_proportion_z'] = two_proportion_z(correct, n, other_correct, other_n)
    return out


def render_markdown(result):
    """Headline scalars as a paste-ready markdown table (brief S4.4h)."""
    cols = ['top1', 'macro_accuracy', 'macro_f1', 'balanced_accuracy', 'cohen_kappa', 'matthews_mcc']
    return to_markdown_table({c: result[c] for c in cols})
