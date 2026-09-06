"""4c safety cost + 4d operating-point selector."""
import numpy as np

from eval_utils.cost_matrix import default_cost_matrix, total_cost, cost_report
from eval_utils.safety_metrics import cost_sensitive_score
from eval_utils.operating_point import confidence_threshold_sweep, select_operating_point_by_safety_recall


def test_default_cost_matrix_shape_and_diagonal():
    m = default_cost_matrix(4)
    assert m.shape == (4, 4)
    assert np.all(np.diag(m) == 0)
    assert np.all(m[~np.eye(4, dtype=bool)] == 10.0)  # all-unsafe default -> every off-diag = miss_cost


def test_cost_matrix_backward_compatible_with_cost_sensitive_score():
    """The declared-matrix total cost must equal the pre-existing scalar helper's
    output under equivalent parameters (all classes unsafe, same miss/false-alarm costs)."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=200)
    pred = rng.integers(0, 4, size=200)
    m = default_cost_matrix(4, unsafe_classes=range(4), miss_cost=10.0, false_alarm_cost=1.0)
    got = total_cost(y, pred, m)
    ref = cost_sensitive_score(y, pred, miss_cost=10.0, false_alarm_cost=1.0, unsafe_classes=(0, 1, 2, 3))
    assert abs(got - ref) < 1e-9, (got, ref)


def test_cost_matrix_penalises_miss_more_than_false_alarm():
    """8-class ontology: classes 0-3 unsafe, 4-7 safe. Missing an unsafe clip (true
    unsafe, pred safe) must cost strictly more than a false alarm (true safe, pred
    unsafe), given equal counts of each error type."""
    y = np.array([0, 4])       # one true-unsafe, one true-safe
    pred = np.array([4, 0])    # both errors: miss + false alarm
    m = default_cost_matrix(8, unsafe_classes=range(4))
    miss_only = total_cost(np.array([0]), np.array([4]), m)
    fa_only = total_cost(np.array([4]), np.array([0]), m)
    assert miss_only > fa_only, (miss_only, fa_only)


def test_cost_report_declares_matrix_alongside_number():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 0])
    out = cost_report(y, pred, n_classes=4)
    assert 'cost_matrix' in out and 'total_cost' in out and 'mean_cost' in out
    assert len(out['cost_matrix']) == 4


def test_threshold_sweep_monotonic_coverage():
    """Coverage (fraction fired) must be non-increasing as tau increases."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, 4, size=100)
    probs = rng.dirichlet([1, 1, 1, 1], size=100)
    taus = [0.0, 0.3, 0.5, 0.7, 0.9]
    rows = confidence_threshold_sweep(y, probs, thresholds=taus)
    coverages = [r['coverage'] for r in rows]
    assert all(coverages[i] >= coverages[i + 1] - 1e-9 for i in range(len(coverages) - 1))


def test_operating_point_selected_by_safety_recall_not_accuracy():
    """Construct a case where the highest-accuracy tau is NOT the one meeting the
    target recall, and confirm the selector picks the recall-satisfying one."""
    # 10 clips: confidences such that tau=0.9 keeps only the 5 easy (always-correct)
    # clips (100% acc, 50% coverage); tau=0.0 keeps all 10 (70% acc, 100% coverage).
    y = np.array([0] * 10)
    probs = np.zeros((10, 2))
    probs[:5, 0] = 0.95   # confident + correct
    probs[5:8, 0] = 0.95  # confident but WRONG -> pred class 0 still (prob col0 highest) so make wrong differently
    probs[5:8, 1] = 0.05
    probs[5:8] = probs[5:8][:, ::-1]  # now class1 has 0.95 -> pred=1, wrong (true=0)
    probs[8:, 0] = 0.4; probs[8:, 1] = 0.6  # low-confidence, also wrong (pred=1)
    probs[:5, 1] = 0.05
    out = select_operating_point_by_safety_recall(y, probs, target_recall=0.95)
    assert out['coverage'] >= 0.95 - 1e-9
    assert out['test_selected'] is True


def test_operating_point_uses_independent_selection_set_when_given():
    rng = np.random.default_rng(4)
    y_sel = rng.integers(0, 2, size=50)
    probs_sel = rng.dirichlet([2, 2], size=50)
    y_test = rng.integers(0, 2, size=50)
    probs_test = rng.dirichlet([2, 2], size=50)
    out = select_operating_point_by_safety_recall(
        y_test, probs_test, target_recall=0.8,
        selection_y_true=y_sel, selection_y_prob=probs_sel)
    assert out['test_selected'] is False


if __name__ == '__main__':
    import sys
    from eval_utils.tests.run_all import run_module
    sys.exit(run_module(sys.modules[__name__]))
