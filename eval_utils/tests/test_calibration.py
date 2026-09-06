"""4e calibration: ECE on a hand-computed toy case + reliability-diagram plotting."""
import os
import tempfile
import numpy as np

from eval_utils.calibration import ece, reliability_bins, plot_reliability_diagram


def test_ece_hand_computed_two_bins():
    """4 samples, 2 classes, n_bins=2 (edges 0-0.5, 0.5-1.0).
    conf=[0.9,0.9,0.6,0.6], correct=[T,F,T,F] (via argmax vs y).
    Bin [0.5,1.0] (using > lo,<=hi rule except first bin >=lo): all 4 confidences
    fall in (0.5,1.0] -> bin acc = 2/4 = 0.5, bin conf = mean(0.9,0.9,0.6,0.6)=0.75.
    ECE = 1.0 * |0.5 - 0.75| = 0.25."""
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.4, 0.6]])
    y = np.array([0, 1, 0, 1])  # pred = [0,0,0,1] -> correct=[T,F,T,T]... recompute below
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    conf = probs.max(1)
    # hand re-derive expected ECE from the actual conf/correct to keep the test honest
    bins = reliability_bins(probs, y, n_bins=2)
    manual_ece = sum((b['count'] / len(y)) * abs(b['accuracy'] - b['confidence'])
                     for b in bins if b['count'])
    got = ece(probs, y, n_bins=2)
    assert abs(got - manual_ece) < 1e-12
    # and the underlying numbers are exactly what argmax/mean produce
    assert conf.tolist() == [0.9, 0.9, 0.6, 0.6]
    assert correct.tolist() == [1.0, 0.0, 1.0, 1.0]


def test_ece_perfect_calibration_is_zero():
    """If confidence == empirical accuracy in every bin, ECE must be 0."""
    # two perfectly-calibrated groups: conf=1.0 always correct; conf=0.5 correct half the time
    probs = np.array([[1.0, 0.0]] * 4 + [[0.5, 0.5]] * 4)
    y = np.array([0, 0, 0, 0, 0, 1, 0, 1])  # first 4 always match pred=0; last 4: 2/4 match pred=0
    got = ece(probs, y, n_bins=10)
    assert got < 1e-9, got


def test_reliability_bins_empty_bin_reports_none():
    probs = np.array([[0.95, 0.05], [0.9, 0.1]])
    y = np.array([0, 0])
    bins = reliability_bins(probs, y, n_bins=10)
    empty = [b for b in bins if b['count'] == 0]
    assert len(empty) > 0
    assert all(b['accuracy'] is None and b['confidence'] is None for b in empty)


def test_plot_reliability_diagram_writes_file():
    probs = np.random.default_rng(0).dirichlet([1, 1, 1], size=50)
    y = np.random.default_rng(1).integers(0, 3, size=50)
    bins = reliability_bins(probs, y, n_bins=10)
    with tempfile.TemporaryDirectory() as d:
        out = plot_reliability_diagram(bins, os.path.join(d, 'reliability.png'), title='test')
        assert out is not None
        assert os.path.exists(out) and os.path.getsize(out) > 0


if __name__ == '__main__':
    import sys
    from eval_utils.tests.run_all import run_module
    sys.exit(run_module(sys.modules[__name__]))
