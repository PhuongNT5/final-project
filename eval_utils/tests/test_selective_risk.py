"""Risk-coverage curve / AURC: hand-derived tiny examples + oracle-vs-model
ordering sanity checks."""
import numpy as np

from eval_utils.selective_risk import (
    risk_coverage_curve, oracle_risk_coverage_curve, aurc, excess_aurc,
    risk_at_coverage, coverage_at_risk,
)


def test_risk_coverage_curve_hand_computed():
    # confidences descending order already: rows 0,1 correct, rows 2,3 wrong
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.3, 0.7]])
    labels = np.array([0, 0, 0, 0])
    cov, risk = risk_coverage_curve(probs, labels)
    assert np.allclose(cov, [0.25, 0.5, 0.75, 1.0])
    # preds = argmax = [0,0,1,1] -> correct=[T,T,F,F], ranked by confidence
    # (already descending: 0.9,0.8,0.6,0.7 -> order stays [0,1,3,2] by conf desc)
    # cumulative risk after re-sorting by descending confidence:
    conf = probs.max(1)
    order = np.argsort(-conf, kind='stable')
    correct = (probs.argmax(1) == labels)[order]
    expected_risk = 1 - np.cumsum(correct.astype(float)) / np.arange(1, 5)
    assert np.allclose(risk, expected_risk)


def test_oracle_curve_is_never_worse_than_model_curve_pointwise_on_aurc():
    rng = np.random.default_rng(0)
    n, C = 64, 4
    labels = rng.integers(0, C, n)
    probs = rng.dirichlet(np.ones(C), size=n)
    out = excess_aurc(probs, labels)
    assert out['excess_aurc'] >= -1e-9  # oracle AURC <= model AURC (oracle ranks correctness perfectly)
    assert out['aurc_oracle'] <= out['aurc'] + 1e-9


def test_aurc_perfect_model_is_zero():
    n = 20
    labels = np.zeros(n, dtype=int)
    probs = np.zeros((n, 2))
    probs[:, 0] = 1.0  # always fully confident and always correct
    cov, risk = risk_coverage_curve(probs, labels)
    assert np.allclose(risk, 0.0)
    assert abs(aurc(cov, risk)) < 1e-12


def test_aurc_always_wrong_model_is_one():
    n = 10
    labels = np.zeros(n, dtype=int)
    probs = np.zeros((n, 2))
    probs[:, 1] = 1.0  # always confidently wrong
    cov, risk = risk_coverage_curve(probs, labels)
    assert np.allclose(risk, 1.0)
    assert abs(aurc(cov, risk) - 1.0) < 1e-9


def test_risk_at_coverage_matches_manual_slice():
    rng = np.random.default_rng(3)
    n, C = 50, 4
    labels = rng.integers(0, C, n)
    probs = rng.dirichlet(np.ones(C), size=n)
    out = risk_at_coverage(probs, labels, coverage_target=0.8)
    cov, risk = risk_coverage_curve(probs, labels)
    m = int(round(0.8 * n))
    assert out['m'] == m
    assert abs(out['risk'] - risk[m - 1]) < 1e-12


def test_coverage_at_risk_unreachable_reports_none():
    n = 10
    labels = np.zeros(n, dtype=int)
    probs = np.zeros((n, 2)); probs[:, 1] = 1.0  # always wrong -> risk=1 everywhere
    out = coverage_at_risk(probs, labels, risk_target=0.01)
    assert out['coverage'] is None


if __name__ == '__main__':
    import sys
    from eval_utils.tests.run_all import run_module
    sys.exit(run_module(sys.modules[__name__]))
