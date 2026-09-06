"""Split-conformal: hand-derived small examples + finite-sample coverage
guarantee sanity check (empirical marginal coverage should track nominal
1-alpha on average, per Vovk et al. 2005 Lemma 1)."""
import numpy as np

from eval_utils.conformal import (
    nonconformity_score, conformal_quantile, prediction_sets,
    evaluate_prediction_sets, split_conformal_once, split_conformal_repeated,
    class_conditional_conformal,
)


def test_nonconformity_score_hand_computed():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    labels = np.array([0, 2])
    got = nonconformity_score(probs, labels)
    assert np.allclose(got, [0.3, 0.4])


def test_conformal_quantile_matches_manual_higher_quantile():
    cal_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    n_cal = len(cal_scores)
    alpha = 0.10
    level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    expected = np.quantile(cal_scores, level, method='higher')
    got = conformal_quantile(cal_scores, alpha)
    assert abs(got - expected) < 1e-12


def test_conformal_quantile_clips_level_to_one_when_ncal_small():
    # n_cal=4, alpha=0.05 -> (5)(0.95)/4 = 1.1875 > 1 -> must clip to 1.0 (= max)
    cal_scores = np.array([0.2, 0.4, 0.6, 0.8])
    q = conformal_quantile(cal_scores, alpha=0.05)
    assert q == cal_scores.max()


def test_prediction_sets_and_coverage_hand_computed():
    # 2 test rows, 3 classes; q_hat=0.3 -> threshold = 0.7
    probs_test = np.array([[0.8, 0.1, 0.1], [0.5, 0.3, 0.2]])
    labels_test = np.array([0, 0])
    sets = prediction_sets(probs_test, q_hat=0.3)
    assert sets[0].tolist() == [0]           # only class 0 >= 0.7
    assert sets[1].tolist() == []            # no class >= 0.7 -> empty set
    res = evaluate_prediction_sets(sets, labels_test)
    assert res['coverage'] == 0.5            # row0 covered, row1 not (label not in empty set)
    assert res['avg_set_size'] == 0.5        # (1+0)/2


def test_split_conformal_once_perfectly_confident_model_gives_size_one_sets():
    rng = np.random.default_rng(0)
    n, C = 40, 4
    labels = rng.integers(0, C, n)
    probs = np.full((n, C), 0.01)
    probs[np.arange(n), labels] = 1 - 0.01 * (C - 1)  # near-certain on true label
    cal_idx, test_idx = np.arange(20), np.arange(20, 40)
    res = split_conformal_once(probs, labels, alpha=0.10, cal_idx=cal_idx, test_idx=test_idx)
    assert res['coverage'] == 1.0
    assert res['avg_set_size'] <= 1.5  # near-perfectly confident -> tight sets


def test_split_conformal_repeated_marginal_coverage_near_nominal():
    """A reasonably-confident model's empirical marginal coverage, averaged over
    many random splits, should be close to (not necessarily >=, at tiny n_cal per
    split) the nominal 1-alpha target -- this is the qualitative guarantee check,
    not a proof of the theorem."""
    rng = np.random.default_rng(1)
    n, C = 64, 4
    labels = rng.integers(0, C, n)
    probs = rng.dirichlet(np.ones(C) * 0.5, size=n)
    # boost true-label mass so the model is informative, not random
    probs[np.arange(n), labels] += 1.5
    probs = probs / probs.sum(axis=1, keepdims=True)
    alpha = 0.10
    out = split_conformal_repeated(probs, labels, alpha=alpha, cal_frac=0.5, n_splits=300, seed=2)
    assert out['nominal_coverage'] == 0.90
    assert abs(out['coverage_mean'] - 0.90) < 0.12  # loose band, small-n regime
    assert out['set_size_mean'] > 0


def test_split_conformal_repeated_rejects_cal_frac_too_large():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=10)
    labels = rng.integers(0, 3, 10)
    try:
        split_conformal_repeated(probs, labels, alpha=0.1, cal_frac=1.0, n_splits=5)
        assert False, 'expected ValueError'
    except ValueError:
        pass


def test_class_conditional_conformal_refuses():
    try:
        class_conditional_conformal()
        assert False, 'expected NotImplementedError'
    except NotImplementedError:
        pass


if __name__ == '__main__':
    import sys
    from eval_utils.tests.run_all import run_module
    sys.exit(run_module(sys.modules[__name__]))
