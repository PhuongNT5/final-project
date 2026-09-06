"""Split-conformal prediction for the 4-class clip head (Vovk, Gammerman & Shafer
2005; Angelopoulos & Bates 2023 tutorial, arXiv:2107.07511).

Finite-sample-exact MARGINAL coverage guarantee: P(Y in C(X)) >= 1-alpha for any
finite n, any distribution, as long as calibration/test are exchangeable. This
holds even at n=64 -- unlike every other framework in this project, conformal's
VALIDITY is not an asymptotic approximation (only its RESOLUTION is coarse at
small n, see split_conformal_repeated).

Class-conditional (Mondrian) guarantees are NOT implemented here: the smallest
class in this benchmark has n=8, too few to support even one meaningful
per-class conformal quantile (a step size of ~1/9 ~= 11pp). This gives a
coverage 'guarantee' that cannot move by less than one whole clip's worth of
mass and is not meaningfully verifiable. class_conditional_conformal() raises
rather than silently returning a number that looks rigorous but isn't.
"""
from __future__ import annotations
import numpy as np


def nonconformity_score(probs, labels):
    """LAC / softmax nonconformity score: 1 - P(true label). Lower = more
    conforming. probs: (n, C) softmax rows. labels: (n,) int class ids."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    return 1.0 - probs[np.arange(len(labels)), labels]


def conformal_quantile(cal_scores, alpha):
    """Empirical ceil((n_cal+1)(1-alpha))/n_cal quantile of calibration
    nonconformity scores (Vovk et al. 2005, Lemma 1). The level is clipped to
    1.0 when n_cal is small enough that the exact level would exceed 1 -- the
    guarantee still holds (the resulting set trivially covers every class for
    such calibration draws), it just cannot be resolved above that alpha with
    this n_cal."""
    cal_scores = np.asarray(cal_scores, dtype=float)
    n_cal = len(cal_scores)
    if n_cal < 1:
        raise ValueError('need at least 1 calibration point')
    level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
    return float(np.quantile(cal_scores, level, method='higher'))


def prediction_sets(probs_test, q_hat):
    """C(x) = {y : prob(x)_y >= 1 - q_hat}, one index array per test row."""
    probs_test = np.asarray(probs_test, dtype=float)
    thresh = 1.0 - q_hat
    return [np.where(row >= thresh)[0] for row in probs_test]


def evaluate_prediction_sets(pred_sets, labels_test):
    """coverage = fraction of test rows whose true label is IN its set;
    avg_set_size = mean |C(x)| (the informative quantity -- size 1 is a
    confident decision, size ~C is a near-abstention)."""
    labels_test = np.asarray(labels_test)
    covered = np.array([labels_test[i] in pred_sets[i] for i in range(len(pred_sets))])
    sizes = np.array([len(s) for s in pred_sets])
    return {'coverage': float(covered.mean()), 'avg_set_size': float(sizes.mean()),
            'covered': covered, 'sizes': sizes}


def split_conformal_once(probs, labels, alpha, cal_idx, test_idx):
    """One calibration/test split -> coverage + avg set size on that split."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    cal_scores = nonconformity_score(probs[cal_idx], labels[cal_idx])
    q_hat = conformal_quantile(cal_scores, alpha)
    pred_sets = prediction_sets(probs[test_idx], q_hat)
    return evaluate_prediction_sets(pred_sets, labels[test_idx])


def split_conformal_repeated(probs, labels, alpha, cal_frac=0.5, n_splits=1000, seed=0):
    """Average marginal coverage + set size over `n_splits` random cal/test
    splits of the SAME n clips. At n=64 a single split is a noisy point
    estimate of empirical coverage (Angelopoulos & Bates 2023 sec 3) -- this
    reports the distribution across splits, not one lucky draw.

    MARGINAL coverage only, by design -- see module docstring and
    class_conditional_conformal() for why per-class is refused, not just
    unreported."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    n = len(labels)
    n_cal = max(2, int(round(n * cal_frac)))
    if n_cal >= n:
        raise ValueError(f'cal_frac too large: n_cal={n_cal} >= n={n}')
    rng = np.random.default_rng(seed)
    coverages = np.empty(n_splits)
    sizes = np.empty(n_splits)
    for s in range(n_splits):
        perm = rng.permutation(n)
        cal_idx, test_idx = perm[:n_cal], perm[n_cal:]
        res = split_conformal_once(probs, labels, alpha, cal_idx, test_idx)
        coverages[s] = res['coverage']
        sizes[s] = res['avg_set_size']
    return {
        'nominal_coverage': 1 - alpha, 'n_splits': n_splits, 'cal_frac': cal_frac,
        'n_cal': n_cal, 'n_test': n - n_cal,
        'coverage_mean': float(coverages.mean()), 'coverage_std': float(coverages.std()),
        'coverage_p05': float(np.percentile(coverages, 5)),
        'coverage_p95': float(np.percentile(coverages, 95)),
        'set_size_mean': float(sizes.mean()), 'set_size_std': float(sizes.std()),
    }


def class_conditional_conformal(*_args, **_kwargs):
    """Deliberately unimplemented -- see module docstring. Raises instead of
    emitting a per-class coverage guarantee that this benchmark's smallest
    class (n=8) cannot support."""
    raise NotImplementedError(
        'Per-class (Mondrian) conformal guarantees are not attainable at this '
        'N (smallest class n=8): quantile granularity ~1/9 (~11pp) makes a '
        'per-class coverage guarantee unverifiable. Use split_conformal_repeated() '
        'for the marginal (whole-test) guarantee instead.')
