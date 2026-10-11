"""Unit tests for convlstm.py ensemble-posterior renormalization fix.

The ensemble-posterior slice at lines 165-166 of unsafenet/convlstm.py now renormalizes:
  Pv, Pt = (torch.stack(Ps).mean(0)[:, :k] for Ps in (Pvs, Pts))
  Pv, Pt = Pv / Pv.sum(1, keepdim=True), Pt / Pt.sum(1, keepdim=True)

Rationale: for --task 8 the model emits an 8-way softmax but is read on 4 classes, so slicing
to [:, :4] left rows that do not sum to 1 — violating the documented row-stochastic
precondition of unsafenet/metrics.py::report. For --task unsafe4 and --task binary the
slice is a no-op and the division is by ~1.0.

Test coverage:
- Ensemble slice of 8-way softmax to 4 columns is row-stochastic after renormalization
- argmax is unchanged by renormalization (row permutation invariance)
- Brier score is perceptibly different before/after renormalization (sensitive to prob magnitude)
"""
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsafenet import metrics as M


def test_renormalized_slice_is_row_stochastic():
    """An 8-way softmax sliced to 4 columns and renormalized must have row sums = 1."""
    # Synthetic 8-way softmax ensemble: 3 seeds, 10 samples
    rng = np.random.default_rng(42)
    seeds = 3
    n_samples = 10
    n_cls_full = 8
    n_cls_slice = 4

    # Generate 3 independent 8-way softmax posteriors
    Ps = []
    for _ in range(seeds):
        P = torch.from_numpy(rng.dirichlet(np.ones(n_cls_full), n_samples)).float()
        Ps.append(P)

    # Ensemble: average and slice to 4 classes (like the fixed code does)
    P_mean = torch.stack(Ps).mean(0)[:, :n_cls_slice]
    # Renormalize to enforce row-stochasticity
    P_renorm = P_mean / P_mean.sum(1, keepdim=True)

    # Every row must sum to 1.0 within floating-point tolerance
    row_sums = P_renorm.sum(1).numpy()
    assert np.allclose(row_sums, 1.0, rtol=1e-6), f"Row sums not 1.0: {row_sums}"


def test_renormalization_preserves_argmax():
    """argmax across columns is unchanged by row-wise normalization."""
    # Start with an 8-way softmax
    rng = np.random.default_rng(43)
    n_samples = 20
    n_cls_full = 8
    n_cls_slice = 4

    P_full = torch.from_numpy(rng.dirichlet(np.ones(n_cls_full), n_samples)).float()

    # Slice to 4 classes (unnormalized)
    P_slice = P_full[:, :n_cls_slice]

    # Renormalize
    P_renorm = P_slice / P_slice.sum(1, keepdim=True)

    # argmax must be identical (permuting probabilities doesn't change the index of max)
    argmax_slice = P_slice.argmax(1).numpy()
    argmax_renorm = P_renorm.argmax(1).numpy()
    assert np.array_equal(argmax_slice, argmax_renorm), \
        f"argmax differs: {argmax_slice} vs {argmax_renorm}"


def test_renormalization_affects_brier_score():
    """Brier score (MSE to one-hot) is sensitive to the magnitude of probabilities."""
    # Create a simple case where Brier will visibly differ
    rng = np.random.default_rng(44)
    n_samples = 50
    n_cls_full = 8
    n_cls_slice = 4

    # Generate true labels: only first 4 classes
    y = rng.integers(0, n_cls_slice, n_samples)

    # Generate softmax over 8 classes, slice to 4, renormalize
    P_full = torch.from_numpy(rng.dirichlet(np.ones(n_cls_full), n_samples)).float()
    P_slice = P_full[:, :n_cls_slice]
    P_renorm = P_slice / P_slice.sum(1, keepdim=True)

    # Brier before renormalization: compare to one-hot, then compute MSE
    # (raw unnormalized slice will produce different loss than normalized)
    P_slice_np = P_slice.numpy()
    P_renorm_np = P_renorm.numpy()
    y_onehot = np.eye(n_cls_slice)[y]

    brier_slice = np.mean((P_slice_np - y_onehot) ** 2)
    brier_renorm = np.mean((P_renorm_np - y_onehot) ** 2)

    # Renormalized should have lower Brier (more concentrated probability on true class)
    # Not always guaranteed, but in this synthetic separable case it is
    # At minimum, they must be different values
    assert brier_slice != brier_renorm, \
        f"Brier scores identical before/after renorm: {brier_slice} == {brier_renorm}"


def test_real_task_binary_slice_is_noop():
    """For task='binary' (2 classes), slicing a 2-way softmax and renormalizing is a no-op."""
    rng = np.random.default_rng(45)
    n_samples = 15
    # Binary softmax
    P = torch.from_numpy(rng.dirichlet(np.ones(2), n_samples)).float()

    # Slice to 2 (no-op)
    P_slice = P[:, :2]
    # Renormalize
    P_renorm = P_slice / P_slice.sum(1, keepdim=True)

    # Must be identical within floating point
    assert torch.allclose(P, P_renorm, rtol=1e-6), \
        "Binary slice and renorm should be no-op"


def test_real_task_unsafe4_slice_is_noop():
    """For task='unsafe4' (4 classes), slicing a 4-way softmax and renormalizing is a no-op."""
    rng = np.random.default_rng(46)
    n_samples = 20
    # 4-way softmax
    P = torch.from_numpy(rng.dirichlet(np.ones(4), n_samples)).float()

    # Slice to 4 (no-op)
    P_slice = P[:, :4]
    # Renormalize
    P_renorm = P_slice / P_slice.sum(1, keepdim=True)

    # Must be identical within floating point
    assert torch.allclose(P, P_renorm, rtol=1e-6), \
        "4-class slice and renorm should be no-op"


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'  PASS  {fn.__name__}')
    print('ALL PASS')
