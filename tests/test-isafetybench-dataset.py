"""Unit tests for isafetybench_dataset.py — token contract, leave-source-out fold integrity, and the
R42 count-rule feature definitions. Reads the REAL tokens_isb_all.pt (no mocks), so a pass is also a
regression check that the token build still matches the documented corpus counts.

No pytest in the project venv (see tests/test_dedupe_binary.py precedent) — plain assert functions
named test_*, runnable standalone.

Usage: .venv/bin/python tests/test-isafetybench-dataset.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isafetybench_dataset as ISB

TOK_OK = os.path.exists(f'{ISB.TOK}/tokens_isb_all.pt')
_CACHE = {}


def _load():
    if 'd' not in _CACHE:
        _CACHE['d'] = ISB.load('isb_')
    return _CACHE['d']


def _skip(name):
    print(f"  SKIP  {name} (tokens_isb_all.pt not found — run build_isafetybench_tokens.py first)")


def test_load_shapes_and_label_domain():
    if not TOK_OK:
        return _skip('test_load_shapes_and_label_domain')
    X, M, y, groups, cids = _load()
    assert X.shape[1:] == (16, 8, 74), X.shape          # T=16, N_MAX=8, FEAT 10 + APP 64
    assert M.shape == X.shape[:3], M.shape
    assert len(y) == len(groups) == len(cids) == X.shape[0]
    assert set(y.unique().tolist()) == {0, 1}, y.unique()
    assert not torch.isnan(X).any()


def test_corpus_counts_match_feasibility_report():
    if not TOK_OK:
        return _skip('test_corpus_counts_match_feasibility_report')
    X, M, y, groups, cids = _load()
    assert len(y) == 1100, len(y)                        # 420 hazard + 680 normal on disk
    assert int(y.sum()) == 420, int(y.sum())
    assert len(set(groups)) >= 140, len(set(groups))     # ~150 distinct YouTube sources


def test_every_source_is_single_class_r41():
    """R41: 0/150 sources carry both classes. If this ever fails the grouping assumption changed and
    the leave-source-out protocol must be re-argued, so assert it rather than trusting the report."""
    if not TOK_OK:
        return _skip('test_every_source_is_single_class_r41')
    _, _, y, groups, _ = _load()
    yn = y.numpy()
    mixed = [g for g in set(groups) if len(set(yn[groups == g].tolist())) > 1]
    assert mixed == [], mixed


def test_geometry_columns_are_zero_and_declared():
    """iSafetyBench ships no zone polygons, so cols 6-9 must be exactly zero — a non-zero value would
    mean the builder silently picked up Unsafe-Net zones."""
    if not TOK_OK:
        return _skip('test_geometry_columns_are_zero_and_declared')
    X, _, _, _, _ = _load()
    assert bool((X[..., 6:10] == 0).all())


def test_folds_never_split_a_source():
    if not TOK_OK:
        return _skip('test_folds_never_split_a_source')
    _, _, y, groups, _ = _load()
    splits = ISB.folds(y, groups, n_splits=10, seed=0)
    assert len(splits) == 10
    covered = np.zeros(len(y), dtype=bool)
    for tr, te in splits:
        assert set(groups[tr]).isdisjoint(set(groups[te]))    # leave-SOURCE-out, not leave-clip-out
        assert len(set(y.numpy()[tr].tolist())) == 2          # both classes present in every train pool
        assert not covered[te].any()                          # folds are disjoint
        covered[te] = True
    assert covered.all()                                      # and exhaustive -> pooled OOF is complete


def test_count_features_shape_and_ranges():
    if not TOK_OK:
        return _skip('test_count_features_shape_and_ranges')
    X, M, y, _, _ = _load()
    F = ISB.count_features(X, M)
    assert F.shape == (len(y), 3), F.shape
    assert (F[:, 0] <= F[:, 1] + 1e-6).all()                  # persons/frame <= objects/frame
    assert (F[:, 1] <= 8.0 + 1e-6).all()                      # at most N_MAX boxes per frame
    empty = M.sum(dim=(1, 2)).numpy() == 0
    assert (F[empty, 2] == 0).all()                           # no detections -> mean conf 0, not NaN
    assert not np.isnan(F).any()


def test_count_features_persons_match_manual_count():
    """Cross-check the vectorised persons/frame against an explicit loop on a 5-clip slice."""
    if not TOK_OK:
        return _skip('test_count_features_persons_match_manual_count')
    X, M, y, _, _ = _load()
    F = ISB.count_features(X[:5], M[:5])
    for i in range(5):
        n = sum(1 for t in range(X.shape[1]) for j in range(X.shape[2])
                if M[i, t, j] > 0 and int(X[i, t, j, 0]) == ISB.PERSON_GROUP)
        assert abs(F[i, 0] - n / X.shape[1]) < 1e-5, (i, F[i, 0], n / X.shape[1])


if __name__ == '__main__':
    import traceback
    failures = 0
    mod = sys.modules[__name__]
    for name in sorted(dir(mod)):
        if not name.startswith('test_'):
            continue
        try:
            getattr(mod, name)()
            print(f"  PASS  {name}")
        except Exception:
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
    sys.exit(1 if failures else 0)
