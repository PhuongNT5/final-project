"""Tests for feature_ablation.apply on the REAL iSafetyBench tokens (no mocks).

The ablation study's whole value rests on zeroing the column the report claims was zeroed, so these
tests check the column identity against the token file's own feat_layout, not against a constant
copied from the module under test.

Usage: .venv/bin/python tests/test-feature-ablation.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feature_ablation as A

TOK = 'sweep_results/relation_tokens/tokens_isb_all.pt'
TOK_OK = os.path.exists(TOK)
_CACHE = {}


def _tokens():
    if 'd' not in _CACHE:
        _CACHE['d'] = torch.load(TOK, weights_only=False)
    d = _CACHE['d']
    return d['tokens'].float(), d['mask'].float(), d['feat_layout']


def _skip(name):
    print(f"  SKIP  {name} ({TOK} not found -- run from repo root)")


def test_none_is_identity_object():
    X = torch.rand(2, 3, 4, 74)
    assert A.apply(X, torch.ones(2, 3, 4), 'none') is X
    assert A.apply(X, torch.ones(2, 3, 4), None) is X


def test_unknown_mode_raises():
    try:
        A.apply(torch.rand(1, 1, 1, 74), torch.ones(1, 1, 1), 'no_geometry')
        assert False, 'expected ValueError'
    except ValueError:
        pass


def test_layout_mismatch_raises_instead_of_zeroing_wrong_column():
    bad = ['xc', 'class', 'yc', 'w', 'h', 'conf', 'a', 'b', 'c', 'd']
    try:
        A.apply(torch.rand(1, 1, 1, 74), torch.ones(1, 1, 1), 'no_conf', bad)
        assert False, 'expected ValueError on reordered feat_layout'
    except ValueError:
        pass


def test_input_tensor_is_not_mutated():
    X = torch.rand(2, 3, 4, 74)
    before = X.clone()
    for m in A.MODES:
        A.apply(X, torch.ones(2, 3, 4), m)
    assert torch.equal(X, before)


def test_real_tokens_no_conf_zeroes_only_the_conf_column():
    if not TOK_OK:
        return _skip('test_real_tokens_no_conf_zeroes_only_the_conf_column')
    X, M, fl = _tokens()
    ci = list(fl).index('conf')
    Xa = A.apply(X, M, 'no_conf', fl)
    assert float(Xa[..., ci].abs().sum()) == 0.0
    assert float(X[..., ci].abs().sum()) > 0.0                      # there WAS something to remove
    keep = [i for i in range(X.shape[-1]) if i != ci]
    assert torch.equal(Xa[..., keep], X[..., keep])


def test_real_tokens_no_appearance_keeps_all_ten_geometry_columns():
    if not TOK_OK:
        return _skip('test_real_tokens_no_appearance_keeps_all_ten_geometry_columns')
    X, M, fl = _tokens()
    Xa = A.apply(X, M, 'no_appearance', fl)
    assert float(Xa[..., A.N_FEAT:].abs().sum()) == 0.0
    assert float(X[..., A.N_FEAT:].abs().sum()) > 0.0
    assert torch.equal(Xa[..., :A.N_FEAT], X[..., :A.N_FEAT])


def test_real_tokens_blind_class_collapses_class_to_one_value():
    if not TOK_OK:
        return _skip('test_real_tokens_blind_class_collapses_class_to_one_value')
    X, M, fl = _tokens()
    Xa = A.apply(X, M, 'blind_class', fl)
    assert len(torch.unique(Xa[..., 0])) == 1
    assert len(torch.unique(X[..., 0])) > 1
    assert torch.equal(Xa[..., 1:], X[..., 1:])


def test_real_tokens_padding_slots_stay_all_zero():
    if not TOK_OK:
        return _skip('test_real_tokens_padding_slots_stay_all_zero')
    X, M, fl = _tokens()
    pad = (M == 0)
    assert float(X[pad].abs().sum()) == 0.0, 'precondition: builder pads with zeros'
    for m in A.MODES:
        assert float(A.apply(X, M, m, fl)[pad].abs().sum()) == 0.0, m


def test_real_tokens_changed_fraction_is_all_valid_slots():
    if not TOK_OK:
        return _skip('test_real_tokens_changed_fraction_is_all_valid_slots')
    X, M, fl = _tokens()
    # every detection carries a nonzero conf and a nonzero appearance vector, so both ablations must
    # touch 100% of valid slots; anything less means slots are silently empty
    assert A.changed_fraction(X, A.apply(X, M, 'no_conf', fl), M) == 1.0
    assert A.changed_fraction(X, A.apply(X, M, 'no_appearance', fl), M) == 1.0
    assert A.changed_fraction(X, A.apply(X, M, 'none', fl), M) == 0.0


def test_real_tokens_boxes_only_leaves_exactly_the_four_box_columns():
    if not TOK_OK:
        return _skip('test_real_tokens_boxes_only_leaves_exactly_the_four_box_columns')
    X, M, fl = _tokens()
    Xa = A.apply(X, M, 'boxes_only', fl)
    keep = {list(fl).index(c) for c in A.BOX_COLS}
    for i in range(A.N_FEAT):
        nonzero = float(Xa[..., i].abs().sum()) > 0
        assert nonzero == (i in keep), f'{fl[i]} nonzero={nonzero}'
    assert float(Xa[..., A.N_FEAT:].abs().sum()) == 0.0
    for i in keep:
        assert torch.equal(Xa[..., i], X[..., i])            # surviving columns untouched


def test_real_tokens_no_boxes_is_the_converse_of_boxes_only():
    if not TOK_OK:
        return _skip('test_real_tokens_no_boxes_is_the_converse_of_boxes_only')
    X, M, fl = _tokens()
    Xb = A.apply(X, M, 'no_boxes', fl)
    for c in A.BOX_COLS:
        assert float(Xb[..., list(fl).index(c)].abs().sum()) == 0.0
    assert float(Xb[..., list(fl).index('conf')].abs().sum()) > 0.0
    assert float(Xb[..., A.N_FEAT:].abs().sum()) > 0.0
    # union of what the two modes keep must cover every column that carried signal
    Xa = A.apply(X, M, 'boxes_only', fl)
    kept = (Xa.abs().sum((0, 1, 2)) > 0) | (Xb.abs().sum((0, 1, 2)) > 0)
    had = X.abs().sum((0, 1, 2)) > 0
    assert torch.equal(kept, had)


def test_real_tokens_appearance_only_is_the_converse_of_no_appearance():
    if not TOK_OK:
        return _skip('test_real_tokens_appearance_only_is_the_converse_of_no_appearance')
    X, M, fl = _tokens()
    Xa = A.apply(X, M, 'appearance_only', fl)
    assert float(Xa[..., :A.N_FEAT].abs().sum()) == 0.0
    assert torch.equal(Xa[..., A.N_FEAT:], X[..., A.N_FEAT:])
    Xn = A.apply(X, M, 'no_appearance', fl)
    kept = (Xa.abs().sum((0, 1, 2)) > 0) | (Xn.abs().sum((0, 1, 2)) > 0)
    assert torch.equal(kept, X.abs().sum((0, 1, 2)) > 0)


def test_ablating_conf_degrades_the_count_rule_floor_feature():
    """The conf ablation must reach the R42 floor too: isafetybench_dataset.count_features reads the
    same column, so mean_conf has to collapse to 0 while the two count features survive."""
    if not TOK_OK:
        return _skip('test_ablating_conf_degrades_the_count_rule_floor_feature')
    import isafetybench_dataset as ISB
    X, M, fl = _tokens()
    F0 = ISB.count_features(X, M)
    F1 = ISB.count_features(A.apply(X, M, 'no_conf', fl), M)
    assert F0[:, 2].std() > 0.01                       # mean_conf varied before
    assert np.abs(F1[:, 2]).max() == 0.0               # and is gone after
    assert np.allclose(F0[:, :2], F1[:, :2])           # persons/objects untouched


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
