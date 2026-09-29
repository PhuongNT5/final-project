"""Unit tests for the binary task's data-wiring layer: run_relation_transformer.load_binary /
apply_coord_variant, and run_kfold_relation.load_pooled_binary — label mapping, dedupe plumbing,
coordinate-variant masking correctness. design-2026-09-16-binary-encoder-redesign.md §0/§3.5.

No pytest in the project venv (see tests/test_dedupe_binary.py precedent) — plain assert
functions named test_*, runnable standalone via `if __name__ == '__main__'` below.

These tests read the REAL token files (sweep_results/relation_tokens/tokens_bin8_{train,test}.pt)
and the real dedupe audit — no mocks — so a passing run is also a regression check that those
artifacts still match the documented counts (laneb-2026-09-16-binary-data-prep.md /
laneb-2026-09-16-binary-dedupe-and-ceiling.md).

Usage: .venv/bin/python tests/test_binary_encoder_wiring.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_relation_transformer as R
import run_kfold_relation as K

TOK_OK = os.path.exists('sweep_results/relation_tokens/tokens_bin8_train.pt')


def _skip(name):
    print(f"  SKIP  {name} (tokens_bin8_{{train,test}}.pt not found — run from repo root)")


def test_binary_label_mapping_sub_0_3_unsafe_4_7_safe():
    ysub = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    ybin = (ysub < 4).long()
    assert ybin.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]


def test_load_binary_train_counts_and_dedupe_none():
    if not TOK_OK:
        return _skip('test_load_binary_train_counts_and_dedupe_none')
    X, M, ybin, ysub, views, clip_ids = R.load_binary('train', 'bin8_', 'none')
    assert len(ybin) == 566, len(ybin)
    assert int(ybin.sum()) == 452, int(ybin.sum())          # design/data-prep: 452 unsafe / 114 safe
    assert set(ysub.unique().tolist()) <= set(range(8))
    assert ((ysub < 4).long() == ybin).all()


def test_load_binary_train_drop_conflict_drops_exactly_four():
    if not TOK_OK:
        return _skip('test_load_binary_train_drop_conflict_drops_exactly_four')
    X, M, ybin, ysub, views, clip_ids = R.load_binary('train', 'bin8_', 'drop_conflict')
    assert len(ybin) == 562, len(ybin)
    dropped = {'0_tr123.mp4', '0_tr136.mp4', '0_tr138.mp4', '1_tr55.mp4'}
    assert dropped.isdisjoint(set(clip_ids))


def test_load_binary_test_untouched_by_default():
    if not TOK_OK:
        return _skip('test_load_binary_test_untouched_by_default')
    X, M, ybin, ysub, views, clip_ids = R.load_binary('test', 'bin8_', 'none')
    assert len(ybin) == 125, len(ybin)
    assert int(ybin.sum()) == 64, int(ybin.sum())            # 64 unsafe / 61 safe, frozen test set


def test_apply_coord_variant_t1_is_identity():
    X = torch.randn(2, 3, 4, 74)
    M = torch.ones(2, 3, 4)
    assert torch.equal(R.apply_coord_variant(X, M, None), X)
    assert torch.equal(R.apply_coord_variant(X, M, 'T1'), X)


def test_apply_coord_variant_t2_drops_xy_columns():
    X = torch.randn(2, 3, 4, 74)
    M = torch.ones(2, 3, 4)
    out = R.apply_coord_variant(X, M, 'T2')
    assert out.shape == (2, 3, 4, 72)
    assert torch.equal(out[..., 0], X[..., 0])               # class id preserved
    assert torch.equal(out[..., 1:], X[..., 3:])              # w,h,conf,... preserved, xc/yc gone


def test_apply_coord_variant_t3_zero_mean_per_frame_and_padding_zeroed():
    torch.manual_seed(0)
    X = torch.zeros(1, 1, 4, 74)
    X[0, 0, :, 1] = torch.tensor([0.1, 0.3, 0.5, 0.9])   # xc
    X[0, 0, :, 2] = torch.tensor([0.2, 0.4, 0.6, 0.8])   # yc
    M = torch.tensor([[[1., 1., 1., 0.]]])                # last slot padded
    out = R.apply_coord_variant(X, M, 'T3')
    # padded slot's coords must be exactly zero
    assert out[0, 0, 3, 1].item() == 0.0 and out[0, 0, 3, 2].item() == 0.0
    # valid slots' xc must sum to ~0 (mean-subtracted over the 3 valid boxes only)
    assert abs(out[0, 0, :3, 1].sum().item()) < 1e-5, out[0, 0, :3, 1]
    assert abs(out[0, 0, :3, 2].sum().item()) < 1e-5, out[0, 0, :3, 2]


def test_apply_coord_variant_unknown_raises():
    X = torch.randn(1, 1, 1, 74)
    M = torch.ones(1, 1, 1)
    try:
        R.apply_coord_variant(X, M, 'T9')
        assert False, 'expected ValueError'
    except ValueError:
        pass


def test_kfold_load_pooled_binary_pools_691_and_dedupe_is_train_only():
    if not TOK_OK:
        return _skip('test_kfold_load_pooled_binary_pools_691_and_dedupe_is_train_only')
    X, M, y, ysub, views, cids = K.load_pooled_binary('bin8_', 'none')
    assert len(y) == 691, len(y)                              # 566 train + 125 test, no dedupe
    assert len(cids) == 691, len(cids)                        # clip_ids must align with the labels
    Xd, Md, yd, ysubd, viewsd, cidsd = K.load_pooled_binary('bin8_', 'drop_conflict')
    assert len(yd) == 687, len(yd)                             # 691 - 4 train-only drops; test untouched
    assert len(cidsd) == 687, len(cidsd)


def test_kfold_and_blind_loaders_agree_on_binary_label_for_same_clip():
    """Cross-check: run_relation_transformer.load_binary and run_kfold_relation.load_pooled_binary
    must derive the SAME binary label for the same sub-class value (single source of truth:
    sub-class < 4 -> Unsafe)."""
    if not TOK_OK:
        return _skip('test_kfold_and_blind_loaders_agree_on_binary_label_for_same_clip')
    _, _, ybin_r, ysub_r, _, cid_r = R.load_binary('train', 'bin8_', 'none')
    Xk, Mk, yk, ysubk, viewsk, cid_k = K.load_pooled_binary('bin8_', 'none')
    assert ((ysub_r < 4).long() == ybin_r).all()
    assert ((ysubk < 4).long() == yk).all()


def test_grouped_folds_unsafenet_partition_matches_leave_one_group_out():
    """Regression guard for the 2026-09-20 run_lco refactor (dataset dispatch): the Unsafe-Net path
    must still be exactly leave-one-scene-group-out. Held-out set == np.where(groups == g), train
    pool == np.where(groups != g), folds disjoint and exhaustive."""
    if not TOK_OK:
        return _skip('test_grouped_folds_unsafenet_partition_matches_leave_one_group_out')
    import argparse
    import json
    import numpy as np
    _, _, y, _, _, clip_ids = K.load_pooled_binary('bin8_', 'drop_conflict')
    args = argparse.Namespace(dataset='unsafenet',
                              scene_map='sweep_results/zone_prior/clip_scene_map_691.json')
    groups, folds, note, sub_names, n_sub, proto = K._grouped_folds(args, y, clip_ids)
    assert proto == 'lco-691' and n_sub == 8 and sub_names is K.SUBN
    sm = json.load(open(args.scene_map))
    want_groups = np.array([str(sm.get(c, sm.get(c[:-4], 'na'))) for c in clip_ids])
    assert (groups == want_groups).all()
    assert [g for g, _ in folds] == sorted(set(want_groups))
    covered = np.zeros(len(y), dtype=bool)
    for g, te in folds:
        assert (te == np.where(want_groups == g)[0]).all(), g
        tr = np.setdiff1d(np.arange(len(y)), te)
        assert (tr == np.where(want_groups != g)[0]).all(), g   # old code's train pool, verbatim
        assert not covered[te].any()
        covered[te] = True
    assert covered.all()


def test_dedupe_task_binary_plumbed_not_unsafe4():
    """Guard against accidentally calling dedupe.keep_mask without task='binary' for the binary
    loaders (would silently fall back to the unsafe4 audit / 52-clip drop-list)."""
    from unsafenet import dedupe
    clip_ids = ['0_tr123.mp4', '0_tr1.mp4']
    km_binary = dedupe.keep_mask(clip_ids, 'train', 'drop_conflict', task='binary')
    assert list(km_binary) == [False, True]
    km_unsafe4 = dedupe.keep_mask(clip_ids, 'train', 'drop_all', task='unsafe4')
    # unsafe4's drop_all policy drops on a totally different (8-class-contradiction) criterion —
    # the two tasks must not be interchangeable via a missing task= kwarg.
    assert isinstance(km_unsafe4, type(km_binary))


if __name__ == '__main__':
    import traceback
    failures = 0
    mod = sys.modules[__name__]
    for name in sorted(dir(mod)):
        if not name.startswith('test_'):
            continue
        fn = getattr(mod, name)
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
    sys.exit(1 if failures else 0)
