"""Unit tests for binary_gate_lib.py — the S1/S2 Tier-1 gate helpers (phase-06 binary reframe).

Covers: (1) the S1 8-dim "max detector confidence per entity class" descriptor construction,
(2) the binary label mapping (source class 0-3 unsafe / 4-7 safe), (3) NO LEAKAGE across
train/held-out in both kfold_oof and lco_oof — the single most load-bearing correctness property
for the LCO gate, since clip_scene_map_691.json groups are severely class-skewed and any leak
would silently inflate the S1/S2 numbers this report's gate verdict depends on.

No pytest in the project venv (tests/test_dedupe_binary.py precedent) — plain assert functions
named test_*, runnable standalone.

Usage: .venv/bin/python tests/test_binary_baseline_gates.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import binary_gate_lib as G
from unsafenet.metrics import balanced_accuracy as _balanced_accuracy


def test_to_binary_maps_source_classes_correctly():
    sub = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y = G.to_binary(sub)
    assert list(y) == [1, 1, 1, 1, 0, 0, 0, 0], y


def test_s1_descriptor_picks_max_confidence_per_class_over_masked_valid_boxes():
    # 1 clip, T=2 frames, N_MAX=2 boxes, feat width 6 (class,xc,yc,w,h,conf) is enough to exercise
    # the [...,0]=class / [...,5]=conf indexing the real 74-wide bin8 tokens also use.
    tok = np.zeros((1, 2, 2, 6), dtype=np.float32)
    mask = np.zeros((1, 2, 2), dtype=np.float32)
    # frame0 box0: class 1, conf 0.3 (valid)
    tok[0, 0, 0] = [1, 0, 0, 0, 0, 0.3]; mask[0, 0, 0] = 1
    # frame0 box1: class 1, conf 0.9 (valid) -> class 1 max should be 0.9, not the padded box below
    tok[0, 0, 1] = [1, 0, 0, 0, 0, 0.9]; mask[0, 0, 1] = 1
    # frame1 box0: class 5, conf 0.99 but INVALID (mask=0) -> must be ignored
    tok[0, 1, 0] = [5, 0, 0, 0, 0, 0.99]; mask[0, 1, 0] = 0
    desc = G.s1_descriptor(tok, mask)
    assert desc.shape == (1, 8)
    assert abs(desc[0, 1] - 0.9) < 1e-6, desc[0, 1]          # class 1 -> max of valid (0.3, 0.9)
    assert desc[0, 5] == 0.0, desc[0, 5]                     # class 5 detection was masked out -> 0
    for k in (0, 2, 3, 4, 6, 7):
        assert desc[0, k] == 0.0                             # never detected -> 0


def test_s1_descriptor_empty_clip_is_all_zero():
    tok = np.zeros((3, 16, 8, 74), dtype=np.float32)
    mask = np.zeros((3, 16, 8), dtype=np.float32)             # no detections at all
    desc = G.s1_descriptor(tok, mask)
    assert desc.shape == (3, 8)
    assert (desc == 0.0).all()


class _SpyModel:
    """Records (train_values, test_values) seen by every _fit_predict_proba call so the test can
    assert the two sets never intersect — i.e. no row used for held-out prediction was also used
    to fit the model. X is a single column of UNIQUE per-row IDs so any overlap is detectable."""
    def __init__(self):
        self.calls = []

    def __call__(self, model_name, Xtr, ytr, Xte):
        self.calls.append((set(Xtr[:, 0].tolist()), set(Xte[:, 0].tolist())))
        # cheap deterministic "prediction": doesn't need to be a real classifier for a leakage test
        p = np.zeros((len(Xte), 2)); p[:, 0] = 1.0
        return p


def test_kfold_oof_never_fits_on_a_held_out_row():
    n = 60
    X = np.arange(n, dtype=np.float64).reshape(-1, 1)          # unique row IDs
    y = np.array([0, 1] * (n // 2))
    spy = _SpyModel()
    orig = G._fit_predict_proba
    G._fit_predict_proba = spy
    try:
        G.kfold_oof(X, y, 's1_logreg', k=5, repeats=3)
    finally:
        G._fit_predict_proba = orig
    assert len(spy.calls) == 5 * 3
    for tr_ids, te_ids in spy.calls:
        assert tr_ids.isdisjoint(te_ids), 'a fold fit on a row it was asked to predict'


def test_lco_oof_never_fits_on_the_held_out_scene_cluster():
    n = 40
    X = np.arange(n, dtype=np.float64).reshape(-1, 1)
    y = np.array(([0, 1] * (n // 2)))
    group = np.array([str(i % 4) for i in range(n)])            # 4 synthetic scene clusters
    spy = _SpyModel()
    orig = G._fit_predict_proba
    G._fit_predict_proba = spy
    try:
        _, uniq = G.lco_oof(X, y, group, 's1_logreg')
    finally:
        G._fit_predict_proba = orig
    assert uniq == ['0', '1', '2', '3']
    assert len(spy.calls) == 4
    for tr_ids, te_ids in spy.calls:
        assert tr_ids.isdisjoint(te_ids), 'a held-out scene cluster leaked into its own training fold'
    # every row appears as a held-out id in EXACTLY one call (each clip belongs to one cluster)
    all_test_ids = [i for _, te in spy.calls for i in te]
    assert sorted(all_test_ids) == sorted(X[:, 0].tolist())


def test_lco_oof_degenerate_single_class_training_fold_falls_back_to_train_prior_not_a_crash():
    # group 'deg' is the only one holding class 1; every OTHER group's training pool for the
    # held-out group's complement still has both classes UNLESS we engineer it away for this
    # specific held-out group -> construct a group whose complement (all other rows) is single-class.
    n = 6
    X = np.arange(n, dtype=np.float64).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 0, 1])                            # only row 5 is class 1
    group = np.array(['a', 'a', 'a', 'a', 'a', 'b'])             # holding out 'b' leaves an all-0 train
    prob, uniq = G.lco_oof(X, y, group, 's1_logreg')
    assert uniq == ['a', 'b']
    assert prob.shape == (6, 2)
    # held-out group 'b' (row 5): train pool (group 'a') is single-class -> prior fallback, not a
    # sklearn crash; prediction must be a valid probability row.
    assert abs(prob[5].sum() - 1.0) < 1e-6


def test_saved_oof_reproduces_balanced_accuracy():
    # real (non-spy) kfold_oof run on a small separable synthetic problem -> compute bal_acc from
    # the in-memory prob/pred, save_oof to a scratch dir, reload the JSON, recompute bal_acc from
    # the saved y_true/y_pred, and assert they match exactly (byte-identical predictions).
    rng = np.random.default_rng(0)
    n = 80
    y = np.array([0, 1] * (n // 2))
    X = (y[:, None] * 3.0 + rng.normal(size=(n, 4)))              # separable-ish, real logreg fit
    clip_ids = np.array([f'clip{i}' for i in range(n)])

    prob, facc, ntr, nte, fold_id = G.kfold_oof(X, y, 's1_logreg', k=5, repeats=1)
    pred = prob.argmax(1)
    bal_acc_direct = _balanced_accuracy(y, pred, 2)

    scratch = tempfile.mkdtemp()
    orig_dir = G.OOF_DIR
    G.OOF_DIR = scratch
    try:
        path = G.save_oof('s1_logreg', 'kfold-691', 'none', clip_ids, y, prob, pred, fold_id)
        saved = json.load(open(path))
    finally:
        G.OOF_DIR = orig_dir
        shutil.rmtree(scratch, ignore_errors=True)

    assert saved['n'] == n
    y_true_saved = np.array([r['y_true'] for r in saved['rows']])
    y_pred_saved = np.array([r['y_pred'] for r in saved['rows']])
    assert np.array_equal(y_true_saved, y)
    assert np.array_equal(y_pred_saved, pred)
    bal_acc_reloaded = _balanced_accuracy(y_true_saved, y_pred_saved, 2)
    assert abs(bal_acc_reloaded - bal_acc_direct) < 1e-12, (bal_acc_reloaded, bal_acc_direct)
    # prob_unsafe threshold consistency: y_pred must equal (prob_unsafe > 0.5), matching argmax
    # tie-breaking (continuous logreg scores essentially never land exactly on 0.5).
    prob_unsafe_saved = np.array([r['prob_unsafe'] for r in saved['rows']])
    assert np.array_equal((prob_unsafe_saved > 0.5).astype(int), y_pred_saved)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'  PASS  {fn.__name__}')
    print('ALL PASS')
