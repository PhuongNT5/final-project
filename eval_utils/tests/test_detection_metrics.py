"""Tests for eval_utils/detection_metrics.py — the unsafe-DETECTION framing of the binary task.

Positive class is unsafe, so the quantities are asymmetric (recall, false-alarm rate, precision)
and several of them are UNDEFINED on degenerate slices: the paper's 64-clip test set is all-unsafe,
where only recall exists.  A metric that returns 0.0 where it means "undefined" would be read as a
measured value, so each degenerate case is asserted explicitly.

`pick_thresholds` may look at validation data ONLY. Re-reading one saved test posterior at many
thresholds is test probing unless every threshold is fixed beforehand on val.

Usage: PYTHONPATH=. .venv/bin/python eval_utils/tests/test_detection_metrics.py
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval_utils.detection_metrics import (counts, duplicate_floor, pick_thresholds,
                                          scene_alarm_profile)


def test_counts_matches_hand_computation_with_unsafe_as_positive():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    dec = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    c = counts(y, dec)
    assert (c['tp'], c['fn'], c['fp'], c['tn']) == (3, 1, 1, 3)
    assert abs(c['recall_unsafe'] - 0.75) < 1e-12
    assert abs(c['false_alarm_rate'] - 0.25) < 1e-12
    assert abs(c['specificity'] - 0.75) < 1e-12
    assert abs(c['precision_unsafe'] - 0.75) < 1e-12
    assert abs(c['f1_unsafe'] - 0.75) < 1e-12
    assert abs(c['youden_j'] - 0.5) < 1e-12


def test_f1_is_undefined_not_zero_when_truth_has_no_positive():
    """A metric with no positive to recall must say so; 0.0 would read as a measured value."""
    c = counts(np.array([0, 0, 0]), np.array([0, 1, 0]))
    assert np.isnan(c['recall_unsafe']) and np.isnan(c['f1_unsafe'])
    assert abs(c['false_alarm_rate'] - 1 / 3) < 1e-12


def test_f1_is_zero_when_positives_exist_but_nothing_is_flagged():
    c = counts(np.array([1, 1, 0, 0]), np.array([0, 0, 0, 0]))
    assert c['recall_unsafe'] == 0.0 and np.isnan(c['precision_unsafe'])
    assert c['f1_unsafe'] == 0.0


def test_unsafe_only_slice_leaves_the_safe_side_undefined():
    """NP-64 is all-unsafe: only recall is defined, which is why nothing else is printed."""
    c = counts(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 0]))
    assert abs(c['recall_unsafe'] - 0.75) < 1e-12
    assert np.isnan(c['false_alarm_rate']) and np.isnan(c['specificity'])
    assert np.isnan(c['youden_j'])


def _val():
    sv = np.array([0.05, 0.15, 0.25, 0.45, 0.55, 0.65, 0.85, 0.95])
    yv = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return sv, yv


def test_pick_thresholds_reads_only_the_arrays_it_is_given():
    """Signature-level guarantee: no test array can reach the choice, and the choice is stable."""
    sv, yv = _val()
    first = pick_thresholds(sv, yv, 0.25)
    assert first == pick_thresholds(sv, yv, 0.25)
    assert first['OP0_argmax'] == 0.5          # argmax is fixed a priori, not fitted at all
    assert set(first) == {'OP0_argmax', 'OP1_val_max_f1', 'OP2_val_fpr_le_10pct',
                          'OP3_val_alarm_matched_to_scene'}


def test_op2_respects_the_alarm_budget_on_val_and_takes_the_lowest_such_threshold():
    sv, yv = _val()
    taus = pick_thresholds(sv, yv, 0.25)
    t = taus['OP2_val_fpr_le_10pct']
    assert counts(yv, (sv >= t).astype(int))['false_alarm_rate'] <= 0.10
    lower = [u for u in np.unique(sv) if u < t
             and counts(yv, (sv >= u).astype(int))['false_alarm_rate'] <= 0.10]
    assert not lower                           # nothing cheaper also met the budget


def test_op3_matches_the_reference_alarm_rate():
    sv, yv = _val()
    t = pick_thresholds(sv, yv, 0.50)['OP3_val_alarm_matched_to_scene']
    assert abs(counts(yv, (sv >= t).astype(int))['false_alarm_rate'] - 0.50) < 1e-12


def _audit(path, unsafe_twin_name):
    groups = [{'kind': 'opposite',
               'members': [{'name': 'a_safe.mp4', 'binary': 'safe'},
                           {'name': unsafe_twin_name, 'binary': 'unsafe'}]},
              {'kind': 'same', 'members': [{'name': 'c_other.mp4', 'binary': 'safe'},
                                           {'name': 'd_other.mp4', 'binary': 'safe'}]}]
    json.dump({'groups': groups}, open(path, 'w'))


def test_duplicate_floor_counts_only_label_forced_false_alarms():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'audit.json')
        _audit(path, 'b_unsafe_in_train.mp4')            # twin lives outside the scored set
        names = ['a_safe.mp4', 'c_other.mp4']
        y = np.array([0, 0])
        loud = duplicate_floor(names, y, np.array([1, 1]), path)
        assert loud['n_forced_candidates'] == 1          # the 'same'-label group is not forced
        assert loud['n_actually_false_alarmed'] == 1
        assert abs(loud['label_induced_fpr_floor'] - 0.5) < 1e-12
        assert loud['perfect_detection_possible']        # no opposite pair wholly inside the set
        quiet = duplicate_floor(names, y, np.array([0, 1]), path)
        assert quiet['n_actually_false_alarmed'] == 0 and quiet['label_induced_fpr_floor'] == 0.0


def test_duplicate_floor_rules_out_perfection_when_both_twins_are_scored():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'audit.json')
        _audit(path, 'b_unsafe.mp4')                     # both members inside the scored set
        out = duplicate_floor(['a_safe.mp4', 'b_unsafe.mp4'], np.array([0, 1]),
                              np.array([0, 1]), path)
        assert out['detail'][0]['twin_in_test']
        assert not out['perfect_detection_possible']


def test_scene_alarm_profile_reports_an_undefined_correlation_instead_of_inventing_one():
    y = np.array([1, 0, 1, 0])
    dec = np.array([1, 1, 1, 1])                 # constant alarm rate -> correlation undefined
    sid = np.array([0, 0, 1, 1])
    out = scene_alarm_profile(y, dec, sid)
    assert np.isnan(out['pearson_alarm_vs_true_rate'])
    assert out['within_scene_youden_j'] == {'0': 0.0, '1': 0.0}


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
