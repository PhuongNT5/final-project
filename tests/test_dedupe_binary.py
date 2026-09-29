"""Unit tests for unsafenet/dedupe.py — binary Safe/Unsafe drop-list + keep-mask, plus a
backward-compatibility guard that the original task='unsafe4' behaviour is untouched.

plans/2026-09-06-thesis-2week-final/reports/laneb-2026-09-16-binary-dedupe-and-ceiling.md

No pytest in the project venv (see tests/test_paired_significance.py precedent) — plain assert
functions named test_*, runnable standalone via `if __name__ == '__main__'` below.

These tests read the real audit files (sweep_results/duplicate_audit.json,
sweep_results/duplicate_audit_binary.json) rather than mocking them, so a passing run is also a
regression check that the checked-in audits still match the documented group counts.

Usage: .venv/bin/python tests/test_dedupe_binary.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsafenet import dedupe

# The one duplicate group entirely inside TEST with opposite binary labels — the hard-ceiling
# driver (124/125). Both classes are 0-prefixed -> unsafe / 4-prefixed -> safe.
BOTH_TEST_OPPOSITE = ('0_te23.mp4', '4_te9.mp4')

# The one 3-member group: two TRAIN members (0_tr123 unsafe, 1_tr55 unsafe) + one TEST member
# (4_te4, safe) — opposite binary label across the group.
THREE_MEMBER_TRAIN = ('0_tr123.mp4', '1_tr55.mp4')
THREE_MEMBER_TEST = '4_te4.mp4'

# The other two 2-member train/test opposite groups.
OTHER_OPPOSITE_TRAIN = ('0_tr136.mp4', '0_tr138.mp4')
OTHER_OPPOSITE_TEST = ('5_te13.mp4', '4_te7.mp4')

ALL_OPPOSITE_TRAIN = set(THREE_MEMBER_TRAIN) | set(OTHER_OPPOSITE_TRAIN)
ALL_OPPOSITE_TEST = {BOTH_TEST_OPPOSITE[0], BOTH_TEST_OPPOSITE[1], THREE_MEMBER_TEST} | set(OTHER_OPPOSITE_TEST)


def test_binary_default_policy_is_none_and_drops_nothing():
    """Project principle: default policy stays 'none' so historical commands don't silently
    change. Must hold for BOTH splits under task='binary', unlike the unsafe4 task whose train
    default is already 'drop_all'."""
    assert dedupe.drop_names('train', task='binary') == set()
    assert dedupe.drop_names('test', task='binary') == set()
    assert dedupe.drop_names('train', policy='none', task='binary') == set()
    assert dedupe.drop_names('test', policy='keep', task='binary') == set()


def test_binary_train_drop_conflict_is_exactly_the_four_opposite_train_clips():
    got = dedupe.drop_names('train', 'drop_conflict', task='binary')
    assert got == ALL_OPPOSITE_TRAIN, got
    assert len(got) == 4, got


def test_binary_three_member_group_drops_both_train_members_not_the_test_one():
    """0_tr123/1_tr55/4_te4: dropping train must remove BOTH train members of the 3-member group,
    and must NOT touch the test member (4_te4) even though it belongs to the same group."""
    train_drop = dedupe.drop_names('train', 'drop_conflict', task='binary')
    for name in THREE_MEMBER_TRAIN:
        assert name in train_drop, (name, train_drop)
    assert THREE_MEMBER_TEST not in train_drop

    test_keep = dedupe.drop_names('test', 'keep', task='binary')
    assert THREE_MEMBER_TEST not in test_keep  # frozen by default
    test_drop = dedupe.drop_names('test', 'drop_conflict', task='binary')
    assert THREE_MEMBER_TEST in test_drop, test_drop  # only removed under the stricter policy


def test_binary_both_in_test_group_untouched_by_train_policy():
    """0_te23/4_te9 has NO train member at all, so no train-only policy can ever remove it —
    it is the irreducible source of the 124/125 hard ceiling under every train policy."""
    train_drop_none = dedupe.drop_names('train', 'none', task='binary')
    train_drop_conflict = dedupe.drop_names('train', 'drop_conflict', task='binary')
    for name in BOTH_TEST_OPPOSITE:
        assert name not in train_drop_none
        assert name not in train_drop_conflict

    test_keep = dedupe.drop_names('test', 'keep', task='binary')
    for name in BOTH_TEST_OPPOSITE:
        assert name not in test_keep

    test_drop_conflict = dedupe.drop_names('test', 'drop_conflict', task='binary')
    for name in BOTH_TEST_OPPOSITE:
        assert name in test_drop_conflict, test_drop_conflict


def test_binary_test_drop_conflict_is_exactly_five_clips():
    got = dedupe.drop_names('test', 'drop_conflict', task='binary')
    assert got == ALL_OPPOSITE_TEST, got
    assert len(got) == 5, got


def test_binary_leakage_and_benign_groups_never_dropped_by_any_policy():
    """The 6 cross-split same-label ('leakage') and 22 same-split same-label ('benign') groups
    must never appear in a binary drop-list — only the 4 'opposite' groups are eligible."""
    leakage_train_names = {'0_tr168.mp4', '0_tr169.mp4', '1_tr24.mp4', '1_tr88.mp4', '1_tr89.mp4',
                            '4_tr10.mp4'}
    leakage_test_names = {'1_te4.mp4', '1_te5.mp4', '0_te7.mp4', '0_te10.mp4', '0_te11.mp4',
                           '6_te11.mp4'}
    train_drop = dedupe.drop_names('train', 'drop_conflict', task='binary')
    test_drop = dedupe.drop_names('test', 'drop_conflict', task='binary')
    assert train_drop.isdisjoint(leakage_train_names), train_drop & leakage_train_names
    assert test_drop.isdisjoint(leakage_test_names), test_drop & leakage_test_names
    # a same-split benign duplicate pair (both train, both unsafe: 0_tr109 <-> 1_tr73)
    assert '0_tr109.mp4' not in train_drop
    assert '1_tr73.mp4' not in train_drop


def test_binary_keep_mask_aligns_with_clip_id_order_and_drops_only_conflicts():
    clip_ids = ['0_tr123.mp4', '0_tr1.mp4', '1_tr55.mp4', '2_tr1.mp4', '0_tr136.mp4']
    mask = dedupe.keep_mask(clip_ids, 'train', 'drop_conflict', task='binary')
    assert mask.dtype == bool
    assert list(mask) == [False, True, False, True, False], mask
    # 'none' keeps everything, same order
    mask_none = dedupe.keep_mask(clip_ids, 'train', 'none', task='binary')
    assert list(mask_none) == [True, True, True, True, True]


def test_binary_invalid_policy_raises():
    try:
        dedupe.drop_names('train', 'drop_all', task='binary')  # 'drop_all' is an unsafe4 policy
        assert False, 'expected ValueError for an unsafe4 policy under task=binary'
    except ValueError:
        pass
    try:
        dedupe.drop_names('train', 'x', task='nonsense')
        assert False, 'expected ValueError for unknown task'
    except ValueError:
        pass


def test_unsafe4_task_unchanged_backward_compat():
    """Regression guard: task='unsafe4' (and calling with no `task=` at all, as every existing
    call site does) must reproduce the pre-existing counts documented in
    plans/2026-09-05-thesis-final-push/reports/dedupe-procedure.md ("drop_all -> 52 | keep_one ->
    22 | test -> 0")."""
    assert len(dedupe.drop_names('train', 'drop_all')) == 52
    assert len(dedupe.drop_names('train', 'keep_one')) == 22
    assert len(dedupe.drop_names('train', 'none')) == 0
    assert len(dedupe.drop_names('test')) == 0  # default policy 'keep'
    assert len(dedupe.drop_names('test', 'drop_contradictory')) == 4
    # same results explicitly passing task='unsafe4'
    assert dedupe.drop_names('train', 'drop_all') == dedupe.drop_names('train', 'drop_all', task='unsafe4')


def test_audit_binary_group_taxonomy_matches_documented_counts():
    """Regression guard on the checked-in sweep_results/duplicate_audit_binary.json: 32 groups
    total, 22 benign / 6 leakage / 4 opposite, matching the report's group taxonomy table."""
    import json
    audit_path = dedupe.AUDIT_BINARY
    data = json.load(open(audit_path))
    assert data['n_groups'] == 32, data['n_groups']
    kinds = [g['kind'] for g in data['groups']]
    assert kinds.count('benign') == 22, kinds.count('benign')
    assert kinds.count('leakage') == 6, kinds.count('leakage')
    assert kinds.count('opposite') == 4, kinds.count('opposite')


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
