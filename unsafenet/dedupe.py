"""Drop-list for byte-identical clips filed under contradictory labels.

`run_duplicate_audit.py` finds 32 groups of md5-identical videos carrying >=2 different class
labels (26 of them WalkwayViolation<->UnauthorizedIntervention). This module turns that audit into
a reusable keep-mask. It NEVER deletes footage - callers filter at load time, so every experiment
can be re-run on the raw set by passing policy='none'.

Two task framings share this module, selected by `task=`:

`task='unsafe4'` (default) - the ORIGINAL 8-class-label framing used by the 4-class unsafe
    experiments. Unchanged since 2026-09-06; every existing call site (which never passes
    `task=`) keeps its old behaviour exactly.

    Train policy
      'drop_all'  (default) - drop every TRAIN member of a contradictory group. We cannot tell
                    which label of a pair is correct, so keeping either injects a suspect label
                    and keeping both injects a contradiction. Also removes the train twin of
                    contaminated test clips.
      'keep_one'  - keep the lexicographically first member per group (ablation; retains a
                    known-suspect label on ~half the pairs).
      'none'      - raw set.

    Test policy
      'keep'      (default) - the 64-clip benchmark is frozen, for comparability with the paper.
                    NOTE: 0_te8.mp4 and 1_te1.mp4 are byte-identical, both in the 64, with
                    different 8-class labels -> any deterministic classifier is wrong on exactly
                    one. Ceiling = 63/64.
      'drop_contradictory' - also drop both members of a test-test contradictory group,
                    restoring a 100%-attainable test set (62 clips). Report as a secondary
                    number, never silently.

`task='binary'` - the Safe/Unsafe framing (unsafenet.config.UNSAFE = classes 0-3 "unsafe",
    4-7 "safe"). Re-auditing the SAME 32 groups under binary labels
    (`run_duplicate_audit_binary.py` -> `sweep_results/duplicate_audit_binary.json`) shows most
    of the old "contradictions" are not contradictions here: 22 groups already share split AND
    binary label (harmless duplicate copy), 6 more cross train/test but still share the binary
    label (byte-level train/test leakage, no label conflict), and only 4 groups actually disagree
    on the binary label - including 1 group entirely inside test (0_te23.mp4 vs 4_te9.mp4), which
    alone sets a hard ceiling of 124/125 regardless of any train policy. See
    `plans/2026-09-06-thesis-2week-final/reports/laneb-2026-09-16-binary-dedupe-and-ceiling.md`
    for the full derivation and the recommended-policy reasoning.

    Per the project's dedupe principle, the default policy is 'none' for BOTH splits, so any
    historical command that does not pass `--dedupe` reproduces its old (contaminated) number.

    Train policy
      'none'          (default) - raw set (566 clips: 452 unsafe / 114 safe).
      'drop_conflict' - drop the TRAIN members of the 4 binary-opposite groups only (4 clips:
                        0_tr123, 0_tr136, 0_tr138, 1_tr55 - all 4 happen to be UNSAFE-class
                        members, so this policy does not worsen the train class imbalance; it
                        marginally improves it, 452/114 -> 448/114). The 6 leakage groups (same
                        binary label, cross-split) are deliberately left untouched: dropping them
                        would cost train clips for zero label-conflict benefit under this task.

    Test policy
      'keep'          (default) - the 125-clip test set (64 unsafe / 61 safe) is frozen, for
                        comparability across every run.
      'drop_conflict' - ALSO drop the 5 TEST members of the 4 binary-opposite groups (0_te23,
                        4_te4, 4_te7, 4_te9, 5_te13), restoring a 120-clip, 100%-attainable test
                        set (63 unsafe / 57 safe). Report as a sensitivity analysis only, never
                        the headline - it breaks comparability with the frozen 125-clip set and
                        costs 4 of the already-smaller safe class.

Usage:
    from unsafenet import dedupe
    dedupe.drop_names('train', 'drop_all')                    # -> set of 52 names (unsafe4)
    dedupe.keep_mask(clip_ids, 'train', 'drop_conflict', task='binary')  # bool array, binary task
"""
import json
import os
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(_ROOT, 'sweep_results', 'duplicate_audit.json')
AUDIT_BINARY = os.path.join(_ROOT, 'sweep_results', 'duplicate_audit_binary.json')

_BINARY_TRAIN_POLICIES = ('none', 'drop_conflict')
_BINARY_TEST_POLICIES = ('keep', 'drop_conflict')
_BINARY_NOOP_POLICIES = ('none', 'keep')  # synonyms: 'none' is train's no-op name, 'keep' is test's


def _groups(audit=AUDIT):
    if not os.path.exists(audit):
        raise FileNotFoundError(f'{audit} missing - run: .venv/bin/python run_duplicate_audit.py')
    return json.load(open(audit))['groups']


def _binary_groups(audit=AUDIT_BINARY):
    if not os.path.exists(audit):
        raise FileNotFoundError(f'{audit} missing - run: '
                                 '.venv/bin/python run_duplicate_audit_binary.py')
    return json.load(open(audit))['groups']


def _drop_names_unsafe4(split, policy, audit):
    policy = policy or ('drop_all' if split == 'train' else 'keep')
    if policy == 'none':
        return set()
    drop = set()
    for g in _groups(audit or AUDIT):
        mem = sorted([r for r in g if r['split'] == split], key=lambda r: r['name'])
        if not mem:
            continue
        if split == 'train':
            drop.update(r['name'] for r in (mem[1:] if policy == 'keep_one' else mem))
        elif policy == 'drop_contradictory' and len(mem) > 1:
            # a contradictory pair wholly inside the test split - unattainable by construction
            drop.update(r['name'] for r in mem)
    return drop


def _drop_names_binary(split, policy, audit):
    policy = policy or 'none'
    valid = _BINARY_TRAIN_POLICIES if split == 'train' else _BINARY_TEST_POLICIES
    # 'none' (train's no-op name) and 'keep' (test's no-op name) are accepted as synonyms on
    # either split so `policy=None` always resolves to a harmless no-op regardless of split.
    if policy in _BINARY_NOOP_POLICIES:
        return set()
    if policy not in valid:
        raise ValueError(f"binary task: unknown {split} policy {policy!r}, expected one of {valid}")
    # policy == 'drop_conflict' for either split: drop this split's members of every group whose
    # members disagree on the binary label (the 4 "opposite" groups; leakage/benign untouched).
    drop = set()
    for g in _binary_groups(audit or AUDIT_BINARY):
        if g['kind'] != 'opposite':
            continue
        drop.update(m['name'] for m in g['members'] if m['split'] == split)
    return drop


def drop_names(split, policy=None, task='unsafe4', audit=None):
    """Set of clip file names to exclude from `split`. See module docstring for policies."""
    if task == 'binary':
        return _drop_names_binary(split, policy, audit)
    if task == 'unsafe4':
        return _drop_names_unsafe4(split, policy, audit)
    raise ValueError(f"unknown task {task!r}, expected 'unsafe4' or 'binary'")


def keep_mask(clip_ids, split, policy=None, task='unsafe4', audit=None):
    drop = drop_names(split, policy, task, audit)
    return np.array([str(c) not in drop for c in clip_ids], dtype=bool)


def summary(audit=None, task='unsafe4'):
    out = {}
    if task == 'binary':
        for sp, pols in (('train', _BINARY_TRAIN_POLICIES), ('test', _BINARY_TEST_POLICIES)):
            for pol in pols:
                out[f'{sp}/{pol}'] = sorted(drop_names(sp, pol, task='binary', audit=audit))
        return out
    for sp in ('train', 'test'):
        for pol in (('drop_all', 'keep_one', 'none') if sp == 'train' else ('keep', 'drop_contradictory')):
            out[f'{sp}/{pol}'] = sorted(drop_names(sp, pol, task='unsafe4', audit=audit))
    return out


if __name__ == '__main__':
    print('-- task=unsafe4 (original 8-class-label framing) --')
    for k, v in summary(task='unsafe4').items():
        print(f'{k:28s} drops {len(v):3d}  {v[:6]}{" ..." if len(v) > 6 else ""}')
    print('-- task=binary (Safe/Unsafe framing) --')
    for k, v in summary(task='binary').items():
        print(f'{k:28s} drops {len(v):3d}  {v[:6]}{" ..." if len(v) > 6 else ""}')
