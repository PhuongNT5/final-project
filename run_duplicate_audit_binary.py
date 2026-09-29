#!/usr/bin/env python3
"""Binary Safe/Unsafe duplicate-clip audit over all 691 clips.

The original `run_duplicate_audit.py` (-> `sweep_results/duplicate_audit.json`) classifies
byte-identical groups as "conflicting" whenever their 8-class folder labels differ. That is
the right question for the 4-class unsafe task, but it over-counts for the BINARY Safe/Unsafe
task: classes 0-3 are all "unsafe" and 4-7 are all "safe" (see `unsafenet/config.py`
`UNSAFE`/`SAFE_OF`), so e.g. a 0-vs-1 group (WalkwayViol vs UnauthorInt) carries the SAME
binary label on both copies and is not a contradiction under this task.

This script reuses `run_duplicate_audit.scan()` (identical hashing logic, so the group
membership is guaranteed to match the original audit) and re-classifies each group under
binary semantics:

  benign    - same split, same binary label -> harmless duplicate, no action needed
  leakage   - crosses train/test, same binary label -> no label conflict, but the exact
              bytes of a test clip were also placed in train (train/test contamination
              independent of labelling)
  opposite  - members disagree on binary label -> genuine label contradiction under the
              binary task; the classifier cannot be correct on every member

Writes `sweep_results/duplicate_audit_binary.json`. Does NOT touch `duplicate_audit.json`,
which backs the published 4-class numbers, and does NOT delete or move any clip.

Usage: .venv/bin/python run_duplicate_audit_binary.py
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_duplicate_audit import scan

UNSAFE_PREFIXES = {'0', '1', '2', '3'}  # matches unsafenet.config.UNSAFE = [0, 1, 2, 3]


def binary_label(cls):
    return 'unsafe' if cls.split('_')[0] in UNSAFE_PREFIXES else 'safe'


def audit(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r['md5']].append(r)
    groups = [v for v in by.values() if len(v) > 1]

    out_groups = []
    counts = collections.Counter()
    for g in groups:
        members = [{
            'split': r['split'],
            'cls': r['cls'],
            'name': r['name'],
            'path': os.path.join('data', r['split'], r['cls'], r['name']),
            'binary': binary_label(r['cls']),
        } for r in g]
        splits = {m['split'] for m in members}
        labels = {m['binary'] for m in members}
        if len(labels) > 1:
            kind = 'opposite'
        elif len(splits) > 1:
            kind = 'leakage'
        else:
            kind = 'benign'
        counts[kind] += 1
        out_groups.append({'md5': g[0]['md5'], 'kind': kind, 'members': members})

    # groups entirely inside test with opposite binary labels -> hard ceiling driver
    hard_ceiling_groups = [g for g in out_groups
                            if g['kind'] == 'opposite' and {m['split'] for m in g['members']} == {'test'}]
    opposite_groups = [g for g in out_groups if g['kind'] == 'opposite']
    opposite_train_names = sorted({m['name'] for g in opposite_groups for m in g['members']
                                    if m['split'] == 'train'})
    opposite_test_names = sorted({m['name'] for g in opposite_groups for m in g['members']
                                   if m['split'] == 'test'})

    return {
        'n_files': len(rows),
        'n_distinct': len(by),
        'n_groups': len(groups),
        'n_benign': counts['benign'],
        'n_leakage': counts['leakage'],
        'n_opposite': counts['opposite'],
        'n_hard_ceiling_groups': len(hard_ceiling_groups),
        'opposite_train_clip_names': opposite_train_names,
        'opposite_test_clip_names': opposite_test_names,
        'groups': out_groups,
    }


if __name__ == '__main__':
    rows = scan('data')
    r = audit(rows)
    os.makedirs('sweep_results', exist_ok=True)
    out = 'sweep_results/duplicate_audit_binary.json'
    json.dump(r, open(out, 'w'), indent=2)
    print(f"{r['n_files']} clips, {r['n_distinct']} distinct -> {r['n_groups']} duplicate groups")
    print(f"  binary-benign  (same split, same label)   : {r['n_benign']}")
    print(f"  binary-leakage (cross split, same label)  : {r['n_leakage']}")
    print(f"  binary-opposite (label contradiction)     : {r['n_opposite']}"
          f"  ({r['n_hard_ceiling_groups']} entirely in test)")
    print(f"  opposite-group train clips to drop        : {len(r['opposite_train_clip_names'])}"
          f" {r['opposite_train_clip_names']}")
    print(f"  opposite-group test clips (sensitivity)   : {len(r['opposite_test_clip_names'])}"
          f" {r['opposite_test_clip_names']}")
    print(f"-> written {out}")
