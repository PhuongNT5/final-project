"""Drop-list for byte-identical clips filed under contradictory labels.

`run_duplicate_audit.py` finds 32 groups of md5-identical videos carrying >=2 different class
labels (26 of them WalkwayViolation<->UnauthorizedIntervention). This module turns that audit into
a reusable keep-mask. It NEVER deletes footage - callers filter at load time, so every experiment
can be re-run on the raw set by passing policy='none'.

Train policy
  'drop_all'  (default) - drop every TRAIN member of a contradictory group. We cannot tell which
                label of a pair is correct, so keeping either injects a suspect label and keeping
                both injects a contradiction. Also removes the train twin of contaminated test clips.
  'keep_one'  - keep the lexicographically first member per group (ablation; retains a known-suspect
                label on ~half the pairs).
  'none'      - raw set.

Test policy
  'keep'      (default) - the 64-clip benchmark is frozen, for comparability with the paper.
                NOTE: 0_te8.mp4 and 1_te1.mp4 are byte-identical, both in the 64, with different
                labels -> any deterministic classifier is wrong on exactly one. Ceiling = 63/64.
  'drop_contradictory' - also drop both members of a test-test contradictory group, restoring a
                100%-attainable test set (62 clips). Report as a secondary number, never silently.

Usage:
    from unsafenet import dedupe
    keep = dedupe.keep_mask(clip_ids, split='train')     # bool array aligned to clip_ids
"""
import json
import os
import numpy as np

AUDIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'sweep_results', 'duplicate_audit.json')


def _groups(audit=AUDIT):
    if not os.path.exists(audit):
        raise FileNotFoundError(f'{audit} missing - run: .venv/bin/python run_duplicate_audit.py')
    return json.load(open(audit))['groups']


def drop_names(split, policy=None, audit=AUDIT):
    """Set of clip file names to exclude from `split`."""
    policy = policy or ('drop_all' if split == 'train' else 'keep')
    if policy == 'none':
        return set()
    drop = set()
    for g in _groups(audit):
        mem = sorted([r for r in g if r['split'] == split], key=lambda r: r['name'])
        if not mem:
            continue
        if split == 'train':
            drop.update(r['name'] for r in (mem[1:] if policy == 'keep_one' else mem))
        elif policy == 'drop_contradictory' and len(mem) > 1:
            # a contradictory pair wholly inside the test split - unattainable by construction
            drop.update(r['name'] for r in mem)
    return drop


def keep_mask(clip_ids, split, policy=None, audit=AUDIT):
    drop = drop_names(split, policy, audit)
    return np.array([str(c) not in drop for c in clip_ids], dtype=bool)


def summary(audit=AUDIT):
    out = {}
    for sp in ('train', 'test'):
        for pol in (('drop_all', 'keep_one', 'none') if sp == 'train' else ('keep', 'drop_contradictory')):
            out[f'{sp}/{pol}'] = sorted(drop_names(sp, pol, audit))
    return out


if __name__ == '__main__':
    for k, v in summary().items():
        print(f'{k:28s} drops {len(v):3d}  {v[:6]}{" ..." if len(v) > 6 else ""}')
