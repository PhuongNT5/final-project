"""G12: prove `--split random` really leaks source identity and `--split group` really does not.

The whole protocol-inflation claim rests on the two fold constructions differing in exactly one
way, so that difference is asserted here rather than trusted to a flag name.
"""
import sys
import types
import numpy as np
import torch

sys.path.insert(0, '.')
import isafetybench_dataset as ISB
import run_kfold_relation as RK


def _args(split):
    return types.SimpleNamespace(dataset='isafetybench', split=split, folds=10, tag='isb_')


def _load():
    X, M, y, groups, clip_ids = ISB.load('isb_')
    return y, clip_ids, groups


def _held_out_index(folds, n):
    seen = np.zeros(n, dtype=int)
    for _, te in folds:
        seen[te] += 1
    return seen


def main():
    y, clip_ids, groups = _load()
    n = len(y)
    fails = []

    for split in ('group', 'random'):
        _, folds, note, sub_names, n_sub, proto = RK._grouped_folds(_args(split), y, clip_ids, groups)
        seen = _held_out_index(folds, n)
        if not (seen == 1).all():
            fails.append(f'{split}: every clip must be held out exactly once, got {sorted(set(seen))}')
        if len(folds) != 10:
            fails.append(f'{split}: expected 10 folds, got {len(folds)}')
        if n_sub != 2:
            fails.append(f'{split}: binary corpus must report n_sub=2, got {n_sub}')
        if sub_names is not None:
            fails.append(f'{split}: iSafetyBench has no sub-class taxonomy, got {sub_names}')

        # the property that defines the protocol: does any source span the boundary?
        spanning = 0
        for _, te in folds:
            held = set(groups[te])
            trained = set(groups[np.setdiff1d(np.arange(n), te)])
            spanning += len(held & trained)

        if split == 'group':
            if spanning != 0:
                fails.append(f'group: {spanning} source/fold pairs span the boundary -- NOT leave-source-out')
            if proto != f'lso-{n}':
                fails.append(f'group: proto tag must be lso-{n}, got {proto}')
            if 'StratifiedGroupKFold' not in note:
                fails.append('group: note must name the fold constructor')
        else:
            if spanning == 0:
                fails.append('random: 0 sources span the boundary -- the leak this run exists to '
                             'measure did not happen, so the G12 comparison would be meaningless')
            if proto != f'rand-{n}':
                fails.append(f'random: proto tag must be rand-{n}, got {proto}')
            for must in ('DIAGNOSTIC', 'SOURCE-LEAKING', 'NOT a generalisation result'):
                if must not in note:
                    fails.append(f'random: note must carry {must!r} so the number cannot be quoted bare')
            print(f'  [random] {spanning} source/fold pairs span the train/held-out boundary')

    # the two protocols must not produce the same partition, or there is nothing to compare
    _, fg, *_ = RK._grouped_folds(_args('group'), y, clip_ids, groups)
    _, fr, *_ = RK._grouped_folds(_args('random'), y, clip_ids, groups)
    if all(np.array_equal(np.sort(a[1]), np.sort(b[1])) for a, b in zip(fg, fr)):
        fails.append('group and random produced identical partitions')

    # determinism: seed 0 both sides, a re-run must reproduce the folds
    _, fr2, *_ = RK._grouped_folds(_args('random'), y, clip_ids, groups)
    if not all(np.array_equal(a[1], b[1]) for a, b in zip(fr, fr2)):
        fails.append('random split is not reproducible at seed 0')

    # The R42 floor must be scored on the ENCODER's folds. run_kfold_relation used to rebuild them
    # with ISB.folds(), which ignores --split: the encoder then ran on random folds while the
    # shortcut ran on leave-source-out folds, and the log still said "identical folds".
    all_idx = np.arange(n)
    for split in ('group', 'random'):
        _, folds, *_ = RK._grouped_folds(_args(split), y, clip_ids, groups)
        derived = [(np.setdiff1d(all_idx, te), te) for _, te in folds]
        for tr, te in derived:
            if len(tr) + len(te) != n or len(np.intersect1d(tr, te)):
                fails.append(f'{split}: derived floor split is not a partition of the corpus')
        rebuilt = ISB.folds(y, groups, n_splits=10, seed=0)
        same = all(np.array_equal(np.sort(a[1]), np.sort(b[1])) for a, b in zip(derived, rebuilt))
        if split == 'group' and not same:
            fails.append('group: derived floor splits must equal ISB.folds()')
        if split == 'random' and same:
            fails.append('random: derived floor splits equal the leave-source-out folds -- the floor '
                         'would be scored on a DIFFERENT protocol than the encoder (the bug this '
                         'test exists to catch)')

    if fails:
        print('FAIL')
        for f in fails:
            print('  -', f)
        return 1
    print(f'PASS  {n} clips / {len(set(groups))} sources: group leaks 0 sources, random leaks many, '
          f'both partition the corpus exactly once, both reproducible')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
