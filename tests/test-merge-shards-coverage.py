"""Guards on tools/merge-esta-shards.py.

Sharding is only safe if a partial merge is impossible. These tests assert the merger refuses the
three ways a sharded run can silently go wrong -- a lost shard, a duplicated fold, and shards run
against different feature files -- and reconstructs the exact OOF vector when the split is clean.

Run: .venv/bin/python tests/test-merge-shards-coverage.py
"""
import importlib.util
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location('merge_esta_shards',
                                              os.path.join(ROOT, 'tools/merge-esta-shards.py'))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

N = 40


def shard(idx, feats='f.pt', sid=0, ns=2):
    idx = np.asarray(idx)
    rng = np.random.default_rng(int(idx.sum()))
    soft = rng.random((len(idx), 2)).astype(np.float32)
    return {'model': 'ESTA', 'shard': sid, 'nshards': ns, 'folds': [f's{sid}'], 'idx': idx,
            'pred': soft.argmax(1), 'soft': soft, 'in_dim': 512, 'feats': feats}


def expect_exit(fn, needle, name):
    try:
        fn()
    except SystemExit as e:
        assert needle in str(e), f'{name}: wrong message: {e}'
        print(f'  ok   {name} -> refused ({needle})')
        return
    raise AssertionError(f'{name}: merged when it should have refused')


def test_clean_split_reconstructs():
    a, b = shard(np.arange(0, N, 2), sid=0), shard(np.arange(1, N, 2), sid=1)
    pred, soft = M.assemble([a, b], N)
    assert (pred != -1).all(), 'clean split left clips unscored'
    np.testing.assert_allclose(soft[a['idx']], a['soft'])
    np.testing.assert_allclose(soft[b['idx']], b['soft'])
    assert (pred == soft.argmax(1)).all(), 'pred does not match the merged softmax'
    print('  ok   clean split reconstructs every clip in clip order')


def test_missing_shard_refused():
    expect_exit(lambda: M.assemble([shard(np.arange(0, N, 2), sid=0)], N),
                'unscored', 'lost shard')


def test_overlapping_shards_refused():
    expect_exit(lambda: M.assemble([shard(np.arange(N), sid=0), shard(np.arange(N), sid=1)], N),
                'scored twice', 'duplicated fold')


def test_mismatched_feature_files_refused():
    with tempfile.TemporaryDirectory() as d:
        M.TOK = d
        torch.save(shard(np.arange(0, N, 2), feats='a.pt', sid=0),
                   f'{d}/shard_isbclip_ESTA_0of2.pt')
        torch.save(shard(np.arange(1, N, 2), feats='b.pt', sid=1),
                   f'{d}/shard_isbclip_ESTA_1of2.pt')
        expect_exit(lambda: M.gather('ESTA', '', 2), 'different feature files', 'mixed features')


def test_wrong_shard_count_refused():
    with tempfile.TemporaryDirectory() as d:
        M.TOK = d
        torch.save(shard(np.arange(N), sid=0), f'{d}/shard_isbclip_ESTA_0of2.pt')
        expect_exit(lambda: M.gather('ESTA', '', 2), 'expected 2 shards', 'short shard count')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f'{len(tests)}/{len(tests)} pass')
