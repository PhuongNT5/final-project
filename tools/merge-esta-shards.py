"""Merge sharded LSO-1100 runs into the single OOF artifact the adjudicator consumes.

Sharding splits independent folds across processes: it changes wall clock and nothing else. This
script refuses unless the shards together cover every clip exactly once, so a lost, duplicated or
mismatched shard cannot quietly produce a partial result that still looks like a complete one.
Scoring and file writing are delegated to run-esta-lso.finalise, so a merged run and an unsharded
run emit identical artifacts by construction.

Usage: .venv/bin/python tools/merge-esta-shards.py --model ESTA --nshards 4
"""
import argparse
import glob
import importlib.util
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TOK = 'sweep_results/relation_tokens'


def driver():
    """run-esta-lso.py is hyphenated (repo convention), so it is loaded by path, not imported."""
    spec = importlib.util.spec_from_file_location('run_esta_lso',
                                                  os.path.join(ROOT, 'run-esta-lso.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gather(model, tag_suffix, nshards):
    paths = sorted(glob.glob(f'{TOK}/shard_isbclip_{model}{tag_suffix}_*of{nshards}.pt'))
    if len(paths) != nshards:
        raise SystemExit(f'expected {nshards} shards, found {len(paths)}: {paths}')
    shards = [torch.load(p, weights_only=False) for p in paths]
    if len({sh['feats'] for sh in shards}) != 1:
        raise SystemExit('shards ran on different feature files -- refusing to merge')
    if {sh['shard'] for sh in shards} != set(range(nshards)):
        raise SystemExit(f'shard ids are not 0..{nshards-1} -- refusing to merge')
    return shards


def assemble(shards, n):
    """Scatter the shard predictions back into clip order, refusing on imperfect coverage.

    Folds partition the corpus, so every clip must be scored by exactly one shard. A gap means a
    shard was lost; an overlap means two shards ran the same fold. Either way the merged vector
    would not be the OOF vector the protocol specifies."""
    pred = np.full(n, -1)
    soft = np.zeros((n, 2), dtype=np.float32)
    seen = np.zeros(n, dtype=int)
    for sh in shards:
        idx = np.asarray(sh['idx'])
        seen[idx] += 1
        pred[idx] = sh['pred']
        soft[idx] = sh['soft']
    if (seen != 1).any():
        raise SystemExit(f'coverage broken: {int((seen == 0).sum())} clips unscored, '
                         f'{int((seen > 1).sum())} scored twice -- refusing to merge')
    return pred, soft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--tag_suffix', default='')
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()

    shards = gather(args.model, args.tag_suffix, args.nshards)
    D = driver()
    _, y, groups, clip_ids = D.load_aligned(shards[0]['feats'])
    yn = y.numpy()

    pred, soft = assemble(shards, len(yn))
    folds = sorted({f for sh in shards for f in sh['folds']})
    D.finalise(args.model, args.tag_suffix, yn, y, pred, soft, clip_ids, groups, args.seeds,
               len(folds), note=f', merged from {args.nshards} shards')


if __name__ == '__main__':
    main()
