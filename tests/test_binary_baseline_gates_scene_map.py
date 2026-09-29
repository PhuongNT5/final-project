"""Real-data integration checks for the LCO-691 grouping used by run_binary_baseline_gates.py.

Unlike tests/test_binary_baseline_gates.py (synthetic, structural no-leakage proof), this file
reads the ACTUAL clip_scene_map_691.json + tokens_bin8_{train,test}.pt built by Lane B
(laneb-2026-09-16-binary-data-prep.md) and checks the precondition the gate's LCO grouping
depends on: every clip that will be pooled has a real (non-'na') scene-cluster group, so no clip
silently falls into a shared 'unmapped' bucket that would mix clips from different real clusters
under one fake group id (a leakage channel distinct from the fold-splitting logic itself).

Usage: .venv/bin/python tests/test_binary_baseline_gates_scene_map.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import binary_gate_lib as G


def test_every_pooled_bin8_clip_has_a_real_scene_group():
    sm = json.load(open(G.SCENE_MAP_691))
    _, _, _, cid_tr = G.load_bin8('train')
    _, _, _, cid_te = G.load_bin8('test')
    cids = list(cid_tr) + list(cid_te)
    assert len(cids) == 691, len(cids)
    missing = [c for c in cids if c not in sm]
    assert not missing, f'{len(missing)} pooled clips absent from clip_scene_map_691.json: {missing[:5]}'


def test_scene_groups_are_the_documented_k8_clusters():
    sm = json.load(open(G.SCENE_MAP_691))
    groups = sorted(set(sm.values()))
    assert groups == list(range(8)), groups
    sizes = {g: sum(1 for v in sm.values() if v == g) for g in groups}
    # documented cluster sizes, laneb-2026-09-16-binary-data-prep.md Step 1
    expected = {0: 22, 1: 189, 2: 58, 3: 170, 4: 123, 5: 101, 6: 14, 7: 14}
    assert sizes == expected, sizes


def test_group_5_is_the_documented_all_unsafe_degenerate_cluster():
    # phase-06 Key Insights: cluster 5 (101 clips) is 100% unsafe (0 safe clips) — the LCO fold
    # most likely to force the single-class-training-pool fallback path (see the synthetic
    # degenerate-fold test in test_binary_baseline_gates.py) when held out under a small pool.
    sm = json.load(open(G.SCENE_MAP_691))
    _, _, lab_tr, cid_tr = G.load_bin8('train')
    _, _, lab_te, cid_te = G.load_bin8('test')
    cid_all = np.concatenate([cid_tr, cid_te]); lab_all = np.concatenate([lab_tr, lab_te])
    group5 = np.array([sm[c] for c in cid_all]) == 5
    sub_in_group5 = set(lab_all[group5].tolist())
    assert sub_in_group5 == {0, 1}, sub_in_group5          # only WalkwayViol/UnauthorInt, both unsafe
    assert int(group5.sum()) == 101, int(group5.sum())


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'  PASS  {fn.__name__}')
    print('ALL PASS')
