#!/usr/bin/env python3
"""D1 fix: build a CLASSIFIER-INDEPENDENT scene grouping for leave-scene-out eval.

The existing `views` are KMeans(8) on the SAME appearance features used for classification
(tools/apply_zone_prior.py:62) -> circular. No real camera IDs exist in filenames ({class}_tr{N}.mp4).
This clusters on each clip's static BACKGROUND (temporal-median of its 32 pre-extracted frames,
downscaled grayscale) -> a scene descriptor independent of the moving subject / classifier features.
Saves sweep_results/zone_prior/clip_scene_map.json (additive; does NOT touch clip_view_map.json).
"""
import argparse
import os, glob, json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'sweep_results', 'zone_prior', 'clip_scene_map.json')
DOWN = 32  # background downscale side


def bg_descriptor(npy_path):
    a = np.load(npy_path, mmap_mode='r')          # (32,224,224,3) uint8
    med = np.median(np.asarray(a[::4]), axis=0)   # temporal median over ~8 frames -> static bg
    g = med.mean(2)                               # grayscale (224,224)
    # block-average downscale to DOWNxDOWN
    s = 224 // DOWN
    g = g[:s*DOWN, :s*DOWN].reshape(DOWN, s, DOWN, s).mean(axis=(1, 3))
    v = g.flatten()
    v = (v - v.mean()) / (v.std() + 1e-6)          # per-clip contrast normalize (lighting invariance)
    return v.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    # KMeans random_state was hardcoded 0, so the LCO grouping carried no measurable
    # split-choice uncertainty (brief.md S16/S18). Exposed so it can be swept.
    ap.add_argument('--cluster_seed', type=int, default=0, help='KMeans random_state')
    ap.add_argument('--clusters', type=int, default=8)
    ap.add_argument('--out', default=None,
                    help='output json. Default: the canonical path for cluster_seed=0/clusters=8, '
                         'otherwise a suffixed name - a sweep must never overwrite the canonical map '
                         '(the S6/S9 silent-overwrite failure mode).')
    args = ap.parse_args()
    out = args.out
    if out is None:
        canon = args.cluster_seed == 0 and args.clusters == 8
        out = OUT if canon else OUT.replace('.json', f'_k{args.clusters}cs{args.cluster_seed}.json')

    # map basename -> npy path (4-class unsafe subset only, labels 0-3)
    npys = glob.glob(os.path.join(ROOT, 'data', '*', '*', '*.npy'))
    by_base = {}
    for p in npys:
        base = os.path.basename(p)[:-4]  # strip .npy -> '0_tr1.mp4'
        cls = int(os.path.basename(os.path.dirname(p)).split('_')[0])
        if cls < 4:
            by_base[base] = p

    bases = sorted(by_base)
    X = np.stack([bg_descriptor(by_base[b]) for b in bases])
    print(f"clips: {len(bases)} | descriptor dim: {X.shape[1]}")

    km = KMeans(args.clusters, random_state=args.cluster_seed, n_init=10).fit(X)
    groups = km.labels_
    rec = {b: int(g) for b, g in zip(bases, groups)}
    json.dump(rec, open(out, 'w'), indent=2)
    from collections import Counter
    print("scene-group sizes:", dict(sorted(Counter(groups).items())))
    print(f"[saved] {out}")

    # compare to the old KMeans-appearance grouping (quantify circularity/divergence)
    old = json.load(open(os.path.join(ROOT, 'sweep_results', 'zone_prior', 'clip_view_map.json')))
    old_by_base = {r['basename']: r['view'] for r in old}
    common = [b for b in bases if b in old_by_base]
    a = [rec[b] for b in common]; c = [old_by_base[b] for b in common]
    ari = adjusted_rand_score(c, a)
    print(f"ARI(background-scene vs appearance-KMeans) over {len(common)} clips = {ari:.3f}")
    print("  (near 1 = old grouping already ~scene-based; near 0 = old grouping was appearance-confounded)")


if __name__ == '__main__':
    main()
