"""Representation ladder under both protocols, scored against the no-pixel floor.

Three systems on identical clips:
  1. no-pixel floor      - camera id only (unsafenet.scene_confound)
  2. frozen CLIP + probe - generic appearance, no geometry
  3. YOLO+relation       - entity geometry (numbers from run_lco_eval.py / run_kfold_relation.py)

Same-distribution ranks these almost identically; leave-one-camera-out separates them. That
separation is the evidence that the pipeline learns behaviour rather than the camera.

Usage: .venv/bin/python run_representation_ladder.py
"""
import argparse, collections, json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from unsafenet import dedupe

CLS = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']
OUT = 'sweep_results/representation_ladder.json'


def prep(path, split=None, dedupe_policy='none'):
    d = torch.load(path, weights_only=False)
    X = normalize(d['feats'].float().numpy().mean(1))          # temporal mean-pool + L2
    y = np.array(list(d['labels']))
    v = np.array([str(x) for x in d['views']])
    m = y < 4                                                   # 4 unsafe classes = the task
    if split and dedupe_policy != 'none':
        keep = dedupe.keep_mask(d['names'], split, dedupe_policy)
        print(f"[dedupe:{dedupe_policy}] {split}: dropped {int((~keep[m]).sum())} of {int(m.sum())} clips")
        m = m & keep
    return X[m], y[m], v[m]


def probe(Xa, ya, Xb):
    return LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced').fit(Xa, ya).predict(Xb)


def score(y, p):
    rec = [float((p[y == c] == c).mean()) for c in range(4)]
    return {'top1': float((p == y).mean()), 'macro': float(np.mean(rec)),
            'per_class': dict(zip(CLS, [round(100 * r, 1) for r in rec]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dedupe', default='none', choices=['none', 'drop_all', 'keep_one'],
                    help='filter TRAIN clips with a byte-identical, differently-labelled twin; '
                         'the 64-clip test set is left frozen for paper comparability')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    Xtr, ytr, vtr = prep('sweep_results/r2/frameout_clip_train.pt', 'train', args.dedupe)
    Xte, yte, vte = prep('sweep_results/r2/frameout_clip_test.pt', 'test', args.dedupe)
    res = {'dedupe': args.dedupe}

    # --- same-distribution (the paper's protocol) ---
    maj = {v: collections.Counter(ytr[vtr == v]).most_common(1)[0][0] for v in set(vtr)}
    res['same_dist'] = {'n': len(yte),
                        'floor': score(yte, np.array([maj[x] for x in vte])),
                        'clip_probe': score(yte, probe(Xtr, ytr, Xte))}

    # --- leave-one-camera-out over the pooled unsafe set ---
    X, y, v = np.concatenate([Xtr, Xte]), np.concatenate([ytr, yte]), np.concatenate([vtr, vte])
    pred, floor, folds = np.zeros_like(y), np.zeros_like(y), []
    for vv in sorted(set(v), key=int):
        te = v == vv
        pred[te] = probe(X[~te], y[~te], X[te])
        floor[te] = collections.Counter(y[~te]).most_common(1)[0][0]   # unseen camera -> global majority
        folds.append({'view': vv, 'n': int(te.sum()),
                      'clip_probe': float((pred[te] == y[te]).mean()),
                      'floor': float((floor[te] == y[te]).mean())})
    res['lco'] = {'n': len(y), 'folds': folds,
                  'floor': score(y, floor), 'clip_probe': score(y, pred)}

    # relation-Transformer LCO, measured previously by run_lco_eval.py (seeds=3, epochs=60)
    res['lco']['relation_reference'] = {
        'top1': 0.771, 'macro': 0.786,
        'per_class': {'WalkwayViol': 77.1, 'UnauthorInt': 44.4, 'OpenedPanel': 93.0, 'CarryOverload': 100.0},
        'source': 'plans/2026-09-02-fair-headtohead-vs-paper/reports/fair-headtohead-scorecard.md'}

    json.dump(res, open(args.out, 'w'), indent=2)
    for prot in ('same_dist', 'lco'):
        print(f"--- {prot} (n={res[prot]['n']}) ---")
        for k in ('floor', 'clip_probe', 'relation_reference'):
            if k in res[prot]:
                s = res[prot][k]
                print(f"  {k:20s} top1={100*s['top1']:5.1f}%  macro={100*s['macro']:5.1f}%  {s['per_class']}")
    print(f'-> written {args.out}')


if __name__ == '__main__':
    main()
