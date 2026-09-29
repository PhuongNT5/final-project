"""Train the deployment checkpoint for the ESTA lane (edge measurement + demo).

Separate from run-esta-lso.py on purpose. That script produces the *evidence* — pooled out-of-fold
predictions, where every clip is scored by a model that never saw its source video. This script
produces the *artifact* — one set of weights trained on the whole corpus, which has no honest
accuracy number of its own and must never be quoted as one.

Epoch selection uses a held-out source group (the first LSO fold), so no clip from a validation
source is in the training pool. The 3 seeds are saved together: the OOF numbers are 3-seed ensemble
numbers, so the deployed artifact has to be the same ensemble or the reported accuracy is not the
accuracy of the thing that ships.

Usage: .venv/bin/python train-esta-deployment-checkpoint.py --model ESTA
"""
import argparse
import importlib.util
import os

import numpy as np
import torch

import esta_training as T
import isafetybench_dataset as ISB

OUT = 'sweep_results/relation_tokens/deploy'


def driver():
    spec = importlib.util.spec_from_file_location(
        'run_esta_lso', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run-esta-lso.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ESTA', choices=['ESTA', 'B1', 'B1w', 'B2'])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=T.EPOCHS)
    ap.add_argument('--feats', default='sweep_results/relation_tokens/feats_isbclip_all.pt')
    args = ap.parse_args()

    feats, y, groups, clip_ids = driver().load_aligned(args.feats)
    in_dim = feats['wholeframe'].shape[-1]
    # The first LSO fold's held-out sources become the validation pool: epoch selection therefore
    # never sees a clip from a source it is training on, exactly as in the evaluation runs.
    _, val = ISB.folds(y, groups, n_splits=10, seed=0)[0]
    fit = np.setdiff1d(np.arange(len(y)), val)
    print(f"[deploy] {args.model} | train {len(fit)} clips / {len(set(groups[fit]))} sources | "
          f"val {len(val)} clips / {len(set(groups[val]))} held-out sources", flush=True)

    states, vals = [], []
    for s in range(args.seeds):
        m, best = T.train_one(args.model, feats, fit, val, y, seed=s, epochs=args.epochs,
                              in_dim=in_dim)
        states.append({k: v.cpu() for k, v in m.state_dict().items()})
        vals.append(best['val_macro'])
        print(f"  seed {s}: best val macro {best['val_macro']:.1f}% at epoch {best['epoch']}",
              flush=True)

    os.makedirs(OUT, exist_ok=True)
    path = f'{OUT}/{args.model.lower()}-deploy-{args.seeds}seed.pt'
    torch.save({'model': args.model, 'in_dim': in_dim, 'seeds': args.seeds, 'states': states,
                'val_macro': vals, 'val_sources': sorted(set(groups[val].tolist())),
                'note': 'trained on 9/10 LSO sources; NOT an accuracy artifact -- quote the OOF '
                        'runs from run-esta-lso.py, never this checkpoint'}, path)
    print(f"[deploy] saved {path}  (val macro per seed: "
          f"{', '.join(f'{v:.1f}' for v in vals)})")


if __name__ == '__main__':
    main()
