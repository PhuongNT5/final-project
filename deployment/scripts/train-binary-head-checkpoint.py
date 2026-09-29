"""train-binary-head-checkpoint.py — Train a BINARY (Safe/Unsafe) relation-head checkpoint
(E3 primary freeze candidate or E0 fallback) on the TRAIN split only, val-selected, and save a
state_dict consumable by deployment/scripts/export-p3-relation-head-binary.py --head_ckpt.

ZERO TEST TOUCHES BY CONSTRUCTION, not by discipline: this file imports only `load_binary` and
`train_one` from run_relation_transformer.py (never `main_binary`, `predict`, or `evaluate_suite`),
and the ONLY `load_binary(...)` call below hardcodes the split argument to the literal string
`'train'`. `sweep_results/relation_tokens/tokens_bin8_test.pt` is therefore never opened by this
process — grep this file for `load_binary(` to confirm there is exactly one call site and it is not
`'test'`. Model/epoch/seed selection uses only the (label,view)-stratified val slice carved out of
TRAIN below (same carve as run_relation_transformer.py::main_binary L226-231, copied verbatim so this
script does not need to import main_binary). No metric is computed, printed, or saved on the 125-clip
blind test — see reports/test-touch-log.md.

Usage:
    .venv/bin/python3 deployment/scripts/train-binary-head-checkpoint.py --arch E3 --seeds 1 \
        --epochs 60 --dedupe drop_conflict --out deployment/export/ckpt-bin8-e3.pt
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from run_relation_transformer import apply_coord_variant, load_binary, train_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arch', required=True, choices=['E0', 'E1', 'E2', 'E3'])
    ap.add_argument('--tag', default='bin8_')
    ap.add_argument('--dedupe', default='drop_conflict',
                     choices=['none', 'drop_all', 'keep_one', 'drop_conflict'])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--seeds', type=int, default=1,
                     help='deployment artifact under time pressure: 1 seed is acceptable (see report)')
    ap.add_argument('--dim', type=int, default=None, help='E1 width override; ignored by E0/E3')
    ap.add_argument('--coord_variant', default=None, choices=['T1', 'T2', 'T3'])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    coord = args.coord_variant or ('T3' if args.arch == 'E1' else 'T1')
    # The only load_binary() call in this file — split hardcoded to 'train'. tokens_bin8_test.pt is
    # never opened by this script.
    Xtr, Mtr, ytr, ysub_tr, views_tr, _ = load_binary('train', args.tag, args.dedupe)
    Xtr = apply_coord_variant(Xtr, Mtr, coord)
    n_num = Xtr.shape[-1] - 1

    # Same (label,view)-stratified val carve as main_binary() (run_relation_transformer.py L226-231).
    strat_key = [f'{int(l)}_{v}' for l, v in zip(ytr.numpy(), views_tr)]
    try:
        fit, val = train_test_split(np.arange(len(ytr)), test_size=0.18, stratify=strat_key, random_state=0)
    except ValueError as e:
        print(f"[val split] (label,view) stratify infeasible ({e}) -> falling back to label-only stratify")
        fit, val = train_test_split(np.arange(len(ytr)), test_size=0.18, stratify=ytr.numpy(), random_state=0)
    Xf, Mf, yf, ysubf = Xtr[fit], Mtr[fit], ytr[fit], ysub_tr[fit]
    Xv, Mv, yv = Xtr[val], Mtr[val], ytr[val]
    print(f"[binary-train-only] arch={args.arch} tag={args.tag} dedupe={args.dedupe} coord={coord} "
          f"fit={len(yf)} val={len(yv)}  (test split NEVER loaded by this process)")

    rows = []
    t0 = time.time()
    for s in range(args.seeds):
        model, best = train_one(Xf, Mf, yf, Xv, Mv, yv, seed=s, epochs=args.epochs, use_cw=True,
                                 n_num=n_num, arch=args.arch, binary=True, ysub_f=ysubf, d=args.dim)
        rows.append({'seed': s, 'model': model, **best})
        print(f"  seed{s}: val_macro={best['val_macro']:.2f} val_top1={best['val_top1']:.2f} "
              f"@epoch{best['epoch']}  ({time.time() - t0:.1f}s elapsed)")

    blind = max(rows, key=lambda r: r['val_macro'])
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                             text=True).stdout.strip()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'state_dict': blind['state'],
        'ctor': {'arch': args.arch, 'n_num': n_num, 'd': args.dim},
        'blind_seed': blind['seed'], 'val_macro': blind['val_macro'], 'val_top1': blind['val_top1'],
        'epoch': blind['epoch'], 'seeds': args.seeds, 'epochs': args.epochs, 'tag': args.tag,
        'dedupe': args.dedupe, 'coord_variant': coord, 'git_commit': commit, 'argv': ' '.join(sys.argv),
        'train_fit_n': len(yf), 'val_n': len(yv),
    }, args.out)
    print(f"[saved] {args.out}  blind_seed={blind['seed']} val_macro={blind['val_macro']:.2f} "
          f"commit={commit[:12]}  total_time={time.time() - t0:.1f}s")
    print("No test-set file was opened by this process.")


if __name__ == '__main__':
    raise SystemExit(main())
