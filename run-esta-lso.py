"""R48 / G14-G16: ESTA and its whole-frame baselines on the LSO-1100 leave-source-out folds.

Every model here is scored on the SAME folds, in the SAME clip order, as the 64-d relation encoders
(E3/E1/E0): the fold partition is rebuilt from isafetybench_dataset.load's clip order, and the CLIP
feature tensor is permuted onto that order rather than the other way round. So the resulting OOF
files join to the existing ones on clip_id with no re-derivation, and every comparison is paired.

No threshold tuning, no per-fold model selection on held-out data, no re-runs for a better number.
0 test touches: LSO-1100 is cross-validation over the development corpus.

Usage: .venv/bin/python run-esta-lso.py --model ESTA [--seeds 3] [--epochs 60]
"""
import argparse
import json

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

import esta_training as T
import isafetybench_dataset as ISB
from model_entity_spatiotemporal import count_params
from run_kfold_relation import report_binary

TOK = 'sweep_results/relation_tokens'
FEATS = f'{TOK}/feats_isbclip_all.pt'
# Set by load_aligned from the feature file itself. The run banner used to hard-code
# "CLIP ViT-B/32", which silently mislabels any run on a different frozen tower (E2 ladder).
BACKBONE = 'unknown'


def load_aligned(path):
    """CLIP features permuted onto isafetybench_dataset.load's clip order.

    Refuses on any id or label disagreement: identical ids in identical order carrying identical
    labels is the only evidence that ESTA and the 64-d encoders scored the same corpus."""
    _, _, y, groups, ids = ISB.load('isb_')
    d = torch.load(path, weights_only=False)
    global BACKBONE
    BACKBONE = str(d.get('backbone', 'unknown'))
    pos = {str(c): i for i, c in enumerate(d['clip_ids'])}
    missing = [c for c in ids if c not in pos]
    if missing:
        raise SystemExit(f'{len(missing)} clips missing from {path} (first: {missing[:3]}) -- '
                         f'refusing to score a different corpus')
    perm = torch.as_tensor([pos[c] for c in ids])
    lab = torch.as_tensor(d['labels']).long()[perm]
    if not torch.equal(lab, y):
        raise SystemExit('label mismatch after permutation -- refusing')
    keys = [k for k in ('entity', 'geom', 'mask', 'wholeframe') if k in d]
    feats = {k: torch.as_tensor(d[k]).float()[perm] for k in keys}
    return feats, y, groups, ids


def finalise(model, tag_suffix, yn, y, pred, soft, clip_ids, groups, seeds, n_groups,
             note='', proto='lso-1100'):
    """Score the completed OOF vector and write the two artifacts downstream tools read.

    Shared by the single-process path and tools/merge-esta-shards.py so a sharded run and an
    unsharded one cannot produce differently-shaped artifacts."""
    print(f"\n=== {proto.upper()} ({model}, {int((pred != -1).sum())}/{len(yn)} clips scored, "
          f"{n_groups} held-out folds{note}) ===")
    report_binary(yn, y, pred, soft, model, seeds,
                  protocol_tag=f'binary-{proto}-{model}', sub_names=None)

    rows = [{'clip_id': str(clip_ids[i]), 'y_true': int(yn[i]), 'prob_unsafe': float(soft[i, 1]),
             'y_pred': int(pred[i]), 'fold_id': str(groups[i])} for i in range(len(yn))]
    jp = f'{TOK}/enc_isbclip_{model}{tag_suffix}_{proto}_headline-none_oof.json'
    json.dump({'model': f'clip_{model}', 'protocol': proto, 'dedupe_policy': 'none',
               'tag': 'isbclip_', 'n': len(rows), 'rows': rows}, open(jp, 'w'))
    pt = (f'{TOK}/lco_oof_softmax_isbclip_binary_{model}{tag_suffix}_none.pt' if proto == 'lso-1100'
          else f'{TOK}/oof_softmax_isbclip_binary_{model}{tag_suffix}_{proto}.pt')
    torch.save({'ens_softmax': torch.from_numpy(soft), 'labels': y, 'clip_ids': clip_ids,
                'groups': groups, 'tag': 'isbclip_', 'arch': model, 'oof_pred': pred}, pt)
    print(f"  [saved per-clip OOF] {jp}\n  [saved OOF tensor] {pt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ESTA', choices=['ESTA', 'B1', 'B1w', 'B2', 'TSF', 'B1xl'])
    ap.add_argument('--split', default='group', choices=['group', 'random'],
                    help="'group' = leave-source-out, the only protocol a headline number may come "
                         "from. 'random' = StratifiedKFold ignoring source identity: the G12 "
                         "diagnostic, for measuring how much the protocol inflates the number. A "
                         "random-split run is NEVER a generalisation result and its artifacts are "
                         "written under a rand-1100 name so they cannot be mistaken for one.")
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=T.EPOCHS)
    ap.add_argument('--folds', type=int, default=10)
    ap.add_argument('--feats', default=FEATS)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1,
                    help='split the folds across processes. Folds are independent, so this changes '
                         'wall clock only -- every model still sees the same folds, the same clips '
                         'and the same arithmetic. Merge with tools/merge-esta-shards.py.')
    ap.add_argument('--tag_suffix', default='',
                    help='appended to the output filenames; use for plumbing checks so a short '
                         'run can never be mistaken for, or overwrite, a headline artifact')
    args = ap.parse_args()

    feats, y, groups, clip_ids = load_aligned(args.feats)
    in_dim = feats['wholeframe'].shape[-1]
    yn = y.numpy()
    if args.split == 'random':
        proto = f'rand-{len(yn)}'
        splits = list(StratifiedKFold(n_splits=args.folds, shuffle=True,
                                      random_state=0).split(np.zeros(len(yn)), yn))
    else:
        proto = 'lso-1100'
        splits = ISB.folds(y, groups, n_splits=args.folds, seed=0)
    folds = [(f'f{i+1}', te) for i, (_, te) in enumerate(splits)]
    if args.nshards > 1:
        folds = folds[args.shard::args.nshards]
        print(f"[shard {args.shard}/{args.nshards}] folds: {', '.join(g for g, _ in folds)}")
    if args.split == 'random':
        print(f"[G12 DIAGNOSTIC -- SOURCE-IGNORING SPLIT] {args.folds} StratifiedKFold folds over "
              f"{len(yn)} clips. Clips of the same source video sit on BOTH sides of every "
              f"boundary, and 0 of {len(set(groups))} sources carry both classes (R41), so source "
              f"identity is a perfect label proxy here. This number measures PROTOCOL INFLATION "
              f"and is NOT a generalisation result; it must never be reported as one.")
    else:
        print(f"[grouping] source YouTube video id: {len(set(groups))} sources, 0 carry both "
              f"classes (R41) -> {args.folds} StratifiedGroupKFold folds; no source spans the "
              f"boundary.")
    print(f"[{proto}] model={args.model} backbone={BACKBONE} dim={in_dim} pooled {len(yn)} clips "
          f"(unsafe {int(yn.sum())}) | folds={len(folds)} | seeds={args.seeds} "
          f"| params={count_params(T.build(args.model, in_dim)):,}", flush=True)

    pred = np.full(len(yn), -1)
    soft = np.zeros((len(yn), 2), dtype=np.float32)
    for g, te in folds:
        tr = np.setdiff1d(np.arange(len(yn)), te)
        fit, val = train_test_split(tr, test_size=0.15, stratify=yn[tr], random_state=0)
        acc = torch.zeros(len(te), 2)
        for s in range(args.seeds):
            m, best = T.train_one(args.model, feats, fit, val, y, seed=s, epochs=args.epochs,
                                    in_dim=in_dim)
            acc += T.predict_softmax(m, args.model, feats, te)
        soft[te] = (acc / args.seeds).numpy()
        pred[te] = soft[te].argmax(1)
        print(f"  held-out group {g}: n={len(te):3d} unsafe={int(yn[te].sum())} "
              f"acc={100 * (pred[te] == yn[te]).mean():5.1f}%", flush=True)

    if args.nshards > 1:
        sp = f'{TOK}/shard_isbclip_{args.model}{args.tag_suffix}_{args.shard}of{args.nshards}.pt'
        torch.save({'model': args.model, 'shard': args.shard, 'nshards': args.nshards,
                    'folds': [g for g, _ in folds], 'idx': np.where(pred != -1)[0],
                    'pred': pred[pred != -1], 'soft': soft[pred != -1],
                    'in_dim': in_dim, 'feats': args.feats}, sp)
        print(f'  [shard saved] {sp}  ({int((pred != -1).sum())} clips scored)')
        return

    finalise(args.model, args.tag_suffix, yn, y, pred, soft, clip_ids, groups,
             args.seeds, len(folds), proto=proto)


if __name__ == '__main__':
    main()
