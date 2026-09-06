#!/usr/bin/env python3
"""E — stratified random k-fold + Nadeau–Bengio for the relation-Transformer.

Complements the leave-camera-out (camera-group) protocol with a conventional stratified random k-fold,
and runs the Nadeau–Bengio corrected resampled t-test (relation-Transformer vs a scene-only baseline)
across folds. Pools 516 clips. CPU.

--repeats R re-splits the data R times with a different fold seed. A SINGLE k-fold split gives the
Nadeau-Bengio test only k=5 difference samples (df=4), which is too few to trust; the reported
"+/-" is then a 5-point spread, not a confidence interval. Repeats pool k*R diffs (df=kR-1).

Usage: .venv/bin/python run_kfold_relation.py [--k 5] [--repeats 10] [--seeds 1] [--epochs 60]
"""
import argparse
import numpy as np
import torch
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split

import run_relation_transformer as R
from unsafenet import dedupe
from eval_utils.metrics import macro_accuracy, top1, confusion
from eval_utils.paired_tests import nadeau_bengio

TOK = 'sweep_results/relation_tokens'


def load_pooled(tag, dedupe_policy='none'):
    """Pool train+test. dedupe_policy drops clips whose byte-identical twin carries a different
    label (see unsafenet/dedupe.py); 'none' reproduces the raw, contaminated set."""
    Xa, Ma, ya = R.load('train', tag); Xb, Mb, yb = R.load('test', tag)
    da = torch.load(f'{TOK}/tokens_{tag}train.pt', weights_only=False)
    db = torch.load(f'{TOK}/tokens_{tag}test.pt', weights_only=False)
    X = torch.cat([Xa, Xb]); M = torch.cat([Ma, Mb]); y = torch.cat([ya, yb])
    views = np.array([str(v) for v in (list(da['views']) + list(db['views']))])
    keep = np.concatenate([dedupe.keep_mask(da['clip_ids'], 'train', dedupe_policy),
                           dedupe.keep_mask(db['clip_ids'], 'test',
                                            'keep' if dedupe_policy == 'none' else None)])
    if not keep.all():
        print(f"[dedupe:{dedupe_policy}] dropped {int((~keep).sum())} of {len(keep)} pooled clips")
    ki = torch.from_numpy(np.where(keep)[0])
    return X[ki], M[ki], y[ki], views[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=1, help='independent k-fold re-splits')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--tag', default='yoloapp_')
    ap.add_argument('--dedupe', default='none', choices=['none', 'drop_all', 'keep_one'],
                    help='drop clips with a byte-identical, differently-labelled twin')
    args = ap.parse_args()

    X, M, y, views = load_pooled(args.tag, args.dedupe)
    yn = y.numpy(); n_num = X.shape[-1] - 1
    rel = np.full(len(yn), -1); scene = np.full(len(yn), -1)
    rel_soft = np.zeros((len(yn), 4), dtype=np.float32)  # pooled out-of-fold probabilities (leakage-free)
    diffs, n_tr, n_te, fold_rel, fold_scene = [], [], [], [], []
    splits = [(r, tr, te) for r in range(args.repeats)
              for tr, te in StratifiedKFold(n_splits=args.k, shuffle=True,
                                            random_state=r).split(np.zeros(len(yn)), yn)]
    for fi, (rep, tr, te) in enumerate(splits):
        # scene-only baseline within the fold (view→majority-class from this fold's train)
        vmaj, cnt = {}, {}
        for i in tr:
            cnt.setdefault(views[i], Counter())[yn[i]] += 1
        vmaj = {v: c.most_common(1)[0][0] for v, c in cnt.items()}
        gmaj = Counter(yn[tr]).most_common(1)[0][0]
        scene[te] = [vmaj.get(views[i], gmaj) for i in te]
        # relation-Transformer seed-ensemble
        fit, val = train_test_split(tr, test_size=0.15, stratify=yn[tr], random_state=0)
        soft = torch.zeros(len(te), 4)
        for s in range(args.seeds):
            m, _ = R.train_one(X[fit], M[fit], y[fit], X[val], M[val], y[val],
                               seed=s, epochs=args.epochs, use_cw=False, n_num=n_num, aux_w=0.5)
            soft += torch.softmax(R.predict(m, X[te], M[te]), 1)
        pr = soft.argmax(1).numpy()
        if rep == 0:                      # repeat 0 is one complete partition -> the OOF prediction set
            rel[te] = pr
            rel_soft[te] = (soft / args.seeds).numpy()
        ra = 100 * (pr == yn[te]).mean(); sa = 100 * (scene[te] == yn[te]).mean()
        fold_rel.append(ra); fold_scene.append(sa); diffs.append(ra - sa)
        n_tr.append(len(tr)); n_te.append(len(te))
        print(f"  rep {rep+1} fold {fi % args.k + 1}: relation {ra:.1f}%  scene-only {sa:.1f}%  Δ={ra-sa:+.1f}", flush=True)

    nb = nadeau_bengio(diffs, int(np.mean(n_tr)), int(np.mean(n_te)))
    # richer multi-metric scorecard on the pooled out-of-fold predictions (additive)
    from sklearn.metrics import f1_score, cohen_kappa_score, matthews_corrcoef, precision_score, recall_score
    f1m = f1_score(yn, rel, average='macro') * 100
    kap = cohen_kappa_score(yn, rel); mcc = matthews_corrcoef(yn, rel)
    prec = precision_score(yn, rel, average='macro', zero_division=0) * 100
    rec = recall_score(yn, rel, average='macro', zero_division=0) * 100
    se = np.std(fold_rel, ddof=1) / np.sqrt(len(fold_rel))
    corr_se = np.sqrt(np.var(fold_rel, ddof=1) * (1.0 / len(fold_rel) + np.mean(n_te) / np.mean(n_tr)))
    print(f"\n=== STRATIFIED {args.k}-FOLD x {args.repeats} REPEATS "
          f"({len(splits)} folds, pooled {len(yn)} clips) ===")
    print(f"  relation-Transformer: top-1 {np.mean(fold_rel):.1f}±{np.std(fold_rel):.1f}% | "
          f"pooled {int((rel==yn).sum())}/{len(yn)} | macro {macro_accuracy(yn, rel)*100:.1f}%")
    print(f"  scorecard: macro-F1 {f1m:.1f}% | macro-precision {prec:.1f}% | macro-recall {rec:.1f}% | "
          f"Cohen κ {kap:.3f} | MCC {mcc:.3f}")
    print(f"  fold-mean top-1 {np.mean(fold_rel):.1f}%  naive SE {se:.2f}  NB-corrected SE {corr_se:.2f} "
          f"(95% CI {np.mean(fold_rel)-1.96*corr_se:.1f}-{np.mean(fold_rel)+1.96*corr_se:.1f}%)")
    print(f"  scene-only baseline : top-1 {np.mean(fold_scene):.1f}±{np.std(fold_scene):.1f}%")
    print(f"  Nadeau–Bengio (relation vs scene-only): mean Δ={nb['mean_diff']:.1f}pp  t={nb['t']:.2f}  df={len(diffs)-1}  p={nb['p_value']:.4f}"
          f"  {'(significant)' if nb['p_value']<0.05 else '(n.s.)'}")
    print("  NOTE: random k-fold has every camera in train, so the per-view-majority scene baseline is")
    print("        informative here. Under leave-camera-out it degenerates to a CONSTANT global-majority")
    print("        predictor (macro 25% by arithmetic) — the two baselines are different estimators, so the")
    print("        old '77.1% -> 16.9% collapse' contrast is WITHDRAWN. See brief.md S2; task A3b rebuilds it.")

    # dump leakage-free out-of-fold probs — a valid selection set for τ/T (each clip predicted by folds
    # that never saw it), disjoint from the blind 64-clip test. Consumed by run_operating_point.py --select_logits.
    # dedupe policy goes in the filename: a drop_all run must never overwrite the canonical
    # `--dedupe none` OOF set (the S6/S9 silent-overwrite failure mode).
    suffix = '' if args.dedupe == 'none' else f'_{args.dedupe}'
    oof_path = f'{TOK}/kfold_oof_softmax_{args.tag}{suffix}.pt'
    torch.save({'ens_softmax': torch.from_numpy(rel_soft), 'labels': y, 'tag': args.tag}, oof_path)
    print(f"  [saved OOF selection set] {oof_path}")


if __name__ == '__main__':
    main()
