#!/usr/bin/env python3
"""
WS2 — k-fold cross-validation (rigor upgrade over the single N=64 test split).

Why: the thesis headline rested on ONE 64-clip test split → bootstrap CI [79.7, 95.3],
no significant difference vs baseline. Unsafe-Net carries NO day/session/camera metadata
(verified 2026-06-28: filenames are sequential `{cls}_tr/te{N}`, mp4s have no
creation_time tag, frame_manifest.csv has no day column), so a true day-grouped /
leave-one-day-out CV is impossible. This script instead pools ALL 691 clips and runs
stratified k-fold over the 516 four-class unsafe clips (effective N=516, ~8x the single
split). It reports:
  - per-fold + pooled top-1, macro-F1, per-class recall AND false-negative rate (FNR)
  - 5000-sample bootstrap CI on the pooled predictions
  - a SCENE-LEAKAGE-ROBUST variant: StratifiedGroupKFold over KMeans clusters of
    time-averaged appearance features (proxy for the missing session/camera grouping)
  - a linear-probe baseline on the same folds + Nadeau-Bengio corrected resampled t-test
    (does the Transformer head significantly beat a linear probe on these features? the
    temporal-stream ablation suggests it may not — reported honestly either way)

Reuses run_pairwise_contrastive (features, TransformerHead) and run_blind_eval
(stratified_fit_val, fit_select_model, probs_on). Read-only on the dataset.

Usage:
  .venv/bin/python3 run_kfold_eval.py [--k 5] [--seeds 42 123 314] [--clusters 24]
  .venv/bin/python3 run_kfold_eval.py --smoke     # fast correctness check
"""
import os, json, time, argparse
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from scipy import stats as sps

import run_pairwise_contrastive as pw
import run_blind_eval as be

CLASS_NAMES_4 = pw.CLASS_NAMES_4
OUT_DIR = 'plans/260628-thesis-improvement-strategy/reports'


def per_class(y, pred, nc=4):
    """Return per-class recall and false-negative rate (miss rate) as percentages."""
    rec, fnr = [], []
    for c in range(nc):
        m = (y == c); n = int(m.sum())
        r = int((pred[m] == c).sum()) / max(1, n)
        rec.append(round(100 * r, 1)); fnr.append(round(100 * (1 - r), 1))
    return rec, fnr


def bootstrap_ci(y, pred, iters=5000, seed=0):
    rng = np.random.RandomState(seed); n = len(y)
    correct = (pred == y).astype(float); top1, macro = [], []
    for _ in range(iters):
        idx = rng.randint(0, n, n)
        top1.append(100 * correct[idx].mean())
        macro.append(100 * f1_score(y[idx], pred[idx], average='macro', zero_division=0))
    pct = lambda a, q: round(float(np.percentile(a, q)), 1)
    return [pct(top1, 2.5), pct(top1, 97.5)], [pct(macro, 2.5), pct(macro, 97.5)]


def nadeau_bengio(diffs, n_train, n_test):
    """Corrected resampled paired t-test (Nadeau & Bengio 2003) for k-fold model diffs."""
    d = np.asarray(diffs, float); n = len(d)
    mean, var = d.mean(), d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0, round(mean, 3)
    t = mean / np.sqrt(var * (1.0 / n + n_test / float(n_train)))
    p = 2 * (1 - sps.t.cdf(abs(t), n - 1))
    return round(float(t), 3), round(float(p), 4), round(mean, 3)


def build_feats(feats, mode):
    """feats: (N,T,16,512) → (N,T,D).
    'stats'   = cat[mean,std,max] over the 16 spatial tokens (1536-d) — destroys location.
    'spatial' = stats + per-cell energy map (16) + activation centroid x,y (2) + spread (1):
                encodes WHERE in the 4x4 grid activity sits (WS4 zone-discriminative signal).
    Tokens are row-major over the 4x4 grid (idx = row*4 + col)."""
    stats = torch.cat([feats.mean(2), feats.std(2), feats.max(2).values], -1)
    if mode == 'stats':
        return stats
    e = feats.norm(dim=-1)                              # (N,T,16) per-cell magnitude
    p = e / (e.sum(-1, keepdim=True) + 1e-8)            # spatial distribution
    idx = torch.arange(16, dtype=torch.float32)
    gx, gy = idx % 4, torch.div(idx, 4, rounding_mode='floor')
    cx = (p * gx).sum(-1, keepdim=True); cy = (p * gy).sum(-1, keepdim=True)
    spread = (p * ((gx - cx) ** 2 + (gy - cy) ** 2)).sum(-1, keepdim=True).sqrt()
    return torch.cat([stats, e, cx, cy, spread], -1)   # (N,T,1536+16+3)


def eval_fold(stats4, y4, tr_idx, te_idx, seed, device, stats_dim, lr, epochs, patience):
    """Train Transformer head + linear-probe baseline on a fold; return their test preds."""
    fit_m, val_m = be.stratified_fit_val(y4[tr_idx], 0.2, 0)
    tr, va = tr_idx[fit_m], tr_idx[val_m]
    pw.set_seed(seed)
    model = pw.TransformerHead(stats_dim, 4, 384, 8, 6, 0.3).to(device)
    model = be.fit_select_model(model, stats4[tr], y4[tr], stats4[va], y4[va],
                                lr, device, epochs=epochs, patience=patience)
    tf_pred = be.probs_on(model, stats4[te_idx], device).argmax(1)
    lp = LogisticRegression(max_iter=2000, class_weight='balanced')
    lp.fit(stats4[tr].mean(1).numpy(), y4[tr].numpy())
    lp_pred = lp.predict(stats4[te_idx].mean(1).numpy())
    return tf_pred, lp_pred


def run_scheme(name, stats4, y4, ynp, groups, seeds, device, stats_dim, args):
    """Run one k-fold scheme over all seeds; return aggregated metrics dict."""
    per_seed_top1, per_seed_macro, per_class_rec, per_class_fnr = [], [], [], []
    diffs, cat_y, cat_tf = [], [], []
    n_tr, n_te = [], []
    for seed in seeds:
        tf_pred = np.full(len(ynp), -1, int); lp_pred = np.full(len(ynp), -1, int)
        if groups is None:
            splitter = StratifiedKFold(args.k, shuffle=True, random_state=seed).split(stats4, ynp)
        else:
            splitter = StratifiedGroupKFold(args.k, shuffle=True, random_state=seed).split(stats4, ynp, groups)
        for fi, (tr_idx, te_idx) in enumerate(splitter):
            a, b = eval_fold(stats4, y4, tr_idx, te_idx, seed, device, stats_dim,
                             args.lr, args.epochs, args.patience)
            tf_pred[te_idx] = a; lp_pred[te_idx] = b
            diffs.append(100 * (a == ynp[te_idx]).mean() - 100 * (b == ynp[te_idx]).mean())
            n_tr.append(len(tr_idx)); n_te.append(len(te_idx))
            print(f"    [{name}] seed{seed} fold{fi+1}: tf={100*(a==ynp[te_idx]).mean():.1f} "
                  f"lp={100*(b==ynp[te_idx]).mean():.1f}", flush=True)
        per_seed_top1.append(100 * (tf_pred == ynp).mean())
        per_seed_macro.append(100 * f1_score(ynp, tf_pred, average='macro', zero_division=0))
        rec, fnr = per_class(ynp, tf_pred); per_class_rec.append(rec); per_class_fnr.append(fnr)
        cat_y.append(ynp.copy()); cat_tf.append(tf_pred.copy())
    cy, ct = np.concatenate(cat_y), np.concatenate(cat_tf)
    ci_top1, ci_macro = bootstrap_ci(cy, ct)
    t, p, mdiff = nadeau_bengio(diffs, int(np.mean(n_tr)), int(np.mean(n_te)))
    return {
        'scheme': name, 'k': args.k, 'seeds': seeds, 'n_clips': len(ynp),
        'top1_mean': round(float(np.mean(per_seed_top1)), 2), 'top1_std': round(float(np.std(per_seed_top1)), 2),
        'top1_per_seed': [round(x, 1) for x in per_seed_top1],
        'macro_mean': round(float(np.mean(per_seed_macro)), 2), 'macro_std': round(float(np.std(per_seed_macro)), 2),
        'top1_bootstrap_ci95': ci_top1, 'macro_bootstrap_ci95': ci_macro,
        'per_class_recall_mean': [round(float(np.mean([s[c] for s in per_class_rec])), 1) for c in range(4)],
        'per_class_fnr_mean': [round(float(np.mean([s[c] for s in per_class_fnr])), 1) for c in range(4)],
        'transformer_vs_linear': {'mean_acc_diff_pp': mdiff, 'nadeau_bengio_t': t, 'p_value': p,
                                  'significant_05': bool(p < 0.05)},
        'confusion_matrix_seed0': confusion_matrix(cat_y[0], cat_tf[0], labels=list(range(4))).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 314])
    ap.add_argument('--clusters', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1.5e-4)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--patience', type=int, default=60)
    ap.add_argument('--feature', choices=['stats', 'spatial'], default='stats',
                    help="'spatial' adds location features (WS4 zone-prior test)")
    ap.add_argument('--schemes', nargs='+', default=['stratified', 'scene_group'],
                    choices=['stratified', 'scene_group'])
    ap.add_argument('--smoke', action='store_true', help='fast correctness check')
    args = ap.parse_args()
    if args.smoke:
        args.k, args.seeds, args.clusters, args.epochs, args.patience = 2, [42], 8, 15, 6

    t0 = time.time()
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | k={args.k} seeds={args.seeds} clusters={args.clusters} "
          f"epochs={args.epochs}\n")

    tr, te = pw.extract_all_features(device)
    feats = torch.cat([tr['features'], te['features']], 0)
    labels = torch.cat([tr['labels'], te['labels']], 0)
    stats = build_feats(feats, args.feature)  # (N,T,D) per --feature
    m4 = labels < 4
    stats4, y4 = stats[m4], labels[m4]
    ynp = y4.numpy(); stats_dim = int(stats4.shape[-1])
    print(f"  Pooled {len(labels)} clips → 4-class subset N={len(ynp)} "
          f"(per-class {[int((y4==c).sum()) for c in range(4)]}) | feature={args.feature} "
          f"dim={stats_dim}\n")

    # scene-leakage proxy groups: cluster clips by time-averaged appearance feature
    clip_repr = stats4.mean(1).numpy()
    groups = KMeans(args.clusters, random_state=0, n_init=10).fit_predict(clip_repr)
    print(f"  Scene clusters: {args.clusters} groups, sizes "
          f"{np.bincount(groups).tolist()}\n")

    results = {}
    if 'stratified' in args.schemes:
        print("── SCHEME: stratified k-fold (random) ──", flush=True)
        results['stratified'] = run_scheme('strat', stats4, y4, ynp, None, args.seeds,
                                           device, stats_dim, args)
    if 'scene_group' in args.schemes:
        print("\n── SCHEME: scene-group k-fold (leakage-robust) ──", flush=True)
        results['scene_group'] = run_scheme('group', stats4, y4, ynp, groups, args.seeds,
                                            device, stats_dim, args)

    print("\n" + "=" * 70)
    print(f"K-FOLD SUMMARY (4-class unsafe, pooled 691→N=516, feature={args.feature})")
    print("=" * 70)
    for key in results:
        r = results[key]
        print(f"\n[{r['scheme']}] top-1 {r['top1_mean']}±{r['top1_std']}% "
              f"CI{r['top1_bootstrap_ci95']} | macro {r['macro_mean']}±{r['macro_std']}% "
              f"CI{r['macro_bootstrap_ci95']}")
        print(f"   per-class recall {dict(zip(CLASS_NAMES_4, r['per_class_recall_mean']))}")
        print(f"   per-class FNR    {dict(zip(CLASS_NAMES_4, r['per_class_fnr_mean']))}")
        tv = r['transformer_vs_linear']
        print(f"   Transformer vs linear-probe: Δ={tv['mean_acc_diff_pp']}pp "
              f"N-B t={tv['nadeau_bengio_t']} p={tv['p_value']} "
              f"{'SIGNIFICANT' if tv['significant_05'] else 'not significant'}")
    if 'stratified' in results and 'scene_group' in results:
        leak = round(results['stratified']['top1_mean'] - results['scene_group']['top1_mean'], 2)
        print(f"\nScene-leakage gap (stratified − scene-group top-1): {leak} pp "
              f"{'→ possible scene leakage' if leak > 3 else '→ no strong scene leakage'}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {'protocol': 'pooled 691 clips; 4-class subset; per-fold inner-val epoch select; '
                       'no day metadata available → clip-level + scene-cluster grouping',
           'args': vars(args), 'runtime_min': round((time.time() - t0) / 60, 1), 'results': results}
    tag = ('-smoke' if args.smoke else '') + ('' if args.feature == 'stats' else f'-{args.feature}')
    path = os.path.join(OUT_DIR, f"kfold-eval{tag}.json")
    json.dump(out, open(path, 'w'), indent=2)
    print(f"\n[SAVED] {path} | {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
