#!/usr/bin/env python3
"""
GZT GO/NO-GO PROBE (walkway + danger). Tests the one claim GZT rests on:
  "A person's floor position separates WalkwayViolation (0) from UnauthorizedIntervention (1)."

Reports per-feature ROC-AUC (Walk vs Unauth) + Mann-Whitney p, and two pooled logistic-regression
5-fold CV models — WALKWAY-only (z1,z2,z3) vs WALKWAY+DANGER (adds z4,z5) — so you can see whether
hand-drawn machine zones add separation over the auto walkway. Prints a clear verdict.

Decision: GO if combined AUC>=0.70 or LR accuracy gain>=+0.12; WEAK-GO if >=0.62/+0.06; else NO-GO.

Usage:
  .venv/bin/python3 tools/check_zone_separation.py --min-coverage 0.2
"""
import os, json, argparse
import numpy as np
import torch
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

OUT_DIR = 'sweep_results/zone_prior'


def load(min_cov):
    X, y, v, names = [], [], [], None
    for subset in ['train', 'test']:
        p = os.path.join(OUT_DIR, f'zone_features_{subset}.pt')
        if not os.path.exists(p):
            continue
        d = torch.load(p, map_location='cpu', weights_only=False)
        names = d['feature_names']
        ok = d['coverage'].numpy() >= min_cov
        X.append(d['features'].numpy()[ok]); y.append(d['labels'].numpy()[ok]); v.append(d['views'].numpy()[ok])
    if not X:
        raise SystemExit("No zone_features_*.pt — run tools/apply_zone_prior.py first.")
    return np.concatenate(X), np.concatenate(y), np.concatenate(v), names


def directed_auc(yc, score):
    walk = (yc == 0).astype(int)                 # positive = WalkwayViolation
    if walk.sum() == 0 or (walk == 0).sum() == 0:
        return float('nan'), ''
    auc = roc_auc_score(walk, score)
    return auc, ('Walk>Unauth' if auc >= 0.5 else 'Unauth>Walk')


def lr_eval(X, y, cols):
    """Pooled logistic regression CV on the given feature columns (rows finite in all of them)."""
    sub = X[:, cols]
    fin = np.isfinite(sub).all(1)
    Xs, ys = sub[fin], y[fin]
    cnt = np.bincount(ys, minlength=2)
    if cnt[0] < 5 or cnt[1] < 5:
        return None
    k = int(min(5, cnt[0], cnt[1]))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
    cv = StratifiedKFold(k, shuffle=True, random_state=0)
    acc = cross_val_score(clf, Xs, ys, cv=cv, scoring='accuracy')
    auc = cross_val_score(clf, Xs, ys, cv=cv, scoring='roc_auc')
    maj = max(cnt) / cnt.sum()
    return {'n': int(fin.sum()), 'k': k, 'acc': float(acc.mean()), 'acc_std': float(acc.std()),
            'auc': float(auc.mean()), 'auc_std': float(auc.std()), 'majority': float(maj),
            'gain': float(acc.mean() - maj)}


def verdict(auc, gain):
    if auc >= 0.70 or gain >= 0.12:
        return 'GO', 'Floor position separates the confused pair — build the Zone-Token head.'
    if auc >= 0.62 or gain >= 0.06:
        return 'WEAK-GO', 'Partial separation — build GZT expecting a partial fix; refine zones / per-camera.'
    return 'NO-GO', 'Geometry does not separate the pair — revert to deployment-led, or refine zones and re-probe.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-coverage', type=float, default=0.2)
    args = ap.parse_args()

    X, y, V, names = load(args.min_coverage)
    m = np.isin(y, [0, 1])
    Xb, yb, Vb = X[m], y[m], V[m]
    nW, nU = int((yb == 0).sum()), int((yb == 1).sum())
    print(f"=== GZT go/no-go: WalkwayViol vs UnauthorInt  (Walk={nW} Unauth={nU}, min-cov={args.min_coverage}) ===\n")

    print(f"{'feature':<20}{'n':>5}{'AUC':>7}{'dir':>13}{'Walk_mu':>9}{'Unauth_mu':>11}{'MWU_p':>9}")
    feat_auc = []
    for j, name in enumerate(names):
        col = Xb[:, j]; fin = np.isfinite(col)
        if fin.sum() < 10 or len(set(yb[fin].tolist())) < 2:
            print(f"{name:<20}{int(fin.sum()):>5}   (not drawn / too few)"); continue
        s, ys = col[fin], yb[fin]
        auc, d = directed_auc(ys, s)
        feat_auc.append(max(auc, 1 - auc))
        try:
            p = mannwhitneyu(s[ys == 0], s[ys == 1]).pvalue
        except ValueError:
            p = float('nan')
        print(f"{name:<20}{int(fin.sum()):>5}{auc:>7.3f}{d:>13}{s[ys==0].mean():>9.3f}{s[ys==1].mean():>11.3f}{p:>9.4f}")

    wk = lr_eval(Xb, yb, [0, 1, 2])
    has_danger = X.shape[1] >= 5 and np.isfinite(Xb[:, 3:5]).all(1).sum() >= 10
    cb = lr_eval(Xb, yb, [0, 1, 2, 3, 4]) if has_danger else None
    print("\nPooled logistic-regression CV:")
    for tag, r in [('walkway-only (z1,z2,z3)', wk), ('walkway+DANGER (z1..z5)', cb)]:
        if r is None:
            print(f"  {tag:<26}: (danger zones not drawn yet)" if 'DANGER' in tag else f"  {tag}: n/a")
            continue
        print(f"  {tag:<26}: acc={r['acc']:.3f}±{r['acc_std']:.3f}  AUC={r['auc']:.3f}  "
              f"(majority {r['majority']:.3f}, gain {r['gain']:+.3f}, n={r['n']}, {r['k']}-fold)")

    has5 = Xb.shape[1] >= 5
    print("\nper-view z2(walk) / z5(danger) means [Walk | Unauth]:")
    for vw in sorted(set(Vb.tolist())):
        mm = Vb == vw; w = yb[mm] == 0; u = yb[mm] == 1
        if w.sum() and u.sum():
            dtxt = " | z5 --"
            if has5 and np.isfinite(Xb[mm][w, 4]).any():
                dtxt = f" | z5 {np.nanmean(Xb[mm][w,4]):+.3f}|{np.nanmean(Xb[mm][u,4]):+.3f}"
            print(f"  view {vw}: n={w.sum()+u.sum():3d}  z2 {Xb[mm][w,1].mean():+.3f}|{Xb[mm][u,1].mean():+.3f}{dtxt}")

    best = cb or wk
    auc_combined = max([best['auc']] + feat_auc) if best else max(feat_auc or [0.5])
    gain = best['gain'] if best else 0.0
    tag, msg = verdict(auc_combined, gain)
    bar = '=' * 64
    src = 'walkway+danger' if cb else 'walkway-only'
    print(f"\n{bar}\nVERDICT: {tag}   (basis: {src})\n  combined AUC={auc_combined:.3f}  LR gain={gain:+.3f}\n  {msg}\n{bar}")

    rep = {'n_walk': nW, 'n_unauth': nU, 'has_danger': bool(has_danger),
           'per_feature_auc': {names[j]: float(a) for j, a in enumerate(feat_auc)} if len(feat_auc) == len(names) else None,
           'lr_walkway_only': wk, 'lr_walkway_danger': cb,
           'combined_auc': float(auc_combined), 'lr_gain': float(gain), 'verdict': tag, 'message': msg}
    json.dump(rep, open(os.path.join(OUT_DIR, 'separation_report.json'), 'w'), indent=2)
    print(f"[SAVED] {OUT_DIR}/separation_report.json")


if __name__ == '__main__':
    main()
