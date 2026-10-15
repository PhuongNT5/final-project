#!/usr/bin/env python3
"""POST-HOC measure study: score frozen predictions with measures the thesis does not report.

NOT a pre-registered gate. Re-trains nothing and writes no new prediction -- it only re-scores
prediction files that already exist, so no number in chapter 6 can move because of this script.

Why this exists
---------------
Section 2.4.5 concedes that balanced accuracy is SYMMETRIC -- it charges a missed unsafe clip and a
false alarm the same -- while the deployment task is not: a missed unsafe behaviour is the cost the
system exists to avoid, a false alarm is an annoyance. Section 8.3.9 nevertheless keeps balanced
accuracy as the headline, correctly, because it was the pre-registered quantity. The gap that
leaves is that the asymmetric reading was never MEASURED, only admitted. This script measures it:

  * recall of unsafe at a fixed false-alarm rate (1%, 5%, 10%) -- the deployment question
  * F2, which weights recall four times as heavily as precision
  * AUROC and average precision, which need no threshold at all, so they are immune both to the
    0.5 cut and to the checkpoint-selection drift of section 4.5.6
  * balanced accuracy at the best in-sample threshold, which separates "the representation is
    weak" from "the 0.5 cut is misplaced"

Every figure gets a cluster-bootstrap CI at the SOURCE unit (B = 5000, seed = 0, alpha = 0.05,
percentile), the same unit and the same settings as section 3.7.4, never the clip unit.

Threshold honesty: the FAR thresholds and the best-threshold row are chosen on the very scores
being reported, so they are IN-SAMPLE and are labelled `[trong mau]`. They bound what a tuned
threshold could give; they are not an out-of-sample estimate. Round-2 artifacts additionally carry
`val_rows`, which allows the nested, out-of-sample version; that is reported separately when present.

Usage: .venv/bin/python analyze_posthoc_asymmetric_measures.py [--out report.json]
"""
import argparse
import glob
import json
import os

import numpy as np

from eval_utils.cluster_bootstrap import balanced_accuracy, metric_cluster_bootstrap

PUBLISHED = 'sweep_results/relation_tokens/enc_isbclip_B1_lso-1100_headline-none_oof.json'
ROUND1 = 'sweep_results/posthoc_sensitivity/*_oof.json'
ROUND2 = 'sweep_results/posthoc_sensitivity/round2/*_oof.json'
FAR_TARGETS = (0.01, 0.05, 0.10)


def far_threshold(y, p, target):
    """Lowest threshold whose false-alarm rate on the negatives stays at or under `target`."""
    neg = np.sort(p[y == 0])[::-1]
    k = int(np.floor(target * len(neg)))
    return float(neg[k]) if k < len(neg) else 0.0


def recall_at_far(y, p, target):
    t = far_threshold(y, p, target)
    return float((p[y == 1] > t).mean())


def fbeta(y, pred, beta=2.0):
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    b2 = beta * beta
    return float((1 + b2) * prec * rec / (b2 * prec + rec))


def auroc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return float('nan')
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(y, p):
    order = np.argsort(-p)
    yo = y[order]
    tp = np.cumsum(yo)
    prec = tp / np.arange(1, len(yo) + 1)
    pos = yo.sum()
    return float((prec * yo).sum() / pos) if pos else float('nan')


def best_threshold_ba(y, p):
    cands = np.unique(p)
    best = max(((balanced_accuracy(y, (p > t).astype(int)), float(t)) for t in cands),
               key=lambda x: x[0])
    return best


def _ci(y, p, groups, fn, scale=100.0):
    point, lo, hi, nb = metric_cluster_bootstrap(y, p, groups, metric=fn, n_boot=5000, seed=0)
    return {'value': round(point * scale, 2), 'lo': round(lo * scale, 2),
            'hi': round(hi * scale, 2), 'draws': nb}


def score(name, y, p, groups):
    """Every measure, each with a source-unit cluster-bootstrap interval. `p` = P(unsafe)."""
    pred = (p > 0.5).astype(int)
    # Every metric below is closed over the SCORES, not over hardened labels, so the bootstrap
    # resamples scores and labels together and a threshold-based metric is re-thresholded per draw.
    out = {'config': name, 'n': int(len(y)), 'n_groups': int(len(np.unique(groups))),
           'prevalence_unsafe': round(100 * float((y == 1).mean()), 2)}
    out['ba_at_0.5'] = _ci(y, p, groups, lambda yt, pp: balanced_accuracy(yt, (pp > 0.5).astype(int)))
    out['f2_at_0.5'] = _ci(y, p, groups, lambda yt, pp: fbeta(yt, (pp > 0.5).astype(int), 2.0))
    out['f1_at_0.5'] = _ci(y, p, groups, lambda yt, pp: fbeta(yt, (pp > 0.5).astype(int), 1.0))
    out['auroc'] = _ci(y, p, groups, auroc)
    out['average_precision'] = _ci(y, p, groups, average_precision)
    for t in FAR_TARGETS:
        out[f'recall_unsafe_at_far_{int(t * 100)}pct'] = _ci(
            y, p, groups, lambda yt, pp, t=t: recall_at_far(yt, pp, t))
        out[f'threshold_for_far_{int(t * 100)}pct'] = round(far_threshold(y, p, t), 4)
    ba_star, t_star = best_threshold_ba(y, p)
    out['ba_at_best_threshold_IN_SAMPLE'] = round(100 * ba_star, 2)
    out['best_threshold_IN_SAMPLE'] = round(t_star, 4)
    out['recall_unsafe_at_0.5'] = round(100 * float((pred[y == 1] == 1).mean()), 2)
    out['far_at_0.5'] = round(100 * float((pred[y == 0] == 1).mean()), 2)
    return out


def nested_threshold(rec, y, p):
    """Threshold maximising BA on the within-fold VAL split only, applied to the held-out fold.

    Out-of-sample with respect to the test fold. Still optimistic in one named way: the val split
    was already used to pick the checkpoint, so it is not a fresh split.
    """
    val = rec.get('val_rows')
    if not val:
        return None
    by_fold = {}
    for r in val:
        by_fold.setdefault(r['fold'], []).append(r)
    ts = []
    for f, rows in sorted(by_fold.items()):
        yv = np.array([r['y_true'] for r in rows])
        pv = np.array([r['prob_unsafe'] for r in rows])
        ts.append(best_threshold_ba(yv, pv)[1])
    t = float(np.mean(ts))
    return {'threshold_mean_over_folds': round(t, 4),
            'ba': round(100 * balanced_accuracy(y, (p > t).astype(int)), 2),
            'recall_unsafe': round(100 * float(((p > t)[y == 1]).mean()), 2),
            'far': round(100 * float(((p > t)[y == 0]).mean()), 2)}


def load(path, group_of):
    rec = json.load(open(path))
    rows = rec['rows']
    y = np.array([r['y_true'] for r in rows])
    p = np.array([r['prob_unsafe'] for r in rows], dtype=float)
    g = np.array([r.get('fold_id') or group_of[r['clip_id']] for r in rows])
    return rec, y, p, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='sweep_results/posthoc_sensitivity/asymmetric_measures.json')
    args = ap.parse_args()
    pub = json.load(open(PUBLISHED))
    group_of = {r['clip_id']: r['fold_id'] for r in pub['rows']}
    results = []
    paths = [PUBLISHED] + sorted(glob.glob(ROUND1)) + sorted(glob.glob(ROUND2))
    for path in paths:
        rec, y, p, g = load(path, group_of)
        # The three thread-control artifacts all carry config `B1_published_config` inside, so
        # the file stem is what distinguishes them; name by stem and keep the artifact path.
        name = os.path.basename(path).replace('_oof.json', '')
        if path == PUBLISHED:
            name = 'B1_PUBLISHED_29-09 (frozen, 1 thread era)'
        row = score(name, y, p, g)
        row['source_artifact'] = path
        nt = nested_threshold(rec, y, p)
        if nt:
            row['nested_val_threshold'] = nt
        results.append(row)
        print(f"{name:42s} BA={row['ba_at_0.5']['value']:6.2f} "
              f"[{row['ba_at_0.5']['lo']:.2f},{row['ba_at_0.5']['hi']:.2f}]  "
              f"AUROC={row['auroc']['value']:6.2f}  AP={row['average_precision']['value']:6.2f}  "
              f"R@FAR5={row['recall_unsafe_at_far_5pct']['value']:6.2f}  "
              f"F2={row['f2_at_0.5']['value']:6.2f}  "
              f"BA*={row['ba_at_best_threshold_IN_SAMPLE']:6.2f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({'note': 'POST-HOC measure study. Not a gate. Re-scores frozen predictions only.',
               'bootstrap': {'unit': 'source video (fold_id)', 'B': 5000, 'seed': 0,
                             'alpha': 0.05, 'kind': 'percentile'},
               'results': results}, open(args.out, 'w'), indent=1)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
