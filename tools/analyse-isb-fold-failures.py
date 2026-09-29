"""Per-fold / per-source failure analysis for an iSafetyBench LSO run.

Fold 6 of the E3 headline run sat at chance (50.9%) while its neighbours reached 87%. A 26-point
spread across held-out source packets is a finding about scene dependence and has to be characterised
rather than averaged away.

Usage: .venv/bin/python tools/analyse-isb-fold-failures.py <encoder_oof.json> [--worst 2]
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, '.')
import isafetybench_dataset as ISB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('encoder_oof')
    ap.add_argument('--tag', default='isb_')
    ap.add_argument('--folds', type=int, default=10)
    ap.add_argument('--worst', type=int, default=2, help='how many worst folds to break down')
    args = ap.parse_args()

    X, M, y, groups, clip_ids = ISB.load(args.tag)
    yn = y.numpy()
    rows = {r['clip_id']: r for r in json.load(open(args.encoder_oof))['rows']}
    pred = np.array([rows[str(c)]['y_pred'] for c in clip_ids])
    F = ISB.count_features(X, M)                    # persons/frame, objects/frame, mean conf
    splits = ISB.folds(y, groups, n_splits=args.folds, seed=0)

    def ba(idx):
        vals = [(pred[idx][yn[idx] == c] == c).mean() for c in (0, 1) if (yn[idx] == c).any()]
        return float(np.mean(vals)) * 100 if len(vals) == 2 else float('nan')

    print(f"{'fold':>5} {'n':>5} {'haz':>5} {'BA%':>6} {'src':>5} {'pers/f':>7} {'obj/f':>7} {'conf':>6}")
    per = []
    for k, (_, te) in enumerate(splits, 1):
        per.append((ba(te), k, te))
        print(f"{k:>5} {len(te):>5} {int(yn[te].sum()):>5} {ba(te):>6.1f} "
              f"{len(set(groups[te])):>5} {F[te, 0].mean():>7.2f} {F[te, 1].mean():>7.2f} "
              f"{F[te, 2].mean():>6.3f}")
    print(f"\ncorpus mean: pers/f={F[:, 0].mean():.2f} obj/f={F[:, 1].mean():.2f} "
          f"conf={F[:, 2].mean():.3f}")

    for b, k, te in sorted(per)[:args.worst]:
        print(f"\n--- fold {k} (BA {b:.1f}%) sources, worst first ---")
        srcs = sorted(set(groups[te]))
        stats = []
        for s in srcs:
            i = te[groups[te] == s]
            stats.append(((pred[i] == yn[i]).mean() * 100, s, len(i), int(yn[i][0]),
                          F[i, 0].mean(), F[i, 1].mean(), F[i, 2].mean()))
        for acc, s, n, lab, p_, o_, c_ in sorted(stats):
            print(f"  {s:<14} n={n:>3} label={'hazard' if lab else 'normal'} acc={acc:>5.1f}% "
                  f"pers/f={p_:.2f} obj/f={o_:.2f} conf={c_:.3f}")


if __name__ == '__main__':
    main()
