"""The three pre-registered SECONDARY readouts of the E2 backbone-ladder test.

tools/compare-oof-runs.py already prints the primary number (paired cluster-bootstrap CI and
McNemar). This file prints only what the pre-registration of 2026-10-10 §4 named as the sharper
test, because the headline delta alone cannot separate the two hypotheses:

  if the limit is LABEL/DATA  -> fold spread does NOT narrow, the all-wrong clips stay wrong,
                                 and error stays concentrated in the same folds
  if the limit is REPRESENTATION -> at least the first two must change

The fold assignment comes from `isafetybench_dataset.folds` -- the same call the training driver
makes -- so it cannot drift from what was actually held out. A hand-built fold map was the known
weakness of the earlier audit (open question #2 of the council minute) and is not used here.

Usage:
  .venv/bin/python tools/analyse-backbone-ladder.py BASE_oof.json NEW_oof.json \
      [--allwrong sweep_results/relation_tokens/e2_baseline_allwrong.json]
"""
import argparse
import json
import sys

sys.path.insert(0, '.')
import isafetybench_dataset as ISB


def fold_map(n_splits=10, seed=0):
    """clip_id -> held-out fold label, from the driver's own splitter."""
    _, _, y, groups, ids = ISB.load('isb_')
    out = {}
    for i, (_, te) in enumerate(ISB.folds(y, groups, n_splits=n_splits, seed=seed)):
        for j in te:
            out[str(ids[j])] = f'f{i + 1}'
    return out


def rows_of(path):
    return {str(r['clip_id']): r for r in json.load(open(path))['rows']}


def per_fold_ba(rows, fmap):
    """Balanced accuracy per held-out fold. A fold missing a class yields None, not a crash."""
    acc = {}
    for cid, r in rows.items():
        f = fmap.get(cid)
        if f is None:
            continue
        acc.setdefault(f, [0, 0, 0, 0])
        y, p = int(r['y_true']), int(r['y_pred'])
        acc[f][0 if (y == 1 and p == 1) else 1 if (y == 1) else 2 if (p == 0) else 3] += 1
    out = {}
    for f, (tp, fn, tn, fp) in acc.items():
        out[f] = None if not (tp + fn) or not (tn + fp) else 50 * (tp / (tp + fn) + tn / (tn + fp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base'); ap.add_argument('new')
    ap.add_argument('--allwrong', default='sweep_results/relation_tokens/e2_baseline_allwrong.json')
    ap.add_argument('--label_base', default='base'); ap.add_argument('--label_new', default='new')
    args = ap.parse_args()

    fmap = fold_map()
    A, Bn = rows_of(args.base), rows_of(args.new)
    if set(A) != set(Bn):
        raise SystemExit(f'clip sets differ ({len(A)} vs {len(Bn)}) -- refusing')

    # (1) fold spread
    print(f'--- (1) BA theo fold (dự đoán: nếu giới hạn là NHÃN, dải KHÔNG co lại) ---')
    fa, fb = per_fold_ba(A, fmap), per_fold_ba(Bn, fmap)
    keys = sorted(fa, key=lambda k: int(k[1:]))
    print(f"{'fold':>5} {args.label_base:>9} {args.label_new:>9} {'Δ':>7}")
    for k in keys:
        va, vb = fa[k], fb[k]
        d = '' if None in (va, vb) else f'{vb - va:+7.1f}'
        print(f'{k:>5} {("--" if va is None else f"{va:8.1f}%")} '
              f'{("--" if vb is None else f"{vb:8.1f}%")} {d}')
    for lbl, f in ((args.label_base, fa), (args.label_new, fb)):
        v = [x for x in f.values() if x is not None]
        print(f'  dải {lbl}: {max(v) - min(v):.1f} pp  (min {min(v):.1f} / max {max(v):.1f})')

    # (2) the clips the whole B/32 family got wrong
    aw = json.load(open(args.allwrong))
    ids = [c for c in aw['all_wrong_clip_ids'] if c in Bn]
    fixed = [c for c in ids if int(Bn[c]['y_pred']) == int(Bn[c]['y_true'])]
    print(f'\n--- (2) {len(ids)} clip sai ở CẢ họ {aw["family"]} ---')
    print(f'  {args.label_new} sửa được: {len(fixed)}/{len(ids)} = {100 * len(fixed) / len(ids):.1f}%')
    print(f'  (dự đoán: nếu giới hạn là NHÃN thì phần lớn VẪN sai)')

    # (3) where the error sits
    print(f'\n--- (3) phân bố lỗi theo fold ---')
    for lbl, R in ((args.label_base, A), (args.label_new, Bn)):
        err = {}
        for cid, r in R.items():
            if int(r['y_pred']) != int(r['y_true']):
                err[fmap.get(cid, '?')] = err.get(fmap.get(cid, '?'), 0) + 1
        tot = sum(err.values())
        top = sorted(err.items(), key=lambda kv: -kv[1])[:3]
        share = 100 * sum(v for _, v in top) / tot
        print(f'  {lbl:>9}: {tot} lỗi | 3 fold tệ nhất {[k for k, _ in top]} gánh {share:.1f}%')


if __name__ == '__main__':
    main()
