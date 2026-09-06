#!/usr/bin/env python3
"""Leave-Camera-Out (LCO) scene-robust evaluation of the relation-Transformer.

The dataset is scene-confounded (view predicts behaviour; scene-only baseline = 52/64 same-distribution).
LCO holds out one camera view at a time: train on the other views, test on the held-out view's clips →
the TRUE cross-camera behaviour-generalization. Pools all 516 clips (452 train + 64 test) grouped by
view (this is a separate generalization analysis, not the paper's fixed-64 test).

A3b (2026-09-06): the old single "scene-only" baseline conflated a trivial constant predictor with a
"model" claim. THREE baselines are now reported, matched to the k-fold script where possible:
  1. `prior`            — training class-prior / global-majority CONSTANT. Trivial floor. Macro accuracy
                           of a constant predictor over k classes is exactly 100/k% by construction
                           (100% recall on the predicted class, 0% on the rest) — printed explicitly so
                           the number is never mistaken for a scene/appearance model.
  2. `scene-appearance`  — HONEST baseline: logistic regression on whole-frame (background-encoding)
                           features (`tokens_wf_*.pt`, MobileNetV3, T=16, mean-pooled over time), fit on
                           the TRAINING cameras only, applied to the held-out camera. This is the
                           actually-transferable scene classifier the withdrawn claim needed.
  3. `per-view-majority` — the k-fold script's baseline (view → majority class within that view). Under
                           LCO the held-out view was never seen in training, so this is UNDEFINED for the
                           held-out clips; we degenerate it to `prior` and say so explicitly, to make the
                           k-fold/LCO protocol asymmetry visible instead of hidden.

The relation-Transformer's margin is then reported against baseline 2 (the meaningful comparison), with
a paired bootstrap CI, not against baseline 1.

Usage: .venv/bin/python run_lco_eval.py --seeds 5 --epochs 60 --tag yolo26L1_ --aux_w 0.5 \
           --scene_map sweep_results/zone_prior/clip_scene_map.json
"""
import argparse
import numpy as np
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import run_relation_transformer as R
from unsafenet import dedupe
from eval_utils.metrics import macro_accuracy, top1, per_class_recall, confusion

NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']
N_CLASSES = 4
TOK = 'sweep_results/relation_tokens'
DEDUPE = ['none']   # set from --dedupe before load_pooled()
KEEP = [None]       # pooled boolean keep-mask, set by load_pooled(), reused by load_wf_pooled()


def load_pooled(tag):
    Xa, Ma, ya = R.load('train', tag)
    Xb, Mb, yb = R.load('test', tag)
    va = torch.load(f'{TOK}/tokens_{tag}train.pt', weights_only=False)['views']
    vb = torch.load(f'{TOK}/tokens_{tag}test.pt', weights_only=False)['views']
    X = torch.cat([Xa, Xb]); M = torch.cat([Ma, Mb]); y = torch.cat([ya, yb])
    views = np.array([str(v) for v in (list(va) + list(vb))])
    da = torch.load(f'{TOK}/tokens_{tag}train.pt', weights_only=False)
    db = torch.load(f'{TOK}/tokens_{tag}test.pt', weights_only=False)
    cids = np.array([str(c) for c in (list(da['clip_ids']) + list(db['clip_ids']))])
    if DEDUPE[0] != 'none':
        keep = np.concatenate([dedupe.keep_mask(da['clip_ids'], 'train', DEDUPE[0]),
                               dedupe.keep_mask(db['clip_ids'], 'test', None)])
        print(f"[dedupe:{DEDUPE[0]}] dropped {int((~keep).sum())} of {len(keep)} pooled clips")
        KEEP[0] = keep
        ki = torch.from_numpy(np.where(keep)[0])
        X, M, y, views, cids = X[ki], M[ki], y[ki], views[keep], cids[keep]
    return X, M, y, views, cids


def load_wf_pooled(labels_ref):
    """Whole-frame (background-encoding, crop-free) features for the scene-appearance baseline.
    Same clip-glob order as any tag built by build_relation_tokens.py (verified: labels match
    1:1 against the main token set) — asserted below rather than assumed."""
    da = torch.load(f'{TOK}/tokens_wf_train.pt', weights_only=False)
    db = torch.load(f'{TOK}/tokens_wf_test.pt', weights_only=False)
    feats = torch.cat([da['feats'], db['feats']])           # (N, T, 64)
    labels = torch.cat([da['labels'], db['labels']])
    if KEEP[0] is not None:                                  # same dedupe filter as load_pooled()
        ki = torch.from_numpy(np.where(KEEP[0])[0])
        feats, labels = feats[ki], labels[ki]
    if not torch.equal(labels, labels_ref):
        raise RuntimeError(
            "tokens_wf_{train,test}.pt clip order does not match the main token set — "
            "cannot align the scene-appearance baseline. Rebuild both from the same clip glob."
        )
    return feats.mean(dim=1).numpy()                         # (N, 64) clip-level pooled background feature


def paired_bootstrap_diff(yn, pred_a, pred_b, n_boot=5000, seed=0):
    """Percentile bootstrap CI on macro_accuracy(pred_a) - macro_accuracy(pred_b), resampling clips
    (both predictions were produced out-of-fold w.r.t. the held-out camera, so this is a valid
    non-parametric CI on the paired difference)."""
    rng = np.random.default_rng(seed)
    n = len(yn)
    point = macro_accuracy(yn, pred_a) - macro_accuracy(yn, pred_b)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = macro_accuracy(yn[idx], pred_a[idx]) - macro_accuracy(yn[idx], pred_b[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--tag', default='app_')
    ap.add_argument('--aux_w', type=float, default=0.5)
    ap.add_argument('--scene_map', default=None,
                    help='D1: JSON {clip_id: group} to OVERRIDE views with a classifier-independent '
                         'background-KMeans scene grouping (still a PROXY, not verified camera metadata '
                         '— see tools/build_scene_grouping.py docstring / thesis limitations)')
    ap.add_argument('--dedupe', default='none', choices=['none', 'drop_all', 'keep_one'],
                    help='drop TRAIN clips that are byte-identical to a differently-labelled clip '
                         '(drop_all = both members of every contradictory group; keep_one = keep the '
                         'alphabetically-first member). See unsafenet/dedupe.py.')
    args = ap.parse_args()

    DEDUPE[0] = args.dedupe
    X, M, y, views, cids = load_pooled(args.tag)
    if args.scene_map:
        import json
        sm = json.load(open(args.scene_map))
        views = np.array([str(sm.get(c, sm.get(c[:-4], 'na'))) for c in cids])
        print(f"[scene_map] overriding views from {args.scene_map} | mapped "
              f"{int((views!='na').sum())}/{len(views)} clips")
        print("[LIMITATION] this grouping is a KMeans(8) proxy on per-clip static-background descriptors "
              "(tools/build_scene_grouping.py) — NOT verified camera metadata (no camera IDs exist in "
              "filenames, per r4). LCO strength is bounded by how well this proxy tracks real cameras.")
    else:
        print("[LIMITATION] no --scene_map given: using the OLDER appearance-KMeans `views` field "
              "(clip_view_map.json), which r4 flags as circular (clustered on the same kind of features "
              "the classifier consumes). Pass --scene_map for the classifier-independent grouping.")
    yn = y.numpy(); n_num = X.shape[-1] - 1
    uniq = sorted(set(views))
    print(f"pooled {len(yn)} clips | views {uniq} | feat {X.shape[-1]} | seeds {args.seeds} epochs {args.epochs}")

    wf_X = load_wf_pooled(y)

    model_pred = np.full(len(yn), -1)
    prior_pred = np.full(len(yn), -1)
    appear_pred = np.full(len(yn), -1)
    # brief.md S18 inferred that LCO per-class recall carries large TRAINING variance from a single
    # perturbation. Retaining each seed's own argmax makes that variance directly measurable instead
    # of inferred: the ensemble above hides it.
    seed_pred = np.full((args.seeds, len(yn)), -1)
    for v in uniq:
        te = np.where(views == v)[0]
        tr = np.where(views != v)[0]
        if len(te) == 0:
            continue
        # baseline 1 — prior: training global-majority CONSTANT (held-out camera unseen)
        fold_majority = Counter(yn[tr]).most_common(1)[0][0]
        prior_pred[te] = fold_majority

        # baseline 2 — scene-appearance: logistic regression on whole-frame background features,
        # fit on TRAINING cameras only, applied to the held-out camera.
        try:
            sc = StandardScaler().fit(wf_X[tr])
            clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=0)
            clf.fit(sc.transform(wf_X[tr]), yn[tr])
            appear_pred[te] = clf.predict(sc.transform(wf_X[te]))
        except Exception as e:
            print(f"  [scene-appearance] held-out view {v}: fit failed ({e}) — fell back to prior")
            appear_pred[te] = fold_majority

        # relation-Transformer: seed-ensemble, epoch selected on a val slice carved from training views
        fit, val = train_test_split(np.arange(len(tr)), test_size=0.15,
                                    stratify=yn[tr], random_state=0)
        soft = torch.zeros(len(te), 4)
        for s in range(args.seeds):
            model, _ = R.train_one(X[tr][fit], M[tr][fit], y[tr][fit],
                                   X[tr][val], M[tr][val], y[tr][val],
                                   seed=s, epochs=args.epochs, use_cw=False, n_num=n_num, aux_w=args.aux_w)
            sm_s = torch.softmax(R.predict(model, X[te], M[te]), 1)
            seed_pred[s, te] = sm_s.argmax(1).numpy()
            soft += sm_s
        model_pred[te] = soft.argmax(1).numpy()

        macc = 100 * (model_pred[te] == yn[te]).mean()
        pacc = 100 * (prior_pred[te] == yn[te]).mean()
        aacc = 100 * (appear_pred[te] == yn[te]).mean()
        print(f"  held-out view {v}: n={len(te):3d} classes={sorted(set(yn[te].tolist()))} "
              f"(train majority class={fold_majority}) → "
              f"transformer acc={macc:5.1f}%  prior acc={pacc:5.1f}%  scene-appearance acc={aacc:5.1f}%")

    n = len(yn)
    mc = int((model_pred == yn).sum())
    pc = int((prior_pred == yn).sum())
    ac = int((appear_pred == yn).sum())
    model_macro = macro_accuracy(yn, model_pred) * 100
    prior_macro = macro_accuracy(yn, prior_pred) * 100
    appear_macro = macro_accuracy(yn, appear_pred) * 100

    print(f"\n=== LEAVE-CAMERA-OUT (scene-robust, {n} clips pooled, {len(uniq)} camera-groups) ===")
    print(f"  relation-Transformer      : {mc}/{n} = {100*mc/n:.1f}% top-1 | macro {model_macro:.1f}%")
    print(f"  [1] prior (trivial floor) : {pc}/{n} = {100*pc/n:.1f}% top-1 | macro {prior_macro:.1f}% "
          f"(arithmetic expectation for a CONSTANT predictor over {N_CLASSES} classes = "
          f"{100/N_CLASSES:.1f}% macro EXACTLY, by construction — 100% recall on one class, 0% on the "
          f"rest; the measured macro above will equal this whenever a single class is predicted for the "
          f"whole held-out camera, which is the common case here)")
    print(f"  [2] scene-appearance      : {ac}/{n} = {100*ac/n:.1f}% top-1 | macro {appear_macro:.1f}% "
          f"(logistic regression on whole-frame MobileNetV3 features, background/scene-encoding, "
          f"NOT crop-based — trained on training cameras only, no access to the held-out camera)")
    print(f"  [3] per-view-majority     : UNDEFINED under LCO (the held-out view has zero training "
          f"clips, so 'majority class within this view from training data' cannot be computed) — "
          f"degenerates to [1] prior by construction. This is the honest reason the k-fold script's "
          f"77.1% baseline and this script's numbers are NOT comparable estimators; see brief.md S2.")
    print("  per-class recall (LCO transformer)    :",
          {NAMES[k]: round(per_class_recall(yn, model_pred).get(k, float('nan')) * 100, 1) for k in range(4)})
    print("  per-class recall (LCO scene-appearance):",
          {NAMES[k]: round(per_class_recall(yn, appear_pred).get(k, float('nan')) * 100, 1) for k in range(4)})


    # per-seed spread — the ensemble numbers above average this away (brief.md S18/S19)
    print(f"\n  --- per-seed spread over {args.seeds} training seeds (same folds, same data) ---")
    st = {NAMES[k]: [] for k in range(4)}
    tops = []
    for s_ in range(args.seeds):
        pr = seed_pred[s_]
        tops.append(100 * (pr == yn).mean())
        rec = per_class_recall(yn, pr)
        for k in range(4):
            st[NAMES[k]].append(round(rec.get(k, float('nan')) * 100, 1))
    print(f"  top-1 per seed : {[round(t,1) for t in tops]}  "
          f"mean {np.mean(tops):.1f}  sd {np.std(tops, ddof=1):.2f}  spread {max(tops)-min(tops):.1f} pp")
    for k in range(4):
        v = st[NAMES[k]]
        print(f"  recall {NAMES[k]:14s}: {v}  sd {np.std(v, ddof=1):5.2f}  spread {max(v)-min(v):5.1f} pp")
    print("  confusion (LCO transformer):\n", confusion(yn, model_pred, 4))
    print("  confusion (LCO scene-appearance):\n", confusion(yn, appear_pred, 4))

    diff, lo, hi = paired_bootstrap_diff(yn, model_pred, appear_pred)
    print(f"\n  → HONEST margin (relation-Transformer macro - scene-appearance macro) = "
          f"{diff*100:+.1f} pp  [95% bootstrap CI {lo*100:+.1f}, {hi*100:+.1f} pp, n_boot=5000, "
          f"resampled over {n} pooled clips]")
    if diff <= 0:
        print("  *** NEGATIVE RESULT: the scene-appearance baseline matches or beats the relation-"
              "Transformer under LCO. Report this plainly — it means camera-transferable behaviour "
              "signal is not demonstrated beyond what background appearance alone predicts. ***")
    print(f"\n  (withdrawn framing, DO NOT USE: the old +60.3pp figure compared the model against the "
          f"trivial constant baseline [1], not against a transferable classifier — see brief.md S2)")
    print("\nContrast: SAME-DISTRIBUTION scene-only=52/64 (81.2%), our S1a=54/64 (84.4%).")


if __name__ == '__main__':
    main()
