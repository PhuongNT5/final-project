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
import json
import numpy as np
import torch
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split

import run_relation_transformer as R
import feature_ablation as ABL
from model_relation_head import count_params
from model_binary_encoders import build_encoder
from unsafenet import dedupe
from eval_utils.metrics import macro_accuracy, top1, confusion
from eval_utils.paired_tests import nadeau_bengio, mcnemar_exact
from eval_utils.suite import evaluate_suite

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


def load_pooled_binary(tag, dedupe_policy='none'):
    """Pool train+test for the binary Safe/Unsafe task (design-2026-09-16). Labels are derived
    from the source sub-class id (0-3 -> Unsafe=1, 4-7 -> Safe=0). dedupe uses task='binary'
    with the pre-registered asymmetric policy: TRAIN gets `dedupe_policy` (headline
    'drop_conflict'), TEST is always left untouched ('keep') per reports/test-touch-log.md
    2026-09-17 — this is a screening protocol (pooled OOF), it does not consume a test touch."""
    da = torch.load(f'{TOK}/tokens_{tag}train.pt', weights_only=False)
    db = torch.load(f'{TOK}/tokens_{tag}test.pt', weights_only=False)
    Xa, Ma, ysuba = da['tokens'].float(), da['mask'].float(), da['labels'].long()
    Xb, Mb, ysubb = db['tokens'].float(), db['mask'].float(), db['labels'].long()
    X = torch.cat([Xa, Xb]); M = torch.cat([Ma, Mb]); ysub = torch.cat([ysuba, ysubb])
    views = np.array([str(v) for v in (list(da['views']) + list(db['views']))])
    keep = np.concatenate([dedupe.keep_mask(da['clip_ids'], 'train', dedupe_policy, task='binary'),
                           dedupe.keep_mask(db['clip_ids'], 'test', 'keep', task='binary')])
    if not keep.all():
        print(f"[dedupe:binary/{dedupe_policy}] dropped {int((~keep).sum())} of {len(keep)} pooled clips "
              f"(train={dedupe_policy}, test untouched)")
    ki = torch.from_numpy(np.where(keep)[0])
    y = (ysub < 4).long()
    clip_ids = np.array(list(da['clip_ids']) + list(db['clip_ids']))[keep]
    return X[ki], M[ki], y[ki], ysub[ki], views[keep], clip_ids


SUBN = ['0:WalkwayViol', '1:UnauthorInt', '2:OpenedPanel', '3:CarryOverload',
        '4:SafeWalkway', '5:AuthorInt', '6:ClosedPanel', '7:SafeCarrying']


def report_binary(yn, ysub, rel, rel_soft, arch, seeds, protocol_tag, sub_names=SUBN):
    """Shared Tier 2/Tier 3 binary scorecard (design §4): eval_utils suite + per-sub-class recall.
    Used by both the k-fold and LCO drivers so the two protocols print an identical metric set.
    `sub_names=None` skips the per-sub-class block for corpora with no sub-class taxonomy."""
    res = evaluate_suite(yn, rel_soft, class_names=['Safe', 'Unsafe'],
                          protocol_tag=protocol_tag, n_seeds=seeds, unsafe_classes=[1])
    print(f"  balanced_acc={res['balanced_accuracy']['value']*100:.1f}% macro_f1={res['macro_f1']['value']*100:.1f}% "
          f"unsafe_recall={res['per_class']['Unsafe']['recall']*100:.1f}% "
          f"unsafe_precision={res['per_class']['Unsafe']['precision']*100:.1f}% "
          f"PR-AUC={res['pr_auc_ovr']['per_class'][1]:.3f} ROC-AUC={res['roc_auc_ovr']['per_class'][1]:.3f}")
    if sub_names:
        print("  per-sub-class recall (pooled OOF, folded back to source 0-7):")
        for c in range(len(sub_names)):
            sel = (ysub.numpy() == c)
            if sel.sum():
                r = float((rel[sel] == yn[sel]).mean()) * 100
                print(f"    {sub_names[c]:16s} n={int(sel.sum()):3d} recall={r:5.1f}%")
    return res


def _grouped_folds(args, y, clip_ids, groups=None):
    """-> (groups, folds, note, sub_names, n_sub, proto). One held-out unit per fold.

    unsafenet: one fold per scene group (true leave-camera-out over 8 KMeans groups).
    isafetybench: 150 single-class source videos make 150 leave-one-source-out folds both
    prohibitively slow and degenerate per fold, so sources are packed into `--folds` grouped folds
    (StratifiedGroupKFold) — still leave-source-out, since no source is ever split across the
    train/held-out boundary."""
    if args.dataset == 'isafetybench':
        import isafetybench_dataset as ISB
        if args.split == 'random':
            # G12 diagnostic ONLY: the source-leaking protocol the baseline literature uses. Sources
            # are ignored, so a source's clips land on both sides of the boundary and source identity
            # becomes a perfect label proxy (0/151 sources carry both classes, R41). Never a headline.
            skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
            splits = list(skf.split(np.zeros(len(y)), y.numpy() if torch.is_tensor(y) else y))
            note = (f"[G12 DIAGNOSTIC — SOURCE-LEAKING PROTOCOL] StratifiedKFold over {len(y)} clips "
                    f"IGNORING the {len(set(groups))} source ids: clips from one YouTube video appear "
                    f"in train AND held-out. Source identity is a perfect label proxy here (R41). "
                    f"This number measures protocol inflation and is NOT a generalisation result.")
            proto = f'rand-{len(y)}'
        elif args.permute_sources is not None:
            # G19 CONTROL: grouped folds over PLACEBO sources -- same group count, same per-class
            # group-size multiset, same class purity, same StratifiedGroupKFold machinery, but each
            # real video's clips are scattered. Isolates source identity from fold geometry: if this
            # lands near the random-split number, the leave-source-out drop is leakage, not difficulty.
            real = groups
            groups = ISB.placebo_groups(y, real, seed=args.permute_sources)
            splits = ISB.folds(y, groups, n_splits=args.folds, seed=0)
            note = (f"[G19 CONTROL — PLACEBO GROUPING, seed={args.permute_sources}] "
                    f"{len(set(groups))} fake sources matched to the {len(set(real))} real ones on "
                    f"count, per-class size multiset and class purity, but carrying no provenance. "
                    f"Fold geometry is held constant; only source identity is destroyed. This number "
                    f"is a CONTROL for the inflation measurement and is NOT a generalisation result.")
            proto = f'plc{args.permute_sources}-{len(y)}'
        else:
            splits = ISB.folds(y, groups, n_splits=args.folds, seed=0)
            note = (f"[grouping] source YouTube video id from filename: {len(set(groups))} sources, "
                    f"0 carry both classes (R41) -> packed into {args.folds} StratifiedGroupKFold "
                    f"folds; no source spans the train/held-out boundary.")
            proto = f'lso-{len(y)}'
        folds = [(f'f{i+1}', te) for i, (tr, te) in enumerate(splits)]
        return groups, folds, note, None, 2, proto
    sm = json.load(open(args.scene_map))
    groups = np.array([str(sm.get(c, sm.get(c[:-4], 'na'))) for c in clip_ids])
    folds = [(g, np.where(groups == g)[0]) for g in sorted(set(groups))]
    note = (f"[LIMITATION] scene groups from {args.scene_map} are a KMeans(8) background-descriptor "
            f"proxy, not verified camera metadata (see run_lco_eval.py / tools/build_scene_grouping.py).")
    return groups, folds, note, SUBN, 8, 'lco-691'


def run_lco(args):
    """Tier 3 leave-group-out (design §4 / coordinator 2026-09-17 step 2): holds out one camera/scene
    group (Unsafe-Net) or one packet of source videos (iSafetyBench) at a time. Pooled OOF, 0 test
    touches (same drop_conflict/train-only dedupe asymmetry as load_pooled_binary). Saves per-clip OOF
    JSON in the same row schema as the gates agent's sweep_results/binary_gates/*_oof.json files
    ({clip_id, y_true, prob_unsafe, y_pred, fold_id}) so McNemar vs S1/S2 can join on clip_id."""
    if args.dataset == 'isafetybench':
        import isafetybench_dataset as ISB
        X, M, y, src, clip_ids = ISB.load(args.tag)
        ysub = y                                  # no sub-class taxonomy on this corpus
        feat_layout = ISB.feat_layout(args.tag)
    else:
        X, M, y, ysub, views, clip_ids = load_pooled_binary(args.tag, args.dedupe)
        src = None
        feat_layout = None
    groups, folds, note, sub_names, n_sub, proto = _grouped_folds(args, y, clip_ids, src)
    if args.ablate != 'none':
        # Applied to the RAW tokens (before apply_coord_variant, which can delete columns) and kept
        # for the R42 floor below, so the encoder and the shortcut are denied the same cue.
        Xa = ABL.apply(X, M, args.ablate, feat_layout)
        frac = ABL.changed_fraction(X, Xa, M)
        print(f"[ablation] {args.ablate}: {ABL.describe(args.ablate)} | "
              f"{frac:.1%} of valid slots changed")
        if frac == 0.0:
            raise SystemExit(f'ablation {args.ablate} changed nothing -- channel already neutral; '
                             f'refusing to report it as a null result')
        X = Xa
    coord = args.coord_variant or ('T3' if args.arch == 'E1' else 'T1')
    X = R.apply_coord_variant(X, M, coord)
    yn = y.numpy(); n_num = X.shape[-1] - 1
    print(note)
    params = count_params(build_encoder(args.arch, n_num=n_num, n_sub=n_sub, d=args.dim))
    print(f"[{proto}] arch={args.arch} d={args.dim or (64 if args.arch == 'E1' else '-')} coord={coord} "
          f"pooled {len(yn)} clips (unsafe {int(yn.sum())}) | folds={len(folds)} | params={params:,}")

    rel = np.full(len(yn), -1)
    rel_soft = np.zeros((len(yn), 2), dtype=np.float32)
    for g, te in folds:
        tr = np.setdiff1d(np.arange(len(yn)), te)
        if len(te) == 0 or len(set(yn[tr].tolist())) < 2:
            print(f"  held-out group {g}: skipped (degenerate train pool)"); continue
        fit, val = train_test_split(tr, test_size=0.15, stratify=yn[tr], random_state=0)
        soft = torch.zeros(len(te), 2)
        for s in range(args.seeds):
            m, _ = R.train_one(X[fit], M[fit], y[fit], X[val], M[val], y[val],
                               seed=s, epochs=args.epochs, use_cw=True, n_num=n_num,
                               aux_w=0.0, arch=args.arch, binary=True, ysub_f=ysub[fit], d=args.dim,
                               n_sub=n_sub)
            soft += torch.softmax(R.predict(m, X[te], M[te]), 1)
        rel_soft[te] = (soft / args.seeds).numpy()
        rel[te] = rel_soft[te].argmax(1)
        acc = 100 * (rel[te] == yn[te]).mean()
        print(f"  held-out group {g}: n={len(te):3d} unsafe={int(yn[te].sum())} acc={acc:5.1f}%", flush=True)

    fit_mask = (rel != -1)
    print(f"\n=== {proto.upper()} ({args.arch}{'_d'+str(args.dim) if args.dim else ''}, "
          f"{int(fit_mask.sum())}/{len(yn)} clips scored, {len(folds)} held-out groups) ===")
    res = report_binary(yn, ysub, rel, rel_soft, args.arch, args.seeds,
                        protocol_tag=f'binary-{proto}-{args.arch}', sub_names=sub_names)

    if args.dataset == 'isafetybench':
        # Pre-registered R42 floor, recomputed on the SAME clips and the SAME folds as the encoder
        # (the gate itself used a 300-clip probe at T=8). Any accuracy claim on this corpus must be
        # printed beside this number.
        import isafetybench_dataset as ISB
        # Derive the floor's splits from the ENCODER's own `folds` (complement of each held-out
        # index), never by rebuilding them. Rebuilding via ISB.folds() ignored --split and silently
        # scored the shortcut on leave-source-out folds while the encoder ran on random ones, so the
        # printed "identical folds" margin compared two different protocols.
        all_idx = np.arange(len(yn))
        splits = [(np.setdiff1d(all_idx, te), te) for _, te in folds]
        cpred, cprob = ISB.count_rule_oof(X, M, y, splits)
        cba = float(np.mean([(cpred[yn == c] == c).mean() for c in (0, 1)])) * 100
        rba = res['balanced_accuracy']['value'] * 100
        print(f"  [R42 floor, identical folds] count-rule (persons/frame, objects/frame, mean conf) "
              f"balanced_acc={cba:.1f}%  |  encoder-minus-shortcut = {rba - cba:+.1f} pp")

    dsuf = f'_d{args.dim}' if args.arch == 'E1' and args.dim else ''
    # coord_variant is a confound (E1 defaults to T3, everything else to T1) -- only suffix the
    # filename when an explicit --coord_variant overrides that arch's implied default, so the
    # already-produced default-recipe OOF files are never silently clobbered by an ablation run.
    implied_default = 'T3' if args.arch == 'E1' else 'T1'
    csuf = f'_{coord}' if coord != implied_default else ''
    csuf += '' if args.ablate == 'none' else f'_{args.ablate}'
    rows = [{'clip_id': str(clip_ids[i]), 'y_true': int(yn[i]), 'prob_unsafe': float(rel_soft[i, 1]),
             'y_pred': int(rel[i]), 'fold_id': str(groups[i])} for i in range(len(yn))]
    json_path = f'{TOK}/enc_{args.tag}{args.arch}{dsuf}{csuf}_{proto}_headline-{args.dedupe}_oof.json'
    json.dump({'model': f'enc_{args.arch}{dsuf}{csuf}', 'protocol': proto, 'dedupe_policy': args.dedupe,
               'tag': args.tag, 'n': len(rows), 'rows': rows}, open(json_path, 'w'))
    # proto in the tensor name too, else a --split random run silently clobbers the
    # leave-source-out tensor of the same arch (they are different protocols, not variants).
    psuf = '' if proto.startswith('lso-') or proto.startswith('lco-') else f'_{proto}'
    pt_path = f'{TOK}/lco_oof_softmax_{args.tag}binary_{args.arch}{dsuf}{csuf}{psuf}_{args.dedupe}.pt'
    torch.save({'ens_softmax': torch.from_numpy(rel_soft), 'labels': y, 'sub_labels': ysub,
                'clip_ids': clip_ids, 'groups': groups, 'tag': args.tag, 'arch': args.arch,
                'dim': args.dim, 'coord_variant': coord, 'oof_pred': rel}, pt_path)
    print(f"  [saved LCO per-clip OOF] {json_path}\n  [saved LCO OOF tensor] {pt_path}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=1, help='independent k-fold re-splits')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--tag', default='yoloapp_')
    ap.add_argument('--dedupe', default='none', choices=['none', 'drop_all', 'keep_one', 'drop_conflict'],
                    help='drop clips with a byte-identical, differently-labelled twin')
    # --- binary Safe/Unsafe reframe additions (design-2026-09-16-binary-encoder-redesign.md, Tier 2) ---
    ap.add_argument('--binary', action='store_true', help='Safe(0)/Unsafe(1) task instead of the 4-class one')
    ap.add_argument('--arch', default='E0', choices=['E0', 'E1', 'E2', 'E3'], help='binary encoder (design §2)')
    ap.add_argument('--dim', type=int, default=None, help='E1 width override (64 primary, 96 capacity check)')
    ap.add_argument('--dedupe_task', default='unsafe4', choices=['unsafe4', 'binary'],
                    help='informational; --binary already forces task=binary in load_pooled_binary')
    ap.add_argument('--coord_variant', default=None, choices=['T1', 'T2', 'T3'],
                    help='token coordinate variant (design §3.5); default T3 for E1, T1 otherwise')
    ap.add_argument('--lco', action='store_true',
                    help='Tier 3 leave-camera-out (pre-registered decision rule) instead of k-fold; requires --binary')
    ap.add_argument('--scene_map', default='sweep_results/zone_prior/clip_scene_map_691.json',
                    help='LCO grouping JSON {clip_id: group} (KMeans(8) background-descriptor proxy)')
    ap.add_argument('--dataset', default='unsafenet', choices=['unsafenet', 'isafetybench'],
                    help='token corpus; isafetybench reads tokens_{tag}all.pt and groups by source video')
    ap.add_argument('--ablate', default='none', choices=list(ABL.MODES),
                    help='zero one channel group in the tokens; applied to the encoder AND the R42 '
                         'count-rule floor so both are denied the same cue')
    ap.add_argument('--split', default='group', choices=['group', 'random'],
                    help='iSafetyBench fold construction. group = leave-source-out (the only valid '
                         'protocol on this corpus). random = G12 diagnostic that deliberately leaks '
                         'source identity, to measure protocol inflation; never a headline number.')
    ap.add_argument('--permute_sources', type=int, default=None,
                    help='G19 control: replace real source ids with structure-matched placebo ones '
                         'at this seed (iSafetyBench, --split group only). Never a headline number.')
    ap.add_argument('--folds', type=int, default=10,
                    help='grouped folds for --dataset isafetybench (ignored by unsafenet, which uses '
                         'one fold per scene group)')
    args = ap.parse_args()
    if args.lco:
        assert args.binary, '--lco is only defined for the binary task'
        run_lco(args); return
    n_cls = 2 if args.binary else 4

    if args.binary:
        X, M, y, ysub, views, _clip_ids = load_pooled_binary(args.tag, args.dedupe)
        coord = args.coord_variant or ('T3' if args.arch == 'E1' else 'T1')
        X = R.apply_coord_variant(X, M, coord)
    else:
        X, M, y, views = load_pooled(args.tag, args.dedupe)
        ysub, coord = None, None
    yn = y.numpy(); n_num = X.shape[-1] - 1
    rel = np.full(len(yn), -1); scene = np.full(len(yn), -1)
    rel_soft = np.zeros((len(yn), n_cls), dtype=np.float32)  # pooled out-of-fold probabilities (leakage-free)
    diffs, n_tr, n_te, fold_rel, fold_scene = [], [], [], [], []
    splits = [(r, tr, te) for r in range(args.repeats)
              for tr, te in StratifiedKFold(n_splits=args.k, shuffle=True,
                                            random_state=r).split(np.zeros(len(yn)), yn)]
    if args.binary:
        params = count_params(build_encoder(args.arch, n_num=n_num, d=args.dim))
        print(f"[binary] arch={args.arch} d={args.dim or (64 if args.arch == 'E1' else '-')} coord={coord} "
              f"pooled {len(yn)} clips (unsafe {int(yn.sum())}) | feat={X.shape[-1]} n_num={n_num} | params={params:,}")
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
        soft = torch.zeros(len(te), n_cls)
        for s in range(args.seeds):
            m, _ = R.train_one(X[fit], M[fit], y[fit], X[val], M[val], y[val],
                               seed=s, epochs=args.epochs, use_cw=args.binary, n_num=n_num,
                               aux_w=0.0 if args.binary else 0.5, arch=args.arch, binary=args.binary,
                               ysub_f=ysub[fit] if args.binary else None, d=args.dim)
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
    se = np.std(fold_rel, ddof=1) / np.sqrt(len(fold_rel))
    corr_se = np.sqrt(np.var(fold_rel, ddof=1) * (1.0 / len(fold_rel) + np.mean(n_te) / np.mean(n_tr)))
    print(f"\n=== STRATIFIED {args.k}-FOLD x {args.repeats} REPEATS "
          f"({len(splits)} folds, pooled {len(yn)} clips) ===")
    print(f"  relation-Transformer: top-1 {np.mean(fold_rel):.1f}±{np.std(fold_rel):.1f}% | "
          f"pooled {int((rel==yn).sum())}/{len(yn)} | macro {macro_accuracy(yn, rel)*100:.1f}%")

    if args.binary:
        # Tier 2 screening scorecard (design §4, phase-06): eval_utils suite, explicit binary class map.
        report_binary(yn, ysub, rel, rel_soft, args.arch, args.seeds, protocol_tag=f'binary-kfold-{args.arch}')
    else:
        # richer multi-metric scorecard on the pooled out-of-fold predictions (additive)
        from sklearn.metrics import f1_score, cohen_kappa_score, matthews_corrcoef, precision_score, recall_score
        f1m = f1_score(yn, rel, average='macro') * 100
        kap = cohen_kappa_score(yn, rel); mcc = matthews_corrcoef(yn, rel)
        prec = precision_score(yn, rel, average='macro', zero_division=0) * 100
        rec = recall_score(yn, rel, average='macro', zero_division=0) * 100
        print(f"  scorecard: macro-F1 {f1m:.1f}% | macro-precision {prec:.1f}% | macro-recall {rec:.1f}% | "
              f"Cohen κ {kap:.3f} | MCC {mcc:.3f}")
    print(f"  fold-mean top-1 {np.mean(fold_rel):.1f}%  naive SE {se:.2f}  NB-corrected SE {corr_se:.2f} "
          f"(95% CI {np.mean(fold_rel)-1.96*corr_se:.1f}-{np.mean(fold_rel)+1.96*corr_se:.1f}%)")
    print(f"  scene-only baseline : top-1 {np.mean(fold_scene):.1f}±{np.std(fold_scene):.1f}%")
    print(f"  Nadeau–Bengio (relation vs scene-only): mean Δ={nb['mean_diff']:.1f}pp  t={nb['t']:.2f}  df={len(diffs)-1}  p={nb['p_value']:.4f}"
          f"  {'(significant)' if nb['p_value']<0.05 else '(n.s.)'}")
    if not args.binary:
        print("  NOTE: random k-fold has every camera in train, so the per-view-majority scene baseline is")
        print("        informative here. Under leave-camera-out it degenerates to a CONSTANT global-majority")
        print("        predictor (macro 25% by arithmetic) — the two baselines are different estimators, so the")
        print("        old '77.1% -> 16.9% collapse' contrast is WITHDRAWN. See brief.md S2; task A3b rebuilds it.")

    # dump leakage-free out-of-fold probs — a valid selection set for τ/T (each clip predicted by folds
    # that never saw it), disjoint from the blind 64-clip test. Consumed by run_operating_point.py --select_logits.
    # filename encodes variant + dedupe policy: never let one run silently overwrite another's OOF file
    # (the S6/S9 failure mode) — critical here since Tier 2 screens 5 archs x up to 2 widths.
    suffix = '' if args.dedupe == 'none' else f'_{args.dedupe}'
    if args.binary:
        dsuf = f'_d{args.dim}' if args.arch == 'E1' and args.dim else ''
        oof_path = f'{TOK}/kfold_oof_softmax_{args.tag}binary_{args.arch}{dsuf}{suffix}.pt'
        torch.save({'ens_softmax': torch.from_numpy(rel_soft), 'labels': y, 'sub_labels': ysub, 'tag': args.tag,
                    'arch': args.arch, 'dim': args.dim, 'coord_variant': coord, 'oof_pred': rel}, oof_path)
    else:
        oof_path = f'{TOK}/kfold_oof_softmax_{args.tag}{suffix}.pt'
        torch.save({'ens_softmax': torch.from_numpy(rel_soft), 'labels': y, 'tag': args.tag}, oof_path)
    print(f"  [saved OOF selection set] {oof_path}")


if __name__ == '__main__':
    main()
