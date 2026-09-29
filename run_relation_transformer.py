#!/usr/bin/env python3
"""Stage B — train the box-token spatial-relation Transformer.

Two task framings share this file:
  - Legacy 4-class unsafe-behaviour path (`--binary` unset) — UNCHANGED, byte-identical to the
    pre-2026-09-17 version. Blind held-out protocol: a stratified val slice is carved from the
    452 TRAIN clips; epoch and seed are selected on val macro; the 64-clip TEST set is touched
    once per seed for reporting.
  - Binary Safe(0)/Unsafe(1) path (`--binary`) — design-2026-09-16-binary-encoder-redesign.md.
    Selects among E0/E1/E2/E3 encoders (model_binary_encoders.py) under one identical recipe:
    2-logit main + 8-way sub-class aux (lambda=0.3) + per-frame binary aux (lambda=0.5), label
    smoothing 0.05, class weights ON. Val is carved from TRAIN stratified on (label, view) per
    brief S4h.1. Metrics via eval_utils.suite.evaluate_suite (class_names=['Safe','Unsafe'],
    unsafe_classes=[1]). Test touches are governed by reports/test-touch-log.md — this script
    does not enforce the "two touches only" budget, the caller must.

Trains on CPU (the MPS training bug corrupts Adam updates on macOS 14 — see reports/).

Usage: .venv/bin/python run_relation_transformer.py [--seeds 5] [--epochs 120]
       .venv/bin/python run_relation_transformer.py --binary --arch E1 --dim 64 --seeds 5 \
           --dedupe drop_conflict --dedupe_task binary
"""
import argparse
import copy
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from model_relation_head import RelationTransformer, count_params
from model_binary_encoders import build_encoder
from unsafenet import dedupe
from eval_utils.metrics import top1, macro_accuracy, per_class_recall, per_class_fnr, confusion
from eval_utils.bootstrap_ci import bootstrap_ci
from eval_utils.paired_tests import paired_vs_aggregate_baseline, mcnemar_exact
from eval_utils.suite import evaluate_suite

ROOT = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(ROOT, 'sweep_results', 'relation_tokens')
NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']
NAMES_BIN = ['Safe', 'Unsafe']   # index 1 = Unsafe = positive (design convention, repo-wide)
DEVICE = 'cpu'   # never MPS for training on macOS 14 (addcmul_ bug)


def load(sp, tag='', dedupe_policy='none'):
    """dedupe_policy drops TRAIN clips that are byte-identical to a differently-labelled clip
    (unsafenet/dedupe.py). The 64-clip TEST split is never filtered — it is the paper's frozen
    evaluation set."""
    d = torch.load(os.path.join(TOK, f'tokens_{tag}{sp}.pt'), weights_only=False)
    X, M, y = d['tokens'].float(), d['mask'].float(), d['labels'].long()
    if sp == 'train' and dedupe_policy != 'none':
        keep = dedupe.keep_mask(d['clip_ids'], 'train', dedupe_policy)
        print(f"[dedupe:{dedupe_policy}] train: dropped {int((~keep).sum())} of {len(keep)} clips")
        ki = torch.from_numpy(np.where(keep)[0])
        X, M, y = X[ki], M[ki], y[ki]
    return X, M, y


def load_binary(sp, tag='bin8_', dedupe_policy='none'):
    """Binary loader. `labels` in the token file is the SOURCE sub-class folder id (0-7);
    binary label = 1 (Unsafe) for sub-class 0-3, 0 (Safe) for 4-7 (design §0, task pre-reg
    2026-09-17). dedupe uses task='binary' (unsafenet/dedupe.py + duplicate_audit_binary.json).
    Returns (tokens, mask, y_binary, y_subclass, views, clip_ids)."""
    d = torch.load(os.path.join(TOK, f'tokens_{tag}{sp}.pt'), weights_only=False)
    X, M, ysub = d['tokens'].float(), d['mask'].float(), d['labels'].long()
    views = np.array([str(v) for v in d['views']])
    clip_ids = list(d['clip_ids'])
    if dedupe_policy != 'none' and dedupe_policy != 'keep':
        keep = dedupe.keep_mask(clip_ids, sp, dedupe_policy, task='binary')
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"[dedupe:binary/{dedupe_policy}] {sp}: dropped {n_drop} of {len(keep)} clips "
                  f"({[c for c, k in zip(clip_ids, keep) if not k]})")
        ki = np.where(keep)[0]
        X, M, ysub, views = X[ki], M[ki], ysub[ki], views[ki]
        clip_ids = [clip_ids[i] for i in ki]
    ybin = (ysub < 4).long()
    return X, M, ybin, ysub, views, clip_ids


def apply_coord_variant(X, M, variant):
    """Token variants (design §3.5), applied at LOAD time (no re-tokenization):
    T1 = unchanged (absolute xc,yc — shortcut-prone upper-bound reference).
    T2 = xc,yc dropped entirely (feat-74 -> feat-72, matches yolo26L1_ layout).
    T3 = xc,yc replaced by per-frame-relative coords (xc/yc minus the mean over valid boxes in
    that frame), zeroed on padded slots — E1's default, translation-invariant camera fingerprint
    removed while inter-entity geometry is preserved."""
    if variant in (None, 'T1'):
        return X
    if variant == 'T2':
        return torch.cat([X[..., :1], X[..., 3:]], dim=-1)
    if variant == 'T3':
        X = X.clone()
        cnt = M.sum(-1, keepdim=True).clamp(min=1)
        mx = (X[..., 1] * M).sum(-1, keepdim=True) / cnt
        my = (X[..., 2] * M).sum(-1, keepdim=True) / cnt
        Mb = M.bool()
        X[..., 1] = torch.where(Mb, X[..., 1] - mx, torch.zeros_like(X[..., 1]))
        X[..., 2] = torch.where(Mb, X[..., 2] - my, torch.zeros_like(X[..., 2]))
        return X
    raise ValueError(f'unknown coord_variant {variant!r}, expected T1/T2/T3')


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def class_weights(y, k=4):
    cnt = torch.bincount(y, minlength=k).float()
    w = cnt.sum() / (k * cnt.clamp(min=1))
    return (w / w.mean()).to(DEVICE)


def run_epoch(model, X, M, y, opt, lossfn, bs=32, train=True, aux_w=0.0,
              binary=False, ysub=None, lambda_sub=0.3, lambda_frame=0.5):
    idx = torch.randperm(len(y)) if train else torch.arange(len(y))
    model.train(train)
    tot = 0.0
    preds = torch.empty(len(y), dtype=torch.long)
    for i in range(0, len(y), bs):
        b = idx[i:i + bs]
        xb, mb, yb = X[b].to(DEVICE), M[b].to(DEVICE), y[b].to(DEVICE)
        with torch.set_grad_enabled(train):
            if train and binary:                    # design §3.2: main + 8-way sub + frame aux, always
                logits, sub_logits, flog, fvalid = model(xb, mb, return_frame=True, return_sub=True)
                loss = lossfn(logits, yb)
                ysb = ysub[b].to(DEVICE)
                loss = loss + lambda_sub * F.cross_entropy(sub_logits, ysb, label_smoothing=0.05)
                Bn, T, _ = flog.shape
                ybT = yb.view(Bn, 1).expand(Bn, T)
                fl, yl = flog[fvalid], ybT[fvalid]
                if fl.numel():
                    loss = loss + lambda_frame * F.cross_entropy(fl, yl, label_smoothing=0.05)
            elif train and aux_w > 0:                # S1a: legacy 4-class per-frame auxiliary loss
                logits, flog, fvalid = model(xb, mb, return_frame=True)
                loss = lossfn(logits, yb)
                Bn, T, _ = flog.shape
                ybT = yb.view(Bn, 1).expand(Bn, T)
                fl, yl = flog[fvalid], ybT[fvalid]
                if fl.numel():
                    loss = loss + aux_w * F.cross_entropy(fl, yl, label_smoothing=0.05)
            else:
                logits = model(xb, mb)
                loss = lossfn(logits, yb)
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        tot += float(loss.detach()) * len(b)
        preds[b] = logits.argmax(1).cpu()
    return tot / len(y), preds


def train_one(Xf, Mf, yf, Xv, Mv, yv, seed, epochs, use_cw=True, n_num=9, aux_w=0.0,
              arch='E0', binary=False, ysub_f=None, d=None, n_sub=8):
    """`n_sub` sizes the auxiliary sub-class head. 8 for Unsafe-Net's taxonomy; corpora without one
    (iSafetyBench) pass n_sub=2 with ysub_f=the binary label, which makes the aux head a mild
    reweighting of the main objective rather than a head predicting classes that never occur."""
    set_seed(seed)
    if binary:
        model = build_encoder(arch, n_num=n_num, n_cls=2, n_sub=n_sub, d=d).to(DEVICE)
    else:
        model = RelationTransformer(n_num=n_num).to(DEVICE)   # unchanged ctor -> unchanged param count
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    w = class_weights(yf, k=2 if binary else 4) if use_cw else None
    lossfn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    best = {'val_macro': -1}
    for ep in range(epochs):
        run_epoch(model, Xf, Mf, yf, opt, lossfn, train=True, aux_w=aux_w,
                  binary=binary, ysub=ysub_f); sched.step()
        with torch.no_grad():
            _, pv = run_epoch(model, Xv, Mv, yv, opt, lossfn, train=False, binary=binary)
        vm = macro_accuracy(yv.numpy(), pv.numpy()) * 100
        vt = top1(yv.numpy(), pv.numpy()) * 100
        if vm > best['val_macro']:
            best = {'val_macro': vm, 'val_top1': vt, 'epoch': ep, 'state': copy.deepcopy(model.state_dict())}
    model.load_state_dict(best['state'])
    return model, best


@torch.no_grad()
def predict(model, X, M):
    model.eval()
    out = [model(X[i:i+32].to(DEVICE), M[i:i+32].to(DEVICE)) for i in range(0, len(X), 32)]
    return torch.cat(out).cpu()


@torch.no_grad()
def predict_frames(model, X, M):
    """Per-frame behaviour softmax + frame-valid mask (for S1b behaviour-gate)."""
    model.eval()
    fl, fv = [], []
    for i in range(0, len(X), 32):
        _, f, v = model(X[i:i+32].to(DEVICE), M[i:i+32].to(DEVICE), return_frame=True)
        fl.append(torch.softmax(f, -1).cpu()); fv.append(v.cpu())
    return torch.cat(fl), torch.cat(fv)


def gated_predict(frame_soft, frame_valid, tau):
    """Paper's YOLO-v4 mechanism: keep frames whose per-frame behaviour confidence ≥ τ (drop
    irrelevant frames), aggregate the kept per-frame softmax → clip prediction. Falls back to all
    valid frames if none pass. Returns (pred, aggregated_softmax, mean_kept_fraction)."""
    conf = frame_soft.max(-1).values
    keep = frame_valid.bool() & (conf >= tau)
    empty = keep.sum(1) == 0
    keep[empty] = frame_valid[empty].bool()
    w = keep.float().unsqueeze(-1)
    agg = (frame_soft * w).sum(1)
    agg = agg / agg.sum(-1, keepdim=True).clamp(min=1e-6)
    kept_frac = (keep.sum(1).float() / frame_valid.sum(1).clamp(min=1)).mean().item()
    return agg.argmax(1).numpy(), agg, kept_frac


def main_binary(args):
    tag = args.tag or 'bin8_'
    coord = args.coord_variant or ('T3' if args.arch == 'E1' else 'T1')
    Xtr, Mtr, ytr, ysub_tr, views_tr, _ = load_binary('train', tag, args.dedupe)
    Xte, Mte, yte, ysub_te, views_te, _ = load_binary('test', tag, 'none')
    Xtr = apply_coord_variant(Xtr, Mtr, coord); Xte = apply_coord_variant(Xte, Mte, coord)
    n_num = Xtr.shape[-1] - 1
    params = count_params(build_encoder(args.arch, n_num=n_num, d=args.dim))
    print(f"[binary] arch={args.arch} d={args.dim or (64 if args.arch == 'E1' else '-')} coord={coord} "
          f"train {len(ytr)} (unsafe {int(ytr.sum())}) | test {len(yte)} (unsafe {int(yte.sum())}) "
          f"| feat={Xtr.shape[-1]} n_num={n_num} | params={params:,}")

    strat_key = [f'{int(l)}_{v}' for l, v in zip(ytr.numpy(), views_tr)]
    try:
        fit, val = train_test_split(np.arange(len(ytr)), test_size=0.18, stratify=strat_key, random_state=0)
    except ValueError as e:
        print(f"[val split] (label,view) stratify infeasible ({e}) -> falling back to label-only stratify")
        fit, val = train_test_split(np.arange(len(ytr)), test_size=0.18, stratify=ytr.numpy(), random_state=0)
    Xf, Mf, yf, ysubf = Xtr[fit], Mtr[fit], ytr[fit], ysub_tr[fit]
    Xv, Mv, yv = Xtr[val], Mtr[val], ytr[val]
    yte_np = yte.numpy()

    rows, prob_sum = [], torch.zeros(len(yte), 2)
    for s in range(args.seeds):
        model, best = train_one(Xf, Mf, yf, Xv, Mv, yv, seed=s, epochs=args.epochs,
                                 use_cw=not args.no_class_weights, n_num=n_num, arch=args.arch,
                                 binary=True, ysub_f=ysubf, d=args.dim)
        lg = predict(model, Xte, Mte)
        prob = torch.softmax(lg, 1); prob_sum += prob
        pred = lg.argmax(1).numpy()
        c = int((pred == yte_np).sum())
        rows.append({'seed': s, 'val_macro': best['val_macro'], 'epoch': best['epoch'],
                     'test_top1': c, 'test_macro': macro_accuracy(yte_np, pred) * 100,
                     'pred': pred, 'softmax': prob, 'state': best['state']})
        print(f"  seed{s}: val_macro={best['val_macro']:.1f}@e{best['epoch']} -> test {c}/{len(yte_np)} "
              f"top1={100*c/len(yte_np):.1f} macro={rows[-1]['test_macro']:.1f}")

    blind = max(rows, key=lambda r: r['val_macro'])
    ens_prob = (prob_sum / args.seeds).numpy()
    res = evaluate_suite(yte_np, blind['softmax'].numpy(), class_names=NAMES_BIN,
                          protocol_tag=f'binary-blind-{args.arch}', n_seeds=args.seeds, unsafe_classes=[1])
    print(f"\n=== RESULTS (binary Safe/Unsafe, arch={args.arch}, {len(yte_np)}-clip test) ===")
    print(f"BLIND (val-selected seed {blind['seed']}): balanced_acc={res['balanced_accuracy']['value']*100:.1f}% "
          f"macro_f1={res['macro_f1']['value']*100:.1f}% top1={res['top1']['value']*100:.1f}% "
          f"unsafe_recall={res['per_class']['Unsafe']['recall']*100:.1f}% "
          f"unsafe_precision={res['per_class']['Unsafe']['precision']*100:.1f}%")
    print(f"  PR-AUC(Unsafe)={res['pr_auc_ovr']['per_class'][1]:.3f} ROC-AUC(Unsafe)={res['roc_auc_ovr']['per_class'][1]:.3f}")
    ens_pred = ens_prob.argmax(1)
    mbe = mcnemar_exact(blind['pred'] == yte_np, ens_pred == yte_np)
    print(f"McNemar blind vs ensemble: b={mbe['b']} c={mbe['c']} p={mbe['p_value']:.4f}")

    if args.save_logits:
        torch.save({'ens_softmax': torch.from_numpy(ens_prob), 'blind_pred': torch.tensor(blind['pred']),
                    'labels': yte, 'sub_labels': ysub_te, 'blind_top1': blind['test_top1'],
                    'blind_softmax': blind['softmax'], 'blind_seed': blind['seed'],
                    'per_seed_softmax': torch.stack([r['softmax'] for r in rows]),
                    'arch': args.arch, 'dim': args.dim, 'coord_variant': coord}, args.save_logits)
        print(f"[saved] binary seed-averaged + blind softmax -> {args.save_logits}")

    if args.save_model:
        torch.save({'state_dict': blind['state'], 'ctor': {'arch': args.arch, 'n_num': n_num, 'd': args.dim},
                    'blind_seed': blind['seed'], 'blind_top1': blind['test_top1'],
                    'blind_softmax': blind['softmax'], 'labels': yte, 'argv': ' '.join(sys.argv),
                    'tag': tag, 'coord_variant': coord, 'seeds': args.seeds, 'epochs': args.epochs},
                   args.save_model)
        print(f"[saved] binary blind-seed weights -> {args.save_model}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--dedupe', default='none', choices=['none', 'drop_all', 'keep_one', 'drop_conflict'],
                    help='drop TRAIN clips with a byte-identical, differently-labelled twin')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--no_class_weights', action='store_true', help='disable inverse-freq class weighting')
    ap.add_argument('--tag', default='', help="token set: '' (v1), 'v2_', or 'bin8_' (binary)")
    ap.add_argument('--save_logits', default=None, help='save seed-averaged test softmax to this path')
    # The blind headline model's WEIGHTS were never persisted - only its softmax - so P3 could not be
    # exported to ONNX/TensorRT for the Jetson benchmark (deployment/JETSON-DEPLOYMENT.md Lane B).
    # --save_model writes the val-selected blind seed's state_dict plus everything needed to rebuild it.
    ap.add_argument('--save_model', default=None,
                    help='save the val-selected blind seed state_dict + ctor config to this path')
    ap.add_argument('--aux_w', type=float, default=0.0, help='S1a: per-frame behaviour aux-loss weight (e.g. 0.5)')
    ap.add_argument('--frame_gate', action='store_true', help='S1b: behaviour-confidence frame-gate + aggregation')
    ap.add_argument('--eval_test_tags', default='', help='F/G: comma alt test sets tokens_<tag><X>_test.pt (e.g. flip,jpeg,noise,blur,dark)')
    # --- binary Safe/Unsafe reframe additions (design-2026-09-16-binary-encoder-redesign.md) ---
    ap.add_argument('--binary', action='store_true', help='Safe(0)/Unsafe(1) task instead of the 4-class one')
    ap.add_argument('--arch', default='E0', choices=['E0', 'E1', 'E2', 'E3'], help='binary encoder (design §2)')
    ap.add_argument('--dim', type=int, default=None, help='E1 width override (64 primary, 96 capacity check)')
    ap.add_argument('--dedupe_task', default='unsafe4', choices=['unsafe4', 'binary'],
                    help='which dedupe audit to use (informational; --binary already forces task=binary)')
    ap.add_argument('--coord_variant', default=None, choices=['T1', 'T2', 'T3'],
                    help='token coordinate variant (design §3.5); default T3 for E1, T1 otherwise')
    args = ap.parse_args()

    if args.binary:
        main_binary(args)
        return

    # ---- everything below is the ORIGINAL 4-class path, unchanged ----
    Xtr, Mtr, ytr = load('train', args.tag, args.dedupe); Xte, Mte, yte = load('test', args.tag)
    n_num = Xtr.shape[-1] - 1
    print(f"train {len(ytr)} clips | test {len(yte)} clips | tag='{args.tag}' feat={Xtr.shape[-1]} (n_num={n_num}) | params ~{count_params(RelationTransformer(n_num=n_num)):,}")
    fit, val = train_test_split(np.arange(len(ytr)), test_size=0.18, stratify=ytr.numpy(), random_state=0)
    Xf, Mf, yf = Xtr[fit], Mtr[fit], ytr[fit]
    Xv, Mv, yv = Xtr[val], Mtr[val], ytr[val]
    yte_np = yte.numpy()

    rows, logits_sum = [], torch.zeros(len(yte), 4)
    g_rows, g_logits_sum = [], torch.zeros(len(yte), 4)
    TAUS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alts, alt_sum = {}, {}                                    # F/G: alternate (corrupted/flipped) test sets
    for x in filter(None, args.eval_test_tags.split(',')):
        dd = torch.load(os.path.join(TOK, f'tokens_{args.tag}{x}_test.pt'), weights_only=False)
        alts[x] = (dd['tokens'].float(), dd['mask'].float()); alt_sum[x] = torch.zeros(len(yte), 4)
    for s in range(args.seeds):
        model, best = train_one(Xf, Mf, yf, Xv, Mv, yv, seed=s, epochs=args.epochs, use_cw=not args.no_class_weights, n_num=n_num, aux_w=args.aux_w)
        lg = predict(model, Xte, Mte); logits_sum += torch.softmax(lg, 1)
        for x, (Xx, Mx) in alts.items():
            alt_sum[x] += torch.softmax(predict(model, Xx, Mx), 1)
        pred = lg.argmax(1).numpy()
        c = int((pred == yte_np).sum())
        rows.append({'seed': s, 'val_macro': best['val_macro'], 'epoch': best['epoch'],
                     'test_top1': c, 'test_macro': macro_accuracy(yte_np, pred) * 100, 'pred': pred,
                     'softmax': torch.softmax(lg, 1), 'state': best['state']})
        print(f"  seed{s}: val_macro={best['val_macro']:.1f}@e{best['epoch']} → test {c}/64 "
              f"top1={100*c/64:.1f} macro={rows[-1]['test_macro']:.1f}")
        if args.frame_gate:
            fsv, fvv = predict_frames(model, Xv, Mv)
            fst, fvt = predict_frames(model, Xte, Mte)
            btau, bvm = 0.0, -1                                   # select τ on VAL (blind)
            for tau in TAUS:
                vp, _, _ = gated_predict(fsv, fvv, tau)
                vm = macro_accuracy(yv.numpy(), vp) * 100
                if vm > bvm:
                    bvm, btau = vm, tau
            tp, tagg, kf = gated_predict(fst, fvt, btau)
            g_logits_sum += tagg
            gc = int((tp == yte_np).sum())
            g_rows.append({'seed': s, 'val_macro': bvm, 'tau': btau, 'kept_frac': kf,
                           'test_top1': gc, 'test_macro': macro_accuracy(yte_np, tp) * 100, 'pred': tp})
            print(f"    +gate τ={btau}: test {gc}/64 top1={100*gc/64:.1f} macro={g_rows[-1]['test_macro']:.1f} kept_frac={kf:.2f}")

    # BLIND selection: pick seed by best VAL macro (never test)
    blind = max(rows, key=lambda r: r['val_macro'])
    ens_pred = logits_sum.argmax(1).numpy()
    ens_c = int((ens_pred == yte_np).sum())

    print("\n=== RESULTS (4-class unsafe, 64-clip test) ===")
    t1_mean = np.mean([r['test_top1'] for r in rows]); t1_std = np.std([r['test_top1'] for r in rows])
    mac_mean = np.mean([r['test_macro'] for r in rows])
    print(f"across {args.seeds} seeds: test top1 = {t1_mean:.1f}±{t1_std:.1f}/64 | macro = {mac_mean:.1f}%")
    print(f"BLIND (val-selected seed {blind['seed']}): {blind['test_top1']}/64 = {100*blind['test_top1']/64:.1f}% top1, macro {blind['test_macro']:.1f}%")
    print(f"ensemble (test-selected caveat): {ens_c}/64 = {100*ens_c/64:.1f}% top1, macro {macro_accuracy(yte_np, ens_pred)*100:.1f}%")
    # own-vs-own, same 64 clips, per-clip preds already in memory -> exact McNemar (not a bound).
    mbe = mcnemar_exact(blind['pred'] == yte_np, ens_pred == yte_np)
    print(f"McNemar blind vs ensemble: b={mbe['b']} c={mbe['c']} p={mbe['p_value']:.4f} "
          f"diff={mbe['diff_accuracy_pp']:+.2f}pp CI95=[{mbe['diff_accuracy_ci95_pp'][0]:+.1f},"
          f"{mbe['diff_accuracy_ci95_pp'][1]:+.1f}]pp")

    lo, hi = bootstrap_ci(yte_np, blind['pred'], lambda a, b: top1(a, b)*100, n_boot=5000, seed=0)[1:3]
    print(f"blind top1 BCa 95% CI = [{lo:.1f}, {hi:.1f}]")
    pcr = per_class_recall(yte_np, blind['pred']); fnr = per_class_fnr(yte_np, blind['pred'])
    print("per-class (blind):")
    for c in range(4):
        n = int((yte_np == c).sum())
        print(f"  {NAMES[c]:14s} n={n:2d}  recall={pcr[c]*100:5.1f}%  FNR={fnr[c]*100:5.1f}%")
    print("confusion (rows=true, cols=pred):\n", confusion(yte_np, blind['pred'], 4))
    print("\nPaper: no-bbox 57/64 (85.78% macro) | full-bbox 62/64 (95.81% macro)")
    # Paper per-clip predictions unpublished -> paired table can't be built -> McNemar BOUND is the
    # valid stat, not two_proportion_z (wrongly assumes 2 independent n=64 samples; same clips).
    # See review-2026-09-13-eval-statistics.md finding #1.
    z = paired_vs_aggregate_baseline(64, blind['test_top1'], 57, label='paper no-bbox (57/64)')
    print(f"blind vs no-bbox baseline: Δ={z['delta_pp']:+.1f}pp McNemar bound "
          f"p∈[{z['min_p_over_splits']:.4f},{z['max_p_over_splits']:.4f}] "
          f"(diagnostic-only two_proportion_z p={z['diagnostic_two_proportion_z']['p_value']:.3f}, invalid independence)")

    if args.frame_gate and g_rows:
        gb = max(g_rows, key=lambda r: r['val_macro'])          # val-selected τ+seed
        gens_pred = g_logits_sum.argmax(1).numpy(); gens_c = int((gens_pred == yte_np).sum())
        gt1 = np.mean([r['test_top1'] for r in g_rows]); gt1s = np.std([r['test_top1'] for r in g_rows])
        kf = np.mean([r['kept_frac'] for r in g_rows])
        print("\n=== S1b BEHAVIOUR FRAME-GATE (paper's YOLO-v4 mechanism) ===")
        print(f"  kept_frac mean = {kf:.2f}  → reduced-duration proxy (paper target 1.87s / 7.72s ≈ 0.24)")
        print(f"  across {args.seeds} seeds (gated): top1 = {gt1:.1f}±{gt1s:.1f}/64 | macro = {np.mean([r['test_macro'] for r in g_rows]):.1f}%")
        print(f"  BLIND gated (val-selected seed{gb['seed']} τ={gb['tau']}): {gb['test_top1']}/64 = {100*gb['test_top1']/64:.1f}% top1, macro {gb['test_macro']:.1f}%")
        print(f"  ensemble gated: {gens_c}/64 = {100*gens_c/64:.1f}% top1, macro {macro_accuracy(yte_np, gens_pred)*100:.1f}%")
        gpcr = per_class_recall(yte_np, gb['pred'])
        print("  per-class (blind gated):", {NAMES[k]: round(gpcr[k]*100, 1) for k in range(4)})
        print("  confusion (blind gated):\n", confusion(yte_np, gb['pred'], 4))
        gz = paired_vs_aggregate_baseline(64, gens_c, 57, label='paper no-bbox (57/64)')
        print(f"  ensemble-gated vs no-bbox baseline (57/64): Δ={gz['delta_pp']:+.1f}pp McNemar bound "
              f"p∈[{gz['min_p_over_splits']:.4f},{gz['max_p_over_splits']:.4f}] "
              f"(diagnostic-only two_proportion_z p={gz['diagnostic_two_proportion_z']['p_value']:.3f}, invalid independence)")

    if alts:
        print("\n=== F/G — robustness / TTA (seed-ensemble) ===")
        print(f"  clean   : {ens_c}/64 = {100*ens_c/64:.1f}%")
        for x in alts:
            p = alt_sum[x].argmax(1).numpy(); cc = int((p == yte_np).sum())
            print(f"  {x:8s}: {cc}/64 = {100*cc/64:.1f}%  (Δ vs clean {cc-ens_c:+d})")
        if 'flip' in alts:                                   # TTA = mean(clean + flip), label-preserving
            tc = int((((logits_sum + alt_sum['flip']) / 2).argmax(1).numpy() == yte_np).sum())
            print(f"  TTA(clean+flip): {tc}/64 = {100*tc/64:.1f}%  (Δ vs clean {tc-ens_c:+d})")

    if args.save_logits:
        # blind_softmax is the val-selected SINGLE seed's own probabilities. Without it, every
        # calibration / conformal / selective-risk figure characterises the seed-ENSEMBLE decision
        # rule (a different predictor with a different top-1) rather than the blind headline.
        torch.save({'ens_softmax': logits_sum / args.seeds, 'blind_pred': torch.tensor(blind['pred']),
                    'labels': yte, 'blind_top1': blind['test_top1'],
                    'blind_softmax': blind['softmax'], 'blind_seed': blind['seed'],
                    'per_seed_softmax': torch.stack([r['softmax'] for r in rows])}, args.save_logits)
        print(f"[saved] seed-averaged softmax + blind-seed softmax → {args.save_logits}")

    if args.save_model:
        # blind_softmax travels with the weights so a reload can be verified bit-exact against the
        # run that produced the headline, rather than trusted.
        torch.save({'state_dict': blind['state'],
                    'ctor': {'n_num': n_num},          # every other ctor arg is a RelationTransformer default
                    'blind_seed': blind['seed'], 'blind_top1': blind['test_top1'],
                    'blind_softmax': blind['softmax'], 'labels': yte,
                    'argv': ' '.join(sys.argv), 'tag': args.tag,
                    'seeds': args.seeds, 'epochs': args.epochs, 'aux_w': args.aux_w},
                   args.save_model)
        print(f"[saved] blind-seed weights (seed {blind['seed']}, {blind['test_top1']}/64) → {args.save_model}")


if __name__ == '__main__':
    main()
