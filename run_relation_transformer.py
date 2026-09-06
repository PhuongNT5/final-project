#!/usr/bin/env python3
"""Stage B — train the box-token spatial-relation Transformer on the 4-class unsafe subset.

Blind held-out protocol: a stratified val slice is carved from the 452 TRAIN clips; epoch and
seed are selected on val macro; the 64-clip TEST set is touched once per seed for reporting.
Trains on CPU (the MPS training bug corrupts Adam updates on macOS 14 — see reports/).

Usage: .venv/bin/python run_relation_transformer.py [--seeds 5] [--epochs 120]
"""
import argparse
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from model_relation_head import RelationTransformer, count_params
from eval_utils.metrics import top1, macro_accuracy, per_class_recall, per_class_fnr, confusion
from eval_utils.bootstrap_ci import bootstrap_ci
from eval_utils.paired_tests import two_proportion_z

ROOT = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(ROOT, 'sweep_results', 'relation_tokens')
NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']
DEVICE = 'cpu'   # never MPS for training on macOS 14 (addcmul_ bug)


def load(sp, tag=''):
    d = torch.load(os.path.join(TOK, f'tokens_{tag}{sp}.pt'), weights_only=False)
    return d['tokens'].float(), d['mask'].float(), d['labels'].long()


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def class_weights(y, k=4):
    cnt = torch.bincount(y, minlength=k).float()
    w = cnt.sum() / (k * cnt.clamp(min=1))
    return (w / w.mean()).to(DEVICE)


def run_epoch(model, X, M, y, opt, lossfn, bs=32, train=True, aux_w=0.0):
    idx = torch.randperm(len(y)) if train else torch.arange(len(y))
    model.train(train)
    tot = 0.0
    preds = torch.empty(len(y), dtype=torch.long)
    for i in range(0, len(y), bs):
        b = idx[i:i + bs]
        xb, mb, yb = X[b].to(DEVICE), M[b].to(DEVICE), y[b].to(DEVICE)
        with torch.set_grad_enabled(train):
            if train and aux_w > 0:                       # S1a: per-frame auxiliary behaviour loss
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


def train_one(Xf, Mf, yf, Xv, Mv, yv, seed, epochs, use_cw=True, n_num=9, aux_w=0.0):
    set_seed(seed)
    model = RelationTransformer(n_num=n_num).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    w = class_weights(yf) if use_cw else None
    lossfn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    best = {'val_macro': -1}
    for ep in range(epochs):
        run_epoch(model, Xf, Mf, yf, opt, lossfn, train=True, aux_w=aux_w); sched.step()
        with torch.no_grad():
            _, pv = run_epoch(model, Xv, Mv, yv, opt, lossfn, train=False)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--no_class_weights', action='store_true', help='disable inverse-freq class weighting')
    ap.add_argument('--tag', default='', help="token set: '' (v1) or 'v2_' (with relational features)")
    ap.add_argument('--save_logits', default=None, help='save seed-averaged test softmax to this path')
    ap.add_argument('--aux_w', type=float, default=0.0, help='S1a: per-frame behaviour aux-loss weight (e.g. 0.5)')
    ap.add_argument('--frame_gate', action='store_true', help='S1b: behaviour-confidence frame-gate + aggregation')
    ap.add_argument('--eval_test_tags', default='', help='F/G: comma alt test sets tokens_<tag><X>_test.pt (e.g. flip,jpeg,noise,blur,dark)')
    args = ap.parse_args()

    Xtr, Mtr, ytr = load('train', args.tag); Xte, Mte, yte = load('test', args.tag)
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
                     'softmax': torch.softmax(lg, 1)})
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

    lo, hi = bootstrap_ci(yte_np, blind['pred'], lambda a, b: top1(a, b)*100, n_boot=5000, seed=0)[1:3]
    print(f"blind top1 BCa 95% CI = [{lo:.1f}, {hi:.1f}]")
    pcr = per_class_recall(yte_np, blind['pred']); fnr = per_class_fnr(yte_np, blind['pred'])
    print("per-class (blind):")
    for c in range(4):
        n = int((yte_np == c).sum())
        print(f"  {NAMES[c]:14s} n={n:2d}  recall={pcr[c]*100:5.1f}%  FNR={fnr[c]*100:5.1f}%")
    print("confusion (rows=true, cols=pred):\n", confusion(yte_np, blind['pred'], 4))
    print("\nPaper: no-bbox 57/64 (85.78% macro) | full-bbox 62/64 (95.81% macro)")
    z = two_proportion_z(blind['test_top1'], 64, 57, 64)
    print(f"blind vs no-bbox baseline: Δ={z['delta_pp']:+.1f}pp z={z['z']:+.2f} p={z['p_value']:.3f}")

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
        gz = two_proportion_z(gens_c, 64, 57, 64)
        print(f"  ensemble-gated vs no-bbox baseline (57/64): Δ={gz['delta_pp']:+.1f}pp z={gz['z']:+.2f} p={gz['p_value']:.3f}")

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


if __name__ == '__main__':
    main()
