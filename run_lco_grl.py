#!/usr/bin/env python3
"""Leave-Camera-Out eval of a DANN/GRL relation-Transformer — the OBJECTIVE-axis lever.

Every feature lever tried (zones-detector, L2 homography, L5 proximity, motion) failed the same way:
scene-coupled features leak camera identity (k-fold up / LCO down) or shift WalkwayViol priors. GRL attacks
the LCO gap from the other side — it does not add features, it changes the training objective so the shared
representation is CAMERA-INVARIANT.

Mechanism (Ganin & Lempitsky 2015, DANN): a domain (camera-group) classifier is attached to the CLS
representation through a gradient-reversal layer. The domain head learns to predict which training camera a
clip came from; the reversal makes the backbone UNlearn that, so the representation keeps behaviour signal
while discarding camera-identifiable structure. The hope for LCO: invariance across the N-1 training cameras
transfers to the held-out camera. Domain labels only exist for training cameras (the held-out one is unseen),
so the adversary is trained across training groups exactly as the LCO protocol allows — no test leakage.

Champion reference (run_lco_eval.py, yolo26L1_, identical protocol): LCO macro 82.0 / k-fold 84.7.
Keep iff LCO macro rises with no leak (k-fold up / LCO down = reject).

Usage: .venv/bin/python run_lco_grl.py --tag yolo26L1_ --scene_map sweep_results/zone_prior/clip_scene_map.json \
           --seeds 3 --epochs 60 --grl_lambda 1.0
"""
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import run_relation_transformer as R
from run_lco_eval import load_pooled
from model_relation_head import RelationTransformer
from eval_utils.metrics import macro_accuracy, per_class_recall, confusion

NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']
DEVICE = 'cpu'


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


def grad_reverse(x, lamb):
    return GradReverse.apply(x, lamb)


def dann_lambda(ep, epochs, lam_max, gamma=10.0):
    """Standard DANN ramp: 0 -> lam_max over training, so the adversary engages only once the backbone
    has some behaviour signal (avoids destroying the representation at init)."""
    p = ep / max(1, epochs - 1)
    return lam_max * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)


def make_domain_head(n_dom, d=128):
    return nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, n_dom))


def train_one_grl(Xf, Mf, yf, df, Xv, Mv, yv, seed, epochs, n_num, n_dom, aux_w, lam_max, bs=32):
    R.set_seed(seed)
    model = RelationTransformer(n_num=n_num).to(DEVICE)
    dom = make_domain_head(n_dom).to(DEVICE)
    opt = torch.optim.AdamW(list(model.parameters()) + list(dom.parameters()), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    beh_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
    dom_loss = nn.CrossEntropyLoss()
    best = {'val_macro': -1, 'state': copy.deepcopy(model.state_dict())}
    for ep in range(epochs):
        lamb = float(dann_lambda(ep, epochs, lam_max))
        model.train(); dom.train()
        idx = torch.randperm(len(yf))
        for i in range(0, len(yf), bs):
            b = idx[i:i + bs]
            xb, mb, yb, db = Xf[b].to(DEVICE), Mf[b].to(DEVICE), yf[b].to(DEVICE), df[b].to(DEVICE)
            if aux_w > 0:
                logits, flog, fvalid, rep = model(xb, mb, return_frame=True, return_repr=True)
                loss = beh_loss(logits, yb)
                Bn, T, _ = flog.shape
                fl, yl = flog[fvalid], yb.view(Bn, 1).expand(Bn, T)[fvalid]
                if fl.numel():
                    loss = loss + aux_w * F.cross_entropy(fl, yl, label_smoothing=0.05)
            else:
                logits, rep = model(xb, mb, return_repr=True)
                loss = beh_loss(logits, yb)
            loss = loss + dom_loss(dom(grad_reverse(rep, lamb)), db)   # DANN adversary
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(dom.parameters()), 5.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = R.predict(model, Xv, Mv).argmax(1).numpy()
        vm = macro_accuracy(yv.numpy(), pv) * 100
        if vm > best['val_macro']:
            best = {'val_macro': vm, 'epoch': ep, 'state': copy.deepcopy(model.state_dict())}
    model.load_state_dict(best['state'])
    return model, best


def fold_grl(Xtr, Mtr, ytr, dom, Xte, Mte, n_num, n_dom, seeds, epochs, aux_w, lambdas):
    """Per-fold GRL with lambda selected on the fold's OWN training-view val slice (never the held-out
    camera). For each lambda: seed-ensemble the test softmax and record mean val macro; return the test
    softmax of the lambda with the best val macro. This treats lambda like epoch — a hyperparameter tuned
    blind — so the reported LCO number is not test-tuned."""
    fit, val = train_test_split(np.arange(len(ytr)), test_size=0.15, stratify=ytr.numpy(), random_state=0)
    best = {'val': -1, 'soft': None, 'lam': None}
    for lam in lambdas:
        soft = torch.zeros(len(Xte), 4)
        vms = []
        for s in range(seeds):
            model, bi = train_one_grl(Xtr[fit], Mtr[fit], ytr[fit], dom[fit],
                                      Xtr[val], Mtr[val], ytr[val],
                                      seed=s, epochs=epochs, n_num=n_num, n_dom=n_dom,
                                      aux_w=aux_w, lam_max=lam)
            soft += torch.softmax(R.predict(model, Xte, Mte), 1)
            vms.append(bi['val_macro'])
        vm = float(np.mean(vms))
        if vm > best['val']:
            best = {'val': vm, 'soft': soft, 'lam': lam}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='yolo26L1_')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--aux_w', type=float, default=0.5)
    ap.add_argument('--grl_lambda', type=float, default=1.0, help='max GRL strength (DANN ramp 0->this)')
    ap.add_argument('--grl_lambdas', default=None,
                    help='comma list of lambdas to sweep; per fold the best is picked on train-view val '
                         '(e.g. "0.05,0.1,0.3,1.0"). Overrides --grl_lambda.')
    ap.add_argument('--scene_map', default=None)
    args = ap.parse_args()
    lambdas = [float(x) for x in args.grl_lambdas.split(',')] if args.grl_lambdas else [args.grl_lambda]

    X, M, y, views, cids = load_pooled(args.tag)
    if args.scene_map:
        import json
        sm = json.load(open(args.scene_map))
        views = np.array([str(sm.get(c, sm.get(c[:-4], 'na'))) for c in cids])
        print(f"[scene_map] {args.scene_map} | mapped {int((views!='na').sum())}/{len(views)} clips")
    yn = y.numpy(); n_num = X.shape[-1] - 1
    uniq = sorted(set(views))
    print(f"pooled {len(yn)} clips | groups {uniq} | feat {X.shape[-1]} | seeds {args.seeds} "
          f"epochs {args.epochs} | GRL lambdas {lambdas} (val-selected per fold) aux_w {args.aux_w}")

    model_pred = np.full(len(yn), -1)
    for v in uniq:
        te = np.where(views == v)[0]
        tr = np.where(views != v)[0]
        if len(te) == 0:
            continue
        tr_views = views[tr]
        dom_map = {g: i for i, g in enumerate(sorted(set(tr_views)))}   # training cameras -> 0..K-2
        dom = torch.tensor([dom_map[g] for g in tr_views], dtype=torch.long)
        best = fold_grl(X[tr], M[tr], y[tr], dom, X[te], M[te], n_num, len(dom_map),
                        args.seeds, args.epochs, args.aux_w, lambdas)
        model_pred[te] = best['soft'].argmax(1).numpy()
        macc = 100 * (model_pred[te] == yn[te]).mean()
        print(f"  held-out group {v}: n={len(te):3d} domains={len(dom_map)} → GRL acc={macc:5.1f}% "
              f"(val-selected lambda={best['lam']}, val_macro={best['val']:.1f})")

    n = len(yn); mc = int((model_pred == yn).sum())
    model_macro = macro_accuracy(yn, model_pred) * 100
    print(f"\n=== LEAVE-CAMERA-OUT · GRL (scene-robust, {n} clips) ===")
    print(f"  GRL relation-Transformer: {mc}/{n} = {100*mc/n:.1f}% top-1 | macro {model_macro:.1f}%")
    print(f"  champion (yolo26L1_, no GRL): 78.9% top-1 | macro 82.0%  [reference]")
    print("  per-class recall (LCO GRL):",
          {NAMES[k]: round(per_class_recall(yn, model_pred).get(k, float('nan')) * 100, 1) for k in range(4)})
    print("  confusion (LCO GRL):\n", confusion(yn, model_pred, 4))
    print(f"\n  Δ vs champion macro = {model_macro - 82.0:+.1f} pp — keep iff >0 AND k-fold does not diverge up.")


if __name__ == '__main__':
    main()
