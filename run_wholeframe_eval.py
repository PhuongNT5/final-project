#!/usr/bin/env python3
"""S3 diagnostic — whole-frame classifier (paper's scene approach) under same-distribution AND
leave-camera-out. A per-frame MLP probe on frozen whole-frame features, clip label propagated to
frames, mean-softmax aggregation. If it scores high same-distribution but collapses under LCO, that
proves the whole-frame/scene approach is scene-inflated — unlike our scene-robust entity model.

Usage: .venv/bin/python run_wholeframe_eval.py [--seeds 3] [--epochs 60]
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split
from eval_utils.metrics import top1, macro_accuracy, per_class_recall, confusion

TOK = 'sweep_results/relation_tokens'
NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']


def load(sp):
    d = torch.load(f'{TOK}/tokens_wf_{sp}.pt', weights_only=False)
    return d['feats'].float(), d['labels'].long(), np.array([str(v) for v in d['views']])


class FrameMLP(nn.Module):
    def __init__(self, d=64, h=128, n=4, p=0.4):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, n))

    def forward(self, x):                    # x (B,T,d) → per-frame logits (B,T,n)
        return self.net(x)


def train_probe(Xf, yf, Xv, yv, seed, epochs):
    np.random.seed(seed); torch.manual_seed(seed)
    m = FrameMLP()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    T = Xf.shape[1]
    best = {'vm': -1}
    for ep in range(epochs):
        m.train()
        idx = torch.randperm(len(Xf))
        for i in range(0, len(Xf), 32):
            b = idx[i:i+32]
            logit = m(Xf[b])                 # (B,T,4)
            yb = yf[b].view(-1, 1).expand(-1, T)
            loss = nn.functional.cross_entropy(logit.reshape(-1, 4), yb.reshape(-1), label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vp = torch.softmax(m(Xv), -1).mean(1).argmax(1).numpy()   # clip = mean per-frame softmax
        vm = macro_accuracy(yv.numpy(), vp) * 100
        if vm > best['vm']:
            best = {'vm': vm, 'state': {k: v.clone() for k, v in m.state_dict().items()}}
    m.load_state_dict(best['state'])
    return m


@torch.no_grad()
def clip_softmax(m, X):
    m.eval()
    return torch.softmax(m(X), -1).mean(1)   # (N,4)


def same_dist(seeds, epochs):
    Xtr, ytr, _ = load('train'); Xte, yte, _ = load('test')
    fit, val = train_test_split(np.arange(len(ytr)), test_size=0.18, stratify=ytr.numpy(), random_state=0)
    soft = torch.zeros(len(yte), 4)
    for s in range(seeds):
        m = train_probe(Xtr[fit], ytr[fit], Xtr[val], ytr[val], s, epochs)
        soft += clip_softmax(m, Xte)
    pred = soft.argmax(1).numpy(); y = yte.numpy()
    print(f"\n=== WHOLE-FRAME, SAME-DISTRIBUTION (64-clip test) ===")
    print(f"  top-1 = {int((pred==y).sum())}/64 = {100*top1(y,pred):.1f}% | macro {macro_accuracy(y,pred)*100:.1f}%")
    print("  per-class:", {NAMES[k]: round(per_class_recall(y,pred)[k]*100,1) for k in range(4)})


def lco(seeds, epochs):
    Xa, ya, va = load('train'); Xb, yb, vb = load('test')
    X = torch.cat([Xa, Xb]); y = torch.cat([ya, yb]); views = np.concatenate([va, vb]); yn = y.numpy()
    pred = np.full(len(yn), -1)
    for v in sorted(set(views)):
        te = np.where(views == v)[0]; tr = np.where(views != v)[0]
        fit, val = train_test_split(np.arange(len(tr)), test_size=0.15, stratify=yn[tr], random_state=0)
        soft = torch.zeros(len(te), 4)
        for s in range(seeds):
            m = train_probe(X[tr][fit], y[tr][fit], X[tr][val], y[tr][val], s, epochs)
            soft += clip_softmax(m, X[te])
        pred[te] = soft.argmax(1).numpy()
    print(f"\n=== WHOLE-FRAME, LEAVE-CAMERA-OUT (516 clips) ===")
    print(f"  top-1 = {int((pred==yn).sum())}/{len(yn)} = {100*top1(yn,pred):.1f}% | macro {macro_accuracy(yn,pred)*100:.1f}%")
    print("  per-class:", {NAMES[k]: round(per_class_recall(yn,pred)[k]*100,1) for k in range(4)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=60)
    args = ap.parse_args()
    same_dist(args.seeds, args.epochs)
    lco(args.seeds, args.epochs)
    # canonical 5-seed values (canonical/lco_seeds5_yolo26L1_CANONICAL.log); the 75.6/16.9 pair
    # printed here previously was a superseded 3-seed run.
    print("\nContrast — entity-relation model: same-dist 84.4% | LCO 77.7% (macro 81.2%).")
    print("scene-appearance (transferable baseline): LCO 53.1% (macro 64.4%).")
    print("constant floor (NOT a competitor): LCO 16.5% (macro 10.1%).")


if __name__ == '__main__':
    main()
