#!/usr/bin/env python3
"""M — operating-point / safety-recall threshold sweep on the canonical model (selective classification).

The 4-class test is all-unsafe, so the deployable decision is: fire the alert when the model is confident
(max-softmax ≥ τ), else abstain (defer to a human). We sweep τ and report the risk-coverage trade-off:
coverage (= fraction of unsafe clips confidently flagged = firing recall) vs accuracy-on-flagged. The
safety operating point is the highest τ that keeps firing-recall ≥ 0.95 (miss ≤ 5%). One-vs-rest ROC/PR
AUCs and per-class miss-rate (FNR) are reported for the safety lens. (False-alarm rate needs safe clips —
noted as a binary-8-class extension.)

Usage: .venv/bin/python run_operating_point.py [--logits logits_yoloroi.pt]
"""
import argparse
import json
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from eval_utils.metrics import top1, per_class_fnr

TOK = 'sweep_results/relation_tokens'
NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']


def risk_coverage(P, y, taus):
    """Return [(τ, coverage, acc-on-fired)] over the given thresholds."""
    pred = P.argmax(1); conf = P.max(1)
    rows = []
    for tau in taus:
        keep = conf >= tau
        cov = keep.mean(); acc = (pred[keep] == y[keep]).mean() if keep.sum() else 0.0
        rows.append((tau, cov, acc))
    return rows


def select_tau(rows, min_recall=0.95):
    """Highest τ whose coverage (firing-recall) ≥ min_recall on the SELECTION set."""
    ok = [r for r in rows if r[1] >= min_recall]
    return (max(ok, key=lambda r: r[0]) if ok else rows[0])[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--logits', default='logits_yoloroi.pt')
    ap.add_argument('--select_logits', default=None,
                    help='D3: leakage-free selection set (e.g. kfold_oof_softmax_*.pt) to CHOOSE τ; '
                         'applied untouched to test. Without it, τ cannot be selected honestly — only the '
                         'risk-coverage curve is printed (a curve is not a test-selected operating point).')
    args = ap.parse_args()
    d = torch.load(os.path.join(TOK, args.logits), weights_only=False)
    P = d['ens_softmax'].numpy(); y = d['labels'].numpy()
    pred = P.argmax(1); conf = P.max(1); n = len(y)
    Y = np.eye(4)[y]

    taus = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    print("=== M — risk-coverage (selective classification) ===")
    print(f"  {'τ':>5} {'coverage(firing-recall)':>24} {'acc-on-fired':>13}  abstained  (TEST curve, diagnostic)")
    rows = risk_coverage(P, y, taus)
    for (tau, cov, acc) in rows:
        print(f"  {tau:>5.2f} {100*cov:>22.1f}% {100*acc:>12.1f}%  {int((conf < tau).sum())}")

    if args.select_logits:
        ds = torch.load(os.path.join(TOK, args.select_logits), weights_only=False)
        Ps = ds['ens_softmax'].numpy(); ys = ds['labels'].numpy()
        sel_rows = risk_coverage(Ps, ys, taus)
        tau_star = select_tau(sel_rows, 0.95)
        keep = conf >= tau_star
        cov = keep.mean(); acc = (pred[keep] == y[keep]).mean() if keep.sum() else 0.0
        print(f"\n  τ* selected on {args.select_logits} (n={len(ys)}, firing-recall ≥ 95%) = {tau_star:.2f}")
        print(f"  SAFETY OPERATING POINT (val-fit / test-apply): τ={tau_star:.2f} → TEST coverage {100*cov:.1f}%, "
              f"accuracy-on-fired {100*acc:.1f}%")
        op = (tau_star, cov, acc)
    else:
        tau_star = select_tau(rows, 0.95)
        keep = conf >= tau_star; cov = keep.mean(); acc = (pred[keep] == y[keep]).mean() if keep.sum() else 0.0
        op = (tau_star, cov, acc)
        print(f"\n  [!] WARNING: no --select_logits given. τ={tau_star:.2f} shown below is TEST-SELECTED "
              f"(coverage {100*cov:.1f}%, acc-on-fired {100*acc:.1f}%) — NOT evidence-grade.")
        print(f"      For a reportable operating point run:  run_kfold_relation.py  then re-run with")
        print(f"      --select_logits kfold_oof_softmax_<tag>.pt  (selects τ off-test, applies here).")
    print(f"  one-vs-rest ROC-AUC={roc_auc_score(Y, P, average='macro', multi_class='ovr'):.3f}  "
          f"PR-AUC={np.mean([average_precision_score(Y[:,c], P[:,c]) for c in range(4)]):.3f}")
    fnr = per_class_fnr(y, pred)
    print("  per-class miss-rate (FNR) at full coverage:", {NAMES[c]: round(fnr[c]*100, 1) for c in range(4)})
    print("  NOTE: false-alarm rate requires safe clips (classes 4–7) — a binary/8-class extension; the")
    print("        4-class test is all-unsafe, so only firing-recall + abstention are measurable here.")

    json.dump({'risk_coverage': [(round(t,2), round(c,3), round(a,3)) for t,c,a in rows],
               'operating_point': {'tau': op[0], 'coverage': round(op[1],3), 'acc_on_fired': round(op[2],3)}},
              open(os.path.join(TOK, 'operating_point.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
