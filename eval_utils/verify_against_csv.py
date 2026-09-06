"""Verify eval_utils reproduces the committed headline (56/64 top-1, 89.28% macro)
from deployment/artifacts/per-clip-logits-test64.csv — the Phase-00 success gate.

Run from the repo root:  .venv/bin/python -m eval_utils.verify_against_csv
"""
import csv
import os
import numpy as np

from eval_utils.metrics import top1, macro_accuracy, per_class_recall, per_class_fnr, confusion
from eval_utils.bootstrap_ci import bootstrap_ci
from eval_utils.calibration import ece
from eval_utils.safety_metrics import (
    per_class_miss_rate, cost_sensitive_score, roc_pr_auc, threshold_for_recall,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'deployment', 'artifacts', 'per-clip-logits-test64.csv')
CLASSES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']


def load():
    with open(CSV) as f:
        rows = [ln for ln in f if not ln.lstrip().startswith('#')]
    rd = csv.DictReader(rows)
    yt, yp, probs = [], [], []
    for r in rd:
        yt.append(int(r['true_label']))
        yp.append(int(r['ensemble_pred']))
        probs.append([float(r[f'ensemble_prob_{c}']) for c in range(4)])
    return np.array(yt), np.array(yp), np.array(probs)


def main():
    y, p, probs = load()
    n = len(y)
    correct = int((y == p).sum())
    t1 = top1(y, p) * 100
    mac = macro_accuracy(y, p) * 100

    print(f"Loaded {n} clips from {os.path.relpath(CSV, ROOT)}")
    print(f"top-1 = {correct}/{n} = {t1:.2f}%   macro (mean per-class recall) = {mac:.2f}%")

    pcr = per_class_recall(y, p)
    fnr = per_class_fnr(y, p)
    for c in range(4):
        nc = int((y == c).sum())
        print(f"  {CLASSES[c]:14s} n={nc:2d}  recall={pcr[c]*100:6.2f}%  FNR={fnr[c]*100:6.2f}%")

    _, lo, hi, sd = bootstrap_ci(
        y, p, lambda a, b: macro_accuracy(a, b) * 100,
        n_boot=5000, seed=20260505, method='bca')
    _, tlo, thi, _ = bootstrap_ci(
        y, p, lambda a, b: top1(a, b) * 100,
        n_boot=5000, seed=20260505, method='bca')
    print(f"  top-1 BCa 95% CI = [{tlo:.1f}, {thi:.1f}]   macro BCa 95% CI = [{lo:.1f}, {hi:.1f}]")

    print(f"  ECE(15 bins) = {ece(probs, y):.4f}")

    # binary unsafe-vs-safe is degenerate here (all 64 clips are unsafe) — show the API works
    unsafe_score = 1.0 - probs.min(axis=1) * 0  # placeholder P(unsafe)=1 (all unsafe subset)
    miss = per_class_miss_rate(y, p)
    cost = cost_sensitive_score(y, p, miss_cost=10.0, false_alarm_cost=1.0)
    print(f"  per-class miss-rate = {[round(miss[c]*100,1) for c in range(4)]}  cost(miss x10) = {cost:.0f}")

    print("confusion matrix (rows=true, cols=pred):")
    print(confusion(y, p, 4))

    assert correct == 56, f"FAIL expected 56/64 top-1, got {correct}/64"
    assert abs(mac - 89.28) < 0.15, f"FAIL expected 89.28% macro, got {mac:.2f}%"
    assert pcr[2] == 1.0 and pcr[3] == 1.0, "FAIL OpenedPanel/CarryOverload should be 100%"
    print("\nPASS: eval_utils reproduces 56/64 top-1 and 89.28% macro from the committed artifact.")


if __name__ == '__main__':
    main()
