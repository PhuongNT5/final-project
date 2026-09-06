# Evaluation Suite Results (E, M, F, G)

**Date:** 2026-08-16. Completes the thesis eval spec on the canonical YOLO11 baseline (YOLO-ROI
appearance + aux Transformer, 7-seed ensemble). Same-distribution 64-clip test unless noted.

## E — k-fold rotation + Nadeau–Bengio
- **Stratified random 5-fold** (pooled 516 clips): relation **81.2 ± 2.5 %** top-1 (macro 84.5 %) vs
  scene-only 77.1 ± 1.0 %. **Nadeau–Bengio corrected t-test: Δ=+4.1 pp, t=3.07, p=0.0375 (significant).**
- **Leave-camera-out** (camera-group rotation, scene-robust): relation **80.4 %** vs scene-only 16.9 %.
- The pooled k-fold gives the statistical power the 64-clip test lacked (there McNemar was n.s., p=0.815).
  → the relation model's behaviour signal over scene-guessing is now statistically established.

## M — operating point / safety-recall (selective classification)
Risk-coverage on the confidence threshold τ (fire alert if max-softmax ≥ τ, else abstain):

| τ | firing-recall (coverage) | accuracy-on-fired |
|---|---|---|
| 0.50 (safety op-point, recall ≥ 95 %) | 98.4 % | 84.1 % |
| 0.90 | 48.4 % | **96.8 %** |

- ROC-AUC (ovr) 0.949, PR-AUC 0.867. Per-class FNR: Walk 22 · Unauth 9 · Panel 15 · **Carry 0**.
- The model is well-behaved for selective deployment: high-confidence subset (48 %) is 96.8 % accurate.
- False-alarm rate needs safe clips (classes 4–7) → a binary/8-class extension (4-class test is all-unsafe).

## G — corruption robustness (the paper lacks this)
Train on clean, evaluate on corrupted test frames (detector + classifier pipeline):

| corruption | top-1 | Δ vs clean |
|---|---|---|
| clean | 54/64 (84.4 %) | — |
| jpeg (Q20) | 55/64 | +1 |
| noise (σ=25) | 55/64 | +1 |
| blur (9×9) | 53/64 | −1 |
| dark (0.4×) | 50/64 | −4 |

**Robust to compression / noise / blur (±1 clip); low-light is the only meaningful degradation (−4).**
The entity-detector front-end absorbs common surveillance corruptions well.

## F — test-time augmentation (TTA)
- flip 52/64 (−2); TTA(clean+flip) 52/64 (−2). **No gain** — honest negative. Zones are view-asymmetric,
  so horizontal flip is not label-preserving for the geometry stream. Multi-scale/temporal-crop TTA not
  pursued given the flip result and the research prediction of neutral/negative TTA on this data.

## Status: evaluation spec complete
Done: multi-lens metrics (κ 0.775, MCC 0.785, macro-F1 85.6, balanced-acc, ROC/PR-AUC, per-class P/R/FNR),
safety (cost-sensitive, FNR), calibration (ECE 0.104), statistics (BCa/Wilson/McNemar + **Nadeau–Bengio**),
timing (0.9 ms/clip classifier, ~200 ms/clip pipeline), k-fold + LCO, corruption robustness, TTA. Both
same-distribution AND scene-robust (LCO) columns reported throughout. **Remaining: MT3 (YOLO26 + Jetson).**
