# Phase 03 — Spatial-Relation Transformer v1 (geometry-only)

**Date:** 2026-08-09 · Stage B on the 4-class unsafe subset (64-clip blind test).
Token feat[10] = [class, xc,yc,w,h, conf, iou_walkway, iou_restricted, foot_in_walkway, norm_dist].
Model: RelationTransformer (~0.5 M params), CPU training (MPS training bug), 5 seeds, blind held-out
(val carved from 452 train clips; test touched once/seed; seed selected on val macro).

## Result (blind, val-selected seed)
- **top-1 = 48/64 = 75.0%** (BCa 95% CI [60.9, 82.8]) · **macro = 83.7%**
- 5-seed: 48.6 ± 0.5 /64 · ensemble (test-selected caveat) 49/64
- Per-class recall: WalkwayViol 59.4% · UnauthorInt 90.9% · OpenedPanel 84.6% · CarryOverload 100%

Confusion (rows=true): WalkwayViol → 19 correct, **12 → UnauthorInt**; UnauthorInt 10/11; OpenedPanel 11/13; CarryOverload 8/8.

## Comparison
| | top-1 | macro |
|---|---|---|
| v1 geometry-only (blind) | 48/64 (75.0%) | 83.7% |
| Paper no-bbox | 57/64 (89.1%) | 85.78% |
| Paper full-bbox | 62/64 (96.9%) | 95.81% |
| Thesis frozen-appearance (blind) | 53/64 | ~85% |

blind vs no-bbox: Δ = −14.1 pp top-1 (z=−2.07, p=0.038).

## Findings
- **Macro competitive, top-1 not.** Model excels on UnauthorInt/OpenedPanel/CarryOverload but WalkwayViolation
  leaks to UnauthorInt (12/32) — the target confusion is not broken by geometry alone.
- **Geometry scalars too weak:** foot_in_walkway ~0.12 for *both* WalkwayViol & UnauthorInt; iou_restricted tiny
  (0.01 vs 0.03). Per-view canonical zones don't align tightly per clip; the discriminative info is largely
  appearance/position, which geometry-only tokens discard.
- **Class weighting side-effect:** inverse-freq weights up-weight minority UnauthorInt → over-prediction (adds to the leak).

## Next → v2 (the original ROI-appearance design)
Add a per-box **appearance feature** (detector-backbone ROI feature or crop embedding) to each token:
`token = [ROI appearance ⊕ coords ⊕ class ⊕ geometry ⊕ conf]`. This restores what the frozen-appearance
approach had (53/64) *and* adds geometry — the actual hypothesis. Secondary levers: milder/no class weights
(blind-selected), person↔Panel/Machine proximity features, per-clip zone refinement.
