# Phase 1 · L2 — homography-registered zone geometry [REJECTED — honest negative]

**Date:** 2026-09-04. Additive (only my own experiment scripts edited; no thesis files touched).
**Verdict: REJECTED in both variants.** L2 was the *promoted top lever* (D2 predicted the most headroom
here). Experiment refutes the hypothesis: homography-registered zone geometry does **not** improve
scene-robust generalization — it acts as a per-camera scene-fingerprint proxy and *reduces* LCO macro.

## Hypothesis (from D2)
The 4 canonical geom cols `[iou_walkway, iou_restricted, foot_in_walkway, norm_dist]` use ONE polygon per
view with no per-clip alignment → `foot_in_walkway ≈ 0.12` for BOTH WalkwayViol and UnauthorInt (polygons
don't line up), leaving the pair unresolvable. D2 predicted per-clip ORB+RANSAC homography-warped polygons
+ continuous signed distances (`unsafenet/geometry.py`) would ground the confusion.

## Two variants tested (both on the L1 winner base; xc,yc stripped)
- **L2 (replace):** swap the 4 canonical geom for 5 homography geom `[in_walkway, sdist_walkway,
  in_restricted, sdist_restricted, green_under_foot]`. `build_l2_tokens.py --drop_xcyc`. feat 73.
- **L2b (augment):** *keep* the 4 canonical geom, *add* the 5 homography geom (rules out the
  registration-failure confound below). `build_l2_tokens.py --drop_xcyc --keep_old`. feat 77.

## Results (YOLO26, seeds=3, epochs=60, repaired background grouping)

| protocol | **L1 base** | L2 (replace) | L2b (augment) |
|---|---|---|---|
| **LCO macro** (headline) | **82.0%** | 79.6% (−2.4) | **76.4% (−5.6)** |
| LCO top-1 | 77.9% | 76.6% | 72.9% |
| k-fold macro | 84.7% | 84.1% (−0.6) | **85.0% (+0.3)** |
| k-fold top-1 | 80.6% | 81.0% | 82.0% |
| k-fold NB p vs paper | 0.070 | 0.040 | 0.032 |

### Per-class LCO recall
| class | L1 | L2 | L2b |
|---|---|---|---|
| WalkwayViol | 68.1 | 73.8 | 74.8 |
| UnauthorInt | 70.4 | 63.0 | 67.6 |
| OpenedPanel | 89.4 | 81.7 | **63.4** |
| CarryOverload | 100.0 | 100.0 | 100.0 |

## Why it fails — the leak signature
L2b **raises k-fold (+0.3) while dropping LCO (−5.6)**. That divergence is the campaign's canonical
scene-leak fingerprint: a feature that helps in-distribution (every camera in train) but fails to transfer
to a held-out camera is encoding camera identity, not behaviour. The homography features are **per-clip
registered to a per-view reference background** → they carry scene-specific structure. Adding them let the
model lean on that structure (k-fold up), which then does not exist for the held-out camera (LCO down,
OpenedPanel collapses 89.4→63.4 as 47 clips bleed into UnauthorInt).

### Registration-failure confound (ruled out as the *sole* cause)
11.3% of clips (51/452 train) get all-zero homography geom (ORB+RANSAC finds no valid homography → empty
warped polygons). L2-replace nulled working canonical geometry on those clips — a plausible partial driver.
But L2b (augment) preserves canonical geom for all clips and still regresses LCO *more*, so registration
failure is not the whole story: the homography features are actively scene-confounding, not merely noisy.

## Decision
**Reject L2 (both variants). Keep L1 as the Phase-1 base (LCO macro 82.0%).** The D2 "geometric fix via
homography" hypothesis is falsified: the WalkwayViol↔UnauthorInt confusion is geometric, but no
per-camera-registered geometry can resolve it cross-camera without leaking scene identity. Residual
confusion must be attacked with a camera-invariant signal (→ L5 relational-proximity × appearance, which
uses only within-frame relations, not absolute/registered scene coordinates).

## Unresolved questions
1. Would a *view-agnostic* zone signal (e.g. relative person↔nearest-restricted-box distance from the
   detector's own boxes, no reference polygon) avoid the leak while still grounding the confusion? This is
   effectively L5 — try there.
2. OpenedPanel's L2b collapse suggests the homography green-mask feature specifically confuses panel
   scenes; not worth chasing since L2 is rejected wholesale.
