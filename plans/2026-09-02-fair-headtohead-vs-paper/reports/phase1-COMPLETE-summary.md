# Phase 1 — COMPLETE (feature levers L1–L5)

**Date:** 2026-09-04. Additive (only my own experiment scripts edited; no thesis files touched).
**Outcome: one confirmed winner (L1). Every grounding lever for the WalkwayViol↔UnauthorInt confusion
either leaked scene identity or failed to transfer cross-camera — a detector-level limitation, not a
feature-tuning gap.**

## Levers evaluated (all on repaired background grouping; headline = LCO macro)

| lever | what | LCO macro | k-fold macro | verdict |
|---|---|---|---|---|
| — | Phase-0 baseline (`app_`) | 80.1% | 84.0% | — |
| **L1** | strip absolute xc,yc (camera fingerprint) | **82.0%** | **84.7%** | **KEEP** |
| L2 | homography zones (replace canonical geom) | 79.6% | 84.1% | reject (leaks) |
| L2b | homography zones (augment) | 76.4% | 85.0% | reject (leaks) |
| L5 | view-agnostic person-context | 79.2% | 84.3% | reject (misleads) |
| L3 | motion/dwell | — | — | not run (D2 rejected motion) |
| L4 | trajectory straightness | — | — | not run (motion family; needs tracking infra) |

**Phase-1 winner set = {L1}. Carry-forward base = `yolo26L1_` (LCO macro 82.0%, k-fold 84.7%).**

## The one win — L1
Removing absolute box centre (a camera fingerprint) lifts LCO macro +1.9pp and rebalances the confused
pair (UnauthorInt LCO recall 52.8→70.4). Cannot add a leak — it removes a signal. Details:
`phase1-L1-strip-abs-coords.md`.

## Why L2/L2b/L5 all failed — the same root cause
All three tried to supply the missing zone/geometry signal that separates WalkwayViol from UnauthorInt.
Each carried scene-specific information:
- **L2/L2b** homography polygons are per-camera registered → helped in-distribution (L2b k-fold +0.3)
  while dropping LCO (−2.4 / −5.6). Textbook scene-leak signature.
- **L5** person-density is per-camera characteristic → same in-distribution/out-of-distribution split;
  UnauthorInt recall craters −15.8pp on held-out cameras.

**Structural cause (token audit):** the detector emits boxes for only **5 of 8** ontology entities.
**Safe_Walkway, Restricted_Zone, Forklift_Safe are never detected.** The WalkwayViol↔UnauthorInt pair is
two Person_Unauthorized instances in *undetected* zones — there is no camera-invariant feature that
distinguishes them, because the discriminating entity (the zone) is not perceived. This bounds the whole
feature-lever family.

## Implication for Phase 2 (model levers — GATED on user go-ahead)
The residual confusion is a **detector/representation** problem, not a head-feature problem:
1. **Retrain the detector to emit zone boxes** (Safe_Walkway, Restricted_Zone) so person↔zone relations
   become real, camera-invariant features — the highest-leverage next step given this finding.
2. **M1 domain-invariance (GRL / adversarial camera head)** to force the head to discard scene-tied signal
   — directly targets the leak family that killed L2/L5.
3. M2/M3 (architecture/temporal) as originally planned.

## Corrected carry-forward numbers
- **Scene-robust headline (LCO, L1):** macro **82.0%**, top-1 77.9%, +61.4pp over scene-only.
- **k-fold (L1):** top-1 80.6±2.8%, macro **84.7%**, κ 0.731, MCC 0.738, NB Δ=3.5pp p=0.070.
- Test set still untouched (touched once at campaign end, per protocol).

## Unresolved questions
1. Proceed to Phase 2 (detector zone-retrain + M1 GRL)? Requires user go-ahead + likely detector
   retraining compute. **Recommend detector zone-retrain FIRST** — it is the root cause this phase surfaced.
2. Run L4 (straightness) before closing feature levers? Low expected value (motion pre-rejected, needs new
   tracking infra). Recommend skip.
3. Re-run YOLO11 LCO/k-fold under repaired grouping + L1 for a matched detector-ablation row? (cheap, still
   pending from Phase 0.)
