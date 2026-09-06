# Phase 1 · L1 — strip absolute (xc, yc) [KEEP]

**Date:** 2026-09-03. Additive (only my own experiment scripts edited; no thesis files touched).
**Verdict: KEEP.** Improves the headline LCO macro and rebalances the confused pair; no leak possible
(a signal was *removed*, not added).

## Lever
`tools/build_l1_tokens.py` drops columns 1,2 (absolute box centre `xc,yc`) from the existing
`yolo26app_` tokens — pure column surgery, no re-detection/appearance recompute. Rationale (D2): absolute
centre re-encodes camera identity (a subject recurs in the same screen region per camera), inflating
same-dist/random-kfold and rotting leave-scene-out. Token 74→72; mask/labels/clip_ids/views preserved so
repaired-scene LCO and k-fold run unchanged. New layout head: `[class, w, h, conf, iou_walkway,
iou_restricted, foot_in_walkway, norm_dist, app_0…]`.

## Result vs Phase-0 baseline (YOLO26, seeds=3, epochs=60, repaired background grouping)

| protocol | baseline `app_` | **L1** | Δ macro |
|---|---|---|---|
| **LCO macro** (headline) | 80.1% | **82.0%** | **+1.9pp** |
| LCO top-1 | 78.5% (405/516) | 77.9% (402/516) | −0.6pp (noise) |
| LCO scene-only macro | 10.1% | 10.1% | — |
| **k-fold macro** | 84.0% | **84.7%** | +0.7pp |
| k-fold top-1 | 80.8±3.1% | 80.6±2.8% (416/516) | flat |
| k-fold κ / MCC | 0.731 / 0.735 | 0.731 / 0.738 | flat |
| k-fold NB Δ vs paper | p=0.134 | Δ=3.5pp t=2.46 **p=0.070** | tighter |

### Per-class LCO recall (the important effect)
| class | baseline | **L1** | Δ |
|---|---|---|---|
| WalkwayViol | 79.5 | 68.1 | −11.4 |
| **UnauthorInt** | 52.8 | **70.4** | **+17.6** |
| OpenedPanel | 88.0 | 89.4 | +1.4 |
| CarryOverload | 100.0 | 100.0 | — |

## Reading
Removing absolute coordinates forces the model off the "where in frame" shortcut that favoured the
majority WalkwayViol class, onto relational/zone evidence — which is exactly what the weak, geometrically
confused UnauthorInt class needs. Net: macro up on **both** protocols, top-1 flat (within seed noise),
and the WalkwayViol↔UnauthorInt imbalance narrows sharply (26.7pp gap → 2.3pp gap in recall). This is
de-confounding, not curve-fitting: the leak-guard is automatic because L1 can only remove a fingerprint.

## Decision
**Keep L1.** It becomes the base for L2. The L1+L2 combo is built by
`tools/build_l2_tokens.py --src yolo26app_ --dst yolo26L12_ --drop_xcyc` (homography-registered zone
geometry recomputed from `yolo26app_`'s retained xc,yc, *then* xc,yc stripped) and evaluated against the
**L1 baseline (LCO macro 82.0%)**.

## Unresolved questions
1. WalkwayViol recall fell 11.4pp — acceptable as a rebalancing trade (macro rose), but L2 should aim to
   recover it via grounded walkway geometry rather than absolute position.
