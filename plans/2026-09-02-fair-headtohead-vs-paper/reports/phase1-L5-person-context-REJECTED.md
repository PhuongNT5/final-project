# Phase 1 · L5 — view-agnostic relational-context [REJECTED — honest negative]

**Date:** 2026-09-04. Additive (only my own experiment scripts edited; no thesis files touched).
**Verdict: REJECTED.** Person-density context separates the confused pair in-distribution but is itself
scene-tied and misleads cross-camera → LCO macro drops.

## Structural finding that reshaped L5 (detector emits only 5/8 entities)
Token audit of `yolo26app_` (26,438 train boxes): the detector produces boxes for **5 of 8** ontology
classes only —
| class | share of detected boxes |
|---|---|
| Person_Unauthorized | 69.9% |
| Panel_Closed | 14.7% |
| Panel_Opened | 8.9% |
| Person_Authorized | 3.5% |
| Forklift_Overload | 2.9% |
| **Safe_Walkway (2)** | **0.0%** |
| **Restricted_Zone (3)** | **0.0%** |
| **Forklift_Safe (6)** | **0.0%** |

Zones are **never detected as boxes** — they live only as the static per-view canonical polygon prior
(the `iou_walkway/iou_restricted/foot_in_walkway/norm_dist` cols, ~0.12 for both confused classes per D2).
So the original L5 design (proximity-to-detected-zone-box) is identically zero (verified) and impossible.
The WalkwayViol↔UnauthorInt pair (both Person_Unauthorized, no zone boxes, no other distinguishing
entity) has **no clean detected signal separating it.**

## Redesigned L5 (the one remaining view-agnostic signal)
Person↔person spatial context + person↔Panel/Forklift offset, all translation-invariant (no absolute
coords): `[n_persons, nn_person_dist, mean_person_dist, n_within_0.2, panel_present, panel_dist,
fork_present, fork_dist]`. `build_l5_tokens.py --drop_xcyc`. feat 80.

**Separation IS non-zero (train):** WalkwayViol denser than UnauthorInt —
nn_person_dist 0.223 vs 0.289, mean_person_dist 0.373 vs 0.459, n_within_0.2 0.671 vs 0.594
("shared walkway" vs "isolated intrusion"). Modest but real → justified an eval run.

## Result (YOLO26, seeds=3, epochs=60, repaired grouping)
| protocol | L1 base | **L1+L5** | Δ |
|---|---|---|---|
| **LCO macro** (headline) | **82.0%** | 79.2% | **−2.8** |
| LCO top-1 | 77.9% | 76.7% | −1.2 |
| k-fold macro | 84.7% | 84.3% | −0.4 |
| k-fold NB p | 0.070 | 0.103 (n.s.) | — |

### Per-class LCO recall
| class | L1 | L1+L5 |
|---|---|---|
| WalkwayViol | 68.1 | 74.8 |
| **UnauthorInt** | **70.4** | **54.6** |
| OpenedPanel | 89.4 | 87.3 |
| CarryOverload | 100.0 | 100.0 |

## Why it fails
Person-density is scene-dependent (each camera frames a characteristic number of people at characteristic
spacing). In-distribution the model exploits it (k-fold barely moves); on a held-out camera the learned
density prior is wrong, so ambiguous cases collapse to the denser default class (WalkwayViol) and
UnauthorInt recall craters −15.8pp. Same failure family as L2: discriminative in-distribution, scene-tied
out-of-distribution.

## Decision
**Reject L5. Keep L1 as the base.** With L2 and L5 both rejected and L3/L4 (motion) pre-rejected by D2,
the feature-lever space for the WalkwayViol↔UnauthorInt confusion is exhausted: no camera-invariant signal
that resolves it exists at the feature level, because the detector does not ground zones. Resolving it is a
Phase-2 problem (retrain detector to emit zone boxes, or model-level domain-invariance / GRL — M1).

## Unresolved questions
1. Run L4 (trajectory straightness)? Needs new tracking infra AND D2 already rejected the motion
   hypothesis — low expected value; recommend skipping unless Phase-2 zone grounding still leaves residual
   confusion. (Flagged for user decision.)
