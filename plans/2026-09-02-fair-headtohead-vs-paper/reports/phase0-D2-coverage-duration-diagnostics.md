# Phase 0 · D2 — coverage / duration diagnostics (near-free)

**Date:** 2026-09-03. Additive (no thesis files edited). Source: token dicts
`sweep_results/relation_tokens/tokens_yolo26app_{train,test}.pt` (74-dim; layout
`[class,xc,yc,w,h,conf,iou_walkway,iou_restricted,foot_in_walkway,norm_dist]`+64 appearance).
Entity ontology: `sweep_results/entity_detector/data_resolved.yaml`.

## D2(i) — Which entities actually reach the tokens? (zone-grounding audit)

Valid-token counts by entity class (train split):

| entity | name | valid tokens | person-geometry cols |
|---|---|---|---|
| 0 | Person_Authorized | 931 | non-zero (iou_restr .18, foot_walk .02) |
| 1 | Person_Unauthorized | 18 487 | non-zero (iou_walk .01, foot_walk .13, ndist .24) |
| 2 | **Safe_Walkway** | **0** | — |
| 3 | **Restricted_Zone** | **0** | — |
| 4 | Panel_Closed | 3 898 | all-zero (geometry is person-only) |
| 5 | Panel_Opened | 2 356 | all-zero |
| 6 | Forklift_Safe_Load | **0** | — |
| 7 | Forklift_Overload | 766 | all-zero |

**Findings:**
1. **The two zone entities (Safe_Walkway, Restricted_Zone) are NEVER detected** — 0 tokens.
   So the `iou_walkway / iou_restricted / foot_in_walkway` features are NOT computed against a
   detected zone box; they come from a **static zone prior** (annotated polygon / homography), not
   from perception. Zone grounding is therefore fixed-per-view and weak → **L2 (homography-registered
   signed-distance zones) has genuine headroom** — it replaces an absent perceptual signal, it doesn't
   duplicate one.
2. **Person_Unauthorized outnumbers Person_Authorized 20:1** (18 487 vs 931). The authorized/
   unauthorized ontology split is effectively collapsed — nearly every person is tagged unauthorized.
   The auth-distinction carries little signal as currently detected.
3. **Forklift_Safe_Load (6) never detected**; only Forklift_Overload (7) appears — the safe/overload
   forklift split is one-sided (consistent with CarryOverload being the only forklift behaviour class).

## D2(ii) — Duration/dwell hypothesis: is UnauthorInt more stationary than WalkwayViol?

Motion proxy = std of per-frame mean person-xc across the 16 frames (higher = more lateral translation):

| behaviour | n | mean xc-std | median |
|---|---|---|---|
| WalkwayViol | 177 | 0.0541 | 0.0456 |
| **UnauthorInt** | 97 | **0.0599** | 0.0530 |
| OpenedPanel | 127 | 0.0568 | 0.0532 |

**Finding — hypothesis NOT supported.** UnauthorInt is *not* lower-motion than WalkwayViol; if anything
marginally higher (and the gap is tiny/noisy). The WalkwayViol↔UnauthorInt confusion is therefore
**not duration/dwell-based** — it is **boundary-geometry ambiguity** (which zone the person's feet
occupy), consistent with D2(i) showing zones are ungrounded.

→ **Ordering decision: prioritize L2 (zone geometry) over L3 (motion).** Motion (L3) targets a
discriminator that the data says is not the confusion's cause. This resolves plan Unresolved-Q#2:
the confused pair is geometric, not temporal.

## D2(iii) — Per-scene detector coverage (identity re-encoding risk)

Mean boxes/frame by repaired scene-group (D1 grouping): min 1.00, max 5.07, **CV 0.45**. The two
low-density scenes (2: 1.00, 5: 1.04) are the forklift/CarryOverload scenes (CarryOverload = 1.01
boxes/frame overall). So token *count* partly tracks behaviour/scene — a mild identity-re-encoding
channel, but benign (a forklift clip genuinely has fewer person boxes). **Caveat, not a bug:** do not
add raw token-count as a feature; it would re-encode scene. Coverage is otherwise adequate (3–5
boxes/frame) on the person-behaviour scenes.

## Consequences for the campaign
- **Re-order Phase 1: L1 → L2 → (skip/deprioritize L3) → L4/L5.** L2 is now the highest-value lever
  (fixes the actual confusion cause + fills an absent perceptual signal). L3 (motion) demoted to
  "only if L2 leaves WalkwayViol↔UnauthorInt residual."
- Consider an L2b micro-lever: since zones aren't detected, feed the person foot-point's signed
  distance to the (static) walkway/restricted polygon boundary — a continuous version of the current
  binary `foot_in_walkway`.

## Unresolved questions
1. Are zones absent because they were never annotated for detection, or detected-but-filtered? (If
   never annotated, L2 must rely on the homography polygon prior — which is fine, but confirms zone
   perception is out of scope.)
2. Does the 20:1 unauthorized:authorized imbalance mean the auth-split should be dropped from the token
   (fewer classes = simpler), or kept for the rare authorized cue? (Test as an L-ablation.)
