# Finding: only ONE sub-task requires relational reasoning — and it is the one that fails

Date: 2026-09-19. Phase-03 kill gate + follow-on diagnostic. Evidence recomputed from
`tokens_bin8_{train,test}.pt` (the headline tokens) x `clip_view_map_691.json` x the LCO OOF dumps.
**Zero test touches, zero GPU hours, zero new training.**

## 1. Kill-gate result: detector classes 2/3 were NEVER annotated

`annotation_guide/dataset/labels/{train,val}/*.txt`, all 6,933 instances:

| cls0 P_Auth | cls1 P_Unauth | **cls2 Safe_Walkway** | **cls3 Restricted_Zone** | cls4 PanelClosed | cls5 PanelOpen | cls6 FkSafe | cls7 FkOver |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 365 | 4497 | **0** | **0** | 1181 | 525 | 170 | 195 |

Zero in every file — a genuine data gap, not lost annotation. The **zone polygons do exist and are
rich**: 7 view files, each 1-3 `safe_walkway` and 1-5 `restricted_zone` polygons, full vertex lists
at 1920x1080 (`view_1.json` alone has 3 walkway + 2 restricted polygons).

So L1 is *instantiable*. But the diagnostic below says the L1 as written attacks the wrong thing.

## 2. What each sub-task actually rests on

Detector-class composition of tokens, per behaviour sub-class (% of valid tokens):

| sub-class | clips | P_Auth | P_Unauth | PanelClosed | PanelOpen | FkSafe | FkOver |
|---|--:|--:|--:|--:|--:|--:|--:|
| 0 walkway_violation | 210 | 1.1 | 75.5 | 21.4 | 2.0 | 0.0 | 0.0 |
| 4 walkway_ok | 75 | 4.0 | 71.2 | 17.9 | 6.9 | 0.0 | 0.0 |
| 1 restricted_zone | 108 | 2.8 | 77.7 | 16.8 | 2.7 | 0.0 | 0.0 |
| 5 zone_ok | 38 | **23.1** | 49.6 | 13.5 | 13.7 | 0.0 | 0.0 |
| 2 panel_open | 142 | 9.3 | 58.6 | 3.7 | **28.4** | 0.0 | 0.0 |
| 6 panel_closed | 32 | 2.0 | 66.7 | 31.2 | **0.0** | 0.0 | 0.0 |
| 3 carry_overload | 56 | 0.0 | 0.6 | 0.0 | 0.0 | 0.0 | **99.4** |
| 7 carry_safe | 30 | 0.0 | 0.0 | 0.0 | 0.0 | **100.0** | 0.0 |

Person-token geometry (dims 6-9), mean over valid person detections:

| sub-class | iou_walkway | iou_restricted | **foot_in_walkway** | norm_dist |
|---|--:|--:|--:|--:|
| 0 walkway_violation | 0.0140 | 0.0101 | **0.1476** | 0.2234 |
| 4 walkway_ok | 0.0195 | 0.0268 | **0.2285** | 0.2280 |
| 1 restricted_zone | 0.0114 | **0.0268** | 0.1192 | 0.2658 |
| 5 zone_ok | 0.0055 | **0.1193** | 0.0798 | 0.2045 |

## 3. The decomposition

| sub-task | what separates the pair | LCO result |
|---|---|---|
| **carry** (3 vs 7) | **detector class alone** — 99.4% FkOver vs 100.0% FkSafe, perfect separation | 100% / 100% |
| **panel** (2 vs 6) | **detector class alone** — panel_closed contains *zero* PanelOpen tokens | unanswerable (single-camera) |
| **zone** (1 vs 5) | **class + geometry** — P_Auth 2.8%→23.1% (8x) *and* iou_restricted 0.027→0.119 (4.4x) | **76-90% both sides** |
| **walkway** (0 vs 4) | **geometry alone** — detector composition is near-identical (P_Unauth 75.5 vs 71.2); only `foot_in_walkway` 0.148 vs 0.229 separates | **17-36% safe recall** |

**The ordering is exact and it is the thesis result.** Where the task reduces to a detector class
lookup, it is solved (~100%). Where it is class *plus* geometry, it works and generalises
cross-camera (76-90%) — this is the genuine positive result and the closest thing in this dataset to
relational reasoning. Where it requires **geometry alone**, it fails (17-36%). The relational
component is the weakest link, and most of the pipeline's apparent success is detector lookup.

Note the geometry direction is semantically *correct*, not inverted: `safe_walkway` is the designated
pedestrian path, so being on it (foot_in 0.229) is SAFE and being off it (0.148) is the violation.
The signal exists and points the right way — it is simply too weak to survive 64 appearance dims.

## 4. Consequence: L1 as written attacks the wrong thing — redirect it

L1 says "populate `Safe_Walkway`/`Restricted_Zone` as zone-entity tokens from existing polygons."
Two objections:

1. **The person-zone relation is already encoded** — `iou_walkway`, `iou_restricted`,
   `foot_in_walkway`, `norm_dist` are exactly that relation, and they are why the zone sub-task
   works. Classes 2/3 are not missing information; they are a second encoding of it.
2. **A zone as an entity token is a per-view CONSTANT.** The polygons are static per camera. Injected
   as tokens they carry camera identity, which is precisely the shortcut LCO exists to delete — so
   the most likely effect under LCO is neutral-to-harmful, while in-distribution k-fold would rise.
   That failure mode would look like success on the wrong protocol.

**Redirect L1 to the measured defect: `foot_in_walkway` is too weak.** It averages 0.229 on
`walkway_ok` where a person genuinely standing on the walkway should score near 1.0 — i.e. only ~23%
of person-frames register as on-path. `iou_walkway` is near-useless by construction (0.014-0.020: a
person bbox against a large polygon cannot produce meaningful IoU) and `norm_dist` carries no signal
at all (0.223 vs 0.228). Concrete candidates, cheapest first: audit per-view walkway polygon
placement (`tools/check_zone_separation.py` exists); replace IoU with **signed distance from the foot
point to the walkway boundary**; fix foot-point estimation (bottom-centre of bbox vs actual stance);
temporally aggregate the foot-in signal over T=16 instead of per-frame.

Same pre-registered bar: must move panel-disclosed LCO BA past **76.6%** (E3) or dead and written up
as a negative result.

## 5. What this is worth at the defence

It converts "our accuracy plateaued" into a specific, falsifiable claim with a measurement behind
every clause: *on Unsafe-Net, sub-tasks separable by detector class are saturated; the one sub-task
separable by class-plus-geometry generalises cross-camera; the one sub-task requiring geometry alone
fails, and its geometry channel is measurably under-powered.* That also states plainly what the
relation-token contribution is and is not — which is the question a committee will actually ask.

## Unresolved questions

1. Does fixing `foot_in_walkway` move walkway_ok recall? The redirected L1 bet.
2. Are the per-view `safe_walkway` polygons correctly placed? Unaudited; a systematic offset would
   explain 0.229 directly and would be the cheapest possible fix.
3. `carry_overload` shows only 1 clip / 5 person-tokens in the person-geometry table (§2) because
   99.4% of its tokens are FkOver, not people — worth confirming no clip is being silently dropped.
4. Does the zone sub-task's 76-90% survive if `Person_Authorized`/`Person_Unauthorized` is blinded?
   That is the clean ablation separating "relational reasoning" from "authorized-person lookup", and
   it decides how strongly the positive result can be worded.
