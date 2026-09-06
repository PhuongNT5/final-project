# Scene-Bias Finding — Critical Analysis Before S3

**Date:** 2026-08-14. Deep research + own-data analysis requested before building S3. Result: **S3 as a
"beat-the-baseline" move is the wrong play** — the benchmark is heavily scene-confounded. This reframes
the whole accuracy question and yields a stronger, honest thesis contribution.

## The finding: this dataset is scene-confounded
Camera view predicts the behaviour class strongly (from `clip_view_map.json`):

| view | n (train) | class mix | dominant |
|---|---|---|---|
| 0 | 61 | mostly OpenedPanel | OpenedPanel 85% |
| 1 | 110 | mostly WalkwayViol | WalkwayViol 90% |
| **2** | 27 | **only CarryOverload** | **100%** |
| 3 | 65 | mostly OpenedPanel | OpenedPanel 69% |
| **4** | 16 | **only UnauthorInt** | **100%** |
| 5 | 95 | Walkway/Unauth mix | WalkwayViol 57% |
| 6 | 57 | mixed | OpenedPanel 56% |
| **7** | 21 | **only CarryOverload** | **100%** |

**3 of 8 camera views are single-class** (pure scene shortcuts).

### Scene-only baseline (predict behaviour from camera view alone)
Using the train view→majority-class map on the 64 test clips: **52/64 = 81.2 %.**
A model that ignores behaviour entirely and looks only at *which camera* scores **81.2 %**.

## Implications (sobering + important)
1. **Our best model (S1a, 54/64 = 84.4 %) is only +2 clips above the scene-only baseline (52/64).**
   Most of our "accuracy" — and everyone's — is scene recognition, not behaviour understanding.
2. **The paper's 95.81 % is almost certainly heavily scene-inflated.** YOLOv4 detecting behaviour-as-object
   on fixed cameras with these view↔class correlations largely memorises the scene/view. The paper runs
   no camera-holdout, no background ablation → the headline is confounded. **A defensible, novel critique.**
3. **S3 (whole-frame classifier) would "reach" ~57–62/64 — but mostly by scene memorisation** (floor is
   already 81.2 % from view alone). Beating the no-bbox baseline this way would be scientifically hollow.
4. Literature is unambiguous that this is the dominant failure mode of appearance/scene models in action
   recognition (RESOUND ECCV'18; SDN NeurIPS'19; StillMix ICCV'23 — all Kinetics models focus on
   background over motion). See `researcher-s3-frame-behaviour.md`.

## Reframed strategy — the scientifically strong (and honest) path
Instead of chasing a scene-inflated number, **measure and control the scene bias** — this is a genuine
contribution and fits the thesis's evaluation-rigor theme.

1. **Scene-only baseline** (done): 52/64 = 81.2 % — the floor every method must be judged against.
2. **Leave-Camera-Out (LCO) evaluation** of our entity-Transformer + a scene-only baseline: train on
   some views, test on held-out views → reveals TRUE behaviour-generalisation (expected: all methods drop).
3. **S3 as a diagnostic bracket, not a headline:** build S3-whole-frame AND S3-actor-crop (person crops,
   background removed, using our detector). Gap = scene-bias magnitude. Report both, never the whole-frame
   number alone.
4. **Positioning:** our entity-relation approach reasons over person×zone relations → structurally more
   scene-agnostic than whole-frame/behaviour-detector methods; the right design for cross-camera deployment.
   The thesis reports scene-confounded (same-distribution) AND scene-robust (LCO) numbers for all methods,
   and critiques the paper's un-ablated 95.81 %.

## RESULTS — Leave-Camera-Out scene-robust evaluation (2026-08-14)
`run_lco_eval.py` — hold out each camera view, train on the rest, test on the held-out camera
(516 clips pooled, relation-Transformer app-tokens+aux, 3-seed ensemble, 60 ep, blind val).

| Protocol | relation-Transformer | scene-only baseline |
|---|---|---|
| **Same-distribution** (paper 64-clip test) | 54/64 = 84.4 % (S1a) | **52/64 = 81.2 %** (≈ model → confounded) |
| **Leave-Camera-Out** (scene-robust, 516) | **75.6 % top-1 / 76.6 % macro** | **16.9 % / 10.4 %** (collapses) |

**Behaviour signal beyond scene = +58.7 pp** (75.6 vs 16.9 under LCO). Per-class LCO recall:
- **CarryOverload 100 %, OpenedPanel 93.0 %** — entity-object behaviours generalise near-perfectly to
  unseen cameras (forklift/panel entities are camera-invariant).
- **WalkwayViol 78.1 %, UnauthorInt 35.2 %** — the zone-relational confusion pair is the residual
  cross-camera challenge (zones are camera-specific); motivates cheap per-view zone calibration
  (the thesis's "virtual fence" future work).

**Interpretation.** Same-distribution accuracy is scene-confounded for *everyone* (model ≈ scene-only).
Under scene-robust LCO the scene shortcut collapses (16.9 %) while our entity-relation model keeps a large
genuine behaviour signal (75.6 %) — i.e. **the entity ontology's real payoff is scene-agnostic,
cross-camera generalisation**, the property a deployable factory system actually needs. The paper's
95.81 % (fixed cameras, no camera-holdout) almost certainly does not survive an LCO protocol — a concrete,
defensible critique + a sounder benchmark this thesis contributes.

## COMPLETED SCENE-BIAS BRACKET (2026-08-14) — 3 methods × 2 protocols
`run_wholeframe_eval.py` (whole-frame frozen features + per-frame MLP probe, mean-aggregate, 3 seeds):

| Method | same-dist top-1 | LCO top-1 | LCO macro |
|---|---|---|---|
| scene-only (camera→class) | 81.2 % | 16.9 % | 10.4 % |
| whole-frame (paper-style) | **81.2 %** | 52.5 % | 60.3 % |
| entity-relation (ours) | 84.4 % | **75.6 %** | **76.6 %** |

Per-class LCO recall — whole-frame vs ours: OpenedPanel 100/93, CarryOverload 100/100,
**WalkwayViol 28/78, UnauthorInt 13/35**.

**Conclusions (definitive):**
1. **Whole-frame same-distribution (81.2 %) == scene-only (81.2 %)** → the paper-style whole-frame
   approach adds nothing over camera-guessing at same-distribution; it is pure scene.
2. **Under LCO it collapses (52.5 %)**, entirely on the spatial-relational confusion pair (WalkwayViol
   28 %, UnauthorInt 13 %); object-visible classes (panel/forklift) stay ~100 % for everyone.
3. **Our entity-relation model is the only scene-robust method (75.6 % LCO)** — it carries the confusion
   pair cross-camera (78 %/35 %). The entity ontology's payoff is cross-camera deployability, not
   same-distribution top-1. The paper's 95.81 % would not survive this protocol.

## Recommendation
- **Do NOT build S3 to "beat 57/64"** — the number would be scene memorisation.
- **Do build S3 in the diagnostic framing** (whole-frame vs actor-crop) to *quantify* the scene ceiling,
  **and add the LCO evaluation** — together these are a stronger, more honest thesis result than any
  same-distribution top-1, and they turn a plateau into a rigorous benchmark critique.
- Ties directly to the thesis's existing "honest evaluation / data-bound" narrative.
