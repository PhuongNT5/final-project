# Chapter 7 — Multi-Entity Spatial-Relation Recognition and Scene-Robust Evaluation

> **New contribution chapter (2026).** Consolidates the multi-entity ontology + YOLOv11/Transformer
> pipeline and the scene-confounding analysis of the Unsafe-Net benchmark. Additive to the existing
> deployment-led thesis; integration notes in §7.8. All numbers reproduce from committed scripts/artifacts.

## 7.1 Motivation

The published Unsafe-Net pipeline [1] pairs a YOLO v4 *behaviour detector* (trained on 2,262 per-frame
behaviour-bounding-boxes) with a ConvLSTM temporal classifier, reporting 95.81 % macro on the 4-class
unsafe test subset. This chapter re-implements that pipeline from scratch with two modernisations —
**YOLOv11s (→ YOLO26) replacing YOLO v4** and a **box-token Transformer replacing ConvLSTM** — but with a
different supervision philosophy: instead of labelling *behaviours* per frame, we label the **entities**
that compose a behaviour (people, zones, object-states) and let the Transformer infer the behaviour from
their spatial relations. The chapter's central empirical finding is that the Unsafe-Net benchmark is
**scene-confounded**: same-distribution accuracy largely measures which camera a clip came from, not the
behaviour. Under a scene-robust (leave-camera-out) protocol the entity-relation approach is the only
method that generalises, and the paper's 95.81 % is shown to be substantially scene-inflated.

## 7.2 Multi-entity annotation ontology

Rather than "behaviour-as-object", we annotate an **8-class entity ontology**: `Person_Authorized`
(green vest), `Person_Unauthorized` (red-black vest), `Safe_Walkway` and `Restricted_Zone` (floor
polygons), `Panel_Closed/Opened`, `Forklift_Safe_Load` (≤2 blocks) / `Forklift_Overload` (≥3). Behaviour
is *derived* downstream (e.g. Walkway-Violation = a person whose foot-point lies outside `Safe_Walkway`;
Unauthorized-Intervention = `Person_Unauthorized` inside `Restricted_Zone`).

**2,262 frames** were sampled from the 566 training clips only (the 64-clip test set untouched),
viewpoint-balanced across the 8 fixed cameras and over-weighted on the WalkwayViol↔UnauthorInt confusion
pair, and annotated in CVAT (rectangles for entities, polygons for zones; foot-point-tight person boxes).
The train/val split is **clip-level** (no clip's frames cross the split), fixing the leakage of the
earlier V10 pilot (`val: images/train`). The frame count matches the paper's YOLO training set (2,262).

## 7.3 Stage A — entity detector

A COCO-pretrained **YOLOv11s** was fine-tuned (two-stage: freeze-backbone 30 ep → unfreeze 100 ep, SGD
lr 0.01→0.001, imgsz 800, `hsv_h=0` to preserve the vest-colour signal). Training ran on a CUDA GPU
(Colab) after establishing that Apple-MPS training is numerically broken on macOS 14 (silent Adam
`addcmul_` failure — see `researcher-mps-training-fix.md`); inference runs locally.

**Detector validation (343-image held-out val):** overall **mAP@50 = 0.969**, mAP@50-95 = 0.894.
Per-class mAP@50: Person_Authorized 0.938 (R 0.921), Person_Unauthorized 0.928 (R 0.937), Panel_Closed
0.989, Panel_Opened 0.972, Forklift_Safe 0.995, Forklift_Overload 0.995. The rare `Person_Authorized`
class (365 instances) was learned well — confirming the vest-colour cue survives with `hsv_h=0`.

## 7.4 Stage B — box-token spatial-relation Transformer

Per detected entity per frame we build a token
`[class, x, y, w, h, conf, IoU_walkway, IoU_restricted, foot_in_walkway, norm_dist, appearance(64)]`
(geometry computed against the per-view zone polygons; appearance = MobileNet crop-embedding). A
[CLS]+box-token Transformer (2 layers, 4 heads, ~0.5 M params) classifies the 4 unsafe behaviours; a
per-frame auxiliary behaviour head propagates the clip label to frames (452 → ~7 k supervision signals).
All configurations are evaluated under a **blind held-out protocol** (epoch/seed selected on a
train-carved val slice; 64-clip test touched once); the small val set (68 clips) makes single-seed
selection high-variance, so we report the **7-seed ensemble** as the robust estimator.

**Ablation (blind seed-ensemble, 64-clip test):**

| Config | closes gap-cause | top-1 | macro |
|---|---|---:|---:|
| v1 geometry only | weak zone signal | 51/64 | 86.1 % |
| v2a + person↔object proximity | (no gain) | 49/64 | 83.0 % |
| v2b + appearance | lossy tokenisation | 52/64 | 84.6 % |
| **S1a + per-frame supervision** | **supervision placement** | **54/64** | **88.4 %** |
| S1b + behaviour frame-gate | (no gain, τ→0) | 53/64 | 87.6 % |

Each rung targets a specific cause identified by the paper back-check (§7.6). The best config (**S1a**,
appearance + per-frame auxiliary loss) reaches **54/64 (84.4 % top-1, 88.4 % macro)** — above the paper's
no-bbox macro baseline (85.78 %). The frame-gate (S1b) yielded no gain (τ→0, kept-frac ≈ 0.99): our
clips are already single-action segments, so the paper's frame-reduction step has nothing to reduce.

## 7.5 Full multi-lens evaluation (best model, 64-clip test)

> **Two decision rules, and previously three sources — read the basis column.** Metrics that need only
> hard predictions are computed from `blind_pred` (the val-selected single seed, **54/64**). Metrics that
> need probabilities can only be computed from `ens_softmax` (the **seed-averaged ensemble, 52/64**),
> because the blind seed's own softmax was never persisted (brief.md S4). These are two different
> predictors and the table must not be read as one model's joint scorecard.
>
> **Correction (2026-09-06):** the ROC-AUC (0.953), PR-AUC (0.874) and ECE (0.104) previously printed
> here were **not from this pipeline at all** — they reproduce `logits_s1a.pt` (≡ `logits_yolo11_baseline.pt`,
> the YOLO11 track) to four decimals, not `logits_yolo26.pt`. Recomputed from the correct artifact below.

| Metric | value | basis | Metric | value | basis |
|---|---|---|---|---|---|
| top-1 | 84.4 % (54/64), BCa CI [71.9, 90.6] | blind | Cohen's κ | 0.775 | blind |
| macro-recall | 88.4 %, BCa CI [77.8, 94.4] | blind | Matthews MCC | 0.785 | blind |
| macro-F1 | 85.6 % | blind | ROC-AUC (ovr-macro) | **0.950** | ensemble (52/64) |
| balanced-acc | 88.4 % | blind | PR-AUC (ovr-macro) | **0.869** | ensemble (52/64) |
| ECE (15-bin) | **0.101** | ensemble (52/64) | cost (miss×10) | 100 | blind |

Per-class precision / recall / FNR: WalkwayViol 96.2 / 78.1 / 21.9 · UnauthorInt 58.8 / 90.9 / 9.1 ·
OpenedPanel 84.6 / 84.6 / 15.4 · CarryOverload 100 / 100 / 0. Confusion + reliability figures in
`figures/relation_confusion.png`, `figures/relation_reliability.png`. **Action-recognition time:** the
Transformer classifier runs at **0.9 ms/clip**; the full detector+classifier pipeline ≈ **200 ms/clip**
on an M3 (detector-bound), vs the paper's 0.14 s on a 2×GTX-1080 workstation.

Two statistics are decisive and honest: **McNemar vs a scene-only baseline is not significant**
(p = 0.815) and **vs the paper's no-bbox baseline Δ = −4.7 pp is not significant** (p = 0.435). i.e. on
the *same-distribution* protocol our model is statistically indistinguishable from camera-guessing — which
motivates §7.7.

## 7.6 Why the gap to 95.81 % — a supervision-placement diagnosis

Back-checking the paper: its ConvLSTM trains on the **same 452/64 split** we use; the +10 pp from
85.78 % → 95.81 % is **entirely** the YOLO-v4 behaviour-frame-reduction. YOLO v4 is a *per-frame
behaviour detector* (2,262 behaviour-bboxes, 95.66 % mAP) that selects behaviour frames and splits clips
into single actions. The paper thus spends its 2,262-frame budget on the *classification answer* at the
frame level; we spend the same budget on *entities*, deferring discrimination to a Transformer trained on
452 clip-level labels (~5× less, less granular). This is a **supervision-placement** effect, not an
architectural deficiency — S1a (adding frame-level supervision) recovered ground exactly as predicted.

## 7.7 The Unsafe-Net benchmark is scene-confounded (central finding)

The dataset has **8 fixed cameras**, and camera view strongly predicts behaviour (3 of 8 views are
single-class). A **scene-only baseline** — predict the behaviour from the camera view alone, ignoring the
video — scores **52/64 = 81.2 %** on the test set. A **whole-frame classifier** (frozen features + probe,
the paper's scene-dependent style) scores **81.2 %** same-distribution — *identical* to camera-guessing,
i.e. it adds nothing over knowing the camera.

We therefore evaluate all methods under a **leave-camera-out (LCO)** protocol — train on some cameras,
test on held-out cameras — the honest test of behaviour generalisation:

| Method | same-distribution top-1 | **leave-camera-out top-1** | LCO macro |
|---|---:|---:|---:|
| scene-appearance (trained, transferable) | — | 53.1 % | 64.4 % |
| whole-frame (paper-style, trained) | 81.2 % | 52.5 % | 60.3 % |
| **entity-relation (ours)** | 84.4 % | **77.7 %** | **81.2 %** |
| *prior / constant floor (not a model)* | *—* | *16.5 %* | *10.1 %* |

**Honest margin: +16.9 pp macro over the scene-appearance baseline, 95 % bootstrap CI
[+12.5, +21.2] pp** (n_boot = 5 000, resampled over the 516 pooled clips). Canonical run:
5 seeds, 60 epochs, 8 camera-groups, `sweep_results/relation_tokens/canonical/
lco_seeds5_yolo26L1_CANONICAL.log`.

Two independently constructed transferable baselines — a logistic regression on whole-frame
MobileNetV3 background features (53.1 %) and a trained whole-frame classifier (52.5 %) — land
within 0.6 pp of each other. That agreement is what makes the margin credible: it is not an
artifact of one baseline being badly built.

**The "prior" row is a floor, not a competitor, and must never be used as the comparison.** A
camera→class lookup is undefined for a camera never seen in training, so an earlier version of
`run_lco_eval.py` fell back to predicting the training **global-majority class** — a *constant*.
Its 16.5 % top-1 / 10.1 % macro is that constant's arithmetic, and it reproduces the 16.9 % that
an earlier draft of this chapter reported as a "scene-only collapse". **The former "+58.7 pp over
scene-only" claim is withdrawn**; it compared a trained model against a constant. Per-view-majority,
the informative baseline used under random k-fold, is likewise **undefined** under LCO (the held-out
view contributes zero training clips), which is precisely why the k-fold 77.1 % baseline and these
numbers are not comparable estimators.

Per-class LCO recall (**scene-appearance vs ours**, canonical run): OpenedPanel 88.0/88.7,
CarryOverload 100/100, **WalkwayViol 17.6/70.5, UnauthorInt 51.9/65.7**. Object-visible behaviours (panel, forklift) generalise for everyone; the
**spatial-relational confusion pair - the whole point of the task - collapses for the whole-frame
approach and is carried by our entity+geometry reasoning**. That per-class pattern, not the aggregate,
is the strongest evidence here: the two classes that a scene-memorising model cannot get right are
exactly the two that entity geometry is designed to separate.

**Implications.** (i) **The same-distribution metric is scene-inflated on this benchmark.** With these
view<->class correlations and no camera holdout, a model can score 81.2 % by identifying the camera. That
is a claim about the *metric*, and it is directly measured here. It is **not** a claim that the paper's
system would fail under LCO: their weights and per-clip predictions are unreleased, so that experiment
cannot be run, and asserting its outcome would be speculation dressed as a result. State the metric
critique; leave their number alone. (ii) The entity ontology's real payoff is **scene-agnostic,
cross-camera generalisation** - the property a deployable factory-safety system needs, since cameras are
added, moved, and re-aimed. Same-distribution top-1 is the wrong metric on this benchmark; scene-robust
accuracy is the right one, and there our approach leads the trained whole-frame baseline by +23.1 pp
(provisional, pending the canonical run).

## 7.8 Limitations and integration

- **Residual cross-camera confusion pair.** UnauthorInt LCO recall is **65.7 %** and WalkwayViol
  **70.5 %** (canonical 5-seed run) — the weakest two classes, and both are the spatially-defined ones:
  zones are camera-specific, so a wholly unseen camera lacks a zone reference. A per-view zone
  calibration (one polygon per camera — the thesis's "virtual fence" future work) directly addresses
  this and is cheap (~5 min/camera). *(An earlier draft quoted 35 % here from a 3-seed run; the
  canonical run nearly doubles it. Seed count mattered a great deal on this class — which is itself
  the argument for the 5-seed floor, and a caution against reading any single-seed per-class recall
  in this project as stable.)*
- **Camera grouping is a proxy, not ground truth.** Unsafe-Net publishes no camera IDs, so the eight
  "cameras" are a KMeans(8) clustering of per-clip static-background descriptors
  (`tools/build_scene_grouping.py`). Every LCO number here is therefore bounded by how well that proxy
  tracks real cameras — state this wherever LCO is cited; it is the single biggest caveat on the
  chapter's headline result.
- **Small N.** 64-clip test, 68-clip val → wide CIs and high seed variance; the seed-ensemble and LCO
  pooling (516 clips) mitigate this. Statistics reported with BCa/Wilson CIs and McNemar throughout.
- **MT3 (edge) deferred.** Porting the YOLO26 detector + Transformer to Jetson TensorRT-INT8 (latency,
  FPS, multi-stream, energy) is the remaining deployment step; the earlier pipeline's Jetson artifacts
  stand as the deployment baseline.

**Integration map (for final compilation):** §7.2–7.4 extend Chapter 3 (methodology); §7.5–7.7 extend
Chapter 4 (experiments) and should be cross-referenced from Chapter 6's honest-reading — the scene-bias
finding strengthens the thesis's data-bound diagnosis into a *supervision-placement + scene-confounding*
diagnosis, and reframes the accuracy result as a scene-robustness advantage rather than a top-1 deficit.

## 7.9 Reproducibility
Scripts: `build_relation_tokens.py`, `augment_relation_tokens.py`, `build_wholeframe_features.py`,
`model_relation_head.py`, `run_relation_transformer.py` (`--aux_w`, `--frame_gate`), `run_lco_eval.py`,
`run_wholeframe_eval.py`, `run_full_eval.py`; detector `sweep_results/entity_detector/detector_best.pt`;
metrics `sweep_results/relation_tokens/full_eval.json`; reports under
`plans/2026-07-09-spatial-relation-yolo26-3month/reports/`.

## Unresolved questions
1. Would per-view zone calibration lift the cross-camera confusion pair (UnauthorInt LCO 65.7 %, up from an old 3-seed 35 %) materially?
2. Does the paper's 95.81 % survive an LCO re-evaluation (requires their per-camera labels — filed as a request)?
3. Is 2,262 entity frames vs 2,262 behaviour frames a fair head-to-head, or should a behaviour-detector control (S3) be run to bound the scene ceiling under LCO?

## References
[1] O. Önal and E. Dandıl, "Unsafe-Net: YOLO v4 and ConvLSTM based computer vision system for
real-time detection of unsafe behaviours in workplace," *Multimedia Tools and Applications*, vol. 84,
pp. 34967–34993, 2024.
