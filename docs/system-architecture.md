# System Architecture — P1 / P2 / P3

**Read this file before citing any number in this repo.** The single most common source of error
in this project's own internal reports has been attributing a number (accuracy, Jetson latency,
ECE) to the wrong pipeline. There are **three non-interchangeable pipelines**. They share the
general shape "YOLO spatial extractor -> token/feature sequence -> Transformer temporal head -> 4-
or 8-class softmax" (matching the architecture diagram in `PROPOSAL.txt`), but differ in what the
YOLO stage is trained on and how the temporal head consumes its output.

## Why three pipelines exist

They are chronological evolution, not parallel competing designs kept for comparison:

1. **P1** — the original approach: frozen COCO-pretrained YOLO features feed a multi-stream
   Transformer classifier. Fast to build (no detector training needed), reached first result.
2. **P2** — after review found P1 never reproduced the paper's actual mechanism (a per-frame
   *fine-tuned* behaviour detector), P2 is a from-scratch paper-faithful rewrite.
3. **P3** — an entity-detector + spatial-relation Transformer design, built after finding the
   detector in P1/P2's line was undertrained; P3 is the only pipeline with evidence at N > 64
   (k-fold over 516 pooled clips, leave-camera-out), and is the **current accuracy headline**.

## P1 — frozen-COCO YOLO -> multi-stream Transformer

**Code:** root `run_*.py` (e.g. `run_pairwise_contrastive.py`, `run_yoloworld_push.py`,
`run_safe_unsafe.py`, `run_temporal_fusion.py`, `run_scenario_cascade.py`, `run_finetune_e2e.py`),
`model_arch.py`-equivalent backbone hooks, `feature_dataset.py`, `temporal_video_loader.py`.

**Data flow:**
```
raw clip (1920x1080, 24fps)
  -> uniform frame sampling (T=16/32/48)
  -> YOLOv11s backbone, COCO-pretrained, frozen (or partially unfrozen for E2E variant)
  -> pooled spatial tokens (T x 4x4 grid x 768 channels)
  -> multi-stream Transformer head: raw tokens + 1st/2nd-order temporal derivatives
     + windowed statistics, learned gating -> Transformer encoder -> CLS token -> MLP -> softmax
  -> 4-class (unsafe subset) or 8-class prediction
```

The YOLO stage here never sees the behaviour labels — it is a generic COCO feature extractor, not
a detector trained on this task's classes. This is why the paper's mechanism (a detector trained
specifically to recognise behaviour-relevant boxes) is not actually replicated by P1.

**Best measured (per `plans/2026-09-06-thesis-2week-final/brief.md` §1):** 56/64 = 87.5% top-1,
89.28% macro (test-selected, deployed frozen ensemble); 53/64 = 82.8% top-1, 85.4% macro under the
honest blind protocol. 64-clip N, so CI width is approximately ±11 pp.

**This is the pipeline currently exported to and benchmarked on Jetson** (see
`deployment-guide.md`). All Jetson latency/FPS/power/soak numbers in this repo today are P1
numbers.

## P2 — paper-faithful rewrite

**Code:** `unsafenet/` package (`config.py`, `behaviour_dataset.py`, `train_frame.py`,
`train_temporal.py`, `convlstm.py`, `frame_cache.py`, `fusion.py`, `evaluate.py`, `metrics.py`).
Driven by `run_unsafenet_r2.py` at the root.

**Data flow:**
```
raw clip -> T=32 frame cache (691 clips, ~2.8 GB, cache_r2/)
  -> entity detection (sweep_results/entity_detector/detector_best.pt)
  -> ORB/RANSAC-registered zone geometry + person-motion streams
  -> 2 per-frame behaviour CNNs trained on all 8 classes (train_frame.py)
  -> 12 training-free aggregation members (incl. the paper's frame-reduction gate)
  -> 22 temporal heads (mean / BiLSTM / Transformer, train_temporal.py)
  -> bagged Caruana fusion selected on val -> one declared test touch
```

This is the pipeline that actually attempts the paper's core mechanism (a behaviour-aware
detector feeding a temporal classifier), evaluated under a frozen val protocol with a single test
touch.

**Measured:** 50/64 = 78.12% top-1, 80.81% macro, blind (95% CI [66.0, 87.5] top-1 /
[70.6, 90.2] macro, per `PROGRESS.md`); 55/64 test-selected. Weaker than P1/P3 but the most
methodologically honest of the three (frozen protocol, disclosed test touch).

## P3 — YOLO entity detector + spatial-relation Transformer (current accuracy headline)

**Code:** `build_relation_tokens.py` (token builder), `model_relation_head.py`
(`RelationTransformer`), `run_relation_transformer.py` (training, blind protocol),
`run_kfold_relation.py` (stratified k-fold + Nadeau-Bengio), `run_lco_eval.py` (leave-camera-out),
`run_full_eval.py` (multi-lens report generator).

**Data flow:**
```
raw clip -> sample T=16 frames
  -> entity detector (YOLO11/YOLO26, sweep_results/entity_detector/detector_best.pt)
  -> per-detection 10-dim token: [class_id, xc, yc, w, h, conf,
       iou_walkway, iou_restricted, foot_in_walkway, norm_dist]
     (last 4 dims computed against per-view zone polygons in annotation_guide/zones/,
      for person-class detections only; zero for non-person entities)
  -> up to 8 entity tokens per frame -> RelationTransformer (model_relation_head.py)
  -> 4-class softmax, blind decision (best-val-seed) + ensemble decision (seed-averaged)
```

Key distinction from P1: the detector stage is trained specifically on this task's entity/zone
ontology (not generic COCO features), and the Transformer reasons over **entity-to-zone spatial
relations** per frame rather than raw appearance tokens. Key distinction from P2: no ConvLSTM,
frame-level behaviour CNN is replaced by an object detector + geometric relation features.

**P3 is the only pipeline in this project with evidence at N > 64** (k-fold pools 516 clips;
leave-camera-out pools all 516 grouped by camera view).

### P3 headline decision rules — two different things, do not conflate

Every `logits_*.pt` artifact in `sweep_results/relation_tokens/` carries two predictions from the
same training run, and they are **different models**, not two views of one calibration step:

- `blind_pred` — the raw prediction of the single seed with the best held-out validation macro,
  selected before touching test. This is the pre-registration-compliant number the brief's
  headlines are built from.
- `ens_softmax` (argmax'd) — the softmax averaged across all trained seeds (an ensemble of N
  independently-trained models).

`blind_pred`'s own single-seed softmax was never persisted to disk, so no calibration number
(ECE, reliability diagram, conformal/selective-risk analysis) exists for the exact decision rule
behind the 54/64 headline. Any such figure computed from `ens_softmax` characterises the
**ensemble** decision (52/64 for the yolo26 headline artifact), a different rule. State the basis
explicitly whenever citing calibration numbers for P3; do not pair the 54/64 headline with an ECE
computed from `ens_softmax` in the same table row.

### P3 canonical numbers (traced to specific artifacts)

| Claim | Value | Canonical artifact |
|---|---|---|
| Same-distribution 64-clip, blind | 54/64 = 84.4% top-1, macro 88.4% | `sweep_results/relation_tokens/logits_yolo26.pt`, key `blind_pred` |
| Same-distribution ECE (ensemble basis, NOT the 54/64 blind rule) | 0.1009 (15-bin) | `logits_yolo26.pt`, key `ens_softmax`, via `eval_utils/calibration.py` |
| k-fold, 516 pooled clips | 417/516 = 80.8 ± 3.1% top-1, 84.0% macro | `sweep_results/relation_tokens/kfold_oof_softmax_yolo26app_.pt`, generated by `run_kfold_relation.py --tag yolo26app_` |
| Leave-camera-out, 516 pooled clips | **77.7% top-1 / 81.2% macro** (LOCKED, canonical) | `sweep_results/relation_tokens/canonical/lco_seeds5_yolo26L1_CANONICAL.log` — 5 seeds, 60 epochs, 8 camera-groups. Supersedes the 78-82% range and all 3-seed runs. |
| LCO scene-appearance baseline (transferable) | 53.1% top-1 / 64.4% macro | same log; logreg on whole-frame MobileNetV3 features, fit on training cameras only |
| **LCO honest margin** | **+16.9 pp macro, 95% bootstrap CI [+12.5, +21.2]** | same log, n_boot=5000 over 516 clips |

**In-flight items — do not write as settled facts anywhere:**
- ~~**LCO single headline value**~~: **LOCKED 2026-09-06** at 77.7% top-1 / 81.2% macro (canonical
  5-seed run). The 78-82% range is superseded — do not report it.
- **k-fold Nadeau-Bengio significance** (relation-Transformer vs scene-only baseline):
  **SETTLED for the YOLO11 track (2026-09-06).** Canonical `--repeats 10` run, 50 folds, df=49:
  mean Δ = 3.6 pp, t = 1.95, **p = 0.0574 — not significant**; relation top-1 80.7 ± 3.6 %,
  macro 85.1 %, NB 95 % CI 77.0–84.4 %, scene-only 77.1 ± 2.5 %. Log:
  `sweep_results/relation_tokens/canonical/kfold_repeats10_yoloapp_YOLO11.log`. The earlier
  0.0375/0.1013 spread was single runs at df=4 — do not cite either. Write "p = 0.0574 (n.s.)";
  never "marginally significant".
  **SETTLED for the YOLO26 headline too (2026-09-06):** canonical `--repeats 10`, 50 folds, df=49 —
  top-1 80.8 ± 4.0 %, macro 84.2 %, macro-F1 83.5 %, κ 0.741, MCC 0.744, NB 95 % CI 76.7–85.0 %,
  **Δ = 3.7 pp, t = 1.93, p = 0.0600 (n.s.)**. Log:
  `sweep_results/relation_tokens/canonical/kfold_repeats10_yolo26app_HEADLINE.log`. The earlier
  single-run p = 0.134 is superseded. The two tracks agree to 0.003 in p, and 80.8 vs 80.7 % in top-1
  on 50 folds each — which is what now evidences "the pipeline is not detector-limited".
- **Nadeau-Bengio correction is mandatory, not optional.** On both canonical runs the naive fold SE is
  0.51 vs 1.87 (YOLO11) and 0.58 vs 2.12 (YOLO26) corrected — a stable 3.7x on this design — **any k-fold CI in this repo computed without it is ~3.7x too narrow.**
- ~~**Cross-camera baseline margin**~~: **RESOLVED 2026-09-06.** The old "+60.3 pp vs scene-only"
  comparison was against a *constant* predictor and is withdrawn. The rebuilt transferable baseline
  (scene-appearance logreg) gives an honest **+16.9 pp macro, CI [+12.5, +21.2]**. Proof the old
  baseline was the constant: the rebuilt `prior` row scores 16.5%, reproducing the old "16.9%".
  **Mandatory caveat wherever LCO is cited:** the 8 "cameras" are a KMeans(8) proxy on background
  descriptors — Unsafe-Net publishes no camera IDs.

## Which pipeline backs which claim — quick reference

| Evidence axis | Backing pipeline |
|---|---|
| Same-paper-protocol 64-clip accuracy | P3 (current headline) — P1 and P2 numbers also exist and are reported for context, never as the primary claim |
| Statistical reliability at scale (k-fold, ±3pp CI) | P3 only — P1/P2 have no N>64 evidence |
| Cross-camera generalisation (LCO) | P3 only |
| Safety metrics / calibration / cost-sensitive scoring | P3 (via `eval_utils.evaluate_suite`, see `codebase-summary.md`) |
| **Jetson edge deployment (latency, FPS, power, soak)** | **P1 — exclusively.** No P3 Jetson benchmark exists yet. Never attribute a Jetson number to P3. |
| Paper-mechanism faithful reproduction | P2 |

## Unresolved questions

1. Whether the thesis ultimately presents P1/P2/P3 side-by-side with explicit pipeline-attribution
   labels in every table, or converges on P3 as the sole headline with P1/P2 relegated to an
   ablation/history appendix, is a framing decision noted as open in
   `plans/2026-09-06-thesis-2week-final/plan.md` ("Unresolved" section) — not settled as of this
   writing.
2. The exact CLI invocation that produced the canonical `logits_yolo26.pt` artifact is not logged
   anywhere on disk (inferred as `--tag yolo26app_` from file-modification-time proximity to the
   token build, per the w2-number-integrity-fixes report). Treat the pipeline description above as
   architecturally accurate; treat the precise hyperparameters as best-effort reconstruction.
