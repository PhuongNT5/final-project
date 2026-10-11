# System Architecture — P1 / P2 / P3

> ## ⚠️ SUPERSEDED — NOT THE SUBMITTED RESULT (as of 2026-10-05)
>
> **The submitted thesis is the sole authority for every number.** It lives in `report/vn/`
> (11 files, Vietnamese). This file is a **development log** kept for provenance; it was written
> while the project's headline was still the YOLO lane, and it has **not** been rewritten to match
> the submission.
>
> What changed, in one paragraph: the thesis's headline configuration is **not** a YOLO pipeline.
> It is **B1** — a frozen CLIP ViT-B/32 tower feeding a 201,090-parameter temporal head — scoring
> **84.2 % balanced accuracy** on iSafetyBench (1,100 clips, binary) under **leave-source-out**
> folds grouped by source video, reported against a pre-registered **63.3 %** count-rule floor.
> The entity-decomposition hypothesis that motivated the YOLO lane was tested as a pre-registered
> gate (G14) and **failed** (−0.5 pp, cluster CI [−5.5, +5.0] containing 0, McNemar p = 0.683).
> Eight pre-registered gates were adjudicated: **two passed, six failed.**
>
> **Numbers in this file that the thesis does NOT stand behind** — do not cite them, from here or
> from anywhere else: any "+3.5 pp macro win"; "57/64"; "95.7 % mAP"; the P3 cadence figures
> "331 ms / 3.02 dec/s / 3.91 J" and any claim that P3 "passes" or "fails" the 3.0 dec/s contract
> as a single verdict (the composed measurement of 2026-10-03 is clip- **and** clock-dependent);
> and any per-stage **sum** presented as a system cadence. The thesis's own Chapter 7 documents why
> each was withdrawn.
>
> Cross-lane comparison is also forbidden: the 1,100-clip binary iSafetyBench numbers and the
> 64-clip 4-class Unsafe-Net numbers are different corpora, different tasks and different metrics.
> They are never averaged and never set against each other (thesis §6.1.1).


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
   detector in P1/P2's line was undertrained; P3 is the only pipeline of these three with evidence
   at N > 64 (k-fold over 516 pooled clips, leave-context-out). **P3 is no longer the accuracy
   headline.** A fourth lane, **B1/R48** (frozen CLIP ViT-B/32 tower + 201,090-parameter temporal
   head), is the submitted headline and the deployed graph — 84.2 % balanced accuracy on
   iSafetyBench under leave-source-out, and the only lane whose composed cadence is measured on the
   device (5.978 decisions/s, thesis §5.4.4). P3 **has** since been measured on the Jetson too
   (composed, 2026-10-03), and that measurement is clip- and clock-dependent; see
   `deployment-guide.md` and thesis §5.5.

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
`train_temporal.py`, `convlstm.py`, `frame_cache.py`, `fusion.py`, `evaluate.py`, `metrics.py`,
`behaviour_clip.py`). Driven by `run_unsafenet_r2.py` at the root.

**Data flow:**
```
raw clip -> T=32 frame cache (691 clips, ~2.8 GB, cache_r2/)
  -> entity detection (sweep_results/entity_detector/detector_best.pt)
  -> behaviour detector, nc=8 (default runs/detect/sweep_results/r2/beh_ctx/weights/best.pt)
     └─ detector_gate_status(): peak mAP50(B) read from that run's own results.csv, stamped
        into every member artifact alongside gate_status
  -> ORB/RANSAC-registered zone geometry + person-motion streams
  -> 2 per-frame behaviour CNNs trained on all 8 classes (train_frame.py)
  -> 12 training-free aggregation members (incl. the paper's frame-reduction gate)
  -> 22 temporal heads (mean / BiLSTM / Transformer, train_temporal.py)
  -> bagged Caruana fusion selected on val -> one declared test touch
```

This is the pipeline that actually attempts the paper's core mechanism (a behaviour-aware
detector feeding a temporal classifier), evaluated under a frozen val protocol with a single test
touch.

**Detector quality gate (G22):** `behaviour_clip.detector_gate_status()` reads peak mAP50(B) from
detector run's `results.csv` and compares against floor 0.85 / gate threshold 0.90. Every saved
`temporal_det*.pt` artifact is stamped with `detector_weights`, `detector_map50_peak`, and
`gate_status` fields so downstream tables cannot present below-floor detector results as gated
measurements. The figure is read from the run's own `results.csv`, never hand-entered; the measured
values for each trained detector live in the plan folder, not here.

**Measured:** read from `sweep_results/r2/report_fusion.json` (the fused P2 headline), blind 64-clip
test: **51/64 = 79.69% top-1, 81.59% macro**, 95% CI [68.8, 89.1] top-1 / [71.6, 90.9] macro;
55/64 test-selected. Weaker than P1/P3 but the most methodologically honest of the three (frozen
protocol, disclosed test touch).

> `PROGRESS.md:128` and earlier revisions of this line report 50/64 = 78.12% / 80.81% macro with CI
> [66.0, 87.5] / [70.6, 90.2] for this same row, citing this same artifact. The artifact on disk
> (untracked, unmodified since 2026-09-05) carries exactly one test block and it is the one quoted
> above; none of that other quadruple appears in it. Do not re-propagate the 50/64 figures for P2
> fusion without re-deriving them. Note 50/64 **is** correct for the bare ConvLSTM member
> (`temporal_convlstm.pt`, macro 75.64%) — a different system that coincidentally shares the top-1,
> which is the likely origin of the mix-up. The frozen thesis text does not contain either figure.

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
| k-fold, 516 pooled clips | fold-mean **80.8 ± 4.0%** top-1; OOF-pooled **421/516 = 81.6%**, macro **84.2%**, κ 0.741, MCC 0.744 | `sweep_results/relation_tokens/canonical/kfold_repeats10_yolo26app_HEADLINE.log` (5×10 folds) and `kfold_oof_softmax_yolo26app_.pt` (argmax of `ens_softmax`). Corrected 2026-09-09 — the row previously read `417/516 = 80.8 ± 3.1%, 84.0% macro`, which matched neither artifact. Fold-mean 80.8% and OOF-pooled 81.6% are two different estimators; quote whichever, never mix their cells. |
| Leave-camera-out, 516 pooled clips | **77.7% top-1 / 81.2% macro** (LOCKED, canonical) | `sweep_results/relation_tokens/canonical/lco_seeds5_yolo26L1_CANONICAL.log` — 5 seeds, 60 epochs, 8 camera-groups. Supersedes the 78-82% range and all 3-seed runs. |
| LCO scene-appearance baseline (transferable) | 53.1% top-1 / 64.4% macro | same log; logreg on whole-frame MobileNetV3 features, fit on training cameras only |
| **LCO honest margin** | **+16.9 pp macro, 95% bootstrap CI [+12.5, +21.2]** | same log, n_boot=5000 over 516 clips |

### Benchmark integrity — the provable 63/64 ceiling

`run_duplicate_audit.py` hashes every clip: 691 videos → 658 distinct md5. **32 groups of byte-identical
videos carry ≥2 different class labels** (26 of them WalkwayViolation↔UnauthorizedIntervention). The `.npy`
frame sidecars are identical too, so no hidden annotation distinguishes the members.

Two consequences for the evaluated 64-clip unsafe test set:

| Fact | Value | How to re-verify |
|---|---|---|
| **Ceiling on the 64-clip benchmark** | **63/64 = 98.44%** | `data/test/0_safe_walkway_violation/0_te8.mp4` and `data/test/1_unauthorized_intervention/1_te1.mp4` share md5 `de03a3421d18b5f7f6d209f89b0c35f3` under different labels. A deterministic classifier maps identical bytes to one label, so it is wrong on exactly one of the two. `md5 data/test/*/0_te8.mp4 data/test/*/1_te1.mp4` |
| Test clips sitting in a contradictory group | **8 of 64 = 12.5%** | `unsafenet/dedupe.py::_groups()` |
| …of those, with a byte-identical TRAIN twin (cross-split leak) | 5 — `0_te7`, `0_te10`, `0_te11`, `1_te4`, `1_te5` | same |

`0_te23.mp4` ≡ `4_te9.mp4` (md5 `bef275463d146ad04cf9df6e00585119`) is a second test-test collision but class 4
is a *safe* class, outside the 4-class unsafe task, so it forces no error inside the 64.

**Reporting standard for the 64-clip set — three numbers, never merged:**
1. **raw 64** — directly comparable with the paper's 62/64.
2. **de-contaminated 59** — drop the 5 test clips whose exact bytes appear in train under another label.
3. **ceiling 63/64** — the provable upper bound for *any* deterministic classifier on this benchmark.

The third reframes the headline gap: the paper's 62/64 = 96.88% sits exactly one clip below the benchmark's
theoretical maximum, on a benchmark where 12.5% of the test clips are label-corrupted.

### Dedupe ablation (`--dedupe drop_all`) — contamination and confound are separable

`unsafenet/dedupe.py` drops every TRAIN member of a contradictory group (51 of 516 pooled clips); the 64-clip
test set is never filtered. Log: `sweep_results/relation_tokens/dedupe_drop_all/`.

| Protocol | `--dedupe none` (headline) | `--dedupe drop_all` |
|---|---|---|
| k-fold 5×10 vs scene-only | 80.8 ± 4.0%, NB Δ=+3.7pp, **p=0.0600 n.s.** | 86.6 ± 3.1%, κ 0.802, NB Δ=**+6.0pp**, t=3.39, **p=0.0014** |
| LCO (8 camera-groups) | 77.7% top-1 / 81.2% macro, honest margin +16.9pp [+12.5,+21.2] | 75.9 / 79.3, honest margin **+17.7pp** [+12.9,+22.4] |
| Same-distribution blind 64 | **54/64 = 84.4%**, macro 88.4% | **54/64 = 84.4%**, macro 87.3% (`logits_yolo26_drop_all.pt`) |
| vs no-pixel camera-ID floor | floor 52/64, McNemar **p=0.804** | floor 52/64, McNemar **p=0.804** (`scene_confound_drop_all_matched.json`) |

The two blind predictors are genuinely different — `blind_pred` differs on 5 of 64 clips, discordant 2–2,
blind seed 0 vs 1 — yet score identically. Meanwhile the confound *strengthens* after dedupe (Cramér's V
0.784→0.799, I(view;label) 1.101→1.175 bits, 59.2%→63.1% of label entropy), because the dropped clips are
precisely the ones that weakened the camera↔label association.

**Conclusion to carry into the thesis:** dedupe sharpens the margin over baselines under both resampling
protocols but moves the same-distribution 64-clip number by exactly zero. Label contamination is not what
caps the 64-clip benchmark — the camera confound is. The two criticisms are independent and this ablation
separates them.

All headline numbers elsewhere in this document remain `--dedupe none` by design: that is the raw benchmark,
comparable with the paper. The deduped run is reported alongside as an ablation, never substituted in.

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
2. ~~The exact CLI invocation that produced the canonical `logits_yolo26.pt` artifact is not logged
   anywhere on disk.~~ **RESOLVED 2026-09-06.** Recovered by pre-registered reproduction (brief.md S4):
   `run_relation_transformer.py --tag yolo26app_ --seeds 5 --aux_w 0.5 --epochs 120`, blind seed 1;
   reproduces the stored artifact bit-for-bit (`ens_softmax` max abs diff 0.00e+00). The hyperparameters
   above are exact, not reconstructed.
3. Dedupe ablation and the provable 63/64 benchmark ceiling: both settled and written up above
   ("Benchmark integrity" and "Dedupe ablation"). Nothing open here.
