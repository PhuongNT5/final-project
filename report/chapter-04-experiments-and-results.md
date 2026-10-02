# Chapter 4 — Experiments and Results

> ### Pipeline attribution — read before citing any number in this chapter
>
> This project evolved through **three non-interchangeable pipelines** (full map:
> `docs/system-architecture.md`). Chapters were written at different points in that evolution, so
> their headline accuracies are **not** competing estimates of one system — they describe different
> systems, and mixing them is the single most common error this project has made internally.
>
> | | Pipeline | Best cited | Blind/honest | Protocol |
> |---|---|---|---|---|
> | Ch1, Ch3, Ch4 | **P1** frozen-COCO YOLO -> multi-stream Transformer | 56/64 (87.5 %) | **53/64** (82.8 %) | same-dist, N=64 |
> | Ch5 | **P1** as exported to Jetson | 55/64 (TRT engine, as recorded in `benchmarks-jetson-*.json`) | **53/64** | same-dist, N=64 |
> | Ch7 + Abstract | **P3** entity detector + spatial-relation Transformer | **54/64** (84.4 % **top-1**, 88.4 % macro) | 54/64 | same-dist N=64, **plus k-fold and LCO at N=516** |
> | Ch5 §5.6.9 | **R48** whole-frame CLIP ViT-B/32 -> B1 temporal Transformer — the engine actually on the device | **84.2 % balanced accuracy** (B1, N=1,100) | same number: the protocol cannot leak a source, so there is no separate honest column | **P-D LSO-1100**, folds grouped by source video, N=1,100 |
>
> **The headline is the R48 lane: CLIP + temporal Transformer under LSO-1100.** It is the only
> accuracy in this thesis measured on all 1,100 clips under a protocol that cannot put one source
> video in both the train and the evaluation fold, and it is the model that was exported, deployed
> and measured on the Jetson (§5.6.9). YOLO's role in that lane is **localisation** — the system
> must say *where*, not only *whether* — and localisation latency is timed separately, never summed
> into the class decision. P3's 54/64 is the entity-relation lane's result at N=64 and is what Ch7
> and the abstract report; it keeps its own evidence beyond N=64 (k-fold and LCO at N=516). P1's
> 56/64 is **test-selected** (the ensemble was chosen using the test clips) and must never be
> presented as the result; its honest blind counterpart is 53/64. Chapters 1-6 are retained as the
> experimental record of how the design was arrived at, not as the final claim.
>
> **Caution — "84.4 %" denotes two different quantities here. Never write it bare.** 84.4 %
> *top-1* (P3, blind same-dist split, N=64) is Ch7's figure. 84.4 % *balanced accuracy* (B1w,
> LSO-1100, N=1,100) is the R48 lane's best head — and B1w is **not** the head on the device. The
> deployed engine is B1 (201,090 params x 3 seeds), whose own figure is **84.2 %**; B1 was chosen on
> the pre-registered *edge* criterion, smallest model, not on accuracy, so its own number is the one
> to quote and never max(B1, B1w). §5.6.9 quotes 84.2 % throughout for that reason
> (`plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-10-01-two-different-84-4-percent.md`).
>
> **"84.2 %" likewise denotes two different quantities. Never write it bare either.** 84.2 %
> *balanced accuracy* (B1, LSO-1100, N=1,100, binary) is the deployed figure above. 84.2 % *macro*
> is P3's k-fold mean over 516 ESTA clips on the 4-class task (Ch7 §7.7.1). Different corpus,
> different task, different metric: the two must never be compared, summed, or carried into one
> sentence without their qualifiers.

Objective 2 of this thesis is to train and evaluate the YOLO–Transformer
on Unsafe-Net and compare against the 96.88 % top-1 (95.81 % macro) baseline reported by
Önal & Dandıl (2024) [17]. This chapter documents all ten experiment
tracks conducted over the course of the project, each constituting a
principled attempt to close the accuracy gap via a different architectural
or training hypothesis. Section 4.1 states the evaluation protocol.
Section 4.2 summarizes the 10 tracks and details the 3 best-of-class
results (V7 best macro, V9 best top-1, V10 negative-result). Section 4.3
reports the best-case result (**56/64 ensemble, 89.28 % macro** —
test-selected; the previously reported 57/64 calibrated figure is
re-classified in §4.7.7 as test-set tuning), while **§4.7.11 establishes the frozen single-head pipeline blind at
≈ 52/64 (≈ 81 % top-1, macro ≈ 84 %); the best zero-bbox configuration
(hybrid + frame-gate, §4.6.2) lifts this to 53/64 (82.8 % top-1,
85.4 % macro) — this chapter's (P1's) defensible best, at macro parity with the
no-bbox baseline even blind. On this 4-class ESTA corpus the best figure is P3's
54/64 (Ch7); the *thesis* headline is the R48 lane (84.2 % balanced accuracy,
N = 1,100, Ch5 §5.6.9). See the banner above.** Section 4.4
presents per-class breakdown and
confusion analysis. Section 4.5 diagnoses the persistent error clips
and argues that the residual gap is attributable to missing
bounding-box annotations.

## 4.1 Evaluation Protocol

**Test subset.** The 4-class unsafe subset (64 held-out clips), matching
the confusion matrix reported in Table 6 of [17].

**Metrics.**
- **Top-1 accuracy**: argmax of the ensemble distribution vs ground-truth label.
- **Macro accuracy**: mean of per-class accuracies; reported because the
  class distribution is imbalanced (32 / 11 / 13 / 8).
- **Per-class accuracy**: `correct_c / total_c` for each class; useful for
  direct comparison with Table 7 of the paper.

**Inference configuration.** T = 32 uniformly sampled frames per clip,
224 × 224 resolution, ImageNet normalization. No test-time augmentation
(hflip was evaluated and rejected — see §4.2.3). Temperature
calibration uses a single scalar T* fit on a 12-clip held-out slice of
the training set.

**Hardware.** All experiments in this chapter run on an Apple MacBook Air
M3 (8-core CPU: 4 performance + 4 efficiency, 16 GB unified memory, macOS
14.6, MPS backend). The one exception is the YOLO detector retrain, which
runs on a Kaggle GPU runtime; its weights are downloaded and every
downstream experiment consumes them as a frozen checkpoint on the M3 host.
Training time per experiment is summarized in Table 4.1.

**Disclosure — re-use of the test set during head training.** The 64-clip
unsafe-test subset was used as a *validation set during head training*:
each head in `run_pairwise_contrastive.py` selects its best epoch by
test-set accuracy (early-stopping criterion at `train_classifier:255`).
This means the *individual heads* see test labels for model selection,
even though the test set is not used for gradient updates. Two
consequences are flagged honestly: (a) per-head accuracies in §4.7.1 and
§4.7.2 are an upper bound on what blind held-out evaluation would yield;
(b) the §4.7.7 audit of post-hoc bias calibration (the originally
reported 57/64) is a *separate* layer of test-set tuning on top of this
head-level model selection — and it is the post-hoc bias layer that has
been withdrawn (Table 4.13a). The best-case 56/64 ensemble number
inherits the head-level tuning but does not add a further calibration
step. **This leakage is not merely flagged — it is measured and removed
in §4.7.11**, where epoch, configuration, and seed are all selected on a
validation slice carved from the *training* set: the resulting honest
blind held-out accuracy is ≈ 52/64 (≈ 81 % top-1, macro ≈ 84 %), and that
blind number — not the best-case 56/64 — is this pipeline's defensible
accuracy claim (the best ESTA figure is P3's 54/64, Ch7; the thesis headline is
the R48 lane, Ch5 §5.6.9). The 56/64 checkpoint is retained only because it is the
P1 engine exported and timed in Chapter 5 §§5.1–5.5. It is **not** the engine on
the device in the live demo: that is the R48 B1 decision path, Ch5 §5.6.9.

## 4.2 Experiment Tracks

Two experiment tracks produced thesis-headline metrics and are reported
in detail in this section: **V9 — Pairwise contrastive ensemble** (best
top-1: 56/64 = 87.5 %) and **V7 — YOLO-World open-vocabulary features**
(best macro: 89.54 %, with 100 % on UnauthorInt). A third track, **V10
— Behaviour-class bbox finetune** (54/64), does *not* improve accuracy
but is retained as a negative result that pins down the cause of the
residual gap (§4.6). Six earlier tracks (V3, V5, V6-det, V6-ft,
VideoMAE, Final sweep, CrossEns) all converged on the 52–55/64 ceiling
and are preserved in `sweep_results/` and the project git history at
commit 854554a; their key takeaway — that no frozen-backbone variant
breaks through the 55-sample plateau — is summarised in §4.5
(data-bound evidence).

**Table 4.1 — Retained experiment tracks**

|  # | Track       | Method                                                          |   Best | Macro %  | Key Finding                                              |
| -: | ----------- | --------------------------------------------------------------- | -----: | -------: | -------------------------------------------------------- |
|  7 | YOLO-World  | YOLO-World-v2 open-vocabulary backbone + Transformer head       |  55/64 | **89.54** | **Best macro**; 99 % person det, UnauthorInt = 100 %     |
|  9 | Pairwise    | 8-class contrastive + direct4 + scenario-pair cascade           | **56/64** | 89.28 | **Best top-1**; only track to cross the 55-sample threshold |
| 10 | V10-bbox    | Behaviour-class bbox finetune (400 frames) + same head pipeline |  54/64 |   84.73  | Negative result: data-volume bound (§4.6)                 |

### 4.2.1 V7 — YOLO-World open-vocabulary features (best macro 89.54 %)

Hypothesis: YOLO-World's open-vocabulary prompting allows zero-shot
detection of categories ("worker", "forklift", "restricted zone") not
present in COCO. Replacing the YOLOv11s backbone with YOLO-World-v2
and running the same Transformer head pipeline produced the **highest
macro accuracy across all tracks (89.54 %)** and **100 % on the
UnauthorInt class** — the class most reliant on disambiguating
"who/what is in this zone". Top-1 stays at 55/64, indicating that
open-vocabulary detection helps macro-balance per-class performance
but does not add the missing zone-boundary information needed to
break the 55-sample top-1 plateau. The YOLO-World backbone is retained
in `sweep_results/yoloworld/` as a drop-in alternative input that the
ensemble in §4.2.2 can incorporate when macro-balanced metrics are
preferred over raw top-1.

### 4.2.2 V9 — Pairwise contrastive ensemble (best top-1 56/64, macro 89.28 %)

The final track combines three ideas:

1. **Pairwise contrastive training** — four binary heads specialized to
   each visually similar class pair (0-1, 0-2, 0-3, 2-3), trained with
   triplet margin loss on [CLS] embeddings.
2. **Scenario-pair cascade** — `p_cascade = p_scenario × p_pair`, as
   defined in §3.5.
3. **Direct4 + cascade averaging ensemble** — 0.5 / 0.5 fusion of the
   direct 4-class classifier and the cascade distribution.

Result: **56/64 top-1 (87.5 %), 89.28 % macro** — exactly one sample
better than the prior ~55/64 plateau and the **first time any
experiment in this project crossed the 55-sample threshold**. The
earlier code added a per-class bias search that yielded 57/64, but
§4.7.7 (Table 4.13a) shows that bias direction is specific to the 64
test labels — refitting the same search honestly on the training set
*reduces* test accuracy to 53.4 ± 1.2 / 64. The 57/64 figure is
withdrawn; the best-case (test-selected) top-1 is **56/64**, while the
honest blind held-out top-1 of this pipeline is **≈ 52/64** (§4.7.11).
Per-class breakdown is in §4.3.

### 4.2.3 Rejected: Test-Time Augmentation (hflip)

A final attempt to gain samples via TTA (`run_tta_track_a.py`, committed
as 97698d1) evaluated three variants: baseline, horizontal flip, and
temporal offset ±4 frames. The hflip variant **reduced** direct4 head
accuracy from 55 to 34 correct samples. Diagnosis: factory surveillance
scenes are not horizontally symmetric — the direction a worker walks
along a corridor is a discriminative cue for the WalkwayViol class —
so hflip corrupts rather than augments. TTA is therefore excluded from
the headline result.

## 4.3 Headline Result

> **Read this table as a best-case upper bound, not the headline accuracy.**
> The 56/64 below is the *test-selected* checkpoint (epoch, seed, and ensemble
> composition all chosen with reference to the 64-clip test set). P1's
> **honest, defensible best is 53/64 (82.8 % top-1, 85.4 % macro) for the
> best zero-bbox config (hybrid + frame-gate, §4.6.2)**; the *thesis* headline
> is P3's blind 54/64 (Ch7). The frozen single-head
> blind estimate is ≈ 52/64 (≈ 81 % top-1, macro ≈ 84 %), established in
> §4.7.11. The 56/64 figure is retained here because it is the checkpoint
> actually deployed in Chapter 5. The "+3.5 pp macro win" over the no-bbox
> baseline (Table 4.3) does **not** survive the blind protocol and is withdrawn
> (§4.7.11).

The best-case reported numbers (reproduced from `sweep_results/pairwise/
results.json`):

**Table 4.2 — Best-case result (test-selected ensemble: direct4_s314 + cascade_s7)**

| Metric                                | Value                          |
| ------------------------------------- | ------------------------------ |
| Best-case top-1 (ensemble, test-selected) | **56/64 = 87.5 %** (upper bound) |
| **Frozen single-head blind (original pipeline, §4.7.11)** | **≈ 52/64 ≈ 81 %** |
| **Hybrid+gate blind (best zero-bbox config, §4.6.2) — honest headline** | **53/64 = 82.8 % top-1 \| 85.4 % macro** |
| Top-1 with bias-search calibration *(withdrawn — test-set tuning, §4.7.7)* | 57/64 = 89.1 % |
| Macro accuracy (best-case ensemble)   | 89.28 % (blind ≈ 84 %)         |
| Ensemble members                      | direct4_s314, scenario_s7 × pair{0-3}_s7 (cascade) |

Comparison to the paper baseline:

**Table 4.3 — Comparison to Önal & Dandıl (2024) [17]**

| System                                                    | 4-class top-1                  | Macro %                                       |
| --------------------------------------------------------- | -----------------------------: | --------------------------------------------: |
| Paper: ConvLSTM only (Table 5 confusion of [17])          |  57/64 = **89.06**             | 85.78 (Table 7 — mean of per-class recalls)   |
| Paper: YOLO v4 + ConvLSTM (Table 6 confusion of [17])     |  62/64 = **96.88**             | **95.81** (Table 7 — mean of per-class recalls) |
| Ours: ensemble (direct4 + cascade) — **best-case (test-selected)** |  56/64 = 87.50             | 89.28                                     |
| Ours: frozen single-head blind (§4.7.11) |  **≈ 52/64 ≈ 81 %**           | **≈ 84 %**                                |
| Ours: hybrid+gate — **blind (honest, §4.6.2)** |  **53/64 = 82.8 %**  | **85.4 % (macro parity w/ no-bbox baseline)**  |
| Ours: ensemble + bias-search calibration *(withdrawn — §4.7.7)* | ~~57/64 = 89.06~~        | —                                             |

## 4.4 Per-Class Breakdown

**Table 4.4 — Per-class accuracy vs paper (Table 7 of [17])**

| Class ID | Name                 | n | Ours (best ens.) — per-class % | Paper ConvLSTM (Table 7) — per-class % | Paper YOLO+ConvLSTM (Table 6) — per-class % |
| -------: | -------------------- | -: | -----------------------: | -----------------------: | -----------------------: |
|        0 | WalkwayViol          | 32 |    27/32 = 84.38         |    31/32 = 96.88         |    **32/32 = 100.00**    |
|        1 | UnauthorInt          | 11 |     8/11 = 72.73         |    9/11  = 81.82         |       10/11 = 90.91      |
|        2 | **OpenedPanel**      | 13 | **13/13 = 100.00**       |   10/13  = 76.92         |       12/13 = 92.31      |
|        3 | **CarryOverload**    |  8 |  **8/8 = 100.00**        |    7/8   = 87.50         |    **8/8 = 100.00**      |
|          | **Top-1 (overall)**  | 64 |  **56/64 = 87.50 %**     |   57/64 = 89.06 %         |   **62/64 = 96.88 %**    |
|          | **Macro (mean of 4)** | — |  **89.28 %**             |   85.78 % (paper-reported, Table 7) |  **95.81 %** (paper-reported, Table 7) |

**Observations.**

- **OpenedPanel and CarryOverload are saturated.** The frozen-backbone
  pipeline produces perfect classification on these two classes, matching
  (CarryOverload) or exceeding (OpenedPanel +7.69 pp) the paper's
  YOLO+ConvLSTM system. These classes are dominated by distinctive
  scene-level cues (an open cabinet, an overloaded forklift) that are
  already salient in COCO-pretrained features.

- **WalkwayViol and UnauthorInt account for the entire gap.** The paper
  achieves 32/32 and 10/11 on these two classes; this thesis achieves
  27/32 and 8/11. All six missing samples come from these two classes.
  §4.5 argues these two classes cannot be reliably discriminated from
  frozen COCO features.

## 4.5 Persistent Error Analysis

**Table 4.5 — The 9 persistently misclassified clips (pairwise ensemble)**

| # | True Class     | Predicted   | Confidence | Diagnosis                                                                 |
| -: | ------------- | ----------- | ---------: | ------------------------------------------------------------------------- |
|  1 | WalkwayViol   | UnauthorInt |    80–90 % | Worker near equipment; unauthorized-access cue dominates                  |
|  2 | WalkwayViol   | UnauthorInt |    80–90 % | (same as #1)                                                              |
| 14 | WalkwayViol   | OpenedPanel |     56 %   | Low-confidence ambiguous scene                                            |
| 29 | WalkwayViol   | UnauthorInt |    80–90 % | (same as #1)                                                              |
| 30 | WalkwayViol   | UnauthorInt |    80–90 % | (same as #1)                                                              |
| 37 | UnauthorInt   | WalkwayViol |    94–96 % | Worker walking; walkway cue dominates                                     |
| 38 | UnauthorInt   | WalkwayViol |    94–96 % | (same as #37)                                                             |
| 44 | OpenedPanel   | WalkwayViol |     67 %   | Worker foregrounds scene; panel not visually salient                      |
| 51 | OpenedPanel   | UnauthorInt |     74 %   | Worker standing near panel; ambiguous intent                              |

**Pattern.** Six of nine errors (67 %) are on the
**WalkwayViol ↔ UnauthorInt** confusion pair — the two classes that are
visually identical at the token level (a worker is present in a
restricted area). The **discriminator is which restricted area**: the
walkway boundary vs a machine-access zone. This information is
geometric, not photometric.

**Evidence of data-bound ceiling.**

1. **All ten experiment tracks attempted (V3, V5, V6-det, V6-ft,
   VideoMAE, Final sweep, YOLO-World, CrossEns, Pairwise, V10-bbox)
   converge on ≤56/64.** Most produced 52–55 correct samples; only V9
   (Pairwise) broke through by exactly one sample, and V10 (bbox
   finetune, §4.6) *regressed* to 54/64 despite 8.6 hours of additional
   training. Detailed per-track results for the eight tracks not
   reported in §4.2 are preserved in `sweep_results/`.

2. **A 17× larger model reaches the same ceiling.** VideoMAE-base (86 M
   params, Kinetics-400 self-supervised pretraining) reaches 55/64 ≈
   YOLOv11s (5 M params) = 55/64. Model capacity is not the binding
   constraint.

3. **Open-vocabulary detection fixes only the macro metric.**
   YOLO-World (V7, §4.2.1) achieves 99 % person detection on Unsafe-Net
   and lifts macro accuracy to 89.54 % (UnauthorInt = 100 %), but
   top-1 stays at 55/64 because detecting "a person is here" does
   not answer "is this restricted zone A or B?"

4. **Feature-level similarity confirms it.** Pairs of
   WalkwayViol / UnauthorInt clips produce [CLS]-embedding cosine
   similarity in the 0.94–0.996 range — the two classes are
   approximately colinear in the frozen representation. §4.8.1
   reports the geometric details.

**Conclusion.** The paper's 95.81 % depends on a training signal that is
not available to this thesis: per-frame bounding-box labels that encode
behaviour class as a detector class. The paper's YOLO v4 detector is
behaviour-aware (it detects "WalkwayViolation" as an object class, not
"person"); our YOLOv11s is behaviour-agnostic (it detects "person").
§4.6 reports a direct experimental test of this argument.

## 4.6 V10 — Behaviour-aware bbox finetune (negative result)

To test the data-bound argument empirically rather than rely on
indirect evidence, ~50 frames per class (400 frames, 8 classes) were
manually annotated with behaviour-class bounding boxes (`annotation_guide/
annotations/`, YOLO format). YOLOv11s was then fine-tuned on these
behaviour-class detection labels for 100 epochs, and the same
direct4 / scenario / pair / cascade head pipeline of §3.3–§3.5 was
re-trained on features extracted from the fine-tuned backbone.

**Threat to validity — single-annotator design.** All 400 bbox frames
were annotated by a single person (the author) over a continuous
labelling session, without inter-rater agreement statistics or a
second-pass review. Two systematic biases follow from this and are
disclosed here so that the negative result is not over-interpreted:
(i) box-tightness conventions and the boundary between WalkwayViol
and UnauthorInt were resolved by the annotator's judgment alone, with
no Cohen's κ available; (ii) frame selection (~50/class) was
purposive — frames that were visually clearer were preferred — which
biases the training distribution towards the easier end of each class.
These design weaknesses *strengthen* the negative result: even a
favourable annotation distribution could not lift V10 above the
frozen-COCO ceiling at this volume, so a more rigorous (multi-rater,
randomly-sampled) protocol is unlikely to reverse the conclusion that
400 frames is insufficient. They do, however, mean V10 cannot be
treated as a clean test of the *minimum viable* annotation budget;
future work on paper-scale annotation (Ch6 §6.4 P1) requires at least
two independent annotators and reported κ.

**Update — Phase-2 corrected-mechanism investigation (post-chapter-freeze).**
Two refinements to the V10 effort were carried out after the initial
chapter draft and are reported here for completeness. **(a) Mechanism
correction.** A close re-reading of the source paper's Tables 5–7 revealed
that YOLO v4 in Unsafe-Net functions as a *frame filter* — it keeps only
frames where an unsafe-behaviour object is detected and discards the rest,
passing the reduced sequence to ConvLSTM — and that the paper's +10 pp gain
(85.78 % → 95.81 %) is attributable to this **filtering step**, not to
YOLO-derived spatial features. V10 implemented a *different* mechanism (the
fine-tuned YOLO as a backbone feature extractor whose features replace the
frozen-COCO ones). This architectural misalignment — independent of
annotation volume — is a second, plausible explanation for the V10
regression, and motivated a corrected *frame-gate* experiment (§6.4 P1).
**(b) Inter-annotator agreement.** The annotation effort was scaled to a
1 200-frame pilot (4 unsafe classes; safe frames as background) and a
second annotator independently re-labelled a 100-frame stratified subset,
yielding **Cohen's κ = 0.65** (substantial, Landis–Koch; n = 100, 5 labels
= 4 unsafe + background; raw agreement 72 %). Critically, the off-diagonal
disagreement concentrates on the **WalkwayViol ↔ UnauthorInt boundary**
(6/20 UnauthorInt frames called WalkwayViol by the second annotator; 7/20
background frames called UnauthorInt) — *the same pair the model fails on*.
This is convergent evidence for the data-bound diagnosis: the boundary is
semantically ambiguous at the single-frame level even for human experts. It
also implies the single-annotator 1 200-frame labels carry ~30 % ambiguity
on the hardest pair, so any frame-gate result on this pair should be read as
a lower bound. (κ artifact: `plans/260613-thesis-defense-revision/reports/kappa-result.json`.)

**Table 4.6 — Bbox-finetuned vs frozen-COCO branch (4-class unsafe)**

| Branch                                 | Best single | Cascade  | Ensemble | Macro %  |
| -------------------------------------- | ----------: | -------: | -------: | -------: |
| **Frozen COCO** (V9 headline)          |       55/64 |    51/64 | **56/64**|  **89.28** |
| **Bbox-finetuned** (V10)               |       54/64 |    52/64 |    54/64 |    84.73 |
| Δ (bbox − frozen)                      |          −1 |       +1 |       −2 |    −4.55 |

*Source: `sweep_results/finetune_bbox/results.json`, 8.6 h training.*

**Diagnosis.** With only 400 annotated frames against the YOLOv11s
backbone's 9.4 M parameters, the COCO inductive prior is washed out
faster than the 4-class behavioural signal can replace it. The pair
heads, which depend on fine-grained features the frozen YOLO already
encodes, suffer most — their individual accuracies collapse to 16–34 / 64
versus 13–25 / 64 from the frozen branch (and the cascade as a whole
drops from 51 to 52 / 64, but the ensemble loses 2 net samples).

### 4.6.1 Cross-ensemble of frozen + bbox branches

The bbox branch failure could in principle be salvaged by ensembling
across branches: if the two backbones make decorrelated errors, even a
weaker bbox branch could lift the overall accuracy. Both feature caches
(`sweep_results/pairwise/features/all8_test.pt` for frozen,
`sweep_results/finetune_bbox/features/bbox_test.pt` for bbox) and both
sets of head checkpoints were present when the blend was run, so the
cross-ensemble was evaluated without retraining (`run_cross_ensemble.py`).
The frozen cache is still in the tree; the bbox cache has since been
deleted and its directory is under `.gitignore`, so reproducing the blend
requires regenerating that cache first. The rows below are read from the
saved `cross_ensemble.json`, not from a live re-run.

**Table 4.6a — Cross-ensemble blends, sorted by top-1**

| Blend                                            | Correct | Macro % | Walkway | UnauthorInt | OpenedPanel | CarryOverload |
| ------------------------------------------------ | ------: | ------: | ------: | ----------: | ----------: | ------------: |
| Frozen ensemble alone (baseline)                 |   56/64 |   89.28 |  84.4 % |      72.7 % |     100.0 % |       100.0 % |
| 0.5·frozen_ens + 0.5·bbox_avg                    |   55/64 |   88.49 |  81.2 % |      72.7 % |     100.0 % |       100.0 % |
| 0.9·frozen_ens + 0.1·bbox_avg                    |   55/64 |   88.49 |  81.2 % |      72.7 % |     100.0 % |       100.0 % |
| avg(frozen_d4, frozen_cas, bbox_avg)             |   55/64 |   88.49 |  81.2 % |      72.7 % |     100.0 % |       100.0 % |
| 0.5·frozen_ens + 0.5·bbox_d4_s7                  |   55/64 |   88.49 |  81.2 % |      72.7 % |     100.0 % |       100.0 % |
| 0.5·frozen_d4 + 0.5·bbox_avg                     |   52/64 |   83.52 |  78.1 % |      63.6 % |      92.3 % |       100.0 % |

*Source: `sweep_results/ablations/cross_ensemble.json`. `bbox_avg` =
mean of bbox direct4 seeds {s42, s123, s314, s7}.*

**Result.** No blend exceeds the frozen-only baseline. Even at 90 %
frozen / 10 % bbox weight, the result holds at 55/64 — meaning the
bbox branch flips at least one frozen-correct WalkwayViol prediction
with high enough confidence to survive heavy down-weighting. The
errors of the two branches are *not* decorrelated — both fail on
WalkwayViol, with the bbox branch failing more.

**Falsifies a stronger hypothesis.** §4.5 argued the gap is data-bound;
this experiment further shows that for fixed annotation budget, more
behaviour-specific supervision *hurts* unless the supervision is dense
enough to replace the COCO prior. The paper's 95.81 % is therefore not
just data-bound but data-volume-bound: the annotation effort to match
their YOLO v4 detector is at least an order of magnitude larger than
this thesis's 400-frame pilot. The 56/64 frozen ensemble represents
the realistic accuracy ceiling within the available annotation budget.

### 4.6.2 Corrected-mechanism investigation at 1,200-frame scale (V11)

The §4.6 mechanism correction (YOLO is a *frame filter*, not a feature
extractor) and the scaled 1,200-frame annotation were taken to a full
experiment: the fine-tuned YOLOv11s detector (mAP@50 = 0.89 on the 1,200
bbox frames) was used both as a **frame-gate** (keep frames with a
confident unsafe detection, conf ≥ τ; τ selected on a train-carved
validation slice) and as a **feature extractor**, across frozen-COCO,
bbox-finetuned, and hybrid (concatenated) feature spaces. All numbers are
direct-4-head, blind held-out (val-selected) and best-case (test-selected),
on the 64-clip 4-class test set. Source:
`plans/260613-thesis-defense-revision/reports/feature-sweep-result.json`
(fixed gate τ = 0.15, every row) and
`plans/260613-thesis-defense-revision/reports/framegate-result.json`
(the τ sweep, τ ∈ {0.0, 0.15, 0.25, 0.35, 0.5}).

**Table 4.6b — Corrected-mechanism sweep (1,200-frame detector)**

| Configuration | Blind top-1 | Blind macro | Best-case top-1 | Best-case macro |
|---|---|---|---|---|
| Frozen-COCO (baseline) | 51/64 | 83.1 % | 53/64 | 85.4 % |
| Frozen + frame-gate | 52/64 | 83.2 % | 52/64 | 83.2 % |
| Bbox-finetuned features | 50/64 | 79.3 % | 50/64 | 80.0 % |
| Bbox + frame-gate (paper mechanism) | 51/64 | 80.1 % | 52/64 | 80.9 % |
| **Hybrid (frozen ⊕ bbox)** | 52/64 | 85.4 % | 53/64 | 85.4 % |
| **Hybrid + frame-gate** ⭐ | **53/64** | **85.4 %** | 53/64 | **86.9 %** |
| **Ensemble (frozen ⊕ hybrid)** | — | — | **54/64** | **87.7 %** |
| *Paper — no-bbox baseline* | *57/64* | *85.78 %* | *57/64* | *85.78 %* |
| *Paper — full-bbox* | *62/64* | *95.81 %* | *62/64* | *95.81 %* |

> **Protocol correction (2026-10-02).** An earlier draft gave the bbox + frame-gate
> row as 49/64 blind, 75.3 % macro. Those two values come from `framegate-result.json`,
> where τ *itself* is selected on validation (τ = 0.5 → 49.33/64, 75.32 %), whereas every
> other row in the table holds τ fixed at 0.15. Mixing the two protocols made the paper's
> mechanism look worse than a like-for-like comparison supports, so the row now carries the
> fixed-τ = 0.15 values from `feature-sweep-result.json` like its neighbours. The
> val-selected-τ variant is reported here rather than in the table. Neither choice changes
> the conclusion below: the gate moves at most one sample at fixed τ, and selecting τ costs
> samples.

Three results follow, all consistent with the data-bound thesis and
together its strongest confirmation.

**(i) Detection fine-tuning regresses; the frame-gate does not rescue it.**
Bbox-finetuned features drop macro to 79.3 % (vs frozen 83.1 %) because
fine-tuning the backbone for *detection* over-specialises layer-22 and
discards the general COCO representation the temporal head relies on. The
frame-gate — the paper's actual gain mechanism — moves at most one sample on
either feature space at the matched τ = 0.15 (+1 on frozen, +1 on bbox), i.e.
within seed noise; and when τ is itself selected on validation it *costs*
samples (49.3/64, 75.3 % macro). Replicated
faithfully at 1,200 frames, the paper's mechanism does **not** close the gap
at this annotation scale.

**(ii) Combining the two feature spaces recovers and slightly exceeds the
ceiling.** The hybrid (frozen ⊕ bbox-finetuned, concatenated) restores the
macro the bbox branch lost (85.4 %) and, with the frame-gate, gives the
best single configuration: **blind 53/64 (85.4 % macro)**, best-case 54/64
(86.9–87.7 % macro). The combination outperforms either branch alone — a
small but real finding: the detection-finetuned features carry
*complementary* behaviour signal that is only useful *alongside* the COCO
prior, not as a replacement.

**(iii) The honest comparison to the no-bbox baseline is now favourable on
macro, even blind.** At the matched zero-bbox annotation budget, the
hybrid + frame-gate pipeline reaches **macro parity-to-advantage** over the
published no-bbox baseline (blind 85.4 % ≈ 85.78 %; best-case 86.9–87.7 %
> 85.78 %), while remaining below it on top-1 (53–54/64 vs 57/64) and well
below the full-bbox 95.81 % cascade. The gap to the full-bbox result is
confirmed **annotation-volume-bound**: across 12+ tracks — including the
paper's exact frame-gate mechanism at 1,200 frames — the ceiling holds at
≈ 54/64 / 88 % macro. The hybrid + frame-gate is adopted as the thesis's
best zero-bbox configuration; the path to the full-bbox corner is
paper-scale (≥ 2,262-frame) bbox supervision, set out in §6.4.

## 4.7 Ablations

This section presents controlled studies that isolate the contribution of
each design choice. All ablations evaluate on the 4-class unsafe subset
(n = 64). Where the same checkpoint can be re-evaluated with a perturbed
input or fusion rule, the experiment is **inference-only**; where the
study requires retraining, this is noted explicitly. All raw numbers are
preserved under `sweep_results/ablations/` (`inference_ablation.json`,
`cascade_decomposition.json`, `decorrelation.json`, `calibration.json`,
`embedding_probe.json`).

### 4.7.1 Cascade decomposition (inference-only)

The headline ensemble averages the direct4 head with the scenario × pair
cascade. Table 4.7 isolates each component's individual contribution;
Table 4.8 fuses direct4 with each component in turn.

**Table 4.7 — Component-level accuracies**

| Component                        | Top-1 | Macro % | Walkway | UnauthorInt | OpenedPanel | CarryOverload |
| -------------------------------- | ----: | ------: | ------: | ----------: | ----------: | ------------: |
| direct4 only                     | 55/64 |   87.35 |  84.4 % |      72.7 % |      92.3 % |       100.0 % |
| scenario only (probabilistic 4-class) | 54/64 |   85.43 |  84.4 % |      72.7 % |      84.6 % |       100.0 % |
| pair-unsafe only (no scenario)   | 11/64 |   30.27 |   0.0 % |      18.2 % |      15.4 % |        87.5 % |
| cascade (no pair, scenario only) | 54/64 |   85.43 |  84.4 % |      72.7 % |      84.6 % |       100.0 % |
| cascade (full, sc × pair)        | 51/64 |   84.23 |  71.9 % |      72.7 % |      92.3 % |       100.0 % |
| **ensemble (0.5·direct4 + 0.5·cascade)** | **56/64** |   **89.28** |  **84.4 %** |      **72.7 %** |     **100.0 %** |       **100.0 %** |

**Two non-obvious results.** First, the cascade *alone* is worse than
direct4 alone (51 vs 55) — pair heads do not produce a competitive
classifier in isolation. Second, the pair heads' standalone projection
into 4-class space (treating each pair's "unsafe" probability as the
class score) achieves only 11/64 — barely above the 25 % chance level on
the largest class. Pair heads contribute *only* through the cascade
multiplication, not as classifiers in their own right.

**Table 4.8 — Ensemble alpha sweep: α·direct4 + (1−α)·cascade**

| α    | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | **0.5** | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
| ---: | --: | --: | --: | --: | --: | -------: | --: | --: | --: | --: | --: |
| Top-1 | 51 | 52 | 53 | 54 | 55 | **56** | **56** | **56** | 55 | 55 | 55 |

The 56/64 plateau spans α ∈ [0.5, 0.7], confirming the equal-weight blend
chosen in §3.5.2 is not a fragile tuning choice. The cascade and
direct4 produce different-but-overlapping correct sets; their average
recovers exactly one OpenedPanel sample that the cascade alone
misclassifies (12/13 in cascade → 13/13 in ensemble) by virtue of the
direct4 head's higher confidence on that clip.

### 4.7.2 Per-pair-head leave-one-out (inference-only)

To attribute the cascade's 56/64 contribution to specific pair heads,
each pair head is replaced with a uniform 1.0 in turn:

**Table 4.9 — Leave-one-pair-out cascade**

| Removed pair head        | Cascade alone | Ensemble (cas + d4) | Δ vs full ensemble |
| ------------------------ | ------------: | ------------------: | -----------------: |
| Walkway (pair0)          |         54/64 |               56/64 |       0 (no loss)  |
| **Intervention (pair1)** |     **49/64** |         **52/64**   |     **−4** **(critical)** |
| Panel (pair2)            |         52/64 |               56/64 |       0 (no loss)  |
| Forklift (pair3)         |         51/64 |               56/64 |       0 (no loss)  |

Only **pair1 (Intervention)** is critical to the headline result.
Removing pair1 drops the ensemble from 56 to 52 — the same UnauthorInt
samples that cause confusion in §4.5 are exactly the ones pair1's
binary supervision rescues. The other three pair heads can be removed
with no effect, indicating considerable redundancy in the cascade.

### 4.7.3 Test-time T robustness (inference-only)

Trained at T = 32, the model degrades gracefully when test-time T is
reduced via uniform sub-sampling.

**Table 4.10 — Subsampled test-time T**

|  T | direct4 | cascade | Ensemble | Macro % | Walkway | UnauthorInt | OpenedPanel | CarryOverload |
| -: | ------: | ------: | -------: | ------: | ------: | ----------: | ----------: | ------------: |
|  8 |   52/64 |   48/64 |   52/64 |   83.17 |  78.1 % |      54.5 % |     100.0 % |       100.0 % |
| 16 |   52/64 |   51/64 |   51/64 |   83.09 |  75.0 % |      72.7 % |      84.6 % |       100.0 % |
| 24 |   53/64 |   51/64 |   54/64 |   87.71 |  78.1 % |      72.7 % |     100.0 % |       100.0 % |
| **32** | **55/64** | **51/64** | **56/64** | **89.28** | **84.4 %** | **72.7 %** | **100.0 %** | **100.0 %** |

The 24-frame variant loses only 2 samples (54 vs 56), suggesting the
deployed system can trade T for 25 % latency reduction at
near-isoaccuracy.

### 4.7.4 Spatial pool grid (inference-only)

The 4 × 4 grid of 16 spatial tokens is compared against coarser pools
applied at test time only (the head was trained on 4 × 4):

**Table 4.11 — Spatial pool granularity**

| Pool grid | Tokens | direct4 correct | Δ vs 4×4 |
| --------: | -----: | --------------: | -------: |
|       1×1 |      1 |           32/64 |      −23 |
|       2×2 |      4 |           53/64 |       −2 |
|     **4×4** |  **16** |       **55/64** |       **0** |

Reducing to a single global token collapses accuracy to 50 % (chance
level on Walkway, the largest class). The 4 × 4 grid is the right
choice; 2 × 2 is a viable compression option for memory-limited edge.

### 4.7.5 Ensemble decorrelation analysis

Why does adding the bbox-finetuned branch *not* improve accuracy
(§4.6.1)? The Pearson correlation matrix over per-clip error vectors
quantifies how independent each member's mistakes are.

**Table 4.12 — Pairwise error-correlation matrix (ρ across 64 test clips)**

| Member          | frozen_d4 | frozen_cas | bbox_d4_s42 | bbox_d4_s123 | bbox_d4_s314 | bbox_d4_s7 |
| --------------- | --------: | ---------: | ----------: | -----------: | -----------: | ---------: |
| frozen_d4       |      1.00 |       0.69 |        0.27 |         0.27 |         0.44 |       0.03 |
| frozen_cas      |      0.69 |       1.00 |        0.25 |         0.25 |         0.39 |       0.01 |
| bbox_d4_s42     |      0.27 |       0.25 |        1.00 |         0.79 |         0.13 |       0.48 |
| bbox_d4_s123    |      0.27 |       0.25 |        0.79 |         1.00 |         0.13 |       0.48 |
| bbox_d4_s314    |      0.44 |       0.39 |        0.13 |         0.13 |         1.00 |       0.24 |
| bbox_d4_s7      |      0.03 |       0.01 |        0.48 |         0.48 |         0.24 |       1.00 |

**Reading.** bbox_d4_s7 is essentially uncorrelated with the frozen
branch (ρ = 0.03 with frozen_d4, 0.01 with frozen_cas). Yet bbox_d4_s7
in isolation reaches only 45/64 (macro 74.7 %), and the cross-ensemble
fails (§4.6.1). This is the textbook ensemble-failure case: decorrelation
is necessary but not sufficient when individual member accuracy is
substantially below the strongest member. Within-branch correlations
(0.69 for frozen, 0.79 for bbox seeds 42/123) confirm the decorrelation
gain comes from the *across-branch* axis, but the gain cannot overcome
the bbox branch's accuracy deficit.

### 4.7.6 Temperature calibration (inference-only)

The headline 57/64 reported in earlier drafts was produced by random
search over per-class additive bias on test-set probabilities — a
test-set-tuning protocol that is not a defensible calibration. We
re-evaluate using **scalar temperature scaling** applied to each
component's logits before cascade fusion, sweeping T ∈ [0.5, 2.0]:

**Table 4.13 — Temperature sweep on the (direct4 + cascade) ensemble**

|   T | Top-1 | Macro % |    NLL |   ECE | Mean confidence |
| --: | ----: | ------: | -----: | ----: | --------------: |
| 0.5 | 55/64 |   88.49 | 0.4452 | 0.146 |           0.892 |
| 0.8 | 55/64 |   88.49 | 0.4066 | 0.110 |           0.827 |
| **1.0** | **56/64** | **89.28** | **0.4200** | **0.114** |     **0.786** |
| 1.5 | 56/64 |   89.28 | 0.4956 | 0.182 |           0.693 |
| 2.0 | 56/64 |   89.28 | 0.5845 | 0.257 |           0.618 |

Top-1 saturates at 56/64 for T ≥ 1.0; NLL is minimized at T ≈ 0.8 (where
top-1 is 55/64 — the trade-off between sharpness and accuracy). The
**defensible test-selected top-1 is 56/64**, not 57/64 (the honest blind
figure is ≈ 52/64, §4.7.11). The bias-search calibration
of the original code is now reported as a methodological caveat in §4.7.7.

### 4.7.7 Methodology note on the previously reported 57/64

The earlier "57/64 = 89.1 % calibrated" figure originates from a 20 000-step
random search over per-class additive biases (`run_pairwise_contrastive.py`
line 519) that was performed on the test set itself. The search range is
narrow ([−0.5, +0.5]) and the lift is only one sample, but the protocol
constitutes test-set tuning. To verify whether the +1 sample reflects a
real, generalizable calibration signal or pure memorization of the 64
test labels, the same bias search was re-run honestly on the training
set instead (`run_ablation_calibration_holdout.py`). The result is
decisive (Table 4.13a): the bias direction that improves training
accuracy *hurts* test accuracy by 2–3 samples.

**Table 4.13a — Honest 57/64 retry: bias search on train holdout (sweep_results/ablations/calibration_holdout.json)**

| Calibration regime                                        | Fit on        | Bias direction         |   Test top-1 | Test macro |
| --------------------------------------------------------- | ------------- | ---------------------- | -----------: | ---------: |
| Baseline — no calibration (headline)                      | n/a           | b = 0                  | **56/64**    | **89.28**  |
| **Test-set tuning (audit target)**                        | test (N = 64) | [+0.47, +0.15, −0.03, −0.34] | 57/64  | 90.06    |
| Honest fit on full 4-class train (N = 452)                | train         | [+0.36, +0.20, −0.21, +0.27] |  53/64 | 82.36    |
| Honest fit, 5 random-search seeds, full train             | train         | mean over 5 seeds            | 53.6 ± 0.5 | —      |
| Honest fit, 5-fold stratified holdout (frac = 0.2/fold)   | train holdout | mean over 5 folds            | 53.4 ± 1.2 (range 51–54) | — |

Three reads of the table: (i) the test-set bias direction differs in
sign and magnitude from any train-fit bias — the +1 sample on test is
specific to those 64 clips, not a generalizable correction; (ii) every
honest variant (full train, multi-seed, k-fold holdout) returns a worse
test result than the uncalibrated 56/64 — the calibration cannot be
defended; (iii) the std of 0.5–1.2 across folds and seeds is small
enough that this is not noise. The previously reported 57/64 is
therefore *withdrawn*. Within this calibration audit the defensible
figure over the calibrated 57/64 is the uncalibrated **56/64 = 87.5 %,
macro 89.28 %** — but note this is still a *test-selected* number;
§4.7.11 takes the remaining model-selection leakage (epoch/config/seed)
off the test set; the honest blind number for the frozen pipeline is
established there (≈ 52/64), while the best zero-bbox config (hybrid+gate)
is in §4.6.2 (53/64, 85.4 % macro). This subsection replaces the §3.5.3 calibration claim that
the calibration "shifts exactly one borderline sample from the correct
majority class" — that statement was true on the test-set fit but does
not generalize.

### 4.7.8 Architecture and training-stage ablations (retraining)

Three retraining studies were run on the frozen-COCO train feature cache
(566 train clips, all 8 classes, 593 MB) using `run_ablation_retrain.py`.
Each variant trains three seeds (42, 123, 7) on the 4-class subset for up
to 200 epochs (early stop after 40 stale epochs); the same focal-γ=2,
label-smoothing 0.05, EMA 0.998, cosine schedule, and weighted-random
sampler from §3.4 are applied. Mean ± std top-1 is reported. All
results below are from re-trained heads with these standard seeds —
*not* from the lucky-seed checkpoints used in the headline ensemble.
The headline number (56/64 = direct4_s314 + cascade) reflects a single
high-performing seed selected from the original sweep; the retraining
seed pool here probes the *typical* behaviour of each architecture.

**Table 4.16 — Stream contribution (4-class direct head, dim=384, L=6, H=8)**

| Streams used                                            | Top-1 (mean ± std) | Macro-F1 (mean ± std) | Per-seed accs |
| ------------------------------------------------------- | ------------------ | --------------------- | ------------- |
| Appearance only (motion zeroed)                         | 53.67 ± 0.47       | 0.854 ± 0.005         | 54, 54, 53    |
| Motion only (appearance zeroed)                         | 53.00 ± 0.82       | 0.832 ± 0.017         | 52, 54, 53    |
| Both (baseline)                                         | 53.33 ± 0.47       | 0.848 ± 0.007         | 54, 53, 53    |

The three configurations are **statistically indistinguishable** within
the seed-to-seed noise. Removing the motion stream costs nothing on
average; removing the appearance stream costs ~0.7 samples. The
multi-stream design — a central architectural feature of the head —
**does not produce a measurable accuracy gain** at this dataset scale.
This is consistent with the §4.5 data-bound argument: the bottleneck is
the cosine collapse of the raw features (Table 4.19), which is upstream
of any temporal-stream design choice.

This result has a direct naming consequence. On Unsafe-Net the pipeline
operates, in effect, as a **frame-level pooled classifier with an
unrewarded temporal path** — the motion-derivative streams the proposal
framed as "temporal reasoning" carry no measurable discriminative signal
here. The honest reading is that Unsafe-Net's class signal is largely
per-frame appearance, not motion dynamics; this is itself a finding about
the dataset's temporal information content. Accordingly this thesis
describes the component as a *multi-stream aggregation head* rather than a
*temporal-reasoning* module, and does not claim temporal modelling as a
demonstrated contribution (see §1.3.1).

**Table 4.17 — Depth × heads sweep (4-class direct head, dim=384, both streams)**

| (Layers L, Heads H) | Top-1 (mean ± std) | Macro-F1 (mean) | Per-seed accs |
| ------------------- | ------------------ | --------------- | ------------- |
| L = 2, H = 4        | 53.67 ± 0.47       | 0.854           | 54, 54, 53    |
| L = 4, H = 4        | 53.33 ± 0.47       | 0.851           | 54, 53, 53    |
| L = 4, H = 8        | 52.33 ± 1.25       | 0.796           | 52, 51, 54    |
| **L = 6, H = 8** (baseline) | **53.67 ± 0.47** | 0.837   | 54, 53, 54    |
| L = 8, H = 8        | 51.33 ± 1.25       | 0.803           | 53, 50, 51    |

The smallest configuration (L = 2, H = 4) ties the deepest configuration
that improves performance (L = 6, H = 8). Going beyond L = 6 *degrades*
accuracy. The Transformer head is **over-parameterised** for the
available 452 training clips: variance explodes (1.25 std) at the
two largest configurations, and the baseline L = 6 was retained only
because of the original sweep's slightly better calibrated F1 on the
full ensemble — not because deeper is meaningfully better.

**Table 4.18 — Gated vs. unweighted ensemble fusion**

| Fusion rule                                          | Top-1 | Per-seed |
| ---------------------------------------------------- | ----- | -------- |
| Unweighted: (direct4 + cascade) / 2 (baseline)       | 56    | —        |
| Learned per-clip gate (2-D softmax MLP on stats)     | 55.0 ± 0.0 | 55, 55, 55 |

The learned gate **underperforms the unweighted average** by exactly
one sample, deterministically across three seeds. This is a
counter-intuitive but interpretable result: with N_train = 452 clips
and only N = 64 test clips, the gate's two extra trainable parameters
per clip introduce more variance on the held-out set than they
remove on the train set. The unweighted average is, in this regime,
the maximum-margin-on-test fusion rule. The decorrelation result of
§4.7.5 (ρ = 0.21) already places (direct4 + cascade) / 2 close to
the optimal linear blend; learned gating cannot improve on it
without overfitting.

**Reading.** Three independent retraining ablations all return the same
verdict: **at this annotation scale, architecture-level choices —
streams, depth, fusion — are within seed-to-seed noise of each other**.
This complements the cross-component analyses in §4.7.1–§4.7.7 and the
geometric finding in §4.8: the residual top-1 gap is dominated by the
input-feature cosine collapse, and no head-side knob in the explored
space changes that fact.

### 4.7.9 Bootstrap confidence intervals on the test set (N = 64)

A test set of 64 clips is small enough that any single-number accuracy
must be reported with an interval. The 95 % bootstrap CI (5 000 resamples
with replacement of the 64-clip test set) is given below:

**Table 4.14 — Bootstrap 95 % CIs on top-1 and macro-F1**

| Component | Top-1 acc | 95 % CI | Macro-F1 | 95 % CI |
| --- | --- | --- | --- | --- |
| direct4 (s314) | 0.860 | [0.766, 0.938] | 0.861 | [0.771, 0.938] |
| scenario | 0.844 | [0.750, 0.922] | 0.847 | [0.761, 0.926] |
| cascade (sc × pair) | 0.798 | [0.688, 0.891] | 0.819 | [0.733, 0.899] |
| **ensemble (d4 + cas)/2** | **0.875** | **[0.797, 0.953]** | **0.879** | **[0.797, 0.950]** |

Paired bootstrap deltas (matched test set) clarify which component
contributes statistically:

**Table 4.15 — Paired bootstrap deltas (95 % CI)**

| Comparison | Δ acc | 95 % CI | P(A > B) |
| --- | --- | --- | --- |
| ensemble − direct4 | +0.015 | [+0.000, +0.047] | 0.628 |
| ensemble − cascade | +0.078 | [+0.016, +0.141] | 0.996 |
| cascade − direct4 | −0.062 | [−0.141, +0.000] | 0.021 |

**Honest reading.** The +1 sample lift of the full ensemble over direct4
alone (56/64 vs. 55/64) is *not statistically significant* on N = 64
(P = 0.63, CI lower bound touches zero). The cascade alone is *worse*
than direct4 with high confidence (P = 0.98). The ensemble is unambiguously
better than the cascade alone (P > 0.99) — i.e. the *direct4 contribution
inside the ensemble* is doing the statistically detectable work. Only a
larger held-out set would resolve whether the ensemble's small lift is
real or a finite-sample artefact.

This finding is consistent with the §4.7.5 decorrelation result: although
direct4 and the cascade make uncorrelated errors (ρ = 0.21), the cascade's
absolute accuracy is too low to consistently rescue direct4's mistakes
within the test sample.

#### 4.7.9.1 Bootstrap CIs on macro accuracy and per-class accuracy

The §4.7.9 top-1 CI is complemented by macro-accuracy and per-class CIs.
The 5 000-sample bootstrap is stratified within each class for the
per-class rows so that the resampled set preserves class size.
(`run_ablation_significance.py` → `sweep_results/ablations/significance.json`.)

**Table 4.16a — Bootstrap CIs (5 000 resamples) on the headline ensemble**

| Metric                  | n  | Point estimate | Bootstrap mean | 95 % CI            | Std    |
| ----------------------- | -: | -------------: | -------------: | ------------------ | -----: |
| Top-1 (overall)         | 64 |     87.50 %    | 87.46 %        | [79.69, 95.31]     | 4.04   |
| **Macro (mean of 4)**   |  — |   **89.28 %**  | **89.34 %**    | **[81.32, 96.32]** | 3.79   |
| Per-class WalkwayViol   | 32 |     84.38 %    | 84.43 %        | [71.88, 96.88]     | 6.36   |
| Per-class UnauthorInt   | 11 |     72.73 %    | 72.85 %        | [45.45, 100.00]    | 13.53  |
| Per-class OpenedPanel   | 13 |    100.00 %    | 100.00 %       | [100.00, 100.00]   | 0.00   |
| Per-class CarryOverload |  8 |    100.00 %    | 100.00 %       | [100.00, 100.00]   | 0.00   |

The two saturated classes have zero-width CIs (every bootstrap resample
hits 100 %). UnauthorInt's CI spans [45, 100] — the +Δ vs paper for
this class is dominated by sample-size noise at n = 11.

#### 4.7.9.2 Two-proportion z-tests vs the paper

Unpaired two-sample z-tests against the published Önal & Dandıl (2024) [17]
numbers (the paper does not release per-clip predictions, so an
unpaired test is the conservative baseline; paired McNemar bounds
follow in §4.7.9.3). *These tests use the best-case (test-selected)
56/64 figure; even so the macro lift is already non-significant, and
under the blind protocol (§4.7.11) it disappears entirely — the row
below is retained to document the best-case comparison, not to claim a
macro advantage.*

**Table 4.16b — Two-proportion z-tests (unpaired, best-case 56/64)**

| Comparison                                 | Δ        |     z   |    p   | Interpretation                  |
| ------------------------------------------ | -------- | ------: | -----: | ------------------------------- |
| top-1: ours 56/64 vs paper 57/64 (no-bbox, Table 5)  | −1.56 pp | −0.270  | 0.7872 | not significant on N = 64 — parity |
| top-1: ours 56/64 vs paper 62/64 (full, Table 6)     | −9.38 pp | −1.976  | 0.0481 | borderline significant deficit  |
| macro: ours 89.28 vs paper 85.78 (no-bbox, Table 7)  | +3.50 pp | +0.599  | 0.5490 | not significant — directional lift |
| macro: ours 89.28 vs paper 95.81 (full, Table 7)     | −6.53 pp | −1.406  | 0.1596 | not significant                 |

Both directions of the comparison fail at the 1 % level on N = 64. The
−1.56 pp top-1 deficit vs the no-bbox ConvLSTM baseline (one sample on
N = 64) cannot be defended as a *significant* deficit, and the +3.50 pp
macro lift over the same baseline cannot be defended as a *significant*
improvement.

> **Superseded in part, 2026-09-29.** The z-test is the wrong instrument here (it
> assumes two independent samples; both classifiers are scored on the *same* 64 clips),
> and §4.7.9.3's paired bounds replace it. Two consequences. First, the row reading
> "top-1 56/64 vs 62/64, p = 0.048, borderline significant deficit" is not the paired
> answer: the exact paired bound is [0.031, 0.109], which brackets 0.05 and decides
> nothing. Second — and this cuts against the thesis — for the figures actually
> defended (54/64, 51/64) the paired deficit against the full-bbox pipeline **is**
> significant under every feasible split. The sentence this paragraph used to end with,
> that the deficit vs the paper's full pipeline "cannot be defended as a significant
> deficit either", is **withdrawn**. The macro rows are untestable by McNemar at all
> (macro accuracy is not a per-clip binary outcome), so they remain z-tests and remain
> non-significant.

The structural argument (V10 negative result, §4.6.2) remains the more
reliable evidence for the data-bound thesis — and it now agrees with,
rather than merely substitutes for, the significance test.

#### 4.7.9.3 McNemar marginal bounds

The paper's per-clip predictions are not published, so a paired
McNemar test cannot be computed exactly. The bound below uses the
known marginals only: with both classifiers evaluated on the same
N = 64, the discordant cell sum D = b₀₁ + b₁₀ satisfies
|b₀₁ − b₁₀| = |k_a − k_b| and is bounded by both classifiers'
individual accuracy levels.

**Table 4.16c — McNemar marginal bounds (paired N = 64, exact binomial)**

*Revised 2026-09-29: these are now **exact** two-sided McNemar p-values, replacing the
asymptotic χ² approximation the table previously carried, and the enumeration now
respects each side's own wrong-count (the earlier enumerator constrained only the
difference, admitting infeasible 2×2 tables — see
`plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-09-29-mcnemar-bound-marginal-constraint.md`).*

| Comparison                                   | D range   | exact p range        | Reading                             |
| -------------------------------------------- | --------- | -------------------- | ----------------------------------- |
| best-case 56/64 vs paper full 62/64          | [6, 10]   | **[0.031, 0.109]**   | brackets 0.05 — undetermined        |
| best-case 56/64 vs paper baseline 57/64      | [1, 15]   | **[1.000, 1.000]**   | p = 1 at *every* split              |
| blind-ladder 54/64 vs paper full 62/64       | [8, 12]   | **[0.008, 0.039]**   | **significant at every split**      |
| blind-ladder 54/64 vs paper baseline 57/64   | [3, 17]   | **[0.250, 0.629]**   | not significant at any split        |
| hybrid 51/64 vs paper full 62/64             | [11, 15]  | **[0.001, 0.007]**   | **significant at every split**      |
| hybrid 51/64 vs paper baseline 57/64         | [6, 20]   | **[0.031, 0.263]**   | brackets 0.05 — undetermined        |

Three readings, and one of them is uncomfortable for this thesis and is stated
plainly. **(i) Against the paper's full-bbox pipeline the deficit is statistically
significant** for the figures the thesis actually defends (54/64 and 51/64): every
feasible per-clip overlap yields p < 0.05. An earlier version of this table, computed
with an unconstrained enumerator, reported [0.008, 0.382] and [0.001, 0.207] and
concluded the deficit was not robustly significant; that conclusion is **withdrawn**.
The deficit is real, and it is precisely what the data-volume-bound diagnosis (§4.6.2,
V10 negative result) predicts — 400–1,200 bbox frames against the paper's 2,262.
**(ii) Against the no-bbox ConvLSTM baseline nothing is significant** at any bound, so
the macro-parity claim stands unchanged. **(iii) The one-clip gap (56/64 vs 57/64)
yields p = 1.000 at every split** — not merely non-significant but entirely
uninformative, which is the honest way to describe a 1-sample swing on N = 64 and is
weaker than the z-test's p = 0.787 implied.

### 4.7.10 Input resolution ablation (224² vs 384², direct4 single-head)

The full pipeline ingests frames at 224 × 224 — a YOLOv11s-friendly
resolution that keeps the dominant cost (per-frame backbone forward)
roughly 3× cheaper than 384 × 384 and matches the operating point
measured for edge deployment (Ch 5 §5.4). The choice is not free:
higher input resolution preserves more spatial detail at layer 22
before the 4 × 4 pool collapses each frame to 16 tokens. To audit the
trade-off, the YOLOv11s + direct4 head was re-extracted and retrained
end-to-end at 384 × 384 (`run_resolution_ablation.py`,
`sweep_results/ablations/resolution_ablation.json`). The cascade and
ensemble are *not* re-run — direct4 covers 55 of the 56 ensemble
samples, so it is the most-leveraged measurement of the resolution
effect; full-ensemble retraining at 384² would consume ~12 hr of MPS
time without changing the conclusion.

**Table 4.22 — Input resolution ablation, direct4 single-head, seed = 314**

| Resolution     | Top-1 | Top-1 % | Macro  | Walk  | Unauth | Panel | Forklift |
| -------------- | ----: | ------: | -----: | ----: | -----: | ----: | -------: |
| **224² (operating)** | **55/64** | **85.94** | **0.874** | 0.844 | 0.727 | 0.923 | 1.000 |
| 384² (ablation)      |    52/64  |    81.25  |    0.772  | 0.906 | **0.182** | 1.000 | 1.000 |
| Δ (384 − 224)        |     −3    |    −4.69 pp |  −10.16 pp | +0.062 | **−0.545** | +0.077 | 0.000 |

**Honest reading.** Higher resolution is *not* a free win. 384²
*improves* WalkwayViol (+6.2 pp), OpenedPanel (+7.7 pp), and saturates
CarryOverload — three of four classes — but **collapses UnauthorInt
from 8/11 to 2/11 (−54.5 pp)**, dragging macro down by 10.16 pp. The
collapse is consistent with the class-imbalance behaviour described
in §4.4 and §4.7.9.1: UnauthorInt has the smallest support (n = 11,
~17 % of the test set) and depends on disambiguating *who is in the
zone* rather than appearance — the head is more sensitive to
representation drift at the minority class. Higher input resolution
shifts the spatial-token statistics enough that the seed-fixed
classifier, retrained from scratch with patience-based early stopping,
no longer recovers the UnauthorInt manifold at 384² (early-stopped at
epoch 80, best 52/64). Compute is not the only axis on which 224² is
the right choice — it is also the better single-axis accuracy
operating point.

This finding is consistent in direction with the §4.7.4 spatial-pool
ablation: once the 4 × 4 pool grid is fixed, *adding* spatial
resolution upstream does not propagate through to the classifier at
this dataset scale (566 train clips, 8 classes). The 224² choice
defended in §3.2 is therefore retained on accuracy grounds in
addition to compute grounds.

### 4.7.11 Blind held-out evaluation (removing model-selection leakage)

§4.1 disclosed that every head selects its best epoch by accuracy on the
64-clip test set, and the headline ensemble members were greedy-selected
on the same set. The 56/64 figure therefore inherits model-selection
leakage. This subsection measures what the pipeline yields once that
leakage is removed end-to-end.

**Protocol.** A stratified validation slice is carved from the *training*
set (≈ 20 %, 91 of the 452 four-class train clips; 8-class-stratified,
fixed split seed). Every head is retrained from scratch and its best epoch
selected on that validation slice; the ensemble configuration (direct-4
vs cascade vs the direct-4 + cascade ensemble) and the seed are *also*
selected on validation. The 64-clip test set is touched exactly once, for
the final number. The procedure is in `run_blind_eval.py`; two protocols
were run (val_frac 0.20 with the fixed headline ensemble, and val_frac
0.15 with validation-based config + seed selection) over seeds
{314, 42, 123, 7}.

**Table 4.24 — Blind held-out vs the test-selected headline**

| Selection regime | top-1 | macro | Notes |
|---|---|---|---|
| Best-case (test-selected epoch, lucky seed 314, greedy ensemble) | 56/64 = 87.5 % | 89.28 % | the deployed checkpoint; optimistic upper bound |
| Typical (test-selected, standard seeds, single direct-4) | 53.3/64 | ~84.8 % | seed-honest, still selects on test |
| **Blind — single direct-4, val-selected epoch** | **51.7 ± 2.0 / 64 ≈ 80.8 %** | **~84 %** | honest generalization estimate |
| Blind — direct-4 + cascade ensemble, val-selected | 50.7 / 64 | 83.9 % | ensemble *underperforms* the single head |
| Paper ConvLSTM-only (no bbox) baseline | 57/64 = 89.06 % | 85.78 % | protocol not reported |

*Source:* `plans/260613-thesis-defense-revision/reports/blind-eval-result.json`
(val_frac 0.20, seeds 314/42/123: direct-4 51.7 ± 1.9, macro 85.0 %) and
`plans/260613-thesis-defense-revision/reports/blind-eval-v2-vf0.15.json`
(val_frac 0.15, seeds 314/42/123/7: direct-4 51.8 ± 2.3, macro 83.1 %).
The row quotes the two runs rounded together, hence the ± 2.0 and the
hedged macro.

Three conclusions. **(i)** The honest, leakage-free top-1 is **≈ 52/64
(≈ 81 %)** — roughly 4–5 samples of the best-case 56/64 were
model-selection optimism (≈ 2 from epoch selection on the single head,
the remainder from the lucky seed and the greedy-on-test ensemble).
**(ii)** The macro advantage over the baseline does **not** survive: the
best blind macro (single direct-4, ~85 %) is at parity-minus-one with the
baseline's 85.78 %, and the ensemble macro (83.9 %) is below it. The
"+3.5 pp macro win" reported in earlier drafts was a test-selection
artifact and is withdrawn. **(iii)** When the configuration is chosen on
validation rather than on the test set, the **single direct-4 head is
selected on every seed** — the pairwise-contrastive cascade and the
ensemble are honestly rejected, confirming §4.7.9 (P = 0.63, CI touches
zero) in the strongest form. Running multiple honest selection protocols
(two val fractions, val-based config and seed selection) did not lift the
estimate above ~52/64, which indicates the number is a stable property of
the frozen-feature representation, not an artifact of one unlucky split.

This blind ≈ 52/64 is the frozen single-head pipeline's defensible number.
P1's best — the best zero-bbox configuration (hybrid + frame-gate,
§4.6.2) — lifts this to **53/64 (82.8 % top-1, 85.4 % macro)**, at macro
parity with the no-bbox baseline even under the blind protocol. The deployed
artifact (Chapter 5) still uses the best-case 56/64 checkpoint — the model
one would actually ship — but the *scientific* statement about generalization
credits the hybrid+gate blind result, and all three numbers are kept distinct
throughout.

## 4.8 Interpretability — what does the head learn?

The accuracy numbers in §4.3–§4.7 establish *that* the multi-stream
gated head works; this section asks *what it learns*. The probe targets
the [CLS] embedding produced by the encoder (§3.3.3) — the 384-D vector
fed to the classifier MLP — and measures how well each model separates
the four behaviour classes in cosine geometry. Raw frozen-YOLO features
serve as the baseline.

### 4.8.1 The raw-feature collapse

Mean-pooled YOLOv11s layer-22 features (no head) place all four classes
at near-identical cosine similarity, confirming the §4.5 claim quantitatively:

**Table 4.19 — Raw frozen-YOLO cosine similarity between class pairs**

| Class pair                          | Between-class cos | Max-pair cos |
| ----------------------------------- | ----------------: | -----------: |
| **WalkwayViol  vs  UnauthorInt**    |         **0.934** |    **1.000** |
| UnauthorInt  vs  OpenedPanel        |             0.888 |        0.995 |
| WalkwayViol  vs  OpenedPanel        |             0.873 |        0.996 |
| UnauthorInt  vs  CarryOverload      |             0.834 |        0.918 |
| WalkwayViol  vs  CarryOverload      |             0.831 |        0.921 |
| OpenedPanel  vs  CarryOverload      |             0.823 |        0.924 |

All six pairs cluster in the 0.823–0.934 range, and the hard
WalkwayViol ↔ UnauthorInt pair is the *most* collapsed of the six, with
a max-pair cosine of 1.000 — some clip pairs across the two classes are
indistinguishable in raw feature space. The frozen backbone, by itself,
places the four behaviour classes essentially co-linear in embedding
space. This is the geometric statement of the data-bound argument in
§4.5. Source: `sweep_results/ablations/embedding_probe.json`
(`raw_yolo_baseline`), regenerable with `run_probe_tsne.py`.

> **Correction (2026-10-02).** An earlier draft of this table listed four
> rows with 0.834 / 0.821 / 0.819 / 0.823 and attributed 0.834 to the
> WalkwayViol ↔ UnauthorInt pair. Re-read against the artifact: 0.834 is
> the **UnauthorInt ↔ CarryOverload** pair, 0.821 and 0.819 appear nowhere
> in the probe output, and the hard pair's true value is **0.934**. Only the
> OpenedPanel ↔ CarryOverload row (0.823 / 0.924) was correct. The table now
> carries all six pairs as computed. The correction strengthens §4.8.1's
> claim — the raw collapse is more severe than reported — and enlarges every
> Δ in Table 4.20; it does not affect any accuracy number, because the
> cosine probe feeds no decision.

### 4.8.2 What the direct4 head learns

After end-to-end training on the 4-class label, the [CLS] embedding
geometry changes substantially:

**Table 4.20 — direct4_s314 [CLS]-embedding cosines**

| Class pair                          | Between-class cos | Δ vs raw |
| ----------------------------------- | ----------------: | -------: |
| WalkwayViol  vs  UnauthorInt        |        **+0.375** |   −0.559 |
| WalkwayViol  vs  OpenedPanel        |            −0.353 |   −1.226 |
| WalkwayViol  vs  CarryOverload      |            −0.023 |   −0.854 |
| UnauthorInt  vs  OpenedPanel        |            −0.348 |   −1.237 |
| UnauthorInt  vs  CarryOverload      |            −0.233 |   −1.067 |
| OpenedPanel  vs  CarryOverload      |            −0.061 |   −0.885 |

Δ is computed against the corrected Table 4.19 raw values. Source:
`sweep_results/ablations/embedding_probe.json` (`models.direct4_s314`).

Three of the four "easy" pairs are pushed to *negative* cosine, i.e.
the head actively separates them. The hard WalkwayViol ↔ UnauthorInt
pair is reduced from 0.934 to 0.375 — a substantial separation, but the
two clusters remain in the same hemisphere. This is the geometric
fingerprint of the residual error pattern: the head learns a partial
separator on the confusion pair, exactly enough to score 8/11
UnauthorInt and 27/32 WalkwayViol but no more.

### 4.8.3 Pair heads do not separate; they classify

If the pair heads worked by clustering, the within-pair classes would
be pushed apart in [CLS] space. They are not:

**Table 4.21 — Binary pair-head [CLS]-embedding cosines on their target pair**

| Pair head             | Target pair cosine | Within-class cosines (c1 / c2) |
| --------------------- | -----------------: | ----------------------------: |
| pair0 Walkway-vs-?    |              0.676 |                  0.660 / 0.654 |
| pair1 Intervention    |              0.788 |                  0.835 / 0.748 |
| pair2 Panel           |              0.843 |                  0.880 / 0.794 |
| **pair3 Forklift**    |          **0.984** |              **0.986 / 0.981** |

The pair heads do *not* push their target classes apart in cosine space
— between-pair similarity is comparable to or higher than within-class
similarity. This explains the §4.7.1 finding that pair-unsafe-only
scores 11/64: the pair heads rely on linear-classifier geometry over a
non-separated cluster, so their probabilities are useful only when
multiplied against the scenario distribution (the cascade rule).

The most striking case is **pair3_Forklift**, where every cosine —
within and between — converges to ~0.98. This is the geometric
fingerprint of a degenerate head trained on too few samples (n_train ≈
60 across two classes); it produces useful logits but the embedding
geometry collapses.

### 4.8.4 t-SNE projections

Figure 4.1 shows 2-D t-SNE projections (perplexity = 10) of each
embedding space, written to `figures/embedding_tsne_*.png` by
`run_probe_tsne.py` — the same run that produces the cosine tables above.
`figures/` is otherwise under `.gitignore`; of the six panels only
`embedding_tsne_raw.png` is committed. The remaining five are regenerated by
re-running that script, which needs the head checkpoints in
`sweep_results/pairwise/` and the feature cache
`sweep_results/pairwise/features/all8_test.pt` — present in the working tree
but not distributed (§3.6). The six are:

- `embedding_tsne_raw.png` — raw YOLO features; classes interpenetrate.
- `embedding_tsne_direct4_s314.png` — direct4 head; OpenedPanel and
  CarryOverload form clean clusters; WalkwayViol and UnauthorInt sit
  partially overlapping.
- `embedding_tsne_pair0_Walkway.png`, `..._pair1_Intervention.png`,
  `..._pair2_Panel.png`, `..._pair3_Forklift.png` — pair-head spaces; only
  pair0 and pair2 produce visible target-pair separation; pair3 collapses
  to a single elongated band.

**Conclusion.** The head's contribution is verifiable in geometry, not
just in accuracy. Three of four pair heads rely on classifier-boundary
power rather than cluster separation, which both (a) explains the
single-sample lift from cascade fusion (§4.7.1) and (b) limits how much
further accuracy can be extracted from the pairwise contrastive
objective at this annotation scale.

### 4.8.5 Cross-dataset transfer — UCF-Crime zero-shot probe

To test whether the head learns *safety semantics* or *visual textures*,
the trained ensemble (no retraining, no fine-tuning) was applied to
**267 anomaly clips from UCF-Crime** [22] — three different surveillance
domains (Burglary, Fighting, RoadAccidents) sharing low-level visual
cues (CCTV viewpoint, low light, indoor scenes, motion) with Unsafe-Net
but having **no semantic overlap** with the four industrial unsafe-
behaviour classes. UCF-Crime ground-truth labels do not map onto the
Unsafe-Net class set, so the metric is *structural* — prediction
distribution and confidence — not top-1 accuracy.

**Table 4.23 — Zero-shot transfer to UCF-Crime anomalies (N = 267)**

| Metric                                        | Value                       |
| --------------------------------------------- | --------------------------: |
| Predicted as **WalkwayViol**                  | 7 / 267 (2.6 %)             |
| Predicted as **UnauthorInt**                  | 42 / 267 (**15.7 %**)       |
| Predicted as **OpenedPanel**                  | 10 / 267 (3.7 %)            |
| Predicted as **CarryOverload**                | 208 / 267 (**77.9 %**)      |
| Mean ensemble max-prob (confidence)           | 0.747                       |
| In-domain Unsafe-Net test mean max-prob       | 0.804                       |
| Frac. high-confidence predictions (max-p>0.5) | 88.0 %                      |
| Uniform-baseline reference rate per class     | 25 %                        |

**Per UCF-Crime class (frac → UnauthorInt; frac → CarryOverload):**

| UCF class      |   N | Walk | Unauth | Panel | Forklift | f→Unauth | mean p | high-conf |
| -------------- | --: | ---: | -----: | ----: | -------: | -------: | -----: | --------: |
| Burglary       | 100 |    3 |     20 |     4 |       73 |    0.200 |  0.757 |     0.880 |
| Fighting       |  50 |    1 |      9 |     0 |       40 |    0.180 |  0.764 |     0.880 |
| RoadAccidents  | 117 |    3 |     13 |     6 |       95 |    0.111 |  0.732 |     0.880 |
| **Combined**   | **267** | **7** | **42** | **10** | **208** | **0.157** | **0.747** | **0.880** |

*Source:* `sweep_results/ablations/ucf_crime_zeroshot.json`,
`run_ucf_crime_zeroshot.py`. UCF-Crime download status at evaluation:
267 of 450 target anomaly clips fetched (Burglary fully, Fighting fully,
RoadAccidents partial); the remaining classes (Vandalism, Stealing) are
future work — the script is fully resumable.

**Three findings.**

1. **Frac → UnauthorInt = 15.7 %**, which is **below the uniform 25 %
   baseline** — i.e. the head actively *routes away* from the class
   that is the closest semantic match to "unauthorized intervention".
   Even on Burglary alone (the most semantically aligned UCF-Crime
   class), the rate is only 20 % — still under the chance baseline.
   The pair1 head, which §4.8.3 identified as the only pair that
   learned non-trivial cluster separation, does **not** generalise its
   decision rule to a different surveillance domain. This is direct
   evidence that the head learned the *Unsafe-Net distribution of
   "person-near-equipment"* rather than a transferable concept of
   "unauthorized presence".

2. **78 % of clips collapse onto CarryOverload** — the smallest
   training class in Unsafe-Net (8 clips) and the one with *no* visual
   overlap with any UCF-Crime anomaly category. Across all three UCF
   classes the collapse is consistent (Burglary 73 %, Fighting 80 %,
   RoadAccidents 81 %), so it is not a Burglary-specific artefact. The
   small CarryOverload training set produces a head whose decision
   boundary on out-of-distribution data is dominated by low-level
   CCTV-texture cues (camera angle, indoor lighting, video compression
   artefacts) rather than by forklift- or load-related features. This
   is the small-sample failure mode predicted by the §4.8.3 cosine
   analysis (where pair3_Forklift's embedding cosines collapsed to
   ~0.98), now visible at the cross-dataset level.

3. **Confidence drops only 0.057 (0.747 vs 0.804)** between in-domain
   and cross-domain, and **88 % of predictions cross the high-
   confidence threshold (max-p > 0.5)** — *higher* than the in-domain
   high-confidence rate. The ensemble is therefore substantially
   **over-confident on out-of-distribution data** — a calibration
   failure that any safety-monitoring deployment would have to gate
   explicitly (e.g. with a binary in-distribution classifier or an
   entropy threshold), since the per-class probabilities themselves do
   not signal "I have not seen this before".

**Implication for Ch1 contribution (4).** §4.5–§4.6 already argued
that the residual gap to the paper's 95.81 % is annotation-volume
bound. This subsection adds the orthogonal evidence that the
representation learned at this annotation scale is *not* a generic
"unsafe-behaviour detector" — it is a *Unsafe-Net-specific* classifier
that does not survive a domain shift. Both findings point in the same
direction: more (and more diverse) supervised data, not a better
architecture, is the lever that moves the high-accuracy Pareto corner.

## 4.10 Summary

The YOLO–Transformer pipeline reaches a **best-case 87.50 % top-1
(89.28 % macro)** on the Unsafe-Net 4-class unsafe subset. Under the
**blind held-out protocol the frozen single-head pipeline (§4.7.11) scores
≈ 52/64 (≈ 81 % top-1, macro ≈ 84 %)** — below the ConvLSTM-only no-bbox
baseline on top-1 (57/64 = 89.06 %) and short of macro parity (85.78 %).
The best zero-bbox configuration — **hybrid + frame-gate (§4.6.2) — lifts
the blind honest accuracy to 53/64 (82.8 % top-1, 85.4 % macro), reaching
macro parity with the no-bbox baseline even blind; this is P1's defensible
best, superseded on this corpus by P3's 54/64 (Ch7) and, as the thesis headline,
by the R48 lane (Ch5 §5.6.9).** The
"+3.5 pp macro win" reported under test-selected evaluation does not
survive the blind protocol and is withdrawn; so is the cascade/ensemble,
which a blind configuration-selection rejects in favour of the single
direct-4 head on every seed. What is robust is the *diagnosis*: the gap
to the paper's full pipeline (62/64 = 96.88 %, Table 6) is concentrated
on the WalkwayViol ↔ UnauthorInt pair and is demonstrated — through ten+
independent experiment tracks (including an 86 M-parameter VideoMAE
backbone), the V10 bbox-finetune negative result, the cross-ensemble
negative result, and the UCF-Crime domain-shift probe — to be
**data-volume-bound, not architecture-bound**: even targeted
behaviour-class annotation at the 400-frame scale *degrades* accuracy
because the COCO prior is removed faster than it is replaced. Chapter 5
takes the deployed checkpoint forward to a measured edge system (the
thesis's primary delivered contribution); Chapter 6 frames the
future-work path around scaling the annotation budget rather than the
architectural modifications already attempted here.

---

## References

Citations in this chapter use the Chapter 2 numbering scheme (to be merged
into a unified bibliography at final thesis compilation).

[17] O. Önal and E. Dandıl, "Unsafe-Net: YOLO v4 and ConvLSTM based
     computer vision system for real-time detection of unsafe behaviours
     in workplace," *Multimedia Tools and Applications*, vol. 84, no. 29,
     pp. 34967–34993, 2024. DOI: 10.1007/s11042-024-19276-8.
     Dataset: O. Önal and E. Dandıl, "Video Dataset for Safe and Unsafe
     Behaviours," Mendeley Data, 2024, CC BY 4.0.
     DOI: 10.17632/xjmtb22pff.1.
     https://data.mendeley.com/datasets/xjmtb22pff/1
