# Chapter 3 — Methodology

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

This chapter specifies (3.1) the Unsafe-Net dataset used throughout the
thesis, (3.2) the YOLO–Transformer architecture, (3.3) the multi-stream
gated temporal fusion design of the Transformer head, (3.4) the training
pipeline, and (3.5) the ensemble and probability-calibration procedure that
yields the best-case 56/64 test-selected result. (The originally reported
57/64 calibrated number is re-classified in §4.7.7 as test-set tuning;
the honest, leakage-free accuracy of the frozen V9 pairwise-contrastive
pipeline is the blind held-out ≈ 52/64 established in §4.7.11; the best
zero-bbox configuration (hybrid + frame-gate, §4.6.2) raises this to
53/64 (85.4 % macro) — the 56/64 here is a best-case upper bound,
retained because it is the deployed checkpoint.)

## 3.1 Datasets

Two corpora are used, in two lanes that are never compared. **Unsafe-Net**
(the remainder of §3.1) carries the 4-class multi-entity lane of Chapter 7
and the edge-export measurements of Chapter 5. **iSafetyBench** (§3.1.1)
carries the binary headline lane, measured in Ch5 §5.6.9 and carried into
Chapters 4 and 6. Different label spaces, different protocols, different N:
no number from one lane may be compared with, summed with, or carried into
the same sentence as a number from the other without its qualifiers.

The Unsafe-Net dataset [17 — Chapter 2] is used without modification.

**Source and size.** 691 MP4 clips at 1920 × 1080 and 24 FPS, captured
across 39 days from a single industrial facility, with a single behaviour
label per clip. The published train / test split is **566 train / 125
test**.

**Class distribution.** 8 classes (Table 3.1), grouped into 4 safe and 4
unsafe categories. The headline evaluation metric — consistent with the
paper [17] — restricts the test set to the 4-class **unsafe subset** of 64
clips (chosen because the original paper's confusion matrix, Table 6, uses
this subset).

**Table 3.1 — 4-class unsafe test subset composition**

| Class ID | Name                              | # Test Clips |
| -------: | --------------------------------- | -----------: |
|        0 | Safe-Walkway Violation            |           32 |
|        1 | Unauthorized Intervention         |           11 |
|        2 | Opened Panel Cover                |           13 |
|        3 | Carrying Overload with Forklift   |            8 |
|          | **Total**                         |       **64** |

**Data-loader pipeline.** Each clip is decoded with OpenCV, then uniformly
sub-sampled to T = 32 frames. Each frame is resized to 224 × 224 (PIL
bilinear) and normalized with ImageNet statistics. Augmentation during
training includes random horizontal flip (*disabled for ensemble inference —
see §4 TTA analysis*), temporal offset jitter (±4 frames), and color
jitter. The same 32-frame sampling is reused at inference for reproducible
evaluation.

**Feature cache.** To amortize the cost of backbone inference across the
many architecture sweeps in Chapter 4, spatial tokens are pre-extracted
once and cached to disk under `features/` (one `.pt` file per clip;
shape `(T, 16, 512)` for the YOLOv11s layer-22 output). Runtime-backed
online extraction is retained for the end-to-end deployment in Chapter 5.

### 3.1.1 iSafetyBench — the headline corpus (binary hazard/normal)

The thesis headline is measured on a second, larger corpus:
**iSafetyBench** [26], a video-language benchmark for safety in industrial
environments released at the VISION'25 workshop of ICCV 2025. It is
distributed under CC BY-NC-SA 4.0 (non-commercial, share-alike) and used
here for academic research only. Clips are YouTube-sourced industrial
footage, trimmed by the benchmark authors.

**The original task is not our task.** iSafetyBench is built for
*open-vocabulary multi-label* recognition over 165 categories — 98 routine
activities and 67 hazardous events — evaluated by multiple-choice prompting
of vision-language models. This thesis recasts the corpus as **binary**
hazard/normal clip classification with a trained head. The recasting is a
deliberate simplification, and it makes every accuracy reported here
**non-comparable with any number in the source paper**: different label
space, different protocol, different model class. No comparison with the
benchmark's published VLM results is made or implied anywhere in this
thesis.

**Composition (measured from the distributed files, not quoted).** 1,100
clips: 420 `hazard` (38.2 %) and 680 `normal` (61.8 %). Majority-class
accuracy is therefore 61.8 %, so **balanced accuracy** is the reported
metric throughout and the imbalance cannot inflate a headline. Resolution
spans 320 × 240 to 1280 × 720; frame rate is median 29.970 fps (Chapter 5
measures both over all 1,100 clips for the real-time contract).

**Source grouping and the LSO-1100 protocol.** The loader derives a source
key from the first 11 characters of each filename — the YouTube video id —
via `SRC_RE = r'([A-Za-z0-9_-]{11})_'`. That yields **151 groups**, of which
**0 contain both classes** (76 hazard-only, 75 normal-only). All splits are
GroupKFold over this key, referred to below as **LSO-1100** (leave-source-out
over 1,100 clips), so no source video is ever split across train and test.

Two properties of the key are recorded for exactness. One filename,
`VFCTDVt5B8_trim_7.mp4`, carries a 10-character id, fails the pattern, falls
back to its own basename, and forms a singleton group. And the key is
strictly *coarser* than a filename-level source parse: it merges
`3UVQRVJy9g0_clip0` with `3UVQRVJy9g0_clip1`, two trims of one video that a
naive `_trim_` parse separates (giving 152 groups). The converse was checked
and never occurs — no filename-level source is split across two key groups —
so the grouping can only be stricter than necessary, never leaky.

**Pre-registered floor, printed beside every accuracy claim.** Before any
encoder was trained on this corpus, a 3-feature detector-count rule
(persons/frame, objects/frame, and their ratio) was fitted under identical
GroupKFold folds as a gate against the count shortcut: a corpus a count rule
can solve is not worth a temporal encoder. On a 300-clip balanced probe it
reaches **63.3 % balanced accuracy leave-source-out** (67.0 % under random
folds; 50.0 % majority). The gate passed — above chance, far from saturating
— and that number became the **pre-registered floor**: every
balanced-accuracy claim on iSafetyBench is reported against 63.3 %,
recomputed on the same folds, never against 50 %.

**Three limitations inherited from the corpus**, each repeated where the
affected result appears:

1. *Source–label confounding is diluted, not cured.* With 0 of 151 sources
   carrying both classes, a model can still in principle fit "this scene is
   a hazard scene". Leave-source-out leaves that strategy unrewarded across
   folds, and 151 groups make the test statistically powered where
   Unsafe-Net's 8 groups do not. But the property is intrinsic to any
   YouTube-collected safety corpus; the protocol bounds it, it does not
   remove it.
2. *No ground-truth boxes.* The corpus ships no detection annotations, so no
   detection mAP is quoted on it anywhere in this thesis, and the detection
   quality measured on Unsafe-Net must not be assumed to transfer.
3. *The entity-to-zone geometry channel is identically zero.* That channel is
   defined only where per-view zone polygons ship (§7.4); iSafetyBench ships
   none. Relations on this corpus are therefore entity-to-entity, and the
   zone-geometry framing of Chapter 7 does not apply to the headline lane.

## 3.2 Architecture Overview

### 3.2.0 Deviation from the approved proposal — RT-DETR → Transformer head

The approved proposal cited RT-DETR [4] as a candidate temporal
encoder (Ch1 §1.1). The implemented system departs from that
suggestion in two respects, disclosed here for transparency:

1. **No RT-DETR encoder.** Preliminary trials with an RT-DETR-style
   decoder over YOLOv11 layer-22 tokens did not improve accuracy
   over a plain Transformer encoder on the same features (52–53/64
   vs 55/64), while substantially increasing training time and
   parameter count. RT-DETR was therefore dropped before the
   experiment-track sweeps reported in Chapter 4.
2. **Frozen spatial backbone.** The proposal left open whether the
   YOLO backbone would be fine-tuned end-to-end. Fine-tuning was
   attempted (V6-ft, V10) and is reported as a negative result in
   §4.6; the headline pipeline freezes the COCO-pretrained YOLOv11s
   and trains only the Transformer head and ensemble.

The framing in this thesis is therefore **engineering composition of
existing building blocks** (frozen YOLOv11s + standard
Transformer-encoder head + multi-stream gating) optimised for
small-N edge deployment, not a new architectural mechanism.

Figure 3.1 (see `deployment/artifacts/figures/arch-diagram.mmd`) shows the overall
YOLO–Transformer pipeline. The pipeline comprises four stages:

```
Video (T = 32 frames)
   │
   ▼ 224×224, ImageNet norm
[YOLOv11s Backbone] — frozen COCO weights
   │  layer-22 C3k2 output, 4×4 spatial pool
   ▼
Spatial tokens:  (T, 16, 512)
   │
   ▼ mean / std / max across 16 tokens
Statistics sequence:  (T, 1536)
   │
   ▼
[Transformer Head]  — multi-stream gated temporal fusion (§3.3)
   │
   ▼
4-class logits
```

### 3.2.1 Spatial Encoder — YOLOv11s

The YOLOv11s backbone (5.0 M parameters, 21.5 GFLOPs at 640×640) is used
as a **frozen** feature extractor with COCO-pretrained weights. Inputs are
resized to 224 × 224, which keeps the end-to-end latency under the 33 ms
real-time budget while preserving enough spatial resolution for the
downstream Transformer to distinguish behaviour categories. The hook-based
extractor in `run_pairwise_contrastive.py` (`target_idx = 22`,
`pool_size = 4`) taps the **layer-22 C3k2 output** and applies adaptive
average pooling to a 4 × 4 spatial grid, yielding 16 tokens of 512-dim
features per frame.

Rationale for layer 22 over deeper (layer 23) or shallower (layer 15/19)
taps was established in preliminary multi-scale fusion sweeps:
concatenating layers 15, 19, and 22 does not improve accuracy over the
single layer-22 tap (53/64 vs 55/64), suggesting the layer-22 bottleneck
already contains the information the Transformer can use.

### 3.2.2 Stats Aggregator

The 16 spatial tokens per frame are summarized by concatenating three
pooling statistics — **mean**, **std**, and **max** — across the spatial
axis, producing a 1536-dim feature vector per frame. Stats aggregation is
preferred over flattening (which would produce 16 × 512 = 8192 dims) for
three reasons: (a) it enforces permutation invariance over the 4 × 4 grid,
(b) it compresses the per-frame feature by 5.3×, and (c) it outperformed
max-pool-only and mean-pool-only ablations in preliminary sweeps.

## 3.3 Transformer Head — Multi-Stream Gated Temporal Fusion

The Transformer head is the trained component of the pipeline and is the
primary design contribution. Figure 3.2 (to be produced) illustrates the
four input streams and the gating mechanism.

### 3.3.1 Input Streams

Four temporal streams are constructed from the (T = 32, 1536) sequence:

1. **Raw appearance** — the unmodified sequence.
2. **1st-order temporal derivative** — `x[t] - x[t-1]`, zero-padded at
   `t = 0`. Captures per-frame motion.
3. **2nd-order temporal derivative** — derivative of the derivative.
   Captures acceleration / change of motion.
4. **Windowed statistics** — sliding-window mean, std, max across a
   w = 4-frame window. Captures local temporal context invariant to
   frame alignment.

Each stream is independently projected into a shared d = 256 embedding
space by a linear layer, producing four (T, 256) tensors.

### 3.3.2 Learned Gating

A small gating MLP consumes the concatenated projections
((T, 4 × 256)) and produces per-timestep, per-stream scalar weights via
softmax over streams. The four streams are fused as a weighted sum,
yielding a single (T, 256) token sequence. Gating weights are learned
end-to-end with classification loss. The retraining ablation in §4.7.8
(Table 4.18) shows that on the V9 ensemble, the gated variant
*underperforms* the unweighted average (55/64 vs 56/64) at N = 64 —
consistent with small-sample overfitting of the gate parameters; the
multi-stream design is retained because the streams are not redundant
in the per-stream ablation (Table 4.16).

### 3.3.3 Transformer Encoder

A standard Transformer encoder follows, with the configuration:

| Parameter        | Value |
| ---------------- | ----- |
| Layers           | 4     |
| Attention heads  | 4     |
| Model dim (d)    | 256   |
| MLP dim          | 1024  |
| Dropout          | 0.3   |
| Positional enc.  | Learned, length 32 |

A learnable **[CLS]** token is prepended; after the final encoder layer,
the [CLS] embedding is consumed by a 2-layer MLP classifier (Linear 256 →
128 → 4) with GELU activation and 0.2 dropout.

### 3.3.4 Parameter Count

The full Transformer head has **~2.1 M parameters** (about 40% of the
backbone's 5.0 M). The complete pipeline (backbone + head) weighs in at
~7.1 M parameters, an order of magnitude smaller than VideoMAE-base
(86 M), which is relevant for the edge-deployment objective.

## 3.4 Training Pipeline

### 3.4.1 Loss Function

Primary loss is **cross-entropy** on the 4-class output with label
smoothing ε = 0.05. The pairwise-contrastive track (V9, §4.2.2)
additionally employs a triplet margin loss on [CLS] embeddings for
pairs of visually similar classes (WalkwayViol vs UnauthorInt),
which yielded the best *test-selected* result (56/64 = 87.5 %, macro
89.28 %). Note that this is a best-case figure; the honest blind held-out accuracy
of the frozen V9 pairwise-contrastive pipeline is ≈ 52/64 (§4.7.11),
and a blind configuration-selection rejects the contrastive cascade in
favour of the single direct-4 head. (The best zero-bbox configuration,
hybrid + frame-gate §4.6.2, achieves 53/64 = 85.4 % macro.)

### 3.4.2 Optimizer and Schedule

- **Optimizer:** AdamW, lr = 1e-4, weight_decay = 0.05.
- **Schedule:** cosine decay to 1e-6 over 60 epochs, with 5-epoch warmup.
- **Batch size:** 16 clips (limited by the host's 16 GB unified memory).
- **Device:** Apple M3, MPS backend in PyTorch. (The YOLO detector is
  retrained separately on a Kaggle GPU runtime; every head trained in this
  chapter runs on the M3 host.)
- **Early stopping:** validation macro accuracy, patience 10 epochs.
- **Seed sweep:** 7–10 random seeds per configuration to quantify
  run-to-run variance; best single seed is selected for ensemble
  membership.

### 3.4.3 Head Variants Trained

The pairwise-contrastive track (best ensemble source) trains five related
heads on the same features:

- **direct4** — direct 4-class classifier on the unsafe subset.
- **scenario** — binary safe/unsafe classifier across all 8 classes.
- **pair{0-1, 0-3, 2-3, 0-2}** — four binary classifiers specialized to
  each visually similar class pair.

Pair heads are trained on the relevant two-class subset of the training
data, then combined at inference via scenario-gated probability fusion
(§3.5).

## 3.5 Ensemble and Calibration

### 3.5.1 Cascade Probability Fusion

The cascade prediction combines the scenario classifier's unsafe
probability with each pair head's within-pair probability:

```
p_cascade(class c) = p_scenario(unsafe_pair containing c) × p_pair(c | pair)
```

Pair assignments: class 0 ∈ pairs {0-1, 0-2, 0-3}; class 1 ∈ pair {0-1};
class 2 ∈ pairs {2-3, 0-2}; class 3 ∈ pairs {0-3, 2-3}. Per-class
probabilities are L1-normalized across the 4 classes.

### 3.5.2 Ensemble Averaging

The final ensemble averages **direct4** and **cascade** softmax
distributions:

```
p_final(c) = 0.5 × p_direct4(c) + 0.5 × p_cascade(c)
```

This ensemble is the best-case 56/64 test-selected result; under blind
configuration-selection (§4.7.11) the single direct-4 head is preferred
over this ensemble, and the honest accuracy of the frozen V9
pairwise-contrastive pipeline is ≈ 52/64. The best zero-bbox
configuration (hybrid + frame-gate, §4.6.2) raises this to 53/64
(85.4 % macro).

### 3.5.3 Temperature Calibration

The deployed pipeline reports the **uncalibrated** ensemble probabilities.
The §4.7.6 temperature sweep shows that test-set top-1 is robust over
T ∈ [0.7, 1.3] and that no temperature setting improves accuracy
above 56/64. The earlier code additionally performed a 20 000-step
random search over per-class additive biases on the test set itself.
§4.7.7 audits and *withdraws* this calibration: when the same search
is run honestly on the training set (Table 4.13a), the resulting bias
*reduces* test accuracy to 53.4 ± 1.2 / 64 — the +1 sample lift on
test is pure test-set memorization, not a generalizable signal. The
best-case uncalibrated figure is **56/64 = 87.5 %**; the defensible
*honest* top-1 of the frozen V9 pairwise-contrastive pipeline, once
epoch/config/seed selection is also moved off the test set, is the blind
held-out **≈ 52/64** of §4.7.11. The best zero-bbox configuration
(hybrid + frame-gate, §4.6.2) raises this to **53/64 (85.4 % macro)**,
which is the best honest accuracy **of the P1 pipeline described in this
chapter**. It is *not* the thesis headline: the headline is the R48
lane (CLIP ViT-B/32 + temporal Transformer, 84.2 % balanced accuracy over
N = 1,100 under LSO-1100, Ch5 §5.6.9). On this 4-class ESTA corpus the best
figure is P3's blind 54/64 (84.4 % top-1, 88.4 % macro, Ch7), the only ESTA
pipeline carrying evidence beyond N = 64. See the pipeline-attribution banner
at the head of this chapter.

## 3.6 Reproducibility

All training and inference scripts are in the repository, together with
the small result artifacts the thesis cites as evidence (the ablation and
evaluation JSONs under `sweep_results/`, ~164 KB total, explicitly
un-ignored in `.gitignore`). The **trained weights and feature caches are
not distributed** — `sweep_results/**/*.pt` (~300 MB, including the 125 MB
`all8_test.pt` feature cache and the 18 MB `detector_best.pt`) stays under
`.gitignore`, as does `deployment/artifacts/` and the 94 MB `figures/gradcam/`.
Every number in Chapters 4 and 7 is therefore re-derivable from the committed
JSONs, and re-*computable* only after regenerating those caches with the
scripts below:

- `run_pairwise_contrastive.py` — produces the 56/64 ensemble.
- `run_yoloworld_push.py` — produces the 55/64 YOLO-World baseline.
- `run_probe_tsne.py` — regenerates the Chapter 4 interpretability
  artifacts: the cosine tables (§4.8.1–§4.8.3) and the Figure 4.1 t-SNE
  panels, from the checkpoints in `sweep_results/pairwise/`.
- `run_ablation_*.py` (7 scripts) — regenerate the ablation JSONs under
  `sweep_results/ablations/`.

Two scripts cited in earlier drafts have been removed from the tree and are
not part of the reproduction path: `run_finetune_e2e.py` (end-to-end
fine-tuning, never used in the final ensemble) and `plot_figures_binary.py`
(a superseded 125-clip *binary* experiment, not a Chapter 4/5 figure
generator — its last version is recoverable from git history at `6f780cd`
and `cedc678` respectively).

Each training run produces a `results.json` checkpoint containing: seed,
config, per-class confusion, per-epoch training / validation curves, and
the full set of test-set logits for every clip. These logit tensors are
the inputs to the ensemble and calibration code and guarantee exact
reproducibility of the headline numbers — for a reader who regenerates
them, since the `.pt` tensors themselves are not in the distribution.

---

## References

Citations in this chapter use the Chapter 2 numbering scheme (to be merged
into a unified bibliography at final thesis compilation).

[17] O. Önal and E. Dandıl, "Unsafe-Net: YOLO v4 and ConvLSTM based
     computer vision system for real-time detection of unsafe behaviours
     in workplace," *Multimedia Tools and Applications*, vol. 84, no. 29,
     pp. 34967–34993, 2024. DOI: 10.1007/s11042-024-19276-8.

[26] R. Abdullah, Y. S. Rawat, and S. Vyas, "iSafetyBench: A video-language
     benchmark for safety in industrial environment," in *Proc. IEEE/CVF
     Int. Conf. Computer Vision (ICCV) Workshops — VISION'25*, 2025.
     arXiv:2508.00399. Licensed CC BY-NC-SA 4.0.
     Project page: https://isafetybench.github.io/

