# Chapter 3 — Methodology

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

## 3.1 Dataset

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

Figure 3.1 (see `figures/arch-diagram.mmd`) shows the overall
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
extractor in `model_arch.py` taps the **layer-22 C3k2 output** and applies
adaptive average pooling to a 4 × 4 spatial grid, yielding 16 tokens of
512-dim features per frame.

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
- **Batch size:** 16 clips (limited by Apple M4 36 GB unified memory).
- **Device:** Apple M4 Pro (MPS backend in PyTorch).
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
which is the thesis headline accuracy.

## 3.6 Reproducibility

All training and inference scripts, trained weights (~300 MB), and ONNX
exports are included in the repository:

- `run_pairwise_contrastive.py` — produces the 56/64 ensemble.
- `run_yoloworld_push.py` — produces the 55/64 YOLO-World baseline.
- `run_finetune_e2e.py` — end-to-end fine-tuning script (not used in
  the final ensemble).
- `plot_figures_binary.py` — regenerates Chapter 4 and Chapter 5 figures
  from saved artifacts.

Each training run produces a `results.json` checkpoint containing: seed,
config, per-class confusion, per-epoch training / validation curves, and
the full set of test-set logits for every clip. These logit tensors are
the inputs to the ensemble and calibration code and guarantee exact
reproducibility of the headline numbers.

---

## References

Citations in this chapter use the Chapter 2 numbering scheme (to be merged
into a unified bibliography at final thesis compilation).

[17] K. B. Önal and E. Dandıl, "Unsafe-Net: YOLO v4 and ConvLSTM based
     computer vision system for real-time detection of unsafe behaviours
     in workplace," *Multimedia Tools and Applications*, vol. 84, pp.
     34967–34993, 2025. DOI: 10.1007/s11042-024-19276-8.
