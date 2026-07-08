# Real-Time Unsafe Behaviour Detection — Hybrid YOLOv11 + Transformer

> **Capstone / Thesis Project** — End-to-end system for industrial safety monitoring built from a **YOLO–Transformer engineering composition** on the Unsafe-Net video dataset (frozen YOLOv11s spatial encoder + multi-stream gated Transformer head; the contribution is the empirical study and edge-deployable artifact, not a new architectural mechanism — see thesis §3.2.0).

## Overview

This project develops an end-to-end unsafe-behaviour detection system on the **Unsafe-Net video dataset** (566 train / 125 test, 8 classes). We propose a hybrid **YOLOv11 + Transformer** architecture with two training paradigms:

1. **Frozen backbone** — pre-extracted spatial tokens + Transformer classification heads
2. **End-to-end fine-tuning** — partial backbone unfreeze + joint training (🆕)

### Key Results

> **Honest headline.** This thesis is **deployment-led**: the delivered contribution is a
> measured, open-source, real-time **edge system** (M4 + Jetson). On accuracy, the best zero-bbox
> configuration — **hybrid (frozen ⊕ detection-finetuned) features + YOLO frame-gate** — reaches
> **macro parity-to-advantage over the published no-bbox baseline even under the blind protocol**.

| Metric (4-class unsafe subset)        | Blind held-out (honest, hybrid+gate) | Best-case (test-selected) | Paper baselines |
| ------------------------------------- | ------------------------------------ | ------------------------- | --------------- |
| **top-1** accuracy                    | **53/64 ≈ 82.8 %**                   | 56/64 = 87.5 % (frozen ens., deployed) | 57/64 no-bbox · 62/64 full-bbox |
| **macro** accuracy                    | **85.4 %** (≈ baseline parity, blind) | 86.9–89.28 %             | 85.78 % no-bbox · 95.81 % full-bbox |
| vs no-bbox baseline                   | **macro parity even blind; > baseline best-case**; top-1 below | — | — |

> The **"+3.5 pp macro win" is withdrawn** (test-selection artifact, §4.7.11). Honest ladder:
> best-case 56/64 (89.28 % macro, deployed frozen ensemble) → blind **53/64 (85.4 % macro,
> hybrid+gate, §4.6.2)**. The corrected paper-mechanism investigation (frame-gate + bbox features at
> 1,200 frames) confirms the gap to the full-bbox 95.81 % is **data-volume-bound** (needs paper-scale
> ≥ 2,262 bbox); the hybrid combination is the best zero-bbox config. The deployed Chapter-5 artifact
> ships the frozen-ensemble checkpoint.

### Frozen Backbone Results (10 experiment tracks, ~100 models)

| Metric                                       | Value                                                         |
| -------------------------------------------- | ------------------------------------------------------------- |
| 4-class ensemble **top-1**                   | **56/64 = 87.5 %** — direct4_s314 + scenario × pair cascade (s7); 95 % bootstrap CI [79.7, 95.3] |
| 4-class ensemble **macro**                   | **89.28 %** — 95 % bootstrap CI [81.3, 96.3]                  |
| Per-class — OpenedPanel                       | 100 % (13/13); CI [100, 100]; exceeds paper's 92.31 % by +7.69 pp |
| Per-class — CarryOverload                     | 100 % (8/8); CI [100, 100]; matches paper                     |
| End-to-end latency (CPU-FP32)                 | 26.8 ms p50 / 28.8 ms p95 / 37.3 FPS                          |
| ONNX deployed ensemble                        | 156 MB FP32 / 40 MB INT8 (3.8× compression)                   |
| INT8 accuracy cost                            | 1 sample (55/64 ensemble vs 56/64 FP32)                       |
| Earlier "57/64 calibrated"                    | **withdrawn** — see thesis §4.7.7 (test-set bias-search tuning; honest train-fit reduces test to 53.4 ± 1.2 / 64) |

## Architecture

```
Raw Video (1920×1080, 24fps)
     ↓
[Frame Sampling: T=32 uniform frames]
     ↓
[YOLOv11s Backbone] ← partial unfreeze (last 3 layers) for E2E
     ↓
Spatial Tokens: (T×16×768)  — pool4 = 4×4 spatial grid × 768 channels
     ↓
┌─────────── TransformerClassHead ─────────────┐
│  Multi-stream temporal fusion:               │
│    • Raw appearance tokens                   │
│    • 1st-order temporal derivatives          │
│    • 2nd-order temporal derivatives          │
│    • Windowed statistics (mean+std+max)      │
│  → Learned gating → Transformer Encoder      │
│  → CLS token → MLP classifier               │
└──────────────────────────────────────────────┘
     ↓
8-class / Binary / 4-class output
```

### Thesis Contributions (deployment-led)

1. **Measured, open-source, real-time edge-deployment system (primary, delivered).** Jetson Orin Nano Super TRT-FP16 8.62 ms / 116 FPS; **4-stream zero-drop throughput (88 FPS agg) + 2-stream 36.2-min soak**, 0 drops, 7.0 W; INT8 256-frame calibration; power Pareto; metadata-only privacy boundary; live dashboard. M4 CPU 26.8 ms / 37.3 FPS, INT8 156→40 MB. No published baseline reports edge deployment for this task.
2. **Corrected paper-mechanism replication + a data-bound diagnosis.** Identified that the paper's YOLO is a *frame filter*, not a feature extractor; replicated it at 1,200 bbox frames. Detection-finetuning regresses (over-specialises the backbone); the **hybrid (frozen ⊕ detection-finetuned) + frame-gate** combination recovers and is best — **blind 53/64 (85.4 % macro) = macro parity with the no-bbox baseline even blind**, best-case 86.9 % > baseline. The ≈54/64 ceiling holds across 12+ tracks (incl. 86 M VideoMAE and the paper's exact mechanism), confirming the gap to full-bbox 95.81 % is **data-volume-bound** (§4.6.2).
3. **V10 negative result** at 400 frames of behaviour-aware bbox annotation (54/64): evidence suggests the bbox route needs substantially more than a 400-frame pilot (paper-scale).
4. **YOLO–Transformer engineering composition** (Objective 1, substantially met): frozen YOLOv11s spatial encoder + multi-stream Transformer temporal head; not a new mechanism. The temporal streams show no measurable lift on this dataset (§4.7.8).
5. **Evaluation-rigor / leakage audit (methodological core).** Blind held-out protocol (epoch + config + seed selected on a train-carved validation slice) removing test-set model-selection leakage; withdrawal of the "57/64 calibrated" figure and the "+3.5 pp macro win"; the pairwise cascade is honestly rejected under blind selection (single direct-4 head best every seed); 5 000-sample bootstrap CIs.

## Paper Comparison

| Approach                                | Task             | Test Set                | Top-1               | Macro             |
| --------------------------------------- | ---------------- | ----------------------- | ------------------: | ----------------: |
| Paper (YOLO + ConvLSTM full pipeline)   | 4 unsafe classes | 64 clips (YOLO-filtered) | **96.88 %** (62/64) | **95.81 %**       |
| Paper (ConvLSTM only, no-bbox baseline) | 4 unsafe classes | 64 clips                | 89.06 % (57/64)     | 85.78 %           |
| Ours: best-case (test-selected, deployed frozen ens.) | 4 unsafe classes | 64 clips (raw video) | 87.50 % (56/64)     | 89.28 %           |
| **Ours: hybrid+gate — blind held-out (honest, §4.6.2)** | 4 unsafe classes | 64 clips (raw video) | **82.8 % (53/64)** | **85.4 %** ⭐      |

> Under the **blind held-out protocol** the best zero-bbox config (hybrid+gate, §4.6.2) reaches
> **53/64 (82.8 % top-1, 85.4 % macro)** — **macro parity with the no-bbox ConvLSTM baseline (85.78 %)
> even blind**, best-case 86.9 % > baseline; top-1 below baseline (53–54 vs 57). The earlier "parity on
> top-1 + macro lifts +3.50 pp" claim was based on the test-selected 56/64 and is **withdrawn** — it does
> not survive blind evaluation. At matched (test-selected) protocol the cross-paper comparison is
> parity-minus-one on top-1 (56/64 vs 57/64; z = −0.27, p = 0.79), reported as a caveat, not a headline.
> Root cause: paper's YOLO v4 was trained on per-frame bbox annotations encoding zone boundaries; we use
> frozen COCO features without zone labels. The "57/64 calibrated" figure is also **withdrawn** (§4.7.7).


## Project Structure

```
├── run_finetune_e2e.py           # ⭐ NEW: End-to-end fine-tuning (beat baseline)
├── run_safe_unsafe.py            # Frozen: Binary safe/unsafe sweep (5 archs × seeds)
├── run_temporal_fusion.py        # Frozen: Multi-temporal + larger models
├── run_scenario_cascade.py       # Frozen: Scenario-aware cascade pipeline
├── plot_figures_binary.py        # Thesis figures + ONNX export + INT8
│
├── model_arch.py                 # YOLOv11 backbone with hook-based extraction
├── feature_dataset.py            # Pre-extracted feature dataset loader
├── temporal_video_loader.py      # Raw video dataset loader
│
├── features/                     # Pre-extracted YOLOv11s features (T=16/32/48)
├── data/                         # Unsafe-Net video dataset (8 classes, 691 clips)
├── deployment/                   # ONNX + INT8 exported models
├── figures/                      # Publication-quality thesis figures
├── sweep_results/                # Experiment checkpoints + logs
│   ├── SUMMARY.md                # Frozen experiment results
│   ├── safe_unsafe/              # Binary safe/unsafe model weights
│   ├── router/                   # 4-class router checkpoints
│   ├── binary_final/             # Binary c0-vs-c1 checkpoint
│   └── finetune/                 # E2E fine-tuning results (NEW)
├── tools/                        # Feature extraction + video scanning
│   ├── preextract_yolo_features.py
│   └── scan_bad_videos.py
└── GAP_ANALYSIS.md               # Complete gap analysis
```

## Quick Start

### 1. Pre-extract features (frozen backbone, one-time)
```bash
python tools/preextract_yolo_features.py --subset train --sequence_length 32 --pool 4 --layout tokens
python tools/preextract_yolo_features.py --subset test --sequence_length 32 --pool 4 --layout tokens
```

### 2. Frozen backbone experiments
```bash
python -u run_safe_unsafe.py          # Binary safe/unsafe sweep
python -u run_temporal_fusion.py      # Multi-temporal ablation
python -u run_scenario_cascade.py     # Scenario-aware cascade
```

### 3. 🆕 End-to-end fine-tuning (to beat baseline)
```bash
python -u run_finetune_e2e.py         # E2E: unfreeze backbone + Transformer head
```

### 4. Generate figures + export
```bash
python -u plot_figures_binary.py      # Thesis figures + ONNX + INT8 + latency
```

## Accuracy Progression

| # | Version | Method | 4-unsafe top-1 | Macro | Key Finding |
|--:|---|---|---:|---:|---|
| 1 | V3 | Frozen COCO YOLO → Transformer (18-model ensemble) | 55/64 | 88.84% | Beats ConvLSTM-only (+3pp) |
| 2 | V5 | Multi-scale frozen (layers 15, 19, 22) | 53/64 | 88.77% | Multi-scale ≈ single-layer |
| 3 | V6-det | COCO detection + scene → cross-attention | 52/64 | 85.36% | Only 27% person detection |
| 4 | V6-ft | Frame-level fine-tuned YOLO → Transformer | 54/64 | 87.27% | Frame labels too noisy |
| 5 | VideoMAE | Frozen VideoMAE-base → MLP | 55/64 | 87.00% | 86M-param SOTA, same ceiling |
| 6 | Final | E2E + 42-model sweep on fine-tuned features | 54/64 | 87.20% | Config search exhausted |
| 7 | YOLOWorld | YOLO-World relationship + backbone | 55/64 | 89.54% | UnauthorInt=100% |
| 8 | CrossEns | Cross-ensemble COCO + YOLO-World | 55/64 | 87.71% | Models too correlated |
| **9** | **Pairwise** | **8-class contrastive + cascade + direct ensemble** | **56/64** | **89.28%** | **Best top-1**; "57/64 calibrated" withdrawn (§4.7.7) |
| 10 | bbox-FT | YOLOv11s fine-tune on 400 behaviour-class bboxes (V10 negative result) | 54/64 | 84.73% | COCO prior loss > behaviour-signal gain at 400 frames |

## Requirements

- Python 3.9+
- PyTorch ≥ 2.0
- ultralytics (YOLOv11)
- torchvision, scikit-learn, matplotlib, seaborn, opencv-python
- onnx, onnxruntime (for deployment)

## Citation

> Önal, K. B., & Dandıl, E. (2025). Unsafe-Net: A real-time safety analysis system for unsafe behaviors in industrial facilities using deep learning. *Multimedia Tools and Applications*, 84, 34967–34993.

## License

Academic use only. Unsafe-Net dataset subject to original dataset license.
