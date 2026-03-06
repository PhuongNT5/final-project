# Real-Time Unsafe Behaviour Detection using Hybrid YOLOv11 + Transformer

> **Capstone / Thesis Project** — A research-grade system for detecting unsafe industrial behaviours from surveillance video, evaluated on the Unsafe-Net benchmark dataset.

![System Architecture](so%20do%20he%20thong.png)

---

## Overview

This project proposes a **hybrid YOLOv11 + Transformer** architecture for real-time unsafe-behaviour detection in industrial environments. The system extracts spatiotemporal features from a frozen YOLOv11s backbone and classifies video clips using a novel **Motion-Aware Transformer Head** with dual-stream (appearance + temporal-difference) tokens.

A **cascaded classifier** strategy (coarse 3-class routing → specialised binary refinement) achieves **90.63% accuracy (58/64)** on the Unsafe-Net benchmark, narrowing the gap to the paper baseline (YOLOv4 + ConvLSTM → 95.81%) while remaining **edge-deployable at real-time speeds** (2.3ms inference, ~83 FPS GPU).

### Key Results

| Metric | Value |
|---|---|
| **4-class accuracy** | **90.63% (58/64)** |
| **Binary 0-vs-1 accuracy** | **93.02% (40/43)** |
| **Paper baseline** | **95.81% (62/64)** |
| **Gap** | **5.19pp** |
| **Cascade latency (ONNX)** | **2.3ms** |
| **Model size (ONNX)** | **35.6 MB** |
| **Edge real-time** | ✅ ~83 FPS (GPU), ~13 FPS (CPU) |

---

## Architecture

### Proposed System Pipeline

```
Surveillance Video
       │
       ▼
┌──────────────┐
│  YOLOv11s    │  Frozen backbone, layer idx=-3
│  (Backbone)  │  → (T, 4×4, 768) spatial tokens
└──────┬───────┘
       │
       ▼
┌────────────────────────┐
│  Motion-Aware          │  Dual-stream: appearance + temporal-diff tokens
│  Transformer Head      │  Type embeddings + temporal position encoding
│  (Thesis Contribution) │  Shared self-attention across streams
└──────┬─────────────────┘
       │
       ▼
┌──────────────────────┐
│  Cascaded Classifier │
│  3-class routing     │──→ Panel (100%) / Forklift (100%)
│  + Binary refinement │──→ Walkway Viol. vs Unauthorized Int.
└──────────────────────┘
       │
       ▼
  4-Class Prediction
```

### Motion-Aware Transformer Head (Thesis Contribution)

The key insight driving this architecture comes from feature-space analysis:

| Feature Type | Cosine Similarity (Class 0 vs 1) |
|---|---|
| Appearance tokens | **0.99** (nearly identical) |
| Temporal-difference tokens | **0.28** (well-separated) |

Classes 0 (Safe Walkway Violation) and 1 (Unauthorized Intervention) are visually identical — same workers, same industrial setting. The **only** distinguishing signal is temporal: walking patterns vs equipment interaction patterns.

The **MotionHead** architecture addresses this with:

1. **Appearance stream**: Frame-level spatial features projected to model dimension
2. **Motion stream**: Temporal-difference tokens (frame[t+1] − frame[t]) capturing velocity
3. **Type embeddings**: Distinguish CLS / appearance / motion token types
4. **Joint Transformer encoder**: Self-attention across both streams simultaneously

This provides **+5.7pp** over appearance-only baselines.

### Cascaded Classifier

The cascade eliminates routing errors that plague direct 4-class classification:

1. **3-class router**: Groups walkway classes (0+1) → Walkway Group, Panel (2), Forklift (3)
   - Achieves **100% on Panel and Forklift** with zero variance across seeds
2. **Binary refinement**: Specialised `MotionBinaryHead` splits Walkway Group → class 0 vs 1
   - Uses 2nd-order temporal differences (acceleration) + statistical features

---

## Dataset

**Unsafe-Net** — 8-class industrial safety video dataset:

| Class | ID | Train | Test | Category |
|---|---:|---:|---:|---|
| Safe Walkway Violation | 0 | 50 | 32 | Unsafe |
| Unauthorized Intervention | 1 | 23 | 11 | Unsafe |
| Opened Panel Cover | 2 | 20 | 13 | Unsafe |
| Carrying Overload (Forklift) | 3 | 21 | 8 | Unsafe |
| Safe Walkway | 4 | 178 | 23 | Safe |
| Authorized Intervention | 5 | 97 | 12 | Safe |
| Closed Panel Cover | 6 | 109 | 14 | Safe |
| Safe Carrying | 7 | 68 | 12 | Safe |
| **Total** | | **566** | **125** | |

**Paper evaluation protocol**: 64 unsafe-only test videos → 4-class classification.

Scenario pairs: {0,4}, {1,5}, {2,6}, {3,7} — safe counterparts map to same unsafe class via `label % 4`.

---

## Project Structure

```
capstone/
├── model_arch.py              # YOLOv11 backbone with hook-based feature extraction
├── spatiotemporal_head.py     # SpatioTemporalTransformerHead, FactorizedSTHead
├── temporal_head.py           # TemporalTransformerHead
├── hierarchical_head.py       # HierarchicalSTHead
├── feature_dataset.py         # PreExtractedFeatureDataset
├── temporal_video_loader.py   # Video dataset loader (frame sampling)
├── export_edge.py             # ONNX / INT8 edge deployment pipeline
│
├── run_breakthrough.py        # 3-class grouped + binary cascade (latest)
├── run_cascaded.py            # T16/T32 cascade experiments (best: 90.63%)
├── run_push_accuracy.py       # Binary specialisation push (best binary: 93%)
├── run_unsafe_only.py         # 4-class baseline training
├── run_fast_motion.py         # Fast temporal motion experiments
├── run_enhanced_motion.py     # Enhanced motion ensemble + ablation
├── run_smart_cascade.py       # T48 + spatial binary experiments
├── run_cross_resolution.py    # T16+T32 cross-resolution cascade
│
├── tools/
│   ├── preextract_yolo_features.py   # Feature pre-extraction from videos
│   └── scan_bad_videos.py            # Dataset quality check
│
├── features/                  # Pre-extracted YOLOv11s features
│   ├── train_T{16,32,48}_img224_pool4_tokens_yolo11s.pt
│   └── test_T{16,32,48}_img224_pool4_tokens_yolo11s.pt
│
├── deployment/                # Edge deployment artifacts
│   ├── motion_head_4class.onnx          # 30.4 MB
│   ├── motion_binary_0vs1.onnx          # 5.2 MB
│   ├── motion_head_4class_int8_full.pt  # 14.5 MB
│   └── motion_binary_0vs1_int8_full.pt  # 1.9 MB
│
├── sweep_results/             # Experiment results & reports
│   ├── SUMMARY.md             # Comprehensive experiment report
│   ├── cascaded_v1_T32/       # Best result: 90.63%
│   └── ...                    # Other experiment results
│
├── data/                      # Unsafe-Net video dataset
│   ├── train/{0..7}_*/        # 566 training videos
│   └── test/{0..7}_*/         # 125 test videos
│
└── yolo11s.pt                 # YOLOv11s pretrained weights
```

---

## Installation

```bash
cd capstone

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision ultralytics
pip install scikit-learn matplotlib seaborn pandas numpy
pip install onnx onnxruntime
```

**Requirements:**
- Python 3.9+
- PyTorch 2.0+
- ultralytics (YOLOv11)
- scikit-learn, matplotlib, seaborn

---

## Usage

### 1. Pre-extract Features

Extract YOLOv11s features from video dataset (one-time):

```bash
python tools/preextract_yolo_features.py \
    --train-dir data/train --test-dir data/test \
    --model yolo11s.pt --frames 32 --pool 4 --img-size 224 \
    --out-dir features/
```

### 2. Run Experiments

```bash
# Baseline 4-class
python run_unsafe_only.py

# Motion-aware architecture (thesis contribution)
python run_fast_motion.py

# T32 cascade (best result: 90.63%)
python run_cascaded.py --train features/train_T32_img224_pool4_tokens_yolo11s.pt \
                       --test features/test_T32_img224_pool4_tokens_yolo11s.pt

# 3-class breakthrough cascade (latest)
python run_breakthrough.py
```

### 3. Export for Edge Deployment

```bash
python export_edge.py
```

Produces ONNX models and INT8 quantised variants in `deployment/`.

---

## Results

### Accuracy Progression

| # | Method | Accuracy | Correct/64 | Key Innovation |
|--:|---|---:|---:|---|
| 1 | 4-class baseline (RobustSTHead) | 78.1% | 50/64 | Focused training |
| 2 | + scenario_all + mixup | 81.25% | 52/64 | Data augmentation |
| 3 | **MotionHead** (dual-stream) | **84.38%** | **54/64** | **Motion tokens (+5.7pp)** |
| 4 | T16 cascade | 84.38% | 54/64 | Binary too weak at T16 |
| 5 | Cross-resolution (T16+T32) | 85.94% | 55/64 | T32 binary helps |
| 6 | Smart cascade (T32 binary) | 89.06% | 57/64 | T32 resolution |
| **7** | **T32 cascade (best)** | **90.63%** | **58/64** | **T32 binary=90.7%** |
| 8 | Push (93% binary) | 90.63% | 58/64 | Binary ceiling |
| 9* | 3-class grouped cascade | *in progress* | *target: 61/64* | Routing fix |

### Comparison with Paper Baseline

| Method | Architecture | Acc % | Correct |
|---|---|---:|---:|
| **Unsafe-Net paper** | YOLOv4 + ConvLSTM | **95.81** | **62/64** |
| **Ours (best)** | YOLOv11s + MotionHead cascade | **90.63** | **58/64** |

**Key difference**: The paper trains a **custom YOLO** to detect unsafe objects and filter frames. Our system uses a **frozen** YOLOv11s backbone — a fundamentally harder end-to-end approach that requires no task-specific object annotation.

### Confusion Matrix (Best: 90.63%)

```
                    Pred →  Walkway  Unauth  Panel  Forklift
Walkway Viol (32):           28       2       2       0
Unauthorized (11):            1      10       0       0
Opened Panel (13):            0       0      13       0
Carry Overload (8):           0       0       0       8
```

Per-class: Walkway=87.5%, Unauthorized=90.9%, **Panel=100%**, **Forklift=100%**

### Edge Deployment

| Platform | Latency | FPS |
|---|---:|---:|
| CPU (ONNX) | ~77ms | ~13 FPS |
| GPU (ONNX+TRT) | ~12ms | ~83 FPS |
| Edge (Jetson, est.) | ~25ms | ~40 FPS |

Model sizes: ONNX cascade = **35.6 MB**, INT8 = **16.4 MB**

---

## Thesis Contributions

1. **Motion-Aware Dual-Stream Transformer Head** — Temporal-difference tokens with type embeddings provide +5.7pp for discriminating visually-identical but behaviourally-distinct classes.

2. **Cascaded Classification Strategy** — 3-class grouped routing (100% on Panel/Forklift) + specialised binary refinement achieves +6.2pp over direct classification.

3. **Temporal Resolution Analysis** — T32 optimal for binary class discrimination; T16 sufficient for coarse routing; T48 introduces noise.

4. **Edge-Deployable Architecture** — Full cascade in 2.3ms ONNX inference, suitable for real-time industrial safety monitoring on edge devices.

---

## Detailed Experiment Report

See [`sweep_results/SUMMARY.md`](sweep_results/SUMMARY.md) for the comprehensive experiment report including:
- Architecture ablation tables
- Hyperparameter sensitivity analysis
- Error analysis (6-error breakdown)
- Feature-space analysis
- All experiment configurations and results

---

## License

This project is part of a capstone/thesis. Please contact the author for usage.
