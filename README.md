# Real-Time Unsafe Behaviour Detection — Hybrid YOLOv11 + Transformer

> **Capstone / Thesis Project** — End-to-end system for industrial safety monitoring using a novel **YOLO–Transformer** architecture on the Unsafe-Net video dataset.

## Overview

This project develops an end-to-end unsafe-behaviour detection system on the **Unsafe-Net video dataset** (566 train / 125 test, 8 classes). We propose a hybrid **YOLOv11 + Transformer** architecture with two training paradigms:

1. **Frozen backbone** — pre-extracted spatial tokens + Transformer classification heads
2. **End-to-end fine-tuning** — partial backbone unfreeze + joint training (🆕)

### Key Results

| Metric | Frozen Backbone | E2E Fine-Tune |
|---|---|---|
| **4-class unsafe accuracy** | 90.63% (cascade) | 🔄 In progress |
| **Binary safe/unsafe accuracy** | 83.2% (125 clips) | 🔄 In progress |
| **Paper baseline (YOLO+ConvLSTM)** | — | **95.81%** (target to beat) |
| **Paper baseline (ConvLSTM only)** | — | **85.78%** |

### Frozen Backbone Results (Completed)

| Metric | Value |
|---|---|
| 🏆 **4-class cascade accuracy** | **90.63% (58/64)** — beats ConvLSTM baseline by +4.85pp |
| 🏆 **Binary accuracy (best)** | **83.2% (104/125)** |
| 📊 **F1-macro (binary)** | **83.0%** |
| ⚡ **ONNX inference latency** | **1.39 ms** (CPU, head only) |
| 📦 **ONNX model size** | **18.1 MB** |
| 🔧 **INT8 quantized size** | **13.6 MB** (no accuracy loss) |

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

### Thesis Contributions

1. **Novel YOLO–Transformer architecture** for video-based unsafe behaviour detection with multi-stream gated temporal fusion
2. **End-to-end fine-tuning pipeline** that breaks the frozen-backbone accuracy ceiling
3. **Evidence that frozen object-detection features lack safety-rule context** — cosine similarity 0.94–0.996 between safe/unsafe pairs
4. **Edge-deployable** system: ONNX export + INT8 quantization, 1.39ms inference on CPU
5. **Comprehensive evaluation**: multi-seed, multi-architecture, ablation study, cascade vs direct

## Paper Comparison

| Approach | Task | Test Set | Accuracy |
|---|---|---|---:|
| **Paper (YOLO+ConvLSTM)** | 4 unsafe classes | 64 clips (YOLO-filtered) | **95.81%** |
| Paper (ConvLSTM only) | 4 unsafe classes | 64 clips | **85.78%** |
| **Ours: frozen cascade** | 4 unsafe classes | 64 clips (raw video) | **90.63%** |
| **Ours: frozen binary** | **Safe/Unsafe (8→2)** | **125 clips** | **83.2%** |
| **Ours: E2E fine-tune** | 8-class / binary / 4-class | 125 / 64 clips | 🔄 |

> See [`GAP_ANALYSIS.md`](GAP_ANALYSIS.md) for detailed comparison and analysis.

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

| # | Method | 4-unsafe | Binary | Key Innovation |
|--:|---|---:|---:|---|
| 1 | Majority class | — | 51.2% | Baseline |
| 2 | Linear probe (spatial stats) | — | 68.0% | Mean+std+max pooling |
| 3 | Frozen Transformer (best) | — | **83.2%** | Gated multi-stream |
| 4 | Frozen cascade (router+binary) | **90.63%** | — | Scenario specialization |
| **5** | **E2E fine-tune** | **🔄** | **🔄** | **Backbone unfreeze** |

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
