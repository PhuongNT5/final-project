---
title: "Track B: Edge Deployment — YOLO-Transformer Unsafe Behavior Detection"
description: "Export, benchmark, and dashboard the best pairwise ensemble on Apple M4 for thesis Objective 3."
status: done
priority: P1
effort: 34h (actual ~10h)
branch: main
tags: [edge, onnx, deployment, streamlit, benchmark, thesis]
created: 2026-04-17
---

# Track B: Edge Deployment

Deadline: 2026-05-17 (1 month). Track B is targeted for **Week 1** (2026-04-17 to 2026-04-24).

## Objective

Build the full edge deployment stack for the best ensemble (direct4_s314 + cascade_s7 = 56/64,
89.28% macro) and generate all thesis Chapter 5 artifacts. No new training. Models are frozen.

## Phase Overview

| # | Phase | Status | Est. Time | Dependency |
|---|-------|--------|-----------|------------|
| 01 | Export Pipeline | done | 6h | none |
| 02 | Benchmark Harness | done | 6h | phase-01 |
| 03 | Multi-Stream Pipeline | done | 8h | phase-01 |
| 04 | Streamlit Dashboard | done | 8h | phase-03 |
| 05 | Thesis Artifacts | done | 6h | phase-02 + phase-03 |

### Phase 02 Results (2026-04-19)

| Config | Ensemble Accuracy | e2e p50 | FPS |
|--------|-------------------|---------|-----|
| CPU-FP32 | **56/64 (87.5%)** ✓ target hit | 26.9 ms | 37.2 |
| CPU-INT8 | 55/64 (85.9%) | 34.0 ms | 29.4 |
| COREML-FP32 | 56/64 (87.5%) | 47.8 ms | 20.9 |
| COREML-INT8 | 55/64 (85.9%) | 88.2 ms | 11.3 |

Three bugs resolved during Phase 02:
1. Backbone pooling: `adaptive_avg_pool2d(7→4)` → `F.interpolate(bilinear, 7→4)` (matches training).
2. Cascade column: `pair_p[:, 0:1]` (P(safe)) → `pair_p[:, 1:2]` (P(unsafe)).
3. Preprocessing: `cv2.resize(INTER_LINEAR)` → `PIL.Image.BILINEAR` (matches torchvision training).

Fastest production config: CPU-FP32 at 37 FPS — exceeds 30 FPS real-time threshold.

Total: ~34h — feasible within Week 1 at ~5h/day.

## Key Models (Frozen)

| File | Role | Shape |
|------|------|-------|
| `yolo11s.pt` (COCO) | Backbone feature extractor | layer-22 → (T, 16, 512) |
| `runs/detect/sweep_results/finetune_bbox/yolo_finetuned/weights/best.pt` | Alt backbone (fine-tuned) | same interface |
| `sweep_results/pairwise/direct4_s314.pt` | Primary 4-class head | stats_dim→4 |
| `sweep_results/pairwise/scenario_s7.pt` | Scenario router (cascade) | stats_dim→4 |
| `sweep_results/pairwise/pair{0..3}_*_s7.pt` | Per-pair binary heads | stats_dim→2 |
| `sweep_results/pairwise/eight_s7.pt` | 8-class head (optional) | stats_dim→8 |

`stats_dim = 512 * 3 = 1536` (mean+std+max over 16 spatial tokens per frame, T=32 frames).

## Deployment Directory Layout

```
deployment/
├── export/
│   ├── backbone_fp32.onnx
│   ├── backbone_int8.onnx
│   ├── direct4_s314_fp32.onnx
│   ├── direct4_s314_int8.onnx
│   ├── scenario_s7_fp32.onnx
│   ├── scenario_s7_int8.onnx
│   ├── pair{0..3}_s7_fp32.onnx
│   └── pair{0..3}_s7_int8.onnx
├── scripts/
│   ├── export-backbone.py
│   ├── export-transformer-heads.py
│   ├── verify-onnx-parity.py
│   ├── benchmark-harness.py
│   ├── multi-stream-pipeline.py
│   └── colab-tensorrt-template.py
├── dashboard/
│   └── app.py
├── artifacts/
│   ├── benchmarks.json
│   ├── benchmarks.md
│   ├── projections.md
│   └── figures/
└── README.md
```

## Phase Files

- [Phase 01 — Export Pipeline](./phase-01-export-pipeline.md)
- [Phase 02 — Benchmark Harness](./phase-02-benchmark-harness.md)
- [Phase 03 — Multi-Stream Pipeline](./phase-03-multi-stream-pipeline.md)
- [Phase 04 — Streamlit Dashboard](./phase-04-streamlit-dashboard.md)
- [Phase 05 — Thesis Artifacts](./phase-05-thesis-artifacts.md)

## Critical Path

```
Phase 01 (Export) → Phase 02 (Benchmark) → Phase 05 (Artifacts)
                 ↘ Phase 03 (Multi-stream) → Phase 04 (Dashboard)
```

Phase 05 waits on benchmark numbers from Phase 02 AND multi-stream numbers from Phase 03.

## Previously Claimed Numbers (Need Re-verification)

- 1.39ms CPU INT8 latency — was transformer head only on frozen COCO backbone; must re-measure
  with current pairwise heads on M4
- 13.6 MB INT8 ONNX — was single transformer; cascade = multiple smaller models; re-measure total
- Both claims require Phase 01 export + Phase 02 benchmark before citing in thesis
