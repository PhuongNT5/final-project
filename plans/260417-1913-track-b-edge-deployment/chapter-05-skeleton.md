# Chapter 5 — Edge Deployment (Skeleton)

Prose goes in the thesis document; this file is the structured outline with measured
numbers and figure references. All numbers below are from `deployment/artifacts/`.

---

## 5.1 Introduction

- Objective 3 of the thesis: demonstrate that the trained hybrid pipeline (YOLOv11s + Transformer) can run on commodity edge hardware without accuracy loss.
- Scope: Apple M4 CPU + CoreML EP for measured numbers; Jetson Orin Nano Super for cited projection.
- Best model from Track A: ensemble of `direct4-s314` and pairwise cascade (`scenario-s7` × 4 × `pair{0..3}-s7`), ensemble top-1 = 56/64 = 87.5% on 4-class unsafe subset.
- Frame budget: 33 ms @ 30 fps real-time. Target: ≥ 30 fps on at least one configuration.

## 5.2 System Architecture

- Figure 1 (`figures/arch-diagram.mmd`): end-to-end pipeline.
- Components: Video Source → Frame Sampler (T=32) → PIL resize 224×224 + ImageNet normalize → Backbone ONNX → Stats Aggregation (mean/std/max over 16 spatial tokens) → Sliding Window → {Direct4 head, Scenario head, Pair0..3 heads} → Cascade probability fusion → Softmax-normalized ensemble.
- Why ONNX? Language-agnostic, runtime-independent; runs on ONNX Runtime (CPU), CoreML EP (ANE/GPU), and TensorRT EP (Jetson) without retraining.
- Why dynamic INT8? Weight-only quantization, no calibration dataset required, preserves accuracy better than full static quant for transformer heads.

## 5.3 Export Pipeline

- `YOLOv11BackboneTokenizer` wrapper exposes layer 22 (C3k2 output) as a flat token sequence.
- Critical detail: training pipeline uses `F.interpolate(bilinear, 7→4)` to align feature map to pool size; `adaptive_avg_pool2d(4)` drops accuracy by 8/64 (measured).
- TransformerHead exported verbatim from training code to match weights exactly; hyperparameters in `export-transformer-heads.py`.
- Dynamic INT8 via ORT `quantize_dynamic`; value_info stripping required to work around dynamo-exported reshapes in transformer attention.
- Parity verified on synthetic random inputs (Table 1): FP32 matches PyTorch within 5e-6; INT8 backbone drift 0.36 (large activation range), heads drift < 0.04.

## 5.4 Benchmarks

- Hardware: Apple M4 (10-core), macOS 14.6, ONNX Runtime 1.24.4.
- Figure 2 (`figures/latency-by-stage.png`): stacked-bar breakdown by stage across 4 configs.
- Figure 3 (`figures/accuracy-by-config.png`): ensemble accuracy across 4 configs vs training baseline.
- Figure 4 (`figures/model-size-comparison.png`): FP32 vs INT8 per ONNX file.
- Key findings:
  - CPU-FP32 end-to-end: **26.9 ms p50 / 32.5 ms p95 / 37.2 FPS**; meets 30 fps real-time.
  - CPU-INT8 end-to-end: 34.0 ms p50 / 29.4 FPS; backbone INT8 slower than FP32 on M4 due to unoptimized dequantize kernels on AMX; heads INT8 2× faster than FP32.
  - CoreML-FP32 backbone accelerates to 1.9 ms p50 (ANE) — 4.6× faster than CPU — but transformer heads fall back to CPU and become the bottleneck.
  - Accuracy preservation: FP32 matches 56/64 baseline; INT8 drops 1 sample (55/64, 85.9%).

## 5.5 Multi-Stream Pipeline

- Design: single shared backbone ORT session; frames from N streams batched to (N, 3, 224, 224) per tick. Per-stream sliding window (T=32) and transformer heads run sequentially after backbone.
- Figure 5 (`figures/fps-vs-streams.png`): aggregate FPS vs N, with ideal linear reference.
- Key findings:
  - Aggregate FPS peaks at **29.1 FPS at N=2**, declines to 20.8 at N=4 — shared CPU backbone saturates as per-batch cost grows.
  - Zero dropped frames across N=1..4 with 30 FPS reader pacing and 4-slot queue per stream — drain-on-shutdown path guarantees every enqueued frame is inferred.
  - Per-stream p50 head latency scales from 34.8 ms (N=2) → 44.6 ms (N=4) — contention on the shared backbone is visible but bounded.
  - Backbone batch utilization ~50-57% at N > 1: 5 ms bucket window too short relative to inference time; trade-off documented as limitation.

## 5.6 Jetson Orin Projection (Unverified)

- Method: cite published Ultralytics YOLOv11s benchmarks on Jetson Orin Nano Super and AGX Orin; apply ratio to our M4 end-to-end figures.
- Source: `plans/reports/researcher-track-b-jetson-orin-projection.md` — Orin Nano Super backbone 5.25 ms, AGX 2.98 ms for YOLOv11s @ 640×640.
- Adjusted for 224×224 backbone input and ratio of backbone : full pipeline at M4: projected Orin Nano Super e2e ~15 ms (67 FPS single-stream, ≥ 120 FPS aggregate at N=4 with TensorRT EP).
- Caveats: projection assumes comparable CPU-side head cost on Orin Cortex-A78AE (slower than M4 E-core); real measurement required to confirm.

## 5.7 Limitations

- **Input resolution mismatch**: training used 640×384 image crops into YOLO; deployment resizes to 224×224. Accuracy preserved on this dataset but could degrade on finer-grained scenes.
- **Test set size**: 64 videos in 4-class subset; confidence intervals wide. 1-sample swing changes headline accuracy by 1.5%.
- **Jetson numbers are projected, not measured**. Projection accuracy depends on vendor benchmarks being representative of our workload.
- **No live camera capture** tested in dashboard — only file upload. Webcam path would need `cv2.VideoCapture(0)` + browser MediaDevices bridge.
- **Batch utilization below target** in multi-stream (~53% vs 75% goal) — real-time pacing conflicts with batching efficiency; documented trade-off.

## 5.8 Conclusion

- Pipeline deployed end-to-end on M4 CPU at 37 FPS single-stream with zero accuracy loss vs PyTorch baseline.
- Four concurrent streams sustained with 20.8 FPS aggregate and no dropped frames; per-stream head p50 ~45 ms.
- Full reproduction in ≤ 7 commands from clean clone (see `deployment/README.md`).
- Artifacts for Chapter 5: 4 PNG figures + 1 Mermaid diagram + 3 data tables, all auto-generated from JSON.
