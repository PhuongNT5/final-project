# Edge Deployment Latency — Apple M3 vs Jetson Orin

## Introduction

This table mixes three kinds of row, and the distinction is load-bearing — read the
**Source** column before citing any figure:

- **`This work (measured, M3)`** — our pipeline, measured on Apple M3 via ONNX Runtime.
- **`This work (measured, Jetson)`** — our pipeline, measured on physical Jetson Orin Nano
  hardware (`phuongnguyen-desktop`, Linux-5.15.185-tegra-aarch64, TensorRT 10.3.0,
  2026-06-03), from `benchmarks-jetson-{fp16,int8}.json`. **These rows benchmark the P1
  pipeline** (frozen-COCO detector + multi-stream Transformer), not the P3 accuracy headline.
- **`[Ult.2025]` / `[arXiv:...]`** — **published third-party figures for a different model and
  workload**, included only as external context. They are *not* measurements of this system
  and must never be reported as our results.

M3 ORT-CPU and Jetson Ampere-GPU-with-TensorRT are distinct accelerator architectures; no
scaling ratio or projection between them is claimed.

## Side-by-Side Latency Table

| Device | Accelerator | Model | Format | Latency p50 (ms) | FPS | Source |
|--------|------------|-------|--------|-----------------|-----|--------|
| MacBook Air M3 | ORT CPU | YOLO11s backbone | FP32 | 27.3 | 37 | This work (measured, M3) |
| MacBook Air M3 | ORT CPU | YOLO11s backbone | INT8 | 31.8 | 31 | This work (measured, M3) |
| MacBook Air M3 | ORT CoreML | YOLO11s backbone | FP32 | 45.1 | 22 | This work (measured, M3) |
| MacBook Air M3 | ORT CoreML | YOLO11s backbone | INT8 | 85.0 | 12 | This work (measured, M3) |
| **Jetson Orin Nano 8GB** | **TensorRT (Ampere GPU)** | **P1 full pipeline** | **FP16** | **8.62** (p95 9.13 / p99 9.21) | **116.0** | **This work (measured, Jetson)** |
| **Jetson Orin Nano 8GB** | **TensorRT (Ampere GPU)** | **P1 full pipeline** | **INT8** | **8.24** (p95 8.58 / p99 8.71) | **121.4** | **This work (measured, Jetson)** |
| Jetson Orin Nano Super 8GB | TensorRT (Ampere GPU) | YOLO11s | INT8 | 5.25 | 190 | [Ult.2025] |
| Jetson AGX Orin 64GB | TensorRT (Ampere GPU) | YOLO11s | INT8 | 2.98 | 336 | [Ult.2025] |
| Jetson Orin NX 16GB | TensorRT (Ampere GPU) | YOLO11s | INT8 | 6.05 | 165 | [Ult.2025] |
| Jetson Orin Nano 8GB (15W std) | TensorRT | YOLOv8s | INT8 | 28.25 | 35 | [arXiv:2502.15737] |

*M3 end-to-end includes: backbone + stats-aggregation + direct4-head + cascade heads (scenario + 4 pairs). Backbone-only FP32 CPU: ~10.2 ms (consistent with prior researcher measurement). Jetson figures cover YOLO11s backbone only at 640×640.*

## Hardware Context

| Device | INT8 Compute | Power | Accelerator Type |
|--------|-------------|-------|-----------------|
| Apple M3 (MacBook Air — this work's host) | ANE present; vendor TOPS not verified here | not measured (fanless chassis) | ARM CPU + Neural Engine |
| Jetson Orin Nano Super 8GB | 67 TOPS (Ampere sparse) | 25 W | NVIDIA Ampere GPU + CUDA Tensor Cores |
| Jetson AGX Orin 64GB | 275 TOPS (Ampere sparse) | 60 W | NVIDIA Ampere GPU + CUDA Tensor Cores |

## Footnotes

1. **Orin Nano Super = standard Orin Nano 8GB module.** The "Super" designation refers to
   a firmware unlock released December 2024 (`nvpmodel -m 2`, 25 W mode) on any existing
   Jetson Orin Nano 8GB kit with JetPack 6.x. No separate hardware is required. Source:
   NVIDIA Developer Blog, December 2024 [3].

2. **Accelerator class mismatch.** M3 ORT-CPU uses ARM Cortex cores with no dedicated
   matrix unit for ONNX INT8 workloads. Jetson Orin TensorRT uses NVIDIA Ampere GPU with
   hardware INT8 Tensor Cores. Latency ratios between the two reflect accelerator
   architecture differences, not merely clock speed or TOPS. No linear projection between
   M3 and Orin latency is claimed.

3. **CoreML EP behavior.** ORT CoreML EP on M3 delegates ops to the Apple Neural Engine
   or GPU. The INT8 CoreML result (85 ms) is slower than FP32 (45 ms) because CoreML does
   not support dynamic INT8 ORT quantization natively — it falls back to CPU for
   quantized ops, adding overhead. A native CoreML `.mlpackage` export would access the ANE
   directly (expected 2–4 ms backbone-only, consistent with researcher measurement of 2.3 ms).

4. **The third-party Jetson figures are backbone-only; ours are full-pipeline.** The
   `[Ult.2025]`/`[arXiv]` rows time a YOLO backbone alone at 640×640. Our two measured Jetson
   rows time the **complete** pipeline — backbone 2.76 ms + direct4 head 1.41 ms + cascade heads
   4.44 ms = 8.62 ms end-to-end (FP16 p50) — so they are **not comparable** to the cited rows and
   the difference is not a performance gap.
   *(Corrected 2026-09-06: this footnote previously stated that a full-pipeline Jetson benchmark
   "would require hardware access not available in this study." That is no longer true — the
   measurements above were taken on physical hardware on 2026-06-03. The claim is withdrawn.)*

6. **Pipeline attribution.** The measured Jetson rows are the **P1** pipeline, whose 4-class
   accuracy on the 64-clip test set is 55/64. The thesis accuracy headline is **P3**, a different
   model; its Jetson re-export and re-benchmark is Lane B work and is **not** represented in this
   table. Do not pair P3 accuracy numbers with P1 latency numbers.

5. **Orin Nano 15W reference.** The arXiv:2502.15737 figure (28.25 ms for YOLOv8s INT8 at
   15 W) covers the standard shipping mode of Orin Nano 8GB. The Nano Super at 25 W yields
   ~1.7× speedup (5.25 / 3.0 ≈ 1.75 for YOLO11s vs YOLOv8s, same silicon). The 15W
   YOLOv8s number serves as a conservative reference for constrained-power deployment.

## References

[Ult.2025] Ultralytics. "Quick Start Guide: NVIDIA Jetson with Ultralytics YOLO."
  https://docs.ultralytics.com/guides/nvidia-jetson/. Accessed 2026-04-19.

[arXiv:2502.15737] Araujo et al. "A Performance Analysis of YOLO Models for Deployment
  on Constrained Edge Devices in Drone Applications." arXiv:2502.15737, Feb 2025.

[3] NVIDIA Developer Blog. "NVIDIA Jetson Orin Nano Developer Kit Gets a 'Super' Boost."
  https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/.
  December 2024.

[4] Seeed Studio Wiki. "Deploy YOLOv8 with TensorRT and DeepStream SDK."
  https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/. Accessed 2026-04-19.
