# Jetson Orin Nano / AGX INT8 Latency — YOLO Benchmarks & Thesis Projection Strategy

**Date:** 2026-04-17
**Author:** researcher (Track B)
**Purpose:** Cite defensible Orin Nano/AGX INT8 latency in thesis Ch.5 — no hardware available

---

## 1. Hardware Disambiguation (Critical)

| SKU | GPU clock | INT8 TOPS (sparse) | Power target | Notes |
|---|---|---|---|---|
| Jetson Orin Nano 8GB (original) | 625 MHz | 40 TOPS | 15 W | HW available since 2023 |
| Jetson Orin Nano Super (= Orin Nano 8GB + sw unlock) | 1,050 MHz | 67 TOPS | 25 W | Dec 2024 free FW update; **same module**, higher clock |
| Jetson AGX Orin 64GB | full Orin SoC | 275 TOPS | 60 W | Top-of-line |

**Key fact:** Orin Nano Super is NOT separate HW. It is the standard 8GB module running at 25 W instead of 15 W via `nvpmodel -m 2`. All Ultralytics 2025 benchmarks on "Orin Nano Super" are therefore reachable on any Jetson Orin Nano 8GB kit with JetPack 6.x and super mode enabled. Source: NVIDIA Developer Blog, Dec 2024.

---

## 2. YOLO11s INT8 TensorRT Latency — Cited Numbers

### 2a. Jetson Orin Nano Super Developer Kit (= Orin Nano 8GB @ 25 W, Super Mode)

**Source:** Ultralytics official docs — `docs.ultralytics.com/guides/nvidia-jetson/`
(accessed 2026-04-17; live page, data current as of early 2025)

| Model | Format | Latency (ms/im) | FPS (derived) | mAP50-95 |
|---|---|---|---|---|
| YOLO11n | TensorRT FP32 | 7.53 | 133 | ~0.479 |
| YOLO11n | TensorRT FP16 | 4.57 | 219 | ~0.479 |
| YOLO11n | TensorRT **INT8** | **3.80** | **263** | ~0.427 |
| YOLO11s | TensorRT FP32 | 13.60 | 74 | ~0.566 |
| YOLO11s | TensorRT FP16 | 7.17 | 139 | ~0.566 |
| YOLO11s | TensorRT **INT8** | **5.25** | **190** | ~0.548 |

- Resolution: 640×640 (default)
- Runner: Ultralytics Python benchmarking harness
- JetPack: 6.x (exact version not stated in docs)
- Power mode: MAX power mode + jetson_clocks recommended but not confirmed in docs text

### 2b. Jetson AGX Orin Developer Kit 64GB

**Source:** Same — Ultralytics `docs.ultralytics.com/guides/nvidia-jetson/` (accessed 2026-04-17)

| Model | Format | Latency (ms/im) | FPS (derived) |
|---|---|---|---|
| YOLO11n | TensorRT INT8 | **2.30** | 435 |
| YOLO11s | TensorRT INT8 | **2.98** | 336 |

### 2c. Jetson Orin NX 16GB (nearest related device, for cross-check)

**Source:** Ultralytics docs / community reports (via search aggregation, accessed 2026-04-17)

| Model | Format | Latency (ms/im) |
|---|---|---|
| YOLO11n | TensorRT INT8 | 4.54 |
| YOLO11s | TensorRT INT8 | **6.05** |
| YOLO11m | TensorRT INT8 | 10.43 |

Note: The 6.05 ms figure for YOLO11s on Orin NX 16GB is consistent with the 5.25 ms on Orin Nano Super — the Nano Super with unlocked clocks performs comparably to NX 16GB (confirmed by Ultralytics blog).

---

## 3. YOLOv8s INT8 Fallback Numbers (if YOLO11s unavailable)

### 3a. Jetson Orin Nano 8GB (standard, 15 W mode)

**Source:** Araujo et al., "A Performance Analysis of YOLO Models for Deployment on Constrained Edge Devices in Drone Applications," arXiv:2502.15737, Feb 2025.

| Model | Precision | Latency (ms) | FPS (derived) |
|---|---|---|---|
| YOLOv8n | INT8 | 23.16 | 43 |
| YOLOv8n | FP16 | 26.70 | 37 |
| YOLOv8s | **INT8** | **28.25** | **35** |
| YOLOv8s | FP32 | 42.97 | 23 |

- Power: 7.4–8.7 W observed (standard 15 W mode, not super)
- Resolution: 640×640, TensorRT engine
- Caveat: this is the **original Orin Nano 8GB at 15W**, not Super mode — ~1.7× slower than Super mode numbers above.

### 3b. Jetson AGX Orin 32GB — YOLOv8s INT8

**Source:** Seeed Studio Wiki / DeepStream TRT guide (wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/, accessed via search 2026-04-17)

- YOLOv8s INT8 trtexec: **mean latency 3.2 ms = 313 FPS**
- Device: Jetson AGX Orin 32GB H01 Kit
- Resolution: 640×640
- Runner: trtexec (pure TensorRT, no Python overhead)
- Date of original post: 2023

---

## 4. Apple M-Series CPU + ANE Reference (Baseline for Projection)

### 4a. Available Data

No published benchmark found for **YOLOv8s or YOLO11s with ONNX Runtime CPU INT8 on Apple M4** specifically.

Closest available data:

| Device | Model | Runtime | Latency | FPS | Source |
|---|---|---|---|---|---|
| MacBook Air M3 | YOLO11n | CoreML (Elixir) | 15.61 ms inference | ~50 FPS | hexdocs.pm/yolo/macbook_air_m3.html |
| MacBook Air M3 | YOLO11n | PyTorch CPU | 25.8 ms | ~39 FPS | same |
| MacBook Air M3 | YOLO11m | CoreML | 39.68 ms | ~22 FPS | same |
| MacBook M4 Pro | YOLOv8n | (unspecified, likely CoreML/MPS) | ~10.8 ms | 92.6 FPS | blog.roboflow.com/putting-the-new-m4-macs-to-the-test |
| generic CPU x86 | YOLOv8s | ONNX CPU | 128.4 ms | ~7.8 FPS | Ultralytics model card |

**Critical gap:** No M4 ORT CPU INT8 numbers exist for YOLOv8s or YOLO11s in any indexed public source as of 2026-04-17. The M4 Roboflow benchmark framework is not ONNX Runtime but an unspecified inference CLI. The 128.4 ms figure is for an unidentified x86 CPU, not Apple Silicon.

### 4b. CoreML ANE vs Orin GPU (Architectural Note)

Apple M4 Neural Engine (ANE): ~38 TOPS (M4 base chip)
Jetson Orin Nano (Super): 67 TOPS (sparse INT8, Ampere tensor cores)
Jetson AGX Orin 64GB: 275 TOPS

ANE vs Orin GPU are both INT8 hardware accelerators but with fundamentally different ISAs. A direct ms-to-ms ratio would be misleading without confirmed identical workloads on both paths.

---

## 5. Recommended Thesis Presentation Strategy

### DO: Side-by-Side Table

Present three rows with independent measurements and explicit hardware + accelerator columns:

```
| Device                      | Accelerator           | Model   | Precision | Latency  | FPS  | Source     |
|-----------------------------|----------------------|---------|-----------|----------|------|------------|
| MacBook Air M4 (measured)   | ORT CPU              | YOLO11s | INT8      | X ms*    | Y    | own exp.   |
| Jetson Orin Nano Super 8GB  | TensorRT (GPU)       | YOLO11s | INT8      | 5.25 ms  | ~190 | [Ult.2025] |
| Jetson AGX Orin 64GB        | TensorRT (GPU)       | YOLO11s | INT8      | 2.98 ms  | ~336 | [Ult.2025] |
```
*Fill X ms from own ORT-CPU-INT8 measurement on M4.

Add a table footnote: "M4 ORT-CPU and Jetson TensorRT-GPU accelerators are not architecturally equivalent; numbers are independent measurements, not projections."

### DO NOT: Derive a scaling ratio

M4 CPU (ARM Cortex cores, no dedicated matrix unit for ONNX workloads) vs Orin GPU (Ampere CUDA + Tensor Cores) is CPU vs GPU — latency ratio varies by batch size, memory bandwidth, kernel optimization. Citing a ratio would be misleading and indefensible under thesis review.

### If YOLO11s Orin number is unavailable in a future snapshot

Use **YOLOv8s on Orin Nano 8GB @ 15 W = 28.25 ms** (arXiv:2502.15737) as fallback and note: "YOLOv11s is architecturally ~10–15% lighter than YOLOv8s with similar accuracy; YOLO11s latency on equivalent hardware is expected to be comparable or marginally lower." Provide YOLO11s Orin NX 6.05 ms as upper-bound reference for the Orin Nano class.

### Disclosure paragraph (suggested thesis wording)

> "Jetson Orin Nano Super 8GB latency figures are cited from Ultralytics official benchmarks (docs.ultralytics.com/guides/nvidia-jetson/, accessed April 2026), measured with TensorRT INT8 at 640×640 using the Ultralytics Python benchmarking harness. Mac M4 figures represent our own measurements using ONNX Runtime CPU EP with INT8 static quantization. The two devices employ distinct inference accelerators (ARM CPU vs. NVIDIA Ampere GPU with Tensor Cores); direct latency comparison is provided for contextual reference only. No linear projection between platforms is claimed."

---

## 6. Source Credibility Assessment

| Source | Type | Reliability | Caveats |
|---|---|---|---|
| Ultralytics docs.ultralytics.com/guides/nvidia-jetson/ | First-party (Ultralytics) | High | Power mode unconfirmed; batch=1 assumed; Python overhead included |
| arXiv:2502.15737 (Feb 2025) | Peer-reviewed preprint | Medium-High | Standard 15W Orin Nano (not Super); TensorRT used; resolution 640 |
| Seeed Studio wiki.seeedstudio.com (2023) | Third-party community | Medium | AGX Orin 32GB only; trtexec (no Python overhead) — faster than Ultralytics harness; older data |
| NVIDIA Developer Blog (Orin Nano Super announce) | First-party (NVIDIA) | High | Spec sheets, not inference benchmarks |
| blog.roboflow.com M4 benchmark | Third-party | Medium-Low | Framework unspecified; YOLOv8n only highlighted; no ms per model breakdown |
| hexdocs.pm/yolo/macbook_air_m3.html | Third-party | Low-Medium | Elixir library, not ORT; CoreML EP not CPU EP |

---

## 7. Recommended Thesis Citations

1. Ultralytics. "Quick Start Guide: NVIDIA Jetson with Ultralytics YOLO." docs.ultralytics.com/guides/nvidia-jetson/. Accessed 17 Apr 2026.
2. Araujo et al. "A Performance Analysis of You Only Look Once Models for Deployment on Constrained Computational Edge Devices in Drone Applications." arXiv:2502.15737, Feb 2025.
3. NVIDIA. "NVIDIA Jetson Orin Nano Developer Kit Gets a 'Super' Boost." developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/. Dec 2024.
4. Seeed Studio. "Deploy YOLOv8 with TensorRT and DeepStream SDK." wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/. Accessed 17 Apr 2026.

---

## 8. Unresolved Questions

1. **Exact JetPack version** used in Ultralytics 2025 benchmarks — docs say "JetPack 6.x" generically; matters for TensorRT version (8.x vs 10.x) which affects INT8 performance.
2. **Power mode confirmation** — Ultralytics docs recommend MAX + jetson_clocks but don't confirm it was used for the tabulated numbers. Issue #22479 shows a user got 3–4× worse results at 25W without jetson_clocks — gap is real.
3. **YOLO11s + small transformer head overhead** — our model adds a ~2M-param transformer decoder. Published benchmarks cover the YOLO backbone only. Additional latency for the head is unmeasured on Orin; estimate ~10–20% overhead (rough), needs ablation.
4. **M4 ORT CPU INT8 numbers** — no public baseline exists; we must measure ourselves. Recommend running `onnxruntime.InferenceSession` with `CPUExecutionProvider` and INT8 ONNX model, 640×640 batch=1, 100 warmup + 500 timed iterations.
5. **YOLOv11s vs YOLO11s naming** — Ultralytics recently rebranded. Confirm the 5.25 ms figure is for the detection model (not seg/pose variant) and that model size matches our backbone (~13 MB INT8).
6. **No Orin Nano 8GB (15W) YOLO11s number exists** — only the Super (25W) mode is benchmarked by Ultralytics. If thesis committee asks about 15W (default shipping mode), scale the 5.25 ms figure by ~1.7× → ~8.9 ms as a rough estimate, citing the TOPS ratio (67 → 40 TOPS at 15W).
