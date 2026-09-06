# Jetson Orin Deployment Guide (JetPack 6)

> **Status (2026-09-06):** **executed on physical hardware.** The procedure below was
> run end-to-end on a Jetson Orin Nano 8GB (`phuongnguyen-desktop`,
> Linux-5.15.185-tegra-aarch64, TensorRT 10.3.0) on 2026-06-03, producing
> `deployment/artifacts/benchmarks-jetson-{fp16,int8}.json`,
> `soak-jetson-summary.json`, `multi-stream-jetson-fp16-realtime.json` and
> `power-sweep-summary.json`. Toolchain risk is retired.
>
> *(Supersedes the 2026-05 status "scaffolds only — no physical Jetson available in the
> thesis budget", which was stale.)*
>
> **Two caveats before citing anything here.** (1) Those artifacts benchmark the **P1**
> pipeline (frozen-COCO detector + multi-stream Transformer), **not** the P3 accuracy
> headline — re-export and re-benchmark of P3 is Lane B work and is still pending, and
> `deployment/scripts/jetson-benchmark.py` still hardcodes the old P1 `MODELS` list.
> (2) Graph-compatibility was additionally pre-checked on M4 (`run_check_trt_compat.py`,
> `deployment/artifacts/trt-compat-report.md`): all FP32 ONNXs in
> `deployment/export/*-fp32.onnx` parse cleanly into TRT 10 networks; the ORT-quantized
> `*-int8.onnx` files are *not* used on Jetson — INT8 is done by TRT's own entropy
> calibrator.

## Target hardware

- **NVIDIA Jetson Orin Nano 8 GB** (cost-optimised) or **Orin AGX 32 GB**
  (headroom for multi-stream).
- JetPack 6.0 (L4T 36.x), CUDA 12.2, cuDNN 8.9, TensorRT 10.x.
- 64 GB NVMe (engines + features cache + JetPack image).

## JetPack 6 image

Download the latest JetPack 6 SD card image from the NVIDIA Developer site
(login required; URL changes each release). After flashing with Etcher /
sdkmanager, on first boot:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-pip python3-venv libopenblas-dev libjpeg-dev zlib1g-dev
pip install --upgrade pip
```

JetPack 6 ships TensorRT 10 inside `/usr/lib/python3.10/dist-packages/`;
do **not** `pip install tensorrt` — that pulls the x86 wheel.

## Project install on Jetson

```bash
git clone <thesis-repo> ~/unsafe-net
cd ~/unsafe-net
python3 -m venv .venv-jetson --system-site-packages   # inherit JetPack TRT
source .venv-jetson/bin/activate
pip install onnx onnxruntime numpy psutil pycuda
# torch: use the JetPack-matched wheel from
#   https://forums.developer.nvidia.com/t/pytorch-for-jetson/  (pinned page)
```

Copy from M4 to Jetson:

| Local path                                          | Purpose                          |
| --------------------------------------------------- | -------------------------------- |
| `deployment/export/*-fp32.onnx*`                    | TRT input graphs                 |
| `deployment/artifacts/calib-jetson/`                | INT8 calibration batches         |
| `deployment/artifacts/features/all8_test_onnx.pt`   | Test set for accuracy parity     |
| `deployment/scripts/jetson-*.py`                    | Build + benchmark scaffolds      |

## Build engines

```bash
# FP16 (default; ~2× faster than FP32 on Orin)
python3 deployment/scripts/jetson-build-trt-engine.py --mode fp16

# INT8 (calibration batches must already exist; produced on M4 by
#       jetson-prep-int8-calib.py before transfer)
python3 deployment/scripts/jetson-build-trt-engine.py --mode int8 \
    --calib-dir deployment/artifacts/calib-jetson
```

Engines land in `deployment/export/jetson/*.engine` and are device-specific
(do **not** copy across Orin Nano ↔ Orin AGX or across L4T versions).

## Benchmark

```bash
python3 deployment/scripts/jetson-benchmark.py --mode fp16 --runs 100
python3 deployment/scripts/jetson-benchmark.py --mode int8 --runs 100
```

Outputs: `deployment/artifacts/benchmarks-jetson-{fp16,int8}.json`.

## Measured results (Jetson Orin Nano 8GB, 2026-06-03)

Superseded the former "Expected results (projected...)" table, which was written
before hardware access and is no longer the best available evidence. Every number
below is read from the committed artifacts; the M4 column is measured on this
development machine. Full derivation and the third-party-vs-measured taxonomy are
in `deployment/artifacts/projections.md`.

| Stage | M4 CPU FP32 (measured) | **Orin Nano FP16 (measured)** | **Orin Nano INT8 (measured)** |
| ----- | ---------------------: | ----------------------------: | ----------------------------: |
| Backbone p50   | ~22 ms   | **2.76 ms** (p95 3.24 / p99 3.25) | **2.39 ms** (p95 2.68 / p99 2.76) |
| Cascade p50    | -        | **4.44 ms** (p95 4.50)            | **4.43 ms** (p95 4.49)            |
| End-to-end p50 | 28.8 ms  | **8.62 ms** (p95 9.13 / p99 9.21) | **8.24 ms** (p95 8.58 / p99 8.71) |
| FPS (single stream) | 34.7 | **116.0**                        | **121.4**                         |

Source: `benchmarks-jetson-fp16.json`, `benchmarks-jetson-int8.json` (warmup 10, runs 100).
INT8 buys ~4 % end-to-end - the pipeline is not backbone-bound, so quantising the
backbone alone cannot move the total much. Say that, rather than selling INT8 hard.

Multi-stream (`multi-stream-jetson-fp16-realtime.json`, 24 fps real-time feeds):

| Streams | Aggregate FPS | Frames | Dropped | Worst per-stream p95 |
| ------: | ------------: | -----: | ------: | -------------------: |
| 1 | 24.28 | 240 | **0** | 7.11 ms |
| 2 | 48.16 | 480 | **0** | 8.53 ms |
| 3 | 70.57 | 720 | **0** | 8.59 ms |
| 4 | 88.06 | 960 | **0** | 7.87 ms |

Zero-drop holds to 4 concurrent real-time streams; head p95 stays under 9 ms and
does not degrade monotonically with load. **Disclosure required wherever this is
cited:** each config runs only ~10 s (240 frames/stream), so this measures steady-state
throughput, not thermal or memory behaviour over a shift - the 30-minute soak
(`soak-jetson-summary.json`) is the evidence for that, and neither may be extrapolated
to longer horizons (brief.md 4i).

## Known gaps

The three gaps previously listed here ("no real device measurement", "synthetic
INT8 calibration", "multi-stream not ported") are all **closed** - see the tables
above, and note that `artifacts/calib-jetson/*.npy` are real ImageNet-normalised
frames (value range [-2.118, 2.640] is the exact signature of normalised [0,1]
pixels, not of sampled noise). The remaining gaps are:

1. **These artifacts benchmark P1, not the P3 accuracy headline.** The 55/64 accuracy
   recorded inside the benchmark JSONs is the deployed P1 cascade, and
   `deployment/scripts/jetson-benchmark.py` still hardcodes the P1 `MODELS` list.
   P3 export + re-benchmark is Lane B and is **not done**. No Ch5 latency number may
   be presented next to a P3 accuracy number without saying so.
2. **Single device, single session.** All Jetson numbers come from one machine on
   one date (2026-06-03) at one power mode. Run-to-run and unit-to-unit variance is
   unmeasured; report as a single measured operating point, not as a device spec.
3. **No Orin AGX measurement.** The former projection table quoted AGX figures with
   no hardware behind them. AGX is now simply out of scope - do not carry those
   numbers into the thesis.
