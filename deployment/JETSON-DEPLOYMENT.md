# Jetson Orin Deployment Guide (JetPack 6)

> **Status (2026-05):** scaffolds only — no physical Jetson available in the
> thesis budget. The procedure below has been validated on M4 only as a
> graph-compatibility sanity-check (`run_check_trt_compat.py`,
> `deployment/artifacts/trt-compat-report.md`). All FP32 ONNXs in
> `deployment/export/*-fp32.onnx` parse cleanly into TRT 10 networks; the
> ORT-quantized `*-int8.onnx` files are *not* used on Jetson — INT8 is
> done by TRT's own entropy calibrator.

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

## Expected results (projected from M4 + Jetson published specs)

These are the targets the thesis Ch5 §5.5 projection table expects to be
verified once hardware is available; see `deployment/artifacts/projections.md`
for derivation.

| Stage | M4 CPU FP32 | Jetson Orin Nano FP16 (proj.) | Jetson Orin AGX FP16 (proj.) |
| ----- | ----------: | ----------------------------: | ---------------------------: |
| Backbone p50 | ~22 ms      | ~6 ms                         | ~2.5 ms                      |
| End-to-end p50 | 28.8 ms   | ~10 ms                        | ~4 ms                        |
| FPS (single stream) | 34.7 | ~100                          | ~250                         |

## Known gaps

1. **No real device measurement.** All Jetson numbers in Ch5 §5.5 are
   projections; replace with measured numbers once the device arrives.
2. **Calibration source for backbone is synthetic.** `jetson-prep-int8-calib.py`
   uses random-noise calibration tensors for the YOLOv11s backbone.
   For tighter INT8 accuracy on Jetson, replace with 32 real frames sampled
   from `data/train/`.
3. **Multi-stream pipeline not ported.** `multi-stream-pipeline.py` (M4
   CoreML version) needs a Jetson-side rewrite using `cv2.VideoCapture` on
   the GStreamer pipeline; deferred to future work (Ch6 §6.4 P3).
