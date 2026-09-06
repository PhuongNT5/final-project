# Deployment Guide

Full procedure lives in `deployment/JETSON-DEPLOYMENT.md` — this file summarises the path and
states the pipeline-attribution caveat that must travel with every number below. Do not duplicate
the full procedure here; link to it.

## Pipeline attribution — read this first

**Every Jetson number in this repo today belongs to P1** (frozen-COCO detector + multi-stream
Transformer — see `system-architecture.md`), the pipeline currently deployed and benchmarked.
**No Jetson benchmark of P3 (the current accuracy headline) exists yet.** If you see a Jetson
latency/FPS/power number anywhere in this repo's docs, thesis chapters, or slides, it describes
P1, not the accuracy headline. Re-export and re-benchmark of P3 on Jetson is planned but not done
as of this writing — treat any P3-Jetson figure you encounter as an error until corrected.

## Hardware target

NVIDIA Jetson Orin Nano 8GB, JetPack 6.0 (L4T 36.x), CUDA 12.2, cuDNN 8.9, TensorRT 10.x. Full
setup instructions (image flashing, venv setup, engine build commands): `deployment/JETSON-DEPLOYMENT.md`.

The measured artifacts below were captured on real physical hardware, 2026-06-03: machine
`phuongnguyen-desktop`, `Linux-5.15.185-tegra-aarch64`, TensorRT 10.3.0 — confirmed via the
`platform` field embedded in each JSON artifact and independently cross-checked in
`plans/2026-09-06-thesis-2week-final/reports/qa1-adversarial-review-and-defence.md` item 18.

## Export -> TensorRT -> benchmark path

```
PyTorch checkpoint (sweep_results/pairwise/*.pt, runs/detect/.../yolo_finetuned/weights/best.pt)
  -> ONNX export (deployment/scripts/export-backbone.py, export-transformer-heads.py)
  -> INT8 quantization, ORT dynamic (deployment/scripts/quantize-int8.py) — for CPU/CoreML path only;
     Jetson INT8 uses TensorRT's own entropy calibrator instead, NOT the ORT-quantized files
  -> parity check (deployment/scripts/verify-onnx-parity.py)
  -> [Jetson] TensorRT engine build, FP16 or INT8
     (deployment/scripts/jetson-build-trt-engine.py --mode {fp16,int8})
  -> [Jetson] latency/FPS benchmark (deployment/scripts/jetson-benchmark.py)
  -> [Jetson] multi-stream + soak + power sweep
     (jetson-multi-stream-pipeline.py, jetson-power-sweep.sh)
```

Full reproduce commands (M4-side export steps 1-7): `deployment/README.md` "Reproduce" section.
Jetson-side build/benchmark commands: `deployment/JETSON-DEPLOYMENT.md` "Build engines" /
"Benchmark" sections.

## Measured results (P1, TRT-FP16, real hardware)

All figures below read directly from the JSON artifacts under `deployment/artifacts/` — not from
narrative claims in other docs, which is why some numbers here differ slightly from earlier
project drafts. **All belong to P1.**

| Metric | Value | Source |
|---|---|---|
| End-to-end latency p50 / p95 / p99 | 8.62 / 9.13 / 9.21 ms | `benchmarks-jetson-fp16.json` |
| Throughput, single stream | 116.0 FPS | `benchmarks-jetson-fp16.json` |
| Accuracy at benchmark time (P1, 4-class, same-dist 64) | 55/64 | `benchmarks-jetson-fp16.json` field `accuracy_4class_correct` — **this is P1's number, not the 54/64 P3 headline** |
| Multi-stream aggregate throughput, 4 streams | 88.06 FPS aggregate, 0 dropped frames across all 4 streams | `multi-stream-jetson-fp16-realtime.json` |
| Soak test duration | 36.2 min active / 42.5 min total | `soak-jetson-summary.json` |
| Soak throttle events | 0 | `soak-jetson-summary.json` |
| Soak mean power draw | 7033.8 mW (~7.03 W) of a 15 W (MAXN) budget | `soak-jetson-summary.json` |
| Energy per inference (best power-sweep mode) | 45.69 mJ/inference at 116.3 FPS, 5.31 W mean | `power-sweep-summary.json`, config `mode: 0` |

**Soak-test caveat:** 36.2 minutes is a pilot run, not a 24/7-availability claim. Do not
extrapolate it as such (per `plans/2026-09-06-thesis-2week-final/brief.md` §4i's explicit ban on
this exact extrapolation).

**A known internal contradiction to flag, not resolve here:** `deployment/JETSON-DEPLOYMENT.md`'s
header banner (top of file) states the procedure "was run end-to-end on a Jetson Orin Nano 8GB...
producing" the real artifacts above, but a later section of the same file ("Expected results
(projected from M4 + Jetson published specs)") still describes Jetson numbers as unmeasured
projections. The header banner and the artifact JSONs (verified above) are the current truth; the
later "Expected results" table is stale and should be corrected or removed by whoever owns
`deployment/JETSON-DEPLOYMENT.md` (Lane B, not this file's owner).

## INT8

INT8 engines on Jetson are built via TensorRT's own entropy calibrator (calibration batches at
`deployment/artifacts/calib-jetson/`), not via the ORT-quantized `*-int8.onnx` files used for the
CPU/CoreML (M4) benchmark path — those two INT8 paths are separate and should not be conflated.

## What is NOT yet available

- P3 (accuracy headline) Jetson export, TensorRT engine, or benchmark of any kind.
- Real (non-synthetic) INT8 calibration data for the backbone — `jetson-prep-int8-calib.py`
  currently uses random-noise calibration tensors per `deployment/JETSON-DEPLOYMENT.md`'s "Known
  gaps" section; only the FP16 engine numbers above should be treated as production-representative.

## Unresolved questions

1. Whether the P3 Jetson re-benchmark will land before thesis submission is a schedule risk noted
   in the active plan (`plans/2026-09-06-thesis-2week-final/plan.md`, phase-03) — not resolved as
   of this writing. Do not backfill a P3 Jetson number without a corresponding artifact JSON.
2. `deployment/JETSON-DEPLOYMENT.md`'s stale "Expected results (projected)" section (see above) has
   not been fixed as of this writing; this doc links to it as the authoritative source anyway
   because the fix is out of this file's ownership scope (Lane B owns `deployment/`).
