# Deployment Guide

> ## ⚠️ SUPERSEDED — NOT THE SUBMITTED RESULT (as of 2026-10-05)
>
> **The submitted thesis is the sole authority for every number.** It lives in `report/vn/`
> (11 files, Vietnamese). This file is a **development log** kept for provenance; it was written
> while the project's headline was still the YOLO lane, and it has **not** been rewritten to match
> the submission.
>
> What changed, in one paragraph: the thesis's headline configuration is **not** a YOLO pipeline.
> It is **B1** — a frozen CLIP ViT-B/32 tower feeding a 201,090-parameter temporal head — scoring
> **84.2 % balanced accuracy** on iSafetyBench (1,100 clips, binary) under **leave-source-out**
> folds grouped by source video, reported against a pre-registered **63.3 %** count-rule floor.
> The entity-decomposition hypothesis that motivated the YOLO lane was tested as a pre-registered
> gate (G14) and **failed** (−0.5 pp, cluster CI [−5.5, +5.0] containing 0, McNemar p = 0.683).
> Eight pre-registered gates were adjudicated: **two passed, six failed.**
>
> **Numbers in this file that the thesis does NOT stand behind** — do not cite them, from here or
> from anywhere else: any "+3.5 pp macro win"; "57/64"; "95.7 % mAP"; the P3 cadence figures
> "331 ms / 3.02 dec/s / 3.91 J" and any claim that P3 "passes" or "fails" the 3.0 dec/s contract
> as a single verdict (the composed measurement of 2026-10-03 is clip- **and** clock-dependent);
> and any per-stage **sum** presented as a system cadence. The thesis's own Chapter 7 documents why
> each was withdrawn.
>
> Cross-lane comparison is also forbidden: the 1,100-clip binary iSafetyBench numbers and the
> 64-clip 4-class Unsafe-Net numbers are different corpora, different tasks and different metrics.
> They are never averaged and never set against each other (thesis §6.1.1).


Full procedure lives in `deployment/JETSON-DEPLOYMENT.md` — this file summarises the path and
states the pipeline-attribution caveat that must travel with every number below. Do not duplicate
the full procedure here; link to it.

## Pipeline attribution — read this first

**This section is out of date and is corrected here rather than deleted.** It used to say that
every Jetson number in the repo belonged to P1 and that no Jetson benchmark of P3 existed. Both
statements are now false, and the pipeline attribution that must travel with each number is:

- **B1/R48** (frozen CLIP tower + temporal head) — the **submitted headline and the deployed
  graph**. Composed cadence measured on the Orin Nano Super: **167.291 ms/decision = 5.978
  decisions/s** (FP16, default DVFS; thesis §5.4.4). This is the only lane whose decision loop was
  measured end to end with nothing omitted.
- **P3** (entity detector + relation Transformer) — measured on the device **composed** on
  2026-10-03. The result is **clip-dependent and clock-dependent** and must never be quoted as a
  single pass/fail verdict against the 3.0 decisions/s contract (thesis §5.5). Earlier P3 headlines
  built as a **sum of separately-timed stages** are withdrawn: that construct runs 7.98 % too fast
  when the clocks are pinned and 39.3 % too slow under DVFS.
- **P1** — the provenance of its Jetson numbers is **UNDETERMINED**, neither clean nor excluded
  (thesis §5.6.1). The engine was never committed, so the figures cannot be verified in place. No
  thesis table vouches for a P1 number.

DVFS is additionally **not a repeatable measurement condition on this board**: 11 runs across two
days gave 309.08–311.78 ms pinned (0.49 % spread) against 365.88–440.01 ms under DVFS (20.3 %
spread), with the two sets disjoint. A single DVFS run adjudicates nothing (thesis §5.4.6).

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

Full reproduce commands (M3-side export steps 1-7): `deployment/README.md` "Reproduce" section.
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
(projected from M3 + Jetson published specs)") still describes Jetson numbers as unmeasured
projections. The header banner and the artifact JSONs (verified above) are the current truth; the
later "Expected results" table is stale and should be corrected or removed by whoever owns
`deployment/JETSON-DEPLOYMENT.md` (Lane B, not this file's owner).

## INT8

INT8 engines on Jetson are built via TensorRT's own entropy calibrator (calibration batches at
`deployment/artifacts/calib-jetson/`), not via the ORT-quantized `*-int8.onnx` files used for the
CPU/CoreML (M3) benchmark path — those two INT8 paths are separate and should not be conflated.

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
