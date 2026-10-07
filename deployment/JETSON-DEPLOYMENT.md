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
> (2) Graph-compatibility was additionally pre-checked on M3 (`run_check_trt_compat.py`,
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

Copy from M3 to Jetson:

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

# INT8 (calibration batches must already exist; produced on M3 by
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
below is read from the committed artifacts; the M3 column is measured on this
development machine. Full derivation and the third-party-vs-measured taxonomy are
in `deployment/artifacts/projections.md`.

| Stage | M3 CPU FP32 (measured) | **Orin Nano FP16 (measured)** | **Orin Nano INT8 (measured)** |
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

   The gap is now *quantified* (`plans/2026-09-06-thesis-2week-final/reports/laneb-p3-edge-cost-and-export.md`).
   Per decision, fused, on arithmetic alone:

   | | GFLOPs/frame | per decision | GFLOPs/decision |
   | --- | ---: | ---: | ---: |
   | P1 backbone (yolo11s L0-22 @224) | 2.24 | 32 frames | 71.6 |
   | P3 A1 detector (yolo26s @800)    | 32.09 | 16 frames | 513.4 |
   | P3 A2 appearance (mnv3-small @224) | 0.1217 | 57.8 crops | 7.0 |

   **P3 is ~7.3x P1 per decision**, driven by detecting at 800² rather than by the
   appearance branch (A2 is 1.4 % of A1). Crop counts are measured over all 516 clips
   from the stored `mask` tensors (mean 57.8, p95 106, cap 128), not assumed.

   **No feasibility claim for P3 may be made from this.** Fixing the unit as ms per
   decision, measured P1 is `32 x 2.76 + 1.41 + 4.44 = 94.2 ms` (10.6 decisions/s);
   24 fps at `EMIT_EVERY=8` requires 3.0 decisions/s. Projecting P3 depends entirely on
   assumed throughput — the measured P1 backbone achieves only 0.81 TFLOP/s, i.e. it is
   overhead-bound, so scaling it linearly is pessimistic:

   | assumption | effective TFLOP/s | P3 ms/decision | decisions/s |
   | --- | ---: | ---: | ---: |
   | A — scale P1 backbone linearly | 0.81 | 643 | **1.6** |
   | B — 3 TFLOP/s | 3.0 | 176 | **5.7** |
   | C — 6 TFLOP/s (optimistic FP16) | 6.0 | 89 | **11.3** |

   The band **straddles** the 3.0 requirement, so arithmetic alone cannot decide whether
   P3 runs in real time here. **Superseded by the measurement below** — the calibrated
   answer falls under even row A, so treat B and C as refuted.

   **MEASURED 2026-09-09 (Apple M3, ORT-CPU, parity-gated).** All three P3 graphs are now
   exported and the head reproduces the headline predictor (max abs softmax diff 2.98e-07,
   argmax 64/64, ONNX top-1 54/64), so the timings below belong to the real predictor:

   | stage | p50 | share |
   | --- | ---: | ---: |
   | A1 detector | 109.06 ms/frame x16 = **1745.0** | **95.6 %** |
   | A2 appearance | 79.6 ms (58 crops, batched) | 4.4 % |
   | B relation head | 0.46 ms | 0.03 % |
   | **total** | **1825.0 ms = 0.55 decisions/s** | |

   Mac->Jetson transfer is *calibrated*, not assumed: P1 is measured on both machines
   (26.80 ms Mac ORT-CPU vs 12.47 / 8.62 / 8.24 ms Jetson TRT fp32/fp16/int8 = **2.15x /
   3.11x / 3.25x**). Applying those: **1.13 / 1.64 / 1.71 decisions/s — FAIL against 3.0
   at every precision**, i.e. P3 misses the cadence by ~2x. Still a projection, and the
   factor is a proxy (P1 is a 224-px backbone; a GPU should favour P3's 800-px conv, so
   the factor is if anything conservative) — but the direction is now measured. Cite as a
   projection, never as a Jetson result.

   > **SUPERSEDED 2026-09-29 by on-device measurement — the projection above was wrong, and
   > wrong in the direction its own caveat predicted. Do not cite 1.13/1.64/1.71 dec/s as a
   > result or as an expectation.** See the measured block below.

   **MEASURED 2026-09-29 (Jetson Orin Nano Super, JetPack R36.5.0, TensorRT 10.3.0 FP16,
   MAXN_SUPER, device idle, warmup 10 / runs 100).** Per-DECISION on the documented
   composition — 16 frames of detection + postprocess (select + Shapely zone geometry) +
   crop extraction + the clip's whole crop set through appearance in ONE batched pass + one
   head pass. Artifact:
   `deployment/artifacts/jetson-measurements/decision-latency-fp16-20260929T214556.json`
   (`dry_run: false`, `Linux-5.15.185-tegra-aarch64`).

   > **DO NOT QUOTE THE TABLE BELOW (standing ban, 2026-09-29, reaffirmed 2026-10-01).**
   > It is a DECODE-FREE control: the runs it comes from decoded no frames, and no deployment
   > decodes zero frames. Counting decode makes the detector **9.4% slower** in-process (CPU
   > decode contends), so every reportable figure — including the 3.62 dec/s total on the last
   > row — is unavailable. Use the second table. Kept only to document the decode cost.

   | stage | p50 (decode-free control) | share |
   | --- | ---: | ---: |
   | A1 detector | 11.982 ms/frame x16 = **191.71** | **69.5 %** |
   | postprocess (16 frames) | 46.835 | 17.0 % |
   | crop extraction (54 crops) | 17.96 | 6.5 % |
   | A2 appearance (54 crops, batched) | 19.31 | 7.0 % |
   | B relation head (binary E3) | 0.154 | 0.06 % |
   | **total** | **275.97 ms = 3.62 decisions/s** | |

   **Deployment-representative figures — RE-MEASURED 2026-10-01 on rebuilt engines, in BOTH
   clock conditions (n=3 each, decode measured in-process; artifacts
   `decision-latency-fp16-20261001T{222854,223216,223538}.json` pinned,
   `...T{220144,220543,220925}.json` DVFS). The 09-29 table that stood here is withdrawn: its
   engines did not match the device record and its clock state was never captured.**

   | stage | p50, clocks PINNED | share |
   | --- | ---: | ---: |
   | A1 detector | 11.724 ms/frame x16 = **187.58** | **60.5 %** |
   | postprocess (16 frames) | 47.35 | 15.3 % |
   | frame decode, 8 sequential 1080p frames (CPU/FFMPEG, no NVDEC) | 37.61 | 12.1 % |
   | A2 appearance (54 crops, batched) | 19.34 | 6.2 % |
   | crop extraction (54 crops) | 17.95 | 5.8 % |
   | B relation head (binary E3) | 0.158 | **0.05 %** |
   | **stage sum** | **310.00 ms = 3.226 decisions/s** | |

   **⚠ The "total" row is a STAGE SUM, not a cadence, and NOT a bound in either direction
   (2026-10-03).** Each stage above was timed **alone**, back-to-back, on synthetic fixed tensors;
   `jetson-measure-latency.py:211` adds the p50s. Two independent errors run in opposite
   directions. (a) **Six costs are omitted**, three of them entire stages — the largest is
   **detector input preprocessing** (16 frames to 800x800 resize + normalise + HWC->CHW), which
   appears **nowhere** in the measurement file; omissions make the sum too optimistic. (b) The sum
   **double-charges the detector**: it multiplies the detector p50 by `T_FRAMES = 16` while the
   loop emits a decision every `DECISION_EMIT_EVERY = 8` frames (`jetson-measure-latency.py:105-106`
   defines the stride on the line after T_FRAMES and never uses it), so the sum charges 16 detector
   passes per decision where a sliding window amortises 8; that makes it too pessimistic. Which
   error wins **depends on the clock state**:

   | arm | published stage sum | stride-coherent recomposition | COMPOSED measured | published vs composed |
   | --- | ---: | ---: | ---: | ---: |
   | pinned | 310.37 ms = 3.222 dec/s | 174.24 ms = 5.739 | **287.44 ms = 3.479** | **7.98 % too HIGH** |
   | DVFS | 421.52 ms = 2.372 dec/s | 230.82 ms = 4.332 | **694.00 ms = 1.441** | **39.3 % too LOW** |

   The bias **changes sign with the clock state**, so no correction factor can rescue the
   construct — it had to be replaced by a composed measurement, not adjusted. See
   `plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-10-03-p3-decision-latency-is-a-stage-sum.md`.

   **Clock state decides the cadence.** Same engines, same crop loads, same
   evening; the only difference is `sudo jetson_clocks`. Margin-to-contract columns are
   **withdrawn** — a stage sum cannot adjudicate a cadence contract:

   | crop load | default DVFS (`schedutil`) | clocks PINNED |
   | --- | ---: | ---: |
   | p50 = 54 | 365.88 ms = 2.733 dec/s | 310.00 ms = 3.226 dec/s |
   | p95 = 106 | 403.11 ms = 2.481 dec/s | 346.87 ms = 2.883 dec/s |
   | p99 = 119 | 413.03 ms = 2.421 dec/s | 355.93 ms = 2.810 dec/s |
   | cap = 128 | 420.25 ms = 2.380 dec/s | 362.32 ms = 2.760 dec/s |

   **What to claim (MEASURED 2026-10-03, supersedes both earlier claims): P3's cadence is
   clip-dependent and holds only with the clocks pinned.** Two earlier claims are both
   **withdrawn**: "clears the contract with +7.5 % margin when pinned" (it rested on a stage sum)
   and its first correction, "P3 misses the contract in BOTH clock conditions" (it rested on
   arithmetic over that same sum — and the measurement shows it is **false in the pinned arm**).
   `deployment/scripts/jetson-p3-multi-stream.py` now runs the whole loop under one wall clock —
   file decode → 800x800 letterbox preprocess → detector → select + zone geometry → 224² context
   crops → appearance encoder → 74-d relation-token assembly → binary E3 head — closing all six
   omissions. Free-running (`--fps 1000`, N=1, 480 frames, 3 test clips, FP16, MAXN_SUPER, both
   arms thermally gated at entry tj <= 56 °C / load <= 0.60 after a 180 s settle):

   | clip | PINNED dec/s | PINNED ms/frame | DVFS dec/s | DVFS ms/frame |
   | --- | ---: | ---: | ---: | ---: |
   | 1_te10 (view 3) | **3.479 MEETS** | 35.32 | 1.441 FAILS | 85.32 |
   | 3_te1 (view 7) | **3.965 MEETS** | 31.01 | 1.511 FAILS | 81.37 |
   | 4_te10 (view 1) | 2.493 FAILS (by 1.20x) | 49.31 | 1.141 FAILS | 107.76 |
   | median | **3.479** | | **1.441** | |

   Pinned **MEETS on 2 of 3 clips**; DVFS **FAILS on 3 of 3** (by 1.99x–2.63x). The two arms are
   **disjoint**: max DVFS 1.511 < min pinned 2.493. Pinning is therefore worth **+141.4 % (2.41x)**
   on the composed cadence but only **+26.4 % (1.26x)** on the stage sum — because the omitted work
   is exactly the part the governor starves. Machine-recorded mechanism: under `schedutil`,
   **`policy4` (cores 4-5) sat at the 115200 kHz floor in every DVFS sample**, before and after
   every run, while `policy0` wandered 192000–1728000; pinned, both clusters read
   `cur = min = 1728000` and the GPU `cur = min = max = 1020 MHz`. The composed loop is CPU-bound
   on preprocessing, which is what the stage sum omits. Never report a P3 cadence figure without
   naming **both** the clock condition and the clip. Artifacts:
   `deployment/artifacts/jetson-measurements/p3-composed-cadence-fp16-{pinned,dvfs}-free-*.json`.

   **Three reading rules for the composed artifacts.** (1) **`--fps F` is an additive pacer, not a
   deadline** — `stream-pipeline-utils.py:84` sleeps `period` *after* all per-frame work, so the
   achievable rate is `1/(1/F + work) < F`. The `--fps 24` rows are **not** a contract verdict;
   only the `--fps 1000` rows are. (2) **`drop_rate = 0` is not evidence of real-time operation at
   N <= 2** — the reader is a coroutine on the consumer's event loop with `preprocess_fn` inline,
   so it is back-pressured into lockstep: `read_fps == aggregate_fps` to the cent in all five
   N <= 2 configs, including the run that requested 1000 fps and delivered 11.72. Cite `read_fps`
   vs `target_fps`, never `drop_rate`. (3) **`clock_state_stable` means the pin *mode* and
   contention did not change, NOT that frequencies held** — it compares `clocks_pinned` and `quiet`
   before/after (`stream-pipeline-utils.py:140`), so it reads `true` for every DVFS run while
   `policy0 cur_khz` moved 1574400 → 192000 inside one run. The code matches its docstring; the
   field **name** is the hazard.

   **Operator action:** run `sudo jetson_clocks` after boot, before starting the pipeline.
   It is not persistent across reboots.

   The residual p95 analysis below is a **stage-sum** statement and is retained only as a
   cost-structure argument — there is no composed p95, because the composed harness reports a
   per-run cadence rather than a per-decision distribution. Postprocess is clock-insensitive
   (−0.7 % under a change that moves the detector −17.3 %) and is the binding term *within the
   sum*: at pinned p95 the gap to contract is 13.5 ms against a 47.4 ms postprocess cost, so a
   ~29 % postprocess reduction would clear the sum's p95 without touching the models. Do not read
   this as a statement about the contract.

   Why the projection missed: the Mac->Jetson factor was calibrated on P1, a 224-px
   backbone, and applied to an 800-px conv detector. The real detector factor is
   109.06 / 11.724 = **9.3x**, not the 3.11x proxy — a ~3x understatement of the GPU
   benefit, exactly the conservatism the paragraph above flagged. The qualitative
   conclusion that survives unchanged is the cost *structure*: the detector dominates a
   decision (60.5 % with decode counted, ~70 % without) and the relation head is free
   (0.05 %) on both latency and energy: 0.00245 J of 4.02 J per decision on the total-board
   basis, or 0.00118 J of 2.90 J marginal over the settled 4.691 W idle baseline — **0.041 %**.

   **Three limits on this number — two of them now closed.** (1) ~~Frame decode is excluded~~
   — **decode was measured the same evening**; the stage-sum figures above already include it.
   ~~Frame *preprocessing* is CPU-side and unmeasured~~ — **it was measured inside the composed
   loop on 2026-10-03** (`cv2.resize` 1920x1080->800 letterbox, BGR->RGB, /255, HWC->CHW, per
   frame, plus 54 crops at 224²) and it is the single largest omitted cost: **18.89 ms/frame
   pinned (2.11x the stage sum's per-frame figure) and 62.91 ms/frame under DVFS (3.64x)**. What
   remains excluded is NVDEC (this OpenCV build reports `GStreamer: NO`, so decode here is CPU and
   is an upper bound). (2) ~~Single stream, synthetic detector/appearance inputs~~ — the composed
   runs decode real test clips and feed real tensors through every stage; the multi-stream arm
   covers N = 1, 2, 3, with the aggregate ceiling roughly **flat in N** (~28–30 fps pinned,
   ~13–15 fps DVFS — concurrency redistributes rather than adds, and **no** N >= 2 config reaches
   3.0 dec/s per stream). The stage-sum rows above remain single-stream and synthetic.
   (3) The Mac baseline it is compared against is **in dispute** — the same graphs
   under the same EP gave 1825 ms/decision on 2026-09-09 and 13727.54 ms on 2026-09-16, the
   latter with a 91 % p50->p95 detector spread that indicates a contended machine. **Do not
   quote 13,727 ms / 0.07 dec/s.** Full adjudication:
   `plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-09-29-jetson-decision-latency.md`.

   **Cost curve — what would clear the bar** (`--cost_imgsz` probes, fp16 factor;
   *accuracy unmeasured for every row but the first*):
   800x16 **1.64** (54/64) | 800x8 3.14 | 640x16 2.51 | **512x16 3.77** | 416x16 4.92.
   T is part of the head's input shape `(1,16,8,74)` so changing it forces a re-architecture;
   imgsz changes only the token *values*. **imgsz 800->512 at T=16 is therefore the cheapest
   credible route** — rebuild tokens, retrain the head, re-measure accuracy.

   Export status: all three graphs exported by `deployment/scripts/export-p3-graphs.py` —
   `p3-detector-fp32.onnx` (38 MB, imgsz 800, NMS-free `(1,300,6)`),
   `p3-appearance-fp32.onnx` (4.2 MB), `p3-relation-head-fp32.onnx` (1.2 MB). The
   `TRTRunner` multi-input blocker is **fixed** (2026-09-29: buffers are keyed by tensor name
   and bound via kwargs, so `tokens`+`mask` bind correctly); `benchmark()`'s accuracy path is
   still P1-shaped (`stats_pool_np` + cascade), and the multi-stream harness is P1-only.

   **Cadence caveat, applies to P1 too.** The 116 FPS above is
   `t_backbone(1 frame) + t_heads(1 clip)` — a per-frame and a per-clip term summed, i.e.
   the *stride-1* figure (a decision every frame). `jetson-multi-stream-pipeline.py` runs
   `T_WINDOW=32, EMIT_EVERY=8`. Neither is wrong and 116 is the conservative one, but any
   P3-vs-P1 comparison must fix the cadence first or it compares units.

2. **Single device, single session.** All Jetson numbers come from one machine on
   one date (2026-06-03) at one power mode. Run-to-run and unit-to-unit variance is
   unmeasured; report as a single measured operating point, not as a device spec.
3. **No Orin AGX measurement.** The former projection table quoted AGX figures with
   no hardware behind them. AGX is now simply out of scope - do not carry those
   numbers into the thesis.
