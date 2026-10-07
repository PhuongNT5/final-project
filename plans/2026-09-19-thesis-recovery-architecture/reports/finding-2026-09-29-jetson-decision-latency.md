# Finding 2026-09-29 — P3 per-decision latency on Jetson: the cadence claim flips

> **⚠ WITHDRAWN IN PART — 2026-10-03, REPLACED BY MEASUREMENT THE SAME DAY.** Every P3
> "per-decision" figure below (310.00 ms / 3.226 dec/s pinned, 365.88 / 2.733 DVFS, and the
> p95/p99/cap rows) is a **sum of separately-timed stages**, not a composed loop:
> `jetson-measure-latency.py:211` adds five stage p50s, each measured with only that stage
> running, on synthetic fixed tensors. **The +7.5 % contract margin is withdrawn.**
> **An intermediate correction is ALSO withdrawn:** the morning's arithmetic ("six costs are
> omitted, every omission can only add latency, so 3.226 dec/s is an upper bound; corrected pinned
> >= 360.87 ms = <= 2.771 dec/s, therefore P3 misses the contract in both clock conditions") is
> **false in the pinned arm**, and the sum is not a one-sided bound at all — it also
> **double-charges the detector** (16 passes per decision where the loop's stride of 8 amortises
> 8), so its bias **changes sign with the clock state**: the published sum is 7.98 % too HIGH when
> pinned and 39.3 % too LOW under DVFS.
> **The measured composed cadence (2026-10-03 17:33–17:47, both arms thermally gated, N=1,
> 480 frames, 3 test clips, FP16):** pinned **2.493–3.965 dec/s (median 3.479), MEETS on 2 of 3
> clips**; DVFS **1.141–1.511 (median 1.441), MEETS on none**. The arms are **disjoint**
> (max DVFS < min pinned), so pinning is worth **+141.4 % (2.41x)** composed against only
> **+26.4 % (1.26x)** on the stage sum. **P3's cadence is clip-dependent and holds only with the
> clocks pinned.** Quote `p3-composed-cadence-fp16-*` artifacts, never a stage sum, for any
> contract statement. Adjudication:
> `reports/finding-2026-10-03-p3-decision-latency-is-a-stage-sum.md` (§11).


Artifact: `deployment/artifacts/jetson-measurements/decision-latency-fp16-20260929T214556.json`
Provenance: `dry_run: false`, `Linux-5.15.185-tegra-aarch64-with-glibc2.35`, `nvpmodel_q: MAXN_SUPER`,
python 3.10.12, TensorRT 10.3.0, warmup 10 / runs 100, device load avg 0.44 at start.

> **SUPERSEDED IN FULL, 2026-10-01 — no number in this report is quotable.** The engines it
> measured did not match the device record, and the run's clock state was never captured. The whole
> session was re-run on rebuilt engines in both clock conditions:
> `finding-2026-10-01-engine-provenance-invalidates-0929-latency.md` is the authority. Headline
> now **310.00 ms = 3.226 dec/s (+7.5 %) pinned** / **365.88 ms = 2.733 dec/s (−8.9 %, FAIL)**
> under default DVFS, p95 and above FAIL in both. Kept as the record of what that session measured
> and of the decode-cost and crop-percentile method, which both carried over.
>
> **Earlier banner, 2026-09-29 22:35 — §1's table is a decode-free control, not the result. Quote §5c.**
> Decode was measured (unresolved item 2) and it removes the PASS at every crop load except p50.
> With n=5 the picture is settled: the decode-free condition below and the deployment-representative
> condition differ by a systematic +9.4% on the detector, each internally reproducible to <2.2%.
> Since no deployment decodes zero frames, **§1's PASS-everywhere row must not be quoted at all.**
> The honest headline is **"on the 3.0 dec/s boundary"** — 3.02-3.04 dec/s at median crop load,
> 2.60-2.72 above it.

## 1. Result as first measured — DECODE-FREE CONTROL, do not quote (see §5c for the result)

In a process that decodes no frames, P3 **appears to meet** the 3.0 decisions/s cadence contract
(24 fps at EMIT_EVERY=8) at every crop-count percentile under TensorRT FP16. §5c shows this
condition is unreachable in deployment; the table is kept for the cost-structure argument only.

| crops | crop-extract p50 | appearance p50 (batched) | ms/decision p50 | dec/s | cadence |
|---|---:|---:|---:|---:|:--:|
| p50 = 54 | 17.96 | 19.31 | **275.97** | **3.62** | PASS |
| p95 = 106 | 37.33 | 36.78 | 312.81 | 3.20 | PASS |
| p99 = 119 | 41.99 | 41.08 | 321.77 | 3.11 | PASS |
| cap = 128 | 45.31 | 44.14 | 328.15 | **3.05** | PASS |

Shared terms: detector 11.982 ms/frame x 16 = 191.71 ms; postprocess (select + Shapely zone
geometry, 16 frames) 46.835 ms; relation head (binary E3) 0.154 ms.
Percentiles are tight (detector p50 11.982 / p99 12.033, 0.4% spread) — the device was idle.

Cost structure at p50: detector **69.5%**, postprocess 17.0%, crop-extract 6.5%, appearance 7.0%,
head 0.06%. The detector remains the entire cost; the relation head is still free. That part of
the prior analysis survives intact — on the deployment-representative runs the shares become
detector **62.1%**, postprocess 14.3%, decode 11.7%, appearance 6.1%, crop-extract 5.5%, head
**0.05%** (§5c, §5e).

## 2. Why this differs from the projection

The thesis currently projects 1.13-1.71 dec/s on Jetson (FAIL) from a Mac ORT-CPU baseline scaled
by P1 transfer factors. The runbook's own instruction for this step applies: the measurement
REPLACES the projection, including when it comes out faster. **D2's framing gets revised, not the
number.** P3 not being real-time was never measured on the target device; it was extrapolated.

## 3. Two engine-build defects found and fixed (both would have corrupted this number)

1. **`jetson-build-trt-engine.py` pinned every dynamic dim to batch 1** ("We deploy at batch=1
   only"). Correct for P1 and for the P3 detector/head; **wrong for `p3-appearance`**, whose ONNX
   batch dim (`boxes`) is the clip's whole crop set. A batch-1 appearance engine forces a
   54-iteration loop where the Mac ONNX baseline does one batched pass — 54 x 1.393 = 75.2 ms
   instead of the measured 19.31 ms, i.e. it would have inflated the decision cost by ~56 ms and
   broken like-for-like comparison. Added `--batch_profile MIN:OPT:MAX`; appearance rebuilt at
   1:54:128. Batch-1 engine preserved as `p3-appearance-fp16-batch1.engine`.
2. **`throughput_decisions_per_sec_naive = 73.051`** in `latency-fp16-20260929T212712.json` is NOT
   a decision rate — it sums one pass per stage, i.e. 1 detector frame (not 16) and 1 crop (not
   54). It overstates the rate by ~20x. Must never be reported. `run_decision_latency()` was added
   to `jetson-measure-latency.py` (`--decision`) to compute the real composition.

## 4. Mac baseline conflict — diagnosed, resolution BLOCKED

Two Mac ORT-CPU measurements of the same graphs disagree by 7.4x:

| | 2026-09-09 (report, "idle M3") | 2026-09-16T20:27 (canonical JSON) |
|---|---:|---:|
| detector | 109.06 ms/frame | 811.05 (p95 1552.52, p99 1588.03) |
| head | 0.46 ms | 1.35 ms |
| total | 1825.0 ms = 0.55 dec/s | 13727.54 ms = 0.07 dec/s |

The 09-16 run is very likely **contended, not slower hardware**: (a) the head is a tiny graph and
cannot legitimately get 2.9x slower on the same machine with the same EP; (b) its detector
p50->p95 spread is 91%, versus 0.4% on the idle Jetson — the signature of CPU contention;
(c) `benchmark-p3-onnx.py`'s own docstring warns "Run this on an IDLE machine".

Confirmed mechanism, not proof: at the time of writing this the Mac carries **load average 51.95**
with `build-isafetybench-videomae-features.py` and 6x `run-esta-lso.py` running. A clean re-run is
the only way to settle it and is **blocked until those finish** — they are live thesis experiments
and must not be killed.

Consequence: **do not quote 13,727 ms / 0.07 dec/s anywhere.** Until the re-run, the Mac figure is
in dispute. The Jetson number above does not depend on it.

## 5. What is NOT in this number (must be disclosed)

- ~~**Frame decode is excluded**~~ — **no longer true: measured 22:18, see §5c.** It was the single
  biggest threat to the PASS verdict and it did in fact remove the PASS everywhere except the
  median crop load. Kept here so the exclusion list reads in the order it was discovered.
- Synthetic tensors, not real footage; detector input is random, so postprocess box counts come
  from a random draw rather than real detections.
- **Frame preprocessing is also excluded**, and this was not previously noticed. The detector is
  timed with `_fixed_inputs`, i.e. an already-shaped `(1, 3, 800, 800)` float tensor. The real
  per-frame conversion — `cv2.resize` from 1920x1080 to 800, BGR->RGB, `/255`, HWC->CHW, x16
  frames — is inside neither `detector_ms_per_frame` nor `postprocess_ms_per_decision` (which runs
  *after* detection). Like decode it is CPU-side and overlappable, but like decode it is not zero,
  and it has not been measured. Same exclusion applies to the Mac ONNX baseline, so the two remain
  comparable to each other; both understate an end-to-end figure.
- Single stream. Multi-stream (runbook §6) not yet measured for P3.
- **Power mode: MAXN_SUPER (mode 2), not the 15 W mode 0 that `risks.md` R32 specifies.** Correctly
  recorded in each artifact's `device_state`, but it means the PASS is a max-performance result and
  says nothing about a 15 W envelope (risks.md R32b).
- Crop-count percentiles came from `FALLBACK_CROP_PCTS` (token files absent on device). Values are
  identical to the `source: "measured"` values in `benchmarks-p3-onnx-cpu.json` (54/106/119/128),
  so the numbers are unaffected — only the provenance label differs.
- **Those percentiles are computed on the wrong corpus, and the effect is immaterial.**
  `crop_count_percentiles()` reads `tokens_yolo26app_{train,test}.pt` = **516 unsafe-only** clips,
  not the 691-clip binary corpus. Recomputed on `tokens_bin8_*`: p50 **53** (not 54), p95 **104**
  (not 106), p99 119, cap 128 — so the p50 row moves ~0.7 ms (3.02 -> 3.03 dec/s) and the p99/cap
  rows not at all. Disclosed, not re-run. Detail:
  `finding-2026-09-29-zone-geometry-leakage-check.md` §4.
- `git_commit: unknown` on device (the rsync'd tree is not a git checkout).

## 5b. FP16 vs FP32 parity (2026-09-29, `parity-20260929T221055.json`)

FP32 engines were built for all four P3 stages (appearance with the same `1:54:128` batch profile
as its FP16 counterpart, so the comparison is like-for-like) and each precision was run on the same
fixed input.

| stage | max abs diff vs FP32 | argmax agreement |
|---|---|---|
| `p3-appearance` | 2.53e-02 | **1.000** |
| `p3-relation-head` (4-class) | 1.66e-03 | **1.000** |
| `p3-relation-head-bin-e3` | 0.0 | **1.000** |
| `p3-detector` | 794.4 | 0.633 — **not a valid measurement, see below** |

The three classifier stages agree perfectly: FP16 quantization does not move a single decision on
the path that produces the safety verdict, and the binary E3 head — the frozen headline encoder —
is bit-identical. That is the parity claim the thesis needs, and it is now measured.

**The detector row must not be reported.** It is an artefact of applying a classifier decision rule
to a stage that is not a classifier. The end2end detector emits a `(300, 6)` table of
`[x1, y1, x2, y2, conf, cls]`, so `decide()`'s `argmax(-1)` returns *which of the six columns holds
the largest value* per box — a question with no meaning. The 794.4 difference is likewise box
coordinates in an 800-px space, not a probability. Compounding this, the harness feeds every stage
`standard_normal` noise; that is a legitimate stimulus for a fixed-width classifier but not for a
detector, whose behaviour on noise is undefined. A defensible detector check compares the boxes
that survive the pipeline's own selection (`BRT.CONF` + `N_MAX`) on **real decoded frames**, matched
by IoU and class, across the two precisions. That has not been run.

## 5c. Decode measured — the PASS survives only at median crop load (2026-09-29 22:18)

Artifact: `decision-latency-fp16-20260929T221836.json`. Unresolved item 2 is now closed.
`run_decode_cost()` times each of 20 distinct clips once, two ways: seek-sampled (the offline
token-extraction path) and stream (8 consecutive reads after priming, no seek — what a live feed
pays per decision at EMIT_EVERY=8). Only the stream term enters the verdict.

- Stream decode: **4.839 ms/frame p50** (p95 6.797) -> **38.71 ms per decision** (p95 54.38).
- Seek-sampled 16 frames: 5577 ms p50. Offline extraction only — ~17x the entire compute budget,
  and the reason token building is slow. Must never be added to the cadence figure.

| crops | compute-only ms | dec/s | + stream decode ms | dec/s | cadence |
|---|---:|---:|---:|---:|:--:|
| p50 = 54 | 292.48 | 3.42 | **331.19** | **3.02** | PASS (margin 0.7%) |
| p95 = 106 | 330.55 | 3.03 | 369.26 | 2.71 | **FAIL** |
| p99 = 119 | 339.72 | 2.94 | 378.43 | 2.64 | **FAIL** |
| cap = 128 | 346.44 | 2.89 | 385.15 | 2.60 | **FAIL** |

**Reproducibility, n=5 (same session, same engines, 22:35 repeats added).** The spread is not
random drift — it is **bimodal by condition**, and the condition is whether decode was measured in
the same process:

| run | `--clips-dir` | detector p50 ms | cap dec/s (compute) | cap dec/s (+decode) |
|---|:--:|---:|---:|---:|
| 21:45 | no | 11.982 | 3.05 | — |
| 21:59 | no | 11.720 | 3.08 | — |
| 22:18 | yes | 12.903 | 2.89 | 2.60 |
| 22:31 | yes | 12.846 | 2.90 | 2.62 |
| 22:34 | yes | 12.852 | 2.89 | 2.61 |

Within each condition the detector is tight (2.2% across the two decode-free runs, **0.4%** across
the three with decode); between conditions it is a systematic **+9.4%**. Running OpenCV CPU decode
in the same process costs the detector ~1.1 ms/frame — CPU contention, not noise. This matters for
which number is deployment-representative: **no deployment decodes zero frames**, so the decode-free
rows (cap 3.05/3.08) describe a lab condition that cannot occur in service, and the compute-only
PASS at p99/cap must not be reported at all. The three in-condition runs agree to within 0.8% at
every crop load: p50 **3.02 / 3.04 / 3.03** dec/s with decode, cap **2.60 / 2.62 / 2.61**.

Decode itself is stable across the three runs: 38.71 / 37.14 / 38.29 ms per decision
(4.839 / 4.643 / 4.786 ms/frame), a 2% spread.

**Decode figure is an upper bound, not the floor.** `cv2.getBuildInformation()` on the device
reports `GStreamer: NO`, `FFMPEG: YES` — so these 4.8 ms/frame are **CPU** decode of 1920x1080
H.264 (test clips are 1080p @ 24.8 fps). NVDEC hardware decode, which a real camera pipeline would
use, is not wired into this harness and is unmeasured. It would likely lower the decode term and
could restore the p95 row; until measured, that is a hypothesis, not a result.

## 5d. Why seek costs 72x, and why serial addition is an upper bound *(second claim WITHDRAWN 2026-10-03 — see the end of this section)*

**Seek vs sequential.** The first decode attempt returned 5,414 ms per decision — 338 ms to obtain
one frame — because `CAP_PROP_POS_FRAMES` random access forces a re-decode from the preceding
keyframe on every call. That is the **offline token-extraction** cost `build_relation_tokens.process_clip`
pays, and it is **72x** the sequential cost of the same frames. It must never enter a cadence
statement; it belongs in the thesis only as an operational fact about batch feature extraction.

**Which compute run to compose with.** An earlier draft of this section composed decode against
the 21:45 run alone (compute-only 275.97 / 312.81 / 321.77 / 328.15 ms), on the grounds that the
22:18 run's own compute figures show a detector p50->p95 spread of 20.9% against 0.3% at 21:45 —
the contention signature this report uses to reject the Mac baseline. That single-run table is
**superseded by the three-run range in §5c**: picking the cleanest run is the same selection
pressure the report criticises elsewhere, and the range is the honest object. The p50 row is the
only one stable across all three runs.

**~~Serial addition is an upper bound, not the deployment figure.~~ WITHDRAWN 2026-10-03.** The
overlap argument itself still holds — decode is CPU work, inference is GPU work, and
`stream-pipeline-utils.py`'s `video_reader` does call `await loop.run_in_executor(None, cap.read)`,
so a 38.71 ms CPU decode can hide inside a GPU decision. What is withdrawn is the **conclusion
drawn from it**, and the dilemma this section posed.

Three things changed on 2026-10-03:

1. **The either/or was a false choice, because both horns were PASS.** This section asked the
   thesis to pick between the *serial upper bound* (PASS at the median crop count) and the
   *compute path with decode overlappable* (PASS throughout). Neither is the answer. The composed
   measurement gives **clip-dependent, clock-dependent**: pinned 3,479 / 3,965 MEETS and 2,493
   FAILS; DVFS 1,141 / 1,441 / 1,511 FAILS 3 of 3. No single verdict was available to pick.
2. **"Serial addition is an upper bound" is false in the direction it claims.** Every omission is
   signed upward, but the stage sum also *double-charges* a stride of per-frame work
   (`DECISION_T_FRAMES`=16 where the contract is defined at `DECISION_EMIT_EVERY`=8). The two
   errors have opposite sign and neither dominates reliably: the construct runs **7,98 % too HIGH
   pinned** and **39,3 % too LOW under DVFS**. It bounds the cadence in neither direction.
3. **"P3 running end-to-end through that pipeline has not been measured" is no longer true.** It
   was measured the same day. `deployment/scripts/jetson-p3-multi-stream.py` drives the **P3**
   engines (`--detector-stage p3-detector --head-stage p3-relation-head-bin-e3`) through decode →
   800² letterbox preprocess → detector → zone geometry → 224² crops → appearance encoder → 74-d
   token assembly → binary E3 head under one wall clock. The sentence above describes the harness
   as it stood on 2026-09-29, when it drove P1 engines only.

What survives from this section unchanged: the **72x seek penalty** (an operational fact about
offline token extraction, never a cadence statement), the decode stability across runs, and the
rejection of single-run selection in favour of the three-run range. Quote
`p3-composed-cadence-fp16-*` artifacts for any cadence or contract statement.

## 5e. Memory, engine size and energy (2026-09-29 22:25-22:28)

Artifacts: `memory-fp16-20260929T222545.json`, `model-size-20260929T222545.json`,
`energy-p3-{detector,appearance,relation-head-bin-e3}-fp16-2026092922*.json`.

**Memory.** Host peak RSS **33.3 MB**; device used **3397.74 / 7607.06 MB** (free 4209.32). The
8 GB part is not the binding constraint for P3 — under half is in use with all engines loaded.

**Engine sizes (MB).** detector 21.457 FP16 / 38.299 FP32; appearance 2.862 / 4.473; binary E3 head
**0.190 / 0.190** (identical — the head is too small for FP16 to shrink the plan); 4-class head
0.996 / 1.420. Total FP16 deployment set = **24.5 MB**.

Correction, 2026-09-30: these sizes are from `model-size-…T222545.json`, which never left the device.
The earlier `model-size-…T212708.json`, which is held locally, reports appearance = **2.949 MB** —
the engine was rebuilt between 21:27 and 22:25 — giving a deployment set of **24.6 MB**. §5.6.8 cites
the local, reader-checkable artifact and therefore prints 24.6, not the 24.5 above. The detector and
E3-head sizes agree across both runs.

**Energy, per stage, 30 s saturated each** (`VDD_IN` module total, 500 ms sampling, n=59):

| stage | inf/s | avg power W | J/inference | inf/s per W |
|---|---:|---:|---:|---:|
| p3-detector | 77.433 | 17.356 | 0.22414 | 4.46 |
| p3-appearance (batch 54) | 49.167 | 15.741 | 0.32017 | 3.12 |
| p3-relation-head-bin-e3 | 5857.433 | 8.763 | 0.00150 | 668.4 |

Idle baseline measured separately (20 s tegrastats, n=40): **7.563 W**, CPU 56.3 °C / GPU 58.2 °C.

**Derived J per decision** (16 detector frames + 1 appearance batch of 54 + 1 head pass):

- total attribution: 16(0.22414) + 0.32017 + 0.0015 = **3.91 J/decision**
- marginal over idle: 16(0.1264) + 0.1663 + 0.0002 = **2.19 J/decision**

Report the marginal figure when comparing algorithms and the total when sizing a power budget; say
which. The relation head is **0.04%** of decision energy — the same negligible share it holds in
latency (0.06%), so the Transformer contribution is free on both axes and the detector is the whole
cost.

**Caveat that must travel with these numbers.** Each stage was measured *alone*, saturating the
GPU. The composed P3 pipeline's average wall power is **not** measured: at ~3 dec/s the GPU is busy
roughly 226 ms of each ~330 ms period while CPU decode/postprocess/crop run alongside, so the true
system draw sits somewhere between 7.6 W idle and 17.4 W saturated. Do not state a composed-pipeline
wattage. Also: `joules_per_inference` for the appearance stage is per *batch of 54 crops*, not per
crop — a per-crop restatement would be 5.9 mJ and is not what the JSON holds.

## 5f. Thermal soak, 30 min (2026-09-29 22:39-23:09)

Artifact: `deployment/artifacts/jetson-measurements/thermal-soak-p3-detector-fp16-20260929T230913.json`
(1790 tegrastats samples at 1 s, 30 x 60 s latency windows, ~5100 detector inferences per window).
Power mode **logged this time**: `nvpmodel_q = "NV Power Mode: MAXN_SUPER\n2"`.

**Result: no thermal throttling.** `possible_throttling = False`.

| | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|
| windows 8-29 (t=540-1800 s), min | 11.665 | 12.007 | 12.034 |
| windows 8-29, max | 11.725 | 12.058 | 12.340 |
| spread across 22 min | **0.51 %** | 0.42 % | — |

Over those 22 minutes tj rose 69.3 → **70.7 °C max** while p50 stayed flat within half a percent.
Orin Nano software throttle is ~95 °C, so the part ran **~24 °C below throttle** and the detector
stage is thermally stable at full duty for the measured half hour.

**The `latency_drift_pct = -7.86 %` field in the JSON must not be quoted.** It is computed
last-window-vs-first-window, and windows 0-7 are contaminated: they read p50 ≈ 12.70 ms, then the
value steps down to 11.70 ms at t ≈ 480-540 s and never returns. This is not DVFS warm-up (a step,
not a ramp) and not cooling (tj was already at 69 °C). Cause identified from the tegrastats raw
lines and device mtimes: **I was `scp`-ing files to the device during the first 8 minutes** —
`jetson-p3-demo-video.py`, `clip_view_map.json`, `clip_view_map_691.json` landed at 22:47:13-22:49:15,
i.e. exactly the step — and the transfer held ~35 % of one CPU core and ~39 MB (RAM 3369 → 3330 MB at
t=480 s, second busy core disappears). Operator error, mine, not a device behaviour. The valid soak
is the clean 22-minute segment; the honest headline is *drift within 0.51 %*, and the -7.86 % figure
is an artifact of my own interference.

Two limits that must travel with this result: the soak exercises the **detector stage alone with
fixed inputs**, not the composed pipeline (no decode, no crop extraction, no appearance batch), and
`possible_throttling` is a **latency heuristic**, not a clock or thermal-throttle register reading.

**Power under sustained load: 17.94 W mean, 18.76 W peak** (`VDD_IN`, module total), GR3D 85-87 %.
This is a worst-case stress figure — the soak runs the detector back-to-back at ~85 inf/s, whereas
the deployed cadence calls it 24 times/s. It is not the deployed system's wall power (§5e).

## 5g. Attempt to recover the P1 soak's power mode from the power trace — FAILED, do not use

**This section is a retraction of my own reasoning. I record it because the argument is seductive
and someone (me, later) will try it again.**

R32c recorded that the June P1 soak's "8 W below the 15 W MAXN budget" headroom claim rested on an
unlogged power mode. Tonight I tried to recover that mode from the power numbers:

| | P1 soak (June) | tonight, MAXN_SUPER |
|---|---:|---:|
| VDD_IN mean | 7.033 W | idle-with-engine **7.563 W** (§5e); detector soak **17.94 W** |
| VDD_IN max | 7.340 W | 18.76 W |
| GPU util mean | 16.0 % (max 44 %) | 85-87 % |
| tj mean / max | 55.88 / 56.81 °C | 69.15 / 70.69 °C |

The argument was: the P1 soak drew 7.033 W mean *while working* at 16 % GPU util, which is below
tonight's 7.563 W *idle* at MAXN_SUPER; a non-idle load cannot draw less than idle at the same
profile; therefore the June run was not at MAXN_SUPER and chapter 5's label is wrong.

**It does not hold, on two independent grounds, both already in our own evidence.**

1. **Chapter 5's own Table 5.5D refutes the premise that power identifies the profile.** The June
   power sweep set all three profiles deliberately and measured 5.31 / 5.29 / **5.22 W** average
   for 15 W / 25 W / MAXN_SUPER — the chapter's stated finding is that "average power is
   essentially identical across all three profiles (within 1.7 %)". If average power barely moves
   with the profile, **average power cannot be used to infer the profile.** The whole inference is
   circular against a result we already published.
2. **The 7.563 W "idle" is not a floor.** `_idle_power_w` samples for 10-20 s with **the TensorRT
   engine resident and the CUDA context alive**, immediately after a saturating run, on a hot SoC,
   on a device that also runs a graphical session. It is an idle-with-engine baseline defined for
   the marginal-energy subtraction, not a cold-machine minimum. The June sweep reaching 5.22 W at
   ~2 % GPU util shows the real floor is well below it.

So 7.033 W at 16 % GPU util sits unremarkably between 5.22 W at ~2 % and 17.94 W at 85 %. It is
consistent with MAXN_SUPER, consistent with the 15 W profile, and consistent with the other one.
**R32c stands exactly as originally written**: the mode was not logged, and it is not recoverable
from the artifacts. The conservative phrasing already installed in chapters 1, 5 and 6 (absolute
watts, no share-of-budget) is the correct and final treatment. Nothing further needs changing, and
the proposed 5-min nvpmodel-0 reading would prove nothing either — by (1), it would reproduce ~7 W
at any profile.

The real lesson is narrower and worth keeping: **on this device, board power tracks GPU duty cycle,
not the nvpmodel cap.** That is why the "race to idle" result in §5.6.4 works, and it is why no
power figure in this thesis may be used to argue about power *budgets* or profiles — only about
absolute draw at a stated duty cycle.

## 6. Claim permitted by this evidence

Permitted: "Under TensorRT FP16 on Jetson Orin Nano Super at MAXN_SUPER, P3 sits **on the boundary**
of the 3.0 decisions/s cadence contract: 3.02 dec/s at the median crop load (54 crops) once CPU
frame decode is included, falling to 2.60-2.71 dec/s at the p95-cap crop loads. The detector
accounts for ~70% of the compute cost and the relation head for 0.06%."

Forbidden: "P3 meets the cadence contract" unqualified (true only at p50 crop load, with 0.7%
margin, and only with CPU decode); "P3 is real-time"; any use of 73 dec/s; any restatement of the
1.13-1.71 dec/s projection as a result; the compute-only cap row 3.05-3.08 dec/s as a headline —
it was produced in a decode-free process, a condition no deployment has (§5c, n=5).

Reproducibility is now n=5 with a 0.8% spread inside the deployment-representative condition, so
the boundary result is stated as a measurement, not an estimate.

## Unresolved

1. Re-run the Mac baseline on an idle machine to settle 1825 vs 13727 (blocked on running jobs).
2. ~~Measure frame-decode cost on device~~ — **closed 22:18, see §5c.** It removes the PASS at
   every crop load above the median.
3. Detector FP16 parity is still unmeasured; the existing harness rule cannot produce it.
4. D2 said P1 is the deployed system partly because P3 missed cadence. The premise is measured
   false at median crop load and measured **true** at p95 and above once decode is counted, so D2's
   conclusion is no longer overturned by this evidence — it is narrowed. Author's decision.
5. NVDEC decode unmeasured (OpenCV built without GStreamer). Decides whether the p95 row is
   recoverable.
6. ~~Still not run on device: multi-stream, energy, thermal soak, memory~~ — energy, memory and
   engine size **closed 22:28, see §5e**; thermal soak **closed 23:09, see §5f** (no throttling,
   0.51 % drift over the clean 22 min, ~24 °C below throttle). Still open:
7. **Multi-stream cannot be measured for P3 with the current harness (blocker).**
   `jetson-multi-stream-pipeline.py:_load_runners` loads a hardcoded P1 stage list
   (`backbone`, `direct4-s314`, `scenario-s7`, `pair0..3-s7`) and `inference_loop` runs the P1
   stats-aggregation + direct4/scenario/pair ensemble. There is no P3 path (detector -> select ->
   crops -> appearance -> relation head) in it. Runbook §6 run as written would produce a **P1**
   saturation curve labelled as a P3 one. Options: (a) write a P3 streaming harness (~150 lines,
   not tonight); (b) run it as-is and report it explicitly as *P1 multi-stream*, which is still a
   real datum for D2's deployed system; (c) drop multi-stream from the thesis and state that
   single-stream cadence is the measured claim. **Author's decision.**
8. Composed-pipeline average wall power unmeasured (§5e) — stages were saturated individually.
9. **P1 soak power mode (§5g).** Measured to be *not* MAXN_SUPER; identity unconfirmed. Recommend
   a ~5-min idle+light-load reading at nvpmodel 0 to settle it. Author's decision.
10. Soak covers the detector only. A composed-pipeline soak would need the P3 streaming harness of
    item 7, so it shares that blocker.
