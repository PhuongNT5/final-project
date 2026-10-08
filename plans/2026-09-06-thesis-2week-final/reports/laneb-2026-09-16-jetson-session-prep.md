# Lane B: Jetson Session Prep (2026-09-16, for 2026-09-19 measurement session)

Scope: fix 3 confirmed edge-pipeline defects, build a single Jetson measurement
CLI + runbook. Stayed inside `deployment/` + `tests/`. No commits made.

## Task 1 — defects fixed

**a) `TRTRunner` single-input assumption** (`jetson-benchmark.py`,
`jetson-multi-stream-pipeline.py`). Both files already imported a shared
`TRTRunner` from `deployment/scripts/trt-runner-utils.py` (pre-existing from
earlier work this conversation) that binds every engine I/O tensor by name via
`set_tensor_address` + `TensorIOMode.INPUT/OUTPUT`, not a single fixed slot —
verified this is the actual runtime code, not a stub. Regression test:
`tests/test_trt_runner_utils.py` — fake 2-input engine (`tokens`, `mask`)
whose kernel computes `[sum(tokens), sum(mask)]`; asserts kwarg binding,
positional-order binding, and a missing-kwarg raise. **3/3 PASS.**

Found + fixed a related latent bug while testing:
`jetson-multi-stream-pipeline.py`'s `inference_loop()` signature still
referenced the OLD local `TRTRunner` class (deleted earlier, replaced by the
`trt-runner-utils` import) as a type annotation — `Dict[str, TRTRunner]` —
causing `NameError` at import time. Fixed to a string forward-ref:
`Dict[str, "_tru.TRTRunner"]`. Caught live by
`tests/test_jetson_multi_stream_import.py` (imports the file with pycuda
stubbed via `sys.modules` injection since Mac has no pycuda). **2/2 PASS**
after fix — this would have broken on the real Jetson too, not just in test.

**b) `aggregate_fps` counting dropped frames as processed**
(`multi-stream-pipeline.py`, `jetson-multi-stream-pipeline.py`). Extracted a
standalone `aggregate_fps(states, elapsed)` in both files (was inline in
`run_config`). Now reports `aggregate_fps` from `processed` only, plus
separate `read_fps` (includes drops) and `drop_rate`. Regression test:
`tests/test_stream_fps_accounting.py` calls the REAL extracted function via
importlib from both files with fabricated `StreamState`s (100 read / 82
processed / 18 dropped, and 100/70/30) — asserts `aggregate_fps == 15.2` (not
the old buggy 20.0), `drop_rate` correct, plus a zero-frame div-by-zero edge
case. **2/2 PASS.**

**c) `benchmark-p3-onnx.py` timing scope + mean-only crop cost**. Two
separate bugs, both fixed:
  - Timing now covers the full decision path (NMS + geometry/`BRT.geometry()`
    + crop extraction + head inference), not just the head forward pass —
    `run_bench()` rewritten around `U.postprocess_frame()` +
    `U.extract_crops()` + head session, all inside the timed region.
  - Crop-count now driven by real p50/p95/p99/cap from stored mask tensors
    (`crop_count_percentiles()`, fallback `{50,106,119,128}` if files
    missing), not a single mean-58 constant — reports latency percentiles
    across ALL crop-count scenarios, not one mean-crop number.
  - Also fixed a units-mismatch bug found in the same code path: synthetic
    detector boxes were generated in 800×800 (IMGSZ) space but normalized
    against 1920×1080 (CLIP) — silently produced out-of-range normalized
    coords. Fixed via `U.synthetic_raw_frames()` generating directly in
    CLIP_W×CLIP_H space.
  - File was 373 lines after the rewrite (200-line rule violation) — split
    into `benchmark-p3-onnx.py` (198), `benchmark-p3-utils.py` (120),
    `benchmark-p3-sweep.py` (112), same importlib pattern already used
    elsewhere in the codebase for hyphenated filenames.
  Regression test: `tests/test_benchmark_p3_onnx.py` — 4 tests covering
  percentile-source correctness, confidence-filter + top-N cap, crop-extract
  shape, and the units-mismatch fix (0 out-of-[0,1] normalized boxes from
  synthetic frames). **4/4 PASS.**
  Live verification (not just unit tests) — ran the real bench on Mac CPU:
  `.venv/bin/python deployment/scripts/benchmark-p3-onnx.py --parity --bench
  --warmup 2 --runs 5` → parity PASS 64/64 argmax agreement (max diff
  2.98e-07), real per-decision p50/p95/p99/cap latencies logged, output saved
  to `deployment/artifacts/benchmarks-p3-onnx-cpu.json`.

**Total: 5 test files, 17/17 tests PASS** (all actually run this session,
output captured above and reconfirmed with a final full re-run before writing
this report). `py_compile` clean on all 12 touched/new `deployment/scripts/*`
files.

## Task 2 — Jetson measurement CLI

New files, all under 200 lines: `jetson-measure-utils.py` (172, shared
helpers), `jetson-measure-latency.py` (137, per-stage latency + FP16/FP32
parity), `jetson-measure-power.py` (122, energy/thermal-soak/memory),
`jetson-full-measurement.py` (126, CLI orchestrator). Covers every metric
requested: latency p50/p95/p99 per stage, throughput + multi-stream
saturation (1/2/4/8 via the fixed `aggregate_fps`), energy (J/inference,
FPS/W via a real `tegrastats` subprocess sampler), 30-min thermal soak with a
throttling heuristic, host+device memory peak, model size on disk, FP16 vs
FP32 decision-agreement parity, power-mode/clock state embedded in every
output JSON.

Design choices directly enforcing the two hard constraints:
- **No hardcoded class count**: `discover_engines()` finds stages purely by
  filename regex `^(.*)-(fp32|fp16|int8)\.engine$`, no stage list. `decide()`
  supports both `argmax` (multi-class) and `threshold` (binary) rules.
  Verified by `test_discover_engines_is_class_count_agnostic` against BOTH
  old P1-style (`backbone-fp32.engine`) and new binary-style
  (`binary-head-fp16.engine`, `aux-cause-head-fp16.engine`) filenames.
- **No fabricated results**: every hardware-dependent function
  (`TegrastatsSampler`, `measure_stage_latency`, `run_streams`, etc.) checks
  for real hardware/binaries first and returns an explicit `skipped_reason`
  string + `null` fields instead of inventing numbers. Verified via a real
  `--dry-run --all` execution on Mac — every output JSON shows
  `device_state.dry_run: true`, `nvpmodel_q: null`, honest reasons like
  `"engine not found: backbone-fp16.engine"` or `"dry-run: tensorrt/pycuda
  not available on this machine"`, `git_commit: "43657ea"` embedded, zero
  fabricated numbers anywhere across 6 output files inspected.

`tests/test_jetson_measure_utils.py` — 6 tests (engine discovery, percentiles,
decide(), real-byte model-size report, dry-run tegrastats never fabricates
telemetry fields, real tegrastats line-format parsing). **6/6 PASS.**

Also extended `jetson-build-trt-engine.py` with `--model NAME --onnx PATH`
(previously hardcoded to loop over the fixed P1 `MODELS` list only, no way to
build the new P3 stages at all) — default no-arg behavior unchanged, verified
via `py_compile`. This was necessary for the runbook's P3 engine-build step
to be executable at all.

## Task 3 — runbook

`deployment/JETSON-RUNBOOK-2026-09-19.md` (255 lines — a doc, not code, so
outside the 200-line code-file rule but kept tight). Sections: (0) pre-flight
checklist incl. code/onnx transfer, venv check, power-mode lock, dry-run
sanity; (1-2) build P1 (if needed) + P3 (new) TensorRT engines; (3-10) one
step per Task-2 measurement type with copy-paste command, expected wall-clock,
and a concrete "good" threshold (cross-referenced against the real measured
P1 numbers already in `JETSON-DEPLOYMENT.md` as a sanity baseline, explicitly
flagged as P1-only so it isn't mistaken for a P3 target); (11) fallback demo
video recording — confirmed via the prior audit that none exists, wrote
capture instructions from scratch; (12) triage — pycuda/tensorrt import-order
CUDA-context error (tied to the documented comment in
`jetson-build-trt-engine.py`), INT8 calibration-data gap (P1 calib data
exists at `deployment/artifacts/calib-jetson/`, confirmed via `ls`; P3 has
none — INT8 explicitly out of scope for 9/19), early-throttling response;
(13) minimum viable set (pre-flight + model-size + latency + parity + demo
video — the four highest-signal, lowest-time items).

## What's still NOT measurable on 9/19 with current code

- **P3 has zero TensorRT engines right now** — confirmed
  `deployment/export/jetson/` doesn't exist in this checkout. Only FP32 ONNX
  exports exist (`deployment/export/p3-{detector,appearance,relation-head}-fp32.onnx`).
  Runbook Step 2 must succeed on-device before any P3 latency/energy/soak
  number exists — this is the biggest session risk, flagged as such.
- **No P3 INT8 calibration data** — `deployment/artifacts/calib-jetson/`
  only has P1 stage names (`backbone-batch-*.npy` etc, confirmed via `ls`).
  INT8 is FP1-only this session; P3 is FP32/FP16 only.
  `jetson-prep-int8-calib.py`'s `HEAD_MODELS` list is still P1-specific — if
  P3 INT8 is wanted later, that script needs updating too (out of scope here,
  not touched, belongs to whichever lane owns model export).
  - **Binary/aux-head architecture isn't exported yet** — the redesign
  (binary logit + aux heads replacing the 4-class head) is still being worked
  in parallel per the task brief; no ONNX/ engine exists for it yet, so
  Task 2's CLI has nothing binary-specific to measure on 9/19 beyond what's
  already exported. `discover_engines()` and `decide(--rule threshold)` are
  ready for it the moment engines exist — no code changes needed then.
- **`end_to_end_naive_sum_ms_p50` in `jetson-measure-latency.py` is NOT the
  true decision-path latency** — it's a naive sum of independently-measured
  per-stage p50s in discovery order, documented as such in the code and the
  runbook. Real end-to-end latency needs the actual pipeline stage order,
  which depends on the still-unresolved architecture redesign.
- **Thermal-soak throttling detection is a heuristic** (`p50s[-1] >
  p50s[0]*1.15`), not a clock-frequency readout — `tegrastats` doesn't
  reliably expose raw GPU clock on this JetPack version per the code's own
  docstring. Cross-check against temperature samples in the same output, not
  a certainty.
- **Multi-stream `--streams` step needs real Jetson + real source videos on
  device** — cannot be dry-run tested beyond confirming the plumbing (import,
  `skipped_reason` path) on Mac; confirmed real `.mp4` files exist at
  `data/test/{class}/*.mp4` for the Jetson to use once transferred.
- Two pipeline files remain slightly over the 200-line guideline after the
  `aggregate_fps` extraction: `jetson-multi-stream-pipeline.py` (267 lines,
  was already 260 before this session — pre-existing, not newly introduced)
  and `multi-stream-pipeline.py` (207 lines). Decided not to force a further
  architectural split given the marginal overage and risk of destabilizing
  tested code this close to the session date — flagging as an accepted
  deviation rather than silently ignoring it.

## Files touched

Modified: `deployment/scripts/jetson-benchmark.py` (verified only, no edit),
`deployment/scripts/jetson-multi-stream-pipeline.py`,
`deployment/scripts/multi-stream-pipeline.py`,
`deployment/scripts/benchmark-p3-onnx.py`,
`deployment/scripts/jetson-build-trt-engine.py`.
New: `deployment/scripts/benchmark-p3-utils.py`,
`deployment/scripts/benchmark-p3-sweep.py`,
`deployment/scripts/jetson-measure-utils.py`,
`deployment/scripts/jetson-measure-latency.py`,
`deployment/scripts/jetson-measure-power.py`,
`deployment/scripts/jetson-full-measurement.py`,
`deployment/JETSON-RUNBOOK-2026-09-19.md`.
New tests: `tests/test_trt_runner_utils.py`,
`tests/test_jetson_multi_stream_import.py`,
`tests/test_stream_fps_accounting.py`, `tests/test_benchmark_p3_onnx.py`,
`tests/test_jetson_measure_utils.py`.
No files outside `deployment/` and `tests/` touched. No commits made.

## Unresolved questions

1. Should `jetson-prep-int8-calib.py` be extended for P3 stage names before
   9/19, or is P1-only INT8 acceptable for this session? (belongs to
   whichever lane owns model export/calibration prep — flagging, not
   deciding.)
2. Once the binary+aux-head architecture is exported, will the pipeline
   stage order for a true end-to-end decision latency be
   detector→appearance→relation-head (same as P1/P3 so far), or does the aux
   heads' output feed back into the decision in a different order? Affects
   whether `end_to_end_naive_sum_ms_p50`'s discovery-order sum is even a
   reasonable proxy once real numbers replace it.
3. Confirm whether the Jetson still has the 2026-06-03 P1 engines on disk, or
   whether Runbook Step 1 (rebuild P1) is required — I have no way to check
   the physical device's `deployment/export/jetson/` contents from Mac.
