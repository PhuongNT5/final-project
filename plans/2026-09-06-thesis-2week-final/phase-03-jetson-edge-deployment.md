---
title: "Phase 03 — Lane B: Jetson Edge Deployment (MT3)"
status: pending
---

# Phase 03 — Jetson Edge Deployment (Lane B: B1-B3)

**File ownership (exclusive):** `deployment/`, `tools/jetson_benchmark.py`. Concurrent with Lane A
(separate hardware — Jetson vs Mac/Kaggle/Colab).

## Context links
- Binding brief: `plans/2026-09-06-thesis-2week-final/brief.md` D2 (§0), §4g (Jetson metric
  contract), hard rule 5 (§9: only measured numbers, literature labelled as context)
- `plans/2026-09-05-thesis-final-push/reports/t1-objectives-audit.md` unresolved q2 — flags
  `deployment/artifacts/projections.md` "measured M4 + Jetson" line as overstated (Jetson rows there
  are literature citations, not our pipeline)
- `deployment/JETSON-DEPLOYMENT.md`, `deployment/JETSON_QUICKSTART.md` — existing (partly stale)
  setup docs
- `colab/README-yolo26.md` §Step 3 — the intended real-Jetson metric checklist

## Overview
Priority P1. Status pending. D2 locks full physical Jetson Orin Nano access — no literature rows
allowed in the deployment chapter (brief hard rule 5). Re-export the FROZEN headline pipeline
(from phase-01 A4, day 6) to TensorRT FP16/INT8, measure the full §4g metric set on real hardware,
and wire the live demo. Independent of Lane A's compute (different device) but data-dependent on
A4's frozen weights.

## Key insights — critical repo finding
- **`deployment/JETSON-DEPLOYMENT.md` header says "no physical Jetson available in the thesis
  budget" (dated 2026-05) — this is STALE.** `deployment/artifacts/benchmarks-jetson-{fp16,int8}.json`
  (timestamp 2026-06-03, host `phuongnguyen-desktop`, platform `Linux-...-tegra-aarch64`),
  `soak-jetson-summary.json` (real tegrastats log, 42.5 min), `multi-stream-jetson-fp16-realtime.json`,
  and `power-sweep-summary.json` are ALL real on-device measurements. The toolchain (TRT 10.3.0,
  tegrastats power sampling, multi-stream harness) already works end-to-end.
- **BUT those artifacts benchmark the OLD P1 pipeline** (`accuracy_4class_correct: 55, total: 64` in
  `benchmarks-jetson-fp16.json` = the direct4+cascade heads, not the P3 relation-Transformer that
  phase-01 will freeze as headline). B1/B2 here means **re-export + re-benchmark the NEW headline**,
  reusing the proven toolchain — this is faster than the brief's schedule implies since the hard
  Jetson-bring-up work (bootstrap, TRT build, tegrastats power sampling, multi-stream harness) is
  already done and validated.
- `deployment/artifacts/projections.md` mixes measured M4 numbers with literature-cited Jetson
  numbers (Ultralytics official benchmarks, arXiv:2502.15737) for a YOLO11s-backbone-only model —
  per brief hard rule 5, this file must be corrected to clearly separate "our measurement" from
  "literature context", or dropped from the final deployment chapter once real numbers exist.
- Existing `deployment/scripts/jetson-benchmark.py` benchmarks 7 named heads (`backbone, direct4-s314,
  scenario-s7, pair0-3-s7`) hardcoded — this MODELS list must be updated once phase-01 freezes the
  new headline's head names (relation-Transformer, not direct4/cascade).

## Requirements
- Functional: TensorRT FP16 + INT8 engines for the frozen headline; latency p50/p95/p99 ms/frame
  and ms/clip; FPS; multi-stream aggregate throughput + zero-drop verification; params/FLOPs; model
  size; INT8 compression ratio; J/inference and FPS/W; 30-min soak; live demo end-to-end.
- Non-functional: every number measured on real Jetson hardware (brief hard rule 5); no literature
  citation presented as "our result".

## Architecture
Frozen headline weights (phase-01 A4 output) -> `deployment/scripts/export-backbone.py` +
`export-transformer-heads.py` (adapt head list to new headline) -> ONNX -> Jetson (rsync via
`deployment/scripts/jetson-rsync-from-mac.sh`) -> `deployment/scripts/jetson-build-trt-engine.py
--mode {fp16,int8}` -> `deployment/scripts/jetson-benchmark.py` (latency/FPS/size) ->
`deployment/scripts/jetson-multi-stream-pipeline.py` (aggregate throughput + zero-drop) ->
`deployment/scripts/jetson-power-sweep.sh` (J/inference, FPS/W via tegrastats) -> soak 30 min ->
`deployment/dashboard/app.py` + `jetson-live-monitor.py` (live demo).

## Related code files
- Run/adapt: `deployment/scripts/export-backbone.py`, `export-transformer-heads.py`,
  `transformer-head.py` (update head list to new headline architecture from phase-01)
- Run: `deployment/scripts/jetson-rsync-from-mac.sh`, `jetson-bootstrap.sh`,
  `jetson-build-trt-engine.py`, `jetson-prep-int8-calib.py`
- Modify: `deployment/scripts/jetson-benchmark.py` (`MODELS` list, `ENGINE_DIR`/`FEAT_PATH` if new
  headline uses different feature files), `tools/jetson_benchmark.py` (works on any Ultralytics
  detector `.pt` already — point `--weights` at phase-01's new detector)
- Run: `deployment/scripts/jetson-multi-stream-pipeline.py`, `jetson-power-sweep.sh`
- Run/verify: `deployment/dashboard/app.py`, `deployment/dashboard/jetson-live-monitor.py`
- Correct: `deployment/JETSON-DEPLOYMENT.md` (remove stale "no physical Jetson" header),
  `deployment/artifacts/projections.md` (separate measured vs literature, or retire)
- Output artifacts (new, will overwrite/supersede old): `deployment/artifacts/benchmarks-jetson-
  {fp16,int8}.json`, `soak-jetson-summary.json`, `multi-stream-jetson-fp16-*.json`,
  `power-sweep-summary.json`

## Implementation steps
### B1 — TRT export FP16/INT8 (day 1-3, parallel with Lane A)
1. Day 1: `bash deployment/scripts/jetson-bootstrap.sh` — re-verify JetPack/TRT/PyTorch/ORT versions
   on the physical device (toolchain already proven working per artifacts above; this is a
   verification pass, not first bring-up).
2. Day 2 (spike, using CURRENT P1 weights as a placeholder while phase-01 A1/A3 run): confirm
   `export-backbone.py` + `export-transformer-heads.py` -> ONNX -> `jetson-build-trt-engine.py
   --mode fp16` still round-trips cleanly end to end (regression check on the toolchain).
3. Day 3: once phase-01 A1 detector lands (Gate 1), re-export ONNX for the new detector +
   whatever head phase-01's A3 produces; `jetson-build-trt-engine.py --mode fp16` then
   `--mode int8 --calib-dir deployment/artifacts/calib-jetson` (INT8 calibration batches must be
   regenerated via `jetson-prep-int8-calib.py` on the new feature distribution).

### B2 — Full §4g metric set (day 4-7)
4. Day 4: `deployment/scripts/jetson-benchmark.py --mode fp16 --runs 300` then `--mode int8
   --runs 300` — latency p50/p95/p99 ms/frame, ms/clip (multiply by clip frame count), FPS, model
   size, params (`model.info()`), FLOPs.
5. Day 5: INT8 calibration + parity check — compare INT8 vs FP16/FP32 accuracy on the held-out test
   features (`deployment/scripts/verify-onnx-parity.py` pattern) -> INT8 compression ratio
   (size(FP16 engine) / size(INT8 engine)).
6. Day 6: `deployment/scripts/jetson-multi-stream-pipeline.py --streams 1 2 3 4 --source <4 test
   clips>` — multi-stream aggregate FPS + per-stream `dropped` count (zero-drop verification is
   `dropped == 0` for all streams).
7. Day 7: `deployment/scripts/jetson-power-sweep.sh` across power modes -> J/inference and FPS/W
   (already computed in `power-sweep-summary.json`'s `energy_mj_per_inference` field — extend to
   the new headline). 30-min soak: rerun the tegrastats soak harness, confirm `throttle_events: 0`.

### B3 — Live demo (day 8-10)
8. Day 8: wire `deployment/dashboard/app.py` (Streamlit) + `jetson-live-monitor.py` to the new
   TensorRT engine end to end: camera/video -> Jetson inference -> dashboard display.
9. Day 9: rehearse the full pipeline (camera -> edge -> server -> dashboard per PROPOSAL.txt's
   architecture diagram); fix any drop/latency spike found.
10. Day 10: **Gate 3 check** — run unattended for 10 min. PASS -> live demo at defence. FAIL -> cut
    a fallback recording (`deployment/artifacts/live-demo-stream.jsonl` pattern already exists from
    a prior run — reuse format) and the defence uses the recording, stated explicitly as such.

## Todo list
- [ ] `deployment/JETSON-DEPLOYMENT.md` header corrected (physical access confirmed, D2)
- [ ] B1: new-headline ONNX exported, TRT FP16 + INT8 engines built on-device
- [ ] B2: latency p50/p95/p99 (ms/frame + ms/clip), FPS, params/FLOPs, model size measured
- [ ] B2: INT8 compression ratio computed and accuracy-parity checked
- [ ] B2: multi-stream aggregate FPS + zero-drop confirmed (or drops documented)
- [ ] B2: 30-min soak completed, throttle_events == 0 or documented
- [ ] B2: J/inference and FPS/W measured across power modes
- [ ] B3: live demo wired camera->edge->server->dashboard
- [ ] Gate 3 outcome recorded (live or fallback-recorded, stated explicitly)
- [ ] `deployment/artifacts/projections.md` corrected or retired (no literature-as-our-result)

## Success criteria
- `deployment/artifacts/benchmarks-jetson-{fp16,int8}.json` regenerated with a 2026-09 timestamp
  and the NEW headline's model names (not `direct4-s314`/`scenario-s7` from the old pipeline).
- `soak-jetson-summary.json` shows `throttle_events: 0` and `duration_min >= 30`.
- `multi-stream-jetson-fp16-realtime.json` shows `dropped: 0` for every stream at the target
  concurrency (2-4 streams per PROPOSAL.txt's multi-stream requirement).
- `power-sweep-summary.json` contains `energy_mj_per_inference` for the new headline.
- Gate 3: either an unattended 10-min live run log, or a fallback recording file path, explicitly
  labelled as one or the other in `deployment/artifacts/`.
- Zero rows in the final deployment chapter sourced from `[Ult.2025]`/`[arXiv:2502.15737]` presented
  as our measurement (brief hard rule 5).

## Risk assessment
- **A3 (phase-01) slips past day 5**: B1 step 3 (re-export new headline) delays correspondingly —
  mitigate by keeping the B1 step-2 regression spike on old weights so the TOOLCHAIN risk is retired
  early, independent of Lane A's timeline.
- **INT8 accuracy collapse** on the new detector's activation ranges (backbone INT8 drift was
  already 0.36 max on the old pipeline per `deployment/README.md`) — mitigate: keep FP16 as the
  reported headline if INT8 parity fails, per brief's "measured numbers only" rule (report the
  failure, don't hide it, per hard rule 2).
- **Multi-stream drops** under 4 concurrent streams on Orin Nano (8GB) — mitigate: report the
  measured max zero-drop stream count honestly rather than forcing 4; PROPOSAL.txt only requires
  "multi-stream" capability, not a specific count.

## Security considerations
- Live demo dashboard (`deployment/dashboard/app.py`) — bind to localhost/LAN only during rehearsal,
  no public exposure of the Jetson's inference endpoint.
- No PII in the Unsafe-Net clips beyond generic factory-floor footage already in the dataset.

## Next steps
- B2's full metric set feeds phase-04 Ch5 (edge deployment chapter) directly — Ch5 is blocked on B2
  completion (day 7), not just B1.
- B3's demo readiness feeds phase-06 (defence) Gate 3 status directly into the Q&A brief.

## Unresolved questions
1. Is the Jetson device still physically available for the full 2-week window, or only for a subset
   of days? Brief D2 says "full physical access" but doesn't state a date range — schedule assumes
   continuous day 1-10 access.
2. Should `deployment/artifacts/benchmarks-jetson-fp16.json` etc. (old P1 pipeline numbers) be kept
   as a historical-ladder comparison in Ch5, or fully overwritten? Recommend: rename old files with
   a `-p1-archive` suffix before overwriting, so the "old vs new pipeline, same hardware" comparison
   is preservable if useful for the thesis narrative.
