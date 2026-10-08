# Phase 05 — Jetson Measurement Session (2026-09-19, FIXED)

## Context Links

- `deployment/JETSON-DEPLOYMENT.md`, `deployment/scripts/jetson-{benchmark,build-trt-engine,prep-int8-calib,multi-stream-pipeline}.py`
- `plans/2026-09-06-thesis-2week-final/reports/audit-2026-09-14-edge-and-demo-readiness.md`
- `reports/test-touch-log.md` (2026-09-09 zero-touch precedent)
- Phase 04 `deployment/export/MANIFEST.md` is the input contract.

## Overview

- **Priority:** P0 — **the only immovable date.** **Status:** pending. **Effort:** 8h on-device.
- Capture the complete mandatory metric set for the binary pipeline on the device.

## Key Insights

- **The user operates the device. Notify the user before EVERY on-device step** and wait for confirmation.
  The runbook is therefore written as a numbered list of copy-pasteable commands with expected output and a
  pass/fail line for each — the operator should never have to improvise.
- **Zero test touches.** Latency/throughput/energy/thermal/memory use dummy or train-split tensors; parity
  is FP16/INT8 **agreement vs FP32**, not accuracy. Log this explicitly in `reports/test-touch-log.md`.
- P1 (TRT FP16, 8.62 ms) is a known-good artifact — run it first as a smoke test that the environment is sane.
- P3 edge budget currently **fails**: detector = 1745.0 ms of a 1825.0 ms decision (95.6%) → 0.55 dec/s on
  Mac; Jetson projected 1.13/1.64/1.71 dec/s vs 3.0 required. Measured cost curve
  `800x16 1.64 | 640x16 2.51 | 512x16 3.77 | 416x16 4.92` → **measure 800/512/416 on device** and report
  the curve, not a single verdict.
- **Benchmark BOTH E0 and the winning encoder** (architecture frozen 18/9) so the deployment chapter can
  quote a like-for-like cost comparison between the baseline and the redesign.
- **The redesigned encoder does NOT fix the edge failure and must not be presented as fixing it.** The
  1825 ms/decision is dominated by detector + appearance extraction; the relation head is a small fraction,
  so a 3.2× smaller encoder moves the total very little. Measure the relation-head share explicitly so the
  claim is quantified rather than asserted.

## Requirements — mandatory metric set (all must be captured)

| # | Metric | Protocol |
|---|---|---|
| M1 | Latency p50/p95/p99 | 500 iters after 50 warmup, per engine, FP16 + INT8 |
| M2 | Throughput | batch 1 and batch 8, decisions/s |
| M3 | Multi-stream saturation | 1,2,4,8 streams; record drop rate separately (D3 fix) |
| M4 | Energy | `tegrastats` @100 ms → J/inference and FPS/W |
| M5 | Thermal soak | **30 min** continuous; temp + clock + sustained FPS curve |
| M6 | Memory | peak host + device RSS per engine |
| M7 | Model size | on-disk bytes per engine/precision |
| M8 | FP16 vs INT8 parity | output agreement vs FP32 on 100 train clips (≥99% argmax agreement target) |
| M9 | Detector cost curve | imgsz 800/512/416 end-to-end decisions/s |

## Architecture

Transfer `deployment/export/` + calib → build TRT engines on device (FP16, INT8-calibrated) → run
`jetson-benchmark.py` (M1/M2/M6/M7) → `jetson-multi-stream-pipeline.py` (M3) → tegrastats wrapper (M4/M5)
→ parity script (M8) → detector sweep (M9). Every script writes JSON to `deployment/results/2026-09-19/`.

## Related Code Files

Modify (before 19/9): `jetson-benchmark.py` (MODELS hardcodes P1 at lines 31-32 → parameterise; TRTRunner
multi-input, phase-03 D2), `jetson-multi-stream-pipeline.py`, `jetson-build-trt-engine.py` (P1-only → accept
the binary head). Create: `deployment/JETSON-RUNBOOK-2026-09-19.md`, `deployment/scripts/jetson-thermal-soak.py`,
`deployment/scripts/jetson-parity-check.py`.

## Implementation Steps

**Pre-session (by 18/9 EOD):** runbook written; all scripts parameterised; D2 fix verified; dry-run done;
artifacts staged with checksums; a **fallback**: screen recording + raw `tegrastats` logs so numbers can be
re-parsed offline if a script fails on device.

**Session (19/9), each step preceded by a user notification:**
1. Environment check: JetPack/TRT versions, `nvpmodel` mode, `jetson_clocks` state → record verbatim.
2. Smoke test: rebuild and run the **P1** FP16 engine; expect ≈8.62 ms. If it deviates >20%, stop and
   record the environment difference before continuing.
3. Build binary-head engines: FP16, then INT8 with the train-split calibration cache.
4. M1/M2/M6/M7 via `jetson-benchmark.py`.
5. M8 parity (100 train clips) — must run **before** the long thermal job, in case of time loss.
6. M3 multi-stream 1/2/4/8.
7. M9 detector sweep 800/512/416.
8. M4+M5: 30-min thermal soak with tegrastats sampling (longest single job — run last).
9. Copy `deployment/results/2026-09-19/` off the device; checksum; append a **zero-touch** entry to
   `reports/test-touch-log.md`.

## Todo List

- [ ] Runbook written and dry-run passed (18/9)
- [ ] User notified + session scheduled
- [ ] M1..M9 captured
- [ ] Results copied off-device + checksummed
- [ ] test-touch-log zero-touch entry appended

## Success Criteria

G5: M1-M8 captured for the binary head in FP16 and INT8; M9 captured for ≥2 image sizes. Partial credit
order if time runs out: M1 > M8 > M2 > M4 > M3 > M5 > M9.

## Risk Assessment

R2 (no second date) — mitigated by the dry-run, the copy-pasteable runbook, the P1 smoke test and the raw-log
fallback. R9 (D2) is a hard blocker. R11 (P3 budget fails) — expected; report the curve as an engineering
finding, and state the imgsz-512 configuration as the deployable candidate.

## Security Considerations

Footage on the device must be removed after the session. No credentials in the runbook. The user physically
controls the device throughout.

## Next Steps

Feeds the deployment chapter and the number sheet (phase 07). Accuracy work continues in phase 06
independently — the two do not block each other.
