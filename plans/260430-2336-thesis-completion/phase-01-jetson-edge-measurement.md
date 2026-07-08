---
priority: must-have
effort: 5 person-days
depends_on: []
primary_device: Jetson Orin Nano 8 GB
branches: A=Jetson Orin Nano 8 GB (primary), B=NVIDIA LaunchPad cloud (fallback), C=x86 NUC+RTX A2000 (final fallback)
---

# Phase 01 — Jetson Edge Measurement

## Context Links

- `deployment/README.md` — measured M4 CPU-FP32/INT8 numbers (replicate protocol here)
- `deployment/scripts/benchmark-harness.py` — source of M4 benchmark; adapt for Jetson
- `deployment/artifacts/benchmarks.md` — M4 results to compare against
- `deployment/export/` — all ONNX FP32 + INT8 graphs ready to transfer
- `report/chapter-05-edge-deployment.md` — §5.6 currently a projection; this phase replaces it
- `plans/260417-1913-track-b-edge-deployment/phase-02-benchmark-harness.md` — prior harness spec

## Overview

Chapter 5 §5.6 currently reports Jetson throughput as a projection derived from published Orin benchmarks, not a direct measurement. This phase transfers the exported ONNX graphs to Jetson hardware (or a fallback), runs the benchmark harness end-to-end, measures latency/FPS/power/memory for N=1..4 streams, and replaces the projection with a real table. The contribution becomes the only paper in the Unsafe-Net ecosystem with a reported edge deployment number.

## Key Insights

- Paper reports zero edge numbers, zero model-size numbers, zero energy figures (Abstract + §3, paper). Any measured Jetson number is a direct novelty claim.
- M4 canonical numbers (Chapter 5 Table 5.2 — GROUND_TRUTH.md): CPU-FP32 p50 = **28.8 ms / 34.7 FPS**; CPU-INT8 = **32.0 ms / 31.3 FPS**. Jetson Orin Nano TRT INT8 projected ≈ 4–6 ms backbone; total pipeline likely 10–15 ms.
- INT8 is slower than FP32 on M4 (dequantize overhead). On Jetson, INT8 via TRT typically *does* help (matrix-multiply units support INT8 natively); reporting both FP32 and INT8 Jetson numbers makes for a clean cross-device comparison.
- ONNX Runtime + TensorRT EP path already validated on M4 CoreML EP; same script parameterization applies.
- Power envelope: Jetson Orin Nano ~5–10 W active. Paper hardware (dual GTX 1080 + i9-9900) ≈ 400–500 W. Energy efficiency ratio is a strong secondary contribution.
- Multi-stream pipeline (`deployment/scripts/multi-stream-pipeline.py`) already tested at N=1..4 on M4; transfer unchanged.
- **Primary branch: Jetson Orin Nano 8 GB.** Do not spend more than 0.5 d on procurement. If unavailable by end of Day 1, go to Branch B immediately.

## Requirements

**Functional**
- Measure e2e p50/p95/p99 latency (ms) for ensemble inference (backbone + stats-agg + direct4 + cascade) at N=1,2,3,4 concurrent streams.
- Measure FPS per stream and aggregate.
- Measure peak memory (RSS + GPU VRAM where applicable).
- Measure board power draw (W) during sustained inference using `tegrastats` or `jtop`.
- Sustain 30-minute soak run at N=2 without frame drops (matches M4 protocol).
- Report accuracy parity on 64-clip test set to confirm ONNX TRT graphs are bit-equivalent to CPU-FP32 ensemble (56/64 or 55/64).

**Non-functional**
- All runs reproducible from a single shell script.
- Results logged to `deployment/artifacts/jetson-benchmarks.json` and `jetson-benchmarks.md`.
- Protocol identical to M4 benchmark (100 warm runs, batch=1, 32-frame window) for fair comparison.

**Acceptance thresholds — one row per (device, precision) for FP32 / FP16 / INT8 on Jetson**
- Produce: e2e p50/p95 latency (ms), FPS, board power (W from tegrastats), peak memory (MB).
- Latency ≤ 33 ms p50 for N=1 stream on Jetson TRT INT8 (real-time budget).
- Accuracy drop vs. CPU-FP32 ≤ 1 sample (55/64 or better).
- Zero frame drops across 30-min soak.
- If Jetson unavailable: Branch B or C must still produce one real GPU row; projection-only is not acceptable.

## Architecture / Approach

### Branch A — Jetson Orin Nano/AGX direct

1. Transfer ONNX graphs via SCP: `scp -r deployment/export/ jetson:~/unsafe-net/export/`
2. Install TensorRT 8.6+ via JetPack 5.x/6.x (already in JetPack, no extra install).
3. Install ONNX Runtime with TensorRT EP: `pip install onnxruntime-gpu` (ORT 1.18+ for opset 18 support).
4. Adapt `benchmark-harness.py`: set `providers=['TensorrtExecutionProvider','CUDAExecutionProvider']`, re-run.
5. Run `jtop` or `tegrastats --interval 500` in a background process to log power + memory.
6. Run `multi-stream-pipeline.py --streams 1 2 3 4` with same video sources.
7. Collect 100-run benchmark; log to `deployment/artifacts/jetson-benchmarks.json`.

### Branch B — NVIDIA LaunchPad cloud Jetson (fallback)

Same as Branch A but remote. Obtain LaunchPad access at `developer.nvidia.com/launchpad`.
Transfer files via LaunchPad SSH. Use `tegrastats` remotely. Note "cloud-hosted Jetson" in thesis §5.6 wording. Script everything in one bash script before session start; LaunchPad sessions expire.

### Branch C — x86 + RTX A2000 NUC (final fallback)

Rename contribution to "commodity GPU edge device." Use TensorRT EP via ONNX Runtime on x86.
Measure with `nvidia-smi dmon` for power. Title §5.6 "GPU Edge Node Measurement." This is the last resort; activate only if Branches A and B both fail.

### Scripts to produce

- `deployment/scripts/jetson-benchmark-harness.py` — thin wrapper over existing harness; parameterized for ORT provider selection.
- `deployment/scripts/jetson-power-logger.sh` — wraps `tegrastats` → CSV.
- `deployment/scripts/jetson-soak-test.py` — 30-min loop at N=2, logs drop count.

## Related Code Files

| Action | Path |
|--------|------|
| Modify | `deployment/scripts/benchmark-harness.py` — add `--provider` arg |
| Create | `deployment/scripts/jetson-benchmark-harness.py` |
| Create | `deployment/scripts/jetson-power-logger.sh` |
| Create | `deployment/scripts/jetson-soak-test.py` |
| Create | `deployment/artifacts/jetson-benchmarks.json` |
| Create | `deployment/artifacts/jetson-benchmarks.md` |
| Modify | `report/chapter-05-edge-deployment.md` — replace §5.6 with real table |

## Implementation Steps (Day-by-Day)

**Day 1 — Hardware access + environment**
1. Confirm Jetson Orin Nano 8 GB access. If unavailable by EOD, activate Branch B (LaunchPad). Do not spend > 0.5 d on procurement.
2. Prepare transfer package: `tar czf unsafe-net-deploy.tar.gz deployment/export/ data/test/ deployment/scripts/`.
3. Transfer to device. Verify ORT ≥ 1.18: `python -c "import onnxruntime; print(onnxruntime.__version__)"`. Upgrade if needed.

**Day 2 — Accuracy parity + FP32 latency**
4. Run `verify-onnx-parity.py` on device; confirm graphs load.
5. Run ensemble inference on all 64 test clips; report correct/64 (accept ≥ 55/64).
6. Run `benchmark-harness.py --provider TensorrtExecutionProvider --precision fp32 --runs 100`. Log p50/p95/p99 per stage.

**Day 3 — INT8 + FP16 latency + power**
7. Run `benchmark-harness.py --provider TensorrtExecutionProvider --precision int8 --runs 100`. Log p50/p95/p99.
8. If device supports FP16 (Orin does): run FP16 variant. Log results.
9. Start `jetson-power-logger.sh` (tegrastats --interval 500); run 100-iter benchmark; kill logger; compute mean/peak W.

**Day 4 — Multi-stream + soak**
10. Run `multi-stream-pipeline.py --streams 1 2 3 4 --max-frames 240`. Record aggregate FPS + drop count per N.
11. Run `jetson-soak-test.py --streams 2 --duration 1800` (30 min). Log drop count, memory HWM.

**Day 5 — Artifacts + chapter update**
12. Write `deployment/artifacts/jetson-benchmarks.json` and `jetson-benchmarks.md` (one row per device×precision).
13. Update `report/chapter-05-edge-deployment.md` §5.6: replace projection paragraph with measured table (latency, FPS, power, memory, multi-stream). Add §5.x Energy Efficiency (W per stream vs. paper desktop GPU ~400 W).

## Todo List

- [ ] Day 1: Determine hardware branch (A/B/C); set up device access
- [ ] Day 1: Prepare and transfer deployment package to target device
- [ ] Day 1: Verify ORT ≥ 1.18; install TensorRT EP if needed
- [ ] Day 2: Run accuracy parity check (64-clip test set, expect ≥ 55/64)
- [ ] Day 2: Run FP32 100-run latency benchmark; log p50/p95/p99 per stage
- [ ] Day 3: Run INT8 latency benchmark; run FP16 if supported
- [ ] Day 3: Run power measurement (tegrastats or nvidia-smi dmon)
- [ ] Day 4: Run multi-stream scaling test N=1..4
- [ ] Day 4: Run 30-min soak test at N=2
- [ ] Day 5: Write jetson-benchmarks.json + jetson-benchmarks.md
- [ ] Day 5: Update §5.6 + add §5.x Energy Efficiency in chapter-05-edge-deployment.md

## Success Criteria

- §5.6 contains a measured latency table (not a projection) with at minimum: e2e p50, e2e p95, FPS, board power (W), memory (MB) for N=1 stream on TRT INT8.
- Multi-stream table shows N=1,2,3,4 aggregate FPS.
- Accuracy on Jetson is ≥ 55/64 (within 1-sample of M4 CPU-FP32 baseline).
- Soak test shows 0 dropped frames over 30 min at N=2.
- Jetson latency ≤ M4 CPU-FP32 (28.8 ms p50 per GROUND_TRUTH.md) due to GPU acceleration; failure to beat M4 is a reportable finding, not a blocker.

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Jetson unavailable | High | Branch B (LaunchPad) ready; Branch C (RTX A2000) as final fallback; all fallbacks produce real GPU numbers |
| TensorRT version mismatch | Medium | Pin ORT 1.24.4; document version matrix; use CUDA EP as fallback if TRT EP fails |
| ONNX dynamo opset 18 not supported on older TRT | Medium | Export backup with `opset_version=17` and `torch.onnx.legacy_export=True`; already tested on M4 ORT 1.24.4 |
| Power logger permission denied (`tegrastats` needs sudo) | Low | Use `jtop` Python library (no sudo required) as alternative |
| Soak test instability (memory leak in pipeline) | Low | Run with `--max-frames 0` to fully drain; monitor RSS with `psutil` inside harness |
| LaunchPad session timeout (Branch B) | Medium | Script everything in one bash script; re-run from checkpoint; export results before session expires |

## Security / Ethics / Privacy

- Test videos are from Unsafe-Net public dataset (ethics-approved per §2.1 of paper). No live camera needed for benchmarking.
- If using Branch B (cloud LaunchPad): do not upload full training dataset; use only the 64 held-out test clips.
- ONNX model weights are research artifacts; do not publish on public endpoints without clearing Unsafe-Net dataset terms.

## Dependencies + Next Steps

- Output consumed by: **Phase 07** (thesis restructure) to populate §5.6 and energy table; **Phase 08** (demo script).
- Landing chapter: `report/chapter-05-edge-deployment.md` §5.6 + new §5.x (Energy).
- Unresolved: Student must confirm hardware access before this phase can begin.
