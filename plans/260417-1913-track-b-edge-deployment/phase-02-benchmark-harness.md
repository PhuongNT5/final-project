# Phase 02 — Benchmark Harness

**Status:** ready (research-validated)
**Priority:** P1
**Estimated time:** 5h (Jetson citations already found)
**Dependency:** Phase 01 (all ONNX files in `deployment/export/`)

---

## Research Update (2026-04-18)

Researcher report: [plans/reports/researcher-track-b-jetson-orin-projection.md](../reports/researcher-track-b-jetson-orin-projection.md)

**Jetson citations ready (Ultralytics official, YOLO11s INT8 TensorRT 640×640):**
- Orin Nano Super 8GB (25W): **5.25 ms / 190 FPS**
- AGX Orin 64GB: **2.98 ms / 336 FPS**
- Orin NX 16GB: 6.05 ms
- Fallback YOLOv8s on Orin Nano 15W: 28.25 ms (arXiv:2502.15737)

**Thesis strategy — NOT a scaling projection.** Side-by-side table: measured M4 ORT (CPU+CoreML) + cited Orin TensorRT numbers + explicit footnote that CPU vs GPU-tensorcore are different accelerator classes. Orin Nano Super = standard Orin Nano 8GB + firmware unlock (same silicon).

**Measured baseline from Phase 01 researcher:** backbone ORT CPU = 10.2ms, CoreML ANE = 2.3ms. Use as sanity check for Phase 02 harness (should reproduce within ±10%).

---

## Context Links

- Input: `deployment/export/*.onnx` (FP32 + INT8 for backbone + 7 heads)
- Output: `deployment/artifacts/benchmarks.json`, `deployment/artifacts/benchmarks.md`
- Output: `deployment/artifacts/projections.md`, `deployment/scripts/colab-tensorrt-template.py`
- Reference: plan.md §Previously Claimed Numbers (1.39ms, 13.6 MB — superseded by researcher measurements)
- Researcher report: [../reports/researcher-track-b-jetson-orin-projection.md](../reports/researcher-track-b-jetson-orin-projection.md)

---

## Overview

Measure latency, throughput, and memory of the full ONNX pipeline on Apple M4 across three
execution providers: ORT-CPU-FP32, ORT-CPU-INT8, ORT-CoreML (targeting ANE/GPU).
Break measurements down per stage so each claim in thesis Chapter 5 is independently verifiable.
Produce machine-readable JSON + thesis-ready Markdown table. Also provide a Colab TensorRT
template (template only, not executed) and a Jetson Orin Nano projection table.

---

## Key Insights

- ORT CoreML EP is available on macOS via `onnxruntime` >= 1.17 with `coremltools` installed;
  requires `--providers CoreMLExecutionProvider` and may fall back to CPU silently — log actual
  EP used via `session.get_providers()`.
- Warmup is mandatory: CoreML JIT-compiles on first run; use 10 warmup + 100 measurement runs.
- Stage breakdown requires explicit timing of: (1) frame-sample + preprocess, (2) backbone ONNX
  forward, (3) stats aggregation (mean/std/max numpy), (4) transformer head ONNX forward,
  (5) softmax + ensemble decision — wrap each with `time.perf_counter_ns()`.
- Peak memory via `psutil.Process().memory_info().rss` before/after each 100-run block.
- Model size: `os.path.getsize(onnx_path)` — report MB. INT8 ≈ 25% of FP32 for weight-dominated
  models; backbone will be larger than heads.
- Jetson projection: use published NVIDIA YOLOv8n/11n CPU→Orin Nano latency ratio (≈ 3-4×
  speedup vs x86; M4 is approximately ARM-class; treat as roughly equivalent ARM baseline).
  Projection is conservative estimate — cite NVIDIA developer blog placeholder.
- RESEARCH FLAG: confirm published NVIDIA Orin Nano latency numbers for ONNX INT8 YOLOv11s
  to use as calibration ratio for projection.

---

## Requirements

### Functional
- Benchmark all three EPs for: backbone (FP32+INT8) + direct4_s314 head (FP32+INT8) + full cascade
- Stage timing: 5 stages × 3 EPs × 2 precisions = report all combinations where applicable
- p50/p95/p99 from 100 samples for each stage and end-to-end
- Throughput: FPS = 1000 / p50_ms (end-to-end)
- Peak RSS memory delta per config
- Model size on disk for all 9 ONNX files (FP32 and INT8)
- Write `deployment/artifacts/benchmarks.json` (structured) + `benchmarks.md` (Markdown table)
- Write `deployment/artifacts/projections.md` with Jetson Orin Nano projection table + citation placeholder
- Provide `deployment/scripts/colab-tensorrt-template.py` as template (header comment only)

### Non-functional
- Script < 200 lines (`benchmark-harness.py` may call helper in `deployment/scripts/bench-utils.py`)
- Idempotent: skip re-run if `benchmarks.json` already newer than all ONNX files
- No GPU rental required; CoreML EP best-effort

---

## Related Code Files

**Create:**
- `deployment/scripts/benchmark-harness.py` — main harness
- `deployment/scripts/bench-utils.py` — timing helpers, stats computation
- `deployment/scripts/colab-tensorrt-template.py` — template, no execution
- `deployment/artifacts/benchmarks.json` — output
- `deployment/artifacts/benchmarks.md` — output
- `deployment/artifacts/projections.md` — output

**Read (do not modify):**
- `deployment/export/*.onnx` — all exported models from Phase 01
- `deployment/scripts/export-backbone.py` — reference for input shape assumptions

---

## Implementation Steps

### Step 1 — Write `bench-utils.py` (30 min)

```python
# deployment/scripts/bench-utils.py
import numpy as np, psutil, os, time

def timed_runs(fn, warmup=10, n=100):
    """Run fn() n times after warmup; return sorted latency array (ms)."""
    for _ in range(warmup): fn()
    t = []
    for _ in range(n):
        s = time.perf_counter_ns()
        fn()
        t.append((time.perf_counter_ns() - s) / 1e6)
    return np.array(sorted(t))

def percentiles(arr):
    return {
        'p50': float(np.percentile(arr, 50)),
        'p95': float(np.percentile(arr, 95)),
        'p99': float(np.percentile(arr, 99)),
    }

def model_size_mb(path): return os.path.getsize(path) / (1024 ** 2)

def peak_rss_mb():
    return psutil.Process().memory_info().rss / (1024 ** 2)
```

### Step 2 — Write `benchmark-harness.py` (3h)

```python
# deployment/scripts/benchmark-harness.py
# Usage: python benchmark-harness.py [--ep cpu|coreml] [--precision fp32|int8|both]

PROVIDERS = {
    'cpu':    ['CPUExecutionProvider'],
    'coreml': ['CoreMLExecutionProvider', 'CPUExecutionProvider'],
}
HEADS = ['direct4_s314', 'scenario_s7', 'pair0_s7', ...]  # from plan.md

# Stage 1: frame-sample + preprocess
def preprocess_stage(frame_path):
    img = cv2.imread(frame_path)
    # resize to 224x224, normalize, return (1,3,224,224) float32
    ...

# Stage 2: backbone forward
backbone_sess = ort.InferenceSession('deployment/export/backbone_fp32.onnx',
                                      providers=PROVIDERS[ep])

# Stage 3: stats aggregation (numpy — not ONNX)
def stats_agg(features):
    # features: (T, 16, 512)
    return np.concatenate([features.mean(1), features.std(1), features.max(1)], axis=-1)
    # output: (T, 1536)

# Stage 4: transformer head forward
# Stage 5: softmax + ensemble

# Collect results into dict, write JSON
results = { 'machine': platform.node(), 'timestamp': ..., 'configs': [...] }
json.dump(results, open('deployment/artifacts/benchmarks.json', 'w'), indent=2)
```

### Step 3 — Generate `benchmarks.md` (30 min)

Script generates Markdown table from `benchmarks.json`:
```
| Config | Stage | p50 (ms) | p95 (ms) | p99 (ms) | FPS | Mem (MB) |
```
Row per (EP × precision × stage). Also include model sizes table.

### Step 4 — Generate `projections.md` (30 min)

```
# Jetson Orin Nano Latency Projection

Baseline: M4 INT8 end-to-end = {X} ms measured.
Projection ratio: Orin Nano INT8 / M4 INT8 ≈ R (from [CITATION]).
Projected Orin Nano end-to-end: {X * R} ms = {1000/(X*R)} FPS.

[CITATION PLACEHOLDER]: NVIDIA Developer Blog, "NVIDIA Jetson Orin Nano Performance",
https://developer.nvidia.com/blog/... (to be filled after confirming numbers)
```

### Step 5 — Write `colab-tensorrt-template.py` (30 min)

Template only — no execution on M4. Provides shell for Colab T4:
```python
# deployment/scripts/colab-tensorrt-template.py
# Template for running TensorRT benchmark on Google Colab T4.
# DO NOT run locally — requires CUDA GPU.
# Steps: install tensorrt, export onnx→trt, warm up, measure.
# Fill in ONNX_PATH and run on Colab with GPU runtime.
```

---

## Todo List

- [ ] Write `bench-utils.py` with timing and memory helpers
- [ ] Write `benchmark-harness.py` — 5-stage timing loop, 3 EPs, FP32+INT8
- [ ] Confirm CoreML EP available: `ort.get_available_providers()`
- [ ] Run benchmark; confirm warmup eliminates JIT outliers
- [ ] Generate `benchmarks.json` with all configs
- [ ] Generate `benchmarks.md` thesis table
- [ ] Compute Jetson projection ratio from NVIDIA published numbers
- [ ] Write `projections.md` with projection table + citation placeholder
- [ ] Write `colab-tensorrt-template.py` (template only)
- [ ] Log actual EP used per session (detect silent CPU fallback)

---

## Success Criteria

1. `benchmarks.json` contains p50/p95/p99 for all 5 stages × ≥ 2 EPs (CPU FP32, CPU INT8)
2. `benchmarks.md` renderable Markdown table with ≥ 6 rows
3. CoreML EP attempted; result logged even if fallback to CPU
4. INT8 end-to-end p50 < FP32 p50 (expected; else document anomaly)
5. `projections.md` present with projection formula and citation placeholder
6. `colab-tensorrt-template.py` present with clear header comment
7. All scripts ≤ 200 lines each

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CoreML EP unavailable / silent fallback | Medium | Low | Detect via `session.get_providers()`; thesis can cite CPU-only numbers |
| Benchmark noise on M4 (thermal throttle) | Medium | Medium | Pin process affinity; run with low background load; report std dev |
| Memory measurement inaccurate (shared libs) | Low | Low | Report delta RSS, not absolute; note limitation in benchmarks.md |
| Jetson projection ratio hard to find | Medium | Medium | Use conservative 3× from NVIDIA YOLOv8n Nano data; cite clearly |
| ONNX file missing (Phase 01 incomplete) | Low | High | Guard with `assert os.path.exists(path)` early in script |

---

## Research Flag

**DELEGATE TO RESEARCHER:** "Find published ONNX INT8 inference latency on Jetson Orin Nano
for YOLOv8n or YOLOv11s from NVIDIA developer blog or Ultralytics benchmarks (2024-2025).
Need: model name, task, input resolution, latency (ms), source URL."

---

## Next Phase

Phase 03 (Multi-Stream Pipeline) can start in parallel once Phase 01 is done.
Phase 05 (Thesis Artifacts) waits on `benchmarks.json` from this phase.
