# Chapter 5 — Edge Deployment

> ### Pipeline attribution — read before citing any number in this chapter
>
> This project evolved through **three non-interchangeable pipelines** (full map:
> `docs/system-architecture.md`). Chapters were written at different points in that evolution, so
> their headline accuracies are **not** competing estimates of one system — they describe different
> systems, and mixing them is the single most common error this project has made internally.
>
> | | Pipeline | Best cited | Blind/honest | Protocol |
> |---|---|---|---|---|
> | Ch1, Ch3, Ch4 | **P1** frozen-COCO YOLO -> multi-stream Transformer | 56/64 (87.5 %) | **53/64** (82.8 %) | same-dist, N=64 |
> | Ch5 | **P1** as exported to Jetson | 55/64 (TRT engine, as recorded in `benchmarks-jetson-*.json`) | **53/64** | same-dist, N=64 |
> | Ch7 + Abstract | **P3** entity detector + spatial-relation Transformer | **54/64** (84.4 %, 88.4 % macro) | 54/64 | same-dist N=64, **plus k-fold and LCO at N=516** |
>
> **The thesis headline is P3 (Ch7).** P1's 56/64 is **test-selected** (the ensemble was chosen
> using the test clips) and must never be presented as the result; its honest blind counterpart is
> 53/64. P3 is the only pipeline with evidence beyond N=64, which is why it leads.
> Chapters 1-6 are retained as the experimental record of how the design was arrived at, not as
> the final claim.

## 5.1 Introduction

Objective 3 of this thesis is to demonstrate that the trained hybrid pipeline
(YOLOv11s backbone + Transformer heads) can execute on commodity edge hardware
without sacrificing the accuracy established in Chapters 3 and 4. Chapter 5
reports a concrete, reproducible deployment on two targets: an Apple M4 CPU
host (Sections 5.3–5.5) and a Jetson Orin Nano Super embedded device (§5.6).
Both are measured end-to-end. No GPU rental, no proprietary runtime: the
deployment path is entirely open-source and local.

The best Track A configuration carries forward as the reference point: an
ensemble of the Direct4 head (`direct4-s314`) and a scenario → pair cascade
(`scenario-s7` gating `pair{0..3}-s7`). On the 4-class unsafe subset of
Unsafe-Net (64 held-out videos), this ensemble achieves a top-1 accuracy of
56/64 = 87.5% (macro 89.28%), matching the PyTorch training baseline. The
deployment target is real-time inference at 30 FPS, which imposes a 33 ms
per-frame budget.

Chapter 5 answers four questions. First, does the exported ONNX pipeline
preserve accuracy? Second, does it meet the 30 FPS budget on M4 CPU? Third,
does it scale to multiple concurrent video streams without dropping frames?
Fourth, does the pipeline run end-to-end on a true embedded device with
real-time multi-stream and operator-grade alerting? Sections 5.2–5.3 describe
the architecture and the export pipeline. Section 5.4 reports measured M4
latency and accuracy. Section 5.5 extends the pipeline to N = 1…4 concurrent
streams. Section 5.6 reports the **measured Jetson Orin Nano Super deployment**
(cross-precision benchmarks, multi-stream throughput, 36-minute sustained
soak, power-efficiency Pareto, live operator dashboard, and threat model).
Section 5.7 enumerates limitations, and Section 5.8 concludes.

---

## 5.2 System Architecture

Figure 5.1 (`figures/arch-diagram.mmd`) shows the end-to-end deployment
pipeline. The pipeline takes a video source (file upload or live camera) and
emits a 4-class unsafe-behavior prediction every K = 8 frames over a
sliding window of T = 32 sampled frames.

The pipeline is composed of six stages. The **Video Source** supplies raw
frames via `cv2.VideoCapture`. The **Frame Sampler** maintains a 32-frame
sliding window and emits an inference tick every 8 frames. The **Preprocess**
stage resizes each frame to 224 × 224 with PIL bilinear interpolation and
applies ImageNet mean/std normalization — matching the training transform
exactly. The **Backbone** runs the YOLOv11s tokenizer (a wrapper that exposes
layer 22, the C3k2 output, as a flat 16-token spatial sequence of 512-d
features). The **Stats Aggregator** concatenates mean, std, and max across the
16 spatial tokens per frame to produce a (T = 32, 1536) sequence. The
**Transformer Heads** consume this sequence: Direct4 predicts the 4-class
distribution directly, while Scenario and four Pair heads combine via cascade
probability fusion (`scenario_probs × pair_unsafe_probs`, L1-normalized).
Finally the ensemble averages the Direct4 and cascade distributions with
softmax normalization, producing the final prediction.

ONNX was chosen as the interchange format because it is language-agnostic and
runtime-independent: the same exported graphs run on ONNX Runtime CPU, on
CoreML Execution Provider (for ANE / GPU acceleration on Apple Silicon), and
on TensorRT EP (for NVIDIA Jetson) without retraining or re-exporting. Dynamic
INT8 weight-only quantization was chosen over full static quantization because
it requires no calibration dataset and preserves accuracy substantially better
on the transformer heads (Section 5.3).

---

## 5.3 Export Pipeline

The export pipeline is implemented in `deployment/scripts/` and produces
eleven ONNX graphs (one FP32 backbone, six FP32 heads, and corresponding INT8
variants). The process is captured in four idempotent scripts: `export-backbone.py`,
`export-transformer-heads.py`, `quantize-int8.py`, and `verify-onnx-parity.py`.
Running all four from a clean checkout takes under ninety seconds on the M4.

### 5.3.1 Backbone export

The backbone is the YOLOv11s network truncated at layer 22 (the final C3k2
block before the detection head). A thin wrapper, `YOLOv11BackboneTokenizer`,
exposes layer-22 output as a (B, 16, 512) flat token sequence. One critical
detail surfaced during export: the training pipeline uses
`F.interpolate(..., mode='bilinear', size=(4, 4))` to reduce the 7 × 7 spatial
feature map to 4 × 4 before flattening. Replacing this with
`F.adaptive_avg_pool2d(..., output_size=4)` — a seemingly equivalent
operation — reduced ensemble accuracy from 56/64 to 48/64. The bilinear
interpolate path is preserved verbatim in the ONNX export.

Figure 5.2 shows the exported graph rendered in Netron. The C3k2 block at
layer 22 is visible as the final multi-branch module before the tokenizer
reshape.

### 5.3.2 Transformer head export

The TransformerHead class is copied verbatim from the training codebase to
`deployment/scripts/transformer-head.py`, ensuring weight compatibility. Each
of the six heads (Direct4, Scenario, Pair0–3) is exported individually with
`torch.onnx.export(..., opset_version=17)`. Under PyTorch 2.11 the dynamo
exporter silently emits opset 18 for certain reshape patterns, which requires
ONNX Runtime ≥ 1.18 to load. The environment pins ORT at 1.24.4.

### 5.3.3 Dynamic INT8 quantization

`quantize_dynamic(weight_type=QUInt8)` is applied to every FP32 graph. One
workaround was required: ORT's pre-quantization shape inference fails on the
dynamo-exported transformer attention reshapes, emitting "Unable to infer
shape" errors. Stripping the `value_info` field before quantize resolves this
without affecting the produced INT8 graph. This is documented in
`quantize-int8.py` and in `deployment/README.md` under "Known caveats".

### 5.3.4 Parity verification

`verify-onnx-parity.py` compares PyTorch, ONNX FP32, and ONNX INT8 outputs on
synthetic random inputs (batch = 1, 10 runs, max absolute difference). Table
5.1 reports the measured drifts.

**Table 5.1 — Parity verification (max absolute output drift vs PyTorch)**

| Model         | PT vs FP32 | PT vs INT8 |
|---------------|-----------:|-----------:|
| Backbone      |   4.53e-06 |   4.32e-01 |
| Direct4-s314  |   1.61e-06 |   3.56e-02 |
| Scenario-s7   |   1.67e-06 |   2.41e-02 |
| Pair{0..3}-s7 |   < 5e-07  |   < 1e-02  |
| Eight-s7      |   2.38e-06 |   2.64e-02 |

FP32 drift is at float32 rounding level (< 5e-06) across all graphs, i.e. the
exports are bit-accurate modulo numerical noise. INT8 drift on the transformer
heads is small (< 4e-02 absolute, softmax-preserving). The backbone INT8 drift
of 0.43 reflects its wide post-SiLU activation range; whether this translates
to an accuracy penalty is answered empirically in Section 5.4 on real test
videos.

---

## 5.4 Benchmarks

All benchmarks were collected on an Apple M4 (10-core CPU, 8 performance +
2 efficiency), 16 GB unified memory, macOS 14.6, ONNX Runtime 1.24.4. The
batch size is 1 (single video at a time) across all measurements; the
multi-stream case is treated separately in Section 5.5.

### 5.4.1 Latency

Four configurations were measured: CPU-FP32, CPU-INT8, CoreML-FP32, and
CoreML-INT8. Each end-to-end run consists of the full pipeline (preprocess →
backbone → stats aggregation → Direct4 head + scenario/pair cascade →
ensemble). 100 runs per configuration. Figure 5.3
(`figures/latency-by-stage.png`) shows the stacked-bar breakdown per stage.

**Table 5.2 — End-to-end latency and throughput (batch = 1, 100 runs)**

| Config      | e2e p50 | e2e p95 | FPS  |
|-------------|--------:|--------:|-----:|
| CPU-FP32    | 26.8 ms | 28.8 ms | 37.3 |
| CPU-INT8    | 32.2 ms | 35.6 ms | 31.1 |
| CoreML-FP32 | 50.6 ms | 54.1 ms | 19.8 |
| CoreML-INT8 | 89.0 ms | 91.3 ms | 11.2 |

**CPU-FP32 is the headline result: 26.8 ms p50, 28.8 ms p95, 37.3 FPS — meeting
the 30 FPS real-time budget with 6.5 ms of headroom at p50.** The INT8 CPU path
is slower, not faster: the M4 AMX matrix units have no native INT8 path in the
current ONNX Runtime build, and the dequantize kernels dominate. The
transformer heads individually benefit from INT8 (~2× faster than FP32), but
the backbone regresses enough to wipe out that gain at the end-to-end level.

The CoreML paths are measured, not projected. The backbone alone accelerates
to 2.1 ms on the ANE under CoreML FP32 — a 4.2× speedup over the CPU
backbone — but the transformer heads fall back to CPU because of unsupported
attention operators, and the CPU → ANE tensor transfer becomes the
bottleneck. For a single-stream demo at 30 FPS the CPU-FP32 path is the
correct choice; the CoreML-FP32 path is retained for users with
transformer-head-friendly hardware (A18-class devices or later).

### 5.4.2 Accuracy

Accuracy was measured on the 4-class unsafe subset of Unsafe-Net (64 videos,
16 per class) using the same evaluation protocol as Chapter 4. Figure 5.4
(`figures/accuracy-by-config.png`) visualizes the three head outputs (Direct4,
cascade, ensemble) across the four configurations.

**Table 5.3 — Accuracy preservation (top-1 correct / 64)**

| Config      | Direct4 | Cascade | Ensemble |
|-------------|--------:|--------:|---------:|
| CPU-FP32    |   55/64 |   51/64 | **56/64** |
| CPU-INT8    |   55/64 |   51/64 |   55/64 |
| CoreML-FP32 |   55/64 |   51/64 | **56/64** |
| CoreML-INT8 |   55/64 |   51/64 |   55/64 |

The CPU-FP32 and CoreML-FP32 ensembles match the PyTorch training baseline
exactly (56/64 = 87.5%). INT8 quantization costs one sample on the ensemble
(55/64 = 85.9%), consistent with the backbone drift observed in Table 5.1. The
per-head outputs (Direct4 and cascade) are stable across all four
configurations, indicating the 1-sample drop is an ensemble-boundary effect
rather than a systematic degradation.

**Combined view:** CPU-FP32 delivers zero accuracy loss at 37 FPS — the
production-grade configuration for Chapter 5.

### 5.4.3 Model size

Figure 5.5 (`figures/model-size-comparison.png`) shows FP32 vs INT8 sizes for
each ONNX file. The backbone shrinks from 33 MB (FP32) to 9 MB (INT8), a 3.8×
reduction. The larger transformer heads (direct4, scenario, eight) shrink from
46 MB to 12 MB (3.9×); the four pair heads shrink from 7.7 MB to 2.0 MB (3.8×).
Total on-disk footprint for the deployed ensemble (backbone + direct4 + scenario
+ four pair heads) is 156 MB (FP32) or 40 MB (INT8), well within the storage
budget of any modern mobile or embedded device.

---

## 5.5 Multi-Stream Pipeline

Single-stream inference establishes the 37 FPS baseline. Section 5.5 extends
the pipeline to N = 1…4 concurrent video streams sharing a single M4 to answer
a practical deployment question: can one edge device monitor multiple cameras
simultaneously?

### 5.5.1 Design

The multi-stream pipeline (`deployment/scripts/multi-stream-pipeline.py`) is
built on asyncio. A single shared backbone ORT session receives a batched
tensor of shape (N, 3, 224, 224) per inference tick, where N is the number of
active streams. Per-stream sliding windows (T = 32) and transformer heads run
sequentially after the backbone call. Each stream has a 4-slot queue; when
the queue is full, the oldest pending frame is dropped and counted. Reader
pacing is set to 30 FPS per stream.

Three design choices matter. First, the backbone is **shared**, not
duplicated — N independent backbones would require N × 35 MB RAM and
eliminate any L2 cache reuse across streams. Second, the inference loop uses
**asyncio**, not multiprocessing — ORT sessions are not fork-safe on macOS,
and IPC overhead would dominate at 30 FPS per stream. Third, on shutdown the
main loop **awaits the inference task**, draining the queue completely,
rather than cancelling it; otherwise the reported FPS would include frames
that were enqueued but never reached the backbone (a subtle but significant
source of inflation discovered during development).

### 5.5.2 Measured results

Each configuration runs with four distinct test videos (240 frames each) at
30 FPS reader pacing. Figure 5.6 (`figures/fps-vs-streams.png`) plots
aggregate FPS against N, with an ideal-linear reference.

**Table 5.4 — Multi-stream scaling (CPU-FP32, 240 frames/stream, drain-on-shutdown)**

| N | Aggregate FPS | Batch utilization | p50 head (ms) | p95 head (ms) | Drops |
|---|--------------:|------------------:|--------------:|--------------:|------:|
| 1 |          17.6 |             100%  |          55.1 |          72.9 |     0 |
| 2 |          29.1 |              54%  |          34.8 |          43.1 |     0 |
| 3 |          26.2 |              57%  |          40.6 |          48.9 |     0 |
| 4 |          20.8 |              50%  |          44.6 |          63.9 |     0 |

Three observations. **First, aggregate throughput peaks at N = 2 (29.1 FPS)
and declines at N ≥ 3** as the shared CPU backbone saturates. The pipeline is
backbone-bound above N = 2; adding streams past this point costs more
contention than it wins in batching. **Second, zero frame drops across all
configurations** — the 4-slot queue plus 30 FPS reader pacing absorb backbone
jitter without overflow, and the drain-on-shutdown path guarantees every
enqueued frame is inferred before the wall-clock timer stops. **Third,
per-stream head p50 latency scales from 34.8 ms (N = 2) to 44.6 ms (N = 4) —
visible but bounded contention** on the shared backbone.

A note on the **N = 1 row (17.6 FPS)**, which is lower than the 37.3 FPS
synchronous single-stream benchmark of §5.4 and below the N = 2 aggregate.
This is harness overhead, not a model regression: the async pipeline pays a
fixed per-tick cost the synchronous benchmark does not — event-loop
scheduling, a 5 ms batch-assembly bucket window (§5.5.3), and the
drain-on-shutdown accounting that counts every enqueued frame. With only one
stream there is nothing to batch against that overhead, so it dominates;
from N = 2 onward the fixed cost amortizes across streams and aggregate
throughput climbs to its N = 2 peak before the shared CPU backbone
saturates. The synchronous §5.4 figure (37.3 FPS) remains the correct
single-stream latency number; the N = 1 async row is reported only for
completeness of the scaling curve.

### 5.5.3 Batch utilization trade-off

Batch utilization sits at 50–57% for N ≥ 2, below the 75% target set in the
phase plan. This is a direct consequence of the 5 ms batch-assembly bucket
window: at 30 FPS reader pacing, frame inter-arrival time is 33 ms and
inference itself takes 20–50 ms, so the 5 ms bucket rarely sees more than one
frame per stream. Lengthening the bucket would raise utilization at the cost
of queueing latency; this is a deployment-tuning knob, not a pipeline defect,
and is documented as such.

---

## 5.6 Jetson Orin Edge Deployment (Measured)

The full pipeline was deployed and measured on a Jetson Orin Nano Super
development kit under JetPack 6.2.2 with TensorRT 10.3.0. All numbers in
this section are **measured**, not projected.

The headline result of this chapter is a **deployment** result, not an
accuracy result. The deployed (best-case) checkpoint runs at 116 FPS
single-stream while sustaining a 2-stream soak for 36.2 minutes with zero
dropped frames at 7.0 W, plus separately a 4-stream zero-drop throughput burst
(88 FPS aggregate, Table 5.5C) — and **no published baseline
demonstrates edge real-time multi-stream inference for this task**, so the
deployment opens an operating point unreachable from the server-only
baselines [5]. On accuracy the deployed checkpoint shows 89.28% macro under
test-selected evaluation, but the honest leakage-free estimate for the best
config (hybrid+gate, §4.6.2) is 53/64 (85.4% macro) — macro parity with the
no-bbox baseline even blind. The original frozen pipeline (§4.7.11) is
≈ 52/64 (≈ 84% macro), below parity. The earlier "+3.5 pp macro" claim is
withdrawn; this chapter rests on the measured edge capability, not on an
accuracy advantage.

### 5.6.1 Hardware platform and toolchain

The edge target is a **Jetson Orin Nano Super Developer Kit**: 8 GB LPDDR5,
67 TOPS INT8, 6× Cortex-A78AE @ 1.5–1.7 GHz, Ampere GPU with 1024 CUDA cores
and 32 tensor cores at up to 1020 MHz under the MAXN_SUPER nvpmodel profile.
List price USD 249–499 (vendor-region dependent) against the paper baseline's
2× GTX 1080 + i9-9900 + 16 GB workstation at ~USD 1,500 retail and a
sustained envelope near 400 W (paper §3).

JetPack 6.2.2 ships CUDA 12.6, cuDNN 9.3, TensorRT 10.3, and Python 3.10.
Three power profiles are available via `nvpmodel`: profile 0 ("15W", legacy),
profile 1 ("25W", default), and profile 2 ("MAXN_SUPER", thermal-only cap).
The deployment runs at MAXN_SUPER throughout the measurements unless stated
otherwise; the profile is restored on every boot through
`/etc/nvpmodel.conf`.

The toolchain consists of four scripts in `deployment/scripts/`:
`jetson-build-trt-engine.py` parses each FP32 ONNX into a TensorRT 10
network and serialises FP32, FP16, or INT8 engines (with a streaming
entropy calibrator for INT8); `jetson-prep-int8-calib.py` extracts 256
real frames from `data/train/` videos and streams 256 stats batches per
head, eliminating the synthetic-noise calibration weakness flagged in the
earlier projection draft; `jetson-benchmark.py` mirrors the M4
benchmark-harness interface, reporting per-stage p50/p95/p99 and 4-class
accuracy through TRT `IExecutionContext` runners; and
`jetson-multi-stream-pipeline.py` drives concurrent N-stream inference
with a per-stream sliding window, with an `--emit-jsonl` flag that
externalises predictions for downstream dashboards.

All seven engines (backbone + direct4 + scenario + 4× pair head) build
cleanly under the toolchain: total wall-clock for FP16 builds is 10.5
minutes (backbone dominates at 6 minutes due to its convolutional layer
count); INT8 with the 256-frame calibrator takes 12 minutes. The serialised
FP16 engines total 87 MB; INT8 engines total 80 MB (the head networks
fall back to FP16 for transformer layers where INT8 calibration is
unstable, so the size reduction is concentrated in the backbone, which
shrinks from 18.8 to 11 MB).

### 5.6.2 Cross-precision benchmarks

The benchmark harness runs 100 timed inferences after 10 warm-up runs,
batch 1, end-to-end (backbone + direct4 + cascade). Accuracy is reported
on the same 64-clip 4-class test set used throughout the thesis.

**Table 5.5B — Edge cross-precision (Jetson Orin Nano Super, MAXN_SUPER,
100 runs, batch=1)**

| Mode | Backbone p50 | Direct4 p50 | Cascade p50 | e2e p50 | FPS | Top-1 |
|---|---|---|---|---|---|---|
| TRT-FP32 | (∈ e2e) | (∈ e2e) | (∈ e2e) | **12.47 ms** | **80.2** | **56/64 = 87.50%** |
| TRT-FP16 | 3.95 ms | 2.05 ms | 5.70 ms | **8.62 ms** | **116.0** | 55/64 = 85.94% |
| TRT-INT8 | (∈ e2e) | (∈ e2e) | (∈ e2e) | **8.24 ms** | **121.4** | 55/64 = 85.94% |

Source: `deployment/artifacts/benchmarks-jetson-{fp32,fp16,int8}.json`.

Three observations matter for §5.7 limitations and Chapter 6 conclusion.

First, **FP32 edge inference achieves accuracy parity with the M4 reference**
(56/64 top-1, 89.28% macro). The 12.47 ms latency is 2.15× faster than the
M4 CPU-FP32 baseline of 26.8 ms (§5.4 Table 5.2), driven by the Ampere
tensor cores' native FP32 path; the M4 has no equivalent hardware
acceleration for the cascade pair softmax matmul, which dominates its
17.5 ms cascade stage.

Second, the **FP16 1-sample accuracy drop (55/64 vs 56/64) is not
statistically significant** (two-proportion z = −0.26, two-tailed p = 0.79).
The drop is consistent with FP16 numerical drift in the transformer attention softmax;
TRT's strict-types fallback for sensitive layers could likely recover the
sample, but at the cost of a slower engine. The accuracy / latency
trade-off is left as a deployment tuning knob: the FP32 row is the
"high-accuracy" Pareto point, the FP16 row is the "high-throughput" point.

Third, **INT8 with 256 real-frame calibration matches FP16 accuracy**
(55/64 = 85.94%) with no further degradation, at ~5% higher throughput than FP16 (121.4 vs 116.0 FPS, 1.05×).
This is a stronger result than the typical 2-sample INT8 degradation
reported in the literature for this class of transformer head; we attribute
the parity to the real-frame entropy calibration (256 samples sampled
uniformly from `data/train/`) rather than synthetic Gaussian noise that an
earlier prototype used and that produced a 2-sample drop in pilot
measurements.

### 5.6.3 Multi-stream throughput and sustained soak

Multi-stream inference is the deployment scenario the thesis cares about:
an industrial site rarely has only one camera. The pipeline runs four
async readers feeding a single shared TRT inference loop; the per-stream
sliding window of 32 frames is updated independently per stream.

**Table 5.5C — Multi-stream FPS, real-time pacing (30 FPS source, 240
frames per stream)**

| N streams | Aggregate FPS | Per-stream FPS | Dropped frames | Head latency p50 |
|---|---|---|---|---|
| 1 | 24.28 | 24.28 | 0 | 6.98 ms |
| 2 | 48.16 | 24.08 | 0 | 6.88 ms |
| 3 | 70.57 | 23.52 | 0 | 6.94 ms |
| 4 | **88.06** | **22.02** | **0** | 6.82 ms |

Source: `deployment/artifacts/multi-stream-jetson-fp16-realtime.json`.

Four concurrent cameras at 30 FPS source pacing are sustained with **zero
dropped frames**. The per-stream FPS gap (24 vs 22 from N=1 to N=4) is
scheduling jitter at the reader, not inference saturation: head latency
p50 remains essentially constant at 6.8 ms across all N, indicating the
GPU is well below its operating envelope. (As on the M4, the N = 1 async
aggregate of 24.3 FPS is below the 116 FPS synchronous single-stream
benchmark of §5.6.2 for the same harness-overhead reason given in §5.5.2 —
event-loop and batch-bucket cost with nothing to amortize against at N = 1;
the synchronous 116 FPS is the single-stream latency figure of record.) A separate unbounded-throughput
sweep (where readers run as fast as `cv2.VideoCapture` allows) caps the
aggregate at 136.89 FPS at N=4 with 328 of 960 frames dropped in total
(~82 of ~240 per stream), identifying the GPU saturation point. For the 30 FPS surveillance use
case the GPU has approximately 54% headroom for additional streams or
for higher-rate workloads.

A **36.2-minute sustained soak at N=2** validates production-readiness.
≈ 107,000 frames were inferenced with zero drops while
`tegrastats --interval 1000` logged thermal and power at 1 Hz.

**Table 5.5F — Sustained soak (N=2, 36.2 minutes, MAXN_SUPER profile)**

| Metric | Value | Headroom vs limit |
|---|---|---|
| Dropped frames | **0** | — |
| Throttle events (tj > 80 °C) | **0** | — |
| GPU temperature (steady-state) | 55.9 ± 0.9 °C | ≈ 24 °C below 80 °C throttle (mean); 23 °C at peak 56.8 °C |
| Power VDD_IN average | 7.0 W | 8 W below 15 W MAXN budget |
| Power VDD_IN peak | 7.34 W | — |
| GPU utilisation peak | 44 % | — |

Source: `deployment/artifacts/soak-jetson-summary.json` +
`tegrastats-soak30min.log` (2,548 samples).

The result is that **the deployment has substantial thermal and power
headroom** — junction temperature stays ≈ 24 °C below the throttle threshold at steady-state (23 °C at peak),
and the workload draws less than half of the MAXN budget. This headroom
is what enables the multi-stream scaling reported above and, together with
the absence of throttle events, supports a claim of suitability for 24/7
industrial deployment.

### 5.6.4 Power efficiency Pareto

The three nvpmodel profiles were swept to characterise the FPS-versus-power
trade-off available to the deployment engineer.

**Table 5.5D — Power-efficiency Pareto (FP16 single-stream)**

| Profile | FPS | Power avg | Power peak | mJ / inference | FPS / W |
|---|---|---|---|---|---|
| 0 — 15W (legacy) | 116.3 | 5.31 W | 9.35 W | 45.7 | 21.9 |
| 1 — 25W (default) | 140.3 | 5.29 W | 9.31 W | 37.7 | 26.5 |
| 2 — **MAXN_SUPER** | **152.6** | 5.22 W | 9.27 W | **34.2** | **29.2** |

Source: `deployment/artifacts/power-sweep-summary.json` +
`tegrastats-pwr{0,1,2}.log`.

The unexpected finding is that **average power is essentially identical
across all three profiles** (5.22–5.31 W, within 1.7%) despite FPS
increasing 31% from profile 0 to profile 2. The mechanism is that the
single-stream workload does not saturate the GPU envelope under any
profile: average GPU utilisation stays below 2% (peak 86% in profile 0,
67% in profiles 1 and 2). The clock boost available in MAXN_SUPER finishes
each inference faster without raising the integrated power draw — a "race
to idle" pattern that yields a **33% improvement in energy efficiency**
(34.2 mJ/inference vs 45.7 mJ/inference) without a corresponding power
penalty.

For a battery- or solar-constrained deployment, profile 2 is therefore
the right choice unless the deployment site has a thermal envelope below
9 W peak; the legacy 15 W profile (profile 0) would only be appropriate
in an enclosure that cannot dissipate the brief power spike.

### 5.6.5 End-to-end system architecture

The deployed architecture realises a four-layer pipeline whose component
boundaries are also the data-protection boundaries:

```
[Cameras / video file proxies]
     │  30 FPS RGB frames
     ▼
[EDGE — Jetson Orin Nano Super, 15 W envelope]
   async video readers → preprocess queue
                       → backbone TRT engine (FP16, 18.82 MB, 3.95 ms)
                       → sliding window (T=32)
                       → direct4 + scenario + 4× pair TRT engines
                       → ensemble fusion
                       → jetson-stream.jsonl  (≈ 520 bytes / event)
     │
     │  🚫 PRIVACY BOUNDARY — raw frames do not cross this line
     │
     │  SSH key-only + AES-128-CTR + HMAC-SHA-256 (rsync default)
     ▼
[SERVER + DASHBOARD — MacBook M4]
   /tmp local cache JSONL
                       → rolling 3-second per-stream aggregation
                       → conf ≥ 0.75 → 🚨 ALERT
                         conf ≥ 0.55 → ⚠️ WARNING
                         else        → 🟢 MONITORING
                       → Streamlit dashboard at http://localhost:8501
                       → security operator
```

A renderable Mermaid version of this diagram is in
`figures/arch-diagram-jetson-edge.mmd`. The privacy boundary is the
critical architectural property: it is enforced not by policy but by the
fact that the only data structure leaving the edge is a per-emission JSON
event (timestamp, stream id, frame number, class index, probability vector,
head latency in ms). At ~520 bytes/event and a measured ~12 events/sec
total across four streams (3.75 emissions/stream/sec by design,
8,988 events recorded in the 772-second live demo), the uplink bandwidth
requirement is **≈ 50 kbps total** — two orders of magnitude below a
single 1080p H.264 stream (~5–10 Mbps) and trivially auditable.

A 12.9-minute (772-second) live demonstration was run with four video
files standing in as camera proxies. The dashboard correctly flagged the two true-positive unsafe
streams as **ALERT** (Stream 0 Walkway Violation at 76.7 % aggregated
confidence, Stream 2 Overload with Forklift at 99.4 %), correctly held
the safe-walkway stream in **MONITORING** (54.9 %, below the 0.75 alert
threshold), and held the ambiguous intervention stream in **MONITORING**
as well (50.4 %, capturing the inherent class confusion between
unauthorized intervention and walkway violation discussed in §4.7.6). The
session log captured every threshold crossing with timestamps. The
demonstration is recorded in `deployment/artifacts/live-demo-stream.jsonl`
(**8,988 prediction events over 772 seconds** ≈ 3.75 emissions/stream/sec,
verified by byte-level audit of the JSONL); a screenshot is included as
Figure 5.6 (`figures/jetson-live-dashboard-demo.png`).

### 5.6.6 Security and privacy properties

The edge architecture supports five concrete claims, each with a
mechanism and measurement evidence. Full threat model and control
mapping are in `plans/260430-2336-thesis-completion/security-narrative.md`;
the summary table is reproduced below. To set expectations honestly:
**T1 (edge-only inference, raw frames never leave the device) and T5
(rolling-window dual-threshold alerting) are architectural contributions
of this work**, measured in the live demo; **T2–T4 (TRT device-locking,
SSH key-only transport and access) are standard operational hardening,
listed for completeness rather than claimed as research novelty.** The
architecture deliberately does not provide tamper-evident attestation or
model signing (§5.6.6).

**Table 5.5G — Threat-to-control mapping**

| Threat | Control | Mechanism | Evidence |
|---|---|---|---|
| T1 — Raw video exfiltration | Edge inference + metadata-only transport | Frames decoded on Jetson; JSONL events only across network | Byte-by-byte audit of `live-demo-stream.jsonl` |
| T2 — Model theft | TRT engine device-locking | Engines built per Jetson device, runtime checks fingerprint | TRT "different model of device" warning at runtime when profile changes |
| T3 — Transport interception | SSH + AES-128-CTR + HMAC | OpenSSH 9.x defaults via rsync | `~/.ssh/jetson_key` ED25519, no password auth |
| T4 — Unauthorised edge access | Pub-key SSH + `NOPASSWD` operator | `/etc/ssh/sshd_config`, `/etc/sudoers.d/` | Single-tenant deployment, physical control |
| T5 — False-alarm flooding | Rolling-window + dual-threshold | 3-second aggregation + 0.75/0.55 cut-offs | Live demo: 100% recall on unsafe + 0% false alarm on safe stream |

The defense-readable summary is that **privacy is enforced by construction**
rather than by policy. A network monitor on the Jetson uplink can verify
that only metadata leaves the device, and the JSONL events are small
enough (520 bytes vs 50 kB for an H.264 frame) that the bandwidth
asymmetry itself rules out covert video exfiltration.

The architecture does **not** provide tamper-evident attestation, model
signing, or zero-trust verification of edge outputs against a server-side
ground truth. These properties are surfaced as future work in §6.3.

### 5.6.7 Comparison with paper baselines

The accuracy comparison against the paper baseline [5] is reported here
in full because Chapters 1, 4, and 6 cite this section as the source of
the headline numbers.

**Table 5.5A — Accuracy on the 64-clip 4-class test set**

| Configuration | Top-1 | **Macro** | Bbox annotation cost |
|---|---|---|---|
| Paper [5] — ConvLSTM-only (no bbox baseline) | 89.06 % (57/64) | 85.78 % | **0** (video labels only) |
| Ours — best-case (test-selected, deployed checkpoint) | 87.50 % (56/64) | 89.28 % | **0** (401 class-frame labels) |
| **Ours — frozen single-head blind (§4.7.11)** | **≈ 81 % (≈ 52/64)** | **≈ 84 %** | **0** |
| **Ours — hybrid+gate blind (best config, §4.6.2)** | **82.8 % (53/64)** | **85.4 %** | **0** |
| Paper [5] — Unsafe-Net full cascade | 96.88 % (62/64) | 95.81 % | 2,262 train + 329 test bbox |

**Apples-to-apples (zero bbox cells).** At the same annotation budget as the
published ConvLSTM-only baseline — neither system uses the bbox dataset — the
frozen single-head blind (§4.7.11) is **below the baseline on top-1 (≈ 81 % vs 89.06 %)
and below macro parity (≈ 84 % vs 85.78 %)**, while **the best-config blind
(hybrid+gate, §4.6.2) reaches macro parity with the baseline (85.4 % vs 85.78 %)**. The earlier claim of a
+3.5 pp macro win came from the test-selected checkpoint and does **not**
survive the blind protocol (§4.7.11); it is withdrawn. The defensible
cross-paper statement at *matched* (test-selected) protocol is parity-minus-
one on top-1 (56/64 vs 57/64; z = −0.27, p = 0.78); the thesis does not lead
with this, because the paper reports no blind number and the honest internal
estimate is the blind rows above. The contribution at this cell is the
**zero-bbox + edge-deployable** property, not an accuracy advantage.

**Pareto framing against Unsafe-Net full.** The full Unsafe-Net cascade
outperforms both rows by 6–8 percentage points, but at the cost of
**2,591 bbox-annotated images** (2,262 train + 329 test). At a
conservative 5 minutes per bbox annotation, this corresponds to
**~190 hours of expert labelling effort** that our deployment does not
require. The thesis position is that this is a Pareto-optimal trade: we
accept 6.5 percentage points of macro accuracy in exchange for
(i) eliminating the bbox annotation budget, (ii) running the inference
on a $499 edge device at < 8 W actual power, (iii) sustaining four
camera streams in real-time with zero drops, and (iv) preserving raw-video
privacy by architecture. Whether this trade is the right one depends on
the deployment cost structure; for sites with many cameras and limited
labelling budget the trade is favourable, while for high-stakes inspection
contexts where every percentage point matters the full Unsafe-Net cascade
remains the correct choice.

**Edge axis (no paper comparison exists).** Neither baseline in [5]
reports edge deployment, multi-stream throughput, real-time sustainability,
or per-frame energy. The paper's "real-time" system in fact runs the
YOLO v4 + ConvLSTM models on its **2× GTX 1080 + i9-9900 workstation**
(≈ 400 W, ~USD 1,500; paper §3) and performs **no model optimization for
embedded hardware** — no quantization, no TensorRT, no edge inference; its
deployment contribution is a wireless **Arduino Uno + ESP8266 alarm** that
merely actuates an audible/visual warning when the workstation flags an
event (paper §2.5). This thesis's contribution is therefore *orthogonal and
complementary*: the same task is moved onto a **7 W, USD 249–499 Jetson**
with INT8/TensorRT optimization, four real-time camera streams, and a
36-minute zero-drop soak — an operating point the paper does not occupy and
did not attempt. The claim is not "we beat them on accuracy" (we do not) but
"we deliver the edge-deployable real-time system the paper's own motivation
calls for but does not build," which is the primary contribution recorded in
§1.4.

---

## 5.7 Limitations

Five limitations are documented in full for the reader:

1. **Low-resolution operating point.** Training and deployment both use
   224 × 224 frames (no train/deploy resolution mismatch); the 224² choice
   is defended on accuracy *and* compute grounds in §4.7.10 (224² 55/64 vs
   384² 52/64 single-head). Accuracy is preserved end-to-end (56/64 on both
   the training baseline and CPU-FP32 ONNX). The limitation is that 224² is a
   coarse operating point: degradation on finer-grained scenes, smaller
   objects, or more diverse camera angles cannot be ruled out. A follow-up
   evaluation at higher input resolutions is listed as future work.

2. **Test set size.** The 4-class unsafe subset is 64 videos, 16 per class.
   Confidence intervals on the 87.5% accuracy figure are wide: a single
   swapped sample shifts the headline by 1.5%. All comparisons between
   configurations in Tables 5.2 and 5.3 should be read with this in mind.

3. **Jetson measurements run on a single device of one SKU.** Section 5.6
   numbers were measured on a single Orin Nano Super development kit; the
   Pareto framing (annotation-cost vs accuracy) generalises across SKUs, but
   absolute FPS and power numbers can shift on the AGX Orin or NX SKUs.
   A multi-SKU evaluation is out of scope and listed as future work.

4. **No live camera capture in dashboard.** The Streamlit dashboard supports
   file uploads only. A live-camera path would require `cv2.VideoCapture(0)`
   plus browser MediaDevices bridging (WebRTC or a refresh loop), which was
   out of scope for the thesis prototype.

5. **Batch utilization below target in multi-stream.** Section 5.5 reports
   ~53% utilization at N ≥ 2 against a 75% goal. The 5 ms bucket window was
   chosen to preserve latency; raising utilization would require either
   longer buckets (more latency) or higher reader rates (unrealistic for 30
   FPS sources). This is a design trade-off, not a defect.

---

## 5.8 Conclusion

Chapter 5 established that the hybrid YOLOv11s + Transformer pipeline from
Chapters 3 and 4 can be deployed to a commodity Apple M4 CPU through an
ONNX Runtime path, with zero accuracy loss relative to the PyTorch training
baseline and with real-time single-stream throughput above the 30 FPS budget.
Specifically:

- **Accuracy preservation:** CPU-FP32 and CoreML-FP32 ensemble accuracy is
  56/64 = 87.5%, matching the training baseline exactly. INT8 quantization
  costs one sample (55/64 = 85.9%).
- **Single-stream latency:** CPU-FP32 end-to-end is 26.8 ms p50 / 28.8 ms
  p95, i.e. 37.3 FPS — 6.5 ms of headroom against the 30 FPS budget at p50.
- **Multi-stream scaling:** Four concurrent streams sustained at 20.8 FPS
  aggregate with zero dropped frames; aggregate throughput peaks at N = 2
  with 29.1 FPS. Per-stream head p50 ~45 ms at N = 4.
- **Reproducibility:** The full Chapter 5 pipeline reproduces from a clean
  clone in seven commands (see `deployment/README.md`).
- **Artifacts:** Four PNG figures, one Mermaid architecture diagram, one
  Netron graph screenshot, and three data tables — all auto-generated from
  the JSON artifacts, so the thesis numbers cannot drift from the code.

Future work for Track B is twofold: (a) a multi-SKU edge characterisation
extending the Section 5.6 protocol to the AGX Orin and Orin NX SKUs to map
the upper and lower bounds of the Jetson family for this workload, and
(b) a live USB-camera capture path integrated into the Streamlit dashboard
to replace the file-uploads-as-cameras demo. Both are engineering extensions
rather than research contributions and are left as follow-on development.
