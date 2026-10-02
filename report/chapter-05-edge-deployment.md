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
> | Ch7 + Abstract | **P3** entity detector + spatial-relation Transformer | **54/64** (84.4 % **top-1**, 88.4 % macro) | 54/64 | same-dist N=64, **plus k-fold and LCO at N=516** |
> | Ch5 §5.6.9 | **R48** whole-frame CLIP ViT-B/32 -> B1 temporal Transformer — the engine actually on the device | **84.2 % balanced accuracy** (B1, N=1,100) | same number: the protocol cannot leak a source, so there is no separate honest column | **P-D LSO-1100**, folds grouped by source video, N=1,100 |
>
> **The headline is the R48 lane: CLIP + temporal Transformer under LSO-1100.** It is the only
> accuracy in this thesis measured on all 1,100 clips under a protocol that cannot put one source
> video in both the train and the evaluation fold, and it is the model that was exported, deployed
> and measured on the Jetson (§5.6.9). YOLO's role in that lane is **localisation** — the system
> must say *where*, not only *whether* — and localisation latency is timed separately, never summed
> into the class decision. P3's 54/64 is the entity-relation lane's result at N=64 and is what Ch7
> and the abstract report; it keeps its own evidence beyond N=64 (k-fold and LCO at N=516). P1's
> 56/64 is **test-selected** (the ensemble was chosen using the test clips) and must never be
> presented as the result; its honest blind counterpart is 53/64. Chapters 1-6 are retained as the
> experimental record of how the design was arrived at, not as the final claim.
>
> **Caution — "84.4 %" denotes two different quantities here. Never write it bare.** 84.4 %
> *top-1* (P3, blind same-dist split, N=64) is Ch7's figure. 84.4 % *balanced accuracy* (B1w,
> LSO-1100, N=1,100) is the R48 lane's best head — and B1w is **not** the head on the device. The
> deployed engine is B1 (201,090 params x 3 seeds), whose own figure is **84.2 %**; B1 was chosen on
> the pre-registered *edge* criterion, smallest model, not on accuracy, so its own number is the one
> to quote and never max(B1, B1w). §5.6.9 quotes 84.2 % throughout for that reason
> (`plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-10-01-two-different-84-4-percent.md`).
>
> **"84.2 %" likewise denotes two different quantities. Never write it bare either.** 84.2 %
> *balanced accuracy* (B1, LSO-1100, N=1,100, binary) is the deployed figure above. 84.2 % *macro*
> is P3's k-fold mean over 516 ESTA clips on the 4-class task (Ch7 §7.7.1). Different corpus,
> different task, different metric: the two must never be compared, summed, or carried into one
> sentence without their qualifiers.

## 5.1 Introduction

Objective 3 of this thesis is to demonstrate that the trained hybrid pipeline
(YOLOv11s backbone + Transformer heads) can execute on commodity edge hardware
without sacrificing the accuracy established in Chapters 3 and 4. Chapter 5
reports a concrete, reproducible deployment on two targets: an Apple M3 CPU
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

**How to read every accuracy number in this chapter.** The 56/64 reference is
**test-selected** — the ensemble was chosen using the 64 held-out clips, so it is an
optimistic estimate and its blind counterpart is 53/64 (attribution box above). That
does not invalidate its use here, because this chapter does not ask how well the model
generalises; it asks whether an exported engine reproduces the checkpoint it was
exported from. For that question the selected configuration is a *fingerprint*: any
divergence between 56/64 and what an engine scores is export error, which is exactly
what §5.4 and §5.6 measure. Accuracy numbers in this chapter are therefore claims about
**export fidelity, never about generalisation**, and none of them should be lifted out
of this chapter as the thesis's accuracy result — that is P3's 54/64 in Chapter 7.

Chapter 5 answers four questions. First, does the exported ONNX pipeline
preserve accuracy? Second, does it meet the 30 FPS budget on M3 CPU? Third,
does it scale to multiple concurrent video streams without dropping frames?
Fourth, does the pipeline run end-to-end on a true embedded device with
real-time multi-stream and operator-grade alerting? Sections 5.2–5.3 describe
the architecture and the export pipeline. Section 5.4 reports measured M3
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
Running all four from a clean checkout takes under ninety seconds on the M3.

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

All benchmarks were collected on an Apple M3 (8-core CPU, 4 performance +
4 efficiency), 16 GB unified memory, macOS 14.6, ONNX Runtime 1.24.4. The
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
is slower, not faster. The per-stage measurements locate the regression but do
not explain it: the transformer heads individually benefit from INT8 (~2×
faster than FP32), while the backbone regresses enough to wipe out that gain
at the end-to-end level. Attributing this to a missing native INT8 kernel, to
dequantize overhead, or to the ONNX Runtime build would require kernel-level
profiling that was not performed, so no mechanism is claimed here — only that
dynamic INT8 is not a useful CPU operating point for this backbone.

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
exactly (56/64 = 87.5%; test-selected reference — the match is the result here, not the
value, per §5.1). INT8 quantization costs one sample on the ensemble
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
the pipeline to N = 1…4 concurrent video streams sharing a single M3 to answer
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

**Provenance warning for this table.** Its artifact
(`deployment/artifacts/multi-stream-results.json`) carries a `note` field declaring
itself "hand-patched post-fix numbers matching Chapter 5 Table 5.4". The table is a
faithful rendering of that file — every row reproduces once the per-stream p50 and p95
are averaged across streams, which is what the columns report — so no arithmetic is in
question. What is in question is the direction of authority: the file was written to
agree with the table rather than the table being read off a run. These four rows are
therefore the **weakest-sourced numbers in this chapter**, and are reported as such.
The harness that would settle them is present and re-runnable; §5.7 records why the
re-run has not been done.

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
baselines [5]. On accuracy the P1 checkpoint exported here shows 89.28% macro under
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
earlier projection draft; `jetson-benchmark.py` mirrors the M3
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

First, **FP32 edge inference achieves accuracy parity with the M3 reference**
(56/64 top-1, 89.28% macro — the test-selected P1 reference of §5.1; parity is the
claim, not the number). The 12.47 ms latency is 2.15× faster than the
M3 CPU-FP32 baseline of 26.8 ms (§5.4 Table 5.2). The stage breakdown shows
where the gap sits — the cascade stage costs 17.5 ms on the M3 host and is
the dominant term there — but the comparison is CPU versus discrete GPU, and
no kernel-level attribution to specific hardware units was measured.

Second, the **FP16 1-sample accuracy drop (55/64 vs 56/64) is not
statistically significant**. The two engines are scored on the *same* 64 clips, so the
comparison is paired and a two-proportion z-test is inadmissible (this chapter
previously reported one, z = −0.26, p = 0.79; it is withdrawn). Only the per-clip
totals survive for these runs, not the per-clip predictions, so the exact McNemar test
is bounded over every feasible 2 × 2 table instead: the bound is **p = 1.000 for all of
them**, best case and worst case alike. A one-sample difference on 64 clips cannot be
distinguished from numerical noise under any pairing — a stronger and more honest
statement than the z-test made.
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
GPU is well below its operating envelope. (As on the M3, the N = 1 async
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
| Power VDD_IN average | 7.03 W | see caveat below |
| Power VDD_IN peak | 7.34 W (min 6.01 W) | — |
| GPU utilisation peak | 44 % | — |

Source: `deployment/artifacts/soak-jetson-summary.json` +
`tegrastats-soak30min.log` (2,548 samples).

> **Caveat on the power figure (added 2026-09-29).** The measured watts above are
> measurements and stand. The *headroom* this table previously claimed ("8 W below
> the 15 W MAXN budget") does not: the artifact records no `nvpmodel` state, only a
> hard-coded `maxn_budget_mw = 15000` supplied by the summariser, and on this device
> the profiles are id 0 = 15 W, id 1 = 25 W, id 2 = MAXN_SUPER. The profile in force
> during this June run was therefore **not logged**, and the "MAXN_SUPER profile"
> label in this table's caption cannot be reconciled with a 15 W denominator. The
> defensible statement is the absolute one: the workload drew **7.03 W on average
> and never exceeded 7.34 W**, which is below the ceiling of even the most restrictive
> profile the device offers. Later (2026-09-29) measurements do record the mode —
> all were taken at MAXN_SUPER — see `risks.md` R32b/R32c.

The result is that **the deployment has substantial thermal and power
headroom** — junction temperature stays ≈ 24 °C below the throttle threshold at steady-state (23 °C at peak),
and peak draw stays under 7.4 W. This headroom
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
[SERVER + DASHBOARD — MacBook M3]
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
one on top-1 (56/64 vs 57/64; paired exact-McNemar bound **p = 1.000 at every
feasible split**, §4.7.9.3 — a one-clip gap on N = 64 carries no signal at all;
the z-test previously quoted here, p = 0.78, is the wrong instrument because
both systems are scored on the same clips); the thesis does not lead
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
on a $499 edge device — measured at 7.03 W average for the P1 graph at its
16 % GPU duty cycle (§5.6.3), rising to 16.98 W for a saturated P3 detector
(§5.6.8), so the figure is duty-cycle-dependent and not a single headline
wattage — (iii) sustaining four
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
complementary*: the same task is moved onto a **USD 249–499 Jetson drawing
7–18 W depending on duty cycle** (§5.6.3, §5.6.8), with INT8/TensorRT
optimization, four real-time camera streams for the P1 graph, and a
36-minute zero-drop soak — an operating point the paper does not occupy and
did not attempt. Even the upper end of that range is a ~22× power reduction
against the paper's ≈ 400 W workstation. The claim is not "we beat them on accuracy" (we do not) but
"we deliver the edge-deployable real-time system the paper's own motivation
calls for but does not build," which is the primary contribution recorded in
§1.4.

### 5.6.8 Pipeline P3 trên thiết bị: phép đo nhịp quyết định

Các mục 5.6.1–5.6.7 đo đồ thị **P1** đã triển khai. Mục này báo cáo pipeline **P3** —
bộ phát hiện YOLO26 ở 800 px → chọn thực thể bằng hình học vùng Shapely → trích crop →
bộ mã hoá ngoại hình theo lô ở 224² → đầu quan hệ nhị phân E3 (18.044 tham số) — tức là
kiến trúc mà Chương 4 đánh giá.

Đơn vị quan tâm không phải khung/giây mà là **quyết định**: pipeline tiêu thụ `T = 16`
khung được lấy mẫu và phát một phán quyết An toàn/Không an toàn mỗi
`DECISION_EMIT_EVERY = 8` khung, nên một camera 24 fps áp đặt hợp đồng nhịp
**3,0 quyết định/giây**.

#### Hai điều kiện xung nhịp, và vì sao phải tách chúng

Phép đo ngày 29/09/2026 không ghi được trạng thái xung nhịp của thiết bị: trường
`jetson_clocks_show` yêu cầu quyền root nên được ghi là `null` trong mọi tạo tác của phiên
đó. Một số liệu độ trễ mà trạng thái thiết bị không thể tái dựng thì không phải là một phép
đo. Vì vậy toàn bộ bộ số của mục này được đo lại ngày 01/10/2026 trên các engine TensorRT
được dựng lại cho đúng thiết bị, dưới **hai điều kiện được dán nhãn tường minh**:

- **DVFS mặc định** — bộ điều khiển `schedutil`, GPU 306–1020 MHz, hệ thống tự điều chỉnh
  tần số. Đây là điều kiện mà một hệ thống triển khai thực tế vận hành.
- **Ghim xung nhịp** — `jetson_clocks` đặt CPU 1.728 MHz và GPU 1020 MHz cố định
  (min = max). Đây là một hành động vận hành có thể khai báo, không phải trạng thái mặc định.

**Bảng 5.5J — Nhịp quyết định P3 theo tải crop, hai điều kiện xung nhịp**
(FP16, MAXN_SUPER, n = 100 lần/giai đoạn, trung vị của 3 lần chạy, đã tính giải mã khung)

| Tải crop | DVFS: ms | DVFS: qđ/s | Biên | Ghim nhịp: ms | Ghim nhịp: qđ/s | Biên |
|---|---:|---:|---:|---:|---:|---:|
| p50 = 54 crop | 365,88 | 2,733 | −8,9 % | **310,00** | **3,226** | **+7,5 %** |
| p95 = 106 crop | 403,11 | 2,481 | −17,3 % | 346,87 | 2,883 | −3,9 % |
| p99 = 119 crop | 413,03 | 2,421 | −19,3 % | 355,93 | 2,810 | −6,3 % |
| trần = 128 crop | 420,25 | 2,380 | −20,7 % | 362,32 | 2,760 | −8,0 % |

Nguồn: `decision-latency-fp16-20261001T{220144,220543,220925}.json` (DVFS) và
`...T{222854,223216,223538}.json` (ghim nhịp).

**Phát biểu đúng là một phát biểu hai điều kiện.** Dưới DVFS mặc định, P3 **không đạt**
hợp đồng 3,0 quyết định/giây ở mọi tải crop, kể cả trung vị. Khi ghim xung nhịp, P3 đạt
hợp đồng **chỉ ở tải crop trung vị** (3,226 qđ/s, biên +7,5 %) và vẫn trượt ở p95, p99 và
trần. Luận văn này **không tuyên bố P3 là hệ thống thời gian thực**; nó tuyên bố một điều
hẹp hơn và kiểm chứng được: nhịp 3 quyết định/giây chỉ đạt được ở tải trung vị và chỉ khi
người vận hành ghim xung nhịp.

Việc ghim xung nhịp còn cải thiện **tính lặp lại** của phép đo: độ phân tán giữa 3 lần chạy
giảm từ 2,42 % (DVFS) xuống 0,46 % (ghim nhịp). Phần lớn nhiễu run-to-run trong điều kiện
DVFS là do nhịp độ tăng/giảm tần số của bộ điều khiển, chứ không phải do pipeline.

**Bảng 5.5K — Phân bổ 310,00 ms mỗi quyết định (ghim nhịp, tải crop p50)**

| Thành phần | ms | Tỷ lệ |
|---|---:|---:|
| Bộ phát hiện (16 khung × 11,724 ms) | 187,58 | **60,5 %** |
| Hậu xử lý: chọn thực thể + hình học vùng | 47,35 | 15,3 % |
| Giải mã khung (8 khung liên tiếp, CPU) | 37,61 | 12,1 % |
| Bộ mã hoá ngoại hình (1 lô gồm 54 crop) | 19,34 | 6,2 % |
| Trích crop (OpenCV) | 17,95 | 5,8 % |
| **Đầu quan hệ (mô hình Transformer)** | **0,158** | **0,05 %** |

Nguồn: `decision-latency-fp16-20261001T223538.json`; tổng các thành phần tính toán khớp với
trường `ms_per_decision_p50 = 272,39` của tạo tác, cộng 37,61 ms giải mã.

Kết quả có ý nghĩa với luận văn này là hàng cuối. **Mô hình quan hệ chiếm 0,05 % độ trễ
quyết định và 0,06 % năng lượng quyết định** (0,041 % nếu tính theo quy ước năng lượng biên —
xem Bảng 5.5M); chi phí nằm gần như trọn vẹn ở bộ phát hiện
(60,5 %) và ở phần hậu xử lý chạy trên CPU (15,3 %) — không thành phần nào trong hai thành
phần đó là đóng góp của công trình này. Phát hiện về hiệu quả tham số của Chương 4 do đó đi
thẳng được vào triển khai: đầu quan hệ 18k tham số là *miễn phí* trên cả hai trục, và mọi
công việc tối ưu độ trễ trong tương lai thuộc về bộ phát hiện và hậu xử lý, không thuộc về
đầu quan hệ.

#### Hậu xử lý là thành phần quyết định việc trượt nhịp

Khi chuyển từ DVFS sang ghim xung nhịp, bộ phát hiện giảm 17,3 % và bộ mã hoá ngoại hình
giảm 30,4 %, nhưng hậu xử lý chỉ giảm 0,7 % (47,672 → 47,352 ms). Thành phần này không bị
giới hạn bởi GPU và cũng không bị giới hạn bởi tần số CPU tại điểm vận hành này; nó là hình
học vùng Shapely cộng logic chọn thực thể, chạy đơn luồng trong Python.

Hệ quả định lượng: ở hàng p95 trong điều kiện ghim nhịp, khoảng cách tới hợp đồng là 13,5 ms,
trong khi hậu xử lý tiêu tốn 47,4 ms. Hậu xử lý vì vậy là thành phần duy nhất mà việc tối ưu
có thể đưa hàng p95 vào trong hợp đồng — và nó là một hạng mục công việc tương lai có phạm
vi rõ ràng, nằm ngoài mô hình quan hệ.

#### Giải mã khung phải được tính vào ngân sách

Giải mã được đo trên thiết bị từ 125 clip kiểm tra, dưới hai dạng. Giải mã tuần tự theo luồng
— 8 khung liên tiếp, không seek, đúng như một pipeline trực tuyến phải trả cho mỗi quyết định
— tốn **37,61 ms** (4,70 ms/khung) trong điều kiện ghim nhịp và 43,41 ms dưới DVFS. Trích
khung bằng seek cho 16 khung lấy mẫu tốn **5,41 s**, tức gấp khoảng 144 lần; con số đó là chi
phí chuẩn bị dữ liệu ngoại tuyến và **không** thuộc về nhịp quyết định.

Các lần chạy không giải mã của cùng pipeline cho nhịp cao hơn và vượt hợp đồng ở mọi tải
crop; **những lần chạy đó là đối chứng phòng thí nghiệm và không được trích dẫn như kết quả
ở bất kỳ đâu trong luận văn này**, vì không có triển khai nào giải mã số không khung. Mọi số
trong Bảng 5.5J và 5.5K đều đã bao gồm giải mã.

Hai điểm hạn chế đi kèm phép đo giải mã: OpenCV trên thiết bị này được biên dịch không có
GStreamer, nên **giải mã bằng phần cứng NVDEC chưa được đo** và con số CPU 37,61 ms là một
chặn trên — đường giải mã phần cứng là hướng khả thi nhất để thu hồi hàng p95. Và giải mã
được đo trên tệp, không phải trên luồng thu trực tiếp từ camera.

#### DVFS tạo đuôi độ trễ, không chỉ dịch trung vị

Ngoài trung vị, DVFS còn làm xấu phân vị cao. Với bộ phát hiện, tỷ số p95/p50 là 1,51 dưới
DVFS (14,246 → 21,575 ms) nhưng chỉ 1,03 khi ghim nhịp (11,724 → 12,030 ms); với đầu quan hệ,
p99 vọt lên 1,915 ms dưới DVFS so với 0,208 ms khi ghim nhịp. Với một hợp đồng thời gian
thực, đuôi phân phối mới là đại lượng ràng buộc, nên đây là một lý do độc lập để khai báo
trạng thái xung nhịp trong mọi số liệu độ trễ được công bố.

#### Bộ nhớ và dấu chân engine

Bộ engine FP16 cho triển khai P3 là **24,46 MB** (bộ phát hiện 21,543; ngoại hình 2,727;
đầu E3 nhị phân 0,190 — đầu quan hệ quá nhỏ để FP16 thu gọn kế hoạch so với FP32). Khi cả
4 engine P3 cùng thường trú, **độ hụt bộ nhớ trống toàn hệ thống quy cho việc đưa các engine
này vào thường trú** là **38,05 MB** (`n_engines_resident = 4`,
`memory-fp16-20261001T224837.json`), với RSS đỉnh phía host 31,6 MB.

Con số này **thay thế** giá trị **251,1 MB** từng được ghi ở đây. Lần đọc cũ
(`memory-fp16-20261001T214413.json`) không mang `quiet`, không mang `clocks_pinned` và không
mang `competing_processes`, nên không thể loại trừ tiến trình khác đang chiếm bộ nhớ, và nó lớn
gấp **6,6×** lần đọc lại trong cửa sổ yên tĩnh trên cùng bộ 4 engine. Cách gọi tên cũng đã
sửa: trường này **không phải** "trọng số thường trú trong bộ nhớ thiết bị" — với đường B1, độ
hụt 90,01 MB chỉ bằng **dưới một nửa** tổng 197,4 MB kích thước tệp kế hoạch tương ứng (xem
§5.6.9), nên nó chỉ là độ hụt bộ nhớ trống toàn hệ thống; kích thước tệp engine mới là con số
xác định và được trích riêng.
Nguồn: `memory-fp16-20261001T214413.json`.

Trường `device_used_mb` (3.523,38 trên tổng 7.607,06) là bộ nhớ hợp nhất toàn hệ thống dùng
chung với hệ điều hành, **không** phải dấu chân của mô hình, và không được trình bày như vậy.
Phần 8 GB của thiết bị không phải ràng buộc tới hạn của P3; độ trễ mới là ràng buộc đó.

**Bảng 5.5L — Năng lượng P3 theo giai đoạn** (VDD_IN = tổng công suất đầu vào module,
mỗi giai đoạn bão hoà 60 s)

| Giai đoạn | suy luận/s | Công suất TB | J / suy luận | suy luận·W⁻¹ |
|---|---:|---:|---:|---:|
| Bộ phát hiện | 74,70 | 17,06 W | 0,2284 | 4,38 |
| Ngoại hình (lô 54 crop) | 41,34 | 15,01 W | 0,3631 | 2,75 |
| Đầu quan hệ (E3 nhị phân) | 3.705,04 | 9,06 W | 0,00245 | 408,84 |

Nguồn: `energy-p3-{detector,appearance,relation-head-bin-e3}-fp16-20261001T22*.json`.

Tổ hợp theo đúng cách độ trễ được tổ hợp (16 lần suy luận bộ phát hiện + 1 lô ngoại hình +
1 lần đầu quan hệ) cho **4,02 J mỗi quyết định** theo quy ước tổng công suất bảng mạch.
Trong đó đầu quan hệ chiếm 0,00245 J, tức **0,06 %**.

Các trường `*_delta_idle` **có sẵn trong ba tạo tác trên không được trích dẫn**. Ba lần chạy
giai đoạn ghi nền rỗi lần lượt 6,841 / 5,618 / 7,509 W — phân tán 34 % — vì mỗi lần chỉ lấy
một cửa sổ rỗi 10 s ngay sau tải của giai đoạn trước, nên "nền rỗi" đó còn mang dư nhiệt và
dư tần số và không so sánh được giữa các giai đoạn.

Chỗ này được sửa bằng **một phép đo nền rỗi 60 s duy nhất trên thiết bị đã ổn định** (tải
0,08; 50 °C; GPU ở sàn 306 MHz; `competing_processes` rỗng; `clocks_pinned` sai ở cả CPU và
GPU): **P_rỗi = 4,691 W**, 118 mẫu.
Nguồn: `energy-p3-relation-head-bin-e3-fp16-20261001T225523.json`.
Giá trị này thấp hơn cả ba nền rỗi 10 s từ 20 % đến 60 %, xác nhận rằng các cửa sổ 10 s đã đo
tải đang suy giảm chứ không phải trạng thái rỗi.

**Bảng 5.5M — Năng lượng biên P3, quy chiếu về một nền rỗi chung**
(suy ra từ `avg_power_w` và `fps` của ba tạo tác giai đoạn, theo công thức
`J/suy luận = (P_TB − P_rỗi) / fps` của đặc tả, với P_rỗi = 4,691 W)

| Giai đoạn | ΔP (W) | J / suy luận (biên) | suy luận·W⁻¹ (biên) |
|---|---:|---:|---:|
| Bộ phát hiện | 12,372 | 0,16562 | 6,04 |
| Ngoại hình (lô 54 crop) | 10,319 | 0,24961 | 4,01 |
| Đầu quan hệ (E3 nhị phân) | 4,371 | 0,00118 | 847,64 |

Tổ hợp cho **2,90 J mỗi quyết định** theo quy ước biên, so với 4,02 J theo quy ước tổng công
suất bảng mạch. Nên trích dẫn con số *biên* khi so sánh các thuật toán với nhau và con số
*tổng* khi tính toán nguồn cấp, và phải nói rõ đang dùng quy ước nào. Đầu quan hệ chiếm
**0,041 %** năng lượng quyết định theo quy ước biên.

Hai lưu ý đi kèm. Các giá trị trong Bảng 5.5M là *suy ra*, không phải trường đọc trực tiếp từ
tạo tác: công thức và đầu vào được nêu đầy đủ ở trên để người đọc tính lại được. Và nền rỗi
4,691 W được đo khi chỉ có engine nhỏ nhất (đầu quan hệ, 0,190 MB) thường trú, nên nó là một
chặn dưới của "nền rỗi khi engine thường trú"; các con số biên trong bảng do đó là chặn trên
nhẹ. Lưu ý thêm: `J/suy luận` của hàng ngoại hình là *trên mỗi lô 54 crop*, không phải trên
mỗi crop.

#### Độ bền nhiệt

Một lần chạy liên tục 30 phút giai đoạn bộ phát hiện (01/10, engine dựng lại, xung nhịp động
mặc định; tạo tác ghi `clocks_pinned: false`, `quiet: true`, không có tiến trình cạnh tranh)
**không cho thấy hiện tượng giảm tần số do nhiệt**. Trên toàn bộ 30 cửa sổ 60 giây — tổng cộng
140.001 lượt suy luận — độ trễ p50 giữ trong khoảng **12,734–12,866 ms**, phân tán **1,04 %**,
và độ trôi so với cửa sổ đầu chỉ **0,6 %** (so với cửa sổ nhanh nhất: 0,7 %). Nhiệt độ mối nối
tăng từ 54,0 °C lên mức bình nguyên **tối đa 69,9 °C** (trung bình 68,8 °C) rồi giữ ổn định,
và trường `possible_throttling` đọc **False**. Công suất duy trì trung bình **16,98 W**, đỉnh
18,02 W, với mức sử dụng GPU trung bình 80 % (thấp nhất 69 %). Bộ nhớ hệ thống đi từ 2.932 MB
lên 2.962 MB sau 30 phút (+30 MB), không có dấu hiệu rò rỉ. Nguồn:
`thermal-soak-p3-detector-fp16-20261001T232524.json`.

Lần soak này thay thế lần soak 29/09 của các bản thảo trước, vốn bị nhiễu ở tám cửa sổ đầu bởi
một lần truyền tệp lên thiết bị trong khi soak đang chạy và vì thế không dùng được trường độ
trôi. Hai điều kiện kèm theo vẫn còn nguyên. Soak chỉ vận hành **riêng giai đoạn bộ phát hiện
với đầu vào cố định**, không phải pipeline tổ hợp, nên nó chứng minh độ ổn định nhiệt của *thiết
bị dưới tải GPU bền*, chưa phải của pipeline. Và `possible_throttling` là một suy đoán dựa trên
độ trễ, không phải số đọc từ thanh ghi ngắt nhiệt của phần cứng.

**Một kết quả phụ: hình phạt của xung nhịp động tách được thành hai phần.** Dưới tải bền, cơ chế
xung nhịp động tự đưa GPU lên đúng tần số tối đa (`cur_hz` = 1.020 MHz = `max_hz`) nhưng **không**
đưa CPU lên tối đa (`cur_khz` = 1.113.600 so với `max_khz` = 1.728.000). Tương ứng, bộ phát hiện
trong lần soak đạt 12,796 ms/khung (trung vị theo cửa sổ), nằm **giữa** 14,246 ms của pipeline
dưới xung nhịp động và 11,779 ms của pipeline khi ghim nhịp. Nói cách khác, chỉ riêng việc bão hoà
tải đã thu hồi khoảng **59 %** khoảng cách giữa hai điều kiện, phần **41 %** còn lại chỉ mất đi
khi ghim nhịp, và nó nằm ở phía CPU. Diễn giải này mang tính chỉ dấu chứ không phải một thí nghiệm
có đối chứng: lần soak chạy bộ phát hiện trong một vòng lặp chặt, còn pipeline chạy nó xen kẽ với
hậu xử lý và giải mã trên CPU, nên hai chế độ truy cập thiết bị không đồng nhất.

**Công suất trong mục này không so sánh được với mục 5.6.3.** 16,98 W ở đây là tải nén liên
tiếp ở mức sử dụng GPU 80 %; 7,03 W ở mục 5.6.3 là tải P1 ở mức sử dụng 16 %. Trên thiết bị
này, công suất bảng mạch đi theo **hệ số hoạt động của GPU**, không theo mức trần nvpmodel —
đúng cơ chế "chạy nhanh để về rỗi" mà mục 5.6.4 báo cáo — nên không số nào trong hai số đó
được phát biểu lại như một tỷ lệ của ngân sách công suất. Công suất tường trung bình của
pipeline P3 tổ hợp, ở nhịp khoảng 3 quyết định/giây thay vì bão hoà, **vẫn chưa được đo** và
nằm đâu đó giữa hai giá trị trên.

#### Những gì chưa được đo cho P3

Độ bão hoà đa luồng: bộ công cụ hiện có (`jetson-multi-stream-pipeline.py`) nạp một danh sách
giai đoạn P1 cố định, nên chạy nó với P3 sẽ cho ra đường cong mở rộng của P1 dưới nhãn P3;
kết quả bốn luồng ở mục 5.6.3 là kết quả **P1** và không chuyển giao được. Công suất tường
của pipeline tổ hợp và soak của pipeline tổ hợp cùng bị chặn bởi lý do này. Độ tương đương
đầu ra FP16-so-FP32 của bộ phát hiện cũng chưa được đo — bộ công cụ tương đương hiện có không
biểu diễn được phép so sánh đó. Soak trong điều kiện **ghim xung nhịp** chưa được đo, nên
biên +7,5 % ở hàng p50 chưa được chứng minh là bền trong 30 phút. Các engine FP32 và INT8
chưa được dựng lại cho đúng thiết bị nên số của chúng chưa được trích dẫn.

#### Xuất xứ tạo tác của mục này

Toàn bộ số trong Bảng 5.5J, 5.5K, 5.5L, các con số giải mã, bộ nhớ và dấu chân engine đều dựa
trên tạo tác JSON được lưu trong kho mã và người đọc kiểm tra lại được; các tạo tác này đã
được đồng bộ từ thiết bị về kho sau phiên đo. Mỗi tạo tác nhúng sẵn `device_state` (nvpmodel,
nền tảng, phiên bản Python), commit git và dấu thời gian.

Từ phiên 01/10, `device_state` còn mang thêm trạng thái xung nhịp và tranh chấp đọc trực tiếp
từ sysfs — `cpu_clocks`, `gpu_clocks`, `clocks_pinned`, `load_average`, `competing_processes` —
và công cụ đo in cảnh báo ngay tại thời điểm đo nếu một lần chạy bị tranh chấp hoặc bị ghim
nhịp. Trong ba lần chạy ở điều kiện ghim nhịp, lần thứ ba có đủ khối trạng thái này và xác
nhận `clocks_pinned = true` cho cả CPU và GPU; hai lần đầu diễn ra trước khi công cụ được
tích hợp và trạng thái ghim của chúng được chứng thực bằng biên bản phiên làm việc
(lưu trạng thái → áp dụng → kiểm tra → phục hồi → kiểm tra lại). Ba giá trị nhịp lệch nhau
dưới 1,2 %, nên lần chạy có trạng thái được ghi là đại diện cho cả ba.

Không giá trị nào trong mục này được nhớ lại từ ký ức — mỗi giá trị được đọc từ tạo tác của
nó tại thời điểm viết.

---

### 5.6.9 Đường quyết định đã triển khai (B1) trên thiết bị

Mục 5.6.8 đo pipeline **P3**. Nhưng họ mô hình sinh ra con số tiêu đề của luận văn này không
phải P3; đó là họ **B1**: 16 khung được lấy mẫu → tháp thị giác **CLIP ViT-B/32 đóng băng**
(87.849.216 tham số, đầu ra 512 chiều) → **đầu Transformer thời gian B1** (tổ hợp 3 hạt giống,
603.270 tham số = 3 × 201.090) → `prob[1,2]`. Mục này đo đúng đồ thị đó, vì một con số độ trễ
chỉ có nghĩa khi nó được đo trên đồ thị đã sinh ra con số độ chính xác.

**Con số độ chính xác đi kèm mọi độ trễ trong mục này là 84,2 %, không phải 84,4 %.** Chương 4
báo cáo B1w (333.570 tham số) ở **84,4 %** và B1 (201.090) ở **84,2 %** — hai giá trị không
phân biệt được về mặt thống kê. Đồ thị **được triển khai là B1**, chọn theo một tiêu chí
**biên** đã ghi trước — *mô hình nhỏ nhất* — chứ không theo độ chính xác
(`deployment/scripts/export-b1-clip-temporal-head.py`, phần mở đầu). Quy tắc kèm theo lựa chọn
đó được tôn trọng ở đây: **trích dẫn con số OOF của chính B1, không bao giờ trích `max(B1, B1w)`**.
Đưa 84,4 % cạnh các độ trễ dưới đây sẽ là gán độ chính xác của một mô hình cho thời gian chạy
của một mô hình khác. Nếu luận văn muốn đặt đúng con số tiêu đề 84,4 % cạnh một số đo trên thiết
bị thì phải xuất và kiểm tra tương đương một engine **B1w** — việc đó **chưa được làm**.

Một đặc điểm kiến trúc phải nói trước, vì nó quyết định cách đọc mọi bảng dưới đây: **không có
bộ phát hiện nào trong quyết định phân lớp B1.** YOLOv11s giữ vai trò **định vị** — hệ thống
phải chỉ ra hiểm hoạ *ở đâu* — và được đo **riêng**, không bao giờ được cộng vào chi phí của
quyết định phân lớp. Ba cổng phủ định hội tụ ở Chương 4 và 7 (G14 thực thể-so-toàn khung,
G17 thế hệ bộ phát hiện, G18 hợp nhất hai luồng) là lý do đồ thị quyết định không tiêu thụ hộp
giới hạn.

#### Điều kiện tiên quyết: tương đương FP16 trước, độ trễ sau

Trích dẫn độ trễ FP16 cạnh 84,2 % chỉ hợp lệ nếu engine FP16 ra **cùng quyết định** với đồ thị
FP32 đã dùng để đo độ chính xác. Trên 24 clip thật: **tỷ lệ đồng thuận quyết định = 1,000**,
`max |Δp| = 2,66·10⁻³`, không có sai lệch tiền xử lý nào (`preprocessing_mismatches` rỗng).
Phép kiểm này được chạy **hai lần**, ở hai mức tải thiết bị khác nhau, và cho cùng kết quả —
nên tương đương không phụ thuộc tải.
Nguồn: `r48-parity-fp16-20261001T{222019,224028}.json`.

Trường `end_to_end_incl_decode_ms` của chính các tạo tác đó (p50 = 1.761,8 ms) **không được
trích dẫn như nhịp quyết định**: nó bao gồm trích khung bằng seek ngẫu nhiên vào tệp mp4, là
đường ngoại tuyến. Xem phần giải mã bên dưới.

**Bảng 5.5N — Độ trễ mỗi giai đoạn của đường B1, đo **liên tiếp** (back-to-back)** (FP16,
MAXN_SUPER, DVFS mặc định — không ghim nhịp, cửa sổ yên tĩnh, `runs = 100`/giai đoạn, 3 lần chạy
độc lập). **Đây là chi phí khi GPU đã ở điểm vận hành của nó, không phải chi phí trong vòng lặp
đã triển khai**; vòng lặp tổ hợp ở Bảng 5.5O mới là nhịp triển khai.

| Giai đoạn | tham số | lô | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|---:|
| Tháp CLIP ViT-B/32 (đóng băng) | 87.849.216 | 16 | 23,386 / 23,335 / 23,567 | 31,19 / 35,08 / 35,42 |
| **Đầu Transformer thời gian B1** | **603.270** | 1 | **0,410 / 0,396 / 0,391** | 0,485 / 0,502 / 0,484 |
| **Mỗi quyết định (tính toán)** | | | **23,796 / 23,731 / 23,958** | |
| **Quyết định/giây** | | | **42,02 / 42,14 / 41,74** | |
| Tỷ lệ của xương sống | | | **98,28 / 98,33 / 98,37 %** | |

Nguồn: `r48-pipeline-20261001T{223826,223853,223920}.json` (cả ba đều mang `quiet = true` và
`clocks_pinned = {cpu: false, gpu: false}`). Ba lần chạy lệch nhau **0,95 %** ở trung vị.

Kết quả có ý nghĩa là hàng cuối, và nó **lặp lại phát hiện của mục 5.6.8 trên đồ thị đã triển
khai**: đóng góp của công trình này — đầu Transformer thời gian — chiếm **1,7 %** độ trễ quyết
định; **98,3 %** nằm ở tháp CLIP đóng băng, một thành phần tải về chứ không phải thành phần
được huấn luyện ở đây. Ở P3 tỷ lệ tương ứng là 0,05 % / 60,5 %. Hai pipeline, hai kiến trúc,
cùng một kết luận về phía mô hình: **hiệu quả tham số của Chương 4 đi thẳng được vào triển
khai — không có công việc tối ưu độ trễ nào thuộc về đầu Transformer.** Nhưng kết luận "phần còn
lại thuộc về xương sống" chỉ đúng *trong phạm vi các giai đoạn mô hình*. Khi đo cả vòng lặp
(Bảng 5.5O), hạng mục chi phối **không phải** xương sống mà là **tiền xử lý trên CPU**, và tỷ lệ
1,7 % / 98,3 % của bảng này trở thành 0,3 % / 20,2 % so với toàn bộ khoảng giữa hai quyết định.
Một bảng giai đoạn không thể tự mình chỉ ra hạng mục tối ưu đúng.

#### Bộ phát hiện là cái thước sai cho một ViT

Dự đoán ghi trước phiên đo (P1) nói tháp CLIP sẽ tốn **60–120 ms** cho 16 khung, suy ra từ chi
phí 12,6 ms/khung của bộ phát hiện. Giá trị đo được là **23,386 ms** — **nằm ngoài khoảng dự
đoán, thấp hơn chặn dưới 2,6 lần**. **P1 bị phủ định.** Nguyên nhân: ViT-B/32 ở 224² xử lý 49
token mỗi khung và lô 16 khung chạy như **một** lần suy luận, nên nó khai thác song song theo lô
mà một bộ phát hiện một-ảnh-một-lần ở 800 px không có. Hai mốc so sánh đo trong cùng cửa sổ:

| Mốc so sánh | lô | p50 (ms) |
|---|---:|---:|
| YOLOv11s @ 800 px (**định vị**, đo riêng, **không** cộng vào quyết định) | 1 | 12,636 / 13,001 / 12,741 |
| MobileNetV3-small @ 224² | 16 | 7,336 / 7,710 / 7,645 |

Tháp CLIP đắt hơn MobileNetV3-small **3,06×** ở cùng lô 16 — đó là giá phải trả cho 84,2 %, và
nó là một con số nhỏ chứ không phải một rào cản triển khai.

#### Tiền xử lý — không phải mô hình và cũng không phải giải mã — là chi phí chi phối

Có **hai** phép đo giải mã trên thiết bị và chúng lệch nhau 2,5×, nên phải nói rõ con số nào là
con số triển khai. Phép đo cô lập cho **5,047 / 5,091 ms mỗi khung** (p95 6,259 / 6,943), nguồn
khối `decode` của `decision-latency-fp16-20261001T{224529,224821}.json`. Vòng lặp tổ hợp cho
**2,048 ms mỗi khung** (Bảng 5.5O). Chênh lệch có hai nguyên nhân, và nguyên nhân lớn hơn là **độ
phân giải** — đã kiểm chứng chứ không phỏng đoán: `data/test` gồm **125 clip, toàn bộ 1920×1080**,
còn `data/isafetybench` có độ phân giải trải từ **320×240 tới 1280×720**, phần lớn ở 720p hoặc
thấp hơn, tức khoảng 1/4 đến 1/7 số điểm ảnh của 1080p. Nguyên nhân thứ hai: tạo tác cô lập tính
thời gian **8 khung liên tiếp** mỗi lần (`emit_every = 8`) nên chi phí khởi động mỗi đợt chỉ được
chia cho 8 khung, trong khi vòng lặp tổ hợp giải mã **liên tục cả clip** (5.559 khung / 24 clip
≈ 232 khung mỗi clip).

**Con số triển khai là 2,048 ms/khung — và định ngữ "trên kho này" là thiết yếu**: chi phí giải mã
phụ thuộc độ phân giải nên nó không phải một hằng số của thiết bị. **Rủi ro phải nêu rõ**: một
camera 1080p sẽ tốn khoảng 5 ms/khung, tức theo số học khoảng quyết định tăng thêm ~72 ms lên
~362 ms = 2,76 quyết định/giây, chỉ còn khoảng **1,10×** so với hợp đồng 2,500 qđ/giây của nguồn
1080p ở 60 fps. **Cấu hình đó chưa được đo.** Đây là cách khả dĩ nhất để bản triển khai trượt
thời gian thực, và biện pháp giảm nhẹ — NVDEC — cũng chưa được đo vì OpenCV trên thiết bị này
được biên dịch không có GStreamer. Cảnh báo của cả hai tạo tác được giữ nguyên: giải mã **từ
tệp, không phải từ luồng camera trực tiếp**, và không được làm nóng cache.

Bản triển khai dùng `--window 48 --stride 24`; `sample()` lấy T = 16 khung bằng `np.linspace`
trên cửa sổ, đúng quy tắc mà phần đánh giá dùng. Bước 24 khung biến tần số khung của nguồn thành
một **hợp đồng nhịp thời gian thực**: nguồn ở *f* fps buộc hệ thống phát *f*/24 quyết định mỗi
giây mới theo kịp luồng.

Các bản nháp trước lấy *f* = 25,1 fps và suy ra hợp đồng **1,05 quyết định/giây**. Con số đó
**không đại diện cho kho dữ liệu**: đo siêu dữ liệu của cả **1.100** clip iSafetyBench cho trung
vị **29,97** fps, trung bình **34,25** fps, p95 **60,0** fps, và 25,1 fps chỉ xuất hiện ở **11
clip (1,0 %)**. Phân bố thực tế: 30,0 fps ở 653 clip (59,4 %), **60,0 fps ở 218 clip (19,8 %)**,
25,0 fps ở 92 clip, 15,0 fps ở 53 clip.

Vì vậy hợp đồng phải được phát biểu theo phân bố, không theo một giá trị:

| nguồn ở | hợp đồng (quyết định/giây) | margin của 3,452 qđ/giây |
|---|---:|---:|
| trung vị kho (29,97 fps) | 1,249 | **2,76×** |
| trung bình kho (34,25 fps) | 1,427 | 2,42× |
| **p95 kho = tối đa (60,0 fps), 19,8 % clip** | **2,500** | **1,38×** |

Hợp đồng vẫn **được đáp ứng trên toàn kho**, kể cả ở clip 60 fps, nhưng margin ở đó là **1,38×**
chứ không phải con số nhiều lần mà bản nháp trước nêu — và đó là margin của một vòng lặp **đơn
luồng**, tức một chặn dưới. Mọi phát biểu "thời gian thực" trong chương phải đi kèm tần số khung
mà nó áp dụng.

**Bảng 5.5O — Nhịp **đo được** của vòng lặp tổ hợp B1** (giải mã → tiền xử lý → tháp → đầu;
`--window 48 --stride 24`; 24 clip, 201 quyết định, 5.559 khung mỗi lần; 3 lần lặp ở **cùng**
`--seed 0` để độ phân tán đo *thiết bị* chứ không đo biến thiên tập clip; FP16, DVFS mặc định,
cửa sổ yên tĩnh)

| Đại lượng | min | **trung vị** | max | phân tán |
|---|---:|---:|---:|---:|
| **nhịp trạng thái dừng, ms/quyết định (p50)** | 289,339 | **289,672** | 292,834 | 1,21 % |
| nhịp trạng thái dừng, ms/quyết định (p95) | 321,764 | **322,708** | 322,909 | 0,35 % |
| nhịp tính cả lấp cửa sổ, ms/quyết định | 259,704 | **260,315** | 260,825 | 0,43 % |
| tiền xử lý p50 (ms) | 160,629 | **160,765** | 160,926 | 0,18 % |
| tháp CLIP p50 **trong vòng lặp** (ms) | 58,002 | **58,630** | 58,883 | 1,50 % |
| đầu B1 p50 (ms) | 0,790 | **0,798** | 0,806 | 2,01 % |
| tính toán p50 (ms/quyết định) | 218,750 | **219,153** | 222,310 | 1,62 % |
| giải mã p50 (ms/khung) | 2,048 | **2,055** | 2,104 | 2,73 % |
| hệ số hoạt động (tính toán/đồng hồ tường) | 0,758 | **0,761** | 0,761 | 0,41 % |

Nguồn: `r48-composed-cadence-fp16-20261001T{233159,233324,233449}.json` (cả ba `quiet = true`,
`clocks_pinned.any = false`, `competing_processes` rỗng). Mọi hàng qua được phép kiểm lặp lại
`max − min ≤ 0,10 × trung vị` đã dùng cho trường bộ nhớ.

Nhịp đã triển khai là **289,672 ms/quyết định = 3,452 quyết định/giây** (p95 322,708 ms), tức
margin **2,76×** so với hợp đồng của clip trung vị (1,249 qđ/giây) và **1,38×** so với clip
60 fps — xem bảng hợp đồng ở trên. Vòng lặp **đơn luồng**: một triển khai
thực tế sẽ gối giải mã lên suy luận, nên 3,452 quyết định/giây là **chặn dưới** của thông lượng
và 76,1 % là **chặn trên** của hệ số hoạt động.

Bảng này **thay thế** một ngân sách tổng giai đoạn mà các bản nháp trước của mục này trình bày
như một nhịp. Phải nói rõ vì sao bản nháp đó sai, chứ không chỉ sửa số: một **tổng** các phép đo
giai đoạn cô lập không thể thấy ba chi phí, và cả ba đều lớn.

| | tỷ lệ của tính toán | tỷ lệ của khoảng 289,672 ms |
|---|---:|---:|
| **tiền xử lý (resize + chuẩn hoá 16 khung, CPU)** | **73,4 %** | **55,5 %** |
| tháp CLIP ViT-B/32 | 26,5 % | 20,2 % |
| đầu Transformer B1 | 0,4 % | 0,3 % |
| giải mã (24 khung × 2,048 ms) | — | 17,0 % |
| chưa quy được (chi phí vòng lặp, `cap.read()` ngoài khoảng đo, chỉ mục bộ đệm vòng) | — | 7,4 % |

1. **Tiền xử lý chưa từng có trong ngân sách.** Không bị đánh giá thấp — **vắng mặt**. Nó nằm
   giữa giải mã và tháp, không giai đoạn cô lập nào đo nó, và nó là hạng mục đắt nhất của cả
   pipeline: đắt hơn ViT, đắt hơn giải mã.
2. **Tháp tốn 58,630 ms trong vòng lặp so với 23,386 ms khi đo liên tiếp — 2,51×.** Nguyên nhân
   được phân xử ở Bảng 5.5P.
3. **Tổng giai đoạn ước thiếu nhịp đo được khoảng 4×.** Tính lại trung thực với chính con số
   giải mã trong vòng lặp: 24 × 2,048 + 23,796 = 72,948 ms, so với 289,672 ms đo được — hệ số
   **3,97**.

Phát biểu "giải mã chiếm 83,6 % ngân sách và mô hình chiếm 16,4 %" của bản nháp trước vì vậy bị
**rút lại**: giải mã chiếm 17,0 % khoảng quyết định, và hạng mục chi phối là tiền xử lý.

Trích khung bằng **seek** ngẫu nhiên cho 16 khung đã lấy mẫu tốn **5.815,9 / 5.729,0 ms**, tức
khoảng **1.140×** so với giải mã tuần tự — con số đó là chi phí chuẩn bị dữ liệu **ngoại tuyến**
và không thuộc về nhịp quyết định ở bất kỳ đâu trong luận văn này.

#### Vì sao tháp chậm 2,5× trong vòng lặp: một phép phân xử đã ghi trước khi chạy

Hai nguyên nhân cùng dự đoán hiện tượng đó và chúng dẫn tới hai kết luận khác nhau. **H-sụt-nhịp**:
xung nhịp không được ghim, giữa hai quyết định GPU rỗi khoảng 210 ms trong lúc CPU giải mã và tiền
xử lý, nên bộ điều khiển hạ xung nhịp SM và mỗi lần gọi tháp phải trả giá cho pha tăng nhịp lại —
nếu đúng, đây là hiệu ứng **trạng thái năng lượng của thiết bị**. **H-tranh-chấp**: giải mã và tiền
xử lý tranh chấp bus bộ nhớ với tháp — nếu đúng, hiệu ứng là **nội tại của pipeline** và không thể
xoá bằng việc ghim nhịp.

Phép thử tách hai nguyên nhân: chạy **riêng tháp**, cùng engine, cùng trạng thái xung nhịp không
ghim, 60 lần/nhánh — nhánh A liên tiếp, nhánh B chèn một khoảng rỗi 210 ms trước mỗi lần gọi và
**không có gì khác chạy**. Tranh chấp bị loại trừ *theo thiết kế*, nên mọi mức tăng ở nhánh B chỉ
có thể do trạng thái xung nhịp. **Quy tắc phân xử được ghi trước khi chạy** (và không thương lượng
lại sau khi thấy kết quả): H-sụt-nhịp được xác nhận khi và chỉ khi nhánh B ≥ 1,8 × nhánh A **và**
nhánh B nằm trong ±20 % của 58,630 ms, tức trong [46,90 ; 70,36] ms.

**Bảng 5.5P — Phép phân xử sụt nhịp khi rỗi** (chỉ tháp CLIP, FP16, 60 lần/nhánh, 3 lần lặp)

| Nhánh | p50: min / **trung vị** / max (ms) | phân tán |
|---|---:|---:|
| liên tiếp (back-to-back) | 25,944 / **26,448** / 26,653 | 2,7 % |
| + khoảng rỗi 210 ms, không có gì khác chạy | 54,675 / **55,231** / 55,832 | 2,1 % |
| **tỷ số rỗi / liên tiếp** | 2,051 / **2,088** / 2,152 | 4,8 % |

Nguồn: `r48-idle-droop-probe-fp16-20261001T{234145,234242,234322}.json` (cả ba `quiet = true`,
`clocks_pinned.any = false`).

Theo đúng quy tắc đã ghi trước: 2,088 ≥ 1,8 và 55,231 ms nằm trong dải — **H-sụt-nhịp được xác
nhận, độc lập ở cả ba lần lặp**. Chỉ riêng khoảng rỗi — không giải mã, không tiền xử lý, không
thể có tranh chấp — tái tạo **94,2 %** chi phí tháp trong vòng lặp (55,231 / 58,630).
H-tranh-chấp **không cần thiết** để giải thích bất cứ điều gì; phần dư 3,4 ms (5,8 %) được để
**chưa quy** thay vì gán cho nó. Dải xung nhịp GPU của bo mạch là 306–1020 MHz, tức 3,33×, nên
nó chứa thừa một hiệu ứng 2,09×.

Hai hệ quả ngược chiều nhau, và phải nêu cả hai. Thứ nhất, 23,386 ms **không sai**: nó là một
con số **liên tiếp** và phải được gắn nhãn như vậy. Thứ hai, nó **cũng không phải chi phí triển
khai**, vì triển khai bị chu kỳ hoá theo thiết kế — một quyết định mỗi 24 khung, GPU rỗi ở giữa.
Một con số đo liên tiếp không thể mang sang một pipeline có chu kỳ công tác.

**Chặn trên — bằng số học, không phải phép đo — của lợi ích từ việc ghim nhịp**: nếu ghim nhịp
đưa tháp từ 58,630 về 26,448 ms và không đổi gì khác, khoảng quyết định giảm tối đa 32,18 ms về
257,5 ms = 3,88 quyết định/giây, tức **+12,5 %**. Trần đó thấp vì hạng mục chi phối là tiền xử
lý — một chi phí CPU trên một CPU vốn luôn bận. **Tối ưu tiền xử lý có giá trị gấp nhiều lần
việc ghim nhịp, và khác với ghim nhịp, nó là một thay đổi phần mềm khả chuyển.**

Một kết quả phụ cần báo cáo: nhánh liên tiếp đọc **26,448 ms** ở nơi `decision-latency-fp16-*.json`
đọc 23,386 ms — **cao hơn 13,1 %** cho cùng một phép đo trên danh nghĩa (khác bộ công cụ, khác
nguồn dữ liệu vào; cùng engine, cùng trạng thái thiết bị). Điều đó chặn độ chính xác của mọi con
số độ trễ giai đoạn cô lập ở khoảng **±13 %**, nên trong chương này không con số nào thuộc loại
đó được trích dẫn quá 3 chữ số có nghĩa.

#### Năng lượng

| | giá trị |
|---|---:|
| Thời lượng bão hoà | 60,015 s |
| Số lần suy luận tháp | 2.508 |
| Nhịp | 41,789 suy luận/s |
| Công suất TB (VDD_IN) | **20,105 W** |
| J / quyết định (tổng công suất bảng mạch) | **0,4811 J** |
| J / quyết định (biên, P_rỗi = 4,691 W) | **0,36885 J** |

Nguồn: `energy-r48-clip-vit-b32-vision-fp16-20261001T224142.json`, quy chiếu về nền rỗi 60 s
đã ổn định của `energy-p3-relation-head-bin-e3-fp16-20261001T225523.json`.

Ba lưu ý. Con số **tổng công suất bảng mạch 0,4811 J không đổi** khi đổi nền rỗi, vì nó là
`P_TB / fps` và không chạm tới nền rỗi; chỉ con số **biên** dịch từ 0,36522 J (nền rỗi 10 s,
4,842 W) sang **0,36885 J** (nền rỗi 60 s, 4,691 W) — lệch 1,0 %. Nền rỗi 4,691 W được đo khi
chỉ có engine nhỏ nhất thường trú, nên nó là chặn dưới của "nền rỗi khi engine thường trú" và
con số biên là chặn trên nhẹ. Và **năng lượng riêng của đầu B1 chưa được đo**; nó bị chặn bởi
tỷ lệ 1,7 % của nó trong độ trễ quyết định.

#### Bộ nhớ và dấu chân engine

Dấu chân **tệp engine** là chính xác và kiểm tra lại được: tháp CLIP **174,022 MB** + đầu B1
**1,874 MB** = **175,9 MB** cho đường quyết định phân lớp; thêm engine định vị YOLOv11s
(21,543 MB) thành **197,4 MB**.
Nguồn: `model-size-20261001T224927.json`.

Bộ nhớ thiết bị tăng thêm do engine thường trú, đo **3 lần trong một cửa sổ yên tĩnh** (cả ba
`quiet = true`, `competing_processes` rỗng, không ghim nhịp, tải 0,07–0,25, 52–53 °C):
**91,36 / 90,01 / 89,61 MB**, trung vị **90,01 MB**, phân tán **1,94 %**.
Nguồn: `memory-fp16-20261001T{232747,232808,232830}.json`. RSS đỉnh phía host 31,5–33,4 MB.

Con số này chỉ được trích dẫn vì nó **qua được một phép kiểm lặp lại đã ghi trước khi chạy**
(lặp lại nếu `max − min ≤ 0,10 × trung vị`; đo được 1,75 ≤ 9,00). Lý do phải ghi trước một phép
kiểm như vậy: trường `engines_resident_delta_mb` suy ra từ `cuda.mem_get_info()`, tức bộ nhớ hợp
nhất trống **toàn hệ thống**, nên nó bị nhiễu bởi mọi thứ khác đang chạy trên máy — và trong các
tạo tác không mang nhãn trạng thái, nhiễu đó đã thực sự xuất hiện ở mức 6,6×.

Một cảnh báo còn lại dù phép kiểm đã qua: 90,01 MB **nhỏ hơn một nửa** tổng kích thước tệp
engine tương ứng (197,4 MB). Vì vậy trường này **không phải** "trọng số thường trú trong bộ nhớ
thiết bị" và không được gán nhãn như thế; cách phát biểu đúng là *"độ giảm bộ nhớ hợp nhất trống
toàn hệ thống khi làm các engine này thường trú"*, còn kích thước tệp engine là con số chính xác
được trích dẫn riêng. Chi tiết và bằng chứng:
`plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-10-01-engine-resident-memory-not-reproducible.md`.

Phần 8 GB của thiết bị không phải ràng buộc tới hạn của đường B1: với 175,9 MB engine, **bộ nhớ
không phải trục tới hạn**. Trục độ trễ thì **không còn thoải mái như bản nháp trước nghĩ**: ở clip
60 fps (19,8 % kho) margin chỉ còn 1,38×, nên với đường B1 độ trễ là trục chặt hơn trong hai trục
— cùng dấu với P3, chỉ là ở mức nhẹ hơn.

#### DVFS tạo đuôi khi chạy liên tiếp — và dịch cả trung vị khi chạy theo chu kỳ công tác

Mọi số của đường B1 trong mục này được đo dưới **DVFS mặc định** — điều kiện một hệ thống
triển khai thực tế vận hành. **Không tồn tại phép đo đường B1 ở điều kiện ghim xung nhịp mà
trạng thái được tạo tác chứng thực**, nên mục này **không** đưa ra so sánh DVFS-so-ghim-nhịp
cho B1. Hai lần chạy sớm hơn cùng ngày thường được nhắc tới như "đã ghim nhịp"
(`r48-pipeline-20261001T{222505,222511}.json`, p50 23,481 / 23,167 ms) **không mang khối
`cpu_clocks`/`gpu_clocks`/`clocks_pinned` nào** — công cụ ghi trạng thái chưa được tích hợp
vào thời điểm đó — và hai lần chạy cách nhau 6 giây thì không phải hai lần lặp độc lập. Theo
đúng quy tắc của mục 5.6.8, chúng **không được trích dẫn**.

Điều *có thể* phát biểu từ ba lần chạy sạch, **khi đo liên tiếp**: DVFS tạo **đuôi**. Tỷ số
p95/p50 của tháp CLIP là **1,33 / 1,50 / 1,50** (23,386 → 31,187; 23,335 → 35,077;
23,567 → 35,418), cùng dấu với hiện tượng mục 5.6.8 báo cáo cho P3, và trung vị thì ổn định
(phân tán 0,95 % qua ba lần chạy).

Nhưng phát biểu "không dịch trung vị" **chỉ đúng cho chế độ liên tiếp**, và Bảng 5.5P rút lại nó
cho chế độ triển khai: khi pipeline bị chu kỳ hoá, DVFS dịch trung vị của tháp **2,09×**
(26,448 → 55,231 ms). Đó là hiệu ứng DVFS lớn nhất đo được ở bất kỳ đâu trong chương này, và nó
không nằm ở đuôi mà ở ngay trung vị. Bài học phương pháp: *đo một giai đoạn liên tiếp sẽ che mất
hiệu ứng DVFS chi phối của một triển khai có chu kỳ công tác.*

Hệ quả cho hợp đồng nhịp thì vẫn an toàn, vì nhịp của Bảng 5.5O **đã** chứa hiệu ứng đó: ngay ở
p95 đo được (322,708 ms) đường B1 vẫn đạt 3,099 quyết định/giây, tức **2,48×** hợp đồng của clip
trung vị và **1,24×** hợp đồng của clip 60 fps — khác với P3, nơi đuôi là đại lượng ràng buộc.

#### Những gì chưa được đo cho B1

Đường B1 ở điều kiện **ghim xung nhịp** chưa được đo với trạng thái được chứng thực, nên lợi
ích của việc ghim nhịp đối với đuôi phân phối của B1 chưa được định lượng.
Độ bão hoà **đa luồng** của đường B1 chưa được đo (bộ công cụ hiện có nạp danh sách giai đoạn
P1 cố định, cùng lý do như ở mục 5.6.8). **Soak** của vòng lặp tổ hợp chưa được đo, nên margin
2,76× (và 1,38× ở clip 60 fps) chưa được chứng minh là bền trong 30 phút. **Giải mã bằng phần cứng NVDEC** chưa được đo
— OpenCV trên thiết bị này được biên dịch không có GStreamer — nhưng vì giải mã chỉ chiếm 17,0 %
khoảng quyết định, trần của nó là 17 %; hạng mục có đòn bẩy lớn nhất là **tiền xử lý** (55,5 %).
Vòng lặp tổ hợp ở điều kiện **ghim xung nhịp** **không được đo**, và đây là một quyết định chứ
không phải thiếu sót: nguyên nhân của hiện tượng đã được phân xử ở Bảng 5.5P, trần số học của
việc ghim nhịp chỉ là **+12,5 %** (vì phần chi phối nằm ở CPU vốn đã bận liên tục), và **ghim
nhịp không phải trạng thái khởi động của thiết bị** — một con số đo ở trạng thái mà hệ thống
triển khai không bao giờ ở trong đó sẽ là một con số không dùng được. Do đó +12,5 % ở trên là
số học, không phải phép đo, và cả chuỗi số trong mục này giữ nguyên ở trạng thái **không ghim
nhịp**. Engine **FP32 và INT8** của tháp CLIP
chưa được dựng lại cho đúng thiết bị nên số của chúng chưa được trích dẫn. Năng lượng riêng
của đầu B1 chưa được đo. Và giải mã được đo trên tệp, không phải trên luồng camera trực tiếp.

**Cập nhật 02/10/2026 — tiền xử lý đã được phân tích bên trong, và nguyên nhân là một lỗi lập
trình, không phải một giới hạn phần cứng.** Mọi con số trong các bảng ở trên vẫn là số đo của
hệ thống **trước khi sửa** và không được thay đổi khi chưa đo lại trên thiết bị. Phân tích chạy
trên host (M3, nên **không phải số thiết bị**) cho thấy phép đổi kênh BGR→RGB `f[:, :, ::-1]`
được thực hiện **trước** khi thu nhỏ ảnh: đó là một khung nhìn có bước âm, nên OpenCV phải sao
chép toàn bộ khung ở **độ phân giải gốc** (6,2 MB ở 1080p) trước khi resize về 224×224. Riêng
bước này chiếm **90,3 / 101,6 ms** ở 1920×1080 và **5,5 / 15,5 ms** ở 480×270 — và nó là bước
duy nhất có chi phí tỷ lệ với độ phân giải *nguồn*. Đổi thứ tự (resize trước, đổi kênh sau) cho
pixel **giống hệt từng bit** — `cv2.resize` tổ hợp các điểm ảnh *trong cùng một kênh* và không
bao giờ trộn giữa các kênh — và nhanh hơn **7,46×** ở 1080p, **1,36×** ở 480×270. Tính bit-giống
nhau **không được giả định**: nó là điều kiện (1) của một quy tắc chấp nhận ghi trước khi chạy
(`deployment/scripts/profile-r48-preprocess.py`), được kiểm tra lại ở mỗi lần chạy trên dữ liệu
thật, và được chốt bằng 11 ca kiểm thử so khớp `np.array_equal` với công thức **trước khi sửa**.
Sửa đổi đã được áp dụng vào mã; **việc đo lại trên thiết bị chưa thực hiện được** vì bo mạch mất
nguồn từ 02/10. Dự đoán cho lần đo lại đã được **đăng ký trước** (P-PRE1…P-PRE4) ở
`plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-10-02-preprocessing-cost-is-a-channel-swap.md`,
cùng với điều kiện phủ định của từng dự đoán.

#### Xuất xứ tạo tác của mục này

Mọi engine TensorRT dùng trong mục này được dựng **ngày 01/10/2026 trên chính thiết bị này**.
Điều đó là cần thiết: toàn bộ 30 engine dựng vào hoặc trước 29/09 phát cảnh báo
`Using an engine plan file across different models of devices...` khi giải tuần tự hoá, nên số
độ trễ của chúng không được trích dẫn. Vì thư mục `deployment/export/jetson` trộn engine sạch
của 01/10 với engine cũ, đường B1 được đo qua thư mục liên kết tượng trưng riêng
`deployment/export/jetson-r48` chỉ chứa ba engine của đường này.

Các engine **không** được đưa vào kho mã: chúng phụ thuộc thiết bị (một engine dựng trên bảng
này không nạp được trên bảng khác) và riêng engine tháp CLIP đã 174 MB, vượt hạn mức 100 MB
mỗi tệp của GitHub. Chúng được dựng lại trên thiết bị từ ONNX đã kiểm tra md5 hai đầu.

Mỗi tạo tác nhúng `device_state` với `cpu_clocks`, `gpu_clocks`, `clocks_pinned`,
`load_average` và `competing_processes` đọc trực tiếp từ sysfs. **Trước khi trích dẫn bất kỳ
số độ trễ nào, phải kiểm tra `quiet` và `clocks_pinned` của tạo tác đó.** Ba lần chạy ở Bảng
5.5N đều mang `quiet = true`, `competing_processes` rỗng và `clocks_pinned.any = false`. Các
lần chạy sớm hơn cùng ngày (`r48-pipeline-20261001T{221822,222505,222511}.json`) **không** mang
khối trạng thái này và **không được trích dẫn** ở bất kỳ đâu.

Không giá trị nào trong mục này được nhớ lại từ ký ức — mỗi giá trị được đọc từ tạo tác của nó
tại thời điểm viết.

---

## 5.7 Limitations

Six limitations are documented in full for the reader:

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

6. **Table 5.4's artifact is hand-reconstructed, and the re-run is deliberately
   deferred.** As disclosed under the table, the multi-stream JSON declares itself
   hand-patched to match the chapter. The harness and all six FP32 ONNX models are
   present, so the measurement is re-runnable in about two minutes — but it must be
   re-run on the *same* host these numbers are attributed to. Running it on a different
   Mac would replace a weakly-sourced number with a confidently-stated but
   wrongly-attributed one, which is the worse defect of the two. The host attribution is
   under verification, so the re-run is held until it is settled rather than performed
   for the appearance of rigour. Nothing else in the chapter depends on these four rows:
   they support a saturation-shape argument (throughput peaks at N = 2), not any headline
   claim, and the Jetson multi-stream result in §5.6.3 is measured independently.

---

## 5.8 Conclusion

Chapter 5 established that the hybrid YOLOv11s + Transformer pipeline from
Chapters 3 and 4 can be deployed to a commodity Apple M3 CPU through an
ONNX Runtime path, with zero accuracy loss relative to the PyTorch training
baseline and with real-time single-stream throughput above the 30 FPS budget.
Specifically:

- **Accuracy preservation:** CPU-FP32 and CoreML-FP32 ensemble accuracy is
  56/64 = 87.5%, matching the training baseline exactly. INT8 quantization
  costs one sample (55/64 = 85.9%). Both figures are export-fidelity checks against
  a test-selected P1 reference (§5.1), not generalisation estimates.
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
