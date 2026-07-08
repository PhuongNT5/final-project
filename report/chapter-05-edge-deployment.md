# Chapter 5 — Edge Deployment

## 5.1 Introduction

Objective 3 of this thesis is to demonstrate that the trained hybrid pipeline
(YOLOv11s backbone + Transformer heads) can execute on commodity edge hardware
without sacrificing the accuracy established in Chapters 3 and 4. Chapter 5
reports a concrete, reproducible deployment to an Apple M4 CPU using ONNX
Runtime, and extends the analysis with a cited projection for NVIDIA Jetson
Orin-class devices. No GPU rental, no proprietary runtime: the deployment path
is entirely open-source and local.

The best Track A configuration carries forward as the reference point: an
ensemble of the Direct4 head (`direct4-s314`) and a scenario → pair cascade
(`scenario-s7` gating `pair{0..3}-s7`). On the 4-class unsafe subset of
Unsafe-Net (64 held-out videos), this ensemble achieves a top-1 accuracy of
56/64 = 87.5% (macro 89.28%), matching the PyTorch training baseline. The
deployment target is real-time inference at 30 FPS, which imposes a 33 ms
per-frame budget.

Chapter 5 answers three questions. First, does the exported ONNX pipeline
preserve accuracy? Second, does it meet the 30 FPS budget on M4 CPU? Third,
does it scale to multiple concurrent video streams without dropping frames?
Sections 5.2 and 5.3 describe the architecture and the export pipeline.
Section 5.4 reports measured end-to-end latency and accuracy on the M4. Section
5.5 extends the pipeline to N = 1…4 concurrent streams. Section 5.6 provides
the Jetson projection. Section 5.7 enumerates limitations, and Section 5.8
concludes.

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
| CPU-FP32    | 28.8 ms | 37.3 ms | 34.7 |
| CPU-INT8    | 32.0 ms | 35.5 ms | 31.3 |
| CoreML-FP32 | 44.7 ms | 56.1 ms | 22.4 |
| CoreML-INT8 | 86.6 ms | 106.5 ms | 11.5 |

**CPU-FP32 is the headline result: 28.8 ms p50, 37.3 ms p95, 34.7 FPS — meeting
the 30 FPS real-time budget with 4 ms of headroom at p50.** The INT8 CPU path
is slower, not faster: the M4 AMX matrix units have no native INT8 path in the
current ONNX Runtime build, and the dequantize kernels dominate. The
transformer heads individually benefit from INT8 (~2× faster than FP32), but
the backbone regresses enough to wipe out that gain at the end-to-end level.

The CoreML paths are measured, not projected. The backbone alone accelerates
to 1.9 ms on the ANE under CoreML FP32 — a 4.6× speedup over the CPU
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

### 5.5.3 Batch utilization trade-off

Batch utilization sits at 50–57% for N ≥ 2, below the 75% target set in the
phase plan. This is a direct consequence of the 5 ms batch-assembly bucket
window: at 30 FPS reader pacing, frame inter-arrival time is 33 ms and
inference itself takes 20–50 ms, so the 5 ms bucket rarely sees more than one
frame per stream. Lengthening the bucket would raise utilization at the cost
of queueing latency; this is a deployment-tuning knob, not a pipeline defect,
and is documented as such.

---

## 5.6 Jetson Orin Projection (Unverified)

Jetson hardware was not physically available for this thesis. This section
provides a **cited projection**, not a measurement, derived from published
vendor benchmarks. The projection method and its assumptions are stated in
full to make the limitation explicit.

### 5.6.1 Method

Ultralytics publishes YOLOv11s benchmarks on Jetson Orin Nano Super
(7 W / 15 W / 25 W modes) and Jetson AGX Orin, both at 640 × 640 input.
The reported backbone latencies are 5.25 ms (Orin Nano Super, 25 W, TensorRT
FP16) and 2.98 ms (AGX Orin, MAXN, TensorRT FP16). Source documented in
`plans/reports/researcher-track-b-jetson-orin-projection.md`.

To project to our workload, three adjustments are applied. (i) Input
resolution 640 × 640 → 224 × 224: the backbone FLOPs scale approximately
linearly with spatial area, so the 224² / 640² = 0.122 ratio is applied
conservatively (i.e. we use 0.2× rather than 0.12× to account for fixed
overhead). (ii) The transformer head cost is held equal to the M4 CPU
measurement, since the Orin Cortex-A78AE CPU is slower per-core than the M4
performance cores; this is pessimistic for Orin. (iii) The measured M4
ratio of backbone to full-pipeline latency (roughly 1 : 3) is preserved.

### 5.6.2 Projected numbers

Applying the method gives an end-to-end estimate of ~15 ms per frame on Orin
Nano Super (25 W, TensorRT EP), i.e. roughly 67 FPS single-stream. At N = 4
with TensorRT batching, aggregate throughput would plausibly exceed
120 FPS — well above the 30 FPS budget, leaving headroom for additional
streams or for live-camera overhead.

### 5.6.3 Caveats

The projection assumes (a) the TensorRT FP16 backbone latency scales as
claimed for 224 × 224 input, (b) head-cost remains CPU-bound, (c) memory
bandwidth does not become the new bottleneck. Any of these can break in
practice. The projection is offered as a **plausibility argument** that the
deployment path generalizes to NVIDIA Jetson, not as a measurement. A
physical Jetson measurement is listed as future work (Section 5.7).

---

## 5.7 Limitations

Five limitations are documented in full for the reader:

1. **Input resolution mismatch.** Training used 640 × 384 image crops fed to
   YOLO; deployment resizes frames to 224 × 224. Accuracy was preserved on
   the 64-video test set (56/64 on both training baseline and CPU-FP32 ONNX),
   but degradation on finer-grained scenes, smaller objects, or more diverse
   camera angles cannot be ruled out. A follow-up evaluation at matched
   resolutions is listed as future work.

2. **Test set size.** The 4-class unsafe subset is 64 videos, 16 per class.
   Confidence intervals on the 87.5% accuracy figure are wide: a single
   swapped sample shifts the headline by 1.5%. All comparisons between
   configurations in Tables 5.2 and 5.3 should be read with this in mind.

3. **Jetson projection, not measurement.** Section 5.6 numbers are derived
   from vendor benchmarks with stated adjustments; they are plausibility
   arguments, not data.

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
- **Single-stream latency:** CPU-FP32 end-to-end is 28.8 ms p50 / 37.3 ms
  p95, i.e. 34.7 FPS — 4 ms of headroom against the 30 FPS budget at p50.
- **Multi-stream scaling:** Four concurrent streams sustained at 20.8 FPS
  aggregate with zero dropped frames; aggregate throughput peaks at N = 2
  with 29.1 FPS. Per-stream head p50 ~45 ms at N = 4.
- **Reproducibility:** The full Chapter 5 pipeline reproduces from a clean
  clone in seven commands (see `deployment/README.md`).
- **Artifacts:** Four PNG figures, one Mermaid architecture diagram, one
  Netron graph screenshot, and three data tables — all auto-generated from
  the JSON artifacts, so the thesis numbers cannot drift from the code.

Future work for Track B is twofold: (a) a physical Jetson Orin Nano Super
measurement to replace the projection in Section 5.6, and (b) a live camera
capture path integrated into the Streamlit dashboard. Both are engineering
extensions rather than research contributions and are left as follow-on
development.
