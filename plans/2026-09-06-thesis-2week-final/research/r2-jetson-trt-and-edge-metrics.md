# R2 — Jetson/TensorRT export + edge metrics for the YOLO26/11 + relation-Transformer pipeline

Scope: MT3 (§4g). New headline = entity detector (YOLO26/YOLO11) → `RelationTransformer`
(`model_relation_head.py`) over fixed-shape padded box tokens. Prior Jetson numbers
(`benchmarks-jetson-fp16.json`, 8.62 ms p50, 116 FPS) are for the **old P1 ensemble**
(backbone + direct4/scenario/pair0-3 heads) — different model, different token scheme. Do not
reuse those numbers for the new headline; re-measure.

## 0. Architecture fact that changes the whole export story

`RelationTransformer` (`model_relation_head.py`) is **already fixed-shape**, not variable-length:
`T=16, N_MAX=8` (`build_relation_tokens.py:28`) → sequence len = `T*N_MAX+1` (CLS) = **129 tokens,
static**, feature dim 10 (`[class_id, xc,yc,w,h,conf,iou_walkway,iou_restricted,foot_in_walkway,norm_dist]`).
Padding to N_MAX per frame + `mask (B,T,N)` already implemented; converted to
`key_padding_mask` (`kpm`, bool, `m==0`) fed to `nn.TransformerEncoder(nn.TransformerEncoderLayer(...))`
with `batch_first=True`. Model is small: d=128, 2 layers, 4 heads, ffn=256 — few hundred K params.

**Consequence:** the "variable-length token" export problem the brief worries about does not exist
at the model level — it was solved at the *data* level (fixed padding + mask), which is exactly
the standard fix TensorRT needs anyway. The only remaining export risk is operator coverage for
`nn.TransformerEncoderLayer` (internally may lower to `scaled_dot_product_attention` depending on
PyTorch version/backend) and mask handling, not dynamic shapes.

## 1. Export path: Transformer over token sequence → TensorRT 10.3 / JetPack 6

**Static shape is the fix, and it's already yours.** Export with fixed input shapes
`tok: (1,16,8,10)`, `mask: (1,16,8)` — no `dynamic_axes`/`dynamic_shapes` at all for T/N (batch=1
fixed too, matching the project's existing "dynamo bakes batch=1" constraint noted in
`deployment/README.md`). This sidesteps TRT dynamic-shape optimization profiles entirely — only
use them if you later want batch>1 for multi-stream fusion (not needed if you run N independent
engine contexts, see §2).

**What breaks, concretely:**
- `torch.onnx.export(..., opset_version=17)` on `nn.TransformerEncoderLayer` may internally call
  `F.scaled_dot_product_attention` (PyTorch ≥2.1 fast path) which **has no native ONNX opset ≤18
  translation** in the legacy (non-dynamo) exporter — raises `UnsupportedOperatorError` for opset
  14/17/18 [pytorch#96944, #106612, #97262]. TensorRT 10.7+ added an `Attention` op at **opset 23**
  via the *dynamo* exporter path, but has open correctness/If-subgraph bugs as of 2026
  [NVIDIA/TensorRT#4333, #4739] — do not depend on it for a thesis deadline.
- Fix used project-wide already (`deployment/scripts/transformer-head.py`, per README "Opset"
  caveat): force PyTorch to use the **math/eager attention backend** during export so SDPA lowers
  to explicit matmul+softmax+add ops (all in `run_check_trt_compat.py`'s `TRT_OPS` allow-list:
  `MatMul, Softmax, Add, Mul, Div, Erf`, etc.). In code: wrap export in
  `torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)`
  or simply export with the older `torch.nn.MultiheadAttention` compat mode (PyTorch <2.1 always
  did this; the project pins PyTorch 2.11 per README, so an explicit context manager or a
  hand-written attention forward is needed).
- **Mask correctness under export:** `key_padding_mask` becomes an `Add` of a large-negative bias
  (`-inf` or `-1e9`) before `Softmax` when traced through math attention — verify with
  `verify-onnx-parity.py`-style comparison (repo already has this pattern) that padded positions
  get ~0 attention weight post-export; `NonZero`/boolean-indexing ops sometimes appear from
  `mask==0` and are NOT in TRT's supported list reliably for dynamic content — since shape is
  static here this degrades to a static `Where`/`Add`, which **is** supported (already in
  `run_check_trt_compat.py:TRT_OPS`).
- **Opset choice:** target **opset 17** (matches the rest of the project's transformer heads per
  `trt-compat-report.md`) unless dynamo forces 18 (README already notes this happens silently
  under PyTorch 2.11 — pin `dynamo=False` or set `opset_version` explicitly and verify with
  `onnx.checker` + the existing `run_check_trt_compat.py` after export, do not assume).
- `LayerNorm` → `LayerNormalization`: native ONNX op since opset 17, in TRT's op list already.
  GELU: torch exports `Erf`-based GELU by default at opset 17 (not the native `Gelu` op, opset 20+)
  — fine, `Erf` is in the allow-list.
- **torch.onnx.export vs dynamo export:** use the **legacy TorchScript-based exporter**
  (`dynamo=False`, the project's current default per README) for this small encoder — dynamo
  export is where the SDPA→opset23-Attention auto-fusion happens and where TRT 10 has open bugs.
  Legacy exporter + forced eager attention is the boring, reliable path for a 2-layer/4-head model
  this small; performance difference between fused-attention-op and explicit-matmul is negligible
  at this scale (129 tokens, d=128) on Orin Nano.
- After export: run the project's own `run_check_trt_compat.py` against the new `.onnx` (it already
  has the right op allow-list and reports opset/unsupported-ops/shape info) **before** touching the
  device — this is the offline gate the project already uses for the old heads.

**Concrete recipe for this exact model:**
1. `model.eval()`, force math attention backend (see above) or rewrite `enc` using explicit
   `nn.MultiheadAttention` with `need_weights=False` and confirm it still traces to matmul+softmax.
2. Dummy inputs: `tok=torch.zeros(1,16,8,10)`, `mask=torch.ones(1,16,8)`.
3. `torch.onnx.export(model, (tok,mask), 'relation-fp32.onnx', input_names=['tok','mask'],
   output_names=['logits'], opset_version=17, dynamo=False)` — no dynamic_axes.
4. `python3 run_check_trt_compat.py` (extend `EXPORT_DIR` scan or point at the new file manually).
5. On Jetson: `trtexec --onnx=relation-fp32.onnx --saveEngine=relation-fp16.engine --fp16
   --verbose 2>&1 | tee trtexec-relation.log` — inspect log for "unsupported node" / fallback-to-FP32
   layer warnings (TRT silently falls back per-layer; this is a silent accuracy/speed trap — check).

## 2. Two-stage pipeline on device: detector + Transformer in one process

- **Keep separate engines**, do not fuse the detector and the RelationTransformer into one TRT
  network. Different input cadence (detector runs once per frame; RelationTransformer runs once
  per T=16-frame window on accumulated tokens), different precision needs likely (detector may
  tolerate INT8 better than the small attention head — see §3), and fusing forces a single
  optimization profile for two very different compute graphs for no measured benefit at this
  model size (RelationTransformer is <1 ms class per the old heads' analogue: `direct4_ms.50=1.41`
  in the old fp16 benchmark — expect similar order for the new head, dwarfed by detector cost).
- **Avoid host round-trips between stages**: token-building (`build_relation_tokens.py` logic —
  box coords, IoU-with-zone, foot-in-walkway) is currently NumPy/CPU. On Jetson this is a
  CPU↔GPU sync point per frame if not careful. Two options, cheapest first:
  1. Keep token math on CPU (it's tiny: ≤8 boxes × 10 floats × 16 frames) but use **pinned
     (page-locked) host memory** (`cuda.pagelocked_empty` via pycuda, or `torch.empty(...,
     pin_memory=True)` if using Torch-TRT) for the tensor handed to `enqueue_v3`/`execute_async_v3`
     so the H2D copy is async and overlaps with the next frame's detector inference.
  2. If profiling shows the CPU token-build is the bottleneck (unlikely at N_MAX=8), port IoU/mask
     math to a tiny CUDA kernel or vectorized Torch-on-GPU op — YAGNI unless measured to matter.
- **CUDA streams**: run detector inference and RelationTransformer inference on **separate
  `cudaStream_t`** (two `pycuda.driver.Stream()` / two TRT `IExecutionContext`s) so that while
  stream A is running frame N's detector, stream B can run the *previous* window's transformer
  pass concurrently — genuine overlap only if GPU has spare SM capacity; Orin Nano's single GPU
  will serialize compute-bound kernels but can still overlap memory copies with compute
  (`cudaMemcpyAsync` + kernel on different streams). For a first pass, do NOT over-engineer this —
  measure single-stream-serial latency first (§4), then only add stream overlap if p50/frame
  doesn't already exceed the 1/target-FPS budget.
- **Multi-stream (N=4) architecture choice — DeepStream vs plain TRT Python API:**
  - Plain TensorRT Python (pycuda/`tensorrt` bindings) + OpenCV/GStreamer capture, one thread or
    process per stream, is what the project's existing `multi-stream-pipeline.py` /
    `jetson-multi-stream-pipeline.py` already do. **Recommendation: keep this path.** It is already
    built, understood, and produces the honest zero-drop numbers seen in
    `multi-stream-jetson-fp16-realtime.json`. Rewriting onto DeepStream (Gst-nvinfer +
    Gst-nvstreammux batching, NVMM zero-copy buffers) would likely raise aggregate FPS at N≥3 (batched
    inference amortizes per-call overhead) but is a non-trivial new pipeline (custom parser for a
    Transformer secondary stage is not a stock DeepStream pattern — `nvinfer` secondary GIEs expect
    per-object classifiers, not a temporal-window-over-multiple-frames graph like RelationTransformer).
  - **Verdict: plain TensorRT Python API for the thesis deadline.** DeepStream buys batched-decode/
    infer throughput at the cost of days of unfamiliar pipeline-config work for a payoff (aggregate
    FPS) that is not what MT3 is graded on (zero-drop + measured p50/p95/p99 + energy matter more
    than squeezing extra FPS). Note this tradeoff explicitly in Ch5 as a "future work" line — it's
    a legitimate, citable limitation.

## 3. INT8 calibration — both stages

- **Detector stage (YOLO26/11, conv-heavy):** `IInt8EntropyCalibrator2` — TensorRT's default,
  KL-divergence-based, is the standard and Ultralytics' own `.export(format='engine', int8=True,
  data=...)` (used already in `tools/jetson_benchmark.py:export_engines`) drives Ultralytics'
  internal calibrator which is entropy-based over its own calibration image loader. Use this path
  as-is — do not hand-roll a calibrator for the detector.
- **RelationTransformer stage (attention-heavy):** community/NVIDIA guidance favors
  **`IInt8MinMaxCalibrator`** over entropy for attention/Transformer blocks — entropy calibration
  can clip outlier activations that self-attention produces (large logit magnitudes pre-softmax),
  hurting accuracy more than a simple min/max range would. Given the head is tiny (129 tokens,
  d=128) and accuracy (not speed) is what's graded per §4g, **INT8 on the RelationTransformer is
  low-value / high-risk — recommend FP16-only for this stage** unless a quick MinMax-calibrated
  INT8 run shows negligible accuracy drop (test on the 64-clip parity check, not train). Justify
  this choice explicitly rather than skip silently (Hard Rule 4: "failed run gets a control").
- **Calibration batch count:** no single NVIDIA-mandated number; practical guidance from TRT docs
  + community consensus is **500–1000 representative images total**, batched (e.g., `batch=8` ×
  ~64–128 batches) is more than sufficient for a detector this size; **the project already has
  `deployment/artifacts/calib-jetson/backbone-batch-{0..~130}.npy`** (confirmed present, ~130
  files seen) — reuse this count as precedent, but verify these are **real frames from
  `data/train/`**, not synthetic noise (`JETSON-DEPLOYMENT.md` "Known gaps" #2 explicitly flags the
  OLD calibration set as synthetic-noise — **check whether `calib-jetson/` was regenerated for the
  new pipeline or is stale from P1**; if stale, regenerate from real Unsafe-Net train clips run
  through the actual detector preprocessing pipeline before trusting any INT8 number).
- **Building a representative calibration set from video clips:** sample frames stratified across
  the 8 behaviour classes and across cameras/views (same stratification axis as the val split,
  §4h protocol) — 8–16 frames per clip from ~40–60 train clips gives ~500 frames at batch=8. Do
  NOT calibrate on test clips (protocol violation, §4h rule 2/3).
- **Verifying accuracy parity FP32 vs FP16 vs INT8, properly:** re-run the SAME 64-clip test set
  (or better, the k-fold 516 val fold used for A3, never test beyond the one declared touch) through
  each engine and report top-1/macro exactly as `verify-onnx-parity.py` already does for FP32/INT8
  ONNX drift (project pattern: numeric drift table + downstream-accuracy table, not just drift).
  **Constraint from §4h: the test touch is singular** — decide calibration/precision choice on
  val/k-fold accuracy, and only run the frozen INT8-vs-FP16-vs-FP32 comparison on the declared test
  touch once, logged in `reports/test-touch-log.md` alongside the accuracy test touch (or reuse the
  same touch if numbers come from one pass through the pipeline in all three precisions).

## 4. Measurement methodology per §4g metric

### 4.1 Latency p50/p95/p99, per FRAME and per CLIP
- **Timer:** `time.perf_counter()` around each stage call (matches project convention already in
  `tools/jetson_benchmark.py` and `run_relation_transformer.py`'s clip-time claim). For GPU-bound
  TRT calls, **must synchronize** before stopping the timer: call `context.execute_async_v3(stream)`
  then `stream.synchronize()` (pycuda) or `torch.cuda.synchronize()` if using Torch-TRT — otherwise
  you measure enqueue time, not compute time (a documented, common measurement bug).
- **Warmup:** ≥10 iterations discarded before timing starts (project convention:
  `--warmup 10` in `jetson-benchmark.py`, `--warmup 30` default in `tools/jetson_benchmark.py`) —
  use ≥30 for the new pipeline's first-call TRT JIT/cache effects (`JETSON_QUICKSTART.md` notes
  "first inference takes 3 min" — that's engine-build JIT, separate from steady-state warmup, but
  confirm engine is pre-built/cached before any timing run).
- **Per-frame latency** = detector-only p50/p95/p99 over ≥300 single-frame calls (dummy or replayed
  real frames), each independently timed and sorted (existing `bench_one()` pattern in
  `tools/jetson_benchmark.py` is directly reusable — extend it, don't rewrite).
- **Per-clip latency** = full clip pipeline time: N_frame detector calls + 1 RelationTransformer
  call + token-build overhead, measured end-to-end per clip (`time.perf_counter()` clip-open to
  clip-classified), matching the paper's own "0.14 s/clip" measurement convention (§4a) — run over
  all 64 test clips (or the k-fold val clips) for the distribution, not just N synthetic reps.
- **Run count:** ≥100 for per-frame (matches existing convention), all-clips-once (~64 or ~516) for
  per-clip since each real clip differs in length/detections — report percentiles over the clip set,
  not repeated timing of one clip.
- **Percentile computation:** sort + index (`ts.sort(); p(q)=ts[int(q*len(ts))]`) — the exact
  pattern already in `tools/jetson_benchmark.py:bench_one` — keep it, it's fine for n≥100.

### 4.2 FPS and multi-stream aggregate throughput with a rigorous zero-drop definition
- **Single-stream FPS** = `1000/mean_ms` (existing convention) or `1000/p50_ms` — pick one and state
  it (project currently uses mean in `tools/jetson_benchmark.py`, p50-derived FPS appears
  inconsistently elsewhere — **standardize on `1000/mean_e2e_ms` project-wide for this metric and
  say so once** in the new benchmark script's docstring).
- **Aggregate throughput (N streams)** = `sum(frames_processed_i) / wall_clock_elapsed` across all
  N concurrent streams — this is exactly `aggregate_fps` in the existing
  `multi-stream-jetson-fp16-*.json` schema; reuse the schema.
- **Zero-drop, precise definition:** a frame is "dropped" if the producer (video source at its
  native FPS, e.g., 30) pushes a new frame while the consumer queue is full and the frame is
  discarded rather than processed. The existing artifacts already demonstrate BOTH regimes:
  - `multi-stream-jetson-fp16-realtime.json` — rate-limited to real-time (source paced at nominal
    FPS, consumer keeps up) → **0 drops at all N=1..4** tested. This is the ONLY config that can
    honestly be called "zero-drop real-time."
  - `multi-stream-jetson-fp16-unbounded.json` — consumer runs flat-out (no pacing), N≥3 shows
    `dropped: 33`, `23` etc. — this is throughput-under-saturation, NOT a zero-drop result, and
    must be labeled as such, not conflated with the realtime numbers.
  - **Recommended definition for the thesis:** "Zero-drop" = every stream is fed at a fixed
    real-world FPS (e.g. 30 fps camera-equivalent pacing via `time.sleep` between frame pushes or
    actual `cv2.VideoCapture` real-time playback), a bounded queue of depth K (state K, e.g. 2–4
    frames) sits between producer and consumer, and **0 frames are discarded across the full run**
    at each N. Report queue depth, source FPS, and drop count explicitly per N — exactly the
    `streams[].dropped` field the existing JSON schema already has. Reuse it.
  - Prove no drops: log `frames_pushed` vs `frames_processed` per stream, assert equality, report
    both numbers (not just a boolean) — add `frames_pushed` alongside existing `frames`/`dropped`
    keys if not already distinguished.

### 4.3 Params, FLOPs, model size, INT8 compression ratio
- **Params:** `sum(p.numel() for p in model.parameters())` on the **PyTorch model**, not the TRT
  engine (TRT engines don't expose a clean param count) — pattern already in
  `tools/jetson_benchmark.py` (`info['params_M']`) and `model_relation_head.py:count_params`.
  Report detector params + RelationTransformer params separately and summed.
- **FLOPs:** compute on the **PyTorch model** with `thop` or `fvcore.nn.FlopCountAnalysis` (fvcore
  preferred — more actively maintained, used by Detectron2/timm) at the real fixed input shapes
  (detector: imgsz×imgsz×3; RelationTransformer: (1,16,8,10)+(1,16,8) mask) — do NOT try to derive
  FLOPs from a `trtexec` layer profile; TRT's `--profilingVerbosity=detailed` JSON gives per-layer
  *latency*, not FLOPs, and fused/quantized kernels make FLOP-from-engine unreliable and
  non-standard for citation. State clearly in the thesis: "FLOPs computed on the FP32 PyTorch
  graph via fvcore; TRT engine layer count reported separately for transparency, not as a FLOPs
  proxy."
- **Model size on disk:** `os.path.getsize(engine_path)` for each precision variant (already the
  pattern in `bench_one()`'s `size_mb`) — report `.onnx` size too for the pre-TRT reference point.
- **INT8 compression ratio:** `size_mb(fp16_engine) / size_mb(int8_engine)` — already implemented
  exactly this way in `tools/jetson_benchmark.py` (`info['int8_compression']`). Compute for detector
  and (if INT8 is attempted) the RelationTransformer separately.

### 4.4 Energy — J/inference and FPS/W (the weakest-defined metric; be precise)

**Which rail(s) to sum:** report **VDD_IN** as the headline number — it is the single rail closest
to "total module power draw during compute" (covers CPU+GPU+CV engine on Orin Nano's INA3221 layout;
does NOT include board-level overhead like fan/USB/NVMe, so it is a lower bound on true wall power,
but it is the standard rail cited in Jetson power literature). Optionally report `VDD_CPU_GPU_CV`
alongside as a secondary breakdown (isolates compute-only draw, excludes `VDD_SOC` = memory
controllers/video codecs) — do not report `VDD_SOC` alone as if it were total power. **State which
rail in every number** — the project's own `power-sweep-summary.json` already reports a generic
`power_mw` field without naming the rail; this is a defensible naming gap to fix in the new script
(name the field `vdd_in_mw` explicitly).

**Sampling rate:** `tegrastats --interval 100` (100 ms) is the project's existing convention
(`tools/jetson_benchmark.py:PowerSampler`) and matches literature practice (20–100 ms typical);
use 100 ms for benchmark runs, the existing 500ms/1000ms intervals used for the soak log
(`tegrastats-soak30min.log`) are coarser but adequate for a 30+ minute thermal trend (not per-
inference energy) — **do not use the soak-log sampling rate for per-inference J/inf calculations**,
only for the thermal/throttle narrative (§4.5).

**tegrastats log → joules-per-inference, exact procedure:**
1. Start `tegrastats --interval 100` as a background subprocess, parse each line for the VDD_IN
   field (format varies by JetPack/board rev — grep for `VDD_IN` token specifically; the existing
   `PowerSampler.run()` in `tools/jetson_benchmark.py` parses ANY `mW.../` token, which is fragile —
   **name the exact rail token expected** and fail loudly if not found, don't silently average
   whatever rail happens to match the regex).
2. Record wall-clock start/end of the timed inference loop (N inferences).
3. Average sampled power over the *exact* time window the loop ran: `mean_power_w = mean(samples
   within [t_start, t_end])`.
4. `energy_j_total = mean_power_w * (t_end - t_start)`; `j_per_inference = energy_j_total / N`.
   This is discrete-integration-by-mean-power, adequate at 100 ms sampling for inference calls in
   the 3–15 ms range only if N is large (≥300) so the sampling-vs-inference-rate mismatch averages
   out — **this is the key subtlety**: at 100 ms sampling and ~9 ms/inference, you get ~1 power
   sample per ~11 inferences, so per-inference energy is a *statistical* estimate over the whole
   run, not a per-call measurement. State this explicitly as a limitation; do not imply per-call
   energy precision you don't have. (Equivalent to `PolyThrottle`/similar edge-energy papers'
   discrete-integration approach — cite as methodology precedent, not as validation of any specific
   number.)
5. `fps_per_watt = fps / mean_power_w` (already implemented correctly in
   `tools/jetson_benchmark.py:bench_one`).

**Idle-subtracted vs total power — report BOTH, labeled:**
- **Total power** = mean VDD_IN during the active inference loop (what §4.4 step 3 gives).
- **Idle power** = mean VDD_IN over ≥30s with the process NOT running (screen off/no GUI, network
  idle) — measure once per nvpmodel mode (idle power differs by mode).
- **Idle-subtracted (compute-marginal) power** = `total - idle`; report `j_per_inference` computed
  BOTH ways (total-power-based, and idle-subtracted-based) — the total-power number is what a
  deployment actually costs (fair for a thesis "system cost" claim); the idle-subtracted number
  isolates the *model's* marginal energy cost (fair for comparing architectures/precisions to each
  other). Existing `power-sweep-summary.json` reports only total (`power_w_mean` ~5.2-5.3W, which
  is suspiciously close to a plausible idle floor — **this strongly suggests no idle baseline was
  subtracted in the P1 numbers; do not carry that gap into the new pipeline's script**).

**Practical tool:** `tegrastats` (built-in, what the project already uses) is sufficient and is
what NVIDIA's own docs point to for INA3221 rail readout; `jtop`/`jetson-stats` (already installed
per `jetson-bootstrap.sh`) is a nicer live-viewing/Python-API wrapper over the same sysfs data —
use `jtop`'s Python API (`from jtop import jtop; jetson.stats['Power VDD_IN']` — verify exact key
name on-device, jtop's dict schema varies by jetson-stats version) if you want structured logging
instead of regex-parsing raw `tegrastats` text; either is acceptable, but **pick one and be
consistent**, and prefer jtop's Python API for the new benchmark script since it avoids the fragile
text-parsing the existing `PowerSampler` does.

### 4.5 Thermal soak / throttling detection, 30+ min
- Reuse the existing pattern exactly: `tegrastats --interval 1000`, log for ≥30 min active
  inference (the old soak did 36.2 min active out of 42.5 min wall — some idle gaps are fine, just
  report active-vs-wall duration like `soak-jetson-summary.json` already does).
  Note - Existing artifact's soak was 42.5 min wall / 36.2 min active. Under similar circumstances, run this on the NEW pipeline (detector+transformer running the actual clip loop, not the old backbone+heads).
- **Throttle detection:** TRT/Jetson throttles when junction temp (`tj_c`) approaches ~cutoff (Orin
  Nano soft/hard throttle thresholds are in the 90-105°C range depending on carrier; the existing
  script uses an 80°C conservative `throttle_threshold_c` — keep this conservative threshold, it's
  a reasonable early-warning line even if not the hardware's actual cutoff) AND/OR when
  `tegrastats` reports a clock frequency drop under sustained load (compare GPU/CPU freq fields
  across the log — a genuine throttle event shows freq dropping while load stays high; the existing
  script's `throttle_events: 0` presumably counts freq-drop-under-load events — confirm the exact
  detection logic in whatever script produced `soak-jetson-summary.json` before reusing it
  unmodified for the new pipeline, since detection thresholds/logic weren't in the files read).
- Report: max/mean/min of `cpu_c`, `gpu_c`, `tj_c`, `power_mw`, `gpu_util` over the active window
  (existing schema), plus explicit throttle event count and timestamps if any occur.

## 5. nvpmodel power modes + defensible power-Pareto table

**Confirm actual modes on YOUR device before assuming numbering** — `nvpmodel -p` output differs
by JetPack/L4T revision and whether "Super" firmware is flashed:
- Pre-Super (JetPack 6.0/6.1, 8GB module): **Mode 0 = 15W (default), Mode 1 = 7W** — only 2 modes,
  NOT 3. (Source: NVIDIA JetPack 6 Developer Guide, Orin Nano/NX power page.)
- Post-Super (**JetPack 6.2+**, `jetson-orin-nano-devkit-super.conf` flash config or in-place
  update): adds **25W and MAXN SUPER** modes. The project's `JETSON_QUICKSTART.md` assumes
  "0=MAXN, 1=15W, 2=7W" (3 modes) — this is **only valid if Super mode is actually flashed/enabled**;
  verify JetPack version on the physical device (`cat /etc/nv_tegra_release`) and run
  `sudo nvpmodel -p` (list all profiles) as the FIRST command in the session checklist below —
  do not assume the quickstart's mode-ID mapping is correct until confirmed on-device.
- After any `nvpmodel -m N`, **always run `sudo jetson_clocks`** to lock clocks to the mode's max
  (without it, DVFS may not reach the mode ceiling, making cross-mode comparisons unfair) — already
  done correctly in `jetson-power-sweep.sh`.
- **Power-Pareto table for the thesis:** one row per nvpmodel mode × precision (at minimum FP16;
  add INT8 for detector-only if pursued), columns = accuracy (should be IDENTICAL across power
  modes for a given precision — power mode changes clock/power ceiling, not numerics; if accuracy
  differs across modes something is wrong and needs a control run), FPS, mean VDD_IN watts,
  J/inference, FPS/W, tj_max. Existing `power-sweep-summary.json` already has almost this exact
  shape (`mode, fps, e2e_p50_ms, energy_mj_per_inference` per mode 0/1/2) — extend its schema with
  `watts_mean` (already has it), `accuracy` (missing — add per-mode accuracy check as a control,
  even though it should be invariant), and the new pipeline's detector+transformer numbers instead
  of the old ensemble's.

## 6. Reality check on `deployment/artifacts/projections.md` — cited vs measured

**Confirmed by reading the file directly.** It is explicit about labeling ("This work" vs
"[Ult.2025]"/"[arXiv:2502.15737]") and includes a disclaimer paragraph — this is NOT a case of
fabricated-as-measured, but the table format still risks a reader skimming rows and missing the
Source column. Flags:

| Location | Risk | Required fix |
|---|---|---|
| `projections.md` main table rows 5-8 | Jetson Orin Nano/AGX/NX INT8 latency figures cited from Ultralytics benchmarks and arXiv:2502.15737 sit in the SAME table as "This work" M4 rows — a reader/reviewer skimming the table (not the Source column) could mistake `5.25 ms / 190 FPS` for a measured number from this thesis | Once real Jetson numbers exist (this task), **delete the cited Jetson rows entirely** from any thesis-facing table — cited figures are for a DIFFERENT model (YOLO11s/YOLOv8s COCO detection, backbone-only) and a different task than the unsafe-behaviour pipeline; keep citation-context only in an appendix/footnote if at all (Hard Rule 5: "deployment chapter contains only measured numbers... literature figures... labelled as context and never as our result") |
| `JETSON-DEPLOYMENT.md` "Expected results" table (M4 vs "Jetson Orin Nano FP16 (proj.)" / "Orin AGX FP16 (proj.)") | Explicitly labeled "(proj.)" in header — lower risk, but these are PROJECTIONS derived by scaling M4 CPU numbers by assumed GPU speedup, not citations of another paper; still not our measurement | Replace entire table with the new pipeline's real measured Jetson numbers once benchmarked; delete "Expected results" section, this doc's whole premise ("no physical Jetson available") is now false (already flagged in `plan.md` as stale) |
| `JETSON-DEPLOYMENT.md` header banner "Status (2026-05): scaffolds only — no physical Jetson available" | Stale — directly contradicts brief D2 (full physical access) | Rewrite header; this is Lane B / phase-03 work, not this report's job, but flagging per task scope |
| `deployment/README.md` "Measured results" section | These are real M4 measurements, correctly NOT labeled as Jetson — no issue, but note this file's numbers are for the OLD P1 ensemble pipeline (backbone+direct4/cascade/ensemble heads), not the new headline — a reader could conflate "the deployment chapter's numbers" across pipelines if Ch5 isn't explicit about which pipeline is measured | Ch5 must state explicitly, once, near the top: "all Jetson numbers in this chapter are for pipeline P3 (YOLO26/11 entity detector → RelationTransformer), re-exported and re-measured on [date]; P1 numbers from earlier project phases are superseded and not reused" |
| `benchmarks-jetson-{fp16,fp32,int8}.json`, `soak-jetson-summary.json`, `power-sweep-summary.json`, `multi-stream-jetson-fp16-*.json` | These ARE real Jetson measurements (TensorRT 10.3.0, aarch64, verified) — genuinely measured, not cited — but they measure **pipeline P1**, not P3. Risk is *pipeline* mismatch, not source mismatch | Must be fully re-run on the new pipeline; old files should be renamed/archived (e.g. `*-p1-superseded.json`) rather than overwritten, so the audit trail survives (Hard Rule 2 spirit: report what ran) |
| `deployment/artifacts/asset-checklist.md` row 11 "Jetson projection table ... done (projected, unverified)" | Explicitly marked unverified — good hygiene already, just needs status flip to "measured" once new numbers land, and source repointed from the projection report to the new benchmark JSON | Update status + source column after re-measurement |

**Net: no fabrication found**, but multiple places where cited/projected numbers sit adjacent to
"This work" framing in ways a thesis committee could misread if carried into Ch5 verbatim. Every
number in Ch5 §5.4-5.6 must trace to a JSON produced by the NEW benchmark script on the NEW
pipeline, on the physical Orin Nano, dated after this task.

## 7. Benchmark script outline (implementation-ready)

Single script `tools/jetson_benchmark_v2.py` (new — do not overload `tools/jetson_benchmark.py`,
which is detector-only; extend it or write a sibling that imports its `export_engines`/`bench_one`
where reusable, per DRY).

**Arguments:**
```
--detector-weights   path to .pt (yolo26 or yolo11 finetuned)
--relation-weights   path to trained RelationTransformer state_dict
--data               data.yaml for detector INT8 calibration
--calib-clips        dir of representative video clips/frames for RelationTransformer calibration
--test-clips         path to the 64-clip test set (or k-fold val fold) manifest — same one used
                     for accuracy eval elsewhere, for parity checks
--imgsz              detector input size (match training, e.g. 800)
--precisions         fp32,fp16,int8 (subset allowed; int8 may be detector-only, flag explicitly)
--runs               per-frame timing runs (default 300)
--warmup             default 30
--clip-runs          default: all clips in --test-clips (no subsampling)
--streams            list, e.g. 1 2 3 4, for multi-stream section
--stream-mode        realtime|unbounded (must run BOTH, label outputs distinctly per §4.2)
--stream-duration    seconds per multi-stream config
--queue-depth        bounded queue size for realtime mode (state explicitly)
--soak-duration      seconds, default 1800 (30 min)
--soak-streams       N for soak run (match multi-stream protocol, e.g. N=2)
--power-rail         vdd_in (default) | vdd_cpu_gpu_cv | both
--power-sample-ms    default 100 for per-inference energy, allow override for soak (1000)
--nvpmodel-modes     list, e.g. 0 1 2 — sweep across these, auto jetson_clocks after each switch
--idle-baseline-sec  default 30, measured once per nvpmodel mode before the sweep
--out-dir            deployment/artifacts/jetson-v2/  (new dir, don't clobber P1 artifacts)
```

**Run counts:** per-frame ≥300 (detector), full test-clip set for per-clip (~64 or ~516, no
subsample), multi-stream ≥120s per N per mode (realtime+unbounded), soak ≥1800s active, power
sweep = len(nvpmodel-modes) full FP16 benchmark passes.

**Output JSON schema** (one file per (precision, nvpmodel-mode) combo, plus separate multi-stream/
soak/power files, mirroring existing artifact naming so downstream figure scripts need minimal
changes):
```json
{
  "run_id": "iso-timestamp",
  "pipeline": "P3-yolo26-relation-transformer",
  "hardware": {"platform": "...", "tensorrt_version": "...", "jetpack": "...", "nvpmodel_mode": 0},
  "precision": "fp16",
  "detector": {
    "weights": "...", "params_M": 0.0, "flops_g": 0.0, "size_mb_onnx": 0.0, "size_mb_engine": 0.0,
    "latency_ms": {"mean":0,"p50":0,"p95":0,"p99":0}, "fps": 0.0
  },
  "relation_transformer": {
    "weights": "...", "params_M": 0.0, "flops_g": 0.0, "size_mb_onnx": 0.0, "size_mb_engine": 0.0,
    "seq_len": 129, "latency_ms": {"mean":0,"p50":0,"p95":0,"p99":0}
  },
  "end_to_end": {
    "per_frame_ms": {"p50":0,"p95":0,"p99":0},
    "per_clip_ms": {"p50":0,"p95":0,"p99":0, "n_clips": 64},
    "fps": 0.0
  },
  "accuracy": {"top1": 0.0, "macro": 0.0, "n_clips": 64, "protocol_tag": "same-dist-64|kfold-val"},
  "int8_compression_ratio": null,
  "energy": {
    "rail": "vdd_in", "sample_ms": 100,
    "power_w_total_mean": 0.0, "power_w_idle_mean": 0.0, "power_w_marginal_mean": 0.0,
    "j_per_inference_total": 0.0, "j_per_inference_marginal": 0.0,
    "fps_per_watt_total": 0.0
  },
  "thermal_soak": {
    "active_duration_min": 0.0, "wall_duration_min": 0.0,
    "cpu_c": {"min":0,"mean":0,"max":0}, "gpu_c": {"min":0,"mean":0,"max":0}, "tj_c": {"min":0,"mean":0,"max":0},
    "throttle_events": 0, "throttle_threshold_c": 80
  },
  "multi_stream": {
    "mode": "realtime|unbounded",
    "configs": [{"n_streams":1,"aggregate_fps":0.0,"streams":[{"id":0,"frames_pushed":0,"frames_processed":0,"dropped":0,"queue_depth":0,"head_latency_ms":{"50":0,"95":0,"99":0}}]}]
  }
}
```

## 8. Jetson session checklist, ordered

**Off-device (prep now, no Jetson needed):**
1. Convert `RelationTransformer` export to fixed-shape ONNX (§1), verify with
   `run_check_trt_compat.py`, fix opset/attention issues until clean — do this on the dev machine.
2. Write `tools/jetson_benchmark_v2.py` per §7 schema (code, not run).
3. Verify/regenerate `deployment/artifacts/calib-jetson/` from REAL train clips for the new
   detector+transformer (check current contents aren't stale P1 synthetic-noise batches, §3).
4. Prepare a stratified calibration clip list (8-classes × cameras) and a frozen 64-clip / k-fold
   test manifest to copy to device.
5. Pre-write the idle-baseline, power-sweep, soak, and multi-stream driver logic (can't run, but
   can code+dry-run on CPU with mocked TRT calls).

**On-device, in order (each step gates the next):**
1. `sudo nvpmodel -p` (list real profiles) + `cat /etc/nv_tegra_release` — confirm mode numbering
   and JetPack/Super status BEFORE assuming §5's mapping.
2. rsync new ONNX + calib data + test manifest (extend `jetson-rsync-from-mac.sh`).
3. Build FP16 detector engine + FP16 RelationTransformer engine via `trtexec` (inspect logs for
   silent per-layer FP32 fallback, §1 step 5).
4. Single-frame / single-clip smoke test: run one clip end-to-end, sanity-check the predicted class
   matches the known M4 accuracy-eval output for that clip (correctness gate before any timing).
5. Per-frame + per-clip latency benchmark (§4.1), FP16, MAXN mode — this is the number everything
   else compares against.
6. INT8 calibration + engine build for detector (§3); attempt RelationTransformer INT8 only as a
   secondary experiment, compare accuracy parity before trusting it.
7. Multi-stream realtime (zero-drop) N=1..4 (§4.2) — the headline "real-time multi-stream" claim.
8. Multi-stream unbounded N=1..4 — saturation reference point, clearly labeled non-zero-drop.
9. Idle-baseline power measurement per nvpmodel mode (§4.4), THEN 30-min soak at the chosen
   production N (§4.5) — soak is long, schedule it to run overnight/unattended if possible.
10. nvpmodel power sweep (7W/15W/25W or whatever modes actually exist) with FP16 benchmark +
    jetson_clocks each switch (§5) — restore MAXN/default at the end.
11. rsync all artifacts back to Mac (extend the existing daily-rsync habit in `JETSON_QUICKSTART.md`).

## Unresolved questions

1. Was `deployment/artifacts/calib-jetson/` regenerated for the new detector/pipeline, or is it
   still the P1 synthetic-noise set flagged stale in `JETSON-DEPLOYMENT.md`? Must check on-device
   or via file timestamps/provenance before trusting any INT8 detector number.
2. Exact `nvpmodel -p` mode table on the physical device is unconfirmed (Super firmware status
   unknown) — §5 gives both pre- and post-Super mappings; resolve on first device session.
3. Whether the RelationTransformer's `nn.TransformerEncoderLayer` actually lowers to
   `scaled_dot_product_attention` under the project's pinned PyTorch 2.11 during ONNX export, or
   still uses the legacy explicit-op path — untested; §1's forced-math-attention fix should be
   applied defensively regardless, but confirm via `onnx.helper.printable_graph` after first export
   attempt.
4. Detection logic behind `throttle_events` in the existing `soak-jetson-summary.json` (freq-drop?
   fixed temp threshold? something else?) — the producing script wasn't in the files read; confirm
   before reusing its logic unmodified for the new soak run.
5. Whether DeepStream is truly off the table for the thesis timeline given the "future work" framing
   in §2, or whether a scout should spend 0.5 day confirming Gst-nvinfer can host a custom secondary
   Transformer stage at all (uncertain, not verified against DeepStream docs directly in this pass).
