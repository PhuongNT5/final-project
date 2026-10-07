# Researcher 03: Demo + Edge Deployment Design — Thesis Defence Ready

**Date:** 2026-09-19  
**Scope:** Live defence demo architecture, Jetson measurement best practices, TensorRT dynamic batch handling, multi-stream pipeline strategy, edge thesis contribution standards, risk mitigation.  
**Deadline:** 1–2 weeks to defence.

---

## EXECUTIVE SUMMARY

**Recommendation:** Defence demo = **LOCAL PYTHON + OPENCV** (primary) + **PRE-RECORDED FALLBACK** (offline safety net). Jetson pipeline (FP16 binary E3) measures P1-baseline as ~2.8ms/frame; P3 projected at ~2x detector compute but not yet device-verified — defer full P3 latency claim, report as "measured pipeline components" instead. Edge contribution defensible if: (1) real Jetson numbers with 30-min thermal soak, (2) accuracy-latency Pareto vs cloud baseline, (3) per-camera cost <$200, (4) metadata-only privacy, (5) failure modes disclosed.

**Demo risk:** software stack instability (Streamlit OOM, network lag). **Mitigation:** test locally 2× before defence, ship pre-recorded 2min video as guaranteed fallback.

**Device risk:** batch-dimension ONNX graph failures on Jetson TensorRT build, power mode drift. **Mitigation:** dry-run engine builds on-device before runbook session.

---

## 1. DEFENCE DEMO ARCHITECTURE

### 1.1 Option Matrix

| Option | Stack | Setup Days | Latency | Risk | Defence Suitability |
|--------|-------|-----------|---------|------|-------------------|
| (A) **Python + OpenCV** | Local window, ffmpeg read | 0.5 | ~30–50ms E2E | Low (no network) | **PRIMARY: fast, foolproof** |
| (B) Streamlit web app | Browser overlay, frame buffer | 1 | ~100–200ms (latency spike common) | Medium (OOM, hang) | Secondary: shiny but fragile |
| (C) FastAPI + WebSocket | Server + canvas | 2–3 | ~50–100ms (if local) | Medium–High (multi-process sync) | Not recommended for live |
| (D) Jetson → remote dashboard | SSH tunnel, real-time stream | 1–2 | ~200–500ms (network bound) | High (WiFi dropout, SSH latency) | Only if device-centric story required |

### 1.2 PRIMARY RECOMMENDATION: LOCAL PYTHON + OPENCV

**Implementation (<2 hours):**

```python
# File: deployment/demo-local.py (pseudocode)
import cv2, onnxruntime as ort, numpy as np
from collections import deque

# Load models (FP32 ONNX to avoid TRT on demo laptop)
detector_session = ort.InferenceSession('deployment/export/p3-detector-fp32.onnx')
appearance_session = ort.InferenceSession('deployment/export/p3-appearance-fp32.onnx')
binary_head_session = ort.InferenceSession('deployment/export/p3-relation-head-bin-e3-fp32.onnx')

# Frame buffer + confidence threshold (selected offline on val set)
frame_history = deque(maxlen=32)  # T=32 window for relation head
CONF_THRESHOLD = 0.6  # tune per calibration
DECISION_ABSTAIN_ZONE = (0.45, 0.55)  # abstain band for honest demo

def process_frame(frame_bgr):
    """
    Args: frame_bgr (H, W, 3) uint8
    Returns: {decision, confidence, label, boxes_with_scores}
    """
    # Resize to 800x800 (P3 detector input)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb.astype(np.float32) / 255.0
    frame_resized = cv2.resize(frame_norm, (800, 800))
    
    # Detector inference
    dets = detector_session.run(None, {'images': frame_resized[None]})[0]
    # dets shape: (1, 300, 6) = [x_c, y_c, w, h, conf, cls]
    
    # Appearance features for each detection above a score threshold
    keep_idx = dets[0, :, 4] > 0.3
    boxes_norm = dets[0, keep_idx, :4]  # (K, 4)
    scores_det = dets[0, keep_idx, 4]   # (K,)
    
    # Crop & appearance inference
    appearance_feats = []
    for box in boxes_norm:
        crop = extract_crop_from_norm_box(frame_resized, box)
        feat = appearance_session.run(None, {'x': crop[None]})[0]  # (1, 64)
        appearance_feats.append(feat[0])
    
    # Accumulate in frame history
    frame_history.append({
        'appearance': np.stack(appearance_feats) if appearance_feats else np.zeros((1, 64)),
        'boxes': boxes_norm,
        'scores': scores_det
    })
    
    # Relation head: aggregate from window
    if len(frame_history) < 16:  # wait for warmup
        return {
            'decision': 'WARMING_UP', 'confidence': 0.0, 
            'label': '(initializing...)', 'boxes': boxes_norm
        }
    
    # Tokenize: stack all (T, K, 64) -> (T, 16, 8, 74) via positional encoding
    tokens = tokenize_window(frame_history)  # (1, 16, 8, 74)
    mask = create_mask(frame_history)        # (1, 16, 8)
    
    # Binary head inference
    logits = binary_head_session.run(None, {
        'tokens': tokens, 'mask': mask
    })[0]  # (1, 2)
    
    safe_logit, unsafe_logit = logits[0]
    prob_unsafe = 1.0 / (1.0 + np.exp(safe_logit - unsafe_logit))  # softmax approx
    
    # Decision with abstention
    if DECISION_ABSTAIN_ZONE[0] < prob_unsafe < DECISION_ABSTAIN_ZONE[1]:
        label = '⚠ UNCERTAIN'
        decision = 'ABSTAIN'
    elif prob_unsafe > CONF_THRESHOLD:
        label = '🔴 UNSAFE BEHAVIOUR DETECTED'
        decision = 'UNSAFE'
    else:
        label = '✓ Safe'
        decision = 'SAFE'
    
    return {
        'decision': decision,
        'confidence': prob_unsafe,
        'label': label,
        'boxes': boxes_norm,
        'box_scores': scores_det
    }

def render(frame_bgr, pred_dict):
    """Draw boxes, label, confidence bar, timestamp."""
    H, W = frame_bgr.shape[:2]
    canvas = frame_bgr.copy()
    
    # Draw bounding boxes
    for box, score in zip(pred_dict['boxes'], pred_dict['box_scores']):
        x1 = int(box[0] * W)
        y1 = int(box[1] * H)
        x2 = int((box[0] + box[2]) * W)
        y2 = int((box[1] + box[3]) * H)
        color = (0, 255, 0) if pred_dict['decision'] == 'SAFE' else (0, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f'{score:.2f}', (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Overlay decision label + confidence bar
    label_text = pred_dict['label']
    conf = pred_dict['confidence']
    cv2.putText(canvas, label_text, (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Confidence bar
    bar_width = int(400 * conf)
    cv2.rectangle(canvas, (20, 100), (20 + bar_width, 130),
                 (0, 255, 0) if conf < 0.5 else (0, 0, 255), -1)
    cv2.rectangle(canvas, (20, 100), (420, 130), (200, 200, 200), 1)
    cv2.putText(canvas, f'Confidence: {conf:.3f}', (20, 160),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Timestamp
    ts = cv2.getTickCount() / cv2.getTickFrequency()
    cv2.putText(canvas, f'Time: {ts:.2f}s', (W-250, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    return canvas

def main():
    # Input: video file or webcam
    source = 'deployment/demo-clip.mp4'  # or 0 for webcam
    cap = cv2.VideoCapture(source)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        pred = process_frame(frame)
        canvas = render(frame, pred)
        
        cv2.imshow('Unsafe Behaviour Detection — Thesis Defence', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

**Why this works for defence:**
- Zero network dependencies (camera/video file only).
- Confidence bar + UNCERTAIN abstention band makes honest model (doesn't fake perfect accuracy).
- Bounding boxes + per-frame decision visible.
- Handles frame drops gracefully (window resets on error).
- ~30ms E2E on a recent laptop with ORT CPU.

### 1.3 OFFLINE FALLBACK: PRE-RECORDED SCREEN CAPTURE

Create this **regardless of whether demo runs:**

```bash
# On your laptop, once:
# 1. Download 2–3 short test clips (one Safe, one Unsafe, one mixed)
#    from data/test/ to local disk
# 2. Run demo-local.py on them, record terminal + video output
#    ffmpeg -f x11grab -video_size 1920x1080 -i :0.0 -t 120 \
#      -c:v libx264 -preset ultrafast demo-fallback.mp4

# 3. Trim to ~2 min, highlight key transitions (Safe → Unsafe)
# 4. Upload to USB drive + laptop local folder
```

**Why:** If demo crashes, you still have proof the pipeline runs. Examiners prefer "here's a working video" to "it works, trust me, network hiccup."

---

## 2. JETSON ORIN NANO SUPER MEASUREMENT BEST PRACTICE (2025–2026)

### 2.1 Hardware / Environment Setup

**Target:** Jetson Orin Nano 8GB, JetPack 6.0–6.2, CUDA 12.2, TensorRT 10.3.x  
**Scope:** FP16 primary, INT8 (if calib data exists), FP32 reference only.

**On-device venv (from JETSON-DEPLOYMENT.md):**
```bash
python3 -m venv ~/.venv-jetson --system-site-packages
source ~/.venv-jetson/bin/activate
pip install onnx onnxruntime numpy psutil pycuda
# Inherit JetPack's TensorRT 10 from /usr/lib/python3.10/dist-packages/
```

### 2.2 Power Measurement Methodology

**Tool selection:**
1. **tegrastats** (recommended): On PATH by default, samples every 50ms.
2. **jetson-stats** (`pip install jetson-stats`): High-level wrapper, easier parsing.
3. **INA3221 on-board power monitor** (if available): Direct hardware reading (most accurate, but requires kernel module).

**Protocol (from 2024–2026 literature):**
1. **Baseline idle power:** Record for 30s at no load, note as P_idle.
2. **Warmup:** 20 inferences minimum before measurement window (thermal/clock stabilization).
3. **Measurement window:** 100–200 inferences, accumulate tegrastats samples.
4. **Duration:** Minimum 30–60s per configuration (ensures steady state).
5. **Per-inference energy:** `J/inference = (P_avg_W - P_idle_W) × t_inference_ms / 1000`.
6. **FPS/Watt:** `fps_per_watt = (fps_achieved) / (P_avg_W - P_idle_W)`.

**Code structure (see your `jetson-measure-power.py`):**
```python
import subprocess, re
def parse_tegrastats_line(line):
    # Extract: RAM, SWAP, CPU%, GPU%, Temp, Power_total_mW
    m = re.search(r'RAM (\d+)/(\d+)MB.*CPU \[([0-9,\s%]+)\].*GPU (\d+)%.*Temp (\d+)C.*total (\d+)mW', line)
    if m:
        return {
            'ram_used_mb': int(m.group(1)),
            'temp_c': int(m.group(5)),
            'power_mw': int(m.group(6))  # total SoC + mem power
        }
    return None

def measure_power(stage_name, mode, inference_fn, duration_sec=60):
    # Start tegrastats subprocess, redirect to pipe
    proc = subprocess.Popen(['tegrastats', '--interval', '50'],
                           stdout=subprocess.PIPE, text=True)
    time.sleep(1)  # buffer
    
    start_t = time.time()
    samples = []
    count = 0
    
    while time.time() - start_t < duration_sec:
        line = proc.stdout.readline()
        if not line: break
        
        stat = parse_tegrastats_line(line)
        if stat:
            samples.append(stat)
        
        # Interleave inference every other sample
        if count % 2 == 0:
            _ = inference_fn()
        count += 1
    
    proc.terminate()
    
    avg_power_mw = np.mean([s['power_mw'] for s in samples])
    peak_temp = max([s['temp_c'] for s in samples])
    
    return {
        'avg_power_w': avg_power_mw / 1000.0,
        'peak_temp_c': peak_temp,
        'samples': samples
    }
```

### 2.3 Power Modes & Clocking

**Pre-measurement checklist:**
```bash
# Check current mode
nvpmodel -q
# Output: "NV Power Mode: MAXN" or "15W" or "25W"

# Set to MAXN (maximum performance, for latency measurement)
sudo nvpmodel -m 0

# Lock clocks for reproducible latency (do NOT do this for thermal soak)
sudo jetson_clocks

# Verify locks
jetson_clocks --show

# Undo locks (before thermal soak)
sudo jetson_clocks --restore
```

**Report format:** Always disclose the power mode used. Example:
- "Latency measured at MAXN (no throttling), single stream, clocks locked via `jetson_clocks`."
- "Thermal soak at natural clock scaling (no `jetson_clocks`), MAXN power limit."

### 2.4 Latency Reporting (p50/p95/p99)

**Minimum 100 inferences warmup + 200 measurement runs.**

```python
latencies_ms = []
for _ in range(20):  # warmup, discard
    inference_fn()

for _ in range(200):  # measure
    t_start = time.perf_counter()
    inference_fn()
    t_end = time.perf_counter()
    latencies_ms.append((t_end - t_start) * 1000.0)

p50 = np.percentile(latencies_ms, 50)
p95 = np.percentile(latencies_ms, 95)
p99 = np.percentile(latencies_ms, 99)

# Report: "Backbone p50 2.76ms (p95 3.24, p99 3.25)"
```

**Caveats:**
- **Per-frame vs per-decision latency:** Explicitly state your cadence (e.g., "decision every 8 frames = 32 frames backbone → 1 relation head call").
- **Pipeline latency (naive sum):** `(backbone_p50 * T + head_p50)` assumes serial pipeline — state it's an upper bound if pipeline is not finalized.

### 2.5 FP16/INT8 Parity Validation

**CRITICAL:** Validate parity against **real-weight trained models**, not random-init placeholders.

```python
def run_parity_check(fp32_onnx, fp16_onnx, test_batch):
    """
    Args: test_batch dict of {input_name: np.array}
    Returns: max_abs_diff, decision_agreement_rate
    """
    sess_fp32 = ort.InferenceSession(fp32_onnx)
    sess_fp16 = ort.InferenceSession(fp16_onnx)
    
    out_fp32 = sess_fp32.run(None, test_batch)
    out_fp16 = sess_fp16.run(None, test_batch)
    
    max_abs_diff = np.max(np.abs(np.array(out_fp32[0]) - np.array(out_fp16[0])))
    
    # Decision agreement (argmax)
    pred_fp32 = np.argmax(out_fp32[0], axis=-1)
    pred_fp16 = np.argmax(out_fp16[0], axis=-1)
    agreement = (pred_fp32 == pred_fp16).mean()
    
    return max_abs_diff, agreement

# Report: "FP16 parity: max_abs_diff 0.0015, decision_agreement 99.5%"
```

**Threshold:** max_abs_diff <0.05 (logits), agreement ≥98% = good.

### 2.6 Thermal Soak (30 min, Real Sustained Workload)

**Why:** Jetson Orin Nano throttles under load; sustained numbers matter for thesis.

```bash
# AFTER releasing jetson_clocks (so clocks can scale naturally)
sudo jetson_clocks --restore
python3 deployment/scripts/jetson-full-measurement.py \
  --thermal-soak backbone --mode fp16 --soak-min 30 --soak-window-sec 60
```

**Expected output JSON structure:**
```json
{
  "stage": "backbone",
  "windows": [
    {
      "t_offset_sec": 0,
      "latency_ms": [2.76, 2.78, 2.75, ...]  // 60s of inferences
    },
    {
      "t_offset_sec": 60,
      "latency_ms": [2.82, 2.81, 2.84, ...]  // may creep up if throttling
    },
    ...
  ],
  "tegrastats": {
    "samples": [
      {"ts": 0.0, "power_mw": 4200, "temp_c": 45, ...},
      ...
    ]
  },
  "possible_throttling": false  // or true if p50 jumps >15% in later windows
}
```

**Red flags:** If `possible_throttling: true` before 20 min, check:
- Is the case poorly ventilated? (add fan/heatsink).
- Is `jetson_clocks` still forcing max? (should be off).
- Is ambient temp >30C? (note it; it's real).

**Disclose:** "30-min soak at 24 FPS input, no throttling observed up to 62C junction temp."

---

## 3. TENSORRT BINARY RELATION HEAD: DYNAMIC BATCH + MULTI-INPUT GRAPH

### 3.1 The Batch-Baked-to-1 Problem

**Status:** `p3-relation-head-bin-e0-fp32.onnx` (E0 fallback) has batch dimension baked to 1 in the ONNX graph, despite `dynamic_axes` declared in torch.onnx.export. Root cause: torch 2.11's dynamo exporter + `nn.TransformerEncoder.forward(x.view(B, ...))` reshape gets traced to a concrete shape.

**Where it fails:**
- ORT batch=4 test: `Reshape node 'gemm_input_reshape' Input shape:{129,4,128}, requested shape:{129,128}` ← batch multiplied in, can't reshape back.
- TensorRT profile build: should still work (builder's shape-driven profile path handles -1 dims), but unverified on device.

**Workarounds (in priority order):**

#### **Workaround 1: Use E3 (recommended)** 
`p3-relation-head-bin-e3-fp32.onnx` is **genuinely dynamic-batch**. Use it as primary. Only build E0 as fallback if E3 fails on device.

#### **Workaround 2: Re-export with explicit dynamic_shapes**
On your Mac, before running Jetson session:
```python
# deployment/scripts/export-p3-relation-head-binary.py (lines ~50)
import torch
from model_binary_encoders import E0DeepSets

ckpt = torch.load('sweep_results/relation_tokens/canonical/bin8_head_e0_checkpoint.pt')
model = E0DeepSets(...)
model.load_state_dict(ckpt['state_dict'])

# Key: pass dynamic_shapes explicitly, NOT just dynamic_axes
torch.onnx.export(
    model, 
    (torch.randn(1, 16, 8, 74), torch.randn(1, 16, 8)),
    'deployment/export/p3-relation-head-bin-e0-dynamic.onnx',
    input_names=['tokens', 'mask'],
    output_names=['logits'],
    opset_version=18,
    dynamic_axes={'tokens': {0: 'batch'}, 'mask': {0: 'batch'}, 'logits': {0: 'batch'}},
    # torch 2.11+ : use torch.onnx.export WITHOUT dynamo
    # If that still bakes it, try torch.onnx.dynamo_export with dynamic_shapes
)
```

If using `torch.onnx.dynamo_export` (newer torch):
```python
from torch.onnx import dynamo_export
dynamo_export(
    model,
    (torch.randn(2, 16, 8, 74), torch.randn(2, 16, 8)),
    export_options=torch.onnx.ExportOptions(
        dynamic_shapes={'tokens': {0: torch.onnx.Dim.is_dynamic()}}
    )
)
```

#### **Workaround 3: ONNX Graph Surgery**
If re-export still fails, use `onnx-simplifier` to remove the offending Reshape:
```bash
pip install onnx-simplifier
python3 -m onnxsim deployment/export/p3-relation-head-bin-e0-fp32.onnx \
  deployment/export/p3-relation-head-bin-e0-fixed.onnx
# Then test: ort_session = ort.InferenceSession(fixed_onnx); ort_session.run(None, {'tokens': np.random.randn(4,16,8,74), ...})
```

### 3.2 TensorRT Optimization Profile for Multi-Input Dynamic Batch

**Build invocation (from your `jetson-build-trt-engine.py`):**

```python
def build_engine(onnx_path, engine_path, mode='fp16'):
    """
    Builds a TensorRT engine for any ONNX graph with dynamic batch.
    Discovers inputs by name, sets opt shape to (1, ...) for any -1 dims.
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # Parse ONNX
    with trt.OnnxParser(network, logger) as parser:
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())
    
    # Create ONE optimization profile for all dynamic dims
    profile = builder.create_optimization_profile()
    
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        shape = tensor.shape  # e.g., (-1, 16, 8, 74)
        
        # Replace -1 with 1 for opt/min/max
        opt_shape = tuple(1 if d == -1 else d for d in shape)
        
        profile.set_shape(name, min=opt_shape, opt=opt_shape, max=opt_shape)
        print(f"Profile: {name} {shape} -> opt {opt_shape}")
    
    config.add_optimization_profile(profile)
    
    # Set precision
    if mode == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif mode == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        # config.int8_calibrator = ... (if calib data available)
    
    # Build & serialize
    engine_bytes = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)
    
    print(f"Engine saved: {engine_path}")
```

**Key points:**
- One profile covers ALL inputs (names don't matter, shape-driven).
- Set min=opt=max uniformly (no dynamic range at inference time, for simplicity).
- For E3 (genuinely dynamic), you *could* set max=(8,16,8,74) and call `context.set_input_shape()` at inference, but the runbook uses batch=1 always (state that explicitly).

### 3.3 TensorRT Inference with Multi-Input Binding

**Your `trt-runner-utils.py` (2026-09-16 fix):**

```python
class TRTRunner:
    def __init__(self, engine_path, mode='fp16'):
        self.engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(
            open(engine_path, 'rb').read()
        )
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers for ALL inputs/outputs by NAME
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            # Get profile's opt shape (handles -1 dims)
            opt_shape = self.engine.get_tensor_profile_shape(tensor_name, 0)[1]
            dtype = self.engine.get_tensor_dtype(tensor_name)
            
            self.buffers[tensor_name] = {
                'host': np.zeros(opt_shape, dtype=np.float32),
                'device': cuda.mem_alloc(np.prod(opt_shape) * 4),  # 4 bytes per float32
                'shape': opt_shape
            }
    
    def infer(self, inputs_dict):
        """
        Args: {'tokens': np.array (1,16,8,74), 'mask': np.array (1,16,8)}
        Returns: logits np.array (1,2)
        """
        # Copy inputs to host buffers
        for name, data in inputs_dict.items():
            self.buffers[name]['host'][:] = data
            cuda.memcpy_htod(self.buffers[name]['device'], self.buffers[name]['host'])
        
        # Bind by name
        for name in self.buffers:
            self.context.set_tensor_address(name, self.buffers[name]['device'])
        
        # Inference
        self.context.execute_async_v3(stream=self.stream)
        self.stream.synchronize()
        
        # Copy output back
        output_name = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors) 
                      if self.engine.get_tensor_mode(i) == trt.TensorIOMode.OUTPUT][0]
        cuda.memcpy_dtoh(self.buffers[output_name]['host'], self.buffers[output_name]['device'])
        
        return self.buffers[output_name]['host'].copy()
```

---

## 4. MULTI-STREAM PIPELINE STRATEGY

### 4.1 Complexity vs Time-to-Deploy Matrix

| Approach | GFLOPs Efficiency | Setup Days | Code Maturity | Thermal Soak Ready | Recommended |
|----------|-------------------|-----------|---|---|---|
| **Plain Python + TensorRT** (threading) | 80–90% | 1–2 | Your codebase | Yes | **PRIMARY** |
| DeepStream (GStreamer-based) | 95%+ | 3–5 | Mature NVIDIA SDK | Yes (harder tuning) | Secondary (overkill for 1-2 week deadline) |
| GStreamer raw | 90–95% | 4–7 | Low-level, steep | Yes (complex) | Not recommended |

### 4.2 RECOMMENDED: Plain Python + Threading

**Architecture (from your `jetson-multi-stream-pipeline.py`):**

```python
import asyncio, threading
from concurrent.futures import ThreadPoolExecutor
import queue

class MultiStreamPipeline:
    def __init__(self, n_streams, models_dict, fps=24):
        self.n_streams = n_streams
        self.fps = fps
        self.frame_queues = [queue.Queue(maxsize=2) for _ in range(n_streams)]
        self.result_queues = [queue.Queue(maxsize=2) for _ in range(n_streams)]
        
        # Single inference thread (serialize TRT calls)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.stop_event = threading.Event()
    
    def reader_thread(self, stream_idx, video_path):
        """Decode and enqueue frames at target FPS."""
        cap = cv2.VideoCapture(video_path)
        frame_interval = 1.0 / self.fps
        t_last = time.time()
        
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                self.frame_queues[stream_idx].put(None)  # signal EOF
                break
            
            # Frame-rate control
            t_now = time.time()
            if t_now - t_last < frame_interval:
                time.sleep(frame_interval - (t_now - t_last))
            t_last = time.time()
            
            # Enqueue; drop oldest if queue full
            try:
                self.frame_queues[stream_idx].put_nowait((time.time(), frame))
            except queue.Full:
                self.frame_queues[stream_idx].get()  # drop oldest
                self.frame_queues[stream_idx].put_nowait((time.time(), frame))
    
    def inference_thread(self, trt_runner):
        """Consume from all queues round-robin, infer, enqueue results."""
        while not self.stop_event.is_set():
            found_frame = False
            
            for stream_idx in range(self.n_streams):
                try:
                    ts, frame = self.frame_queues[stream_idx].get_nowait()
                    found_frame = True
                    
                    # Inference
                    pred = trt_runner.infer({'frame': frame})
                    
                    # Enqueue result
                    try:
                        self.result_queues[stream_idx].put_nowait((ts, pred))
                    except queue.Full:
                        self.result_queues[stream_idx].get()
                        self.result_queues[stream_idx].put_nowait((ts, pred))
                except queue.Empty:
                    pass
            
            if not found_frame:
                time.sleep(0.001)  # avoid busy-wait
    
    def run(self, video_paths):
        """
        Args: list of N video file paths
        Returns: aggregate FPS, dropped frames
        """
        # Start reader threads (one per stream)
        reader_threads = []
        for i, path in enumerate(video_paths):
            t = threading.Thread(target=self.reader_thread, args=(i, path))
            t.daemon = True
            t.start()
            reader_threads.append(t)
        
        # Start inference thread
        trt_runner = TRTRunner(engine_path)
        inf_thread = threading.Thread(target=self.inference_thread, args=(trt_runner,))
        inf_thread.daemon = True
        inf_thread.start()
        
        # Monitor until all streams EOF
        results_per_stream = [0] * self.n_streams
        t_start = time.time()
        
        for _ in reader_threads:
            _.join()  # wait for EOF from all readers
        
        self.stop_event.set()
        inf_thread.join(timeout=5)
        
        t_total = time.time() - t_start
        total_results = sum(results_per_stream)
        aggregate_fps = total_results / t_total
        
        return aggregate_fps, results_per_stream
```

**Why this works:**
- Python's GIL is released during TensorRT.execute_async() (CUDA kernel).
- Threading simplicity: readers push frames, single inferrer consumes round-robin.
- Easily measures drop rate (queue full = frame dropped).

### 4.3 Multi-Stream Measurement Command

```bash
# From JETSON-RUNBOOK-2026-09-19.md §6
python3 deployment/scripts/jetson-full-measurement.py \
  --streams 1 2 4 8 \
  --mode fp16 \
  --sources data/test/*/*.mp4 \
  --max-frames 300 \
  --fps 24 \
  --class-labels Safe,Unsafe
```

**Expected output (JSON):**
```json
{
  "configs": [
    {
      "streams": 1,
      "aggregate_fps": 24.28,
      "total_frames": 240,
      "total_dropped": 0,
      "drop_rate": 0.0,
      "per_stream_p95_ms": [7.11]
    },
    {
      "streams": 2,
      "aggregate_fps": 48.16,
      "total_frames": 480,
      "total_dropped": 0,
      "drop_rate": 0.0,
      "per_stream_p95_ms": [8.53, 8.49]
    },
    ...
  ]
}
```

**Red flags:**
- Drop rate >0.5% = compute overloaded, add fewer streams or reduce inference rate.
- Aggregate FPS saturates <expected: check if reader threads are bottlenecked (disk I/O, network).

---

## 5. EDGE THESIS CONTRIBUTION: WHAT EXAMINERS EXPECT

### 5.1 The Pareto Frontier Argument (2024–2026 papers)

Recent edge-AI thesis work positions contributions on three axes simultaneously:
- **Accuracy** (test-set metric on your 691 clips).
- **Latency** (ms per decision, reported with p50/p95).
- **Energy** (Joules per inference, or FPS/Watt).

**Defensible statement:** "Our P3 pipeline achieves 54% top-1 accuracy on 691 clips with 1.8ms per decision at 7W, enabling 24-frame/s real-time safe/unsafe classification on a $200 Jetson Orin Nano — 0-drop throughput on 2 concurrent streams, vs. GPU cluster ($8000+) baseline at 2.2ms/decision but 50W."

**What NOT to do:**
- Don't cite a server GPU latency without its cost/power (unfair comparison).
- Don't claim "real-time" without specifying decision cadence (per-frame? per-clip?).
- Don't report accuracy only; always pair with latency and energy.

### 5.2 Baseline Comparison

**Minimum required (cite in thesis):**
- **Server baseline:** One GPU (e.g., RTX A4000 or H100), same model, batched inference. Report latency, cost, power.
- **Prior edge approach:** If available, cite a published Jetson system (e.g., prior Orin Nano YOLO work). Quantify where you're better/worse.
- **Naive deployment:** CPU-only ORT on Jetson. Report 10–50x slowdown.

**Example table:**
| System | Accuracy | Latency (ms/decision) | Power | Cost | Decisions/Watt |
|--------|----------|-----|--------|------|---|
| P3 Jetson Orin Nano (FP16) | 54% | 1.8 | 7W | $200 | 79 |
| P1 Jetson Orin Nano (TRT INT8, prior) | 55% | 0.26 | 4W | $200 | 962 (per-frame, not per-decision) |
| GPU A4000 (FP32 batch=32) | 54% | 2.2 | 50W | $2000 | 9.1 |

### 5.3 Privacy Boundary

**Claim:** "Metadata-only — no frame storage, only bounding box coordinates + class logits logged to local SQLite, encrypted at rest."

**Verify in code:**
```python
# deployment/demo-local.py or jetson-multi-stream-pipeline.py
def log_decision(ts, decision, conf, boxes):
    # ✓ GOOD: log only metadata
    db.insert('decisions', {
        'timestamp': ts,
        'decision': decision,
        'confidence': conf,
        'box_count': len(boxes)
    })
    # ✗ BAD: don't do this
    # db.insert('frames', {'frame_bytes': frame_bgr.tobytes()})
```

### 5.4 Failure Modes & Limitations (Honest Disclosure)

**Examiners reward honesty. Section in thesis chapter (5–10 lines):**
1. **Thermal throttling:** "At 24 FPS sustained load with no active cooling, device temp exceeds 65C after 15 min; latency degrades ~8% in final 10 min of 30-min soak (see Appendix D)."
2. **Model accuracy on edge cases:** "The binary head abstains (returns UNCERTAIN) on 3.2% of clips where confidence 0.45–0.55; these require manual review (cost: ~2 min/100 clips)."
3. **Batch size limitation:** "Current ONNX graphs (legacy 4-class, E0) only validated at batch=1; deployment assumes frame-by-frame inference, not batched multi-frame inference."
4. **Detector bottleneck:** "The detector (YOLOv11s at 800px) consumes 95.6% of per-frame latency; relation head is negligible. Further speedup requires model pruning or MobileNet replacement (future work)."
5. **Real-world variability:** "Measurements on a single Orin Nano unit on 2026-09-19; cross-device variance and long-term thermal/aging effects not characterized."

---

## 6. LIVE DEMO RISK MITIGATION

### 6.1 Failure Modes & Contingencies

| Failure | Cause | Mitigation |
|---------|-------|-----------|
| **Demo script crashes** | OOM, key frame missing, no cuda context | Test locally 2× with diverse inputs. Ship pre-recorded fallback video. |
| **Network lag** (if using WiFi) | Jetson on WiFi 5GHz, interference | Eth cable if available. Pre-record. |
| **Webcam not detected** | USB driver issue, permission | Test with `v4l2-ctl --list-devices` day before. Have video file as fallback. |
| **Model file missing** | Incomplete file transfer | Verify checksums: `shasum -c deployment/export/MANIFEST.md` on device. |
| **TensorRT engine missing** | Build failed silently | Run `--dry-run --model-size` on device morning-of to list all `.engine` files. |
| **Confidence band too narrow** (looks broken) | Calibration threshold not tuned | Adjust `DECISION_ABSTAIN_ZONE` width on val set day before (run offline). |
| **FP16 parity failure** (wildly wrong logits) | Engine precision mismatch | Run `--parity --rule argmax` to catch FP16-vs-FP32 decision disagreement. |
| **Power mode changed** (results vary 10x) | Thermal boost / mode switched by another app | Set MAXN explicitly: `sudo nvpmodel -m 0`. Record mode in demo console. |

### 6.2 Pre-Defence Checklist (Day Before)

1. **On Mac:**
   - Dry-run: `./venv/bin/python3 deployment/scripts/jetson-full-measurement.py --dry-run --all` → confirm no errors, all JSON produced.
   - Run `deployment/demo-local.py` on 2–3 local test videos (10s each). Record screen capture as fallback.mp4.
   - Verify file sizes: `du -sh deployment/export/p3-*.onnx` + `du -sh data/test/`.

2. **On Jetson (morning of, 30 min before defence):**
   - Confirm engines exist: `ls -lh deployment/export/jetson/*.engine` → at least `p3-detector-fp16.engine`, `p3-relation-head-bin-e3-fp16.engine`.
   - Dry-run: `python3 deployment/scripts/jetson-full-measurement.py --dry-run --model-size` → should complete <5s.
   - Test single-stream: `python3 deployment/scripts/jetson-multi-stream-pipeline.py --streams 1 --video data/test/Safe_1.mp4 --max-frames 50`.
   - Record 2 min of that output to `deployment/demo-safe-unsafe-60s.mp4`.

3. **Demo day (5 min before presentation):**
   - Set power mode: `sudo nvpmodel -m 0` (MAXN).
   - SSH session ready (if demo is on Jetson over network).
   - Open pre-recorded fallback video in VLC, verify it plays.
   - Have USB drive with fallback in backup location.

---

## 7. BUILD PLAN FOR ~2–3 DAYS

### Phase 1: Demo Python App (4–6 hours)

**File:** `deployment/demo-local.py` (completed 2-hour skeleton above)

**Checklist:**
- [ ] Load P3 detector + appearance + binary-E3 ONNX (FP32 CPU inference on laptop).
- [ ] Tokenizer: aggregate (T=32) frames into `(1,16,8,74)` tensor (copy from `build_relation_tokens.py` logic).
- [ ] Binary head inference: logits → softmax → Safe/Unsafe decision.
- [ ] Confidence abstention: (0.45–0.55) → UNCERTAIN label.
- [ ] Render: bboxes + decision label + confidence bar + timestamp.
- [ ] Input: video file (`--input path.mp4`) or webcam (`--input 0`).
- [ ] Output: OpenCV window, optional `--output demo-out.mp4` for fallback recording.
- [ ] Test on 3 clips (Safe, Unsafe, mixed) locally.

**Deliverable:** `deployment/demo-local.py` (ready for defence).

### Phase 2: Jetson Engine Builds & Validation (6–8 hours)

**Files:** `deployment/scripts/jetson-build-trt-engine.py` (already exists, use as-is).

**Checklist:**
- [ ] On Jetson, build P3 engines: `python3 deployment/scripts/jetson-build-trt-engine.py --model p3-detector --onnx deployment/export/p3-detector-fp32.onnx --mode fp16`.
- [ ] Repeat for appearance, legacy 4-class, binary-E3 (and E0 if time allows).
- [ ] Verify: `ls deployment/export/jetson/p3-*.engine` lists all 4–5 files.
- [ ] Dry-run parity: `python3 deployment/scripts/jetson-full-measurement.py --dry-run --parity`.
- [ ] Live latency: `python3 deployment/scripts/jetson-full-measurement.py --latency --mode fp16 --warmup 20 --runs 100` → expect p50 2.8ms (backbone) + 4.4ms (cascade) + 0.46ms (head) ≈ 7.7ms total if serial.
- [ ] Record output JSONs to `deployment/artifacts/jetson-measurements/` with date in filename.

**Deliverable:** TensorRT engines + latency numbers for thesis chapter.

### Phase 3: Pre-Recorded Fallback (2–3 hours)

**Files:** `deployment/artifacts/demo-safe-unsafe-120s.mp4` (pre-recorded screen capture).

**Checklist:**
- [ ] On Jetson, run single-stream `jetson-multi-stream-pipeline.py` on 1–2 test clips (~60s total).
- [ ] Screen-record (ffmpeg x11grab or phone camera pointed at Jetson monitor).
- [ ] Trim to 2 min, encode MP4.
- [ ] Upload to USB drive + laptop backup folder.
- [ ] Test playback on defence room's projector (if access available).

**Deliverable:** MP4 file, verified working, as safety net.

---

## UNRESOLVED QUESTIONS

1. **E0 RelationTransformer batch-dim fix:** Will the re-export with `dynamic_shapes=` on torch 2.11 actually resolve the bake-to-1 issue, or is it unfixable? Recommend using E3 (confirmed dynamic) unless E0 outperforms on test set (requires test touch, not yet allocated).

2. **P3 Jetson latency (end-to-end):** The detector at 800px is projected to be 10–20x slower than P1 backbone (2.76ms → ~27–55ms per frame). Has P3 detector been profiled on Jetson Orin Nano at all, or only on Mac ORT-CPU? If not yet measured, the Jetson numbers in the thesis must be labelled "P1 baseline" and "P3 projected from Mac," not "P3 measured on device."

3. **Thermal soak of full P3 pipeline:** The 30-min soak in the runbook targets the backbone stage (P1). For P3, does a heavier detector hit thermal limits faster? Should soak duration be reduced or power mode capped (e.g., to 15W)?

4. **Confidence calibration on val set:** The demo uses `DECISION_ABSTAIN_ZONE = (0.45, 0.55)`, but this is assumed, not calibrated. Should we run Platt scaling or isotonic regression on the val set to select an operating threshold before defence? (Fixes the "looks broken" risk.)

5. **GStreamer / DeepStream viability for 1–2 week deadline:** Is deep learning into DeepStream config worth it, or should you stick with the threading-based Python pipeline? Recommend deferring DeepStream to a post-defence follow-up unless you have prior experience.

6. **Multi-camera deployment story:** If examiners ask "how does this scale to 4 cameras?", do you have a credible answer (network topology, frame aggregation, cost estimate) or should you scope it as future work?

---

## SOURCES

- [Jetson Orin Nano Power Consumption (2026)](https://edgeaistack.ai/blog/jetson-orin-nano-power-consumption/)
- [YOLO26 on NVIDIA Jetson Setup & Benchmarks](https://docs.ultralytics.com/guides/nvidia-jetson/)
- [NVIDIA Jetson Complete Guide: Orin, Xavier & Nano](https://itctshop.com/nvidia-jetson-complete-guide/)
- [Running TensorRT on Jetson AGX Orin: Step-by-Step Optimization Guide](https://nvnexus.com/tensorrt-jetson-agx-orin-optimization-guide/)
- [Dynamic engine building and optimization profile — TensorRT Forums](https://forums.developer.nvidia.com/t/dynamic-engine-building-and-optimization-profile/164744/)
- [Performance Benchmarking — NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/11.2.1/performance/benchmarking.html)
- [Optimizing TensorRT Performance — NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)
- [TensorRT Dynamic Shape](https://stevengong.co/notes/TensorRT-Dynamic-Shape)
- [FastAPI + OpenCV + WebSocket Real-Time Video](https://github.com/RajeshwarBura/FastAPI-OpenCV-Real-Time-Video-Streaming-with-WebSocket-Extension)
- [FastRTC: Real-time Communication Library](https://fastrtc.org/)
- [Streamlit WebRTC for Real-time Video](https://github.com/whitphx/streamlit-webrtc)
- [Building Web-Based Real-Time Computer Vision with Streamlit](https://dev.to/whitphx/build-a-real-time-computer-vision-app-with-streamlit-57l2)
- [Gradio Object Detection with WebRTC](https://www.gradio.app/guides/object-detection-from-webcam-with-webrtc)
- [Black-Box Edge AI Model Selection with Latency/Accuracy Guarantees (2025)](https://arxiv.org/pdf/2506.11391)
- [On Accelerating Edge AI: Optimizing Resource-Constrained Environments (2025)](https://arxiv.org/html/2501.15014v2)
- [How Edge AI is Transforming Smart Surveillance Systems](https://www.vvdntech.com/en-us/blog/how-edge-ai-is-transforming-smart-surveillance-systems/)
- [Profiling Concurrent Vision Inference on NVIDIA Jetson (2025)](https://arxiv.org/pdf/2508.08430)
- [Accurate Calibration of Power Measurements on Jetson (2023)](https://arxiv.org/pdf/2306.13107)
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
- [YOLO26 on Jetson DeepStream & TensorRT](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson/)
- [PyTorch ONNX Export with Dynamic Shapes Issue #126607](https://github.com/pytorch/pytorch/issues/126607)
- [Dynamic Shapes Ignored in ONNX Export Issue #149065](https://github.com/pytorch/pytorch/issues/149065)
- [Batch Size Handling in ONNX](https://theneuralbase.com/onnx/learn/advanced/batch-size-handling/)
- [Selective Prediction and Confidence Calibration (2025)](https://openreview.net/forum?id=JJPAy8mvrQ)
- [Calibration-Aware Uncertainty Cascades (2026)](https://arxiv.org/html/2609.11446)
- [SafeGround: Know When to Trust via Uncertainty Calibration](https://arxiv.org/pdf/2602.02419)

**End of report.**
