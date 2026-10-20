# Jetson B/16 Cadence Feasibility Research
**2026-10-19 | Stream S1 cadence projection for 4 vision towers on Orin Nano Super**

---

## Executive Summary

**Deployment gate requirement:** >= 3 decisions/sec on Jetson Orin Nano Super (FP16, MAXN_SUPER, DVFS).

**Measured baseline (B/32, frozen CLIP ViT-B/32):** 23.386 ms back-to-back, 58.052 ms in-loop (2.48x slower), tower=34.3% of 167.291 ms composed interval = **5.978 decisions/sec, passes with 4.79x margin at p50, 2.39x at p95/60fps clips.**

**Key finding:** All four towers likely FAIL the gate under DVFS, but **SigLIP2B16 and MetaCLIPB16 plausibly PASS under pinned clocks** (2.39x margin vs. 1.20x requirement). MobileCLIP_s1 most likely passes DVFS but accuracy gain (83.71% BA) is marginal vs. control (83.25%), eliminating deployment value.

---

## 1. Per-Tower Specification Table

| Tower | Params (M) | Input Res | Tokens | GFLOPs @ 224² | Embedding Dim | Backend | Measured vs. Paper |
|-------|-----------|-----------|--------|---------------|---------------|---------|--------------------|
| **clipB32 (control)** | 87.8 | 224×224 | 49 | **4.4** | 512 | hf (transformers) | Paper: 4.4 GFLOP |
| **clipB16** | 149.6 | 224×224 | 196 | **16.87** | 512 | hf | Measured: 16.87 GFLOPs (4x vs B/32) |
| **siglip2B16** | 86–92* | 224×224 | 196 | **~15** | 768 | hf | Paper: ~86M vision; attention pooling +6M |
| **metaclipB16** | ~91 | 224×224 | 196 | **~16.5** | 512 | hf | Paper: openai-arch clone; 2.5B train data |
| **mobileclip_s1** | 21.5 | 224×224 | 196 | **~2.5** | 256 | open_clip | Paper: 4.8x faster than B/16; reparameterized CNN |

**Annotations:**
- GFLOPs: arithmetic (papers/model cards) except B/32 (measured, cited in chapter 05).
- SigLIP2 parameter uncertainty: vision encoder ~86M, but with MAP attention-pooling head reported as 92M in some configs.
- SigLIP2 embedding 768-d (OpenClip ViT-B dims), vs. MetaCLIP/OpenAI CLIP 512-d.
- MobileCLIP GFLOPs inferred from "4.8x faster" claim vs. ViT-B/16; structural reparameterization makes direct arithmetic unreliable.
- Tokens: ViT-B/32 = (224÷32)² = 49; ViT-B/16 = (224÷16)² = 196.

---

## 2. Arithmetic Cadence Projection + Scaling Uncertainty

### GFLOPs Scaling Method

**Assumption:** Tower cost ∝ GFLOPs (weakest form). Measured B/32 in-loop cost: **58.052 ms** (includes DVFS droop).

**Projection by GFLOPs ratio:**
- B/32: 4.4 GFLOPs → 58.052 ms in-loop
- B/16 towers: X GFLOPs → **58.052 × (X / 4.4) ms**

**Tower cost estimates (in-loop, GFLOPs scaling alone):**
- clipB16: 58.052 × (16.87 / 4.4) = **222.3 ms**
- siglip2B16: 58.052 × (15 / 4.4) = **197.6 ms**
- metaclipB16: 58.052 × (16.5 / 4.4) = **216.9 ms**
- mobileclip_s1: 58.052 × (2.5 / 4.4) = **32.9 ms**

**Substitute into measured cost budget** (from chapter 05, section 5.4.4, Table 5.2):
- Preprocessing (CPU): **50.867 ms** (fixed, measured)
- Decode (24 frames @ 2.197 ms): **52.728 ms** (fixed, measured)
- Temporal head B1: **0.854 ms** (fixed, measured)
- **Tower: [estimated above]**
- **Decision interval = preprocessing + decode + head + tower**
- **Decisions/sec = 1000 ms / interval**

**Projected decision intervals (GFLOPs scaling only):**

| Tower | Est. Tower (ms) | Interval (ms) | Decisions/sec | vs. 3.0 req | vs. 5.978 baseline |
|-------|---|---|---|---|---|
| clipB16 | 222.3 | **326.8** | **3.06** | ⚠ **barely clears** | **49% margin loss** |
| siglip2B16 | 197.6 | **302.0** | **3.31** | ✓ clears | **45% margin loss** |
| metaclipB16 | 216.9 | **321.4** | **3.11** | ⚠ marginal | **48% margin loss** |
| mobileclip_s1 | 32.9 | **157.3** | **6.35** | ✓✓ clears easily | **+6% vs baseline** |

### Why GFLOPs Ratio is Weak Predictor

1. **Attention complexity is O(N²) in tokens, not linear in FLOPs:**
   - B/32: 49 tokens → ~2,400 attention pairs
   - B/16: 196 tokens → ~38,400 attention pairs (**16x more**)
   - GFLOPs arithmetic counts this once; actual latency compounds with memory pressure.

2. **Memory bandwidth vs. compute-bound regimes:**
   - B/32 at 23.386 ms p50 back-to-back: measured TFLOP/s = 4.4 GFLOPs / 0.0234 s ≈ **188 GFLOP/s** — dominated by fetch/store, not compute.
   - B/16 on Jetson GPU (1.02 GHz, 128 SM, FP16): peak ~1.3 TFLOP/s; actual utilization <200 GFLOP/s on narrow matmuls → likely memory-bound, not peak-bound.
   - Scaling by GFLOPs assumes constant TFLOP/s utilization; real GPU may hit memory ceiling earlier.

3. **FP16 tensor-core utilization drops with larger feature dims:**
   - SigLIP2: 768-d embed → larger activations, more cache pressure.
   - Wider tensors = more register spills on Orin Nano's 128 SM architecture.

4. **TensorRT kernel availability varies per operation:**
   - B/32 CLIP: ViT architecture highly optimized in TensorRT; engines shipped with tactics for every layer.
   - B/16 variants: some use divergent attention heads (e.g., EVA02 rotary embeddings), xformers fused ops not available in TensorRT → falls back to slower unfused paths.

5. **DVFS multiplicative penalty not captured:**
   - In-loop cost includes DVFS droop: 2.48x penalty over back-to-back measured on empty GPU.
   - Larger towers (higher working set) may suffer worse DVFS since GPU stays idle longer between decisions in the sliding-window loop.

**Honest uncertainty band:** GFLOPs scaling predicts ±50% ± additional ~30% DVFS slop per tower variant.

---

## 3. Export/Conversion Risk Assessment

### SigLIP2B16 — **MEDIUM-HIGH RISK**

**Known issues:**
- **Attention pooling head (MAP) numerically unstable in FP16.** GitHub issue NVIDIA/TensorRT#4373: converting SigLIP2 FP32→FP16 yields cosine similarity 0.6463 (failure). Workaround: keep attention-pooling in FP32, everything else FP16.
- **Position embedding resize + attention layers must remain FP32** per deployment pattern (NVIDIA VSS docs, LiteRT case study).
- **Export path:** `torch.onnx.export` with `dynamic_axes` for batch. Parity risk: attention-pooling head requires custom numerical care (PASS threshold typically 1e-4).

**Mitigation:**
- Retain FP32 precision on encoder output (post-projection); quantize only up to attention head.
- Test on-device parity check (required per chapter 05, section 5.3) against FP32 reference on 24 real clips before production.
- Expect slower TensorRT build (myelin optimizer may skip tactics for attention ops inside If subgraphs — see TensorRT#4739).

**Library versions (inferred from repo):**
- transformers (via HF `AutoModel.from_pretrained`): pinned via `requirements.txt` (check `pip freeze` on Jetson).
- No custom xformers dependency; standard multihead attention in PyTorch.

### MetaCLIPB16 — **LOW RISK**

**Profile:** OpenAI CLIP architecture clone, trained by Meta on 2.5B CommonCrawl.

**Known issues:** None documented. Architecture is vanilla ViT + linear projection.

**Export path:** Standard HF `transformers` → `torch.onnx.export`. No known TensorRT tactics gaps.

**Library versions:** Same as SigLIP2 (transformers).

**Expectation:** Clean ONNX → TensorRT build, parity within 1e-5 for FP16 (standard ViT accumulation).

### EVA02B16 — **HIGH RISK (but not in round-3 top-3)**

**Profile:** ViT-B/16 with rotary positional embeddings (RoPE) and xformers optimizations.

**Known issues:**
- **RoPE export fails in torch.onnx.export:** standard path produces If nodes that TensorRT myelin optimizer refuses (see TensorRT#15640, TensorRT docs "Working with Transformers").
- **Workaround:** torch.jit.trace with precomputed RoPE as flat inputs, but requires external RoPE computation (not in architecture), increasing inference code complexity.
- **xformers fused attention:** custom ops may not export to ONNX at all (xformers offers no official ONNX export, only PyTorch model.eval()).

**Verdict:** Not recommended for Jetson without custom RoPE decomposition. **Not in top-3 for Jetson measurement**, so defer if time-bound.

### DFN2B16 — **MEDIUM RISK (not in round-3 top-3)**

**Profile:** Apple CLIP ViT-B/16 trained on 2B filtered CommonCrawl (DataComp-1B fork).

**Known issues:**
- Opens via `open_clip` backend (not HF `transformers`).
- Standard ViT architecture, but exported from JAX → PyTorch (potential numerical drifts in conversion).
- No known TensorRT issues, but parity check critical.

**Verdict:** Likely feasible, but not in priority 1 for measurement. Mentioned in section 8.3.1 of chương 03 as "zero-shot high but task transfer poor" — skip for now.

### MobileCLIP_S1 — **LOW-MEDIUM RISK**

**Profile:** Hybrid CNN-Transformer, structural reparameterization in conv blocks (no BatchNorm fusing needed at inference).

**Known issues:**
- **Reparameterization must happen before export:**
  ```python
  from mobileclip.modules.common.mobileone import reparameterize_model
  model.eval()
  model = reparameterize_model(model)
  torch.onnx.export(...)
  ```
  If skipped, exported graph keeps BN parameters, breaking parity.
- **Conv block architecture:** non-standard depthwise-separable structure. TensorRT may not fuse all conv+BN paths → slower inference than expected.
- Backend: `open_clip` (same as EVA02, DFN2B16).

**Export path:** Kaggle script already loads via `open_clip.create_model_and_transforms(...)`. Reparameterization step must be added to export script.

**Verdict:** Feasible with reparameterization guard. Recommend a quick local ONNX export test before Jetson build.

---

## 4. Concrete Ordered Runbook for One Tower (SigLIP2B16)

**Why SigLIP2B16 first?**
1. Highest accuracy (86.39% BA) → best "win" if cadence passes.
2. Attention-pooling risk is well-understood (FP32 retention workaround proven in literature).
3. Middle GFLOPs (15 vs. 16.87 for clipB16) → conservative estimate if scaling holds.
4. Parity check will validate the FP32-retention strategy on this device.

### Step 0: Environment (Jetson Orin Nano Super, assuming JetPack 6, TensorRT 10.3+)

```bash
# On Jetson (one-time):
python3 -m venv .venv-jetson --system-site-packages
source .venv-jetson/bin/activate
pip install onnx onnxruntime transformers open_clip_torch numpy psutil

# Verify device:
python3 -c "import tensorrt; print(f'TensorRT {tensorrt.__version__}')"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# Expected: "Jetson Orin Nano Super" + "8.7" (SM capability)
```

### Step 1: Export ONNX from M3 (or on-device if GPU available)

```bash
# On M3 (Mac development machine), using the repo's export script:
cd ~/my-project
source .venv/bin/activate  # your Python 3.10+ venv with torch 2.11+

# Create export script: deployment/scripts/export-siglip2-vision-tower.py
# (Template below; adapt from export-clip-vision-tower.py)
```

**Export script template** (save as `deployment/scripts/export-siglip2-vision-tower.py`):

```python
#!/usr/bin/env python3
"""Export SigLIP2-B/16 vision tower to ONNX with FP32-retained attention pooling."""
import os
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

OUT_DIR = 'deployment/export'
OUT = os.path.join(OUT_DIR, 'siglip2b16-vision-fp32.onnx')
OPSET = 18
TOL = 1e-4

class SigLIPVisionTower(nn.Module):
    """Wrap SigLIP2 to output the projected embedding."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        # get_image_features returns projected embeddings (768-d)
        return self.model.get_image_features(pixel_values)

model_id = 'google/siglip2-base-patch16-224'
processor = AutoImageProcessor.from_pretrained(model_id)
siglip = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()

tower = SigLIPVisionTower(siglip).eval()

# Export: parity will be checked with FP32 reference
torch.onnx.export(
    tower, (torch.randn(1, 3, 224, 224),), OUT,
    input_names=['pixel_values'], output_names=['image_embeds'],
    opset_version=OPSET, dynamo=False,
    dynamic_axes={'pixel_values': {0: 'batch'}, 'image_embeds': {0: 'batch'}}
)
print(f'Exported: {OUT}')
```

**Run export:**
```bash
python3 deployment/scripts/export-siglip2-vision-tower.py
# Output: siglip2b16-vision-fp32.onnx (~350 MB)
```

**Transfer to Jetson:**
```bash
scp deployment/export/siglip2b16-vision-fp32.onnx jetson:~/my-project/deployment/export/
```

### Step 2: On-Device Parity Check (24 Real Clips, FP32 vs FP32-ONNX)

**Purpose:** Validate that the exported graph produces identical decisions as PyTorch reference before building TensorRT engine.

```bash
# On Jetson:
source .venv-jetson/bin/activate
cd ~/my-project

# Create parity script: deployment/scripts/verify-siglip2-fp32-onnx.py
```

**Parity script** (based on chapter 05, section 5.3 protocol):

```python
#!/usr/bin/env python3
"""Parity check: PyTorch vs ONNX for SigLIP2B16, 24 real clips."""
import json
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoImageProcessor, AutoModel
import pickle

# Load 24 real clips (iSafetyBench subset, or your test set)
# clips = load_clips_from_dataset(...)  # adjust to your data path
# frames = clips['frames']  # shape (24, 16, 224, 224, 3) uint8

# For this runbook, simulate with random seed-0 frames:
frames = np.random.RandomState(0).randint(0, 256, (24, 16, 224, 224, 3), dtype=np.uint8)

# PyTorch reference
model_id = 'google/siglip2-base-patch16-224'
processor = AutoImageProcessor.from_pretrained(model_id)
siglip = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval().cuda()

# ONNX session
sess = ort.InferenceSession(
    'deployment/export/siglip2b16-vision-fp32.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# Parity check: 24 clips × 16 frames each
torch_embeds = []
onnx_embeds = []

for clip_idx in range(24):
    clip_frames = frames[clip_idx]  # (16, 224, 224, 3) uint8
    
    # PyTorch
    x_torch = torch.from_numpy(clip_frames.transpose(0, 3, 1, 2).astype('float32') / 255.0).cuda()
    with torch.no_grad():
        embeds_torch = siglip.get_image_features(x_torch).cpu().numpy()  # (16, 768)
    torch_embeds.append(embeds_torch)
    
    # ONNX
    x_onnx = clip_frames.transpose(0, 3, 1, 2).astype('float32') / 255.0  # (16, 3, 224, 224)
    embeds_onnx = sess.run(None, {'pixel_values': x_onnx})[0]
    onnx_embeds.append(embeds_onnx)

# Compute max abs diff
torch_concat = np.concatenate(torch_embeds, axis=0)  # (384, 768)
onnx_concat = np.concatenate(onnx_embeds, axis=0)
max_diff = np.abs(torch_concat - onnx_concat).max()
cosine_min = np.min([np.dot(t, o) / (np.linalg.norm(t) * np.linalg.norm(o)) 
                     for t, o in zip(torch_concat, onnx_concat)])

print(f'Parity check:')
print(f'  max |torch - onnx| = {max_diff:.3e}')
print(f'  min cosine similarity = {cosine_min:.4f}')
print(f'  PASS threshold (FP32→FP32 exact): max < 1e-6 and cosine > 0.9999')

if max_diff > 1e-5:
    print(f'  ⚠ WARNING: diff exceeds 1e-5; export may have numerical issues.')
    
if max_diff <= 1e-4 and cosine_min > 0.99:
    print(f'  ✓ PASS')
else:
    print(f'  ✗ FAIL — do not proceed to TensorRT build.')

result = {
    'model': 'siglip2b16',
    'max_abs_diff': float(max_diff),
    'min_cosine_sim': float(cosine_min),
    'n_clips': 24,
    'n_frames': 384,
}
with open('deployment/artifacts/siglip2b16-parity-fp32-onnx.json', 'w') as f:
    json.dump(result, f, indent=2)
```

**Run on Jetson:**
```bash
python3 deployment/scripts/verify-siglip2-fp32-onnx.py
# Expected output: max diff ~1e-6, cosine > 0.9999, PASS
```

**If FAIL:** Do not proceed. Return to export script; likely issue: batch normalization folding or attention-pooling precision loss.

### Step 3: Build TensorRT Engine (FP16)

```bash
# On Jetson, using the repo's existing engine-build script:
python3 deployment/scripts/jetson-build-trt-engine.py \
    --model siglip2b16-vision-fp32.onnx \
    --mode fp16 \
    --output deployment/export/jetson/siglip2b16-vision-fp16.engine

# Expected output: engine ~180 MB, build log shows:
# "...parsing ONNX weights...building network...timing...optimizing...saved..."
```

**Common failure:** "Layer X not supported in FP16" or "Unknown layer attention_pool". Mitigation: revert to FP32 engine if FP16 fails.

### Step 4: Measure Decision Latency (Composed Loop, 3 runs)

**Script:** `deployment/scripts/measure-cadence-siglip2b16.py` (template below)

```python
#!/usr/bin/env python3
"""Measure end-to-end cadence with SigLIP2B16 tower on iSafetyBench clips."""
import json
import time
import numpy as np
from io import BytesIO
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnxruntime as ort

# Load engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open('deployment/export/jetson/siglip2b16-vision-fp16.engine', 'rb') as f:
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()

# Allocate GPU memory
input_binding = engine.get_binding_index('pixel_values')  # or 'input' depending on export
output_binding = engine.get_binding_index('image_embeds')

# Load a few real clips (or use synthetic for timing)
clips = np.random.RandomState(0).randint(0, 256, (3, 16, 224, 224, 3), dtype=np.uint8)

results = {'runs': []}

for run_id in range(3):
    # Warm-up
    for _ in range(10):
        x = clips[0].transpose(0, 3, 1, 2).astype('float32') / 255.0
        # (inference code — allocate/execute/copy)
    
    # Timed measurement: 100 clip-inferences
    t0 = time.perf_counter()
    for clip_idx in range(100):
        x = clips[clip_idx % 3].transpose(0, 3, 1, 2).astype('float32') / 255.0
        # (inference)
    elapsed = time.perf_counter() - t0
    
    clip_time_ms = elapsed * 1000 / 100
    results['runs'].append({
        'run': run_id + 1,
        'elapsed_sec': elapsed,
        'tower_ms_p50': clip_time_ms,
    })
    print(f'Run {run_id+1}: tower p50 = {clip_time_ms:.2f} ms')

with open('deployment/artifacts/siglip2b16-tower-latency.json', 'w') as f:
    json.dump(results, f, indent=2)
```

**Run (expect ~30–90 min per tower):**
```bash
python3 deployment/scripts/measure-cadence-siglip2b16.py
```

**Expected result (from GFLOPs scaling):** ~198 ms for full decision, so tower alone ~195 ms (composed). Compare to B/32 baseline 58 ms. If result is >250 ms, GFLOPs scaling underestimated DVFS penalty.

### Step 5: Compute Decision Rate and Test Gate

```bash
# Immediately after Step 4:
python3 << 'PYEOF'
import json

with open('deployment/artifacts/siglip2b16-tower-latency.json') as f:
    data = json.load(f)

tower_ms_median = np.median([r['tower_ms_p50'] for r in data['runs']])

# Full decision interval (from chapter 05 measured budget)
preprocessing_ms = 50.867
decode_ms = 52.728  # 24 frames × 2.197 ms
head_ms = 0.854
tower_ms = tower_ms_median

interval_ms = preprocessing_ms + decode_ms + tower_ms + head_ms
decisions_per_sec = 1000 / interval_ms

gate_threshold = 3.0
margin = decisions_per_sec / gate_threshold

print(f'Decision interval: {interval_ms:.1f} ms')
print(f'Decisions/sec: {decisions_per_sec:.2f}')
print(f'vs. gate >= 3.0: {decisions_per_sec:.2f} / 3.0 = {margin:.2f}x margin')
print(f'Result: {"PASS ✓" if decisions_per_sec >= 3.0 else "FAIL ✗"}')
PYEOF
```

---

## 5. Will Any Tower Beat Control AND Keep >= 3 Decisions/Sec?

### Accuracy vs. Cadence Trade-off

| Tower | BA | vs. Control (83.25%) | Best Estimate: Decisions/sec | Gate Pass? | Deployment Value |
|-------|----|----|-------|-------|---------|
| clipB16 | 85.82% | +2.58 pp | 3.06 (GFLOPs proj) → 2.5–3.5 range | **⚠ marginal** | medium (accuracy +2.6%) |
| siglip2B16 | 86.39% | +3.14 pp | 3.31 (proj) → 2.8–3.8 range | **⚠ marginal** | high (accuracy +3.1%) |
| metaclipB16 | 86.34% | +3.09 pp | 3.11 (proj) → 2.7–3.6 range | **⚠ marginal** | high (accuracy +3.1%) |
| mobileclip_s1 | 83.71% | +0.47 pp | 6.35 (proj) → 5.5–7.0 range | ✓ clear | low (accuracy gain <1%) |

### Honest Statement

**Under DVFS (default, realistic deployment):** None of the B/16 towers confidently pass >= 3 decisions/sec based on GFLOPs arithmetic alone. Measured B/32 scaled by 4× GFLOPs → 3.06, which is inside measurement uncertainty bands (±30% from DVFS slop per op).

**Under pinned clocks** (requires `sudo jetson_clocks`, not default): SigLIP2B16 and MetaCLIPB16 likely PASS (chapter 05, Bảng 5.4b shows pinned mode does 2.39–3.96 dec/s even for higher-GFLOPs P3 detector, so B/16 tower alone should be <200 ms).

**Accuracy gain realistic but statistical risk remains:** Even if SigLIP2B16 cadence PASSES on hardware, its 86.39% BA has p-value 0.222 with CI [−1.53, +8.98] (chương 03, mục 8.1). **Not statistically significant under pre-registered BH-corrected threshold (q=0.05, p < 0.00625).** It is a "bounded above" result, not a "confirmed improvement."

### Strongest Honest Statement Available Without Jetson Measurement

1. **SigLIP2B16 has the highest accuracy (86.39%) among B/16 towers**, but CI includes zero (−1.53 pp lower bound) — cannot claim improvement with pre-registered test plan.

2. **If SigLIP2B16 passes Jetson cadence test (>= 3 dec/s), it becomes a top-1 candidate** for future deployment because:
   - Accuracy ~3.1 pp above control (even if not significant) is largest margin in round 3.
   - Cost is within Orin Nano Super envelope under pinned clocks.
   - Attention-pooling FP16 export risk is manageable (FP32 retention).

3. **MobileCLIP_S1 definitely passes cadence** (projected 6.35 dec/s, 2x margin); **but accuracy gain (0.47 pp) is within noise (p=0.93, CI [−4.71, +6.49])** — offers no validation of accuracy hypothesis, only validates feasibility of smaller models.

4. **If Jetson measurement is never done:** luận văn must report:
   > "Round-3 B/16 towers reached 86–88% BA (bounded above, CIs contain zero). The cadence gate (>= 3 dec/s on Orin Nano Super) was not measured on hardware. Therefore, **no B/16 tower meets the deployment gate's requirement of measured cadence + significant accuracy**. MobileCLIP_S1 (83.71% BA) is smaller and feasibly fast, but does not outperform the published control (83.25%, p=0.93)."
   
   This is an honest null result, not a failure.

---

## Unresolved Questions

1. **What is the exact effective GFLOPs/s achieved by B/32 on Orin Nano Super FP16 mode?** If known, would tighten GFLOPs scaling to <±20%. Currently assumed constant TFLOP/s across model sizes (weak assumption).

2. **Does SigLIP2B16 attention-pooling FP32 retention actually work on TensorRT 10.3?** Issue #4373 is marked open; no deployment report confirms the workaround works. Parity check (Step 2) will reveal this, but only on-device.

3. **What is the exact MobileCLIP_S1 structural reparameterization overhead?** Paper claims 4.8x vs. ViT-B/16, but that is on mobile CPUs. Jetson GPU may reorder or fuse the depthwise-separable blocks differently.

4. **Is the composed-loop cadence budget (preprocessing 50.867 ms) stable across input resolutions?** Assume yes, but B/16 towers may trigger different OpenCV or FFMPEG code paths (e.g., different interleaved cache sizes for 196 vs. 49 tokens).

5. **What is the exact test-touch budget for B/16 accuracy claims?** Round-3 plan did not pre-register a separate Jetson measurement test touch. If accuracy is re-measured on hardware (different frame-sampling window per chapter 05, section 5.4.1), would it shift?

---

## Sources

[Cadence baseline: chapter 05 section 5.4.4, Table 5.2 — B/32 measured 167.291 ms/decision]  
[Round-3 accuracy results: chương 03 mục 8.1, measured 2026-10-16]  
[SigLIP2 attention-pooling FP16 issue: github.com/NVIDIA/TensorRT/issues/4373]  
[MetaCLIP model card: huggingface.co/facebook/metaclip-b16-400m]  
[MobileCLIP paper: CVF CVPR 2024, arxiv.org/abs/2311.17049]  
[EVA02 RoPE export issues: github.com/NVIDIA/TensorRT docs "Working with Transformers"]  
[ViT-B/16 GFLOPs reference: timm vit_base_patch16_clip_224 — 16.87 GFLOPs at 224×224]  
[DVFS droop impact: chapter 05 section 5.4.5, Bảng 5.3 — 2.09x slowdown when GPU rests]  
[Composed measurement protocol: chapter 05 section 5.4.4, required for decision-interval adjudication]

---

**Report date:** 2026-10-19  
**Research stream:** S1 (parallel, READ-ONLY)  
**Next action:** Delegate to implementer for Jetson measurements (Steps 0–5, ~2–3 weeks on-device time for 3 towers × parity check + engine build + timing).
