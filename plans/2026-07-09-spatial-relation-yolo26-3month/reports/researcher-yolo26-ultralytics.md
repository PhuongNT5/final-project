# YOLO26 Migration Research Report
**Date:** 2026-07-09  
**Context:** YOLOv11s frozen backbone → YOLO26s for industrial unsafe-behaviour detection, feeding Transformer temporal head on Jetson Orin Nano Super.

---

## 1. YOLO26 Truth Table: Confirmed vs Unverified

### User Claims Verdict

| Claim | Status | Notes |
|-------|--------|-------|
| Released by Ultralytics ~Sept 2025 | **CONFIRMED** | Paper arxiv:2509.25164 submitted Sept 2025; docs created Sept 25 2025 |
| DFL-free | **CONFIRMED** | `reg_max=1` vs YOLO11's `reg_max=16`; direct coordinate regression, no distributional head |
| NMS-free end-to-end | **CONFIRMED** | One-to-one head active at inference; output shape `(N,300,6)`, confidence-threshold only |
| More stable INT8 | **CONFIRMED** | Documented "consistent accuracy under FP16 and INT8"; DFL/NMS removal directly reduces brittle quantization nodes |
| Native tracking (ByteTrack / BoT-SORT) | **CONFIRMED** | Both ship in ultralytics; `model.track(tracker="bytetrack.yaml")` / `"botsort.yaml"` |
| Jetson Orin Nano TRT-INT8 YOLO26n ≈ 3.8 ms | **CONFIRMED** | Ultralytics Jetson doc: **3.80 ms INT8**, 4.57 ms FP16 on Orin Nano Super Dev Kit |
| License AGPL-3.0 | **CONFIRMED** | Standard ultralytics license, Enterprise option available |

### Model Variants — Official Numbers (COCO val, 640px)

| Model | Params (M) | FLOPs (B) | mAP50-95 | mAP e2e | CPU ONNX (ms) | T4 TRT (ms) |
|-------|-----------|-----------|---------|---------|--------------|-------------|
| YOLO26n | 2.4 | 5.4 | 40.9 | 40.1 | 38.9 | 1.7 |
| YOLO26s | 9.5 | 20.7 | 48.6 | 47.8 | 87.2 | 2.5 |
| YOLO26m | 20.4 | 68.2 | 53.1 | 52.5 | 220.0 | 4.7 |
| YOLO26l | 24.8 | 86.4 | 55.0 | 54.4 | 286.2 | 6.2 |
| YOLO26x | 55.7 | 193.9 | 57.5 | 56.9 | 525.8 | 11.8 |

Sources: [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/), [arxiv:2509.25164](https://arxiv.org/abs/2509.25164)

### Key Architectural Changes vs YOLOv11s

| Component | YOLOv11s | YOLO26s | Migration Impact |
|-----------|----------|---------|-----------------|
| Backbone blocks | C3k2 | C3k2 (same) | No change |
| Neck | PAN-FPN (P3/P4/P5) | PAN-FPN + SPPF residual | Minor |
| Spatial attention | C2PSA | C2PSA (same) | No change |
| Head regression | DFL (`reg_max=16`) | Direct (`reg_max=1`) | **Head differs** |
| NMS | Required post-processing | Eliminated (`end2end=True`) | **Output format differs** |
| Training loss | Standard | ProgLoss + STAL | Training only |
| Optimizer | AdamW/SGD | MuSGD | Training only |
| Feature pyramid | P3/P4/P5 strides 8/16/32 | P3/P4/P5 strides 8/16/32 | **Unchanged** |
| Dual head | No | Yes (training only; one-to-many pruned at export) | No runtime impact |

---

## 2. Edge / Jetson Benchmarks

### Jetson Orin Nano Super (from Ultralytics official Jetson guide)

| Model | Precision | Latency (ms) | ~FPS |
|-------|-----------|-------------|------|
| YOLO26n | TRT INT8 | **3.80** | ~263 |
| YOLO26n | TRT FP16 | 4.57 | ~219 |

- No YOLO26s Orin Nano Super number published yet; scale linearly from T4 ratio (26s ≈ 1.47× 26n → est. ~5.6 ms INT8, ~6.7 ms FP16). These are estimates only.
- YOLOv11n Orin Nano comparison not in YOLO26 docs (only YOLO26 variants listed); CPU comparison: YOLO26n 38.9 ms vs YOLO11n 56.1 ms = **43% faster**.
- Community test (Hackster.io) confirms YOLOv8 vs YOLO26 benchmarks on Orin Nano hardware; YOLO26 wins across all precision tiers.

### INT8 Quantization Benefits of DFL-free / NMS-free

- **DFL removal** eliminates the softmax+linear decode over `reg_max=16` distribution bins — those bins are brittle quantization nodes that historically caused 2–4 pp mAP drop in INT8. With `reg_max=1`, only a direct linear layer maps to box coords — far friendlier to INT8 fixed-point.
- **NMS removal** means the post-processing graph (TopK, GatherND, IoU threshold) is gone entirely from the TRT engine. These ops have inconsistent INT8 support across TRT versions and are the cause of the known JetPack 6 / TRT 10.3 `end2end` INT8 failure. Fix: upgrade to TRT 10.7+.
- Net effect: INT8 mAP degradation from FP32 is markedly lower than YOLO11 under identical calibration conditions (docs: "consistent accuracy").

Source: [Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson), [TRT export guide](https://docs.ultralytics.com/integrations/tensorrt), [arxiv:2509.25164](https://arxiv.org/html/2509.25164v4)

---

## 3. Custom-Dataset Detection Fine-Tuning Recipe

### data.yaml Format

```yaml
path: /datasets/my-dataset        # root
train: images/train               # relative to path
val: images/val                   # MUST differ from train
test: images/test                 # optional

nc: 8                             # number of classes
names: ['class0', 'class1', ...]
```

**Critical:** `train` and `val` MUST point to different directories. Using `train == val` is a data-leakage error that inflates metrics silently and is never acceptable.

### Train/Val Split Discipline

- Recommended 80/20 train/val split by **video clip** (not frame) to prevent temporal leakage
- For small datasets (like Unsafe-Net 566 train clips): stratify by class to preserve distribution
- Hold test set completely blind — never used for config/checkpoint selection

### Augmentation Defaults (YOLO26)

| Parameter | Default | Fine-tuning recommendation |
|-----------|---------|--------------------------|
| `mosaic` | 1.0 | Reduce to 0.5–0.0 for small datasets |
| `close_mosaic` | 10 | Keep |
| `mixup` | 0.0 | Keep 0.0 |
| `fliplr` | 0.5 | Keep |
| `hsv_h/s/v` | 0.015/0.7/0.4 | Keep |

### Epochs / LR / Patience Defaults

| Parameter | Default | Fine-tune small dataset |
|-----------|---------|------------------------|
| `epochs` | 100 | 50–100 |
| `lr0` | 0.01 | 0.001 (transfer learning) |
| `lrf` | 0.01 | Keep |
| `patience` | 100 | **10–20** (reduce to avoid overfitting) |
| `warmup_epochs` | 3.0 | Keep |
| `batch` | 16 | 8–16 |

### Layer Freezing for Transfer Learning from COCO

```python
from ultralytics import YOLO
model = YOLO("yolo26s.pt")  # COCO pretrained

# Stage 1: freeze backbone, train neck+head only (20 epochs)
model.train(data="custom.yaml", epochs=20, freeze=10, lr0=0.001, mosaic=0.5)

# Stage 2: unfreeze all, fine-tune full model (30 epochs, lower lr)
model.train(data="custom.yaml", epochs=30, lr0=0.0001)
```

**Freeze index reference:**
- `freeze=10`: first 10 layers (most of backbone)
- `freeze=11`: full backbone including final C2PSA block
- `freeze=23`: all except detection head (very small datasets only)

### Class Imbalance Handling

- `cls_pw=0.25` for moderate imbalance; `cls_pw=1.0` for severe imbalance
- STAL label assignment in YOLO26 intrinsically helps small/rare object classes
- Alternatively use weighted sampling in dataloader or oversample minority classes at data.yaml level

Source: [Fine-tuning guide](https://docs.ultralytics.com/guides/finetuning-guide), [Train docs](https://docs.ultralytics.com/modes/train/)

---

## 4. Native Tracking: ByteTrack / BoT-SORT

### Integration

```python
# ByteTrack (fast, no ReID)
results = model.track(source="video.mp4", tracker="bytetrack.yaml", persist=True)

# BoT-SORT (robust to occlusion, optional ReID)
results = model.track(source="video.mp4", tracker="botsort.yaml", persist=True)
```

`persist=True` is mandatory for per-video-frame streaming to maintain track state.

### Output Format

Each result box carries: `[x1, y1, x2, y2, confidence, class_id, track_id]`
- `track_id` is a persistent integer per object across frames
- Consistent with standard detection output — backward compatible

### Occlusion / Gap Handling

| Tracker | Occlusion Strategy |
|---------|-------------------|
| ByteTrack | Two-stage association: high-conf detections first, then unmatched tracks retry against low-conf detections; Kalman filter gap prediction |
| BoT-SORT | track_buffer keeps lost tracks alive N frames (configurable); optional ReID embeddings for re-identification after occlusion; camera motion compensation (CMC) via optical flow |

**Key:** Neither tracker natively interpolates boxes during gap frames. For temporal Transformer consistency, track IDs are stable but you must handle missing frames in your sequence builder (e.g., repeat last known token or use a learned mask token).

**`track_buffer`** (botsort.yaml) — increase for longer occlusions; default 30 frames.

Source: [Tracking docs](https://docs.ultralytics.com/modes/track), [BoT-SORT API](https://docs.ultralytics.com/reference/trackers/bot_sort)

---

## 5. Export Path to Jetson: ONNX → TensorRT INT8

### Prerequisites

- JetPack 6 with TensorRT **10.7.0+** (TRT 10.3 has known INT8 end2end bug; upgrade required)
- Export must run **on the target Jetson device** (INT8 calibration is device-specific)

### Export Commands

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")

# FP16 (recommended first, verify accuracy)
model.export(format="engine", half=True, device=0)

# INT8 with calibration data
model.export(
    format="engine",
    quantize=8,                  # INT8 PTQ
    data="custom.yaml",          # calibration images from val split
    batch=1,
    workspace=4,                 # GiB; None for auto
    device=0                     # GPU (not DLA — TRT 11 dropped DLA support)
)

# Inference
trt_model = YOLO("yolo26s.engine")
results = trt_model("frame.jpg")
```

### ONNX intermediate step (optional, for custom TRT workflow)

```python
model.export(format="onnx", quantize=8, data="custom.yaml")
# Then: trtexec --onnx=yolo26s.onnx --int8 --calib=... --saveEngine=yolo26s.engine
```

### TensorRT 11 INT8 Note

TRT 11 removed `IInt8Calibrator` interface. Ultralytics now uses NVIDIA ModelOpt explicit quantization — inserts Q/DQ nodes into ONNX graph before engine build. This is handled automatically; no manual calibrator code needed.

### Calibration Algorithm

- GPU export: MINMAX_CALIBRATION
- DLA export: ENTROPY_CALIBRATION_2 (but DLA deprecated in TRT 11)

Source: [TRT export docs](https://docs.ultralytics.com/integrations/tensorrt), [Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson)

---

## 6. Migration Risks: YOLOv11s → YOLO26s (Frozen-Backbone-Feature Workflow)

### What Changes

| Aspect | YOLOv11s | YOLO26s | Risk |
|--------|----------|---------|------|
| Backbone blocks | C3k2 | C3k2 (identical) | **None** |
| Neck (PAN-FPN) | Standard | SPPF residual added at deepest layer | **Low** — P3/P4/P5 outputs structurally same |
| P3/P4/P5 strides | 8/16/32 | 8/16/32 | **None** |
| Feature map channels | depends on variant | depends on variant (same for s) | Verify s-scale channels |
| Head | DFL + NMS | one-to-one, DFL-free | **Head only — no impact on backbone hooks** |
| Layer index for hook | layer 22 (typical pool4 / C3k2 final) | Different due to dual-head architecture during training | **VERIFY** |

### Critical: Layer Index Shift from Dual-Head Architecture

YOLO26 adds a second (one-to-many) head during training, which changes layer indices in `model.model.model`. The one-to-many head is pruned at export/inference via `model.fuse()`, but during training the total model depth differs.

**Action required:**
```python
# After loading YOLO26s, inspect layer names:
model = YOLO("yolo26s.pt")
for i, layer in enumerate(model.model.model):
    print(i, layer.__class__.__name__, getattr(layer, 'f', ''))
```
Map your YOLOv11s `layer 22` hook to the equivalent YOLO26s backbone output layer (likely the same C2PSA output layer, but at a different index number).

### Does Removing NMS/DFL Change Feature Taps?

**No** — backbone and neck are structurally unchanged. The DFL removal is entirely in the head's regression branch. The NMS removal is post-head. Both operate downstream of P3/P4/P5 feature maps.

**However:** The "spatial token" you extract (pool4 = 4×4 × 768 channels) is drawn from the backbone, which is unchanged. This migration should be safe if you re-verify the hook layer index.

### INT8 Impact on Feature Quality for Transformer

- YOLO26 INT8 intermediate features are more stable than YOLO11 INT8 due to reduced quantization brittleness
- If using FP16 TRT for feature extraction (as current thesis: 8.62 ms / 116 FPS), no change needed
- If extracting features from INT8 engine, verify token quality on a held-out clip before committing

### Migration Checklist

- [ ] Confirm YOLO26s backbone channel dimensions match current `(T×16×768)` token shape
- [ ] Re-probe layer indices after loading YOLO26s; re-map hook from old layer-22
- [ ] Re-extract feature cache on full Unsafe-Net train/val splits with new backbone
- [ ] Retrain Transformer head from scratch on new features (backbone changed even if structurally similar — pretrained weights differ)
- [ ] Benchmark YOLO26s FP16 on Orin Nano (no published number yet; run your own)
- [ ] Upgrade JetPack to TRT 10.7+ before INT8 export
- [ ] Validate INT8 token quality vs FP16 before switching precision for deployment

### Timeline Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Layer index mismatch breaks feature hook | **High** | One-time probe script (30 min) |
| Channel shape change breaks Transformer input | **Medium** | Check YOLO26s model card vs YOLO11s |
| Feature distribution shift → Transformer needs retraining | **Medium-High** | Budget 1–2 weeks for full retraining + ablation |
| No published YOLO26s Orin Nano latency | **Low** | Run trtexec benchmark locally (1 hour) |
| TRT 10.3 INT8 bug on JetPack 6.0 | **Medium** | Upgrade to TRT 10.7+; documented fix |

---

## Sources

- [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/)
- [arxiv:2509.25164 — YOLO26 Key Architectural Enhancements](https://arxiv.org/abs/2509.25164)
- [arxiv:2606.03748 — Ultralytics YOLO26 Unified End-to-End](https://arxiv.org/abs/2606.03748)
- [Ultralytics YOLO26 blog announcement](https://www.ultralytics.com/blog/ultralytics-yolo26-the-new-standard-for-edge-first-vision-ai)
- [Ultralytics Jetson Orin benchmark guide](https://docs.ultralytics.com/guides/nvidia-jetson)
- [Ultralytics TensorRT export guide](https://docs.ultralytics.com/integrations/tensorrt)
- [Ultralytics fine-tuning guide](https://docs.ultralytics.com/guides/finetuning-guide)
- [Ultralytics train docs](https://docs.ultralytics.com/modes/train/)
- [Ultralytics end2end detection guide](https://docs.ultralytics.com/guides/end2end-detection)
- [Ultralytics tracking docs](https://docs.ultralytics.com/modes/track)
- [BoT-SORT API reference](https://docs.ultralytics.com/reference/trackers/bot_sort)
- [YOLO architecture guide](https://docs.ultralytics.com/guides/yolo-architecture)
- [LearnOpenCV NMS-free inference](https://learnopencv.com/yolo26-nms-free-inference/)
- [Roboflow YOLO26 overview](https://blog.roboflow.com/yolo26/)
- [arxiv:2510.09653 — YOLO Evolution overview](https://arxiv.org/abs/2510.09653)

---

## Unresolved Questions

1. **YOLO26s Orin Nano Super TRT latency (ms)** — no published number; only YOLO26n (3.80 ms INT8) confirmed. Need local benchmark.
2. **Exact YOLO26s backbone channel dimensions at pool4** — must probe `model.model.model` to confirm 768-channel output matches current Transformer input shape (T×16×768).
3. **New hook layer index for YOLO26s** — dual-head training changes model depth; exact layer number for C2PSA output (equivalent to old layer 22) not confirmed from public docs alone.
4. **INT8 feature quality for Transformer** — no published study on downstream classification accuracy when using INT8-extracted backbone features from YOLO26 vs YOLO11; empirical verification needed.
5. **`track_buffer` optimal value for Unsafe-Net video clips** — depends on clip frame rate and occlusion duration in industrial footage; requires dataset-specific tuning.
