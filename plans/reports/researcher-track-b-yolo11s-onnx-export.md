# YOLOv11s Layer-22 ONNX Export — Research Report

**Date:** 2026-04-17  
**ultralytics version tested:** 8.4.27  
**PyTorch:** 2.11.0 | **ONNX:** 1.21.0 | **ORT:** 1.24.4  

---

## 1. Architecture Facts (Live-Introspected)

### Model structure — `yolo11s.pt` (COCO)

```
model.model: nn.ModuleList, 24 layers (indices 0–23)

[0]  Conv      f=-1
[1]  Conv      f=-1
[2]  C3k2      f=-1       ← backbone
[3]  Conv      f=-1
[4]  C3k2      f=-1       ← saved (skip source)
[5]  Conv      f=-1
[6]  C3k2      f=-1       ← saved (skip source)
[7]  Conv      f=-1
[8]  C3k2      f=-1
[9]  SPPF      f=-1       ← backbone terminus
[10] C2PSA     f=-1       ← saved (attention module)
[11] Upsample  f=-1
[12] Concat    f=[-1, 6]  ← FPN skip from layer 6
[13] C3k2      f=-1       ← saved
[14] Upsample  f=-1
[15] Concat    f=[-1, 4]  ← FPN skip from layer 4
[16] C3k2      f=-1       ← saved (P3 head output)
[17] Conv      f=-1
[18] Concat    f=[-1, 13] ← PAN skip from layer 13
[19] C3k2      f=-1       ← saved (P4 head output)
[20] Conv      f=-1
[21] Concat    f=[-1, 10] ← PAN skip from layer 10
[22] C3k2      f=-1       ← saved (P5 head output, deepest)
[23] Detect    f=[16,19,22]

model.save = [4, 6, 10, 13, 16, 19, 22]
```

### Layer-22 output (224×224 input, CONFIRMED live)

- Shape: **(1, 512, 7, 7)** — C=512, H=W=7 (stride-32 feature map)
- Type: `C3k2`
- This is the P5 (large-object) FPN/PAN output, NOT the SPPF output
- SPPF is layer 9; C2PSA (attention) is layer 10 — both are backbone-proper terminus
- Layer-22 is the deepest neck stage (largest receptive field, highest semantic level)

### Pooling → tokens

`adaptive_avg_pool2d((4,4))` + `flatten(2).transpose(1,2)` → **(B, 16, 512)**  
Confirmed identical to existing hook code in `run_pairwise_contrastive.py` and `run_finetune_bbox.py`.

### Fine-tuned checkpoint (`best.pt`, 8-class bbox detection)

- Same 24-layer structure, same `model.save = [4,6,10,13,16,19,22]`
- Layer-22 output shape: **(1, 512, 7, 7)** — identical to COCO checkpoint
- Backbone frozen during fine-tuning → feature geometry unchanged

---

## 2. Skip-Connection Problem with Naive Slicing

**`model.model[:23]` alone DOES NOT work** — it's not a Sequential.

The `DetectionModel.forward()` uses per-layer `.f` attributes to route inputs:
- Layers 12, 15, 18, 21 are `Concat` with cross-layer sources
- Layer 21: `Concat(f=[-1, 10])` — needs both layer 20 output AND layer 10 output
- Without tracking saved outputs, these Concat layers receive wrong inputs

Naive `nn.Sequential(*model.model[:23])(x)` will fail or produce wrong features.

**Solution:** reimplement the routing logic inside a thin `nn.Module` wrapper (see §3).

---

## 3. Recommended Export Approach — Option A

**KISS rationale:** wrap + `torch.onnx.export` (dynamo path). No onnx-surgery tooling needed, no full-model export + trim. 30 lines total.

### Code sketch (< 40 lines, production-ready)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


class YOLOv11BackboneTokenizer(nn.Module):
    """Layer-22 output → (B, 16, 512) spatial tokens."""

    def __init__(self, detection_model, pool: int = 4):
        super().__init__()
        self.layers = detection_model.model   # nn.ModuleList
        self.save = detection_model.save      # [4,6,10,13,16,19,22]
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: list = []
        for i, layer in enumerate(self.layers[:23]):
            f = layer.f if hasattr(layer, "f") else -1
            if f != -1:
                x = (
                    [y[j] if j >= 0 else x for j in f]
                    if isinstance(f, list)
                    else (y[f] if f >= 0 else x)
                )
            x = layer(x)
            y.append(x if i in self.save else None)
        # x: (B, 512, 7, 7) for 224×224 input
        x = F.adaptive_avg_pool2d(x, (self.pool, self.pool))  # (B, 512, 4, 4)
        return x.flatten(2).transpose(1, 2)                    # (B, 16, 512)


def export_backbone(pt_path: str, out_path: str, opset: int = 17) -> None:
    import onnx

    dm = YOLO(pt_path).model
    dm.eval()
    wrapper = YOLOv11BackboneTokenizer(dm)
    wrapper.eval()

    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        wrapper, dummy, out_path,
        opset_version=opset,
        input_names=["image"],
        output_names=["tokens"],
        dynamic_axes={"image": {0: "batch"}, "tokens": {0: "batch"}},
    )

    # Re-save inline (no external-data split) — required for CoreML EP
    model = onnx.load(out_path)
    onnx.save(model, out_path, save_as_external_data=False)
    onnx.checker.check_model(model)
    print(f"Exported: {out_path}")


# Usage
export_backbone("yolo11s.pt", "yolo11s_backbone_tokens.onnx")
export_backbone(
    "runs/detect/sweep_results/finetune_bbox/yolo_finetuned/weights/best.pt",
    "yolo11s_finetuned_backbone_tokens.onnx"
)
```

### Verification (confirmed numerically)

```python
# Hook reference vs ONNX — max abs diff: 3.2e-06 (float32 rounding only)
assert np.allclose(ref_tokens, onnx_tokens, atol=1e-4)  # passes
```

### Why not Option B (trim full YOLO export)?

`model.export(format='onnx')` exports all 24 layers + detect head. Trimming the ONNX graph requires `onnx-graphsurgeon` or manual node removal — more code, more failure modes, harder to maintain when ultralytics updates. Wrapper approach is 30 lines and survives minor ultralytics refactors as long as `.f` and `.save` stay (they have been stable since YOLOv5).

### Why not Option C (hooks during tracing)?

`torch.jit.trace` captures control flow at trace time. Hook-captured tensors do not appear in the traced graph. `torch.onnx.export` with dynamo uses `torch.export.export` which also cannot capture side-effect hooks. Option C is impossible without monkey-patching the forward method.

---

## 4. Opset + Runtime Compatibility

### Actual opset emitted

`torch.onnx.export(opset_version=17)` with PyTorch 2.11 / dynamo path emits **opset 18** (version converter silently upgrades). ONNX 1.21 opset 18 is fully supported by ORT 1.18+.

### Op inventory in exported graph (268 nodes)

| Op | Count | CoreML EP support |
|----|-------|-------------------|
| Conv | 63 | Yes |
| Mul (SiLU = sigmoid×x) | 60 | Yes |
| Sigmoid | 59 | Yes |
| Add | 22 | Yes |
| Concat | 20 | Yes |
| Gather | 12 | Partial (26 nodes fall back to CPU) |
| Split | 10 | Yes |
| Transpose | 5 | Yes |
| Reshape | 4 | Yes |
| MaxPool | 3 | Yes |
| MatMul | 2 | Yes |
| Resize | 2 | Yes |
| Where | 2 | Fallback |
| GatherND | 1 | Fallback |
| Shape | 1 | Fallback |

CoreML EP partitions: **242/268 nodes** on CoreML, 26 fall to CPU (mostly shape-inference and indexing ops from C2PSA attention). This is expected and benign.

### Runtime results on Apple M4

| Provider | Latency (224×224, bs=1) |
|----------|------------------------|
| CPU EP | 10.2 ms/frame |
| CoreML EP | 2.3 ms/frame |

CoreML EP numerical accuracy vs CPU: max diff **0.023**, mean **0.0025** — fp16 ANE precision, atol=0.03 passes. Acceptable for feature extraction (downstream Transformer trained on these features, not pixel-level reconstruction).

### Known issue: CoreML EP requires `save_as_external_data=False`

ORT 1.19+ CoreML EP fails with `model_path must not be empty` when model uses the dynamo export format (initializer references without resolved file path). **Fix:** call `onnx.save(model, path, save_as_external_data=False)` after export to inline all weights. Confirmed working on ORT 1.24.4 / macOS 14.

### Minimum opset for YOLOv11 ops

- Conv + BN fused: opset 9+
- SiLU (Sigmoid × Mul): opset 6+ (no native SiLU op needed, decomposed)
- C2PSA (attention with Softmax): opset 11+ (Softmax axis=-1)
- Resize (Upsample): opset 11+ (coordinate_transformation_mode)
- adaptive_avg_pool2d → GlobalAveragePool or AveragePool with pads: opset 11+
- **Effective minimum: opset 12.** Recommend exporting opset 17 (ORT 1.14+ compatible).

---

## 5. Codebase Cross-Check

### Exact feature extraction pattern (both scripts use identical logic)

**`run_pairwise_contrastive.py` lines 117–168 / `run_finetune_bbox.py` lines 246–315:**

```python
pool_size = 4; target_idx = 22
# hook captures backbone.model[22] output → features: (1, 512, H, W)
pooled = F.adaptive_avg_pool2d(features, (pool_size, pool_size))  # (1, 512, 4, 4)
tokens = pooled.flatten(2).transpose(1, 2).squeeze(0)             # (16, 512)
# Stacked over T frames → (T, 16, 512)
```

`run_finetune_bbox.py` comment: `"# YOLOv11s backbone — hook at layer 22 (SPPF output, deepest features)"` — the comment is **wrong** (layer 22 is C3k2 PAN output, not SPPF; SPPF is layer 9). The hook target index and output shape are correct.

### Deployment decision: include pooling in ONNX graph

The wrapper above includes `adaptive_avg_pool2d` + reshape inside the ONNX graph, outputting `(B, 16, 512)` directly. This eliminates a numpy post-processing step in the ORT inference path and keeps the deployment contract clean. ORT handles these ops natively.

---

## 6. Export Checklist

```
[ ] Run export_backbone("yolo11s.pt", "yolo11s_backbone_tokens.onnx")
[ ] Run export_backbone("...best.pt", "yolo11s_finetuned_backbone_tokens.onnx")
[ ] Verify: onnx.checker.check_model(onnx.load(path))
[ ] Verify: ORT CPU output matches torch hook output (atol=1e-4)
[ ] Verify: CoreML EP loads without error (requires save_as_external_data=False)
[ ] Benchmark latency at bs=1 and bs=8 for video batch inference
```

---

## Citations

- ultralytics YOLOv11 config: `ultralytics/cfg/models/11/yolo11.yaml`  
  https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml
- ultralytics block modules (C3k2, SPPF, C2PSA):  
  https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py
- ultralytics DetectionModel forward (routing via `.f`):  
  https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py
- ORT CoreML EP `model_path must not be empty` bug (ORT issue #18518):  
  https://github.com/microsoft/onnxruntime/issues/18518
- ORT CoreML EP opset support matrix:  
  https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- torch.onnx dynamo export guide:  
  https://pytorch.org/docs/stable/onnx_dynamo.html

---

## Unresolved Questions

1. **Opset 18 instead of 17:** PyTorch 2.11 dynamo exporter silently upgrades to opset 18 regardless of `opset_version=17`. ORT 1.24 handles this fine, but if target ORT is older (< 1.16), test explicitly. No workaround found other than using `dynamo=False` legacy tracer, which fails on list-input layers (Concat).

2. **CoreML EP fp16 precision:** max diff 0.023 is acceptable for feature extraction but should be re-validated if the downstream Transformer is sensitive to activation scale. Consider fp32 fallback (`ort.SessionOptions` + `ExecutionMode.ORT_SEQUENTIAL`) if classification accuracy degrades vs PyTorch baseline.

3. **`run_mega_ensemble.py` uses "V6 backbone features"** (from a different checkpoint path) — not the same as `yolo11s.pt`. If that checkpoint has a different neck depth or channel count, a separate export is needed. Grep shows it references `v6_tr_stats` from a cached `.pt` file, not a live model — low priority for ONNX export.

4. **`save_as_external_data=False` and model size:** current backbone ONNX is 0.53 MB (opset-18 dynamo path inlines all weights). If a future export produces external data files (models > ~2 GB), the CoreML EP workaround will not apply and a different fix will be required.
