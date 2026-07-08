# Phase 01 — Export Pipeline

**Status:** ready (research-validated)
**Priority:** P1 (blocks everything else)
**Estimated time:** 4h (reduced from 6h — researcher already validated architecture and measured latency)
**Dependency:** none

---

## Research Update (2026-04-18)

Researcher report: [plans/reports/researcher-track-b-yolo11s-onnx-export.md](../reports/researcher-track-b-yolo11s-onnx-export.md)

**Key corrections — MUST follow researcher report, not original plan notes:**
- Layer 22 is `C3k2` (P5 PAN output), NOT SPPF. SPPF is layer 9. Output (1,512,7,7).
- `model.model[:23]` slicing silently corrupts features. Must reimplement routing via each layer's `.f` attribute (layers 12/15/18/21 are Concat with cross-layer skips).
- Exact wrapper code + parity verification in researcher report §3.
- CoreML EP requires `onnx.save(model, path, save_as_external_data=False)` after export (ORT issue #18518).
- **Already measured**: backbone ORT CPU INT8 = 10.2ms, CoreML ANE = 2.3ms (4.4× speedup). Supersedes old PROGRESS.md "1.39ms" claim.

---

## Context Links

- Source: `run_pairwise_contrastive.py` — reference for TransformerHead arch and feature extraction hook
- Models: `sweep_results/pairwise/*.pt`, `runs/detect/.../yolo_finetuned/weights/best.pt`
- ONNX docs: onnxruntime 1.17+, onnx 1.15+
- Researcher report: [../reports/researcher-track-b-yolo11s-onnx-export.md](../reports/researcher-track-b-yolo11s-onnx-export.md)

---

## Overview

Rebuild `deployment/` from scratch. Export every model used in the best ensemble to ONNX
(FP32 + dynamic INT8). Verify numerical parity against PyTorch on the 64-clip test set.
Output is a self-contained `deployment/` directory that Phase 02-05 all depend on.

---

## Key Insights

- TransformerHead uses `torch.arange(t, device=d)` inside `forward()` — dynamic shapes must be
  handled via `dynamic_axes` at export, or the shape must be fixed at T=32 (simpler, KISS).
- The `forward()` method uses in-graph embedding lookups (`self.tpos`, `self.te`) — these are
  compatible with ONNX export but need opset >= 13 for `Embedding` with dynamic indices.
- Feature aggregation (mean+std+max over 16 spatial tokens → stats_dim=1536) happens OUTSIDE
  the TransformerHead in training scripts. For deployment, this aggregation must be either:
  (a) fused into the head's ONNX graph, or (b) kept as a numpy pre-processing step.
  Recommendation: keep as numpy step for simplicity (KISS) — avoids rewriting TransformerHead.
- The backbone extraction uses a forward hook on `model.model[22]`. ONNX export of the full
  backbone with a hook is non-trivial. Better approach: wrap backbone in a thin `BackboneWrapper`
  nn.Module that calls backbone and returns layer-22 output directly.
- INT8 dynamic quantization via `torch.quantization.quantize_dynamic` is simplest for CPU
  deployment. Alternative: ONNX Runtime quantization tool. Use ORT tool (more portable).
- RESEARCH FLAG: Confirm exact ONNX export path for YOLOv11s spatial-feature extractor.
  The ultralytics YOLO model structure may have changed in recent versions — hook index 22
  may not be stable. Need to validate `backbone.model[22]` still outputs (B, 512, H, W) tokens
  for yolo11s. Recommend delegating a researcher to verify against ultralytics 8.x changelog.

---

## Requirements

### Functional
- Export `BackboneWrapper(yolo11s.pt)` → `backbone_fp32.onnx`, input: (1,3,224,224), output: (1,16,512) [T=1 frame]
- Export `BackboneWrapper(yolo_finetuned/best.pt)` → `backbone_ft_fp32.onnx` (same interface)
- Export each TransformerHead `.pt` → `{name}_fp32.onnx`, input: (1,32,1536), output: (1,nc)
- Apply ORT INT8 dynamic quantization to all ONNX models → `{name}_int8.onnx`
- Verify: max abs diff FP32 ONNX vs PyTorch < 1e-3 on all 64 test clips
- Verify: INT8 accuracy on 64-clip test set preserved (must not drop below 55/64 = 85.9%)

### Non-functional
- Each export script < 200 lines (dev rules)
- Scripts idempotent: skip re-export if file exists (like training scripts)
- No new model training

---

## Related Code Files

**Create:**
- `deployment/scripts/export-backbone.py` — backbone wrapper + ONNX export
- `deployment/scripts/export-transformer-heads.py` — bulk export all 7 heads
- `deployment/scripts/verify-onnx-parity.py` — parity check on 64 test clips
- `deployment/export/` — output directory for all .onnx files

**Read (do not modify):**
- `run_pairwise_contrastive.py` — TransformerHead class definition (copy verbatim)
- `sweep_results/pairwise/*.pt` — model weights
- `sweep_results/pairwise/features/all8_test.pt` — test features for parity check

---

## Implementation Steps

### Step 1 — Create `deployment/` skeleton (15 min)
```
deployment/
├── export/
├── scripts/
├── dashboard/
├── artifacts/figures/
└── README.md
```

### Step 2 — Write `export-backbone.py` (1.5h)

```python
# deployment/scripts/export-backbone.py
# Wraps YOLOv11s backbone; exports layer-22 features to ONNX.
# Input:  (1, 3, 224, 224) RGB normalized frame
# Output: (1, 16, 512) spatial tokens after adaptive_avg_pool2d(4,4) + flatten

class BackboneWrapper(nn.Module):
    def __init__(self, yolo_path, pool_size=4, target_layer=22):
        # load ultralytics YOLO, freeze, register hook to capture layer output
        ...
    def forward(self, x):
        # run backbone, return pooled tokens (B, pool_size*pool_size, channels)
        ...

# Export: torch.onnx.export(..., opset_version=17, dynamic_axes={'x': {0: 'batch'}})
# Also export fine-tuned variant
```

Key concern: hook-based extraction doesn't naturally map to ONNX graph.
Solution: restructure `forward()` to call backbone layers 0..22 sequentially by slicing
`backbone.model[:23]` — avoids hooks entirely, fully exportable.

### Step 3 — Write `export-transformer-heads.py` (1h)

Copy `TransformerHead` class from `run_pairwise_contrastive.py` verbatim.
Export loop:
```python
HEADS = {
    'direct4_s314': ('sweep_results/pairwise/direct4_s314.pt', 4),
    'scenario_s7':  ('sweep_results/pairwise/scenario_s7.pt',  4),
    'pair0_s7':     ('sweep_results/pairwise/pair0_Walkway_s7.pt', 2),
    'pair1_s7':     ('sweep_results/pairwise/pair1_Intervention_s7.pt', 2),
    'pair2_s7':     ('sweep_results/pairwise/pair2_Panel_s7.pt', 2),
    'pair3_s7':     ('sweep_results/pairwise/pair3_Forklift_s7.pt', 2),
    'eight_s7':     ('sweep_results/pairwise/eight_s7.pt', 8),
}
# stats_dim = 1536, fixed T=32, B=1
dummy_input = torch.zeros(1, 32, 1536)
torch.onnx.export(model, dummy_input, out_path, opset_version=17,
                  input_names=['features'], output_names=['logits'],
                  dynamic_axes={'features': {0: 'batch'}})
```

### Step 4 — Apply ORT INT8 quantization (30 min)

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
```
Run for all 9 ONNX files (1 backbone + 7 heads + 1 backbone_ft).

### Step 5 — Write `verify-onnx-parity.py` (1.5h)

Load cached test features from `sweep_results/pairwise/features/all8_test.pt`.
For each head:
1. Run PyTorch model → logits_pt
2. Run ORT FP32 session → logits_ort_fp32
3. Run ORT INT8 session → logits_ort_int8
4. Assert max abs diff (pt vs fp32) < 1e-3
5. Compute INT8 accuracy on 64-clip subset; report vs original 56/64

Output: `deployment/artifacts/parity-report.txt`

### Step 6 — Smoke test full inference chain (30 min)

Manual run: pick 1 test video → frame sample → ONNX backbone → stats aggregation →
ONNX direct4 head → softmax. Confirm class and confidence match PyTorch pipeline.

---

## Todo List

- [ ] Create `deployment/` directory structure
- [ ] Validate layer-22 hook index for yolo11s in current ultralytics version
- [ ] Write `export-backbone.py` with sequential backbone slicing (no hooks)
- [ ] Write `export-transformer-heads.py` bulk exporter
- [ ] Apply ORT INT8 quantization to all models
- [ ] Write `verify-onnx-parity.py` parity checker
- [ ] Run parity check; confirm FP32 diff < 1e-3
- [ ] Run INT8 accuracy check; confirm ≥ 55/64 on direct4 head
- [ ] Update `deployment/README.md` with exported model inventory

---

## Success Criteria

1. All 9 `.onnx` files (FP32) present in `deployment/export/`
2. All 9 `_int8.onnx` files present
3. Parity report: max abs diff < 1e-3 for all FP32 heads
4. INT8 direct4_s314: accuracy ≥ 55/64 (no catastrophic degradation from 56/64)
5. Backbone ONNX: output shape (1, 16, 512) confirmed on dummy input
6. All scripts < 200 lines each

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| layer-22 hook index wrong for current ultralytics | Medium | High | Read ultralytics yolo11s source to confirm; use model introspection `print(backbone.model)` |
| TransformerHead `torch.arange` non-exportable | Low | Medium | Fix: pass `t` as fixed 32 at export; dynamic shape not needed for deployment |
| INT8 quant drops accuracy below 55/64 | Low | Medium | Fall back to FP32 for affected head; thesis can cite FP32 latency |
| ORT CoreML provider unavailable on macOS | Low | Low | Phase 02 handles this; export doesn't require CoreML |
| ONNX opset incompatibility with TransformerHead | Low | Medium | Test opset 14, 16, 17 — use highest compatible |

---

## Research Flag

**DELEGATE TO RESEARCHER:** "What is the correct ONNX export path for YOLOv11s spatial-feature
extractor using ultralytics 8.x? Specifically: does `model.model[:23]` slice work for layer-0 to
layer-22, and does layer-22 output (B, 512, H, W) feature maps compatible with backbone wrapper
design? Check ultralytics GitHub for yolo11s architecture docs or source code."

---

## Next Phase

Phase 02 (Benchmark Harness) depends on all ONNX files in `deployment/export/`.
Phase 03 (Multi-stream) also depends on Phase 01 completion.
