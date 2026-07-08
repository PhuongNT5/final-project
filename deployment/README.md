# Deployment — Edge Inference Pipeline

ONNX exports, INT8 quantization, parity verification for the YOLO-Transformer
unsafe behavior detection pipeline. Source models: frozen checkpoints from
`runs/detect/.../yolo_finetuned/weights/best.pt` and `sweep_results/pairwise/*.pt`.

## Contents

```
deployment/
├── export/                          # ONNX outputs (FP32 + INT8)
│   ├── backbone-fp32.onnx           # yolo11s COCO, 35 MB
│   ├── backbone-ft-fp32.onnx        # fine-tuned (bbox), 35 MB
│   ├── *-fp32.onnx                  # 7 transformer heads
│   ├── *-int8.onnx                  # 9 dynamic INT8 variants
│   └── *.data                       # external weight sidecars (auto-generated)
├── scripts/
│   ├── export-backbone.py           # YOLOv11BackboneTokenizer → ONNX
│   ├── export-transformer-heads.py  # 7 heads: direct4, scenario, pair0-3, eight
│   ├── transformer-head.py          # TransformerHead class (verbatim copy)
│   ├── quantize-int8.py             # ORT dynamic INT8
│   └── verify-onnx-parity.py        # synthetic FP32/INT8 drift check
└── artifacts/
    └── parity-report.txt
```

## Reproduce

```bash
# 1. Export + quantize + parity check (Phase 01)
.venv/bin/python3 deployment/scripts/export-backbone.py
.venv/bin/python3 deployment/scripts/export-transformer-heads.py
.venv/bin/python3 deployment/scripts/quantize-int8.py
.venv/bin/python3 deployment/scripts/verify-onnx-parity.py

# 2. Extract ONNX-backbone features for accuracy eval (Phase 02)
.venv/bin/python3 deployment/scripts/extract-features-onnx.py

# 3. Benchmark (100 runs, 4 configs: CPU FP32/INT8 + CoreML FP32/INT8)
.venv/bin/python3 deployment/scripts/benchmark-harness.py
#   → deployment/artifacts/benchmarks.{json,md} + accuracy-report.txt

# 4. Multi-stream pipeline scaling test (Phase 03)
.venv/bin/python3 deployment/scripts/multi-stream-pipeline.py \
    --streams 1 2 3 4 \
    --source data/test/0_safe_walkway_violation/0_te1.mp4 \
             data/test/1_unauthorized_intervention/1_te1.mp4 \
             "data/test/2_opened_panel cover/2_te1.mp4" \
             data/test/3_carrying_overload_with_forklift/3_te1.mp4 \
    --max-frames 240 --fps 30
#   → deployment/artifacts/multi-stream-results.json

# 5. Streamlit demo (Phase 04)
.venv/bin/python3 -m streamlit run deployment/dashboard/app.py

# 6. Generate thesis figures (Phase 05)
.venv/bin/python3 deployment/scripts/generate-figures.py
#   → deployment/artifacts/figures/*.png

# 7. Netron graph screenshot (manual)
netron deployment/export/direct4-s314-fp32.onnx
# Save screenshot → deployment/artifacts/figures/onnx-graph-direct4.png
```

All scripts are idempotent; pass `--force` to re-export/re-extract.

## Parity (Phase 01)

Synthetic random-input comparison, batch=1 loop:

| model         | PT vs FP32 | PT vs INT8 |
|---------------|-----------:|-----------:|
| backbone      |   2.98e-06 |   3.61e-01 |
| direct4-s314  |   1.61e-06 |   3.56e-02 |
| scenario-s7   |   1.67e-06 |   2.41e-02 |
| pair{0..3}-s7 |   < 5e-07  |   < 1e-02  |
| eight-s7      |   2.38e-06 |   2.64e-02 |

FP32 drift is float32 rounding only. INT8 drift on heads is small. Backbone INT8
max drift (0.36) reflects wide activation range; end-to-end accuracy is verified
in Phase 02 on real test videos.

## Known caveats

- **Dynamo export bakes batch=1.** Exported graphs have reshape nodes that fail
  at batch > 1 despite `dynamic_axes` declaration. Use batch=1 inference (which
  is our production path anyway: one video at a time).
- **External data sidecars.** `.onnx.data` files accompany large backbones.
  Keep them together when copying. CoreML EP requires them present; OK to leave.
- **Shape inference + dynamo.** Quantization trips `onnx.shape_inference` on
  transformer attention reshapes. `quantize-int8.py` strips `value_info` before
  quantize to work around it.
- **Opset.** `torch.onnx.export(opset_version=17)` silently emits opset 18 under
  PyTorch 2.11 dynamo. Needs ORT >= 1.18 (we use 1.24.4).
- **Previous PROGRESS.md claim of "1.39 ms CPU INT8" is superseded.** Researcher
  measured 10.2 ms CPU / 2.3 ms CoreML ANE for the backbone alone (bs=1) on M4;
  full-pipeline numbers are in Phase 02 (CPU-FP32 e2e 26.9 ms p50).

## Measured results (Phase 02, 03)

**Accuracy (4-class unsafe subset, 64 videos):**

| Config | Direct4 | Cascade | Ensemble |
|--------|--------:|--------:|---------:|
| CPU-FP32 | 55/64 | 51/64 | **56/64** ← matches training baseline |
| CPU-INT8 | 55/64 | 51/64 | 55/64 |
| COREML-FP32 | 55/64 | 51/64 | **56/64** |
| COREML-INT8 | 55/64 | 51/64 | 55/64 |

**Latency p50 / p95 (batch=1, 100 runs):**

| Config | e2e p50 | e2e p95 | FPS |
|--------|--------:|--------:|----:|
| CPU-FP32 | 26.9 ms | 32.5 ms | 37.2 |
| CPU-INT8 | 34.0 ms | 37.5 ms | 29.4 |
| COREML-FP32 | 47.8 ms | 53.4 ms | 20.9 |
| COREML-INT8 | 88.2 ms | 108.3 ms | 11.3 |

**Multi-stream scaling (CPU-FP32, 4 distinct videos, 240 frames each, honest end-to-end — queue fully drained):**

| N | agg FPS | batch util | p50 head (ms) | drops |
|---|--------:|-----------:|--------------:|------:|
| 1 | 17.6 | 100% | 55.1 | 0 |
| 2 | 29.1 | 54% | 34.8 | 0 |
| 3 | 26.2 | 57% | 40.6 | 0 |
| 4 | 20.8 | 50% | 44.6 | 0 |

Aggregate throughput peaks at N=2 (~29 FPS) then declines at N≥3 as the shared
CPU backbone saturates. No frame drops across all configurations. Head latency
p50 reflects per-stream head inference; single-stream N=1 is higher because the
window emits less frequently relative to elapsed wall time.

## Reference

- Architecture + wrapper rationale: [plans/reports/researcher-track-b-yolo11s-onnx-export.md](../plans/reports/researcher-track-b-yolo11s-onnx-export.md)
- Jetson projection targets: [plans/reports/researcher-track-b-jetson-orin-projection.md](../plans/reports/researcher-track-b-jetson-orin-projection.md)
- Phase spec: [plans/260417-1913-track-b-edge-deployment/phase-01-export-pipeline.md](../plans/260417-1913-track-b-edge-deployment/phase-01-export-pipeline.md)
