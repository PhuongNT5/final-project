# MT3 — YOLO26 detector swap + edge benchmark

Final roadmap track. Two comparisons the thesis wants, both isolated to *one thing changed* (the detector):
1. **Accuracy** — YOLO26 entity detector vs the frozen YOLOv11s baseline, re-run through the *same* S1a
   Transformer pipeline; paired McNemar + LCO scene-robust protocol.
2. **Edge** — YOLO26 params/FLOPs, model size, per-format latency/FPS, TensorRT-INT8 compression, energy.

## Step 1 — Colab (train YOLO26 detector) — ~1 h on T4/L4
1. `unsafe-net-detector-dataset.zip` is already in My Drive root (same zip the YOLO11 detector used). Reuse it.
2. Open `colab/train_yolo26_detector_colab.ipynb`, Runtime → GPU, Run all.
3. Recipe is **identical** to the YOLO11 detector (2-stage freeze→fine-tune, SGD lr 0.01→0.001, imgsz 800,
   hsv_h=0, mosaic 0.5, seed 0) — so any accuracy delta is the detector, not the training.
4. Cell 6 prints per-class mAP → compare to YOLO11 (**mAP@50 0.969 / mAP@50-95 0.894**).
5. Cell 7 prints the **edge metrics** (params, GFLOPs, per-format size + latency incl. INT8) — Colab GPU as a
   Jetson proxy.
6. Cell 8 saves `yolo26_best.pt` + `runs26.zip` to `My Drive/unsafe-net-yolo26-out/`.
7. If `yolo26s.pt` is unavailable in the installed ultralytics, the cell notes it — fall back to the newest
   YOLO variant and record the version (the swap logic is unchanged).

## Step 2 — Local (accuracy comparison) — I run this when weights land
```bash
# re-extract tokens with the YOLO26 detector (+ its ROI-appearance backbone), clean + LCO
.venv/bin/python build_relation_tokens.py --detector yolo26_best.pt \
    --appearance --appearance_backbone yolo   # -> tokens_yolo26app_{train,test}.pt
# re-run the frozen S1a pipeline (same seeds/config as the YOLO11 baseline)
.venv/bin/python run_relation_transformer.py --tag yolo26app_ --aux_w 0.5 --save_logits logits_yolo26.pt
# paired comparison vs the YOLO11 baseline
.venv/bin/python run_full_eval.py --logits logits_yolo26.pt          # multi-lens on YOLO26
.venv/bin/python run_lco_eval.py  --tag yolo26app_                   # scene-robust (LCO)
```
Report table: detector mAP · same-dist top-1/macro · LCO top-1/macro · McNemar vs YOLO11 (`logits_yoloroi.pt`).

## Step 3 — Jetson (real edge metrics) — needs the device
Copy `yolo26_best.pt` to the Jetson, then run cell 7's `benchmark(...)` + INT8 `export(format='engine',
int8=True, ...)` **on the Jetson**. Collect:

| metric | how |
|---|---|
| latency p50/p95/p99 (ms) | 200-run timed loop, batch 1, imgsz 800 |
| throughput (FPS) | 1000 / mean-latency; + multi-stream (2/4 cams) zero-drop |
| model size (MB) | `.pt` / `.onnx` / `.engine` (FP16, INT8) |
| params / GFLOPs | `model.info()` |
| INT8 compression | size(FP16 engine) / size(INT8 engine) |
| energy | `tegrastats` → J/inference, FPS/W |

Without a Jetson: the Colab-GPU numbers stand as a proxy, the earlier pipeline's Jetson artifacts remain the
deployment baseline, and full Jetson benchmarking is documented future work.

## Deliverables
- `logits_yolo26.pt`, YOLO26 mAP + edge tables → a new additive report
  `plans/2026-07-09-spatial-relation-yolo26-3month/reports/mt3-yolo26-results.md` (no edits to existing files).
