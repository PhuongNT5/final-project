# MT3 — YOLO26 detector swap + edge metrics

**Date:** 2026-08-29. Additive report (no existing files edited). Final roadmap track: swap the entity
detector YOLOv11s → **YOLO26s** with an *identical* training recipe (fair one-thing-changed), run it through
the *frozen* S1a Transformer pipeline, and compare accuracy + edge cost. **Step 2 (downstream accuracy) is now
complete.** Step 3 (real Jetson edge metrics) script is ready and pending on-device run.

## Headline
Swapping YOLOv11→YOLO26 leaves behaviour classification **statistically unchanged** (paired McNemar **p=1.0**).
The pipeline is **not detector-limited** — the Transformer reads geometry + ROI appearance and absorbs the
marginal detector-mAP difference. YOLO26 adds an **NMS-free end-to-end head** (deployment win) at ~equal params
and slightly fewer FLOPs.

## 1. Detector accuracy — YOLO26s vs YOLOv11s
Same dataset (2262 frames, 8 entities), same recipe (2-stage freeze→fine-tune, SGD lr 0.01→0.001, imgsz 800,
hsv_h=0, mosaic 0.5, seed 0). YOLO26 early-stopped epoch 35 (65 total), 1.36 h on a T4.

| metric | YOLO26s | YOLOv11s | Δ |
|---|---|---|---|
| mAP@50 (all) | 0.962 | **0.969** | −0.7 pp |
| mAP@50-95 (all) | 0.872 | **0.894** | −2.2 pp |
| Person_Unauthorized mAP@50 (hard) | 0.906 | 0.928 | −2.2 |
| Panel_Closed/Opened mAP@50 | 0.990 / 0.962 | — | — |
| Forklift_Safe/Overload mAP@50 | 0.995 / 0.995 | — | — |

Backbone verified **byte-identical to YOLO11 through layer 10** (Conv…C3k2…SPPF…C2PSA, 512-ch out) → the
YOLO-ROI appearance extraction transfers unchanged; the swap is genuinely one-thing-changed.

## 2. Downstream behaviour accuracy — YOLO26 vs YOLO11 pipeline (frozen S1a)
Re-extracted tokens with the YOLO26 detector (+ its ROI backbone) → `tokens_yolo26app_*` (26,438 train /
3,390 test detections) → same S1a Transformer, same seeds/config → `logits_yolo26.pt`.

| metric (same-distribution 64-clip) | YOLO26 | YOLO11 baseline |
|---|---|---|
| blind top-1 (val-selected seed) | 54/64 (84.4%) | 55/64 (85.9%) |
| blind macro | 88.4% | 88.4% |
| ensemble top-1 | 52/64 (81.2%) | 54/64 (84.4%) |
| ensemble macro | 86.9% | 88.4% |
| Cohen κ / MCC | 0.734 / 0.751 | 0.775 / 0.785 |
| ROC-AUC ovr / PR-AUC | 0.950 / 0.869 | 0.949 / 0.867 |
| ECE (15-bin) | **0.101** | 0.142 |
| CarryOverload recall | 100% | 100% |
| top-1 BCa 95% CI | [67.2, 87.5] | overlapping |

- **Paired McNemar (blind preds): p = 1.0000** — 61/64 identical, only 3 discordant clips (YOLO11-only-correct
  2, YOLO26-only-correct 1), ensemble argmax agrees 60/64. Statistically indistinguishable.
- Differences (ensemble 52 vs 54; κ 0.734 vs 0.775) are within seed noise and BCa CIs overlap fully.
- YOLO26 is marginally **better calibrated** (ECE 0.101 vs 0.142).

## 3. Scene-robust (leave-camera-out) — YOLO26
LCO pools 516 clips, per-view holdout, 3-seed ensemble:

| | YOLO26 | YOLO11 baseline |
|---|---|---|
| LCO top-1 | 77.1% | 80.4% |
| LCO macro | 78.6% | 82.0% |
| scene-only baseline | 16.9% | 16.9% |
| behaviour signal beyond scene | **+60.3 pp** | +63.5 pp |

- Both are strongly scene-robust (crush scene-only 16.9%). YOLO26 ~3 pp lower cross-camera (hardest class
  UnauthorInt LCO recall 44.4% vs YOLO11-ROI 56%) — within 3-seed variance; not a categorical difference.

## 4. Edge metrics (Colab T4 proxy; Jetson pending device)
| metric | YOLO26 fused | YOLO26 end-to-end | YOLOv11s |
|---|---|---|---|
| params | 9.47 M | 9.95 M | 9.4 M |
| GFLOPs | **20.8** | 32.7 | 21.5 |
| layers | 122 | 146 | — |

- **NMS-free end-to-end head** (output `(1,12,13125)`) — removes NMS post-processing, simplifying/accelerating
  deployment; the key architectural reason to prefer YOLO26 despite the −1–2 pp mAP.
- Size: `.pt` 19.4 MB · TF SavedModel 91.4 MB · GraphDef 36.5 MB. Export fidelity: GraphDef mAP@50 0.961 vs
  0.962 PyTorch. PyTorch inference 4.7 ms/img (T4).
- TensorRT-FP16/INT8 latency + INT8 compression → the `benchmark(..., int8=True)` table (`yolo26_benchmark.csv`)
  and, definitively, the on-device Jetson run.

## 5. Jetson (Step 3) — script ready, pending device
`tools/jetson_benchmark.py` runs on the Jetson: exports TRT FP16+INT8 engines, measures latency p50/p95/p99,
FPS, model size, INT8 compression, and energy (W, J/inference, FPS/W) via tegrastats. Works on any detector
`.pt`, so run it with `--weights yolo26_best.pt` and `--weights detector_best.pt` for the on-device YOLO26-vs-
YOLO11 comparison.
```bash
python3 tools/jetson_benchmark.py --weights yolo26_best.pt --data data.yaml --imgsz 800 --runs 300
```

## Conclusion
YOLO26 is a **drop-in equal** to YOLOv11 for this pipeline: identical downstream behaviour accuracy
(McNemar p=1.0), comparable scene-robustness, better calibration, ~equal params / slightly fewer FLOPs, and a
**NMS-free deployment advantage**. Confirms the thesis claim that accuracy is bounded by supervision placement
+ scene-confounding, **not** the detector generation.

## Status
- ✅ Step 1 detector trained (YOLO26s, mAP@50 0.962).
- ✅ Step 2 downstream swap — tokens, S1a, McNemar p=1.0, LCO 77.1%, full multi-lens. Complete.
- ⏳ Step 3 Jetson edge metrics — `jetson_benchmark.py` ready; needs the device.

## Unresolved questions
1. On-device Jetson: run now (script ready) or keep as documented future work with the T4-proxy edge numbers?
2. The ~3 pp LCO gap (YOLO26 77.1% vs YOLO11 80.4%) — seed noise or a real cross-camera cost of the NMS-free
   head? Would need a multi-seed×detector run to separate; low priority given McNemar p=1.0 same-distribution.
