# Phase 02 — Entity Detector Results (Stage A) ✅

**Date:** 2026-08-09 · **Model:** `sweep_results/entity_detector/detector_best.pt` (YOLOv11s, 18 MB)

## Training
- **Venue:** Google Colab (CUDA T4) — local MPS training is broken on macOS 14.6 (addcmul_ bug; see `researcher-mps-training-fix.md`). Inference/val run fine locally on MPS.
- **Recipe:** yolo11s COCO-pretrained → 2-stage (freeze 10 / 30 ep @ lr0=0.01 → unfreeze / 100 ep @ lr0=0.001), SGD momentum 0.937, imgsz 800, batch 16, seed 0.
- **Key strategy:** `hsv_h=0` (no hue jitter) to preserve the green (Authorized) vs red-black (Unauthorized) vest signal. mosaic 0.5, patience 30.
- **Data:** 2262 frames (1919 train / 343 val), clip-level split (no leakage). Zones (cls 2,3) excluded from detection by design → per-view constants in `annotation_guide/zones/`.

## Validation (local, 343 val images)
**overall mAP50 = 0.969 · mAP50-95 = 0.894**

| Class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Person_Authorized | 0.979 | 0.921 | 0.938 | 0.853 |
| Person_Unauthorized | 0.816 | 0.937 | 0.928 | 0.732 |
| Panel_Closed | 0.987 | 0.946 | 0.989 | 0.964 |
| Panel_Opened | 0.881 | 1.000 | 0.972 | 0.913 |
| Forklift_Safe_Load | 0.970 | 1.000 | 0.995 | 0.919 |
| Forklift_Overload | 0.972 | 1.000 | 0.995 | 0.982 |

## Findings
- **Person_Authorized imbalance (365 samples) is NOT a problem** — mAP50 0.938, R 0.921. hsv_h=0 preserved the colour cue → **keep the 2-class ontology** (single-`Person`+colour-feature fallback not needed).
- Person_Unauthorized precision 0.816 (some false positives) but recall 0.937 — good for feeding the Transformer (recall > precision priority: don't drop people/tokens).
- Detector quality is strong across all 6 movable entities → solid Stage-A signal for Stage B.

## Open items
- Grab `runs.zip` from Drive (`unsafe-net-detector-out/`) for thesis figures (PR/F1 curves, confusion matrix).
- Person_Unauthorized precision could improve with a higher conf threshold at token-build time (tune in Stage B).

## Next → Stage B
`build_relation_tokens`: run `detector_best.pt` (+ByteTrack) over the clips → per-frame entity boxes;
combine with per-view zone geometry (foot-in-polygon, IoU person∩zone, norm distance) → box-token sequences
for the Transformer. Runs locally (inference on MPS/CPU is fine).
