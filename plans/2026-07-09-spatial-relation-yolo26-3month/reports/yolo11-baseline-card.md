# YOLO11 Baseline Card (frozen comparison anchor)

**Date:** 2026-08-16. The canonical YOLOv11 result — the anchor for the paper comparison and the future
YOLO26 swap. Reproducible: CPU-deterministic, fixed seeds [0..6], per-clip logits saved.

## Configuration (frozen)
- **Detector:** YOLOv11s, COCO-pretrained, fine-tuned on 2,262 multi-entity frames (Colab CUDA),
  imgsz 800, mAP@50 = 0.969. Checkpoint `sweep_results/entity_detector/detector_best.pt`.
- **Classifier (S1a):** box-token spatial-relation Transformer, tokens =
  [class, xyxy(norm), conf, IoU_walkway, IoU_restricted, foot_in_walkway, norm_dist, **YOLOv11s-ROI-appearance(64)**]
  + per-frame auxiliary behaviour loss (aux_w=0.5), no class-weights, 7-seed ensemble, CPU.
  Appearance = the fine-tuned detector's own backbone ROI features (entity-conditioned) — chosen over
  MobileNet crops because it improves cross-camera (LCO) generalisation by +4.8 pp at equal same-dist.
- **Protocol:** blind held-out (epoch/seed on train-carved val; 64-clip test once) + leave-camera-out.

## Frozen numbers (64-clip 4-class unsafe test)
| Metric | same-distribution | leave-camera-out |
|---|---|---|
| top-1 | **54/64 = 84.4 %** (ensemble); mean 51.6±2.4; best-seed 55/64 | **80.4 %** |
| macro (mean recall) | **88.4 %** | **82.0 %** |
| macro-F1 | 85.6 % | — |
| balanced-acc | 88.4 % | — |
| Cohen κ / MCC | 0.775 / 0.785 | — |
| ROC-AUC / PR-AUC (ovr) | 0.949 / 0.867 | — |
| ECE (15-bin) | 0.142 | — |
| BCa CI (top-1 / macro) | [71.9, 90.6] / [78.0, 94.3] | — |
| per-class recall | Walk 78 · Unauth 91 · Panel 85 · Carry 100 | **Walk 80 · Unauth 56 · Panel 91 · Carry 100** |

**ROI vs MobileNet appearance:** same-distribution tie (54/64), but LCO **80.4 %** vs 75.6 % (+4.8 pp),
driven by UnauthorInt cross-camera recall **56 %** vs 35 % (+21 pp). Entity-conditioned features transfer
across cameras far better — the scene-robust metric that matters. Confirms the ceiling is data-bound, not
feature-bound (same-dist unchanged).

Artifacts: `logits_yoloroi.pt` (canonical, 64×4 seed-avg softmax), `logits_yolo11_baseline.pt` (MobileNet,
superseded), `full_eval.json`, `figures/relation_*.png`.

## Positioning
- vs paper no-bbox (57/64, 85.78 % macro): macro **above** (88.4 %); top-1 −4.7 pp (not sig, p=0.44).
- vs paper full-bbox (62/64, 95.81 %): below — but the paper is scene-inflated (whole-frame = scene-only
  = 81.2 % same-dist; collapses to 52.5 % under LCO). Our **75.6 % LCO** is the scene-robust headline.

## Comparison discipline for YOLO26 (MT3)
Change ONE thing — the detector `.pt` + re-extracted tokens — keep classifier/protocol/seeds identical;
compare against `logits_yolo11_baseline.pt` with a paired McNemar on the same 64 clips.

## Optional pre-freeze lever (research-backed, pending decision)
Swap MobileNet appearance → fine-tuned YOLOv11s ROI features (entity-conditioned): expected +1–2 clips
same-dist, LCO neutral/+, scene-robust, ~2 h. If it reaches 55–56/64 it replaces this card; else S1a stands.
