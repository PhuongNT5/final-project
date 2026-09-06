# Phase 02 — YOLO26 Migration + Entity Detector Fine-Tune

**Priority:** P1 · **Weeks:** W2–W6 · **Status:** ☐
**Context:** [researcher-yolo26-ultralytics.md](reports/researcher-yolo26-ultralytics.md), [Ultralytics tutorial](https://raw.githubusercontent.com/ultralytics/ultralytics/main/examples/tutorial.ipynb)

## Overview
Two independent tracks: (A) **backbone-swap baseline** — drop YOLO26 in as the frozen feature
extractor, re-extract, retrain the existing head → validates migration with *zero new labels*, start
W2. (B) **entity detector fine-tune** — train YOLO26 on the phase-01 multi-entity ontology → this is
the detector that feeds the relation Transformer, starts when 1.5k frames land (~W3).

## YOLO26 facts (verified)
Released 25 Sep 2025. **DFL-free** (`reg_max=1`), **NMS-free end-to-end** (one-to-one head, output
`(N,300,6)`), more stable INT8, native ByteTrack/BoT-SORT (`model.track(persist=True)`), AGPL-3.0.
Variants n/s/m/l/x (2.4–55.7 M params). **Backbone (C3k2) + neck (PAN-FPN P3/P4/P5) identical to
YOLOv11** → feature-tap workflow migrates; only the head differs. Orin Nano YOLO26n INT8 ≈ 3.8 ms.

## Track A — Backbone-swap baseline (no new labels, W2)
1. Load `yolo26s.pt`; **re-map the layer hook** — dual training head shifts indices:
   `for i,l in enumerate(model.model.model): print(i, l.__class__.__name__)` → find the C3k2 tap
   equivalent to current layer-22; verify pool4 output shape matches Transformer input (`T×16×C`).
2. Re-extract full feature cache on Unsafe-Net ([tools/preextract_yolo_features.py](../../tools/), adapt for YOLO26).
3. Retrain the current best head on YOLO26 features; evaluate under **blind + k-fold** ([phase-00](phase-00-preflight-and-review-fixes.md) template).
4. **Report:** YOLO26-frozen vs YOLOv11-frozen at matched protocol → is the newer backbone alone worth anything? (Likely marginal; documents the migration is clean.)

## Track B — Entity detector fine-tune (needs phase-01, W3–W6)
Two-stage transfer from COCO (research-recommended defaults):
- **Stage 1** (adapt head): `freeze=10, epochs=20, lr0=0.001, mosaic=0.5`
- **Stage 2** (refine backbone): `epochs=30, lr0=0.0001`, unfreeze; `patience=15`
- `data.yaml` with a **real train/val split** (never `val: images/train`); `cls_pw` tuned for class imbalance.
- Retrain on 1.5k first (W3), refresh at 3k (W5–W6).
- Enable **ByteTrack** for track-id continuity → stable box sequences for phase-03; interpolate occluded frames (trackers do not fill gaps).

## Geometric feature pipeline (bridge to phase-03)
Build `tools/build_relation_tokens.py`:
- Run YOLO26 detector (+ByteTrack) over each clip → per-frame boxes with `class, (x,y,w,h), track_id, conf`.
- Load per-view zone polygon (phase-01).
- Compute **relative geometry** per person box: `IoU(person, Safe_Walkway)`, `IoU(person, Restricted)`,
  `foot_in_walkway = polygon.contains(Point(cx, y_max))` (shapely), `norm_dist(person_centroid, zone_centroid)/zone_w`.
- Emit per-clip token tensors → cached like the current feature cache (input to [phase-03](phase-03-spatial-relation-transformer.md)).

## Migration risks (budget for these)
- Layer-index shift from dual head → re-map + shape-assert (blocking; do first).
- Backbone weights changed even if architecture same → **must** re-extract features (no reuse of old cache).
- Retrain head from scratch on new features (~1–2 wk, overlaps).
- No published YOLO26s Orin Nano number → measure our own in [phase-05](phase-05-jetson-and-thesis-integration.md).
- INT8 export needs TensorRT ≥ 10.7 on JetPack 6.2 (10.3 has an end2end INT8 bug).

## Todo
- [ ] Install/verify YOLO26 (ultralytics ≥ version shipping YOLO26); confirm license note for thesis
- [ ] Track A: re-map hook, shape-assert, re-extract features, retrain+eval frozen baseline
- [ ] Track B: build `data.yaml`; Stage-1/Stage-2 fine-tune on 1.5k → 3k
- [ ] Enable ByteTrack; add occlusion interpolation
- [ ] `tools/build_relation_tokens.py` → cached geometric token tensors
- [ ] Detector eval: mAP@50 per class on held-out annotation val; confusion-pair detection recall

## Success criteria
- Clean YOLO26 feature extraction (shape-verified); frozen baseline reported blind + k-fold vs YOLOv11.
- Entity detector **≥ 0.80 mAP@50** on annotation val; `Person_Authorized`/`Unauthorized` separated reliably.
- Geometric token cache produced for all 691 clips.

## Deliverables
- `yolo26_backbone_swap` report; fine-tuned entity `best.pt`; `build_relation_tokens.py` + token cache; migration notes.
