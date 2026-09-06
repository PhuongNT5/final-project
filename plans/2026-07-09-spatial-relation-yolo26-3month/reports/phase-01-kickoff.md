# Phase 01 — Kickoff (multi-entity annotation started)

**Date:** 2026-07-11 · **Status:** scaffolding done; ready for CVAT annotation.

## 1. Removed the old frames + annotation (as requested)
- Old V10 detection dataset **1600 frames + 1600 labels (896 MB, 4 behaviour classes, `val: images/train` leakage)**
  moved to `.archive/v10-yolo-dataset-20260711/` (not hard-deleted — it was gitignored, so unrecoverable
  otherwise; it holds real hand-labels + is the V10 negative-result training data).
- **Preserved (untouched):** V10 trained heads `sweep_results/finetune_bbox/direct4_s*.pt`, `results.json`,
  `features/bbox_*.pt`, and the git-tracked finetuned YOLO `runs/.../yolo_finetuned/weights/best.pt`.
  → the thesis's V10 evidence is intact; only the raw frames/labels were cleared.
- To reclaim the 896 MB permanently: `rm -rf .archive/` (your call — I left it recoverable).

## 2. New ontology (entities, not behaviours)
`annotation_guide/ontology-multi-entity.md` — 8 entity classes:
`Person_Authorized(green vest)`, `Person_Unauthorized(red-black)`, `Safe_Walkway(polygon)`,
`Restricted_Zone(polygon)`, `Panel_Closed/Opened`, `Forklift_Safe_Load/Overload`.
Behaviour is inferred downstream from geometry (foot-point-in-walkway, IoU person∩zone). Key rules:
foot-point-tight person boxes, zones drawn **once per view**, track-id continuity with occlusion interpolation.

## 3. Frame set sampled — 2,262 frames (paper-matched)
`tools/sample_frames_for_annotation.py` → `annotation_guide/frame_manifest.csv`. From **train clips only**
(64-clip test set untouched). Verified: **0 clips span both splits** (no leakage), all 566 train clips covered.
Count **matches the paper's YOLO training set exactly** (Önal & Dandıl: 2,262 labelled images).

| Source class | Frames | | Split | Frames |
|---|--:|---|---|--:|
| 0 WalkwayViol (confusion pair) | 550 | | train | 1,919 |
| 1 UnauthorInt (confusion pair) | 510 | | val | 343 (15 %) |
| 2 OpenedPanel | 290 | | **total** | **2,262** |
| 3 CarryOverload | 200 | | | |
| 4 SafeWalkway (contrastive neg.) | 200 | | **confusion pair** | 1,060 |
| 5 AuthorizedInterv | 184* | | entity coverage | 818 |
| 6 ClosedPanel | 152* | | contrastive neg. | 384 |
| 7 SafeCarrying | 176* | | | |

\* classes 5/6/7 cap below target — only 23/19/22 source clips exist; shortfall redistributed to the
confusion pair + opened-panel (the scientific priority). View-balanced across the 8 camera views
(safe-class frames have no view label — assign zones per physical view in CVAT).

## 4. YOLO-ready dataset skeleton
`annotation_guide/dataset/` with `images/{train,val}`, `labels/{train,val}`, `data.yaml` (**nc=8**, entity
names, **real train≠val split**). Zones store: `annotation_guide/zones/`. CVAT schema: `annotation_guide/cvat-labels.json`.

## 5. Git hygiene
`.gitignore` updated: ignores `.archive/`, `dataset/images/`, `frames_to_annotate/`, `*.jpg/*.png` under
`annotation_guide/`. **Tracked:** manifest CSV, ontology, CVAT schema, `data.yaml`, `labels/**`, `zones/**`.

## Next actions
1. ✅ **Frames extracted**: **2,262 JPEGs** at native 1920×1080 (1,919 train / 343 val), ≈1.2 GB in
   `annotation_guide/dataset/images/{train,val}/`. Manifest rows == images on disk == 2,262 (4 undecodable
   last-frames auto-replaced by same-class readable frames, split-matched to avoid leakage; re-verified 0 leak).
   Detector decision: **YOLOv11s first → YOLO26 for edge** (Stage A).
2. **CVAT setup:** one Project (import `cvat-labels.json`) → one Task per camera view; draw each
   `Safe_Walkway`/`Restricted_Zone` polygon once per view (Track mode) → store to `annotation_guide/zones/<view>.json`.
3. **Annotate** entities with track-ids + occlusion interpolation (SAM2-assist if on CVAT cloud).
   Est. 50–75 h SAM2-assisted / ~100 h track+interp.
4. **QA:** 2-annotator 100-frame overlap → Cohen's κ (reuse `run_kappa_prepare.py`/`run_kappa_compute.py`); target κ > 0.8.
5. **Export:** CVAT "Ultralytics YOLO 1.0" → `dataset/labels/{train,val}` → feeds Phase 02 YOLO26 detector.

## Open questions (advisor, from plan.md)
- WalkwayViolation definition (person-outside-walkway vs any-person-in-restricted) — drives whether cls4/cls5 negatives suffice.
- Zones at inference: training-label only, or bundled runtime rule engine?
- Confirm 2,992 ≈ 3,000 is the agreed budget (vs Ch6's ≥4,000 recommendation).
