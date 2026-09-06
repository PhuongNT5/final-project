# Phase 01 — Multi-Entity Annotation (~3,000 frames, CVAT)

**Priority:** P0 (long pole) · **Weeks:** W1–W5 (rolling) · **Status:** ☐
**Context:** [researcher-annotation-eval-tooling.md](reports/researcher-annotation-eval-tooling.md), [annotation_guide/](../../annotation_guide/)

## Overview
Annotate ~3,000 frames with a **multi-entity spatial-context ontology** so the detector emits the
entities a relation Transformer needs to *infer* behaviour geometrically — instead of asking YOLO to
name behaviours directly. This is the supervision the thesis identified as missing.

## Ontology (finalize W1 with advisor)
Detect entities, not behaviours. Behaviour is inferred downstream from geometry.

**Human entities**
- `Person_Authorized` — green safety vest
- `Person_Unauthorized` — red-black vest

**Environment zones** (annotate as **polygons** — painted walkway is non-rectangular)
- `Safe_Walkway` — green painted path (drawn once per view, see below)
- `Restricted_Zone` / `Machine_Area` — control-panel / machine region

**Object states**
- `Panel_Closed` / `Panel_Opened`
- `Forklift_Safe_Load` (≤2 blocks) / `Forklift_Overload` (≥3 blocks)

Total ≤ 8 entity classes (keep tight; blob classes hurt small-N detectors).

## CVAT workflow (concrete)
- One **Project** (shared label schema) → one **Task per camera view** (~6–8 fixed views).
- **Zones drawn once per view:** Track mode, polygon on frame 0 + identical keyframe at clip end →
  CVAT interpolates a static polygon pinned to every frame. ~5 min/zone/view (~30–45 min total).
- **People/objects:** Track mode + keyframe interpolation (1-in-5 keyframes); **SAM2-assisted** boxes
  on CVAT cloud where available → 2–4× throughput. Preserve `track_id` for temporal consistency.
- **Tightness rule (critical):** person box must hug the **feet** — `y_max` (foot point) is the decisive
  signal for in/out-of-walkway. Interpolate boxes through occlusion so the token sequence never breaks.
- **Export:** "Ultralytics YOLO 1.0" (→ `data.yaml` + normalized labels) for detection; "COCO 1.0"
  (→ `segmentation` polygons) for zones. `track_id` preserved in both.

## Sampling strategy (3,000 frames from 691 clips)
1. **Viewpoint balance:** ~400–500 frames/camera view.
2. **Confusion-pair focus (~40 % of budget):** oversample frames where WalkwayViolation and
   UnauthorizedIntervention co-occur or are visually confusable (the entire error budget lives here).
3. **Hard-case mining:** run current detector, collect top-2-confused frames → reserve 300–400 slots.
4. **Temporal density:** unsafe windows every 3rd frame; safe windows every 15th.
5. **Rare-class guard:** if `Forklift_Overload` raw clips < 20, reallocate quota + augment.

## Quality assurance
- 2 annotators on a shared 100-frame sample → Cohen's κ (`sklearn.cohen_kappa_score`) on class, IoU>0.5.
  Target **κ > 0.8**; adjudicate disagreements; refine guide (feeds the `run_kappa_*` scripts already in repo).
- CVAT QA/honeypot module for spot checks.

## Effort estimate
- Manual bbox: ~50 boxes/hr → ~300 h. Track+interp: ~150/hr → ~100 h. **SAM2-assisted track: ~50–75 h** (recommended).
- Plan for staged delivery: **1,500 frames by W3** (unblocks detector fine-tune), **3,000 by W5**.

## Related files
- Create/extend: `annotation_guide/ontology-multi-entity.md`, `tools/sample_frames_for_annotation.py` (sampling logic), zone polygon store `annotation_guide/zones/<view>.json`.
- Reuse: [extract_frames_for_annotation.py](../../extract_frames_for_annotation.py), [tools/prepare_cvat_import.py](../../tools/prepare_cvat_import.py), [tools/make_zones_auto.py](../../tools/make_zones_auto.py), [run_kappa_prepare.py](../../run_kappa_prepare.py), [run_kappa_compute.py](../../run_kappa_compute.py).

## Todo
- [ ] Audit raw clip/frame counts per class (feasibility of 3k budget)
- [ ] Finalize ontology + annotation guide (advisor sign-off on WalkwayViolation definition)
- [ ] CVAT project + per-view tasks; draw + store zone polygons per view
- [ ] Sampling script → export 3k frame list (viewpoint-balanced, hard-case-weighted)
- [ ] Annotate batch 1 (1,500) by W3; κ agreement pass
- [ ] Annotate batch 2 (→3,000) by W5
- [ ] Export YOLO + COCO; commit `data.yaml`, `labels/`, zone polygons

## Success criteria
- 3,000 frames labelled across 8 entity classes, viewpoint-balanced, confusion-pair-weighted.
- Static zone polygon per view committed and reusable at train + inference.
- κ > 0.8 on the agreement sample; clean YOLO `data.yaml` with a **train/val split that is not identical** (fixes the [run_finetune_bbox.py](../../run_finetune_bbox.py) `val: images/train` defect).

## Risks
- Overload class scarcity → reallocate + augment. Occlusion → interpolate. Velocity slip → SAM2 assist + train on partial.
