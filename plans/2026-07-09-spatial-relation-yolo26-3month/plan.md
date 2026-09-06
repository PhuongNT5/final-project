# 3-Month Plan — Spatial-Context Ontology + YOLO26 to Break the WalkwayViol ↔ UnauthorInt Ceiling

**Window:** 2026-07-13 → 2026-10-09 (~13 weeks) · **Owner:** thesis author · **Status:** DRAFT / not started

## Thesis hypothesis under test
The current thesis concludes the accuracy ceiling (~54/64) is **zone-supervision-bound, not
architecture-bound** (Ch6 Finding 2/3). This plan is the *decisive experiment*: supply the missing
spatial supervision via a **multi-entity bbox ontology** (people + zones + object-states), let a
**box-token spatial-relation Transformer** infer behaviour from geometry (person-in-zone via
attention), and re-measure under the blind protocol. Outcome is publishable either way:
- **Positive:** confusion pair breaks → top-1 climbs toward the full-bbox 95.81 %; hypothesis confirmed at scale.
- **Negative:** ceiling holds even at 3k frames + relations → hypothesis *strengthened* (bound is even higher), a stronger negative result than the 400-frame V10 pilot.

## Core design decision — paper-faithful re-implementation (2026-07-11)
**Re-implement the paper's 3-stage pipeline from scratch, swapping components:** YOLOv4 → **YOLO26**
(frame-selection gate + entity detector), ConvLSTM → **Transformer** (temporal + spatial-relation
classifier), warning system unchanged. Detect **entities**, not behaviours; the Transformer infers
behaviour from geometry. Per-entity token = ROI appearance feature ⊕ normalized coords [xc,yc,w,h] ⊕
entity-class embedding ⊕ relative geometry (IoU person∩zone, foot-in-walkway, norm distance) ⊕
ByteTrack id. Relative geometry >> absolute coords at small N (DAB-DETR / Relation DETR / Action
Genome). Full spec: [paper-faithful-pipeline-design.md](reports/paper-faithful-pipeline-design.md);
head detail: [phase-03](phase-03-spatial-relation-transformer.md).

## Phases
| # | Phase | Weeks | Depends on | Status |
|---|-------|-------|-----------|--------|
| 00 | [Pre-flight: review fixes + rigor scaffolding](phase-00-preflight-and-review-fixes.md) | W1 | — | ☐ |
| 01 | [Multi-entity annotation — ~3,000 frames (CVAT)](phase-01-multi-entity-annotation.md) | W1–W5 | 00 | ☐ |
| 02 | [YOLO26 migration + entity detector fine-tune](phase-02-yolo26-and-detector.md) | W2–W6 | 00, (01 for detector) | ☐ |
| 03 | [Box-token spatial-relation Transformer](phase-03-spatial-relation-transformer.md) | W6–W9 | 01, 02 | ☐ |
| 04 | [Evaluation rigor — paper replication + multi-lens + robustness](phase-04-evaluation-rigor.md) | W9–W11 | 03 | ☐ |
| 05 | [Jetson YOLO26 deployment + thesis integration](phase-05-jetson-and-thesis-integration.md) | W10–W13 | 02, 04 | ☐ |

## Timeline (parallelism)
```
W1   W2   W3   W4   W5   W6   W7   W8   W9   W10  W11  W12  W13
[00]
     [====== 01 annotation (rolling) ======]
     [===== 02 YOLO26 backbone-swap baseline =====][detector FT]
                              [===== 03 relation transformer =====]
                                                  [== 04 eval ==]
                                        [===== 05 Jetson + writeup =====]
```
Annotation (01) is the long pole → start W1, run rolling. YOLO26 **backbone-swap baseline** (02) needs
no new labels → run in parallel to validate the migration early. The **entity detector fine-tune**
(02) and the **relation head** (03) consume annotations as they land (train on first 1.5k, refresh at 3k).

## Success criteria (plan-level)
1. **Primary (science):** blind held-out top-1 on the 64-clip 4-class test measurably beats the current
   honest 53/64 (85.4 % macro), with the WalkwayViol↔UnauthorInt confusion (currently ~5–6 of 8 errors)
   at least halved — OR a rigorous negative result at 3k frames. Report on the blind protocol only.
2. **Engineering:** YOLO26 entity detector ≥ 0.80 mAP@50 on held-out annotation val; runs on Jetson.
3. **Rigor:** every headline number carries a BCa bootstrap CI; blind-protocol + k-fold both reported and
   reconciled (closes the 53/64 vs 71.9 % gap); no test-set model selection anywhere.
4. **Deployment:** YOLO26 INT8 TensorRT on Orin Nano, multi-stream zero-drop preserved, power measured.
5. **Thesis:** review fixes applied; abstract + unified bibliography + missing figures done.

## Key dependencies / assets already in repo
- Annotation scaffolding: [annotation_guide/](../../annotation_guide/), [tools/prepare_cvat_import.py](../../tools/prepare_cvat_import.py), [tools/make_zones_auto.py](../../tools/make_zones_auto.py), [tools/draw_zone.py](../../tools/draw_zone.py)
- Blind protocol (reuse, do not re-invent): [run_blind_eval.py](../../run_blind_eval.py), [run_kfold_eval.py](../../run_kfold_eval.py)
- Frame-gate + bbox finetune precedent: [run_yolo_framegate.py](../../run_yolo_framegate.py), [run_finetune_bbox.py](../../run_finetune_bbox.py)
- Feature extraction / backbone hook: [model_arch.py](../../model_arch.py), [tools/preextract_yolo_features.py](../../tools/)

## Risk register (top 5)
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Class Forklift_Overload has too few raw clips for 3k budget | Med | Med | Audit clip counts W1; reallocate budget; oversample rare class (phase-01) |
| YOLO26 dual-head shifts layer indices → breaks feature hook | High | Low | Re-map hook indices on load; verify pool4 shape (phase-02) |
| Relation head overfits at N=3k (64-clip test) | Med | High | Relative-geometry features + heavy reg + blind selection; report k-fold CI (phase-03/04) |
| Annotation slips past W5 | Med | High | SAM2-assisted track mode; train detector on partial 1.5k first; hard-cap ontology to 8 classes |
| Ceiling still holds (negative) | Med | Med(→OK) | Pre-framed as a stronger negative result; keep V10→3k as the scientific narrative |

## Reports feeding this plan
- [researcher-yolo26-ultralytics.md](reports/researcher-yolo26-ultralytics.md)
- [researcher-cvpr2025-spatial-relation.md](reports/researcher-cvpr2025-spatial-relation.md)
- [researcher-annotation-eval-tooling.md](reports/researcher-annotation-eval-tooling.md)

## Open questions for advisor (resolve W1)
1. WalkwayViolation definition: *unauthorized person outside walkway* vs *any person in restricted area*? (drives sampling quotas)
2. Zones at inference: training-label only, or a runtime rule engine bundled with the deployment artifact?
3. Accept the honesty upgrade: report blind + k-fold as headline (not the test-selected 56/64)?
4. Is 3,000 frames the agreed budget, or scale toward the Ch6 ≥4,000 recommendation if annotation velocity allows?
