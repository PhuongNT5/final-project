# Phase 03 — Box-Token Spatial-Relation Transformer

**Priority:** P0 (the scientific core) · **Weeks:** W6–W9 · **Status:** ☐
**Context:** [researcher-cvpr2025-spatial-relation.md](reports/researcher-cvpr2025-spatial-relation.md)

## Overview
Replace "YOLO names the behaviour" with "YOLO names entities → Transformer infers behaviour from
geometry." The head consumes per-box tokens (appearance + coordinates + entity class + relative
geometry + track-id) and lets self-attention learn *who is where relative to what* — directly
targeting the WalkwayViol ↔ UnauthorInt confusion (person-in-zone via attention).

## Design (research-backed, tuned for small N)
Lineage: DAB-DETR (box-as-token) → Relation DETR / ECCV'24 Oral (pairwise IoU as attention bias) →
Action Genome (action = spatio-temporal scene graph). **Relative geometry beats absolute coords at
N<5k** — do not feed raw pixel coordinates as the primary signal.

```
Input: YOLO26 detections, T=16 frames, N_max=8 boxes/frame
Per-box token (paper-faithful, per paper-faithful-pipeline-design.md):
  roi_feature(YOLO box)                   # ROI appearance feature — the user's coordinate-embedding spec
  || sin_embed(x,y,w,h)                    # DAB-DETR style positional, d≈64
  || class_embed(entity_label)            # 8 entity classes, d≈32
  || [IoU_walkway, IoU_restricted,        # relative geometry — the task signal
      foot_in_walkway(bin+cont),
      norm_dist_to_zone_centroid]         # ~4–5 scalars
  || track_embed(track_id % 16)           # temporal identity
  → MLP → d (128–256)
Sequence: [CLS] + (T × N_max) box tokens → learned temporal + slot pos-enc
  → Transformer Encoder: 2 layers, 4 heads, d=128, FFN=256
  → CLS → Linear(4)  (4-class unsafe; 8-class variant as ablation)
Regularization: drop_path=0.1, dropout=0.3, label_smoothing=0.1, weight_decay=0.01
Optional (ablation): pairwise-relation attention bias (Relation DETR) — person×zone IoU as additive bias.
```

## Why this should break the confusion pair
The two classes are cosine-0.834 in appearance but **geometrically distinct**: WalkwayViolation =
person foot-point *outside* Safe_Walkway; UnauthorizedIntervention = `Person_Unauthorized` *inside*
Restricted_Zone near a panel/machine. `foot_in_walkway` and `IoU(person, Restricted)` separate them
with almost no learning — the model just has to route on them.

## Ablation ladder (all under blind protocol, [phase-00](phase-00-preflight-and-review-fixes.md) wrapper)
1. **Appearance-only** (current head on YOLO26 features) — the control.
2. **+ box coords** (sin_embed only) — does absolute geometry help?
3. **+ entity class embedding** — does knowing Authorized vs Unauthorized help?
4. **+ relative geometry scalars** — the hypothesized main lever.
5. **+ pairwise relation-attention bias** — is the fancier mechanism worth it at this N?
6. **+ track-id / temporal consistency** — does identity across frames help?
Report each rung's blind top-1, macro, and **confusion-pair error count** with BCa CIs. The rung where
the confusion pair collapses is the thesis result.

## Selection discipline (non-negotiable)
- Epoch, seed, config, and any threshold selected on a **train-carved validation slice** only
  ([run_blind_eval.py](../../run_blind_eval.py)). 64-clip test touched once per final config.
- Report **both** blind single-split and stratified/scene-group k-fold ([run_kfold_eval.py](../../run_kfold_eval.py)).
- Seed sweep 7–10; fix MPS seed ([phase-00](phase-00-preflight-and-review-fixes.md) fix). Ensemble macro reported honestly (not `max` over members).

## Overfitting guardrails (N=3k frames, 64-clip test)
Relative-geometry features (few, meaningful) over absolute coords; shallow encoder (2 layers); strong
dropout/drop_path/weight-decay; label smoothing; augment token sequences (temporal offset, box jitter).
If lift appears only on the single split but not in k-fold CI → treat as noise, not a result.

## Related files
- Create: `model_relation_head.py` (<200 lines), `run_relation_transformer.py` (train+blind eval, reuses [run_blind_eval.py](../../run_blind_eval.py) helpers), consumes `build_relation_tokens.py` cache from [phase-02](phase-02-yolo26-and-detector.md).
- Reuse: [run_kfold_eval.py](../../run_kfold_eval.py), `eval_utils/` (phase-00).

## Todo
- [ ] Implement per-box token builder (sin coords + class + geometry + track embeds)
- [ ] Implement `model_relation_head.py` (encoder + CLS classifier)
- [ ] Wire blind-protocol training loop; seed sweep
- [ ] Run ablation ladder rungs 1–6; log confusion-pair counts
- [ ] Optional Relation-DETR pairwise bias variant
- [ ] k-fold on the best rung; BCa CIs; McNemar vs appearance-only control

## Success criteria
- **Primary:** a rung with blind top-1 > 53/64 AND WalkwayViol↔UnauthorInt errors ≥ halved vs control, with a CI that clears the control — OR a clean negative (no rung beats control under k-fold), reported as the stronger-negative result.
- Ablation isolates *which* signal (geometry vs class vs relation-attention) does the work — the interpretable finding.

## Deliverables
- `model_relation_head.py`, `run_relation_transformer.py`, ablation results table (blind+k-fold+CIs), confusion-pair delta analysis, best checkpoint.
