# Phase 2 · Zones-detector retrain — REJECTED (LCO regressed)

**Date:** 2026-09-05 · **Verdict:** REJECT (LCO macro −2.3 pp vs champion) · **Arm:** zones (experimental)

## What was tested
Retrained the entity detector on Kaggle GPU (YOLO11s, 2-stage COCO transfer, imgsz 960, stage1 20ep
frozen + stage2 40ep finetune, vest-colour-safe aug hsv_h=0). **Zone arm**: zone polygons →
axis-aligned YOLO boxes (classes 2 Safe_Walkway, 3 Restricted_Zone) appended to labels via
`build-zone-labels.py`, so the detector now EMITS zone boxes (the old CPU detector never did).
Weights: `sweep_results/retrain-detector/runs_zones/stage2_finetune/weights/best (1).pt`.

Handoff (repo-side): `build_relation_tokens.py --detector <zones.pt> --appearance --appearance_backbone
yolo --out_tag zones_app_` → `tools/build_l1_tokens.py --src zones_app_ --dst zones_l1_` →
`run_lco_eval.py` + `run_kfold_relation.py`. Champion `yolo26L1_` re-run under IDENTICAL protocol
(3 seeds, repaired scene-map, 60ep) for a fair side-by-side — not just the cached 82.0%.

Detector emits far more tokens: **44219+6017** detections (train+test) vs old **25891+3267** — zone
boxes now grounded on every frame.

## Result (identical protocol)
| Metric | Zones (zones_l1_) | Champion (yolo26L1_) | Δ |
|---|---|---|---|
| **LCO macro** (headline) | **79.7%** | **82.0%** | **−2.3 pp** |
| LCO top-1 | 77.5% | 77.9% | −0.4 |
| k-fold macro (secondary) | 83.6% | 84.7% | −1.1 |
| k-fold top-1 | 80.0±2.3% | 80.6±2.8% | −0.6 |

Down on BOTH → not the scene-leak signature (kfold↑/LCO↓); detector just weaker. Per campaign rule
(keep iff LCO↑ and no leak) → **rejected**.

## Per-class LCO recall — the one real effect
| | Zones | Champion |
|---|---|---|
| WalkwayViol | **75.7** (+7.6) | 68.1 |
| UnauthorInt | **53.7** (−16.7) | 70.4 |
| OpenedPanel | 89.4 | 89.4 |
| CarryOverload | 100.0 | 100.0 |

Zone tokens SHIFT the WalkwayViol↔UnauthorInt tradeoff toward WalkwayViol (LCO confusion: 39
UnauthorInt→WalkwayViol vs champion 22) — helps walkway recall, collapses intrusion recall, net macro ↓.
Mechanism: a grounded walkway box makes "person overlaps walkway" fire more, over-predicting WalkwayViol.

## Critical confound (why this is not the final word)
Champion base = **YOLO26** detector; retrain = **YOLO11s**. The −2.3 pp conflates architecture
(YOLO11s < YOLO26) with the zone-token effect. The clean control is the **baseline arm** (YOLO11s,
no zones), which **never produced weights** — its Kaggle run crashed on the data.yaml path bug before
the `{BASE}` fix; only the crashed `args.yaml` remains under `runs_baseline/`.

## Artifacts (all additive — no narrative files touched)
- Full eval log: `sweep_results/relation_tokens/_zones_eval.log` (KEPT — provenance)
- Extraction log: `sweep_results/relation_tokens/_build_zones.log` (KEPT — provenance)
- **PURGED 2026-09-05 (workspace cleanup, result preserved above):** zones detector weights
  (`sweep_results/retrain-detector/runs_zones/…/best (1).pt`), tokens `tokens_zones_app_*`,
  `tokens_zones_l1_*`, OOF `kfold_oof_softmax_zones_l1_.pt`. Regenerate by re-running the Kaggle
  zones arm + the handoff commands documented above.

## Unresolved questions
1. Does a YOLO11s **baseline** (no zones) beat, match, or trail YOLO26? Needed to de-confound the −2.3.
   Requires re-running the baseline arm on Kaggle (path bug now fixed via notebook cell 0b / `{BASE}`).
2. Could zone tokens help if fed to the relation head as a SEPARATE geometry channel rather than as
   competing detection tokens (which bias class priors)? The +7.6 WalkwayViol recall hints zone signal
   is real but mis-integrated.
3. Zone AABBs are loose (diagonal walkway → near-full-frame box) — would oriented/polygon IoU features
   recover the lost UnauthorInt recall without the WalkwayViol over-fire?
