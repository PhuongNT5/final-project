# Phase 0 — COMPLETE (integrity fixes & diagnostics)

**Date:** 2026-09-03. Additive (only my own experiment scripts edited; no thesis files touched).
All three blocking items done & verified. Details in `phase0-D1/D2/D3-*.md`.

## Outcomes

| item | status | key result |
|---|---|---|
| **D1** [CRITICAL] repair LCO grouping | ✅ | LCO headline **survives** classifier-independent grouping: macro **80.1%** (was 78.6% circular), scene-only 10.1%, **+62.0pp**. Claim is now defensible. |
| **D2** coverage/duration diagnostics | ✅ | Zones **never detected** (static prior only) → L2 has headroom. Motion hypothesis **rejected** → confusion is geometric → **reorder L2 before L3**. |
| **D3** fix 2 test-fit leakage bugs | ✅ | Calibration: honest **ECE 0.114** (cited 0.101 was test-fit). Operating point: honest **τ=0.50 → 100% coverage @ 81.2%** (OOF-selected). |

## Corrected / new baseline numbers (carry forward)
- **Scene-robust headline (LCO, repaired):** macro **80.1%**, top-1 78.5%, +62.0pp over scene-only.
  ← the number Phase-1 levers must beat.
- **k-fold (reproduced):** top-1 80.8±3.1%, macro **84.0%**, κ 0.731, MCC 0.735, NB p=0.134 (n.s.).
- **Calibration (honest):** ECE **0.114** uncalibrated; temperature scaling does NOT help.
- **Operating point (honest):** τ=0.50, TEST coverage 100%, acc-on-fired 81.2%, ROC-AUC 0.950.

## Artifacts produced
- `tools/build_scene_grouping.py` → `sweep_results/zone_prior/clip_scene_map.json` (516-clip background grouping)
- `run_lco_eval.py --scene_map` (D1 override); `run_kfold_relation.py` now dumps
  `kfold_oof_softmax_<tag>.pt` (leakage-free selection set)
- `run_ablation_calibration.py` (train-fit T / test-apply); `run_operating_point.py`
  (`--select_logits`, off-test τ)
- `sweep_results/relation_tokens/kfold_oof_softmax_yolo26app_.pt`; refreshed
  `sweep_results/ablations/calibration.json`

## Revised Phase-1 order (per D2)
1. **L1 — strip absolute `xc,yc`** (camera fingerprint; ship first, low risk).
2. **L2 — homography-registered zones** (signed-distance to walkway/restricted polygon + continuous
   foot-distance). **Promoted to top lever** — D2 shows zones are ungrounded and the WalkwayViol↔
   UnauthorInt confusion is geometric.
3. **L3 — motion/dwell — DEPRIORITIZED** (D2 rejected the duration hypothesis); revisit only if L2
   leaves residual WalkwayViol↔UnauthorInt confusion.
4. L4 straightness, L5 relational-proximity×appearance — minor, later.

Each lever: re-extract tokens → `run_lco_eval.py --scene_map …` (primary, repaired grouping) +
`run_kfold_relation.py` → per-view leak-guard → keep iff LCO↑ and no leak. Test touched once at the end.

## Unresolved questions
1. Run YOLO11 LCO/k-fold under the repaired grouping too, for a matched detector-ablation row? (cheap, pending)
2. Report the operating point at all given the test is all-unsafe (no false-alarm measurable)? (deployment value limited)
3. Proceed to Phase 1 L1→L2 now? (awaiting user go-ahead per campaign protocol)
