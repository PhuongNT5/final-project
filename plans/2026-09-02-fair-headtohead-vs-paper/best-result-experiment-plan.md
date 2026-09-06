# Best-result experimental campaign — Unsafe-Net, YOLO11/26 + Transformer

**Date:** 2026-09-02. Additive (no thesis files edited). Synthesis of 5 parallel research reports
(motion, relation-transformer small-N, zone-geometry, eval-rigor, codebase-map) + direct code
verification. Goal: the BEST honest result on Unsafe-Net, prioritized by lift/risk, reusing existing
code. Source reports in `reports/researcher-*.md`.

## Convergent finding across ALL researchers: the enemy is camera/scene-fingerprint leakage
Every lever is judged by its LEAVE-CLUSTER-OUT (scene-robust) delta, not same-dist. Honest ceiling
~82-84% LCO (from 78.6%). **Any run reporting >90% LCO is a leak, not a win.** Guard every new feature
with a per-view correlation check (features that re-encode camera identity — appearance cosine, track
persistence, absolute coords — inflate same-dist and rot LCO).

## Headline protocol (eval-rigor rec): LCO (grouped 8-fold) primary · k-fold secondary · 64-clip footnote
Report: BCa CI + Clopper-Pearson per-group + Nadeau-Bengio vs scene-only & paper. Full metric battery
via `run_full_eval.py` (top1/macro/F1/κ/MCC/ROC-PR-AUC/ECE/McNemar). Test touched once for final.

---

## Phase 0 — Integrity fixes & diagnostics (BLOCKING, cheap, do first)

- **D1 [CRITICAL] Fix the LCO grouping.** VERIFIED: `views` = `KMeans(8)` on the SAME 8192-d
  appearance features used for classification (`tools/apply_zone_prior.py:62`). No real camera IDs in
  `data/`. → (a) try to recover real camera/session from clip filenames; (b) else re-cluster on a
  CLASSIFIER-INDEPENDENT scene descriptor (per-clip temporal-median background — `unsafenet/geometry.py`
  already computes these + view references). Re-run `run_lco_eval.py` under the repaired grouping; this
  becomes the true headline. Removes the circularity in the scene-only baseline.
- **D2 Coverage/duration diagnostics (near-free).** (i) Are `Safe_Walkway`/`Restricted_Zone` (YOLO
  class 2/3) already present in `tokens_*` but geometry-zeroed? (ii) Duration hypothesis: is UnauthorInt
  higher-dwell than WalkwayViol? (decides whether motion P2 will pay off). (iii) Per-view detector
  recall (does worse recall on some clusters re-encode identity?).
- **D3 Fix 2 test-fit leakage bugs** before any calibration/operating-point number is used as evidence:
  `run_ablation_calibration.py` (T) and `run_operating_point.py` (τ) fit their knob on TEST — change to
  val-fit / test-apply. (Same class of bug as the withdrawn "+3.5pp".)

## Phase 1 — Feature levers (each: re-extract tokens → LCO+kfold → keep only if LCO↑ & leak-guard passes)

| ID | Lever | Reuse | Exp. LCO lift | Leak risk |
|---|---|---|---|---|
| **L1** | **Strip absolute `xc,yc`** (keep w,h + relative/zone geom only) — camera fingerprint | edit `build_relation_tokens.py` | +1-3pp / decouples same-dist inflation | low — ship first |
| **L2** | **Homography-registered zones** (signed-dist + green-mask) into `geometry()` — fixes AUC 0.629→~0.73-0.83 | wire `unsafenet/geometry.py::clip_polygons` | **big** (targets the confused pair) | low-med |
| **L3** | **Motion via track-linking** — per-person bbox-height-normalized speed + dwell indicator (n_num +2), and clip-level motion vector via `clip_proj` fusion | port `unsafenet/entities.py::link_tracks` + `streams.py::_motion_vec` | **+2-4pp** (the physical discriminator) | med — per-view guard EACH feature |
| **L4** | **Straightness ratio** net_disp/path_length (camera-scale-invariant) | derived once L3 deltas exist | +0-1pp | low |
| **L5** | **Combine relational-proximity (v2) WITH appearance** (never combined yet) | `augment_relation_tokens.py` on app tokens | +0-1pp | low |

Normalization rule (motion): normalize displacement by bbox height (pinhole trick), NOT raw pixels.
Skip: optical flow (no gain over track-deltas given mAP .969), trajectory curvature (wrong cue).

## Phase 2 — Model / regularization levers (no re-extract; train-side)

| ID | Lever | Reuse | Exp. | Note |
|---|---|---|---|---|
| **M1** | **Domain-adversarial (GRL) head** off CLS → predict view, gradient-reversed → camera-invariant CLS | add head to `model_relation_head.py` | attacks LCO gap directly | needs repaired D1 groups |
| **M2** | **Cross-camera token MixUp** — same-label, different-cluster pairs ONLY | train loop | small, robustness | generic MixUp = do NOT |
| **M3** | **Logit adjustment** (closed-form) for imbalance + label-smoothing 0.1 + token-dropout | `train_one` | stabilizes n=8/11 classes | prefer over tunable class-weights |

Do NOT: scale up transformer (298k is right-sized), fine-tune appearance backbone, stochastic depth,
supervised-contrastive pretext (camera-shortcut risk), Caruana greedy (overfits tiny val).

## Phase 3 — Squeeze & final blind eval

- Safe squeeze (val-selected only): seed-ensemble (+1-3pp), mild train-aug targeting confound, TTA
  (+0-1pp realistic), temperature scaling (ECE only, 0pp top-1).
- Final: `run_full_eval.py` battery on the winning config; LCO headline + k-fold + 64-clip footnote;
  ablation table = each lever's marginal LCO contribution; report ties/losses honestly.

## Execution loop (per lever)
1. change (token re-extract OR model edit) → 2. `run_lco_eval.py --tag <new>` (primary) +
`run_kfold_relation.py --tag <new>` → 3. per-view leak-guard correlation check → 4. keep iff LCO↑ and
no leak; log marginal delta → 5. next lever. Cumulative winners compound into the final token+model.

## Expected honest outcome
LCO 78.6% → ~82-84% if D1(repaired)+L1+L2+L3(+M1) land; same-dist/k-fold roughly flat (already
scene-saturated). The deliverable = a defensible best-config with an ablation attributing each pp to a
mechanism, under a repaired scene-robust protocol — not a bigger noise-band 64-clip number.

## Unresolved questions
1. Can real camera IDs be recovered from filenames? If not, is temporal-median-background clustering an
   acceptable scene grouping for the thesis headline? (Blocks D1 → blocks the whole headline.)
2. Is the WalkwayViol↔UnauthorInt confusion duration-based (motion pays) or pure boundary-geometry
   ambiguity (geometry pays)? D2 decides L2-vs-L3 ordering.
3. Are 8 clusters enough domains for stable GRL/DANN (M1)? If unstable, drop M1.
4. Does `--appearance` currently help or HURT LCO? (Never isolated; a quick ablation before building on it.)
