# Phase 2 · Baseline-detector de-confound — SETUP + pre-registered rule

**Date:** 2026-09-06 · **Status:** awaiting Kaggle GPU run (user-executed) · **Purpose:** de-confound the
zones-detector rejection.

## The confound being closed
The zones-detector arm was REJECTED at LCO 79.7 vs champion 82.0 (−2.3). But the arms differ in TWO ways:
- champion detector = **YOLO26**, zones detector = **YOLO11s** (architecture)
- zones detector also emits **zone boxes** (the zone-token effect)
So the −2.3 conflates architecture with zone-tokens. The missing control is a **YOLO11s baseline (no zones)**.
Its Kaggle run crashed on the old data.yaml path bug and never produced weights (only `runs_baseline/args.yaml`).
Now fixed (notebook cell 0b builds `/kaggle/working/base`; cell-4 uses `--data_root {BASE}`).

## Kaggle run (baseline arm only) — ~3–5 h on a T4
Dataset already uploaded (the zones run used it). In the notebook, run in order:
1. **cell-1** — env + auto-discover the dataset input (`IN`).
2. **cell 0b** (`8f2416f5`) — build the corrected self-contained root (`BASE`). REQUIRED (fixes the path bug).
3. **cell-4** — baseline 2-stage train (`--data_root {BASE}`, 20 frozen + 40 finetune). NOT the zones cells.
4. **cell-13** — collect weights → downloads `detector_baseline_best.pt`.
Skip the smoke cell (cell-3) unless you want the ~2-min plumbing check first. Do NOT run the zones cells
(cell-6/7) or the behaviour-detector cells (Arm C) for this — baseline only.

## Handoff (local, one command once weights are downloaded into the repo)
```
tools/eval-detector-arm.sh <path>/detector_baseline_best.pt baseline_ cpu
```
Runs: extract tokens (YOLO appearance) → L1 strip → LCO (seeds 3) → k-fold (seeds 3). Fully local; the
behaviour-clip frames are already present (same path the zones arm used).

## Pre-registered decision rule (fixed BEFORE seeing the number — no post-hoc tuning)
Reference: champion (YOLO26) LCO 82.0 / k-fold 84.7; zones (YOLO11s+zones) LCO 79.7 / k-fold 83.6.
Let B = baseline (YOLO11s, no-zones) LCO macro.
- **B ≈ 79.7 (within ±1.0 of zones):** the −2.3 is **ARCHITECTURE** (YOLO11s < YOLO26); zone tokens are
  approximately neutral. Conclusion: zones weren't the culprit — the detector arch was. Champion (YOLO26)
  stays; no reason to revisit zones on a stronger backbone unless separately motivated.
- **B ≈ 82.0 (within ±1.0 of champion):** zone tokens **HURT** (they cost the −2.3 on the same arch).
  Conclusion: zones rejection stands on its own merits, arch-independent.
- **79.7 < B < 82.0:** **MIXED** — report both contributions proportionally; neither effect is sole cause.
Whichever bucket: this does NOT change the champion (YOLO26, 82.0). It only attributes the zones −2.3
correctly for the thesis methods/limitations section. The blind test set is NOT touched by this step.

## Expectation (hypothesis, not the rule)
Likely **B ≈ 79.7** (architecture-dominated): YOLO26 is a newer/stronger detector than YOLO11s, and the
zones arm's per-class shift (WalkwayViol +7.6 / UnauthorInt −16.7) looked like a class-prior effect that a
no-zones YOLO11s would not reproduce — pointing to arch as the main driver of the gap. To be confirmed.

## Unresolved questions
1. Does the baseline YOLO11s also fail to emit Safe_Walkway/Restricted_Zone (as the CPU detector did), or
   does higher-res GPU training recover them even without zone labels? (Side observation from the run logs.)
2. If B shows arch dominance, is a YOLO26 + zone-tokens arm worth one run to test zones on the strong
   backbone? Only if the thesis narrative needs it — otherwise YAGNI (zones already rejected on LCO logic).
