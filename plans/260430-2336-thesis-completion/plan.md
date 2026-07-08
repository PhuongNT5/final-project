---
title: "Thesis Completion — YOLO-Transformer Edge Deployment"
description: "5-week plan (defense in 5 weeks) to complete missing experiments, reconcile numerical errors, restructure thesis around Pareto headline, and produce defense materials."
status: in-progress
priority: P1
effort: 25d
branch: main
tags: [thesis, edge, yolo-transformer, unsafe-net, defense]
created: 2026-04-30
revised: 2026-04-30
---

# Thesis Completion Plan (5-Week Revision)

> **Defense in 5 weeks.** Phases 04/05 run as parallel tracks alongside the critical path (see below).
> All numbers must trace to [GROUND_TRUTH.md](GROUND_TRUTH.md) — the single source of truth.

## Headline Contribution (corrected — Chapter 5 canonical numbers)

> Annotation-efficient, edge-deployable YOLO–Transformer establishing a new
> accuracy/annotation-cost/latency Pareto point: **89.28 % macro / 87.50 % top-1
> (56/64)** at **zero behavior-bbox annotations** and **28.8 ms CPU-FP32 p50
> latency (34.7 FPS)**, vs. paper's **95.81 % macro / 96.88 % top-1 (62/64)**
> requiring 2 262 manually-labeled behavior bboxes and unreported deployment cost.
> Versus the paper's **no-bbox** baseline (ConvLSTM-only: 89.06 % top-1 / 85.78 %
> macro, [5] Tables 5 & 7): **+3.50 pp on macro**, **−1 sample (parity) on top-1**
> (z = −0.27, p = 0.79 — not significant on N = 64).

## 5-Week Schedule

| Week | Days | Phases active | Output |
|---|---|---|---|
| 1 | 5 | Phase 01 — Jetson edge measurement | Real Jetson FP16/INT8 numbers OR documented fallback |
| 2 | 5 | Phase 02 — multi-stream polish (2d) + Phase 06 — stats rigor (3d) | INT8/CoreML multi-stream + 30-min soak; ablation table + bootstrap CIs |
| 3 | 5 | Phase 03 — data-efficiency curve (3d) + Phase 07 start (2d) | 3-point label-budget curve from existing checkpoints; Ch 1 + Ch 4 outline |
| 4 | 5 | Phase 07 — thesis restructure (5d) | All chapters numerically reconciled; Ch 6 + Ch 7 drafted |
| 5 | 5 | Phase 08 — defense materials (2d) + buffer (3d) | Slides, demo script, Q&A appendix, one-pager |

## Phase Status Table

| File | Priority | Effort | Status |
|---|---|---|---|
| [phase-01-jetson-edge-measurement.md](phase-01-jetson-edge-measurement.md) | must-have | 5d | pending |
| [phase-02-multi-stream-live-system.md](phase-02-multi-stream-live-system.md) | must-have | **2d** (was 4d — N=1..4 FP32 already measured) | pending |
| [phase-03-data-efficiency-curve.md](phase-03-data-efficiency-curve.md) | must-have | **3d** (was 5d — existing checkpoints cover 401-frame point) | pending |
| [phase-04-weakly-supervised-pseudo-bbox.md](phase-04-weakly-supervised-pseudo-bbox.md) | **parallel** | 10d cap | spare cycles only |
| [phase-05-cross-dataset-generalization.md](phase-05-cross-dataset-generalization.md) | **parallel** | 4–6d post-DL | download in background NOW |
| [phase-06-statistical-rigor.md](phase-06-statistical-rigor.md) | must-have | **3d** (was 4d — run_ablation_inference.py exists) | pending |
| [phase-07-thesis-restructure.md](phase-07-thesis-restructure.md) | must-have | 6d | pending |
| [phase-08-defense-materials.md](phase-08-defense-materials.md) | must-have | **2d** | pending |

**Total must-have: ~21d across 5 weeks. Buffer: 4d.**

## Parallel-Track Phases (off critical path)

Phases 04 and 05 are reactivated as **parallel tracks** running alongside the 5-week
critical path. They are not required for defense; the student writes report and
defense materials simultaneously with these experiments. If the critical path slips,
parallel tracks pause without affecting defense readiness.

- **Phase 04** (weakly-supervised pseudo-bbox) — 10-day hard cap; Branch B negative
  result is the realistic outcome and ties to Phase 03 data-efficiency story. Lands
  in §4.7 if completed; otherwise stays as Future Work in Ch 7.
- **Phase 05** (cross-dataset generalization on **UCF-Crime**) — pre-download starts
  NOW (background, ~30 GB curated subset; disk constraint = 41 GB free). Active work
  (zero-shot + light-finetune) ≈ 4–6 person-days post-download. Lands in §4.8 if
  completed; otherwise Future Work bullet in Ch 7.
- In Phase 08 slides, §4.7/§4.8 placeholders read "Result reported if completed by
  defense; otherwise listed as Future Work."

## Numerical Errors Inherited by Prior Plan (NOW CORRECTED)

| Prior plan said | Correct value | Fix location |
|---|---|---|
| `26.5 ms / 37.7 FPS` (headline) | `28.8 ms / 34.7 FPS` (Ch 5 Table 5.2) | plan.md headline, phase-01, phase-07, phase-08 |
| `32.2 ms / 31.1 FPS` (CPU-INT8) | `32.0 ms / 31.3 FPS` (Ch 5 Table 5.2) | phase-07, GROUND_TRUTH.md |

(PROGRESS.md errors `1.39 ms` and `13.6 MB` are tracked in GROUND_TRUTH.md §7 and corrected in Phase 07 Step 0.)

## Key Numbers (locked — trace to GROUND_TRUTH.md)

- Best ensemble: **56/64 = 87.50 % top-1 / 89.28 % macro** (CPU-FP32 and CoreML-FP32)
- Paper no-bbox baseline (ConvLSTM-only): **57/64 = 89.06 % top-1 (Table 5) / 85.78 % macro (Table 7)**
- Paper full cascade (YOLO+ConvLSTM, bbox): **62/64 = 96.88 % top-1 (Table 6) / 95.81 % macro (Table 7); 2 262 bbox images**
- V10 finetune (401 frames, s2024): **84.73 %** — see `sweep_results/finetune_bbox/results.json`
- M4 CPU-FP32 e2e p50: **28.8 ms / 34.7 FPS** (Chapter 5 Table 5.2 — canonical)
- M4 CPU-INT8 e2e p50: **32.0 ms / 31.3 FPS** (INT8 slower than FP32 — legitimate finding)
- INT8 ensemble total: **~40 MB** (8.66 + 11.79 + 11.79 + 4×2.03)
- Annotation budget: **401 labeled frames** (not "~400")

## Dependency Graph

```
Critical path (5 weeks → defense):
  Ph1 ──────────────────────────────┐
  Ph2 (N=1..4 FP32 done; INT8 left)─┤
  Ph3 (401-frame point exists) ─────┤──→ Ph7 (thesis restructure) ──→ Ph8
  Ph6 (ablation_inference.py exists)┘

Parallel tracks (off critical path):
  Ph4 (10d cap, spare cycles) ──────────→ §4.7 (if done) / Future Work
  Ph5 (UCF-Crime download NOW; 4–6d) ───→ §4.8 (if done) / Future Work
```
