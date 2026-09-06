# Phase 05 — Jetson YOLO26 Deployment + Thesis Integration

**Priority:** P1 · **Weeks:** W10–W13 · **Status:** ☐
**Context:** [researcher-yolo26-ultralytics.md](reports/researcher-yolo26-ultralytics.md), [deployment/](../../deployment/), [report/](../../report/)

## Overview
Carry the YOLO26 + relation-head pipeline to the edge (preserving the thesis's strongest contribution —
the measured Orin Nano system), then integrate all new results into the thesis and close the review gaps.

## A. Jetson YOLO26 deployment (W10–W11)
- Export YOLO26 entity detector → ONNX → **TensorRT INT8** on-device (`model.export(format="engine", quantize=8, data=..., device=0)`); requires **TRT ≥ 10.7** (10.3 has an end2end INT8 bug). 256-frame real-image calibration (reuse `deployment/artifacts/calib-jetson/`).
- Export the relation head (small) to ONNX/TRT; wire the detector→geometry→head graph end-to-end.
- **Re-run the full edge protocol** on Orin Nano Super, matching the existing artifacts so numbers are comparable:
  - Single-stream FP16/INT8 latency p50/p95/p99 + FPS (measure our own YOLO26s number — none published).
  - Multi-stream aggregate throughput + **zero-drop** verification (target: preserve current 4-stream 88 FPS).
  - Soak (≥30 min) 0-throttle; power sweep → J/inference and FPS/W.
- Compare vs the current YOLOv11 pipeline: does NMS-free/DFL-free YOLO26 improve INT8 latency/stability at equal accuracy?

## B. Metrics to report (MT3)
- Latency p50/p95/p99 (ms/frame and ms/clip), FPS, multi-stream agg + zero-drop.
- Model size / params / FLOPs; INT8 compression ratio.
- Energy: J/inference and FPS/W (power Pareto vs server baseline — extend the existing Pareto framing).

## C. Thesis integration (W11–W13)
Fold the new work into [report/](../../report/):
- **New results chapter/section:** spatial-relation ontology + YOLO26 + the ablation ladder result (confusion-pair outcome). Frame explicitly as the test of Ch6 Finding 2/3 (data-bound hypothesis).
- **Update Ch5** with YOLO26 edge numbers (fix the FP16 per-stage table + TRT-size defects from [phase-00](phase-00-preflight-and-review-fixes.md)).
- **Integrate the reframe-memo orphans** the review flagged as missing: k-fold section, WS4 spatial negative result, Transformer-vs-linear-probe finding.
- **Reconcile the two honest protocols** (blind vs k-fold) in one authoritative methodology note; remove the ~12× repeated disclaimer down to one prominent statement.
- **Fix internal contradictions:** Ch2 line ~164 stale "Jetson projected/deferred"; Ch1 "temporal-reasoning head" terminology.

## D. Finish the thesis skeleton (review found these missing)
- Write the **abstract** (from the deployment-led one-paragraph framing).
- Merge the **unified bibliography** (per-chapter numbering → single list).
- Produce **Figure 3.2** (multi-stream gating) and any dangling figure refs; add the new relation-head architecture figure.
- Optional: create `docs/` per CLAUDE.md structure; remove dangling `GAP_ANALYSIS.md` / `BASELINE.md` refs or write them.

## Todo
- [ ] YOLO26 detector + head → TRT INT8 on Orin Nano (TRT ≥ 10.7)
- [ ] Re-run latency / multi-stream / soak / power protocol; commit artifacts alongside existing ones
- [ ] YOLO26-vs-YOLOv11 edge comparison table
- [ ] Write new results section (relation ontology + ablation outcome)
- [ ] Update Ch5 edge numbers; fix table + TRT-size defects
- [ ] Integrate k-fold / WS4 / linear-probe orphans; reconcile protocols; de-duplicate disclaimers
- [ ] Abstract + unified bibliography + Figure 3.2 + relation-head figure
- [ ] Final consistency pass (`git grep` for withdrawn figures / stale numbers)

## Success criteria
- YOLO26 pipeline runs on Orin Nano with measured INT8 latency, multi-stream zero-drop preserved, power reported.
- Thesis has: new results integrated, abstract, unified bibliography, all figures resolved, no internal contradictions, single honest accuracy headline with CIs.
- A clean-clone reproduction path (scripts + artifacts) for every headline number.

## Deliverables
- YOLO26 edge artifacts + comparison; updated thesis chapters; abstract; unified bibliography; figures; final review-fix pass.
