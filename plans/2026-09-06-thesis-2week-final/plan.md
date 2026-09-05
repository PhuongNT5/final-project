---
title: "Thesis 2-Week Final Push — Execution Plan"
description: "14-day plan to close MT1-MT3, freeze headline evidence, redeploy on Jetson, and assemble VN thesis"
status: pending
priority: P1
effort: 14d
branch: next-round
tags: [thesis, yolo-transformer, unsafe-net, jetson, vietnamese]
created: 2026-09-06
---

# Plan — Thesis 2-Week Final Push (2026-09-06 -> 2026-09-19)

Expands `brief.md` (binding, do not re-litigate) into 6 executable phases. Owner locked D1-D5
(brief §0): VN thesis, real Jetson, no beat-62/64 claim, all 4 deliverables, owner's §4 eval contract.

## Claim structure (brief §2, binding)
On the paper's own 64-clip metric we are **statistically indistinguishable** (54/64=84.4%, macro
88.4% vs paper no-bbox 57/64/85.78%, z=-0.78, p=0.435) — and by construction (paper wrong only 2/64)
**no system can beat it significantly at N=64** (max McNemar p=0.50). On every axis the paper never
measured — reliability (k-fold 516, ±3pp CI), cross-camera generalisation (LCO 78.6% macro vs
scene-only 16.9%), safety/calibration (FNR, ECE), and measured Jetson edge performance — this work
provides evidence. Fixing the detector (A1, mAP50 0.79->0.90 target) makes the reproduction claim
(axis 2) credible; it is not a bid to beat the paper.

## Lanes (brief §5)
| Lane | Scope | Owns (exclusive) | Phase |
|---|---|---|---|
| A Accuracy | A1-A5 | `sweep_results/`, `runs/`, `cache_r2/`, `train_behaviour_detector.py`, `run_kfold_relation.py`, `run_lco_eval.py`, `kaggle-detector-retrain/` | [phase-01](phase-01-accuracy-detector-and-pipeline.md) |
| F Metrics | §4 suite as code | `eval_utils/` | [phase-02](phase-02-evaluation-metric-suite.md) |
| B Edge | Jetson export/bench/demo | `deployment/`, `tools/jetson_benchmark.py` | [phase-03](phase-03-jetson-edge-deployment.md) |
| C Thesis | Ch1-6 VN + abstract | `report/` | [phase-04](phase-04-thesis-writing-vietnamese.md) |
| D Repro/docs | docs, README, scripts | `docs/`, `README.md`, `scripts/` | [phase-05](phase-05-reproducibility-and-docs.md) |
| E Defence | slides, Q&A, figures | `plans/2026-09-06-*/slides/`, `figures/` | [phase-06](phase-06-defence-slides-and-qa.md) |

## Phase list & status
| Phase | Lane | Days | Status |
|---|---|---|---|
| 01 accuracy detector & pipeline | A | 1-8 | pending |
| 02 evaluation metric suite | F | 1-2 (front-loaded) | pending |
| 03 Jetson edge deployment | B | 1-10 | pending |
| 04 thesis writing (VN) | C | 1-13 | pending |
| 05 reproducibility & docs | D | 5-13 | pending |
| 06 defence slides & Q&A | E | 6-13 | pending |

## Go/no-go gates (brief §8)
- **Gate 1 (end day 2):** A1 mAP50 >= 0.90 AND all 8 classes emitted -> proceed A2/A3. Else (mAP50 <
  0.85 or classes still dead): stop detector attack, keep existing P3 k-fold/LCO as headline,
  reallocate days 3-6 to B/C/D. **Either outcome is reportable** (phase-01).
- **Gate 2 (end day 6):** headline frozen, test touch spent, §4 suite complete on headline. No further
  accuracy experiments after (phase-01, phase-02).
- **Gate 3 (end day 10):** demo runs unattended 10 min, else fallback recording becomes the demo
  (phase-03).

## Critical path
A1 (day1-2) -> A2/A3 (day3-5) -> A4 freeze+touch (day6) -> Ch4 (day7-8) -> VN assembly (day12) ->
freeze (day14). Only a day-1/2 slip in A1 threatens the thesis (brief §7). Phase-02 (eval_utils
consolidation) must land before A4's day-6 touch — it is a hard blocker on Gate 2, not just Lane F
housekeeping, since A4 reports off `eval_utils/`.

## Key dependencies found in repo (see phase files for detail)
1. `eval_utils/` and `unsafenet/metrics.py`+`metrics_stats.py` are TWO overlapping, undocumented
   implementations of most of §4 (DRY violation) — phase-02 must consolidate into one, not add a third.
2. Jetson artifacts in `deployment/artifacts/benchmarks-jetson-{fp16,int8}.json`,
   `soak-jetson-summary.json`, `multi-stream-jetson-fp16-realtime.json`, `power-sweep-summary.json`
   are REAL measurements (2026-06-03, tegra aarch64 host) on the OLD P1 pipeline (55/64 heads), not
   the P3 headline pipeline A3 will produce — `deployment/JETSON-DEPLOYMENT.md` header ("no physical
   Jetson available") is stale and must be corrected (phase-03, phase-05).
3. `docs/` directory does not exist yet — phase-05 creates all 7 files from scratch, not an update.
4. `deployment/artifacts/projections.md` mixes literature-cited Jetson numbers with measured M4 numbers
   (flagged in T1 audit) — must be corrected or dropped per brief hard-rule 5 (phase-03/phase-04).

## Unresolved (see also each phase file)
- Whether k-fold/LCO (fair-headtohead, pipeline P3) becomes the single unified headline or two
  side-by-side pipelines is a T5/committee-framing call (brief doesn't resolve; phase-01 flags it).
