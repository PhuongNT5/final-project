# Project Roadmap

Authoritative tracker: `plans/2026-09-06-thesis-2week-final/plan.md`. This file is a snapshot for
`docs/` readers — link back to the plan for live status, do not treat this as the source of truth.

## Window

2026-09-06 -> 2026-09-19 (14 days). Owner decisions locked in
`plans/2026-09-06-thesis-2week-final/brief.md` §0: Vietnamese thesis + English abstract, full
physical Jetson access, no claim to beat the paper's headline metric, all 4 deliverables (thesis,
slides, reproducible repo, live demo), owner's evaluation contract binding.

## Lanes and phases

| Lane | Scope | Phase file | Concurrency |
|---|---|---|---|
| A — Accuracy | Detector retrain, P3 rebuild, k-fold/LCO, headline freeze, safety/calibration | `phase-01-accuracy-detector-and-pipeline.md` | Serialized — one heavy job at a time |
| F — Metrics | `eval_utils/` §4 suite as reusable code | `phase-02-evaluation-metric-suite.md` | Front-loaded, must land before Lane A's headline freeze |
| B — Edge | Jetson export/benchmark/demo | `phase-03-jetson-edge-deployment.md` | Concurrent with A (separate hardware) |
| C — Thesis | VN chapters 1-6 + English abstract | `phase-04-thesis-writing-vietnamese.md` | Concurrent; Ch4/Ch5 gated on Lane A/B freezes |
| D — Repro & docs | `docs/` (this directory), README, repro scripts | `phase-05-reproducibility-and-docs.md` (this file's own phase) | Concurrent |
| E — Defence | Slides, Q&A brief, figures | `phase-06-defence-slides-and-qa.md` | Gated on Lane A/B freezes |

## Go/no-go gates (brief §8)

| Gate | Condition | Consequence |
|---|---|---|
| Gate 1 (end day 2) | Detector retrain reaches mAP50 >= 0.90 and all 8 classes emitted | Pass -> proceed to relation-Transformer rebuild. Fail (mAP50 < 0.85 or classes still dead) -> stop the detector attack, keep the existing P3 k-fold/LCO evidence as headline, reallocate to Lanes B/C/D. Either outcome is reportable. |
| Gate 2 (end day 6) | Headline frozen, one declared test touch spent, §4 eval suite complete on the headline | After this: no further accuracy experiments — a new number would land inside the existing confidence interval and is not a result. |
| Gate 3 (end day 10) | Demo runs end-to-end unattended for 10 minutes | Pass -> live demo at defence. Fail -> fallback recording becomes the demo. |

## Critical path

Detector retrain (day 1-2) -> pipeline rebuild / k-fold / LCO (day 3-5) -> headline freeze (day 6)
-> Ch4 experiments chapter (day 7-8) -> Vietnamese assembly (day 12) -> final freeze (day 14). A
slip in the day-1/2 detector retrain is the single highest-risk item for the whole schedule.

## Day-by-day schedule (brief §7)

| Day | Date | Lane A | Lane B | Lanes C/D/E/F |
|---|---|---|---|---|
| 1 | Sep 6 | Plan lock; GPU detector training launched | Jetson env verify, JetPack/TRT inventory | F: §4 suite spec; C: Ch1/Ch2 VN outline |
| 2 | Sep 7 | Detector results: card, PR curves, 8/8 class coverage | Export path spike | F: §4 suite implemented |
| 3 | Sep 8 | Behaviour-clip + frame-gate; paper-anchor replication (0.14s/clip, 1.87s reduced duration) | ONNX -> TRT FP16 | C: Ch1, Ch2 drafted |
| 4 | Sep 9 | Relation-Transformer retrain (>=5 seeds) | Latency p50/95/99, FLOPs, size | C: Ch3 drafted |
| 5 | Sep 10 | k-fold 516 + LCO 516 | INT8 calibration + parity | D: `docs/` skeleton (this directory) |
| 6 | Sep 11 | **Headline freeze + single test touch + full §4 suite**; safety/calibration | Multi-stream + zero-drop | E: figures regenerated |
| 7 | Sep 12 | Buffer / contingency | Soak 30 min + power | C: Ch4 drafted from frozen headline |
| 8 | Sep 13 | Ablation table + error analysis | Demo pipeline wiring | C: Ch4 finalised |
| 9 | Sep 14 | — | Dashboard + live run | C: Ch5 drafted from edge results |
| 10 | Sep 15 | — | Rehearsal + fallback recording | C: Ch6 + English abstract |
| 11 | Sep 16 | — | — | D: README, repro scripts, `docs/` complete |
| 12 | Sep 17 | — | — | C: full VN assembly + consistency pass |
| 13 | Sep 18 | — | — | E: slides + Q&A brief; code review |
| 14 | Sep 19 | — | — | Final review, freeze, buffer |

## Status as of this writing (docs creation pass)

This `docs/` directory was created as part of Lane D. As of this writing:

- Lane A: detector retrain and headline-freeze status not independently re-verified by this pass
  — see `plans/2026-09-06-thesis-2week-final/reports/` for the latest lane reports before citing
  a freeze status.
- Lane F: `eval_utils/` exists and has a passing-looking test suite (`eval_utils/tests/`), consumed
  by `run_full_eval.py` and referenced as the single entry point (`evaluate_suite()`) — appears
  substantially built, per `codebase-summary.md`.
- Lane B: real Jetson artifacts exist for P1 only (see `deployment-guide.md`); P3 Jetson
  re-benchmark not yet done.
- Lane D (this directory): `docs/` populated for the first time in this pass (previously did not
  exist at all). README rewrite and `scripts/repro-*.sh` wrappers are explicitly out of this pass's
  scope (owned by whoever runs the full Lane D phase) — see phase-05's own todo list for what
  remains: README rewrite, repro scripts, environment lockfile, code review pass.

## Unresolved questions

1. Whether Gate 1 passed, failed, or is still pending is not verifiable from static file reads
   alone (it depends on a training run's live outcome) — this roadmap states the gate criteria,
   not a verified pass/fail; check `plans/2026-09-06-thesis-2week-final/reports/` for the latest
   dated report before citing a gate outcome in the thesis.
2. This docs-creation pass did not perform the phase-05 todo items "README rewritten," "scripts/
   repro-*.sh created," or "environment pinned" — those remain open per phase-05's own todo list
   and are explicitly out of this pass's file-ownership scope (`docs/` only).
