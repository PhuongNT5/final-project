# Thesis Final Push - 5 parallel tracks

**Goal:** produce a defensible thesis result against Onal & Dandil (Unsafe-Net, 62/64 = 96.88 % top-1,
95.81 % macro on 64 unsafe test clips) - either beating it or establishing, with statistics, why the
comparison cannot be decided at N=64.

**Status entering this plan (measured, not projected):**
- Blind test 50/64 = 78.12 % top-1, 80.81 % macro. Test-selected 55/64 = 85.94 %, 85.86 % macro.
- Behaviour detector (paper's YOLO stage) NOW WORKS on cpu: mAP50 0.791 (tight boxes) / 0.806 (ctx).
  Previously 0.0144 / 0.00212 - an Ultralytics-on-MPS training bug, not a method failure.
- Box-context ablation: negative (0.791 vs 0.806) - literal paper formulation is fine.
- Val macro does not predict test macro across 34 configs: Pearson r = -0.066 (p = 0.71).
- Paper's own frame-reduction ablation (62 vs 57) is not significant: McNemar p >= 0.0625.

## Tracks

| # | Track | Owns (write) | Parallel-safe | Depends on |
|---|---|---|---|---|
| T1 | Objectives & results audit -> master plan | `plans/2026-09-05-*/plan*.md`, `phase-01*` | yes (read-only on code) | - |
| T2 | Deep research: paper + related work | `plans/2026-09-05-*/research/*` | yes (read-only + web) | - |
| T3 | Build model code | `unsafenet/*.py`, `train_*.py` | no - serialize vs T4 | T1 decision, T2 findings |
| T4 | Training & experiments | `runs/`, `sweep_results/`, `cache_r2/` | queue of 1 (hardware-bound) | T3 for new code |
| T5 | Validation, metrics, defence | `plans/2026-09-05-*/reports/*`, `docs/` | yes (protocol now, verdict later) | T4 for final numbers |

**Honest parallelism note:** T1, T2, T5-protocol are genuinely concurrent (read-only, disjoint files).
T3 and T4 are *not* concurrent with each other - one machine, CPU-bound YOLO training, and T4 must
train what T3 ships. Presenting them as 5 simultaneous workers would be a scheduling fiction; they run
as 3 concurrent lanes with a serialized build->train lane in the middle.

## Phase files
- `phase-01-objectives-and-results-audit.md` (T1)
- `phase-02-deep-research-paper-and-related-work.md` (T2)
- `phase-03-build-model-code.md` (T3)
- `phase-04-training-and-experiments.md` (T4)
- `phase-05-validation-metrics-and-defence.md` (T5)

## Critical path
T1 (2 h) -> T3 (4 h) -> T4 (6-12 h compute) -> T5 verdict (2 h). T2 and T5-protocol run alongside.

## Hard rules for this plan
1. One test touch per declared protocol. Blind number is the headline; test-selected reported beside it.
2. No claim of "beating the paper" without an interval or a test statistic attached.
3. Every failed run gets a control before it gets an explanation (learned the hard way: 9 h lost to an
   unverified MPS diagnosis).
4. Report what ran, including what failed.
