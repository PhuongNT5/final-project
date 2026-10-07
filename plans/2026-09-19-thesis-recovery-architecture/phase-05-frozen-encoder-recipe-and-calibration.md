# Phase 05 — Frozen-Encoder Recipe: Prior Correction, Calibration, Operating Point

## Context Links
- [plan.md](plan.md) · [phase-02](phase-02-lco-confound-decomposition-and-reporting.md) · [phase-03](phase-03-walkway-geometry-repowering.md) · [risks.md](risks.md)
- Code: `eval_utils/calibration.py`, `eval_utils/operating_point.py`, `eval_utils/cost_matrix.py`,
  `eval_utils/safety_metrics.py`, `eval_utils/selective_risk.py`, `binary_gate_lib.py`
- `reports/finding-2026-09-19-lco-failure-is-camera-label-confounding.md` §6 — architecture search cut

## Overview
- **Priority** P2 — small, near-certain gains on the *safety* framing; no architecture work.
- **Status** not started. Needs only frozen-E3 OOF dumps, so it can run in parallel with phase 03.
- **Effort** 6h · **Lane B** · **Gate G4** (operating point fixed before the freeze)
- **0 test touches.** Everything is fitted and selected on OOF/val folds only.

## Key Insights
1. **E3 DeepSets (18,044 params) is frozen.** Architecture, width, depth, loss zoo: all cut. Six
   encoders over a 16× parameter range failed identically per sub-class — further search is
   uninformative by construction, and saying so is stronger than another sweep.
2. What is *not* cut: how the frozen model's scores are turned into decisions. Threshold, temperature
   and prior correction change the safe/unsafe recall trade-off without touching the model — and the
   thesis's operating point (0.95 unsafe recall) lives exactly there.
3. Train and test priors differ (pool is unsafe-heavy; test is 64U/61S). Report at the
   **test-observed prior**; logit adjustment (Δ = log(3.96/1.0) = 1.38) belongs in an appendix as a
   sensitivity analysis, never in the headline.
4. Calibration is a *reportable deliverable* for a safety system (Brier, ECE, reliability diagram),
   independent of whether it moves accuracy. A miscalibrated safety classifier is a finding.
5. Cross-architecture ensembling is cut with the rest of architecture search: averaging models that
   fail on the same clips cannot fix those clips. Seed-averaging within E3 stays, as it is already
   part of the training recipe.

## Requirements
**Functional** — temperature scaling fitted per LCO fold on training folds only; prior correction as
a declared variant; operating point at ≥0.95 unsafe recall with the threshold chosen on OOF; cost at
10:1; selective-risk/abstention curve for the demo's abstain band.
**Non-functional** — no fitting on the 125-clip blind split; every fitted scalar recorded in the
number sheet with the folds it came from.

## Architecture
```
frozen E3 ──> per-fold OOF probs ──┬─> temperature T (fit on train folds)      ──> calibrated probs
                                   ├─> prior shift Δ = log(pi_test/pi_train)   ──> declared variant
                                   └─> threshold tau  s.t. unsafe recall >= 0.95 on OOF
                                                │
                                                v
                              operating-point table + reliability diagram + cost@10:1
                                                │
                                                v
                              DECISION_ABSTAIN_ZONE for the demo (phase 08)
```

## Related Code Files
**Create** — `tools/eval-operating-point.py` (<150 lines): fits T per fold, sweeps tau, emits the
operating-point table, reliability data, cost@10:1 and the abstention curve as markdown + JSON.
**Modify** — `reports/test-touch-log.md` (0 touches recorded).
**Delete** — none.

## Implementation Steps
1. **[M4-CPU, 60 min] Temperature scaling per LCO fold**, fitted on the fold's *training* clips,
   applied to its held-out clips; pool the calibrated OOF probabilities.
   ```bash
   cd /Users/nguyenthiphuong/Learning/Proposal/project/my-project
   .venv/bin/python tools/eval-operating-point.py --oof \
     sweep_results/relation_tokens/enc_E3_lco-691_headline-drop_conflict_oof.json \
     --calibrate temperature --per-fold --out reports/operating-point-e3
   ```
2. **[M4-CPU, 30 min] Calibration report** — Brier, ECE (15-bin, smoothed), MCE, reliability diagram
   data, before and after temperature. Ship it whether or not it helps accuracy.
3. **[M4-CPU, 45 min] Prior correction as a declared variant** — add Δ = log(π_test/π_train) to the
   unsafe logit; report **alongside** the uncorrected number, in the same table, labelled. Headline
   stays at the test-observed prior.
4. **[M4-CPU, 45 min] Operating point** — sweep τ on OOF, pick the smallest τ with unsafe recall
   ≥0.95, report the safe recall / precision / cost@10:1 it buys, with BCa CIs. The τ ships as a
   constant in the deployment config and is never re-tuned on test.
5. **[M4-CPU, 45 min] Selective risk / abstention** — risk-coverage curve; derive the demo's abstain
   band empirically instead of inheriting the literature default (0.45, 0.55), and record both.
6. **[M4-CPU, 30 min] Sub-task-aware operating points (DROPPABLE)** — the same τ sweep restricted to
   the walkway sub-task, to show whether a single global threshold is even appropriate.
7. **[Kaggle-T4, 8h, `[OVERNIGHT]`, DROPPABLE — L6] Background mixing.** Only if phase 03 finished
   early. It attacks the "walkway scene ⇒ Unsafe" shortcut directly, which is the one mechanism the
   finding identifies. Falsification: <+1.5pp panel-disclosed LCO BA with a CI excluding 0 → dead.

### Pre-registered decision rules
| Item | ADOPT if | ABANDON if |
|---|---|---|
| Temperature scaling | ECE improves on pooled OOF | ECE flat or worse → report uncalibrated, disclose |
| Prior correction | — | never adopted into the headline; appendix variant regardless of sign |
| Threshold τ | unsafe recall ≥0.95 is achievable at safe recall > 0 | if no τ reaches 0.95 unsafe recall, say so and report the achievable frontier |
| Abstain band | empirical band beats the default on risk-coverage | otherwise keep (0.45, 0.55) and label it a default, not a result |
| L6 background mixing | ≥+1.5pp panel-disclosed LCO BA, CI excludes 0 | otherwise dead; one attempt only, no hyperparameter hunt |

## Todo List
- [ ] `tools/eval-operating-point.py` written
- [ ] Per-fold temperature fitted (train folds only) and calibration report emitted
- [ ] Prior-correction variant computed and labelled as appendix-only
- [ ] τ at 0.95 unsafe recall fixed, with CI, and frozen into the deployment config
- [ ] Abstention curve produced; demo abstain band decided and recorded
- [ ] [DROPPABLE] sub-task-aware operating points
- [ ] [DROPPABLE] L6 background mixing run once against its falsification rule
- [ ] **G4 declared**; `reports/test-touch-log.md` updated (0 touches)

## Success Criteria
| Check | Pass condition |
|---|---|
| No leakage | every fitted scalar (T, τ, Δ) traceable to training folds only |
| Safety framing | operating point, cost@10:1 and abstention all reported with CIs |
| Freeze integrity | no encoder architecture is modified anywhere in this phase |
| Demo linkage | the abstain band used by phase 08 comes from this phase's curve |

## Risk Assessment
| Risk | L | Impact | Mitigation |
|---|---|---|---|
| Temperature fitted on pooled OOF (leakage) | Med | Invalidates calibration | `--per-fold` is mandatory; assert fold disjointness in the script |
| τ tuned until the number looks good | Med | **Selection leakage** | τ defined by a rule (smallest τ with unsafe recall ≥0.95), not by search |
| 0.95 unsafe recall unreachable | Low | Narrative | Pre-committed: publish the achievable frontier instead |
| Prior correction mistaken for the headline | Med | Credibility | Appendix-only rule, written into the table caption |
| L6 turns into a hyperparameter hunt | Med | Schedule | One configuration, one run, one falsification rule |

## Security Considerations
- Operates on probability dumps only; no footage decoded, no PII touched.
- τ and the abstain band ship into `deployment/` config — Lane C consumes them read-only; Lane B
  never edits deployment files.

## Next Steps
- Feeds phase 06 (the frozen suite reports the calibrated, thresholded numbers) and phase 08 (abstain
  band + τ).
- If L6 is skipped, record it as *deliberately not attempted, hours available but prior too low* —
  do not invent work to fill the unspent Kaggle budget.

**Unresolved:** whether one global threshold is defensible when sub-task difficulty differs this much
(step 6 probes it; a per-sub-task threshold would need a sub-task classifier, which is out of scope).
