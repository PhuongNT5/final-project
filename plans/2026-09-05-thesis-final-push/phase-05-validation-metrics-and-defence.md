# T5 - Validation, metrics & defence

**Priority:** P0 for protocol (now), P1 for verdict (after T4). **Owns:** `reports/*`, `docs/*`.

## Overview
Two halves. The protocol half runs NOW, in parallel, and must be frozen before T4 finishes - a metric
defined after seeing the result is not a metric. The defence half assembles the argument.

## Part A - Metrics protocol (do first, independent of results)
- A1 Primary: top-1 on 64 unsafe test clips + macro (mean per-class recall) - matches paper Tables 5-7.
- A2 Every number carries: Clopper-Pearson exact CI (top-1), bootstrap CI 5000 resamples (macro),
  seed count, and which protocol produced it (blind vs test-selected).
- A3 Head-to-head vs paper: McNemar. We lack the paper's per-clip predictions, so report the bound
  across all discordance splits, not a single p. Already computed: 62 vs 57 gives p >= 0.0625.
- A4 Declare test-touch budget and log every touch.
- A5 Pre-register: which configuration is the headline, chosen by which rule, before T4 item 2 runs.

## Part B - Result verdict (after T4)
- B1 Ablation table: entity-only / +behaviour-detector / +reduction / leaky-vs-leak-free decomposition.
- B2 Per-class recall vs paper's per-class (32/32, 10/11, 12/13, 8/8).
- B3 Reproducibility findings section: the MPS training bug (0.0002 vs 0.940 control), the box-context
  negative result, the val/test independence (r = -0.066).
- B4 Figures: val-vs-test scatter over 34 configs; CI overlap plot vs paper; detector PR curves.

## Part C - Defence preparation
Anticipated questions, each needing an evidence-backed answer:
- "You did not beat the paper." -> the interval argument + minimum detectable effect at N=64 (T1 step 3).
- "Did you reproduce their method?" -> ConvLSTM baseline result (T4 item 3), detector at mAP 0.79-0.81.
- "Their YOLO got 97 mAP, yours 79 - is that not the gap?" -> answer with the ablation, not a guess.
- "Is their evaluation fair?" -> the decomposition leak analysis (T3 step 3, T2 R1). Must be phrased as
  a documented reading of their text, with line cites - not an accusation.
- "Why is your blind number lower than your test-selected number?" -> r = -0.066; explain honestly.
- "What is the contribution then?" -> whichever claim T1 selects, plus the deployment measurements.

## Todo
- [x] Part A frozen and written BEFORE T4 item 2 completes (`reports/t5a-metrics-protocol.md`)
- [x] Pre-registered headline config (SS5 of protocol: max val_macro, ties by name, defined pre-results)
- [ ] B1-B4 after T4
- [ ] Part C brief with evidence pointers
- [ ] `docs/` populated (currently empty): system-architecture, codebase-summary, project-roadmap

## Success criteria
Every defence question maps to a file and a number. No question answered with an opinion.

## Risks
- Protocol written after results = invalid. Mitigation: Part A is gated to finish before T4 item 2.

## Unresolved questions
1. Which claim does T1 pick - beat (a), negative-result (b), or deployment-axis (c)?
2. Do we have the compute budget for the ConvLSTM baseline, or is the paper-baseline anchor dropped?
3. Is the decomposition leak a defensible published claim, or presented only as "our reading"?
