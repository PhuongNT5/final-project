# T5-A - Frozen metrics protocol

Written BEFORE T4's final round results. If a metric or rule here needs to change after seeing
numbers, that is a violation of this protocol - it must be flagged to T1, not silently patched.

Implementation: `unsafenet/metrics.py` (200 lines) + `unsafenet/metrics_stats.py` (94 lines).
No sklearn (checked: sklearn 1.8.0 and scipy 1.17.1 both installed, but metrics computed from
numpy/scipy primitives only, per instruction). Verified on 3 real `sweep_results/r2/temporal_*.pt`
members (Task 2 output below) - not a dry run.

## 1. Primary metric

**Top-1 accuracy on the 64-clip blind test set, plus macro accuracy (mean per-class recall).**
Why: this is the only pair of numbers the paper (Onal & Dandil) reports in Tables 5-7 (62/64 =
96.88% top-1, 95.81% macro). Anything else is not comparable to their headline claim. Both numbers
always reported together - top-1 alone hides the CarryOverload/UnauthorizedIntervention (n=8,
n=11) classes that macro protects.

## 2. Secondary / diagnostic metrics (all implemented, none primary)

- Balanced accuracy - numerically identical to macro accuracy for single-label multiclass; kept
  as a separate reported name only because reviewers ask for both terms.
- Per-class precision/recall/F1, macro-F1, weighted-F1, micro-F1 (micro-F1 == top-1 accuracy here,
  reported for completeness, not as new information).
- Matthews correlation coefficient (multiclass, Gorodkin 2004) and Cohen's kappa - chance-corrected
  agreement, useful because our class supports are uneven (32/11/13/8).
- Confusion matrix + row-normalised confusion matrix - diagnostic for B2 (per-class comparison).
- One-vs-rest ROC-AUC and PR-AUC, per class and macro - diagnostic for calibration/ranking quality
  independent of the argmax threshold; NOT comparable to the paper (they report only top-1/macro).
- Calibration: ECE, MCE (15 equal-width bins), multiclass Brier score, NLL - diagnostic only, used
  in Part C to argue about deployment confidence, never in the headline comparison.
- Per-class top-1 error and per-camera-view top-1 error (via `unsafenet.data_index` view map) -
  diagnostic for B3 reproducibility findings, not a claim vs the paper (paper does not report by
  view).
- Statistical layer (`metrics_stats.py`): Clopper-Pearson exact CI, Wilson interval, 5000-resample
  bootstrap CI (macro accuracy, macro-F1, any scalar metric via closure), exact McNemar, bounded
  McNemar (see SS5), permutation test (paired, 10000 resamples, works on any scalar metric_fn).

## 3. Reporting format (mandatory for every number in the thesis)

`<value> (95% CI [lo, hi], N seeds, <protocol label>)`

- CI: Clopper-Pearson for top-1 (exact, matches a proportion), bootstrap for macro/F1-family
  (no closed form). Wilson reported alongside top-1 as a cross-check, not a replacement.
- Seed count: number of independent training seeds averaged, or 1 if a single deterministic run
  (e.g. `agg*` members, which are training-free aggregations of one frame-model run).
- Protocol label: exactly one of `blind` (scored once, never used for selection) or
  `test-selected` (best of several configs by test score - reported for transparency, never as
  the headline). A number with no label attached must not appear in the thesis.

## 4. Test-touch budget and log

Hard rule from `plan.md`: one test touch per declared protocol per configuration. A "touch" = any
time test labels (`yt`) or test predictions inform a decision (including "which of N configs do we
report" - evaluating N configs on test and keeping the best is N touches, not 1).

Already spent, entering this plan (from `plan.md` status): blind pass across the round producing
50/64, plus the test-selected sweep across configs producing 55/64. Both already counted as spent
- do not re-run these as if new.

Budget for the remainder of this plan: **2 touches** -
1. one blind touch on the pre-registered headline config (SS5) once T4's final round lands,
2. one test-selected touch reporting the best of the final round's configs, disclosed as such.

No further test touches after that without a written exception logged below.

| # | Date | Config(s) touched | Protocol | Result | Logged by |
|---|---|---|---|---|---|
| 1 | (round 0-1, pre this plan) | round history | blind | 50/64 | T4 (prior) |
| 2 | (round 0-1, pre this plan) | best of sweep | test-selected | 55/64 | T4 (prior) |
| 3 | TBD | pre-registered headline (SS5) | blind | pending | T4/T5 |
| 4 | TBD | best of final sweep | test-selected | pending | T4/T5 |

T4 and T5 must both append a row here at the moment of touch, before discussing the number.

## 5. Pre-registration rule - headline config, defined now

The headline configuration is the one with the **highest validation macro accuracy** among configs
completed in this plan's final round, ties broken by earliest name lexicographically. This rule is
picked *because* T1's audit already showed val/test correlation is r = -0.066 (p=0.71) - val cannot
reliably pick the test winner, so no rule based on val selection can be defended as "finding the
best model." Declaring the val-macro-max rule now, before results, converts that weakness into a
feature: whatever it selects is the pre-registered blind number, honestly obtained, and its
possible mismatch with the test-selected number IS the r=-0.066 finding restated at N=1, not a
flaw to hide.

Concretely: the rule is executed by picking `max(val_macro)` across every `sweep_results/r2/*.pt`
member written after this protocol's freeze timestamp, using the `val_macro` field already stored
in each file. That member's `Pt`/`yt` are scored exactly once (touch #3 above) and reported as the
blind headline. No other config's test score may be called "the result" in the abstract/conclusion.

## 6. Head-to-head vs the paper without their per-clip predictions

We only have the paper's aggregate counts (62/64 top-1, per-class 32/32, 10/11, 12/13, 8/8), not
which specific clips they got right. Exact McNemar needs the per-clip discordance, which we cannot
compute. Two honest alternatives, both implemented, both to be reported together (never just one):

- **`mcnemar_bound(n_total, correct_a, correct_b)`**: enumerates every discordant split (b, c)
  consistent with the two known correct-counts and returns the smallest exact p across all of
  them - a lower bound on the true p-value (true p cannot be shown smaller than this). Already
  verified: `mcnemar_bound(64, 62, 57)` = 0.0625, matching the number already in `plan.md` for the
  paper's own full-vs-nobbox ablation. This is the number we cite for "us vs paper."
- Interval-overlap framing (T1's minimum-detectable-effect): report our CI and note whether 62/64
  falls inside it, alongside the bound above - two independent ways of saying the same thing, not
  redundant if the reader trusts one test framework over another.

Report format: "Our blind result is X/64 (CI). Paper reports 62/64. McNemar bound across all
consistent discordance splits: p >= 0.0625 [or the actual computed value once X is known] - the
comparison cannot be decided as more/less significant than the paper's own internal ablation
without per-clip data, which is not published."

## Task 2 verification (real data, not simulated)

Loaded 3 members from `sweep_results/r2/`, ran `metrics.report()` on each `Pt`/`yt` (64-clip test):

| name | n | top1 | macro_accuracy | macro_f1 | mcc | kappa | roc_auc_macro | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| detT-mean | 64 | 0.7812 | 0.8002 | 0.7887 | 0.6877 | 0.6807 | 0.9429 | 0.1516 | 0.3261 |
| aggb0-gate50 | 64 | 0.8125 | 0.8123 | 0.8059 | 0.7229 | 0.7217 | 0.9360 | 0.1067 | 0.2730 |
| tf-xlstm-lg-s8 | 64 | 0.8281 | 0.8351 | 0.8277 | 0.7471 | 0.7451 | 0.9572 | 0.1132 | 0.2652 |

(table transcribed from the actual run; `markdown_table()` output matches, see command below.)
`mcnemar_bound(64, 62, 57)` = `{'diff': 5, 'min_p_over_splits': 0.0625}` - reproduces the number
already asserted in `plan.md`, confirming the implementation. `permutation_test` and
`mcnemar_exact` between `detT-mean` and `tf-xlstm-lg-s8` both return p > 0.4 - consistent with
T1's finding that these configs are not distinguishable at N=64.

Repro: `.venv/bin/python -c "from unsafenet import metrics as M; ..."` loading any
`sweep_results/r2/temporal_*.pt` and calling `M.report(d['yt'], d['Pt'].numpy(), d['name'],
names=d['test_names'])`.

## Unresolved questions

1. T1 has not yet picked claim (a)/(b)/(c) (phase-01 R1) - SS6's reporting emphasis (bound vs
   interval) may shift depending on which claim wins; this protocol supports either.
2. Per-camera-view breakdown assumes `clip_view_map.json` covers all test clips - not verified for
   100% coverage; clips missing from the map fall into view `'na'`, diluting that bucket silently.
3. Bootstrap CI for macro accuracy/F1 can resample a fold missing a class entirely (small per-class
   n, e.g. CarryOverload n=8) - `per_class_recall` returns NaN for that resample and
   `np.nanpercentile` skips it, which biases the CI narrower than a naive implementation. Flagging
   so T1/reviewers know this is a deliberate choice, not an oversight.
4. Test-touch log rows 1-2 are back-filled from `plan.md`'s prose, not from a contemporaneous log -
   exact dates/configs for those two touches are not recoverable; going forward every touch must be
   logged at the moment it happens.
