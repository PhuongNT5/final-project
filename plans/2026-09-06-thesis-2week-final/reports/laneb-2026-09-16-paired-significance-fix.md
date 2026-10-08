# Paired-significance fix — McNemar exact/bound replaces two_proportion_z on same-clip comparisons

Date: 2026-09-16. Scope per assignment: `run_full_eval.py`, `ensemble_relation.py`,
`eval_utils/paired_tests.py` (shared statistics helper), statistics-only call sites in
`run_relation_transformer.py` (lines 220, 236 — nothing else in that file touched), new tests
under `tests/`. No commit made (per instructions).

Source of the defect: `plans/2026-09-06-thesis-2week-final/reports/review-2026-09-13-eval-statistics.md`
finding #1 — `two_proportion_z` was used as the "vs paper" stat on 3 live scripts even though our
result and the paper's are scored on the SAME 64 clips (paired data), not 2 independent n=64
samples. Confirmed by reading the code before editing — the review's line numbers matched exactly.

## Existing statistics infra found (used, not reimplemented)

`eval_utils/paired_tests.py` already had `mcnemar_exact` (exact, needs per-clip predictions),
`mcnemar_discordance_bound` (bound, aggregate-counts-only, wraps `unsafenet.metrics_stats.mcnemar_bound`),
`two_proportion_z`, `wilson_ci`, `nadeau_bengio` — and `eval_utils/suite.py::_compare` already
computed both bound and z side by side. The gap (per the review) was that the 3 number-producing
scripts never called any of this — they hand-rolled `two_proportion_z` as the sole "vs paper" line.
Fix: wire the scripts to the existing module instead of duplicating comparison logic (DRY), adding
the two pieces the review flagged as missing (the max-p bound and a paired CI) to the shared module.

## What changed in `eval_utils/paired_tests.py` (the ONE shared place)

1. `mcnemar_exact` — unchanged math (still exact binomial CDF on discordant pairs, never
   chi-square), now also returns `b`/`c` (task-standard naming; `b01`/`b10` kept for backward
   compat with 3 existing eval_utils tests) and a **paired bootstrap CI** for the accuracy
   difference (`diff_accuracy_pp`, `diff_accuracy_ci95_pp`).
2. New `paired_diff_bootstrap_ci(correct_a, correct_b)` — percentile bootstrap over clips (same
   resampled index applied to both arrays each draw, so pairing is preserved), 5000 resamples.
   Chosen over the closed-form Wald SE on `(b-c)/n` because b+c is typically ~6-8 of N=64 here —
   too small for the Wald normal approximation to be trustworthy; the bootstrap needs no
   distributional assumption and costs nothing extra at this N.
3. `mcnemar_discordance_bound` — unchanged `min_p_over_splits` computation (still calls
   `unsafenet.metrics_stats.mcnemar_bound`, so the existing audit-table test keeps passing
   byte-for-byte); added a new companion `_max_p_over_discordant_splits` giving
   `max_p_over_splits`, so the bound is now the full `[min_p, max_p]` interval the task asked for
   ("the most and least significant p-value consistent with the observed marginal accuracies"),
   not just the one-sided lower bound it returned before.
4. New `paired_vs_aggregate_baseline(n, our_correct, other_correct, label)` — the single entry
   point all 3 scripts now call for "vs paper" comparisons. Paper per-clip predictions are
   unpublished (only 57/64, 62/64 are known), so the paired table can't be reconstructed and
   `mcnemar_exact` can't run — this returns the McNemar bound as `primary_test`, clearly labelled,
   plus `two_proportion_z` renamed `diagnostic_two_proportion_z` with an explicit
   `diagnostic_warning` string calling out the invalid independence assumption. No silent
   substitution: the bound is structurally always used when only aggregate counts are available.

`eval_utils/paired_tests.py` is 195 lines (under the 200-line cap).

## Call sites changed

| File | Line(s) | Before | After |
|---|---|---|---|
| `run_full_eval.py` | import + 123-124, print 166-168 | `two_proportion_z(our, n, 57/62, n)` as sole stat | `paired_vs_aggregate_baseline(n, our, 57/62)` primary; z kept as labelled diagnostic. Also enriched the `mcnemar_vs_scene_only` print with `b`, `c`, and the new paired CI (this comparison already had real per-clip predictions for both sides, so it now gets the FULL fix — exact p + paired CI, not just a bound). |
| `ensemble_relation.py` | import + 42-44 | same pattern, loop over `('no-bbox baseline',57),('full-bbox',62)` | same swap; dropped an unused `mcnemar_exact` import that was never called in this file. |
| `run_relation_transformer.py` | import (line 24) + line 220 (blind vs no-bbox) + line 236 (ensemble-gated vs no-bbox) | `two_proportion_z(...)` | `paired_vs_aggregate_baseline(64, ..., 57, label=...)`. **Only these 3 spots touched** — no other line in this file was edited, per instructions (training/model code here is slated for rewrite by another workstream). |

`unsafenet/metrics_stats.py` (owns `mcnemar_bound`, imported by `mcnemar_discordance_bound`) was
**not modified** — it's outside my file-ownership grant, and I didn't need to touch it: the new
`max_p_over_splits` is computed by a small companion function added to `paired_tests.py` instead,
using the same enumeration logic independently rather than editing the out-of-scope file.

Not touched (out of scope, flagged only): `run_ablation_significance.py` has its own inline
McNemar-bound reimplementation (uses continuity-corrected chi-square, not exact binomial, at the
bound extremes) — same category of issue as finding #1 but a separate call site outside my file
ownership; `deployment/`, `tools/`, `build_relation_tokens.py` untouched (other work streams).

## Tests — actual run output

New file `tests/test_paired_significance.py` (125 lines), plain-assert style matching the
project's existing no-pytest convention (`eval_utils/tests/run_all.py`). Ran directly:

```
$ .venv/bin/python tests/test_paired_significance.py
  PASS  test_b_plus_c_zero_edge_case
  PASS  test_bound_max_p_never_below_min_p_over_random_cases
  PASS  test_bound_path_when_only_aggregate_counts_known
  PASS  test_paired_diff_bootstrap_ci_matches_point_estimate
  PASS  test_perfectly_discordant_case
  PASS  test_worked_example_hand_computed_exact_p
ALL PASS
```

Coverage: (1) worked example n=6, b=1,c=5, hand-computed p=2*(1+6)/64=0.21875, cross-checked
against `scipy.stats.binomtest`; (2) b+c=0 edge case (identical predictions) — p=1.0, CI collapses
to a point at 0; (3) perfectly-discordant case (a always right, b always wrong) — p=2*0.5^n,
CI collapses to a point at +100pp; (4) bound path reproduces the pre-existing hand-verified
audit-table value (min_p=0.03125 for 56 vs 62 of 64) and adds max_p >= min_p plus the
diagnostic-warning field; (5) a property test that max_p never falls below min_p across 20 random
(n, correct_a, correct_b) triples; (6) bootstrap CI point estimate sanity check.

Also re-ran the full pre-existing suite to confirm no regression:

```
$ PYTHONPATH=. .venv/bin/python eval_utils/tests/run_all.py
... (60 tests) ...
ALL PASS
```

(60/60, same count the prior audit reported — nothing broken by the new `b`/`c`/CI fields added to
`mcnemar_exact`'s return dict, since existing consumers only read the pre-existing keys.)

## Scripts re-run for real (no training, only loading saved `.pt` logits)

`run_full_eval.py` (loads `sweep_results/relation_tokens/logits_yolo26.pt`) and
`ensemble_relation.py` (loads `logits_v1nocw.pt` + `logits_v2bnocw.pt`) both executed end-to-end
successfully with the new code and produced live output (shown below). `run_relation_transformer.py`
was **not** re-run (it trains from scratch — explicitly disallowed); instead its 2 comparison call
sites were recomputed directly from already-saved/already-logged aggregate counts (54/64 blind,
confirmed identical to `run_full_eval.py`'s `blind_pred_top1_correct` field in the just-regenerated
`full_eval.json`; 53/64 ensemble-gated, from the already-committed `sweep_results/relation_tokens/train_s1b.log`)
— this is exactly the "re-run cheaply from saved predictions" case the task asked for, with zero
retraining and zero new touches to the test split.

## Old (two_proportion_z) vs new (McNemar bound) — the numbers the thesis needs

n=64 in every row (the paper's 4-class unsafe-subset clip count).

| Comparison | Old two_proportion_z p (verdict) | New McNemar bound p∈[min,max] (verdict) |
|---|---|---|
| `run_full_eval.py`: S1a ensemble 54/64 vs no-bbox 57/64 | p=0.435 (not sig.) | [0.250, 0.801] (cannot reach p<0.05 at all — min_p itself is 0.25) |
| `run_full_eval.py`: S1a ensemble 54/64 vs full-bbox 62/64 | **p=0.015 (SIGNIFICANT)** | [0.008, 0.382] (best-case split could be significant; true p unknown, plausibly up to 0.38 — **no longer a confirmed significant result**) |
| `ensemble_relation.py`: hybrid ensemble 51/64 vs no-bbox 57/64 | p=0.144 (not sig.) | [0.031, 0.532] (best-case split could be significant; true p unknown, plausibly up to 0.53) |
| `ensemble_relation.py`: hybrid ensemble 51/64 vs full-bbox 62/64 | **p=0.003 (SIGNIFICANT)** | [0.001, 0.207] (best-case split could be significant; true p unknown, plausibly up to 0.21 — **no longer a confirmed significant result**) |
| `run_relation_transformer.py`: blind 54/64 vs no-bbox 57/64 | p=0.435 (not sig.) | [0.250, 0.801] (cannot reach p<0.05 at all) |
| `run_relation_transformer.py`: ensemble-gated (hybrid+gate) 53/64 vs no-bbox 57/64 | p=0.309 (not sig.) | [0.125, 0.708] (cannot reach p<0.05 at all) |

`run_full_eval.py`'s own new `mcnemar_vs_scene_only` line (a case where per-clip predictions on
BOTH sides are actually available, so this gets the exact test, not a bound): b=9, c=7, p=0.804,
diff=+3.1pp, paired bootstrap CI95=[-9.4,+15.6]pp — unchanged conclusion (not significant), but
now reports b/c and a real paired CI instead of just p.

### Explicit statement: which previously-claimed "significant" results survive

**Neither of the two results that two_proportion_z called "significant" (S1a ensemble 54/64 vs
paper's full-bbox 62/64, p=0.015; hybrid ensemble 51/64 vs full-bbox 62/64, p=0.003) survives as a
confirmed significant result under the correct paired treatment.** The McNemar bound shows the
true (unknown, because the paper's per-clip predictions are unpublished) p-value could be as low
as 0.008–0.001 (in the most favourable discordant split) but could equally be as high as 0.38–0.21
(in the least favourable split consistent with the same aggregate counts) — the honest statement is
"inconclusive, bounded, not proven significant," not "significant." This matches the review's own
prediction: the earlier hand-worked audit table (`t1-objectives-audit.md`) already concluded no
our-vs-paper comparison is confirmably significant either way, and this fix now makes the
number-producing scripts themselves say so instead of printing an unguarded, falsely-precise
two-proportion z p-value. No comparison in this table flips from "not significant" to
"significant" in the other direction — the correction only removes false-positive "significant"
claims, it never manufactures new ones (consistent with the review's point that the independence
violation is anti-conservative, i.e. it can only have inflated apparent significance, never
deflated it).

## Issues / deviations

- `run_relation_transformer.py` is 280 lines, over the 200-line guideline — pre-existing, not
  touched beyond the 2 statistics call sites per explicit instruction (file is slated for a
  training-code rewrite by another work stream).
- The historical `train_s1b.log` / `train_yolo11_baseline.log` reused for the 53/64 gated-ensemble
  number predate the `2026-09-06` `logits_yolo26.pt` provenance-recovery fix mentioned in
  `run_full_eval.py`'s own comments; I used them only for their aggregate correct-count (53/64),
  which is a simple integer re-derivable from any correctly-trained run at this config and is not
  sensitive to that fix (which only concerns which softmax array is persisted, not the top-1 count
  already printed to the log at training time).

## Follow-up (2026-09-16, same day): `run_ablation_significance.py` DRY-fix + explicit answers

Coordinator asked for two follow-ups so the fix is complete, not partial. Both done, not committed.

### Follow-up 1 — routed `run_ablation_significance.py` through the shared module

This file was outside the original ownership grant; coordinator explicitly authorized touching
its statistics call sites only (nothing else — model/feature-extraction code in this file
untouched).

**Removed** (dead code, DRY): local `two_prop_z(k1,n1,k2,n2)` (duplicate of
`eval_utils.paired_tests.two_proportion_z`) and local `mcnemar_marginal_bounds(k_a,k_b,n)` — the
"separate, drifting bound implementation" the coordinator flagged. That local function used the
**continuity-corrected chi-square approximation** (`chi2=(max(0,|diff|-1))**2/D`, converted via
`1-chi2.cdf(chi2,df=1)`) at both bound extremes, not the exact binomial CDF `paired_tests.py` uses.

**Added**: `from eval_utils.paired_tests import paired_vs_aggregate_baseline, mcnemar_exact`;
`main()`'s `[3]` section now calls `paired_vs_aggregate_baseline` for both paper comparisons
(55/64 ConvLSTM baseline, 62/64 full pipeline) exactly like the other 3 fixed scripts. Kept
`macro_z_simulated` untouched — separate methodological question (macro accuracy is a mean of
per-class recalls, not per-clip binary correctness, so it is not a McNemar/paired-proportion case
at all; noted inline in the file so it isn't mistaken for an oversight).

**Did the old bound disagree with the new one? Yes, on one of the two cases — materially:**

| Comparison (k_a=ours, k_b=paper, n=64) | OLD chi2-continuity bound p∈[min,max] | NEW exact-binomial bound p∈[min,max] | Disagreement |
|---|---|---|---|
| ours 56/64 vs paper baseline 55/64 (diff=1) | [1.0000, 1.0000] | [1.0000, 1.0000] | **None** — both floor to p=1 at this tiny diff (chi2 continuity-corrects to 0 at diff=1; exact-binomial independently finds no significant split possible) |
| ours 56/64 vs paper full 62/64 (diff=-6) | [0.0412, 0.1138] | [0.0312, 0.5323] | **Large, one-sided** — the OLD max-p (least-significant/most-conservative bound) was 0.114, the TRUE max-p is 0.532. The old implementation understated the width of the uncertainty interval by >4x at the conservative end. Both old and new min-p stay under 0.05 (0.041 vs 0.031, no qualitative flip there), but a reader trusting the old max-p=0.114 would wrongly conclude "borderline, maybe salvageable with more data" when the honest worst-case is p=0.53 — solidly non-significant. |

Live re-run reproduced these exact numbers (`.venv/bin/python run_ablation_significance.py`):
```
[3] McNemar bound vs paper (primary) + two_proportion_z (labelled diagnostic only)
  top-1  ours 56/64 vs paper 55/64:  McNemar bound p∈[1.0000,1.0000]  Δ=+1.56pp  (diagnostic two_proportion_z p=0.7945, invalid independence)
  top-1  ours 56/64 vs paper 62/64:  McNemar bound p∈[0.0312,0.5323]  Δ=-9.38pp  (diagnostic two_proportion_z p=0.0481, invalid independence)
```
Neither of these two comparisons was ever reported as "significant" under either old or new bound
(the old method's own min-p of 0.041 was reported as a bound, not a point p, so no prior claim of
confirmed significance exists here to retract) — the practical effect of this fix is tightening
what "not disprovable" means for the 62/64 case from a falsely-narrow [0.041,0.114] to the honest
[0.031,0.532].

### Follow-up 2, part (a) — is the wide bound on paper comparisons permanent?

**Yes. Confirmed. Stated plainly: no valid paired significance test against the published
paper baselines is possible in principle, for any of the 6 paper comparisons in this fix
(57/64, 62/64, 55/64) — this is not a temporary limitation to be fixed later, it is the correct
and final answer.**

Reasoning: McNemar's test (exact or otherwise) requires the per-clip (b, c) discordant-pair table
— i.e., knowing, clip-by-clip, whether the paper's model got each of the same 64 test clips right
or wrong, so it can be cross-tabulated against our own per-clip correctness. The paper (Önal &
Dandıl 2024, Unsafe-Net) published only aggregate Table 7 top-1 counts (57/64, 62/64) and a
separate 55/64 ConvLSTM ablation baseline number — never per-clip predictions, and the paper's
own model/weights/inference code needed to regenerate per-clip predictions are not available to
us either (it is somebody else's already-published, already-frozen experiment, not one we can
re-run). There is no way to obtain that table now or in the future short of the original authors
publishing per-clip results, which is outside anyone's control here. Checked all 6 paper
comparisons across the 4 fixed scripts (`run_full_eval.py` x2, `ensemble_relation.py` x2,
`run_relation_transformer.py` x2, `run_ablation_significance.py` x2 — 8 call sites total, 6
distinct aggregate targets: 57/64, 62/64, 55/64) — every one is aggregate-only vs the paper, none
has a path to per-clip data. **No comparison exists where the paper's per-clip predictions could
be plumbed through** — the bound is the ceiling on precision this comparison can ever have, and
the thesis should word every "vs paper" comparison as a bound (McNemar discordance interval), never
as a point p-value, permanently.

### Follow-up 2, part (b) — which own-vs-own comparisons use the exact test vs the bound?

Audited every McNemar/bound call site (grep across all touched files + `eval_utils/suite.py` +
`unsafenet/scene_confound.py`). Split into 3 categories:

**(i) Already using the exact paired test** (per-clip data for both sides available in-memory,
same clips):
- `run_full_eval.py`: `mcnemar_vs_scene_only` — our model vs the scene-only floor predictor
  computed live in-script. Already exact before this follow-up (unchanged by it).
- `unsafenet/scene_confound.py`: model-vs-floor comparison — same pattern, already exact.
- `eval_utils/suite.py::_compare`: branches to exact McNemar whenever the caller supplies
  `other_pred` (per-clip); only falls back to the bound when the caller doesn't have per-clip data
  for the other side. Conditional by design, not a gap.

**(ii) Fixed in this follow-up session (were gaps, now exact)** — identified because per-clip
predictions for BOTH sides were already sitting in memory/on disk in these scripts but no
significance test was being computed between them at all:
- `ensemble_relation.py`: hybrid ensemble vs `v1 geom no-weights` and vs `v2b appearance nw`
  (`a['blind_pred']`, `b['blind_pred']` were already loaded from the saved `.pt` files but unused
  for testing). Added `mcnemar_exact` calls; live output: hybrid(51/64) vs v1(51/64) b=1 c=1
  p=1.0000 diff=+0.00pp; hybrid(51/64) vs v2b(52/64) b=2 c=3 p=1.0000 diff=-1.56pp — both
  unsurprisingly non-significant (the hybrid is a 50/50 blend of these two, so high agreement is
  expected).
- `run_ablation_significance.py`: ensemble vs direct4 and ensemble vs cascade
  (`pred_ens`/`pred_d4`/`pred_cas` all computed in-script on the same 64 clips, printed side by
  side with no test). Added `mcnemar_exact` calls; live output: ensemble(56/64) vs
  direct4(55/64) b=1 c=0 p=1.0000 diff=+1.56pp; ensemble(56/64) vs cascade(51/64) b=5 c=0
  p=0.0625 diff=+7.81pp CI95=[+1.6,+14.1]pp — the ensemble-vs-cascade gap is the closest to
  significant of any own-vs-own comparison here (p=0.0625, just above 0.05) but does not cross the
  conventional threshold.
- `run_relation_transformer.py`: BLIND (val-selected seed) vs ensemble (test-selected caveat)
  (`blind['pred']`, `ens_pred` both in-script, same `yte_np`). Added `mcnemar_exact` call at the
  one new statistics call site (directly adjacent to the pre-existing paper-comparison call sites,
  within the "statistics call sites" scope of the standing restriction on this file — no
  model/training code touched). **Not re-run** (requires training, disallowed) — code path
  verified via `ast.parse` only, same as the file's other pre-existing call sites.

No own-vs-own comparison remains that has per-clip data for both sides and still falls back to the
bound — all 3 identified gaps are now fixed.

## Updated old-vs-new summary table (all files, after both follow-ups)

| Comparison | Bound type | p (or p∈[min,max]) | Verdict |
|---|---|---|---|
| `run_full_eval.py` ensemble 54/64 vs no-bbox 57/64 | bound (paper, permanent) | [0.250, 0.801] | not significant, cannot be |
| `run_full_eval.py` ensemble 54/64 vs full-bbox 62/64 | bound (paper, permanent) | [0.008, 0.382] | **no longer confirmed significant** |
| `run_full_eval.py` vs scene-only floor | exact (own vs own) | p=0.804, b=9,c=7 | not significant |
| `ensemble_relation.py` hybrid 51/64 vs no-bbox 57/64 | bound (paper, permanent) | [0.031, 0.532] | not significant |
| `ensemble_relation.py` hybrid 51/64 vs full-bbox 62/64 | bound (paper, permanent) | [0.001, 0.207] | **no longer confirmed significant** |
| `ensemble_relation.py` hybrid vs v1 (NEW) | exact (own vs own) | p=1.000, b=1,c=1 | not significant |
| `ensemble_relation.py` hybrid vs v2b (NEW) | exact (own vs own) | p=1.000, b=2,c=3 | not significant |
| `run_relation_transformer.py` blind 54/64 vs no-bbox 57/64 | bound (paper, permanent) | [0.250, 0.801] | not significant |
| `run_relation_transformer.py` ensemble-gated 53/64 vs no-bbox 57/64 | bound (paper, permanent) | [0.125, 0.708] | not significant |
| `run_relation_transformer.py` blind vs ensemble (NEW) | exact (own vs own) | not re-run (needs training; code verified) | n/a |
| `run_ablation_significance.py` ours 56/64 vs paper baseline 55/64 | bound (paper, permanent) | [1.000, 1.000] | not significant |
| `run_ablation_significance.py` ours 56/64 vs paper full 62/64 | bound (paper, permanent) | [0.031, 0.532] | **not significant (was falsely narrow [0.041,0.114] under old chi2 bound)** |
| `run_ablation_significance.py` ensemble vs direct4 (NEW) | exact (own vs own) | p=1.000, b=1,c=0 | not significant |
| `run_ablation_significance.py` ensemble vs cascade (NEW) | exact (own vs own) | p=0.0625, b=5,c=0 | not significant (closest own-vs-own result to the threshold) |

**No previously-"significant" claim survives** (same conclusion as the original report — the two
`two_proportion_z` p<0.05 results, S1a-vs-full-bbox and hybrid-vs-full-bbox, remain unconfirmed
under the bound). No new significant result was introduced by fixing the 3 own-vs-own gaps either.

## Tests / regression re-check after follow-up

```
$ PYTHONPATH=. .venv/bin/python eval_utils/tests/run_all.py     # 60/60 PASS
$ .venv/bin/python tests/test_paired_significance.py            # 6/6 PASS
$ git status --porcelain -- run_full_eval.py ensemble_relation.py run_relation_transformer.py \
    run_ablation_significance.py eval_utils/paired_tests.py tests/
 M ensemble_relation.py
 M eval_utils/paired_tests.py
 M run_ablation_significance.py
 M run_full_eval.py
 M run_relation_transformer.py
?? tests/
```
Only the 5 authorized files (+ new tests/) touched. No commit made.

## Unresolved questions

1. `README.md` (out of my file ownership) currently states "z = −0.27, p = 0.79" for the
   56/64-vs-57/64 comparison as a caveat line — that's the same category of invalid
   independence-assumption stat as finding #1, just a different pair of counts than any of my
   call sites computed live. Someone with README ownership should either swap it for the
   McNemar-bound treatment or add the same "assumes independent samples, does not hold here"
   caveat inline. (`run_ablation_significance.py` follow-up is now done, so this is the one
   remaining known stale stat outside this fix's file ownership.)
