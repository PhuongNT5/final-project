# Eval/statistics audit — eval_utils, unsafenet, run_*.py

Scope: eval_utils/*.py + tests, unsafenet/metrics.py+metrics_stats.py+scene_confound.py,
run_full_eval.py, run_kfold_eval.py, run_kfold_relation.py, run_operating_point.py,
run_blind_eval.py, run_relation_transformer.py, ensemble_relation.py, run_ablation_calibration.py.
Method: read code, cross-checked formulas against sklearn/scipy numerically (see snippets below),
ran `eval_utils/tests/run_all.py` (all 60 pass), grepped for call-site misuse across the repo.

Verdict up front: formula implementations are correct (verified against sklearn/scipy to 1e-9-1e-16).
The codebase is unusually disciplined about test-set leakage (self-flagging `test_selected=True`,
val-fit/test-apply patterns, honest touch log). The one real, live finding is #1 below.

## Ranked findings

### 1. HIGH — `two_proportion_z` reported as the "vs paper" stat in 3 live scripts; the paired-safe tool exists but isn't wired in
- Files: `run_full_eval.py:123-124,168`, `run_relation_transformer.py:220,236`, `ensemble_relation.py:43-46`
- `two_proportion_z(k1,n1,k2,n2)` (`eval_utils/paired_tests.py:40-47`) assumes 2 INDEPENDENT samples
  of n=64. Our result and the paper's 57/64 or 62/64 are scored on the SAME 64 clips (2 raters on one
  item set), not 2 disjoint samples of 64 → double-counts n as 128 independent Bernoulli draws.
  Correct treatment: paired (McNemar), but paper's per-clip predictions are unpublished so exact
  McNemar is impossible — the only valid move is a BOUND, which the repo already has
  (`mcnemar_discordance_bound` / `unsafenet.metrics_stats.mcnemar_bound`, enumerates every discordant
  split consistent with the aggregate counts, returns the smallest possible p as a LOWER bound).
- Evidence, computed: for 56/64 vs paper 62/64, `two_proportion_z` gives p=0.048 (reads "borderline
  significant"); the true bound is p ≥ 0.03125 with the *actual* p unknowable and plausibly much
  higher (independence violated in the direction that inflates apparent significance when
  correctness is positively correlated across raters on the same items, which it almost certainly is
  here — same clips, same difficulty). Numbers reproduced live:
  ```
  56/64 vs 62/64: two_proportion_z p=0.048   mcnemar_discordance_bound min_p=0.03125 (bound only, not exact)
  54/64 vs 57/64: two_proportion_z p=0.435   mcnemar_discordance_bound min_p=0.25
  ```
- The correct tool is already implemented and even wired into `eval_utils/suite.py::_compare` (computes
  BOTH `mcnemar_discordance_bound` and `two_proportion_z` side by side), but `evaluate_suite` is called
  ONLY from tests (`grep` confirms zero production call sites) — none of the actual number-producing
  scripts (`run_full_eval.py`, `run_relation_transformer.py`, `ensemble_relation.py`) go through it.
  Those 3 scripts print/save ONLY `two_proportion_z`'s z/p as the headline "vs paper" line.
- Impact: if a printed p-value from these scripts is pasted into the thesis as "significant" or
  "not significant" vs the paper, it rests on an invalid independence assumption. Mitigating: a prior
  audit (`plans/2026-09-05-thesis-final-push/reports/t1-objectives-audit.md:135-171`) already worked
  the McNemar-bound table by hand and concluded no our_correct value beats the paper significantly
  either way — so the CONCLUSION these scripts would produce is probably not wrong, but the SCRIPTS
  themselves don't carry that caveat/bound, so re-running them for a new config gives an unguarded,
  falsely-precise number.
- Fix: either (a) swap these 3 scripts to call `mcnemar_discordance_bound` as the primary "vs paper"
  stat, printing `two_proportion_z` only as a labeled "diagnostic, assumes independence which does
  not hold here" secondary line, matching `suite.py`'s own pattern; or (b) route them through
  `evaluate_suite`/`_compare` instead of duplicating the comparison logic.

### 2. MEDIUM — Nadeau-Bengio applied to pooled repeated-k-fold diffs (J=k·r): correct convention but undisclosed extra-correlation caveat
- Files: `eval_utils/paired_tests.py:92-105`, `run_kfold_eval.py:73-81,140-142` (duplicate inline copy),
  `run_kfold_relation.py:20,88` (uses the shared one, J=k·repeats, comment at line 8-9 is explicit)
- Formula itself matches spec exactly: `corr = 1/J + n_test/n_train`, `t = mean/sqrt(var*corr)`,
  `df=J-1`, verified against hand formula in `test_nadeau_bengio_matches_hand_formula`. Using
  J=k·r (pool all seed×fold diffs into one array) for repeated k-fold is the standard extension in
  the literature (Bouckaert & Frank 2004's "corrected repeated k-fold CV test"), so this is not a bug.
- Caveat not stated anywhere in code or (as far as searched) in report text: Bouckaert & Frank (2004)
  showed this correction, derived for a SINGLE k-fold's train/test overlap, still under-corrects
  variance for repeated CV because it does not model the extra correlation introduced by re-using the
  same fixed pool of clips across different seeds/repeats (repeats are not independent replications —
  they all draw folds from the same 516 or 64 clips). The known result is elevated Type-I error
  ("too liberal") that WORSENS as more repeats are pooled into J, i.e. more "significant" results than
  the nominal alpha would justify. `run_kfold_eval.py` pools 3 seeds × 5 folds = J=15;
  `run_kfold_relation.py` allows `--repeats` up to 10× (J up to 50) — the more repeats, the more this
  approximation is stressed.
- Fix: add one line to thesis text/report wherever an NB p-value from these scripts is quoted: "the
  corrected resampled t-test is known to be anti-conservative under repeated k-fold pooling
  (Bouckaert & Frank 2004); treat p-values near 0.05 as indicative, not exact." No code change needed
  — this is a disclosure gap, not a formula bug.

### 3. LOW — duplicate, drifting implementations of the same statistics (DRY)
- `nadeau_bengio`: `eval_utils/paired_tests.py:92-105` and inline copy in `run_kfold_eval.py:73-81`
  (functionally identical today, verified by inspection, but nothing prevents drift — `run_kfold_eval.py`
  even predates the shared module per its own docstring history).
- `mcnemar_exact`: two DIFFERENT signatures for the same test — `eval_utils.paired_tests.mcnemar_exact(correct_a, correct_b)`
  (takes boolean correctness arrays) vs `unsafenet.metrics_stats.mcnemar_exact(y, pred_a, pred_b)` (takes
  raw predictions). All current call sites use the right one for their import (`run_full_eval.py` /
  `eval_utils/suite.py` use the `eval_utils` form correctly; `scene_confound.py` uses the `unsafenet`
  form correctly) — no live misuse found, but the signature mismatch is a footgun for future callers
  who copy a call from one module into a script importing the other.
- `bootstrap_ci`: `eval_utils/bootstrap_ci.py` supports `method='percentile'|'bca'`; `unsafenet/metrics_stats.py::bootstrap_ci`
  is percentile-only but is called from `unsafenet/metrics.py::report()` under the key
  `macro_accuracy_ci_bootstrap` / `macro_f1_ci_bootstrap` with no "percentile" qualifier in the key
  name, even though brief S4.4f asks for BCa preferred at small N. Currently harmless: `report()` is
  only invoked via `eval_utils/suite.py`, which overrides these fields with its own explicit
  `bootstrap_bca_5000`-labeled CIs before exposing them (`suite.py:58-63,71-76`) — so today's exposed
  numbers are BCa. But if `unsafenet.metrics.report()` is ever called directly (e.g. a new script,
  a notebook) its `..._ci_bootstrap` keys will silently be percentile, not BCa, under a generic name.
- Fix (optional, non-blocking): either delete the inline `nadeau_bengio` in `run_kfold_eval.py` and
  import the shared one, or rename `unsafenet/metrics.py`'s report keys to `..._ci_bootstrap_percentile`
  to make the method explicit at the call site.

## Formula correctness — verified, no issues
All checked numerically against an independent reference (sklearn/scipy or a hand-derived closed form):
- Wilson CI (`eval_utils/paired_tests.py:50-59`, `unsafenet/metrics_stats.py:19-28`): matches Newcombe
  (1998) worked example and hand-solved quadratic (repo tests already do this, confirmed passing).
- BCa bootstrap (`eval_utils/bootstrap_ci.py:33-56`): z0/a formula standard Efron-Tibshirani; repo test
  cross-checks against `scipy.stats.bootstrap(method='BCa')`, agrees within tolerance (passing).
- McNemar exact (`eval_utils/paired_tests.py:17-37`, `unsafenet/metrics_stats.py:45-52`): exact binomial
  on discordant pairs, two-sided, `2*binom.cdf(min(b,c), n, 0.5)` — matches `scipy.stats.binomtest`
  exactly in repo test and in an independent re-check.
- Conformal quantile (`eval_utils/conformal.py:29-41`): `level = min(1, ceil((n+1)(1-alpha))/n)` —
  exact match to the Vovk et al. Lemma-1 formula given in the task spec; clipping to 1.0 for small n_cal
  is the correct documented edge case, not a bug.
- AUROC/AUPRC one-vs-rest macro averaging (`unsafenet/metrics.py:77-106`): independently re-verified
  against `sklearn.roc_auc_score(...,multi_class='ovr',average='macro')` and
  `average_precision_score` per-class — diffs ~1e-16 (see transcript). NaN-safe exclusion of absent
  classes from the macro mean is correct and matches brief's intent.
- ECE binning (`eval_utils/calibration.py:9-27`, `unsafenet/metrics.py:109-122`): equal-width bins,
  `(lo,hi]` except first bin `[lo,hi]` — standard Guo et al. (2017) convention, both implementations
  agree.
- Cohen's kappa / MCC (`unsafenet/metrics.py:61-74`): exact match to `sklearn.cohen_kappa_score` /
  `sklearn.matthews_corrcoef` (diff 0.0).
- `required_n_noninferiority` / `fisher_z_ci` (`eval_utils/power_analysis.py`): standard closed-form
  normal-approximation formulas; repo tests cross-check against a separately published table
  (`r5-alternative-evaluation-frameworks.md`) and hand-solved Fisher-z — passing.

## Test suite quality (task item 3)
`eval_utils/tests/run_all.py` → 60/60 PASS. Genuinely external-reference-checked, not tautological, for
the load-bearing formulas: Wilson vs Newcombe 1998, McNemar vs `scipy.binomtest`, BCa vs
`scipy.stats.bootstrap`, power/Fisher-z vs the independently published r5 table, classification metrics
vs hand-computed toy cases, real-artifact tests reproduce committed thesis numbers
(`test_real_artifacts.py`) including an inline independent sklearn re-derivation. The one weak spot:
`test_two_proportion_z_symmetry` only checks internal antisymmetry, not correctness against an external
reference (e.g. `statsmodels.stats.proportion.proportions_ztest`) — low priority given finding #1 means
this function's use case is itself questionable, not the formula.

## Protocol discipline — test-set leakage (task item 4)
No test-set-selection leakage found in the current scripts; the codebase actively defends against it:
- `eval_utils/operating_point.py::select_operating_point_by_safety_recall` self-flags `test_selected=True`
  with an explicit "diagnostic only, not evidence-grade" note when no independent selection set is
  passed (lines 62-63, 89-91).
- `run_operating_point.py` prints a `[!] WARNING` banner and refuses to call the result evidence-grade
  when `--select_logits` isn't given (lines 76-79).
- `run_ablation_calibration.py`: temperature T selected via `argmin NLL` on the TRAIN sweep, applied
  once to TEST (val-fit/test-apply, explicitly labeled "D3 FIX", lines 115-125); the TEST-side sweep is
  printed only as a labeled diagnostic, never used to pick T.
- `run_blind_eval.py`: epoch/config/seed selection all done on a stratified slice carved from TRAIN;
  test touched once per config, explicitly built to fix an earlier acknowledged leakage bug (module
  docstring lines 3-14).
- `unsafenet/scene_confound.py`: the train-only floor (`floor_test`) is kept separate from the
  pooled-fit optimistic floor (`floor_pooled`), which is explicitly labeled "optimistic ceiling" and
  not used for the model-vs-floor McNemar comparison (that uses the train-fit floor correctly).
- `eval_utils/conformal.py::split_conformal_repeated`: re-splits the 64 (or pooled) clips into fresh
  cal/test partitions 1000×, reporting the distribution — not a cherry-picked favorable split; this is
  the correct way to use conformal prediction when no separate calibration set exists.
- `reports/test-touch-log.md` shows an active, honored one-touch-per-variant discipline (integrity
  checks before believing anomalous results, e.g. the 2026-09-07 LCO cs=2 entry).

## Unresolved questions
1. Where (if anywhere) do the `two_proportion_z` printed lines from `run_full_eval.py` /
   `run_relation_transformer.py` / `ensemble_relation.py` currently get copied into thesis prose? If
   nowhere yet, finding #1 is a preventive fix; if already quoted somewhere, that text needs a check/
   correction pass.
2. Confirm whether `run_kfold_relation.py --repeats` was ever run above ~5 for a number destined for the
   thesis (higher repeats stress the NB anti-conservativeness noted in finding #2 more).
