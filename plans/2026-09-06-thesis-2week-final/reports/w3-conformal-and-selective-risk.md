# W3 — Conformal prediction, selective risk, power analysis, resolving power

Date 2026-09-06. New library code: `eval_utils/conformal.py`, `eval_utils/selective_risk.py`,
`eval_utils/power_analysis.py` (+ tests). Full suite verified: `.venv/bin/python -m
eval_utils.tests.run_all` -> **ALL PASS** (54 tests total across all eval_utils modules, including
the 45 pre-existing ones from the concurrent lane — none broken).

Analysis driver (not committed, per file-ownership — reads eval_utils + sweep_results only, writes
JSON+PNGs into this reports dir): kept in session scratchpad, re-runnable, no library logic lives
there. Raw numbers: `reports/w3-figures/w3_results.json`. Figures: `reports/w3-figures/risk_coverage_*.png`.

Data: `sweep_results/relation_tokens/logits_{yolo26,s1a,yolo11_baseline,yoloroi,v1nocw,v2bnocw}.pt`,
each `ens_softmax` (64,4) + `labels` (64,), the SAME 64-clip test set as every other headline number
in this project. **Disclosure: `logits_s1a.pt` and `logits_yolo11_baseline.pt` are bit-identical**
(`ens_softmax`, `labels`, `blind_pred` all `torch.equal`) — verified directly, not assumed. Treated
below as one underlying result reported under two names, not two independent configs; do not double-count
it as corroborating evidence across "5 distinct configs."

Per-config top-1 on this run (for context, not a new headline — matches existing project numbers where
overlapping): yolo26 52/64 (81.25%), s1a=yolo11_baseline 54/64 (84.38%), yoloroi 54/64 (84.38%),
v1nocw 49/64 (76.56%), v2bnocw 52/64 (81.25%).

---

## 1. Split-conformal prediction (marginal coverage)

Method: nonconformity score alpha_i = 1 - softmax(true label) (LAC score, Vovk/Angelopoulos-Bates).
Per config: 2000 random 50/50 calibration/test splits of the same 64 clips (n_cal=32, n_test=32),
quantile level `ceil((n_cal+1)(1-alpha))/n_cal` clipped to <=1. Reporting the DISTRIBUTION across
splits, not one split — n=64 makes a single split's empirical coverage a noisy point estimate.

| config | alpha | nominal cov | mean emp. coverage | std | [p05,p95] | mean set size | std |
|---|---|---|---|---|---|---|---|
| yolo26 | .10 | .90 | .9408 | .0564 | [.844, 1.00] | 1.56 | .199 |
| yolo26 | .05 | .95 | .9700 | .0403 | [.906, 1.00] | 2.49 | .824 |
| s1a / yolo11_baseline | .10 | .90 | .9405 | .0573 | [.813, 1.00] | 1.42 | .144 |
| s1a / yolo11_baseline | .05 | .95 | .9694 | .0416 | [.875, 1.00] | 1.98 | .508 |
| yoloroi | .10 | .90 | .9413 | .0572 | [.844, 1.00] | 1.45 | .127 |
| yoloroi | .05 | .95 | .9707 | .0403 | [.875, 1.00] | 1.99 | .505 |
| v1nocw | .10 | .90 | .9408 | .0575 | [.813, 1.00] | 1.50 | .206 |
| v1nocw | .05 | .95 | .9697 | .0408 | [.875, 1.00] | 2.05 | .461 |
| v2bnocw | .10 | .90 | .9408 | .0569 | [.844, 1.00] | 1.53 | .250 |
| v2bnocw | .05 | .95 | .9695 | .0418 | [.875, 1.00] | 1.96 | .329 |

**Reading this.** Mean empirical coverage (.940-.941 at target .90; .969-.970 at target .95) sits
*above* nominal in every config — consistent with the theorem (marginal coverage is guaranteed
>= 1-alpha in expectation over the calibration draw; at this n_cal=32 the discreteness of the
`ceil(...)/n_cal` quantile level tends to over-cover rather than land exactly on target, a known
small-n conformal artifact, not a violation). The wide std (.04-.06) and the p05 dropping to
.81-.84 confirm the resolution caveat: **any single split can look like it "misses" 90% coverage
by several points purely from resampling noise; only the averaged-over-splits number is
informative at n=64.**

**Set size is the deployment-relevant number.** Average set size 1.4-1.6 at 90% target (out of 4
classes) — the models are genuinely discriminative, not defaulting to "cover everything." At 95%
target, set size rises to ~2.0-2.5, i.e. the extra 5pp of guaranteed coverage is bought by roughly
doubling how often two classes are returned instead of one (a near-abstention on class identity,
useful framing for a human-review UI: "system says either A or B, defer to operator").

**What this buys and does not buy.** The coverage GUARANTEE itself (P(Y in C(X)) >= 1-alpha) is
exact and distribution-free for any n, including n=64, PROVIDED calibration/test exchangeability
holds (Vovk et al. 2005, Lemma 1) — this is the one number in the whole evaluation-frameworks
program whose validity is not asymptotic. It does **not** mean the paper is beaten, matched, or
compared at all; conformal calibrates uncertainty on OUR OWN model, it says nothing about the
opponent, who reports no confidence scores at all. It also does not mean any given single 90%-target
prediction set on new data is 90% likely correct in a per-clip sense beyond the marginal average.

**Mondrian per-class refused by construction.** `eval_utils.conformal.class_conditional_conformal()`
raises `NotImplementedError` unconditionally — verified by
`test_class_conditional_conformal_refuses`. Smallest class (CarryOverload) is n=8 in this 64-clip
test set; a per-class conformal quantile at n=8 has ~1/9 (~11pp) step granularity, i.e. moving the
"guarantee" at all requires re-weighting one whole clip's nonconformity mass — not a verifiable
guarantee at any usable alpha. No per-class coverage number is reported anywhere in this deliverable.

---

## 2. Selective risk / risk-coverage / AURC

Ranking by max-softmax confidence, 0/1 loss, oracle = rank by actual correctness (isolates the
irreducible error floor from confidence-ranking failure).

| config | AURC (model) | AURC* (oracle) | excess AURC | risk@80% cov | risk@50% cov | full-coverage risk (n=64) |
|---|---|---|---|---|---|---|
| yolo26 | .0634 | .0188 | .0446 | 11.8% (m=51) | 3.1% (m=32) | 18.75% |
| s1a/yolo11_baseline | .0660 | .0129 | .0531 | 13.7% (m=51) | 6.3% (m=32) | 15.63% |
| yoloroi | .0633 | .0129 | .0504 | 15.7% (m=51) | 3.1% (m=32) | 15.63% |
| v1nocw | .0707 | .0299 | .0408 | 15.7% (m=51) | 3.1% (m=32) | 23.44% |
| v2bnocw | .0584 | .0188 | .0396 | 13.7% (m=51) | 0.0% (m=32) | 18.75% |

Coverage needed to hit a target risk ceiling (largest achievable coverage with risk <= target,
literal max over the curve, curve not assumed monotone):

| config | coverage @ risk<=5% | coverage @ risk<=2% |
|---|---|---|
| yolo26 | 53.1% | 42.2% |
| s1a/yolo11_baseline | 45.3% | 40.6% |
| yoloroi | 57.8% | 39.1% |
| v1nocw | 54.7% | 32.8% |
| v2bnocw | 57.8% | 54.7% |

**Deferral-to-human framing (the actual safety-relevant reading).** At 80% forced coverage (defer
the least-confident 20% of clips), selective risk drops from a 15.6-23.4% full-coverage error rate to
11.8-15.7% — a real, measurable reduction, but a much smaller one than the paper's own headline gap
would suggest is available: even deferring 20% of clips, best-case config still errs on ~12% of the
remaining ones. At 50% coverage the reduction is much larger (0-6.3% risk on the most-confident half) —
this is the genuinely strong result: **if a human reviews the least-confident half of clips, the
system's own decisions on the other half are near-perfect in 4 of 5 configs (v2bnocw: literally 0
errors on its most-confident 32 clips).**

Excess AURC (.04-.05) says the confidence ranking is doing real but imperfect work — an oracle
ranking (using ground truth, unavailable at deployment) would roughly halve AURC. This is a
diagnostic about ranking quality, not a coverage guarantee (contrast with section 1): it describes
these 64 clips' curve, it does not promise anything about held-out clips the way conformal does.

**Wording discipline.** None of this is "our system beats the paper's" — the paper reports zero
confidence scores, zero abstention option, and zero risk-coverage curve, so this is a one-sided
demonstration of an axis the paper cannot be evaluated on, not a comparison with a p-value. Frame in
the thesis as "additional evaluation lens the published system offers no evidence on," per brief 4j /
r5 guardrail 7.

Figures: `reports/w3-figures/risk_coverage_<config>.png` (model vs oracle per config) and
`risk_coverage_all_configs.png` (all 6 overlaid). Raw curve data (coverage/risk arrays are
regenerable from the .pt files + `eval_utils.selective_risk.risk_coverage_curve`, not itself dumped
to keep the JSON small — summary statistics only in `w3_results.json`).

---

## 3. Power analysis / required-n for non-inferiority

Formula (Chow, Shao & Wang 2003 ch.4, one-sided two-proportion NI, normal approx):
`n = (z_alpha + z_beta)^2 * [p1(1-p1)+p2(1-p2)] / delta^2`, z_alpha one-sided (alpha=.05 ->
z=1.6449), z_beta at power=.80 -> z=0.8416. Implemented independently in
`eval_utils/power_analysis.py::required_n_noninferiority`, scipy.stats.norm only (no statsmodels
needed — the formula is closed-form).

| delta (margin) | n per arm, p~=0.90 | n per arm, p~=0.9581 (paper's own rate) |
|---|---|---|
| 0.05 | **446** | **199** |
| 0.10 | **112** | **50** |
| 0.15 | **50** | **23** |
| 0.20 | **28** | **13** |

**Compared to r5's numbers (~445, ~111, ~50, ~28 at p=.90; ~199, ~50 at p=.958):** agreement to
within 1 unit everywhere. The 1-unit differences (446 vs 445, 112 vs 111, 23 vs r5's unlisted-but-
implied ~22) are `ceil()` — r5 presumably reported the raw (unceiled) value rounded to nearest
integer, we ceil to protect the nominal 80% power (a required-n that's rounded DOWN under-powers the
study). **Independently confirmed, no disagreement in substance** — recommend citing the ceiling
convention explicitly in the thesis so a reader checking the arithmetic doesn't flag a mismatch.

**Achievable margin at n=64** (invert the formula: smallest delta resolvable at n=64, alpha=.05,
power=.80): **13.19pp** at p~=0.90, **8.81pp** at p~=0.958 — matches r5's 13.2pp / 8.8pp to the
decimal. `test_achievable_margin_matches_r5_at_n64` locks this in.

**Recommendation for the thesis (benchmark-design finding):** a defensible safety-context NI margin
(5-15pp conventional) is only reachable at n=64 in the best case (p near the paper's own ~96%, margin
~15pp) — most of the conventional NI-margin range (5-10pp) requires 50-450 clips per arm depending on
the true accuracy level and chosen margin. **~111 per arm at delta=10pp / p=.90 is the single number
worth headlining** — it is an order of magnitude above the current dataset, and it is a concrete,
falsifiable target for a future data-collection round, not a vague "more data would help."

---

## 4. Fisher-z CI on the val/test correlations (RQ8, resolving power)

Using `eval_utils.power_analysis.fisher_z_ci(r, n)` on the three (r, N) pairs already computed and
audited in `plans/2026-09-05-thesis-final-push/reports/t1-objectives-audit.md` S1b (no raw
per-config table was re-derived here — this consumes the already-verified r/N pairs, per task scope):

| subset | r | N | 95% CI (Fisher-z) | spans zero? |
|---|---|---|---|---|
| all 46 round-2 configs | -0.247 | 46 | **[-0.501, +0.047]** | yes |
| 22 "legitimate" trained heads only | -0.136 | 22 | **[-0.527, +0.303]** | yes |
| prior thesis-process-review subset | -0.066 | 34 | **[-0.395, +0.278]** | yes |

**Reading.** Every subset's 95% CI spans zero AND has a lower bound around -0.40 to -0.53 — i.e. the
data are statistically consistent with anywhere from "moderately strong negative val/test
relationship" to "mild positive relationship." This converts the audit's qualitative "r is small,
negative, never significant" into a numbered claim: **the benchmark's val-set ranking signal, at
this N of configs, cannot be distinguished from zero, and the CI is wide enough that a moderately
strong true correlation in either direction cannot be ruled out either.** This is a measured
statement about resolving power, not a claim that no signal exists in principle — a larger config
sweep (or the required-n style argument) is needed to resolve it, mirroring section 3's logic
one level up (config selection instead of clip-level accuracy).

---

## 5. What this section does NOT claim (binding, brief 4j / r5 guardrail)

1. No claim of beating, matching, or being statistically indistinguishable from the paper's 62/64 —
   untouched by anything above.
2. Conformal's coverage number is a guarantee ABOUT OUR OWN MODEL's uncertainty quantification, not
   evidence of superiority over a system that reports no confidence scores at all.
3. Per-class conformal coverage is not reported anywhere (refused by code, not just by omission).
4. The 111-per-arm / 445-per-arm numbers are a benchmark-design recommendation (data-scale finding),
   not evidence that runs at n=64 are invalid or that a specific effect exists at that scale.
5. Fisher-z CI describes resolving power of the internal config sweep, not a new comparison to the
   paper (the paper released no per-config sweep to correlate against).
6. `logits_s1a.pt` / `logits_yolo11_baseline.pt` being bit-identical is disclosed, not hidden or
   silently deduplicated into "5 configs agree" language.

---

## Unresolved questions

1. Whether `s1a` and `yolo11_baseline` are *intentionally* the same run under two names (e.g. one is
   a symlink/copy of the other in the export pipeline) or an accidental overwrite — worth a 30-second
   check against `run_full_eval.py` / whatever script wrote these two files before citing both names
   separately anywhere else in the thesis.
2. The 50/50 calibration split fraction was my choice (not specified in the task) since only the
   64-clip test set has softmax vectors — no separate val-softmax file exists in
   `sweep_results/relation_tokens/`. `kfold_oof_softmax_yolo26app_.pt` (516 out-of-fold clips) IS
   available and could serve as a much larger, more stable calibration set for a DIFFERENT model
   family (the k-fold relation-transformer pipeline) if the thesis wants a tighter-CI conformal
   number elsewhere — not used here since it's a different pipeline than the 6 sweep configs.
3. required-n ceiling-vs-round convention (r5 raw vs my `np.ceil`) should be stated explicitly
   wherever both numbers appear in the same chapter, to avoid an apparent (but not real) discrepancy.
