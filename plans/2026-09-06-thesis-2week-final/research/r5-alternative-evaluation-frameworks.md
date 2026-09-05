# R5 — Alternative Evaluation Frameworks (metric-contract extension)

Scope: metrics that let the thesis prove value WITHOUT a head-to-head accuracy win at N=64 (structurally
closed, see `plans/2026-09-05-thesis-final-push/reports/t1-objectives-audit.md` §3b: even our_correct=64
gives McNemar best-case p=.50, Fisher p=.496, vs paper 62/64). Excludes (cite only, do not re-derive):
top-1/macro-F1/bal-acc/per-class P-R, kappa, MCC, confusion matrix, per-class FNR, declared cost matrix,
ROC/PR-AUC, threshold sweep, ECE+reliability, bootstrap 5000, Wilson/BCa CI, exact McNemar, Nadeau-Bengio
— these are `eval_utils/{metrics,paired_tests,bootstrap_ci,calibration,safety_metrics}.py`, being
implemented concurrently. All numbers below computed fresh (scipy 1.x / statsmodels 0.15, scratchpad
`/private/tmp/.../scratchpad/`) against the audited counts: paper 62/64=96.88%; ours blind 50/64=78.12%;
ours test-selected 55/64=85.94% (56/64=87.5% round-0/1 best, shown for context only, not headline).

---

## A. Master table

| # | Metric family | Formula (core) | Claim it enables | Feasible at n=64? | Cost (h) | Priority |
|---|---|---|---|---|---|---|
| 1 | TOST / non-inferiority (paired/unpaired proportions) | score test on Δp vs margin −δ (Miettinen-Nurminen) | "non-inferior at margin δ, p=..." OR "δ this small is not achievable, need n=X" | **Marginal-negative but quantifiable** — unpaired only (no per-clip opponent labels); NI fails at all defensible δ, but required-n IS a result | 2 | **P0** |
| 2 | Bayesian Beta-Binomial + ROPE | posterior Beta(a+k,b+n−k); P(θ1>θ2); P(\|Δ\|≤ROPE) | "P(practically equivalent)=X%", "P(ours better)=Y%" | **Yes** — works at any n, just wide credible intervals | 3 | **P0** |
| 3 | Effect size (Cohen's h, OR, RD, MDE) | h=2asin√p1−2asin√p2; OR=(ad)/(bc); RD=p1−p2 | quantify magnitude+CI where p-value is silent | **Yes**, cheap | 2 | **P0** |
| 4 | Selective prediction + conformal | risk-coverage curve, AURC; split-conformal quantile | "abstain-and-escalate is more deployable"; finite-sample coverage guarantee | **Yes for marginal; NO for per-class Mondrian** (rarest class n=8) | 5 | **P0** |
| 5 | Decision-curve / net benefit / cost curves | NB(pt)=TP/n−FP/n·pt/(1−pt) | "dominates over cost-ratio range [a,b]" | Yes, needs probability scores (already produced for ROC/PR) | 3 | **P1** |
| 6 | Friedman+Nemenyi, Wilcoxon+Holm | χ²_F; CD=q_α√(k(k+1)/6N) | turns 46-config sweep into evidence, not noise | Yes if per-fold scores per config exist | 4 | **P1** |
| 7 | Learning-curve extrapolation | acc=1−a·N^(−b), fit log-log | "need N≈X clips to reach 95.81%, CI[..]" | **Expensive** — needs retrain at ≥4 data fractions | 10+ | **P2** |
| 8 | Val/test resolving power (Fisher-z CI on r, MDE-as-resolving-power) | z=arctanh r; CI=tanh(z±z_crit·SE); resolving power = MDE at n | "this benchmark cannot rank models below ~15pp true gap" — formal, numbered | **Yes, nearly free** (r already computed in audit) | 1 | **P0** |
| 9 | Robustness (mCE-style, corruption+temporal) | CE_c=ΣE_{s,c}/ΣE_{s,c}^{baseline}; mCE=mean_c CE_c | "robust to X, degrades under Y, unmeasured by paper" | Yes but needs corruption pipeline + re-inference | 8 | **P2** |

---

## B. Five highest-value additions — full math + exact thesis sentence

### B1. Non-inferiority testing (RQ1) — P0

**Design constraint (state honestly first):** opponent's per-clip labels are NOT published (only
aggregate 62/64). A **paired** TOST (needs discordant pairs b,c) is impossible. Only the **unpaired**
two-independent-proportions non-inferiority test is computable. This is strictly less powerful than a
paired design — flag this explicitly in the thesis, don't let a reader assume it's the strong version.

**Formula — Miettinen-Nurminen score test for risk difference, one-sided NI:**
H0: p_ours − p_paper ≤ −δ  vs  H1: p_ours − p_paper > −δ
Score statistic uses the MLE of p1,p2 constrained to p1−p2=−δ (restricted MLE, closed-form quadratic).
`statsmodels.stats.proportion.test_proportions_2indep(count1, nobs1, count2, nobs2, value=-delta, method='score', compare='diff', alternative='larger')`.

**Computed** (ours vs paper 62/64, n=64 each):

| our k/64 | δ=0.10 p | δ=0.15 p | δ=0.20 p | δ needed for p=.05 |
|---|---|---|---|---|
| 50 (blind headline) | .946 | .750 | .415 | .290 |
| 54 | .692 | .318 | .092 | .220 |
| 55 (test-selected) | .576 | .219 | .053 | **.202** |
| 56 (round-0/1 best) | .448 | .140 | .029 | **.184** |

**Interpretation:** even our single best number (56/64) needs a **margin ≥18.4pp** to be declared
non-inferior at α=.05 — a margin larger than the entire span any safety-NI literature would call
"practically equivalent" (clinical/industrial NI margins are conventionally 5-15pp). **Non-inferiority
is NOT achievable at n=64 with the unpaired design at any defensible δ.** This itself is the honest
finding — do not force a positive NI claim.

**Choosing δ in a safety context (how to justify it, for the required-n calc below):** derive δ from the
declared cost matrix (already in `eval_utils/safety_metrics.py`): δ_max = the accuracy drop that keeps
expected cost non-worse than baseline given miss:false-alarm cost ratio C. If misses cost 10x false
alarms (contract §4c convention) and errors are ~evenly split miss/FA at baseline, a δ of 5pp translates
to a nontrivial expected-cost increase; conventionally δ=0.05-0.10 is the safety-context default absent
a formal cost-indifference derivation — state this convention explicitly rather than invent false
precision.

**Required n (the reportable result):** independent two-proportion NI sample size (per arm),
assuming true accuracy ≈90%, α=.05 one-sided, power=.80:
n = (z_α+z_β)² · [p₁(1−p₁)+p₂(1−p₂)] / δ²  (Chow, Shao & Wang 2003, ch.4)

| δ (margin) | n per arm (p≈0.90) | n per arm (p≈0.9581) |
|---|---|---|
| 0.05 | 445 | 199 |
| 0.10 | 111 | 50 |
| 0.15 | 50 | 22 |
| 0.20 (our own threshold) | 28 | 12 |

**Achievable margin AT n=64** (invert the formula, solve δ given n=64, power=.80): δ=**13.2pp** (p≈.90)
to **8.8pp** (p≈.958). This is narrower than our own required-δ-for-p=.05 above (18-20pp) because that
number is empirical (uses our actual observed k), while this one is the theoretical best-case power
curve — the gap between them (13pp achievable-by-design vs 18-20pp needed-in-practice) is itself
evidence the true effect is not small enough for n=64 to resolve either way.

**Exact thesis sentence:** *"Under an unpaired Miettinen-Nurminen non-inferiority test (the only design
possible without the opponent's per-clip labels), our best measured accuracy (56/64) would require a
margin of 18.4 percentage points to be declared non-inferior at α=.05 — larger than any margin
defensible in a safety context (conventional 5-15pp) — so non-inferiority CANNOT be claimed at n=64. A
conclusive non-inferiority claim at a defensible margin of 10pp, power 80%, would require ≈111 clips per
arm (≈50 at the paper's own ~96% accuracy level); this required-n is itself reported as a quantified
data-scale finding."*

**Python sketch:**
```python
from statsmodels.stats.proportion import test_proportions_2indep
res = test_proportions_2indep(k_ours, 64, 62, 64, value=-delta,
                                method='score', compare='diff', alternative='larger')
res.pvalue
```

---

### B2. Bayesian comparison (RQ2) — P0

**Model:** unpaired Beta-Binomial. Prior θ_i ~ Beta(a,b) (flat a=b=1, or Jeffreys a=b=0.5 — report both,
they barely move the answer at n=64). Posterior θ_i | k_i,n_i ~ Beta(a+k_i, b+n_i−k_i) (conjugate).

**Quantities:**
- P(θ_ours > θ_paper | data) — Monte Carlo (draw both posteriors, compare) or closed-form via the
  Beta-difference CDF (no simple closed form for two different Beta's in general — MC is standard).
- ROPE: P(|θ_ours − θ_paper| ≤ ε | data) for ε ∈ {0.05, 0.10} — the ROPE width is a modeling choice,
  justify via the same cost/margin logic as B1's δ.
- Bayes factor for equality (Gunel & Dickey 1974 for 2×2 tables): BF01 = marginal likelihood under
  θ1=θ2 (pooled Beta-Binomial) / marginal likelihood under θ1≠θ2 (independent Beta-Binomials). Both are
  closed-form via Beta functions:
  `m(k,n)=B(a+k,b+n-k)/B(a,b)`; independent model evidence = m(k1,n1)·m(k2,n2); pooled = m(k1+k2,n1+n2).

**Computed** (flat Beta(1,1) prior, MC N=2e6):

| our k/64 | P(ours>paper) | P(\|Δ\|≤5pp) | P(\|Δ\|≤10pp) | 95% credible interval on Δ |
|---|---|---|---|---|
| 50 (blind) | 0.06% | 0.9% | 7.1% | [−29.8, −7.3]pp |
| 55 (test-sel) | 1.5% | 12.7% | 46.4% | [−21.0, −1.0]pp |
| 56 (round0/1) | 2.7% | 19.6% | 58.8% | [−19.2, +0.2]pp |

Jeffreys prior moves these by <1.5pp absolute — conclusion is prior-robust at this n.

**Interpretation:** data give substantially more support to "practically tied within 10pp" (46-59%) than
to "we are better" (1.5-2.7%) — but 46-59% is NOT "probably equivalent," it is "can't rule out
inequality either." Report as a probabilistic hedge, not a win.

**Bayes factor** (equality vs difference, k1=56 vs k2=62, both n=64, flat prior): compute via the
formula above — expect BF01 in the 1-3 range (weak evidence for equality, Kass & Raftery 1995 scale) at
this n; do not oversell, "weak" is the honest label here, and report it alongside the ROPE number rather
than instead of it (a single BF without a scale is not self-interpreting in a thesis).

**ROPE via Bayesian correlated t-test for k-fold (Benavoli/Corani 2016)** — applicable ONLY to
*our own* internal cross-validated comparisons (e.g., pipeline P3 fold-level scores vs an ablation),
because it needs per-fold paired differences for BOTH systems, and the paper released no fold data.
Formula: differences d_i (fold i) modeled as Student-t with dof=k−1, mean adjusted for the
train/test-overlap correlation ρ≈n_test/n_train (Nadeau-Bengio correction, already in
`eval_utils/paired_tests.py::nadeau_bengio` — reuse its variance term inside a Bayesian posterior instead
of a frequentist t-test): posterior mean = mean(d), posterior variance = corrected-t variance; then
compute P(mean∈ROPE), P(mean>0), P(mean<0) from the resulting t-posterior. Package: `baycomp` (Python,
Benavoli et al.'s own library, function `baycomp.two_on_single` for correlated-t comparison) — cite
it explicitly; else replicate manually with `scipy.stats.t`.

**Exact thesis sentence:** *"Under a Beta(1,1) posterior over clip-level accuracy, P(our accuracy ≥ the
published 62/64 | data) = 1.5% (test-selected 55/64), but P(the two are practically equivalent within a
10-percentage-point ROPE | data) = 46.4% — i.e. the evidence favors 'statistically indistinguishable'
far more than it favors either 'we win' or 'we lose decisively', consistent with the frequentist
non-significance finding but expressed as a calibrated probability rather than a failure to reject."*

**Python sketch:**
```python
import numpy as np
rng = np.random.default_rng(0)
post_ours  = rng.beta(1+k1, 1+n1-k1, 2_000_000)
post_paper = rng.beta(1+62, 1+2,     2_000_000)
p_superior = (post_ours > post_paper).mean()
diff = post_ours - post_paper
p_rope = (np.abs(diff) <= 0.10).mean()
np.percentile(diff, [2.5, 97.5])          # 95% credible interval
# Bayes factor (Gunel-Dickey), flat prior:
from scipy.special import betaln
def log_m(k, n, a=1, b=1): return betaln(a+k, b+n-k) - betaln(a, b)
log_bf01 = (log_m(k1+62, n1+64) ) - (log_m(k1, n1) + log_m(62, 64))
```

---

### B3. Effect size table (RQ3) — P0

**Formulas:**
- Cohen's h (arcsine, variance-stabilizing, right tool for two proportions): h = 2·asin(√p1) − 2·asin(√p2). |h|<0.2 small, 0.2-0.5 medium, ≥0.8 large (Cohen 1988).
- Odds ratio: OR=(k1·(n2−k2))/((n1−k1)·k2); log-OR SE=√(1/a+1/b+1/c+1/d); Wald CI on log scale, exponentiate.
- Risk difference: RD=p1−p2; Wald SE=√(p1(1−p1)/n1+p2(1−p2)/n2).
- Minimum detectable effect (MDE) at n=64, two-sided α=.05, power=.80: δ=√((z_{.975}+z_{.80})²·2p(1−p)/n).

**Computed (n1=n2=64, vs paper 62/64):**

| our k/64 | Cohen's h | \|h\| label | OR [95% CI] | RD pp [95% CI] |
|---|---|---|---|---|
| 50 | −0.618 | large | 0.115 [0.025, 0.531] | −18.8 [−29.7, −7.8] |
| 54 | −0.457 | medium | 0.174 [0.037, 0.830] | −12.5 [−22.4, −2.6] |
| 55 | −0.413 | medium | 0.197 [0.041, 0.952] | −10.9 [−20.5, −1.4] |
| 56 | −0.367 | medium | 0.226 [0.046, 1.108] | −9.4 [−18.5, −0.2] |

MDE at n=64 (2-sided): **14.9pp** (p≈.90 baseline) to **9.9pp** (p≈.958 baseline).

**Note the OR-vs-RD disagreement at k=56**: OR's 95% CI just crosses 1 (upper 1.108, "no sig. odds
difference") while RD's CI just excludes 0 (upper −0.2pp, "sig. difference") — an artifact of different
variance-approximation scales at small n; report BOTH and flag the disagreement as a small-sample
symptom, not resolve it by picking whichever supports the preferred conclusion.

**Exact thesis sentence:** *"The accuracy gap between our best measured result (56/64) and the published
62/64 has Cohen's h=−0.37 (conventionally 'medium'), odds ratio 0.23 [95% CI 0.05, 1.11], and risk
difference −9.4pp [95% CI −18.5, −0.2] — a real but imprecisely-estimated, medium-magnitude gap, neither
'no difference' (as a bare non-significant p-value might imply) nor a large practical difference."*

**Python sketch:**
```python
import numpy as np
h = 2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p2))
# OR/RD: use statsmodels.stats.contingency_tables.Table2x2([[a,b],[c,d]]).oddsratio_confint()
from statsmodels.stats.contingency_tables import Table2x2
t = Table2x2([[k1, n1-k1], [62, 2]])
t.oddsratio, t.oddsratio_confint()
t.riskdiff if hasattr(t, 'riskdiff') else None  # else compute RD manually as above
```

---

### B4. Selective prediction + conformal prediction (RQ4) — P0, single strongest addition

**Risk-coverage curve:** rank n test clips by model confidence (max softmax or calibrated score) s_i
descending. At coverage c=m/n (keep top-m most confident), selective risk r(c) = (1/m)·Σ_{i=1}^m loss_i
(0/1 loss on the m most-confident predictions only). AURC = ∫₀¹ r(c) dc ≈ Riemann sum over the n
achievable coverage points (El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017, "Selective Classification
for Deep Neural Networks"). **Excess-AURC** = AURC − AURC*, AURC* = the oracle curve (rank by
correctness, not confidence) — isolates how much of the risk-coverage gap is a confidence-ranking
failure vs an irreducible-error floor.

**Report:** selective risk at fixed coverage (e.g., risk@80% coverage — "if we auto-decide the 80% most
confident clips and defer the rest to a human"), and coverage at fixed risk (e.g., "coverage needed to
guarantee ≤2% risk"). This is the **deferral-to-human framing**: an industrial safety system that
abstains on ambiguous clips and escalates is MORE deployable, not less — the paper offers zero evidence
on this axis because it never reports confidence scores or an abstain option.

**Conformal prediction (split-conformal, Vovk, Gammerman & Shafer 2005; Angelopoulos & Bates 2023
tutorial arXiv:2107.07511):**
1. Split cal/test (or reuse val=83 as calibration, test=64 as evaluation — do NOT recalibrate on test).
2. Nonconformity score α_i = 1 − f̂(x_i)_{y_i} (softmax miscoverage score), or APS score (Romano,
   Sesia & Candès 2020, arXiv:2009.14193): cumulative softmax mass to include true label rank.
3. q̂ = the ⌈(n_cal+1)(1−α)⌉/n_cal empirical quantile of {α_i} on calibration set.
4. Prediction set C(x) = {y : f̂(x)_y ≥ 1−q̂} (softmax) or the APS/RAPS analog (RAPS adds a size
   regularizer, Angelopoulos et al. 2021, penalizes large sets — use for the 8-class detector-token
   output if brought into scope, less needed for the 4-class clip head).
5. **Finite-sample coverage guarantee (exact, distribution-free):** P(Y∈C(X)) ≥ 1−α EXACTLY for any n,
   any distribution, as long as calibration/test are exchangeable (Vovk et al. 2005, Lemma 1). This
   holds at n=64 — conformal is the one framework in this whole report whose validity is NOT an
   asymptotic approximation.

**n=64/83 caveat (state explicitly):** the guarantee's VALIDITY is exact at any n; its RESOLUTION
(achievable α granularity) is coarse: with n_cal=83, the empirical quantile only takes ~84 distinct
values → achievable coverage levels step in ~1.2pp increments, fine for one α=0.10 target. But
**class-conditional (Mondrian) conformal needs a per-class calibration set**: our smallest class
(CarryOverload, n=8 in the test loader; ~similarly small in val) gives quantile granularity of only
1/9≈11pp steps — a per-class 90% coverage guarantee on 8 examples is not meaningfully verifiable
(you'd need to add ≥1 clip's worth of nonconformity mass to move the guarantee at all). **Report ONLY
marginal (whole-test) conformal coverage as a guarantee; report per-class prediction-set size as a
descriptive (not guaranteed) diagnostic.**

**Prediction-set size as a metric:** average |C(x)| across test clips — smaller is better at fixed
coverage; a well-calibrated 4-class head should show avg size ≈1.1-1.3 at 90% target coverage if
genuinely confident, size→4 (uninformative) if the model can't discriminate. This is a genuinely new
axis vs the paper (they report zero uncertainty quantification).

**Exact thesis sentence:** *"At a target marginal coverage of 90%, split-conformal calibration on the
val set yields empirical test coverage of X% [exact finite-sample guarantee ≥90% by construction,
Vovk et al. 2005] with average prediction-set size Y; at 80% forced coverage (defer the least-confident
20% of clips to human review), selective risk drops to Z% (from W% at full coverage) — demonstrating a
viable deferral operating point that the published system, which reports no confidence scores, cannot
offer."* [X,Y,Z,W to be filled by implementer from real softmax outputs — not fabricable now, no
model-output file was read for this report; this is a formula spec, not a result].

**Python sketch:**
```python
import numpy as np
# risk-coverage / AURC
conf = scores.max(axis=1)           # (n,) confidence per clip
correct = (preds == y_true).astype(float)
order = np.argsort(-conf)
cum_correct = np.cumsum(correct[order])
coverage = np.arange(1, len(y_true)+1) / len(y_true)
risk = 1 - cum_correct / np.arange(1, len(y_true)+1)
aurc = np.trapz(risk, coverage)

# split-conformal (softmax score), calibrate on val, evaluate on test
cal_scores = 1 - val_probs[np.arange(len(y_val)), y_val]
alpha = 0.10
n_cal = len(y_val)
q = np.quantile(cal_scores, np.ceil((n_cal+1)*(1-alpha))/n_cal, method='higher')
pred_sets = [np.where(test_probs[i] >= 1-q)[0] for i in range(len(y_test))]
coverage_test = np.mean([y_test[i] in pred_sets[i] for i in range(len(y_test))])
avg_set_size = np.mean([len(s) for s in pred_sets])
```
No scipy/statsmodels function exists for conformal — implement per above (standard, ~30 lines); for
APS/RAPS use `crepes` or `mapie` (pip installable, `MapieClassifier(method='raps')` from `mapie`
package) if an off-the-shelf implementation is preferred over the manual one.

---

## C. What still CANNOT be claimed (guardrail — read before writing Ch4/Ch6)

1. **"We beat/match the paper's headline 62/64 in a statistically significant sense"** — closed by T1
   audit §3b, unaffected by anything in this report (structural: paper's 2 errors cap discordant
   advantage). No metric reframing changes this arithmetic fact.
2. **"We are non-inferior to the paper"** (B1) — computed and FAILED at every defensible margin (need
   ≥18.4pp, unreasonable for a safety NI claim). Do not state non-inferiority as achieved; state the
   required-n instead.
3. **"Bayesian analysis proves practical equivalence"** — P(ROPE)=46-59% is "more support for tied than
   for different," not "proven equivalent." Never write P>50% as if it were a p<.05-style threshold —
   there is no accepted "significance" cutoff for a probability like this; report the number, not a
   verdict.
4. **Any per-class conformal coverage guarantee** — smallest classes (n=8-11) cannot support Mondrian
   guarantees at this N. Marginal only.
5. **Learning-curve extrapolation (RQ7) as a precise number** — with only one dataset size regime
   effectively available (452 train clips, no systematically varied fractions yet), a power-law fit has
   enormous extrapolation uncertainty; report the fitted curve with WIDE CI and frame as "order of
   magnitude," not a precise clip count, unless multi-fraction training runs are actually done (cost
   10h+, P2, may not fit the 2-week window per brief.md schedule).
6. **Friedman/Nemenyi as "our method is best"** — a significant Friedman test across the 46-config sweep
   shows the configs differ from each other; it says nothing about vs-the-paper (paper is one config, no
   per-fold data). Useful only for internal ablation legitimacy, not for the headline claim.
7. **Robustness numbers (RQ9) as "safer than the paper in deployment"** — paper reports zero robustness
   data, so this is a one-sided demonstration ("we measured X, they measured nothing"), not a comparison
   with a p-value or effect size. Frame as "additional evidence axis," not "we are more robust than
   them" (unmeasurable for them).
8. **Any claim from B1-B4 as replacing the pre-registered headline (blind fusion 50/64, per brief.md
   §5 / audit §5.4)** — these are ADDITIONAL evaluation lenses on the SAME frozen numbers, not new test
   touches. No new model selection or fusion re-run is licensed by anything in this report.

---

## D. Implementation order & eval_utils/ dependencies

All B1-B4 (+B5/B6 if time) consume **outputs of** `eval_utils/`, they do not modify it. Suggested new
module `eval_utils_ext/` (or a clearly separate file, per file-ownership rule — this report's author does
not touch `eval_utils/`) — implementer's call on exact filename, kebab-case, <200 lines/file per
dev-rules.

| Step | Module | Depends on (already in eval_utils/) | Depends on (new data needed) |
|---|---|---|---|
| 1 | `resolving_power.py` (RQ8: Fisher-z CI, MDE) | none (pure stats on existing val/test macro pairs) | the 46-config val/test macro table (audit §1b, already in scratchpad `rows.json` — regenerate, read-only) |
| 2 | `effect_sizes.py` (RQ3) | `metrics.py::top1` output (k,n pairs) | none — just counts |
| 3 | `noninferiority.py` (RQ1, TOST) | `metrics.py`, needs `statsmodels` (not yet a project dep — add to requirements) | none — just counts |
| 4 | `bayes_compare.py` (RQ2) | `metrics.py`; reuses `paired_tests.py::nadeau_bengio`'s variance term for the correlated-t/ROPE k-fold variant | k-fold per-fold scores (pipeline P3, `run_kfold_relation.py` output) for the correlated-t part; counts-only for the Beta-Binomial part |
| 5 | `selective_conformal.py` (RQ4) | none direct, but needs **per-clip softmax probability vectors** (not just argmax) — confirm the inference pipeline saves these; if only hard labels are saved, this blocks until a re-run adds `probs` to the output | val (calibration) + test (evaluation) softmax arrays |
| 6 | `decision_curve.py` (RQ5) | `safety_metrics.py::roc_pr_auc` scores | same probability vectors as step 5 |
| 7 | `multi_model_stats.py` (RQ6) | none direct | per-fold, per-config score matrix (46 configs × k folds) — assemble from `sweep_results/r2/*.pt` |
| 8 | learning curve (RQ7) — **do not implement without new training runs**, flag as stretch | — | multi-fraction training runs (10h+, competes with A1-A3 lane) |
| 9 | robustness (RQ9) — **stretch**, needs corruption pipeline (imgaug/albumentations for blur/noise/JPEG; custom for frame-drop) | none direct | re-inference on corrupted copies of the 64 test clips |

Add to `requirements.txt`/pinned env: `statsmodels` (not currently a dependency — verified absent from
project venv until this report's research installed it in scratchpad only; installer must add it
properly, pinned version, not ad hoc).

---

## Unresolved questions

1. Whether per-clip **softmax probability vectors** (not just argmax predictions) are actually saved by
   the current inference/eval pipeline — B4 (conformal) and B5 (decision-curve) are blocked without them.
   Whoever owns eval_utils/ implementation should confirm and expose this before day 6 (A4 freeze).
2. Whether pipeline P3's k-fold run (`run_kfold_relation.py`) stores **per-fold, per-config** scores in a
   form usable for RQ6 (Friedman/Nemenyi) and the Bayesian correlated-t (RQ2 extension) — needed as a
   matrix, not just the pooled 516-clip headline number currently reported.
3. δ/ROPE width (5pp vs 10pp) is a judgment call justified here by convention, not a formally derived
   cost-indifference point — if the cost matix (`eval_utils/safety_metrics.py::cost_sensitive_score`) is
   finalized with concrete miss:FA numbers, δ should be re-derived from it for consistency rather than
   left as a round-number convention.
4. Whether learning-curve (RQ7) and robustness (RQ9) fit the 14-day window at all — both P2/stretch by
   design; recommend explicit go/no-go check against brief.md's day-7 buffer slot before committing.
