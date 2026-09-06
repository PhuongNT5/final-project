# Alternative metrics & statistics beyond the headline number

Scope: what's MISSING from `unsafenet/metrics.py`+`metrics_stats.py` (top1, macro-recall,
per-class P/R/F1, MCC, kappa, ROC/PR-AUC, ECE/MCE/Brier/NLL, Clopper-Pearson, Wilson, bootstrap,
exact+bounded McNemar, permutation test — all already implemented per t5a). Also builds directly
on t1's finding: paper N=64 fixed forever, 2 discordant-pair floor on McNemar. That same frozen-N
fact reappears below as a **variance floor on equivalence testing** — this is the connective
thread across RQ1.

## RQ1 — Equivalence / non-inferiority (TOST)

**Recipe.** TOST = two one-sided z/t-tests. H0_lower: diff ≤ -margin. H0_upper: diff ≥ +margin.
Reject both at level α ⇒ conclude |diff| < margin at confidence 1-2α (equivalently: the
(1-2α)*100% CI, e.g. 90% CI for α=.05, lies entirely inside [-margin, margin]). For two
proportions (our accuracy p_ours,n_ours vs paper p_paper,n_paper=64, **unpaired** — we lack
per-clip labels, same limitation as McNemar in t5a §6):
SE = sqrt(p_ours(1-p_ours)/n_ours + p_paper(1-p_paper)/64); CI = diff ± z_{1-α}*SE (one-sided
z, e.g. z=1.645 for α=.05); equivalence holds if CI ⊂ [-margin,+margin]. [Schuirmann 1987;
Julious 2004 "Sample Sizes for Clinical Trials" ch.10 for the proportions case; Chow/Shao/Wang
2003 "Sample Size Calculations in Clinical Research".] Modern ML-specific treatment: Koobs &
Koning, "Equivalence testing with data-dependent and post-hoc equivalence margins," arXiv
2603.16213 (2025-26) — argues margin choice is usually done backwards (picked after seeing
results) and proposes disciplined data-dependent/pre-registered margin selection; directly
citable for "how do we defend our margin," not just "how do we run the test."

**Margin choice.** No universal ML convention. Concrete anchors found: Benavoli/Corani/Demšar
use ROPE=[-1%,+1%] accuracy difference as "practically equivalent" for generic classifier
comparison (arbitrary but widely cited, JMLR 2017). Clinical-trial literature (EMA guideline,
BJA Education) insists margin be picked on domain grounds *before* seeing data, independent of
power considerations — directly transferable rule: pick margin from what a safety reviewer
would call "operationally indistinguishable" (e.g. "differs by less than 1 clip in the smallest
class," 1/8=12.5%, or "less than half a McNemar-detectable effect from 3a," ~5-6pp), not from
whatever number makes the test pass.

**Original calculation for this project (not from a single citable source — combines the
standard TOST-for-two-proportions formula above with t1's fixed-paper-N fact).** Because
n_paper=64 is frozen forever (no way to re-run the paper's model on more clips), it sets a hard
floor on the achievable CI half-width no matter how large n_ours grows:
floor = z_{.05}*sqrt(p_paper(1-p_paper)/64). Computed floors (α=.05, one-sided z=1.645):
- vs full headline (p_paper=.9688, top1): floor = **3.6pp**
- vs no-reduction ablation (p_paper=.8906, top1): floor = **6.4pp**
- vs no-reduction ablation (p_paper=.8578, macro): floor = **5.8pp**

i.e. **even with n_ours → ∞, the TOST-equivalence CI can never be narrower than ~4-6pp around
the true difference**, because the paper's own uncertainty (64 clips) doesn't shrink. Sample
size needed on OUR side to approach that floor (standard two-proportion TOST sample-size
formula, n≈(z_α+z_β)²·Σp(1-p)/margin², true-diff-under-alt=0, p≈0.90 both sides, α=.05 one-sided,
power=.80): **margin=10pp → n≈111/arm; margin=5pp → n≈445/arm; margin=3pp → n≈1237/arm;
margin=2pp → n≈2782/arm** (formula-derived, not independently validated against a canonical ML
example — flagged as an assumption). These numbers land almost exactly on our existing k-fold
protocol scale (N=516, fair-headtohead, t1 §1d) — i.e. **TOST becomes numerically plausible only
around the N=516 k-fold scale for a ~5-7pp margin, never at N=64.**

**Behaviour at N=64:** CI half-width computed directly (top1, no-reduction comparator,
n_ours=n_paper=64) = **9.6pp** — worse than the asymptotic floor (6.4pp), confirms N=64 is too
small even against the *closer* comparator. At N=516 (our side only, paper still frozen at 64):
half-width drops to **6.9pp**, i.e. 90% of the way to the theoretical floor already — diminishing
returns past N≈500-800 on our side, because the floor is paper-side-limited, not us-side-limited.

**Two different equivalence targets, two different verdicts:**
- vs full headline (96.88/95.81): point gap is 11-19pp (our best legitimate numbers) — **larger
  than even the asymptotic floor**, so TOST cannot succeed regardless of N; the gap must
  actually close (better model), not be argued away by more test data. Matches t1 (a) "not
  achievable."
- vs the paper's own no-reduction ablation (89.06/85.78 — the paper's ConvLSTM-only row without
  their bbox-reduction trick): our test-selected macro (85.86%) is **0.08pp** from theirs — inside
  the floor even at N=64 in principle, IF this were the pre-registered blind number (it isn't;
  see t5a protocol on test-selected vs blind). Blind macro (80.81%) gap to no-reduction row is
  4.97pp — also inside a defensible 5-6pp margin. **Top-1 gaps are less favorable** (blind
  78.12% vs 89.06% = 10.9pp, outside any defensible margin) — so a TOST equivalence claim, if
  pursued, should target **macro-recall vs the no-reduction ablation specifically**, using the
  blind number, not top-1 and not the full headline.

**VERDICT (asked explicitly):** TOST is **not viable at N=64 as a standalone frequentist test** —
under-powered by a wide margin against any defensible margin, and structurally capped by the
paper's frozen N=64 no matter how large our side grows. It becomes *numerically plausible* only
if (1) the cross-validation/k-fold protocol (N≈500+) is used for OUR side, (2) the comparator is
the paper's no-reduction ablation (not the full headline — that gap is unbridgeable by definition
of the ceiling already proven in t1 §3b), and (3) the metric is macro-recall (not top-1, where the
gap is bigger). Even then it is a **secondary/supporting** result, not primary, because the
margin choice will look chosen-after-the-fact unless pre-registered now, before T4's final numbers
land (Koobs & Koning's whole point). Recommend: pre-register a 6pp macro-recall margin against the
no-reduction row NOW (this document), so it isn't post-hoc.

## RQ2 — Bayesian alternatives (the N=64-friendly path)

**Beta-Binomial posterior over one classifier's accuracy.** Prior Beta(a,b) (a=b=1 uniform, or
Jeffreys a=b=0.5); posterior Beta(a+correct, b+wrong) given correct/wrong counts. At n=64 this
is a real distribution, not a point estimate — degrades gracefully, no "insufficient power"
failure mode the way NHST does. [Standard conjugate-prior result; used generically in ML eval,
e.g. Goutte & Gaussier 2005 for F-score, though beta-binomial-for-accuracy itself is textbook
Bayesian stats, not one paper's invention.]

**Posterior over the DIFFERENCE, single test set, unpaired (our situation exactly — no per-clip
correspondence to the paper).** Draw p_ours ~ Beta(1+c_ours, 1+w_ours), p_paper ~
Beta(1+62,1+2) independently (independence is the honest assumption given we cannot pair clips);
Monte Carlo N=100k draws of Δ=p_ours-p_paper; report P(Δ>0), P(|Δ|<ROPE), and the 95% credible
interval of Δ directly — no p-value, no arbitrary α. This is the single most useful addition:
it works at n=64, requires no minimum sample size to be "valid" (unlike TOST), and directly
answers "how likely is it we're worse, and by how much" rather than "can we reject a null."

**ROPE (region of practical equivalence).** Benavoli/Corani/Demšar/Zaffalon, "Time for a change:
a tutorial for comparing multiple classifiers through Bayesian analysis," JMLR 18 (2017) —
canonical citation. ROPE=[-1%,+1%] is their example value for generic accuracy comparison;
recommend widening to match RQ1's margin logic (e.g. ±5-6pp for macro vs no-reduction row) rather
than blindly importing their 1% (which was chosen for a different, larger-N benchmark context).
Decision rule: HPD interval entirely inside ROPE ⇒ practically equivalent; entirely outside ⇒
practically different; straddling ⇒ "undecided," reported honestly as such — this third outcome
is itself useful (avoids the false dichotomy NHST forces).

**Bayesian correlated t-test (for repeated cross-validation, NOT for the N=64 single test).**
Same JMLR 2017 paper. Models k-fold CV replicate differences as correlated Gaussian
(correlation ρ=1/k, corrects the well-known non-independence of CV folds that plain t-tests
ignore — Nadeau & Bengio 2003 correction, already partially reflected in t1 §1d's "Nadeau-Bengio
±3pp CI" note for the fair-headtohead k-fold protocol). Posterior over mean difference is a
non-standard (shifted/scaled Student-t) distribution; `bayesiantests` Python package
(Benavoli et al.) implements `correlated_ttest` directly — reusable off-the-shelf for the
N=516 k-fold data already collected (pipeline O). This is the correct Bayesian counterpart to
the k-fold TOST discussed in RQ1, and should be run on the SAME k-fold data, reported alongside
the frequentist CI, not instead of it.

**Recipe summary (concrete, ready to implement in `metrics_stats.py`):**
```
beta_ci(k, n, a=1, b=1) -> posterior mean/HPD via scipy.stats.beta
posterior_diff(k_a, n_a, k_b, n_b, rope, n_mc=100_000) -> P(diff>0), P(in ROPE), 95% CI
```
Both are ~20 lines, no new dependency (scipy.stats.beta already imported in the codebase).

## RQ3 — Cost-sensitive / safety-critical metrics

**Cost matrix + expected cost.** Assign cost C_FN (miss an unsafe behaviour) ≫ C_FP (false
alarm); expected cost = Σ_ij C_ij * confusion[i,j] / N. Threshold selection: optimal cut-off for
a calibrated posterior is C_FP/(C_FP+C_FN) [textbook cost-sensitive learning, e.g.
Elkan 2001 "The Foundations of Cost-Sensitive Learning"]. Practically: pick 2-3 cost ratios
(e.g. 5:1, 10:1, 20:1 miss:false-alarm) and report expected cost per ratio — a sensitivity table,
not one number, since the true ratio is a policy choice, not a measured quantity.

**FNR@FPR / miss-rate-at-fixed-false-alarm.** Standard in biometrics/surveillance: report the
miss rate (1-recall) for the unsafe classes at a FIXED, operationally-meaningful false-alarm
rate (e.g. FPR=1%, 5%, 10% — "how often does the system cry wolf if we insist on catching X% of
real violations"), read directly off the already-implemented per-class ROC curve
(`roc_pr_auc_ovr` — the curve exists, this metric is just a specific point read off it, near-zero
new code). This is a fully citable, standard safety/surveillance-literature metric (used broadly
in face-recognition/biometrics eval, e.g. NIST FRVT reports FNMR@FMR).

**Neyman-Pearson classification.** Formalizes "constrain Type-I (false-alarm) error, minimize
Type-II (miss) error" as the objective (rather than minimizing total error) — the right framing
for a safety system where false alarms are tolerable but misses are not. Tony Cai & Xin Tong,
"A Neyman-Pearson Approach to Statistical Learning" (JMLR 2013), and follow-ups (e.g. distribution-
free NP rates, arXiv 2402.09560, 2024) give the formal apparatus; practically this is "report
recall at FPR≤α for each α of interest" (same as FNR@FPR above) — cite NP theory to justify WHY
that's the right axis to report on for this domain, not just report it ad hoc.

**Decision curve analysis / net benefit.** Net benefit(threshold t) = TP/N - FP/N *
(t/(1-t)); plot across t, compare against "treat/flag all" and "flag none" baselines
[Vickers & Elkin 2006, origin in clinical prediction models; Wikipedia "Decision curve analysis"
for the formula]. Directly portable: swap "treat patient" for "flag clip for human review."
Gives a single plot showing the operating range where the model adds value over blanket
alerting or no alerting — a strong, visual, safety-framed argument that plain accuracy cannot
make, and the paper does not report anything like it.

**Where this can show superiority:** because the paper reports only top-1/macro at their default
argmax threshold, ANY of the above (cost-weighted expected cost, FNR@FPR, net-benefit curve) is
information the paper simply does not have — not a claim of beating their number, but a claim of
reporting a materially more relevant number for the stated application (worker safety) that
survives even though top-1 parity cannot be shown. This is the strongest "lead, don't follow"
opportunity in the whole list.

## RQ4 — Effect size + uncertainty reporting instead of p-values

**Cohen's h** (arcsine-transformed distance between two proportions): h = 2·arcsin(√p1) -
2·arcsin(√p2); |h| 0.2/0.5/0.8 = small/medium/large [Cohen 1988; Wikipedia "Cohen's h"]. Unlike
raw pp-difference, h is a variance-stabilized effect size usable directly in TOST/power formulas
(sample-size tables keyed on h exist, e.g. `pwr::ES.h` in R) — report alongside the raw pp gap,
not instead of it, since raw pp is what a reader intuitively understands and h is what a power
calc needs.

**Common-language effect size (CLES) / probability of superiority.** P(random correct clip from
A also correct under B) — intuitive complement to h, e.g. "if you sampled one clip our model got
right, there's an X% chance the paper's model also got it right" — reads well in a defence.

**"Report intervals, not p-values" movement.** Dror, Baumer, Shlomov, Reichart, "The Hitchhiker's
Guide to Testing Statistical Significance in NLP," ACL 2018 (aclanthology.org/P18-1128) — surveyed
ACL/TACL 2017 papers, found significance testing "ignored or misused" as the NORM; gives a
decision protocol for which test to use per metric/setup (directly reusable structure: their
protocol → McNemar for paired single-test-set classification comparisons, matches what's already
implemented). Bouthillier et al. 2021 (RQ5 below) makes the parallel claim for benchmark variance
broadly: report the full distribution of outcomes across variance sources, not a single seed's
p-value. Both support: **at N=64, the honest move is a full report — point estimate + CI (already
done) + effect size (Cohen's h, new) + Bayesian posterior (RQ2, new) — deliberately NOT reduced
to a single pass/fail p-value**, because p-values at this N have almost no resolving power
(t1 §3a/3b already proves this quantitatively for McNemar).

## RQ5 — Variance decomposition; validation-set unreliability at small N

**Bouthillier, Delaunay, et al., "Accounting for Variance in Machine Learning Benchmarks,"
MLSys 2021 (arXiv 2103.03098).** Decomposes benchmark variance into seed/init variance,
data-order variance, hyperparameter-search variance, and **data-split variance — found to be the
LARGEST single source**, larger than seed variance. Directly citable for our val↔test
anti-correlation finding (t1 §1b, r=-0.07 to -0.25): if split variance dominates seed variance in
general, a val set carved from a small 83-clip pool inheriting a fixed 64-clip test split is
exactly the regime where val-based model selection is expected to be unreliable — this paper is
the load-bearing citation for "this isn't just our bad luck, it's a documented property of small
benchmarks." Their prescription: report over MULTIPLE splits (not one), which is what the
existing k-fold/LCO protocol (fair-headtohead, N=516) already does — cite Bouthillier to justify
why that protocol, not the single-split unsafenet/ round, is the methodologically stronger
evidence source (already t1's own conclusion, now literature-grounded).

**Isaksson, Wallman, Göransson, Gustafsson, "Cross-validation and bootstrapping are unreliable in
small sample classification," Pattern Recognition Letters (2008)** — directly on-point older
result: at small n, CV/bootstrap accuracy estimates have such high variance that model ranking
by them is little better than random; exactly our val-macro-can't-pick-the-test-winner finding,
21 years earlier and in a different subfield (bioinformatics) — strengthens the claim that this
is a known general phenomenon, not specific to video/behaviour classification.

**"Crossing the Validation Crisis" (arXiv 2606.12552, 2026)** — argues repeated/nested CV
substantially reduces (but does not eliminate) this benchmarking variance; supportive citation
for why the plan's k-fold/LCO protocol is the right mitigation, not a cure — consistent with
RQ1's finding that variance shrinks toward a floor, not to zero.

**No paper found reporting the exact "val anti-correlates with test" sign** (r<0) as a named
phenomenon — the closest literature gives magnitude/unreliability, not sign. Flagging this as
possibly a genuinely novel empirical note worth stating carefully as "consistent with, but more
extreme than, the split-variance-dominance literature" rather than "directly predicted by it."

## RQ6 — Temporal/video metrics (paper does not report any of these — lead here)

**Per-frame vs per-clip agreement.** If the pipeline scores per-frame before aggregating to a
clip decision, report frame-level accuracy alongside clip-level, and the gap between them —
large gaps indicate the temporal aggregator (not the frame model) is doing most of the work,
or masking frame-level noise. Simple, no citation needed beyond standard practice in temporal
action recognition (frame-AP vs video-AP framing appears throughout the online-action-detection
survey literature, e.g. arXiv 2107.05140 "Review of Video Predictive Understanding").

**Temporal stability / "flicker" of predictions.** Borrowed from video-quality literature
(flicker = mean absolute frame-to-frame change in output; "Evaluating the Effect of Compression
on Video Temporal Consistency," arXiv 2605.18378, 2026, gives the formal metric definition for
continuous signals) — adapted to discrete per-frame class predictions as: fraction of
consecutive-frame pairs where the argmax class changes ("switch rate"), or entropy of the
run-length distribution of stable segments. Useful diagnostic for whether errors are isolated
frame glitches vs sustained misclassification — the paper reports neither.

**Early-detection latency.** Standard formal setup: sample predictions at k=10%,20%,...,100% of
clip length, report accuracy-vs-observed-fraction curve; report the minimum fraction (or frame
count) at which accuracy plateaus/reaches a target (e.g. "correct after seeing first 30% of
clip") ["Online human action detection and anticipation in videos: A survey," ScienceDirect 2022;
methodology used broadly in early-action-prediction literature, e.g. arXiv 1904.13085]. **This is
explicitly flagged by the research brief as the strongest under-reported opportunity: a
real-time safety system's value is largely about HOW SOON it flags an unsafe behaviour, and the
paper (offline clip classification) reports zero timing/latency-to-decision numbers of any kind.**
Concrete recipe for this codebase: for each test clip, run the temporal head on the first k
frames (k=1..T), find the first k at which the running prediction locks onto the final (correct)
answer and stays there; report median/p90 "frames-to-lock" and "seconds-to-lock" (given known
fps) per class and overall.

## RQ7 — Reliability/deployment metrics

Already measured per t1 §1a-1d/M4 benchmarks: p50 latency 26.8-28.8ms (CPU-FP32, ONNX). Missing
pieces per current best practice (edge-AI benchmarking guides, 2025-26 sources): report the full
**p50/p95/p99 latency distribution** from ≥1000 real on-device inferences (not just p50 — tail
latency is what matters for a safety system that must never silently stall), **throughput at the
target latency bound** (not raw max throughput — "how many clips/sec at p99<X ms," per
DigitalOcean/Mirantis-style guidance), model size (params + on-disk size, already knowable from
existing checkpoints), and — flagged as a known caveat already in t1 §4(c) — do NOT claim Jetson
numbers that are literature citations, not measured. Energy-per-inference not currently measured;
lowest priority since no on-device power meter data exists — recommend stating this as an
explicit non-claim rather than estimating it from unrelated literature (same discipline as the
Jetson caveat already flagged).

## Ranked adoption list

**Primary evidence (adopt for the thesis's core statistical argument):**
1. Bayesian posterior over the accuracy difference (Beta-Binomial, unpaired, RQ2) — works at
   N=64, degrades gracefully, directly answers "how confident are we this isn't worse" without
   needing power we don't have. ~20 lines of new code.
2. Pre-registered TOST vs the paper's **no-reduction ablation row** (not full headline), macro-
   recall only, margin=6pp, run on the **k-fold N=516 data** (pipeline O) — the one place TOST is
   numerically plausible; pre-register margin in this document, before final numbers land.
3. Cost-sensitive / FNR@FPR reporting (RQ3) — the one place this thesis can show a materially
   more relevant number than the paper reports at all, framed correctly for "worker safety," using
   metrics already partially computable from the existing ROC curves (near-zero new code).
4. Bouthillier 2021 + Isaksson 2008 citations grounding the val/test anti-correlation finding —
   turns the project's strongest empirical result from an anecdote into a literature-supported
   methodological finding.

**Supporting evidence (secondary, strengthens but doesn't carry the argument alone):**
5. Cohen's h / common-language effect size alongside every reported pp-gap (RQ4) — cheap,
   standard, makes effect-size reporting explicit per the Dror et al. protocol.
6. Bayesian correlated t-test (Benavoli et al., `bayesiantests` package) on the existing k-fold
   N=516 data — Bayesian counterpart to item 2, report both side by side.
7. Early-detection latency / frames-to-lock (RQ6) — genuinely novel axis the paper never reports;
   good "we lead here" material but needs new instrumentation (per-frame running-prediction
   logging), so scope depends on remaining time budget.
8. Decision curve analysis / net benefit plot (RQ3) — strong visual, but ~1-2 days of new
   plotting code for a single figure; include only if item 3's numbers alone don't feel like
   "enough."
9. p50/p95/p99 latency + throughput-at-target-latency table (RQ7) — mostly already measured,
   just needs re-reporting in percentile form instead of a single p50 number.

**Do not pursue / low priority:**
- TOST vs the full 96.88/95.81 headline at any N — proven above to be structurally unreachable
  (point-gap exceeds even the asymptotic variance floor).
- Energy-per-inference — no instrumentation exists; state as explicit non-claim, don't estimate.
- Temporal-flicker metric (RQ6) as a standalone claim — interesting diagnostic, not decision-
  relevant enough to spend scarce remaining time on relative to items 1-4.

## Sources
- Julious, S. (2004). *Sample Sizes for Clinical Trials*, ch.10 (TOST formulas).
- Chow, Shao, Wang (2003). *Sample Size Calculations in Clinical Research*.
- Koobs, S. & Koning, N.W. (2025-26). "Equivalence testing with data-dependent and post-hoc
  equivalence margins." arXiv:2603.16213.
- Benavoli, A., Corani, G., Demšar, J., Zaffalon, M. (2017). "Time for a change: a tutorial for
  comparing multiple classifiers through Bayesian analysis." JMLR 18, 1-36.
  https://www.jmlr.org/papers/volume18/16-305/16-305.pdf ; package: `bayesiantests` (PyPI).
- Elkan, C. (2001). "The Foundations of Cost-Sensitive Learning." IJCAI.
- Cai, T. & Tong, X. (2013). "A Neyman-Pearson Approach to Statistical Learning." JMLR.
- Vickers, A.J. & Elkin, E.B. (2006). Decision curve analysis (origin); Wikipedia "Decision curve
  analysis" for formula.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (Cohen's h);
  https://en.wikipedia.org/wiki/Cohen%27s_h
- Dror, R., Baumer, G., Shlomov, S., Reichart, R. (2018). "The Hitchhiker's Guide to Testing
  Statistical Significance in NLP." ACL. https://aclanthology.org/P18-1128/
- Bouthillier, X. et al. (2021). "Accounting for Variance in Machine Learning Benchmarks." MLSys.
  arXiv:2103.03098.
- Isaksson, A., Wallman, M., Göransson, H., Gustafsson, M.G. (2008). "Cross-validation and
  bootstrapping are unreliable in small sample classification." Pattern Recognition Letters.
- "Crossing the Validation Crisis: Cross-Validation Reduces Benchmarking Variance Surprisingly
  Well." arXiv:2606.12552 (2026).
- Online action detection/early-prediction survey: ScienceDirect (2022) S0925231222003617;
  early-action-prediction GAN paper arXiv:1904.13085.
- Video temporal-consistency/flicker metric definition: arXiv:2605.18378 (2026).
- Edge-inference latency-percentile practice: Mirantis "Improving Inference Latency"; DigitalOcean
  "The LLM Inference Trilemma" (general practice, not video-specific, transferred by analogy).

## Unresolved questions
1. TOST floor/sample-size numbers above are derived by me from the standard two-proportion TOST
   formula (Julious 2004 form), not independently cross-checked against a canned example from a
   citable ML paper — verify with `statsmodels.stats.power` or R `TOSTER::power_twoprop` before
   quoting exact ns in the thesis text; qualitative conclusion (floor exists, N=64 insufficient,
   ~500-1000 needed for defensible margins) is robust to small formula variants.
2. Whether "no-reduction ablation row" (89.06/85.78) is an ethically fair equivalence target vs
   the full headline is a framing/committee-acceptance question, not a statistical one — flagging
   for T1/T5 to decide jointly with the claim-(b)/(c) framing already in t1's report.
3. No literature found for the SIGN of val/test anti-correlation specifically (only magnitude/
   unreliability generally) — if a committee member pushes on this, honest answer is "consistent
   with, more extreme than, published small-N unreliability results; the negative sign itself may
   be this dataset's own finding."
4. Early-detection-latency and decision-curve-analysis items require new instrumentation/plotting
   code not yet scoped for time budget — needs a go/no-go call from whoever owns the remaining
   implementation schedule.

---

## ADDENDUM (post fair-headtohead-scorecard read) — mixed-verdict reporting, unpaired-vs-published power loss, scene-bias grounding

Context: scorecard shows a genuinely mixed result — nominal same-dist top-1 tie/against-us (54/64 vs
62/64, but paired test impossible), macro/calibration/scene-robustness superiority (LCO +60.3pp over
scene-only vs paper's same-dist regime which never tested this), ~10 metrics reported total. Two new
RQs below; scene-bias literature grounds the "protocols rank by memorization" finding.

### RQ8 — Formally stating a mixed verdict (superior on A, non-inferior on B, untestable on C) + multiplicity

**Frame: hierarchical/fixed-sequence gatekeeping**, not simultaneous multi-metric significance
testing. Clinical-trial multi-endpoint methodology (Dmitrienko & Tamhane-style gatekeeping;
Bretz et al. 2009 "graphical approaches for multiple testing procedures," Biometrical Journal) pre-
orders endpoints into tiers; a lower tier is only tested for significance if the tier above already
"passed" (or is tested at a reduced, propagated α if it didn't) — this controls family-wise error
WITHOUT needing a flat Bonferroni/BH correction across all ~10 metrics, which would be overly
conservative here since the metrics are not independent (macro-F1, MCC, kappa all derive from the
same confusion matrix) [Dmitrienko gatekeeping overview; PMC5255051 on weighted-FDR-controlling
gatekeeping for clinical trials — same logic, directly transferable]. Concretely for this thesis:
- **Tier 1 (primary, pre-registered, alpha-controlled):** macro-recall non-inferiority/equivalence
  vs the paper's no-bbox ablation (RQ1 recommendation), on the k-fold N=516 protocol. This is the
  ONE claim that gets a formal significance/equivalence test at α=.05.
- **Tier 2 (secondary, gated on Tier 1, or reported as an independent superiority claim on a
  DIFFERENT axis the paper never measured — LCO/scene-robustness):** because Tier 2 here is not
  "the same claim tested again" but a genuinely different question (generalization, not accuracy
  parity), it does not need to be gated behind Tier 1 passing — it can run in parallel as a second,
  independently pre-specified primary-type claim (a "co-primary endpoint" in trial language, e.g.
  EMA/FDA guidance permits >1 co-primary endpoint if each is individually alpha-protected and the
  overall claim requires ALL co-primaries to succeed, which is more conservative than either alone —
  appropriate here since we want "both accuracy-parity AND robustness," not "accuracy-parity OR
  robustness").
- **Tier 3 (exploratory/descriptive, no significance claim attached):** calibration (ECE), MCC,
  kappa, F1 variants, latency, cost-sensitive numbers — reported with CIs for transparency, but
  explicitly labeled "descriptive, not part of the formal claim" — this is exactly what avoids the
  "metric-shopping" appearance: pre-declare (in this document, before final numbers) which 1-2
  metrics are the confirmatory claim, and label everything else as context.
- **Untestable tier (explicit, separate from the above):** same-dist top-1 vs the paper's full
  headline. State plainly: "no valid test exists (per-clip labels unreleased ⇒ no paired McNemar;
  unpaired test is available but under-powered/structurally capped per t1 §3b and RQ9 below) — this
  axis is reported descriptively, not adjudicated." This is a legitimate, citable category
  (clinical-trial guidance explicitly allows "non-formally-tested" secondary/exploratory endpoints
  reported without a claim of statistical significance).

**On correcting across ~10 metrics:** flat Bonferroni across macro/F1/kappa/MCC/ROC-AUC/PR-
AUC/ECE/etc. is not appropriate because (a) most are point estimates/descriptive statistics, not
hypothesis tests (no p-value to correct), and (b) they are highly non-independent (same confusion
matrix). Multiplicity correction is only needed for the tests that carry a formal p-value/
significance claim — under the tiered scheme above that's exactly 1-2 tests (Tier 1 equivalence,
Tier 2 co-primary superiority), which is small enough that Bonferroni/Holm across just those 2 is
sufficient and cheap (α/2 = .025 each) if a stricter reviewer demands it.

### RQ9 — Unpaired comparison vs a published point estimate; quantified power loss

**Options, ranked by validity given no raw/per-clip data from the paper:**
1. **Bayesian Beta-Binomial posterior comparison (RQ2, restated for this exact use case)** — treat
   the paper's 62/64 as one more Binomial(64,p_paper) observation, posterior Beta(1+62,1+2);
   ours similarly; Monte-Carlo the difference. Best choice here specifically BECAUSE it makes no
   claim of pairing (models the two as independent processes honestly) and returns a full
   distribution rather than a brittle reject/fail-to-reject call at an arbitrary N.
2. **Two-proportion z-test** — valid, standard, but normal-approximation can misbehave near
   ceiling (p→1, small n(1-p)) — exactly our regime (paper 96.88%, small-count tail). The
   "N-1 two-proportion test" variant (Campbell 2007, popularized via measuringu.com) corrects a
   known small-sample bias in the classic z by using N-1 in the pooled-variance term — cheap
   drop-in improvement over vanilla two-proportion z, recommended over plain z here given the
   near-ceiling proportions and small n.
3. **Fisher's exact test** — the right choice for the tiny expected-cell-count regime (paper's own
   error count = 2/64), already implemented (t1's Fisher table). Preferred over the z-test/N-1-z
   for reporting the confirmatory p-value given how close counts are to boundary values (t1 already
   uses this correctly).
4. **Unconditional exact test (Boschloo/Barnard)** — more powerful than Fisher's exact for 2x2
   tables while remaining exact (not just asymptotic); R package `Exact` implements it. Marginally
   better power than Fisher at this N — worth a footnote as "the more powerful exact alternative,"
   but unlikely to change any conclusion given how far from significance the current numbers sit.

**Quantified power loss from being forced unpaired (worked example, this dataset):** paired
McNemar variance depends only on the DISCORDANT-pair count n_disc = b+c (SE ≈ sqrt(n_disc)/n);
unpaired two-proportion variance depends on BOTH full binomial variances p1(1-p1)/n + p2(1-p2)/n,
which includes variance contributed by clips where the two classifiers agree (concordant pairs) —
those clips carry zero information about the DIFFERENCE but still inflate the unpaired SE. This
is the textbook reason paired designs dominate unpaired ones for correlated raters/classifiers on
shared items [general result, e.g. Fleiss/Levin/Paik *Statistical Methods for Rates and
Proportions* ch.9 on paired vs unpaired proportion comparisons]. Worked numbers for n=64,
p_paper=.9688, p_ours=.84 (54/64, illustrative): unpaired SE=sqrt(.9688*.0312/64 + .84*.16/64)
= **0.0507**; paired SE (McNemar, using the FEASIBLE discordance range from t1's bound, n_disc
8-12 given diff=8 correct-count-gap and c≤2 forced) = sqrt(n_disc)/64 ≈ **0.044-0.054** — i.e. **in
this specific dataset the two are close in magnitude**, so the textbook power advantage of pairing
is largely NEUTRALIZED here, because the paper's near-ceiling accuracy caps how large n_disc (and
hence how small paired SE) can get in our favor — the SAME structural fact (paper wrong only twice)
that produces t1's "2-pair floor" also caps how much pairing WOULD have helped even if we had the
per-clip labels. **Correct, precise statement for the thesis: being unable to pair costs us
comparatively little extra power beyond what the paper's near-ceiling result already costs us
structurally** — a more defensible and more specific claim than a generic "unpairing loses power"
statement, and directly reinforces (does not merely repeat) t1 §3b's ceiling finding.

### RQ5 addendum — scene-bias / shortcut-learning grounds the LCO finding

The scorecard's "same-dist and random k-fold rank models by camera memorization; only LCO isolates
behaviour" finding is a specific instance of a well-documented phenomenon in action recognition:
**static/background bias**. Key citations: Li, Li, Vasconcelos, "RESOUND: Towards Action
Recognition without Representation Bias," ECCV 2018 — shows standard action-recognition datasets
are biased toward static scene/object/person representations rather than motion, proposes
representation-bias quantification and the Diving48 dataset (near-identical backgrounds across
classes) specifically to defeat this; directly analogous to our LCO protocol (same idea: remove the
confound by construction, i.e. hold out cameras rather than hold out random clips). Newer: "Evaluating
and Mitigating Static Bias of Action Representations in the Background and the Foreground," arXiv
2211.12883 — introduces SCUBA (static cues in background) and SCUFO (foreground) benchmarks
specifically to quantify how much accuracy is attributable to static cues vs motion, i.e. exactly a
formalized version of our "scene-only baseline" ablation; and ALBAR (arXiv 2502.00156, 2025) —
adversarial de-biasing method for action recognition, whose motivation section restates the same
diagnosis (models rely on static/background shortcuts, generalize poorly out-of-distribution) as our
own LCO-vs-random-kfold gap. **This trio is the direct literature grounding for the scorecard's
central claim** — cite RESOUND as the origin, SCUBA/SCUFO as the modern quantification methodology
(closest analogue to our scene-only-baseline ablation), ALBAR for the current (2025-26) framing of
why this matters for OOD/deployment generalization — strengthens "protocols rank by memorization,
not behaviour" from an in-house finding to a named, literature-recognized failure mode.

### Updated ranked adoption list (supersedes prior list's ordering at the top; items below are ADDED,
prior list's items 1-9 still stand as originally ranked, folded into the tiers below)

**Primary (Tier 1, formally tested):**
1. Macro-recall TOST/equivalence vs no-bbox ablation, k-fold N=516 protocol, margin=6pp (RQ1) —
   the one confirmatory significance claim.

**Co-primary (Tier 2, independently alpha-protected, must ALSO hold for the strong claim):**
2. LCO scene-robustness superiority (+60.3pp over scene-only, scorecard §1/§3) grounded in
   RESOUND/SCUBA-SCUFO/ALBAR (RQ5 addendum) — different axis, different literature, reported as a
   co-primary not a tie-breaker.
3. Bayesian posterior-over-difference (RQ2) as the SUPPORTING quantification for both 1 and 2 —
   reported alongside the frequentist test, not replacing it, per Tier-1/2 gatekeeping framing.

**Secondary (Tier 3, descriptive, explicitly non-confirmatory — avoids metric-shopping appearance):**
4. Cost-sensitive/FNR@FPR (RQ3), Cohen's h/CLES (RQ4), calibration (ECE, already measured), F1/
   kappa/MCC family (already measured) — all reported as context, explicitly labeled "descriptive."
5. Early-detection latency (RQ6) — descriptive, novel-axis lead, not part of the formal claim.

**Untestable (explicit fourth category, not folded into 1-3):**
6. Same-dist top-1 vs paper's full 96.88% headline — state plainly per RQ9: no valid paired test
   exists; the unpaired alternatives (N-1 z, Fisher, Boschloo) are available but structurally capped
   near-uninformative by the paper's own near-ceiling result (RQ9 worked example) — report the
   number and the Fisher/Boschloo p descriptively, make no significance claim.

## Additional sources (addendum)
- Bretz, F. et al. (2009). "Graphical approaches for multiple comparison procedures using weighted
  Bonferroni, Simes, or parametric tests." Biometrical Journal 51(6).
- Dmitrienko, A. & Tamhane, A. "Gatekeeping procedures in clinical trials" (overview slides,
  ww2.amstat.org/meetings/fdaworkshop).
- Weighted-FDR gatekeeping for clinical trials: PMC5255051.
- Campbell, I. (2007) "Chi-squared and Fisher-Irwin tests of two-by-two tables with small sample
  recommendations" — origin of the N-1 two-proportion/chi-square correction; practitioner summary
  at measuringu.com "How to Compare Two Proportions with the N-1 Two-Proportion Test."
- Fisher's exact vs unconditional exact (Boschloo/Barnard): R package `Exact`
  (packages.oit.ncsu.edu/cran/web/packages/Exact).
- Fleiss, Levin, Paik, *Statistical Methods for Rates and Proportions*, 3rd ed., ch.9 (paired vs
  unpaired proportion comparison variance).
- Li, Y., Li, Y., Vasconcelos, N. (2018). "RESOUND: Towards Action Recognition without
  Representation Bias." ECCV. Diving48 dataset.
- "Evaluating and Mitigating Static Bias of Action Representations in the Background and the
  Foreground." arXiv:2211.12883 (SCUBA/SCUFO benchmarks).
- "ALBAR: Adversarial Learning approach to mitigate Biases in Action Recognition." arXiv:2502.00156
  (2025).

## Additional unresolved questions
5. Whether Tier 2 (LCO robustness) should be framed as "co-primary" (must hold jointly with Tier 1)
   or as an independent, freestanding superiority claim not gated to Tier 1 at all — a
   committee-framing choice; co-primary is the more conservative/defensible option and is what's
   recommended above, but a less cautious framing is defensible too if the committee prefers a
   simpler narrative.
6. RQ9's worked SE numbers use illustrative (54/64) counts, not the final pre-registered blind
   number — recompute once T4's headline lands; the QUALITATIVE conclusion (pairing advantage
   mostly neutralized by paper's near-ceiling result) should hold across nearby our_correct values
   but has not been checked at every value in t1's table.
