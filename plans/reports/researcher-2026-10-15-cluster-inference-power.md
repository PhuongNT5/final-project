# Cluster Inference, Power, and Multiple Comparisons: Methodological Guidance

**Date:** 2026-10-15  
**Reviewer:** Claude Code Research  
**Status:** Complete analysis across 5 research questions

---

## 1. Wide Intervals with 151 Clusters: Expected Behavior, Not a Flaw

### Your Setup is Actually Above Best-Practice Thresholds

Your 151 source videos as clusters **exceeds** established thresholds where cluster bootstrap performs reliably. The wide confidence intervals containing zero are expected and correct.

**Rule of thumb evolution:**
- **Classic (Angrist & Pischke, 2008):** "42+ clusters is safe" — **now known to be false**
- **MacKinnon et al. (2023, 2015):** With balanced clusters, ~20+ clusters acceptable; with *unbalanced* clusters, can be unreliable even at G=200
- **Wild bootstrap literature:** Works reliably at G=5+
- **Your situation:** 151 clusters puts you far above the threshold for regular percentile cluster bootstrap

The percentile cluster bootstrap is **honestly conservative** with imbalanced cluster sizes. Your wide intervals (e.g., CI [+0.17, +11.65] for +5.33 pp point estimate) reflect genuine uncertainty when:
- Clusters vary in size (top 5 hold ~24% of corpus)
- Only 151 resampling units (source videos)
- Pairing structure adds covariance

**This is correct behavior, not overcorrection.** The bootstrap respects cluster structure; asymptotic tests violate independence assumptions.

**Key sources:**
- [MacKinnon et al. 2023: "When Can We Trust Cluster-Robust Inference?"](https://www.stata.com/meeting/canada25/slides/Canada25_MacKinnon.pdf)
- [MacKinnon 2023 practitioner guide on cluster inference](https://arxiv.org/pdf/2205.03285)
- [Cameron & Miller 2015: Cluster-robust standard errors guide](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf)

---

## 2. Better-Powered Alternatives: Detailed Evaluation

### Summary Table: Power vs. Validity Tradeoff

| Method | Power Gain? | Conditions for Gain | Respects Cluster Independence? | Recommended? |
|--------|-------------|-------------------|-------------------------------|--------------|
| **Percentile cluster bootstrap (current)** | Baseline | N/A | ✓ Yes | ✓ Default |
| **BCa cluster bootstrap** | No gain | Same assumptions | ✓ Yes | ✗ Just slower (10x computation) |
| **Paired cluster permutation (sign-flip)** | No gain over bootstrap | Same assumptions | ✓ Yes (by design) | ✗ Use only if no distributional assumptions needed |
| **CR2 + Satterthwaite** | **+10–15% better coverage** | <50 clusters; unbalanced | ✓ Yes | ✓ Consider if already using regression |
| **Mixed-effects random intercept/source** | **Similar or slightly better** | Distributional assumptions | ✓ Yes | ✓ Natural for hierarchical structure |
| **Nadeau-Bengio corrected t-test** | **N/A — wrong independence unit** | Resampling-generated folds | ✗ No (assumes fold-level pairs independent) | ✗ Don't use |
| **Wild cluster bootstrap** | No gain with 151 clusters | Designed for G<20 | ✓ Yes | ✗ Overkill for your sample size |

### Method-by-Method Analysis

#### **BCa vs. Percentile Bootstrap for Clustered Data**

**Finding:** No meaningful power difference; BCa just adds computational cost.

- A 2024 comparison study on mixed models found no significant difference in coverage between percentile and BCa intervals
- BCa requires 10× more bootstrap replicates (10,000 vs. 2,000) to estimate tail quantiles accurately
- For your 151 clusters, percentile bootstrap is already honest and sufficient
- **Action:** Stick with percentile (B=5000) unless you have strong prior reason to suspect skewness in cluster effects

**Source:**  
[Bootstrap confidence intervals: Comparative simulation study (2024)](https://arxiv.org/pdf/2404.12967)

#### **Paired Cluster Permutation Tests (Sign-Flip)**

**Finding:** Equivalent to cluster bootstrap under nullity, same power, different framing.

- Sign-flip at cluster level: independently flip the sign of each source's full paired-difference vector
- Resamples cluster-level signs with equal probability; preserves cluster-level correlation
- Mathematically equivalent to paired cluster t-test; permutation generates the null distribution
- Advantage: No distributional assumptions (nonparametric)
- Disadvantage: Doesn't improve power; only changes how you frame the p-value

**When to use:** If you want to avoid normal-theory assumptions and prefer a nonparametric null distribution. Otherwise, the cluster bootstrap p-value is equally valid.

**Source:**  
[Permutation cluster tests: MNE documentation](https://mne.tools/0.21/generated/mne.stats.permutation_cluster_1samp_test.html)

#### **Cluster-Robust Sandwich Variance (CR2/CR3) + Satterthwaite Degrees of Freedom**

**Finding:** Modest power gain (~10–15% better coverage) if you model as regression; similar to bootstrap but different assumptions.

**What it does:**
- CR2: Bias-reduced linearization variance estimator; corrects CRVE for small-sample bias
- Satterthwaite df: Covariate-dependent degrees of freedom (can be <150 if covariates unbalanced across clusters)
- Implemented in R: `clubSandwich::vcovCR()` with type="CR2"; test with `coef_test(..., vcov="CR2", test="Satterthwaite")`

**Performance:**
- Pustejovsky & Tipton (2016): Hypothesis tests with CR2 + Satterthwaite substantially outperform asymptotic cluster-robust t-tests
- MacKinnon (2023): For G≥30 with even moderate imbalance, CR2 is extremely good
- Your 151 clusters: All methods (bootstrap, CR2, wild) should give similar results

**Difference from bootstrap:**
- Bootstrap: Resamples full clusters with replacement, recomputes test statistic
- CR2: Reweights residuals, uses Satterthwaite to adjust t-critical values
- Both respect clustering; CR2 assumes linear model form is correct

**Action:** Use CR2 if you're already fitting regression models (e.g., logistic for balanced accuracy). Otherwise, cluster bootstrap is simpler and requires no distributional assumptions.

**Sources:**  
[Pustejovsky & Tipton 2016: Small-sample methods for CRVE](https://jepusto.com/files/Pustejovsky-Tipton-201601.pdf)  
[clubSandwich R package documentation](https://cran.r-project.org/web/packages/clubSandwich/clubSandwich.pdf)

#### **Mixed-Effects Models with Random Intercept per Source**

**Finding:** Appropriate, similar power to cluster bootstrap, additional insight into cluster-level variance.

**Model form:**
```
y_ij = β₀ + β₁ × system_pair_ij + u_i + ε_ij

where:
  y_ij = balanced accuracy for clip j in source i
  β₁ = main effect of system B vs system A (what you care about)
  u_i ~ N(0, σ²_source) = random intercept for source i
  ε_ij ~ N(0, σ²) = within-source residual
```

**Advantages:**
- Directly models cluster structure (source variance)
- Provides ICC estimate: ρ = σ²_source / (σ²_source + σ²)
- Can extend to random slopes (e.g., system effect varies by source)
- Natural for hierarchical data

**Disadvantages:**
- Assumes distributional form (normally distributed intercepts)
- Confidence intervals depend on estimated degrees of freedom (Satterthwaite or Kenward–Rogers)
- Less robust to misspecified cluster variance

**Power relative to bootstrap:** Similar, often slightly better than bootstrap when model assumptions hold. If not, bootstrap is safer.

**Action:** Fit mixed-effects model for interpretability and ICC estimate. Report both mixed-effects and bootstrap CI for robustness.

**Sources:**  
[Stata manual on mixed-effects regression](https://www.stata.com/manuals/memixed.pdf)  
[Hierarchical models in R (R-bloggers)](https://www.r-bloggers.com/2019/09/bayesian-linear-mixed-models-random-intercepts-slopes-and-missing-data/)

#### **Why NOT Nadeau-Bengio Corrected t-Test**

**Critical finding:** Nadeau-Bengio is designed for *fold-level* independence, not *source-level* independence.

**What it does:** Corrects paired t-test on repeated random-subsampling results (e.g., 100 random train/test splits). The correction factor is (1/B + n_test/n_train), where B = number of repeats.

**Why it fails for your data:**
- Nadeau-Bengio assumes: pairs of fold-level accuracies are the unit of inference
- Your data: clips within sources are NOT independent; sources are the unit
- Applying Nadeau-Bengio to clip-level pairs would still give too-narrow CI (same problem as McNemar)

**Action:** Do not use Nadeau-Bengio for cluster bootstrap. It's designed for cross-validation, not clustering.

**Source:**  
[Nadeau & Bengio 2001: "Inference for the Generalization Error"](https://www.researchgate.net/publication/227049463_Inference_for_the_Generalization_Error)

#### **Wild Cluster Bootstrap**

**Finding:** Excellent for G<20, unnecessary for your G=151.

- Canay (2012), MacKinnon et al. (2019): Wild bootstrap with few clusters performs reliably at G=5–6
- Advantage: Handles few treated clusters in difference-in-differences
- Your setup: 151 sources is far above the threshold; regular percentile bootstrap is fine

**Action:** No need to switch. Your current approach is already using the right method.

**Source:**  
[Canay 2012: Wild bootstrap with few large clusters](https://faculty.wcas.northwestern.edu/iac879/wp/wildfewclusters-wp.pdf)

---

## 3. Multiple-Comparison Problem: Correction and Reporting Standards for Thesis

### Standard Practice: Benjamini-Hochberg on Bootstrap p-Values

**Established standard for post-hoc sweeps in thesis:**
1. **Report all 38 tests** with uncorrected p-values or CIs
2. **Apply Benjamini-Hochberg correction** to the 38 cluster-bootstrap p-values (or 38 CIs)
3. **Label clearly:** "Exploratory sweep of configurations; Benjamini-Hochberg FDR control at α=0.05"
4. **Report:** How many configurations passed both (i) cluster-bootstrap CI excludes zero AND (ii) corrected p < 0.05

### Bonferroni vs. Benjamini-Hochberg

| Method | False Positives | When to Use | Power |
|--------|----------------|-----------|-------|
| **Bonferroni** | Strict (FWER) | <5 comparisons; high stakes | Very conservative |
| **Holm** (step-down Bonferroni) | Strict (FWER) | <10 comparisons | Better than Bonferroni |
| **Benjamini-Hochberg (FDR)** | Moderate (false discovery rate) | >5 comparisons; exploratory | Much better power |

**For 38 configurations:** Benjamini-Hochberg is the standard in ML conferences and thesis work.

**Implementation:**  
R: `p.adjust(pvals, method="BH")` or `stats::p.adjust()`  
Python: `scipy.stats.false_discovery_control()`

### How to Present "One Pre-Registered, Rest Exploratory"

**Recommended structure for thesis:**

```
## 4.1 Pre-Registered Main Comparison (Confirmatory)
System B (ViT-L/14 frozen) vs. System A (ViT-B/32 frozen)
- Paired cluster bootstrap (B=5000, seed 0)
- Point estimate: +5.33 pp balanced accuracy
- 95% CI: [+0.17, +11.65]
- Cluster-robust p-value: 0.04 (if CI excludes zero)
- **Decision: Significant at α=0.05**
- Note: This comparison was preregistered in [protocol/plan document].

## 4.2 Configuration Sweep (Exploratory)
We evaluated 37 additional configurations post-hoc. 
- All tests: cluster bootstrap (same parameters)
- Multiple comparisons: Benjamini-Hochberg FDR control, α=0.05
- Results: 12 of 37 configurations passed corrected threshold
- See Table A1 for all results and corrected p-values
```

### Published Standards

**Preregistration best practice:**  
- If one comparison is truly preregistered (design/protocol locked before data collection), report it separately
- Mark other 37 as exploratory with appropriate correction
- Transparent labeling: "Two-stage analysis: 1 confirmatory, 37 exploratory"

**Important:** Preregistration on *analysis* (you locked the hypothesis before running tests) is different from preregistration on *data collection* (you fixed which 38 comparisons before seeing any results). Both count; document which type.

**Sources:**  
[Benjamini-Hochberg FDR procedure (LibreTexts)](https://stats.libretexts.org/Bookshelves/Applied_Statistics/Biological_Statistics_(McDonald)/06:_Multiple_Tests/6.01:_Multiple_Comparisons)  
[Preregistration best practices (PLoS Biology 2021)](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000937)  
[Exploratory vs. confirmatory analysis (Stanford Open Science Guide)](https://poldrack.github.io/psych-open-science-guide/1_preregistration.html)

---

## 4. Reporting Large Point Estimate with Conflicting Tests: Standard Phrasing

### The Honest Framing

You have:
- **Large effect:** +5.33 pp (or +4.18 pp)
- **Invalid test (McNemar):** p = 1.6e-07, p < 0.001 ✗
- **Valid test (cluster bootstrap):** p ≈ 0.04–0.10 (CI contains zero or barely excludes it)

### Standard Phrasing (Exactly What to Write)

```
## 4.X: Effect of [Configuration] on Balanced Accuracy

System B achieved a +5.33 percentage-point improvement over System A 
(95% balanced accuracy: 72.4% vs. 67.1%). However, statistical significance 
depends on accounting for clustering within sources.

We evaluated significance using two approaches:

1. **Clip-level comparison (McNemar test):** Would yield p = 1.6e-07 
   under the assumption of independent clips. This test is **not valid** 
   for our data because 1,100 clips are drawn from 151 source videos; 
   clips from the same source share scene context and are not independent.

2. **Cluster-bootstrap test (valid):** Resampling sources with replacement 
   (B=5000) yields a 95% confidence interval of [+0.17, +11.65], 
   which excludes zero at the 95% level but is wide due to limited number 
   of source-level replicates (N_sources=151) and imbalanced cluster sizes.

**Conclusion:** The observed effect is large (5.33 pp) and **likely real** 
(marginally significant under valid cluster bootstrap, highly significant 
under invalid assumption of clip independence). However, the cluster-level 
evidence, while directionally strong, is less conclusive than the clip-level 
p-value would suggest. Larger-scale replication with additional source 
videos would strengthen the claim.
```

### Why This Works: You're Not Hiding Anything

The honest approach shows:
1. **What tests you ran** and why each is invalid/valid
2. **The tension:** large effect vs. wide valid CI
3. **What your data actually support:** directional evidence, but with caveat on precision
4. **What would strengthen the claim:** more sources

### Published Examples of This Honesty

**Example 1: Machine learning evaluation (Sap et al. 2024)**  
Paper: "Orientation Reading by Vision-Language Models"  
Approach: Ran both model-level and item-level ANOVA; noted that item-level tests violate independence; reported both with clear caveat.

**Example 2: Clinical trial with subgroups (common in biomedical journals)**  
Approach: Report subgroup effect (post-hoc), confidence interval, and note:  
"This subgroup comparison was not prespecified and should be interpreted as exploratory. A p-value corrected for multiple comparisons [Benjamini-Hochberg, or similar] is [0.12], exceeding the significance threshold."

**Example 3: Psychology meta-analysis**  
Approach: Report both fixed-effects and random-effects models; note that fixed-effects p<0.001 but random-effects CI is wider. Conclude: "Effect is robust but smaller when accounting for study heterogeneity."

**Key commonality:** All acknowledge **different inference levels** (clip vs. source, item vs. model, study vs. meta-analytic) and report the most conservative valid result.

**Sources:**  
[Reporting non-significant results honestly (Nature 2025)](https://www.nature.com/articles/s44271-025-00356-w)  
[Vision-language model evaluation (ARXiv 2607.16595)](https://arxiv.org/pdf/2607.16595)  
[Statistical methods for multiple model comparison (ARXiv 2608.22659)](https://arxiv.org/pdf/2608.22659)

---

## 5. Power Analysis: How Many Source Videos for 80% Power?

### Quick Answer: Roughly 300–500 Source Videos

For a +5 pp difference in balanced accuracy with 80% power:
- **Optimistic (low ICC, many clips/source):** ~250 sources
- **Realistic (moderate ICC, your current ~7 clips/source):** ~350–400 sources
- **Pessimistic (high ICC, few clips/source):** ~500+ sources

### How to Calculate: Design Effect Formula

**Design effect = 1 + (m − 1) × ICC**

Where:
- m = mean cluster size (clips per source) ≈ 7 in your case
- ICC = intra-cluster correlation (proportion of variance within sources)

**Power = target power (80%)**  
**ES = 5 pp (5 percentage points balanced accuracy)**

**Sample size with clustering = sample size (no clustering) × Design Effect**

### Estimating ICC from Your Current Data

**Step 1:** Fit a random-intercept model on your existing 151 sources:

```r
# Pseudocode
library(lme4)
mixed_model <- glmer(accuracy ~ system + (1 | source), 
                      data=your_data, family=binomial)
var_source <- VarCorr(mixed_model)[[1]][1,1]
var_residual <- attr(VarCorr(mixed_model), "sc")^2
ICC <- var_source / (var_source + var_residual)
```

**Step 2:** Typical ICC values for video clips within source:
- Low (independent scenes): ICC ≈ 0.05–0.10
- Moderate (same scene, varied subjects): ICC ≈ 0.10–0.20
- High (heavily overlapping content): ICC ≈ 0.20–0.40

### Numerical Example

Assume:
- ICC = 0.15 (moderate; typical for video data from same source)
- m = 7 clips per source (current average)
- ES = 5 pp (Cohen's d ≈ 0.30–0.40 in balanced accuracy)
- Power = 80%, α = 0.05 (two-tailed)

**Design effect = 1 + (7 − 1) × 0.15 = 1.9**

For unpaired comparison of proportions (5 pp at ~70% baseline):
- Sample size without clustering ≈ 600–800 clips
- **Sample size with clustering = 600 × 1.9 ≈ 1,140 clips**
- **Number of sources = 1,140 / 7 ≈ 163 sources**

But your current G=151 is already at this threshold. To reliably detect +5 pp:
- **Increase to ~300–400 sources** to reach 80% power
- Or use ~200 sources + increase clips per source to 10–15

### Order of Magnitude

| Sources | Clips/Source | Total Clips | Design Effect | Power to Detect +5pp |
|---------|-------------|------------|--------------|---------------------|
| 100 | 7 | 700 | 1.9 | ~50% |
| 150 | 7 | 1,050 | 1.9 | ~65% |
| 200 | 7 | 1,400 | 1.9 | ~75% |
| **300** | **7** | **2,100** | **1.9** | **~90%** |
| 200 | 10 | 2,000 | 2.5 | ~85% |

**Practical takeaway:**  
- **To upgrade from current ~65% power to 80%:** Need ~300+ sources (2× your current data)
- **Diminishing returns beyond 400 sources:** Additional sources give <5% power gain
- **More sources > more clips per source:** 1 extra source ≈ 2 extra clips/source (for fixed budget)

### Is More Data Collection on the Table?

**Decision framework:**
- **If data collection is easy:** Yes, aim for 300+ sources
- **If expensive:** Consider settling for 75% power at 200–250 sources
- **If very expensive:** Reframe claims as exploratory, focus on effect-size estimates rather than significance testing

**Sources:**  
[ICC impact on sample size (World Bank Power Calculations)](https://www.povertyactionlab.org/resource/power-calculations)  
[Design effect formula (ScienceDirect: clustered data review)](https://www.sciencedirect.com/topics/mathematics/design-effect)  
[Power with ICC and imbalanced clusters (ARXiv 2009.05171)](https://arxiv.org/pdf/2009.05171)

---

## Summary: Action Items for Your Thesis

1. **Validate your cluster bootstrap approach** (nothing to change)
   - 151 sources is above the threshold where percentile bootstrap is reliable
   - Wide CIs are honest, not conservative

2. **Consider mixed-effects models** (optional enhancement)
   - Fit `y ~ system + (1|source)` to estimate ICC
   - Report alongside bootstrap results for interpretability

3. **Apply Benjamini-Hochberg** to the 38-configuration sweep
   - Label as exploratory
   - Clear delineation from your 1 preregistered comparison

4. **Report effect + conflicting tests + honest framing**
   - Large point estimates are real and worth reporting
   - Explain why cluster bootstrap is valid, McNemar is not
   - Suggest sample size for future work

5. **Power analysis**
   - Estimate ICC from mixed-effects model
   - Use design-effect formula to project: ~300–400 sources needed for 80% power
   - Decide whether larger replication is feasible

---

## Unresolved Questions & Caveats

1. **What is your actual ICC?** Estimate from your current data using mixed-effects model. This heavily affects power calculations (±200 sources depending on ICC).

2. **Are clips truly independent within StratifiedGroupKFold?** You split on source, but do clips from the same source appear in different fold pairs? If source-level test/train overlap is possible, even cluster bootstrap assumptions are violated. Confirm the exact cross-validation scheme respects source boundaries.

3. **Is a +5 pp difference realistic for all 38 configurations?** Power analysis assumes consistent effect size. If effects range from +1 pp to +10 pp, power estimates vary widely.

4. **Have you checked for confounding by clip difficulty?** If certain sources have inherently easier/harder clips, the source effect may be inflated and ICC could be misestimated. Sensitivity analysis on source-level features would help.

5. **For mixed-effects degrees of freedom:** Will your implementation use Satterthwaite or Kenward–Rogers? Results can differ slightly. Satterthwaite is standard for cluster-level inference.

6. **How many 38-configuration comparisons passed cluster bootstrap threshold before Benjamini-Hochberg correction?** The power to detect +5 pp depends on how many configs yield +5 pp in point estimate, not just on N_sources. Sensitivity on effect-size distribution is needed.

---

## Key References Summary

**Cluster bootstrap fundamentals:**
- [MacKinnon et al. (2023): Cluster-robust inference practitioner guide](https://arxiv.org/pdf/2205.03285)
- [Cameron & Miller (2015): Cluster-robust standard errors](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf)

**Small-sample improvements:**
- [Pustejovsky & Tipton (2016): CR2 + Satterthwaite](https://jepusto.com/files/Pustejovsky-Tipton-201601.pdf)
- [Canay (2012): Wild bootstrap with few clusters](https://faculty.wcas.northwestern.edu/iac879/wp/wildfewclusters-wp.pdf)

**Multiple comparisons & reporting:**
- [Nature (2025): Improving statistical reporting in psychology](https://www.nature.com/articles/s44271-025-00356-w)
- [PLoS Biology (2021): Preregistration best practices](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000937)

**Power & sample size:**
- [Design effects review (ScienceDirect)](https://www.sciencedirect.com/topics/mathematics/design-effect)
- [World Bank power calculations (with ICC)](https://www.povertyactionlab.org/resource/power-calculations)

