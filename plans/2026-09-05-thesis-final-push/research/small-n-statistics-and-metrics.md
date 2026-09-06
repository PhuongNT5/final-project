# Small-N Statistics Practice & Recommended Metrics

## 1. How the field handles <100-test-clip significance - survey

- **Practice in surveyed unsafe-behaviour/action-recognition papers: none report confidence intervals or significance tests.** Unsafe-Net (our baseline), Yang et al. CMA paper, and the transformer-comparison Buildings 2025 paper all report single point-estimate accuracy numbers with no CI, no repeated-run variance, no paired test. This is the norm, not the exception, in this sub-field - which is itself the gap our thesis can point to.
- **General ML methodology literature says the opposite should be done:**
  - Binomial/Wilson score interval is the standard tool for reporting a proportion (e.g. top-1 accuracy) computed on a small N. Wilson is preferred over the naive normal-approximation interval because it stays valid and doesn't produce nonsensical bounds (e.g. >100%) when the observed proportion is near 0 or 1, and remains reasonably tight/valid down to small n [1][2].
  - At N=64, a rule-of-thumb width: getting a tight (~4%) CI on a >90%-accuracy classifier needs on the order of 1,000 test samples; at N=64 the CI half-width will be large (typically +-8-12 points depending on the point estimate) - i.e. 62/64 vs 55/64 may not be statistically distinguishable once a CI is attached [3].
  - McNemar's test is the recommended paired test for comparing two classifiers evaluated once on the *same* test set (exactly our situation: our model vs. Unsafe-Net's reported numbers, or our own ablations against each other). It directly tests whether the disagreement pattern (cases where one model is right and the other wrong) is asymmetric enough to reject equal error rates, and is specifically recommended for single-train/single-test-set deep learning comparisons because it doesn't assume repeated resampling [4].
  - For small-N *repeated-run* comparisons (not quite our case, since neither we nor the paper report multiple seeds), the RL evaluation literature (Agarwal et al., NeurIPS 2021 Outstanding Paper, "Deep RL at the Edge of the Statistical Precipice") is the most rigorous small-N precedent: stratified bootstrap percentile CIs give good coverage down to as few as N=10 runs, and the paper argues point-estimate-only reporting at small N actively slows scientific progress by making noise look like signal [5][6]. This is directly citable as methodological support for "small-N without a CI is not defensible," independent of whether it's RL or vision.
  - Confidence intervals for the F1 score specifically (not just raw accuracy) have dedicated methodology (4 competing CI construction methods compared) - relevant if we report per-class F1 with intervals [7].

## 2. This supports a "negative result with teeth" thesis claim

Because (a) no comparable paper in this space reports a CI or test, (b) the paper's own frame-reduction ablation (62 vs 57 /64) is not significant under McNemar (p>=0.0625, per plan.md), and (c) a Wilson CI on N=64 will be wide enough that 55/64 (ours, test-selected) and 62/64 (theirs) plausibly overlap - the defensible thesis claim is: *at N=64, this comparison class cannot be decided by a single point-accuracy number; the field's practice of not reporting intervals obscures this.* This is a real, citable methodological contribution, not just an excuse.

## 3. Recommended metric set (beyond top-1/macro accuracy)

| Metric | Why | Priority |
|---|---|---|
| Wilson 95% CI on top-1 accuracy | mandatory at N=64; makes point estimates honest | must-have |
| McNemar's test (paired) vs. paper's reported per-clip correctness pattern, and vs. our own ablations | direct significance test for "did X beat Y" claims | must-have |
| Per-class F1 (or precision/recall pair) | class sizes are unequal (32/11/13/8) - top-1 hides small-class performance | must-have |
| Balanced accuracy (macro recall) | already implicitly what the paper's "95.81%" is (verified in method-verification doc) - report explicitly and name it | must-have |
| Confusion matrix (4x4, raw counts) | shows structural errors (paper's own Fig 12e/f shows Unauthorized Intervention <-> Opened Panel Cover confusion) - lets reader see if errors cluster on genuinely visually-similar classes | must-have |
| Matthews Correlation Coefficient (MCC) | single-number summary that stays informative under class imbalance, uses all 4 confusion-matrix cells; more robust than accuracy or F1 alone when classes are skewed (8:1 ratio here) | should-have |
| Cohen's kappa | chance-corrected agreement; useful as a secondary imbalance-robust check alongside MCC, and it is the paper's OWN safety-inspection domain convention for inter-rater style agreement, so it reads naturally to a reviewer from this field | nice-to-have |
| Calibration (if using a model producing probabilities, e.g. CLIP-based) | reliability diagram / ECE - relevant if we report confidence-thresholded operating points | nice-to-have if applicable |

**Do not report:** additional exotic metrics with no established use in this literature (e.g. temporal IoU - not applicable, we do whole-clip classification, not temporal localization).

## Sources
[1] https://rdrr.io/cran/fastR2/man/wilson.ci.html
[2] general Wilson-vs-Clopper-Pearson consensus from ML stats literature searched (no single canonical paper found; standard stats reference)
[3] https://arxiv.org/pdf/2210.14677 (precision of performance estimates, medical imaging, small-N CI width discussion)
[4] https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/ (McNemar's test recommendation for single train/test set DL comparison)
[5] https://arxiv.org/abs/2108.13264 (Agarwal et al., Deep RL at the Edge of the Statistical Precipice, NeurIPS 2021)
[6] https://github.com/google-research/rliable
[7] https://arxiv.org/pdf/2309.14621 (Confidence Intervals for the F1 Score: A Comparison of Four Methods)

## Unresolved questions
- No single canonical paper found specifically for "Wilson vs normal-approx for classifier accuracy CIs" - cited as general statistical practice; if the thesis needs a hard textbook citation, recommend Brown, Cai & DasGupta (2001) "Interval Estimation for a Binomial Proportion," Statistical Science - not independently re-verified here, flagging for T1/T5 to add if needed.
