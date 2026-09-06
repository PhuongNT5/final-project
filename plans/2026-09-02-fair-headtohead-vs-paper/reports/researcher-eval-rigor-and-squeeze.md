# Eval rigor + legit accuracy squeeze on tiny scene-confounded benchmark

Scope: 452 train / 64 test (516 pooled, 8 cameras), 4 unsafe classes, per-class test n = 32/11/13/8.
Existing tooling verified in repo: `run_kfold_relation.py` (stratified 5-fold + Nadeau-Bengio vs
scene-only), `run_lco_eval.py` (leave-camera-out), `ensemble_relation.py` (2-config hybrid, BCa CI,
McNemar/z-test), `eval_utils/{bootstrap_ci,paired_tests,metrics}.py`, `run_ablation_calibration.py`
(temperature sweep), `run_operating_point.py` (risk-coverage τ), `run_tta_track_a.py` (hflip/toffset
TTA). This is already close to best-practice; recommendations below are additive, not a rewrite.

## 1. Evaluation protocols — which is the headline

| Protocol | What it measures | Power on this data | Confound-robust? |
|---|---|---|---|
| Fixed 64-clip same-dist test | paper-comparability | very low (±11pp bootstrap) | **No** — 8 cameras all in train, scene leaks |
| Stratified repeated k-fold (pooled 516) | in-distribution generalization | 5x more test events pooled across folds → tighter Nadeau-Bengio CI; still the standard "compare two algorithms" test | **No** — every camera still in every fold's train set (script docstring itself flags this) |
| Leave-camera-out (group CV, 8 folds) | cross-scene behaviour generalization | n per held-out camera small (~65 avg) but pools to 516 for the headline number | **Yes** — only protocol immune to the scene confound already measured (81.2%→16.9% collapse) |
| Nested CV | hyperparam selection + unbiased estimate | adds outer/inner splits; on 8 groups this is LCO nested inside itself (leave-1-camera-out with hyperparams tuned on remaining 7) — feasible but expensive, marginal benefit since config space is already small and blind-selected on a train-carved val slice | orthogonal to confound, mainly guards against config leakage |
| LOO (leave-one-clip-out) | max data efficiency | 516 folds, computationally heavy for retrain-per-fold DL; unbiased but very high variance per fold, and LOO CV is documented to have systematic bias/variance failure modes ([Distributional bias compromises LOOCV, arXiv 2406.01652](https://arxiv.org/pdf/2406.01652)) | no group control, same confound as k-fold unless grouped |

**Recommendation: LEAVE-CAMERA-OUT (grouped, 8-fold) is the thesis headline**, not k-fold and not the
fixed 64-clip split. Rationale: this dataset's central threat to validity is the scene confound
(scene-only 81.2% → 16.9% under LCO) — any protocol that leaves cameras in train cannot certify the
model learned *behaviour* rather than *background*. A reviewer of a safety/CV thesis will ask "does
this generalize to a new camera" first; LCO is the only existing protocol that answers it. This
mirrors the accepted fix for context confounds in action recognition: Leave-One-Scene-Out CV was
introduced specifically because "networks can overfit context rather than the actual motion" and
"Leave-One-Subject-Out is the default protocol" when identity/scene could leak
([leave-one-scene/subject-out precedent](https://arxiv.org/pdf/1908.10899)).

**Reporting stack for the headline (all already have code):**
- Point estimate: pooled top-1 + macro-recall across all 8 held-out folds (`run_lco_eval.py` already prints this).
- Uncertainty: BCa bootstrap CI on the pooled 516 (`eval_utils/bootstrap_ci.py`) — BCa preferred over
  percentile because it corrects skew/bias at small N (Efron & Tibshirani 1993, already cited in code).
- Group-level exact CI: Clopper-Pearson per held-out camera (small n per camera) — exact, no normal
  approximation, standard recommendation for small-sample binomial accuracy
  ([Clopper-Pearson for small-sample classification](https://www.afit.edu/STAT/statcoe_files/12_Binomial%20proportion%20intervals%20DRAFT%20-%20PA%20copy(1).pdf)); add a `clopper_pearson()` helper to `eval_utils/paired_tests.py` (scipy `beta.ppf`, ~10 lines).
- Significance vs scene-only / vs paper-modern: Nadeau-Bengio corrected resampled t-test across the 8
  camera-folds (reuse the existing `nadeau_bengio()` — it already handles this shape; `run_lco_eval.py`
  just needs the same diff-collection + call that `run_kfold_relation.py` has). NB correction matters
  because folds share overlapping training data, so the naive paired t-test underestimates variance and
  inflates false positives ([Nadeau & Bengio 2003](http://papers.neurips.cc/paper/1661-inference-for-the-generalization-error.pdf); [correctR vignette](https://cran.r-project.org/web/packages/correctR/vignettes/correctR.html)).
- Secondary/legacy: keep the fixed 64-clip same-dist number as a **paper-comparability footnote only**
  (with BCa CI + McNemar), explicitly labeled non-headline given the r=-0.066 val↔test correlation
  already measured (34-config sweep, p=0.71 — selection on this split is statistically indistinguishable
  from random).
- k-fold (script E) stays as a **secondary "in-distribution power" row** — useful to show tighter CIs
  than the 64-clip split, but must carry the same caveat printed in its own docstring: "every camera in
  train (scene-confounded)."
- Nested CV / LOO: **not worth the compute.** Nested-CV's main value (protecting against hyperparam
  leakage) is already achieved by the blind protocol (config chosen only on train-carved val slice,
  never on fold-test). LOO-clip adds no group-robustness and has known small-sample pathologies. Skip
  both; document why in the thesis methods section (1 paragraph) to preempt the "why not nested CV"
  question.

## 2. Legit accuracy levers, ranked by lift/risk

Ordered by (expected lift ÷ leakage risk), assuming **blind selection**: all val-only decisions come
from the train-carved 15% val slice already used in `R.train_one`, never from fold-test/LCO-test.

| Lever | Mechanism | Expected lift | Leakage risk | Blind-safe procedure | Already have code |
|---|---|---|---|---|---|
| **Seed-ensembling (3-7 seeds)** | averages softmax over independent inits, reduces variance | +1-3pp top-1, biggest CI-width reduction (variance ↓ ∝ 1/√seeds) | none — seeds are a training hyperparam, not a selection choice, as long as seed *count* is fixed a priori (not "run until best") | fix seed count before touching test; already done (`R.train_one` loop in both k-fold/LCO scripts) | Yes |
| **Caruana greedy ensemble selection (blind)** | greedily add configs to ensemble by val-loss, with replacement | +1-2pp over equal-weight if val set has enough events (~50+); on a 15%-of-452 val slice (~68 clips) this is right at the edge of reliable | **moderate** — greedy selection can overfit small val sets, esp. with many correlated candidate models (documented failure mode: "overfitting happens dramatically when there are many correlated models and only ~1k val points" — here it's far fewer than 1k) | (a) only select among the small pre-registered candidate set (not all 34 sweep configs), (b) use "with replacement" + early stopping on val-loss plateau per Caruana's anti-overfit trick, (c) never re-tune after seeing test | `ensemble_relation.py` currently does *equal* 50/50 weighting (blind, safe) — extending to greedy Caruana selection is the correct next increment but only if candidate pool ≤ ~5 configs given tiny val |
| **Temperature scaling** | post-hoc softmax rescale fit on val NLL | **0pp top-1 change on its own** (monotonic, doesn't change argmax) — real value is ECE/NLL improvement for the calibration story and for making *softmax-averaging ensembles* combine fairly across differently-confident heads | none if T fit on val only | fit T on val NLL (`run_ablation_calibration.py` already sweeps T against test — must be changed to fit-on-val, apply-to-test to be blind) | Yes, but **currently fits/reports T against TEST — this is the exact "reporting best-of-N" pitfall in §4**, must be moved to val before it's used as evidence, not just an ablation curve |
| **Test-time augmentation (hflip + temporal-offset)** | averages predictions across augmented views | **modest, domain-variable**: general CV lit shows ~+0.5-2pp for a couple of views (flip alone ~+0.7pp on ImageNet-scale); TTA can also *hurt* on small/OOD-flavoured data (documented -pp drops in medical imaging) — expect **0 to +1pp**, not more | low if variant set fixed a priori and averaged blind (not "pick best variant") | fix {baseline, hflip, toffset} averaging weights before test touch; never select which variant subset wins on test | Yes (`run_tta_track_a.py`) — confirm it averages fixed variants, not test-selects |
| **Class-balanced loss / logit adjustment** | reweight or logit-shift for imbalance (esp. n=8/11 classes) | targets macro-recall specifically: **can move macro ±3-5pp** by trading majority-class top-1 for minority recall — net top-1 may drop while macro rises | low — it's a training-time prior, no test contact | tune the single balance hyperpar(class-weight strength / logit-adjust τ) on val macro-recall, not test | Partial: `class_weights()` + `use_cw` flag exist; consider adding logit-adjustment as alternative (Menon et al. 2021, [Long-tail learning via logit adjustment](https://arxiv.org/pdf/2007.07314) — Bayes-optimal post-hoc offset by log class-prior) since with n=8 CarryOverload, inverse-freq weighting can be unstable (huge weight on 8 samples inflates gradient variance) vs. logit-adjustment which is more stable at this imbalance ratio (4:1) |
| **Threshold / operating-point tuning** | pick confidence τ for abstention, on val | not an accuracy lever per se — trades coverage for accuracy-on-fired; useful for the safety framing, not for beating a top-1 number | none if τ chosen on val | `run_operating_point.py` already computes on test logits directly — should be refit on val, applied to test, to be blind-correct | Yes, needs the same val/test split fix as calibration |
< mild augmentation (train-time, e.g. brightness/temporal jitter) | more effective train distribution coverage | **+0-2pp**, and mostly helps LCO/cross-camera generalization more than same-dist (since aug approximates camera diversity) | none, train-time only | no test contact by construction | not seen in repo — would need adding to token-extraction pipeline; medium effort |

**Ranking by lift/risk for the LCO headline specifically:** seed-ensembling > mild train-time
augmentation (targets the actual scene-confound problem) > class-imbalance handling (macro metric) >
TTA > temperature scaling (calibration only) > Caruana selection (skip unless candidate pool is tiny
and val is enlarged).

**Honest ceiling:** given bootstrap CI width ~±11pp on 64 clips and the r=-0.066 val↔test correlation
already measured, no combination of the above will produce a defensible >2-3pp "win" on the fixed
64-clip split — that number is noise-dominated. The levers' real payoff is on the LCO headline (516
pooled events, tighter CI) and on macro/calibration/robustness axes, not on squeezing the 64-clip
top-1 further.

## 3. Smallest classes (CarryOverload n=8, UnauthorInt n=11)

- **Do not** report per-class CI as if precise — n=8 binomial CI is enormous (Clopper-Pearson 95% CI
  for 8/8 correct is [63%, 100%]; for 7/8 it's [35%, 100%]). Report the interval, not just the point.
- **Pool across LCO folds, not within one split**: under LCO, CarryOverload's 8 test-time appearances
  are spread across whichever cameras hold it out — report recall pooled over all folds where it
  appears in the held-out set, not per-fold (per-fold n could be 0-3).
- **Macro-recall is already the thesis "macro" metric** (mean per-class recall, per
  `eval_utils/metrics.py` docstring) — this is the right choice for imbalance, but its variance is
  dominated by the smallest class; report macro alongside a **variance decomposition**: how much of
  macro-CI width comes from CarryOverload vs the 32-sample WalkwayViol. A 1-line note ("macro CI driven
  by n=8 class; per-class CIs in Table X") preempts reviewer pushback.
- **Avoid over-engineering for 8 samples**: complex class-balancing on a class with only 8 test
  instances mostly changes *variance*, not *bias* — inverse-freq class weights with n=8 can overweight
  a handful of possibly-atypical clips and destabilize training (high-variance gradient signal from a
  tiny class). Prefer **logit adjustment (fixed, closed-form, no extra hyperparam search)** over
  tunable class-weight strength, precisely because it doesn't add another blind-val-tuned knob for such
  a small class.
- **Do not stratify further** (e.g., ensure exact per-class-per-fold counts) beyond what
  `StratifiedKFold`/label-stratified split already gives — with n=8, any extra constraint over-fits
  the *split design* itself.
- Consider reporting **Cohen's κ / MCC** (already computed in `run_kfold_relation.py`) as
  imbalance-robust supplementary metrics rather than inventing new n=8-specific statistics.

## 4. Pitfalls reviewers will flag

1. **Model/config selection leakage** — already caught by the project (r=-0.066 val↔test finding).
   Enforce: the blind protocol document should list the *exact* val-chosen config *before* any test
   run, timestamped/logged, not reconstructed post-hoc.
2. **TTA leakage** — picking which TTA variant combo to use *after* seeing test scores (`run_tta_track_a.py`
   currently prints numbers vs "baseline to beat: 56/64" — framing as "beat this test number" is a
   red flag; must select variant set on val, report test once).
3. **Ensemble-weight leakage** — `ensemble_relation.py` is currently blind (fixed 50/50), which is
   correct; if moving to Caruana-style learned weights, those weights **must** be fit on val, frozen,
   then applied once to test/LCO-fold.
4. **Calibration-on-test leakage** — `run_ablation_calibration.py` fits/sweeps T directly against the
   test set (`FEAT = .../all8_test.pt`) and reports which T is "best" — this is test-set model
   selection by another name. Fix: fit T on val NLL, report the single resulting T applied to test,
   remove the "best T" framing from any headline claim (the sweep can remain as an *ablation figure*,
   clearly labeled non-blind/exploratory, not as evidence for the headline number).
5. **Operating-point leakage** — same issue: `run_operating_point.py` computes τ from test logits
   directly. Fix analogous to above.
6. **Reporting best-of-N** — the withdrawn "+3.5pp macro win" (already documented in
   `plans/2026-09-01.../thesis-process-review.md`) is the canonical example; the fix already adopted
   (blind protocol, single test touch) is correct — keep applying it to every new lever above.
7. **Scene confound not addressed by headline metric** — if the thesis reports only same-dist or
   random-k-fold as headline, reviewers of a safety/CV thesis will immediately ask about
   cross-camera generalization; pre-empt by making LCO the headline (§1).
8. **Underpowered significance claims** — `mcnemar_exact` already self-flags `underpowered: n_disc<10`
   on N=64; carry the same discipline to any new pairwise test (LCO pooled n=516 has much better power,
   prefer it for any "our model beats baseline" claim).
9. **Multiple comparisons** — 34 legit configs were swept; any table showing several metrics × several
   configs without a comparison-count caveat invites a p-hacking read. State N-configs-tried and that
   the reported config was picked blind (val-only) exactly once.

## 5. Best-practice checklist (condensed)

- [ ] Headline = leave-camera-out, pooled 516, BCa CI + Clopper-Pearson per-camera + Nadeau-Bengio vs baselines.
- [ ] Fixed 64-clip split kept only as paper-comparability footnote with r=-0.066 caveat inline.
- [ ] All post-hoc knobs (T, τ, ensemble weights) fit on val, frozen, applied once to test/fold — audit
      `run_ablation_calibration.py` and `run_operating_point.py` for the val/test fix.
- [ ] Seed-ensembling count fixed a priori (already true).
- [ ] TTA variant set fixed a priori, no "beat the number" framing.
- [ ] Small classes: report Clopper-Pearson CI per class, note macro-CI is n=8-class-dominated, prefer
      logit-adjustment over tunable class-weights.
- [ ] State config-search count (34) explicitly wherever a "best config" number appears.
- [ ] Nested-CV/LOO explicitly declined with 1-paragraph justification (compute vs. marginal benefit
      already covered by blind protocol + group CV).

## Sources
- [Nadeau & Bengio, "Inference for the Generalization Error" (2003)](http://papers.neurips.cc/paper/1661-inference-for-the-generalization-error.pdf)
- [correctR vignette — corrected resampled t-test implementations](https://cran.r-project.org/web/packages/correctR/vignettes/correctR.html)
- [Distributional bias compromises leave-one-out cross-validation, arXiv 2406.01652](https://arxiv.org/pdf/2406.01652)
- [Leave-one-scene/subject-out precedent for confounded video/context, arXiv 1908.10899](https://arxiv.org/pdf/1908.10899)
- [Caruana et al., Ensemble Selection from Libraries of Models](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)
- [Guo et al. 2017, On Calibration of Modern Neural Networks (temperature scaling)](https://www.researchgate.net/publication/317591245_On_Calibration_of_Modern_Neural_Networks)
- [Menon et al. 2021, Long-Tail Learning via Logit Adjustment](https://arxiv.org/pdf/2007.07314)
- [Bouthillier et al. 2021, Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/pdf/2103.03098)
- [Clopper-Pearson intervals for small-sample binomial classification accuracy](https://www.afit.edu/STAT/statcoe_files/12_Binomial%20proportion%20intervals%20DRAFT%20-%20PA%20copy(1).pdf)
- Repo: `run_kfold_relation.py`, `run_lco_eval.py`, `ensemble_relation.py`, `eval_utils/{bootstrap_ci,paired_tests,metrics}.py`, `run_ablation_calibration.py`, `run_operating_point.py`, `run_tta_track_a.py`; `plans/2026-09-01-unsafenet-r2-rewrite/reports/thesis-process-review.md` (r=-0.066 finding origin).

## Unresolved questions
- Is the r=-0.066/34-config correlation report itself saved anywhere as a script/artifact, or only
  narrated in `thesis-process-review.md`? If not scripted, worth codifying as a reusable
  `run_val_test_correlation.py` so the "selection is near-random" claim is reproducible on demand.
- Candidate pool size for Caruana selection not yet decided — needs a fixed pre-registered list (≤5
  configs) before implementation, else the "greedy selection on small val" risk in §2 applies.
- Whether mild train-time augmentation (brightness/temporal jitter) is worth the pipeline change given
  YOLO+box-token extraction is already built around fixed frame sampling — needs an engineering-cost
  vs. expected-LCO-lift tradeoff call from the team, not purely a research question.
- `run_ablation_calibration.py` / `run_operating_point.py` fit directly on test — confirm whether any
  existing headline claim in the thesis already relies on those test-fit numbers (would need retraction
  similar to the withdrawn +3.5pp, if so).
