# Evaluation Protocols & Robustness — Research Report

**REVISED**: much of the originally-scoped protocol was already implemented and run
(`run_kfold_relation.py`, `run_lco_eval.py`, `eval_utils/paired_tests.py`, results in
`plans/2026-09-02-fair-headtohead-vs-paper/reports/fair-headtohead-scorecard.md`). This revision
leads with an audit of that existing work, then keeps the remaining (not-yet-done) research
sections, re-ranked against what already exists.

## PART A — Audit of the existing k-fold / LCO / NB implementation

### A1. Verified correct

- **NB correction formula is right.** `eval_utils/paired_tests.py::nadeau_bengio`:
  `corr = 1/n + n_test/n_train`, `t = mean_diff / sqrt(corr * var)`, `df = n-1`. This is exactly the
  general Nadeau & Bengio (2003) resampling correction (`t = d̄ / sqrt((1/n + n2/n1)·σ̂²)`), and using
  the empirical mean fold sizes for n_test/n_train is equivalent to the standard k-fold
  simplification n2/n1 = 1/(k-1) (here ≈103/413≈0.249 vs the theoretical 1/(5-1)=0.25 — matches to
  3 decimals). **No formula bug.** [Nadeau & Bengio 2003, Machine Learning; Bouckaert & Frank 2004;
  correctR CRAN vignette — cross-checked against this report's earlier general-research pass]
- **Minority-class stratification concern (raised in my original Part B, Q1) does NOT apply to the
  516-clip pool as actually used.** I originally worried an ~8-clip minority class would starve
  k=5 folds. Verified against `sweep_results/zone_prior/clip_view_map.json` + `data/{train,test}/*`
  clip counts: the 516-clip pool is the 4 UNSAFE classes only (labels 0-3), pooled sizes
  **210 / 108 / 142 / 56** (WalkwayViol/UnauthorInt/OpenedPanel/CarryOverload) — smallest class is
  56, not 8. (The "~8" figure in the brief is the TEST-SPLIT-ONLY count for CarryOverload, 8/64; the
  pooled train+test count for that class is 56.) At k=5 that's ~11/fold minimum — stratification is
  in no danger. **My original k=4-fallback recommendation is withdrawn as unnecessary for this pool.**
  It would still apply if/when the 4 SAFE classes (sizes 75/38/32/30 pooled, see A3) are folded in
  for an 8-class task — still comfortably ≥30/class, also fine at k=5.
- **The scene-only-baseline-vs-model design is sound and already delivers exactly the Q4 evidence
  this task asked for** (see Part B for the literature grounding it should cite).

### A2. Statistical defect found — the one worth fixing

**`run_kfold_relation.py` runs a SINGLE 5-fold split (`StratifiedKFold(..., random_state=0)`), not a
repeated k-fold.** The `--seeds 3` argument only re-seeds model *training* inside each fixed fold
(reduces model variance, i.e. a 3-model ensemble per fold) — it does **not** redraw the fold
partition. Consequence:
1. The NB test runs on **n=5 paired differences, df=4**. This is a very low-power test — the
   reported "NB p=0.134 (n.s.)" for YOLO26 and "NB p=0.0375 (sig)" for YOLO11 are each resting on
   only 5 data points; at df=4 a single atypical fold materially swings both the mean and the
   variance, so the "p=0.0375 (sig)" result for YOLO11 should be treated as fragile, not a stable
   finding, until re-run with more folds.
2. The reported **"80.8 ± 3.1%"** (and the YOLO11/26 equivalents) is `mean(fold_rel) ±
   std(fold_rel)` over only 5 numbers — a population std (ddof=0) over n=5, not a confidence
   interval, and not the corrected variance actually used inside the NB test. It understates true
   uncertainty and should not be read as a rigorous CI in the thesis; a 5-point std is itself a
   noisy estimate of spread.
3. This is precisely the failure mode my original research (Part B, Q1) already flagged generically
   ("repeated stratified k-fold, not a single k-fold run") — it applies directly here.

**Fix (small, additive, reuses all existing code):** wrap the existing per-fold loop in an outer
repeat loop, R=10-20 times, each with a different `random_state` for `StratifiedKFold`, collect all
kR per-fold differences into one array, and:
- feed all kR diffs into `nadeau_bengio` using the **repeated-k-fold form**:
  `t = ( (1/(kR)) Σ_{r,i} x_{r,i} ) / sqrt( (1/(kR) + n2/n1) · σ̂² )`, df = kR−1 (Bouckaert & Frank
  2004 repeated-CV extension of Nadeau-Bengio — same file already imports scipy, this is a ~3-line
  change to accept `n` as `k*R` and `d` as the pooled array; the existing function signature already
  generalizes to this with no logic change, just call it with the longer `diffs` list).
- report mean ± a proper CI (e.g. percentile bootstrap over the kR fold-diffs, or t-interval using
  df=kR-1) instead of the raw 5-point std.
- Cost: ~4-8x the current compute (R=10-20 vs R=1 today) — this is the same tradeoff flagged in
  original Part B (§1): at 5 seeds × 5 folds already run once, going to R=10 with seeds=3 means
  ~10x today's run; if wall-clock-bound on the M4, R=10 (giving df=49) is the recommended minimum
  that meaningfully fixes the df=4 fragility without an unreasonable compute bill; R=20 (df=99) if
  time allows.
- **This is now the single highest-value statistical fix remaining** — same conclusion as before,
  just against the real code instead of a hypothetical design.

### A3. The 175 missing view labels — resolved, concretely, from the data itself

Coordinator asked to check whether 516 = clips with a view label. Verified directly:
`clip_view_map.json` has exactly 516 entries, label distribution {0:210, 1:108, 2:142, 3:56} — this
is **exactly** the 4 unsafe classes' full train+test pool (452 train + 64 test, confirmed by
counting `*.mp4` files per class dir: labels 0-3 sum to 452 train / 64 test = 516). The 4 SAFE
counterpart classes (labels 4-7, dirs `4_safe_walkway`, `5_authorized_intervention`,
`6_closed_panel_cover`, `7_safe_carrying`) are **entirely absent** from the view map: pooled sizes
75/38/32/30 = 175 clips, matching the brief's "175/691 missing a view label" exactly (516+175=691).

**Conclusion: the 175 missing-view clips are not a random scatter inside the unsafe classes — they
are precisely, and only, the 4 safe classes.** The unsafe-class pool used for LCO/k-fold already has
**100% view-label coverage**, so `run_lco_eval.py`'s leave-one-camera-out result is NOT compromised
by the missing-label problem at all — it already correctly uses the complete, gold-labeled 516.

**Recommendation on recovery: do not recover the 175 labels unless the 8-class (safe+unsafe)
task specifically needs cross-view evaluation.** Reasons:
- YAGNI: the paper-comparison task, the LCO headline, and the NB comparison are all defined on the
  4-unsafe-class 516-clip pool, which is already fully labeled. Recovering safe-class labels adds
  engineering risk (silver-label validation, spot-checks) for zero benefit to the numbers already
  reported.
- If an 8-class LCO or 8-class k-fold IS wanted later, recovery is comparatively *low-risk* here
  versus my original generic proposal, because the 8 physical cameras are shared between safe and
  unsafe clips at the same site — nearest-centroid matching of each safe clip's first-frame
  background embedding against the 8 already-known view centroids (built from the 516 gold-labeled
  clips) should be high-accuracy (same cameras, same backgrounds, just a different behaviour in
  frame). Validate with a manual spot-check (e.g., 20-30 of 175) before trusting it for a metric,
  same caveat as originally written.
- **Action: state explicitly in the thesis that LCO/k-fold report on the fully-labeled 516-clip
  unsafe-class pool; the 175 safe-class clips are out of scope for this evaluation axis and their
  missing labels do not affect the reported cross-view numbers.** This is a one-sentence fix to a
  potential reviewer objection, at zero engineering cost.

### A4. One leakage gap not yet closed by the existing work

`run_kfold_relation.py`'s random `StratifiedKFold` is **not** grouped by session — its own code
comment admits this ("random k-fold has every camera in train (scene-confounded)"), which
`run_lco_eval.py` correctly addresses for the *camera* axis. But nothing yet checks for
**within-camera adjacent-clip leakage** (two different clips, same camera, same recording session,
same worker, temporally adjacent) — LCO removes camera-level leakage but a clip and its
near-duplicate neighbour from the same session could still land in different folds of the *same*
camera under either protocol. This is my original Part B §3 concern, now narrowed: it is not fixed
by the existing LCO work and is still open. **Recommended audit** (cheap, ~1-2 hrs, reuses cached
frames): per camera view, compute a first-frame embedding (mean-pooled frozen YOLO feature already
available in this pipeline) for all clips of that view, hierarchically cluster/nearest-neighbour
search for near-duplicate backgrounds (cosine sim > ~0.98), flag clusters as session groups. Rerun
the repeated k-fold (A2 fix) with `StratifiedGroupKFold` using these session groups instead of plain
`StratifiedKFold`, compare resulting accuracy to the ungrouped number — the gap (if any) is the
leakage estimate, report it explicitly. This is the one piece of Part B's original leakage section
still fully applicable and not yet subsumed by existing work.

## PART B — The scene-confound finding: literature grounding

The `run_lco_eval.py` result — scene-only baseline 77.1% (or 81.2% same-dist) collapsing to
16.9% under leave-camera-out, while the relation-Transformer holds 78.6% macro — is a textbook
instance of a well-documented failure mode in action recognition, not an anomaly. Citable grounding:

- **Li, Li & Vasconcelos, "RESOUND: Towards Action Recognition Without Representation Bias" (ECCV
  2018).** Introduces "representation bias": a dataset is biased toward a representation type
  (e.g., static scene) if that representation alone achieves unexpectedly strong accuracy without
  the true discriminative signal (action/behaviour). Proposes the Diving48 dataset specifically
  because prior diving datasets let scene/background predict the dive class. **Directly maps to our
  finding**: our scene-only baseline (majority class per camera view — a pure background proxy) gets
  77-81% same-distribution accuracy despite carrying zero behaviour information — that IS a
  RESOUND-style representation-bias measurement, applied to camera identity instead of scene
  category.
- **Choi, Gao, Messou & Huang, "Why Can't I Dance in the Mall? Learning to Mitigate Scene Bias in
  Action Recognition" (NeurIPS 2019).** Explicitly frames scene as a confound in action recognition
  (e.g., basketball-court scenes predict "playing basketball" independent of the actual action) and
  proposes adversarial scene-debiasing losses. Their evaluation protocol — train on one scene
  distribution, test on a scene-disjoint split — is structurally identical to our leave-one-camera-
  out design; their headline point (models trained/tested with scene overlap inflate accuracy
  relative to scene-disjoint testing) is exactly what our 77.1%→16.9% (scene-only) and 84.0%→78.6%
  (relation-Transformer, smaller drop) demonstrate. [arXiv:1912.05534; github.com/vt-vl-lab/SDN]
- **SCUBA/SCUFO benchmarks** (background/foreground bias evaluation for action recognition on
  Kinetics/UCF101/HMDB51, "Evaluating and Mitigating Static Bias of Action Representations,"
  arXiv:2211.12883) and the broader **"Mitigating Representation Bias in Action Recognition:
  Algorithms and Benchmarks"** survey (arXiv:2209.09393) — establish that scene/background bias is a
  known, actively-benchmarked problem class in mainstream action recognition, not specific to
  small/idiosyncratic datasets. Useful to cite for "this is a recognised failure mode across the
  field, we independently reproduced it in a new domain (industrial safety CCTV)."
- **How to phrase the thesis contribution**: the LCO experiment is a **domain-specific instance of a
  known bias-detection protocol** (train/test on disjoint viewpoints/scenes, per Choi et al./RESOUND
  methodology), applied for the first time (to our knowledge — not independently verified in this
  search) to industrial-safety CCTV behaviour classification, where it reveals the paper's own
  same-distribution protocol (and ours, under naive random k-fold) rewards camera memorisation over
  behaviour recognition. This reframes the LCO result as *methodologically grounded*, not an
  isolated ad hoc check.

## PART C — Remaining research, re-ranked against what already exists

Everything below was NOT found to already exist in the codebase/reports (checked
`plans/2026-09-02-fair-headtohead-vs-paper/`, `run_kfold_relation.py`, `run_lco_eval.py`,
`eval_utils/`, `unsafenet/`). Kept from the original pass, condensed, re-ranked.

### C1. Repeat the existing NB test properly (see A2) — do this first, cheapest high-value fix
Already covered above; restated here only for the ranking table.

### C2. Session-leakage audit (see A4) — second priority
Already covered above (A4); ~1-2 hrs, protects validity of the A2 fix's numbers.

### C3. Data-efficiency / learning curve — still not done, still high value/hour
Not present anywhere in the existing runs. Standard recipe: train on {25,50,75,100}% of the 452
unsafe-train pool, R≥10 repeats per size (reuse the same repeated-fold infrastructure from A2), plot
mean macro-accuracy ± bootstrap CI vs size, fit power law `acc(n) = c − a·n^b` (b∈[-0.5,0]) via
`scipy.curve_fit`, extrapolate toward paper-scale training volume. This directly quantifies the
existing "data-bound not method-bound" claim (§4.6.2) instead of asserting it qualitatively. Cheapest
way to turn an existing narrative into a number — reuses 100% of existing training code, no new
protocol design needed beyond a train-set-size filter. [Hendrycks/Cortes-line power-law learning-
curve literature; arXiv:2010.11029; arXiv:2103.10948 review; arXiv:2311.18025 small-pilot
extrapolation — unchanged from original research pass]

### C4. Robustness stress tests — still not done, medium value/hour
Not present in existing runs (existing work is entirely about split protocol, not input corruption).
Recipe unchanged from original pass: frame-rate reduction (32→16→8), resolution downsample
(960×540→480×270→240×135), H.264/JPEG compression ladders (CRF 23/30/37/44/51 or JPEG q
90/70/50/30/10), motion blur (kernel 3-21px), low-light (gamma/brightness scale 0.8→0.1), partial
occlusion (5-50% area cutout), camera shift (small affine jitter) — all inference-only on the
existing cached-frame pipeline, no retraining. Report accuracy-vs-severity per corruption + a single
mean-relative-accuracy-drop summary (Hendrycks & Dietterich 2019, arXiv:1903.12261, methodology only,
skip their AlexNet-normalized mCE since no reference model exists here). Lower priority than C3
because it doesn't speak to the paper-comparison headline, more of a deployment-realism bonus.

### C5. External validation — still not viable, confirmed, deprioritise further
No taxonomy-matched public dataset exists (checked: our own Onal & Dandil set is already public on
Mendeley/HuggingFace — not "external"; STR-Transformer's 1,595-clip 7-category set has no confirmed
public download; CHAD is public but parking-lot anomaly domain, no taxonomy overlap; PPE/Roboflow
sets are image-level detection, wrong granularity). **Given LCO already exists and delivers a
stronger, in-house generalisation story (scene-confound proof), external validation is now even
lower priority than originally assessed** — LCO subsumes the "does this generalise beyond the
training distribution" question more rigorously than a domain-mismatched external dataset would.
Recommend dropping this axis entirely from the remaining work plan.

### C6. Reporting checklist citations — unchanged, cheap, do alongside writeup
REFORMS (Kapoor et al., Science Advances 2024, arXiv:2308.07832, 32-item ML-science checklist),
TRIPOD+AI (2024, PMC11019967), Cawley & Talbot (JMLR 2010, model-selection bias — cite specifically
re: A2's "no further tuning after freeze" and the blind-protocol already in place), Agarwal et al.
2021 (arXiv:2108.13264, small-N CI precedent, already cited in
`research/small-n-statistics-and-metrics.md`). No compute cost — just citation work during writeup.

## Ranking: evidential value per hour of work (revised)

1. **A2 — repeated k-fold + correct NB df (was already 80% built).** Cheapest remaining fix with
   the highest payoff: turns a fragile df=4 result into a properly-powered df=49-99 one, using code
   that already exists almost verbatim. Do this before anything else.
2. **A4 — session-leakage audit.** ~1-2 hrs, protects A2's numbers from a leakage mode neither
   existing script checks (they check camera leakage via LCO, not within-camera session leakage).
3. **C3 — learning curve.** Reuses 100% of existing training pipeline; converts an existing
   qualitative claim into the strongest possible quantitative figure.
4. **Part B literature grounding for LCO** — near-zero cost (already-run experiment, just needs
   citations), meaningfully upgrades the LCO section from "we noticed this" to "this is the field's
   known failure mode, independently confirmed in a new domain."
5. **C4 — robustness stress tests.** Cheap to run, but secondary to the paper-comparison and
   scene-confound narratives that already exist and are now the thesis's strongest material.
6. **C5 — external validation.** Drop. LCO already provides better generalisation evidence than any
   obtainable external dataset would, at far lower cost/risk.

## Unresolved questions

- Exact number of repeats (R) for A2 bounded by wall-clock on the M4 — R=10 (df=49) recommended
  minimum, R=20 (df=99) if time allows; not empirically timed here, T4 should benchmark one repeat's
  wall-clock first before committing to R.
- Whether `StratifiedGroupKFold` (needed for the A4 fix) can maintain adequate class balance once
  grouped by session, given group sizes are unknown until the clustering audit (A4) is run — verify
  empirically once session groups exist, not assumed here.
- Whether the near-duplicate/session-clustering approach (A4) will find any sessions at all — it is
  possible the dataset's clips are already suf1ficiently separated in time/session that grouping
  changes nothing; the audit could come back negative, which is itself a useful (cheap) result to
  report ("we checked, no session leakage found").
- 8-class safe/unsafe view-label recovery (A3, if ever pursued): no canonical citable method found
  for automatic CCTV camera-ID recovery via background clustering; would need to be described as a
  plain engineering method, not a citable technique, same caveat as the original research pass.
- `baycomp` (Bayesian correlated t-test) was researched as a secondary statistic option in the
  original pass but not checked against what's already implemented — if a second statistic beyond
  NB is still wanted, verify `baycomp`'s current maintenance/Python-compatibility before adopting;
  not otherwise blocking since NB is already correctly implemented and is sufficient on its own.
