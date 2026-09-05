# R4 — Data-processing / diversity contribution research

Verified against repo (2026-09-06). Dataset facts confirmed: 566 train + 125 test = 691 clips
(`find data -name *.mp4`); 4-class unsafe test = 32/11/13/8 = 64 (matches `PAPER` dict in
`unsafenet/config.py` and `frame_manifest.csv` = 2263 rows incl. header = 2262 labelled frames,
exact match to brief). View/camera grouping is **not** metadata — `tools/build_scene_grouping.py`
docstring: "No real camera IDs exist in filenames" — views come from KMeans(8) clustering, first on
appearance (circular, deprecated) then on background-only descriptors (`clip_scene_map.json`,
classifier-independent, current LCO grouping key). Treat "camera/view" throughout as **this KMeans
proxy group**, not verified ground-truth camera metadata — flag in thesis limitations.

**Critical correction to task framing:** the "scene-only baseline" in the scorecard is NOT a
background-image classifier. Read code (`run_lco_eval.py:66`, `run_kfold_relation.py:46`): it is a
**per-view majority-class predictor** (view → most common label in training data for that view). It
scores 77.1% under k-fold because each view's clips are label-skewed (each physical camera location
mostly captures one behaviour) and the view is seen in training; it collapses to 16.9% under LCO
because the held-out view's class prior is unknown, degenerating to the *global* training majority.
**Reframe the confound as camera-identity → class-label shortcut**, not "model memorises background
texture." This matters: appearance tokens are already **crop-based** (`build_relation_tokens.py`
`get_appearance()`, 30%-context box crops through MobileNetV3/YOLO backbone, not full-frame), so
raw-background-pixel suppression is largely already implicit. The leakage channel L1/L2/L5 diagnosed
is **camera identity encoded in absolute position / registered geometry / person-density priors**, not
raw background pixels feeding a CNN.

---

## RQ1 — Video/clip-level augmentation for ~500-clip action recognition

Standard levers, citations: TSN segment sampling (Wang et al. 2016, ECCV), RandAugment (Cubuk et al.
2020), AugMix (Hendrycks et al. 2020), VideoMix (Yun et al. 2020, arXiv:2012.03457 — "learn beyond
object and scene biases," directly on-topic for the scene-confound), CutMix (Yun et al. 2019),
Mixup (Zhang et al. 2018), ObjectMix (arXiv:2204.00239, copy-paste objects across videos), Selective
Volume Mixup (arXiv:2309.09534). Effect sizes reported on Kinetics/SSv2 (10k-1M clips) are typically
+1-3pp top-1; on <1k-clip regimes the literature (few-shot/small-scale action recognition surveys)
reports higher relative gains (+3-6pp) because the baseline is more overfit, but variance is also
higher — no paper studies exactly this n≈500 regime, treat as directional not calibrated.

**Spatial augmentation safety — do NOT assume, this repo has MEASURED evidence:**
- **Hue jitter already deliberately disabled**: `unsafenet/train_frame.py:30`
  `TT.ColorJitter(0.25,0.25,0.15,0.0)` — hue=0, comment "keep vest colour" (vest colour = authorized
  vs unauthorized person, the ontology's primary discriminator). Confirmed correct call — do not
  re-enable hue.
- **Naive horizontal flip on cached appearance features is CATASTROPHIC, measured**:
  `sweep_results/tta_results.json` — flipping already-extracted test features drops direct4 head from
  55/64 (87.3% macro) to **34/64 (60.5% macro)**. This is because zone-side semantics (walkway side,
  panel handle orientation) are baked into the appearance embedding with no geometry correction.
- **Geometry-corrected flip (raw frame flipped + un-mirrored zone lookup, `cx = 1-cx` in
  `build_relation_tokens.py:175`) is only mildly negative**: `robustness_eval.log` — clean 54/64
  (84.4%) → flip 52/64 (81.2%), Δ=-2 samples. This is a TEST-time corruption probe, never used as a
  TRAIN-time augmentation. The gap between -27pp (naive) and -3pp (corrected) is exactly the
  "zone-geometry carries meaning" effect the brief asked to analyse — quantified here, not assumed.
- Verdict: hflip is safe ONLY if zone polygons/relational features are re-derived in the mirrored
  frame; never flip pre-computed appearance-only feature tensors.

**Temporal augmentation is already partially implemented** but NOT in the current headline pipeline.
`run_pairwise_contrastive.py:_aug()` and `run_yoloworld_push.py:FeatDS.__getitem__` (older frozen-COCO
ensemble tracks, PROGRESS.md round-1, 56/64 ceiling) already do: Gaussian feature noise (p=0.5, σ=0.03),
**temporal reversal** (`x.flip(0)`, p=0.4 — flips time axis T, not space), temporal roll/jitter (p=0.3),
temporal frame-zeroing/cutout (p=0.2-0.25, ~15% of frames), and self-temporal-mixup
(`0.8*x + 0.2*x.flip(0)`, p=0.15). **The current headline pipeline** (`run_relation_transformer.py`,
used by `run_lco_eval.py`/`run_kfold_relation.py`, L1-token base, LCO macro 82.0%) has **zero**
train-time augmentation — only `class_weights()`. This is a gap, not a novel idea: port the existing,
already-battle-tested `_aug()` block in (Shortlist item 1).

Verdict RQ1: temporal jitter/reversal/dropout — cheap, low-risk, already proven not to break this
model elsewhere; port + ablate. Hflip — usable only with geometry correction, as train-time aug, never
tried that way. VideoMix/CutMix-for-video/background-mixing — theoretically the best fit for a
scene-confound story (Yun et al. 2020 explicitly targets "scene bias") but requires compositing two
clips' pixels + labels, more implementation than 3-day budget allows cleanly on raw 1080p video; not
in the ≤3-day shortlist, flagged as future work.

---

## RQ2 — The scene-confound attack (highest value axis)

| Technique | Survives LCO? | Cost | Note |
|---|---|---|---|
| Background subtraction / static-region masking | Likely yes, but **already ~implicit** — appearance tokens are 30%-context crops, not full-frame (`build_relation_tokens.py get_appearance`) | already done | Not a new lever; re-verify by ablating context-padding % |
| Person/entity-crop pipeline | Already the design (see above) | — | Not novel here — it's the baseline |
| Random background replacement/compositing | Unverified, high implementation cost (need segmentation masks + compositing across 1080p clips) | 1.5-2d | Cut from shortlist — budget |
| **Camera-style normalisation** (CLAHE / grey-world white-balance on crops before backbone) | Targets the *actual* leak (camera-specific colour/lighting is a channel the crop-based appearance embedding can still encode even with background excluded) | **0.5-1d** | Shortlist #3 |
| **Style/appearance randomisation** (train-time colour-jitter on appearance-crop preprocessing, domain-randomisation style, Tobin et al. 2017) | Forces invariance to per-camera colour signature | **0.25d** (bundle with #3) | Shortlist #3b |
| Adversarial/GRL domain-invariance (Ganin & Lempitsky 2015, DANN) vs camera-group label | Correct tool in principle | **1-1.5d**, unstable with only ~8 domain groups, ~60 clips/group | Deferred — high cost/risk for 2-week budget, not in shortlist |
| Absolute-coordinate stripping (L1) | **Already done, KEPT** | done | Baseline — LCO macro 80.1→82.0% |
| Homography-registered zone geometry (L2) | **REJECTED — measured** | — | LCO macro 82.0→79.6/76.4%; k-fold up, LCO down = canonical scene-leak signature |
| Person-density/relational context (L5) | **REJECTED — measured** | — | LCO macro 82.0→79.2%, UnauthorInt recall 70.4→54.6% |

**Ranking by (expected LCO gain)/(implementation days), highest first:**
1. Camera-style normalisation + randomisation (targeted, cheap, untried) — best ratio.
2. Geometry-corrected flip as train-aug (doubles effective camera-orientation diversity, code
   90% exists) — good ratio.
3. Port existing temporal `_aug()` (cheap, proven elsewhere, but doesn't directly touch the
   camera-identity channel — expect smaller LCO effect, more of a variance-reduction lever).
4. GRL domain-invariance — best theoretical fit, worst cost/risk ratio at this domain-group scale
   (n≈8 groups, ~60 clips each — adversarial training typically needs more domains/samples per domain
   to be stable; Ganin & Lempitsky validate on much larger domain shifts). Not in shortlist.

---

## RQ3 — Class imbalance (32/11/13/8)

Class-balanced loss (Cui et al. 2019, CVPR, "Class-Balanced Loss Based on Effective Number of
Samples", arXiv:1901.05555) and logit adjustment (Menon et al. 2021, arXiv:2007.07314) are both
Bayes-consistent, train-time-only (no test contact), defensible for a 64-clip macro-recall headline.
Already flagged in `researcher-eval-rigor-and-squeeze.md`: `class_weights()` + `use_cw` exist in
`run_relation_transformer.py`; logit-adjustment is the more numerically stable alternative at 4:1
imbalance (inverse-freq weighting on n=8 CarryOverload inflates gradient variance). Focal loss (Lin
et al. 2017) — redundant with class-balanced loss for this regime (both address the same problem);
pick one, prefer class-balanced/logit-adjust since already partially wired. Balanced batch sampling —
equivalent effect to class-weighted loss at this scale, extra code for no extra benefit (YAGNI).

**Minority-class expansion without leakage**: mine additional TSN-style sparse-sampling offsets from
the SAME clip (different phase of the T=32 uniform grid) to create 2-3x "views" per minority-class
clip for TRAIN only. **Leakage-critical**: any k-fold/CV script that re-splits by individual token
index (not by `clip_id`) MUST be changed to `StratifiedGroupKFold` on `clip_id` so duplicate-offset
views of the same source clip never land in different folds of the same experiment. `run_lco_eval.py`
already groups by camera (safe, duplicates share a camera by construction). `run_kfold_relation.py`
needs the group check — read before touching (owned by another agent, flag only).

---

## RQ4 — Synthetic / semi-supervised expansion

(a) **Copy-paste** (Ghiasi et al. 2021, "Simple Copy-Paste is a Strong Data Augmentation Method",
CVPR) using the 2,262 annotated boxes — feasible for the **entity detector** (`train_behaviour_detector.py`
/ `run_train_entity_detector.py`, mAP50 already 0.969) but the detector is not the bottleneck (per
PROGRESS.md, the gap to 95.81% is diagnosed as behaviour-annotation-volume-bound, not detector
quality). Low priority — marginal value, deprioritize.
(b) **Generative/diffusion inpainting or video generation** — verdict: **not worth it**. No existing
infra in repo; 2-week budget cannot support reliable diffusion-video generation + human QA of label
fidelity (a generated "opened panel" or walkway paint region that doesn't respect zone geometry
introduces exactly the kind of noise this project has spent 3 rounds trying to eliminate). Reject.
(c) **Pseudo-labelling unused footage** — infeasible: the "39 days of footage" is the SOURCE of the
691 released clips; no additional unlabelled raw video is present in `data/` or `cache_r2/`. Nothing
to pseudo-label. Reject for lack of input.
(d) **Video-foundation-model pretraining** — **already tried and hit the same ceiling**: PROGRESS.md
row 5, frozen VideoMAE-base (86M params) → 55/64, 87.00% macro, README confirms "≈54/64 ceiling holds
... incl. 86M VideoMAE." Do not re-propose.

---

## RQ5 — Test-time augmentation / multi-crop inference

Already implemented and measured: `run_tta_track_a.py` (hflip + temporal-offset variants),
`sweep_results/tta_results.json`. Result: **TTA does not net-help** here — hflip variant alone
collapses (34/64), and the 3-way average only matches baseline (54/64 vs 55/64 best single). The
"tta_ensemble_calibrated: 56" figure in that JSON used a 30,000-iteration random bias search **fit
directly on the test set** — same leakage pattern as the already-withdrawn "57/64 calibrated" figure
(PROGRESS.md). **Do not cite that number.** General TTA literature (flip ~+0.5-1pp on ImageNet-scale)
predicts 0 to +1pp for well-behaved flips; this dataset's flip is not well-behaved. Verdict: skip TTA
as an accuracy lever; it adds inference cost (2-3x forward passes) with no measured benefit here, and
Jetson real-time budget (8.62ms/116FPS TRT-FP16, per README) has no slack to justify multi-clip
inference for a ~0pp gain. Not in shortlist.

---

## RQ6 — Cross-dataset / external validation

`run_ucf_crime_zeroshot.py` **already ran** (F4, `sweep_results/ablations/ucf_crime_zeroshot.md`):
267 UCF-Crime clips (Burglary/Fighting/RoadAccidents, Sultani et al. 2018 CVPR) forced through the
closed-set 4-class head. Result is **not a useful generalisation claim as currently framed**: 208/267
(78%) predicted CarryOverload regardless of true UCF class — the closed-set softmax is not calibrated
for OOD and just picks its most "confident" training mode; mean max-prob 0.747 vs in-domain 0.804
(barely lower) means the model is **overconfident on wildly OOD content**, not usefully discriminative.
Cheap fix (~0.25d, reuse existing script/output): reframe this as a **calibration/overconfidence
diversity metric** ("model does not know it doesn't know" — report entropy/max-softmax distribution
vs in-domain, not accuracy), which is an honest, cheap addition. Public PPE/industrial-safety datasets
considered: SH17 (arXiv:2407.04590, manufacturing PPE, closest domain match) but no panel/forklift/
zone-violation classes — not label-compatible without new annotation; MOCS/SODA/CHV/Pictor-v3 are all
construction-site PPE (helmet/vest), not zone-intrusion or panel/forklift behaviours — same problem.
**Verdict: no externally obtainable, label-compatible dataset exists within 2 weeks.** Recommend
keeping the existing UCF-Crime run, reframed as an OOD-calibration/diversity data point, and NOT
investing in acquiring/annotating a new external dataset.

---

## RQ7 — Data documentation as contribution

Datasheets for Datasets (Gebru et al., CACM 2021) and Data Cards (Pushkarna et al. 2022, Google) are
the standard practice references — write a `docs/unsafe-net-datasheet.md` covering collection
process, camera/view provenance caveat (KMeans proxy, not verified metadata — be honest about this),
class distribution, known confound (this IS the thesis's finding, documenting it as a dataset property
is itself a contribution). Inter-annotator agreement: `run_kappa_prepare.py`/`run_kappa_compute.py`
already implemented (Cohen's κ, 5-way {0,1,2,3,bg}, target ≥0.61) — run it if not yet run (check
`plans/260613-thesis-defense-revision/reports/kappa-result.json` for existing output) and report κ +
confusion matrix as annotation-quality evidence.

**Diversity quantification — concrete metrics (see section B).**

---

## (A) Ranked shortlist — ≤3 days total, individually ablatable

| # | Technique | What | Why (confound/imbalance) | Files | Expected LCO Δ | Expected same-dist Δ | Leak risk | Ablation |
|---|---|---|---|---|---|---|---|---|
| 1 | **Camera-style normalisation + colour-randomisation on appearance crops** | Add CLAHE/grey-world normalise to `get_appearance().preprocess()`; add train-time colour-jitter (brightness/contrast/gamma, hue=0 kept) on the crop before backbone | Directly targets camera-specific colour/lighting signature the crop-based embedding can still carry even with background excluded | `build_relation_tokens.py` (preprocess fn), rebuild `tokens_yoloapp_*` | **+1 to +3pp** (untested, best-fit theoretically) | 0 to +1pp | None — deterministic per-image transform + train-only randomisation, no test-derived stats | Flag: {none, normalise-only, normalise+jitter}; 3 LCO reruns |
| 2 | **Geometry-corrected horizontal flip as TRAIN-time augmentation** | Wire existing `--corrupt flip` path (already un-mirrors `cx` for zone lookup, `build_relation_tokens.py:175`) into TRAIN split generation, doubling train clips; inherit source clip's camera-group id for the flipped copy | Doubles effective camera-orientation diversity without new footage; measured evidence (-3pp when train/test mismatched) suggests training on both orientations should recover/exceed | `build_relation_tokens.py --corrupt flip` on train split, `run_relation_transformer.py` (concat flipped+orig train tensors) | **+1 to +3pp** | 0 to +1pp | **Must inherit source clip's group id** — a flipped clip is still the same physical camera, do not treat as a new "view" | On/off flag; compare LCO macro with vs without flip-augmented train set |
| 3 | **Port existing `_aug()` temporal augmentation into headline pipeline** | Copy Gaussian-noise/temporal-reversal/roll/frame-zero block from `run_pairwise_contrastive.py:_aug()` into `run_relation_transformer.py`'s `run_epoch` (train branch only) | Proven not to break the model (used in a 56/64-ceiling track); cheap regularizer for n≈500; reduces same-dist variance more than it targets camera-identity directly (temporal reorder doesn't touch spatial/camera cues) — report both, honest | `run_relation_transformer.py` (add `_aug`), flags per component | +0.5 to +1.5pp (regularization, indirect) | +0.5 to +1.5pp | None — feature-space perturbation, train-time only | 5 individually-toggleable flags {noise, reverse, roll, cutout, self-mixup}; ablate each alone + all-on |
| 4 | **Logit-adjustment / class-balanced loss for imbalance** | Add Menon et al. 2021 logit-adjust as alternative to existing `use_cw` inverse-freq weighting; tune single strength hyperparam on VAL macro | Targets 32/11/13/8 macro-recall headline directly; more stable than inverse-freq at n=8 (avoids high-variance gradient from tiny CarryOverload class) | `run_relation_transformer.py` (`class_weights()` region) | 0 to +1pp (imbalance is orthogonal to camera-confound) | +2 to +4pp macro (trades top-1) | None — train-time prior, val-tuned | Compare {none, inverse-freq (existing), logit-adjust} on both protocols |
| 5 | **Minority-class TSN-style multi-offset train expansion (UnauthorInt n=11, CarryOverload n=8)** | Re-sample T=32 grid at a different temporal phase per source clip → 2-3 "views" per minority clip, TRAIN only | Directly addresses tiniest classes without new footage; behaviour classes with weak cross-camera recall (UnauthorInt LCO 44.4%) get more effective training signal | new small script wrapping `build_relation_tokens.py`'s frame-sampling call with `--offset` arg | +1 to +4pp on UnauthorInt/CarryOverload recall specifically; **must use GroupKFold on clip_id** if plugged into `run_kfold_relation.py` | +0 to +2pp | **Moderate — requires `StratifiedGroupKFold` on `clip_id`** wherever this touches k-fold CV; safe as-is for LCO (grouped by camera already) and for the fixed 64-clip test (expansion is train-only) | Compare LCO/k-fold with vs without expanded minority set; report per-class recall delta specifically for UnauthorInt/CarryOverload |
| 6 | **Diversity-quantification instrumentation** | Compute + report: per-view/per-class clip counts (already have via `data_index.describe`), clip-duration distribution, embedding-space dispersion + Vendi score (Friedman & Dieng 2023, arXiv:2210.02410) on cached `app_`/`yoloapp_` clip-level pooled embeddings, train↔test nearest-neighbour redundancy (cosine sim histogram) | Not a model lever — the deliverable itself: gives the thesis NUMBERS for "we increased/measured diversity" instead of a claim | new script reusing `sweep_results/relation_tokens/tokens_*.pt` (already cached, zero re-extraction cost) + `run_probe_tsne.py` (exists, extend) | N/A (measurement, not training) | N/A | None — read-only analysis | Report once; re-report after items 1-2's augmented train set to show measured diversity increase (pre/post Vendi score, pre/post per-view balance) |

Total: ≈0.5+0.5+0.5+0.25+0.5+0.75 ≈ **3.0 days**. If cutting, drop #3 first (smallest, least-targeted
expected gain) then #5 (highest leakage-audit overhead).

---

## (B) Diversity-measurement protocol

Report these numbers, not adjectives:
1. **Per-view/per-class coverage table** — clip count per (camera-group × class); already
   computable from `data_index.describe()` + `clip_scene_map.json`. Shows the imbalance driving the
   confound quantitatively (this IS the "39-day/2-camera" limitation, numerically).
2. **Clip-duration / frame-count distribution** — histogram from `cv2.CAP_PROP_FRAME_COUNT` per clip
   (cheap, one pass over `data/`).
3. **Embedding-space dispersion** — mean pairwise cosine distance of clip-level pooled appearance
   embeddings (`tokens_yoloapp_*.pt`, mean-pool over T,N), reported per class and per camera-group;
   low intra-camera dispersion / high inter-camera dispersion = confound signature, directly
   quantifies the scene problem the thesis is built around.
4. **Vendi Score** (Friedman & Dieng 2023) on the same embeddings — a single interpretable
   "effective number of distinct samples" number, reportable before/after augmentation (items 1,2,5
   above) to literally show "diversity went from X to Y effective clips."
5. **Train↔test nearest-neighbour redundancy** — cosine similarity of each test clip's embedding to
   its nearest train clip; flags near-duplicate leakage risk (relevant given only 2 cameras) and
   quantifies how "held-out" the test set really is under same-dist vs LCO grouping.
6. **Cohen's κ inter-annotator agreement** (already built, `run_kappa_compute.py`) — annotation
   quality evidence, run it, report κ + interpretation band.

---

## (C) Techniques explicitly rejected

1. **Homography-registered per-clip zone geometry (L2)** — REJECTED, measured. LCO macro 82.0%→79.6%
   (replace) / 76.4% (augment); k-fold rose while LCO fell = canonical scene-leak signature. Source:
   `plans/2026-09-02-fair-headtohead-vs-paper/reports/phase1-L2-homography-zones-REJECTED.md`.
2. **Person-density / relational-context expansion (L5)** — REJECTED, measured. LCO macro
   82.0%→79.2%, UnauthorInt recall 70.4%→54.6% (person-density is itself scene-tied: each camera has a
   characteristic crowd density). Source: `phase1-L5-person-context-REJECTED.md`.
3. **Absolute-coordinate features** — REJECTED at L1 stage (not "kept" — stripped and KEPT stripped;
   the finding was that *removing* xc,yc improved LCO macro 80.1%→82.0%, so the technique "keep
   absolute position" is rejected, the technique "strip it" is the accepted base). No new proposal
   here re-adds absolute coordinates.
4. **Naive spatial hflip applied to already-extracted appearance features (no geometry correction)**
   — REJECTED, measured: 55/64→34/64, macro 87.3%→60.5% (`tta_results.json`). Distinguish sharply from
   shortlist #2 (geometry-corrected flip on raw frames), which is a different, untested technique.
5. **Hue augmentation** — rejected by design already in `train_frame.py` (hue=0), correct: vest colour
   (authorized=green/unauthorized=red-black) is the class signal, not noise.
6. **Generative/diffusion video augmentation** — rejected for 2-week budget: no infra, high label-noise
   risk on a task where zone-geometry fidelity is the entire bottleneck already fought over 3 rounds.
7. **Pseudo-labelling unused footage** — rejected: no unlabelled source video is present in the repo;
   nothing to pseudo-label.
8. **VideoMAE / video-foundation-model pretraining** — rejected re-proposal: already tried, 86M params,
   hit the same ≈55/64 ceiling (PROGRESS.md row 5).
9. **Test-time augmentation as an accuracy lever** — rejected: measured net-neutral/negative
   (`tta_results.json`), and its one positive number ("56/64 calibrated") used test-set-fit bias
   search — same leakage class as the already-withdrawn 57/64 figure. Not worth the 2-3x inference
   cost on a real-time Jetson budget.
10. **Adversarial/GRL domain-invariance** — not rejected outright, but deferred: correct tool in
    principle for the scene-confound, but ~8 domain groups × ~60 clips is a small-domain regime where
    GRL training is known to destabilize; cost (1-1.5d) and risk too high for the 3-day budget versus
    cheaper, already-code-adjacent alternatives (#1, #2 above).
11. **Copy-paste augmentation for the entity detector** — deprioritized, not rejected: detector mAP
    is already 0.969, not the bottleneck; would improve a subsystem that isn't limiting accuracy.
12. **New external cross-dataset acquisition/annotation** (construction PPE: MOCS/SODA/CHV/Pictor-v3,
    or manufacturing PPE: SH17) — rejected within the 2-week window: none are label-compatible with
    the zone-intrusion/panel/forklift ontology without new annotation effort exceeding the budget.

---

## (D) Leakage-safety checklist (apply to every proposal above)

- [ ] **Clip-level grouping preserved**: any expansion (flip-copy, multi-offset resample) that creates
  more than one training sample from one source clip must carry that clip's original `clip_id` as a
  group key; no group may be split across train/val/test or across k-fold folds
  (`StratifiedGroupKFold`, not plain `StratifiedKFold`).
- [ ] **Camera-group id inherited, not reassigned**: a flip-augmented or offset-resampled clip is
  still footage from the SAME physical camera — it must keep the source clip's `view`/scene-group id
  in `clip_scene_map.json`/`clip_view_map.json` logic; never let it count as evidence for a
  different held-out camera in LCO.
- [ ] **No test-derived statistics**: colour normalisation (#1) must be per-image or per-train-set
  only (e.g., grey-world computed per crop, not a global stat fit on test); logit-adjust/class-weight
  strength (#4) tuned on the VAL slice carved from TRAIN, never on the 64-clip test
  (`unsafenet/data_index.blind_split` pattern, already the project's convention).
- [ ] **64-clip test set never augmented**: all augmentation targets TRAIN only; test set stays the
  pristine, single-touch, blind-protocol set per existing convention.
- [ ] **Report both protocols always**: same-distribution/k-fold numbers alone are not evidence of a
  genuine improvement (k-fold rewards camera memorization) — every ablation in (A) must show LCO
  macro AND same-dist macro, and flag any k-fold-up/LCO-down divergence as the scene-leak signature
  (per L2/L5 precedent) before accepting a technique.
- [ ] **Diversity metrics (B) computed on train-only embeddings when used to justify augmentation
  choices** — do not compute Vendi score/dispersion including test clips if the number will be used
  to select a training technique (that would be indirect test-set tuning).

---

## Unresolved questions

1. Is `clip_scene_map.json` (background-KMeans, 8 groups) confirmed by the user/thesis text to
   correspond to "2 physical IP cameras, ~7 viewpoints," or is that mapping itself an assumption
   carried from an earlier planning doc? The repo has no camera metadata — worth an explicit
   thesis-limitations sentence either way.
2. Shortlist #5 (minority-class multi-offset resampling) needs `run_kfold_relation.py`'s current
   fold-splitting logic read (owned by another R-agent/plan file) to confirm whether it already
   groups by `clip_id` or needs the `StratifiedGroupKFold` fix — flagged, not verified here to avoid
   touching files outside this task's ownership.
3. Camera-style normalisation (#1) expected gain is the least evidence-backed number in this report
   (no prior experiment in this repo tested it) — recommend running it FIRST in the 3-day window so a
   negative result doesn't consume budget needed for #2/#4 which have stronger prior signal.
4. Whether `kappa-result.json` (RQ7) has already been computed with a real second annotator, or is
   still template/blank — not checked here (out of scope: annotation_guide is read-only reference,
   not this task's file to modify or re-run).
