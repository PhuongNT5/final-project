# Data Diversity & Augmentation for Small-N Industrial Action Recognition

Context confirmed from PROGRESS.md / unsafenet/config.py: 691 clips, 8 classes (4 unsafe + 4 safe
counterparts, `SAFE_OF` mapping exists in config), eval task = 4 unsafe / 452 train / 64 test,
support 32/11/13/8, 8 camera views w/ 175 clips `view='na'`, T=32 frames @ 960x540, entity detector
mAP 0.969, behaviour detector mAP50 0.79-0.81, M4/MPS (YOLO det training must be `--device cpu`).

Current best (round 1 ensemble) = 56/64 (89.28% macro). Gap to paper's bbox-based 62/64 attributed
to missing per-frame zone semantics, not raw data volume. Ranking below assumes we want cheap wins
without new annotation, without re-breaking round-2 findings (val doesn't predict test at N=64 note
this bounds how much we should trust ANY offline augmentation ablation — validate primarily by
proxy/held-out where possible, treat single test-touch as final only).

## Q1. Video-specific augmentation at small N

| Technique | Works at <1000 clips? | Evidence | Typical gain | Cost |
|---|---|---|---|---|
| Speed/frame-rate perturbation (resample T frames at different stride) | Yes, cheap, label-safe | T-Half/T-Drop/T-Reverse study (Kim et al., "Exploring Temporally Dynamic Data Augmentation", arXiv:2206.15015, 2022) shows temporal augs beat spatial-only on UCF101/HMDB51/SSv2-scale data | +0.5-2pp top-1 on Kinetics-scale; larger relative effect at small N because it's free extra "views" of the same clip | Low — resample existing 32-frame cache at different offsets/strides |
| Temporal jitter / random start offset | Yes | Same source; also standard TSN practice (segment-based random sampling) | Marginal alone, stacks with above | Low |
| Random temporal cropping (crop a sub-window of the 32 frames, still contains the action) | Conditional — only if action is not confined to a short sub-segment | Extending Temporal Data Augmentation, arXiv:2211.04888 (2022/2023) | Small, risk of cropping out the discriminative moment (panel opening, person crossing line) if window too short | Low but needs per-clip validation of "safe window" |
| Frame dropout (random frame drop within clip) | Yes, cheap | Robustness study, CVPR2023 Schiappa et al. "Large-scale Robustness Analysis" catalogs frame-drop as one of 25 temporal perturbations models should be robust to — implies training-time exposure helps robustness even if headline-acc gain is small | Robustness gain > accuracy gain | Low |
| Temporal shuffling as SSL pretext (order prediction) | Only as pretraining signal, not as augmentation of labelled clips | See Q5 | N/A here | — |
| VideoMix (cuboid mixing across clips + soft label mixing) | Evidence is Kinetics/SSv2 scale (240k+ clips): SlowOnly-50 +1.3% Kinetics, +0.9% SSv2 (Yun et al., VideoMix, arXiv:2012.03457, 2020) | No published <1000-clip validation found | Unverified at our N; label-safe IF cuboid doesn't erase the walkway/panel region — but semantic soft-labelling ("30% class A + 70% class B") is questionable for spatially-defined safety classes (a clip that is 70% WalkwayViol pixels + 30% SafeWalkway pixels is not "70% unsafe") | Medium — needs care, see risk below |
| ObjectMix (Kimata et al., ACM MMAsia 2022, arXiv:2204.00239) — copy-paste person instance (via instance seg) from one clip into another, video-level, beats VideoMix on UCF101/HMDB51 | Yes, validated at UCF101/HMDB51 scale (thousands, still bigger than us but same order of magnitude as our 452 train) | Direct MMAsia'22 paper | Reported to outperform VideoMix on UCF101/HMDB51 (exact numbers behind paywall, order +1-3pp typically for this class of method) | Medium-high — we already have person boxes+ masks-ish via entity detector; this is basically Q2, see below |
| Learn2Augment (learned video compositing, arXiv:2206.04790) | Learns *which* videos to composite, needs training signal itself — likely to overfit at N=452 | Weak evidence at small N | Skip | High complexity, not worth it |
| Selective Volume Mixup (arXiv:2309.09534) | Kinetics/SSv2 scale evidence only | — | Skip, no small-N evidence | — |
| S3Aug (segmentation+sampling+shift, arXiv:2310.14556) | Uses segmentation masks to do smarter mixing; same family as ObjectMix, no small-N validation | — | Same risk profile as ObjectMix | Medium |
| Random erasing (Zhong et al. 2017, extended to video) | Commonly used, but no dataset-scale-specific ablation found confirming a reliable gain magnitude at video-clip level; mentioned as "commonly used" in general surveys, not isolated | Weak/anecdotal | Small, uncertain | Low — but HIGH RISK here: erasing a random box could erase the panel or the zone boundary, our exact discriminative signal. See Q3. |
| Background mixing / substitution (swap background, keep person foreground) | Designed for the OPPOSITE goal — to REDUCE background reliance ("static bias") | Li et al. ICCV2023 "Mitigating Static Bias" (arXiv:2211.12883), StillMix, Action-Swap (human+Places365 bg) | For generic action recognition this helps because background is a nuisance/shortcut. **For us background is not a nuisance — it is the label carrier (walkway lines, restricted-zone geometry, panel identity are scene-fixed).** Background substitution would likely DESTROY our labels, not just reduce a shortcut. | Do not use naively — see Q3 |

**Q1 verdict**: worth doing — temporal resampling/jitter/speed-perturb/frame-dropout (free, safe, low
gain but zero risk). Uncertain/needs-validation — ObjectMix-style person copy-paste (see Q2). Skip —
VideoMix cuboid label-mixing (semantics of soft label wrong for us), Learn2Augment/Selective-Volume-Mixup
(no small-N evidence, high complexity), naive background substitution (actively harmful, our background
IS the label).

## Q2. Object-level copy-paste (persons/panels/forklifts into scenes)

Feasibility: we have the ingredients — entity detector (mAP 0.969) gives person/panel/forklift boxes
in any frame; 2262 frames have behaviour boxes. Ghiasi et al. "Simple Copy-Paste" (Instance seg, 2021)
and video successor ObjectMix (Kimata et al. 2022) establish the mechanism: cut object region, paste
into target frame/clip, adjust label.

Pitfalls specific to us:
1. **Temporal consistency**: pasting a static person crop into 32 frames looks frozen/no-motion — a
   near-perfect give-away artifact a CNN/Transformer will learn to exploit as a shortcut ("if person is
   perfectly still across all pixels → synthetic → ignore" or worse, learn spurious correlation with
   whatever real class happens to share that artifact). Need per-frame re-tracking or affine jitter
   of the pasted patch to fake motion — nontrivial without a segmentation model (we only have boxes,
   not masks) → box-paste will paste background pixels too (bbox is axis-aligned rectangle over a person,
   includes wall/floor around them), producing visible seams.
2. **We only have detector BOXES, not instance masks.** True copy-paste (Ghiasi) needs masks for clean
   edges. Box-paste = crude rectangle compositing → visible blending artifacts, likely to hurt more than
   help unless paired with feathering/Poisson blending, which is nontrivial extra engineering for a
   thesis with M4/CPU-only constraints.
3. **Label correctness depends on absolute scene position** (which zone, which side of the walkway line).
   Pasting a person "into a restricted zone" requires knowing the zone geometry in the TARGET clip
   (homography/zone map) so the paste lands in a semantically-valid location — we do have
   `ZONES_DIR`/zone_prior infrastructure per PROGRESS.md (ORB/RANSAC-registered zone geometry from
   round 2), so this is technically feasible, unlike a generic copy-paste.
4. **Defensibility in a safety-critical thesis**: examiners will ask "did you validate synthetic unsafe
   examples look plausible / are not physically impossible (e.g., person overlapping forklift, wrong
   scale)?" This needs a manual QA pass on a sample of generated clips — which reintroduces manual
   labour, defeating "no new annotation budget" unless QA sample is small (e.g., spot-check 20-30 clips,
   cheap).

**Q2 verdict**: feasible in principle given our zone-registration infra, but effort/risk is HIGH relative
to expected gain at N=452, and box-only (no masks) crops make artifacts likely. **Recommend: worth a
scoped experiment (paste person crops into empty-zone frames of the SAME clip's own background, i.e.
"self-paste" to create synthetic UnauthorizedIntervention/WalkwayViol variants without cross-clip
background mismatch) rather than full cross-clip Ghiasi-style copy-paste.** Full cross-clip copy-paste:
not recommended given cost/artifact risk vs. our small remaining time budget — flag as "commonly proposed,
not clearly justified at this N/effort ratio" rather than a hard "don't do."

## Q3. Label-destructive augmentations — CRITICAL, ranked by severity

Our 4 classes are defined by **absolute spatial semantics relative to fixed scene geometry** (which side
of a line, which zone, is a specific panel open, is a specific load overloaded), not by generic motion
pattern. This inverts the usual assumption that appearance/background is a "nuisance shortcut" (Q1's
background-bias literature, e.g. arXiv:2211.12883, arXiv:2512.17953) — for us the scene IS signal.

**MUST NOT USE:**
1. **Horizontal flip.** Mirrors which side of the walkway/zone boundary the person is on →
   WalkwayViolation becomes indistinguishable from SafeWalkway under flip (a flipped "violation" frame
   can look exactly like a legitimate position on the other side). General principle confirmed by
   augmentation-design guidance: flip is invalid whenever the label is not left-right invariant (asymmetric
   letters b/d, medical laterality — same logic applies to fixed-camera zone geometry). This is the single
   most dangerous default augmentation for us because it's on by default in almost every video-aug library.
2. **Random/aggressive spatial cropping that can exclude the panel or the zone boundary.** Confirmed
   general failure mode: random-resized-crop can produce "non-informative" crops carrying the original
   label but missing the discriminative region, acting as noisy/mislabeled samples (dashcam driving-behavior
   analogue, arXiv:2405.05354, makes the same point: periphery objects that carry the label get cropped
   out by standard RandomResizedCrop). For OpenedPanel/ClosedPanel and WalkwayViolation/SafeWalkway this
   is not a "difficult but valid" example — it becomes an unanswerable frame with the wrong label attached.
   Any cropping used must be geometry-aware (never crop below a min box covering the annotated zone/panel).
3. **Full background substitution / green-screen style bg swap** (StillMix/Action-Swap style, arXiv:2211.12883).
   Designed for generic action recognition where background is noise; for us, background (walkway markings,
   which panel, forklift path) is the primary evidence. Swapping it either destroys the label or creates
   an impossible scene (person in Restricted_Zone geometry that doesn't exist in the substituted background).
4. **Video-level soft-label mixing (VideoMix cuboid mix, CutMix/MixUp-for-video) applied across DIFFERENT
   classes with continuous soft labels.** A clip that is spatially/temporally 70% WalkwayViolation pixels
   + 30% SafeWalkway pixels is not a real point on a "0.7 unsafe" continuum — our classes are categorical
   safety judgments, not a linearly-interpolable attribute (unlike e.g. dog/cat blending). Soft-label
   mixup here manufactures physically nonsensical composite scenes and a training signal that has no
   real-world referent. If used at all, restrict mixing to WITHIN the same unsafe/safe pair (e.g., mix two
   WalkwayViolation clips together, not WalkwayViolation with OpenedPanel) — but even this is unvalidated
   at our N, low priority.
5. **Random erasing landing on the zone-boundary/panel/forklift region.** Same-class issue as cropping —
   if the erased patch removes the one discriminative object, label becomes wrong. If used, must be
   masked out from a region-aware erasure that never touches the annotated behaviour-object bbox.
6. **Aggressive rotation (viewpoint change beyond a few degrees).** Fixed CCTV cameras never rotate;
   any rotation augmentation creates an out-of-distribution artifact that also risks flipping perceived
   walkway orientation similarly to horizontal flip. Low value, avoid beyond +/-3-5 deg jitter for lens/
   camera-shake realism only.

**Conditionally OK (label-preserving):**
- Speed/temporal resampling, frame dropout, temporal jitter, brightness/contrast/color jitter, slight
  Gaussian noise/blur, small-magnitude rotation (<5deg), geometry-aware crop with hard floor (never below
  a bbox covering the annotated zone/panel/person), same-class mixing only.

**Q3 verdict**: this section's finding is the load-bearing one. Any augmentation library default (flip=True,
RandomResizedCrop with aggressive scale range) applied naively WILL corrupt labels for this dataset. Any
existing training pipeline must be audited for these defaults before doing anything else in this list.

## Q4. Using the 8-class safe/unsafe pairs (contrastive / pairwise)

- Round 1 already found this promising: PROGRESS.md #9 "Pairwise" = "8-class contrastive + cascade +
  direct ensemble" gave best top-1 (56/64) and near-best macro (89.28%) — i.e. **we already have direct
  in-house evidence this works**, not just literature.
- Literature support: Supervised Contrastive Learning with Hard Negatives (arXiv:2209.00078) — using
  labels to pick true hard negatives (same-looking, different-label) is exactly our safe/unsafe pair
  structure (WalkwayViolation vs SafeWalkway look near-identical minus zone-crossing). This is the
  textbook hard-negative-mining setup: mine negatives specifically from the matched safe counterpart
  clips rather than random other classes.
- Recommendation: train the representation with supervised contrastive loss over all 8 classes,
  constructing hard-negative pairs preferentially from `SAFE_OF` counterparts (config.py already encodes
  this mapping), then classify/marginalize to 4. This directly targets the confusion pairs the error
  analysis already found (WalkwayViol <-> UnauthorInt confusion, OpenedPanel <-> UnauthorInt) — though
  note those are confusions BETWEEN unsafe classes, not with their own safe counterpart, so hard-negative
  mining should ALSO include cross-unsafe-class pairs (WalkwayViol vs UnauthorInt), not only the
  safe/unsafe pairs explicitly asked about.
- Pair-difference modelling (train on embedding(unsafe) - embedding(safe-counterpart) as a signed
  "violation vector") is a reasonable extension but unvalidated in literature for this exact setup —
  flag as an experimental idea, not a cited technique.

**Q4 verdict**: worth doing, already has in-house evidence (round-1 best result) + literature backing.
Highest-confidence recommendation in this report. Extend hard-negative mining to include the actual
confused pairs from the error table (WalkwayViol/UnauthorInt, OpenedPanel/UnauthorInt), not just the
safe counterparts.

## Q5. Self-supervised pretraining on our own clips (VideoMAE-style)

VideoMAE paper (Tong et al., NeurIPS 2022, arXiv:2203.12602) explicitly states it achieves strong results
on datasets as small as **3k-4k videos** using extreme (90-95%) masking ratios, without extra data — and
this is presented as VideoMAE's headline "data-efficient" claim. **691 clips is ~5-6x BELOW that
already-small floor.** No evidence found that VideoMAE-style masked pretraining works below ~3k clips;
the mechanism (learning from reconstruction of masked tubelets) needs enough video diversity to avoid the
autoencoder trivially memorizing/copying near-duplicate scenes — with 8 fixed camera views and repetitive
backgrounds, 691 clips have LESS effective diversity than 691 random clips would, making this worse than
the headline number suggests.

Also directly consistent with round-1 in-house result #5: frozen VideoMAE-base (86M params, ImageNet+
Kinetics pretrained, i.e. NOT trained on our data) already hit the same 55/64 ceiling as everything
else — meaning generic large-scale video pretraining is already saturating whatever this task allows;
pretraining a NEW encoder from scratch on our 691 clips would have far less data and far less compute
than the frozen VideoMAE checkpoint already tried, so it is very unlikely to beat it.

Temporal-order-prediction / CVRL / MoCo-style contrastive SSL also generally need thousands-to-millions of
clips (their published results are Kinetics-400/UCF101-scale, i.e. >>10k clips) — general literature
review confirms these methods target pretraining set >> downstream set, not the reverse.

**Q5 verdict**: NOT worth doing. Below the smallest N in the VideoMAE paper's own regime by 5-6x, we
already have evidence (in-house frozen VideoMAE result) that large-scale generic pretraining plateaus at
the same ceiling, and our 8 fixed-camera-view redundancy reduces effective diversity below the raw clip
count. This is squarely "commonly done, not supported at this N."

## Q6. Semi-supervised (pseudo-labelling, FixMatch/FlexMatch, mean-teacher)

All located evidence (CMPL arXiv:2112.09690 CVPR2022, SVFormer 2023, Stable Mean Teacher arXiv:2412.07072
AAAI2025) targets the standard semi-supervised setup: **small labeled set + LARGE unlabeled pool**
(e.g., 1% labeled Kinetics = ~2.4k labeled clips out of 240k). We have 452 labeled train clips and — per
given data description — no large stated pool of additional UNLABELED clips beyond the 691 total (the
safe-class clips are labeled too, just not part of the evaluated 4-class task). Without a genuinely large
unlabeled pool, FixMatch/mean-teacher/pseudo-labelling has nothing to pseudo-label that adds information;
running SSL machinery on the same finite labeled+unlabeled set we already fully know the labels for is
circular. If there exist additional un-annotated raw CCTV footage (beyond the 691 clips) that could serve
as the unlabeled pool, semi-supervised pretraining/pseudo-labelling on THAT footage would be legitimate;
otherwise, skip.

**Q6 verdict**: not applicable/not worth doing UNLESS a genuinely separate large pool of unlabeled raw
footage exists (unclear from provided data — flagged as unresolved question).

## Q7. Test-time augmentation (TTA) for video

Mixed evidence. Positive: video TTA via multiple temporal views + consistency gives ~2% gain (temporal-
augmentation TTA literature); large-scale video segment classification TTA gave 0.94-1.71% relative MAP
gain (arXiv:1912.01127). Negative: "I Can't Believe TTA Is Not Better" (arXiv:2604.09697) shows TTA
consistently DEGRADES accuracy on medical image benchmarks and should never be assumed a free win —
must be validated per model/dataset, not applied blindly.

For us: TTA using multiple temporal samplings of the same 32-frame clip (different start offset/stride)
is cheap, label-safe (doesn't touch spatial semantics), and directly testable — run at inference, compare
against single-pass on a held-out val split before trusting it on the N=64 test set. Given round-2's
finding that val doesn't predict test macro at this N (Pearson r=-0.066), do NOT tune TTA aggressiveness
by test-set feedback; pick one fixed, simple TTA config a priori (e.g., average over 3 temporal
offsets) and apply once.

**Q7 verdict**: worth trying, cheap, but must be validated not assumed; keep it simple (temporal-only,
no spatial TTA given Q3's flip/crop risks) and lock the config before touching test data.

## Q8. Class imbalance (test support 32/11/13/8, imbalanced train)

- **Class-Balanced Loss via Effective Number of Samples** (Cui et al., CVPR2019, arXiv:1901.05555):
  well-established, cheap (one loss-function swap), the effective-number reweighting is smoother than
  naive inverse-frequency and specifically designed to not over-amplify at very small counts — most
  applicable single change here.
- **Focal loss**: helps with easy/hard imbalance, weaker specifically for extreme few-shot minority
  (7/8-example classes) — general finding: class weighting "most beneficial for categories with a few
  hundred to several thousand samples," rare categories (<200, and by extension our single-digit-per-class
  train counts) "remain difficult" regardless of loss reweighting — i.e. loss-function tricks have limits
  once a class has ~10 train examples; CarryOverload class (8 test / correspondingly few train) is in this
  regime already and round-1 already hit 100% on it, so further loss engineering has low marginal value
  for that specific class but may help WalkwayViol (largest, currently weakest at 84.4%) less.
- **LDAM-DRW** (Cao et al. 2019): enforces larger margins for minority classes, deferred reweighting.
  Reasonable to try, same caveat as above re: floor effect at n<10.
- **Decoupled representation/classifier learning** (Kang et al., ICLR2020, arXiv:1910.09217): train
  representation on natural (imbalanced) distribution, re-train/re-balance ONLY the final classifier
  layer on a class-balanced resample. This is cheap (no need to retrain full backbone), and directly
  suited to us since we likely already have frozen/cheap-to-extract features (round-1 pipeline reuses
  cached features across many configs). Recommended over full resampling of raw clips (oversampling 8
  clips to match 32 = severe duplication → overfitting risk on those exact 8 clips).
- **Resampling** (oversample minority clips): NOT recommended as primary tool at n=8 — duplicating 8
  clips 4x just means memorizing those exact 8, no new information; combine with cheap label-safe
  augmentation (Q1's safe list) if resampling, never oversample raw duplicates alone.

**Q8 verdict**: worth doing — swap to Class-Balanced (effective-number) loss or try decoupled
classifier retraining as a cheap add-on to the existing pairwise/contrastive pipeline. Manage
expectations: with 8-13 test examples in 2 of 4 classes, gains from loss reweighting alone are capped;
round-1 already got 100% on the two smallest classes (OpenedPanel, CarryOverload) so the real leverage
is on WalkwayViol (n=32, weakest at 84%) and UnauthorInt (n=11, 72.7%) — the imbalance techniques help
less for WalkwayViol (it's not the minority class) and its errors are semantic confusion (see Q4/Persistent
Error Clips), not lack of examples.

## Q9. Cross-view robustness + recovering the 175 missing view labels

**Augmentation for view-invariance**: standard approach is multi-view contrastive learning (pull same-
action-different-view together) — but we don't have paired multi-view footage of the SAME event (single
fixed camera per clip), so classic multi-view-contrastive setups don't directly transfer. More applicable:
train/eval stratified by view to check per-view accuracy variance; if certain views underperform, apply
view-specific color/exposure normalization (fixed CCTV cameras often have different white-balance/exposure
settings) rather than geometric augmentation. This is a data-diagnostic step more than an augmentation
technique.

**Recovering 175 missing view labels automatically**: this is a well-matched proxy problem, not the
person-reID literature that came back from search (that targets cross-camera identity matching under
appearance change, different goal). What we actually want is much simpler: cluster clips by their STATIC
background (16 fixed camera views total per the "8 camera views" — wait, dataset says 8 views, so
clustering 175 unlabeled + assigning to one of the already-known 8 view clusters is a closed-set
classification problem, not open clustering):
1. Extract a background/median frame per clip (median or mode pixel over the 32 cached frames removes
   moving persons/forklifts, leaves static scene).
2. Compute a simple perceptual descriptor (e.g., color histogram, or a frozen CNN embedding of the
   median frame, or ORB/SIFT keypoints — round 2 already built "ORB/RANSAC-registered zone geometry"
   per PROGRESS.md, i.e. we already have homography infra for zone registration that can be reused here).
3. Nearest-centroid or k-NN classification against the 516 ALREADY-LABELED clips' view clusters (8
   known classes = supervised classification, not unsupervised clustering — much more reliable than blind
   clustering since we know the target classes and have 516 labeled exemplars).
4. Validate: manually spot-check 15-20 of the 175 assignments (small effort, not full annotation).

This is a **closed-set nearest-neighbor classification using existing ORB/RANSAC infra**, not a novel
unsupervised camera-clustering research problem — should be fast and reliable given fixed, non-moving
cameras (background is highly distinctive per view once persons are median-filtered out).

**Q9 verdict**: view-invariance augmentation — low priority/skip (no real multi-view-same-event data to
exploit; exposure/color normalization is a minor diagnostic tweak, not a big lever). View-label recovery
— HIGH VALUE, LOW COST: reuse existing ORB/zone-registration code, treat as supervised nearest-neighbor
against 516 known-view clips + median-frame background descriptor. Directly increases usable metadata
(view-stratified error analysis, view-balanced sampling) at zero annotation cost.

---

## RANKED SHORTLIST (actionable, no new annotation)

1. **Recover 175 missing view labels via median-background + nearest-neighbor against 516 labeled
   clips (reuse existing ORB/RANSAC zone-registration code).** Gain: unlocks view-stratified analysis
   and view-balanced sampling/eval for ALL other techniques below. Cost: low (hours, reuses existing
   code). Label-destruction risk: none (metadata only). Not a citation-driven technique — an engineering
   reuse of existing infra; justify methodologically as closed-set NN classification, not novel research.

2. **Push the 8-class supervised-contrastive / pairwise-hard-negative direction further** (already
   validated in-house as round-1's best result, 56/64). Mine hard negatives explicitly from (a) safe
   counterparts via `SAFE_OF` and (b) the actual confused unsafe pairs in the error table (WalkwayViol/
   UnauthorInt, OpenedPanel/UnauthorInt). Gain: incremental over 56/64, medium confidence given prior
   success. Cost: medium (extend existing pairwise pipeline). Risk: none to labels; risk of overfitting
   to N=64 test if hard-negative mining is tuned ON the test errors — mine on TRAIN confusions only.
   Citation: Robinson et al. hard-negative contrastive (arXiv:2209.00078); in-house PROGRESS.md #9.

3. **Label-safe temporal augmentation stack**: multi-offset/stride resampling of the 32-frame cache +
   temporal jitter + frame dropout + brightness/contrast jitter, applied at train time. Gain: small
   (+0.5-2pp order of magnitude, extrapolated from Kinetics-scale numbers, unverified at our N but
   zero-risk). Cost: low (data-loader change only). Risk: none — purely temporal/photometric, doesn't
   touch spatial semantics. Citation: arXiv:2206.15015, arXiv:2211.04888.

4. **Class-balanced loss (effective number of samples) or decoupled classifier retraining**, applied on
   top of existing cached features. Gain: small-medium, concentrated on WalkwayViol (n=32, currently
   weakest at 84.4%) rather than the already-100%-scoring minority classes. Cost: low (loss swap /
   cheap classifier-head retrain on existing features). Risk: none to labels. Citation: Cui et al. CVPR2019
   (arXiv:1901.05555), Kang et al. ICLR2020 (arXiv:1910.09217).

5. **Cheap temporal-only TTA at inference** (average logits over 2-3 temporal offsets of the same
   clip), config locked BEFORE touching test set (round-2 already proved val-test correlation is broken
   at this N, so don't tune TTA on test). Gain: uncertain, historically ~1-2% in video literature but
   also documented to sometimes HURT (arXiv:2604.09697) — treat as a coin-flip experiment, validate on a
   train-side holdout, not the one-shot test. Cost: very low. Risk: none to labels (temporal only, no
   spatial TTA given Q3 flip/crop dangers).

**Worth a scoped, lower-confidence experiment (not top-5, but not "skip")**: self-paste augmentation —
paste person/panel crops from entity detector into other frames of the SAME clip's own zone geometry
(using existing zone/homography infra) rather than cross-clip Ghiasi/ObjectMix copy-paste. Full
cross-clip copy-paste: too much engineering/artifact risk for remaining budget.

**Commonly done, NOT supported at this N (explicitly do not pursue)**:
- VideoMAE-style masked SSL pretraining from scratch on 691 clips (5-6x below the paper's own smallest
  validated regime of 3k-4k videos; in-house frozen-VideoMAE result already shows the ceiling this would
  chase is already hit).
- FixMatch/mean-teacher/pseudo-labelling — no confirmed large separate unlabeled pool exists; without
  one this is circular.
- Cross-clip VideoMix/CutMix-style soft-label mixing across different classes (physically nonsensical
  composite labels for categorical safety judgments — see Q3 #4).
- Background substitution / StillMix-style debiasing (inverts our label mechanism — background IS signal
  here, not a shortcut to remove).
- Any TTA/augmentation tuned using feedback from the N=64 test set (round-2 already showed val-test
  divergence at this N; treat test as one-shot).

## AUGMENTATIONS WE MUST NOT USE (label-destructive, ranked by severity)

1. **Horizontal flip** — mirrors walkway/zone side, WalkwayViolation <-> SafeWalkway become
   indistinguishable. Most dangerous because it's a common default=True in aug libraries.
2. **Aggressive/unconstrained random-resized-crop** — can crop out the panel, zone boundary, or forklift,
   leaving a wrong-labeled frame that looks like a valid different-class example.
3. **Background substitution / greenscreen swap** — background is the label carrier here, not noise.
4. **Cross-class soft-label mixing (VideoMix/CutMix/MixUp across different classes)** — manufactures
   physically-impossible composite scenes with no real safety referent.
5. **Unconstrained random erasing over the behaviour-object region** — same failure mode as #2 if the
   erased patch is the panel/zone/person.
6. **Rotation beyond a few degrees** — fixed CCTV never rotates; breaks orientation cues similarly to flip.

## Unresolved questions
- Does a genuinely separate pool of unlabeled/unannotated raw CCTV footage exist beyond the 691 clips?
  Determines whether Q6 semi-supervised is viable at all.
- Do we have instance MASKS (not just boxes) anywhere, or only boxes? Determines whether clean
  Ghiasi-style copy-paste is feasible vs. only crude box-paste with visible artifacts.
- Is the zone/homography registration from round-2 (`ZONES_DIR`, ORB/RANSAC) still working/available to
  reuse for view-label recovery (Q9) and self-paste (Q2), or was it dropped in the round-2 rewrite?
- What is the actual per-view clip distribution among the 516 labeled clips — are all 8 views
  well-represented enough to serve as nearest-neighbor exemplars, or are some views themselves sparse?
- Given round-2's finding that val macro doesn't predict test macro (r=-0.066) at N=64, what is the
  agreed protocol for validating ANY of the above techniques before the final one-shot test run?
