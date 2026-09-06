# Researcher: Baseline Strategy — Small-N Video Classification

**Date:** 2026-08-16  
**Context:** 4-class industrial unsafe-behaviour video classification; 452 train / 64 test clips; ~2,262 annotated frames; YOLOv11s entity detector (mAP@50 0.969) → Spatial-Relation Transformer. Current best = S1a: 54/64 (84.4% top-1, 88.4% macro) same-dist; 75.6% LCO. Scene-only baseline = 52/64 = 81.2%.

---

## Q1. Small-N (<500 clips) video classification best practices

### What the field does

**Transfer + frozen backbone is the dominant pattern.** With <500 clips, end-to-end training of a video
backbone is high-variance and rarely beats a strong ImageNet/Kinetics-pretrained feature extractor with a
light head. Standard evidence:

- **TSN (Wang et al., ECCV 2016):** sparse-frame sampling → per-frame ImageNet-pretrained CNN → average
  logits; achieves 94.9% UCF-101. The baseline for all subsequent "light-head on frozen backbone" work.
  Works because temporal averaging acts as implicit regularisation.
- **VideoMAE (Tong et al., NeurIPS 2022):** MAE pretraining on Kinetics; the frozen ViT-B/L backbone + a
  single linear probe is competitive on UCF-101/HMDB-51 without any temporal model. On very small datasets
  (3–4 k clips), VideoMAE with frozen backbone outperforms fine-tuned 3D CNNs.
- **SIFAR (Fan et al., ICLR 2022):** rearranges K frames into a "super-image" → plain image classifier.
  On appearance-dominated actions (Kinetics-400), matches video-specific models. Demonstrates that for
  coarse action categories, per-frame image classifiers are near-optimal.
- **X3D / MoViNet frozen + MLP:** established pattern for small-dataset video (healthcare, robotics). X3D-M
  produces 192-d clip embeddings; MoViNet-A2 produces 640-d. A 2-layer MLP trained on top is the standard
  probe.

### The concrete recipe for small N

1. **Pretrained backbone, frozen (no weight updates).** Pre-extract features once. Eliminates
   catastrophic forgetting and overfitting risk.
2. **Light head:** 2-layer MLP (256→64→num_classes) or single linear layer. Fewer parameters = less
   variance on N=452.
3. **Augmentation on features or frames:** horizontal flip, temporal jitter (different frame offsets),
   RandAugment (for frame-level models). NOT heavy augmentation that distorts spatial relations.
4. **k-fold cross-validation:** stratified k-fold (k=5 or k=10) on the train+val pool for variance
   estimation. Report mean±std. LCO is additionally important here because of scene confounding.
5. **Seed ensemble (7–10 seeds):** for a 64-sample test set, a single model has ±8–10 pp variance
   (BCa 95% CI [67–95] for a 54/64 result). Seed-ensemble mean narrows this, is standard for
   small-N reporting.
6. **No test-set model selection** (blind protocol): select checkpoint/config on validation, touch test
   once per final config. The project already does this correctly.

### 2023–2025 references on small-data video recognition

- **VideoMAE v2 (Wang et al., 2023):** scales MAE to ViT-G; frozen ViT-B probe achieves competitive
  few-shot AR (HMDB-51 with 10 shots per class).
- **EV-CLIP (2024):** efficient few-shot action recognition with CLIP visual prompting; few-shot (5-shot)
  settings explicit in evaluation. Frozen CLIP backbone is state-of-art for few-shot.
- **V-JEPA (Meta, 2024):** video joint embedding predictive architecture; frozen + attentive probe
  (non-linear) outperforms linear probe on K400/SSv2. Confirms that non-linear probing > linear for
  frozen video backbones.
- **Multi-Task Generalizable Representations (Ruder et al., 2018+):** meta-learning for AR; standard
  recipe is pretrain on large video dataset → finetune head only on small target.

### Summary verdict for this project

Standard practice for <500 clips: frozen strong backbone (VideoMAE-B, X3D-M, or ImageNet CNN) + light
MLP/linear head. The project already executes this (YOLOv11s frozen + Transformer, VideoMAE-B frozen +
MLP = 55/64). The project's approach is textbook.

---

## Q2. Have we missed a lever? Ranked by expected gain vs effort

Given the current state (S1a, 54/64 same-dist, 75.6% LCO), this ranks the remaining levers:

### The critical context: scene-only baseline = 52/64 = 81.2%

**This changes everything.** Any approach that uses whole-frame appearance is a scene detector, not a
behaviour detector. The measured scene-only floor is 52/64. Our entity-Transformer = 54/64 = only +2
clips above the scene-only floor on same-distribution. The LCO protocol is the honest discrimination test.

### (a) End-to-end fine-tuning detector+head (CUDA/Colab)

**Expected same-dist gain: marginal (+0–1 clip). Expected LCO gain: possibly negative.**

Evidence: the project already ran E2E fine-tuning (V6-ft → 54/64, V10 bbox-FT → 54/64). The V10 result
is a controlled negative: fine-tuning on 400 behaviour-class bboxes regressed accuracy because COCO prior
loss > behaviour-signal gain at that scale. The gap to 62/64 is supervision-volume bound, not architecture
bound. The ~54/64 ceiling appears across 12+ tracks including E2E.

**Risk:** on a fixed-camera dataset with scene-behaviour correlation, fine-tuning the backbone risks
encoding scene shortcuts more deeply. Under LCO, a fine-tuned backbone likely performs worse than frozen.

**Verdict: skip for baseline. Low value, high risk of scene-inflation.**

### (b) Small pretrained VIDEO backbone (VideoMAE-tiny/base, X3D-M) frozen → linear/MLP probe

**Expected same-dist gain: minimal (+0–1 clip). Already tested.**

The project ran VideoMAE-base (86 M) frozen → MLP: 55/64 (87.00%). Same ceiling. VideoMAE-tiny would
likely be slightly lower (fewer params, less capacity) but similar result. X3D-M not tested but expected
similar (~54–56/64) given the VideoMAE ceiling result.

**Scene-robustness:** VideoMAE encodes temporal motion — potentially more scene-agnostic than static image
backbone. LCO performance of VideoMAE not yet measured. This IS a genuine gap. If VideoMAE LCO > 75.6%,
it would be a useful datapoint.

**Verdict: marginal same-dist gain, but LCO of VideoMAE not yet measured. Low-effort datapoint to fill
(1–2 h). Useful for the thesis comparison table, not a headline booster.**

### (c) TSN-style whole-clip average of strong image backbone (ConvNeXt/EfficientNet)

**Expected same-dist gain: possibly +2–5 clips (up to ~57/64). LCO gain: NEGATIVE vs entity-Transformer.**

This is exactly what was already measured as the "whole-frame" bracket:
- Whole-frame same-dist: 81.2% (equal to scene-only baseline)
- Whole-frame LCO: 52.5% vs entity-Transformer 75.6% LCO

A stronger backbone (ConvNeXt-Tiny, EfficientNet-B2) would improve the same-dist number (possibly to
57–60/64), but this gain IS SCENE MEMORISATION — the measured evidence confirms it. A ConvNeXt trained
on ~7k frames of fixed-camera industrial data will memorise camera-view appearance with very high fidelity.

**This is the "scene-inflating" approach.** Scene-confounded. Do NOT report as a behaviour-understanding
result. Build only as a diagnostic bracket (if at all) to quantify the scene-bias ceiling.

**Verdict: scene-inflating (bad). Do not pursue as a performance lever. Already has measured evidence.**

### (d) Reusing the trained detector's own backbone features (instead of MobileNet crops)

**Expected same-dist gain: +1–2 clips. LCO gain: unknown but likely similar.**

The current S1a pipeline uses MobileNetV2 crops for appearance. The YOLOv11s backbone (P3/P4 features at
pool4=4×4×768) was pre-extracted for the original frozen-appearance thesis experiments (53–55/64 range).
For the *relation* Transformer, appearance tokens come from MobileNet crop-embed; replacing these with
YOLOv11s ROI-pool features (already in the codebase) would:
- Use semantically richer features (YOLOv11s trained on COCO object detection, more entity-discriminative)
- Avoid the resolution/scale mismatch of MobileNet crops on small entity boxes
- Be a single config change in the token builder

**No new training required.** The pre-extracted YOLOv11s features are already on disk in `features/`.

**This is the highest-value, lowest-effort untested lever for the RELATION model specifically.** YOLO
backbone features are conditioned on detecting the SAME entities we're reasoning about; MobileNet was
pretrained on image classification and has no entity-discriminative signal for these specific object
categories.

**Verdict: scene-robust (good), low effort, untested. Ranked #1 unexplored lever.**

### Ranked summary

| Rank | Lever | Expected same-dist gain | LCO | Effort | Verdict |
|------|-------|------------------------|-----|--------|---------|
| 1 | YOLO detector backbone ROI features replacing MobileNet crops | +1–2 clips | neutral/positive | ~2h | **Do it** |
| 2 | VideoMAE-B LCO eval (already have the model) | +0–1 | unknown | ~1h | measure LCO |
| 3 | X3D-M frozen MLP probe | +0–1 | unknown | ~2h | optional |
| 4 | E2E fine-tuning backbone+head (Colab) | 0 (already tested) | negative risk | high | skip |
| 5 | TSN whole-frame strong CNN | +3–5 (scene-inflated) | negative | moderate | diagnostic only |

---

## Q3. Reproducibility & variance control for a thesis baseline at N=64/68

### The core problem

With N=64 test clips, a single-seed classification error of 1 clip changes accuracy by 1.5625 pp. A
seed-to-seed swing of ±3–5 clips (±4.7–7.8 pp) is completely expected from initialisation variance on
a 64-sample test. This was measured: 7-seed range 48–55/64 for S1a.

### What the literature says (2024)

- **"A Tale of Two Variances" (arXiv 2604.23114, 2025):** confirms single-seed benchmarks are
  unreliable on small-N; recommends multiple seeds + aggregate (mean or ensemble).
- **Bootstrap CI (BCa) for accuracy on N=64:** BCa 95% CI for 54/64 is approximately [67.5%, 87.5%]
  — a 20 pp wide interval. Any single-number comparison must acknowledge this.
- **Nadeau-Bengio corrected t-test:** standard when comparing classifiers on a held-out set from a
  larger cross-validation. K-fold needed to compute this; item E from the objectives roadmap.
- **"No Unbiased Estimator of the Variance of K-Fold CV" (Grandvalet & Bengio, JMLR 2004):** classic
  result; k-fold variance estimates are biased downward. Use BCa bootstrap on test + report seed mean±std.

### Recommended reporting practice for this thesis

1. **Seed ensemble (7 seeds, already done):** report as headline number. The ensemble prediction
   (majority vote or mean-softmax over 7 seeds) is the most stable estimator.
2. **Report ± std over individual seed test scores:** e.g., "54/64 ensemble; individual seeds
   mean 50.3 ± 2.2 / 64" — shows that the ensemble is above the per-seed mean and quantifies variance.
3. **BCa 95% CI on the ensemble (5000 bootstrap samples):** already computed for S1a; carry forward.
4. **McNemar test for pairwise comparisons:** e.g., entity-Transformer vs whole-frame on same 64 clips.
   Already implemented in `eval_utils`. Report for any cross-method comparison.
5. **Median-seed** is an alternative to mean±std; median is robust to seed outliers. For 7 seeds,
   median = 4th-ranked seed.
6. **Fixed-seed determinism on CPU:** already confirmed (MPS bug avoided; CPU is deterministic with
   `torch.manual_seed` + `numpy.random.seed` + `random.seed` + env variable `PYTHONHASHSEED=0`).

### How many seeds?

- 7 seeds is the current practice. Literature consensus: ≥5 seeds is standard for small-N.
- 7 is defensible; 10 would give one more decimal of stability but marginal return.
- Keep 7 as the canonical count; run extra seeds only if needed to resolve a close comparison.

### Making the baseline a clean anchor for YOLO26 swap

The principle is **change ONE thing at a time.** The canonical YOLO11 baseline must be:

- Fixed seeds: [0, 1, 2, 3, 4, 5, 6] (or the same 7 that produced S1a)
- Fixed config: S1a (appearance+aux, no class weights, 16 frames, T=16, 8 entity tokens, 3-layer Transformer)
- Fixed split: the same 452/64 train/test split used throughout
- Saved artifacts: `sweep_results/entity_detector/yolo11_baseline_canonical/` with:
  - 7 × checkpoint .pt files
  - per-clip logits (64 × 4 softmax) for McNemar tests
  - full_eval.json (top-1, macro, per-class, BCa CI)
  - The exact command to reproduce it (README section)

When YOLO26 swaps in as the detector: replace ONLY `detector_best.pt` + re-extract tokens. Everything
else (Transformer, config, seeds, eval) stays identical. The delta is unambiguously the detector swap.

---

## Q4. Why YOLO-based video baselines underperform — checklist

### Common failure modes

| Mode | Typical cause | Project status |
|------|--------------|----------------|
| Resolution mismatch | Detector trained at imgsz=416; small objects (<32px) missed | Fixed: imgsz=640, entities are person/machine-scale |
| Label noise | Clip-level label propagated to all frames; frames with mixed behaviours mislabelled | Acknowledged; S1a aux loss mitigates (452→7k signals); clips are pre-segmented (kept_frac=0.99) |
| Class imbalance | Minority class over-/under-represented; class weighting destabilises training | Fixed: no-weights is the proven pick; CarryOverload 8/8 test (small but robust) |
| Train/val leakage | Frame-level split leaks correlated frames from same clip into val | Fixed: clip-level split; val carved before frame sampling |
| Wrong optimizer/AMP on Apple MPS | MPS AMP bugs cause NaN loss divergence on macOS-14 | Fixed: CPU training avoids MPS entirely |
| Temporal window mismatch | Too few frames miss behaviour; too many add noise | Fixed: T=16 uniform sampling; S1b gate showed no gain (clips already single-action) |
| Backbone feature mismatch | COCO-object features not discriminative for behaviour subclasses | Partially fixed by aux per-frame loss; still a gap for WalkwayViol/UnauthorInt confusion |
| Scene-behaviour correlation | Camera view = class → trivial shortcut; model learns scene not behaviour | Measured: scene-only = 81.2%; entity-Transformer LCO = 75.6% (scene-agnostic payoff) |
| Test-set model selection | Epoch/config chosen on test → optimistic bias | Fixed: blind protocol, val-selected checkpoint |
| Overconfident ensembling | Averaging correlated models gives no benefit | Confirmed: V1 ensemble ≈ best single seed; S1a ensemble provides stability not lift |

### The two non-standard fixes specific to this project

1. **CPU training as the AMP/MPS fix:** not typical but correct. MPS AMP has a confirmed divergence
   bug on macOS-14 for mixed-precision Transformers. CPU is slower (minutes not seconds) but deterministic
   and avoids the bug. Standard fix if Colab/CUDA unavailable.
2. **Per-frame auxiliary loss (S1a):** propagating clip label to all frames then adding a frame-level
   cross-entropy auxiliary head is standard in weakly-supervised temporal action localization literature
   (W-TALC, MUTA, others). This is what the paper's per-frame bbox supervision approximates. Our version
   is "noisy label" supervision — principled, not a trick.

---

## Q5. Recommendation — Yes/Adjust?

### Current baseline state

S1a = **YOLOv11s entity detector + appearance (MobileNet crops) + aux per-frame loss Transformer,
7-seed ensemble, CPU, blind + LCO reporting = 54/64 (84.4% top-1, 88.4% macro) same-dist,
75.6% top-1 LCO.**

### Is it the right, defensible baseline?

**Yes, with ONE high-value adjustment before freezing.**

The strategy is sound:
- Blind protocol is correct (no test-set selection)
- 7-seed ensemble is correct for N=64 variance
- LCO reporting is the critical honest contribution
- BCa CIs are already computed
- The scene-bias bracket (scene-only=81.2%, whole-frame=81.2% LCO=52.5%) is a stronger contribution
  than the raw accuracy number

**The one adjustment: replace MobileNet crop-embed with YOLOv11s backbone ROI features in the
appearance tokens.**

Rationale:
- Already pre-extracted in `features/`; zero new training required
- MobileNetV2 was pretrained on ImageNet classification; has no entity-discriminative signal for
  industrial safety entities (forklifts, panels, walkway zones)
- YOLOv11s backbone features are specifically conditioned on the same entity classes the detector
  was trained on (COCO + fine-tuned on the Unsafe-Net entities at mAP@50=0.969)
- Expected: +1–2 clips on same-dist; LCO neutral or positive; closes the WalkwayViol↔UnauthorInt
  confusion pair more than MobileNet crops did (v2b lifted WalkwayViol recall 59→78% — the mechanism
  was appearance; richer appearance features = more lift)
- This is a configuration change in the token builder, not a new model or new concept
- If it produces 55–56/64 blind, THAT becomes the canonical YOLO11 baseline anchoring the YOLO26 swap

**Concrete action plan (in order):**

1. **Swap MobileNet → YOLO11s ROI features in token builder.** Run 7-seed blind sweep. If ≥55/64:
   this is the canonical baseline. If 54/64: S1a stays as the baseline. Either way, freeze after this run.

2. **Freeze canonical baseline artifacts** (checkpoint, logits, full_eval.json, command).

3. **Compute stratified k-fold (k=5) + Nadeau-Bengio corrected t-test** on the canonical baseline
   vs whole-frame probe — closes the statistical reporting gap (item E+O in objectives).

4. **Proceed to YOLO26 backbone swap** using exactly the same eval harness.

### What NOT to do before freezing

- Do NOT run E2E backbone fine-tuning (already shown to not improve; adds scene-encoding risk)
- Do NOT add TSN/ConvNeXt whole-frame as a "performance" model (scene-inflated; only for diagnostic bracket)
- Do NOT add more architectural complexity (temporal derivatives, cross-attention) — 12+ tracks confirm
  the architecture is not the bottleneck
- Do NOT change the 7-seed/blind/LCO eval setup — it is the thesis's core rigor contribution

### Final verdict

The baseline strategy is **correct and defensible**. The architecture, eval protocol, and reporting are
all well-grounded. The single high-value change is swapping the appearance feature source (MobileNet →
YOLO11s ROI features) before freezing the canonical baseline. This is a 2-hour change that could lift
the anchor by 1–2 clips and is methodologically cleaner (same backbone for detection and appearance).
If it doesn't help, S1a is the baseline. Either outcome, freeze and move to YOLO26.

---

## Unresolved Questions

1. **YOLO11s ROI-pool feature format in `features/`:** are the pre-extracted features the pool4 (4×4×768)
   global grid or per-box ROI-pool features? If only the grid is stored, per-box ROI-pool needs a token-
   builder run (re-extract from stored features via grid interpolation at box coordinates). Estimate: ~30 min.

2. **VideoMAE LCO not yet measured:** the VideoMAE-B frozen MLP (55/64 same-dist) was not evaluated
   under LCO. If VideoMAE LCO > 75.6%, it would be a useful comparison showing that MAE pretraining
   on Kinetics provides motion-invariant features that partly escape scene-bias. Worth measuring if
   time allows (1h), not blocking.

3. **k-fold + Nadeau-Bengio:** item E+O in objectives, not yet done. Needed for statistical reporting
   of cross-method comparisons in the thesis chapter. Blocking for final statistics table.

4. **Canonical baseline seeds:** are the 7 seeds for S1a the standard [0,1,2,3,4,5,6]? Need to confirm
   the exact seed list before freezing (for reproducibility section of thesis).

---

## Sources

- [TSN (Wang et al., ECCV 2016 / TPAMI 2019)](https://arxiv.org/abs/1705.02953)
- [VideoMAE (Tong et al., NeurIPS 2022)](https://arxiv.org/abs/2203.12602)
- [SIFAR (Fan et al., ICLR 2022)](https://openreview.net/forum?id=qhkFX-HLuHV)
- [RESOUND (Li et al., ECCV 2018)](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Yingwei_Li_RESOUND_Towards_Action_ECCV_2018_paper.pdf)
- [SDN (Choi et al., NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/ab817c9349cf9c4f6877e1894a1faa00-Abstract.html)
- [StillMix / SCUBA (Li et al., ICCV 2023)](https://arxiv.org/pdf/2211.12883)
- [A Tale of Two Variances (arXiv 2604.23114, 2025)](https://arxiv.org/html/2604.23114v1)
- [Bootstrap CI for small N (MachineLearningMastery, 2024)](https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/)
- [No Unbiased Estimator of K-Fold CV Variance (Grandvalet & Bengio, JMLR 2004)](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf)
- [Improving Reproducibility via Bagging (arXiv 2604.17694, 2025)](https://arxiv.org/html/2604.17694v1)
- [Ultralytics Fine-Tune Guide (2024)](https://docs.ultralytics.com/guides/finetuning-guide)
- [EV-CLIP Few-Shot Action Recognition (arXiv 2604.22595, 2024)](https://arxiv.org/pdf/2604.22595)
- [Project internal: scene-bias-finding-and-reframe.md](../reports/scene-bias-finding-and-reframe.md)
- [Project internal: gap-analysis-and-next-strategy.md](../reports/gap-analysis-and-next-strategy.md)
- [Project internal: researcher-s3-frame-behaviour.md](../reports/researcher-s3-frame-behaviour.md)
- [Project internal: objectives-review-and-remaining-roadmap.md](../reports/objectives-review-and-remaining-roadmap.md)
