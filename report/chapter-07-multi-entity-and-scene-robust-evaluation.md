# Chapter 7 — Multi-Entity Spatial-Relation Recognition and Scene-Robust Evaluation

> ### Pipeline attribution — read before citing any number in this chapter
>
> This project evolved through **three non-interchangeable pipelines** (full map:
> `docs/system-architecture.md`). Chapters were written at different points in that evolution, so
> their headline accuracies are **not** competing estimates of one system — they describe different
> systems, and mixing them is the single most common error this project has made internally.
>
> | | Pipeline | Best cited | Blind/honest | Protocol |
> |---|---|---|---|---|
> | Ch1, Ch3, Ch4 | **P1** frozen-COCO YOLO -> multi-stream Transformer | 56/64 (87.5 %) | **53/64** (82.8 %) | same-dist, N=64 |
> | Ch5 | **P1** as exported to Jetson | 55/64 (TRT engine, as recorded in `benchmarks-jetson-*.json`) | **53/64** | same-dist, N=64 |
> | Ch7 + Abstract | **P3** entity detector + spatial-relation Transformer | **54/64** (84.4 % **top-1**, 88.4 % macro) | 54/64 | same-dist N=64, **plus k-fold and LCO at N=516** |
> | Ch5 §5.6.9 | **R48** whole-frame CLIP ViT-B/32 -> B1 temporal Transformer — the engine actually on the device | **84.2 % balanced accuracy** (B1, N=1,100) | same number: the protocol cannot leak a source, so there is no separate honest column | **P-D LSO-1100**, folds grouped by source video, N=1,100 |
>
> **The headline is the R48 lane: CLIP + temporal Transformer under LSO-1100.** It is the only
> accuracy in this thesis measured on all 1,100 clips under a protocol that cannot put one source
> video in both the train and the evaluation fold, and it is the model that was exported, deployed
> and measured on the Jetson (§5.6.9). YOLO's role in that lane is **localisation** — the system
> must say *where*, not only *whether* — and localisation latency is timed separately, never summed
> into the class decision. P3's 54/64 is the entity-relation lane's result at N=64 and is what Ch7
> and the abstract report; it keeps its own evidence beyond N=64 (k-fold and LCO at N=516). P1's
> 56/64 is **test-selected** (the ensemble was chosen using the test clips) and must never be
> presented as the result; its honest blind counterpart is 53/64. Chapters 1-6 are retained as the
> experimental record of how the design was arrived at, not as the final claim.
>
> **Caution — "84.4 %" denotes two different quantities here. Never write it bare.** 84.4 %
> *top-1* (P3, blind same-dist split, N=64) is Ch7's figure. 84.4 % *balanced accuracy* (B1w,
> LSO-1100, N=1,100) is the R48 lane's best head — and B1w is **not** the head on the device. The
> deployed engine is B1 (201,090 params x 3 seeds), whose own figure is **84.2 %**; B1 was chosen on
> the pre-registered *edge* criterion, smallest model, not on accuracy, so its own number is the one
> to quote and never max(B1, B1w). §5.6.9 quotes 84.2 % throughout for that reason
> (`plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-10-01-two-different-84-4-percent.md`).
>
> **"84.2 %" likewise denotes two different quantities. Never write it bare either.** 84.2 %
> *balanced accuracy* (B1, LSO-1100, N=1,100, binary) is the deployed figure above. 84.2 % *macro*
> is P3's k-fold mean over 516 ESTA clips on the 4-class task (Ch7 §7.7.1). Different corpus,
> different task, different metric: the two must never be compared, summed, or carried into one
> sentence without their qualifiers.

> **New contribution chapter (2026).** Consolidates the multi-entity ontology + YOLOv11/Transformer
> pipeline and the scene-confounding analysis of the Unsafe-Net benchmark. Additive to the existing
> deployment-led thesis; integration notes in §7.8. All numbers reproduce from committed scripts/artifacts.

## 7.1 Motivation

The published Unsafe-Net pipeline [1] pairs a YOLO v4 *behaviour detector* (trained on 2,262 per-frame
behaviour-bounding-boxes) with a ConvLSTM temporal classifier, reporting 95.81 % macro on the 4-class
unsafe test subset. This chapter re-implements that pipeline from scratch with two modernisations —
**YOLOv11s (→ YOLO26) replacing YOLO v4** and a **box-token Transformer replacing ConvLSTM** — but with a
different supervision philosophy: instead of labelling *behaviours* per frame, we label the **entities**
that compose a behaviour (people, zones, object-states) and let the Transformer infer the behaviour from
their spatial relations. The chapter's central empirical finding is that the Unsafe-Net benchmark is
**scene-confounded**: same-distribution accuracy largely measures which camera a clip came from, not the
behaviour. Under a scene-robust (leave-camera-out) protocol the entity-relation approach is the only
method that generalises, and the paper's 95.81 % is shown to be substantially scene-inflated.

## 7.2 Multi-entity annotation ontology

Rather than "behaviour-as-object", we annotate an **8-class entity ontology**: `Person_Authorized`
(green vest), `Person_Unauthorized` (red-black vest), `Safe_Walkway` and `Restricted_Zone` (floor
polygons), `Panel_Closed/Opened`, `Forklift_Safe_Load` (≤2 blocks) / `Forklift_Overload` (≥3). Behaviour
is *derived* downstream (e.g. Walkway-Violation = a person whose foot-point lies outside `Safe_Walkway`;
Unauthorized-Intervention = `Person_Unauthorized` inside `Restricted_Zone`).

**2,262 frames** were sampled from the 566 training clips only (the 64-clip test set untouched),
viewpoint-balanced across the 8 fixed cameras and over-weighted on the WalkwayViol↔UnauthorInt confusion
pair, and annotated in CVAT (rectangles for entities, polygons for zones; foot-point-tight person boxes).
The train/val split is **clip-level** (no clip's frames cross the split), fixing the leakage of the
earlier V10 pilot (`val: images/train`). The frame count matches the paper's YOLO training set (2,262).

## 7.3 Stage A — entity detector

A COCO-pretrained **YOLOv11s** was fine-tuned (two-stage: freeze-backbone 30 ep → unfreeze 100 ep, SGD
lr 0.01→0.001, imgsz 800, `hsv_h=0` to preserve the vest-colour signal). Training ran on a CUDA GPU
(Colab) after establishing that Apple-MPS training is numerically broken on macOS 14 (silent Adam
`addcmul_` failure — see `researcher-mps-training-fix.md`); inference runs locally.

**Detector validation (343-image held-out val):** overall **mAP@50 = 0.969**, mAP@50-95 = 0.894.
Per-class mAP@50: Person_Authorized 0.938 (R 0.921), Person_Unauthorized 0.928 (R 0.937), Panel_Closed
0.989, Panel_Opened 0.972, Forklift_Safe 0.995, Forklift_Overload 0.995. The rare `Person_Authorized`
class (365 instances) was learned well — confirming the vest-colour cue survives with `hsv_h=0`.

## 7.4 Stage B — box-token spatial-relation Transformer

Per detected entity per frame we build a token
`[class, x, y, w, h, conf, IoU_walkway, IoU_restricted, foot_in_walkway, norm_dist, appearance(64)]`
(geometry computed against the per-view zone polygons; appearance = MobileNet crop-embedding). A
[CLS]+box-token Transformer (2 layers, 4 heads, ~0.5 M params) classifies the 4 unsafe behaviours; a
per-frame auxiliary behaviour head propagates the clip label to frames (452 → ~7 k supervision signals).
All configurations are evaluated under a **blind held-out protocol** (epoch/seed selected on a
train-carved val slice; 64-clip test touched once); the small val set (68 clips) makes single-seed
selection high-variance, so we report the **7-seed ensemble** as the robust estimator.

**Ablation (blind seed-ensemble, 64-clip test):**

| Config | closes gap-cause | top-1 | macro |
|---|---|---:|---:|
| v1 geometry only | weak zone signal | 51/64 | 86.1 % |
| v2a + person↔object proximity | (no gain) | 49/64 | 83.0 % |
| v2b + appearance | lossy tokenisation | 52/64 | 84.6 % |
| **S1a + per-frame supervision** | **supervision placement** | **54/64** | **88.4 %** |
| S1b + behaviour frame-gate | (no gain, τ→0) | 53/64 | 87.6 % |

Each rung targets a specific cause identified by the paper back-check (§7.6). The best config (**S1a**,
appearance + per-frame auxiliary loss) reaches **54/64 (84.4 % top-1, 88.4 % macro)** — above the paper's
no-bbox macro baseline (85.78 %). The frame-gate (S1b) yielded no gain (τ→0, kept-frac ≈ 0.99): our
clips are already single-action segments, so the paper's frame-reduction step has nothing to reduce.

## 7.5 Full multi-lens evaluation (best model, 64-clip test)

> **One decision rule — resolved 2026-09-06.** Every metric below is now computed from the **same
> predictor**: the val-selected single seed (`blind_pred` / `blind_softmax`, **54/64**). This was not
> previously possible. Two defects had to be closed first:
>
> 1. **Wrong artifact.** The ROC-AUC (0.953), PR-AUC (0.874) and ECE (0.104) printed here until
>    2026-09-06 were **not from this pipeline at all** — they reproduced `logits_s1a.pt` (≡ the YOLO11
>    track) to four decimals, not `logits_yolo26.pt`.
> 2. **Wrong predictor.** After correcting (1), the probability metrics still had to come from
>    `ens_softmax` (the seed-*averaged* ensemble, which scores 52/64), because the blind seed's own
>    softmax had never been persisted — so the calibration row described a different model from the
>    accuracy row.
>
> Defect 2 is now closed. The generating command was recovered and verified to reproduce the stored
> artifact **bit-for-bit** (`--tag yolo26app_ --seeds 5 --aux_w 0.5 --epochs 120`, blind seed 1; max abs
> difference on `ens_softmax` exactly 0.00e+00 — see `reports/test-touch-log.md`), which yields the blind
> seed's own probabilities. The table is therefore a single model's joint scorecard.

| Metric | value | Metric | value |
|---|---|---|---|
| top-1 | 84.4 % (54/64), BCa CI [71.9, 90.6] | Cohen's κ | 0.775 |
| macro-recall | 88.4 %, BCa CI [77.8, 94.4] | Matthews MCC | 0.785 |
| macro-F1 | 85.6 % | ROC-AUC (ovr-macro) | **0.951** |
| balanced-acc | 88.4 % | PR-AUC (ovr-macro) | **0.880** |
| ECE (15-bin) | **0.097** | cost (miss×10) | 100 |

All rows: blind rule, 54/64, `logits_yolo26.pt`. For reference the superseded ensemble-basis values were
ROC-AUC 0.950 / PR-AUC 0.869 / ECE 0.101; the blind seed is very slightly better separated and better
calibrated than the 5-seed average, but the point of the change is *coherence*, not the third decimal.

Per-class precision / recall / FNR: WalkwayViol 96.2 / 78.1 / 21.9 · UnauthorInt 58.8 / 90.9 / 9.1 ·
OpenedPanel 84.6 / 84.6 / 15.4 · CarryOverload 100 / 100 / 0. Confusion + reliability figures in
`figures/relation_confusion.png`, `figures/relation_reliability.png`. **Action-recognition time:** the
Transformer classifier runs at **0.9 ms/clip**; the full detector+classifier pipeline ≈ **200 ms/clip**
on an M3 (detector-bound), vs the paper's 0.14 s on a 2×GTX-1080 workstation.

Two statistics are decisive and honest: **McNemar vs a scene-only baseline is not significant**
(p = 0.804, blind basis; 0.815 on the superseded ensemble basis) and **vs the paper's no-bbox baseline Δ = −4.7 pp is not significant** (p = 0.435). i.e. on
the *same-distribution* protocol our model is statistically indistinguishable from camera-guessing — which
motivates §7.7.

## 7.6 Why the gap to 95.81 % — a supervision-placement diagnosis

Back-checking the paper: its ConvLSTM trains on the **same 452/64 split** we use; the +10 pp from
85.78 % → 95.81 % is **entirely** the YOLO-v4 behaviour-frame-reduction. YOLO v4 is a *per-frame
behaviour detector* (2,262 behaviour-bboxes, 95.66 % mAP) that selects behaviour frames and splits clips
into single actions. The paper thus spends its 2,262-frame budget on the *classification answer* at the
frame level; we spend the same budget on *entities*, deferring discrimination to a Transformer trained on
452 clip-level labels (~5× less, less granular). This is a **supervision-placement** effect, not an
architectural deficiency — S1a (adding frame-level supervision) recovered ground exactly as predicted.

## 7.7 The Unsafe-Net benchmark is scene-confounded (central finding)

The dataset has **8 fixed cameras**, and camera view strongly predicts behaviour (3 of 8 views are
single-class). A **scene-only baseline** — predict the behaviour from the camera view alone, ignoring the
video — scores **52/64 = 81.2 %** on the test set. A **whole-frame classifier** (frozen features + probe,
the paper's scene-dependent style) scores **53/64 = 82.8 %** same-distribution (macro 79.5 %) — **one clip
above camera-guessing**, i.e. all that training buys over simply knowing which camera you are looking at
is a single test clip, and it is *worse* on macro.

### 7.7.1 Random k-fold at N = 516 — the confound measured from the inside

Before turning to cross-camera evaluation, the confound can be measured *within* the standard protocol,
at eight times the test size. We pool the 516 four-class clips and run **stratified 5-fold × 10 repeats
(50 folds)**, comparing the entity-relation model against a **per-view-majority scene-only baseline**
(predict each clip from its camera's training-majority class). Under random k-fold every camera appears
in training, so this baseline is informative rather than trivial.

| | fold-mean top-1 | pooled OOF | macro | macro-F1 | κ | MCC |
|---|---:|---:|---:|---:|---:|---:|
| **entity-relation (YOLO26, headline)** | **80.8 ± 4.0 %** | 421/516 = 81.6 % | 84.2 % | 83.5 % | 0.741 | 0.744 |
| entity-relation (YOLO11, ablation) | 80.7 ± 3.6 % | 421/516 = 81.6 % | 85.1 % | 83.9 % | 0.744 | 0.749 |
| scene-only (per-view majority) | 77.1 ± 2.5 % | — | — | — | — | — |

**The behaviour model's advantage over camera-guessing is +3.7 pp and is _not_ statistically
significant: Nadeau–Bengio corrected resampled *t* = 1.93, df = 49, p = 0.0600.** The YOLO11 ablation
independently gives +3.6 pp, t = 1.95, p = 0.0574 — also n.s. We report this against our own interest.
It is the direct measurement of the claim in §7.1: on a benchmark whose protocol lets every camera into
training, a model that ignores behaviour entirely is already within noise of one that models it.

Two methodological points travel with this table.

*Nadeau–Bengio is not optional here.* Folds share training data, so fold scores are positively
correlated and the naive standard error is badly anti-conservative: on the headline run naive
SE = 0.58 pp versus **NB-corrected SE = 2.12 pp**, a factor of 3.7. The uncorrected interval would have
made the same +3.7 pp look comfortably significant. Corrected 95 % CI: **76.7–85.0 %**.

*Fold-mean and pooled top-1 are different estimators* and are reported separately above rather than
merged: the fold-mean (80.8 %) is the quantity the NB interval attaches to, while the pooled
out-of-fold count (421/516 = 81.6 %) scores every clip once. An earlier draft quoted "80.8 % (417/516)",
which silently paired one with the other; they agreed only by numerical coincidence in a superseded
single-repeat run.

*Detector generation is not the bottleneck.* YOLO26 (80.8 ± 4.0 %) and YOLO11 (80.7 ± 3.6 %) are
indistinguishable across 50 folds each with overlapping NB intervals, so the ceiling documented in
Chapter 4 is not detector-limited. Canonical runs:
`sweep_results/relation_tokens/canonical/kfold_repeats10_yolo26app_HEADLINE.log` and
`…/kfold_repeats10_yoloapp_YOLO11.log`.

### 7.7.2 Leave-camera-out — the confound measured from the outside

We therefore evaluate all methods under a **leave-camera-out (LCO)** protocol — train on some cameras,
test on held-out cameras — the honest test of behaviour generalisation:

| Method | same-distribution top-1 | **leave-camera-out top-1** | LCO macro |
|---|---:|---:|---:|
| scene-appearance (trained, transferable) | — | 53.1 % | 64.4 % |
| whole-frame (paper-style, trained) | 82.8 % | 52.5 % | 60.3 % |
| **entity-relation (ours)** | 84.4 % | **77.7 %** | **81.2 %** |
| *prior / constant floor (not a model)* | *—* | *16.5 %* | *10.1 %* |

**Honest margin: +16.9 pp macro over the scene-appearance baseline, 95 % bootstrap CI
[+12.5, +21.2] pp** (n_boot = 5 000, resampled over the 516 pooled clips). Canonical run:
5 seeds, 60 epochs, 8 camera-groups, `sweep_results/relation_tokens/canonical/
lco_seeds5_yolo26L1_CANONICAL.log`.

Two independently constructed transferable baselines — a logistic regression on whole-frame
MobileNetV3 background features (53.1 %) and a trained whole-frame classifier (52.5 %) — land
within 0.6 pp of each other. That agreement is what makes the margin credible: it is not an
artifact of one baseline being badly built.

**The "prior" row is a floor, not a competitor, and must never be used as the comparison.** A
camera→class lookup is undefined for a camera never seen in training, so an earlier version of
`run_lco_eval.py` fell back to predicting the training **global-majority class** — a *constant*.
Its 16.5 % top-1 / 10.1 % macro is that constant's arithmetic, and it reproduces the 16.9 % that
an earlier draft of this chapter reported as a "scene-only collapse". **The former "+58.7 pp over
scene-only" claim is withdrawn**; it compared a trained model against a constant. Per-view-majority,
the informative baseline used under random k-fold, is likewise **undefined** under LCO (the held-out
view contributes zero training clips), which is precisely why the k-fold 77.1 % baseline and these
numbers are not comparable estimators.

Per-class LCO recall (**scene-appearance vs ours**, canonical run): OpenedPanel 88.0/88.7,
CarryOverload 100/100, **WalkwayViol 17.6/70.5, UnauthorInt 51.9/65.7**. Object-visible behaviours (panel, forklift) generalise for everyone; the
**spatial-relational confusion pair - the whole point of the task - collapses for the whole-frame
approach and is carried by our entity+geometry reasoning**.

**How much weight this per-class pattern can bear (revised).** An earlier draft called this pattern
"the strongest evidence here, not the aggregate". That is withdrawn - the stability study in §7.8
reverses the ordering. Under perturbations that leave the aggregate margin essentially unchanged, the
individual recalls move a great deal: re-seeding the scene clustering swings UnauthorInt by 20.4 pp in
the k-fold scene-group scheme, and removing the 51 label-contradictory duplicate clips moves OpenedPanel
by 18.5 pp under LCO even though only **one** of the removed clips belongs to that class - i.e. the move
is training variance, not a change of population. The aggregate margin is the robust quantity here
(+16.9 pp macro, CI [+12.5, +21.2], reproduced as +17.7 pp CI [+12.9, +22.4] on the deduplicated set);
the four recalls are reported as descriptive support for the intended mechanism, not as its proof.

**Implications.** (i) **The same-distribution metric is scene-inflated on this benchmark.** With these
view<->class correlations and no camera holdout, a model can score 81.2 % by identifying the camera. That
is a claim about the *metric*, and it is directly measured here. It is **not** a claim that the paper's
system would fail under LCO: their weights and per-clip predictions are unreleased, so that experiment
cannot be run, and asserting its outcome would be speculation dressed as a result. State the metric
critique; leave their number alone. (ii) The entity ontology's real payoff is **scene-agnostic,
cross-camera generalisation** - the property a deployable factory-safety system needs, since cameras are
added, moved, and re-aimed. Same-distribution top-1 is the wrong metric on this benchmark; scene-robust
accuracy is the right one. Against the *weaker* of the two transferable baselines - the trained
whole-frame classifier - the lead is **+20.9 pp macro (81.2 % vs 60.3 %) and +25.2 pp top-1
(77.7 % vs 52.5 %)**; against the *stronger* scene-appearance baseline it is **+16.9 pp macro
(81.2 % vs 64.4 %)**. **The +16.9 pp figure is the headline claim throughout this thesis** - it is the
margin over the harder baseline, and quoting the larger number as the result would be baseline-shopping.
Both sides of both comparisons are canonical 5-seed runs; the earlier "+23.1 pp (provisional)" paired a superseded 3-seed 75.6 % top-1 with
this baseline and is superseded.

## 7.8 Limitations and integration

### 7.8.1 Stability of the LCO result under perturbation (what each number can bear)

Three perturbations were applied that a reader could reasonably object were arbitrary choices, to see
which conclusions survive them — the scene clustering, the training seed, and the label-contradictory
duplicates. All run on the pooled 516-clip set and consume **no test evaluations**.

**(a) Re-seeding the scene clustering.** The camera proxy is `KMeans(8, random_state=0)`; the seed was
hardcoded, so the grouping carried no measurable uncertainty. Exposing it (`--cluster_seed`) and
re-running shows the aggregate is stable while the per-class recalls are not. In the k-fold
scene-group scheme, top-1 moves 70.6 / 72.3 / 72.4 % across three clusterings (spread 1.7 pp, SD 0.97)
— *smaller* than the training-seed SD of 1.93 pp measured over 8 seeds at fixed clustering. Combining
both sources gives ±2.2 pp, which is the interval this thesis quotes. But UnauthorInt recall over the
same three clusterings is **38.6 / 24.7 / 45.1 % — a 20.4 pp swing from re-seeding alone**, and
WalkwayViol moves 9.7 pp, while the object-visible classes barely move (OpenedPanel 5.0 pp,
CarryOverload 0.0 pp).

The same sweep was then run on the **LCO** scene grouping itself, which is the grouping the
cross-camera claim actually rests on. Re-clustering moves the aggregate margin over the
scene-appearance baseline to **+16.9 / +13.3 / +16.9 pp** (CIs [+12.5, +21.2], [+9.2, +17.3],
[+12.5, +21.2]). All three exclude zero and mutually overlap, so the sign and the significance of the
cross-camera margin survive re-clustering; the magnitude moves by 3.6 pp, and the margin is therefore
quoted in this thesis as **+13.3 to +16.9 pp**, not as a single figure.

One caveat is stated because it limits how much the sweep is worth. The `cluster_seed = 2` grouping
returned results identical to the canonical grouping in every field. This was checked rather than
accepted: the two partitions genuinely differ (ARI 0.9952), the run demonstrably loads its own map
and builds its own folds from it, and the identity turns out to be forced by the data — the only
clips the re-clustering moves are 56 clips of a single class (CarryOverload), reshuffled between two
folds whose union is unchanged and which the model classifies perfectly under both groupings, so
every per-clip prediction is necessarily unchanged. `cluster_seed = 2` is thus a degenerate
perturbation rather than an independent replicate, and the clustering term above rests on **two**
distinct partitions, not three.

**(b) Varying only the training seed.** Holding the clustering fixed and varying the seed (five seeds, identical folds,
data and hyperparameters) separates the last confounded term. Aggregate LCO top-1 moves 4.8 pp
(72.9–77.7 %, SD 2.08), but per-class LCO recall moves **15.3–21.1 pp** — WalkwayViol 15.3,
UnauthorInt 19.4, OpenedPanel 21.1, CarryOverload 0.0. This is a direct measurement of the effect
that (b) below can only infer, and it is larger than the 18.5 pp that (b) attributes to training
variance. It is also the reason the headline is reported as a **5-seed ensemble**: 77.7 % is the
ensemble, while individual seeds average 75.2 %.

**(c) Removing the label-contradictory duplicates.** The dataset contains 691 clips of which only 658
are byte-distinct, in **32 md5-identical groups carrying two or more different class labels** (26 of the
32 conflate WalkwayViolation with UnauthorizedIntervention). These clips are unlearnable by
construction: identical pixels, contradictory targets. Dropping them (465 clips) leaves the LCO
aggregate margin intact — **+17.7 pp macro, CI [+12.9, +22.4]**, against the canonical
**+16.9 pp, CI [+12.5, +21.2]** — but moves OpenedPanel LCO recall by **18.5 pp** (88.7 → 70.2 %) even
though **one** single clip of that class was removed. A change that large from a one-clip change of
population is training variance, not a population effect — and (b) measures that variance directly at
21.1 pp for this class, which is larger than the 18.5 pp being explained.

**What follows.** The aggregate cross-camera margin is robust to all three perturbations and is the
quantity this thesis claims. Individual LCO per-class recalls are **not** stable to within the
differences the mechanism argument would need to read off them, and are reported here as descriptive.
This ordering is the reverse of an earlier draft's and is stated against our own interest: the
per-class pattern is the more rhetorically appealing evidence, and it is the weaker evidence.

*A second, unrelated finding from (c), reported because it bears on the k-fold analysis of §7.7.1:*
removing those 51 clips raises random-k-fold top-1 from 80.8 to 86.6 % and moves the Nadeau–Bengio test
from p = 0.057 (n.s.) to p = 0.0014 (significant). **This is not evidence of a better model.** Scoring
the unchanged canonical model's own out-of-fold predictions on the two subsets shows it gets
**43.1 % (22/51)** on the dropped clips — at the coin-flip level their 25/25 class-0/class-1 mix implies
— and 85.8 % on the 465 kept clips. So removing them from the *denominator* alone accounts for
80.8 → 85.8 %; at most ≈ 0.8 pp of the 5.8 pp gain is attributable to training on cleaner labels. The
headline k-fold figures in §7.7.1 therefore remain **80.8 % and p = 0.057 (n.s.)** on the full 516
clips. The finding worth reporting is the diagnosis, not the number: the benchmark's own duplicate
contradictions are large enough to move a significance test across the 0.05 threshold, and they
systematically penalise an input-dependent behaviour model relative to an input-independent scene
baseline on exactly the class pair this task is about.

### 7.8.2 Other limitations

- **Residual cross-camera confusion pair.** UnauthorInt LCO recall is **65.7 %** and WalkwayViol
  **70.5 %** (canonical 5-seed run) — the weakest two classes, and both are the spatially-defined ones:
  zones are camera-specific, so a wholly unseen camera lacks a zone reference. A per-view zone
  calibration (one polygon per camera — the thesis's "virtual fence" future work) directly addresses
  this and is cheap (~5 min/camera). *(An earlier draft quoted 35 % here from a 3-seed run; the
  canonical run nearly doubles it. Seed count mattered a great deal on this class — which is itself
  the argument for the 5-seed floor, and a caution against reading any single-seed per-class recall
  in this project as stable.)*
- **Camera grouping is a proxy, not ground truth.** Unsafe-Net publishes no camera IDs, so the eight
  "cameras" are a KMeans(8) clustering of per-clip static-background descriptors
  (`tools/build_scene_grouping.py`). Every LCO number here is therefore bounded by how well that proxy
  tracks real cameras — state this wherever LCO is cited; it is the single biggest caveat on the
  chapter's headline result.
- **Small N.** 64-clip test, 68-clip val → wide CIs and high seed variance; the seed-ensemble and LCO
  pooling (516 clips) mitigate this. Statistics reported with BCa/Wilson CIs and McNemar throughout.
- **MT3 (edge) deferred.** Porting the YOLO26 detector + Transformer to Jetson TensorRT-INT8 (latency,
  FPS, multi-stream, energy) is the remaining deployment step; the earlier pipeline's Jetson artifacts
  stand as the deployment baseline.

**Integration map (for final compilation):** §7.2–7.4 extend Chapter 3 (methodology); §7.5–7.7 extend
Chapter 4 (experiments) and should be cross-referenced from Chapter 6's honest-reading — the scene-bias
finding strengthens the thesis's data-bound diagnosis into a *supervision-placement + scene-confounding*
diagnosis, and reframes the accuracy result as a scene-robustness advantage rather than a top-1 deficit.

## 7.9 Reproducibility
Scripts: `build_relation_tokens.py`, `augment_relation_tokens.py`, `build_wholeframe_features.py`,
`model_relation_head.py`, `run_relation_transformer.py` (`--aux_w`, `--frame_gate`), `run_lco_eval.py`,
`run_wholeframe_eval.py`, `run_full_eval.py`; detector `sweep_results/entity_detector/detector_best.pt`;
metrics `sweep_results/relation_tokens/full_eval.json`; reports under
`plans/2026-07-09-spatial-relation-yolo26-3month/reports/`.

## Unresolved questions
1. Would per-view zone calibration lift the cross-camera confusion pair (UnauthorInt LCO 65.7 %, up from an old 3-seed 35 %) materially?
2. Does the paper's 95.81 % survive an LCO re-evaluation (requires their per-camera labels — filed as a request)?
3. Is 2,262 entity frames vs 2,262 behaviour frames a fair head-to-head, or should a behaviour-detector control (S3) be run to bound the scene ceiling under LCO?

## References
[1] O. Önal and E. Dandıl, "Unsafe-Net: YOLO v4 and ConvLSTM based computer vision system for
real-time detection of unsafe behaviours in workplace," *Multimedia Tools and Applications*, vol. 84,
no. 29, pp. 34967–34993, 2024. DOI: 10.1007/s11042-024-19276-8.
