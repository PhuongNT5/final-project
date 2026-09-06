# Thesis Process Review — why results stalled, and what the rewrite changes

**Date:** 2026-09-01. Scope: full re-review of the pipeline against Önal & Dandıl (*Unsafe-Net*),
plus the design of the round-2 rewrite (`unsafenet/`). Numbers filled in from the blind run.

## 1. What the paper actually does (verified, `paper/paper.txt` L205, L269-273, L304-308, L323-342)

- YOLO v4 is trained on **2,262 frames with `nc=4` = the four behaviour classes**. It is a
  *behaviour* detector, not an object detector (imgsz 416, batch 65, mAP ~97 %).
- It is used twice at inference: (a) **frame reduction** — keep only frames where a behaviour fires,
  (b) **single-action separation** — split a raw clip into segments containing one action.
- ConvLSTM then classifies the reduced segment, on the **same 452 train / 64 test unsafe split** we use.
- Tables 5-7: ConvLSTM **without** reduction = 57/64 (89.06 %) top-1, **85.78 %** macro.
  Unsafe-Net **with** reduction = 62/64 (**96.88 %**) top-1, **95.81 %** macro.
  The whole +5 clips / +10 pp is the frame-reduction step.

## 2. Root cause of the stall — supervision placement

Same annotation budget (~2,262 frames), spent differently:

| | Paper | This thesis (rounds 0-1) |
|---|---|---|
| 2,262-frame budget | **behaviour** bboxes = the answer | **entity** bboxes = an intermediate |
| Per-frame model | behaviour detector → frame select + split | entity detector (mAP .969), no behaviour selection |
| Temporal model | ConvLSTM on cleaned segments | Transformer on unfiltered uniform T=16 |
| Discriminative signal | ~2,262 frame-level | 452 clip-level (~5x less) |

Consequence: rounds 0-1 reproduce the paper's *no-reduction* regime almost exactly
(our best blind macro 85.4 vs the paper's 85.78) and cannot reach the reduced regime, because the
mechanism worth +10 pp was never built.

**Second finding, from re-reading the code:** the "whole-frame paper-style baseline"
(`build_wholeframe_features.py`) used **frozen MobileNetV3-small features random-projected to 64
dims**. That is not the paper's mechanism — it is a strawman. A properly fine-tuned per-frame CNN,
which is exactly the role YOLO v4 plays, had **never been implemented** in 12+ experiment tracks.
This was the single largest untapped lever, and it is what round 2 builds.

**Third finding:** the zone geometry was weak for a mechanical reason. One canonical polygon per
*view* was never aligned to individual clips, so `foot_in_walkway ~= 0.12` for both confused classes
(LR AUC 0.629). Round 2 registers each clip to its view reference by ORB+RANSAC homography and adds
an independent HSV green-paint prior.

**Fourth finding:** motion was never encoded, yet it is the physical difference between the two
classes that actually confuse — a walkway violation is a person *translating across open floor*, an
unauthorised intervention is a person *stationary at a machine*.

## 3. What round 2 (`unsafenet/`) does differently

1. **Stage B — per-frame behaviour recogniser (the paper's mechanism).** Fine-tuned CNN, clip label
   propagated to frames, trained on **all 8 classes** so the safe counterparts (SafeWalkway,
   AuthorInt, ClosedPanel, SafeCarry; +114 train clips) supply the contrast that *defines* each
   unsafe class. Marginalised to 4 at inference via `pair_contrast` = unsafe logit - safe logit.
2. **Frame reduction as a first-class member.** `aggregate.py` keeps the top-q frames by
   P(any unsafe) and averages — a training-free replication of the paper's reduction step, at
   q in {25, 50, 75} %, plus mean/max/pair members.
3. **Registered geometry + motion streams.** Per-clip homography, green-paint mask, dwell fraction,
   net displacement, path length.
4. **Temporal heads** (mean / BiLSTM ~= ConvLSTM / Transformer) over frame embeddings + logits + geom,
   with a per-frame auxiliary loss so frame-level supervision reaches the temporal model too.
5. **Blind fusion.** Caruana greedy selection with replacement, weights chosen on the val slice only.

## 4. Protocol integrity (non-negotiable)

Epoch, config, seed, ensemble weights are selected **only** on a val slice carved from TRAIN clips
(83 unsafe clips, stratified on (label, view)). The 64-clip test set is touched once per final
config. Test posteriors are stored but never used for any selection. Bootstrap CIs (5,000 resamples)
accompany every reported number.

## 5. Results of the rewrite (blind, single test touch)

Pipeline executed end to end: 691 clips cached at T=32 (2.8 GB) -> entity detection (691 clips) ->
registered geometry + motion streams -> two per-frame behaviour recognisers (b0 resnet34, best ep5
val macro 83.7; b1 effb0, best ep4 val macro 84.7) -> 6 training-free aggregation members per
backbone -> 22 temporal heads (mean / BiLSTM / Transformer x stream subsets x 3-8 seeds) ->
bagged Caruana fusion selected on val -> one evaluation of the 64-clip test set.

**Final blind result: 50/64 = 78.12 % top-1 [66.0, 87.5], 80.81 % macro [70.6, 90.2].**
Per-class recall: WalkwayViol 24/32, UnauthorInt 7/11, OpenedPanel 11/13, CarryOverload 8/8.

This is **worse** than the project's previous blind result (53/64 = 82.8 % top-1, 85.4 % macro).
The rewrite did not beat the paper, and did not beat round 1. Reported as measured.

## 6. The two findings that matter more than the number

### 6.1 On this benchmark, validation accuracy carries no information about test accuracy

Across the **34 legitimate configurations** trained in this round (all selected without touching test):

- val macro range 80.6 - 88.2 %, test macro range 73.4 - 85.9 %
- **Pearson r = -0.066 (p = 0.71); Spearman rho = 0.024 (p = 0.89)**

Val and test macro are statistically independent. The val slice (83 clips) and the test set
(64 clips) are both far too small for the differences anyone reports on this dataset to be real.
Concretely: the config with the *best* val macro (t1-lstm-lg, 88.2) scored 80.8 on test, while a
config ranked 24th on val (tx-lstm-elg, 83.1) scored 85.9. Selection is a coin flip.

This retro-explains the whole thesis history: rounds 0-2 produced a ladder of "improvements"
(52 -> 53 -> 54 -> 56/64) that were never improvements, only re-draws from a +/-10 pp noise band.
The bootstrap CI on 64 clips is about +/-11 pp wide; every result the project has ever produced sits
inside every other result's interval.

### 6.2 The paper's headline mechanism is not statistically significant, and does nothing here

**Direct ablation** (frame reduction = keep top-q frames by P(any unsafe), the YOLO-v4 step):

| backbone | no reduction | gate 25 % | gate 50 % | gate 75 % |
|---|---|---|---|---|
| b0 resnet34 | val 83.7 / test 81.2 | 81.5 / 81.2 | 82.6 / 81.2 | 82.6 / 81.2 |
| b1 effb0 | val 84.7 / test 79.7 | 83.2 / 81.2 | 81.3 / 78.9 | 83.9 / 78.9 |

Frame reduction costs 0.8 - 3.4 pp on val and moves test by -0.8 to +1.5 pp. It gives nothing.
The reason is mechanical: the paper credits reduction with *"ensuring there is only one action in the
same video"*, but the **released clips are already single-action**, so the step has nothing to
remove. Whatever produced the paper's +10 pp is not reproducible from the released data.

**And the paper's own evidence is weak.** Its central ablation is 62/64 vs 57/64 on one 64-clip test
set. Exact Clopper-Pearson intervals overlap heavily (96.88 % [89.2, 99.6] vs 89.06 % [78.8, 95.5]).
A McNemar test on 64 paired clips, under the *most favourable possible* discordance (all 5 flips one
way, none the other), gives **p = 0.0625** - not significant at alpha = 0.05. Under realistic
discordance (7 vs 2), p = 0.18. The result the entire method rests on is a 5-clip difference that
its own test set cannot resolve.

## 7. What this means for the thesis

The honest framing is no longer "we are 15 pp short of the paper". It is:

1. **The benchmark cannot support the claims made on it.** 64 test clips, 8 fixed cameras, and a
   camera-view-only baseline of 52/64 (81.2 %) mean the reported spread between every published and
   in-house system is inside sampling noise. Demonstrated here with r = -0.066 over 34 configs.
2. **The paper's mechanism does not transfer**, and its supporting ablation is not significant.
3. **The remaining honest contribution is the deployment system and the evaluation protocol** -
   leave-camera-out, blind held-out selection, bootstrap CIs, and the scene-confound analysis -
   which is what a reader can actually trust and reuse.

Chasing a higher number on this test set is not a research activity; it is resampling. The thesis is
stronger asserting that, with the evidence above, than claiming a 56/64 that means nothing.

## Unresolved questions

- Should the thesis be re-scoped around the negative result + deployment, or should new data
  (more cameras, more clips) be collected to make the accuracy claim testable at all?
- Is leave-camera-out over 8 views (not the 64-clip split) the right headline metric to report?
- The authors' 2,262 behaviour-frame annotations were never released; without them the paper's
  frame-reduction claim cannot be reproduced even in principle. Worth stating in the thesis?

---

## CORRECTION (2026-09-03): behaviour-detector failure was an MPS training bug, not the task

Two behaviour-detector runs failed totally: run #1 (tight boxes, 416 px, 25 ep, 2.31 h) mAP50 = **0.0144**;
run #2 (context boxes, 512 px, 36 ep, 6.59 h) mAP50 = **0.00212**. I attributed this to the behaviour-as-object
formulation being unlearnable on our data (person boxes ~14x50 px, class defined by out-of-box context).
**That diagnosis was wrong.** Controls:

| control | data | device | amp | epochs | mAP50 |
|---|---|---|---|---|---|
| A | entity (known-good) | mps | on | 3 | **0.0002** |
| B | entity (known-good) | mps | off | 3 | **0.0002** |
| val | entity, `detector_best.pt` | mps (inference) | - | - | **0.9752** |
| C | entity (known-good) | **cpu** | off | 1 | **0.940** |

Control A/B reproduce total failure on the *entity* dataset that previously trained to 0.969 -> failure is
not the behaviour task, not box size, not AMP. Inference on MPS is fine (0.9752). CPU reaches 0.94 after
one epoch with the identical recipe.

**Root cause: Ultralytics YOLO *detection training* does not converge on the MPS backend in this stack
(torch 2.11.0 / ultralytics 8.4.27).** Training losses stay high (box 3.5 / cls 4.8 on MPS vs 1.16 / 1.60
on CPU at epoch 1). ~9 h of compute lost. Pure-PyTorch training on MPS is unaffected - the stage-B frame
recognisers (`frame_b0`/`frame_b1`) trained on MPS and reached val macro 83.7 / 84.7.

Consequence: the paper's YOLO-behaviour-detector stage was **never actually tested**. It is now being
trained on CPU (`--device cpu --amp False`), in two variants to settle the box-context question as an
ablation rather than an assumption:
- `beh_tight` - literal paper formulation, tight person boxes, 416 px (paper's imgsz), 25 ep;
- `beh_ctx` - person boxes expanded 4.0x wide / 1.8x tall to enclose the cue.

`train_behaviour_detector.py` now defaults to cpu and carries the bug note.
