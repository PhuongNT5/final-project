# Phase 06 — Binary Experiment Suite

## Context Links

- **`design-2026-09-16-binary-encoder-redesign.md` §4-§6** — protocol, decision rule, cost table. Authoritative.
- `laneb-2026-09-16-binary-data-prep.md` — the 691-clip co-clustering measurement.
- `brief.md` §4h eval contract; findings S2, S3, S11, S12, S14-S19.
- Phase 02 (encoders), phase 03 (code changes), phase 12 (ceilings).

## Overview

- **Priority:** P1. **Status:** pending. **Effort:** ~8-42 CPU-hours across tiers; ~16h attended.
- Decide, under a pre-registered rule, whether the redesigned encoder is better relational modelling or
  merely scene fitting — and produce every number the thesis quotes.

## Key Insights

- **Two baselines are GATES, run before any encoder training.** S1 (detector-rule) is the number most
  likely to force a reframing of the entire accuracy claim: the detector's 8 classes are safe/unsafe
  counterpart pairs, so a 3-feature rule may already solve the binary task.
- **The 125-clip test split gives a Wilson half-width of ±8.7 pp** — too wide to resolve the effect sizes
  at stake. Therefore **k-fold-691 is the primary evidential protocol**; test is one confirmatory touch.
- **Background co-clustering (measured, 691 clips, k=8):** carrying 3↔7 **1.00**, walkway 0↔4 **0.73**,
  intervention 1↔5 **0.46**, panel 2↔6 **0.12**. Cluster 5 (101 clips, all class 0/1) has **zero** safe
  clips. A binary classifier can therefore score well by recognising *the scene*, not *the behaviour*.
  Panel caveat: the descriptor is a temporal median of the raw frame and an open door is itself a large
  static background change, so "different camera" and "different panel state" **cannot be disentangled**
  by this descriptor. The other three pairs differ behaviourally → their numbers are cleaner.
- **LCO folds are severely class-skewed** (group 5 = 101 unsafe / 0 safe; group 4 = 61% safe; groups 6,7
  n=14). Therefore **pool OOF predictions across all 8 folds and report pooled metrics with clip-level
  bootstrap CI — never per-fold accuracies.**
- **Equal-recipe rule (non-negotiable):** the existing 4-class numbers (blind 54/64, k-fold 80.8±4.0%,
  LCO macro 81.2%) may appear as **historical context only, never as a baseline row in a binary table** —
  five things differ at once (label space, task, class prior, 516→691 pool, token/geometry tag).
- Test-touch accounting: pooled k-fold/LCO over the 691 pool = **0 touches**. Budget exactly **two** touches.

## Requirements

Every experiment states command, protocol, seeds, wall-clock, touches, claim. Selection on val/OOF only;
≥5 seeds for every quoted number; mean ± sd **and** range always.

## Architecture — systems

| # | System | Role |
|---|---|---|
| S0 | constant/prior | information-free floor |
| **S1** | **detector-rule baseline** — LogReg / depth-≤2 tree over the 8-dim "max confidence per entity class" clip descriptor, fit on train only | **GATE, runs first** |
| **S2** | **scene-appearance baseline** — whole-frame MobileNet feats, T=16 mean-pooled, LogReg `class_weight='balanced'` | **PRIMARY CONTROL** (not a courtesy row); needs a 691 rebuild |
| S3 | **E0 retrained binary** | null hypothesis |
| S4 | **E1 d=64, T3** | the proposal |
| S5 | E1 d=96 | capacity check |
| S6 | E2 | attribution control |
| S7 | E3 | anti-overfitting floor |

## Protocols

- **P-A blind** — 566 train / 125 test; val = 18% of train **stratified on (label, view)** per brief §4h.1.
  The current code stratifies on **label only** → a code change (phase 03). 5 seeds. **Two declared test
  touches total: E0-binary and the winning encoder**, pre-registered in `reports/test-touch-log.md`
  *before* running.
- **P-B k-fold-691** — stratified, 5 folds × 2 repeats, 5 seeds. **Primary evidence.** Paired comparisons
  use **Nadeau–Bengio corrected t** (SE inflation ×3.7).
- **P-C LCO-691** — 8 background-cluster groups from `sweep_results/zone_prior/clip_scene_map_691.json`.
  **Required for every headline claim.** Pooled-OOF reporting only.

## Pre-registered decision rule (fixed BEFORE any run)

Let `Δ_rand = metric(E1) − metric(E0)` under k-fold-691 and `Δ_LCO` = the same under LCO-691.

1. **E1 is credited as better relational modelling only if** `Δ_LCO > 0` with its NB-corrected /
   paired-bootstrap CI excluding 0, **AND** `E1 − S2` under LCO has a paired-bootstrap CI excluding 0.
2. **If `Δ_rand > 0` but `Δ_LCO`'s CI includes 0 → conclude scene fitting**, state it **in those words**,
   and report the gain as non-transferable across cameras.
3. If `E1 ≈ S2` under LCO → the finding is that the binary task is largely scene recognition on this dataset.
4. If `|Δ|` small with CI including 0 → pre-committed wording: *"E1 matches E0 with 3.2× fewer parameters
   and 12.7× fewer attention pairs"* — a **parameter-efficiency result** for the deployment chapter.
5. If `Δ < 0`, CI excludes 0 → report the loss plainly; E2/E3 identify which inductive bias was wrong.

*A committee accepts a well-instrumented negative result. It does not accept a positive result whose CI
overlaps zero, nor a comparison against a differently-trained baseline.*

## Metrics

All via `evaluate_suite` (no new metric code), passing `class_names=['Safe','Unsafe']` and
**`unsafe_classes=[1]` explicitly** — `eval_utils/cost_matrix.py` defaults `unsafe_classes` to ALL classes
and will otherwise **silently** produce a wrong cost matrix.
Accuracy; **balanced accuracy / macro-F1 (headline)**; unsafe recall/precision/F1; PR-AUC + ROC-AUC;
expected cost under the declared **10:1** matrix; operating point at 0.95 unsafe recall; ECE/MCE 15-bin +
reliability diagram pre/post temperature scaling (fitted on val, seconds, no retraining).
τ is selected on val/OOF only and **frozen before any test touch**.
**Per-sub-class recall for all 8 sub-classes** — marked **descriptive, not evidential** (measured per-class
seed spreads 15.3-21.1 pp forbid inference). **Pre-committed: the panel pair's numbers are published even
if embarrassing.**
Statistics: clip-level block bootstrap 5,000 resamples; Wilson intervals; NB-corrected t for k-fold means.

**Paired-significance rule (binding for all new binary experiments).**
- Whenever we own **both** sets of per-clip predictions — encoder variants, the S1 detector-rule gate, the
  S2 scene control, ablations — the **exact paired test (McNemar exact)** must be used.
- The **bound** (`mcnemar_discordance_bound`) is acceptable *only* when comparing against a published
  aggregate whose per-clip outputs we cannot obtain.
- `two_proportion_z` on paired data is never acceptable (phase 03 D1, landed). It assumes independent
  samples and was being applied to two classifiers scored on the same clips.
- Every reported comparison carries **b, c (discordant counts), the exact p or the bound interval, and the
  paired CI** — never a bare p-value.

**Structural consequence for the thesis narrative:** comparisons against the published Unsafe-Net numbers
**can never be tested for paired significance**, because their per-clip outputs do not exist for us. The
thesis therefore must **not** rest its central claim on beating the paper's aggregate. The defensible
claims are exactly those where we control both sides — encoder variants, the detector-rule gate, the
scene-appearance control — which is what this suite is built around.

## Tiered schedule

| Tier | Content | Wall-clock | Touches |
|---|---|---|---|
| **0 Prep** | bin8 tokens (~28 min), `build_wholeframe_features.py` 691 rebuild (~10 min), scene map (**done**), **calibration run (~10 min)** | ~50 min | 0 |
| **1 GATES** | S1 rule + S2 scene, train-CV + k-fold + LCO (LogReg/tree, no NN) | **<20 min** | 0 |
| **2 Screening** | E0, E1@64, E1@96, E2, E3 — k-fold-691 only, 3 seeds × 1 repeat | 1.4-8.6 h (overnight) | 0 |
| **3 Confirmatory** | E0 + winner: k-fold 5 seeds × 2 repeats + LCO 5 seeds + blind 5 seeds | 4.3-25.6 h (2-3 nights) | 0 until the touch |
| **4 Ablations** | winner only, k-fold 3 seeds × 1 repeat | 0.9-5.4 h | 0 |
| **5 Edge** | phase 04/05, E0 + winner | ~1.5 h on 19/9 | 0 |

Ranges are 6× wide because two logged throughput anchors disagree; the calibration run collapses them.
**Lane discipline: one heavy job at a time** — violating it is what produced the 6× spread.

## Ablations (winner only, k-fold-691, 3 seeds)

| Ablation | Question |
|---|---|
| T1 vs T2 vs T3 coordinates | how much of the score is absolute-position camera identity? |
| **class-id-blinded** (`cls_embed` zeroed) | **the single most thesis-defensible experiment** — if binary accuracy survives, the claim is about geometry+appearance; if it collapses, the encoder is decoration on S1 |
| geometry-free (4 zone columns zeroed) | bounds the wrong-polygon noise (≈50 safe clips inherited a low-confidence view) |
| appearance-free (64 ROI dims zeroed) | separates relational geometry from texture |
| *nice-to-have:* re-cluster on a motion / non-panel descriptor | tests the panel-confound hypothesis — **off the critical path** |

## Related Code Files

Modify: `run_kfold_relation.py` (`n_cls`, `--scene_map`→691, **`use_cw=True`**), `run_lco_eval.py`
(`N_CLASSES`, default map → `clip_scene_map_691.json`, pooled-OOF reporting),
`build_wholeframe_features.py:21` (`UNSAFE=[0..7]` → 691 rebuild for S2).
No change to `eval_utils/suite.py`, `unsafenet/metrics.py`, `eval_utils/cost_matrix.py` — class-count agnostic.

## Implementation Steps

1. Tier 0 prep + **calibration run**.
2. **Tier 1 gates. Read them before spending any CPU-hours.** If S1 is near ceiling → **stop and replan**
   (see Risk R1); notify the user immediately.
3. **Verify in the built tokens that safe-clip geometry columns are not identically zero** before trusting
   any number — the original fatal leak ("zero geometry ⇒ safe") is fixed by `clip_view_map_691.json`
   (175/175 assigned, 0 `na`), and this check confirms it.
4. Tier 2 screening overnight → pick the winner → **freeze architecture 18/9**.
5. Tier 3 confirmatory; report S1/S2/E0/E1 side by side under **both** k-fold and LCO.
6. Tier 4 ablations.
7. Apply the decision rule mechanically; write the conclusion sentence it dictates.
8. **The two test touches, last:** pre-register both entries in `reports/test-touch-log.md`, get user
   go-ahead, then run. After that, **no new accuracy experiments** (G6).
9. Print **both contamination ceilings (124/125 hard, ≈121/125 practical)** in every results table (phase 12).

## Todo List

- [ ] Tier 0 + calibration
- [ ] **S1 gate** read (escalate if near ceiling)
- [ ] **S2 gate** (691 rebuild) under k-fold + LCO
- [ ] Safe-clip geometry non-zero verified
- [ ] Tier 2 screening; winner chosen; arch frozen 18/9
- [ ] Tier 3 k-fold + LCO + blind
- [ ] Tier 4 ablations (incl. class-id-blinded)
- [ ] Decision rule applied; wording fixed
- [ ] **Two pre-registered test touches (G6)**
- [ ] Ceilings printed in every table

## Success Criteria

G6: winner decided by the pre-registered rule; S1/S2/E0/E1 reported side by side under k-fold **and** LCO;
per-sub-class table complete incl. the panel pair; exactly two logged test touches; ceilings printed; every
number reproducible from a logged command.

## Risk Assessment

**R1 (high) S1 ≈ encoder** → restructure the claim: the contribution becomes the evaluation protocol + LCO
transfer + edge characterisation, with the honest sentence *"a 3-feature detector rule matches a 298k-param
Transformer on this dataset"*. The thesis survives; the accuracy claim does not.
**R6 (high) scene fitting** → the §decision rule exists to prevent quoting a random-split win.
**R3 seed variance** (2.08 pp aggregate sd, 11 pp same-distribution) → ≥5 seeds, paired seeds, never one seed.
**R4** test too small → k-fold is the evidence. **R5** prior shift → weights ON, balanced metrics.
**R9 scope creep** → candidate list frozen; no loss zoo, no augmentation search, no architecture search.
**R10** contaminated clips misread as model failure → print both ceilings.

## Security Considerations

The test split is governed by the eval contract; an accidental test read is a reportable incident. No
selection, tuning, calibration or threshold-fitting on test.

## Next Steps

Feeds phase 07 (number sheet) and the pre-registered limitations in phase 08.
