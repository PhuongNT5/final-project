# Design — Redesigned Transformer Encoder for the Binary Safe/Unsafe Clip Classifier

Date: 2026-09-16 · Scope: DESIGN ONLY (no code modified, no training run) · Author: planner

Grounded in measured repo facts, not in the paywalled paper. Every parameter count below was computed
from the actual constructor in `model_relation_head.py`; every timing anchor is a logged wall-clock
number from this repo. Where Agent A's review
(`research-2026-09-16-unsafe-net-paper-review-and-binary-task-design.md`) is hypothetical, it is
marked as such.

---

## 0. Recommendation in one paragraph

Replace the flat 129-token encoder with **E1 = a factorized set-then-time encoder at d=64**:
a permutation-invariant intra-frame stage over the ≤8 entity tokens of each frame (no positional
embedding at all — box slots are a confidence-ranked *set*, not a sequence), masked mean‖max pooling
to one embedding per frame, then a 17-token temporal stage (CLS + 16 frames) carrying the only
positional embedding in the model. **94,380 params vs 297,860 today (0.32×)**, positional params
1,088 vs 16,512 (0.07×). Heads: binary main + 8-way sub-class aux + per-frame binary aux. Default
token variant is **frame-relative coordinates** (Δ to the per-frame valid-box centroid), which keeps
inter-entity geometry while removing the absolute-position camera fingerprint — computable from the
existing tokens with no re-tokenization. The headline claim is decided by **LCO-691, not the random
split**, and the old encoder must be retrained under a byte-identical binary recipe or the comparison
is worthless.

---

## 1. Diagnosis of the current encoder

`RelationTransformer(n_cls, n_entity=8, d=128, layers=2, heads=4, ffn=256, dropout=0.3, t=16,
n_max=8, cls_embed_dim=32, n_num=73)` — `model_relation_head.py`.

### 1.1 Parameter budget (measured, feat-74 token set)

| Component | Params | Note |
|---|---:|---|
| `enc` (2 × TransformerEncoderLayer, d=128, ffn=256) | 264,960 | 89% of the model |
| `pos` (1+T·N = 129 slots × 128) | 16,512 | free table, learned from 566 clips |
| `head` (128→64→n_cls) | 8,386 | n_cls=2 |
| `num_proj` (73→96) | 7,104 | |
| `frame_head` (128→n_cls) | 258 | n_cls=2 |
| `cls_embed` (8×32) | 256 | |
| `token_ln` | 256 | |
| `cls_token` | 128 | |
| **Total (binary)** | **297,860** | 4-class variant = 298,248 |

Blind protocol fits on 566 − 18% val = **464 clips → 642 params per training clip**. Pooled k-fold
(691 clips, ~470 per fold fit) → 634. There is no regime in this project where this model is not
over-parameterized by 2-3 orders of magnitude relative to sample count. The already-observed
symptoms: same-distribution per-seed spread of **11.0 pp** (47–54/64, brief S20), LCO per-class recall
seed spreads of **15.3–21.1 pp** (S16/S18), aggregate top-1 seed sd 2.08 pp. Those are
variance-of-an-overfit-model signatures, and they are the reason the thesis currently cannot quote a
per-class number as evidence.

### 1.2 No separation of intra-frame co-occurrence from temporal evolution

All 128 entity tokens + CLS go into one flat attention over 129 positions (16,641 attention pairs).
Nothing in the architecture encodes "these 8 tokens are the same instant" versus "these 16 are the
same slot across time". The only carrier of that distinction is the `pos` table — i.e. the model must
*learn from 566 clips* a fact that is free structural knowledge. Two consequences:

- The two physically distinct questions — *what co-occurs right now* (person + opened panel in one
  frame) and *how it evolves* (person crosses the walkway boundary over time) — are collapsed into
  one undifferentiated soup. Neither is expressed cleanly.
- 89% of capacity sits in layers that could, in principle, memorize clip-specific token constellations.

### 1.3 The positional embedding entangles frame index with box slot — and the slot is meaningless

`pos` index = `1 + t*8 + j`, where `j` is the **confidence rank** within frame `t`
(`build_relation_tokens.py`: `np.argsort(-conf)[:N_MAX]`). There is no tracker. Box `j` in frame `t`
and box `j` in frame `t+1` need not be the same physical object — the rank permutes whenever detector
confidence jitters. So:

- The model **cannot** express "the same entity moved" (no identity across time to attend along).
- The model **can** express "position 37 is special" — which on 566 clips is pure memorization.
- 128 of the 129 position vectors are learned for slots whose semantics are unstable *within a single
  clip*, let alone across clips. 16,512 params — 29 params per training clip — spent on noise.

This is the single clearest design defect, and it is cheap to fix.

### 1.4 Padding and masking

Measured on `tokens_yolo26app_train.pt` (452 unsafe train clips):

- mean **3.66 of 8** valid boxes per frame → **54.3% of all token slots are padding**;
- frames with zero detections: 0.3%.

Handling is *correct but wasteful*: `src_key_padding_mask` excludes padded keys, and CLS is never
masked (so no all-masked row → no NaN, the usual failure mode is avoided). But padded slots still
receive their `pos` vector, still occupy the sequence, and still cost attention compute. More
importantly, the model spends half its sequence length on nothing, while the positional table is
sized for the padded worst case.

The per-frame aux path takes a masked mean over *encoder-output* tokens — i.e. post-attention,
already globally contextualized. It is therefore not really a "per-frame local" supervision signal;
it is a weak consistency regularizer. Also note the aux labels are the clip label broadcast to all 16
frames, which asserts the violation is visible in every frame — false for transient events, so the
aux carries label noise. (Keep it, but treat `aux_w` as a 2-point val-only choice, §3.4.)

### 1.5 Task-specific defect exposed by the binary reframe

`cls_embed` gives every token its detector class id, and the detector's 8 classes are *counterpart
pairs* (Person_Unauthorized/Authorized, Panel_Opened/Closed, Forklift_Overload/Safe_Load). Under the
binary task, the entity class id alone may nearly solve the problem — the same pathology already
documented for CarryOverload (100% recall from a 1-line detector-class rule, audit B4). If so, all
297,860 params are decoration on a lookup table. This must be *measured before* the redesign is
credited with anything (§4.1 gate, §4.6 class-id-blinded ablation).

---

## 2. Candidate encoders

Shared: input token = detector class id + 73 numeric (geometry + conf + 64-d ROI appearance);
T=16, N_MAX=8; masks as today.

### E0 — control: current flat encoder, retrained binary (**mandatory**, not optional)

```
[CLS] b0,0 b0,1 ... b0,7 | b1,0 ... b1,7 | ... | b15,0 ... b15,7     (129 tokens)
  +  pos[0..128]  (16,512 params; frame index × conf-rank entangled)
  →  2 × TransformerEncoderLayer(d=128, h=4, ffn=256, p=0.3)
  →  x[:,0] → head → 2 logits
```
**297,860 params.** Role: the only legitimate reference point. It is *not* a candidate design; it is
the null hypothesis.

### E1 — RECOMMENDED: factorized set-then-time ("F-STE")

```
per frame t (independently, B·T sequences of length 8):
    tok → [cls_embed(16) ‖ num_proj(73→d-16)] → LayerNorm
    ── NO positional embedding ──  (entities in a frame are an unordered SET;
                                    conf-rank is arbitrary → enforce permutation invariance)
    → 1 × TransformerEncoderLayer(d, h=4, ffn=2d, p=0.2)   [INTRA-FRAME RELATIONS]
    → masked mean ‖ masked max over the ≤8 valid boxes  (2d)
    → Linear(2d → d)  → frame embedding f_t        (empty frame → learned `empty_frame` vector)

temporal stage (1 sequence of length 17):
    [CLS] f_0 f_1 ... f_15
      + temporal_pos[0..16]        (17 × d only — 1,088 params at d=64)
    → 1 × TransformerEncoderLayer(d, h=4, ffn=2d, p=0.2)   [TEMPORAL EVOLUTION]
    → pool = CLS ‖ mean(f'_t) ‖ max(f'_t)   (3d)
    → heads (§3)
```

| | d=64 (**primary**) | d=96 (capacity check) |
|---|---:|---:|
| intra-frame layer | 33,472 | 74,784 |
| temporal layer | 33,472 | 74,784 |
| `frame_proj` (2d→d) | 8,256 | 18,528 |
| `temporal_pos` (17×d) | 1,088 | 1,632 |
| embed (`cls_embed`+`num_proj`+LN) | 3,808 | 6,240 |
| `cls_token` + `empty_frame` | 128 | 192 |
| heads (bin + 8-way sub + frame) | 14,156 | 21,132 |
| **Total** | **94,380** (0.32× E0) | **197,292** (0.66× E0) |
| params / training clip (464 fit) | **203** | 425 |

Inductive biases asserted, each matched to a measured defect:

| Bias | Fixes |
|---|---|
| Intra-frame stage is permutation-invariant (no slot embedding) | §1.3 — removes 15,424 memorizable positional params and the unstable conf-rank identity |
| Attention is factorized set→time | §1.2 — co-occurrence and evolution become architecturally distinct |
| Position exists only over frame index (17 slots) | §1.3 — time order is real and stable; slot order is not |
| Pooling is mean‖max, not CLS-only | a single-entity, single-frame violation survives max-pooling; mean carries scene-level prevalence. Also 54% padding (§1.4) makes masked pooling the natural reduction |
| Attention pairs: 16·8² + 17² = **1,313** vs 16,641 | ~1.5-2.5× cheaper per epoch (§6) |

What could go wrong:
- **Identity is destroyed at the pooling step.** The temporal stage sees frame *summaries*, so
  "person X specifically moved into the restricted zone" is inexpressible; only "some person was in
  the zone at t, and the zone-IoU distribution shifted" survives. Mitigation: max-pooling preserves
  the single most extreme entity per dimension, which is precisely the violation signal. Tracking is
  deliberately **not** added (YAGNI; no tracker exists in the pipeline, and adding one would confound
  the encoder comparison).
- Two stacked stages = a deeper gradient path on 566 clips. Mitigation: 1 layer each, d=64, and E3
  exists to tell us if depth is buying anything at all.
- If the true signal is genuinely global across frames *and* entities simultaneously, factorization
  is a hard constraint that costs accuracy. E2 exists to separate that hypothesis from the positional
  one.

### E2 — surgical control: flat attention, factorized position only

Identical to E0 except `pos` (129×128) is replaced by a frame-only table (17×128 = 2,176) broadcast
across the 8 slots of each frame; nothing else changes. **283,524 params** (+1,032 with the 8-way aux
head).

Purpose: this is the *scientific* control that makes E1's result interpretable. E0→E2 isolates "was
the problem the positional prior?"; E2→E1 isolates "was the problem the flat attention?". Without E2,
a win by E1 is a win by an unattributable bundle of three changes — exactly the kind of result a
committee pushes back on.

Risk: it keeps 283k params and 16,641 attention pairs over a sequence that is 54% padding; the most
likely outcome is "indistinguishable from E0", which is itself a publishable attribution.

### E3 — deliberately smaller anti-overfitting control (no attention at all)

```
tok → [cls_embed(8) ‖ num_proj(73→56)] → LayerNorm(64)
    → masked MEAN over the ≤8 boxes            (DeepSets; no attention)
    → Linear(64→64) + GELU                     (per-frame MLP)
    → mean ‖ max over the 16 frames   (128)
    → heads
```
**18,044 params (0.06× E0; 39 params per training clip).**

Purpose: the honest floor for *learned* models. If E3 ≈ E1 within CI, then at n=691 attention buys
nothing and the defensible thesis sentence is "a permutation-invariant pooled encoder matches a
Transformer on this dataset; the Transformer's extra 5.2× parameters are unjustified at this sample
size" — a stronger, more honest contribution than a 1-pp win with overlapping CIs.

Risk: cannot express conjunctions ("person AND opened panel in the same frame"); if it loses clearly,
that loss *is* the evidence that relational modelling matters — which is the thesis' actual claim.
Either outcome is informative, which is why this control is non-negotiable.

### Summary

| | params | pos params | attn pairs | params/clip | role |
|---|---:|---:|---:|---:|---|
| E0 flat (today) | 297,860 | 16,512 | 16,641 | 642 | null hypothesis, must be retrained binary |
| E1 F-STE d=64 | **94,380** | 1,088 | 1,313 | 203 | **recommended** |
| E1 F-STE d=96 | 197,292 | 1,632 | 1,313 | 425 | capacity check |
| E2 flat + frame-pos | 283,524 | 2,176 | 16,641 | 611 | attribution control |
| E3 pooled MLP | 18,044 | 0 | 0 | 39 | anti-overfitting floor |

---

## 3. Head design

### 3.1 Output parameterization — 2-logit softmax, not BCE

Use `n_cls=2` CE, **not** a single BCE logit. Reason: `eval_utils/suite.py::evaluate_suite` and
`unsafenet/metrics.py::report` derive `n_cls` from `len(class_names)` / `probs.shape[1]` and are
already class-count agnostic. A 2-column softmax plugs into the entire §4 contract (ROC/PR, ECE/MCE,
cost matrix, bootstrap, McNemar) **with zero new metric code**. A scalar BCE logit would require a
parallel metrics path for no modelling benefit. Agent A's weighted-BCE proposal is functionally
equivalent to weighted 2-class CE; take the form that reuses the contract.

Convention (fix once, repo-wide): `class_names = ['Safe','Unsafe']`, **index 1 = Unsafe = positive**.
`evaluate_suite(..., unsafe_classes=[1])` — note `eval_utils/cost_matrix.py` defaults
`unsafe_classes` to ALL classes, so omitting it silently produces a wrong cost matrix.

### 3.2 Multi-task heads

| Head | Input | Output | Weight | Rationale |
|---|---|---|---:|---|
| **Main binary** | pooled (3d) | 2 | 1.0 | the thesis task |
| **Sub-class aux** | pooled (3d) | 8 | λ_sub = 0.3 | labels are free (folder id); preserves continuity with the existing 4-class results; forces the encoder to keep sub-class structure instead of collapsing to one scene cue; yields per-sub-class diagnostics for §4.5 |
| **Per-frame aux** | frame token (d) | 2 | λ_frame = 0.5 | keeps the existing `aux_w=0.5` recipe (S1a); see label-noise caveat §1.4 |

`L = CE_bin(w) + 0.3·CE_sub + 0.5·CE_frame`, label smoothing 0.05 on all three (matches the existing
recipe in `run_relation_transformer.py`).

λ_sub ∈ {0, 0.3} and λ_frame ∈ {0, 0.5} are the **only** two hyper-parameters allowed to be tuned,
selected on validation/OOF only, on E1 only, then frozen and applied identically to E0/E2/E3. Any
further tuning breaks §4h and the equal-recipe requirement.

### 3.3 Class imbalance (452 unsafe : 114 safe in train)

Use the existing helper with `k=2`: `class_weights(y, k=2)`. Note it **mean-normalizes**
(`w / w.mean()`), so the actual weights are `w_unsafe = 0.403`, `w_safe = 1.597` (ratio 3.96), not
the `0.626 / 2.48` Agent A quotes — same ratio, different scale, and the scale matters because it
interacts with the learning rate. Quote the normalized values in the thesis.

**The decisive fact is prior shift, and it points to weights-ON:** train is 3.96:1 unsafe-heavy while
the **test set is near-balanced (64 unsafe / 61 safe)**. Inverse-frequency weighting makes the
effective training prior uniform, which *matches the evaluation prior*. This is the opposite of the
k-fold script's current `use_cw=False`, and the change must be stated explicitly in the thesis as a
deliberate, prior-matched decision — not slipped in. For pooled k-fold (516:175 = 2.95:1) the same
argument holds, and **balanced accuracy / macro-F1, not top-1, is the headline metric**.

### 3.4 Operating point and threshold

- The declared cost matrix (fixed 2026-09-06 in `reports/test-touch-log.md`) is **10:1** —
  missing an unsafe clip costs 10, a false alarm costs 1. Therefore argmax (τ=0.5) is *not* the
  reportable operating point.
- Select τ minimizing expected cost **on validation / OOF only**, via
  `evaluate_suite(..., selection_y_true=, selection_y_prob=, cost_matrix=, target_safety_recall=0.95)`.
  Report both the argmax row and the cost-selected row; the τ is frozen before any test touch.
- Calibrate with single-parameter temperature scaling fitted on val (no retraining, seconds). Report
  ECE/MCE (15-bin) and the reliability diagram before and after. With 2 classes and max-softmax
  binning the existing `ece_mce` works unchanged.

### 3.5 Absolute coordinates (the camera-identity confound)

`xc, yc` encode camera identity (the L1 ablation exists precisely for this), and the parallel
data-prep agent has now *measured* a scene/label correlation (below). But dropping `xc,yc` outright
also destroys all inter-entity geometry, which is the whole point of a relational encoder.
Resolution — three token variants, all derived from the **existing** tensors (no re-tokenization):

- **T1** full feat-74 (absolute coords) — shortcut-prone, run as an upper-bound reference only.
- **T2** feat-72 / `yolo26L1_` (xc,yc dropped) — matches the existing canonical LCO setup.
- **T3 (default for E1)** frame-relative coords: replace `xc,yc` with `xc − mean_valid(xc)`,
  `yc − mean_valid(yc)` computed per frame over valid boxes only, zeroed on padded slots.
  Translation-invariant w.r.t. scene layout, preserves entity-to-entity geometry.

E1's headline number uses **T3**; T1 and T2 are reported as ablations (§4.6).

---

## 4. Comparison protocol

### 4.0 The equal-recipe rule (non-negotiable)

**The old encoder must be retrained from scratch under the identical binary recipe.** Any comparison
against the existing 4-class numbers (blind 54/64, k-fold 80.8±4.0%, LCO macro 81.2%) is confounded
by *five* simultaneous changes: label space (4→2), task, class prior, data pool (516→691 clips),
and token tag/geometry (new `clip_view_map_691.json`). Those numbers may appear in the thesis only as
historical context, never as a baseline row in a binary table.

Held identical across E0/E1/E2/E3: token set, splits, seeds {0..4}, epochs, optimizer
(AdamW 1e-3, wd 0.01, cosine), batch 32, grad-clip 5.0, label smoothing, λ_sub, λ_frame, class
weights, val fraction, model-selection criterion, threshold policy. **Only the encoder module differs.**

### 4.1 Systems (baselines first — two of them are gates)

| # | System | Notes |
|---|---|---|
| S0 | constant/prior | information-free floor |
| **S1** | **detector-rule baseline** — logistic regression or depth-≤2 decision tree over the 8-dim "max confidence per entity class" clip descriptor, fit on train only | **GATE. Run before any encoder training.** The detector's classes are safe/unsafe counterpart pairs; if S1 is near ceiling, the encoder has no headroom and the thesis claim must be restructured (§5 R1) |
| **S2** | **scene-appearance baseline** — whole-frame MobileNet features, T=16 mean-pooled, LogReg `class_weight='balanced'` | **PRIMARY CONTROL, promoted from "a baseline" (see 4.2). Must be rebuilt for 691** (`build_wholeframe_features.py:21` hardcodes `UNSAFE=[0,1,2,3]`) and run under **both** k-fold and LCO |
| S3 | **E0 retrained binary** | the null hypothesis |
| S4 | **E1 d=64, T3** | the proposal |
| S5 | E1 d=96 | capacity check |
| S6 | E2 | attribution control |
| S7 | E3 | anti-overfitting floor |

### 4.2 The scene-fitting problem, and the comparison that decides it

Measured by the data-prep agent on all 691 clips (k=8 background clustering): safe/unsafe counterpart
pairs co-cluster **unevenly** — carrying 1.00, walkway 0.73, intervention 0.46, **opened vs closed
panel 0.12**. Cluster 5 (101 clips, all class 0/1) contains **zero** safe clips. A binary classifier
can therefore score well by recognising *the scene* rather than *the behaviour*. (The panel figure is
partly confounded: an open door is itself a large static change in the background descriptor, so
"different camera" and "different panel state" cannot be disentangled with this descriptor. Report as
an explicit limitation.)

Consequences, all mandatory:

1. **S2 is a primary control on the binary task over all 691 clips**, not a courtesy row.
2. **Every headline claim requires LCO-691 alongside the random split.** LCO groups come from
   `sweep_results/zone_prior/clip_scene_map_691.json` (exists, 691/691 mapped, k=8).
3. **Pre-registered decision rule.** Let Δ_rand = metric(E1) − metric(E0) under k-fold-691 and
   Δ_LCO = the same under LCO-691.
   - E1 is credited as better relational modelling **only if** Δ_LCO > 0 with its NB-corrected /
     paired-bootstrap CI excluding 0, **and** E1 − S2 under LCO has a paired-bootstrap CI excluding 0.
   - **If Δ_rand > 0 but Δ_LCO's CI includes 0 → conclude scene fitting**, state it in those words,
     and report the gain as non-transferable across cameras. This is decided by the *pair* of
     comparisons, not by either alone.
   - If E1 ≈ S2 under LCO, the finding is that the binary task is largely a scene-recognition task on
     this dataset — report it (§5 R1).

### 4.3 Protocols

- **P-A blind** (566 train / 125 test). Val = 18% carved from train, **stratified on (label, view)**
  per brief §4h.1 — the current code stratifies on label only; this deviation must be corrected for
  the binary runs. 5 seeds, ensemble + per-seed range. **One declared test touch per pipeline
  variant**, pre-registered in `reports/test-touch-log.md` *before* running. Budget only **two**
  touches: E0-binary and the winning encoder.
- **P-B k-fold-691** (stratified, 5 folds × 2 repeats, 5 seeds). **Primary evidential protocol** —
  125 test clips give a Wilson half-width of ≈±8.7 pp, too wide to resolve the effect sizes at stake.
  Paired comparisons use Nadeau–Bengio corrected t (SE inflation ×3.7 as already established).
- **P-C LCO-691** (8 background-cluster groups). Required for every headline claim (4.2).
  **Caveat to print alongside the numbers:** the folds are severely class-skewed — group 5 is
  101 unsafe / 0 safe (per-fold accuracy is meaningless there), group 4 is 61% safe, groups 6 and 7
  have n=14. Therefore **pool OOF predictions across all 8 folds and report pooled metrics with
  clip-level bootstrap CI; do not report per-fold accuracies.**

### 4.4 Metrics (all via `evaluate_suite`, no new metric code)

Accuracy; balanced accuracy / macro-F1 (headline under prior shift); **unsafe recall / precision /
F1**; PR-AUC (positive = Unsafe) and ROC-AUC; expected cost under the declared 10:1 matrix; operating
point at 0.95 unsafe recall; ECE/MCE 15-bin + reliability diagram, pre- and post-temperature.

**Per-sub-class recall for all 8 sub-classes** (binary prediction grouped by true sub-class 0..7, plus
the 8-way aux head's own recalls) so the panel pair's contribution is visible rather than absorbed
into the binary aggregate. Mark these **descriptive, not evidential** — measured per-class seed
spreads of 15.3–21.1 pp forbid inference from them.

Statistics: clip-level block bootstrap, 5,000 resamples; Wilson intervals for proportions; **McNemar
exact** for paired system comparisons on identical clips (blind + pooled OOF); Nadeau–Bengio for
k-fold means. **≥5 seeds for every quoted number**; always report mean ± sd *and* range.

### 4.5 Ceilings and contamination — must be printed in every results table

From `sweep_results/duplicate_audit.json` (691 files, 658 distinct, 32 md5 groups):

- **Hard binary test ceiling = 124/125 (99.2%).** `0_te23.mp4` and `4_te9.mp4` are byte-identical and
  both in the test set with **opposite binary labels**. Any deterministic classifier is necessarily
  wrong on exactly one of them.
- **Practical ceiling ≈ 121/125 (96.8%)** for any model that fits its training data: three safe test
  clips have byte-identical **unsafe** train twins — `4_te4` (≡ `0_tr123`, `1_tr55`), `4_te7`
  (≡ `0_tr138`), `5_te13` (≡ `0_tr136`). The exact pixels are labelled Unsafe in training and Safe at
  test. This is a *binary-specific* contamination: it did not exist in the 4-class protocol because
  the safe clips were not in the dataset.
- 4 of the 32 groups are safe↔unsafe contradictory (binary); **all 32** are contradictory for the
  8-way aux head (e.g. `4↔6` is safe-safe, `1↔2` unsafe-unsafe — harmless binary, harmful to aux).
- Dedupe policy: headline = `keep_one` (train) + `keep` (test), unchanged from the 4-class canonical;
  add one sensitivity row with `drop_contradictory` (`unsafenet/dedupe.py`).

Geometry caveats (from the Lane-B prep):
- `annotation_guide/zones/` has **no `view_2.json` / `view_7.json`** → 56 unsafe + 30 safe clips get
  all-zero geometry. Pre-existing gap, now applied symmetrically to both sides of a counterpart pair —
  report it, do not "fix" it mid-experiment.
- 175 safe clips inherit views from cluster-dominant unsafe views; clusters 1 (dominance 0.40) and 4
  (0.60) are low-confidence, so ~50 safe clips may have geometry computed against the **wrong**
  camera's polygon — an asymmetric noise source affecting safe clips only. Bound it with a
  **geometry-free ablation** (zero the 4 geometry columns for all clips).
- The original fatal leak — safe clips having no view at all → all-zero geometry → "zero geometry ⇒
  safe" — **is fixed** by `clip_view_map_691.json` (175/175 assigned, 0 `na` fallbacks). Verify in the
  built tokens that the safe-clip geometry columns are not identically zero before trusting any number.

### 4.6 Ablations (winner encoder only, k-fold-691, 3 seeds)

| Ablation | Question |
|---|---|
| T1 vs T2 vs T3 coordinates | how much of the score is absolute-position camera identity? |
| **class-id-blinded** (`cls_embed` zeroed / class column dropped) | is the task a detector-class lookup? **The single most thesis-defensible experiment** — if binary accuracy survives, the claim is about geometry+appearance; if it collapses, the encoder is decoration on S1 |
| geometry-free (4 zone columns zeroed) | bounds the wrong-polygon noise of §4.5 |
| appearance-free (64 ROI dims zeroed) | separates relational geometry from texture |

---

## 5. Risk assessment and fallback

| # | Risk | Likelihood | Detection | Response |
|---|---|---|---|---|
| R1 | **Rule baseline S1 ≈ encoder** — the detector's counterpart classes already solve the task | **High** | S1 gate, train-CV, before any encoder run | Restructure the claim: contribution becomes the evaluation protocol + LCO transfer + edge characterization, with the honest sentence "a 3-feature detector rule matches a 298k-param Transformer on this dataset". Thesis survives; the accuracy claim does not |
| R2 | **E1 does not beat E0** | Moderate-High | P-B with NB-corrected CI | **Planned negative result — see 5.1** |
| R3 | Seed variance swamps the effect (measured 2.08 pp sd aggregate, 11 pp same-dist) | High | per-seed range always reported | ≥5 seeds; paired seeds for McNemar; k-fold as primary; never quote a single seed |
| R4 | Test set too small (±8.7 pp) | Certain | Wilson width | k-fold-691 is the evidence; test = one confirmatory touch |
| R5 | Prior shift (train 3.96:1, test ~1:1) | Certain | — | class weights ON (§3.3); balanced metrics headline |
| R6 | Scene fitting mistaken for relational modelling | **High** (0.12 panel overlap) | §4.2 decision rule | pre-registered Δ_rand/Δ_LCO rule; S2 as primary control |
| R7 | Gate 2 (brief §8 bans accuracy experiments after day 6) | Certain | — | written exception in `brief.md` + pre-registered test-touch entries **before** any run; deadline flex 1-2 weeks covers it |
| R8 | Jetson 2026-09-19 needs a frozen architecture | Certain | calendar | **freeze by 09-18**; export both E0 and the current best. Note the edge budget currently FAILS (1825 ms/decision, 1.13-1.71 dec/s vs 3.0 required) and is dominated by detector+appearance, **not** the relation head — a 3.2× smaller encoder will **not** fix it. Do not claim it will |
| R9 | Scope creep / architecture search | Moderate | — | candidate list frozen at E0/E1(d=64,d=96)/E2/E3 + the 4 ablations. No new architectures, no loss-function zoo, no augmentation search |
| R10 | Contaminated test clips (§4.5) read as model failure | Certain | duplicate audit | print both ceilings in every table |

### 5.1 Fallback: the negative result, planned in advance

Pre-register the claim **"E1 ≥ E0 under an identical binary recipe"** and report the NB-corrected
k-fold difference with its CI *whichever way it falls*. Three outcomes, all pre-committed:

1. **Δ_LCO > 0, CI excludes 0** → the designed claim holds; report it with the ablations that explain
   *why* (positional factorization vs attention factorization, via E2).
2. **|Δ| small, CI includes 0** → report: *"no significant difference at n=691; E1 reaches E0's
   accuracy with 3.2× fewer parameters and 12.7× fewer attention pairs."* This is a
   **parameter-efficiency result**, it is honest, and it is the better story for the deployment
   chapter. Framing it this way *in advance* is what makes the outcome useful instead of
   embarrassing.
3. **Δ < 0, CI excludes 0** → report the loss plainly, with E2 and E3 identifying which inductive
   bias was wrong (e.g. if E3 ≈ E0 too, the dataset does not support *any* of these distinctions and
   the finding is about the dataset, not the architecture).

A committee accepts a well-instrumented negative result. It does not accept a positive result whose
CI overlaps zero, or a comparison against a differently-trained baseline. The only genuinely losing
move is to run E1, see a small win on the random split, and quote it — which §4.2's decision rule
exists to prevent.

---

## 6. Cost table (CPU wall-clock)

### 6.1 Anchors — and an honest 6× uncertainty

Two logged anchors in this repo disagree:
- `w2-number-integrity-fixes.md`: **8.9 s/epoch at ~400 train clips** → 2.2 s per 100 clips per epoch.
- `a3b-lco-baseline-rebuild.md`: 5 seeds × 8 folds × 60 epochs in **~65 min** = 40 trainings →
  ~1.6 s/epoch at ~450 clips → 0.36 s per 100 clips per epoch.

Cause of the spread is machine contention (a3b itself notes ">90 CPU-min without finishing fold 1/8"
under 3 concurrent jobs, and warns to expect 1-3+ h). **Action: run a 10-minute calibration first**
(E0-binary, 1 seed, 5 epochs, machine otherwise idle) to collapse this before committing the
schedule. All figures below are given as **[fast .. slow]** using c ∈ [0.36 .. 2.2] s/100 clips/epoch,
60 epochs, per-variant multipliers E0 1.0, E1@96 0.75, E1@64 0.5, E2 0.95, E3 0.15.

### 6.2 Per-training and per-protocol (E0 units)

| Protocol | fit clips | trainings | per training | per variant, 5 seeds |
|---|---:|---:|---|---|
| P-A blind | 464 | 1/seed | 1.7 – 10.2 min | **8 – 51 min** |
| P-B k-fold-691, 1 repeat | ~470 | 5/seed | 1.7 – 10.3 min | **42 – 258 min** |
| P-B k-fold-691, 2 repeats | ~470 | 10/seed | " | **1.4 – 8.6 h** |
| P-C LCO-691 (8 groups) | ~514 | 8/seed | 1.9 – 11.3 min | **1.2 – 7.5 h** |

### 6.3 Scheduled plan (tiered — running everything on every protocol is not affordable)

| Tier | Content | Trainings | Wall-clock |
|---|---|---:|---|
| **0 Prep** | bin8 token build 691 (running, ~28 min), `build_wholeframe_features.py` rebuild for 691 (~10 min), scene map 691 (**done**), calibration run (~10 min) | – | **~50 min** |
| **1 Gates** | S1 rule + S2 scene, train-CV + k-fold + LCO (LogReg/tree, no NN) | – | **< 20 min** |
| **2 Screening** | E0, E1@64, E1@96, E2, E3 — k-fold-691 only, 3 seeds × 1 repeat (15 trainings each) | ~50 units | **1.4 – 8.6 h** (one overnight) |
| **3 Confirmatory** | E0 + winner only: k-fold 5 seeds × 2 repeats (50) + LCO 5 seeds (40) + blind 5 seeds (5) | ~142 units | **4.3 – 25.6 h** (2-3 nights) |
| **4 Ablations** | winner only: T1/T2/T3, class-id-blinded, geometry-free, appearance-free — k-fold 3 seeds × 1 repeat | ~30 units | **0.9 – 5.4 h** |
| **5 Edge** | ONNX export + Jetson benchmark, E0 + winner | – | **~1.5 h on 09-19** |
| | **Total** | | **≈ 8 – 42 CPU-hours** |

Lane discipline: **one heavy job at a time** (violating it is what produced the 6× timing spread).

### 6.4 Calendar (deadline flexible 1-2 weeks; Jetson 2026-09-19)

| Day | Work |
|---|---|
| 09-16 (tonight) | Tier 0 finishes (tokens already at 50/566); calibration run; **Tier 1 gates** |
| 09-17 | Read gates. If R1 fires → replan before spending any GPU-free CPU-hours. Else launch Tier 2 overnight |
| 09-18 | Pick winner; **freeze architecture for Jetson**; start Tier 3 k-fold |
| **09-19** | **Jetson: export + benchmark E0 and winner** (architecture frozen 09-18) |
| 09-20/21 | Tier 3 LCO + Tier 4 ablations |
| 09-22 | Two pre-registered blind test touches (E0-binary, winner); calibration + reliability |
| 09-23/24 | Write-up: results tables, ceilings, limitations, negative-result framing |

---

## 7. Implementation surface (specification only — no code changed by this document)

| File | Change needed |
|---|---|
| `model_relation_head.py` | add E1/E2/E3 classes; leave `RelationTransformer` untouched (E0 = pass `n_cls=2`); add the 8-way aux head |
| `run_relation_transformer.py` | `NAMES`; `class_weights(y, k=2)`; two `torch.zeros(..., 4)`; `range(4)` loops; `two_proportion_z(..., 64, 57, 64)` hardcodes; val split stratify on **(label, view)**; `--arch` and `--binary` flags; token-variant T3 transform |
| `run_kfold_relation.py` | `n_cls`; `--scene_map` → 691 map; `use_cw=True` |
| `run_lco_eval.py` | `N_CLASSES`; default scene map → `clip_scene_map_691.json`; pooled-OOF reporting (no per-fold accuracy) |
| `build_wholeframe_features.py:21` | `UNSAFE = [0..7]` → rebuild S2 features for 691 |
| `eval_utils/suite.py`, `unsafenet/metrics.py`, `eval_utils/cost_matrix.py` | **no change** — class-count agnostic; pass `class_names=['Safe','Unsafe']`, `unsafe_classes=[1]` |
| `reports/test-touch-log.md` | 2 new pre-registered entries, written **before** the runs |
| `plans/.../brief.md` §8 | written Gate-2 exception |

## 8. Explicitly out of scope (YAGNI)

No tracker / IoU box-linking; no graph neural nets; no pretrained video backbones; no augmentation
search; no architecture search beyond the frozen candidate list; no focal/BCE loss zoo (one weighted
CE); no new metric code; no detector retraining (the 8 classes already cover safe entities).

---

## 9. Unresolved questions

1. **Timing**: which anchor is real — 8.9 s/epoch or 1.6 s/epoch? Blocks precise scheduling; resolved
   by the 10-minute calibration run (§6.1). Every wall-clock figure here is a range because of it.
2. **S1 rule-baseline strength is still unmeasured** — the bin8 token build was at 50/566 at the time
   of writing, so the safe-clip entity-class distribution is unknown. This is the single number most
   likely to change the thesis' framing, and it is the first thing to compute.
3. **Panel pair (2↔6) 0.12 cluster overlap** cannot be disentangled into "different camera" vs
   "panel state changes the background pixels" with the current temporal-median descriptor. A masked
   descriptor (exclude detected-panel regions) would separate them, but that is new preprocessing —
   proposed as a limitation statement rather than an experiment.
4. **Cluster 5** (101 clips, zero safe) makes one LCO fold degenerate for the binary label. Pooled-OOF
   reporting handles it, but whether a k=10 map (`clip_scene_map_691_k10.json`, already built) gives a
   better-balanced fold structure has not been checked.
5. **λ_sub = 0.3** is an assumption, not a measurement; the 2-point val-only sweep is budgeted but its
   outcome may not transfer between encoders (it is fitted on E1 and imposed on E0/E2/E3 for recipe
   equality — a deliberate, documented compromise).
6. Whether the three train-twin-contaminated safe test clips should be **excluded** from the headline
   test number or merely footnoted. Recommendation: footnote + a `drop_contradictory` sensitivity row;
   excluding them from the declared test set after the fact would itself be a protocol violation.
