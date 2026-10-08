# Binary encoder family (E0-E3): implementation + Tier 2 screening

Deadline: architecture freeze 2026-09-18. This report covers design-2026-09-16-binary-encoder-redesign.md
§2 implementation, unit tests, 10-min calibration, and Tier 2 screening (k-fold-691, 0 test touches).

## 1. Implementation

Files:
- `model_relation_head.py` (71->84 lines): additive-only edit. `n_sub=None` ctor param (+`sub_head` if
  set), `_positional(x)` hook extracted from `forward()` (subclass point for E2), `forward(..., return_sub=False)`.
  Output packing rewritten as list-accumulation; verified against every existing call-site shape
  (plain call, `return_repr`, `return_frame` with/without `return_repr`) — legacy 4-class path is
  byte-identical when `n_sub=None, return_sub=False` (both defaults).
- `model_binary_encoders.py` (NEW, 156 lines): `E2FrameOnlyPos(RelationTransformer)` (17-slot
  frame-only positional table via `_positional()` override), `E1FSTE` (standalone: permutation-invariant
  intra-frame TransformerEncoderLayer, no pos-embed on entity slots -> masked mean||max pool -> temporal
  stage with CLS + 17-slot pos table -> pooled CLS||mean||max -> heads), `E3DeepSets` (masked-mean box
  pooling, no attention, frame MLP, mean||max pool -> heads). All three share E0's forward contract:
  `(cls_logits[, sub_logits][, frame_logits, frame_valid][, cls_repr])`. `build_encoder(arch, n_num, n_cls=2, n_sub=8, d=None)` factory.
- `run_relation_transformer.py` (286->443 lines): added `load_binary()`, `apply_coord_variant()`
  (T1/T2/T3), `main_binary()`, and binary-aware `run_epoch()`/`train_one()` (multi-head loss
  `CE_bin(w) + 0.3*CE_sub + 0.5*CE_frame`, label_smoothing=0.05). Legacy 4-class `main()` path left
  byte-unchanged (verified by re-run, see §2).
- `run_kfold_relation.py` (125->195 lines): added `load_pooled_binary()` (asymmetric dedupe: train=
  policy arg, test=always 'keep'), `--binary/--arch/--dim/--dedupe_task/--coord_variant` flags, binary
  branch in the fold loop (`use_cw=True` per design, unlike the legacy `use_cw=False` which is
  untouched), `evaluate_suite`-based scorecard, OOF filename `kfold_oof_softmax_bin8_binary_{ARCH}[_dN]_{policy}.pt`.
- `tests/test_binary_encoder_forward.py` (NEW), `tests/test_binary_encoder_wiring.py` (NEW).

### Param counts (n_num=73, feat=74, bin8_ tag) vs design table

| variant | measured | design | match |
|---|---|---|---|
| E0 | 298,892 | ~297,860 | +1,032 = the 8-way sub_head (design's 297,860 excluded it; identical otherwise) |
| E1@64 | 94,380 | 94,380 | exact |
| E1@96 | 197,292 | (capacity check, no exact target given) | consistent scaling |
| E2 | 284,556 | (n/a in table; E0-sized minus 2 positional tables) | consistent |
| E3 | 18,044 | 18,044 | exact |

E0/E1@64/E3 verified exactly against hand-derived component math from the design doc (intra+temporal
attention blocks, frame_proj, temporal_pos, embed, heads). E0's design figure (297,860) omits the
8-way sub-class head (+1,032 params); E0 including sub_head = 298,892, matching my measurement.

## 2. Unit tests (actual output)

`tests/test_binary_encoder_forward.py` (9 tests) — forward shapes all 4 archs x all return-flag
combos, all-padding-frame/-clip no-NaN, E1 intra-frame permutation invariance, E0 backward-compat:

```
PASS  test_all_archs_build_with_expected_param_order_of_magnitude
PASS  test_all_padding_clip_does_not_nan_all_archs
PASS  test_all_padding_frame_does_not_nan_all_archs
PASS  test_e0_backward_compat_default_call_matches_pre_binary_contract
PASS  test_e1_intra_frame_permutation_invariance
PASS  test_forward_all_flags_combined_all_archs
PASS  test_forward_default_shape_all_archs
PASS  test_forward_return_frame_shape_all_archs
PASS  test_forward_return_sub_shape_all_archs
ALL PASS
```

`tests/test_binary_encoder_wiring.py` (12 tests) — label mapping, coord variants (T1 identity, T2
drops xy, T3 zero-mean+zero-pad), dedupe task=binary plumbing, blind/kfold loader agreement:

```
PASS  test_apply_coord_variant_t1_is_identity
PASS  test_apply_coord_variant_t2_drops_xy_columns
PASS  test_apply_coord_variant_t3_zero_mean_per_frame_and_padding_zeroed
PASS  test_apply_coord_variant_unknown_raises
PASS  test_binary_label_mapping_sub_0_3_unsafe_4_7_safe
PASS  test_dedupe_task_binary_plumbed_not_unsafe4
PASS  test_kfold_and_blind_loaders_agree_on_binary_label_for_same_clip
PASS  test_kfold_load_pooled_binary_pools_691_and_dedupe_is_train_only
PASS  test_load_binary_test_untouched_by_default
PASS  test_load_binary_train_counts_and_dedupe_none
PASS  test_load_binary_train_drop_conflict_drops_exactly_four
ALL PASS
```

Legacy 4-class path re-run (`run_relation_transformer.py --seeds 1 --epochs 1`, no `--binary`):
params 292,104 (unchanged from pre-edit baseline), runs and reports identically to before —
confirms the additive edits to `model_relation_head.py` did not alter default behaviour.

## 3. Calibration (E0-binary, 1 seed, 5 epochs)

`ps` checked first: one other agent's job running (`build_wholeframe_features.py`, ~540% CPU, 8-core
machine) — proceeded anyway (light job). `.venv/bin/python run_relation_transformer.py --binary --arch E0 --seeds 1 --epochs 5 --dedupe drop_conflict`:

```
train 562 (unsafe 448) | test 125 (unsafe 64) | feat=74 n_num=73 | params=298,892
seed0: val_macro=80.7@e4 -> test 97/125 top1=77.6 macro=77.4
BLIND: balanced_acc=77.4% macro_f1=77.4% unsafe_recall=85.9% unsafe_precision=74.3% PR-AUC=0.829 ROC-AUC=0.821
```

Wall time 18.2s total (5 epochs, incl. ~4.5s fixed overhead for token load + model build) — a 25-epoch
follow-up run (72.2s total) isolates the marginal cost: **~2.7s/epoch for E0** at ~478 fit clips
(0.57s/100 clips/epoch), inside design's [0.36..2.2] anchor range, closer to the fast end. Cross-arch
epoch-cost calibration (10-epoch runs, same overhead subtracted): **E1@64 ~1.1s/epoch, E1@96
~1.35s/epoch, E2 ~2.76s/epoch, E3 ~0.13s/epoch**. E3's no-attention design is ~20x cheaper than E0, as
predicted; E1@64 is ~2.4x cheaper than E0 (design predicted ~2x).

## 4. Tier 2 screening (k-fold-691 pooled OOF, 0 test touches)

Launched in background (nohup, tracked via the harness's background-job mechanism — no busy polling):
`run_kfold_relation.py --binary --tag bin8_ --dedupe drop_conflict --k 5 --repeats 1 --seeds 3 --epochs 60`
for `--arch {E0, E1 --dim 64, E1 --dim 96, E2, E3}`, sequentially (avoids CPU oversubscription with the
concurrent agent job). ETA from the per-epoch calibration above: 15 trainings/arch x 60 epochs x
per-epoch cost ~= E0 41min + E1@64 17min + E1@96 20min + E2 41min + E3 2min ~= **2.0h total**, within
design's 1.4-8.6h window.

Dedupe: train='drop_conflict' (drops the 4 binary-opposite train clips), test='keep' (untouched) —
this is a screening protocol on pooled OOF, consumes 0 test touches per reports/test-touch-log.md
2026-09-17 pre-registration.

Actual wall time: 65 min total (22:39-23:44), faster than the 2.0h calibration-based ETA — the
concurrent agent job (`build_wholeframe_features.py`) finished partway through, freeing CPU.

### Results table (pooled OOF, 687 clips = 691 - 4 drop_conflict, 5-fold x 1 repeat x 3-seed ensemble)

| variant | params | s/epoch | bal_acc | safe recall | macro-F1 [95% CI] | unsafe recall | unsafe precision | PR-AUC | ROC-AUC | McNemar p (vs E0) | paired-boot Δmacc vs E0 [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 298,892 | 1.99 | 83.6% | 81.1% | 81.3% [78.0,84.5] | 86.1% | 93.0% | 0.976 | 0.926 | - (baseline) | - |
| E1@64 | 94,380 | 0.45 | 84.1% | 82.9% | 81.3% [78.1,84.6] | 85.4% | 93.6% | 0.970 | 0.923 | 1.000 (n.s., b=24 c=23) | +0.5pp [-2.3,+3.2] (n.s.) |
| E1@96 | 197,292 | 0.58 | 84.5% | 84.6% | 81.2% [78.0,84.3] | 84.4% | 94.1% | 0.973 | 0.922 | 0.784 (n.s., b=28 c=25) | +0.8pp [-2.1,+3.8] (n.s.) |
| E2 | 284,556 | 1.28 | 84.6% | 83.4% | 81.9% [78.7,85.0] | 85.7% | 93.8% | 0.972 | 0.918 | 0.868 (n.s., b=17 c=19) | +0.9pp [-1.2,+3.3] (n.s.) |
| E3 | 18,044 | 0.03 | 82.5% | 83.4% | 78.8% [75.4,82.1] | 81.6% | 93.5% | 0.968 | 0.910 | **0.027 (sig. worse, b=43 c=24)** | -1.1pp [-4.2,+2.1] (n.s.) |

Macro-F1 CI95 (BCa bootstrap, n_boot=5000, via `eval_utils.suite.evaluate_suite`) and the paired
bootstrap Δmacro-accuracy-vs-E0 CI (clip-resampled, n_boot=5000, same mechanism as
`run_lco_eval.py::paired_bootstrap_diff`, reimplemented in the scratch report script since that file
is another agent's) all include 0 or overlap E0's CI for every variant — **no k-fold-691 comparison
reaches significance under the bootstrap**, including E3 vs E0, despite E3's McNemar p=0.027. This is
not a contradiction: McNemar exact tests the discordant-pair count directly (more sensitive to a
consistent per-clip effect), while the clip-resampling bootstrap also captures between-clip metric
variance (macro-accuracy over a resampled class balance) — both point the same direction (E3 worse)
but the bootstrap CI is the more conservative, appropriately wider estimate. Read together: Tier 2
k-fold-691 supports "E3 is probably worse" and "E0/E1@64/E1@96/E2 are statistically indistinguishable"
— it does not support any accuracy claim for E1 over E0, consistent with the freeze recommendation below.

Fold-to-fold spread (5 folds, NB-corrected SE, top-1): E0 84.9±4.6% (NB-SE 3.48), E1@64 84.7±1.9%
(NB-SE 1.43, tightest), E1@96 84.4±2.3% (NB-SE 1.70), E2 85.2±3.5% (NB-SE 2.62), E3 82.1±3.3%
(NB-SE 2.49). All 5 variants beat the per-fold scene-only baseline (76.4±1.0%) at p<0.05 by
Nadeau-Bengio except E0 itself (p=0.066, n.s. — smallest sample-efficiency of the 5, ironically,
though its raw accuracy is mid-pack).

### Per-sub-class recall (all 8 source classes, pooled OOF)

| sub-class | n | E0 | E1@64 | E1@96 | E2 | E3 |
|---|---|---|---|---|---|---|
| 0 WalkwayViol | 207 | 89.9% | 88.4% | 86.0% | 87.0% | 83.6% |
| 1 UnauthorInt | 107 | 86.0% | 86.0% | 85.0% | 90.7% | 82.2% |
| 2 OpenedPanel | 142 | 75.4% | 74.6% | 75.4% | 74.6% | 71.1% |
| 3 CarryOverload | 56 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 SafeWalkway | 75 | 69.3% | 72.0% | 73.3% | 74.7% | 73.3% |
| 5 AuthorInt | 38 | 78.9% | 86.8% | 89.5% | 78.9% | 84.2% |
| 6 ClosedPanel | 32 | 93.8% | 87.5% | 90.6% | 93.8% | 90.6% |
| 7 SafeCarrying | 30 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

Class 2 (OpenedPanel) and class 4 (SafeWalkway) are the weak points across every variant (71-75% and
69-75% recall respectively) — a shared confusability, not an architecture-specific weakness.

### Tier 2 provisional read (SUPERSEDED by Tier 3 §5 below — kept for the record, do not cite alone)

E3 trails on this random-split screen (McNemar p=0.027 vs E0) and E1@64/E1@96/E2 are all statistically
tied with E0. A Tier-2-only read provisionally favoured freezing on E1@64 (smallest tied-accuracy
option). **This read is withdrawn**: Tier 3 (LCO-691, the pre-registered protocol) inverts it — see §5.

## 5. Tier 3: LCO-691 (pre-registered decision protocol, 0 test touches)

Ran `run_kfold_relation.py --binary --lco` (new `--lco`/`--scene_map` mode added this session) for
all 5 variants, leave-one-scene-group-out over the 8 KMeans(8) background-descriptor groups in
`clip_scene_map_691.json` (`[LIMITATION]`: proxy, not verified camera metadata — same caveat as the
read-only `run_lco_eval.py`), pooled OOF, drop_conflict/train-only dedupe (687 clips), 3 seeds x 60
epochs. Per-clip OOF saved to `sweep_results/relation_tokens/enc_{ARCH}[_dN]_lco-691_headline-drop_conflict_oof.json`
in the same row schema as the gates agent's `sweep_results/binary_gates/*_oof.json` (`{clip_id,
y_true, prob_unsafe, y_pred, fold_id}`) specifically so clip_id joins work directly — verified 687/687
join against both S1-logreg and S2 gate files (exact match, no coverage loss).

**Coordinator-flagged confound, checked before any conclusion:** `apply_coord_variant`'s default
differs by arch (E1->T3 frame-relative coords, E0/E2/E3->T1 absolute), so a raw E1-vs-E0 comparison
mixed architecture with input representation. Ran E1@64 LCO again with `--coord_variant T1` forced
(isolating arch from coords): balanced_acc 61.8% vs T3's 62.3%, Δ(T1-T3) on macro-acc across the whole
LCO pool is well inside single-digit-pp noise (E1@64-T1 vs E0: -5.0pp [-7.8,-2.0], vs T3's -4.5pp
[-7.3,-1.6] — same sign, same order of magnitude, CIs overlap heavily). **The confound is not
material**: forcing T1 does not rescue E1, if anything it is marginally worse. E0-with-T3 was
therefore NOT additionally run (the 2x2 would cost ~35min more CPU for a contrast already shown to be
sub-1pp against a 4-5pp architecture gap) — flagged here rather than silently skipped.

### Results table (pooled OOF, 687 clips, 8 scene-groups, 3-seed ensemble)

| variant | params | bal_acc | macro-F1 [95% CI] | vs Tier-2 k-fold bal_acc |
|---|---|---|---|---|
| E0 | 298,892 | 66.7% | 67.9% [63.7,71.7] | 83.6% (k-fold rank: mid) |
| E1@64 (T3, default) | 94,380 | 62.3% | 62.5% [58.5,66.5] | 84.1% (k-fold rank: 3rd) |
| E1@64 (T1, confound check) | 94,380 | 61.8% | 61.7% [57.8,65.7] | n/a (ablation) |
| E1@96 | 197,292 | 63.9% | 64.7% [60.6,68.9] | 84.5% (k-fold rank: 2nd) |
| E2 | 284,556 | 66.7% | 66.8% [62.8,70.7] | 84.6% (k-fold rank: 1st) |
| E3 | 18,044 | **69.6%** | **70.4%** [66.4,74.4] | 82.5% (k-fold rank: last) |
| S1-logreg (gates) | - | 59.7% | 60.6% [56.4,65.0] | 60.3% |
| S2 (gates) | - | 52.7% | 46.6% [43.0,50.3] | 74.1% |

**The random-split (Tier 2) ranking inverts under LCO (Tier 3).** k-fold ranked E2>E1@96>E1@64>E0>E3;
LCO ranks E3>E0=E2>E1@96>E1@64 — the cheapest, no-attention architecture goes from last to first, and
both E1 widths go from "tied-or-ahead of E0" to "behind E0." This inversion is itself a result: it says
the attention machinery (especially E1's narrower, no-positional-embed intra-frame stage) is fitting
something that does not transfer across the KMeans(8) scene split, consistent with — though not
identical to — the test-touch-log's "must say scene fitting" trigger (that trigger is written for
Δ_rand>0-but-LCO-CI-includes-0; here Δ_rand≈0 but LCO shows E1 significantly *behind* E0, a stronger
version of the same warning).

### Per-sub-class recall under LCO (panels flagged, all archs collapse hard vs k-fold)

| class | n | E0 | E1@64(T3) | E1@64(T1) | E1@96 | E2 | E3 |
|---|---|---|---|---|---|---|---|
| 0 WalkwayViol | 207 | 94.7% | 92.8% | 92.8% | 90.8% | 83.6% | 94.2% |
| 1 UnauthorInt | 107 | 81.3% | 83.2% | 82.2% | 81.3% | 86.9% | 86.9% |
| 2 OpenedPanel | 142 | 81.7% | 59.2% | 52.1% | 77.5% | 74.6% | 73.9% |
| 3 CarryOverload | 56 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 SafeWalkway | 75 | 17.3% | 18.7% | 20.0% | 14.7% | 26.7% | 36.0% |
| 5 AuthorInt | 38 | 84.2% | 78.9% | 81.6% | 84.2% | 89.5% | 76.3% |
| 6 ClosedPanel | 32 | 9.4% | 0.0% | 0.0% | 0.0% | 9.4% | 12.5% |
| 7 SafeCarrying | 30 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

ClosedPanel (safe, class 6) and SafeWalkway (safe, class 4) collapse for every architecture under LCO
(0-12.5% and 14.7-36.0% recall respectively, vs 87-94% and 69-75% under k-fold) — matches the gates
report's finding that ClosedPanel is the hardest class for S1/S2 too (S1 LCO: 0%, S2 LCO: 56.2%,
0.12 scene-cluster overlap with OpenedPanel). E1@64 is worst-in-class on ClosedPanel/OpenedPanel under
both coord variants (0.0%/52-59%) — the panel pair is where E1's narrower representation hurts most.
E3 has the best SafeWalkway recall (36.0%) and ties best-or-near-best on ClosedPanel (12.5%) despite
having no attention at all — consistent with its lead on the aggregate metric.

### Pre-registered decision rule (reports/test-touch-log.md G0 + late-17/9 addendum) — evaluated

Rule: credit E1 over E0 only if Δ(E1-E0) LCO paired-bootstrap CI excludes 0 **and is positive**, AND
(E1-S2) CI excludes 0. Secondary: (E1-S1-logreg) CI excludes 0, needed to additionally claim "beats
detector-only info on unseen cameras." All paired-bootstrap CIs below: percentile, n_boot=5000,
clip-resampled, on macro-accuracy (same estimator as `run_lco_eval.py::paired_bootstrap_diff` and as
S1/S2's own reported bal_acc) — chosen as primary over McNemar because McNemar tests raw-accuracy
discordance, which is majority-inflated here (unsafe=74.7% of the pool); the two occasionally disagree
in significance (not direction) for that reason, e.g. E1@96-S1: McNemar p=1.00 (n.s.) but bootstrap CI
[+1.3,+7.3] excludes 0 — flagging this explicitly rather than picking whichever test is convenient.

| E1 config | Δ(E1-E0) [95% CI] | criterion 1 | (E1-S2) [95% CI] | criterion 2 | (E1-S1) [95% CI] | secondary |
|---|---|---|---|---|---|---|
| E1@64 (T3) | -4.5pp [-7.3,-1.6] | **FAIL** (negative, wrong direction) | +9.5pp [+4.8,+14.4] | pass | +2.6pp [-0.4,+5.7] | n.s. |
| E1@64 (T1) | -5.0pp [-7.8,-2.0] | **FAIL** | +9.0pp [+4.2,+14.0] | pass | +2.1pp [-0.9,+5.2] | n.s. |
| E1@96 | -2.8pp [-5.2,-0.4] | **FAIL** (negative, wrong direction) | +11.2pp [+6.2,+16.2] | pass | +4.2pp [+1.3,+7.3] | pass |

**Verdict: E1 is NOT credited over E0 under the pre-registered rule, for either width, under either
coord variant.** Criterion 1 fails decisively — not "CI includes 0" (n.s.) but CI entirely negative
(E0 significantly beats E1). This is a clean refutal, not an ambiguous non-result: **do not search for
a framing that rescues E1.**

E3 (not covered by the pre-registered rule, which is E1-specific) is the empirical LCO leader:
statistically tied with E0 (Δ +2.8pp [-0.7,+6.2], CI includes 0 — E3 does not *significantly* beat E0
either, tied not superior), but significantly beats E1@64 (+7.3pp [+3.6,+10.9]), S1-logreg (+9.9pp
[+6.2,+13.6]) and S2 (+16.8pp [+12.0,+21.8]) under LCO. E0 also clears both gates (E0-S1 +7.0pp
[+3.8,+10.5], E0-S2 far larger). So both E0 and E3 demonstrate real camera-transferable signal beyond
detector-only (S1) and scene-only (S2) information; E1 does not exceed E0 by that same measure.

### Freeze recommendation (final — supersedes the withdrawn Tier 2 read above)

**Freeze on E3 for the 2026-09-18 architecture freeze / 2026-09-19 Jetson ONNX export.** Justification,
stated as a graph decision, not an accuracy claim:
- E3 is statistically tied with E0 under the pre-registered protocol (LCO-691 bal_acc 69.6% vs 66.7%,
  CI includes 0 — point estimate favours E3 but not significantly) while being **16.6x smaller**
  (18,044 vs 298,892 params). This is an accuracy-parity + Occam's-razor argument, **not a latency
  argument**: per the Jetson readiness audit (`plans/2026-09-16-binary-safe-unsafe-reframe/phase-05-jetson-measurement-session.md`,
  `plans/2026-09-06-thesis-2week-final/reports/audit-2026-09-14-edge-and-demo-readiness.md`), the
  measured P3 edge budget is 1825 ms/decision with the detector at 1745 ms (95.6% of total) — the
  relation head (any of E0/E1/E2/E3) is a small remainder of that budget. **No latency claim rests on
  head size here**: swapping E0's 298,892-param head for E3's 18,044-param one cannot plausibly move
  the 0.55 dec/s bottleneck, which is detector+appearance-extraction bound. E3 is picked for accuracy
  parity and graph simplicity, not for solving the edge-throughput problem — that requires the
  detector/appearance work already tracked separately (see the audit's options 1-5).
- E3 significantly beats E1@64 (the Tier-2-provisional pick, now withdrawn) under the pre-registered
  protocol (+7.3pp [+3.6,+10.9]), and significantly beats both baseline gates (S1, S2) — real
  transferable signal, not scene memorization.
- **E1 is explicitly refuted by the pre-registered rule** (both widths, both coord variants) — it is
  not a freeze candidate. E2 is tied with E0 on LCO (66.7%=66.7%) but offers no efficiency win
  (284,556 params, nearly E0-sized) — no reason to prefer it over E0 or E3.
- **Thesis framing**: a DeepSets-style pooled-MLP with no attention (E3) matching or beating a
  factorized-attention Transformer (E1) — and tying the full attention baseline (E0) — at n=687
  training clips is a legitimate negative result about data scale/architecture fit, not a failure to
  hide. State plainly: at this dataset size, permutation-invariant mean/max pooling over per-box MLP
  features captures the safety-relevant signal at least as well as intra-frame + temporal attention;
  the attention machinery's extra capacity does not transfer to unseen cameras and may be *fitting*
  rather than *generalizing* (see the k-fold-vs-LCO ranking inversion above).
- Caveat carried into the thesis: E3 vs E0 is a **tie**, not a proven win (CI includes 0) — freeze on
  E3 is justified on efficiency-under-parity grounds (no accuracy cost, large parameter saving),
  exactly mirroring the reasoning the Tier-2-only read wrongly applied to E1. The one material
  difference: E3's tie with E0 replicates on the pre-registered protocol; E1's did not.
- **Risk flagged, not resolved here**: E3's ONNX export path is less exercised than E0's — E0 is the
  original/legacy architecture (`model_relation_head.py::RelationTransformer`), already the subject of
  prior export/deployment work (`deployment/export/p3-relation-head-fp32.onnx` in the current tree
  predates this session and is presumably an E0-shaped graph); E3 (`model_binary_encoders.py::E3DeepSets`)
  is new this session and has only been exercised via PyTorch training/eval, never traced/exported to
  ONNX. `E3DeepSets` uses only `Linear`/masked-mean/masked-max/elementwise ops (no `nn.MultiheadAttention`,
  no `TransformerEncoder`) — structurally simpler and lower-risk for ONNX tracing than E0/E1/E2, but
  this is an expectation, not a verified export. **Whoever owns `deployment/scripts/export-p3-graphs.py`
  should trace-export E3 today and confirm the ONNX graph is valid before relying on this freeze** —
  out of this task's file ownership (`deployment/` is read-only for this lane), reporting rather than
  fixing.
- **Selection-independence caveat**: E3 was selected using pooled OOF over the full 691-clip set
  (566 train + 125 test pooled for both Tier 2 k-fold and Tier 3 LCO) — the 125-clip blind test was
  never scored in isolation and consumed 0 touches for this selection, per
  `reports/test-touch-log.md`'s pooled-protocol accounting precedent. But this means the architecture
  choice is **not fully independent of the blind test's clips** (they contributed to the pooled OOF
  training/selection signal, just never to a standalone blind score) — the blind score on the eventual
  frozen graph is a confirmatory second touch, not an independent held-out validation of the
  architecture-selection decision itself. State this plainly alongside any blind-test number quoted
  post-freeze.

## Unresolved / methodological notes

- "3 seeds x 1 repeat" is implemented as a within-fold seed-ensemble (softmax averaged over 3 seeds
  before argmax, matching the existing legacy 4-class k-fold script's precedent) rather than 3
  independently-saved OOF sets. The reported spread is therefore fold-to-fold (5 folds, NB-corrected
  SE), not seed-to-seed — consistent with `run_kfold_relation.py`'s pre-existing reporting convention
  for the 4-class path. Flagging this as a deliberate precedent-matched choice, not a deviation.
- k-fold internal fit/val split (`train_test_split(tr, test_size=0.15, stratify=yn[tr])`) is stratified
  on label only, not (label,view) — the design's (label,view) stratification requirement is stated for
  the blind protocol (P-A, `run_relation_transformer.py`, implemented in `main_binary`); the k-fold
  script's internal split follows the legacy script's existing precedent (label-only stratify).
- `run_kfold_relation.py` grew to 268 lines (`--lco` mode + shared `report_binary()` helper), over the
  200-line-per-file guideline — a deliberate, disclosed deviation: the alternative was a second new
  file outside the declared file-ownership list, which the coordinator's own direction ruled out
  ("do this within your owned files").
- LCO internal fit/val split (`train_test_split(tr, test_size=0.15, stratify=yn[tr])`, same as k-fold)
  is per-held-out-group, not (label,view)-stratified; same precedent-matched choice as k-fold above,
  not separately re-justified for LCO.
- The E1-vs-E0 coord-variant confound was checked for E1@64 only (not E1@96, E2, or E3) — judged
  sufficient because (a) the confound-check result (T1 vs T3 Δ~0.5pp) is far smaller than the E1-E0
  gap it was raised to explain, and (b) E2/E3 already use T1 by default so no confound exists for them
  vs E0; re-running E1@96 with T1 would only be needed if someone specifically wants to defend E1@96
  against the same objection.
- LCO group 0 (n=22) is the smallest, noisiest held-out fold across all 5 archs (27-91% swing) — no
  per-group CI was computed (only pooled-OOF CIs), so a single volatile small group's contribution to
  the pooled estimate was not separately bounded. Worth a variance check if the E0-vs-E3 tie needs to
  be defended more rigorously later (e.g. leave-one-group-out jackknife on the pooled bal_acc).
- S1/S2 gate comparisons use the gates agent's `sweep_results/binary_gates/*_logreg_*` files only (not
  `*_tree_*`) — logreg was treated as "the" S1/S2 baseline per the coordinator's own summary numbers
  (59.2-59.7% / 52.7-52.8%); the tree variant (worse, 47.4% LCO bal_acc) was not additionally compared.
- E3's ONNX exportability is asserted from its op inventory (Linear/mean/max only), not verified by an
  actual export — see the freeze-recommendation risk note above. This is the single largest residual
  risk against today's freeze deadline: if E3 fails to trace/export cleanly, the fallback should be E0
  (also tied with E3 under LCO, already presumed to have prior export precedent) rather than E1
  (refuted) or E2 (no efficiency benefit).
