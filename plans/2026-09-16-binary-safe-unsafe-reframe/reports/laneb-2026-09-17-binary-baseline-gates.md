# Binary Safe/Unsafe Tier-1 GATES: S1 detector-rule, S2 scene-appearance (2026-09-17)

Pre-encoder gates per phase-06 §Implementation Steps 2. Zero test touches: k-fold-691/LCO-691 pool
train+test (691) per existing 4-class precedent (`run_kfold_relation.py`/`run_lco_eval.py` already
pool 516) and phase-06's explicit accounting ("pooled k-fold/LCO over the 691 pool = 0 touches") —
the 125-clip blind test was never scored in isolation or used for any selection decision.

## Timing (S2 rebuild — announced per instructions)

`build_wholeframe_features.py --classes 0,1,2,3,4,5,6,7 --view_map clip_view_map_691.json
--out_tag bin8_` ran as PID 11488, ~35 min wall (22:23→22:58), CPU-heavy (MobileNetV3 inference,
691 clips × 16 frames). Finished cleanly: `tokens_wf_bin8_train.pt` (566,16,64),
`tokens_wf_bin8_test.pt` (125,16,64). Machine free for the encoder-screening job from 22:58 on.

## Files changed

- `build_wholeframe_features.py` (96 lines) — added `--classes/--view_map/--out_tag` argparse CLI
  (backward-compat default reproduces the original 4-class `tokens_wf_{split}.pt`), added
  `clip_ids` to saved dict. No existing file overwritten (new `tokens_wf_bin8_*.pt`).
- `binary_gate_lib.py` (NEW, 140 lines) — data loading (`load_bin8`, `load_wf`, `load_pool`), S1
  descriptor, `kfold_oof`/`lco_oof`. Self-contained: only `eval_utils`, `unsafenet.dedupe`,
  `sklearn`, `torch`, `numpy` — no import of `model_relation_head.py` /`run_relation_transformer.py`
  /`run_kfold_relation.py` (owned by the concurrent encoder agent). This was a deliberate scope
  choice: rather than editing `run_lco_eval.py` (which imports `run_relation_transformer.py` at
  module level and would create a read/write race with that agent), the gate got its own runner.
- `run_binary_baseline_gates.py` (NEW, 106 lines) — CLI/orchestration, `evaluate_suite` calls,
  McNemar + NB-corrected significance.
- `tests/test_binary_baseline_gates.py` (NEW, 129 lines) — 6 synthetic/structural tests.
- `tests/test_binary_baseline_gates_scene_map.py` (NEW, 63 lines) — 3 real-data integration tests.
- No commits made.

## Test output (actual, both files, `.venv/bin/python`)

```
$ .venv/bin/python tests/test_binary_baseline_gates.py
  PASS  test_kfold_oof_never_fits_on_a_held_out_row
  PASS  test_lco_oof_degenerate_single_class_training_fold_falls_back_to_train_prior_not_a_crash
  PASS  test_lco_oof_never_fits_on_the_held_out_scene_cluster
  PASS  test_s1_descriptor_empty_clip_is_all_zero
  PASS  test_s1_descriptor_picks_max_confidence_per_class_over_masked_valid_boxes
  PASS  test_to_binary_maps_source_classes_correctly
ALL PASS

$ .venv/bin/python tests/test_binary_baseline_gates_scene_map.py
  PASS  test_every_pooled_bin8_clip_has_a_real_scene_group
  PASS  test_group_5_is_the_documented_all_unsafe_degenerate_cluster
  PASS  test_scene_groups_are_the_documented_k8_clusters
ALL PASS
```
`test_kfold_oof_never_fits_on_a_held_out_row` / `test_lco_oof_never_fits_on_the_held_out_scene_cluster`
are the load-bearing leakage proofs (task 7): spy-based, unique per-row IDs, assert train/held-out
ID sets disjoint across every one of 5×3=15 kfold calls and all 4 LCO calls, plus every row held out
exactly once in LCO. `test_group_5_...` additionally confirms on real data that scene-cluster 5 (101
clips) is 100% unsafe — the LCO fold most likely to hit the degenerate single-class-training-pool
fallback, which the synthetic test in the same file proves does not crash.

## Ceilings (from `laneb-2026-09-16-binary-dedupe-and-ceiling.md`)

| policy | hard ceiling | practical ceiling |
|---|---|---|
| `none` | 124/125 = 99.2% | 121/125 = 96.8% |
| `drop_conflict` (headline, train only; test unchanged) | 124/125 = 99.2% | 124/125 = 99.2% |

## S1 (detector-rule) — LogReg / depth≤2 tree, fit train-folds-only

| protocol | dedupe | model | bal_acc | macro-F1 (CI95) | unsafe R/P | safe R | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| k-fold-691 | none | logreg | 60.5% | 61.5% [56.9,66.0] | 97.5/79.0 | 23.4% | 0.714 | 0.710 |
| k-fold-691 | none | tree | 58.6% | 58.5% [54.2,63.3] | 100.0/78.1 | 17.1% | 0.690 | 0.707 |
| k-fold-691 | drop_conflict | logreg | 60.3% | 61.4% [56.7,65.8] | 97.3/78.8 | 23.4% | 0.713 | 0.706 |
| k-fold-691 | drop_conflict | tree | 58.6% | 58.4% [54.1,63.2] | 100.0/77.9 | 17.1% | 0.700 | 0.716 |
| LCO-691 | none | logreg | 59.2% | 60.0% [55.8,64.3] | 90.9/78.7 | 27.4% | 0.584 | 0.461 |
| LCO-691 | none | tree | 47.4% | 47.4% [44.0,51.5] | 70.2/73.3 | 24.6% | 0.487 | 0.391 |
| LCO-691 | drop_conflict | logreg | 59.7% | 60.6% [56.4,65.0] | 90.8/78.8 | 28.6% | 0.594 | 0.479 |
| LCO-691 | drop_conflict | tree | 47.4% | 47.5% [43.9,51.2] | 70.3/73.2 | 24.6% | 0.468 | 0.406 |

S1 per-sub-class recall is a fixed pattern across all 8 runs: near-perfect on 3 of 4 unsafe classes
(WalkwayViol/UnauthorInt/OpenedPanel/CarryOverload all ≥73%, mostly ≥90%) and near-zero on 3 of 4
safe classes (SafeWalkway ~4-5%, AuthorInt 0-42%, **ClosedPanel 0-3.1%**) — S1 has essentially
learned "predict Unsafe" with detector confidence as a weak modulator, not real discrimination.

## S2 (scene-appearance) — whole-frame T=16 mean-pooled, LogReg `class_weight='balanced'`

| protocol | dedupe | bal_acc | macro-F1 (CI95) | unsafe R/P | safe R | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| k-fold-691 | none | **74.6%** | **70.4%** [66.8,74.0] | 73.8/89.9 | 75.4% | 0.784 | 0.830 |
| k-fold-691 | drop_conflict | **74.1%** | **70.0%** [66.4,73.6] | 73.4/89.5 | 74.9% | 0.794 | 0.836 |
| LCO-691 | none | **52.8%** | **46.6%** [42.9,50.4] | 43.2/77.2 | 62.3% | 0.543 | 0.530 |
| LCO-691 | drop_conflict | **52.7%** | **46.6%** [43.0,50.3] | 43.2/77.0 | 62.3% | 0.545 | 0.533 |

S0 majority (predict Unsafe always): 74.5-74.7% raw accuracy, 50% balanced accuracy by construction.

## Significance (own-vs-own McNemar exact, S2-logreg vs S1-logreg on same pooled OOF clips)

| protocol | dedupe | b (S2 wrong,S1 right) | c (S2 right,S1 wrong) | p | diff |
|---|---|---|---|---|---|
| k-fold-691 | none | 101 | 132 | 0.0491 | +4.49pp (S2 better) |
| k-fold-691 | drop_conflict | 102 | 134 | 0.0434 | +4.66pp (S2 better) |
| LCO-691 | none | 81 | 266 | <0.0001 | **-26.77pp (S2 far worse)** |
| LCO-691 | drop_conflict | 79 | 264 | <0.0001 | **-26.93pp (S2 far worse)** |
NB-corrected fold-mean test (k-fold only, 25 fold-diffs): none t=-2.05,p=0.0515; drop_conflict
t=-2.28,p=0.0319 — consistent direction/significance with McNemar.

## Panel pair (class 2 OpenedPanel vs class 6 ClosedPanel, 0.12 co-cluster overlap, L1)

| system | protocol | OpenedPanel recall | ClosedPanel recall |
|---|---|---|---|
| S1 logreg | k-fold | 98.6-99.3% | **3.1%** |
| S1 logreg | LCO | 73.9-74.6% | **0.0%** |
| S2 logreg | k-fold | 54.2-55.6% | 81.2-84.4% |
| S2 logreg | LCO | 19.0% | 56.2% |

## Gate verdict

1. **S1 is nowhere near ceiling.** bal_acc 47-61% vs 96.8-99.2% practical ceiling. No replan
   forced — the detector-rule alone does not saturate the binary task, real headroom exists.
2. **S2 (background alone) solves a large fraction of the task under k-fold-691, and it is
   scene memorization, not signal.** Under k-fold (train/test scenes overlap), S2 reaches
   74-75% balanced accuracy / ~70% macro-F1, 24pp above chance and clearly above S1 (McNemar
   p<0.05, both dedupe policies). Under LCO-691 (camera held out), S2 collapses to ~53% balanced
   accuracy / ~47% macro-F1 — statistically indistinguishable from chance (ROC-AUC 0.53) and
   McNemar-significantly *worse* than S1 (p<0.0001, -27pp). This is the central finding: nearly
   all of S2's k-fold performance is background/camera memorization that does not generalize to
   an unseen scene. It validates LCO-691 as the protocol that must gate any encoder claim of
   real behavior understanding — a candidate encoder beating S2's k-fold number tells you little;
   beating S2's LCO number (~53% bal_acc / ~47% macro-F1) is the real bar, and beating S1's LCO
   number (~59-60% bal_acc) on top of that is needed to show detector-only info isn't enough.
3. **Panel pair confirms L1 and sharpens it.** ClosedPanel (safe) is the single hardest sub-class
   for both baselines everywhere (S1: 0-3.1% recall; S2 drops it from 81-84% under k-fold to 56%
   under LCO). OpenedPanel (unsafe) similarly halves for S2 under LCO (55%→19%). With only 0.12
   scene co-cluster overlap between the two, any encoder's panel-pair recall under LCO is the
   most diagnostic single number for whether it reasons about panel state vs. background.

## Follow-up (same day): per-clip OOF dump + bal_acc CI95

Added (own files only, `binary_gate_lib.py` +16 lines / 160 total, `run_binary_baseline_gates.py`
+7 lines / 113 total, both still under the 200-line cap):
- `binary_gate_lib.save_oof(model, protocol, policy_tag, clip_ids, y_true, prob, pred, fold_id)` —
  writes `sweep_results/binary_gates/{model}_{protocol}_{policy}_oof.json` (clip_id, y_true,
  prob_unsafe, y_pred at the same argmax/0.5 threshold used for the reported metrics, fold_id).
  `kfold_oof` now also returns `fold_id` (repeat-0 fold index per row); LCO's fold_id is the scene
  group itself (`pool['group']`).
- `run_binary_baseline_gates.py` calls `save_oof` after every system×protocol×policy run (12 files:
  {s1_logreg, s1_tree, s2_logreg} × {kfold-691, lco-691} × {none, headline-drop_conflict}) and now
  prints `balanced_acc=...% CI95=[lo,hi]` — sourced from `res['macro_accuracy']['ci95']`, since
  `unsafenet/metrics.py` defines `balanced_accuracy = macro_accuracy` (literal alias, same
  bootstrap-BCa CI applies to both).
- New test `test_saved_oof_reproduces_balanced_accuracy` (tests/test_binary_baseline_gates.py):
  runs real (non-spy) `kfold_oof`, computes bal_acc from the in-memory prob/pred, saves via
  `save_oof` to a scratch dir, reloads the JSON, recomputes bal_acc from the saved rows, asserts
  exact match, plus checks `y_pred == (prob_unsafe > 0.5)` for every row. 7/7 tests PASS.

Re-ran `run_binary_baseline_gates.py --k 5 --repeats 5` in the foreground (~9.5s total, all sklearn,
zero test touches — same pooled 691-clip data as before). Diffed the new log against the prior run's
log (ignoring the new "OOF saved" lines and the new bal_acc CI95 field): **byte-identical**, all
numbers in the tables above reproduce exactly. Balanced-accuracy CI95 (bootstrap BCa, 5000 resamples):

| protocol | dedupe | s1_logreg | s1_tree | s2_logreg |
|---|---|---|---|---|
| k-fold-691 | none | 60.5% [57.3,63.8] | 58.6% [56.1,61.7] | 74.6% [70.8,78.3] |
| k-fold-691 | drop_conflict | 60.3% [57.2,63.7] | 58.6% [56.0,61.6] | 74.1% [70.3,77.7] |
| LCO-691 | none | 59.2% [55.8,62.9] | 47.4% [43.8,51.6] | 52.8% [48.5,56.9] |
| LCO-691 | drop_conflict | 59.7% [56.3,63.5] | 47.4% [43.7,51.2] | 52.7% [48.6,57.0] |

Note: S1-logreg's LCO CI [55.8,62.9]/[56.3,63.5] does not overlap S2-logreg's LCO CI
[48.5,56.9]/[48.6,57.0] at the lower/upper boundary only marginally — consistent with the
McNemar significance (p<0.0001) already reported: S1 reliably beats S2 under LCO.

12 OOF files written under `sweep_results/binary_gates/` (new dir, not overwriting anything),
e.g. `s1_logreg_kfold-691_none_oof.json` — 691 rows each, verified `n` matches table `n` for every
file. These give the encoder work an exact-match clip set for own-vs-own McNemar against S1/S2
without re-fitting the baselines.

## Unresolved questions

- Full `evaluate_suite` per-metric CI95 (PR-AUC/ROC-AUC/per-class recall) beyond macro-F1 and
  balanced-accuracy still not printed — reproducible in seconds from the saved OOF + `eval_utils`
  if needed for the thesis chapter.
- Not verified: whether any unsafe-clip geometry columns are identically zero (phase-06 step 3
  mention) — judged out of scope for a detector-confidence/whole-frame gate, no geometry columns
  are touched by S1/S2.
- `tests/test_binary_baseline_gates*.py` (2 files, 10 tests total) — confirm this satisfies task 7's
  glob; no second gate-specific test file beyond the scene-map integration file was written.
