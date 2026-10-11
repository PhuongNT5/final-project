# Review: NP-125-D detection scorer (eval_utils/detection_metrics.py, run_binary_detection_scorecard.py, eval_utils/cluster_bootstrap.py)

Scope: 3 files, ~320 LOC. No numbers changed, no scoring rerun. Verified by reading code +
spot-checking pure functions against the frozen artifact `sweep_results/r2/temporal_convlstm_binary.pt`
(no inference, no write to `ket-qua-np125d-phat-hien.json`).

## Priority 1 — test-label leakage into threshold selection: NONE FOUND

All 4 operating points are chosen by `pick_thresholds(sv, yv, scene_fpr_val)`
(detection_metrics.py:42-51) from the 103-clip val slice only. `st`/`yt` (test) are read only
after `taus` is fixed (run_binary_detection_scorecard.py:44-47), with `>=` used consistently in
both grid construction and application. OP3's scene-only reference FPR is the HONEST
train-fit scene control evaluated on val (`scene_v = ctrl_v['scene_majority_train_fit']`,
line 41), not the oracle/test-label-derived control — correct per prereg §3. The oracle control
(`scene_majority_oracle`, computed from each split's own labels in
`run_convlstm_paper_baseline.scene_controls`) is computed for TEST too but only surfaced
descriptively in `controls_in_detection_terms` (line 76-77); it never feeds OP3's threshold.
Worth a one-line comment at that call site since the function's generic name doesn't make the
"descriptive only" intent obvious to a future reader.

## Findings, most severe first

### 1. [HIGH] within_scene_auc's bootstrap CI uses the WRONG unit, undisclosed
`eval_utils/detection_metrics.py:113-130` (`ci_within_scene_clip_resample`). The loop fixes all
8 scenes every draw and only resamples CLIPS WITHIN each scene (comment at line 117 says so
honestly) — it never varies which scenes appear. That is exactly the "wrong unit" the
pre-registration's own §6 risk calls out (n=8 groups ⇒ wide CI, likely contains 0/chance), but
unlike `recall_ci_CLIP_UNIT_wrong_unit` in run_binary_detection_scorecard.py:56, this field is not
flagged as the wrong unit, and no group-resampled alternative is reported for this statistic.

Empirically verified on the real artifact (seed 0, 5000 draws, same scene ids):
- as-implemented (within-scene clip resample): `[0.5106, 0.7680]` (width 0.257)
- true group-cluster bootstrap (resample WHICH of the 8 scenes appear, same style as
  `cluster_bootstrap.metric_cluster_bootstrap`): `[0.4886, 0.8860]` (width 0.397), noticeably
  closer to chance (0.5).

Failure scenario: a reader cites "the within-scene AUC's 95% CI excludes chance" off this field,
when the correctly-unit'd CI almost touches 0.5 — understating exactly the uncertainty the
pre-registration pre-committed to disclosing.

Fix: either replace the internal loop with a scene-level resample (pick 8 scene ids with
replacement, as `metric_cluster_bootstrap` already does for recall/FPR), or rename the field to
carry a `WRONG_UNIT` flag and add a true group CI alongside it, matching the
`recall_ci_CLIP_UNIT_wrong_unit` / `recall_ci_cluster_8_scene_groups` pairing already used
elsewhere in the same JSON.

### 2. [MEDIUM] counts() silently turns an undefined F1 into 0.0 (not nan) for an empty class
`eval_utils/detection_metrics.py:27`. `f1 = 2*prec*rec/(prec+rec) if prec+rec and not
np.isnan(prec*rec) else 0.0`. When the ground truth has ZERO positives (not just zero
*predicted* positives), `recall_unsafe` correctly comes out `nan`, but `f1_unsafe` comes out
`0.0`:
```
counts([0,0,0,0], [1,0,0,0]) -> recall_unsafe=nan, f1_unsafe=0.0   # should be nan
```
`bool(nan)` is `True` in Python, so the `prec+rec` truthiness check never catches this case; only
`np.isnan(prec*rec)` does, and by then the `else` branch has already been chosen, overwriting a
genuinely-undefined quantity with a confident-looking zero.

Does not change any number in the current run (every subset `counts()` is called on here — full
val, full test, every group-cluster-bootstrap draw — always has ≥1 unsafe clip per included
scene, so `tp+fn>0` everywhere; all 8 test scenes have `n_unsafe≥1`). But it is precisely the
"silently wrong number instead of an error" trap: the first reuse on a single-class scene/slice
(e.g. a future scene-stratified F1 breakdown) will print 0.0, not "undefined."

Fix: distinguish "prec and rec both legitimately 0" (→ 0.0, correct) from "prec or rec is nan"
(→ nan), e.g. `float('nan') if np.isnan(prec) or np.isnan(rec) else (2*prec*rec/(prec+rec) if
prec+rec else 0.0)`.

### 3. [LOW] fragile substring-membership test for the NP-64 slice mask
`run_binary_detection_scorecard.py:80`: `n.split('_')[0] in '0123'` uses Python's string
substring-containment operator against a literal, not membership in a set/tuple. The correct
pattern already exists two files over: `run_convlstm_paper_baseline.py:136` uses
`n.split('_')[0] in ('0','1','2','3')`. Verified the two forms agree on all 125 current test
names (0 mismatches, both give 64), so today's `recall_unsafe_at_OP0` for NP-64 is correct. But
the substring form has real failure modes it doesn't need: `'' in '0123'` is `True` (an
empty-prefix name would be wrongly included), and a two-digit prefix like `'10'` would be wrongly
EXCLUDED (`'10' in '0123'` is `False`) with no error either way. Switch to the tuple/set form
used elsewhere.

### 4. [MEDIUM] zero test coverage for all three files
`eval_utils/tests/test_cluster_bootstrap.py` only covers the pre-existing
`paired_metric_cluster_bootstrap`; the newly appended `metric_cluster_bootstrap`, every function
in `detection_metrics.py` (`counts`, `pick_thresholds`, `duplicate_floor`,
`per_unsafe_class_recall`, `within_scene_auc`, `scene_alarm_profile`), and
`run_binary_detection_scorecard.py` have no automated tests. This is the code computing the
headline numbers for a pre-registered thesis-defense measurement; development-rules.md and
primary-workflow.md both call for tester coverage before this is treated as reviewed-and-done.

### 5. [LOW] minor disclosed-but-unflagged design choices
- `pick_thresholds` (detection_metrics.py:46, 49): OP1/OP3 tie-breaks both resolve ties toward
  the SMALLEST threshold (most liberal). Deterministic and harmless, but undocumented; one
  comment line would save the next reader from re-deriving it.
- `run_binary_detection_scorecard.py:49-55` discards `n_valid_draws` (4th return value of
  `metric_cluster_bootstrap`) when building `recall_ci_cluster_8_scene_groups` /
  `false_alarm_ci_cluster_8_scene_groups`. Two of the 8 test scenes are single-class
  (scene 5 and 6 are 100% unsafe, `n_safe=0` — verified from the artifact), so a (today
  negligible, ~1e-5 per draw) tail of the 5000 group resamples could be nan-discarded for the FPR
  statistic. The module's own docstring promises this count "for disclosure"; the script drops
  it. Keep `r[3]` in the output.
- `scene_alarm_profile`'s `np.corrcoef(ar, tr)` (detection_metrics.py:148) emits a
  RuntimeWarning and returns nan silently on zero-variance input (e.g. an operating point with a
  uniform decision across all 8 scenes). Not triggered by OP0 on the real data; undefended.

## Spot-checks that came back correct (for completeness)
- `duplicate_floor` reproduces exactly the 4 pairs/names in prereg §5
  (`4_te4`, `5_te13`, `4_te7` straddling train→test; `0_te23`≡`4_te9` wholly in test), with
  0/4 actually tripped at τ=0.5.
- `within_scene_auc`'s point estimate (0.6426 on the real artifact) matches an independent
  re-derivation; tie handling (`diff==0` weighted 0.5) and the "scenes needing both classes"
  guard are correct; `n_mixed_scenes=6` correctly excludes the 2 single-class scenes.
- `cluster_bootstrap.metric_cluster_bootstrap` (cluster_bootstrap.py:52-76) correctly resamples
  the 8 scene GROUPS (not clips) with replacement and discards nan draws — this is the one place
  group-unit CIs are done right, which is why finding #1 reads as an inconsistency rather than a
  systemic problem.
- Class/column convention: `POS=1`="unsafe" matches `(ytr < 4).astype(int)` in
  `unsafenet/convlstm.py:132`; `Pt[:, 1]` is therefore P(unsafe), confirmed by cross-reading the
  training code, not assumed.
- Trains nothing / infers nothing: `main()` only calls `torch.load`; no model instantiation,
  no forward pass. `py_compile` + bare `import` of all three files succeed cleanly.
- File sizes: 151 / 93 / 76 lines, all under the 200-line cap.

## Project-rule notes
- Filenames use snake_case, not kebab-case (`detection_metrics.py`, not `detection-metrics.py`).
  This is consistent with 100% of the existing Python codebase and with Python's own import
  syntax (hyphens aren't legal in `import` statements) — not a new deviation, no action needed.
- No mocks/fake data; real artifact reads only. Confirmed compliant.

## Unresolved questions
- Is the within-scene AUC headline number expected to carry a group-unit CI at all, or is the
  clip-conditional-on-scene bootstrap an intentional, separate diagnostic? If intentional, it
  still needs the `WRONG_UNIT` label per the project's own convention (finding #1).
- Should `metric_cluster_bootstrap`'s `n_valid_draws` be surfaced in the JSON output for the two
  single-class test scenes' sake, or is the current ~1e-5 discard probability considered
  negligible enough to not disclose?
