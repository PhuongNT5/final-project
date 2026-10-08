# P3 pipeline code review — 2026-09-13

Scope: build_relation_tokens.py, model_relation_head.py, run_relation_transformer.py (+uncommitted diff),
run_kfold_relation.py, run_lco_eval.py, unsafenet/dedupe.py, tools/build_scene_grouping.py,
tools/apply_zone_prior.py, tools/make_zones_auto.py, kaggle-detector-retrain/{train-detector-gpu,build-zone-labels}.py,
augment_relation_tokens.py, eval_utils/{metrics,paired_tests}.py. Verified by reading code + running
targeted checks against on-disk artifacts (frame_manifest.csv, clip_view_map.json, zones/). Read-only,
no files edited. Codebase already carries an unusually deep internal audit trail (brief.md S1-S19,
dedupe.py, scene-appearance baseline swap) — findings below are additive, cross-checked against that
trail to avoid duplicating what's already tracked, and flagged explicitly where something IS already known.

## CRITICAL

**C1 — entity detector was fit on frames from 100% of the 452 P3-train clips and 0% of the 64 P3-test
clips → the blind 64-clip number is clean, but k-fold/LCO (which pool all 516) are not.**
Verified: `annotation_guide/frame_manifest.csv` (2262 rows, all `clip_path` under `data/train/`, none
under `data/test/`) → `set(4-class clips in manifest) == set(452 P3-train clips on disk)` exactly (452/452
overlap, 0/64 for P3-test). This manifest feeds `annotation_guide/dataset` (movable-entity detector,
`kaggle-detector-retrain/train-detector-gpu.py` baseline arm) AND `dataset_behaviour_*` — and per
`plans/2026-07-09-.../reports/mt3-yolo26-results.md` §1, the YOLO26 swap detector used for the headline
`yolo26app_`/`yoloapp_` tag ("Same dataset (2262 frames, 8 entities)") is trained on the *same* manifest.
`build_relation_tokens.py:238` (`det = YOLO(DET)`) then runs this exact detector over every clip, train
**and** test, to build tokens — including its ROI/appearance backbone for the `app`/`yoloapp_` tags
(`get_appearance(..., backbone='yolo')`, build_relation_tokens.py:45-49).
- Blind 64-clip headline (`run_relation_transformer.py`, task's exact command): clean on this axis — real
  strength, worth stating explicitly in the thesis as a positive control.
- `run_kfold_relation.py`/`run_lco_eval.py` (`load_pooled`, both files) concatenate train+test into 516
  clips then re-split by `StratifiedKFold` or by camera view. Any fold's held-out partition is therefore,
  in expectation, ~452/516 = 87.6% clips whose own footage frames were annotated and trained into the
  entity detector (train or val arm of its own 90/10 split — clip-level clean, i.e. no single clip's
  frames straddle detector-train/detector-val, verified). Evaluating relation-Transformer tokens built
  from a detector that has directly seen a held-out clip's own frames is leakage upstream of the
  Transformer itself — distinct from, and additive to, the already-documented camera-clustering
  circularity (run_lco_eval.py's own `[LIMITATION]` prints) and the S11 dedupe contamination. It inflates
  k-fold 80.8±4.0% and LCO 77.7% by an unknown, unquantified amount; direction is unambiguous (biases
  upward), magnitude is not measured anywhere in the repo (grepped brief.md/reports — no hit).
- Impact: both headline generalization claims (k-fold, LCO) rest partly on detector memorization of
  specific clip backgrounds/entities, not just camera-type generalization. This does not touch the blind
  64-clip 54/64, which remains the one clean same-distribution number.
- Fix options (cheapest→most rigorous): (a) explicitly caveat k-fold/LCO as upper bounds contaminated by
  detector-level clip memorization; (b) re-run k-fold with the entity detector's own train/val membership
  as an additional stratification/exclusion variable (drop folds' test clips that were in detector-train);
  (c) retrain the detector per-fold excluding that fold's clips (expensive, most correct); (d) at minimum
  quantify by comparing per-clip detector confidence/box quality between detector-train-origin clips vs the
  genuinely-unseen 64-test clips within the pooled eval — cheap, no retraining needed.

## HIGH

**H1 — headline k-fold/LCO numbers quoted in the task are the `--dedupe none` (contaminated) variant;
a cleaner number already exists but isn't the one reported.** `run_kfold_relation.py --tag yoloapp_` and
`run_lco_eval.py` default to `--dedupe none`; neither headline command in the task passes `--dedupe`. Per
brief.md S11 (already executed, not new), `--dedupe drop_all` gives k-fold 86.6±3.1% (vs none's
80.8±4.0%, NB p 0.0600→0.0014) and LCO 75.9%/79.3% macro (vs none's 77.7%/81.2%, margin +16.9→+17.7pp).
~10% of the 516 pooled clips (32 groups, mostly WalkwayViol↔UnauthorInt) carry a contradictory label under
`none`. Since the drop_all run already exists and is *more* favorable on k-fold significance, there's no
reason to report the contaminated number as primary — report drop_all (or both, clearly labelled).

**H2 — missing zone polygons for camera views 2 and 7 silently zero out the model's safety-relevant
geometry features for 56/516 (10.9%) clips; specifically cripples 2 of 8 LCO folds.** Verified:
`annotation_guide/zones/view_*.json` exists for views {0,1,3,4,5,6,na} only; `clip_view_map.json` shows
views 2 and 7 hold 32 and 24 clips respectively (56 total). `build_relation_tokens.py:100-101`
(`zones.get(v)`) returns `None` for those clips; `geometry()` (`build_relation_tokens.py:104-123`) then
takes the `if zones else 0.0` / `if zones and ...` branches and returns `[0,0,0,0]` for
iou_walkway/iou_restricted/foot_in_walkway/norm_dist for every person/unauthorized-person token in those
clips — **no warning, log line, or count printed anywhere in the script**, so this is invisible unless you
diff `zones.keys()` against `vmap.values()` by hand (as done here). Effect: (1) ~11% of both train and
pooled-eval data get 4 of ~9-73 features hard-zeroed regardless of actual walkway/restricted position; (2)
under LCO, when view 2 or view 7 is the held-out camera, **every** test clip in that fold has zero
geometry signal at inference, which the model was also never trained to rely on for that view (since
train-side clips of views 2/7 are equally zeroed) — likely a real contributor to the per-class LCO recall
volatility already noted in brief.md/mt3. **Already tracked** (not new): flagged in
`plans/2026-09-06-thesis-2week-final/reports/review-2026-09-13-thesis-alignment.md` item 11 ("draw the two
missing zone polygons") — confirmed here at the code level with the exact mechanism and blast radius (56
clips / 2 of 8 LCO folds), which the other review didn't quantify. Fix: draw the 2 missing polygons (cheap,
per that review) and re-extract tokens + re-run LCO before finalizing the LCO headline number.

**H3 — the three flagship numbers (54/64 blind, 80.8% k-fold, 77.7% LCO) come from three different
training recipes, not one model evaluated three ways.** `run_kfold_relation.py:77` and
`run_lco_eval.py:172-174` both call `R.train_one(..., use_cw=False, ...)` (class weights OFF); the
headline blind command has no `--no_class_weights`, so class weights are ON (default,
`run_relation_transformer.py:172`). Epochs also differ: headline `--epochs 120` vs k-fold/LCO default
`--epochs 60`. This is a real confound if the thesis frames k-fold/LCO as "the same model, generalization-
tested" — they are a same-architecture, differently-tuned sibling model. Either match the recipe or state
the difference explicitly wherever the three numbers are placed side by side.

## MEDIUM

**M1 — "Transformer temporal reasoning" is a weakly-supported claim as currently built; verify with an
ablation before asserting it.** `model_relation_head.py:29` (`self.pos`) is a single learned **absolute**
positional embedding over the flattened `1 + T*N_MAX` sequence; encoding is full joint self-attention (no
causal mask, no relative-time bias, no per-frame recurrence/RNN). Within a frame, the N_MAX=8 slots are
filled by **confidence-sorted order** (`build_relation_tokens.py:175`, `order = np.argsort(-conf)`), not
stable entity identity, so an entity's position in the sequence is not tracked across frames — only the
frame index `t` (via `t*N_MAX` in the flattened offset) is a stable temporal signal, and it is a coarse,
non-relative one. Circumstantial evidence elsewhere in the repo
(`plans/2026-09-06-.../research/r4-data-processing-and-diversity.md:58,65`: temporal reversal `x.flip(0)`
"proven not to break the model") suggests a sibling track's classifier is largely frame-order-invariant —
i.e., closer to a permutation-invariant "bag of frames" set-Transformer than sequential temporal reasoning.
That evidence is from a different pipeline (track A / pairwise-contrastive), not P3's relation-Transformer
directly — no shuffle-frame-order or zero-`self.pos` ablation for `model_relation_head.py` was found in the
reviewed files. Recommend running one (shuffle T at inference only; or ablate `self.pos`→0) before the
thesis asserts "temporal reasoning" as a mechanism, rather than "spatial relation + coarse frame-position
prior."

**M2 — validation-set reuse across 120 epochs × 5 seeds (600 snapshots) scored against one fixed ~81-clip
val split.** `run_relation_transformer.py:159` carves val once (`random_state=0`, fixed for every seed);
`train_one` selects best epoch by val_macro (`>` strict, ties keep earliest — reasonable), then `main()`
selects the "blind" seed by max val_macro across seeds again on the same val split
(`run_relation_transformer.py:200`). No test leakage (test is never touched by any of this selection), so
the reported 54/64 remains an honest single-model estimate — but val_macro itself is optimistically biased
by the amount of implicit search against a tiny fixed slice, and should not be quoted as an independent
generalization estimate anywhere in the thesis.

**M3 — per-frame aux loss (`--aux_w 0.5`, used in the headline command) assigns the clip-level label to
every sampled frame, including frames before/after the unsafe act.** `run_relation_transformer.py:65-72`:
`ybT = yb.view(Bn,1).expand(Bn,T)` broadcasts one label across all T=16 frames of a clip for the frame-head
cross-entropy. This is a documented design choice (weak/noisy supervision), not a bug, but it means the
"per-frame behaviour" auxiliary signal used in the champion recipe is intentionally coarse — worth stating
as a limitation, since it also underlies the S1b frame-gate's "kept_frac" interpretation
(`run_relation_transformer.py:229`, "reduced-duration proxy").

## LOW

**L1 — no logging when the N_MAX=8 per-frame detection cap is hit.** `build_relation_tokens.py:175`
(`order = np.argsort(-conf)[:N_MAX]`) silently drops lower-confidence detections beyond the top 8 in
crowded frames; no counter/print exists to quantify how often this occurs across the 516 clips.

**L2 — token-build reproducibility is not fully hardware-pinned.** `build_relation_tokens.py` lets
`--device` default to `mps` if available; detector inference on MPS vs CPU can, in principle, reorder
near-tied-confidence detections differently, changing which boxes fall inside the top-8 N_MAX cut and thus
the saved `tokens_*.pt` (downstream training is CPU-only and seeded correctly, model_relation_head.py side
is fine). Low practical impact (ties are rare) but worth a one-line note in the reproducibility section
since tokens are precomputed artifacts, not regenerated per training run.

**L3 — `augment_relation_tokens.py` (v2 person-entity-proximity tokens) is not on the headline path**
(`--tag yolo26app_` bypasses it entirely) — confirmed dead code relative to the reported numbers, no action
needed, just noting scope so it isn't mistaken for part of the champion recipe.

## Verified correct (no issues found)

- **Masking/padding** in `model_relation_head.py`: CLS is never masked (`kpm` col 0 forced `False`);
  padded keys are correctly excluded via `src_key_padding_mask`; `token_ln` is per-token so padding can't
  skew LayerNorm stats; frame-level masked-mean (`frame_emb`) and `frame_valid` correctly restrict the aux
  loss and frame-gate to frames with ≥1 real detection.
- **Softmax ensemble averaging** (`logits_sum += torch.softmax(...)`), **blind seed selection** (test never
  touched), **τ selection for the frame-gate** (val-only sweep over `TAUS`) — all correctly test-blind.
- **clip_id / dedupe alignment**: `unsafenet/dedupe.py` audit `name` field (`'0_tr109.mp4'`) matches
  `build_relation_tokens.py`'s `clip_ids` format exactly (`os.path.basename(path)`, extension included) —
  no silent mismatch.
- **Label mapping**: class id parsed as `int(base.split('_')[0])` for folders `0_safe_walkway_violation` …
  `3_carrying_overload_with_forklift`, matching `NAMES` order in `run_relation_transformer.py` 1:1.
- **eval_utils/metrics.py, paired_tests.py**: `macro_accuracy` = mean per-class recall (matches the
  thesis's own stated definition, correctly NOT macro-F1); `nadeau_bengio` applies the train/test-ratio
  variance correction; `two_proportion_z`, `wilson_ci`, `mcnemar_exact` all standard-correct. No bugs.
- **Fold-local fitting** in `run_lco_eval.py`'s scene-appearance baseline (`StandardScaler`/
  `LogisticRegression` fit on `wf_X[tr]` only, per held-out view) and in `run_kfold_relation.py`'s
  per-view-majority baseline (`vmaj` built from `tr` only, per fold) — both correctly leakage-free.
- KMeans camera-clustering (`clip_view_map.json`, `clip_scene_map.json`) is fit on the full 516-clip pool,
  but unsupervised over background/appearance descriptors, not labels — expected and not leakage (this is
  the "which physical camera is this" question, analogous to calibration metadata); the appearance-based
  `clip_view_map.json`'s circularity (same feature family as the classifier) is already self-flagged in
  `run_lco_eval.py`'s printed `[LIMITATION]` and mitigated via `--scene_map`'s background-only KMeans.

## Unresolved questions

1. C1's leak magnitude is unmeasured — is there budget for the cheap version (compare detector
   confidence/box-count distributions for detector-train-origin clips vs the genuinely-unseen 64, within
   the pooled k-fold/LCO sets) before the thesis defence?
2. Does the entity detector's own internal 90/10 train/val arm (within the 566-clip annotation pool)
   correlate with which clips land in which LCO camera-group — i.e., is the C1 leak evenly spread across
   folds or concentrated in a few (compounding effect)? Not checked here (out of time budget).
2b. H2's fix (draw 2 missing zone polygons) requires re-extracting tokens and re-running LCO/k-fold — is
   that scheduled before the numbers in report/chapter-07 are frozen?
3. Was M1's temporal-reversal-invariance evidence (from the sibling pairwise-contrastive track) ever
   reproduced directly on `model_relation_head.py`/P3 tokens? If not, recommend adding it to the remaining
   task list before writing the "Transformer temporal reasoning" claim in the thesis text.
