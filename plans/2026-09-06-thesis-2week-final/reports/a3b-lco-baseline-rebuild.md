# A3b — LCO baseline rebuild (S2 fix)

Task: rebuild the leave-camera-out (LCO) baseline in `run_lco_eval.py` so axis-4 (cross-camera
generalisation) rests on a defensible comparison, not a constant predictor. File owned exclusively:
`run_lco_eval.py`. No other root `run_*.py` touched.

## 0. Canonical run (reproducible, single declared config)

```
.venv/bin/python -u run_lco_eval.py --seeds 5 --epochs 60 --tag yolo26L1_ --aux_w 0.5 \
    --scene_map sweep_results/zone_prior/clip_scene_map.json
```

- Token set: `tokens_yolo26L1_{train,test}.pt` — the only LCO-relevant tag whose cache files still
  exist on disk (`tokens_zones_l1_*.pt`, cited in `_zones_eval.log` at 79.7%, was deleted/renamed at
  some point — confirmed absent via `find`; not reproducible, dropped from consideration).
- Feature layout: 72-dim (`class,w,h,conf,iou_walkway,iou_restricted,foot_in_walkway,norm_dist` + 64-d
  crop appearance, `xc,yc` already stripped = the L1 ablation, kept per r4/L1 finding).
- **seeds = 5** (0..4, ensemble-averaged softmax per fold), **epochs = 60**, `aux_w=0.5` (S1a per-frame
  aux loss), `use_cw=False` inside the LCO loop (matches pre-existing script behaviour — class weights
  disabled for the per-fold retrain).
- `--scene_map sweep_results/zone_prior/clip_scene_map.json`: overrides `views` with the
  classifier-independent KMeans(8) background grouping (see §3). All 516/516 clips mapped.
- Env: `torch==2.11.0`, `scikit-learn==1.8.0`, CPU-only training (`DEVICE='cpu'` hardcoded in
  `run_relation_transformer.py`, MPS Adam bug on macOS 14).
- Raw stdout saved: `sweep_results/relation_tokens/lco_eval_canonical_a3b.log` (this is the canonical,
  citable log — do not cite `_zones_eval.log` going forward, see §4).
- Wall clock: **~65 min** (started 01:46:17, finished 02:51:29), sharing the machine with 1-2 other
  concurrent Lane-A training jobs (`run_kfold_relation.py --repeats 10`, and briefly another
  `run_lco_eval.py --tag yolo26motL1_`) — this violates the plan's "Lane A: one heavy job at a time"
  rule but those jobs were not mine to kill. An earlier attempt at the same 5-seed config, under worse
  contention (3 concurrent heavy jobs), used >90 CPU-minutes without finishing even fold 1/8 and was
  killed and restarted rather than quoted. Expect **highly variable wall-clock cost (1-3+ hours)** for
  this config depending on machine load; budget accordingly, do not assume 65 min is guaranteed.

## 1. Three baselines, measured (pooled 516 clips, 8 camera-groups, LCO)

| Baseline | top-1 | macro | Note |
|---|---|---|---|
| **[1] prior** (trivial floor) | 85/516 = 16.5% | **10.1%** | training global-majority class, recomputed per fold (majority CLASS differs by fold — view1's removal makes class2 the majority elsewhere, etc. — so this is not one single global constant across all 516 clips, it's 8 fold-local constants pooled). Arithmetic floor for a SINGLE global constant over 4 classes is exactly 25.0% macro (100% recall on one class, 0% on the rest) — the measured 10.1% is *below* that because different folds pick different majority classes, so pooled per-class recall is worse than the single-constant case. Either way: **information-free**, must not be read as a scene model. |
| **[2] scene-appearance** (honest baseline) | 274/516 = 53.1% | **64.4%** | Logistic regression (`class_weight='balanced'`, `StandardScaler`) on whole-frame MobileNetV3 features (`tokens_wf_{train,test}.pt`, background-encoding, crop-free, T=16 mean-pooled to 64-d), fit on training cameras only, applied to the held-out camera. This is genuinely transferable — no access to the held-out camera at fit time. |
| **[3] per-view-majority** (k-fold-matched) | — | — | **Undefined under LCO** by construction: the held-out view has zero training clips, so "majority class within this view, from training data" cannot be computed. It degenerates to [1]. This is the exact, stated reason the k-fold script's 77.1% (real, per-view-majority-within-fold, every camera seen in training) and any LCO "scene-only" number are **different estimators** and were never comparable — see brief.md S2. |
| **relation-Transformer** (5-seed ensemble) | 401/516 = 77.7% | **81.2%** | — |

Per-class recall (transformer vs scene-appearance):

| Class | transformer | scene-appearance |
|---|---|---|
| WalkwayViol | 70.5% | 17.6% |
| UnauthorInt | 65.7% | 51.9% |
| OpenedPanel | 88.7% | 88.0% |
| CarryOverload | 100.0% | 100.0% |

Confusion (rows=true, cols=pred, order WalkwayViol/UnauthorInt/OpenedPanel/CarryOverload):

transformer:
```
[[148  44  18   0]
 [ 26  71  11   0]
 [  3  13 126   0]
 [  0   0   0  56]]
```
scene-appearance:
```
[[ 37 161  12   0]
 [ 34  56  18   0]
 [  6  11 125   0]
 [  0   0   0  56]]
```

CarryOverload (n=56, held out only in view-2/view-5 folds) is trivially 100% for BOTH — both cameras
holding that class are visually distinctive enough that background alone solves it; not evidence the
transformer is doing anything class-specific there. OpenedPanel is also close (88.7 vs 88.0) —
appearance alone nearly matches the model on this class. The transformer's real advantage is
concentrated in WalkwayViol (+52.9pp) and UnauthorInt (+13.8pp) — plausibly because zone-relative
geometry (iou_walkway/foot_in_walkway features) and person-relation tokens carry information the
whole-frame background embedding cannot.

## 2. The honest margin (claim axis 4)

**relation-Transformer macro − scene-appearance macro = +16.9 pp** [95% bootstrap CI **+12.5, +21.2 pp**,
percentile method, n_boot=5000, resampled over the 516 pooled clips, **model = 5-seed ensemble**].

This is the number for the thesis. The CI excludes zero by a wide margin, so this is a real, if modest,
result: under leave-camera-out, the relation-Transformer retains cross-camera behavioural signal beyond
what a simple background-appearance classifier can transfer. It is **much smaller than the withdrawn
+60.3 pp** (which compared the model to the constant baseline [1], not to a transferable classifier) —
report it as measured, not adjusted upward.

**Also newly measured, and itself reportable:** baseline [2] alone reaches 64.4% macro under LCO — far
above the 25% floor. Whole-frame background pixels carry substantial cross-camera-transferable
behaviour-correlated signal by themselves. This reframes, but does not eliminate, the r4 "camera-identity
→ label shortcut" concern: even a simple background classifier picks up a large chunk of the signal, so
part of what the transformer captures may still be background-adjacent rather than purely behavioural —
the +16.9pp margin is the part that survives after subtracting that channel, and it is the correct number
to defend, not the raw 81.2% model macro against the 10.1%/25% floor.

## 3. Proxy-grouping limitation (investigated, per task caveat)

`--scene_map` points at `sweep_results/zone_prior/clip_scene_map.json`, built by
`tools/build_scene_grouping.py`: KMeans(8) on each clip's temporal-median grayscale background
(32×32 downscale of the pre-extracted 32-frame cache, per-clip contrast-normalised). This is
**classifier-independent** — unlike the older `clip_view_map.json` (KMeans on the SAME appearance
features the classifier consumes, which r4 flags as circular) — so it is the better of the two available
groupings and is what the canonical run uses.

**But it is still a proxy, not verified camera metadata.** No camera IDs exist anywhere in the repo —
filenames are `{class}_tr{N}.mp4` / `{class}_te{N}.mp4`, confirmed via
`tools/build_scene_grouping.py` docstring ("No real camera IDs exist in filenames") and independently by
grep across the dataset. The 8 KMeans groups are an unsupervised clustering of background pixels; whether
they correspond 1:1 to 8 physical cameras, to fewer cameras at different pan/zoom settings, or to lighting
conditions on a shared camera, is **unconfirmed** (same open question r4 raised). **The strength of the
LCO result is bounded by how well this proxy tracks real camera identity** — if the true camera count is
smaller than 8 (e.g. the "~2 physical cameras" figure floated in earlier planning docs, itself
unconfirmed per r4), some "held-out cameras" could actually be held-out viewpoints of an already-seen
physical camera, which would inflate the apparent generalisation. State this plainly as a limitation;
do not claim genuine unseen-camera generalisation without this caveat attached.

## 4. Additional finding (per coordinator instruction): pre-existing LCO numbers also violate the ≥5-seed rule

`sweep_results/relation_tokens/_zones_eval.log` line 3 (`zones_l1_` / "ZONES", **79.7%** macro) and line
28-ish (`yolo26L1_` / "CHAMPION", **82.0%** macro) both print `seeds 3 epochs 60`. The already-withdrawn
78.6% figure (per brief.md, not found in any current log) is presumably from the same family of runs.
**All were seeds=3, violating brief.md §4h rule 5 ("≥5 seeds for any quoted number")**, independent of the
baseline-definition problem this task was opened to fix. This is a second, independent reason those
numbers were never fit to headline: not just baseline-vs-constant, but also underpowered on seed count.
The canonical run in this report (seeds=5) supersedes all three prior LCO figures; `_zones_eval.log`
should not be cited going forward.

Note the canonical seeds=5 model macro (81.2%) is close to but not identical to the old seeds=3
"CHAMPION" figure (82.0%) for the same tag/scene_map — the ~0.8pp gap is normal seed-to-seed variance,
not a regression; it is, if anything, a more trustworthy estimate for having 5 seeds instead of 3.

## 5. What changed in `run_lco_eval.py`

- Three baselines computed per fold and pooled, replacing the single constant-predictor "scene-only"
  baseline (which is kept as baseline [1], relabelled and given its arithmetic-floor context).
- New `load_wf_pooled()`: loads `tokens_wf_{train,test}.pt` (whole-frame MobileNetV3 features, built by
  `build_wholeframe_features.py`, already cached — no re-extraction needed), asserts its clip order
  matches the main token set via an exact label-tensor equality check (this file has no `clip_ids`
  field, so order-alignment is asserted, not merely assumed) before trusting it for baseline 2.
- New `paired_bootstrap_diff()`: 5000-resample percentile bootstrap CI on
  `macro(model) - macro(scene-appearance)`, reported alongside the point estimate.
- Explicit print of the k-fold/LCO asymmetry (baseline 3 undefined under LCO) and the proxy-grouping
  limitation on every invocation with `--scene_map`.
- A `*** NEGATIVE RESULT ***` guard prints automatically if baseline 2 ever matches or beats the model —
  did not trigger here (margin is positive and CI excludes zero), but the check is load-bearing for any
  future re-run with a different tag/config.

## 6. Not done here (explicitly out of scope / other owners)

- `run_kfold_relation.py:83` (prints the withdrawn 16.9% contrast) needs the same correction — flagged
  per brief.md, not edited (owned by another phase, task explicitly restricted me to `run_lco_eval.py`
  only).
- No `test-touch-log.md` entry created — LCO pools all 516 clips (train+test), doesn't consume the
  single declared 64-clip test touch; leaving that log to whichever lane owns it.

## Unresolved questions

1. Does the physical-camera count behind the 8 KMeans background-groups match reality, or could some
   "held-out cameras" be viewpoints of an already-seen camera (§3)? No ground truth exists in-repo to
   settle this — would need external annotation/metadata not currently available.
2. Should `build_wholeframe_features.py`'s `VIEW_MAP` (currently the old circular `clip_view_map.json`)
   be updated to default to `clip_scene_map.json`? Not changed here since that file is outside this
   task's exclusive ownership and the `--scene_map` override in `run_lco_eval.py` already fixes the
   grouping used at eval time regardless of what's baked into the `.pt` file.
3. Wall-clock cost for this canonical config is volatile under Lane-A contention (65 min this run, >90
   CPU-min without finishing fold 1 on a worse-contended prior attempt). If A4's headline freeze needs
   a tighter timing guarantee, recommend re-running once other Lane-A jobs are confirmed idle.
4. The scene-appearance baseline (64.4% macro) is higher than expected for "just background pixels" —
   worth a sentence in Ch3/Ch4 flagging that background alone is a stronger signal than the earlier
   framing assumed, which is exactly why the +16.9pp (not +71.1pp = 81.2-10.1) is the number to defend.
