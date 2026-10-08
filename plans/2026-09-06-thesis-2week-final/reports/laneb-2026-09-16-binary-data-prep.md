# Lane B — Binary Safe/Unsafe Data Prep (2026-09-16)

Scope: extend scene grouping + zone/view assignment + relation-token build to all 691 clips
(566 train / 125 test; unsafe=classes 0-3, safe=classes 4-7) for the binary reframe. No training,
no test-split touches beyond building tokens (which is a one-time artifact, not an eval).

## Status: DONE — all 5 steps complete.

## Step 1 — Scene grouping extended to 691 clips

`tools/build_scene_grouping.py`: added `--classes {unsafe,all,<comma list>}` (default `unsafe` =
original 4-class behaviour, verified byte-identical output vs the existing 516-clip canonical map).
`--classes all` clusters all 691 clips (same background descriptor, `random_state=0`) into a NEW
file `sweep_results/zone_prior/clip_scene_map_691.json` — `clip_scene_map.json` (516) untouched.

k=8 (canonical) cluster sizes: `{0:22, 1:189, 2:58, 3:170, 4:123, 5:101, 6:14, 7:14}`.
k=10 sensitivity run saved to `clip_scene_map_691_k10.json` — sizes
`{0:35,1:189,2:15,3:170,4:56,5:22,6:94,7:15,8:29,9:66}`: k=10 mostly splits clusters 4 and 6
into finer pieces, clusters 1/3/5 (the largest, most homogeneous ones) are stable across k=8→10.
k=8 kept canonical (no clear reason to move; k=10 doesn't merge/resolve the weak clusters below,
it just subdivides).

ARI(new 691-map vs old appearance-KMeans view) over the 516 shared unsafe clips = 0.720 (k=8) /
0.649 (k=10), close to the original 4-class-only run's 0.690 — adding safe clips to the descriptor
pool doesn't materially change the unsafe-clip grouping.

## Step 2 — Zone/view assignment for safe clips

### Cluster composition (k=8, 691-clip map)

| cluster | total | unsafe | safe | unsafe class mix | unsafe known-view dist | dominant view (frac) | safe class mix |
|---|---|---|---|---|---|---|---|
| 0 | 22  | 18  | 4  | {1:18}            | {4:18}              | **4 (1.00)** | {5:4} |
| 1 | 189 | 154 | 35 | {0:18,1:11,2:125} | {0:58,3:61,6:35}    | 3 (0.40) — near 3-way tie | {4:20,5:15} |
| 2 | 58  | 36  | 22 | {3:36}            | {2:28,7:8}          | 2 (0.78) | {7:22} |
| 3 | 170 | 139 | 31 | {0:125,1:14}      | {1:137,5:2}         | **1 (0.99)** | {4:31} |
| 4 | 123 | 48  | 75 | {0:10,1:21,2:17}  | {0:10,3:9,6:29}     | 6 (0.60) | {4:24,5:19,6:32} |
| 5 | 101 | 101 | 0  | {0:57,1:44}       | {5:101}             | 5 (1.00) | (none — no safe clip fell here) |
| 6 | 14  | 12  | 2  | {3:12}            | {2:2,7:10}          | 7 (0.83) | {7:2} |
| 7 | 14  | 8   | 6  | {3:8}             | {2:2,7:6}           | 7 (0.75) | {7:6} |

No cluster is safe-only at k=8, so **every safe clip inherits a real (non-`na`) view** — the `na`
fallback path is unused. Confidence varies a lot though: clusters 3/5/0 have a near-unanimous
dominant view (≥0.99/1.00/1.00); cluster 1 is a weak plurality (0.40, split ~evenly 0/3/6) and
cluster 4 is moderate (0.60) — flag these as lower-confidence view assignments (50 safe clips:
35 in cluster1 + not all of cluster4's 75, but the ambiguity concentrates there).

New file: `sweep_results/zone_prior/clip_view_map_691.json` (691 records — the 516 unsafe records
are byte-copies of the original `clip_view_map.json` entries with `source=original_kmeans`; the
175 safe records carry `source=cluster{N}_dominant_unsafe_view`, `cluster`, `dominance_frac`).
`clip_view_map.json` itself is untouched.

Safe-clip view assignment: **175/175 got a real view, 0 got `na` fallback.**
Distribution: `{view1:31, view2:22, view3:35, view4:4, view6:75, view7:8}`.

Caveat: `annotation_guide/zones/` has NO `view_2.json` or `view_7.json` (pre-existing gap — 56 of
the *existing* 516 unsafe clips, those on views 2/7, already get zero geometry today). The new
safe-clip mapping inherits this: all 30 safe-clip `7_safe_carrying` (view 2/7) will also get zero
geometry — same coverage gap, applied symmetrically to the safe/unsafe counterpart pair, not a new
asymmetric artifact.

### Same-camera validity check (the load-bearing question)

Per-counterpart-pair cluster co-occurrence (does the safe class share the unsafe class's clusters?):

| pair | unsafe class top clusters | safe class top clusters | shared-cluster coverage (unsafe / safe) |
|---|---|---|---|
| 0 walkway_violation ↔ 4 safe_walkway | (3:125),(5:57),(1:18) | (3:31),(4:24),(1:20) | 0.73 / 1.00 |
| 1 unauthorized_intervention ↔ 5 authorized_intervention | (5:44),(4:21),(0:18) | (4:19),(1:15),(0:4) | 0.46 / 1.00 |
| 2 opened_panel ↔ 6 closed_panel | (1:125),(4:17) | (4:32) | **0.12 / 1.00** |
| 3 carrying_overload ↔ 7 safe_carrying | (2:36),(6:12),(7:8) | (2:22),(7:6),(6:2) | 1.00 / 1.00 |

**Verdict: mixed, not a clean yes.** Carrying (3↔7) and, to a lesser extent, walkway (0↔4)
co-cluster well — consistent with same-camera safe/unsafe counterparts. Intervention (1↔5) is
weak (only 46% of unauthorized-intervention clips share a cluster with any authorized-intervention
clip — cluster 5, the single largest unsafe cluster at 101 clips, has ZERO safe clips in it).
Panel (2↔6) is the worst case: only 12% overlap — the 125 opened-panel clips are concentrated in
cluster 1 (which has 0 closed-panel clips), while all 32 closed-panel clips sit in cluster 4 (which
has only 17/125 opened-panel clips). **This is a real, load-bearing negative finding for the panel
pair specifically** — but with an important confound: the background descriptor is a temporal
median of the RAW frame (not a masked region), and an open vs. closed panel door is itself a large,
static visual change in the "background" for that camera. So low overlap for 2↔6 may reflect
"panel state changes the background pixels" rather than "these are different physical cameras" —
the descriptor can't cleanly separate camera identity from panel-state appearance for this specific
pair. For walkway/intervention/carrying there's no equivalent confound (the safe/unsafe distinction
there is behavioural, not a static scene change), so their overlap numbers are cleaner evidence.
Net: the binary task is **not clearly a pure camera-ID task**, but for the panel sub-pair specifically
the safe/unsafe split correlates with a background feature that is either a real different-camera
effect or a panel-state-confounded scene-descriptor artifact — cannot fully disentangle with this
descriptor. Recommend reporting this as an explicit limitation in the thesis binary-task section.

## Step 3 — Token build (691 clips, `bin8_` tag)

`build_relation_tokens.py`: added `--classes` (comma list, default unset = original `[0,1,2,3]`,
verified unchanged) and `--view_map` (override, default `clip_view_map.json`). Smoke-tested with
`--limit 2` first, then full build:

```
.venv/bin/python build_relation_tokens.py --subsets train,test --classes 0,1,2,3,4,5,6,7 \
  --view_map sweep_results/zone_prior/clip_view_map_691.json \
  --detector sweep_results/entity_detector/yolo26_best.pt \
  --appearance --appearance_backbone yolo --out_tag bin8_
```

Ran in background (nohup), log at `sweep_results/relation_tokens/build_bin8_2026-09-16.log`.
Output: `tokens_bin8_train.pt` (566 clips, 31,873 detections), `tokens_bin8_test.pt` (125 clips,
6,774 detections). feat=74 (matches `yolo26app_` layout: 6 base + 4 geometry + 64 appearance).
Existing `tokens_yolo26app_*` / `tokens_yolo26L1_*` untouched.

## Step 4 — Token verification

**Clip counts per class/split** (matches `data/{train,test}/*` directory counts exactly):

| split | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | total |
|---|---|---|---|---|---|---|---|---|---|
| train | 178 | 97 | 129 | 48 | 50 | 23 | 19 | 22 | 566 |
| test  | 32  | 11 | 13  | 8  | 25 | 15 | 13 | 8  | 125 |

**Detections per class** (total / mean-per-clip), train:
0: 12265/68.9, 1: 6846/70.6, 2: 6556/50.8, 3: 771/16.1, 4: 2886/57.7, 5: 1216/52.9, 6: 964/50.7,
7: 369/16.8. Test: 0: 1915/59.8, 1: 655/59.5, 2: 688/52.9, 3: 132/16.5, 4: 1621/64.8, 5: 954/63.6,
6: 684/52.6, 7: 125/15.6. (Classes 3/7 — carrying/safe-carrying — have the lowest detections/clip
in both splits; consistent across safe/unsafe, not a new artifact.)

**Zero-detection clips: 0/566 train, 0/125 test.** Detector fires on every clip.

**Non-zero geometry fraction** (>=1 person box with non-zero iou_walkway/iou_restricted/foot_in/norm_dist):

| split | safe (n) | safe w/ nonzero geom | unsafe (n) | unsafe w/ nonzero geom |
|---|---|---|---|---|
| train | 114 | 90 (78.9%) | 452 | 401 (88.7%) |
| test  | 61  | 53 (86.9%) | 64  | 56 (87.5%) |

Safe and unsafe rates are close (within ~10pp) in both splits — **no "zero-geometry ⇒ safe"
shortcut**: plenty of unsafe clips also have zero geometry (views 2/7 missing zone files, or no
person detected) and most safe clips DO get real geometry from their derived view.

**Unsafe-clip token equivalence vs `tokens_yolo26app_*`**: for all 452 common train / 64 common
test unsafe clip_ids, max abs token diff = **0.000000**, max abs mask diff = **0.000000**, view
mismatches = **0**. Byte-identical, as expected (unsafe view-map entries were copied unchanged,
frame sampling/detector are deterministic).

## Step 5 — Rule-based baseline (TRAIN split only, test untouched)

Rule: predict Unsafe if >=`min_frames` of the T=16 sampled frames have max unsafe-entity conf
(`Person_Unauthorized`, `Panel_Opened`, `Forklift_Overload`) >= tau. Detection pass cached once
(conf floor 0.05) then swept over tau in {0.25..0.9} x min_frames in {1,2,3,5,8,12,16}.

n_train=566 (452 unsafe / 114 safe, 79.9% base rate).

Best operating points (by accuracy):

| tau | min_frames | accuracy | unsafe recall | safe recall | F1 (unsafe-positive) |
|---|---|---|---|---|---|
| 0.25-0.60 | 2-3 | **0.843** | 0.998 | 0.228 | 0.910 |
| 0.25 | 1 (natural default) | 0.839 | 0.998 | 0.211 | 0.908 |
| 0.90 | 16 (strictest) | 0.518 | 0.518 | 0.766 | — (unsafe recall collapses) |

**Trivial "always predict Unsafe" baseline: accuracy = 0.799** (452/566), unsafe recall 1.0, safe
recall 0.0.

**Headline number: the best detector-only rule (0.843 acc) beats the trivial majority baseline by
only +4.4pp**, and does so almost entirely by keeping unsafe recall near-perfect (~99.8%) while
safe recall stays very low (~21-23%) across nearly the whole reasonable tau/min_frames grid — the
detector fires "unsafe entity present" on ~75-80% of genuinely SAFE clips too. Safe recall only
climbs above 0.5 at tau>=0.8 with min_frames>=8, but then unsafe recall collapses in lockstep
(0.52 unsafe recall at the tau=0.9/mf=16 corner) — there is no operating point that is good on
both sides simultaneously. **Conclusion: the detector-alone signal explains very little of the
binary task beyond the class base rate; the relation/temporal model has real work to do.** This is
a useful, unglamorous number for the thesis — it rules out "the detector already solves it."

## Files produced (all new, nothing overwritten)
- `tools/build_scene_grouping.py` — edited: `--classes {unsafe,all,list}` option added
- `build_relation_tokens.py` — edited: `--classes`, `--view_map` options added
- `sweep_results/zone_prior/clip_scene_map_691.json` — 691-clip k=8 scene grouping
- `sweep_results/zone_prior/clip_scene_map_691_k10.json` — k=10 sensitivity check
- `sweep_results/zone_prior/clip_view_map_691.json` — 691-clip view map (516 unsafe unchanged + 175 safe derived)
- `sweep_results/relation_tokens/tokens_bin8_{train,test}.pt` — 691-clip tokens, headline recipe
- `sweep_results/relation_tokens/build_bin8_2026-09-16.log` — build log

## Unresolved questions
1. Panel pair (2 opened <-> 6 closed) has only 12% cluster overlap — is this a real different-camera
   effect, or an artifact of the background descriptor being sensitive to panel open/closed state
   itself? Cannot disentangle without either real camera metadata or a masked/panel-excluded
   descriptor. Recommend trying a descriptor that masks out the panel region if this matters for a
   headline claim.
2. Cluster 1's dominant-view assignment is a weak plurality (39.6%, 3-way near-tie between views
   0/3/6) — 35 safe clips (class4/5) inherit this uncertain view. If downstream zone-prior features
   for these 35 clips turn out to matter, worth a manual spot-check of a few cluster-1 safe clips'
   actual visual background against views 0/3/6's zone polygons.
3. Rule-baseline used conf floor 0.05 then swept tau on cached results — the entity detector itself
   was trained on the 4-class-derived entity ontology; is there a known reason safe-clip detections
   fire so often on "unsafe" entity classes (e.g. Panel_Opened firing on Panel_Closed frames)? Not
   investigated here (would need per-frame visual inspection); flagging as a detector-quality
   question, not a data-prep bug.
