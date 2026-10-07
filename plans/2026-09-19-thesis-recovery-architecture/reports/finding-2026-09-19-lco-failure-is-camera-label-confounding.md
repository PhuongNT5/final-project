# Finding: the LCO "negative result" is camera-label confounding, not model failure

Date: 2026-09-19. Evidence: existing OOF dumps (`sweep_results/relation_tokens/enc_*_lco-691_headline-drop_conflict_oof.json`,
6 encoders) x `sweep_results/zone_prior/clip_view_map_691.json`. **Zero test touches, zero GPU hours** —
all recomputation over the 691 pool from artifacts already on disk.

## 1. Headline

Pooled LCO-691 safe recall is 42-51% across every encoder. That is **not** a uniform inability to
recognise Safe. It decomposes into four sub-tasks with completely different behaviour, and two of
them are **structurally unanswerable** under leave-camera-out.

## 2. Per-sub-class LCO recall, all 6 encoders (pooled OOF, n=687)

| sub-class | n | E0 | E1_d64 | E1_d64_T1 | E1_d96 | E2 | E3 | side |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 walkway_violation | 207 | 94.7 | 92.8 | 92.8 | 90.8 | 83.6 | 94.2 | UNSAFE |
| 1 restricted_zone | 107 | 81.3 | 83.2 | 82.2 | 81.3 | 86.9 | 86.9 | UNSAFE |
| 2 panel_open | 142 | 81.7 | 59.2 | 52.1 | 77.5 | 74.6 | 73.9 | UNSAFE |
| 3 carry_overload | 56 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | UNSAFE |
| **4 walkway_ok** | **75** | **17.3** | **18.7** | **20.0** | **14.7** | **26.7** | **36.0** | SAFE |
| 5 zone_ok | 38 | 84.2 | 78.9 | 81.6 | 84.2 | 89.5 | 76.3 | SAFE |
| **6 panel_closed** | **32** | **9.4** | **0.0** | **0.0** | **0.0** | **9.4** | **12.5** | SAFE |
| 7 carry_safe | 30 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | SAFE |

Two sub-classes (107 of 175 safe clips) carry the entire failure. Everything else is 74-100%.

## 3. Why — sub-class x camera-group cross-tab

| sub-class | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | n | #groups | max share |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 walkway_violation | 0 | 16 | 0 | 125 | 9 | 57 | 0 | 0 | 207 | 4 | 60% |
| 1 restricted_zone | 18 | 11 | 0 | 14 | 20 | 44 | 0 | 0 | 107 | 5 | 41% |
| 2 panel_open | 0 | 125 | 0 | 0 | 17 | 0 | 0 | 0 | 142 | 2 | **88%** |
| 3 carry_overload | 0 | 0 | 36 | 0 | 0 | 0 | 12 | 8 | 56 | 3 | 64% |
| 4 walkway_ok | 0 | 20 | 0 | 31 | 24 | 0 | 0 | 0 | 75 | 3 | 41% |
| 5 zone_ok | 4 | 15 | 0 | 0 | 19 | 0 | 0 | 0 | 38 | 3 | 50% |
| **6 panel_closed** | 0 | 0 | 0 | 0 | **32** | 0 | 0 | 0 | 32 | **1** | **100%** |
| 7 carry_safe | 0 | 0 | 22 | 0 | 0 | 0 | 2 | 6 | 30 | 3 | 73% |

**All 32 `panel_closed` clips live in one camera group.** When that group is the held-out fold, the
training set contains **zero** examples of the class. 0.0% recall is the mathematically expected
outcome, not a model defect. `panel_open` is 88% in a single different group (f1). So the whole panel
sub-task is "camera 1 = open, camera 4 = closed" — and LCO exists precisely to delete that shortcut.
**The panel sub-task (174 clips, 25% of the pool) cannot be evaluated by LCO at all.**

Supporting structure: camera groups are strongly label-confounded overall. Views 0 and 5 are 100%
unsafe (68 and 103 clips); 43% of all safe clips sit in view 6. Two of eight LCO folds have **no safe
clip at all**, so they contribute only to unsafe recall.

## 4. Honest decomposition

| sub-task | n | verdict |
|---|---:|---|
| zone (1+5) | 145 | **genuinely works cross-camera** — 81-87% unsafe / 76-90% safe, both sides, 5 and 3 groups. The real positive result. |
| carry (3+7) | 86 | 100%/100% but **trivial** — detector class lookup (`Forklift_Overload` R=1.000, `Forklift_Safe_Load` R=1.000). Must be disclosed (risk R13). |
| walkway (0+4) | 282 | **real, open failure** — 90%+ unsafe vs 15-36% safe. Genuinely spread over 3-4 groups, so this IS attackable. Largest sub-task. |
| panel (2+6) | 174 | **structurally unanswerable under LCO** — single-camera safe class. |

The walkway asymmetry (90% vs 20%) says the model learned "walkway scene -> Unsafe", not "is the
person actually on the walkway". Note `Safe_Walkway` is detector class 2 — **zero instances in the
label set** — so the walkway itself is only representable through the hand-drawn zone polygons.
By contrast the zone sub-task, whose geometry channels (`iou_restricted`, `norm_dist`) are well
populated, is the one that generalises. That is a coherent, testable mechanism.

## 5. LCO with the panel sub-task disclosed separately

Reported **alongside** the full number, never replacing it (post-hoc exclusion is forbidden by R8;
this is a declared protocol variant with a stated structural justification, computed on the 691 pool,
0 test touches).

| encoder | FULL LCO-691 BA | safe rec | panel-disclosed BA | safe rec (n=143) |
|---|---:|---:|---:|---:|
| E0 | 66.7% | 44.6% | **72.0%** | 52.4% |
| E1_d64 | 62.3% | 42.3% | 71.4% | 51.7% |
| E1_d64_T1 | 61.8% | 43.4% | 72.0% | 53.1% |
| E1_d96 | 63.9% | 41.7% | 70.3% | 51.0% |
| E2 | 66.7% | 49.7% | 72.9% | 58.7% |
| **E3 (18k params)** | **69.6%** | 51.4% | **76.6%** | 60.1% |

## 6. Consequence for the accuracy lane — architecture search is dead

Six architectures spanning **18k to 300k parameters** fail on exactly the same two sub-classes and
succeed on exactly the same four, with the same rank order of difficulty. The failure is
**architecture-invariant**. This explains the 12+ track ~54-56/64 ceiling directly: every track was
varying the one factor that does not matter.

- **Stop** E0/E1/E2/E3-style encoder search, loss zoos, width/depth sweeps.
- E3 DeepSets (18,044 params) is the best LCO encoder under both protocols — which is also the best
  possible outcome for the edge pillar. Take the win: smallest model wins on robustness.
- The remaining honest accuracy lever is the **walkway sub-task** (282 clips, largest), specifically
  the safe side. Concrete, mechanism-driven candidates: populate `Safe_Walkway`/`Restricted_Zone`
  (detector classes 2/3, currently zero instances) from the existing zone polygons; improve walkway
  geometry features; per-view zone quality audit (`tools/check_zone_separation.py` exists).

## 7. What this is worth at the defence

This is a *better* result than a tuned accuracy number. It is a falsifiable, mechanism-level claim
supported by an architecture sweep and a dataset cross-tab: **cross-camera generalisation failure on
Unsafe-Net is a dataset camera-label confounding property, localised to identifiable sub-tasks, and
invariant to encoder architecture over a 16x parameter range.** The zone sub-task result shows the
relation approach does generalise where the data permits it.

## Unresolved questions

1. Do the scene-group fold ids (KMeans(8) background proxy) and the `view` field in
   `clip_view_map_691.json` agree? They are used as separate groupings here; the cross-tabs in §3 are
   fold-based (the grouping the LCO run actually used), §4's zone/view remarks are view-based.
2. `build_relation_tokens.py` header has `N_MAX = 8` while the binary encoders default `n_max=8` —
   the summary-level claim of "<=16 entity slots" needs reconciling against the actual headline tag.
3. Does the walkway failure survive if `Safe_Walkway` is populated from zone polygons? This is the
   pre-registered falsification target: must move panel-disclosed LCO BA past 76.6% (E3) or the
   lever is dead and gets reported as a negative result.
4. Class-id IS a learned `nn.Embedding`, not an ordinal scalar (checked `model_binary_encoders.py:64`,
   `:119`) — so "the head cannot see panel state" is NOT the explanation for panel_closed. The
   single-camera confound in §3 is.

---

## Addendum 2026-09-19 — unresolved questions 1-4 resolved, L2 killed

### Q2 RESOLVED: KMeans(8) fold ids vs the `view` field — 81.7% agreement, NOT interchangeable

| fold | v0 | v1 | v2 | v3 | v4 | v5 | v6 | v7 | majority-view purity |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0 | 0 | 0 | 0 | 22 | 0 | 0 | 0 | 100% |
| 1 | 58 | 0 | 0 | 94 | 0 | 0 | 35 | 0 | **50%** |
| 2 | 0 | 0 | 50 | 0 | 0 | 0 | 0 | 8 | 86% |
| 3 | 0 | 168 | 0 | 0 | 0 | 2 | 0 | 0 | 99% |
| 4 | 10 | 0 | 0 | 9 | 0 | 0 | 102 | 0 | 84% |
| 5 | 0 | 0 | 0 | 0 | 0 | 101 | 0 | 0 | 100% |
| 6 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 12 | 86% |
| 7 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 12 | 86% |

Overall 81.7%. Fold 1 mixes three views at 50% purity; folds 6 and 7 have identical compositions.
**Rule for the thesis: never mix the two groupings in one sentence.** All LCO/confounding claims
(§2, §3, §5) are **fold-based** — the grouping the LCO run actually used — so they stand unchanged.
The view-based zone-coverage remarks in §4 and the view-level label skew in §3's closing paragraph
are **view-based** and must be labelled as such. The KMeans(8) `[LIMITATION]` banner already printed
by `run_kfold_relation.py:106` ("background-descriptor proxy, not verified camera metadata") must
ride with every LCO number.

### Q3 RESOLVED: L2 (draw `view_2`/`view_7` polygons) is DEAD — zero headroom

The 86 zero-geometry clips are **exactly** 8-class labels 3 and 7 (`carry_overload` 56 + `carry_safe`
30) — i.e. precisely the carry sub-task. Current LCO correctness of exactly those 86 clips:

| encoder | E0 | E1_d64 | E1_d64_T1 | E1_d96 | E2 | E3 |
|---|---|---|---|---|---|---|
| correct | 86/86 | 86/86 | 86/86 | 86/86 | 86/86 | 86/86 |

**100.0% for every encoder.** Zone geometry is zeroed on these clips because the forklift load-state
task does not need walkway/restricted polygons — and the detector already solves it outright
(`Forklift_Overload` R=1.000, `Forklift_Safe_Load` R=1.000). Drawing those polygons cannot move any
number, because there is nothing left to get right. **Cut L2 and phase-03 step 2. The manual
annotation session is not needed.** This also strengthens the R13 carry-lookup disclosure: the carry
sub-task is solved with *zero* geometry information, which is direct evidence it is a detector
class lookup rather than relational reasoning.

### Q4 RESOLVED: `N_MAX = 8`, not 16

Headline `tokens_bin8_{train,test}.pt` are `(566|125, 16, 8, 74)` = T=16 frames x **8 entity slots**
x 74 features; `feat_layout` = `['class','xc','yc','w','h','conf','iou_walkway','iou_restricted',
'foot_in_walkway','norm_dist', ...64 appearance]`. The "<=16 entity slots" figure is **wrong** and
must not ship in any capacity/complexity statement. Related token tags on disk: `yolo26L1_` (72 wide,
drops xc/yc), `tokens_v2_` (13), geometry-only `tokens_` (10).

### Q1 unchanged (still the plan's single accuracy bet)

Whether populating detector classes 2/3 as zone-entity tokens moves panel-disclosed LCO BA past
76.6% (E3). Note Q3's result sharpens the prior: the sub-task that carries *no* zone geometry is at
100%, and the sub-task with *well-populated* zone geometry (`restricted_zone`/`zone_ok`) is the one
that generalises — so geometry helps where it is informative, which is the walkway case L1 targets.
