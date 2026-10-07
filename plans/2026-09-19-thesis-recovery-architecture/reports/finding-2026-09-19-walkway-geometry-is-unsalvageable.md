# Finding: the walkway geometry channel cannot be repaired — phase-03 is dead

Date: 2026-09-19. Zero test touches (recomputed from stored tokens, no re-detection, no training).

## Why this was run

Phase-03 ("walkway geometry repowering") was the plan's **single accuracy bet**: the walkway pair
(`walkway_violation` n=209 vs `walkway_ok` n=75) is the only sub-task that must be separated by
geometry alone, and it is the worst performer (walkway_ok LCO recall 17-36%). The bet assumed the
geometry signal was real but under-exploited. It is not.

## 1. The feature is mis-normalised (real defect, but not the real problem)

`build_relation_tokens.py:109-115` computes `iou_walkway` as intersection / **union**. The walkway
polygons are 53k-113k px^2; the median person box is 19.5k px^2. So for a person standing *entirely
inside* the walkway the feature still reads only 0.18-0.34, and it conflates "person is small/far"
with "person is outside the walkway". It is a size feature wearing a containment feature's name.

Replaced it with containment = intersection / **person-box area** (size-invariant, 1.0 = fully inside),
plus a signed foot-to-boundary distance. Recomputed over all 284 walkway clips:

| feature | violation (0) | walkway_ok (4) | Cohen d | best single-threshold BA (in-sample) |
|---|--:|--:|--:|--:|
| `iou_walkway` (current) | 0.0151 | 0.0201 | 0.239 | 69.6% |
| containment (proposed fix) | 0.1202 | 0.1552 | **0.339** | **63.4%** |
| signed foot distance (proposed) | -0.2574 | -0.2555 | 0.014 | 58.0% |
| `foot_in_walkway` (current) | 0.1513 | 0.2533 | 0.603 | 67.8% |

**The fix does not work.** d rises 0.24 -> 0.34 (both negligible) and in-sample threshold accuracy
*falls*. Note these BA figures are fitted on the same data they are scored on — they are optimistic
upper bounds, not estimates. The best existing feature remains `foot_in_walkway` at d=0.603.

## 2. The real problem: the zone polygons do not describe the scenes

The number that settles it: **in clips labelled `walkway_ok` — a person correctly using the walkway —
the person's foot is inside the annotated walkway polygon only 25% of the time** (containment 0.155).
A correct annotation would put this near 1.0.

Three cheap rescue hypotheses were tested. **All three are falsified.**

**H1 - the clip->view map is wrong** (cheap fix: remap + rebuild). Test: score each of the 284
walkway clips against *every* view's polygons and take the best. That is an oracle (it chooses per
clip using information we would not have), so it upper-bounds anything H1 could deliver.

| | foot-inside rate |
|---|--:|
| assigned view's polygons (current) | 0.178 |
| **best-of-any-view polygons (ORACLE, cheating)** | **0.194** |

Assigned view == best-fitting view for only 105/284 clips (37.0%), yet fixing all 179 disagreements
buys **+1.6 pp**. H1 falsified. Under the oracle, 96 of 284 clips still have <10% foot-inside.

**H2 - a global scale error** (polygons drawn at a different resolution). Swept scale x0.75 to x2.5
about the frame centre: x1.0 = 22.8% foot-inside, every other scale is WORSE (x1.25 -> 1.4%,
x1.5+ -> 0.0%). H2 falsified.

**H3 - a global translation offset** (polygons drawn on a crop/pad). Swept 18 offsets, dx -800..+200,
dy -200..+200. dx=0, dy=0 is already best of all 18. H3 falsified.

### What is actually going on

The polygons are drawn where they were meant to be. The problem is the premise.

Every view's `safe_walkway` polygon occupies the **rightmost ~27% of the frame** (x from 1391-1489 to
1920) while people in walkway clips are centred near the middle (x ~= 1000). And the two classes'
people stand in **the same place**:

| view | `walkway_violation` foot centroid | `walkway_ok` foot centroid |
|---|---|---|
| 1 | (1016, 619) n=5934 | (1169, 564) n=1513 |
| 3 | (1082, 477) n=554 | (1023, 656) n=788 |
| 6 | (885, 553) n=559 | (982, 609) n=1087 |

In view 3 the "ok" people are further **left** than the violators — the sign of the difference is not
even consistent across views. At the best possible offset the foot-inside gap between the two classes
is **+8.1 pp** (22.8% vs 14.8%): that is the entire geometric separation available.

**The two classes are spatially indistinguishable.** No polygon, however drawn, separates them by
person position, because their person positions coincide. Whatever distinguishes `walkway_violation`
from `walkway_ok` here, it is not where the person is standing — so the geometry premise of the
sub-task is wrong at the root, not the geometry implementation.

## 3. Consequence — phase-03 is killed

Repowering the geometry cannot beat the 76.6% panel-disclosed LCO BA bar, because there is no signal
to repower. The only route to a working walkway feature is redrawing walkway polygons **per clip**
(284 clips, manual) and even then §2 gives no reason to expect the labels to agree with the redrawn
geometry.

This closes the last open accuracy lever on Unsafe-Net. Standing evidence, now complete:

| sub-task | status | why |
|---|---|---|
| carry (3v7) | solved, worthless | detector class label alone, 100/100; zero relational content |
| panel (2v6) | unanswerable | all 32 `panel_closed` clips in ONE camera group -> LCO recall is arithmetic |
| zone (1v5) | **works, and survives class-blinding** | class + geometry, `iou_restricted` separates 4.4x |
| walkway (0v4) | **dead (this report)** | geometry-only, and the geometry annotation does not describe the scenes |

Combined with the earlier result that six encoders from 18k to 300k params produce near-identical
per-sub-class profiles, **the dataset is the binding constraint, not the model.** No architecture
and no feature engineering on Unsafe-Net changes the headline.

## Unresolved questions

1. **RESOLVED (H2/H3 above):** scale and translation were the only cheap explanations left. Both
   swept, both falsified — x1.0 and dx=dy=0 are already optimal.
2. Do the `walkway_ok` labels actually mean "in the walkway", or something looser (e.g. "not
   obstructing")? If the latter, the geometry premise of the whole sub-task is wrong, not the data.
3. Does the zone pair's `iou_restricted` suffer the same union-normalisation flaw? It still works
   (4.4x separation), so it is not urgent — but the same fix might strengthen the one positive result.
