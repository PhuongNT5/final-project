# Phase 0 · D1 [CRITICAL] — repaired LCO scene grouping (classifier-independent)

**Date:** 2026-09-03. Additive (only my own scripts edited; no thesis files touched).
**Verdict: the scene-robust LCO headline is NOT an artifact of the circular grouping — it survives the
repair and marginally improves.**

## The bug
The `views` used by LCO/k-fold were `KMeans(8)` clustered on the **same 8192-d appearance features used
for classification** (`tools/apply_zone_prior.py:62`). No real camera IDs exist in `data/`
(filenames are `{class}_tr{N}.mp4`). A leave-"camera"-out split whose groups are defined by the
classifier's own features is circular — the scene-only baseline and the held-out difficulty are both
contaminated.

## The fix
`tools/build_scene_grouping.py`: cluster on each clip's **static background** — temporal-median of its
32 pre-extracted frames, grayscale, block-downscaled to 32×32, per-clip contrast-normalized → a scene
descriptor independent of the moving subject and of the classifier features. `KMeans(8)` →
`sweep_results/zone_prior/clip_scene_map.json` (516 clips). `run_lco_eval.py --scene_map` overrides the
views with this grouping.
- **ARI(background-scene vs old appearance-KMeans) = 0.690** → the old grouping already tracked real
  scenes ~69%, so the old LCO was not invalidated, but the new grouping is principled (no circularity).

## Result — LCO under the repaired grouping (YOLO26, seeds=3, epochs=60, 516 pooled)

| metric | OLD (appearance-KMeans, circular) | **NEW (background, classifier-independent)** |
|---|---|---|
| top-1 | 398/516 = 77.1% | **405/516 = 78.5%** |
| macro | 78.6% | **80.1%** |
| scene-only top-1 | 16.9% | 16.5% |
| scene-only macro | 10.4% | 10.1% |
| **behaviour signal vs scene** | +60.3 pp | **+62.0 pp** |

Per-class recall (repaired LCO): WalkwayViol 79.5 · **UnauthorInt 52.8** (was 44.4) · OpenedPanel 88.0 ·
CarryOverload 100.0. Confusion: WalkwayViol↔UnauthorInt still the only real confusion (25+40 off-diag),
consistent with D2 (geometric, not temporal).

Per-held-out-scene: mix of easy (view1 84%, view2/5 100%, view6 81%) and hard (view0 56%, view3 55%,
view4 68%) — the hard scenes are the low-n / mixed-class ones, as expected under a genuine cross-scene
split.

## Why this matters (the whole headline rested on it)
The campaign's central claim is "modern entity+relation model wins on **scene-robust** generalization,
where the paper's scene-confounded whole-frame regime collapses." That claim is only as good as the LCO
grouping. **Under a grouping that cannot see the classifier's features, the result holds (80.1% macro,
+62pp over scene-only).** The scene-only baseline still collapses to ~10% macro — it never saw the
held-out background. Plan Unresolved-Q#1 is resolved: temporal-median-background clustering is an
acceptable, defensible scene grouping for the thesis headline.

## New honest headline (Phase-0 baseline, pre-Phase-1 levers)
**LCO macro 80.1% (top-1 78.5%) under a classifier-independent scene grouping; scene-only 10.1%
(+62.0pp).** This replaces 78.6% as the true scene-robust baseline the Phase-1 levers (L1→L2→…) must beat.

## Unresolved questions
1. Re-run YOLO11 LCO + k-fold under the repaired grouping for a matched ablation table? (YOLO26 done;
   YOLO11 pending — expected within noise given detector-agnostic k-fold.)
2. 8 background clusters vs more/fewer — is 8 the right domain count for the eventual GRL (M1)? (D1
   grouping now also feeds M1's view labels.)
