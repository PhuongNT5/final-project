---
priority: must-have
effort: 3 person-days
depends_on: []
branches: A=subsample 401→200→100 from existing labeled pool (primary), B=2-point curve if subsampling infeasible
---

# Phase 03 — Data-Efficiency Curve

## Context Links

- `PROGRESS.md` — V10 bbox-finetune (400 frames) = 84.73 %; gap analysis §"Why 95.81% Cannot Be Reached"
- `report/chapter-04-experiments-and-results.md` §4.6 — V10 negative result; needs reframing
- `/tmp/unsafe-net-paper.txt` Table 3 — 2,262 labeled training images (paper budget)
- `sweep_results/v6_detection/features/` — cached fine-tuned YOLO features (existing)
- `run_pairwise_contrastive.py` — best ensemble training script (re-use for head training)
- `run_ablation_inference.py` — referenced in git status; not yet analyzed

## Overview

V10 showed that fine-tuning YOLOv11 on behavior-class bbox frames regressed accuracy to 84.73 %
(54/64, seed s2024 — `sweep_results/finetune_bbox/results.json`), worse than the frozen baseline
(89.28 %). **Already available:** 5 trained V10 checkpoints (s42, s123, s314, s7, s2024) using the
full 401-frame labeled pool (`annotation_guide/`). `run_cross_ensemble.py` provides the
cross-ensemble comparison. This phase only needs to:
(a) run two subsampled budgets (200 and 100 frames — one seed each, to save GPU time),
(b) collect the 3-point curve (100 / 200 / 401 + frozen baseline at 0),
(c) extrapolate to the paper's 2,262-frame budget.

Use **401** (not "~400") as the upper budget — exact count from `annotation_guide/`.
The 84.73 % V10 result is the 401-frame anchor; no re-run needed for that point.

## Key Insights

- **401 labeled frames confirmed** in `annotation_guide/` (8 classes, class_0…class_7). YAML at `annotation_guide/annotations/dataset.yaml`.
- V10 result = 84.73 % macro (54/64, seed s2024). This is the 401-frame anchor. 5 seeds already trained; no re-run needed.
- `run_cross_ensemble.py` (untracked): cross-ensemble of frozen-COCO and V10 bbox-finetuned branches; max 55/64. Reuse output.
- `run_ablation_inference.py` (untracked): runs T-robustness and per-member accuracy — not a budget sweep, but can be reused for head-training at reduced budgets.
- Paper budget: 2,262 images. Extrapolation gap from 401 → 2,262 is 5.65×.
- Subsample strategy: 401 → 200 (50 per class) and 401 → 100 (25 per class) for the two new curve points. Use single seed (s42) for each subsampled run to save GPU time.
- Expected curve: monotonic increase (100 < 200 < 401) but all below frozen baseline (89.28 %). That is the finding: even 401 real bboxes are insufficient; you need ~2,262.
- Max available = 401. No additional labeling planned. Use "401 = max available" as upper label.

## Requirements

**Functional**
- Three training runs using V10 protocol: label budgets 100 / 400 / max-feasible frames.
- Each run: fine-tune YOLOv11s on bbox-subset → extract features → train best-head pipeline (direct4 + cascade ensemble) with 5-seed sweep → report mean macro accuracy.
- Plot: x-axis = annotation count (log scale: 100, 400, max, [2262 extrapolated]), y-axis = macro accuracy %. Mark paper's 95.81 % at 2,262 as a reference point.
- Table: annotation cost (hours per 100 frames estimated), accuracy, vs. frozen-backbone 89.28 % at 0 annotations.

**Non-functional**
- Each budget run ≤ 10 hours training time (comparable to V10's 8.6 h).
- Reuse existing 400-frame labeled set for the 400-budget run (no re-labeling).
- Figure saved to `figures/data-efficiency-curve.png`.

**Acceptance thresholds**
- At least 3 data points (including 0 = frozen baseline = 89.28 %).
- Curve is presentable (not all points identical) — even if all bbox-finetune runs are below 89.28 %, that is the finding.
- Extrapolation line reaches ≥ 93 % at 2,262 to justify future-work claim; if not, report honestly.

## Architecture / Approach

### Label Budget Construction

```
100-frame budget:  sample 25 frames per class from the 400-frame labeled pool
400-frame budget:  use existing 400-frame V10 dataset (already labeled)
max-feasible:      label additional N frames (if time permits) OR use all available
```

If no additional labeling is possible, run 100 / 400 only (2-point curve + frozen baseline = 3 points).

### Training Protocol (per budget)

1. Fine-tune YOLOv11s on bbox subset with V10 hyperparameters (see `sweep_results/v6_detection/` for config).
2. Extract features for all training clips (T=32, 224×224) using fine-tuned backbone.
3. Train direct4 head + scenario+pair cascade: 5-seed sweep (seeds 1,7,42,314,999).
4. Record per-seed macro accuracy on 64-clip test set.
5. Compute mean ± std; plot.

### Extrapolation

Fit a log-linear model: `accuracy = a * log(n_annotations) + b` on the 2–3 measured points. Extrapolate to n=2,262. Report 95 % CI of extrapolation. Frame as: "at paper budget, our architecture is expected to reach X % ± Y %."

### Figure

```
100% |                                      * 95.81 (paper, 2262 bbox)
     |                        extrapolated ↗
     |                   * max-feasible
 89% | * frozen (0)    / ↓ V10 results
     |      * 100     * 400
     |_________________________ annotation count (log)
     0   100  400  max  2262
```

## Related Code Files

| Action | Path |
|--------|------|
| Read/reuse | `run_ablation_inference.py` — check if it implements budget sweep |
| Modify | `run_pairwise_contrastive.py` — add `--bbox-budget N` flag to subsample training data |
| Create | `run_data_efficiency.py` — orchestrates 3-budget sweep, outputs results JSON |
| Create | `figures/data-efficiency-curve.png` |
| Modify | `report/chapter-04-experiments-and-results.md` §4.6 — reframe V10 as data-efficiency finding, add curve figure |

## Implementation Steps (Day-by-Day, 3 days)

**Day 1 — Subsample construction + 100-frame run**
1. Verify `annotation_guide/` contains 401 frames across 8 classes. Confirm YAML is valid.
2. Build 100-frame stratified subsample: 25 frames per class (8×25 = 200... adjust if some classes < 25; use all available for minority classes). Save to `annotation_guide/subsets/budget_100/`.
3. Build 200-frame subsample: 50 per class → `annotation_guide/subsets/budget_200/`.
4. Run budget=100 experiment (seed s42 only): fine-tune backbone 30 epochs → extract features → train heads → record accuracy. Expected ≈ 3–5 h.

**Day 2 — 200-frame run + anchor points**
5. Run budget=200 experiment (seed s42 only). Record accuracy.
6. Collect 401-frame anchor from existing results: V10 s2024 = 84.73 % (54/64). Also run `run_cross_ensemble.py` to get cross-ensemble max (55/64 max per ground truth).
7. Frozen baseline anchor = 89.28 % at 0 bbox annotations.

**Day 3 — Extrapolation + figure + chapter update**
8. Fit log-linear extrapolation: `np.polyfit(np.log([100, 200, 401]), [acc_100, acc_200, 84.73], 1)`. Predict at n=2,262 with 95 % CI.
9. Generate `figures/data-efficiency-curve.png`: log-scale x-axis, 4 points (0=frozen baseline, 100, 200, 401), extrapolation line to 2,262, paper 95.81 % annotated.
10. Update `report/chapter-04-experiments-and-results.md` §4.6: rename to "V10 — Behaviour-Class Bbox Finetune: Data-Efficiency Analysis"; embed curve figure; add extrapolation sentence. Use 401 (not ~400) throughout.

## Todo List

- [ ] Day 1: Confirm 401 frames in annotation_guide/; verify YAML valid
- [ ] Day 1: Build budget_100 subsample (25/class stratified)
- [ ] Day 1: Build budget_200 subsample (50/class stratified)
- [ ] Day 1: Run budget=100 experiment (seed s42); record accuracy
- [ ] Day 2: Run budget=200 experiment (seed s42); record accuracy
- [ ] Day 2: Collect 401-frame V10 anchor (84.73%, s2024) + cross-ensemble (55/64)
- [ ] Day 2: Confirm frozen baseline anchor = 89.28% at 0 bboxes
- [ ] Day 3: Fit log-linear extrapolation to n=2262; compute 95% CI
- [ ] Day 3: Generate data-efficiency-curve.png (4 points + extrapolation + paper ref)
- [ ] Day 3: Update §4.6 with data-efficiency framing, figure, and 401 (not ~400)

## Success Criteria

- ≥ 3 data points on the curve (including frozen baseline at 0 annotations).
- Figure `figures/data-efficiency-curve.png` exists and is referenced in §4.6.
- Extrapolation reaches a stated predicted accuracy at n=2,262 with stated uncertainty.
- §4.6 no longer reads as a regret; reframes V10 as "under-budget fine-tuning confirms the frozen baseline is the annotation-efficient operating point."
- Budget=400 result matches previously reported V10 macro 84.73 % (within ±1 pp for run-to-run variance).

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| All bbox-finetune runs below frozen baseline (U-shape absent, monotonic low) | Medium | That IS the finding: frozen COCO beats any sub-2262-bbox fine-tune. Report curve with that interpretation. |
| Insufficient labeled data for 3-point curve | Medium | Use 2 points (100, 400) + extrapolation; 3-point curve is nice-to-have |
| Budget=400 re-run gives different result from V10 84.73% | Low | Variance of ±1–2 pp expected; acceptable; report mean ± std across 5 seeds |
| run_ablation_inference.py does something incompatible | Low | It is untracked new code; read it first and decide; worst case, write run_data_efficiency.py from scratch |
| Extrapolation line misses 95.81% by a large margin | Low | Report the gap honestly; thesis already says "data, not architecture"; extrapolation that reaches 90–93% still supports the narrative |

## Security / Ethics / Privacy

- Bbox annotation labeling (if extended): use LabelImg on factory footage. Consistent with paper's ethics approval (§2.1 paper). Labeled files should not include personal identifiers.
- No new data collection required for this phase.

## Dependencies + Next Steps

- Output consumed by: **Phase 04** (pseudo-bbox framing depends on this curve); **Phase 06** (seed runs here feed ablation table); **Phase 07** (§4.6 rewrite, §1.4 Contribution 4 update).
- Landing chapter: `report/chapter-04-experiments-and-results.md` §4.6, plus one paragraph in §1.4 Contribution 4.

**Unresolved questions**
- How many bbox-labeled frames currently exist? PROGRESS.md says "~400 frames" but does not give an exact count. The student must check the actual directory before starting Step 2.
- Does `run_ablation_inference.py` implement any budget logic already?
