# Phase 01 — Binary Data Prep + Rule Baseline

## Context Links

- Dependency (in flight): `plans/2026-09-06-thesis-2week-final/reports/laneb-2026-09-16-binary-data-prep.md`
  — steps 1-2 landed, steps 3-5 pending. **Do not redo steps 1-2.**
- `plans/2026-09-06-thesis-2week-final/reports/research-2026-09-16-unsafe-net-paper-review-and-binary-task-design.md`
  — recommends running the rule baseline FIRST: "if it achieves >85% accuracy alone, the added Transformer value is marginal."
- `brief.md` §4h eval contract.

## Overview

- **Priority:** P1 — blocks everything.
- **Status:** partially done (external agent).
- **Description:** produce the 691-clip binary relation-token set and a train-split-only rule baseline.

## Key Insights

- `build_relation_tokens.py` **already supports** `--classes`, `--out_tag`, `--view_map`, `--detector`,
  `--appearance{,_backbone}` (lines 228-241). The earlier audit claim that `UNSAFE=[0,1,2,3]` is hardcoded
  and needs an edit is **wrong**. No code change is needed for the token build.
- True clip counts (each `.mp4` has a paired `.mp4.npy` cache — folder file counts are 2×):
  train `0:178 1:97 2:129 3:48 | 4:50 5:23 6:19 7:22` (452 U / 114 S);
  test `0:32 1:11 2:13 3:8 | 4:25 5:15 6:13 7:8` (64 U / 61 S).
- Safe sub-classes are **tiny** (19-50 train clips) — stratify on the 8 sub-classes, not on the binary label.
- 175/175 safe clips received a real view; 30 safe clips sit on views 2/7 which have **no zone polygons** →
  zero geometry, symmetric with the 56 existing unsafe clips in the same views. Not a new asymmetry.
- Building tokens over the test split is an **artifact build, not an evaluation** → **0 test touches**.

## Requirements

Functional: `bin8_{train,test}.pt` tokens for all 691 clips, shape `(T=16, N_MAX=8, 74)`, 8-way sub-class
label + derived binary label per clip; a `clip_id` field that survives into fold assignment.
Non-functional: build ≈2.4 s/clip → ~28 min for 691 clips; deterministic given the fixed detector weights.

## Architecture

`data/{subset}/{c}_*/*.mp4` → YOLO26 entity detector (`sweep_results/entity_detector/yolo26_best.pt`)
→ per-frame top-8 boxes by confidence → geometry vs `clip_view_map_691.json` zone polygons
→ 10 geometric dims + 64 appearance dims = 74-d tokens → `bin8_{subset}.pt`.
Binary label = `1 if sub_class in {0,1,2,3} else 0`.

## Related Code Files

Modify: none required for the build. Create: `tools/verify_bin8_tokens.py`,
`tools/rule_baseline_binary.py`. Read: `build_relation_tokens.py`, `tools/build_scene_grouping.py`.

## Implementation Steps

1. **Token build (step 3, running).** Confirm completion of:
   ```
   python -u build_relation_tokens.py --classes 0,1,2,3,4,5,6,7 --appearance \
     --appearance_backbone yolo --detector sweep_results/entity_detector/yolo26_best.pt \
     --view_map sweep_results/zone_prior/clip_view_map_691.json \
     --out_tag bin8_ --subsets train,test
   ```
   Log: `sweep_results/relation_tokens/build_bin8_2026-09-16.log`. Wall-clock ≈28 min. **0 test touches.**
2. **Verify (step 4).** Assert: 566+125 clips present; no NaN/Inf; `conf` ∈[0,1]; geometry dims exactly 0 for
   the 86 view-2/7 clips and non-zero elsewhere; per-sub-class counts match the table above; binary label
   derivation correct for all 691.
3. **Rule baseline (step 5) — train split only.** Hand-written rules per sub-class concept
   (foot_in_walkway, entity-in-restricted-zone, panel-box present, carried-object count/area).
   Report binary accuracy/precision/recall/F1 on **train+val only**. **0 test touches.**
4. Record the baseline number in the number sheet as `RULE-BIN-TRAINVAL`.

## Todo List

- [ ] Token build finishes cleanly (exit 0, 691 clips)
- [ ] `tools/verify_bin8_tokens.py` written and passing (G1)
- [ ] Rule baseline implemented, train/val only
- [ ] Baseline number logged; if >85% escalate to user immediately
- [ ] Data-prep report steps 3-5 completed by the lane-B agent

## Success Criteria

G1: verification script exits 0 on both subsets and the per-sub-class counts match. Rule baseline number
exists with an explicit statement that no test clip was scored.

## Risk Assessment

R5 (tiny safe sub-classes) — mitigate by stratified folds. If the rule baseline exceeds 85%, the
Transformer's marginal value collapses; escalate rather than quietly proceeding (see Next Steps).

## Security Considerations

Dataset is local factory footage; no PII leaves the machine. No artifact overwrite: new files use the
`691`/`bin8_` suffixes, the 516-clip canonical maps stay untouched (guards against S9-style silent overwrite).

## Next Steps

Unblocks phase 02 (training) and phase 06. If rule baseline >85% → notify the user before phase 02 starts,
because the thesis narrative would need to shift to "learned model vs strong rule baseline".
