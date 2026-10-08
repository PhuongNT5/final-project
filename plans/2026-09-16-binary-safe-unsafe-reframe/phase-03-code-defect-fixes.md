# Phase 03 — Code-Defect Fixes

## Context Links

- `plans/2026-09-06-thesis-2week-final/reports/review-2026-09-13-eval-statistics.md`
- `plans/2026-09-06-thesis-2week-final/reports/review-2026-09-13-edge-deployment.md`
- `plans/2026-09-06-thesis-2week-final/reports/audit-2026-09-14-edge-and-demo-readiness.md`
- `design-2026-09-16-binary-encoder-redesign.md` §7 (implementation surface) — source of D6-D8.
- Background agents are already working on the McNemar replacement and the deployment fixes —
  **do not duplicate**; this phase tracks and verifies them.

## Overview

- **Priority:** P1 for D2 (blocks 19/9) and D6/D7 (block correct binary evaluation). **Effort:** 8h.

## Key Insights

- D2 is the only defect on the **Jetson critical path**: the binary head exports with **2 inputs**
  (`tokens [B,16,8,74]`, `mask [B,16,8]`) and the current runner binds one. An unbound input reads garbage
  device memory → **silently wrong outputs, not a crash**.
- D6/D7 are *correctness* defects for the binary task specifically, not cosmetic: they change which number
  the thesis reports.

## Requirements

| ID | Defect | Location | Fix |
|---|---|---|---|
| D1 | `two_proportion_z` on **paired** data (inflates significance) | `run_full_eval.py:123-124,168`; `run_relation_transformer.py:220,236`; `ensemble_relation.py:43-46` | McNemar exact + `mcnemar_discordance_bound` from `eval_utils/paired_tests.py` |
| D2 | TRT runner binds only input 0 | `jetson-benchmark.py:63`; `jetson-multi-stream-pipeline.py:96` | bind all engine inputs by name; assert `n_inputs == len(feeds)` |
| D3 | `aggregate_fps` counts dropped frames as processed | `multi-stream-pipeline.py:151,154`; `jetson-multi-stream-pipeline.py:219,222` | count completed decisions; report drop rate separately |
| D4 | P3 ONNX benchmark excludes NMS/crop/geometry, uses mean crops (58) | `benchmark-p3-onnx.py:44` | full per-decision path; report p95 (106) and cap (128) |
| D5 | k-fold / LCO have no clip include-exclude flag | `run_kfold_relation.py`, `run_lco_eval.py` | `--exclude_clips <file>` for the detector-leakage-free variant |
| **D6** | **val split stratifies on label only**; brief §4h.1 requires **(label, view)** | `run_relation_transformer.py` | stratify on the (label, view) pair for all binary runs |
| **D7** | **`use_cw=False`** in the k-fold script contradicts the prior-matched decision (train 3.96:1, test ~1:1) | `run_kfold_relation.py` | `use_cw=True`; `class_weights(y, k=2)` → 0.403 / 1.597 |
| **D8** | `build_wholeframe_features.py:21` hardcodes `UNSAFE=[0,1,2,3]` → S2 scene baseline cannot cover 691 | `build_wholeframe_features.py` | accept all 8 classes; rebuild features for 691 |
| **D9** | `eval_utils/cost_matrix.py` defaults `unsafe_classes` to **all** classes → silently wrong 10:1 cost | call sites | always pass `unsafe_classes=[1]` explicitly |

## Architecture

`eval_utils/paired_tests.py` already implements the correct tests — D1 is a call-site swap, not new stats
code (DRY). D6-D9 are one-line-to-one-function changes with outsized correctness impact.

## Related Code Files

Modify: the files above. Create: `tests/test_trt_runner_binding.py` (mock engine, CPU-runnable),
`tests/test_aggregate_fps.py`, `tests/test_val_stratification.py`.

## Implementation Steps

1. Collect the background agents' outputs; **verify their diffs** rather than re-writing them.
2. D1: swap every call site; mark any previously printed number superseded in the sheet.
3. D2: iterate engine I/O tensor names; add a mock-engine test that **fails against the pre-fix code**. **G3.**
4. D3: separate `decisions_completed` from `frames_ingested`; emit both.
5. D4: wrap the full decision path; parameterise crop count; default p95.
6. D5: `--exclude_clips`; verify the excluded set is absent from every fold.
7. D6: stratify val on (label, view); test that both marginals are preserved.
8. D7: `use_cw=True`; assert the computed weights equal 0.403 / 1.597 on the train split.
9. D8: generalise to 8 classes; rebuild whole-frame features for 691 (~10 min) — **unblocks the S2 gate**.
10. D9: grep every `evaluate_suite` / cost-matrix call site; pass `unsafe_classes=[1]`.

## Todo List

- [x] **D1 LANDED** (`laneb-2026-09-16-paired-significance-fix.md`; 6 new paired tests pass, pre-existing
      60-test eval suite still passes). **Neither previously-"significant" result survives as confirmed** —
      follow-ups below.
- [ ] D1-followup-a: `README.md` still quotes a raw two-proportion-z line (56/64 vs 57/64) → **lane D, phase 09**
- [ ] D1-followup-b: `run_ablation_significance.py` has a separate drifting bound implementation → returned
      to the same agent; verify it lands
- [ ] **D2 landed + unit test (G3, due 18/9)**
- [ ] D3, D4 landed
- [ ] D5 landed
- [ ] **D6 (label,view) stratification landed**
- [ ] **D7 class weights ON**
- [ ] **D8 whole-frame features rebuilt for 691**
- [ ] D9 `unsafe_classes=[1]` at every call site
- [ ] Full test suite green

## Success Criteria

G3 passes and fails against pre-fix code. `grep two_proportion_z` clean on paired call sites. S2 features
exist for 691 clips. Class weights verified numerically.

## Risk Assessment

R9 — if D2 misses 18/9, phase 05 falls back to benchmarking the 4-class P1 engine only, leaving the binary
head unmeasured on device. That is the worst schedule failure available; D2 outranks every other defect.
D6-D8 are gates on the *validity* of phase 06, so they must land before Tier 2 screening.

## Security Considerations

None; all changes are local and test-covered.

## Next Steps

D2 → phases 04/05. D1/D5/D9 → phase 06 statistics. D6/D7/D8 → phase 06 Tier 1-2.
