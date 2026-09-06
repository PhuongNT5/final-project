# Phase 0 · D3 — fix 2 test-fit leakage bugs (val-fit / test-apply)

**Date:** 2026-09-03. Additive (only my own experiment scripts edited; no thesis files touched).
Idiom follows the repo's existing honest-fix pattern (`run_ablation_calibration_holdout.py`): fit the
knob on non-test data, apply to test untouched, report both the test-fit reference and the honest number.

## Bug 1 — temperature scaling fit on TEST (`run_ablation_calibration.py`)

**Before:** swept T∈[0.5,2.0] on the 64-clip TEST set; the minimum-ECE point was read off the test
sweep → test-fit.

**Fix:** factored the ensemble into `ens_logits()`; compute the sweep on TRAIN, select
`T* = argmin NLL(train)`, apply that single T* to TEST. Test sweep kept but relabelled DIAGNOSTIC ONLY.
JSON now stores `test_sweep_diagnostic`, `train_sweep`, `T_star_selected_on_train`,
`honest_test_at_T_star`, `test_at_T1`.

**Result (verified, ran end-to-end):**
| quantity | ECE | NLL | top-1 |
|---|---|---|---|
| test-fit min (old, leaked) | **0.110** (T≈0.8) | 0.407 | 55/64 |
| **honest — uncalibrated T=1.0** | **0.114** | 0.420 | 56/64 |
| honest — train-selected T*=0.50, applied to test | 0.146 | 0.445 | 55/64 |

**Finding:** train wants T*=0.50 (sharper) but that makes TEST calibration *worse* (0.146 vs 0.114) —
train and test miscalibration differ. **Temperature scaling does not help here.** The honest calibration
number is **ECE ≈ 0.114 (uncalibrated)**. The previously-reported **ECE 0.101 was a test-minimized
value** (test-fit) and is mildly optimistic → correct any citation to 0.114 uncalibrated.
(Note: this pairwise-cascade ensemble scores 56/64 at T=1.0 — a different pipeline than `logits_yolo26`
54/64; the ECE cell in the scorecard came from THIS script and is what the 0.114 correction applies to.)

## Bug 2 — abstention threshold τ selected on TEST (`run_operating_point.py`)

**Before:** swept τ on the 64-clip TEST softmax and picked the "safety operating point" (highest τ with
firing-recall ≥95%) using TEST labels → test-fit.

**Fix:** added `--select_logits`. τ* is selected on a **leakage-free selection set** and applied
untouched to test. Without `--select_logits` the script prints only the risk-coverage CURVE (a curve is
not a selected point) and a loud WARNING that any τ shown is test-selected / not evidence-grade.

**Selection set (new, leakage-free):** added out-of-fold softmax dumping to `run_kfold_relation.py` →
`sweep_results/relation_tokens/kfold_oof_softmax_<tag>.pt` (516 clips, each predicted by folds that
never saw it; disjoint from the blind 64-clip test). Consume via
`run_operating_point.py --select_logits kfold_oof_softmax_<tag>.pt`.

**Status:** code fixed + verified end-to-end. OOF file generated (`kfold_oof_softmax_yolo26app_.pt`,
516 clips).

**Result (honest — τ selected on OOF, applied to `logits_yolo26.pt` test):**
| | τ | TEST coverage (firing-recall) | acc-on-fired |
|---|---|---|---|
| old test-fit (leaked) | 0.60 | 95.3% | 82.0% |
| **honest (OOF-selected)** | **0.50** | **100.0%** | **81.2%** |

OOF (516) picks the lowest τ meeting ≥95% recall = 0.50, which on test yields full coverage at 81.2%
accuracy — slightly more conservative than the test-fit τ=0.60. ROC-AUC 0.950, PR-AUC 0.869. Per-class
miss-rate at full coverage: WalkwayViol 28.1%, UnauthorInt 9.1%, OpenedPanel 15.4%, CarryOverload 0.0%.

## Net integrity effect
- Calibration: reportable ECE corrected **0.101 → 0.114** (uncalibrated; temp-scaling doesn't help).
- Operating point: no longer emits a test-selected τ; requires an off-test selection set for any
  reported point. Same class of bug as the previously-withdrawn "+3.5pp" is now structurally prevented
  in both scripts.

## Unresolved questions
1. Report the operating point at all, or drop it (the 4-class test is all-unsafe → only firing-recall +
   abstention are measurable; false-alarm needs safe clips)? Deployment value is limited without safe
   clips.
2. Use OOF-softmax (516) or a train-holdout as the τ selection set for the final number? OOF is
   leakage-free and larger — recommended.
