# Phase 04 — Evaluation Rigor: Paper Replication + Multi-Lens + Robustness

**Priority:** P1 · **Weeks:** W9–W11 · **Status:** ☐
**Context:** [researcher-annotation-eval-tooling.md](reports/researcher-annotation-eval-tooling.md), `eval_utils/` (phase-00)

## Overview
Evaluate the best phase-03 model the way the source paper does **and** through a stronger multi-lens
battery, so the thesis's accuracy claim is unimpeachable. Everything runs through the blind protocol;
no metric is chosen on the test set.

## A. Match the paper's four headline metrics
- **Macro classification accuracy (4-class)** = mean per-class recall. Paper full pipeline **95.81 % (62/64)**; **no-bbox baseline 89.06 % top-1 / 85.78 % macro** = the number to beat.
- **Action-recognition time / clip** — paper **0.14 s**. `time.perf_counter()`, 10-clip warmup, all 64, report p50/p95/p99; pin CPU (`taskset -c 0-3`).
- **Reduced avg video duration** — paper **1.87 s**. Measure via the YOLO26 frame-gate ([run_yolo_framegate.py](../../run_yolo_framegate.py) analogue): avg clip duration with vs without gating.
- Present as a like-for-like table vs paper Table 6/7 with our blind number + CI beside each.

## B. Multi-lens classification metrics
- top-1, macro-F1, balanced accuracy, per-class precision/recall, Cohen's κ, Matthews MCC, confusion matrix.
- All with **BCa bootstrap 5,000** CIs (`eval_utils/bootstrap-ci.py`).

## C. Safety metrics (most important for the domain)
- Per-class **FNR / miss-rate**; cost-sensitive score penalizing missed-unsafe ×10 vs false-alarm.
- **ROC-AUC / PR-AUC** + threshold sweep → pick operating point at safety-recall ≥ 0.95; report the resulting false-alarm rate. This is the deployment-relevant operating curve.

## D. Calibration
- **ECE** (`netcal`, 15 bins) + reliability diagram, before/after temperature scaling (fit on val, not test).

## E. Robustness / test-set strengthening
- **k-fold / rotation:** stratified + scene-group k-fold ([run_kfold_eval.py](../../run_kfold_eval.py)); report mean±CI. Reconcile with the single-split blind number (closes the 53/64 vs 71.9 % gap flagged in review).
- **TTA:** horizontal flip + 3–5 temporal crops → average softmax (`eval_utils`). Expect +1–3 clips; report with/without. Avoid noise-based TTA (hurts).
- **Corruption robustness:** `imagecorruptions` + `albumentations` suite — gaussian_noise, shot_noise, jpeg_compression, motion_blur, defocus_blur, brightness, contrast × severities {1,3,5} = ~1,152 cases. Report per-corruption accuracy drop vs clean → a robustness curve the paper lacks.

## F. Statistics
- **McNemar exact** (paired, vs appearance-only control and vs paper no-bbox on the 64 clips) — with the honest "underpowered on N=64" caveat (CIs are the primary evidence).
- **Nadeau–Bengio** corrected t-test for k-fold comparisons.
- Wilson CI for any single proportion.

## Framing rule
Report the **blind** number as the scientific headline; keep any test-selected best-case explicitly
labelled as an upper bound (do not repeat the review's "56/64 presented without qualifier" issue).

## Related files
- Create: `run_full_eval.py` (drives A–F on a given checkpoint → one JSON + figures), `run_corruption_eval.py`.
- Reuse: `eval_utils/`, [run_kfold_eval.py](../../run_kfold_eval.py), [run_blind_eval.py](../../run_blind_eval.py), [run_export_perclip_csv.py](../../run_export_perclip_csv.py).

## Todo
- [ ] Paper 4-metric replication table (macro, time/clip, reduced duration) with CIs
- [ ] Multi-lens metric block + confusion matrix figure
- [ ] Safety block: FNR, cost score, ROC/PR + operating-point selection
- [ ] Calibration: ECE + reliability diagram
- [ ] k-fold (stratified + scene-group); reconcile with blind single-split
- [ ] TTA sweep; corruption suite + robustness curves
- [ ] McNemar + Nadeau–Bengio + Wilson CIs
- [ ] `run_full_eval.py` one-command reproducibility

## Success criteria
- Every headline number carries a BCa CI; blind + k-fold both reported and reconciled.
- Paper's four metrics reproduced/compared like-for-like.
- Safety operating point (recall ≥ 0.95) and its false-alarm cost stated.
- Robustness curve + calibration diagram produced (both new vs the paper).

## Deliverables
- `run_full_eval.py`, `run_corruption_eval.py`, `eval-report.md` + figures (confusion, ROC/PR, reliability, robustness), stats JSON.
