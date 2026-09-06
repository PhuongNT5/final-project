# Phase 00 — Pre-flight: Review Fixes + Rigor Scaffolding

**Priority:** P0 (unblocks everything) · **Weeks:** W1 · **Status:** ☐

## Overview
Cheap, high-value cleanups from the project review + set up shared rigor utilities the later phases
reuse (stats, calibration, blind-protocol wrappers). Do this first so new experiments are honest and
reproducible from day one.

## Review fixes (from the whole-project review)
Correctness/consistency defects found during review — fix before adding new results.

1. **PROGRESS.md is stale** — remove/flag the withdrawn "+3.5 pp macro win" presented as live
   ([PROGRESS.md](../../PROGRESS.md) lines ~79, ~104); fix p50/p95 mislabel (line ~95: 28.8 ms is p95, not p50).
2. **Ch5 Table 5.5B FP16 per-stage latencies wrong** — table shows 3.95/2.05/5.70 ms (sum 11.7 > 8.62 e2e);
   artifact `deployment/artifacts/benchmarks-jetson-fp16.json` says 2.76/1.41/4.44 ms. Replace with artifact values.
3. **Unsupported TRT engine size 18.82 MB** (Ch5 §5.6.5) — either measure the real `.engine` size and commit it, or drop the figure.
4. **Missing referenced files** — README/PROGRESS reference `docs/`, `GAP_ANALYSIS.md`, `BASELINE.md` that do not exist.
   Create a minimal `docs/` (per CLAUDE.md structure) or remove the dangling references.
5. **Code hygiene** (see review): `run_ablation_significance.py` hardcodes z-test inputs (56, 64, 89.28) →
   derive from computed `c_ens`/macro; `set_seed` never seeds MPS (`torch.mps.manual_seed`); `final_macro`
   takes `max` over individual models (report the *ensemble* macro); k-fold bootstrap uses macro-**F1** —
   switch to macro-**accuracy** (mean per-class recall) to match the thesis metric.
6. **Terminology drift** — Ch1 "temporal-reasoning head" vs Ch4 retraction; "Unsafe-Ne" typo (source paper only).

## Rigor scaffolding (build once, reuse in phases 03–05)
Create `eval_utils/` (kebab-case module files, <200 lines each):
- `bootstrap-ci.py` — BCa 5,000-resample CI (`scipy.stats.bootstrap(method='BCa')`) for top-1, macro, per-class.
- `paired-tests.py` — McNemar exact (`statsmodels ... mcnemar(exact=True)`) + Nadeau–Bengio corrected t-test (`correctipy`) with an explicit "underpowered on N=64" note (n_discordant ≈ 6–8).
- `calibration.py` — ECE (`netcal`, 15 bins) + reliability diagram; optional temperature scaling fit on val.
- `safety-metrics.py` — per-class FNR/miss-rate, cost matrix (miss unsafe ×10), ROC/PR-AUC + threshold sweep to pick operating point at recall ≥ 0.95.
- `blind-eval-wrapper.py` — thin wrapper enforcing the [run_blind_eval.py](../../run_blind_eval.py) protocol for *any* new head (train-carved val slice; test touched once). All phase-03/04 numbers go through this.

## Protocol reconciliation (important)
The review found two "honest" numbers that disagree: blind held-out **53/64 (82.8 %)** vs the reframe
memo's k-fold scene-robust **71.9 %**. Standardize reporting: run **both** the single-split blind
protocol AND stratified + scene-group k-fold ([run_kfold_eval.py](../../run_kfold_eval.py)) for every
model from now on; report them side by side and explain the delta (single-split optimism vs
scene-generalization). This becomes the reporting template for phases 03–04.

## Todo
- [ ] Fix PROGRESS.md withdrawn-claim + p50/p95 mislabel
- [ ] Fix Ch5 Table 5.5B FP16 per-stage + TRT size
- [ ] Resolve dangling `docs/`, `GAP_ANALYSIS.md`, `BASELINE.md` references
- [ ] Patch `run_ablation_significance.py`, `set_seed` (MPS), `final_macro`, k-fold bootstrap metric
- [ ] Build `eval_utils/` (5 modules) + verify against current 56/64 artifact
- [ ] Run blind + k-fold on the existing best checkpoint → establish the reporting template baseline
- [ ] Advisor sign-off on the 4 open questions in [plan.md](plan.md)

## Success criteria
- All reviewed defects fixed; `git grep` finds no live withdrawn figure.
- `eval_utils/` reproduces the committed 56/64 top-1 and 89.28 % macro from `per-clip-logits-test64.csv`.
- Baseline table: current best model reported with blind top-1, k-fold mean±CI, ECE, per-class FNR — the template every later phase fills in.

## Deliverables
- Clean PROGRESS/README/Ch5; `eval_utils/`; a `baseline-rigor-report.md` (the reporting template, filled for the current model).
