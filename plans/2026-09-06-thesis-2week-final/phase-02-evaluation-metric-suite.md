---
title: "Phase 02 — Lane F: Evaluation Metric Suite (§4 contract as code)"
status: pending
---

# Phase 02 — Evaluation Metric Suite (Lane F)

**File ownership (exclusive):** `eval_utils/`. Concurrent with all lanes; front-loaded — must land
before A4's day-6 test touch (phase-01 step 15 calls this code).

## Context links
- Binding brief: `plans/2026-09-06-thesis-2week-final/brief.md` §4 (full contract, a-h), §5
- `eval_utils/{metrics.py,bootstrap_ci.py,calibration.py,paired_tests.py,safety_metrics.py,
  verify_against_csv.py}` — current partial implementation
- `unsafenet/metrics.py` + `unsafenet/metrics_stats.py` — a SECOND, more complete implementation of
  overlapping metrics, currently used only by the `unsafenet/` P1/P2 pipeline
- `run_kfold_relation.py` lines computing sklearn `f1_score/cohen_kappa_score/matthews_corrcoef/
  precision_score/recall_score` inline — a THIRD ad-hoc implementation

## Overview
Priority P1 (blocking). Status pending. The brief mandates one binding evaluation contract (§4a-h)
that every headline number must come from. Today that contract is split across three places with
duplicated, non-identical logic. This phase makes `eval_utils/` the single source of truth,
reusing the already-correct math in `unsafenet/metrics.py`/`metrics_stats.py` (production-tested on
P1/P2) rather than re-deriving it, and folding in the sklearn calls currently inlined in
`run_kfold_relation.py`. No lane needs the OTHER metric files after this — `unsafenet/metrics.py`
callers get migrated or the module stays temporarily for P1/P2-internal calls only (do not delete
without a compat check, `unsafenet/` may still be imported by round-2 scripts).

## Key insights
- `eval_utils/metrics.py` today has only: `top1, per_class_recall, macro_accuracy, per_class_fnr,
  confusion`. Missing vs brief §4b: **macro-F1, balanced accuracy (distinct name, same number as
  macro_accuracy — document this explicitly, do not silently duplicate), per-class precision, Cohen
  kappa, MCC**.
- `unsafenet/metrics.py::report()` ALREADY computes the full superset in one call: top1 (+CP/Wilson
  CI), macro_accuracy (+bootstrap CI), balanced_accuracy (aliased, documented "same number, kept as
  separate name because both terms appear in the literature"), precision/recall/f1 per class,
  macro/weighted/micro F1, MCC (Gorodkin 2004, multiclass, no sklearn), Cohen kappa, confusion
  (+normalized), ROC/PR-AUC one-vs-rest (numpy-only), ECE/MCE, Brier, NLL, per-class error,
  per-view error. This is the correct target shape for `eval_utils.metrics.report()`.
- `eval_utils/safety_metrics.py` (per_class_miss_rate, cost_sensitive_score, roc_pr_auc,
  threshold_for_recall) and `eval_utils/paired_tests.py` (mcnemar_exact, two_proportion_z,
  wilson_ci, nadeau_bengio) are NOT duplicated in `unsafenet/` — keep these as-is, just call them
  from the same consolidated module.
- `eval_utils/bootstrap_ci.py` (BCa, 5000 default) and `unsafenet/metrics_stats.py::bootstrap_ci`
  (percentile only) differ in method — brief §4f requires **Wilson/BCa**, so `eval_utils`'s BCa
  version is the one to keep; percentile-only stays internal to `unsafenet/` if still used there.
- `eval_utils/verify_against_csv.py` is the existing regression gate (reproduces 56/64 top-1 /
  89.28% macro from `deployment/artifacts/per-clip-logits-test64.csv`) — any refactor MUST keep this
  script passing unmodified in its assertions (only import paths may change).

## Requirements
- Functional: one `eval_utils` call produces every number in brief §4a-h for a given
  (y_true, probs) pair, with n/CI/seed-count/protocol-tag attached (§4h.5).
- Non-functional: no new dependency beyond numpy/scipy/sklearn (sklearn already used by
  `run_kfold_relation.py`, acceptable); each file stays under ~200 lines per project convention
  (split like `unsafenet/metrics.py` + `metrics_stats.py` does).

## Architecture
`eval_utils/metrics.py` gains: `macro_f1`, `weighted_f1`, `precision_recall_f1` (per-class),
`cohen_kappa`, `matthews_corrcoef`, `balanced_accuracy` (= alias of `macro_accuracy`, documented).
Port implementations from `unsafenet/metrics.py` (already correct, tested) rather than reimplement.
`eval_utils/report.py` (new, thin) composes `metrics.py` + `bootstrap_ci.py` + `calibration.py` +
`paired_tests.py` + `safety_metrics.py` into one `full_report(y, probs, protocol_tag, seed_count,
n_seeds, cost_matrix=None)` dict — this is what phase-01's A4 step calls. Keep files small: split
new metrics into `eval_utils/metrics.py` (pure metric math) vs `eval_utils/report.py`
(orchestration + CI wiring), mirroring the existing `metrics.py`/`metrics_stats.py` split pattern.

## Related code files
- Modify: `eval_utils/metrics.py` (add macro_f1/weighted_f1/precision_recall_f1/cohen_kappa/
  matthews_corrcoef/balanced_accuracy)
- Create: `eval_utils/report.py` (full_report orchestration, calls into metrics/bootstrap_ci/
  calibration/paired_tests/safety_metrics)
- Do not modify (already correct, reuse as reference): `unsafenet/metrics.py`,
  `unsafenet/metrics_stats.py`, `eval_utils/bootstrap_ci.py`, `eval_utils/calibration.py`,
  `eval_utils/paired_tests.py`, `eval_utils/safety_metrics.py`
- Must keep passing unchanged in behaviour: `eval_utils/verify_against_csv.py`
- Consumers to update imports (read-only inspection, actual call sites owned by phase-01/phase-03):
  `run_relation_transformer.py`, `run_kfold_relation.py`, `run_lco_eval.py`, `run_operating_point.py`

## Implementation steps (day 1-2, front-loaded)
1. Day 1: spec the `full_report()` signature and required output keys against brief §4a-h line by
   line (checklist below); confirm against `unsafenet/metrics.py::report()`'s existing key set —
   most keys already match 1:1.
2. Day 1: port `macro_f1, weighted_f1, precision_recall_f1, cohen_kappa, matthews_corrcoef` from
   `unsafenet/metrics.py` into `eval_utils/metrics.py`, adapting signature to eval_utils's
   `(y_true, y_pred)` convention (vs unsafenet's `(y, pred, n_cls)`) — keep n_classes as an explicit
   optional arg, default inferred like existing `confusion()`.
3. Day 1: add `balanced_accuracy = macro_accuracy` alias in `eval_utils/metrics.py` with the same
   documenting comment `unsafenet/metrics.py` uses (avoid future "why are there two" confusion).
4. Day 2: write `eval_utils/report.py::full_report(y, probs, protocol_tag, n_seeds, cost_matrix)` —
   composes: top1 (+Wilson CI via `paired_tests.wilson_ci`, +Clopper-Pearson if kept), macro/
   balanced accuracy (+BCa bootstrap via `bootstrap_ci.bootstrap_ci`), per-class P/R, macro-F1,
   kappa, MCC, confusion, FNR/miss-rate + cost-sensitive score (`safety_metrics`), ROC/PR-AUC +
   threshold sweep (`safety_metrics.threshold_for_recall`), ECE + reliability bins (`calibration`),
   McNemar exact (`paired_tests.mcnemar_exact`, needs a second model's predictions — optional arg),
   Nadeau-Bengio (`paired_tests.nadeau_bengio`, needs per-fold diffs — optional arg for k-fold call
   sites). Every field tagged with `protocol_tag` and `n_seeds` per §4h.5.
5. Day 2: run `eval_utils/verify_against_csv.py` — confirm it still reproduces 56/64=87.5%/89.28%
   macro after the refactor (no regression).
6. Day 2: smoke-test `full_report()` against `run_kfold_relation.py`'s existing pooled OOF softmax
   (`sweep_results/relation_tokens/kfold_oof_softmax_<tag>.pt`) — confirm macro-F1/kappa/MCC match
   the inline sklearn numbers already printed by that script (80.8% top-1 / 84.0% macro / F1 83.0% /
   kappa 0.731 / MCC 0.735 from the fair-headtohead scorecard) before phase-01 relies on it.
7. Day 2: hand off to phase-01 — A4 (day 6) imports `eval_utils.report.full_report` directly; do not
   let phase-01 reimplement any metric inline.

## Todo list
- [ ] `eval_utils/metrics.py`: macro_f1, weighted_f1, precision_recall_f1, cohen_kappa,
      matthews_corrcoef, balanced_accuracy added
- [ ] `eval_utils/report.py` created with `full_report()`
- [ ] `eval_utils/verify_against_csv.py` still passes (56/64, 89.28% macro reproduced)
- [ ] `full_report()` smoke-tested against `run_kfold_relation.py`'s stored OOF probabilities,
      numbers match within float tolerance
- [ ] Phase-01 notified `eval_utils.report.full_report` is ready (before day 6)

## Success criteria
- `python -m eval_utils.verify_against_csv` exits 0, prints 56/64 top-1 and 89.28% macro (unchanged).
- `eval_utils/report.py::full_report()` returns a dict containing every metric named in brief
  §4a-4g (top-1, macro-F1, balanced accuracy, per-class P/R, kappa, MCC, confusion, FNR, cost
  score, ROC-AUC, PR-AUC, threshold-selected operating point, ECE, reliability bins, bootstrap CI,
  McNemar p when a comparator given, Nadeau-Bengio p when fold diffs given) — verified by a
  checklist diff against brief §4, not by opinion.
- No metric file exceeds ~200 lines (project convention); `git diff --stat eval_utils/` shows no
  file rewritten from scratch, only additive changes to `metrics.py` + new `report.py`.

## Risk assessment
- **Signature mismatch** between `unsafenet/metrics.py`'s `(y, pred, n_cls)` and `eval_utils`'s
  `(y_true, y_pred)` causes silent n_classes bugs on the 4-class subset — mitigate: always pass
  `n_classes=4` explicitly at P3 call sites, never rely on inference from `y_true.max()`.
- **Two modules drift again** if `unsafenet/metrics.py` is later modified for P1/P2 without mirroring
  `eval_utils` — mitigate: `eval_utils` becomes canonical for ALL new (A4-forward) reporting;
  `unsafenet/metrics.py` is frozen (bugfix-only) since P1/P2 are historical-ladder evidence only.

## Security considerations
No secrets/PII. Pure numeric library code.

## Next steps
- Phase-01 A4 (day 6) is the first and primary caller of `full_report()` for the headline freeze.
- Phase-03 B2 (Jetson accuracy-parity check) should also call `eval_utils.metrics.top1`/
  `macro_accuracy` for consistency, not reimplement.

## Unresolved questions
1. Should `unsafenet/metrics.py` be refactored to import FROM `eval_utils` (true DRY) once
   consolidated, or left frozen as a separate, intentionally-duplicated-but-tested module for
   historical P1/P2 reporting? Recommend frozen (lower risk in a 2-week window), but flagging since
   it leaves a permanent, documented duplication.
2. `eval_utils/verify_against_csv.py`'s reference CSV is P1's headline (56/64) — once phase-01
   freezes a new P3 headline, does a second `verify_against_<new>.py` get created, or is the old one
   kept as a P1-regression check only? Recommend keep old as-is (P1 regression), add new for P3 in
   phase-01's A4 step, not phase-02's scope.
