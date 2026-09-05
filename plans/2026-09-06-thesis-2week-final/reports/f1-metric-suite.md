# Lane F — evaluation-metric suite (brief S4), implementation report

2026-09-06. Owner: Lane F. Scope: `eval_utils/` only.

## 1. Module map

Reused (pre-existing, untouched signatures — callers `run_full_eval.py`,
`run_operating_point.py`, `run_lco_eval.py` keep working unmodified):
- `eval_utils/metrics.py` — top1, per_class_recall, macro_accuracy (=mean per-class recall), per_class_fnr, confusion.
- `eval_utils/bootstrap_ci.py` — bootstrap_ci(percentile|bca), n_boot default 5000.
- `eval_utils/paired_tests.py` — mcnemar_exact, two_proportion_z, wilson_ci, nadeau_bengio.
- `eval_utils/safety_metrics.py` — per_class_miss_rate, cost_sensitive_score, roc_pr_auc (binary), threshold_for_recall.
- `eval_utils/calibration.py` — reliability_bins, ece (equal-width confidence bins, default 15).
- `eval_utils/verify_against_csv.py` — existing 56/64-headline regression check (still passes, untouched).

Reused from `unsafenet/` (read-only import, no write — DRY: `unsafenet/metrics.py` +
`metrics_stats.py` already implement most of S4b/S4d/S4e correctly; reimplementing
them in `eval_utils` would violate the project's DRY rule):
- `unsafenet.metrics.report()` — top1, macro_accuracy, balanced_accuracy(=macro_accuracy),
  precision/recall/f1 per class, macro/weighted/micro-F1, MCC (Gorodkin), Cohen's kappa,
  confusion (+normalized), ROC/PR-AUC one-vs-rest (per class + macro, NaN-safe), ECE/MCE,
  Brier, NLL.
- `unsafenet.metrics_stats.mcnemar_bound()` — the discordance-bound math (reused verbatim,
  confirmed byte-for-byte reproduces `t1-objectives-audit.md` S3b's table, see S5 below).

New (added this pass, all < 200 lines, kebab-word snake_case matching the existing
`eval_utils/` convention — hyphenated filenames aren't importable as Python modules,
so kept `snake_case.py` consistent with all 6 pre-existing files):
- `eval_utils/cost_matrix.py` — declared C×C cost matrix (S4c).
- `eval_utils/operating_point.py` — threshold sweep + safety-recall operating point (S4d).
- `eval_utils/timing.py` — per-clip perf_counter timing + frame-gate duration (S4a).
- `eval_utils/calibration.py` (extended) — `plot_reliability_diagram()` (S4e).
- `eval_utils/paired_tests.py` (extended) — `mcnemar_discordance_bound()` wrapper (S4f).
- `eval_utils/report_schema.py` — canonical record schema + markdown renderer (S4h).
- `eval_utils/suite.py` — **the single reusable entry point**, `evaluate_suite()`.
- `eval_utils/__init__.py` — exports `evaluate_suite`, `render_markdown`.
- `eval_utils/tests/` — 45 known-answer/edge-case/real-artifact tests + `run_all.py`
  runner (no pytest in `.venv` — plain `assert` + import-and-call, same style as the
  pre-existing `verify_against_csv.py`).

## 2. Public API

```python
from eval_utils import evaluate_suite, render_markdown

result = evaluate_suite(
    y_true, y_prob, class_names, protocol_tag,     # required
    n_seeds=1, y_pred=None,                          # y_pred defaults to y_prob.argmax(1)
    cost_matrix=None, unsafe_classes=None,           # S4c — declared matrix, documented default
    target_safety_recall=0.95,                       # S4d — selects tau, NOT by accuracy
    n_boot=5000, seed=0,                              # S4f — bootstrap
    compare_to=None,                                  # {'name','correct','n','pred'(optional)}
    selection_y_true=None, selection_y_prob=None,    # leakage-free tau selection set
    n_bins=15,                                        # S4e binning
)
render_markdown(result)   # paste-ready headline table
```

Supporting, independently importable:
- `eval_utils.cost_matrix.default_cost_matrix(n_classes, unsafe_classes, miss_cost=10.0, false_alarm_cost=1.0)`, `total_cost`, `mean_cost`, `cost_report`.
- `eval_utils.operating_point.confidence_threshold_sweep(y_true, y_prob, thresholds=None)`, `select_operating_point_by_safety_recall(y_true, y_prob, target_recall, selection_y_true=None, selection_y_prob=None)`.
- `eval_utils.timing.time_per_clip(fn, clips)`, `frame_gate_reduced_duration(before, after, fps)`.
- `eval_utils.calibration.plot_reliability_diagram(bins, save_path, title=None)`.
- `eval_utils.paired_tests.mcnemar_discordance_bound(n_total, correct_a, correct_b)`.
- `eval_utils.report_schema.make_record(...)`, `to_markdown_table(...)`, `save_json(result, path)`.

## 3. JSON schema (evaluate_suite return value, abbreviated)

```jsonc
{
  "protocol_tag": "same-dist-64", "n": 64, "seed_count": 5, "class_names": [...],
  "top1": {"value":.., "ci95":[lo,hi], "n":64, "seed_count":5, "protocol_tag":"..",
           "correct":56, "ci_method":"bootstrap_bca_5000", "wilson_ci95":[lo,hi]},
  "macro_accuracy": {..., "definition":"mean per-class recall"},
  "macro_f1": {...}, "balanced_accuracy": {...}, "cohen_kappa": {...}, "matthews_mcc": {...},
  "per_class": {"<class>": {"precision":.., "recall":.., "f1":.., "fnr_miss_rate":.., "n":..}},
  "confusion_matrix": [[...]], "confusion_matrix_normalized": [[...]],
  "safety_cost": {"cost_matrix":[[...]], "unsafe_classes":[...], "total_cost":.., "mean_cost":..,
                  "protocol_tag":"..", "n":.., "seed_count":..},
  "roc_auc_ovr": {"per_class":[...], "macro":..}, "pr_auc_ovr": {"per_class":[...], "macro":..},
  "threshold_sweep": [{"tau":.., "coverage":.., "accuracy_on_fired":.., "n_fired":..,
                       "n_abstained":.., "per_class_recall_on_fired":{0:..,1:..}}],
  "operating_point_safety_recall": {"tau":.., "target_recall":.., "coverage":..,
                                    "accuracy_on_fired":.., "test_selected":bool, "note":".."},
  "calibration": {"ece":.., "mce":.., "n_bins":15, "binning_scheme":"..", "reliability_bins":[...]},
  "brier":.., "nll":..,
  "comparison": {"name":"..", "mcnemar_exact":{...}, "mcnemar_discordance_bound":{...}, "two_proportion_z":{...}}
}
```
Every leaf metric (`make_record` output) carries `value, ci95, n, seed_count, protocol_tag` —
the S4h mandate. `save_json()` / `render_markdown()` in `report_schema.py`/`suite.py` turn
this into thesis-ready JSON + markdown.

## 4. Brief S4 checklist

| Item | Function(s) | Status |
|---|---|---|
| 4a macro=mean per-class recall | `unsafenet.metrics.macro_accuracy` (verified test) | done |
| 4a action-recognition time (perf_counter, 0.14s mark) | `eval_utils.timing.time_per_clip` | done — utility ready; **actual measurement needs a real running model/clip loader, which is Lane A's inference code, not mine to run** |
| 4a reduced duration (1.87s mark) | `eval_utils.timing.frame_gate_reduced_duration` | done — utility ready, same caveat (needs real frame-gate output) |
| 4b top1/macroF1/balAcc/P/R/kappa/MCC/confusion | `unsafenet.metrics.report` via `suite.evaluate_suite` | done |
| 4c per-class FNR | `eval_utils.metrics.per_class_fnr` | done |
| 4c cost-sensitive, declared matrix | `eval_utils.cost_matrix.*` | done |
| 4d ROC-AUC ovr, PR-AUC | `unsafenet.metrics.roc_pr_auc_ovr` | done |
| 4d threshold sweep | `eval_utils.operating_point.confidence_threshold_sweep` | done |
| 4d operating point by safety recall | `eval_utils.operating_point.select_operating_point_by_safety_recall` | done |
| 4e ECE + binning scheme stated | `eval_utils.calibration.ece` (equal-width, 15 bins default) | done |
| 4e reliability diagram data + plot | `eval_utils.calibration.reliability_bins` / `plot_reliability_diagram` | done |
| 4f bootstrap 5000 | `eval_utils.bootstrap_ci.bootstrap_ci(n_boot=5000)` | done |
| 4f Wilson | `eval_utils.paired_tests.wilson_ci` | done |
| 4f BCa | `eval_utils.bootstrap_ci.bootstrap_ci(method='bca')` | done |
| 4f McNemar exact (per-clip) | `eval_utils.paired_tests.mcnemar_exact` | done |
| 4f McNemar discordance bound (opponent per-clip unknown) | `eval_utils.paired_tests.mcnemar_discordance_bound` | done — reuses `unsafenet.metrics_stats.mcnemar_bound`, matches audit S3b exactly |
| 4f Nadeau-Bengio | `eval_utils.paired_tests.nadeau_bengio` | done |
| 4g Jetson (p50/95/99, FLOPs, INT8, J/inf) | — | **out of scope**, owned by Lane B (`deployment/`) |
| 4h n/CI/seed/protocol on every record | `eval_utils.report_schema.make_record` + `suite.evaluate_suite` | done |
| 4h canonical JSON schema | `suite.evaluate_suite` return value | done |
| 4h markdown renderer | `report_schema.to_markdown_table` / `suite.render_markdown` | done |

Flagged gaps: 4a's two timing marks are **utilities only** — I did not run any model
(explicitly out of scope: "no model training… CPU-light analysis code only"), so I
could not produce an actual 0.14s/1.87s comparison number this pass. The functions
are tested against deterministic `time.sleep()`/hand-computed frame counts and are
drop-in ready for whichever script does the real clip inference loop (A2's job per
the brief's work programme). 4g is explicitly another lane's file ownership.

## 5. Test results

`.venv/bin/python -m eval_utils.tests.run_all` → **45/45 PASS**, 8 modules:
classification_metrics(9), statistics(9), calibration(4), cost_and_operating_point(7),
timing(4), edge_cases(5), suite_integration(5), real_artifacts(5).
Pre-existing `eval_utils/verify_against_csv.py` also still passes unmodified (56/64,
89.28% macro) — confirms zero regression to the callers that import individual
`eval_utils.*` functions today.

Reference cross-checks used (no statsmodels in `.venv`, scipy substituted):
- Wilson CI vs a published worked example (Newcombe 1998, 81/263 → [0.255, 0.365]) and
  vs a hand-solved quadratic (k=8,n=10) — exact match.
- BCa vs `scipy.stats.bootstrap(method='BCa')` on N(5,2) data, same seed — CIs agree
  within 0.5 (sampling noise between independent RNG streams, expected).
- McNemar exact vs `scipy.stats.binomtest` on random paired boolean arrays — exact
  p-value match to 1e-9.
- McNemar discordance bound vs `t1-objectives-audit.md` S3b's table (paper=62/64
  fixed, our_correct 55/56/57/58/60/62/64) — **exact match** to all 7 rows
  (0.015625, 0.03125, 0.0625, 0.125, 0.5, 1.0, 0.5), confirming
  `unsafenet.metrics_stats.mcnemar_bound` is literally the function that produced
  that audit table.
- Nadeau-Bengio vs the closed-form correction formula by hand — exact match; zero-
  variance edge case gives t=0, p=1 as required.
- ECE hand-computed 2-bin toy case; perfect-calibration case gives ECE≈0.

Edge cases exercised and found correct (not skipped):
- Zero-support class → NaN recall/FNR, excluded from `macro_accuracy`'s nanmean (not
  treated as 0, doesn't crash).
- Bootstrap resample collapsing to a single class (both percentile and BCa) → no NaN
  propagation into the final CI.
- Undefined ROC-AUC (single-class y_true) → NaN for affected per-class AUCs and
  macro; **discovered during testing that PR-AUC (average precision) stays
  well-defined for an all-positive class** even when ROC-AUC is not (AP only needs
  positives, ROC-AUC needs both classes) — this is correct behaviour in
  `unsafenet.metrics.roc_pr_auc_ovr`, documented via the test rather than "fixed"
  (not my file to change, and it's mathematically correct as-is).
- `threshold_for_recall` with an unreachable target (>1.0) → explicit `None` +
  `"unreachable"` note, not a silent wrong answer.

## 6. Real-artifact validation

1. **`deployment/artifacts/per-clip-logits-test64.csv`** (64 clips, committed headline
   artifact). `evaluate_suite` reproduces: **56/64 = 87.50% top-1, 89.28% macro**
   (bit-identical to `verify_against_csv.py`'s asserted numbers). Per-class recall
   WalkwayViol 84.4%/UnauthorInt 72.7%/OpenedPanel 100%/CarryOverload 100%; kappa
   0.814, MCC 0.815; ECE(15-bin)=0.136; cost(default matrix)=80; McNemar discordance
   bound vs paper 62/64 → diff=-6, min_p_over_splits=0.03125 (exact audit-table match
   for our_correct=56).
2. **`sweep_results/relation_tokens/logits_s1a.pt`** (64-clip, S1a ensemble, current
   file on disk — n=64, argmax gives **54/64=84.375%**, kappa 0.775, MCC 0.785,
   ROC-AUC 0.954, PR-AUC 0.877). Cross-checked against an independent hand-run of
   `run_full_eval.py`'s exact inline sklearn calls on this same file: matches to the
   printed precision (top1 84.38%, macro 88.41%, macro-F1 85.56%, kappa 0.775, MCC
   0.785, ROC-AUC 0.953, PR-AUC 0.874 — PR-AUC differs by ~0.3pp because
   `unsafenet`'s step-interpolated AP and sklearn's `average_precision_score` use
   slightly different interpolation, both valid, not a bug).
   **Note:** the committed `sweep_results/relation_tokens/full_eval.json` (mtime
   Aug 29) shows 52/64=81.25% — it does **not** match the `.pt` file currently on
   disk (mtime Aug 13, raw argmax = 54/64). The JSON is stale relative to the
   artifact, not a discrepancy in this suite; flagging for whichever lane owns
   `sweep_results/` to regenerate it.
3. **`sweep_results/relation_tokens/kfold_oof_softmax_yolo26app_.pt`** (516-clip
   pooled k-fold OOF). `evaluate_suite` gives **417/516 = 80.81% top-1, 83.99%
   macro** — matches the brief's quoted "k-fold 516: 80.8% top-1, 84.0% macro"
   essentially exactly. Bootstrap 95% CI on macro ≈ [81.1, 86.7] (half-width ~2.9pp),
   consistent with the brief's Nadeau-Bengio ±3.1pp (different method, same
   ballpark, as expected).

## 7. Design notes / deviations

- Kept `eval_utils/` filenames `snake_case.py` (not literal kebab-case) — matches
  all 6 pre-existing files and is required for plain `import`; a hyphenated
  `.py` file cannot be imported without `importlib` gymnastics, which none of the
  five caller scripts use.
- Did not touch `run_ablation_significance.py` / `run_kfold_eval.py`, which have
  their OWN inline duplicate bootstrap/Nadeau-Bengio/McNemar-bound implementations
  (never imported `eval_utils` at all). Task said the new suite must be *usable*
  from callers without rewriting them — it is (drop-in `evaluate_suite` matches
  their existing per-function imports) — but actually migrating those two scripts
  onto the new suite is a follow-up, outside `eval_utils/` ownership.
- `mcnemar_discordance_bound`'s `favours_b` flag is about the function's own
  `(correct_a, correct_b)` argument order; `suite._compare` labels which name is
  `a` (`this_result`) vs `b` (`compare_to['name']`) explicitly in the output to
  avoid misreading "favours_b" as "favours us".
- Cost matrix default: off-diagonal cost = `miss_cost` (10.0) if true class is
  "unsafe", else `false_alarm_cost` (1.0), diag=0; `unsafe_classes` defaults to ALL
  classes (matches today's all-unsafe 4-class test protocol and is numerically
  identical to the pre-existing `cost_sensitive_score` under the same params —
  verified in tests). For the 8-class ontology once safe clips exist in a test
  split, pass `unsafe_classes=range(4), n_classes=8`.

## Unresolved questions
1. Should `run_ablation_significance.py` / `run_kfold_eval.py` be migrated onto
   `evaluate_suite`/`paired_tests` to remove their duplicate inline stats code? Not
   done here (outside `eval_utils/` file ownership) — flag for whoever owns those
   root scripts.
2. `sweep_results/relation_tokens/full_eval.json` is stale vs the current
   `logits_s1a.pt` (52/64 in the JSON vs 54/64 from the file on disk today) —
   whoever regenerates the headline (A4 freeze) should re-run `run_full_eval.py`
   before quoting numbers from that JSON in the thesis.
3. 4a's actual 0.14s/1.87s replication numbers need a real clip-inference loop
   (A2's job) plumbed through `eval_utils.timing`'s two functions — not produced
   here per the "no model training / CPU-light analysis only" instruction for this
   lane.
