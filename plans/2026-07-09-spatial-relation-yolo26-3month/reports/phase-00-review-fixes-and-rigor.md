# Phase 00 — Review Fixes & Rigor Scaffolding (report)

**Date:** 2026-07-11 · **Constraint:** keep existing results/files untouched; corrections live here, not in the source.

## How to read this
Per instruction, **no existing file was edited**. Two temporary edits made earlier were reverted —
`git diff PROGRESS.md report/chapter-05-edge-deployment.md` is empty (byte-for-byte original).
Everything below is a **recommendation** you apply when ready. The only NEW files added this phase:
- `eval_utils/` — additive rigor toolkit (does not touch any existing script).
- this report.

---

## A. Documentation errata (recommended — NOT applied)

| # | File:line | Current (as-is) | Recommended | Why | Sev |
|---|-----------|-----------------|-------------|-----|-----|
| A1 | [PROGRESS.md:96](../../../PROGRESS.md) | `Inference latency (e2e p50)` = **28.8 ms** (FP32) | `26.8 ms p50 / 28.8 ms p95` | Factual mislabel: 28.8 ms is **p95**; `benchmarks.json` p50 = 26.796 ms. README:40 & Ch5 already correct. | High |
| A2 | [PROGRESS.md:104](../../../PROGRESS.md) | Macro row: **Ours** (Δ = **+3.50 pp**) | mark **withdrawn** (§4.7.11); blind = parity 85.4 % vs 85.78 % | Withdrawn figure presented as a live advantage. | High |
| A3 | [PROGRESS.md:11,79](../../../PROGRESS.md) | "**+3.50 pp on macro**" (headline + contribution 2) | annotate as test-selected / withdrawn | Same withdrawn claim; README/Ch4/Ch6 already withdraw it. | Med |
| A4 | [chapter-05:359](../../../report/chapter-05-edge-deployment.md) | FP16 per-stage `3.95 / 2.05 / 5.70 ms` (sum 11.70 > 8.62 e2e) | `2.76 / 1.41 / 4.44 ms` | Stages sum above e2e; artifact `benchmarks-jetson-fp16.json` p50 = 2.76/1.41/4.44 (sum 8.61 ≈ e2e). | High |
| A5 | [chapter-05:492](../../../report/chapter-05-edge-deployment.md) | `backbone TRT engine (FP16, 18.82 MB, 3.95 ms)` | drop `18.82 MB` (no artifact) or commit the real `.engine` size; latency → `2.76 ms p50` | Engine size unsupported by any committed file; 3.95 ms is the wrong per-stage number. | Med |
| A6 | [chapter-01:84,135,137](../../../report/chapter-01-introduction.md) vs [chapter-04:672](../../../report/chapter-04-experiments-and-results.md) | Ch1 calls it "**Transformer temporal-reasoning head**"; Ch4:672 renames it "*multi-stream aggregation head*" | pick one term repo-wide (recommend "multi-stream aggregation head", per §4.7.8 finding that motion streams carry no signal) | Terminology contradiction between chapters. | Med |
| A7 | [chapter-02:162-164](../../../report/chapter-02-related-work.md) | "Chapter 5 **projects** … to Jetson … **deferred to future work**" | rewrite: Ch5 reports **measured** Jetson numbers | Stale sentence from a pre-measurement draft; contradicts Ch5 §5.6 & Ch6 Finding 1. | High |
| A8 | [README.md:122](../../../README.md), [PROGRESS.md:121](../../../PROGRESS.md) | reference `GAP_ANALYSIS.md`, `BASELINE.md`, and a `docs/` tree | create the files, or remove the dangling references | Referenced artifacts do not exist. | Low |

> None of the accuracy/Jetson/model-size headline numbers themselves are wrong — the full-audit found them
> internally consistent and artifact-backed. A1–A8 are the only doc defects.

---

## B. Code-patch recommendations (NOT applied — diffs for you to apply)

**B1 — `run_ablation_significance.py`: derive our numbers from computed values (currently hardcoded `56`, `89.28`).**
Paper baselines (55/62/85.78/95.81) stay literal; only *our* numbers should come from the model, else z-tests/JSON go stale if the model changes.
```python
# after line 166 (c_ens computed):
macro_pt = round(macro_acc(pred_ens, y), 2)
# line 179-180 prints:
print(f"  top-1 :  point={100*c_ens/n:.2f}  bootstrap mean={top1_mu:.2f} ...")
print(f"  macro :  point={macro_pt:.2f}  bootstrap mean={macro_mu:.2f} ...")
# line 191/195/209/210: two_prop_z(c_ens, n, 55, 64) / (c_ens, n, 62, 64);
#                       mcnemar_marginal_bounds(c_ens, 62, n) / (c_ens, 55, n)
# line 199/203: macro_z_simulated(macro_pt, 85.78, ...) / (macro_pt, 95.81, ...)
# out dict (232-245): ours_k=c_ens, ours_n=n, ours_macro=macro_pt, delta_pp computed
```

**B2 — `run_pairwise_contrastive.py:48` `set_seed`: seed MPS (M4 runs MPS; results not reproducible without it).**
```python
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        torch.mps.manual_seed(s)          # <-- add
    torch.backends.cudnn.deterministic = True
```
(Also imported by `run_blind_eval.py`, so the fix propagates to the blind protocol.)

**B3 — `run_pairwise_contrastive.py:534` `final_macro`: report the ensemble macro, not the max over individual heads.**
```python
# current (inflates: picks best individual head's macro):
final_macro = max(ens_macro, max(r['macro'] for r in all_results.values()))
# recommended (the ensemble is the reported config):
final_macro = ens_macro          # keep best-individual as a SEPARATE reported line
```

**B4 — `run_kfold_eval.py:59,128`: macro is computed as macro-**F1**, but the thesis metric is macro-**accuracy** (mean per-class recall). They differ unless precision==recall.**
```python
# line 33: from sklearn.metrics import confusion_matrix          # drop f1_score (now unused)
# add helper:
def macro_acc(y, pred, nc=4):
    r = [(pred[y==c]==c).mean() for c in range(nc) if (y==c).sum()]
    return 100*float(np.mean(r)) if r else 0.0
# line 59:  macro.append(macro_acc(y[idx], pred[idx]))
# line 128: per_seed_macro.append(macro_acc(ynp, tf_pred))
```
(Equivalently, `import eval_utils; 100*macro_accuracy(y, pred)`.)

---

## C. New additive tooling — `eval_utils/`

Reusable, dependency-light (numpy/scipy/sklearn; **no** statsmodels/netcal needed — McNemar-exact and ECE implemented directly). Every future experiment imports these so numbers are computed identically.

| Module | Provides |
|--------|----------|
| `metrics.py` | `top1`, `macro_accuracy` (mean per-class recall), `per_class_recall`, `per_class_fnr`, `confusion` |
| `bootstrap_ci.py` | `bootstrap_ci` — 5000-resample **BCa** (bias+skew corrected) or percentile |
| `paired_tests.py` | `mcnemar_exact` (+underpowered flag), `two_proportion_z`, `wilson_ci`, `nadeau_bengio` |
| `calibration.py` | `ece` (15-bin) + `reliability_bins` |
| `safety_metrics.py` | `per_class_miss_rate`, `cost_sensitive_score`, `roc_pr_auc`, `threshold_for_recall` |
| `verify_against_csv.py` | Phase-00 success gate (below) |

**Naming note:** modules are snake_case (kebab-case is not importable in Python) — an intentional, documented deviation from the repo file-naming guideline for `eval_utils/` only.

### Verification (success gate) — PASS
`.venv/bin/python -m eval_utils.verify_against_csv` on `deployment/artifacts/per-clip-logits-test64.csv`:
```
top-1 = 56/64 = 87.50%   macro (mean per-class recall) = 89.28%
  WalkwayViol    n=32  recall= 84.38%  FNR= 15.62%
  UnauthorInt    n=11  recall= 72.73%  FNR= 27.27%
  OpenedPanel    n=13  recall=100.00%  FNR=  0.00%
  CarryOverload  n= 8  recall=100.00%  FNR=  0.00%
  top-1 BCa 95% CI = [75.0, 92.2]   macro BCa 95% CI = [79.6, 95.6]
  ECE(15 bins) = 0.1364
  per-class miss-rate = [15.6, 27.3, 0.0, 0.0]  cost(miss x10) = 80
confusion matrix (rows=true, cols=pred):
[[27  4  1  0]
 [ 3  8  0  0]
 [ 0  0 13  0]
 [ 0  0  0  8]]
PASS: eval_utils reproduces 56/64 top-1 and 89.28% macro from the committed artifact.
```
Confirms: headline reproduces exactly; the entire error budget is the **WalkwayViol↔UnauthorInt** block
(4+3 = 7 of the 8 errors) — the pair Phase-03 targets. New finding: **ECE = 0.136** → the deployed
ensemble is over-confident; temperature scaling on a val slice is worth adding (phase-04).

> BCa CIs here differ slightly from the committed percentile CIs ([79.7,95.3]/[81.3,96.3]) — expected:
> BCa vs percentile method + different resample seed. Both are valid; BCa is the recommended default going forward.

---

## D. Baseline rigor reporting template (the format every future model fills)

Filled below for the **current deployed frozen ensemble** (the reference row all Phase-03 models compare against). This is the template phases 02–05 reuse — never report a bare accuracy again.

| Model | Protocol | top-1 | top-1 CI (BCa) | macro | macro CI | WalkwayViol R | UnauthorInt R | OpenedPanel R | CarryOverload R | ECE | cost(×10) |
|-------|----------|-------|----------------|-------|----------|---|---|---|---|-----|-----------|
| Frozen ens. (direct4_s314 ⊕ cascade_s7) | **test-selected** (deployed) | 56/64 = 87.5 % | [75.0, 92.2] | 89.28 % | [79.6, 95.6] | 84.4 % | 72.7 % | 100 % | 100 % | 0.136 | 80 |
| _same_ | **blind held-out** (§4.7.11) | ≈52/64 | _fill_ | ≈84 % | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| _same_ | **k-fold scene-robust** | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |

Every model reports **all three protocol rows** + the McNemar-exact paired test vs this reference.

---

## E. Protocol reconciliation (open, feeds phase-04)
Review found two "honest" numbers that disagree: **blind single-split 53/64 (82.8 %)** vs reframe-memo
**k-fold scene-robust 71.9 %**. Recommendation: from now on report **both** for every model (template §D),
and treat the ~10 pp gap as *single-split optimism vs scene-generalization* — quantify and explain it in
phase-04, do not pick the flattering one. `eval_utils.nadeau_bengio` is provided for the k-fold comparison.

---

## F. Status & open items
- ✅ `eval_utils/` built + verified (PASS, reproduces 56/64 / 89.28 %).
- ✅ All earlier source edits reverted; originals byte-for-byte intact.
- ☐ Apply A1–A8 doc errata + B1–B4 code patches — **your call** (not applied, per instruction).
- ☐ Advisor sign-off on the 4 questions in [plan.md](../plan.md) (WalkwayViolation definition; zones at inference; honesty-upgrade headline; 3k vs ≥4k budget).
- ☐ Fill the blind + k-fold rows of the §D template (needs a training run; scheduled in phase-03/04).
