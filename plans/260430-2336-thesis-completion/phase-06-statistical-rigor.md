---
priority: must-have
effort: 3 person-days
depends_on: [phase-03 provides seed runs for data-efficiency curve]
branches: none
---

# Phase 06 — Statistical Rigor

## Context Links

- `PROGRESS.md` — best result 56/64 = 89.28 % macro; V3 baseline 55/64 = 88.84 %; V10 = 84.73 %
- `report/chapter-04-experiments-and-results.md` §4.2 — experiment ledger; currently lacks per-seed CIs
- `sweep_results/pairwise/` — best ensemble checkpoints with multiple seeds available
- `run_pairwise_contrastive.py` — training script; add `--seeds` argument

## Overview

**Already available:** `run_ablation_inference.py` (untracked) runs T-robustness {8, 16, 24, 32},
spatial-pool robustness {1, 2, 4}, and per-ensemble-member accuracy. This covers most of the
ablation work. Phase 06 only needs to:
(a) execute `run_ablation_inference.py` and collect output,
(b) add a 5-seed re-run of the top-3 configs (direct4-s314, V3, pairwise ensemble),
(c) compute paired bootstrap CI vs V3 baseline and vs V10 (84.73 %).

Drop the wide ablation table from the original plan. Keep only the 4-row version
(backbone-only / +Transformer / +ensemble / +calibration) — seed replication is cheaper
with a smaller table, and 4 rows is sufficient for committee credibility.

V10 accuracy anchor from GROUND_TRUTH.md: 84.73 % (54/64, seed s2024).

## Key Insights

- 64-clip test set is small: 95 % Wilson CI for 56/64 = [78.0 %, 95.7 %]. A CI table must acknowledge this width explicitly.
- V3 (55/64) and best ensemble (56/64) differ by 1 clip; a paired bootstrap test is the correct statistical tool (not McNemar, which requires paired predictions, which we have).
- PROGRESS.md shows training times: Pairwise (2.5 h), V3 (3.5 h), YOLO-World (2.5 h). 3 configs × 5 seeds = 15 runs ≈ 37.5 h total. Plan for ≈ 2 days of compute (run overnight if needed).
- The ablation table prescribed in the strategic framing must cover 4 levels: backbone-only / +Transformer / +ensemble / +calibration. These map cleanly to V3 subset, V3 full, Pairwise, and Pairwise-calibrated (57/64).
- Temperature calibration (57/64 calibrated) is already implemented; just needs per-seed runs.

## Requirements

**Functional**
- Re-run top-3 configurations (direct4-s314, YOLO-World, V3-frozen) with 5 seeds each.
- Compute per-seed macro accuracy; report mean ± std and 95 % CI (Wilson or bootstrap).
- Paired bootstrap test (B=10,000 bootstrap samples) comparing:
  - Pairwise ensemble vs. V3 baseline (is +1 clip significant at α=0.05?)
  - Pairwise ensemble vs. V10 bbox-finetune (is regression significant?)
- Formal ablation table: 4 rows (backbone-only / +Transformer / +ensemble / +calibration), showing macro accuracy mean ± CI.
- All statistical code in a standalone `run_statistical_analysis.py`.

**Non-functional**
- 10,000 bootstrap iterations ≤ 60 s on M4 CPU.
- Confidence intervals use Wilson score for per-clip binomial, not normal approximation (appropriate for small n).

**Acceptance thresholds**
- Ablation table exists with ≥ 4 rows and CI column.
- p-value for pairwise ensemble vs. V3 is reported (even if p > 0.05; that is honest).
- Chapter 4 text explicitly states the test set size (n=64) and warns readers that CIs are wide.

## Architecture / Approach

### Configuration Map (top-3 + ablation)

| Config | Training script | Seeds | Expected mean accuracy |
|--------|----------------|-------|----------------------|
| backbone-only (ablation row 1) | `run_pairwise_contrastive.py --model linear` | 1,7,42,314,999 | ≈ 70–75 % (no temporal modelling) |
| V3 (backbone + Transformer, no ensemble) | `run_pairwise_contrastive.py --model direct4 --no-ensemble` | 1,7,42,314,999 | ≈ 85–87 % |
| Pairwise ensemble (direct4 + cascade) | `run_pairwise_contrastive.py` | 1,7,42,314,999 | ≈ 87–89 % |
| Pairwise + temperature calibration | `run_pairwise_contrastive.py --calibrate` | 1,7,42,314,999 | ≈ 87–89 % |

### Paired Bootstrap Test

```python
import numpy as np

def paired_bootstrap(preds_a, preds_b, labels, B=10000, rng_seed=0):
    """Test H0: macro_accuracy(A) == macro_accuracy(B)"""
    rng = np.random.default_rng(rng_seed)
    obs_delta = macro_acc(preds_a, labels) - macro_acc(preds_b, labels)
    # Pool under H0
    combined = np.stack([preds_a, preds_b])  # shape (2, n)
    deltas = []
    for _ in range(B):
        idx = rng.integers(0, 2, size=len(labels))
        resampled_a = combined[idx, np.arange(len(labels))]
        resampled_b = combined[1-idx, np.arange(len(labels))]
        deltas.append(macro_acc(resampled_a, labels) - macro_acc(resampled_b, labels))
    p_value = np.mean(np.abs(deltas) >= np.abs(obs_delta))
    return obs_delta, p_value
```

### Wilson Score CI

```python
from statsmodels.stats.proportion import proportion_confint
ci_low, ci_high = proportion_confint(correct, n_total, alpha=0.05, method='wilson')
```

### Ablation Table Format

| Configuration | Macro Acc (mean ± std) | 95 % CI | vs. V3 p-value |
|---|---|---|---|
| Backbone-only (no temporal) | X.X ± Y.Y % | [a, b] | — |
| + Transformer head (V3) | X.X ± Y.Y % | [a, b] | reference |
| + Pairwise ensemble | X.X ± Y.Y % | [a, b] | p=Z.ZZ |
| + Temperature calibration | X.X ± Y.Y % | [a, b] | p=Z.ZZ |

## Related Code Files

| Action | Path |
|--------|------|
| Create | `run_statistical_analysis.py` — 5-seed re-runs + CI + bootstrap |
| Modify | `run_pairwise_contrastive.py` — add `--seeds` list arg, `--model linear` (backbone-only mode), `--calibrate` flag |
| Create | `figures/ablation-table.tex` (or embedded in chapter as Markdown table) |
| Modify | `report/chapter-04-experiments-and-results.md` — add §4.9 Statistical Significance; update §4.2 experiment ledger with CI column |

## Implementation Steps (Day-by-Day, 3 days)

**Day 1 — Execute run_ablation_inference.py**
1. Run `run_ablation_inference.py` end-to-end. Capture T-robustness {8,16,24,32}, spatial-pool {1,2,4}, per-member accuracy output.
2. Verify output matches expected format; save to `sweep_results/stats/ablation-inference-results.json`.
3. Add `--seeds` argument to `run_pairwise_contrastive.py` (if not present). Add backbone-only mode (`--model linear`).

**Day 2 — 5-seed re-runs of top-3 configs**
4. Run backbone-only for 5 seeds (≈ 0.5 h total). Record per-seed macro accuracy.
5. Run Pairwise ensemble for 5 seeds (note: seed s314 already trained; re-run other 4). Save per-seed logit arrays (64×4) to `sweep_results/stats/`.
6. Run V3 (no-ensemble) for 3 seeds minimum (3.5 h each; run overnight if needed). Accept 3 seeds if compute is tight.

**Day 3 — Statistical analysis + chapter update**
7. Write `run_statistical_analysis.py`: load all seed results; compute mean ± std; Wilson CI; paired bootstrap (pairwise vs. V3, pairwise vs. V10 = 84.73 %); output JSON + Markdown table.
8. Add calibrated row: temperature scaling on each seed's logits (T* on 12-clip validation slice).
9. Run analysis; check p-values (expect p > 0.05 given n=64 — report honestly).
10. Update `report/chapter-04-experiments-and-results.md`: add CI column to §4.2 ledger; add §4.9 (≤ 2 pages) with ablation table, bootstrap p-values, honest n=64 limitation discussion.

## Todo List

- [ ] Day 1: Execute run_ablation_inference.py; capture T-robustness + spatial-pool output
- [ ] Day 1: Save ablation output to sweep_results/stats/ablation-inference-results.json
- [ ] Day 1: Add --seeds argument + backbone-only mode to run_pairwise_contrastive.py
- [ ] Day 2: Run backbone-only 5-seed sweep (≈ 0.5 h)
- [ ] Day 2: Run Pairwise ensemble 5-seed sweep (reuse s314; run 4 new seeds)
- [ ] Day 2: Run V3 no-ensemble ≥ 3-seed sweep (overnight)
- [ ] Day 2: Collect per-seed logit arrays to sweep_results/stats/
- [ ] Day 3: Write run_statistical_analysis.py (Wilson CI + paired bootstrap vs V3 and V10)
- [ ] Day 3: Add calibrated row (temperature scaling per seed)
- [ ] Day 3: Execute analysis; record p-values
- [ ] Day 3: Update §4.2 CI column + write §4.9

## Success Criteria

- Ablation table with 4 rows exists in Chapter 4 with mean ± std and 95 % CI.
- Paired bootstrap p-value reported for pairwise ensemble vs. V3 (regardless of significance).
- §4.9 text explicitly acknowledges n=64 test-set limitation and that the ±1 clip gap between V3 and best ensemble is below the CI width.
- `run_statistical_analysis.py` is reproducible from a single command.

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| V3 5-seed sweep takes > 17 h wall time | Medium | Run 3 seeds instead of 5; widen CI; note in text |
| All bootstrap p-values > 0.05 | High (likely given n=64) | This is the honest result; state "test set is underpowered to distinguish 1-clip differences; larger test set is future work" |
| Backbone-only ablation fails to train (MLP head unstable) | Low | Use simple logistic regression instead; still provides ablation row |
| Seed variance reveals high instability (std > 3 pp) | Low | Report honestly; high std means the best result may be a lucky seed; chapter 4 already positions results as "ceiling" |

## Security / Ethics / Privacy

- No new data. Statistical analysis on existing test results.

## Dependencies + Next Steps

- Phase 03 seed runs can be reused here if same protocol (5 seeds, same splits). Coordinate to avoid duplicate compute.
- Output consumed by: **Phase 07** (ablation table → §4.9; CI column → §4.2 rewrite).
- Landing chapter: `report/chapter-04-experiments-and-results.md` §4.2 (updated) and §4.9 (new).

**Unresolved questions**
- Does the current training script already support multi-seed iteration, or does it require a new `--seeds` argument?
- Are V3 individual seed logit files preserved in `sweep_results/` or only the best checkpoint?
