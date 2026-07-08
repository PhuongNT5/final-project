# WS2 Results — Leakage-Free k-Fold Cross-Validation

**Run:** k=5, seeds {42,123,314}, 24 scene-clusters, 300 epochs, MPS, 302 min.
**Source:** `run_kfold_eval.py` → `reports/kfold-eval.json`. Pooled all 691 Unsafe-Net clips;
4-class unsafe subset N=516 (per-class: Walk 210, Unauth 108, Panel 142, Forklift 56).
**Model evaluated:** single Direct-4 `TransformerHead` on frozen-COCO stats features — the
*honest best config* (blind protocol §4.6 rejected the cascade; single head best every seed).
Per-fold inner-val epoch selection; **the held-out fold is never seen in training/selection.**

> **Why this evaluation:** the canonical thesis numbers (56/64 best-case, 53/64 blind) rest on
> ONE 64-clip split → bootstrap CI **[79.7, 95.3]** (15.6 pp wide). k-fold over 516 clips with
> tight CIs is the rigor fix the committee asked for. No day/session metadata exists in
> Unsafe-Net (verified), so a scene-cluster GroupKFold substitutes for day-grouping.

## Headline table

| Protocol | top-1 (mean±std) | top-1 CI₉₅ | macro (mean±std) | macro CI₉₅ |
|---|---|---|---|---|
| **Stratified k-fold (random)** | **75.3 ± 1.2 %** | [73.1, 77.5] | **77.8 ± 1.0 %** | [76.0, 79.6] |
| **Scene-group k-fold (leakage-robust)** | **71.9 ± 1.6 %** | [69.7, 74.2] | **71.8 ± 1.0 %** | [70.0, 73.8] |

- **CI width collapses from 15.6 pp (single split) → ~4.4 pp (k-fold)** — the core rigor win.
- **Scene-leakage gap = 3.4 pp top-1 / 6.0 pp macro.** Modest, *not* catastrophic memorisation,
  but real: the scene-robust 71.9 % / 71.8 % is the honest **deployable generalisation** estimate.

## Three findings (all defensible, all honest)

### F1 — The Transformer head does **not** beat a linear probe (and is significantly worse on random split)
Logistic regression on **time-averaged** frozen features vs the Direct-4 Transformer, same folds,
Nadeau–Bengio corrected resampled t-test:

| Protocol | Δ (TF − linear) | N-B t | p | verdict |
|---|---|---|---|---|
| Stratified | **−4.65 pp** (TF≈75.3 vs linear≈80.0) | −3.15 | **0.007** | TF **significantly worse** |
| Scene-group | +1.70 pp (TF≈71.9 vs linear≈70.2) | 0.75 | 0.47 | tie |

→ The temporal/attention machinery earns nothing on this dataset; a logistic regression on
mean-pooled features is as good or better. Quantifies the temporal-null ablation (§4.7.8) with a
proper test. **Strong support for dropping "Transformer" from the headline / reframing (W1).**
Citable as a valuable negative result (Beyer et al. 2017).

### F2 — Safety view (per-class false-negative / miss rate)
| Class | FNR random | FNR scene-robust |
|---|---|---|
| WalkwayViolation | 32.4 % | 29.4 % |
| **UnauthorizedIntervention** | **44.8 %** | **68.2 %** |
| OpenedPanel | 7.7 % | 6.8 % |
| CarryOverload (forklift) | 0.0 % | 0.0 % |

→ Essentially **all** safety risk sits in the two zone-ambiguous classes; the other two are
near-perfect. **UnauthorizedIntervention is the critical failure** (misses 2/3 of true cases under
scene-robust eval). This is the safety-framed result (W3) and the precise motivation for WS4
(virtual fencing to inject zone knowledge).

### F3 — Confusion is the predicted Walk↔Unauth zone ambiguity (now at N=516, not 64)
Stratified seed-0 confusion (rows=true, cols=pred; order Walk/Unauth/Panel/Forklift):
```
Walk     [140,  52,  18,   0]   ← 52 Walk → Unauth
Unauth   [ 32,  57,  19,   0]   ← 32 Unauth → Walk
Panel    [  3,  12, 127,   0]
Forklift [  0,   0,   0,  56]   ← perfect
```
The dominant errors are exactly the bidirectional Walk↔Unauth zone confusion (52+32 swaps),
reproducing §4.7.6 at full dataset scale. Forklift is perfectly separable (distinct object).

## Honest caveats (state in thesis)
1. **k-fold ≠ the deployed pipeline.** It evaluates the *single* Direct-4 head, not the deployed
   ensemble+cascade+hybrid+gate stack. So the honest statement is: *"the core single-head
   classifier cross-validates at 72–75 % top-1"*, NOT "the deployed system gets 75 %." The 56/64
   deployed-checkpoint number remains valid for the canonical split.
2. The k-fold numbers (72–75 %) are **below** the canonical-split headline (82.8–87.5 %): the
   single split is favourable + the headline used ensembling/selection. k-fold is the *more
   representative generalisation* estimate. Report **both**, framed as complementary.
3. Scene clusters are a KMeans proxy for sessions, not ground-truth day labels (which don't exist).

## Recommended thesis framing
Lead Chapter 4 with: tight-CI k-fold generalisation (75.3 ± 1.2 % random / 71.9 ± 1.6 %
scene-robust) **+ F1 negative result + F2 safety FNR**, then present the canonical-split deployed
checkpoint (56/64) as the shipped artifact. This converts "weak, wide-CI accuracy" into
"rigorously characterised, honestly bounded, safety-aware" — exactly the committee's ask.
