# Blind Held-Out Evaluation — Result

**Protocol:** epoch selected on a stratified validation slice carved from TRAIN (91 of 452
4-class train clips held out; split_seed=0, val_frac=0.2); 64-clip test set untouched until one
final eval. Fixed headline ensemble (direct4 + scenario×pair cascade); no greedy-on-test member
selection; no bias calibration. Seeds 314/42/123. Runtime 77.7 min on M4 MPS.

Artifact: `blind-eval-result.json`. Reproduce: `python run_blind_eval.py --seeds 314 42 123`.

## Numbers

| Config | Blind top-1 (mean ± std) | per-seed | Blind macro (mean ± std) |
|---|---|---|---|
| **direct4 (single head)** | **51.67 ± 1.89 /64 (80.7%)** | 53/53/49 | **85.01 ± 1.77%** |
| cascade | 50.67 ± 1.25 /64 (79.2%) | 49/52/51 | 84.35 ± 2.02% |
| ensemble (headline def.) | 50.67 ± 1.25 /64 (79.2%) | 51/52/49 | 83.85 ± 0.86% |

## The ladder (how much each optimism source was worth)

| Number | top-1 | macro | What it is |
|---|---|---|---|
| Best-case (test-selected, lucky s314, greedy ensemble) | 56/64 = 87.5% | 89.28% | optimistic upper bound |
| Typical (test-selected, standard seeds, single head) | 53.3/64 | ~84.8% | seed-honest, still peeks test |
| **Blind (val-selected, honest)** | **~51.7/64 = 80.7%** | **~85.0%** | clean generalization estimate |
| Paper ConvLSTM-only (no bbox) baseline | 57/64 = 89.06% | 85.78% | comparison target (protocol unknown) |
| Paper full bbox cascade | 62/64 = 96.88% | 95.81% | upper bound w/ 2,591 bbox |

## Three findings (all material to the defense)

1. **The +3.5pp macro "win" does not survive.** Best blind macro is the single direct4 head at
   **85.01%** vs the baseline's **85.78%** — i.e. roughly macro *parity-minus-one*, not a win. The
   ensemble macro (83.85%) is clearly *below* baseline. The headline "+3.5pp macro" was a
   test-selection artifact.

2. **The cascade/ensemble adds nothing — and slightly hurts.** Blind, the single direct4 head
   (51.67) BEATS the ensemble (50.67) on both metrics. This confirms §4.7.9 (P=0.63, CI touches 0)
   in the strongest form: the most-engineered component (pairwise-contrastive cascade) is not a
   demonstrated contribution. The honest technical core is the **single frozen-feature direct4 head**.

3. **~5 samples of "56/64" were leakage.** s314 single head: 55 test-selected → 53 blind (2 samples
   = epoch selection). Ensemble: 56 test-selected → 51 blind (5 samples = epoch selection + lucky-seed
   + greedy-on-test member selection).

## v2 — "try to improve it honestly" (val_frac 0.15, 4 seeds, val-based config+seed selection)

Levers added: (a) more training data (val_frac 0.2→0.15), (b) config selection (direct4 vs cascade
vs ensemble) **on validation**, (c) seed selection **on validation**. Artifact: `blind-eval-v2-vf0.15.json`.

| Config | Blind test top-1 (mean ± std) | per-seed (314/42/123/7) | macro |
|---|---|---|---|
| **direct4** | **51.75 ± 2.28 /64 (80.9%)** | 48/52/53/54 | 83.14% |
| cascade | 45.75 ± 3.19 | 41/50/46/46 | 77.14% |
| ensemble | 48.50 ± 3.91 | 42/52/51/49 | 81.64% |
| val-selected config (always picked direct4) | **51.75 /64** | — | 83.14% |
| val-selected seed (=42) + direct4 | **52/64 (81.2%)** | — | 80.88% |

**Running more did NOT rescue the number — and that is itself a result.** Two robust facts:
- Honest config selection on validation **always picks direct4** and rejects the cascade/ensemble
  every seed → the cascade is honestly, not just statistically, not a contribution.
- The blind number is **stable at ~51–52/64 (~81% top-1)** across both protocols (v1 ensemble-fixed
  50.7, v2 direct4 val-selected 51.8). It is not an artifact of one unlucky split; it is the real
  generalization level. Macro lands ~83–85% — **at or just below** the 85.78% baseline (parity, not win).

## Consequence for the thesis

The accuracy-competition framing (parity / macro-win vs the no-bbox baseline) is **no longer
defensible** once leakage is removed. The thesis must pivot its spine to what is unaffected and now
*more* strongly supported:
- **Negative result / data-bound diagnosis** (frozen COCO features are insufficient — the blind gap
  to even the no-bbox baseline reinforces this).
- **Measured open-source edge deployment** (Jetson/M4 — independent of accuracy).

Honest cross-paper caveat: the paper reports no blind number, so a *matched-protocol* cross-paper
comparison is necessarily test-selected (56 vs 57 = parity-minus-one). Report this as a caveat, NOT
as a way to resurrect a parity claim. The defensible internal estimate is the blind ~51.7/64.
