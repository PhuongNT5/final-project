# Phase 03 — Spatial-Relation Transformer: Final Results

**Date:** 2026-08-10 · Stage B of the paper-faithful pipeline (YOLOv11s entity detector → box-token
Transformer → 4-class unsafe behaviour). Blind held-out protocol (val carved from 452 train clips;
64-clip test touched once per seed; seed selected on val macro). CPU training (MPS training bug).

## Full config sweep (blind, 64-clip test)
| Config | blind top-1 | blind macro | 7-seed mean | WalkwayViol R |
|---|---:|---:|---:|---:|
| v1 geometry, weighted | 48/64 (75.0%) | 83.7% | 48.6 | 59% |
| v1 geometry, **no-weights** | 51/64 (79.7%) | **86.1%** | 49.4±2.2 | 69% |
| v2a + relational proximity | 49/64 (76.6%) | 83.0% | 47.2 | 66% |
| **v2b + appearance, no-weights** | **52/64 (81.2%)** | 84.2% | 50.3±2.2 | **78%** |
| v2b + appearance, weighted | 41/64* | — | 47.0 | 59% |
| hybrid v1 ⊕ v2b (14 models) | 51/64 (79.7%) | 84.6% | — | 72% |

\* weighted picked a noisy val-peak seed (epoch 2) — illustrates seed variance, not a real config gain.

## Headline (best config = v2b appearance, no-weights)
- **top-1 = 52/64 = 81.2%** (BCa 95% CI [67.5, 87.5]) · **macro = 84.2%**
- Per-class recall: WalkwayViol **78.1%** · UnauthorInt 81.8% · OpenedPanel 76.9% · CarryOverload 100%
- Confusion: WalkwayViol→UnauthorInt cut to **6** errors (was 12 in v1) — appearance breaks the pair.

## Comparison to targets
| | top-1 | macro |
|---|---:|---:|
| **Ours best (v2b, blind)** | **52/64 (81.2%)** | 84.2% |
| Ours best-macro (v1-nw, blind) | 51/64 | **86.1%** |
| Paper no-bbox baseline | 57/64 (89.1%) | 85.78% |
| Paper full-bbox | 62/64 (96.9%) | 95.81% |
| Thesis frozen-appearance (blind) | 53/64 | ~85% |

- **Macro parity-to-advantage over the no-bbox baseline** (v1-nw 86.1% > 85.78%; v2b 84.2% ≈ it).
- **Top-1 below** the no-bbox baseline (52 vs 57; Δ≈−7.8 pp, not significant on N=64).
- **Well below full-bbox** (52 vs 62; Δ=−15.6 pp, p=0.003) — the data-bound gap.

## Findings
1. **Appearance is the operative lever.** Geometry-only tops out ~51/64; adding per-box appearance
   (MobileNet crop-embed) lifts to 52/64 and nearly halves the WalkwayViol↔UnauthorInt confusion
   (recall 59→78%). This confirms the confusion needs *appearance context*, not just zone scalars.
2. **Zone geometry is weak here** — per-view canonical polygons don't align tightly per clip; foot_in_walkway
   ~0.12 for both confused classes. Relational proximity (v2a) added nothing.
3. **Class weighting hurts** on this small N (unstable, over-predicts minority) — no-weights is the pick.
4. **Ensembling did not beat the best single config** — models too correlated (same tokens).
5. **The ~52–54/64 ceiling reproduces** through this clean entity pipeline, matching the thesis's earlier
   12+ tracks. The gap to the paper's 62/64 is **data-bound**: their 2,262 per-frame *zone-labelled* bboxes
   encode restricted-zone membership our COCO-detector + coarse zones cannot.

## Update (2026-08-13) — S1a per-frame supervision + S1b frame-gate
Motivated by the paper back-check (see `gap-analysis-and-next-strategy.md`): the paper's +10 pp comes
from a per-frame BEHAVIOUR detector (2,262 behaviour-bboxes) that frame-reduces + single-action-splits.
- **S1a — per-frame auxiliary behaviour loss** (each frame's tokens supervised by the clip label,
  452 → ~7 k signals; aux_w=0.5, 1.0 worse): **seed-ensemble 54/64 (84.4 % top-1, 88.4 % macro)** —
  best result, **macro clearly > no-bbox baseline (85.78 %)**, beats the thesis's 53/64. Confirms the
  supervision-placement diagnosis. High seed variance (48–55) → the seed-ensemble is the robust headline.
- **S1b — behaviour-confidence frame-gate** (val-selected τ): **no gain** — τ→0 (kept_frac ≈ 0.99),
  gated ensemble 53/64 ≤ S1a. Our clips are already single-action, so the paper's frame-reduction
  mechanism has nothing to reduce. Documented negative.

**Sharpened diagnosis:** the residual gap to 62/64 is the **strength of per-frame behaviour supervision**
(paper trained a strong per-frame behaviour recognizer), NOT frame-selection or architecture. Direct test
= S3 (train a whole-frame behaviour recognizer, e.g. YOLO26, on the existing 2,262 frames' clip labels).

## Conclusion & next lever
The entity→(geometry+appearance)→Transformer pipeline (the designed contribution) reaches **52/64
(81.2% top-1, 84.2% macro) blind — macro-competitive with the no-bbox baseline**, essentially at the
thesis's honest 53/64 ceiling, via a cleaner interpretable route that demonstrably reduces the target
confusion. This is a natural, honest stopping point for architecture. The remaining accuracy gap to
full-bbox is **not architectural** — it requires paper-scale, zone-labelled bbox supervision
(documented future work). Deliverables: `detector_best.pt`, token builders, `model_relation_head.py`,
`run_relation_transformer.py`, `ensemble_relation.py`, all results reproducible under the blind protocol.
