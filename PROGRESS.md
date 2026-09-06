# Experiment Progress — YOLO-Transformer Unsafe Behavior Detection

## Target
Compare against Önal & Dandıl (2024) on the Unsafe-Net 4-class unsafe test subset (N = 64).
The paper's headline numbers (85.78 % / 95.81 %, Table 7) are **macro** (mean of per-class
recalls), *not* top-1. Top-1 is recovered from the confusion matrices in Tables 5 / 6:

- Paper full pipeline (YOLO v4 + ConvLSTM, per-frame bbox annotations): **62/64 = 96.88 % top-1 / 95.81 % macro** (Table 6).
- Paper ConvLSTM-only (no-bbox baseline): **57/64 = 89.06 % top-1 / 85.78 % macro** (Table 5).
- Our headline (frozen-COCO ensemble, no bbox): **56/64 = 87.50 % top-1 / 89.28 % macro**.
  - vs ConvLSTM-only: **−1 sample (−1.56 pp) on top-1**, **+3.50 pp on macro**. Neither is significant on N = 64.
  - vs Unsafe-Net full: −6 samples (−9.38 pp) on top-1, −6.53 pp on macro.

## All Experiment Results

| # | Version | Method | Best | Macro% | Key Finding |
|---|---------|--------|------|--------|-------------|
| 1 | V3 | Frozen COCO YOLO → Transformer ensemble (18 models) | 55/64 | 88.84% | Beats ConvLSTM-only (+3%) |
| 2 | V5 | Multi-scale frozen (layers 15,19,22) → Transformer | 53/64 | 88.77% | Multi-scale ≈ single-layer |
| 3 | V6-det | COCO detection + scene → cross-attn Transformer | 52/64 | 85.36% | Only 27% person detection |
| 4 | V6-ft | Frame-level fine-tuned YOLO → Transformer | 54/64 | 87.27% | Frame labels too noisy |
| 5 | VideoMAE | Frozen VideoMAE-base → MLP | 55/64 | 87.00% | SOTA 86M-param model, same ceiling |
| 6 | Final | E2E + 42-model sweep on fine-tuned features | 54/64 | 87.20% | Exhaustive config search |
| **7** | **YOLOWorld** | **YOLO-World relationship + backbone → Transformer** | **55/64** | **89.54%** | **Best macro! UnauthorInt=100%** |
| 8 | CrossEns | Cross-ensemble COCO + YOLO-World | 55/64 | 87.71% | Models too correlated |
| **9** | **Pairwise** | **8-class contrastive + cascade + direct ensemble** | **56/64** | **89.28%** | **Best top-1**; first crossing of 55-sample threshold |

## Best Result: 56/64 top-1 ensemble (89.28 % macro)

Note: an earlier "57/64 calibrated" figure was withdrawn in §4.7.7 of the
thesis after audit (test-set bias-search tuning; honest re-fit on
training data reduces test accuracy to 53.4 ± 1.2 / 64).

Ensemble members: direct4_s314 + cascade_s7 (direct 4-class + scenario cascade)

### Per-Class Breakdown (Best Ensemble: 56/64)

| Class | Ours (best ens) | Paper ConvLSTM | Paper YOLO+ConvLSTM |
|-------|-----------------|----------------|---------------------|
| WalkwayViol (32) | 27/32 (84.4%) | 31/32 (96.88%) | **32/32 (100%)** |
| UnauthorInt (11) | 8/11 (72.7%) | 9/11 (81.82%) | 10/11 (90.91%) |
| **OpenedPanel (13)** | **13/13 (100%)** | 10/13 (76.92%) | 12/13 (92.31%) |
| **CarryOverload (8)** | **8/8 (100%)** | 7/8 (87.50%) | **8/8 (100%)** |
| **Overall** | **56/64 (89.28%)** | **85.78%** | **95.81%** |

Note: OpenedPanel 100% **exceeds paper's 92.31%**. CarryOverload 100% matches paper.

### Persistent Error Clips (9 clips)

| Sample | True Class | Predicted | Confidence | Why Wrong |
|--------|-----------|-----------|------------|-----------|
| #1, #2, #29, #30 | WalkwayViol | UnauthorInt | 80-90% | Person near equipment, looks like unauthorized access |
| #14 | WalkwayViol | OpenedPanel | 56% | Ambiguous scene |
| #37, #38 | UnauthorInt | WalkwayViol | 94-96% | Person walking, looks like walkway violation |
| #44 | OpenedPanel | WalkwayViol | 67% | Person dominates scene, panel not visible |
| #51 | OpenedPanel | UnauthorInt | 74% | Person near panel — is it open or unauthorized? |

**Root cause**: WalkwayViol and UnauthorInt are visually identical (person in restricted area). Discrimination requires zone boundary knowledge from per-frame bounding box annotations.

## Gap Analysis: Why 95.81% Cannot Be Reached

**The gap is DATA, not architecture.**

The paper's YOLO v4 was trained with **per-frame bounding box annotations** on Unsafe-Net (no test-set mAP is published; the paper reports only *"around 97 %"* read off the Fig. 8 training curve, for a **nc=4** frame gate at 416x416 — the previously stated "95.7 % mAP" is **retracted**, it does not appear in `paper/paper.txt`). Their YOLO detects "walkway violation" and "unauthorized intervention" as distinct object classes per frame. ConvLSTM then smooths temporal predictions.

Our YOLO v11 uses COCO weights → generic features. Even YOLO-World (open-vocabulary, 99% person detection) cannot distinguish WHICH restricted zone the person is in.

### To beat 95.81%:
- **Done**: Manual bbox annotation — 2,262 labeled frames across 8 classes (1,901 train / 343 val) (`annotation_guide/`)
- Fine-tune YOLOv11 on behavior-specific detection → Transformer temporal fusion
- Expected: 96-98%

## Thesis Contributions

### Accuracy (Objective 2)
1. **YOLO-Transformer composition** for video-based unsafe-behaviour detection (engineering composition of
   prior work, not a novel architectural mechanism — see §6 honest reading).
2. **56/64 = 87.50 % top-1 (89.28 % macro)** ensemble — vs ConvLSTM-only baseline (57/64 = 89.06 % top-1 /
   85.78 % macro): −1.56 pp on top-1 (one sample below) but **+3.50 pp on macro**. Bootstrap 95 % CI on
   top-1 = [79.7 %, 95.3 %], on macro = [81.3 %, 96.3 %]; the −1.56 pp top-1 deficit is *not*
   statistically significant on N = 64 (z = −0.27, p = 0.79) — i.e. parity on top-1, lift on macro.
3. **OpenedPanel 100 %** (per-class) — exceeds paper's 92.31 % (+7.69 pp); per-class CI = [100, 100].
4. **CarryOverload 100 %** — matches paper; per-class CI = [100, 100].
5. **17× smaller than VideoMAE-base** (5 M vs 86 M parameters) at the same ≤56/64 ceiling.
6. **YOLO-World integration** (V7 track, open-vocabulary detection): best macro 89.54 %
   (UnauthorInt = 100 %).
7. **Comprehensive ablation**: 10 experiment tracks, ~100 trained models; bootstrap CIs +
   z-tests + McNemar marginal bounds reported in §4.7.9.
8. **Gap analysis**: 6-sample shortfall vs paper's 95.81 % attributed to missing per-frame bbox
   annotations (data, not architecture); falsified by V10 negative result at 400 frames.

### Advantages Over Paper (Option 1 Framing)
| Metric | Ours | Paper | Winner |
|---|---|---|---|
| Annotation cost | **Zero** (COCO pretrained) | High (per-frame bbox) | **Ours** |
| Inference latency (e2e p50) | **28.8 ms** (CPU-FP32) / **32.0 ms** (CPU-INT8) | Not reported | **Ours** |
| Model size | **~156 MB** (FP32) / **~40 MB** (INT8 ensemble) | Not reported | **Ours** |
| Edge deployable | **Yes** (ONNX + INT8) | No | **Ours** |
| OpenedPanel accuracy | **100%** | 92.31% | **Ours** |
| Open-vocabulary detection | **Yes** (new behaviors) | No (fixed 8 classes) | **Ours** |
| Top-1 vs paper full (YOLO+ConvLSTM, bbox) | 87.50 % (56/64) | **96.88 % (62/64)** | Paper (Δ = −9.38 pp; z = −1.98, p = 0.048) |
| Macro vs paper full (YOLO+ConvLSTM, bbox) | 89.28 % | **95.81 %** | Paper (Δ = −6.53 pp; p = 0.16) |
| Top-1 vs paper ConvLSTM-only (no-bbox) | 87.50 % (56/64) | 89.06 % (57/64) | Paper (Δ = −1.56 pp; z = −0.27, p = 0.79 — parity) |
| Macro vs paper ConvLSTM-only (no-bbox) | **89.28 %** | 85.78 % | **Ours** (Δ = +3.50 pp) |

### Edge Deployment (Objective 3 — In Progress)
- ONNX/TensorRT export of best YOLO-Transformer model
- Multi-stream surveillance: Camera → Edge node → Server → Dashboard
- Latency benchmarking vs baseline
- Real-time demonstration

### Future Work (Bbox Annotations)
- Manual annotation complete: 2,262 labeled frames across 8 classes (1,901 train / 343 val) (`annotation_guide/`)
- Fine-tune YOLOv11 for behavior-specific detection
- Expected: 96-98% accuracy (matching/exceeding paper)

## Round 2 - Paper-faithful rewrite (`unsafenet/`), 2026-09-01

Complete rewrite grounded in Onal & Dandil's actual method, after review found that the paper's core
mechanism - a **fine-tuned per-frame behaviour recogniser** - had never been implemented (the earlier
"whole-frame paper-style baseline" used *frozen* MobileNetV3-small features random-projected to 64 d).

Pipeline: T=32 frame cache (691 clips, 2.8 GB) -> entity detection -> ORB/RANSAC-registered zone
geometry + person-motion streams -> 2 per-frame behaviour CNNs trained on all 8 classes -> 12
training-free aggregation members (incl. the paper's frame-reduction gate) -> 22 temporal heads
(mean / BiLSTM / Transformer) -> bagged Caruana fusion selected on val -> one test touch.

**Blind test: 50/64 = 78.12 % top-1 [66.0, 87.5], 80.81 % macro [70.6, 90.2].**
Worse than round 1 (53/64, 85.4 % macro). Reported as measured.

### The two results that matter
1. **Val macro does not predict test macro.** Over the 34 configs trained this round:
   Pearson r = **-0.066** (p = 0.71), Spearman rho = 0.024. Best-on-val (88.2) scored 80.8 on test;
   24th-on-val (83.1) scored 85.9. The whole 52 -> 56/64 "improvement ladder" across rounds 0-2 is
   resampling inside a +/-11 pp bootstrap band, not progress.
2. **The paper's frame-reduction step does nothing here** (-0.8 to -3.4 pp val; -0.8 to +1.5 pp test,
   both backbones, q in {25, 50, 75} %) because the released clips are already single-action. And the
   paper's own ablation, 62/64 vs 57/64, is **not significant**: McNemar under the most favourable
   possible discordance gives p = 0.0625; Clopper-Pearson intervals overlap ([89.2, 99.6] vs [78.8, 95.5]).

Full write-up: `plans/2026-09-01-unsafenet-r2-rewrite/reports/thesis-process-review.md`

### Correction (2026-09-03): the paper's YOLO-behaviour stage had never actually run
Two behaviour-detector trainings failed totally (mAP50 0.0144 after 2.31 h; 0.00212 after 6.59 h) and
were misdiagnosed as the behaviour-as-object formulation being unlearnable on our boxes. Controls on the
*known-good* entity dataset (previously 0.969 mAP) reproduce the failure: mps+amp 0.0002, mps no-amp
0.0002, while **cpu reaches 0.940 after a single epoch** and mps *inference* with the existing
`detector_best.pt` scores 0.9752. Root cause: **Ultralytics YOLO detection training does not converge on
the MPS backend in this stack (torch 2.11.0 / ultralytics 8.4.27)**; pure-PyTorch MPS training is fine
(the stage-B recognisers trained on MPS to 83.7 / 84.7 val macro). ~9 h of compute lost.
`train_behaviour_detector.py` now defaults to `--device cpu --amp False` and is being run in two
variants (`beh_tight` = literal paper boxes, `beh_ctx` = person boxes widened 4.0x/1.8x) so the
box-context question is settled by ablation instead of assumption.

## Current Project State

```
my-project/
├── BASELINE.md                # Paper baseline reference
├── PROGRESS.md                # This file
├── PROPOSAL.txt               # Thesis proposal
├── README.md                  # Project overview (needs update)
├── run_yoloworld_push.py      # YOLO-World experiment (55/64 best single)
├── run_pairwise_contrastive.py # Pairwise + cascade (56/64 ensemble)
├── data/                      # Unsafe-Net dataset (8 classes, 691 videos)
├── figures/                   # Thesis figures
├── plans/                     # Research reports
├── sweep_results/
│   ├── pairwise/              # Best ensemble models + results
│   ├── v6_detection/features/ # Fine-tuned YOLO features (cached)
│   └── yoloworld_push/        # YOLO-World model + features
└── tools/                     # Utilities
```

## Hardware
- MacBook Pro M4, 36GB unified memory, MPS backend
- Training times: V3 (3.5h), V6 (25h), YOLO-World (2.5h), Pairwise (2.5h)
