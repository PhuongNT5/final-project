---
title: "Phase 13 — ESTA: entity spatio-temporal attention as the thesis contribution"
status: in-progress
priority: P0
created: 2026-09-29
tags: [esta, yolo11, yolo26, clip, transformer, lso-1100, g14, g15, g16, g17]
---
# Phase 13 — ESTA (Entity SpatioTemporal Attention)

## Context Links
- Pre-registration + falsification rules: `risks.md` (R48, G14-G17, integrity log 2026-09-29)
- Headline reframe: `plan.md` -> DECISIONS UPDATED 2026-09-29 (supersedes D1, amends D4)
- Protocol evidence this lane rests on: `reports/finding-2026-09-21-protocol-inflation-and-the-capacity-contrast.md`
- Superseded accuracy lane: `phase-12-isafetybench-accuracy-lane.md`

## Overview
Priority P0. The thesis contribution is an architecture for unsafe-behaviour detection, evaluated
under leave-source-out cross-validation. Prior lanes established the *evaluation*; this lane
supplies the *system*.

## Key Insights
- The seven-condition ablation showed the signal on iSafetyBench is entirely crop appearance
  (`appearance_only` 75.6% vs full 75.2%, p=0.551): all ten hand-built geometric/relational FEAT
  columns were dispensable.
- Appearance in the old tokens was MobileNetV3-small crushed through a **fixed random projection
  576 -> 64**, and the whole-frame baseline used the *identical* crippled projection. So the two
  arms of the old G11 comparison were both bottlenecked, and the bottleneck - not the relational
  head - is the prime suspect for the accuracy ceiling (R48).
- Even through that bottleneck the entity pipeline beat whole-frame by +6.2 pp (E0, p=1.59e-05),
  so entity structure carries signal that survived it. Lifting the bottleneck is the experiment.

## Architecture
```
video -> 16 sampled frames -> YOLOv11s (top-8 boxes/frame, conf 0.25, 30% context crop)
      -> frozen CLIP ViT-B/32 -> 512-d per entity  (+ class id, box xywh, confidence)
      -> stage 1: attention over entities WITHIN a frame  -> per-frame CLS state
      -> stage 2: attention over the 16 frame states      -> clip CLS -> binary head
```
Factorised (ViViT-style) rather than joint space-time attention: joint attention over 8x16=128
tokens costs 128^2 attention on 1,100 training clips, which this corpus cannot support.

| model | role | params |
|---|---|---|
| ESTA | the contribution | 352,258 |
| B1 | ESTA with the entity axis deleted, nothing else changed | 201,090 |
| B1w | B1 with a second temporal layer — capacity-matched to ESTA | 333,570 |
| B2 | frozen VideoMAE-base **fine-tuned on Kinetics-400**, no detector | 233,858 |
| E0@64-d | the old relation encoder at the old bottleneck | 298,118 |

## Requirements
- Functional: every model scored on the identical LSO-1100 folds, identical clip order, identical
  training recipe (AdamW 1e-3 / wd 0.01, cosine, inverse-freq class weights, label smoothing 0.05,
  60 epochs, 3 seeds, best-val-macro checkpoint, 15% stratified val inside each training pool).
- Non-functional: 0 test touches; every comparison paired (cluster bootstrap over the 151 source
  videos + exact McNemar); no threshold tuning; no per-fold selection; no re-runs for a better number.

## Related Code Files
Created: `model_entity_spatiotemporal.py`, `esta_training.py`, `run-esta-lso.py`,
`frame_sampling.py`, `build-isafetybench-clip-features.py`,
`build-isafetybench-videomae-features.py`, `train-esta-deployment-checkpoint.py`,
`tools/adjudicate-r48-gates.py`, `tests/test-esta-alignment-and-guards.py`,
`demo/esta-window-pipeline.py`, `demo/run-esta-demo.py`.
Modified: `plan.md`, `risks.md`.

## Implementation Steps
1. Extract CLIP features for both arms from ONE decode and ONE detector pass. **Done** — 1,100
   clips, `entity (1100,16,8,512)`, `wholeframe (1100,16,512)`, **52,997 detections, identical to
   the old token build**, box coords max|diff| 0.0.
2. Extract Kinetics-VideoMAE features for B2 from the same frames (`frame_sampling.decode`).
3. Run ESTA, B1, B1w, B2 on LSO-1100. **In progress.**
4. Adjudicate G14/G15/G16 mechanically via `tools/adjudicate-r48-gates.py`.
5. G17 detector ablation: re-extract with `yolo26s.pt` (NMS-free, end2end), re-run ESTA.
6. Train the deployment checkpoint, export ONNX, measure on Jetson **only if G14 passes**.
7. Demo on held-out footage; write the finding report.

## Todo List
- [x] CLIP feature build (both arms, one pass)
- [x] Architecture + training module + driver + guard tests (4/4 pass)
- [x] Pre-register G14-G17 and the plumbing-check integrity log
- [x] ESTA / B1 / B1w / B2 LSO-1100 runs (ESTA sharded 4-way; merge coverage-asserted)
- [x] G14/G15/G16 adjudication + finding report — **G14 FAILED (-0.5 pp, p=0.683)**
- [x] G17 YOLO26 ablation — **newer detector is 2.2 pp WORSE** (p=0.00412); G14 not detector-limited
- [x] Deployment checkpoint — **B1**, not ESTA: `deploy/b1-deploy-3seed.pt`
- [x] ONNX export — B1 3-seed head (2.56 MB, parity 1.19e-07) + CLIP ViT-B/32 tower (351.6 MB,
      parity 1.57e-05, gitignored: over GitHub's 100 MB per-file limit)
- [x] Edge-cost benchmark (host CPU) — tower 375.0 ms vs head 0.3 ms per 16-frame clip; the
      backbone is **99.9 % of wall clock** and costs **18.3x** the MobileNetV3 graph it replaced
- [x] Complementarity diagnostic — entity vs whole-frame error correlation **phi=0.49** (control:
      0.86 within the whole-frame family), oracle 91.2 % vs 84.4 % best-single. Fusion scores from
      that sweep are **selection-contaminated and banned from the thesis** (risks.md R49)
- [x] G18 pre-registered (two tiers) BEFORE the fusion model existed
- [x] `TwoStreamFusion` implemented — both G14 arms kept whole, attention fusion, aux heads at
      lambda=0.3; 686,726 params; 6/6 arm-reuse tests pass incl. bit-exact encode/head refactor
- [x] G18 adjudication — **FAILED.** TSF 84.1 % vs B1w 84.4 % = -0.3 pp, p=0.401. Fusion collapsed
      onto the whole-frame stream (discordance vs B1w fell 150 -> 51). Pre-registered consequence
      applied: thesis headlines B1/B1w, YOLO is localisation + edge
- [x] G18 capacity control — B1xl (696,962 params, **1.5 % larger than TSF**) scores **82.7 %**,
      1.7 pp BELOW B1w at half the size. Capacity ruled out across a 3.5x range, both directions
- [ ] Jetson measurement (author operates the device)
- [ ] Demo run on real footage
- [ ] **AUTHOR DECISION — now decidable by G18 rather than by preference.** The open question was
      whether to headline B1 (no detector in the accuracy path) or keep the entity pipeline despite
      G14. G18 replaces that choice with a measurement: a PASS puts YOLOv11 back in the accuracy
      path as a *complementary* stream and the thesis headlines the two-stream model; a FAIL
      headlines B1 with YOLO as localisation + edge subject. Either way the author confirms the
      framing, but no longer has to arbitrate it on taste.

## Success Criteria
G14 passes: ESTA minus **max(B1, B1w)** >= +3 pp, cluster CI over 151 sources excluding 0,
McNemar p < 0.05. Anything less is a fail, and the pre-registered consequence stands.

## Risk Assessment
- **B1 improves too and the gap closes.** This is exactly how G11 failed once. Probability G14
  passes, estimated before the run: roughly even.
- **B2 beats ESTA.** Then the detector is not justified on accuracy grounds even if G14 passes,
  and the thesis must say so. B2 is reported whatever it shows.
- **Schedule.** This lane spends the slack the 2026-09-21 freeze was protecting. If Jetson
  re-measurement does not fit, the edge chapter stays E3's and the thesis states that plainly.

## Next Steps
Adjudicate, then write `reports/finding-2026-09-29-esta-entity-spatiotemporal.md` with the full
metric suite, the paired statistics, and the negative results in the same table as the positive ones.
