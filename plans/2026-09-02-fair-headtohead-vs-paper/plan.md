# Fair head-to-head vs Unsafe-Net paper — plan (additive, non-destructive)

**Date:** 2026-09-02. Trigger: "modern models should beat the paper; re-review paper steps, re-plan,
re-run, evaluate on many metrics." This plan does NOT edit existing thesis files; all outputs go to
new files under this plan dir + new reports.

## Verified facts (do not re-litigate)
- Paper 95.81% = **macro over the 4 UNSAFE classes**, 64 unsafe test clips (paper.txt L340, Table 7).
  Our comparison is apples-to-apples 4-class. The 8-vs-4 question is CLOSED.
- Paper = 3 stages: (A) YOLOv4 behaviour-detector (nc=4, 2262 frames) used as frame-reduction +
  single-action split; (B) ConvLSTM classifier; (C) warning system.
- Paper's +10pp (85.78→95.81) is entirely the frame-reduction step (Table 7 vs Table 5-6).
- On 64 clips the benchmark cannot distinguish systems: val↔test macro r=−0.066 (p=0.71, 34 configs);
  62 vs 57 McNemar p≥0.0625 (n.s.). Chasing the 64-clip number is resampling, not research.

## The only valid way to claim "modern > paper": adequately-powered paired head-to-head

### Pipelines (same data, same splits)
- **P (paper-modern):** behaviour-CNN per-frame recogniser + frame-reduction (top-q by P(unsafe)) +
  temporal head. Faithful modern stand-in for YOLOv4+ConvLSTM. (code: `unsafenet/`)
- **O (ours):** YOLO26 entity detector → box-token spatial-relation Transformer. (code:
  `build_relation_tokens.py`, `run_relation_transformer.py`, `ensemble_relation.py`)

### Protocols
1. same-dist 64-clip (paper comparability) + BCa CI + McNemar(P vs O) + McNemar(each vs paper 57/62).
2. **stratified k-fold over pooled 516 clips** (tight CI; the protocol that can show a real diff) —
   Nadeau-Bengio corrected t-test P vs O.
3. **leave-camera-out** (honest cross-camera generalization) — per-view + pooled.

### Multi-metric scorecard (both pipelines, all protocols)
top-1 · macro-F1 · per-class recall · Cohen κ · MCC · ROC-AUC(ovr) · PR-AUC · ECE(15-bin) ·
corruption-robustness (already have `--corrupt`) · latency/FPS/energy (edge, Jetson script ready).

### Expected honest outcome
- same-dist top-1: statistical tie with paper (report as such, not a loss).
- macro-F1 / ECE / LCO / edge: ours likely wins — dimensions the paper never measured.
- Deliverable = a scorecard where "modern pipeline matches paper on its own noisy metric and wins on
  every rigor/robustness/deployment axis," backed by paired stats. Defensible, not p-hacked.

## Execution order
1. Confirm scope (headline protocol) with user. ← BLOCKING
2. Lock k-fold + LCO splits (seed-fixed, stratified on label×view). Reuse `run_kfold_relation.py`,
   `run_lco_eval.py`.
3. Run O across 3 protocols × multi-metric (mostly built; wire scorecard emitter).
4. Run P across same 3 protocols (unsafenet/ evaluate.py; add reduction ablation already present).
5. Paired stats + scorecard table → new report `reports/fair-headtohead-scorecard.md`.
6. (Opt) Jetson edge rows via `tools/jetson_benchmark.py`.

## Non-negotiables
- Blind protocol: epoch/config/seed/weights selected ONLY on train-carved val slice.
- Test/fold touched once per final config. No test-set model selection.
- Report measured numbers, including ties/losses. No fabrication, no test-tuning.

## Unresolved (for user)
- Headline protocol: k-fold vs LCO vs same-dist? (Recommend k-fold as headline, LCO as generalization,
  same-dist as paper-comparability footnote.)
- Include the faithful paper-modern pipeline P in the head-to-head, or evaluate O only vs paper's
  published 57/62? (Recommend include P for a fair same-code comparison.)
