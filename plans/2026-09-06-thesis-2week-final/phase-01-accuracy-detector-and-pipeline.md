---
title: "Phase 01 — Lane A: Accuracy, Detector Retrain, Pipeline Rebuild"
status: pending
---

# Phase 01 — Accuracy (Lane A: A1-A5)

**File ownership (exclusive):** `sweep_results/`, `runs/`, `cache_r2/`, `train_behaviour_detector.py`,
`run_kfold_relation.py`, `run_lco_eval.py`, `kaggle-detector-retrain/`, `colab/*yolo26*`,
`build_relation_tokens.py`, `tools/build_l1_tokens.py`, `run_relation_transformer.py`,
`run_yolo_framegate.py`, `unsafenet/behaviour_clip.py`, `run_operating_point.py`,
`run_ablation_calibration.py`, new `reports/test-touch-log.md` (repo-root `reports/`, does not
exist yet — create it). Serialized lane: one heavy GPU/CPU job at a time.

## Context links
- Binding brief: `plans/2026-09-06-thesis-2week-final/brief.md` §1, §2, §4h, §8 Gate 1/2
- `plans/2026-09-05-thesis-final-push/reports/t1-objectives-audit.md` — full run table, statistical
  power analysis, why-not-claim-(a) proof
- `plans/2026-09-02-fair-headtohead-vs-paper/reports/fair-headtohead-scorecard.md` — P3 k-fold/LCO
  numbers, LCO weak class
- `kaggle-detector-retrain/README.md` — detector-fix rationale (5/8 classes dead) and run recipe

## Overview
Priority P1. Status pending. Fix the CPU-trained, 3-dead-class behaviour detector on free GPU
(A1), replicate the paper's frame-reduction mechanism honestly (A2), rebuild the P3 relation-
Transformer on the improved detector and re-run all 3 protocols (A3), freeze one headline with one
test touch (A4), and add safety/calibration analysis (A5). Serves claim axes 1-6 (brief §2).

## Key insights (from research, do not re-derive)
- Current behaviour detector: mAP50 0.79 (tight) / 0.81 (ctx), trained CPU/yolo11s/416/25ep/batch8,
  emits boxes for only 5/8 classes (`Safe_Walkway`, `Restricted_Zone`, `Forklift_Safe_Load` dead).
  Root cause is a documented Ultralytics-MPS training bug forcing CPU, not the method.
- The paper's own bbox annotation scale is already matched here: 2,262 imgs / 2,245 labels,
  1,901 train / 343 val (`annotation_guide/dataset_behaviour_{tight,ctx}/`).
- P3 (YOLO26 entity + spatial-relation Transformer, `run_kfold_relation.py`/`run_lco_eval.py`) is
  the ONLY N>64 evidence in the project: k-fold 516 = 80.8±3.1% top-1/84.0% macro; LCO 516 = 78.6%
  macro vs scene-only baseline collapsing to 16.9%. This is the headline candidate, not P1/P2.
  P3 detector-generation ablation already shows YOLO11-vs-YOLO26 is within noise (81.2% vs 80.8%
  k-fold) — **A1's payoff is class coverage and mAP50, not swapping detector architecture.**
- Excluded by prior proof (brief §2, do not spend a day): further config/seed/ensemble search on
  the 64 clips (val<->test r=-0.07..-0.25, never significant); re-running P1/P2 fusion to chase
  55->56/64; the box-context ablation (settled negative, ctx +1.5pp mAP50 but no clip-decision win).
- LCO weak class: UnauthorInt recall 44.4% — `unsafenet/streams.py` already implements a
  motion/dwell stream (D_CLIP=10, walkway-vs-stationary cue) not yet wired into the relation tokens.
- Statistical ceiling (audit §3b): even a perfect 64/64 cannot beat paper's 62/64 at p<0.05
  (McNemar best-case p=0.50, Fisher p=0.496) — 2 discordant pairs is an information floor. Do not
  chase claim (a); claim (b) (statistical tie + rigor wins) is the recommended primary claim.

## Requirements
- Functional: mAP50 >=0.90 target, all 8 ontology classes emitting boxes; k-fold(516)+LCO(516)+
  same-dist(64) re-run on the improved-detector tokens; one declared test touch per pipeline variant.
- Non-functional: >=5 seeds per quoted number (brief §4h.5); every reported number carries n, CI,
  seed count, protocol tag; a failed run gets a control before an explanation (§4h.6).

## Architecture
`kaggle-detector-retrain/` (GPU, Kaggle/Colab) -> new detector weights -> `build_relation_tokens.py
--detector <new>.pt` (re-extract tokens, entity+geometry+optional appearance) -> `tools/
build_l1_tokens.py` (strip camera-fingerprint xc/yc) -> optionally fuse `unsafenet/streams.py`
motion/dwell vector -> `run_relation_transformer.py` (train) -> `run_kfold_relation.py` +
`run_lco_eval.py --scene_map` + same-dist eval (headline suite, phase-02's `eval_utils/`).
Parallel/independent: `unsafenet/behaviour_clip.py` + `run_yolo_framegate.py` (paper-mechanism
replication, feeds axis 2 only, does not gate A3).

## Related code files
- Modify/run: `train_behaviour_detector.py` (CPU fallback reference only — do not retrain here,
  GPU path is `kaggle-detector-retrain/train-detector-gpu.py` / `colab/kaggle_yolo26_onecell.py`)
- Run: `kaggle-detector-retrain/prepare-upload.sh`, `kaggle-detector-retrain/retrain-detector.ipynb`,
  `colab/train_yolo26_detector_kaggle.ipynb`, `colab/train_yolo26_detector_colab.ipynb`
- Run: `build_relation_tokens.py`, `tools/build_l1_tokens.py`, `unsafenet/streams.py`
- Modify: `run_relation_transformer.py` (wire streams.py motion vector as optional extra feature
  columns — additive, keep existing `--tag` mechanism), `run_kfold_relation.py`, `run_lco_eval.py`
- Run: `unsafenet/behaviour_clip.py` (frame_scores + det-gateQ), `run_yolo_framegate.py`
- Run: `run_operating_point.py`, `run_ablation_calibration.py`
- Create: `reports/test-touch-log.md` (repo root, new dir)
- Do not touch: `unsafenet/` round-2 fusion code (P1/P2) beyond reading — excluded by prior proof

## Implementation steps

### A1 — GPU detector retrain (day 1-2)
1. `bash kaggle-detector-retrain/prepare-upload.sh` -> assembles Kaggle payload from
   `annotation_guide/dataset_behaviour_{tight,ctx}/`.
2. Kaggle: run `kaggle-detector-retrain/retrain-detector.ipynb` (baseline arm, GPU T4) — yolo11s
   AND yolo26s, imgsz 640-800, 150-300 epochs, full aug (per brief §2 A1), both `tight` and `ctx`
   box variants. Alternatively `colab/kaggle_yolo26_onecell.py` (single-cell YOLO26, imgsz 800,
   2-stage freeze->finetune, already configured for nc=8).
3. Download `detector_{baseline}_best.pt` per variant; record per-class AP50 (`model.info()` /
   `benchmark()` output already printed by the notebook/one-cell script).
4. **Gate 1 check (end day 2):** mAP50 >= 0.90 AND all 8 classes emit boxes (verify via per-class
   AP50 table, a 0.0 AP50 row = dead class). Log result in `reports/test-touch-log.md` header
   (this is a val-set metric, not a test touch).
   - PASS -> proceed to A2/A3 with the new weights.
   - FAIL (mAP50 < 0.85 or classes still dead) -> **stop the detector attack**, keep existing P3
     k-fold(80.8%)/LCO(78.6%) as the headline, reallocate days 3-6 to Lane B/C/D. Document the
     failure + a control run (per §4h.6) in `reports/test-touch-log.md`. **This is a reportable
     result either way** — the thesis states which branch was taken and why.

### A2 — Behaviour-clip + frame-gate replication (day 3, if Gate 1 passed; else skip, report N/A)
5. `unsafenet/behaviour_clip.py::frame_scores(weights=<new_detector>.pt, ...)` — per-frame
   max-confidence cache (8 classes x T frames).
6. Run `run_yolo_framegate.py --weights <new_detector>.pt` — gates frames by conf, measures the
   frame-reduction step directly.
7. Instrument both scripts with `time.perf_counter()` around the per-clip inference call (brief
   §4a) to produce **action-recognition time per clip** (paper mark 0.14s) and **reduced average
   clip duration** (paper mark 1.87s, count kept-frames/original-fps). Report where these hold and
   where they diverge — do not force agreement.

### A3 — Relation-Transformer rebuild (day 3-5)
8. `build_relation_tokens.py --detector <new_detector>.pt --appearance --appearance_backbone yolo
   --out_tag <newtag>_` — re-extract tokens for train+test.
9. `tools/build_l1_tokens.py --src <newtag>_ --dst <newtag>L1_` — strip camera-fingerprint xc/yc.
10. Optionally fuse `unsafenet/streams.py` D_CLIP=10 motion/dwell vector into the L1 tokens
    (targets UnauthorInt LCO recall 44.4% -> higher; additive feature columns, not a new pipeline).
11. `run_relation_transformer.py --tag <newtag>L1_ --seeds 5 --epochs 120 --save_logits
    logits_<newtag>.pt` — >=5 seeds per brief §4h.5.
12. `run_kfold_relation.py --tag <newtag>L1_ --k 5 --seeds 5 --epochs 60` — pooled 516, Nadeau-
    Bengio vs scene-only.
13. `run_lco_eval.py --tag <newtag>L1_ --seeds 5 --epochs 60 --scene_map <camera-group.json>` —
    scene-robust, primary generalisation evidence.

### A4 — Headline freeze + single test touch (day 6)
14. Pick ONE pipeline variant (new-detector P3, or fallback existing P3 if Gate 1 failed) as THE
    headline. Pre-register the configuration (detector weights, token tag, seed count, epoch) in
    `reports/test-touch-log.md` BEFORE touching test.
15. Run the phase-02 `eval_utils` full §4 suite (top-1, macro-F1, balanced acc, per-class P/R,
    kappa, MCC, confusion, FNR/miss-rate, cost-sensitive score, ROC/PR-AUC, threshold sweep, ECE +
    reliability diagram, bootstrap 5000 + Wilson/BCa, McNemar exact, Nadeau-Bengio) across all 3
    protocols (same-dist 64, k-fold 516, LCO 516). One touch, timestamp-logged.
16. No further accuracy experiments after this step (Gate 2, brief §8).

### A5 — Safety + calibration analysis (day 6, parallel with A4 writeup)
17. `run_operating_point.py --logits logits_<newtag>.pt` — threshold sweep, safety-recall operating
    point, per-class FNR.
18. `run_ablation_calibration.py` (adapt `FEAT`/`CKPT` paths to the new pipeline's saved features) —
    ECE + reliability diagram, temperature sweep.
19. Declare the cost matrix (miss >> false-alarm) BEFORE reading results (brief §4c) — write it into
    `reports/test-touch-log.md` alongside the A4 pre-registration.

### Buffer (day 7)
20. Contingency: finish any slipped A1-A5 step; ablation table + error analysis if on schedule.

## Todo list
- [ ] A1: Kaggle/Colab GPU retrain launched, both tight+ctx, yolo11s+yolo26s
- [ ] Gate 1 evaluated and logged (pass or documented fallback)
- [ ] A2: 0.14s/clip and 1.87s duration measured with `time.perf_counter()`
- [ ] A3: tokens re-extracted, L1-stripped, streams.py motion fused (if pursued)
- [ ] A3: relation-Transformer retrained >=5 seeds; k-fold 516 + LCO 516 + same-dist 64 re-run
- [ ] A4: headline pre-registered in `reports/test-touch-log.md` before touch
- [ ] A4: full §4 suite run on headline across 3 protocols, one touch each
- [ ] A5: cost matrix declared before results; FNR, threshold sweep, ECE, reliability diagram done
- [ ] Gate 2 confirmed: no further accuracy experiments after day 6

## Success criteria
- Gate 1: mAP50 >= 0.90 AND 8/8 classes with AP50 > 0 (artifact: Kaggle/Colab output CSV/log).
- A2: `time.perf_counter()` measurements written to a results file (path TBD by implementer, e.g.
  `sweep_results/framegate_timing.json`) with per-clip mean/median vs paper's 0.14s and 1.87s.
- A3: `sweep_results/relation_tokens/tokens_<newtag>L1_{train,test}.pt` exist; k-fold macro with
  ±CI narrower or equal to current ±3.1pp; LCO macro reported with per-class recall table.
- A4: exactly one entry per pipeline variant in `reports/test-touch-log.md` with timestamp; all
  §4a-h numbers present for the headline (no missing cell in the reporting table).
- A5: cost matrix committed to `reports/test-touch-log.md` with a timestamp earlier than the A4
  test-touch timestamp; ECE numeric value + reliability diagram image path recorded.

## Risk assessment
- **A1 GPU access unavailable/slow**: Kaggle 12h session cap — mitigate by `--epochs2` reduction or
  imgsz 800->640 per `kaggle-detector-retrain/README.md`. Falls into Gate 1 FAIL branch, which is
  pre-planned (reallocate to B/C/D), not a crisis.
  Note: even in the FAIL branch, the current best P3 evidence is retained as headline — Gate 1 has
  no path that leaves the thesis without an accuracy chapter.
- **A3 motion/dwell fusion (streams.py) regresses k-fold while improving LCO**: per
  `kaggle-detector-retrain/README.md`'s own rule — k-fold up / LCO down = reject. Apply symmetric
  rule: LCO up / k-fold materially down = also needs justification, report both directions honestly.
- **Test-touch discipline slip**: two people (this lane + phase-02 eval author) touching test
  independently would double-count touches. Mitigate: only the pre-registered headline (step 14)
  may call the phase-02 test-set evaluation function; phase-02 delivers library code only.

## Security considerations
No secrets/PII in this lane. Kaggle/Colab uploads (`kaggle-detector-retrain/upload/`) contain only
frame images + YOLO labels, already used in prior uploads — no new exposure surface.

## Next steps
- A4 output (headline logits/tokens + `reports/test-touch-log.md`) is the hard input to phase-04
  Ch4 and phase-06 slides — both are blocked on this phase's day-6 completion.
- A3's exported detector weights and tokens feed phase-03's B1 TRT export (new headline pipeline
  must be re-exported, not the old P1 heads currently benchmarked on Jetson).

## Unresolved questions
1. Whether k-fold/LCO (P3, currently reported as its own pipeline outside `unsafenet/`) becomes the
   SOLE thesis headline, or P1/P2/P3 are reported side by side — a committee-framing call not
   resolved in the brief (t1-audit unresolved q4). Recommend: P3 headline, P1/P2 as historical
   ladder in Ch4 negative-results section.
2. Exact camera-group JSON for `run_lco_eval.py --scene_map` on the NEW detector's tokens — does the
   existing `clip_view_map.json` / zone view files still apply, or must it be regenerated per
   `kaggle-detector-retrain/README.md`'s zone-arm caveat (only 1550/2262 images have a view)?
3. Whether `unsafenet/streams.py` motion/dwell fusion is in-scope for the 2-week window given it is
   additive R&D on top of an already-tight day-3-5 window — flagged as a stretch, not a hard gate.
