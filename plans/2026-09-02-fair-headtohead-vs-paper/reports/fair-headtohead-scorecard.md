# Fair head-to-head scorecard — Ours (YOLO11/26 + Transformer) vs Unsafe-Net paper

**Date:** 2026-09-02. Additive (no thesis files edited). Answers: "do modern models beat the paper?" —
run under the ONLY statistically valid protocol (adequately-powered), across many metrics.
Pipeline O only (user decision); compared to paper's PUBLISHED numbers. Runs: `run_kfold_relation.py`,
`run_lco_eval.py` (seeds=3, epochs=60, pooled 516 clips), same-dist from `logits_yolo26.pt`.

> **CORRECTION NOTICE (2026-09-06).** Three claims in this 2026-09-02 scorecard are superseded; see
> `plans/2026-09-06-thesis-2week-final/brief.md` (S2) and `reports/qa1-adversarial-review-and-defence.md`.
> 1. **ECE 0.101 is mis-attributed.** It was computed from the **P1 pairwise-cascade ensemble (56/64)**,
>    not the P3 pipeline whose top-1/macro fill the rest of that column. Do not cite it alongside 54/64.
>    **Superseded 2026-09-06 — now fixed at source.** P3's generating command was recovered and verified
>    bit-exact, so the blind seed's own softmax exists. The correct P3 blind-basis values, coherent with
>    the 54/64 in the same column, are **ECE(15) 0.097 · ROC-AUC 0.951 · PR-AUC 0.880**. Use these; the
>    0.950/0.869/0.101 triple (P3 ensemble, 52/64) and the 0.101 above (P1) are both retired.
> 2. **"scene-only collapses to 16.9 % (+60.3 pp)" is WITHDRAWN — now proven, not just argued.**
>    Under LCO that baseline is a **constant global-majority predictor**, not the per-view-majority
>    estimator used in k-fold. The rebuilt `prior` row scores **16.5 %**, reproducing the old "16.9 %"
>    to within rounding: the old baseline *was* the constant.
>    **Replacement (canonical, use this):** against a transferable scene-appearance baseline
>    (logreg on whole-frame MobileNetV3 features, fit on training cameras only), the honest margin is
>    **+16.9 pp macro, 95 % bootstrap CI [+12.5, +21.2]**, n_boot = 5 000.
> 3. **LCO values are now LOCKED** by the canonical 5-seed run
>    (`sweep_results/relation_tokens/canonical/lco_seeds5_yolo26L1_CANONICAL.log`):
>    ours **77.7 % top-1 / 81.2 % macro**; scene-appearance 53.1 % / 64.4 %; constant floor 16.5 % / 10.1 %.
>    All earlier LCO figures (78.6, 79.7, 82.0, 75.6 %) were 3-seed and are superseded.
>    Per-class recall ours vs scene-appearance: OpenedPanel 88.7/88.0 and CarryOverload 100/100 **tie**,
>    while WalkwayViol 70.5/17.6 and UnauthorInt 65.7/51.9 separate — the mechanism claim, visible.
>
> 4. **The k-fold YOLO11 NB p-value below (0.0375, "sig") is SUPERSEDED.** A canonical
>    `--repeats 10` run (50 folds, df=49; archived at
>    `sweep_results/relation_tokens/canonical/kfold_repeats10_yoloapp_YOLO11.log`) gives
>    **mean Δ = 3.6 pp, t = 1.95, p = 0.0574 - not significant.** Both 0.0375 and 0.10 were
>    single runs at df = 4. Use 0.0574 and call it n.s.; do not write "marginally significant".
>    Canonical YOLO11 k-fold also supersedes the row values: top-1 80.7 ± 3.6 %, macro 85.1 %,
>    macro-F1 83.9 %, κ 0.744, MCC 0.749, NB 95 % CI 77.0-84.4 %.
> 5. **Any k-fold CI here computed without the NB correction is ~3.7x too narrow** (naive SE 0.51
>    vs NB-corrected 1.87 on the canonical run).

## 0. Task parity confirmed (paper.txt L340, Table 7)
Paper 95.81% = macro over the **4 UNSAFE classes**, 64 unsafe test clips. Our eval is identical 4-class.
The 8-vs-4 question is CLOSED — the gap is not a task-definition artifact.

## 1. Scorecard — Ours (YOLO26) across 3 protocols vs paper

| Metric | same-dist 64-clip (ours, blind) | **k-fold 516 (headline)** | LCO 516 (scene-robust) | Paper (published) |
|---|---|---|---|---|
| top-1 | 54/64 = 84.4% | **80.8 ± 4.0%** fold-mean (pooled 421/516 = 81.6%) | **77.7% (401/516)** | 57/64 (nobbox) · 62/64 (full) |
| macro | **88.4%** | **84.0%** | **81.2%** | 85.78% (nobbox) · 95.81% (full) |
| macro-F1 | — | 83.0% | — | not reported |
| Cohen κ | 0.734 | 0.731 | — | not reported |
| MCC | 0.751 | 0.735 | — | not reported |
| ROC-AUC (ovr) | 0.951 | — | — | not reported |
| PR-AUC | 0.880 | — | — | not reported |
| **ECE (calibration)** | **0.097** | — | — | not reported |
| scene baseline | 52/64 (81.2%) | 77.1% (per-view majority) | 53.1% / 64.4% macro (scene-appearance) | never tested |
| behaviour signal vs scene | +3.2 pp | **+3.7 pp (NB p=0.0600, n.s., 50 folds)** | **+16.9 pp macro, CI [+12.5,+21.2]** | — |
| CI width (test size) | ±11 pp (64 clips) | **±3 pp (516 clips)** | tight (516) | ±11 pp (64 clips) |

YOLO11 ablation (k-fold): 419/516 = 81.2%, macro 84.5%, F1 83.4%, κ 0.737, MCC 0.741, NB p=0.0375 (sig).
**[SUPERSEDED — canonical 50-fold runs: YOLO11 80.7±3.6% top-1 / macro 85.1% / p=0.0574 (n.s.);
YOLO26 80.8±4.0% / macro 84.2% / p=0.0600 (n.s.).]**
→ Detector generation (YOLO11 vs YOLO26) is within noise — pipeline is NOT detector-limited. This now
rests on 50 folds per track with overlapping NB CIs, not on the single runs it originally rested on.

LCO per-class recall (YOLO26): Walkway 77.1 · UnauthorInt 44.4 · OpenedPanel 93.0 · CarryOverload 100.

## 2. Did we beat the paper? — honest verdict per axis

**Where the paper is (nominally) ahead — same-dist top-1, and only there:**
- Ours 54/64 vs full-bbox 62/64. Unpaired two-proportion z≈−2.4 (p≈0.016) — nominally against us,
  BUT the correct paired McNemar cannot be run (paper's per-clip labels unreleased), and the paper's
  OWN 62-vs-57 ablation is n.s. (McNemar p≥0.0625). The +10pp rests entirely on frame-reduction, which
  is **unreproducible on the released single-action clips** (direct ablation: −0.8 to +1.5 pp = zero).
- vs no-bbox baseline (57/64): Δ=−4.7pp, z=−0.78, **p=0.435 (n.s.)** → statistical tie on top-1.

**Where WE win (defensibly, on axes the paper never measured):**
1. **Macro on same-dist:** 88.4% > paper no-bbox 85.78% (blind, honest protocol).
2. **Statistical reliability:** k-fold macro 84.0% ± 3pp over 516 clips vs paper's single 64-clip point
   (±11pp). Our estimate is ~3.5× tighter and reproducible — the paper never provided one.
3. **Scene-robust generalization (LCO):** **81.2% macro vs a transferable scene-appearance baseline's
   64.4% — +16.9 pp, 95% bootstrap CI [+12.5, +21.2]** (canonical 5-seed run). The mechanism is visible
   per-class: object-visible behaviours **tie** (OpenedPanel 88.7/88.0, CarryOverload 100/100) while the
   spatially-defined pair separates (WalkwayViol 70.5/17.6, UnauthorInt 65.7/51.9).
   The paper NEVER ran cross-camera — its 95.81% does not measure behaviour generalization. Note this is
   a claim about the *protocol*, not a prediction about their model, which cannot be run (weights
   unreleased). Caveat: the 8 "cameras" are a KMeans(8) proxy; Unsafe-Net publishes no camera IDs.
4. **Calibration:** ECE 0.097 (blind basis) — paper reports none.
5. **Edge deployment:** measured M4 + Jetson (Ch5) — paper reports none.

## 3. The one finding that reframes everything (verified this round + R2)
Random k-fold: scene-only baseline is already **77.1%** (every camera in train → scene leakage), so the
behaviour model adds only **+3.7 pp — NB p = 0.0600, n.s.** over 50 folds (canonical, YOLO26 headline;
the YOLO11 ablation independently gives p = 0.0574). Under LCO, where
camera identity is structurally unavailable, the same model leads a transferable scene baseline by
**+16.9 pp macro with a CI clearing zero**. That contrast — n.s. under the confounded protocol,
solidly positive under the honest one — *is* the scene-confound, measured from both sides. Same-dist and
random k-fold rank systems substantially by camera memorization; only LCO isolates behaviour.

*(An earlier version of this section claimed +60.3 pp under LCO. Withdrawn — that compared the model
against a constant predictor. The honest figure is +16.9 pp. Smaller, real, and it needs no inflating.)*

→ "Do modern models beat the paper?" On the paper's own noisy same-dist metric: **statistical tie**
(neither side's advantage is significant on 64 clips). On rigor, calibration, and scene-robust
generalization: **yes, on axes the paper never measured** — a modest but statistically clean
cross-camera margin, tighter interval estimates, and measured edge deployment. Not "decisively"; the
evidence does not support that word, and the thesis is stronger for not using it.

## 4. What this is NOT
Not a p-hacked 64-clip number. We deliberately did NOT tune on test to manufacture a >62/64. The
honest, higher-power evidence (k-fold + LCO + calibration) is a stronger thesis claim than a
noise-band 64-clip win.

## Unresolved questions
1. Report k-fold (84.0% macro, tight CI) as the headline accuracy number in the thesis, demoting the
   64-clip comparison to a paper-comparability footnote? (Recommended.)
2. LCO UnauthorInt recall 44.4% is the weak class cross-camera — worth a targeted motion/dwell stream
   (R2 `unsafenet/streams.py` has it) to lift it, or accept as measured?
3. Paper's per-clip test labels are unreleased → paired McNemar vs paper impossible; state as a
   benchmark-limitation in the thesis?
