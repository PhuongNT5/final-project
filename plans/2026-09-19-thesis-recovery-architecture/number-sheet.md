# Number sheet — single source of truth for every headline number

Phase-09 deliverable, created 2026-09-20 (skeleton + first measured rows). **NUMBER FREEZE is
2026-10-05**; until then every row is `measured` or `placeholder`, never `frozen`, and nothing here
may be propagated into `report/**`, `README.md`, `PROGRESS.md` or `docs/**`.

Rules this sheet enforces:
- **Protocol is its own column.** `P-C LCO-full` and `P-C LCO-panel-disclosed` never share a row; the
  same applies to `P-D LSO-1100` (iSafetyBench) vs anything on Unsafe-Net. Two corpora, two blocks,
  never joined into one table in prose.
- Every value is verified by opening its `log_path` and reading the line — never from memory.
- `test_touch_cost` is 0 unless the row was produced by scoring the frozen 125-clip blind split.
  Running total of touches spent: **0 of 2**.

## Block 1 — iSafetyBench accuracy lane (P-D LSO-1100, grouped by YouTube source)

1,100 clips / 420 hazard / 151 sources, `StratifiedGroupKFold(10)`, seed 0, 3 seeds/fold, 60 epochs.

| metric | protocol | value | 95% CI / range | token_tag | script | log_path | date | touch | status |
|---|---|---|---|---|---|---|---|---|---|
| balanced accuracy, E3 | P-D LSO-1100 | 75.2% | per-fold **top-1** 50.9–87.5; per-fold **BA** 57.3–91.2 | `isb_` | `run_kfold_relation.py --lco --dataset isafetybench --arch E3` | `sweep_results/relation_tokens/logs/lso-isb-E3.log:15` | 2026-09-20 | 0 | measured |
| macro F1, E3 | P-D LSO-1100 | 73.9% | — | `isb_` | same | `…/lso-isb-E3.log:15` | 2026-09-20 | 0 | measured |
| hazard recall, E3 | P-D LSO-1100 | 78.3% | — | `isb_` | same | `…/lso-isb-E3.log:15` | 2026-09-20 | 0 | measured |
| hazard precision, E3 | P-D LSO-1100 | 63.4% | — | `isb_` | same | `…/lso-isb-E3.log:15` | 2026-09-20 | 0 | measured |
| ROC-AUC, E3 | P-D LSO-1100 | 0.815 | — | `isb_` | same | `…/lso-isb-E3.log:15` | 2026-09-20 | 0 | measured |
| PR-AUC (hazard), E3 | P-D LSO-1100 | 0.692 | — | `isb_` | same | `…/lso-isb-E3.log:15` | 2026-09-20 | 0 | measured |
| **R42 count-rule floor** | P-D LSO-1100, identical folds | **67.2%** | — | `isb_` | same run, floor block | `…/lso-isb-E3.log:16` | 2026-09-20 | 0 | measured |
| R42 floor, random stratified 10-fold | control (not a claim) | 68.2% | — | `isb_` | `isafetybench_dataset.count_rule_oof` | feasibility report §floor | 2026-09-20 | 0 | measured |
| **encoder − floor (G10 margin)** | P-D LSO-1100 | **+8.0 pp** | **cluster bootstrap over 151 sources [+1.9, +13.5]** | `isb_` | `eval_utils/cluster_bootstrap.py` | G10 finding report §3 | 2026-09-20 | 0 | measured |
| same margin, clip-level bootstrap | WRONG UNIT — disclosure only | +8.0 pp | [+4.6, +11.4] | `isb_` | `eval_utils/paired_tests.py` | G10 finding report §3 | 2026-09-20 | 0 | measured |
| McNemar exact, encoder vs floor | P-D LSO-1100 | p = 1.02e-05 | b=211 c=129, 340 discordant | `isb_` | `eval_utils/paired_tests.mcnemar_exact` | G10 finding report §3 | 2026-09-20 | 0 | measured |
| **balanced accuracy, E0** | P-D LSO-1100 | **80.3%** | macro-F1 79.0, rec 84.0, prec 68.9, PR-AUC 0.747, ROC-AUC 0.857; floor 67.2% -> +13.1 pp | `isb_` | `… --arch E0` | `…/lso-isb-E0.log:15` | 2026-09-20 | 0 | measured |
| **paired Δ, E0 − whole-frame** | P-D LSO-1100 | **+6.2 pp** | cluster CI [+2.1, +10.5] excludes 0, McNemar p=1.59e-05 | `isb_` vs `wf_isb_` | `tools/compare-oof-runs.py` | G10 finding §10 | 2026-09-20 | 0 | **clears G11 arithmetic but G11 named E3 — NOT a pass** |
| **paired Δ, E0 − E3** | P-D LSO-1100 | **+5.1 pp** | cluster CI [+2.0, +8.3] excludes 0, McNemar p=7.74e-06 | `isb_` | `tools/compare-oof-runs.py` | G10 finding §10c | 2026-09-20 | 0 | **R47: architecture invariance falsified** |
| E3 `--ablate no_conf` | P-D LSO-1100 | 76.1% | floor 53.0%, margin +23.2 [+15.3, +31.4] | `isb_` | `… --ablate no_conf` | `sweep_results/relation_tokens/logs/lso-isb-E3-ablations.log` | 2026-09-20 | 0 | measured |
| E3 `--ablate no_appearance` | P-D LSO-1100 | 73.8% | floor 67.2%, margin +6.5 [+1.8, +11.7] | `isb_` | `… --ablate no_appearance` | `sweep_results/relation_tokens/logs/lso-isb-E3-ablations.log` | 2026-09-20 | 0 | measured |
| E3 `--ablate blind_class` | P-D LSO-1100 | 76.6% | floor 68.1%, margin +8.5 [+2.2, +14.0] | `isb_` | `… --ablate blind_class` | `sweep_results/relation_tokens/logs/lso-isb-E3-ablations.log` | 2026-09-20 | 0 | measured |
| E3 `--ablate no_boxes` | P-D LSO-1100 | 76.2% | floor 67.2%, margin +9.0 [+2.4, +15.0] | `isb_` | `… --ablate no_boxes` | `…/lso-isb-E3-ablations-2.log` | 2026-09-20 | 0 | measured |
| E3 `--ablate boxes_only` | P-D LSO-1100 | 64.2% | floor 47.4%, margin +16.8 [+7.3, +26.6] | `isb_` | `… --ablate boxes_only` | `…/lso-isb-E3-ablations-2.log` | 2026-09-20 | 0 | measured |
| **E3 `--ablate appearance_only`** | P-D LSO-1100 | **75.6%** | floor 47.4%, margin +28.3 [+19.5, +37.0] | `isb_` | `… --ablate appearance_only` | `…/lso-isb-E3-appearance-only.log` | 2026-09-20 | 0 | measured |
| paired Δ, full − appearance_only | P-D LSO-1100 | −0.4 pp | [−2.9, +2.4], McNemar p=0.551 | `isb_` | `tools/compare-oof-runs.py` | G10 finding report §5 | 2026-09-20 | 0 | measured |
| paired Δ, full − boxes_only | P-D LSO-1100 | +11.0 pp | [+3.4, +18.8], McNemar p=2.6e-13 | `isb_` | `tools/compare-oof-runs.py` | G10 finding report §5 | 2026-09-20 | 0 | measured |
| **G11 whole-frame baseline (best probe = linear-mean)** | P-D LSO-1100, no detector | **74.1%** | other probes: meanstd 72.7%, mlp 73.6% | `wf_isb_` | `run_wholeframe_baseline.py` | `…/logs/g11-wholeframe-baseline.log:4` | 2026-09-20 | 0 | measured |
| **paired Δ, E3 − whole-frame (G11)** | P-D LSO-1100 | **+1.1 pp** | cluster CI **[−4.8, +6.9] includes 0**, McNemar p=0.625 (b=138 c=129) | `isb_` vs `wf_isb_` | `tools/compare-oof-runs.py` | G10 finding report §9 | 2026-09-20 | 0 | **G11 FAILS both criteria** |
| **balanced accuracy, E1 d64** (93,222) | P-D LSO-1100 | **78.1%** | floor 67.2% -> +10.9 pp | `isb_` | `… --arch E1 --dim 64` | `sweep_results/relation_tokens/logs/g13-lso-isb-E1-d64.log:17` | 2026-09-21 | 0 | measured |
| **balanced accuracy, E1 d96** (195,558) | P-D LSO-1100 | **77.0%** | 1.1 pp BELOW E1 d64 at 2x params — the G13 inversion, must be disclosed | `isb_` | `… --arch E1 --dim 96` | `sweep_results/relation_tokens/logs/g13-lso-isb-E1-d96.log:17` | 2026-09-21 | 0 | measured |
| **capacity spread across a 17x ladder** | P-D LSO-1100 | **5.1 pp** | E3 75.2 / E1d64 78.1 / E1d96 77.0 / E0 80.3 | `isb_` | `tools/protocol-inflation-table.py` | `sweep_results/relation_tokens/logs/g12-protocol-inflation-table.log` | 2026-09-21 | 0 | measured |
| balanced accuracy, E3 | **RAND-1100 — SOURCE-LEAKING PROTOCOL, shown for comparison** | 90.6% | floor 68.2% -> +22.4 pp | `isb_` | `… --split random --arch E3` | `sweep_results/relation_tokens/logs/g12-rand-isb-E3.log:15` | 2026-09-21 | 0 | **diagnostic artefact, never a headline** |
| balanced accuracy, E1 d64 | RAND-1100 — source-leaking, for comparison | 91.3% | floor 68.2% | `isb_` | `… --split random` | `sweep_results/relation_tokens/logs/g12-rand-isb-E1-d64.log:17` | 2026-09-21 | 0 | diagnostic artefact |
| balanced accuracy, E1 d96 | RAND-1100 — source-leaking, for comparison | 91.9% | floor 68.2% | `isb_` | `… --split random` | `sweep_results/relation_tokens/logs/g12-rand-isb-E1-d96.log:17` | 2026-09-21 | 0 | diagnostic artefact |
| balanced accuracy, E0 | RAND-1100 — source-leaking, for comparison | 90.7% | floor 68.2% -> +22.5 pp | `isb_` | `… --split random --arch E0` | `sweep_results/relation_tokens/logs/g12-rand-isb-E0.log:15` | 2026-09-21 | 0 | diagnostic artefact |
| **capacity spread, same 17x ladder** | RAND-1100 leaking | **1.4 pp** | vs 5.1 pp under LSO — the protocol loses the power to rank encoders | `isb_` | `tools/protocol-inflation-table.py` | `sweep_results/relation_tokens/logs/g12-protocol-inflation-table.log` | 2026-09-21 | 0 | measured |
| **Δ_inflation, E3 (G12)** | RAND minus LSO, paired | **+15.4 pp** | cluster CI over 151 sources [+10.6, +21.5] excludes 0; McNemar b=186 c=8 p=3.58e-45 | `isb_` | `tools/compare-oof-runs.py` | `sweep_results/relation_tokens/logs/g12-inflation-paired.log` | 2026-09-21 | 0 | **G12 PASSES** |
| **Δ_inflation, E0 (G12)** | RAND minus LSO, paired | **+10.4 pp** | [+6.3, +15.3] excludes 0; McNemar b=150 c=22 p=1.3e-24 | `isb_` | same | `sweep_results/relation_tokens/logs/g12-inflation-paired.log` | 2026-09-21 | 0 | **G12 PASSES** |
| **Δ_inflation, R42 count-rule floor** | RAND minus LSO, control | **+1.0 pp** | 67.2% -> 68.2%; 3 features cannot memorise source identity — this is the mechanism control | `isb_` | `isafetybench_dataset.count_rule_oof` | `reports/finding-2026-09-21-protocol-inflation-and-the-capacity-contrast.md §5` | 2026-09-21 | 0 | measured |
| **paired Δ, E0 − E3 under the LEAKING protocol** | RAND-1100 | **+0.1 pp** | cluster CI [−1.2, +1.5] includes 0, McNemar p=0.665 — vs +5.1 pp p=7.74e-06 under LSO. **The headline: leakage destroys discriminative power.** | `isb_` | `tools/compare-oof-runs.py` | `sweep_results/relation_tokens/logs/g13-capacity-contrast.log` | 2026-09-21 | 0 | measured |
| widest pair on the leaking ladder (E1 d96 − E3) | RAND-1100 | +1.4 pp | [−0.4, +3.0] includes 0, McNemar p=0.105 — nothing on the random ladder is significant | `isb_` | same | `reports/finding-2026-09-21-protocol-inflation-and-the-capacity-contrast.md §6` | 2026-09-21 | 0 | measured |

### Block 1b — R48 CLIP-representation lane (P-D LSO-1100, same folds, same clips, same recipe)

Added 2026-09-29. All five rows are the **3-seed ensemble pooled OOF** over the identical
`StratifiedGroupKFold(10, seed=0)` partition as every row above, joined on `clip_id`; alignment is
asserted by `run-esta-lso.load_aligned`, which aborts rather than score a different corpus.

| metric | protocol | value | 95% CI / range | script | log_path | date | touch | status |
|---|---|---|---|---|---|---|---|---|
| **balanced accuracy, B1w** (333,570) — whole-frame CLIP, 2 temporal layers | P-D LSO-1100 | **84.4%** | macro-F1 83.6, rec 84.8, prec 76.6, PR-AUC 0.801, ROC-AUC 0.884 | `run-esta-lso.py --model B1w` | `…/logs/r48-lso-B1w.log` | 2026-09-29 | 0 | measured |
| **balanced accuracy, B1** (201,090) — whole-frame CLIP, 1 temporal layer | P-D LSO-1100 | **84.2%** | macro-F1 83.5, rec 84.5, prec 76.3, PR-AUC 0.792, ROC-AUC 0.897 | `run-esta-lso.py --model B1` | `…/logs/r48-lso-B1.log` | 2026-09-29 | 0 | measured |
| **balanced accuracy, ESTA** (352,258) — entity CLIP, factorised space-time | P-D LSO-1100 | **83.8%** | macro-F1 83.1, rec 84.3, prec 75.8, PR-AUC 0.760, ROC-AUC 0.895 | `run-esta-lso.py --model ESTA` (4 shards, merged) | `…/logs/r48-lso-ESTA-shard{0..3}.log` | 2026-09-29 | 0 | measured |
| **balanced accuracy, B2** (233,858) — VideoMAE-base Kinetics-400 | P-D LSO-1100 | **81.5%** | macro-F1 80.7, rec 82.1, prec 72.6, PR-AUC 0.763, ROC-AUC 0.875 | `run-esta-lso.py --model B2 --feats …isbvmae…` | `…/logs/r48-lso-B2.log` | 2026-09-29 | 0 | measured |
| **paired Δ, ESTA − B1w (G14, THE GATE)** | P-D LSO-1100 | **−0.5 pp** | cluster CI **[−5.5, +5.0] includes 0**, McNemar b=72 c=78 p=0.683 | `tools/adjudicate-r48-gates.py` | `…/logs/r48-adjudication.log` | 2026-09-29 | 0 | **G14 FAILS — entity decomposition adds nothing** |
| paired Δ, ESTA − E0@64d (G15) | P-D LSO-1100 | +3.5 pp | cluster CI **[−2.3, +10.0] includes 0**, McNemar p=0.00148 | same | same | 2026-09-29 | 0 | **G15 FAILS — margin clears, CI does not** |
| paired Δ, ESTA − B2 (G16) | P-D LSO-1100 | +2.3 pp | [−2.3, +7.5] includes 0, McNemar p=0.0828 | same | same | 2026-09-29 | 0 | no separation; B2 does not beat ESTA |
| **paired Δ, B1w − E0@64d** (representation swap) | P-D LSO-1100 | **+4.0 pp** | cluster CI **[−0.9, +8.8] includes 0**; clip-level [+1.4, +6.7] — WRONG UNIT, disclosure only; McNemar b=131 c=78 p=3.01e-04 | `tools/compare-oof-runs.py` | — | 2026-09-29 | 0 | **the R48 result; CI does not certify the margin** |
| paired Δ, B1 − E0@64d | P-D LSO-1100 | +3.8 pp | [−1.0, +8.6] includes 0, McNemar p=4.51e-04 | same | — | 2026-09-29 | 0 | measured |
| paired Δ, B1w − B2 | P-D LSO-1100 | +2.9 pp | [−1.8, +7.0] includes 0, McNemar p=0.0178 | same | — | 2026-09-29 | 0 | measured |
| paired Δ, B2 − E0@64d | P-D LSO-1100 | +1.2 pp | [−2.9, +5.3] includes 0, McNemar p=0.184 | same | — | 2026-09-29 | 0 | measured |

| **balanced accuracy, ESTA @ YOLO26s (G17)** | P-D LSO-1100 | **81.6%** | macro-F1 80.3, rec 85.0, prec 70.7, PR-AUC 0.737, ROC-AUC 0.866 | `run-esta-lso.py --model ESTA --feats …isbclip26… --tag_suffix _y26` | `…/logs/r48-lso-ESTAy26-shard{0..3}.log` | 2026-09-29 | 0 | measured |
| **paired Δ, ESTA@yolo11s − ESTA@yolo26s (G17)** | P-D LSO-1100 | **−2.2 pp for the newer detector** | cluster CI [−1.7, +6.0] includes 0, McNemar b=75 c=43 p=0.00412 | `tools/compare-oof-runs.py` | — | 2026-09-29 | 0 | **newer detector generation is WORSE here; G14 is not detector-limited** |
| paired Δ, B1w − ESTA@yolo26s | P-D LSO-1100 | +2.7 pp | [−3.2, +8.8] includes 0, McNemar p=0.0044 | same | — | 2026-09-29 | 0 | measured |

**TSF (two-stream fusion, G18) added 2026-09-30 — FAILED.** 686,726 params, **84.1 %** balanced
accuracy (macro-F1 83.1, unsafe recall 86.0, unsafe precision 74.9, PR-AUC 0.796, ROC-AUC 0.901).
Against B1w: **-0.3 pp**, cluster CI [-3.7, +2.5], McNemar b=22 c=29 **p=0.401**. Against its own
entity arm: +0.2 pp, b=81 c=82, **p=1.000**. Note the discordance collapse — ESTA vs B1w disagreed
on 150 clips, TSF vs B1w on 51 — the fusion converged onto the whole-frame stream rather than
combining two views. **Bans:** (a) TSF's +1.4 pp over B1xl may not be quoted as evidence for the
fusion (registered as a condition on a passing claim; there is no passing claim); (b) the 86.0 %
unsafe recall may not be quoted as an improvement — balanced accuracy is the lane's metric and it
is down; (c) the banned complementarity-sweep fusion numbers remain banned and are NOT what this
row measures.

**Capacity control added 2026-09-30 (B1xl, pre-registered before the TSF result was visible).**
A whole-frame model at **696,962 params** — 2.09x B1w, and 1.5 % larger than the TSF fusion model —
scores **82.7 %** balanced accuracy (macro-F1 82.0, unsafe recall 82.6, PR-AUC 0.836, ROC-AUC
0.893). That is **1.7 pp BELOW B1w's 84.4 % at 333,570 params.** Scaling the whole-frame arm does
not buy accuracy in this lane; it costs some. Together with B1 (201,090) -> B1w (333,570) moving
only +0.2 pp, capacity is ruled out as the mechanism across a 3.5x parameter range. Any later claim
that a larger model wins here must explain why the two pre-registered capacity controls both say
otherwise.

**Standing caveat on every Block 1b row:** at 151 sources with 0 carrying both classes (R41),
**every cluster CI in this lane includes 0** while the clip-level CIs exclude it. No 3–4 pp margin
is certifiable on iSafetyBench at cluster resolution. Quoting a clip-level CI as the headline is
forbidden — it is the same protocol inflation Block 1 already documents against this project.

### Block 1c — R48 edge cost (host CPU, **NOT Jetson** — must never be quoted as a device number)

Measured 2026-09-29 by `deployment/scripts/benchmark-r48-clip-pipeline.py`, onnxruntime
CPUExecutionProvider on arm64 Darwin 23.6.0, 20 runs after 3 warmups, median and p95. One clip =
**16 frames** through the tower + one head call — the exact evaluated configuration (T=16,
whole-frame, **no detector in B1's accuracy path**).

| stage | graph | params | median | p95 |
|---|---|---:|---:|---:|
| CLIP ViT-B/32 vision tower, batch 16 | `r48-clip-vit-b32-vision-fp32.onnx` | 87,849,216 | **375.0 ms** | 392.9 ms |
| B1 temporal head, 3-seed ensemble | `r48-b1-clip-temporal-3seed-fp32.onnx` | 603,270 | **0.3 ms** | 0.3 ms |
| **end-to-end per clip** | — | — | **375.2 ms** | — |

Derived: **2.67 clips/s, 42.6 frames/s**; the backbone is **99.9 % of wall clock**. Reference on the
same host and batch: `p3-appearance-fp32.onnx` (MobileNetV3-small) **20.4 ms** — CLIP costs
**18.3×** the appearance backbone it replaced.

12. **The R48 accuracy gain is a backbone purchase, not a head design (2026-09-29).** B1's head is
   *smaller* than the E0/E3 heads it beats, and quoting the head size alone would misrepresent the
   cost. The +4.0 pp balanced-accuracy gap (B1w − E0@64d, McNemar p=3.01e-04, **cluster CI
   [−0.9, +8.8] includes 0**) is bought with an 18.3× slower, 35× larger frozen backbone. Chapter 5
   must state both halves in the same sentence. Three bans: (a) **these are host arm64 CPU numbers
   and may not be presented as Jetson Orin Nano Super figures** — no R48 graph has been measured on
   the device; (b) the 0.3 ms head figure may not be quoted without the 375.0 ms tower beside it;
   (c) no throughput claim for a deployed system may be derived from this — it is a fixed-input
   single-stream microbenchmark with no decode, no I/O and no detector.

Detector-generation control: the whole-frame arm of `feats_isbclip26_all.pt` (YOLO26s) is
**byte-identical** (max |diff| = 0.0) to `feats_isbclip_all.pt` (YOLOv11s), because no detector
touches it. So B1/B1w carry over unchanged and only ESTA is re-run for G17. Detections:
YOLOv11s 52,997 (90.2% of frames non-empty) vs YOLO26s 45,512 (87.9%).

Per-fold balanced accuracy (the log prints per-fold **top-1**, not BA — do not quote the log's
per-fold numbers as balanced accuracy): 91.2 / 73.0 / 72.5 / 66.6 / 63.0 / **57.3** / 81.9 / 87.0 /
84.7 / 71.7, via `tools/analyse-isb-fold-failures.py`.

Corpus facts (not metrics, but cited in prose and therefore traceable here): 1,100 clips, 420 hazard
/ 680 normal, 151 distinct sources, **0/151 sources carry both classes (R41)**, 52,997 detections,
geometry columns (`iou_walkway`, `iou_restricted`, `foot_in_walkway`, `norm_dist`) identically zero
(R45) — `tests/test-isafetybench-dataset.py`, all pass.

## Block 2 — Unsafe-Net binary lane (P-A blind / P-B k-fold-691 / P-C LCO-691)

Seeded as placeholders; each must be re-verified against its log at freeze time, NOT copied from
prose in `report/**` (that prose is exactly what phase-10's trace audit is checking).

The capacity-ladder rows below were recomputed on 2026-09-21 from the saved per-clip OOF tensors
(`tools/protocol-inflation-table.py`), **not** re-trained and **not** parsed from prose. Unsafe-Net
cluster bootstraps use only **8** camera groups and are thin — treat every Unsafe-Net CI here as
indicative; the 151-cluster iSafetyBench CIs in Block 1 carry the weight. The k-fold tensors saved no
clip ids or group labels and are row-aligned onto the LCO tensors via
`compare-oof-runs.py --ids_from`, which asserts label equality row by row before trusting the
alignment.

| metric | protocol | value | 95% CI / range | token_tag | script | log_path | date | touch | status |
|---|---|---|---|---|---|---|---|---|---|
| macro accuracy, headline | P-A blind (125) | _pending_ | — | `bin8_` | `run_relation_transformer.py --binary` | — | — | 1 (unspent) | placeholder |
| macro accuracy | P-C LCO-panel-disclosed | _pending_ | — | `bin8_` | same, declared variant | — | — | 0 | placeholder |
| **balanced accuracy, E3** (18,044) | **P-C LCO-691** | **69.6%** | the SMALLEST encoder is nominally best on this corpus | `bin8_` | `tools/protocol-inflation-table.py` | `sweep_results/relation_tokens/logs/g12-protocol-inflation-table.log` | 2026-09-21 | 0 | measured |
| balanced accuracy, E1 d64 (94,380) | P-C LCO-691 | 62.3% | — | `bin8_` | same | same | 2026-09-21 | 0 | measured |
| balanced accuracy, E1 d96 (197,292) | P-C LCO-691 | 63.9% | — | `bin8_` | same | same | 2026-09-21 | 0 | measured |
| balanced accuracy, E2 (284,556) | P-C LCO-691 | 66.7% | — | `bin8_` | same | same | 2026-09-21 | 0 | measured |
| balanced accuracy, E0 (298,892) | P-C LCO-691 | 66.7% | — | `bin8_` | same | same | 2026-09-21 | 0 | measured |
| **capacity spread across a 17x ladder** | P-C LCO-691 | **7.3 pp** | ordering is INVERTED vs iSafetyBench — the G13 corpus contrast | `bin8_` | same | same | 2026-09-21 | 0 | measured |
| balanced accuracy, E3 | **P-B k-fold-691 — CAMERA-LEAKING, for comparison** | 82.5% | — | `bin8_` | same | same | 2026-09-21 | 0 | diagnostic artefact |
| balanced accuracy, E1 d64 | P-B k-fold-691 leaking | 84.1% | — | `bin8_` | same | same | 2026-09-21 | 0 | diagnostic artefact |
| balanced accuracy, E1 d96 | P-B k-fold-691 leaking | 84.5% | — | `bin8_` | same | same | 2026-09-21 | 0 | diagnostic artefact |
| balanced accuracy, E2 | P-B k-fold-691 leaking | 84.6% | — | `bin8_` | same | same | 2026-09-21 | 0 | diagnostic artefact |
| balanced accuracy, E0 | P-B k-fold-691 leaking | 83.6% | — | `bin8_` | same | same | 2026-09-21 | 0 | diagnostic artefact |
| **capacity spread, same ladder** | P-B k-fold-691 leaking | **2.1 pp** | vs 7.3 pp under LCO — same non-discriminative effect as on iSafetyBench | `bin8_` | same | same | 2026-09-21 | 0 | measured |
| **Δ_inflation range across the ladder** | k-fold minus LCO | **+13.0 to +21.8 pp** | E3 +13.0 · E1d64 +21.8 · E1d96 +20.5 · E2 +17.9 · E0 +16.9 | `bin8_` | same | same | 2026-09-21 | 0 | measured |
| **paired Δ, E0 − E3** | P-C LCO-691 | **−2.8 pp** | cluster CI over only **8** groups [−7.1, +0.1] includes 0, McNemar p=0.606 — capacity buys nothing here | `bin8_` | `tools/compare-oof-runs.py` | `sweep_results/relation_tokens/logs/g13-capacity-contrast.log` | 2026-09-21 | 0 | measured |
| paired Δ, E0 − E3 | P-B k-fold-691 leaking | +1.1 pp | [−1.6, +5.2] includes 0; McNemar p=0.0271 **disagrees with the CI** — 8 clusters is thin, conservative CI reading taken | `bin8_` | same | same | 2026-09-21 | 0 | measured, disputed cell |
| detector mAP@50 / mAP@50-95 | detector eval | 0.966 / 0.975 | — | — | — | — | — | 0 | placeholder |

## Block 3 — Edge / Jetson (Unsafe-Net graphs)

| metric | protocol | value | token_tag | script | log_path | date | touch | status |
|---|---|---|---|---|---|---|---|---|
| **P3 per-decision latency incl. CPU frame decode, p50 crops (54)** — HEADLINE | on-device TRT FP16, MAXN_SUPER, n=3 | 331.19 / 328.88 / 329.69 ms = **3.02 / 3.04 / 3.03 dec/s** (margin 0.7-1.3%) | `bin8_` | `--decision --clips-dir data/test` | `decision-latency-fp16-20260929T{221836,223133,223401}.json` | 2026-09-29 | 0 | measured |
| **P3 per-decision latency incl. CPU frame decode, cap crops (128)** | same, n=3 | 385.15 / 382.30 / 383.73 ms = **2.60 / 2.62 / 2.61 dec/s — FAIL**; p95 **2.71-2.72**, p99 **2.64-2.66**, also FAIL | `bin8_` | same | same | 2026-09-29 | 0 | measured |
| P3 per-decision latency, p50 crops, **compute only, same 3 runs** | same, n=3 | 292.48 / 291.74 / 291.40 ms = **3.42 / 3.43 / 3.43 dec/s** (0.4% spread) | `bin8_` | same | same | 2026-09-29 | 0 | measured |
| P3 per-decision latency, cap crops, **compute only, same 3 runs** | same, n=3 | 346.44 / 345.16 / 345.44 ms = **2.89 / 2.90 / 2.89 dec/s — FAIL** | `bin8_` | same | same | 2026-09-29 | 0 | measured |
| _(decode-free process — not deployment-representative)_ compute only, 2 earlier runs | same, n=2, no `--clips-dir` | p50 275.97 / 272.40 ms = 3.62 / 3.67; cap 328.15 / 324.55 ms = 3.05 / 3.08 — **do not quote**: decode running in-process costs the detector +9.4% (11.85 -> 12.87 ms/frame), and no deployment decodes zero frames | `bin8_` | `--decision` (no clips) | `decision-latency-fp16-20260929T{214556,221408}.json` | 2026-09-29 | 0 | measured |
| P3 detector stage | on-device TRT FP16, n=3 | 12.846-12.903 ms/frame (x16 = **205.6 ms, 62% of the 331 ms decision**) | `bin8_` | same | same | 2026-09-29 | 0 | measured |
| P3 postprocess (select + Shapely geometry, 16 frames) | on-device, CPU, n=3 | 47.25-47.30 ms/decision (14.3%) | `bin8_` | same | same | 2026-09-29 | 0 | measured |
| P3 crop extraction, 54 crops | on-device, CPU, n=3 | 18.07-18.30 ms/decision (5.5%) | `bin8_` | same | same | 2026-09-29 | 0 | measured |
| P3 appearance, batched 54 crops | on-device TRT FP16, n=3 | 20.27-20.45 ms/decision (6.2%) | `bin8_` | same | same | 2026-09-29 | 0 | measured |
| P3 relation head, binary E3 | on-device TRT FP16, n=3 | **0.162-0.169 ms/decision (0.05%)** — the Transformer is free | `bin8_` | same | same | 2026-09-29 | 0 | measured |
| P1 per-stage latency (naive sum) | on-device TRT FP16 | see artifact — **`throughput_decisions_per_sec_naive` (73.051) is NOT a decision rate** | `bin8_` | `jetson-full-measurement.py --latency` | `deployment/artifacts/jetson-measurements/latency-fp16-20260929T212712.json` | 2026-09-29 | 0 | measured |
| frame decode, stream path (8 consecutive frames = 1 decision) | on-device, 20 real 1080p clips, CPU/FFMPEG (OpenCV built without GStreamer, **no NVDEC**), n=3 | **4.643-4.839 ms/frame p50 -> 37.14-38.71 ms/decision** (11.7% of the decision) | `bin8_` | `--decision --clips-dir data/test` | `decision-latency-fp16-20260929T{221836,223133,223401}.json` | 2026-09-29 | 0 | measured |
| frame decode, seek-sampled 16 frames (offline token extraction only) | on-device, same 20 clips, n=3 | 5577 / 5674 / 5614 ms p50 — **never add to the cadence figure**; 72x the sequential cost | `bin8_` | same | same | 2026-09-29 | 0 | measured |
| FP16 vs FP32 parity, 3 classifier stages (appearance, head-4class, head-bin-E3) | on-device TRT, argmax vs FP32 ref | **agreement 1.000 / 1.000 / 1.000**; max abs diff 2.53e-02 / 1.66e-03 / 0.0 | `bin8_` | `jetson-full-measurement.py --parity --rule argmax` | `deployment/artifacts/jetson-measurements/parity-20260929T221055.json` | 2026-09-29 | 0 | measured |
| FP16 vs FP32 parity, detector stage | on-device | **NOT MEASURED — do not quote the 0.633 in the artifact** | `bin8_` | same | same | 2026-09-29 | 0 | placeholder |
| energy per decision, **total attribution** | on-device, per-stage 30 s saturated runs, VDD_IN module total | **3.91 J/decision** = 16(0.22414) detector + 0.32017 appearance + 0.0015 head | `bin8_` | `--energy <stage> --energy-sec 30` | `energy-p3-{detector,appearance,relation-head-bin-e3}-fp16-2026092922{2624,2705,2811}.json` | 2026-09-29 | 0 | measured |
| energy per decision, **marginal over idle** | same, minus 7.563 W idle baseline | **2.19 J/decision** | `bin8_` | same | same + idle tegrastats 20 s n=40 | 2026-09-29 | 0 | measured |
| stage power / efficiency | on-device TRT FP16, 30 s each | detector 17.356 W, 77.4 inf/s, 0.22414 J/inf; appearance(batch 54) 15.741 W, 49.2 inf/s, 0.32017 J/batch; head-bin-E3 8.763 W, 5857 inf/s, **0.0015 J/inf** | `bin8_` | same | same | 2026-09-29 | 0 | measured |
| idle power baseline | on-device, tegrastats 20 s, n=40, MAXN_SUPER, no load | **7.563 W**; CPU 56.3 °C / GPU 58.2 °C | `bin8_` | tegrastats direct | — (console, 22:29) | 2026-09-29 | 0 | measured |
| composed-pipeline average wall power | — | **NOT MEASURED** — stages were saturated one at a time; true draw lies between 7.6 W idle and 17.4 W. Do not state a single wattage. | `bin8_` | — | — | — | 0 | placeholder |
| device memory, all P3 engines loaded | on-device TRT FP16 | host peak RSS **33.3 MB**; device used **3397.74 / 7607.06 MB** (44.7%) | `bin8_` | `--memory` | `memory-fp16-20260929T222545.json` | 2026-09-29 | 0 | measured |
| engine sizes, FP16 deployment set | on-device TRT build | detector 21.457 + appearance 2.862 + head-bin-E3 0.190 = **24.5 MB** (FP32: 38.299 / 4.473 / 0.190) | `bin8_` | `--model-size` | `model-size-20260929T222545.json` | 2026-09-29 | 0 | measured |
| ONNX↔PyTorch max abs logit diff, E3 | numerics only | 1.431e-06 | `bin8_` | `deployment/scripts/export-p3-relation-head-binary.py` | `deployment/export/MANIFEST.md` | 2026-09-19 | 0 | measured |
| ONNX↔PyTorch max abs logit diff, E0 | numerics only | 2.623e-06 | `bin8_` | same | `deployment/export/MANIFEST.md` | 2026-09-19 | 0 | measured |

## Historical — 4-class task (context only, never joined into the binary tables)

Left deliberately empty until freeze; the 4-class numbers exist in `report/**` and are superseded by
the binary reframe. Any 4-class number that appears in the final thesis must first get a row here.

## Unresolved questions

1. Does the thesis headline become the iSafetyBench 75.2% or stay Unsafe-Net? Blocks 1 and 2 are two
   corpora and must never be averaged or compared directly (different detector, different labels,
   zero geometry on Block 1). Framing decision, not a measurement one — plan.md unresolved Q4.
2. Block 2 rows need a verification pass against logs whose seed-count was never recorded
   (`audit-2026-09-19-repo-capability-and-feasibility.md`). If a log cannot substantiate a number,
   the row stays `placeholder` and the number leaves the thesis — it does not get re-run to match.
3. **Block 3 (2026-09-29, session run).** Compute-path rows are filled. **The cadence verdict is
   "on the boundary", not PASS** (n=5, settled): with CPU decode included only the p50 crop load
   clears 3.0 dec/s (**3.02-3.04**, margin 0.7-1.3%); p95/p99/cap come in at 2.71-2.72 / 2.64-2.66 /
   2.60-2.62. Compute-only on the same runs the cap row is 2.89-2.90, also FAIL. Energy, memory and
   engine-size rows are now filled. **Thermal soak closed 23:09** (Note 10); **multi-stream remains
   the open device item, and is blocked for P3** — `jetson-multi-stream-pipeline.py:_load_runners`
   hardcodes the P1 stage list, so running it as written would label a P1 saturation curve as P3.
   Three constraints on how they may be used:
   (a) **decode is now measured** (38.71 ms/decision, CPU/FFMPEG — OpenCV on the device has no
   GStreamer, so NVDEC is unmeasured and the decode term is an upper bound); a latency row that
   excludes decode must say so, since that exclusion is what separates PASS from FAIL;
   (b) the `throughput_decisions_per_sec_naive`
   figure of 73.051 in the `--latency` artifact sums one pass per stage and overstates the
   decision rate ~20x — it may never be quoted; (c) the Mac ORT-CPU baseline is **in dispute**
   (1825 ms/decision on 2026-09-09 vs 13727.54 on 2026-09-16, same graphs and EP; the latter run
   shows a 91% p50->p95 detector spread, the signature of a contended machine). Do not quote
   13,727 ms / 0.07 dec/s. Re-run on an idle Mac is blocked on the author's running experiments.
   Details: `reports/finding-2026-09-29-jetson-decision-latency.md`.
4. **G12 / G13 (2026-09-21).** G12 passes; G13 does not pass cleanly (E1 d96 sits 1.1 pp below E1
   d64 at double the parameters). Every RAND-1100 and k-fold-691 row above is a **diagnostic
   artefact** and may never be propagated as this thesis's accuracy — each appearance carries the
   label "source-leaking protocol, shown for comparison". Adjudication and consequences:
   `reports/finding-2026-09-21-protocol-inflation-and-the-capacity-contrast.md`, `risks.md`
   G12/G13 OUTCOME blocks.
5. ~~**The thesis currently states P3 misses cadence (1.13-1.71 dec/s projected, FAIL).**~~
   **CLOSED 2026-09-29 23:00.** The projection is **refuted by direct measurement**: P3 runs at
   **3.02-3.04 dec/s** at median crop load and **2.60-2.72** above it, i.e. *on* the 3.0 boundary,
   not ~2x below it. D2's framing is revised, the contract is not. Text sweep done in
   `deployment/JETSON-DEPLOYMENT.md`, `deployment/JETSON-RUNBOOK-2026-09-19.md`,
   `report/vn/bao-cao-tien-do-2026-09-17.txt`, `report/abstract.md`, `README.md`, and the
   2026-09-20 strategy report. **New standing ban:** the decode-free control rows
   (3.62 / 3.05-3.08 dec/s) may not be quoted either — see Note 3 and the finding report §5c/§6.
6. **Detector FP16 parity is not measured (2026-09-29).** `parity-20260929T221055.json` reports
   `p3-detector` agreement 0.633 / max abs diff 794.4. **Both are artefacts of an inapplicable
   decision rule, not evidence of quantization damage, and neither may be quoted.** The end2end
   detector emits a `(300, 6)` `[x1,y1,x2,y2,conf,cls]` table, so `decide()`'s `argmax(-1)` asks
   which of the six *columns* is largest — not what was detected; the inputs are additionally
   `standard_normal` noise, not images. The three classifier stages in the same artifact are
   unaffected (their outputs really are class-score vectors) and all agree at 1.000. A real
   detector check means running decoded frames through both precisions and comparing the boxes
   that survive `BRT.CONF` + `N_MAX` selection (IoU-matched, class-matched) — not yet run.
7. ~~**Two runs disagree on the compute base**~~ — **ADJUDICATED 2026-09-29 22:35 by two more runs
   (n=5).** The disagreement is not contention and not drift: it is **systematic and conditional**.
   The two runs without `--clips-dir` give detector p50 11.982 / 11.720; the three with it give
   12.903 / 12.846 / 12.852 — a **0.4% spread inside the with-decode condition** (so the 22:18 run
   was never contended) and a **+9.4% step between conditions**. Running OpenCV CPU decode in the
   same process costs the detector ~1.1 ms/frame. **Resolution: the sheet uses the with-decode
   condition throughout**, because no deployment decodes zero frames; the decode-free rows are
   retained above only as a labelled non-deployment control. The p50 margin is therefore 0.7-1.3%,
   not 6.0%, and the 3.18 dec/s composition proposed below is withdrawn. Original text follows.
   ~~The
   rows "P3 per-decision latency incl. CPU frame decode" (3.02 dec/s at p50 crops, margin 0.7%;
   2.71/2.64/2.60 at the tail) take **both** compute and decode from
   `decision-latency-fp16-20260929T221836.json`. That run's compute figures are contention-
   affected: detector p50 **12.903 ms** and p95 **15.602 ms**, a **20.9%** p50->p95 spread. The
   21:45 run on an idle device gives p50 **11.982** / p95 **12.022** — a **0.3%** spread. TensorRT
   on an idle Jetson is near-deterministic, so 20.9% is the same contention signature this project
   uses to reject the 2026-09-16 Mac baseline; applying that rule consistently means the 22:18
   compute numbers should not set the headline. Composing the clean 21:45 compute with the 22:18
   decode term gives **314.68 ms = 3.18 dec/s PASS at p50 crops** and 2.84 / 2.77 / **2.73 FAIL**
   at p95 / p99 / cap. Either basis may be defended, but the sheet must state which it uses and
   apply it to every row; right now the cadence margin at p50 reads as 0.7% on one basis and 6.0%
   on the other.~~ Detail: `reports/finding-2026-09-29-jetson-decision-latency.md` §5c (n=5 table)
   and §5d.
8. **Concurrent editing (2026-09-29).** Block 3 and this list were written by more than one session
   in the same hour. Before the freeze, re-read every Block 3 row against its named artifact rather
   than trusting the table text.
9. **McNemar aggregate-only bound was too loose — two conclusions reversed (2026-09-29 23:05).**
   `mcnemar_bound` / `_max_p_over_discordant_splits` enumerated discordant splits under `b+c <= n`
   only, ignoring each side's wrong-count, so they admitted infeasible 2x2 tables and inflated
   `max_p`. **`min_p` is unaffected; every published lower bound stands.** Corrected upper bounds:
   54/64 vs paper full-bbox 62/64 → **[0.008, 0.039]** (was [0.008, 0.382]) and 51/64 vs 62/64 →
   **[0.001, 0.007]** (was [0.001, 0.207]) — i.e. **the deficit against the paper's full-bbox
   pipeline IS significant at every feasible split**, reversing the earlier "neither
   previously-significant comparison survives". The error ran in the direction that flattered this
   thesis, so it is stated plainly wherever it appeared. Nothing changes against the no-bbox
   baseline, and the macro-parity claim is untouched. Rows in this sheet that use `mcnemar_exact`
   (both sides' per-clip predictions available) are **not** affected — the bound is only for
   comparisons against the paper's published totals. Code fixed, regression test added
   (`test_mcnemar_bound_respects_marginal_wrong_counts`), full suite 78 passed. Analysis:
   `reports/finding-2026-09-29-mcnemar-bound-marginal-constraint.md`. No test touch consumed
   (arithmetic over published aggregates only).
   **Follow-up, 23:20:** code review found the fix still incomplete in the mirror direction — the
   both-wrong cell was capped, the **both-right** cell was not (`b <= correct_a`, `c <= correct_b`
   missing), so tables with a negative both-right count were still admitted. It binds only when
   `correct_a + correct_b < n_total`, which no thesis comparison approaches (all ~50-62 of 64), so
   **every row above is unchanged** — re-verified after the fix: 56/62 [0.031, 0.109], 56/57
   [1.000, 1.000], 54/62 [0.008, 0.039], 54/57 [0.250, 0.629], 51/62 [0.001, 0.007], 51/57
   [0.031, 0.263]. Feasibility now lives in one shared enumerator
   (`unsafenet.metrics_stats.feasible_discordant_splits`) consumed by both call sites, since the
   same rule had to be fixed twice in two files; out-of-range counts now raise instead of silently
   returning `max_p = 0.0` ("significant at every split"). Second regression test added, this one
   checked against an independent four-cell ground truth over a swept low-accuracy grid. Full repo
   suite **184 passed**. Still no test touch consumed.

10. **Thermal soak, P3 detector, 30 min (2026-09-29 23:09).** Artifact
   `deployment/artifacts/jetson-measurements/thermal-soak-p3-detector-fp16-20260929T230913.json`;
   power mode logged (MAXN_SUPER). **No throttling** (`possible_throttling = False`): over the
   clean 22-min segment p50 held 11.665-11.725 ms, a **0.51 %** spread, while tj rose to
   **70.7 °C** — ~24 °C below the ~95 °C throttle. Sustained load draw **17.94 W mean / 18.76 W
   peak** (VDD_IN). Three bans: (a) the JSON's `latency_drift_pct = -7.86 %` **may not be quoted** —
   it is first-vs-last window and windows 0-7 are contaminated by my own `scp` to the device during
   the first 8 minutes (finding §5f); (b) the soak is the **detector stage alone with fixed
   inputs**, not the composed pipeline, and must be labelled as such; (c) 17.94 W is a back-to-back
   ~85 inf/s stress figure, not the deployed wall power at 24 fps — the composed-pipeline draw
   stays unmeasured.

11. **Power may never be used to infer a power *profile* on this device (2026-09-29 23:25, finding
   §5g, risk R32c).** I attempted to recover the June P1 soak's unlogged nvpmodel from its power
   trace (7.033 W mean is below tonight's 7.563 W "idle", so it looked like a lower profile) and
   **the attempt failed and is withdrawn**: that idle figure is an idle-*with-engine-resident*
   baseline on a hot SoC, not a floor, and Table 5.5D's own published sweep shows average power is
   **within 1.7 % across all three profiles** (5.31 / 5.29 / 5.22 W) — so power carries essentially
   no information about the profile. R32c stands as first written; chapters 1, 5 and 6 keep the
   conservative phrasing already installed (absolute watts, no share-of-budget) and **no further
   device measurement is warranted**. Standing rule: board power here tracks **GPU duty cycle, not
   the nvpmodel cap** — quote absolute draw at a stated duty cycle, never a budget share.

12. **This sheet's core rule is currently unfollowable for every Jetson row (2026-09-30, R32d/R32e).**
   `.gitignore:30` ignores `deployment/artifacts/` wholesale, so **none** of the artifacts chapter 5
   cites — `soak-jetson-summary.json`, `power-sweep-summary.json`, `tegrastats-soak30min.log`,
   `benchmarks-p3-onnx-cpu.json`, and the whole 2026-09-29/30 P3 set — is in version control. "Every
   value is verified by opening its `log_path`" cannot be carried out by anyone but the author on
   this one machine, and an examiner cannot resolve the `Source:` paths printed in the chapter at
   all. The 48 text artifacts total **2.7 MB** and contain only latency/power/temperature numbers —
   no frames, no identifiable workers, no credentials — so force-adding them (with the 574 MB of
   `.npy`/`.pt` caches left ignored) closes it. **Must land before the 2026-10-05 freeze. Not
   committed — awaiting instruction.** Separately, five P3 artifacts are still device-only and were
   never mirrored (the two `decision-latency` repeats, `memory`, `model-size`, the three `energy`
   files); a `scp` pattern failed silently and the device went unreachable at ~00:30. Until they are
   pulled, §5.6.8's energy table, memory row and n=3 spread rest on readings recorded in the finding
   report at 22:25-22:35, **not** on a locally re-openable artifact. The decision-latency table and
   the thermal-soak section were re-verified line-by-line against local JSON on 2026-09-30 and are
   sound.

13. **§5.6.8 written (2026-09-30).** Chapter 5 previously had no P3 section at all — §5.6.1-5.6.7
   are entirely P1, while chapter 4's architecture is P3, so the edge chapter did not measure the
   thesis's own model. §5.6.8 now carries Tables 5.5G (cadence by crop load), 5.5H (cost
   decomposition) and 5.5I (per-stage energy), the thermal-soak result, and an explicit
   not-measured list. Two P1 power claims in §5.6.7 and one in chapter 6 were rescoped at the same
   time: "< 8 W actual power" and "a 7 W Jetson" now state the duty-cycle dependence, because P3
   draws 17.94 W saturated and a single headline wattage misrepresents both graphs.

   **Re-verification pass, same day.** Every arithmetic claim in §5.6.8 was recomputed from
   `decision-latency-…T221836.json`: the six-stage sum reproduces 331.19 ms exactly, all six
   percentage shares round as printed, the 144× seek penalty and the +8.9 % decode-induced detector
   slowdown both check out. Three corrections were made. (a) The decode-free control was quoted as a
   single "3.62 dec/s"; there are **two** control runs (3.62 and 3.67 at p50, 3.05 and 3.08 at the
   crop cap) and both are now stated. (b) The engine-size row cited appearance = 2.86 MB from the
   **device-only** `model-size-…T222545.json`, while the locally held `model-size-…T212708.json`
   reports **2.949**; the engine rebuilt between 21:27 and 22:25. The chapter now cites the local,
   reader-checkable artifact — detector 21.457 + appearance 2.949 + E3 head 0.190 = **24.6 MB**, not
   24.5 — and carries that file as its `Source:`. The finding report's §5e figure of 24.5 MB is the
   later run and is not what the thesis quotes. (c) §5.6.8 now ends with an artifact-provenance
   paragraph naming the four claims that rest on the five unmirrored device-only files (Table 5.5I,
   the 7.56 W idle baseline, the 3.40 GB memory row, the within-1 % repeat claim), stating they are
   reported on the author's reading alone and are flagged for re-verification before the freeze.

14. **Banned test found live in the thesis text, and the 56/64 disclosures (2026-09-30).**
    Standing rule 1 forbids `two_proportion_z` on paired predictions. Chapter 5 §5.5 was still
    justifying the FP16 1-sample drop with exactly that — "two-proportion z = −0.26, two-tailed
    p = 0.79" — on two engines scored over the *same* 64 clips. The per-clip predictions for those
    runs no longer exist, only the totals, so the exact test is unavailable; the aggregate bound is
    used instead and gives **p = 1.000 at every feasible 2 × 2 table, best case and worst case
    alike**. The conclusion ("not significant") is unchanged and in fact strengthened, and the
    withdrawn z-test is named in the text rather than quietly deleted. A sweep of all report
    chapters found **no other instance**: chapter 4's ensemble-vs-direct4 comparison and the k-fold
    summary use bootstrap CIs with P(A > B), which are admissible. — On the 56/64 disclosures,
    §5.1 now carries a "how to read every accuracy number in this chapter" paragraph establishing
    that chapter 5 measures **export fidelity, not generalisation**: the test-selected reference
    functions as a fingerprint, so any engine-vs-reference divergence is export error. The three
    inline quotation sites (Table 5.3, the Jetson TRT table, the §5.8 summary bullet) now each carry
    a short test-selected tag pointing back to it. Nothing in this note changes a measured value.

15. **Host-CPU attribution is wrong throughout the thesis (2026-09-30) — R32f.** Chased down while
    checking whether Table 5.4 could be re-measured. It cannot, and the reason matters more than the
    table: this machine is an **Apple M3 MacBook Air, 8 cores (4P+4E), 16 GB**, while the thesis says
    "Apple M4" in 24 places and embellishes it with a 10-core count (§5.6 of ch05), a 12-core count
    (ch04), 36 GB of unified memory and an "M4 Pro" variant (ch03). The two M-series artifacts both
    record `Phgs-Mac.lan` / `macOS-14.6-arm64`, which is this host, and **macOS 14.6 cannot run on an
    M4 at all** — so the benchmarks were not produced on an M4. No measured value changes; the
    hardware label does. Deferred to the author rather than rewritten, because the training-hardware
    claims may legitimately point at a different machine than the benchmark host.

16. **Table 5.4 (M-series multi-stream) rests on a self-declared hand-patched artifact.** The file's
    own `note` field reads "Hand-patched post-fix numbers matching Chapter 5 Table 5.4". The table is
    a faithful rendering of that file — all four rows reproduce once the per-stream p50/p95 are
    averaged across streams, which is what the table reports — so there is no arithmetic error. The
    defect is provenance: the artifact was reconstructed to match the chapter instead of emitted by
    a run. The harness (`deployment/scripts/multi-stream-pipeline.py`, 164 lines) and all six P1 FP32
    ONNX models are present, so it is re-runnable — **but not on this machine**, because re-running
    here would silently substitute M3 numbers for numbers the chapter attributes to an M4 (note 15).
    A re-run was prepared and abandoned for that reason; nothing was overwritten. Blocked on note 15.
