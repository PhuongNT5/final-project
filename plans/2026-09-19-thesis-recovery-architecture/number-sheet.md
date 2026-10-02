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

## Numeral collisions between the two lanes (added 2026-10-02 — read before any defence)

Three figures repeat across lanes that mean entirely different things. Each is correct; each is a
trap. **No sentence may quote one of these without its protocol and its N**, and an examiner who
hears "eighty-four point four" must be told immediately which lane it is.

| Digits | Lane A | Lane B | Why it matters |
|---|---|---|---|
| **84.4 %** | **B1w balanced accuracy**, iSafetyBench, P-D LSO-1100, N=1,100 (Block 1b) | **blind rule top-1**, Unsafe-Net, P-A same-distribution, N=64 (H6) | Identical digits; different corpus, metric, protocol and N. They must never appear in the same sentence, and neither may be used to corroborate the other. |
| **84.2 %** | **B1 balanced accuracy** — **the deployed headline**, iSafetyBench, P-D LSO-1100, N=1,100 (Block 1b) | **k-fold macro**, Unsafe-Net 4-class, P-B 50 folds, 516 pooled clips (`KF` L53) | The headline number collides with a 4-class macro. Ch6 §"Second" quotes the 4-class one; it now says so explicitly in-text. |
| **81.2 %** | **scene-only same-distribution top-1**, 52/64 (H5) | **relation-Transformer LCO macro** (H1) | Both inside the 4-class lane. The scene-only tie at same-distribution and the model's LCO macro are the two ends of the central argument, and they happen to print the same. |

The deployed headline is **84.2 % balanced accuracy, B1, LSO-1100, N=1,100, against a 63.3 % floor**.
Its own OOF number is the one to quote — never `max(B1, B1w)`, and never the 4-class neighbour.

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

Source split re-verified 2026-10-02 with the loader's own key `SRC_RE = r'([A-Za-z0-9_-]{11})_'`:
151 groups = **76 hazard-only / 75 normal-only**, 0 cross-class. One filename
(`VFCTDVt5B8_trim_7.mp4`, 10-char id) misses the pattern and forms a singleton. The key is strictly
coarser than a filename-level parse (it merges `3UVQRVJy9g0_clip0/1`; a `_trim_` parse gives 152)
and no filename-level source is split across two key groups — grouping can only be stricter, never
leaky. Earlier "150 sources / normal 74" in risks.md R41 was stale and is corrected.

Provenance, now cited in the thesis (Ch3 §3.1.1, ref [26]): Abdullah, Rawat & Vyas, "iSafetyBench:
A video-language benchmark for safety in industrial environment," VISION'25 workshop @ ICCV 2025,
arXiv:2508.00399, licence **CC BY-NC-SA 4.0**, project page `https://isafetybench.github.io/`.
Original task is **open-vocabulary multi-label over 165 categories** (98 routine + 67 hazardous),
evaluated by multiple-choice prompting of VLMs — our binary recasting makes every number here
**non-comparable with the source paper**, and the thesis says so.

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
| **P3 per-decision latency incl. CPU frame decode, p50 crops (54), CLOCKS PINNED** — HEADLINE | on-device TRT FP16, MAXN_SUPER, `jetson_clocks` applied (CPU/GPU min=max), n=3 | 311.78 / 308.16 / 310.00 ms = **3.207 / 3.245 / 3.226 dec/s** — **PASS, margin +7.5%** (median basis) | `bin8_` | `--decision --clips-dir data/test --engine-dir deployment/export/jetson-p3` | `decision-latency-fp16-20261001T{222854,223216,223538}.json` | 2026-10-01 | 0 | measured |
| **P3 per-decision latency incl. CPU frame decode, p50 crops (54), DEFAULT DVFS** | same engines, `schedutil`, GPU 306-1020 MHz, n=3 | 360.01 / 371.69 / 365.88 ms = **2.778 / 2.690 / 2.733 dec/s — FAIL, -8.9%** | `bin8_` | same | `decision-latency-fp16-20261001T{220144,220543,220925}.json` | 2026-10-01 | 0 | measured |
| P3 per-decision, tail crop loads, **clocks pinned** | same, n=3, median basis | p95 346.87 ms = **2.883 (-3.9%)**; p99 355.93 = **2.810 (-6.3%)**; cap 362.32 = **2.760 (-8.0%)** — all FAIL | `bin8_` | same | `decision-latency-fp16-20261001T{222854,223216,223538}.json` | 2026-10-01 | 0 | measured |
| P3 per-decision, tail crop loads, **default DVFS** | same, n=3, median basis | p95 403.11 ms = **2.481 (-17.3%)**; p99 413.03 = **2.421 (-19.3%)**; cap 420.25 = **2.380 (-20.7%)** — all FAIL | `bin8_` | same | `decision-latency-fp16-20261001T{220144,220543,220925}.json` | 2026-10-01 | 0 | measured |
| P3 detector stage | on-device TRT FP16, n=3 | **pinned 11.724 ms/frame** (x16 = 187.58, 60.5% of the 310.00 ms decision); **DVFS 14.246** (x16 = 227.94) | `bin8_` | same | `decision-latency-fp16-20261001T{222854,223216,223538}.json` / `decision-latency-fp16-20261001T{220144,220543,220925}.json` | 2026-10-01 | 0 | measured |
| P3 detector p95/p50 ratio — DVFS tail penalty | same, n=3 | **pinned 1.03** (11.724 -> 12.030); **DVFS 1.51** (14.246 -> 21.575). DVFS widens the tail, not just the median | `bin8_` | same | same | 2026-10-01 | 0 | measured |
| P3 postprocess (select + Shapely geometry, 16 frames) | on-device, CPU, n=3 | **47.352 ms pinned / 47.672 DVFS** — only -0.7% under a clock change that moves the detector -17.3%: **clock-insensitive, single-threaded Python, now the binding term** (15.3%) | `bin8_` | same | same | 2026-10-01 | 0 | measured |
| P3 crop extraction, 54 crops | on-device, CPU, n=3 | 17.953 ms pinned / 18.801 DVFS (5.8%) | `bin8_` | same | same | 2026-10-01 | 0 | measured |
| P3 appearance, batched 54 crops | on-device TRT FP16, n=3 | 19.343 ms pinned / 27.779 DVFS (6.2%) | `bin8_` | same | same | 2026-10-01 | 0 | measured |
| P3 relation head, binary E3 | on-device TRT FP16, n=3 | **0.158 ms pinned / 0.284 DVFS = 0.05% of the decision** — the Transformer is free | `bin8_` | same | same | 2026-10-01 | 0 | measured |
| frame decode, stream path (8 consecutive frames = 1 decision) | on-device, 20 real 1080p clips, CPU/FFMPEG (OpenCV built without GStreamer, **no NVDEC**), n=3 | **37.61 ms/decision pinned** (4.70 ms/frame), 43.41 DVFS — 12.1% of the decision | `bin8_` | same | same | 2026-10-01 | 0 | measured |
| frame decode, seek-sampled 16 frames (offline token extraction only) | on-device, same 20 clips, n=3 | 5405.9 ms p50 pinned — **never add to the cadence figure**; ~144x the sequential cost | `bin8_` | same | same | 2026-10-01 | 0 | measured |
| run-to-run repeatability of the decision measurement | n=3 per condition | **pinned 0.46% spread; DVFS 2.42%** — pinning the clocks is also what makes the measurement repeatable | `bin8_` | same | same | 2026-10-01 | 0 | measured |
| **P3 detector thermal soak, 30 min** (replaces the 09-29 soak) | on-device, detector stage alone, fixed input, default DVFS, 30x60 s windows, 140,001 inferences | **no thermal throttling**: p50 **12.734-12.866 ms**, spread **1.04%**, drift vs first window **0.6%** (vs fastest 0.7%); `possible_throttling: false` | `bin8_` | `--thermal-soak p3-detector --soak-min 30 --soak-window-sec 60 --engine-dir deployment/export/jetson-p3` | `thermal-soak-p3-detector-fp16-20261001T232524.json` | 2026-10-01 | 0 | measured |
| soak thermal + power envelope | same run, 1789 tegrastats samples | junction temp 54.0 -> **69.9 °C max** (mean 68.8, plateaued); sustained power **16.98 W mean**, peak 18.02 W; GPU util mean 80% (min 69%); RAM 2932 -> 2962 MB (+30 MB, no leak) | `bin8_` | same | `thermal-soak-p3-detector-fp16-20261001T232524.json` | 2026-10-01 | 0 | measured |
| soak measurement conditions, recorded in-artifact | same run | `clocks_pinned.any false`, `competing_processes []`, `quiet true`, `nvpmodel_q MAXN_SUPER`, cpu governor `schedutil` (policy0 cur 1,113,600 of max 1,728,000), GPU `cur_hz 1,020,000,000 = max_hz`. **This is the first soak whose clock state and contention are in the artifact** | `bin8_` | same | `thermal-soak-p3-detector-fp16-20261001T232524.json` | 2026-10-01 | 0 | measured |
| _(indicative, not controlled)_ DVFS penalty decomposition | detector p50 in three regimes, same rebuilt engine | in-pipeline DVFS **14.246** / saturated soak **12.796** / in-pipeline pinned **11.779** ms per frame — saturation alone recovers **59%** of the DVFS-to-pinned gap, pinning the remaining **41%** (CPU-side, since DVFS already reaches max GPU clock under saturation). **Soak times the detector in a tight loop, the pipeline times it interleaved with CPU postprocess and decode — different access patterns, so this is a decomposition sketch, not an experiment** | `bin8_` | derived from the soak + the 6 decision-latency artifacts | `thermal-soak-p3-detector-fp16-20261001T232524.json` + `decision-latency-fp16-20261001T22{0925,3216}.json` | 2026-10-01 | 0 | derived |
| P3 engine memory, all 4 P3 engines resident | on-device, P3 engine dir only | `engines_resident_delta_mb` = **251.1**, `n_engines_resident` = 4, host peak RSS 31.6 MB. Supersedes 417.84 (an 11-engine P1+P3 aggregate from the default dir). `device_used_mb` 3523.38 is system-wide unified memory, **not** a model footprint | `bin8_` | `--memory --engine-dir deployment/export/jetson-p3` | `memory-fp16-20261001T214413.json` | 2026-10-01 | 0 | measured |
| P3 FP16 engine set size | on-device, rebuilt engines | **24.46 MB** (detector 21.543 + appearance 2.727 + bin-E3 head 0.190) | `bin8_` | `stat` on `deployment/export/jetson-p3` | device filesystem | 2026-10-01 | 0 | measured |
| P3 per-stage energy, **total-board basis only** | on-device, VDD_IN, 60 s saturated/stage | detector 74.70 inf/s @ 17.06 W = **0.2284 J/inf**; appearance (54-crop batch) 41.34 @ 15.01 W = **0.3631 J/batch**; bin-E3 head 3705.0 @ 9.06 W = **0.00245 J/inf**. Composed: **4.02 J/decision**, head share **0.06%** | `bin8_` | `--energy <stage> --energy-sec 60` | `energy-p3-{detector,appearance,relation-head-bin-e3}-fp16-20261001T22*.json` | 2026-10-01 | 0 | measured |
| _(do not quote)_ `*_delta_idle` energy figures | same artifacts | idle baseline recorded 6.841 / 5.618 / 7.509 W across three consecutive stages (**34% spread**, 55% including the concurrent R48 run's 4.842 W) because each takes only a 10 s window right after the previous stage's load. **`*_delta_idle` is not comparable across stages and is not quoted**; one settled 60 s baseline is being measured to repair it | `bin8_` | same | same | 2026-10-01 | 0 | measured, unusable |
| settled idle power baseline, **the one to quote** | on-device, 60 s window, engine resident, no work submitted, taken after a 300 s settle | **4.691 W**, n=118 samples, `quiet: true`, clocks NOT pinned, load 0.08, 50 °C | `bin8_` | `--energy p3-relation-head-bin-e3 --energy-sec 5 --idle-sec 60` | `energy-p3-relation-head-bin-e3-fp16-20261001T225523.json` | 2026-10-01 | 0 | measured |
| P3 per-decision energy, **marginal basis** (spec `(P_avg − P_idle)/fps`) | composed from the three 60 s stage runs against the single 4.691 W settled baseline | detector ΔP 12.372 W = **0.16562 J/inf**; appearance ΔP 10.319 W = **0.24961 J/batch**; bin-E3 head ΔP 4.371 W = **0.00118 J/inf**. Composed: **2.90 J/decision marginal** vs **4.02 J/decision total-board**; head share **0.041%** | `bin8_` | derived, see Note 20 | `energy-p3-*-fp16-20261001T22*.json` + `energy-p3-relation-head-bin-e3-fp16-20261001T225523.json` | 2026-10-01 | 0 | derived |
| **P3 per-decision latency incl. CPU frame decode, p50 crops (54)** — HEADLINE | on-device TRT FP16, MAXN_SUPER, n=3 | 331.19 / 328.88 / 329.69 ms = **3.02 / 3.04 / 3.03 dec/s** (margin 0.7-1.3%) | `bin8_` | `--decision --clips-dir data/test` | `decision-latency-fp16-20260929T{221836,223133,223401}.json` | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — engines did not match the device record; clock state unrecorded |
| **P3 per-decision latency incl. CPU frame decode, cap crops (128)** | same, n=3 | 385.15 / 382.30 / 383.73 ms = **2.60 / 2.62 / 2.61 dec/s — FAIL**; p95 **2.71-2.72**, p99 **2.64-2.66**, also FAIL | `bin8_` | same | same | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — engines did not match the device record; clock state unrecorded |
| P3 per-decision latency, p50 crops, **compute only, same 3 runs** | same, n=3 | 292.48 / 291.74 / 291.40 ms = **3.42 / 3.43 / 3.43 dec/s** (0.4% spread) | `bin8_` | same | same | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — engines did not match the device record; clock state unrecorded |
| P3 per-decision latency, cap crops, **compute only, same 3 runs** | same, n=3 | 346.44 / 345.16 / 345.44 ms = **2.89 / 2.90 / 2.89 dec/s — FAIL** | `bin8_` | same | same | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — engines did not match the device record; clock state unrecorded |
| _(decode-free process — not deployment-representative)_ compute only, 2 earlier runs | same, n=2, no `--clips-dir` | p50 275.97 / 272.40 ms = 3.62 / 3.67; cap 328.15 / 324.55 ms = 3.05 / 3.08 — **do not quote**: decode running in-process costs the detector +9.4% (11.85 -> 12.87 ms/frame), and no deployment decodes zero frames | `bin8_` | `--decision` (no clips) | `decision-latency-fp16-20260929T{214556,221408}.json` | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — engines did not match the device record; clock state unrecorded |
| P3 detector stage | on-device TRT FP16, n=3 | 12.846-12.903 ms/frame (x16 = **205.6 ms, 62% of the 331 ms decision**) | `bin8_` | same | same | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — engines did not match the device record; clock state unrecorded |
| P3 postprocess (select + Shapely geometry, 16 frames) | on-device, CPU, n=3 | 47.25-47.30 ms/decision (14.3%) | `bin8_` | same | same | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — engines did not match the device record; clock state unrecorded |
| P3 crop extraction, 54 crops | on-device, CPU, n=3 | 18.07-18.30 ms/decision (5.5%) | `bin8_` | same | same | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — engines did not match the device record; clock state unrecorded |
| P3 appearance, batched 54 crops | on-device TRT FP16, n=3 | 20.27-20.45 ms/decision (6.2%) | `bin8_` | same | same | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — engines did not match the device record; clock state unrecorded |
| P3 relation head, binary E3 | on-device TRT FP16, n=3 | **0.162-0.169 ms/decision (0.05%)** — the Transformer is free | `bin8_` | same | same | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — engines did not match the device record; clock state unrecorded |
| P1 per-stage latency (naive sum) | on-device TRT FP16 | see artifact — **`throughput_decisions_per_sec_naive` (73.051) is NOT a decision rate** | `bin8_` | `jetson-full-measurement.py --latency` | `deployment/artifacts/jetson-measurements/latency-fp16-20260929T212712.json` | 2026-09-29 | 0 | measured |
| frame decode, stream path (8 consecutive frames = 1 decision) | on-device, 20 real 1080p clips, CPU/FFMPEG (OpenCV built without GStreamer, **no NVDEC**), n=3 | **4.643-4.839 ms/frame p50 -> 37.14-38.71 ms/decision** (11.7% of the decision) | `bin8_` | `--decision --clips-dir data/test` | `decision-latency-fp16-20260929T{221836,223133,223401}.json` | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — engines did not match the device record; clock state unrecorded |
| frame decode, seek-sampled 16 frames (offline token extraction only) | on-device, same 20 clips, n=3 | 5577 / 5674 / 5614 ms p50 — **never add to the cadence figure**; 72x the sequential cost | `bin8_` | same | same | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — engines did not match the device record; clock state unrecorded |
| FP16 vs FP32 parity, 3 classifier stages (appearance, head-4class, head-bin-E3) | on-device TRT, argmax vs FP32 ref | **agreement 1.000 / 1.000 / 1.000**; max abs diff 2.53e-02 / 1.66e-03 / 0.0 | `bin8_` | `jetson-full-measurement.py --parity --rule argmax` | `deployment/artifacts/jetson-measurements/parity-20260929T221055.json` | 2026-09-29 | 0 | measured |
| FP16 vs FP32 parity, detector stage | on-device | **NOT MEASURED — do not quote the 0.633 in the artifact** | `bin8_` | same | same | 2026-09-29 | 0 | placeholder |
| energy per decision, **total attribution** | on-device, per-stage 30 s saturated runs, VDD_IN module total | **3.91 J/decision** = 16(0.22414) detector + 0.32017 appearance + 0.0015 head | `bin8_` | `--energy <stage> --energy-sec 30` | `energy-p3-{detector,appearance,relation-head-bin-e3}-fp16-2026092922{2624,2705,2811}.json` | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — 30 s windows on engines that did not match the device record; the 7.563 W idle baseline was unsettled (settled value is 4.691 W), so both the total and the marginal figure are retired |
| energy per decision, **marginal over idle** | same, minus 7.563 W idle baseline | **2.19 J/decision** | `bin8_` | same | same + idle tegrastats 20 s n=40 | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — 30 s windows on engines that did not match the device record; the 7.563 W idle baseline was unsettled (settled value is 4.691 W), so both the total and the marginal figure are retired |
| stage power / efficiency | on-device TRT FP16, 30 s each | detector 17.356 W, 77.4 inf/s, 0.22414 J/inf; appearance(batch 54) 15.741 W, 49.2 inf/s, 0.32017 J/batch; head-bin-E3 8.763 W, 5857 inf/s, **0.0015 J/inf** | `bin8_` | same | same | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — 30 s windows on engines that did not match the device record; the 7.563 W idle baseline was unsettled (settled value is 4.691 W), so both the total and the marginal figure are retired |
| idle power baseline | on-device, tegrastats 20 s, n=40, MAXN_SUPER, no load | **7.563 W**; CPU 56.3 °C / GPU 58.2 °C | `bin8_` | tegrastats direct | — (console, 22:29) | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — 30 s windows on engines that did not match the device record; the 7.563 W idle baseline was unsettled (settled value is 4.691 W), so both the total and the marginal figure are retired |
| composed-pipeline average wall power | — | **NOT MEASURED** — stages were saturated one at a time; true draw lies between 7.6 W idle and 17.4 W. Do not state a single wattage. | `bin8_` | — | — | — | 0 | placeholder |
| device memory, all P3 engines loaded | on-device TRT FP16 | host peak RSS **33.3 MB**; device used **3397.74 / 7607.06 MB** (44.7%) | `bin8_` | `--memory` | `memory-fp16-20260929T222545.json` | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — measured on the pre-rebuild engine set; see the 2026-10-01 memory and engine-size rows above |
| engine sizes, FP16 deployment set | on-device TRT build | detector 21.457 + appearance 2.862 + head-bin-E3 0.190 = **24.5 MB** (FP32: 38.299 / 4.473 / 0.190) | `bin8_` | `--model-size` | `model-size-20260929T222545.json` | 2026-09-29 | 0 | **SUPERSEDED 2026-10-01** — measured on the pre-rebuild engine set; see the 2026-10-01 memory and engine-size rows above |
| ONNX↔PyTorch max abs logit diff, E3 | numerics only | 1.431e-06 | `bin8_` | `deployment/scripts/export-p3-relation-head-binary.py` | `deployment/export/MANIFEST.md` | 2026-09-19 | 0 | measured |
| ONNX↔PyTorch max abs logit diff, E0 | numerics only | 2.623e-06 | `bin8_` | same | `deployment/export/MANIFEST.md` | 2026-09-19 | 0 | measured |

## Block 3b — Edge / Jetson, R48 deployment path (B1: CLIP tower + temporal head)

Orin Nano **Super** 8GB `phuongnguyen-desktop`, JetPack R36.5.0, TensorRT 10.3.0, nvpmodel 2
(MAXN_SUPER), **default DVFS (clocks NOT pinned)**, device verified **quiet** (no competing
measurement process) by the provenance block now embedded in every artifact. Engines built
2026-10-01 and verified free of the stale-device warning. n=3 independent repeats, 100 runs each
after 3 warmups. One class decision = **16 frames, whole frame, no detector**.

Read `Block 3b-note` below before quoting anything here: three earlier runs today were withdrawn.

| metric | protocol | value | token_tag | script | log_path | date | touch | status |
|---|---|---|---|---|---|---|---|---|
| **B1 class-decision compute latency, stages measured BACK-TO-BACK** | on-device TRT FP16, quiet, unpinned, n=3 | **23.796 / 23.731 / 23.958 ms = 42.02 / 42.14 / 41.74 dec/s** (0.95% spread). **Not a deployed cadence** — these are back-to-back stage costs; the deployed loop is the row below and is 12.2x slower per decision. See 3b-note 9 | `r48_` | `benchmark-r48-clip-pipeline.py --backend tensorrt --mode fp16 --runs 100 --detector --power --json` | `r48-pipeline-20261001T{223826,223853,223920}.json` | 2026-10-01 | 0 | measured |
| **B1 composed live-pipeline cadence** — HEADLINE for deployment | on-device TRT FP16, quiet, unpinned, **whole loop** (decode -> preprocess -> tower -> head), `--window 48 --stride 24`, 24 clips / 201 decisions / 5559 frames, n=3 at fixed seed 0 | **289.672 ms/decision = 3.452 dec/s** (289.339 / 289.672 / 292.834; spread 1.21%), p95 **322.708 ms**; incl. window fill 260.315 ms; **duty 76.1%**. Single-threaded, so throughput is a LOWER bound and duty an UPPER bound | `r48_` | `benchmark-r48-composed-cadence.py --engine-dir deployment/export/jetson-r48 --mode fp16 --clips 24 --seed 0 --json` | `r48-composed-cadence-fp16-20261001T{233159,233324,233449}.json` | 2026-10-01 | 0 | measured |
| **preprocess (CPU resize+normalise, 16 frames)** — the largest cost in the whole pipeline | same composed runs, n=3 | p50 **160.765 ms** (160.629 / 160.765 / 160.926; spread 0.18%) = **73.4% of compute, 55.5% of the decision interval**. Measured by **no** isolated stage benchmark, so it was absent from every stage sum | `r48_` | same | same | 2026-10-01 | 0 | measured |
| **preprocess, sub-step split and the fix** — *HOST numbers, NOT device numbers* | host M3, 8 cv2 threads, 16 frames, median of 3 repeats x 30 runs, real decoded frames | the BGR->RGB swap runs **before** the resize on a negative-stride view, so OpenCV copies the whole frame at source resolution: that one sub-step is **90.3 of 101.6 ms** at 1920x1080 and **5.5 of 15.5 ms** at 480x270. Swapping after the resize is **bit-identical** (`np.array_equal`, no tolerance) and **7.46x** faster at 1080p, **1.36x** at 480x270. Applied to `preprocess` and `to_batch`. **No device re-measurement yet** — the board lost power on 02-10; predictions P-PRE1..P-PRE4 are registered in `reports/finding-2026-10-02-preprocessing-cost-is-a-channel-swap.md` before that run | `r48_` | `profile-r48-preprocess.py --data-root data/{test,isafetybench} --json` | `r48-preprocess-profile-20261002T214004.json` (`is_device_number: false`) | 2026-10-02 | 0 | **host-only, pending device re-measure** |
| CLIP tower **in-loop** vs back-to-back | same composed runs, n=3 | in-loop p50 **58.630 ms** (58.002 / 58.630 / 58.883) against **23.386 ms** back-to-back — **2.51x**. Cause adjudicated: GPU clock droop over the ~210 ms idle gap, not contention (next row) | `r48_` | same | same | 2026-10-01 | 0 | measured |
| B1 head **in-loop** | same composed runs, n=3 | p50 **0.798 ms** (0.790 / 0.798 / 0.806) = **0.4% of compute, 0.3% of the decision interval** | `r48_` | same | same | 2026-10-01 | 0 | measured |
| frame decode **in the deployed loop** (continuous, mixed resolution) | same composed runs, n=3, on `data/isafetybench` itself | p50 **2.048 / 2.055 / 2.104 ms/frame** -> **49.2 ms per 24-frame advance = 17.0% of the decision interval**. This, not 5.047, is the deployed decode cost on this corpus | `r48_` | same | same | 2026-10-01 | 0 | measured |
| **idle-droop adjudication: why the tower costs 2.5x more in the loop** | on-device TRT FP16, tower only, quiet, unpinned, 60 runs/arm, n=3; pre-registered rule | back-to-back p50 **26.448 ms** vs **+210 ms idle gap, nothing else running** p50 **55.231 ms** — ratio **2.088x** (2.051 / 2.088 / 2.152). Rule (B >= 1.8xA **and** B within +-20% of 58.630) **PASSES in all 3**: idle alone reproduces **94.2%** of the in-loop cost, so the cause is **GPU power state**, not memory-bus contention | `r48_` | `benchmark-r48-composed-cadence.py --probe-idle --idle-gap-ms 210 --probe-runs 60` | `r48-idle-droop-probe-fp16-20261001T{234145,234242,234322}.json` | 2026-10-01 | 0 | measured |
| **real-time contract, corpus fps distribution** | metadata of all **1100** iSafetyBench clips; stride 24 means a source at *f* fps demands *f*/24 dec/s | median **29.970** fps -> contract 1.249 dec/s, margin **2.76x**; mean 34.245 -> 1.427, 2.42x; **p95 = max 60.0 fps (218 clips, 19.8%) -> 2.500 dec/s, margin 1.38x**. **25.1 fps occurs in 11 clips (1.0%)** and the "1.05 dec/s contract" derived from it is withdrawn | `r48_` | `cv2.CAP_PROP_FPS` over `data/isafetybench/*/*.mp4` | derived, see 3b-note 9 | 2026-10-01 | 0 | measured |
| CLIP ViT-B/32 vision tower, batch 16 | same, 87,849,216 params | p50 **23.386 / 23.335 / 23.567 ms**; p95 31.187 / 35.077 / 35.418; min 22.859-22.912 | `r48_` | same | same | 2026-10-01 | 0 | measured |
| B1 temporal head, 3-seed ensemble, batch 1 | same, 603,270 params | p50 **0.410 / 0.396 / 0.391 ms**; p95 0.484-0.502 — the Transformer is free, as in P3 | `r48_` | same | same | 2026-10-01 | 0 | measured |
| backbone share of class-decision compute | same, n=3 | **98.28 / 98.33 / 98.37%** | `r48_` | same | same | 2026-10-01 | 0 | measured |
| **YOLOv11s localisation, isolated** — a *separate* obligation, never summed into the class decision | on-device TRT FP16 @imgsz **800**, batch 1, n=3 | p50 **12.636 / 13.001 / 12.741 ms/frame**; p95 14.638-14.736 | `r48_` | same | same | 2026-10-01 | 0 | measured |
| YOLOv11s localisation, **with decode running in-process** | same, n=2 | p50 **13.317 / 14.491 ms/frame** — the decode tax the 09-29 session measured at +9.4%; quote the isolated figure for the stage and this one for a composed pipeline | `r48_` | `jetson-full-measurement.py --decision --engine-dir deployment/export/jetson-p3` | `decision-latency-fp16-20261001T{224529,224821}.json` | 2026-10-01 | 0 | measured |
| frame decode, stream path (sequential, no seek) — **1080p, 8-frame bursts** | on-device, 20 clips of `data/test` (**all 1920x1080**, verified), CPU/FFMPEG (**no NVDEC**), 8 consecutive frames per timing, n=2 | p50 **5.047 / 5.091 ms/frame**, p95 6.259 / 6.943 -> 40.373 / 40.728 ms per 8-frame advance. **Condition-specific: this is 1080p in short bursts, and it is NOT the deployed decode cost** — see the composed row below (2.048 ms/frame) and 3b-note 9 | `r48_` | same | same | 2026-10-01 | 0 | measured |
| frame decode, seek-sampled 16 frames | same, n=2 | **5815.9 / 5729.0 ms** — OFFLINE token extraction only; **never add to a cadence figure** (1150x the sequential cost per frame) | `r48_` | same | same | 2026-10-01 | 0 | measured |
| **FP16 vs FP32 parity, full B1 decision path** | on-device TRT FP16 vs onnxruntime CPU FP32 ref, 24 real clips (12 hazard / 12 normal) | **decision agreement 1.000** (gate 1.000), **0 preprocessing mismatches**, max abs prob diff **2.66e-03** | `r48_` | `verify-r48-device-parity.py --mode fp16` | `r48-parity-fp16-20261001T224028.json` | 2026-10-01 | 0 | measured |
| end-to-end per clip **incl. seek decode** | same 24 clips | p50 **1761.8 ms**, p95 4245.2, p99 5428.6 — this is the **offline file-evaluation path**; it is NOT the deployed cadence and must never be quoted as one | `r48_` | same | same | 2026-10-01 | 0 | measured |
| **energy per class decision** | on-device, 60.015 s saturated tower run, 2508 invocations, VDD_IN module total | **0.4811 J/decision** total board; **0.36522 J/decision** marginal over idle; 41.789 inf/s, 20.105 W avg | `r48_` | `jetson-full-measurement.py --energy r48-clip-vit-b32-vision --energy-sec 60 --idle-sec 10` | `energy-r48-clip-vit-b32-vision-fp16-20261001T224142.json` | 2026-10-01 | 0 | measured |
| engine sizes, B1 deployment set | on-device TRT build, FP16 | class path **175.9 MB** (tower 174.022 + head 1.874); + localisation 21.543 = **197.4 MB** total | `r48_` | `--model-size --engine-dir deployment/export/jetson-r48` | `model-size-20261001T224927.json` | 2026-10-01 | 0 | measured |
| device memory, B1 set resident | on-device TRT FP16, 3 engines, **quiet window**, n=3 | **90.01 MB** median (91.36 / 90.01 / 89.61; spread **1.94%**, passes the pre-registered `max−min ≤ 0.10×median` rule); host peak RSS 31.5–33.4 MB. Field is a **system-wide free-memory delta**, NOT "weights resident" — it is less than half the 197.4 MB engine-file total. The withdrawn `quiet: false` reading was 92.24 MB (2.5% off) | `r48_` | `--memory --engine-dir deployment/export/jetson-r48` | `memory-fp16-20261001T{232747,232808,232830}.json` | 2026-10-01 | 0 | measured |
| reference: MobileNetV3-small appearance, batch 16 | same conditions, n=3 | p50 **7.336 / 7.710 / 7.645 ms** -> CLIP costs **3.06x** the appearance backbone it replaced (host CPU said 18.3x; see Block 3b-note 3) | `r48_` | same as headline | same | 2026-10-01 | 0 | measured |
| _(declared second condition)_ same stages, **clocks pinned** by `jetson_clocks` | on-device TRT FP16, n=2 — **pinning is NOT substantiated by these artifacts** | tower p50 23.481 / 23.167 ms, p95 24.084-25.359; detector p50 11.743 / 11.802 ms. **Do not quote.** Both files predate `jetson-clock-state-utils.py`: `cpu_clocks`/`gpu_clocks`/`clocks_pinned` are all absent, so "pinned" rests only on recollection — and note 1 **withdraws these very two artifacts**. They are also 6 s apart, so they are not two independent repeats. See 3b-note 2 and 3b-note 9 | `r48_` | same as headline | `r48-pipeline-20261001T{222505,222511}.json` | 2026-10-01 | 0 | **withdrawn** |

### Block 3b-note — how these numbers were obtained, and what was thrown away

1. **Three runs taken earlier the same evening are withdrawn.** A second session was measuring on the
   same device. `r48-pipeline-20261001T221822.json` was taken while a concurrent 60 s energy sweep
   saturated the GPU (tower 41.8 ms, detector 21.1 ms); `...222505/...222511` were taken while that
   session had pinned all clocks. The spread between the contended run and the clean run is
   **1.79x on the same engine and the same board**, and all three carried **byte-identical**
   `device_state`, because `nvpmodel -q` still reads MAXN_SUPER under a clock pin and
   `jetson_clocks --show` needs root and was serialised as null. Full account:
   `reports/finding-2026-10-01-concurrent-session-contamination.md`. Closed in code by
   `deployment/scripts/jetson-clock-state-utils.py`, folded into the shared `device_state()`, so every
   artifact from this commit forward records per-policy CPU/GPU min/cur/max, a `clocks_pinned` flag,
   `competing_processes` and `quiet`. The 22:38 headline runs carry `quiet: true`,
   `clocks_pinned: {cpu: false, gpu: false, any: false}`, `competing_processes: []`.
   **Consequence that is not comfortable: no device artifact written before this commit — 09-29
   included, and note 8 records that 09-29 was also a multi-session day — can have contention or
   pinning excluded from its own record.**

2. **CORRECTED 2026-10-01 (3b-note 9): true back-to-back, false for the deployed duty cycle — and
   the pinned side of the comparison is withdrawn anyway.** The claim below rests on
   `r48-pipeline-20261001T{222505,222511}.json`, which note 1 **withdraws** and which carry no
   `clocks_pinned` block at all, so the "pinned" label is unsubstantiated; and the idle-droop probe
   shows that when the pipeline is duty-cycled, DVFS shifts the tower **median** by **2.088x**
   (26.448 -> 55.231 ms), the largest DVFS effect measured anywhere in this sheet. **Do not quote
   "DVFS is a tail cost" without "when stages are run back-to-back".**
   ~~Pinning the clocks does not change the median; it removes the tail.~~ p50 is 23.17-23.48 ms
   pinned against 23.34-23.57 ms unpinned (indistinguishable), while p95 is 24.1-25.4 ms pinned
   against 31.2-35.4 ms unpinned. The DVFS ramp is a **tail** cost, not a throughput cost. The
   detector, being batch-1 with shorter kernels, is the one stage pinning actually speeds up
   (11.74-11.80 vs 12.64-13.00 ms, ~7%). Default DVFS is quoted as the headline because it is the
   state the device boots and deploys in and the state the 09-29 numbers were taken in; the pinned
   pair is kept above as a labelled condition, not merged.

3. **The host-CPU extrapolation in Block 1c was wrong by 16x, and in the safe direction.** Block 1c
   measured the tower at 375.0 ms on host CPU with the backbone at 99.9% of wall clock. On device it
   is **23.386 ms** — 16.0x faster. The CLIP-vs-MobileNetV3 ratio also collapses from 18.3x on CPU to
   **3.06x** on device: tensor cores and batch-16 amortisation favour the large regular GEMMs of a
   ViT far more than they favour a small depthwise-separable CNN. Block 1c stays in the sheet as the
   host number it always was, and the lesson is explicit: **a CPU ratio is not an edge ratio.**

4. **SUPERSEDED BY MEASUREMENT 2026-10-01 — see 3b-note 9.** A composed live-pipeline measurement
   for B1 now exists and refutes the budget below by ~4x. The paragraph is kept only so the
   superseded construction is on the record; **none of its numbers may be quoted.**
   ~~No composed live-pipeline measurement exists for B1.~~ Everything above is per-stage. A duty
   budget can be assembled from it — at 25 fps, localising every frame and deciding on the demo's
   48-frame window with 24-frame stride: decode 25x5.047 = 126.2 ms, localisation 25x12.741 =
   318.5 ms, classification 23.796/0.96 s = 24.8 ms, total **469.5 ms per second of video = 47.0%
   duty, 2.1x headroom** — but this is a **naive stage sum**, the exact construction Block 3 already
   warns against (`throughput_decisions_per_sec_naive` is not a decision rate). It may be reported as
   a budget with that caveat attached and must not be presented as a measured cadence.

5. **B1 against P3 on energy, since both now have a measured per-decision figure.**
   *(Revised 2026-10-01 after the settled idle baseline; the original ratio used the retired
   3.91 J P3 figure.)* On the **total-board** basis B1's class decision costs **0.4811 J** against
   P3's **4.02 J** (16 detector invocations plus appearance plus head) — **B1 is 8.4x cheaper
   per decision**. On the **marginal** basis, recomputing both against the one settled 4.691 W
   baseline, B1 costs **0.3689 J** = (20.105 − 4.691)/41.789 and P3 **2.90 J** —
   **7.9x**. Quote the ratio with its basis named; the two differ because the board floor is a
   larger fraction of B1's smaller bill. B1 is also the more accurate arm (+4.0 pp balanced
   accuracy, cluster CI includes 0 — the accuracy claim is not certified at cluster resolution
   and the energy claim does not change that). The 4.842 W idle figure recorded inside the B1 run
   itself (10 s, n=19) and the 09-29 figure of 7.563 W (20 s, n=40) are both retired as baselines;
   per note 11 neither may be used to infer a power *profile* either.

9. **The stage sum is refuted by the composed measurement; three further defects fell out of it.**
   Full account with the pre-registered probe rule:
   `reports/finding-2026-10-01-stage-sum-is-not-a-cadence.md`.
   (a) **Preprocessing** — CPU resize+normalise of 16 frames, **160.765 ms = 73.4% of compute** — is
   measured by no isolated stage benchmark and was therefore *absent*, not underestimated, from every
   budget. It is the largest single cost in the pipeline, larger than the ViT and larger than decode,
   and it is the highest-leverage optimisation target (a portable software change, unlike a clock
   pin). (b) **The tower costs 2.51x more in the loop**, and a pre-registered probe isolating the
   idle gap (no decode, no preprocessing, so contention is excluded by construction) reproduces
   **94.2%** of that, so the cause is **GPU power state**, not memory-bus contention; the arithmetic
   ceiling on pinning clocks is therefore only **+12.5%**, because preprocessing dominates and runs on
   a continuously busy CPU. (c) **A sum understates the measured cadence by 3.97x** even when
   recomputed with the in-loop decode figure. (d) **The "1.05 dec/s real-time contract" is withdrawn**
   — it assumed 25.1 fps, which 11 of 1100 clips (1.0%) have; the corpus median is 29.97 fps and
   **19.8% of clips are 60 fps**, where the margin is **1.38x**, not a comfortable multiple. (e) The
   two decode figures (5.047 vs 2.048 ms/frame) differ because the isolated one times **8-frame
   bursts on all-1080p `data/test`** while the composed one decodes **continuously on
   `data/isafetybench`**, whose resolutions are mostly 720p or below — so decode cost is
   **resolution-dependent and 2.048 ms/frame is a corpus figure, not a device constant**; a 1080p
   source would pay roughly 5 ms/frame, which by arithmetic would push the interval to ~362 ms
   (2.76 dec/s) and leave only **~1.10x** over a 60 fps 1080p contract. That case is **not measured**
   and must be stated as the open risk it is, not as a result.

10. **CAUSE FOUND 2026-10-02, FIX APPLIED, DEVICE RE-MEASURE PENDING.** Sub-defect (a) above —
   preprocessing dominating at ~10 ms/frame — has a cause, and it is a programming defect, not a
   hardware limit. `preprocess` applied the BGR->RGB channel swap as `f[:, :, ::-1]` **before** the
   resize. That is a negative-stride view, so OpenCV materialises a contiguous copy of the frame at
   **source** resolution (6.2 MB at 1080p) before downscaling to 224x224. On host that single
   sub-step is **90.3 of 101.6 ms** at 1920x1080 and **5.5 of 15.5 ms** at 480x270, and it is the
   only sub-step whose cost tracks source resolution (15.4x the pixels, 16.5x the time). Swapping
   *after* the resize is **bit-identical** — `cv2.resize` combines pixels within a channel and never
   across channels — and 7.46x / 1.36x faster at those two resolutions. Bit-equality is not assumed:
   it is leg (1) of an acceptance rule written before the first run
   (`deployment/scripts/profile-r48-preprocess.py`), re-checked on real frames at every run, and
   pinned by 11 tests comparing against the hard-coded pre-change formula. **Consequences for this
   sheet:** every Block 3b row above still describes the pre-fix system and none may be edited on
   the strength of this note; the open 1080p risk in 9(d)/9(e) is the case expected to improve most,
   which is exactly why it must be measured rather than asserted. Predictions P-PRE1..P-PRE4 with
   their falsifiers are registered in
   `reports/finding-2026-10-02-preprocessing-cost-is-a-channel-swap.md` **before** the device run.
   The device has been unreachable since 02-10 ~21:00 (ping 100% loss, ARP incomplete = powered
   off), so the re-measurement is queued, not skipped.

11. **NOT MEASURED BY DECISION, not by omission: the composed loop with clocks pinned.** The cause of
   the in-loop tower slowdown is already adjudicated (idle-droop row, 3 of 3 repeats passing a
   pre-registered rule); the arithmetic ceiling on pinning is only **+12.5%**, because the dominant
   stage is CPU-side and that CPU is continuously busy; and **pinned clocks are not the state this
   device boots or deploys in**. A cadence measured in a state the deployed system is never in would
   not be quotable. The whole Block 3b series therefore stays **unpinned**, and the +12.5% figure is
   labelled arithmetic, never measurement, wherever it appears.

### Block 3b-adjudication — pre-registered predictions P1-P4

Written 2026-10-01 in `deployment/JETSON-RUNBOOK-2026-10-01.md` **before any device number existed**,
and not renegotiated. Adjudicated only against the quiet, unpinned 22:38 runs.

| # | prediction | falsifier | measured | verdict |
|---|---|---|---|---|
| **P1** | tower batch-16 FP16 in **60-120 ms** | outside the band | **23.386 ms** (23.335-23.567, n=3) | **REFUTED** — 2.6x below the floor |
| **P2** | per decision **100-160 ms** incl. decode -> **6-10 dec/s**; B1 **MEETS** the 3.0 dec/s contract | **< 3.0 dec/s** ⇒ refuted | **composed measurement 2026-10-01: 289.672 ms/decision = 3.452 dec/s** (n=3, spread 1.21%), p95 322.708 ms = 3.099 dec/s | **CONFIRMED, but by 15% not by 130%** — 3.452 >= 3.0, so the pre-registered falsifier is not met and the verdict stands unchanged. **The evidence behind it is replaced**: the stage-sum figures this cell previously carried (144.9 ms, 6.9 dec/s) are refuted by the composed measurement, which is 4x slower (3b-note 9). The **band** 100-160 ms is **missed by 1.8x** — the prediction's point estimate was wrong even though its falsifiable claim survived, and at p95 the margin over the contract is **1.03x**. Two defects in the prediction stand recorded: it failed to specify the stride, and it was adjudicated on a sum rather than a cadence until the composed run existed |
| **P3** | backbone **>= 90%** of compute (decode excluded) | < 90% | **98.28 / 98.33 / 98.37%** | **CONFIRMED** |
| **P4** | FP16 decision agreement **100%** on the 24-clip fixture | any disagreement ⇒ FP16 latency may **NOT** be quoted beside 84.4% accuracy | **1.000**, 0 preprocessing mismatches, max abs diff 2.66e-03 — obtained twice (22:20 contended, 22:40 clean); agreement is load-independent | **CONFIRMED** — FP16 latency may be quoted beside the accuracy |

**Why P1 was wrong, stated plainly.** The 60-120 ms band came from FLOP-parity scaling off the
09-29 detector: 70.4 GFLOPs per 16-frame decision against YOLOv11s' 14.7 GFLOPs/frame at 12.90 ms.
That assumed the two graphs convert FLOPs to time at the same rate. They do not. ViT-B/32 at 224 is
a short stack of large, regular GEMMs with a batch of 16 to amortise over — close to the best case
for Orin's tensor cores — while YOLOv11s at 800 is batch-1, memory-bound, and spends much of its
time in many small layers. **A detector is the wrong yardstick for a ViT**, and no amount of care
with the arithmetic would have fixed a baseline that was wrong in kind. This is the one prediction
of the four that was quantitative rather than directional, and it is the one that failed.

## Historical — 4-class task (context only, never joined into the binary tables)

The 4-class numbers exist in `report/**` and are superseded as the headline by the binary reframe.
Any 4-class number that appears in the final thesis must first get a row here.

**P1 edge measurements (ESTA 4-class engine).** Quoted in Ch1 §1.6, Ch5 Table 5.5 and Ch6 §6.2.
Device `phuongnguyen-desktop`, Linux 5.15.185-tegra, TensorRT 10.3.0, warmup 10 / 100 runs,
measured 2026-06-03. Added here 2026-10-02 by the trace audit: these numbers were in the thesis
with no sheet row, and their artifacts were untracked (`deployment/artifacts/` is gitignored); the
artifacts are now force-added so each row resolves.

| Quantity | Protocol | Value | Spread | Artifact | Status |
|---|---|---|---|---|---|
| P1 end-to-end, TRT **FP32** | on-device, n=100 | **12.47 ms** / 80.2 FPS | p95 12.56, p99 12.64 | `deployment/artifacts/benchmarks-jetson-fp32.json` | measured |
| P1 end-to-end, TRT **FP16** | on-device, n=100 | **8.62 ms** / 116.0 FPS | p95 9.13, p99 9.21 | `deployment/artifacts/benchmarks-jetson-fp16.json` | measured |
| P1 end-to-end, TRT **INT8** (256-frame entropy calib) | on-device, n=100 | **8.24 ms** / 121.4 FPS | p95 8.58, p99 8.71 | `deployment/artifacts/benchmarks-jetson-int8.json` | measured |
| P1 4-class accuracy on the exported engines | same-dist N=64, **test-selected** | FP32 56/64; FP16 55/64; INT8 55/64 | — | same three files | measured; 56/64 is test-selected and must never be quoted as honest |
| Soak: sustained run, zero throttle | tegrastats, 2,548 samples / 2,170 active | **36.2 min active** (42.5 min wall), **0 throttle events** | — | `deployment/artifacts/soak-jetson-summary.json` | measured |
| Soak power | same | **7.03 W mean** | min 6.01 W, **max 7.34 W** | same | measured |
| Soak junction temperature | same, throttle threshold 80 °C | mean 55.88 °C (**24.1 °C below throttle**) | max 56.81 °C (**23.2 °C below**) | same | measured |
| Power/latency vs nvpmodel mode | tegrastats sweep | mode 0: 8.6 ms / 116.3 FPS / 45.69 mJ per inference; mode 1: 7.13 ms / 140.3; mode 2: 6.55 ms / 152.6 / 34.18 mJ | — | `deployment/artifacts/power-sweep-summary.json` | measured |
| Multi-stream, zero drop | fp16, 240 frames per stream | 1–4 streams all zero-drop; aggregate 24.3 / 48.2 / 70.6 FPS at 1/2/3 streams | head p50 6.86–6.98 ms | `deployment/artifacts/multi-stream-jetson-fp16-realtime.json` | measured |

**~~Open against these rows.~~ Closed 2026-10-02.** Ch6 calls the soak a *2-stream* soak;
`soak-jetson-summary.json` records duration, power and temperature but not the stream count. The
qualifier traces to a different artifact, `deployment/artifacts/soak-30min.log`, whose header reads
`=== N=2 streams, mode=fp16 === aggFPS=49.24 drops=0 elapsed=2193.509s` — 2,193.5 s = 36.6 min,
consistent with the summary's 36.2 min active. So **"2-stream" is substantiated**.

**"At 30 FPS" was not, and is corrected.** 49.24 FPS aggregate over two streams is **24.6 FPS per
stream**, not 30. The per-stream pacing target is unlogged, and the JSON that `soak-30min.log`
points to (`multi-stream-jetson-fp16.json`) was never pulled off the device, so no stronger
statement is available. Both occurrences in Ch6 now quote **49.2 FPS aggregate (24.6 per stream)**.

| Quantity | Protocol | Value | Spread | Artifact | Status |
|---|---|---|---|---|---|
| Soak stream count and rate | 2 streams, fp16, 2,193.5 s | **N=2, 49.24 FPS aggregate, 0 drops** | 24.6 FPS/stream | `deployment/artifacts/soak-30min.log` | measured |

### Historical — 4-class accuracy surface (added 2026-10-02 by the trace audit)

The trace audit found 19 4-class accuracy numbers live in `report/**` with **no sheet row**, in
breach of this sheet's own rule. Each is given a row below. None was re-run to make it fit: every
value here was read back out of an artifact that already existed, except the Stage-A detector rows,
whose training log is truncated and which were therefore **re-measured from the saved weights** on
the same held-out split (noted per row). Two corrections came out of the exercise and are recorded
at the end.

Artifact keys used in the Source column:
`LCO` = `sweep_results/relation_tokens/canonical/lco_seeds5_yolo26L1_CANONICAL.log`;
`KF` = `sweep_results/relation_tokens/canonical/kfold_repeats10_yolo26app_HEADLINE.log`;
`KF11` = `sweep_results/relation_tokens/canonical/kfold_repeats10_yoloapp_YOLO11.log`;
`LOGITS` = `sweep_results/relation_tokens/logits_yolo26.pt`;
`DEDUPE` = `sweep_results/relation_tokens/dedupe_drop_all/dedupe_drop_all_ablation.log`;
`REVAL` = `sweep_results/entity_detector/reval-2026-10-02-detector-best-val343.log`.

| # | Quantity | Protocol | Value | Source | Status |
|---|---|---|---|---|---|
| H1 | relation-Transformer, LCO | P-C LCO, 516 pooled, 8 camera-groups, 5 seeds | **401/516 = 77.7 % top-1, macro 81.2 %** | `LCO` L14 | measured |
| H2 | scene-appearance baseline, LCO | same folds; logistic regression on whole-frame MobileNetV3 features, train cameras only | **274/516 = 53.1 % top-1, macro 64.4 %** | `LCO` L16 | measured |
| H3 | honest LCO margin | H1 macro − H2 macro, bootstrap over the 516 pooled clips | **+16.9 pp, 95 % CI [+12.5, +21.2], n_boot = 5 000** | `LCO` L31 | measured |
| H4 | trivial constant floor, LCO | train global-majority predictor | 85/516 = 16.5 % top-1, macro 10.1 % | `LCO` L15 | measured; **arithmetic, not a model** — the withdrawn "+58.7 pp / +60.3 pp" framing was against this |
| H5 | scene-only, same-distribution | P-A blind, N=64 | **52/64 = 81.2 %** | `LCO` L35 | measured |
| H6 | relation model, same-distribution | P-A blind, N=64, val-selected blind rule | **54/64 = 84.4 % top-1** | `LCO` L35; `LOGITS` `blind_top1` | measured |
| H7 | LCO per-class recall, relation | P-C LCO | WalkwayViol 70.5, UnauthorInt 65.7, OpenedPanel 88.7, CarryOverload 100.0 | `LCO` L18 | measured; **descriptive only** — §7.8 finds up to ~20 pp instability |
| H8 | LCO per-class recall, scene-appearance | same | WalkwayViol 17.6, UnauthorInt 51.9, OpenedPanel 88.0, CarryOverload 100.0 | `LCO` L19 | measured; descriptive only |
| H9 | k-fold fold-mean top-1 | P-B stratified 5-fold × 10 repeats, 516 pooled | **80.8 %**, naive SE 0.58 pp, **NB-corrected SE 2.12 pp**, 95 % CI 76.7–85.0 % | `KF` L55 | measured |
| H10 | k-fold margin vs scene-only | same; Nadeau–Bengio corrected resampled *t* | Δ = +3.7 pp, t = 1.93, df = 49, **p = 0.0600 (n.s.)** | `KF` L57 | measured |
| H11 | YOLO11 ablation, k-fold | same protocol, YOLOv11 features | 80.7 ± 3.6 % fold-mean, pooled 421/516 = 81.6 %, macro 85.1 %; Δ = +3.6 pp, t = 1.95, **p = 0.0574 (n.s.)** | `KF11` L53–57 | measured |
| H12 | §7.5 scorecard — macro-recall / balanced accuracy | P-A blind, N=64, blind predictor | **88.4 %** (both) | `LOGITS` | **re-derived 2026-10-02**, see note below |
| H13 | §7.5 scorecard — macro-F1 | same | **85.6 %** | `LOGITS` | re-derived 2026-10-02 |
| H14 | §7.5 scorecard — Cohen's κ | same | **0.775** | `LOGITS` | re-derived 2026-10-02 |
| H15 | §7.5 scorecard — MCC | same | **0.785** | `LOGITS` | re-derived 2026-10-02 |
| H16 | §7.5 scorecard — ROC-AUC (one-vs-rest) | same, `blind_softmax` | **0.951** | `LOGITS` | re-derived 2026-10-02 |
| H17 | §7.5 scorecard — PR-AUC (one-vs-rest) | same | **0.880** | `LOGITS` | re-derived 2026-10-02 |
| H18 | §7.5 scorecard — ECE (15-bin) | same | **0.097** | `LOGITS` | re-derived 2026-10-02 |
| H19 | Stage-A entity detector | 343-image held-out val, 988 instances, imgsz 800 | **mAP@50 = 0.9695**, mAP@50-95 = 0.8939, P 0.934, R 0.967 | `REVAL` | **re-measured 2026-10-02** from `detector_best.pt`; matches the published 0.969 / 0.894 |
| H20 | Stage-A per-class mAP@50 | same | Person_Authorized 0.9377 (R 0.9211), Person_Unauthorized 0.9277 (R 0.9368), Panel_Closed 0.9895, Panel_Opened 0.9719, Forklift_Safe_Load 0.9950, Forklift_Overload 0.9950 | `REVAL` | re-measured 2026-10-02; all match |
| H21 | dedupe ablation (context for §7.9) | P-B k-fold after dropping 51 of 516 near-duplicate clips | 86.6 % fold-mean, pooled 400/465, p = 0.0014 | `DEDUPE` L60–64 | measured; **not a better model** — 80.8 → 85.8 pp of the gain is denominator change |
| H22 | paper baseline (Önal & Dandıl, 2024) | their protocol, not ours | 95.81 % macro, YOLO v4 trained on 2,262 boxes | citation [17] | **not our measurement**; never re-run, never compared same-protocol |

**Two corrections this exercise produced.**

1. **§7.3's detector numbers had no artifact.** `train_run.log` stops mid-epoch 20 of the 30-epoch
   freeze stage, and the only `results.csv` in the repo (`stage1_frozen/`, 3 epochs) peaks at
   mAP@50 **0.462** — three epochs of a frozen backbone, not the finished detector. An examiner
   grepping for 0.969 would have found nothing. The weights and the 343-image split were both
   still present, so the section was re-measured end to end (H19, H20) and **every published value
   reproduced**. The number was right; the provenance was missing, and now exists.
2. **§7.9 quoted the wrong p-value.** It paired the 80.8 % headline with `p = 0.057`, which is the
   **YOLO11 ablation's** p (H11, on 80.7 %). The headline run's own p is **0.0600** (H10), as
   §7.7.1 states and as `DEDUPE`'s header line records for its canonical counterpart. Corrected in
   both places in §7.9.

**On the §7.5 re-derivation (H12–H18).** §7.5 records that these five metrics were wrong twice
before — first computed on the wrong artifact, then on the wrong predictor (the 5-seed ensemble,
52/64, because the blind seed's softmax had not been persisted) — and they carried no generating
command. They were therefore recomputed on 2026-10-02 from `LOGITS` using `blind_pred`,
`blind_softmax` and `labels` (the val-selected blind rule, seed 1, 54/64), with scikit-learn.
**All nine values reproduce the thesis exactly.** The artifact's own provenance string is
`run_relation_transformer.py --tag yolo26app_ --seeds 5 --aux_w 0.5 --epochs 120`
(recovered+verified 2026-09-06). The ensemble/blind mismatch is resolved and is not re-opened.

## Citation-existence check (2026-10-02 — every cited path opened, not trusted)

85 backtick-quoted artifact paths across `report/*.md`; 5 did not resolve, all now closed.
Full table in `reports/review-2026-10-02-adversarial-defence-audit.md`. Sheet-relevant outcome:

| Key | Artifact | Status | Backs |
|---|---|---|---|
| KAPPA | `plans/260613-thesis-defense-revision/reports/kappa-result.json` | **restored** from `a93a997^` | ch04 §4.5 κ = 0.65, raw 72 %, n = 100; off-diagonal 6/20 and 7/20. Reproduce the thesis exactly. |
| BLIND1 | `.../reports/blind-eval-result.json` | **restored** | val_frac 0.20, seeds 314/42/123. direct-4 51.7 ± 1.9 (per-seed 53/53/49), macro 85.0 %. Ensemble 50.7 ± 1.3, macro 83.9 %. |
| BLIND2 | `.../reports/blind-eval-v2-vf0.15.json` | **restored** | val_frac 0.15, seeds 314/42/123/7. direct-4 51.8 ± 2.3, macro 83.1 %. |
| SEC | `plans/260430-2336-thesis-completion/security-narrative.md` | **restored** | ch05 §5.6.6 threat model / T1–T5. |
| BBOXCACHE | `sweep_results/finetune_bbox/features/bbox_test.pt` | **gone, documented** | ch04 §4.6.1. Dir absent and `.gitignore`d, so never in git. Prose now says so; Table 4.6a reads from `sweep_results/ablations/cross_ensemble.json`, which is present. |

**Correction this exercise produced.** ch04's blind-eval row quotes **51.7 ± 2.0 / 64 ≈ 80.8 %,
macro ~84 %**. Neither restored run yields exactly that: it is the two runs rounded together
(± 1.9 and ± 2.3; macro 85.0 and 83.1). The source line now states both runs separately, so the
blend is visible rather than implied. **The headline numbers are unchanged** — 51.7 and ≈80 %
hold under both.

**Figure paths.** Five callouts cited `figures/…` for files that live in
`deployment/artifacts/figures/` (`arch-diagram.mmd` — cited by *both* ch03 Fig 3.1 and ch05
Fig 5.1 — `latency-by-stage.png`, `accuracy-by-config.png`, `model-size-comparison.png`,
`fps-vs-streams.png`). All exist and are tracked; repointed. `figures/jetson-live-dashboard-demo.png`
never existed — that screenshot claim is removed, and `Figure 5.6` is no longer double-assigned.
**Caveat:** `fps-vs-streams.png` plots the same hand-reconstructed rows as Table 5.4, so it carries
that table's deferred-re-run warning, not independent support.

## Artifact-number check (2026-10-03 — every cited number read back out of its file)

Two tables disagreed with their own evidence. Both corrected; see
`reports/review-2026-10-02-adversarial-defence-audit.md` §"Artifact-number sweep".

| Key | Table | Was | Is | Artifact |
|---|---|---|---|---|
| `GATEROW` | 4.6b, bbox + frame-gate, blind | 49/64, 75.3 % | **51/64, 80.1 %** | `plans/260613-thesis-defense-revision/reports/feature-sweep-result.json` (fixed τ = 0.15, as every other row) |
| `GATETAU` | same config, τ selected on val | — | 49.33/64, 75.32 % | `plans/260613-thesis-defense-revision/reports/framegate-result.json` (`blind_val_selected_tau: 0.5`) — now disclosed in a note, not in the table |
| `RAWCOS` | 4.19, WalkwayViol ↔ UnauthorInt raw cosine | 0.834 | **0.934** (max-pair **1.000**) | `sweep_results/ablations/embedding_probe.json` → `raw_yolo_baseline` |
| `RAWCOSRANGE` | 4.19 range over all six pairs | 0.82–0.93 (4 rows) | **0.823–0.934** (6 rows) | same |
| `D4DELTA` | 4.20 Δ vs raw, hard pair | −0.459 | **−0.559** | recomputed against corrected 4.19 |

**Numeral warning.** 0.834 is now a *collision*: it was printed as the
WalkwayViol ↔ UnauthorInt raw cosine (wrong) and it is genuinely the
**UnauthorInt ↔ CarryOverload** raw cosine (right). Any surviving 0.834 in a
draft must be checked against which pair it names.

Both corrections move the thesis's own claims in the honest direction: `GATEROW`
makes the *paper's* mechanism look better, and `RAWCOS` makes the raw-feature
collapse this thesis argues for look worse. Neither touches an accuracy number.

Tables 4.20 and 4.21 were checked value-by-value against the same artifact and
match exactly. §4.3–§4.5 have **not** been checked at this level.

## Distribution check (2026-10-03 — cited vs `git ls-files`, not the filesystem)

30 of 123 cited paths resolved on disk but were absent from the repository
(directory-level ignores). 22 are now tracked (~3.5 MB total: result JSONs,
four cited figures, the tegrastats logs, the demo stream, the frame manifest);
the large binaries are **disclosed as not distributed** instead — weights and
feature caches (~300 MB), `train_run.log` (2.8 MB), `figures/gradcam/` (94 MB).
§3.6's claim that the weights "are included in the repository" was false and is
corrected. Privacy re-verified: no image or video becomes committable.

**Open gap (blocking).** Six artifacts behind the headline edge numbers exist
nowhere on this machine:
`r48-composed-cadence-fp16-20261001T{233159,233324,233449}.json` (Bảng 5.5O) and
`r48-idle-droop-probe-fp16-20261001T{234145,234242,234322}.json` (Bảng 5.5P).
They carry **3,452 decisions/s**, p95 322,708 ms, the 2,76×/1,38× margins, the
76,1 % duty factor, and the whole H-sụt-nhịp adjudication. Measured on the device
2026-10-01 and never pulled back. **Until they are rsync'd, every number in those
two tables is quoted but not re-readable.** First device action in the runbook.

## Unresolved questions

1. ~~Does the thesis headline become the iSafetyBench 75.2% or stay Unsafe-Net?~~ **RESOLVED
   2026-10-02 (framing decision).** The headline is the **R48 lane**: CLIP ViT-B/32 + temporal
   Transformer, **84.2 % balanced accuracy** (B1, N=1,100) under P-D LSO-1100 — Block 1b, quoted in
   Ch5 §5.6.9. YOLOv11s is localisation plus the edge chapter's measurement subject, not part of the
   class decision. The standing constraint is unchanged: Blocks 1 and 2 are two corpora and must
   never be averaged or compared directly (different detector, different labels, zero geometry on
   Block 1).
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
   **CLOSED 2026-09-29 23:00.** *(Numbers withdrawn 2026-10-01 — engine provenance; the verdict
   below survives, the figures do not. Current: 3.226 dec/s pinned / 2.733 DVFS at median crop
   load.)* The projection is **refuted by direct measurement**: ~~P3 runs at **3.02-3.04 dec/s**
   at median crop load and **2.60-2.72** above it~~, i.e. *on* the 3.0 boundary,
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

10. **Thermal soak, P3 detector, 30 min (2026-09-29 23:09).** **SUPERSEDED 2026-10-01 by
   Note 21 — no number in this note is quotable.** It was re-run clean on engines matching the
   device record; the replacement figures are 16.98 W mean / 18.02 W peak and tj 69.9 °C max.
   Retained for the contamination diagnosis only. Artifact
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
   trace (7.033 W mean is below tonight's 7.563 W "idle", so it looked like a lower profile — and that
   7.563 W figure is **itself now retired**: the settled baseline is **4.691 W**, which puts the
   June mean *above* idle and dissolves the anomaly entirely, strengthening this withdrawal) and
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

13. **§5.6.8 written (2026-09-30).** **HISTORICAL — every number in this note was superseded
   on 2026-10-01 (see Note 21 and the two 10-01 rows above); the table labels it cites were
   also renamed. Retained as the record of when the section was written and why.**
   Chapter 5 previously had no P3 section at all — §5.6.1-5.6.7
   are entirely P1, while chapter 4's architecture is P3, so the edge chapter did not measure the
   thesis's own model. §5.6.8 now carries Tables 5.5G (cadence by crop load), 5.5H (cost
   decomposition) and 5.5I (per-stage energy), the thermal-soak result, and an explicit
   not-measured list. Two P1 power claims in §5.6.7 and one in chapter 6 were rescoped at the same
   time: "< 8 W actual power" and "a 7 W Jetson" now state the duty-cycle dependence, because P3
   draws 17.94 W saturated and a single headline wattage misrepresents both graphs.

   **Re-verification pass, same day.** *(Historical, as above — it verified the arithmetic of a
   headline that was later withdrawn on provenance grounds, not on arithmetic grounds.)* Every
   arithmetic claim in §5.6.8 was recomputed from
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

17. **A shared device silently produced a 1.79× latency spread (2026-10-01).** Two sessions measured
    on the same Orin Nano within the same half hour. One of my runs was taken while a concurrent
    60 s energy sweep saturated the GPU, another while that session had pinned every clock with
    `jetson_clocks`, and the quiet unpinned run came third — tower 41.8 / 23.5 / 23.4 ms on one
    engine, one board, one power mode, one host commit. **All three carried byte-identical
    `device_state`**: `nvpmodel -q` reports MAXN_SUPER under a full clock pin, and
    `jetson_clocks --show` needs root so it was serialised as `null`, which is indistinguishable
    from ordinary off-device degradation. Contention was not recorded at all, because Orin's
    integrated GPU exposes no per-process accounting. Fixed in
    `deployment/scripts/jetson-clock-state-utils.py` — sysfs-based, unprivileged, with
    `clocks_pinned` returning `None` rather than `False` when nothing was read — folded into the
    shared `device_state()` so all six measurement scripts inherit it, and warning on stderr at
    measurement time. 12 tests in `tests/test-jetson-clock-state.py` against real sysfs/`/proc`
    trees on disk. The guard fired correctly on its first live contention (the R48 memory run
    self-labels `quiet: false`). Three artifacts withdrawn; see Block 3b-note 1 and
    `reports/finding-2026-10-01-concurrent-session-contamination.md`. The consequence for the rest
    of the sheet is the uncomfortable one: **no device artifact predating this commit can have
    contention or pinning excluded from its own record**, and note 8 already establishes that
    2026-09-29 was also a multi-session day. One cross-check survives: the 09-29 detector figure
    (12.903 ms) sits inside today's clean spread (12.636–13.001 ms) on the same graph and
    configuration, which is evidence that *that* row was taken quiet — for that row only.

18. **Block 3 re-measured under two labelled clock conditions; all 09-29 decision rows superseded
    (2026-10-01).** Every P3 decision-latency row in Block 3 was retaken on engines rebuilt for this
    device, three repeats per condition, with the crop load and decode source unchanged. The result
    is a **two-condition** finding, and the sheet now carries both: under default DVFS P3 misses
    3.0 dec/s at every crop load including p50 (2.733, −8.9 %); with `jetson_clocks` pinned it
    clears the contract **at p50 only** (3.226, +7.5 %) and still misses p95/p99/cap. Twelve 09-29
    rows are marked SUPERSEDED above. Full derivation in
    `reports/finding-2026-10-01-engine-provenance-invalidates-0929-latency.md`.

    **The earlier attribution in this project was wrong and is withdrawn.** The p50 PASS→FAIL flip
    was first attributed to the engine rebuild. The pinned condition disproves that: rebuilt engines
    at pinned clocks are **faster than the 09-29 stale engines on every GPU term** (detector 11.724
    vs 12.852; appearance 19.343 vs 20.274; head 0.158 vs 0.169). A rebuild that produced slower
    engines cannot produce those numbers. The DVFS deficit is a **governor** result, not an engine
    result.

    **Correction to note 17's surviving cross-check.** Note 17 keeps the 09-29 detector row on the
    grounds that 12.903 ms "sits inside today's clean spread (12.636–13.001 ms) on the same graph and
    configuration". Same engine, yes — `r48-pipeline-…T223920.json` times
    `p3-detector-fp16.engine` at 12.741 median — but **not the same configuration**: today's clean
    spread is a *standalone* batch-1 loop with no decode in the process, while 09-29's 12.903 is an
    *in-pipeline* figure with OpenCV decode running in-process. Unresolved 7 of this sheet already
    establishes that decode-in-process costs the detector ≈ +9.4 %, so the two numbers are one
    condition apart and their agreement is not evidence of a quiet device. Under the **matched**
    condition (in-pipeline, with decode, DVFS) today's figure is **14.246 ms** — 09-29 is 9.4 %
    *faster* than it, which is unexplained unless 09-29 ran at higher clocks. That is consistent
    with, but does not prove, the inference that 09-29 was a near-max-clock session; its
    `jetson_clocks_show` was null, so it cannot be assigned to either condition. **The 09-29
    detector row is superseded rather than reinterpreted**, and note 17's cross-check should not be
    relied on for it.

19. **§5.6.8 rewritten in Vietnamese (2026-10-01).** Per the standing decision that the whole report
    is Vietnamese in scientific-report register, and that new text is drafted directly in Vietnamese
    rather than translated later, the rewritten §5.6.8 is Vietnamese. Chapter 5's remaining sections
    are still English and await conversion. Table labels changed: the section's tables are now
    **5.5J** (cadence, two clock conditions), **5.5K** (cost decomposition at pinned p50) and
    **5.5L** (per-stage energy). This also fixes a real collision — 5.5G was previously used twice
    in the same chapter, by the threat-to-control table at §5.6.6 and by the cadence table. Notes 12
    and 13 above refer to the old 5.5G/5.5H/5.5I labels and to a 331.19 ms headline; both are
    historical. The chapter-wide table lettering remains out of order (5.5A–5.5L appear as B, C, F,
    D, G, A, J, K, L) and is a separate editorial cleanup.

    **The provenance gap notes 12 and 13 describe is now closed.** All 27 artifacts from the
    2026-10-01 session were rsynced from the device into
    `deployment/artifacts/jetson-measurements/`, so §5.6.8 no longer rests on any device-only
    reading: the energy table, the memory row and the n=3 spread are all locally re-openable. The
    "five unmirrored device-only files" caveat in note 13 is discharged.

20. **`*_delta_idle` energy figures are not quotable (2026-10-01).** The three P3 stage runs recorded
    idle baselines of 6.841 / 5.618 / 7.509 W — a 34 % spread, 55 % including the concurrent R48
    run's 4.842 W — because `run_energy` samples only a 10 s idle window immediately after the
    previous stage's load, so the baseline carries residual heat and clock state. Every
    `joules_per_inference_delta_idle` and `fps_per_watt_delta_idle` value in those artifacts is
    therefore **not comparable across stages and is not quoted**.

    **Repaired.** One settled 60 s baseline was taken on a quiet device — `idle_power_w` =
    **4.691 W**, 118 samples, `clocks_pinned` false on both CPU and GPU, `competing_processes`
    empty, `quiet: true`, load 0.08, 50 °C, GPU at its 306 MHz floor
    (`energy-p3-relation-head-bin-e3-fp16-20261001T225523.json`). That is 20–60 % below all three
    10 s baselines, confirming they measured decaying load rather than idle. Re-referencing the
    three stage runs to this common baseline via the spec's own
    `J/inference = (P_avg − P_idle) / fps` gives detector 0.16562 J, appearance 0.24961 J/batch,
    bin-E3 head 0.00118 J, composing to **2.90 J/decision marginal** against **4.02 J/decision
    total-board**; head share 0.041 % marginal. These are **derived** values, not artifact fields —
    §5.6.8 Table 5.5M states the formula and inputs so they recompute. Caveat: the baseline was
    taken with only the 0.190 MB head engine resident, so it is a lower bound on engine-resident
    idle and the marginal figures are a mild upper bound.

21. **The 30-minute thermal soak was re-run clean (2026-10-01 23:25).** The 09-29 soak that §5.6.8
    rested on had eight contaminated windows (a file transfer to the device ran during it), so its
    drift field was unquotable, and it was taken on engines that did not match the device record.
    The replacement run is the first soak whose artifact records its own clock state and contention:
    `clocks_pinned.any false`, `competing_processes []`, `quiet true`. Verdict unchanged and now
    better supported — **no thermal throttling over 30 minutes**: 30 windows of 60 s, 140,001
    inferences, p50 12.734–12.866 ms (spread 1.04 %), drift 0.6 % against the first window,
    junction temperature plateauing at 69.9 °C, sustained 16.98 W, and RAM flat to +30 MB. The
    suite now writes two drift fields (`latency_drift_pct_vs_first`, `..._vs_fastest`) instead of
    the single first-to-last field whose 09-29 value read −7.86 %.

    **Deliberate condition choice, and what it costs.** The soak ran under **default DVFS, not
    pinned**, because a concurrent R48 session held the device and a 35-minute pin would have
    corrupted its artifacts the way an earlier 12-minute pin already did. DVFS is also the condition
    a deployment ships in. The cost is that **the +7.5 % pinned cadence margin is still not shown to
    survive 30 minutes** — that needs an exclusive device window and remains open.

    **One result fell out of it.** Under sustained saturation DVFS drives the GPU to its maximum
    clock on its own (`cur_hz` 1,020,000,000 = `max_hz`) but leaves the CPU below maximum
    (policy0 `cur_khz` 1,113,600 of 1,728,000). The detector accordingly measures 12.796 ms/frame in
    the soak, between the 14.246 ms of the in-pipeline DVFS runs and the 11.779 ms of the in-pipeline
    pinned runs: saturation alone recovers ~59 % of the gap and pinning the rest, which places the
    residual penalty on the CPU side. **Indicative only** — the soak times the detector in a tight
    loop and the pipeline times it interleaved with CPU postprocess and decode, so the two are not a
    controlled pair. It does explain why the 09-29 soak (11.665–11.725 ms, old engine, saturated)
    looked faster than tonight's in-pipeline DVFS figures.
