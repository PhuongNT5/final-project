# Risk Register — Thesis Recovery (2026-09-19)

Extends `plans/2026-09-16-binary-safe-unsafe-reframe/risks.md` (R1-R21). **R1-R21 remain in force**
unless a status note below supersedes them. New risks start at R22. Columns: L = likelihood,
I = impact, Ph = owning phase.

## Status changes to inherited risks
| # | Inherited risk | New status |
|---|---|---|
| R1 / R13 | Detector-rule baseline ≈ the encoder; carry triviality | **Confirmed with a sharper proof.** The carry sub-task's 86 clips have geometry **identically zero** and are still 86/86 correct in all six encoders — it is solved with no relational information at all. Disclosure mandatory in every table footnote; the sentence ships in the thesis. |
| R2 | E1 does not beat E0 | **Moot.** Encoder search is cut; E3 is frozen. The parameter-efficiency wording survives and is now the *headline* framing: the smallest encoder (18,044 params) wins on LCO. |
| R3 | Accuracy explained by background alone | **Partially confirmed and localised.** The walkway sub-task shows exactly this ("walkway scene ⇒ Unsafe", 90% vs 15-36%); the zone sub-task shows it is not universal. S2 scene control stays primary. |
| R4 | Single Jetson date | **Downgraded.** Device is available repeatedly; two sessions scheduled (see R32). |
| R5-R8 | Seed variance, small test, prior shift, contamination | **Unchanged and still binding.** R8's ban on post-hoc exclusion is the reason the panel-disclosed variant is a *declared* variant reported beside the full number (see R23). |
| R9, R16, R17 | TRT/ONNX/edge budget | **Improved.** E3 exports genuinely dynamic ONNX, so the `nn.TransformerEncoder` batch-baking defect no longer blocks the headline path; it is documented as a characterised limitation. R17 stands — the cost is detector+appearance dominated. |
| R12, R13 | Panel confound, carry triviality | **Upgraded to headline findings**, with measured cross-tabs, not limitations buried in a list. |
| R14, R15, R18-R21 | Wall-clock, test touches, scope creep, number drift, deadline | **Unchanged.** R18 is now enforced by evidence rather than by discipline (see R24). |

## New risks
| # | Risk | L | I | Mitigation | Ph |
|---|---|---|---|---|---|
| R22 | **The one accuracy lever (L1 zone-entity injection) fails its 76.6% bar** | **High** | High | Pre-committed: it ships as a *written negative result* that supports the confounding claim — the mechanism was stated, tested and rejected. The thesis spine does not depend on it. No re-tuning, no second attempt, no moving the bar. | 03, 11 |
| R23 | **Panel-disclosed LCO variant read as cherry-picking** | **High** | **Critical** | Never emitted alone: it is a paired column beside the full number, with the structural justification travelling in the same artifact. Declared *before* any further experiment. Argument stated plainly: it is not selection of favourable clips, it is the removal of a sub-task that LCO cannot score (all 32 `panel_closed` clips in one camera group ⇒ zero training examples in the held-out fold). | 02, 06, 09 |
| R24 | Architecture-invariance over-read as "architecture never matters" | Med | Med | Scope the claim exactly: *invariant over six encoders spanning 18k-300k params, on this token representation, on this dataset*. It licenses stopping the search **here**; it is not a general claim about model capacity. | 02, 09 |
| R25 | Detector retrain reopened by a later agent or by the committee | Med | High | Phase 01 is the written kill record: measured mAP50 0.966 / R 0.945 for the checkpoint that built the headline tokens, with provenance. Any request to retrain must first falsify that table. | 01 |
| R26 | **Research reports carry fabricated / misattributed / misdated citations** | **Confirmed** | **Critical** | All three reports are marked UNVERIFIED in plan.md. Phase 10 runs a first-class citation audit: every arXiv ID decoded from its YYMM prefix, pass/fail table, known-bad list replaced (fabricated "Dey et al. 2024"; Nadeau-Bengio → Nadeau & Bengio, *Machine Learning* 2003; focal loss → Lin et al., ICCV 2017; SOAR → thecvf.com / arXiv:2309.01265). No citation enters the bibliography unchecked. | 10 |
| R27 | **researcher-02 §6 Q6 answer is logically inverted — defence-fatal if said aloud** | **Confirmed** | **Critical** | Excluded from this plan entirely. Phase 10 rehearses a corrected answer verbatim and records the inversion so it cannot be reintroduced from the source report. | 10 |
| R28 | researcher-02 §7.1 presents unmeasured binary numbers as measured | **Confirmed** | High | The results table is built only from the phase-06 frozen artifact. Any number without a JSON row and a log path does not exist. | 06, 09 |
| R29 | ~~Injected zone tokens starve the `N_MAX = 8` slots~~ **MOOT** — zone-entity injection cut (R39) | — | — | Geometry repowering adds channels, not tokens, so the 8-slot budget is untouched. The budget itself is still reported as a stated architectural constraint. | 03 |
| R30 | **Fold ids and view ids conflated in prose — measured agreement is only 81.7%** (fold 1 mixes views 0/3/6 at 50% purity; folds 6 and 7 are compositionally identical) | **Confirmed** | High | Claims are **fold-based** and stand unchanged; the risk is wording, not validity. Fold-based phrasing mandatory, `run_kfold_relation.py:106` `[LIMITATION]` banner attached to every LCO number, non-mixing assertion in the emitted tables, grep check in phase 10. | 02, 10 |
| R31 | Unspent Kaggle/Colab budget invites invented work | Med | Med | Retrain and architecture search are cut; hours are deliberately underspent. Only L6 (background mixing, one configuration, one falsification rule) may consume GPU time. Record the underspend honestly rather than filling it. | 05 |
| R32 | Jetson device state drifts between the two sessions | Med | Med | Both sessions run the same runbook with `nvpmodel -m 0` + `jetson_clocks` recorded as measured configuration; session 2 is confirmatory and re-measures a subset; any delta is reported, not averaged away. Every on-device block carries a **NOTIFY USER BEFORE RUNNING** marker. | 07 |
| R32b | **Measured power mode differs from the one this table specifies** | **Confirmed 2026-09-29** | Med | R32 specifies `nvpmodel -m 0`, which on Orin Nano Super is the **15 W** mode; every 2026-09-29 measurement was taken at **mode 2 = MAXN_SUPER**, the max-performance mode (recorded in each artifact's `device_state.nvpmodel_q`, so the numbers are correctly labelled — the plan text is what is stale). Consequence: the cadence PASS is a MAXN_SUPER result and must always be stated with that qualifier; it does **not** establish that P3 meets cadence in a 15 W power envelope. Fix: either re-run `--decision` at `-m 0` and report both envelopes, or restate R32 to specify MAXN_SUPER and disclose the untested 15 W case as a limit. | 07 |
| R32c | **The June P1 soak's power headroom claim rests on an unlogged power mode** | **Confirmed 2026-09-29** | Med | Unlike the 09-29 artifacts, `deployment/artifacts/soak-jetson-summary.json` (2026-06-03) records **no** `nvpmodel` state — only a hard-coded `maxn_budget_mw = 15000`. Chapter 5 Table 5.5F labels the same run "MAXN_SUPER profile" while quoting "8 W below 15 W MAXN budget", and those two cannot both be right: on this device `nvpmodel` id 0 = 15W, id 1 = 25W, id 2 = MAXN_SUPER (read from `/etc/nvpmodel.conf`, 2026-09-29). The **measurements** stand (6.01 / 7.03 / 7.34 W min/mean/max, 0 throttle events, 55.9 °C) — only the *denominator* of the headroom claim is unsubstantiated. Fix applied: Ch5 now reports the measured watts and states the profile was not logged, instead of quoting headroom against an assumed 15 W cap. Re-running the P1 soak to pin the mode is **not** scheduled — P1 multi-stream is already the stronger, published result and the temptation to re-measure it is scope creep. **Escalation attempted and withdrawn, 23:25 (finding §5g).** I briefly concluded the mode was measurably *not* MAXN_SUPER, because 7.033 W mean is below tonight's 7.563 W "idle". Both legs fail: that idle is an **idle-with-engine-resident** baseline on a hot SoC (not a floor — the June sweep reached 5.22 W at ~2 % util), and Table 5.5D's own published finding is that average power is **within 1.7 % across all three profiles**, so power cannot identify a profile at all. **This risk stands exactly as first written** and the installed conservative phrasing (absolute watts, no share-of-budget) is final; no further device measurement would settle it. Standing rule extracted: on this device board power tracks **GPU duty cycle, not the nvpmodel cap**, so no power figure in this thesis may be used to argue about budgets or profiles — only absolute draw at a stated duty cycle. | 07 |
| R32d | **Every measurement artifact chapter 5 cites as its evidence is gitignored and untracked** | **Confirmed 2026-09-30** | **High** | `.gitignore:30` ignores `deployment/artifacts/` wholesale. A handful of P1-era files (`benchmarks.json`, `multi-stream-results.json`, `parity-report.txt`, figures) were committed before that rule landed, but **everything the edge chapter actually sources is untracked**: `soak-jetson-summary.json`, `power-sweep-summary.json`, `tegrastats-soak30min.log`, `benchmarks-p3-onnx-cpu.json`, and **all** 2026-09-29/30 P3 artifacts (decision-latency x5, energy x3, memory, model-size, parity, thermal-soak). Consequences: (i) chapter 5's "Source: deployment/artifacts/…" lines name paths an examiner cannot resolve from the repository; (ii) the number-sheet's own rule — *every value verified by opening its `log_path`* — is unfollowable by anyone but the author on this one machine; (iii) the evidence for the entire edge contribution lives on exactly two unversioned filesystems (this Mac and the Jetson), so a reimage or disk loss deletes it with no history. Fix is cheap and should land before the freeze: the **48 text artifacts total 2.7 MB** (largest: the 1.2 MB thermal-soak JSON) and contain only latency/power/temperature numbers — no frames, no identifiable workers, no credentials — so `git add -f` them and add a `!` negation for `*.json`/`*.log`/`*.txt`/`*.md` under that directory, while leaving the 574 MB of `.npy` calibration caches and `.pt` feature dumps ignored. **Not committed — awaiting instruction.** | 07, 09 |
| R32e | **P3 measurement artifacts exist only on the Jetson, not mirrored** | **Confirmed 2026-09-30** | Med | Of the 2026-09-29 P3 set, only `decision-latency-…T221836.json`, `…T214556`, `…T221408`, `parity-…T221055`, `model-size-…T212708`, `latency-…T212712` and the new `thermal-soak-…T230913` reached the Mac. Still device-only: the two repeat runs `decision-latency-…T223133/223401`, `memory-…T222545`, `model-size-…T222545`, and the three `energy-p3-*` files. A `scp` brace-expansion pattern failed silently on 2026-09-30 and the device then went unreachable (~00:30), so the copy is incomplete. A second, per-file retry loop was then run and **exited 0 while every one of the seven files failed** with `scp: Connection closed` — the loop did not propagate per-file status, so a green exit code here means nothing; the mirror must be confirmed by listing the destination, never by the exit code. Mirror them at the next connection **before** quoting their values as verified; until then the n=3 spread, the energy table and the memory row in §5.6.8 are sourced from readings taken at 22:25-22:35 and recorded in the finding report, not from a locally re-openable artifact. | 07, 09 |
| R32f | **The thesis names the wrong host CPU in 24 places across 6 chapters** | **Confirmed 2026-09-30** | **High** | Both M-series artifacts (`benchmarks.json` 2026-06-13, `benchmarks-p3-onnx-cpu.json` 2026-09-16) record `machine: Phgs-Mac.lan`, `platform: macOS-14.6-arm64`. That host is *this* machine: hostname `Phgs-Mac.lan`, macOS 14.6 build 23G80, and `system_profiler` reports **Apple M3, MacBook Air, 8 cores (4P+4E), 16 GB**. Decisive: **macOS 14.6 cannot boot on any M4** (M4 requires macOS 15.1+), so no artifact bearing a 14.6 platform string was produced on an M4. The thesis nevertheless claims "Apple M4" throughout and adds false specifics that also contradict each other — ch05:166 "10-core CPU, 8 performance + 2 efficiency", ch04:61 "MacBook Pro M4 (12-core)", ch03:225 "M4 36 GB unified memory", ch03:226 "M4 Pro", plus a mechanism claim at ch05:190 attributing behaviour to "the M4 AMX matrix units". The mutually inconsistent core counts indicate these were written from assumption, not from the machine. **Not yet corrected**: the deployment rows are decisively wrong, but the *training* claims (ch03:225-226, and ch04:61's "all experiments") may refer to a genuinely different machine, so the author must confirm which hardware trained and which benchmarked before 24 lines are rewritten. Latency, FPS and accuracy values are unaffected — only the hardware attribution. | 05, 09 |
| R33 | **The accuracy push fails in full** — no lever beats the frozen E3 baseline with a CI excluding zero | **High** | High | Pre-committed wording (full verbatim text in phase 11, invoked 2026-10-04, user signs off): *"Across the levers tested — zone-entity injection, containment geometry, detector choice, appearance representation and calibration — none produced a cross-camera improvement whose confidence interval excluded zero. This is reported as a negative result. The contribution of this work is therefore threefold: an evaluation protocol that makes cross-camera failure measurable; a mechanism-level finding that the failure is a dataset camera-label confounding property, localised to identifiable sub-tasks and invariant to encoder architecture over a 16× parameter range; and a measured edge deployment of the frozen 18k-parameter model."* | 11 |
| R34 | Concurrent lane agents write outside their globs | Med | High | Ownership globs are stated in plan.md per lane and repeated in every phase file. Cross-lane writes are forbidden (reads are free); a conflict is a stop-and-escalate event, never a merge. | plan |
| R35 | ~~Interactive polygon drawing stalls waiting on the human~~ **CLOSED 2026-09-19** | — | — | L2 is cut: the 86 zero-geometry clips are exactly the carry sub-task and are already 86/86 correct in every encoder. No hand-drawing is scheduled and the user is never asked for it. Replaced by R38. | 03 |
| R38 | ~~Classes 2/3 un-instantiable ⇒ L1 dies before it starts~~ **RESOLVED 2026-09-19** | — | — | Gate run: classes 2/3 were **never annotated** (0 instances across all 6,933 labels in `labels/{train,val}`) — a genuine data gap, not lost work. The polygons are rich (7 view files, 1-3 walkway + 1-5 restricted each, full vertices at 1920×1080), so geometry work is instantiable. L1 survives, redirected to channel repowering. | 03 |
| R39 | **The protocol trap: a change that carries camera identity improves k-fold while flat or worse under LCO, and is mistaken for success** | **High** | **Critical** | Named and pre-rejected. Its archetype is zone-as-entity-token: a polygon is static per camera, so a zone token is a per-view **constant** encoding camera identity — precisely the shortcut LCO deletes. **Rule: LCO decides. Any variant that gains on k-fold without gaining on LCO is rejected, not reported**, and the k-fold/LCO divergence is itself reported as evidence of camera-bound signal. Zone-entity injection is cut on these grounds *before* being run. | 03, 06 |
| R40 | Apparent successes are detector lookups, and the thesis over-claims "relational reasoning" | **Confirmed for 2 of 4 sub-tasks** | High | Measured separability ordering (carry/panel = detector class alone; zone = class + geometry; walkway = geometry alone) is published as a headline table. Phase 02's class-blinded E3 LCO ablation bounds how strongly the zone positive result may be worded. No sentence claims relational reasoning for a pair the detector class already separates. | 02, 09, 10 |
| R36 | Committee expects an accuracy number and discounts a mechanism result | Med | High | Phase 09 leads the chapter with the finding *and* its falsifiability; phase 10 rehearses the "why is this a contribution" question directly; the edge deployment provides a second, independently measurable pillar. Acceptance remains outside this plan's control. | 09, 10, 11 |
| R37 | Post-freeze defect forces a re-run | Med | Med | T2 exists exactly for this. Any re-run re-stamps the entire frozen artifact with a new git SHA; single rows are never patched. | 06 |

## Standing rules (violations are defects, not style choices)
1. **Paired data ⇒ paired tests.** McNemar exact + discordance bound for every own-vs-own comparison;
   Nadeau-Bengio for k-fold means. `two_proportion_z` on paired predictions is banned.
2. **Selection on val/OOF only.** Thresholds, temperature, protocol variants and token-set choice are
   never selected on the 125-clip blind split.
3. **Two test touches, pre-registered, user-approved.** T1 headline, T2 reserve. Pooled
   k-fold-691/LCO-691 and all recomputation from existing OOF dumps cost 0.
4. **Equal-recipe rule.** Four-class numbers are historical context only, never a baseline row.
5. **No post-hoc exclusion** (R8). Declared variants are permitted only when stated in advance with a
   structural justification and reported beside the full number.
6. **Every falsification rule is written before the run and is not renegotiated after seeing results.**
| R41 | **iSafetyBench has the same source-label confounding** — 0/150 YouTube source videos contain both `hazard` and `normal` clips, so a model can still fit "this scene = hazard" | **Confirmed** | High | It is *diluted*, not cured: 150 groups vs Unsafe-Net's 8, so leave-source-out is statistically powered and no class is trapped in one group (hazard 76 sources / normal 74). Every headline must be leave-source-out with the group count stated, and the confounding disclosed as a property of any YouTube-collected safety corpus. | new |
| R42 | **Detector-count shortcut on iSafetyBench** — first probe shows persons/frame 1.32 (hazard) vs 2.08 (normal) and objects/frame 2.51 vs 4.48; a 2-feature count rule may match the relation head, exactly as R1 feared for Unsafe-Net | **High** | Critical | Run the count-rule baseline under leave-source-out **as a gate before any encoder training on this dataset** — the same S1 discipline. If it fires, the dataset is rejected before any time is sunk, not after. | new |
| R42-RESULT | **Gate resolved 2026-09-19: PASSES.** 300-clip balanced probe, 3 features, GroupKFold(10) over source: **63.3% BA leave-source-out** (67.0% random, 50% majority). Above chance, far from saturating -> dataset not rejected. | **Resolved** | — | The number becomes the pre-registered FLOOR: every accuracy claim on this corpus prints it beside the headline, recomputed on identical folds (`run_kfold_relation.py --dataset isafetybench` does this automatically). | 12 |
| R44 | **COCO detector quality on iSafetyBench is unmeasured** — the corpus ships no ground-truth boxes, so unlike Unsafe-Net (mAP50 0.966/0.975) there is no detection mAP to quote under the relation head | **Confirmed** | Med | State as an explicit limitation; do not imply detection quality is comparable across the two corpora. Report per-clip detection density instead (persons/frame, objects/frame), which IS measured and is also the R42 shortcut's feature set. | 12 |
| R46 | **Cross-corpus contamination** — a clip present in both Unsafe-Net and iSafetyBench would invalidate any joint claim | **RULED OUT** | — | `tools/check-cross-dataset-contamination.py`, 64-bit dHash x3 frames, 2,059 vs 3,300 hashes: **0 exact collisions, minimum cross-corpus Hamming distance 13/64** (threshold 6). Nearest pair `3_tr3.mp4` vs `MEEGwePfDgM_trim_49.mp4` is a false near-neighbour at that distance. Feasibility Q4 closed. | 12 |
| R45 | **Geometry channel is identically zero on iSafetyBench** (no zone polygons ship), so the thesis's entity-to-zone "relation" framing does not apply to the headline corpus | **Confirmed** | Med | Reframe as entity-to-entity relations on this corpus and say so plainly. Defensible only because entity-to-zone geometry was already falsified on Unsafe-Net (phase-03 kill), which must be cited in the same paragraph rather than left implicit. | 12 |
| R43 | Switching the headline dataset with ~2-3 weeks left leaves neither dataset fully finished | Med | High | Unsafe-Net work is already banked as a complete failure-analysis chapter (4 findings) and the Jetson/edge artefacts are label-independent, so they survive a switch untouched. Gate the switch on R42's probe; if it fires, stay on Unsafe-Net and ship the negative result. | new |

## Unresolved
- Whether the fold/view groupings agree (R30) — measured first in phase 02, and several narrative
  sentences depend on the answer.
- Whether T1 is worth spending at all if the cross-camera result is the headline (phase 06).
- Whether a dataset-confounding finding plus a measured deployment is accepted as a Master's
  contribution without a headline accuracy gain (R36) — not a technical question.
- **Framing assumption in force (2026-09-20, reversible until the 10-05 number freeze):** R42 passed,
  so work proceeds on the feasibility report's §6 reassignment — accuracy -> iSafetyBench, edge ->
  Unsafe-Net, Unsafe-Net -> failure-analysis chapter. This follows the binding accuracy-first
  objective order. If the user prefers "transfer experiment only", nothing built so far is wasted:
  the same runs become the transfer table.
- **G10 pre-registered (2026-09-20, before the first encoder run on this corpus):** E3 under
  leave-source-out must beat the count-rule floor recomputed on the same 1,100 clips and the same
  folds, by a margin whose paired bootstrap CI excludes 0; a margin under ~3 pp is declared not a
  result regardless of sign. Failure => written up as a negative result. No third dataset, and no
  retry with a different encoder (architecture invariance forbids it).
  **G10 outcome: PASSES** — 75.2% vs 67.2%, +8.0 pp, cluster CI [+1.9, +13.5] excludes 0, McNemar
  p = 1.02e-05. Disclosed caveat: the CI's lower bound (+1.9 pp) sits BELOW the 3 pp triviality
  threshold even though the point estimate clears it.
- **G11 pre-registered (2026-09-20, WRITTEN BEFORE `run_wholeframe_baseline.py` PRODUCED ANY NUMBER
  — the whole-frame features were still being extracted when this was committed to):** the ablations
  showed no single token channel is necessary (conf, appearance, class and box geometry are each
  individually removable within noise), which makes "does the detector buy anything at all?" the
  decisive committee question. So: E3 under leave-source-out must beat the **best** of the
  whole-frame frozen-feature probes (MobileNetV3-small ImageNet -> fixed 64-d projection, identical
  16-frame sampling, identical clips, identical folds; best-of-three, not best-for-us) by a margin
  whose **cluster-bootstrap 95% CI over the 151 sources excludes 0**, with the same ~3 pp triviality
  threshold on the point estimate.
  - If the CI includes 0, or the point margin is under 3 pp: **the thesis may not claim the detector
    + relation head add accuracy on this corpus.** iSafetyBench is then written up as a parity /
    negative result, the accuracy pillar reverts to a stated limitation, and the edge-efficiency
    pillar carries the contribution. The +8.0 pp over the count-rule floor survives as a statement
    about the shortcut, not about whole-frame CNNs.
  - No retry with a different encoder, a different backbone, a different pooling, or a tuned
    threshold. The three probes and the metric are fixed as written here.
  - G11 failing does NOT retract G10; they answer different questions (shortcut vs whole-frame).
  **G11 OUTCOME: FAILS.** Best-of-three whole-frame probe = linear-mean, balanced acc **74.1%**
  (linear-meanstd 72.7%, mlp-meanstd 73.6%). E3 = 75.2%. Paired on the same 1100 clips / same 10
  leave-source-out folds: **+1.1 pp**, cluster-bootstrap 95% CI over 151 sources **[-4.8, +6.9]
  (includes 0)**, McNemar exact b=138 c=129 n_disc=267 **p=0.625**. Both pre-registered criteria
  fail — the point margin is under the 3 pp triviality threshold AND the CI includes 0.
  Consequence, applied as written and not renegotiated: **the thesis may NOT claim the detector +
  relation head add accuracy on iSafetyBench.** iSafetyBench is written up as a parity / negative
  result against a whole-frame CNN baseline; the accuracy pillar reverts to a stated limitation on
  this corpus; the edge-efficiency pillar carries the contribution. G10's +8.0 pp survives only as a
  statement about the count-rule shortcut, not about whole-frame CNNs.
  - Consistent with the §5 ablation finding: `appearance_only` already matched the full model, so the
    detector's 10 geometry/class/conf columns were carrying nothing. G11 closes the loop — the
    per-entity crops themselves also carry nothing over the whole frame. What works here is
    ImageNet appearance statistics, with or without a detector.
  - **E0 may not be used to rescue this.** The pre-registration forbids retry with a different
    encoder; E0's LSO run (started before G11 was adjudicated) is reported as an architecture-
    invariance row only, whatever its number.
  - **No efficiency escape either.** "Equal accuracy at 17k params" is false: 17k is the head only,
    and E3's input costs a YOLOv11s pass + up to 8 crop-CNN passes per frame vs the baseline's single
    whole-frame CNN pass. The detector-free baseline is equally accurate AND cheaper. Only an
    output-granularity (per-entity localisation) argument survives, and it is unvalidated.
  - **E0 OUTCOME (2026-09-20, after G11 was adjudicated): 80.3% BA.** E0 - whole-frame = +6.2 pp,
    cluster CI [+2.1, +10.5], McNemar p=1.59e-05. The arithmetic clears G11's thresholds.
    **G11 nonetheless remains FAILED**: it named E3, and substituting the encoder after the
    designated one fails is exactly what the rule forbids. Evidence E0 was not launched in reaction:
    `lso-isb-E0.log` born 19:08:43 vs first whole-frame number 20:07:54. Evidence against promoting
    it: phase-12 step 5 planned E0 as an invariance CROSS-CHECK, not a headline candidate, so the
    decision rule "report whichever encoder beats the baseline" was never declared. Both facts are
    reported together, failure first. Author/committee decision; not this lane's to resolve.
- **R47 — ARCHITECTURE INVARIANCE IS FALSIFIED AS A GENERAL CLAIM (2026-09-20).** E0 - E3 on
  iSafetyBench = +5.1 pp, cluster CI [+2.0, +8.3], McNemar p=7.74e-06. The project's invariance
  finding (six encoders, 18k-300k params, identical pass/fail) holds on Unsafe-Net only. Correct
  scoped statement: capacity was irrelevant where the task had no extractable signal (camera-
  confounded labels) and decisive where it did. Invariance was a symptom of unlearnability, not a
  property of the architecture. **Every document asserting it generally must be scoped**:
  `phase-04` L23, `phase-09` step 5 (chapter-07 subsection), `phase-11` narrative template,
  `phase-12` L26. Also: E3 is the frozen deployment artefact and all Jetson/ONNX measurements are
  E3's; headlining E0 costs a device re-measurement (`p3-relation-head-bin-e0-fp32.onnx` exists,
  no Jetson numbers for it).

- **G12 pre-registered (2026-09-20, WRITTEN BEFORE `--split random` EXISTS IN THE CODE AND BEFORE ANY
  RANDOM-SPLIT NUMBER HAS BEEN PRODUCED):** the protocol-inflation measurement. Same tokens, same
  encoders (E3 and E0), same 10 folds, same epochs — the ONLY change is that folds are drawn with
  `StratifiedKFold` ignoring the YouTube source id, i.e. the protocol the baseline literature uses.
  Reported quantity is **Δ_inflation = BA(random split) − BA(leave-source-out)** per encoder, with a
  cluster-bootstrap CI over the 151 sources and exact McNemar on the paired per-clip predictions.
  - **Prediction:** Δ_inflation > 0 and materially large (>= 5 pp) for at least one encoder, because
    0/151 sources carry both classes, so a random fold lets the model memorise source identity as a
    perfect label proxy.
  - **Falsification:** if Δ_inflation < 3 pp for BOTH E3 and E0 with CIs including 0, then protocol
    choice does not inflate accuracy on this corpus, the protocol-critique contribution may NOT be
    claimed from this experiment, and it is written up as a null.
  - The random-split number is a **diagnostic artefact only**. It must never be reported as this
    thesis's accuracy, in any table, abstract or slide. Every appearance carries the label
    "source-leaking protocol, shown for comparison".
  - Fixed now: encoders E3 + E0 only, 10 folds, seed 0, no threshold tuning, no re-runs for a better
    Δ. 0 test touches.
- **G13 pre-registered (2026-09-20, WRITTEN BEFORE E1/E2 HAVE BEEN RUN ON iSafetyBench):** the direct
  test of R47's mechanism claim ("invariance was a symptom of unlearnability, not a property of
  encoders"). Run the capacity ladder E3 (17,270) -> E1 d=64 -> E1 d=96 -> E0 (298,118) under
  leave-source-out on iSafetyBench, identical folds.
  - **Prediction:** balanced accuracy increases with capacity on iSafetyBench (signal present),
    while the already-measured Unsafe-Net sweep over the same encoder family shows no such trend
    (signal absent). The claim is the CONTRAST between corpora, not either curve alone.
  - **Falsification:** if the iSafetyBench ladder is flat (max − min < 3 pp, or the E0−E3 gap is not
    reproduced by the intermediate encoders showing an ordered trend), R47's mechanism claim is
    withdrawn and E0's advantage is reported as an unexplained single-encoder effect, not as
    evidence about learnability.
  - This does NOT re-run G11 and may not be used to revive it. G11 named E3 and stays failed.
  - Fixed now: those four points, 10 folds, seed 0, no tuning. 0 test touches.

- **R48 / G14-G16 pre-registered 2026-09-29, WRITTEN BEFORE ANY CLIP FEATURE HAS BEEN EXTRACTED AND
  BEFORE THE ARCHITECTURE EXISTS IN CODE.** New direction, at the author's instruction: drop the
  protocol critique as the thesis's headline and pursue a constructive architecture contribution for
  unsafe-behaviour detection (YOLOv11 detection + Transformer). This is a NEW experiment with NEW
  features, not a re-adjudication of G11 - G11 named E3 on 64-d MobileNet features and stays FAILED
  regardless of what happens here. Declaring a new gate on a new architecture is legitimate; moving
  an old gate's goalposts is not, and is not what this is.

  **Hypothesis (R48):** the accuracy ceiling on iSafetyBench is the *appearance representation*, not
  the relational head. Evidence assembled before the gate: (a) the seven-condition ablation showed
  `appearance_only` matches the full model (75.6% vs 75.2%, p=0.551) - all ten FEAT columns are
  dispensable and the signal is entirely crop appearance; (b) that appearance is MobileNetV3-small
  features crushed through a **fixed random projection 576 -> 64**; (c) the whole-frame G11 baseline
  uses the identical crippled projection, so G11 compared two handicapped arms; (d) at 64-d the
  entity pipeline nevertheless beat whole-frame by +6.2 pp (E0, p=1.59e-05), so entity structure
  carries signal that survived the bottleneck.

  **Architecture under test (fixed now, no search, no tuning):** per-entity tokens from YOLOv11s
  (COCO-80 -> 8 coarse groups, unchanged), each token = frozen **CLIP ViT-B/32** crop embedding
  (512-d, 30% context padding, 224x224 - the existing crop geometry) + box (4) + entity-class
  embedding + temporal index; **factorised space-time attention** (ViViT-style): stage 1 attends over
  entities within a frame to a per-frame scene state, stage 2 attends over the T=16 frame states to a
  clip state; binary head. T=16, N_MAX=8 unchanged so the clip set and folds are byte-identical to
  every existing run.

  **Baselines, all backbone-matched - this is the load-bearing control.** A win over a weaker
  backbone would be a backbone result, not an architecture result:
  - **B1 whole-frame CLIP** + the *same* stage-2 temporal Transformer, no detector, no entity axis.
    Isolates the entity decomposition with the representation held constant.
    - **Amended 2026-09-29, BEFORE ANY MODEL HAS BEEN TRAINED (no result seen).** Deleting the
      entity axis also deletes its parameters: ESTA is 352,258 and the structural B1 is 201,090, so
      a raw ESTA win would be open to "you simply gave it more capacity". B1 is therefore
      **best-of-two**: the structural B1 (1 temporal layer, 201,090) and a capacity-matched B1
      (2 temporal layers, ~332k, within 6% of ESTA). The *higher* of the two is the baseline G14
      must clear. This is the same best-of-N convention the G11 whole-frame probe used, and it moves
      the bar against us, not for us.
  - **B2 frozen VideoMAE-base** (16 frames, matches T) + the same temporal head - a modern
    detector-free video Transformer, frozen like ours so the comparison is fair.
  - **B3** the existing E0/E3 at 64-d - measures what the representation upgrade bought.
  - **B4** the R42 count-rule floor, unchanged, on identical folds.

  **Protocol, fixed now:** LSO-1100 `StratifiedGroupKFold(10)`, seed 0, 3 seeds/fold, 60 epochs, no
  threshold tuning, no per-fold model selection, no re-runs for a better number. 0 test touches.
  Every comparison paired: cluster bootstrap over the 151 sources + exact McNemar.

  - **G14 (THE GATE - the entity decomposition must earn its place):** ESTA minus B1 >= **+3 pp**,
    cluster CI over 151 sources **excluding 0**, McNemar p < 0.05.
    - **Falsification:** margin < 3 pp, or CI includes 0. Then the entity decomposition adds nothing
      over a whole-frame representation *at matched backbone*, the detector is not justified on
      accuracy grounds on this corpus, and that is final - no third architecture, no fourth
      baseline, no swapping in a different encoder after the designated one fails. It is written up
      as a negative result and the accuracy pillar closes for good.
  - **G15 (component ablation, NOT the contribution):** ESTA minus E0@64-d >= +3 pp with CI
    excluding 0. Tests R48's representation-bottleneck hypothesis. Failing G15 while passing G14
    would mean the gain came from the head, not the features - reportable, but it falsifies R48.
  - **G16 (the honest strong opponent):** if **B2 VideoMAE >= ESTA** with a CI excluding 0, the
    entity pipeline is not justified even if G14 passes, and the thesis must say so plainly rather
    than omit the baseline. B2 is reported whatever it shows.
    - **Fixed 2026-09-29, BEFORE B2 HAS BEEN TRAINED (no result seen).** B2's backbone is
      `MCG-NJU/videomae-base-finetuned-kinetics`, not the self-supervised `MCG-NJU/videomae-base`
      that happened to be in the local cache: an MAE encoder never trained on a labelled action task
      is a weak opponent, and beating a weak opponent would prove nothing about ESTA. B2 sees the
      same 16 frames of the same clips (shared `frame_sampling.decode`), emits 8 tubelet states of
      768-d, and feeds the SAME temporal Transformer B1 uses - it differs from B1 only in backbone,
      and from ESTA in backbone plus the entity axis.

  - **G17 (detector generation - a REPORTED ABLATION, explicitly NOT a selection). Pre-registered
    2026-09-29, before any YOLO26 feature has been extracted.** The entity arm is re-extracted with
    `yolo26s.pt` (10,009,784 params, 80 COCO classes, `end2end=True` - NMS-free) in place of
    `yolo11s.pt`, everything else held fixed: same frames, same CLIP tower, same crops rule, same
    folds, same recipe. ESTA is then re-run under the identical protocol.
    - **The headline detector stays YOLOv11s whatever this shows.** Reason, stated in advance: the
      frozen deployment artifact, the ONNX exports and every edge number already measured are
      YOLOv11s, and with the thesis deadline inside two weeks there is no time to re-measure the
      edge lane on a second detector. Taking the accuracy of whichever detector scored higher would
      be selection on cross-validation with 0 test touches - it is forbidden here.
    - What G17 buys is a sensitivity statement ("the result does/does not depend on the detector
      generation") and, for the edge chapter, an NMS-free end-to-end detector to measure against.
      The YOLO26 number is reported whichever direction it points, including if it is higher.

  **Integrity log, 2026-09-29 (recorded the moment it happened, not afterwards).** Before the
  headline runs were launched, a 2-epoch / 1-seed plumbing check of the ESTA driver was run on the
  full LSO-1100 folds to verify the fold construction, the feature permutation and the OOF writer.
  It printed balanced_acc 82.9%, so **an ESTA number was seen before the pre-registered run
  started**. Nothing was changed in response: the protocol below (60 epochs, 3 seeds, the same
  folds, the same recipe as the 64-d encoders) was already fixed, and the check's artifacts carry a
  `_smoke` suffix so they can never be mistaken for or overwrite a headline file. A 2-epoch,
  1-seed, off-protocol number is not evidence and is not quotable as a result. It is logged here
  only so that nobody has to take on trust that the protocol was not tuned after seeing it.

  **Declared risk, recorded before the run:** B1 is expected to improve substantially too, and the
  gap may close - this is exactly how G11 failed once already. Estimated probability G14 passes:
  roughly even. If it fails, the deliverable is a second, better-controlled negative result, and the
  author must be told plainly rather than shown a fourth attempt.

- **G12 OUTCOME 2026-09-21 - PASSES.** Delta_inflation E3 **+15.4 pp** (cluster CI over 151 sources
  [+10.6, +21.5], excludes 0; McNemar b=186 c=8 p=3.58e-45); E0 **+10.4 pp** ([+6.3, +15.3]; b=150
  c=22 p=1.3e-24). Prediction (>= 5 pp for at least one encoder) met by both; the falsification
  condition (< 3 pp for BOTH with CIs including 0) is not approached. Full 2x2 in
  `reports/finding-2026-09-21-protocol-inflation-and-the-capacity-contrast.md`.
  - **Disclosure:** G12 fixed "encoders E3 + E0 only". E1 d64 (+13.2 pp) and E1 d96 (+14.9 pp) were
    also run under the random split to complete G13's contrast. The verdict is adjudicated on E3 and
    E0 as written; the E1 rows are disclosure and could only have weakened it.
  - **The decisive control:** the 3-feature R42 count rule, which has no capacity to memorise source
    identity, gains only **+1.0 pp** (67.2% -> 68.2%) from the same protocol change. Inflation is
    identity memorisation, not split arithmetic.
  - **The result that outranks the predicted one:** leakage is not merely inflationary, it is
    **non-discriminative**. E0 minus E3 (a 17x capacity gap) = **+5.1 pp, p=7.74e-06** under LSO-1100
    but **+0.1 pp, CI [-1.2, +1.5], p=0.665** under RAND-1100; the widest random-protocol pair on the
    whole ladder is +1.4 pp, CI [-0.4, +3.0], p=0.105. A leaking protocol cannot support ANY
    architecture comparison, favourable or not. This invalidates the comparisons in such papers, not
    only their absolute numbers.
  - The random-split numbers stay diagnostic artefacts under the original constraint: never this
    thesis's accuracy, always labelled "source-leaking protocol, shown for comparison". The
    constraint is inherited by every downstream document.

- **G13 OUTCOME 2026-09-21 - DOES NOT PASS CLEANLY (conservative reading).** iSafetyBench LSO ladder:
  E3 17,270 -> 75.2%, E1 d64 93,222 -> 78.1%, E1 d96 195,558 -> **77.0%**, E0 298,118 -> 80.3%.
  - Flatness clause does NOT fire: spread 5.1 pp > 3 pp.
  - Ordered-trend clause DOES fire: E1 d96 sits **1.1 pp below** E1 d64 at double the parameters. The
    ladder is not monotone.
  - The pre-registered phrase "ordered trend" was ambiguous - monotonicity vs positive association
    was never stated. **That ambiguity is my drafting error**, and resolving it after seeing results
    in the direction that favours our own claim is precisely the move the no-renegotiation rule
    exists to prevent. Conservative reading adopted.
  - **Permitted claim:** "capacity class matters where extractable signal exists, and does not where
    it does not" - iSafetyBench +5.1 pp (p=7.74e-06) against Unsafe-Net LCO-691 -2.8 pp (CI
    [-7.1, +0.1] includes 0, p=0.606). The claim is the CONTRAST, as pre-registered.
  - **Forbidden claim:** "balanced accuracy increases with parameter count". Must not be smoothed
    over by plotting only the E3 and E0 endpoints.
  - Does not revive G11. G11 named E3 and stays failed.
  - R47 survives only in its already-scoped form (invariance = symptom of an unlearnable task on
    Unsafe-Net, not a property of the encoder family).
  - **Caveat on the Unsafe-Net side:** its cluster bootstraps use **8** clusters and are thin; in the
    k-fold-691 cell McNemar (p=0.0271) and the cluster CI ([-1.6, +5.2]) disagree and the
    conservative CI reading is taken. Disclosed, not resolved.

- **DEFECT FOUND AND FIXED 2026-09-20 (`run_kfold_relation.py` L199, pre-existing):** the R42 count-rule
  floor rebuilt its own folds with `ISB.folds(...)`, which ignores `--split`. Under `--split random`
  the encoder was scored on random folds while the shortcut was scored on leave-source-out folds, and
  the log still printed "[R42 floor, identical folds]". The first G12 E3 log
  (`g12-rand-isb-E3.log`) carries that wrong line: it says floor 67.2% / margin +23.4 pp; the correct
  random-fold floor is **68.2%** and the margin **+22.4 pp**. The encoder's own 90.6% is unaffected
  (it always ran on the right folds). Fixed by deriving the floor's splits from the encoder's `folds`
  (set complement) so identity holds by construction, not by re-derivation; regression asserted in
  `tests/test-split-protocol-modes.py`. Affected log to be regenerated; no other run is affected,
  because for `--split group` the rebuilt folds were identical to the encoder's.

- **EXECUTION NOTE 2026-09-29 (ESTA run sharded across 4 processes; no arithmetic change).** ESTA
  runs ~5x slower per fold than the whole-frame baselines: its spatial stage is `b*t = 512`
  sequences of length 9, so the workload is latency-bound on many small matmuls, not thread-bound.
  I first misdiagnosed this as thread starvation and relaunched ESTA with `OMP_NUM_THREADS=6`; CPU
  stayed at 109%, the restart bought nothing, and ~25 min of compute was lost. The abandoned log is
  kept as `logs/r48-lso-ESTA-abandoned-thread-starved.log`. The actual lever is **fold-level
  parallelism**: `run-esta-lso.py --shard i --nshards 4` slices the same `ISB.folds(y, groups,
  n_splits=10, seed=0)` list `folds[i::4]`, so shard 0 runs f1/f5/f9, shard 1 f2/f6/f10, shard 2
  f3/f7, shard 3 f4/f8 — the same folds, the same clips, the same seeds, the same 60 epochs, the
  same arithmetic, in four processes instead of one.
  - `tools/merge-esta-shards.py` scatters the shards back into clip order and **refuses to merge**
    unless every clip is covered exactly once (a lost shard, a duplicated fold, a shard-count
    mismatch, or shards run against different feature files all abort). It then calls
    `run-esta-lso.finalise`, the same function the unsharded path calls, so a merged artifact and an
    unsharded one are identical by construction rather than by inspection.
  - Guards asserted in `tests/test-merge-shards-coverage.py` (5/5 pass).
  - **This is a scheduling change, disclosed here so it is on the record, and it is not a protocol
    amendment**: no fold, split, seed, epoch budget, threshold or selection rule moved.

- **G14 ADJUDICATED AND FAILED 2026-09-29.** ESTA 83.8% vs B1w 84.4% = **-0.5 pp**, cluster 95% CI
  [-5.5, +5.0], McNemar b=72 c=78 **p=0.683**. ESTA is behind, not merely short of the margin, and
  the discordance is symmetric. The pre-registered consequence is applied verbatim and is **not**
  renegotiated: the entity decomposition adds nothing over a whole-frame representation at matched
  backbone. G14 is closed. ESTA may appear in the thesis only as a falsified hypothesis.
  - **G15 FAIL on the CI rule**: ESTA - E0@64d = +3.5 pp, McNemar p=0.00148, but cluster CI
    [-2.3, +10.0] includes 0. The gate required *both*; the margin alone does not carry it. The
    McNemar/CI disagreement is reported as the finding, not resolved in our favour.
  - **G16 no separation**: ESTA - B2 = +2.3 pp, CI [-2.3, +7.5], p=0.083. B2 does not beat ESTA, so
    the G16 failure clause does not fire.
  - Descriptive, non-gate: B1w - E0@64d = +4.0 pp (McNemar p=3.01e-04, cluster CI [-0.9, +8.8]);
    B2 - E0@64d = +1.2 pp (p=0.184). **Every cluster CI in this lane includes 0** while the
    clip-level CIs exclude it. At 151 sources with 0 carrying both classes (R41), between-source
    variance dominates and no 3-4 pp margin is certifiable on this corpus at cluster resolution.
    Quoting the clip-level CI as the headline is forbidden — it is the same protocol inflation this
    project already documented against itself.
  - **Forbidden claims:** "entity-aware attention improves unsafe-behaviour recognition"; any margin
    quoted from a clip-level CI; any post-hoc lowering of the 3.0 pp margin or dropping of the CI
    requirement now that the margin alone would have passed G15.
  - **Permitted claim:** the representation swap (fixed random 576->64 projection -> frozen CLIP
    ViT-B/32) lifts LSO-1100 balanced accuracy 80.3% -> 84.4%, with the source-clustered CI too wide
    to certify the margin — stated with that caveat attached, every time.

- **DEFECT FOUND AND FIXED 2026-09-29 (`tools/adjudicate-r48-gates.py`, display only).** `paired()`
  assigned the key `'b'` twice — first the baseline run's name, then McNemar's discordant count b —
  so the printed table showed `ESTA 83.8% - 72 84.4%` instead of the baseline's name. Counts renamed
  `mc_b`/`mc_c`. **No verdict was affected**: the PASS/FAIL logic reads `diff`, the CI exclusion flag
  and `p`, none of which collided. The first adjudication log was regenerated after the fix.

- **DEFECT FOUND AND FIXED 2026-09-29 (`demo/esta-window-pipeline.py`, would have crashed).** The
  demo unpacked `crop_box(...)` as four coordinates, but that function returns a cropped *image*; the
  demo would have raised on its first detection. Geometry split into `box_px()` in
  `build-isafetybench-clip-features.py`, with `crop_box` now slicing with it. Because that file
  produced every measured feature tensor, `tests/test-crop-geometry-matches-token-builder.py`
  asserts pixel-identity against the pre-refactor expression over 700 boxes, 259 of them degenerate
  (2/2 pass). No feature tensor changes.

- **G17 ADJUDICATED 2026-09-29 (detector generation).** ESTA re-run end-to-end on YOLO26s entity
  features: **81.6%** vs **83.8%** on YOLOv11s = **-2.2 pp**, McNemar b=75 c=43 **p=0.00412**,
  cluster CI [-1.7, +6.0] includes 0. The newer detector generation makes this task **worse**, not
  better: at conf 0.25 YOLO26s emits 45,512 detections over 87.9% of frames against YOLOv11s's
  52,997 over 90.2%, so the entity arm is fed a sparser scene.
  - **This closes the defence question "was your detector the problem?"** No. G14's failure is not
    detector-limited — a newer, larger, NMS-free detector moves the entity arm further from the
    whole-frame baseline (B1w - ESTA@yolo26s = +2.7 pp, McNemar p=0.0044), not closer.
  - The pre-registered rule ("the headline detector stays YOLOv11s whatever YOLO26 shows") is
    honoured, and it happens to cost us nothing. **That is luck, not method**, and is recorded as
    such: the rule was written before the run and would have bound us the other way.
  - Fairness control: the whole-frame arm of `feats_isbclip26_all.pt` is **byte-identical**
    (max |diff| = 0.0) to `feats_isbclip_all.pt`, because no detector touches it. So B1/B1w need no
    re-run and the only changed variable between the two ESTA runs is the detector.
  - YOLO26s stays in the thesis as the **edge** subject (`end2end=True`, NMS-free graph), never as
    an accuracy claim.

---

## R49 — two-stream fusion (TSF): the direction G14's failure actually opened

### The diagnostic that motivated it, and why it is NOT a result

2026-09-29, `scratchpad/complementarity.py` on the six existing LSO-1100 OOF vectors (no training,
no new compute). Error-correlation phi over the correctness vectors:

|            | ESTA | B1 | B1w | B2 | E0 |
|---|---:|---:|---:|---:|---:|
| **ESTA**   | 1.00 | 0.43 | **0.49** | 0.35 | 0.37 |
| **B1w**    | 0.49 | **0.86** | 1.00 | 0.45 | 0.37 |

The entity stream and the whole-frame stream fail on **different clips** (phi 0.49), while two
whole-frame variants of the same family agree at 0.86 — which is the control that says the metric
is measuring what it claims. ESTA+B1w oracle ("either stream right") = **91.2 %** against 84.4 %
best-single, i.e. **+6.8 pp of headroom**, with only 9.2 % of clips both-wrong.

**BAN — the fusion scores from that script may never be quoted, in the thesis or anywhere else.**
I swept 6 pairs and 3 triples and read all nine scores before choosing. The best of nine
(+3.1 pp) is a selection statistic on the evaluation vectors, not a measurement. It is recorded
here as **hypothesis-generating only**. The number that may be reported is the one produced by the
pre-registered run below, whatever it turns out to be.

### G18 — PRE-REGISTERED 2026-09-29, BEFORE the model exists or any fusion is trained

**Claim under test:** the entity decomposition is not a *replacement* for a whole-frame
representation (G14 settled that, FAIL), but it is a *complementary view*, and a two-stream model
that keeps both beats the whole-frame stream alone.

**Configuration, fixed now and not to be tuned after seeing results:**
- Streams: entity (YOLOv11s crops -> frozen CLIP ViT-B/32 -> ESTA factorised attention) +
  whole-frame (frozen CLIP ViT-B/32 -> B1 temporal transformer). **Exactly the two arms already
  built** — no new feature extraction, no new backbone.
- Fusion: learned, inside the model, trained jointly with both heads. Not post-hoc probability
  averaging (that is what the banned sweep did).
- Detector: **YOLOv11s**, per the standing rule (G17). YOLO26s is edge-only.
- Protocol: LSO-1100, `StratifiedGroupKFold(10, seed=0)`, the identical folds/clips/recipe as every
  Block 1b row. 3 seeds. Baseline = **B1w (84.4 %)**, the incumbent best single stream.
- Statistics: cluster bootstrap over the 151 sources + exact McNemar. **0 test touches.**

**Verdicts, written before the run:**
- **G18a (strong) PASS** — diff >= +3.0 pp AND cluster CI excludes 0 AND McNemar p < 0.05.
  Permits: "the two-stream architecture outperforms the whole-frame baseline."
- **G18b (weak) PASS** — diff >= +2.0 pp AND McNemar p < 0.05, cluster CI may include 0.
  Permits **only**: "the two streams are complementary and fusion improves accuracy on this
  corpus", stated together with the R41 caveat that at 151 sources with 0 carrying both classes,
  source-level generalisation of a margin this size is not certifiable. Forbidden under G18b: any
  unqualified claim of superiority, and any use of the clip-level CI as the headline.
- **FAIL** — neither tier met. Consequence, accepted in advance: the entity stream adds nothing
  even as a complementary view; the thesis headlines B1/B1w, and YOLO's role is localisation plus
  the edge chapter. No renegotiation, no third tier invented afterwards.

**Why this is the right use of the remaining days:** it is the only open direction that puts
YOLOv11 back in the *accuracy* path on measured evidence rather than on assumption, it reuses
features that already exist (hours, not GPU-days), and both of its outcomes are publishable — a
PASS is an architecture, a FAIL closes the question with a paired test instead of a hunch.

### G18 capacity control — PRE-REGISTERED 2026-09-29, before any TSF fold had been scored

TSF is 686,726 params against B1w's 333,570, so "the two-stream model is simply bigger" is the
first objection a committee will raise. The existing evidence against it is that capacity did not
buy accuracy anywhere else in this lane: B1 (201,090) -> B1w (333,570) moved balanced accuracy by
**+0.2 pp**. That is suggestive but only spans a 1.66x increase, while TSF is 2.06x over B1w.

So a matched control is added and fixed now: **B1xl**, a whole-frame model selected by the rule
*"of the whole-frame configurations, the one whose parameter count is closest to TSF's"* — which
resolves to d=192, ff=384, t_layers=2 at **696,962 params, 1.5 % LARGER than TSF**. Same folds,
same recipe, same 3 seeds, same 0 test touches. The rule was applied to parameter counts alone,
with no model trained and no accuracy visible.

**How it binds the G18 verdict:** if G18a or G18b passes, the claim is additionally required to
survive TSF vs B1xl at the same tier. If TSF does not beat a *larger* whole-frame model, the
fusion result is reported as a capacity effect and the entity stream gets no credit — regardless
of what TSF vs B1w showed. This condition is part of G18 as of now and is not renegotiable.

### G18 ADJUDICATED AND FAILED — 2026-09-30

TSF **84.1 %** vs B1w **84.4 %** = **-0.3 pp**, cluster CI [-3.7, +2.5] includes 0, McNemar exact
b=22 c=29 **p=0.401**. Neither tier met. LSO-1100, identical folds/clips/recipe, 3 seeds,
**0 test touches**.

**Pre-registered consequence, applied verbatim and not renegotiated:** the entity stream adds
nothing even as a complementary view. The thesis headlines B1/B1w; YOLO's role is localisation plus
the edge chapter.

**Why it failed, and this is the interesting part.** The complementarity was real — ESTA and B1w
disagreed on **150 clips** (b=72, c=78) with an oracle of 91.2 %. TSF disagrees with B1w on only
**51** (b=22, c=29), and with its own entity arm at b=81 c=82, **p=1.000**. So the fusion did not
split the difference between two views: it **collapsed onto the whole-frame stream** and kept the
entity arm only as a passenger, despite the auxiliary head holding lambda=0.3 of the loss.

The lesson is precise and worth stating in the thesis: **oracle headroom is not reachable without
an oracle.** A +6.8 pp gap between "either stream is right" and "the better stream" only converts
into accuracy if something at training time can predict *which* stream to trust on a given clip.
At 1,100 clips over 151 sources, no such signal was learnable. This is a statement about the
selector, not about the streams.

**Capacity condition — also FAIL, and it does not rescue anything.** TSF - B1xl = +1.4 pp
(CI [-0.3, +3.2] includes 0, p=0.121). TSF does beat a 1.5 %-larger whole-frame model numerically,
but not at either registered tier, so under the registration the capacity control sustains no
claim. **Forbidden:** quoting +1.4 pp over B1xl as evidence for the fusion. It was registered as a
condition on a passing claim, and there is no passing claim.

**What the capacity controls DO establish (independent of G18).** B1 201,090 -> 84.2 %; B1w
333,570 -> 84.4 %; B1xl 696,962 -> **82.7 %**. Over a 3.5x parameter range the whole-frame arm
gains +0.2 pp and then loses 1.7 pp. Capacity is ruled out as the mechanism in this lane, in both
directions. This is a clean positive result and is quotable.

**Standing: three independent tests now say the same thing.** G14 (entity vs whole-frame,
p=0.683), G17 (detector generation, newer detector 2.2 pp *worse*, p=0.00412) and G18 (fusion,
p=0.401) each fail to find any accuracy contribution from the entity decomposition. That
convergence is itself the finding, and it was reached by three pre-registered tests rather than by
one disappointing run.
