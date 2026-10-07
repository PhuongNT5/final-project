---
title: "Thesis Recovery — Confounding Finding, Walkway Lever, Edge Evidence, Defence Package"
description: "Three-week recovery: bank the detector and camera-confounding findings, attack the one surviving accuracy lever, freeze E3, ship measured edge + demo."
status: pending
priority: P1
effort: 80h
branch: next-round
tags: [thesis, binary-classification, unsafe-net, relation-tokens, lco, confounding, jetson, defence]
created: 2026-09-19
---
# Thesis Recovery — Master Plan
Supersedes `plans/2026-09-16-binary-safe-unsafe-reframe/`; its landed work is **input, not redo**, its phase-06 decision-rule style inherited verbatim. **User's objective order (binding): 1. ACCURACY · 2. EDGE · 3. ROBUSTNESS.** Framing keeps that order — note only that the accuracy chapter's *content* is now largely a cross-camera result. **Number freeze 2026-10-05, defence package 2026-10-10. 2 test touches pre-registered, 0 spent.**

## DECISIONS UPDATED 2026-09-29 (author instruction; SUPERSEDES D1 and amends D4 below)

Author instruction, verbatim: *"khong nen phe phan, tim huong khac lien quan den Unsafe behavior
detection voi YOLOv11/26 va TRansformer"* - do not build the thesis on criticism; find a different
direction, on unsafe-behaviour detection with YOLOv11/26 and Transformers.

- **D1 SUPERSEDED - the headline is an architecture contribution, not a protocol critique.** The
  spine becomes **ESTA** (YOLOv11s entities -> frozen CLIP ViT-B/32 -> factorised space-time
  attention), evaluated on leave-source-out folds. **The protocol-inflation result is not
  discarded and not wasted**: it moves from headline to the methodology chapter, where it does
  the job it is actually best at - justifying *why* LSO-1100 is the evaluation protocol and why a
  random-fold number on this corpus means nothing. A reviewer who asks "why are your numbers lower
  than the literature's" is answered by that chapter; the thesis no longer *leads* with it.
- **D4 AMENDED - the cut on G14 and architecture search is lifted for the ESTA lane only.** Same
  instruction. Cost, stated plainly: this spends the schedule slack that the freeze was protecting,
  and it is the author's call, made with the deadline known. Everything else D4 cut (third corpus,
  L5, L6) stays cut. The mandatory Jetson session stays mandatory.
- **D3 CLARIFIED, not overturned.** E3 remains the frozen *deployment* artifact and every edge
  number on the sheet is E3's. ESTA is a candidate for the accuracy chapter; promoting it into the
  edge chapter requires re-measuring on device, which happens only if G14 passes AND the schedule
  allows. If it passes and there is no time, the thesis says exactly that.
- **What is still forbidden.** The gates in `risks.md` (G14/G15/G16/G17) were written before the
  code existed and are not renegotiable after seeing results. If G14 fails, the deliverable is a
  second, better-controlled negative result plus the methodology chapter - not a fifth attempt.

## DECISIONS LOCKED 2026-09-20 (author sign-off received; binding, supersede conflicting text below)

- **D1 — Headline = protocol inflation, not an accuracy number.** Thesis spine: cross-camera /
  leave-source-out evaluation costs **12.9-21.8 pp** balanced accuracy vs random folds on
  Unsafe-Net (E3/18,044 82.5->69.6, E1d64/94,380 84.1->62.3, E1d96/197,292 84.5->63.9, E2/284,556 84.6->66.7,
  E0/298,892 83.6->66.7 — five encoders, none escapes; identical tokens,
  identical arch, identical params, only the fold construction differs) and **+15.4 pp** on
  iSafetyBench (E3 90.6 random vs 75.2 LSO, G12). Replicated across two verified-disjoint corpora,
  two detectors, two label vocabularies, four capacities. LCO has MORE training data per fold
  (8 held-out groups vs 5-fold), so the drop is not a data-volume artifact.
- **D2 — Deployed system is P1; stop forcing P3 to real-time.** P1 carries the full measured edge
  package; P3's measured cost curve is itself the result (it quantifies the compute price of
  per-entity granularity). Chapter attribution of the two pipelines must be disclosed explicitly.
- **D3 — E3 stays the frozen headline encoder; E0 is NOT promoted.** G11 named E3; the
  pre-registration forbids retry with a different encoder. Report both facts, failure first.
- **D4 — Experiment freeze from 2026-09-21, WITH ONE AMENDMENT: on-device Jetson results are
  MANDATORY, not optional** (author decision 2026-09-20). Rationale in the author's favour: the
  "P3 misses the 3.0 dec/s cadence" conclusion is currently a Mac-derived projection; measuring it
  on device converts it into a first-class measurement and strengthens the thesis. Allowed
  experiments and nothing else: (a) the in-flight G12 E0/E1 runs; (b) the T1 test touch on the
  125-clip blind split (the protocol the PROPOSAL itself designates); (c) the mandatory Jetson
  session per `deployment/JETSON-RUNBOOK-2026-09-19.md`, E3 only per D3; (d) L3 calibration, only
  if writing finishes early. CUT: G14, any third corpus, L5, L6, all architecture search.

**Deadline and language fixed 2026-09-20 by the author:** defence **mid-October**; number freeze
**2026-10-05** and package **2026-10-10** stand. The **entire report is in Vietnamese**, written in
**scientific research-report register**. English abstract per D1 of the 2026-09-06 plan. Consequence:
new chapters are drafted DIRECTLY in Vietnamese, never written in English and translated.

Full review and schedule: `reports/strategy-2026-09-20-duyet-lai-du-an-va-ke-hoach-nop-hoi-dong.md`.

**Day-0 findings — three premises died, a fourth finding replaced them.** Evidence: `reports/finding-2026-09-19-detector-starvation-root-cause.md`, `reports/finding-2026-09-19-lco-failure-is-camera-label-confounding.md`, `reports/finding-2026-09-19-what-each-subtask-actually-rests-on.md`, `reports/critique-2026-09-19-research-factcheck.md`.
1. **Detector is not undertrained.** `yolo26_best.pt` (built the headline tokens) = mAP50 0.966 / R 0.945; `best.pt` 0.975/0.956; both 100-epoch GPU runs; the 3-epoch CPU artifact is unused. **Retrain cut, not deferred.** Imbalance 11.8:1 did not kill minority detection (P_Auth R 0.899).
2. **Architecture search is dead.** Six encoders over a 16× parameter range (18k→300k) fail on exactly the same two sub-classes and succeed on the same four, same difficulty order. The 12+ track ~54-56/64 ceiling = every track varied the one factor that does not matter. **All encoder/width/loss search is cut. E3 DeepSets (18,044 params) is frozen** as headline and deployment model — best LCO *and* best edge story.
3. **The LCO collapse is camera-label confounding, not model failure.** All 32 `panel_closed` clips sit in one camera group → held-out fold has zero training examples → 0.0% recall is arithmetic, not a defect. Sub-tasks: zone (145) **generalises both sides 76-90%**; carry (86) 100%/100% but a **trivial detector lookup — must be disclosed (R13)**; walkway (282) the **real open failure**, 90% unsafe vs 15-36% safe, spread over 3-4 groups so it is attackable; panel (174) **unanswerable under LCO**. **Carry is solved with geometry identically zero** (the 86 zero-geometry clips are exactly labels 3/7, 86/86 correct for all six encoders) — direct evidence it is a detector class lookup, not relational reasoning. Put that sentence in the thesis.

4. **What each sub-task rests on — the ordering IS the thesis result.** carry (3v7): **detector class alone** (99.4% FkOver vs 100% FkSafe) → 100/100. panel (2v6): **detector class alone** (panel_closed has zero PanelOpen tokens) → unanswerable. zone (1v5): **class + geometry** (P_Auth 2.8→23.1%, `iou_restricted` 0.027→0.119) → **76-90% both sides**. walkway (0v4): **geometry alone** (detector composition near-identical, P_Unauth 75.5 vs 71.2; only `foot_in_walkway` 0.148 vs 0.229) → **17-36%**. Exactly monotone: sub-tasks separable by detector class are saturated, the one separable by class+geometry generalises, the one needing geometry alone fails. **The relational component is the weakest link and most apparent success is detector lookup.**

**Headline strategy.** The thesis result is the mechanism-level claim: *cross-camera failure on Unsafe-Net is a dataset camera-label confounding property, localised to identifiable sub-tasks, invariant to encoder architecture over a 16× parameter range — and sub-task success tracks exactly how much detector-class information separates the pair, so the relational component is the weakest link* — plus the zone sub-task as the positive result, plus a measured Jetson deployment. A tuned accuracy number is a bonus, not the spine. Honest prior on a large jump: **low**.

**Two measured constraints on wording.** (i) Fold ids and view ids agree only **81.7%** — fold 1 mixes views 0/3/6 at 50% purity, folds 6 and 7 are compositionally identical. All LCO/confounding claims are **fold-based** (the grouping LCO actually used) and stand unchanged, but **no sentence may mix the two groupings**, and the `run_kfold_relation.py:106` `[LIMITATION]` banner (KMeans(8) background proxy, not verified camera metadata) rides with every LCO number. (ii) `N_MAX = 8`: headline `tokens_bin8_*.pt` are `(566|125, 16, 8, 74)` = 16 frames × **8** slots × 74 feats. The "≤16 entity slots" figure is **wrong** and must not appear in any capacity or complexity statement.

## Surviving accuracy levers by EV/hour (everything else cut)
| # | Lever | Ph | Cost | Pre-registered falsification |
|---|---|---|---|---|
| L1 | **Walkway geometry repowering** — the measured defect: `foot_in_walkway` averages only 0.229 on `walkway_ok` (should approach 1.0; ~23% of person-frames register), `iou_walkway` is useless by construction (0.014-0.020), `norm_dist` has no signal (0.223 vs 0.228). Cheapest-first: per-view polygon placement audit → signed foot-to-boundary distance replacing IoU → foot-point estimation fix → temporal aggregation of foot-in over T=16 | 03 | 6h | **Must move panel-disclosed LCO BA past 76.6% (E3). Else dead and written up as a negative result.** |
| L3 | Prior correction + temperature calibration + threshold at 0.95 unsafe recall | 05 | 2h | Dead if safe recall at fixed unsafe recall is flat on OOF |
| L4 | Detector A/B token rebuild: YOLO11s (.975) vs YOLO26 (.966) — the choice was never justified | 03 | 1.5h | Attribution only; report either way, never re-run |
| L5 | Appearance probe + upgrade (fixed 512→64 Gaussian; 1.3× context crop) **DROPPABLE** | 04 | 6h | Dead if appearance-free ≥ full under LCO in the P04 probe |
| L6 | Background mixing — attacks the "walkway scene ⇒ Unsafe" shortcut directly **DROPPABLE** | 05 | 8h | Dead if <+1.5pp panel-disclosed LCO with CI excluding 0 |
| — | **CUT:** encoder/width/loss search · cross-arch ensembling · VLM feature fusion · detector retrain · class-id representation (a learned `nn.Embedding`, `model_binary_encoders.py:64,:119`) · **zone-entity token injection** (a polygon is a per-view **constant**, so injecting it as a token carries camera identity — the exact shortcut LCO deletes; expected neutral-to-harmful under LCO while k-fold rises, i.e. success on the wrong protocol, see R39) · **zone-polygon repair for `view_2`/`view_7`** — the 86 zero-geometry clips are exactly the carry sub-task and are already 86/86 correct in every encoder: zero headroom, and no hand-drawing is asked of the user | | | Architecture-invariance (finding §6) makes these unfalsifiable-by-construction here |

## Lanes — exclusive write ownership (concurrent agents; reads free)
- **A** tokens: `build_relation_tokens.py`, `annotation_guide/**`, `kaggle-detector-retrain/**`, `sweep_results/entity_detector/**`, `sweep_results/relation_tokens/tokens_*.pt`, `sweep_results/relation_tokens/build_*.log`, `tools/detector-*.py`, `tools/tokens-*.py`, `tools/zones-*.py`
- **B** eval: `model_*.py`, `run_*.py`, `binary_gate_lib.py`, `eval_utils/**`, `unsafenet/**`, `tools/eval-*.py`, `sweep_results/relation_tokens/**` *except A's two patterns above*, `reports/test-touch-log.md`
- **C** edge: `deployment/**`, `tools/jetson_benchmark.py` — **D** writing: `report/**`, `docs/**`, `README.md`, `PROGRESS.md`, `plans/2026-09-19-*/number-sheet.md` — **E** review: `plans/2026-09-19-*/reports/review-*.md` only, read-only elsewhere
- A→B hands tokens by filename tag; B→D via number sheet; C→D via measurement JSON. Cross-lane write → stop and escalate.

## Phases
| # | Phase | Lane | Effort | Gate |
|---|---|---|---|---|
| 01 | [Detector evidence package](phase-01-detector-evidence-package.md) | A | 2h | **G1** mAP table + caveats banked |
| 02 | [LCO confound decomposition + reporting variant](phase-02-lco-confound-decomposition-and-reporting.md) | B | 5h | **G2** sub-task tables frozen |
| 03 | ~~[Walkway geometry repowering](phase-03-walkway-geometry-repowering.md)~~ **KILLED 2026-09-19** | A | 0h | **G3 RESOLVED — falsified.** See `reports/finding-2026-09-19-walkway-geometry-is-unsalvageable.md`. Three rescue hypotheses (view-map, scale, translation) all swept and falsified; the two classes' foot-point centroids coincide, so no polygon separates them. Write up as the pre-registered negative result. |
| 04 | [Appearance probe (droppable)](phase-04-appearance-probe-and-upgrade.md) | A/B | 6h | needs G2 |
| 05 | [Frozen-encoder recipe: calibration + thresholds](phase-05-frozen-encoder-recipe-and-calibration.md) | B | 6h | **G4** operating point fixed |
| 06 | [Frozen metric suite + statistics](phase-06-frozen-metric-suite-and-statistics.md) | B | 8h | **G5** freeze 10-05 |
| 07 | [Jetson measurement runbooks](phase-07-jetson-measurement-runbooks.md) | C | 10h | **G6** metrics captured |
| 08 | [Live demo + mandatory fallback](phase-08-live-demo-and-fallback-recording.md) | C | 8h | **G7** fallback .mp4 exists |
| 09 | [Number sheet + thesis integration](phase-09-number-sheet-and-thesis-integration.md) | D | 20h | **G8** every number traces |
| 10 | [Adversarial review + citation audit + Q&A](phase-10-adversarial-review-and-defence-qa.md) | E | 6h | **G9** every cite checked |
| 11 | [Contingency + negative-result narrative](phase-11-contingency-and-negative-result-narrative.md) | D/E | 3h | wording pre-committed |
| 12 | [**iSafetyBench accuracy lane**](phase-12-isafetybench-accuracy-lane.md) — external dataset, replaces the dead phase-03 lever | A/B | 10h | **G10** must beat the R42 count-rule floor (63.3% BA leave-source-out) on identical folds, paired CI excluding 0 |

**Critical path** `02 → ~~03~~ → **12** → 06 → 09` (03 killed 2026-09-19; the accuracy lever moved to phase-12's external-dataset lane, see risks R41-R43). 02 gates 03 and 04; 01 blocks nothing; 05 needs only frozen E3 OOF; C and D run from day 1 and never block A/B.

## Calendar (DROPPABLE items marked inside phases)
- **09-20→21** G1 + G2 banked: detector package, sub-task tables, panel-disclosed variant, carry-lookup disclosure. C starts demo, D starts number sheet + the confounding chapter section.
- **09-22→25** L1 geometry repowering: polygon-placement audit → signed-distance + foot-point + temporal-aggregation rebuild; L4 A/B rebuild; **L3 pulled forward** (cheap and near-certain). L2 is cut, so this window is short — **do not backfill it with invented work**. **Jetson #1.**
- **09-26→29** L1 verdict against the 76.6% bar. L3 on frozen E3. L5/L6 only if L1 landed early.
- **09-30→10-03** Phase 06 frozen suite: per-sub-class, per-group/worst-group, both protocol variants, calibration. **Jetson #2** confirmatory.
- **10-04→05** Decision rules applied; contingency call (phase 11); test touches **if earned**. **NUMBER FREEZE 10-05.**
- **10-06→10** Writing, trace audit, citation audit, **demo rehearsal ×2. No new experiments.**

**Compute.** Kaggle ≤30h/wk now almost entirely unspent — retrain cut and architecture search cut. Reserve for L6 and any phase-06 forced re-run; expect to underspend and say so rather than invent work. **One heavy job at a time** (6× wall-clock spread traced to contention). Anchors: LCO-691 5-arch 95m CPU; **E3 5-fold LCO 51s**; token rebuild ≈28m/tag on M4-MPS. **Ultralytics detection training never on MPS** (`train_run.log` stage-2 val mAP identically 0); pure-PyTorch head training on MPS is fine.

**Test touches.** Pooled k-fold-691/LCO-691 = **0 touches** and is the primary evidence; every recomputation from existing OOF dumps is also 0. **T1** = frozen E3 on the 125-clip blind split; **T2** reserve, spendable only if a post-freeze defect invalidates T1. Both pre-registered in `reports/test-touch-log.md` before running, with explicit user go-ahead. Thresholds, temperature, protocol variants: **val/OOF only**. See [risks.md](risks.md).

**Research-report status.** All three research reports are **UNVERIFIED** pending per-citation checks (phase 10). Known bad: a fabricated "Dey et al. 2024"; Nadeau-Bengio cited to an unrelated AUC-fairness paper; focal loss misattributed; SOAR on a dead domain; ~6 arXiv IDs misdated by 1-2 years. researcher-02 §6 Q6 is logically inverted and §7.1 presents unmeasured binary numbers as measured — **neither is carried into this plan**.

## Unresolved questions
1. ~~Does walkway geometry repowering move panel-disclosed LCO BA past 76.6%?~~ **ANSWERED 2026-09-19: NO, and it cannot.** Not a training result — the feature carries no signal to repower. `walkway_violation` and `walkway_ok` foot-point centroids coincide per view (view 3 has them inverted), best achievable foot-inside gap +8.1pp. The failure is neither polygon placement (3 hypotheses falsified) nor model capacity. This closes the last accuracy lever on Unsafe-Net and is why the external-dataset option was opened.
2. ~~How much of the zone positive result survives with the detector-class channel blinded?~~ **ANSWERED 2026-09-19: it survives.** Auth-blinded E3 LCO 68.9% BA / 75.2% panel-disclosed vs 69.6%/76.6% baseline; `zone_ok` recall -5.3pp on n=38 (2 clips, descriptive not significant). See `reports/finding-2026-09-19-auth-blinding-ablation-and-oof-filename-defect.md`. Stronger "relational" wording is permitted for the zone sub-task.
3. Is a dataset-confounding finding plus a measured edge deployment accepted as a Master's contribution without a headline accuracy gain? Phase 11 pre-commits the wording; committee acceptance is not a technical question this plan settles.

*Resolved 2026-09-19 (see the three finding reports): classes 2/3 were **never annotated** — 0 instances across all 6,933 labels, a genuine data gap, not lost work; polygons are rich (7 view files, 1-3 walkway + 1-5 restricted each, full vertices at 1920×1080) so geometry work is instantiable. The 86 zero-geometry clips are the carry sub-task at 86/86 → L2 cut. `N_MAX = 8`, not 16. Fold/view agreement 81.7% → fold-based wording enforced.*
4. **Headline vs transfer framing for iSafetyBench.** Proceeding on the feasibility report's §6 reassignment (accuracy -> iSafetyBench, edge -> Unsafe-Net, Unsafe-Net -> failure-analysis chapter) because it follows the binding accuracy-first order. Reversible until the number sheet freezes (10-05). The two test touches stay pointed at the Unsafe-Net blind split, 0 spent.
