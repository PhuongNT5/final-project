# Examiner-style review — thesis alignment, 2026-09-13 (D-6 to defence)

Scope: PROPOSAL.txt vs brief.md (binding D1-D5, S1-S20) vs report/*.md vs README/docs vs actual repo state.
Method: full read of PROPOSAL.txt, brief.md (1150 lines), plan.md, phase-04, abstract.md, README, PROGRESS.md,
docs/system-architecture.md, docs/project-roadmap.md, grep sweeps for stale numbers/broken refs/language,
spot checks of chapter-03/04/07, gate1 file, kaggle-detector-retrain, deployment artifacts, git log.
Also folds in two parallel code-audit reports run same-day: `review-2026-09-13-edge-deployment.md`
(deployment/*, Ch5, 5 findings C1/C2/H1-H5 below) and `review-2026-09-13-eval-statistics.md` (eval_utils,
unsafenet stats, run_*.py, 2 findings below). A third parallel code review ("P3 core pipeline") was still
running at write time — see Unresolved Questions.

## 1. Objective coverage (PROPOSAL.txt MT1-MT3, Nội dung 1-3)

**MT1 (YOLO+Transformer architecture, CNN speed + Transformer temporal reasoning).** Built, twice
(P1: frozen-COCO YOLO11s → multi-stream Transformer; P3: YOLO26 entity detector → spatial-relation
Transformer). RT-DETR deviation is explicitly and correctly handled: `chapter-03-methodology.md`
§3.2.0 and `chapter-01-introduction.md` L103 both state the proposal named RT-DETR but a plain
Transformer encoder realizes the "temporal reasoning" intent, RT-DETR is an object detector not a
sequence encoder, and it would hit the same COCO-pretrained ceiling anyway. This is a well-argued,
defensible deviation — a good answer if a committee member asks "where is RT-DETR". Good: this is
already correct and does not need more work.

**Nội dung 2 kết quả (3 components: YOLO spatial extractor → Transformer temporal → fusion/classify;
sparse-sampling data loader; test accuracy >95.8%):**
- 3-component structure: present in Ch3 (`## 3.3 Transformer Head`, cascade fusion §3.5.1) for P1;
  present in Ch7/system-architecture for P3 (entity detector → relation tokens → RelationTransformer).
- Data loader: README says "Frame Sampling: T=32 uniform frames"; P3 uses 16 frames @800. Neither is
  described anywhere as "sparse sampling" in the proposal's sense (this is fixed-stride/uniform, not
  adaptive/thưa sampling) — cosmetic gap, rename or add one sentence, do not re-engineer.
- >95.8% accuracy: NOT achieved, and per D3 the thesis explicitly does not attempt to claim it — this
  is a locked, defensible framing (statistical tie + N=64 ceiling proof), not a gap, provided the
  framing is stated consistently everywhere (see §2 below — currently it is not).

**8-class / safe-vs-unsafe classification — the sharpest specific gap.** Proposal Nội dung 2 literally
asks for classification into "4 an toàn – 4 nguy hiểm" (8 classes, safe+unsafe). Every headline number
in this repo (54/64, 56/64, 80.8% k-fold, 77.7% LCO, etc.) is evaluated on the **4-unsafe-class subset
only** (the paper's own 64-clip protocol, which itself only test-labels the unsafe clips). Ch7's
"8-class entity ontology" is a different axis — it is 8 *entity* classes (people/zones/panels/forklift
states) used as Transformer input, not the 8 *behaviour* classes as classifier output. **Nowhere in
report/ is a full 8-class safe-vs-unsafe test accuracy reported.** This is defensible if framed as "we
follow the paper's own comparison protocol, which is 4-class"; it is a real gap if an examiner reads
Nội dung 2 literally. Recommendation: one explicit sentence in Ch4/Ch3 stating the paper's own
evaluation is 4-unsafe-class and this thesis follows it for comparability, with a named limitation that
full 8-class safe/unsafe accuracy was not separately reported. Cheap (~30 min), high committee-risk if
skipped.

**YOLO26 vs YOLO11 coherence.** Ch3/Ch4 (P1, the deployed/benchmarked pipeline) use YOLO11s only.
YOLO26 appears only in Ch7 (10 hits) — the P3/accuracy-headline chapter. This is internally coherent
(different pipelines, different YOLO versions, documented in `docs/system-architecture.md`) but is
**never stated as a deliberate comparison** anywhere — there is no "YOLO11 vs YOLO26" ablation table in
the thesis despite brief.md's own k-fold table showing both tracks agree (80.7% vs 80.8%, S3). That
table exists in brief.md/logs but never made it into Ch4/Ch7 prose. Cheap add, strengthens MT1.

**MT3 (TensorRT INT8, multi-stream, low latency + high accuracy, edge).** Real, physically-measured
Jetson Orin Nano evidence exists and is strong — but **only for P1** (the older/lower-accuracy
pipeline). P3 (the accuracy headline) has no physical Jetson measurement at all; only a Mac-side
projection that *fails* the 3.0 decisions/s requirement (S20, see §2). So MT3 is satisfied for "a"
pipeline but not for "the" pipeline the thesis leads with on accuracy. This is the single most
important unresolved coherence problem — detailed next.

## 2. Central coherence problem — P3 (accuracy) vs P1 (deployed, real-time)

**The facts, cross-checked against S20 and `laneb-p3-edge-cost-and-export.md`:**
- P3 same-dist headline: 54/64 = 84.4% top-1, 88.4% macro (blind, pre-registered). Its k-fold
  (516 clips, 50 folds) is 80.8±4.0%, and its LCO cross-camera margin is +16.9pp macro over a
  transferable scene baseline — the only N>64 evidence in the project and the strongest scientific
  result here.
- P3 measured (M3 Mac, 2026-09-09) at **0.55 decisions/s**; projected onto Jetson via a transfer
  factor calibrated on P1 (the only pipeline benchmarked on both machines): **1.13–1.71 dec/s vs
  3.0 required — fails at every TensorRT precision, ~2x miss.** P3 costs 7.3x P1's FLOPs/decision,
  driven by 800² detection resolution. No physical Jetson run of P3 exists yet; this number is a
  calibrated projection, not a device measurement.
- P1 is real-time and physically measured on Jetson (8.62ms p50/116 FPS single-stream TRT-FP16;
  4-stream 88 FPS aggregate zero-drop; 36.2-min soak 0 drops; INT8 export) — but its accuracy is the
  older, weaker, single-64-clip point estimate (±11pp CI) that the thesis's own audit trail (S9-S20)
  has repeatedly found fragile (withdrawn 57/64, withdrawn +3.5pp, S9's silent-overwrite near-miss).

**Which chapters claim what (as of this read):**
| Chapter | Anchors on | Notes |
|---|---|---|
| abstract.md | P3 (LCO scene-robust framing is the headline claim) | "honest headline" paragraph centers LCO +16.9pp, not P1 edge system |
| README.md | P1 ("deployment-led", 56/64 headline table) | Explicitly says "P3 accuracy-headline re-benchmark pending" — i.e. README's own headline table is stale relative to abstract.md |
| Ch1 Introduction | Mixed — objectives table cites P3 architecture, results table cites P1 hybrid+gate numbers (53/64) | A third number, different from both abstract (54/64) and README (56/64) |
| Ch3 Methodology | P1 architecture in detail; P3 only via the RT-DETR-deviation note | |
| Ch4 (+ kfold section) | P1-era accuracy ladder, withdrawn-number history; explicit pipeline-attribution banner added retroactively | 71KB, likely still carries residual P1-only framing despite the banner |
| Ch5 Edge deployment | P1 Jetson numbers only, with a banner flagging P3 re-benchmark as pending | Consistent with what is actually measured |
| Ch6 Conclusion | Repeats the withdrawn-number history, P1-flavoured | |
| Ch7 (new, extra chapter) | P3 entity-relation + LCO — the chapter that matches the abstract | Only chapter consistent with the abstract's headline |
| docs/system-architecture.md | Explicit P1/P2/P3 map, most trustworthy single source | Best-maintained document in the repo |

**Verdict: the narrative does not currently survive as one document.** Four different top-1 numbers
(53/64 Ch1, 54/64 abstract/Ch7, 56/64 README, plus P1's various ladder entries) are each locally correct
for the artifact they cite, but nothing tells a first-time reader which one is "the" number, and the
abstract's central scientific claim (P3/LCO) is not the same system as README's central engineering
claim (P1/edge). An examiner who reads only the abstract and then only the README will believe two
different theses.

**What framing an examiner would accept (and the brief already argues for it, D3 — just not
executed consistently):** state explicitly, once, in Ch1 and again in Ch6, that this thesis produces
**two pipelines answering two different halves of MT1-MT3**, not one pipeline that does everything:
- P3 = the scientific/accuracy contribution: statistically reliable (k-fold), the only pipeline with
  a genuine cross-camera generalisation result (LCO), and the pipeline that best answers MT2.
- P1 = the engineering/deployment contribution: the only pipeline with physically measured real-time
  edge performance (MT3), at older/weaker accuracy.
- Name the gap explicitly and quantitatively: P3 is 7.3x the FLOPs/decision of P1, measured at
  0.55 dec/s, projected 1.1–1.7 dec/s vs the 3.0 requirement — **P3 does not currently satisfy MT3**,
  and P1 does not carry the P3 accuracy evidence. State the identified fix (imgsz 800→512, S20) as
  future work / in-progress, and if time allows (see action list) actually run it partway.
This is a coherent, examiner-defensible thesis (it is arguably a *better* thesis than claiming one
pipeline does everything) — but it must be the *same* sentence in the abstract, README, Ch1 and Ch6,
which it currently is not.

**Sharper evidence from the parallel Ch5 code review (`review-2026-09-13-edge-deployment.md` C1/C2):**
Ch5 itself sets a hard rule at its own top ("P1's 56/64 is test-selected... must never be presented as
the result; its honest blind counterpart is 53/64") and then **breaks its own rule in 4 places** in the
body (`chapter-05-edge-deployment.md:36, 203-209, 377-386, 700-702` — intro, §5.4.2, §5.6.2 Jetson
table, §5.8 conclusion all restate bare 56/64 with no hedge; only §5.6.7 does it correctly). Worse, the
entire polished Ch5 deployment narrative (TRT export, multi-stream zero-drop, 36-min soak, power Pareto,
live dashboard, threat model) is **P1's story presented with no caveat that P1 is not the pipeline the
abstract/Ch7 lead with** — `deployment/JETSON-DEPLOYMENT.md` discloses the P3-not-yet-benchmarked
caveat honestly, but that file is two hops away from where a reader meets the Ch5 narrative. This is
independent corroboration, from actual line numbers, of the coherence problem identified above — it
is not a hypothetical risk, it is present in committed chapter text today.

The same review's Mac→Jetson projection-soundness check (their §3): the FAIL verdict (§2 above) is
methodologically **sound as a bounded, disclosed projection**, and the margin (0.55 measured vs 3.0
required, ~2x even at the most optimistic projected end) is large enough that the qualitative FAIL
conclusion is probably robust even though the P1-derived transfer factor applied to P3's structurally
different (conv-heavy vs MLP-heavy) workload is unverified in either direction — treat "FAILS at every
precision" as safe to state, treat the exact "1.64 dec/s" figure as illustrative, not measured.

**Additional finding from the parallel stats/eval-code audit (`review-2026-09-13-eval-statistics.md`
#1), directly bearing on claim axis 1 (paper-protocol tie):** the three scripts that actually produce
"vs paper" significance numbers (`run_full_eval.py`, `run_relation_transformer.py`, `ensemble_relation.py`)
print `two_proportion_z`, which assumes two **independent** samples of n=64 — invalid here, since our
result and the paper's are both scored on the *same* 64 clips (paired, not independent). The correct
tool (`mcnemar_discordance_bound`, an enumeration bound since the paper's per-clip predictions are
unpublished) already exists in the codebase and is wired into `eval_utils/suite.py`, but none of the
three number-producing scripts call it. Live re-check: `two_proportion_z` gives p=0.048 for 56/64 vs
paper 62/64 ("borderline significant"); the true McNemar bound is p ≥ 0.03125 with the actual value
unknowable and plausibly much higher. A prior hand-worked audit already reached the correct honest
conclusion (no tie is broken either way) — so the thesis's *stated* conclusion is not currently wrong —
but any number freshly re-run from these 3 scripts between now and the deadline would print an
unguarded, invalid p-value. Cheap fix, see action list.

## 3. Chapter state

**Language — hard non-compliance with D1.** `grep -P` for Vietnamese diacritics across every
`report/*.md` file returns **zero** real hits (one false positive: an inline English gloss
"multi-luồng" in Ch2). **100% of report/ is English.** D1 (brief.md §0, binding) requires the thesis
body in Vietnamese with an English abstract. As of day 8/14 (2026-09-13, 6 days to deadline), **no
Vietnamese translation work has started** despite phase-04 stating Ch1/Ch2 could start "immediately,
ungated." This is the single largest compliance and schedule risk in the project.

**P1-era vs P3 chapters:** see table in §2. Ch1/Ch3/Ch4/Ch4-kfold-section/Ch5/Ch6 are P1-anchored
(with retroactively-added pipeline-attribution banners, per git log `c2538ed "make Ch3/4/6 body text
agree with the pipeline-attribution banner"`); Ch7 alone is P3-anchored and is the only chapter that
matches the abstract's headline claim.

**Stale numbers — better than expected.** Grepped for the exact withdrawn figures named in the task
(57/64 calibrated, +3.5pp, 16.9%, 417/516, 95.7% mAP, marginally significant, ECE 0.104, AUC 0.953,
2025 citation). Every hit in `report/`, `docs/`, `README.md`, `PROGRESS.md` appears **inside an
explicit withdrawal/correction sentence**, not asserted as current truth — the audit discipline (S1-S20)
has actually been propagated into the prose, which is a real strength. The 2025→2024 citation-year
correction (S7) is applied in README and (per grep) Ch2's bibliography entry. One caveat: Ch4 is 71KB
(~2,300 lines) and was only spot-checked; `audit-cross-chapter-number-consistency.md` (the tool built
for this) was last run 2026-09-06, three days before S16-S20 landed — it is now stale and should be
re-run before final freeze, not trusted as current.

**Ch7 / 6-chapter template mismatch.** Confirmed: report/ currently has Ch1-Ch7 plus a separate
`chapter-04-section-kfold-cross-validation.md` and a superseded `thesis-reframe-deployment-led.md`
(marked superseded in git log `ff5a272` but the file is still present, untouched, and could be cited by
mistake). Phase-04's own plan file names this exact ambiguity as unresolved: whether the kfold section
and Ch7 merge into Ch4 or stay separate. **Nothing has been decided or executed.** S01 Mẫu 2 (Vietnamese
thesis template implied by PROPOSAL.txt and brief §6) expects 6 chapters. Current file count, if
assembled as-is, would produce 7-8 chapters/sections — needs an explicit merge decision now.

**`report/thesis-assembled.md`** (phase-04's own day-12 deliverable and success criterion) **does not
exist.** Confirms writing/assembly has not progressed past the pre-existing English draft state.

## 4. README accuracy

**Broken file references — confirmed, all missing.** Checked every file named in README's "Project
Structure" / "Quick Start" sections against disk:
`run_finetune_e2e.py`, `model_arch.py`, `feature_dataset.py`, `temporal_video_loader.py`,
`tools/preextract_yolo_features.py`, `tools/scan_bad_videos.py`, `GAP_ANALYSIS.md`,
`run_safe_unsafe.py`, `run_temporal_fusion.py`, `run_scenario_cascade.py`, `plot_figures_binary.py`,
`BASELINE.md` — **all 12 are missing.** The actual root scripts that matter now
(`run_relation_transformer.py`, `run_kfold_relation.py`, `run_lco_eval.py`, `build_relation_tokens.py`,
`model_relation_head.py`, `run_full_eval.py`) are not mentioned in README at all. The README describes a
codebase layout from an earlier project phase that has been fully superseded.

**Headline table is stale.** README's "Key Results" table still leads with the **P1** number
(56/64 = 87.5%, 89.28% macro) and explicitly flags "P1 pipeline; P3 accuracy-headline re-benchmark
pending" — but P3 *has* been re-benchmarked and frozen (abstract.md, Ch7, system-architecture.md all
cite 54/64 / 88.4% macro as current). README has not been updated since that freeze landed.

**One-command reproduction (D4) — does not exist.** No `scripts/` directory at repo root (confirmed).
`docs/project-roadmap.md` itself states this explicitly as an open phase-05 item, not yet started.
Deliverable D4 ("reproducible repo... one-command repro per headline number") is currently unmet.

## 5. Over-complexity / recommended examiner-readable structure

The project is scientifically strong but is currently organized as an engineering audit trail (brief.md
alone documents S1 through S20, each a found-bug/retraction/correction cycle) rather than as a thesis.
That rigor is a genuine asset — keep it, but move it out of the reading path.

**Recommended Ch1-Ch6 (VN) map:**
- **Ch1 Giới thiệu** — motivation, MT1-3, one paragraph stating the two-pipeline framing (§2 above) as
  the thesis's actual claim structure, contributions.
- **Ch2 Tổng quan** — YOLO family, Transformer/RT-DETR (with the deviation note), industrial safety
  datasets, Unsafe-Net paper (correct 2024 citation).
- **Ch3 Phương pháp** — lead with **P3** (entity ontology → YOLO26 detector → relation-Transformer),
  since it is the scientific/accuracy-headline system the abstract already centers; describe P1 as the
  deployment variant in one comparison table (architecture, params, FLOPs/decision), not a second full
  methodology section.
- **Ch4 Thực nghiệm và Kết quả** — merge current Ch4 + kfold-section + Ch7 into one chapter, organized
  by the brief §2 axes: (1) paper-protocol tie, N=64 ceiling proof, (2) faithful-reproduction result,
  (3) k-fold reliability, (4) LCO cross-camera generalisation (lead result), (5) safety/FNR, (6)
  calibration. Negative results / withdrawn-number history as one short named subsection at the end, not
  scattered through the chapter.
- **Ch5 Triển khai biên** — P1 as "what is deployed and measured today" (the real Jetson numbers); P3
  edge cost as an honest, quantified limitation (7.3x FLOPs, 0.55 dec/s measured, projected fail) plus
  the named future-work route (imgsz 512).
- **Ch6 Kết luận** — restate the two-pipeline honest headline, contributions, limitations, future work.
- **Appendices (not in the 6-chapter body):** full S1-S20 audit trail, the P2 paper-faithful-rewrite
  attempt (currently gets a full narrative thread in PROGRESS.md/README — compress to one paragraph:
  "attempted, underperformed at 50/64, abandoned"), detailed seed/cluster-sensitivity tables (S16/S19),
  dedupe ablation (S17/S18), test-touch log.

**5-7 headline numbers with exact protocol labels** (for slides + Ch4 lead table):
1. Same-dist paper-protocol, P3, blind, pre-registered 2026-09-06: **54/64 = 84.4% top-1, 88.4% macro**
   — statistical tie vs paper no-bbox baseline (85.78% macro), not vs full pipeline (95.81%).
2. k-fold reliability, P3/YOLO26, 516 clips, 50 folds, Nadeau-Bengio corrected: **80.8±4.0% top-1,
   84.2% macro**, CI [76.7, 85.0]; vs scene-only baseline **p = 0.0600 (n.s.)** — quote as n.s., never
   "marginally significant."
3. Leave-camera-out cross-camera generalisation, 516 clips, 5-seed canonical, KMeans(8) camera proxy:
   entity-relation **81.2% macro (77.7% top-1)** vs transferable scene-appearance baseline **64.4%
   macro** = **+16.9pp margin, CI [+12.5, +21.2]** (range +13.3 to +16.9 across re-clustering — quote
   as a range, not a point).
4. Calibration/safety, blind-basis predictor (54/64): ROC-AUC **0.951**, PR-AUC **0.880**, ECE
   **0.097**, κ = 0.775, MCC = 0.785.
5. N=64 ceiling proof: paper errs on only 2/64 clips → max possible McNemar significance at this N is
   **p = 0.50**; required-n for a real non-inferiority claim ≈ 111 clips (Δ=10pp) to 445 (Δ=5pp).
6. Edge deployment, P1, physical Jetson Orin Nano, TensorRT-FP16, measured 2026-06-03: **8.62ms
   p50 / 116 FPS** single-stream; 4-stream **88 FPS aggregate, zero-drop**; 36.2-min soak, 0 drops,
   7.0W; INT8 156→40MB.
7. P3 edge-cost honest negative result, measured M3 Mac 2026-09-09, projected via P1-calibrated
   transfer factor: **0.55 dec/s measured; 1.1-1.7 dec/s projected on Jetson vs 3.0 required — fails
   at every TensorRT precision**, ~7.3x P1's FLOPs/decision.

## 6. Prioritised action list, 6 days remaining (Sep 13→19)

**MUST (blocks a defensible defence):**
1. Lock the two-pipeline reconciliation narrative (§2) in one paragraph; paste it identically into
   abstract, README, Ch1, Ch6. ~2-3h, no compute.
2. Vietnamese translation, D1-binding. Given zero progress and 6 days left, **do not literally
   translate the existing ~250KB of EN prose** — write condensed new VN chapters per the §5 structure
   (normal VN thesis chapter length, not the current 40-70KB-per-chapter sprawl), citing the detailed
   English material as appendices. Largest remaining item; realistically 3-4 days if condensed, likely
   infeasible in 6 days if attempted as full literal translation of current content.
3. Decide and execute the chapter-count merge (Ch4 + kfold-section + Ch7 → one Ch4; retire/relabel
   `thesis-reframe-deployment-led.md`); produce `report/thesis-assembled.md`. ~0.5 day, concurrent with #2.
4. Rewrite README: correct headline table to the current P3 number (or an explicit dual P1/P3 table
   matching #1's narrative), remove/replace all 12 broken file references, add manual repro commands
   per headline number even without a `scripts/` wrapper layer. ~3-4h.
5. Re-run the cross-chapter number-consistency audit against whatever Ch4/Ch5/abstract/README end up
   as (last run 2026-09-06, now stale relative to S16-S20). ~2-3h.
6. Fill in the blank Gate-1 outcome table (`gate1-detector-retrain-prediction.md`) from the existing
   `runs/detect/sweep_results/r2/beh_{tight,ctx}/results.csv` — data already on disk, just needs
   reading and a pass/fail sentence in Ch4. ~1h.
7. Add one sentence to Ch4/Ch7 naming that CarryOverload's 100% recall is produced by a single-rule
   detector artifact (single Forklift_Overload token → CarryOverload, precision/recall 100% with no
   Transformer needed — `labeling-strategy-review.md` §2) and is not evidence of relational reasoning.
   Near-zero cost, high committee-attack-risk if omitted.
8. Add the literal-8-class caveat sentence from §1 (thesis follows the paper's 4-unsafe-class protocol;
   full 8-class safe/unsafe accuracy not separately headlined). ~30 min.
9. Fix Ch5 body to stop contradicting its own attribution rule (C1: 4 sites, `:36,203-209,377-386,700-702`
   — append "(test-selected; blind 53/64)" or replace bare 56/64 with the pair everywhere); expand C2's
   caveat into a real paragraph/subsection stating P1-deployed vs P3-headline explicitly, using the
   edge-cost table already built in Lane B. ~0.5-1 day, writing only, no re-measurement — this is option
   D from the edge-deployment review, recommended unconditionally regardless of what else is attempted.
10. Swap `run_full_eval.py`/`run_relation_transformer.py`/`ensemble_relation.py`'s headline "vs paper"
    stat from `two_proportion_z` to `mcnemar_discordance_bound` (already implemented, already wired into
    `eval_utils/suite.py`, just not called from these 3 scripts); keep `two_proportion_z` only as a
    labeled "assumes independence, does not hold here" diagnostic. Prevents any number re-run before the
    deadline from printing an invalid p-value. ~1-2h.

**SHOULD (materially strengthens the defence, not launch-blocking):**
11. Get at least one physical-Jetson measurement of P3 (even partial: detector-only TRT-FP16 latency
    at 800px) to replace the current pure-projection FAIL claim with one measured anchor point. D2 says
    full physical access exists. Requires first fixing `TRTRunner`'s silent multi-input corruption bug
    (edge-deployment review H3: 2-input models silently drop the second input's data instead of
    erroring) and verifying `trt-compat-report.md` covers the P3 graphs, not just P1's. ~1.5-2 days.
12. Attempt the imgsz 800→512 P3 cost-reduction route (S20's own cheapest identified fix): one token
    rebuild + one retrain + re-benchmark. Even a partial/negative result gives Ch5/Ch6 a concrete
    in-progress data point instead of speculation. ~1-2 days (accuracy at 512² is unmeasured; risk of
    needing multiple iterations — timebox it).
13. Draw the two missing zone polygons (view 2, view 7) per `labeling-strategy-review.md` — cheapest
    fix in that report, removes a "geometry ⟺ label" leakage channel currently uncaught in the thesis
    text. ~1-2h.
14. Recompute the Ch5 "GPU saturation point" 136.89 FPS figure (`chapter-05-edge-deployment.md:437-439`)
    from processed frames, not raw-read frames — it currently counts 328/960 dropped frames as
    throughput (edge-deployment review H4). Also caveat the detector-only ONNX bench as excluding
    NMS/crop/decode stages (H1) and using mean not p95 crop count (H2) — both make the already-failing
    P3 dec/s figures optimistic, not pessimistic, which strengthens rather than undercuts the FAIL
    conclusion, but should be stated. ~1-2h combined.
15. Add the Nadeau-Bengio anti-conservativeness caveat (eval-statistics review #2) wherever a k-fold
    p-value is quoted in Ch4: "corrected resampled t-test is known to be anti-conservative under
    repeated-CV pooling (Bouckaert & Frank 2004); treat p near 0.05 as indicative." No code change.
    ~15 min.
16. Repo hygiene before final submission: `unsafe-net-detector-dataset.zip` (1.3GB) and `best.pt`
    (19MB) sit at repo root — confirm `.gitignore` (recently modified per git status) actually excludes
    them; a 1.3GB blob in the "reproducible repo" deliverable is a bad first impression.
17. Start slides/Q&A brief (phase-06) once #1/#3 lock the numbers — already behind the original
    schedule (was meant to start day 9-10).

**COULD (cut first under time pressure — brief.md itself says so for most of these):**
18. Full streaming/event-level operational metrics (§4i of the eval contract) — brief already names
    this the first thing to cut if Lane A/B is behind.
19. A third independent LCO re-clustering (cs=3+) to firm up the clustering-uncertainty term — S19
    explicitly says "not blocking."
20. Person_Authorized/Unauthorized class-imbalance rebalancing (labeling-strategy-review.md §4) —
    real finding, but fixing it (relabel/reweight) is a data-pipeline change this late in the
    schedule; better to name it as a limitation than to touch training data with 6 days left.
21. Token-reuse/sliding-window amortisation for P3 (edge-deployment review option C: `EMIT_EVERY=8`
    already means consecutive decisions overlap 8/16 frames, so detector cost could partially amortise)
    — intellectually the most interesting cost-reduction route but needs new caching logic plus a
    recall sanity-check on the two seed/cluster-fragile classes (WalkwayViol, UnauthorInt); treat as
    future work, not a 6-day item.
22. Misc code hygiene noted by the edge-deployment review, non-blocking: label the two different INT8
    paths (ORT dynamic vs TensorRT static calibrated) explicitly wherever both are cited near each
    other (M2); add a `parse-tegrastats.py` so `power-sweep-summary.json`/`soak-jetson-summary.json`
    are regenerable from the committed raw logs rather than seemingly hand-computed (M3); scope
    `deployment/dashboard/app.py` (no auth, renders raw worker video) as "demo only, localhost, not
    covered by the Ch5 threat model" in one sentence if it's ever shown over a shared network (M5).

## Unresolved questions
1. Is Gate 1 (detector mAP50 ≥ 0.90, all 8 classes) actually pass/fail? `gate1-...prediction.md`'s
   outcome table is blank; `runs/detect/sweep_results/r2/beh_ctx/results.csv` has numbers that look
   consistent with a pass on the `ctx` arm but the column header wasn't verified against the training
   script's CSV schema in this pass — verify column mapping before writing a Gate-1 sentence into Ch4.
2. Required assembly format for S01 Mẫu 2 (Markdown→PDF, university template, .docx?) is still
   unconfirmed with the advisor per phase-04's own open item — blocks final formatting, not content.
3. Whether the owner wants a genuinely single unified headline pipeline in future work, or is content
   to defend the two-pipeline framing as the permanent thesis position — affects how strongly Ch6's
   future-work section should commit to "P3 will be made real-time" vs "P3 and P1 serve different
   purposes by design."
4. `report/thesis-reframe-deployment-led.md` is marked superseded in a commit message but the file is
   still present and un-flagged in its own text — confirm it should be deleted/archived, not merely
   ignored, before assembly.
5. A third parallel code review ("Review P3 core pipeline code") was still running at the time this
   report was written and is not yet incorporated — check its findings separately once it lands; it may
   surface additional core-pipeline (detector/token/relation-head) issues beyond what the edge-deployment
   and eval-statistics reviews covered.
6. Does `trt-compat-report.md` cover the P3 ONNX graphs or only P1's (edge-deployment review, gates
   action item #11/MUST-adjacent SHOULD #11 above)?
7. Where do `power-sweep-summary.json`/`soak-jetson-summary.json` actually come from if no committed
   parser produces them from the raw tegrastats logs (edge-deployment review M3) — hand-computed
   numbers presented as pipeline output would conflict with the project's own no-manual-results rule.
