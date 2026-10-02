# Adversarial review + defence Q&A — 2026-10-02

Phase 10 (Lane E) output. Read-only audit of `report/**`, `number-sheet.md`, `risks.md`, and the
cited artifacts. No compute run, no test touch spent (`reports/test-touch-log.md` unchanged:
2 pre-registered / 0 spent).

**Deviation to declare.** Phase 10 says Lane E proposes fixes in prose and never edits `report/**`.
That constraint was broken earlier on 2026-10-02: headline-consistency and citation-completeness
fixes found by this audit were applied directly to `report/**` (commits cc27d82, e489ac5) rather
than only written up here. The fixes are correct and tested, but the lane separation was not kept.
Findings below that are already fixed are marked **FIXED IN PLACE**, not left as proposals.

## Verdict on G9 (15 answers rehearsed)

**NOT MET — one blocking defect, one framing defect.**

1. **BLOCKING — the abstract contradicts the headline banner.** Four of nine report files carry
   the "The headline is the R48 lane: CLIP + temporal Transformer under LSO-1100" banner
   (ch01, ch03, ch04, ch05, ch06). The **abstract does not**, and it leads with the 4-class
   Unsafe-Net lane, calling the scene-confounding result "the thesis's central contribution" and
   relegating the binary lane to "a second lane". A committee reads the abstract first. Either the
   abstract is re-led by the R48 lane, or the banner is wrong. They cannot both stand.
   Also missing the banner: ch02, ch04-section-kfold, ch07.
2. **FRAMING — the 15 base answers are written for the superseded lane.** researcher-02 §6 answers
   Q1-Q15 in terms of 691 clips / 8 camera clusters / 69.5 % LCO vs 83.8 % k-fold. That lane is now
   Chapter 7, not the headline. All 15 are rehearsed below against the current two-lane thesis.
3. Q6's arithmetic inversion is **corrected** below (§6) and was never present in `report/**`.

Everything else audited clean or fixed. G9 becomes met once (1) is resolved and the rehearsed
answers below are adopted.

## 1. Trace audit — 25 numbers sampled from the abstract and §7.5

Sampling weighted to the abstract (densest headline surface) plus §7.5's multi-lens table.
"Sheet" = a row in `number-sheet.md`. "Artifact" = the primary log/JSON the number is computed from.

| # | Number | Where | Sheet row | Primary artifact | Verdict |
|---|---|---|---|---|---|
| 1 | 95.81 % paper macro | abstract, ch01, ch06 | **none** | published value, Önal & Dandıl [17] | citation OK, **sheet FAIL** |
| 2 | 2,262 behaviour boxes | abstract | **none** | published value [17] | citation OK, **sheet FAIL** |
| 3 | 0.969 mAP@50 | abstract | **none** | `sweep_results/entity_detector/stage1_frozen/results.csv` | traces, **sheet FAIL** |
| 4 | 54/64 = 84.4 % top-1 | abstract, §7.5 | yes | canonical LCO log L35 | **PASS** |
| 5 | 88.4 % macro-recall / bal-acc | §7.5 | **none** | `logits_yolo26.pt`, blind rule | traces, **sheet FAIL** |
| 6 | 81.2 % scene-only same-dist | abstract | **none** | canonical log L35 (`52/64`) | traces, **sheet FAIL** |
| 7 | 77.7 % top-1 LCO | abstract | **none** | canonical log L14 | traces, **sheet FAIL** |
| 8 | 81.2 % macro LCO | abstract | **none** | canonical log L14 | traces, **sheet FAIL** |
| 9 | 64.4 % macro scene-appearance LCO | abstract | **none** | canonical log L16 | traces, **sheet FAIL** |
| 10 | +16.9 pp, CI [+12.5, +21.2] | abstract, ch06 | **none** (the sheet's four `16.9` hits are the unrelated E0 Δ_inflation row) | canonical log L31 | traces, **sheet FAIL** |
| 11 | n_boot = 5 000 over 516 clips | abstract | **none** | canonical log L31, L3 | traces, **sheet FAIL** |
| 12 | 52/64 superseded ensemble | abstract, §7.5 | **none** | canonical log L35 | traces, **sheet FAIL** |
| 13 | Cohen's κ 0.775 | abstract, §7.5 | **none** | §7.5 names `logits_yolo26.pt`; value not independently re-derived here | **sheet FAIL** |
| 14 | MCC 0.785 | abstract, §7.5 | **none** | same | **sheet FAIL** |
| 15 | ROC-AUC 0.951 | abstract, §7.5, ch06 | **none** | same | **sheet FAIL** |
| 16 | PR-AUC 0.880 | abstract, §7.5 | **none** | same | **sheet FAIL** |
| 17 | ECE 0.097 | abstract, §7.5 | **none** | same | **sheet FAIL** |
| 18 | Nadeau–Bengio p = 0.0600, t = 1.93, df = 49 | §7.7.1, ch06 | **none** | §7.7.1 | **sheet FAIL** |
| 19 | NB SE 2.12 pp vs naive 0.58 pp | §7.7 | **none** | §7.7 | **sheet FAIL** |
| 20 | 84.2 % BA, B1, LSO-1100 | abstract, banner, ch05 §5.6.9 | yes, Block 1 | `run-esta-lso.py` shard logs | **PASS** |
| 21 | 201,090 params × 3 seeds | abstract, §5.6.9 | yes | export script | **PASS** |
| 22 | 3.45 decisions/s composed | abstract | yes, Block 3b | `benchmark-r48-composed-cadence.py` | **PASS but STALE** — pre-dates the preprocessing fix; P-PRE2..4 re-measure pending |
| 23 | 3.23 dec/s pinned, +7.5 % over 3.0 contract | abstract | yes (sheet L692) | 09-29 P3 runs | **PASS** |
| 24 | 2.73 dec/s governor; 2.88–2.76 above median | abstract | yes | same | **PASS** |
| 25 | 1,100 / 420 / 680 / 151 sources | ch03 §3.1.1, ch05 | yes, corpus-facts block | measured from `data/isafetybench/` | **PASS** |

**Result: 9 PASS, 16 sheet-coverage FAIL.** Fifteen of the sixteen failures do trace to a real
primary artifact — they are not fabricated — but the number sheet states its own rule in the
Historical block: *"Any 4-class number that appears in the final thesis must first get a row here."*
The entire 4-class accuracy surface (rows 1–19) violates it. The Historical block currently holds
only the nine **P1-edge** rows added by the 2026-10-02 trace audit; it holds **no 4-class accuracy
rows at all**.

**Proposed fix (not applied).** Add a "Historical — 4-class accuracy" sub-block with one row per
number 1–19, each naming `sweep_results/relation_tokens/canonical/lco_seeds5_yolo26L1_CANONICAL.log`
or `logits_yolo26.pt`. Until then an examiner who audits the abstract through the sheet finds
nothing. Rows 13–17 additionally need their generating command recorded, since §7.5 documents that
these five values were **wrong twice before** (wrong artifact, then wrong predictor) — they are the
highest-risk numbers in the thesis and the least traceable.

## 2. Citation audit

All 22 arXiv-style citations across `report/**` decoded by YYMM prefix and compared to the inline
year. **22/22 consistent** (preprint year ≤ claimed publication year in every case; the four
one-year-later claims — 1506.02640/2016, 2304.08069/2024, 1812.03982/2019, 2104.04527/2022,
2308.11681/2024 — are preprint-vs-venue, which is correct, not the misdating pattern the fact-check
flagged).

Known-bad citations from `critique-2026-09-19-research-factcheck.md`, re-checked verbatim:

| Known-bad citation | Status 2026-10-02 | Evidence |
|---|---|---|
| Dey et al. 2024 (fabricated attribution) | **ABSENT** | `grep -rn 'Dey' report/*.md` → no match |
| Nadeau–Bengio via Semantic Scholar AUC-fairness slug | **method used, bad link gone, no bibliography entry** | 4 files use the test by name; no reference entry anywhere |
| Focal loss misattributed to insurance-fraud paper | **ABSENT** | no focal-loss citation in `report/**` |
| SOAR, dead `thecvpr.org` domain | **ABSENT** | no match |
| "Scene Memorization" (wrong paper) | **ABSENT** | no match |
| Zhu et al. 2019 (phase-10 left this unresolved) | **ABSENT — resolved by removal** | no match; nothing left to verify |

**Remaining citation defect (open).** Nadeau–Bengio is applied as the decisive statistic in ch06,
ch07 §7.7.1 and `chapter-04-section-kfold-cross-validation.md` with **no bibliography entry in any
of them**. Correct entry to add: C. Nadeau and Y. Bengio, "Inference for the Generalization Error,"
*Machine Learning*, vol. 52, no. 3, pp. 239–281, 2003. DOI: 10.1023/A:1024068626366.

**FIXED IN PLACE 2026-10-02** (commits e489ac5, fb20043): Önal & Dandıl gains issue "no. 29" in
ch03/ch04/ch07; DETR [2] gains pages + DOI; ViT [3] gains its arXiv ID; SlowFast page range
corrected 6201–6210 → 6202–6211; **iSafetyBench added as [26]** with venue, arXiv ID, licence and
project page — the headline corpus had been cited nowhere.

## 3. Defect hunt

| # | Category | Verdict | Evidence |
|---|---|---|---|
| a | `two_proportion_z` on paired data | **PASS with one exception** | `run_full_eval.py`, `ensemble_relation.py`, `run_ablation_significance.py`, `run_relation_transformer.py` all compute McNemar as the test and print `two_proportion_z` labelled "diagnostic-only ... invalid independence". **Exception:** `eval_utils/suite.py:115` writes `out['two_proportion_z']` unconditionally with no `diagnostic_` prefix and no flag, unlike `paired_tests.py:186` which namespaces it as `diagnostic_two_proportion_z`. Any consumer reading `suite.py`'s dict by key gets an unlabelled unpaired p-value. **Proposed fix:** rename the key to `diagnostic_two_proportion_z` in `suite.py` and update `eval_utils/tests/test_suite_integration.py:49`. |
| b | Selection on test | **PASS** | No "chosen/selected/tuned on test" phrasing in `report/**`. Threshold is the untuned 0.5; folds fixed seed 0 before scoring; test-touch log shows 2 pre-registered / 0 spent. |
| c | 4-class / binary mixing in comparison tables | **PASS** | No line in `report/**` carries 84.4 % and 84.2 % together outside the explicit caution blocks. The byte-identical banner + the "84.2 % likewise denotes two quantities" block now sit in ch01/03/04/05/06. **Residual risk:** the banner is absent from the abstract, ch02, ch04-section-kfold and ch07 — see the blocking finding. |
| d | Residual "detector undertrained" language | **PASS** | no match for `undertrained`, `under-trained`, `retrain the detector` in `report/**`. |
| e | "zones are unused" phrasing | **PASS** | verbatim grep clean. |
| f | "16 entity slots" (tokens are 16 frames × 8 slots) | **PASS** | no match for `16 slots` / `16 entity slots`. |
| g | Headline consistency across files | **FIXED IN PLACE then re-broken at the abstract** | Five sentences naming P3 as the thesis headline were corrected (e489ac5). The abstract was not, and still leads with the 4-class lane. |
| h | Stale numbers pending re-measure | **OPEN, declared** | abstract's `3.45 decisions/s`, Bảng 5.5N/5.5O/5.5P and §5.6.9 shares all pre-date the preprocessing fix. Predictions P-PRE1..P-PRE4 are pre-registered in `finding-2026-10-02-preprocessing-cost-is-a-channel-swap.md`; device unreachable since 2026-10-02, so the re-measure has not run. Not a defect — a declared pending state — provided no number moves before the falsifiers are adjudicated. |

## 4. Confounding-finding framing check (phase-10 step 7)

Required framing, each checked against `report/**`:

| Sub-task | Required statement | Present? |
|---|---|---|
| carry (CarryOverload, 86 clips, labels 3/7) | trivial detector lookup, **not** a generalisation win | **YES** — abstract names the 100 vs 100 tie and explicitly refuses to rest the claim on the per-class numbers; §7.5 reports 100/100/0 |
| panel | structurally unanswerable by protocol construction, **not** a model defect | **PARTIAL** — §7.5 reports OpenedPanel 88.7 vs 88.0 as a tie, but the "structurally unanswerable" mechanism is not stated in the abstract; the abstract's hedge ("consistent with", "do not rest the claim") covers it without naming it |
| walkway | the one real, still-open, attackable failure | **YES** — 70.5 vs 17.6 reported; `finding-2026-09-19-walkway-geometry-is-unsalvageable.md` stands |
| zone | the genuine cross-camera positive | **YES** — the +16.9 pp margin is the zone-carried result |
| encoder search | **closed** (E3 frozen), not "ongoing" | **YES** — G14/G15/G16/G17 all adjudicated in Block 1; G14 FAILS is reported against our own interest |

No deviation serious enough to block. Recommend naming the panel mechanism explicitly in §7.5 so the
hedge does not have to carry it.

## 5. The 15 defence questions, rehearsed against the current thesis

Base text: researcher-02 §6, written for the superseded 691-clip / 8-cluster lane. Rewritten here
for the two-lane thesis. `[PENDING FREEZE]` marks any answer whose number is not yet final.

**Q1 — "Which number is your real performance?"**
Two lanes, never averaged. Binary hazard/normal on iSafetyBench under LSO-1100 (1,100 clips, 151
source groups, 0 carrying both classes): **84.2 % balanced accuracy**, against a pre-registered
63.3 % count-rule floor. 4-class Unsafe-Net, same-distribution 64-clip test: 84.4 % top-1 — but that
protocol is scene-confounded and a camera-guessing baseline scores 81.2 % on it, so the honest
4-class number is the leave-camera-out one, 81.2 % macro. The deployed number is 84.2 %, because
that is the graph on the device.

**Q2 — "Why is the deployed head 84.2 % when 84.4 % exists?"**
B1 (201,090 params) was exported, not B1w (333,570), under a pre-registered *edge* criterion —
smallest model — written into the export script before the run, not chosen after seeing accuracy.
The two are statistically indistinguishable. The rule we bound ourselves to is: quote the deployed
model's own OOF number, never `max(B1, B1w)`. Putting 84.4 % beside a B1 latency would attribute
one model's accuracy to another model's runtime. A B1w engine has **not** been exported or
parity-checked, and we say so.

**Q3 — "Your corpus is imbalanced (420/680). Did you account for prior shift?"**
Balanced accuracy is the reported metric throughout; it is macro-average recall and invariant to
test prior. Majority-class accuracy on this corpus is 61.8 % and the count-rule floor is 63.3 %;
both are printed beside the headline so that neither the imbalance nor a trivial detector statistic
can be mistaken for signal.

**Q4 — "Could cross-corpus contamination explain anything?"**
Ruled out by measurement, not assumption: 64-bit dHash over 3 frames per clip, 2,059 vs 3,300
hashes, **0 exact collisions, minimum cross-corpus Hamming distance 13/64** against a threshold of
6 (R46). The nearest pair is a false near-neighbour at that distance.

**Q5 — "Is 1,100 clips enough?"**
Modest, and the intervals say so. The binding constraint is not clip count but **source count**: at
151 groups with 0 carrying both classes, the unit of independence is the source, so every interval
is a cluster bootstrap over 151 sources, not a clip-level CI. The G10 margin over the floor is
+8.0 pp with cluster CI [+1.9, +13.5]. A clip-level CI here would be anti-conservative and is
never quoted as the headline.

**Q6 — "Your scene-only baseline gets 74.6 % k-fold. Isn't that a failure?" (CORRECTED — paste verbatim)**
"The scene baseline bounds how much of the in-distribution score is scene. It is a strong feature in
distribution. Out of distribution it collapses: scene-only falls from 74.6 % to **52.7 %**, a
**22 pp** gap, and 52.7 % on a balanced binary task is statistically indistinguishable from chance —
**no transferable signal survived at all**. The full model falls 14 pp from a much higher base
(83.8 %) and lands well above chance. A **larger** gap is **worse** generalisation, not better. The
earlier draft of this answer had the inequality backwards: it read the scene baseline's bigger drop
as evidence that scene 'generalises better', when a bigger drop from a lower base to chance is the
strongest possible evidence that it does not generalise at all. The correct reading: the scene
baseline is the control that **validates the protocol** — leave-camera-out detects scene
memorisation, because it destroys a pure scene model, exactly as designed."

**Q7 — "Why the recall asymmetry?"**
Operating point, not pathology. Threshold is the untuned 0.5 and was never tuned on test. Where the
asymmetry appears it is reported per class with its miss-rate and its cost-sensitive score, and the
deployment recommendation is advisory-with-human-review, not autonomous.

**Q8 — "Why no temperature scaling?"**
Calibration is measured (ECE 0.097 on the 4-class blind rule) and reported raw rather than
cherry-picked. Under domain shift the learned temperature does not transfer; per-camera temperature
is recommended as future work, not claimed as a result.

**Q9 — "Your 10:1 cost matrix — is it defensible?"**
It is a *choice*, pre-specified, not a measurement, and reported as such. Results are shown across a
range of ratios so the conclusion does not hinge on the exact number.

**Q10 — "Balanced accuracy is not a proper scoring rule."**
Correct, and both are reported: balanced accuracy as the operational metric (a threshold must be
chosen), PR-AUC and ROC-AUC as threshold-free companions. On the binary lane: PR-AUC 0.760,
ROC-AUC 0.895 for ESTA; the B1 headline is reported with its own interval.

**Q11 — "151 source groups — how do you know this generalises to new sites?"**
We do not claim it does. LSO-1100 is evidence within the corpus we have, and the honest statement is
the one in §3.1.1: source–label confounding is **diluted, not cured**. 151 groups make the test
statistically powered where Unsafe-Net's 8 do not, and the pre-registered count-rule control
(63.3 %) shows the protocol detects a scene-free shortcut when one exists. True multi-site
generalisation needs a multi-facility corpus; that is future work, named as such.

**Q12 — "Is the model useful at these numbers?"**
Advisory deployment, with the measured cadence: the composed loop sustains 3.45 decisions/s
`[PENDING FREEZE — pre-dates the preprocessing fix]` against a corpus median contract of 1.249 dec/s
(margin 2.76×), falling to 1.38× at the p95 source rate of 60 fps. The 1080p case is **not measured**
and is declared the most likely way the deployment slips real time.

**Q13 — "Why does one group contain only one class?"**
Every group does: 0 of 151 sources carry both classes. That is a property of YouTube-collected
safety corpora, disclosed as R41, and it is exactly why leave-source-out is mandatory rather than
optional — a random fold turns source identity into a perfect label proxy and inflates balanced
accuracy by **+15.4 pp** (G12, cluster CI [+10.6, +21.5] excluding 0, McNemar b=186 c=8 p=3.6e-45).
The mechanism control is the 3-feature count rule, which cannot memorise source identity and moves
only **+1.0 pp** under the same swap.

**Q14 — "Several protocols — are you picking the flattering one?"**
The opposite: the pre-registered primary is the strictest one (LSO-1100), and the inflated protocol
is reported **as a measurement of inflation**, not as a result. Three gates failed and are reported
failing — G14 (entity decomposition adds nothing, −0.5 pp, CI includes 0), G15 and G16. Zero test
touches have been spent of the two pre-registered.

**Q15 — "What is the contribution, if the architecture gates failed?"**
(1) A leak-free headline on 1,100 clips under source-grouped folds, with a pre-registered floor
beside it. (2) A quantified protocol-inflation result (+15.4 pp) with a capacity control proving the
mechanism is source memorisation. (3) A falsified entity-decomposition hypothesis, reported against
our own interest rather than buried. (4) An end-to-end edge characterisation of the *composed* loop
— not a sum of stage timings, which understates by 3.97× — on the graph that actually produced the
accuracy number. The contribution is a measured, honest system plus the protocol corrections the
benchmark needed.

## Unresolved questions

1. Does the abstract get re-led by the R48 binary lane, or does the banner get withdrawn? These
   contradict each other today and the abstract is what a committee reads first. **This is the one
   blocking item.**
2. Will the 4-class accuracy surface (19 numbers) get sheet rows, or will the sheet's own
   "must first get a row here" rule be relaxed and stated as relaxed?
3. `eval_utils/suite.py:115` — rename the unlabelled `two_proportion_z` key? It is the last place an
   unpaired p-value is exposed without the diagnostic prefix.
4. Nadeau–Bengio has no bibliography entry in any of the three files that rely on it.
5. §7.5's κ/MCC/ROC-AUC/PR-AUC/ECE were wrong twice before (wrong artifact, then wrong predictor).
   They have no recorded generating command. Re-derive from `logits_yolo26.pt` before the freeze?
6. Device unreachable since 2026-10-02, so P-PRE1..P-PRE4 remain un-adjudicated and every composed
   cadence number in the thesis is pre-fix. Re-run this trace audit after that measurement.
