# Audit — thesis document & non-code deliverables readiness, 2026-09-14 (D-5 to defence)

Scope: report/, docs/, README, plans/.../slides, per user's 5-point checklist. Re-verifies
`review-2026-09-13-thesis-alignment.md` against current repo state (git HEAD `43657ea` + uncommitted
changes per gitStatus) rather than re-deriving from scratch. Method: file listing/wc, git log/diff,
grep sweeps for headline numbers and VN diacritics, targeted reads of flagged lines.

Bottom line up front: **VN translation is still at 0%, 5 days left.** This is the single blocker for
D1/D4. Everything else below is fixable in hours; this is not.

---

## 1. report/ inventory

| File | Lang | Words | Maps to D4 | State | Stale/superseded markers |
|---|---|---|---|---|---|
| abstract.md | EN | 839 | EN abstract | Draft, actively maintained | None found, but see §2 new finding (Jetson/YOLO26 claim) |
| chapter-01-introduction.md | EN | 2714 | Ch1 Giới thiệu | Draft, has pipeline-attribution banner (added Sep6, `effc9aa`) | None |
| chapter-02-related-work.md | EN | 2262 | Ch2 Tổng quan | Draft | One false-positive VN grep hit: L195 "multi-luồng" gloss, harmless |
| chapter-03-methodology.md | EN | 2247 | Ch3 Phương pháp | Draft, P1-anchored, P3 only via RT-DETR-deviation note | Per Sep13 review: should be restructured to lead with P3 |
| chapter-04-experiments-and-results.md | EN | 10699 | Ch4 (part 1/3) | Draft, has pipeline-attribution banner (top of file) | 71KB; last full number-consistency audit run 2026-09-06, stale relative to S16-S20 |
| chapter-04-section-kfold-cross-validation.md | EN | 2080 | Ch4 (part 2/3) | **Standalone draft, marked "slots into Chapter 4 after §4.8"** — merge not executed | Still unmerged, same as Sep13 |
| chapter-05-edge-deployment.md | EN | 5994 | Ch5 Triển khai biên | Draft, P1-only content | Own top-of-file rule broken in body (see §2); **contains zero mention of the P3 edge-cost/FAIL result (S20)** — new finding, worse than Sep13 review stated |
| chapter-06-conclusion.md | EN | 2893 | Ch6 Kết luận | Draft | Correctly hedges P1 (56/64) at L145; no full two-pipeline banner like Ch1/Ch4 |
| chapter-07-multi-entity-and-scene-robust-evaluation.md | EN | 3844 | Ch4 (part 3/3), P3 headline | Draft, **actively updated today** (uncommitted diff adds LCO seed-variance analysis, S19 material) — best-maintained chapter | None; matches abstract |
| thesis-reframe-deployment-led.md | EN | 1348 | none (pre-P3 draft) | **Marked SUPERSEDED 2026-09-06, file still present, unflagged in its own render** | Same open item as Sep13 review's unresolved Q4 — still not deleted/archived |

**Total report/ word count: 34,920 words, 100% English** (grep -P for VN diacritics across all files returns zero real hits — same finding as Sep13 review, unchanged).

**No assembled single document.** `report/thesis-assembled.md` does not exist (checked directly). Phase-04's own day-12 deliverable has not been started.

**Figures/tables:** `figures/` has 136 files (PNG/PDF/mmd, calibration curves, confusion matrices, embeddings, Grad-CAM); `deployment/artifacts/figures/` has 6 more. Populated and non-trivial, but per `phase-06-defence-slides-and-qa.md`'s own note, `figures/` "currently has content from the OLD pipeline" and needs re-pointing at the P3 headline — not verified fresh in this pass beyond file count.

**7-chapter-vs-6-chapter problem confirmed unresolved:** current file count (Ch1-Ch7 + kfold-section + superseded reframe) would produce 8 sections if assembled as-is. No merge executed.

---

## 2. Headline-number consistency — grep results and contradictions

Counts across report/*.md: 54/64=43, 56/64=86, 53/64=52, 80.8=10, 81.2=16, 16.9=13, 13.3=2, 62/64=12, 8.62=4.

**D3 check (must not claim beating paper): PASS.** Grepped for "beat/surpass/outperform/exceed" near
62/64 or 95.81 across all report/*.md — the only hit is abstract.md:45, which explicitly states the
thesis does **not** beat 95.81%. No violation found.

**Ch5 self-contradiction — CONFIRMED STILL PRESENT, unchanged from Sep13 review** (file not touched
since last commit per git status). Ch5's own banner (chapter-05-edge-deployment.md:16-18) states: "P1's
56/64 is test-selected... must never be presented as the result; its honest blind counterpart is
53/64." Body then states bare, unhedged 56/64 at:
- **L36**: "this ensemble achieves a top-1 accuracy of 56/64 = 87.5%... matching the PyTorch training baseline" — no hedge.
- **L205, L208**: Table row "CPU-FP32 | 55/64 | 51/64 | **56/64**" + prose "match the PyTorch training baseline exactly (56/64 = 87.5%)" — no hedge.
- **L381**: TRT-FP32 table row "**56/64 = 87.50%**" — no hedge.
- **L700-701**: §5.8 conclusion, "CPU-FP32 and CoreML-FP32 ensemble accuracy is 56/64 = 87.5%, matching the training baseline exactly" — no hedge.

4 sites, exactly as the Sep13 review found. Fix is mechanical: append "(test-selected; blind 53/64)" or replace with the pair, per that review's action item #9. ~30min-1h.

**New finding, not in Sep13 review — abstract.md:36 overstates edge-deployment evidence.**
> "The work is supported by ... and an edge-deployment characterisation (**YOLO26 + Jetson
> TensorRT-INT8**)." (abstract.md L36)

This reads as if P3 (YOLO26, the headline pipeline this abstract leads with) has a physical Jetson
TensorRT-INT8 measurement. It does not. Per brief.md S20 (verified against
`reports/laneb-p3-edge-cost-and-export.md`, measured 2026-09-09): P3 was benchmarked only on an idle M3
Mac (0.55 dec/s) and the Jetson number is a **calibrated projection that FAILS** the 3.0 dec/s
requirement at every TensorRT precision (1.13-1.71 dec/s projected). The only physical Jetson
TensorRT-INT8 measurement in the project belongs to **P1** (YOLO11s), not YOLO26/P3. As written, a
reader of only the abstract — the most-read page — would believe the scientific-headline pipeline has
measured, passing edge performance. It does not, and the honest negative result (P3 fails real-time on
Jetson, P1 is the deployed system) is itself a defensible thesis finding per the two-pipeline framing —
but the abstract currently obscures it rather than stating it. Fix: either drop "Jetson TensorRT-INT8"
from this sentence (keep it P1-scoped) or add one clause naming the P1/P3 split explicitly, matching
the two-pipeline paragraph recommended in the Sep13 review. ~15-30min once the two-pipeline paragraph
(item 1 below) is drafted.

**Cross-chapter number spread — largely mitigated since Sep13, not fully.** The Sep13 review flagged
"four different top-1 numbers, nothing tells a reader which is 'the' number." Verified now: Ch1
(L12-18), Ch4 (top-of-file), and Ch5 (L16-18) all carry the same pipeline-attribution banner table
(P1/P1-Jetson/P3 rows with test-selected vs blind columns), dated Sep6 — **this predates the Sep13
review**, so that review's characterisation of Ch1 as "mixed, a third number" is technically accurate
per-number but undersells that the banner already labels it. Two things still genuinely missing:
(a) **README has no such banner** and still leads bare with P1's 56/64 (see §4); (b) **Ch6 has only one
hedged line (L145), not the full banner** — a reader who reads only Ch6 still lacks the reconciliation.

**YOLO11-vs-YOLO26 ablation — Sep13 review's claim is FALSE, correcting it here.** That review states
"there is no YOLO11 vs YOLO26 ablation table in the thesis." Verified: chapter-07 §7.7, lines 140-166,
has exactly this: a table with rows "entity-relation (YOLO26, headline)" and "entity-relation (YOLO11,
ablation)" (80.8±4.0% vs 80.7±3.6%, both non-significant vs scene-only), plus the prose sentence
"Detector generation is not the bottleneck." Committed 2026-09-06 (`b2f530d`), before the Sep13 review
was written — this was a miss in that review, not a regression. No action needed here; MT1 coverage of
the YOLO version comparison is adequate, it just needs pulling into Ch3/Ch4 prose for a reader who
doesn't reach §7.7.

**Two-proportion-z code fix — CONFIRMED STILL NOT DONE.** `run_relation_transformer.py:24,220,236`,
`ensemble_relation.py:13,43`, `run_full_eval.py:20,123-124` all still call `two_proportion_z` (assumes
independent samples) rather than `mcnemar_discordance_bound` for "vs paper" comparisons, including one
call at `run_full_eval.py:124` that compares directly against the paper's full-bbox 62/64. The printed
z=-0.26/-0.27, p=0.79 figures already in README.md:91 and chapter-05-edge-deployment.md:393 are vs the
**no-bbox baseline (57/64)**, not vs 62/64, so no D3 violation is currently printed — but any fresh run
of these three scripts before the deadline will still emit an invalid-independence p-value. Sep13
review's action item #10, ~1-2h, unchanged priority.

---

## 3. Coverage of the thesis title

**YOLO11 vs YOLO26 comparison:** present (chapter-07 §7.7, see above) but confined to Ch7; not
referenced from Ch3 (methodology) where a reader would expect to see it introduced. Cheap fix: one
cross-reference sentence.

**Transformer's contribution isolated by ablation:** yes — chapter-04-section-kfold-cross-validation.md
and chapter-04's Table 4.9.2 report a head-vs-linear-probe test (Δ=-4.65pp, NB p=0.007, later
re-verified at p=0.015 after the clip-bootstrap fix, S6/S6b) showing the Transformer head is NOT simply
matched by a linear probe under random split — a real, negative-leaning, audited finding. Present and
sound.

**4-class vs 8-class scope:** **still not explained anywhere**, confirmed via grep (`grep -n "follows
the paper|4-unsafe-class protocol|8-class safe/unsafe" report/*.md` → zero hits). Ch3 states "8 classes
(Table 3.1), grouped into 4 safe and 4 unsafe" and Ch7 describes an "8-class entity ontology" — but
these are two different 8s (8 *behaviour* classes vs 8 *entity* classes used as Transformer input), and
nowhere does the text say explicitly "we follow the paper's own 4-unsafe-class evaluation protocol;
full 8-class safe/unsafe accuracy is not separately headlined." Sep13 review's action item #8, still
open, ~30min, high committee-risk if skipped (proposal literally asks for "4 an toàn – 4 nguy hiểm").

**CarryOverload single-rule-detector caveat:** grepped, zero hits — still not added anywhere in
report/. Sep13 review's action item #7, still open, near-zero cost, high committee-attack-risk (a
sharp examiner who checks the confusion matrix and the token-construction code will ask why one class
is always 100%).

---

## 4. Slides / Q&A brief, docs/, README

**Slides: not started.** `plans/2026-09-06-thesis-2week-final/slides/` exists but is **empty** (0
files, checked directly). `phase-06-defence-slides-and-qa.md` is a plan file only (status: pending),
gated on headline freeze (done, Sep6/9) and Jetson metrics (done for P1). Nothing blocks starting now
except time. This was scheduled to start day 9-10 (Sep14-15) — today.

**docs/ — all 7 files present**, matching the CLAUDE.md-mandated structure exactly: code-standards.md,
codebase-summary.md, deployment-guide.md, design-guidelines.md, project-overview-pdr.md,
project-roadmap.md, system-architecture.md. `system-architecture.md` is uncommitted-modified today and
is, per Sep13 review, "the best-maintained document in the repo" (has explicit P1/P2/P3 map). Not
independently re-verified line-by-line in this pass, but no contradicting evidence found.
`project-roadmap.md` itself documents that README rewrite / repro scripts were explicitly out of its
own creation-pass scope and remain open — self-consistent, not stale in a misleading way.

**README — confirmed still stale, identically to Sep13 findings:**
- Headline table still leads with P1's 56/64 / 89.28% macro (README.md:21-22,36-37), with a banner
  saying "P3 accuracy-headline re-benchmark pending" (README.md:15) — false as of Sep6/9, P3 has been
  frozen and reported for over a week.
- All 12 previously-flagged broken file references confirmed still missing on disk (re-checked
  directly): `run_finetune_e2e.py`, `model_arch.py`, `feature_dataset.py`, `temporal_video_loader.py`,
  `tools/preextract_yolo_features.py`, `tools/scan_bad_videos.py`, `GAP_ANALYSIS.md`,
  `run_safe_unsafe.py`, `run_temporal_fusion.py`, `run_scenario_cascade.py`, `plot_figures_binary.py`,
  `BASELINE.md`. Current-generation scripts (`run_relation_transformer.py`, `run_kfold_relation.py`,
  `run_lco_eval.py`, `build_relation_tokens.py`, `model_relation_head.py`, `run_full_eval.py`) are not
  mentioned.
- **No `scripts/` directory** — confirmed, `ls scripts/` fails. D4's "one-command repro per headline
  number" is unmet.

---

## 5. Day-by-day writing plan, Sep 14 → 19 (5 days + freeze day)

**Constraint stated plainly: 34,920 words of English draft, 0 words of Vietnamese, 5 days.** Per the
Sep13 review (correct, restate it since it's still the right call): **do not literally translate.**
Write condensed new VN chapters at normal-thesis length (~15,000-20,000 VN words total, not 35K),
using the EN draft as source material / mechanism reference, citing the fuller EN detail as an
appendix. Full literal translation of 35K words at thesis quality in 5 days is not achievable
alongside the other D4 deliverables (slides, README, repro, demo). This is the load-bearing schedule
risk; everything else in this audit is hours-scale.

**Day 0 remainder — today, Sep 14 (do before any VN writing starts, ~3-4h):**
1. Lock the two-pipeline reconciliation paragraph (P3 = scientific/accuracy contribution via k-fold +
   LCO; P1 = engineering/edge contribution, real-time on Jetson; P3 fails real-time, quantified 7.3x
   FLOPs/decision, 0.55 dec/s measured, projected FAIL). Paste identically into abstract.md, README.md,
   chapter-01, chapter-06. ~1.5h.
2. Fix abstract.md:36 (drop or scope the "Jetson TensorRT-INT8" claim to P1). ~15min.
3. Fix the 4 bare-56/64 sites in chapter-05 (L36, 205/208, 381, 700-701). ~30min.
4. Write the missing P3-edge-cost-FAIL section into chapter-05 (currently absent entirely) — pull
   directly from `reports/laneb-p3-edge-cost-and-export.md` and brief.md S20; this is also the raw
   material the new VN Ch5 will condense, so doing it in EN first is not wasted work. ~1h.
5. Add the two one-sentence caveats: 4-class-vs-8-class scope (Ch3 or Ch4), CarryOverload single-rule
   detector artifact (Ch4 or Ch7). ~45min.
6. Decide the chapter-merge structure explicitly (Ch4 + kfold-section + Ch7 → one Ch4, organized by
   brief §2's 8 axes as Sep13 review recommended) — decision only, execution happens as part of VN Ch4
   writing on Day 3. ~15min.

**Day 1, Sep 15 — Ch1 + Ch2 (VN):**
- Ch1 Giới thiệu (~1,800-2,500 VN words): motivation, MT1-3, the two-pipeline claim as one paragraph
  (already drafted Day 0), contributions. Source: existing EN Ch1 (2,714 words) is close to right
  length already — condense/restructure rather than expand; mostly translatable with light editing.
- Ch2 Tổng quan (~2,500-3,500 VN words): YOLO family, Transformer/RT-DETR deviation note, industrial
  safety datasets, Unsafe-Net paper (citation already corrected to 2024). Source: EN Ch2 (2,262 words)
  is survey material, largely translatable as-is with trimming.
- Also: start README rewrite (headline table + broken refs), since it is independent writing work.

**Day 2, Sep 16 — Ch3 (VN) + README/docs:**
- Ch3 Phương pháp (~3,000-4,000 VN words): **must be restructured, not translated** — lead with P3
  (entity ontology → YOLO26 → relation-Transformer) since it's the pipeline the abstract centers;
  describe P1 as the deployment variant via one comparison table (architecture/params/FLOPs), not a
  second full methodology section. This is new organizational writing using both EN Ch3 (2,247 words,
  currently P1-anchored) and Ch7's P3 architecture description as raw material.
- Finish README rewrite: correct headline table (dual P1/P3, matching Day-0 paragraph), remove/replace
  12 broken refs, add manual repro commands per headline number.
- Fill blank Gate-1 outcome table from `runs/detect/sweep_results/r2/beh_{tight,ctx}/results.csv`.

**Day 3, Sep 17 — Ch4 (VN), the hardest chapter:**
- Merge chapter-04 (10,699 words) + chapter-04-section-kfold (2,080 words) + chapter-07 (3,844 words) =
  16,623 EN words → **one VN chapter, ~4,500-6,000 VN words**, organized by brief §2's 8 axes: (1)
  paper-protocol tie + N=64 ceiling, (2) faithful reproduction, (3) k-fold reliability, (4) LCO
  cross-camera (lead result), (5) safety/FNR, (6) calibration, (7) YOLO11-vs-YOLO26 ablation pulled in
  from §7.7, (8) negative-results/withdrawn-number history as one short subsection at the end. This is
  the highest-effort chapter — condensation ratio here is ~3.5:1, steepest of any chapter, because the
  EN source is an audit trail (S1-S20) as much as a results chapter. Push the full S1-S20 trail to an
  appendix, keep only the load-bearing numbers in the VN body.
- Produce `report/thesis-assembled.md` skeleton alongside this (structure only; fill as chapters land).

**Day 4, Sep 18 — Ch5, Ch6, abstract, references, assembly:**
- Ch5 Triển khai biên (~2,500-3,000 VN words): P1 as "what is deployed and measured" (real Jetson
  numbers, condensed from 5,994 EN words); P3 edge cost as a named, quantified limitation (using the
  section written Day 0) plus future-work route (imgsz 800→512).
- Ch6 Kết luận (~1,200-1,800 VN words): two-pipeline headline restated, contributions, limitations,
  future work. Source EN Ch6 (2,893 words) translates fairly directly, it is already the most
  concise/consistent chapter.
- EN abstract: finalize (already EN per D1, mostly done at 839 words — verify it reflects Day-0 fixes).
- References: verify full bibliography against the corrected Önal & Dandıl (2024) entry and any other
  citations; VN thesis reference formatting per S01 Mẫu 2.
- Full VN assembly into `report/thesis-assembled.md`; consistency pass (re-run whatever tool produced
  `audit-cross-chapter-number-consistency.md`, last run Sep6, now 8 days stale).
- Delete or explicitly archive `thesis-reframe-deployment-led.md` (superseded, still present).

**Day 5, Sep 19 — slides, Q&A, freeze, buffer:**
- This is very late to start slides (originally scheduled day 9-10, now day 6/6) — if Day 0-4 run to
  schedule, compress slide-writing into Day 4 evening / Day 5 morning instead of leaving it entirely to
  Day 5. VN slides + Q&A brief mapping every anticipated question to a file+number (the two hardest
  questions are pre-answered in brief.md: "why not beat 62/64" = N=64 ceiling proof; "is k-fold/LCO
  comparable to the paper's 64" = different N, different question).
- Final review, freeze, defence rehearsal, buffer for whatever slipped.

**Word-count estimate, total VN thesis body:** ~15,000-20,000 words across Ch1-Ch6, roughly a third of
the current English sprawl (34,920 words) — matches the Sep13 review's own estimate and is deliberately
smaller, not smaller by accident. Detailed audit trail (S1-S20), P2 attempt narrative, seed/cluster
tables move to appendices, one paragraph each.

**What is realistically NOT achievable in this window and should be named as scoped-out now, not
discovered as a slip on Day 4:**
- Physical Jetson measurement of P3 (Sep13 review's SHOULD #11, 1.5-2 days) — timebox-incompatible
  with 5 days that also need full VN writing + slides. Recommend explicitly dropping it and keeping the
  honest Mac-projected FAIL as the Ch5 finding, unless Lane B has spare capacity this exact audit did
  not check.
- imgsz 800→512 P3 retrain (SHOULD #12, 1-2 days) — same reasoning, drop or timebox hard.
- Streaming/event-level operational metrics (§4i) — brief.md itself already names this first-cut.
- Third independent LCO re-clustering (cs=3+) — brief.md already says not blocking.

---

## Unresolved questions

1. Whether the owner wants the P3-Jetson-FAIL and imgsz-512 items attempted at all this week, given
   they cost 1.5-4 days combined against a 5-day VN-writing-dominated schedule — this audit recommends
   dropping both but that is a scope call, not a fact-check.
2. S01 Mẫu 2 assembly format (Markdown→PDF vs university .docx template) — still unconfirmed with
   advisor per Sep13 review's Q2; blocks final formatting of `thesis-assembled.md`, not its content.
3. Whether `thesis-reframe-deployment-led.md` should be deleted outright or moved to an
   `appendix/`/`archive/` subfolder — the plan above says "delete or archive," owner should pick one so
   assembly isn't ambiguous.
4. Ch3's restructuring (lead with P3, P1 as a comparison-table variant) is a bigger rewrite than a
   translation pass — confirm this is still the intended framing before Day 2, since it changes the
   shape of the chapter rather than just its language.
5. Whether any Lane B capacity exists Sep14-19 for even a partial P3 Jetson run (Sep13 review's SHOULD
   #11) was not re-checked in this pass (this is a document/deliverables audit, not a code/hardware
   audit) — if capacity exists, the schedule above would need adjusting.
