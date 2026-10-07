# Phase 10: Adversarial Review + Defence Q&A

## Context Links
- Master plan: `plan.md` — Lane E scope, gate G9 (15 answers rehearsed)
- `research/researcher-02-evaluation-protocol-and-metric-suite.md` §6 — the 15 defence Q&A (base text; Q6 is broken, see below)
- `reports/critique-2026-09-19-research-factcheck.md` — full citation-verification table (24 entries), fatal/major/minor severity ranking, the Q6 arithmetic-inversion proof, the "table presented as real" defect
- `reports/finding-2026-09-19-lco-failure-is-camera-label-confounding.md` — the confounding decomposition this review must confirm is stated correctly, not diluted into "zones are unused"
- `reports/finding-2026-09-19-detector-starvation-root-cause.md` — detector-exoneration facts to cross-check against any residual "detector undertrained" claim
- `reports/audit-2026-09-19-repo-capability-and-feasibility.md` §8 — `two_proportion_z` misuse status (3 scripts, unresolved as of audit date)
- Input: `plans/2026-09-19-thesis-recovery-architecture/number-sheet.md` (produced by Phase 09) — read-only
- Input: `report/**` — read-only
- Depends on: Phase 09 substantially drafted (needs a report + number sheet to review); runs continuously 09-20 onward on whatever draft exists, final pass 10-06→10-10 per calendar

## Overview
- **Priority:** P1 (gate G9, defence-readiness blocker)
- **Status:** pending — Lane E, independent of Lane D's drafting, starts as soon as any report draft + partial number sheet exist
- **Brief:** An independent agent adversarially reviews the thesis draft, the number sheet, and every citation, and rehearses all 15 defence questions with corrected written answers. Output is a single `review-*.md` report; this lane never edits `report/**` or code — only writes its own review file.

## Key Insights
- Two known FATAL defects already found in researcher-02's own material must not survive into the thesis, and this review's first job is confirming they were actually fixed, not just noted: (1) Q6's answer is logically inverted — a **bigger** gap (22pp scene vs 14pp full-model) is **worse** generalization, not better; scene-only's 52.7% is ≈chance, i.e. no signal survived, the opposite of "generalizes better." (2) §7.1's numeric results table was written as if binary-task numbers were already measured; the audit shows 0/2 binary test touches spent as of 2026-09-19 — any thesis table that still reads this way is presenting fabricated-by-omission data.
- Citation integrity is now a first-class review task, not a footnote. The fact-check found one **fabricated attribution** (Dey et al. 2024 — URL resolves to an unrelated real paper, no author named Dey), one Nadeau-Bengio citation pointing to an unrelated AUC-fairness paper via a Semantic Scholar slug, one focal-loss citation misattributed to an insurance-fraud paper, one dead-domain citation (SOAR, `thecvpr.org`), and a systematic pattern of arXiv IDs misdated by one to two years (YYMM prefix decodes later than the inline year given). Every bibliography entry that reaches the final thesis must be re-decoded by hand.
- "Zones are unused" is FALSE and must never appear: zones are wired into token dims 6-9 (`iou_walkway`, `iou_restricted`, `foot_in_walkway`, `norm_dist`). The correct, narrow claim: they are computed for person classes 0/1 only, absent from the detector label space (classes 2/3 have zero instances), and identically zero on `view_2`/`view_7` — which is **correct behaviour, not a gap**, because those 86 clips are exactly the carry sub-task (labels 3/7) and are 86/86 correct in all six encoders. That last fact is itself evidence for the carry-triviality disclosure and should be checked as a *required* sentence, not just a forbidden one.
- Two further required greps: (a) **"16 entity slots" is wrong** — headline tokens are `(566|125, 16, 8, 74)`, i.e. 16 frames x 8 slots; fail any capacity/complexity statement that says 16 slots. (b) **fold/view conflation** — fold ids and view ids agree only 81.7%, so every LCO claim must be phrased in fold terms and carry the `run_kfold_relation.py:106` `[LIMITATION]` banner; flag any view-phrased LCO sentence.
- The confounding finding raises the bar for this review: it is not enough to check that headline numbers trace — the review must confirm the thesis states the **mechanism** correctly (panel = structurally unanswerable by protocol construction, not a model defect; carry = trivial detector lookup, not evidence of generalization; walkway = the one real, still-open, attackable failure).

## Requirements
**Functional**
- 15 defence Q&A from researcher-02 §6, each rehearsed with a written answer in the review file. Q6 replaced with the corrected reasoning (see Implementation Steps). Any answer whose supporting number is not yet frozen is marked `[PENDING FREEZE]`, never answered with a placeholder presented as real.
- Trace audit: N >= 20 numbers sampled at random from the current thesis draft, each checked against `number-sheet.md`; pass/fail table in the review output.
- Citation audit: every bibliography / in-text citation checked by decoding arXiv YYMM prefix (or DOI/venue) against the claimed year; pass/fail table, one row per citation, explicit correct-citation replacement where wrong.
- Explicit hunt, each with a pass/fail verdict and evidence: (a) unpaired test (`two_proportion_z`) applied to paired same-clip-set data — cross-check `run_full_eval.py`, `ensemble_relation.py`, `run_ablation_significance.py` against audit §8's unresolved-as-of-date flag; (b) selection-on-test leakage — any threshold/temperature/ensemble-weight chosen on test rather than val/OOF; (c) 4-class/binary recipe mixing in any comparison table; (d) any claim the detector work does not support (e.g. residual "detector undertrained" language after the exoneration finding); (e) any "zones are unused" phrasing, verbatim grep.

**Non-functional**
- Lane E is read-only everywhere except its own output file — it proposes fixes in prose, it does not apply them.
- Review output goes only to `plans/2026-09-19-*/reports/review-*.md`, named `review-<date>-<slug>.md`.
- No compute run, no experiments, no test-touch consumption — this is a document/code-reading review only.

## Architecture
```
Lane D output (read-only inputs)          Lane E (this phase)
report/**              ----------------->  adversarial read
number-sheet.md        ----------------->  trace audit (sample N>=20)
                                            citation audit (decode every arXiv ID)
                                            15 Q&A rehearsal (Q6 corrected)
                                            defect hunt (5 categories above)
                                                  |
                                                  v
                        plans/2026-09-19-*/reports/review-<date>-<slug>.md
                                                  |
                                                  v
                                    handed back to Lane D / lead for fixes
                                    (Lane E never edits report/** itself)
```

## Related Code Files
| Type | Path | Why |
|---|---|---|
| create | `plans/2026-09-19-thesis-recovery-architecture/reports/review-2026-XXXX-adversarial-defence-audit.md` | this phase's sole output (date filled at run time) |
| modify | none | Lane E is read-only outside its own reports glob |
| delete | none | — |

Read-only inputs (not edited): `report/**`, `plans/2026-09-19-thesis-recovery-architecture/number-sheet.md`, `plans/2026-09-19-thesis-recovery-architecture/plan.md`, `reports/test-touch-log.md`.

## Implementation Steps
1. **(0.5h)** Read current state of `report/**` + `number-sheet.md` in full; note which sections are still `[PLACEHOLDER]` (pre-freeze) vs frozen.
2. **(1h)** Trace audit: randomly sample >=20 numbers from thesis text (weight toward chapter-04/06/07 and abstract), verify each against a `number-sheet.md` row + its cited log path. Record pass/fail + the row id per sample.
3. **(1.5h)** Citation audit: for every arXiv-style citation, decode YYMM from the ID itself and compare to the inline year claimed; for non-arXiv citations, verify DOI/venue/authors resolve to the claimed content. Known-bad citations to explicitly re-verify are fixed, not re-trusted: `Dey et al. 2024` (fabricated — replace with Sagawa et al. 2019 / verify Zhu et al. 2019 before use), Nadeau-Bengio (replace Semantic Scholar AUC-fairness link with Nadeau & Bengio, "Inference for the Generalization Error," Machine Learning 2003), focal loss (replace insurance-fraud paper with Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017 / arXiv:1708.02002), SOAR (replace dead `thecvpr.org` link with `thecvf.com` or arXiv:2309.01265), "Scene Memorization" (wrong paper entirely — either find the correct shortcut-learning citation or drop the claim's citation and mark it uncited). Output one pass/fail row per citation.
4. **(1h)** Rehearse Q1-Q5, Q7-Q15 as written (researcher-02 §6), updating any answer whose supporting number changed post-freeze or whose framing depends on the pre-confounding-finding narrative (Q1, Q2, Q6-adjacent Q7, Q11, Q13 all likely need updating to reflect panel/walkway/zone/carry decomposition rather than a single undifferentiated "domain gap").
5. **(0.5h)** Replace Q6 with corrected reasoning: state explicitly that a **larger** gap (scene: 22pp) is **worse**, not better, generalization; scene-only's 52.7% LCO score is statistically indistinguishable from chance (50%) on a balanced binary task, meaning no transferable signal survived; the full model's smaller 14pp gap from a much higher base (83.8%) is the actual evidence of partial generalization. Write this out verbatim in the review file, ready to paste into the thesis answer bank.
6. **(1h)** Defect hunt, one pass/fail line per category: (a) grep `run_full_eval.py`, `ensemble_relation.py`, `run_ablation_significance.py` for `two_proportion_z` usage on paired same-clip data — confirm current status (audit flagged unresolved as of 2026-09-19, these files show `M` in git status; re-check if fixed); (b) scan methods/results chapters for any threshold/temperature/weight chosen with test-set visibility; (c) scan every comparison table for 4-class rows mixed with binary rows; (d) grep for residual "detector undertrained"/"detector retrain" language contradicting the exoneration finding; (e) grep verbatim for "zones are unused" or equivalent phrasing.
7. **(0.5h)** Confirm the confounding-finding subsection (Phase 09 step 5) states: carry = trivial lookup (not a generalization win), panel = structurally unanswerable by protocol construction (not a model defect), walkway = the one real open failure, zone = the genuine cross-camera positive result, encoder search = closed (E3 frozen, not "ongoing"). Flag any deviation.
8. **(1h)** Write `review-<date>-adversarial-defence-audit.md`: trace-audit table, citation-audit table, 15 Q&A (Q6 corrected) block, defect-hunt table, overall verdict (ready/not-ready for G9), unresolved questions.

## Todo List
- [ ] Read current `report/**` + `number-sheet.md` state
- [ ] Trace audit: >=20 sampled numbers vs sheet, pass/fail table
- [ ] Citation audit: every citation, YYMM-decoded, pass/fail table, replacements for the 5 known-bad citations
- [ ] Rehearse Q1-Q5, Q7-Q15 with current numbers
- [ ] Write corrected Q6 verbatim
- [ ] Defect hunt: two_proportion_z, test-leakage, 4-class/binary mixing, unsupported detector claims, "zones unused" grep
- [ ] Confirm confounding-finding subsection framing (carry/panel/walkway/zone/encoder-closed)
- [ ] Publish `review-<date>-adversarial-defence-audit.md` with overall verdict

## Success Criteria
- Review file exists with all four audit tables (trace, citation, Q&A, defect-hunt) fully populated, no `[TODO]` left inside it.
- Q6 corrected answer present verbatim, arithmetically checked (22 > 14, both parsed as "gap," bigger gap = worse generalization).
- Zero surviving instances of the 5 known-bad citations; zero surviving "zones are unused" phrasing; zero surviving 4-class/binary mixing in comparison tables.
- Overall verdict states explicitly whether G9 (15 answers rehearsed) is met.

## Risk Assessment
| Risk | Mitigation |
|---|---|
| Review runs against a stale pre-freeze draft, findings go stale | Re-run trace + citation audit once more after 2026-10-05 freeze, before 10-10 defence package close |
| Lane E accidentally edits `report/**` while checking something | Read-only discipline — if a fix is obvious, write it in the review file as a proposed diff, never apply it directly |
| Citation audit missed one bad reference (only sampled, not exhaustive) | Audit is exhaustive (every citation), not sampled — sampling is trace-audit only (numbers) |
| New confounding-finding subsection introduces a fresh unverified claim | Steps 2-3 apply equally to any new numbers/claims added by Phase 09's dedicated subsection |

## Security Considerations
- No secrets in scope. Review file may quote clip identifiers/camera group ids — internal only, not PII, but avoid quoting raw video file paths that could imply facility identity if this review is ever shared externally.
- Lane E's read-only-elsewhere constraint is itself a security/integrity control — do not weaken it even temporarily to "quickly fix" something found during review.

## ESTA-specific attack surface (added 2026-09-29 with the reframe)

The hardest questions a committee can ask about the new headline, and where each is already
answered by design rather than by argument. Answers marked **GAP** are not answered and must be
conceded in the thesis rather than defended.

1. *"Your gain is just a better backbone, not your architecture."* — B1 and B1w use the **same**
   frozen CLIP ViT-B/32, the same frames, the same recipe. G14 is ESTA minus the better of them, so
   the backbone is held constant and only the entity axis varies. G15 separately quantifies what
   the backbone alone bought (ESTA vs the old 64-d E0).
2. *"You gave the entity model more parameters."* — B1w is the capacity-matched baseline (333,570
   vs ESTA's 352,258, within 5.3%), and G14 must clear the **higher** of B1/B1w.
3. *"A video model with no detector would do as well."* — that is B2, and it is a Kinetics-400
   fine-tuned VideoMAE, not an unsupervised encoder. If it wins, the pre-registration says the
   detector is not justified on accuracy grounds and the thesis reports that.
4. *"Different detector, different answer."* — G17 re-extracts with YOLO26s (NMS-free, end2end) and
   reports the result whichever way it points. The headline detector was fixed in advance.
5. *"Did you pick the folds/threshold/seed that worked?"* — folds are `StratifiedGroupKFold(10)`
   seed 0 over the 151 source ids, fixed before the corpus was ever scored; the threshold is 0.5
   and was never tuned; all 3 seeds are ensembled, none selected. 0 test touches.
6. *"Why is your accuracy below the published literature?"* — the methodology chapter: on this
   corpus 0/151 sources carry both classes, so a random fold makes source identity a perfect label
   proxy and inflates balanced accuracy by +15.4 pp (G12, McNemar p=3.6e-45). The comparison is
   protocol-to-protocol, not number-to-number.
7. *"Your detector is COCO-pretrained; helmets, harnesses and guarding are not COCO classes."* —
   **GAP.** Entities are COCO groups, so safety-specific objects are invisible to the entity arm
   and whatever the model learns about them must come through CLIP appearance of the crops. The
   domain detector in `best.pt` does not fix this: it was trained on Unsafe-Net's 8-class warehouse
   taxonomy over 2 IP cameras, and applying it to 151 YouTube sources is a domain shift that would
   confound the very comparison it was meant to strengthen. Concede it; name it as the first item
   of future work.
8. *"Why not fine-tune the backbone end-to-end?"* — **GAP, resource-bound.** 1,100 clips over 151
   single-class sources is far too little to fine-tune a ViT without memorising source identity,
   which is precisely the failure mode the protocol chapter documents. Frozen backbones are the
   conservative choice here, and the thesis should say that is a choice, not a result.
9. *"Does the demo match the measured system?"* — the demo imports the detector, crop rule, CLIP
   tower and token layout from the feature builder itself and runs the same 3-seed ensemble, so it
   cannot drift; the threshold shown is the untuned 0.5 with the probability displayed beside it.

## Next Steps
- Findings hand back to Phase 09 (Lane D) for report fixes and Phase 11 (contingency wording) if the confounding-finding framing needs adjustment.
- Final re-run of this phase folds into the 10-06->10-10 "writing, trace check, adversarial review" calendar week, immediately before demo rehearsal.
- Unresolved: whether `Zhu et al. 2019` (cited alongside the fabricated Dey citation) is itself verifiable — critique report flags it as not independently checked; this phase must resolve it, not re-defer it.
