# Phase 11: Contingency + Negative-Result Narrative

## Context Links
- Master plan: `plan.md` — EV-ranked levers L1-L7, kill rules, decision date 2026-10-04, freeze 2026-10-05
- `reports/finding-2026-09-19-lco-failure-is-camera-label-confounding.md` — the mechanism-level finding this phase's wording is built on: zone genuinely generalizes, carry is trivial, walkway is the one real open failure, panel is structurally unanswerable, effect is architecture-invariant across 6 encoders **on this corpus only (R47)**
- `reports/finding-2026-09-19-detector-starvation-root-cause.md` — detector exonerated, mAP50 0.966, not part of any negative-result framing
- `research/researcher-02-evaluation-protocol-and-metric-suite.md` §6 Q15 — "what's your real contribution" framing (base text; corrected per Phase 10's citation/reasoning fixes before reuse)
- Phase 09 (`phase-09-number-sheet-and-thesis-integration.md`) — owns `chapter-06-conclusion.md`, where this wording lands
- Phase 10 (`phase-10-adversarial-review-and-defence-qa.md`) — reviews whether this wording is actually used correctly and consistently with the frozen numbers
- `plans/2026-09-19-thesis-recovery-architecture/reports/` — Lane E's home for any review of this phase's wording

## Overview
- **Priority:** P2 (contingency — used only if invoked, but must be ready before the decision date regardless)
- **Status:** pending — wording drafted now (Lane D), reviewed by Lane E, invoked-or-not decided 2026-10-04
- **Brief:** Pre-commit exact, ready-to-paste wording for the accuracy objective's possible negative outcome, and pre-commit the decision rule + date + sign-off for invoking it. The wording is written now, deliberately, before anyone knows if it is needed, so it cannot be spun after the fact.

## Key Insights
- **This is not a pure-failure fallback.** The camera-label-confounding finding changes what "the accuracy lever fails" actually means for this thesis: even in the worst case (no lever beats the E3 DeepSets baseline with a CI excluding zero), the project still has (a) a falsifiable, mechanism-level explanation for *why* — architecture-invariant across 6 encoders spanning 18k-300k params **on Unsafe-Net** (R47: not on iSafetyBench, where the same ladder spans 5.1 pp), localized to 2 of 4 identifiable sub-tasks, one of which (panel) is unanswerable by LCO's own construction, not a model defect — and (b) a genuine positive result (zone sub-task, 145 clips, 76-90% both classes, cross-camera) proving the relation-token approach generalizes where the data permits it. The wording below must lead with this, not apologize for a missing accuracy number.
- The decision of whether to *invoke* contingency wording is separate from whether the confounding-finding subsection appears — that subsection (Phase 09 step 5) appears in the thesis regardless of the accuracy-lever outcome, because it is true and useful either way. This phase's wording is specifically the closing/framing paragraph for chapter-06, conditioned on the L1-L7 lever outcome.
- Kill rules for L1-L7 already exist per-lever in `plan.md`; this phase does not re-litigate them. It only pre-commits what to *say* in the aggregate case where none of the surviving levers clears its bar.
- Sign-off requirement exists so that invoking the negative-result framing is never a unilateral drafting decision made under deadline pressure — it must be confirmed by the thesis author.

## Requirements
**Functional**
- Verbatim wording block, ready to paste into `report/chapter-06-conclusion.md` (and echoed, shortened, into `abstract.md`), for the case: no lever L1-L7 beats the E3 DeepSets LCO-full baseline (69.6% BA) with a bootstrap CI excluding zero.
- Decision rule stated precisely: check every surviving (non-killed) lever's paired-bootstrap CI vs baseline; if all include zero, invoke this wording; if any excludes zero, do not invoke (use the positive framing instead, still including the confounding-finding subsection).
- Decision date: **2026-10-04** (day before number freeze). Sign-off: **the user (thesis author)** — explicit go-ahead required before the wording is treated as final, not assumed from silence.
- Wording must not contradict anything Phase 10's citation/reasoning audit requires (e.g. must not repeat the Q6-style inverted-gap reasoning).

**Non-functional**
- Wording is honest, specific, falsifiable — no hedge language that could be read as overselling ("promising," "suggests strong potential") without a number attached.
- Stays inside Lane D/E globs: `report/chapter-06-conclusion.md`, `abstract.md` (Lane D, via Phase 09) for application; this phase file itself and any Lane E review of it stay in `plans/2026-09-19-*/`.

## Architecture
```
2026-10-04: decision date
     |
     v
Check L1-L7 surviving-lever CIs vs E3 baseline (from number-sheet.md, Phase 09)
     |
     +-- any lever CI excludes zero  ---> positive framing (still includes
     |                                     confounding-finding subsection,
     |                                     Phase 09 step 5, unconditionally)
     |
     +-- all levers CI include zero  ---> invoke wording below, pasted into
                                            chapter-06-conclusion.md + abstract.md,
                                            after sign-off from the user
```

## Related Code Files
| Type | Path | Why |
|---|---|---|
| modify | `report/chapter-06-conclusion.md` | destination for the verbatim wording if invoked (edit executed under Phase 09's ownership, this phase supplies the exact text) |
| modify | `report/abstract.md` | shortened echo of the same framing if invoked |
| create | none (this phase's own file is the plan document itself) | — |
| delete | none | — |

Note: this phase does not itself hold write access beyond planning — the actual paste-in happens as part of Phase 09's execution, using the text pre-committed here. Listed as "modify" because that is where the words land, not because Phase 11 is a separate write pass outside Phase 09/Lane D's glob.

## Implementation Steps
1. **(0.5h)** Confirm from `number-sheet.md` (once frozen) whether any surviving L1-L7 lever's paired-bootstrap CI vs E3 baseline excludes zero. This is a read of already-produced numbers, not a new computation.
2. **(0.5h)** If invoking: get explicit sign-off from the user before treating the wording as final for the thesis. Record the sign-off (date, method) in this phase file or a short note in `chapter-06-conclusion.md`'s revision history.
3. **(1h)** Paste the verbatim wording block (below) into `report/chapter-06-conclusion.md`, filling only the bracketed numeric placeholders from frozen `number-sheet.md` rows (e.g. the exact baseline BA, exact lever deltas and CIs) — no other prose changes to the pre-committed wording.
4. **(0.5h)** Paste the shortened echo into `abstract.md`.
5. **(0.5h) DROPPABLE** — if partial win (some lever's CI excludes zero but is small, e.g. <2pp), use the bridging paragraph (below) instead of the full negative-result block; droppable because it's an edge case, not the primary contingency.

### Pre-committed wording — full negative-result case (all L1-L7 CIs include zero)

> No accuracy lever tested in this work (L1-L7: zone repair, appearance upgrade, prior-correction and calibration, ensemble, detector A/B, background mixing, pretrained-VLM features) improved leave-camera-out balanced accuracy over the E3 DeepSets baseline ([BASELINE_BA]%) by a margin whose bootstrap confidence interval excludes zero. This is reported as a primary finding, not concealed or reframed.
>
> This thesis's contribution is threefold. First, a reproducible, pre-registered evaluation protocol for safety-critical binary video classification under domain shift — leave-camera-out with a declared panel-disclosed variant, per-cluster and worst-group reporting, prior-shift handling, and McNemar/Nadeau-Bengio significance testing — that goes beyond the aggregate accuracy reporting of the baseline paper (Önal & Dandıl 2024). Second, a mechanism-level, falsifiable finding: the leave-camera-out safe-recall collapse is not a uniform model failure but decomposes into four sub-tasks with a pattern that is invariant across six encoders spanning 18,044 to ~300,000 parameters **on this corpus**. That invariance is itself diagnostic rather than architectural: on a second corpus (iSafetyBench) the identical encoder ladder spans 5.1 pp and the largest encoder wins by 5.1 pp (p=7.74e-06), so flatness indicates an unlearnable task, not an insensitivity of the encoder family. The zone sub-task (145 clips) genuinely generalises across cameras (76-90% both classes) — a positive result showing the relation-token approach works where the data supports it. The carry sub-task (86 clips, 100%/100%) is disclosed as a trivial detector-class lookup, not claimed as generalization evidence. The panel sub-task (174 clips) is structurally unanswerable under leave-camera-out by construction — all 32 `panel_closed` clips occupy a single camera group, so 0% recall there is the protocol's expected mathematical output, not a model defect, and is reported separately (panel-disclosed variant) rather than hidden. The walkway sub-task (282 clips, the largest) is the one real, open, still-attackable failure (90%+ unsafe recall vs 15-36% safe recall), correctly isolating where future work should focus. Third, a measured, reproducible edge-deployment package (Jetson latency, power, memory across quantization levels) for E3 DeepSets — the smallest model in the sweep — establishing that the most cross-camera-robust architecture found here is also the cheapest to deploy. **This may no longer be stated as general evidence against "a bigger model would fix this" (R47)** - on iSafetyBench a bigger model demonstrably does help (+5.1 pp, p=7.74e-06). The defensible form is narrower: where the labels are camera-confounded, capacity buys nothing.
>
> Consistent with this thesis's stated objective order (accuracy, edge, robustness), we report a negative result against the accuracy objective and positive, falsifiable results against the evaluation-methodology and edge-deployment objectives. The ordering constrains what would have been the headline; it does not constrain what is disclosed.

### Bridging paragraph — partial-win case (some lever CI excludes zero, DROPPABLE default text, adapt per actual numbers)

> Lever [X] improved leave-camera-out balanced accuracy over the E3 DeepSets baseline by [DELTA]pp (95% CI [LOW, HIGH]pp, excluding zero). This is reported as the accuracy headline. The remaining levers did not clear this bar and are reported as tested-and-inconclusive, not silently dropped. The camera-label-confounding finding (sub-task decomposition, architecture-invariance) is reported unconditionally alongside this result, as it explains both why most levers failed to move the number and where the surviving lever's gain plausibly came from (state which sub-task[s] the surviving lever's gain concentrates in, if traceable from per-sub-class OOF recall).

## Todo List
- [ ] Read frozen `number-sheet.md`, determine invoke/no-invoke on decision date 2026-10-04
- [ ] Obtain user sign-off before treating wording as final (if invoking)
- [ ] Paste full-negative-result wording into `chapter-06-conclusion.md`, fill bracketed numbers only (if invoking)
- [ ] Paste shortened echo into `abstract.md` (if invoking)
- [ ] Use bridging paragraph instead if partial-win case applies (droppable, adapt numbers)
- [ ] Confirm Phase 10 review does not flag this wording for reasoning errors

## Success Criteria
- Wording exists verbatim, ready to paste, before 2026-10-04 — not drafted under pressure on the decision date itself.
- If invoked: sign-off recorded, bracketed placeholders filled only from frozen `number-sheet.md` rows, no other prose deviation from the pre-committed text.
- Passes Phase 10's adversarial review with no reasoning-inversion defect (the Q6 class of error) and no unsupported claim.
- Thesis conclusion never reads as unexplained failure — every negative claim is paired with the mechanism that explains it.

## Risk Assessment
| Risk | Mitigation |
|---|---|
| Wording drafted/rewritten under 10-04 deadline pressure, introduces a reasoning error | Text is fully pre-committed now (this phase), decision date only checks which branch applies — no live drafting under pressure |
| Sign-off skipped, wording goes in without user approval | Step 2 is a hard gate; Phase 10's review should also check for a recorded sign-off before treating chapter-06 as final |
| Partial-win bridging paragraph used sloppily (wrong lever's numbers) | Bracketed placeholders only, filled from `number-sheet.md` rows with cited log paths, same discipline as Phase 09 |
| Confounding-finding framing undercut elsewhere in the thesis by leftover pre-finding language (e.g. old "12+ tracks hit a ceiling, cause unknown" phrasing) | Phase 10's defect hunt should also grep for stale pre-finding framing in chapters 1/2/6 |

## Security Considerations
- No secrets or PII in scope. Wording references sub-task/camera-group identifiers only, no facility-identifying detail.
- Sign-off record (date/method) should not embed credentials or private contact detail beyond noting "confirmed by the user."

## Next Steps
- Depends on Phase 09's `number-sheet.md` being frozen (2026-10-05) to fill bracketed numbers — decision itself happens 2026-10-04 on lever CIs, which may be available slightly before full freeze if Phase 06 finishes early.
- Hands to Phase 10 for final adversarial pass before 10-10 defence package close.
- Unresolved: exact mechanism for recording "user sign-off" (a plans/ note vs a chapter-06 revision-history line) — pick whichever the lead prefers; not specified elsewhere in this plan.
