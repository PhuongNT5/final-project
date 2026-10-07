# Phase 09: Number Sheet + Thesis Integration

## Context Links
- Master plan: `plan.md` (lanes, gates, calendar, EV levers, test-touch discipline)
- `research/researcher-02-evaluation-protocol-and-metric-suite.md` — frozen table layout §7.1, appendix structure §7.3, prior-shift handling §3, defence Q&A §6
- `reports/critique-2026-09-19-research-factcheck.md` — **FATAL**: researcher-02 §7.1's numeric table is not verified binary-task data (audit: 0/2 test touches spent as of 2026-09-19); its Q6 answer is logically inverted. Do not copy either verbatim.
- `reports/audit-2026-09-19-repo-capability-and-feasibility.md` §6, §9 — test-touch ledger, binary head status
- `reports/finding-2026-09-19-detector-starvation-root-cause.md` — detector exonerated, mAP50 0.966, retrain cut
- `reports/finding-2026-09-19-lco-failure-is-camera-label-confounding.md` — panel sub-task structurally unanswerable under LCO; walkway is the real open failure; zone sub-task genuinely generalizes
- `docs/system-architecture.md` — "read this file before citing any number in this repo" (P1/P2/P3 provenance rule this phase must obey)
- Existing report files: `report/abstract.md`, `chapter-01..07`, `chapter-04-section-kfold-cross-validation.md`
- Depends on: Phase 06 (frozen metric suite + statistics, gate G5, freeze 2026-10-05)

## Overview
- **Priority:** P1 (blocks G8, feeds Phase 10/11)
- **Status:** pending — starts 2026-09-20 (day 0), placeholder phase runs through freeze
- **Brief:** Build and maintain `number-sheet.md`, the single traceable source for every headline number in the thesis; propagate frozen numbers into `report/**`, `README.md`, `PROGRESS.md`, `docs/**` only after NUMBER FREEZE 2026-10-05.

## Key Insights
- **The thesis headline is no longer a tuned accuracy number.** `finding-2026-09-19-lco-failure-is-camera-label-confounding.md` decomposes the LCO safe-recall collapse into four sub-tasks: zone (145 clips, genuinely generalises both sides, 76-90%), carry (86, 100%/100% but a trivial detector-class lookup — MUST be disclosed as trivial, not claimed as a win), walkway (282, the real open failure, 90% unsafe vs 15-36% safe), panel (174, structurally unanswerable under LCO — all 32 `panel_closed` clips sit in one camera group, so 0% recall there is the mathematically expected outcome of the protocol, not a model defect). The failure is **architecture-invariant on Unsafe-Net** across six encoders spanning 18k-300k params (NOT a general property - **R47 scope (2026-09-21):** architecture invariance holds **on Unsafe-Net only**. On iSafetyBench capacity is decisive (E0 - E3 = +5.1 pp, cluster CI [+2.0, +8.3], McNemar p=7.74e-06). Invariance is a symptom of an unlearnable, camera-confounded task - not a property of the encoder family. Every sentence below that says "architecture-invariant" means "architecture-invariant on this corpus" and must be written that way in the thesis.). This is the strongest result in the project — a falsifiable, mechanism-level claim backed by an architecture sweep plus a dataset cross-tab — and gets a dedicated chapter-07 subsection, not a footnote.
- **Encoder architecture search is cut, not merely deprioritized.** E3 DeepSets (18,044 params) is frozen as the headline and deployment model (best under both full-LCO and panel-disclosed variants, smallest, fastest). No E0-vs-E1 decision rule, no width/depth sweeps get reported as live work in the thesis — report the sweep as the evidence that *closed* the question, not as ongoing exploration.
- **Panel-disclosed LCO is a first-class declared protocol variant, not a replacement.** Reported alongside the full LCO number always, per pre-registration rule R8 (post-hoc exclusion forbidden) — it is declared, structurally justified (single-camera safe class), computed on the full 691 pool, 0 test touches. `number-sheet.md` must carry protocol as its own column value (`LCO-full` vs `LCO-panel-disclosed`) so the two numbers can never be silently swapped or confused in prose.
- Two structural risks already found in this repo's own research artifacts, both must NOT recur here: (1) a results table presented as frozen/real when the underlying test touches were never spent (researcher-02 §7.1); (2) a defence-answer reasoning error caught only by re-deriving the arithmetic (Q6, 22pp vs 14pp — see Phase 10). The number sheet's job is to make (1) structurally impossible — no number enters `report/**` without a sheet row citing its log path.
- Lane D cannot compute anything — it only transcribes what Lane B (eval) and Lane C (edge) produce. Placeholder-first drafting exists so writing isn't blocked on Lane B/C's 09-26→10-03 timeline.
- 4-class numbers (P1/P2 pipelines, baseline paper's own ~97% train-curve figure) are historical/context only — thesis narrative arc, never a row in the binary headline comparison table. Mixing recipes here is exactly the "4-class/binary recipe mixing" defect Phase 10 is tasked to hunt.
- `docs/system-architecture.md`'s own stated purpose ("read before citing any number") means this phase must keep it in sync whenever P3's headline numbers change — it is the disambiguation layer between P1/P2/P3 pipelines.

## Requirements
**Functional**
- `number-sheet.md` has one row per headline number: metric, protocol (P-B k-fold / P-C LCO-full / P-C LCO-panel-disclosed / scene control), token tag, producing script, log/artifact path, date, test-touch cost, status (`placeholder`/`frozen`). Protocol is its own column — `LCO-full` and `LCO-panel-disclosed` never share a row.
- Every number appearing in `report/**` chapters 4, 6, 7 and `abstract.md` after 2026-10-05 traces to exactly one sheet row.
- Frozen metric suite present on every headline number: balanced accuracy, macro-F1, MCC, Cohen's kappa, per-class recall, PR-AUC, ROC-AUC, Brier, ECE, cost @ 10:1 unsafe-miss, operating point @ 0.95 unsafe recall, BCa bootstrap CI. Paired own-vs-own comparisons: McNemar exact only (never `two_proportion_z` — 3 scripts in this repo still misuse it per audit §8, unresolved; do not import that error into the thesis text). K-fold means: Nadeau-Bengio corrected t.
- Equal-recipe rule enforced: any table comparing "old" vs "new" results uses the same task (binary vs binary, never binary vs legacy-4-class-collapsed) and same dedupe policy.

**Non-functional**
- No number written into `report/**` before 2026-10-05 may be marked anything but a placeholder token (e.g. `TBD`, `[PLACEHOLDER]`) — never a plausible-looking fake number.
- Every edit stays inside Lane D's glob: `report/**`, `docs/**`, `README.md`, `PROGRESS.md`, `plans/2026-09-19-*/number-sheet.md`. No edits to `eval_utils/**`, `run_*.py`, `model_*.py` (Lane B) even to "just check a number."

## Architecture
```
Lane B (eval)              Lane C (edge)             Lane D (this phase)
measured JSON/log   --->   measurement JSON   --->   number-sheet.md (ledger)
per phase 05/06                per phase 07/08              |
                                                              v
                                                report/chapter-04, -07, abstract.md,
                                                README.md, PROGRESS.md
                                                              |
                                                              v
                                                docs/system-architecture.md,
                                                docs/project-roadmap.md (sync)
```
Hand-off contract: B->D and C->D are by filename tag / JSON path only, never verbal recall of a number. D never re-derives a metric — it transcribes and cites.

## Related Code Files
| Type | Path | Why |
|---|---|---|
| create | `plans/2026-09-19-thesis-recovery-architecture/number-sheet.md` | single source of truth ledger (this phase's core deliverable) |
| modify | `report/chapter-04-experiments-and-results.md` | headline binary results table, placeholder then frozen |
| modify | `report/chapter-04-section-kfold-cross-validation.md` | k-fold-691 section, Nadeau-Bengio results |
| modify | `report/chapter-07-multi-entity-and-scene-robust-evaluation.md` | LCO-691, per-cluster/worst-group, scene control, gap headline |
| modify | `report/abstract.md` | headline claim sentence, hedge until freeze |
| modify | `report/chapter-06-conclusion.md` | contribution framing, coordinate with Phase 11 |
| modify | `report/chapter-01-introduction.md` | any early numeric claims (e.g. legacy "8-class ~78%" citation flagged unresolved by audit — hedge or remove) |
| modify | `README.md` | top-line accuracy table |
| modify | `PROGRESS.md` | status/progress numbers |
| modify | `docs/system-architecture.md` | P3 section headline numbers, keep pipeline-attribution rule intact |
| modify | `docs/project-roadmap.md` | phase status per documentation-management triggers |
| delete | none | — |

## Implementation Steps
1. **(1h)** Inventory every numeric claim currently in `report/**`, `README.md`, `docs/system-architecture.md`; tag each as P1/P2/P3, 4-class/binary, measured/placeholder. Flag the audit's open item ("8-class ~78% top-1, no findable source") for removal or explicit caveat.
2. **(2h)** Create `number-sheet.md` skeleton: columns `metric | protocol | value | 95% CI/range | token_tag | script | log_path | date | test_touch_cost | status`, where `protocol` distinguishes `P-B k-fold`, `P-C LCO-full`, `P-C LCO-panel-disclosed`, `Scene Control`. Seed one placeholder row per frozen-suite metric x each protocol, plus a clearly separate "Historical (4-class, context only)" section that is never joined into the binary table.
3. **(2h)** Draft `report/chapter-04-experiments-and-results.md` and `chapter-07` headline tables referencing sheet row IDs / `[PLACEHOLDER]` tokens, not literal numbers. Table layout follows researcher-02 §7.1's column structure (P-B / P-C / Scene Control) with an added `LCO-panel-disclosed` column shown beside (never instead of) `LCO-full`; every cell is a placeholder until step 6.
4. **(1h) DROPPABLE** — hedge `abstract.md` and `chapter-01-introduction.md` headline sentences to future tense / bracketed placeholder; low priority, defer if time-boxed, must finish before freeze regardless.
5. **(2h)** Write a dedicated chapter-07 subsection for the camera-label-confounding finding: the four-sub-task decomposition table (zone/carry/walkway/panel, n, verdict), the architecture-invariance claim, **scoped to Unsafe-Net** (six encoders, 18k-300k params, identical pass/fail pattern) and explicitly contrasted with iSafetyBench where the same ladder spans 5.1 pp and the largest encoder wins by +5.1 pp (p=7.74e-06), and the panel-structural-unanswerability argument (32/32 `panel_closed` clips in one camera group => 0% recall there is the protocol's expected output, not a defect). State plainly that carry (100%/100%) is a trivial detector-class lookup, not a generalization win. State plainly that encoder search is closed — E3 DeepSets is frozen as headline/deployment model, the sweep is reported as the evidence that closed the question.
6. **(4h, gated on Phase 06 G5, earliest 2026-10-05)** NUMBER FREEZE: pull Lane B's measured JSON/logs and Lane C's measurement JSON; fill every `number-sheet.md` row (both `LCO-full` and `LCO-panel-disclosed` rows), flip status `placeholder`->`frozen`. Verify each figure against its log path before writing it down (open the file, grep the line, do not trust memory).
7. **(4h)** Propagate frozen numbers from `number-sheet.md` into `chapter-04`, `chapter-07` (including the confounding subsection and panel-disclosed column), `abstract.md`, `chapter-06-conclusion.md`, `README.md` headline table, `PROGRESS.md`. Every propagated number carries an inline citation to its sheet row (footnote or parenthetical).
8. **(2h)** Sync `docs/system-architecture.md` P3 section and `docs/project-roadmap.md` phase status per `documentation-management.md` triggers (phase status change, feature landed).
9. **(1h)** Self-trace pass: grep every number in `report/**` chapters 4/6/7 + abstract, confirm each maps to a `number-sheet.md` row, and confirm no sentence conflates `LCO-full` with `LCO-panel-disclosed`. This is a pre-check only — Phase 10's independent trace audit is the real gate, not a substitute for it.
10. **(1h) DROPPABLE** — add appendix stubs A-J (per researcher-02 §7.3) as empty headers in `chapter-07` or a new appendix section, to be filled opportunistically; skip if 10-06 writing week is tight.

## Todo List
- [ ] Inventory existing numeric claims, tag provenance (P1/P2/P3, task, measured/placeholder)
- [ ] Create `number-sheet.md` skeleton with frozen-suite rows + historical-4-class section
- [ ] Draft placeholder tables in `chapter-04`, `chapter-07`
- [ ] Hedge `abstract.md`/`chapter-01` headline claims (droppable, but finish pre-freeze)
- [ ] Write per-cluster/worst-group/panel-disclosed table shell
- [ ] Write dedicated chapter-07 confounding-finding subsection (four-sub-task decomposition, architecture-invariance, panel unanswerability, carry triviality, E3 frozen)
- [ ] NUMBER FREEZE: fill all sheet rows from Lane B/C artifacts (gated 2026-10-05)
- [ ] Propagate frozen numbers into report chapters, README, PROGRESS
- [ ] Sync `docs/system-architecture.md`, `docs/project-roadmap.md`
- [ ] Self-trace every report number against the sheet
- [ ] Appendix stubs A-J (droppable)

## Success Criteria
- `number-sheet.md` exists, every frozen-suite metric row present for P-B / P-C LCO-full / P-C LCO-panel-disclosed / Scene Control, no orphan numbers.
- Zero numbers in `report/**` after 2026-10-05 without a traceable sheet row (Phase 10's trace audit is the verification method — this phase's own self-trace is necessary but not sufficient).
- No 4-class number appears inside a binary-task comparison table.
- Chapter-07 confounding subsection present, states carry's triviality and panel's structural unanswerability explicitly, does not claim encoder search as ongoing.
- `docs/system-architecture.md` and `docs/project-roadmap.md` consistent with frozen numbers (gate G8 condition).

## Risk Assessment
| Risk | Mitigation |
|---|---|
| Placeholder number accidentally looks real (e.g. a plausible percentage left in a draft) | Use unmistakable tokens (`[PLACEHOLDER]`, `TBD`) never a numeric-looking stand-in; grep for stray digits before freeze |
| Repeating researcher-02's §7.1 error (presenting unmeasured binary numbers as frozen) | Every propagated number requires a `number-sheet.md` row with a real log path checked open, not copied from a research report |
| Freeze date slips because Lane B/C run late | Phase 09 has no compute dependency until step 6 — steps 1-5, 9-10 proceed regardless; only steps 6-8 block on G5 |
| Docs drift from report (system-architecture.md not updated) | Step 8 explicit, tied to documentation-management.md triggers |

## Security Considerations
- No credentials/secrets touched. `number-sheet.md` and report files contain no PII — clip filenames/camera IDs are internal identifiers only, verify none embed real-world facility names before any external sharing.
- Do not commit `.pt`/model weight files while editing docs (out of Lane D's glob regardless; confirm `.gitignore` still excludes them per current modified `.gitignore` in git status).

## Next Steps
- Feeds Phase 10 (adversarial review reads `number-sheet.md` + `report/**` for the trace audit).
- Feeds Phase 11 (contingency wording lands in `chapter-06-conclusion.md`, owned here).
- Unresolved: whether `docs/development-roadmap.md` / `docs/project-changelog.md` (named in `documentation-management.md` but absent from current `docs/`) should be created — defer to lead; not required for G8.
