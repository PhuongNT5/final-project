---
title: "Phase 04 — Lane C: Thesis Writing (Vietnamese)"
status: pending
---

# Phase 04 — Thesis Writing, Vietnamese (Lane C)

**File ownership (exclusive):** `report/`. Concurrent with all lanes (writing/CPU-light); Ch4/Ch5
gated on phase-01 A4 / phase-03 B2.

## Context links
- Binding brief: `plans/2026-09-06-thesis-2week-final/brief.md` D1 (§0), §2 claim structure, §6
  deliverable 1
- Existing EN drafts to translate/rewrite: `report/chapter-{01..07}-*.md`, `report/abstract.md`,
  `report/thesis-reframe-deployment-led.md`
- PROPOSAL.txt — MT1/MT2/MT3 objective wording (Vietnamese), S01 Mẫu 2 structure hints

## Overview
Priority P1. Status pending. D1 locks thesis language = Vietnamese with English abstract. All
`report/*.md` (currently English) must be rewritten to VN per S01 Mẫu 2 structure. Six chapters +
EN abstract + assembled single document. This phase can start immediately on ungated chapters
(Ch1, Ch2, Ch3, Ch6-skeleton) while Ch4/Ch5 wait on experiment freezes.

## Key insights
- Substantial EN content already exists and is NOT throwaway — `report/chapter-01-introduction.md`
  through `chapter-07-*.md` contain the technical substance (architecture, methodology, prior
  round-0/1/2 results). Translation + reconciliation with the NEW brief claim structure (§2), not a
  from-scratch rewrite.
- `report/chapter-04-experiments-and-results.md` (69KB) and
  `chapter-04-section-kfold-cross-validation.md` and `chapter-07-multi-entity-and-scene-robust-
  evaluation.md` currently document the OLD P1/P2 numbers and framing (e.g. withdrawn 57/64
  calibrated, "+3.5pp macro win") — brief hard rule 2 says these negative results STAY in the
  thesis as methodological contribution, they are not deleted, just correctly framed as historical.
- `report/thesis-reframe-deployment-led.md` already argues a "deployment-led" framing similar in
  spirit to the brief's claim (b) — reuse its argument structure, update its numbers.
- The brief's claim structure (§2) is the organizing spine for Ch4: axis 1 (paper-protocol
  accuracy, statistical tie) -> axis 2 (faithful reproduction) -> axis 3 (statistical reliability,
  k-fold) -> axis 4 (LCO cross-camera) -> axis 5 (safety) -> axis 6 (calibration). Ch5 = axis 7
  (edge). Ch6/negative-results section = axis 8.

## Requirements
- Functional: Ch1-6 in Vietnamese, English abstract + keywords, references, single assembled doc.
- Non-functional: every accuracy/latency claim carries an interval or test statistic in the same
  sentence (brief hard rule 1); no test-set peeking language implied anywhere (hard rule 3); every
  failed run/control documented (hard rule 4, e.g. MPS bug, withdrawn 57/64).

## Architecture
Per-chapter VN files in `report/` (keep existing filenames, translate in place — do not create
`-vn` suffixed duplicates per dev-rules "do not create new enhanced files, update existing"), plus
`report/abstract.md` split into VN body + EN abstract section, then a final assembly pass
concatenating all chapters into one document (target: `report/thesis-assembled.md` or per
university template if S01 Mẫu 2 specifies a required file format — check with advisor/template if
unclear, flagged below).

## Related code files (report/ only — this lane writes docs, not code)
- Modify in place: `report/chapter-01-introduction.md`, `chapter-02-related-work.md`,
  `chapter-03-methodology.md`, `chapter-04-experiments-and-results.md`,
  `chapter-04-section-kfold-cross-validation.md`, `chapter-05-edge-deployment.md`,
  `chapter-06-conclusion.md`, `chapter-07-multi-entity-and-scene-robust-evaluation.md`,
  `abstract.md`, `thesis-reframe-deployment-led.md`
- Create: `report/thesis-assembled.md` (final concatenation, day 12)
- Read-only inputs: phase-01's `reports/test-touch-log.md` + eval_utils full_report outputs;
  phase-03's `deployment/artifacts/*.json`; PROPOSAL.txt for MT1-3 wording

## Implementation steps (per brief §7 schedule)
1. Day 1: outline Ch1 (Giới thiệu) + Ch2 (Tổng quan) in VN — NOT gated, can start immediately.
   Source: PROPOSAL.txt intro/motivation section (already VN) + `chapter-01-introduction.md`,
   `chapter-02-related-work.md` EN content for technical substance.
2. Day 3: draft Ch1, Ch2 in full VN.
3. Day 4: draft Ch3 (Phương pháp) — architecture description (YOLO backbone + relation-Transformer
   + fusion), sourced from `chapter-03-methodology.md` + phase-01's A3 architecture once finalized.
   Ch3 is NOT strictly gated (architecture is known before A3 finishes numbers) but final layer
   dimensions/config should be confirmed against phase-01's actual A3 run.
4. Day 7: draft Ch4 (Thực nghiệm & Kết quả) — **GATED on phase-01 A4 (day 6 headline freeze) and
   phase-02's `eval_utils.report.full_report()` output.** Structure Ch4 around brief §2's 6 axes
   table; include the withdrawn-number history (56/64->50/64 blind->new headline) as the
   methodological-rigor section per hard rule 2. Do not draft numeric claims before A4 lands.
5. Day 8: finalize Ch4 with A5's safety/calibration analysis (FNR, cost matrix, ECE, reliability
   diagram) folded in.
6. Day 9: draft Ch5 (Triển khai biên) — **GATED on phase-03 B2 (day 7 full Jetson metric set).**
   Source: `deployment/artifacts/benchmarks-jetson-*.json`, `soak-jetson-summary.json`,
   `multi-stream-*.json`, `power-sweep-summary.json` (real numbers only, per hard rule 5 — do not
   carry over `projections.md`'s literature rows unless clearly labelled as context).
7. Day 10: draft Ch6 (Kết luận) + English abstract — abstract can be drafted in skeleton form
   earlier (day 1-2) and finalized once Ch4/Ch5 numbers are locked.
8. Day 12: full VN assembly + consistency pass — check every number in Ch4/Ch5 matches the source
   JSON/log files exactly (no transcription drift), check terminology consistency (VN technical
   terms for "macro accuracy", "calibration", "leave-camera-out" etc. — establish a glossary early,
   day 1, to avoid inconsistent translation across chapters written on different days).

## Todo list
- [ ] Day 1-3: Ch1, Ch2 VN drafted (ungated, start immediately)
- [ ] Day 4: Ch3 VN drafted
- [ ] Day 1-2: VN technical-term glossary established (macro accuracy, calibration, LCO, etc.)
- [ ] Day 7-8: Ch4 VN drafted from phase-01 A4 + A5 output (gated)
- [ ] Day 9: Ch5 VN drafted from phase-03 B2 output (gated)
- [ ] Day 10: Ch6 + EN abstract drafted
- [ ] Day 12: full VN assembly, numeric consistency pass vs source artifacts
- [ ] Withdrawn numbers / MPS bug / negative results present in final text (hard rule 2 check)

## Success criteria
- All 6 `report/chapter-*.md` files are in Vietnamese (spot-check: no English body paragraphs
  remaining, headers may keep bracketed EN terms for clarity).
- `report/abstract.md` contains an English section, distinct from the VN body.
- `report/thesis-assembled.md` exists and concatenates all 6 chapters + abstract + references in
  order, day 12.
- Every accuracy number in Ch4 traces to a specific cell in phase-01's A4 `full_report()` output or
  `reports/test-touch-log.md`; every latency/power number in Ch5 traces to a specific
  `deployment/artifacts/*.json` field — spot-checkable, not paraphrased from memory.
- Ch4 contains the withdrawn "57/64 calibrated" and "+3.5pp macro win" as explicit named negative
  results (hard rule 2), and the N=64 ceiling proof (max McNemar p=0.50) is stated (brief §1).

## Risk assessment
- **Ch4/Ch5 gating slip**: if phase-01 A4 or phase-03 B2 slip past day 6/7, Ch4/Ch5 drafting slips
  1:1 — mitigate by using the buffer day (brief day 7 Lane A buffer) and continuing Ch1-3/Ch6-
  skeleton/glossary work in the meantime (never idle).
- **Terminology drift**: multiple chapters translate the same term differently (e.g. "macro
  accuracy" vs "độ chính xác trung bình theo lớp") — mitigate with the day-1 glossary (todo above).
- **Numeric transcription errors** copying from JSON/log files into prose — mitigate: day-12
  consistency pass explicitly re-opens every source artifact cited in Ch4/Ch5 and diffs against text.

## Security considerations
No secrets in thesis text. Confirm no internal file paths/credentials leak into the assembled
document (e.g. accidental `.env` reference, local absolute paths from artifact JSON `machine` field
like `phuongnguyen-desktop` — scrub or anonymize such fields before quoting JSON in the thesis body).

## Next steps
- Phase-06 (defence slides) is downstream of this phase's Ch4/Ch5 numbers — slides should not be
  finalized before Ch4/Ch5 text is numerically locked (day 8/9).
- Phase-05 (docs/README) cross-references thesis chapter numbers for reproducibility — coordinate
  chapter numbering before day 11.

## Unresolved questions
1. Exact required assembly format for S01 Mẫu 2 (single .docx via template? Markdown->PDF pipeline?
   university LaTeX template?) — not specified in the brief; assumed Markdown assembly here,
   flag for advisor/template confirmation before day 12.
2. Whether `chapter-04-section-kfold-cross-validation.md` and `chapter-07-multi-entity-and-scene-
   robust-evaluation.md` merge into the main Ch4 file or stay as separate referenced sections in
   the assembled document — brief doesn't specify; recommend merging into Ch4 proper since brief's
   claim axes 3/4 (k-fold, LCO) are core Ch4 content, not an appendix.
