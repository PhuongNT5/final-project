---
title: "Phase 06 — Lane E: Defence Slides & Q&A Brief"
status: pending
---

# Phase 06 — Defence Slides & Q&A (Lane E)

**File ownership (exclusive):** `plans/2026-09-06-thesis-2week-final/slides/`, `figures/`. Gated on
phase-01 A4 (headline freeze) and phase-03 B2 (Jetson metrics).

## Context links
- Binding brief: `plans/2026-09-06-thesis-2week-final/brief.md` §5 (Lane E), §6 deliverable 2
- `figures/` (existing directory, currently populated from prior rounds — needs regeneration once
  headline changes)
- `deployment/scripts/generate-figures.py` — existing figure-generation script (M4/P1 pipeline)
- `plans/2026-09-05-thesis-final-push/reports/t1-objectives-audit.md`,
  `plans/2026-09-02-fair-headtohead-vs-paper/reports/fair-headtohead-scorecard.md` — source of the
  hardest anticipated questions (statistical power, why not beat 62/64, scene confound)

## Overview
Priority P2. Status pending. Vietnamese slides + a Q&A brief where every anticipated committee
question maps to a specific file and a specific number — no answer without a traceable source.
Hard-gated on day 6 (headline) and day 7 (Jetson metrics); cannot be finalized before both land.

## Key insights
- The hardest anticipated question is baked into the brief itself: "why didn't you beat 62/64?" —
  answer already fully derived in the audit (§3b): even a perfect 64/64 cannot reach p<0.05 against
  the paper (McNemar best-case p=0.50, Fisher p=0.496) because the paper is only 2 clips from
  perfect. This is a ceiling proof, not a shortfall — the Q&A brief must state this precisely, not
  hand-wave it.
- Second-hardest: "is your k-fold/LCO evidence comparable to the paper's 64-clip number?" — answer:
  different N (516 vs 64), different question (statistical reliability / cross-camera
  generalisation vs a single point estimate) — the brief's axis table (§2) is the direct answer.
- `deployment/scripts/generate-figures.py` exists and already generates M4/P1 figures — likely
  needs re-pointing at the new headline + new Jetson artifacts (phase-03), not rewriting from
  scratch.
- `figures/` currently has content from the OLD pipeline (see git status: many
  `runs/detect/sweep_results/finetune_bbox/yolo_finetuned/*.png` deletions already staged — this
  directory is mid-cleanup; verify what remains before assuming any figure is current).

## Requirements
- Functional: VN slide deck covering problem/motivation/architecture/results/deployment/
  conclusion; Q&A brief with >=1 file+number citation per anticipated question.
- Non-functional: every number on a slide matches phase-04's Ch4/Ch5 text exactly (no drift between
  slides and thesis body).

## Architecture
`plans/2026-09-06-thesis-2week-final/slides/` (deck source, format TBD — Markdown->reveal.js,
PowerPoint, or Google Slides per advisor preference, unresolved below) + `figures/` (regenerated
plots: confusion matrix, ROC/reliability diagram, LCO bar chart, Jetson latency/power charts) +
a `qa-brief.md` mapping table (question -> source file -> number).

## Related code files
- Create: `plans/2026-09-06-thesis-2week-final/slides/outline.md` (VN slide-by-slide outline, then
  final deck in whatever format is confirmed)
- Create: `plans/2026-09-06-thesis-2week-final/slides/qa-brief.md`
- Modify/run: `deployment/scripts/generate-figures.py` (re-point at new headline data +
  phase-03's new Jetson artifacts)
- Regenerate into: `figures/` (confusion matrix, reliability diagram, LCO comparison bar chart,
  ROC/PR curves, Jetson latency/power plots)
- Read-only inputs: phase-01's `reports/test-touch-log.md` + full_report outputs; phase-03's
  `deployment/artifacts/*.json`; phase-04's `report/chapter-04-*.md`, `chapter-05-*.md`

## Implementation steps (per brief §7 schedule, day 6-13)
1. Day 6: figures regenerated for whatever A4/A5 produced that day (confusion matrix, ROC/PR,
   reliability diagram) — `deployment/scripts/generate-figures.py` adapted to read the new headline
   logits path.
2. Day 6-9 (parallel, non-gated): draft slide outline structure (title, motivation, related work,
   architecture, methodology) — VN, mirrors Ch1-3.
3. Day 9: once phase-03 B2 lands (Jetson metrics), regenerate Jetson figures (latency percentile
   chart, power/FPS chart, multi-stream throughput chart).
4. Day 13: finalize full VN slide deck — results slides sourced verbatim from phase-04's finalized
   Ch4/Ch5 numbers (no independent recomputation, copy the locked numbers).
5. Day 13: build `qa-brief.md` — draft the anticipated-question list from: (a) the audit's own
   "why not beat 62/64" ceiling proof, (b) brief §2's 8-axis table (each axis is a likely question),
   (c) methodology questions about the MPS bug / withdrawn numbers (hard rule 2 material). For each
   question, cite the exact source file + number (e.g. "Q: why not 62/64? -> A: [ceiling proof] ->
   `plans/2026-09-05-thesis-final-push/reports/t1-objectives-audit.md` §3b, McNemar best-case
   p=0.50").
6. Day 13: code review of any generation scripts touched (`generate-figures.py` changes) per
   primary-workflow.md.

## Todo list
- [ ] Day 6: figures regenerated from A4/A5 headline data
- [ ] Day 6-9: VN slide outline for motivation/related-work/architecture/methodology drafted
- [ ] Day 9: Jetson figures regenerated from phase-03 B2 data
- [ ] Day 13: full slide deck finalized, numbers verified against phase-04's locked Ch4/Ch5 text
- [ ] Day 13: `qa-brief.md` complete, every question has a file+number citation
- [ ] Day 13: "why not beat 62/64" ceiling proof present verbatim in the Q&A brief
- [ ] Gate 3 (phase-03) outcome reflected in the demo slide (live vs recorded, stated explicitly)

## Success criteria
- `plans/2026-09-06-thesis-2week-final/slides/qa-brief.md` exists with >= 8 questions (one per
  brief §2 axis at minimum), each with an explicit `file:line-or-section` citation and a number.
- Every numeric claim on a results slide is byte-identical (or CI-consistent) with the corresponding
  claim in `report/chapter-04-experiments-and-results.md` / `chapter-05-edge-deployment.md` after
  phase-04 locks them — spot-checkable side by side.
- Demo slide states explicitly "live demo" or "recorded fallback" per phase-03's Gate 3 outcome —
  never ambiguous.
- Figures in `figures/` are all dated after phase-01's day-6 freeze (no stale pre-freeze plots
  presented as current).

## Risk assessment
- **Late Gate 2/3 landing (day 6/10)** compresses slide-finalization to day 11-13 — mitigate by
  front-loading everything gate-independent (outline, motivation, methodology slides) in the first
  9 days, leaving only results/deployment/demo slides for the compressed window.
- **Figure/number drift** if slides are drafted from intermediate (non-final) numbers before
  phase-04 locks Ch4/Ch5 — mitigate by explicit day-13 cross-check step (todo above), not trusting
  earlier drafts.
- **Hard committee question not covered**: the ceiling proof (McNemar p=0.50 at best) is
  counterintuitive and may draw follow-up ("so no method could ever win here?") — mitigate by
  including the full derivation logic (not just the number) in `qa-brief.md`, not a one-liner.

## Security considerations
No secrets. Confirm no internal absolute file paths or hostnames appear on public-facing slides
(e.g. `phuongnguyen-desktop` from Jetson JSON metadata).

## Next steps
- Slides are the final consumer in the dependency chain (phase-01 -> phase-02 -> phase-04/03 ->
  phase-06) — no downstream phase depends on this one; day-14 buffer covers any last slide fix.

## Unresolved questions
1. Slide deck output format (Markdown/reveal.js vs PowerPoint/Google Slides) not specified by the
   brief — assumed Markdown outline first, final format decided by advisor/defence-venue
   requirement; flag before day 13 to avoid a late format-conversion scramble.
2. Whether the Q&A brief itself is presented to the committee or kept as private prep material —
   affects whether it needs polishing beyond internal clarity. Assumed private prep (brief says
   "Q&A brief", not "handout") — confirm if wrong.
