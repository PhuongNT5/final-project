---
title: "Phase 05 — Lane D: Reproducibility & Docs"
status: pending
---

# Phase 05 — Reproducibility & Docs (Lane D)

**File ownership (exclusive):** `docs/`, `README.md`, `scripts/`. Concurrent with all lanes
(writing/CPU-light).

## Context links
- Binding brief: `plans/2026-09-06-thesis-2week-final/brief.md` §5 (Lane D), §6 deliverable 3
- `.claude/rules/documentation-management.md` — required `docs/` structure (7-file convention)
- `README.md` (current, describes P1 pipeline as headline — needs rewrite once phase-01 freezes
  the new headline)
- `deployment/README.md`, `kaggle-detector-retrain/README.md` — existing good examples of
  repro-oriented docs to model the new `docs/` files on

## Overview
Priority P2. Status pending. `docs/` directory does not exist yet in this repo — this is creation,
not an update. Populate the 7 standard files, rewrite `README.md` to reflect the new (phase-01)
headline, add one-command repro scripts per headline number, pin the environment, and pass a code
review before day 13.

## Key insights
- `docs/` is genuinely empty (`ls docs` -> No such file or directory) — every one of the 7 files
  named in `.claude/rules/documentation-management.md` is new work, not a refresh.
- The repo already has strong per-subsystem READMEs (`deployment/README.md` has a full "Reproduce"
  section with exact commands; `kaggle-detector-retrain/README.md` has a clear run recipe) — the
  new `docs/` files should link to / summarize these, not duplicate their content (DRY).
- `README.md`'s current "Key Results" table cites P1 numbers (56/64, 89.28% macro) as the headline
  and frames the project as "deployment-led" around that pipeline — this must be rewritten once
  phase-01 freezes the new P3-based headline, and the frozen-COCO/no-bbox framing needs updating to
  reflect the fixed-detector (A1) story.
- No `scripts/` directory currently exists at repo root (all repro is via root `run_*.py` files +
  subsystem scripts) — "one-command repro per headline number" likely means small wrapper scripts
  in a new `scripts/` dir chaining the existing `run_*.py`/`eval_utils` calls, not new pipelines.

## Requirements
- Functional: `docs/` populated (7 files per the standard structure); README rewritten around the
  new headline; one wrapper command per headline number in brief §2's evidence table; pinned
  environment (requirements/lockfile); code review passed.
- Non-functional: no confidential info committed (dev-rules); repro commands must actually run
  against files that exist in this repo (verified, not aspirational).

## Architecture
`docs/project-overview-pdr.md` (objectives, MT1-3 mapping) · `docs/code-standards.md` (existing
project conventions: kebab-case, 200-line cap, try/catch, etc. — largely already stated in
`.claude/rules/development-rules.md`, distilled here for repo-external readers) ·
`docs/codebase-summary.md` (map of `unsafenet/`, `eval_utils/`, `deployment/`, root `run_*.py`
scripts, `kaggle-detector-retrain/`) · `docs/design-guidelines.md` (N/A or minimal — no UI beyond
the Streamlit dashboard, document dashboard conventions) · `docs/deployment-guide.md` (Jetson
setup, superseding/linking `deployment/JETSON-DEPLOYMENT.md` once phase-03 corrects it) ·
`docs/system-architecture.md` (YOLO->relation-Transformer->fusion pipeline, data flow, matches
PROPOSAL.txt architecture diagram) · `docs/project-roadmap.md` (this 14-day plan's phase status,
links to `plans/2026-09-06-thesis-2week-final/plan.md`). New `scripts/repro-<headline-name>.sh`
per major number (e.g. `scripts/repro-kfold-headline.sh`, `scripts/repro-jetson-benchmark.sh`).

## Related code files
- Create: `docs/project-overview-pdr.md`, `docs/code-standards.md`, `docs/codebase-summary.md`,
  `docs/design-guidelines.md`, `docs/deployment-guide.md`, `docs/system-architecture.md`,
  `docs/project-roadmap.md`
- Modify: `README.md` (full rewrite of "Key Results"/"Architecture" sections around new headline)
- Create: `scripts/` directory with one-command repro wrappers, e.g.
  `scripts/repro-kfold-lco.sh` (wraps `run_kfold_relation.py` + `run_lco_eval.py` with the frozen
  A3 tag), `scripts/repro-jetson-metrics.sh` (wraps phase-03's B2 commands),
  `scripts/repro-full-eval.sh` (wraps `eval_utils.report.full_report` over the headline logits)
- Create: environment pin — `requirements.txt` or `pyproject.toml` lockfile (check if one already
  exists at repo root before creating; `.venv/` present suggests pip-based, verify with `pip freeze`
  against what's actually installed)
- Read-only: `deployment/README.md`, `kaggle-detector-retrain/README.md` (style/content reference)

## Implementation steps (per brief §7 schedule, day 5-13)
1. Day 5: `docs/` skeleton — create all 7 files with headers + TOC stubs, no content yet (unblocks
   parallel writing by section).
2. Day 5-10 (parallel, non-gated): fill `docs/code-standards.md`, `docs/design-guidelines.md`,
   `docs/project-overview-pdr.md` — these don't depend on phase-01/03 results.
3. Day 11: `docs/system-architecture.md`, `docs/codebase-summary.md` finalized (need phase-01's A3
   final architecture to be accurate); `docs/deployment-guide.md` finalized (needs phase-03's
   corrected `JETSON-DEPLOYMENT.md`).
4. Day 11: `README.md` full rewrite — new headline numbers (from phase-01 A4), new architecture
   diagram description, corrected Jetson claims (from phase-03).
5. Day 11: write `scripts/repro-*.sh` wrappers — verify each one actually runs against the frozen
   headline artifacts (not aspirational commands); pin `requirements.txt`/lockfile via
   `pip freeze > requirements-lock.txt` (or update existing lockfile if present).
6. Day 11: `docs/project-roadmap.md` — snapshot of this plan's phase statuses at time of writing.
7. Day 13: code review pass — delegate to `code-reviewer` agent per primary-workflow.md step 3,
   covering all lane deliverables (phase-01 through phase-03's code changes) for syntax errors,
   security issues (no committed secrets/credentials), and basic quality.

## Todo list
- [ ] `docs/` directory created with all 7 files
- [ ] `docs/code-standards.md`, `design-guidelines.md`, `project-overview-pdr.md` filled (ungated)
- [ ] `docs/system-architecture.md`, `codebase-summary.md`, `deployment-guide.md` filled (gated on
      phase-01/03)
- [ ] `README.md` rewritten with new headline + corrected Jetson claims
- [ ] `scripts/repro-*.sh` created and each one verified to run end-to-end
- [ ] Environment pinned (`requirements.txt`/lockfile committed)
- [ ] `docs/project-roadmap.md` reflects final phase statuses
- [ ] Code review passed (via `code-reviewer` agent), findings addressed

## Success criteria
- `docs/` contains exactly the 7 files named in `.claude/rules/documentation-management.md`, each
  non-empty.
- Every headline number in brief §2's evidence table has a corresponding one-command repro script
  in `scripts/` that runs without error and reproduces the number (or a documented CI-tolerance
  match, given seeds).
- `README.md`'s headline table matches phase-01's `reports/test-touch-log.md` exactly (no stale P1
  numbers left as the primary claim).
- `pip install -r requirements.txt` (or lockfile equivalent) succeeds in a clean venv.
- Code review report exists (delegate output) with all flagged issues either fixed or explicitly
  deferred with reason.

## Risk assessment
- **Docs finalization gated on phase-01/03**: `system-architecture.md`/`deployment-guide.md` can't
  be finalized before day 6/7 — mitigate by drafting skeleton + placeholder sections day 5, filling
  numbers day 11 only (small, bounded task).
- **Repro scripts rot** if phase-01/03 change file/tag names after day 11 — mitigate by writing
  scripts LAST (day 11, after freezes), not early.
- **No existing environment lockfile** — if none found, creating one late risks missing a
  dependency; mitigate by running `pip freeze` against the actual working `.venv` used throughout
  the 2 weeks, not a fresh install.

## Security considerations
- Scrub local absolute paths (e.g. `/Users/nguyenthiphuong/...`) and machine hostnames from
  committed docs/scripts where feasible; use relative paths / `$REPO_ROOT` placeholders.
- Verify no `.env`, API keys (Kaggle/Colab tokens), or credentials are included when creating
  `scripts/` or referencing `kaggle-detector-retrain/upload/`.

## Next steps
- `docs/project-roadmap.md` should link back to `plans/2026-09-06-thesis-2week-final/plan.md` as
  the authoritative phase tracker, not duplicate its content.
- Phase-06 (defence Q&A brief) references `docs/` and `scripts/` for "every question maps to a file
  and a number" — coordinate final paths before day 13.

## Unresolved questions
1. Does an environment lockfile already exist somewhere untracked (e.g. `.venv/pyvenv.cfg` only,
   no `requirements.txt` at all)? Needs a repo-root check before day 11 to know create-vs-update.
2. Should `docs/design-guidelines.md` cover only the Streamlit dashboard, or also the (currently
   undocumented) demo UX flow end-to-end? Brief doesn't specify depth; recommend minimal (dashboard
   conventions only) given YAGNI and the 2-week budget.
