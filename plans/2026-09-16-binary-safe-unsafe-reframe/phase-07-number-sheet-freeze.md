# Phase 07 — Number-Sheet Freeze

## Context Links

- `reports/test-touch-log.md`
- `plans/2026-09-06-thesis-2week-final/reports/laneb-2026-09-16-paired-significance-fix.md` (**landed**)
- `plans/2026-09-06-thesis-2week-final/reports/laneb-2026-09-16-binary-dedupe-and-ceiling.md` (incoming)
- Phase 05 `deployment/results/2026-09-19/`, phase 06 result dirs, phase 12 ceilings.
- `brief.md` S2/S3/S11/S17/S19 (withdrawn and corrected claims).

## Overview

- **Priority:** P1. **Status:** pending. **Effort:** 5h.
- Build the **single number sheet** that is the ONLY source of numbers for lane C. After G7, lane C may not
  read a result JSON, a log, or an English chapter for any figure.

## Key Insights

- Every historical numeric error here came from a number travelling between documents without an ID
  (S2 +60.3pp, S9 silent overwrite, the withdrawn 57/64). An ID-addressed, **generated** sheet is the fix.
- **Significance claims changed materially.** The paired-significance fix replaced a two-proportion z-test
  (which assumes *independent* samples) applied to two classifiers scored on the **same clips**. Result:
  ~~**neither previously-significant comparison survives as confirmed.**~~ **CORRECTED 2026-09-29:
  both DO survive.** The bound column below was computed by an enumerator that constrained only
  `b + c <= n`, not each side's wrong-count, so it admitted infeasible 2×2 tables and inflated the
  upper bound. Fixed in `unsafenet/metrics_stats.py` + `eval_utils/paired_tests.py`, regression test
  added; see `plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-09-29-mcnemar-bound-marginal-constraint.md`.

  | Comparison | Old z-test | Bound as published (LOOSE — withdrawn) | **Corrected bound** |
  |---|---|---|---|
  | S1a ensemble 54/64 vs paper no-bbox 57/64 | p=0.435 | [0.250, 0.801] | **[0.250, 0.629]** |
  | S1a ensemble 54/64 vs paper full-bbox 62/64 | **p=0.015 "significant"** | [0.008, 0.382] | **[0.008, 0.039] — significant at every split** |
  | hybrid 51/64 vs no-bbox 57/64 | p=0.144 | [0.031, 0.532] | **[0.031, 0.263]** |
  | hybrid 51/64 vs paper full-bbox 62/64 | **p=0.003 "significant"** | [0.001, 0.207] | **[0.001, 0.007] — significant at every split** |
  | blind 54/64 vs no-bbox 57/64 | p=0.435 | [0.250, 0.801] | **[0.250, 0.629]** |
  | gated 53/64 vs no-bbox 57/64 | p=0.309 | [0.125, 0.708] | **[0.125, 0.481]** |

  The two comparisons that the loose bound "rescued" were both findings that **we are worse than the
  paper's full-bbox pipeline**, i.e. the error ran in the direction that flattered this thesis. Under
  the corrected bound both deficits are significant at α = 0.05 under every feasible split. Nothing
  changes against the no-bbox baseline.

  ~~No comparison flips the other way — the fix only removes false positives, the anti-conservative
  direction the earlier statistics review predicted. The true p is **unknown** and plausibly as high as
  0.38 / 0.21.~~ **Withdrawn 2026-09-29.** 0.38 / 0.21 were the artefacts of the unconstrained
  enumerator; the true p is bounded above by **0.039 / 0.007**, so it is not "plausibly non-significant"
  at all. Two comparisons did flip back — the z-test→McNemar switch removed no false positive here.
- Ranges must be stored as ranges: the LCO margin is **+13.3 to +16.9 pp**, not a point estimate (S19).

## Requirements

One file: `docs/thesis-number-sheet.md`. Each row:
`ID | value (+CI/range) | protocol | seeds | source command | commit | touch? | policy | status`.

**Every comparison row must additionally carry `b`, `c` (the discordant counts), the exact p OR the bound
interval, and the paired CI. A bare p-value is not acceptable. A z-test p-value is never acceptable.**

## Architecture

ID namespaces: `BIN-*` binary headline · `SCN-*` scene control (S2) · `RULE-*` detector-rule gate (S1) ·
`LCO-*` · `KF-*` · `SUB-*` per-sub-class · `EDGE-*` Jetson · `CEIL-*` ceilings · `CMP-*` paired comparisons ·
`LEG-*` legacy 4-class component results · `WDR-*` withdrawn.

## Related Code Files

Create: `docs/thesis-number-sheet.md`, `tools/collect_numbers.py` (scrape result JSONs → rows, so the sheet
is **generated, not hand-typed**). Modify: `reports/test-touch-log.md`.

## Implementation Steps

1. Write `tools/collect_numbers.py`; emit rows from phase 05/06/12 artifacts.
2. Populate `BIN/SCN/RULE/LCO/KF/SUB/EDGE` rows.
3. **`CMP-*` rows:** for every paired comparison emit `b`, `c`, exact p (when we own both prediction sets)
   or the bound interval (when comparing to a published aggregate), plus the paired CI.
4. **`CEIL-*` rows:** `CEIL-HARD` 124/125 (99.2%), `CEIL-PRACTICAL` ≈121/125 (96.8%),
   `CONTRA-REMOVED-BY-BINARY` ≈28 of 32, `DEDUPE-POLICY` (declared policy + date).
5. `LEG-*` rows for the kept component results (C5): blind 54/64, k-fold 516 = 80.8±4.0%, LCO macro 81.2%
   vs 64.4% scene-appearance baseline (+16.9pp, CI [+12.5,+21.2]), YOLO11-vs-YOLO26, P1 TRT FP16 8.62 ms.
   **Mark every one "historical context — not a binary baseline row"** (equal-recipe rule).
6. `WDR-*` rows with reasons: +60.3pp macro (S2); "57/64 calibrated"; "+3.5pp macro win"; "8-class ~78%"
   (unsourced, likely a miscopy of "78% of clips collapse onto CarryOverload"); the YOLO26 Jetson INT8
   characterisation claimed at `abstract.md:36` but never measured; **and every z-test-derived significance
   claim**. WITHDRAWN rows exist so the VN thesis cannot resurrect them.
7. Pre-registered disclosure rows: `SUB-PANEL-CONFOUND`, `SUB-CARRY-RULE-ARTIFACT`,
   `BIN-SCOPE-BINARY-NOT-8WAY`, `CMP-NO-PAIRED-TEST-VS-PAPER`.
8. Freeze: append the commit hash; mark rows `frozen`. **G7.**

## Todo List

- [ ] `tools/collect_numbers.py` written
- [ ] Generated rows populated
- [ ] **`CMP-*` rows carry b, c, exact-p-or-bound, paired CI**
- [ ] `CEIL-*` + dedupe policy rows
- [ ] `LEG-*` marked historical-context-only
- [ ] `WDR-*` incl. all z-test significance claims
- [ ] Disclosure rows
- [ ] **G7 frozen + commit hash**

## Success Criteria

Every number the VN thesis will cite has an ID and a reproducible source command; no comparison row lacks
b/c and an interval; withdrawn claims present and marked.

## Risk Assessment

R12 — drift; mitigated by generation-not-typing. Post-G7 changes need an explicit un-freeze note plus a
re-run of the G8 trace check.

## Security Considerations

Each row records test-touch status; a `touch=1` row absent from the test-touch log is a contract violation
and blocks the freeze.

## Next Steps

G7 unblocks lane C (phase 08) and the final review (phase 10).
