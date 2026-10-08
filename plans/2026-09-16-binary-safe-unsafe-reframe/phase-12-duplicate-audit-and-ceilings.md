# Phase 12 — Duplicate Audit, Dedupe Policy, Ceilings

> **Scheduling: runs immediately after phase 01, BEFORE any encoder training and BEFORE the pre-registered
> test touches.** The phase number is high only because the issue was found late; its place in the critical
> path is early.

## Context Links

- **Incoming (dedicated agent):** `plans/2026-09-06-thesis-2week-final/reports/laneb-2026-09-16-binary-dedupe-and-ceiling.md`
  — recomputes the audit for binary/691 and extends the policy. **Do not duplicate that work.**
- `unsafenet/dedupe.py` — **existing machinery; reuse, do not reinvent.**
- `design-2026-09-16-binary-encoder-redesign.md` §4.5; `sweep_results/duplicate_audit.json`.
- `brief.md` S11 (dedupe never applied to the headline), S15 (cross-clip box repetition).

## Overview

- **Priority:** P1 — contamination silently caps every headline number. **Status:** verified; policy pending.
- **Effort:** 3h (mostly policy + reporting; the scan is done).

## Key Insights — measured and independently verified

- **691 clips → 658 unique hashes, 32 duplicate groups, 33 extra copies.** Confirmed twice (design report
  and an independent md5 pass). Spot-check reproduced here: `0_te23.mp4` and `4_te9.mp4` both hash
  `bef275463d146ad04cf9df6e00585119`.
- Classified by **binary** label and split:
  - **22 groups** — same binary label, same split → benign for the binary task.
  - **6 groups** — cross train/test, same binary label → leakage without contradiction.
  - **4 groups — OPPOSITE binary labels:**
    - `0_te23.mp4` ≡ `4_te9.mp4`, **both in TEST** → **hard ceiling 124/125 (99.2%)**. Any deterministic
      classifier is necessarily wrong on exactly one of them.
    - `4_te4` ≡ `0_tr123` ≡ `1_tr55`; `5_te13` ≡ `0_tr136`; `4_te7` ≡ `0_tr138` → three TEST clips
      byte-identical to TRAIN clips of the **opposite** binary label → **practical ceiling ≈ 121/125 (96.8%)**
      for any model that fits its training data.
- **Favourable nuance — state it with exact counts, do not overstate.** The old 4-class audit found all 32
  groups label-contradictory, of which **26/32 were the pair `0↔1`** (walkway violation ↔ unauthorized
  intervention). Both are UNSAFE, so **under the binary reframe those contradictions disappear**: the binary
  framing removes roughly **28 of 32** label contradictions, leaving 4. This is a genuine argument *in
  favour* of the reframe and belongs in the thesis — as a counted fact, not a rhetorical flourish.
- **What is genuinely new:** the old policy was "dedupe train only, freeze the test split at 64 unsafe
  clips". That no longer suffices, because the contradiction is **no longer confined to train** — three test
  clips are byte-identical to train clips of the opposite binary label.
- Note the asymmetry: all 32 groups remain contradictory for the **8-way aux head** (`4↔6` safe-safe,
  `1↔2` unsafe-unsafe — harmless binary, harmful to aux).

## Requirements

(a) binary/691 audit recomputed (agent, in flight); (b) a **dated, written policy fixed before any test
touch**; (c) both ceilings printed in **every** binary results table; (d) headline reported under a declared
policy **with the contaminated number shown beside it**.

## Architecture — reuse the existing machinery

`unsafenet/dedupe.py` already implements drop-lists and keep-masks with policies `none` (default) /
`keep_one` / `drop_all`, already wired into `run_kfold_relation.py`, `run_lco_eval.py`,
`run_relation_transformer.py`, `unsafenet/scene_confound.py`, `run_representation_ladder.py`, with OOF
filenames suffixed by policy. **Its governing principles are already agreed and carry into this plan:**

1. **Never delete source files** — filter at load time.
2. **Always report BOTH the contaminated and the deduped number.**
3. **Default `none`**, so historical commands reproduce.

The extension needed is a **test-side contradiction policy**, not a new framework.

## Related Code Files

Modify: `unsafenet/dedupe.py` (extend to binary-label contradictions spanning the test split),
phase-06 reporting helpers (ceiling row), `docs/thesis-number-sheet.md`.
Create: `sweep_results/duplicate_audit_691.json` (agent). Delete: **nothing** — principle 1.

## Implementation Steps

1. Ingest the agent's report when it lands; reconcile against the verified counts above. Recomputed numbers
   win over quoted ones.
2. Extend `unsafenet/dedupe.py` with the binary contradiction classes, preserving the three principles and
   the existing policy names.
3. **Decide and document the policy, dated, BEFORE the test touches.** Recommended:
   headline = existing canonical (`keep_one` train / `keep` test) **plus** a `drop_contradictory`
   sensitivity row reported beside it.
   **Excluding contaminated clips from the declared test set after the fact is a protocol violation** — the
   test set was declared before this was known. Footnote + sensitivity row, never silent removal.
4. Quantify the aux-head impact (all 32 groups contradictory for 8-way): either caveat the aux numbers or
   restrict aux evaluation to non-duplicated clips; state which.
5. Wire both ceilings into the results-table template so they cannot be omitted.
6. Add `CEIL-HARD`, `CEIL-PRACTICAL`, `DEDUPE-POLICY`, `CONTRA-REMOVED-BY-BINARY` rows to the number sheet.

## Todo List

- [ ] Agent report ingested and reconciled
- [ ] `unsafenet/dedupe.py` extended for test-side binary contradictions
- [ ] **Policy decided, dated, written — before any test touch**
- [ ] Aux-head contamination quantified
- [ ] Ceilings wired into every results table
- [ ] Sheet rows added (incl. the 28-of-32 favourable count)

## Success Criteria

Policy is written and dated before the first test touch; every binary table shows both ceilings and both
(contaminated / deduped) numbers; no source file deleted.

## Risk Assessment

R10 — a 96%-accurate model looks like failure against a naive 100% expectation; printing ceilings prevents
both false disappointment and false triumph. **R16** — the temptation to drop contaminated test clips to
raise the headline: explicitly forbidden here, and the policy's date stamp is the audit trail.

## Security Considerations

No source deletion. The policy is a reporting decision, not a data operation.

## Next Steps

Blocks phase 06 headline reporting and phase 07 sheet rows. Feeds a limitation **and** the favourable
reframe argument in phase 08.
