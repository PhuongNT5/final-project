# Phase 10 — Final Consistency Review

## Context Links

- `docs/thesis-number-sheet.md` (G7), `report/vn/*.txt` (phase 08), `reports/test-touch-log.md`.

## Overview

- **Priority:** P1 — last gate before submission. **Status:** pending. **Effort:** 5h.
- Verify that every claim in the Vietnamese thesis is traceable, contract-compliant and internally consistent.
  **No EN/VN cross-check exists** — the English thesis was dropped.

## Key Insights

- The check is mechanical where possible: extract every numeral from `report/vn/*.txt` and require each to
  match a frozen sheet row. Human judgement is reserved for claim *strength* (hedging), which is where this
  project has historically failed (S2, S3).

## Requirements

G8: every number traces to a sheet ID. Plus: no withdrawn claim present; exactly one test touch logged for
the binary variant; every "significant" claim backed by a paired test; binary framing consistent.

## Related Code Files

Create: `tools/check_number_traceability.py`. Modify: `report/vn/*.txt` (fixes only).

## Implementation Steps

1. **Automated trace.** Extract numerals from `report/vn/*.txt`; match against frozen sheet rows; report
   orphans. Orphan count must be 0 (excluding an allow-list: years, equation constants, clip counts).
2. **Strike-list grep.** Assert absence of: "78%" 8-class claim, "+60.3", "57/64", "+3.5pp",
   "marginally significant" / "gần như có ý nghĩa", the unmeasured YOLO26 INT8 claim.
2b. **Significance audit.** Assert **no bare p-value** appears against a published-baseline comparison, and
   **no z-test-derived p-value appears anywhere**. Every comparison must show `b`, `c`, and either an exact
   p (both prediction sets owned) or a bound interval (published aggregate). Cross-check each against its
   `CMP-*` sheet row.
2c. **Ceiling audit.** Assert the hard (124/125) and practical (≈121/125) ceilings appear beside every
   binary accuracy table, and that the declared dedupe policy is named with its date.
3. **Limitation presence.** Assert L1-L6 (phase 08) are present in chương 6 — especially **L1 panel confond**
   and **L5 binary scope**.
4. **Protocol pairing.** Assert the scene-appearance control (S1), random-split k-fold (S3) and LCO (S4)
   appear side by side wherever the headline is stated.
5. **Sub-class table.** Assert all 8 sub-classes appear with recall, including the panel pair.
6. **Contract audit.** Cross-check `reports/test-touch-log.md`: exactly one declared touch for the binary
   pipeline; the Jetson session logged as zero-touch; no selection-on-test anywhere.
7. **Claim-strength read-through.** Manual pass for over-claiming; every causal claim needs a control.
8. **Terminology pass.** VN technical vocabulary consistent against `report/vn/thuat-ngu.txt`.

## Todo List

- [ ] `tools/check_number_traceability.py` written; orphan count 0 (G8)
- [ ] Strike-list grep clean
- [ ] Significance audit: no bare/z-test p-values; every comparison has b, c + interval
- [ ] Ceiling audit: both ceilings + dated dedupe policy present
- [ ] L1-L8 present
- [ ] S1/S3/S4 reported side by side
- [ ] All 8 sub-class recalls present
- [ ] Test-touch audit passes
- [ ] Claim-strength read-through done
- [ ] Terminology consistent

## Success Criteria

G8 green and all seven checks above pass. Deliverable handed to the user for template formatting.

## Risk Assessment

R12/R13/R14 all terminate here. If an orphan number is found late, prefer deleting the claim over inventing
provenance.

## Security Considerations

Final check for footage stills, device IPs or credentials in the appendix.

## Next Steps

Submission-ready `report/vn/*.txt`. Phase 11 only if the user activates it.
