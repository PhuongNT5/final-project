# Phase 11 — Slides + Defence Q&A (DEFERRED)

> **Trạng thái: CHƯA KÍCH HOẠT — user sẽ yêu cầu sau nếu cần.**
> Not on the critical path. Consumes **no** scheduled time. Do not start this phase unless the user
> explicitly activates it.

## Context Links

- `docs/thesis-number-sheet.md`, `report/vn/*.txt`.

## Overview

- **Priority:** deferred. **Status:** not activated. **Effort:** unscheduled (~8h if activated).

## Scope if activated

1. Defence slides in Vietnamese, plain text outline first (no styling work — same rationale as the .txt
   thesis: the user formats personally).
2. Q&A preparation targeting the known attack surface:
   - "Is binary Safe/Unsafe just camera identification?" → scene-appearance control (S1) + LCO (S4).
   - "Why is the panel pair's co-clustering only 0.12?" → L1, stated as a limitation, with the
     descriptor-confound caveat.
   - "Is CarryOverload's 100% recall real?" → L2, one-line detector rule.
   - "Why only one test evaluation?" → eval contract §4h, pre-registration.
   - "Does it run in real time?" → phase-05 measured curve; imgsz-512 candidate; the 800px configuration
     does not meet 3.0 dec/s.
   - "Why redesign the encoder?" → S7 ablations (PE factorisation, slot association).
3. A 3-minute live or recorded demo using the Jetson footage captured on 19/9.

## Requirements if activated

Numbers cited by sheet ID only (G7/G8 discipline applies to slides too).

## Next Steps

None. Await user activation.
