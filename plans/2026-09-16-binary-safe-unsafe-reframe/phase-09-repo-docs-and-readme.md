# Phase 09 — README + docs/

## Context Links

- `README.md` (root), `docs/system-architecture.md`, `deployment/JETSON-DEPLOYMENT.md`
- `plans/2026-09-06-thesis-2week-final/reports/audit-2026-09-14-experiment-inventory-and-feasibility.md`

## Overview

- **Priority:** P3 — droppable if the deadline tightens. **Status:** pending. **Effort:** 6h.
- Make the repo match the binary reframe and stop pointing at files that do not exist.

## Key Insights

- The README references **12 files that do not exist**: `run_finetune_e2e.py`, `model_arch.py`,
  `feature_dataset.py`, `temporal_video_loader.py`, `tools/preextract_yolo_features.py`,
  `tools/scan_bad_videos.py`, `GAP_ANALYSIS.md`, `run_safe_unsafe.py`, `run_temporal_fusion.py`,
  `run_scenario_cascade.py`, `plot_figures_binary.py`, `BASELINE.md` — while the scripts actually in use
  are unmentioned. A reader (or examiner) cannot reproduce anything from it today.
- README currently headlines the **P1 56/64 (87.5%)** figure, which is superseded (blind = 54/64) and is
  4-class, not binary.

## Requirements

README reflects: the binary objective, the real script inventory, the 691-clip data layout, how to
reproduce each phase-06 experiment, and the eval contract. `docs/` updated per documentation-management.

## Related Code Files

Modify: `README.md`, `docs/system-architecture.md`, `docs/codebase-summary.md`,
`docs/project-changelog.md`, `docs/development-roadmap.md`, `deployment/JETSON-DEPLOYMENT.md`.

## Implementation Steps

0. **Remove the raw two-proportion-z significance line (56/64 vs 57/64).** The z-test assumes independent
   samples and was applied to two classifiers scored on the same clips; the corrected result is a McNemar
   bound, not a p-value. Replace with the `CMP-*` sheet row (b, c, bound interval) or delete the claim.
   *(Assigned here as lane D per the paired-significance fix hand-off.)*
1. Inventory the real top-level scripts; write the true command list.
2. Remove/replace all 12 dead references.
3. Replace the headline block with binary numbers **cited by number-sheet ID** (after G7).
4. Document the `bin8_` token artifacts and the `*_691` maps.
5. Update `docs/system-architecture.md` with the redesigned encoder.
6. Changelog entry: binary reframe, encoder redesign, defect fixes, Jetson session.
7. Add a reproduction section: one command per phase-06 experiment.

## Todo List

- [ ] **Two-proportion-z significance line removed**
- [ ] 12 dead references removed
- [ ] Real script inventory documented
- [ ] Binary headline (sheet IDs) replaces the 56/64 block
- [ ] `docs/system-architecture.md` updated
- [ ] Changelog + roadmap updated
- [ ] Reproduction section added

## Success Criteria

Every path mentioned in README resolves on disk (grep-verified). No superseded headline number remains.

## Risk Assessment

Low. R12 — README numbers must come from the sheet, same as the thesis.

## Security Considerations

No credentials, dataset paths or device IPs in public docs.

## Next Steps

Optional input to phase 10's trace check.
