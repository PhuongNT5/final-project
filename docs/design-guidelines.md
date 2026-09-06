# Design Guidelines

This project has no end-user UI beyond an operator-facing monitoring dashboard. Scope of this file
is minimal by design (YAGNI) — the project has no design system, brand guidelines, or component
library to document.

## Streamlit dashboard (`deployment/dashboard/app.py`)

Operator-facing view for the live demo (camera -> edge node -> server -> dashboard, per
`PROPOSAL.txt`'s architecture). Built with Streamlit — no custom CSS/theming layer found in the
codebase; follow Streamlit's default component conventions when extending it.

## Live monitor (`deployment/dashboard/jetson-live-monitor.py`)

Alert-loop script, not a visual UI: polls the edge node and raises alerts on unsafe-behaviour
detections. Currently implements 2-tier confidence thresholds with no debounce/refractory timer
(see `plans/2026-09-06-thesis-2week-final/plan.md` phase-03 for the planned N-of-M debounce
addition — not yet implemented as of this writing).

## Conventions for any new UI work

- Keep operator-facing text in Vietnamese where the thesis and defence audience are Vietnamese
  (matches the thesis-language decision in `plans/2026-09-06-thesis-2week-final/brief.md` §0 D1);
  keep code identifiers and comments in English, matching the rest of the codebase.
- No raw video frames leave the edge device over the network — metadata-only egress is a stated
  privacy/security property of the demo (per `README.md`); any new dashboard feature that adds
  network calls must preserve this.

## Unresolved questions

1. Whether the live demo's UX flow (beyond the current dashboard) will be extended before the
   defence is not specified in any plan file read for this pass — flagged as open, not addressed
   here per phase-05's own recommendation to keep this file minimal given the 2-week budget.
