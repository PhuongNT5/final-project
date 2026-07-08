# Advisor Alignment Brief — 1 page for the supervisor meeting

**Purpose:** get the supervisor's decision on framing + scope before committing the remaining month. The thesis is *printed in a deployment-led, honest-negative voice*; the defense narrative must match it, so this must be settled first.

## Where the thesis stands vs the approved proposal
| Objective | Promised | Delivered | Status |
|---|---|---|---|
| Obj1 — YOLO+RT-DETR hybrid | RT-DETR temporal reasoning | YOLOv11s + Transformer head; RT-DETR dropped on evidence (52–53 vs 55/64); temporal lift ≈ 0 | Partial (reframed) |
| Obj2 — accuracy **> 95.8%** | > 95.8% | Blind **≈52/64 ≈81%** top-1; best-case 56/64 = 87.5% | **Missed** (honest negative) |
| Obj3 — TensorRT/INT8 multi-stream Jetson | edge deployment | 8.62 ms/116 FPS FP16; N=4 zero-drop; 7 W soak — all measured | **Exceeded** |

## The honest accuracy reality (important for the supervisor)
- On the **N=64** test set, anything below **60/64 is statistically indistinguishable** from the current 56/64 (bootstrap ±2.6 clips). So the proposal's **95.8% (=62/64) is, in practice, out of statistical reach** on this dataset.
- The accuracy gap is **causally isolated** to missing per-frame *zone-boundary* annotations (WalkwayViol↔UnauthorInt cosine 0.834; human κ only 0.65 on that pair; an 86M VideoMAE hits the same ceiling). This is a defensible scientific finding, not a coding failure.

## The decision — two paths
- **Path A — Defend the negative result (low risk, ready now).** Lead with Obj3 (the measured edge system) + the data-bound finding. Document already supports it. Work = fix 17 errors + Jetson re-measurement + rehearsal.
- **Path B — Attempt a positive bump (added ~1 week, gated).** Path A **plus** a *corrected* experiment. Note: our V10 pilot that "tried bbox" used the **wrong mechanism** (feature-extractor + full unfreeze) — the paper's YOLO is actually a **frame gate/filter**. We already have **1,600 annotated frames on disk**, so the corrected frame-gate experiment costs **~1 week, no new annotation**, and is hard-gated (stop if < 58/64). A month-long 4,000-frame annotation is **not recommended** — structurally unlikely and can't clear the N=64 bar.

## My recommendation
**Do Path A in full + the Path B Step-1 corrected experiment (1 week, existing data), and do NOT commit to month-scale new annotation.** Rationale: Step 1 can only help — either a modest accuracy bump *or* a stronger, cleaner negative finding (a 400→1,600-frame mechanism+scaling result that replaces our single weak V10 data point). The guaranteed de-riskers are the 17 fixes + the real Jetson N=4 soak, not annotation.

## Decisions I need from you
1. **Approve the deployment-led / honest-negative framing** as the defense narrative? (The printed thesis already commits to it.)
2. **Approve ~1 week on the corrected frame-gate experiment** (existing 1,600 frames), with a hard stop at < 58/64 — and decline the month-long 4,000-frame annotation?
3. **Abstract accuracy number:** report both with labels, blind **≈52/64** as the primary scientific claim, 56/64 as the deployed-checkpoint upper bound? (current thesis posture)
4. **Email Önal & Dandıl now** for their original annotation files? (Either helps the experiment or documents good-faith effort — neutralizes a likely committee question.)

*Supporting detail: `defense-strategy.md` (full Q&A), `c1-bbox-feasibility.md` (the experiment), `execution-roadmap.md` (schedule), `must-fix-checklist.md` (the 17 fixes).*
