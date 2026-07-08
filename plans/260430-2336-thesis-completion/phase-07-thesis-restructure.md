---
priority: must-have
effort: 6 person-days
depends_on: [phase-01, phase-02, phase-03, phase-06]
branches: none (content adapts based on which phases completed; phase-04 and phase-05 deferred)
---

# Phase 07 — Thesis Restructure

## Context Links

- `report/chapter-01-introduction.md` — contributions list; §1.3 objectives; §1.5 scope
- `report/chapter-04-experiments-and-results.md` — full chapter; V3→V10 ledger; §4.5 gap analysis
- `report/chapter-05-edge-deployment.md` — §5.6 projection to be replaced by Phase 01 results
- `PROGRESS.md` — all numbers; use as source of truth over chapter drafts
- `BASELINE.md` — paper table numbers (Table 4, 5, 6, 7)
- `/tmp/unsafe-net-paper.txt` — cite specific section numbers when updating fair-comparison text

## Overview

This phase makes concrete edits to every existing chapter and drafts two new chapters. The primary goals are: (1) lock in the Pareto headline framing throughout; (2) replace the "we tried to beat 95.81 %" narrative with "we established a new Pareto point"; (3) integrate Phase 01–06 results; (4) add a Discussion chapter (Ch6) covering ethics/limitations/Hawthorne critique; (5) add a Conclusions+Future Work chapter (Ch7). No new experiments. Work from the chapter drafts; do not rewrite from scratch.

## Key Insights

- **Step 0 (numerical audit) must run before any prose changes.** See GROUND_TRUTH.md §7 for the list of wrong values. Run the grep commands first; do not rely on memory.
- **Corrected headline numbers:** CPU-FP32 = **28.8 ms p50 / 34.7 FPS** (Ch 5 Table 5.2). Do NOT use benchmarks.md's earlier run (26.5 ms / 37.7 FPS). The old plan.md also inherited this error — already corrected in plan.md revision.
- **INT8-slower-than-FP32 finding:** CPU-INT8 = 32.0 ms vs CPU-FP32 = 28.8 ms. Chapter 5 already explains this correctly. PROGRESS.md does not mention it. Ch 6 Discussion and Ch 7 Future Work must reference this as a legitimate engineering finding, not a defect.
- **Wrong values in PROGRESS.md:** "1.39 ms" and "13.6 MB" must be corrected before submission. Phase 07 Step 0 is the fix point.
- **Critical framing fix**: contributions list must be reordered — Pareto point first, architecture second.
- **§4.5 gap analysis**: reframe from "we could not reach 95.81 %" to "the 6.5 pp gap is annotation-volume-bounded."
- **Ch5 §5.6**: replace Jetson projection with Phase 01 measured table. If Phase 01 incomplete, label as "Projection (Jetson pending)."
- **New Ch6 (Discussion)**: Hawthorne critique of paper's 75 % claim; honest INT8 discussion; deployment limitations.
- **New Ch7 (Conclusions + Future Work)**: pseudo-bbox and cross-dataset listed as future work citing `.deferred` phase files.

## Requirements

**Functional**
- All chapter files exist: Ch1–Ch5 revised, Ch6 and Ch7 drafted.
- Ch1 §1.4 contributions list reordered with Pareto headline first.
- Ch1 new fair-comparison table (before or after §1.4).
- Ch1 §1.3 Objective 2 updated to cite 85.78 % as primary comparison.
- Ch4 §4.2 experiment ledger updated with CI column (from Phase 06).
- Ch4 §4.5 gap analysis reframed as a finding.
- Ch4 §4.7 added (Phase 04 result — pseudo-bbox or negative result).
- Ch4 §4.8 added (Phase 05 cross-dataset table, if complete).
- Ch4 §4.9 added (Phase 06 statistical significance).
- Ch5 §5.5 updated with live-system numbers (Phase 02).
- Ch5 §5.6 updated with Jetson measured table or updated projection (Phase 01).
- Ch5 new §5.x Energy Efficiency (Jetson W per stream vs. paper's desktop GPU).
- Ch6 drafted: ethics, false-alarm cost, deployment limitations, threats to validity, Hawthorne critique.
- Ch7 drafted: conclusions, future work (annotation campaign, behavior-class detector finetune, multi-camera spatial fusion, online learning).

**Non-functional**
- Each chapter ≤ 30 pages (thesis norm for Vietnamese CS master's).
- All numerical claims match PROGRESS.md or verified paper text; no invented numbers.
- Consistent figure numbering throughout.

## Architecture / Approach

### Chapter 1 Edit Plan

**§1.3 Objective 2** — change "measure accuracy against the published baseline (YOLO v4 + ConvLSTM, 95.81% top-1)" to:
> "Train, fine-tune, and evaluate the YOLO–Transformer on Unsafe-Net. The primary accuracy comparison is against the ConvLSTM-only baseline (85.78 %, §3 Table 7) — which uses no behavior-class bounding boxes, matching our annotation-free setting. The full Unsafe-Net cascade (95.81 %) uses 2,262 manually-labeled behavior-class bbox images (Table 3, §2.1) and is analyzed separately in §4.6 to quantify the annotation cost of that gap."

**§1.4 Contributions** — reorder:
1. (NEW) Pareto headline — accuracy/annotation/latency tradeoff.
2. (WAS 1) Architecture.
3. (WAS 2) Empirical study.
4. (WAS 3) Accuracy result — updated to cite 89.28 % vs. 85.78 % as headline.
5. (WAS 4) Gap analysis — updated as "data-efficiency finding."
6. (WAS 5) Edge deployment — updated with Phase 01/02 measured numbers.
7. (WAS 6) Open-source artifacts.

**§1.5 Scope** — update Jetson language from "projected" to "measured" if Phase 01 complete.

**NEW TABLE after §1.4**: fair-comparison table.

| Metric | Our YOLO-Transformer | Paper ConvLSTM-only (§3 Table 7) | Paper Unsafe-Net full cascade (Table 7) |
|---|---|---|---|
| Macro accuracy | **89.28 %** | 85.78 % | **95.81 %** |
| Behavior-bbox annotations required | **0** | 0 | **2,262** |
| Inference latency (per clip) | **28.8 ms p50 (M4 CPU-FP32)** | 0.14 s avg (§3 paper) | 0.14 s avg (§3 paper) |
| Model size (INT8 ensemble) | **≈ 40 MB** | Not reported | Not reported |
| Edge deployable | **Yes (ONNX)** | Not reported | Not reported |
| OpenedPanel accuracy | **100 %** | 76.92 % | 92.31 % |

### Chapter 4 Edit Plan

- §4.2: add "Mean ± CI" column to experiment ledger (from Phase 06 data).
- §4.5: rewrite opening sentence from "we could not reach 95.81 %" to "the persistent 6.5 pp gap is annotation-volume-bounded; evidence follows."
- §4.6 (current V10): rename to "§4.6 V10 — Behaviour-Class Bbox Finetune (Negative Result and Data-Efficiency Analysis)"; integrate Phase 03 curve figure; add extrapolation sentence.
- §4.7 (new): Weakly-Supervised Pseudo-Bbox (Phase 04 result — Branch A or B).
- §4.8 (new): Cross-Dataset Generalization (Phase 05, if complete). If not complete: 1-sentence placeholder "deferred to future work."
- §4.9 (new): Statistical Significance (Phase 06 ablation table + bootstrap results).

### Chapter 5 Edit Plan

- §5.5: Add RTSP input description, alert latency table (P50/P95/P99), 30-min soak result.
- §5.6: Replace projection with Phase 01 measured table. If Phase 01 incomplete: add "(Projection)" to heading; add sentence "Direct Jetson measurement is deferred pending hardware access; projected targets remain as stated."
- §5.x NEW: Energy Efficiency. Table: hardware / TDP (W) / streams / FPS / W per stream-FPS. Compare paper's dual GTX 1080 + i9-9900 (≈ 400 W total) to Jetson Orin Nano (5–10 W) to M4 CPU (≈ 15 W).

### Chapter 6 — Discussion (new, ≤ 8 pages)

Sections:
1. **Ethical considerations**: factory surveillance consent requirements; data retention; false-alarm harm; bias (system may perform differently across demographic groups in factory footage — not tested here, should be stated as a limitation).
2. **False-alarm cost analysis**: current best ensemble achieves 56/64 on unsafe subset, but safe-class accuracy is not evaluated in this work. A false alarm triggers unnecessary shutdown; cost model: 1 false alarm = X minutes of production loss. State as quantification of deployment risk.
3. **Deployment limitations**: single-factory dataset; fixed camera positions; no adaptation to new factories without retraining; bright/dark/occlusion robustness untested.
4. **Threats to validity**: test set n=64 (wide CIs); Phase 06 bootstrap shows 1-clip difference is not statistically significant at α=0.05; Jetson numbers may vary by JetPack version.
5. **Hawthorne Effect critique**: the paper's §4 claims "75 % reduction in recurrence of unsafe behaviour" after system installation. Per §4 of paper, this is a behavioral/compliance effect from audible alarms alerting workers — not a model quality result. The thesis must state this explicitly to distinguish model accuracy from system behavioral impact.
5b. **INT8 performance finding**: CPU-INT8 (32.0 ms p50) is slower than CPU-FP32 (28.8 ms p50) on M4. Root cause: ORT's dequantize kernels dominate; M4 AMX units lack native INT8 path in current ORT build. This is documented in Ch 5 §5.4.1. Ch 6 frames it as: "INT8 ≠ free on all hardware; runtime and hardware co-design is required for quantization to deliver speedup." This finding motivates the INT8 future work item above.
6. **Comparison fairness**: re-state the apples-to-apples framing from §1.4 fair-comparison table.

### Chapter 7 — Conclusions + Future Work (new, ≤ 5 pages)

Sections:
1. **Summary of contributions** (1 page): echo §1.4 reordered list; state the Pareto point in 2 sentences.
2. **Research questions answered**: (Q1) YOLO-Transformer feasible on edge? Yes, **28.8 ms CPU-FP32 p50 / 34.7 FPS** (Ch 5 Table 5.2). (Q2) Annotation-free approach competitive? Yes, +3.5 pp vs. ConvLSTM-only at 0 annotations. (Q3) Gap to 95.81 % bridgeable? Evidence says data-volume-bounded.
3. **Future Work**:
   - **Annotation campaign**: Phase 03 curve extrapolates to ≈ X % at 2,262 bboxes. A 6-month effort to label the remaining ≈ 1,861 frames (2,262 − 401) is the clearest path to matching paper accuracy. Estimated 2,262 images ≈ 60 annotation-hours at 38 images/hour.
   - **Behavior-class detector fine-tuning at paper scale**: replicate the paper's YOLO v4 training recipe with YOLOv11 at full 2,262-image budget.
   - **Weakly-supervised pseudo-bbox**: cite `phase-04-weakly-supervised-pseudo-bbox.md.deferred` — Grad-CAM bootstrapping of zone-aware detection at zero annotation cost. Deferred from this thesis; full 10-day experiment plan archived.
   - **Cross-dataset generalization**: cite `phase-05-cross-dataset-generalization.md.deferred` — light-finetune on HMDB51 or UCF-Crime. Deferred from this thesis; full plan archived.
   - **INT8 hardware-aware quantization**: current INT8 is slower than FP32 on M4 (ORT dequantize overhead). ONNX Runtime INT8 with hardware-specific kernel (QNN EP on Snapdragon, TRT INT8 on Jetson) expected to reverse this. Future benchmarks should test native INT8 paths.
   - **Multi-camera spatial fusion** and **online learning**: as described in Ch 5 §5.7 limitations.

## Related Code Files

| Action | Path |
|--------|------|
| Modify | `report/chapter-01-introduction.md` — §1.3, §1.4, §1.5, new fair-comparison table |
| Modify | `report/chapter-04-experiments-and-results.md` — §4.2, §4.5, §4.6, §4.7, §4.8, §4.9 |
| Modify | `report/chapter-05-edge-deployment.md` — §5.5, §5.6, new §5.x |
| Create | `report/chapter-06-discussion.md` |
| Create | `report/chapter-07-conclusions.md` |

## Implementation Steps

**Step 0 — Reconcile to GROUND_TRUTH.md (MUST run before any prose edits)**
0a. Run: `grep -rn "1\.39\|1\.4 ms\|13\.6 MB\|13\.6MB\|~400 frame\|26\.5 ms\|37\.7 FPS" report/ PROGRESS.md README.md`
0b. For each match, replace with canonical value from GROUND_TRUTH.md §7.
0c. Reconcile latency table in every chapter to Chapter 5 Table 5.2 values: 28.8 / 32.0 / 44.7 / 86.6 ms p50.
0d. Reconcile ensemble model size to 40 MB INT8 (sum of components).
0e. Reconcile annotation budget to **401** (not "~400") in all chapters and PROGRESS.md.
0f. After replacements, verify no instance of the wrong values remains.

1. **Read all existing chapters** to catch any additional inconsistencies beyond Step 0.
2. **Resolve remaining inconsistencies** using GROUND_TRUTH.md as the single reference.
3. **Edit Ch1**: reorder §1.4 contributions; add fair-comparison table; update §1.3 Objective 2 wording; update §1.5 Jetson language.
4. **Edit Ch4 §4.2**: add CI column (use Phase 06 data; if Phase 06 not yet complete, add placeholder "[TBD - Phase 06]").
5. **Edit Ch4 §4.5**: reframe gap analysis opening paragraph (5–6 sentences).
6. **Edit Ch4 §4.6**: rename and integrate Phase 03 data-efficiency curve figure.
7. **Add Ch4 §4.7**: copy Phase 04 result (Branch A or B write-up from that phase).
8. **Add Ch4 §4.8**: cross-dataset table (from Phase 05) or 1-sentence placeholder.
9. **Add Ch4 §4.9**: ablation table + bootstrap p-values (from Phase 06).
10. **Edit Ch5 §5.5**: add RTSP + alert latency table from Phase 02.
11. **Edit Ch5 §5.6**: replace projection with Phase 01 table (or update heading).
12. **Add Ch5 §5.x**: energy efficiency table.
13. **Draft Ch6**: 6 sections as outlined above; ≤ 8 pages.
14. **Draft Ch7**: 3 sections as outlined; ≤ 5 pages; include future-work annotation-cost estimate.
15. **Final consistency pass**: check all figure references (figures/ directory), table numbers, and cross-references between chapters.

## Todo List

- [ ] 1. Read all existing chapters; build number reference table (PROGRESS.md vs. chapter text)
- [ ] 2. Resolve number inconsistencies (1.39 ms superseded claim, etc.)
- [ ] 3. Edit Ch1: reorder contributions, add fair-comparison table, update §1.3 Objective 2
- [ ] 4. Edit Ch4 §4.2: add CI column (or TBD placeholder)
- [ ] 5. Edit Ch4 §4.5: reframe gap analysis as a finding
- [ ] 6. Edit Ch4 §4.6: rename, integrate Phase 03 data-efficiency curve
- [ ] 7. Add Ch4 §4.7: Phase 04 result
- [ ] 8. Add Ch4 §4.8: Phase 05 cross-dataset table or placeholder
- [ ] 9. Add Ch4 §4.9: Phase 06 ablation + bootstrap
- [ ] 10. Edit Ch5 §5.5: RTSP + alert latency (Phase 02)
- [ ] 11. Edit Ch5 §5.6: Phase 01 Jetson table or updated projection
- [ ] 12. Add Ch5 §5.x: energy efficiency table
- [ ] 13. Draft Ch6 (Discussion) — 6 sections
- [ ] 14. Draft Ch7 (Conclusions + Future Work) — 3 sections
- [ ] 15. Final consistency pass: figures, table numbers, cross-references

## Success Criteria

- All 7 chapter files exist and are internally consistent (no contradictory numbers).
- Ch1 fair-comparison table present; Pareto framing is the opening contribution.
- Ch1 Objective 2 cites 85.78 % as primary comparison.
- Ch4 §4.5 opens with "finding," not "failure to reach."
- Ch5 §5.6 has a measured table OR an explicitly-labeled projection.
- Ch6 exists with Hawthorne Effect critique paragraph.
- Ch7 exists with future-work annotation cost estimate.
- No claim of "1.39 ms INT8" (superseded); all latency numbers from `deployment/artifacts/benchmarks.md`.

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Phase 01–06 not complete when restructure begins | Medium | Write placeholders for missing sections with "[TBD - Phase XX]" tags; complete in order |
| Number inconsistencies between chapters (latency, accuracy) | High | Build reference table in Step 2 before editing; all edits reference the table |
| Ch6/Ch7 scope creep (trying to add too much new content) | Medium | Hard length limits: Ch6 ≤ 8 pages, Ch7 ≤ 5 pages; cut to meet limit |
| Hawthorne critique offends paper authors | Low | Frame as academic distinction: "behavioral compliance effect" vs. "model accuracy"; cite the Hawthorne effect literature (Adair 1984) |

## Security / Ethics / Privacy

- Surveillance ethics discussion in Ch6 must acknowledge: any real deployment must comply with applicable data protection law (GDPR or Vietnamese Decree 13/2023/ND-CP on personal data); recordings of workers require consent.
- No code changes in this phase — documentation only.

## Dependencies + Next Steps

- Depends on Phases 01–06 for content to integrate.
- Output consumed by: **Phase 08** (defense deck builds from restructured thesis).
- Landing chapters: all of `report/`.

**Unresolved questions**
- Is there a currently unresolved `1.39 ms` vs. `benchmarks.md` discrepancy in Ch1 §1.4? Check and fix in Step 2.
- Does the Vietnamese thesis submission require a fixed chapter structure? If yes, Ch6 and Ch7 numbering may need to follow a specific format prescribed by the university.
