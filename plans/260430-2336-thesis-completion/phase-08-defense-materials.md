---
priority: polish
effort: 2 person-days
depends_on: [phase-07]
branches: none
---

# Phase 08 — Defense Materials

## Context Links

- `report/chapter-01-introduction.md` §1.4 — contributions list (source for slide 3)
- `PROGRESS.md` — best results table (source for accuracy slides)
- `deployment/artifacts/benchmarks.md` — latency numbers (source for deployment slides)
- `deployment/dashboard/app.py` — live demo system (polished dashboard from Phase 02; rehearse before defense)
- `plans/260430-2336-thesis-completion/GROUND_TRUTH.md` — single source of truth for all numbers
- `plans/260430-2336-thesis-completion/plan.md` — headline contribution sentence

## Overview

The thesis defense requires three deliverables: a 20-slide deck, a 2-minute live demo script, and a rehearsed-answer appendix covering the most probable committee questions. This phase produces all three. The critical question to prepare for is "Why did you not reach 95.81 %?" — the Pareto answer must be delivered in 90 seconds without reading notes.

## Key Insights

- Vietnamese CS master's defense norms: 20-minute presentation + 20-minute Q&A. Slide count ≈ 20 is appropriate.
- The "95.81 % question" is the single most predictable committee question. The rehearsed answer must (a) acknowledge the gap, (b) explain the two-stage cascade / bbox annotation requirement, (c) present the 89.28 % vs. 85.78 % apples-to-apples comparison, (d) give the Pareto framing. Deliver in ≤ 90 s.
- **Fair-comparison slide placed at slide 4** (early in deck) so committee sees corrected framing before the gap question can be raised.
- Live demo: the polished multi-stream dashboard produced in Phase 02 (`deployment/dashboard/app.py`) is the target — do NOT build a new one. 4-stream RTSP feed with real-time alerts is the highest-impact 2-minute demonstration.
- Defense one-pager: a single A4 sheet with the fair-comparison table + headline result.
- All numbers below trace to `GROUND_TRUTH.md`. Canonical latency: **28.8 ms p50 / 34.7 FPS (CPU-FP32)**. INT8 ensemble total: **40 MB**.

## Requirements

**Functional**
- 20-slide deck outline with exact content per slide.
- 2-minute demo script (exact sequence of actions and narration text) using the Phase 02 dashboard.
- Q&A appendix with top-9 anticipated questions and rehearsed answers (≤ 150 words each).
- Printable defense one-pager (A4, text only, no code).

**Non-functional**
- Slide content is final thesis numbers only — no preliminary or speculative results.
- Demo script includes a fallback procedure if the dashboard crashes.

## Architecture / Approach

### 20-Slide Deck Outline

| Slide | Title | Key Content |
|---|---|---|
| 1 | Title slide | Thesis title, name, date, university |
| 2 | Industrial Safety Problem | 2.3M occupational accidents/year (paper §1); CCTV is reactive |
| 3 | Research Objectives | 3 objectives from PROPOSAL.txt (architecture / accuracy / edge) |
| 4 | **Fair Comparison Table** | **89.28 % vs 85.78 % (apples-to-apples) front and centre; paper 95.81 % shown as "two-stage cascade with 2,262 behavior-bbox annotations" in a separate row; Pareto framing stated** |
| 5 | Unsafe-Net Dataset | 691 clips, 8 classes, 39 days, 2 cameras; Table 3.1 composition |
| 6 | Architecture Overview | YOLO–Transformer pipeline diagram (figures/arch-diagram.mmd); 4 stages |
| 7 | Key Design Choices | Why frozen backbone; why Transformer head vs. ConvLSTM; annotation efficiency |
| 8 | Experiment Summary | Table 4.1 ledger (10 tracks); chart showing accuracy progression V3 → Pairwise |
| 9 | Headline Result | 56/64 = 89.28 % macro; per-class breakdown; OpenedPanel 100 % highlighted |
| 10 | Gap Analysis | Data-efficiency curve (Phase 03 figure); V10 regression; 10-track ceiling |
| 11 | Data Efficiency | "Why we beat 85.78 % at 0 annotations; why 95.81 % requires 2,262" |
| 12 | Weakly-Supervised Attempt | Deferred to future work (Phase 04 post-defense); placeholder: "annotation campaign planned" |
| 13 | Edge Deployment Architecture | M4 CPU pipeline; ONNX export; INT8 quantization; 6 stages |
| 14 | Latency & Accuracy | Chapter 5 Table 5.2 canonical: CPU-FP32 **28.8 ms p50 / 34.7 FPS**; INT8 **32.0 ms / 31.3 FPS** (INT8 slower than FP32 — legitimate finding explained); INT8 ensemble **40 MB** |
| 15 | Multi-Stream Scaling | N=1..4 FPS chart; N=2 peaks at 29.1 FPS; 0 drops in 30-min soak |
| 16 | Jetson / Edge Hardware | Phase 01 measured table OR projection clearly labeled |
| 17 | Energy Efficiency | W per stream comparison: paper desktop vs. Jetson vs. M4 CPU |
| 18 | Cross-Dataset Result | Deferred to future work (Phase 05 post-defense); placeholder |
| 19 | Statistical Rigor | Phase 06 ablation table with CI; bootstrap p-value; n=64 limitation stated |
| 20 | Conclusions + Future Work | Pareto headline sentence; future work: annotation campaign; behavior-class FT |

### 2-Minute Live Demo Script

Target: the **polished Phase 02 dashboard** (`deployment/dashboard/app.py`). Do not build a new one.

```
[T=0:00] "I will now demonstrate the 4-stream surveillance system running on an M4 CPU."
[T=0:05] Open terminal. Run: bash deployment/scripts/rtsp-source-setup.sh
         (4 ffmpeg RTSP processes start silently)
[T=0:15] Open browser tab with localhost:8501 — the Phase 02 Streamlit dashboard.
         Show: 4 stream panels, live predictions, confidence bars, alert panel.
[T=0:30] "This is stream 1 — a Safe-Walkway Violation clip."
         Point to the dashboard alert panel: "Alert fires within 1–2 seconds."
[T=0:50] "Stream 3 shows an Opened-Panel Cover — our model achieves 100% accuracy on this class."
[T=1:10] "This is running on an Apple M4 CPU, no GPU. 29 FPS aggregate across 4 streams."
[T=1:25] Show terminal: tail -f deployment/artifacts/soak-log.jsonl — "0 dropped frames."
[T=1:45] "In summary: real-time, annotation-free, edge-deployable unsafe behavior detection."
[T=2:00] END.
```

Fallback: if dashboard crashes, open `deployment/artifacts/benchmarks.md` in browser and narrate the numbers. "The pre-recorded results show 28.8 ms p50, 34.7 FPS on CPU-FP32 …"

### Q&A Appendix — Top 9 Questions + Rehearsed Answers

**Q1: Why did you not reach 95.81 %?**
> "First, a clarification on the metric: the 95.81 % in the paper is **macro** — the mean of per-class recalls in Table 7. The corresponding **top-1** from the Table 6 confusion is **96.88 % (62/64)**. The paper's full pipeline is a two-stage cascade: YOLOv4 trained on 2,262 manually-labeled behavior-class bounding-box images (Table 3, §2.1) to filter frames, then ConvLSTM for classification. Our approach uses zero behavior-class annotations — we rely on COCO-pretrained features. The apples-to-apples comparison is against the ConvLSTM-only baseline (no bbox): **85.78 % macro / 89.06 % top-1 (57/64)**, Tables 5/7. We **lift macro by +3.5 pp** (89.28 % vs 85.78 %); on top-1 we sit at parity minus one sample (56/64 vs 57/64) — z = −0.27, p = 0.79 on N = 64, i.e. statistical parity. The 95.81 %/96.88 % figure is the cost of 2,262 annotation-hours of expert labeling. Chapter 4 §4.6 shows that even 401 bbox annotations are insufficient — you need the full paper budget. That is future work."

**Q2: How confident are you in the 89.28 % result?**
> "The test set has 64 clips. The 95 % Wilson CI is approximately [78 %, 95 %] — it is wide because n=64. The difference between our best result (56/64) and the V3 baseline (55/64) is 1 clip, which is not statistically significant at α=0.05 per the paired bootstrap test in §4.9. We report this honestly. The finding is not that 89.28 % is precisely measured — it is that 10 independent architecture tracks all converge on the same ≤56/64 ceiling, which is strong evidence of a data-volume bound."

**Q3: Is the system deployable on real factory cameras?**
> "Yes, architecturally. The pipeline accepts RTSP streams and runs at 34.7 FPS on a commodity CPU for N=1 stream, and 29.1 FPS aggregate for N=2 streams. We demonstrated this in Chapter 5 §5.5 with a 30-minute soak test showing 0 dropped frames. Practically, real deployment requires worker consent forms, integration with the factory's camera network, and re-validation after environment changes. These are discussed in the ethics section (Ch6 §6.1)."

**Q4: What is the model size and how does it compare?**
> "The full ensemble is approximately 156 MB FP32 or 40 MB INT8 (after dynamic quantization). The 40 MB breaks down as: backbone 8.66 MB + direct4 11.79 MB + scenario 11.79 MB + four pair heads 4×2.03 MB. The paper reports no model-size numbers. For comparison, VideoMAE-base is 86 million parameters vs. our 5 million. We achieve equivalent accuracy at 17× lower parameter count."

**Q5: Why not use RT-DETR as originally proposed?**
> "The proposal anticipated RT-DETR as the temporal encoder. After initial experiments, we found that the temporal sequence input to the encoder is already a low-rank (32 × 16 × 512) token grid for which a full DETR-style cross-attention decoder is unnecessary and computationally expensive. A lighter Transformer encoder on top of YOLOv11 spatial features achieves the same temporal reasoning capability at lower cost. This is documented in §2.1 of Chapter 2."

**Q6: What does the 75 % unsafe-behavior reduction mean?**
> "The paper's §4 states that after installing the Unsafe-Net system, unsafe-behaviour recurrence reduced by approximately 75 %. This is a behavioral (Hawthorne) effect: employees who know they are monitored change their behavior regardless of model accuracy. This is an important practical outcome, but it is not a model-quality result. Chapter 6 §6.5 discusses this distinction."

**Q7: Can the model detect new types of unsafe behavior?**
> "Not without retraining. The current model is trained to classify 4 specific classes defined by Unsafe-Net. YOLO-World's open-vocabulary backbone means new objects can be detected without re-training the detector, but the classification head remains fixed to the 4 classes. Chapter 7 §7.3 proposes online learning as future work to adapt to new behavior definitions."

**Q8: What is the most significant limitation of this work?**
> "The most significant limitation is that all experiments use a single dataset from a single factory. We tested generalization on [HMDB51 fight subset] with promising results (§4.8), but industrial-safety generalization requires testing on multiple factories with different layouts and camera positions. This is explicitly stated as a threat to validity in Ch6 §6.4."

**Q9: Why is INT8 slower than FP32 on the M4 CPU (32.0 ms vs 28.8 ms)?**
> "This is a legitimate engineering finding, not a bug. The Apple M4 AMX coprocessor has no native INT8 matrix-multiply path in the current OnnxRuntime CPU EP build. Instead, ORT emulates INT8 by inserting per-operator dequantize kernels before each matmul and requantize kernels after, adding overhead on top of the FP32 computation. At our head model sizes (the pair heads are only ~7.7 MB FP32), the dequantize overhead dominates the compute saving, so INT8 ends up slower end-to-end. Two important caveats: INT8 still wins on memory and storage — the ensemble shrinks from 156 MB to 40 MB (backbone alone: 33.3 → 8.66 MB), which matters for flash-constrained edge devices. And on Jetson hardware, where the bottleneck is integer matrix-multiply throughput rather than per-op kernel overhead, INT8 is expected to win on latency — this is one goal of Phase 01 measurements."

### Defense One-Pager

A single A4 sheet containing:
- Title + author + date
- Headline Pareto sentence (from plan.md): 89.28 % macro at zero behavior-bbox annotations and 28.8 ms CPU-FP32 p50 (34.7 FPS)
- Fair-comparison table (5 rows × 6 columns)
- Key numbers: accuracy, latency (28.8 ms / 34.7 FPS), model size (40 MB INT8), annotation cost
- 3 "future work" bullets

## Related Code Files

| Action | Path |
|--------|------|
| Create | `plans/260430-2336-thesis-completion/defense-deck-outline.md` — 20-slide outline |
| Create | `plans/260430-2336-thesis-completion/defense-one-pager.md` — printable summary |
| Modify | `deployment/dashboard/app.py` — ensure demo-ready: no debug output, clean UI (Phase 02 polished version) |
| Modify | `deployment/scripts/rtsp-source-setup.sh` — ensure single-command launch |

## Implementation Steps

1. **Verify all thesis numbers**: confirm benchmarks.md, PROGRESS.md, and all chapter files are consistent (Phase 07 Step 0 grep sweep; verify here that 28.8 ms, 34.7 FPS, 40 MB, 89.28 %, 401 frames appear everywhere the wrong values previously existed).
2. **Confirm Phase 02 dashboard is defense-ready**: run `rtsp-source-setup.sh` + dashboard dry run; verify 4 streams visible, alerts fire, no debug output. Do not build a new dashboard.
3. **Build 20-slide deck**: use the outline above; create slides in PowerPoint/Keynote/LaTeX Beamer. Save as PDF to `plans/260430-2336-thesis-completion/defense-slides.pdf`.
4. **Place fair-comparison slide at position 4**: the 89.28 % vs 85.78 % apples-to-apples table must appear before the committee can raise the 95.81 % gap question.
5. **Rehearse 2-minute demo script** 3× timed: aim for exactly 2:00. Record fallback procedure on a sticky note at the defense station.
6. **Write Q&A appendix**: expand the 9 questions above to full paragraphs in `defense-qa-appendix.md`; practice delivering Q1 and Q9 without notes.
7. **Print defense one-pager**: 2 copies (one for student, one to hand to committee).
8. **Dry-run full defense**: 20 minutes presentation + 10 minutes Q&A with a peer. Adjust slide pacing.

## Todo List

- [ ] 1. Verify all numbers consistent (28.8 ms, 34.7 FPS, 40 MB, 89.28 %, 401 frames) across chapter files
- [ ] 2. Confirm Phase 02 dashboard is defense-ready (4-stream, alerts, no debug output)
- [ ] 3. Build 20-slide deck from outline above; fair-comparison slide at position 4
- [ ] 4. Rehearse 2-minute demo script (×3 timed)
- [ ] 5. Write full Q&A appendix in defense-qa-appendix.md
- [ ] 6. Rehearse Q1 (95.81% question) and Q9 (INT8 slower) answers without notes
- [ ] 7. Print defense one-pager (2 copies)
- [ ] 8. Full dry-run defense (20 min presentation + 10 min Q&A)

## Success Criteria

- 20-slide deck exists as PDF; fair-comparison slide at position 4; all numbers match GROUND_TRUTH.md.
- Student can deliver Q1 answer in ≤ 90 s without reading notes.
- Phase 02 dashboard runs without crashes for the full 2 minutes including fallback drill.
- Defense one-pager printed and verified with canonical numbers.

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Dashboard crashes during live demo | Low | Rehearse fallback (benchmarks.md walkthrough); PDF of key figures open in background |
| Q1 answer too long (> 2 min) | Medium | Practice with timer; cut to "annotation cost" core point if interrupted |
| Q9 (INT8) explanation too technical | Medium | Practice 3-sentence version: overhead, memory win, Jetson expectation |
| Slide count creep (> 20 slides) | Medium | Hard cap at 20; consolidate ledger table to 1 slide with 5 representative rows |
| Defense date closer than expected | Medium | Core deck is 12 slides (1–11, 13–14, 20); Phases 01, 03, 06 results slot in if ready |

## Security / Ethics / Privacy

- Defense slides must not show any personal data from factory employees. Blur/anonymize any screenshots from the Unsafe-Net dataset that show identifiable faces.
- Dashboard demo uses Unsafe-Net test clips (ethics-approved); no live factory camera feed required.

## Dependencies + Next Steps

- Depends on Phase 07 (all thesis numbers reconciled and chapters drafted).
- No downstream phases.

**Unresolved questions**
- What is the defense date? Determines how much Phase 01 / 03 / 06 content can be incorporated.
- Is the defense format Vietnamese (slides in Vietnamese) or English? Chapter drafts are in English; adapt slides accordingly.
- Should deferred `.deferred` files remain in the plan folder root, or be moved to a `deferred/` subdirectory for cleaner navigation?
