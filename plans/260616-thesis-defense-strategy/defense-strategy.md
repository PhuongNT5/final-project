# Master's Thesis Defense Strategy & Completion Plan

**Thesis:** Real-time Unsafe Behaviour Detection in Industrial Surveillance Based on YOLO and Transformer
**Student:** PhuongNT5 · **Defense context:** Vietnamese hội đồng phản biện
**Prepared:** 2026-06-16 · Source: 18-agent audit (10 readers + 2 auditors + 5 examiner personas + synthesis)

---

## 1. Executive Verdict

**Defensible now? YES — medium pass-risk, drops to LOW after the must-fix corrections + accuracy-ladder rehearsal.**

The thesis is honestly written, statistically rigorous, and produces real measured artifacts on actual Jetson hardware. It can pass. The danger is *not* the missed accuracy target itself (that is defensible as a negative finding) — it is **17 verifiable factual errors** in the submitted text, any of which a prepared examiner can check in 30 seconds against the thesis's own JSON files, turning "honest negative result" into "sloppy methodology."

### The 3 highest-leverage actions
1. **Fix the N=4 / N=2 soak conflation (CRITICAL).** Ch6 §6.1 + README + Ch1 say "4 cameras × 30 FPS sustained 36.2 min @ 7.0 W." The artifact (`soak-jetson-summary.json`) says the 36.2-min soak was **N=2**; N=4 zero-drop is a *separate ~10 s* test. This is the headline deployment claim and it is wrong as printed.
2. **Rehearse the 3-tier accuracy ladder until automatic.** The accuracy question lands in the first 5 minutes. Deliver instantly: **56/64** (test-selected upper bound, = deployed checkpoint) → **53.3/64** (typical) → **≈52/64** (blind, the scientific claim). Confusing the tiers under pressure is the single most likely cause of a fail verdict.
3. **Resolve ρ=0.21 vs ρ=0.69 (CRITICAL).** §§4.7.8–4.7.9 cite ρ=0.21 but Table 4.12 shows ρ=0.69 for the same pair. Either footnote the distinction (binary error-vector vs probability-vector correlation) or correct the text.

---

## 2. Objective Scorecard

| Objective | Status | Justification |
|---|---|---|
| **Obj1** — YOLO-Transformer hybrid w/ RT-DETR temporal reasoning | **Partially met (reframed)** | Hybrid built & ablated; RT-DETR dropped after controlled comparison (52–53/64 vs 55/64, §3.2.0); temporal streams add **zero** measurable lift (Table 4.16) — itself the dataset finding. |
| **Obj2** — Unsafe-Net accuracy/F1 **> 95.8%** | **MISSED — honest negative result** | Blind held-out ≈**52/64 ≈81% top-1 / ≈84% macro** (§4.7.11); ~14–15 pp below target, and below the no-bbox baseline (57/64=89.06%). Gap causally isolated to missing per-frame bbox zone annotations. |
| **Obj3** — TensorRT + INT8 multi-stream Jetson edge | **EXCEEDED** | TRT-FP16 8.62 ms/116 FPS; INT8 8.24 ms/121 FPS; N=4 zero-drop; 36.2-min N=2 soak, 0 throttle, 7.0 W — all measured, JSON-backed. |

**Net:** 1 exceeded, 1 partial, 1 missed. The defense lives or dies on reframing Obj2's miss as a *deliberate scientific contribution* and leaning on Obj3 as an independent deliverable.

---

## 3. Central Strategic Problem & Winning Narrative

**The problem:** the proposal is effectively a research contract approved on the basis of >95.8%. Delivered blind result ≈81% — below the paper's *own* no-bbox baseline. In Vietnamese academic culture this risks reading as "dự án thất bại" rather than "kết quả khoa học phủ định nghiêm ngặt."

**The frame — "The annotation-bottleneck thesis." Do not lead with the accuracy number. Lead with the question you answered:**
> *Can the paper's 95.8% be reproduced WITHOUT per-frame bounding-box annotations, on edge hardware, in real time?* Answer: **no** — and the thesis proves *why* with four independent experimental chains, then delivers the first measured edge-deployable system for this task.

**Reframe the three objectives as one story:**
- **Obj1:** built the hybrid → discovered the signal is per-frame *appearance*, not temporal motion (dataset characterization finding).
- **Obj2:** 100+ models / 10 tracks → proved the ceiling is **annotation-bound, not architecture-bound** (the causal answer the paper never gave).
- **Obj3:** delivered what the paper never attempted — measured, privacy-preserving, sub-9 ms edge inference on a USD 249 device.

**Data-bound argument chain (memorize — deliver in ≤2 min, no notes):**
1. VideoMAE (86M params, Kinetics) hits the **same ≤55/64 ceiling** as YOLOv11s (5M, COCO) → not a capacity problem.
2. WalkwayViol ↔ UnauthorInt frozen-feature **cosine = 0.834** → geometrically indistinguishable without zone cues.
3. V10: 400 bbox frames, full fine-tune → **54/64 regression** → annotation *scale* is the binding variable.
4. Cross-ensemble frozen+bbox → **no blend exceeds 55/64** → blending ≠ dense annotation.

**Thesis statement (memorize verbatim):**
> "This thesis set out to replicate Önal & Dandıl's 95.8% accuracy *without* per-frame bounding-box annotations, on a USD 249 edge device running in real time. The evidence — 100+ trained models, an 86M-parameter VideoMAE control, cosine-similarity analysis, and a direct bbox fine-tuning pilot — proves the gap is **annotation-bound, not architecture-bound**: without zone-level spatial labels, no model reliably separates WalkwayViol from UnauthorInt. The thesis delivers the causal explanation of this gap and the first measured real-time edge-deployment system for this task — 116 FPS on Jetson Orin Nano Super, zero dropped frames across four concurrent streams."

**Vietnamese language note:** avoid "phủ định giả thuyết / falsified hypothesis." Use **"kết quả thực nghiệm âm tính có cơ sở khoa học"** (evidence-based negative result) or **"phát hiện giới hạn dữ liệu qua thực nghiệm kiểm soát."**

---

## 4. Defense Q&A Prep (hardest questions + model answers)

### Theme A — Objectives & Accuracy (existential)
**Q1. Promised >95.8%, blind ≈81%, below the no-bbox baseline. How is this passing, not failed?**
A Master's thesis demonstrates independent research capability + original contribution; it need not hit every target. This delivers (1) a 100-model controlled study proving the ceiling is annotation-bound; (2) the first measured edge system for this task; (3) a blind-protocol correction withdrawing two inflated figures — methodological integrity. Causal isolation of the annotation bottleneck advances beyond the paper's own baseline study. That is original scholarship.

**Q2. The >95.8% target = the paper's full-bbox result needing 2,591 unreleased per-frame annotations. Did you know?**
Honest answer: no — the proposal underweighted the role of the bbox-trained YOLOv4 detector. Once experimentation began, "scene-level appearance classification" vs "zone-boundary-aware detection" became the central research question; V10 (§4.6) + gap analysis (§4.5) answer it directly. The annotation request to the authors is documented (§1.5) — good-faith effort.

**Q3. Even best-case 56/64 < no-bbox 57/64. Why is a sub-baseline system a contribution?**
Blind ≈52/64 CI [79.7, 95.3]; vs 57/64, z=−0.27, p=0.79 — the gap is one sample, statistically non-significant. The contribution is not top-1 parity; it is (a) an edge system the paper never attempted, (b) a causal diagnosis of the no-bbox ceiling, (c) two perfect per-class results (OpenedPanel 13/13, CarryOverload 8/8) at **zero annotation cost** vs the paper's 2,591 labels.

**Q4. Proposal said 125 test clips; you report 64. Why change the denominator?**
64 = the 4-class unsafe subset (32+11+13+8) matching the paper's Table 6 confusion matrix exactly — enables a head-to-head comparison. Using all 125 mixes the 4 safe classes and makes the number incomparable to the paper. The 8-class result (~78%, §6.4) is reported as future work.

### Theme B — Methodology & Statistics
**Q5. Architecture choices (layer 22, 4×4 pool, T=32) were also test-informed. Is blind ≈52/64 truly leakage-free?**
§4.7.11 corrects *final-checkpoint* leakage, not upstream architecture-search leakage — acknowledged. Mitigation: choices had large margins (1×1 pool 32/64 vs 4×4 55/64 — not a one-sample call) and VideoMAE (totally different architecture, no shared design) hits the same ceiling. Blind ≈52/64 is labeled exactly as "final-checkpoint honest estimate under fixed architecture"; a fully unbiased study needs a separate arch-search val set, impossible at 691 clips — stated as a limitation.

**Q6. Table 4.12 shows ρ=0.69; §§4.7.8–4.7.9 cite ρ=0.21. Source?**
0.69 = Pearson over continuous probability vectors; 0.21 = Pearson over the binary correct/incorrect error indicator. High prob-correlation + low error-overlap is consistent. *(If unverifiable before defense: concede openly, declare Table 4.12's 0.69 authoritative, note the ensemble still adds one sample 55→56, directionally consistent.)*

**Q7. The 57/64 calibrated figure came from a 20k-step test-set bias search. Did you tune on test, and why trust other numbers?**
A discipline error, not fraud. When the honest train-set retry returned 53.4±1.2/64 the figure was **immediately withdrawn** and a 3-level transparency framework introduced. The §4.7.7 disclosure was self-initiated. Other numbers: 56/64 inherits ~4–5 samples of selection optimism (quantified §4.7.11); blind ≈52/64 is the corrected claim.

**Q8. UnauthorInt 8/11 with CI [45, 100] spans 54 pp — is macro 89.28% meaningful?**
N=11 is too small for a precise per-class estimate; the wide CI correctly captures that. Macro is an interval ([81.3, 96.3]), not a point. Two saturated classes dominate. This *is* the structural diagnosis: UnauthorInt↔WalkwayViol disambiguation needs zone annotations. Per-class CIs reported §4.7.9.1.

### Theme C — Architecture & Novelty
**Q9. Appearance-only 53.67 ≈ motion-only 53.00 ≈ both 53.33. Transformer adds nothing; RT-DETR dropped; gating < average. What is the architectural contribution?**
The empirical proof that *no* Transformer temporal head — including 86M VideoMAE — recovers class-discriminative signal absent from per-frame appearance when zone info is missing: a principled negative finding about the dataset's temporal structure. The pipeline (frozen YOLOv11s + Transformer aggregator + multi-stream fusion + TRT) is a complete engineering framework whose value is Obj3 (116 FPS). The Transformer is present and ablated; its temporal value is zero on *this* dataset — that is the finding.

**Q10. RT-DETR is a detector, not a sequence model. How implemented; were the 52–53/64 trials adequately trained?**
Adapted the RT-DETR cross-attention decoder over the (T=32,16,512) token sequence with T learned queries (time-steps as "objects"). Same 60-epoch + early-stopping schedule as the final pipeline. 52–53/64 reflects a structural mismatch: the decoder targets spatial localization diversity, not single-label temporal classification; CLS-token aggregation is the better inductive bias.

**Q11. Did you test a plain MLP (mean-pool) baseline?**
Not directly — acknowledged limitation. The depth-heads sweep (Table 4.17) partially answers: smallest config (L=2/H=4, 53.67) ties the deepest (L=8/H=8, 51.33), spanning near-MLP to full Transformer at a flat ceiling; VideoMAE confirms the bottleneck is input-feature quality. *(2-hour fix available — Tier B1.)*

### Theme D — Deployment & Systems
**Q12. Ch6 says "4 streams, 36.2 min, 7.0 W"; the soak log says N=2. Explain.**
An editing error conflating two experiments. `soak-jetson-summary.json`: 36.2 min, **N=2**, 7.03 W, 0 throttle. N=4 zero-drop is a separate shorter throughput benchmark (Table 5.5E). Correct statement (Ch5 Table 5.5F has it right): N=2 sustained 36.2 min @ 7.0 W; N=4 demonstrated zero drops in a shorter run. Both are real.

**Q13. Table 5.5B FP16 per-stage sums to 11.70 ms vs measured 8.62 ms e2e — impossible.**
The e2e 8.62 ms/116 FPS is correct (`benchmarks-jetson-fp16.json`). The per-stage values are stale planning-doc numbers; measured stages are 2.76 + 1.41 + 4.44 = 8.61 ms. Headline stands; the breakdown table must be corrected.

**Q14. "33% energy improvement (34.2 vs 45.7 mJ)" — arithmetic is 25.2%.**
Correct, calculation error. (45.7−34.2)/45.7 = 25.2%. The 33% applies to the FPS/W metric (29.2 vs 21.9). Correct statement: "25% lower energy/inference (45.7→34.2 mJ) with +31% FPS at near-identical power."

**Q15. Live-demo says both "8,988 events / 772 s" and "3,826 events / 5 min"; the dashboard screenshot doesn't exist.**
8,988 events / 772.1 s is correct (`live-demo-stream.jsonl`, exact line count + timestamps). The "5-minute / 3,826" sentence is a stale draft — delete it. The screenshot was not committed (production gap, will regenerate); the JSONL is the primary evidence.

### Theme E — Scope & Framing
**Q16. You swapped RT-DETR for a plain encoder without committee approval. Justify the mid-project deviation.**
§1.3.1 (scope-evolution table) + §3.2.0 proactively disclose it with a data-driven reason (52–53 vs 55/64). The spirit of Obj1 (YOLO spatial + Transformer temporal aggregation) is preserved; only the variant changed on evidence. Disclosing deviations with supporting data is the correct academic posture.

**Q17. Is ~81% top-1 suitable for industrial safety? You miss ~1 in 5.**
No — and the thesis does not claim standalone safety certification (§5.6.7: for high-stakes inspection the full Unsafe-Net cascade remains correct). It is positioned as a zero-annotation-cost, privacy-preserving, real-time *alert layer* where the alternative is no automated monitoring; dual-threshold (0.75/0.55) routes borderline cases to a human in the loop. A supplementary aid, not a safety-rated system — and 81% ≫ 0%.

**Q18. The data-bound claim rests on V10 — one annotator, 400 frames, no Cohen's κ. Could 54/64 just be annotation noise?**
V10 is one of four pillars, not the proof. Triangulated by: VideoMAE same ceiling (no annotation involved), cosine 0.834 (feature-space, annotation-independent), cross-ensemble failure, plus V10. Single-annotator limitation disclosed §4.6/§6.3; the ≥4,000-frame threshold is framed as extrapolation, not measurement.

---

## 5. Must-Fix Before Submission

See `must-fix-checklist.md` for the full 17-item checklist with sources. Critical (catchable in 30 s):
1. **N=4/N=2 soak conflation** — Ch6 §6.1, §6.5, Ch1 §1.4, README, Ch5 intro → state N=2 for the 36.2-min/7.0 W soak.
2. **ρ=0.21 vs 0.69** — Ch4 §§4.7.8–4.7.9 vs Table 4.12.
3. **95.81% mislabeled "top-1"** — Ch4 line 4 → it is *macro*; top-1 is 96.88%.
4. **Table 5.5B FP16 per-stage** — 11.70 ms sum vs 8.62 ms e2e → use 2.76/1.41/4.44.
5. **"33%" energy** — → 25.2%.
6. **Live-demo 5-min/3,826** contradiction → delete; use 772 s / 8,988.
7. **Missing dashboard screenshot** (`jetson-live-dashboard-demo.png`) → regenerate or replace.

---

## 6. Tiered Completion Plan

### Tier A — MUST DO (0–7 days, regardless of timeline)
- **A1.** Fix all 17 inconsistencies (1 day) → kills the easy credibility attacks (Q12–Q15).
- **A2.** Regenerate/commit the dashboard screenshot (2 h) → removes the only missing visual evidence.
- **A3.** Resolve ρ=0.21 provenance (1–2 h) → neutralizes Q6.
- **A4.** Rehearse the 3-tier accuracy ladder, Vietnamese script (2 h) → the make-or-break answer.
- **A5.** Rehearse the 4-pillar data-bound chain in ≤2 min (2 h) → neutralizes Q1/Q2/Q18.
- **A6.** Add explicit H0 hypothesis statement to §3.4/§1.3.1 (30 min) → neutralizes Q3 framing.
- **A7.** Correct Table 5.5B power-mode header (nvpmodel=0, 15 W) (15 min).
- **+** Merge per-chapter bibliographies into one; add missing ref [22] UCF-Crime; produce Figure 3.2 (mandatory pre-submission).

### Tier B — HIGH-VALUE (if 1–3 weeks)
- **B1.** Plain MLP (mean-pool) baseline, ~52–55/64 expected (2–4 h) → resolves Q9/Q11 conclusively.
- **B2.** Compute both ρ (prob + binary-error) for direct4 vs cascade, add to §4.7.5 (1 h) → closes Q6.
- **B3.** Per-class bootstrap CIs for WalkwayViol + UnauthorInt in Table 4.4 (30 min) → preempts Q8.
- **B4.** Run a real **N=4 ≥30-min Jetson soak** (1 day, needs Jetson) → makes the Ch6 headline *true*, closes Q12 fully.
- **B5.** 8-class result on all 125 clips (~78%, 2 h) → answers Q4, preempts cherry-pick accusation.
- **B6.** Draw Figure 3.2 multi-stream gating diagram (2 h).

### Tier C — STRETCH (3+ months / journal paper)
- **C1.** ≥4,000-frame bbox annotation + retrain (25–40 h labelling) → expected ≥90%, converts negative→positive finding.
- **C2.** Multi-SKU Jetson (AGX/NX) validation (1 day).
- **C3.** Inter-rater Cohen's κ for bbox annotations (5 h) → closes Q18 for publication.
- **C4.** Live camera (USB/CSI) capture integration (1 week).

---

## 7. Recommended Slide Structure (13 slides)
1. Title + framing (annotation-bottleneck study + measured edge system; bold VN bottom line).
2. Research context & gap (paper needs 2,591 bbox labels + workstation; nobody did edge).
3. System architecture (YOLOv11s frozen → layer-22 tokens → multi-stream Transformer → direct4/cascade → ONNX/TRT → Jetson).
4. **Obj3 results — the STRENGTH slide** (M4 vs Jetson FP32/FP16/INT8; N=4 zero-drop; N=2 soak; 156→40 MB).
5. **Accuracy — 3-tier ladder** (56/64 → 53.3/64 → ≈52/64; withdrawn 57/64 noted).
6. Paper comparison (4 rows; "−1 sample vs no-bbox, p=0.79, at zero annotation cost"; OpenedPanel/CarryOverload 100%).
7. The data-bound finding (4 pillars).
8. Experiment scale & ablations (10 tracks/100+ models; blind protocol).
9. Obj1 architecture findings (RT-DETR dropped on evidence; temporal lift = 0).
10. Per-class + error analysis (9 error clips; 5 are WalkwayViol↔UnauthorInt).
11. Live-demo evidence (dashboard; 772 s / 8,988 events; ~20.6 kbps privacy).
12. Scope deviations & transparency (3-row promised/delivered/why table).
13. Contributions & future work (4 contributions; Priority 1 = ≥4,000-frame bbox).

---

## 8. Open Decisions for the Student
1. **Time before defense?** <1 wk → Tier A only. 1–3 wk → add B1, B4, B5. >3 mo → consider C1 + journal.
2. **Does your advisor agree with the "deployment-led / honest negative" framing?** Non-negotiable to settle first — the printed thesis is written this way; you cannot defend a different story than what is on paper.
3. **Jetson access before defense?** If yes: regenerate dashboard, run N=4 soak, re-verify per-stage TRT numbers in 1 day.
4. **Which accuracy number on the abstract/title page?** Recommend: report both with labels, blind ≈52/64 as primary (already what §1.3.1 does — do not change).
5. **Email Önal & Dandıl for the bbox files now** — either you retrain (changes the story) or you have documented good-faith effort the committee cannot fault.
