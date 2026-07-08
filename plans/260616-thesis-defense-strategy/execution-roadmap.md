# Execution Roadmap — >1 Month Timeline, Full Jetson Access

**Decided constraints (2026-06-16):** Timeline **> 1 month** · Jetson **full access** · Advisor framing **not yet discussed (GATING)**.

This timeline unlocks the high-reward path: **attempt to actually meet Objective 2** (>95.8% / at least beat the no-bbox baseline) via paper-scale bbox annotation, instead of only defending the negative result. That decision is gated on the advisor (D0) and on the C1 feasibility study (`c1-bbox-feasibility.md`).

---

## Critical-path decision (D0) — do this FIRST
**Advisor alignment.** Before committing a month of work, settle the strategic fork with your advisor (see `advisor-alignment-brief.md`):
- **Path A — Defend the negative result** (deployment-led; thesis as-printed). Low risk, the document already supports it. Work = Tier A + B + Jetson re-measurement.
- **Path B — Attempt the positive flip** (Path A *plus* the corrected experiment). Directly addresses the committee's #1 attack. **Refined by the feasibility study:** do NOT bet a month on 4,000-frame annotation; instead run a ~1-week **corrected frame-gate experiment on the 1,600 frames already on disk** (V10 tested the wrong mechanism), hard-gated at <58/64.

> **Final recommendation (post-feasibility):** **Path A in full + Path B Step 1 only** (corrected experiment on existing data, ~1 week). Decline month-scale new annotation — on N=64 anything <60/64 can't be claimed as a win, and the structural zone-context problem is unlikely to yield to more person-boxes. Step 1 can only help: a modest bump *or* a stronger negative finding (mechanism correction + scaling curve). The guaranteed de-riskers are the 17 fixes + the real Jetson N=4 soak.

---

## Phase 0 — Day 0 (parallel, no dependencies)
- [ ] **Advisor meeting** on the Path A/B fork (D0 gate).
- [ ] **Email Önal & Dandıl** requesting the original Unsafe-Net per-frame bbox annotation files. If shared, C1 collapses from ~30 h annotation to a retrain — and either outcome documents good-faith effort (neutralizes Q2). Cite the attempt in §1.5.
- [ ] **Kick off Tier A** factual corrections (framing-independent — safe to start before D0).

## Phase 1 — Stabilize document + deployment evidence (Week 1)
*Framing-independent; do regardless of Path A/B.*
- [ ] **Tier A:** fix all 17 errors in `must-fix-checklist.md` (1 day).
- [ ] **Jetson — exploit full access (highest value here):**
  - [ ] **B4 — run a REAL N=4, ≥30-min soak** with tegrastats logging. This converts your headline claim from "technically defensible" to **measured-true**, and closes Q12 entirely. Then rewrite Ch6 §6.1/§6.5 + Ch1 + README to the measured N=4 soak.
  - [ ] **A2 — regenerate the live dashboard screenshot** (re-run the Streamlit pipeline, capture, commit as `figures/jetson-live-dashboard-demo.png`); fix the Figure 5.6→5.7 collision.
  - [ ] **Re-verify FP16 per-stage latencies** on-device → replace the stale Table 5.5B (2.76/1.41/4.44 ms) and confirm the nvpmodel power mode.
  - [ ] **Re-confirm INT8 native-TRT calibration** path vs the TRT-incompatible ONNX-INT8 (DynamicQuantizeLinear) — make the distinction explicit so it can't be attacked.
- [ ] **Structural:** merge per-chapter references → unified bibliography; add missing ref [22] UCF-Crime; **produce Figure 3.2** (multi-stream gating diagram).

## Phase 2 — Strengthen the science (Weeks 2–3, parallel to Phase 3)
*Each item neutralizes a specific committee question.*
- [ ] **B1 — plain MLP (mean-pool) baseline** (~2–4 h). If it ties the Transformer (~52–55/64), it *conclusively* proves "head architecture doesn't matter, features do." Closes Q9 + Q11.
- [ ] **B2 — compute both ρ** (probability-vector AND binary-error-vector) for direct4 vs cascade; add to §4.7.5. Closes Q6 (the ρ=0.21/0.69 contradiction) with documented source.
- [ ] **B3 — per-class bootstrap CIs** for WalkwayViol + UnauthorInt in Table 4.4. Preempts Q8.
- [ ] **B5 — 8-class result on all 125 clips** (~78% expected). Answers the 64-vs-125 denominator question (Q4); report honestly.

## Phase 3 — The corrected experiment (Week 2, GATED by D0), NOT month-scale annotation
*Refined after the C1 feasibility study (`c1-bbox-feasibility.md`). Two findings changed the plan: (1) **1,600 frames are already annotated on disk** — no new annotation needed to start; (2) **V10 tested the wrong mechanism** (feature-extractor + full unfreeze) — the paper's YOLO is a **frame gate**, so a corrected run is genuinely untested.*

- [ ] **Step 1 — corrected frame-gate experiment on EXISTING 1,600 frames (~3–5 days, no new annotation).** Train YOLOv11s as a binary frame gate with the two-stage recipe (`freeze=10`→`freeze=0`, COCO replay ~10%/batch, batch=16, BN eval() in Stage 1); integrate as a **sequence filter** over **frozen-COCO** Transformer features (do NOT replace features). Pre-register evaluation: McNemar vs 56/64 + bootstrap CI.
  - **Gate G2: if < 58/64 → STOP.** Days spent, not a month; you've banked a clean corrected-mechanism data point.
- [ ] **Step 2 (only if G2 passes + advisor approves) — zone-aware re-annotation of the confusion pair.** 700 frames, classes 0+1 only, worker box **+ zone-boundary box**, exclude zone-ambiguous; SAM2-assisted (~8–12 h); κ≥0.70 quality gate. **G3: ≥59/64** to continue.
- [ ] **Step 3 (only if trending up) — scale 2k (G4 ≥60/64) → 4k (parity ≥62/64).** Budget ~50–60 h. Most-likely even at 4k: 57–61/64 (modest, likely not significant on N=64).
- [ ] **C3 — inter-rater Cohen's κ** on a 100-frame subset (closes Q18 for publication).

> **Reality check (settle with advisor):** on N=64, anything < 60/64 is statistically indistinguishable from the current 56/64. The realistic prize from Phase 3 is a **sharper finding** (corrected mechanism + 400→1,600 scaling curve) — NOT hitting 95.8%. Do **not** commit month-scale new annotation.

## Phase 4 — Defense preparation (final 1–2 weeks)
- [ ] Build the **13-slide deck** (strategy doc §7). Lead with Obj3 strength, then the 3-tier accuracy ladder, then the data-bound finding (or, on Path B success, the positive result).
- [ ] **Rehearse cold:** the 3-tier accuracy ladder (VN script) + the 4-pillar data-bound chain in ≤2 min.
- [ ] **Mock defense** with the 18 Q&A (strategy doc §4); have someone play each examiner persona.
- [ ] Final consistency pass — re-run the numeric audit against the corrected report.

---

## Decision gates
| Gate | When | Question | If NO |
|---|---|---|---|
| **D0** | Day 0 | Advisor approves Path A + Step-1 corrected experiment? | Resolve before any Phase 3 work |
| **G1** | Authors reply (~2 wk) | Original bbox files available? | Proceed with corrected frame-gate on existing data |
| **G2** | Step 1 done (~day 5) | Corrected frame-gate ≥ 58/64? | Stop Phase 3; report corrected-mechanism + 400→1,600 curve as the contribution; Path A still strong |
| **G3** | Step 2 done (if run) | Zone-aware 700-frame ≥ 59/64 (κ≥0.70)? | Stop; the scaling curve is the win |
| **G4** | 2 wk pre-defense | Document audit clean + answers rehearsed? | Freeze scope, focus on rehearsal |

## What each path delivers to the committee
- **Path A (negative result):** strongest-possible honest defense; Obj3 exceeded, Obj2 a rigorous data-bound negative finding, document airtight.
- **Path B success:** Obj2 **met** (≥90%, possibly >95.8%) — removes the existential question entirely; thesis becomes a positive, complete result. Best outcome.
- **Path B partial (G2/G3 stop):** even a stopped C1 yields a **measured annotation-scaling curve** (400→1k→2k frames), which converts the thesis's weakest point (a single extrapolated V10 data point) into evidence — a net upgrade over Path A.
