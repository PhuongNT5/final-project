# Tier C1 — BBox Annotation Bet: Feasibility Verdict

**Question:** Should you spend ~1 month on paper-scale bbox annotation to flip Objective 2 from negative to positive?
**Verdict:** **No to the month-long annotation. Yes to a ~1-week corrected experiment on data you already have.**

Two findings (from the research + skeptic + your own codebase) change the calculus:

### Finding 1 — You already have 1,600 annotated frames on disk (not 400)
`annotation_guide/frames_to_annotate/` + `annotation_guide/annotations/` hold **1,600 labelled frames** across 8 classes (350+350+250+250+100×4) — the Phase-2 expansion is **already done**. The corrected experiment can run on existing data with **zero new annotation**.

### Finding 2 — V10 tested the WRONG mechanism
The paper's YOLOv4 is a **frame GATE** (it discards frames with no unsafe-behaviour object, passing a reduced sequence to ConvLSTM) — the +10 pp jump (85.78→95.81%) comes from *filtering*, not richer features. V10 instead used the fine-tuned YOLO as a **feature-extractor replacement** with **full unfreeze on 400 frames** (catastrophic forgetting — see seed `direct4_s7` collapsing to 34/64). **V10 is not a valid test of the paper's approach.** A corrected frame-gate experiment is a genuinely different, untested hypothesis.

> **So V10's regression does NOT falsify C1 — it falsifies a misimplemented C1.** That's the opening. But the structural and statistical ceilings below are real.

---

## The sober reality you must accept before deciding

**1. N=64 caps what any improvement can claim.** Bootstrap CI = ±4.13 pp (~±2.6 clips). Anything **below 60/64 is statistically indistinguishable** from your current 56/64 (McNemar). So "meeting >95.8%" (=62/64) is, on this test set, **essentially out of statistical reach** — even a "successful" C1 at 58/64 can't be claimed as a significant win. **Reframe the goal:** not "hit 95.8%," but "get a defensible bump toward 60/64 *or* strengthen the negative finding with a corrected-mechanism + scaling-curve contribution."

**2. The structural problem may be unfixable by bboxes.** The confusion is WalkwayViol↔UnauthorInt (7 of 9 errors), whose discriminator is **which zone** a worker stands in — *scene context*, not worker appearance. Raw cosine 0.934; after contrastive training still 0.788. A bbox around a *person* cannot encode the zone. Human annotators agree only at **κ=0.65** on this pair (6/20 flipped). More person-boxes ≠ zone signal.

**3. The fix requires zone-aware annotation, not just more frames.** Each frame needs a *second* box around the zone-boundary marker (painted line/barrier/gate), and frames with no visible boundary must be **excluded**. This is a different, harder annotation task than V10 did — and is what would consume the 50–60 real hours (the "25–40 h" estimate is optimistic).

---

## Recommended plan — cheap first, gated, honest

### Step 1 (DO THIS — ~3–5 days, no new annotation): corrected frame-gate experiment
Run the genuinely-untested mechanism on the existing 1,600 frames:
- **Train YOLOv11s as a frame gate** (binary unsafe-with-zone vs background), **two-stage recipe**: Stage 1 `freeze=10`, lr0=0.001, 20 ep, BN in eval(), **COCO replay ~10% of each batch**, mosaic=0.5; Stage 2 `freeze=0`, lr0=0.005, 80 ep, close_mosaic, batch=16 (not V10's batch=4). Monitor COCO mAP — if it drops >3 pp in Stage 1, lower lr0.
- **Integrate as a gate, not a backbone:** keep the Transformer head on **frozen-COCO** features; use the detector's per-frame score to *mask/filter* the sequence (low threshold T≈0.3, swept on val). Retrain only the head, seeds {42,123,314,7,2024}.
- **Pre-register the gate:** evaluate on the same N=64; report McNemar vs 56/64 + bootstrap CI.

**Gate G2:** if this corrected run does **not** reach **≥58/64**, STOP. You've spent days, not a month, and gained a clean second data point.

### Step 2 (ONLY if Step 1 hits ≥58/64 AND advisor approves): zone-aware re-annotation of the confusion pair
- Annotate **700 frames, classes 0+1 only** (skip CarryOverload — already 100%), **zone-aware** (worker box + zone-boundary box; exclude zone-ambiguous). ~8–12 h with SAM2-assisted CVAT. Second-annotator κ check every 200 frames; require **κ≥0.70** or revise the guide.
- Retrain; **Gate G3: ≥59/64** to continue.

### Step 3 (ONLY if Step 2 trends up): scale to 2k (Gate ≥60/64), then 4k (parity attempt ≥62/64)
Budget realistically **~50–60 h** for 4k zone-aware frames. Most-likely outcome even at 4k: **57–61/64** — a modest, probably-not-significant bump.

---

## Accuracy projection (honest, mostly interpolated)
| Frames | Zone-naive (like V10) | Zone-aware | Confidence |
|---|---|---|---|
| 400 (V10) | **54/64 (measured)** | not tested | measured |
| 700 pilot | ~54/64 | 54–57/64 | low |
| 1,000 | ~53–54/64 | 55–58/64 | low |
| 2,000 | ~53–55/64 | 56–59/64 | low–med |
| 4,000 | ~53–55/64 | 57–61/64 | uncertain |
| 10,000 | ~54–56/64 | 59–63/64 | speculative |

Zone-naive stays flat (volume doesn't fix a structural problem). Paper's 62/64 is at the outer edge and needs annotation quality matching their unreleased schema.

---

## The fallback win (why even a "no" is a contribution)
If C1 gates out at any stage, you still publish a **positive methodological result**:
1. **Mechanism correction quantified** — measured difference between V10 (feature-extractor, zone-naive) and the corrected frame-gate, zone-aware protocol. Replaces your single weak V10 point with a curve.
2. **Annotation-scaling curve** (400→1k→2k) — fills the exact extrapolation your thesis currently flags as a limitation (§6.4).
3. **Empirical grounding for the data-bound claim** — κ=0.65 persisting shows the signal isn't single-frame-recoverable even by humans, explaining *why* the authors' annotations were essential and unreleased.

This converts your thesis's weakest spot (one extrapolated data point) into evidence — a net upgrade over the current Path A, for ~1 week of work.

---

## Bottom line for the decision
- **Don't** commit a month to 4,000-frame annotation — structurally unlikely to clear the N=64 significance bar, and it would be the only thing at risk if Objective 3 weren't already done (it is).
- **Do** run the **corrected frame-gate experiment on the existing 1,600 frames (~1 week)** — it tests a real untested hypothesis, can only help (bump *or* stronger negative), and is fully gated.
- **Reframe** the target with your advisor: on N=64 the honest ceiling for a *claimable* win is ~60/64; the realistic prize is a sharper finding, not 95.8%.
- **In parallel**, lock the guaranteed wins: Tier A fixes + Jetson N=4 soak + dashboard. These, not C1, are what most de-risk the defense.
