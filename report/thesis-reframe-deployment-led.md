# Deployment-Led Reframe & Consolidated Conclusion (draft for review)

> ### SUPERSEDED 2026-09-06 — pre-P3 draft, retained for framing only. Do not compile.
>
> This draft was written **before the P3 pivot** (entity detector + spatial-relation Transformer,
> Ch7) and describes only the **P1** pipeline. Three of its central claims no longer hold:
>
> 1. **"does not beat even the no-bbox baseline on top-1"** — overstated. P3 blind is 54/64 vs the
>    baseline's 57/64: Δ = −4.7 pp, **p = 0.435, a statistical tie**, not a defeat. §7.4 further
>    shows *no* system could demonstrate superiority at N = 64.
> 2. **The title suggestions in §2, and "drop YOLO and Transformer from the title"** — rejected.
>    They rest on P1's §4.8 finding that the Transformer head is not the operative component; in P3
>    the spatial-relation Transformer over entity tokens *is* the mechanism carrying the
>    cross-camera result (§7.7.2). The thesis title keeps YOLO and Transformer.
> 3. **The two-part contribution (edge system + data-bound diagnosis)** — incomplete. The thesis
>    now leads with the **scene-confound finding** (§7.7.1–7.7.2) and the N = 516 evidence; see
>    Ch6 §6.5 for the current four-part statement.
>
> **What is still usable:** the deployment-led *narrative stance* and the Jetson measurements. The
> Vietnamese Ch6 drafting plan cites this file for framing only
> (`plans/2026-09-06-thesis-2week-final/research/r3-…md`), which remains valid under these caveats.
>
> ---
>
> *Original header:* Standalone draft consolidating the thesis around what was actually delivered.
> Feeds Chapter 1 (§1.3 objectives, §1.4 contributions), Chapter 4/§4.8, and Chapter 6
> (conclusion). All claims are backed by committed artifacts. EN to match existing chapters.

## 1. The honest one-paragraph thesis

This thesis does **not** deliver a new architecture, does not reach the paper's 95.81 % macro, and —
under leakage-free evaluation — does not beat even the no-bbox baseline on top-1. What it *does*
deliver is (a) a **measured, reproducible, privacy-by-construction real-time edge system** for
industrial unsafe-behaviour detection (Jetson Orin Nano Super, INT8/TensorRT, 4-stream zero-drop,
36-min soak at 7 W) that **no prior work on this task provides**, and (b) a **rigorous, triangulated
diagnosis** that the accuracy ceiling on this dataset is **bound by missing zone supervision, not by
model capacity** — established from three independent negative results. The contribution is a
deployable system plus an honest, well-bounded scientific account of why the accuracy stops where it
does. That is the thesis.

## 2. Reframed title (pick one)

1. *A Real-Time, Privacy-Preserving Edge System for Unsafe-Behaviour Detection in Industrial
   Surveillance: Deployment, Rigorous Evaluation, and the Limits of Zero-Annotation Video Models*
2. *Edge-Deployable Unsafe-Behaviour Detection without Bounding-Box Supervision: A Measured System
   and a Data-Bound Diagnosis*
3. *From Server to Edge: A 7-Watt Real-Time Multi-Stream Unsafe-Behaviour Monitor and an Honest
   Account of the Zero-Bbox Accuracy Ceiling*

(Drop "YOLO and Transformer" from the title — §4.8 shows the Transformer head is not the operative
component. Keep YOLO/Transformer as *method* detail inside, not as the headline claim.)

## 3. Reframed objectives (replace proposal MT1–MT3)

| Old (proposal) | New (delivered) |
|---|---|
| MT1: novel YOLO–Transformer architecture (RT-DETR temporal layer) | **O1: Build a zero-bbox YOLO-feature + lightweight-head video classifier; characterise honestly what it can and cannot learn on Unsafe-Net.** (engineering composition, not new mechanism) |
| MT2: accuracy > 95.8 % | **O2: Rigorously evaluate generalisation (leakage-free k-fold, scene-robust, safety FNR) and diagnose the accuracy ceiling.** |
| MT3: edge optimisation + multi-stream real-time | **O3 (primary): Deploy and measure a real-time, multi-stream, privacy-by-construction edge system; quantify the accuracy–energy–latency Pareto vs the server baseline.** (unchanged — met/exceeded) |

## 4. The methodological core — a triangulated negative result

The thesis's scientific value is that it pins the accuracy ceiling to **zone supervision** from
three independent directions, each properly tested:

1. **Temporal modelling adds nothing** (§4.7.8 ablation): hand-built motion/derivative streams give
   no measurable lift over appearance.
2. **The Transformer head does not beat a linear probe** (§4.8 k-fold): on the random split it is
   *significantly worse* (Δ = −4.65 pp, Nadeau–Bengio p = 0.007); scene-robust it ties. A logistic
   regression on mean-pooled frozen features is as good or better → the attention machinery is not
   the operative component.
3. **Crude automated spatial features do not help** (WS4): adding per-cell energy + activation
   centroid changes top-1 by +0.1 pp (random) and **−3.8 pp** (scene-robust); they track scene
   layout, not the person, and hurt on unseen scenes.

All three point to the same conclusion the paper's design implies: the WalkwayViolation ↔
UnauthorizedIntervention confusion (which carries essentially the entire safety miss-rate — Unauth
FNR 68 % scene-robust) is **only resolvable with explicit restricted-zone supervision** (the paper's
2,262 per-frame bounding boxes, or an equivalent virtual fence). Image appearance alone, frozen or
fine-tuned, cannot recover which floor zone a person occupies. Reported as a negative result, this
spares future work three dead ends and is itself a contribution (cf. Beyer et al., 2017).

Feasibility note (positive): viewpoint clustering shows only ~6–8 fixed camera scenes and the
walkway is painted on the floor — so a *virtual fence* (one polygon per view + person foot-point
test) is the cheap path to the supervision the model lacks; it is scoped as immediate future work
(tooling and per-view frames already prepared in `annotation_guide/zones/`).

## 5. Primary contribution — the edge system (headline)

Recap as the lead result (Chapter 5, measured on hardware, not projected):
- **8.62 ms / 116 FPS** single-stream FP16; INT8 8.24 ms / 121 FPS; FP32 accuracy parity.
- **4 concurrent streams, 88 FPS aggregate, zero dropped frames**; 36.2-min soak, 0 throttle, **7.0 W**.
- **Privacy by construction:** raw frames never leave the device; ~50 kbps metadata uplink (2 orders
  below one 1080p stream) — auditable by bandwidth asymmetry alone.
- **Pareto position:** trades ~6.5 pp macro vs the full Unsafe-Net cascade for **zero bbox annotation
  (~190 h saved), a $249–499 / 7 W device** (vs the paper's ~400 W, ~$1,500 2×GTX-1080 workstation),
  4-stream real-time, and privacy. No prior work on this task occupies the edge operating point.

## 6. Honest accuracy positioning (one table, no spin)

| Configuration | Top-1 | Macro | Note |
|---|---|---|---|
| Paper full (2,262 bbox) | 96.88 % | 95.81 % | upper bound; heavy annotation |
| Paper no-bbox baseline | 89.06 % | 85.78 % | zero bbox |
| Ours — deployed checkpoint (canonical 64-clip split) | 87.5 % (56/64) | 89.28 % | shipped artifact |
| **Ours — k-fold random (single head, leakage-free)** | **75.3 ± 0.2 %** | ~~77.8~~ **77.6 ± 0.3 %** | representative generalisation (std/CI corrected by the 2026-09-06 re-run, §4.9.3) |
| **Ours — k-fold scene-robust** | ~~71.9 ± 1.6 %~~ **70.5 ± 1.9 %** | ~~71.8 ± 1.0 %~~ **72.5 ± 1.0 %** | superseded by the 2026-09-06 clip-level-bootstrap re-run (§4.9.3) |

State plainly: the canonical-split 56/64 is the deployed checkpoint; the leakage-free generalisation
of the core method is 72–75 %; we do not claim an accuracy advantage. The contribution is the
edge system + the rigorous diagnosis, on a tightened CI (15.6 pp → 4.4 pp).

## 7. Integration map (where each piece goes)

- **Ch.1 §1.3/§1.4:** swap objectives (table §3) + contributions = [edge system (primary), triangulated
  data-bound diagnosis, evaluation rigor, negative results]. Drop "novel architecture".
- **Ch.1/title:** adopt a title from §2.
- **Ch.4:** insert §4.8 (`chapter-04-section-kfold-cross-validation.md`); add the WS4 spatial negative
  result as §4.9 short subsection (1–2 paragraphs from §4 above).
- **Ch.6 conclusion:** lead with §1 paragraph; structure as (i) primary edge contribution, (ii)
  triangulated data-bound finding, (iii) limitations (single dataset, N, single SKU), (iv) future
  work = virtual fence (tooling ready) + multi-SKU + live camera.
- **Abstract:** rewrite from §1.

## 8. Residual risks / to flag to advisor
- The leakage-free numbers (72–75 %) are below current draft headlines — confirm advisor accepts the
  honesty upgrade (it is defensible and stronger than an unstated inflated number, but it is a change).
- "Transformer significantly worse than linear" is blunt — phrasing can be softened to "provides no
  significant advantage over a linear probe" while keeping the statistic in a footnote.
