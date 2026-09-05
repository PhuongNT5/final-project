# Chapter 6 — Conclusion and Future Work

> ### Pipeline attribution — read before citing any number in this chapter
>
> This project evolved through **three non-interchangeable pipelines** (full map:
> `docs/system-architecture.md`). Chapters were written at different points in that evolution, so
> their headline accuracies are **not** competing estimates of one system — they describe different
> systems, and mixing them is the single most common error this project has made internally.
>
> | | Pipeline | Best cited | Blind/honest | Protocol |
> |---|---|---|---|---|
> | Ch1, Ch3, Ch4 | **P1** frozen-COCO YOLO -> multi-stream Transformer | 56/64 (87.5 %) | **53/64** (82.8 %) | same-dist, N=64 |
> | Ch5 | **P1** as exported to Jetson | 55/64 (TRT engine, as recorded in `benchmarks-jetson-*.json`) | **53/64** | same-dist, N=64 |
> | Ch7 + Abstract | **P3** entity detector + spatial-relation Transformer | **54/64** (84.4 %, 88.4 % macro) | 54/64 | same-dist N=64, **plus k-fold and LCO at N=516** |
>
> **The thesis headline is P3 (Ch7).** P1's 56/64 is **test-selected** (the ensemble was chosen
> using the test clips) and must never be presented as the result; its honest blind counterpart is
> 53/64. P3 is the only pipeline with evidence beyond N=64, which is why it leads.
> Chapters 1-6 are retained as the experimental record of how the design was arrived at, not as
> the final claim.

## 6.1 Summary of Findings

This thesis investigated whether a hybrid YOLO–Transformer pipeline,
trained on a small public industrial-safety dataset (Unsafe-Net, 691
short clips), can match a published baseline (96.88 % top-1 / 95.81 %
macro with full per-frame bbox annotations) and run on commodity edge
hardware. The accuracy answer is **no** — and the thesis's value lies
elsewhere. Four findings stand out, in order of contribution strength.

**Finding 1 — Real-time edge deployment is solved and measured on
embedded hardware (the primary delivered contribution).** End-to-end
inference on Apple M4 CPU (FP32 ONNX) runs at 26.8 ms p50 / 37.3 FPS
single-stream, sustains 29.1 FPS aggregate at N = 2 concurrent streams
with zero dropped frames, and INT8 dynamic weight quantization shrinks
the deployed graph from 156 MB → 40 MB (3.8 ×). The same pipeline was
then deployed and **measured** on a Jetson Orin Nano Super (USD 249–499,
JetPack 6.2 + TensorRT 10.3): TRT-FP16 single-stream **8.62 ms p50 /
116 FPS** (3.11 × faster than M4 CPU FP32); TRT-INT8 with 256 real-frame
entropy calibration **8.24 ms / 121 FPS**; **4-stream zero-drop
throughput demonstrated (88 FPS aggregate, Table 5.5C)**; and a **2-stream
soak sustained for 36.2 minutes at 30 FPS with zero dropped frames**
at **7.0 W average power** (47 % of the 15 W MAXN budget, junction
temperature ≈ 24 °C below throttle at steady-state, 23 °C at peak). A live operator dashboard with
rolling-window dual-threshold alerting (≥ 0.75 ALERT, ≥ 0.55 WARNING)
achieved 100 % recall on true-unsafe streams and 0 % false-alarm on a
safe-input stream in a 12.9-minute multi-stream demonstration, with raw
frames never leaving the edge (metadata-only uplink). The path is fully
open-source (no proprietary runtimes, no GPU rental); no published
baseline for this task reports any of these measurements (§5.6).

**Finding 2 — Accuracy is data-bound, but the best zero-bbox configuration
reaches macro parity-to-advantage over the no-bbox baseline even under
blind evaluation.** A blind held-out protocol (epoch, configuration, and
seed all selected on a train-carved validation slice; the 64-clip test
touched once) was applied across frozen-COCO, bbox-finetuned, and hybrid
feature spaces with the corrected YOLO frame-gate (§4.6.2). The best
configuration — **hybrid (frozen ⊕ detection-finetuned) features +
frame-gate** — achieves **blind 53/64 (82.8 % top-1, 85.4 % macro)**, i.e.
**macro parity with the no-bbox ConvLSTM baseline (85.78 %) even under the
honest protocol**, and best-case (matched test-selected protocol) 86.9–
87.7 % macro, *exceeding* the no-bbox baseline. Top-1 remains below baseline
(53–54/64 vs 57/64) and well below the full-bbox cascade (62/64, 95.81 %),
which is the data-volume-bound corner. Reported as a transparent ladder
(best-case 56/64 → blind 53/64). Twelve+ independent tracks — including an
86 M VideoMAE backbone, end-to-end fine-tuning, and **the paper's exact
frame-gate mechanism replicated at 1,200 bbox frames** — converge on the
same ≈ 54/64 ceiling. That neither an SOTA video backbone nor the paper's
own mechanism beats this ceiling is the core evidence the gap is
**data-volume-bound, not architecture-bound** (Chapter 4 §4.6.2).

**Finding 3 — Annotation, not architecture, is the bottleneck.** The
residual error is concentrated on the WalkwayViol ↔ UnauthorInt
confusion pair, where the frozen-COCO features place the two classes
at cosine 0.834 (Table 4.19). A direct test — annotating 400 frames
with behaviour-class bounding boxes and fine-tuning YOLOv11s on them
(V10) — *regressed* to 54/64 (Table 4.6), a cross-ensemble of frozen
and bbox branches capped at 55/64, and a 267-clip UCF-Crime zero-shot
probe showed the head routes out-of-domain anomalies *away* from the
nearest semantic class (§4.8.5) — a Unsafe-Net-specific classifier, not
a generic unsafe-behaviour concept. All three support the data-bound
conclusion (Chapter 4, §4.5–§4.6, §4.8.5).

**Finding 4 — The pairwise-contrastive cascade is not a contribution.**
Under blind configuration-selection the single direct-4 head is the best
configuration on *every* seed; the cascade and the direct-4 + cascade
ensemble are rejected. This confirms, in the strongest form, the N = 64
significance analysis (ensemble − direct-4: P = 0.63, CI touches zero;
cascade alone significantly worse, Table 4.15). The most-engineered
component of the pipeline is reported honestly as tried-but-unhelpful
(Chapter 4, §4.7.9).

## 6.2 Honest Reading

This thesis does not match the paper's 96.88 % top-1 (95.81 % macro),
and it does not present a new architecture that ought to. What it
presents instead is a **delivered, measured edge-deployment system**
(§6.1 Finding 1) and a **falsified hypothesis** with a quantitative
cause: architecture changes do not close the gap under a frozen-backbone
or 400-frame finetune budget, because the COCO features place visually
similar safe/unsafe pairs at cosine ~0.83. The negative result on V10 is
itself a contribution: it documents that the bbox-annotation route
requires (the evidence suggests) substantially more than a 400-frame
pilot.

On the accuracy number the thesis is deliberately, and unusually,
transparent. The honest headline is the **blind held-out estimate of the
best zero-bbox configuration — hybrid (frozen ⊕ detection-finetuned)
features + frame-gate (§4.6.2): 53/64 (82.8 % top-1, 85.4 % macro)** —
obtained by selecting epoch, configuration, threshold, and seed on a
validation slice carved from the training set, never on the 64 test clips.
This is at **macro parity with the no-bbox baseline (85.78 %) even under the
blind protocol**, and best-case (matched test-selected) 86.9–87.7 % macro
*exceeds* it; top-1 stays below baseline (53–54 vs 57). The numbers are
reported as a ladder so the reader sees what each source of optimism was
worth:

| Number | top-1 | macro | What it is |
|---|---|---|---|
| Best-case (test-selected, lucky seed, frozen ensemble) | 56/64 | 89.28 % | optimistic upper bound; the *deployed* checkpoint |
| Frozen single-head blind | ≈ 52/64 | ≈ 84 % | the original-pipeline honest estimate |
| **Hybrid + frame-gate blind (best config)** | **53/64** | **85.4 %** | the defensible generalization estimate |

The previously reported **57/64 = 89.1 % calibrated** figure is withdrawn
(test-set bias tuning, §4.7.7), and so is the **+3.5 pp macro win**, which
does not survive the blind protocol. The deployed system uses the best-case
frozen-ensemble checkpoint (the artifact one would actually ship); the
*scientific claim* about accuracy is the blind 53/64 (85.4 % macro). The
two are kept distinct on purpose.

## 6.3 Limitations

**(a) Test set size and model-selection leakage.** 64 clips on the
4-class subset (16 per class on average) is sufficient to demonstrate
operational viability but limits statistical claims about small accuracy
differences (§4.7.9). More importantly, the small N forced a
single-test-set design in which epoch/seed/configuration were originally
selected with reference to the test set; §4.7.11 measures and removes
this leakage with a train-carved validation slice, establishing the
frozen single-head honest blind at ≈ 52/64. The best zero-bbox
configuration (hybrid + frame-gate, §4.6.2) raises this to 53/64 (85.4 %
macro) — the thesis's defensible accuracy headline. A larger dataset
permitting a permanent held-out validation split would remove the need
for this correction entirely.

**(b) Cross-domain coverage is partial.** A zero-shot cross-domain probe
on a second safety-anomaly dataset (UCF-Crime) *was completed* and is
reported in §4.8.5: 267 clips across three anomaly classes (Burglary 100,
Fighting 50, RoadAccidents 117). The limitation is coverage, not
execution — three of the five target UCF anomaly classes were evaluated
(RoadAccidents partial; Vandalism and Stealing were not fetched in the
download window). The probe is sufficient to show the trend (the head
collapses out-of-domain anomalies onto CarryOverload, §4.8.5) but a full
five-class balanced transfer evaluation remains future work.

**(c) Bbox finetune scale.** The 400-frame V10 pilot established that
*small-scale* targeted bbox annotation hurts the COCO prior more than
it adds behavioural signal. Whether the trade-off inverts at a
paper-scale annotation budget (≥ 4 000 frames) is the central open
question (§6.4).

**(d) Three-seed retraining ablations.** The retraining-based
ablations (appearance-only, motion-only, gated vs. unweighted
ensemble, depth × heads sweep) are completed with three seeds per
variant and reported in Tables 4.16–4.18 of §4.7.8. The std bands
(0.0–1.25 clips) are tight enough to rank configurations, but a
larger seed pool would reduce noise further; this is a
quality-of-evidence limitation rather than a missing experiment.

**(e) Single-SKU edge measurement.** Chapter 5 §5.6 reports measured
Jetson Orin Nano Super numbers, but on a single device of one Jetson
SKU. The Pareto framing (annotation cost × accuracy × edge-readiness)
generalises across SKUs, but absolute FPS and power numbers can shift
on the AGX Orin or NX SKUs. A multi-SKU evaluation is future work.

## 6.4 Future Work

**Priority 1 — Paper-scale bounding-box annotation.** The V10 negative
result in §4.6 points to the most direct route the present evidence
supports for closing the WalkwayViol ↔ UnauthorInt gap: scaling the
bounding-box annotation budget by an order of magnitude (4 000 – 10 000
frames). The 400-frame pilot regressed; whether the trade-off inverts,
and at what budget, is a hypothesis this thesis does not establish —
intermediate scales (1 000, 2 000 frames) were not run, so the specific
≥ 4 000-frame figure is an extrapolation, not a measured threshold. At that scale
the YOLOv11s detector can both retain the COCO prior (via a low-
learning-rate schedule on the backbone) and acquire enough
behavioural signal on workers/forklifts to push the cosine similarity
of the confusion pair below 0.5. Three concrete steps make this
tractable: (i) public release of the original Unsafe-Net bbox
annotations by the paper's authors (filed as a request, not yet
fulfilled); (ii) failing that, a 4 000-frame annotation pass using
SAM-assisted segmentation and a single-pass `Worker` class (one
unsafe vs. one ambient class), targeting a doubling of behaviour
samples relative to the V10 pilot; (iii) a learning-rate schedule
that freezes the backbone for the first 30 epochs and only unfreezes
the deepest two C2f blocks. Estimated annotation effort: 25–40 hours
of skilled labelling.

**Priority 2 — Cross-domain validation.** Extend the completed
three-class UCF-Crime zero-shot probe (§4.8.5; 267 clips: Burglary,
Fighting, RoadAccidents) to the two remaining anomaly classes
(Vandalism, Stealing) and complete the RoadAccidents fetch, then frame
the full five-class set as a binary anomaly-detection task. The current
probe already shows the head collapses out-of-domain anomalies onto
CarryOverload; the full set would quantify whether any
*behaviour-feature transfer* (useful) exists beyond *Unsafe-Net artefact
memorization* (not useful).

**Priority 3 — Multi-SKU edge characterisation.** The Jetson Orin Nano
Super measurement (§5.6) anchors the edge corner of the Pareto frontier
but only on one SKU. Repeating the cross-precision + multi-stream +
power-sweep protocol on the AGX Orin and the Orin NX SKUs would map the
upper and lower bounds of the Jetson family for this workload, and
would let an industrial deployer choose the SKU that matches their
camera count, power envelope, and budget. The toolchain is already
SKU-agnostic; the work is operating-cost rather than engineering-cost.

**Priority 4 — Multi-class scaling.** Unsafe-Net contains 8 labelled
classes (4 unsafe + 4 safe paired). Extending the head from 4-class
unsafe to 8-class joint safe/unsafe classification — and reframing
the problem as binary safe/unsafe with 8 sub-class evidence — would
provide a more deployment-realistic target. Preliminary 8-class runs
(§4.4) show ~78 % top-1, which is not yet competitive but is a
sensible next milestone after Priority 1 closes the bbox gap.

**Priority 5 — Larger seed pool for retraining ablations.** Tables
4.16–4.18 use three seeds per variant; expanding to seven or ten
would tighten the std bands enough to detect the small (≤ 1 clip)
deltas that are currently within noise. No new data or design
changes required.

## 6.5 Closing Remarks

The contribution of this thesis is a **Pareto-optimal point**, not a single
competitive top-1 number. The best zero-bbox configuration (hybrid + frame-
gate) reaches **macro parity with the no-bbox baseline even under blind
evaluation (85.4 % vs 85.78 %)** and exceeds it best-case — at zero bbox
annotation cost and with an edge deployment the baseline lacks. The
contribution is three-fold. First, a **delivered, measured, open-source
real-time edge-deployment system**: the trained pipeline sustains 4-stream
zero-drop throughput and a 2-stream 36.2-minute soak at 30 FPS on a
USD 249–499 Jetson at 7 W, keeps raw video on the device, and reproduces
from a clean clone — a capability no published baseline for this task
demonstrates. Second, the **corrected-mechanism replication and hybrid-
feature finding**: identifying that the paper's YOLO is a frame filter (not
a feature extractor), replicating it at 1,200 bbox frames, and showing that
combining frozen and detection-finetuned features recovers what either alone
loses. Third, the **causal isolation of the accuracy gap**: across twelve-plus
tracks — including an SOTA video backbone and the paper's exact mechanism —
the ≈ 54/64 ceiling holds, so the gap to the full-bbox 95.81 % is shown to
be **data-volume-bound**, not architecture-bound. The working system, the
mechanistic diagnosis, and the macro-competitive zero-bbox result are the
deliverables; the next milestone, set out in §6.4, is to scale the bbox
annotation budget to paper scale and re-run the
same pipeline. The system is ready; the data is the remaining work.

---

## References

Per-chapter numbering; merged at final compilation.

[1] K. B. Önal and E. Dandıl, "Unsafe-Net: YOLO v4 and ConvLSTM based
    computer vision system for real-time detection of unsafe behaviours
    in workplace," *Multimedia Tools and Applications*, vol. 84, pp.
    34967–34993, 2025.

[2] N. Carion et al., "End-to-End Object Detection with Transformers,"
    *Proc. ECCV*, 2020. arXiv:2005.12872.

[3] A. Dosovitskiy et al., "An Image is Worth 16×16 Words: Transformers
    for Image Recognition at Scale," *Proc. ICLR*, 2021.
