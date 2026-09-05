# Chapter 1 — Introduction

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

## 1.1 Background and Motivation

Industrial facilities are hazardous environments. Workers share floor space
with moving forklifts, rotating machinery, exposed electrical panels, and
restricted robot cells. Traditional surveillance — CCTV monitored by human
operators — is fundamentally **reactive**: it records incidents for later
review but cannot intervene before an accident occurs. Collisions between
workers and forklifts in blind spots, unauthorized entry into robot cells,
and PPE violations continue to cause injuries and production losses despite
camera coverage.

Modern deep learning provides an opportunity to convert passive surveillance
into **proactive** safety monitoring: a system that recognizes unsafe
behaviour as it unfolds and alerts the operator in real time. Two model
families dominate the state of the art. **YOLO** [1, 2] offers real-time
object detection with state-of-the-art throughput on commodity hardware.
**Transformer**-based architectures [3, 4] — including RT-DETR and video
Transformers such as TimeSformer and SlowFast — excel at modelling temporal
context and complex, cluttered scenes typical of factory floors, but are
computationally expensive.

Neither family alone satisfies the combined requirement of (a) real-time
per-frame perception, (b) temporal reasoning across multi-second behaviour
windows, and (c) on-device deployment without a cloud round-trip. Cloud-based
inference introduces latency, bandwidth cost, and data-security risk — all
unacceptable in industrial deployments where a 500 ms alert delay can be the
difference between prevention and injury.

This thesis investigates a hybrid **YOLO–Transformer** architecture that
combines the throughput of a YOLO spatial encoder with the temporal
reasoning capability of a Transformer sequence encoder, optimized for edge
deployment on commodity hardware.

## 1.2 Problem Statement

**Input.** A continuous video stream from a factory surveillance camera
(1920×1080 @ 24 FPS).

**Output.** For every sliding window of T = 32 sampled frames, a probability
distribution over four unsafe-behaviour classes:

| Class ID | Class Name                         | Description                                |
| -------: | ---------------------------------- | ------------------------------------------ |
|        0 | Safe-Walkway Violation             | Worker walking outside the painted walkway |
|        1 | Unauthorized Intervention          | Worker entering a restricted machine zone  |
|        2 | Opened Panel Cover                 | Electrical panel left exposed              |
|        3 | Carrying Overload with Forklift    | Forklift carrying unsafe load              |

The system must run at **≥ 30 FPS** on a commodity edge CPU and sustain
**multiple concurrent streams** without dropping frames.

## 1.3 Research Objectives

This thesis pursues three objectives, drawn from the approved proposal:

**Objective 1 — Architecture.** Combine the speed of CNN-based detection
(YOLO family) with the temporal reasoning of Transformer-based sequence
encoders into a unified YOLO–Transformer pipeline for video behaviour
classification.

**Objective 2 — Training & Evaluation.** Train, fine-tune, and evaluate the
YOLO–Transformer on the **Unsafe-Net** dataset [5] — an eight-class
industrial-safety video benchmark — and measure accuracy against the
published baseline (YOLO v4 + ConvLSTM, 96.88 % top-1 / 95.81 % macro on
the 4-class unsafe subset; the paper-reported 95.81 % is the macro of
per-class recalls in [5] Table 7, while top-1 is recovered from the
Table 6 confusion matrix as 62/64).

**Objective 3 — Edge Deployment.** Apply model-optimization techniques
(ONNX export, INT8 quantization, potentially TensorRT compilation) so that
the pipeline meets real-time multi-stream constraints on edge hardware
(Apple M4 CPU and NVIDIA Jetson Orin Nano Super, both measured end-to-end).

### 1.3.1 Scope evolution from the approved proposal

Three commitments in the approved proposal changed during the work. They are
stated here, up front, so that the findings in Chapters 4–6 are read as
deliberate scientific outcomes rather than as gaps to be discovered.

| Proposal commitment | What was delivered | Why it changed |
|---|---|---|
| YOLO (spatial) → **Transformer temporal-reasoning** stage → MLP, with the encoder named **RT-DETR** | **Substantially met:** YOLOv11s spatial encoder → a multi-stream **Transformer encoder** temporal-reasoning head → MLP. The Transformer temporal stage *is* built; only the specific *RT-DETR* encoder was dropped. | RT-DETR is an end-to-end *object detector*, not a sequence/temporal encoder; the proposal conflated "Transformer" with "RT-DETR." A Transformer encoder over the frame sequence realizes the temporal-reasoning intent directly. RT-DETR is itself COCO-pretrained and would hit the same data-bound ceiling, so building it would not change the finding (argued in §6.2). |
| Test accuracy **> 95.8 %** | Best zero-bbox config (hybrid+gate, §4.6.2): **blind 53/64 (82.8 % top-1, 85.4 % macro)** — macro parity with the no-bbox baseline (85.78 %) even blind; best-case 86.9–87.7 % macro > baseline; top-1 below (53–54 vs 57). Full-bbox 95.81 % not reached. Ladder: best-case 56/64 → blind 53/64. | Twelve+ tracks (incl. VideoMAE and the paper's exact frame-gate mechanism @1,200 bbox) converge on the ≈54/64 ceiling; the gap to full-bbox is **data-volume-bound**, not architecture-bound (cosine ≈ 0.83 collapse, §4.6.2). The earlier "+3.5 pp macro win" was a test-set-selection artifact and is withdrawn (§4.7.11). |
| **8-class** (4 safe + 4 unsafe) classification | **4-class unsafe** subset as the headline; 8-class joint (~78 %) and binary safe/unsafe scoped to future work (§6.4) | The 4-class subset isolates the WalkwayViol ↔ UnauthorInt confusion pair that drives the entire shortfall, giving a clean object of study. |

The consequence for **positioning** is explicit, and it defines the spine of
this thesis. The contribution is *not* a novel architectural mechanism, and
*not* a competitive accuracy number. It is two-fold: **(i) a delivered,
measured, fully open-source real-time edge-deployment system** (Objective 3,
met in full — Chapter 5), and **(ii) a rigorous empirical characterization of
why frozen object-detection features hit a data-bound accuracy ceiling on this
task** (Objectives 1–2, which produced a data-bound ceiling diagnosis and a
hybrid-feature finding — the best zero-bbox configuration reaching macro parity
with the no-bbox baseline even under blind evaluation (§4.6.2) — rather than the
proposed competitive top-1 number). Accuracy is presented honestly as *evidence for the
empirical study*, not as a competitive claim. This deployment-led, empirically-
honest framing is declared here, up front, and carried through §1.4, Chapter 4,
and §6.2 — not revealed at the end.

## 1.4 Contributions

The contributions are presented in **deployment-led order**, matching the
thesis spine declared in §1.3.1: one objective (edge deployment) delivered in
full, and two (architecture, accuracy) recast as a rigorous empirical study.
Accuracy is reported honestly as *evidence for that study*, not as a
competitive claim — the paper of record [5] remains ahead on accuracy, and
this thesis does not contest that corner. What it adds is a measured,
reproducible edge system the paper does not report, and a causal explanation
of where the accuracy gap lives.

1. **A measured, reproducible, open-source real-time edge-deployment
   system (Objective 3 — delivered in full).** The trained pipeline runs
   end-to-end on two commodity targets, both measured (no projection, no
   proprietary runtime). On a **Jetson Orin Nano Super** (USD 249–499,
   JetPack 6.2 + TensorRT 10.3): TRT-FP16 **8.62 ms p50 / 116 FPS**
   single-stream; INT8 with 256 real-frame entropy calibration at FP16
   accuracy; **4-stream zero-drop throughput (88 FPS aggregate, Table 5.5C)
   and a 2-stream soak sustained for 36.2 minutes with zero dropped frames
   at 7.0 W average power** (47 % of the 15 W budget, ≈ 23 °C below
   throttle); a power-efficiency Pareto across nvpmodel
   profiles; and a live operator dashboard with rolling-window
   dual-threshold alerting whose only network egress is metadata
   (≈ 50 kbps, *privacy by construction* — raw frames never leave the
   edge). On **Apple M4 CPU**: 26.8 ms p50 / 37.3 FPS, ONNX + dynamic INT8
   compressing the deployed ensemble 156 → 40 MB (3.8×). No published
   baseline for this task reports edge deployment, multi-stream
   throughput, sustained soak, or per-frame energy. This is the primary
   delivered contribution. (Chapter 5)

2. **A YOLO–Transformer engineering composition** realizing the proposal's
   architecture intent (Objective 1, substantially met): a frozen YOLOv11s
   spatial encoder coupled with a multi-stream **Transformer encoder**
   temporal-reasoning head over appearance, 1st/2nd-order temporal
   derivatives, and windowed-statistics streams, ending in an MLP. The
   Transformer temporal stage realizes the "temporal reasoning" the
   proposal intended; the specific RT-DETR encoder was dropped on
   principled grounds (§1.3.1). The composition reuses prior building
   blocks — it is an artefact, not a new mechanism — and the temporal
   streams show no measurable accuracy lift on this dataset (§4.7.8),
   itself a finding about Unsafe-Net's predominantly per-frame signal.
   (Chapter 3; §6.2 for the honest reading)

3. **Comprehensive empirical study** of frozen-backbone approaches on
   Unsafe-Net spanning 10 experiment tracks (V3, V5, V6-det, V6-ft,
   VideoMAE, Final sweep, V7 YOLO-World, V8 CrossEns, V9 Pairwise,
   V10 bbox-finetune), 100+ trained models, with three tracks reported
   in detail in Ch4: V7 (best macro 89.54 %, UnauthorInt = 100 %), V9
   (best-case top-1, 56/64), and V10 (negative result, 54/64). Ten+
   independent architecture choices — including an 86 M-parameter
   VideoMAE backbone and end-to-end fine-tuning — converge on the same
   ≈ 54/64 ceiling (the best-config blind is 53/64; the frozen single-head
   blind is ≈ 52/64, §4.7.11; §1.4 contribution 5). That an SOTA
   video backbone does not beat frozen YOLO features is the principal
   evidence that the residual gap is **data-bound, not
   architecture-bound**. (Chapter 4)

4. **Why the zero-annotation corner cannot be moved with small-budget
   bbox supervision** — a falsified hypothesis (V10) plus a
   cross-dataset transfer test. Manually annotating 400 frames with
   behaviour-class bounding boxes and fine-tuning YOLOv11s on them
   *regressed* to 54/64; cross-ensembling frozen and bbox-finetuned
   branches caps at 55/64. A complementary **zero-shot transfer probe
   on 267 UCF-Crime anomaly clips (Burglary, Fighting, RoadAccidents)**
   shows the trained ensemble routes only **15.7 %** of clips to
   *UnauthorInt* — *below* the uniform 25 % baseline, i.e. the head
   actively routes *away* from the closest semantic match — and **78 %**
   to *CarryOverload* (= the smallest training class, with no visual
   overlap with burglary, fighting, or road accidents), while keeping
   high confidence (mean max-prob 0.747 vs 0.804 in-domain; **88 %** of
   predictions above 0.5). Combined with the raw-feature
   cosine probe (0.94–0.996 between visually similar safe/unsafe
   pairs), this localises the 6-sample shortfall to missing
   zone-boundary annotations rather than missing model capacity, and
   shows the head learned a **Unsafe-Net-specific classifier**, not a
   generic unsafe-behaviour concept. The paper's corner is therefore
   **annotation-volume bound and domain-bound**; it moves only with
   paper-scale (~order-of-magnitude larger) and more diverse bbox
   supervision. (Chapter 4 §4.5–§4.6, §4.8.5)

5. **Evaluation-rigor and leakage audit — the methodological core.**
   This thesis applies a **blind held-out protocol** that removes the
   test-set model-selection leakage present in the original pipeline:
   epoch, ensemble configuration, and seed are all chosen on a
   validation slice carved from the training set, with the 64-clip test
   set touched exactly once. This exposes ≈4–5 samples of the best-case
   56/64 as selection optimism. On the frozen single-head pipeline the
   blind estimate is ≈52/64; the **best zero-bbox configuration
   (hybrid+gate, §4.6.2) lifts this to a blind 53/64 (85.4 % macro)** —
   macro parity with the no-bbox baseline even under the honest protocol.
   Reported as a transparent ladder (best-case 56/64 → blind 53/64). Under
   blind configuration-selection the pairwise-contrastive cascade is
   rejected on *every* seed — the single direct-4 head is consistently
   best — so the cascade is reported honestly as a tried-but-unhelpful
   component, not a contribution. The audit also **withdraws** the originally reported
   "57/64 calibrated" figure (test-set bias tuning; honest train-fit
   reduces test to 53.4 ± 1.2 / 64), and reports 5 000-sample bootstrap
   CIs and a [CLS]-embedding cosine probe quantifying the raw-feature
   collapse (0.82–0.93) and head-level recovery (WalkwayViol ↔
   UnauthorInt cosine 0.834 → 0.375). (Chapter 4 §4.7)

6. **Reproducible open-source artifacts** — all trained weights, ONNX exports, INT8-quantized
   models, parity reports, ablation JSON outputs, t-SNE figures, the
   per-clip CSV under CC BY 4.0 attribution, and benchmark logs are
   included in `deployment/artifacts/`, `sweep_results/ablations/`,
   and `figures/`, and can be re-executed end-to-end from the
   `run_ablation_*.py` and `plot_figures_binary.py` scripts.

## 1.5 Scope and Limitations

**In scope.** Classification of behaviours on the Unsafe-Net 4-class unsafe
subset; frozen-backbone + trained-head approaches; ONNX + INT8 deployment
on Apple M4 CPU; **TensorRT FP16/INT8 deployment with measured cross-precision,
multi-stream, and 36-minute sustained-soak data on Jetson Orin Nano Super
(§5.6)**.

**Out of scope.** *Paper-scale* fine-tuning of YOLOv11 on Unsafe-Net
per-frame bounding-box annotations (the original paper's training
regime) — bounding boxes for Unsafe-Net were not publicly released at
the time of writing. A pilot at 400 frames was attempted and is
reported as a negative result (V10, §4.6); scaling that effort by an
order of magnitude is outlined as future work in Section 6.2.
Multi-SKU edge characterisation (AGX Orin, Orin NX) and live USB-camera
capture in the dashboard are deferred to future work.

**Primary limitation and positioning.** The thesis does not match the
full-bbox paper result (95.81 % macro) and does not claim to. Its best
zero-bbox configuration — the **hybrid (frozen ⊕ detection-finetuned)
frame-gate pipeline** (§4.6.2) — reaches **blind 53/64 (82.8 % top-1,
85.4 % macro)**: macro parity with the published no-bbox baseline (85.78 %)
*even under the blind protocol*, and best-case 86.9–87.7 % macro
*exceeding* it, while remaining below on top-1 (53–54/64 vs 57/64). The
gap to the full-bbox corner is shown to be **data-volume-bound** — the
≈54/64 ceiling holds across 12+ tracks including the paper's exact
frame-gate mechanism replicated at 1,200 bbox frames (§4.6.2), so the
remaining route is paper-scale (≥ 2,262-frame) annotation, not a different
architecture. The thesis's delivered value is the edge-deployment system
(Objective 3), the corrected-mechanism diagnosis, and the hybrid-feature
finding — a Pareto-optimal point (macro-competitive at zero bbox cost,
plus edge), not a single competitive top-1 number.

## 1.6 Thesis Structure

- **Chapter 2 — Related Work** surveys YOLO detectors, video Transformers,
  and industrial safety-monitoring benchmarks, positioning the contribution
  relative to prior art.
- **Chapter 3 — Methodology** describes the Unsafe-Net dataset, the
  YOLO–Transformer architecture, the multi-stream temporal fusion design,
  the training pipeline, and the ensemble / calibration procedure.
- **Chapter 4 — Experiments and Results** documents the 9 experiment tracks,
  the per-class accuracy breakdown, confusion analysis, and the gap
  analysis supporting the data-not-architecture conclusion.
- **Chapter 5 — Edge Deployment** reports the ONNX export, INT8
  quantization, M4 CPU latency benchmarks, multi-stream pipeline
  measurements, and the **measured Jetson Orin Nano Super deployment**
  (cross-precision benchmarks, 4-stream zero-drop throughput + 2-stream
  36.2-minute sustained soak, power-efficiency
  Pareto, live operator dashboard, and threat-model-to-control mapping).
- **Chapter 6 — Conclusion** summarises the four headline findings,
  the limitations (test-set size, single-dataset validation, 400-frame
  bbox pilot, pending retraining ablations, single-SKU edge measurement),
  and the prioritised future-work path centred on paper-scale bbox
  annotation and cross-domain validation.

---

## References

Per-chapter numbering; to be merged into a unified bibliography at final
thesis compilation.

[1] J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You Only Look Once:
    Unified, Real-Time Object Detection," *Proc. IEEE Conf. Computer
    Vision and Pattern Recognition (CVPR)*, 2016, pp. 779–788.
    arXiv:1506.02640. DOI: 10.1109/CVPR.2016.91.

[2] Ultralytics, "YOLO11 — Documentation," 2024. [Online]. Available:
    https://docs.ultralytics.com/models/yolo11/ (accessed 2026-04).

[3] A. Vaswani et al., "Attention Is All You Need," *Advances in Neural
    Information Processing Systems (NeurIPS)*, vol. 30, 2017.
    arXiv:1706.03762.

[4] Y. Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection,"
    *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*,
    2024, pp. 16965–16974. arXiv:2304.08069.

[5] O. Önal and E. Dandıl, "Unsafe-Net: YOLO v4 and ConvLSTM based
    computer vision system for real-time detection of unsafe behaviours in
    workplace," *Multimedia Tools and Applications*, vol. 84, no. 29,
    pp. 34967–34993, 2024. DOI: 10.1007/s11042-024-19276-8.
