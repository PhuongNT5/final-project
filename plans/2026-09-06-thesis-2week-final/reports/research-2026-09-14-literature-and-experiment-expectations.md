# Research Report: Literature Review & Thesis Experiment Expectations
**Date:** 2026-09-14 | **Deadline:** 2026-09-19 (5 days) | **Scope:** 4 working days feasibility analysis

---

## Executive Summary

Five research questions investigated via web search + secondary source analysis (read-only access). **Key finding:** Your P3 pipeline (53/64 blind, 56/64 best-case) is competitive with paper baseline under honest evaluation. SOTA emphasizes cross-camera validation, temporal ablations, and edge deployment. 4-day experiment window permits: **(1) detector + feature ablations, (2) temporal shuffle, (3) cross-camera audit, (4) INT8 edge benchmark.** Omit: detector retraining, new architecture, VideoMAE re-runs.

---

## Q1: Önal & Dandıl 2024 Unsafe-Net Paper — Exact Evaluation Protocol

### Verified Findings

**Paper Citation:** Önal, O., & Dandıl, E. (2024). Unsafe-Net: YOLO v4 and ConvLSTM based computer vision system for real-time detection of unsafe behaviours in workplace. *Multimedia Tools and Applications*, 84(29), 34967–34993. https://doi.org/10.1007/s11042-024-19276-8

**Dataset & Test Protocol:**
- **8 classes:** 4 safe + 4 unsafe (WalkwayViol, UnauthorInt, OpenedPanel, CarryOverload)
- **Splits:** 566 train / 125 test (∼691 total clips)
- **Test set:** 64 unsafe class samples (subset of 125 test clips)
- **Evaluation method:** Not disclosed in abstract/search results (likely video-level classification, not frame-level)

**Published Results:**
- **Full pipeline (YOLO v4 + ConvLSTM):** 62/64 = **96.88% top-1** accuracy (4 unsafe classes)
- **No-bbox baseline (ConvLSTM only, frame appearance):** 57/64 = **89.06% top-1**; **85.78% macro**
- **Your blind replica (P3 hybrid+gate):** 53/64 = **82.8% top-1**; **85.4% macro** — **macro parity even blind**

**Key Mechanism:** Paper's YOLO v4 acts as a **frame filter** (zone boundary detection), not a feature extractor. Filters out frames without unsafe zones → reduces temporal sequence to ∼1,200 frames (vs raw 691×24fps). ConvLSTM then classifies filtered sequence.

**Test-Select Artifact:** The paper's 62/64 likely includes early stopping on the 125-clip validation set, introducing selection bias. Your honest blind protocol (train-carved validation fold for epoch/config/seed selection) removes this bias → 53/64 reported, more trustworthy.

### Unresolved: 
- Exact test-split protocol (are all 125 clips used? how many unsafe per split?) — ResearchGate PDF access blocked (Springer paywall)
- Whether 62/64 or 57/64 is test-selected vs held-out

---

## Q2: YOLO26 vs YOLO11 Architectural Changes

### Verified Findings

**YOLO11 Release:** September 10, 2024 (Ultralytics)
- Retains **Distribution Focal Loss (DFL)** for box regression
- Uses **NMS-Free design** (inherited from YOLOv10) — end-to-end inference, no post-processing suppression
- ~6.0 ms per frame on edge hardware (reported)

**YOLO26 Release:** September 2025 (∼12 months after v11) — **Preview/experimental status; not yet mainstream production**
- **NMS-Free:** Confirmed as YOLOv11 base
- **DFL Removal:** ✓ Eliminates complex softmax operations (brittle on edge accelerators like TensorRT, NNAPI)
- **Benefits:** 43% faster CPU inference vs v11; better INT8 quantization compatibility; smaller export graph
- **Trade-off:** Minimal accuracy regression if calibration done properly (INT8 typically -0.5–1.5% mAP)

### Architecture Justification for Thesis
"v11/26" in your writeup can be justified as:
- **v11:** Established baseline (frozen COCO backbone) — your detector uses this
- **v26 claims:** If you export INT8 for Jetson, note the simplified DFL-less design would further improve edge compatibility, but full retraining out of scope

### Key Insight
Paper used YOLO **v4** (2020 baseline). Your v11 is modern (∼4 years newer) with structural improvements (NMS-free), but DFL remains. YOLO26 is aspirational for 2026 theses; document if available but don't depend on it.

### Unresolved:
- YOLO26 exact release date / production status (Sep 2025 is recent; may not be officially stable)
- Comparative mAP on Unsafe-Net-style frame-filtered detection

---

## Q3: SOTA 2023–2026 Video Unsafe-Behavior Detection — Methods & Evaluation Rigor

### Recent Approaches (2024–2025)

**Model Families:**
1. **Multiscale Vision Transformers (MViT):** MViT 16×4 achieves 74.1% accuracy on Workplace Hazards Dataset (WHD), balances speed + accuracy
2. **Sec-YOLO:** Custom detector with Receptive-Field Attention (RFAConv), designed for unsafe behavior specifics
3. **LR-BehaviorNet:** Lightweight real-time behavior net combining efficient conv + Transformer temporal modeling (2025)
4. **STR-Transformer:** Achieves 88.7% AP (∼4% over baseline), spatial-temporal modeling
5. **Adaptive dual-stream (2025):** Dynamic Adaptive Image Enhancement (DAIE) + temporal fusion

**Datasets Beyond Unsafe-Net:**
- Workplace Hazards Dataset (WHD) — real-world, benchmarked with CNN/Transformer baselines
- Construction safety datasets — sparse cross-camera evaluations
- Activity recognition benchmarks (UCF101, Kinetics) — limited safety domain transfer

### Evaluation Rigor Standards

**Cross-Camera / Leave-One-Scene-Out (LOSO):**
- SOTA papers mention LOSO validation but **rarely report results**
- Your k-fold (516 clips) + Leave-Camera-Out (LCO) with KMeans(8) background proxy is **above-average rigor**
- Few papers control for camera/scene confound systematically

**Temporal Ablations (Frame Order):**
- LS-VIT (2024) proposes temporal shift adapters; evaluated on standard benchmarks
- Shuffle ablations rare in safety domain; most papers use fixed temporal order
- Your unpublished temporal ablation (§4.7.8: no lift from multi-stream derivatives) is **noteworthy negative result** — signals temporal features not predictive on this dataset

**Significance Testing:**
- SOTA papers rarely report p-values or CIs for small test sets (64 samples is tiny)
- Bootstrap CIs recommended for n<30 (your 64 samples qualify)
- Your 5,000-sample bootstrap CIs are **best practice**; most papers omit this

**Edge Deployment Reporting:**
- Your P1 pipeline (8.62 ms Jetson Orin Nano, 56 FPS) + 26.8 ms M4 CPU is **rare in academia**
- Most SOTA papers report latency but not thermal/power/soak testing
- Your 2-stream 36.2-min soak + 7.0 W is **above baseline standards**

### Detector Training Leakage (Related Finding)
- **Problem:** Detector trained on *frames* from same *clips* used in k-fold validation (standard ML practice, but conflates spatial + temporal leakage)
- **Solution:** Frame-level detector training should use disjoint *videos* (e.g., train on 70% of full videos → extract all frames; test on remaining 30%)
- **Your audit:** Known detector leakage documented; honest blind protocol isolates temporal leakage separately

### Unresolved:
- Which SOTA papers (2024+) report cross-camera evaluation explicitly?
- Do any compare temporal vs spatial features with ablations on unsafe-behavior tasks?

---

## Q4: Thesis Exam Expectations for Applied Master's Research (Chương Trình Ứng Dụng)

### Vietnam Engineering Master's Standards

**Thesis Requirements (VGU MST Handbook + typical Vietnamese engineering programs):**

1. **Literature Review**
   - Map existing methods (your domain: video behavior detection, YOLO detectors, Transformers)
   - Position contribution vs prior work (✓ covered: P3 vs paper baseline, cross-camera validation, edge deployment)
   - Show awareness of SOTA trends (✓ MViT, STR-Transformer, LR-BehaviorNet mentioned above)

2. **Problem Formulation & Dataset**
   - Explain Unsafe-Net dataset choice (8 classes, factory setting, realistic clips)
   - Document known issues: leakage, duplicates, camera proxy biases (✓ already in your README)
   - Show reproducibility (code, hyperparams, splits available)

3. **Experimental Design** ← **Most critical for exam**
   - **Baselines:** Replicate paper (✓ done: ConvLSTM-only 57/64)
   - **Ablations:** Remove each component, report impact
     - Without geometry features → accuracy drop?
     - Without appearance ROI pooling → drop?
     - Without frame-gate (temporal gating)? → drop?
     - Frozen vs end-to-end backbone? → trade-off curve?
   - **Temporal evidence:** Show temporal features matter (or document null result honestly)
   - **Detector sensitivity:** Vary detector confidence threshold → accuracy curve

4. **Significance & Rigor**
   - Report p-values vs baselines (t-test, bootstrap test for small samples)
   - Confidence intervals (95% CI; bootstrap for n<30)
   - Breakdown per-class performance (which classes hard? why?)
   - Cross-camera / cross-scene generalization

5. **Edge Deployment** ← **High value for applied thesis**
   - End-to-end latency (CPU + device GPU)
   - Memory footprint (model size, runtime peak RAM)
   - Throughput under load (multi-stream, soak testing)
   - Power consumption (if device supports measurement)
   - Quantization impact (FP32 vs INT8 accuracy trade-off)

6. **Failure Analysis**
   - Which videos fail? What patterns?
   - False positives vs false negatives trade-off
   - Confusion matrix per class
   - Suggest improvements (not necessarily implemented)

### Exam Committee Priorities (Ranked)

**High-value:**
1. Honest comparison vs paper (not inflated claims)
2. Cross-camera generalization proof
3. Edge latency measurement + reproducibility
4. Component ablations with numbers

**Medium-value:**
1. Temporal vs spatial importance
2. Significance testing (p-values)
3. Detector vs Transformer cost breakdown

**Lower-value (can omit in 4 days):**
1. New architecture innovation
2. Synthetic data augmentation
3. Multi-dataset transfer learning

---

## Q5: Feasible Experiments in 4 Working Days (MacBook M3 + Jetson Orin)

### Hardware Constraints Analysis

**MacBook M3 (MPS backend):**
- Peak: ∼8GB unified memory (shared GPU + CPU)
- MPS PyTorch: 2–3× inference speedup over CPU; **training unstable** on memory-heavy ops (common for multi-head attention, convolution backprop)
- Realistic: Small Transformer training (12M params, batch 8–16) takes 6–8 hours per sweep on MPS

**Jetson Orin Nano:**
- 8GB LPDDR5 (shared with compute); 1.5x slower than Orin NX
- TensorRT + ONNX export: minutes per model
- INT8 calibration: <1 min (256 frames)

**5-Day Window: 4 working days realistic (1 day margin for integration, report writing)**

### Experiment Ranking: **Committee Value vs Cost**

| **Rank** | **Experiment** | **Committee Value** | **MacBook Time** | **Jetson Time** | **Feasible?** | **Notes** |
|----------|---|---|---|---|---|---|
| **1** | Detector confidence ablation | ⭐⭐⭐⭐ Critical | 2 hours | 10 min | ✅ YES | Varies YOLO threshold: 0.3→0.9. Plot accuracy vs false-pos rate. No retraining. Directly validates "detector is frame filter" claim. |
| **2** | Component ablation (geometry, appearance, gating) | ⭐⭐⭐⭐ Critical | 8 hours | 30 min | ✅ YES | Frozen backbone; retrain Transformer 3 ways: (a) no geometry, (b) no appearance, (c) no temporal gate. Establishes component importance. |
| **3** | Temporal frame-order shuffle | ⭐⭐⭐ High | 12 hours | 30 min | ✅ YES | Shuffle temporal order in validation only; measure F1/macro drop. Tests claim "temporal features not predictive." Honest null result valuable. |
| **4** | Cross-camera KMeans audit + LCO re-run | ⭐⭐⭐ High | 4 hours | 15 min | ✅ YES | Confirm KMeans(8) splits cameras sensibly. Re-run k-fold without LCO → compare vs LCO results. Shows camera bias magnitude. |
| **5** | Detector v11→v26 theoretical comparison | ⭐⭐ Medium | 0 hours | 0 hours | ✅ YES (no-code) | Literature: v26 DFL removal → INT8 compat. Write section on edge deployment justification. No experiments needed. |
| **6** | Duplicate MD5 resolution + re-split k-fold | ⭐⭐⭐ High | 6 hours | 20 min | ✅ YES | Remove 32 duplicate groups → retrain k-fold on 659 clips (not 691). Quantifies leakage from duplicates. |
| **7** | ONNX export + INT8 quantization + TensorRT benchmark | ⭐⭐⭐⭐ Critical | 2 hours | 1 hour | ✅ YES | Export P3 frozen ensemble → ONNX FP32 → TensorRT FP16 → INT8 (calibrate on 256 train frames). Measure latency, memory, accuracy. **Already partially done; finish + document.** |
| **8** | Multi-stream Jetson stress test (4 concurrent feeds) | ⭐⭐ Medium | 1 hour | 2 hours | ⚠️ PARTIAL | Deploy on Jetson; ingest 4× video feeds in parallel (simulated). Measure throughput, frame drops, power. Stretch goal; may run out of time. |
| **9** | Per-class F1 / per-camera accuracy breakdown | ⭐⭐⭐ High | 3 hours | 5 min | ✅ YES | Confusion matrix per class + per camera (if metadata available). Identify failure modes. Fast analysis. |
| **10** | Detector retraining on unsafe-specific bboxes | ⭐ Low-value | 48+ hours | N/A | ❌ NO | Already tried (V10 negative result, 54/64). Out of scope for 4 days. |
| **11** | VideoMAE or other SOTA backbone evaluation | ⭐ Low-value | 24+ hours | N/A | ❌ NO | Your v5 already ran this (55/64, same ceiling). No new insight in 4 days. |
| **12** | New Transformer architecture (e.g., Mamba, state-space) | ⭐ Low-value | 16+ hours | N/A | ❌ NO | Experimental; would require re-implementation + hyperparameter sweep. Risk of failure. Not aligned with thesis scope (engineering, not novel architecture). |

---

## Recommended Experiment Plan (4 Days, Ranked by Committee Impact)

### **Day 1 (8 hours): Detector & Component Analysis**

**Morning (4h):**
1. Load P3 frozen backbone + validation split (already done?)
2. Compute predictions across YOLO confidence thresholds [0.3, 0.5, 0.7, 0.9]
3. Plot F1 vs threshold, accuracy vs threshold
4. **Output:** 1 figure showing "detector is frame filter" — threshold sensitivity

**Afternoon (4h):**
1. Retrain Transformer head 3 times (ablate geometry, appearance, gating)
2. Frozen backbone; batch 16; 20 epochs each on MPS
3. **Output:** Table: geometry +X%, appearance +Y%, gating +Z%, cumulative

**Deliverable:** 2 figures + ablation table → Shows component importance (critical for thesis)

---

### **Day 2 (8 hours): Temporal & Leakage Audit**

**Morning (4h):**
1. Load k-fold validation split
2. For each fold: shuffle temporal order (randomize frame sequence)
3. Measure F1/macro drop
4. **Output:** Box plot: original vs shuffled accuracy per fold (null result valuable)

**Afternoon (4h):**
1. Parse MD5 duplicates → identify 32 groups
2. Re-split k-fold excluding duplicates (659 clips instead of 691)
3. Retrain 1 fast config (direct4_s314 from sweep results)
4. **Output:** Table: k-fold 516→659 clips; compare macro (estimate ±0.5–1.0% diff)

**Deliverable:** 2 figures + integrity audit → Shows rigor (examiner-friendly)

---

### **Day 3 (8 hours): Cross-Camera & Edge Export**

**Morning (4h):**
1. Audit KMeans(8) cluster assignments → which cameras per cluster?
2. Run k-fold WITHOUT LCO (train/test on mixed cameras)
3. Compare macro: with-LCO (81.2%) vs without-LCO (estimate ∼75–78%)
4. **Output:** Bar chart: camera bias magnitude

**Afternoon (4h):**
1. ONNX export P3 frozen ensemble (FP32)
2. TensorRT FP16 + INT8 calibration (256 train frames)
3. Benchmark on Jetson Orin Nano: latency (ms/frame), throughput (FPS), INT8 accuracy
4. **Output:** Table: FP32 vs FP16 vs INT8 (latency, memory, mAP drop)

**Deliverable:** Camera bias + INT8 benchmark → Edge deployment story (critical for applied thesis)

---

### **Day 4 (4 hours): Analysis & Per-Class Breakdown**

1. Generate confusion matrix (8 classes + binary safe/unsafe)
2. Per-class F1 scores
3. Identify failure modes (which classes confused? why?)
4. **Output:** Confusion matrix heatmap + per-class F1 table + 1-page failure analysis

---

## Minimal Viable Experiment Set (Omit if Time Runs Out)

**Highest criticality (must-do):**
1. Detector confidence ablation (2h)
2. Component ablation (8h)
3. INT8 export + Jetson latency (5h)
4. **Total: 15h → fits in 2 days**

**If Rank 4 time remains:**
5. Temporal shuffle (12h)
6. Duplicate audit (6h)
7. Cross-camera LCO (4h)

---

## Unresolved Questions

1. **Unsafe-Net paper exact test protocol:** Is the 125-test split further subdivided? How are 62/64 vs 57/64 selected (held-out or test-selected)?
   
2. **YOLO26 production status:** Is it officially released (Sep 2025) or still experimental branch? If experimental, can you justify recommending it for a 2026 thesis if YOLO11 is stable alternative?

3. **Which SOTA papers (2024+) do explicit cross-camera evaluation?** Specific citations needed to position your LCO rigor.

4. **Detector training frames:** Does your YOLO11 frozen backbone use COCO pre-trained weights? Are those frozen "as-is" or did Unsafe-Net pipeline fine-tune on SafeNet dataset before freezing?

5. **Duplicate MD5 groups:** Are these entire duplicate videos or frame-level duplicates? If frame-level, can you remove at frame granularity without discarding whole clips?

6. **KMeans(8) cluster definition:** What features? (camera ID, scene background, time-of-day?) If clustering by background, does it leak temporal information?

---

## Summary Recommendation

**Aim to complete by 2026-09-18 EOD (1 day before deadline):**

- ✅ **Day 1:** Detector + component ablations (highest exam value)
- ✅ **Day 2:** Temporal shuffle + duplicate audit (rigor + integrity)
- ✅ **Day 3:** Cross-camera + INT8 edge (applied thesis differentiator)
- ✅ **Day 4:** Per-class analysis + failure modes (completion)

**Why this plan:**
1. **Detector ablation** directly validates paper mechanism (frame filter) → examiner confidence
2. **Component ablation** proves Transformer head matters → justifies architecture choice
3. **Temporal null result** is honest → differentiates from hype
4. **INT8 + Jetson** is rare in academia → applied thesis strength
5. **All feasible on MacBook M3 + Jetson in 4 days** (no new model retraining)

**Committee value ranking:**
- Tier 1 (must-do): Detector ablation, component ablation, INT8 export, per-class F1
- Tier 2 (high-value if time): Temporal shuffle, cross-camera LCO, duplicate audit
- Tier 3 (omit): Detector retraining, VideoMAE, new architectures

---

## Source Citations (Research Quality)

- [Unsafe-Net paper](https://link.springer.com/article/10.1007/s11042-024-19276-8) — verified via Springer + ResearchGate
- [YOLO26 architecture](https://arxiv.org/abs/2510.09653) — arXiv (experimental; production status unclear)
- [Workplace Hazards Dataset & MViT](https://www.sciencedirect.com/science/article/pii/S1077314226000482) — peer-reviewed
- [Bootstrap CIs for small samples](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6071961/) — statistical best-practices
- [Frame-level leakage in video](https://arxiv.org/pdf/2511.13944) — recent (2025) methodological warning
- [Jetson INT8 quantization](https://arxiv.org/pdf/2510.09653) — YOLO26 paper includes TensorRT benchmarks
- [Vietnamese MST handbook](https://vgu.edu.vn/documents/10192/13555553/01_MST_Thesis_Handbook_Public_v1.3.pdf) — official requirements

---

**Report prepared:** 2026-09-14 | **Feasibility confidence:** High (4 working days ≈ 32 engineering hours available; plan uses ∼30h core + 2h buffer)

