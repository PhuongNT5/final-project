# Research Report: Concrete Techniques for Unsafe-Net Binary Classification Cross-Camera Generalization
**Status:** Complete literature synthesis + prioritized recommendations  
**Date:** 2026-09-19  
**Scope:** 1-week sprint, 691 clips, PyTorch + MPS, ~67-70% LCO ceiling target  

---

## Executive Summary: The Core Problem & Viable Pathways

**Measured Diagnosis:** Current E3 DeepSets achieves 69.6% LCO balanced accuracy, but Safe-class recall collapses to 42–51% (models default to "Unsafe"). K-fold shows 83.6–84.6% (scene memorization). Scene co-clustering ranges 0.12–1.00 across safe/unsafe counterpart pairs. The ceiling is data-bound AND likely requires active scene/background disentanglement.

**Strategic Insight:** The 18k-param DeepSets generalization > 299k-param RelationTransformer aligns with implicit bias + Occam's razor literature — smaller models have inbuilt regularization toward simpler, more generalizable solutions. Adding raw capacity will worsen LCO unless coupled with explicit domain-invariance mechanisms.

**Viable 1-Week Pathways:**
1. **Low-effort, high-confidence:** Scene-stratified training + background mixing → directly attacks scene memorization, proven effective (CoMix, NeurIPS 2021).
2. **Medium-effort, diagnostic value:** Motion-only representation experiments → if geometry alone fails, isolating motion establishes why.
3. **Medium-effort, defensible contribution:** Diagnostic benchmark (LCO protocol refinement) → if numbers stay modest, a rigorous protocol + negative-result analysis is publishable.
4. **Higher-effort, incremental gain:** Adversarial camera ID + stratified focal loss → targets class imbalance + camera shift simultaneously.

**Unresolved Question:** Whether Safe-class recall collapse is fundamentally a label ambiguity issue (coders disagree on frame-level safety) or model capacity/data issue. Recommend label audit as prerequisite for all below work.

---

## 1. Detector Layer: YOLO26 vs. Alternatives

### YOLO26 Status (Released Jan 2026)

**What It Is:** Unified real-time detection family, NMS-free end-to-end by default (dual-head supports both), removes DFL (complex + brittle under quantization), trains with MuSGD + ProgLoss + STAL.

**Performance:** +1.6–2.8 AP over YOLOv11 at matched scales; YOLOv26n runs 43% faster than YOLOv11n on CPU ONNX. Removes NMS post-processing overhead.

**For Unsafe-Net Entity Tokens:**
- **Mechanism:** Better entity localization → tighter bounding boxes → higher-quality relation tokens (box geometry is 6/74 features, but anchor for spatial context).
- **Why It Should Help:** Entity tokens depend on detector bbox quality. YOLO26 improves small-object detection (ProgLoss + STAL) and speed (important for video pipelines).
- **Expected Effect:** +0.5–1.5% LCO balanced accuracy (indirect gain; detector quality is not the bottleneck; scene/appearance is).
- **Implementation Cost:** ~2 hours. Ultralytics API stable; swap YOLOv11s to YOLO26s in inference, re-extract relation tokens.
- **Risk:** Marginal gain if scene memorization is the driver. Could waste implementation time.

### Temporal Entity Tracking (ByteTrack / BoT-SORT)

**Mechanism:** Link detections across frames → persistent entity ID within video → aggregate temporal identity instead of treating each frame independently.

**Why It Should Help THIS Failure Mode:** 
- Current pipeline: 16–32 frames of entity slots (up to 16 entities per frame), but NO temporal identity — slot 5 at t=0 ≠ slot 5 at t=1.
- Problem: Spatial entity tokens collapse temporal continuity; Transformer head sees jumbled entity sequences.
- ByteTrack tracks with IoU + confidence; BoT-SORT adds appearance features + camera motion compensation.

**Expected Effect:** +1–3% LCO if entity consistency is indeed lost; negligible if slots already capture temporal patterns implicitly.

**Implementation Cost:** ~4 hours.
- Use Ultralytics tracking API (YOLO26 supports BoT-SORT by default).
- Store tracked entity IDs alongside bboxes.
- Modify relation token pipeline to reorder/mask slots by track ID continuity.
- Risk: BoT-SORT ReID features require model training; ByteTrack alone may under-cluster in crowded scenes. Test on 20 clips first.

**Recommendation:** Low priority unless manual inspection shows entity slot churn is breaking temporal patterns.

### YOLO-Pose + STGCN Skeleton

**Mechanism:** Extract 17-joint skeleton → feed to STGCN (spatio-temporal graph convolution over skeleton edges). STGCN explicitly captures human kinematic constraints.

**Why It Should Help:** Skeleton is pose-invariant, appearance-agnostic, camera-robust. If unsafe behaviors are kinematic (reaching, climbing, falling), skeleton-only could decouple from background.

**Expected Effect:** +2–5% LCO if unsafe behavior is strongly kinematic; 0–1% if appearance/context dominates (e.g., operator near panel = unsafe regardless of pose).

**Implementation Cost:** ~6 hours.
- Extract YOLO-Pose skeletons per frame.
- Convert to graph (joints=nodes, skeleton edges).
- Add STGCN module in parallel to relation tokens (ensemble or early fusion).
- Requires new model definition (~50 lines PyTorch).

**Risk:** High — if unsafe behavior is contextual (panel open = unsafe), skeleton alone will fail. Labor-intensive if requires architecture redesign. Recommend as Phase 2 if Scene Stratification (below) plateaus.

### Open-Vocab Detection (YOLO-World)

**Status:** Emerging; ultralytics integration available.  
**Mechanism:** Zero-shot detection via text prompts (e.g., "open panel", "person climbing").  
**For Unsafe-Net:** Could extract task-specific attributes (panel state, person proximity) without re-annotation.

**Expected Effect:** +1–2% if attributes are class-correlated; 0% if ground-truth safety is orthogonal.

**Implementation Cost:** ~3 hours (inference only, no training).

**Risk:** Attribute extraction adds latency; requires careful prompt engineering; prone to false-positive "open panel" detections in crowded scenes.

**Recommendation:** Defer to Phase 2 if main techniques plateau.

---

## 2. Background & Scene Invariance: The Decisive Axis

This is WHERE the 67→70% LCO ceiling must be broken. Scene co-clustering analysis shows 100% unsafe clusters exist (cluster 101 = all unsafe); Safe/Unsafe counterparts range 0.12–1.00 similarity. **Current model learns scene more than behavior.**

### Background Mixing / CoMix (HIGH CONFIDENCE)

**Source:** "Contrast and Mix: Temporal Contrastive Video Domain Adaptation with Background Mixing" (NeurIPS 2021, IBM Research).

**Mechanism:**
1. Identify source/target domain pairs (e.g., source=outdoor, target=indoor).
2. Extract background frames from target domain videos.
3. Paste source-domain action videos onto target-domain backgrounds → synthetic videos with source action + target appearance.
4. Train jointly on real + synthetic videos with temporal contrastive learning.

**For Unsafe-Net:** 
- Scene clusters as proxy for "domains" (cluster 1 vs. cluster 5 = different backgrounds).
- Randomly insert Safe-class actors into Unsafe-dominant backgrounds (cluster 101) and vice versa.
- Forces model to rely on actor behavior, not background.

**Expected Effect:** +2–4% LCO balanced accuracy; +5–15 percentage points Safe-class recall (directly targets class collapse).

**Implementation Cost:** ~6–8 hours.
1. Segment actors (bounding boxes) from relation tokens (already extracted).
2. Build background banks per cluster using frame averages or INPAINTING (10 min).
3. Paste strategy: random location, blend edges with Poisson blending or alpha channel.
4. Double dataset size with augmented videos; retrain E3 head (2–3 hours).

**Risk:**  
- Inpainting artifacts could inject spurious visual cues.
- Mitigation: Use simple alpha-blended paste, test on 50 clips before full run.
- Requires careful label accounting (mixed videos get blended labels, e.g., 0.7×Safe + 0.3×Unsafe).

**Why It Should Help:**
- Directly breaks scene-behavior spurious correlation.
- Proven effective in domain adaptation (CoMix SOTA on UCF-to-HMDB transfer).
- Low-data friendly (no new model architecture, just augmentation).

**Priority: HIGHEST. Implement first.**

### Scene-Debiasing Losses (SOAR / DEVIAS)

**Sources:** 
- SOAR: Scene-debiasing Open-set Action Recognition (ICCV 2023).
- DEVIAS: Learning Disentangled Video Representations (ECCV 2024).

**Mechanism:**
1. **SOAR:** Adversarial scene reconstruction head + adaptive adversarial scene classification. Encourages feature extractor to forget scene while preserving action.
2. **DEVIAS:** Encoder-decoder with disentangled action + scene latent factors. Decode action from action-only bottleneck.

**For Unsafe-Net:**
- Add auxiliary scene classifier predicting cluster ID (8 clusters).
- Gradient reversal layer (GRL) on scene classifier: features optimized to confuse scene predictor.
- Secondary loss: scene reconstruction MSE on auxiliary decoder; backprop GRL to main feature head.

**Expected Effect:** +1–3% LCO if GRL is effective; comparable to Background Mixing but architecture-heavy.

**Implementation Cost:** ~8–10 hours.
- Define scene classifier (2-layer MLP, 100 dims, 8 outputs).
- Insert GRL layer (Gradient Reversal Layer, ~5 lines PyTorch, multiplies grad by -λ).
- Add scene reconstruction decoder if doing full DEVIAS (adds complexity; start with GRL alone).
- Hyperparameter tuning: λ schedule (init 0.1, anneal to 1.0 over epochs).

**Risk:**  
- GRL can be unstable; requires careful λ tuning and batch balancing.
- If 8 clusters are semantic (Safe vs. Unsafe clusters mixed), adversarial loss may wash out true signal.
- Empirical: GRL rarely beats simpler domain-shift methods in low-data regimes (group-dro/irm literature shows mixed results).

**Recommendation:** Implement after Background Mixing succeeds. Use as fallback if Background Mixing + motion experiments plateau.

### Coordinate Normalization & Entity-Centric Representations

**Mechanism:** Normalize entity box coordinates to [0, 1] or center-relative. Decouple box position from camera viewpoint.

**For Unsafe-Net:** 
- Currently relation tokens include absolute box coordinates (likely in pixel space).
- Normalize to video frame dimensions → position-agnostic.
- Further: use centered-coordinate system (center of video = 0, edges = ±1) → shifts background contribution.

**Expected Effect:** +0.5–1.5% LCO (small gain; coordinates are one feature among 74).

**Implementation Cost:** ~1 hour (modify relation token extraction in `build_relation_tokens.py`).

**Risk:** Minimal; worst case is no change.

**Recommendation:** Do this as REQUIRED BASELINE before any other experiment. Measure k-fold vs. LCO impact to confirm scene leakage.

---

## 3. Relational & Temporal Architectures for Small Video Data

**Empirical Puzzle:** E3 (18k DeepSets) > E0 (299k Transformer) on LCO. Literature explanation: Implicit bias of small models toward simpler, generalizable functions (Occam's razor in deep learning).

### Why Small Models Win Here

**Sources:** 
- "Deep neural networks have an inbuilt Occam's razor" (Nature Commun. 2024).
- "On the Implicit Bias in Deep-Learning Algorithms" (arXiv 2208.12591).

**Insight:** Over-parameterized models reach a "double-descent" point where too much capacity memorizes spurious features (scene, camera artifacts). DeepSets, by design, processes entities as a set (permutation-invariant aggregation) — this structural inductive bias acts as implicit regularization.

**Actionable Finding:** Don't increase model size. Instead, ADD STRUCTURAL INDUCTIVE BIASES (permutation invariance, equivariance, entity-centric) while keeping capacity <50k params.

### Slot Attention for Object-Centric Learning

**Sources:** Recent 2024–2025 papers; Dual-State Slot Attention, MetaSlot, FORLA.

**Mechanism:** Each slot = latent code; iterative attention between slots and frames. Slots learn to bind to individual entities (person, object) across frames.

**For Unsafe-Net:**
- Replace fixed 16-entity slots with learned slot attention.
- Slot attention naturally handles variable entity counts and temporal tracking.
- Compute cross-frame attention per slot → slot consistency metric for temporal loss.

**Expected Effect:** +1–2% LCO (improves entity consistency; modest gain if current slots already effective).

**Implementation Cost:** ~10–12 hours.
- Implement slot attention module (~80 lines PyTorch).
- Integrate into E3 head (modify relation-token aggregation).
- Add temporal consistency loss (MSE between slot embeddings t=i and t=i+1).
- Requires hyperparameter tuning (num_slots, iters, dim).

**Risk:**  
- Slot attention is unsupervised; may bind to spurious features (shadows, background texture).
- Unstable training; requires careful learning rate and gradient clipping.
- Best results on large clean datasets (CleVR, Multi-Object Tracking); limited low-data evidence.

**Recommendation:** Implement only if Background Mixing + motion experiments plateau AND time permits. Medium priority.

### Motion-Only Representation (Diagnostic Value)

**Sources:** "Appearance-Free Action Recognition", optical flow literature (Sevilla-Lara 2018, Sun 2018).

**Mechanism:**
1. Extract optical flow (frame-to-frame pixel motion) as I_flow ∈ R^(H×W×2).
2. Compute relation tokens from flow-only (geometry + motion, NO appearance RGB).
3. Compare E3 head performance on (RGB-based) vs. (flow-only) relation tokens.

**Why It Should Help THIS FAILURE MODE:**
- If LCO fails because scene/appearance dominates, motion-only will show larger drop than geometry-only.
- Diagnostic: if motion-only recovers 50%+ of LCO accuracy, unsafe behavior is kinematic; if <40%, it's contextual/appearance.

**Expected Effect (diagnostic):**
- If motion recovers 65%+ LCO → behavior is kinematic, pursue skeleton/motion paths.
- If motion recovers <50% → behavior is contextual, focus on GRL/scene debiasing.

**Implementation Cost:** ~5–6 hours.
1. Compute optical flow per video (use openCV / torchvision.transforms.functional).
2. Modify relation-token builder to extract from flow instead of RGB geometry.
3. Run E3 inference on 50 test clips (diagnostic), then full LCO evaluation.

**Risk:** Flow computation is expensive (CPU-bound); may not fit on M4 MPS easily. Compute on CPU/GPU if available.

**Recommendation:** HIGH priority as diagnostic. Results directly inform next phase (if motion recovers truth, pursue skeleton; if not, pursue scene debiasing). ~6 hours for diagnostic confidence.

---

## 4. Pretrained Video Models & VLM Under Camera Shift

### VideoMAE (Low-Data Regime)

**Status:** Masked autoencoding, strong on UCF101/HMDB51 (<4k clips per dataset). Outperforms supervised methods in low-data.

**For Unsafe-Net:**
- VideoMAE-base pretrained on Kinetics-400 or UCF101 → frozen backbone.
- Extract frame-level features via forward pass (no re-training, linear probe on top).
- Advantages: data-efficient, proven low-data generalization.

**Expected Effect:** +0–2% LCO (VideoMAE-base was already tried on 4-class task; hit same ceiling ~56%). Likely because 691 clips is still too small for fine-tuning; frozen probe may miss Unsafe-Net-specific patterns.

**Implementation Cost:** ~3 hours (load pretrained, run inference, verify feature dimensions).

**Risk:** Ceiling known to be ~54–56% on 4-class; binary task may escape, but no guarantee. Sunk cost if doesn't work.

**Recommendation:** Low priority. Skip if Background Mixing + motion diagnostics show promise. Defer to Phase 2 if all else fails.

### Frozen CLIP / SigLIP2 Features + Few-Shot Adaptation

**Sources:** 
- "Learning to Adapt Frozen CLIP for Few-Shot Test-Time Domain Adaptation" (ICLR 2025).
- SigLIP-2: Multilingual vision-language encoders (Feb 2025).

**Mechanism:**
1. Extract frame-level features from frozen SigLIP2 (large multimodal CLIP variant).
2. Linear probe on unsafe-net training data (very cheap, 1–2 hours on 566 clips).
3. Test-time domain adaptation (TTA): fine-tune probe on unlabeled test cluster frames.

**Why It Should Help:**
- SigLIP2 trained on 3B diverse examples; strong OOD generalization.
- Frozen backbone prevents overfitting to spurious scene features.
- Few-shot adaptation leverages test-time unlabeled data (LCO test split is available).

**Expected Effect:** +2–4% LCO over E3 baseline (speculative; depends on SigLIP2 alignment with safety concepts).

**Implementation Cost:** ~5–7 hours.
1. Download SigLIP2 checkpoint (Hugging Face, ~1GB).
2. Extract per-frame features (T × 768 or T × 1024 dims).
3. Average over time or use simple MLP aggregator.
4. Train linear probe on train split.
5. TTA: fine-tune on unlabeled test frames (optional, requires impl. of simple TA algorithm).

**Risk:**  
- SigLIP2 may not encode safety concepts (designed for general vision-language tasks).
- Frozen features may lose fine-grained motion/temporal information.
- TTA requires careful hyperparameter tuning to avoid overfitting to test artifacts.

**Recommendation:** MEDIUM priority. Implement after Background Mixing baseline to understand if visual foundation model helps. ~5 hours for linear-probe baseline; TTA adds another 3–4 hours.

### VLM as Judge (Zero-Shot Safety Prompts)

**Sources:**
- "Unlocking Vision-Language Models for Video Anomaly Detection via Fine-Grained Prompting" (Oct 2024).
- "MonitorVLM" safety detection for mining (Oct 2024).
- **Warning:** "Prompt-Induced Score Variance in Zero-Shot Binary Vision-Language Safety Classification" (Jun 2025) shows semantic equivalent prompts give different unsafe probabilities.

**Mechanism:**
- Query pretrained VLM (e.g., LLaVA, GPT-4V API) on sampled frames: *"Is a worker performing an unsafe action in this frame? Answer: yes/no."*
- Aggregate frame-level judgments (majority vote or confidence threshold).
- Use as auxiliary weak label or evaluation metric.

**Why It Should Help:**
- VLM may capture safety semantics not in training labels.
- Zero-shot = no additional labeled data required.

**Risk (CRITICAL):**
- Prompt variance destabilizes probability scores (2025 finding).
- VLM designed for general vision, not domain-specific safety.
- Adds dependency on external API (GPT-4V cost, latency, rate limits).
- Empirical: VLMs struggle with temporal reasoning in videos; sample single frames loses motion.

**Expected Effect:** -1 to +2% (likely negative due to prompt variance and frame-level sampling).

**Recommendation:** SKIP for 1-week sprint. Use only as post-hoc evaluation diagnostic (e.g., "VLM judges 70% of LCO test set as unsafe — model aligns with VLM judgment").

---

## 5. Making Weak Results Defensible: Benchmark & Diagnostic Contributions

If LCO stays 68–72%, absolute numbers are "weak" by SOTA standards. But THREE published pathways exist for defensible contributions:

### (A) Diagnostic Benchmark: LCO-Protocol Refinement

**Precedent:** 
- "Video-Oasis: Rethinking Evaluation of Video Understanding" (2024) — decouples visual/temporal cues, shows 54% of benchmark samples solvable without video.
- "Do Video-LLMs Actually Watch?" (2025) — diagnostic failures (character tracking) masked by option bias.
- "EgoSchema" (NeurIPS 2023) — very-long-form diagnostic benchmark.

**For Unsafe-Net:**
- Formalize LCO-8 as a benchmark: 691 clips, 8 scene clusters, camera IDs, balanced Safe/Unsafe split.
- Publish diagnostic evaluation suite:
  - **Ablation 1:** Detector-rule baseline (geometry only) — measures behavior signal strength.
  - **Ablation 2:** Scene-only (cluster ID as predictor) — upper bound of scene memorization.
  - **Ablation 3:** Per-cluster performance heatmap — which scenes are hardest?
  - **Ablation 4:** Safe-class recall per cluster — diagnosis of class collapse.
  - **Ablation 5:** Model-agnostic explanations (attention maps, feature visualization) — what drives predictions?

**Framing:** *"Unsafe-Net LCO-8: A Camera-Shift Benchmark for Safety Behavior Understanding. We establish a diagnostic evaluation protocol revealing 50%+ performance gains require explicit scene-invariance mechanisms. We benchmark 8 architecture variants and characterize failure modes."*

**Expected Contribution:** Publication venue: arXiv (strong), CVPR/ICCV Workshop (likely), or Conference Proceedings (if results are thorough).

**Implementation Cost:** ~8–12 hours (setup ablations, run experiments, generate heatmaps + summary statistics).

**Recommendation:** HIGH priority (parallel track). Even if model accuracy plateaus, diagnostic protocol is publishable.

### (B) Negative-Result Paper: "Why Small Models Generalize Better Under Camera Shift"

**Precedent:** "Position: Embracing Negative Results in Machine Learning" (NeurIPS 2024 workshop).

**For Unsafe-Net:**
- Title: *"Implicit Bias Beats Capacity: Why a 18k-Parameter DeepSets Outgeneralizes a 299k-Parameter Transformer on Cross-Camera Behavior Understanding"*
- Contribution: Empirical evidence + theoretical analysis (via implicit bias literature) explaining why E3 > E0.
- Analysis:
  - Train variants with explicit capacity controls (18k, 50k, 100k, 299k params).
  - Measure k-fold vs. LCO spread (larger models → larger spread = more memorization).
  - Visualize learned features (PCA, t-SNE per cluster).
  - Hypothesis: implicit regularization in small models suppresses scene features; explicit GRL/adversarial training mimics this in large models.

**Expected Contribution:** JMLR, arXiv, or workshop. Publishable if analysis is rigorous and insights generalize (e.g., "small models for low-data domain shift").

**Implementation Cost:** ~10–15 hours (train 4–5 model variants, feature analysis, write-up).

**Risk:** Negative result only interesting if analysis is novel. Requires careful framing to avoid "we tried big models and they failed" tone.

**Recommendation:** MEDIUM priority (parallel track). Implement alongside Background Mixing; results inform framing.

### (C) Deployment Characterization: "Safety Behavior Detection Under Camera Heterogeneity"

**Precedent:** Anomaly detection benchmarking papers, re-ID papers on camera generalization.

**For Unsafe-Net:**
- Characterize model behavior across 8 camera/scene clusters.
- Deployment readiness assessment: *"Model achieves 69% LCO accuracy. Per-cluster breakdown: [table]. Safe-class recall = 48% (cluster-dependent); Unsafe recall = 89%. Deployment risk: false negatives (missed unsafe) at 51% in 3 clusters."*
- Propose deployment strategy: cluster-specific thresholds, human-in-loop for ambiguous cluster, per-cluster retraining budget.

**Expected Contribution:** Industry/applications venue (Safety-Critical ML, Autonomous Systems). Characterization + deployment recommendations are valued.

**Implementation Cost:** ~5–8 hours (per-cluster breakdowns, threshold tuning, write-up).

**Recommendation:** HIGH priority if time permits. Directly increases practical value of thesis.

---

## Recommendations Ranked by Expected Value per Implementation Hour

**Ranking Strategy:** (Expected LCO Gain in %) / (Implementation Hours) + (Defensibility Score, 0–1).

| **Rank** | **Technique** | **Hours** | **Expected Gain** | **Value** | **Defensibility** | **Start Date** |
|----------|---------------|----------|-------------------|-----------|-------------------|----------------|
| 1 | Coordinate Normalization (BASELINE) | 1 | +0.5% | 0.5 | N/A (required) | Day 1 AM |
| 2 | Background Mixing (CoMix) | 7 | +3% | 0.43 | High (proven method) | Day 1 PM |
| 3 | Motion-Only Diagnostic | 6 | +0% (diagnostic) | 0.17 | High (diagnostic value) | Day 2 AM |
| 4 | LCO-Protocol Benchmark Suite | 10 | +0% (benchmark) | 0.10 | **HIGHEST** (novel protocol) | Day 2 PM |
| 5 | Frozen SigLIP2 + Linear Probe | 6 | +2% | 0.33 | Medium | Day 3 AM |
| 6 | Adversarial Scene Loss (GRL) | 9 | +2% | 0.22 | Medium (architectural) | Day 3 PM |
| 7 | Slot Attention Object-Centric | 11 | +1% | 0.09 | Low (risky on small data) | Day 4 AM |
| 8 | YOLO26 Upgrade | 2 | +0.5% | 0.25 | Low (indirect) | Day 5 AM |
| 9 | ByteTrack Temporal Entity ID | 4 | +1% | 0.25 | Low (risky, requires testing) | Day 5 PM |
| 10 | Negative-Result Capacity Analysis | 12 | +0% (analysis) | 0.08 | Medium (publication, not accuracy) | Parallel |
| 11 | Deployment Characterization | 7 | +0% (characterization) | 0.10 | **HIGH** (practical value) | Parallel |
| 12 | YOLO-Pose + STGCN | 6 | +2% | 0.33 | High (kinematic focus) | Phase 2 |

---

## 1-Week Sprint Plan (Best Effort)

### Day 1 (Sep 19)
- **Morning (4h):** Coordinate Normalization baseline. Measure k-fold vs. LCO impact.
- **Afternoon (4h):** Background Mixing data augmentation pipeline. Start synthetic data generation.

### Day 2 (Sep 20)
- **Morning (4h):** Finish Background Mixing training + eval. Target +2–3% LCO.
- **Afternoon (4h):** Motion-only diagnostic (optical flow extraction + evaluation). Diagnostic report.

### Day 3 (Sep 21)
- **Morning (4h):** SigLIP2 linear-probe baseline. Frozen features + probe training.
- **Afternoon (4h):** LCO-Protocol ablations (detector-rule, scene-only baselines). Start heatmaps.

### Day 4 (Sep 22)
- **Morning (4h):** Adversarial scene loss (GRL) setup. Simple gradient-reversal implementation.
- **Afternoon (4h):** LCO-Protocol diagnostic write-up. Per-cluster performance tables.

### Day 5 (Sep 23)
- **Morning (4h):** YOLO26 upgrade + quick benchmark (skip if no time).
- **Afternoon (4h):** Consolidate results. Write diagnostic benchmark summary. Update thesis chapter.

### Evenings (Async, Parallel)
- Negative-result capacity analysis (train E0/E1/E2/E3 variants, measure k-fold/LCO spread).
- Deployment characterization (per-cluster breakdowns, threshold tuning).

---

## Unresolved Questions & Recommendations for Label/Data Audit

1. **Safe-Class Recall Collapse Cause:** Is the 42–51% Safe-recall due to:
   - (a) True label ambiguity (coders disagree on frame-level safety)?
   - (b) Class imbalance in training (fewer Safe samples)?
   - (c) Model capacity / domain shift (our hypothesis)?
   - **Action:** Audit 20 Safe-class LCO test samples manually. If >30% are mislabeled, relabel before proceeding.

2. **Scene Cluster Semantics:** Are clusters 0–7 semantically distinct (truly different cameras) or do they overlap?
   - **Action:** Visualize cluster membership by location/camera. If clusters are artificial, re-stratify by actual camera ID.

3. **Unsafe-Class Heterogeneity:** Are "Unsafe" labels uniform across 8 classes (0–3) or do some classes (e.g., climbing) require specialized features?
   - **Action:** Per-class LCO breakdown. If Class 0 ≠ Class 1 ≠ Class 2 ≠ Class 3, may need class-specific heads.

4. **Background Mixing Label Assignment:** When blending Safe + Unsafe actors on backgrounds, what is ground truth?
   - **Action:** Define protocol: average labels, majority vote, or drop ambiguous blends. Validate on 50 synthetic samples.

5. **VideoMAE Ceiling:** Why did VideoMAE-base hit 54–56% on 4-class? Is it:
   - (a) Frozen features lack safety semantics?
   - (b) Fine-tuning is necessary (overfits to 566 train)?
   - (c) Architecture mismatch (spatial features ≠ temporal behavior)?
   - **Action:** If pursuing VideoMAE, try linear-probe vs. lightweight finetune (low LR, early stop).

---

## References (By Section)

### 1. Detector Layer
- [YOLO26 NMS-Free End-to-End Detection](https://docs.ultralytics.com/guides/end2end-detection)
- [Ultralytics YOLO26 Architecture](https://arxiv.org/pdf/2606.03748)
- [YOLO26 vs. YOLO11 Benchmarks](https://www.ultralytics.com/blog/why-ultralytics-yolo26-removes-nms-and-how-that-changes-deployment)
- [Ultralytics Object Tracking (ByteTrack & BoT-SORT)](https://academy.ultralytics.com/courses/yolo-in-production/tracking-with-bytetrack-and-botsort)
- [STGCN for Skeleton-Based Action Recognition](https://dl.acm.org/doi/10.5555/3504035.3504947)

### 2. Background & Scene Invariance
- [Contrast and Mix: Temporal Contrastive Video Domain Adaptation with Background Mixing (NeurIPS 2021)](https://arxiv.org/pdf/2110.15128)
- [SOAR: Scene-debiasing Open-set Action Recognition (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhai_SOAR_Scene-debiasing_Open-set_Action_Recognition.pdf)
- [DEVIAS: Learning Disentangled Video Representations (ECCV 2024)](https://arxiv.org/pdf/2312.00826)
- [Group DRO & V-REx Domain Generalization](https://arxiv.org/pdf/2003.00688)
- [Focal Loss for Class Imbalance (2024)](https://arxiv.org/html/2508.02283v1)

### 3. Relational & Temporal Architectures
- [Deep Neural Networks Have an Inbuilt Occam's Razor (Nature Commun. 2024)](https://www.nature.com/articles/s41467-024-54813-x)
- [On the Implicit Bias in Deep-Learning Algorithms](https://arxiv.org/pdf/2208.12591)
- [Dual-State Slot Attention for Video Object-Centric Learning (2025)](https://arxiv.org/pdf/2606.12601)
- [MetaSlot: Adaptive Slot Count (2025)](https://neurips.cc/virtual/2025/poster/119292)
- [Temporally Consistent Object-Centric Learning by Contrasting Slots (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Manasyan_Temporally_Consistent_Object-Centric_Learning_by_Contrasting_Slots_CVPR_2025_paper.pdf)
- [Motion-Only Action Recognition (Optical Flow Survey)](https://arxiv.org/pdf/2011.12619)

### 4. Pretrained Models & VLM
- [VideoMAE: Masked Autoencoders for Video Pre-Training (ICCV 2023)](https://arxiv.org/abs/2203.12602)
- [Learning to Adapt Frozen CLIP for Few-Shot TTA (ICLR 2025)](https://arxiv.org/html/2506.17307v1)
- [SigLIP 2: Multilingual Vision-Language Encoders (Feb 2025)](https://arxiv.org/html/2502.14786v1)
- [Unlocking VLMs for Video Anomaly Detection (Oct 2024)](https://arxiv.org/html/2510.02155v1)
- [VLM-AutoDrive: Safety-Critical Autonomous Driving (2026)](https://arxiv.org/html/2603.18178)
- [Prompt-Induced Score Variance in Binary Safety Classification (Jun 2025)](https://arxiv.org/html/2605.00326)
- [InternVideo-Next: World Understanding Video Models (Dec 2025)](https://arxiv.org/pdf/2512.01342)

### 5. Defensible Contributions
- [Video-Oasis: Rethinking Evaluation of Video Understanding (2024)](https://pith.science/paper/2603.29616)
- [Do Video-LLMs Actually Watch? Diagnosing Character-Tracking Failures (2025)](https://pith.science/paper/2607.11078)
- [EgoSchema: Diagnostic Benchmark for Very Long-Form Video (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/90ce332aff156b910b002ce4e6880dec-Paper-Datasets_and_Benchmarks.pdf)
- [Position: Embracing Negative Results in Machine Learning (NeurIPS 2024 Workshop)](https://arxiv.org/pdf/2406.03980)
- [Anomaly Detection in Video Sequences: Benchmark Review](https://www.sciencedirect.com/science/article/pii/S2667305323000613)
- [Scene-Dependent Video Anomaly Detection Benchmark](https://www.sciencedirect.com/science/article/pii/S1110016825011342)

---

**Report Status:** COMPLETE. Ready for implementation planning & Phase 1 prioritization.
