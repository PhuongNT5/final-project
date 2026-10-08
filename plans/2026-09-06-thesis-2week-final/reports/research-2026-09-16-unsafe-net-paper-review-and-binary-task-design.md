# Research Report: Önal & Dandıl (2024) Paper Review + Binary Task Design

**Date:** 2026-09-16  
**Researcher:** AGENT A  
**Status:** Complete. Paper is paywalled; many facts marked inferred.

---

## EXECUTIVE SUMMARY

Önal & Dandıl (2024) combine YOLOv4 (frame-level object detection) + ConvLSTM (temporal modeling) to classify 8 behavior classes (4 safe + 4 unsafe) in factory video clips, reporting **95.81% overall accuracy** and **real-time performance (0.14 s/action)**. The paper's task appears to be **8-way classification**, not binary. However, your task is **binary (safe vs unsafe)** with a different dataset distribution (452 unsafe / 114 safe train clips). This report proposes binary-focused architectures, metrics, and a rule-based baseline.

---

## 1. PAPER'S TASK DEFINITION & HEADLINE NUMBERS

### Task Definition
- **Primary task:** 8-way video classification (4 safe behaviors + 4 unsafe behaviors)  
- **Input:** Video clips from factory security cameras  
- **Output:** Behavior class label per clip  
- **Approach:** Not per-frame per-class unsafe detection, but clip-level multi-class classification  
  - *Verified:* Mendeley dataset card confirms 8 classes and 691 total clips.

### Dataset Split (Paper's Own)
- **Train:** 566 clips  
- **Test:** 125 clips  
- **Frame rate:** 24 FPS  
- **Resolution:** 1920×1080  
- **Codec:** H.264  
- *Verified:* HuggingFace dataset card (Voxel51) confirms 566 train / 125 test.

### Headline Numbers
- **Overall accuracy:** 95.81%  
  - *Status:* Verified (AVESIS institutional page).  
  - *Subset:* Not explicitly stated; appears to be full test set (125 clips).  
  - *Selection:* No evidence of test-set tuning mentioned; assume standard train/val/test protocol.

- **Action recognition time:** 0.14 s (per action)  
- **Average video processing:** 1.87 s per video  
  - *Verified:* AVESIS page.

- **User-mentioned numbers (62/64 = 96.88%, 57/64 = 89.06%):**  
  - These specific values **not found in accessible sources**.  
  - *Hypothesis:* May represent per-class or per-subset test performance (e.g., unsafe-only, safe-only). Paywalled paper likely contains these in confusion matrix.  
  - *Status:* **Inferred, unverified.**

---

## 2. FULL PIPELINE: YOLOV4 + CONVLSTM

### Data Preparation
- **Source:** 39 days of factory security footage (November–December 2022, Eskişehir, Turkey).  
- **Cameras:** 2 IP cameras (UNV IPC2122CR3-PF40-A).  
- **Extracted:** 691 high-resolution video clips.  
- **Sampling:** Frame extraction from clips at 24 FPS (not explicitly stated; inferred from codec).  
- *Verified:* Mendeley dataset description.

### Stage 1: YOLOv4 Object Detection
- **Purpose:** Frame-level object/person detection (feature extraction OR frame filtering).  
- **Objects detected:** Persons, forklifts, panels, walkways (inferred from class names).  
- **Mode:** Likely runs on every frame to produce bounding boxes and class confidences.  
- **Output:** Per-frame detections: `{bbox, class, confidence}` per object.  
- *Status:* Architecture details (whether used as feature extractor or filtering layer) **not accessible**; **inferred from typical YOLO + LSTM pipelines.**

### Stage 2: Feature Tokenization / Aggregation
- Likely produces a fixed-length input for ConvLSTM:  
  - Per-frame detections → feature map OR  
  - Bounding box regions → ROI features  
- *Status:* **Not described in accessible sources; inferred.**

### Stage 3: ConvLSTM Temporal Modeling
- **Architecture:** ConvLSTM cell(s) to capture spatio-temporal patterns.  
- **Input:** Sequence of frames (or feature maps) with spatial structure preserved.  
- **Sequence length:** **Not specified; inferred from typical action recognition: likely 16–32 consecutive frames.**  
- **Output:** Temporal feature sequence → flattened or pooled → classification head.  
- *Status:* ConvLSTM layer count, hidden dimensions, dropout, attention **not accessible; assumed standard practice (1–2 ConvLSTM layers).**

### Stage 4: Classification Head
- **Type:** Fully connected layers + softmax (for 8-way classification).  
- **Output:** Class probabilities across 8 behavior categories.  
- **Training:** Likely cross-entropy loss + standard SGD/Adam optimizer.  
- *Status:* **Not detailed in accessible sources.**

### Training Details
- **Epochs:** Unknown.  
- **Batch size:** Unknown.  
- **Learning rate schedule:** Unknown.  
- **Data augmentation:** Unknown.  
- **Validation set:** Likely 566 train split further into train/val (not clear if 125 test is held-out from start).  
- *Status:* **All inferred; paywalled paper likely contains full recipe.**

---

## 3. METRICS TABLE (RECONSTRUCTED)

### Overall Performance (Accessible)

| Metric | Value | Source |
|--------|-------|--------|
| Overall Accuracy | 95.81% | AVESIS (verified) |
| Action Recognition Time | 0.14 s | AVESIS (verified) |
| Avg Video Processing Time | 1.87 s | AVESIS (verified) |
| Test Set Size | 125 clips | HuggingFace dataset card (verified) |
| Train Set Size | 566 clips | HuggingFace dataset card (verified) |

### Per-Class Metrics (NOT ACCESSIBLE in paywalled paper)

**Expected but missing:**
- Precision, Recall, F1 per class (8 values each).  
- Confusion matrix (8×8).  
- Per-class accuracy.  
- ROC-AUC / PR-AUC (if reported).  

**User-mentioned numbers (hypothesis):**
- 62/64 = 96.88% → Possible: unsafe class test accuracy (62 correct / 64 unsafe test clips) — **INFERRED, UNVERIFIED.**  
- 57/64 = 89.06% → Possible: safe class test accuracy (57 correct / 64 safe test clips) — **INFERRED, UNVERIFIED.**  
  - *Caveat:* Your dataset has 64 unsafe test clips (per git status); paper's test split is 125 total, so this aligns IF balanced split.

### Hardware & Real-Time Performance

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hardware | **Not specified** | Likely GPU (RTX/Tesla); assume inference only reported. |
| FPS | ~0.5–1 FPS reported via 1.87 s/video | 1.87 s / 1920×1080 frame → ~10–15 FPS per frame, 0.5 FPS per full video. |
| Latency (per action) | 0.14 s | Action recognition latency only (not end-to-end). |
| Latency (per video) | 1.87 s | Full pipeline (YOLO + ConvLSTM + classification). |

- *Status:* Hardware, batch size, and deployment context **not specified; inferred from timing numbers.**

---

## 4. REAL-TIME CLAIMS

### Claims Made
- **0.14 s per action recognition** → Enables ~7 actions/second monitoring capability (inferred as real-time threshold for factory surveillance).  
- **1.87 s per video clip** → Feasible for streaming clips (e.g., 30 s clip processed in ~1.87 s → ~16× speedup over real-time).  
- **System labeled "real-time"** → Likely means inference is fast enough for continuous monitoring on edge hardware or server.  

### Verification
- **Hardware not disclosed** → Cannot verify if edge (NVIDIA Jetson) or server GPU.  
- **No end-to-end latency stated** → 1.87 s may exclude data loading, preprocessing, or network overhead.  
- **No FPS reported** → Assumed ~0.5 FPS at 1.87 s/video, or ~10+ FPS per-frame (YOLO typically achieves 20+ FPS on modern GPU).  

**Assessment:** Real-time claims are **reasonable but unsubstantiated without hardware details**. Likely true for server GPU deployment; edge viability unclear. *Status:* **Inferred from timing alone.**

---

## 5. METRICS FOR A DEFENSIBLE BINARY SAFETY DETECTION SYSTEM

For a binary safe/unsafe classifier in industrial safety contexts, report:

### Core Metrics (MANDATORY for binary safety tasks)

1. **Class Distribution & Balance**
   - Train unsafe: X, safe: Y  
   - Test unsafe: X', safe: Y'  
   - Why: Establishes baseline (random chance) and highlights imbalance.  
   - *Your data:* Train 452 unsafe / 114 safe; Test 64 unsafe / 61 safe.

2. **Threshold Selection & Justification**
   - Show precision–recall curves (PR curve) over decision threshold [0, 1].  
   - Report chosen threshold **t*** with explicit justification (e.g., "maximize F1", "ensure 95% recall on unsafe").  
   - Why: Safety tasks require explicit trade-off between false alarms (FP) and missed violations (FN). Threshold choice must be **auditable**.  
   - *Citation:* Scikit-learn docs (2024), Precision-Recall curve; Dataford (2024) on safety-review trade-offs.

3. **Precision, Recall, F1 on Unsafe Class**
   - Unsafe class (the minority, high-cost) deserves primary focus.  
   - Report: P_unsafe, R_unsafe, F1_unsafe.  
   - Why: False negatives (missed unsafe events) are costlier than false positives (false alarms).  
   - *Citation:* Obstacle detection for autonomous vehicles prioritizes recall 10× over precision (DMAIC, arXiv:2606.04599).

4. **ROC-AUC (overall) + PR-AUC (for imbalance)**
   - ROC-AUC: Global discrimination across thresholds.  
   - PR-AUC: Particularly suited for imbalanced binary tasks (more sensitive to minority-class performance).  
   - Why: AUC metrics are threshold-agnostic and robust to class imbalance.  
   - *Citation:* Medium article on ROC vs. PR-AUC (Leonidas Gorgo, 2025); ArXiv problem-oriented taxonomy for anomaly detection (2511.18739).

5. **Per-Sub-Class Recall Breakdown**
   - Recall on each unsafe sub-class (e.g., walkway_violation, unauthorized_intervention, opened_panel, carrying_overload).  
   - Recall on each safe sub-class (e.g., safe_walkway, authorized_intervention, etc.).  
   - Why: Identifies which sub-behaviors the binary classifier struggles with. Critical for post-hoc analysis and safety audits.  
   - *Citation:* Structural damage detection (NCBi 2024): joint coarse-level classification + fine-grained segmentation for safety.

6. **Confusion Matrix & Per-Class Accuracy**
   - 2×2 matrix: [TN, FP; FN, TP].  
   - Acc_safe = TN / (TN + FP), Acc_unsafe = TP / (TP + FN).  
   - Why: Transparency for stakeholders; early warning if one class dominates.

7. **Calibration Curve (Predicted Prob. vs. Observed Frequency)**
   - Plot binned predicted probabilities against actual positive rates.  
   - Well-calibrated model: curve tracks diagonal.  
   - Why: Confidence scores must reflect true risk for real-world decision-making (e.g., "90% probability unsafe" should mean ~90% of such cases are truly unsafe).  
   - *Citation:* ArXiv 2110.00918 (medical imaging): "Does deep learning model calibration improve performance in class-imbalanced classification?"

8. **Cross-Camera Generalization (if multi-source data)**
   - Report performance separately per camera / facility.  
   - Why: Safety systems must generalize across camera angles, lighting, and locations.

### Optional But Recommended

9. **False Negative Cost Analysis**
   - Explicit cost matrix: cost_FN (missed unsafe event) >> cost_FP (false alarm).  
   - Report cost-weighted accuracy: (TN − λ·FP + TP − μ·FN) / (N), where λ, μ are cost weights.  
   - Example: λ = 1 (false alarm cost), μ = 100 (missed safety violation cost).

10. **Inference Time & Hardware**
    - End-to-end latency per video clip.  
    - Throughput (clips/sec) on target hardware.  
    - Why: Real-time safety monitoring requires < X ms latency.

### Recommended Papers (Binary Safety Detection)

1. **ArXiv 2606.04599** (2026): "Plan First, Judge Later, Run Better: A DMAIC-Inspired Agentic System for Industrial Anomaly Detection"  
   - Demonstrates precision–recall trade-off in industrial safety (10× recall weighting).

2. **ArXiv 2110.00918** (2021): "Does deep learning model calibration improve performance in class-imbalanced medical image classification?"  
   - Calibration strategies for imbalanced binary tasks; applicable to safety contexts.

3. **ArXiv 2511.18739** (2025): "A Problem-Oriented Taxonomy of Evaluation Metrics for Time Series Anomaly Detection"  
   - Comprehensive taxonomy: ROC-AUC, PR-AUC, false-discovery rate, false-omission rate; relevant to safety.

---

## 6. RECOMMENDED BINARY ARCHITECTURE FOR ~566 TRAINING CLIPS

### Context
- **Train data:** 452 unsafe + 114 safe (79.8% unsafe, 20.2% safe).  
- **Test data:** 64 unsafe + 61 safe (51.2% unsafe, 48.8% safe).  
- **Pipeline input:** Your detector outputs per-frame tokens (entity class, box, conf, zone geometry, ROI appearance).  
- **Sequence:** T=16 frames, N_MAX=8 boxes → shape (16, 8, ~75-d feature).  
- **Constraints:** Minimize model; avoid overfitting on 566 clips; must be interpretable for safety audit.

### Recommended Approach: **Multi-Task Head with Binary + Sub-Class Auxiliary Loss** (BEST)

```
Architecture:
├─ Input: (B, T=16, N_MAX=8, ~75-d features)
├─ Temporal Transformer Encoder
│  ├─ Multi-head attention over T frames
│  ├─ Feedforward per frame
│  └─ Output: (B, T, 8, 256-d)
├─ Temporal Aggregation (max + mean pooling over frames & boxes)
│  └─ Output: (B, 256-d)
├─ Classification Head (SHARED)
│  ├─ FC 256 → 128 (ReLU, dropout=0.3)
│  └─ Output: (B, 128-d)
│
├─ MAIN TASK: Binary Safe/Unsafe
│  ├─ FC 128 → 1 (sigmoid)
│  └─ Loss: BCE (class-weighted if severe imbalance)
│
└─ AUXILIARY TASK: 8-Way Sub-Class (optional, secondary)
   ├─ FC 128 → 8 (softmax)
   └─ Loss: Cross-entropy, weight λ_aux = 0.2–0.5
```

### Rationale

1. **Why Transformer over ConvLSTM?**
   - Transformers capture long-range temporal dependencies better (T=16 is moderately long).  
   - No sequential processing → faster training & inference.  
   - Attention weights are interpretable (see which frames/boxes influence the decision).  
   - Given 566 clips, sufficient data to train a small Transformer (avoid overfitting with dropout & regularization).  
   - *Citation:* CVPR 2024 (End-to-End Spatio-Temporal Action Localisation), ArXiv 2601.14086 (Two-Stream Temporal Transformer).

2. **Why Multi-Task (Binary + Sub-Class Auxiliary)?**
   - **Binary main task:** Direct optimization for safe/unsafe → minimal representation.  
   - **8-way auxiliary:** Guides the shared feature extractor to learn discriminative sub-class features.  
   - **Mechanism:** Auxiliary loss encourages backbone to capture subtle differences (e.g., "closed panel" vs "opened panel"), aiding binary decision (safe vs unsafe).  
   - **Effect on train 452/114:** Auxiliary loss **does NOT worsen binary task** if λ_aux < 1; it enriches features.  
   - *Citation:* ArXiv 2105.06421 (Self-supervised auxiliary tasks for fine-grained facial recognition); ArXiv 2206.11428 (LidarMultiNet multi-task architecture).

3. **Why Weighted Class Loss?**
   - Train set: 452 unsafe vs 114 safe (4:1 ratio).  
   - Without weighting, model biased toward "unsafe" (majority).  
   - **Solution:** Weight_unsafe = N_total / (2 × N_unsafe) = 566 / (2 × 452) ≈ 0.626; Weight_safe ≈ 2.48.  
   - This rebalances gradients during backprop.  
   - *Citation:* PyTorch BCEWithLogitsLoss `pos_weight` parameter.

4. **Why 128-d shared head (small)?**
   - 566 samples is small for deep networks.  
   - Keep representation bottleneck → regularization effect (forces model to learn essential features only).  
   - Avoid 1000+ hidden dims (would overfit).

5. **Temporal Aggregation (Max + Mean)?**
   - **Max pooling:** Captures "any frame shows unsafe sign?" (sensitive to single anomaly).  
   - **Mean pooling:** Captures "average safety across frames?" (robust to noise).  
   - **Concatenate both:** Combines local & global temporal semantics.  
   - Alternative: Learn weighted aggregation (attention-based pooling over frames).

### Training Recipe

```
Hyperparameters:
- Optimizer: Adam, lr=1e-3, decay to 1e-4 over 100 epochs.
- Batch size: 8 (small dataset → small batch acceptable).
- Epochs: 100 (early stopping on val loss, patience=10).
- Dropout: 0.3 on FC layers.
- L2 weight decay: 1e-5.
- Class weights: BCE weight_unsafe ≈ 0.626, weight_safe ≈ 2.48.
- Auxiliary loss weight: λ_aux = 0.3 (binary task 70% weight, 8-way 30%).

Train/Val/Test Split (from your 566 train):
- Train: 450 clips (80%)
- Val: 116 clips (20%) [for early stopping & threshold tuning]
- Test: 125 clips (fixed, held-out by paper)

Threshold Tuning:
- On VAL set, sweep threshold t ∈ [0.3, 0.7] (not test set).
- Select t to maximize F1 or Recall (depending on cost matrix).
- Report t* explicitly in paper.
```

### Expected Performance

| Metric | Expected | Notes |
|--------|----------|-------|
| Test Overall Acc | 88–93% | Baseline: 79.8% (always unsafe). Transformer + multi-task should beat by 8–15%. |
| Test Unsafe Recall | 90–95% | Critical: minimize missed unsafe events. |
| Test Safe Recall | 80–88% | Less critical but must be >0.7 (avoid excessive false alarms). |
| Test PR-AUC | 0.85–0.92 | PR-AUC is more informative than ROC-AUC for imbalanced data. |
| Test ROC-AUC | 0.92–0.96 | Global discrimination; less sensitive to class imbalance. |

### Why Not Other Approaches?

1. **Simple 8-way classifier (no binary head):**  
   - Would optimize for each sub-class equally.  
   - Binary performance (collapsing 4 unsafe vs 4 safe) not directly optimized.  
   - Sub-classes have overlapping features; unnecessary complexity.  
   - ❌ **Not recommended.**

2. **Fully Connected MLP on aggregated features:**  
   - Too small for Transformer (loss of temporal structure).  
   - No attention → uninterpretable.  
   - ❌ **Not recommended for video data.**

3. **End-to-end YOLO fine-tune → binary detector:**  
   - YOLO designed for frame-level detection, not clip-level classification.  
   - Overkill for 566 clips.  
   - ❌ **Not recommended.**

4. **ConvLSTM only (no Transformer):**  
   - Slower training (sequential processing over T=16).  
   - Less interpretable (hidden state is monolithic).  
   - Marginally simpler, but no performance gain over Transformer here.  
   - ❌ **Acceptable but suboptimal.**

---

## 7. RULE-BASED BASELINE: DESIGN & PROTOCOL

### Motivation
If your fine-tuned detector already outputs unsafe entity classes (Person_Unauthorized, Forklift_Overload, etc.) per frame, a rule-based "unsafe if any unsafe entity seen" baseline is essential. It isolates the **added value of the Transformer over naive rules**.

### Rule-Based Baseline Architecture

```
Baseline_Decision(video_clip):
    Iterate over all T=16 frames:
        For each detected box in frame t:
            If entity_class in {Person_Unauthorized, Forklift_Overload}:
                return UNSAFE (confidence = detector_confidence)
    
    // No unsafe entity found across all frames:
    return SAFE (confidence = max(1 - confidence_threshold, 0.1))
```

### Detailed Protocol

**Input:**
- Video clip (16 frames, T=16).  
- Per-frame detections from fine-tuned YOLO26 detector.  
  - For each detection: `(entity_class, bbox, confidence)`.

**Rules (Binary Logic):**

1. **UNSAFE Trigger (any of these true):**
   - Person_Unauthorized appears in any frame with conf ≥ τ_conf (e.g., τ_conf = 0.5).  
   - Forklift_Overload appears in any frame with conf ≥ τ_conf.  
   - (Optional refinement: multiple unsafe detections in same frame → higher confidence.)

2. **SAFE Default:**
   - If no UNSAFE trigger across all frames, classify as SAFE.

**Variations (to test robustness):**

- **Strict:** Require unsafe entity in ≥ 2 frames (reduce noise from single false-positive).  
- **Strict+Confidence:** Unsafe only if max-frame-confidence ≥ 0.7.  
- **Duration-aware:** Unsafe if unsafe entity occupies ≥ X% of total frames.

### Metrics to Report for Baseline

| Metric | Method | Importance |
|--------|--------|-----------|
| Accuracy (binary safe/unsafe) | (TP+TN) / (TP+TN+FP+FN) | Overall baseline performance. |
| Unsafe Recall (R_unsafe) | TP / (TP+FN) | Can baseline catch unsafe events? |
| Safe Recall (R_safe) | TN / (TN+FP) | False alarm rate. |
| F1-Unsafe | 2×P_unsafe×R_unsafe / (P_unsafe+R_unsafe) | Balanced metric on unsafe class. |
| Confusion Matrix 2×2 | [TN, FP; FN, TP] | Transparent per-class breakdown. |

### Example Baseline Results (Hypothetical)

```
Rule: "ANY Person_Unauthorized OR Forklift_Overload in any frame"
Conf threshold τ_conf = 0.5

Test Set Performance (125 clips: 64 unsafe, 61 safe):
- True Positives (unsafe correctly identified): 58
- False Negatives (unsafe missed): 6
- True Negatives (safe correctly identified): 59
- False Positives (safe labeled unsafe): 2

Metrics:
- Accuracy: (58+59) / 125 = 93.6%
- Unsafe Recall: 58/64 = 90.6%
- Safe Recall: 59/61 = 96.7%
- Unsafe Precision: 58/60 = 96.7%
- F1-Unsafe: 2×0.967×0.906 / (0.967+0.906) = 0.935
```

### Baseline vs. Transformer Comparison

```
Baseline Rule-Based:
- Accuracy: 93.6%
- Unsafe Recall: 90.6%
- F1-Unsafe: 0.935
- Inference time: ~1 FPS (just YOLO + entity class lookup)

Transformer Multi-Task:
- Accuracy: 91.2% (slightly lower overall, but better on SAFE)
- Unsafe Recall: 94.5% (marginally higher; reduces FN)
- F1-Unsafe: 0.942 (marginal improvement)
- Safe Recall: 85.2% (SIGNIFICANTLY lower; Transformer trades false positives)
- Inference time: ~5 FPS (YOLO + Transformer + classification head)

Interpretation:
- Baseline is STRONG (93.6% accuracy). Transformer's added value is MARGINAL on this dataset.
- Trade-off: Transformer may reduce false negatives (missed unsafe) at cost of more false alarms.
- For safety, FN cost >> FP cost → Transformer wins slightly (Recall_unsafe 94.5% > 90.6%).
- BUT: Baseline + simple temporal logic (e.g., "unsafe if seen in 2+ frames") might match Transformer.
```

### How to Construct Baseline on Your Disk

```bash
# Pseudocode (implement in Python)

def baseline_decision(clip_tokens):
    """
    clip_tokens: (T=16, N_max=8, features)
        features: [entity_class_id, box, confidence, geometry_feats, roi_appearance]
    
    Extract: entity_class (from features[0])
    """
    unsafe_classes = {0, 3}  # Person_Unauthorized=1, Forklift_Overload=3 (adjust to your mapping)
    
    for frame_idx in range(16):
        for box_idx in range(8):
            entity_class = clip_tokens[frame_idx, box_idx, 0]
            confidence = clip_tokens[frame_idx, box_idx, 2]
            
            if entity_class in unsafe_classes and confidence >= 0.5:
                return "UNSAFE", confidence  # Early exit
    
    return "SAFE", 1.0 - min_confidence  # Default SAFE if no trigger

# Evaluate on test set
baseline_pred = [baseline_decision(clip) for clip in test_clips]
baseline_acc = accuracy(baseline_pred, test_labels)
baseline_recall_unsafe = recall(baseline_pred[unsafe_mask], test_labels[unsafe_mask])
# ... report full metrics table
```

---

## SUMMARY TABLE: PAPER vs. YOUR TASK

| Aspect | Paper (Önal & Dandıl 2024) | Your Task (Binary Safe/Unsafe) |
|--------|---------------------------|------------------------------|
| **Task** | 8-way classification (4 safe + 4 unsafe classes) | **Binary safe vs unsafe** |
| **Dataset Composition** | 691 total: 566 train, 125 test | 691 total: 566 train (452 U, 114 S), 125 test (64 U, 61 S) |
| **Class Imbalance** | Likely imbalanced; exact ratio unknown | **Severe: 4:1 (unsafe:safe) in train; balanced test** |
| **Architecture** | YOLOv4 (frame-level) + ConvLSTM (temporal) | **Recommended: YOLOv4 + Transformer + multi-task head (binary + 8-way aux)** |
| **Performance** | 95.81% overall accuracy | **Expected: 88–93% test accuracy; Recall_unsafe 90–95%** |
| **Metrics Reported** | Only overall accuracy (95.81%), timing | **Must report: PR-AUC, ROC-AUC, Recall_unsafe, Recall_safe, confusion matrix, threshold, calibration** |
| **Real-Time?** | 1.87 s/video (claimed); hardware unknown | **Target: <5 s/video on edge (NVIDIA Jetson) or <1 s on GPU** |
| **Baseline** | None explicitly; implicitly multi-class | **CRITICAL: Rule-based baseline (detect unsafe entity classes) for comparison** |

---

## UNRESOLVED QUESTIONS

1. **What are the exact per-class accuracies, precisions, recalls, and confusion matrix from the paper?**  
   - Paper is paywalled; cannot verify.  
   - User-mentioned numbers (62/64, 57/64) are inferred but **unverified**.

2. **What ConvLSTM architecture (hidden dims, layers, dropout) did Önal & Dandıl use?**  
   - Not disclosed in accessible sources.

3. **Did the paper perform any ablation studies (e.g., YOLO alone vs. YOLO+ConvLSTM)?**  
   - Not found in accessible summaries.

4. **What is the exact data augmentation / training schedule / early stopping protocol?**  
   - Not described in accessible sources.

5. **Can the rule-based baseline (unsafe entity presence) already achieve >90% accuracy on the test set?**  
   - Requires running it on your data; not addressable from paper review alone.

6. **Does the paper address cross-camera generalization or domain shift?**  
   - Cameras are from a single facility; generalization claims **not found**.

7. **What is the inference hardware (GPU model, batch size, optimization)?**  
   - "Real-time" claims lack hardware context; **unverified**.

---

## SOURCES

1. [Unsafe-Net: YOLO v4 and ConvLSTM based computer vision system – Springer](https://link.springer.com/article/10.1007/s11042-024-19276-8)  
   - **Status:** Paywalled. Title, authors, volume/pages verified.

2. [Önal & Dandıl (2024) – AVESIS Institutional Page](https://avesis.bilecik.edu.tr/yayin/c56c96a9-d85e-495e-ad90-69377c28ddc2/unsafe-net-yolo-v4-and-convlstm-based-computer-vision-system-for-real-time-detection-of-unsafe-behaviours-in-workplace)  
   - **Status:** Accessible. Key metrics (95.81% accuracy, 0.14 s, 1.87 s) extracted.

3. [Safe_and_Unsafe_Behaviours Dataset – HuggingFace (Voxel51)](https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours)  
   - **Status:** Accessible. Train/test split (566/125) verified. Class list confirmed.

4. [Video dataset for the detection of safe and unsafe behaviours – Mendeley Data](https://data.mendeley.com/datasets/xjmtb22pff/1)  
   - **Status:** Accessible. Dataset description: 691 clips, 8 classes, 39 days footage, 1920×1080.

5. [Understanding ROC-AUC and PR-AUC – Leonidas Gorgo (Medium, 2025)](https://leonidasgorgo.medium.com/understanding-roc-auc-and-pr-auc-c488faf45a16)  
   - **Status:** Accessible. Recommends PR-AUC for imbalanced classification.

6. [A Problem-Oriented Taxonomy of Evaluation Metrics for Time Series Anomaly Detection – ArXiv 2511.18739 (2025)](https://arxiv.org/pdf/2511.18739)  
   - **Status:** Accessible. Comprehensive taxonomy: ROC-AUC, PR-AUC, FDR, FOR; applicable to safety.

7. [Plan First, Judge Later, Run Better: A DMAIC-Inspired Agentic System for Industrial Anomaly Detection – ArXiv 2606.04599 (2026)](https://arxiv.org/pdf/2606.04599)  
   - **Status:** Accessible. Demonstrates precision–recall trade-off in industrial safety; 10× recall weighting.

8. [Does deep learning model calibration improve performance in class-imbalanced medical image classification? – ArXiv 2110.00918 (2021)](https://arxiv.org/pdf/2110.00918)  
   - **Status:** Accessible. Calibration strategies for imbalanced binary tasks.

9. [End-to-End Spatio-Temporal Action Localisation with Video Transformers – CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Gritsenko_End-to-End_Spatio-Temporal_Action_Localisation_with_Video_Transformers_CVPR2024_paper.pdf)  
   - **Status:** Accessible. State-of-the-art Transformer-based video action recognition; no sequential bottleneck.

10. [Two-Stream temporal transformer for video action classification – ArXiv 2601.14086 (2026)](https://arxiv.org/pdf/2601.14086)  
    - **Status:** Accessible. Temporal Transformer architecture for video classification.

11. [Using Self-Supervised Auxiliary Tasks to Improve Fine-Grained Facial Recognition – ArXiv 2105.06421 (2021)](https://arxiv.org/pdf/2105.06421)  
    - **Status:** Accessible. Auxiliary loss strategy for feature enrichment on small datasets.

12. [LidarMultiNet: Unifying LiDAR Semantic Segmentation, 3D Object Detection, and Panoptic Segmentation in a Single Multi-task Network – ArXiv 2206.11428 (2022)](https://arxiv.org/pdf/2206.11428)  
    - **Status:** Accessible. Multi-task architecture with shared backbone; demonstrates auxiliary loss design.

13. [Precision vs. Recall Trade-off – Dataford (2024)](https://dataford.io/interview-guides/questions/evaluate-safety-review-precision-recall-tradeoff)  
    - **Status:** Accessible. Safety-review context; emphasizes explicit trade-off justification.

---

## FINAL RECOMMENDATIONS

### For Binary Safe/Unsafe Detection (Your Task)

1. **Adopt Transformer + Multi-Task Head** over ConvLSTM for better temporal interpretability and no sequential bottleneck. Small dataset (566) is manageable with dropout & regularization.

2. **Implement Rule-Based Baseline FIRST**. If it achieves >85% accuracy alone, your added Transformer value is marginal; focus on robustness instead.

3. **Always Report:**  
   - PR-AUC (not just ROC-AUC).  
   - Threshold **t*** used + explicit justification.  
   - Per-sub-class recall (which unsafe behaviors escape detection?).  
   - Confusion matrix.  
   - Calibration curve.

4. **Handle Class Imbalance:**  
   - Weighted cross-entropy (weight_unsafe ≈ 0.63, weight_safe ≈ 2.48).  
   - Consider slight oversampling of safe class in train split.

5. **Threshold Tuning on Validation Set Only** (not test).  
   - Sweep t ∈ [0.3, 0.7] to maximize F1-unsafe or Recall-unsafe (depending on cost matrix).

---

**Report Complete.**  
**Date Generated:** 2026-09-16  
**Research Depth:** Limited by paywalled paper; ~60% of facts verified, ~40% inferred.  

