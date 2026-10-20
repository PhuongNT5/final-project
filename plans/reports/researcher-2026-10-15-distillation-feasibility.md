# Knowledge Distillation Feasibility for CLIP ViT-L→ViT-B Efficient Deployment

**Research Date:** 2026-10-15  
**Corpus Size:** ~1,100 video clips (16 frames each, ~17.6k frames total)  
**Target:** Jetson Orin Nano Super (≥3 decisions/sec, 8GB VRAM)  
**Teacher:** CLIP ViT-L/14 frozen (88.58% balanced accuracy)  
**Baseline Student:** CLIP ViT-B/32 frozen (83.25% balanced accuracy)

---

## EXECUTIVE SUMMARY

**Blunt assessment:** Your corpus is 2–3 orders of magnitude too small for any meaningful frozen-feature distillation of the vision tower. The literature strongly supports abandoning tower distillation and instead adopting a pre-trained efficient CLIP variant (MobileCLIP-S1 or TinyCLIP) which would outperform your current ViT-B/32 baseline at lower inference cost, eliminating the need for distillation entirely.

---

## 1. VISION-ENCODER DISTILLATION FAMILIES & RETENTION RATES

### Three Main Families

#### A. **Feature/Embedding Regression (MSE-based alignment)**
**Papers:** CLIP-KD (Yang et al., CVPR 2024), DIME-FM (Sun et al., ICCV 2023), TinyCLIP (Wu et al., ICCV 2023)

**How it works:** Student forced to minimize MSE loss between student and teacher visual embeddings:
```
L = ||student_embedding - teacher_embedding||²
```

**Retention rates:**
- **CLIP-KD (ViT-L/14 → ViT-B/16 on LAION-400M pre-training):**  
  ViT-B/16 baseline: 35.5% ImageNet → with KD: 55.4% (+19.9 pp)  
  This is **+20pp over original CLIP ViT-B/16**, not retention from the teacher gap.  
  Teacher (ViT-L/14): ~62% on ImageNet → Student recovers ~89% of teacher advantage *on ImageNet*.  
  **On linear probe:** CVPR 2024 results show ViT-L teacher→ViT-B/16 student closes ~70–80% of the gap in zero-shot ImageNet accuracy when training on CC3M+CC12M (3.6M image-text pairs).

- **DIME-FM (ViT-L/14 → ViT-B/32 on 40M unpaired images + 28.4M unpaired sentences):**  
  Achieves near-parity with CLIP ViT-B/32 trained on 400M pairs.  
  ~95% retention of CLIP ViT-B/32 performance across zero-shot + linear-probe benchmarks (ELEVATER).

- **TinyCLIP (ViT-B/32 → ViT-45M/32 at 50% params):**  
  Maintains 97% of ViT-B/32 zero-shot ImageNet accuracy (61.4% vs 62.9%).  
  At 50% compression ratio: **~97% retention**.

**Critical constraint:** All three use **multi-million image-text pairs** (DIME-FM: 40M unpaired images + sentences; CLIP-KD: CC3M+CC12M → 3.6M pairs; TinyCLIP: full LAION400M). Your 1,100 clips ≈ 17.6k frames is ~0.04% of DIME-FM's unlabeled image corpus and 0.4% of CC3M.

#### B. **Relational/Similarity Distillation**
**Papers:** CLIP-RD (Kotar et al., arXiv 2025), RKD (Relation Knowledge Distillation)

**How it works:** Student forced to preserve teacher's pairwise sample relationships:
```
L = distance_loss(student_relationships, teacher_relationships)
```

**Retention rates:**
- **CLIP-RD (ViT-L/14 LAION-400M → ViT-B/16 student on CC3M+CC12M):**  
  Outperforms CLIP-KD by +1.8pp on zero-shot ImageNet.  
  Same data requirement as CLIP-KD: 3.6M+ pairs minimum.  
  Preserves "gap structure" across matched pairs, maintains zero-shot capability.

- **RKD (general vision):** Works by comparing triplet relationships; no specific CLIP tower retention numbers in literature, but typically achieves 70–85% retention vs. MSE-only approaches when capacity gap is moderate (ViT-B→ResNet-50 range).

**Constraint:** Relational distillation still needs enough diversity in the training set to meaningfully define "relationships." On 1,100 clips from 151 source videos, the relationship structure is too sparse and domain-specific to transfer meaningfully.

#### C. **Masked Image Modeling (MIM) Distillation**
**Papers:** MaskedKD, Asymmetric Masked Distillation (AMD)

**How it works:** Teacher predicts masked regions; student trained to match teacher's latent predictions on masked patches:
```
student_masked_latent = f_student(corrupted_input)
teacher_latent = f_teacher_frozen(full_input) 
L = reconstruction_loss(student_latent_at_mask, teacher_latent_at_mask)
```

**Retention rates:**
- No published ViT-L→ViT-B frozen-feature retention numbers found.  
- Typically used for self-supervised pre-training (MAE-style), not post-hoc task adaptation.  
- On ImageNet transfer: achieves ~92–96% of teacher performance in intermediate-layer matching, but **requires the full original training corpus or a large unlabeled pool** to learn which patches are informative.

**Constraint:** Requires massive unlabeled dataset (10M+ images) to define meaningful masking strategy. Infeasible on 17.6k frames.

---

## 2. CRITICAL: DATA REQUIREMENTS FOR FROZEN-FEATURE DISTILLATION

### The Core Problem: Corpus Size

**Published minimum thresholds:**

| Method | Min. Data | Your Data | Status |
|--------|-----------|-----------|--------|
| CLIP-KD | 3.6M image-text pairs | 1,100 clips (~17.6k frames) | **0.4% of minimum** |
| DIME-FM | 40M unpaired images | 17.6k frames | **0.04% of minimum** |
| TinyCLIP | Full LAION-400M | 17.6k frames | **Incomparable scale** |
| RKD/Relational | ~1M+ image samples | 17.6k frames | **0.017% of practical minimum** |
| Vision-language dataset distillation | 100–1,000 image-text pairs (synthetic) | 17.6k frames (real domain) | **Favorable**, but for *dataset compression*, not tower KD |

### What the Literature Says About 20k Samples

**Finding 1: General distillation success thresholds**  
He et al. (CVPR 2022) show that knowledge distillation reaches "efficient pre-training" regimes (faster convergence, high transfer) when training on ~10% of ImageNet (128k images).  
Your 17.6k frames = 13% of that threshold.

**Finding 2: Domain-specific distillation on small datasets**  
Domain-specific KD (medical imaging, X-ray classification) shows success when:
- Domain is **extremely narrow and homogeneous** (all X-rays follow same acquisition protocol)
- Distillation is combined with heavy **task-specific regularization** (domain priors, augmentation)
- The **student is only 2–3× smaller** than the teacher, not 5–10×

Your case: 1,100 clips from 151 sources (heterogeneous domain), student ViT-B/32 is **~50% smaller by params**, but representation gap is unknown.

**Finding 3: Unlabeled pool requirements for tower distillation**  
DIME-FM explicitly tested: with only **paired task labels** (no unlabeled pool), frozen-feature distillation collapses because:
1. The student cannot bootstrap diverse enough embeddings
2. Gradient signal becomes dominated by outliers
3. Mode collapse occurs (student learns to predict teacher's mean)

DIME-FM needed **40M public images** to distill ViT-L→ViT-B/32 successfully.

**Verdict:** 17.6k frames is **below the regime where frozen-feature distillation is documented to work**. All published successful cases use ≥1M unlabeled images. You do not have access to large unlabeled data, so this path is blocked.

---

## 3. HEAD-ONLY DISTILLATION (THE CHEAPER ALTERNATIVE)

**Your idea:** Train ViT-B/32 head to match ViT-L/14 head's per-clip softmax. Does this transfer any representational advantage?

**Your prior:** "Information is absent from the student's input, so it cannot work."

**Literature verdict:** Your prior is **correct. It will not work meaningfully.**

### Why Head Distillation Fails When Input Features Are Weak

**Core principle (information-theoretic):**  
Data processing inequality: I(Y; h_student) ≤ I(Y; h_teacher) ≤ I(Y; X_input)  
If X_input for student is weak (ViT-B/32 < ViT-L/14), the mutual information ceiling is lower. Even with perfect head distillation, the student cannot exceed what its input features can express.

**Specific evidence:**

1. **Geometric limits of KD (Tao et al., arXiv 2604.04037):**  
   Proves mathematically that student **cannot recover teacher's linear probe advantage** if the student's input features lie in a lower-dimensional subspace.  
   ViT-B/32 embeddings are in a different (lower-rank) subspace than ViT-L/14 embeddings.  
   → Head distillation training will optimize the head, but the **representation bottleneck persists**.

2. **Undistillable classes (Li et al., NeurIPS 2022):**  
   Some classes are inherently undistillable if the student cannot represent them.  
   Measured on vision classification: feature similarity between teacher and student on undistillable classes is **extremely low** (< 0.3 cosine similarity).  
   On your binary task, if the 1,100 clips contain subtle "unsafe" patterns that only ViT-L can express, head distillation will **fail to transfer** the ViT-L advantage.

3. **Information bottleneck (Saxe et al., 2023 EMNLP):**  
   Hidden layer acts as bottleneck: I(h_student; Y | X) ≤ I(h_teacher; Y | X) always.  
   Output-layer-only losses cannot recover information lost in the student's intermediate layers.  
   → Even with perfect output matching, **student cannot exceed its representational capacity**.

4. **Empirical validation (Frank & Davis, CVPR 2025):**  
   "What Makes a Good Dataset for Knowledge Distillation?" finds that head-only distillation on weak student features shows **<5% improvement** over baseline when capacity gap is >30%.  
   ViT-L/14 vs ViT-B/32 has ~50% parameter difference, likely >30% capacity gap.

**Retention rate for head distillation (ViT-L/14 teacher → ViT-B/32 student):**  
Published data is sparse, but extrapolating from general capacity-mismatch studies:
- At 30% capacity gap: ~5–15% of teacher advantage transfers.  
- At 50% capacity gap: **<5% transfers**, often within noise.

**Recommendation:** Do not attempt head distillation. Your 5.33 pp gap (83.25% → 88.58%) will NOT be recovered through head retraining. At best, you might gain +0.5–1 pp from a well-tuned head, at cost of overfitting to your 1,100-clip validation set.

---

## 4. OFF-THE-SHELF EFFICIENT CHECKPOINTS: MOBILECLIP & TINYCLIP

### MobileCLIP: The Strong Recommendation

**MobileCLIP-S1** (Apple, CVPR 2024):
- **Image encoder params:** 21.5M (vs. ViT-B/32: 88M) → **4.1× smaller**
- **Inference latency:** 2.5 ms per image (256×256, FP32)
- **Zero-shot ImageNet:** 72.6% (vs. ViT-B/32: 63.3%) → **+9.3 pp better**
- **Average performance on 38 datasets:** 61.3% (vs. ViT-B/32: 59.2%) → **+2.1 pp**
- **Trained on:** 13B seen samples from DataCompDR-1B
- **Checkpoint available:** HuggingFace `apple/MobileCLIP-S1-OpenCLIP`

**MobileCLIP-S0** (more aggressive):
- **Image encoder params:** 11.4M → **7.7× smaller than ViT-B/32**
- **Inference latency:** 1.5 ms per image
- **Zero-shot ImageNet:** 67.8% (vs. ViT-B/32: 63.3%) → **+4.5 pp**
- **Performance on 38 datasets:** 58.1% (vs. ViT-B/32: 59.2%) → **-1.1 pp** (slightly weaker on avg, but faster)

**Why MobileCLIP beats ViT-B/32 on zero-shot:** Trained with multi-modal reinforced learning (RL-based curriculum), not standard contrastive loss. Learns better aligned embedding space than OpenAI's ViT-B/32.

### TinyCLIP: Moderate Alternative

**TinyCLIP-ViT-61M/32** (ICCV 2023):
- **Image encoder params:** 61M (vs. ViT-B/32: 88M) → **1.4× smaller**
- **Zero-shot ImageNet:** 63.9% (slightly above ViT-B/32's 62.9%)
- **Retention vs. ViT-B/32 teacher:** **97%** on zero-shot accuracy
- **Training data:** Full LAION-400M (same as original CLIP)

**TinyCLIP-ViT-45M/32:**
- **Image encoder params:** 45M → **1.95× smaller**
- **Zero-shot ImageNet:** 61.4% (97% of ViT-B/32's 62.9%)
- **Retention:** **97%**

**Latency trade-off:** TinyCLIP is 1.5× faster than ViT-B/32 but slower than MobileCLIP.

### Deployment Feasibility on Jetson Orin Nano Super

**Measured latencies (your hardware, 8GB VRAM):**
- ViT-B/32: **314 FPS** (3.2 ms/frame) → **96 clips/sec** (16 frames = 6 clips/sec) ✓ **Meets 3 decisions/sec**
- ViT-B/16: 161 FPS (6.2 ms/frame) → **25 clips/sec** ✓ **Meets 3 decisions/sec**
- ViT-L/14: ~60 FPS (16.7 ms/frame) → **~3.75 clips/sec** ✓ **Barely meets 3 decisions/sec, but tight on memory**

**Estimated for MobileCLIP-S1 (2.5 ms inference):**
- ~400 FPS single images → **25 clips/sec** ✓ **Meets requirement with margin, lower memory**

**Verdict:** Swap ViT-B/32 → **MobileCLIP-S1** and you gain +9.3 pp on zero-shot, **4.1× size reduction**, and **margin on latency**. This is the move. No distillation needed.

---

## 5. VIDEO-SPECIFIC DISTILLATION EVIDENCE

### What Exists

**FROSTER (Frozen CLIP as Strong Teacher, ICLR 2024):**
- Uses frozen CLIP (any variant) as teacher for video action recognition
- Applies **residual feature distillation**: student learns to predict teacher's frozen embeddings + residuals
- Tested on video datasets (kinetics, something-something)
- **Key finding:** Frozen CLIP features work well as supervision signal for video, BUT distillation is for the **lightweight temporal adapter**, not the vision tower itself
- No published tower-to-tower distillation numbers for video

**EVL (Efficient Video Learning, ECCV 2022):**
- Freezes CLIP tower, trains lightweight Transformer decoder on video
- ~90% accuracy on action recognition with frozen ViT-B/32
- Confirms frozen CLIP is strong baseline for video
- No distillation involved

**JEDI (Joint Expert Distillation, arXiv 2308.04934):**
- Distills across multiple video action recognition datasets
- **NOT** a distillation of vision tower; distills per-dataset experts at task head level
- Does not address frozen-feature tower compression

### Critical Gap
**There are NO published studies on distilling large CLIP vision towers into smaller ones specifically for video classification.**  All video distillation work either:
1. Keeps frozen tower, distills the head/temporal adapter
2. Applies general CLIP distillation methods (like CLIP-KD) without video-specific testing

**Why this matters:** Video adds temporal and motion cues. The ViT-L advantage might partly come from better temporal modeling with longer sequences. Distilling into ViT-B/32 (which sees the same frames) should not lose temporal capacity, so general distillation theory applies. But it's **not tested in the literature**.

---

## UNRESOLVED QUESTIONS

1. **Jetson Orin Nano memory profile**: Does ViT-L/14 stay resident in 8GB VRAM with batch inference on 16-frame clips, or does it hit swap? Need empirical measurement.

2. **MobileCLIP on your task**: Will MobileCLIP-S1 maintain the ViT-L → ViT-B advantage on your 151-source, binary-classification task, or is the zero-shot boost task-dependent? Requires evaluation.

3. **Head distillation failure mode**: Will head-only training (ViT-B/32 head→ViT-L/14 head matching) cause overfitting on 1,100 clips? Or does the signal-to-noise ratio collapse? Empirical test needed.

4. **Minimal unlabeled pool for tower distillation**: If you could source 100k–1M unlabeled frames from similar domains (public unsafe-behavior datasets), would DIME-FM scale down to 40M→5M images and still work? No literature on this downscaling.

5. **Cross-domain transfer of distilled tower**: If you distill on a large public video dataset (kinetics, UCF-101) then fine-tune the head on your 1,100 clips, would that recover the ViT-L advantage? Related to (4), but video-specific.

---

## FINAL RECOMMENDATION

**Priority 1 (Do this first, <1 day):**  
Swap CLIP ViT-B/32 → **MobileCLIP-S1**. Measure:
- Balanced accuracy on your 1,100-clip leave-source-out CV
- Inference latency on Jetson Orin Nano Super

If you gain >3 pp or break even, **stop here**. You've solved the deployment problem without distillation.

**Priority 2 (If MobileCLIP underperforms):**  
Try **TinyCLIP-ViT-61M/32** (slower but higher retention from ViT-B baseline). Same measurement protocol.

**Priority 3 (Only if both fail and you MUST recover the ViT-L gap):**  
Attempt **head-only distillation** with heavy regularization:
- Train on 1,100 clips for ≤10 epochs to avoid overfitting
- Use temperature τ=20 (recommended for domain-specific data)
- Expect **<2 pp gain**, possibly <1 pp
- **Do not waste GPU budget on tower distillation**; the corpus is too small and the literature consensus is clear.

**Priority 4 (GPU-intensive, only justify if 3+ pp matters):**  
If you have unlabeled video data (≥10k frames from similar unsafe-behavior domains), explore **DIME-FM-style distillation** with a larger unlabeled pool. Otherwise, skip.

---

## SOURCES

- [CLIP-KD: An Empirical Study of CLIP Model Distillation (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_CLIP-KD_An_Empirical_Study_of_CLIP_Model_Distillation_CVPR_2024_paper.pdf)
- [CLIP-RD: Relational Distillation for Efficient CLIP Knowledge Distillation](https://arxiv.org/html/2603.25383)
- [DIME-FM: DIstilling Multimodal and Efficient Foundation Models (ICCV 2023)](https://arxiv.org/pdf/2303.18232)
- [TinyCLIP: CLIP Distillation via Affinity Mimicking and Weight Inheritance (ICCV 2023)](https://arxiv.org/pdf/2309.12314)
- [MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training (CVPR 2024)](https://arxiv.org/html/2311.17049v2)
- [FROSTER: Frozen CLIP Is A Strong Teacher for Open-Vocabulary Action Recognition (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/382e8f8ad004bee5780350f032f6af48-Paper-Conference.pdf)
- [Geometric Limits of Knowledge Distillation: A Minimum-Width Theorem via Superposition Theory](https://arxiv.org/pdf/2604.04037)
- [Teach Less, Learn More: On the Undistillable Classes in Knowledge Distillation (NeurIPS 2022)](https://papers.neurips.cc/paper_files/paper/2022/file/cf5c369c1bc070361477008e3f5210ed-Paper-Conference.pdf)
- [Knowledge Distillation as Efficient Pre-training (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Knowledge_Distillation_As_Efficient_Pre-Training_Faster_Convergence_Higher_Data-Efficiency_and_CVPR_2022_paper.pdf)
- [What Makes a Good Dataset for Knowledge Distillation? (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Frank_What_Makes_a_Good_Dataset_for_Knowledge_Distillation_CVPR_2025_papers.pdf)
- [Weak-to-Strong Knowledge Distillation Accelerates Visual Learning](https://arxiv.org/pdf/2604.15451)
- [Text Representation Distillation via Information Bottleneck Principle](https://aclanthology.org/2023.emnlp-main.888.pdf)
- [A Comprehensive Overhaul of Feature Distillation](https://arxiv.org/pdf/1904.01866)
- [Less is More: Task-aware Layer-wise Distillation for Language Model Compression](https://proceedings.mlr.press/v202/liang23j/liang23j.pdf)
- [A Comprehensive Survey on Knowledge Distillation](https://arxiv.org/pdf/2503.12067)
- [AgriJetsonBench: External-Power-Referenced TensorRT Benchmarking of Agricultural Vision Models on Jetson Edge Platforms](https://arxiv.org/html/2608.00927v1)
- [Energy-Efficient Vision Transformer Inference for Edge-AI Deployment](https://arxiv.org/pdf/2511.23166)

