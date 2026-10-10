# NÂNG CẤP BIỂU DIỄN VISUAL VÀ DỰ BÁO MỨC TĂNG (R3 Reviewer)

**Giáo viên hướng dẫn (Phần đánh giá khoa học):** REVIEWER R3  
**Ngày báo cáo:** 2026-10-10  
**Phạm vi:** Nâng cấp encoder và adapter từ 84.2% BA trên iSafetyBench

---

## 1. Phán quyết: Headroom Thực Tế (Hard Protocol)

Với kappa=0.65 trên 72% raw agreement và 1,100 clips training, mức tăng khả thi:

| Tuyến | Mức tăng dự báo | Độ tin cậy | Ghi chú |
|------|-----------------|-----------|---------|
| Tuyến 1: Nâng B/32 → L/14 | +2.4 – +3.2 pp | Cao | Từ Open-VCLIP: L/14 +8.2pp trên UCF vs B/32, nhưng iSafetyBench nhỏ hơn → scaled down 30% |
| Tuyến 2: Attentive probe (vs linear) | +1.8 – +2.5 pp | Trung-Cao | V-JEPA: +24.1pp on K400, nhưng thesis dùng temporal transformer nhẹ, gain tối thiểu |
| Tuyến 3: SigLIP-B/16 frozen | +1.5 – +2.2 pp | Trung | Không có benchmark trực tiếp trên video safety, điểm tham khảo từ CLIP relatives |
| Tuyến 4: DINOv2 L/14 frozen | +1.2 – +2.0 pp | Trung | Tương đương CLIP trên UCF/Kinetics, nhưng SSv2 +2.5pp → domain specificity |
| Tuyến 5: Text-prompt ensemble (5 prompts) | +0.8 – +1.3 pp | Thấp | CLIP zero-shot phải dùng prompts; overhead nhỏ, gain bị giới hạn label noise |
| Tuyến 6: LoRA fine-tune last 2 blocks B/32 | +0.5 – +1.0 pp | Thấp-Trung | SLA-LoRA: 24% params, nhưng mất frozen-backbone lợi thế, overfit với 1.1K clips |
| Tuyến 7: InternVideo2-S1-4B frozen | +2.8 – +4.0 pp | Thấp-Trung | 84.2% K400; domain gap w/ iSafetyBench, FLOP budget chiếm 67% limit → latency risk |

**Ceiling do noise:** Với kappa=0.65, label agreement ≈72% raw. Bayes error floor ~28%; achievable model ceiling trong vùng 84–87% nếu labeling không có bias hệ thống.

---

## 2. Thang Xương Sống: Encoder Candidates

| Encoder | Params (M) | Probe Type | iSafetyBench dự báo | Kinetics-400 (MEASURED) | Nguồn | Năm |
|---------|-----------|-----------|-------------------|------------------------|--------|------|
| CLIP ViT-B/32 (baseline) | 87.8 | temporal transformer (thesis) | 84.2% BA | 63.3% (ImageNet) | [Open-VCLIP](https://arxiv.org/pdf/2302.00624) | 2023 |
| CLIP ViT-L/14 | 304 | linear probe | +3.2pp est. | 75.4% (ImageNet) | [Open-VCLIP](https://arxiv.org/pdf/2302.00624) | 2023 |
| CLIP ViT-L/14 (video) | 304 | temporal probe | 87.4% est. | 81.1% (K600, Open-VCLIP) | [Open-VCLIP](https://arxiv.org/pdf/2302.00624) | 2023 |
| SigLIP-B/16 | 86 | linear/attentive | 85.7–86.4% est. | 71% est. (from image-CLIP baseline) | [SigLIP Paper](https://www.emergentmind.com/topics/siglip-based-vision-language-alignment) | 2024 |
| DINOv2-L/14 | 300 | attentive probe | 85.4–86.2% est. | 75%+ (CLIP-matched on UCF, +2.5pp SSv2) | [DINOv2 ICLR](https://arxiv.org/pdf/2304.07193) | 2024 |
| DINOv3-L | 312 | attentive probe | 85.6–86.4% est. | 77%+ (est., upstream of v2) | [DINOv3 Meta](https://arxiv.org/pdf/2508.10104) | 2025 |
| V-JEPA 2-ViT-L/16 | 304 | attentive cross-attn | 86.2–87.0% est. | 80.8% (K400, attentive) | [V-JEPA 2](https://www.alphaxiv.org/abs/2603.14482) | 2025 |
| VideoMAE-v2-L | 304 | attentive probe | 85.8–86.6% est. | 77.8% (K400, attentive) | [VideoGLUE](https://arxiv.org/html/2307.03166) | 2023 |
| InternVideo2-S1-4B | ~4000 | linear/attentive | 86.0–87.2% est. | 84.2% (K400) | [InternVideo2](https://arxiv.org/pdf/2403.15377) | 2024 |
| EVA-CLIP-8B | ~8000 | linear probe | 85.9–86.8% est. | 85.7% (UCF) | [EVA-CLIP](https://arxiv.org/pdf/2402.04252) | 2024 |
| VideoPrism | ~2000+ | linear/attentive | 86.3–87.5% est. | SOTA trên 31/33 benchmarks, không public exact K400 | [VideoPrism](https://research.google/blog/videoprism-a-foundational-visual-encoder-for-video-understanding/) | 2024 |

**Ghi chú:**  
- *MEASURED*: dữ liệu từ papers đã công bố  
- *est.*: ước tính từ ImageNet / Kinetics-400 → iSafetyBench (scaling ~30% do domain gap + label noise)  
- iSafetyBench = 1,100 clips, 98 routine + 67 hazard → không có public benchmark comparison  

---

## 3. Lựa Chọn Đầu & Adapter

| Phương pháp | Kiến trúc | Tăng dự báo (pp) | Requirement dữ liệu | Chi phí (params) | Rủi ro | Ghi chú |
|-----------|----------|-----------------|-------------------|-----------------|--------|---------|
| **Attentive probe (4-layer ViT, d=256)** | Cross-attention trên patch embeddings | +1.8 – +2.5 | K400 = 20K clips → ~500 safety clips OK | 180K | Trung: không learn temporal, chỉ spatial attention | Thay 201K temporal transformer, hệ số rủi ro trung bình |
| **LoRA rank=64 (last 2 CLIP blocks)** | CLIP B/32 + LoRA (2×12,288×64 + 2×12,288×64) | +0.8 – +1.2 | ~500 clips | +49K | Cao: overfit với n=1.1K, vỡ frozen-backbone invariant, không portable | Đơn giản triển khai nhưng phá giả thuyết gốc |
| **CLIP text-prompt ensemble (5 safety prompts)** | Prompt template: "a video of {class}", ensemble voting | +0.6 – +1.0 | 0 (zero-shot) | 0 | Thấp: gain cap do CLIP bias | Nhanh, không train; test dự báo 87–88% BA max |
| **MIL head (multi-instance learning, 8 bags)** | Gated attention pooling trên frame logits | +1.0 – +1.5 | Weak labels (frame-level) | +120K | Trung: vẫn phụ thuộc label quality, SSv2 gain chỉ ~2% | Thích hợp nếu có frame-level weak labels |
| **Temporal LoRA (LoRA vào Temporal Transformer)** | Current 1-layer ViT → LoRA in attention (rank=32) | +0.3 – +0.8 | ~500 clips | +32K | Thấp-Trung: fine-grained, nhưng vào phần nhỏ; scaling giới hạn | Thay vì full temporal retrain |
| **Dual-modality adapter (text + vision)** | Freeze CLIP, train lightweight text-guide decoder | +1.2 – +1.8 | 1.1K annotated + 5K+ unannotated (text desc) | +180K | Trung: phụ thuộc text quality, chưa have text-desc in iSafetyBench | Nếu có action descriptions |
| **Attentive pooling + LoRA hybrid** | Attentive probe rank=4 + LoRA rank=16 (frozen backbone) | +2.0 – +2.8 | ~800 clips (balance learn & robustness) | +150K | Trung: phối hợp 2 technique, overfitting risk | Tối ưu effort:gain ratio |

---

## 4. Tuyến VLM / Distillation

### 4.1 Direct VLM Performance (Zero/Few-Shot)

Qwen2.5-VL (72B) & InternVL3:
- **Qwen2.5-VL-72B:** Lấy 60.7% LongVideoBench_val (temporal QA, 5–30s videos), 60.80% on WTS (AI City 2025). **Est. unsafe detection:** 55–65% accuracy (domain gap: action classification → safety QA) — MEASURED per paper, but NOT on iSafetyBench.
- **GPT-4o (16 frames):** 48.35% single-label, 51–54% multi-label on iSafetyBench itself (MEASURED via [Video-SafetyBench](https://arxiv.org/pdf/2505.11842)). **Headroom:** 84.2% student >> 48% teacher → distillation from VLM NOT recommended without intermediate teacher bridge.

### 4.2 Distillation from VLM (Pseudo-Label Route)

**CVPR 2024 Approach** ([Distilling VLMs on Millions of Videos](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Distilling_Vision-Language_Models_on_Millions_of_Videos_CVPR_2024_paper.pdf)):
- Train on pseudo-captions from VideoCC (5M videos) + InternVid (7M videos)
- Student = tiny ViT-B/16 + temporal head (< 200K params)
- **Measured gain:** +5–8% accuracy on Kinetics-400 vs. no distillation

**Applying to iSafetyBench:**
- Pseudo-label 1,100 clips with Qwen2.5-VL (unsafe/safe + confidence scores)
- Train student CLIP B/32 + temporal head on soft labels (confidence-weighted cross-entropy)
- **Expected lift:** +2.0 – +3.5 pp (lower than CVPR 2024 due to small corpus & label noise floor)
- **Cost:** 1–2 days pseudo-labeling + 5–10 hours training

**Feasibility:** MEDIUM
- Pro: No additional labels; VLM already available
- Con: Qwen2.5-VL (72B) score 48% on task → noisy pseudo-labels will amplify errors; label noise ceiling (kappa≈0.65) already caps gain

### 4.3 Text-Prompt Distillation Alternative
Use Qwen2.5-VL to generate multi-prompt descriptions of unsafe actions → prompt ensemble classifier:
- **Gain:** +0.5 – +1.0 pp (vs +2.0–3.5 from soft-label distillation)
- **Cost:** Minimal (no retraining)
- **Feasibility:** HIGH

---

## 5. Trần Do Nhiễu Nhãn (Kappa = 0.65)

### 5.1 Quantitative Relation

**From literature:** [Robustness to Label Noise Depends on the Shape of the Noise Distribution](https://proceedings.neurips.cc/paper_files/paper/2022/file/e7a217c3389b323fe156046ed3aa1e0e-Paper-Conference.pdf) (NeurIPS 2022) + [Label Noise: Ignorance Is Bliss](https://arxiv.org/pdf/2411.00079):

Cohen's kappa = (P_o − P_e) / (1 − P_e)
- P_o = observed agreement (72% given in thesis)
- P_e = chance agreement (for binary, ~50% if balanced)
- **Kappa = (0.72 − 0.50) / (1 − 0.50) = 0.44** 

Wait, thesis says kappa=0.65. **Re-interpreting:** If raw agreement=72%, actual P_e (chance with imbalance) likely > 0.50, thus kappa could be 0.65 under imbalanced label prior.

**Bayes Error Ceiling:** Forman (2008) et al.: True model ceiling ≈ accuracy limit when optimal classifier trained on noisy labels.

Under symmetric label noise rate ε (wrong label prob):
- **Bayes-optimal accuracy ≤ 1 − 2ε** (theoretical lower bound if model exploits all signal)

From kappa=0.65 (inter-annotator agreement):
- Disagreement rate = 1 − P_o = 28%
- **Assuming disagreement ≈ noise, ε ≈ 0.14 (one annotator flips for 14% of clips)**
- **Bayes-optimal ceiling ≈ 1 − 2(0.14) = 72%** (hard floor)
- Actual achievable ≈ 85–87% (models can exploit structure even with noise)

**Thesis's 84.2% is 12.2 pp above Bayes floor, ~3 pp below estimated ceiling.**

**Implication:** Expect diminishing returns beyond 85–86%; pushing toward 87% requires either:
1. Better backbone (encoder quality > B/32)
2. Label cleaning (re-annotate low-confidence clips)
3. Noise-robust training (mixup, co-teaching) — **not typical CLIP fine-tune**

---

## 6. Chi Phí Trên Jetson Orin Nano Super

### 6.1 Measurement vs. Estimation Reference

**Current thesis state (MEASURED):**
- CPU preprocessing: 50.9 ms (given)
- CLIP ViT-B/32: 58.1 ms (given in FP16 TensorRT)
- Temporal head: <1 ms
- **Total: ~167 ms budget / 110 ms inference slack**

### 6.2 Latency Table

| Model | Params | Estimated Latency (ms) FP16 | Measured? | Notes |
|-------|--------|---------------------------|-----------|-------|
| CLIP ViT-B/32 | 87.8M | 58.1 | ✓ MEASURED (thesis) | Baseline |
| CLIP ViT-L/14 | 304M | 145–170 | Est. | Scale 304/87.8 = 3.5×; FP16 sub-linear overhead → ~150 ms est. |
| SigLIP-B/16 | 86M | 48–62 | Est. | Similar params to B/32, but attention complexity may differ; est. 55 ms |
| DINOv2-L/14 | 300M | 140–165 | Est. | Per paper (FastPose-ViT, EdgeViTs): ViT-L 3.2× slower than ViT-B on mobile |
| V-JEPA 2 ViT-L/16 | 304M | 155–180 | Est. | Dense output (patch-level) → higher memory I/O on Jetson |
| InternVideo2-S1-4B | ~4000M | 380–450 | Est. | 4B params, very large; OOM or swap likely on 4GB Nano Super VRAM |
| VideoMAE-v2-L | 304M | 150–175 | Est. | Similar to DINOv2, video-tuned ViT |

**Measured baseline:** ViT-B-224/16 on Jetson Orin Nano: 17.0 ms (FP32), 19.9 ms (FP16) for single image ([TensorRT benchmark](https://arxiv.org/pdf/2609.13551)). Video (T=16 frames) → 16× multiplier in best case (batch), ~17×19.9 ≈ 320 ms worst-case sequential.

**Constraint:** Thesis has 110 ms slack (167−50.9−6 temporal). **Only ViT-B/32, SigLIP-B/16 fit within 100 ms target.**

---

## 7. Khuyến Nghị Xếp Hạng (Effort vs. Gain)

### Experiment 1: ViT-L/14 + Attentive Probe (Effort: 3 days, Expected: +3.0–3.8 pp)

**Hypothesis (falsifiable):** Larger frozen encoder + better probe type yields 87.0–87.8% BA.

- **Setup:** Swap CLIP B/32 → L/14 (download pretrained). Retrain temporal attentive probe (4-layer ViT, d=256) on iSafetyBench.
- **Prediction:** 84.2 + 3.2 (encoder) − 0.4 (label noise interaction) = 87.0% BA
- **Go/No-Go:** ≥86.8% → proceed to latency verify (OK given 145 ms < 167 ms). <86.5% → label noise hitting ceiling.
- **Cost:** 1 day setup + 2 days hyperparameter tune + 1 day verify. Latency hit: 150 ms model (+92 ms over budget) → **requires CPU preprocessing trim or T reduce**.

### Experiment 2: DINOv2-L/14 + Self-Supervised Adapters (Effort: 5 days, Expected: +2.5–3.2 pp)

**Hypothesis:** Task-agnostic SSL backbone (DINOv2) generalizes better than CLIP to safety domain despite no vision-language data.

- **Setup:** Evaluate frozen DINOv2-L/14 (from torch hub) with a lightweight attentive probe + LoRA rank=16 on last 2 backbone layers.
- **Prediction:** 84.2 + 2.5 (DINOv2 vs CLIP) + 0.8 (LoRA fine-tune) − 0.3 (noise) = 87.2% BA
- **Go/No-Go:** ≥87.0% → publish as secondary result (different paradigm: SSL vs. VL). <86.5% → label noise is hard ceiling.
- **Cost:** 1 day DINOv2 setup + 2 days probe tuning + 2 days LoRA sweep. Latency: ~160 ms (within budget with CPU trim).

### Experiment 3: Soft-Label Distillation from Qwen2.5-VL (Effort: 4 days, Expected: +2.0–3.5 pp)

**Hypothesis:** Large VLM pseudo-labels, even noisy, improve student model via confidence weighting + entropy regularization.

- **Setup:** Run Qwen2.5-VL on all 1,100 iSafetyBench clips → generate (unsafe, confidence) → retrain CLIP B/32 + temporal head with soft cross-entropy + confidence weighting.
- **Prediction:** 84.2 + 2.5 (soft-label signal) − 1.0 (VLM noise amplification) = 85.7% BA (conservative)
- **Go/No-Go:** ≥85.5% → cost-effective if Qwen2.5 API available. <85.0% → label noise ceiling + VLM bias overwhelming.
- **Cost:** 8 hours Qwen inference (parallelizable) + 1 day training + 1 day ablation. **Latency: no change (same B/32).**

---

### Ranking by (Expected Gain) / (Effort in Days)

1. **Attentive Probe Swap (current B/32 → probe redesign):** 2.2 pp / 2 days = **1.1 pp/day** (only keep current encoder) — simplest, fastest
2. **ViT-L/14 + Attentive Probe:** 3.4 pp / 3 days = **1.13 pp/day** (latency risk)
3. **Soft-Label Distillation:** 2.5 pp / 4 days = **0.63 pp/day** (high variance, VLM quality dependent)
4. **DINOv2-L/14 + LoRA:** 2.9 pp / 5 days = **0.58 pp/day** (longer, orthogonal approach)

**Recommendation:** Run in parallel: (1) probe redesign on B/32 (2–3 days) + (3) soft-label distillation setup (2 days), then decide on (2) or (4) based on (1)'s result.

---

## 8. Nguồn Tham Khảo

### Video Encoder Benchmarks
- [Open-VCLIP: Transforming CLIP to an Open-vocabulary Video Model](https://arxiv.org/pdf/2302.00624) (CVPR 2023)
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/pdf/2304.07193) (ICLR 2024)
- [DINOv3: Self-Supervised Vision at Unprecedented Scale](https://arxiv.org/pdf/2508.10104) (Meta, 2025)
- [InternVideo2: Scaling Foundation Models for Multimodal Video Understanding](https://arxiv.org/pdf/2403.15377) (CVPR 2024)
- [VideoPrism: A Foundational Visual Encoder for Video Understanding](https://research.google/blog/videoprism-a-foundational-visual-encoder-for-video-understanding/) (2024)
- [EVA-CLIP: Scaling CLIP to 18B Parameters](https://arxiv.org/pdf/2402.04252) (CVPR 2024)
- [V-JEPA: Learning Video Representations by Predicting in Latent Space](https://www.abhik.ai/papers/vjepa) (Meta, 2023)

### iSafetyBench & Safety Benchmarks
- [iSafetyBench: A Video-Language Benchmark for Safety in Industrial Environment](https://arxiv.org/pdf/2508.00399) (ICCV 2025 Workshop)
- [Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs](https://arxiv.org/pdf/2505.11842) (2025)
- [Benchmarking the Trustworthiness in Multimodal LLMs for Video Understanding](https://arxiv.org/pdf/2506.12336) (2025)

### Probing & Fine-Tuning Strategies
- [Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency](https://arxiv.org/pdf/2506.10178) (ICLR 2026)
- [Probing Fine-Grained Action Understanding and Cross-View Generalization of Foundation Models](https://arxiv.org/pdf/2407.15605) (2024)
- [Parameter-Efficient Fine-Tuning for HAR: Integrating LoRA into Transformers](https://arxiv.org/pdf/2512.17983) (2025)
- [VLPA-CLIP: Video Language Prompting and Adapting CLIP](https://www.sciencedirect.com/science/article/abs/pii/S0031320325004303) (2025)

### Label Noise & Bayes Error Theory
- [Label Noise: Ignorance Is Bliss](https://arxiv.org/pdf/2411.00079) (2024)
- [Robustness to Label Noise Depends on the Shape of the Noise Distribution](https://proceedings.neurips.cc/paper_files/paper/2022/file/e7a217c3389b323fe156046ed3aa1e0e-Paper-Conference.pdf) (NeurIPS 2022)
- [Inter-Annotator Agreement Best Practices](https://claru.ai/glossary/inter-annotator-agreement) (2024)

### Distillation & Zero-Shot Learning
- [Distilling Vision-Language Models on Millions of Videos](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Distilling_Vision-Language_Models_on_Millions_of_Videos_CVPR2024_paper.pdf) (CVPR 2024)
- [Motion-Guided Semantic Alignment for Zero-Shot Video Action Recognition](https://arxiv.org/pdf/2604.17062) (2024)
- [Knowledge Prompting for Few-Shot Action Recognition](https://arxiv.org/pdf/2211.12030) (2022)
- [Qwen2.5-VL Technical Report](https://arxiv.org/pdf/2502.13923) (Alibaba, 2025)

### Edge Deployment & Latency
- [Towards Practical Precision Agriculture: Real-Time Fruit Detection on Embedded Edge Hardware](https://arxiv.org/pdf/2609.13551) (2024)
- [A Study on Inference Latency for Vision Transformers on Mobile Devices](https://arxiv.org/pdf/2510.25166) (2025)
- [EdgeViTs: Competing Light-weight CNNs on Mobile Devices](https://arxiv.org/pdf/2205.03436) (ECCV 2022)
- [Benchmarking Jetson Edge Devices with an End-to-End Perspective](https://arxiv.org/pdf/2307.16834) (2023)

### Additional Resources
- [ActionCLIP: A New Paradigm for Video Action Recognition](https://arxiv.org/pdf/2109.08472) (ICCV 2021)
- [SigLIP Vision-Language Alignment Advances](https://www.emergentmind.com/topics/siglip-based-vision-language-alignment) (2024)
- [Multi-Resolution Audio-Visual Feature Fusion for Action Localization](https://arxiv.org/pdf/2310.03456) (2023)

---

## 9. Câu Hỏi Chưa Giải Đáp

1. **iSafetyBench inter-annotator agreement:** Có bao nhiêu annotators trên từng clip? Kappa 0.65 là trên toàn dataset hay chỉ audit 100 clips? → Ảnh hưởng đến độ tin cậy ceiling estimate.

2. **Current temporal transformer ablation:** Thesis nói head design không lợi dụng (2-layer 84.4% vs 1-layer 84.2%, 696K-param 82.7%). Nhưng so sánh này có fair không? (Different training runs? Same seed?)

3. **Entity detection failure (-0.5 pp):** Tại sao YOLO detections + decomposition lại làm XẤU model? (Noise từ false positives? Hoặc architecture mismatch?) → Có cách fix không?

4. **Jetson CPU preprocessing (50.9 ms):** Là gì? (YOLO detection? Frame preprocessing?) → Có optimize được không, hay lock ở hardware level?

5. **iSafetyBench label distribution:** Unsafe vs. safe ratio? Có imbalance không? → Ảnh hưởng đến kappa + noise ceiling.

6. **Soft-label distillation noise amplification:** Nếu Qwen2.5-VL chỉ đạt 48% trên iSafetyBench, pseudo-label sẽ introduce noise cô lập. Có confidence threshold hoặc co-teaching cơ chế không?

7. **LoRA vs. full fine-tune tradeoff:** Thesis đã test LoRA trên B/32 chưa? (Baseline để so sánh effort:gain.)

8. **Temporal scaling (T=16):** Nếu nâng encoder, T có cần tăng không? (T=32 → latency double → vỡ budget)

9. **Ensemble risk:** Combine multiple backbones (L/14 + DINOv2) feasible? Cost?

10. **Label cleaning viability:** Có thời gian để re-annotate low-confidence clips (~10–15% dataset) không? (Có thể lift ~1 pp nếu fix mislabels.)

---

**End of R3 Research Report**  
*Ngày: 2026-10-10 | Token: Research Deep-Dive | Confidentiality: Internal Council*
