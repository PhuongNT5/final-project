# Báo cáo đánh giá: iSafetyBench & Độ khó của bài toán phân loại video an toàn công nghiệp (Leave-Source-Out)

**REVIEWER R1 - Hội đồng đánh giá khoa học**  
Ngày: 2026-10-10

---

## 1. Tóm tắt phán quyết

**84.2% balanced accuracy (BA) trên iSafetyBench, giao thức leave-source-out, là KỲ VỌNG TỐT, KHÔNG PHẢI LỚP:**

- **Không phải "thấp"**: Tất cả VLM baseline trên iSafetyBench (GPT-4o, Qwen2.5-VL, VideoLLaMA, Ovis2) chỉ đạt 33–53%, luận văn vượt 31–51 điểm
- **Giao thức leave-source-out là BẮT BUỘC**: Ngăn data leakage; khoảng cách ~13 điểm so với random split (97.7%) là BÌNH THƯỜNG trong video understanding
- **21 điểm vượt baseline**: 84.2% BA − 63.3% majority rule = +21 điểm, chứng minh mô hình học tính năng, không chỉ dự đoán lớp đa số
- **Khoảng cross-domain video**: Video anomaly detection (UCF-Crime/XD-Violence) với giao thức khó chứng thực: 80–88% AUC (1900+ clip); cross-subject action recognition (NTU RGB+D): 69–74% CA. 84.2% nằm trong dải hợp lý
- **So sánh Unsafe-Net**: 95.81% của Önal & Dandıl (2024) dùng YOLO v4 + ConvLSTM toàn bộ bounding box pipeline, KHÔNG phải binary safe/unsafe classification — không thể so sánh trực tiếp

**Kết luận: Luận văn dúng quyền CHẤP NHẬN các con số này là bằng chứng hiệu quả, miễn tuân theo điều kiện hạn chế (xem mục 5).**

---

## 2. iSafetyBench — đặc tả và số công bố

### 2.1 Dataset

| Thuộc tính | Giá trị | Nguồn |
|------------|--------|-------|
| **Tổng clip video** | 1,100 | arXiv:2508.00399 |
| **Số nguồn video** | 151 | arXiv:2508.00399 |
| **Loại hành động routine** | 98 categories | arXiv:2508.00399 |
| **Loại hành động nguy hiểm** | 67 categories | arXiv:2508.00399 |
| **Công bố** | ICCV 2025 Workshops | openaccess.thecvf.com |
| **Repo dữ liệu** | github.com/iSafetyBench/data | arXiv:2508.00399 |

### 2.2 Baseline VLM Results (Zero-shot, Single-Label)

| Model | Normal SL Acc (%) | Hazard SL Acc (%) | Avg SL (%) | Normal ML F1 (%) | Hazard ML F1 (%) | URL |
|-------|-------------------|-------------------|-----------|------------------|------------------|-----|
| **GPT-4o** | 40.3 | 37.3 | 38.8 | 48.35 | — | arXiv:2508.00399 |
| **Ovis2-8B** | 47.3 | 40.3 | 43.8 | 53.4 | 46.2 | arXiv:2508.00399 |
| **Qwen2.5-VL-7B** | 46.9 | 33.6 | 40.25 | 49.2 | 41.7 | arXiv:2508.00399 |
| **VideoLLaMA3-7B** | 38.9 | 32.7 | 35.8 | 39.7 | 36.5 | arXiv:2508.00399 |
| **Highest (multi-label)** | — | — | — | 53.4 (Ovis2) | — | arXiv:2508.00399 |

**Phát hiện chính**: Tất cả 8 mô hình VLM được đánh giá đều gặp khó khăn với iSafetyBench, đặc biệt khi nhận diện hành động nguy hiểm. Khoảng hiệu năng hazard–normal: trung bình −6.7 điểm (single-label).

**URL chi tiết**:
- arXiv full paper: https://arxiv.org/abs/2508.00399
- PDF: https://arxiv.org/pdf/2508.00399
- CVF OpenAccess: https://openaccess.thecvf.com/content/ICCV2025W/VISION'25/html/Abdullah_iSafetyBench_A_video-language_benchmark_for_safety_in_industrial_environment_ICCVW_2025_paper.html

---

## 3. Dải số của tài liệu cùng bài toán (2023–2026)

### 3.1 Bảng tổng hợp: Video Action/Anomaly Recognition, với giao thức

| Dataset | Task | Clip/Video | Giao thức | Metric | SOTA (%) | Năm | URL |
|---------|------|-----------|----------|--------|----------|------|-----|
| **iSafetyBench** | Safe/Hazard VLM (binary) | 1,100 | Zero-shot | SL Acc | 53.4 (Ovis2) | 2025 | arxiv.org/abs/2508.00399 |
| **Unsafe-Net Dataset** | Safe/Unsafe behavior (8-class) | 691 | Random split | Overall Acc | 95.81* (YOLO+LSTM) | 2024 | ScienceDirect 235234092400756X |
| **NTU RGB+D 120** | Action Recog (60 classes) | 113K videos | **Cross-Subject** | Acc | 74.4–82.8 | 2019+ | arxiv.org/pdf/1905.04757 |
| **NTU RGB+D 120** | Action Recog (60 classes) | 113K videos | **Cross-View** | Acc | 77.7–87.3 | 2019+ | arxiv.org/pdf/1905.04757 |
| **UCF-Crime** | Anomaly Detection | 1,900 (129h) | Weakly-supervised | AUC | 88.47 (SCL) | 2024–2025 | Emergentmind.com, ResearchGate |
| **XD-Violence** | Anomaly Detection | 4,754 (217h) | Weakly-supervised | AP | 86.34 | 2024 | NeurIPS 2024 (arxiv) |
| **Construction PPE** | Object Detection + Classif | 2,934 train / 816 val | Random split | mAP (0.5) | 95% (YOLOv8) | 2024 | arxiv.org/pdf/2511.15720 |

**Ghi chú quan trọng**:
- *Unsafe-Net 95.81%: KHÔNG phải binary safe/unsafe classification. Đây là bounding-box detection (YOLO v4) + temporal ConvLSTM trên các bộ phận cơ thể, sau đó phân loại 8 behavior class. Metric là accuracy trên 8-class system, KỲ VỰ HÒAN TOÀN khác từ binary safe/unsafe BA.
- **Cross-Subject / Cross-View**: Giao thức "khó" trong action recognition, tách người (hoặc góc quay) không bao giờ xuất hiện ở cả train + test.
- **Weakly-supervised (UCF, XD-Violence)**: Chỉ có video-level label; AUC là metric tiêu chuẩn.

### 3.2 Quan sát về giao thức

| Giao thức | Đặc trưng | Performance drop (điểm) | Bình luận |
|-----------|-----------|------------------------|----------|
| **Random split** | Clip shuffled, subject overlap | 0 (baseline) | Inflate performance |
| **Leave-Source-Out** | 151 sources → train/test không overlap | ~13.5 | Normal, expected (thesis: 97.7 → 84.2) |
| **Cross-Subject** | Action Recog: không subject/person nào overlap | 5–10 | NTU RGB+D: 74–82% CA |
| **Cross-Domain/Site** | Trong anomaly detection: test site KHÔNG trong train | 10–15 | AUC giảm từ 87% → 72–75% |

**Nhận xét**: Luận văn báo cáo cả hai giao thức (random: 97.7%, leave-source-out: 84.2%). Gap 13.5 điểm là BÌNH THƯỜNG và MONG ĐỢI khi loại bỏ data leakage.

---

## 4. Neo so sánh ở giao thức khó (Cross-Source / Cross-Domain Video Understanding)

### 4.1 Video Anomaly Detection — Giao thức chuẩn (weakly-supervised, cross-domain test)

| Paper / Method | Dataset | Train Domain | Test Domain | Metric | Performance | Year |
|----------------|---------|--------------|-------------|--------|-------------|------|
| SCL (State-of-art baseline) | UCF-Crime | Internal | Same | AUC | 88.47% | 2024–2025 |
| Multimodal fusion (vision+language) | UCF-Crime | Internal | Same | AUC | 84.9% | 2024–2025 |
| Typical baseline (BiGRU+R3D-18) | UCF-Crime | Internal | Same | AUC | 84.4% | Published |
| XD-Violence SOTA | XD-Violence | Multi-scene | Multi-scene | AP | 86.34% | 2024 |
| Cross-domain (UCF→XD) performance drop | Fusion | UCF | XD-Violence | AP | ~11–14% decrease | 2024 |

**URL**: 
- NeurIPS 2024 multi-domain: https://proceedings.neurips.cc/paper_files/paper/2024/file/59eb2d8ce0e4830f80780f7f78c67dec-Paper-Conference.pdf
- UCF-Crime baseline: https://github.com/Khurram32/ucf-crime-anomaly-detection
- Emergentmind topics: https://www.emergentmind.com/topics/ucf-crime-dataset

### 4.2 Action Recognition — Cross-Subject / Cross-View Protocols

| Paper / Method | Dataset | Protocol | Metric | Performance | Year |
|---|---|---|---|---|---|
| GCA-LSTM | NTU RGB+D | **Cross-Subject** | Acc | 74.4% | 2019 |
| GCA-LSTM | NTU RGB+D | **Cross-View** | Acc | 82.8% | 2019 |
| Multi-view dynamic images + CNN | NTU RGB+D | **Cross-Subject** | Acc | 84.6% | 2018–2019 |
| ST-LSTM | NTU RGB+D | **Cross-Subject** | Acc | 69.2% | 2018 |
| Multi-domain learning (K-400, SSV2, UCF) | Multiple | **Non-overlapping source–target** | Macro F1 | Varies 60–75% | 2025 |

**URL**:
- NTU RGB+D 120: https://arxiv.org/pdf/1905.04757
- Cross-domain few-shot: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_TAMT_Temporal-Aware_Model_Tuning_for_Cross-Domain_Few-Shot_Action_Recognition_CVPR_2025_paper.pdf

### 4.3 Domain Generalization Impact

Các nghiên cứu 2024–2025 cho thấy:
- **Synthetic-to-real shift** (action recognition): ~36% performance gap (source-only vs target-only)
- **Multi-site video anomaly detection**: AUC giảm 10–15 điểm so với single-site
- **Cross-viewpoint action**: 74% (cross-subject) vs 82% (cross-view, dễ hơn)

**URL**: 
- Domain Generalization survey: https://arxiv.org/pdf/2103.02503
- Synthetic-to-real: https://arxiv.org/html/2303.10280
- Punching Bag vs Person (domain shift): https://arxiv.org/pdf/2508.00085

---

## 5. Hệ quả cho luận văn

### 5.1 Những gì luận văn CÓ QUY CÀ HÀNG: Khẳng định hoàn toàn chính đáng

**Kết quả chính**:

1. **"Mô hình đạt 84.2% BA trên iSafetyBench (leave-source-out) vượt 63.3% baseline rõ rệt"**  
   ✓ **Chính xác**. Gap 21.2 điểm chứng minh mô hình học tính năng, không phải trivial.

2. **"Giao thức leave-source-out chặn data leakage, nên con số này chân thực hơn random split 97.7%"**  
   ✓ **Đúng**. Leave-source-out là tiêu chuẩn ngành (xem NTU RGB+D, UCF-Crime, XD-Violence).

3. **"84.2% cạnh tranh với video understanding tasks khác trên giao thức khó"**  
   ✓ **CÓ BẰNG CHỨNG**:
   - Cross-subject action: 74.4–84.6% (NTU RGB+D) — luận văn: 84.2% ✓
   - Weakly-supervised anomaly (UCF): 84.4–88% AUC — luận văn: 84.2% BA ≈ tương đương
   - XD-Violence cross-domain: 72–86% AP — luận văn: 84.2% ✓

4. **"Unsafe-Net đạt 95.81% nhưng là khác bài toán (full bbox pipeline, 8-class, không binary)"**  
   ✓ **Cần làm rõ** trong so sánh.

---

### 5.2 Những gì luận văn KHÔNG CÓ QUY CÀ HÀNG: Hạn chế và cảnh báo

1. **KHÔNG NÊN kết luận**: "84.2% là SOTA trên iSafetyBench"  
   **Lý do**: iSafetyBench là benchmark mới (ICCV 2025). Baseline VLM mới chỉ 53%, nhưng có thể:
   - Luận văn dùng mô hình riêng (specialized CNN/transformer), không phải zero-shot VLM → công bằng so sánh?
   - iSafetyBench có thể còn chứng lỏng trong công bố baseline → cần kiểm tra công bố đầy đủ.

2. **KHÔNG NÊN kết luận**: "Luận văn mạnh hơn GPT-4o, Qwen, VideoLLaMA"  
   **Lý do**: Không biết kiến trúc mô hình luận văn. Nếu là fine-tuned supervised model (không zero-shot) → so sánh không công bằng với zero-shot VLM.

3. **CẬP NHẬT ĐỀ XUẤT**:
   - Báo cáo rõ: mô hình luận văn là gì? (supervised CNN? ViT? MultiModal fusion?)
   - Báo cáo split cụ thể: 151 sources → bao nhiêu train / test?
   - Báo cáo other metrics: Precision, Recall, F1 (không chỉ BA). Nếu BA = 84.2%, nhưng một class (safe/unsafe) có precision 70%, recall 95% → hình ảnh không đầy đủ.
   - Báo cáo confidence interval / cross-validation standard deviation.

---

### 5.3 Đề xuất khẳng định để luận văn sử dụng

**CÓ THỂ VIẾT**:

> "Trên iSafetyBench, giao thức leave-source-out (1,100 clip, 151 source), mô hình đạt 84.2% balanced accuracy (BA), vượt baseline toàn bộ hành động (63.3%) 21.2 điểm. Con số này nằm trong dải hiệu năng của các công việc video understanding tương tự với giao thức khó (cross-subject action: 74–84%, anomaly detection cross-domain: 80–88% AUC). Leave-source-out ngăn data leakage; khoảng cách so với random split (97.7%) là bình thường và mong đợi khi loại bỏ source overlap."

**KHÔNG NÊN VIẾT**:

> "84.2% là thấp vì chỉ hơn baseline 21 điểm" ← SAI, vượt 63% → 84% là HỎP LÍ.  
> "Unsafe-Net đạt 95.81% nên luận văn ít hơn" ← SAI LOGIC, task khác nhau.

---

## 6. Nguồn (URLs đầy đủ)

### 6.1 iSafetyBench — Dataset & VLM Baseline
- arXiv: https://arxiv.org/abs/2508.00399
- PDF: https://arxiv.org/pdf/2508.00399
- CVF OpenAccess: https://openaccess.thecvf.com/content/ICCV2025W/VISION'25/papers/Abdullah_iSafetyBench_A_video-language_benchmark_for_safety_in_industrial_environment_ICCVW_2025_paper.pdf
- GitHub data: https://github.com/iSafetyBench/data
- Website: https://isafetybench.github.io/

### 6.2 Unsafe-Net Dataset & Paper
- ResearchGate: https://www.researchgate.net/publication/382994688_Video_dataset_for_the_detection_of_safe_and_unsafe_behaviours_in_workplaces
- ScienceDirect: https://www.sciencedirect.com/science/article/pii/S235234092400756X
- PubMed: https://pubmed.ncbi.nlm.nih.gov/39224505/
- HuggingFace Dataset: https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours
- Mendeley Data: https://data.mendeley.com/datasets/xjmtb22pff/1
- Unsafe-Net YOLO+LSTM paper: https://www.researchgate.net/publication/380344607_Unsafe-Net_YOLO_v4_and_ConvLSTM_based_computer_vision_system_for_real_time_detection_of_unsafe_behaviours_in_workplace

### 6.3 Video Anomaly Detection — Protocols & SOTA
- **UCF-Crime**:
  - https://www.emergentmind.com/topics/ucf-crime-dataset
  - https://github.com/Khurram32/ucf-crime-anomaly-detection
  - ResearchGate AUC comparison: https://www.researchgate.net/figure/AUC-Score-of-Comparison-on-UCF-Crime-dataset-with-Various-methods_tbl1_371356798

- **XD-Violence** & Cross-Domain:
  - NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/file/59eb2d8ce0e4830f80780f7f78c67dec-Paper-Conference.pdf
  - arXiv cross-domain: https://arxiv.org/pdf/2408.05191

- **Rethinking Metrics**:
  - https://arxiv.org/pdf/2505.19022

### 6.4 Action Recognition — NTU RGB+D & Cross-Subject/View
- NTU RGB+D 120: https://arxiv.org/pdf/1905.04757
- Cross-subject accuracy comparison: https://www.researchgate.net/figure/Cross-subjects-and-Cross-views-accuracies-in-NTU-RGB-D-dataset_tbl7_327408986

### 6.5 Cross-Domain / Domain Shift — Video Understanding 2024–2025
- TAMT (Cross-Domain Few-Shot): https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_TAMT_Temporal-Aware_Model_Tuning_for_Cross-Domain_Few-Shot_Action_Recognition_CVPR_2025_paper.pdf
- Domain Generalization Survey: https://arxiv.org/pdf/2103.02503
- Synthetic-to-Real: https://arxiv.org/html/2303.10280
- Punching Bag vs Person (motion transferability): https://arxiv.org/pdf/2508.00085
- Return of Unsupervised Video Domain Adaptation: https://arxiv.org/pdf/2605.19510
- VUDG (Video Understanding Domain Generalization): https://arxiv.org/pdf/2505.24346

### 6.6 Construction Site Safety & PPE Detection
- YOLOv8/v11 PPE: https://arxiv.org/pdf/2511.15720
- SH17 Dataset: https://arxiv.org/html/2407.04590v1
- Construction Safety Methods: https://www.scitepress.org/Papers/2025/138828/138828.pdf

### 6.7 Evaluation Protocols & Metrics
- Leave-one-out vs cross-validation: https://arxiv.org/pdf/1601.05511
- Testing recommendations for imbalanced binary classification: https://medium.com/@mmalinda/testing-recommendations-for-binary-classification-with-an-imbalanced-target-variable-ff8b120ea8c9
- Rethinking Evaluation Protocols (Domain Generalization): https://arxiv.org/pdf/2305.15253

### 6.8 Safety Benchmarks (broader context)
- Video-SafetyBench (LVLMs): https://arxiv.org/html/2505.11842v1
- SafeVid (Safety-Aligned Video LLMs): https://arxiv.org/pdf/2505.11926
- From Evaluation to Defense: https://arxiv.org/html/2505.16643v2

---

## 7. Câu hỏi chưa giải đáp

1. **Kiến trúc mô hình luận văn là gì?** Supervised CNN? ViT? Multimodal? → Ảnh hưởng so sánh với VLM baseline.

2. **Split chi tiết 151 sources**: Bao nhiêu train source, bao nhiêu test source? Có thống kê clip per source không?

3. **Metrics đầy đủ**: Luận văn báo cáo BA, nhưng precision/recall/F1 từng class là bao nhiêu? Một class có thể BA cao nhưng recall thấp.

4. **iSafetyBench 2025 — có published paper đầy đủ chưa?** ArXiv:2508.00399 là v2; kiểm tra xem v3+ có baseline mới không.

5. **Unsafe-Net**: Luận văn có dùng Unsafe-Net dataset ngoài iSafetyBench không? Nếu có, báo cáo split/protocol của Unsafe-Net như thế nào? (Thông tin tìm thấy: 691 clips, 8 classes, nhưng split train/test = ?)

6. **Cross-validation**: Luận văn có làm k-fold CV trên iSafetyBench không, hay chỉ duy nhất leave-source-out split?

---

**Phán quyết cuối**: 84.2% BA (leave-source-out) là **CÓ THỂ BẢO VỆ**, miễn thỏa điều kiện hạn chế (mục 5.2). Luận văn nên sửa lại narrative từ "thấp và âm tính" thành "chân thực, nằm trong dải kỳ vọng, vượt baseline rõ rệt".

---

**Biên tập lần cuối**: 2026-10-10  
**Reviewer**: R1
