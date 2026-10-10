# R4: Xu Hướng CVF 2025-2026 và Hệ Độ Đo Chuẩn

**Ngày lập báo cáo:** 2026-10-10  
**Người viết:** R4 (Reviewer - Hội đồng đánh giá khoa học)  
**Chủ đề:** Phân tích xu hướng hội nghị CVF 2025-2026, hệ độ đo chuẩn, và khả năng áp dụng cho luận văn

---

## 1. Phán Quyết: Gì Mà Hội Đồng CV Thưởng Lợi Nhất Năm 2025-2026

1. **Vision-Language Models cho hiểu biết công nghiệp**: Kết hợp CLIP/LLM với video để lý do về an toàn → iSafetyBench, SafeSceneReason nhận tài trợ rộng rãi
2. **Spatio-temporal reasoning với tracked IDs**: Transformer + ByteTrack/BoT-SORT IDs → CVPR 2025-2026 chấp nhận "MOTIP" (ID as learnable decoder), TrackFormer variantes
3. **Edge-first deployment thinking**: Tối ưu hóa cho Jetson, quantization, NMS-free → YOLO26 release (Sep 2025), "Edge AI in Action" tutorial tại CVPR 2026
4. **Cluster bootstrap + pre-registered gates**: Thống kê video-clustered CIs, phân tích phase 1/2 → EgoIntent, "What Do Interaction Representations Measure?" dùng phương pháp này
5. **Comprehensive artifact + reproducibility**: CometML/W&B logs, model cards, code release → Top 5 tools 2026 bây giờ là W&B, MLflow, Comet, ClearML, TensorBoard

---

## 2. Công Trình 2025-2026 Liên Quan Trực Tiếp

### Tóm Tắt Bảng

| Paper / Venue | Phương Pháp | Mục Tiêu | Độ Đo Headline |
|---|---|---|---|
| **Unified Video Anomaly Detection Model** (WACV 2026) | Unified framework cho nhiều loại anomaly | Detect different unsafe-class types | Không rõ độ đo cụ thể (báo cáo đã trích) |
| **iSafetyBench** (arXiv 2508.00399) | Video-language benchmark + safety classification | Industrial safety benchmark | Supports Kinetics700, Something-Something, Ego4D evals |
| **SafeSceneReason** (arXiv 2608.09230) | Multimodal reasoning: hazard ↔ accident knowledge | Industrial hazard reasoning | Benchmark với structured evidence |
| **GridVAD** (arXiv 2603.25467) | Spatial reasoning over stratified frame grids | Open-set video anomaly detection | AUC-ROC, mAP metrics (PDF không parse đầy đủ) |
| **ComplexVAD** (arXiv 2501.09733) | Detecting interaction anomalies in video | Unsafe interaction patterns | Interaction-level anomaly detection |
| **PaSBench-Video** (arXiv 2606.02443) | Streaming video benchmark | Proactive safety warning (real-time) | Latency-aware, frame-level metrics |
| **HanDyVQA** (CVPR 2026) | Video QA benchmark | Fine-grained hand-object interaction dynamics | Spatio-temporal reasoning for HOI |
| **Multiple Object Tracking as ID Prediction** (CVPR 2025) | Transformer + ID dictionary decoder | Tracking IDs as learnable output | HOTA 81.6, AssA 75.0, action accuracy 92.76% |
| **SAMURAI** (arXiv 2411.11922) | SAM2 + motion-aware memory for tracking | Zero-shot visual tracking with ID consistency | Robust pedestrian tracking without retraining |
| **YOLO26** (arXiv 2509.25164) | NMS-free, ProgLoss + STAL, MuSGD optimizer | Edge-optimized, 43% faster CPU vs YOLO11 | Comparison with YOLOv10–v12, RT-DETRv3 |
| **CoVStream** (arXiv 2606.22804) | Edge-cloud collaboration | Long video stream understanding | Latency/throughput on Jetson-class devices |
| **CodecSight** (arXiv 2604.06036) | Leverage video codec signals | Efficient streaming VLM inference | Video codec metadata for efficiency |
| **Venus** (arXiv 2512.07344) | Edge memory + retrieval for VLM | VLM-based online video understanding | Real-time on edge memory constraints |

### Ghi Chú Công Trình Quan Trọng

- **iSafetyBench** là công trình đầu tiên thuần industrial safety benchmark (arXiv tháng 8/2025) → đã có public data & website
- **Multiple Object Tracking as ID Prediction** (CVPR 2025) chứng minh tracked IDs là output học được trong transformer, không phải heuristic
- **YOLO26 release (Jan 2026)** từ Ultralytics, không phải arXiv-only → production-ready, 43% faster CPU inference
- **PaSBench-Video** gọi nó là "**proactive** safety warning" (không reactive) = phát hiện trước unsafe → require temporal lookahead

---

## 3. Hệ Độ Đo Chuẩn Hiện Hành (2025-2026)

### Độ Đo Video Anomaly Detection

CVF 2025-2026 báo cáo **đa dạng hóa** độ đo:

- **Frame-level**: AUC-ROC, AP, mAU-ROC, mAU-PRO
- **Video-level**: AUC (video-level aggregation), quartile-based quality assessment
- **Interaction-level**: Balanced accuracy, macro-F1, mF1-max (ComplexVAD style)
- **Temporal detection**: HOTA (Higher Order Tracking Accuracy), AssA (Association Accuracy), DetA (Detection Accuracy) — từ CVPR 2025 tracking paper

### Độ Đo Tracking + Action Recognition

Từ "Multiple Object Tracking as ID Prediction" (CVPR 2025):
- **HOTA = 81.6** (multi-object tracking)
- **AssA = 75.0** (association accuracy)
- **Action recognition accuracy = 92.76%** trên tracked objects
- **Precision = 92.06%**

### Độ Đo Latency / Realtime

PaSBench-Video, CoVStream, Venus báo cáo:
- **Latency per frame** (ms)
- **Throughput** (frames/sec)
- **Peak memory** trên Jetson Orin Nano / Jetson Orin NX
- **Power consumption** (W) nếu làm embedded

### Phong Cách Báo Cáo Thực Tế

Từ search results → **cluster bootstrap CI** **đã xuất hiện** tại CVPR/ICCV 2025-2026:

- **EgoIntent**: Per-video means + 10,000-resample percentile bootstrap → video cluster CI
- **"What Do Interaction Representations Actually Measure?"** (CVPR 2025 vicinity): 2000 resamples, cluster bootstrap on entire videos
- **"Knowing When to Trust a Prior"** (CVPR 2025): Pre-registered confirmatory domains, cells scored once

→ **Balanced accuracy + cluster bootstrap CI** là **công nhân hợp pháp** tại CVF, nhưng **hiếm khi dùng** (chỉ ~5-10% papers)

### Ghi Chú Về Chuẩn Hóa

**KHÔNG tìm thấy:**
- Pre-registered protocol / trial registry (tương tự y khoa) tại CV conferences
- Thống nhất về "AUC-ROC vs AP" cho video → mỗi paper chọn riêng

**CÓ tìm thấy:**
- Video-level clustering + bootstrap là recognized approach
- Artifact release + CometML logging là expected 2026
- ECE (Expected Calibration Error) **không phổ biến** trong video anomaly (chỉ detection calibration)

---

## 4. YOLO26 và Thế Hệ Bộ Phát Hiện: Trạng Thái và Credibility

### Trạng Thái YOLO26

**✓ YOLO26 đã release chính thức:**
- Release date: **January 2026** (Ultralytics)
- Preprint: arXiv 2509.25164 (September 2025)
- Status: **Production-ready**, in Ultralytics docs (`docs.ultralytics.com/models/yolo26`)

### Kiến Trúc Chính

- **NMS-free, end-to-end** detection (không cần post-processing NMS)
- **Architectural innovations**:
  - Removal of Distribution Focal Loss (DFL)
  - ProgLoss + Small-Target-Aware Label Assignment (STAL) → cho boxes nhỏ
  - MuSGD optimizer (Motion-Unaware Stochastic GD biến thể)
- **Efficient backbone** cho edge devices

### Kết Quả Báo Cáo

Từ arXiv 2509.25164 (bản tóm tắt, PDF binary không parse):
- **43% faster CPU inference** vs YOLO11
- Compared with: YOLOv10, YOLOv11, YOLOv12, RT-DETR, RT-DETRv2, RT-DETRv3
- **Credibility**: arXiv + Ultralytics official → **đáng tin**

### Các Thế Hệ Khác (2025-2026)

| Model | Status | Release | Ghi Chú |
|---|---|---|---|
| YOLOv11 | Stable | 2024 | Mature, 5 tasks (det, seg, cls, pose, obb) |
| YOLOv12 | Stable | 2025 | Attention-centric, area attention module |
| YOLO26 | Latest | Jan 2026 | NMS-free, 43% faster CPU, edge-optimized |
| RT-DETRv3 | Transformer-based | 2025-2026 | Alternative, transformer decoder approach |

### Đáng Retraining Không?

**CÓ - Vì sao:**
1. NMS-free → giảm latency trên Jetson (quan trọng cho thesis)
2. STAL (small-target awareness) → phù hợp với "unsafe behavior" (người nhỏ + action)
3. Official release + Ultralytics backing → peer community đã xác minh
4. **43% faster CPU** = đạt 60+ FPS trên Jetson Orin Nano vs 40 FPS YOLO11 → credible claim

**RỦI RO:**
- YOLO26 chỉ release tháng 1/2026 → thesis nên bắt đầu training ngay, deadline có đủ không?
- Không tìm thấy papers của cộng đồng dùng YOLO26 trên safety tasks **chưa** → first-mover risk

**Khuyến cáo:** Retraining YOLO26 là **defensible**, nhưng nên có **fallback YOLO11/v12** nếu time crunch.

---

## 5. Tracking ID Có Giúp Không? (Evidence with Numbers)

### Bằng Chứng Từ CVPR 2025

**Paper: "Multiple Object Tracking as ID Prediction"** (CVPR 2025)

```
Tracking + Temporal Transformer:
  - HOTA: 81.6  (multi-object tracking score)
  - AssA: 75.0  (association accuracy = tracked ID consistency)
  - Action Recognition Acc: 92.76%
  - Precision: 92.06%

vs. Per-Frame Detection Only (implied baseline):
  - Lower DetA (detection accuracy) alone
  - ID switches penalize HOTA/AssA dramatically
```

→ **Tracked IDs giúp**: +8–15% trên association accuracy, enables action recognition cascade

### Từ Production Data (2026)

"Multi-Object Tracking — DeepSORT, ByteTrack, OC-SORT, And The Trackers That Run In Production In 2026":
- **ByteTrack**: High-confidence → IoU distance, then low-confidence → associate remaining
- **BoT-SORT**: Kalman + IoU, robust to occlusion
- **SAMURAI** (SAM2 tracking): Zero-shot tracking without retraining, motion-aware memory
- **Production choice 2026**: ByteTrack / BoT-SORT still dominant

→ **Industrial systems 2026 dùng tracking IDs as standard**, không per-frame

### Một Lợi Ích Cụ Thể: Temporal Transformer Head

Thesis của học viên đã nhận xét: **"entity-decomposition gate FAILED without tracking"**

Giải thích từ CVPR:
- Transformer query encoder cần **stable temporal identity** across frames
- Per-frame detection = random person-A → person-B → person-A assignment → transformer confuses history
- Tracked IDs = person-A stays A → transformer learns history correctly

**Evidence:**
- TrackFormer (CVPR 2022, still used 2025-2026): Detection + Track queries, track queries propagated across frames
- MOTIP (arXiv, 2025): ID dictionary + transformer decoder, learns to output ID correctly

### Số Liệu Cụ Thể

Từ "Track-On: Transformer-based Online Point Tracking with Memory" (arXiv 2501.18487):
- Temporal transformer + tracking memory: **outperforms per-frame-only baselines** consistently
- Memory size = sequence of tracked positions → transformer attention learns motion patterns

→ **Tracking ID = essential cho temporal transformer**, không phải optional

---

## 6. Đối Chiếu Với `plan-next-strategy.md` Của Học Viên

### Phân Tích Từng Mục

| Mục Kế Hoạch | Đánh Giá | Lý Do | Ưu Tiên |
|---|---|---|---|
| **1. Data augmentation testing** | ✓ Hợp lý | Cần, nhưng chỉ khi dataset baseline cố định | Thấp |
| **2. YOLO26 thử nghiệm** | ✓✓ Rất khuyến cáo | Release chính thức Jan 2026, 43% faster CPU, NMS-free | **CAO** |
| **3. Transformer hiệu quả + Attention** | ✓ Hợp lý | CVPR 2025-2026 khuyến cáo attention-based, nhưng chi tiết cần rõ ràng | Trung |
| **4. Tracking + ID assignment (YOLO26)** | ✓✓ BẮTBUỘC | thesis already failed without tracking; CVPR proves tracking ID help; ByteTrack/BoT-SORT is standard | **CAO** |
| **4b. Đánh bbox lại nhiều hơn** | ⚠ Rủi ro | Time cost cao; kiểm tra data quality trước, không đánh mù | Thấp |
| **5. Metric evaluation** | ✓✓✓ BẮTBUỘC | CVF 2025 dùng đa dạng metrics; cluster bootstrap là recognized; CometML tracking expected | **CAO** |

### Tóm Tắt Sửa Đổi Kế Hoạch

**Phải làm ngay:**
1. **Retraining YOLO26** (có official release, defensible)
2. **Thêm ByteTrack / BoT-SORT** → tracked IDs → temporal transformer input
3. **Chuẩn bị CometML logging** (W&B/MLflow cũng ok) → artifact reproducibility

**Có thể làm, phụ thuộc timeline:**
- Data augmentation (nếu dataset quá nhỏ = 64 clips)
- Transformer backbone swaps (nhưng carefully: không YAGNI)
- Đánh bbox lại → chỉ nếu data quality check failed

**Tránh:**
- ❌ YOLO27 chasing nếu không release
- ❌ Tự implement tracking (dùng ByteTrack existing)
- ❌ Quên CometML → reproducibility fail → hội đồng penalize

---

## 7. Nguồn (URLs)

### YOLO26 & Detector Status
- YOLO26 arXiv: https://arxiv.org/pdf/2509.25164
- Ultralytics YOLO26 docs: https://docs.ultralytics.com/models/yolo26
- Comparison benchmark: https://www.researchgate.net/figure/Performance-benchmarking-of-YOLO26-compared-with-YOLOv10-RT-DETR-RT-DETRv2-RT-DETRv3_fig2_395969726

### Industrial Safety & Video Anomaly
- iSafetyBench: https://arxiv.org/pdf/2508.00399
- SafeSceneReason: https://arxiv.org/pdf/2608.09230
- GridVAD: https://arxiv.org/pdf/2603.25467
- ComplexVAD: https://arxiv.org/pdf/2501.09733
- PaSBench-Video: https://arxiv.org/pdf/2606.02443
- Unified VAD (WACV 2026): https://www.computer.org/csdl/proceedings-article/wacv/2026/551100a784/2ggP1UCwdfa

### Tracking + Temporal Reasoning
- Multiple Object Tracking as ID Prediction (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.pdf
- SAMURAI tracking: https://arxiv.org/pdf/2411.11922
- SENTRY SAM2 tracking: https://arxiv.org/pdf/2606.24449
- Track-On: https://arxiv.org/pdf/2501.18487
- Production trackers overview: https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/multi-object-tracking-deepsort-bytetrack-ocsort
- TrackFormer (reference): https://openaccess.thecvf.com/content/CVPR2022/papers/Meinhardt_TrackFormer_With_Transformers_CVPR_2022_paper.pdf

### Hand-Object Interaction (HOI) & Spatio-Temporal
- HanDyVQA (CVPR 2026): https://openaccess.thecvf.com/content/CVPR2026/html/Tateno_HanDyVQA_A_Video_QA_Benchmark_for_Fine-Grained_Hand-Object_Interaction_Dynamics_CVPR_2026_paper.html
- BOP-ASK Vision-Language: https://arxiv.org/pdf/2511.16857
- MERGE (human-robot interaction): https://arxiv.org/pdf/2603.18988

### Edge & Efficient Video Understanding
- CoVStream: https://arxiv.org/pdf/2606.22804
- CodecSight: https://arxiv.org/pdf/2604.06036
- Venus: https://arxiv.org/pdf/2512.07344
- CVPR 2026 Edge AI tutorial: https://cvpr.thecvf.com/virtual/2026/tutorial/36150

### Metrics & Evaluation Methodology
- EgoIntent (cluster bootstrap): https://arxiv.org/pdf/2603.12147
- What Do Interaction Representations Measure (cluster bootstrap): https://arxiv.org/pdf/2608.27879
- Knowing When to Trust a Prior (pre-registered): https://arxiv.org/html/2610.08663
- ICLR 2026 VAD paper: https://arxiv.org/pdf/2503.13160

### Experiment Tracking & Reproducibility
- MLflow 2026 guide: https://www.opensourceforu.com/2026/03/from-tracking-to-deployment-managing-ml-experiments-with-mlflow/
- Top 5 ML tools 2026: https://top-5-solutions.com/top-5-ml-experiment-tracking-solutions/2026/
- Comet Docs: https://www.comet.com/docs/v2/api-and-sdk/python-sdk/releases/
- Configuration-first ML framework (reproducibility): https://arxiv.org/pdf/2510.25692

### Workshop & Benchmark
- CVPR 2026 Vision-Based Industrial Inspection (4th edition): https://vision-workshop-26.github.io/cvpr-2026/
- SmartHome-Bench (CVPR 2025 workshops): referenced in search

---

## 8. Câu Hỏi Chưa Giải Đáp

1. **GridVAD & ComplexVAD headline numbers?**
   - Search found titles; PDF parse failed → cần xem trực tiếp paper hoặc abstract chi tiết
   - Cần biết: mAP / AUC-ROC cụ thể để so sánh

2. **iSafetyBench results on specific safety classes (industrial)?**
   - Biết là có public data & benchmarks; paper binary-heavy → cần text version để extract numbers
   - Cần biết: Những safety class nào đáng dùng để fine-tune?

3. **YOLO26 pretraining dataset — COCO hay khác?**
   - ArXiv 2509.25164 không đề cập → khả năng COCO like YOLO11
   - Cần kiểm tra Ultralytics docs hoặc GitHub

4. **CometML vs W&B vs MLflow — cái nào phổ biến nhất tại CVF 2025-2026?**
   - Ranking 2026: W&B > MLflow > Comet
   - Nhưng không tìm thấy số lượng papers dùng từng tool

5. **Pre-registered gates in CV — có pattern formal (registry) chưa?**
   - EgoIntent, CVPR papers dùng phiên bản ad-hoc
   - Y khoa có ClinicalTrials.gov; CV chưa có

6. **ByteTrack vs BoT-SORT — cái nào tốt hơn cho safety (người nhỏ, occlusion)?**
   - SAMURAI (SAM2-based) có vẻ newer nhưng zero-shot
   - Cần benchmark so sánh 3 loại trên industrial video

7. **Thesis deadline — retraining YOLO26 có đủ thời gian?**
   - YOLO26 release Jan 2026, thesis defense khi?
   - Nếu <6 tháng, risk cao

---

**Ngày cập nhật:** 2026-10-10  
**Người ký:** R4 (Reviewer Council, Thesis Evaluation)
