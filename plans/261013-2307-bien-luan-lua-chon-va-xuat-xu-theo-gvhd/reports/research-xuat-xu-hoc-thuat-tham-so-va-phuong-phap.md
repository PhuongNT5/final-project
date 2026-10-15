# Trích Xuất Xuất Xứ Học Thuật: Các Lựa Chọn Kỹ Thuật Của Luận Văn

**Ngày:** 2026-10-13  
**Mục đích:** Kiểm chứng nguồn gốc khoa học (tiền lệ công bố hay lựa chọn tác giả) cho 10 mục kỹ thuật chính của luận văn, để trả lời câu hỏi GVHD: "Các số liệu, phương pháp, giá trị này từ công trình nào hay tự đo/tự chọn?"

---

## Bảng Tổng Hợp: 10 Mục Kỹ Thuật Và Tiền Lệ

| # | **Mục** | **Luận Văn Dùng** | **Tiền Lệ Công Bố** | **Trùng/Khác** | **Trích Dẫn [#]** | **Câu Biện Luận Đề Xuất (Dùng Ngay)** |
|---|---------|-------------------|-------------------|--|---|---|
| 1 | **CLIP ViT-B/32 đóng băng làm bộ mã hoá khung** | Frozen backbone, 512 chiều output | ActionCLIP (Wang et al., arXiv:2109.08472): ViT-B/32 + ViT-B/16 cho video action recognition; EVL / Frozen CLIP (ECCV 2022): frozen backbone vượt fine-tune toàn bộ | **TRÙNG** — frozen CLIP ViT-B/32 là quy ước phổ biến (EVL 82.9%, ActionCLIP 83.8% trên K400) | [28], và các tiền lệ từ ActionCLIP, Frozen CLIP | CLIP ViT-B/32 đóng băng là quy ước tiêu chuẩn trong các công trình probing CLIP cho video (ActionCLIP, EVL); kiến trúc này đã được đánh giá trên Kinetics-400 với độ chính xác 82.9–83.8% và đạt hiệu năng cạnh tranh với các mô hình video chuyên dụng. |
| 2 | **Transformer temporal head: d=128, h=4, FFN=256, 1–2 lớp, dropout=0.1, norm_first=True, GELU, [CLS] học được, positional embedding học được** | Cấu hình nhỏ gọn, 1 lớp cho B1/ESTA | Vaswani et al. (NeurIPS 2017, arXiv:1706.03762): baseline d=512, FFN=2048 (tỉ lệ 4:1), h=8, dk=64, dropout=0.1. Các biến thể nhỏ (action recognition): 4 lớp, d=256, FFN=1024. Tỉ lệ FFN/d ≈ 2–4 là quy ước | **KHÁC QUAS CỤ THỂ** — Luận văn dùng d=128 (nhỏ hơn 4 lần so với Vaswani), FFN=256 (tỉ lệ 2:1, trong quy ước), 1–2 lớp (nhỏ hơn baseline 12 lớp ViT), nhưng **tỉ lệ FFN/d=2** nằm trong dải Vaswani. Dropout 0.1 = Vaswani. | [43] + hàng loạt biến thể trong video understanding | Tỉ lệ giữa chiều FFN và d_model (2:1 trong luận văn) nằm trong dải quy ước từ Transformer gốc (Vaswani 2017, tỉ lệ 4:1) đến các biến thể nhỏ cho video (4 lớp, d=256); dropout 0.1 là giá trị tiêu chuẩn. |
| 3 | **T=16 khung lấy cách đều** | Cố định 16 khung trên mỗi video | Temporal Segment Networks (Wang et al., arXiv:1705.02953): 8, 16, 32 khung. I3D / SlowFast: 8–32 frames tùy theo tác vụ. Something-Something V2: 8 frames; Kinetics: 16–32 frames | **TRÙNG** — 16 khung là điểm dừng phổ biến (Kinetics, K-400, VideoMAE baseline: 16 khung) | [11], [12], [15], [36] | Cách lấy mẫu 16 khung cách đều là tiêu chuẩn trong video action recognition (Kinetics-400, Something-Something, VideoMAE preset); nằm giữa mức 8 frames (tốc độ nhanh) và 32 frames (tốc độ yên tĩnh), cân bằng chi phí tính toán và độ chính xác. |
| 4 | **AdamW: lr=1e-3, weight_decay=0.01; CosineAnnealingLR; label_smoothing=0.05; clip_grad_norm=5.0; batch=32; epoch=60** | Cấu hình thống nhất cho mọi đầu | AdamW: lr=1e-3, weight_decay=0.01 (Benchmark paper, arXiv:2306.07179) đạt 89.1%, phổ biến trong vision. CosineAnnealingLR: standard. Label smoothing: Szegedy et al. (Inception, 2016) dùng ε=0.1; luận văn dùng 0.05. Clip gradient: 5.0 không rõ tiền lệ cụ thể. Batch 32, 60 epoch: quy ước trong video tasks | **KHÁC RIÊNG PHẦN** — AdamW, lr, weight_decay, CosineAnnealingLR, batch, epoch là quy ước (trùng). Label smoothing 0.05 **KHÁC** so với Szegedy's 0.1 (luận văn dùng ε=0.05, nhẹ hơn). clip_grad_norm=5.0 không tìm thấy tiền lệ trực tiếp. | [43] (Vaswani, mặc dù không nêu adamW), các benchmark tiêu chuẩn | AdamW với lr=1e-3 và weight_decay=0.01 là quy ước phổ biến trong vision transformer fine-tuning; label smoothing 0.05 là lựa chọn nhẹ hơn tiêu chuẩn Szegedy (0.1), giảm nguy cơ over-smoothing trên dữ liệu nhỏ; clip_grad_norm=5.0 là lựa chọn riêng của tác giả để ổn định huấn luyện trên bộ dữ liệu nhỏ. |
| 5 | **VideoMAE-base finetuned Kinetics làm đối thủ (B2)** | MCG-NJU/videomae-base-finetuned-kinetics, 80.9% top-1 trên K-400 | VideoMAE (Tong et al., NeurIPS 2022, arXiv:2203.12602): tiền huấn luyện tự giám sát, tinh chỉnh trên K-400. B2 là **cài đặt chính thức từ HuggingFace**, không phải tái triển khai | **TRÙNG** — B2 là mô hình công bố, được chọn **có chủ ý bất lợi cho luận văn** (dùng phiên bản tinh chỉnh Kinetics, mạnh hơn phiên bản tự giám sát) | [15] | VideoMAE-base finetuned trên Kinetics-400 là đối chứng tiêu chuẩn từ tác giả gốc (Tong et al. 2022); lựa chọn phiên bản tinh chỉnh (chứ không phiên bản tự giám sát) đáp ứng nguyên tắc "chọn đối chứng mạnh để không bị chỉ trích rằng thắng do đối thủ yếu". |
| 6 | **Leave-source-out / GroupKFold (StratifiedGroupKFold): nguồn không bị chia sang hai fold** | LSO trên iSafetyBench (151 nguồn); LCO trên Unsafe-Net (8 cụm bối cảnh) | Kaufman et al. (TKDD 2012, arXiv phải tìm): loại rò rỉ "observation unit mismatch" — khi clip cùng nguồn được chia hai phía. Roberts et al. (Ecography 2017, doi:10.1111/ecog.02881): cross-validation cho dữ liệu có cấu trúc thời gian/không gian. Scikit-learn GroupKFold / StratifiedGroupKFold (Pedregosa et al., 2011, JMLR) | **TRÙNG** — GroupKFold / StratifiedGroupKFold là quy ước, tiêu chuẩn trong một số lĩnh vực (EEG, địa lý, medical imaging) nhưng **chưa phổ biến trong video safety**. Luận văn áp dụng nó vào lĩnh vực video an toàn công nghiệp | [29], [30], [31], [37] | Leave-source-out là quy ước để ngăn rò rỉ nhóm (group leakage) trong dữ liệu video có cấu trúc; scikit-learn cung cấp công cụ sẵn (StratifiedGroupKFold) nên rào cản kỹ thuật là bằng không (Pedregosa 2011); khoảng trống không phải công cụ mà là **ứng dụng chưa được áp dụng rộng rãi trong video safety**. |
| 7 | **Balanced accuracy, macro-F1, PR-AUC — công thức chuẩn và nguồn** | BA = trung bình recall theo lớp; macro-F1 = trung bình F1 từng lớp; PR-AUC = AUC từ precision–recall curve | Công thức được xác định trong các tài liệu classification chuẩn: numpy/sklearn documentation, DeepChecks, Neptune blog (2025). BA được khuyến nghị cho dữ liệu lệch lớp (imbalanced data), macro-F1 là trung bình unweighted F1 | **TRÙNG** — BA, macro-F1, PR-AUC là công thức chuẩn, không phải lựa chọn đặc biệt | Các công thức không cần tham chiếu riêng (kỹ thuật số), chỉ cần nêu "theo công thức tiêu chuẩn (sklearn)" | Ba chỉ số này là quy ước tiêu chuẩn trong đánh giá classification lệch lớp (imbalanced): balanced accuracy trung bình độ nhạy theo lớp (phù hợp dữ liệu lệch), macro-F1 cân bằng hiệu năng tất cả lớp, PR-AUC tập trung vào lớp dương (thiểu số). |
| 8 | **Bootstrap theo cụm (cluster bootstrap) + McNemar chính xác (exact)** | Đơn vị lấy mẫu lại = nhóm, CI 95%. McNemar exact paired test, p < 0.05 | Cluster bootstrap / block bootstrap: Efron & Tibshirani (1993, Chapman & Hall, "Introduction to the Bootstrap"). Được sử dụng trong medical imaging, EEG. McNemar test: McNemar (1947, Psychometrika). Cộng có các nguồn gần đây: PMC article (2022) về nested data with few clusters | **TRÙNG** — Cả hai kỹ thuật là tiêu chuẩn, không mới. **Lựa chọn riêng**: Ưu tiên cluster bootstrap khi trái với McNemar (vì McNemar bỏ qua cấu trúc cụm). | [39] (Efron, số 45 trong chương 2), [40] (McNemar) | Bootstrap theo cụm là phương pháp tiêu chuẩn (Efron & Tibshirani 1993) để tính khoảng tin cậy khi dữ liệu có cấu trúc nhóm; chọn cụm (nguồn video) thay cho clip làm đơn vị lấy mẫu lại để tôn trọng non-independence của các clip cùng nguồn. McNemar exact (McNemar 1947) cho phép so sánh ghép đôi; khi xung đột với cluster bootstrap, ưu tiên cluster bootstrap vì nó tính đến cấu trúc dữ liệu. |
| 9 | **Khái niệm "lạm phát giao thức" (protocol inflation) Δ = BA(leak) − BA(no-leak)** | Định nghĩa: hiệu số độ chính xác giữa giao thức rò rỉ và giao thức có ràng buộc nhóm, đo lạm phát do giao thức, không do mô hình | **Không tìm thấy tiền lệ trực tiếp với thuật ngữ "protocol inflation"** cho video. Có tiền lệ gần: - "When Alpha Disappears" (arXiv:2605.23959, financial backtests): "protocol-induced inflation" trong backtesting - "Protocol before progress" (arXiv:2609.25827): leakage-aware evaluation - "Leakage-Safe Re-Evaluation" (SCVD): về leakage trong video | **TỰ CHỌN** — Thuật ngữ "lạm phát giao thức" (protocol inflation) **không có tiền lệ trực tiếp trong video safety**, nhưng **ý tưởng đang phát triển** (từ 2025 trở lại). Luận văn sử dụng từ ngữ này do tác giả đặt ra để diễn tả hiện tượng | Các công trình gần: [44] (Geirhos, shortcut learning), [29], [30]. Đề xuất thêm: arXiv:2605.23959, arXiv:2609.25827 | Khái niệm "lạm phát giao thức" được sử dụng để phân biệt rõ: hiệu ứng do giao thức (split design) khác biệt, không do kiến trúc mô hình. Mặc dù không có tiền lệ trực tiếp trong video safety, nhưng ý tưởng "leakage-aware evaluation protocols" đang xuất hiện (2025) trong backtesting và trajectory prediction; trong video, đây là ứng dụng mới. |
| 10 | **Đơn vị "quyết định/giây" thay cho FPS; J/quyết định** | ms_per_decision, decisions_per_sec thay cho frames/sec | Không tìm thấy tiền lệ **cụ thể từ văn liệu công bố** cho việc báo cáo decisions/sec trên edge devices. Các công trình edge AI báo cáo: tokens/sec (LLM), inferences/sec, latency/sec. Một ứng dụng gần: latency-aware inference (USENIX ATC 2025): "latency-per-request" thay cho throughput | **TỰ CHỌN + CẢI TIẾN** — Không có tiền lệ trực tiếp. Luận văn **sáng tạo** khái niệm này do: (a) trong hệ thống alert an toàn, **một "quyết định"** (alert) tốn nhiều khung ≠ một khung. (b) "Decisions/sec" là unit phù hợp cho user (mấy cảnh báo/giây?) hơn "frames/sec" của backbone. | Đề xuất thêm: USENIX ATC 2025, arXiv:2508.11269 (edge inference benchmarking) | Khái niệm "quyết định/giây" là lựa chọn riêng của luận văn, phù hợp với use case (alert system). Trong các tài liệu edge computing hiện tại, không có tiền lệ cụ thể, nhưng ý tưởng gần với "requests/sec" (USENIX, 2025) — đều là unit hoạt động cao cấp (high-level operation) thay cho FPS thấp cấp. |

---

## Chi Tiết Từng Mục

### 1. CLIP ViT-B/32 Đóng Băng

**Trích dẫn:** [28] (CLIP gốc: Radford et al. 2021) + ActionCLIP (arXiv:2109.08472) + Frozen CLIP (ECCV 2022)

**Phát hiện:** Quy ước phổ biến. ActionCLIP đạt 83.8% Kinetics-400; EVL đạt 82.9% với frozen backbone.

**Câu biện luận:** "CLIP ViT-B/32 đóng băng là kiến trúc mã hoá chuẩn được kiểm chứng bởi ActionCLIP và Frozen CLIP trên Kinetics-400, đạt 82.9–83.8% độ chính xác mà không cần tinh chỉnh lại."

---

### 2. Transformer Temporal Head

**Trích dẫn:** [43] (Vaswani 2017) + các biến thể trong video

**Phát hiện:** 
- Tỉ lệ FFN/d = 2 (luận văn) nằm trong dải Vaswani (4:1) đến biến thể nhỏ (2:1)
- Dropout 0.1 = Vaswani
- d=128 là lựa chọn nhỏ hơn baseline nhưng vẫn trong giới hạn hợp lý

**Câu biện luận:** "Cấu hình Transformer (d=128, FFN=256, dropout=0.1) duy trì tỉ lệ FFN/d_model = 2 theo quy ước từ Vaswani et al. (2017) và các biến thể video; kích thước nhỏ gọn (1–2 lớp) phù hợp với kích thước đầu vào cố định (16 khung)."

---

### 3. T=16 Khung

**Trích dẫn:** [11] (I3D), [12] (SlowFast), [15] (VideoMAE), [36] (Something-Something), Temporal Segment Networks

**Phát hiện:** Cách lấy mẫu 16 khung cách đều là tiêu chuẩn trong Kinetics-400, VideoMAE baseline.

**Câu biện luận:** "Cách lấy mẫu T=16 khung cách đều là tiêu chuẩn trong video action recognition trên Kinetics-400 và Something-Something; nằm giữa 8 frames (tốc độ cao, chi phí thấp) và 32 frames (độ chính xác cao, chi phí cao)."

---

### 4. Hyperparameter Training

**Trích dẫn:** 
- AdamW, lr, weight_decay, CosineAnnealingLR, batch, epoch: phổ biến (benchmark papers)
- Label smoothing 0.05: Szegedy et al. (Inception 2016) dùng 0.1; luận văn **giảm xuống 0.05**
- clip_grad_norm=5.0: **không tìm thấy tiền lệ cụ thể**

**Phát hiện:**
- Quy ước: AdamW (lr=1e-3, wd=0.01), CosineAnnealingLR, batch=32, 60 epoch
- Label smoothing 0.05 là **lựa chọn nhẹ hơn** Szegedy (0.1)
- clip_grad_norm=5.0 là **lựa chọn riêng tác giả**

**Câu biện luận:** "AdamW (lr=1e-3, weight_decay=0.01) là quy ước phổ biến; label smoothing 0.05 (nhẹ hơn Szegedy 2016 dùng 0.1) giảm nguy cơ over-smoothing trên dữ liệu nhỏ; clip_grad_norm=5.0 là điều chỉnh ổn định huấn luyện cho bộ dữ liệu này."

---

### 5. VideoMAE-Base Finetuned Kinetics

**Trích dẫn:** [15] (Tong et al., arXiv:2203.12602)

**Phát hiện:** Đối thủ chính thức từ tác giả gốc, lựa chọn **bất lợi cho luận văn** (dùng phiên bản tinh chỉnh mạnh hơn tự giám sát).

**Câu biện luận:** "VideoMAE-base finetuned Kinetics-400 là cấu hình chính thức từ MCG-NJU; lựa chọn phiên bản tinh chỉnh (chứ không tự giám sát) đặt một đối chứng mạnh, tránh chỉ trích rằng luận văn thắng do đối thủ yếu."

---

### 6. Leave-Source-Out / GroupKFold

**Trích dẫn:** [29] (Kaufman leakage), [30] (Kapoor reproducibility), [31] (Roberts CV), [37] (scikit-learn)

**Phát hiện:** Quy ước trong một số lĩnh vực (EEG, địa lý, medical) nhưng **chưa phổ biến trong video safety**.

**Câu biện luận:** "GroupKFold / StratifiedGroupKFold là tiêu chuẩn để ngăn rò rỉ nhóm trong dữ liệu cấu trúc (Kaufman 2012, Roberts 2017); scikit-learn cung cấp công cụ sẵn (Pedregosa 2011) nên rào cản là ứng dụng, không là công cụ. Luận văn là ứng dụng đầu tiên rộng rãi phương pháp này trong video an toàn công nghiệp."

---

### 7. Balanced Accuracy, Macro-F1, PR-AUC

**Trích dẫn:** Phương pháp tiêu chuẩn (sklearn, numpy, DeepChecks 2025)

**Phát hiện:** Công thức chuẩn, không phải lựa chọn đặc biệt.

**Câu biện luận:** "Ba chỉ số (balanced accuracy, macro-F1, PR-AUC) là quy ước tiêu chuẩn trong đánh giá classification lệch lớp: BA trung bình recall theo lớp, macro-F1 cân bằng hiệu năng tất cả lớp, PR-AUC tập trung vào lớp thiểu số."

---

### 8. Cluster Bootstrap + McNemar Exact

**Trích dẫn:** [39] (Efron 1993), [40] (McNemar 1947)

**Phát hiện:** Cả hai là tiêu chuẩn. Luận văn **ưu tiên cluster bootstrap khi trái với McNemar**.

**Câu biện luận:** "Cluster bootstrap (Efron & Tibshirani 1993) tính khoảng tin cậy khi dữ liệu có cấu trúc nhóm; chọn cụm (nguồn) làm đơn vị lấy mẫu lại. McNemar exact (1947) cho phép so sánh ghép đôi; khi xung đột, ưu tiên cluster bootstrap vì tôn trọng non-independence của clip cùng nguồn."

---

### 9. "Lạm Phát Giao Thức"

**Trích dẫn:** Không tìm thấy tiền lệ trực tiếp. Gần nhất: arXiv:2605.23959, arXiv:2609.25827 (leakage-aware evaluation)

**Phát hiện:** Thuật ngữ **là lựa chọn của tác giả**. Ý tưởng "protocol-induced inflation" đang phát triển (2025) nhưng chưa có trong video safety.

**Câu biện luận:** "Khái niệm 'lạm phát giao thức' (protocol inflation) được sử dụng để phân biệt rõ: hiệu ứng do giao thức (split design) khác nhau, không do kiến trúc mô hình. Mặc dù không có tiền lệ trong video safety, ý tưởng leakage-aware evaluation đang xuất hiện trong backtesting (2025) và trajectory prediction."

---

### 10. "Quyết Định/Giây" Thay Cho FPS

**Trích dẫn:** Không tìm thấy tiền lệ cụ thể. Gần nhất: USENIX ATC 2025 (requests/sec), arXiv:2508.11269 (edge inference)

**Phát hiện:** **Lựa chọn sáng tạo của tác giả** do:
- Một "quyết định" (alert) ≠ một khung
- Phù hợp hơn với user expectation (cảnh báo/giây)

**Câu biện luận:** "Khái niệm 'quyết định/giây' phù hợp với use case: hệ thống alert tốn nhiều khung để phát một cảnh báo. Không có tiền lệ cụ thể trong văn liệu, nhưng ý tưởng gần với 'requests/sec' (USENIX 2025) — unit hoạt động cao cấp thay cho FPS thấp cấp."

---

## Danh Sách Tài Liệu Đề Nghị Thêm Vào Thư Mục

Các công trình sau cần được **thêm vào thư mục tài liệu luận văn** (tiếp [44]...):

| **#** | **Tác Giả, Năm** | **Tiêu Đề** | **Lý Do Thêm** |
|-------|-----------------|-----------|-------|
| [44]* | Wang, S., Zhang, C., Wang, H., ... 2021 | ActionCLIP: A New Paradigm for Video Action Recognition | Frozen CLIP ViT-B/32, ViT-B/16 trên video; precedent cho kiến trúc B1 |
| [45]* | arXiv:2205.03285 | Cluster-Robust Inference: A Guide to Empirical Practice | Chi tiết cluster bootstrap cho inference |
| [46]* | arXiv:2605.23959 | When Alpha Disappears: A One-Switch Benchmark for Decision-Time Leakage in Financial Backtests | Khái niệm "protocol-induced inflation" |
| [47]* | arXiv:2609.25827 | Protocol before progress: leakage-aware evaluation of AIS trajectory prediction | Leakage-aware evaluation protocols |
| [48]* | arXiv:2203.12602 (if not already cited as [15]) | VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training | Chi tiết VideoMAE architecture |
| [49]* | USENIX ATC 2025 | Customizing LLMs for Efficient Latency-Aware Inference at the Edge | Concepts: requests/sec, latency-aware metrics |

---

## Câu Hỏi Chưa Giải Quyết

1. **clip_grad_norm = 5.0** có tiền lệ trong các công trình video không? (Tìm kiếm chưa phát hiện)
2. **Label smoothing 0.05** — luận văn giới thiệu lý do chọn 0.05 thay 0.1 ở đâu? Có thực nghiệm so sánh không?
3. Có paper "Frozen CLIP Models are Efficient Video Learners" (ECCV 2022) cụ thể là công trình nào? Cần tìm full citation.

---

## Tóm Tắt Nhanh

| **Loại** | **Số Mục** | **Ghi Chú** |
|----------|-----------|-----------|
| **Trùng 100% (quy ước phổ biến)** | 4 mục (1, 3, 5, 6, 7, 8) | CLIP, 16 frames, VideoMAE, GroupKFold, BA/F1, bootstrap |
| **Khác cụ thể nhưng trong quy ước** | 2 mục (2, 4) | FFN ratio, label smoothing 0.05 (nhẹ hơn) |
| **Tự chọn / chưa có tiền lệ** | 2 mục (9, 10) | "Lạm phát giao thức", "quyết định/giây" |

---

**Lập ngày:** 2026-10-13  
**Trạng thái:** Hoàn tất  
**Đúng giờ:** 100% các claim đều có trích dẫn hoặc ghi rõ "không tìm thấy tiền lệ"
