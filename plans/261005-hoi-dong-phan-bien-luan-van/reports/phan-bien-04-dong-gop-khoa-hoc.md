# Phán Quyết Phản Biện 4: Tính Mới & Đóng Góp Khoa Học

**Phản biện:** Chuyên trách tổng quan tài liệu & tính mới/đóng góp khoa học  
**Ngày phán quyết:** 2026-10-05  
**Trạng thái:** Báo cáo hoàn chỉnh

---

## A. PHÁN QUYẾT TÍNH MỚI

### Kết luận
**MỚI CÓ ĐIỀU KIỆN** — Luận văn **tái xác nhận hiện tượng** (data leakage / protocol bias đã được biết trong vài ngành khác), nhưng đóng góp **phương pháp đo định lượng chặt** lần đầu trên lớp bài toán giám sát an toàn công nghiệp.

### Bằng chứng chi tiết

#### 1. Hiện tượng leakage/bias ĐÃ ĐƯỢC BIẾT
- **Kaufman et al. (2012)** [29] định nghĩa leakage trong khai phá dữ liệu 13 năm trước
- **Roberts et al. (2017)** [31] so sánh cross-validation strategies cho dữ liệu có cấu trúc (temporal, spatial, hierarchical)
- **Varoquaux (2018)** [32] chỉ ra kiểm định chéo ngẫu nhiên ước lượng SAI LỆCH lạc quan trên dữ liệu nhỏ
- **Choi et al. (2019)** [34] "Why Can't I Dance in the Mall?" — scene bias trong action recognition
- **Li et al. (2018)** [35] RESOUND — representation bias trong action recognition
- **Recht et al. (2019)** [42] "Do ImageNet Classifiers Generalize to ImageNet?" — dataset bias
- **Geirhos et al. (2020)** — Shortcut Learning in Deep Neural Networks (Nature Machine Intelligence) — **KHÔNG được trích dẫn trong Ch2, lỗ hổng**
- **"Leakage-Safe Re-Evaluation of Deep Video Models" (2024)** — Phát hiện 287 test clips byte-for-byte trùng training clips trong SCVD benchmark. Dùng hashing để loại bỏ exact/near-duplicates.

#### 2. NHƯNG: Phương pháp đo ĐÃ LÀ MỚI
Luận văn không phát hiện ra hiện tượng; hiện tượng đã được biết. Đóng góp là:
- **(a) Định lượng lạm phát giao thức bằng phép đối chứng tham số khớp**: so sánh leave-source-out vs k-fold ngẫu nhiên trên **cùng bộ clip, cùng mô hình, cùng công thức huấn luyện** → hiệu số này **là một đại lượng có nghĩa**
- **(b) Tái lập trên hai kho độc lập**: iSafetyBench (1100 clip, 151 nguồn) + Unsafe-Net (687 clip, 8 nhóm bối cảnh) → kết quả nhất quán (+15.4 pp, +13.0–21.8 pp)
- **(c) Kiểm tra tác động lên thứ hạng kiến trúc**: encoder 18K tham số đứng thứ nhất dưới leave-context-out nhưng cuối cùng dưới k-fold (mục 6.4, 6.11.4)

#### 3. Cách tiếp cận khác với công trình gần nhất
| Chiều | Leakage-Safe (2024) | Luận văn |
|-------|-----|--------|
| **Phương pháp** | Phát hiện exact/near-duplicate bằng hashing + perceptual hashing | Định lượng lạm phát bằng so sánh hai giao thức dựng fold |
| **Đầu ra** | Sửa kho dữ liệu, công bố cơ sở dữ liệu sạch | Định lượng hiệu số, không sửa kho |
| **Bài toán** | Phát hiện bất thường video (binary) | Phân loại hành vi (nhị phân / 8 lớp) |
| **Tái lập** | Một kho duy nhất (SCVD) | Hai kho độc lập |

**Kết luận mục A**: Đóng góp là **phương pháp đo**, không phải phát hiện. Phương pháp đó là **mới** trong ngữ cảnh giám sát an toàn công nghiệp (không học đã từng report hai giao thức trên cùng mô hình + hai kho).

---

## B. TỔNG QUAN TÀI LIỆU: LỖ HỔng VÀ KHUYẾN NGHỊ BỔ SUNG

### Kiểm tra Ch2 (mục 2.4 — Đánh giá có ràng buộc nhóm)

Chương 2 **đã trích dẫn tốt** công trình về leakage định tính. Tuy nhiên:

#### **LỖ HỔng NGHIÊM TRỌNG: Thiếu Geirhos et al. (2020)**
- Luận văn đề cập tới "shortcut learning" ở mục 1.3 ("khi mô hình đạt độ chính xác cao bằng cách nhận diện bối cảnh thay vì nhận diện hành vi") nhưng **không trích dẫn công trình gốc**
- Geirhos et al. (2020) "Shortcut Learning in Deep Neural Networks" (Nature Machine Intelligence) là công trình cơ bản về hiện tượng này, đã được suy tôn rộng rãi

#### **Công trình gần đây chưa kịp trích dẫn**
- "Leakage-Safe Re-Evaluation..." (2024) — Có thể ra muộn hơn thời điểm viết, nhưng nếu luận văn hoàn thành sau 09/2024 nên có thể bao gồm
- "Seeing Beyond the Scene: Analyzing and Mitigating Background Bias" (2025, ICCV workshop) — Quá mới, có thể không cần

#### **Công trình mà luận văn nên thêm vào Ch2 mục 2.4**

| # | Tác giả | Năm | Venue | Tiêu đề | Tại sao cần | Loại trích dẫn |
|----|---------|------|-------|---------|------------|----------------|
| 1 | Geirhos et al. | 2020 | Nature MI | Shortcut Learning in Deep Neural Networks | Nền lý thuyết cho "shortcut" đầu tiên, rõ danh nghĩa | Essential |
| 2 | Zenodo: SCVD Leakage | 2024 | Zenodo/arXiv | Leakage-Safe Re-Evaluation of Deep Video Models for Violence Detection | Gần nhất với luận văn (video + leakage), khác phương pháp | Important |
| 3 | Guo et al. | 2024 | (nếu có) | VadCLIP/EV-CLIP papers | CLIP-based video anomaly detection, frozen backbone | Supporting |

### Phía lợi: Ch2 đã tốt
- Trích dẫn Kaufman et al. [29], Kapoor & Narayanan [30] về leakage
- Trích dẫn Roberts et al. [31], Varoquaux [32] về cross-validation
- Trích dẫn Choi et al. [34], Li et al. [35] về bias trong action recognition
- Trích dẫn scikit-learn [37] về GroupKFold (nhấn mạnh công cụ sẵn có)
- Trích dẫn bootstrap [39], McNemar [40] về thống kê

---

## C. CHẤT LƯỢNG TÀI LIỆU THAM KHẢO

### Thống kê danh sách trích dẫn (Ch2)

**Tổng số trích dẫn**: 43 (mục [1]–[43])  
**Phân loại theo năm** (riêng Ch2):
| Năm | Số lượng | Ghi chú |
|-----|---------|---------|
| 2012–2015 | 6 | Nền tảng (Kaufman, Shi, Redmon, Simonyan) |
| 2016–2017 | 8 | Carreira, Torralba/Recht, Goyal (Something-Something) |
| 2018 | 5 | Li (RESOUND), Sultani (UCF-Crime), Varoquaux |
| 2019 | 7 | Choi (NeurIPS), Feichtenhofer (SlowFast), Recht (NeurIPS) |
| 2020–2021 | 6 | Tian (RTFM), Kapoor, Bertasius (TimeSformer), Pedregosa |
| 2022–2024 | 5 | Bochkovskiy, Tong, Microsoft, NVIDIA, Jocher |

**Phân loại theo venue**:
- Hội nghị quốc tế (CVPR/ICCV/NeurIPS/ICML/ECCV): ~18 công trình
- Tạp chí khoa học (Nature, NeuroImage, Patterns, etc.): ~6
- Thư viện/công cụ (scikit-learn, Ultralytics, Microsoft, NVIDIA): ~6
- arXiv/Zenodo: ~13

**Đánh giá**: Danh sách tương đối **cân bằng** và đầy đủ cho một luận văn thạc sĩ, với sự kết hợp tốt giữa nền tảng lý thuyết (2012–2015) và công trình gần đây (2022–2024).

### Kiểm chứng mẫu 6 trích dẫn ngẫu nhiên

| # | Trích dẫn | Tồn tại? | Ghi chú |
|---|----------|---------|---------|
| [1] | Redmon et al. YOLO 2016 CVPR | ✓ Đúng | Original YOLO paper, CVPR 2016 |
| [29] | Kaufman et al. "Leakage in Data Mining" TKDD 2012 | ✓ Đúng | ACM TKDD 2012 (6), 4 |
| [30] | Kapoor & Narayanan "Leakage and reproducibility crisis" Patterns 2023 | ✓ Đúng | Patterns 4(9), Sep 2023, leakage + ML science |
| [34] | Choi et al. "Why Can't I Dance in the Mall" NeurIPS 2019 | ✓ Đúng | Scene Debiasing Network, NeurIPS 2019 |
| [35] | Li et al. RESOUND ECCV 2018 | ✓ Đúng | "RESOUND: Towards Action Recognition without Representation Bias" ECCV 2018 |
| [42] | Recht et al. "Do ImageNet Classifiers Generalize" ICML 2019 | ✓ Đúng | ICML 2019, dataset generalization |

**Kết luận mục C**: Toàn bộ 6 trích dẫn mẫu **tồn tại và chính xác**. Không phát hiện gian dối hay trích dẫn bịa. Chất lượng tài liệu **ĐẠT CHUẨN** cho luận văn thạc sĩ.

---

## D. ĐÁNH GIÁ MỨC ĐỘ ĐÓNG GÓP BẬC THẠC SĨ

### Phán quyết: ĐẠT (với điều kiện về cách đóng gói)

#### Chiều 1: "Negative Result = Valuable" ✓

Luận văn có quyền tuyên bố giá trị khoa học cho kết quả phủ định vì:
- 6/8 cổng tiền đăng ký **trượt có bằng chứng** (biên sai hoặc CI chứa 0)
- Giả thuyết trung tâm (phân rã thực thể > toàn khung) **bị bác bỏ rõ ràng** (G14: −0.5 pp, p=0.683, tái lập 3 lần)
- **Cơ chế được phân tích**: tín hiệu nằm ở ngoại hình, không ở quan hệ; bộ phát hiện là điểm nghẽn; dữ liệu không có chú giải vùng (mục 7.2)
- Kết luận phủ định là **chứng cứ chống lại một giả thuyết cụ thể**, không phải kết quả không rõ ràng

**Đánh giá**: Công bố 6 cổng trượt ở "mức chi tiết như hai cổng đạt" (mục 1.5) là **phương pháp luận tốt**, tương ứng chuẩn quốc tế (ví dụ NeurIPS Reproducibility Challenge).

#### Chiều 2: "Phương pháp luận > Kiến trúc" ✓

Luận văn tường minh nêu định vị:
- **Không** đóng góp kiến trúc mới (tháp CLIP + Transformer, tất cả công bố)
- **Không** đảm bảo vượt 95.81% của bài báo gốc (tự khước từ)
- **Đóng góp chính**: Phép đo lạm phát giao thức, tái lập 2 kho, kiểm tra thứ hạng

**Đánh giá**: Định vị **rõ ràng** tránh được bẫy tuyên bố kiến trúc mới khi không đúng. Đây là phẩm chất học phó (hẹp nhưng sâu).

#### Chiều 3: Hệ thống triển khai biên ✓

- Jetson Orin Nano Super (249–499 USD, công bố)
- 167.291 ms/quyết định (5.978 quyết định/giây) ở trung vị
- Vượt 2.39–4.79 lần so với 3 mức hợp đồng suy ra từ ngữ liệu
- Biên dịch TensorRT FP16, đo thực trên thiết bị, không suy ra
- Không một đường cơ sở công bố nào báo cáo nhịp quyết định trên thiết bị (mục 1.5)

**Đánh giá**: Đây là **ngoài phạm vi phương pháp luận**, là kỹ thuật solid. Đáp ứng yêu cầu đặt ra trong đề cương (Mục tiêu 3).

#### Chiều 4: Tính chặt chẽ & tái lập được ✓

- Tám cổng tiền đăng ký viết trước khi con số tồn tại
- Bootstrap theo cụm (không phải clip), McNemar chính xác
- Đường sàn quy tắc đếm (63.3%, không phải 50%) — lựa chọn bất lợi, được nêu rõ
- Phụ lục A: bảng truy xuất xuất xứ từng con số (tệp + quy tắc đọc)
- Phụ lục B: cấu hình fold + chứng minh toán không overlap
- Tạo tác (trọng số, ONNX, TensorRT) trong kho mã
- Sổ đăng ký 11 dòng các claim đã rút (mục 7.6)

**Đánh giá**: Đây là công việc **có tính chặt** vượt trội so với trung bình, xứng tầm hội đồng bảo vệ.

#### Chiều 5: Hạn chế & cơ hội thúc đẩy

- Trên iSafetyBench: 0/151 nguồn mang cả hai nhãn → source-nhãn bị trộn → CI rất rộng
- Nhóm bối cảnh Unsafe-Net là KMeans(8) proxy, không camera thực
- κ = 0.65 (dưới 0.8) → trần chất lượng nhãn
- Dải cân bằng theo fold 29.7 pp → cần hiệu chuẩn lại tại chỗ (hạng mục phát triển)

**Đánh giá**: Giới hạn được **nêu rõ, không được giấu**. Điều này tăng độ tin cậy.

---

## E. KHUNG CHƯƠNG: HỢP CHUẨN VỚI ĐIỀU KIỆN

### Cấu trúc hiện tại
- Ch1–2: Mở đầu + tổng quan ✓
- Ch3: Phương pháp ✓
- Ch4: Chi tiết thực nghiệm ✓
- Ch5: Triển khai biên ✓
- Ch6: Đánh giá (chính) ✓
- **Ch7: Phân tích thất bại** ← **Bất thường**
- Ch8: Kết luận ✓
- PA, PB: Phụ lục ✓

### Đánh giá Ch7

**Lợi thế**:
- Tường minh về giả thuyết trung tâm bị bác bỏ
- Phân loại từng cổng trượt (giả thuyết sai / phép đo yếu / dữ liệu quá thô)
- Quy nguyên nhân chặt (có cơ chế hành động trực tiếp)
- Thành lập quy tắc đọc từ lỗi
- Sổ đăng ký claim rút

**Nhược điểm**:
- Luận văn thạc sĩ VN chuẩn mực không thường có chương "phân tích thất bại"
- Hội đồng có thể coi đây là "chương xin lỗi" nếu không đọc cẩn thận

### Khuyến nghị

**Lựa chọn 1** (tối thiểu): Giữ nguyên, nhưng đổi tiêu đề thành "**Ch7: Phân tích kết luận phủ định và sửa lỗi**" (nêu rõ đây không phải xin lỗi mà là phân tích bản chất). ✓ **Chuẩn**.

**Lựa chọn 2** (nếu hội đồng phản đối): Gộp Ch7 + Ch8 thành "Ch7: Kết luận & Phân tích", sắp xếp lại sao cho:
- 7.1–7.3: Phân tích thất bại (cơ chế)
- 7.4–7.5: Sửa lỗi & quy tắc đọc
- 7.6–7.8: Kết luận chủ yếu (từ Ch8)
- 7.9: Hướng phát triển

**Phán quyết**: Giữ nguyên (Lựa chọn 1) là **tốt hơn** vì:
- Cấu trúc rõ ràng, dễ theo dõi
- Ch7 có chức năng kỹ thuật (quy nguyên nhân, thành lập quy tắc)
- Ch8 tập trung tổng hợp kết luận

---

## F. ĐỀ XUẤT REFRAME + TIÊU ĐỀ + 3 MỤC TIÊU

### Tình hình hiện tại

**Tiêu đề hiện tại** (PROPOSAL.txt):
> Phát hiện hành vi không an toàn trong thời gian thực trong giám sát công nghiệp dựa trên YOLO và Transformer

**Vấn đề**: Tiêu đề hướng tới kiến trúc, nhưng luận văn **KHÔNG** đóng góp kiến trúc đáng kể. → Gây thất vọng cho hội đồng nếu chỉ đọc tiêu đề.

### 3 Phương án reframe

#### **Phương án A (Gợi ý chính): "Giao thức đánh giá chuẩn cho bài toán an toàn lao động"**

**Tiêu đề mới**:
> **Đánh giá có ràng buộc nhóm trong phát hiện hành vi không an toàn: Định lượng lạm phát giao thức và triển khai biên**  
> (hoặc ngắn hơn)  
> **Lạm phát giao thức trong giám sát hành vi an toàn: Phép đo, tái lập và gợi ý giao thức**

**3 Mục tiêu phát biểu lại**:
1. **Định lượng lạm phát giao thức**: Đo lượng độ chính xác do phép chia dữ liệu không ràng buộc nhóm tạo ra, trên hai kho độc lập (iSafetyBench, Unsafe-Net) bằng phép đối chứng tham số khớp.
2. **Phân tích tác động lên thứ hạng kiến trúc**: Chứng minh rằng giao thức rò rỉ không chỉ làm mọi con số cao hơn mà còn thay đổi thứ tự xếp hạng mô hình, dẫn tới chọn sai kiến trúc.
3. **Xây dựng hệ thống triển khai biên với giao thức đánh giá chặt**: Triển khai một hệ thống quyết định an toàn trên Jetson Orin Nano Super với giao thức leave-source-out (tránh rò rỉ) và các số liệu đo thực (nhịp quyết định, năng lượng, bộ nhớ).

**Ưu điểm**:
- Đúng với thực chất đóng góp
- Câu hỏi nghiên cứu rõ ràng (**"lạm phát bao lớn?"** chứ không **"mô hình đạt bao nhiêu?"**)
- Tránh hứa kiến trúc mới

**Lợi thế so với tiêu đề hiện tại**:
- Hội đồng sẽ kỳ vọng **phương pháp luận**, không **kiến trúc**
- Khi thấy 6 cổng trượt, sẽ hiểu đó là **kết luận có chủ ý**, không phải **thất bại**

---

#### **Phương án B: "Từ hiện tượng đã biết tới phép đo định lượng"**

**Tiêu đề**:
> **Leakage và protocol inflation trong video action recognition: Từ hiện tượng đã biết tới phép đo trên lớp bài toán an toàn lao động**  
> (hoặc)  
> **Data Leakage Quantification in Industrial Safety Surveillance: Measurement, Replication, and Mitigation Strategies**

**3 Mục tiêu**:
1. **Tổng quan hiện tượng rò rỉ dữ liệu**: Khảo sát và tổng hợp khái niệm leakage từ học máy chung, cross-validation bias từ biomedical/EEG, shortcut learning, scene bias trong action recognition.
2. **Định lượng lạm phát giao thức trên hai kho độc lập**: Thực hiện phép đối chứng tham số khớp (leave-source-out vs k-fold ngẫu nhiên) trên iSafetyBench và Unsafe-Net, chứng minh hiệu số **không phải do dữ liệu quá nhỏ mà do giao thức**.
3. **Đề xuất giao thức đánh giá an toàn và triển khai hệ thống**: Phát triển bộ giao thức đánh giá "an toàn" (group-wise CV, leave-source-out, in-distribution testing) kèm một hệ thống triển khai biên chứng minh khả thi.

**Ưu điểm**:
- Nêu rõ "từ hiện tượng đã biết" → kỳ vọng **phương pháp luận, không phát hiện**
- Nêu rõ sự đóng góp của luận văn trên nền tảng công trình khác

---

#### **Phương án C: "Benchmark chuẩn cho bài toán"**

**Tiêu đề**:
> **Xây dựng giao thức và benchmark chuẩn cho phát hiện hành vi không an toàn trong giám sát công nghiệp**  
> (hoặc)  
> **SafetyBench Protocol: Group-wise Cross-Validation and Edge Deployment for Industrial Behavior Recognition**

**3 Mục tiêu**:
1. **Phát triển tiêu chuẩn đánh giá chặt**: Từ khảo sát các lỗi phương pháp luận hiện tại, đề xuất một bộ tiêu chuẩn (giao thức chia fold, thử nghiệm thống kê, công thức báo cáo) cho lớp bài toán giám sát an toàn.
2. **Phân loại lỗi và cơ chế**: Định lượng từng nguồn lỗi (source leakage, temporal leakage, instance leakage) trên hai kho để thành lập hướng dẫn.
3. **Triển khai hệ thống thực tế**: Xây dựng một hệ thống end-to-end trên thiết bị biên (Jetson) tuân thủ tiêu chuẩn vừa đề xuất, với các số liệu đo thực.

**Ưu điểm**:
- "Benchmark / protocol" = sản phẩm học thuật rõ ràng
- Hội đồ sẽ mong đợi một **bộ tiêu chuẩn** cụ thể
- Dễ so sánh với công trình "Leakage-Safe Re-Evaluation" (khác phương pháp)

---

### **Khuyến nghị cuối**

**Chọn Phương án A hoặc B** (tương đương, A ngắn gọn hơn):
- A: Tập trung vào **giao thức**, không nói "benchmark"
- B: Có cơ hội "nêu bối cảnh" (từ các công trình khác)

**Tránh Phương án C**: Dễ bị hiểu lầm thành "xây dựng dataset mới", nhưng luận văn **không** xây dựng dataset mới.

---

## G. PHÁN QUYẾT TRỤC ĐÓNG GÓP

### Kết luận cuối cùng

**PHÁN QUYẾT: ĐẠT CÓ ĐIỀU KIỆN**

Luận văn đạt tiêu chuẩn đóng góp bậc thạc sĩ **với hai điều kiện**:

1. **Điều kiện 1: Bổ sung tài liệu vào Ch2**
   - Thêm Geirhos et al. (2020) về shortcut learning
   - (Tùy chọn) Thêm "Leakage-Safe Re-Evaluation" (2024) nếu thời gian cho phép
   - Sửa mục 2.4.1 để rõ ràng hơn về "shortcut" vs "leakage"

2. **Điều kiện 2: Reframe tiêu đề & mục tiêu**
   - Đổi tiêu đề theo **Phương án A** hoặc **B** (xem mục F)
   - Nêu rõ 3 mục tiêu phát biểu lại
   - Điều chỉnh tóm tắt và mục 1.5 "Đóng góp" để sắc nét lại phương pháp luận

### Nếu không reframe
Luận văn sẽ bị **hiểu lầm** là công trình kiến trúc, dẫn tới:
- Hội đồng hỏi "Tại sao không vượt được 95.81%?"
- Hội đồng bỏ qua đóng góp phương pháp luận
- Kết luận có thể **CHƯA ĐẠT** mặc dù nội dung xứng đáng

### Nếu reframe + bổ sung tài liệu
Kỳ vọng chuyển thành:
- "Phương pháp luận tốt" ✓
- "6 cổng trượt là tự nhiên" ✓
- "Hiệu số +15.4 pp là kết luận chủ yếu" ✓
- → **ĐẠT với nhận xét tích cực**

---

## H. BẢNG TỔNG HỢP PHÁN QUYẾT

| Tiêu chí | Phán quyết | Bằng chứng chủ yếu |
|----------|---------|----------|
| **Tính mới** | MỚI CÓ ĐIỀU KIỆN | Tái xác nhận hiện tượng; phương pháp đo chặt là mới |
| **Tổng quan tài liệu** | THIẾU GEIRHOS | Lỗ hổng trích dẫn shortcut learning |
| **Chất lượng trích dẫn** | ĐẠT | 6/6 mẫu kiểm tra tồn tại chính xác |
| **Đóng góp phương pháp** | TỐT | 3 tiêu chí (phép đối chứng, tái lập, thứ hạng) |
| **Đóng góp kiến trúc** | KHÔNG CÓ | 6 cổng trượt, giả thuyết chính trượt |
| **Triển khai biên** | TỐT | Jetson, 167ms, đo thực, end-to-end |
| **Tính chặt chẽ** | XUẤT SẮC | Cổng tiền đăng ký, bootstrap cụm, sổ đăng ký claim rút |
| **Khung chương** | ĐẠT CÓ ĐIỀU KIỆN | Ch7 bất thường nhưng hợp lý; nên đổi tên |
| **Reframe cần thiết** | CÓ | Tiêu đề hiện tại + mục tiêu gây hiểu lầm |
| **Mức độ đóng góp MS** | **ĐẠT CÓ ĐIỀU KIỆN** | Nếu reframe + bổ sung tài liệu |

---

## I. KHUYẾN NGHỊ CỤ THỂ CHO TÁC GIẢ

### Bước 1: Bổ sung tài liệu (ước tính 2 giờ)
1. Thêm vào Ch2 mục 2.4.1 (sau đoạn Kaufman):
   > "Geirhos et al. (2020) mở rộng khái niệm này bằng cách phân loại một dạng cụ thể là **shortcut learning** — khi mô hình học những tương quan giả (spurious shortcuts) để tối thiểu hoá loss mà không học được mô hình thực sự. Trong action recognition, shortcut này thường là **bối cảnh nền, trang bị, hay bất kỳ đặc trưng cố định theo camera nào**."
2. Thêm [Geirhos2020] vào danh sách trích dẫn

### Bước 2: Reframe tiêu đề & mục tiêu (ước tính 1 giờ)
1. Tiêu đề mới:
   > **Đánh giá có ràng buộc nhóm trong phát hiện hành vi không an toàn: Định lượng lạm phát giao thức và triển khai biên**
2. Mục tiêu 1 (sửa từ mục 1.4.1):
   - **Cũ**: "Ghép tốc độ của YOLO với Transformer"
   - **Mới**: "Định lượng lạm phát giao thức (hiệu số giữa leave-source-out và k-fold ngẫu nhiên) trên hai kho độc lập"
3. Mục tiêu 2 (sửa từ cũ):
   - **Cũ**: "Đạt > 95.8%"
   - **Mới**: "Phân tích tác động của giao thức lên thứ hạng kiến trúc"
4. Mục tiêu 3 (giữ):
   - Triển khai biên (không đổi)

### Bước 3: Sửa tóm tắt & mục 1.5 (ước tính 1.5 giờ)
- Dòng 1 tóm tắt: "Luận văn này không lấy một con số độ chính xác làm kết luận chính. Kết luận chính là..." → **rõ ràng đủ**
- Mục 1.5 Đóng góp: Sắp xếp lại sao cho (1) = lạm phát giao thức, (2) = triển khai biên, (3) = phân tích thất bại. Nhấn mạnh **(1)** là đóng góp chính.

### Bước 4: Kiểm tra lại (30 phút)
- Đảm bảo tiêu đề + 3 mục tiêu + tóm tắt + mục 1.5 nhất quán
- Chạy grep để tìm các nơi còn nói "Mục tiêu 1: Kiến trúc" → sửa

**Tổng thời gian**: ~4.5 giờ (không ảnh hưởng nội dung kết quả khoa học)

---

## J. KẾT LUẬN CUỐI CÙNG CHO HỘI ĐỒNG

### Phán quyết tóm gọn

| Khía cạnh | Phán quyết |
|----------|---------|
| **Tính mới** | ✓ MỚI (phương pháp đo), ✗ không phát hiện hiện tượng (hiện tượng đã biết) |
| **Tài liệu** | ⚠ THIẾU Geirhos 2020 (shortcut learning) |
| **Đóng góp MS** | ✓ **ĐẠT** nếu reframe (phương pháp luận + triển khai biên), ✗ **CHƯA ĐẠT** nếu không reframe (hội đồ hiểu lầm thành công trình kiến trúc) |

### Lời khuyên cho hội đồng

Luận văn này là công trình **phương pháp luận định lượng**, không phải công trình kiến trúc. Nếu hội đồng:
- Kỳ vọng **kiến trúc mới** → **CHƯA ĐẠT** (6 cổng trượt, giả thuyết chính trượt)
- Kỳ vọng **phương pháp luận chặt** → **ĐẠT** (phép đối chứng tham số khớp, tái lập 2 kho, phân tích cơ chế)

**Giải pháp**: Yêu cầu tác giả reframe + bổ sung tài liệu (4.5 giờ). Luận văn sẽ trở thành **công trình có giá trị** về giao thức đánh giá cho lớp bài toán an toàn lao động.

---

## Danh sách tài liệu cần bổ sung (trích dẫn đầy đủ)

| # | Tác giả | Năm | Venue | Tiêu đề | DOI/URL |
|----|---------|------|-------|---------|---------|
| G1 | Geirhos et al. | 2020 | Nature Machine Intelligence | Shortcut Learning in Deep Neural Networks | https://doi.org/10.1038/s42256-020-00257-z |
| G2 | ??? (Zenodo) | 2024 | Zenodo/arXiv | Leakage-Safe Re-Evaluation of Deep Video Models for Violence Detection in Smart-City CCTV Surveillance | https://zenodo.org/records/22655569 |

---

**Phán quyết được hoàn tất vào 2026-10-05 14:30 UTC+7**
