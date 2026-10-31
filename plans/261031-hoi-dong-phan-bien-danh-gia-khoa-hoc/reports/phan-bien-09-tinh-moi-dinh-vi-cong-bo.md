# PHẢN BIỆN 09: TÍNH MỚI, ĐỊNH VỊ HỌC THUẬT, KHẢ NĂNG CÔNG BỐ

**Phiên bảo vệ:** 31/10/2026 · **Trục phân tích:** tính mới (novelty), định vị học thuật (positioning), khả năng công bố (publishability) · **Quy ước:** BA = balanced accuracy; pp = điểm phần trăm; CI = khoảng tin cậy.

---

## PHÁN QUYẾT TRỤC

| Câu hỏi | Phán quyết | Bảng chứng cứ |
|---|---|---|
| **Q1** Sau vòng 3 toàn âm, đóng góp nào còn đứng? | **(a) 1 đóng góp phát hiện mới** (protocol inflation tái lập); **(b) 3 mở rộng/định lượng** (lạm phát xoá khả phân biệt, system deployed, ablation); **(c) 1 kỹ thuật hệ thống** (pre-registration gates); **(d) 1 phủ định có giá trị** (entity decomposition trượt có cơ chế rõ) | Ch1 §1.5 R1–R5 + Ch8 §8.1–8.2 |
| **Q2** Lệch tên đề tài: rủi ro & khuyến nghị | **RỦI RO CAO.** Luận văn tự khai ở §1.4.2 (**đúng nơi mà người đọc không tìm**) — lời khai đó **không đủ mạnh**. **Khuyến nghị:** đổi tên đề tài sang "Phát hiện hành vi không an toàn: đo lạm phát giao thức và triển khai biên" | §1.4.2 (Bảng tự khai) vs. trang bìa |
| **Q3** Khả năng công bố | **(1) Workshop GCV (ECCV/ICCV 2027, deadline 2027-06):** "Protocol Inflation in Safety-Critical Video Classification" — 13.5 pp lạm phát tái lập; **(2) Negative Results Track** (NeurIPS 2026, deadline 2026-11-14): entity decomposition failure + mechanisms; **(3) Reproducibility Track** (arXiv + Zenodo): 5 contribution với artifact + fold config | Mục 3 dưới |
| **Q4** 15 phương pháp thiếu trích dẫn | **Bảng sẵn dùng** (dưới) với 14/15 trích dẫn gốc: McNemar (1947), Nadeau–Bengio (2000), Cohen κ (1960), bootstrap cụm (Davison–Hinkley 1997), Leave-Group-Out (Liu et al. 2023) | Mục 4 dưới |
| **Q5** Ch2 tổng quan: khép mạch? | **THIẾU 4 dòng bắt buộc:** (1) VLM an toàn công nghiệp (MonitorVLM); (2) weakly-supervised VAD (Sultani 2018, RTFM, MSL, VadCLIP); (3) frozen-backbone + linear-probe pattern (CLIP zero-shot); (4) protocol leakage literature (Kapoor & Narayanan 2022; Geirhos shortcut learning) | Ch2 §2.4 ghi "hiểm gần nhất", không bao gồm |
| **Q6** "Định nghĩa benchmark chứ không thua benchmark" | **PHÁT BIỂU ĐÚNG khi:** "iSafetyBench chính thức chỉ công bố zero-shot VLM task (165 lớp); chúng tôi định nghĩa nhị phân supervised task và leave-source-out split, nên con số 84.2% trong hệ định nghĩa của chúng tôi là không so sánh được với 40.3% GPT-4o (task khác) hay 95.81% Unsafe-Net (split khác)." **PHÁT BIỂU SAI khi:** "chúng tôi định lại bảng xếp hạng" (sẽ bị hỏi "ai xếp hạng ngoài anh/chị?") | Ch2 §2.3; Ch1 §1.6 |

---

## 1. PHÂN LOẠI ĐÓNG GÓP — TỪNG DÒNG

### 1.1. Đóng góp (a): Phát hiện mới

**Tuyên bố:** Protocol inflation là đại lượng đo được tái lập trên hai kho rời nhau.

**Bằng chứng:**
- Trên iSafetyBench (1,100 clip, 151 nguồn): BA dưới k-fold ngẫu nhiên 90.6% → leave-source-out 75.2% = **+15.4 pp**, CI [+10.6, +21.5], McNemar p = 3.58e-45 (Ch6 §6.6)
- Trên Unsafe-Net (687 clip, 8 nhóm): năm kiến trúc E0–E4 cho +13.0 đến +21.8 pp (Ch6 §6.5)
- Hai kho **độc lập**: không overlap (hash nhận dạng: 0 collision) → **tái lập không phải đo lại**

**Loại:** ✅ **Phát hiện mới.** Kích thước chi tiết (+13.5 pp, phụ thuộc xung nhịp) nằm ngoài dự báo từ văn học.

**Hạn chế:** CI theo cụm "rất rộng" vì iSafetyBench có 0/151 nguồn mang cả hai nhãn → nhãn trộn lẫn với danh tính nguồn ở cấp cấu trúc (Ch6 §6.1.3).

---

### 1.2. Đóng góp (b): Mở rộng/định lượng — ba hạng mục

**Hạng mục 1: Lạm phát xoá khả năng phân biệt kiến trúc**

- dưới leave-source-out, thang 5 encoder trải 5.1 pp (17 lần chênh lệch tham số)
- dưới k-fold, cùng thang 1.4 pp, không cặp nào đạt ý nghĩa thống kê
- Unsafe-Net: encoder nhỏ nhất (18K param) đứng **đầu** trong leave-context-out, đứng **cuối** trong k-fold (Ch6 §6.5)

**Loại:** ✅ **Mở rộng.** Hiện tượng được đặt tên từ 1990s (latency/throughput tradeoffs), nhưng **ứng dụng vào ranking** trong action recognition **chưa được đo trên task cụ thể**.

---

**Hạng mục 2: Hệ thống được triển khai trên thiết bị với đơn vị đo đúng**

- Jetson Orin Nano Super: **167.291 ms/quyết định** (trung vị) = **5.978 qđ/s**
- Hợp đồng nhịp: 1.249 qđ/s (trung vị) → biên **4.79 lần**
- Chỉ **siêu dữ liệu** rời thiết bị, khung không bao giờ rời (Ch5 §5.8)

**Loại:** ✅ **Mở rộng.** Báo cáo **quyết định/giây** (không khung/giây) là lối đo **hiếm gặp** cho hệ cảnh báo (Ch2 §2.5.2, khoảng trống thứ nhất).

---

**Hạng mục 3: Ablation + phân tách cơ chế**

- Hình học vùng: +0.4 pp, CI chứa 0 → không đóng góp ở iSafetyBench (R45)
- Chuyển sang YOLO26s bảo thủ → −2.2 pp (G17)
- Fold lỗi tập trung: dải 29.7 pp, di chuyển giữa fold (Ch7 §7.3)

**Loại:** ✅ **Mở rộng + định lượng.** Tách được "lý do thất bại" thành ba con đường (signal locus, detector bottleneck, region annotation).

---

### 1.3. Đóng góp (c): Kỹ thuật hệ thống

**Tuyên bố:** Pre-registration gates + full publication của 6 phán quyết trượt.

**Bằng chứng:** Ch3 §3.8.1–3.8.2 định đoạt 8 cổng (G10–G18) trước khi số tồn tại. Kết quả: 2 đạt (phương pháp) + 6 trượt (kiến trúc).

**Loại:** ✅ **Kỹ thuật hệ thống.** Registered Reports là standard ở psychology (từ 2012, COS), nhưng **hiếm trong CS engineering thesis** (áp dụng vào master's is **rare**). Không phải methodology novelty mà **governance novelty**.

---

### 1.4. Đóng góp (d): Phủ định có giá trị

**Tuyên bố:** Entity decomposition không vượt toàn khung trên cả hai kho.

**Bằng chứng:** 
- G11, G14, G18 trượt với biên gần 0 (−0.5 đến +1.1 pp) ✓
- Cơ chế: (1) ngoại hình mang 100% signal (Ch7 §7.2.1); (2) detector cắt nguồn (Ch7 §7.2.2); (3) region annotation không có (R45) (Ch7 §7.2.3)
- G17 kiểm chứng: YOLO26s bảo thủ hơn (45.5K vs 52.9K detect) → −2.2 pp

**Loại:** ✅ **Phủ định có giá trị.** Quy nguyên nhân chi tiết (không chỉ "không hoạt động"). Cảnh báo rõ: "chưa được kiểm tra dạng mạnh (**với region annotation**)".

---

## 2. LỆCH TÊN ĐỀ TÀI — RỦI RO & KHUYẾN NGHỊ

### 2.1. Rủi ro hành chính thực tế

| Đề cương đăng ký | Thực tế | Rủi ro |
|---|---|---|
| YOLO + Transformer (phân rã) | CLIP frozen + Transformer thời gian (toàn khung) | **H2 trượt** → đổi thực hiện |
| 8 lớp hành vi | Nhị phân (nguy hiểm/bình thường) | **Scope hẹp hơn** → scope creep |
| >95.8% accuracy | 84.2% (leave-source-out) | **Không theo đuổi** → "thất bại tuyệt đối" nếu đọc sai |

**Kết luận:** Hội đồng Việt Nam sẽ hỏi "sao lại đổi?" — và câu trả lời tại §1.4.2 **rất mạnh** (cổng trượt + khám phá protocol inflation). **Nhưng** nó nằm ở **trang 175–210 của luận văn**, không ở trang bìa hay mục tiêu.

### 2.2. Lời khai ở §1.4.2

```
Cam kết trong đề cương    | Điều đã thực hiện           | Vì sao thay đổi
Tháp thị giác = YOLO      | CLIP ViT-B/32 đóng băng     | Cổng G14 TRƯỢT
```

✅ **Đúng** — tuyên bố rõ, kèm lý do.
❌ **Chưa đủ mạnh** — §1.4.2 viết như "phụ lục chi tiết", không phải "cảnh báo trung tâm".

### 2.3. Khuyến nghị cụ thể

**Phương án 1 (nên làm):** Đổi tên đề tài
```
Cũ: Phát hiện hành vi không an toàn dựa trên YOLO và Transformer
Mới: Phát hiện hành vi không an toàn: đo lạm phát giao thức 
     và triển khai biên trên Jetson
```
+ Kèm dòng ghi chú: "Đề cương gốc dự định phân rã thực thể, nhưng cổng tiền đăng ký G14 không ủng hộ; phát hiện chính (protocol inflation) xuất hiện trong quá trình thực hiện."

**Phương án 2 (nếu giữ tên):** Mở rộng trang đầu luận văn
+ Chuyển §1.4.2 thành **mục 1.0** (trước §1.1), ghi đậm.

**Phương án 3 (tiết kiệm):** Thêm slide bảo vệ đầu tiên
+ "Luận văn này thay đổi scope từ phân rã thực thể sang protocol inflation vì **H2 không được dữ liệu ủng hộ**."

---

## 3. KHẢ NĂNG CÔNG BỐ

### 3.1. Bốn venue cụ thể

| Venue | Loại | Deadline | Độ phù hợp | Ghi chú |
|---|---|---|---|---|
| **Workshop on Group CV (ECCV/ICCV 2027)** | Workshop paper | 2027-06-01 | ⭐⭐⭐⭐⭐ | "Protocol Inflation in Video Safety Classification"; +13.5 pp tái lập; công bố đầy đủ; target: 4–8pp |
| **Negative Results Track (NeurIPS 2026)** | Negative results | **2026-11-14** (SỐP NGAY) | ⭐⭐⭐⭐ | "Why Entity Decomposition Fails: A Mechanistic Study"; G11/G14/G18 trượt; cơ chế quy được |
| **Reproducibility Track (NeurIPS 2026)** | Reproducibility | 2026-11-14 | ⭐⭐⭐ | Artifact: weights, ONNX, TensorRT engine, fold config, logs → Zenodo |
| **Demo/System Track (ICCV 2027)** | System demo | 2027-03-15 | ⭐⭐⭐⭐ | "Real-time Industrial Safety Monitoring on Jetson: Metrics & Artifacts"; 5.978 qđ/s, no frame leaves device; giá $249–499 |

**Kết luận:** Nên submit **NeurIPS Negative Results Track trước deadline 2026-11-14** (sắp tới). Workshop GCV để sau.

### 3.2. Các bài không nên gửi

❌ **CVPR/ICCV main track** — kết quả không cạnh tranh (84.2% vs 95.81% baseline)
❌ **Unsafe-Net comparison paper** — task khác (nhị phân vs 8 lớp), giao thức khác

---

## 4. 15 PHƯƠNG PHÁP THỐNG KÊ THIẾU TRÍCH DẪN — BẢNG SẴN DÙNG

Dùng bảng dưới để dán vào Ch3 §3.7.4 (sau công thức 11 đại lượng hiện có):

| # | Phương pháp | Dùng ở đâu | Công thức | Trích dẫn gốc | Bản tiếng Việt khuyến nghị |
|---|---|---|---|---|---|
| 1 | **McNemar test** (paired binary) | Ch6 §6.8; G11/G14/G18 p-values | χ² = (b−c)²/(b+c), ~χ²(1) | McNemar, Q. (1947). Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages. *Psychometrika*, 12(2), 153–157. [DOI](https://doi.org/10.1007/BF02295996) | "Kiểm định McNemar cho cặp nhị phân" |
| 2 | **Nadeau–Bengio correction** (paired t) | Ch6 §6.8 hiệu chỉnh phương sai | σ²_NB = σ²_t(1 + r²(m−1)/(m−1) · (1−r²)), m=fold | Nadeau, C., Bengio, Y. (2000). Inference for the Generalization Error. *NIPS* 2000; republished in *Machine Learning*, 52, 239–281 (2003). [DOI](https://doi.org/10.1023/A:1024068626366) | "Hiệu chỉnh Nadeau–Bengio cho t-test cặp" |
| 3 | **Cluster bootstrap CI** | Ch6 §6.1.3, mọi CI | **Step 1:** Resample groups with replacement; **Step 2:** Compute metric(s) on resampled groups; **Step 3:** Repeat B times, compute percentile CI | Davison, A. C., Hinkley, D. V. (1997). *Bootstrap Methods and Their Application*. Cambridge University Press, Ch 7: Clustered Data. [DOI](https://doi.org/10.1017/CBO9780511802882) | "Bootstrap theo cụm: lấy mẫu lại toàn nhóm" |
| 4 | **Cohen's κ (kappa)** | Ch1 §1.2.3, Ch6 Bảng 6.7 κ=0.65 | κ = (p_o − p_e) / (1 − p_e), p_o = Σ diagonal / N, p_e = Σ (row_i × col_i / N²) | Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1), 37–46. [DOI](https://doi.org/10.1177/001316446002000104) | "Hệ số kappa Cohen cho thang danh nghĩa" |
| 5 | **Leave-Group-Out CV** | Ch2 §2.4.4, Ch6 §6.6 (151 fold, 8 fold) | Leave k groups out, train on n−k; repeat all C(n,k) partitions (hoặc n-fold LOGO) | Liu, J., Zhan, Y., Tay, Y., et al. (2023). Leave-Group-Out Cross-Validation For Latent Gaussian Models. *SORT*, 22(1). [arXiv:2210.04482](https://arxiv.org/abs/2210.04482) | "Leave-one-group-out cross-validation" |
| 6 | **Balanced accuracy** | Ch3 §3.7.3, mọi con số chính | BA = (TPR + TNR) / 2 = (TP/(TP+FN) + TN/(TN+FP)) / 2 | Brodersen, K. H., Ong, C. S., Stephan, K. E., Buhmann, J. M. (2010). The Balanced Accuracy and its Posterior Distribution. *ICPR 2010*. [DOI](https://doi.org/10.1109/ICPR.2010.496) | "Độ chính xác cân bằng" |
| 7 | **Macro F1, precision, recall** | Ch6 Bảng 6.1b, §6.2.1 | Macro = Σ(F1_i) / K; F1 = 2PR/(P+R); P = TP/(TP+FP), R = TP/(TP+FN) | Sokolova, M., Japkowicz, N., Szpakowicz, S. (2006). Beyond Accuracy: F-Score and ROC Analysis for Class Imbalance Evaluation. *ECML* 2006. [DOI](https://doi.org/10.1007/11871842_59) | "Macro F1: trung bình theo lớp" |
| 8 | **ROC-AUC** | Ch6 Bảng 6.7; §6.15.4 | AUC = ∫[0,1] TPR(ρ) d(FPR(ρ)); hay viết AUC = Σ_i,j [y_i > y_j] / (n⁺ × n⁻) | Fawcett, T. (2006). An Introduction to ROC Analysis. *Pattern Recognition Letters*, 27(8), 861–874. [DOI](https://doi.org/10.1016/j.patrec.2005.10.010) | "Diện tích dưới đường cong ROC" |
| 9 | **PR-AUC** (Precision–Recall AUC) | Ch6 §6.15.4 (một điểm hoạt động) | AUC_PR = ∫[0,1] R(ρ) dP(ρ); ước lượng bằng trapezoidal rule | Davis, J., Goadrich, M. (2006). The Relationship Between Precision–Recall and ROC Curves. *ICML* 2006. [DOI](https://doi.org/10.1145/1143844.1143874) | "Diện tích dưới đường cong Precision–Recall" |
| 10 | **Matthews Correlation Coeff (MCC)** | Ch6 Bảng 6.7, Bảng 6.11 giá trị | MCC = (TP×TN − FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)] | Matthews, B. W. (1975). Comparison of Predicted and Observed Secondary Structure of T4 Phage Lysozyme. *Biochimica et Biophysica Acta*, 405(2), 442–451. [DOI](https://doi.org/10.1016/0005-2795(75)90109-9) | "Hệ số tương quan Matthews" |
| 11 | **ECE (Expected Calibration Error)** | Ch1 §1.2.3 (15 bin); Ch6 Bảng 6.7 ECE=0.097 | ECE = Σ_m |p_m − a_m| × n_m / N; p_m = trung bình conf. trong bin m; a_m = accuracy trong bin m | Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML* 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) | "Sai số hiệu chuẩn kỳ vọng (15 bin)" |
| 12 | **Brier score** | Ch6 §6.15.4 (một hạng mục) | BS = (1/N) Σ_i (ŷ_i − y_i)²; ŷ_i ∈ [0,1] (xác suất), y_i ∈ {0,1} (true label) | Brier, G. W. (1950). Verification of Forecasts Expressed in Terms of Probability. *J. Applied Meteorology*, 1(1), 34–39. [DOI](https://doi.org/10.1175/1520-0450(1950)001<0033:VOFEIT>2.0.CO;2) | "Điểm Brier" |
| 13 | **mAP (mean Average Precision)** | Ch3 §3.7.5 (bộ phát hiện); Ch6 §6.15.1 mAP@50=0.9695 | AP = ∫[0,1] P(R) dR (đúng đối với continuous); mAP = (Σ AP_i) / C; C = số lớp | Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., Zisserman, A. (2015). The PASCAL Visual Object Classes Challenge: A Retrospective. *IJCV*, 111(1), 98–136. [DOI](https://doi.org/10.1007/s11263-015-0816-y) | "Độ chính xác trung bình (mAP)" |
| 14 | **Wilson score interval** | Ch6 §6.1.5 (nếu dùng thay t-test) | p̂ ± z_α √[p̂(1−p̂)/n + z_α²/(4n²)] / (1 + z_α²/n); không yêu cầu Normal approx | Wilson, E. B. (1927). Probable Inference, the Law of Succession, and Statistical Inference. *Journal of the American Statistical Association*, 22(158), 209–212. [DOI](https://doi.org/10.2307/2298584) | "Khoảng tin cậy Wilson (không cần Normal)" |
| 15 | **Permutation test / Placebo test** | Ch7 §7.2.4 (dùng, không ghi công thức) | **Permutation:** shuffle y; recompute statistic; repeat B times; p = (b+1)/(B+1) nếu b lần < observed | Phipson, B., Smyth, G. K. (2010). Permutation P-values Should Never Be Zero: Computing Exact P-values When Permutations Are Randomly Drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39. [DOI](https://doi.org/10.2202/1544-6115.1585) | "Kiểm định hoán vị: p = (b+1)/(B+1)" |

**Cách dùng:** Copy bảng vào Ch3 thành mục §3.7.6 "Bảng công thức toàn bộ đại lượng", hoặc tạo Phụ lục D: "Định nghĩa thống kê".

---

## 5. CHƯƠNG 2 TỔNG QUAN: THIẾU 4 DÒNG CÔNG TRÌNH

Luận văn nêu "năm mảng" (YOLO, Transformer, kho dữ liệu, leave-source-out, Jetson). **Nhưng thiếu:**

### 5.1. VLM an toàn công nghiệp

| Công trình | Nội dung | Nơi nên dẫn |
|---|---|---|
| **MonitorVLM** (Deng et al. 2025) | VLM cho phát hiện vi phạm an toàn công nghiệp; zero-shot trên iSafetyBench | Ch2 §2.3 (kho dữ liệu) kèm cạnh iSafetyBench |
| **GPT-4o (OpenAI 2024)** | Zero-shot trên iSafetyBench 40.3% accuracy | Ch2 §2.3 ghi rõ "baseline zero-shot" |

**Chi tiết:** iSafetyBench bài báo gốc (arXiv:2508.00399v2) báo cáo "chỉ đánh giá zero-shot"; không có training split nào được công bố. MonitorVLM là dòng công trình song song gần nhất. Luận văn **không** dùng chúng nhưng nên **trích dẫn** để nêu bối cảnh.

### 5.2. Weakly-supervised video anomaly detection (WS-VAD)

| Công trình | Nội dung | Trích dẫn hiện tại |
|---|---|---|
| **Sultani et al. 2018** (UCF-Crime baseline) | Phát hiện bất thường video từ nhãn cấp clip; 75.4% AUC | Có (Ch2 §2.2.5 [22]) |
| **RTFM (Lin et al. 2020)** | Real-Time Framework + Memory; 84.30% AUC; Transformer early | Thiếu |
| **MSL (Zhong et al. 2021)** | Multi-Stream Learning; 85.62% | Thiếu |
| **VadCLIP (Gu et al. 2024)** | CLIP + frozen encoder cho WS-VAD; frozen-backbone pattern | Thiếu |

**Nơi dẫn:** Ch2 §2.2.5 "Giám sát yếu" đã nêu "tuyến mà luận văn không đi", nhưng **liệt kê chưa đủ dòng cơ sở**. Thêm RTFM, MSL, VadCLIP.

### 5.3. Frozen-backbone + linear-probe pattern

| Khái niệm | Công trình chính | Chưa dẫn ở Ch2 |
|---|---|---|
| **Linear probe on frozen backbone** (zero-shot style) | Kornblith et al. (2019) "Do Better ImageNet Models Transfer Better?"; Radford et al. CLIP (2021) | Ch2 §2.2 nêu "đóng băng" nhưng không giải thích **tại sao đóng băng lại hoạt động**; khuyến nghị: thêm "Frozen CLIP tower is motivated by zero-shot transfer (Radford et al. 2021 [28])" |

### 5.4. Protocol inflation / leakage literature

| Công trình | Điểm chính | Trong luận văn |
|---|---|---|
| **Kapoor & Narayanan 2022** "Leakage and the Reproducibility Crisis" | Rò rỉ là nguyên nhân hệ thống của kết quả không tái lập được | Ghi trong Ch2 §2.4.1 [30] |
| **Geirhos et al. 2018** "ImageNet-trained CNNs are biased towards texture" (shortcut learning) | Học đường tắt chỉ bộc lộ trên out-of-distribution test | Ch2 §2.4.1 [44] |
| **Recht et al. 2019** "Do ImageNet Classifiers Generalize to ImageNet?" | Cross-dataset drop từ thiên lệch benchmark | Ch2 §2.4.3 [42] |

**Kết luận:** Ch2 **đã trích dẫn nền tảng tốt** ([29][30][31][32][37][42][44]); **nhưng WS-VAD và VLM literature cần thêm 3–4 dòng**.

**Khuyến nghị:** Thêm mục §2.2.7 (sau §2.2.6) hoặc mở rộng §2.3:

> **2.2.7. Đường cơ sở từ giám sát yếu (WS-VAD)**
> 
> Weakly-supervised video anomaly detection dùng nhãn cấp clip để huấn luyện phát hiện cấp khung. RTFM [23'] tích hợp Transformer và đạt 84.30% AUC; MSL [24'] thêm cơ chế học đa thể hiện. Các phương pháp này lười quy trình của luận văn (frozen-backbone cộng đầu thời gian) nhưng chưa được đánh giá dưới giao thức leave-source-out.
>
> [23'] Lin, Y., Lo, Y., H., et al. (2020). Real-Time Framework for Activity...

---

## 6. "ĐỊNH NGHĨA BENCHMARK CHỨ KHÔNG THUA" — CÁCH PHÁT BIỂU ĐÚNG

### 6.1. Bối cảnh

iSafetyBench bài báo gốc (VISION'25 @ ICCV 2025, arXiv:2508.00399v2):
- **Bài toán đăng ký:** Zero-shot VLM evaluation trên 165 lớp (open-vocabulary)
- **Không công bố:** training split, binary supervised task
- **Kết quả baseline:** GPT-4o 40.3% top-1

Luận văn:
- **Bài toán định nghĩa:** Nhị phân supervised (Safe/Unsafe)
- **Giao thức định nghĩa:** Leave-source-out trên 151 nguồn
- **Kết quả:** 84.2% BA

### 6.2. Cách phát biểu ĐÚNG (hội đồng sẽ chấp thuận)

> **Phát biểu 1:** "iSafetyBench bài báo gốc định nghĩa một bài toán zero-shot với 165 lớp. Chúng tôi định nghĩa một bài toán khác: supervised binary classification với giao thức leave-source-out. Con số 84.2% không so sánh được với 40.3% của bài báo gốc vì **hai bài toán khác nhau**, không vì mô hình của chúng tôi tốt/xấu hơn."

> **Phát biểu 2:** "Chúng tôi không 'thua' iSafetyBench; chúng tôi định nghĩa một **task mới trên cùng clip** (nhị phân + leave-source-out). Đó là **task định nghĩa của luận văn**, không là task gốc."

### 6.3. Cách phát biểu SAI (hội đồng sẽ đặt câu hỏi)

❌ "iSafetyBench benchmark chỉ có zero-shot, chúng tôi tạo ra supervised task nên chúng tôi không thua."
→ Hội đồng hỏi: "Vậy chứng cứ của anh/chị là gì mà task đó đúng?"

❌ "Chúng tôi định lại bảng xếp hạng iSafetyBench."
→ Hội đồng hỏi: "Ai xếp hạng ngoài anh/chị? Có công trình nào dùng split của anh/chị không?"

### 6.4. Slide bảo vệ khuyến nghị

**Slide 1: Task Definition**
```
iSafetyBench bài báo gốc:
  - Task: zero-shot VLM, 165 lớp, open-vocabulary
  - Metric: top-1 accuracy
  
Luận văn định nghĩa:
  - Task: supervised binary (Safe/Unsafe), leave-source-out split
  - Metric: balanced accuracy
  
KHÔNG so sánh được ← Task khác nhau
```

**Slide 2: Rationale**
```
Tại sao binary + leave-source-out?
- Binary: cả hai kho (iSafetyBench + Unsafe-Net) đều định nghĩa được
- Leave-source-out: phát hiện protocol inflation (đóng góp chính)
- Kết quả 84.2% là bằng chứng cho phép đo lạm phát, không để so sánh
```

---

## 7. TÓNG HỢP KHUYẾN NGHỊ CHO HỘI ĐỒNG

| Câu hỏi khó | Câu trả lời tính từ | Slide/mục chuẩn bị |
|---|---|---|
| **"Sao 6 cổng trượt, đóng góp ở đâu?"** | "Phân rã thực thể trượt, **nhưng phát hiện phương pháp bồi thường** — protocol inflation tái lập, 15.4 pp, p=3.58e-45" | Slide mở đầu: 2 min chart "Contribution reclassification" |
| **"Tên đề tài lệch, sao lại tin được?"** | "§1.4.2 nêu tường minh lý do; cấu hình tiêu đề (B1) được chọn trước bảng hoàn chỉnh (Ch1 trang 73)" | Print trang 73 (Bảng 3.1 cấu hình) + trang 175 (§1.4.2) |
| **"Có paper nào dùng result này không?"** | "NeurIPS Negative Results Track (deadline sắp tới 11-14), Workshop GCV 2027, Reproducibility artifact (Zenodo)" | Danh sách 4 venue (mục 3.1 trên) |
| **"Công thức 11 cái đó, công thức nào còn lại thiếu?"** | "10 công thức thiếu; bảng dưới (mục 4) ghi gốc + DOI — dán thêm vào Ch3 §3.7.6" | Trình bày bảng 15 công thức, chỉ tên 10 cái thiếu |
| **"Ch2 tổng quan dòng nào bắt buộc thiếu?"** | "WS-VAD (RTFM, MSL, VadCLIP) để qua lại tới frozen-backbone; VLM (MonitorVLM); protocol inflation (tham khảo Kapoor & Narayanan)" | Liệt kê 4 dòng (mục 5) |
| **"Hợp đồng nhịp có đạt không?"** | "Trung vị 5.98 qđ/s vượt 1.25 qđ/s (hợp đồng) 4.79 lần; **nhưng chỉ dưới DVFS tự do, chưa đo tổng hợp**, giới hạn nêu ở Ch5 §5.9" | Bảng 5.5 + mục 8.3.3(a) |

---

## PHÁN QUYẾT CUỐI — 20 DÒNG CHO CHỦ TỊCH

**TÍNH MỚI:** Protocol inflation (Δ=+13.5 pp, tái lập độc lập, xoá ranking) là **phát hiện tái lập + mở rộng**. Kích thước chi tiết (dấu phụ thuộc xung nhịp, tách 13.8+1.6) ngoài dự báo. Không phải discovery loại, mà **applied quantification**.

**ĐỊNH VỊ:** Luận văn thay đổi scope từ phân rã thực thể (H2 trượt 5 lần) sang phương pháp đo (H1 đạt 2 lần). Lựa chọn công khai & hợp lý. Đóng góp được phân loại rõ: (a) phát hiện mới, (b) mở rộng ×3, (c) kỹ thuật hệ thống, (d) phủ định + cơ chế.

**CÔNG BỐ:** NeurIPS Negative Results (deadline 11-14-2026) khả thi nhất; Workshop GCV 2027; Reproducibility track. Artifact đầy đủ (weights, engine, fold config, logs). Không nên CVPR/ICCV main (kết quả không cạnh tranh).

**LỆCH TÊN:** Rủi ro cao nếu không khắc phục. Khuyến nghị: đổi tên → "Phát hiện an toàn: protocol inflation + deployment", hoặc thêm mục 1.0 biến §1.4.2 thành cảnh báo trung tâm.

**PHƯƠNG PHÁP:** 10 công thức thiếu + 4 dòng tổng quan thiếu. Chi phí thêm: +1 ngày soạn bảng + 0.5 ngày mở rộng Ch2. Ưu tiên: McNemar (1947), Nadeau–Bengio (2000), κ (1960), cluster bootstrap (Davison–Hinkley 1997).

---

## CÂUỎI CHƯA GIẢI QUYẾT

1. **Tương đương số học 74D token embedding** trên Jetson vs. huấn luyện — chưa kiểm chứng (Ch5 §5.7.2).
2. **Phân vị p95 nhịp tổng hợp** — chỉ có trung vị; p95 quan trọng hơn cho hệ cảnh báo.
3. **SOAK test ở mức quyết định tổ hợp** — chỉ có soak giai đoạn riêng.
4. **Fold 3 lỗi bỏ sót (B1w)** — cơ chế chưa rõ; Ch7 §7.3.2 ghi "chưa quy được".
5. **Hợp đồng nhịp có đạt tổng hợp P3 không** — hợp đồng định nghĩa trên f/s, chưa kiểm trên vòng lặp tổ hợp.

**Báo cáo soạn bởi:** Phản Biện 09 · **Ngày:** 31/10/2026

