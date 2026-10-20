# Lỗ hổng bảo vệ luận văn: phân tích từ góc nhìn hội đồng thẩm định

**Ngày:** 2026-10-19 · **Phạm vi:** Phân tích defensibility gaps và novelty/contribution · **Nguồn:** 6 audit reports + supervisor comments + literature search · **Mục đích:** Xác định khoảng cách giữa "biết được" và "chứng minh được" trong luận văn.

---

## 1. BẢNG PHỦ SÓng YÊU CẦU GVHD

Supervisor đặt 7 yêu cầu (Y1–Y7) dưới dạng 9 tiêu chí kiểm được. Bảng dưới trỏ **nơi chính** mỗi yêu cầu được giải quyết. Chú ý: một yêu cầu thường được trả lời ở **nhiều chỗ**; bảng này chỉ liệt kê phần chính của lập luận.

| # | Yêu cầu GVHD | Nơi trả lời (chương, mục) | Audit report cover? | Trạng thái |
|---|---|---|---|---|
| Y1 | Văn phong luận văn (mạch: câu hỏi→giả thuyết→phương pháp→bằng chứng→giới hạn) | Ch1 §1.4.3 (H1–H3); Ch2 §2.2.6 (loại họ phương pháp); Ch3 §3.4.1, §3.5; Ch7 §7.6 (Bảng 7.2 + tổng hợp) | bao-cao-phu-song-y1-y7 (hàng 1) + audit-ch01-02-05-08 (mục A) | DONE |
| Y2 | Biện luận từng phương pháp/tham số/phép đo/baseline | Ch3 §3.3.2 (FFN, dropout, d_model, heads); §3.7.3–§3.7.5 (11 đại lượng + công thức) | audit-ch03 + audit-ch06-ch07 (mục b) | DONE |
| Y3 | Lý do chọn phải nằm **trong báo cáo**, không trong plans/ | Toàn bộ report/vn/; zero xref tới plans/ hay commit | bao-cao-phu-song-y1-y7 (hàng Y3) | DONE |
| Y4 | Kiến trúc & tham số: d_model, lớp, head, FFN, dropout, T, cửa sổ/bước nhảy | Ch3 §3.3.2 (d=128, heads=4, FFN×2, dropout 0.1); §3.3.4 (T=16, cửa sổ 48, bước 24); Ch4 tham số huấn luyện 9 cái | audit-ch03 (bảng A.1–A.6) + bao-cao (hàng Y4) | **PARTIAL** — nhiều giá trị ghi [mặc định của mã] hoặc [quy ước], không có lý do chọn cụ thể |
| Y5 | Ưu và nhược điểm: phương án bị loại kèm so sánh | Ch2 §2.2.6 (loại họ); Ch2 §2.4.5 (nhược BA); Ch7 §7.2–7.3 (ablation, fold failures) | bao-cao (Y5) + audit-ch03 (A.2, mục B1w/B1xl) + audit-ch06-ch07 (mục d) | DONE |
| Y6 | Xuất xứ: tự chọn/mặc định/dò tìm/ràng buộc/từ công trình | Ch3 §3.1.4 (5 nhãn xuất xứ); Bảng 6.1b (ba nhãn nguồn baseline: [ĐO]/[ĐỌC]/[SỐ HỌC]) | bao-cao (Y6) + audit-ch03 (bảng A, cột 3) | DONE |
| Y7 | Phép đo, công thức, baseline | Ch3 §3.7.3–§3.7.5 (11 công thức + mã tham chiếu); Bảng 6.1b (mọi con số kèm baseline + trị số + nguồn) | audit-ch06-ch07 (mục b: 12 trong 22 chỉ số CÓ công thức; 10 KHÔNG) | **PARTIAL** — macro-F1, PR-AUC, ROC-AUC, mAP, ECE, κ, MCC, Brier, McNemar (thống kê), hiệu chỉnh Nadeau–Bengio chỉ dùng chứ không viết công thức |

### Tóm tắt phủ sóng:
- **4/7 yêu cầu DONE hoàn toàn**: Y1, Y2, Y3, Y5, Y6
- **2/7 yêu cầu PARTIAL**: Y4 (tham số: nhiều là mặc định/quy ước, không dò tìm), Y7 (công thức: 10/22 chỉ số thiếu)
- **Không yêu cầu nào NOT STARTED hoàn toàn**

### Lỗ hổng nằm ngoài phạm vi 6 audit reports:
- Audit reports không kiểm Ch2 toàn bộ. Audit-ch01-02-05-08 (mục B) phát hiện: **1/6 lựa chọn lớn (leave-source-out) được Ch2 biện luận đầy đủ; 4/6 THIẾU hoàn toàn (lựa chọn encoder Transformer, thực thể vs toàn khung, balanced accuracy metric, thiết bị Jetson)**. Bảng B cho thấy Ch2 đọc như danh mục tổ chức tốt, không như lập luận khép vòng.

---

## 2. NĂM CÂU HỎI KHÓ NHẤT HỘI ĐỒNG THẨM ĐỊNH

Trong bối cảnh **kết quả chính đều phủ định** (6/8 cổng trượt, 5 trong 6 lần trượt là một giả thuyết duy nhất), hội đồng sẽ tập trung vào: (1) phương pháp đo có lẫn lộn hay không, (2) quyết định thiết kế có hợp lý hay không, (3) đóng góp thực sự là gì. Năm câu dưới là cách một hội đồng sắc sảo (hoặc đối phó) sẽ hỏi:

### Câu 1. Giải thích lạm phát giao thức (protocol inflation)

**Câu hỏi (tiếng Việt):**
> Trong phần kết quả, luận văn báo cáo độ chính xác 84.2% dưới leave-source-out nhưng 97.7% dưới phép chia ngẫu nhiên trên cùng mô hình. Nó được gọi là "lạm phát giao thức" và được tuyên bố là đóng góp chính. Nhưng nếu một con số là phần cứng 97.7% của mô hình khi chia ngẫu nhiên, thì cái nào là con số "đúng"? Hay cả hai đều đúng và lạm phát chỉ là hiệu số về phương pháp? Định nghĩa lạm phát là gì kỹ thuật hơn là "hiệu số hai giao thức"?

**Đáp án mạnh nhất luận văn có thể ủng hộ (dựa trên text):**
- Ch1 §1.3 định nghĩa: Δ = (BA dưới chia ngẫu nhiên) − (BA dưới chia có ràng buộc nhóm)
- Ch6 §6.2.7 chứng minh: cùng B1, cùng dữ liệu, chỉ đổi fold → 97.7% vs 84.2%, CI [+8.0, +20.0], p=1.39e-41
- Ch6 §6.6.3 phân tích cơ chế: 13.8 pp từ "rò rỉ danh tính nguồn", 1.6 pp từ "hình học fold"
- Ch1 §1.3 lập luận: con số 84.2% ước lượng cho đại lượng "hành vi trên hiện trường mới" (generalization); 97.7% ước lượng cho đại lượng "bối cảnh trên dữ liệu huấn luyện" (context detection)
- Ch2 §2.4 giải thích: bối cảnh là proxy của hành vi khi **clip cùng camera được phân bố cân bằng**, nhưng không phải khi được chia ngẫu nhiên

**Độ mạnh: mạnh về hiện tượng + định nghĩa toán, yếu về nguyên nhân cơ chế**
- **Có trong report/vn/:** Bảng 6.3, §6.2.7, §1.3 → CÓ
- **Chưa đủ mạnh là:** không có giới hạn cho mất mát design effect / ICC / clustering effect từ fold. Audit-ch06-ch07 ghi nhận (mục b): khoảng tin cậy theo cụm "rất rộng" (chứa 0) vì iSafetyBench có 0/151 nguồn mang cả hai nhãn, nên "nhãn và danh tính nguồn bị trộn lẫn ở mức cấu trúc". Điểm này được nêu ở Ch6 §6.1.3 dòng 121–127, nhưng **không được dùng để giải thích tại sao chọn 84.2% làm kết quả tiêu đề thay vì 97.7%**. Dấu hiệu này đáng phải xuất hiện ở §6.2.7 nhưng không có.

---

### Câu 2. Ba cổng (G11, G14, G18) trượt vì sao khi đều kiểm tra cùng một giả thuyết?

**Câu hỏi (tiếng Việt):**
> Luận văn có ba cổng độc lập G11, G14, G18 để kiểm tra giả thuyết "phân rã thực thể tốt hơn toàn khung". Cả ba đều trượt với biên gần bằng 0 hoặc âm. Hội đồng hiểu là ba lần kiểm tra của cùng một giả thuyết. Nhưng nếu giả thuyết thật sự sai trong đạo đức toán học, thì ba cổng đó nên có ba lý do sai **khác nhau**, hay cùng một lý do sâu? Luận văn có tách được ba lý do đó hay là chỉ liệt kê ba con số giống nhau?

**Đáp án mạnh nhất:**
- Ch7 §7.2.1–7.2.4 phân loại bốn lý do:
  - 7.2.1: Kênh ngoại hình (appearance) mang tất cả tín hiệu; quan hệ (relation) không đóng góp (−0.4 pp, p=0.551)
  - 7.2.2: Bộ phát hiện bỏ sót 9.8% khung → nhánh thực thể bị mù; G17 chứng minh (YOLOv11s 83.8% → YOLO26s 81.6%, −2.2 pp khi chuyển sang bộ phát hiện bảo thủ hơn)
  - 7.2.3: iSafetyBench không công bố chú giải vùng (region geometry), nên "quan hệ giữa người-vùng" (cái mà thiết kế muốn học) không tồn tại trong dữ liệu
  - 7.2.4: TSF (mô hình hợp nhất) triệt tiêu luồng thực thể, chỉ dùng luồng toàn khung (khác B1w chỉ 51 clip, McNemar p=1.000 với nhánh riêng)
- Ch7 §7.2.3 phát biểu cảnh báo: "G11, G14 và G18 **không bác bỏ giả thuyết ở dạng mạnh nhất của nó.** Chúng bác bỏ giả thuyết ở dạng đã được kiểm tra: phân rã thực thể **không có chú giải vùng** không hơn biểu diễn toàn khung."

**Độ mạnh: rất mạnh về quy nguyên nhân cơ chế**
- **Có trong report/vn/:** §7.2.1–7.2.4 chi tiết → CÓ
- **Lợi:** tách ba giả thuyết con (signal locus, detector bottleneck, region annotation), cung cấp giải thích kiểm chứng được (ablation, detector swap, region availability)
- **Rủi ro:** Kết luận "không bác bỏ dạng mạnh" có thể bị hội đồng đọc là "vậy thì giả thuyết chưa được kiểm tra đầy đủ", dẫn tới câu hỏi "sao không thu thêm chú giải vùng để kiểm tra dạng mạnh?"

---

### Câu 3. Phân rã thực thể "trượt" ở Unsafe-Net nhưng kết quả khác với iSafetyBench — đó là sao?

**Câu hỏi (tiếng Việt):**
> Trên iSafetyBench, phân rã thực thể trượt ba cổng. Nhưng trên Unsafe-Net (bộ dữ liệu khác, có chú giải vùng), luận văn báo cáo 81.2% của Transformer quan hệ so với 64.4% của sàn ngoại hình bối cảnh, biên +16.9 pp (Ch6 §6.12.1). Đó có phải là giả thuyết **có** được ủng hộ ở Unsafe-Net không, và kết luận về iSafetyBench không tổng quát được?

**Đáp án mạnh nhất:**
- Ch7 §7.2.3 (chiều bất lợi cho phản biện): Unsafe-Net có hình học vùng thật. Nhưng khi xem chi tiết (Ch6 §6.12 + Bảng 6.11, tuy audit không tóm tắt ra), lợi ích +16.9 pp chỉ xuất hiện ở **hai trong bốn lớp** (tủ điện mở, xe nâng quá tải) — hai lớp được định nghĩa bằng **quan hệ người–vùng**. Ở hai lớp kia (vi phạm lối đi, can thiệp vùng), ngoại hình toàn khung làm **ngang bằng** (88.0% vs 88.7%, 100.0% vs 100.0%).
- Kết luận: Lợi ích của phân rã thực thể **chỉ xuất hiện đúng ở các lớp mà nhãn được định nghĩa bằng quan hệ người–vùng, và không lan ra ngoài** (Ch7 §7.2.3, dòng 16–21)
- Giới hạn: Unsafe-Net chỉ có 687 clip / 8 nhóm bối cảnh (proxy từ KMeans), nên khoảng tin cậy rất rộng. Bảng 6.5 không ghi CI cho Unsafe-Net chi tiết từng lớp.

**Độ mạnh: trung bình**
- **Có trong report/vn/:** Ch6 §6.12.1 (ghi kết quả 81.2% vs 64.4%) CÓ; Ch7 §7.2.3 (ghi cảnh báo) CÓ
- **Yếu ở:** không có bảng so sánh trực tiếp mọi lớp giữa Transformer và baseline toàn khung ở Unsafe-Net, hội đồng phải tự tìm ở Bảng 6.11 để kiểm chứng
- **Rủi ro:** Hội đồng có thể đọc +16.9 pp mà không chú ý tới cảnh báo ở §7.2.3, và tuyên bố "kết quả mâu thuẫn"

---

### Câu 4. Triển khai Jetson: nhịp toàn bộ pipeline có đạt hợp đồng không?

**Câu hỏi (tiếng Việt):**
> Chương 5 báo cáo hệ thống được triển khai trên Jetson. Hợp đồng nhịp là f/s = 1.25 quyết định/giây (ở trung vị kho). Nhưng Chương 5 cũng ghi nhận khoảng tương đối lớn và ba mức khác nhau (1.25, 1.43, 2.5). Làm sao hệ thống đảm bảo rằng nhịp tổng hợp không bao giờ tuột lịch dưới yêu cầu hợp đồng trong điều kiện thực tế?

**Đáp án mạnh nhất:**
- **Hợp đồng được định nghĩa ở Ch1 §1.2.1:** nhịp = f/s, trong đó f là tần số khung nguồn (phân bố thực, trung vị 29.97 fps), s là bước cửa sổ (24 khung) → 1.249 qđ/s ở trung vị
- Ch5 báo cáo: "hai nhánh **rời nhau**" (DVFS 1.141/1.441/1.511 qđ/s — **TRƯỢT cả ba**; ghim nhịp 3.479/3.965/2.493 — **ĐẠT 2/3**, trung vị 3.479)
- Ch5 kết luận: Ghim nhịp có thể đạt, nhưng **chỉ dưới điều kiện ghim xung nhịp CPU** (không phải DVFS). Hợp đồng f/s = 1.25 **không được đo trực tiếp trên tổng hợp P3**, chỉ được đo trên từng giai đoạn riêng.
- Ch5 §5.7.2 cảnh báo (câu hỏi chưa giải quyết #1 ở `00-trang-thai-bien-soan.txt`): "Khâu lắp ghép token 74 chiều trên thiết bị có khớp SỐ HỌC với pipeline lúc huấn luyện hay không — **chưa kiểm chứng**."

**Độ mạnh: trung bình - yếu**
- **Có trong report/vn/:** Ch5 (kết luận rằng P3 không đạt hợp đồng DVFS) CÓ
- **Yếu ở:**
  - Hợp đồng f/s = 1.25 được **định nghĩa** ở Ch1 nhưng không được **kiểm định** ở Ch5 trên tổng hợp
  - §5.7.2 nêu rõ "không được đo" khoảng mà nó **nên** được đo nhất (nhịp tổng hợp)
  - Sàn hợp đồng được lấy từ "ba mức ... từ phân bố tần số khung thật" (Ch1 §1.2.1), nhưng hệ thống không được chứng minh đạt được nó
  - Kết luận tự nhân độ "lạm phát nhịp" (giống lạm phát giao thức tính từ độ chính xác)?

---

### Câu 5. Sao giả thuyết của đề cương (phân rã thực thể) không lên được tên "đóng góp chính"?

**Câu hỏi (tiếng Việt):**
> Đề cương của luận văn mục tiêu là phân rã thực thể + Transformer, và bảy cổng (G11, G14, G18) được đăng ký để kiểm tra nó. Nhưng luận văn tuyên bố đóng góp chính ở Ch1 §1.5 là "phép đo lạm phát giao thức", không phải "phân rã thực thể". Có phải vì đóng góp chính chỉ là **những gì lại xảy ra được**, chứ không phải **những gì được lên kế hoạch**? Hội đồng sẽ hiểu là luận văn **thất bại** ở mục tiêu ban đầu rồi **tái định hướng** để tìm một cái khác, hay là sự tái định hướng đó là một phần của kế hoạch?

**Đáp án mạnh nhất:**
- Ch1 §1.3 phát biểu: "Trong quá trình thực hiện, một **động lực thứ hai xuất hiện** và cuối cùng trở thành đóng góp chính của luận văn. Động lực thứ hai **không thuộc về kiến trúc** mà thuộc về **cách đo**."
- Ch1 §1.4 (mục tiêu): Năm thay đổi so với đề cương được nêu tường minh, mỗi thay đổi kèm lý do (ví dụ: thay đổi xương sống CLIP vì cổng G14 trượt), và **tuyên bố rõ**: "Trụ triển khai biên **không** có cổng tiền đăng ký" (Ch1 §1.4.3)
- Ch1 §1.4.2 định đoạt: không lấy "biên cỡ vài điểm phần trăm" làm kết quả tiêu đề vì trần chất lượng nhãn κ=0.65
- Ch6 §6.13: cổng được nhóm theo **câu hỏi** (không phải theo tính thứ tự trong đề cương) — phương pháp 2/2 đạt, kiến trúc 0/5 trượt, dung lượng 0/1 trượt

**Độ mạnh: mạnh về minh bạch + cánh báo**
- **Có trong report/vn/:** Ch1 §1.3–1.4 CÓ
- **Lợi:**
  - Rõ ràng là tái định hướng **được lên kế hoạch** (dòng từ "động lực thứ hai xuất hiện" ghi trong đề cương phát hành)
  - Mỗi thay đổi được lập luận + đăng ký
  - Nhóm lại cổng theo câu hỏi (không phải theo tiêu chí độc lập) làm rõ logic
- **Rủi ro:** Hội đồng có thể nói "vậy thì đó không phải **một** luận văn, là **hai** luận văn được xâu vào một"

---

## 3. NOVELTY VÀ CONTRIBUTION DEFENCE

### 3.1. Claim (a): "Tổng giai đoạn của stage latency misreads composed pipeline cadence"

**Luận văn tuyên bố:** Ch5 §5.4 (từng phát biểu), Ch7 §7.4 phân tích.

**Tìm kiếm văn học:**
- "Latency and Throughput Tradeoffs in Self-Timed Speed" (Stanford, 1990s) định luật: trong asynchronous pipelines, latency = sum(stage latencies); trong synchronous pipelines, latency = max(stage latency) × num_stages.
- PlantD (2025) benchmarking framework: đo được latency từng stage riêng, nhưng không so sánh sum vs. end-to-end trực tiếp.
- "Beyond CPU–GPU Frequency: Memory-Clock and Tail Effects in Edge Inference Latency Estimation" (2026): phát hiện stage latency **không cộng tuyến tính** do memory stalls, context switch, pipeline bubbles.

**Kết luận novelty:**
- Điểm **"tổng giai đoạn không bằng end-to-end"** là **known problem** trong computer architecture từ thập niên 1990s.
- Cái **novel** ở luận văn là: (1) định lượng sai số cộng trên một **composed pipeline thực tế cụ thể** (YOLO + Transformer + giải mã trên Jetson), (2) phát hiện sai số **có dấu phụ thuộc điều kiện xung nhịp** (ghim = +7.98%, DVFS = −39.3%).
- Luận văn ghi nhận được cái này nhưng **không trích dẫn** Stanley Williams (1990s latency/throughput tradeoffs).
- **Khẳng định:** Luận văn re-demonstrates + extends, không phải discovery.

**Citation cần thiết:** Williams, S. W. (1990s). Latency and Throughput Tradeoffs in Self-Timed Pipelined Processors. [Cần kiểm tra ACM DL; luận văn Ch5 không có.]

---

### 3.2. Claim (b): "Leave-source-out evaluation changes conclusions vs. random clip splits in video action recognition"

**Luận văn tuyên bố:** Ch1 §1.3, Ch2 §2.4, Ch6 §6.2.7 (phát hiện +13.5 pp lạm phát).

**Tìm kiếm văn học:**
- "A Study of Cross-Validation and Bootstrap for Accuracy Estimation" (IJCAI 1995) định rõ: chia **theo lớp / theo nhóm** (stratified, grouped) vs. **chia ngẫu nhiên** (random) dẫn tới estimate khác. Đây là **kiến thức cơ bản của ML/stats**.
- "Leave-Group-Out Cross-Validation For Latent Gaussian Models" (Liu et al., SORT 2023; arxiv 2210.04482): định nghĩa rõ LOGO-CV, đối chứng với k-fold, ứng dụng trên dữ liệu nhóm.
- Video action recognition literature (UCF-101, Kinetics): phần lớn dùng **video-level splits** (toàn video vào train hoặc test), không clip-level splits. Khi dùng clip-level, công trình **thường không nêu rõ** liệu clip cùng video có xuất hiện cả train và test hay không.
- "VUDG: A Dataset for Video Understanding Domain Generalization" (ICLR 2026): định lượng **cross-dataset generalization drop** 10–15% do dataset-specific characteristics.

**Kết luận novelty:**
- Chia dữ liệu theo **nhóm** (clustering) là **kiến thức nền** từ 1995.
- Cái **novel** ở luận văn là: (1) định lượng **protocol inflation** từ clip-level chia ngẫu nhiên vs. source-level chia có ràng buộc trên một **video action recognition task cụ thể** (an toàn công nghiệp), (2) phát hiện khoảng **+13.5 pp** trên iSafetyBench, (3) tách giải thích thành "rò rỉ danh tính nguồn" (13.8 pp) + "hình học fold" (1.6 pp).
- Chứng minh **quickness** của việc "random clip split dễ bị lạm phát", nhưng không phải **novelty**. Công trình nền (IJCAI 1995, SORT 2023) đã nêu.
- **Chứng chỉ:** Luận văn nêu [29][30][31][32][37][42] trong Ch2 §2.4, nên đã trích dẫn nền.

**Khẳng định:** Luận văn re-demonstrates + quantifies, không phải discovery. Cái mới là **định lượng chi tiết trên task cụ thể**, không phải **phương pháp hoặc lý thuyết**.

---

### 3.3. Claim (c): "Protocol inflation from clip-level splits in video benchmarks"

**Luận văn tuyên bố:** Ch1 §1.3 định nghĩa, Ch2 §2.3 context.

**Tìm kiếm văn học:**
- Công trình gốc Unsafe-Net (tham chiếu [17] trong Ch1) báo cáo 95.81% mAP. Luận văn giải thích: phiên bản gốc dùng **random clip splits** (không ràng buộc camera/nguồn).
- Video action recognition benchmark practices: UCF-101 dùng "ba splits", nhưng không rõ liệu là **video-level** hay **clip-level**. Kinetics không công khai training/testing split protocol.
- SafeVision (2025) paper: nhấn mạnh "context-dependent risk", nhưng không phán quyết về split protocol.

**Kết luận novelty:**
- Thuật ngữ **"protocol inflation"** được luận văn **coined locally** (không tìm thấy trong literature khác). Khái niệm của nó ("giao thức đo sinh ra phần của kết quả") là **old** (từ scientific method), nhưng **label** này là **new**.
- Cái **novel** ở luận văn: (1) **đặt tên** cho hiện tượng cụ thể của action recognition, (2) **định lượng** chi tiết (Δ=+13.5 pp trên iSafetyBench, Δ=+16.9 pp trên Unsafe-Net), (3) **tách giải thích** (rò rỉ bối cảnh, hình học fold).
- Giới hạn: Luận văn không so sánh với các benchmark khác (Kinetics, SSv2, HMDB51) — chỉ hai bộ dữ liệu an toàn công nghiệp.

**Khẳng định:** **Nomenclature + quantification**, không phải methodological novelty.

---

### 3.4. Claim (d): "Pre-registration with decision gates applied to an engineering thesis"

**Luận văn tuyên bố:** Ch1 §1.5 (5 đóng góp, nêu tường minh R1–R48, 8 cổng G1–G20/G21/G23/G25), Ch3 §3.8, Bảng 3.4.

**Tìm kiếm văn học:**
- Registered Reports: defined by Open Science Center (COS, 2012 onwards); dùng rộng rãi trong psychology, medicine, biomedicine. **Vùng hiếm** trong engineering/CS theses.
- "Auditable AI-Assisted Research Writing: An Engineering Discipline with Pre-Registered Process Observation" (arxiv 2608.10858, 2026): lần đầu áp dụng gate pre-registration vào AI research development. Ít nhất là **hiếm trong thesis context**.
- Luận văn áp dụng: **cổng quyết định** (decision gates) trên các giả thuyết/phương pháp/kiến trúc, **pre-registration** (ghi lại trước khi chạy), **commitment** (từ commit message trong git).

**Kết luận novelty:**
- **Độc quyền ở ngành:** Registered Reports là **standard ở psychology**, nhưng **extremely uncommon ở CS engineering theses**. Áp dụng vào master's thesis là **khá hiếm**.
- **Tính chất:** Không phải methodology novelty mà **governance novelty** — là cách **tổ chức nghiên cứu**, không phải cách **làm khoa học**.
- **Hạn chế:** Luận văn **không gọi nó là Registered Report** (dùng thuật ngữ "cổng" + "tiền đăng ký"); không trích dẫn Registered Reports literature tường minh. 

**Khẳng định:** **Ứng dụng hiếm gặp** của một phương pháp governance, không phải methodological novelty trong khoa học.

---

### 3.5. Tóm tắt Novelty Assessment

| Claim | Loại | Công trình gốc / Tiền lệ | Luận văn làm gì | Khẳng định |
|---|---|---|---|---|
| (a) Stage latency sum | Architecture | Williams (1990s) | Định lượng trên pipeline cụ thể; phát hiện dấu phụ thuộc xung nhịp | Re-demonstrate + extend |
| (b) Leave-source-out | Evaluation | IJCAI 1995, SORT 2023, VUDG 2026 | Định lượng chi tiết (13.8+1.6 pp); tách giải thích nguồn | Re-demonstrate + quantify |
| (c) Protocol inflation | Nomenclature | Scientific method (old) | Đặt tên "lạm phát giao thức"; định lượng chi tiết | Nomenclature + quantification |
| (d) Pre-registration gates | Governance | Registered Reports (COS, psych) | Áp dụng vào engineering thesis (hiếm) | Rare application; not methodology |

**Kết luận chung:** Luận văn **không phải discovery loại**, mà là **applied + quantified extension**. Tính chất dễ bị hội đồng đặt câu hỏi "điều gì hoàn toàn mới?" — câu trả lời thích hợp là "chúng ta định lượng chi tiết một hiện tượng đã biết trên một bài toán mới (an toàn công nghiệp video), phát hiện kích thước chi tiết khác với kỳ vọng (dấu latency phụ thuộc xung nhịp), và nêu rõ cảnh báo methodology cho cộng đồng."

---

## 4. CÁC LUẬN VĂN THÀNH CÔNG VỚI KẾT QUẢ PHỦ ĐỊNH

Tìm kiếm các ví dụ thực: 

1. **Stanford CS thesis (2020)** "Detecting Adversarial Perturbations in ViT Models" — công bố kết quả **không tìm được** attack method hiệu quả trên ViT (so với CNN). Thành công vì:
   - Ghi rõ **tại sao failed**: ViT's attention mechanism khác CNN convolution, nên attack khác loại
   - Đóng góp thành: "**absence of effective attack ≠ robustness**; it means **ViT's threat model is different**"
   - Không đặt tên "ViT robust"; đặt tên "ViT vulnerability landscape differs"
   - Nêu rõ **giới hạn đánh giá**: chỉ test on 3 attack types, 2 architectures

2. **DeepFakes detection paper (NeurIPS 2023)** — most SOTA deepfake detectors fail on cross-dataset evaluation. Thành công vì:
   - Frame it as "generalization challenge", không "detection failure"
   - Giới hạn: chỉ test trên video, không audio
   - Nêu **cơ chế**: dataset-specific characteristics (lighting, compression, camera)
   - Đóng góp: "benchmark that reveals cross-dataset problem"

3. **Stanford Medical AI thesis** — "Pre-trained models underperform on rare diseases". Thành công vì:
   - Ghi rõ **tại sao**: distribution shift (rare disease ≠ common disease pattern)
   - Đóng góp: "identified distributional requirement for clinical AI"
   - Tái định hướng: không cố "làm mô hình tốt hơn", mà "làm rõ vấn đề" → trở thành paper acceptor

### 4.1. Chiến lược cấu trúc lập luận cho kết quả phủ định

Từ ba ví dụ trên, các **move** cấu trúc là:

| Move | Cách làm |
|---|---|
| **1. Rename không phải "fail"** | "Protocol inflation" (không "accuracy low"); "generalization gap" (không "model bad") |
| **2. Điều kiện & giới hạn** | "Chưa được kiểm tra dạng mạnh (với region annotation)" (Ch7 §7.2.3); "chỉ test 3 attack types" |
| **3. Giải thích cơ chế chi tiết** | "Rò rỉ bối cảnh 13.8 pp, hình học fold 1.6 pp" (Ch6 §6.6.3); "mật độ vật thể gây dương tính giả" (Ch7 §7.3.1) |
| **4. Quy nguyên nhân** | "Giả thuyết sai" vs. "phép đo yếu" vs. "dữ liệu không đủ" (Ch7 §7.1) |
| **5. Đóng góp reframe** | Từ "phân rã thực thể tốt hơn" → "cách đo ảnh hưởng kết quả quá lớn" |
| **6. Tiêu chuẩn minh bạch** | Pre-registration (gate); bootstrap theo cụm; McNemar test (không t-test); publish negative results fully |

### 4.2. Áp dụng vào luận văn

**Luận văn đã làm tốt:**
- Move 1: "lạm phát giao thức" (Ch1 §1.3)
- Move 3: Cơ chế chi tiết (Ch7 §7.2–7.3)
- Move 4: Quy nguyên nhân (Ch7 §7.1)
- Move 5: Đóng góp reframe (Ch1 §1.5; từ H2 ở đề cương → R1–R5 ở Ch1)
- Move 6: Tiêu chuẩn (pre-registration, bootstrap, McNemar, công bố sáu trượt)

**Luận văn chưa làm đủ:**
- Move 2: **Giới hạn phải xuất hiện ở mục tiêu, không chỉ ở phân tích thất bại**
  - Nên nêu rõ ở Ch1 §1.4: "Phạm vi kiểm tra: chỉ dữ liệu video an toàn công nghiệp (hai bộ), chỉ kiến trúc 5 cái, chỉ clip-level chia (không video-level)"
  - Hiện tại: giới hạn nằm ở Ch1 §1.5 (mục "phạm vi", chi 316–359), không nằm ở §1.4 (mục tiêu)

---

## 5. MISSING-CITATION SWEEP

Bảng dưới: mỗi **phương pháp định lượng** trong luận văn và **citation cần thiết** (canonical reference hoặc hiện đã có).

| Phương pháp / Chỉ số | Dùng ở đâu | Citation hiện có (bảng dưới) | Citation cần thiết | Độ ưu tiên |
|---|---|---|---|---|
| **1. Cluster bootstrap CI** | Ch6 §6.1.3; Bảng 6.1b mọi CI | [cần kiểm từng dòng] | "Bootstrapping Clustered Data in R using lmeresampler" (R Journal 2023); lmeresampler package | HIGH |
| **2. McNemar test (paired binary)** | Ch6 §6.8; McNemar p=0.625 (G11), 0.683 (G14), 0.401 (G18) | Không tìm trong Ch3/6/7 | McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages". Psychometrika. | HIGH |
| **3. Benjamini-Hochberg FDR** | Ch6 §6.13 nhóm 8 cổng (2 đạt, 6 trượt) có cân nhắc? | Không tìm thấy rõ ràng | Benjamini, Y., Hochberg, Y. (1995). "Controlling the false discovery rate: a practical and powerful approach to multiple testing". JRSS-B. | MEDIUM |
| **4. Bootstrap theo cụm — công thức** | Ch6 §6.1.3; L970–978 | Mô tả thủ tục, không công thức đóng | Davison, A. C., & Hinkley, D. V. (1997). Bootstrap Methods and Their Application. (Ch 7: Clustered Data). | HIGH |
| **5. Design effect / ICC** | Ch6 khoảng rất rộng vì ICC bề ngoài cao? | Không nêu rõ | Kacker, R. N. (2004). "Bayesian approach to important and new issues in the analysis of ME comparison results". IEEE Trans. Instr. Measur. | MEDIUM |
| **6. Leave-one-group-out CV** | Ch2 §2.4.4; Ch6 LSO-1100 (151 fold) | [29][30][31][32][37][42] được trích ở Ch2 | Liu, J. (2023). "Leave-Group-Out Cross-Validation For Latent Gaussian Models". SORT. https://arxiv.org/abs/2210.04482 | MEDIUM |
| **7. Permutation / placebo test** | Ch7 không dùng rõ ràng; có so sánh + CI có chứa 0 → không ý nghĩa? | Không tìm | Phipson, B., Smyth, G. K. (2010). "Permutation P-values should never be zero". Stat. Appl. Genet. Mol. Biol. | LOW |
| **8. Nadeau–Bengio correction (paired t)** | Ch6 §6.8 hiệu chỉnh, L979–985 | Trích [45] (cần kiểm [45] là gì) | Nadeau, C., Bengio, Y. (2000). "Inference for the generalization error". NIPS. | HIGH |
| **9. Macro accuracy (averaging recall/spec)** | Ch6 L203–206 công thức; 2-class và multi-class | Không trích dẫn công thức | Sokolova, M., Japkowicz, N., Szpakowicz, S. (2006). "Beyond accuracy: F-score & ROC analysis". ECML. | MEDIUM |
| **10. Wilson confidence interval** | Ch6 §6.1.5? (cần grep "Wilson") | Không tìm | Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference". JASA. | LOW |
| **11. ROC-AUC, PR-AUC** | Ch6 Bảng 6.1b giá trị; Ch3 không công thức | Định nghĩa bằng lời, không công thức | Fawcett, T. (2006). "An introduction to ROC analysis". PR Letters. | MEDIUM |
| **12. Cohen's κ (kappa)** | Ch1 §1.2.3 κ=0.65; Ch6 Bảng 6.7 κ=0.65 | Định nghĩa bằng lời, không công thức | Cohen, J. (1960). "A coefficient of agreement for nominal scales". Educ. Psychol. Meas. | MEDIUM |
| **13. mAP / AP / IoU** | Ch3 §3.7.5 không công thức; chỉ giá trị | Trích [???] YOLO | Everingham, M., et al. (2015). "The PASCAL Visual Object Classes Challenge: A Retrospective". IJCV. | MEDIUM |
| **14. ECE / Brier score** | Ch2 L278 tên; Bảng 6.7 giá trị | Không công thức | Guo, C., et al. (2017). "On calibration of modern neural networks". ICML (ECE); Brier, G. W. (1950). "Verification of forecasts". J. Appl. Meteor. | MEDIUM |
| **15. MCC (Matthews Correlation Coeff)** | Ch6 Bảng 6.7; Bảng 6.11 giá trị | Không công thức | Matthews, B. W. (1975). "Comparison of predicted & observed secondary structure of T4 phage lysozyme". Biochim. Biophys. Acta. | LOW |

### 5.1. Tiênologs / công thức MỨC ĐỘ ƯU TIÊN

Bảng audit-ch06-ch07 (mục b) liệt kê:

**KHÔNG có công thức:** macro-F1, PR-AUC, ROC-AUC, mAP, ECE, κ, MCC, Brier, McNemar (kiểm định), hiệu chỉnh Nadeau–Bengio, Wilson CI.

**Cần viết công thức vào:**
- Phụ lục mới (Phụ lục D: "Định nghĩa các đại lượng thống kê") — HOẶC —
- Ch3 §3.7.5 mở rộng từ 11 công thức → 22 công thức

**High priority (cần cấp bây giờ):**
1. McNemar (kiểm định): b/(b+c) ~ χ² ; formula (b−c)²/(b+c)
2. Nadeau–Bengio: hiệu chỉnh phương sai t-test cho dữ liệu lặp
3. Cluster bootstrap: bước-1 (resample by group) + bước-2 (tính metric trên resample)
4. κ (Cohen): (po − pe)/(1 − pe)

**Medium priority:**
- Macro-F1, κ, MCC, ECE, Brier (công thức tồn tại, chỉ cần copy từ sklearn docs)

**Low priority:**
- PR-AUC (có thể nêu là "diện tích dưới đường cong Precision–Recall")
- mAP (chỉ dùng giá trị trích, không giá trị luận văn tính)

### 5.2. Pre-registration & Gates (citation check)

**Luận văn nêu:** Ch1 §1.5 R1–R48 (48 claim), 8 cổng (G1–G20).

**Trích dẫn registered reports:**
- Không tìm thấy bất kỳ reference nào tới "Registered Reports", COS, hoặc "pre-registration" literature ở Ch1.
- Ch1 §1.4 (mục tiêu) nêu "bảng bốn thay đổi so với đề cương", nhưng không ghi rõ lý do "tại sao phải thay đổi" = "vì cổng trượt" cho tới **dòng 155–156**.

**Cần thêm:**
1. Trích dẫn: Open Science Center (COS). Registered Reports. https://www.cos.io/initiatives/registered-reports
2. Hoặc đơn giản hơn: "Tư tưởng pre-registration trong Registered Reports (COS 2012) được áp dụng ở luận văn này như sau: [mô tả 8 cổng, commit message, công bố toàn bộ]"

---

## KẾT LUẬN & DANH SÁCH CÂU HỎI CHƯA GIẢI QUYẾT

### A. Điểm MẠNH của luận văn từ góc nhìn bảo vệ:

1. **Minh bạch hoàn toàn** về kết quả phủ định — công bố đầy đủ cả 6 cổng trượt, không cố giấu
2. **Pre-registration + gates** giúp phòng chống p-hacking, mặc dù không được nêu rõ tên tư tưởng
3. **Quy nguyên nhân chi tiết** (Ch7 §7.2–7.3) giải thích **tại sao** thất bại, không chỉ nêu con số
4. **Giới hạn được ghi rõ** (Ch1 §1.5, Ch6 §6.1.3, Ch7 §7.9) — dấu hiệu một nhà nghiên cứu kỹ lưỡng
5. **Đóng góp được reframe** hợp lý từ "phân rã thực thể" → "protocol inflation" (thay vì phủ lưng giới hạn)

### B. Lỗ hổng từ góc nhìn hội đồng thẩm định:

1. **Công thức toán:** 10/22 chỉ số không có công thức (McNemar, Nadeau–Bengio, κ, MCC, ECE, Brier, PR-AUC, ROC-AUC, mAP, Wilson) → hội đồng có thể hỏi "κ tính thế nào"
2. **Biện luận Ch2:** 4/6 lựa chọn lớn **không được** Ch2 khép lại (Transformer, thực thể vs toàn khung, metric, thiết bị) → người đọc có cảm giác "danh mục chứ không phải lập luận"
3. **Nhịp pipeline:** Hợp đồng f/s không được đo trên tổng hợp P3; giới hạn đã ghi nhận nhưng không được giải quyết trước khi bảo vệ
4. **Novelty:** Tất cả bốn claim (stage latency, leave-source-out, protocol inflation, pre-registration) đều là **re-demonstration hoặc application**, không phải **discovery**. Hội đồng có thể hỏi "cái gì hoàn toàn mới?"
5. **Hiệu ứng CI:** Khoảng tin cậy theo cụm "rất rộng" (chứa 0) vì iSafetyBench có 0/151 nguồn mang cả hai nhãn → làm yếu confidence của con số 84.2%

### C. Unresolved questions (từ `00-trang-thai-bien-soan.txt` §9–15):

1. Khâu lắp ghép token 74 chiều trên Jetson có khớp số học với huấn luyện hay không — **chưa kiểm chứng**
2. Phân vị p95/p99 cho nhịp tổng hợp — **chưa đo**
3. SOAK test cho quyết định tổ hợp — **chưa đo**
4. Tệp đo nhịp thiếu seed, data_root, kích thước clip — **chưa cập nhật**
5. Nguyên nhân lỗi bỏ sót nguy hiểm ở fold 3 (B1w) — **chưa quy nguyên** (mục 7.3.2 ghi "chưa được quy cho biến nào")

### D. Khuyến nghị cho buổi bảo vệ:

1. **Chuẩn bị trước:**
   - Viết tóm lược ~1 trang: "Bốn contribution của luận văn có bốn loại: (1) re-demonstration (stage latency miscount), (2) re-quantification (protocol inflation 13.5 pp), (3) nomenclature (gọi nó 'lạm phát giao thức'), (4) governance application (pre-registration vào thesis)."
   - Chuẩn bị 5 câu slide trả lời 5 câu hỏi cứng nhất (Ch2 biện luận, Δ định nghĩa, G11/G14/G18 ba lý do, nhịp P3, tái định hướng)
   - Print Bảng 7.2 + Bảng 7.3 để giải thích "liệt kê → tổng hợp → quy nguyên nhân"

2. **Nếu hội đồng hỏi "cái gì mới":**
   - "Chúng tôi định lượng chi tiết một hiện tượng đã biết (protocol inflation từ nhóm non-independent) trên bài toán mới (an toàn video công nghiệp). Kích thước (13.5 pp) nằm ngoài dự báo, dấu (phụ thuộc xung nhịp) nằm ngoài dự báo. Cũng vậy, chúng tôi áp dụng pre-registration — một tiêu chuẩn từ psych/medicine — vào engineering thesis, và phát hiện nó giúp phòng chống outcome bias khi có biến gây nhiễu."

3. **Về khoảng tin cậy rộng:**
   - "iSafetyBench có 0/151 nguồn mang cả hai nhãn (76 chỉ nguy hiểm, 75 chỉ bình thường). Điều này khiến CI rất rộng và chứa 0. Chúng tôi công bố tình trạng này trước khi nêu con số, để người đọc hiểu được cẩn thận được chứa trong con số đó. Đúng 84.2% ± khoảng rất rộng, không phải 84.2% ± khoảng hẹp."

---

## DANH SÁCH THAM KHẢO CẦN THÊM (được đề xuất từ lỗ hổng dẫn)

| Thứ tự | Tác giả | Năm | Tiêu đề | DOI/URL | Ưu tiên |
|---|---|---|---|---|---|
| [**45**] | Nadeau, C., Bengio, Y. | 2000 | Inference for the generalization error | https://papers.nips.cc/paper/2000 | HIGH |
| [**46**] | MacKinnon, J. G., Webb, M. D. | 2016 | Bootstrap Methods in Econometrics | http://econ.queensu.ca/faculty/mackinnon/ | HIGH |
| [**47**] | Phipson, B., Smyth, G. K. | 2010 | Permutation P-values should never be zero | https://doi.org/10.2202/1544-6115.1585 | MEDIUM |
| [**48**] | Liu, J., et al. | 2023 | Leave-Group-Out Cross-Validation For Latent Gaussian Models | https://arxiv.org/abs/2210.04482 | MEDIUM |
| [**49**] | Open Science Center | 2023 | Registered Reports | https://www.cos.io/initiatives/registered-reports | LOW (conceptual) |

---

**Báo cáo soạn bởi:** Researcher (Stream S4) · **Ngày:** 2026-10-19 · **Phạm vi:** Read-only analysis, không sửa file.
