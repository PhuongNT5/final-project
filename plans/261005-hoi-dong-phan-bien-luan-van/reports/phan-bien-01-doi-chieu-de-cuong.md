# PHẢN BIỆN 1 — ĐỐI CHIẾU ĐỀ CƯƠNG ĐÃ ĐĂNG KÝ vs LUẬN VĂN ĐÃ VIẾT

Nguồn: `PROPOSAL.txt` (187 dòng) vs `report/vn/*.txt` (11 tệp nộp + 2 tệp hồ sơ).
Quy ước: mọi trích dẫn kèm số dòng của tệp gốc. "KHÔNG TÌM THẤY" = đã grep toàn bộ `report/vn/*.txt`, không có kết quả.

## 0. Tóm tắt 1 câu

Luận văn tự nhận đã đổi hướng đề tài một cách có chủ ý và minh bạch ở hầu hết các chỗ — nhưng (a) chỗ đổi hướng nặng nhất (xương sống YOLO → CLIP ViT-B/32 cho hệ thống triển khai đầu đề) KHÔNG được liệt trong bảng tự khai "thay đổi so với đề cương" của Ch1; (b) không có bằng chứng thủ tục (GVHD/khoa duyệt đổi đề cương); (c) thiếu Confusion Matrix bắt buộc; (d) kiến trúc đa luồng camera→edge→server→dashboard chỉ giao camera→edge.

---

## 1. BẢNG ĐỐI CHIẾU

### 1.1. Ba mục tiêu đề cương

| # | Đề cương (PROPOSAL.txt) | Luận văn thực tế | Dòng bằng chứng | Phán quyết |
|---|---|---|---|---|
| MT1 | "Kết hợp CNN-based (YOLO)... với Transformer-based (RT-DETR)... xây dựng kiến trúc YOLO–Transformer" (dòng 42-46) | Hệ thống **triển khai đầu đề (B1)** = tháp **CLIP ViT-B/32 đóng băng** (87,849,216 tham số) + đầu Transformer thời gian (201,090 tham số). **Không có YOLO trong B1.** YOLO chỉ xuất hiện ở P3 (pipeline phụ, chỉ đo chi phí trên Unsafe-Net, không phải headline) và P1 (giai đoạn trước, không triển khai). RT-DETR bị loại hẳn, thay bằng Transformer encoder thường. | `chuong-05-trien-khai-bien.txt:196` "B1 là đồ thị được triển khai... sinh ra con số độ chính xác tiêu đề"; `chuong-03-phuong-phap.txt:134-145` (CLIP tower); `chuong-03-phuong-phap.txt:72-81` (loại RT-DETR) | **ĐÃ ĐỔI HƯỚNG, CÔNG BỐ KHÔNG ĐẦY ĐỦ.** Lý do kỹ thuật tồn tại và hợp lý (Ch3.3.1: CLIP cho biểu diễn ngữ nghĩa mạnh không cần video có nhãn) nhưng **bảng tự khai 3 thay đổi ở Ch1 mục 1.4.2 (dòng 162-188) KHÔNG liệt chỗ này** — chỉ liệt RT-DETR→MLP, mục tiêu accuracy, và 8 lớp→nhị phân. Người đọc phải tự ghép Ch3.3.1 + Ch5.1 + `00-trang-thai-bien-soan.txt:122-128` mới nhận ra xương sống đổi hẳn. |
| MT2 | "Huấn luyện, tinh chỉnh, đánh giá YOLO–Transformer trên Unsafe-Net, 8 lớp (4 an toàn–4 nguy hiểm)" (dòng 47-49) | Bài toán tiêu đề đổi thành **nhị phân** nguy hiểm/bình thường. "Trên đĩa không tồn tại lần chạy 8 lớp nào" — một bản thảo trước nêu "khoảng 78%" cho 8 lớp, con số đó **đã bị rút** vì không có log. Làn 4 lớp cũ giữ lại như "làn lịch sử", không phải làn tiêu đề. | `chuong-01-mo-dau.txt:180-188` | **ĐÃ ĐỔI HƯỚNG, CÓ CÔNG BỐ** (nằm rõ trong bảng tự khai, có lý do). |
| MT3 | "Tối ưu hoá (TensorRT, INT8) để xử lý đa luồng video thời gian thực trên biên, độ trễ thấp, độ chính xác cao" (dòng 50-54) | Đạt phần "đo trên chính thiết bị" (TensorRT FP16, Jetson Orin Nano Super, 167.291 ms/quyết định = 5.978 qđ/giây). Đạt phần kỹ thuật tối ưu. **Không đạt** phần "đa luồng" cho hệ thống triển khai thật (B1/P3): "Đa luồng cho B1 và P3. Kết quả 4 luồng là của P1 và không chuyển được" — đa luồng chỉ đo trên pipeline cũ (P1, 4 lớp, YOLOv11s), không phải hệ thống đầu đề. INT8 không thấy số liệu triển khai cuối, chỉ FP16. | `chuong-05-trien-khai-bien.txt:1174` | **ĐẠT MỘT PHẦN.** Phần đo trên thiết bị rất kỹ và trung thực; phần "đa luồng" bị thu hẹp về pipeline đã bỏ. |

### 1.2. Ba nội dung nghiên cứu

| Nội dung đề cương | Kết quả dự kiến đề cương | Thực tế | Phán quyết |
|---|---|---|---|
| ND1: tổng quan YOLO/RT-DETR + Transformer/TimeSformer/SlowFast; tổng hợp pruning/quantization/TensorRT | Tài liệu tổng hợp + bộ dữ liệu baseline | Có — Ch2 tổng quan đầy đủ, có thảo luận RT-DETR và lý do loại (dòng 41-44 `chuong-02-tong-quan.txt`), minh bạch. | **ĐẠT.** |
| ND2: mô hình 3 thành phần (YOLO trích đặc trưng không gian + Transformer chuỗi + fusion MLP); data loader sparse sampling; accuracy test Unsafe-Net > 95.8% | Mô hình kết hợp CNN-Transformer 3 khối; accuracy > 95.8% | Cấu trúc 3 khối **tồn tại về bản chất** (trích không gian + chuỗi thời gian + phân loại) nhưng khối 1 là **CLIP, không YOLO** cho headline. Accuracy > 95.8%: **không theo đuổi nữa**, kết luận đầu đề chuyển thành đại lượng lạm phát giao thức (+13.0 đến +21.8 pp). Trên làn 4 lớp cũ (nơi có dùng YOLOv11s/YOLO26), kết quả tốt nhất dưới giao thức mù là 85.4% macro — thấp hơn 95.81% của bài báo gốc, và khoảng tin cậy McNemar cho thấy thiếu hụt này **có ý nghĩa thống kê** so với baseline full-bbox (mọi khoảng <0.05 ở cấu hình tiêu đề), **không phân biệt được** so với baseline no-bbox (85.78%). | **ĐÃ ĐỔI HƯỚNG, CÓ CÔNG BỐ** cho phần accuracy (dòng 171-179 `chuong-01-mo-dau.txt`); **KHÔNG ĐẠT** theo nghĩa đen của cam kết số (>95.8%), khoảng cách **~10.4 pp** (85.4% vs 95.81%) nếu so cấu hình tiêu đề, hoặc ngang bằng nếu so đường sàn không-bbox. |
| ND3: hệ thống đa luồng camera→edge node→server lưu trữ→dashboard real-time; bảng so sánh toàn diện vs baseline; triển khai hệ thống đa luồng chứng minh real-time + bảo mật dữ liệu | Kiến trúc 4 khối; bảng so sánh; hệ thống đa luồng hoạt động | **"dashboard"**: 0 kết quả grep trên toàn bộ 11 tệp nộp — KHÔNG TÌM THẤY bất kỳ đề cập, thiết kế hay số đo nào. **"server lưu trữ"**: KHÔNG TÌM THẤY. Chỉ có camera (giải mã luồng) → edge node (tháp + đầu) được đo thật; "khung hình không bao giờ rời thiết bị" — nghĩa là theo thiết kế **không có** luồng dữ liệu ra server/dashboard. Bảng so sánh toàn diện: **có**, và công bằng (Ch6.12.5, dùng cùng giao thức, báo cáo cả kết quả bất lợi). Đa luồng: chỉ đo trên P1 (pipeline bỏ), không chuyển cho B1/P3. | **ĐẠT MỘT PHẦN.** Bảng so sánh: đạt, và đáng khen vì trung thực. Kiến trúc hệ thống 4 khối: **không đạt** — chỉ giao 2/4 khối (camera, edge); server lưu trữ và dashboard **không tồn tại trong luận văn nộp**, kể cả ở dạng thiết kế/mock. |

### 1.3. Tiêu chí đánh giá bắt buộc của đề cương

| Tiêu chí đề cương (dòng 169-180) | Thực tế | Phán quyết |
|---|---|---|
| Precision trên 125 clip | Thực tế dùng 64 clip (làn 4 lớp) hoặc 687/1,100 clip theo leave-context-out/leave-source-out (làn nhị phân) — không có tập test cố định 125 clip nào được dùng cho kết quả tiêu đề. | **KHÔNG KHỚP THIẾT KẾ ĐÁNH GIÁ** của đề cương, nhưng có lý do phương pháp luận (giao thức ràng buộc nguồn thay cho test cố định). |
| **Confusion Matrix — "Bắt buộc phải có"** (dòng 173-174) | KHÔNG TÌM THẤY ma trận nhầm lẫn mô hình đầy đủ cho kết quả đầu đề (binary/B1 hoặc P3). `chuong-04...txt:142` "Bảng 4.2. Ma trận đồng thuận hai người gán nhãn" là ma trận **đồng thuận annotator**, không phải ma trận dự đoán-vs-thật. Duy nhất một **mảnh** ma trận 4 lớp (`[0,0,0,56]`, một hàng) xuất hiện trong Phụ lục B (`phu-luc-b-cau-hinh-fold.txt:345`) khi phân tích tính suy biến của một fold — không phải bảng/hình chính thức cho kết quả chính. | **KHÔNG ĐẠT.** Đây là yêu cầu "bắt buộc" duy nhất ghi rõ ràng bằng chữ trong đề cương và nó vắng mặt ở dạng đầy đủ. |
| FPS | Đo kỹ, minh bạch, có cả lịch sử sửa sai (289.672→167.291 ms/quyết định sau khi vá lỗi tiền xử lý). | **ĐẠT**, chất lượng cao. |
| Latency (ms) | Đo đầy đủ (p50/p95), trên vòng lặp tổ hợp thật, không bỏ giai đoạn. | **ĐẠT**, chất lượng cao. |

---

## 2. CÁC CÂU HỎI BẮT BUỘC — TRẢ LỜI DỨT KHOÁT

**(a) Đồ thị thực sự triển khai = YOLO hay CLIP?**
→ **CLIP ViT-B/32 đóng băng** (B1, `chuong-05-trien-khai-bien.txt:196`: "B1 là đồ thị được triển khai... sinh ra con số độ chính xác tiêu đề"). YOLO (YOLO26) chỉ ở P3 — đo chi phí trên Unsafe-Net, không phải headline, và P3 "không đạt hợp đồng nhịp dưới điều kiện xung nhịp mặc định" (dòng 206-207). Đây là **lệch đề tài ở mức tên đề tài** ("...dựa trên YOLO và Transformer") vì hệ thống đầu đề không chứa YOLO. Giải trình kỹ thuật tồn tại (Ch3.3.1, dòng 145-148) và hợp lý về mặt khoa học, nhưng **không được đưa vào bảng tự khai thay đổi ở Ch1.4.2** — đây là một lỗ hổng công bố, không phải lỗ hổng khoa học.

**(b) Accuracy > 95.8% — đạt bao nhiêu, giao thức nào, khoảng cách, giải trình?**
→ Không đạt, và luận văn không theo đuổi nữa (từ bỏ có chủ ý, công bố ở `chuong-01-mo-dau.txt:171-179`). Số cụ thể có thể so sánh: làn 4 lớp cũ, cấu hình tiêu đề (54/64 = 84.4% top-1 / 88.4% macro) hoặc con số trung thực nhất dưới dừng sớm (53/64 = 85.4% macro) — khoảng cách tới 95.81% là **~7.4 đến 11.4 điểm phần trăm** tuỳ cách đọc, và `chuong-06-danh-gia.txt:1150-1162` chứng minh khoảng cách này **có ý nghĩa thống kê** ở mọi cách chia khả thi khi so với baseline full-bbox 62/64, nhưng **không phân biệt được** khi so với baseline no-bbox 57/64 (85.78%). Giải trình: rất kỹ, có trích dẫn, minh bạch — đây là điểm mạnh nhất của luận văn.

**(c) 8 lớp hay thu hẹp phạm vi?**
→ Thu hẹp thành **nhị phân** (nguy hiểm/bình thường) làm bài toán tiêu đề. Làn 4 lớp giữ như lịch sử. 8 lớp: **không có kết quả nào trên đĩa**, một con số "~78%" từng xuất hiện ở bản thảo trước đã bị rút vì vô căn cứ (`chuong-01-mo-dau.txt:186-188`). Đánh giá tiêu đề chạy trên iSafetyBench (1,100 clip, 151 nguồn) — **một bộ dữ liệu hoàn toàn không có trong đề cương** (đề cương chỉ đăng ký Unsafe-Net) — đây là một bổ sung phạm vi không đăng ký, dù theo chiều mở rộng/kiểm chứng chứ không che giấu.

**(d) Confusion Matrix có không?**
→ **Không**, ở dạng đầy đủ cho kết quả chính. Xem bảng 1.3 trên.

**(e) Kiến trúc đa luồng camera→edge→server→dashboard giao được mấy phần?**
→ 2/4 (camera + edge node), đo thật. Server lưu trữ và dashboard: **không tồn tại trong văn bản nộp, không một dòng nào**. Multi-stream (4 camera) chỉ có bằng chứng cho pipeline P1 đã bị thay thế, và luận văn tự ghi rõ "không chuyển được" (`chuong-05-trien-khai-bien.txt:1174`) sang hệ thống đầu đề.

**(f) Bảng so sánh baseline — có, công bằng?**
→ Có (`chuong-06-danh-gia.txt:1140-1186`), và công bằng về phương pháp: dùng McNemar trên mọi cách chia khả thi, báo cáo cả kết quả bất lợi (thiếu hụt có ý nghĩa thống kê so với full-bbox), không chỉ chọn baseline dễ thắng. Đây là điểm mạnh.

**(g) RT-DETR — dùng gì, có giải trình bỏ không?**
→ Không dùng. Thay bằng Transformer encoder thường (1-2 lớp). Giải trình kỹ thuật rõ ràng và lặp lại 3 nơi (Ch1 dòng 168-173, Ch2 dòng 41-44, Ch3 dòng 72-81): RT-DETR là bộ phát hiện đầu-cuối, không phải bộ mã hoá chuỗi thời gian — đề cương đã nhầm hai khái niệm.

**(h) Môi trường thực nghiệm MacBook M4 Pro — rủi ro gì?**
→ Văn bản nộp (`chuong-01-mo-dau.txt:278`) ghi "CPU Apple M3" cho khâu xuất ONNX/lượng tử hoá, khác đề cương (MacBook Pro M4). Nghiêm trọng hơn: `00-trang-thai-bien-soan.txt:220` "Khiếm khuyết nhãn phần cứng M3/M4 ở 24 chỗ: để tác giả quyết, **chưa sửa**" — tính đến 04/10/2026, NHÃN PHẦN CỨNG CHƯA NHẤT QUÁN tại 24 vị trí trong các tệp sẽ nộp. Rủi ro khi hội đồng đọc: thấy mâu thuẫn M3/M4 giữa các bảng → nghi ngờ độ tin cậy của toàn bộ việc ghi chép nhật ký đo, dù luận văn khẳng định "mọi giá trị độ trễ/FPS/độ chính xác không bị ảnh hưởng — chỉ quy thuộc phần cứng thay đổi" (`chuong-05-trien-khai-bien.txt:952`).

**(i) Đổi hướng đề tài sang "lạm phát giao thức đánh giá" — có hợp lệ theo chuẩn LVThS VN?**
→ Về khoa học: đổi hướng có cơ sở, minh bạch, tái lập trên 2 kho dữ liệu độc lập, có khoảng tin cậy/p-value cho mọi phát biểu — đây là chất lượng nghiên cứu cao. Về **thủ tục hành chính**: KHÔNG TÌM THẤY bất kỳ bằng chứng nào trong toàn bộ repo (đã grep "GVHD", "giảng viên hướng dẫn", "phê duyệt đề cương", "xin phép đổi", "thay đổi đề cương", "xác nhận hướng dẫn" trên `report/vn/*.txt`, `PROGRESS.md`, `README.md`) về việc đề cương đã được điều chỉnh chính thức, hoặc GVHD/khoa đã ký xác nhận thay đổi phạm vi. Theo quy chế phổ biến ở các trường VN, một luận văn thay đổi **kết luận đầu đề** (từ "đạt accuracy cao trên hệ YOLO-Transformer" sang "phát hiện phương pháp luận về lạm phát giao thức, hệ triển khai là CLIP không phải YOLO") so với đề cương đã nộp **cần** văn bản giải trình thay đổi đề cương + xác nhận của GVHD, nộp trước hoặc cùng hồ sơ bảo vệ. Không có gì trong repo cho thấy bước này đã làm. **Đây là rủi ro thủ tục độc lập với chất lượng khoa học** — hội đồng có quyền yêu cầu xác nhận trước khi chấp nhận bảo vệ.

---

## 3. 8 LỖ HỔNG NGHIÊM TRỌNG NHẤT (xếp theo rủi ro khi bảo vệ)

1. **[CAO NHẤT] Không có bằng chứng thủ tục đổi đề cương/GVHD duyệt** cho một đổi hướng làm thay đổi hẳn kết luận đầu đề và xương sống mô hình. Hội đồng có thể từ chối công nhận phạm vi nộp nếu không có văn bản.
2. **Xương sống YOLO→CLIP cho hệ thống triển khai đầu đề không được liệt trong bảng tự khai 3 thay đổi** ở Ch1.4.2 — đây là thay đổi nặng nhất so với tên đề tài nhưng bị "giấu" trong văn bản phương pháp (Ch3.3.1) thay vì công bố tường minh ngay đầu như 3 thay đổi khác.
3. **Thiếu Confusion Matrix** — yêu cầu "bắt buộc" duy nhất viết rõ chữ trong đề cương, vắng mặt hoàn toàn ở dạng đầy đủ cho kết quả chính.
4. **Kiến trúc đa luồng 4 khối (camera→edge→server→dashboard) chỉ giao 2 khối**; "dashboard" và "server lưu trữ" — 0 bằng chứng tồn tại dù chỉ ở dạng thiết kế.
5. **Khoảng cách accuracy 95.8%**: không đạt, và với cấu hình tiêu đề (54/64 hoặc 53/64), khoảng cách tới baseline full-bbox của bài báo gốc **có ý nghĩa thống kê theo hướng bất lợi** — luận văn tự công bố điều này trung thực nhưng hội đồng có thể hỏi tại sao mục tiêu đăng ký (>95.8%) lại cách xa đến vậy mà không đạt được dù chỉ đường sàn.
6. **Khiếm khuyết nhãn phần cứng M3/M4 ở 24 chỗ CHƯA SỬA** tính đến thời điểm gần đóng gói (04/10/2026) — rủi ro mâu thuẫn hiển thị khi hội đồng đọc các bảng số liệu.
7. **iSafetyBench (1,100 clip) — bộ dữ liệu đầu đề không có trong đề cương đã đăng ký** (đề cương chỉ đăng ký Unsafe-Net). Mở rộng phạm vi không đăng ký trước, dù không gây hại khoa học.
8. **README.md công khai trên GitHub lạc hậu hoàn toàn** so với luận văn tiếng Việt — vẫn mô tả hệ YOLOv11+Transformer, P1 pipeline, 4 lớp, không đề cập lạm phát giao thức, CLIP B1, hay iSafetyBench. Rủi ro nếu hội đồng đối chiếu code/README với luận văn sẽ thấy hai câu chuyện khác nhau.

---

## 4. PHÁN QUYẾT RIÊNG

**ĐẠT CÓ ĐIỀU KIỆN.**

Lý do: Về mặt khoa học, đây là một trong những luận văn tự phê bình nghiêm túc và minh bạch nhất có thể gặp — mọi con số có nguồn, mọi phát biểu rút lại được ghi sổ, phương pháp đối chứng (McNemar, bootstrap theo cụm, cổng tiền đăng ký) đúng chuẩn. Nhưng xét đúng vai trò "đối chiếu đề cương vs luận văn": (i) nội dung thực tế nộp lệch khỏi đề cương ở nhiều điểm cốt lõi (xương sống YOLO→CLIP, mục tiêu accuracy, phạm vi 8 lớp→nhị phân, hệ thống đa luồng 4 khối→2 khối, bộ dữ liệu đầu đề thêm iSafetyBench); (ii) một trong các lệch đó (xương sống) không được tự khai đầy đủ đúng chỗ; (iii) không có bằng chứng thủ tục chính thức cho việc đổi hướng; (iv) thiếu 1 yêu cầu "bắt buộc" rõ chữ (confusion matrix). Những điểm này đủ nghiêm trọng để không xếp "ĐẠT CHUẨN ĐẦU RA" thẳng, nhưng không đủ để xếp "CHƯA ĐẠT" vì chất lượng minh bạch và phương pháp luận cao, và các khoảng trống đều có thể vá trước bảo vệ.

---

## 5. VIỆC PHẢI LÀM TIẾP (ưu tiên)

1. **[Khẩn] Xin xác nhận GVHD + làm văn bản giải trình thay đổi đề cương** (tên đề tài giữ nguyên nhưng hệ triển khai và kết luận đầu đề khác đăng ký) trước ngày đóng gói 10/10/2026. Không có bước này, toàn bộ phán quyết có thể bị hội đồng bác ở cổng thủ tục, bất kể chất lượng khoa học.
2. **Bổ sung vào bảng tự khai Ch1.4.2 một dòng thứ 4**: "xương sống không gian YOLO → tháp CLIP ViT-B/32 (cho hệ thống triển khai đầu đề B1)", cùng lý do kỹ thuật đã có ở Ch3.3.1 — tránh để hội đồng tự phát hiện.
3. **Thêm Confusion Matrix đầy đủ** cho kết quả nhị phân đầu đề (B1 trên iSafetyBench) và/hoặc cho làn 4 lớp trên Unsafe-Net — kể cả dạng phụ lục, để đáp ứng yêu cầu "bắt buộc" của đề cương.
4. **Sửa dứt điểm 24 chỗ nhãn phần cứng M3/M4** trước đóng gói — `00-trang-thai-bien-soan.txt` ghi rõ đây là việc đóng gói còn lại, không phải việc đo, nên không vướng mốc đóng băng số liệu 05/10.
5. **Viết rõ 1 đoạn trong Ch1 hoặc Ch5** nói thẳng: kiến trúc đa luồng camera→edge→server→dashboard của đề cương **chỉ giao được camera→edge**; server lưu trữ và dashboard không nằm trong phạm vi luận văn — tránh im lặng để hội đồng tự hỏi "dashboard đâu".
6. **Đồng bộ README.md** với kết luận headline của luận văn tiếng Việt, hoặc ghi chú rõ README là tài liệu kỹ thuật của pha trước, không phải tóm tắt luận văn — tránh mâu thuẫn công khai trên GitHub.
7. Giải quyết các câu hỏi chưa đóng đã tự liệt ở `00-trang-thai-bien-soan.txt` mục "CÂU HỎI CHƯA GIẢI QUYẾT" (6 câu) — ít nhất phải đọc trước buổi bảo vệ để trả lời được nếu hội đồng hỏi trúng.

---

## 6. CÂU HỎI CHƯA GIẢI QUYẾT (của phản biện, không phải của tác giả)

1. Có văn bản chính thức nào (ngoài repo code) xác nhận GVHD/khoa đã biết và đồng ý với việc đổi hướng đề tài không? Không thể xác minh chỉ từ repo.
2. Đề cương yêu cầu Precision trên "125 clip" — luận văn dùng 64 clip hoặc giao thức leave-source-out/leave-context-out trên 687/1,100 clip. Hai cách này có được trường/hội đồng chấp nhận là "tương đương" với yêu cầu gốc không?
3. `00-trang-thai-bien-soan.txt` tự liệt 6 câu hỏi chưa giải quyết (khớp token 74 chiều, chưa có soak cho quyết định tổ hợp, chưa có p95/p99 tổ hợp, thiếu seed/data_root trong tệp đo, nguồn gốc Bảng 4.5/4.1-4.4, video demo dự phòng chưa tồn tại) — tác giả có kế hoạch đóng các câu này trước 10/10 không, hay sẽ nộp kèm giới hạn đã công bố?
4. Tệp `multi-stream-results.json` tự ghi "hand-patched" vẫn đang là nguồn cho Bảng 5.4 (bản tiếng Anh) — tình trạng này đã được làm rõ chưa tại thời điểm nộp?
