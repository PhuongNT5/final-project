# PHẢN BIỆN 5 — TRIỂN KHAI BIÊN, "THỜI GIAN THỰC", TÁI LẬP

Phạm vi: `report/vn/chuong-05-trien-khai-bien.txt` (BẢN NỘP, 1255 dòng), đối chiếu
`chuong-05-trien-khai-bien-ghim-xung-nhip.txt` (hồ sơ), `00-trang-thai-bien-soan.txt`,
`deployment/` (README, JETSON-*.md, artifacts/*.json/*.md), `docs/deployment-guide.md`,
`docs/system-architecture.md`, `demo/*.py`, `PROPOSAL.txt`. Trích dẫn kèm số dòng của bản nộp
trừ khi ghi khác.

================================================================================
## 1. "HỢP ĐỒNG NHỊP" — TỰ ĐẶT ĐỀ RỒI TỰ CHẤM ĐỖ?

**Cơ sở hợp đồng, nguyên văn:**
- B1: "Hợp đồng nhịp của B1 do bước cửa sổ quy định... nguồn ở *f* khung/giây buộc hệ thống
  phát *f*/24 quyết định mỗi giây mới theo kịp luồng" (dòng 393–396). Ba mức hợp đồng
  (1.249/1.427/2.500 qđ/giây) suy ra từ phân bố khung/giây CỦA CHÍNH KHO iSafetyBench
  (dòng 398–410), KHÔNG từ một yêu cầu an toàn lao động nào.
- P3: "Ở nguồn 24 khung/giây, hợp đồng là 3.0 quyết định/giây" (dòng 647–648) — suy ra từ
  bước nhảy phát quyết định (8 khung) của chính kiến trúc, không từ bài toán an toàn.

**Không tìm thấy** bất kỳ trích dẫn tiêu chuẩn/tài liệu an toàn lao động (OSHA, ISO 45001,
IEC 61508, thời gian phản ứng người vận hành, v.v.) ở BẤT KỲ đâu trong 11 tệp nộp: grep
`OSHA|ISO [0-9]|IEC [0-9]|tiêu chuẩn an toàn|an toàn lao động|thời gian phản ứng|reaction time`
trên toàn bộ `report/vn/*.txt` chỉ trúng ba dòng, cả ba đều là cụm "an toàn lao động công nghiệp"
mang tính mô tả lĩnh vực trong `00-tom-tat.txt`, không phải một con số ngưỡng có dẫn nguồn.

**Đánh giá mức nghiêm trọng: CAO, nhưng có giảm nhẹ.** Luận văn **không che giấu** nguồn gốc của
hợp đồng — tự nhận "suy ra từ chính kho dữ liệu" (abstract dòng 32, 96–97: "ba mức hợp đồng nhịp
suy ra từ phân bố tốc độ khung của chính kho dữ liệu"). Đây là hợp đồng **kỹ thuật** (giữ kịp tốc
độ nguồn), không phải hợp đồng **an toàn** (kịp ngăn một tai nạn). Luận văn không hề định nghĩa
"bao lâu là kịp can thiệp" bằng một con số độc lập — hợp đồng chỉ đảm bảo hệ thống không bị dồn
khung (backlog), không đảm bảo người vận hành có đủ thời gian phản ứng vật lý trước khi tai nạn
xảy ra. Hai khái niệm này **hoàn toàn khác nhau** và luận văn trộn chúng khi dùng chữ "an toàn"
trong tên "quyết định an toàn" ở tiêu đề chương (dòng 143) và trong câu hỏi chương (dòng 163).
Hội đồng có cơ sở để nói: "biên 2.39–4.79 lần" chỉ chứng minh hệ thống không tụt hậu so với nguồn
video — **không chứng minh gì về biên an toàn vận hành thật**.

================================================================================
## 2. DVFS LÀM HEADLINE: HEADLINE THUỘC ĐIỀU KIỆN NÀO, VÀ MÂU THUẪN VỚI RÀNG BUỘC 13

Headline B1 (167.291 ms / 5.978 qđ/giây, dòng 10, tái khẳng định dòng 454, 1216–1219) đo dưới
**DVFS mặc định** — xác nhận tại Bảng 5.2 (dòng 419: "FP16, DVFS mặc định"). Headline P3
(Bảng 5.4b, dòng 675–691) báo cả hai điều kiện, chọn DVFS làm "kết quả tiêu đề" (dòng 692).

`00-trang-thai-bien-soan.txt` ràng buộc 13 (dòng 173–178): "DVFS KHÔNG PHẢI MỘT ĐIỀU KIỆN ĐO LẶP
LẠI ĐƯỢC trên thiết bị này. 11 lần chạy qua hai ngày: ghim nhịp 309.08–311.78 ms (phân tán 0.49%),
DVFS 365.88–440.01 ms (20.3%), hai tập KHÔNG giao nhau." Đây là một ghi chú **nội bộ, không nộp**.

**Phát hiện chính xác:** đây KHÔNG phải một mâu thuẫn lộ liễu trong văn bản nộp (Ch5 không viết
câu nào tự phản bác), nhưng là một **thiếu công bố (omission) nghiêm trọng**:
- Con số phân tán mà Ch5 CÓ công bố cho DVFS của P3 là phân tán TRONG MỘT PHIÊN, 3 lần chạy liên
  tiếp cùng ngày: "2.39% dưới DVFS" (phần tính toán, dòng 731) và "3.19%" khi cộng giải mã
  (dòng 733) — nhỏ, có vẻ ổn.
- Con số 20.3% ở ràng buộc 13 là phân tán GIỮA HAI NGÀY, trên một tập 11 lần chạy rộng hơn — và
  nó **không xuất hiện ở bất kỳ đâu trong Ch5 bản nộp**. Grep toàn văn: không có "20.3%" trong
  `chuong-05-trien-khai-bien.txt`.
- Hệ quả: người đọc Ch5 một mình sẽ tin DVFS là điều kiện ổn định (phân tán ~2–3%). Chỉ người đọc
  thêm tệp ghi chú nội bộ mới biết DVFS có thể trôi tới 20.3% giữa các ngày trên P3 — và B1 chưa
  từng được kiểm tra lặp-lại-được XUYÊN NGÀY dưới DVFS (B1 chỉ có 3 lần chạy, cùng một phiên,
  dòng 420 "3 lần lặp ở cùng hạt giống 0... độ phân tán đo *thiết bị*"). Vì B1 dùng chung
  `schedutil`/governor với P3 trên cùng thiết bị, không có cơ sở để loại trừ rủi ro tương tự cho
  headline B1.

**Mức nghiêm trọng: CAO.** Đây là lỗ hổng tái lập quan trọng nhất của chương: điều kiện được chọn
làm headline (vì "là trạng thái mặc định", dòng 708–711) là chính điều kiện mà tác giả tự phát
hiện kém ổn định nhất trên thiết bị — và phát hiện đó bị giữ lại ngoài bản nộp.

================================================================================
## 3. TRỘN SỐ GIỮA PIPELINE?

Kiểm tra kỹ: Ch5 **không** trộn số B1/P3/P1 ở bất kỳ bảng nào tôi tìm thấy. Mục 5.1.1 (dòng
175–220) thiết lập quy tắc tách bạch rất chặt, lặp lại nhiều lần ("không hàng nào trong đó được
phát biểu lại cho B1 hoặc P3", dòng 206–207; "Hai hợp đồng khác nhau... chúng không được so với
nhau", dòng 1227–1228). Mục 5.6.4 tách rõ kết quả 4-luồng là của P1 và "KHÔNG chuyển sang B1 hoặc
P3" (dòng 929–931). Đây là điểm MẠNH của chương — ngược hẳn với rủi ro mà câu hỏi đặt ra.

**Nhưng** rủi ro trộn số nằm ở NGOÀI Ch5: `report/vn/00-tom-tat.txt` (tóm tắt/abstract) chỉ trích
số của B1 (dòng 31–32, 95–97), không nhắc số P3 cạnh đó — tốt. `README.md` cấp dự án (không phải
tệp nộp luận văn, nhưng là tài liệu công khai cùng kho) VẪN còn đoạn mô tả P3 đạt "+7.5% margin"
khi ghim nhịp làm **headline cho P3** và gọi P1 là "the model currently on the device" — đây là
trạng thái CŨ (trước 03/10), bị Ch5 bản nộp vượt qua (B1 mới là đồ thị triển khai, dòng
196–198; P3 trượt hợp đồng dưới DVFS, không đạt +7.5%). `README.md` và `docs/system-architecture.md`,
`docs/deployment-guide.md` đều **chưa được đồng bộ** với kết luận mới nhất của Ch5 — xem mục 6
dưới đây. Nếu hội đồng đọc các tệp đó (rất có thể, vì Ch5 bản nộp dẫn `deployment/` làm nguồn gốc
tạo tác), họ sẽ thấy ba phát biểu khác nhau về việc P3 đạt hay trượt hợp đồng, tùy tệp nào họ mở.

================================================================================
## 4. LẮP GHÉP TOKEN 74 CHIỀU: LỖ HỔNG CHÍ TỬ VỀ ĐỘ CHÍNH XÁC END-TO-END

Xác nhận từ `00-trang-thai-bien-soan.txt`, "CÂU HỎI CHƯA GIẢI QUYẾT" số 1 (dòng 194–196): "Khâu
lắp ghép token 74 chiều trên thiết bị có khớp SỐ HỌC với pipeline lúc huấn luyện hay không — chưa
kiểm chứng. Huấn luyện lấy 16 khung RẢI ĐỀU trên cả clip; triển khai trượt cửa sổ 16 khung LIÊN
TIẾP." Xác nhận độc lập trong mã: `demo/b1-window-pipeline.py` dòng 12–15 tự thừa nhận: "The
evaluation samples 16 frames spread across a whole short clip. A demo runs on a longer video, so
the same 16-frame tensor is filled from a sliding window instead." Và Ch5 dòng 393–394 xác nhận
"hàm lấy mẫu chọn T = 16 khung cách đều trên cửa sổ, đúng quy tắc mà phần đánh giá dùng" — câu
này chỉ đúng với **cửa sổ**, không chứng minh cửa sổ 48-khung-trượt-24 sinh ra đúng phân bố thống
kê như huấn luyện (lấy mẫu rải đều TRÊN CẢ CLIP, không phải trên một cửa sổ con).

**Hệ quả, đúng như câu hỏi đặt ra:** mọi con số độ trễ của Ch5 (167.291 ms, 5.978 qđ/giây,
biên 2.39–4.79 lần...) là số đo của một **vòng lặp chạy được**, không phải số đo của một **hệ
quyết định đã biết độ chính xác**. Chương 5 tự nhận thức điều này một phần — dòng 328–336 chỉ nói
đúng/sai mô hình B1 so với B1w (độ chính xác offline), không nói gì về độ chính xác CỦA CHÍNH
VÒNG LẶP TRÊN THIẾT BỊ (online, cửa sổ trượt). Không có bảng nào trong Ch5 đo độ chính xác của
vòng lặp triển khai thật.

**Phán quyết: đây là lỗ hổng chí tử về MẶT TUYÊN BỐ, không hẳn về mặt kỹ thuật.** Tuyên bố "hệ
quyết định an toàn chạy trên biên" (tiêu đề chương, dòng 143) ngụ ý một hệ phát hiện hành vi chính
xác tới mức đo trong Ch6 (84.2%) đang chạy trên Jetson. Thực tế: cái chạy trên Jetson là MỘT PHÉP
ĐO TỐC ĐỘ của một vòng lặp có cấu trúc lắp ghép token CHƯA kiểm chứng tương đương toán học với
pipeline đã đánh giá độ chính xác. Độ chính xác 84.2% **không được chứng minh áp dụng được** cho
chuỗi quyết định mà Bảng 5.2 đo tốc độ. Luận văn công bố giới hạn này ở `00-trang-thai-bien-soan.txt`
(tệp nội bộ) nhưng **KHÔNG** nêu rõ trong Ch5 bản nộp — tôi không tìm thấy câu nào trong 1255 dòng
của Ch5 nói "độ chính xác của vòng lặp cửa sổ trượt trên thiết bị chưa được đo". Mục 5.9 (14 điều
chưa đo, dòng 1149–1194) KHÔNG có mục nào nêu đúng vấn đề này — gần nhất là mục (1) "B1 trên
camera thật" (dòng 1155–1163), nhưng mục đó nói về việc chưa đo trên camera trực tiếp, không nói
về việc chưa kiểm chứng khớp số học của phép lấy mẫu cửa sổ trượt so với lấy mẫu rải đều.
**Đây là một thiếu sót công bố cần hội đồng truy vấn trực tiếp.**

================================================================================
## 5. KHÔNG SOAK TỔ HỢP, KHÔNG ĐUÔI PHÂN PHỐI CHO NHỊP TỔ HỢP

Xác nhận kép: Ch5 dòng 1088–1094 tự thừa nhận "Không tồn tại phép soak nào của vòng lặp tổ hợp";
và `00-trang-thai-bien-soan.txt` câu hỏi chưa giải quyết số 2–3 (dòng 197–198): "Chưa có phép đo
SOAK cho quyết định tổ hợp... Chưa có phân vị (p95/p99) cho nhịp tổ hợp — chỉ có trung vị. Không
được viết p95 tổ hợp." Bảng 5.4b (dòng 682–691) chỉ có **3 đoạn × 1 lần chạy** (N=1, 480 khung) —
không có lặp lại, không có phân vị, không có độ bền theo thời gian.

**Đánh giá:** với một hệ tự xưng "quyết định an toàn", chỉ có TRUNG VỊ của 3 điểm dữ liệu
không-lặp-lại là KHÔNG đủ. Một hệ an toàn thực tế bị chi phối bởi đuôi phân phối (worst case),
không phải trung vị — chính Ch5 còn tự nói điều này cho P3 ở mục 5.5.4 (dòng 798–807: "đuôi phân
phối mới là đại lượng ràng buộc"), rồi KHÔNG áp dụng được bài học đó cho chính Bảng 5.4b vì công
cụ đo không ghi phân vị theo quyết định. Đây là một mâu thuẫn nội tại giữa luận điểm phương pháp
(đuôi quan trọng) và năng lực đo đạc hiện có (không đo được đuôi của chính con số headline).
Hội đồng chắc sẽ hỏi: "worst case của 5.978 qđ/giây là bao nhiêu?" — câu trả lời trung thực hiện
tại là "chưa biết".

================================================================================
## 6. ĐA LUỒNG, DASHBOARD, "4 CAMERA": CAM KẾT ĐỀ CƯƠNG SO VỚI THỰC TẾ

`PROPOSAL.txt` dòng 89–91 cam kết: "Xây dựng kiến trúc hệ thống giám sát đa luồng (multi-stream):
Module camera → Edge node (YOLO–Transformer) → Server lưu trữ → Dashboard hiển thị kết quả
real-time." Dòng 186: "...liên tục truyền tải từ camera giám sát."

**Thực tế đối chiếu:**
- `demo/` chỉ có 2 tệp Python (`b1-window-pipeline.py`, `run-unsafe-behavior-demo.py`) — một
  pipeline suy luận đơn-luồng, đơn-video, KHÔNG đọc camera, KHÔNG nối server/dashboard.
- `deployment/dashboard/app.py` và `jetson-live-monitor.py` tồn tại và triển khai đúng kiến trúc
  Camera→Edge→Server→Dashboard (mã nguồn thật, không giả), nhưng:
  - `app.py` chạy trên **CPU/ONNX Runtime trên máy M3**, không phải trên Jetson, và dùng các đầu
    cũ (direct4/cascade/ensemble) — KHÔNG phải B1/CLIP.
  - `jetson-live-monitor.py` đọc luồng sự kiện qua rsync từ `jetson-multi-stream-pipeline.py`,
    nhưng số liệu "8,988 sự kiện / 772.145 giây" (Ch5 dòng 880, nguồn `live-demo-stream.jsonl`) là
    của **P1**, không phải B1 — xác nhận chính Ch5 dòng 929–931 ("kết quả 4 luồng là của P1 và
    KHÔNG chuyển sang B1 hoặc P3").
  - `docs/system-architecture.md` dòng 215: "Jetson edge deployment (latency, FPS, power, soak):
    P1 — exclusively. No P3 Jetson benchmark exists yet." — văn bản này **cũ hơn** cả bản ghim-nhịp
    của Ch5 và hoàn toàn mâu thuẫn với trạng thái hiện tại (P3 và B1 ĐÃ có số đo Jetson riêng).

**"4-stream zero-drop 88 FPS" và "2-stream 36.2-min soak":** cả hai thuộc về đồ thị **P1** (đã xác
nhận ba lần độc lập — Ch5 5.6.4, `docs/deployment-guide.md` dòng 56–57 "All belong to P1",
`README.md` dòng 74). **KHÔNG áp dụng cho B1 hay P3**, và theo chính Ch5 (mục 5.6.4, dòng 921–927),
tạo tác nguồn của số 88.06 FPS **không mang khối xuất xứ nào** (không `clocks_pinned`, không
`competing_processes`) và **không được git theo dõi** — nó thất bại điều kiện (Đ2) của chính
chương này và chỉ được giữ "với tư cách số lịch sử".

**Kết luận mục 6:** kiến trúc camera→edge→server→dashboard **có tồn tại và chạy được bằng mã
thật** (không mô phỏng), nhưng (a) chưa từng chạy với camera thật (toàn bộ là tệp video thay thế —
Ch5 hạn chế (14), dòng 1193–1194), (b) số đo thông lượng đa luồng duy nhất thuộc về một pipeline
đã bị thay thế (P1) và tự thất bại điều kiện xuất xứ của chính luận văn. Tuyên bố "hệ thống phục
vụ N camera" **không có cơ sở đo đạc nào cho hệ headline hiện tại** — đúng như Ch5 tự liệt ở mục
5.9(6).

================================================================================
## 7. `drop_rate = 0` CÓ ĐƯỢC DÙNG LÀM BẰNG CHỨNG TRONG CH5 KHÔNG?

Grep toàn văn Ch5 bản nộp cho `drop_rate`: chỉ 3 lần, cả ba đều trong khối CẢNH BÁO ở đầu tệp
(dòng 108, 112, 703) **cấm** dùng `drop_rate = 0` làm bằng chứng — không có chỗ nào trong THÂN
chương thực sự trích `drop_rate` làm bằng chứng tích cực. **Ch5 tuân thủ ràng buộc 12(2).**

Nhưng README.md cấp dự án (không phải Ch5, nhưng cùng kho, công khai) câu "4-stream zero-drop
throughput (88 FPS agg)... 0 drops" (dòng 74) **chính là** kiểu phát biểu mà ràng buộc 12(2) cấm —
dùng "0 drop" như bằng chứng thời gian thực cho N=4 luồng. Theo đúng lý do kỹ thuật mà Ch5 tự nêu
(dòng 108–112: bộ đọc cùng vòng lặp sự kiện với bộ tiêu thụ ⇒ lockstep, không phải không rơi khung
thật), con số "0 drop" ở N=4 cũng đáng ngờ theo cùng cơ chế — nhưng vì tạo tác N=4 không mang
`read_fps`/`target_fps` để kiểm tra, không thể xác nhận hay bác bỏ. **Rủi ro: nếu hội đồng đọc cả
README lẫn Ch5, họ sẽ thấy hai tiêu chuẩn chứng cứ khác nhau áp dụng cho hai tài liệu của cùng một
tác giả.**

================================================================================
## 8. "KHUNG HÌNH KHÔNG BAO GIỜ RỜI THIẾT BỊ": THIẾT KẾ TRÊN GIẤY HAY CÓ KIỂM MÃ?

Ch5 mục 5.8.1 (dòng 1100–1127) trình bày ranh giới riêng tư như MỘT THUỘC TÍNH KIẾN TRÚC ("không
có đường dẫn nào để chúng vượt", dòng 1117) và tính lại băng thông đúng (221.0 B/sự kiện, 20.6
kbps, sửa lỗi K1 2.35 lần). Kiểm mã xác nhận một phần:
- `jetson-live-monitor.py` dòng 9–19 (docstring kiến trúc) khẳng định "Raw video frames NEVER
  leave the Jetson; only prediction metadata... crosses the network" và mô tả đúng luồng
  Jetson→rsync→M3 dashboard — nhất quán với Ch5.
- NHƯNG: đây là docstring/comment, không phải một bài kiểm (test) hay một log lưu lượng mạng thực
  đo được. Ch5 tự thừa nhận: 20.6 kbps là TÍNH TỪ kích thước tệp sự kiện và tần suất sự kiện
  (dòng 1123–1125), KHÔNG phải đo trực tiếp lưu lượng mạng (ví dụ bằng `tcpdump`/`iftop`) để xác
  nhận KHÔNG có kênh phụ nào khác đang gửi dữ liệu lớn hơn (ví dụ log debug, ảnh crop lỗi, hay
  một đường truyền hình ảnh gỡ lỗi để lại từ phát triển).
- Bằng chứng kỹ thuật MẠNH NHẤT mà luận văn có là một đối số phủ định đúng: không tìm thấy đoạn
  mã nào trong `deployment/scripts/*.py` hay `demo/*.py` ghi/gửi khung ảnh thô ra ngoài thiết bị —
  nhưng đây là một kết luận suy ra từ ĐỌC MÃ (của tôi, trong phiên phản biện này), không phải một
  phép KIỂM TRA TỰ ĐỘNG mà luận văn tự chạy và báo cáo.

**Đánh giá: phát biểu đúng ở mức thiết kế và nhất quán với mã, nhưng CHƯA có một bài kiểm độc lập
đo lưu lượng mạng thực tế để loại trừ kênh rò dữ liệu ngoài ý muốn.** Mức rủi ro: THẤP (kiến trúc
đơn giản, dễ kiểm tra bằng đọc mã) nhưng tuyên bố "bảo đảm bằng kiến trúc" (tên mục 5.8) là một
tuyên bố MẠNH hơn bằng chứng đang có.

================================================================================
## 9. TENSORRT + INT8 CHO HỆ HEADLINE (CLIP ViT-B/32 + ĐẦU 201,090 THAM SỐ)?

Xác nhận từ nhiều nguồn độc lập:
- Ch5 mục 5.3 (dòng 318–320): "engine FP32 cùng INT8 của tháp CLIP chưa được dựng lại cho đúng
  thiết bị nên số của chúng chưa được trích dẫn."
- `deployment/artifacts/trt-compat-report.md`: liệt kê 18 tệp ONNX kiểm compat TRT — **không có
  tệp nào tên `r48-clip-vit-b32-vision` hay bất kỳ biến thể INT8 của nó**. Chỉ có INT8 cho
  backbone YOLO cũ và 4 đầu cũ (direct4/eight/pair*/scenario) — toàn bộ thuộc P1/P2, không thuộc
  B1.
- `find deployment/export -iname "*clip*int8*"` → rỗng. Không một engine INT8 nào tồn tại cho
  thành phần chiếm 52.6% chi phí tính toán của hệ headline (Bảng của mục 5.4.4, dòng 487–488).
- `docs/deployment-guide.md` dòng 74–78, 82–85 xác nhận INT8 trên Jetson dùng calibrator riêng
  của TensorRT, và mục "What is NOT yet available" (dòng 80–85) liệt "Real (non-synthetic) INT8
  calibration data for the backbone" là CHƯA CÓ — dữ liệu calib hiện là **nhiễu ngẫu nhiên**, nên
  dù có build được engine INT8 cho backbone cũ, số liệu của nó cũng không đại diện cho production.

**Kết luận mục 9:** TensorRT ĐƯỢC dùng cho hệ headline (B1 chạy trên engine FP16 dựng bằng
TensorRT trên chính thiết bị — điều kiện Đ1). INT8 **KHÔNG** được dùng/đo cho hệ headline ở bất kỳ
đâu — toàn bộ số INT8 trong kho thuộc các pipeline/thành phần đã bị thay thế. Cam kết đề cương
("biên dịch TensorRT, lượng tử hóa INT8", `PROPOSAL.txt` dòng 51) chỉ được thực hiện **một nửa**
cho hệ cuối cùng.

================================================================================
## 10. KHẢ NĂNG TÁI LẬP

Kiểm chứng trực tiếp trên tạo tác `r48-composed-cadence-fp16-20261003T155728.json` (nguồn của
hàng SAU trong Bảng 5.2, headline 167.291 ms): các trường cấp cao gồm `n_clips`, `n_decisions`,
`frames_decoded`, `window`, `stride`, `wall_sec_total`, các phân vị theo giai đoạn, `per_clip`
(tên tệp clip + số khung + wall_sec mỗi clip) — **KHÔNG có trường `seed`, `data_root`, hay kích
thước khung hình của từng đoạn**. Đúng như `00-trang-thai-bien-soan.txt` câu hỏi 4 (dòng 199–200)
tự nhận: "Tệp đo nhịp thiếu `seed`, `data_root`, và kích thước từng đoạn video (hoãn sửa: thay đổi
mã đo quá sát mốc đóng băng 05/10)."

Script `deployment/scripts/benchmark-r48-composed-cadence.py` CÓ nhận `--seed` (default 0) và
`--data_root` làm đối số CLI (dòng 122, 145 của script) — nghĩa là người chạy lại CÓ THỂ chỉ định
seed/data_root, nhưng **giá trị đã dùng không được ghi lại trong tạo tác JSON đầu ra**, nên một
người ngoài cầm đúng tệp JSON này **không thể biết** lệnh gốc dùng seed nào, trỏ `data_root` nào —
chỉ có thể biết DANH SÁCH TÊN CLIP qua `per_clip`, và from đó suy luận lại.

**Có tái lập được không?** CÓ MỘT PHẦN. Danh sách clip cụ thể được ghi lại (nên có thể tái tạo
đúng tập dữ liệu nếu `data/isafetybench` còn giữ cùng cấu trúc), script tồn tại và có hướng dẫn
chạy tối thiểu (`deployment/README.md`, `JETSON-DEPLOYMENT.md`), nhưng:
- Thiếu `seed`/`data_root` trong tạo tác làm việc ĐỐI CHIẾU ngược (verify một con số đã công bố
  khớp đúng lệnh nào) không thể tự động hoá hoàn toàn — phải dò ngược qua tên clip.
- Không có tệp `requirements.txt`/lockfile phiên bản chính xác nào được dẫn trong Ch5 cho môi
  trường Jetson (JetPack/TensorRT version có ghi trong một số tạo tác cũ nhưng không nhất quán —
  xem mâu thuẫn M3/M4 CPU nhãn sai 24 chỗ, dòng 952–953, mục 5.6.5).
- Engine `.engine` không được commit (dòng 848–849: "174 MB vượt hạn mức 100 MB của GitHub") —
  người ngoài PHẢI tự build lại engine trên đúng dòng thiết bị (Jetson Orin Nano Super, JetPack
  version cụ thể) vì cảnh báo (Đ1) nói engine không portable giữa các thiết bị cùng dòng.

**Đánh giá: tái lập được VỀ NGUYÊN TẮC (mã + hướng dẫn tồn tại, idempotent theo README), nhưng
TỐN CÔNG và PHỤ THUỘC PHẦN CỨNG CỤ THỂ** (phải có chính xác một Jetson Orin Nano Super, build lại
engine tại chỗ — không dùng engine có sẵn được). Thiếu seed/data_root trong tạo tác là một khiếm
khuyết tái lập nhỏ-đến-vừa (SỬA BẰNG CÂU CHỮ không đủ — cần SỬA MÃ ĐO, đã bị tác giả hoãn vì sát
mốc đóng băng).

================================================================================
## 11. VIDEO DEMO DỰ PHÒNG CHƯA TỒN TẠI

Xác nhận: `00-trang-thai-bien-soan.txt` câu hỏi 6 (dòng 203) và mục "VIỆC CÒN LẠI" số 4 (dòng 222):
"Video demo dự phòng chưa tồn tại." Kiểm tra trực tiếp: `demo/` chỉ chứa 2 tệp `.py` và
`__pycache__` — **không có tệp video/`.mp4` nào, không có bản ghi màn hình nào**. Script
`demo/run-unsafe-behavior-demo.py` CÓ THỂ sinh một video chú giải (`--out demo/out`), nhưng
chưa có bằng chứng nó đã được CHẠY và LƯU LẠI kết quả trong kho.

**Rủi ro khi bảo vệ:** nếu demo trực tiếp (live, trên Jetson thật hoặc qua `run-unsafe-behavior-
demo.py`) thất bại (lỗi driver, mất kết nối SSH tới Jetson, phần cứng không khởi động được,
mạng phòng bảo vệ không ổn định cho rsync dashboard) — **không có phương án B**. Với một luận văn
tự xưng "thời gian thực trên thiết bị biên", việc không trình diễn được trực tiếp sẽ không phải
lỗi chí tử về khoa học (số liệu vẫn đứng), nhưng là một lỗ hổng NGHI VẤN lớn trước hội đồng.

**Khuyến nghị cụ thể:** trước 10/10/2026 (mốc đóng gói), ít nhất PHẢI có:
1. Một video màn hình (không cần máy quay) chạy `demo/run-unsafe-behavior-demo.py` trên ít nhất
   2-3 clip (một an toàn, một không an toàn) với đầu ra video chú giải đã lưu sẵn trong
   `demo/out/`.
2. Một ảnh chụp màn hình hoặc video ngắn của `jetson-live-monitor.py` dashboard đang hiển thị
   alert/warning/monitoring tier, dùng `live-demo-stream.jsonl` đã có sẵn (không cần Jetson sống).
Cả hai không tốn thêm "số đo mới" (không vi phạm đóng băng số 05/10) — chỉ là đóng gói bằng chứng
đã có thành định dạng trình diễn được. Việc này nằm trong "việc đóng gói" mà chính tác giả đã xác
định còn phải làm trước 10/10 (dòng 214 "VIỆC CÒN LẠI TRƯỚC ĐÓNG BĂNG 05/10 — việc đóng gói").

================================================================================
## PHỤ LỤC: ĐIỂM MẠNH CẦN GHI NHẬN (để cân bằng)

Để phán quyết công bằng, phải ghi nhận: Ch5 bản nộp là một trong những chương triển khai biên
TỰ-PHÊ-BÌNH nghiêm ngặt nhất có thể gặp — nó tự rút lại 7+ phát biểu của chính mình trong vòng
một tuần (mục 5.2.2, 5.10), phân biệt pipeline rất kỷ luật, công bố cả quy tắc đọc tạo tác sai
(3 "bẫy" ở đầu tệp), và đưa "điều chưa đo" thành MỘT MỤC KẾT QUẢ (5.9) chứ không giấu ở phần hạn
chế. Phần lớn lỗ hổng tôi tìm thấy là lỗ hổng mà tác giả **đã biết** (ghi trong
`00-trang-thai-bien-soan.txt`) nhưng CHƯA kịp sửa vì mốc đóng băng 05/10 — khác về chất với lỗ
hổng do không biết.

================================================================================
# TRẢ LỜI THEO YÊU CẦU ĐẦU RA

## (a) Phán quyết tuyên bố "thời gian thực"

**CHỨNG MINH CÓ ĐIỀU KIỆN.**

Điều kiện phải ghi kèm MỖI LẦN câu "hệ thống đạt thời gian thực" xuất hiện (trong luận văn, trong
slide bảo vệ, trong trả lời hội đồng):
1. "Thời gian thực" = giữ kịp tốc độ khung của NGUỒN VIDEO TỆP, theo một hợp đồng **tự suy ra từ
   kho dữ liệu**, KHÔNG phải một ngưỡng an toàn lao động có dẫn nguồn độc lập.
2. Chỉ đúng cho đường **B1** (CLIP ViT-B/32 + đầu 201,090 tham số), đo dưới **DVFS mặc định**,
   trên TỆP VIDEO (không phải camera sống), với độ chính xác của CHÍNH VÒNG LẶP TRÊN THIẾT BỊ
   **chưa được đo** (chỉ đo tốc độ, không đo độ đúng của quyết định trong cấu hình cửa sổ trượt).
3. Chưa có bằng chứng độ bền (soak) cho vòng lặp tổ hợp, chưa có phân vị p95/p99 cho nhịp tổ hợp —
   chỉ có trung vị của N rất nhỏ.
4. Đường **P3** KHÔNG đạt thời gian thực dưới điều kiện mặc định (trượt 1.99–2.63 lần trên cả 3
   đoạn test) — luận văn ghi đúng điều này và không tuyên bố ngược lại.
5. Đa luồng/nhiều camera: KHÔNG có bằng chứng cho hệ headline (B1/P3); bằng chứng duy nhất
   (4-luồng, 88 FPS) thuộc một pipeline đã bị thay thế (P1) và tự thất bại điều kiện xuất xứ của
   chính luận văn.

Nếu bỏ qua điều kiện 1 (hợp đồng tự suy, không phải ngưỡng an toàn) và điều kiện 2 (độ chính xác
vòng lặp thật chưa đo), tuyên bố sẽ bị hội đồng coi là **tự đặt bài thi rồi tự chấm đỗ** — đúng
như nghi ngại nêu trong nhiệm vụ.

## (b) Lỗ hổng xếp theo rủi ro bảo vệ (8–12 mục)

1. **Hợp đồng nhịp tự suy từ dataset, không dẫn một tiêu chuẩn an toàn lao động nào.**
   SỬA BẰNG CÂU CHỮ — đổi tên "hợp đồng an toàn" → "hợp đồng giữ-kịp-nguồn (source-pacing
   contract)" ở mọi nơi, thêm một câu minh bạch rằng đây không phải ngưỡng an toàn vận hành.

2. **Độ chính xác của vòng lặp triển khai thật (cửa sổ trượt 16 khung liên tiếp) chưa từng được
   đo, và Ch5 không nêu rõ giới hạn này ở bất kỳ câu nào trong thân chương.**
   CẦN ĐO THÊM (lý tưởng) / SỬA BẰNG CÂU CHỮ (tối thiểu — thêm một câu công bố rõ ở mục 5.9 hoặc
   ngay đầu mục 5.4, không chỉ giữ trong tệp nội bộ).

3. **DVFS được chọn làm headline trong khi ghi chú nội bộ (ràng buộc 13) cho thấy DVFS trôi 20.3%
   xuyên ngày trên P3 — thông tin này không xuất hiện trong Ch5 bản nộp.**
   CẦN ĐO THÊM (đo lại B1 dưới DVFS ở một ngày khác để kiểm tra) hoặc tối thiểu SỬA BẰNG CÂU CHỮ
   (công bố rủi ro trôi xuyên-phiên ngay trong Ch5, không chỉ trong ghi chú nội bộ).

4. **Không có phép soak cho vòng lặp tổ hợp (B1 hay P3); không có p95/p99 cho nhịp tổ hợp.**
   CẦN ĐO THÊM — nhưng đã bị đóng băng số 05/10. KHÔNG CHỮA ĐƯỢC TRƯỚC 10/10 nếu giữ mốc đóng
   băng; chỉ có thể SỬA BẰNG CÂU CHỮ (nêu rõ là giới hạn, đã làm ở mục 5.9(4) và 5.7.3).

5. **Đa luồng/nhiều camera: bằng chứng duy nhất thuộc pipeline đã bị thay thế (P1), tự thất bại
   điều kiện xuất xứ (Đ2) của chính luận văn, và tài liệu `docs/` + `README.md` cấp dự án chưa
   đồng bộ với Ch5 bản nộp (vẫn mô tả trạng thái P1-là-chính hoặc P3-đạt-+7.5%-ghim-nhịp).**
   SỬA BẰNG CÂU CHỮ — cập nhật `docs/system-architecture.md`, `docs/deployment-guide.md`,
   `README.md` khớp với Ch5 bản nộp trước khi nộp/bảo vệ (việc này không cần đo mới, chỉ cần viết
   lại theo số đã có).

6. **INT8 không được đo/dùng cho hệ headline (tháp CLIP) — cam kết đề cương chỉ thực hiện một
   nửa.** SỬA BẰNG CÂU CHỮ (nêu rõ ở mục 5.3/5.9 rằng INT8 cho CLIP là công việc tương lai, không
   phải đã làm) — Ch5 đã làm điều này ở dòng 318–320 và mục 5.9(10); chỉ cần đảm bảo không có câu
   nào khác trong luận văn (Ch1, abstract, slide) ngụ ý INT8 đã áp dụng cho hệ headline.

7. **Video demo dự phòng chưa tồn tại — rủi ro khi demo live thất bại trước hội đồng.**
   CẦN ĐO THÊM (thực ra là "cần đóng gói thêm", không cần số đo mới) — làm được trước 10/10, xem
   khuyến nghị cụ thể ở mục 11 trên.

8. **Tái lập: tạo tác nhịp thiếu `seed`, `data_root`; engine không commit; không lockfile phiên
   bản phần mềm nhất quán.**
   SỬA BẰNG CÂU CHỮ (tối thiểu — nêu rõ giới hạn, đã làm một phần ở mục 5.9) + CẦN ĐO THÊM (lý
   tưởng — sửa script để ghi seed/data_root vào tạo tác, nhưng tác giả đã quyết định hoãn vì sát
   mốc đóng băng; đây là quyết định hợp lý, không phải sai sót).

9. **Tuyên bố riêng tư "khung không rời thiết bị" chưa có một bài kiểm lưu lượng mạng độc lập,
   chỉ có đối số kiến trúc + đọc mã.**
   SỬA BẰNG CÂU CHỮ — hạ bớt độ mạnh của câu "bảo đảm bằng kiến trúc" thành "theo thiết kế mã
   nguồn hiện có, không tìm thấy đường dẫn nào gửi khung thô; chưa có phép đo lưu lượng mạng độc
   lập xác nhận".

10. **`README.md` cấp dự án còn dùng `drop_rate=0`/"zero-drop" làm bằng chứng thời gian thực cho
    N=4 — đúng kiểu phát biểu mà chính Ch5 cấm ở ràng buộc 12(2).**
    SỬA BẰNG CÂU CHỮ — sửa `README.md` hoặc ít nhất thêm chú thích giống `docs/deployment-guide.md`
    đã có ("All belong to P1").

11. **Trộn lẫn thông tin giữa các tệp `docs/` (cũ) và Ch5 bản nộp (mới) có thể khiến hội đồng
    nhận hai câu trả lời khác nhau cho cùng một câu hỏi** (ví dụ "P3 có đạt hợp đồng không?").
    SỬA BẰNG CÂU CHỮ — việc đồng bộ `docs/` là bắt buộc trước bảo vệ, không tốn công đo.

## (c) Câu hỏi hội đồng sẽ hỏi + câu trả lời tốt nhất hiện có

- **H: "Hợp đồng 1.249–2.500 qđ/giây ở đâu ra? Dựa trên tiêu chuẩn an toàn lao động nào?"**
  Đ: Không dựa trên tiêu chuẩn an toàn lao động nào; đây là hợp đồng giữ-kịp-tốc-độ-nguồn, suy ra
  từ phân bố khung/giây của chính kho iSafetyBench (mục 5.4.4). Luận văn không tuyên bố đây là
  ngưỡng an toàn vận hành; đó là giới hạn cần nêu rõ hơn trong bản sửa tới.

- **H: "Độ chính xác 84.2% có áp dụng cho cái đang chạy trên Jetson không?"**
  Đ: Chưa chứng minh được bằng số — vòng lặp trên thiết bị dùng cửa sổ trượt 16 khung liên tiếp,
  khác cách lấy mẫu 16 khung rải đều lúc huấn luyện/đánh giá; khớp số học giữa hai cách lấy mẫu
  CHƯA được kiểm chứng (câu hỏi chưa giải quyết số 1, tác giả tự nhận).

- **H: "Tại sao chọn DVFS làm headline nếu nó kém ổn định hơn ghim nhịp?"**
  Đ: Vì DVFS là trạng thái mặc định một hệ triển khai thực tế vận hành, không cần hành động quản
  trị nào; báo cáo số xấu hơn là chiều thận trọng (mục 5.5.2). Cần bổ sung: thừa nhận rủi ro trôi
  xuyên-phiên (20.3%, hiện chỉ ghi nội bộ) ngay trong luận văn.

- **H: "Hệ có chạy được với 4 camera thật không?"**
  Đ: Kiến trúc mã có hỗ trợ (dashboard, rsync, aggregation), nhưng số đo đa luồng duy nhất (88
  FPS, 0 drop) thuộc pipeline cũ (P1) đã bị thay B1 thay thế, và tạo tác của nó không đạt điều
  kiện xuất xứ mà chính chương này đặt ra. Chưa có số đo đa luồng cho B1/P3.

- **H: "INT8 có dùng không?"**
  Đ: Có dùng TensorRT (FP16) cho hệ headline, nhưng INT8 chỉ tồn tại cho các thành phần/pipeline
  cũ (YOLO backbone, các đầu cascade), không cho tháp CLIP — thành phần chiếm hơn nửa chi phí tính
  toán của hệ headline.

- **H: "Nếu demo live thất bại thì sao?"**
  Đ: Hiện chưa có phương án dự phòng dạng video đã ghi sẵn; đây là một rủi ro đã được tác giả tự
  nhận diện (`00-trang-thai-bien-soan.txt`) và cần khắc phục trước 10/10.

## (d) Phán quyết trục triển khai

**ĐẠT CÓ ĐIỀU KIỆN.**

Lý do đạt: phương pháp luận đo đạc (4 điều kiện Đ1–Đ4, phân xử ghi trước, phân biệt pipeline, tự
rút lại số sai) đạt chuẩn mực hiếm thấy; engine TensorRT FP16 thật, chạy thật, đo lại được một
phần; năng lượng/bộ nhớ/nhiệt có số đo thật và nhất quán nội tại.

Lý do có điều kiện: (i) hợp đồng "thời gian thực" là hợp đồng kỹ thuật tự suy, bị gắn nhãn "an
toàn" không tương xứng; (ii) độ chính xác của vòng lặp triển khai thật chưa được chứng minh;
(iii) không có soak/percentile cho nhịp tổ hợp; (iv) cam kết đa luồng/camera/dashboard/INT8 của
đề cương chỉ đạt được cho pipeline đã bị thay thế hoặc không đạt được cho hệ headline; (v) tài
liệu `docs/` và `README.md` cấp dự án chưa đồng bộ với Ch5 bản nộp, tạo rủi ro mâu thuẫn trước hội
đồng; (vi) chưa có video demo dự phòng.

Điều kiện để nâng lên ĐẠT: sửa 11 điểm ở mục (b) — phần lớn là SỬA BẰNG CÂU CHỮ, có thể hoàn tất
trước 10/10/2026 mà không vi phạm mốc đóng băng số 05/10.

## Câu hỏi chưa giải quyết (của phản biện này)

1. Con số phân tán 20.3% DVFS xuyên-ngày (ràng buộc 13) được tính trên tập 11 lần chạy nào chính
   xác — có bao gồm cả các điểm tải p95/p99/trần của Bảng 5.4 hay chỉ p50? Không xác định được chỉ
   từ các tệp đã đọc; cần hỏi tác giả hoặc mở thêm artefact JSON chưa được liệt trong Ch5.
2. `demo/run-unsafe-behavior-demo.py` đã từng được CHẠY THỬ thành công trên dữ liệu thật chưa (có
   `demo/out/*.mp4` nào đã sinh ra ở máy tác giả, chỉ chưa commit) — hay script hoàn toàn chưa
   chạy? Ảnh hưởng trực tiếp đến mức độ cấp thiết của khuyến nghị ở mục 11.
