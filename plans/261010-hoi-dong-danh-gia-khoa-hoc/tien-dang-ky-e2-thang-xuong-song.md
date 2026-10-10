# TIỀN ĐĂNG KÝ — E2: THANG XƯƠNG SỐNG (B/32 → L/14)

**Viết và commit TRƯỚC khi trích đặc trưng L/14. Không một con số nào của thí nghiệm này tồn tại
khi dòng chữ này được ghi.**

| | |
|---|---|
| Ngày | 2026-10-10 |
| Nhánh | `next-round` |
| Commit nền | 90b7310 |
| Căn cứ | Biên bản hội đồng đánh giá khoa học 10/10, §5/E2 |

---

## 1. GIẢ THUYẾT ĐƯỢC KIỂM

Trần 84.2% BA của làn B1 **nằm ở biểu diễn** (xương sống đóng băng CLIP ViT-B/32 không mã hoá đủ
thông tin), **hay nằm ở nhãn/dữ liệu** (nhiễu nhãn + phương sai nguồn)?

Ba phép đo nội bộ 10/10 đã loại trừ các lời giải thích rẻ: ngưỡng (+0.12 pp), ngưỡng nested
(−0.47 pp), ensemble 5 mô hình (+0.33 pp). 87/1,100 clip sai ở **cả 5** mô hình — nhưng cả 5 đều
dùng chung tháp B/32. Thí nghiệm này là cách duy nhất phân biệt hai khả năng trên.

## 2. CAN THIỆP — ĐÚNG MỘT BIẾN

Đổi **duy nhất tháp đóng băng**: `openai/clip-vit-base-patch32` (512-d) → `openai/clip-vit-large-patch14` (768-d).

Giữ nguyên tuyệt đối:
- Đầu phân loại `WholeFrameTemporal(t_layers=1)` — `in_dim` đọc từ tệp đặc trưng, không chỉnh tay.
- Giao thức `StratifiedGroupKFold(10, seed=0)` theo 151 nguồn, LSO-1100, cùng `--seeds`/`--epochs`.
- Lấy mẫu 16 khung cách đều qua `frame_sampling.decode` — cùng khung, cùng thứ tự clip.
- **Mặt nạ hiệu lực khung (`mask`) được sao y từ `feats_isbclip_all.pt`**, khớp theo `clip_id`,
  có assert. Lý do: `esta_training.parts_for` rơi về `torch.ones` khi thiếu `mask`, và điều đó
  sẽ làm B1-L/14 thấy nhiều khung hơn B1-B/32 — một biến thứ hai. Không chấp nhận.

## 3. LUẬT QUYẾT ĐỊNH — CHỐT TRƯỚC KHI THẤY SỐ

Đọc trên **ΔBA so với 84.17%**, kèm CI bootstrap theo cụm (nguồn):

| Kết quả | Kết luận đã chốt trước |
|---|---|
| **≥ +3.0 pp**, CI loại trừ 0 | Dư địa biểu diễn **có thật**. Thang xương sống là khoản đầu tư đúng → chạy tiếp SigLIP2, DINOv3. |
| **+1.5 … +3.0 pp** | Mơ hồ. **Không được tuyên bố gì** cho tới khi có bậc thang thứ hai. |
| **≤ +1.5 pp** | Bằng chứng rằng giới hạn **không nằm ở biểu diễn**. Chuyển toàn bộ công sức sang E1 (trần nhãn) và E3 (nguồn khó). **Đây là kết quả dương, không phải thất bại.** |
| **< −1.5 pp** | Tháp lớn hơn *xấu hơn* trên tác vụ này dù nhiều dung lượng — báo cáo đúng như thế, cùng họ với phát hiện B1xl 696K < B1 201K. |

## 4. DỰ ĐOÁN KHẢ BÁC BỎ, SẮC HƠN CẢ SỐ TIÊU ĐỀ

Nếu giới hạn là **nhãn/dữ liệu** thì đồng thời phải thấy cả ba:
1. Dải BA giữa các fold **không** co lại đáng kể (hiện 29.7 pp).
2. Phần lớn trong 87 clip "sai ở cả 5 mô hình" **vẫn sai**.
3. Lỗi vẫn tập trung ở fold 6/3/4 (hiện gánh 48.0% tổng lỗi).

Nếu giới hạn là **biểu diễn** thì ít nhất (1) và (2) phải đổi. Ba chỉ báo này được ghi ở đây để
không thể giải thích hậu nghiệm theo hướng nào cũng được.

## 5. ĐỘ ĐO ĐƯỢC PHÉP TRÍCH DẪN

Cho phép: BA, macro-F1, BA từng fold, số clip sai-ở-mọi-mô-hình, CI bootstrap theo cụm, McNemar
ghép cặp trên OOF so với B1-B/32.
**Không** cho phép: ECE/hiệu chuẩn (bin quá mỏng — xem câu hỏi mở #5 của biên bản).

## 6. GIỚI HẠN PHẠM VI — BẮT BUỘC

Số của thí nghiệm này đo **sau** mốc đóng băng 05/10. Nó **không** được xuất hiện trong
`report/vn/`, trong bản bảo vệ, hay trong bất kỳ câu trả lời nào trước hội đồng bảo vệ. Nó thuộc
nhánh `next-round` và chỉ dùng cho chương trình sau bảo vệ.

---

## PHỤ LỤC — SỬA ĐỔI TIỀN ĐĂNG KÝ, 2026-10-10, TRƯỚC KHI CÓ BẤT KỲ SỐ NÀO

Trích đặc trưng L/14 vẫn đang chạy; không một con số L/14 nào tồn tại khi phụ lục này được ghi.
Phép thử đường ống (tệp đặc trưng giả 768-d, đã xoá) phơi ra một nhiễu **không thể tránh** mà
§2 nói "giữ nguyên đầu phân loại" đã phát biểu quá mạnh:

**Đầu phân loại không giữ nguyên số tham số.** Cấu hình giống hệt (`WholeFrameTemporal`,
`t_layers=1`, cùng `d`/`heads`/`ff`/dropout/BS/LR/epochs), nhưng lớp chiếu đầu vào phải nở theo
chiều đặc trưng: **201,090 → 233,858 tham số (+32,768, +16.3%)**.

Đây là nhiễu nội tại của *mọi* thí nghiệm thang xương sống: không thể đổi chiều đặc trưng mà
không đổi lớp đầu tiên. Khai ra thay vì giấu. Nó **không** cứu được một kết quả dương, vì:

- B1xl có **696,962** tham số (gấp 3.5×) và cho **82.7%**, tức *thấp hơn* B1 201K là 1.5 pp.
  Trên chính kho này, thêm dung lượng cho đầu phân loại đã được đo là **đi lùi**.
- Nên nếu L/14 thắng, "chỉ vì thêm 32K tham số" không phải lời giải thích khả dụng — hướng của
  bằng chứng hiện có đi ngược lại.
- Ngược lại, nếu L/14 **thua**, +16.3% tham số là một lời giải thích cạnh tranh phải nêu, và khi
  đó luật quyết định ở §3 (ô "< −1.5 pp") được đọc kèm cảnh báo này.

Luật quyết định §3 và ba dự đoán §4 **không đổi**.

---

## KẾT QUẢ — 2026-10-10, ĐỌC THEO LUẬT §3 KHÔNG SỬA NGƯỠNG

Trích đặc trưng: 94.9 phút, 1,100/1,100 clip, `backbone: openai/clip-vit-large-patch14`,
`wholeframe (1100,16,768)`. **Mặt nạ sao y đã kiểm khớp byte-đúng** với `feats_isbclip_all.pt`:
52,997 phát hiện, 15,874 khung hợp lệ ở *cả hai* bên. Tháp là biến duy nhất, như §2 cam kết.
Tham số đầu phân loại: **233,858**, đúng con số phụ lục đã khai trước.

| | BA | macro-F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| B1 / CLIP ViT-B/32 (nền) | 84.17% | 83.5 | 0.792 | 0.897 |
| B1 / CLIP ViT-L/14 (E2) | **87.42%** | 87.0 | 0.905 | 0.940 |
| **ΔBA** | **+3.25 pp** | | | |

- **Cluster bootstrap theo 151 nguồn** (đơn vị đúng, 5000/5000 lần rút hợp lệ): +3.2 pp,
  95% CI **[−1.4, +8.0] — CHỨA 0**.
- Clip-level CI [+1.1, +5.3] — **đơn vị sai**, chỉ khai báo.
- **McNemar chính xác:** b=46, c=84, 130 ô bất đồng, **p = 0.00109**.

### Phán quyết: ô trên cùng của §3 KHÔNG kích hoạt

§3 đòi **phép hội**: "≥ +3.0 pp **và** CI loại trừ 0". Biên độ đạt, CI không đạt. Điều kiện CI có
trong luật chính là để chặn tuyên bố dựa trên điểm ước lượng, nên **không được tuyên bố "dư địa
biểu diễn có thật"**. Chỉ thị hiệu lực là của ô thứ hai: **chưa tuyên bố gì cho tới bậc thang thứ hai.**

Đây **không** phải cách đọc tuỳ ứng: sổ số đã có tiền lệ cùng dạng — **G15 (+3.5 pp, CI chứa 0)
được ghi là "FAILS — margin clears, CI does not"**. E2 nhận đúng chuẩn đó.

**CI chứa 0 ở đây là vấn đề lực thống kê, không phải vắng hiệu ứng.** Hai bằng chứng:
lượt chạy lại cùng cấu hình cho `rows` **bit-identical** (sàn nhiễu run-to-run = **0/1100 clip**),
nên cả 130 ô bất đồng là tín hiệu thật; và cùng giao thức 151 cụm *có* phân giải được một hiệu ứng
6 pp (xem dưới). 151 cụm không đủ cho 3 pp — đó là thuộc tính đo được của giao thức.

### Ba dự đoán §4

| # | Dự đoán nếu giới hạn là NHÃN | Đo được | Phán |
|---|---|---|---|
| 1 | dải fold **không** co | 29.7 → **21.5 pp** (co 8.2) | bác — *nhưng xem cảnh báo* |
| 2 | phần lớn 87 clip **vẫn sai** | **56/87 vẫn sai (64.4%)**; sửa 31/87 = 35.6% | giữ theo văn bản |
| 3 | lỗi vẫn tập trung ở f6/f3/f4 | tập trung **48.0% → 46.7%**; 3 fold tệ nhất đổi thành f10/f5/f4 | hỗn hợp |

**Cảnh báo huỷ giá trị của chỉ báo (1):** B1w — *cùng tháp B/32*, chỉ thêm tham số — đã co dải
xuống 21.0 pp mà chỉ được +0.2 pp BA. Vậy co dải **không phân biệt được tháp với dung lượng**.
Chỉ báo (1) được ghi trước như một phép thử sắc, và phép đo cho thấy **nó không sắc**. Ghi lại
đúng như thế thay vì dùng nó làm bằng chứng.

**Chỉ báo (3) là thứ giá trị nhất:** mức tập trung lỗi gần như bất biến (48.0 → 46.7%) nhưng
*danh tính* các fold khó đổi hoàn toàn (f6 +17.2 pp, f10 −8.7 pp). Tháp tốt hơn **di chuyển** chỗ
khó mà **không giảm mức tập trung** của nó → độ khó theo nguồn là thuộc tính của **phân bố nguồn**,
không của biểu diễn. Đây là luận cứ cho E3, độc lập với phán quyết E2.

**Chỉ báo (2):** 87 clip đó **không** phải sàn nhiễu nhãn thuần — 35.6% được sửa, trong khi mọi
mẹo rẻ trên cùng tháp chỉ cho +0.12 / −0.47 / +0.33 pp. Nhưng 64.4% còn lại vẫn là ứng viên
cho trần nhãn, nên **E1 vẫn là con số chưa đo quan trọng nhất**.

### Thăm dò (KHÔNG tiền đăng ký) — nhiễu tham số bị bác bằng thực nghiệm

Phụ lục khai rằng +32,768 tham số là lời giải thích cạnh tranh. Sổ số đã có hàng bác nó:
**B2 (VideoMAE-base, 768-d) có đúng 233,858 tham số — bằng chính xác B1 trên L/14.**
Dung lượng đầu phân loại và bề rộng đặc trưng khớp tuyệt đối; chỉ tháp đóng băng khác.

**B1@L/14 − B2 = +5.9 pp, cluster CI [+0.6, +10.8] LOẠI TRỪ 0, McNemar b=69 c=137 p=2.49e-06.**

Nên ở dung lượng khớp, lựa chọn biểu diễn đóng băng dịch BA gần 6 pp với CI chứng nhận.
Hai giới hạn phải nêu: cặp này **không** tiền đăng ký, và B2 khác L/14 ở nhiều chiều cùng lúc
(kiến trúc, dữ liệu tiền huấn luyện, video vs ảnh) nên nó **không** là một bậc thang quy mô sạch.
Nó chứng minh "tháp có ý nghĩa", **không** chứng minh "tháp lớn hơn thì tốt hơn".

### Sự cố xuất xứ đã xảy ra và đã sửa

Lượt E2 đầu **ghi đè đúng tệp OOF của bản headline**: đường dẫn hiện vật mã hoá cứng chuỗi
`isbclip_`, không phụ thuộc `--feats`. Baseline phục hồi từ git (`18903f5`) và **tái lập đúng
84.17%**; `run-esta-lso.py` nay suy tag từ tên tệp đặc trưng (đường dẫn mặc định byte-đúng như cũ),
và E2 đã chạy lại bằng mã đã sửa để hiện vật được sinh đúng chỗ, không qua khâu cứu hộ.
Tệp `.pt` softmax của bản headline **chưa từng được git theo dõi và đã mất** — nó là hiện vật dẫn
xuất, không dùng trong bất kỳ phép so sánh nào ở trên (các công cụ đọc JSON), nhưng cần tái sinh
nếu muốn sổ xuất xứ đầy đủ.
