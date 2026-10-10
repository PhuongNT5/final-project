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
