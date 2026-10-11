# HIỆU ĐÍNH BÁO CÁO SAU TÁI LẬP — 2026-10-11

| | |
|---|---|
| Kích hoạt | Tác giả chỉ ra phát biểu "tái hiện đầy đủ bị giới hạn bởi chú giải hộp và tạo tác gốc không đầy đủ" **không còn đúng** sau lượt chạy 11/10 |
| Cơ chế dùng | Ngoại lệ hậu đóng băng, **đúng tiền lệ G20** (mục 6.14): báo riêng, không phán quyết, giữ nguyên mọi số tiêu đề |
| Lần chạm test mới | **0** — không lượt chạy mới nào; chỉ chuyển số đã có vào báo cáo |
| Bảng cổng | **không đổi**: hai đạt, sáu trượt, tám cổng |

## 1. Phát biểu nào sai, và sai ở đâu

Bản đóng băng gộp **hai việc khác nhau** vào một câu:

| Việc | Trạng thái thật | Bản đóng băng nói |
|---|---|---|
| Chạy lại **hệ thống** của [17] | **Không làm được** (trọng số + dự đoán theo clip không công bố) | đúng |
| Chạy lại **giao thức + phép đo** của [17] | **Đã làm 11/10** (452/64, 4 lớp, macro = TB độ nhạy) | sai — ghi "chưa bao giờ được chạy lại" |
| Tái lập **giai đoạn một** (bộ phát hiện hành vi) | **Đã làm** trên hộp tự chú giải; chưa làm trên 2,262 hộp *của họ* | nói quá — ghi "không thể tái lập" |

Gốc của lỗi: `chuong-06:1520` viết "nó chưa bao giờ được chạy lại, và chưa bao giờ được so sánh
cùng giao thức". Đúng ngày 05/10, sai từ 11/10.

## 2. Sáu chỗ đã sửa

| Tệp | Sửa gì |
|---|---|
| `00-trang-thai-bien-soan.txt` | Thêm **ngoại lệ thứ hai** 11/10/2026 + đồng bộ bảng tiến độ (9,221 dòng / 111,427 từ) |
| `chuong-06-danh-gia.txt` §6.12.5 | Bỏ mệnh đề "chưa bao giờ được chạy lại"; giữ phân biệt *hệ thống* ≠ *giao thức*; trỏ §6.15 |
| `chuong-06-danh-gia.txt` **§6.15 mới** | Mục tái lập đầy đủ: §6.15.1 G21, §6.15.2 G23, §6.15.3 G25, §6.15.4 giới hạn kết luận |
| `chuong-06-danh-gia.txt` | Kết luận chương **6.15 → 6.16** (0 tham chiếu chéo bị vỡ, đã kiểm) |
| `00-tom-tat.txt` | Thêm kết quả tái lập vào đoạn "Làn thứ hai"; giữ nguyên "không đạt 95.81%" |
| `chuong-03-phuong-phap.txt` | Thu hẹp "không thể tái lập giai đoạn một" → "không trên dữ liệu *của họ*" |
| `chuong-08-ket-luan.txt` | Hạng mục (9) thu hẹp; (11) giữ phát biểu, đổi **căn cứ**; thêm **§8.3.8** (cổng G22 gác sai đại lượng) |

## 3. Nội dung thực chất được thêm vào báo cáo

Khoảng cách tới 95.81% giờ là **phép đo**, không còn là suy luận từ bảng của họ:

- **Chân 1 rộng ra, không hẹp lại.** G21 bác bỏ giả thuyết thiết bị: 47/64 = 73.44% / 70.22% macro
  ⇒ khoảng cách 10.14 pp → **15.56 pp**. Kèm phát hiện val/test nghịch chiều (val +2.78 pp, test
  −3 clip, c2 rơi 30.77 pp) ⇒ **hàng per-class của [17] không so được bằng một lần chạy**.
- **Cơ chế họ gán công là vô hiệu.** Giảm khung cho **±0.13 pp**, McNemar p = 1.000, ba lần đo
  độc lập, hai hướng dấu — so với **+10.03 pp** họ gán. Tiêu **0** lần chạm test.
- **Nguồn thật của +8.88 pp** là thay khung thô bằng hậu nghiệm bộ phát hiện, không phải giảm
  khung. Hợp nhất **không thêm gì** trên member đơn tốt nhất (88.05% = 88.05%).
- **Nhị phân bị bối cảnh chi phối.** BA 74.19% nhưng p = 0.4049 so scene-only ⇒ G25 trượt.

Đây là **ba kết quả bất lợi** cho giả thuyết của luận văn về nguyên nhân khoảng cách, và được đưa
vào ở đúng mức chi tiết như hai cổng đạt.

## 4. Kiểm chứng

- **37/37 con số** viết vào báo cáo đối chiếu khớp `ket-qua-g21-g25.json` + `ket-qua-g23-val.json`
  (script so sánh từng giá trị, sai số ≤ 0.01 pp). Thêm val top-1 75.90% / macro 79.17% xác nhận
  trực tiếp từ `temporal_convlstm_cpu.pt`.
- **0 tham chiếu chéo vỡ**: không tệp nào trỏ "mục 6.15"/"6.16" trước khi đổi số; mọi tham chiếu
  mới (§6.15, §6.15.1-4, §8.3.8) đều có đích.
- **Không còn phát biểu cũ nào còn sống**: grep "chưa bao giờ được chạy lại / không thể chạy lại"
  chỉ còn khớp ở bản đã sửa hoặc trong chính câu trích dẫn của phần hiệu đính.
- Bảng tiến độ khớp số dòng/từ tính lại bằng script.
- Bộ test: **271 passed** (không mã nào đổi ở lượt này).

## 5. Điều hiệu đính này **không** làm

- **Không** đổi bất kỳ số tiêu đề nào, không đổi bất kỳ phán quyết cổng nào, không đổi số cổng.
- **Không** tuyên bố đạt 95.81%. Luận văn vẫn không đạt và vẫn không tuyên bố đạt.
- **Không** chấm hậu nghiệm `Pt` của hệ hợp nhất — vẫn cần lần chạm còn lại + luật quyết định
  viết trước.
- **Không** nâng bất kỳ số nào ở §6.15 lên trạng thái "đã qua cổng": tất cả dưới sàn G22 0.85.

## 6. Câu hỏi chưa giải quyết

1. **Cổng G22 có còn đúng?** mAP50 mức hộp không đơn điệu với macro mức clip (`beh_tight` kém hộp
   0.791 < 0.806 nhưng hơn clip 88.05% > 85.91%). Đã nêu thành §8.3.8, **chưa** có cổng thay thế
   được tiền đăng ký.
2. **Lần chạm còn lại tiêu cho cái gì** — hệ hợp nhất val 88.05% ngay bây giờ (số tiêu đề sẽ mang
   nhãn "bộ phát hiện dưới sàn" vĩnh viễn), hay chờ bộ phát hiện vượt sàn trên CUDA?
3. **85.78% của [17] có phải số chọn trên test?** `paper.txt:332` vẽ `val_loss` mà không khai báo
   val split. Nếu có, chân 1 là một câu hỏi sai về mặt cấu tạo.
4. Hợp nhất không thắng member đơn tốt nhất — thiếu lực trên n = 83, hay member tương quan quá
   cao? **Chưa đo** tương quan lỗi giữa member.
5. Có nên đưa §6.15 vào bản in nộp hay giữ như phụ lục hậu đóng băng? Quyết định của tác giả;
   hiệu đính này đặt nó vào thân Chương 6 theo đúng cách G20 đã được đặt.
