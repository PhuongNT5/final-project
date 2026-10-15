# Pha 05 — Ch1 / Ch5 / Ch8 / Tóm Tắt / Phụ Lục A–B: Vá Mạch Luận

**Tệp đích:** `00-tom-tat.txt` (229), `chuong-01-mo-dau.txt` (440),
`chuong-05-trien-khai-bien.txt` (1339), `chuong-08-ket-luan.txt` (516),
`phu-luc-a-bang-nhat-ky.txt` (529), `phu-luc-b-cau-hinh-fold.txt` (435)
**Báo cáo nền:** `reports/audit-ch01-02-05-08-mach-luan.md`
**Ưu tiên:** TRUNG BÌNH · **Trạng thái:** chưa bắt đầu

## Nhận xét GVHD được giải quyết

Y1 (liệt kê thay vì luận văn), Y3 (phải đưa lý do lựa chọn thử nghiệm vào báo cáo).

## Ba chỗ đứt mạch phải vá

1. **Ch5** — 141 dòng cảnh báo mở đầu chạy trước khi câu hỏi của chương xuất hiện ở dòng 158.
   Người đọc gặp một chuỗi cảnh báo trước khi biết chương này hỏi gì. Vá: đưa câu hỏi của chương
   lên trước, giữ nguyên nội dung cảnh báo nhưng đặt sau khi đã nêu chương này định trả lời gì.
   **Lưu ý tệp:** chỉ sửa `chuong-05-trien-khai-bien.txt`.
   `chuong-05-trien-khai-bien-ghim-xung-nhip.txt` là bản lưu trữ, **không nộp, không gộp**.
2. **Ch8** — rủi ro G17 xuất hiện ở kết luận mà không được báo trước ở chương trên. Vá: gieo trước
   ở nơi G17 phát sinh (Ch6 hoặc Ch7), để Ch8 chỉ kết lại chứ không giới thiệu rủi ro mới.
3. **Ch1** — thiếu một câu phát biểu giả thuyết tường minh. Vá: thêm phát biểu giả thuyết của luận
   văn (lạm phát giao thức D1), đặt trước phần đóng góp.

## Ràng buộc khi viết

- Đại lượng tiêu đề vẫn là balanced accuracy (mục 8.3.9) — không đổi.
- Kết luận đầu bài vẫn là **lạm phát giao thức (D1)**, không phải một con số độ chính xác.
- "lạm phát giao thức" là thuật ngữ **do luận văn đặt** — ở chỗ nêu lần đầu phải nói rõ như vậy,
  kèm các tiền lệ gần [46][47].
- Đơn vị biên là **quyết định/giây** (không phải FPS), **J/quyết định**, vòng lặp hợp thành;
  cấm trình bày tổng theo chặng như một kết luận. Nguồn trích dẫn duy nhất:
  `p3-composed-cadence-fp16-*.json`.
- Không khôi phục bất kỳ số đã rút ("+3.5 pp", "57/64", "95.7% mAP", "331 ms / 3.02 dec/s / 3.91 J",
  mọi phán quyết đạt/không đạt đơn lẻ cho hợp đồng 3.0 dec/s của P3).

## Phụ lục A và B

Soát nhẹ: bảo đảm nhãn xuất xứ và cách gọi tên thống nhất với Ch3/Ch4 sau khi sửa
("leave-context-out" viết đủ chữ trong văn xuôi, giữ viết tắt LCO; "nhóm bối cảnh" chứ không
"camera" cho Unsafe-Net). Không thêm số liệu mới.

## Todo

- [ ] Ch1 — câu phát biểu giả thuyết tường minh
- [ ] Ch5 — đưa câu hỏi của chương lên trước khối cảnh báo
- [ ] Gieo trước rủi ro G17 ở Ch6 hoặc Ch7
- [ ] Tóm tắt — soát khớp với các sửa đổi ở Ch1/Ch8
- [ ] Phụ lục A, B — soát thống nhất thuật ngữ

## Tiêu chí hoàn thành

Đọc tuyến tính từ tóm tắt tới Ch8 không gặp khái niệm hay rủi ro nào xuất hiện lần đầu ở chỗ nó
được kết luận.
