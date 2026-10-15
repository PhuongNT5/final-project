# Kế Hoạch: Biện Luận Lựa Chọn + Xuất Xứ Số Liệu Theo Nhận Xét GVHD

**Ngày:** 2026-10-13/14 · **Nhánh:** `next-round` · **Loại việc:** VIẾT, không ĐO.

## Mục tiêu

Trả lời 7 nhận xét GVHD (xem `nhan-xet-gvhd-2026-10-13.md`, mục Y1–Y7) bằng cách **sửa inline**
trong 11 tệp đã nộp ở `report/vn/`. Không tạo phụ lục mới. Không viết báo cáo giải trình riêng.

## Ba quyết định phạm vi (do người dùng chốt, bất biến)

1. Chỉ sửa inline trong các chương — không thêm Phụ lục D/E.
2. Không viết báo cáo giải trình riêng cho GVHD.
3. Phạm vi = toàn bộ 11 tệp đã nộp.

## Sáu ràng buộc bất khả xâm phạm

1. **Đóng băng số liệu 05/10/2026** — vòng này không sinh số mới, không đo lại.
2. Không phục hồi bất kỳ số đã rút (xem `README.md` + `00-trang-thai-bien-soan.txt`).
3. Không đổi đại lượng tiêu đề (balanced accuracy — mục 8.3.9).
4. Bảng cổng giữ nguyên: tám cổng / hai đạt / sáu không đạt.
5. **Không bao giờ bịa lý do.** Giá trị là mặc định chưa dò tìm thì phải viết đúng như vậy.
6. Không so sánh chéo hai làn (iSafetyBench ↔ Unsafe-Net).

## Các pha

| # | Pha | Tệp đích | Trạng thái |
|---|---|---|---|
| 01 | [Ch3 — biện luận thiết kế + công thức §3.7](phase-01-chuong-03-bien-luan-thiet-ke.md) | `chuong-03-phuong-phap.txt` | **xong** |
| 02 | [Ch4 — nhãn xuất xứ siêu tham số + sửa 5 sai số dòng](phase-02-chuong-04-xuat-xu-sieu-tham-so.md) | `chuong-04-chi-tiet-thuc-nghiem.txt` | **xong** |
| 03 | [Ch2 — khép mạch 4/6 lựa chọn còn THIẾU + tài liệu [44]–[49]](phase-03-chuong-02-khep-mach-tong-quan.md) | `chuong-02-tong-quan.txt` | **xong** |
| 04 | [Ch6/Ch7 — 11 công thức phép đo + đoạn tổng hợp Bảng 7.2](phase-04-chuong-06-07-cong-thuc-phep-do.md) | `chuong-06-danh-gia.txt`, `chuong-07-phan-tich-that-bai.txt` | **xong** |
| 05 | [Ch1/Ch5/Ch8/tóm tắt — vá mạch luận](phase-05-chuong-01-05-08-tom-tat-mach-luan.md) | 4 tệp còn lại + 2 phụ lục | **xong** |
| 06 | [Kiểm tra định dạng và số liệu](phase-06-kiem-tra-dinh-dang-va-so-lieu.md) | cả 11 tệp | **xong** — xem `reports/bao-cao-phu-song-y1-y7.md` |

Hai điều chỉnh so với kế hoạch ban đầu, ghi lại để không ai đi tìm việc không tồn tại:

- Pha 03 ghi "thêm tài liệu [44]–[49]". **Việc này không tồn tại.** Kiểm toán danh mục cho thấy
  kho có đúng **45** mục, mọi mục đều được trích, không mục treo, và số thứ tự không bao giờ vượt
  45 — nên cả nhóm [44]–[49] lẫn cặp trùng [48]/[15] bị nghi ngờ đều không có. Việc thật sự phải
  làm là bổ sung **sáu mục còn thiếu trong danh mục riêng của Ch3** ([11], [12], [15], [38], [43],
  [45]), chép nguyên văn từ danh mục hợp nhất của Ch2; đã làm.
- Pha 04 ghi sửa Ch6 ở nhiều chỗ. Phần lớn các chỗ đó nằm trong vùng **không được chạm**
  (§6.1–6.1.6, §6.8, §6.13–6.15, Bảng 6.9), nên việc trỏ chéo về công thức Ch3 chỉ làm được ở
  §6.12.4 và cảnh báo (a) của §6.12.1 — là đúng hai chỗ người đọc cần nó.

## Thứ tự và phụ thuộc

Ch3 → Ch4 → Ch2 → Ch6/Ch7 → Ch1/Ch5/Ch8 → kiểm tra.

- Ch3 đi trước vì nó định nghĩa khung "lý do + xuất xứ + phương án bị loại" mà các chương sau trích.
- Ch2 đi **sau** Ch3/Ch4 vì câu khép mạch ở Ch2 phải trỏ đúng tới mục đã viết ở Ch3/Ch4.
- Pha 06 chạy cuối, trên toàn bộ 11 tệp.

## Báo cáo nền

- `reports/audit-ch03-khe-ho-bien-luan.md` — 20 đoạn Ch3 đã viết tốt (giữ nguyên) + các khe hở.
- `reports/audit-ch04-xuat-xu-sieu-tham-so.md`
- `reports/audit-ch06-ch07-phep-do-cong-thuc-baseline.md`
- `reports/audit-ch01-02-05-08-mach-luan.md` — bảng 6 lựa chọn lớn: 1/6 đủ, 4/6 THIẾU.
- `reports/research-xuat-xu-hoc-thuat-tham-so-va-phuong-phap.md` — tiền lệ công bố cho 10 mục.
- `reports/kiem-chung-cau-hinh-huan-luyen-ch04.md` — kiểm chứng độc lập; **bác bỏ** đề nghị
  "công bố patience=30" của audit Ch4 (không tồn tại trong mã).

## Tiêu chí hoàn thành

- Mỗi giá trị thiết kế/siêu tham số được nêu trong 11 tệp có: lý do, xuất xứ (một trong năm nhãn),
  và nếu có — phương án bị loại.
- 11 đại lượng thống kê có công thức viết ra.
- Dòng ≤ 100 ký tự (trừ đường dẫn và hàng bảng rộng); không dòng nào bắt đầu bằng `^[0-9]\.[0-9]`.
- Không số mới, không số đã rút, bảng cổng không đổi.
