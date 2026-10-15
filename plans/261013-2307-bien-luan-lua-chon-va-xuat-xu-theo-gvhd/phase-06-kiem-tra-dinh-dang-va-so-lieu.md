# Pha 06 — Kiểm Tra Định Dạng Và Số Liệu Trên Toàn Bộ 11 Tệp

**Tệp đích:** cả 11 tệp trong `report/vn/`
**Ràng buộc nguồn:** `report/vn/00-trang-thai-bien-soan.txt` (15 ràng buộc viết), `README.md`
**Ưu tiên:** BẮT BUỘC · **Phụ thuộc:** chạy sau Pha 01–05
**Trạng thái:** chưa bắt đầu

## Kiểm định dạng (tự động)

1. **Dòng ≤ 100 ký tự** — miễn trừ: đường dẫn không ngắt được, hàng bảng rộng.
   `awk 'length>100 {print FILENAME":"FNR": "length}' report/vn/*.txt`
2. **Không dòng nào bắt đầu bằng `^[0-9]\.[0-9]`**
   `grep -nE '^[0-9]\.[0-9]' report/vn/*.txt`  → phải rỗng.
3. Mọi số dòng mã được trích dẫn khớp mã hiện tại — kiểm lại bằng `grep -n`.

## Kiểm nội dung (đọc)

4. **Không số mới.** So với bản trước khi sửa, tập các con số phải không đổi.
   Cách kiểm: `git diff` rồi soát mọi chữ số trong phần thêm vào; mỗi chữ số mới phải là
   (a) số dòng mã, (b) số công thức/ký hiệu, hoặc (c) số đã có sẵn ở chỗ khác trong luận văn.
5. **Không số đã rút** — grep các chuỗi cấm: `3.5 pp`, `57/64`, `95.7`, `331 ms`, `3.02`, `3.91 J`.
6. **Bảng cổng không đổi** — vẫn tám cổng, hai đạt, sáu không đạt. G20 và G21/G25/G23/NP-125-D
   vẫn là đối chứng **không phán quyết**.
7. **Không trộn làn** — không câu nào đặt số iSafetyBench cạnh số Unsafe-Net như một so sánh.
8. **Không trộn pipeline** — P1 (116 fps), P3 (quyết định/giây), B1 không dùng chung đơn vị.
9. **`val_macro` 89.15% / 90.92%** vẫn được mô tả là chỉ số chọn mô hình, không phải độ chính xác.
10. **G11** vẫn được báo cáo trước E0.
11. **R47** vẫn giới hạn phạm vi ở Unsafe-Net.
12. Caveat Bảng 5.4 (`multi-stream-results.json` tự khai là vá tay) vẫn còn.
13. Ba ngoại lệ sau đóng băng (G20 ngày 10/10; G21/G25/G23 ngày 11/10; đọc lại NP-125-D ngày
    11/10) vẫn được ghi nhận đúng ở `00-trang-thai-bien-soan.txt`; nếu Pha 01–05 làm phát sinh
    ngoại lệ mới thì **dừng lại** — vòng này không được sinh ngoại lệ mới.

## Kiểm đối chiếu nhận xét GVHD

14. Lập bảng Y1–Y7 → mục nào trong 11 tệp trả lời. Mỗi dòng phải có ít nhất một mục cụ thể.

## Todo

- [ ] Chạy ba kiểm định dạng
- [ ] Soát `git diff` tìm chữ số mới
- [ ] Grep chuỗi số đã rút
- [ ] Soát 8 kiểm nội dung còn lại
- [ ] Lập bảng đối chiếu Y1–Y7

## Tiêu chí hoàn thành

Ba lệnh kiểm định dạng chạy sạch, `git diff` không chứa chữ số nào không giải thích được, và bảng
đối chiếu Y1–Y7 không có dòng trống.
