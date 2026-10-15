# Pha 04 — Ch6/Ch7: Công Thức Phép Đo + Đoạn Tổng Hợp Bảng 7.2

**Tệp đích:** `report/vn/chuong-06-danh-gia.txt` (2776 dòng), `chuong-07-phan-tich-that-bai.txt` (1023)
**Báo cáo nền:** `reports/audit-ch06-ch07-phep-do-cong-thuc-baseline.md`
**Ưu tiên:** CAO · **Trạng thái:** chưa bắt đầu

## Nhận xét GVHD được giải quyết

Y7 ("phép đo, công thức, baseline so sánh") — Ch6 dùng 11 đại lượng thống kê mà **không viết công
thức nào**. Phần baseline thì **đã tốt** (Bảng 6.1b và 6.9 đã khai metric/protocol/giá trị/nguồn
cho từng con số) — không chạm.

## Mười một công thức phải viết

Đặt vào một mục định nghĩa ngắn trong Ch6 (hoặc ngay nơi đại lượng xuất hiện lần đầu), mỗi công
thức kèm một câu nói **đại lượng này trả lời câu hỏi gì**:

1. macro-F1
2. PR-AUC (diện tích dưới đường precision–recall)
3. ROC-AUC
4. mAP / AP / IoU (ba cái liên quan, gom một chỗ)
5. ECE (sai số hiệu chuẩn kỳ vọng)
6. Cohen's κ
7. MCC
8. Brier score
9. Thống kê McNemar (bản chính xác, nhị thức)
10. Hiệu chỉnh Nadeau–Bengio
11. Khoảng tin cậy Wilson

Balanced accuracy và bootstrap theo cụm đã thuộc Pha 01 (§3.7); ở Ch6 chỉ trỏ ngược về Ch3,
**không lặp lại công thức**.

## Đoạn tổng hợp sau Bảng 7.2

Ch7 dòng 607–714 (Bảng 7.2) kết thúc bằng bảng, không có đoạn rút ra ý nghĩa — đúng kiểu "liệt kê"
mà GVHD phê. Viết một đoạn tổng hợp theo **đúng khuôn của Bảng 7.3** (đoạn sau Bảng 7.3 đã làm tốt).

## Việc KHÔNG được làm

- Không chạm §6.1–6.1.6, §6.8, §6.13, §6.14, §6.15, Bảng 6.9, §7.4, §7.8 (audit đánh giá tốt).
- Không sinh số mới. Viết công thức **không** kéo theo tính lại giá trị nào.
- Không đổi bảng cổng (tám cổng / hai đạt / sáu không đạt).
- Khoảng tin cậy mức clip chỉ được in kèm nhãn `WRONG_UNIT`; đơn vị lấy mẫu lại là **nhóm**.
- Khi bootstrap theo cụm và McNemar mâu thuẫn, **bootstrap theo cụm thắng** — giữ nguyên quy tắc.

## Todo

- [ ] Viết 11 công thức + một câu diễn giải mỗi công thức
- [ ] Trỏ ngược BA và bootstrap theo cụm về Ch3 §3.7
- [ ] Đoạn tổng hợp sau Bảng 7.2 (theo khuôn Bảng 7.3)
- [ ] Soát: mọi ký hiệu dùng trong công thức đều được định nghĩa trước khi dùng

## Tiêu chí hoàn thành

Không đại lượng thống kê nào trong Ch6/Ch7 xuất hiện mà không có công thức ở đâu đó trong luận văn,
và Bảng 7.2 không còn kết thúc bằng bảng trần.
