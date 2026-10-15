# Nhận xét GVHD — 13/10/2026 (nguyên văn)

Ghi lại nguyên văn, không diễn giải, để mọi việc sửa sau này đối chiếu được.

> - Báo cáo chỉ là đang liệt kê chứ không phải một bài luận văn nghiên cứu
> - Chưa biện luận giải thích từng loại phương pháp, mô hình sử dụng, tham số, phép đo, baseline nào.
> - Phải đưa lý do lựa chọn sử dụng thử nghiệm, hay đưa vào báo cáo.
> - Tại sao chọn model với layer này, lớp kia, tham số đầu vào, ...
> - Giải thích lý do chọ, kết quả, ưu, nhược điểm của từng loại phương pháp, gía trị, biến, tham số, lớp đưa vào thử nghiệm
> - Các số liệu và phương pháp là tự nghĩ ra, tự đo đạt được hay từ một nghiên cứu nào?
> - Phép đo, công thức, baseline so sánh

## Phân rã thành bảy yêu cầu kiểm được

| # | Yêu cầu | Dạng bằng chứng phải có trong luận văn |
|---|---------|----------------------------------------|
| Y1 | Văn phong luận văn nghiên cứu, không phải danh mục | mỗi mục có mạch: câu hỏi → giả thuyết → vì sao cách này trả lời được → bằng chứng → giới hạn |
| Y2 | Biện luận từng phương pháp / mô hình / tham số / phép đo / baseline | một đoạn lý do cho từng mục, không chỉ một hàng bảng |
| Y3 | Lý do chọn phải nằm TRONG báo cáo | không được chỉ nằm ở plans/, log hay commit |
| Y4 | Vì sao chọn kiến trúc này, số lớp này, tham số đầu vào này | lý do cho d_model, số lớp, số đầu chú ý, FFN, dropout, T, cửa sổ/bước nhảy |
| Y5 | Ưu VÀ nhược điểm của từng phương pháp, giá trị, biến, tham số | bảng ưu/nhược cho từng mô hình và từng lựa chọn tham số, kèm phương án bị loại |
| Y6 | Xuất xứ: tự nghĩ ra / tự đo / lấy từ nghiên cứu nào | ba nhãn xuất xứ cho mọi giá trị và mọi phương pháp, có trích dẫn hoặc có tệp đo |
| Y7 | Phép đo, công thức, baseline so sánh | công thức toán viết ra cho mọi đại lượng; mỗi con số kèm baseline + nguồn baseline |

## Ràng buộc không được vi phạm khi sửa

1. ĐÓNG BĂNG SỐ (05/10/2026, đã có ba ngoại lệ ghi nhận). Việc sửa lần này là **việc viết**,
   không phải việc đo. Không thêm con số mới, không chạy lại để "đẹp hơn".
2. Không phục hồi các con số đã rút lại (95.7% mAP, +3.5 pp macro, 57/64, phán quyết P3 một chiều).
3. Không đổi đại lượng tiêu đề (độ chính xác cân bằng) — mục 8.3.9.
4. Bảng cổng giữ nguyên: tám cổng, hai đạt, sáu trượt.
5. Không được bịa lý do. Nếu một giá trị là mặc định không dò tìm thì phải viết đúng như vậy;
   câu trả lời trung thực "đây là mặc định, không dò tìm, đây là rủi ro kèm theo" là một câu
   trả lời hợp lệ cho GVHD, còn một lý do bịa ra thì không.
6. Cấm so sánh chéo hai làn (iSafetyBench nhị phân 1,100 clip vs Unsafe-Net 4 lớp 64 clip).
