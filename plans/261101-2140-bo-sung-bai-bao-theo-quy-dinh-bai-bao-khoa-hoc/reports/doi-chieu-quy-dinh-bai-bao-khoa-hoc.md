# Đối chiếu bản thảo với Quy định về bài báo khoa học (hiệu lực 15/12/2025)

Bản thảo: `plans/261031-hoi-dong-phan-bien-danh-gia-khoa-hoc/Bao_cao_luan_van_dang_bai_bao_khoa_hoc (1).docx`
Trạng thái sau khi chỉnh sửa ngày 01/11/2026. Ba mức: **Đạt** / **Thiếu** / **Chưa kiểm được**.

## Điều 6.1 — cấu trúc bắt buộc

| Mục | Yêu cầu | Trạng thái | Vị trí / ghi chú |
|---|---|---|---|
| a | Tên bài VN + EN | **Đạt** | Title (VN) + Subtitle (EN) ngay dưới |
| b | Tác giả, đơn vị, liên hệ | **Thiếu — chờ tác giả** | Khối placeholder có nhãn tại đầu bài; người biên tập không có thẩm quyền khai tên/đơn vị/email thay tác giả |
| c | Tóm tắt VN + EN | **Đạt** | "Tóm tắt" + "Abstract" |
| d | Từ khóa VN + EN | **Đạt** | 7 từ khóa mỗi ngôn ngữ, khớp nhau |
| đ | Mở đầu | **Đạt** | §1 Giới thiệu |
| e | Vật liệu / phương pháp | **Đạt** | §3 Thiết kế hệ thống, §4 Thiết kế thực nghiệm |
| g | Kết quả và thảo luận | **Đạt** | §5 Kết quả và phân tích, §6 Thảo luận |
| h | Kết luận | **Đạt** | §7 |
| i | Phụ lục (nếu có) | **Đạt** | Phụ lục A (công bố hậu-đóng-băng), Phụ lục B (dữ liệu / mã / giấy phép / tái lập) |
| k | Đóng góp khoa học | **Đạt** | Mục riêng, ba đóng góp, tự giới hạn là "định lượng và kỹ thuật triển khai" |
| l | Xung đột lợi ích | **Đạt** | Mục riêng, tuyên bố không có |
| m | Lời cảm ơn / tài trợ | **Đạt một phần** | Tuyên bố không nhận tài trợ đã viết; câu cảm ơn GVHD còn placeholder |
| n | TLTK theo chuẩn quốc tế | **Đạt** | IEEE, 16 mục; [1]–[12],[15],[16] có DOI hoặc arXiv; [13],[14] là tài liệu kỹ thuật không có DOI, đã bổ sung nhà phát hành, phiên bản, địa chỉ, ngày truy cập |

## Điều 5 — liêm chính

| Khoản | Yêu cầu | Trạng thái |
|---|---|---|
| 5.1 | Kiểm tra trùng lặp / đạo văn là **bắt buộc** | **Chưa kiểm được** — không có công cụ trong môi trường này; phải chạy trước khi nộp. Rủi ro cao hơn mức thường vì bài dùng chung kho dữ liệu và cùng mục tiêu tái lập với [1][2] |
| 5.2 | Ràng buộc liêm chính theo NĐ 262/2025 | **Đạt** — mọi số hậu-đóng-băng được dán nhãn ngày, nguồn tạo tác, và "không phải cổng tiền đăng ký" |
| 5.2 | Không công bố trùng lặp | **Chưa kiểm được** — bài là bản rút của luận văn; cần xác nhận chính sách của tạp chí đích về quan hệ luận văn ↔ bài báo |

## Điều 7.2 / 8.4.a — xác thực trích dẫn

- 14/16 mục có DOI hoặc arXiv ID → xác thực được bằng máy. **Đạt**.
- [13] ONNX, [14] NVIDIA TensorRT: tài liệu kỹ thuật, **không tồn tại DOI**. Đã nâng lên dạng tham chiếu web đầy đủ theo IEEE (tổ chức, nhan đề, nhà phát hành, phiên bản, năm, địa chỉ, ngày truy cập) và ghi rõ "không có DOI". Theo Điều 8.4.a, điều kiện tính điểm là *nằm trong danh mục, đúng chuẩn quốc tế, đủ thông tin thư mục, kiểm chứng được* — cả hai mục nay thỏa; nhưng nếu tạp chí đích yêu cầu DOI tuyệt đối thì hai mục này phải chuyển thành trích dẫn gián tiếp qua một công bố có DOI. **Chưa kiểm được** cho đến khi biết tạp chí.
- Mọi mục đều có trong danh mục TLTK và đều được dẫn trong thân bài.

## Điều 10.1.a — thang điểm

- 5 điểm cho tên bài / tóm tắt / từ khóa tiếng Anh: **đủ điều kiện** sau khi bổ sung.
- 5 điểm cho định dạng trích dẫn chuẩn: **đủ điều kiện** (IEEE nhất quán, có DOI/arXiv).

## Điều 6.2 — quy định riêng của tạp chí

**Chưa kiểm được.** Chưa biết tạp chí đích. Quy định riêng của tạp chí (thứ tự mục, giới hạn số từ, kiểu trích dẫn, mẫu tuyên bố) **ghi đè** cấu trúc chung của Điều 6.1. Thứ tự hiện tại đặt đúng theo 6.1: Kết luận → Phụ lục → Đóng góp → Xung đột lợi ích → Cảm ơn/Tài trợ → TLTK.

## Còn thiếu, không thuộc Điều 6 nhưng sẽ bị phản biện hỏi

| Hạng mục | Trạng thái |
|---|---|
| Khoảng tin cậy cho mọi dòng bảng kết quả | **Đạt** (bổ sung 01/11) |
| Dải tái lập của con số đầu đề | **Đạt** |
| Phân tách thống kê thứ hạng Bảng 3 | **Đạt** (BH, chỉ B1xl tách, về phía kém hơn) |
| Đối chứng nhiễu bối cảnh | **Đạt** (Cramér's V 0.784; McNemar p = 0.804) |
| Công bố khảo sát siêu tham số hậu nghiệm | **Đạt** (Phụ lục A.1) |
| Bền vững cấp nguồn | **Đạt** (Phụ lục A.2) |
| Dữ liệu / mã nguồn / giấy phép | **Đạt** (Phụ lục B; CC BY-NC-SA 4.0 + ShareAlike, CC BY 4.0) |
| κ cho làn nhị phân iSafetyBench | **Thiếu — chưa đo được**. Chỉ có κ = 0.65 trên làn Unsafe-Net bốn lớp. Phụ lục B nói thẳng điều này và bài **không** phát biểu gì về "gần trần nhãn". Hạn cuối gán nhãn 14/11 |
| Xuất xứ bản ghi Jetson | **Đạt một phần** — Phụ lục B khai cây làm việc có thay đổi chưa cam kết; không bảo đảm thi hành lại trùng byte |

## Câu hỏi chưa giải quyết

1. **Tạp chí đích?** Điều 6.2 cho tạp chí quyền ghi đè cấu trúc; chưa biết tạp chí thì chưa chốt được thứ tự mục, giới hạn số từ và việc [13]/[14] có được chấp nhận không DOI.
2. **Kiểm tra trùng lặp** (Điều 5.1, bắt buộc): cần chạy ở đâu, bằng công cụ gì, ngưỡng nào? Hiện chưa chạy.
3. **Tên, đơn vị, email liên hệ, ORCID** của tác giả — bản thảo đang để placeholder.
4. Có đưa dòng thứ 7 (ESTA @ YOLO26s, 81.6%, CI [74.4, 88.5]) vào Bảng 3 hay giữ ở Phụ lục A.3 như hiện nay?
5. Chính sách tạp chí về việc bài báo rút từ luận văn đã bảo vệ — có tính là công bố trùng lặp theo Điều 5.2 hay không?
6. Q7 của luận văn (2,262 so với 1,901 + 343 ảnh) vẫn mở; bài báo hiện không dùng con số nào trong nhóm này nên chưa chặn việc nộp.
