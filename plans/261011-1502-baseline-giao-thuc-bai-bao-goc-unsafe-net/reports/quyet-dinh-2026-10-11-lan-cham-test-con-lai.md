# Lần chạm tập kiểm thử còn lại — hai phương án khác nhau ở chỗ nào

Ngày 11/10/2026. Viết vì câu hỏi "hai phương án khác nhau thế nào" chưa được trả lời rõ.
Ngân sách: **2 đăng ký / 1 đã tiêu (G21+G25 gộp) / 1 còn**. Không phán quyết gì ở đây; chỉ làm rõ.

## Khác nhau ở MỘT điều duy nhất

Cả hai phương án đều **không** cho luận văn một con số tiêu đề mới. Khác nhau ở chỗ:

- **A — chấm ngay:** biến lần chạm còn lại thành **một con số test cho đường hợp nhất**, nhưng con
  số đó mang nhãn "bộ phát hiện dưới sàn" **vĩnh viễn** và không chứng nhận được gì.
- **B — giữ lại:** luận văn ra hội đồng với **một chỗ trống được khai báo** ("chưa chấm"), và lần
  chạm còn nguyên cho một lượt chạy *có thể* phán quyết sau này.

Nói cách khác: A đổi một **chỗ trống** thành một **con số không dùng được**. B giữ chỗ trống.

## Vì sao con số của A không dùng được

| Điều kiện | Hiện trạng |
|---|---|
| Sàn cổng G22 cho bộ phát hiện | mAP50 ≥ **0.85** |
| Hai bộ phát hiện tốt nhất hiện có | **0.806** và **0.791** — dưới sàn |
| Nhãn tạo tác | `behaviour_clip.detector_gate_status()` đọc đỉnh mAP50 từ `results.csv` và **dập nhãn `gate_status` vào từng member** ⇒ bảng hạ nguồn **không thể** trình bày số dưới sàn như số đã qua cổng |
| G24 (cổng của pha 5) | là **đối chứng, không phán quyết** theo điều khoản đăng ký — kể cả số đẹp cũng không cấp phép kết luận |
| Dự báo từ val | hợp nhất **không thêm gì** trên member đơn tốt nhất: 88.05% = 88.05%; với bộ phát hiện còn lại hợp nhất **kém hơn** member đơn |
| Val có dự báo được test không | **Không.** G21 đã chứng minh val 83 clip **mù** với test: cpu hơn val +2.78 pp nhưng **mất 3 clip test**, lớp c2 rơi **30.77 pp** |

Tức là: A tiêu lần chạm cuối để lấy một con số (a) không qua cổng, (b) do một cổng không phán quyết
sinh ra, (c) mà bằng chứng val đã dự báo là **không có biên**, (d) trong khi chính val vừa bị chứng
minh là không dự báo được test. Bốn điều đó cộng lại thì giá trị kỳ vọng của A gần bằng không.

## Vì sao B không phải là "không làm gì"

B **không** làm luận văn yếu đi so với hiện tại: mục 6.15.5 đã phát biểu rõ là **không** kết luận gì
về độ lớn hệ hợp nhất trên test, hậu nghiệm `Pt` đã lưu, và việc chấm cần lần chạm còn lại **cùng
một luật quyết định viết trước khi tiêu**. Đó là một **sự thiếu vắng phép đo được khai báo**, cùng
loại với cách luận văn xử lý mọi ô "chưa đo". Hội đồng đọc được nó như một giới hạn có kiểm soát,
không phải một lỗ hổng bị che.

Cái B mua: lần chạm cuối vẫn dùng được cho lượt chạy **có** bộ phát hiện vượt sàn (pha 3, hiện
**chặn GPU** — baseline imgsz-416 đã bão hoà ở 0.806/0.791). Sau mốc bảo vệ vẫn tiêu được.

## Một điều làm phức tạp cả hai phương án

Nghịch lý ở mục 6.15.2: `beh_tight` có **mAP50 thấp hơn** (0.791 < 0.806) nhưng **clip macro cao
hơn** (88.05% > 85.91%). mAP50 ở mức hộp **không đơn điệu** với macro ở mức clip ⇒ chính sàn 0.85
có thể là **tiêu chí chọn sai đại lượng** (đã nêu ở mục 8.3.8 là cổng gác sai đại lượng). Hệ quả:
"đợi mAP50 ≥ 0.85" có thể là đợi đúng con số mà cổng đo sai. Nhưng điều đó **không** chuyển phiếu
sang A — nó nói rằng nếu đợi, nên đợi **cổng được viết lại theo đại lượng mức clip**, chứ không chỉ
đợi mAP50 lên.

## Khuyến nghị

**B — giữ lần chạm**, trừ khi tác giả muốn có số test cho đường hợp nhất **vì lý do trình bày** chứ
không vì lý do phán quyết. Nếu chọn A, phải viết trước khi tiêu: (i) luật quyết định, (ii) câu phát
biểu kèm nhãn dưới sàn sẽ in trong báo cáo, (iii) chấp nhận rằng G22 vẫn trượt và G24 vẫn không
phán quyết. Điều kiện làm tôi đổi khuyến nghị sang A: hội đồng **yêu cầu** một con số test cho
đường hợp nhất, hoặc pha 3 bị huỷ hẳn (không còn khả năng có bộ phát hiện vượt sàn) ⇒ khi đó giữ
lần chạm không còn mua được gì.

## Câu hỏi chưa giải quyết

1. Nếu viết lại cổng cho bộ phát hiện theo **macro mức clip** thay vì mAP50 thì sàn bao nhiêu, và
   đặt sàn sau khi đã thấy 85.91%/88.05% có hợp lệ không, hay bắt buộc phải là tiền đăng ký mới?
2. Pha 3 có đường nào khác ngoài CUDA (imgsz lớn hơn, bộ phát hiện khác, chú giải thêm) đưa mAP50
   lên 0.85 trong thời gian còn lại không?
3. Nếu lần chạm cuối không bao giờ tiêu, có nên ghi thành một mục riêng ở Chương 8 (ngân sách còn
   dư là một **kết quả về phương pháp**, không phải một việc chưa làm) hay giữ ở mục 6.15.5?
