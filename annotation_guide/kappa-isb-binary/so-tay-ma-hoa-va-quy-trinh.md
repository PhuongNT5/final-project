# Sổ tay mã hoá + quy trình gắn nhãn — nhãn nhị phân iSafetyBench

Mục đích: gắn nhãn **độc lập** 151 clip để đo **trần nhãn** của iSafetyBench nhị phân.
Chuẩn vàng là **phân vùng gốc của bộ dữ liệu** (`hazard/`→1, `normal/`→0), không phải nhãn do
luận văn suy ra. Vì vậy câu hỏi thực sự là: *"người gán nhãn khác có tái tạo được nhãn của
iSafetyBench không?"*

**ĐỌC HẾT TỆP NÀY TRƯỚC KHI MỞ CLIP ĐẦU TIÊN.** Sau sự kiện gắn nhãn đầu tiên, tệp này **không
được sửa**; mọi thay đổi phải là phụ lục có ngày ở cuối, không phải sửa tại chỗ.

**KHÔNG MỞ `key.json`.** Nó chứa bản đồ K-id → nhãn gốc. Nó bị `.gitignore` loại có chủ ý.
Cũng không mở `data/isafetybench/` bằng Finder trong lúc gắn nhãn: tên thư mục **là** nhãn.

---

## 1. Định nghĩa vận hành

**NGUY HIỂM (nhấn `j`)** — clip cho thấy một **sự cố** gây hại, đang xảy ra hoặc sắp xảy ra rõ
ràng, với người, thiết bị hoặc kết cấu: té/rơi, va chạm, kẹp/nghiền, cuốn vào máy, điện giật,
cháy/nổ, rò rỉ hoá chất, sập kết cấu, rơi tải, xung đột xe–người đi bộ, bị vật đập vào.

**BÌNH THƯỜNG (nhấn `f`)** — mọi trường hợp còn lại, **kể cả công việc trông rất nguy hiểm nhưng
không có sự cố nào xảy ra**.

## 2. Mười luật quyết định — áp dụng theo thứ tự, không ứng biến

| # | Luật | Vì sao có luật này |
|---|---|---|
| **R1** | Phán xét **SỰ CỐ**, không phán xét **MỨC RỦI RO**. Làm việc thiếu an toàn mà không có sự cố ⇒ **bình thường** | nguồn bất đồng lớn nhất; phải chốt trước |
| **R2** | **Cận kề tính là có**: nếu hậu quả đang hiển thị sắp xảy ra trong clip (vật đã rơi, xe đã trên đường va) ⇒ **nguy hiểm**, dù cú đập nằm ngoài khung hoặc sau cắt | nhiều clip bị cắt ngay trước va chạm |
| **R3** | **Mất kiểm soát** (rơi tải, trượt chân, lạng xe) ⇒ nguy hiểm. **Suýt va mà không mất kiểm soát, không tiếp xúc** ⇒ bình thường | tách "suýt" khỏi "đã" |
| **R4** | **Thiếu bảo hộ (PPE) một mình** ⇒ **bình thường**. Đó là vi phạm, không phải sự cố | luật này sẽ bị bản năng chống lại; cứ tuân |
| **R5** | Sự cố **chỉ suy ra**, không có bằng chứng hình ảnh ⇒ **bình thường** | chống việc tự kể chuyện |
| **R6** | Bối cảnh **không phải công nghiệp** (đường phố, nhà, thể thao) ⇒ vẫn dùng đúng luật sự cố. **Không lấy bối cảnh làm căn cứ** | bối cảnh tương quan với nguồn, mà nguồn tương quan hoàn hảo với nhãn ⇒ dùng bối cảnh là gian lận với chính mình |
| **R7** | Phim **dàn dựng / huấn luyện / tái hiện** ⇒ phán xét **sự cố được mô tả** | bộ dữ liệu lấy từ YouTube, có cả loại này |
| **R8** | **Nhiều sự kiện** ⇒ nguy hiểm nếu **có bất kỳ** sự cố đủ điều kiện | nhãn nhị phân là phép gộp "có ít nhất một" của nhãn đa lớp gốc |
| **R9** | **Chỉ thấy hậu quả** (đổ vỡ, không thấy sự kiện): nguy hiểm **nếu** hiện trường cho thấy sự cố vừa xảy ra trong cảnh đó; hư hỏng có sẵn, không liên quan ⇒ bình thường. **Đặt độ tin cậy 1** | ca khó nhất, phải đánh dấu được để cắt riêng khi phân tích |
| **R10** | **Lưỡng lự ⇒ không được bỏ qua.** Chọn phía nhiều khả năng hơn, đặt **độ tin cậy 1** | κ cần nhãn nhị phân cưỡng bức; "không chắc" sẽ phá phép đo |

## 3. Quy trình mỗi phiên

1. **Hiệu chuẩn, chỉ trước phiên đầu tiên:** gắn nhãn thử **10 clip Unsafe-Net** ở `data/train/`
   (tám lớp, tên thư mục có `safe`/`unsafe`) bằng đúng R1–R10, rồi tự đối chiếu. Mục đích là luyện
   áp dụng luật, **không** sinh ra số nào được báo cáo. Unsafe-Net là **làn khác** — không bao giờ
   đặt cạnh số iSafetyBench. Dùng Unsafe-Net chứ không dùng clip iSafetyBench khác vì **cả 151
   nguồn đều đã nằm trong nghiên cứu**: tập hiệu chuẩn lấy từ chính kho sẽ làm lộ nhãn của nguồn đó.
2. Chạy `.venv/bin/python tools/kappa-annotation-server.py`, mở `http://127.0.0.1:8765`.
3. **Xem hết clip ít nhất một lượt** trước khi quyết. `space` xem lại; số lần xem lại được ghi.
4. Chọn **độ tin cậy 1/2/3** (bắt buộc), bật `r` nếu **nhận ra clip từ trước**.
5. `f` = bình thường · `j` = nguy hiểm.
6. **Tối đa 60 sự kiện một phiên**, nghỉ ≥ 30 phút giữa hai phiên, không gắn nhãn khi mệt.
   Có thể dừng bất cứ lúc nào: tiến độ lưu theo từng dòng, mở lại là chạy tiếp.
7. **Sau mỗi phiên: `git add annotation_guide/kappa-isb-binary/annotations.jsonl && git commit`.**
   Commit theo phiên là cách duy nhất khiến dấu thời gian của dữ liệu **độc lập kiểm chứng được** —
   chính là lỗ hổng mà hội đồng 31/10 phát hiện ở các hiến chương cổng trước đây.

## 4. Nút `r` — "tôi nhận ra clip này" — dùng cho đúng

Tác giả đã sống với kho này nhiều tháng, nên **không thể giả vờ là người xa lạ**. Thay vì phủ nhận,
nghiên cứu này **đo** nó: bật `r` **bất cứ khi nào** bạn nhớ đã thấy clip, hoặc nhớ nhãn của nó,
hoặc nhận ra nguồn. Phân tích báo cáo κ **tách riêng** nhóm nhận ra và nhóm không nhận ra.

Trung thực ở đây tốn 0 đồng và là thứ duy nhất khiến con số này đáng tin. Bật `r` quá nhiều làm
nghiên cứu yếu đi một cách công khai; bật thiếu làm nó **sai một cách thầm lặng**.

## 5. Điều không được làm trong lúc gắn nhãn

1. Không mở `key.json`, không mở `data/isafetybench/`, không mở OOF, không mở biểu đồ lỗi.
2. Không đoán "clip này chắc thuộc nhóm hazard vì tôi vừa gắn mấy clip bình thường liên tiếp".
   Giao diện **cố tình không hiện số đếm theo nhãn** để chặn việc cân bằng theo kỳ vọng.
3. Không quay lại sửa nhãn đã gửi vì "nghĩ lại". Nếu buộc phải sửa, gắn lại và **giải thích trong
   commit**; dòng cũ vẫn nằm trên đĩa và vẫn là phần của hồ sơ.
4. Không đọc `tools/analyse-kappa.py` để suy ra kết quả sẽ ra sao. (Script đã được commit **trước**
   khi có dữ liệu — đó là điểm mạnh của thiết kế, đừng biến nó thành chỗ rò rỉ.)

---

## Phụ lục sửa đổi

*(Chưa có. Mọi sửa đổi sau sự kiện gắn nhãn đầu tiên phải ghi tại đây, có ngày, không sửa phần trên.)*
