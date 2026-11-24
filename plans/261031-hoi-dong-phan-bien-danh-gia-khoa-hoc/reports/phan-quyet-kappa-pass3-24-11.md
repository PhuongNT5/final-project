# Phán quyết pass 3 — D3 nổ, nghiên cứu κ được tuyên là không diễn giải được

**Ngày:** 24/11/2026 · **Dữ liệu:** commit `ea3e412` · **Tiền-đăng-ký:** Sửa đổi 3, commit
`0e750e9` · **Script:** `fc0df3b`/`019b061`, cả hai commit **trước** khi có dòng dữ liệu pass 3 ·
**Kết quả:** `ket-qua-kappa-pass3.txt`

## Kết quả dẫn đầu

Test-retest κ sau 7 ngày trên 72 clip là **0,5970**, CI95 [0,3755, 0,7910]. Ngưỡng D3 là 0,80.
Theo tiền-đăng-ký §5, nguyên văn: *"κ nội-người < 0.80 ⇒ **Toàn bộ phép đo tuyên bố KHÔNG DIỄN
GIẢI ĐƯỢC.** Báo cáo đúng như vậy, không kết luận về phía nào."*

**D3 nổ. Nghiên cứu κ không còn chống đỡ được lập luận "84.2 % sát trần nhãn".** Luật này do tôi
chốt ngày 01/11 và tái khẳng định trong Sửa đổi 3 ngày 24/11, trước khi thấy bất kỳ số nào của
pass 3. Nó có hiệu lực.

## Vì sao phán quyết này không thể lách

Có ba đường lách khả dĩ, và cả ba đều đã được kiểm và đều thất bại.

**Thứ nhất, nghịch lý κ.** Tiền-đăng-ký §4 đã chọn Gwet AC1 làm đại lượng đồng-chính với lý do ghi
sẵn rằng κ nhạy với tỉ lệ hiện mắc. Nghịch lý xuất hiện đúng như dự đoán: ở nhóm thử nghiệm, trùng
khớp thô là 75,0 % nhưng κ = −0,1408 trong khi AC1 = +0,6800. Tuy nhiên **AC1 trên cả 72 clip là
0,7160, vẫn dưới 0,80**. Phán quyết vững dưới cả hai thước đo.

**Thứ hai, lệch tầng.** Tập 72 clip được tôi dựng có chủ đích gồm 50 % clip tự tin thấp, trong khi
corpus n = 150 chỉ có 24 %, tức lệch 2,1 lần về phía tầng không đáng tin. Trùng khớp thô tái cân
theo tỉ lệ thật là 87,7 % so với 83,3 % trên 72 clip, nên 0,5970 **là ước lượng lệch thấp** cho
corpus đầy đủ. Đây là **lỗi thiết kế của tôi** trong Sửa đổi 3: tôi đã chỉ định một con số tính trên
mẫu cố ý lệch tầng làm "con số độ tin cậy chính thức", trong khi nhóm đối chứng được thêm vào để
phân biệt E1 với E1b, không phải để gộp vào một ước lượng độ tin cậy. Nhưng luật đã chốt trước trên
"cả 72 clip", nên lỗi của tôi **không cho phép** thay con số sau khi đã thấy nó.

**Thứ ba, phép can thiệp.** E0 thất bại: dwell trung vị nhóm thử nghiệm là **8,4 s**, thấp hơn chính
pass 1 (11,2 s) và xa ngưỡng 20 s. Toàn bộ 72 clip được gắn trong **10,4 phút**. Việc bỏ áp lực thời
gian — mục đích duy nhất của pass 3 — đã không xảy ra, nên E1, E1b và E2 đều không áp dụng. Pass 3
chỉ còn giá trị như một test-retest, và nó không đạt.

## Điều nghiêm trọng hơn cả D3

Hai tầng tách nhau rất xa, và phần đáng tin lại chỉ ngược hướng với kết luận D2 của tuần trước.

| Tầng | Tỉ lệ corpus | Test-retest 7 ngày | So với nhãn gốc (pass 1) |
|---|---|---|---|
| Tự tin cao | 114/150 (76 %) | κ **0,8333** · AC1 0,8345 | đúng **92,1 %** · κ 0,8418 |
| Tự tin thấp | 36/150 (24 %) | κ **−0,1408** · AC1 0,6800 | đúng **61,1 %** · κ 0,1486 |

Trên 76 % corpus mà người đọc **tái lập được chính mình**, độ chính xác so với nhãn iSafetyBench là
**92,1 %**, trong khi mô hình đạt **85,33 %** trên cả 150 clip. Nghĩa là trên phần corpus đáng tin,
**người hơn mô hình khoảng 6,8 pp**. Con số 84,67 % từng chống đỡ D2 bị kéo xuống bởi 36 clip mà
người đọc mặc định trả lời "bình thường" và **không tái lập được chính mình** — test-retest κ âm.

Hướng này trùng hướng của **D1**, tức hướng bất lợi cho luận văn. Tôi **không** chạy lại D1 trên
nhóm con, vì D1 đã chốt là đánh giá trên toàn bộ n = 150 và đã không nổ. Nhưng phải nói thẳng rằng
phân tích theo tầng chỉ **ngược** với kết luận D2, không ủng hộ nó.

## Hệ quả buộc phải viết vào luận văn

Lập luận "84.2 % sát trần nhãn" **mất phần chống đỡ bằng số đo**. §6 không được dẫn κ = 0,6933 hay
mốc người 84,67 % làm bằng chứng nữa, vì D3 tuyên toàn bộ phép đo không diễn giải được. Lập luận có
thể được giữ như một **giả thuyết chưa kiểm chứng**, không phải một kết quả.

Nghiên cứu nên ở lại phụ lục dưới dạng **kết quả phương pháp luận âm**: một nghiên cứu trần nhãn có
tiền đăng ký, ba phụ lục sửa đổi có ngày, và một luật dừng đã nổ đúng như đã cam kết. Phát hiện phụ
dùng được là: **gán nhãn nhị phân hành vi không an toàn bởi một người đọc trên corpus này không tái
lập được ở một phần tư clip nhập nhằng**, và chính độ tự tin tự khai của người đọc chỉ ra được phần
đó trước khi biết đáp án.

## Không đổi

Bảng cổng **không đổi: tám cổng, hai đạt, sáu trượt**. Chạm tập kiểm thử của nghiên cứu này vẫn là
**0 lần**, nên ngân sách còn **1 lần** như trước. Kết quả chính của luận văn, 84,2 % BA với CI
[77,4, 90,0] và dải tái lập 83,25–84,71 %, **không bị ảnh hưởng** — pass 3 không chạm vào chúng.

**Phát ngôn bị cấm, bổ sung:** "κ cho thấy mô hình sát trần nhãn", "độ tin cậy của người gán nhãn
đã được xác nhận", và mọi cách dẫn κ = 0,6933 như một kết quả đang có hiệu lực.

## Câu hỏi chưa giải quyết

1. Có thử pass 4 với **cưỡng chế thời gian trong công cụ** (khoá nút trả lời cho đến khi đủ N giây)
   không? Hai lần liên tiếp phép can thiệp đã thất bại bằng hướng dẫn, nên nếu làm thì phải cưỡng
   chế. Chi phí: sửa công cụ khoảng 15 phút, gắn nhãn tối thiểu 24 phút chờ cưỡng chế cho 72 clip,
   và rủi ro mới là bấm cho qua trong lúc chờ. Còn **23 ngày** đến đóng băng 17/12.
2. Khuyến nghị của tôi là **dừng thu thập** và báo cáo kết quả âm, vì một luật dừng đã nổ được báo
   cáo trung thực thì an toàn hơn ở buổi bảo vệ so với một con số có thể bị chất vấn. Tác giả quyết.
3. §6 viết lại theo hướng "giả thuyết chưa kiểm chứng" — ai viết, khi nào?
4. Còn treo từ trước: đoạn khoảng trống nghiên cứu, Tóm tắt 269 → ≤250 từ, 7 → 5 từ khoá, và câu
   sai ở `chuong-04:1279` ("cả hai lần chạm vẫn còn nguyên").
