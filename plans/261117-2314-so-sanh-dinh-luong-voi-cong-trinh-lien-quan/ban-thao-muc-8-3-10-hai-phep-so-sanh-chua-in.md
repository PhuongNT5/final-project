# Bản thảo mục 8.3.10 — hai phép so sánh mục 2.7.5 từ chối in

Soạn 24/11/2026 theo quyết định của tác giả: hai mục con bị bỏ khỏi mục 2.7 (chi phí tính toán lý
thuyết, năng lượng) đi vào **Chương 8** như hướng phát triển thay vì mất hẳn.

**Chèn vào:** `report/vn/chuong-08-ket-luan.txt`, ngay sau mục 8.3.9 (dòng nội dung cuối là
456, dòng 457 để trống) và trước dòng kẻ mở đầu mục 8.4 (dòng 458). Mục 8.3 hiện có chín hạng mục và **không chỗ nào khai số
lượng hạng mục**, nên thêm một hạng mục thứ mười không làm sai một câu nào đang có.

**Thứ tự chép bắt buộc:** mục này dẫn chiếu mục 2.7.5 ngay câu đầu, nên **mục 2.7 phải vào Chương
2 trước**. Chép ngược thứ tự sẽ tạo một dẫn chiếu trỏ vào chỗ trống.

---

```
8.3.10. Hai phép so sánh mục 2.7.5 từ chối in — hạng mục mở ra từ lượt soạn 24/11/2026

Mục 2.7.5 liệt kê bảy phép so sánh định lượng mà luận văn không thực hiện. Bốn trong số đó bị từ
chối vì hai đại lượng đặt cạnh nhau không ước lượng cho cùng một thứ, và chúng sẽ còn bị từ chối
dù có thêm bao nhiêu công sức; một phép nữa — chi phí đóng băng của chính luận văn — đơn giản là
chưa đo. Hai phép còn lại, điều (5) và điều (6), mỗi phép bị chặn vì một lý do khác hẳn nhau, và
chính sự khác nhau ấy quyết định phép nào mở ra được ở lượt sau.

  (a) **Chi phí tính toán lý thuyết so với các mô hình video đã công bố — một thủ tục.** Đây là
      phép so gần-trực-tiếp nhất mà mảng này có thể có, vì số phép nhân-cộng không phụ thuộc phần
      cứng hay chuỗi công cụ. Nó không được in vì một lý do thủ tục thuần tuý: in nó là đưa vào
      báo cáo những giá trị chưa tồn tại trước mốc đóng băng số 05/10/2026. **Không một lượt chạy
      nào là cần thiết** — các giá trị đã có sẵn. Việc phải làm là một ngoại lệ hậu-đóng-băng
      được ghi nhận, hoặc một lần công bố sau khi mốc ấy hết hiệu lực. Kèm theo là một quy ước
      bắt buộc phải khai: nhiều công trình đếm phép nhân-cộng nhưng ghi nhãn là FLOPs, nên hai
      con số chỉ đặt cạnh nhau được sau khi đã nói rõ bên nào là bên nào.

  (b) **Năng lượng so với tài liệu — không phải một thủ tục.** Hạng mục này **không** mở ra được
      bằng một ngoại lệ, vì chỗ nghẽn không nằm ở phía luận văn. Các đối chứng khả dụng đo công
      suất tổng của bảng mạch, không trừ công suất nền, và đo trên một bảng mạch khác bảng mạch
      của luận văn. Chừng nào tài liệu còn ở trạng thái đó thì chỉ một phép so hạng (c) hoặc (d)
      là khả thi, và luận văn chọn không in một phép so như vậy. Thứ mở được hạng mục này là một
      đối chứng có trừ nền trên một bảng mạch so sánh được — một điều kiện nằm ngoài tầm quyết
      định của tác giả, nên đây là hạng mục **chờ tài liệu**, không phải hạng mục chờ công sức.
      Quy tắc mục 5.6.5 áp nguyên trong mọi trường hợp: không con số công suất nào được dùng để
      lập luận về ngân sách năng lượng hay để khuyến nghị một chế độ nguồn.

Phân biệt giữa (a) và (b) là điều đáng giữ, vì nhìn từ xa hai hạng mục trông như một — cùng là
"so chi phí với tài liệu". Thực ra một bên chờ một quyết định thủ tục của chính tác giả, còn bên
kia chờ một phép đo mà người khác phải công bố trước đã.
```

---

## Vì sao là một hạng mục mới chứ không phải thêm dòng vào mục 8.3.3

Mục 8.3.3 ("Các phép đo biên còn thiếu") thoạt nhìn là chỗ đúng, nhưng nó tự khai phạm vi ngay câu
mở đầu: *"Đây là danh sách **chưa đo**, không phải danh sách đã đo rồi bỏ."* Hai hạng mục ở đây
thuộc đúng loại mà câu ấy loại trừ — chúng **đã đo rồi**, cái thiếu là quyền in. Nhét chúng vào
8.3.3 sẽ làm sai chính câu mở đầu của mục ấy.

## Điều giữ cho bốn ngoại lệ vẫn là bốn

Bản thảo này **cố ý không nêu một con số nào** của hai hạng mục trên. Đó không phải phong cách mà
là ràng buộc: Chương 8 tự khai ở đầu chương rằng nó *"không giới thiệu một con số mới nào"*, và ba
giá trị liên quan (nhân-cộng mỗi ảnh, nhân-cộng mỗi quyết định, năng lượng mỗi khung) chưa từng
xuất hiện trong `report/vn/`. In bất kỳ giá trị nào trong ba giá trị ấy — ở Chương 8 hay chỗ nào
khác — **chính là ngoại lệ hậu-đóng-băng thứ năm**, và phải được ghi nhận như vậy.

Nói cách khác, mục 8.3.10 mô tả hai phép so sánh còn thiếu mà không thực hiện chúng. Đó là lý do
nó không đụng tới mốc 05/10/2026.

## Một quan sát ngoài phạm vi, không tự ý làm

Điều (4) của mục 2.7.5 — chi phí đóng băng của chính luận văn — là một dòng **chưa đo** đúng nghĩa,
nên nó là ứng viên hợp lệ cho danh sách ở mục 8.3.3, khác hẳn hai hạng mục trên. Quyết định ngày
24/11 chỉ nói về hai mục con bị bỏ, nên tôi không tự thêm dòng ấy.

## Câu hỏi chưa giải quyết

1. Tiêu đề mục dùng cách đặt của 8.3.8 và 8.3.9 ("hạng mục mở ra từ..."), nhưng hai mục ấy mở ra từ
   một **phép đo**, còn mục này mở ra từ một **lượt soạn thảo**. Cách đặt ấy có ổn không, hay bạn
   muốn một tiêu đề không gắn ngày?
2. Mục 8.3 xếp theo hệ quả đối với triển khai và giá trị khoa học, không theo độ dễ. Tôi đặt hạng
   mục này ở cuối vì nó là hạng mục yếu nhất về hệ quả khoa học. Nếu bạn thấy (a) đáng đứng trên —
   nó mở ra bằng một quyết định thủ tục, rẻ hơn mọi hạng mục khác — thì chỗ của nó là sau 8.3.4.
