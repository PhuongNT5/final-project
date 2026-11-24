# Bản thảo mục 2.7 — So sánh định lượng với tài liệu

Loại: bản thảo chương · Ngày: 2026-11-24 · Nguồn số: `reports/synthesis-261117-so-sanh-dinh-luong-da-truc.md`

**Trạng thái:** bản thảo nằm ngoài `report/vn/`. Chương đang đóng băng và quyền sửa thuộc về tác
giả, nên tệp này **chỉ để chép vào**, không phải một sửa đổi đã thực hiện.

**Vị trí:** mục **2.7**, sau mục 2.6 và trước TÀI LIỆU THAM KHẢO của Chương 2. Thuần cộng thêm,
không đánh số lại mục nào.

**Phạm vi — đã chốt 24/11: lõi cộng thang chi phí thích nghi.** Năm mục con. Hai mục con từng có
trong bản thảo trước đã **bỏ** vì chúng là hai mục duy nhất cần số chưa có trong báo cáo: mục tính
toán lý thuyết (4.409 GMACs mỗi ảnh, 70.5 GMACs mỗi quyết định) và mục năng lượng (30.07 mJ mỗi
khung). Mục bảng thực hành báo cáo cũng bỏ, vì mệnh đề "không bài nào trong 12 bài" buộc phải liệt
kê cả 12 bài trong danh mục tham khảo.

**Hệ quả quan trọng: bản này KHÔNG cần ngoại lệ hậu-đóng-băng.** Mọi con số trong năm mục con còn
lại đều đã công bố — trong `report/vn/` hoặc trong tài liệu được trích. Không số đo mới nào được
thêm, nên mốc đóng băng 05/10 không bị đụng tới, và bốn ngoại lệ đã ghi vẫn là bốn.

**Phạm vi đã thu hẹp sau khi đọc lại Chương 2.** Chương đã có ba chỗ mà bản thảo đầu viết lại: mục
2.2.3 có bảng B1w so với B2 với đúng các con số ấy, mục 2.2.5 có số của mảng giám sát yếu kèm ba lý
do từ chối, mục 2.2.6 có kết quả dung lượng và điểm 5.1 so với 1.4 điểm phần trăm. Cả ba đã thành
dẫn chiếu. Mục 2.7 chỉ chứa những trục Chương 2 hiện **chưa** có con số nào.

---

```
2.7. SO SÁNH ĐỊNH LƯỢNG VỚI TÀI LIỆU
----------------------------------------------------------------------

Mục 2.6 định vị luận văn bằng lời. Mục này định vị bằng số, và phải mở đầu bằng một khó khăn: phần
lớn các con số được công bố trong hai mảng lân cận **không so sánh trực tiếp được** với các con số
của luận văn. Chúng đo trên kho khác, không gian nhãn khác, giao thức chia khác, phần cứng khác, và
thường không nêu giao thức chia của chúng. Một bảng đặt cạnh nhau mà không nói rõ điều đó sẽ tạo ra
một thứ hạng không có nghĩa.

Giải pháp của luận văn là **không bỏ phép so sánh mà gắn nhãn cho nó**. Mỗi bảng trong mục này nói
rõ các hàng của nó so sánh được tới mức nào, và mục 2.7.5 liệt kê các phép so sánh mà luận văn **từ
chối thực hiện**, kèm lý do kỹ thuật của từng phép.

Mục này không lặp lại các phép so sánh đã có chỗ riêng: hiệu số giữa xương sống hình ảnh đóng băng
và xương sống video đóng băng nằm ở mục 2.2.3; vị thế so với mảng giám sát yếu nằm ở mục 2.2.5; so
sánh với công trình gốc của Unsafe-Net nằm ở mục 6.12.5 và mục 6.15.

2.7.1. Thang nhãn so sánh được

  Nhãn | Nghĩa
  -----+--------------------------------------------------------------------------------
  (a)  | So sánh trực tiếp — cùng chỉ số, cùng kho hoặc giao thức tương đương
  (a*) | So sánh được sau chuẩn hoá đơn vị, hoặc cùng đại lượng trên quần thể khác
  (b)  | Cùng phần cứng, khác mô hình
  (c)  | Cùng mô hình hoặc cùng lớp mô hình, khác phần cứng
  (d)  | Khác đơn vị hoặc khác không gian nhãn — không quy đổi được; chỉ là dải tham chiếu

Hai quy ước đi kèm thang này. Với các trục **hao hụt**, nhãn áp cho hiệu số Δ, không áp cho con số
tuyệt đối: hai hệ có thể cho hai độ chính xác không so được nhưng hai mức sụt so được. Và nhãn
(a*) ở đây gánh hai trường hợp khác nhau — chuyển đổi đơn vị, và cùng một đại lượng đo trên một
quần thể khác — nên mỗi bảng phải nói nó dùng nghĩa nào.

2.7.2. Vị trí trên thang chi phí thích nghi

Đây là trục mà luận văn thuộc hẳn vào một mảng tài liệu đã có tên — **thích nghi tham số hiệu quả**
(parameter-efficient transfer learning) — nhưng Chương 2 chưa dẫn mảng ấy. Thiết kế của luận văn là
một điểm trên thang đó, và cần được đọc như vậy.

  Cấu hình                               | Tham số huấn luyện | % so với tinh chỉnh toàn phần
  ---------------------------------------+--------------------+-------------------------------
  Chỉ tinh chỉnh đầu (STA-VPT)           |       0.07 M       |   0.08%
  **Đầu thời gian B1 của luận văn**      |     **0.201 M**    | **0.23%**
  Đầu dò tuyến tính (ST-Adapter)         |       0.31 M       |   0.36%
  VPT-deep                               |       4.68 M       |   5.4%
  ST-Adapter / EVL một lớp               |  7.20 / 7.41 M     |   8.4 – 8.6%
  AIM                                    |      11 M          |  12.8%
  EVL bốn lớp                            |      28.70 M       |  33.4%
  Vita-CLIP                              |      38.88 M       |  45.3%
  Tinh chỉnh toàn phần (TimeSformer)     |  85.87 – 121.57 M  | 100 – 142%

Hai cột mang hai nhãn khác nhau. Cột **tham số huấn luyện** mang nhãn (a): số tham số được cập nhật
là một đại lượng xác định, không phụ thuộc kho, chỉ số hay giao thức chia, nên các hàng so trực
tiếp được. Cột **phần trăm** mang nhãn (a*): mẫu số "tinh chỉnh toàn phần" khác nhau giữa các công
trình — 85.87 triệu cho một tháp ViT-B/32, 121.57 triệu cho TimeSformer — nên tỉ lệ chỉ đọc được
sau khi khai mẫu số, và hàng cuối vượt 100% chính vì mẫu số của nó không phải mẫu số của các hàng
trên.

Đầu được triển khai, ở 0.23% tham số, nằm giữa "chỉ đầu" và "đầu dò tuyến tính", tức thuộc **nhóm
rẻ nhất** của thang đã công bố: rẻ hơn VPT-deep khoảng 23 lần, rẻ hơn AIM khoảng 55 lần, rẻ hơn
tinh chỉnh toàn phần khoảng 427 lần.

Điều cần nói kèm là **vì sao con số nhỏ này không phải một sự tiết kiệm gượng ép**, và câu trả lời
đã nằm trong luận văn chứ không phải trong tài liệu: mục 2.2.6 cho thấy tăng dung lượng khối thời
gian mua được +0.2 điểm phần trăm, và mục 6.2.6 cho thấy kết quả dương duy nhất đo được đến từ đổi
xương sống chứ không từ thiết kế đầu. Nghĩa là luận văn **không** chọn 0.201 triệu tham số vì ngân
sách bắt buộc rồi chịu một hao hụt; nó đo được rằng ở lớp bài toán này, dung lượng đầu không phải
biến quyết định. Hai mảng gặp nhau ở đúng điểm ấy: tài liệu thích nghi tham số hiệu quả lập luận
rằng phần lớn năng lực đã nằm trong xương sống, và phép đo của luận văn độc lập đi tới cùng kết
luận.

2.7.3. Chi phí của việc đóng băng xương sống

Phải nói trước một điều: **luận văn không đo chi phí đóng băng của chính nó.** Không có lượt tinh
chỉnh toàn phần CLIP ViT-B/32 nào được chạy trên làn iSafetyBench, nên con số đó không tồn tại
trong kho và không được suy ra từ bảng của người khác. Đây là một ô trống và nó được công bố như
một ô trống.

Cái luận văn *có* đo là một phép **đổi xương sống** dưới cùng chế độ đóng băng — mục 2.2.3, với
khoảng tin cậy theo cụm **chứa 0**, nên không phát biểu được rằng tháp hình ảnh thắng xương sống
video. Phép đo ấy **không** trả lời câu hỏi "đóng băng tốn bao nhiêu", vì cả hai nhánh đều đóng
băng. Hai câu hỏi phải để riêng.

Dải tham chiếu cho chi phí đóng băng, toàn bộ nhãn (d), phải đọc theo **ba chế độ tách biệt**:

  Chế độ                                    | Δ = (toàn phần) − (đóng băng) | Ghi chú
  ------------------------------------------+-------------------------------+--------------------
  Đóng băng + module thời gian học được     |      −12.0 … +3.3 pp          | lớp thiết kế của B1
  Chỉ đầu dò tuyến tính, không module       |       +5.1 … +43.1 pp         | B1 KHÔNG thuộc đây
  Chỉ prompt, dữ liệu rất ít (ViFi-CLIP K=2)|       +9.3 / +17.5 pp         | phản ví dụ

Cách đọc đúng: B1 thuộc **chế độ thứ nhất**, nơi việc đóng băng thường không gây lỗ và đôi khi còn
lãi — AIM lãi 8.9 pp so với tinh chỉnh toàn phần ở 1% dữ liệu Kinetics-400. Dải +5.1 … +43.1 pp của
chế độ thứ hai **không được mượn** để nói B1 lỗ nhiều, vì B1 có một lớp Transformer thời gian và vị
trí học được, không phải một đầu dò tuyến tính.

Phản ví dụ ở chế độ thứ ba được công bố chứ không bị giấu: ở K = 2 mẫu mỗi lớp, tinh chỉnh toàn
phần **thắng** cấu hình đóng băng-prompt tới 9.3 và 17.5 điểm phần trăm. Nghĩa là mệnh đề "ít dữ
liệu thì đóng băng thắng" không phải một quy luật; nó phụ thuộc vào việc đóng băng *cái gì*. Một
hội đồng hỏi "sao không tinh chỉnh toàn phần" phải nhận được dải ba chế độ này, kèm câu trả lời
thẳng rằng luận văn chưa đo nhánh ấy.

Chiều của hiệu số đo được ở mục 2.2.3 cũng cần một đối chứng từ tài liệu: VideoGLUE báo đầu dò
xương sống ảnh đóng băng vượt xương sống video đóng băng trên Kinetics-400 (75.2 so với 65.1), và
**đổi chiều** trên các tác vụ nặng chuyển động (Something-Something v2 41.0 so với 53.9; Diving48
44.1 so với 59.5). Điều này củng cố đúng cách đọc thận trọng mà mục 2.2.3 đã chọn: hiệu số nghiêng
về tháp hình ảnh ở lớp bài toán này, và tài liệu cho biết đó không phải một quy luật phổ quát.

2.7.4. Lạm phát do giao thức chia dữ liệu, so với các ngành khác

Luận văn đo độ lạm phát trên cả ba làn (mục 6.2.7, mục 6.5, mục 6.6). Mục này **không** lặp lại các
con số ấy; nó trả lời một câu hỏi khác mà Chương 6 không trả lời được: **độ lớn ấy có bất thường
không.**

Trước khi đọc bảng, phải nhắc quy tắc mục 6.1.1, vì nó áp cho chính bảng này: ba con số của luận
văn là ba phép đo **độc lập** của cùng một *hiện tượng* trên ba biểu diễn khác nhau. Chúng **không**
được lấy trung bình, không chứng thực cho nhau, và không được đọc như ba lần đo của cùng một đại
lượng. Vì vậy ba hàng đầu mang nhãn làn, không mang nhãn so sánh được.

  Nguồn                        | Lĩnh vực            | Đơn vị nhóm   | Độ lạm phát           | Nhãn
  -----------------------------+---------------------+---------------+-----------------------+-----
  **Luận văn, làn A**          | video an toàn       | nguồn video   | **+13.5 pp**          | làn
  **Luận văn, làn quan hệ**    | video an toàn       | nguồn video   | **+10.4 … +15.4 pp**  | làn
  **Luận văn, làn Unsafe-Net** | video an toàn       | bối cảnh      | **+13.0 … +21.8 pp**  | làn
  UniqueShip, DeepShip         | âm dưới nước        | bản ghi nguồn | +10.2 … +12.2 pp      | (a*)
  UniqueShip, VTUAD            | âm dưới nước        | bản ghi nguồn | +47.2 … +47.6 pp      | (a*)
  Tổng hợp EEG                 | EEG                 | đối tượng     | +21.5 … +31 pp        | (a*)
  Zhu 2015                     | EEG                 | đối tượng     | +26.15 pp             | (a*)
  Ploton 2020                  | sinh khối viễn thám | không gian    | R² 0.53 xuống 0.14    | (d)
  Tampu 2022                   | OCT                 | bệnh nhân     | MCC +0.07 … +0.43     | (d)
  Peptide MS                   | protein             | —             | +1.8 pp (chiều ngược) | (a*)

Nhãn (a*) ở bảng này mang nghĩa thứ hai của thang: **cùng một đại lượng trên quần thể khác.** Độ
lạm phát là hiệu của hai độ chính xác dưới hai giao thức chia, đo bằng điểm phần trăm, nên không
cần quy đổi đơn vị; cái khác là lĩnh vực và đơn vị nhóm. Vì vậy dải này là một dải tham chiếu về
**độ lớn**, không phải một bảng xếp hạng, và hai hàng đổi chỉ số (R², MCC) thì tụt xuống nhãn (d).

Câu trả lời: **không bất thường** — và phải phát biểu theo từng làn, không phát biểu gộp. Mỗi làn,
đọc riêng, nằm ở giữa-dưới dải đã công bố: cùng băng với DeepShip, thấp hơn EEG và viễn thám. Việc
ba làn độc lập cùng rơi vào một băng là **bằng chứng độc lập hội tụ**, không phải một phép cộng.

Luận văn vì vậy **không** phát hiện ra rò rỉ do nhóm — hiện tượng đã được đặt tên và định lượng ở
nhiều ngành. Cái luận văn đóng góp là **phép đo trên lớp bài toán này**, trong một mảng mà, trong
phạm vi tìm được, không công trình nhận dạng hành động trên video nào công bố con số tương ứng và
không hệ giám sát an toàn công nghiệp nào dùng chia có ràng buộc nhóm.

Ba kết quả cơ chế mạnh hơn bản thân con số Δ, và cả ba đều là của luận văn:

  - **Rò rỉ xoá khả năng phân biệt kiến trúc** (mục 6.6.2): trên làn quan hệ, hai encoder chênh
    nhau 17 lần tham số (298,118 so với 17,270). Dưới chia theo nguồn, hiệu của chúng là
    **+5.1 pp**, CI [+2.0, +8.3] **loại trừ 0**, McNemar p = 7.74e-06. Dưới chia ngẫu nhiên, hiệu
    ấy còn **+0.1 pp**, CI [−1.2, +1.5] **chứa 0**, p = 0.665 — khác biệt biến mất hoàn toàn. Hệ
    quả: một bảng so sánh kiến trúc chạy trên chia ngẫu nhiên **không đo được gì**, và điều đó áp
    cho phần lớn các bảng trong mảng này.
  - **Phần mất đi không phải thông tin đếm tầm thường** (mục 6.6.1): khi chuyển giao thức, đường
    sàn quy tắc đếm chỉ dịch 1.0 điểm phần trăm (68.2% xuống 67.2%) trong khi bộ mã hoá mất 15.4
    điểm phần trăm. Phần bị mất vì vậy là **thông tin đặc thù theo nguồn**, chỉ khai thác được khi
    một nguồn xuất hiện ở cả hai phía của phép chia.
  - **Và phần ấy là danh tính nguồn, không phải hình học fold** (mục 6.6.3): đối chứng giả dược
    G19 tách 15.4 điểm phần trăm thành **13.8 điểm rò rỉ danh tính nguồn** và 1.6 điểm hình học
    fold, trên 20 hạt giống, p = 0.0476. Đây là chỗ tách luận văn khỏi một báo cáo chỉ nêu Δ: cơ
    chế được **tách ra**, không chỉ được gọi tên.

Kết quả thứ nhất là lý do mục này tồn tại. Nếu phần lớn các bảng trong mảng được chạy dưới chia
ngẫu nhiên, thì các con số trong chúng không đo khả năng tổng quát hoá sang nguồn mới, và việc đặt
một con số đo đúng cạnh chúng sẽ phạt chính phép đo đúng. Đây là lập luận định lượng cho việc
mục 2.7.5 từ chối bảy phép so sánh bên dưới.

2.7.5. Những phép so sánh luận văn KHÔNG thực hiện, và lý do

Bảy phép so sánh dưới đây sẽ làm luận văn trông mạnh hơn, và cả bảy đều bị từ chối. Lý do được nêu
để người đọc kiểm tra được, chứ không phải để biện hộ cho một khoảng trống.

  (1) **Kết quả của luận văn trên iSafetyBench so với bài báo gốc của kho ấy.** Bài gốc là trắc
      nghiệm 16 lựa chọn ở chế độ zero-shot, mức may rủi 6.25%; luận văn là bài toán nhị phân có
      huấn luyện dưới chia theo nguồn. Khác tác vụ, khác chỉ số, khác chế độ (mục 2.3). Ba điều này
      đúng bất kể bài gốc chia dữ liệu theo cách nào, nên phép từ chối không phụ thuộc một câu hỏi
      còn mở về kho.

  (2) **Bất kỳ con số nào của luận văn đặt cùng bảng với mảng giám sát yếu.** Mục 2.2.5 đã nêu ba
      lý do; ở đây bổ sung lý do thứ tư và thứ năm, cả hai đều định lượng: AUC không quy đổi được
      sang độ chính xác cân bằng — chặn một chiều duy nhất, ở AUC 0.88, chỉ cho BA ≥ 0.755, quá
      lỏng để dùng — và chênh lệch xương sống B/16 so với B/32 tự nó đáng khoảng 3.6 điểm phần trăm,
      lớn hơn phần lớn các hiệu số mà một bảng như vậy muốn thể hiện.

  (3) **Nhịp quyết định của luận văn so với khung/giây của một bộ phát hiện.** Hai đại lượng khác
      đơn vị. Trong phạm vi tìm được, không tồn tại con số "quyết định mỗi giây" đã công bố cho
      giám sát video cửa sổ trượt trên dòng thiết bị này, nên con số của luận văn **không có đối
      ứng**, và nó không được đặt cạnh thông lượng một-khung của bộ phát hiện.

  (4) **Chi phí đóng băng của chính luận văn.** Chưa đo (mục 2.7.3).

  (5) **Chi phí tính toán lý thuyết của luận văn so với các mô hình video đã công bố.** Phép so này
      khả thi về nguyên tắc, và là phép so gần-trực-tiếp nhất có thể có, vì số phép nhân-cộng không
      phụ thuộc phần cứng hay chuỗi công cụ. Luận văn không in nó vì một lý do thủ tục chứ không
      phải khoa học: nó sẽ đưa vào báo cáo các số chưa có trước mốc đóng băng số liệu. Một quy ước
      phải được khai nếu phép so này được thêm ở lượt sau: nhiều công trình đếm phép nhân-cộng
      nhưng ghi nhãn là FLOPs, nên hai con số chỉ đặt cạnh nhau được sau khi nói rõ bên nào là bên
      nào.

  (6) **Năng lượng của luận văn so với tài liệu.** Các đối chứng khả dụng đo công suất tổng của
      bảng mạch và **không trừ công suất nền**, trên một bảng mạch khác bảng mạch của luận văn. Chỉ
      một phép so hạng (c)/(d) là khả thi, và luận văn chọn không in nó. Hệ quả: các số năng lượng
      của luận văn chỉ được đọc trong phạm vi mục 5.8, và quy tắc mục 5.6.5 vẫn áp nguyên — không
      con số công suất nào được dùng để lập luận về ngân sách năng lượng hay khuyến nghị chế độ
      nguồn.

  (7) **Ba làn của luận văn so với nhau.** Tỉ lệ lớp khác nhau, định nghĩa đơn vị nhóm khác nhau,
      nhãn khác bản chất. Một hiệu số giữa hai làn **không ước lượng cho đại lượng nào**, kể cả khi
      hai con số trông cùng thang (mục 6.1.1).

Điều rút ra từ danh sách này không phải là luận văn thiếu đối chứng. Điều rút ra là **mảng này chưa
có hạ tầng so sánh**: không giao thức chia chung, không chỉ số chung, không báo cáo tham số chung,
và không thói quen công bố khoảng tin cậy. Mục 2.7.4 cho con số cho mệnh đề ấy. Trong tình trạng
đó, một bảng xếp hạng đặt cạnh nhau sẽ là phát minh ra một thứ hạng, không phải đo nó — và sẽ phạt
đúng những phép đo được làm cẩn thận nhất.
```

---

## Mười lăm mục phải thêm vào TÀI LIỆU THAM KHẢO

Danh mục hiện có 45 mục. Năm mục con này viện đến 15 công trình **chưa** xuất hiện ở đâu trong
`report/vn/` — tôi đã kiểm từng tên, tất cả đều 0 lượt.

  Mục con | Công trình phải thêm
  --------+----------------------------------------------------------------------------
  2.7.2   | STA-VPT · VPT · ST-Adapter · EVL · AIM · Vita-CLIP
  2.7.3   | ViFi-CLIP · VideoGLUE · CLIP-LoRA
  2.7.4   | UniqueShip (chứa cả DeepShip và VTUAD) · tổng hợp EEG · Zhu 2015 ·
          | Ploton 2020 · Tampu 2022 · peptide MS

Chi tiết thư mục của 15 mục này nằm trong `research/researcher-02-frozen-vs-finetune-gap.md` và
`research/researcher-05-lam-phat-giao-thuc-chia-du-lieu.md`, **và phải được kiểm lại trước khi
chép**: đợt khảo sát đã phát hiện hai mã arXiv sai trong chính danh mục hiện tại, nên không mã nào
trong hai báo cáo ấy được tin mà không đối chiếu.

**Một lỗi trong danh mục hiện tại, không liên quan mục 2.7 nhưng cùng khu vực và nên sửa cùng lượt:**
mục [24] (MSL) đang mang mã arXiv `2104.04527`, là một bài vật lý. Mục đúng là DOI
`10.1609/aaai.v36i2.20028`. Mã sai thứ hai là `1807.08362`, thực tế là một bài về định nghĩa công
bằng. Chi tiết trong mục 13 của báo cáo tổng hợp.

## Những chỗ bản thảo này cố ý không làm

1. **Không cần ngoại lệ hậu-đóng-băng.** Mọi số đều đã công bố. Đây là lý do hai mục con tính toán
   và năng lượng bị bỏ, và lý do điều (5) và (6) của mục 2.7.5 nói thẳng rằng hai phép so ấy không
   được in — điều (5) nêu rõ lý do là thủ tục, không phải khoa học, để lượt sau có thể thêm.
2. **Không lặp lại Chương 2.** Bảng B1w–B2 (mục 2.2.3), số WS-VAD (mục 2.2.5) và kết quả dung lượng
   (mục 2.2.6) chỉ được dẫn chiếu. Mục 2.7.3 và 2.7.5(2) cộng thêm vào các mục ấy, không thay.
3. **Không đặt 84.2% cạnh 95.81%.** So sánh với công trình gốc đã tồn tại ở mục 6.12.5 và 6.15 với
   đúng đơn vị và đúng kiểm định, kèm kết luận bất lợi. Thân mục 2.7 không trích một con số độ
   chính xác nào của luận văn, nên quy tắc kèm đường sàn 63.3% không bị đụng tới.
4. **Không in bảng xếp hạng nào.** Mọi bảng là định vị độ lớn kèm nhãn, không có cột "hạng".
5. **Không có dòng κ nào**, và không mệnh đề nào về độ tin cậy nhãn hay trần nhãn. Bảng thực hành
   báo cáo — chỗ duy nhất từng có một dòng như vậy — đã bị bỏ khỏi phạm vi.
6. **Quy tắc không trộn làn của mục 6.1.1 được nêu thẳng trong mục 2.7.4**, vì bảng ở đó đặt ba
   con số lạm phát của ba làn vào ba hàng liền nhau. Chương 6 cho phép đọc chúng như ba phép đo độc
   lập của cùng một hiện tượng, và cấm đọc chúng như ba lần đo của cùng một đại lượng. Nếu một lượt
   sửa sau này gộp ba hàng ấy thành một dải "+10.4 … +21.8 pp" cho gọn, nó sẽ phạm đúng điều mục
   6.2.7 cấm. Ba hàng phải giữ nhãn làn.
7. **Đã bỏ hệ số thiết kế mẫu DEFF ≈ 8.4.** Không phải phép đo: nó là phép tính ngược từ mệnh đề
   "khoảng tin cậy nới 2.9 lần", mệnh đề này không tồn tại trong `report/vn/`, và `researcher-05`
   ghi rõ rằng ρ thực tế chưa được kiểm.

## Câu hỏi chưa giải quyết

1. **Chi tiết thư mục của 15 mục mới chưa được kiểm độc lập.** Đây là việc còn lại duy nhất chặn
   việc chép. Tôi có thể kiểm từng mục nếu bạn muốn, nhưng ba công cụ tra cứu đều trả 403 trong đợt
   khảo sát trước, nên có thể phải tra tay.
2. Phát biểu "trong phạm vi tìm được" đã dùng ở ba chỗ thay cho "không công trình nào", vì lý do
   vừa nêu. Mức hạ cấp này có đủ với bạn không?
3. Hai mục con bị bỏ (tính toán lý thuyết, năng lượng) có nên vào **Chương 8** như hướng phát triển
   thay vì mất hẳn? Điều (5) của mục 2.7.5 đã mở cửa cho chúng, nhưng chưa trỏ tới chỗ nào.
