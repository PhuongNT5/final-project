# Bản thảo mục 2.7 — So sánh định lượng với tài liệu

Loại: bản thảo chương · Ngày: 2026-11-24 · Nguồn số: `reports/synthesis-261117-so-sanh-dinh-luong-da-truc.md`

**Trạng thái:** bản thảo nằm ngoài `report/vn/`. Chương đang đóng băng và quyền sửa thuộc về tác
giả, nên tệp này **chỉ để chép vào**, không phải một sửa đổi đã thực hiện.

**Vị trí đề xuất:** mục **2.7**, đặt sau mục 2.6 và trước TÀI LIỆU THAM KHẢO của Chương 2. Vị trí
này thuần cộng thêm, không đánh số lại mục nào.

**Phạm vi đã bị thu hẹp sau khi đọc lại Chương 2.** Bản thảo đầu tiên của tệp này dài hơn và lặp
lại ba chỗ đã có trong chính chương: mục 2.2.3 đã có bảng B1w so với B2 với đúng các con số ấy,
mục 2.2.5 đã có số của mảng giám sát yếu kèm ba lý do từ chối, mục 2.2.6 đã có kết quả dung lượng
và điểm 5.1 so với 1.4 điểm phần trăm. Những phần đó đã được **bỏ khỏi bản thảo và thay bằng dẫn
chiếu**. Mục 2.7 vì vậy chỉ chứa những trục mà Chương 2 hiện **chưa** có con số nào.

**Một quyết định tác giả phải chốt trước khi chép.** Mục này đưa vào báo cáo ba con số chưa có
trong bản đóng băng: 4.409 GMACs mỗi ảnh, 70.5 GMACs mỗi quyết định, 30.07 mJ mỗi khung. Hai con
số đầu là phép đếm lý thuyết của một mô hình công bố của bên thứ ba, dựng từ cấu hình của
`open_clip` 3.3.0 với trọng số ngẫu nhiên; con số thứ ba là phép chia 0.48110 J ÷ 16 của một số đã
có trong Chương 5. Không lượt huấn luyện nào chạy lại, không tập kiểm thử nào bị chạm, ngân sách
chạm tập kiểm thử tiêu **không lần nào**. Nhưng mốc đóng băng 05/10 cấm thêm số mới, nên nếu chép
thì phải ghi nhận đây là **ngoại lệ hậu-đóng-băng thứ năm**; nếu không, bỏ mục 2.7.3 và mục 2.7.5
và giữ bốn mục còn lại, vốn chỉ dùng số đã công bố.

---

```
2.7. SO SÁNH ĐỊNH LƯỢNG VỚI TÀI LIỆU
----------------------------------------------------------------------

Mục 2.6 định vị luận văn bằng lời. Mục này định vị bằng số, và phải mở đầu bằng một khó khăn: phần
lớn các con số được công bố trong hai mảng lân cận **không so sánh trực tiếp được** với các con số
của luận văn. Chúng đo trên kho khác, không gian nhãn khác, giao thức chia khác, phần cứng khác, và
thường không nêu giao thức chia của chúng. Một bảng đặt cạnh nhau mà không nói rõ điều đó sẽ tạo ra
một thứ hạng không có nghĩa.

Giải pháp của luận văn là **không bỏ phép so sánh mà gắn nhãn cho nó**. Mỗi hàng trong mọi bảng của
mục này mang một nhãn cho biết hàng ấy so sánh được tới mức nào, và mục 2.7.7 liệt kê các phép so
sánh mà luận văn **từ chối thực hiện**, kèm lý do kỹ thuật của từng phép.

Mục này không lặp lại các phép so sánh đã có chỗ riêng: hiệu số giữa xương sống hình ảnh đóng băng
và xương sống video đóng băng nằm ở mục 2.2.3; vị thế so với mảng giám sát yếu nằm ở mục 2.2.5; so
sánh với công trình gốc của Unsafe-Net nằm ở mục 6.12.5 và mục 6.15.

2.7.1. Thang nhãn so sánh được

  Nhãn | Nghĩa
  -----+--------------------------------------------------------------------------------
  (a)  | So sánh trực tiếp — cùng chỉ số, cùng kho hoặc giao thức tương đương
  (a*) | So sánh được sau chuẩn hoá đơn vị (MAC ↔ FLOP, mỗi-khung ↔ mỗi-quyết-định)
  (b)  | Cùng phần cứng, khác mô hình
  (c)  | Cùng mô hình hoặc cùng lớp mô hình, khác phần cứng
  (d)  | Khác đơn vị hoặc khác không gian nhãn — không quy đổi được; chỉ là dải tham chiếu

Với các trục **hao hụt**, nhãn áp cho hiệu số Δ, không áp cho con số tuyệt đối.

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

Đầu được triển khai, ở 0.23% tham số, nằm giữa "chỉ đầu" và "đầu dò tuyến tính", tức thuộc **nhóm
rẻ nhất** của thang đã công bố: rẻ hơn VPT-deep khoảng 23 lần, rẻ hơn AIM khoảng 55 lần, rẻ hơn
tinh chỉnh toàn phần khoảng 427 lần.

Điều cần nói kèm là **vì sao con số nhỏ này không phải một sự tiết kiệm gượng ép**, và câu trả lời
đã nằm trong luận văn chứ không phải trong tài liệu: mục 2.2.6 cho thấy tăng dung lượng khối thời
gian mua được +0.2 điểm phần trăm, và mục 6.2.6 cho thấy kết quả dương duy nhất đo được đến từ đổi
xương sống chứ không từ thiết kế đầu. Nghĩa là luận văn **không** chọn 0.201 M vì ngân sách bắt
buộc rồi chịu một hao hụt; nó đo được rằng ở lớp bài toán này, dung lượng đầu không phải biến quyết
định. Hai mảng gặp nhau ở đúng điểm ấy: tài liệu thích nghi tham số hiệu quả lập luận rằng phần lớn
năng lực đã nằm trong xương sống, và phép đo của luận văn độc lập đi tới cùng kết luận.

2.7.3. Tham số và tính toán lý thuyết

Đây là trục duy nhất cho một phép so sánh gần-trực-tiếp về chi phí tính toán, vì nó không phụ thuộc
phần cứng, chuỗi công cụ hay trạng thái xung nhịp.

  Mô hình                               | Tham số       | Tính toán              | Nhãn
  --------------------------------------+---------------+------------------------+------
  **Tháp CLIP ViT-B/32 của luận văn**   | **87,849,216**| **4.409 GMACs mỗi ảnh**|  (a)
    → mỗi quyết định, 16 khung          |               | **70.5 GMACs ≈ 141 GF**| (a*)
  X-CLIP B/32, 16 khung × 224           | —             | 75 G                   | (a*)
  X3D (XS → L)                          | 3.8 – 6.2 M   | 0.60 – 18.37 G         | (a*)
  VideoMAE-B / -L                       | 87 / 305 M    | 180 / 597 G × 15 view  | (a*)
  TimeSformer                           | 121.4 M       | 0.59 T                 | (a*)
  SlowFast (ba biến thể)                | 34.5 – 62.9 M | không công bố          |  (a)
  ConvLSTM bài gốc [10]                 | không công bố | không công bố          |  (d)

  Nguồn: open_clip 3.3.0, cấu hình ViT-B-32; phép đếm giải tích từ chiều rộng 768, 12 lớp,
         50 token, tỉ lệ MLP 4, đối chiếu với `image_gflops` = 8.818 của chính open_clip.

Ba điều phải nói kèm bảng này. **Thứ nhất, quy ước đơn vị.** Luận văn đếm theo MAC và quy đổi
FLOP = 2 × MAC; 4.409 GMACs mỗi ảnh tương đương 8.818 GFLOPs, trùng giá trị `image_gflops` mà
open_clip công bố cho chính mô hình này. Nhiều công trình đếm MAC nhưng ghi nhãn là GFLOPs, nên quy
ước phải được khai trước khi đặt hai con số cạnh nhau. Phép đếm bằng công cụ tự động cần được kiểm
lại bằng tay: `thop` không có hook cho `nn.MultiheadAttention` — lớp mà `open_clip` dùng — nên nó
bỏ im lặng toàn bộ bốn phép chiếu q/k/v/out và hai phép nhân ma trận chú ý, cho ra 2.947 GMACs,
thiếu 1.462 GMACs so với phép đếm giải tích.

**Thứ hai, sự trùng khớp với X-CLIP là một phép kiểm chéo, không phải một chiến thắng.** X-CLIP
công bố 75 G cho cùng xương sống B/32 ở cùng 16 khung 224 điểm ảnh; luận văn đếm được 70.5 GMACs.
Hai giá trị gần nhau tới mức cách đọc hợp lý duy nhất là X-CLIP cũng đếm MAC dưới nhãn GFLOPs. Điều
này **xác nhận** phép đếm của luận văn chứ không cho luận văn ưu thế nào: hai hệ dùng chung một
xương sống thì chi phí xương sống phải bằng nhau. Chênh lệch còn lại là phần khối thời gian, và nó
nhỏ hơn sai số làm tròn của bảng.

**Thứ ba, ConvLSTM không có hàng số.** Bài gốc [10] là bài dự báo mưa, không công bố tham số, không
công bố FLOPs, không đánh giá trên kho nhận dạng hành động nào. Mọi so sánh tính toán với "ConvLSTM"
vì vậy mang nhãn (d) và không được in như một hiệu số. Đây cũng là lý do mục 2.2.6 loại họ ấy bằng
lập luận thiết kế, không bằng một bảng chi phí.

2.7.4. Chi phí của việc đóng băng xương sống

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

2.7.5. Năng lượng mỗi khung

  Hệ                                   | mJ mỗi khung | Phần cứng         | Nhãn
  -------------------------------------+--------------+-------------------+---------
  CoDAT-S                              |    13.79     | AGX Orin          | (c)/(d)
  **Luận văn B1, quy ước toàn board**  |  **30.07**   | Orin Nano Super   |    —
  MoViNet-A1                           |    86.84     | AGX Orin          | (c)/(d)
  MoViNet-A3                           |   155.37     | AGX Orin          | (c)/(d)
  TimeSformer                          |   591.90     | AGX Orin          | (c)/(d)

  Nguồn: 0.48110 J mỗi quyết định ÷ 16 khung (bảng công suất Chương 5); CoDAT Bảng V.

Hai điều kiện ràng buộc phải in kèm. **Quy ước đo phải khớp:** CoDAT đo bằng jtop/INA3221, lấy công
suất tổng và **không trừ nền**, nên chỉ hàng toàn board 30.07 mJ của luận văn là cùng quy ước; con
số biên 23.05 mJ (đã trừ công suất nền) **không** được đặt vào bảng này. **Phần cứng khác nhau:**
AGX Orin không phải Orin Nano Super, nên mọi hàng đối chứng mang nhãn (c)/(d), không phải (a). Điều
đọc được từ bảng chỉ là một định vị độ lớn: B1 nằm giữa CoDAT-S và MoViNet-A1, và thấp hơn
TimeSformer khoảng 20 lần.

Quy tắc của mục 5.6.5 áp nguyên vào đây: **không một con số công suất nào trong bảng này được dùng
để lập luận về ngân sách năng lượng hay để khuyến nghị một chế độ nguồn.**

Trên mảng giám sát an toàn công nghiệp, đây là trục luận văn **không có đối thủ**: trong phạm vi tìm
được, không hệ nào trong mảng đó báo cáo một phép đo năng lượng nào.

2.7.6. Lạm phát do giao thức chia dữ liệu, so với các ngành khác

Luận văn đo độ lạm phát trên cả ba làn (mục 6.5, mục 6.6). Mục này **không** lặp lại các con số ấy;
nó trả lời một câu hỏi khác mà Chương 6 không trả lời được: **độ lớn ấy có bất thường không.**

  Nguồn                        | Lĩnh vực            | Đơn vị nhóm   | Độ lạm phát
  -----------------------------+---------------------+---------------+---------------------------
  **Luận văn, làn A**          | video an toàn       | nguồn video   | **+13.5 pp**
  **Luận văn, làn quan hệ**    | video an toàn       | nguồn video   | **+10.4 … +15.4 pp**
  **Luận văn, làn Unsafe-Net** | video an toàn       | bối cảnh      | **+13.0 … +21.8 pp**
  UniqueShip, DeepShip         | âm dưới nước        | bản ghi nguồn | +10.2 … +12.2 pp
  UniqueShip, VTUAD            | âm dưới nước        | bản ghi nguồn | +47.2 … +47.6 pp
  Tổng hợp EEG                 | EEG                 | đối tượng     | +21.5 … +31 pp
  Zhu 2015                     | EEG                 | đối tượng     | −26.15 pp (90.97 → 64.82)
  Ploton 2020                  | sinh khối viễn thám | không gian    | R² 0.53 → 0.14
  Tampu 2022                   | OCT                 | bệnh nhân     | MCC +0.07 … +0.43
  Peptide MS                   | protein             | —             | +1.8 pp (chiều ngược)

Câu trả lời: **không bất thường.** Luận văn nằm ở giữa-dưới dải đã công bố, cùng băng với DeepShip,
thấp hơn EEG và viễn thám. Luận văn vì vậy **không** phát hiện ra rò rỉ do nhóm — hiện tượng đã được
đặt tên và định lượng ở nhiều ngành. Cái luận văn đóng góp là **phép đo trên lớp bài toán này**,
trong một mảng mà, trong phạm vi tìm được, không công trình nhận dạng hành động trên video nào công
bố con số tương ứng và không hệ giám sát an toàn công nghiệp nào dùng chia có ràng buộc nhóm.

Hai kết quả cơ chế mạnh hơn bản thân con số Δ, và cả hai đều là của luận văn:

  - **Rò rỉ xoá khả năng phân biệt kiến trúc** (mục 6.6.2): dưới chia ngẫu nhiên, một dải tham số
    rộng 17 lần chỉ trải 1.4 điểm phần trăm; dưới chia theo nguồn, cùng dải ấy trải 5.1 điểm phần
    trăm, khoảng tin cậy loại trừ 0. Hệ quả: một bảng so sánh kiến trúc chạy trên chia ngẫu nhiên
    **không đo được gì**, và điều đó áp cho phần lớn các bảng trong mảng này.
  - **Phần mất đi không phải thông tin đếm tầm thường** (mục 6.6.3): khi chuyển giao thức, đường sàn
    quy tắc đếm chỉ dịch 1.0 điểm phần trăm (68.2% xuống 67.2%) trong khi bộ mã hoá mất 15.4 điểm
    phần trăm. Phần bị mất vì vậy là **thông tin đặc thù theo nguồn**, chỉ khai thác được khi một
    nguồn xuất hiện ở cả hai phía của phép chia.

Kết quả thứ nhất là lý do mục này tồn tại. Nếu phần lớn các bảng trong mảng được chạy dưới chia
ngẫu nhiên, thì các con số trong chúng không đo khả năng tổng quát hoá sang nguồn mới, và việc đặt
một con số đo đúng cạnh chúng sẽ phạt chính phép đo đúng. Đây là lập luận định lượng cho việc
mục 2.7.7 từ chối bảy phép so sánh bên dưới.

2.7.7. Độ chặt thống kê và báo cáo

  Thực hành                            | Luận văn  | WS-VAD (12 bài) | An toàn CN (10 hệ)
  -------------------------------------+-----------+-----------------+--------------------
  Khoảng tin cậy                       |    có     | không bài nào   | không hệ nào
  Đơn vị lấy mẫu là nhóm, không là clip|    có     | không           | không
  Nhiều hạt giống / báo cáo phương sai |    có     | không bài nào   | không
  Hiệu chỉnh đa so sánh                |    có     | không           | không
  Kiểm định có cặp (McNemar chính xác) |    có     | không           | không
  Chia có ràng buộc nhóm               |    có     | không bài nào   | không hệ nào
  Cổng phán quyết đăng ký trước        |    có     | không           | không
  Đối chứng giả dược                   |    có     | không           | không
  Đo trên thiết bị biên                |    có     | một bài         | vài hệ, FPS lẻ
  Đo năng lượng                        |    có     | không           | không hệ nào
  Công khai mã và dữ liệu              |    có     | một phần        | một phần
  Đo độ tin cậy người gắn nhãn         | có, và    | không bài nào   | không hệ nào
                                       | báo trượt |                 |

Trong phạm vi tìm được, **không công trình nào trong cả hai mảng có đủ bốn điều**: đo trên thiết bị
biên kèm năng lượng, khoảng tin cậy, chia có ràng buộc nhóm, và công khai mã cùng dữ liệu.

Hai hàng cuối cần giải thích, vì cả hai đều là kết quả **bất lợi** được công bố như bất lợi. Trong
tám cổng phán quyết đăng ký trước, **hai đạt và sáu trượt**, và cả sáu lần trượt đều được in kèm
bằng chứng; một tỉ lệ trượt như vậy là dấu hiệu việc đăng ký trước có hiệu lực thật chứ không phải
một thủ tục trang trí. Tương tự, phép đo độ tin cậy người gắn nhãn được đăng ký trước kèm một luật
dừng, luật dừng ấy **đã kích hoạt**, và phép đo được tuyên là không diễn giải được (Phụ lục). Luận
văn vì vậy **không** tuyên bố rằng nhãn của kho đã được xác nhận đáng tin, và **không** tuyên bố
rằng mô hình đã chạm trần nhãn. Điều có thể so sánh với tài liệu chỉ là: phép đo ấy đã được thực
hiện và kết quả bất lợi của nó đã được công bố, trong khi không công trình đối chứng nào thực hiện
nó.

2.7.8. Những phép so sánh luận văn KHÔNG thực hiện, và lý do

Bảy phép so sánh dưới đây sẽ làm luận văn trông mạnh hơn, và cả bảy đều bị từ chối. Lý do được nêu
để người đọc kiểm tra được, chứ không phải để biện hộ cho một khoảng trống.

  (1) **Kết quả của luận văn trên iSafetyBench so với bài báo gốc của kho ấy.** Bài gốc là trắc
      nghiệm 16 lựa chọn ở chế độ zero-shot, mức may rủi 6.25%; luận văn là bài toán nhị phân có
      huấn luyện dưới chia theo nguồn. Khác tác vụ, khác chỉ số, khác chế độ (mục 2.3).

  (2) **Bất kỳ con số nào của luận văn đặt cùng bảng với mảng giám sát yếu.** Mục 2.2.5 đã nêu ba
      lý do; mục này bổ sung lý do thứ tư và thứ năm, cả hai đều định lượng: AUC không quy đổi được
      sang độ chính xác cân bằng — chặn một chiều duy nhất, ở AUC 0.88, chỉ cho BA ≥ 0.755, quá
      lỏng để dùng — và chênh lệch xương sống B/16 so với B/32 tự nó đáng khoảng 3.6 điểm phần trăm,
      lớn hơn phần lớn các hiệu số mà một bảng như vậy muốn thể hiện.

  (3) **Nhịp quyết định của luận văn so với khung/giây của một bộ phát hiện.** Hai đại lượng khác
      đơn vị. Trong phạm vi tìm được, không tồn tại con số "quyết định mỗi giây" đã công bố cho
      giám sát video cửa sổ trượt trên dòng thiết bị này, nên con số của luận văn **không có đối
      ứng**, và nó không được đặt cạnh thông lượng một-khung của bộ phát hiện.

  (4) **Chi phí đóng băng của chính luận văn.** Chưa đo (mục 2.7.4).

  (5) **Hao hụt lượng tử hoá INT8 của đường triển khai.** Engine FP32 và INT8 của tháp CLIP chưa
      được dựng lại cho đúng thiết bị, nên không có số để trích. Hàng FP32/FP16/INT8 duy nhất trong
      kho là số lịch sử của một pipeline khác trên một làn khác, và nó không được đọc như hao hụt
      lượng tử hoá của hệ thống triển khai. Cái **đã** đo được cho đường triển khai là tác động của
      FP16: mức đồng thuận quyết định 1.000 và McNemar chính xác p = 1.000, tức không phân biệt
      được khác biệt nào.

  (6) **Năng lượng theo quy ước biên so với tài liệu.** Đối chứng khả dụng không trừ công suất nền,
      nên chỉ quy ước toàn board được so (mục 2.7.5).

  (7) **Ba làn của luận văn so với nhau.** Tỉ lệ lớp khác nhau, định nghĩa đơn vị nhóm khác nhau,
      nhãn khác bản chất. Một hiệu số giữa hai làn **không ước lượng cho đại lượng nào**, kể cả khi
      hai con số trông cùng thang (mục 6.1.1).

Điều rút ra từ danh sách này không phải là luận văn thiếu đối chứng. Điều rút ra là **mảng này chưa
có hạ tầng so sánh**: không giao thức chia chung, không chỉ số chung, không báo cáo tham số chung,
và không thói quen công bố khoảng tin cậy. Mục 2.7.6 cho con số cho mệnh đề ấy. Trong tình trạng
đó, một bảng xếp hạng đặt cạnh nhau sẽ là phát minh ra một thứ hạng, không phải đo nó — và sẽ phạt
đúng những phép đo được làm cẩn thận nhất.
```

---

## Những chỗ bản thảo này cố ý không làm

1. **Không lặp lại Chương 2.** Bảng B1w–B2 (mục 2.2.3), số WS-VAD (mục 2.2.5) và kết quả dung lượng
   (mục 2.2.6) chỉ được dẫn chiếu. Mục 2.7.4 và 2.7.8(2) cộng thêm vào hai mục ấy, không thay.
2. **Không đặt 84.2% cạnh 95.81%.** So sánh với công trình gốc đã tồn tại ở mục 6.12.5 và 6.15 với
   đúng đơn vị và đúng kiểm định, kèm kết luận bất lợi.
3. **Không in bảng xếp hạng nào.** Mọi bảng là định vị độ lớn kèm nhãn, không có cột "hạng".
4. **Không trích số nào đã bị rút**, không dùng `max(B1, B1w)` cho bất kỳ số nào trên thiết bị, và
   mọi lần nhắc 84.2% ở phần còn lại của luận văn vẫn phải đi kèm đường sàn 63.3%.
5. **Không có dòng κ nào** hàm ý độ tin cậy nhãn đã được xác nhận; hàng duy nhất nói rằng phép đo
   đã thực hiện và đã báo trượt.
6. **Đã bỏ hệ số thiết kế mẫu DEFF ≈ 8.4.** Bản thảo đầu có câu "DEFF ≈ 8.4 đo được". Con số ấy
   **không phải phép đo**: nó là phép tính ngược từ mệnh đề "khoảng tin cậy nới 2.9 lần", mệnh đề
   này không tồn tại trong `report/vn/`, và `researcher-05` đã ghi rõ rằng ρ thực tế chưa được kiểm.
   Không có cách nào in nó mà không tạo ra một số mới không nguồn.

## Câu hỏi chưa giải quyết

1. Tác giả có chấp nhận **ngoại lệ hậu-đóng-băng thứ năm** cho ba con số nêu ở đầu tệp không? Nếu
   không, bỏ mục 2.7.3 và 2.7.5; bốn mục còn lại chỉ dùng số đã công bố và chép được ngay.
2. Mục này đặt ở **2.7** (thuần cộng thêm) hay **6.16** (phải đánh số lại mục kết luận chương)?
3. Dải CoDAT có được đặt cùng bảng với 30.07 mJ/khung không, hay quy tắc không trộn làn nên hiểu
   rộng thành cấm luôn so sánh xuyên bảng mạch?
4. Phát biểu "trong phạm vi tìm được" đã dùng ở bốn chỗ thay cho "không công trình nào", vì ba công
   cụ tra cứu đều trả 403 trong đợt khảo sát. Mức hạ cấp này có đủ với tác giả không?
5. Mục 2.7.2 là mục duy nhất đưa một **mảng tài liệu mới** vào Chương 2 (thích nghi tham số hiệu
   quả: VPT, AIM, ST-Adapter, EVL, Vita-CLIP). Chín trích dẫn mới sẽ phải vào TÀI LIỆU THAM KHẢO.
   Tác giả có muốn mở rộng danh mục ở giai đoạn này không?
