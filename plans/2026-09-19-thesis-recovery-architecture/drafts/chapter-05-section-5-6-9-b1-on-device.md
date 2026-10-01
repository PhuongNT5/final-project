### 5.6.9 Đường quyết định đã triển khai (B1) trên thiết bị

Mục 5.6.8 đo pipeline **P3**. Nhưng họ mô hình sinh ra con số tiêu đề của luận văn này không
phải P3; đó là họ **B1**: 16 khung được lấy mẫu → tháp thị giác **CLIP ViT-B/32 đóng băng**
(87.849.216 tham số, đầu ra 512 chiều) → **đầu Transformer thời gian B1** (tổ hợp 3 hạt giống,
603.270 tham số = 3 × 201.090) → `prob[1,2]`. Mục này đo đúng đồ thị đó, vì một con số độ trễ
chỉ có nghĩa khi nó được đo trên đồ thị đã sinh ra con số độ chính xác.

**Con số độ chính xác đi kèm mọi độ trễ trong mục này là 84,2 %, không phải 84,4 %.** Chương 4
báo cáo B1w (333.570 tham số) ở **84,4 %** và B1 (201.090) ở **84,2 %** — hai giá trị không
phân biệt được về mặt thống kê. Đồ thị **được triển khai là B1**, chọn theo một tiêu chí
**biên** đã ghi trước — *mô hình nhỏ nhất* — chứ không theo độ chính xác
(`deployment/scripts/export-b1-clip-temporal-head.py`, phần mở đầu). Quy tắc kèm theo lựa chọn
đó được tôn trọng ở đây: **trích dẫn con số OOF của chính B1, không bao giờ trích `max(B1, B1w)`**.
Đưa 84,4 % cạnh các độ trễ dưới đây sẽ là gán độ chính xác của một mô hình cho thời gian chạy
của một mô hình khác. Nếu luận văn muốn đặt đúng con số tiêu đề 84,4 % cạnh một số đo trên thiết
bị thì phải xuất và kiểm tra tương đương một engine **B1w** — việc đó **chưa được làm**.

Một đặc điểm kiến trúc phải nói trước, vì nó quyết định cách đọc mọi bảng dưới đây: **không có
bộ phát hiện nào trong quyết định phân lớp B1.** YOLOv11s giữ vai trò **định vị** — hệ thống
phải chỉ ra hiểm hoạ *ở đâu* — và được đo **riêng**, không bao giờ được cộng vào chi phí của
quyết định phân lớp. Ba cổng phủ định hội tụ ở Chương 4 và 7 (G14 thực thể-so-toàn khung,
G17 thế hệ bộ phát hiện, G18 hợp nhất hai luồng) là lý do đồ thị quyết định không tiêu thụ hộp
giới hạn.

#### Điều kiện tiên quyết: tương đương FP16 trước, độ trễ sau

Trích dẫn độ trễ FP16 cạnh 84,2 % chỉ hợp lệ nếu engine FP16 ra **cùng quyết định** với đồ thị
FP32 đã dùng để đo độ chính xác. Trên 24 clip thật: **tỷ lệ đồng thuận quyết định = 1,000**,
`max |Δp| = 2,66·10⁻³`, không có sai lệch tiền xử lý nào (`preprocessing_mismatches` rỗng).
Phép kiểm này được chạy **hai lần**, ở hai mức tải thiết bị khác nhau, và cho cùng kết quả —
nên tương đương không phụ thuộc tải.
Nguồn: `r48-parity-fp16-20261001T{222019,224028}.json`.

Trường `end_to_end_incl_decode_ms` của chính các tạo tác đó (p50 = 1.761,8 ms) **không được
trích dẫn như nhịp quyết định**: nó bao gồm trích khung bằng seek ngẫu nhiên vào tệp mp4, là
đường ngoại tuyến. Xem phần giải mã bên dưới.

**Bảng 5.5N — Độ trễ mỗi giai đoạn của đường B1, đo **liên tiếp** (back-to-back)** (FP16,
MAXN_SUPER, DVFS mặc định — không ghim nhịp, cửa sổ yên tĩnh, `runs = 100`/giai đoạn, 3 lần chạy
độc lập). **Đây là chi phí khi GPU đã ở điểm vận hành của nó, không phải chi phí trong vòng lặp
đã triển khai**; vòng lặp tổ hợp ở Bảng 5.5O mới là nhịp triển khai.

| Giai đoạn | tham số | lô | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|---:|
| Tháp CLIP ViT-B/32 (đóng băng) | 87.849.216 | 16 | 23,386 / 23,335 / 23,567 | 31,19 / 35,08 / 35,42 |
| **Đầu Transformer thời gian B1** | **603.270** | 1 | **0,410 / 0,396 / 0,391** | 0,485 / 0,502 / 0,484 |
| **Mỗi quyết định (tính toán)** | | | **23,796 / 23,731 / 23,958** | |
| **Quyết định/giây** | | | **42,02 / 42,14 / 41,74** | |
| Tỷ lệ của xương sống | | | **98,28 / 98,33 / 98,37 %** | |

Nguồn: `r48-pipeline-20261001T{223826,223853,223920}.json` (cả ba đều mang `quiet = true` và
`clocks_pinned = {cpu: false, gpu: false}`). Ba lần chạy lệch nhau **0,95 %** ở trung vị.

Kết quả có ý nghĩa là hàng cuối, và nó **lặp lại phát hiện của mục 5.6.8 trên đồ thị đã triển
khai**: đóng góp của công trình này — đầu Transformer thời gian — chiếm **1,7 %** độ trễ quyết
định; **98,3 %** nằm ở tháp CLIP đóng băng, một thành phần tải về chứ không phải thành phần
được huấn luyện ở đây. Ở P3 tỷ lệ tương ứng là 0,05 % / 60,5 %. Hai pipeline, hai kiến trúc,
cùng một kết luận về phía mô hình: **hiệu quả tham số của Chương 4 đi thẳng được vào triển
khai — không có công việc tối ưu độ trễ nào thuộc về đầu Transformer.** Nhưng kết luận "phần còn
lại thuộc về xương sống" chỉ đúng *trong phạm vi các giai đoạn mô hình*. Khi đo cả vòng lặp
(Bảng 5.5O), hạng mục chi phối **không phải** xương sống mà là **tiền xử lý trên CPU**, và tỷ lệ
1,7 % / 98,3 % của bảng này trở thành 0,3 % / 20,2 % so với toàn bộ khoảng giữa hai quyết định.
Một bảng giai đoạn không thể tự mình chỉ ra hạng mục tối ưu đúng.

#### Bộ phát hiện là cái thước sai cho một ViT

Dự đoán ghi trước phiên đo (P1) nói tháp CLIP sẽ tốn **60–120 ms** cho 16 khung, suy ra từ chi
phí 12,6 ms/khung của bộ phát hiện. Giá trị đo được là **23,386 ms** — **nằm ngoài khoảng dự
đoán, thấp hơn chặn dưới 2,6 lần**. **P1 bị phủ định.** Nguyên nhân: ViT-B/32 ở 224² xử lý 49
token mỗi khung và lô 16 khung chạy như **một** lần suy luận, nên nó khai thác song song theo lô
mà một bộ phát hiện một-ảnh-một-lần ở 800 px không có. Hai mốc so sánh đo trong cùng cửa sổ:

| Mốc so sánh | lô | p50 (ms) |
|---|---:|---:|
| YOLOv11s @ 800 px (**định vị**, đo riêng, **không** cộng vào quyết định) | 1 | 12,636 / 13,001 / 12,741 |
| MobileNetV3-small @ 224² | 16 | 7,336 / 7,710 / 7,645 |

Tháp CLIP đắt hơn MobileNetV3-small **3,06×** ở cùng lô 16 — đó là giá phải trả cho 84,2 %, và
nó là một con số nhỏ chứ không phải một rào cản triển khai.

#### Tiền xử lý — không phải mô hình và cũng không phải giải mã — là chi phí chi phối

Có **hai** phép đo giải mã trên thiết bị và chúng lệch nhau 2,5×, nên phải nói rõ con số nào là
con số triển khai. Phép đo cô lập cho **5,047 / 5,091 ms mỗi khung** (p95 6,259 / 6,943), nguồn
khối `decode` của `decision-latency-fp16-20261001T{224529,224821}.json`. Vòng lặp tổ hợp cho
**2,048 ms mỗi khung** (Bảng 5.5O). Chênh lệch có hai nguyên nhân, và nguyên nhân lớn hơn là **độ
phân giải** — đã kiểm chứng chứ không phỏng đoán: `data/test` gồm **125 clip, toàn bộ 1920×1080**,
còn `data/isafetybench` có độ phân giải trải từ **320×240 tới 1280×720**, phần lớn ở 720p hoặc
thấp hơn, tức khoảng 1/4 đến 1/7 số điểm ảnh của 1080p. Nguyên nhân thứ hai: tạo tác cô lập tính
thời gian **8 khung liên tiếp** mỗi lần (`emit_every = 8`) nên chi phí khởi động mỗi đợt chỉ được
chia cho 8 khung, trong khi vòng lặp tổ hợp giải mã **liên tục cả clip** (5.559 khung / 24 clip
≈ 232 khung mỗi clip).

**Con số triển khai là 2,048 ms/khung — và định ngữ "trên kho này" là thiết yếu**: chi phí giải mã
phụ thuộc độ phân giải nên nó không phải một hằng số của thiết bị. **Rủi ro phải nêu rõ**: một
camera 1080p sẽ tốn khoảng 5 ms/khung, tức theo số học khoảng quyết định tăng thêm ~72 ms lên
~362 ms = 2,76 quyết định/giây, chỉ còn khoảng **1,10×** so với hợp đồng 2,500 qđ/giây của nguồn
1080p ở 60 fps. **Cấu hình đó chưa được đo.** Đây là cách khả dĩ nhất để bản triển khai trượt
thời gian thực, và biện pháp giảm nhẹ — NVDEC — cũng chưa được đo vì OpenCV trên thiết bị này
được biên dịch không có GStreamer. Cảnh báo của cả hai tạo tác được giữ nguyên: giải mã **từ
tệp, không phải từ luồng camera trực tiếp**, và không được làm nóng cache.

Bản triển khai dùng `--window 48 --stride 24`; `sample()` lấy T = 16 khung bằng `np.linspace`
trên cửa sổ, đúng quy tắc mà phần đánh giá dùng. Bước 24 khung biến tần số khung của nguồn thành
một **hợp đồng nhịp thời gian thực**: nguồn ở *f* fps buộc hệ thống phát *f*/24 quyết định mỗi
giây mới theo kịp luồng.

Các bản nháp trước lấy *f* = 25,1 fps và suy ra hợp đồng **1,05 quyết định/giây**. Con số đó
**không đại diện cho kho dữ liệu**: đo siêu dữ liệu của cả **1.100** clip iSafetyBench cho trung
vị **29,97** fps, trung bình **34,25** fps, p95 **60,0** fps, và 25,1 fps chỉ xuất hiện ở **11
clip (1,0 %)**. Phân bố thực tế: 30,0 fps ở 653 clip (59,4 %), **60,0 fps ở 218 clip (19,8 %)**,
25,0 fps ở 92 clip, 15,0 fps ở 53 clip.

Vì vậy hợp đồng phải được phát biểu theo phân bố, không theo một giá trị:

| nguồn ở | hợp đồng (quyết định/giây) | margin của 3,452 qđ/giây |
|---|---:|---:|
| trung vị kho (29,97 fps) | 1,249 | **2,76×** |
| trung bình kho (34,25 fps) | 1,427 | 2,42× |
| **p95 kho = tối đa (60,0 fps), 19,8 % clip** | **2,500** | **1,38×** |

Hợp đồng vẫn **được đáp ứng trên toàn kho**, kể cả ở clip 60 fps, nhưng margin ở đó là **1,38×**
chứ không phải con số nhiều lần mà bản nháp trước nêu — và đó là margin của một vòng lặp **đơn
luồng**, tức một chặn dưới. Mọi phát biểu "thời gian thực" trong chương phải đi kèm tần số khung
mà nó áp dụng.

**Bảng 5.5O — Nhịp **đo được** của vòng lặp tổ hợp B1** (giải mã → tiền xử lý → tháp → đầu;
`--window 48 --stride 24`; 24 clip, 201 quyết định, 5.559 khung mỗi lần; 3 lần lặp ở **cùng**
`--seed 0` để độ phân tán đo *thiết bị* chứ không đo biến thiên tập clip; FP16, DVFS mặc định,
cửa sổ yên tĩnh)

| Đại lượng | min | **trung vị** | max | phân tán |
|---|---:|---:|---:|---:|
| **nhịp trạng thái dừng, ms/quyết định (p50)** | 289,339 | **289,672** | 292,834 | 1,21 % |
| nhịp trạng thái dừng, ms/quyết định (p95) | 321,764 | **322,708** | 322,909 | 0,35 % |
| nhịp tính cả lấp cửa sổ, ms/quyết định | 259,704 | **260,315** | 260,825 | 0,43 % |
| tiền xử lý p50 (ms) | 160,629 | **160,765** | 160,926 | 0,18 % |
| tháp CLIP p50 **trong vòng lặp** (ms) | 58,002 | **58,630** | 58,883 | 1,50 % |
| đầu B1 p50 (ms) | 0,790 | **0,798** | 0,806 | 2,01 % |
| tính toán p50 (ms/quyết định) | 218,750 | **219,153** | 222,310 | 1,62 % |
| giải mã p50 (ms/khung) | 2,048 | **2,055** | 2,104 | 2,73 % |
| hệ số hoạt động (tính toán/đồng hồ tường) | 0,758 | **0,761** | 0,761 | 0,41 % |

Nguồn: `r48-composed-cadence-fp16-20261001T{233159,233324,233449}.json` (cả ba `quiet = true`,
`clocks_pinned.any = false`, `competing_processes` rỗng). Mọi hàng qua được phép kiểm lặp lại
`max − min ≤ 0,10 × trung vị` đã dùng cho trường bộ nhớ.

Nhịp đã triển khai là **289,672 ms/quyết định = 3,452 quyết định/giây** (p95 322,708 ms), tức
margin **2,76×** so với hợp đồng của clip trung vị (1,249 qđ/giây) và **1,38×** so với clip
60 fps — xem bảng hợp đồng ở trên. Vòng lặp **đơn luồng**: một triển khai
thực tế sẽ gối giải mã lên suy luận, nên 3,452 quyết định/giây là **chặn dưới** của thông lượng
và 76,1 % là **chặn trên** của hệ số hoạt động.

Bảng này **thay thế** một ngân sách tổng giai đoạn mà các bản nháp trước của mục này trình bày
như một nhịp. Phải nói rõ vì sao bản nháp đó sai, chứ không chỉ sửa số: một **tổng** các phép đo
giai đoạn cô lập không thể thấy ba chi phí, và cả ba đều lớn.

| | tỷ lệ của tính toán | tỷ lệ của khoảng 289,672 ms |
|---|---:|---:|
| **tiền xử lý (resize + chuẩn hoá 16 khung, CPU)** | **73,4 %** | **55,5 %** |
| tháp CLIP ViT-B/32 | 26,5 % | 20,2 % |
| đầu Transformer B1 | 0,4 % | 0,3 % |
| giải mã (24 khung × 2,048 ms) | — | 17,0 % |
| chưa quy được (chi phí vòng lặp, `cap.read()` ngoài khoảng đo, chỉ mục bộ đệm vòng) | — | 7,4 % |

1. **Tiền xử lý chưa từng có trong ngân sách.** Không bị đánh giá thấp — **vắng mặt**. Nó nằm
   giữa giải mã và tháp, không giai đoạn cô lập nào đo nó, và nó là hạng mục đắt nhất của cả
   pipeline: đắt hơn ViT, đắt hơn giải mã.
2. **Tháp tốn 58,630 ms trong vòng lặp so với 23,386 ms khi đo liên tiếp — 2,51×.** Nguyên nhân
   được phân xử ở Bảng 5.5P.
3. **Tổng giai đoạn ước thiếu nhịp đo được khoảng 4×.** Tính lại trung thực với chính con số
   giải mã trong vòng lặp: 24 × 2,048 + 23,796 = 72,948 ms, so với 289,672 ms đo được — hệ số
   **3,97**.

Phát biểu "giải mã chiếm 83,6 % ngân sách và mô hình chiếm 16,4 %" của bản nháp trước vì vậy bị
**rút lại**: giải mã chiếm 17,0 % khoảng quyết định, và hạng mục chi phối là tiền xử lý.

Trích khung bằng **seek** ngẫu nhiên cho 16 khung đã lấy mẫu tốn **5.815,9 / 5.729,0 ms**, tức
khoảng **1.140×** so với giải mã tuần tự — con số đó là chi phí chuẩn bị dữ liệu **ngoại tuyến**
và không thuộc về nhịp quyết định ở bất kỳ đâu trong luận văn này.

#### Vì sao tháp chậm 2,5× trong vòng lặp: một phép phân xử đã ghi trước khi chạy

Hai nguyên nhân cùng dự đoán hiện tượng đó và chúng dẫn tới hai kết luận khác nhau. **H-sụt-nhịp**:
xung nhịp không được ghim, giữa hai quyết định GPU rỗi khoảng 210 ms trong lúc CPU giải mã và tiền
xử lý, nên bộ điều khiển hạ xung nhịp SM và mỗi lần gọi tháp phải trả giá cho pha tăng nhịp lại —
nếu đúng, đây là hiệu ứng **trạng thái năng lượng của thiết bị**. **H-tranh-chấp**: giải mã và tiền
xử lý tranh chấp bus bộ nhớ với tháp — nếu đúng, hiệu ứng là **nội tại của pipeline** và không thể
xoá bằng việc ghim nhịp.

Phép thử tách hai nguyên nhân: chạy **riêng tháp**, cùng engine, cùng trạng thái xung nhịp không
ghim, 60 lần/nhánh — nhánh A liên tiếp, nhánh B chèn một khoảng rỗi 210 ms trước mỗi lần gọi và
**không có gì khác chạy**. Tranh chấp bị loại trừ *theo thiết kế*, nên mọi mức tăng ở nhánh B chỉ
có thể do trạng thái xung nhịp. **Quy tắc phân xử được ghi trước khi chạy** (và không thương lượng
lại sau khi thấy kết quả): H-sụt-nhịp được xác nhận khi và chỉ khi nhánh B ≥ 1,8 × nhánh A **và**
nhánh B nằm trong ±20 % của 58,630 ms, tức trong [46,90 ; 70,36] ms.

**Bảng 5.5P — Phép phân xử sụt nhịp khi rỗi** (chỉ tháp CLIP, FP16, 60 lần/nhánh, 3 lần lặp)

| Nhánh | p50: min / **trung vị** / max (ms) | phân tán |
|---|---:|---:|
| liên tiếp (back-to-back) | 25,944 / **26,448** / 26,653 | 2,7 % |
| + khoảng rỗi 210 ms, không có gì khác chạy | 54,675 / **55,231** / 55,832 | 2,1 % |
| **tỷ số rỗi / liên tiếp** | 2,051 / **2,088** / 2,152 | 4,8 % |

Nguồn: `r48-idle-droop-probe-fp16-20261001T{234145,234242,234322}.json` (cả ba `quiet = true`,
`clocks_pinned.any = false`).

Theo đúng quy tắc đã ghi trước: 2,088 ≥ 1,8 và 55,231 ms nằm trong dải — **H-sụt-nhịp được xác
nhận, độc lập ở cả ba lần lặp**. Chỉ riêng khoảng rỗi — không giải mã, không tiền xử lý, không
thể có tranh chấp — tái tạo **94,2 %** chi phí tháp trong vòng lặp (55,231 / 58,630).
H-tranh-chấp **không cần thiết** để giải thích bất cứ điều gì; phần dư 3,4 ms (5,8 %) được để
**chưa quy** thay vì gán cho nó. Dải xung nhịp GPU của bo mạch là 306–1020 MHz, tức 3,33×, nên
nó chứa thừa một hiệu ứng 2,09×.

Hai hệ quả ngược chiều nhau, và phải nêu cả hai. Thứ nhất, 23,386 ms **không sai**: nó là một
con số **liên tiếp** và phải được gắn nhãn như vậy. Thứ hai, nó **cũng không phải chi phí triển
khai**, vì triển khai bị chu kỳ hoá theo thiết kế — một quyết định mỗi 24 khung, GPU rỗi ở giữa.
Một con số đo liên tiếp không thể mang sang một pipeline có chu kỳ công tác.

**Chặn trên — bằng số học, không phải phép đo — của lợi ích từ việc ghim nhịp**: nếu ghim nhịp
đưa tháp từ 58,630 về 26,448 ms và không đổi gì khác, khoảng quyết định giảm tối đa 32,18 ms về
257,5 ms = 3,88 quyết định/giây, tức **+12,5 %**. Trần đó thấp vì hạng mục chi phối là tiền xử
lý — một chi phí CPU trên một CPU vốn luôn bận. **Tối ưu tiền xử lý có giá trị gấp nhiều lần
việc ghim nhịp, và khác với ghim nhịp, nó là một thay đổi phần mềm khả chuyển.**

Một kết quả phụ cần báo cáo: nhánh liên tiếp đọc **26,448 ms** ở nơi `decision-latency-fp16-*.json`
đọc 23,386 ms — **cao hơn 13,1 %** cho cùng một phép đo trên danh nghĩa (khác bộ công cụ, khác
nguồn dữ liệu vào; cùng engine, cùng trạng thái thiết bị). Điều đó chặn độ chính xác của mọi con
số độ trễ giai đoạn cô lập ở khoảng **±13 %**, nên trong chương này không con số nào thuộc loại
đó được trích dẫn quá 3 chữ số có nghĩa.

#### Năng lượng

| | giá trị |
|---|---:|
| Thời lượng bão hoà | 60,015 s |
| Số lần suy luận tháp | 2.508 |
| Nhịp | 41,789 suy luận/s |
| Công suất TB (VDD_IN) | **20,105 W** |
| J / quyết định (tổng công suất bảng mạch) | **0,4811 J** |
| J / quyết định (biên, P_rỗi = 4,691 W) | **0,36885 J** |

Nguồn: `energy-r48-clip-vit-b32-vision-fp16-20261001T224142.json`, quy chiếu về nền rỗi 60 s
đã ổn định của `energy-p3-relation-head-bin-e3-fp16-20261001T225523.json`.

Ba lưu ý. Con số **tổng công suất bảng mạch 0,4811 J không đổi** khi đổi nền rỗi, vì nó là
`P_TB / fps` và không chạm tới nền rỗi; chỉ con số **biên** dịch từ 0,36522 J (nền rỗi 10 s,
4,842 W) sang **0,36885 J** (nền rỗi 60 s, 4,691 W) — lệch 1,0 %. Nền rỗi 4,691 W được đo khi
chỉ có engine nhỏ nhất thường trú, nên nó là chặn dưới của "nền rỗi khi engine thường trú" và
con số biên là chặn trên nhẹ. Và **năng lượng riêng của đầu B1 chưa được đo**; nó bị chặn bởi
tỷ lệ 1,7 % của nó trong độ trễ quyết định.

#### Bộ nhớ và dấu chân engine

Dấu chân **tệp engine** là chính xác và kiểm tra lại được: tháp CLIP **174,022 MB** + đầu B1
**1,874 MB** = **175,9 MB** cho đường quyết định phân lớp; thêm engine định vị YOLOv11s
(21,543 MB) thành **197,4 MB**.
Nguồn: `model-size-20261001T224927.json`.

Bộ nhớ thiết bị tăng thêm do engine thường trú, đo **3 lần trong một cửa sổ yên tĩnh** (cả ba
`quiet = true`, `competing_processes` rỗng, không ghim nhịp, tải 0,07–0,25, 52–53 °C):
**91,36 / 90,01 / 89,61 MB**, trung vị **90,01 MB**, phân tán **1,94 %**.
Nguồn: `memory-fp16-20261001T{232747,232808,232830}.json`. RSS đỉnh phía host 31,5–33,4 MB.

Con số này chỉ được trích dẫn vì nó **qua được một phép kiểm lặp lại đã ghi trước khi chạy**
(lặp lại nếu `max − min ≤ 0,10 × trung vị`; đo được 1,75 ≤ 9,00). Lý do phải ghi trước một phép
kiểm như vậy: trường `engines_resident_delta_mb` suy ra từ `cuda.mem_get_info()`, tức bộ nhớ hợp
nhất trống **toàn hệ thống**, nên nó bị nhiễu bởi mọi thứ khác đang chạy trên máy — và trong các
tạo tác không mang nhãn trạng thái, nhiễu đó đã thực sự xuất hiện ở mức 6,6×.

Một cảnh báo còn lại dù phép kiểm đã qua: 90,01 MB **nhỏ hơn một nửa** tổng kích thước tệp
engine tương ứng (197,4 MB). Vì vậy trường này **không phải** "trọng số thường trú trong bộ nhớ
thiết bị" và không được gán nhãn như thế; cách phát biểu đúng là *"độ giảm bộ nhớ hợp nhất trống
toàn hệ thống khi làm các engine này thường trú"*, còn kích thước tệp engine là con số chính xác
được trích dẫn riêng. Chi tiết và bằng chứng:
`plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-10-01-engine-resident-memory-not-reproducible.md`.

Phần 8 GB của thiết bị không phải ràng buộc tới hạn của đường B1: với 175,9 MB engine, **bộ nhớ
không phải trục tới hạn**. Trục độ trễ thì **không còn thoải mái như bản nháp trước nghĩ**: ở clip
60 fps (19,8 % kho) margin chỉ còn 1,38×, nên với đường B1 độ trễ là trục chặt hơn trong hai trục
— cùng dấu với P3, chỉ là ở mức nhẹ hơn.

#### DVFS tạo đuôi khi chạy liên tiếp — và dịch cả trung vị khi chạy theo chu kỳ công tác

Mọi số của đường B1 trong mục này được đo dưới **DVFS mặc định** — điều kiện một hệ thống
triển khai thực tế vận hành. **Không tồn tại phép đo đường B1 ở điều kiện ghim xung nhịp mà
trạng thái được tạo tác chứng thực**, nên mục này **không** đưa ra so sánh DVFS-so-ghim-nhịp
cho B1. Hai lần chạy sớm hơn cùng ngày thường được nhắc tới như "đã ghim nhịp"
(`r48-pipeline-20261001T{222505,222511}.json`, p50 23,481 / 23,167 ms) **không mang khối
`cpu_clocks`/`gpu_clocks`/`clocks_pinned` nào** — công cụ ghi trạng thái chưa được tích hợp
vào thời điểm đó — và hai lần chạy cách nhau 6 giây thì không phải hai lần lặp độc lập. Theo
đúng quy tắc của mục 5.6.8, chúng **không được trích dẫn**.

Điều *có thể* phát biểu từ ba lần chạy sạch, **khi đo liên tiếp**: DVFS tạo **đuôi**. Tỷ số
p95/p50 của tháp CLIP là **1,33 / 1,50 / 1,50** (23,386 → 31,187; 23,335 → 35,077;
23,567 → 35,418), cùng dấu với hiện tượng mục 5.6.8 báo cáo cho P3, và trung vị thì ổn định
(phân tán 0,95 % qua ba lần chạy).

Nhưng phát biểu "không dịch trung vị" **chỉ đúng cho chế độ liên tiếp**, và Bảng 5.5P rút lại nó
cho chế độ triển khai: khi pipeline bị chu kỳ hoá, DVFS dịch trung vị của tháp **2,09×**
(26,448 → 55,231 ms). Đó là hiệu ứng DVFS lớn nhất đo được ở bất kỳ đâu trong chương này, và nó
không nằm ở đuôi mà ở ngay trung vị. Bài học phương pháp: *đo một giai đoạn liên tiếp sẽ che mất
hiệu ứng DVFS chi phối của một triển khai có chu kỳ công tác.*

Hệ quả cho hợp đồng nhịp thì vẫn an toàn, vì nhịp của Bảng 5.5O **đã** chứa hiệu ứng đó: ngay ở
p95 đo được (322,708 ms) đường B1 vẫn đạt 3,099 quyết định/giây, tức **2,48×** hợp đồng của clip
trung vị và **1,24×** hợp đồng của clip 60 fps — khác với P3, nơi đuôi là đại lượng ràng buộc.

#### Những gì chưa được đo cho B1

Đường B1 ở điều kiện **ghim xung nhịp** chưa được đo với trạng thái được chứng thực, nên lợi
ích của việc ghim nhịp đối với đuôi phân phối của B1 chưa được định lượng.
Độ bão hoà **đa luồng** của đường B1 chưa được đo (bộ công cụ hiện có nạp danh sách giai đoạn
P1 cố định, cùng lý do như ở mục 5.6.8). **Soak** của vòng lặp tổ hợp chưa được đo, nên margin
2,76× (và 1,38× ở clip 60 fps) chưa được chứng minh là bền trong 30 phút. **Giải mã bằng phần cứng NVDEC** chưa được đo
— OpenCV trên thiết bị này được biên dịch không có GStreamer — nhưng vì giải mã chỉ chiếm 17,0 %
khoảng quyết định, trần của nó là 17 %; hạng mục có đòn bẩy lớn nhất là **tiền xử lý** (55,5 %),
và tiền xử lý **chưa được phân tích bên trong**: 160,8 ms cho 16 khung là ~10 ms/khung cho
resize + chuẩn hoá, và chưa đo được phần nào trong đó là `cv2.resize`, phần nào là chuyển kiểu
số thực. Vòng lặp tổ hợp ở điều kiện **ghim xung nhịp** chưa được đo, nên con số +12,5 % ở trên
là số học chứ không phải phép đo. Engine **FP32 và INT8** của tháp CLIP
chưa được dựng lại cho đúng thiết bị nên số của chúng chưa được trích dẫn. Năng lượng riêng
của đầu B1 chưa được đo. Và giải mã được đo trên tệp, không phải trên luồng camera trực tiếp.

#### Xuất xứ tạo tác của mục này

Mọi engine TensorRT dùng trong mục này được dựng **ngày 01/10/2026 trên chính thiết bị này**.
Điều đó là cần thiết: toàn bộ 30 engine dựng vào hoặc trước 29/09 phát cảnh báo
`Using an engine plan file across different models of devices...` khi giải tuần tự hoá, nên số
độ trễ của chúng không được trích dẫn. Vì thư mục `deployment/export/jetson` trộn engine sạch
của 01/10 với engine cũ, đường B1 được đo qua thư mục liên kết tượng trưng riêng
`deployment/export/jetson-r48` chỉ chứa ba engine của đường này.

Các engine **không** được đưa vào kho mã: chúng phụ thuộc thiết bị (một engine dựng trên bảng
này không nạp được trên bảng khác) và riêng engine tháp CLIP đã 174 MB, vượt hạn mức 100 MB
mỗi tệp của GitHub. Chúng được dựng lại trên thiết bị từ ONNX đã kiểm tra md5 hai đầu.

Mỗi tạo tác nhúng `device_state` với `cpu_clocks`, `gpu_clocks`, `clocks_pinned`,
`load_average` và `competing_processes` đọc trực tiếp từ sysfs. **Trước khi trích dẫn bất kỳ
số độ trễ nào, phải kiểm tra `quiet` và `clocks_pinned` của tạo tác đó.** Ba lần chạy ở Bảng
5.5N đều mang `quiet = true`, `competing_processes` rỗng và `clocks_pinned.any = false`. Các
lần chạy sớm hơn cùng ngày (`r48-pipeline-20261001T{221822,222505,222511}.json`) **không** mang
khối trạng thái này và **không được trích dẫn** ở bất kỳ đâu.

Không giá trị nào trong mục này được nhớ lại từ ký ức — mỗi giá trị được đọc từ tạo tác của nó
tại thời điểm viết.

---

**Trạng thái chèn.** Mục này **chưa được chèn** vào `report/chapter-05-edge-deployment.md` (1024
dòng tại thời điểm viết, chương chưa có §5.6.9 và đang được một phiên khác chỉnh sửa). Chèn ngay
sau §5.6.8, trước `## 5.7 Limitations`. Nhãn bảng 5.5N / 5.5O / 5.5P đã kiểm tra không trùng với
5.5J-5.5M của §5.6.8. Mọi con số trong mục này có tạo tác thiết bị tương ứng và được đối chiếu với
Block 3b của number sheet; bằng chứng và quy tắc phân xử ở
`plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-10-01-stage-sum-is-not-a-cadence.md`.
