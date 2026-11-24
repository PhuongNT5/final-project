# TIỀN ĐĂNG KÝ — đo trần nhãn của iSafetyBench nhị phân (κ)

Ngày: **01/11/2026** · Nhánh `next-round` · Mã việc: **R3** (`plans/trang-thai-chuong-trinh.md` §11)
Căn cứ: khuyến nghị **số một** của hội đồng 10/10, nhắc lại ở hội đồng 31/10 §2.9.

**Tệp này được commit TRƯỚC khi gắn nhãn clip đầu tiên, cùng commit với ba công cụ và sổ tay mã
hoá.** Đó là chủ đích, không phải tình cờ: hội đồng 31/10 §2.1 phát hiện các hiến chương cổng
trước đây vào git **sau** kết quả chúng phán xử (`risks.md` 01/10 so với kết quả G14 ngày 29/09),
một lỗ hổng xuất xứ không phép chạy lại nào vá được. Nghiên cứu này không lặp lại điều đó.

---

## 1. Câu hỏi

Nhãn nhị phân của iSafetyBench **tái tạo được đến mức nào** bởi một người gán nhãn độc lập, và
**độ chính xác cân bằng của con người** trên chính các clip đó là bao nhiêu so với mô hình B1?

Đây **không** phải câu hỏi về mô hình. Nó quyết định liệu `84.2 %` đã **sát trần nhãn** hay còn
dư địa — và nó **chặn** mọi quyết định nâng cấp xương sống (hội đồng 31/10 §2.9).

## 2. Thiết kế — chốt, không đàm phán lại

**Mẫu: tổng điều tra (census) toàn bộ 151 nguồn, đúng MỘT clip mỗi nguồn.** Không phải lấy mẫu.

Ba lý do, đo được:
1. **Cân bằng nhãn sẵn.** 151 nguồn = **76 hazard / 75 normal** (ở cấp clip là 420/680 lệch 38/62). Cân bằng loại bỏ nghịch lý κ theo tỉ lệ hiện mắc.
2. **Thông tin thống kê cao hơn n = 200.** ICC theo nguồn là 0.43–0.44, nên n = 200 với 2 clip/nguồn có **cỡ hiệu dụng ≈ 140**, thấp hơn n = 151 với các clip **độc lập hoàn toàn**. Mô phỏng (4,000 lượt): n = 151 cho nửa rộng CI95 của κ là **±0.126 ở κ≈0.60, ±0.114 ở κ≈0.70, ±0.086 ở κ≈0.85**; n = 200 độc lập chỉ hơn ±0.01–0.02. Chênh lệch không đáng để đánh đổi điểm 3.
3. **Chặn rò rỉ nhãn trong cùng nguồn.** **0/151 nguồn mang cả hai nhãn.** Clip thứ hai của một nguồn đã gắn nhãn là **đáp án cho sẵn**. Một clip/nguồn khiến việc đó không thể xảy ra.

Hệ quả phải nói thẳng: **n = 151 < 200** mà hội đồng 10/10 nêu. 200 là số tròn; 151 là **trần thật**
dưới ràng buộc độc lập, và mạnh hơn 200 có cặp. Không nâng n bằng cách thêm clip cùng nguồn.

**Tiền định, tái lập được:** mẫu là hàm tiền định của `seed = 20261101`, `order_sha256 =
11f757c01b26b3b14dd7ece9b7640005e7c3dc05807346900e72dfc6d267ae4f`, và `tools/sample-kappa-clips.py`.
Không niêm phong gì — **tái lập được là tính chất mạnh hơn niêm phong.**

| Tạo tác | sha256 |
|---|---|
| `events.json` (176 sự kiện, **không chứa nhãn**) | `dfeab129243009d998f3d97cdf8e1fdce495246c154c8bfa915278b153251009` |
| `key.json` (K-id → nhãn gốc; **bị gitignore có chủ ý**) | `663d660581d1874e60a4105508e120163f250dc86e88038708deb9cef677ac94` |

**Làm mù:** giao diện chỉ thấy K-id. Tên tệp thật chứa mã nguồn, thư mục cha **chính là nhãn** —
nên máy chủ không bao giờ gửi đường dẫn ra trình duyệt, và traceback được chặn để không in đường
dẫn (lỗi tìm ra khi kiểm thử 01/11). Giao diện **không hiện số đếm theo nhãn**: một bảng đếm sẽ
cho phép cân bằng theo tỉ lệ kỳ vọng, đúng loại thiên lệch nghiên cứu này không được có.

**Lặp lại để đo chính người gán nhãn:** 25 clip hiện lần hai, cách lần đầu ≥ 40 sự kiện. Không có
số này thì κ giữa hai bên **không diễn giải được**: κ thấp có thể là nhiễu của nhãn, cũng có thể
chỉ là nhiễu của người.

**Biến đồng thời, tiền đăng ký:** độ tin cậy 1–3 (bắt buộc), cờ **"tôi nhận ra clip này"**, thời
gian xem, số lần xem lại.

## 3. Loại trừ trước khi có dữ liệu

**`K096` bị loại, n = 150.** Trong phiên dựng và kiểm thử công cụ ngày 01/11, một traceback của máy
chủ đã in đường dẫn thật của `K096` vào bản ghi hội thoại mà người gán nhãn đọc được ⇒ nhãn của nó
**có thể đã bị thấy trước**. Loại trước thì rẻ; tranh luận sau thì không. Lỗi đã vá (§2).

## 4. Đại lượng và cách tính — đã viết thành mã, đã commit

`tools/analyse-kappa.py`, commit cùng tệp này. Đọc số ra khỏi đó được phép đổi **kết luận** của
luận văn; không được phép đổi **những gì script tính**.

| Đại lượng | Cách tính |
|---|---|
| **Cohen κ** (chính) | người (pass 1) ↔ nhãn gốc, n = 150 |
| **Gwet AC1** (chính) | cùng dữ liệu; κ nhạy với tỉ lệ hiện mắc, các nhóm con **không** cân bằng |
| **Độ chính xác cân bằng của NGƯỜI** | cùng đại lượng đầu bảng của luận văn (§8.3.9) |
| **Độ chính xác cân bằng của MÔ HÌNH trên đúng 150 clip đó** | đọc từ OOF đã công bố `enc_isbclip_B1_lso-1100_headline-none_oof.json`; **không chạy mô hình** |
| Hiệu NGƯỜI − MÔ HÌNH | kèm CI bootstrap |
| **McNemar chính xác** | trên tính đúng/sai từng clip, cùng phép kiểm luận văn vẫn dùng |
| **κ nội-người** (test–retest) | 25 clip lặp |
| Cắt nhóm | nhận ra / không nhận ra · tin cậy cao / thấp (chỉ báo cáo khi n ≥ 10) |

CI: bootstrap **theo nguồn**, B = 10,000, seed 20261101. Mỗi nguồn góp đúng một clip nên cụm là
đơn lẻ và bootstrap cụm **trùng** bootstrap thường — tính chất **có chủ đích** của thiết kế tổng
điều tra, không phải thiếu hiệu chỉnh cụm.

## 5. Luật diễn giải — chốt TRƯỚC, kể cả khi bất lợi

Gọi κ = Cohen κ, H = BA của người, M = BA của mô hình trên cùng 150 clip.

| Mã | Điều kiện | Kết luận **buộc phải** viết |
|---|---|---|
| **D1** | κ ≥ 0.85 **và** H − M ≥ +5 pp với CI loại 0 | **RÚT lập luận "sát trần nhãn".** Nhãn đáng tin và người hơn mô hình rõ ⇒ **còn dư địa thật**; §6 phải viết lại theo hướng đó, và hướng nâng cấp xương sống **mở lại** |
| **D2** | κ ≤ 0.75 **hoặc** CI của H − M chứa 0 | "84.2 % sát trần nhãn" **được chống đỡ**, phát biểu kèm số đo. **Không** được nói "ngang bằng con người" |
| **D3** | κ nội-người < 0.80 | **Toàn bộ phép đo tuyên bố KHÔNG DIỄN GIẢI ĐƯỢC.** Báo cáo đúng như vậy, không kết luận về phía nào |
| **D4** | tỉ lệ bật cờ "nhận ra" ≥ 50 % | Nghiên cứu dán nhãn **ĐÃ YẾU ĐI** trong văn bản. Đại lượng chính **vẫn là** toàn bộ 150 clip — không đổi đại lượng chính sau khi thấy số |
| **D5** | mọi trường hợp khác | báo cáo số, không nâng thành tuyên bố nào mạnh hơn số cho phép |

**Phát ngôn bị cấm bất kể kết quả:** "mô hình ngang con người", "trần nhãn là X %" như một chặn
cứng, và đặt κ này cạnh **κ = 0.65 cũ** (đo trên Unsafe-Net 4 lớp + nền, n = 100 — **làn khác**).

**Mối đe doạ hiệu lực lớn nhất, nói trước:** nếu sổ tay mã hoá định nghĩa "nguy hiểm" lệch với
định nghĩa của iSafetyBench thì κ thấp đo **sai lệch định nghĩa**, không đo **nhiễu nhãn**. Giảm
nhẹ: R1–R10 chốt trước; hiệu chuẩn trên Unsafe-Net; và **đọc danh sách lớp của bài báo iSafetyBench
(arXiv 2508.00399) trước khi chốt sổ tay** — mọi tinh chỉnh sau đó là phụ lục có ngày.

## 6. Ngân sách chạm tập kiểm thử: **TIÊU 0 LẦN**

Còn 1 lần trong ngân sách 2 (1 đã tiêu cho G21+G25 ngày 11/10). Nghiên cứu này **không tiêu lần nào**:
không huấn luyện, không đánh giá lại mô hình, không tính số mới từ trọng số. Dự đoán OOF là **kết
quả LSO đã công bố**; 150 clip được **chốt bằng seed trước** khi đọc bất kỳ số nào của mô hình trên
chúng. Số của mô hình trên tập con được tính **một lần**, báo cáo **bất kể ra sao**, và `84.2 %`
trên toàn bộ 1,100 clip **vẫn là số đầu bảng**.

## 7. Đưa vào luận văn — **ngoại lệ bốn**

`report/vn/` đã đóng băng (số 09/10, văn bản 15/10). Thêm κ là **phép đo mới** ⇒ phải khai thành
**ngoại lệ bốn** tại `report/vn/00-trang-thai-bien-soan.txt`, theo đúng khuôn ngoại lệ hai và ba:
dẫn chiếu tiền đăng ký này, ghi **tiêu 0 lần chạm test**, và ghi **bảng cổng không đổi — hai đạt,
sáu trượt, tám cổng**.

Khuyến nghị: đặt kết quả vào **phụ lục mới, có ngày**, cộng **một dòng dẫn chiếu** ở §6 — thay vì
chen vào giữa các mục đã đóng băng. Ít xâm phạm nhất, dễ bảo vệ nhất.

## 8. Chính sách sửa đổi

Sau sự kiện gắn nhãn đầu tiên, tệp này **không được sửa**. Mọi thay đổi là phụ lục có ngày ở cuối.
Dữ liệu commit **theo từng phiên**, để dấu thời gian độc lập kiểm chứng được.

## 9. Câu hỏi chưa giải quyết

1. Đã đọc danh sách lớp của bài báo iSafetyBench trước khi chốt sổ tay chưa? (§5, mối đe doạ lớn nhất)
2. Có người gán nhãn **thứ hai** cho một tập con 60–80 clip không? Không bắt buộc, nhưng nó **chặn
   được thiên lệch của chính tác giả** — người có lợi ích trong việc κ thấp. Nếu có, đăng ký bổ sung.
3. Tác giả nhận ra bao nhiêu phần trăm clip? Không biết trước; D4 đã chốt cách xử lý.
4. Nếu D1 xảy ra (kết quả **bất lợi** cho lập luận hiện tại) — còn 60 ngày tới 31/12, có đủ để viết
   lại §6 theo hướng "còn dư địa" không? Nên trả lời **trước** khi gắn nhãn, không phải sau.

---

## Sửa đổi 1 — 17/11/2026, trước khi gắn clip đầu tiên

**Không có người gán nhãn thứ hai.** Tác giả chốt 07/11. Hệ quả, ghi trước khi có dữ liệu:

1. **Không đo được κ liên-người.** Phép đo duy nhất về độ tin cậy là **κ nội-người** trên 25 clip
   hiện lại. Đổi tên estimand ở §4 cho đúng: nghiên cứu này đo **độ tái lập của một người gán nhãn
   so với nhãn của bộ dữ liệu**, không đo **độ đồng thuận giữa những người gán nhãn**.
2. **D3 trở thành luật chịu lực.** Trước đây κ nội-người là phép kiểm tra phụ; nay nếu nó < 0.80
   thì không còn phép đo nào đỡ, và theo D3 toàn bộ nghiên cứu không diễn giải được. Hệ quả thực
   tế: ba phiên ≤ 60 sự kiện và nghỉ ≥ 30 phút không còn là khuyến nghị, mà là điều kiện.
3. **Giới hạn phải khai trong phụ lục, đúng chữ này**: con số đo được là mức trùng khớp giữa **một**
   người đọc sổ tay R1–R10 và phân hoạch thư mục của iSafetyBench. Nó **không** là trần nhãn của
   bộ dữ liệu, và **không** là giới hạn Bayes của bài toán. Phản biện có quyền hỏi "một người thì
   biết gì về nhiễu nhãn" — câu trả lời đúng là: không biết, và phụ lục nói thẳng như vậy.
4. **Phát ngôn bị cấm, bổ sung**: "hai người gán nhãn độc lập", "độ đồng thuận giữa các chuyên gia",
   và mọi cách diễn đạt gợi ý có nhiều hơn một người đọc clip.
5. Phần còn lại của tiền đăng ký **không đổi**: vẫn tổng điều tra 151 nguồn, vẫn loại K096 ⇒ n = 150,
   vẫn hai hash đã chốt, vẫn D1–D5, vẫn **tiêu 0 lần chạm test**, vẫn **bảng cổng không đổi —
   tám cổng, hai đạt, sáu trượt**.

Lý do ghi sửa đổi này thành văn bản thay vì lặng lẽ bỏ qua: hội đồng 10/10 yêu cầu **hai** người.
Thiết kế nay chỉ còn một. Khai trước khi có dữ liệu là cách duy nhất để sự thu hẹp đó không bị đọc
thành kết quả bị cắt gọt sau khi đã thấy số.

---

## Sửa đổi 2 — 17/11/2026, sau khi gắn xong 176 sự kiện, TRƯỚC khi chạy `analyse-kappa.py`

**Điều kiện phiên của Sửa đổi 1 đã KHÔNG đạt.** Đo trực tiếp từ dấu thời gian trong
`annotations.jsonl`, không phải tự thuật:

| Sửa đổi 1 đặt làm **điều kiện** | Đo được |
|---|---|
| 3 phiên, mỗi phiên ≤ 60 sự kiện | **1 phiên, 176 sự kiện** |
| nghỉ ≥ 30 phút giữa các phiên | **nghỉ dài nhất 1,3 phút**; 0 lần nghỉ ≥ 5 phút |
| commit theo từng phiên (§8) | 1 commit duy nhất, sau khi xong |
| — | 26,1 phút tổng ⇒ **8,9 s/sự kiện**; 22/176 dưới 3 giây; dwell trung vị 7,4 s |
| 25 clip lặp cách ≥ 40 sự kiện | **đạt**: 50–173 sự kiện (trung vị 111) |
| — | nhưng khoảng cách **thời gian thực** chỉ **8–26 phút** (trung vị 16) |

Toàn vẹn dữ liệu thì **không** có vấn đề: 176/176 cặp (kid, pass) duy nhất, không thiếu, không
thừa, không dòng ghi đè, `confidence` có ở mọi dòng (3→138, 2→36, 1→2).

### Hệ quả, ghi trước khi thấy bất kỳ số nào

1. **D3 mất một nửa quyền lực, giữ nửa còn lại.** κ nội-người đo trên hai lần gắn cách nhau 8–26
   phút trong **cùng một lần ngồi** thì phần lớn là **ký ức ngắn hạn**, không phải độ tái lập.
   Cụ thể, bất đối xứng phải tôn trọng:
   - κ nội-người **< 0.80** ⇒ D3 vẫn nổ, và nổ **mạnh hơn**: không tái lập được **dù có trợ giúp
     của ký ức**.
   - κ nội-người **≥ 0.80** ⇒ D3 **không** nổ, nhưng **không** được đọc thành "độ tin cậy đã
     được xác nhận". Nó là **chặn trên** của độ tái lập, không phải ước lượng của nó.
2. **D4 hết tác dụng bảo vệ.** Cờ "nhận ra" bật **1/176 (0,6 %)**, xa ngưỡng 50 %, nên theo đúng
   chữ thì D4 không nổ. Nhưng với lần lặp cách nhau 8–26 phút, con số gần bằng không **không**
   chứng minh tác giả thật sự không nhận ra clip — nó chỉ cho thấy tín hiệu đó **không được dùng**.
   Vì vậy: **cấm** dùng tỉ lệ 0,6 % làm bằng chứng rằng ký ức không can thiệp. Cắt nhóm
   "nhận ra / không nhận ra" **không báo cáo** (n = 1 < 10, trùng luật n ≥ 10 ở §4).
3. **Đại lượng CHÍNH không bị thổi phồng.** κ người ↔ nhãn gốc trên n = 150 chỉ dùng pass 1; nén
   phiên và gấp gáp làm nó **giảm**, không làm nó tăng. Do đó:
   - κ chính **thấp** ⇒ có **hai** cách giải thích cạnh nhau — nhiễu nhãn thật, **hoặc** 8,9
     s/sự kiện là quá nhanh. Phải khai cả hai; **không** được chỉ chọn cách có lợi cho §6.
   - κ chính **cao** ⇒ đọc được, và là phía **thận trọng**: cao *dù* gấp gáp.
4. **Phát ngôn bị cấm, bổ sung**: "κ nội-người xác nhận độ tin cậy", "đã kiểm chứng độ tái lập",
   và mọi câu đặt κ nội-người làm bảo đảm chất lượng cho đại lượng chính.
5. **Không đổi `analyse-kappa.py`.** Luật §4 giữ nguyên: đọc số được đổi **kết luận**, không được
   đổi **những gì script tính**. Sửa đổi này chỉ đổi cách **diễn giải** đầu ra.

### Khai báo: tôi đã thấy gì trước khi viết phụ lục này

Khi kiểm tra điều kiện thu thập, script đo đã in **mức trùng khớp thô giữa hai lần gắn: 23/25 =
92,0 %** (2 clip lệch nhãn). **Chưa** có κ nào được tính, **chưa** đọc đại lượng chính, **chưa**
mở OOF của mô hình. Ghi ra đây vì phụ lục này **không** được viết trong trạng thái mù hoàn toàn,
và nói thẳng thì rẻ hơn để phản biện phát hiện sau.

### Cách khôi phục điều kiện, nếu tác giả chọn

Gắn lại 25 clip lặp thành **pass 3**, sau **≥ 24 giờ**, phiên riêng, dòng mới append vào cùng tệp.
Khi đó κ nội-người (pass 1 ↔ pass 3) đo đúng độ tái lập, và D3 lấy lại toàn bộ quyền lực. Chi phí
≈ 10 phút gắn nhãn. Lựa chọn thuộc về tác giả; tiền-đăng-ký **không** bắt buộc.

### Không đổi

Vẫn tổng điều tra 151 nguồn, vẫn loại K096 ⇒ **n = 150**, vẫn hai hash đã chốt
(`events_sha256 dfeab129…1009`, `key_sha256 663d6605…ac94`), vẫn **D1–D5 nguyên văn**, vẫn
**tiêu 0 lần chạm test**, vẫn **bảng cổng không đổi — tám cổng, hai đạt, sáu trượt**.

---

## Sửa đổi 3 — 24/11/2026, TRƯỚC khi gắn clip đầu tiên của pass 3

Tác giả chọn **gắn lại 36 clip tự tin thấp, không giới hạn thời gian**. Phụ lục này chốt thiết kế
và luật diễn giải **trước khi có dữ liệu pass 3**.

### Khai báo trung thực về tình trạng của phụ lục này

1. **Nhóm 36 clip được chọn SAU KHI đã thấy kết quả** (κ nhóm này = 0,1486). Đây **không** phải
   phép thử mù. Hệ quả cứng: pass 3 **không được dùng để tính lại đại lượng chính**. κ = 0,6933
   trên n = 150 **giữ nguyên như đã công bố**. Pass 3 là **phép dò cơ chế trên một nhóm con đã
   chọn hậu nghiệm**, và phải được gọi đúng tên đó trong luận văn.
2. **Người gắn nhãn đã bị nhiễm thông tin.** Báo cáo ngày 17/11 đã nói cho tác giả biết chiều của
   lệch hệ thống (17 bỏ sót vs 6 báo động giả). Vì vậy mức tăng ở nhóm thử nghiệm có **hai** cách
   giải thích: phân xử được clip, **hoặc** chỉ dịch ngưỡng sang gọi hazard nhiều hơn. **Nhóm đối
   chứng tồn tại để tách hai cái này** — chúng dự đoán khác nhau về đối chứng (xem E1 vs E1b).

### Thiết kế

| | |
|---|---|
| Nhóm **thử nghiệm** | 36 clip có `confidence ∈ {1,2}` ở pass 1. Gốc **16 hazard / 20 normal**. Pass 1: đúng **22/36 = 61,1 %**, gọi hazard **4**, bỏ sót 13, báo động giả 1 |
| Nhóm **đối chứng** | 36 clip `confidence = 3`, chọn xác định bằng `sha256("pass3-20261124\|" + kid)`. Gốc **18/18**. Pass 1: đúng **32/36 = 88,9 %**, gọi hazard 18, bỏ sót 2, báo động giả 2 |
| Mốc tầm thường | trả "normal" cả 36 clip thử nghiệm ⇒ đúng **20/36 = 55,6 %**. Mốc so sánh là **55,6 %**, không phải 50 % |
| Trộn | xen kẽ xác định bằng `sha256(seed\|order\|kid)`; nhóm thử nghiệm rải đều, **không thành khối** ⇒ người gắn không biết clip nào thuộc nhóm nào |
| Tệp | `events-pass3.json`, 72 sự kiện, `sha256 f2b92b41…464b`, commit **trước** dòng dữ liệu đầu tiên |
| Khoảng cách | pass 1 là **17/11**, pass 3 là **24/11** ⇒ **7 ngày**. Đây là test-retest **thật**, thay thế con số 0,8397 đã bị thổi phồng |
| Sổ tay | **đóng băng**, `sha256 dc7784c70fd4c1b0…`, không đổi một chữ. Tác giả **đọc lại y nguyên** R1–R10 trước khi bắt đầu, **không** sửa R4 |

**Vì sao không sửa R4 dù đã biết nó lệch:** sửa định nghĩa sau khi thấy nhãn gốc là **khớp dụng cụ
đo vào đáp án**. Khoảng lệch định nghĩa vẫn là **giới hạn chưa giải quyết**, khai ở phụ lục, không
vá bằng cách chỉnh sổ tay.

### E0 — kiểm tra phép can thiệp, xét TRƯỚC mọi luật khác

Dwell trung vị của nhóm thử nghiệm ở pass 1 đã là **11,2 s**. Để kết luận "đã bỏ áp lực thời gian",
pass 3 phải có **dwell trung vị nhóm thử nghiệm ≥ 20 s**. Nếu không đạt ⇒ phép can thiệp **thất
bại**, E1/E1b/E2 **không áp dụng**, pass 3 chỉ báo cáo được dưới dạng test-retest 7 ngày.

### Luật diễn giải — chốt TRƯỚC

L₁ = 61,1 % · C₁ = 88,9 %. L₃, C₃ = độ chính xác pass 3 của hai nhóm.

| Mã | Điều kiện | Kết luận **buộc phải** viết |
|---|---|---|
| **E1** | L₃ − L₁ ≥ **+15 pp** **và** C₃ ≥ C₁ − 5 pp | **Phân xử thật.** Thời gian/chú ý là nguyên nhân, không phải nhập nhằng. ⇒ **trần nhãn CAO HƠN 84,67 %**, mô hình **còn dư địa**, lập luận "sát trần nhãn" ở §6 **YẾU ĐI và phải viết lại**. *Bất lợi cho luận văn.* |
| **E1b** | L₃ − L₁ ≥ +15 pp **và** C₃ < C₁ − 5 pp | **Chỉ dịch ngưỡng, không phân xử.** Tăng ở nhóm thử nghiệm đến từ việc gọi hazard nhiều hơn ở mọi nơi. **Không** nâng ước lượng trần nhãn; báo cáo như một minh chứng điểm làm việc của người đọc đã dịch. §6 **không đổi** |
| **E2** | \|L₃ − L₁\| ≤ 10 pp **và** \|C₃ − C₁\| ≤ 5 pp | **Nhập nhằng thật được xác nhận bằng phép can thiệp.** Thời gian không giới hạn **không** giải được những clip đó ⇒ lập luận "84.2 % sát trần nhãn" **được chống đỡ mạnh**, nay bằng can thiệp chứ không chỉ bằng tương quan |
| **E4** | C₃ ≤ C₁ − 10 pp | Người đọc đã **trôi** khỏi sổ tay. **Toàn bộ pass 3 không diễn giải được**, kể cả nhóm thử nghiệm |
| **E3** | mọi trường hợp khác | báo cáo số, **không** nâng thành tuyên bố nào mạnh hơn số cho phép |

### Báo cáo bất kể mã nào nổ

- **κ(pass 1 ↔ pass 3) trên cả 72 clip** — test-retest 7 ngày, là **con số độ tin cậy chính thức**
  thay cho 0,8397. Nếu **< 0,80** thì **D3 nổ** và áp dụng nguyên văn.
- **Tỉ lệ gọi hazard** hai nhóm (pass 1: thử nghiệm **4/36**, đối chứng **18/36**) — quy tắc mặc
  định "không chắc thì chọn bình thường" còn sống dưới thời gian không giới hạn hay không.
- Dwell trung vị hai nhóm.

### Không đổi

Đại lượng chính **κ = 0,6933 trên n = 150**, **D1–D5 nguyên văn**, **D2 vẫn là phán quyết đang
hiệu lực**, **tiêu 0 lần chạm test**, và **bảng cổng không đổi — tám cổng, hai đạt, sáu trượt**.
