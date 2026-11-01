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
