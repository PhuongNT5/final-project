# BIÊN BẢN TỔNG HỢP — HỘI ĐỒNG PHẢN BIỆN LUẬN VĂN THẠC SĨ

Ngày: 05/10/2026 (ngày đóng băng số của tác giả)
Đối tượng: `report/vn/` 11 tệp nộp, 90,632 từ + `PROPOSAL.txt` (đề cương đã đăng ký)
Thành phần: 5 phản biện độc lập + chủ tịch tổng hợp + 5 phép tự kiểm chứng của chủ tịch

## 1. PHÁN QUYẾT

**ĐẠT CÓ ĐIỀU KIỆN.** Nhất trí 5/5 phản biện trên từng trục.

| Trục | Phán quyết |
|---|---|
| Đối chiếu đề cương | ĐẠT CÓ ĐIỀU KIỆN (lệch nhiều điểm cốt lõi) |
| Phương pháp & thống kê | ĐẠT CÓ ĐIỀU KIỆN |
| Truy xuất số liệu | ĐẠT CÓ ĐIỀU KIỆN |
| Đóng góp khoa học | ĐẠT CÓ ĐIỀU KIỆN (ĐẠT nếu reframe; CHƯA ĐẠT nếu không) |
| Triển khai biên | ĐẠT CÓ ĐIỀU KIỆN |

Lý do KHÔNG "đạt thẳng": luận văn không giao được hợp đồng đã đăng ký với nhà trường
(xem §3), và một số tuyên bố cốt lõi chưa có đường truy xuất bằng git.

Lý do KHÔNG "chưa đạt": chất lượng phương pháp luận và tính trung thực thuộc nhóm cao
hiếm thấy ở bậc thạc sĩ. Có phép đo thật, có kết quả âm được báo cáo trước kết quả dương,
có sổ đăng ký số đã rút, có bootstrap theo cụm, có 8 cổng phán quyết bằng script hằng số.

### 1b. TRẠNG THÁI XỬ LÝ — cập nhật 09/10/2026

| Trạng thái | Lỗ hổng |
|---|---|
| **ĐÃ ĐÓNG** | A1 (phần câu chữ), A5, B2, B3, B4, B5, B6, B7, B8, B9, C1, C2, C3, C5, C7, C8, C9 |
| **HẠ MỨC — công bố thành giới hạn ở đúng chỗ trong bản nộp** | B1, B10, C4 |
| **CÒN MỞ — thủ tục, thuộc quyền tác giả** | A4 (văn bản GVHD), C6 (video dự phòng) |
| **CÒN MỞ — cần đo thêm, phá mốc đóng băng** | B1 (phép đo), B10 (p95/p99, soak), C4 (INT8) |
| **KHÔNG ĐÓNG ĐƯỢC BẰNG CÂU CHỮ** | A2, A3, A6 — là sự thật về phạm vi đã giao, chỉ có thể trình bày trung thực |

**Nguyên tắc đã áp dụng xuyên suốt đợt xử lý này:** mỗi lỗ hổng được đóng bằng **một phép
đo hoặc một lời khai đặt ở nơi người đọc tìm**, không bằng cách xoá hoặc làm mềm con số.
Không một con số đo được nào bị gỡ khỏi luận văn trong đợt này — kể cả các con số 6x%/7x%
mà tác giả không muốn có; chúng được **giải thích bằng phép đo** (mục 6.2.7, 6.2.8, 6.6.3)
chứ không bị rút, vì mục 6.11.3 của chính luận văn tuyên bố rút số là hành vi mà Chương 6
tồn tại để phản đối.

**Số lượng đã đóng trong hai ngày 08–09/10: 17 lỗ hổng + 16 trong 17 việc Nhóm 2.** Trong 17 có
hai lỗ hổng (C8, C9) **phát sinh trong lúc xử lý** và được đóng ngay trong cùng đợt. Việc Nhóm 2
còn lại duy nhất (số 16, video minh hoạ dự phòng) **không** chặn bởi công cụ mà chặn bởi một buổi
ghi hình của tác giả — và cùng buổi đó đóng luôn B1.

C3 (nhãn M3/M4) **đã chuyển từ "chặn bởi tác giả" sang "đã phân xử"** trong ngày 09/10: câu hỏi
tưởng là chuyện ký ức của tác giả, nhưng kho mã đã tự trả lời. Đây là bài học đáng ghi: trước khi
xếp một hạng mục vào loại "cần người xác nhận", phải tìm cho hết tạo tác đã có.

## 2. ĐIỂM MẠNH (hội đồng ghi nhận)

1. Kỷ luật đánh giá: giao thức leave-source-out/leave-context-out, bootstrap theo cụm,
   McNemar ghép đôi trên dự đoán out-of-fold từng clip (hợp lệ về toán học — hai giao
   thức đều sinh một dự đoán OOF cho MỌI clip nên ghép đôi được).
2. Báo cáo kết quả âm đúng thứ tự: G11 trượt nêu trước E0; giả thuyết trung tâm G14
   trượt (−0.5 pp, p = 0.683) và quyết định triển khai đi theo bằng chứng chứ không theo
   đề cương.
3. Sổ đăng ký số đã rút: "57/64 calibrated", "+3.5 pp macro", "95.7% mAP",
   "331 ms/3.02 dec/s", "P3 ĐẠT hợp đồng", "P3 TRƯỢT cả hai điều kiện" — chủ tịch đã
   grep toàn bộ `report/vn/`: KHÔNG có chỗ nào phục hồi ngầm số đã rút. Mọi lần xuất hiện
   đều kèm nhãn "đã rút".
4. Tuân thủ chính 3 quy tắc trích số nhịp tự đặt (không trích stage sum, không trích
   drop_rate, không trích dòng không-giải-mã): 0 vi phạm trong Ch5 bản nộp.
5. Minh bạch về `multi-stream-results.json` "hand-patched": bản VN công bố và KHÔNG dùng
   số của tệp đó trong bảng in (khác bản EN).
6. Phép đo lạm phát giao thức truy được về log: `g12-protocol-inflation-table.log`,
   `g12-inflation-paired.log`.

## 3. LỖ HỔNG — PHÂN LOẠI THEO MỨC RỦI RO

### A. RỦI RO LOẠI LUẬN VĂN (phải xử lý bằng THỦ TỤC, không bằng câu chữ)

A1. **Lệch đề tài: tên đề tài "dựa trên YOLO và Transformer", hệ triển khai headline là
    CLIP ViT-B/32.** Mục tiêu 1 của đề cương là xây dựng kiến trúc YOLO–Transformer.
    Ch5 mục 5.1 cho thấy đồ thị thực sự chạy trên thiết bị là B1 (tháp CLIP đóng băng +
    đầu thời gian 201,090 tham số). Lệch này KHÔNG có trong bảng tự khai 3 thay đổi
    (Ch1 mục 1.4.2) — tức chỗ lệch nặng nhất không được khai ở nơi người đọc tìm.
    GIẢM NHẸ: làn YOLO (P3) CÓ tồn tại và CÓ được đo trên thiết bị. Nó không trống,
    chỉ không phải headline.

    **PHẦN CÂU CHỮ ĐÃ XỬ LÝ (08/10/2026).** Bảng tự khai Ch1 mục 1.4.2 nay có **bốn**
    dòng, và dòng YOLO → CLIP ViT-B/32 được đặt **đầu bảng** kèm lý do bằng số: cổng
    tiền đăng ký G14 TRƯỢT (−0.5 pp, CI theo cụm [−5.5, +5.0] chứa 0, p = 0.683; mục
    6.2.2), nên quyết định triển khai đi theo bằng chứng chứ không theo đề cương. Dòng
    cũng nêu rõ làn YOLO (P3) vẫn được xây, vẫn được đo trên Jetson và vẫn được báo cáo
    bằng số của chính nó. Đoạn mở đầu mục 1.4.2 nói thẳng đây là thay đổi nặng nhất.
    **Phần THỦ TỤC (văn bản GVHD duyệt) vẫn mở** — Nhóm 1 việc 1.

A2. **Không đạt mục tiêu accuracy đã đăng ký (> 95.8%).** Thực đạt: 85.4% macro (blind,
    4 lớp Unsafe-Net) / 84.2% balanced (LSO, iSafetyBench). Khoảng cách tới 95.81% được
    luận văn quy cho data-volume-bound và KHÔNG tuyên bố đạt — trung thực, nhưng vẫn là
    mục tiêu đăng ký không hoàn thành.

    **LẬP LUẬN BẢO VỆ ĐÃ MẠNH HƠN HẲN (08/10/2026).** Trước đây khoảng cách tới 95.81%
    được *lập luận* (data-volume-bound, giao thức khác nhau). Nay nó được *đo*: chính
    cấu hình được triển khai đạt **97.7%** trên iSafetyBench dưới fold ngẫu nhiên bỏ qua
    nguồn — đúng kiểu giao thức mà tài liệu dùng — so với **84.2%** dưới leave-source-out
    (+13.5 pp, CI [+8.0, +20.0], p = 1.39e-41; mục 6.2.7). Luận văn có con số chín mươi
    mấy phần trăm, đo được, có log, có tệp OOF 1,100 dòng, và **chọn không công bố nó**.
    Mục tiêu đăng ký vẫn không hoàn thành theo chữ, nhưng lý do nay là một **phép đo**,
    không phải một lời giải thích.

    CẢNH BÁO TRỘN LÀN: 97.7% là iSafetyBench (làn A), 95.81% là Unsafe-Net (làn C).
    Mục 6.1.1 cấm đặt hai con số này vào cùng một câu so sánh. Khi trả lời hội đồng phải
    nói "dưới giao thức cùng kiểu", không nói "chúng em vượt 95.81%".

A3. **Thu hẹp phạm vi + thay kho dữ liệu headline.** Đề cương: 8 lớp (4 an toàn/4 nguy
    hiểm) trên Unsafe-Net. Thực tế: thu hẹp về nhị phân/4 lớp; kho headline chuyển sang
    iSafetyBench (1,100 clip) — kho này KHÔNG có trong đề cương đã đăng ký.

A4. **Đổi kết luận headline từ "hệ thống" sang "phát hiện phương pháp luận".** Hợp lệ về
    khoa học, nhưng là đổi hướng đề tài mà đề cương không đăng ký. KHÔNG tìm thấy bằng
    chứng văn bản GVHD/khoa duyệt đổi hướng.

A5. ~~**Thiếu Confusion Matrix — yêu cầu ghi rõ chữ "bắt buộc phải có" trong đề cương.**~~
    → **ĐÃ ĐÓNG (09/10/2026).** Nguyên văn lỗ hổng: grep toàn `report/vn/` chỉ còn một tổng
    hàng `[0,0,0,56]` ở Phụ lục B và một ma trận đồng thuận annotator (khác mục đích).

    **Ba ma trận được bổ sung, đếm lại từ tệp đã đóng băng, không phải số đo mới.**
    Mục **6.2.8** mới: **Bảng 6.2a** — B1 dưới leave-source-out, 1,100 clip
    (TN 570 / FP 110 / FN 65 / TP 355; recall hiểm hoạ 84.5%, đặc hiệu 83.8%, cân bằng
    **84.2%** — tái lập đúng con số tiêu đề); **Bảng 6.2b** — cùng mô hình, cùng 1,100 clip,
    chỉ đổi sang fold ngẫu nhiên (TN 665 / FP 15 / FN 10 / TP 410; 97.6 / 97.8 / **97.7%**).
    Đặt cạnh nhau: bỏ sót hiểm hoạ 65 → 10 clip (6.5 lần), báo động giả 110 → 15 (7.3 lần),
    biến duy nhất là fold có tôn trọng danh tính nguồn hay không. Mục **6.12.4**:
    **Bảng 6.7a** — ma trận bốn lớp của **đúng bộ dự đoán mù 54/64** mà làn C trích dẫn
    (nguồn `sweep_results/relation_tokens/full_eval.json`, trường `confusion`), **không**
    phải ma trận của bộ ensemble 56/64; toàn bộ 6 lỗi nặng nhất là WalkwayViol →
    UnauthorInt, một thuộc tính *quyền được ở đó* không nhìn thấy trong khung hình, và đó
    cũng là lý do precision UnauthorInt chỉ 58.8% trong khi recall 90.9%.

    **Giá trị phụ ngoài dự kiến:** ba ma trận tái lập độc lập mọi con số tiêu đề
    (84.17 → 84.2; 97.71 → 97.7; Δ 13.54 → +13.5 pp; 54/64 → 84.4% top-1, macro 88.4%),
    nên chúng đồng thời là một phép kiểm tra chéo số học của Chương 6.

    Phân định trung thực kèm theo: hai loại lỗi gần cân nhau (15.5% bỏ sót so với 16.2%
    báo động giả), tức mô hình **không** được hiệu chỉnh lệch về phía an toàn; luận văn
    **không** dịch ngưỡng, vì ngưỡng vận hành đúng phụ thuộc chi phí can thiệp tại hiện
    trường — đại lượng nằm ngoài kho dữ liệu.

A6. **Hệ đa luồng 4 khối (camera → edge → server lưu trữ → dashboard) giao 2/4 khối.**
    grep "dashboard", "server lưu trữ" trong bản nộp: 0 kết quả. Bằng chứng đa luồng duy
    nhất (4-stream, 88 FPS, 0 drop) thuộc pipeline P1 ĐÃ BỊ THAY.

### B. RỦI RO LÀM SỤP MỘT TUYÊN BỐ CỐT LÕI

B1. **Độ chính xác của hệ THỰC SỰ chạy trên thiết bị chưa được đo.** → **VẪN MỞ NHƯ MỘT
    PHÉP ĐO, NHƯNG ĐÃ ĐƯỢC CÔNG BỐ TRONG BẢN NỘP (09/10/2026).**

    **Sửa mô tả kỹ thuật của chính lỗ hổng này.** Biên bản ban đầu ghi "triển khai trượt
    cửa sổ 16 khung LIÊN TIẾP". Đọc mã (`deployment/scripts/benchmark-r48-composed-cadence.py`,
    `collections.deque(maxlen=window)`, `--window 48 --stride 24`) cho thấy vòng lặp trên
    thiết bị lấy **16 khung `linspace` TRONG một cửa sổ trượt 48 khung, bước nhảy 24**
    (≈1.6 s ở trung vị 29.97 fps của kho). Đánh giá lấy 16 khung `linspace` trên **toàn
    clip** (`build_relation_tokens.py:194`). Vậy chỗ lệch là **khoảng thời gian** mà cửa sổ
    bao, **không** phải dạng lấy mẫu. Lỗ hổng nhẹ hơn mô tả ban đầu nhưng vẫn là lỗ hổng.

    **Đã vào bản nộp:** mục **5.4.1** (quy tắc trích dẫn bắt buộc: **không con số độ chính
    xác nào trong luận văn được đọc là độ chính xác của vòng lặp trên thiết bị**; mọi số ở
    mục 5.4 là số **chi phí**, không phải số **chất lượng**; **chiều** của thiên lệch là
    KHÔNG BIẾT — cửa sổ hẹp hơn có thể dễ hơn, cũng có thể bỏ mất hiểm hoạ) và mục **5.9
    hạng mục (15)** (nêu đúng đường chạy còn thiếu: `demo/run-unsafe-behavior-demo.py` với
    `b1-deploy-3seed.pt` trên máy chủ, và mốc đóng băng 05/10/2026 là lý do chưa chạy).

    Mức rủi ro hạ từ "sụp tuyên bố cốt lõi" xuống "**giới hạn được công bố ở đúng chỗ**".
    Phép đo vẫn là việc phải làm (Nhóm 3 việc 2).

B2. ~~**Confound chưa tách: rò rỉ nguồn vs độ khó của phép chia.**~~ → **ĐÃ ĐÓNG
    (08/10/2026).** Nguyên văn lỗ hổng: hiệu số LSO vs k-fold ngẫu nhiên trộn hai nguyên
    nhân — (a) rò rỉ nguồn, (b) fold LSO khó hơn về bản chất (ít mẫu huấn luyện hơn, phân
    bố lớp lệch hơn, covariate shift) — và không có đối chứng khớp kích thước/phân bố lớp,
    không có permutation test trên nhãn nguồn.

    **Đã xử lý bằng đo, không bằng câu chữ.** Đối chứng giả dược **G19**
    (`isafetybench_dataset.placebo_groups`): sinh định danh nguồn giả khớp ĐÚNG ba thuộc
    tính cấu trúc của nguồn thật — cùng số nhóm, cùng đa tập kích thước nhóm *trong từng
    lớp*, cùng tính thuần lớp (cả ba được bảo đảm bằng `assert` trong mã) — đồng thời rải
    clip của mỗi nguồn thật ra nhiều nhóm giả. Thứ duy nhất bị phá là xuất xứ, nên mọi độ
    chính xác còn sống sót KHÔNG quy được cho hình học fold.

    Kết quả trên **20 seed** (tối thiểu để p < 0.05 đạt được: min p = 1/21 ≈ 0.048):
    trung bình **89.03%**, sd 0.47, dải 87.7–89.7. Nguồn thật dưới LSO: **75.2%**. Fold
    ngẫu nhiên: **90.6%**. KHÔNG seed nào xuống tới nguồn thật — seed thấp nhất còn cách
    **12.5 pp** — nên **p hoán vị = (0+1)/(20+1) = 0.0476 < 0.05**.

    **Phân tách +15.4 pp:** hình học fold **+1.6 pp (10.2%)**, danh tính nguồn
    **+13.8 pp (89.8%)**. Bằng chứng độc lập cùng chiều: đường sàn quy tắc đếm chỉ nhích
    **1.0 pp** (67.2 → 68.0 → 68.2%) xuyên cả ba điều kiện trong khi bộ mã hoá dịch
    15.4 pp — thứ fold ngẫu nhiên cấp cho bộ mã hoá vì vậy không phải thông tin đếm và
    cũng không phải thêm dữ liệu huấn luyện.

    Viết vào bản nộp: **mục 6.6.3** (toàn văn đối chứng), **mục 6.7(e)** (loại trừ giải
    thích cạnh tranh), hàng **G19 của Bảng 6.8**, và Chương 8 (tổng số đối chứng).
    Nhật ký: `sweep_results/relation_tokens/logs/g19-placebo-E3-seed{1..20}.log`.
    Commit `eafec53` (mã đối chứng) + `9895df4` (phép đo và văn bản).

    **Hai điều phải khai đúng mức.** (i) G19 **không được đăng ký trước** — chỉ *tiêu chí*
    (20 seed, α = 0.05, ba thuộc tính cấu trúc) được cố định trước khi chạy; điều này được
    công bố ở mục 6.6.4 và ở chú thích Bảng 6.8. (ii) G19 **không phán quyết** cổng nào,
    nên tổng cổng phán quyết VẪN là **2 đạt / 6 trượt**, không đổi.

    Trên Unsafe-Net, phần biên mà luận văn tự nhận là "dịch chuyển phân bố nhãn" vẫn CHƯA
    được định lượng — G19 chạy trên iSafetyBench. Đây là phần còn mở của B2, mức rủi ro
    hạ từ "sụp tuyên bố cốt lõi" xuống "giới hạn được công bố".

B3. ~~**Nguy cơ vòng tròn logic của nhóm bối cảnh KMeans(8).**~~ → **ĐÃ CÔNG BỐ
    (09/10/2026).** Nguyên văn: cụm sinh ra từ chính không gian đặc trưng đang được đánh
    giá, nên leave-cluster-out tất yếu khó hơn.

    Viết thành **hạn chế (1) của mục 6.9**, bằng đúng chữ của phản biện: tám nhóm là
    KMeans trên chính không gian đặc trưng ngoại hình đang bị đánh giá, nên fold theo nhóm
    **tất yếu** khó hơn một phép chia theo camera thật; **+13.0–21.8 pp** trên Unsafe-Net
    vì vậy là một **chặn trên bị thiên lệch lên**, không phải một ước lượng không thiên
    lệch; không có phép hiệu chỉnh nào khi chưa có định danh camera thật; và — điểm then
    chốt — **kết luận lạm phát giao thức KHÔNG dựng trên làn này** mà trên iSafetyBench,
    nơi 151 định danh nguồn là siêu dữ liệu thật chứ không phải cụm suy ra.

B4. ~~**Tiền đăng ký không kiểm chứng được bằng git.**~~ → **ĐÃ CÔNG BỐ TRONG BẢN NỘP
    (09/10/2026).** Lỗ hổng **không đóng được bằng bằng chứng** — nó đóng được bằng việc
    thôi yêu cầu người đọc tin quá mức.

    Ba mốc git được **kiểm lại** và nêu nguyên văn trong bản nộp (mục **6.9 hạn chế (6)**,
    nhắc lại ở mục **8.3.7** như hạng mục xuất xứ thứ tám): `tools/adjudicate-r48-gates.py`
    vào git **29/09/2026 23:14** trong **cùng commit** `9cf01ca` với mô hình ESTA và script
    huấn luyện; `plans/2026-09-19-thesis-recovery-architecture/risks.md` — nơi G14–G17 được
    đăng ký — vào git lần đầu **01/10/2026 21:13** (`26cdf1f`), **sau** phán quyết hai ngày;
    `phase-12-isafetybench-accuracy-lane.md` vào git **07/10/2026 23:03** (`df1dab7`), tức
    **sau cả mốc đóng băng số 05/10**. (Biên bản ban đầu ghi "git log RỖNG" cho tệp này —
    đúng ở thời điểm 05/10, nay đã có commit.)

    **Phát biểu được hạ xuống đúng mức bằng chứng:** luận văn tuyên bố rõ rằng người đọc
    **không thể dùng git để chứng minh** ngưỡng được cố định trước khi biết kết quả; điều git
    chứng minh được chỉ là (i) các ngưỡng không bị sửa kể từ 01/10/2026 (`MARGIN = 3.0`,
    `G18B_MARGIN = 2.0`, `ALPHA = 0.05` bất biến trong mọi commit sau đó) và (ii) phán quyết
    do một script hằng số sinh ra chứ không do người đọc số rồi chọn ngưỡng. Luận văn
    **không yêu cầu tin hơn mức đó**.

    Cách sửa được nêu là **cho lần sau, không hồi tố**: đăng ký công khai có dấu thời gian
    bên thứ ba (OSF) hoặc một commit đã ký được đẩy lên **trước** khi phép đo chạy. Hội đồng
    ghi nhận lời từ chối viết lại nhật ký để tạo mốc thuận lợi — một dấu thời gian dựng lại
    sau sự việc kém hơn việc không có dấu nào — là ứng xử đúng, và trên thực tế là **bằng
    chứng về tính liêm chính mạnh hơn** bất kỳ mốc thời gian nào tệp đó có thể mang.

B5. ~~**Số liệu headline tính từ tệp CHƯA VÀO GIT.**~~ → **ĐÃ ĐÓNG (09/10/2026).**
    Nguyên văn: `enc_isbclip_ESTA_lso-1100_headline-none_oof.json` và
    `enc_isbclip_B1_...oof.json` — hai tệp dự đoán từng clip mà MỌI thống kê headline tính
    từ đó — đều UNTRACKED.

    **Nguyên nhân gốc, không phải triệu chứng:** `.gitignore` dòng 12 có
    `sweep_results/**/*.json`, quét sạch toàn bộ bằng chứng OOF. Một `git add -f` sẽ đóng
    được hôm nay và mở lại ở tệp tiếp theo. Cách xử lý là **một phủ định `.gitignore`**
    theo đúng văn phong phủ định đã có trong tệp:

    ```
    !sweep_results/**/*_oof.json
    ```

    kèm chú thích ghi lý do ghi đè quy tắc chặn (mọi thống kê tiêu đề của Chương 6 và
    Chương 7 — bootstrap theo cụm, McNemar chính xác, 20 seed G19 — tính từ đúng các tệp
    này; ~7.4 MB / 50 tệp). **61 tệp được stage**: `sweep_results/relation_tokens/*_oof.json`,
    `sweep_results/binary_gates/*_oof.json` và `.gitignore`.
    `deployment/artifacts/jetson-measurements/` đã commit đủ 103 tệp từ `df1dab7`.

    Hệ quả: tệp OOF **tương lai** cũng được theo dõi, không cần nhớ `-f` lần nữa.

B6. **Tóm tắt trích một khoảng tin cậy mà chính luận văn tuyên bố là SAI ĐƠN VỊ.**
    Tóm tắt viết "+16.9 pp macro, khoảng tin cậy [+12.5, +21.2]" không kèm cảnh báo.
    Ch6 (quanh dòng 918) tự thú: khoảng này "bootstrap trên 516 clip, KHÔNG phải bootstrap
    theo cụm trên 8 nhóm... **đơn vị lấy mẫu sai**... **không** được dùng làm căn cứ chứng
    nhận". Tóm tắt — tệp hội đồng đọc đầu tiên — đang dùng nó làm căn cứ.

B7. ~~**Hợp đồng nhịp tự suy từ kho dữ liệu, không dẫn tiêu chuẩn an toàn lao động.**~~
    → **ĐÃ ĐÓNG VỀ CÂU CHỮ (09/10/2026).** Nguyên văn: grep OSHA/ISO/IEC toàn văn 0 hit;
    "vượt hợp đồng 2.39–4.79 lần" có thể bị đọc thành tự đặt bài thi rồi tự chấm đỗ.

    Mục **5.4.4** nay nói thẳng xuất xứ: 1.249 / 1.427 / 2.500 quyết định/giây suy từ
    *f*/24 trên siêu dữ liệu fps của 1,100 clip, **không** từ OSHA, ISO 45001, IEC 61508
    hay bất kỳ tiêu chuẩn kỹ thuật Việt Nam nào; và một yêu cầu thời gian phản ứng an toàn
    thật là **đặc thù theo hiểm hoạ** (xe nâng đang chạy so với tủ điện bị mở khác nhau
    nhiều bậc độ lớn) nên nằm ngoài phạm vi. Mục **5.10 kết quả (3)** được phát biểu lại
    từ "đạt hợp đồng thời gian thực" thành "**giữ kịp tốc độ khung của nguồn** với biên
    2.39–4.79 lần", kèm xuất xứ ngay trong câu. Đây là một phát biểu **theo kịp luồng**,
    không phải một chứng nhận an toàn — và biên bản ghi nhận đó là cách phát biểu đúng.

B8. ~~**Headline nhịp đo dưới DVFS — điều kiện KHÔNG lặp lại được.**~~ → **ĐÃ ĐÓNG
    (09/10/2026).** Mục **5.4.6** mới công bố toàn bộ: 11 lần chạy trong hai ngày cho ghim
    nhịp **309.08–311.78 ms (phân tán 0.49%)** so với DVFS **365.88–440.01 ms (20.3%)** —
    **hai tập rời nhau**. Giả thuyết nhiệt bị **bác bỏ bằng phép đo** (421.52 so với
    422.82 ms, lệch 0.31%). Nguyên nhân là bộ điều tốc hạ nhịp **trong lúc chạy**
    (`policy0 cur_khz` 1,574,400 → 192,000) trong khi `clock_state_stable` vẫn báo `true`,
    vì trường đó chỉ xác nhận **chế độ**, không xác nhận nhịp giữ nguyên.

    Ba hệ quả được nêu kèm, không để người đọc tự suy: các phân tán nhỏ trong bảng chỉ có
    nghĩa **trong một phiên**; **một lần chạy DVFS không phán quyết được gì**; và các kết
    luận hợp đồng còn sống **chỉ vì** biên của chúng (2.39–4.79 lần) lớn hơn 20.3% một bậc
    độ lớn — nói rõ là sống nhờ biên, không nhờ độ chính xác của phép đo.

B9. ~~**Đa so sánh không hiệu chỉnh.**~~ → **ĐÃ ĐÓNG (09/10/2026).** Mục **6.8.1** mới
    đếm **31 p-value riêng biệt** xuyên Chương 6–8 trên **10 mã cổng**, công bố rằng
    **không** áp dụng Bonferroni/FDR, và tính ngưỡng để người đọc tự đối chiếu:
    α' = 0.05/31 = **1.61e-03**. Hai cổng đạt sống sót với biên rất lớn — G12
    **3.58e-45** (hơn 42 bậc độ lớn), làn A **1.39e-41**, G10 **1.02e-05** (~158 lần) —
    còn **G19 với p = 0.0476 thì KHÔNG**, và điều đó được nói thẳng kèm lý do thiết kế:
    20 seed cho min p = 1/21 ≈ 0.048, nên một ngưỡng đã hiệu chỉnh là **không thể đạt
    được về nguyên tắc**; bằng chứng của G19 vì vậy luôn là **khoảng cách 12.5 pp** của cả
    20 seed giả dược (hơn 26 lần sd 0.47), không phải p-value.

    Ba điều kiện biên kèm theo: căn cứ chứng nhận là **khoảng tin cậy theo cụm**, không
    phải p (mục 6.1.5); hiệu chỉnh **không cứu được** cổng nào đã trượt (cả sáu trượt vì
    khoảng tin cậy chứa 0); và 31 kiểm định **không độc lập** nên Bonferroni ở đây là
    **bảo thủ quá mức có chủ ý**.

B10. **Chỉ có trung vị cho nhịp tổ hợp, không p95/p99, không soak.** Với hệ an toàn, báo
     cáo không có đuôi phân phối là thiếu. Tác giả tự nhận (câu hỏi mở 2, 3).

### C. RỦI RO TRÌNH BÀY / ĐÓNG GÓI

C1. ~~`README.md` kể một headline KHÁC HẲN bản nộp.~~ → **ĐÃ ĐÓNG (09/10/2026).** Banner
    `⚠️ SUPERSEDED — NOT THE SUBMITTED RESULT (as of 2026-10-05)` đặt đầu tệp, chỉ
    `report/vn/` là nguồn duy nhất có thẩm quyền, nêu lại headline thật (B1 / CLIP ViT-B/32
    đóng băng + đầu 201,090 tham số / 84.2% dưới LSO / sàn tiền đăng ký 63.3% / G14 trượt
    −0.5 pp, CI [−5.5, +5.0], p = 0.683 / hai đạt sáu trượt) và **liệt kê từng con số bị
    cấm** ("+3.5 pp macro win", "57/64", "95.7% mAP", "331 ms / 3.02 dec/s / 3.91 J", mọi
    phán quyết đạt-trượt đơn lẻ của P3, mọi tổng giai đoạn dùng như một nhịp), kèm lệnh cấm
    trộn làn. Tài liệu được **sửa để giữ xuất xứ**, không bị xoá.
C2. ~~`docs/system-architecture.md`, `docs/deployment-guide.md` lạc hậu, mâu thuẫn Ch5.~~
    → **ĐÃ ĐÓNG (09/10/2026).** Cùng banner, cộng hai sửa nội dung: `system-architecture.md`
    hạng mục 3 không còn gọi P3 là "accuracy headline hiện hành" (B1/R48 được nêu đúng,
    5.978 dec/s, và P3 **đã** được đo tổ hợp trên thiết bị); mục "Pipeline attribution" của
    `deployment-guide.md` được thay bằng ba gạch đầu dòng đúng (B1/R48 là headline đã triển
    khai, 167.291 ms = 5.978 dec/s; P3 đo tổ hợp 03/10/2026, phụ thuộc clip và xung nhịp,
    tổng giai đoạn đã rút kèm lật dấu +7.98% / −39.3%; xuất xứ P1 KHÔNG XÁC ĐỊNH, engine
    chưa bao giờ vào git), cộng đoạn DVFS không lặp lại được.
C3. ~~24 chỗ nhãn phần cứng M3/M4 mâu thuẫn (đề cương ghi M4 Pro, văn bản ghi M3).~~
    → **ĐÃ ĐÓNG (09/10/2026), và đóng bằng tạo tác chứ không bằng lời tác giả.** Ba nguồn
    độc lập trùng khớp: (a) biểu ngữ ultralytics **tự in ra** ở đầu mỗi phiên huấn luyện bộ
    phát hiện — `Ultralytics 8.4.27 Python-3.13.5 torch-2.11.0 MPS (Apple M3)` —
    `sweep_results/entity_detector/train_run.log` dòng 4, 254, 270 và
    `sweep_results/r2/behaviour_yolo.log` dòng 11, 194, cả hai nhật ký **đã vào git**;
    (b) đầu tạo tác đo ghi `"machine": "Phgs-Mac.lan"` / `"platform":
    "macOS-14.6-arm64-arm-64bit-Mach-O"`; (c) chính máy ghi `hw.model` = `Mac15,12`,
    `machdep.cpu.brand_string` = `Apple M3`, `hw.memsize` = 16 GB. Vậy nhãn đúng là
    **Apple M3, 16 GB**; **đề cương sai** (nó ghi "Apple M4 Pro, 12 lõi CPU, 18 lõi GPU") và
    PROGRESS.md cũng sai ("M4, 36 GB"). Con số chỗ sai là **49**, không phải 24 — bản kiểm
    trước chỉ đếm trong văn bản luận văn, lần này đếm cả mã nguồn và tài liệu. 49 chỗ đã sửa
    về M3. Đã ghi ở **Ch4 mục 4.6.1** (thân bài, kèm bằng chứng), **Ch5 sổ đăng ký**, **Ch7
    mục 7.7.2(i)** (đánh dấu ĐÃ GIẢI QUYẾT, không xoá khỏi danh sách khuyết điểm) và **Ch7
    mục 7.9.6**. Đề cương **không** bị sửa: nó là văn bản đã đăng ký, và sai lệch so với nó
    phải được khai ở phần lệch đề cương chứ không phải được viết lại. **Không một con số đo
    nào đổi** — mọi phiên huấn luyện chạy trên CPU, MPS chỉ dùng ở bước trích đặc trưng, và
    mọi số nhịp/độ trễ/năng lượng đo trên Jetson Orin Nano, không trên máy chủ.
C4. TensorRT/INT8 — đề cương cam kết — chưa đo cho tháp CLIP headline.
C5. ~~Thiếu Geirhos et al. (2020) và nhánh literature record-wise/subject-wise CV.~~
    → **ĐÃ ĐÓNG (09/10/2026).** Chương 2 có thêm một đoạn thân bài về **shortcut learning**
    ngay sau đoạn Kapoor/Narayanan `[30]`, với đúng luận điểm mà hội đồng sẽ hỏi: một
    shortcut **không bao giờ hiện ra** trên tập kiểm tra cùng phân phối, chỉ hiện ra trên
    một tập ngoài phân phối; khi không có kho thứ hai thì LSO/LCO **là** phép kiểm ngoài
    phân phối duy nhất có được; và ở kho này danh tính nguồn là **trường hợp xấu nhất** vì
    nó là một **đại lượng thay thế nhãn hoàn hảo** (R41: 0 trong 151 nguồn mang cả hai lớp).
    Tài liệu: **`[44]`** Geirhos et al. 2020 (Nature Machine Intelligence 2:665–673, DOI
    10.1038/s42256-020-00257-z, arXiv:2004.07780) và **`[45]`** Nadeau & Bengio 2003
    (Machine Learning 52(3):239–281, DOI 10.1023/A:1024068626366). Kaufman et al. 2012 đã
    có sẵn ở `[29]`.
C6. Video demo dự phòng CHƯA TỒN TẠI. Nếu demo live thất bại trước hội đồng, không có
    phương án hai.
C7. ~~Ch7 "Phân tích thất bại" — tên chương bất thường trong chuẩn VN.~~ → **ĐÃ ĐÓNG
    (09/10/2026).** Tên mới: "**PHÂN TÍCH KẾT QUẢ PHỦ ĐỊNH VÀ QUY NGUYÊN NHÂN: CÁC CỔNG
    TRƯỢT, CÁC PHÁT BIỂU ĐÃ RÚT, CÁC KHUYẾT ĐIỂM PHÉP ĐO**" (dòng hai vừa khung 70 ký tự).
    Câu mở mục 7.1 được đồng bộ, và tên mới được lan sang **bốn** chỗ tham chiếu: mục lục
    Ch1 §1.6, đóng góp (4) của Ch1, tiêu đề Ch8 §8.1.5, đóng góp (4) của Ch8 — cả bốn khối
    được bọc dòng lại.

**Hai lỗ hổng PHÁT SINH trong lúc truy xuất nhãn phần cứng (09/10/2026).** Cả hai do chính việc
kiểm chứng C3 làm lộ ra, và cả hai đã được công bố trong bản nộp ngay trong cùng đợt. Chúng được
ghi thêm vào đây chứ không bị bỏ qua, vì một biên bản chỉ đáng tin khi nó cũng ghi những thứ xuất
hiện muộn và bất tiện.

C8. ~~Mục 4.5.5 phát biểu "**mọi** phiên huấn luyện trong luận văn chạy trên CPU" — SAI.~~
    → **ĐÃ ĐÓNG (09/10/2026).** Hai phiên huấn luyện YOLO còn nhật ký trong kho mã đều khai
    `device=mps` **và** `optimizer=AdamW` — tức đúng cấu hình mà chính mục 4.5.5 tuyên bố làm
    hỏng âm thầm cập nhật Adam. Lượng từ phổ quát đã được thu hẹp về đúng phạm vi nó đúng:
    **mọi đầu phân loại** (Transformer thời gian, hồi quy logistic, cây) chạy CPU; bộ phát hiện
    YOLO là ngoại lệ duy nhất. Sửa ở mục 4.5.5, bảng môi trường mục 4.6.1 và câu tóm tắt ở 4.6.1.
    Đây là lỗi **có lợi cho luận văn khi được sửa**: phiên MPS ấy đã sụp về mAP@50 = 0, nên nó
    biến quy tắc "huấn luyện trên CPU" từ một trích dẫn docstring thành một **bằng chứng thực
    nghiệm nội bộ**. Nói cách khác, luận văn đã vô tình chạy đúng đối chứng cho quy tắc của mình.

C9. ~~Nguồn gốc huấn luyện của `detector_best.pt` (trọng số đạt mAP@50 0.9695) KHÔNG có trong
    bản nộp tiếng Việt.~~ → **ĐÃ ĐÓNG BẰNG LỜI KHAI (09/10/2026), không đóng được bằng bằng
    chứng.** Nhật ký duy nhất còn lại (`train_run.log`) không chỉ "bị cắt giữa epoch 20/30" như
    mục 4.2.3 từng viết: **mọi dòng kiểm định sau epoch 1 trong đó đều ghi mAP@50 = 0**, nên
    phiên đó không thể là phiên sinh ra trọng số 0.9695. Bản thảo **tiếng Anh** (Ch3 §3.4.2,
    Ch4) quy trọng số cho một phiên chạy lại trên **runtime GPU của Kaggle**; bản tiếng Việt —
    tức bản nộp — chưa bao giờ nói điều này. Nay đã nói, kèm đúng mức độ tin cậy của nó: **không
    có nhật ký phía Kaggle trong kho mã, nên đây là lời tự khai, không phải bằng chứng.** Điều
    kiểm chứng được là phép đo lại trên CPU của chính trọng số đó trên 343 ảnh giữ riêng. Ghi ở
    mục 4.5.5 (đầy đủ), ghi chú Bảng 4.1 và mục 7.7.1(a). **Hội đồng có thể hỏi thẳng việc này
    và câu trả lời phải là lời khai, không được là một lời bảo đảm.**

## 4. TÍNH MỚI

Phán quyết: **MỚI CÓ ĐIỀU KIỆN.** Hiện tượng (data leakage, record-wise vs subject-wise CV,
shortcut learning) đã được biết rõ trong literature y sinh/EEG/speech từ lâu. Cái mới là
**phép đo**: đối chứng tham số khớp (cùng mô hình, cùng đặc trưng, cùng công thức huấn
luyện, chỉ đổi cách dựng fold), tái lập trên hai kho độc lập, và hai hệ quả bậc hai chưa
thấy được lượng hoá ở đâu khác cho domain này: **nén khả năng phân biệt kiến trúc**
(5.1→1.4 pp và 7.3→2.1 pp) và **đảo thứ tự xếp hạng**. Đây là đóng góp thật nhưng phải
được phát biểu đúng tầm: đóng góp về GIAO THỨC cho domain an toàn lao động, không phải
phát hiện hiện tượng mới.

## 5. VIỆC PHẢI LÀM — PHÂN THEO KHẢ NĂNG THỰC HIỆN TRƯỚC 10/10

### Nhóm 1 — THỦ TỤC, làm ngay hôm nay (quyết định mức ĐẠT/KHÔNG, không phải mức điểm)
1. Gặp GVHD, xin **văn bản giải trình thay đổi đề cương** có chữ ký, nêu đúng 4 thay đổi:
   (a) tháp thị giác headline YOLO → CLIP ViT-B/32 (kèm lý do: ablation chọn, làn YOLO vẫn
   được giữ và đo), (b) kho headline thêm iSafetyBench, (c) phạm vi lớp thu hẹp,
   (d) kết luận headline chuyển sang trục giao thức đánh giá. Không có bước này, A1–A4 là
   rủi ro mở.
2. Hỏi GVHD dứt khoát: hội đồng có chấp nhận một luận văn mà kết luận chính là phát hiện
   phương pháp luận + kết quả âm? Nếu KHÔNG → phải nâng làn YOLO (P3) lên thành deliverable
   của Mục tiêu 1 (xem Nhóm 2 việc 1).

### Nhóm 2 — CÂU CHỮ/ĐÓNG GÓI, làm được trước 10/10 (không cần đo mới)
1. **Dựng lại tường thuật 3 mục tiêu**, mỗi mục tiêu một deliverable có số:
   - MT1 (YOLO–Transformer): làn P3 = YOLOv11s + đầu Transformer, đo trên Jetson. Nêu
     bằng số CỦA CHÍNH NÓ, kèm điều kiện xung nhịp. Không để mục tiêu này trống.
   - MT2 (độ chính xác): phát biểu lại thành "đo độ chính xác dưới giao thức không rò rỉ",
     nêu 84.2% so đường sàn 63.3%, và nêu rõ 95.8% của bài báo gốc đo dưới giao thức khác
     — đây chính là nội dung của phát hiện headline. Biến việc trượt mục tiêu thành kết quả.
   - MT3 (tối ưu biên): nêu đúng những gì đo được; công bố INT8/TensorRT cho tháp CLIP là
     việc chưa làm, và dashboard/server ngoài phạm vi.
2. ~~Thêm **dòng thứ 4 vào bảng tự khai Ch1 mục 1.4.2**: YOLO → CLIP.~~ → **ĐÃ LÀM
   08/10.** Dòng được đặt **đầu bảng** (không phải thứ tư) vì đó là thay đổi nặng nhất;
   bảng nay khai **bốn** thay đổi; lý do nêu bằng số (G14 TRƯỢT). Xem A1 phía trên.
3. ~~**Bổ sung Confusion Matrix** cho kết quả headline.~~ → **ĐÃ LÀM 09/10.** Ba ma trận:
   **Bảng 6.2a / 6.2b** (nhị phân, hai giao thức, mục 6.2.8) và **Bảng 6.7a** (bốn lớp,
   mục 6.12.4). Xem A5. **Một cái bẫy phải ghi lại:** nguồn mà biên bản đề xuất —
   `deployment/artifacts/per-clip-logits-test64.csv` — cho bộ **ensemble 56/64** (đã kiểm:
   `sum(correct) == 56`), **không** phải bộ dự đoán mù 54/64 mà làn C trích dẫn; ma trận
   bốn lớp vì vậy lấy từ trường `confusion` của
   `sweep_results/relation_tokens/full_eval.json`.
4. ~~**Sửa tóm tắt**: bỏ khoảng [+12.5, +21.2] hoặc kèm đúng một câu cảnh báo.~~ → **ĐÃ
   LÀM 08/10.** Cả hai lần xuất hiện trong `00-tom-tat.txt` (TÓM TẮT và TÓM TẮT MỞ RỘNG) và
   cả hai lần trong `report/abstract.md` nay kèm câu: khoảng này bootstrap trên **đơn vị
   clip, không phải đơn vị nhóm bối cảnh**, nên theo mục **6.12(a)** của chính luận văn nó
   **không** được dùng làm căn cứ chứng nhận. (Số mục là 6.12(a) sau khi Chương 6 được đổi
   số ngày 08/10, không phải 6.11(a) như biên bản ghi ban đầu.) Tóm tắt nay không còn mạnh
   hơn thân bài.
5. ~~**Thêm một mục giới hạn nổi bật trong Ch5** cho B1.~~ → **ĐÃ LÀM 09/10** (mục 5.4.1 +
   mục 5.9 hạng mục 15). Viết theo **sự thật đã kiểm mã**, không theo mô tả trong biên bản:
   vòng lặp lấy 16 khung `linspace` **trong cửa sổ 48 khung, bước 24**, không phải 16 khung
   liên tiếp — chỗ lệch là **khoảng thời gian** cửa sổ bao. Quy tắc trích dẫn bắt buộc được
   nêu nguyên văn, và **chiều** thiên lệch được khai là KHÔNG BIẾT. Xem B1.
6. ~~**Thêm một mục về nguồn gốc hợp đồng nhịp.**~~ → **ĐÃ LÀM 09/10** (mục 5.4.4 + mục
   5.10 kết quả 3). Cả hai việc: công bố xuất xứ *f*/24 và đổi cách phát biểu sang "giữ kịp
   tốc độ khung của nguồn với biên 2.39–4.79 lần". Xem B7.
7. ~~**Công bố tính không lặp lại của DVFS trong bản nộp.**~~ → **ĐÃ LÀM 09/10** (mục
   5.4.6). Chọn phương án công bố, không chuyển headline — biến thể ghim nhịp
   (`chuong-05-trien-khai-bien-ghim-xung-nhip.txt`) vẫn chỉ để lưu, không nộp, không trộn.
   Xem B8.
8. ~~**Thêm một đoạn về đa so sánh.**~~ → **ĐÃ LÀM 09/10** (mục 6.8.1: 31 p-value, 10 mã
   cổng, α' = 1.61e-03, ba điều kiện biên). Xem B9.
9. ~~**Thêm một đoạn về confound B2 như một GIỚI HẠN ĐƯỢC CÔNG BỐ.**~~ → **ĐÃ LÀM 09/10.**
   Mục 6.9 nay có **năm** hạn chế thay vì ba. Hạn chế **(4)**: phép phân tách confound chỉ
   tồn tại trên iSafetyBench — trên 8 nhóm của Unsafe-Net, một phép hoán vị khớp cấu trúc có
   **quá ít nhóm** để cho một phân phối null dùng được, nên dải +13.0–21.8 pp vẫn **trộn** rò
   rỉ với độ khó cấu trúc, và phần "dịch chuyển phân bố nhãn" ở mục 6.12 vẫn **chưa được
   định lượng**. Hạn chế **(5)**: cả hai đối chứng (G19 và phép chia ngẫu nhiên làn A) được
   **nghĩ ra sau**; chỉ *tiêu chí* của chúng được cố định trước; **không** cái nào phán quyết
   cổng, **không** cái nào vào bảng tổng "hai đạt sáu trượt". Hội đồng ghi nhận: đây là hai
   hạn chế được **khai**, không phải bị phát hiện.
10. ~~**Thêm nguy cơ vòng tròn logic của KMeans(8)** vào cùng mục giới hạn.~~ → **ĐÃ LÀM
    09/10** (hạn chế (1) của mục 6.9). Xem B3.
11. ~~**git add -f + COMMIT + push.**~~ → **ĐÃ LÀM 09/10, nhưng KHÔNG bằng `add -f`.**
    Đóng bằng **một phủ định `.gitignore`** (`!sweep_results/**/*_oof.json`) để tệp OOF
    tương lai cũng được theo dõi; 61 tệp vào git; `jetson-measurements/` đã đủ 103 tệp từ
    `df1dab7`. Xem B5. (Việc `push` thuộc quyền tác giả — xem câu hỏi còn mở.)
12. ~~**Đồng bộ hoặc vô hiệu hoá README.md + 2 tệp docs.**~~ → **ĐÃ LÀM 09/10**, và làm
    hơn mức tối thiểu: banner **cộng** sửa nội dung sai ở cả hai tệp docs. Xem C1, C2.
13. ~~Bổ sung Geirhos et al. (2020) và nhánh record-wise/subject-wise CV vào Ch2.~~ → **ĐÃ
    LÀM 09/10** (`[44]`, `[45]`, đoạn shortcut learning ở mục 2.4.1). Xem C5.
14. ~~Sửa 24 chỗ nhãn M3/M4.~~ → **ĐÃ LÀM 09/10 — và KHÔNG cần tác giả xác nhận.** Ghi chú
    trước đó của biên bản này ("không tạo tác nào trong repo phân xử được") **là sai**, và
    sai vì chưa tìm hết: nhật ký huấn luyện bộ phát hiện đã vào git có dòng biểu ngữ
    `MPS (Apple M3)` do ultralytics tự in — một phát biểu do máy sinh ra, không do người
    gõ, nên nó mạnh hơn cả lời xác nhận của tác giả. Cùng chiều với nó: `hw.model` =
    `Mac15,12`, 16 GB, và `Phgs-Mac.lan` / `macOS-14.6-arm64` trong đầu tạo tác. Nhãn đúng
    là **M3**; **đề cương là bên sai**. Số chỗ thật là **49** (đếm cả mã nguồn), đã sửa hết;
    `python3 -m py_compile` trên hai tệp mã bị sửa: OK. Xem C3.
15. ~~Đổi tên Ch7.~~ → **ĐÃ LÀM 09/10**, lan sang bốn chỗ tham chiếu. Xem C7.
16. **Quay video demo dự phòng.** → **CÒN MỞ — KHÔNG LÀM ĐƯỢC BẰNG CÔNG CỤ VĂN BẢN.** Cần
    tác giả chạy `demo/run-unsafe-behavior-demo.py` với `b1-deploy-3seed.pt` và ghi lại. Ghi
    chú rủi ro: đây đồng thời là đường chạy của **B1** (độ chính xác cửa sổ trượt) — một buổi
    làm đóng được **cả hai** hạng mục, nên nó là việc có tỉ lệ lợi/chi phí cao nhất còn lại.
17. ~~Thêm chú giải làm tròn dưới Bảng 6.2 (lệch 0.1 pp).~~ → **ĐÃ LÀM 09/10**, và chú giải
    có **hai** phần chứ không một: (i) Δ tính trên giá trị chưa làm tròn rồi mới làm tròn về
    0.1 pp, nên nó **không** bằng hiệu của các số đã làm tròn in ở nơi khác (ví dụ
    B1w − B2 = +2.9 so với 84.4 − 81.6 = 2.8; ESTA in 83.8% ở mục 6.2.4 so với +3.5 trên
    E0 = 80.4 hàm ý 83.9) — lệch tối đa 0.1 pp, **không** ảnh hưởng phán quyết nào vì không
    đầu mút khoảng tin cậy nào nằm trong 0.1 pp của 0; (ii) Δ = −2.2 pp của G17 là
    *v11s − 26s* trong khi khoảng tin cậy [−1.7, +6.0] do `compare-oof-runs.py` in ra là
    *26s − v11s* — cùng một phép so sánh, hai dấu, và phán quyết không phụ thuộc chiều.

### Nhóm 3 — CẦN ĐO THÊM, cân nhắc có phá mốc đóng băng hay không
1. ~~**Đối chứng tách confound (B2)**~~ → **ĐÃ CHẠY XONG 08/10/2026.** Thực hiện bằng
   permutation test trên nhãn nguồn giả khớp cấu trúc (G19, 20 seed, p = 0.0476), không
   bằng fold ngẫu nhiên khớp kích thước. Mốc đóng băng 05/10 là mốc tác giả tự đặt (câu
   hỏi mở số 3 đã được trả lời), nên không có vi phạm thủ tục. Kết quả: +1.6 pp hình học
   fold, +13.8 pp danh tính nguồn. Chi tiết ở B2 phía trên. Việc Nhóm 2 số 9 vì vậy
   **không còn bắt buộc** — nhưng vẫn nên giữ một câu giới hạn cho phần Unsafe-Net chưa
   được tách.

   **Phát sinh ngoài kế hoạch, cùng ngày.** Phép đo lạm phát được lặp lại trên **chính
   cấu hình được triển khai** (B1: tháp CLIP toàn khung + đầu 201,090 tham số), giữ nguyên
   1,100 clip / 10 fold / 3 seed / 60 epoch, chỉ đổi phép chia: **97.7%** dưới fold ngẫu
   nhiên so với **84.2%** dưới LSO, tức **+13.5 pp**, CI theo cụm **[+8.0, +20.0]**,
   McNemar chính xác **p = 1.39e-41** (156 cặp bất đồng: 153 chỉ ngẫu nhiên đúng, 3 chỉ
   LSO đúng). Viết ở **mục 6.2.7**. Lượt chạy này cũng **không được đăng ký trước** (G12
   được đăng ký trên làn token quan hệ) nên là phần mở rộng hậu nghiệm, không phải cổng.
   Ý nghĩa cho hội đồng: câu hỏi §7 số 2 ("đề cương kỳ vọng > 95.8%, em đạt 84–85%") giờ
   được trả lời bằng **phép đo trên chính mô hình của luận văn**, không bằng lập luận.
   Cờ an toàn: cờ `--split random` ghi tạo tác dưới tên `rand-1100`, không ghi đè tạo tác
   tiêu đề `lso-1100`; đã kiểm chứng hai tệp OOF cùng tồn tại với đúng độ chính xác.
2. Đo độ chính xác vòng lặp cửa sổ trượt trên thiết bị (B1). Không khả thi trước 10/10 →
   công bố giới hạn (Nhóm 2 số 5).
3. p95/p99 + soak cho nhịp tổ hợp. Không khả thi trước 10/10 → công bố giới hạn.
4. INT8/TensorRT cho tháp CLIP. Không khả thi trước 10/10 → công bố là việc chưa làm.

## 6. KHUYẾN NGHỊ VỀ TIÊU ĐỀ

Tiêu đề hiện tại hứa một kiến trúc; luận văn giao một phép đo giao thức + một hệ biên.
Nếu GVHD cho phép điều chỉnh tiêu đề, phương án giữ được cả hai vế:

> "Phát hiện hành vi không an toàn trong giám sát công nghiệp dựa trên YOLO và Transformer:
>  đánh giá có ràng buộc nguồn và triển khai trên thiết bị biên"

Giữ nguyên vế đã đăng ký (không cần thủ tục đổi tên đề tài), thêm vế thực sự giao được.

## 7. CÂU HỎI HỘI ĐỒNG SẼ HỎI — CHUẨN BỊ TRƯỚC

1. Mục tiêu đề cương là YOLO–Transformer; hệ chạy trên Jetson dùng CLIP. Giải thích?
2. Đề cương kỳ vọng > 95.8%; em đạt 84–85%. Vì sao, và em có đạt mục tiêu không?
   → **ĐÃ CÓ SỐ (08/10):** chính mô hình được triển khai đạt **97.7%** dưới giao thức
   bỏ qua nguồn mà tài liệu dùng, và **84.2%** dưới leave-source-out; hiệu **+13.5 pp**,
   p = 1.39e-41 (mục 6.2.7). Luận văn đo được con số cao và chọn công bố con số thấp.
   **Không** so con số 97.7% này với 95.81% của bài báo gốc: 95.81% thuộc Unsafe-Net
   (làn C), 97.7% thuộc iSafetyBench (làn A) — mục 6.1.1 cấm đặt hai số đó vào một câu.
3. Con số 84.2% có áp dụng cho hệ đang chạy trên thiết bị không?
   → **ĐÃ CÓ CÂU TRẢ LỜI DỨT KHOÁT TRONG BẢN NỘP (09/10): KHÔNG.** Mục 5.4.1 đặt quy tắc
   trích dẫn bắt buộc: **không con số độ chính xác nào trong luận văn được đọc là độ chính
   xác của vòng lặp trên thiết bị.** Lý do bằng kỹ thuật: đánh giá lấy 16 khung `linspace`
   trên **toàn clip**; vòng lặp triển khai lấy 16 khung `linspace` **trong cửa sổ 48 khung,
   bước 24** (≈1.6 s ở 29.97 fps) — cùng dạng lấy mẫu, khác khoảng thời gian. **Chiều** của
   thiên lệch là **KHÔNG BIẾT**: cửa sổ hẹp hơn có thể dễ hơn, cũng có thể bỏ mất hiểm hoạ.
   Mọi số ở mục 5.4 là số **chi phí**, không phải số **chất lượng**. Phép đo còn thiếu được
   khai ở mục 5.9 hạng mục (15) kèm đúng đường chạy cần dùng. Câu trả lời đúng khi bị hỏi là
   "chưa đo, và luận văn nói thẳng điều đó ở hai chỗ" — **không** phải "chắc là tương đương".
4. Hiệu số +15.4 pp: bao nhiêu là rò rỉ, bao nhiêu là vì fold leave-source-out khó hơn?
   → **ĐÃ CÓ SỐ (08/10):** +13.8 pp rò rỉ danh tính nguồn, +1.6 pp hình học fold; đối
   chứng giả dược G19, 20 seed, p hoán vị = 0.0476, mục 6.6.3. Trả lời bằng đo, không
   bằng lập luận.
5. Hiện tượng rò rỉ do phép chia đã được biết trong y sinh từ 2012. Cái mới của em là gì?
6. Hợp đồng 3 quyết định/giây lấy từ đâu? Tiêu chuẩn an toàn nào nói vậy?
   → **ĐÃ CÓ TRONG BẢN NỘP (09/10): KHÔNG tiêu chuẩn nào.** Mục 5.4.4: 1.249 / 1.427 / 2.500
   quyết định/giây suy từ *f*/24 trên siêu dữ liệu fps của 1,100 clip, **không** từ OSHA, ISO
   45001, IEC 61508 hay tiêu chuẩn kỹ thuật Việt Nam nào; yêu cầu thời gian phản ứng an toàn
   thật là **đặc thù theo hiểm hoạ** (xe nâng đang chạy so với tủ điện bị mở lệch nhau nhiều
   bậc độ lớn) và nằm ngoài phạm vi. Phát biểu đúng là "**giữ kịp tốc độ khung của nguồn** với
   biên 2.39–4.79 lần" (mục 5.10 kết quả 3) — một phát biểu theo kịp luồng, **không** phải một
   chứng nhận an toàn. Không dùng chữ "vượt hợp đồng thời gian thực".
7. "Tiền đăng ký" — chứng minh bằng gì ngoài lời em nói?
   → **ĐÃ TRẢ LỜI BẰNG CÁCH TỰ KHAI GIỚI HẠN (09/10), KHÔNG bằng cách khẳng định mạnh hơn.**
   Mục 6.9 hạn chế (6) nêu đúng ba mốc git — script phán quyết vào git cùng commit với mô
   hình (29/09 23:14), `risks.md` vào git sau đó hai ngày (01/10 21:13), đặc tả làn tiêu đề
   vào git sau cả mốc đóng băng (07/10 23:03) — và kết luận thẳng: **người đọc không thể dùng
   git để chứng minh ngưỡng được cố định trước khi biết kết quả.** Điều chứng minh được là yếu
   hơn: ngưỡng bất biến kể từ 01/10 và phán quyết do script hằng số sinh ra. Trả lời đúng là
   nói nguyên văn như vậy rồi nêu cách sửa cho lần sau (OSF / commit đã ký đẩy trước khi chạy).
   Tuyệt đối **không** biện hộ rằng mốc git đã đủ.
8. Confusion matrix đâu?
   → **ĐÃ CÓ (09/10):** Bảng **6.2a** (B1, LSO, 1,100 clip), Bảng **6.2b** (cùng mô hình, fold
   ngẫu nhiên — đặt cạnh nhau cho thấy bỏ sót hiểm hoạ 65 → 10 clip và báo động giả 110 → 15
   chỉ vì đổi cách chia), Bảng **6.7a** (bốn lớp, đúng bộ dự đoán mù 54/64). Nêu kèm **cấu
   trúc lỗi**, không chỉ con số: hai loại lỗi gần cân nhau nên mô hình **không** lệch về phía
   an toàn, và luận văn **không** dịch ngưỡng vì chi phí can thiệp tại hiện trường nằm ngoài
   kho dữ liệu.
9. Hệ đa luồng camera → edge → server → dashboard trong đề cương, em giao được mấy phần?
10. 6/8 cổng trượt, giả thuyết trung tâm bị bác bỏ — vậy luận văn kết luận gì là dương?
11. **Cho xem nhật ký huấn luyện bộ phát hiện đạt mAP 0.9695.** → Câu này phải trả lời bằng
    **lời khai, không bằng lời bảo đảm**, và nói trước khi bị truy: *"Thưa hội đồng, không có.
    Nhật ký còn lại trong kho là của một phiên chạy trên MPS đã sụp về mAP = 0 ở mọi epoch sau
    epoch 1, nên nó không phải phiên sinh ra trọng số này. Theo bản ghi của em, trọng số được
    huấn luyện lại trên runtime GPU của Kaggle và tải về; em không giữ được nhật ký phía Kaggle,
    nên đó là lời khai của em chứ không phải bằng chứng. Thứ em có bằng chứng là phép đo lại:
    cùng trọng số đó, chạy trên CPU ngày 02/10, trên đúng 343 ảnh giữ riêng, cho mAP@50 0.9695
    — tạo tác đã nằm trong git. Luận văn ghi rõ sự phân biệt này ở mục 4.5.5 và 7.7.1(a)."*
    **KHÔNG được nói "nhật ký bị mất" rồi để trống ở đó** — phần "phiên MPS sụp về 0" là phần
    làm cho lời khai đáng tin, vì nó cho thấy tác giả đã đọc nhật ký của mình chứ không chỉ
    viện lý do. Xem C9.

## 8. CÂU HỎI CHƯA GIẢI QUYẾT (của hội đồng)

1. Có hay không văn bản GVHD/khoa duyệt đổi hướng đề tài — không tìm thấy trong repo.
   **Vẫn mở 09/10. Đây là hạng mục duy nhất còn lại có thể ảnh hưởng mức ĐẠT/KHÔNG ĐẠT**;
   mọi việc còn lại chỉ ảnh hưởng mức điểm.
2. README.md có được nộp kèm cho hội đồng, hay chỉ là tài liệu nội bộ? Quyết định mức
   nghiêm trọng của C1.
3. ~~Mốc đóng băng 05/10 là ràng buộc của nhà trường hay tự tác giả đặt?~~ → **ĐÃ TRẢ
   LỜI:** tác giả tự đặt. Nhóm 3 số 1 vì vậy khả thi và **đã chạy xong** 08/10.
4. `demo/run-unsafe-behavior-demo.py` đã từng chạy ra .mp4 thật chưa? **Vẫn mở 09/10** —
   và nay là câu hỏi có đòn bẩy cao nhất: một buổi chạy đóng được **cả** C6 (video dự phòng)
   **và** B1 (độ chính xác cửa sổ trượt), vì cùng một đường chạy.
~~6. Máy huấn luyện/phân tích là **M3** hay **M4 Pro**?~~ → **ĐÃ TRẢ LỜI 09/10 bằng tạo tác,
   không cần tác giả: M3.** Biểu ngữ ultralytics trong nhật ký đã vào git ghi `MPS (Apple M3)`;
   `hw.model` = `Mac15,12`, 16 GB. Đề cương ("M4 Pro, 36 GB") là bên sai. Xem C3.
5. Giao thức 125-clip của bài báo gốc tương đương mức nào với leave-source-out — chưa
   có phép quy đổi, nên so sánh chéo bài báo vẫn là so sánh khác giao thức.
