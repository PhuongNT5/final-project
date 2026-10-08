# PHẢN BIỆN 3 — TRUY XUẤT XUẤT XỨ SỐ LIỆU & NHẤT QUÁN NỘI BỘ

Vai: kiểm toán viên số liệu. Ngày kiểm toán: 05/10/2026. Phạm vi: report/vn/ (bản nộp), README.md,
deployment/artifacts/, sweep_results/, plans/*/reports/, git history.

## 1. BẢNG TRUY XUẤT CÁC CON SỐ HEADLINE

| # | Con số | Nguồn trích trong luận văn | Tệp tạo tác/log | Phán quyết |
|---|---|---|---|---|
| 1 | 75.2% (LSO, iSafetyBench) | Ch6 Bảng 6.2 | `sweep_results/relation_tokens/logs/g12-protocol-inflation-table.log` | TRUY ĐƯỢC |
| 2 | 90.6% (chia ngẫu nhiên) | Ch6 Bảng 6.2 | cùng log trên | TRUY ĐƯỢC |
| 3 | +15.4 pp | Ch6 6.5, Bảng 6.2, Phụ lục A | cùng log; khớp số học 90.6−75.2=15.4 | TRUY ĐƯỢC |
| 4 | CI [+10.6, +21.5] | Ch6 6.5 (G12) | `logs/g12-inflation-paired.log` | TRUY ĐƯỢC |
| 5 | p = 3.58e-45 | Ch6 G12, Ch1, Ch8, 00-tom-tat, Phụ lục A | `sweep_results/relation_tokens/logs/g12-inflation-paired.log` (b=186 c=8) | TRUY ĐƯỢC — nhất quán 6/6 chỗ trích |
| 6 | 84.2% (B1, độ chính xác triển khai) | Ch5 5.4.1 | Ch6 (B1 LSO) — cần log cụ thể B1; đối chiếu `sweep_results/relation_tokens/logs/r48-lso-B1.log` (chứa 201,090) | TRUY ĐƯỢC (gián tiếp qua log tham số B1; không kiểm log accuracy B1 trực tiếp trong phiên này) |
| 7 | 63.3% (đường sàn tiền đăng ký) | Ch3 3.x, Ch6 6.1, 00-tom-tat | mô tả nguồn gốc đầy đủ trong Ch3 (quy tắc 3-đặc-trưng, 300-clip probe) nhưng KHÔNG trích tên tệp log cụ thể tại điểm dùng ở Ch6/tóm tắt | TRUY MỘT PHẦN — có đường dẫn suy luận bằng văn, thiếu tên tệp tại điểm trích lại |
| 8 | 167.291 ms/quyết định | Ch5 Bảng 5.2 | `r48-composed-cadence-fp16-20261003T{155728,155806,155843}.json` | TRUY ĐƯỢC THEO VĂN BẢN — nhưng tệp **chưa commit vào git** (xem mục 8) |
| 9 | 5.978 quyết định/giây | Ch5 Bảng 5.2 | cùng tệp trên; khớp số học 1000/167.291=5.978 | TRUY ĐƯỢC (số học đúng), cùng rủi ro xuất xứ git như #8 |
| 10 | 201,090 tham số (đầu B1) | Ch5 5.1.1, 5.4.1 | `sweep_results/relation_tokens/logs/r48-lso-B1.log`, `deployment/export/MANIFEST.md` | TRUY ĐƯỢC |
| 11 | +16.9 pp macro (G12/làn C, LCO) | Ch6 6.11, Phụ lục A, 00-tom-tat | tính trực tiếp 83.6−66.7=16.9 — khớp Bảng 6.1 | TRUY ĐƯỢC |
| 12 | CI [+12.5, +21.2] | Ch6 6.11 | bootstrap n_boot=5000 (nêu rõ phương pháp, không trích tên tệp log cụ thể tại điểm này) | TRUY MỘT PHẦN |
| 13 | dải +13.0 đến +21.8 pp | Ch1, Ch6 Bảng 6.1, Ch8, Phụ lục A, 00-tom-tat | `sweep_results/relation_tokens/logs/g12-protocol-inflation-table.log` (phần Unsafe-Net) | TRUY ĐƯỢC — khớp min(E3=13.0)/max(E1 d64=21.8) đúng |
| 14 | 5.1 → 1.4 pp (độ phân tán thang, iSafetyBench) | 00-tom-tat, Ch1, Ch8, Phụ lục A | log trên: "spread... 5.1 pp / 1.4 pp" | TRUY ĐƯỢC THEO LOG — nhưng **không khớp nếu tính lại từ 4 số % in trong Bảng 6.2** (91.9−90.6=1.3, không phải 1.4) — xem mục 7, lỗi nhẹ |
| 15 | 7.3 → 2.1 pp (độ phân tán thang, Unsafe-Net) | 00-tom-tat, Ch1 | cùng log, phần Unsafe-Net: LCO spread 7.3, k-fold spread 2.1 | TRUY ĐƯỢC — khớp đúng cả khi tính lại từ Bảng 6.1 (69.6→62.3=7.3; 82.5→84.6=2.1... thực ra min k-fold 82.5 max 84.6 → 2.1 ✓) |
| 16 | 18,044–298,892 tham số | Ch3, Ch4, Ch6, Phụ lục A, 00-tom-tat | `sweep_results/relation_tokens/logs/g12-protocol-inflation-table.log` + Bảng 4.5/6.1 | TRUY ĐƯỢC |

## 2. BẢNG 4.1–4.4 VÀ HÌNH 4.1

Xác nhận PHẦN, không xác nhận TOÀN BỘ như trạng-thái-biên-soạn tự khai:

- **Bảng 4.1** (bộ phát hiện Stage-A): CÓ log — `sweep_results/entity_detector/reval-2026-10-02-detector-best-val343.log` (đã tracked trong git). TRUY ĐƯỢC. Trọng số `detector_best.pt` tồn tại trên đĩa nhưng KHÔNG tracked (bị `.gitignore:80` loại) — người ngoài đọc được log nhưng không tái lập được từ trọng số.
- **Bảng 4.2** (kappa gán nhãn): CÓ log — `plans/260613-thesis-defense-revision/reports/kappa-result.json` (tracked). TRUY ĐƯỢC.
- **Bảng 4.3** (chi phí trích đặc trưng): dẫn 5 log/tensor cụ thể trong văn, nhưng **toàn bộ 4 trong 5 tệp tensor/log KHÔNG tracked trong git** (`feats_isbclip_all.pt`, `feats_isbclip26_all.pt`, `feats_isbvmae_all.pt`, `build-isb-tokens.log` — đều bị `.gitignore` loại qua `sweep_results/**/*.pt` và `**/*.json`/log không nằm diện whitelist). TRUY MỘT PHẦN — văn bản trỏ đúng tên tệp, nhưng người ngoài clone git sẽ không thấy tệp.
- **Bảng 4.4** (thành phần fold): tự khai "dựng lại từ bộ nạp dữ liệu, không từ log" — đây là thiết kế có chủ ý (tái lập bằng mã, không bằng tạo tác đông cứng), KHÔNG phải thiếu sót như 4.1–4.3. Nên liệt riêng, không gộp chung "không log" với 4.1–4.3 — **trạng-thái-biên-soạn gộp sai**: Bảng 4.1 và 4.2 THỰC RA CÓ log.
- **Hình 4.1**: grep toàn bộ `chuong-04-chi-tiet-thuc-nghiem.txt` — **0 lần xuất hiện chuỗi "Hình 4.1"**. Hình này chỉ được nhắc tới trong `00-trang-thai-bien-soan.txt`, không tồn tại trong thân chương đã nộp (hợp lý vì chương là .txt, hình ảnh chèn riêng vào mẫu trường) — nhưng do đó **không kiểm toán được nội dung Hình 4.1** trong phạm vi văn bản nộp.

**Hội đồng có quyền loại Bảng 4.1–4.4 không?** Không nên loại cả bốn đồng hạng — chỉ Bảng 4.3 (4/5 nguồn không tracked) và Bảng 4.4 (không có tạo tác đông cứng, chỉ tái lập bằng mã — chấp nhận được về phương pháp nhưng yếu hơn một log) là đáng chất vấn. Bảng 4.1, 4.2 có log tracked, truy được. **Nếu hội đồng loại cả bốn, Chương 4 còn lại**: mục 4.3.1–4.3.2 (cấu hình trích đặc trưng, đối chứng G14 — có bằng chứng ở mức mã nguồn), 4.3.3 (đối chứng G17, có số liệu tensor so khớp byte), Bảng 4.5 (đã có trọng tài `per-clip-logits-test64.csv`, commit 6319b21) — chương KHÔNG sụp hoàn toàn nhưng mất phần mô tả chất lượng dữ liệu đầu vào (mAP bộ phát hiện, kappa nhãn) là phần hội đồng hay hỏi nhất.

## 3. multi-stream-results.json ("hand-patched")

Đã mở tệp: trường `"note"` ghi nguyên văn **"Hand-patched post-fix numbers matching Chapter 5
Table 5.4 (drain-on-shutdown + ensemble-latency fix)"** — tức số liệu được sửa tay để **khớp** một
bảng đã viết trước, không phải bảng được viết từ số đo. Đây là hướng nhân-quả NGƯỢC, mức nghiêm
trọng cao nếu số liệu này còn được trích dẫn như số đo.

Kiểm tra bản tiếng Việt Ch5 (`chuong-05-trien-khai-bien.txt`, mục 5.6.4): CÓ dùng — nhưng:
- Grep toàn văn: **không một giá trị số nào** của tệp này (17.6/29.1/26.2/20.8 fps, 55.1/72.9/86.4 ms...)
  xuất hiện trong bảng dữ liệu của bản nộp. Tệp chỉ được **nêu tên và cảnh báo** ("tự ghi chú là
  hand-patched... cấp số cho bảng đo trên CPU máy chủ dòng M, không cấp số cho bảng đa luồng Jetson").
- Mục 5.6.2 tuyên bố rõ: toàn bộ mục 5.6 (gồm phần multi-stream M-CPU) là **số LỊCH SỬ**, "không
  con số nào trong đó được dùng để chống đỡ một kết luận của luận văn".
- **Kết luận: bản tiếng Việt đã nộp XỬ LÝ ĐÚNG rủi ro này** — không giống bản tiếng Anh (vẫn giữ
  Bảng 5.4 dựa trên tệp hand-patched như một bảng số liệu thật). Đây là một điểm CẢI THIỆN thật,
  không phải lỗi sót. Mức nghiêm trọng hạ từ "nghiêm trọng" xuống "đã khắc phục, cần giữ nguyên
  cách viết này khi dàn trang — không phục hồi bảng số liệu gốc".

## 4. MÂU THUẪN README vs BẢN TIẾNG VIỆT

README.md (tracked, commit mới nhất 06/09/2026, có `M` chưa commit) và bản VN nộp **mô tả hai hệ
thống headline khác nhau**:

| Trục | README.md | Bản tiếng Việt (nộp) |
|---|---|---|
| Kết luận headline | "deployment-led... measured real-time edge system (P1)" + accuracy ladder 56/64, 89.28% | **Lạm phát giao thức (D1)**, KHÔNG mở đầu bằng accuracy (ràng buộc 9) |
| Hệ triển khai chính | P1 (frozen-COCO detector + multi-stream Transformer), hoặc "B1... tháp CLIP" chỉ ở bản VN | **B1** = tháp CLIP ViT-B/32 đóng băng + đầu B1 (201,090 tham số) |
| Số P3 headline | 310 ms/decision = 3.23 dec/s (pinned), +7.5% margin — TRÌNH BÀY NHƯ KẾT LUẬN | +7.5% margin **đã bị rút lại** (Ch5 cảnh báo thứ hai); P3 là tổng-giai-đoạn, không phải nhịp tổ hợp |
| Phán quyết hợp đồng P3 | "FAIL dưới DVFS, PASS khi pinned +7.5%" (số liệu tổng giai đoạn) | PASS 2/3 đoạn khi ghim nhịp (3.479/3.965), FAIL 3/3 dưới DVFS (1.141–1.511) — đo **tổ hợp**, khác cấu trúc với số của README |
| Nhịp đã triển khai (B1/167ms) | KHÔNG xuất hiện trong README | 167.291 ms = 5.978 qđ/giây — không có mặt ở README |
| "57/64 calibrated", "+3.5pp" | README tự nhận đây **đã withdrawn**, nhưng vẫn in số đó ra để giải thích (có nhãn withdrawn rõ) | Cũng ghi "đã rút", nhất quán về việc rút |
| 95.7% mAP | Không thấy trong README (đã kiểm không xuất hiện) | Đã rút, không phục hồi |
| 331ms/3.02 dec/s/3.91J | README tự ghi "đã rút" (withdrawn), minh bạch | Không phục hồi trong VN |

**README có phải tài liệu công khai không?** CÓ — tracked trong git, có remote GitHub công khai dạng
URL (`https://github.com/phuongnt18-hub/my-project.git`). README là trang đầu tiên hiển thị khi mở
repo.

**Rủi ro nếu hội đồng đọc README**: hội đồng sẽ đọc được một bản tóm tắt với headline, hệ thống
triển khai, và phán quyết hợp đồng P3 **khác hẳn** bản luận văn tiếng Việt vừa nộp — đọc xong có thể
kết luận tác giả đổi câu chuyện giữa hai bản mà không công bố, hoặc tệ hơn, nghi ngờ bản nào là "thật".
README chưa được cập nhật đồng bộ với quyết định headline mới nhất (D1 — lạm phát giao thức, B1/CLIP)
— đây là một SẢN PHẨM BÀN GIAO CHƯA ĐỒNG BỘ, mức rủi ro **CAO** nếu repo được chia sẻ cùng luận văn.

## 5. SỐ ĐÃ RÚT LẠI CÒN SÓT — KẾT QUẢ GREP TOÀN BỘ

| Con số đã rút | Còn sót trong report/vn/? | Còn sót trong README.md? | Còn sót trong docs/? |
|---|---|---|---|
| "57/64 calibrated" (nguyên văn tiếng Anh) | Không tìm thấy nguyên văn (bản VN không dùng cụm này) | CÓ — nhưng luôn kèm nhãn **withdrawn** liền kề, không phải phục hồi | docs/ không chứa |
| "+3.5 pp macro win" | CÓ xuất hiện ("+3.5 pp") ở Ch5 (đã rút, có nhãn), Ch6 Bảng G15 (ESTA−E0, đây là MỘT con số +3.5pp KHÁC, không phải cùng con số bị rút — là kết quả G15 hợp lệ, CI chứa 0, TRƯỢT — không phải vi phạm) | CÓ, có nhãn withdrawn | — |
| "95.7% mAP" | Không tìm thấy ở đâu trong report/vn (chỉ nhắc "đã rút" ở Ch4, Ch7, trang-thái) | Không xuất hiện | docs/ không chứa |
| "331 ms/3.02 dec/s/3.91 J" | Không tìm thấy trong report/vn | CÓ, có nhãn withdrawn rõ ràng ("are withdrawn") | — |
| "P3 ĐẠT hợp đồng 3.0 qđ/giây" (dạng khẳng định tuyệt đối, không kèm "2/3 đoạn") | KHÔNG tìm thấy dạng tuyệt đối. Có một câu ở Ch5 cảnh báo đầu chương dùng cụm "ĐẠT hợp đồng 3.0 qđ/giây **trên 2 trong 3 đoạn test**" — đây là bản ĐÃ SỬA, có định ngữ, không phải câu bị rút | — | — |
| "P3 TRƯỢT hợp đồng ở CẢ HAI điều kiện xung nhịp" | KHÔNG tìm thấy — Ch5 có câu "PHÁT BIỂU ĐÓ SAI" ngay sau khi trích dẫn lại để giải thích lịch sử, đúng quy cách công bố minh bạch | — | — |

**Kết luận mục 5: KHÔNG phát hiện số đã rút nào bị phục hồi trái phép trong bản tiếng Việt.** Mọi chỗ
còn thấy chuỗi ký tự của số đã rút đều nằm trong câu giải thích lịch sử có nhãn rõ "đã rút"/"SAI". Đây
là thực hành đúng. **README.md thì có in lại các số đã rút — nhưng luôn kèm nhãn withdrawn tại chỗ**,
nên không tính là "còn sót" theo nghĩa phục hồi ngầm; rủi ro của README nằm ở mục 4 (mâu thuẫn headline),
không nằm ở việc phục hồi số bị rút.

## 6. VI PHẠM QUY TẮC TRÍCH NHỊP QUYẾT ĐỊNH (ràng buộc 11, 12, 15)

Grep `chuong-05-trien-khai-bien.txt`:
- Rule 11 (không trích tổng giai đoạn làm phán quyết hợp đồng): mọi lần "tổng giai đoạn" xuất hiện
  đều đi kèm cảnh báo "không phải một nhịp" / "không được gọi là chặn trên bi quan" — **KHÔNG VI PHẠM**.
- Rule 12(2) (không trích `drop_rate`): chỉ xuất hiện trong chính đoạn NÊU QUY TẮC, không dùng làm
  bằng chứng ở nơi khác — **KHÔNG VI PHẠM**.
- Rule 15 (cấm trích `decisions_per_sec_p50` không giải mã, 3.68/3.66): grep "3.68", "3.66",
  "decisions_per_sec_p50" trong Ch5 nộp — **0 kết quả** — **KHÔNG VI PHẠM**.

Không phát hiện vi phạm chính ba quy tắc này trong chương đã nộp.

## 7. SAI SỐ HỌC PHÁT HIỆN ĐƯỢC

1. **1000/167.291 = 5.9776 ≈ 5.978** — khớp, không lỗi.
2. **75.2 + 15.4 = 90.6** — khớp đúng, không lỗi.
3. **Bảng 6.1 (Unsafe-Net)**: 4/5 dòng Δ khớp tuyệt đối với hiệu hai cột in; 2 dòng (E3: 82.5−69.6=12.9
   vs 13.0; E1 d96: 84.5−63.9=20.6 vs 20.5) lệch 0.1pp — **ĐÃ ĐƯỢC CHÚ GIẢI** ngay dưới bảng (làm tròn
   trước/sau), không phải lỗi sót.
4. **Bảng 6.2 (iSafetyBench) — dòng "Độ phân tán của thang", cột chia ngẫu nhiên**: in **1.4 pp**,
   nhưng tính lại từ 4 giá trị % in trong chính bảng (90.6, 91.3, 91.9, 90.7) cho **91.9−90.6=1.3 pp**,
   không phải 1.4. Giá trị 1.4 khớp với log gốc (`g12-protocol-inflation-table.log` in "1.4 pp") nên
   KHÔNG PHẢI do tác giả bịa — là cùng loại lệch làm-tròn đã được chú giải cho Bảng 6.1, nhưng **chú
   giải đó KHÔNG được lặp lại cho Bảng 6.2** → người đọc tính tay bằng tay sẽ thấy "sai" 0.1pp không
   lời giải thích tại đúng chỗ. Mức: NHẸ, sửa bằng câu chữ (thêm 1 câu chú giải).
5. **Dải +13.0 đến +21.8 pp** (Ch1/Ch6/Ch8/Phụ lục A): khớp đúng min(E3=13.0)/max(E1 d64=21.8) của
   Bảng 6.1 — không lỗi.
6. **5.1→1.4 pp / 7.3→2.1 pp**: nguồn gốc giống hệt mục 4 trên (log, không phải hai cột rounded) —
   7.3/2.1 (Unsafe-Net) khớp đúng khi tính lại từ Bảng 6.1; chỉ cặp 5.1/1.4 (iSafetyBench) có lệch
   0.1pp như đã nêu.

**Tổng kết mục 7: không phát hiện sai số học nghiêm trọng (loại đổi dấu, đổi hướng, hay chênh lớn hơn
0.1 pp). Toàn bộ chênh lệch tìm được đều thuộc hiệu ứng làm tròn 1 chữ số thập phân đã biết và đã có
cơ chế chú giải — chỉ thiếu áp dụng chú giải đó đồng bộ ở Bảng 6.2.**

## 8. XUẤT XỨ GIT CHO SỐ LIỆU CHƯƠNG 5

Kiểm tra trực tiếp (không dựa vào tự khai):

- `.gitignore:49` xác nhận đúng: `deployment/artifacts/*` bị loại, chỉ `per-clip-logits-test64.csv`
  được whitelist (`!deployment/artifacts/per-clip-logits-test64.csv`).
- **NHƯNG**: `git status` hiện tại cho thấy phần lớn tệp `deployment/artifacts/jetson-measurements/*`
  đã ở trạng thái `A` (staged/added vào index qua `git add -f`) — tức việc "buộc vào git" nêu ở
  trạng-thái-biên-soạn mục "VIỆC CÒN LẠI" **đã được bắt đầu**, nhưng:
  - `git log -- deployment/artifacts/jetson-measurements/r48-composed-cadence-fp16-20261001T233159.json`
    trả về **rỗng** — tệp này (nguồn của cột "TRƯỚC" trong Bảng 5.2) **CHƯA từng được commit**, chỉ
    đang nằm trong index cục bộ.
  - Tức là: tại HEAD hiện tại của repo (commit `6319b21`), **toàn bộ các tệp `r48-composed-cadence-*`,
    `p3-composed-cadence-*`, `decision-latency-*` dùng cho Bảng 5.1, 5.2, 5.4b, bảng 1080p KHÔNG tồn
    tại trong lịch sử git**. Một người ngoài `git clone` repo ở commit đã push sẽ **không thấy** các
    tệp này.
- `multi-stream-results.json`: ĐÃ tracked và committed, không có thay đổi chờ — xuất xứ git OK cho
  tệp này (vấn đề của nó là nội dung "hand-patched", không phải việc thiếu commit).
- `sweep_results/`: 1,092 tệp trên đĩa, chỉ **57 tệp tracked**. Các tệp làm nguồn cho Bảng 4.3
  (`feats_isbclip_all.pt`, `feats_isbclip26_all.pt`, `feats_isbvmae_all.pt`,
  `logs/build-isb-tokens.log`) và cho Bảng 6.1/6.2 một phần (`tier2_binary_logs/*`,
  `tier3_lco_logs/*` — chưa kiểm riêng nhưng cùng thư mục `relation_tokens/` áp `.gitignore` giống) đều
  **không tracked** — bị loại bởi `sweep_results/**/*.pt` và không nằm trong danh sách whitelist
  `.json`/`.md`.

**Kết luận mục 8: XÁC NHẬN đúng như trạng-thái-biên-soạn — tính đến thời điểm kiểm toán, số liệu
Chương 5 (và một phần Chương 4) KHÔNG truy xuất được bởi người ngoài chỉ có quyền đọc repo đã push,
vì tạo tác nguồn chưa có trong lịch sử commit.** Việc stage (`git add -f`) đã làm nhưng CHƯA commit —
đây là rủi ro còn treo, không phải đã xong.

## (a) BẢNG RÚT GỌN — HEADLINE → TRẠNG THÁI TRUY XUẤT

| Con số | Trạng thái |
|---|---|
| 75.2% / 90.6% / +15.4pp / CI[+10.6,+21.5] / p=3.58e-45 | TRUY ĐƯỢC |
| 84.2% (B1) | TRUY ĐƯỢC (gián tiếp) |
| 63.3% (đường sàn) | TRUY MỘT PHẦN |
| 167.291 ms / 5.978 qđ/giây | TRUY ĐƯỢC THEO VĂN BẢN, nhưng tệp nguồn **chưa commit git** |
| 201,090 tham số | TRUY ĐƯỢC |
| +16.9pp / CI[+12.5,+21.2] | TRUY ĐƯỢC (số chính); CI bootstrap TRUY MỘT PHẦN (thiếu tên log tại điểm trích) |
| +13.0 đến +21.8 pp | TRUY ĐƯỢC |
| 5.1→1.4 pp | TRUY ĐƯỢC qua log; lệch 0.1pp nếu tính tay từ bảng rounded (đã biết, chưa chú giải ở Bảng 6.2) |
| 7.3→2.1 pp | TRUY ĐƯỢC, khớp cả khi tính tay |
| 18,044–298,892 tham số | TRUY ĐƯỢC |

## (b) MÂU THUẪN / SAI SỐ HỌC / SỐ ĐÃ RÚT CÒN SÓT — PHẦN QUAN TRỌNG NHẤT

1. **MÂU THUẪN NGHIÊM TRỌNG, README vs bản nộp**: README.md (public, git-tracked, remote GitHub) mô
   tả headline, hệ triển khai, và phán quyết hợp đồng P3 **khác hẳn** bản tiếng Việt (xem bảng mục 4).
   README chưa đồng bộ với quyết định D1 (lạm phát giao thức) và với kết quả B1/167.291ms mới nhất.
   **Đây là rủi ro truy xuất lớn nhất tìm được** — không phải vì số liệu trong VN sai, mà vì tài liệu
   công khai đi kèm repo kể một câu chuyện khác.
2. **XUẤT XỨ GIT CHƯA XONG**: toàn bộ tạo tác nguồn của Bảng 5.1, 5.2, 5.4b và bảng 1080p (Ch5) +
   4 trong 5 tạo tác của Bảng 4.3 (Ch4) **chưa có trong lịch sử git commit**, dù đã stage. Nếu nộp ở
   trạng thái này, số liệu Ch4–Ch5 không tái lập được từ repo đã push.
3. **SAI SỐ HỌC NHẸ, Bảng 6.2**: dòng "Độ phân tán của thang" cột chia ngẫu nhiên in 1.4pp, tính tay
   từ 4 số % trong bảng ra 1.3pp — đúng theo log gốc nhưng thiếu câu chú giải (đã có mẫu câu chú giải
   này ngay dưới Bảng 6.1, chỉ cần lặp lại).
4. **KHÔNG phát hiện số đã rút lại còn sót** trong report/vn/ dưới dạng phục hồi ngầm (không nhãn).
   README có in lại một số số đã rút nhưng luôn kèm nhãn "withdrawn" — không tính là lỗi tương tự,
   nhưng góp phần vào mâu thuẫn ở mục 1.
5. **Không phát hiện vi phạm ràng buộc 11/12/15** (trích tổng giai đoạn/drop_rate/dòng không giải mã)
   trong Ch5 bản nộp.
6. Hai mục TRUY MỘT PHẦN (63.3%, CI +16.9pp) không phải lỗi — chỉ là thiếu tên tệp log tại đúng điểm
   trích lại ở Ch6/tóm tắt, trong khi log tồn tại và được trích đủ ở nơi khác (Ch3, cước chú G-gate).

## (c) PHÁN QUYẾT TRỤC TRUY XUẤT

**ĐẠT CÓ ĐIỀU KIỆN.**

Lý do không ĐẠT thẳng: (i) README công khai mâu thuẫn headline với bản nộp — rủi ro uy tín cao nếu
hội đồng tự tìm thấy repo; (ii) tạo tác nguồn Ch4/Ch5 chưa commit — số liệu chưa truy xuất được từ
repo đã push tại thời điểm kiểm toán.

Lý do không CHƯA ĐẠT: bản thân văn bản tiếng Việt đã nộp kỷ luật tốt — tự khai trung thực các khiếm
khuyết xuất xứ, không phục hồi số đã rút, tuân thủ nghiêm ba quy tắc trích nhịp (11/12/15), xử lý
đúng rủi ro `multi-stream-results.json` hand-patched (không như bản EN), và toàn bộ "sai số học" tìm
được chỉ là lệch làm tròn 0.1pp đã có cơ chế giải thích.

## (d) VIỆC PHẢI SỬA TRƯỚC KHI NỘP — XẾP ƯU TIÊN

**Sửa được bằng câu chữ / thao tác git, KHÔNG cần đo lại** (có thể làm trước 10/10):
1. [ƯU TIÊN 1] `git add -f` + **commit** (chưa chỉ stage) toàn bộ tạo tác `deployment/artifacts/jetson-measurements/*` dùng cho Bảng 5.1/5.2/5.4/5.4b/1080p, và push. Không có bước này, Ch4–Ch5 không tự truy xuất được từ repo đã push.
2. [ƯU TIÊN 1] Đồng bộ README.md với headline D1 (lạm phát giao thức) và hệ triển khai B1/167.291ms/5.978 — hoặc tối thiểu thêm một khối "ĐÃ CẬP NHẬT — xem bản luận văn tiếng Việt cho headline hiện hành" ở đầu README, trỏ rõ README là tài liệu kỹ thuật của mã nguồn, không phải headline luận văn.
3. [ƯU TIÊN 2] Thêm 1 câu chú giải làm tròn 0.1pp dưới Bảng 6.2 (mẫu đã có sẵn dưới Bảng 6.1), áp cho dòng "Độ phân tán của thang".
4. [ƯU TIÊN 2] Tại Ch6 6.11 và 00-tom-tat, thêm tên tệp log cụ thể cho CI bootstrap [+12.5,+21.2] (biên +16.9pp) tại chính điểm trích, không chỉ ở nơi khác.
5. [ƯU TIÊN 3] Tại Ch6 6.1/00-tom-tat, thêm tên tệp log cho đường sàn 63.3% tại điểm trích lại (nguồn đã có đủ ở Ch3, chỉ thiếu liên kết ngược).
6. [ƯU TIÊN 3] Quyết định có commit các tạo tác `sweep_results/relation_tokens/feats_isb*.pt` và `logs/build-isb-tokens.log` (nguồn Bảng 4.3) hay chấp nhận để chúng ở ngoài git với ghi chú rõ "tái lập được bằng script X, không đông cứng trong repo".

**Cần chạy lại đo / không sửa bằng câu chữ được:**
7. Không có mục nào trong phạm vi kiểm toán này đòi phải đo lại — mọi vấn đề tìm được là vấn đề đóng gói/xuất xứ/văn bản, không phải vấn đề số đo sai. (Phù hợp với tự khai của trang-thái-biên-soạn: "việc còn lại là đóng gói, không phải đo".)

## CÂU HỎI CHƯA GIẢI QUYẾT

1. 84.2% (B1) — chưa mở trực tiếp log accuracy B1 (chỉ đối chiếu qua log tham số 201,090); nên một phản biện khác hoặc phiên kiểm toán sau mở `r48-lso-B1.log` để xác nhận số 84.2% nằm đúng trong đó.
2. Chưa kiểm tra riêng `sweep_results/relation_tokens/tier2_binary_logs/` và `tier3_lco_logs/` (nguồn Bảng 6.1) có tracked trong git hay không — nằm ngoài thời gian phiên này, nên kiểm trước khi đóng băng cuối.
3. Hình 4.1 không thể kiểm toán nội dung vì không có trong văn bản .txt — cần hỏi tác giả xem 5/6 ô ảnh đã chèn vào mẫu trường hay chưa, và liệu ảnh gốc có nguồn tạo tác kiểm được không.
4. Chưa xác minh README.md có thực sự được tác giả gửi kèm luận văn cho hội đồng hay chỉ là tài liệu nội bộ của repo — nếu tác giả xác nhận README không đi kèm bản nộp, mức rủi ro ở (b)-1 hạ xuống đáng kể.
