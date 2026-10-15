# Audit Chương 6–7: Phép đo – Công thức – Baseline – Xuất xứ số liệu

Phạm vi: `report/vn/chuong-06-danh-gia.txt` (1–2776, đọc hết), `report/vn/chuong-07-phan-tich-that-bai.txt` (1–1023, đọc hết), đối chiếu `report/vn/00-trang-thai-bien-soan.txt` và `git show` hai commit `06854b4`, `16f75c9`. Không sửa file nguồn. Không đo thêm.

## 0. Xác minh hai commit (không tin commit message)

Đã `git show --stat` + đọc diff đầy đủ, đối chiếu với nội dung hiện tại của hai chương.

| Commit | Tuyên bố | Thực tế kiểm chứng |
|---|---|---|
| `06854b4` (11/10) | Thêm §6.1.6 + Bảng 6.1b: đại lượng/giao thức/baseline+trị số/nguồn cho mọi con số tiêu đề; mang cách làm này sang bảng cổng, Ch7, Ch8 | **Đúng, đã làm thật.** Bảng 6.1b (L220–260) có 12 dòng, đủ 4 cột (đại lượng/giao thức/baseline+trị số/nhãn [ĐO]/[ĐỌC]/[SỐ HỌC]). Diff Ch8 xác nhận: tách 63.3% khỏi 67.2% (hai sàn trùng tên), tách 0.3% khỏi 1.7% (hai mẫu số), viết lại recall KAT = TP/(TP+FN), báo động giả = FP/(FP+TN). Không có số mới, đúng như cam kết "text correction only". |
| `16f75c9` (13/10) | Thêm Bảng 6.9 (bảng so sánh kết quả/baseline duy nhất); đọc Bảng 8 công trình gốc (§6.15.2); nhóm 8 cổng theo câu hỏi | **Đúng, đã làm thật.** Bảng 6.9 (L2637–2693) gộp toàn bộ phép so sánh kết quả/baseline của chương thành 3 khối không trộn lẫn (iSafetyBench, Unsafe-Net, NP-125), đúng tuyên bố "KHÔNG thêm con số mới" (L2635). §6.15.2 (L2391–2448) đọc lại Bảng 8 công trình gốc, tính ra thông lượng bằng nhau (lệch 0.35%), phát hiện Bảng 8 bỏ chi phí YOLO v4. §6.13 (L1787–1803) nhóm 8 cổng: phương pháp 2/2, kiến trúc 0/5, dung lượng 0/1 — khớp log commit. |

**Kết luận phần 0**: hai commit làm đúng và đủ phạm vi đã nêu, không phải "làm nửa vời rồi ghi quá". Đây là điểm mạnh nên giữ nguyên.

## a) Bảng kiểm 5 ô cho từng bảng/con số tiêu đề chính

Ký hiệu: Đ=đại lượng định nghĩa rõ; CT=công thức viết ra; GT=giao thức đo; BL=baseline có trị số+nguồn; CI=CI+đơn vị lấy mẫu độc lập. ✓=đủ, ✗=thiếu, (✓)=đủ nhưng có cảnh báo đi kèm.

| Con số/bảng (dòng) | Đ | CT | GT | BL | CI | Ghi chú |
|---|---|---|---|---|---|---|
| 84.2% BA B1 (L226–227, 6.2.1) | ✓ | ✓ (L203) | ✓ LSO-1100 | ✓ 63.3% [ĐO] | ✓ cụm/nguồn | Đầy đủ nhất của chương |
| 97.7% lạm phát (L228–229, 6.2.7) | ✓ | ✓ | ✓ | ✓ 84.2% [ĐO] | ✓ | — |
| G10–G20 (Bảng 6.8, L1731–1748) | ✓ | ✓ Δ=hiệu 2 BA | ✓ | ✓ mỗi dòng có trị số+nguồn | ✓ (†‡ chú thích đơn vị khác nhau) | Mẫu hình chuẩn nên dùng làm gốc |
| 81.2%/+16.9pp làn C (L236–237, 6.12.1) | ✓ | ✓ | ✓ LCO 8 nhóm | ✓ 64.4% [ĐO] | (✓) CI ở **đơn vị clip**, không phải nhóm — tự khai L2665–2666 | Baseline rõ, nhưng CI sai đơn vị được *tự thú*, không phải thiếu sót ẩn |
| 54/64=84.4%/88.4% (L238–241, 6.12.5) | ✓ | ✓ | ✓ | ✓ 57/64 [ĐỌC] | ✓ McNemar bound | — |
| 95.81% macro (tham chiếu, L2676) | ✓ ghi rõ KHÔNG phải baseline | n/a | n/a | n/a — hệ của họ không chạy lại được | n/a | Xử lý đúng: gọi là "tham chiếu tài liệu", không trừ chéo |
| Bảng 8 công trình gốc, 0.58s/0.14s (L2678–2680) | ✓ | ✓ [SỐ HỌC] thông lượng | ✓ | ✓ [ĐỌC] | n/a | Mẫu xử lý tham chiếu tài liệu tốt nhất của toàn luận văn |
| G25 74.19% BA (L2685–2686, 6.15.3) | ✓ | ✓ | ✓ NP-125 | ✓ scene-only 69.94% [ĐO] | ✓ McNemar | — |
| NP-125-D recall/FAR (L2687–2690, 6.15.4) | ✓ | ✓ TP/(TP+FN) v.v. | ✓ | ✓ scene-only 89.06%/49.18% [ĐO] | ✓ cụm 8 nhóm (rộng, tự khai) | — |
| Bảng 6.1 Lane A (L291–316) | ✓ | (✓) — macro-F1/PR-AUC/ROC-AUC KHÔNG có công thức (xem mục b) | ✓ | ✗ không mỗi dòng có baseline riêng (bảng mô tả, chưa so sánh) | ✗ không có CI trong bảng | Bảng mô tả thuần, baseline/CI nằm ở các mục kế tiếp — chấp nhận được vì không tự xưng là phán quyết |
| Bảng 6.7a confusion matrix + 7 metric (L1588–1641) | (✓) | ✗ macro-F1/κ/MCC/ROC-AUC/PR-AUC/ECE không có công thức | ✓ | (✓) κ=0.65 làm trần, có nguồn | ✗ | Giá trị đúng, nhưng 5/7 chỉ số không công thức hoá — rủi ro cao nhất khi hội đồng hỏi "κ tính sao" |
| §7.4 công thức tổng-giai-đoạn (L322–325) | ✓ | ✓ viết rõ, rồi **tự bác bỏ chính nó** | ✓ | n/a (đây là phân tích lỗi, không phải kết quả) | n/a | Mẫu lập luận tốt nhất của Ch7 |
| Bảng 7.2 sổ đăng ký đã rút (L607–714) | ✗ không nhắc lại đại lượng | ✗ | ✗ mỗi dòng chỉ có "Lý do" 1 dòng | n/a | n/a | Xem mục d — đoạn liệt kê thiếu biện luận rõ nhất |

## b) Công thức từng đại lượng — có/không, ở đâu (grep toàn bộ `report/vn/*.txt`)

| Đại lượng | Công thức có viết ra? | Vị trí |
|---|---|---|
| top-1 | CÓ | chuong-06 L202 |
| BA (độ chính xác cân bằng) | CÓ | chuong-06 L203 |
| macro (làn C) | CÓ (cùng công thức BA, khác số lớp) | chuong-06 L204–206 |
| recall KAT, báo động giả (FPR), precision | CÓ | chuong-06 L207 |
| Youden's J | CÓ (= recall + specificity − 1) | chuong-06 L208 |
| AUC trong-nhóm | CÓ (định nghĩa bằng lời: tỉ lệ cặp xếp đúng thứ tự) | chuong-06 L209–210 |
| Δ lạm phát giao thức | CÓ | chuong-06 §6.3 (L659–665) |
| n_decisions (nhịp) | CÓ | chuong-07 L597 `floor((processed-16)/8)+1` |
| J (độ trễ biên, Ch5) | CÓ | chuong-05 L898 `J=(P_TB−4.691)/fps` |
| **macro-F1 / F1** | **KHÔNG tìm thấy công thức** (chỉ xuất hiện như giá trị, hàng chục lần) | — |
| **PR-AUC** | **KHÔNG** (chỉ giá trị, không có "diện tích dưới đường cong P-R") | — |
| **ROC-AUC** | **KHÔNG** (chỉ giá trị) | — |
| **mAP / mAP@50 / mAP@50-95** | **KHÔNG** (không có công thức Average Precision, không có ngưỡng IoU định nghĩa) | — |
| **ECE** | **KHÔNG** (chỉ dịch tên "sai số hiệu chuẩn kỳ vọng", không có công thức theo bin) | chuong-02 L278 chỉ nêu tên |
| **Cohen's κ** | **KHÔNG** (chỉ giá trị 0.65/0.775/0.486, không có (po−pe)/(1−pe)) | — |
| **MCC** | **KHÔNG** (chỉ giá trị 0.785/0.492, không viết tên đầy đủ, không công thức TP/TN/FP/FN) | — |
| **Brier score** | **KHÔNG** (chỉ giá trị 0.338/0.098) | — |
| **McNemar (chính xác)** | Tên gọi + cách dùng (b, c bất đồng) có giải thích, nhưng **không có công thức/thống kê kiểm định viết ra** | chuong-06 §6.8 |
| **Hiệu chỉnh Nadeau–Bengio** | Có ví dụ số minh hoạ (1.66→2.49) nhưng **công thức hiệu chỉnh không viết ra**, chỉ trích dẫn [45] | chuong-06 L979–985 |
| Bootstrap theo cụm | Mô tả thủ tục (resample theo nguồn), đúng bản chất là thủ tục không phải công thức đóng — không coi là thiếu | chuong-06 L970–978, code `eval_utils/cluster_bootstrap.py` |
| Wilson CI | Chỉ dùng 1 lần, không có công thức | chuong-06 L2468 |

**Danh sách KHÔNG có công thức ở bất kỳ đâu trong `report/vn/`**: macro-F1/F1, PR-AUC, ROC-AUC, mAP (và AP/IoU), ECE, Cohen's κ, MCC, Brier score, McNemar (thống kê kiểm định), hiệu chỉnh Nadeau–Bengio, Wilson CI. Đây là nhóm chỉ số "ai cũng biết công thức chuẩn" nên tác giả có thể đã mặc định không cần viết — nhưng đúng là lỗ hổng nếu hội đồng hỏi trực tiếp "κ của anh tính thế nào", đặc biệt MCC/ECE/Brier vì có nhiều biến thể công thức khác nhau trong literature.

## c) Bảng kiểm Baseline

| Baseline | Trị số | Tự đo hay trích dẫn? | Nêu rõ trong văn bản? (dòng) |
|---|---|---|---|
| Sàn quy tắc đếm, làn A | 63.3% | [ĐO], mẫu thăm dò 300 clip cân bằng | CÓ, Bảng 6.1b L226, cảnh báo trùng tên L236/248 (§7.8) |
| Sàn quy tắc đếm, làn B (G10) | 67.2% | [ĐO], đủ 1,100 clip | CÓ, Bảng 6.1b L230, L1765–1766 |
| Sàn toàn khung tốt nhất (G11) | 74.1% | [ĐO] | CÓ, Bảng 6.8 L2648 |
| Hằng số lớp đa số / ngẫu nhiên | 50.0% | [SỐ HỌC] | CÓ, nhiều nơi, ví dụ L2481, L2664 |
| Sàn không-bbox công trình gốc | 85.78% (57/64) | [ĐỌC] từ bảng gốc | CÓ, §6.12.5/6.15, Bảng 6.1b L239 |
| Bbox đầy đủ công trình gốc | 95.81% macro | [ĐỌC], gọi là "tham chiếu tài liệu" **không phải baseline** | CÓ, minh bạch nhất của toàn chương (L216, L2676) |
| Sàn ngoại hình-bối cảnh, làn C | 64.4% macro | [ĐO] | CÓ, Bảng 6.1b L236–237 |
| scene-only (train majority) | 69.94% BA / 89.06% recall | [ĐO] | CÓ, L2479, L2526 |
| scene-only oracle (chặn trên) | 72.44% BA / 87.50% recall | [ĐO] | CÓ, L2480, L2527 |
| "Luôn báo động" | 100%/100% | [SỐ HỌC] | CÓ, L2528 |
| Original work ConvLSTM (chạy lại) | 78.12%→73.44% top-1 (MPS→CPU) | [ĐO] tái lập giao thức gốc, 11/10 | CÓ, §6.15.1, ngoại lệ hậu đóng băng ghi rõ |
| 3.0 quyết định/giây hợp đồng (Ch5, tham chiếu trong Ch7) | Có nhiều biến thể theo điều kiện clock | [ĐO] | CÓ nhưng phức tạp — đã rút nhiều phiên bản sai (Bảng 7.2 nhóm C) |

**Baseline KHÔNG rõ nguồn trong phạm vi Ch6/Ch7**: không tìm thấy baseline nào thiếu nhãn nguồn hoàn toàn — mọi baseline chính đều có [ĐO]/[ĐỌC]/[SỐ HỌC]. Điểm yếu duy nhất còn lại không phải "thiếu nguồn" mà là **độ phân giải CI sai đơn vị được tự khai nhưng vẫn in ra** (ví dụ L2665–2666, CI cấp clip thay vì cấp nhóm) — rủi ro bị trích dẫn nhầm nếu đọc lướt, dù văn bản đã cảnh báo.

## d) Đoạn liệt kê bảng-chồng-bảng thiếu biện luận

1. **Chuong-07 §7.6 Bảng 7.2 "Sổ đăng ký các phát biểu đã rút" (L607–714)**: ~26 dòng, 4 nhóm A–D. Mỗi dòng chỉ có "Lý do" 1 dòng ngắn, KHÔNG có đoạn văn tổng hợp "vì sao nhóm lỗi này xảy ra / cơ chế chung / giới hạn" ngay sau bảng — chuyển thẳng sang §7.7 (L715) với chủ đề khác. Đây là chỗ RÕ NHẤT khớp than phiền "liệt kê mà không biện luận". So sánh: Bảng 7.3 (§7.8, L835–861, trùng số) có đoạn chẩn đoán nguyên nhân hệ thống ngay sau (L862–869) — mẫu đối chứng cho thấy tác giả BIẾT cách viết đoạn tổng hợp, chỉ là không làm ở Bảng 7.2.
2. Bảng 6.1 Lane A mô tả (L291–316): là bảng mô tả (không phán quyết), chấp nhận được vì baseline/CI được chuyển xuống các mục ngay sau — không coi là lỗi, chỉ ghi nhận để hội đồng khỏi hiểu lầm là bảng kết quả đầy đủ.

**Các phần KHÔNG nên sửa — đã làm tốt** (mục e, gộp vào đây cho liền mạch): §6.1–6.1.6 (khung đọc + Bảng 6.1b), Bảng 6.8/6.13 (gom cổng theo câu hỏi, có 3 đoạn "Thứ nhất/Thứ hai/Thứ ba"), §6.14 (hai nhánh G20, lập luận cơ chế theo từng nhóm/nguồn, rất chặt), §6.15 (G21/G23/G25/NP-125-D, mỗi mục con đều có "không cho phép kết luận gì"), Bảng 6.9 + 5 kết luận cuối chương, §7.4 (phân tích lỗi tổng-giai-đoạn), §7.8 (trùng số + quy tắc trình bày). Đây là phần xương sống phương pháp luận của luận văn — sửa vào đây có rủi ro cao, lợi ích thấp.

## Câu hỏi còn mở (chưa giải quyết được trong phạm vi audit này)

1. Có cần viết công thức chuẩn cho macro-F1/PR-AUC/ROC-AUC/mAP/ECE/κ/MCC/Brier vào một phụ lục chung (ví dụ Phụ lục định nghĩa đại lượng), hay giữ nguyên hiện trạng (coi là kiến thức nền, chỉ cần trích thư viện tính — `sklearn.metrics` — nếu có)? Chưa rõ thư viện cụ thể nào được dùng để tính các chỉ số này có được nêu tên ở đâu khác trong luận văn (Ch3/Ch4) hay không — phạm vi audit này chỉ bao Ch6/Ch7.
2. Bảng 7.2 có nên được bổ sung 1 đoạn tổng hợp cuối mỗi nhóm (A–D), theo đúng mẫu đã làm cho Bảng 7.3? Đây là khuyến nghị, không phải lỗi bắt buộc sửa.
3. McNemar và hiệu chỉnh Nadeau–Bengio: công thức toán học không viết ra, chỉ có ví dụ số + trích dẫn [45]. Có đủ cho hội đồng hay cần công thức tường minh? Tuỳ mức độ hội đồng muốn kiểm tra lại bằng tay.
4. Hai commit đã xác minh là làm đúng tuyên bố — nhưng phạm vi của chúng tự giới hạn ở "mọi con số TIÊU ĐỀ" (Bảng 6.1b có 12 dòng). Các bảng mô tả phụ (Bảng 6.1, 6.2, 6.3...) không có baseline/CI tại chỗ — có cần mở rộng Bảng 6.1b để phủ hết mọi bảng, hay giữ nguyên phạm vi "chỉ con số tiêu đề" như hiện tại? Đây là quyết định phạm vi của tác giả, không phải lỗi.
