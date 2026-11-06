# Báo cáo chỉnh sửa — bản thảo bài báo + tài liệu luận văn, 01/11/2026

Sửa trực tiếp vào tệp đang có, không tạo bản "enhanced".

## 1. Bản thảo bài báo (.docx, sửa tại chỗ)

`plans/261031-hoi-dong-phan-bien-danh-gia-khoa-hoc/Bao_cao_luan_van_dang_bai_bao_khoa_hoc (1).docx`
XML thân bài 107,961 → 135,475 ký tự. Giữ nguyên 15 khối `m:oMath`, 4 `m:oMathPara`, 2 hình, 16 siêu liên kết, 4 bảng, 19 phần trong gói — kiểm lại sau khi ghi, không mất phần nào.

### 1.1 Hạng mục hình thức theo Điều 6.1
- Tên bài tiếng Anh (style `Subtitle`) ngay dưới tên VN.
- Khối tác giả / đơn vị / liên hệ / ORCID — **placeholder có nhãn**, kèm SourceNote giải thích vì sao để trống.
- `Abstract` tiếng Anh + `Keywords` tiếng Anh, nội dung khớp tóm tắt VN kể cả phần khoảng tin cậy.
- Bốn mục mới trước TLTK: Phụ lục A, Phụ lục B, Đóng góp khoa học, Tuyên bố xung đột lợi ích, Lời cảm ơn và nguồn tài trợ.
- TLTK [13] ONNX và [14] NVIDIA TensorRT: thêm nhà phát hành, phiên bản, dạng `[Trực tuyến]. Địa chỉ:`, ngày truy cập, và ghi rõ không có DOI. Siêu liên kết gốc giữ nguyên.

### 1.2 Khuyết điểm số liệu đã bịt
1. **84.2% không còn là hằng số.** Tóm tắt VN, Abstract EN và §7 Kết luận đều mang CI 95% theo nguồn [77.4, 90.0] và dải tái lập 83.25–84.71%.
2. **Bảng 3 có khoảng tin cậy.** Mỗi ô BA xuống dòng thứ hai mang CI bootstrap cụm theo nguồn: B1 [77.4, 90.0], B1w [78.1, 89.8], ESTA [78.2, 89.4], TSF [76.6, 90.2], B1xl [75.7, 88.9], B2 [76.1, 86.4]. Cột 1 hẹp lại 2592→1800 twip, cột BA rộng ra 1728→2520, tổng 9,504 giữ nguyên nên `tblGrid` không lệch.
3. **§5.1 nói rõ nửa độ rộng CI (6.3 pp) lớn hơn mọi hiệu số trong bảng** (hiệu lớn nhất 2.7 pp), kèm tỷ số 2.9× giữa CI theo nguồn và CI theo clip, và dải BA từng fold 67.5–97.2%.
4. **Thứ hạng Bảng 3 được kiểm.** BH q=0.05 trên họ 6 phép so: chỉ B1xl tách khỏi B1, và tách về phía **kém hơn** (−1.46 pp, [−2.75, −0.42], p=0.0032). B1w +0.19 pp p=0.906; TSF −0.09 pp p=0.923. Đoạn mới nói thẳng "0.2 pp B1w hơn B1 nằm sâu trong nhiễu", đồng thời không suy ra "tương đương".
5. **Nhiễu bối cảnh có đối chứng.** §5.2 thêm Cramér's V 0.784, 59.17% entropy nhãn, sàn chỉ-biết-bối-cảnh 52/64 = 81.25% so với đầu quan hệ 54/64 = 84.38%, McNemar p = 0.804, bền qua ba cách xử lý trùng lặp; đối chứng NP-125 AUC 0.8485 → 0.6426, CI [0.4923, 0.8443] chứa 0.50.
6. **Rào va chạm số liệu mới.** 81.2% (macro recall, 516 clip) ≠ 81.25% (tỷ lệ đúng, 52/64) — đã đóng khung trong SourceNote.
7. **Khảo sát hậu nghiệm được công bố** (Phụ lục A.1): 20 cấu hình, 9 cấu hình vượt 84.17%, cao nhất lr=3e-4 ở 85.55% (+2.30 pp so đối chứng cùng môi trường 83.25%, vượt dải 1.46 pp) nhưng CI [−0.32, +6.03] chứa 0, p=0.107, BH không bác bỏ. Có nêu gradclip 5.0 = 83.56%.
8. **Bền vững cấp nguồn** (Phụ lục A.2): 0/151 nguồn mang cả hai nhãn; Gini 0.579; một nguồn 102 clip (9.3%); ICC 0.563; 111 đúng hoàn toàn / 15 sai hoàn toàn / 25 lẫn; đối trọng thuận là macro theo nguồn 85.09% > 84.17% (+0.92 pp), CI [79.89, 89.98].
9. **Đối chứng đổi bộ phát hiện** (Phụ lục A.3): ESTA @ YOLO26s 81.6%, CI [74.4, 88.5]; thấp hơn ESTA @ YOLOv11s 2.2 pp; CI hiệu số [−1.7, +6.0] chứa 0. Dán nhãn đối chứng, không phán quyết.
10. **Giấy phép và dữ liệu** (SourceNote Bảng 1 + Phụ lục B): iSafetyBench CC BY-NC-SA 4.0 với điều kiện ShareAlike, Unsafe-Net CC BY 4.0; không phát hành lại video gốc.
11. **Giới hạn tái lập** (Phụ lục B): bản ghi Jetson sinh từ cây làm việc có thay đổi chưa cam kết; ba bản ghi cùng phiên 167.0 / 167.3 / 167.7 ms → phân tán 0.41%. κ duy nhất là 0.65 trên làn bốn lớp, **chưa có** κ cho làn nhị phân, nên bài không phát biểu gì về trần nhãn.
12. **Cắt độ chính xác giả tạo.** 17 giá trị: 289.672→289.7, 167.291→167.3 (3 chỗ), 322.708→322.7, 198.224→198.2, 160.765→160.8, 50.867→50.9, 58.630→58.6, 58.052→58.1, 0.798→0.80, 0.842→0.84, 2.055→2.06, 2.182→2.18, 227.214→227.2, 5.978→5.98, 1.732→1.73, 5.045→5.05, 4.401→4.40. SourceNote Bảng 4 giải thích: phân tán giữa ba lần chạy là 0.41%, lớn hơn chữ số thập phân thứ ba.
13. Hàng chia ngẫu nhiên 97.7% trong §5.2 nay có CI [96.5, 98.7] — cho thấy khoảng cách LSO ↔ chia ngẫu nhiên là hiệu số duy nhất **không** bị nhiễu hấp thụ.

Mọi số bổ sung đều dán nhãn: tính 01/11/2026, hậu đóng băng, hậu nghiệm, từ **đúng tệp dự đoán ngoài fold đã đóng băng**, không huấn luyện lại, không chạm tập kiểm thử, không đổi phán quyết cổng nào. Không có số nào bắc qua ranh giới hai làn.

## 2. Tài liệu luận văn (`report/vn/`) — 5 lỗi câu chữ, ngoại lệ #4

Theo quyết định của tác giả: sửa đúng 5 lỗi, **không thêm số đo mới**, ghi ngoại lệ thứ 4 vào `00-trang-thai-bien-soan.txt`.

1. `chuong-06-danh-gia.txt` — dấu của G17 sai ở 4 chỗ. Δ = **+2.2 pp** là *v11s − 26s* (83.8 − 81.6), tức bộ phát hiện mới hơn cho kết quả thấp hơn. Lời biện "CI in ngược" bị xóa vì sai. Xác nhận độc lập bằng `sign_audit` trong `detector-number-provenance.json` (toán hạng 83.834 và 81.618, hiệu +2.2164 pp).
2. `chuong-03-phuong-phap.txt` — câu "không có phép so nào giữa gradclip 1.0 và 5.0" được thu hẹp phạm vi về "tập số đã đóng băng 05/10", kèm HIỆU ĐÍNH trỏ `sweep_results/posthoc_sensitivity/gradclip_5.0_oof.json`. Cố ý **không** nhập giá trị 83.56 vào chương.
3. `chuong-04-chi-tiet-thuc-nghiem.txt` — hiệu đính song song, trỏ về Ch3 §3.6.
4. `chuong-05-trien-khai-bien.txt` — `0.49%` → `0.87%` cho dải 309.08–311.78 ms; `53.2 %` → `52.6 %, khớp Bảng 5.2`. Giữ nguyên `46.8 %` vì đó là tỷ lệ giải mã 1080p, không phải 46.6 % của Bảng 5.2.
5. `chuong-06-danh-gia.txt` §6.12.5 — bỏ quy 2,262 cho nhóm tác giả gốc (họ chưa công bố tập nhãn nên không biết quy mô); 2,262 là số của chính luận văn. Kết luận không đổi; số chính xác để mở ở Q7.

Ngoại lệ #4 ghi: 5 sửa đổi, không số đo mới, không huấn luyện lại, ngân sách chạm tập kiểm thử vẫn còn 1 lần, bảng cổng không đổi (**tám cổng, hai đạt, sáu trượt**), kèm danh mục điều cấm.

## 3. Tệp không đóng băng

- `analyze_round3_tower_bh.py` + `chien-luoc-thuc-nghiem-vong-3.md`: "meanpool thắng 4/8" → **6/8**, đếm lại từ `round3_tower_bh.json`.
- `README.md`: khối "Honest headline" không còn dùng lại con số đã thu hồi "3.02–3.04 dec/s"; nêu rõ phép đo P3 phụ thuộc clip **và** xung nhịp nên không đỡ được một phán quyết nhịp duy nhất; "real-time" chỉ còn áp cho P1. Lệnh cấm số thu hồi và cấm so chéo làn giữ nguyên nguyên văn.
- `PROGRESS.md`: thêm nhãn ⛔ SUPERSEDED đầu tệp, liệt kê các khẳng định đã thu hồi (56/64, ladder 52→56/64, +3.50 pp macro, 57/64), tự dẫn r = −0.066 của chính tệp; sửa phép tính 2,262 ở hai chỗ và thay mục cây thư mục chết (`BASELINE.md`, `GAP_ANALYSIS.md`) bằng `docs/`.

## 4. Tạo tác

7 tệp trong `sweep_results/review_261101/` (bị `.gitignore:12` loại khỏi git). Bản ghi cam kết được là 7 tập lệnh `tools/compute-*.py` / `tools/audit-*.py`, chạy lại bằng `PYTHONPATH=. python3 tools/<script>.py`. `headline-cluster-ci.json` đã được xác nhận tái lập trùng hash.

## 5. Chưa làm

- **Kiểm tra trùng lặp (Điều 5.1, bắt buộc)** — chưa chạy; không có công cụ trong môi trường.
- Khối tác giả vẫn là placeholder.
- κ làn nhị phân iSafetyBench chưa đo (hạn 14/11).
- Chưa đối chiếu quy định riêng của tạp chí (Điều 6.2) vì chưa biết tạp chí.

## Câu hỏi chưa giải quyết

1. Tạp chí đích là tạp chí nào? Điều 6.2 cho tạp chí quyền ghi đè cấu trúc Điều 6.1.
2. Chạy kiểm tra trùng lặp ở đâu, ngưỡng bao nhiêu? Rủi ro cao hơn thường vì bài dùng chung kho dữ liệu và cùng mục tiêu tái lập với Önal & Dandıl [1][2].
3. Tên, đơn vị, email công vụ, ORCID của tác giả?
4. Dòng ESTA @ YOLO26s (81.6%, CI [74.4, 88.5]) để ở Phụ lục A.3 hay đưa thành dòng thứ 7 của Bảng 3?
5. Bài rút từ luận văn đã bảo vệ — tạp chí có coi là công bố trùng lặp theo Điều 5.2?
6. Q7 (2,262 so với 1,901 + 343) vẫn mở; bài báo hiện không dùng số nào trong nhóm này.
7. Có cần bản .docx thứ hai rút gọn cho giới hạn số từ của tạp chí? Bản hiện tại đã dài hơn sau khi bổ sung.
