# Audit văn phong nghiên cứu & mạch biện luận — Ch1, Ch2, Ch5, Ch8

Phạm vi đọc: `00-trang-thai-bien-soan.txt` (ràng buộc viết), `00-tom-tat.txt`, `chuong-01-mo-dau.txt`,
`chuong-02-tong-quan.txt`, `chuong-05-trien-khai-bien.txt` (bản NỘP, bỏ qua bản ghim-xung-nhịp),
`chuong-08-ket-luan.txt`. KHÔNG đọc Ch3/4/6/7 — mọi kết luận "THIẾU trong Ch2" chỉ nói về Ch2, không
suy ra là thiếu toàn luận văn (lý do thật có thể nằm ở Ch1/Ch6/Ch7, ngoài phạm vi).

Quy ước: CÓ = có mặt + dòng số dẫn chứng. THIẾU = không tìm thấy trong phạm vi đọc (đã `grep` xác
minh, không suy đoán). Không bịa số.

---

## A. MẠCH LUẬN theo chương (câu hỏi → giả thuyết → vì sao PP này → bằng chứng → giới hạn)

### Ch1 (mở đầu) — mạch MẠNH NHẤT trong 4 chương, nên giữ nguyên cấu trúc

| Khâu | Vị trí | Nội dung |
|---|---|---|
| Động lực 1 (kỹ thuật) | 12-37 | lai YOLO-Transformer, lý do 3 yêu cầu (a)(b)(c) |
| Động lực 2 (PP luận, thành đóng góp chính) | 39-43 | tường minh: "động lực thứ hai... cuối cùng trở thành đóng góp chính" |
| Phát biểu vấn đề + hợp đồng nhịp (công thức f/s) | 46-115 | input/output/nhịp, 2 kho, trần nhãn κ=0.65 |
| Câu hỏi nghiên cứu (dạng đo lường, có công thức Δ) | 131-147 | "Δ lớn bao nhiêu, có ổn định không" — **không phải câu hỏi tối ưu hoá**, nói rõ lý do (144-147) |
| Mục tiêu + bảng 4 thay đổi so với đề cương, mỗi dòng có "cam kết — đã làm — vì sao" | 150-217 | ví dụ: đổi xương sống vì gate G14 trượt (có CI, p-value) |
| Đóng góp (ưu tiên hoá, có số) | 218-315 | 5 đóng góp |
| Phạm vi/giới hạn | 316-359 | trong/ngoài phạm vi + 4 giới hạn |

**Đứt mạch duy nhất đáng ghi (nhẹ):** câu hỏi được đặt là câu hỏi **đo lường** (dòng 139-144), không
có một câu "giả thuyết" tường minh kiểu "chúng tôi giả thuyết rằng Δ > 0" — khâu "giả thuyết" của
chuỗi 5 bước mà GVHD/đề bài yêu cầu bị thay bằng khâu "đại lượng đo được + tiêu chí bác bỏ". Đây là
lựa chọn có chủ ý (dòng 144-147 tự giải thích), không phải lỗi, nhưng nếu đọc đúng nghĩa đen "giả
thuyết" thì khâu này **THIẾU trong Ch1** — bằng chứng kiểm định giả thuyết dưới dạng gate (G-xx) chỉ
xuất hiện ở Ch6/8, ngoài Ch1.

### Ch2 (tổng quan) — xem mục B riêng (trọng tâm GVHD)

### Ch5 (triển khai biên) — mạch chặt chẽ trong TỪNG MỤC, nhưng cấu trúc toàn chương có vấn đề văn phong

- Mục 5.1 ("câu hỏi của chương") đứng ở dòng **158**, nhưng **hai khối CẢNH BÁO/rút lại rất dài đứng
  trước nó ở dòng 1-141** — độc giả gặp lịch sử sửa lỗi và số liệu bị rút trước khi gặp câu hỏi
  chương đang trả lời. Đây là **đứt mạch về trình tự đọc**: bằng chứng/đính chính đến trước câu hỏi,
  đảo ngược trình tự "câu hỏi → bằng chứng". Nội dung từng khối là đúng và cần giữ, nhưng vị trí nên
  xem lại (xem phần "nên sửa gì" ở cuối).
- Từ 5.2 đến 5.8 (222-1222): mỗi mục đều có mẫu "số liệu → vì sao đo thế này → so với gì → giới hạn
  của chính số đó" — xem chi tiết ở mục D.
- 5.9 "những gì chưa được đo" (1223-1277) và 5.10 "kết luận chương + phát biểu có thể bị phủ định"
  (1278-cuối) đóng mạch tốt: nêu rõ điều kiện nào sẽ đảo kết luận.

### Ch8 (kết luận) — đóng đúng 3 câu hỏi của Ch1, mạch tốt, một điểm hở ngược dòng

- 8.4 (457-493) trả lời tường minh 3 câu hỏi, câu 3 là câu trả lời **phủ định** (479-485) — khớp với
  RQ đo lường đặt ra ở Ch1 dòng 139-147 và với "đóng góp chính" nêu ở Ch1 dòng 40-43. Mạch đóng tốt.
- 8.2.1 (135-182): dữ liệu phản biện quyết định (G17, YOLOv11s vs YOLO26s cho ESTA, dòng 152-159)
  **xuất hiện lần đầu ở Ch8**. Ch1/Ch2 (trong phạm vi đọc) không hề đặt trước câu hỏi "thế hệ bộ phát
  hiện có thể là biến gây nhiễu" — khâu "giả thuyết/dự kiến rủi ro" của chuỗi 5 bước bị thiếu ở
  **thượng nguồn** (Ch1/Ch2), chỉ khâu "bằng chứng" xuất hiện ở hạ nguồn (Ch8). (Ch2 dòng 50-53 có
  nêu hiện tượng này dưới dạng "hệ quả thực nghiệm đáng ghi nhận" — nhưng đó là kết quả, không phải
  rủi ro được dự kiến trước.)
- 8.3.9 (421-455): quyết định giữ balanced accuracy làm đại lượng tiêu đề — lập luận đầy đủ, có ngày
  chốt, có 3 ràng buộc cho đợt sau. Đây là **phần tốt, không nên sửa** (xem mục E).

---

## B. CHƯƠNG 2 — có dẫn đến lựa chọn phương pháp, hay chỉ là danh mục? (trọng tâm GVHD)

| # | Lựa chọn lớn | Ch2 có nêu phương án đối thủ? | Ch2 có nêu ưu/nhược mỗi phương án? | Ch2 có nêu rõ *vì sao chọn*? | Kết luận |
|---|---|---|---|---|---|
| 1 | Xương sống CLIP đóng băng thay video backbone | CÓ (92-100: TimeSformer/ViViT/VideoMAE) | MỘT PHẦN — chỉ có số + CI/p (92-100), không có lập luận chi phí/triển khai trong Ch2 | THIẾU — "không bỏ xa" (102-107) là kết quả null, không phải tiêu chí quyết định; tiêu chí thật (triển khai biên, đặc trưng đóng băng) nằm ở Ch1/Ch5 | **CÓ (yếu)** — dòng 83-107 |
| 2 | Đầu Transformer thời gian thay ConvLSTM/3D-CNN | MỘT PHẦN — ConvLSTM (62-75) và 3D-CNN/I3D/SlowFast (77-81) có nêu | THIẾU — ConvLSTM chỉ là baseline mô tả (của bài báo gốc), 3D-CNN bị gạt bằng 3 dòng ("tốn kém, chưa test trên Unsafe-Net"), không có ưu/nhược đối chiếu trực tiếp với Transformer | THIẾU — không có câu "vì vậy luận văn chọn Transformer vì..." | **THIẾU** — dòng 62-81 |
| 3 | Phân rã theo thực thể | THIẾU — 2.2.4 chỉ 6 dòng (109-114), nói "lai YOLO+Transformer còn thưa trong tài liệu", không nêu phương án đối thủ nào (toàn khung là gì, ưu/nhược) | THIẾU | THIẾU — lý do thật (3 gate G11/G14/G18 trượt) nằm ở Ch8 (135-182), ngoài Ch2 | **THIẾU rõ nhất** — dòng 109-114 |
| 4 | Leave-source-out thay k-fold ngẫu nhiên | CÓ — toàn mục 2.4 (179-281): leakage [29][30], group CV ở ngành khác [31][32][37], context bias [33][34][35][36][42] | CÓ — k-fold ngẫu nhiên: ưu (dễ, công cụ sẵn) nhược (lạc quan giả — 216-222); leave-source-out: ưu (test ngoài phân phối thật) nhược (giảm lực phân giải — ngụ ý, không nói thẳng "nhược" của PP mình chọn trong Ch2) | CÓ — 2.4.4 khoảng trống (248-263) | **CÓ** — dòng 179-281, mục mạnh nhất chương |
| 5 | Balanced accuracy làm đại lượng tiêu đề | THIẾU — `grep "balanced accuracy\|cân bằng\|metric\|chỉ số"` trong Ch2: **0 kết quả liên quan đến lựa chọn đại lượng** | THIẾU | THIẾU — quyết định (và lý do từ chối đổi sang recall) chỉ nằm ở Ch8 §8.3.9 (421-455), chốt ngày 11/10/2026, SAU khi Ch2 đã đóng | **THIẾU hoàn toàn** — không một dòng nào trong Ch2 |
| 6 | Triển khai Jetson | MỘT PHẦN — mục 2.5 (284-318) có nêu công cụ (ONNX/TensorRT/CoreML — CoreML chậm hơn CPU, 295-298) và thông lượng Jetson trước đây (300-301) | THIẾU — không so với thiết bị biên khác (Raspberry Pi, Coral, mobile SoC); chỉ so công cụ biên dịch trên cùng một họ thiết bị (Jetson) | THIẾU — không có đoạn "vì sao Jetson, không phải X" | **THIẾU** (chọn thiết bị) — dòng 284-318 nêu được *khoảng trống phương pháp đo* (đơn vị đo, phép cộng giai đoạn) rất tốt, nhưng đó là lựa chọn (6b) "cách đo", không phải lựa chọn (6a) "thiết bị Jetson" |

**Bảng tóm tắt theo đúng format yêu cầu:**

| Lựa chọn lớn | Ch2 có biện luận? |
|---|---|
| (1) CLIP đóng băng thay video backbone | CÓ (yếu — null-result, thiếu tiêu chí quyết định) |
| (2) Transformer thay ConvLSTM/3D-CNN | THIẾU |
| (3) Phân rã thực thể | THIẾU (thiếu rõ nhất) |
| (4) Leave-source-out thay k-fold | CÓ (mạnh nhất) |
| (5) Balanced accuracy | THIẾU (0 dòng) |
| (6) Triển khai Jetson (chọn thiết bị) | THIẾU |

**Nhận định tổng:** 1/6 lựa chọn được Ch2 biện luận đầy đủ (leave-source-out), 1/6 biện luận yếu
(CLIP backbone — có số nhưng thiếu tiêu chí quyết định), 4/6 THIẾU. Đây đúng là bằng chứng cụ thể
cho nhận xét GVHD: với phần lớn lựa chọn lớn, Ch2 đọc như **danh mục công trình có tổ chức tốt**
(mục 2.1-2.3, 2.5 đều có cấu trúc rõ, dẫn chứng tốt) nhưng **không khép vòng về lý do chọn** — vòng
khép chỉ xảy ra đầy đủ ở mục 2.4, và một phần ở 2.2.3.

---

## C. ƯU–NHƯỢC theo mô hình/pipeline (trong phạm vi 5 tệp đọc)

Trước hết: `grep` "B1xl", "B2", "TSF" trên toàn bộ 5 tệp → **0 kết quả**. Ba tên này không xuất
hiện trong Ch1/Ch2/Ch5/Ch8/Tóm tắt — có thể chỉ xuất hiện ở Ch6/Ch7 (ngoài phạm vi đọc của tôi).
Không bịa nội dung cho 3 model này.

| Model/pipeline | Ưu có nêu? (dòng) | Nhược có nêu? (dòng) | Ghi chú |
|---|---|---|---|
| **B1** (CLIP đóng băng + đầu 201,090 tham số, đã triển khai) | CÓ — 84.2% / +20.9pp trên sàn (Ch8:68-73); đầu chỉ tốn 1.7%/0.3% chi phí (Ch8:94-99); quyền riêng tư theo kiến trúc (Ch8:105-108) | CÓ — 84.2% khác 84.4% của B1w, không được trích "max" (Ch5:329-336); đánh giá sliding-window ≠ whole-clip, cảnh báo không đọc như "độ chính xác vòng lặp" (Ch5 mục 5.4.3, dòng 396-421); CI so với encoder thực thể tốt nhất **chứa 0** (Ch8:78) | Đầy đủ nhất trong 5 tệp |
| **B1w** (đầu 333,570 tham số) | CÓ — 84.4% > B1 (Ch2:96; Ch5:329) | CÓ — chưa xuất ONNX, chưa kiểm tra tương đương, "84.4% không được trích là tiêu đề" (Ch5:1259 "Engine B1w. Chưa xuất, chưa kiểm tra tương đương") | |
| **B1xl** | KHÔNG THẤY trong phạm vi đọc | KHÔNG THẤY | Có thể ở Ch6/7 |
| **B2** | KHÔNG THẤY trong phạm vi đọc | KHÔNG THẤY | Có thể ở Ch6/7 |
| **ESTA** | THIẾU — không có câu nêu lợi ích/ưu điểm của ESTA | CÓ — gate G14 trượt, biên −0.5pp, CI chứa 0, p=0.683 (Ch8:143-145); YOLOv11s 83.8% vs YOLO26s 81.6%, thế hệ mới cho kết quả thấp hơn (Ch8:156-159) | Chỉ có nhược, không có ưu trong phạm vi đọc |
| **TSF** | KHÔNG THẤY trong phạm vi đọc | KHÔNG THẤY | Có thể ở Ch6/7; có thể là viết tắt khác không khớp "TSF" |
| **E0–E3** (encoder quan hệ) | MỘT PHẦN — dưới leave-source-out, thang trải 5.1pp có phân biệt được, nêu như minh chứng phương pháp (không phải ưu của riêng từng E) (Ch8:46-51) | CÓ — dưới k-fold rò rỉ, thang sụp xuống 1.4pp, **không cặp nào có ý nghĩa** (Ch8:47-48); trên Unsafe-Net, **thứ tự đảo ngược** theo giao thức (Ch8:49-51); E0 vượt E3 +5.1pp trên iSafetyBench nhưng R47 (bất biến kiến trúc) bị phản chứng (Ch8:177-182) | Không có bảng ưu/nhược TÁCH riêng cho từng E0/E1/E2/E3 — chỉ có như điểm dữ liệu cho luận điểm "giao thức xoá khả năng phân biệt" |
| **P1** (pipeline giai đoạn trước) | CÓ — bằng chứng đa luồng/soak duy nhất hiện có (Ch5:982-1003, "kết quả 4 luồng là của P1") | CÓ, rất đầy đủ — xuất xứ KHÔNG XÁC ĐỊNH (Ch5:908-926); 5 khiếm khuyết K1-K5 phải sửa (Ch5 mục 5.6); các phát biểu đã rút (Ch5:1004-1029) | Nhược áp đảo ưu — đúng bản chất "đồ thị không triển khai" |
| **P3** (pipeline thực thể-quan hệ) | CÓ — chi phí biên của đầu quan hệ chỉ 0.051%/0.158ms ngân sách (Ch5:822-830) | CÓ, rất đầy đủ — TRƯỢT hợp đồng 3qđ/giây dưới DVFS trên 3/3 đoạn test (Ch5:763-773); Bảng 5.4 (tổng giai đoạn) bị RÚT làm phán quyết hợp đồng, thay bằng Bảng 5.4b (Ch5:725-748); đuôi phân phối xấu hơn dưới DVFS (Ch5:869-880); điểm vận hành đo (800px) ≠ điểm đã chọn (640px) (Ch5:880-899) | |
| **Quy tắc đếm** (sàn 63.3%/67.2%) | CÓ, nhưng khung khác: không phải "ưu của một phương pháp được thử" mà là **lựa chọn bất lợi có chủ ý** để làm hẹp biên luận văn (Ch8:37-40) | — (không áp dụng theo khung "nhược", vì nó là baseline tối giản, bản chất là yếu) | Hai sàn cùng tên khác trị số, không trừ chéo (Ch8:69-71) — điểm cần đọc kỹ, dễ nhầm |

**Nhận định C:** Các pipeline/mô hình **đã triển khai hoặc đã đo trên thiết bị** (B1, B1w, P1, P3)
có ưu-nhược rất đầy đủ, chủ yếu ở Ch5/Ch8. Các mô hình **chỉ xuất hiện như kết quả gate** (ESTA,
E0-E3) chỉ có nhược (kết quả âm) trong phạm vi đọc — không có khung "ưu" vì bản thân kết quả là phủ
định; đây không phải lỗi văn phong mà là bản chất phát hiện phủ định. B1xl/B2/TSF không tìm thấy
trong 5 tệp — không thể kết luận có/thiếu.

---

## D. CHƯƠNG 5 — số liệu triển khai có nêu phép đo + công thức + mốc so sánh không?

| Đại lượng | Phép đo nêu? | Công thức nêu? | Mốc so sánh nêu? | Dòng |
|---|---|---|---|---|
| Nhịp quyết định B1 (167.291 ms / 5.978 qđ/giây) | CÓ — vòng lặp tổ hợp, không phải tổng giai đoạn | CÓ — so sánh trước/sau sửa, hệ số 1.7315 | CÓ — 289.672 ms trước sửa; hợp đồng 1.249/1.427/2.500 qđ/giây theo p50/mean/p95 khung hình; biên 4.79/4.19/2.39 lần | Ch5:422-613; Ch8:87-104 |
| Hợp đồng nhịp (f/s quyết định/giây) | CÓ | CÓ — công thức tường minh *f*/*s* | CÓ — 3 mức theo phân bố tần số khung | Ch1:65-71; Ch5 mục 5.5.1 (713-722) |
| Chi phí từng giai đoạn (Bảng 5.2, 5.5) | CÓ | CÓ — phân biệt tổng/marginal, 2 mẫu số khác nhau (1.7% vs 0.3%) | CÓ — trước/sau sửa tiền xử lý; so với đầu E0 | Ch5:366-421, 808-868; Ch8:92-99 |
| Năng lượng theo giai đoạn (Bảng 5.6) | CÓ — quy ước total/marginal nêu rõ | MỘT PHẦN — công thức marginal suy từ idle baseline chung, có nêu | CÓ — % so với tổng; so giữa B1/P3 | Ch5 mục 5.7.1 (1030-1081) |
| Năng lượng vòng lặp TỔ HỢP | **THIẾU — tự nhận "chưa đo"** | — | — | Ch5:1223-1277 (mục 5.9), Ch8:311-312 (8.3.3c) |
| Bộ nhớ (3 con số khác nhau) | CÓ — phân biệt rõ 3 định nghĩa, 1 con số bị RÚT (417.84MB, 251.1MB) | CÓ | CÓ — so giữa B1 (3 engine) / P3 (4 engine) | Ch5 mục 5.7.2 (1082-1124) |
| Độ bền nhiệt 30 phút | CÓ — có log VDD_IN, có cơ chế "idle droop" đối chứng qua quy tắc phân xử tiền đăng ký | CÓ | CÓ — DVFS vs ghim nhịp (Bảng 5.3) | Ch5 mục 5.7.3 (1125-1170) |
| Phép cộng tổng-giai-đoạn (P3, Bảng 5.4) | CÓ nêu, nhưng **bị RÚT làm phương pháp phán quyết hợp đồng** — tác giả tự công bố sai số đổi dấu theo xung nhịp (+7.98%/−39.3%) | CÓ | CÓ — thay bằng Bảng 5.4b (đo tổ hợp thật) | Ch5:710-808; Ch8:126-129 |

**Nhận định D:** Chương 5 đáp ứng tốt yêu cầu "phép đo + công thức + mốc so sánh" cho **hầu hết** số
liệu — đây là điểm mạnh thật, không phải liệt kê bảng trần trụi. Ngoại lệ tự nhận: năng lượng vòng
lặp tổ hợp hoàn toàn chưa đo (không phải lỗi văn phong, là khoảng trống đo đạc được khai báo trung
thực). Vấn đề của Ch5 không nằm ở TỪNG con số (đã đủ phép đo/công thức/mốc so sánh) mà ở **cấu trúc
trình bày toàn chương**: 2 khối cảnh báo dài (141 dòng) đứng trước câu hỏi chương, và việc liệt kê
chuỗi sự cố/rút lại theo trình tự thời gian ở nhiều mục (5.2.2, 5.6.5, 7.6 dẫn lại) — nội dung đúng,
nhưng HÌNH THỨC trình bày dễ đọc thành "biên bản sự cố" hơn là "luận". Đây có thể là phần đang bị
GVHD đọc nhầm thành "liệt kê" dù bản chất là rigor cao.

---

## E. PHẦN ĐÃ TỐT — KHÔNG NÊN SỬA

1. **Ch1 toàn chương** (12-359): chuỗi động lực→vấn đề→câu hỏi đo lường→mục tiêu (có bảng 4 thay đổi
   với lý do bằng chứng)→đóng góp→giới hạn. Đây là mẫu mạch luận chuẩn nhất trong 4 chương, giữ
   nguyên cấu trúc này khi sửa các chương khác.
2. **Ch2 mục 2.4** (179-281): mẫu đúng cho "tổng quan dẫn đến lựa chọn" — có đối thủ, có ưu/nhược
   ngụ ý, có khoảng trống (a)(b)(c), có công cụ thống kê nêu trước để Ch6 không giải thích lại.
   Nên dùng mục này làm khuôn để viết lại các mục 2.2.4 và phần balanced-accuracy.
3. **Ch2 mục 2.1** (21-54): có lý do từ chối RT-DETR tường minh (41-46), có cả một "cảnh báo sớm"
   về rủi ro thế hệ bộ phát hiện (50-53) — mẫu tốt, không phải 1 trong 6 lựa chọn lớn nhưng đáng
   giữ làm ví dụ văn phong.
4. **Ch5 mục 5.9 + 5.10** (1223-cuối): liệt kê "chưa đo" + phát biểu có thể bị phủ định — đúng chuẩn
   khoa học, không nên rút gọn hay xoá vì nó là bằng chứng của rigor.
5. **Ch8 toàn bộ §8.1-8.2 và 8.4** (16-249, 457-493): quy nguyên nhân cho kết quả phủ định (8.1.5),
   bảng đối chiếu đã giao/đã rút (8.2.2), và 8.4 đóng đúng 3 câu hỏi của Ch1 — không sửa.
6. **Ch8 §8.3.9** (421-455): quyết định giữ balanced accuracy, có ngày chốt, có lý do phương pháp
   (không phải tiện lợi), có 3 ràng buộc cho đợt sau — mẫu chuẩn cho cách viết một quyết định có
   tranh cãi. Vấn đề duy nhất là **vị trí**: nội dung này nên có một bản rút gọn/tham chiếu chéo
   TRONG Ch2 (nơi balanced accuracy phải được giới thiệu là lựa chọn), không chỉ ở Ch8.

---

## F. 8 CHỖ ĐỨT MẠCH NẶNG NHẤT (toàn bộ 4 chương đã đọc)

1. **Ch2:109-114** — mục 2.2.4 "Lai YOLO+Transformer" (phân rã thực thể): 6 dòng, chỉ nêu khoảng
   trống tài liệu, không nêu phương án đối thủ/ưu-nhược/lý do chọn. Lý do thật (3 gate G11/G14/G18
   trượt) nằm ở Ch8:135-182, ngoài Ch2. **Nặng nhất** — đúng tâm điểm phê bình GVHD.
2. **Ch2 (toàn chương, 0 dòng)** — balanced accuracy làm đại lượng tiêu đề: không một dòng biện luận
   trong Ch2; quyết định + lý do chỉ xuất hiện ở Ch8:421-455, chốt sau khi đã có số (11/10/2026).
3. **Ch2:284-318** — mục 2.5 triển khai biên: không so Jetson với thiết bị biên khác; chỉ so công cụ
   biên dịch trong cùng một họ thiết bị.
4. **Ch2:62-81** — ConvLSTM (baseline mô tả) và 3D-CNN (gạt bằng 3 dòng) không được đối chiếu ưu/nhược
   trực tiếp với đầu Transformer đã chọn.
5. **Ch5:1-141** — hai khối cảnh báo/rút lại (141 dòng) đứng trước câu hỏi của chương (dòng 158),
   đảo trình tự "câu hỏi → bằng chứng"; nội dung đúng, vị trí gây đọc-nhầm thành liệt kê sự cố.
6. **Ch8:152-159** — dữ liệu phản biện G17 (YOLOv11s vs YOLO26s) xuất hiện lần đầu ở Ch8; không có
   "giả thuyết/rủi ro dự kiến" tương ứng ở Ch1/Ch2 trong phạm vi đọc.
7. **Ch1:139-147** — câu hỏi nghiên cứu được đặt dưới dạng đo lường, không có câu "giả thuyết" tường
   minh; khâu "giả thuyết" của chuỗi 5 bước bị thay bằng "đại lượng đo được + tiêu chí bác bỏ" (có
   chủ ý, nhưng là một đứt mạch hình thức nếu đọc đúng nghĩa đen yêu cầu).
8. **Ch2:92-107** — lựa chọn CLIP đóng băng thay video backbone chỉ được biện luận bằng một kết quả
   null (CI chứa 0, "không bỏ xa"), không có câu nêu tiêu chí quyết định thật (triển khai biên, đặc
   trưng đóng băng) — tiêu chí đó nằm ở Ch1/Ch5, không nằm ở Ch2 nơi lẽ ra phải biện luận lựa chọn.

---

## Câu hỏi chưa giải quyết

1. B1xl, B2, TSF không xuất hiện trong 5 tệp đã đọc (Ch1, Ch2, Ch5, Ch8, Tóm tắt) — cần xác nhận
   chúng có được định nghĩa ở Ch6/Ch7 không, hay là tên khác/đã đổi tên.
2. Ch2 mục 2.2.4 (phân rã thực thể) chỉ 6 dòng — có nên bổ sung ngay trong Ch2 phần ưu/nhược + lý do
   chọn tạm thời (trước khi biết kết quả gate), hay giữ nguyên hiện trạng và chỉ thêm câu dẫn chiếu
   chéo tới Ch8 §8.2.1? Cách nào tránh rủi ro "viết lại sau khi biết kết quả" (hindsight bias) mà
   luận văn đang rất nghiêm ngặt tránh ở các chỗ khác (ví dụ §8.3.9)?
3. Balanced accuracy: Ch8 §8.3.9 đã có đầy đủ lý do giữ nguyên — có nên chuyển/tóm tắt đoạn lý do đó
   sang Ch2 (nơi đáng lẽ phải giới thiệu lựa chọn đại lượng) mà không phạm quy tắc "không viết lại
   sau khi biết kết quả" đã nêu ở 00-trang-thai-bien-soan.txt?
4. Phần "đứt mạch" #5 (hai khối cảnh báo 141 dòng đứng trước câu hỏi chương ở Ch5) — có nằm trong
   phạm vi "KHÔNG sửa tệp luận văn" của nhiệm vụ audit này hay tác giả có ý định tự sửa cấu trúc
   sau? Tôi chỉ chỉ ra, không đề xuất bản sửa cụ thể vì ngoài phạm vi (chỉ đọc, không sửa).
5. Mục C (ưu-nhược) cho thấy ESTA/E0-E3 chỉ có "nhược" (kết quả âm) trong 5 tệp đọc — cần xác nhận
   với Ch6/Ch7 liệu có đoạn nào nêu "ưu" dự kiến ban đầu của các mô hình này (trước khi biết kết quả
   gate) để đối chiếu đầy đủ ưu/nhược như đề bài yêu cầu, vì phạm vi đọc của tôi không phủ Ch6/Ch7.
6. Giữa 2 cách đọc "giả thuyết" (axiom đo lường vs hypothesis cổ điển) ở đứt mạch #7 — đây có phải
   là điểm GVHD thực sự muốn, hay GVHD chỉ muốn Ch2 dẫn rõ hơn tới lựa chọn PP (mục B)? Cần hỏi lại
   GVHD để tránh sửa nhầm chỗ không phải trọng tâm phê bình.
