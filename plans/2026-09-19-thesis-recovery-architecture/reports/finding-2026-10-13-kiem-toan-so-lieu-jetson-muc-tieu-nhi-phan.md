# Kiểm toán số liệu Jetson của mục tiêu bài toán (phát hiện nhị phân An toàn/Không an toàn)

Ngày 2026-10-13. 0 phép đo mới. 0 lần chạm tập kiểm thử. Toàn bộ số dưới đây đọc lại từ
`deployment/artifacts/jetson-measurements/*.json` hoặc là phép chia hai trường trong cùng tạo tác.

## 1. Phạm vi

Mục tiêu đề tài = quyết định nhị phân An toàn/Không an toàn. Đường triển khai = **B1**
(CLIP ViT-B/32 đóng băng → đầu Transformer thời gian). Chỉ số Jetson của mục tiêu nằm trong
`r48-composed-cadence-fp16-*.json`. P3 (quan hệ thực thể) và P1 (giai đoạn trước) **không** là
đầu ra của mục tiêu; số của chúng chỉ vào chương để trả lời câu hỏi chi phí.

## 2. Cách chương 05 chọn con số: trung vị theo nhánh

Mỗi nhánh có n=3 lần chạy. Chương lấy **lần chạy trung vị theo `steady_state_ms_per_decision.p50`**:

| nhánh | ba lần chạy (ms) | lần trung vị được dùng |
|---|---|---|
| trước khi sửa, kho 720p | 289.672 / 289.339 / 292.834 | `20261001T233159` |
| đã triển khai, kho 720p | 167.009 / 167.690 / 167.291 | `20261003T155843` |
| nguồn 1920×1080 | 227.214 / 226.316 / 228.368 | `20261003T160245` |

Bảng ngân sách chi phí dựng đúng trên lần trung vị: 50.867/109.040 = **46.65 %**,
57.399/109.040 = **52.64 %**, 0.854/109.040 = **0.78 %** → in 46.6 / 52.6 / 0.8. Cột khoảng quyết
định: 30.41 / 34.31 / 0.51 / 31.30 %, phần dư 3.5 %. `duty_cycle_compute` là **trường có sẵn**
trong tạo tác (0.7615 → 76.1 %; 0.6311 → 63.1 %; 0.3768 → 37.7 %), không phải tỉ số của hai trung vị.

## 3. Bốn mâu thuẫn: ba tan, một là khiếm khuyết thật

### 3.1 `53.2 %` (ch05:22) so với `52.6 %` (bảng ngân sách) — **KHIẾM KHUYẾT THẬT**

53.2 % là **tỉ số trộn hai lần chạy**: tháp 58.052 ms của `155806` chia cho phần tính toán
109.040 ms của `155843`. Tỉ số đúng trong từng lần chạy:

```
155728: 58.416 / 109.114 = 53.54 %
155806: 58.052 / 110.201 = 52.68 %
155843: 57.399 / 109.040 = 52.64 %   <- lần trung vị, là con số của chương
58.052 / 109.040 = 53.24 %           <- cái đang in ở ch05:22, không thuộc lần chạy nào
```

Sửa văn bản: `53.2 %` → `52.6 %`. Kết luận nút cổ chai **không đổi** (tháp vẫn là hạng mục lớn
nhất ở cả ba lần chạy). Ghi chú kèm: ch05 ý (3) trích 58.052 ms và 2.48 lần (của `155806`; lần
trung vị cho 57.399 ms và 2.45 lần) và hàng giải mã trích 2.182 ms (của `155806`; trung vị 2.197).
Hai chỗ này **hợp lệ** vì Bảng 5.2 in cả ba lần chạy — nhưng nếu muốn nhất quán tuyệt đối thì
chương nên nói rõ "lần trung vị" một lần ở đầu mục 5.4.4.

### 3.2 `46.8 %` (ch05:23) — **ĐÚNG, tái lập được**

24 × 4.432 / 227.214 = **46.81 %**, cả hai số nằm trong `160245`. Bảng 1080p không in hàng phần
trăm nên con số này trước đây không tra được; nay tra được. Không sửa gì.

### 3.3 Độ phân tán DVFS: ch05 `20.3 %` so với ch07 `10.6 %` — **KHÔNG mâu thuẫn, khác phạm vi**

Cả hai đều đúng cho tập chạy mà chúng khai báo; cơ số là **giá trị nhỏ nhất** ở cả bốn phép tính.

```
ch07  "trong cùng phiên 2026-10-03, chín lần chạy tổng giai đoạn"
      ghim 3: 309.08 / 310.37 / 310.60                      -> (310.60-309.08)/309.08 =  0.49 %
      DVFS 6: 397.78 / 399.10 / 421.52 / 422.82 / 437.47 / 440.01 -> 10.62 %
ch05  "11 lần chạy trong hai ngày" (thêm nhánh 2026-10-01)
      DVFS 7: thêm 365.88 -> (440.01-365.88)/365.88 = 20.26 % -> in 20.3 %  OK
      ghim  : in "309.08-311.78 ms - phân tán 0.49 %"          -> KHÔNG KHỚP
```

**Khiếm khuyết thứ hai, cùng câu:** 0.49 % là phân tán của **ba** lần ghim trong phiên 10-03
(309.08–310.60). Dải 309.08–311.78 là dải **hai ngày**, phân tán của nó là
(311.78−309.08)/309.08 = **0.87 %**. Câu hiện tại lấy dải của tập rộng và phần trăm của tập hẹp,
trong khi nhánh DVFS ngay bên cạnh lại lấy cả hai của tập rộng.

**Khiếm khuyết thứ ba, cùng câu:** ch05 viết 11 lần chạy ấy là "trên vòng lặp tổ hợp". Sai —
chúng là **tổng giai đoạn** (`decision-latency-fp16-*.json`), đúng như ch07 gọi. Vòng lặp tổ hợp
của P3 là `p3-composed-cadence-fp16-*` (N=1 mỗi nhánh, 3 phân đoạn). Nhầm nhãn này nguy hiểm vì
cả mục 5.4/5.5 được viết ra để tách hai cấu trúc đó.

Ba sửa văn bản đề xuất cho một câu ở ch05:690-693:
`trên vòng lặp tổ hợp` → `trên tổng giai đoạn`; `0.49 %` → `0.87 %`; và nói rõ tập 11 lần chạy
là tập nào (xem 3.5).

### 3.4 "Tạo tác nào sinh ra 365.88 / 310.00 / 397.78…" — **TAN, không có họ tạo tác nào bị thiếu**

Cùng họ `decision-latency-fp16-*.json`, nhưng **trường khác**:
`by_scenario.p50.ms_per_decision_p50_with_stream_decode`, không phải
`by_scenario.p50.ms_per_decision_p50`. Chênh lệch giữa hai trường là 16 khung giải mã CPU.

```
220925: 322.47 (không giải mã) -> 365.88 (có giải mã)   = số in ở Bảng 5.4 nhánh DVFS
223538: 272.39                 -> 310.00                = số in ở Bảng 5.4 nhánh ghim
172128: 382.03                 -> 421.52
```

Lần dò trước đọc trường không-giải-mã nên không khớp gì. Không có số liệu nào mất.

### 3.5 Tập 11 lần chạy là một tập được **chọn lọc** (số mới, nên nằm ở đây)

Toàn bộ lần chạy tổng giai đoạn có `_with_stream_decode`, sau mốc 2026-09-29 (mốc 09-29 bị loại
bởi `finding-2026-10-01-engine-provenance-invalidates-0929-latency.md`):

```
ghim  (6): 308.16 311.78 310.00 | 309.08 310.37 310.60    -> dải 308.16-311.78, phân tán 1.17 %
DVFS (11): 351.30 360.01 365.88 368.17 371.69 397.78 399.10 421.52 422.82 437.47 440.01
                                                          -> dải 351.30-440.01, phân tán 25.26 %
```

Tập 11 của ch05 = 4 ghim + 7 DVFS, tức nhánh 10-01 được thu về trung vị ở một số chỗ và tính
theo lần chạy lẻ ở chỗ khác (max ghim 311.78 là một lần chạy lẻ, min ghim 309.08 thì không).
**Hai kết luận của chương vẫn đứng, và đứng mạnh hơn** nếu dùng tập đầy đủ: hai tập vẫn
**không giao nhau** (max ghim 311.78 < min DVFS 351.30), và phân tán DVFS tăng 20.3 % → 25.3 %.
Không đề xuất đổi số trong chương (đóng băng); đề xuất chương nói rõ tập chạy của mình.

## 4. Bảng hợp nhất: số Jetson của mục tiêu nhị phân

| đại lượng | trị số | nguồn |
|---|---|---|
| nhịp đã triển khai, p50 | **167.291 ms = 5.978 qđ/giây** | `155843` [ĐO] |
| p95 | 198.224 ms = 5.045 qđ/giây | `155843` [ĐO] |
| trên toàn 201 quyết định | 5.772 qđ/giây | `measured_decisions_per_sec` [ĐO] |
| hệ số hoạt động | 63.1 % (chặn trên, đơn luồng) | `duty_cycle_compute` [ĐO] |
| trước khi sửa tiền xử lý | 289.672 ms = 3.452 qđ/giây, duty 76.1 % | `233159` [ĐO] |
| mức lợi của bản sửa | 1.732 lần (−42.2 %), tiền xử lý −68.4 % | [SỐ HỌC] |
| nguồn 1920×1080 | 227.214 ms = 4.401 qđ/giây, duty 37.7 % | `160245` [ĐO] |
| biên hợp đồng, kho 720p | 4.79 lần (1.249) / 2.39 lần (2.500) | [SỐ HỌC] |
| biên hợp đồng, 1080p | 3.52 lần / 1.76 lần | [SỐ HỌC] |
| bảng giai đoạn liên tiếp | 23.796 ms = 42.02 qđ/giây — **không phải một nhịp** | `r48-pipeline-*` [ĐO] |
| độ chính xác đi kèm | **84.2 %** (không bao giờ 84.4 %) | Chương 6 [ĐO] |

Ba điều kiện bắt buộc kèm bảng này: (i) hợp đồng f/24 suy từ phân bố khung/giây của kho, **không**
từ OSHA / ISO 45001 / IEC 61508 hay tiêu chuẩn Việt Nam nào — "vượt 4.79 lần" chỉ nói B1 theo kịp
tốc độ khung nguồn; (ii) **không** số độ chính xác nào trong luận văn được đọc là độ chính xác của
vòng lặp trên thiết bị (đánh giá lấy T=16 đều trên cả clip, vòng lặp lấy 16 khung trong cửa sổ 48
bước 24) — đại lượng đó chưa đo, mục 5.9 ý 15; (iii) 167.291 và 227.214 **không** trộn, không lấy
trung bình: khác độ phân giải *và* khác kho.

## 5. Thay đổi đề xuất (chỉ sửa văn bản, 1 câu + 1 token)

| tệp | dòng | hiện tại | đề xuất |
|---|---|---|---|
| `report/vn/chuong-05-trien-khai-bien.txt` | 22 | `53.2 %` | `52.6 %` |
| `report/vn/chuong-05-trien-khai-bien.txt` | 690-693 | `trên vòng lặp tổ hợp` | `trên tổng giai đoạn` |
| `report/vn/chuong-05-trien-khai-bien.txt` | 692 | `phân tán 0.49 %` (dải hai ngày) | `phân tán 0.87 %` |
| `report/vn/chuong-05-trien-khai-bien-ghim-xung-nhip.txt` | — | kiểm cùng ba chỗ (bản chỉ lưu hồ sơ) | như trên |

Chưa thực hiện — chờ tác giả đồng ý, như đã làm với Ch1/Ch2.

## Câu hỏi chưa giải quyết

1. Có sửa ba chỗ ở mục 5 không? Cả ba là sửa văn bản trong phạm vi đóng băng, không số mới.
2. Chương 05 có nên khai báo một lần "mọi trị số mục 5.4 lấy từ lần chạy trung vị theo nhánh"
   để các chỗ trích `155806` (58.052 / 2.48 lần / 2.182) không còn trông như trộn lần chạy?
3. Có nên thay tập 11 lần chạy bằng tập đầy đủ 6 ghim + 11 DVFS (phân tán 1.17 % / 25.3 %,
   vẫn không giao nhau)? Mạnh hơn cho luận điểm, nhưng là số mới ⇒ vi phạm đóng băng.
4. ch07:149 TSF "+0.2 pp": BA của riêng luồng thực thể không in ở đâu trong `report/vn/` — còn treo.
5. Làn Unsafe-Net 687 clip 2 lớp (69.6 %) vẫn chưa có đường sàn ở bất kỳ chỗ nào trong `report/vn/`.
