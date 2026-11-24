# So sánh định lượng đa trục: luận văn so với công trình liên quan

Loại: tổng hợp (synthesis) · Ngày: 2026-11-17 · Nguồn: 7 báo cáo `../research/researcher-01..07`
Work context: `/Users/nguyenthiphuong/Learning/Proposal/project/my-project`

---

## 0. Phạm vi và các điều cấm phải tôn trọng

Yêu cầu: so sánh kết quả đo được với phương pháp tương tự, **định lượng hao hụt độ chính xác** và
**hiệu năng**, trên nhiều trục metrics.

Điều cấm còn hiệu lực, áp dụng cho toàn báo cáo này:

1. **Không trộn làn.** Làn A = iSafetyBench nhị phân, tháp CLIP ViT-B/32 đóng băng toàn khung +
   đầu B1, LSO-1100 (§6.2 — cấu hình triển khai). Làn B = **cùng** clip/fold, biểu diễn token
   quan hệ thực thể, encoder E0–E3 (§6.6, §6.11). Làn C = Unsafe-Net, 691 clip, một nhà máy,
   LCO-691 (§6.5 nhị phân, §6.12 bốn lớp). Không trung bình giữa làn, không chứng thực chéo,
   không hai số khác làn trong một câu so sánh. *Tái lập độc lập cùng một phép đo trên hai kho
   rời nhau thì được phép* (§6.1.1) — đó là chuyện khác.
2. **Không so với bài báo gốc trên iSafetyBench** (§2.3). Không trích mAP trên kho đó.
3. **84.2% không bao giờ rời đường sàn 63.3%** của làn A. Và với mọi số trên thiết bị: trích số
   của chính B1, **không bao giờ** `max(B1, B1w)` (§5.4.1).
4. **Số đã rút không được dùng lại**: "+3.5 pp macro", "57/64", "95.7% mAP", cadence P3
   "331 ms / 3.02 dec/s / 3.91 J", "417.84 MB", "73.051 quyết định/giây", "≈520 B/sự kiện",
   "≈107,000 khung", "±0.9 °C", mọi **tổng theo giai đoạn** trình bày như nhịp hệ thống.
5. **Không một con số công suất nào được dùng để lập luận ngân sách/chế độ nguồn** (§5.6.5).
6. **Không đặt số luận văn vào cùng bảng với WS-VAD** (khuyến nghị researcher 07; lý do ở §12).

## 1. Thang nhãn so sánh được

Mọi hàng trong mọi bảng đều mang một nhãn. Với các trục **hao hụt**, nhãn áp cho **Δ**, không áp
cho con số tuyệt đối.

| Nhãn | Nghĩa |
|---|---|
| (a) | So sánh trực tiếp — cùng chỉ số, cùng kho/giao thức tương đương |
| (a\*) | So sánh được **sau chuẩn hoá** (MAC↔FLOP, mỗi-khung↔mỗi-quyết định) |
| (b) | Cùng phần cứng, khác mô hình |
| (c) | Cùng mô hình, khác phần cứng |
| (d) | Khác đơn vị / khác không gian nhãn → **không quy đổi được**; chỉ là dải tham chiếu |

---

## 2. TRỤC 1 — Hao hụt độ chính xác do đóng băng xương sống

**Điều phải nói trước: luận văn KHÔNG đo hao hụt đóng băng của chính nó.** Không có lượt
tinh-chỉnh-toàn-phần CLIP ViT-B/32 nào được chạy trên làn A. Vậy con số này **không tồn tại trong
kho** và không được bịa ra. Những gì *có* đo là một so sánh **chọn xương sống**, không phải chi
phí đóng băng:

| So sánh | Tham số huấn luyện | BA (LSO-1100) | Δ | CI 95% | Kiểm định | Nhãn |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 đóng băng + đầu (B1w) | 333,570 | 84.4% | — | — | — | (a) |
| VideoMAE-base-K400 đóng băng + đầu (B2) | 233,858 | 81.5% | **+2.9 pp** cho CLIP | [−1.8, +7.0] | McNemar p = 0.0178 | (a) |
| B1 triển khai | 201,090 | **84.2%** (sàn 63.3%) | — | — | — | (a) |

Chiều của Δ này **khớp với tài liệu**: VideoGLUE (2307.03166) Hình 1, probe xương sống ảnh đóng
băng so với xương sống video đóng băng — K400 CLIP 75.2 vs VideoMAE 65.1 (**ảnh +10.1**); LoRA
K400 80.2 vs 73.6 (+6.6). Nhưng đổi chiều trên tác vụ nặng chuyển động: SSv2 41.0 vs 53.9
(**video +12.9**), Diving48 44.1 vs 59.5 (video +15.4). (d) — khác không gian nhãn.

### 2.1. Dải tham chiếu cho chi phí đóng băng (toàn bộ là nhãn (d))

Δ = (tinh chỉnh toàn phần) − (đóng băng). Dương = đóng băng lỗ.

| Nguồn | Kho | Cấu hình đóng băng | Tham số | Đóng băng | Full FT | Δ |
|---|---|---|---|---|---|---|
| ST-Adapter 2206.13559 T1 | K400 | probe tuyến tính | 0.31 M | 76.6 | 81.7 | **+5.1** |
| ST-Adapter T1 | K400 | ST-Adapter | 7.20 M | 82.0 | 81.7 | **−0.3** |
| VideoGLUE 2307.03166 | K400 | CLIP probe | — | 75.2 | 81.0 | +5.8 |
| VideoGLUE | SSv2 | CLIP probe | — | 41.0 | 46.6 | +5.6 |
| VideoGLUE | Diving48 | CLIP probe | — | 44.1 | 75.7 | **+31.6** |
| VideoGLUE | Charades (mAP) | CLIP probe | — | 11.2 | 54.3 | **+43.1** |
| VideoGLUE | K400 | VideoMAE probe | — | 65.1 | 78.7 | +13.6 |
| VideoMAE 2203.12602 T3 | SSv2 | probe (800 ep) | — | 38.9 | 69.6 | **+30.7** |
| STA-VPT 2312.10376 T VIII | HMDB51 | chỉ FT đầu | 0.07 M | 55.49 | 70.72 | **+15.2** |
| STA-VPT T VIII | UCF101 | chỉ FT đầu | 0.07 M | 86.41 | 94.50 | +8.1 |
| STA-VPT T VIII | K100 | chỉ FT đầu | 0.07 M | 77.15 | 83.05 | +5.9 |
| STA-VPT T VIII | SSv2 | chỉ FT đầu | 0.07 M | 20.80 | 37.75 | **+17.0** |
| STA-VPT T VIII | HMDB51 | STA-VPT-Parallel | 5.35 M | 73.20 | 70.72 | **−2.5** |
| STA-VPT T VIII | SSv2 | STA-VPT-Parallel | 5.35 M | 49.70 | 37.75 | **−12.0** |
| AIM 2302.03024 | K400 | adapter trên CLIP | 11/97 M | 83.9 | 82.0 (TimeSformer 121 M) | **−1.9** |
| X-CLIP 2208.02816 T7 | K400 | frozen/frozen | — | 79.1 | 82.4 | +3.3 |
| CLIP LoRA (VideoGLUE T6) | SSv2 | LoRA | — | 56.0 | 46.6 | **−9.4** |
| AIM §5 | K400 @ **1% dữ liệu** | adapter | 11 M | — | — | **−8.9** |
| ViFi-CLIP 2212.03640 T3 | UCF101 @ **K=2** | prompt A5 | — | 71.4 | 80.7 | **+9.3** |
| ViFi-CLIP T3 | HMDB51 @ **K=2** | prompt A5 | — | 39.7 | 57.2 | **+17.5** |

**Đọc dải này thế nào.** Ba chế độ tách biệt, không được gộp:

- **Đóng băng + module thời gian** (đúng lớp thiết kế của B1): Δ ∈ **[−12.0, +3.3]** — nghĩa là
  thường *không lỗ*, có khi lãi. Ở chế độ dữ liệu ít, AIM **lãi 8.9 pp** so với full FT ở 1% K400.
  Đây là chỗ B1 đứng, và dải này **biện hộ được** cho thiết kế đóng băng.
- **Chỉ probe/đầu tuyến tính, không module thời gian**: Δ ∈ **[+5.1, +43.1]**. B1 **không** thuộc
  chế độ này (nó có 1 lớp Transformer thời gian + vị trí học được), nên không được mượn dải này
  để nói B1 lỗ nhiều.
- **Phản ví dụ phải công bố, không được ẩn**: ViFi-CLIP ở K=2 cho full FT **thắng** đóng
  băng-prompt +9.3 và +17.5 pp. Tức "ít dữ liệu ⇒ đóng băng thắng" **không phải quy luật**; nó
  phụ thuộc đóng băng *cái gì*. Prompt-only ≠ đầu thời gian học được.

**Kết luận trục 1.** Hao hụt đóng băng của hệ thống này là **chưa đo**. Dải tham chiếu cho đúng
lớp thiết kế (đóng băng + đầu thời gian) là −12.0 đến +3.3 pp, và chiều ưu CLIP-trên-VideoMAE mà
luận văn đo được (+2.9 pp) trùng chiều VideoGLUE trên K400. Không hàng nào ở §2.1 được trích như
hao hụt *của luận văn*.

---

## 3. TRỤC 2 — Hao hụt do lượng tử hoá FP32 → FP16 → INT8

### 3.1. Những gì luận văn thực sự đo (làn A / đường B1 triển khai)

| Phép đo | Giá trị | Nhãn |
|---|---|---|
| Tỷ lệ đồng thuận quyết định FP16 vs FP32, 24 clip thật | **1.000** | (a) |
| Sai khác xác suất tuyệt đối lớn nhất | **2.66e-03** | (a) |
| `preprocessing_mismatches` | rỗng | (a) |
| Số lần chạy lại ở mức tải khác | 2, giống tới từng chữ số | (a) |

Nguồn: `r48-parity-fp16-20261001T{222019,224028}.json` (§5.3).

Phát biểu đúng mà luận văn **đã** dùng, và không cần sửa: "**FP16 không làm giảm độ chính xác một
cách phát hiện được, và dữ liệu không đủ để nói gì mạnh hơn thế**" (§5.6.5). Kiểm định z hai tỷ lệ
đã bị rút, thay bằng chặn **McNemar chính xác p = 1.000 ở mọi bảng 2×2 khả dĩ**. Đây là cách phát
biểu đúng chuẩn: tương đương **đo được** so với một ngưỡng, không phải "FP16 không mất gì".

**Hai hạn chế do chính luận văn khai, phải giữ nguyên trong mọi bảng so sánh:** (i) tương đương
FP16-vs-FP32 của **bộ phát hiện** chưa đo; (ii) engine **FP32 và INT8 của tháp CLIP chưa dựng lại
cho đúng thiết bị**, nên **không có số INT8 nào của đường triển khai được trích dẫn** (§5.3,
hạn chế (10)).

### 3.2. Hàng FP32/FP16/INT8 duy nhất có trong kho là số LỊCH SỬ của P1, làn C

| Nhánh | Độ trễ | Thông lượng | Đúng/64 (4 lớp) | Trạng thái đối chiếu |
|---|---|---|---|---|
| FP32 | 12.47 ms | 80.2 khung/s | 56/64 | **chỉ chép từ bản thảo EN**, chưa mở lại |
| FP16 | 8.62 ms | 116.0 khung/s | 55/64 | đã mở tạo tác, xác nhận |
| INT8 | 8.24 ms | 121.4 khung/s | 55/64 | **chỉ chép từ bản thảo EN**, chưa mở lại |
| ONNX trên CPU M3 | 26.8 ms p50 / 28.8 p95 | 37.3 FPS | — | 156 MB FP32 → 40 MB INT8 (3.8×) |

**Mọi con số mục 5.6 là số LỊCH SỬ và không con số nào được dùng để chống đỡ một kết luận**
(§5.6.2). Thêm nữa chúng là **làn C, 4 lớp, P1** — không phải đường B1 triển khai. Vậy:

> Chi phí INT8 "1 clip" (56/64 → 55/64) **không phải** một phép đo hao hụt lượng tử hoá của hệ
> thống triển khai. Nó là một hàng lịch sử của một pipeline khác, trên một làn khác, với hai
> trong ba hàng chưa được mở lại.

Nếu vẫn muốn đọc nó như một ước lượng độ lớn: 1/64 = **1.56 pp**, SE ≈ **4.13 pp** ở n=64, một
cặp bất đồng ⇒ **McNemar chính xác p = 1.0**. Tức *không phân biệt được với 0*.

### 3.3. Dải tham chiếu hao hụt PTQ trong tài liệu

| Nguồn | Cấu hình | Phạm vi lượng tử | Dải sụt top-1 (pp) | Nhãn |
|---|---|---|---|---|
| PTQ4ViT 2111.12293 T1 | W8A8, chỉ FC, calib 32 ảnh | một phần | **0.12 – 2.14** (DeiT-S xấu nhất 2.14; Swin-B 0.12) | (d) |
| PTQ4ViT T1 | W6A6 | một phần | 1.48 – **11.1** | (d) |
| FQ-ViT 2111.13824 T1 | 8/8/8 **toàn phần** (gồm LayerNorm, Softmax), calib 1000 ảnh | đầy đủ | **0.49 – 1.22** (ViT-B xấu nhất 1.22) | (d) |
| FQ-ViT T1 | 8/8/4 | đầy đủ | 0.73 – 1.85 | (d) |
| X3D-XS (PyTorchVideo zoo) | INT8, CPU di động Samsung S8 | — | **1.6** (68.5 → 66.9), nhanh 1.41× | (d) |
| arXiv 2509.21173 Phụ lục C.2 | W8A8, 20,160 lượt mô phỏng | — | **PTQ hỏng 44.1%** số lượt (sụt > 5%); QAT 25.0% | (d) |
| arXiv 2509.21173 Phụ lục E | Simple PTQ 62.0% hỏng → Rotation 5.6% → Rotation+LSQ **0.0%** | — | kỹ thuật quyết định | (d) |

**Đặt 1.56 pp vào dải:** nằm **trong** dải PTQ4ViT (0.12–2.14), **trên** dải FQ-ViT (0.49–1.22),
sát X3D-XS INT8 (1.6 pp). Nghĩa là *độ lớn không bất thường* — nhưng ở n=64 nó không phân biệt
được với 0, nên phát biểu duy nhất đúng là "**không phát hiện được hao hụt, và cỡ mẫu không cho
phép loại trừ hao hụt tới ±4 pp**".

### 3.4. Rủi ro FP16 mà tài liệu ghi nhận, và vì sao đồng thuận 1.000 là bằng chứng mạnh

TensorRT issue #3893: engine **FP16 cho mIoU 0.0 so với FP32 79.12385607181146** trên RTX 4090,
TRT 24.02; abs diff mean 0.015861 / max 0.15575; nguyên nhân nghi là "FP16 LayerNorm nodes that
may overflow"; chạy debug-precision trên 2,576 lớp vẫn không khoanh được; **issue còn mở, không
có bản sửa**. NVIDIA không công bố ngưỡng số nào ngoài trần FP16 65,504; **không tồn tại ngưỡng
cosine-similarity chính thức**; Polygraphy mặc định `rtol = atol = 1e-5`.

⇒ Bối cảnh này làm phép kiểm §5.3 **tăng giá trị**: 2.66e-03 lớn hơn 1e-5 nhiều bậc nên một phép
so sánh tensor theo mặc định Polygraphy sẽ **báo trượt**, trong khi **đồng thuận quyết định** —
đơn vị đúng cho một bộ phân lớp — là 1.000. Đây là lý do phải báo cáo đồng thuận quyết định thay
vì sai khác tensor, và luận văn đã làm đúng.

### 3.5. Kích thước engine sau lượng tử hoá

| Đối tượng | FP32 | INT8/FP16 | Tỷ lệ | Nhãn |
|---|---|---|---|---|
| ONNX hệ P1 (CPU M3, làn C) | 156 MB | 40 MB INT8 | **3.8×** | (a) nội bộ |
| Engine TRT P3 (trên thiết bị) | — | **24.46 MB** | — | (a) nội bộ |
| Engine TRT B1 (tổng tệp plan) | — | **197.4 MB** | dấu chân chạy < nửa | (a) nội bộ |
| arXiv 2509.18193 | — | — | giảm **35%** | (d) |
| arXiv 2603.28708 | — | — | giảm **49%** | (d) |
| Diễn đàn NVIDIA | — | — | **không bảo đảm**, "highly tactic dependent"; có ca FP16 = FP32 | (d) |

3.8× của ONNX INT8 **vượt** dải 35–49% của engine TRT — hợp lý, vì đó là hai định dạng khác nhau
(ONNX weight-only so với engine đã chọn tactic). **Không được trộn hai tỷ lệ này.**

---

## 4. TRỤC 3 — Lạm phát giao thức chia dữ liệu (đóng góp định lượng chính)

Ba phép đo **độc lập** của cùng một hiện tượng. Không trung bình, không chứng thực chéo.

### 4.1. Làn A — iSafetyBench, CLIP + đầu B1

| Giao thức | BA | Nhãn |
|---|---|---|
| Leave-source-out, 1,100 clip / 151 nguồn (**công bố**) | **84.2%** (sàn tiền đăng ký 63.3%) | (a) |
| k-fold ngẫu nhiên (mở rộng hậu nghiệm, không đăng ký trước) | 97.7% | (a) |
| **Δ** | **+13.5 pp** | (a) |

Ghi chú bắt buộc: lượt chia ngẫu nhiên là **mở rộng hậu nghiệm** của cổng G12 (đăng ký trên làn
B), **không** phải cổng phán quyết, **không** có trong Bảng 6.8 như hàng phán quyết. Luận văn
**thấy** 97.7% của chính mình và **chọn công bố 84.2%**.

### 4.2. Làn B — cùng clip, cùng fold, biểu diễn token quan hệ (Bảng 6.4)

| Encoder | Tham số | LSO | Ngẫu nhiên | Δ | CI 95% (bootstrap cụm 151 nguồn, B=5000) | McNemar chính xác |
|---|---|---|---|---|---|---|
| E3 DeepSets | 17,270 | 75.2% | 90.6% | **+15.4 pp** | [+10.6, +21.5] loại trừ 0 | b=186 c=8, p = 3.58e-45 |
| E1 d=64 | 93,222 | 78.1% | 91.3% | +13.2 pp | — | — |
| E1 d=96 | 195,558 | 77.0% | 91.9% | +14.9 pp | — | — |
| E0 RelationTransformer | 298,118 | 80.3% | 90.7% | +10.4 pp | [+6.3, +15.3] loại trừ 0 | b=150 c=22, p = 1.3e-24 |
| **Độ phân tán của thang (17×)** | | **5.1 pp** | **1.4 pp** | | | |

Luận văn còn in khoảng tin cậy **mức clip** [+13.1, +17.6] kèm nhãn "WRONG UNIT" và **không dùng**
— để người đọc thấy đúng cái giá của việc chọn sai đơn vị lấy mẫu. Đây là thực hành mà **không
công trình nào** trong hai lĩnh vực lân cận làm (§10).

### 4.3. Làn C — Unsafe-Net, LCO-691 vs k-fold ngẫu nhiên (§6.5)

Năm kiến trúc, 18,044 → 298,892 tham số (chênh hơn 16×), **không kiến trúc nào miễn nhiễm**:
Δ ∈ **+13.0 … +21.8 pp** (ví dụ E1 d=64, 94,380 tham số: 84.1% → 62.3% = +21.8 pp).

### 4.4. Hai kết quả cơ chế mạnh hơn bản thân con số Δ

1. **Rò rỉ xoá khả năng phân biệt kiến trúc.** Dưới chia ngẫu nhiên, 17× khoảng tham số chỉ trải
   **1.4 pp**; dưới LSO trải **5.1 pp**. Tức giao thức rò rỉ làm mọi kiến trúc trông như nhau —
   một bảng so sánh kiến trúc chạy trên chia ngẫu nhiên **không đo được gì**.
2. **Phần mất đi không phải thông tin đếm tầm thường.** Đường sàn quy tắc đếm của làn B gần như
   không đổi giữa hai giao thức (68.2% → 67.2%, **giảm 1.0 pp**) trong khi encoder **mất 15.4
   pp**. Vậy phần bị mất là **thông tin đặc thù theo nguồn**, chỉ khai thác được khi nguồn xuất
   hiện ở cả hai phía. Sàn làn B (67.2/68.2%) là **đối tượng khác** với sàn tiền đăng ký 63.3%
   của làn A và **không được trừ chéo** (§6.1.4, §6.6.1).

### 4.5. Dải tham chiếu rò rỉ theo nhóm trong tài liệu

| Nguồn | Lĩnh vực | Đơn vị nhóm | Độ lạm phát | Nhãn |
|---|---|---|---|---|
| UniqueShip 2609.13659 T II | âm dưới nước, DeepShip | bản ghi nguồn | **+10.2 … +12.2 pp** acc (F1 +12.7…+13.7) | (d) — gần nhất về *loại* rò rỉ |
| UniqueShip T II | âm dưới nước, VTUAD | bản ghi nguồn | **+47.2 … +47.6 pp** acc (F1 +60.9…+61.4) | (d) |
| EEG (tổng hợp) | EEG | đối tượng | +21.5 … +31 pp | (d) |
| Zhu 2015 | — | đối tượng | 90.97% → 64.82% (**−26.15**) | (d) |
| Ploton 2020 (DOI 10.1038/s41467-020-18321-y) | sinh khối viễn thám | không gian | R² 0.53 → **0.14**; LOO có vùng đệm ~0.50 → ~0.15 (50 km) → ~0 (>100 km) | (d) |
| Tampu 2022 | OCT | bệnh nhân | MCC +0.07 … +0.43 | (d) |
| Peptide MS | protein | — | **+1.8 pp (chiều ngược)** | (d) |

**Vị trí của luận văn trong dải:** 13.0–21.8 pp (làn C) và 10.4–15.4 pp (làn B) nằm **giữa-dưới**
dải đã công bố — cùng băng với DeepShip (10–12 pp), **thấp hơn** EEG (21–31) và viễn thám
(39–47 pp quy theo R²). Tức **độ lớn không bất thường**.

**Nhưng:** researcher 05 và 06 cùng xác nhận **không tồn tại con số công bố nào** cho nhận dạng
hành động trên video dưới chia theo nguồn, và **không công trình giám sát an toàn công nghiệp nào
dùng chia có ràng buộc nhóm**. Vậy đóng góp của luận văn ở trục này là: *độ lớn tầm thường, miền
ứng dụng có thể là mới*. Yêu cầu trích dẫn mạnh nhất hiện có là **CLAIM 2024 Item 20**: "Sets of
medical images generally should be disjoint at the patient level or higher … so that images of the
same patient do not appear in more than one partition."

Cảnh báo Cameron & Miller (20–50 cụm) áp trực tiếp vào **làn C với 8 cụm bối cảnh** — đơn vị
bootstrap ở đó dưới ngưỡng an toàn và phải nói rõ.

---

## 5. TRỤC 4 — Hiệu quả tham số huấn luyện

| Cấu hình | Tham số huấn luyện | % so với full FT (85.87 M) | Nhãn |
|---|---|---|---|
| Chỉ FT đầu (STA-VPT T VIII) | 0.07 M | 0.08% | (d) |
| **B1 triển khai (luận văn)** | **0.201 M** | **0.23%** | (a\*) |
| Probe tuyến tính (ST-Adapter) | 0.31 M | 0.36% | (d) |
| B1w / B1xl / B2 / ESTA / TSF | 0.334 / 0.697 / 0.234 / 0.352 / 0.687 M | 0.27–0.81% | (a) nội bộ |
| VPT-deep | 4.68 M | 5.4% | (d) |
| STA-VPT-Parallel | 5.35 M | 6.2% | (d) |
| Adapter (ST-Adapter T1) | 6.77 M | 7.9% | (d) |
| ST-Adapter / EVL-1 lớp | 7.20 / 7.41 M | 8.4–8.6% | (d) |
| AIM | 11 M | 12.8% | (d) |
| EVL-4 lớp | 28.70 M | 33.4% | (d) |
| Vita-CLIP | 38.88 M | 45.3% | (d) |
| Full FT (TimeSformer / SA+TA) | 85.87 – 121.57 M | 100–142% | (d) |

**Đọc:** B1 ở **0.23%** tham số huấn luyện nằm giữa "chỉ đầu" và "probe tuyến tính" — tức thuộc
**nhóm rẻ nhất** trong thang PEFT đã công bố, rẻ hơn VPT-deep ~23×, rẻ hơn AIM ~55×, rẻ hơn full
FT ~427×. Và dải Δ ở §2.1 cho đúng lớp này là −12.0…+3.3 pp.

Hai quan sát nội bộ củng cố (nhãn (a)): (i) tăng dung lượng khối thời gian **mua được rất ít** —
đầu triển khai 201,090 **không** bị 298,118 đánh bại, B1xl (696,962) đạt 82.7%, **thấp hơn** B1w
1.7 pp; (ii) B1w hơn B1 **+0.2 pp ở 1.66× tham số đầu**, hai giá trị không phân biệt được về
thống kê. ⇒ Trần của lớp bài toán này **không nằm ở dung lượng đầu thời gian**; phần thắng là
**mua từ xương sống**, không phải từ thiết kế đầu.

**Đối chiếu WS-VAD: không khả thi.** Không một bài WS-VAD nào (Sultani → LAS-VAD) báo cáo tổng
tham số huấn luyện ⇒ 201,090 **không có đối ứng**. Thêm confound: WS-VAD dùng ViT-B/16, luận văn
dùng B/32 — và X-CLIP T1 cho biết riêng chênh backbone này đáng **3.6 pp** ở 16 khung full FT
(81.1 vs 84.7).

---

## 6. TRỤC 5 — Tính toán lý thuyết (GMACs / GFLOPs)

Đây là **trục duy nhất** cho so sánh gần-trực-tiếp về hiệu năng, sau chuẩn hoá MAC.

| Mô hình | Tham số | Tính toán | Top-1 K400 | Nhãn |
|---|---|---|---|---|
| **CLIP ViT-B/32 (tháp luận văn)** | **87,849,216** (≡ 87.85 M trong `model_profile.csv`) | 4.41 GMACs/ảnh → **16 khung ≈ 70.6 GMACs ≈ 141 GFLOPs/quyết định** `[suy ra]` | — | **(a)** cho tham số; **(a\*)** cho tính toán |
| X-CLIP B/32, 16×224 | — | **75 G** | 81.1 | (a\*) — **khớp gần như hoàn toàn** |
| X3D (XS→L) | 3.8 – 6.2 M | 0.60 – 18.37 G | — | (a\*) |
| SlowFast | 34.5 / 34.6 / 62.9 M | — | — | (a) tham số |
| VideoMAE B/L | 87 / 305 M | 180 / 597 G × 15 views | — | (a\*) |
| TimeSformer | 121.4 M | **0.59 T** | — | (a\*) |
| EVL ViT-B/16 (CLIP đóng băng) | +7.41 M (1 lớp) / +28.70 M (4 lớp) | **444 G (T1) vs 454 G (T3) — mâu thuẫn trong chính bài** | — | (a\*) có bảo lưu |
| ConvLSTM gốc 1506.04214 | **không công bố** | **không công bố** | — | **(d)** |

**Hai điều phải ghi:** (i) `image_gflops` = 8.82 trong `model_profile.csv` của open_clip là **2 ×
GMACs** — quy ước FLOP = 2 MAC; luận văn **phải nói rõ dùng quy ước nào** trước khi in bất kỳ số
GFLOPs. (ii) Bài ConvLSTM gốc là bài **dự báo mưa**, không có tham số/FLOPs/K400 ⇒ mọi so sánh
tính toán với "ConvLSTM" là **(d), không quy đổi được**.

---

## 7. TRỤC 6 — Độ trễ, thông lượng, nhịp quyết định

### 7.1. Số của luận văn (Jetson Orin Nano Super 8 GB, R36.5.0, TRT 10.3.0 FP16, MAXN_SUPER)

| Phép đo | Giá trị (3 lần lặp) | Nhãn |
|---|---|---|
| Tháp CLIP ViT-B/32, **liên tiếp** (back-to-back) | 23.386 / 23.335 / 23.567 ms | (a) nội bộ |
| Mỗi quyết định (tính toán) | 23.796 / 23.731 / 23.958 ms | (a) |
| **Quyết định/giây** | **42.02 / 42.14 / 41.74** | (a) |
| Tháp **trong vòng lặp** so với liên tiếp | 58.052 vs 23.386 ms = **2.48×** | (a) |
| Tổng theo giai đoạn so với đo hợp thành | 76.164 vs 167.291 ms = **2.20×** | (a) |
| Hợp đồng nhịp (trung vị kho 29.97 fps / bước 24) | 1.249 quyết định/s | (a) |
| Biên so với 1.249 / 2.500 | 4.79× / 2.39× (trước sửa) → **3.52× / 1.76×** (sau sửa) | (a) |
| Một phép đo *danh nghĩa giống nhau*, khác chuỗi công cụ/nguồn | cao hơn **13.1%** so với 23.386 ms | (a) |
| P3 ghim xung nhịp, tải crop trung vị | 310 ms = **3.23 dec/s** (+7.5% biên) | (a) |
| P3 dưới DVFS mặc định | 366 ms = **2.73 dec/s** → **TRƯỢT** | (a) |
| P3 ở p95 | TRƯỢT cả hai điều kiện (ghim 2.88 / 2.81 / 2.76) | (a) |

**Ba quy tắc bắt buộc đi kèm.** (i) **Tổng theo giai đoạn ≠ nhịp hợp thành**: hệ số 2.20–2.48×
giữa tổng và đo hợp thành là lý do mọi phán quyết nhịp dựa trên tổng giai đoạn bị cấm. (ii)
**Trạng thái xung nhịp phải nêu cùng mọi số cadence của P3** — cùng engine, cùng clip, hai
governor cho PASS và FAIL. (iii) Hợp đồng 1.249 dec/s **không có cơ sở an toàn lao động**; cách
đọc hợp lệ duy nhất là "B1 theo kịp nhịp khung nguồn, biên 4.79× ở clip trung vị và 2.39× ở
60 fps".

### 7.2. Vì sao không có đối chứng ngoài

researcher 03 (trục hiệu năng) và 07 (WS-VAD) kết luận trùng nhau:

- **Không tồn tại phép đo mô hình video nào đã công bố trên Orin Nano Super.** (c) là tốt nhất có
  thể đạt được, và không có hàng nào đủ điều kiện.
- **Không tồn tại con số "quyết định/giây" cho giám sát video cửa sổ trượt trên Jetson** ở bất kỳ
  đâu. Gần nhất là ApS trong RW-GCN (arXiv:2201.05739) — **Xavier NX, dữ liệu khung xương** ⇒ (d).
- Tham chiếu duy nhất **cùng phần cứng** là bộ phát hiện, không phải mô hình video:

| Hệ | Phần cứng | Độ trễ | Thông lượng | Nhãn |
|---|---|---|---|---|
| YOLOv8n TRT FP16 | **Orin Nano Super** | **4.43 ms** | **225.3 FPS** | (b) |
| YOLOv8n TRT INT8 | Orin Nano Super | 3.49 ms | — | (b) |
| YOLOv8n TRT FP32 | Orin Nano Super | 8.36 ms | — | (b) |
| YOLOv8n PyTorch FP32 | Orin Nano Super | 20.88 ms | — | (b) |
| Ultralytics YOLO26 FP16 | Orin Nano Super | 4.57 ms/ảnh | — | (b) |
| Seeed, pipeline **4 luồng 4K** | Orin Nano Super | — | **58.82 FPS tổng hợp**, 11.02 W | (b) — **không phải số một mô hình** |
| MemoVAD 2606.07669 T2 | **AGX Orin** | 0.475 s | 36.2 FPS | (c)/(d) — board khác, tác vụ khác |
| MemoVAD baseline CLIP | AGX Orin | 1.512 s | 11.5 FPS | (c)/(d) |
| MoViNet A0–A5 | V100 | 183 – 2310 ms luồng | 3.7 – 19.2 ms/khung | (d) |
| EVL ViT-B/16 | V100-32G | — | 25.53 views/s | (d) |

**Thông số board phải nêu kèm vì tài liệu NVIDIA tự mâu thuẫn:** 67 TOPS INT8 **sparse** / 33
dense; 7/15/25 W; $249; GPU 1020 vs 635 MHz; **băng thông bộ nhớ 102 vs 65/68 GB/s tuỳ trang**;
"Super" là **nâng cấp chỉ bằng phần mềm** (JetPack 6.1+, `nvpmodel -m 2`); trang sản phẩm 404;
**FP16 TFLOPS không được công bố**. Mọi so sánh "so với TOPS lý thuyết" vì vậy không dùng được.

### 7.3. Số đa luồng P1 — giữ nhưng hạ cấp

4 luồng **88.06 FPS tổng hợp, 0 khung bỏ** và soak **36.2 phút, 0 sự kiện giảm tốc** là của **P1**
và **không chuyển sang B1 hoặc P3** — cả hai pipeline mới **chưa từng** được đo đa luồng. Thêm
nữa `multi-stream-jetson-fp16-realtime.json` **không mang khối xuất xứ nào** (không `machine`,
`platform`, `tensorrt_version`, `timestamp`, trạng thái xung nhịp) và **không được git theo dõi**
⇒ **không thoả (Đ2)**, giữ với tư cách số lịch sử. Mọi phát biểu "hệ thống chạy 4 camera" phải
gắn với P1 + tác vụ 4 lớp, hoặc bị bỏ.

---

## 8. TRỤC 7 — Năng lượng

| Phép đo | Giá trị | Nhãn |
|---|---|---|
| Công suất rỗi đã ổn định (118 mẫu) | **4.691 W** | (a) |
| B1 tháp CLIP ViT-B/32, lô 16 | 41.79 fps; 20.105 W | (a) |
| **Mỗi quyết định — toàn board** | **0.48110 J** | (a) |
| **Mỗi quyết định — biên (trừ nền)** | **0.36885 J** | (a) |
| → quy về mỗi khung (÷16) | **30.07 mJ** toàn board / **23.05 mJ** biên `[suy ra]` | (a\*) |
| Độ nhạy mốc nền (4.842 → 4.691 W) | 0.36522 → 0.36885 J = **1.0%** | (a) |
| P3 | 4.02 J toàn board / 2.90 J biên | (a) |
| P1 MAXN_SUPER | 152.6 fps; **34.18 mJ/suy luận**; tỷ số 34.2 vs 45.7 mJ = **+33.6% suy luận/J** | (a) lịch sử |
| P1 soak | 7,033.761 mW TB; 55.88 °C (min 52.375 / max 56.812) | (a) lịch sử |

### Đối chứng ngoài duy nhất khả dụng — CoDAT arXiv:2608.06691 Bảng V (AGX Orin)

| Mô hình | mJ/khung | Nhãn |
|---|---|---|
| CoDAT-S | 13.79 | (c)/(d) |
| **Luận văn B1, biên** | **23.05** `[suy ra]` | — |
| **Luận văn B1, toàn board** | **30.07** `[suy ra]` | — |
| MoViNet-A1 | 86.84 | (c)/(d) |
| MoViNet-A3 | 155.37 | (c)/(d) |
| TimeSformer | 591.90 | (c)/(d) |

**Điều kiện so sánh phải nêu:** CoDAT đo bằng jtop/INA3221, **công suất tổng, không trừ nền** ⇒
chỉ hàng **toàn board 30.07 mJ** của luận văn là cùng quy ước; hàng **biên 23.05 mJ** thì không.
Board khác (AGX Orin ≠ Orin Nano Super) ⇒ nhãn (c)/(d), **không phải (a)**. Đọc được: B1 nằm
**giữa CoDAT-S và MoViNet-A1**, thấp hơn TimeSformer ~20×.

**Quy tắc §5.6.5 áp vào đây:** không một con số công suất nào ở trên được dùng để lập luận ngân
sách hay khuyến nghị chế độ nguồn. Tỷ số +33.6% suy luận/J **giữ được** vì là tỷ số giữa hai phép
đo; nhãn "ngưỡng 15 W" và khuyến nghị "chế độ 2" **phải bỏ**.

**Trên lĩnh vực lân cận: không công trình giám sát an toàn công nghiệp nào đo năng lượng**
(researcher 06). Đây là trục luận văn **không có đối thủ**.

---

## 9. TRỤC 8 — Bộ nhớ và kích thước triển khai

Xem §3.5. Thêm: con số **417.84 MB đã bị rút** (tổng 11 engine trộn P1 + P3 từ thư mục mặc định,
không phải số của P1). Dấu chân chạy của B1 **dưới nửa** tổng tệp plan 197.4 MB.

Tài liệu không cho đối chứng dùng được: không bài video SOTA nào báo cáo dung lượng engine triển
khai ⇒ (d).

---

## 10. TRỤC 9 — Độ chặt thống kê: nơi luận văn dẫn trước rõ rệt

| Thực hành | Luận văn | WS-VAD (12 bài, 2018–2026) | An toàn công nghiệp (10 hệ) |
|---|---|---|---|
| Khoảng tin cậy | **có** — bootstrap cụm B=5000, seed 0, percentile, có cặp | **không bài nào** | **không bài nào** |
| Đơn vị lấy mẫu = nhóm, không phải clip | **có** (151 nguồn / 8 cụm); in cả CI mức clip kèm nhãn "WRONG UNIT" | không | không |
| Nhiều hạt giống / phương sai | **có** (3 hạt giống, ensemble) | **không bài nào** — mọi số là giá trị đơn | không |
| Hiệu chỉnh đa so sánh | **có** — Benjamini–Hochberg q = 0.05 | không | không |
| Kiểm định có cặp | **có** — McNemar chính xác, kèm b/c | không | không |
| Nadeau–Bengio cho fold lặp | **có** (làn C) | không | không |
| Cổng đăng ký trước | **có** — 8 cổng phán quyết: **2 đạt** (G10, G12), **6 trượt** (G11, G13–G16, G18) | không | không |
| Đối chứng giả dược | **có** (G19) | không | không |
| Chia có ràng buộc nhóm | **có** (LSO/LCO) | **không bài nào** | **không bài nào** |
| Đo trên thiết bị biên + năng lượng | **có** | chỉ MemoVAD (không năng lượng) | chỉ FPS lẻ |
| Công khai mã + dữ liệu | **có** | một phần | một phần |

**Không công trình nào trong cả hai lĩnh vực có đủ cả bốn**: biên + năng lượng, CI, chia theo
nhóm, mã + dữ liệu công khai. Và **6/8 cổng trượt được công bố là trượt** — đây là bằng chứng
đăng ký trước có hiệu lực thật, không phải trang trí.

DEFF = 1 + (m−1)ρ; CI nới **2.9×** khi chuyển từ đơn vị clip sang đơn vị nguồn ⇒ DEFF ≈ **8.4**.
Đây là con số định lượng trực tiếp cho câu "chọn sai đơn vị lấy mẫu làm hẹp CI bao nhiêu".

---

## 11. Bảng hệ thống giám sát an toàn công nghiệp

Nhãn mặc định **(d)** cho mọi hàng: khác kho, khác không gian nhãn, khác tác vụ.

| Hệ | Dữ liệu | Chỉ số | Phần cứng | Nhịp | Chia theo nhóm? | CI? |
|---|---|---|---|---|---|---|
| Önal (thesis + DOI 10.1007/s11042-024-19276-8) | 691 clip, 8 lớp, **2 camera IP, 1 nhà máy**; **2,262 khung bbox train / 329 test** | "Overall" **95.81%** = **trung bình macro không trọng số** của (100; 90.91; 92.31; 100) = 95.805; gộp **62/64 = 96.88%** top-1; ConvLSTM-only macro **85.78%**, gộp 57/64 = 89.06%; YOLOv4 test mAP@0.5 **95.66%**, val 91.18% | 2× GTX 1080Ti + i9-9900, 16 GB | **0.14 s** "eylem tanıma", **phần cứng không nêu**; **1.87 s là độ dài clip trung bình sau lọc, KHÔNG phải độ trễ**; **FPS hệ chưa từng báo cáo** | **Không** — 566/125 cố định theo clip, không ràng buộc camera/phiên/ngày/công nhân | Không (Wilson 95% cho 62/64 tự tính: **[89.3; 99.1]**) |
| Nath 2020 (Pictor-v3) | 1,472 ảnh | mAP 72.3% | — | 11–13 FPS | Không | Không |
| Sec-YOLO (PeerJ CS 2025) | 8,799 ảnh, random 7:2:1 | mAP50 **92.6** / mAP50-95 63.6; **mâu thuẫn nội bộ P 92.2 vs 91.3** | RTX 3090 | **không báo cáo** | Không | Không |
| Mohy (ITcon 2025) | 55,594 ảnh riêng | P 0.89 / mAP95 0.45 | — | — | Không | Không |
| MTA 2026 | — | — | Orin Nano | 30 FPS `[snippet]` | Không | Không |
| JCEMD | — | mAP50 92.52 `[snippet]` | Xavier NX | 9.11 FPS `[snippet]` | Không | Không |
| MonitorVLM 2510.03666 | 9,000 VQA, 10 mỏ, 80/20 | -72B đủ **93.05 / 89.57 / 91.28**; -basic 89.47 / 82.02 / 85.57; Qwen2.5-VL-72B gốc 73.33 / 61.11 / 66.66 | **8× H100**, không biên | K=1 17.46 s → K=5 17.59 s → w/o CF 20.35 s | Không | Không |
| iSafetyBench 2508.00399 | 1,100 clip, 680 thường / 420 hiểm, 18 nhóm | **MCQ 16 lựa chọn, chance 6.25%**, toàn bộ zero-shot; Avg Acc/F1: Ovis2-8B 43.8/49.8 · Valley-Eagle-7B 42.35/44.8 · InternVL2.5-8B-MPO 40.25/49.9 · Qwen2.5-VL-7B 40.25/45.45 · GPT-4o 38.8/48.35 · VideoLLaMA3-7B 35.8/38.1 · VideoChat-Flash-7B 28.9/32.3 · Oryx-7B 23.25/28.45 | — | — | **Không có chia train/test; không nhóm theo nguồn; số nguồn video KHÔNG công bố; không baseline phi-VLM** | Không |

**Lỗi nhãn tương đối phải biết khi đọc MonitorVLM:** "+22.01% / +34.22% / +28.37%" trong tóm tắt
là **tương đối**; tuyệt đối là **+16.14 / +20.91 / +18.91 pp**, và so với biến thể **-basic** của
chính nó, không so baseline Qwen.

**Không công trình nào khác huấn luyện/đánh giá trên kho Önal/Mendeley với số liệu** (giới hạn:
Scholar / ScienceDirect / Semantic Scholar đều trả 403 ⇒ kết luận bị hạn chế bởi phạm vi tìm).

---

## 12. Những gì KHÔNG so sánh được, và vì sao

1. **iSafetyBench của luận văn vs bài báo gốc.** Bài gốc là **MCQ 16 lựa chọn zero-shot** (chance
   6.25%), luận văn là **nhị phân có huấn luyện** dưới LSO. Khác tác vụ, khác chỉ số, khác chế độ.
   §2.3 đã cấm, và luận văn **không đặt bảng nào**. Không trích mAP trên kho này.
2. **Bất kỳ số nào của luận văn với bảng WS-VAD.** Bốn lý do: (i) AUC → BA **không quy đổi được**
   — chặn một chiều duy nhất là max BA ≥ 1 − ½√(2−2·AUC), với AUC 0.88 chỉ cho ≥ 0.755, quá lỏng;
   (ii) không bài nào báo tham số huấn luyện; (iii) confound backbone B/16 vs B/32 trị giá ~3.6 pp;
   (iv) **không giao thức nào tách camera/bối cảnh**. Thêm hai phê phán giao thức: cặp trong-video
   chỉ chiếm **0.071–0.388%** tổng số cặp bất thường–bình thường, và một bộ phân lớp **mức video**
   đạt 81.40–97.18% Micro-AUROC trong khi **Within-AUROC = 50%** (arXiv:2608.21854); AUC gộp thiếu
   độ phân giải và mọi mô hình tách video theo **độ phân giải/mã màu** (arXiv:2608.11985).
3. **Cadence vs FPS xương sống.** Không tồn tại số quyết định/giây công bố cho cửa sổ trượt trên
   Jetson ⇒ 42.02 dec/s **không có đối ứng**, và không được so với FPS một-khung của bộ phát hiện.
4. **Hao hụt đóng băng của luận văn.** Chưa đo (§2).
5. **Hao hụt INT8 của đường triển khai.** Engine FP32 và INT8 của tháp CLIP **chưa dựng lại cho
   đúng thiết bị** ⇒ chưa có số để trích (§3.1).
6. **Năng lượng biên vs tài liệu.** CoDAT không trừ nền ⇒ chỉ hàng toàn board cùng quy ước.
7. **Ba làn của luận văn với nhau.** Tỷ lệ lớp khác (74.5% vs 38.2% dương), định nghĩa nhóm khác
   (cụm bối cảnh vs nguồn video), nhãn khác bản chất — một hiệu số giữa chúng **không ước lượng
   cho đại lượng nào**.

---

## 13. Nhãn tham chiếu phải sửa trước khi nộp

| # | Nhãn hiện tại | Vấn đề | Sửa thành |
|---|---|---|---|
| 1 | Tham chiếu [21] "25–60 FPS" Jetson YOLO | **Số tổng hợp pipeline 4 luồng 4K** (Seeed 58.82 FPS, 11.02 W), không phải số một mô hình. Cùng board, YOLOv8n TRT FP16 đạt **225.3 FPS** | Nêu rõ là thông lượng tổng hợp đa luồng, kèm số một-mô-hình 4.43 ms / 225.3 FPS |
| 2 | Số GFLOPs/GMACs | Quy ước không nêu; `image_gflops` 8.82 = 2 × 4.41 GMACs | Khai quy ước FLOP = 2 MAC; in **70.6 GMACs ≈ 141 GFLOPs/quyết định** kèm nhãn `[suy ra]` |
| 3 | Nhánh INT8 ONNX | ORT `quantize_dynamic` **lượng tử hoá kích hoạt tại thời điểm chạy** ⇒ có thể **không** phải weight-only như mô tả | Kiểm đồ thị tìm `DynamicQuantizeLinear` / `MatMulInteger` / `ConvInteger`, rồi sửa mô tả theo kết quả |
| 4 | "1,901 + 343" (dữ liệu Önal) | **Không có trong nguồn** | **2,262 khung train / 329 test** (Önal thesis Tablo 3.4, tr. 38) |
| 5 | arXiv 1807.08362 (RESOUND) và 2104.04527 (MSL) | **Sai ID**: 1807.08362 là "An Intersectional Definition of Fairness"; 2104.04527 là bài vật lý | Bỏ cả hai ID; MSL trích **DOI 10.1609/aaai.v36i2.20028** |

**Không thuộc danh sách này:** "FP16 không mất gì". §5.3 và §5.6.5 **đã** phát biểu đúng — tương
đương **đo được** (đồng thuận 1.000, max |Δp| 2.66e-03, chạy 2 lần) cộng chặn **McNemar chính xác
p = 1.000**, kèm câu "dữ liệu không đủ để nói gì mạnh hơn thế". Không cần sửa.

**Mâu thuẫn nguồn Önal phải nêu khi trích:** số ngày thu dữ liệu **19 ngày** (thesis §3) vs **39
ngày** (tóm tắt DIB) vs "~40 ngày" (kết luận thesis); ngày **5/11–13/12/2022** (DIB) vs **5–23/11**
(thesis Tbl 3.3). Bài MTA bị tường phí ⇒ mọi số Unsafe-Net đều lấy từ thesis mở của cùng tác giả +
DOI 10.1016/j.dib.2024.110791.

---

## 14. Vị trí tổng hợp — ba câu

1. **Hao hụt độ chính xác:** chi phí đóng băng của chính hệ thống **chưa đo**; nhưng dải tài liệu
   cho đúng lớp thiết kế (đóng băng + đầu thời gian, 0.23% tham số) là **−12.0 đến +3.3 pp**, và ở
   chế độ dữ liệu ít còn **lãi tới 8.9 pp** ⇒ thiết kế biện hộ được, với phản ví dụ ViFi-CLIP K=2
   (+9.3/+17.5 pp nghiêng full FT) phải công bố kèm. Hao hụt lượng tử hoá đo được là **không phát
   hiện được** ở FP16 (đồng thuận 1.000) và **chưa có số** cho INT8 của đường triển khai.
2. **Hiệu năng:** so sánh trực tiếp chỉ khả thi ở **tham số** (87.85 M khớp chính xác) và **tính
   toán lý thuyết** (70.6 GMACs ≈ 141 GFLOPs, khớp X-CLIP B/32 75 G). Độ trễ, cadence và năng
   lượng **không có đối chứng cùng board**; trục năng lượng quy về toàn board (30.07 mJ/khung) đặt
   B1 giữa CoDAT-S và MoViNet-A1, thấp hơn TimeSformer ~20×.
3. **Đóng góp có thể bảo vệ mạnh nhất không phải con số độ chính xác mà là giao thức:** lạm phát
   13.0–21.8 pp (làn C) / 10.4–15.4 pp (làn B) / 13.5 pp (làn A), cộng hai kết quả cơ chế — rò rỉ
   **xoá** khả năng phân biệt kiến trúc (5.1 → 1.4 pp trên thang 17×) và phần mất đi **không phải**
   thông tin đếm tầm thường (sàn chỉ giảm 1.0 pp khi encoder mất 15.4 pp) — trong một lĩnh vực mà
   **không công trình nào** dùng chia ràng buộc nhóm, báo CI, hay đo năng lượng.

---

## 15. Câu hỏi chưa giải quyết

1. **Có chạy một lượt tinh-chỉnh-toàn-phần CLIP ViT-B/32 trên làn A để đo chi phí đóng băng của
   chính hệ thống không?** Hiện là ô trống duy nhất ở trục hao hụt chính. Nhưng §00 cấm thêm số đo
   mới sau 05/10 (đã có 4 ngoại lệ) ⇒ cần quyết định cho ngoại lệ #5, hoặc khai là giới hạn.
2. **Có dựng lại engine FP32 + INT8 của tháp CLIP cho đúng thiết bị không?** Không có nó thì luận
   văn **không có** số hao hụt INT8 cho đường triển khai, chỉ có hàng lịch sử P1 làn C.
3. **Nhánh ONNX INT8 có thực là weight-only?** Phải kiểm đồ thị trước khi mô tả nó trong bài.
4. **Dải CoDAT có được đặt cùng bảng với 30.07 mJ/khung của luận văn không**, với nhãn (c)/(d) rõ
   ràng — hay §6.1.1 nên hiểu rộng thành cấm luôn so sánh xuyên-board?
5. **Làn C có 8 cụm bối cảnh**, dưới ngưỡng 20–50 cụm của Cameron & Miller. Có nêu đây là giới hạn
   của bootstrap cụm làn C, hay chuyển sang bootstrap wild/t-hiệu chỉnh?
6. **Phát biểu "không công trình nào dùng chia ràng buộc nhóm"** bị hạn chế bởi phạm vi tìm
   (Scholar/ScienceDirect/Semantic Scholar trả 403). Có hạ cấp thành "trong phạm vi tìm được" không?
7. **ESTA @ YOLO26s (81.6%, CI [74.4, 88.5])** ở Phụ lục A.3 hay thành hàng thứ 7 của Bảng 3?
   (còn mở từ phiên trước)
8. **κ cho làn nhị phân iSafetyBench** — đã hoàn tất theo commit mới nhất (D2 là phán quyết ràng
   buộc); có cần đưa κ vào bảng độ chặt thống kê §10 như một dòng nữa không?
9. **Kiểm tra trùng lặp Điều 5.1, tên tác giả / đơn vị / e-mail chính thức / ORCID, và tên tạp chí
   đích cho quy tắc nội bộ Điều 6.2** — vẫn chưa có, vẫn chặn việc nộp.
