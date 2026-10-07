# Duyệt lại dự án + Chiến lược nộp hội đồng

Ngày 2026-09-20. Đối tượng đọc: tác giả luận văn. Thay thế phần "chiến lược" rải rác trong
`plan.md` / `phase-11`. Không thay thế các quy tắc pre-registration — chúng vẫn ràng buộc.

## 0. Chẩn đoán: vấn đề không phải thiếu kết quả

Dự án chạy 4 cổng falsification (G10, G11, G12, G13) trong 24 giờ. Tốc độ nghiên cứu rất cao.
Nhưng:

| Hạng mục giao nộp | Trạng thái |
|---|---|
| Bản thảo luận văn tiếng Việt | **0%** — toàn bộ `report/**` là tiếng Anh; yêu cầu D1 là tiếng Việt + abstract tiếng Anh |
| Nội dung 2 tuần gần nhất trong bản thảo | **0 dòng** — không chương nào nhắc iSafetyBench, G10, G11, E0 |
| Demo chạy được | **Không tồn tại** — không có `deployment/demo-*.py` |
| Video dự phòng | **Không tồn tại** |
| Test touch (giao thức 125 clip mà chính đề cương quy định) | **0/2 đã dùng** |
| P3 đo trên Jetson | **0 phép đo** — mọi file `jetson-measurements/*.json` là `dry_run: true` trên macOS |

**Kết luận: giá trị biên của thí nghiệm tiếp theo đang âm.** Mỗi cổng mới đóng một cánh cửa và
sinh thêm caveat phải viết. Thứ đang thiếu là *quyết định* và *tài liệu*, không phải *kết quả*.

---

## 1. Duyệt lại theo đề cương (MT1 / MT2 / MT3)

Hội đồng chấm theo đề cương, không theo mạch nghiên cứu nội bộ. Chấm trung thực từng mục tiêu:

### MT1 — Kiến trúc YOLO-Transformer. **ĐẠT.**
Đã xây, nhiều biến thể (P1/P3, 6 encoder từ 18k đến 299k tham số), export ONNX, parity đã kiểm
(1.43e-06 với E3, 2.62e-06 với E0). Không có gì phải bào chữa.

### MT2 — Huấn luyện/đánh giá trên Unsafe-Net, kỳ vọng > 95,8%. **KHÔNG ĐẠT NHƯ PHÁT BIỂU — nhưng đã được TRẢ LỜI.**

Đây là chỗ gây mất phương hướng. Câu trả lời đã có, và nó mạnh hơn một con số accuracy:

**Con số 95,8% phụ thuộc giao thức đánh giá, và mức phụ thuộc đã được đo.**

Unsafe-Net, 687 clip, cùng token, cùng kiến trúc, cùng tham số — chỉ khác cách chia fold:

| Encoder | Tham số | k-fold ngẫu nhiên | Leave-camera-out | Δ inflation |
|---|--:|--:|--:|--:|
| E3 DeepSets (coord T1) | 18.044 | 82,5% | 69,6% | **+12,9 pp** |
| E1 d=64 (coord T3) | 94.380 | 84,1% | 62,3% | **+21,8 pp** |
| E1 d=96 (coord T3) | 197.292 | 84,5% | 63,9% | **+20,6 pp** |
| E2 phẳng (coord T1) | 284.556 | 84,6% | 66,7% | **+17,9 pp** |
| E0 RelationTransformer (coord T1) | 298.892 | 83,6% | 66,7% | **+16,9 pp** |

Tất cả là **balanced accuracy** trên OOF gộp. Số tham số và cờ `coord` đã được đối chiếu từng dòng
giữa hai nhóm log: `tier2_binary_logs/{E3,E1_d64,E1_d96,E2,E0}.log` so với
`tier3_lco_logs/` cùng tên — trùng khớp tuyệt đối (ví dụ E1 d=64: `coord=T3 ... params=94,380` ở
cả hai), nên khác biệt duy nhất giữa hai cột là **cách dựng fold**. Năm encoder, dải 18k→299k tham
số, không encoder nào thoát: inflation 12,9–21,8 pp.

iSafetyBench, 1.100 clip, 151 nguồn YouTube (G12, đã pre-register trước khi `--split random`
tồn tại trong code):

| Encoder | split ngẫu nhiên | leave-source-out | Δ inflation |
|---|--:|--:|--:|
| E3 | 90,6% | 75,2% | **+15,4 pp** |
| E0 / E1 d64 / E1 d96 | đang chạy | 80,3 / 78,1 / 77,0% | — |

Hai corpus độc lập (đã kiểm không nhiễm chéo: 0 va chạm hash, khoảng cách Hamming tối thiểu
13/64), hai detector khác nhau, hai bộ nhãn khác nhau, bốn mức dung lượng mô hình. **Hiệu ứng
lặp lại ở mọi cấu hình.**

Lưu ý phản biện quan trọng, có lợi cho ta: LCO có **nhiều** dữ liệu huấn luyện hơn mỗi fold
(8 nhóm giữ-lại-1 vs 5-fold), nên điểm thấp hơn **không** do thiếu dữ liệu. Điều này loại bỏ
lời giải thích cạnh tranh duy nhất đáng kể.

### MT3 — Triển khai biên TensorRT/INT8 đa luồng. **ĐẠT ĐẦY ĐỦ, ĐO THẬT.**

Jetson Orin Nano 8GB, JetPack 6.0, TensorRT 10.3.0, pipeline P1:

| Chỉ số | Số đo |
|---|---|
| Độ trễ INT8 p50/p95/p99 | 8,24 / 8,58 / 8,71 ms — **121,4 FPS** |
| Độ trễ FP16 | 8,62 / 9,13 / 9,21 ms — 116,0 FPS |
| Đa luồng 4 stream @24fps thật | 88,1 FPS tổng, **0 drop**, p95 ≤ 8,6 ms |
| Công suất | 5,22–5,31 W; 34–46 mJ/suy luận |
| Soak 30 phút | **0 throttle**, tj max 56,8 °C |
| Độ chính xác cùng lần chạy | 56/64 FP32, 55/64 FP16/INT8 |
| Parity PyTorch↔TRT | mọi stage PASS < 1e-3 |

Đề cương yêu cầu FPS, latency, đa luồng, INT8 — **tất cả đã có số đo thật trên phần cứng thật.**

**Tổng kết: 2/3 mục tiêu đạt đầy đủ, mục tiêu thứ ba được trả lời bằng một phép đo giải thích
tại sao. Đây là một luận văn thạc sĩ bảo vệ được.**

---

## 2. Duyệt lại kết quả khoa học: cái gì sống, cái gì chết

### Sống (dùng được, đã qua falsification của chính nó)
1. **Protocol inflation +12,9 → +21,8 pp**, lặp lại trên 2 corpus. *Đây là xương sống luận văn.*
2. **G10: +8,0 pp trên count-rule floor** (iSafetyBench LSO, CI [+1,9, +13,5], McNemar p=1,02e-05).
   Chỉ là phát biểu về shortcut đếm-đối-tượng, không hơn. Caveat bắt buộc: cận dưới CI (+1,9)
   nằm *dưới* ngưỡng tầm thường 3 pp — luôn viết "+8,0 pp (95% CI +1,9 đến +13,5)", không bao
   giờ viết "+8 pp" trần trụi.
3. **Camera-label confounding trên Unsafe-Net**, phân rã theo 4 sub-task, có cơ chế: sub-task
   tách được bằng *lớp detector* thì bão hoà (carry 100/100, panel không trả lời được);
   tách bằng *class + geometry* thì tổng quát hoá (zone 76–90% cả hai phía);
   tách bằng *geometry đơn thuần* thì hỏng (walkway 17–36%). Đơn điệu hoàn hảo.
4. **Carry được giải với geometry bằng 0 tuyệt đối** — 86/86 đúng ở cả 6 encoder. Bằng chứng
   trực tiếp đó là tra cứu lớp detector, không phải suy luận quan hệ. Câu này phải vào luận văn.
5. **R47 — bất biến kiến trúc bị bác bỏ, và phiên bản đúng thì sắc hơn:** dung lượng encoder
   *không* quan trọng trên Unsafe-Net (18k–299k, cùng mẫu hỏng) nhưng *quyết định* trên
   iSafetyBench (E0−E3 = +5,1 pp, p=7,7e-06). **Bất biến chưa bao giờ là tính chất của kiến
   trúc — nó là triệu chứng của một bài toán không có tín hiệu trích xuất được.** Một cơ chế
   giải thích cả hai bộ dữ liệu.
6. **Toàn bộ gói đo biên P1** (mục MT3 ở trên).

### Chết (đã falsify, viết thành kết quả âm — không hồi sinh)
- **G11 THẤT BẠI.** Probe whole-frame *không có detector* đạt 74,1% vs E3 75,2%: +1,1 pp,
  cluster CI [−4,8, +6,9] chứa 0, McNemar p=0,625. → **Luận văn không được phép tuyên bố
  detector + relation head làm tăng độ chính xác trên corpus này.**
- **`appearance_only` bằng mô hình đầy đủ** (75,6% vs 75,2%, p=0,551). Mọi câu dùng từ "spatial
  relation" cho kết quả iSafetyBench là sai và phải cắt.
- **Walkway geometry không cứu được** — tâm foot-point hai lớp trùng nhau; 3 giả thuyết cứu đều
  bị bác.
- ~~**P3 không đạt cadence biên.** Đo trên M3 ORT-CPU: 1.825 ms/quyết định = 0,55 dec/s
  (detector chiếm 95,6%). Quy đổi sang Jetson: 1,13 / 1,64 / 1,71 dec/s (fp32/fp16/int8) so với
  **3,0 yêu cầu — trượt ở mọi độ chính xác.**~~
  > **CẬP NHẬT 01/10/2026 — các số của phiên 29/09 đã bị thu hồi.** Engine của phiên đó
  > không khớp với bản ghi thiết bị, và trạng thái xung nhịp không được ghi lại. Đo lại trên
  > engine dựng lại, hai điều kiện xung nhịp, n=3 mỗi điều kiện, đều có đếm giải mã:
  > **ghim nhịp (`sudo jetson_clocks`) đạt 310,00 ms = 3,226 dec/s ở tải crop trung vị, vượt
  > ngưỡng 3,0 với biên +7,5 %**; **xung nhịp động mặc định (DVFS) chỉ đạt 365,88 ms =
  > 2,733 dec/s, TRƯỢT ở mọi mức tải kể cả trung vị**. Từ p95 trở lên TRƯỢT ở cả hai điều
  > kiện. Hệ số chuyển đổi Mac→Jetson của detector là **9,3x**, detector chiếm **60,5 %** chi
  > phí. Thẩm quyền:
  > `reports/finding-2026-10-01-engine-provenance-invalidates-0929-latency.md`.
  >
  > **CẬP NHẬT THỨ HAI, 03/10/2026 — biên +7,5 % bị rút lại, VÀ PHÉP TÍNH THAY THẾ NÓ CŨNG BỊ
  > RÚT LẠI; thay vào đó là một PHÉP ĐO.** Mọi con số "mỗi quyết định" của P3 ở trên là **tổng của
  > các giai đoạn được đo rời** (`jetson-measure-latency.py:211` cộng năm p50 giai đoạn, mỗi giai
  > đoạn đo khi chỉ nó chạy, trên tensor tổng hợp), chứ không phải một vòng lặp tổ hợp. Lập luận
  > trung gian — "mọi chi phí bị bỏ đều chỉ làm tăng độ trễ, nên 3,226 qđ/giây là một chặn trên;
  > sau hiệu chỉnh ghim nhịp ≥ 360,87 ms = ≤ 2,771 qđ/giây, vậy P3 TRƯỢT ở CẢ HAI điều kiện xung
  > nhịp" — **SAI Ở NHÁNH GHIM NHỊP**, và tổng giai đoạn cũng không phải một chặn một phía: nó còn
  > **tính trùng bộ phát hiện** (16 lượt mỗi quyết định, trong khi bước nhảy 8 của vòng lặp chỉ
  > phân bổ 8 lượt), nên **DẤU của sai số ĐỔI theo trạng thái xung nhịp**: tổng đã công bố **cao
  > hơn 7,98 %** khi ghim nhịp và **thấp hơn 39,3 %** dưới DVFS. Không hệ số hiệu chỉnh nào cứu
  > được cấu trúc đó — nó phải bị thay thế.
  >
  > **PHÉP ĐO TỔ HỢP (03/10/2026, 17:33–17:47).** `jetson-p3-multi-stream.py` chạy toàn bộ vòng
  > lặp dưới một đồng hồ treo tường (giải mã → tiền xử lý letterbox 800×800 → bộ phát hiện → chọn
  > + hình học vùng → crop 224² → bộ mã hoá ngoại hình → token quan hệ 74 chiều → đầu E3), đóng
  > cả sáu chi phí bị bỏ. Hai nhánh đều có cổng nhiệt, N=1, 480 khung, 3 đoạn test, FP16:
  > **ghim nhịp 2,493–3,965 qđ/giây (trung vị 3,479), ĐẠT trên 2 trong 3 đoạn**; **DVFS
  > 1,141–1,511 (trung vị 1,441), ĐẠT trên 0 đoạn** (cách hợp đồng 1,99–2,63 lần). Hai nhánh
  > **RỜI NHAU** (max DVFS 1,511 < min ghim nhịp 2,493), nên ghim nhịp đáng **+141,4 % (2,41
  > lần)** trên nhịp tổ hợp so với chỉ **+26,4 % (1,26 lần)** trên tổng giai đoạn — vì phần việc
  > bị bỏ chính là phần nhạy nhất với xung nhịp (dưới `schedutil`, `policy4` nằm ở sàn
  > 115.200 kHz trong mọi mẫu DVFS). **Kết luận đúng hiện nay: nhịp của P3 phụ thuộc TRẠNG THÁI
  > XUNG NHỊP VÀ phụ thuộc CHÍNH ĐOẠN VIDEO; nó chỉ đạt hợp đồng khi ghim nhịp, và không phải
  > trên mọi đoạn.** Luận văn không tuyên bố P3 là hệ thống thời gian thực; đường đã triển khai là
  > P1 (D2), và căn cứ cho D2 không cần P3 phải trượt — một hợp đồng phụ thuộc đoạn video và phụ
  > thuộc một can thiệp mức quản trị là đủ để không chọn P3 làm đường triển khai. Thẩm quyền:
  > `reports/finding-2026-10-03-p3-decision-latency-is-a-stage-sum.md` (§11).
  >
  > Số của phiên 29/09 giữ lại làm hồ sơ, **không được trích dẫn**:
  > ~~**BỊ BÁC BỎ 29/09/2026 bằng đo trực tiếp trên Jetson (TRT FP16, MAXN_SUPER, n=3 có decode).**~~
  > ~~Thực đo **331,19 / 328,88 / 329,69 ms/quyết định = 3,02 / 3,03 / 3,04 dec/s** ở mức crop trung~~
  > ~~vị (54) — **vượt ngưỡng 3,0 nhưng chỉ 0,7-1,3 %** — và **2,60-2,62 dec/s ở trần 128 crop~~
  > ~~(TRƯỢT)**. Kết luận đúng: **P3 nằm TRÊN RANH GIỚI cadence**, không phải "trượt ~2 lần". Hệ số~~
  > ~~chuyển đổi Mac→Jetson của detector thực tế là **9,1x**, không phải 3,11x mà phép ngoại suy đã~~
  > ~~dùng. Detector chiếm **62,1 %** chi phí, relation head **0,05 %**. Nguồn:~~
  > ~~`reports/finding-2026-09-29-jetson-decision-latency.md` §5c.~~
- **Không có lối thoát "hiệu quả tham số".** 17k chỉ là phần head; đầu vào của nó cần 1 lượt
  YOLOv11s + tới 8 lượt MobileNetV3 mỗi khung. Baseline whole-frame vừa chính xác ngang vừa
  **rẻ hơn**. Câu "ngang accuracy với 17k tham số" là SAI, cấm viết.

### Điều duy nhất còn lại bênh vực pipeline entity
Nó sinh đầu ra **theo từng thực thể** (công nhân nào, vật thể nào, ở đâu trong khung) mà bộ phân
loại whole-frame về cấu trúc không thể tạo ra — và cảnh báo an toàn khả dụng cần điều đó. Đây là
lập luận về **độ chi tiết đầu ra**, chưa được kiểm chứng, và **không phải** tuyên bố về accuracy
hay hiệu quả. Viết đúng như vậy.

---

## 3. Chỉnh hướng: bốn quyết định — **ĐÃ ĐƯỢC TÁC GIẢ DUYỆT 2026-09-20**

D1 ✅ · D2 ✅ · D3 ✅ · D4 ✅ *có sửa đổi: đo trên Jetson là bắt buộc*

### D1 — Headline là PROTOCOL INFLATION, không phải một con số accuracy

Câu chủ đề của luận văn:

> Đánh giá chéo-camera làm sụt 13–22 điểm balanced accuracy so với chia fold ngẫu nhiên, lặp lại
> trên hai corpus độc lập và bốn mức dung lượng mô hình. Các con số công bố trên loại dữ liệu
> giám sát an toàn này phụ thuộc giao thức đánh giá, và mức phụ thuộc đã được định lượng.

Vì sao đây là lựa chọn đúng:
- Dương tính, định lượng, có CI, đã pre-register, **lặp lại độc lập**.
- Nó **bao trùm** thất bại G11 như bằng chứng hỗ trợ thay vì là vết thương: nếu tín hiệu thật
  yếu đến mức whole-frame ngang entity-pipeline, thì con số cao trong tài liệu đến từ đâu? Từ
  rò rỉ giao thức. Hai mảnh khớp nhau.
- Nó **trả lời MT2** thay vì né MT2.
- Không hội đồng nào bác được một phép đo lặp lại trên hai bộ dữ liệu.

### D2 — Hệ thống triển khai = P1. Ngừng ép P3 đạt thời gian thực.

P1 có gói đo biên đầy đủ trên thiết bị thật. P3 có 0 phép đo và trượt cadence 2 lần. Đường cong
chi phí đã đo (800×16 → 1,64 dec/s; 512×16 → 3,77 dec/s) **tự nó là một kết quả**: nó định lượng
cái giá tính toán của độ chi tiết theo thực thể.

Viết: P1 là hệ thống được triển khai và đo; P3 là pipeline nghiên cứu độ chính xác; đường cong
chi phí giải thích vì sao P3 chưa triển khai được ở điểm vận hành hiện tại. **Công bố rõ ràng
việc hai chương dùng hai pipeline khác nhau** — Ch6 đã có sẵn khối disclaimer quy kết pipeline,
mở rộng nó.

Hệ quả: **headline MT3 không phụ thuộc phiên đo mới** — P1 đã có đủ số trên thiết bị thật, nên
nếu phiên 09-22 hỏng thì chương triển khai vẫn đứng vững. Nhưng theo D4 phiên đó vẫn **bắt buộc
chạy**: nó phục vụ P3 (biến ngoại suy cadence thành số đo), không phải cứu P1.

### D3 — Giữ E3, KHÔNG chuyển sang E0

E0 đạt 80,3%, hơn baseline whole-frame +6,2 pp (CI [+2,1, +10,5], p=1,59e-05). Số học vượt ngưỡng
G11. **Vẫn không được dùng làm headline.**

Lý do, và phải nói thẳng lý do này ở buổi bảo vệ:
G11 chỉ đích danh E3. Pre-registration ghi nguyên văn *"không thử lại với encoder khác"*. Thay
encoder sau khi chứng kiến encoder được chỉ định thất bại chính xác là hành vi mà quy tắc đó tồn
tại để ngăn. Một phép kiểm định falsification được chạy lại với mô hình mới cho đến khi có mô
hình vượt qua thì không còn là kiểm định falsification.

Trình bày **cả hai sự thật, theo thứ tự này**: (1) phép kiểm đã đăng ký trước thất bại;
(2) một encoder lớn hơn, chạy như kiểm tra chéo, vượt baseline có ý nghĩa thống kê. Báo cáo (2)
mà không có (1) là gian lận học thuật.

Đây cũng là đường rẻ hơn: E3 là artifact triển khai đã đóng băng; chọn E0 tốn một phiên đo lại
trên thiết bị.

### D4 — ĐÓNG BĂNG THÍ NGHIỆM TỪ 21-09, **có một sửa đổi: kết quả đo trên Jetson là BẮT BUỘC**

*(Tác giả duyệt 20-09, sửa đổi so với đề xuất ban đầu.)*

Sửa đổi này **có lợi cho luận văn**. Hiện tại kết luận "P3 trượt cadence 3,0 quyết định/giây" chỉ
là **phép ngoại suy từ Mac** qua hệ số quy đổi 2,15×/3,11×/3,25×. Một phép đo thật trên thiết bị
biến nó thành **số đo hạng nhất**. Trước hội đồng, "chúng tôi đo được" mạnh hơn hẳn "chúng tôi
ước lượng được".

Thí nghiệm được phép, và không gì khác:
- (a) Các run G12 E0/E1 đang chạy — để chạy hết, hoàn thiện bảng inflation.
- (b) **T1 test touch** trên 125 clip blind — chính đề cương quy định giao thức này.
  Phải đăng ký trước, có phê duyệt tác giả.
- (c) **Phiên Jetson bắt buộc** theo `deployment/JETSON-RUNBOOK-2026-09-19.md`, **chỉ E3** (theo D3).
- (d) L3 hiệu chỉnh/ngưỡng (2h) — chỉ khi phần viết xong sớm.

Cắt: G14, corpus thứ ba, L5 appearance upgrade, L6 background mixing, mọi tìm kiếm kiến trúc.

#### Phiên Jetson phải thu được gì (thứ tự ưu tiên — mỗi mục là một mốc bảo vệ được nếu phải dừng sớm)

| # | Hạng mục | Vì sao cần | Thời gian |
|---|---|---|---|
| 1 | Build engine TensorRT cho P3 (detector, appearance, head E3) | Chưa từng tồn tại trên thiết bị | 15–20' |
| 2 | Kích thước model trên đĩa | Đề cương yêu cầu bảng so sánh hiệu năng | tức thì |
| 3 | **Độ trễ từng stage p50/p95/p99, FP16** | **Biến FAIL cadence từ ngoại suy thành số đo** | 2–5' |
| 4 | Parity FP16 vs FP32 (tỉ lệ đồng thuận quyết định) | Chất lượng quyết định sau lượng tử hoá | 2–5' |
| 5 | Video demo dự phòng quay trên thiết bị | Bắt buộc có cho buổi bảo vệ | 20' |
| 6 | Đa luồng 1/2/4/8 stream | Đề cương hứa đa luồng; P1 đã có, P3 thì chưa | 10–15' |
| 7 | Năng lượng + soak 30 phút | Tăng cường; P1 đã có số | 40' |

Ghi chú bắt buộc: **INT8 cho P3 ngoài phạm vi** (không có dữ liệu calib cho P3; chỉ P1 có).
FP32/FP16 thôi. Chỉ build head E3, **không build E0** (theo D3).

**Cảnh báo phải mang theo mọi con số từ hai graph binary:** `val_macro` 89,15% (E3) / 90,92% (E0)
là **chỉ số chọn mô hình lúc huấn luyện, KHÔNG phải độ chính xác trên tập test**. Chưa test touch
nào được tiêu cho hai checkpoint này. Cấm báo cáo chúng như accuracy cho tới khi T1 được tiêu.
Số latency/throughput/kích thước thì thật (trọng số thật, graph thật).

---

## 4. Ngân sách thời gian

Còn 15 ngày tới number freeze (05-10), 20 ngày tới gói bảo vệ (10-10). Bảo vệ: **giữa tháng 10** (đã chốt).

| Việc | Giờ | Phụ thuộc thí nghiệm? |
|---|--:|---|
| Dịch/viết lại thân bài sang tiếng Việt (~4.000 dòng) | 25–35 | **Không** |
| Chương kết quả mới (inflation + iSafetyBench + G10/G11/E0) | 10–14 | Không (số đã có) |
| Viết lại Ch7 thành chương phân tích thất bại | 8–10 | Không |
| Cập nhật Ch1 / Ch6 / abstract theo trật tự trụ cột mới | 5–6 | Không |
| Ch5 biên — giữ E3 + số P1 hiện có | 4 | Không |
| Audit truy vết số + audit trích dẫn (cổng G9) | 6 | Không |
| Pass nhất quán liên chương | 4 | Không |
| Demo + video dự phòng | 3 | Không (Mac, CPU) |
| Lắp ráp, mục lục, danh mục hình/bảng, slide | 4–6 | Không |
| **Phiên Jetson (BẮT BUỘC, D4)** | 6 | Có |
| **Tổng** | **~75–98** | |

**Ràng buộc chặt nhất là bản dịch tiếng Việt (25–35h), và nó không phụ thuộc bất kỳ thí nghiệm
nào.** Phải khởi động ngay hôm nay, chạy song song.

Tối ưu quan trọng: **viết chương mới TRỰC TIẾP bằng tiếng Việt**, không viết tiếng Anh rồi dịch.
Tiết kiệm ~10h.

---

## 5. Lịch

| Ngày | Việc |
|---|---|
| **09-21** | Chốt D1–D4 vào `plan.md`. Đóng băng thí nghiệm. Đăng ký trước T1 vào `test-touch-log.md`. Bắt đầu dịch Ch1–Ch3. |
| **09-22** | **PHIÊN JETSON BẮT BUỘC** (mục 1→5 của bảng ưu tiên D4; 6–7 nếu còn giờ). Tối: demo Mac + quay video dự phòng. |
| 09-23 | Đổ số Jetson vào `number-sheet.md` Block 3. Tiếp tục dịch Ch1–Ch3. |
| 09-24→27 | Viết chương kết quả mới **trực tiếp tiếng Việt**: protocol inflation (dẫn đầu), G10, G11, ablation, E0/R47. |
| 09-28→30 | Viết lại Ch7 thành chương phân tích thất bại. Cập nhật Ch1/Ch6/abstract. |
| 10-01→03 | **T1 test touch** (một lần duy nhất, đã đăng ký). Điền number-sheet. Dịch nốt Ch4/Ch5. |
| 10-04 | Quyết định contingency (phase-11), tác giả ký duyệt wording. |
| **10-05** | **NUMBER FREEZE.** Không số mới sau mốc này. |
| 10-06→10 | Audit truy vết + trích dẫn. Slide. Diễn tập demo ×2. Không thí nghiệm mới. |

---

## 6. Chuẩn bị bảo vệ: ba câu hỏi chắc chắn bị hỏi

**H1. "Đề cương hứa >95,8%. Kết quả đâu?"**
> Con số 95,8% được báo cáo dưới giao thức chia dữ liệu ngẫu nhiên. Chúng tôi đo được rằng giao
> thức đó thổi phồng balanced accuracy 12,9–21,8 điểm trên chính Unsafe-Net và 15,4 điểm trên
> một corpus độc lập thứ hai. Dưới giao thức chéo-camera, mô hình của chúng tôi đạt 69,6–80,3%.
> Chúng tôi không tuyên bố vượt con số công bố; chúng tôi định lượng được con số đó phụ thuộc
> giao thức bao nhiêu. Đó là đóng góp thay thế, và nó lặp lại độc lập.

**H2. "Vậy detector + Transformer của anh có ích gì không?"**
> Trên iSafetyBench, không — và chúng tôi đã tự kiểm chứng điều đó bằng một phép kiểm đăng ký
> trước (G11) rồi công bố thất bại. Một CNN whole-frame không dùng detector đạt ngang. Cái còn
> lại là độ chi tiết đầu ra theo từng thực thể, thứ mà cảnh báo an toàn khả dụng cần, và chúng
> tôi phát biểu đó là lập luận triển khai chưa kiểm chứng, không phải tuyên bố độ chính xác.

**H3. "E0 đạt 80,3% và vượt baseline. Sao không dùng nó?"**
> Vì G11 chỉ đích danh E3 và quy tắc đăng ký trước cấm thử lại với encoder khác. Thay mô hình
> sau khi mô hình được chỉ định thất bại sẽ làm phép kiểm mất giá trị. Chúng tôi báo cáo cả hai:
> phép kiểm thất bại trước, kết quả E0 sau, kèm bằng chứng thời gian cho thấy E0 được khởi chạy
> trước khi baseline có số (19:08:43 vs 20:07:54).

---

## 7. Rủi ro / việc phải xác nhận

1. ~~Ngày bảo vệ~~ **CHỐT 20-09: giữa tháng 10.** Freeze 05-10 và gói 10-10 giữ nguyên, biên độ
   an toàn hẹp. Không nén phần đóng băng thí nghiệm để lấy thêm thời gian viết.
2. ~~Ngôn ngữ~~ **CHỐT 20-09: TOÀN BỘ báo cáo bằng tiếng Việt, văn phong báo cáo nghiên cứu khoa
   học.** Đòn bẩy 25–35h không còn; bản dịch là đường găng. Hệ quả bắt buộc: **chương mới soạn
   THẲNG bằng tiếng Việt**, không viết tiếng Anh rồi dịch (tiết kiệm ~10h). Văn phong: ngôi thứ
   nhất số nhiều hoặc thể bị động, thì hiện tại cho phát biểu khoa học, mọi con số đi kèm CI và
   nguồn truy vết.
3. **2.262 khung hình nhà máy có người nhận dạng được** thuộc diện phiếu đồng thuận
   (`annotation_guide/annotation-sop.md`). Clearance do tác giả khẳng định chỉ áp dụng cho
   Kaggle, **không** áp dụng cho việc đẩy footage lên GitHub.
4. **14 commit chưa push** trên nhánh `next-round`.
5. `deployment/artifacts/multi-stream-results.json` tự ghi chú là "hand-patched" cho khớp bảng
   Chapter 5 — **không trích dẫn file này** như số đo thô; dùng các file gắn nhãn `jetson-*`
   ngày 03-06.
6. Ba báo cáo research vẫn UNVERIFIED: có "Dey et al. 2024" **bịa**, Nadeau-Bengio trích sai
   nguồn, focal loss quy sai tác giả, ~6 arXiv ID sai năm. Audit trích dẫn (phase 10) là bắt buộc.
7. researcher-02 §6 Q6 có lập luận **đảo ngược logic** — nếu nói ra ở buổi bảo vệ sẽ rất tai hại.
   Đã loại khỏi kế hoạch; phải tập câu trả lời đã sửa.

## Câu hỏi chưa giải quyết

1. ~~Ngày bảo vệ~~ **Đã chốt: giữa tháng 10.**
2. ~~Ngôn ngữ~~ **Đã chốt: toàn bộ tiếng Việt, văn phong nghiên cứu khoa học.**
3. Unsafe-Net giữ thành chương đầy đủ hay rút thành mục con? Ảnh hưởng trực tiếp 8–10h.
4. G13 phán quyết: thang dung lượng LSO là 75,2 / 78,1 / 77,0 / 80,3 — biên độ 5,1 pp vượt ngưỡng
   3 pp, nhưng d96 < d64 nên **không đơn điệu chặt**. Cần phán quyết tường minh có ghi ngày trước
   khi trích dẫn R47.
