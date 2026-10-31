
# Phản biện 10 — Triển khai biên & Tái lập

Phiên 31/10/2026. Trục: TRIỂN KHAI BIÊN & TÁI LẬP. CHỈ ĐỌC, không đo mới.

## Bảng phán quyết trục

| Hạng mục | Phán quyết | Mức độ |
|---|---|---|
| Tuyên bố "thời gian thực" P1 | ĐỨNG — 167.291 ms/qđ = 5.978 qđ/giây, biên hẹp nhất 2.39× (60 khung/giây), tái lập ±2% qua 16 ngày/3 phiên | VỮNG |
| Tuyên bố "trên mép" của P3 trong đề bài phản biện | **SAI — đang trích số đã bị luận văn TỰ RÚT LẠI** (vi phạm Đ3, dòng 284-287 tệp archive) | SỬA TẠI CHỖ (xem mục 1) |
| Phán quyết đối sánh engine (Q12) | THIẾU TỆP, không chặn số | NHẸ |
| Lỗi TensorRT 10.3.0 với MobileCLIP-S1 (Q14) | TÀI SẢN — chuỗi lập luận KÍN, phát hiện báo lỗi được | TRUNG BÌNH-CAO |
| Mìn GELU-tanh siglip2B16 (Q16) | THẬT, chặn khả năng di chuyển runtime nghiêm trọng | CAO, bắt buộc cảnh báo |
| Tái lập số đầu bảng Ch5 | ĐẾN ĐƯỢC commit, dừng ở "-dirty" | TỐT, có 1 khe hở |
| Tạo tác 11 JSON thiết bị | Đã commit (8610c74), **CHƯA ĐẨY remote** — xác nhận hôm nay | RỦI RO MẤT ĐĨA |
| 3 ONNX tháp vòng 3 | Không vào git, script dựng lại có nhưng phụ thuộc mạng + phiên bản thư viện không khoá | RỦI RO TÁI LẬP TRUNG BÌNH |
| Cấu trúc script | `deployment/scripts/`: 7/45 tệp > 200 dòng, max 358 — chấp nhận được; rủi ro thật nằm ở thiếu lock file | NHẸ (biên); THIẾU (lock) |

---

## 1. Tuyên bố "real-time" — SỬA một lỗi trong chính đề bài phản biện

Đề bài phản biện ghi: *"P3 nằm trên mép (3.02–3.04 qđ/giây ở tải cắt trung vị, 2.60–2.72 ở trên
mức đó)"*. Số **3.02 là số đã bị chính luận văn RÚT LẠI**, ghi rõ tại
`report/vn/chuong-05-trien-khai-bien-ghim-xung-nhip.txt:284-287`:

> "Phát biểu 'P3 đạt 3.02 quyết định/giây với biên dương 0.7%' (vi phạm Đ3: đó là một con số đo
> dưới trạng thái xung nhịp không khai báo)... Con số 3.02 nằm giữa hai điều kiện và không thuộc
> điều [kiện nào]... Giá trị đúng: nhịp tổ hợp **2.733 quyết định/giây (−8.9%)** [DVFS, và đó
> cũng chỉ là tổng giai đoạn, KHÔNG PHẢI MỘT NHỊP]."

Số 2.60–2.72 trong đề bài không khớp bất kỳ hàng nào trong Bảng 5.4/5.4b hiện hành (gần nhất:
p95 DVFS 2.481, p95 ghim 2.883 — `chuong-05-trien-khai-bien.txt:596`). Kết luận: khung "P3 trên
mép" là khung đã lỗi thời/sai nguồn — **không dùng để đánh giá luận văn**.

Phán quyết ĐÚNG, đọc tại nguồn hiện hành (`chuong-05-trien-khai-bien.txt:604-634`, Bảng 5.4b):

| Điều kiện | 3 đoạn test | So cổng 3 qđ/giây |
|---|---|---|
| DVFS mặc định | 1.141 / 1.441 / 1.511 qđ/giây | **TRƯỢT cả 3/3**, cách 1.99×–2.63× |
| Ghim xung nhịp | 2.493 / 3.479 / 3.965 qđ/giây | ĐẠT 2/3, trượt 1.20× ở đoạn còn lại |

Hai nhánh **rời nhau** (max DVFS 1.511 < min ghim 2.493). Luận văn tự phát biểu dòng 634:
**"Luận văn này không tuyên bố P3 là hệ thống thời gian thực; đường đã triển khai là P1."**
Đây là khung đúng và đã có trong văn bản — đề bài phản biện không nên gán nhãn "trên mép" cho P3.

### Kiểm ba quy tắc tự đặt (Đ1–Đ4, `chuong-05-trien-khai-bien.txt:95-151`)

| Quy tắc | Nội dung | Có vi phạm còn sót trong Ch5 hiện hành? |
|---|---|---|
| Đ1 | Engine dựng ≤29/09 không trích | Không thấy vi phạm còn sót |
| Đ2 | Tạo tác phải nhúng trạng thái sysfs | Không thấy; mục 5.2.1 tự xử lý ca biên `quiet:false` giả |
| Đ3 | Điều kiện xung nhịp phải dán nhãn tường minh | Đã tự rút 3.02 (xem trên) — KHÔNG còn vi phạm trong thân chương, chỉ còn trong đề bài phản biện |
| Đ4 | Giải mã phải trong ngân sách | Đã tự rút "giải mã 83.6%" — không còn vi phạm |

Luận văn có một cơ chế tự kiểm rất hiếm thấy: mục 5.2.2 liệt kê **6 phát biểu đã rút lại** (phiên
29/09, 30 engine cũ, 3.02 qđ/giây, 83.6% giải mã, ngân sách tổng-giai-đoạn thiếu 3.97×, "95.7%
mAP"). Không phải "phép đo chi phí bị đọc thành độ chính xác" nào còn sót trong thân chương — tất
cả các ca đó nằm trong lịch sử retraction đã công khai, không bị chôn. Đây là bằng chứng mạnh cho
Y6 (tự đo, tái đo được).

### Mâu thuẫn còn sót trong văn bản (KHÔNG liên quan đề bài, tôi tìm thấy khi đọc)

`chuong-05-trien-khai-bien.txt` mâu thuẫn với CHÍNH NÓ về trạng thái đồng bộ git:
- Dòng 351-353 (§5.4.4): "Cả sáu đã được đồng bộ về kho mã... hạn chế xuất xứ... **đã được khắc
  phục**."
- Dòng 1175 (§5.10, kết luận (3)): "Toàn bộ chín tạo tác nhịp tổ hợp của B1 **đã được đồng bộ**
  về kho mã."
- Dòng 1122-1125 (§5.9, hạng mục (12)): "Sáu tạo tác R48 **chưa đồng bộ** về kho mã... Đây là
  hạn chế **nghiêm trọng nhất** trong danh sách này vì nó treo chính kết quả tiêu đề của chương."

Xác minh tại đĩa (`git ls-files`): cả 6 tệp (`r48-composed-cadence-fp16-20261003T{155728,155806,
155843}` + `r48-idle-droop-probe-fp16-20261001T{234145,234242,234322}`) **ĐÃ** được track từ
commit `df1dab7`. Vậy §5.9 hạng mục (12) là **văn bản tồn đọng (stale) chưa xoá** sau khi (Đ3/Đ2)
vấn đề đã fix — không phải hạn chế thật. Mức độ: nhẹ (văn bản, không chặn số), nhưng đáng sửa vì
tự mâu thuẫn ngay trong một chương và §5.9 tự xưng "danh mục không phải lời xin lỗi".

---

## 2. Q12 — phán quyết đối sánh engine ba tháp vòng 3

| Gì | Có | Thiếu |
|---|---|---|
| Văn xuôi phán quyết | `JETSON-RUNBOOK-2026-10-19.md:318-321` — cos/rel đủ cho cả 3 tháp | — |
| Tệp .npz tham chiếu | 3 tệp, track trong git (`deployment/artifacts/r48-parity-fixture/*.npz`) | — |
| Tệp JSON cấu trúc `r48-parity-fp16-*` | — | KHÔNG CÓ cho 3 tháp mới (chỉ B/32 có, 2026-10-01) |
| STDOUT/log của lần chạy đối sánh trên thiết bị | — | Không tìm thấy trong kho |

Mức nghiêm trọng: **nhẹ, không chặn số nào**. clipB16/siglip2B16 đã ĐẠT đối sánh (cos
0.999973/0.999918) — số đã dùng trong report nội bộ (`ket-qua-thiet-bi...md:59`) hợp lệ vì nguồn
là văn xuôi runbook có timestamp + điều kiện guard sạch. Đây đúng là **việc lưu bản ghi**, như
audit đã kết luận (`audit-2026-10-20-doi-sanh-engine-va-do-chinh-xac-so.md:52,112`). Khuyến nghị:
chẻ ≤10 dòng mã để script tự ghi JSON — không urgent vì không ảnh hưởng kết luận đã đóng băng.

---

## 3. Q14 — lỗi TensorRT 10.3.0 với MobileCLIP-S1: tài sản hay tiêu sản?

**Phán quyết: TÀI SẢN. Chuỗi lập luận kín.**

Chuỗi bằng chứng (`JETSON-RUNBOOK-2026-10-19.md:191-209`):
1. ONNX máy chủ đúng: torch↔ONNX rel 1.22e-05, cos 1.000000 (loại trừ lỗi xuất ONNX)
2. Đầu vào thiết bị bit-đồng nhất: tổng kiểm điểm ảnh lệch đúng 0 (loại trừ lỗi nạp/tiền xử lý/
   thứ tự kênh/chuẩn hoá)
3. Sai **bất biến** qua 2 độ chính xác (FP32 rel 0.241-0.247, FP16 rel 0.241-0.259) × 4 cỡ lô
   (loại trừ cơ chế do độ chính xác — FP32 có độ phân giải ~1e-7, lẽ ra chỉ lệch ~0.006%)
4. Giả thuyết cạnh tranh "lệ thuộc scale embedding/FP16" được **chính tác giả đề ra rồi tự bác**:
   agreement là trùng hợp độ lớn, vì FP32 (không bị trần FP16) sai **giống hệt** (dòng 210-226)

⇒ Chuỗi loại trừ đủ 4 nguồn lỗi cạnh tranh hợp lý (xuất ONNX, nạp dữ liệu, hiệu chuẩn, kênh/chuẩn
hoá) và một giả thuyết tinh vi hơn (precision-scale coincidence) đã bị kiểm chứng-bác bỏ bằng thực
nghiệm đối chứng (FP32 build) — không chỉ bằng suy luận. **Kín.**

Việc chưa làm để đủ sức báo upstream (lên NVIDIA/TensorRT): **chẻ đồ thị bằng engine cắt ngắn**
để định vị op (đã tự ghi nhận, dòng 207, 127 báo cáo). Không làm không chặn kết luận triển khai,
nhưng BẮT BUỘC phải làm trước khi gửi bug report — NVIDIA cần repro tối thiểu (minimal graph),
không nhận "cả graph sai 24%" làm báo cáo hữu ích. Cách báo: minimal reproducible ONNX (bisect),
kèm 2 cặp (FP32 cos 0.932605/rel 0.247, FP16 cos 0.932087/rel 0.259, batch 16) + ONNX gốc đã PASS
đối sánh host. Đây là một phát hiện xuất bản được độc lập với luận văn (không cần chờ luận văn).

---

## 4. Q16 — mìn 3.64e9 trong GELU-tanh siglip2B16

**Phán quyết: THẬT, cảnh báo bắt buộc toàn văn.**

Vị trí: `layers.9/mlp/activation_fn/Mul_1`, hạng x³ trong phân rã GELU-tanh, giá trị **3.64e9**,
vượt trần FP16 (65,504) **~5.56×10⁴ lần**. TensorRT 10.3 **hợp nhất** (fuse) mẫu này trước khi
tràn số nên engine hiện tại qua đối sánh — nhưng đây là **tính chất của chính runtime TensorRT
10.3, không phải của mô hình hay của FP16 nói chung.**

Cảnh báo BẮT BUỘC đi kèm mọi khuyến nghị dùng tháp này ở nơi khác:
1. **Không tái lập được trên runtime khác** (ONNX Runtime, TVM, một bản TensorRT khác không fuse
   đúng mẫu đó, hoặc bất kỳ compiler thực thi phân rã GELU-tanh theo đúng nghĩa chữ) — sẽ tràn số
   FP16 thật, không phải giả thuyết.
2. Không có hàng rào clip/safe-decomposition nào trong đồ thị xuất — rủi ro nằm ở chính ONNX, nếu
   mang sang runtime khác phải tự thêm clip hoặc dùng decomposition khác của GELU-tanh.
3. **Mức mất khả năng di chuyển: CAO** — không phải "chậm hơn" mà là "có thể SAI SỐ hoặc NaN/Inf
   trên runtime không fuse". Đây khác về loại so với vấn đề tốc độ của mobileclip_s1.

---

## 5. Tái lập — truy một tuyên bố đầu bảng Ch5 đến tận commit

**Tuyên bố chọn**: "Nhịp đã triển khai là 167.291 ms/quyết định = 5.978 quyết định/giây"
(`chuong-05-trien-khai-bien.txt:361`).

| Bước | Kết quả |
|---|---|
| Số in trong luận văn | 167.291 ms (p50), p95 198.224 |
| → Tệp JSON | `deployment/artifacts/jetson-measurements/r48-composed-cadence-fp16-20261003T155843.json` → xác minh bằng Python: `steady_state_ms_per_decision.p50 = 167.291` ✓ khớp đúng từng chữ số |
| → Trạng thái thiết bị trong JSON | `device_state.quiet=true`, `clocks_pinned.any=false`, `competing_processes=[]`, `gpu_clocks.cur_hz=714MHz` (giữa dải DVFS) ✓ khớp mô tả thân chương |
| → Nhãn git | `git_commit = "6319b21-dirty"` |
| → Commit | `6319b21` **CÓ TỒN TẠI** trong lịch sử (`docs(report): track per-clip logits...`) |
| → Commit chứa bản sửa tiền xử lý | `cc27d82` ("move BGR->RGB swap after resize") **LÀ TỔ TIÊN** của `6319b21` (`git merge-base --is-ancestor` xác nhận) — khớp mô tả "sau khi sửa" |
| → Script đo | `deployment/scripts/benchmark-r48-composed-cadence.py` còn tồn tại ở HEAD hiện hành |

**Đi được tới đâu**: JSON → device_state → commit tổ tiên → script, đủ cho một người ngoài kho
xác minh **con số khớp JSON** và **điều kiện đo khớp mô tả**. **Khe hở duy nhất**: nhãn
`"6319b21-dirty"` nghĩa là lúc đo có **thay đổi chưa commit** trên nền 6319b21 — kho không giữ
diff đó, nên không dựng lại **bit-for-bit** được trạng thái mã chính xác tại thời điểm đo, chỉ
dựng lại được trạng thái **gần nhất đã biết**. Vì commit tổ tiên `cc27d82` (bản sửa thật, có ý
nghĩa) đã nằm trong lịch sử, phần "-dirty" nhiều khả năng là thay đổi không ảnh hưởng số (đường
dẫn cục bộ, debug print) — nhưng đây là suy đoán, không verify được.

**Người ngoài kho dựng lại được đến đâu**: đọc JSON + script + commit — đủ để **tin cậy số liệu
và điều kiện đo**, KHÔNG đủ để **re-run bit-for-bit trên Jetson khác** vì (a) không có Jetson,
(b) thiếu bản ghi uncommitted diff, (c) không có lock file phiên bản TensorRT/CUDA/driver ngoài
những gì ghi trong `device_state` (JetPack R36.5.0, TRT 10.3.0 — đây ghi trong runbook, không
trong JSON này cụ thể, nhưng khớp nhau giữa các tạo tác).

---

## 6. Rủi ro tạo tác

### 6.1 Trạng thái push — xác minh HÔM NAY (31/10/2026, sau `git fetch origin`)

```
## next-round...origin/next-round [ahead 2]
```

**XÁC NHẬN: `8610c74` và `6a5d01c` (gồm toàn bộ 11 JSON đo thiết bị 205 KB không tái tạo được)
VẪN CHƯA ĐẨY LÊN REMOTE.** Rủi ro mất đĩa còn nguyên — commit cục bộ không bảo vệ khỏi mất máy.
Đây là hành động đơn giản nhất, rẻ nhất, cần làm ngay (`git push origin next-round`), không phụ
thuộc quyết định khoa học nào.

### 6.2 Ba ONNX tháp vòng 3 — `export-round3-towers-onnx.py` dựng lại được thật không?

Không vào git (2 tệp 329/354 MB > hạn 100 MB GitHub, 1 tệp 82.5 MB > hạn 48 MB tự đặt của kho —
xác nhận: `deployment/export/` chỉ track `r48-b1-clip-temporal-3seed-fp32.onnx` và
`r48-b1-head-cost768-3seed-fp32.onnx`, KHÔNG track 3 tệp `r48-tower-{clipB16,siglip2B16,
mobileclip_s1}-fp32.onnx`).

Đọc mã `deployment/scripts/export-round3-towers-onnx.py`:

| Phụ thuộc | Trong kho? | Rủi ro |
|---|---|---|
| Clip fixture `.mp4` (24 clip, `r48-parity-fixture/`) | CÓ, track trong git | Không |
| Trọng số pretrained (HF Hub: `openai/clip-vit-base-patch16`, `google/siglip2-base-patch16-224`; open_clip/timm cho mobileclip) | KHÔNG trong kho — tải qua mạng lúc chạy | **CÓ** — không có SHA/revision pin trong mã (`TOWERS` dict chỉ ghi `model_id` dạng string, không ghi revision hash) |
| Phiên bản thư viện (`transformers`, `open_clip_torch`, `timm`) | KHÔNG có `requirements.txt` ở gốc kho (đã tìm, không có) | **CÓ** — không lock; `pip show` hiện tại cho `transformers 5.4.0`, `open_clip_torch 3.3.0` — không xác minh được đây có phải đúng phiên bản dùng lúc đo 19/10 hay không |
| Ngưỡng đối sánh (rel ≤1e-4, cos ≥0.9999) | Có trong script, tự-gate | Không |

**Kết luận**: script **không chỉ** phụ thuộc trạng thái máy cục bộ — nó còn phụ thuộc **mạng** và
**phiên bản thư viện không khoá**. Script sẽ chạy và tự kiểm (gate pass/fail) nên nếu chạy lại mà
trọng số/thư viện trôi, nó **sẽ tự báo FAIL** (đổi tên `.PARITY_FAILED`) chứ không âm thầm sinh số
sai — đây là điểm thiết kế tốt. Nhưng việc **không dựng lại y hệt engine đã đo** (chỉ dựng lại một
engine *tương đương trong sai số gate*) là một giới hạn tái lập thật, không phải lý thuyết.

---

## 7. Mã & chất lượng công trình — khung thực dụng

| Phạm vi | Số liệu | Đánh giá |
|---|---|---|
| `deployment/scripts/*.py` (script LOAD-BEARING cho mọi số Ch5) | 45 tệp, 7 (16%) > 200 dòng, max 358 (`jetson-measure-latency.py`) | **Chấp nhận được** — không tệp nào > 400 dòng, không ảnh hưởng khả năng đọc-kiểm |
| `run_*.py` ở gốc kho | 42 tệp (không phải "~100" như đề bài nêu) | Không trực tiếp ảnh hưởng tái lập Ch5 (đây là script huấn luyện, không phải script đo thiết bị) |
| Tệp > 200 dòng ở gốc | tối đa 781 dòng (`run_finetune_bbox.py`) | Rủi ro bảo trì thật nhưng KHÔNG ảnh hưởng thẩm định Ch5 — nằm ngoài phạm vi trục này |
| **Lock file phiên bản (`requirements.txt`/`pyproject.toml`)** | **KHÔNG TỒN TẠI** ở gốc kho | **Rủi ro tái lập thật, mức trung bình-cao** — không ai (kể cả tác giả, 6 tháng sau) biết chắc phiên bản `transformers`/`torch`/`tensorrt`-python binding nào đã tạo ra các số trong Ch5 |

Đánh giá thực dụng: cấu trúc ~100 script không phải vấn đề thẩm định tự nó (đây là kho nghiên cứu,
không phải sản phẩm, và script load-bearing nằm riêng trong `deployment/scripts/` ở trạng thái
tốt). Vấn đề thật là **thiếu lock file** — đây mới là thứ chặn tái lập số, không phải độ dài tệp.

---

## Câu hỏi chưa giải quyết

1. Nội dung chính xác của phần "-dirty" trên nền commit `6319b21` khi đo 167.291 ms — không có
   trong kho, không verify được là có ảnh hưởng số hay không.
2. Phiên bản chính xác của `transformers`/`open_clip_torch`/`timm`/TensorRT-python lúc đo 19-20/10
   — không có lock file; `pip show` hiện tại không chứng minh được là cùng phiên bản.
3. Việc TensorRT 10.3.0 chuyển đổi sai MobileCLIP-S1 có trung tính về chi phí không (nếu không,
   161.314 ms không chặn trên chi phí một engine đúng) — câu hỏi do chính luận văn nêu, tôi không
   có cách giải thêm vì cần chẻ đồ thị trên thiết bị (ngoài phạm vi phiên đọc-chỉ này).
   Op nào gây lỗi cũng chưa định vị — cần engine cắt ngắn, chưa làm.
4. §5.9 hạng mục (12) ("chưa đồng bộ") là văn bản tồn đọng mâu thuẫn với §5.4.4/§5.10 — nên sửa
   trước khi bảo vệ, nhưng tôi không rõ đây là sơ suất biên tập hay có ý giữ lại có chủ đích mà
   tôi chưa hiểu.
5. Đề bài phản biện (chủ tịch) trích "P3 3.02-3.04 / 2.60-2.72 qđ/giây" — số 3.02 đã bị luận văn
   tự rút; 2.60-2.72 không khớp bảng nào tôi tìm thấy. Cần hỏi chủ tịch nguồn số này từ đâu, có
   phải bản thảo cũ hơn bản tôi đọc không.
