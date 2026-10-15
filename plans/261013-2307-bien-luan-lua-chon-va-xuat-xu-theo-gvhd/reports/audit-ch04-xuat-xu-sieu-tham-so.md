# Audit Chương 4 — xuất xứ siêu tham số & khe hở biện luận

Phạm vi: `report/vn/chuong-04-chi-tiet-thuc-nghiem.txt` (974 dòng, đọc toàn bộ) +
`report/vn/00-trang-thai-bien-soan.txt` (304 dòng, đọc toàn bộ). Chỉ đọc, không sửa tệp luận văn.
Mọi dòng mã trích dẫn dưới đây đã grep/đọc lại trực tiếp trong kho tại thời điểm viết báo cáo
(13/10/2026), không suy từ tóm tắt nội bộ.

Báo cáo liên quan đã có trong cùng thư mục (không trùng lặp nội dung ở đây, chỉ tham chiếu):
- `research-xuat-xu-hoc-thuat-tham-so-va-phuong-phap.md` — truy xuất xứ HỌC THUẬT (công trình nào)
  cho 10 lựa chọn kỹ thuật, dùng để điền cột "nguồn giá trị" = THEO CÔNG TRÌNH KHÁC dưới đây.
- `audit-ch03-khe-ho-bien-luan.md` — khe hở biện luận của Chương 3, nơi thực sự chứa các giá trị
  kiến trúc (d/heads/ffn/layers/dropout) mà đề bài hỏi nhưng KHÔNG xuất hiện trong Chương 4.
- `audit-ch06-ch07-phep-do-cong-thuc-baseline.md` — công thức & baseline của Chương 6/7.

---

## VIỆC 1 — Bảng xuất xứ siêu tham số

Cột "Phân loại" dùng đúng 5 nhãn đề bài cho. **Không nhãn nào được gán "ĐÃ DÒ TÌM CÓ BẰNG CHỨNG"
trừ khi có tệp log/sweep_results cụ thể trích dẫn được** — rà toàn bộ `sweep_results/`, `runs/`,
`plans/*/reports/*.md` không tìm thấy log nào chứng minh tác giả THỬ giá trị khác cho các siêu
tham số huấn luyện cốt lõi (lr, weight_decay, label_smoothing, clip_grad_norm, batch, T, N_MAX,
CONF detector). Các sweep/ablation có thật trong kho (tau cổng khung, độ phân giải 224/384, kênh
đặc trưng, tổ hợp đặc trưng) đều nhằm **tham số/pipeline KHÁC** câu hỏi, nêu riêng ở cột ghi chú.

### 1.1 Công thức huấn luyện chung (§4.5.1)

| Siêu tham số | Giá trị mã | File:line | Nguồn giá trị | Bằng chứng đã thử giá trị khác? | Ch4 nói xuất xứ? (dòng) | Phân loại |
|---|---|---|---|---|---|---|
| `lr` | 1e-3 | `run_relation_transformer.py:166` | Trùng mặc định `torch.optim.AdamW` (lr=1e-3) | Không | Chỉ nêu giá trị (dòng 527), không nói đây là mặc định thư viện | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** |
| `weight_decay` | 0.01 | `run_relation_transformer.py:166` | Trùng mặc định `AdamW` (wd=0.01) | Không | Dòng 527, không nói là mặc định | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** |
| Lịch học | `CosineAnnealingLR(opt, epochs)` | `run_relation_transformer.py:167` | Lựa chọn loại lịch học của tác giả, không có log thử StepLR/constant | Không | Dòng 528, chỉ nêu tên | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** |
| `label_smoothing` | 0.05 | `run_relation_transformer.py:169` | KHÔNG trùng mặc định `nn.CrossEntropyLoss` (mặc định = 0.0); nhẹ hơn giá trị hay trích (Szegedy 2016 = 0.1) — xem `research-xuat-xu-hoc-thuat-...md` mục 4 | Không | Dòng 529, 541 chỉ nêu giá trị | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** |
| `class_weights(y,k)` | `w = cnt.sum()/(k*cnt.clamp(min=1)); w/w.mean()` | `run_relation_transformer.py:111-114` | Biến thể của công thức inverse-frequency chuẩn (gần `sklearn class_weight='balanced'`), có thêm bước chuẩn hoá theo trung bình không thấy ở sklearn | Không | Dòng 530 chỉ nêu "tính từ phần huấn luyện", không nêu công thức | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** |
| `clip_grad_norm` | 5.0 | `run_relation_transformer.py:150` | Không có mặc định thư viện (hàm bắt buộc truyền `max_norm`); không có tiền lệ trích được (xác nhận ở `research-xuat-xu-...md` câu hỏi mở #1) | Không | Dòng 531, chỉ nêu giá trị | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** |
| Kích thước lô | 32 | `run_relation_transformer.py:117` (`run_epoch(..., bs=32, ...)`) | Mặc định của hàm tác giả viết, không phải mặc định thư viện (PyTorch không có batch-size mặc định) | Không | Dòng 532, chỉ nêu giá trị | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** |
| Số epoch | **60** (k-fold/LCO) **nhưng 120** ở script khác | 60: `run_kfold_relation.py:253` (`--epochs` default); 120: `run_relation_transformer.py:290` (`--epochs` default, đường 4-lớp single-split) | Mặc định script tác giả, hai script hai giá trị | Không | Dòng 533 chỉ nói "60", và dòng 519 khẳng định "**mọi** mô hình làn CLIP dùng **cùng một cấu hình**" — **không đúng cho epoch**, vì script tạo một phần số liệu 4-lớp (ensemble 56/64, 64/64 nhắc ở §4.8.2) mặc định 120 | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** — và thêm: tuyên bố "cùng cấu hình" ở dòng 519 cần soát lại, xem mục Câu hỏi chưa giải quyết |
| `lambda_sub` | 0.3 | `run_relation_transformer.py:118` (default `run_epoch`) | Mặc định hàm tác giả | Không | **Dòng 541-542: Ch4 tự nói rõ "không được dò tìm"** | **MẶC ĐỊNH KHÔNG DÒ (đã công bố trong văn bản)** |
| `lambda_frame` | 0.5 | `run_relation_transformer.py:118` | Mặc định hàm tác giả | Không | **Dòng 541-542: tự công bố** | **MẶC ĐỊNH KHÔNG DÒ (đã công bố trong văn bản)** |

Trích dẫn dòng mã ở Ch4 §4.5.1 ("`run_relation_transformer.py:156-171`", dòng 525) khớp đúng
phạm vi: `set_seed` 107-108, docstring 158-160, `AdamW` 166, `CosineAnnealingLR` 167,
`CrossEntropyLoss` 169, `if vm > best` 177 — `model.load_state_dict`/`return` 179-180 (Ch4 ghi
177-180, đúng). Các trích dẫn này **chính xác**, không có lỗi dòng.

### 1.2 Chọn mô hình & tách validation (§4.5.2)

| Mục | Giá trị mã | File:line thực tế | Ch4 trích dẫn (dòng) | Khớp? | Phân loại |
|---|---|---|---|---|---|
| `test_size`, `random_state` | 0.15, 0 (k-fold path) | `run_kfold_relation.py:317` | "`:300`" (dòng 560) | **Lệch ~17 dòng** — nội dung đúng, số dòng cũ | MẶC ĐỊNH KHÔNG DÒ (chưa công bố) |
| `test_size`, `random_state` | 0.15, 0 (LCO/binary path) | `run_kfold_relation.py:189` | "`:175`" (dòng 560) | **Lệch ~14 dòng** — nội dung đúng, số dòng cũ | MẶC ĐỊNH KHÔNG DÒ (chưa công bố) |
| Vì sao 15% (không 10%/20%) | — | — | Không nói | — | **CHƯA TRUY ĐƯỢC** (không có lý do nào trong mã/log/plan cho đúng số 15) |

**Phát hiện cần nêu riêng vì GVHD khả năng cao sẽ hỏi — `test_size` thứ hai, không có trong Ch4:**
`run_relation_transformer.py` (script tạo một phần số liệu 4-lớp, và hàm `main_binary()` mà
`deployment/scripts/train-binary-head-checkpoint.py` dùng để huấn luyện **chính checkpoint E3 đang
triển khai**) dùng `test_size=0.18`, không phải 0.15:
```
run_relation_transformer.py:231   fit, val = train_test_split(..., test_size=0.18, stratify=strat_key, random_state=0)
run_relation_transformer.py:234   fit, val = train_test_split(..., test_size=0.18, stratify=ytr.numpy(), random_state=0)
run_relation_transformer.py:320   fit, val = train_test_split(..., test_size=0.18, stratify=ytr.numpy(), random_state=0)
```
`plans/2026-09-16-binary-safe-unsafe-reframe/reports/laneb-2026-09-19-binary-head-checkpoints.md`
(dòng 20-45) xác nhận: script triển khai "copies `main_binary()`'s own ... carve verbatim (same
`random_state=0`, `test_size=0.18`) so the split matches what the screening runs used" — nghĩa là
**chính các lần chạy đã chọn kiến trúc E3** cũng dùng 0.18. Ch4 dòng 560 viết test_size=0.15 "giống
nhau cho **cả hai giao thức**" — câu này **đúng** cho hai giao thức mà §4.5.2 đang nói tới (LSO
k-fold và LCO, cả hai qua `run_kfold_relation.py`), nhưng **không nhắc tới** giao thức thứ ba
(`main_binary()`, dùng để chọn kiến trúc & huấn luyện checkpoint triển khai) có giá trị khác hẳn
(0.18). Đây không phải một câu sai, mà là một câu **đúng nhưng phạm vi hẹp hơn người đọc tưởng** —
phân loại: **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** cho riêng giá trị 0.18, và ghi vào câu hỏi mở.

### 1.3 Seed (§4.5.3)

| Mục | Giá trị | File:line | Ch4 có nói? | Phân loại |
|---|---|---|---|---|
| Hàm gieo | `set_seed(s)` | `run_relation_transformer.py:107-108` | Có, dòng 583, đúng | (đã công bố, không cần phân loại) |
| Tập seed (k-fold/LCO, headline) | `{0,1,2}` (`--seeds 3`) | `run_kfold_relation.py:252` | Có, dòng 584 | MẶC ĐỊNH KHÔNG DÒ (**đã công bố giá trị**, nhưng chưa công bố *vì sao 3*) |
| Vì sao 3 seed, không 5/10 | — | — | Không | **CHƯA TRUY ĐƯỢC** |
| Tập seed (checkpoint triển khai E3) | **5 seed**, không phải 3 | `deployment/scripts/train-binary-head-checkpoint.py` + `laneb-2026-09-19-binary-head-checkpoints.md` | **Không** — Ch4 §4.5.3 chỉ nói "3" | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** — cùng kiểu khe hở với mục 1.2: con số headline (3 seed) không phải con số của checkpoint thật đang chạy trên Jetson |

### 1.4 Token quan hệ & bộ phát hiện (§4.3.1, §4.3.2)

| Siêu tham số | Giá trị | File:line | Nguồn | Đã thử giá trị khác? | Ch4 nói xuất xứ? | Phân loại |
|---|---|---|---|---|---|---|
| `T` (số khung) | 16 | `build_relation_tokens.py:43` | Mặc định tác giả; lý do duy nhất tìm được trong kho là "nhất quán với T=16 dùng ở nơi khác" (`plans/2026-07-09-.../researcher-s3-frame-behaviour.md:186`) — đây là lý do NHẤT QUÁN, không phải lý do thực nghiệm so sánh T khác | Không (có T=32 ở `run_resolution_ablation.py` nhưng cho pipeline khác — ảnh 224/384, không phải token quan hệ) | Dòng 245, chỉ nêu giá trị | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** |
| `N_MAX` | 8 | `build_relation_tokens.py:43` | Mặc định tác giả | Không | Dòng 246 | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** |
| `CONF` (ngưỡng tin cậy bộ phát hiện) | 0.25 | `build_relation_tokens.py:43`; CLI `--conf 0.25` ở `build-isafetybench-clip-features.py:147` | **Trùng mặc định `ultralytics` `.predict(conf=0.25)`** | Không (sweep tau tìm được là CHO cổng khung/operating-point, khác ngưỡng này) | Dòng 247, không nói là mặc định thư viện | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** |
| `IoU`/NMS | **không set** → mặc định thư viện (`ultralytics` `.predict()` default `iou=0.7`) | `build_relation_tokens.py:211`, `build-isafetybench-clip-features.py:124` (chỉ truyền `imgsz, conf, verbose, device` — không truyền `iou`) | Mặc định thư viện | Không | **Ch4 không hề nhắc ngưỡng NMS/IoU ở đâu cả** (grep toàn văn 974 dòng = 0 kết quả) | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** — mục bị bỏ sót hoàn toàn, không chỉ "chưa biện luận" |
| `IMGSZ` | 800 (Unsafe-Net) / 640 (iSafetyBench) | `build_relation_tokens.py:43` (800); log `build-isb-tokens.log` dòng đầu (640) | Mặc định tác giả, 2 giá trị cho 2 làn | Không | **Phần phân biệt 800 vs 640 được Ch4 biện luận RẤT TỐT (dòng 255-265, có bằng chứng log)** — nhưng *vì sao 800/640 cụ thể, không 512 hay 1280* thì không | Việc phân biệt 2 làn: đã công bố tốt. Giá trị tuyệt đối: **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** |
| Đệm ngữ cảnh quanh hộp | 30% | `build-isafetybench-clip-features.py:90-91` (`crop_box`) | Mặc định tác giả | Không | Dòng 294, chỉ nêu % | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** |
| `FEAT` | 10 | `build_relation_tokens.py:45` (Ch4 trích "dòng 43", lệch 2 dòng do gộp 5 hằng số vào một trích dẫn) | Mặc định tác giả | Không | Dòng 249 | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** |

### 1.5 Trọng số bộ phát hiện YOLO (Bảng 4.1, §4.3.3, §4.5.5)

Đây là phần được truy xuất xứ **nghiêm ngặt nhất** trong toàn chương (đọc metadata nhúng trong
`.pt`, đối chiếu 2 notebook đã commit, đọc lại ngày 10/10/2026 — Ch4 dòng 674-721). Không phải một
siêu tham số "dò tìm" theo nghĩa sweep, mà là một **chuỗi bằng chứng xuất xứ tạo tác** đầy đủ nhất
trong chương:

| Trường | `detector_best.pt` (YOLOv11s) | `yolo26_best.pt` (YOLO26s) | Phân loại |
|---|---|---|---|
| `optimizer` | SGD | SGD | MẶC ĐỊNH thư viện ultralytics, **không dò** |
| `lr0` | 0.001 | 0.001 | MẶC ĐỊNH KHÔNG DÒ (chưa công bố là mặc định hay không) |
| `imgsz` | 800 | 800 | như trên |
| `epochs` | 100 | 100 | MẶC ĐỊNH thư viện (ultralytics default=100) |
| `patience` | 30 | 30 | **KHÔNG** trùng mặc định thư viện (ultralytics default patience=100) → ghi đè có chủ ý, nhưng không nêu lý do chọn 30 | **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** |
| `batch` | 16 | 16 | Trùng mặc định thư viện | MẶC ĐỊNH KHÔNG DÒ |
| `seed` | 0 | 0 | — | MẶC ĐỊNH KHÔNG DÒ |

Toàn bộ bảng này: **giá trị có xuất xứ xác nhận bằng tạo tác (metadata + notebook), nhưng KHÔNG
có bằng chứng dò tìm** (không log nào cho thấy thử `patience` hay `lr0` khác). Diễn biến theo
epoch (đường hội tụ) **vẫn chưa tái dựng được** — Ch4 tự nhận ở dòng 723-724, và hai tệp `.pt`
**chưa vào git** (dòng 724-728) nên thành viên hội đồng không mở lại được từ kho mã như đang có —
đây là câu hỏi mở quan trọng, xem cuối báo cáo.

### 1.6 Kiến trúc Transformer (d/heads/FFN/layers/dropout) — KHÔNG Ở CHƯƠNG 4

**Sửa phạm vi:** grep toàn văn 974 dòng Chương 4 cho "d=128", "heads=4", "ffn=256"/"FFN", "layers",
"dropout" → **không có giá trị kiến trúc nào trong số này xuất hiện ở Chương 4.** Các giá trị này
nằm ở **Chương 3** (`report/vn/chuong-03-phuong-phap.txt`, dòng 161-170, 339, 365, 430-441), đã
được rà ở `audit-ch03-khe-ho-bien-luan.md`. Ghi lại ở đây để không bỏ sót câu hỏi của đề bài, nhưng
**không tính vào điểm "Chương 4 thiếu biện luận"** vì giá trị không thuộc chương này:

| Kiến trúc | d / heads / FFN / layers / dropout | File:line (mã) | Nguồn | Phân loại |
|---|---|---|---|---|
| `RelationTransformer` (E0), `E2FrameOnlyPos` | 128 / 4 / 256 / **2** / **0.3** | `model_relation_head.py` (ctor), `model_binary_encoders.py` | Tác giả thiết kế, theo `plans/.../design-2026-09-16-binary-encoder-redesign.md`: "Grounded in measured repo facts, not in the paywalled paper... not systematically swept" | MẶC ĐỊNH KHÔNG DÒ (chưa công bố) |
| `TemporalTransformer` (B1/ESTA) | 128 / 4 / 256 / **1** / **0.1** | `model_entity_spatiotemporal.py` | Tỉ lệ FFN=2×d và dropout=0.1 **khớp quy ước Vaswani 2017** (xác nhận ở `research-xuat-xu-hoc-thuat-...md` mục 2) — nhưng *d=128 cụ thể* và *số lớp=1* không có tiền lệ trích được | FFN/d ratio + dropout: **THEO CÔNG TRÌNH KHÁC** (chưa trích dẫn trong luận văn). `d`, số lớp: **MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** |

Lưu ý quan trọng: E0/E2 dùng `dropout=0.3`, **không phải 0.1** như đề bài giả định — đề bài đã
gộp hai gia đình kiến trúc khác nhau (xem `audit-ch03-...md` điểm 9: sự khác biệt này **không
được luận văn giải thích** ở cả Ch3 và Ch4).

### 1.7 Trích đặc trưng CLIP / VideoMAE (§4.3.2, §4.3.4)

| Mục | Giá trị | File:line | Phân loại |
|---|---|---|---|
| Backbone CLIP | `openai/clip-vit-base-patch32`, đóng băng | `build-isafetybench-clip-features.py:37-38` (Ch4 dòng 290, đúng) | **THEO CÔNG TRÌNH KHÁC** — khớp quy ước "frozen CLIP + temporal head" (ActionCLIP Wang 2021, EVL) theo `research-xuat-xu-hoc-thuat-...md` mục 1; luận văn **chưa trích dẫn hai công trình này** trong Ch2/Ch4 |
| `CLIP_DIM` | 512 | dòng 37 | Mặc định kiến trúc CLIP B/32, không phải lựa chọn tác giả — không cần phân loại |
| Backbone VideoMAE | `MCG-NJU/videomae-base-finetuned-kinetics`, đóng băng | `build-isafetybench-videomae-features.py:32` (đọc trực tiếp, xác nhận đúng Ch4 dòng 334) | **THEO CÔNG TRÌNH KHÁC** — checkpoint chính thức MCG-NJU, dùng làm baseline B2 mạnh có chủ đích |
| `frame_sampling` chung CLIP/VideoMAE | Có (`frame_sampling.decode`) | `build-isafetybench-videomae-features.py:8-9,27`; đối chiếu `build-isafetybench-clip-features.py:95-97` | Đã công bố tốt ở Ch4 dòng 339-341 — không có khe hở |

---

## Tổng kết nhanh VIỆC 1 — hai nhóm GVHD sẽ hỏi

**MẶC ĐỊNH KHÔNG DÒ (chưa công bố)** — giá trị đúng trong mã, không có log dò tìm, Ch4 KHÔNG nói
rõ đây là mặc định (thư viện hay tác giả) hay một lựa chọn có lý do:
`lr=1e-3`, `weight_decay=0.01`, loại lịch học CosineAnnealingLR, `label_smoothing=0.05`, công thức
`class_weights`, `clip_grad_norm=5.0`, `batch=32`, `epochs=60` (và đặc biệt: **mâu thuẫn ngầm với
epochs=120** ở script khác), `test_size=0.15` lẫn **`test_size=0.18`** (ở `main_binary()`/checkpoint
triển khai — không nhắc trong Ch4), seed=3 lẫn **seed=5** (ở checkpoint triển khai — không nhắc
trong Ch4), `T=16`, `N_MAX=8`, `CONF=0.25`, **ngưỡng NMS/IoU (im lặng hoàn toàn, dùng mặc định
ultralytics 0.7)**, `IMGSZ` giá trị tuyệt đối 800/640, đệm ngữ cảnh 30%, `patience=30` của bộ phát
hiện YOLO, và (ngoài Ch4, ở Ch3) `d=128`/số lớp của cả hai gia đình Transformer.

**CHƯA TRUY ĐƯỢC** — không tìm được bất cứ lý do/nguồn nào trong kho (mã, log, plan, thiết kế) cho
chính con số đó, dù giá trị trong mã đã xác nhận đúng:
vì sao `test_size=15%` cụ thể (không 10/20%); vì sao 3 seed cụ thể (không 5/10 — càng rõ khi chính
checkpoint triển khai lại dùng 5); vì sao `patience=30` của YOLO (không 50/100); vì sao đệm ngữ
cảnh đúng 30%; đường hội tụ theo epoch của hai lần huấn luyện YOLO (không nhật ký nào còn trong kho).

---

## VIỆC 2 — Khe hở biện luận theo mục (4.2 → 4.9)

Chỉ liệt kê đoạn **có giá trị cụ thể nhưng không có lý do chọn / không có đánh đổi / không có
phương án khác được cân nhắc**. Đoạn chỉ mô tả hệ quả hoặc chỉ dựng lại số liệu (không phải lựa
chọn thiết kế) không tính vào đây dù không "biện luận" — vì không có gì để biện luận.

**4.2.**
- Dòng 46-50: cấu trúc kho 1.100/420/680/151/0 — là tính chất đo được, không phải lựa chọn, **không
  tính là khe hở**.
- Dòng 93 (`KMeans k=8`): nêu giá trị `k=8` không nêu lý do chọn 8 nhóm (không 5/10) — chỉ nêu đó
  là proxy có hạn chế, không nêu *vì sao 8*. Khe hở nhỏ (trị bàn chính ở Ch3, xem báo cáo Ch3 điểm
  10 cho KMeans(8) phía làn 4-lớp).
- Dòng 103-187 (§4.2.3, chưa trích ở trên): giá trị tập huấn luyện/kiểm định bộ phát hiện — cần đọc
  lại để xác nhận có/không lý do split; **chưa rà xong, ghi vào câu hỏi mở.**

**4.3.**
- Dòng 243-250 (`T=16, N_MAX=8, CONF=0.25, IMGSZ=800, FEAT=10`): liệt kê bảng trần 5 giá trị, không
  một dòng lý do nào — đúng kiểu nhận xét GVHD. Đây là ví dụ rõ nhất của toàn chương.
- Dòng 290-295 (backbone CLIP, GEOM=6, đệm 30%, conf=0.25): liệt kê, không lý do — trừ dòng 291
  ("GEOM giữ cạnh embedding, không bị chiếu mất") có nửa câu lý do cho *cách dùng* GEOM, không cho
  *giá trị* GEOM=6.
- Không có đoạn nào nêu **vì sao T=16 mà không 8 hay 32** — câu hỏi mẫu của đề bài, xác nhận có khe
  hở đúng như GVHD nêu.

**4.4.**
- Dòng 386 (`StratifiedGroupKFold(n_splits=10, seed=0)`): dòng 390-398 biện luận RẤT TỐT cho việc
  CHỌN BỘ CHIA (StratifiedGroupKFold, vì phải thoả 2 yêu cầu xung đột) — nhưng **không biện luận
  cho số 10** (vì sao 10 fold, không 5 hay 8 như Unsafe-Net). Khe hở hẹp, nằm đúng trong một mục nói
  chung là mạnh.
- Dòng 493-498 (`k=5` lặp 10 lần cho giao thức "lạm phát" Unsafe-Net 4 lớp): nêu đây là mốc đối
  chiếu không phải giao thức công bố — có lý do VỀ VAI TRÒ, không có lý do vì sao đúng 5×10=50.

**4.5.**
- Dòng 527-533 (bảng 7 siêu tham số huấn luyện): dòng 519-523 biện luận chính sách "dùng chung một
  cấu hình cho mọi kiến trúc" (rất tốt) — nhưng **không một dòng nào trong bảng 527-533 biện luận
  cho GIÁ TRỊ cụ thể** (vì sao lr=1e-3 không 3e-4, vì sao batch=32 không 64, vì sao epoch=60 không
  100) — khớp đúng câu hỏi mẫu "vì sao batch 32" của đề bài.
- Dòng 560-562 (`test_size=0.15`): có lý do CƠ CHẾ (tách từ `tr`, không từ `te`, dòng 564-566) —
  nhưng không lý do cho SỐ 15%.
- Dòng 584 (`seeds=3`): có mô tả CƠ CHẾ gieo và tổ hợp rất kỹ (579-598) — nhưng không một dòng nào
  nói vì sao 3, không 5 hay 10 — khớp đúng câu hỏi mẫu "vì sao 3 seed".

**4.6.** (chưa đọc chi tiết trong lượt này — phần cứng/môi trường). Theo cấu trúc mục (739-803),
đây chủ yếu là khai báo thiết bị, ít giá trị "lựa chọn" cần biện luận; ghi vào câu hỏi mở để xác
nhận không sót khe hở.

**4.7.** (harness đo thiết bị biên, 804-906) — chưa đọc chi tiết trong lượt này; ghi vào câu hỏi
mở.

**4.8.** Không có khe hở — đây là mục quy ước, mọi quy tắc đều có lý do xuất phát từ một thất bại cụ
thể đã xảy ra (xem VIỆC 3).

**4.9.** Không có khe hở đáng kể — giao thức có lý do (chống lạm phát do chọn lọc) nêu ngay dòng
952-953, và 3 ngoại lệ được liệt kê có lý do riêng từng cái (961-969).

---

## VIỆC 3 — Các mục đã biện luận tốt, KHÔNG nên sửa

1. **§4.2.1-4.2.2 (dòng 41-101)** — disclosure rõ ràng về giới hạn cấu trúc (nhãn gắn nguồn, không
   gắn tình huống; KMeans là proxy; không ground-truth box trên iSafetyBench) kèm hệ quả cụ thể cho
   cách đọc số liệu ở chương sau. Văn phong trung thực, không tô vẽ.
2. **§4.3.2-4.3.3 (dòng 271-328)** — phép đối chứng "một lần giải mã, hai nhánh" (cổng G14) và đối
   chứng thế hệ bộ phát hiện (cổng G17), cả hai có **bằng chứng số** (so byte-for-byte giữa hai tệp
   tensor, `max|hiệu|=0.0` cho nhánh không đổi) — mức biện luận bằng thực nghiệm cao nhất trong
   chương.
3. **§4.4.1-4.4.4 (dòng 384-487)** — lý do chọn `StratifiedGroupKFold` (hai yêu cầu xung đột, nêu
   rõ cái gì phải đánh đổi), ba phép kiểm tra bằng số cho tính đúng của phép chia, và đặc biệt đoạn
   "fold 1 không phải fold dễ" (454-483) — một caveat tự nêu, đi ngược lợi ích tường thuật của tác
   giả, kèm số liệu định lượng đầy đủ. Mẫu mực cho cách viết trung thực.
4. **§4.5.5 (dòng 647-728)** — lý do CPU thay MPS có **bằng chứng thất bại thực nghiệm** (một phiên
   YOLO từng chạy MPS, mAP@50 sụp về 0 sau epoch 1), không chỉ trích dẫn docstring; và bảng đối
   chiếu metadata nhúng `.pt` với hai notebook đã commit (674-721) là điều tra xuất xứ kỹ nhất toàn
   luận văn — tự sửa cả một tuyên bố sai trước đó (platform Colab/Kaggle bị quy nhầm) bằng chứng cứ
   mới, và tự khai hai điều còn hở (không có log theo epoch; `.pt` chưa vào git).
5. **§4.8 (dòng 907-946)** — mọi quy ước trình bày số đều bắt nguồn từ một lỗi cụ thể đã xảy ra
   (84.2% bị dùng cho hai thứ khác nhau; "48/64" có 3 cách đọc) — đây là biện luận nhân-quả thực sự,
   không phải quy ước áp đặt từ trên xuống.
6. **§4.9 (dòng 949-974)** — giao thức chạm test có lý do rõ, ba ngoại lệ liệt kê có lý do riêng,
   và hệ quả của việc "còn nguyên 2 lần chạm" được nêu rõ cho việc đọc Chương 6.
7. **Bảng dòng 633-637 (hai giá trị E3 khác nhau, 17,270 và 18,044)** — giải thích đúng cơ chế gây
   lệch (độ rộng đầu vào khác giữa 2 làn) và cảnh báo người đọc đừng coi là lỗi — ngắn nhưng đủ.

---

## Câu hỏi chưa giải quyết

1. Ch4 dòng 519 tuyên bố "mọi mô hình làn CLIP dùng cùng một cấu hình huấn luyện" — nhưng
   `run_relation_transformer.py` (`--epochs` default **120**) và `run_kfold_relation.py`
   (`--epochs` default **60**) khác nhau, và `main_binary()`/checkpoint triển khai dùng
   `test_size=0.18` + **5 seed** thay vì 0.15/3 seed. Cần hỏi tác giả: tuyên bố "cùng cấu hình" có
   nên thu hẹp phạm vi lại (chỉ áp cho k-fold/LCO qua `run_kfold_relation.py`), hay các con số
   120/0.18/5-seed thực ra không được dùng cho bất cứ số liệu nào in trong Ch6 (nếu vậy nên nói
   rõ trong Ch4 để tránh hiểu lầm)?
2. Ngưỡng NMS/IoU của bộ phát hiện — hoàn toàn không được nhắc ở Ch4, dùng mặc định thư viện
   (`iou=0.7`, chưa xác minh triệt để đây có đúng là mặc định hiện hành của phiên bản ultralytics
   8.4.x đang dùng hay không, vì tài liệu ultralytics có đổi mặc định theo version). Nên thêm một
   dòng vào Bảng siêu tham số §4.3.1/§4.5.1.
3. `patience=30` của hai phiên huấn luyện YOLO khác mặc định thư viện (100) — không có log nào giải
   thích vì sao chọn 30; cần hỏi tác giả có nhớ lý do (ví dụ: muốn huấn luyện ngắn hơn trên
   Colab/Kaggle do giới hạn thời gian phiên) hay đã quên.
4. §4.2.3 (dòng 103-187, tập huấn luyện/kiểm định bộ phát hiện) và §4.6-4.7 (phần cứng, harness đo
   thiết bị biên, dòng 731-906) **chưa được rà lại dòng-theo-dòng trong lượt audit này** để tìm khe
   hở biện luận kiểu VIỆC 2 — cần một lượt đọc riêng nếu GVHD hỏi sâu vào các mục đó.
5. Đệm ngữ cảnh 30% quanh hộp (`crop_box`, `build-isafetybench-clip-features.py:90-91`) — không tìm
   được lý do chọn đúng 30% (không 20%/40%) ở bất cứ đâu trong kho; xếp CHƯA TRUY ĐƯỢC.
6. Mục 00-trang-thai-bien-soan.txt tự nhận "Bảng 4.5: nguồn gốc con số độ tin cậy chưa truy được
   hết. Bảng 4.1-4.4 chưa có tệp log tương ứng" — xác nhận một phần các khe hở trên đã được tác giả
   tự biết trước khi audit này chạy; nên đối chiếu danh sách đó với danh sách ở báo cáo này để tránh
   trùng công việc khi sửa.
