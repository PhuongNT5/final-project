# Pha 03 — Bộ phát hiện hành vi trên CUDA (cổng G22)

## Context Links

- Cổng: **G22** — hằng số đã chốt tại `kaggle-detector-retrain/train-behaviour-detector-gpu.py:27`
- Dữ liệu: `annotation_guide/dataset_behaviour{,_ctx,_tight}/`, `annotation_guide/frame_manifest.csv`
- Sinh nhãn: `unsafenet/behaviour_dataset.py::build(ctx, suffix, src_root, dst_root)`
- Runner CUDA: `kaggle-detector-retrain/train-behaviour-detector-gpu.py`, `build-behaviour-labels.py`
- Baseline CPU: `sweep_results/r2/behaviour_yolo.log`, `behaviour_yolo_ctx.log` (được git theo dõi)
- Bài báo: `paper/paper.txt:205` (2262 train + 329 test ảnh), `:323` (**416×416, 4 lớp**, batch 65, 2 GPU)
- Chú giải gốc không công bố: `report/vn/chuong-02-tong-quan.txt:158-165`

## Overview

- **Priority:** P1 — chặn pha 04. Có thể chạy song song pha 02.
- **Status:** pending
- Huấn luyện bộ phát hiện hành-vi-như-đối-tượng trên khung đã chú giải của ta, trên CUDA, và phán
  quyết G22. **[17] không công bố mAP kiểm thử nào** (xem Key Insight 0); baseline CPU của ta 0.791
  (tight) / 0.806 (ctx) ở imgsz 416 — **đã dưới sàn 0.85 của G22**.

## Key Insights

0. **G22 KHÔNG đo được so với [17] — mốc đối chiếu không tồn tại.** Ba sự thật đã kiểm, ghi trước
   khi có số mới: (a) con số "95.66% mAP" **không có trong bài báo** — grep `paper/paper.txt` cho
   `95.66`/`0.957` trả về **0 hit**; nó đã bị thu hồi ở
   `plans/2026-09-06-thesis-2week-final/reports/qa1-adversarial-review-and-defence.md:75`. Đại lượng
   bộ phát hiện duy nhất họ công bố là một **đường cong huấn luyện** ("average mAP ... around 97%",
   `paper.txt:323`), tức mAP **trên tập huấn luyện trong lúc lặp**, không phải mAP giữ-ngoài.
   (b) Họ gán nhãn riêng **329 ảnh của tập test** để đánh giá bộ phát hiện (`paper.txt:205`); ta
   **không có hộp cho bất kỳ clip test nào** — cả 2,262 dòng `frame_manifest.csv` đều từ
   `data/train/` (đã kiểm lại 11/10: `clip_path` 2262/2262 chứa `/train/`). (c) "2,262" của họ là
   **số ảnh**, không phải số hộp; hộp của ta là 5,263 train + 859 val = 6,122 trên 2,244 ảnh.
   ⇒ **Hệ quả đã chốt:** mAP của G22 là một số **giữ-ngoài trong phân bố, cắt ở mức clip, nội bộ
   tập train** — báo cáo **độc lập**, không bao giờ cạnh con số 97% của họ. G22 phán quyết *bộ phát
   hiện của ta có đủ tốt để nuôi pha 04 hay không*, chứ không phán quyết "tái lập được [17] hay không".
0b. **"Đường cong còn đang lên" — tiền đề của công thức GPU — ĐÃ BỊ CHÍNH DỮ LIỆU BÁC BỎ.**
   `kaggle-detector-retrain/train-behaviour-detector-gpu.py:11` nói baseline CPU "early-stopped at
   epoch 15 while mAP50 was still rising" nên đề xuất 25→200 epoch. Đọc lại `results.csv` 11/10:
   **cả hai run chạy đủ 25 epoch, không dừng sớm**, và mAP50 **bão hoà rồi đi xuống**:
   - `beh_tight` (416): đỉnh ep15 = 0.7913, ep25 = 0.7580; 10 epoch cuối dao động 0.743-0.770.
   - `beh_ctx` (416): đỉnh ep15 = 0.8058, ep25 = 0.7665; 10 epoch cuối dao động 0.752-0.788.
   ⇒ **Thêm epoch ở 416 sẽ không đưa mAP50 tới 0.85.** Đòn còn chưa thử là **độ phân giải**
   (416→800) và **dung lượng mô hình** (yolo11s→11m), cả hai chỉ chạy được trên GPU. Phải sửa tiên
   lượng của `gate1-detector-retrain-prediction.md` trước khi chạy, không phải sau.
0c. **Hai run `behaviour_yolo*` là rác MPS, không phải baseline.** mAP50 của chúng ≈ 0 suốt
   (`behaviour_yolo` đỉnh 0.0147; `behaviour_yolo_ctx` đỉnh 0.0021) — dấu hiệu lỗi hội tụ
   Ultralytics-trên-MPS. **Không bao giờ trích dẫn chúng như baseline.** Baseline CPU hợp lệ chỉ là
   `beh_tight` / `beh_ctx` (`device: cpu`, `optimizer: SGD`, `fliplr: 0.0`, `batch: 8`).
0d. **Chi phí CPU đã đo, nên "chặn trên GPU" là một con số, không phải một lời thanh minh.**
   `beh_tight` tốn **14,448 s cho 25 epoch = 578 s/epoch** ở imgsz 416 / batch 8 trên M3 CPU.
   Ở imgsz 800 (gấp ~3.7 lần pixel) ước ~2,100 s/epoch ⇒ 60 epoch ≈ **35 giờ CPU**. Đó là lý do
   arm 800 phải chờ GPU; arm 416 thì đã có và đã bão hoà.
1. **Ba bộ dữ liệu đã tồn tại và đã kiểm.** Cả ba có **1,901 train + 343 val = 2,244** ảnh (so với
   2,262 của [17]) và **phân bố lớp giống nhau hoàn toàn** — chúng chỉ khác **quy ước hộp**:
   `tight` = `ctx=(1.0, 1.0)` (không nở), `base`/`ctx` = `(4.0, 1.8)` (nở theo
   `behaviour_dataset.py::_expand`, cắt theo biên ảnh).
2. **Không có rò rỉ clip test.** `frame_manifest.csv` 2,262 dòng dữ liệu (1,919 train / 343 val),
   mọi clip đến từ `data/train/`; `tools/sample_frames_for_annotation.py` chỉ lấy mẫu từ `data/train/`.
   Chia ở **mức clip**, sửa đúng lỗi rò rỉ `val: images/train` của V10 cũ.
3. **Lệch 18 ảnh cần giải thích.** Manifest ghi 1,919 khung train nhưng bộ dữ liệu có 1,901 ảnh
   train. Chênh 18 — khả năng là khung bị loại vì không còn hộp nào sau khi lọc
   (`PERSON_CLIP_CLASSES={0,1,4,5}`, drop zone/unknown). Phải xác nhận trước khi dùng con số 2,244
   trong bất kỳ câu nào so với 2,262.
4. **Mất cân bằng lớp nghiêm trọng đe doạ tiêu chí "cả 8 lớp có AP > 0" của G22.** Hộp train:
   `c0:1456, c1:1388, c2:446, c3:167, c4:328, c5:337, c6:995, c7:146`. c7 SafeCarry (146) và c3
   CarryOverload (167) là hai lớp mỏng nhất; nếu một trong hai cho AP = 0, G22 trượt dù mAP50 cao.
5. **Số lớp của [17] mâu thuẫn trong chính bài báo.** Table 3 (`paper.txt:205`) mô tả lược đồ 8 lớp
   mà bộ dữ liệu của ta theo, nhưng cấu hình huấn luyện (`paper.txt:323`) ghi rõ **"the number of
   classes was 4"**. Vậy "tái lập trung thực" là 4 lớp hay 8 lớp chưa xác định ⇒ phải chốt trước
   bằng một luật, và nên chạy cả hai (rẻ: cùng nhãn, chỉ lọc lớp).
6. **imgsz 800 là một sai lệch khỏi bài báo, không phải trung thực hơn.** [17] dùng **416×416**.
   Baseline CPU của ta cũng 416. Công thức GPU đặt 800. Vì vậy:
   - **arm trung thực** = imgsz **416** (cùng cấu hình ảnh với họ — *không* so được về số, xem KI 0);
   - **arm tốt-nhất-của-ta** = imgsz **800** (dùng để nuôi pha 04).
   Hai arm phải dán nhãn riêng; không được dùng 800 rồi gọi đó là tái lập.
7. **`fliplr=0.5` xung đột với ngữ nghĩa lớp.** Công thức GPU bật lật ngang, nhưng
   `unsafenet/convlstm.py:105` cố ý **không** lật vì *"walkway side matters"*. c0 WalkwayViol và c4
   SafeWalkway phụ thuộc phía của lối đi. Lật ngang có thể phá đúng hai lớp này. Cần một arm
   `fliplr=0.0` làm đối chứng — hoặc giữ `fliplr=0.5` và xem per-class AP của c0/c4 để chứng thực.
8. **`hsv_h=0.0` là có chủ đích và phải giữ**: c1 UnauthorInt so với c5 AuthorInt là tín hiệu **màu
   áo gilê**. Đây cũng là lớp mà ConvLSTM sụp ở pha 02 — hai pha gặp nhau ở đúng một lớp.
9. **Vi phạm DRY cần xử lý.** `train_behaviour_detector.py` (23 dòng, gốc repo, `device='cpu'`,
   epochs 25, patience 10, `fliplr=0.0`, `amp=False`) tồn tại song song với công thức GPU. Theo
   `development-rules.md` ("DO NOT create new enhanced files, update to the existing files
   directly") phải hợp nhất: một nguồn sự thật cho siêu tham số, hai lớp vỏ thiết bị.
10. **Thay thế kiến trúc phải công bố.** [17] dùng YOLO v4 (darknet, batch 65, subdivisions 32,
    2 GPU). Ta dùng `yolo11s` (ultralytics). Đây là một sai lệch tái lập, phải nêu, không phải
    chi tiết kỹ thuật bỏ qua được.

## Requirements

**Chức năng**
- Dựng nhãn trên Kaggle bằng **chính** `unsafenet.behaviour_dataset.build()` để nhãn bit-đồng nhất
  với local (mẫu đã có: `build-behaviour-labels.py`).
- Chạy arm trung thực (416) và arm tốt-nhất (800) trên cùng một quy ước hộp được chọn.
- Chọn quy ước hộp bằng **một luật chốt trước khi chạy**, không chọn sau khi thấy mAP.
- Xuất `{name}_result.json` gồm per-class P/R/AP50/AP50-95 và ba cờ cổng
  (`gate_map50_pass`, `gate_floor_pass`, `gate_all_classes_pass`).
- Trích được mAP50/mAP50-95 cuối từ log (log CPU hiện chứa mã điều khiển ANSI làm che số cuối).

**Phi chức năng**
- CUDA bắt buộc (MPS huấn luyện hỏng âm thầm, `chuong-03-phuong-phap.txt:484-487`).
- Không đổi một hằng số nào ở `train-behaviour-detector-gpu.py:27` — đổi là mất hiệu lực tiền đăng ký.
- Dataset Kaggle **private**.

## Architecture

```
annotation_guide/dataset (chú giải thực thể) + frame_manifest.csv (nhãn mức clip)
        │
        ▼ unsafenet/behaviour_dataset.py::build(ctx=...)
   dataset_behaviour_tight (1.0,1.0)   dataset_behaviour_ctx (4.0,1.8)
        │                                   │
        └────────── luật chọn chốt trước ───┘
                        │
                        ▼  kaggle-detector-retrain/train-behaviour-detector-gpu.py (CUDA)
        arm trung thực imgsz=416          arm tốt-nhất imgsz=800
                        │                                   │
                        ▼                                   ▼
       mAP giữ-ngoài, báo cáo độc lập          nuôi pha 04 (tầng giảm khung)
                        │
                        ▼  {name}_result.json  ->  phán quyết G22
```

## Related Code Files

**Sửa**
- `train_behaviour_detector.py` — biến thành lớp vỏ CPU gọi cùng khối siêu tham số với bản GPU (bỏ
  trùng lặp công thức), **không** tạo tệp mới
- `kaggle-detector-retrain/train-behaviour-detector-gpu.py` — thêm `--imgsz` làm đối số tường minh
  để chạy được arm 416 mà không sửa hằng số cổng; thêm lọc 4-lớp tuỳ chọn cho arm Table-3-vs-cấu-hình
- `sweep_results/r2/` — thêm log hai arm mới (`.log` được git theo dõi)

**Tạo**
- `plans/.../reports/g22-bo-phat-hien-ket-qua.md`

**Không sửa**
- `unsafenet/behaviour_dataset.py` — nó là nguồn nhãn bit-đồng nhất giữa local và Kaggle
- `annotation_guide/dataset_behaviour*/` — nhãn đã dựng, không chú giải thêm trong pha này
- `train-behaviour-detector-gpu.py:27` — ba hằng số cổng

## Implementation Steps

1. Xác nhận commit tiền đăng ký pha 01 tồn tại.
2. **Giải thích lệch 18 ảnh**: đối chiếu `frame_manifest.csv` (1,919 train) với số tệp ảnh thực
   (1,901). Ghi nguyên nhân vào báo cáo. Nếu là khung rỗng-hộp, nêu rõ con số đúng là 2,244 và
   khoảng cách so với 2,262 của [17] là 18 khung (0.8%).
3. **Chốt luật chọn quy ước hộp, viết trước khi chạy** (đề xuất): chọn quy ước có **mAP50 cao hơn
   trên tập val ở arm 416**, và nếu hai quy ước chênh < 1.0 pp thì chọn `tight` vì nó không đưa thêm
   giả định hình học nào. Luật này phải nằm trong tiền đăng ký pha 01 hoặc một phụ lục commit trước
   khi chạy.
4. Hợp nhất công thức: đưa khối siêu tham số thành một hằng số dùng chung; `train_behaviour_detector.py`
   chỉ còn đặt `device='cpu'`. Chạy lint.
5. Dựng nhãn trên Kaggle qua `build-behaviour-labels.py` cho cả `tight` và `ctx`; assert số ảnh
   1,901/343 mỗi biến thể.
6. **Arm trung thực**: `imgsz=416`, epochs 200, patience 30, `fliplr=0.5`, `hsv_h=0.0`, `yolo11s`.
7. **Arm đối chứng lật**: `imgsz=416`, `fliplr=0.0` — chỉ để đọc per-class AP của c0/c4. Rẻ, và nó
   trả lời được xung đột ở Key Insight 7.
8. **Arm tốt-nhất**: `imgsz=800`, còn lại như arm trung thực.
9. **Arm 4-lớp**: lọc nhãn còn `{0,1,2,3}`, `imgsz=416` — arm tương ứng câu "the number of classes
   was 4" của `paper.txt:323`, tức arm gần cấu hình của họ nhất. Vẫn **không** so số với họ (KI 0).
10. Phán quyết G22 trên arm được luật ở bước 3 chỉ định. Đọc cả ba cờ. Đặc biệt kiểm AP của c3 và c7.
11. Viết `reports/g22-bo-phat-hien-ket-qua.md`: bảng arm × (mAP50, mAP50-95, per-class AP50), phán
    quyết G22, quy ước hộp được chọn và trọng số nào đi tiếp sang pha 04, và danh sách **sai lệch
    tái lập** (yolo11s thay YOLO v4; 2,244 thay 2,262; 8 lớp thay 4 lớp nếu chọn vậy; imgsz nếu 800).

## Todo List

- [ ] Xác nhận commit tiền đăng ký pha 01
- [ ] Giải thích lệch 1,919 manifest so với 1,901 ảnh
- [ ] Chốt + commit luật chọn quy ước hộp (trước khi chạy)
- [ ] Hợp nhất công thức CPU/GPU (bỏ trùng lặp `train_behaviour_detector.py`)
- [ ] Dựng nhãn trên Kaggle cho `tight` và `ctx`, assert 1,901/343
- [ ] Arm trung thực imgsz 416
- [ ] Arm đối chứng `fliplr=0.0` (đọc AP c0/c4)
- [ ] Arm tốt-nhất imgsz 800
- [ ] Arm 4-lớp (theo `paper.txt:323`)
- [ ] Phán quyết G22 + kiểm AP c3/c7 khác 0
- [ ] `reports/g22-bo-phat-hien-ket-qua.md` + danh sách sai lệch tái lập

## Success Criteria

- Nhãn trên Kaggle bit-đồng nhất với local (assert số ảnh + checksum một tệp nhãn mẫu).
- Arm trung thực (416) và arm tốt-nhất (800) đều có log được git theo dõi và `{name}_result.json`.
- G22 được phán quyết bằng **một** arm do luật chốt trước chỉ định, với cả ba cờ được đọc.
- Quy ước hộp được chọn bằng luật, không bằng việc nhìn kết quả.
- Danh sách sai lệch tái lập được viết ra, không ẩn đi.
- `train_behaviour_detector.py` không còn là bản sao công thức thứ hai.

## Risk Assessment

| Rủi ro | Giảm thiểu |
|---|---|
| c3 (167 hộp) hoặc c7 (146 hộp) cho AP = 0 ⇒ G22 trượt dù mAP50 ≥ 0.90 | Nêu trước như kết cục khả dĩ. Không được bỏ tiêu chí "cả 8 lớp" sau khi thấy số. Nếu trượt vì lý do này, kết luận công bố là "ngân sách chú giải chưa đủ cho hai lớp mỏng", và pha 04 bị bỏ theo luật ở `plan.md`. |
| `fliplr=0.5` phá c0/c4 (phía lối đi có nghĩa) | Arm đối chứng `fliplr=0.0` ở bước 7 trả lời trực tiếp. |
| Chọn quy ước hộp sau khi thấy mAP = lạm phát do chọn lọc | Luật chọn commit trước khi chạy (bước 3). |
| imgsz 800 bị trình bày như "tái lập" | Hai arm dán nhãn riêng; arm cùng cấu hình ảnh với [17] là arm 416. |
| Quota GPU miễn phí hết giữa 4 arm | Thứ tự: trung thực 416 → 4-lớp → 800 → đối chứng lật. Arm sau chỉ chạy nếu arm trước không đã kết luận. |
| 2,244 so với 2,262 bị dùng như "bằng nhau" | Luôn viết cả hai số kèm chênh 18 khung. |
| Thay YOLO v4 bằng yolo11s làm so sánh mAP không chặt | Vô hiệu hoá bằng KI 0: **không có so sánh mAP nào với [17]**. mAP50 của ultralytics và mAP darknet khác định nghĩa ngưỡng, và số của họ là đường cong huấn luyện ⇒ hai bên không cùng đại lượng. |

## Security Considerations

- Khung ảnh là ảnh nhà máy thật ⇒ dataset Kaggle **private**, không public notebook có ảnh.
- `kaggle.json` / token API không commit.
- `annotation_guide/dataset_behaviour*/` không được đẩy lên nơi công khai; chỉ nhãn + manifest là
  dẫn xuất, ảnh gốc vẫn thuộc giấy phép dataset Unsafe-Net (dùng học thuật).

## Next Steps

- Phụ thuộc vào: pha 01.
- Chặn: pha 04 (trọng số bộ phát hiện là đầu vào tầng giảm khung).
- Song song với: pha 02.
- Theo sau: pha 06 ghi phán quyết G22.

## Unresolved Questions

1. **"Tái lập trung thực" là 4 lớp hay 8 lớp?** Table 3 của [17] cho 8, cấu hình huấn luyện
   (`paper.txt:323`) cho 4. Nếu là 4 thì tầng giảm khung chỉ cần phát hiện lớp unsafe và luật
   "khung thuộc lớp safe bị loại" trở thành "khung không có phát hiện unsafe bị loại" — đổi hẳn
   ngữ nghĩa pha 04.
2. ~~95.66% đo ở ngưỡng nào~~ — **đã giải quyết 11/10, câu hỏi dựa trên số không tồn tại.**
   `paper/paper.txt` chỉ chứa bốn phần trăm hai chữ số thập phân: 95.81% (×2), 92.31%, 90.91%,
   85.78%. **Không có 95.66%.** Đại lượng bộ phát hiện duy nhất là đường cong huấn luyện "around
   97%" (`:323`). 329 ảnh test của họ không có đối ứng ở phía ta. Xem KI 0.
3. "Gate 1" ở `brief.md` §8 đã từng được phán quyết chưa? Nếu rồi thì G22 là đăng ký lại một cổng
   đã dùng.
4. Lệch 18 ảnh (1,919 manifest so với 1,901 tệp) — khung rỗng-hộp, hay lỗi dựng? Nếu là lỗi dựng
   thì phải dựng lại trước khi huấn luyện.
5. Có nên chú giải thêm hộp cho c3/c7 để cứu tiêu chí "cả 8 lớp" của G22? Việc đó mở lại chiến dịch
   chú giải và vượt phạm vi kế hoạch — nhưng nếu không làm thì G22 có thể trượt vì dữ liệu, không
   vì phương pháp.
