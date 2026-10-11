# Pha 02 — Chân 1: tái lập hoặc bác bỏ mốc no-bbox 57/64 / 85.78%

## Context Links

- Cổng: **G21** (pha 01)
- Mã: `unsafenet/convlstm.py` (182 dòng), `unsafenet/data_index.py::blind_split`, `unsafenet/config.py::PAPER`
- Tạo tác hiện có: `sweep_results/r2/temporal_convlstm.pt` (06/09/2026 01:01, 8,437 B, **gitignored**)
- Cache: `cache_r2/raw_train_112px_t16.npy` (566,16,112,112,3), `cache_r2/raw_test_112px_t16.npy` (125,16,112,112,3)
- Bài báo: `paper/paper.txt:327` (452/64), `:334` (32/11/13/8 và Table 5/6), `:332` (Fig 11 val_loss)
- Văn bản phải sửa: `report/vn/chuong-06-danh-gia.txt:1525-1527`
- Luật thiết bị: `report/vn/chuong-03-phuong-phap.txt:484-488`
- Nhiễm do dừng sớm theo test: `report/vn/chuong-06-danh-gia.txt:1567-1570`

## Overview

- **Priority:** P1 — milestone đầu tiên và cao nhất của kế hoạch.
- **Status:** pending
- Mốc 85.78% hiện được luận văn gọi là *"cấu hình có thể tái dựng từ các bảng của họ"*
  (`chuong-06-danh-gia.txt:1525`) — nghĩa là **đọc ra từ bảng của họ, chưa từng được tái lập**.
  Phép đo của ta cho **75.64%**. Pha này đóng khoảng cách 10.14 pp đó, hoặc chứng minh nó không
  đóng được, rồi làm cho câu chữ của luận văn khớp với điều tìm thấy.

## Key Insights

1. **Khoảng cách tập trung ở một lớp.** c1 UnauthorInt 2/11 = 18.18% so với 9/11 = 81.82% của họ,
   góp **+15.91 pp** trong thiếu hụt macro 10.14 pp. c2 và c3 của ta **vượt** họ (100% so với 76.92%
   và 87.50%). Đây không phải mô hình yếu toàn cục.
2. **Bảng của [17] tự nhất quán.** `unsafenet/config.py::PAPER['nobbox'].per_class = [31/32, 9/11,
   10/13, 7/8]` cho đúng 57 clip đúng và đúng macro 85.78%; `['full']` cho đúng 62 và 95.81%. Cỡ lớp
   test của ta trùng khớp [32, 11, 13, 8]. Vậy mục tiêu so sánh là hợp lệ và đã kiểm số học.
3. **H3 — thiết bị — là nghi phạm mạnh nhất và rẻ nhất để loại trừ.** `convlstm.py:137` mặc định
   `device='mps'`, và chính luận văn ghi rằng trên macOS 14 phép `addcmul_` của AdamW **hỏng âm
   thầm** (`chuong-03:484-487`). Tạo tác 06/09 gần như chắc chắn được huấn luyện ở chế độ đó. Nếu
   đúng, 75.64% không phải một phép đo về ConvLSTM mà là một phép đo về một optimizer bị hỏng.
4. **H1 — thiếu dữ liệu huấn luyện, chưa ai nêu.** `blind_split(val_frac=0.18)` cắt 83 clip làm val
   ⇒ ConvLSTM của ta khớp trên **369** clip, còn [17] khai báo **452**. Thiếu **18%** dữ liệu. [17]
   không khai báo một tập val riêng, nhưng Fig 11 (`paper.txt:332`) vẽ `val_loss`/`val_accuracy`.
5. **H4 — [17] có thể đã chọn trên tập test.** Nếu `val_*` của Fig 11 chính là 64 clip test (hoặc
   `validation_split` của Keras lấy từ 452 mà không khai báo), thì 85.78% là số **được chọn trên
   test** và **không thể tái lập** dưới một giao thức trung thực. Giả thuyết này không kiểm trực tiếp
   được; nó được **chặn** bằng cách báo cáo đồng thời số trung thực của ta và số chọn-trên-test của ta.
6. **H2 — độ phân giải.** 112 px toàn khung từ nguồn 960×540: người cao ~10-15 px. c1 UnauthorInt
   phân biệt với c5 AuthorInt bằng **tín hiệu màu áo gilê** — đúng lý do công thức bộ phát hiện giữ
   `hsv_h=0.0`. Ở 112 px toàn khung tín hiệu đó gần như không tồn tại về không gian. [17] không gặp
   vấn đề này vì YOLO v4 đã lọc/khoanh vùng trước khi ConvLSTM nhìn.
7. **Dừng sớm của ta là theo val, không theo test.** `convlstm.py:131-133` chọn epoch tốt nhất theo
   `val_macro` trên 83 clip val; test chỉ được suy luận một lần ở cuối. Pha này **không thêm đường
   nhiễm thứ ba** như `chuong-06:1567-1570` cảnh báo — và phải giữ đúng như vậy.
8. **Tăng cường dữ liệu gần như không có.** Chỉ nhiễu Gauss cộng 0.02 với xác suất 0.5
   (`convlstm.py:104-105`); cố ý không lật ngang vì *"walkway side matters"*. Với 369 clip, đây là
   một nghi phạm phụ (H5).
9. **Tạo tác yếu xuất xứ.** `.gitignore:11` (`sweep_results/**/*.pt`) ⇒ không theo dõi. Cùng loại
   khiếm khuyết mà Ch5 §5.6 nêu cho tạo tác P1 88.06 FPS đã nghỉ. Cách sửa đúng chuẩn repo **không
   phải** theo dõi tệp nhị phân mà là xuất một JSON **được theo dõi** chứa dự đoán theo clip + cấu
   hình + chỉ số, giống các `*_oof.json` đã có ngoại lệ trong `.gitignore:15-25`.

## Requirements

**Chức năng**
- Chạy lại `unsafenet/convlstm.py` trên **CUDA** với cấu hình **giống hệt** tạo tác 06/09 (một biến:
  thiết bị). Đây là arm A, arm quyết định G21.
- Các arm phụ chốt trước, mỗi arm đổi **đúng một** biến so với arm A: H1 (fit 452 qua k-fold thay vì
  cắt val cố định), H2 (`--px 224`), H5 (tăng cường mạnh hơn).
- Xuất JSON được theo dõi cho mọi arm: dự đoán theo clip, sensitivity từng lớp, cả hai loại macro.
- Tính và dán nhãn **riêng** hai độ đo: `macro_paper` (trung bình sensitivity 4 lớp) và `macro_f1`
  (sklearn). Không được gộp.
- Đưa ConvLSTM vào `run_unsafenet_r2.py` thành một stage, không tạo script mới.

**Phi chức năng**
- Huấn luyện trên CUDA (Kaggle/Colab). Không huấn luyện trên MPS.
- Tệp < 200 dòng; `unsafenet/convlstm.py` đang 182 dòng ⇒ phần xuất JSON đặt ở module khác.
- Mọi arm dùng cùng 5 seed, cùng `blind_split(seed=0)` để so sánh được theo cặp.

## Architecture

```
cache_r2/raw_{train,test}_{px}px_t{t}.npy      (xây lại nếu px đổi)
        │
        ▼
unsafenet/convlstm.py::run(device='cuda')       arm A: 1 biến = thiết bị
        │   ├── arm H1: fit 452 (k-fold thay cắt val cố định)
        │   ├── arm H2: px=224
        │   └── arm H5: tăng cường mạnh
        ▼
sweep_results/r2/temporal_convlstm{arm}.pt      (gitignored, nhị phân)
        │
        ▼
unsafenet/<xuất JSON>  ->  sweep_results/r2/convlstm_{arm}_oof.json   (ĐƯỢC THEO DÕI)
        │                     dự đoán/clip + per_class + macro_paper + macro_f1 + cấu hình
        ▼
so với unsafenet/config.py::PAPER['nobbox']  ->  phán quyết G21
```

## Related Code Files

**Sửa**
- `unsafenet/convlstm.py` — thêm `--device cuda` làm giá trị khuyến nghị trong docstring; thêm tham
  số tăng cường cho arm H5; **không** đổi mặc định hiện có (đổi mặc định sẽ làm arm A không còn là
  bản sao của tạo tác 06/09)
- `run_unsafenet_r2.py` — thêm `convlstm` vào `choices` của `stage` và nhánh gọi tương ứng
- `.gitignore` — thêm ngoại lệ `!sweep_results/r2/convlstm_*_oof.json`

**Tạo**
- `unsafenet/convlstm_report.py` (< 120 dòng) — xuất JSON được theo dõi từ `Pt`/`yt`/`Pv`/`yv`;
  tính `macro_paper` (dùng lại `unsafenet/metrics.py::macro_accuracy`) và `macro_f1`
- `kaggle-detector-retrain/train-convlstm-gpu.py` — runner CUDA, gọi `unsafenet.convlstm.run()`
  giống cách `build-behaviour-labels.py` gọi `unsafenet.behaviour_dataset.build()`

**Không sửa**
- `unsafenet/config.py::PAPER` — hằng số so sánh, đã kiểm số học
- `unsafenet/data_index.py::blind_split` — đổi nó sẽ phá so sánh theo cặp với tạo tác 06/09

**Không xoá**
- `sweep_results/r2/temporal_convlstm.pt` — chứng cứ của phép đo 75.64%

## Implementation Steps

1. **Trước mọi thứ:** xác nhận commit pha 01 đã tồn tại (`git log`).
2. Viết `unsafenet/convlstm_report.py`: nạp một tạo tác `.pt`, xuất JSON gồm `config`, `per_clip`
   (tên clip, nhãn, dự đoán, vector xác suất), `per_class_sensitivity`, `macro_paper`, `macro_f1`,
   `top1`, `n_fit`, `n_val`, `n_test`, `device`, `git_commit`.
3. Chạy nó trên tạo tác 06/09 **trước tiên** → `convlstm_mps0906_oof.json`. Đây là **miễn trừ (3)**
   của §4.9 (tính lại từ dự đoán đã lưu), không tiêu lần chạm. Commit nó: phép đo 75.64% từ đó có xuất xứ.
4. Thêm stage `convlstm` vào `run_unsafenet_r2.py`.
5. Viết `kaggle-detector-retrain/train-convlstm-gpu.py`: cài phụ thuộc, dựng cache 112px/t16 bằng
   chính `clip_tensor()` của repo (bit-đồng nhất với local), gọi `run(device='cuda')`.
6. **Arm A** — chạy trên CUDA, mọi tham số giống tạo tác 06/09: `px=112 t=16 seeds=5 epochs=18 bs=8
   lr=3e-4 n_cls=4`. Xuất JSON. **Đây là arm phán quyết G21.**
7. Đọc `per_class_sensitivity` của arm A. Theo dự đoán khả bác bỏ của G21:
   - c1 rời khỏi ~18% ở đa số seed ⇒ H3 (thiết bị) được chứng thực.
   - c1 vẫn kẹt ⇒ H3 bị loại; chạy tiếp arm H2 rồi H1.
8. **Arm H2** (`--px 224`): dựng cache mới `raw_{train,test}_224px_t16.npy`, chạy lại. Một biến.
9. **Arm H1**: thay cắt val cố định bằng k-fold phân tầng trên 452 clip unsafe để **mọi** clip đều
   vào fit ở một fold nào đó, giữ dừng sớm theo val-trong-fold. Báo cáo macro ngoài-fold trên 452.
   Lưu ý: số này **không** so trực tiếp với 57/64 (khác tập), nó chỉ định lượng ảnh hưởng của 18%
   dữ liệu thiếu.
10. **Arm H5**: bật tăng cường mạnh hơn (jitter thời gian, scale/crop; **không** lật ngang — phía
    walkway có nghĩa, và `convlstm.py:105` đã ghi lý do).
11. Chẩn đoán c1: xuất ma trận nhầm lẫn 4×4 của arm tốt nhất; xác định c1 bị gán sang lớp nào. Nếu
    c1 → c0 chiếm đa số thì đó là lẫn không gian (vùng/zone); nếu c1 → c2 thì là lẫn ngữ cảnh panel.
12. Viết `plans/.../reports/chan-1-ket-qua.md`: bảng mọi arm × (top-1, macro_paper, macro_f1,
    sensitivity 4 lớp), phán quyết G21 theo đúng bảng ba nhánh, và **nhãn nào §6.12.5 được dùng**.
13. Sửa chữ `report/vn/chuong-06-danh-gia.txt:1525-1527` theo kết quả — **chỉ sửa nhãn mô tả, không
    thêm giá trị số mới** (mốc đóng băng 05/10 chỉ cho sửa chữ).

## Todo List

- [ ] Xác nhận commit tiền đăng ký pha 01 tồn tại
- [ ] `unsafenet/convlstm_report.py` + ngoại lệ `.gitignore`
- [ ] Xuất `convlstm_mps0906_oof.json` từ tạo tác cũ, commit (cấp xuất xứ cho 75.64%)
- [ ] Thêm stage `convlstm` vào `run_unsafenet_r2.py`
- [ ] `kaggle-detector-retrain/train-convlstm-gpu.py`
- [ ] **Arm A** trên CUDA, cấu hình giống hệt — phán quyết G21
- [ ] Đọc sensitivity c1 của arm A, đối chiếu dự đoán khả bác bỏ H3
- [ ] Arm H2 (`px=224`) nếu H3 bị loại
- [ ] Arm H1 (k-fold trên đủ 452) để định lượng 18% dữ liệu thiếu
- [ ] Arm H5 (tăng cường) nếu H1/H2 chưa đóng được khoảng cách
- [ ] Ma trận nhầm lẫn + chẩn đoán hướng lỗi của c1
- [ ] `reports/chan-1-ket-qua.md` + phán quyết G21 + nhãn cho §6.12.5
- [ ] Sửa chữ `chuong-06-danh-gia.txt:1525-1527`

## Success Criteria

- Arm A chạy xong trên CUDA với **đúng một** biến khác tạo tác 06/09, và có JSON được theo dõi.
- G21 được phán quyết theo đúng một trong ba nhánh đã chốt ở pha 01, kể cả nhánh "không tái lập được".
- Phép đo 75.64% có xuất xứ kiểm được (JSON theo dõi trong git), không còn là số từ một `.pt` bị ignore.
- `report/vn/chuong-06-danh-gia.txt:1525` nói đúng điều đã tìm thấy: hoặc "đã tái lập", hoặc "đọc từ
  bảng của họ, chưa tái lập".
- Hướng lỗi của c1 được định danh (lẫn sang lớp nào), không chỉ nêu là "thấp".
- Không có lần chạm test mới nào bị tiêu ở pha này.

## Risk Assessment

| Rủi ro | Giảm thiểu |
|---|---|
| **Tiêu một lần chạm test ngoài ý muốn.** `convlstm.py:run()` luôn suy luận trên `Xt` ở cuối mỗi seed. | Mọi arm vẫn tính `Pt` nhưng **không được đọc** `yt` để phán quyết trong pha này; G21 phán quyết trên arm A, và arm A là bản sao cấu hình của một lần chạm **đã xảy ra** (06/09). Nếu kết luận này không đứng được về mặt giao thức, G21 phải chuyển sang phán quyết trên `val_macro` (83 clip) và việc so với 85.78% dồn vào lần chạm gộp ở pha 05. **Phải quyết trước khi chạy** — xem câu hỏi mở #1. |
| Arm A trên CUDA không chạy lại được vì cache thiếu/khác | Dựng cache bằng chính `clip_tensor()` của repo trên Kaggle, giống mẫu `build-behaviour-labels.py`; assert shape `(566,16,112,112,3)` và `(125,16,112,112,3)`. |
| CUDA sửa được khoảng cách ⇒ mọi số làn C huấn luyện trên MPS đều đáng ngờ | Đây là kết quả **có giá trị cao** chứ không phải rủi ro; nhưng phải kiểm ngay nhánh nào khác của làn C cũng huấn luyện trên MPS, và ghi vào pha 06. |
| 4 arm × 5 seed vượt quota GPU miễn phí | Arm A trước, một mình. Arm phụ chỉ chạy khi dự đoán khả bác bỏ chỉ về phía chúng. Đây là lý do thứ tự arm được chốt trước. |
| Nhiều arm ⇒ lạm phát do chọn lọc | Arm A là arm duy nhất phán quyết G21 (`chuong-06:150-152`). Các arm khác là **chẩn đoán**, kết quả của chúng không được trình bày như số tiêu đề. |
| Sửa `report/vn/` sau mốc đóng băng | Chỉ sửa **chữ** (nhãn mô tả). Mọi giá trị số mới ở lại thư mục kế hoạch. Điều kiện G20 được sao y. |

## Security Considerations

- Không đưa dữ liệu video lên Kaggle công khai. Dataset Kaggle phải **private**; cache `.npy`
  (325 MB + 72 MB) chứa khung ảnh nhà máy thật ⇒ coi như dữ liệu hạn chế.
- `kaggle.json` không được commit. Kiểm `.gitignore` trước khi chạy.
- JSON được theo dõi chỉ chứa tên clip, nhãn, xác suất — không chứa khung ảnh.

## Next Steps

- Phụ thuộc vào: pha 01 (commit tiền đăng ký).
- Chặn: pha 05 (cấu hình ConvLSTM thắng G21 là đầu vào của tầng hợp nhất).
- Có thể chạy song song: pha 03 (khác tệp, khác tạo tác).
- Theo sau: pha 06 ghi phán quyết G21 vào bảng cổng.

## Unresolved Questions

1. **Arm A có tiêu một lần chạm test không?** Nó tái tạo một cấu hình mà tập test đã bị chạm cho
   (06/09), nhưng nó sinh **trọng số mới** và một `Pt` mới. §4.9 miễn trừ (3) chỉ bao "tính lại từ
   dự đoán đã lưu". Phải quyết dứt điểm ở pha 01, trước khi chạy — nếu tính là một lần chạm thì kế
   hoạch tiêu cả 2 lần (pha 02 + pha 05) và hết ngân sách.
2. [17] huấn luyện ConvLSTM trên 452 clip rồi lấy `val_loss` của Fig 11 từ đâu? Nếu từ 64 clip test
   thì 85.78% là số chọn-trên-test và G21 không thể "đạt" dưới một giao thức trung thực. Có cách nào
   chặn được điều này từ chính bài báo (ví dụ số iteration, cỡ batch, tỉ lệ val) không?
3. [17] đưa ảnh cỡ nào vào ConvLSTM? `paper.txt` cho 416×416 cho YOLO v4 nhưng không nêu cỡ đầu vào
   ConvLSTM. Nếu không tìm được, arm H2 chỉ là thăm dò, không phải tái lập.
4. Nếu H3 (thiết bị) đúng: những tạo tác làn C nào khác đã huấn luyện trên MPS, và có số nào trong
   `report/vn/` đứng trên chúng? Nếu có, mốc đóng băng 05/10 ngăn sửa giá trị — xử lý thế nào?
5. c1 UnauthorInt chỉ có 11 clip test. Với n=11, một clip = 9.09 pp sensitivity. Biên 2.0 pp của G21
   trên macro có nghĩa gì khi một lớp dao động 9.09 pp mỗi clip? Có cần biên riêng cho c1?
