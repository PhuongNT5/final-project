# PHÁN QUYẾT G23 — CƠ CHẾ GIẢM KHUNG CỦA [17], ĐO HOÀN TOÀN TRÊN VAL

| | |
|---|---|
| Ngày | 2026-10-11 |
| Làn | **C duy nhất** (Unsafe-Net, 691 clip) |
| **Lần chạm test tiêu** | **0** — toàn bộ báo cáo này sống trên val slice 83 clip. Ngân sách vẫn còn **1**. |
| Tạo tác | `ket-qua-g23-val.json`, `fusion_g23_*.pt` (có `Pt` nhưng **chưa chấm**) |
| Mã | `run_g23_framegate_val.py`, `unsafenet/behaviour_clip.py` |

[17] gán **+10.03 pp** macro (85.78 → 95.81) cho việc hợp nhất bộ phát hiện hành vi, với **tầng
giảm khung** ("chỉ giữ khung mang bằng chứng không an toàn") là cơ chế được nêu tên. Tách làm hai
câu hỏi rời nhau; chỉ câu thứ hai cần tập test:

- **(a) Cơ chế** — giảm khung có thêm gì so với tổng hợp phẳng trên **cùng** posterior của **cùng**
  bộ phát hiện? Trả lời được trọn vẹn trên val slice 83 clip dùng chung giữa mọi member 4 lớp.
- **(b) Độ lớn** — hệ hợp nhất có đạt 95.81% trên 64 clip test? Cần test.

Báo cáo này trả lời **(a)** và chốt trọng số hợp nhất chọn-trên-val cho **(b)** mà không chấm test.

## 1. Cơ chế giảm khung: hiệu lực ≈ 0, trên cả ba bộ phát hiện độc lập

`det-gateQ` (giữ top-q khung theo bằng chứng KAT, `U.sum(-1)`) so với `det-mean`/`-max`/`-pair`
(tổng hợp phẳng), cùng val slice, cùng posterior:

| Bộ phát hiện | gate tốt nhất | phẳng tốt nhất | **Δ (gate − phẳng)** | McNemar trên val | [17] gán cho cơ chế này |
|---|---|---|---|---|---|
| `beh_ctx` (mAP50 0.806) | gate50 **85.91%** | mean 85.78% | **+0.13 pp** | **p = 1.0000** | **+10.03 pp** |
| `beh_ctx` (lượt 06/09) | gate50 **85.91%** | mean 85.78% | **+0.13 pp** | **p = 1.0000** | **+10.03 pp** |
| `beh_tight` (mAP50 0.791) | gate75 **87.93%** | max **88.05%** | **−0.13 pp** | **p = 1.0000** | **+10.03 pp** |

Ba lần đo độc lập, hai hướng dấu, không lần nào quá ±0.13 pp, McNemar p = 1.0 (số clip bất đồng
bằng nhau hai phía). **Tầng giảm khung của ta không tái dựng cơ chế +10 pp của [17].** Lượt 06/09
cho **đúng** cùng số như lượt 11/10 ⇒ tất định, không phải nhiễu seed.

## 2. Vậy +10 pp từ đâu? Từ việc **dùng bộ phát hiện**, không từ việc giảm khung

Cùng val slice 83 clip, macro (trung bình recall lớp):

| Hệ | val macro | val top-1 | Δ so ConvLSTM khung thô |
|---|---|---|---|
| **ConvLSTM khung thô** (cpu, G21) | **79.17%** | 75.90% | — |
| ConvLSTM khung thô (mps, 06/09) | 76.39% | — | −2.78 pp |
| đường bộ-phát-hiện, member đơn tốt nhất (`detT-max`) | **88.05%** | — | **+8.88 pp** |
| hợp nhất Caruana, chỉ member `detT-*` | 87.30% | 84.34% | +8.13 pp |
| **hợp nhất Caruana, ConvLSTM ∪ `detT-*`** (kiến trúc [17]) | **88.05%** | **85.54%** | **+8.88 pp** |
| hợp nhất Caruana, ConvLSTM ∪ `detctx-*` | 85.15% | 81.93% | +5.98 pp |

Hai điều đọc được:

1. **Khoảng cách +8.88 pp đến hoàn toàn từ việc thay khung thô bằng posterior bộ phát hiện** — xấp
   xỉ đúng cỡ +10.03 pp mà [17] công bố, nhưng **cơ chế được họ nêu tên không phải nguyên nhân**.
2. **Hợp nhất không thêm gì trên member đơn tốt nhất**: 88.05% (hợp nhất) = 88.05% (`detT-max` một
   mình), tới hai chữ số thập phân. Với `detctx` hợp nhất còn **kém hơn** member đơn tốt nhất
   (85.15% so 85.91%). Caruana chọn trọng số trên val; nó không tạo ra tín hiệu mới.

**Bẫy trùng số, nêu để không ai nhầm:** `detctx-mean` cho val macro **85.78%**, trùng từng chữ số
với macro ConvLSTM mà [17] công bố trên test (85.78%). Hoàn toàn trùng hợp — khác tập, khác tác vụ,
khác giao thức. Không được đặt hai số này cạnh nhau như một sự tái lập.

Nghịch lý đáng ghi: `beh_tight` **mAP50 thấp hơn** (0.791 < 0.806) nhưng cho clip macro **cao hơn**
(88.05 > 85.91). mAP50 mức hộp **không đơn điệu** với macro mức clip ⇒ chọn bộ phát hiện theo mAP50
(chính là điều cổng G22 làm) **không bảo đảm** chọn được bộ tốt cho tác vụ cuối.

## 3. Mọi số ở đây đều **dưới sàn**, theo kiến tạo

`beh_ctx` 0.806 và `beh_tight` 0.791 đều **dưới sàn 0.85** của G22. Trước đây điều này chỉ được
bảo đảm bằng **một dòng chú thích** — không thứ gì ở hạ nguồn đọc được. Nay
`behaviour_clip.detector_gate_status()` đọc đỉnh mAP50 **từ chính `results.csv`** của lượt huấn
luyện (không nhập tay) và **dập nhãn vào từng tạo tác member**:

```
{'name': 'detctx-gate50', 'detector_map50_peak': 0.80585,
 'gate_status': 'BELOW FLOOR - EXPLORATORY ONLY, never cite as a gated result'}
```

## 4. Điều **không** được kết luận

- **Không** kết luận gì về độ lớn trên test. `Pt` của mọi hệ hợp nhất đã lưu nhưng **chưa chấm**;
  việc chấm cần lần chạm còn lại **và** một luật quyết định viết trước khi tiêu.
- **Không** kết luận "cơ chế của [17] vô dụng". Kết luận đúng: *tầng giảm khung **của ta**, trên
  posterior của bộ phát hiện **của ta**, không có hiệu lực đo được*. Hai khả năng còn mở: chẩn
  đoán §7.6 sai, hoặc hai tầng khác nhau ở chỗ chưa định vị được.
- **Không** so sánh với "95.66% mAP" — số đó **không tồn tại** trong `paper.txt` (0 hit khi grep).
- Val slice 83 clip đã được G21 chứng minh là **mù** với test 64 clip (val +2.78 pp ↔ test −3 clip).
  Mọi xếp hạng ở §2 là xếp hạng **trên val**, không phải dự báo test.

## 5. Sửa mã sau review (không số nào đã báo cáo bị đổi)

`code-reviewer` chạy lại toàn bộ scorer và xác nhận đầu ra **trùng từng byte** với
`ket-qua-g21-g25.json` đã commit; không lỗi sống. Hai lỗi **ngủ** đã sửa:

1. `convlstm.py` — `task='8'` cắt softmax 8 chiều xuống 4 cột **không chuẩn hoá lại** ⇒ hàng không
   còn tổng 1, vi phạm tiền điều kiện của `metrics.report`. `argmax` sống sót nhưng Brier/ECE/NLL
   và **mọi xếp hạng per-class cho ROC/PR** thì không. Đã thêm `P / P.sum(1, keepdim=True)`.
   Không tạo tác `task='8'` nào tồn tại ⇒ không số nào đã báo cáo bị ảnh hưởng.
2. `run_convlstm_paper_baseline.py` — sàn/trần NP-125 bị **đóng cứng** 64/125 và 124/125. Nay suy
   ra từ chính nhãn test và các nhóm `opposite` thực có trong split. Kiểm chứng hai nhánh:
   `keep` → 0.5120 / 0.9920 (trùng số cũ), `drop_conflict_test` → **0.5250 / 1.0000** (đúng như
   `dedupe.py` đặc tả — bản cũ sẽ báo sai trần 0.992 cho split 120 clip).
3. Đã **khai báo** việc thay nguồn đối chứng `clip_view_map.json` → `clip_scene_map_691.json` vào
   `phan-quyet-2026-10-11-g21-g25.md` §2 (lý do: bản tiền đăng ký là KMeans trên **cùng** đặc
   trưng dùng để phân lớp ⇒ vòng tròn, và chỉ phủ 516 clip KAT nên không phục vụ được test 125).

## 6. Câu hỏi chưa giải quyết

1. Lần chạm còn lại tiêu cho cái gì: `fusion_g23_convlstm_cpu+detT` (val 88.05%), hay chờ bộ phát
   hiện vượt sàn trên CUDA? Chấm bây giờ ⇒ số tiêu đề mang nhãn "bộ phát hiện dưới sàn" vĩnh viễn.
2. mAP50 không đơn điệu với clip macro (`beh_tight` kém hộp, hơn clip). Vậy **cổng G22 có còn là
   cổng đúng**? Nó gác một đại lượng không dự báo được đại lượng ta cần.
3. Hợp nhất không thắng member đơn tốt nhất trên val slice 83 clip — do Caruana thiếu lực trên n
   nhỏ, hay do các member tương quan quá cao? Chưa đo tương quan lỗi giữa member.
4. `--task 8` có nằm trong pha nào của kế hoạch này không? Không tìm thấy tham chiếu; nếu không,
   nên cân nhắc bỏ khỏi `choices` thay vì để làm bẫy.
