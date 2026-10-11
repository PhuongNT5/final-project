# Pha 05 — Đường ống hợp nhất dưới hai giao thức, độ lạm phát làm sản phẩm (G24)

## Context Links

- Đối chứng: **G24** — đo, **không** phán quyết đạt/trượt (pha 01)
- Giao thức (a) bài báo 452/64: `run_blind_eval.py:169-173` (val-frac 0.2, split-seed 0) — gần nhất hiện có
- Giao thức (b) LCO-691: `run_lco_eval.py` — 8 nhóm bối cảnh, gộp 516 clip 4 lớp
- Đối chứng cảnh: `run_full_eval.py:30-43` (`scene_only_pred`), `sweep_results/zone_prior/clip_view_map.json`,
  `sweep_results/zone_prior/clip_scene_map.json`, `unsafenet/scene_confound.py`
- Lạm phát giao thức, tiền lệ làn A: G12 (`chuong-06-danh-gia.txt:1582-1650`, +15.4 pp, CI [+10.6, +21.5])
- Nhóm bối cảnh là **đại diện** góc quay, không phải nhãn camera: `chuong-06-danh-gia.txt:1247-1249`
- Ngân sách chạm: `chuong-04-chi-tiet-thuc-nghiem.txt:949-973`
- Câu hỏi mở Q3 của chapter-07: `report/chapter-07-...md:404-407`

## Overview

- **Priority:** P1 — pha sinh ra sản phẩm báo cáo được của kế hoạch.
- **Status:** pending
- Đo đường ống hợp nhất (ConvLSTM + tầng giảm khung) dưới **cả hai** giao thức, giữ mọi thứ khác cố
  định, và lấy **hiệu** giữa chúng làm đại lượng công bố. Đây là pha **tiêu một lần chạm test duy nhất**.

## Key Insights

1. **Hai giao thức, không phải một, và báo cáo riêng.**
   (a) 452/64 cùng-phân-bố của [17] — **chỉ** giao thức này làm số của ta so được với 57/64 và 62/64.
   (b) LCO-691 của luận văn — giao thức đo **lạm phát**.
   Đại lượng báo cáo được là **hiệu** (a) − (b) với mọi thứ khác cố định, không phải (a) một mình.
2. **Chỉ (a) tiêu lần chạm.** LCO-691 gộp 516 clip (452 train + 64 test) và báo cáo ngoài-fold; mọi
   số Chương 6 đã là số ngoài-fold (`chuong-04:970-972`). Chỉ phép đo trên tập test 64 clip **giữ
   riêng** theo giao thức (a) mới tiêu một lần chạm. ⇒ Mọi arm của (a) phải chạy trong **một** lần
   suy luận gộp, danh sách arm chốt từ pha 01.
3. **`run_blind_eval.py` không phải giao thức của [17]**, nó là giao thức *blind held-out* của luận
   văn trên cùng cách chia 452/64 (cắt val 20% từ train). Nó là thứ gần nhất. Khác biệt với [17]:
   [17] không khai báo tập val tách khỏi 452 (xem pha 02 câu hỏi #2). Phải nêu sai lệch này, không
   gọi `run_blind_eval.py` là "giao thức bài báo".
4. **Đối chứng cảnh **thoái hoá** dưới LCO và phải xử lý riêng.** Dưới k-fold, "đa số theo view" là
   đường sàn có thông tin: **77.1% ± 2.5** (`chuong-06-danh-gia.txt:1384`). Dưới leave-context-out,
   nhóm giữ lại **không có clip train nào** ⇒ đường sàn suy biến thành hằng số toàn cục (**16.5%**,
   `chuong-06:1394-1396`). Vì vậy:
   - giao thức (a) → đối chứng là **scene-only 52/64 = 81.2%** (`chuong-06:1342`) và whole-frame 53/64 = 82.8%;
   - giao thức (b) → đối chứng là **hằng số toàn cục**, và phải nói rõ rằng đối chứng đa-số-theo-view
     *không áp dụng được*. Dùng sai đối chứng ở (b) sẽ làm độ lạm phát trông lớn hơn thực tế.
5. **Nhóm bối cảnh là KMeans(8) đại diện, không phải camera thật**; đồng thuận view trong nhóm chỉ
   **81.7%** (`chuong-06:1249`). Mọi phát biểu "leave-camera-out" phải viết là "leave-context-out
   trên đại diện góc quay".
6. **Bootstrap cụm trên 8 nhóm thiếu lực — phát biểu trước khi đọc bảng.** Theo khuôn mẫu §6.1.3:
   với n = 8 đơn vị lấy mẫu, khoảng tin cậy theo cụm sẽ rất rộng; **rất có thể mọi biên dưới ~10 pp
   đều cho CI chứa 0**. Hệ quả đã chốt: biên nào có CI chứa 0 thì không được chứng nhận và không
   được làm số tiêu đề, kể cả độ lạm phát. Khoảng ở mức clip in kèm `WRONG UNIT` và không dùng.
7. **Clip mỗi nhóm rất lệch**: 68, 137, 32, 70, 18, 103, 64, 24 (tổng 516). Nhóm nhỏ nhất 18 clip.
   Bootstrap cụm sẽ bị chi phối bởi nhóm 137 và 103. Phải báo cáo phân bố này cùng CI.
8. **Ba trên tám view đơn-lớp** ⇒ một phần kết quả giải thích được bằng cảnh, không bằng hành vi. Vì
   vậy mọi số giao thức (a) phải đặt cạnh đối chứng đa-số-theo-view, nếu không nó không đọc được.
9. **Hai loại macro, luôn dán nhãn riêng.** `macro_paper` = trung bình sensitivity 4 lớp (đúng cách
   của [17]; đã có ở `unsafenet/metrics.py::macro_accuracy`) và `macro_f1` (sklearn). Balanced
   accuracy **là** `macro_paper` cho bài toán đơn nhãn (`unsafenet/metrics.py:34-36`) — nêu rõ để
   không ai tưởng đó là ba số.

## Requirements

**Chức năng**
- Chạy đường ống hợp nhất dưới (a) và (b) với **cùng** trọng số, cùng seed, cùng tầng giảm khung.
- Giao thức (a): một lần suy luận gộp trên 64 clip, bao mọi arm trong danh sách chốt ở pha 01.
- Giao thức (b): LCO-691, dự đoán ngoài-fold gộp, 8 nhóm.
- Đối chứng kèm theo, đúng loại cho đúng giao thức (Key Insight 4).
- Độ đo: balanced accuracy là chính trên dự đoán ngoài-fold gộp; `macro_paper`; `macro_f1`;
  recall/precision lớp unsafe; PR-AUC; ROC-AUC. Dán nhãn riêng.
- Arm **S3** trả lời Q3 của chapter-07: bộ phát hiện **hành vi** (pha 03) thay bộ phát hiện **thực
  thể** dưới LCO, cùng ngân sách 2,262 khung ⇒ chặn trần cảnh.
- Xuất `*_oof.json` được theo dõi cho mọi arm (theo lệ `.gitignore:15-25`).

**Phi chức năng**
- Tiêu **đúng 1** lần chạm test. Sau pha này ngân sách còn 1.
- Không dừng sớm theo test ở bất kỳ arm nào (`chuong-06:1567-1570` — không thêm đường nhiễm thứ ba).
- Không trộn làn: mọi số ở đây là làn C (`chuong-06:82-85`).

## Architecture

```
            ConvLSTM (cấu hình thắng G21)  +  tầng giảm khung (cấu hình chốt pha 04)
                                   │
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
 (a) giao thức [17]: 452 train / 64 test        (b) LCO-691: 8 nhóm bối cảnh
     val 20% cắt từ train                           gộp 516 clip, ngoài-fold
     MỘT lần suy luận gộp = 1 lần chạm              KHÔNG tiêu lần chạm
            │                                             │
            ▼                                             ▼
 đối chứng: scene-only 52/64, whole-frame 53/64   đối chứng: hằng số toàn cục
            │                                             │     (đa-số-theo-view KHÔNG áp dụng)
            └──────────────────────┬──────────────────────┘
                                   ▼
                  G24 = macro(a) − macro(b)   + CI bootstrap cụm (8 nhóm, 5000 mẫu)
                                   │
                                   ▼  arm S3: bộ phát hiện hành vi dưới LCO -> trần cảnh (Q3)
```

## Related Code Files

**Sửa**
- `run_lco_eval.py` — nhận đường ống hợp nhất làm một arm; xuất JSON (hiện chỉ in stdout, không ghi tệp)
- `run_blind_eval.py` — nhận đường ống hợp nhất làm một arm
- `unsafenet/metrics_stats.py` — thêm **một** hàm bootstrap cụm tập trung (hiện cài rải trong
  `run_kfold_eval.py:53-71` và `eval_utils/paired_tests.py` ⇒ vi phạm DRY)

**Tạo**
- `plans/.../reports/g24-do-lam-phat-giao-thuc.md`
- `plans/.../reports/s3-tran-canh-duoi-lco.md` (trả lời Q3 chapter-07)

**Không sửa**
- `sweep_results/zone_prior/clip_view_map.json`, `clip_scene_map.json` — bản đồ nhóm đã cố định;
  đổi nó là đổi giao thức
- `unsafenet/config.py::PAPER`

## Implementation Steps

1. Kiểm G21 đã phán quyết và G23 đã chốt cấu hình. Nếu G22 trượt sàn: pha này thu lại còn arm
   **không-bbox** dưới hai giao thức (vẫn đo được độ lạm phát).
2. Tập trung bootstrap cụm vào một hàm ở `unsafenet/metrics_stats.py`; đơn vị lấy mẫu = **nhóm**,
   `n=5000`; đồng thời in CI mức clip kèm nhãn `WRONG UNIT`.
3. Chạy **(b) LCO-691** trước — không tiêu lần chạm, nên nó là nơi hoàn thiện mã và sửa lỗi.
   Xuất `*_oof.json`. Báo cáo kèm phân bố clip theo nhóm (68/137/32/70/18/103/64/24).
4. Chạy arm **S3** dưới LCO: thay bộ phát hiện thực thể bằng bộ phát hiện hành vi của pha 03, cùng
   ngân sách khung. Đây là đối chứng mà Q3 của chapter-07 đòi.
5. Chạy đối chứng cảnh cho (b): hằng số toàn cục; ghi rõ đa-số-theo-view không áp dụng được dưới LCO.
6. **Chốt danh sách arm cuối cùng** cho (a) và đối chiếu với mục khai báo Section B ở
   `reports/test-touch-log.md`. Nếu lệch, cập nhật mục khai báo **trước** khi chạy.
7. **Tiêu lần chạm**: chạy (a) một lần, mọi arm trong một lần suy luận. Ghi kết quả vào ô kết quả
   của mục Section B.
8. Chạy đối chứng cảnh cho (a): scene-only (đa số theo view, học từ train) và whole-frame.
9. Tính **G24** = `macro_paper`(a) − `macro_paper`(b), kèm CI bootstrap cụm 8 nhóm. Tính cả với
   balanced accuracy. Báo cáo theo đúng cách đã đăng ký ở pha 01.
10. So (a) với 57/64 và 62/64 bằng `feasible_discordant_splits` / `paired_vs_aggregate_baseline`
    (dự đoán theo clip của [17] không công bố). **Không** dùng z hai tỉ lệ (`chuong-06:1553-1556`).
11. Viết hai báo cáo; nêu rõ đâu là số, đâu là hiệu, đâu là đối chứng, và biên nào có CI chứa 0.

## Todo List

- [ ] Kiểm G21 phán quyết, G23 chốt cấu hình, G22 trạng thái
- [ ] Hàm bootstrap cụm tập trung ở `metrics_stats.py` (bỏ cài rải, DRY)
- [ ] Chạy (b) LCO-691 + xuất `*_oof.json` + phân bố clip theo nhóm
- [ ] Arm S3 dưới LCO (trả lời Q3 chapter-07)
- [ ] Đối chứng cảnh cho (b) = hằng số toàn cục, ghi rõ đa-số-theo-view không áp dụng
- [ ] Chốt danh sách arm của (a), đồng bộ mục Section B **trước khi chạy**
- [ ] **Tiêu 1 lần chạm**: chạy (a) gộp một lần, điền ô kết quả Section B
- [ ] Đối chứng cảnh cho (a): scene-only + whole-frame
- [ ] G24 = hiệu hai giao thức + CI bootstrap cụm 8 nhóm (5000)
- [ ] So với 57/64 và 62/64 bằng biên McNemar khả thi (không z hai tỉ lệ)
- [ ] `reports/g24-do-lam-phat-giao-thuc.md`
- [ ] `reports/s3-tran-canh-duoi-lco.md`

## Success Criteria

- Hai giao thức đo trên **cùng** trọng số/seed/tầng giảm khung — hiệu là hiệu giao thức, không phải
  hiệu mô hình.
- Đúng **1** lần chạm tiêu, có mục Section B điền đủ trước-và-sau.
- Mọi arm có `*_oof.json` được git theo dõi.
- Mỗi giao thức đi kèm **đúng loại** đối chứng cảnh của nó; sự thoái hoá dưới LCO được nêu, không bị
  che bằng cách mượn số 77.1% của k-fold.
- `macro_paper`, `macro_f1`, balanced accuracy dán nhãn riêng; nêu rõ balanced accuracy = `macro_paper`.
- CI bootstrap cụm 8 nhóm có, và biên nào chứa 0 được nói thẳng là không chứng nhận được.
- Arm S3 trả lời Q3 của chapter-07 bằng một số, hoặc nói rõ vì sao chưa trả lời được.
- Không một câu nào đặt số làn C cạnh số làn A/B.

## Risk Assessment

| Rủi ro | Giảm thiểu |
|---|---|
| **CI cụm trên 8 nhóm chứa 0 với cả độ lạm phát** ⇒ sản phẩm chính không chứng nhận được | Nêu trước (Key Insight 6), đúng khuôn §6.1.3. Khi đó công bố điểm ước lượng kèm phát biểu "không chứng nhận được ở độ phân giải cụm" và **không** dùng làm số tiêu đề. Đây là kết cục đã lường, không phải thất bại. |
| Tiêu quá 1 lần chạm vì chốt arm sai | Bước 6 bắt buộc đồng bộ danh sách arm với mục Section B **trước** khi chạy; (b) chạy trước để hoàn thiện mã. |
| Dùng đối chứng đa-số-theo-view dưới LCO ⇒ phóng đại độ lạm phát | Key Insight 4 + Success Criteria; đối chứng LCO là hằng số toàn cục. |
| Nhóm 18 clip và nhóm 137 clip ⇒ bootstrap bị chi phối | Báo cáo phân bố clip theo nhóm cùng CI; cân nhắc in thêm CI bỏ-một-nhóm (jackknife) làm chẩn đoán. |
| `run_blind_eval.py` bị gọi là "giao thức bài báo" | Nêu sai lệch: [17] không khai báo tập val tách khỏi 452. |
| Nhóm bối cảnh là đại diện (đồng thuận 81.7%), không phải camera | Luôn viết "leave-context-out trên đại diện góc quay". |
| Lạm phát do chọn lọc vì (b) chạy nhiều lần | (b) không tiêu lần chạm nhưng vẫn là lựa chọn; mọi tham số đã chốt ở pha 04 trên val, không chỉnh ở pha này. |

## Security Considerations

- `*_oof.json` chỉ chứa tên clip, nhãn, xác suất — không ảnh. An toàn để theo dõi.
- Không commit `clip_view_map.json`/`clip_scene_map.json` phiên bản mới; bản đồ nhóm là phần của
  giao thức đã cố định.
- Video test chỉ được đọc trong lần chạm đã khai báo; không sao chép ra ngoài repo.

## Next Steps

- Phụ thuộc vào: pha 02 (G21), pha 03 (G22), pha 04 (G23 cấu hình).
- Chặn: pha 06 (phán quyết thống kê và sửa chữ).
- Theo sau: pha 06.

## Unresolved Questions

1. Nếu CI cụm trên 8 nhóm chứa 0 với **mọi** biên, kế hoạch còn sản phẩm công bố được không — hay
   kết luận duy nhất là "làn C không đủ nhóm để chứng nhận bất kỳ biên nào"? Nếu vậy, phát biểu đó
   tự nó có phải một kết quả đủ?
2. Arm S3 có đủ trả lời Q3 của chapter-07 không? Q3 hỏi 2,262 khung thực thể so với 2,262 khung hành
   vi có phải head-to-head công bằng. Nhưng bộ dữ liệu hành vi chỉ có **2,244** khung (xem pha 03)
   ⇒ không khớp đúng ngân sách. Chênh 18 khung có làm head-to-head mất hiệu lực?
3. Giao thức (a) nên dùng cách cắt val nào để gần [17] nhất: 20% như `run_blind_eval.py`, 18% như
   `blind_split`, hay không cắt (huấn luyện đủ 452, dừng theo số epoch cố định)? Phương án thứ ba
   gần [17] nhất về cỡ dữ liệu nhưng không có cơ chế dừng sớm trung thực.
4. Độ lạm phát có so được với G12 của làn A (+15.4 pp) không? Lệnh cấm trộn làn
   (`chuong-06:82-85`) nói **không** — nhưng cả hai đều là *lạm phát giao thức*, không phải điểm
   chính xác. Có ngoại lệ cho đại lượng bậc hai không?
5. Nếu G22 trượt sàn, độ lạm phát đo trên arm không-bbox có còn là sản phẩm của **kế hoạch này** hay
   nó chỉ lặp lại một phép đo đã có của luận văn?
