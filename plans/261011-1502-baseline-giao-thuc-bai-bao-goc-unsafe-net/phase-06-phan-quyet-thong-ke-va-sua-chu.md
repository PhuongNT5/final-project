# Pha 06 — Phán quyết thống kê, ghi sổ, và sửa chữ

## Context Links

- Luật phán quyết cổng: `report/vn/chuong-06-danh-gia.txt:139-155`
- Bảng cổng G10-G20: `report/vn/chuong-06-danh-gia.txt:1582-1650`
- Cấm z hai tỉ lệ cho dự đoán theo cặp: `report/vn/chuong-06-danh-gia.txt:1553-1556`
- Năm con số mơ hồ + nhãn làn C: `report/vn/chuong-06-danh-gia.txt:123-135`
- §6.12.5 hiện tại: `report/vn/chuong-06-danh-gia.txt:1517-1580` (câu phải sửa: `:1525-1527`)
- Mốc đóng băng số: `report/vn/00-trang-thai-bien-soan.txt:11`; tiền lệ ngoại lệ G20: `:13-17`
- Thống kê: `unsafenet/metrics_stats.py`, `eval_utils/paired_tests.py`
- Hằng số so sánh đã sửa: `run_ablation_significance.py:155` (57, không phải 55)
- Q3 chapter-07: `report/chapter-07-multi-entity-and-scene-robust-evaluation.md:404-407`
- Sổ chạm test: `reports/test-touch-log.md`

## Overview

- **Priority:** P1 — pha kết, biến số thành phán quyết và chữ.
- **Status:** pending
- Phán quyết G21-G23 theo đúng luật đã chốt, ghi G24 như đối chứng, điền sổ chạm test, và sửa **chữ**
  trong `report/vn/` cho khớp điều đã tìm thấy — **không thêm giá trị số mới** vào luận văn.

## Key Insights

1. **Hai điều kiện, không phải một.** Cổng đạt chỉ khi **biên ≥ biên tiền đăng ký** *và* **CI cụm
   loại trừ 0**. Biên đạt mà CI chứa 0 ⇒ **trượt**, đọc theo hướng bảo thủ (`chuong-06:143-146`).
2. **CI cụm thắng McNemar khi hai kết luận trái nhau** (`chuong-06:147-149`) — McNemar bỏ cấu trúc
   cụm nên quá dễ dãi. Phải in cả hai rồi áp luật, không chọn cái thuận.
3. **Một cổng đặt tên một mô hình; thay mô hình để lấy biên lớn hơn không đổi trượt thành đạt**
   (`chuong-06:150-152`). Các arm chẩn đoán của pha 02 không được dùng để phán quyết G21.
4. **z hai tỉ lệ bị cấm cho dự đoán theo cặp.** Luận văn đã **rút** một phép z cho cặp 56/64 so với
   57/64 (`chuong-06:1553-1556`). Vì dự đoán theo clip của [17] không công bố, công cụ đúng là
   `feasible_discordant_splits` → `mcnemar_bound` (p nhỏ nhất) và `_max_p_over_discordant_splits`
   (p lớn nhất), báo cáo thành **khoảng** [min_p, max_p].
5. **Ràng buộc "cả hai đều đúng" đã được sửa 29/09** trong `feasible_discordant_splits` (thêm
   `b ≤ correct_a`, `c ≤ correct_b`). Trước đó biên bị phóng đại và từng đảo kết luận. Phải xác nhận
   đang dùng bản đã sửa, nếu không mọi khoảng p của pha này sai.
6. **Nadeau-Bengio bắt buộc cho k-fold.** Hiệu chỉnh cho tập huấn luyện chồng lấn nâng sai số chuẩn
   lên **2.12 pp**, hệ số 3.7×; không hiệu chỉnh thì p = 0.0000 thay vì p = 0.0600
   (`chuong-06:1382-1410`). Arm H1 của pha 02 và mọi so sánh k-fold phải qua hàm này.
7. **Mốc đóng băng chỉ cho sửa chữ.** Sau 05/10/2026 không thêm số vào `report/vn/`
   (`00-trang-thai-bien-soan.txt:11`). Tiền lệ G20 (chạy 10/10) được phép vì tiền đăng ký kèm điều
   kiện *giữ nguyên mọi số tiêu đề và mọi phán quyết cổng*; tổng vẫn là **8 cổng, 2 đạt, 6 trượt**.
   Kế hoạch này nhận cùng điều kiện ⇒ **G21-G24 không được vào bảng 8 cổng**, chúng là mục sau-đóng-băng.
8. **Câu phải sửa là `:1525-1527`**, hiện đọc: *"đó là cấu hình có thể tái dựng từ các bảng của họ:
   **57/64**, macro 85.78%."* Nếu G21 trượt theo nhánh "không tái lập được", cụm *"có thể tái dựng
   từ các bảng của họ"* trở thành gây hiểu sai — nó gợi ý một việc tái dựng đã làm. Sửa thành một
   mô tả nói rõ: số **đọc từ bảng của họ**, và một lần thử tái lập độc lập đã **không** đạt tới nó.
   Đây là **sửa chữ**: không giá trị số nào đổi.
9. **Nhãn làn C phải được mang theo nguyên vẹn** (`chuong-06:127-135`): 56/64 = 87.5% là **chặn
   trên** (chọn sau khi xem test); 54/64 = 84.4% top-1 / 88.4% macro là **tiêu đề chọn trên val**;
   53/64 = 85.4% macro là **số trung thực nhất**; trần 63/64. Số mới của kế hoạch này **không được**
   chiếm một trong các nhãn đó.
10. **75.64% phải được thêm vào danh sách số cần nhãn.** Nó chưa xuất hiện ở đâu trong `report/vn/`,
    và 50/64 = 78.12% thì đã có ở `docs/system-architecture.md`. Nếu không dán nhãn, nó trở thành
    con số mơ hồ **thứ sáu** của §6.1.4.

## Requirements

**Chức năng**
- Phán quyết G21, G22, G23 theo đúng bảng luật pha 01; ghi G24 như đối chứng không phán quyết.
- Với mỗi cổng: biên, CI bootstrap cụm (8 nhóm, 5000), McNemar/khoảng biên khả thi, và phán quyết
  theo luật hai điều kiện.
- Điền ô kết quả mục Section B của `reports/test-touch-log.md`; cập nhật "số lần đã tiêu" = 1.
- Sửa **chữ** `report/vn/chuong-06-danh-gia.txt:1525-1527` cho khớp phán quyết G21.
- Thêm `75.64%` (và `50/64`) vào §6.1.4 như số cần nhãn — **sửa chữ**, thêm nhãn, không thêm số đo mới.
- Trả lời Q3 của chapter-07 bằng kết quả arm S3, hoặc ghi rõ vì sao chưa trả lời được.
- Cập nhật `docs/` theo `documentation-management.md`: `codebase-summary.md`,
  `system-architecture.md`, `project-changelog.md` (stage mới, module mới).

**Phi chức năng**
- `report/vn/` chỉ nhận sửa chữ. Mọi giá trị số mới ở lại `plans/261011-1502-.../`.
- Lint trước commit; test trước push (`development-rules.md`).
- Conventional commits, không tham chiếu AI.
- Bảng 8 cổng của Chương 6 **không đổi**: vẫn 2 đạt, 6 trượt.

## Architecture

```
JSON ngoài-fold của pha 02/04/05  (được git theo dõi)
        │
        ▼ unsafenet/metrics_stats.py (bootstrap cụm 8 nhóm, n=5000)
        │ eval_utils/paired_tests.py (McNemar chính xác; biên khả thi; Nadeau-Bengio)
        ▼
luật hai điều kiện (biên ≥ tiền đăng ký) AND (CI loại trừ 0)
   CI cụm thắng McNemar khi trái nhau
        │
        ├──► plans/.../reports/phan-quyet-cong-g21-g25.md     (số mới sống ở đây)
        ├──► reports/test-touch-log.md                        (điền ô kết quả, đã tiêu = 1)
        ├──► report/vn/chuong-06-danh-gia.txt                 (CHỈ sửa chữ: :1525-1527, §6.1.4)
        ├──► report/chapter-07-...md                          (trả lời Q3)
        └──► docs/{codebase-summary,system-architecture,project-changelog}.md
```

## Related Code Files

**Sửa**
- `reports/test-touch-log.md` — điền ô kết quả Section B; "số lần đã tiêu" 0 → 1
- `report/vn/chuong-04-chi-tiet-thuc-nghiem.txt:956` — "Số lần đã tiêu **0**" → **1** (sửa một số
  đếm thủ tục, không phải một số đo; nêu rõ trong commit)
- `report/vn/chuong-06-danh-gia.txt` — `:1525-1527` (nhãn mốc 85.78%); §6.1.4 (thêm 75.64% / 50/64)
- `report/chapter-07-multi-entity-and-scene-robust-evaluation.md` — trả lời Q3 ở `:404-407`
- `docs/codebase-summary.md`, `docs/system-architecture.md` — stage `convlstm`/`framereduce`,
  module `unsafenet/clip_split.py`, `unsafenet/convlstm_report.py`
- `docs/project-changelog.md` — mục cho kế hoạch này

**Tạo**
- `plans/.../reports/phan-quyet-cong-g21-g25.md`

**Không sửa**
- `report/vn/chuong-06-danh-gia.txt:1582-1650` — bảng 8 cổng, điều kiện G20 cấm đổi
- Mọi giá trị số tiêu đề của luận văn
- `unsafenet/config.py::PAPER`

## Implementation Steps

1. Xác nhận `feasible_discordant_splits` đang là bản sau 29/09 (có ràng buộc `b ≤ correct_a`,
   `c ≤ correct_b`). Nếu không, dừng và sửa trước — mọi khoảng p phụ thuộc vào nó.
2. Tính, cho mỗi cổng: biên; CI bootstrap cụm 8 nhóm 5000 mẫu; CI mức clip kèm nhãn `WRONG UNIT`;
   McNemar chính xác khi có dự đoán theo cặp; khoảng [min_p, max_p] khi so với [17].
3. Áp Nadeau-Bengio cho mọi so sánh k-fold (arm H1 pha 02).
4. Áp **luật hai điều kiện** cho G21, G22, G23. Khi McNemar và CI cụm trái nhau, lấy CI cụm.
5. Viết `reports/phan-quyet-cong-g21-g25.md`: với mỗi cổng — giả thuyết, mô hình được đặt tên, biên
   tiền đăng ký, biên đo được, CI, p, phán quyết, và **nhánh kết luận nào của pha 01 được kích hoạt**.
   Trượt viết chi tiết bằng đạt (`chuong-06:153-155`).
6. Điền ô kết quả mục Section B của `reports/test-touch-log.md`; đổi "đã tiêu" thành 1; cập nhật
   `chuong-04:956` tương ứng.
7. **Sửa chữ `chuong-06-danh-gia.txt:1525-1527`** theo phán quyết G21:
   - G21 **đạt** → giữ nhãn hiện tại, thêm một câu rằng một lần tái lập độc lập đã đạt tới mốc, và
     trỏ tới tạo tác trong `plans/261011-1502-.../`.
   - G21 **mơ hồ hoặc trượt** → đổi *"cấu hình có thể tái dựng từ các bảng của họ"* thành một mô tả
     nói rõ: giá trị **đọc ra từ Table 5 của họ**, chưa được tái lập độc lập, và một lần thử tái lập
     sau mốc đóng băng đã không đạt tới nó. Không nêu giá trị mới trong câu đó; trỏ tới thư mục kế hoạch.
8. Thêm `75.64%` và `50/64` vào §6.1.4 kèm nhãn ("macro của một lần thử tái lập ConvLSTM nội bộ, làn
   C, không phải số của [17], không phải số tiêu đề của luận văn") để chúng không thành số mơ hồ thứ sáu.
9. Trả lời Q3 của chapter-07 từ arm S3 (pha 05); nếu chênh 18 khung làm head-to-head mất hiệu lực,
   ghi đúng như vậy.
10. Cập nhật `docs/` theo `documentation-management.md`. Nêu `Docs impact:` tường minh.
11. Lint; chạy `pytest` (có `pytest.ini`, `tests/`); commit từng nhóm thay đổi riêng: (i) thống kê +
    báo cáo phán quyết, (ii) sổ chạm test, (iii) sửa chữ `report/vn/`, (iv) `docs/`.

## Todo List

- [ ] Xác nhận `feasible_discordant_splits` là bản sau 29/09
- [ ] Bootstrap cụm 8 nhóm 5000 cho mọi cổng + CI clip kèm `WRONG UNIT`
- [ ] McNemar chính xác / khoảng [min_p, max_p] so với 57/64 và 62/64
- [ ] Nadeau-Bengio cho so sánh k-fold (arm H1)
- [ ] Áp luật hai điều kiện; CI cụm thắng McNemar khi trái nhau
- [ ] `reports/phan-quyet-cong-g21-g25.md` (trượt chi tiết bằng đạt)
- [ ] Điền Section B + "đã tiêu" = 1 + `chuong-04:956`
- [ ] Sửa chữ `chuong-06-danh-gia.txt:1525-1527` theo phán quyết G21
- [ ] Thêm nhãn cho 75.64% / 50/64 vào §6.1.4
- [ ] Trả lời Q3 chapter-07 từ arm S3
- [ ] Cập nhật `docs/` + nêu `Docs impact:`
- [ ] Lint + pytest + 4 commit riêng

## Success Criteria

- Mỗi cổng G21-G23 có phán quyết theo **luật hai điều kiện**, với cả biên và CI cụm được in.
- Không một phép z hai tỉ lệ nào xuất hiện cho dự đoán theo cặp.
- So với [17] dùng khoảng biên khả thi, không dùng điểm p đơn.
- Bảng 8 cổng của Chương 6 **không đổi**; G21-G24 là mục sau-đóng-băng.
- `report/vn/` chỉ nhận sửa chữ; không một giá trị số đo mới nào được thêm.
- Câu `:1525` nói đúng điều đã tìm thấy về mốc 85.78%.
- 75.64% và 50/64 có nhãn, không thành số mơ hồ thứ sáu.
- Q3 chapter-07 được trả lời hoặc được nêu rõ vì sao chưa.
- Lint sạch, `pytest` xanh trước push.

## Risk Assessment

| Rủi ro | Giảm thiểu |
|---|---|
| **Sửa `report/vn/` sau đóng băng bị coi là vi phạm** | Chỉ sửa chữ; sao y điều kiện G20; nêu tường minh trong thông điệp commit và trong `00-trang-thai-bien-soan.txt` rằng số tiêu đề và phán quyết cổng không đổi. |
| Dùng `feasible_discordant_splits` bản trước 29/09 ⇒ khoảng p sai và có thể đảo kết luận | Bước 1 kiểm trước mọi thứ khác. |
| CI cụm chứa 0 với mọi cổng ⇒ không cổng nào đạt | Đã nêu trước ở `plan.md` rủi ro 1 và pha 05. Công bố "không chứng nhận được ở độ phân giải cụm". |
| Cám dỗ đưa số mới vào bảng 8 cổng để có một cổng đạt thứ ba | Điều kiện G20 cấm; bảng không đổi là một Success Criterion kiểm được bằng `git diff`. |
| Sửa `chuong-04:956` (0 → 1) bị coi là đổi số | Đây là một **số đếm thủ tục** của sổ chạm, không phải số đo; nêu rõ trong commit. Nếu vẫn bị coi là vi phạm, để nguyên `report/vn/` và chỉ cập nhật `reports/test-touch-log.md` — xem câu hỏi mở #2. |
| 75.64% được đọc như một số tiêu đề mới của làn C | Dán nhãn ở §6.1.4 **cùng lúc** với khi nó xuất hiện lần đầu; nhãn nêu rõ nó không phải số tiêu đề. |
| Trộn làn khi so độ lạm phát với G12 | Lệnh cấm `chuong-06:82-85`; xem pha 05 câu hỏi #4. |

## Security Considerations

- Không commit dữ liệu bí mật: kiểm `git status` trước mỗi commit (`.env`, `kaggle.json`, khoá API).
- Báo cáo chỉ chứa chỉ số và tên clip, không chứa khung ảnh nhà máy.
- `report/vn/` là tài liệu nộp; mọi sửa phải kiểm lại bằng `git diff` để không vô tình đổi số khác.

## Next Steps

- Phụ thuộc vào: pha 02, 03, 04, 05.
- Theo sau: nếu G21 trượt, mở một kế hoạch riêng cho câu hỏi "những tạo tác làn C nào khác đã huấn
   luyện trên MPS" (pha 02 câu hỏi mở #4) — ngoài phạm vi kế hoạch này.
- Ngân sách chạm test còn lại sau pha này: **1**.

## Unresolved Questions

1. G21-G24 nên đi vào đâu? Bảng 8 cổng bị khoá bởi điều kiện G20. Tạo một bảng "cổng sau đóng băng"
   riêng (như G19/G20 đã được xử lý) hay chỉ để trong thư mục kế hoạch?
2. Đổi "Số lần đã tiêu 0 → 1" ở `chuong-04:956` có phải sửa chữ hay sửa số? Nếu là sửa số thì sổ
   `reports/test-touch-log.md` và luận văn sẽ **lệch nhau** — tình trạng nào tệ hơn?
3. Nếu G21 trượt, luận văn có cần nêu rằng mốc 85.78% của [17] **có thể không tái lập được** — một
   phát biểu về công trình của người khác? Mức mạnh nào của câu là biện hộ được với n=64 và không có
   dự đoán theo clip của họ?
4. 75.64% nên dán nhãn ở §6.1.4 (số mơ hồ) hay ở §6.12.5 (so với công trình gốc)? Hai chỗ có ngữ
   nghĩa khác nhau.
5. Nếu arm S3 cho thấy trần cảnh dưới LCO cao hơn kết quả hợp nhất, kết luận là "hành vi không thêm
   gì ngoài cảnh" — điều đó có cần một cổng riêng, hay nó là một quan sát?
