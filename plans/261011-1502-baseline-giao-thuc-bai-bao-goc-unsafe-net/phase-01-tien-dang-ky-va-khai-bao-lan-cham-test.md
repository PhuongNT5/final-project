# Pha 01 — Tiền đăng ký cổng G21-G24 và khai báo lần chạm test

## Context Links

- Tiền lệ bắt buộc: commit `6df16e1` *"pre-register the backbone-ladder test and its decision rule before measuring"*
- Khuôn mẫu văn bản: `plans/261010-hoi-dong-danh-gia-khoa-hoc/tien-dang-ky-e2-thang-xuong-song.md`
- Luật phán quyết cổng: `report/vn/chuong-06-danh-gia.txt:139-155`
- Bảng cổng G10-G20: `report/vn/chuong-06-danh-gia.txt:1582-1650`
- Ngân sách chạm test: `report/vn/chuong-04-chi-tiet-thuc-nghiem.txt:949-973`; sổ `reports/test-touch-log.md`
- Mốc đóng băng số: `report/vn/00-trang-thai-bien-soan.txt:11`

## Overview

- **Priority:** P1 — chặn mọi pha khác. Không một phép đo nào của kế hoạch được chạy trước khi pha này commit.
- **Status:** pending
- Viết và commit các cổng G21-G24 cùng luật quyết định, **trước khi tồn tại bất kỳ con số nào**.
  Đồng thời khai báo lần chạm test duy nhất và ghi hồi tố lần chạm lịch sử 06/09.

## Key Insights

1. **G10-G20 đã dùng.** ID tự do tiếp theo là **G21**. (`chuong-06-danh-gia.txt:1582-1650`; G19/G20 là
   đối chứng không phán quyết.)
2. **"Gate 1" của bộ phát hiện không thuộc dãy G.** Nó là hằng số đã chốt trong
   `kaggle-detector-retrain/train-behaviour-detector-gpu.py:27` (`GATE_MAP50=0.90`,
   `FLOOR_MAP50=0.85`, `GATE_MIN_RECALL=0.0`), đặt tên trong `brief.md` §8 là Gate 1/2/3. Pha này
   **đăng ký nó vào dãy G với ID G22 mà không đổi một hằng số nào** — nếu sửa hằng số thì tiền đăng
   ký cũ mất hiệu lực.
3. **Lần chạm lịch sử chưa được ghi sổ.** `sweep_results/r2/temporal_convlstm.pt` (06/09/2026
   01:01) lưu cả `Pt` và `yt` ⇒ tập test 64 clip **đã bị chạm** khi đó, nhưng không có mục nào ở
   Section B của `reports/test-touch-log.md`. Phải ghi hồi tố vào Section A (mục hồi tố, không tiền
   đăng ký), đúng cách sổ đã xử lý các tạo tác cũ khác.
4. **Tính lại chỉ số từ `Pt`/`yt` đã lưu không tiêu lần chạm** — miễn trừ (3) của §4.9. Vì vậy số
   50/64 / 75.64% được phép dùng ngay, không cần tiêu ngân sách.
5. **Chạy LCO-691 không tiêu lần chạm.** LCO-691 gộp 516 clip (452+64) và báo cáo ngoài-fold; mọi số
   Chương 6 đã là số ngoài-fold (`chuong-04:970-972`). Chỉ phép đo trên tập test **đóng băng, giữ
   riêng** 64 clip theo giao thức 452/64 mới tiêu một lần chạm.

## Requirements

**Chức năng**
- Một tệp tiền đăng ký trong thư mục kế hoạch, cấu trúc theo khuôn mẫu E2: giả thuyết / can thiệp
  một-biến / luật quyết định chốt trước / dự đoán khả bác bỏ / độ đo được phép trích dẫn / giới hạn phạm vi.
- Mỗi cổng đặt tên **đúng một mô hình**; biên tiền đăng ký; tiêu chí CI; nhánh kết luận cho mọi kết cục.
- Một mục Section B trong `reports/test-touch-log.md` khai báo lần chạm duy nhất (chưa điền kết quả).
- Một mục Section A ghi hồi tố lần chạm 06/09/2026.

**Phi chức năng**
- Commit theo conventional commits, không tham chiếu AI. Tiền đăng ký phải nằm ở commit **trước**
  commit chứa số đầu tiên — kiểm được bằng `git log`.
- Tệp < 200 dòng; kebab-case.

## Architecture

```
phase-01 (commit A: chỉ chữ, không số)
   ├── tien-dang-ky-g21-g25.md        <- luật quyết định
   └── reports/test-touch-log.md      <- +1 mục Section B (declared), +1 Section A (hồi tố)
            │
            ▼
phase-02..05 (commit B..N: chứa số)   <- hợp lệ chỉ khi commit A đã tồn tại
```

## Related Code Files

**Tạo**
- `plans/261011-1502-baseline-giao-thuc-bai-bao-goc-unsafe-net/tien-dang-ky-g21-g25.md`

**Sửa**
- `reports/test-touch-log.md` — thêm mục Section A (hồi tố 06/09) + Section B (khai báo 1 lần chạm)

**Không sửa**
- `kaggle-detector-retrain/train-behaviour-detector-gpu.py` — hằng số cổng G22 phải giữ nguyên bit

## Implementation Steps

1. Đọc `reports/test-touch-log.md` để khớp đúng định dạng hai Section.
2. Viết `tien-dang-ky-g21-g25.md` với năm khối cổng:

   **G21 — chân 1: ConvLSTM của ta có tái lập mốc no-bbox của [17]?**
   - Mô hình được đặt tên: `unsafenet/convlstm.py::ConvLSTM`, `n_cls=4`, 5 seed, **trên CUDA**.
   - Đọc trên **macro kiểu bài báo** (trung bình sensitivity 4 lớp) so với **85.78%**.
   - Luật chốt trước:
     | Kết quả | Kết luận đã chốt |
     |---|---|
     | macro ≥ 83.78% (trong 2.0 pp của 85.78%) **và** CI cụm của hiệu chứa 0 | **Tái lập được.** §6.12.5 được giữ nhãn hiện tại. |
     | macro 78.78-83.78% | **Mơ hồ.** Không tuyên bố gì; mốc 85.78% dán nhãn "chưa tái lập". |
     | macro < 78.78% (tức không khá hơn 75.64% đã đo quá 3.14 pp) | **Không tái lập được.** §6.12.5 phải đổi nhãn: 85.78% là số *đọc từ bảng của họ*, không phải số đã tái dựng. |
   - Dự đoán khả bác bỏ: nếu nguyên nhân là **thiết bị** (H3) thì sensitivity c1 phải rời khỏi vùng
     ~18% ở **đa số** seed; nếu nguyên nhân là **độ phân giải/dữ liệu** (H1/H2) thì c1 vẫn kẹt thấp
     trên CUDA ở `px=112` và chỉ nhấc lên khi `px` tăng. Ghi trước để không giải thích hậu nghiệm.

   **G22 — bộ phát hiện hành vi** (hằng số đã chốt sẵn, chỉ đăng ký vào dãy)
   - Mô hình được đặt tên: `yolo11s` trên quy ước hộp được chọn bởi luật ở pha 03.
   - Đạt: `mAP50 ≥ 0.90` **và** cả 8 lớp có AP > 0. Sàn: `mAP50 < 0.85` ⇒ dừng, bỏ pha 04.

   **G23 — cơ chế +10 pp của [17] có tái dựng được?**
   - Mô hình được đặt tên: ConvLSTM (cấu hình thắng G21) **+** tầng giảm khung/cắt clip ở pha 04.
   - Biên tiền đăng ký: **≥ +5.0 pp macro** so với cùng ConvLSTM không có tầng đó (tức ≥ 50% của
     +10.03 pp mà [17] báo cáo), **và** CI bootstrap theo cụm loại trừ 0.
   - Nhánh: < +5.0 pp hoặc CI chứa 0 ⇒ **trượt**; kết luận công bố là chẩn đoán §7.6 chưa được
     chứng thực bằng thực nghiệm của ta.

   **G24 — độ lạm phát giao thức làn C (đối chứng, KHÔNG phán quyết)**
   - Đại lượng: macro(giao thức 452/64) − macro(LCO-691), mọi thứ khác cố định.
   - Không có biên đạt/trượt: đây là **phép đo**, báo cáo kèm CI cụm và đối chứng scene-only.
   - Đăng ký trước **cách báo cáo** để không chọn cách trình bày sau khi thấy dấu của hiệu.

3. Thêm §"Giới hạn phạm vi" vào tiền đăng ký, sao đúng điều kiện G20: *số của kế hoạch này đo sau
   mốc đóng băng 05/10/2026; nó không được làm đổi một số tiêu đề hay một phán quyết cổng nào;
   tổng vẫn là 8 cổng, 2 đạt 6 trượt.*
4. Thêm mục Section A (hồi tố) vào `reports/test-touch-log.md`: tạo tác `temporal_convlstm.pt`,
   mtime 2026-09-06 01:01, `Pt`/`yt` đã lưu, 50/64 top-1 / 75.64% macro, ghi chú *không tiền đăng ký*.
5. Thêm mục Section B (khai báo, chưa có kết quả): mục đích = phép đo hợp nhất pha 05; cấu hình
   chốt trước; lệnh chính xác; giao thức 452/64; bộ độ đo; lý do được phép; ô kết quả để trống.
6. `git commit` chỉ hai tệp này. Thông điệp: `docs(next-round): pre-register G21-G24 and declare the single test touch before measuring`.

## Todo List

- [ ] Đọc định dạng hai Section của `reports/test-touch-log.md`
- [ ] Viết khối G21 (mô hình, biên 2.0 pp, ba nhánh, dự đoán khả bác bỏ H1-H5)
- [ ] Viết khối G22 (sao hằng số từ `train-behaviour-detector-gpu.py:27`, không sửa)
- [ ] Viết khối G23 (biên +5.0 pp, CI loại trừ 0)
- [ ] Viết khối G24 (đối chứng, không phán quyết, chốt cách báo cáo)
- [ ] Viết §giới hạn phạm vi theo điều kiện G20
- [ ] Ghi hồi tố lần chạm 06/09 vào Section A
- [ ] Khai báo 1 lần chạm vào Section B (ô kết quả trống)
- [ ] Commit, xác nhận `git log` cho thấy commit này đứng trước mọi commit có số

## Success Criteria

- `git log --oneline` cho thấy commit tiền đăng ký **trước** commit đầu tiên chứa số của pha 02.
- Mỗi cổng G21-G23 có: một mô hình được đặt tên, một biên số, một tiêu chí CI, và một nhánh kết luận
  cho **mọi** kết cục khả dĩ (kể cả kết cục bất lợi).
- `reports/test-touch-log.md` có đúng 1 mục khai báo chưa tiêu và 1 mục hồi tố.
- Không một con số đo mới nào xuất hiện trong commit này.

## Risk Assessment

| Rủi ro | Giảm thiểu |
|---|---|
| Biên 2.0 pp của G21 bị cho là tự chọn thuận lợi | Nêu rõ căn cứ: 2.0 pp < 1 clip trên n=64 (1 clip = 1.56 pp top-1), nên 2.0 pp là ngưỡng nhỏ nhất còn có nghĩa; ghi trước nên không điều chỉnh được. |
| Cám dỗ nới cổng sau khi thấy số | `chuong-06-danh-gia.txt:150-152` cấm thay mô hình được đặt tên; commit tiền đăng ký là bằng chứng kiểm được. |
| Viết tiền đăng ký mà quên một kết cục ⇒ giải thích hậu nghiệm | Mỗi bảng luật quyết định phải phủ toàn trục số, kiểm bằng rà soát ba nhánh liền kề. |
| Tiêu quá 1 lần chạm vì pha sau phát sinh nhu cầu | Pha 05 gộp **toàn bộ** arm vào một lần suy luận duy nhất; danh sách arm chốt ở pha 01. |

## Security Considerations

- Không có dữ liệu bí mật. Không commit `.env`, khoá API, token Kaggle (`kaggle.json`) — kiểm
  `git status` trước commit.
- `reports/test-touch-log.md` chứa số liệu nội bộ, không chứa dữ liệu cá nhân; frame đã chú giải
  không được đưa vào commit này.

## Next Steps

- Chặn: pha 02 (G21) và pha 03 (G22) chỉ được bắt đầu sau khi commit pha 01 tồn tại.
- Theo sau: pha 06 điền ô kết quả của mục Section B và ghi phán quyết cổng.

## Unresolved Questions

1. "Gate 1" của bộ phát hiện trong `brief.md` §8 đã bao giờ được phán quyết chưa? Nếu đã có một lần
   chạy bị bỏ dở, việc đăng ký nó thành G22 có vi phạm "tiền đăng ký trước khi có số" không?
2. Lần chạm 06/09 có cần **sự đồng ý tường minh** hồi tố theo §4.9, hay việc ghi sổ hồi tố là đủ?
3. G24 nên là "đối chứng không phán quyết" hay một cổng thật với biên? Nếu để không phán quyết thì
   nó không thể làm số tiêu đề — đó có phải điều ta muốn cho sản phẩm chính của kế hoạch?
4. Có cần một cổng riêng cho câu hỏi mở Q3 của chapter-07 (đối chứng S3), hay nó là một arm trong G24?
