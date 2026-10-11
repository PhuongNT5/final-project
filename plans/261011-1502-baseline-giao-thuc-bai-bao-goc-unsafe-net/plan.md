---
title: "Chạy lại giao thức và phép đo của bài báo gốc Unsafe-Net làm baseline"
description: "Tái dựng mốc no-bbox 57/64 / 85.78% và cơ chế giảm khung +10 pp của [17], đo dưới cả giao thức bài báo (452/64) và LCO-691, lấy độ lạm phát giao thức làm sản phẩm."
status: pending
priority: P1
effort: 50h
branch: next-round
tags: [replication, unsafe-net, convlstm, yolo-framegate, lco-691, pre-registration, lane-c]
created: 2026-10-11
---

# Baseline giao thức bài báo gốc (Önal & Dandıl, ref [17])

**Làn C duy nhất** (Unsafe-Net, 691 clip, 4 lớp); không trộn làn A/B (`chuong-06-danh-gia.txt:82-85`).
**Sau mốc đóng băng 05/10/2026** (`00-trang-thai-bien-soan.txt:11`) chỉ được **sửa chữ**: kế hoạch nhận
đúng điều kiện tiền lệ G20 — *giữ nguyên mọi số tiêu đề và phán quyết cổng*; số mới sống ở plan folder.

## Hai chân của khoảng cách — chân 1 là ưu tiên cao nhất

| Mốc | top-1 | macro (trung bình sensitivity) | Nguồn |
|---|---|---|---|
| ConvLSTM của ta (**đã đo**) | mps 50/64 · **cpu 47/64** | 75.64% · **70.22%** | `temporal_convlstm{,_cpu}.pt`; G21 **bác bỏ** 11/10 |
| ConvLSTM của họ (**họ báo cáo**) | 57/64 = 89.06% | **85.78%** | Table 5 [17]; `unsafenet/config.py::PAPER['nobbox']` |
| Hợp nhất của họ (**họ báo cáo**) | 62/64 = 96.88% | **95.81%** | Table 6 [17] |

Chân 1 **chưa giải thích được và đã rộng ra**. G21 (chỉ đổi mps→cpu) **bị bác bỏ**: cpu hơn val
+2.78 pp nhưng mất 3 clip test, c2 rơi 30.77 pp vì **một biến gây nhiễu thuần tuý** ⇒ val 83 clip mù
với test, **không hàng per-class một-lần-chạy nào so được với [17]**. `reports/phan-quyet-2026-10-11-g21-g25.md`.

## Các pha

| # | Pha | Hiện trạng | Công | Cổng |
|---|---|---|---|---|
| 1 | [Tiền đăng ký + khai báo lần chạm test](phase-01-tien-dang-ky-va-khai-bao-lan-cham-test.md) | **done** 11/10 — `tien-dang-ky-g21-g25.md` | 3h | — |
| 2 | [Chân 1 — tái lập hoặc bác bỏ mốc 57/64 / 85.78%](phase-02-chan-1-tai-lap-moc-no-bbox-convlstm.md) | **done** 11/10 — G21 **bác bỏ** (47/64) | 12h | **G21** |
| 2b | **Nhị phân An toàn/Không an toàn (NP-125, NP-125-D)** — thêm 11/10 | **done** — BA 74.19%, G25 **trượt**; đo lại theo mục tiêu phát hiện (0 lần chạm test): recall 82.81% @ báo động giả 34.43%, AUC trong-bối-cảnh 0.6426 so với gộp 0.8485 | 4h | **G25**; NP-125-D **không cổng** |
| 3 | [Bộ phát hiện hành vi trên CUDA](phase-03-bo-phat-hien-hanh-vi-tren-cuda.md) | **chặn GPU** — baseline 416 đã bão hoà ở 0.791/0.806 (< sàn 0.85) | 8h | **G22** |
| 4 | [Tầng giảm khung + cắt clip đơn-hành-động](phase-04-tang-giam-khung-va-cat-clip.md) | **done (val)** — G23 **trượt**: cơ chế cho +0.13 pp, [17] gán +10.03 pp | 10h | **G23** |
| 5 | [Hợp nhất, hai giao thức, độ lạm phát](phase-05-hop-nhat-hai-giao-thuc-va-do-lam-phat.md) | pending | 10h | **G24** (đối chứng, không phán quyết) |
| 6 | [Phán quyết thống kê + sửa chữ](phase-06-phan-quyet-thong-ke-va-sua-chu.md) | **sửa chữ done** 11/10 — §6.15 mới, §8.3.8, 6 tệp `report/vn/`; 37/37 số khớp JSON | 7h | — |

Phụ thuộc: 1 → (2, 2b) → (3 → 4) → 5 → 6. Pha 3 có thể chạy song song với pha 2 (khác thiết bị, khác tệp).

## Phép đo nhị phân — xung đột "trên 64 clip test" đã được chốt

64 clip test của [17] **toàn bộ là KAT** ⇒ nhị phân trên đúng 64 clip ấy **suy biến**. Hai phép đo, **không gộp**:

| Ký hiệu | Tập | Thành phần | Đại lượng chính | Sàn / trần |
|---|---|---|---|---|
| **NP-125** | 125 clip test, đóng băng | 64 KAT / 61 AT | **BA** | 51.20% / **124/125 cứng** |
| **NP-64** | lát cắt 64 clip của [17] | 64 KAT / 0 AT | chỉ recall KAT | — |
| **NP-125-D** | như NP-125, đọc lại | lớp dương = KAT | recall / báo động giả / F1 / PR-AUC | ngưỡng chốt trên val |

Trần cứng vì `0_te23.mp4` ≡ `4_te9.mp4` từng byte, nhãn đối nghịch, cả hai trong test. Chính sách trùng lặp + đối chứng scene-only: `tien-dang-ky-g21-g25.md` §1, §4.

## Pha nào được bỏ nếu cổng trượt sớm

- **G21 trượt** → vẫn chạy pha 3-6, nhưng khoảng cách thành *của ta so với của ta*; 85.78% dán nhãn "chỉ đọc từ bảng của họ". Độ lạm phát không cần mốc của họ.
- **G22 < 0.85** → bỏ pha 4 + phần hợp nhất pha 5 (tầng giảm khung mất đầu vào); pha 5 còn arm không-bbox dưới hai giao thức. **G22 ∈ [0.85, 0.90)** → pha 4 chạy, số pha 5 mang nhãn "bộ phát hiện dưới cổng".
- **G23 đã trượt** (`reports/phan-quyet-2026-10-11-g23-co-che-tren-val.md`, 0 lần chạm test): giảm khung cho +0.13 pp, McNemar p = 1.0, trên 3 bộ phát hiện. +8.88 pp val đến từ **việc dùng bộ phát hiện**, không từ cơ chế họ nêu tên; hợp nhất không thêm gì trên member đơn tốt nhất (88.05% = 88.05%).

## Rủi ro mức chương trình (nêu trước khi có số)

1. **Bootstrap cụm trên 8 nhóm bối cảnh thiếu lực** (§6.1.3, `chuong-06-danh-gia.txt:104-113`): n = 8
   đơn vị ⇒ CI rất rộng, **rất có thể chứa 0 với mọi biên dưới ~10 pp**. Biên có CI chứa 0 thì không
   chứng nhận được và không làm số tiêu đề. CI mức clip in kèm nhãn `WRONG UNIT`.
2. **Kết quả bất lợi là kết cục khả dĩ nhất và được chấp nhận** (làn C đã công bố 6/8 cổng trượt).
   Không thiết kế lại cổng, đổi mô hình được đặt tên, hay nới biên sau khi thấy số (`:150-152`).
3. **Phụ thuộc CUDA/Kaggle.** MPS huấn luyện hỏng âm thầm trên macOS 14 (`addcmul_` của Adam,
   `chuong-03-phuong-phap.txt:484-487`) — và `unsafenet/convlstm.py` từng mặc định `device='mps'`.
   Pha 3 **chặn** trên quota Kaggle/Colab. M3 chỉ CPU cho đầu, MPS cho trích đặc trưng.
4. **Ngân sách chạm test: 2 tiền đăng ký** (`chuong-04-chi-tiet-thuc-nghiem.txt:949-973`); tiêu
   **đúng 1 lần gộp**. **Dừng sớm theo test đã làm bẩn cả hai nhánh làn C** (`:1567-1570`) — kế hoạch
   không thêm đường nhiễm thứ ba: dừng sớm chỉ theo val cắt từ train, chọn mô hình trên val/LCO-OOF.

## Sản phẩm

Đại lượng báo cáo được là **độ lạm phát giao thức** giữa (a) giao thức cùng-phân-bố 452/64 của [17] và
(b) LCO-691, mọi thứ khác cố định — kèm đối chứng scene-only/đa-số-theo-view, CI bootstrap cụm 5,000 mẫu, đơn vị = nhóm bối cảnh.
