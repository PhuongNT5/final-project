# Báo cáo phủ sóng Y1–Y7 — nơi từng nhận xét GVHD được trả lời trong luận văn

**Ngày:** 2026-10-14 · **Nhánh:** `next-round` · **Phạm vi kiểm:** 11 tệp đã nộp ở `report/vn/`.

Bảng này là tạo tác làm việc, **không** thuộc luận văn (quyết định phạm vi #2: không viết báo cáo
giải trình riêng). Mục đích duy nhất: mỗi nhận xét trỏ được tới mục **trong báo cáo** đã trả lời
nó, để kiểm được rằng không nhận xét nào chỉ được trả lời ở `plans/`.

## Y1 — văn phong luận văn nghiên cứu, không phải danh mục

| Chỗ sửa | Việc đã làm |
|---|---|
| `chuong-01-mo-dau.txt` §1.4.3 (mới) | Ba giả thuyết H1–H3 phát biểu được, mỗi giả thuyết nêu rõ **bị phán quyết ở cổng nào**; nói thẳng giả thuyết sinh ra đề cương (H2) là giả thuyết bị bác bỏ 0/5, và trụ triển khai biên **không** có cổng tiền đăng ký |
| `chuong-02-tong-quan.txt` §2.2.6 (mới) | Bốn họ phương pháp chiếu qua ba tiêu chí, nêu rõ hai họ bị loại **bằng lập luận chứ không bằng phép đo** |
| `chuong-03-phuong-phap.txt` §3.4.1, §3.5 | Chuyển các đoạn đặc tả dày thành mạch "câu hỏi → vì sao cách này trả lời được → giới hạn" (`Mạch lập luận sinh ra đồ thị này`, `Vì sao P1 thôi là đồ thị được triển khai`) |
| `chuong-05-trien-khai-bien.txt` | 141 dòng cảnh báo biên soạn nội bộ chuyển xuống **phụ chú biên soạn cuối tệp** (nguyên văn, không xoá); chương vào câu hỏi §5.1 trong ~18 dòng thay vì ở dòng 158 |
| `chuong-07-phan-tich-that-bai.txt` §7.6 | Bảng 7.2 có đoạn tổng hợp: thứ tự nguy hiểm của bốn nhóm, phân bố 11/21 lần rút về một làn và **lý do cấu trúc** của nó, bốn quy tắc sinh ra từ bốn nhóm, và khuyết điểm của chính sổ đăng ký |

## Y2 — biện luận từng phương pháp / mô hình / tham số / phép đo / baseline

- `chuong-03-phuong-phap.txt` §3.3.2: một đoạn lý do riêng cho **từng** giá trị của khối
  Transformer (FFN ×2, dropout 0.1, token [CLS], d = 128, số lớp, số đầu chú ý) — dòng 204–264.
- §3.3.3: lý do tồn tại của **từng** biến thể toàn khung trong bốn biến thể.
- §3.4.3: thang encoder E0–E3, nêu rõ mức điều chuẩn và độ sâu **khác** đường B1 và vì sao.
- §3.7.3 / §3.7.5: lý do chọn từng chỉ số, kèm ba đại lượng **được tính mà không được trích** và
  một kiểm định **bị cấm**.
- `chuong-02-tong-quan.txt` §2.4.5: vì sao đại lượng báo cáo là độ chính xác cân bằng.

## Y3 — lý do chọn phải nằm TRONG báo cáo

Toàn bộ nội dung bảng này nằm trong `report/vn/`. Không mục nào trỏ ra `plans/`, log hay commit để
lấy lý do. Các tệp đo chỉ được dẫn làm **tạo tác** cho con số, không làm nơi chứa lập luận.

## Y4 — vì sao kiến trúc này, số lớp này, tham số đầu vào này

| Giá trị | Nơi biện luận | Nhãn xuất xứ |
|---|---|---|
| d = 128 | Ch3 §3.3.2 dòng 222 | `[ràng buộc]` — trần dung lượng §3.1.2 ép ra |
| số lớp = 1 (B1) | Ch3 §3.3.2 dòng 229 | `[ràng buộc]` — độ sâu nhỏ nhất chạy được |
| số đầu chú ý = 4 | Ch3 §3.3.2 dòng 250 | `[mặc định của mã]` — **không** có bằng chứng cho riêng giá trị đang dùng, và điều đó được nói ra |
| FFN/d = 2 | Ch3 §3.3.2 dòng 204 | `[quy ước]` |
| dropout = 0.1 | Ch3 §3.3.2 dòng 211 | `[quy ước]` — đúng giá trị bộ mã hoá gốc [43] |
| T = 16 khung | Ch3 §3.3.4 dòng 338 | `[quy ước]` |
| cửa sổ 48 / bước nhảy 24 | Ch3 §3.3.4 dòng 345 | `[mặc định của mã]` |
| `iou = 0.7`, `max_det = 300`, `agnostic_nms` | Ch4 §4.3.1 | `[mặc định thư viện]` — nêu rõ `max_det` không có tác dụng thực tế vì N_MAX = 8, còn `iou` **có** tác dụng và **chưa được dò tìm** |
| chín siêu tham số huấn luyện | Ch4 §4.5.1 | sáu dùng chung + **bốn khác nhau giữa các làn**, có bảng đối chiếu |

Thang kiến trúc (vì sao *lớp này* chứ không phải lớp khác) đã có sẵn ở Bảng 6.1 và không phải viết
mới; cổng G13 ở §6.11.4 là chỗ thang dung lượng bị phán quyết.

## Y5 — ưu VÀ nhược điểm

- Ch2 §2.4.5: nhược điểm của BA nói thẳng — nó **đối xứng**, định giá một clip không an toàn bị
  bỏ sót đúng bằng một báo động giả, trong khi tác vụ thì không đối xứng.
- Ch2 §2.2.6: hai họ bị loại bằng lập luận → đây là nhược điểm của chính lựa chọn, được khai ra.
- Ch2 §2.5 (khối mới): bo mạch cụ thể **không có luận chứng** — là `[ràng buộc]` thiết bị có sẵn;
  kèm hai phép đo nói ngược lại một lời tự khen (8GB không phải ràng buộc tới hạn §5.7.2; bộ giải
  mã cứng không được dùng §4.6.2).
- Ch4 §4.5.1 hệ quả thứ hai: làn bốn lớp huấn luyện **không có trọng số lớp** nhưng bị chấm bằng
  macro accuracy — không ăn khớp, và **không** có lần chạy đối chứng để biết nó đáng bao nhiêu.
- Ch4 §4.5.6 (mới): cái giá của việc chọn điểm kiểm tra bằng một đại lượng đã bão hoà — hàng tiêu
  đề chỉ tái lập trong dải **1.46 pp** khi chỉ đổi số luồng CPU.
- Ch7 §7.6: nhược điểm của chính sổ đăng ký rút số — nó chỉ liệt kê những gì đã được tìm ra, và
  một nhóm thứ năm (*số đúng nhưng chỉ xác định tới một dải*) còn thiếu.

## Y6 — xuất xứ: tự nghĩ / tự đo / lấy từ nghiên cứu nào

- **Năm nhãn xuất xứ** định nghĩa ở Ch3 §3.1.4: `[quy ước]`, `[mặc định thư viện]`,
  `[mặc định của mã]`, `[dò tìm]`, `[ràng buộc]`. Hai quy tắc kèm theo: nhãn
  `[mặc định của mã]` **được viết ra, không được che**; nhãn nói về **xuất xứ**, không nói về
  **chất lượng**.
- **Ba nhãn nguồn đường cơ sở** ở Bảng 6.1b (§6.1.6): `[ĐO]`, `[ĐỌC]`, `[SỐ HỌC]`.
- mAP@50 / mAP@50-95 khai rõ là **số đọc**, không phải số luận văn tự cài đặt (Ch3 §3.7.5).
- Ch3 bổ sung sáu mục tài liệu còn thiếu ([11], [12], [15], [38], [43], [45]); danh mục chương
  nay khớp đúng 12 trích dẫn của nó. Toàn kho: 45 mục, đều được trích, không mục treo.

## Y7 — phép đo, công thức, baseline so sánh

- Ch3 §3.7.3–§3.7.5 viết ra công thức của **11 đại lượng**: macro accuracy (hai dạng, và luận văn
  báo cáo cả hai) ở §3.7.3; bootstrap theo cụm (dạng một chỉ số và dạng ghép đôi) và McNemar chính
  xác ở §3.7.4; còn lại ở §3.7.5 — ROC-AUC
  một-so-với-còn-lại, PR-AUC nội suy bậc thang, ECE/MCE 15 bin, Brier đa lớp, κ của Cohen, MCC
  dạng Gorodkin, khoảng Wilson, t ghép đôi hiệu chỉnh Nadeau–Bengio [45]. Mỗi công thức kèm
  **dòng mã** tính nó.
- Tham số cố định khai một lần: B = 5000, `seed = 0`, α = 0.05, phương pháp phân vị (không BCa).
- Bảng 6.1b (§6.1.6): mỗi con số kết quả khai bốn thuộc tính — đại lượng, giao thức, **trị số**
  của đường cơ sở, và **nguồn** của đường cơ sở.
- Ch6 §6.12 cảnh báo (a) nay trỏ thẳng về §3.7.4: hai khoảng (đơn vị nhóm so đơn vị clip) dùng
  **cùng** B, `seed`, α và phương pháp — chỉ khác thứ được rút có hoàn lại — kèm dòng mã của từng
  dạng (`eval_utils/cluster_bootstrap.py` dòng 23 và 52; `eval_utils/paired_tests.py` dòng 62).

## Kiểm tra cuối (Pha 06)

| Phép kiểm | Kết quả |
|---|---|
| `check.py` 11 tệp | 0 vi phạm NUMSTART; 0 dòng LONG ngoài vùng miễn (đường dẫn, tên trường, 6 vùng bảng rộng Ch6) |
| grep số đã rút | mọi lần xuất hiện nằm trong ngữ cảnh rút/giải thích; `3.91` **0** lần |
| kiểm chữ số trên `git diff` cả nhánh | 1356 dòng thêm / 173 dòng bớt; chỉ **5** token số giảm, cả năm là **dòng mã tham chiếu** được cập nhật (`run_relation_transformer.py:178-180`, `run_kfold_relation.py:187-188`, `:189`, `:191-197`, `:317-328`) — đã đối chiếu với mã hiện tại, cả năm khớp. **Không** con số kết quả nào bị bỏ |
| đếm hàng Bảng 7.2 | A=9, B=5, C=7, D=9 — khớp đúng đoạn tổng hợp ("21 lần rút của A–C", "11 trong 21") |
| bảng cổng | vẫn tám cổng / hai đạt / sáu trượt |

## Câu hỏi chưa giải quyết

1. Phần **phân tích độ nhạy hậu nghiệm** (20 cấu hình, `OMP_NUM_THREADS=4`) đang chạy, 4/20 xong.
   Khi xong phải viết thành một mục có nhãn rõ *hậu nghiệm*, so với **hàng đối chứng 83.25%**,
   không so với 84.2%, và **không** phán quyết lại cổng nào.
2. ~~Hai tạo tác `_REPROCHECK` chưa quyết định giữ hay bỏ.~~ **Đã giải quyết:** giữ, và nay được
   **trích tên trong luận văn** ở Ch4 §4.5.6 cùng hai tệp đối chứng luồng trong
   `sweep_results/posthoc_sensitivity/` — ba hàng của dải 1.46 pp nay đều có tạo tác, đúng quy ước
   §6.1.6. Hậu tố `_REPROCHECK` được giải thích trong luận văn để không đọc lẫn với tạo tác 29/09.
3. Trụ triển khai biên vẫn **không** có cổng tiền đăng ký. §1.4.3 khai ra điều đó thay vì làm
   phẳng nó; nếu hội đồng coi đây là khuyết điểm phạm vi thì nó không sửa được sau ngày đóng băng.
