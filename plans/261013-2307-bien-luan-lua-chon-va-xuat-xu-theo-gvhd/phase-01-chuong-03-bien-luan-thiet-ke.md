# Pha 01 — Ch3: Biện Luận Thiết Kế + Công Thức Mục 3.7

**Tệp đích:** `report/vn/chuong-03-phuong-phap.txt` (958 dòng)
**Báo cáo nền:** `reports/audit-ch03-khe-ho-bien-luan.md`, `reports/research-xuat-xu-hoc-thuat-tham-so-va-phuong-phap.md`
**Ưu tiên:** CAO — pha này định nghĩa khung mà Ch4/Ch2 sẽ trích.
**Trạng thái:** chưa bắt đầu

## Nhận xét GVHD được giải quyết

Y2 (chưa biện luận từng phương pháp/mô hình/tham số), Y4 (tại sao layer này, tham số này),
Y5 (lý do + ưu + nhược của từng lựa chọn), Y7 (phép đo, công thức).

## Khung chuẩn: mỗi giá trị cần ba phần

Mọi giá trị thiết kế được nêu phải kèm:

1. **Lý do** — vì sao giá trị này phục vụ bài toán.
2. **Xuất xứ** — một trong **năm nhãn** (dùng thống nhất cả Ch3 và Ch4):
   - `quy ước công bố` (kèm trích dẫn),
   - `mặc định thư viện` (giữ nguyên, không chạm),
   - `mặc định của mã, chưa dò tìm` (nói đúng như vậy — ràng buộc 5),
   - `dò tìm có bằng chứng` (trỏ tới nhật ký/phụ lục),
   - `ràng buộc triển khai` (bị phần cứng hoặc giao thức ép).
3. **Phương án bị loại** — nếu có; nếu không có thì nói rõ là không thử phương án khác.

## Khe hở phải vá

| Mục Ch3 | Dòng | Giá trị đang liệt kê trần | Việc phải làm |
|---|---|---|---|
| §3.3.2 | 161–170 | d=128, h=4, FFN=256, 1 lớp, dropout 0.1, norm_first, GELU | Thêm đoạn biện luận sau bảng: tỉ lệ FFN/d = 2 trong dải Vaswani [43]; dropout 0.1 = Vaswani; d=128 là lựa chọn quy mô nhỏ cho đầu vào 16 khung cố định |
| §3.3.3 | Bảng 3.2 | biện luận *sự tồn tại* biến thể, không biện luận *chiều* | Thêm cột/đoạn: vì sao B1w tăng lớp, B1xl tăng d |
| §3.3.4 | 242–246 | cửa sổ 48, bước 24, T=16 | T=16 = quy ước Kinetics/VideoMAE [11][12][15]; cửa sổ 48/bước 24 = chồng 50%; nêu rõ phần nào là quy ước, phần nào là lựa chọn tác giả |
| §3.5 | 430–441 | P1 trình bày dạng liệt kê thuần | Viết lại thành văn có mạch: bài toán → lựa chọn → hệ quả |
| §3.7.3 | — | BA định nghĩa bằng lời | **Viết công thức** BA = (1/K)·Σ recall_k |
| §3.7.4 | — | bootstrap theo cụm [39], McNemar [40] không công thức | **Viết công thức** cả hai |
| mục tiền xử lý | — | N_MAX=8, conf 0.25, đệm 30%, IMGSZ=800 | Gắn nhãn xuất xứ; `conf 0.25` là mặc định Ultralytics → nhãn `mặc định thư viện` |
| giao thức | — | K=8, 10 fold, val 15%, 3 seed, mẫu dò 300 clip | Gắn nhãn + lý do |
| cổng | — | lề 3/5/2/3.0 pp | Nêu rõ là lề **tiền đăng ký**, chọn trước khi đo |
| E0–E3 | — | 2 lớp, dropout 0.3 (khác khối chung 1 lớp / 0.1) | Giải thích vì sao khác, hoặc nói rõ là khác biệt lịch sử chưa đồng nhất hoá |

## Nguồn xuất xứ đã kiểm (dùng trực tiếp)

- CLIP ViT-B/32 đóng băng: quy ước (ActionCLIP, Frozen-CLIP/EVL) — [28] + tài liệu mới [44].
- T=16 cách đều: quy ước (Kinetics, VideoMAE) — [11][12][15][36].
- Transformer d/FFN/dropout: tỉ lệ FFN/d = 2 trong dải Vaswani [43].
- Leave-source-out / StratifiedGroupKFold: quy ước [29][30][31][37].
- Bootstrap theo cụm: Efron & Tibshirani [39]; McNemar chính xác [40].
- `clip_grad_norm = 5.0`: **không tìm thấy tiền lệ** → nhãn `mặc định của mã, chưa dò tìm`.
- `label_smoothing = 0.05`: nhẹ hơn Szegedy (0.1) → nêu là lựa chọn tác giả, không bịa thực nghiệm.
- "lạm phát giao thức" và "quyết định/giây": **tự đặt** → phải nói rõ là thuật ngữ của luận văn.

## Việc KHÔNG được làm

- Không chạm 20 đoạn mà audit Ch3 đánh dấu đã biện luận tốt.
- Không viết "chúng tôi đã thử X và thấy Y" nếu không có nhật ký.
- Không đổi bất kỳ con số nào trong Ch3.

## Todo

- [ ] §3.3.2 — đoạn biện luận cấu hình khối Transformer chung
- [ ] §3.3.3 — biện luận chiều của B1w/B1xl
- [ ] §3.3.4 — biện luận T=16, cửa sổ 48, bước 24 và 8
- [ ] §3.5 — viết lại P1 thoát dạng liệt kê
- [ ] §3.7.3 — công thức BA
- [ ] §3.7.4 — công thức bootstrap theo cụm + McNemar chính xác
- [ ] Nhãn xuất xứ cho N_MAX, conf, đệm, IMGSZ, K, fold, val, seed, mẫu dò
- [ ] Nêu lề cổng là tiền đăng ký
- [ ] Giải thích E0–E3 dùng 2 lớp / dropout 0.3

## Tiêu chí hoàn thành

Đọc Ch3 từ đầu đến cuối, không còn bảng giá trị nào đứng một mình không có đoạn biện luận
theo sau, và mục 3.7 có công thức cho mọi đại lượng nó định nghĩa.
