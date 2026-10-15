# Pha 02 — Ch4: Nhãn Xuất Xứ Siêu Tham Số + Sửa Năm Sai Số Dòng

**Tệp đích:** `report/vn/chuong-04-chi-tiet-thuc-nghiem.txt` (974 dòng)
**Báo cáo nền:** `reports/audit-ch04-xuat-xu-sieu-tham-so.md`,
`reports/kiem-chung-cau-hinh-huan-luyen-ch04.md` (kiểm chứng độc lập, 2026-10-14)
**Ưu tiên:** CAO NHẤT — pha này sửa một tuyên bố **sai** đang nằm trong bản đã nộp.
**Trạng thái:** chưa bắt đầu

## Nhận xét GVHD được giải quyết

Y6 ("các số liệu và phương pháp là tự nghĩ ra, tự đo hay từ một nghiên cứu nào?") — trực tiếp.

## A. Sửa tuyên bố quá rộng ở dòng 519 (BẮT BUỘC, đây là lỗi thật)

Câu hiện tại: "Mọi đầu quan hệ và mọi mô hình làn CLIP trong luận văn dùng **cùng một cấu hình
huấn luyện**". Kiểm chứng cho thấy **ba** script huấn luyện dùng **ba** cấu hình khác nhau:

| Script | epochs | seeds | test_size | Vai trò |
|---|---|---|---|---|
| `run_kfold_relation.py` | 60 (`:253`) | 3 (`:252`) | 0.15 (`:189`,`:317`) | sinh **mọi** dòng LSO/LCO công bố |
| `run_relation_transformer.py` | 120 (`:290`) | 5 (`:287`) | 0.18 (`:231`,`:234`,`:320`) | chia đơn, sàng lọc kiến trúc |
| `deployment/scripts/train-binary-head-checkpoint.py` | 60 (`:41`) | 1 (`:42`) | 0.18 (`:59`,`:62`) | xuất checkpoint E3 triển khai |

**Cách sửa:** thu hẹp câu về đúng phạm vi ("mọi dòng kết quả LSO/LCO ở Chương 6 do
`run_kfold_relation.py` sinh ra với cùng một cấu hình"), rồi **thêm một đoạn công bố** hai cấu hình
còn lại và chúng xuất hiện ở đâu. Không được im lặng sửa câu mà bỏ qua hai cấu hình kia.

Luận cứ kỹ thuật để nêu kèm: `CosineAnnealingLR(opt, epochs)` lấy `T_max = epochs`, nên 60 và 120
là **hai lịch học khác hình dạng**, không phải "cùng cấu hình chạy dài hơn".

## B. Sửa năm số dòng sai

| Ch4 đang ghi | Sửa thành |
|---|---|
| `run_kfold_relation.py:239` (mặc định `--epochs`) | `:253` |
| `run_kfold_relation.py:175` và `:300` (`test_size=0.15`) | `:189` và `:317` |
| `run_kfold_relation.py:173-174` (bỏ fold thoái hoá) | `:187-188` |
| `run_kfold_relation.py:301-307` (cộng softmax theo seed) | `:191-197` (nhị phân), `:319-328` (bốn lớp) |
| `run_relation_transformer.py:177-180` (chọn epoch tốt nhất) | `:178-180` |

Số dòng **đúng, không sửa**: `:107-108`, `:117`, `:118`, `:150`, `:156-171`, `:158-160`.

## C. BÁC BỎ một đề nghị của audit

Audit Ch4 đề nghị "công bố `patience=30`". **Không tồn tại** `patience` trong mã. Cơ chế thật:
`train_one` chạy **đủ** ngân sách epoch, **không dừng sớm**, chỉ giữ bản sao trạng thái ở epoch có
`val_macro` cao nhất. Phải viết đúng cơ chế này; **cấm** viết "dừng sớm" hoặc nêu giá trị patience.

## D. Thêm nhãn xuất xứ cho mọi siêu tham số ở §4.5.1

Dùng đúng **năm nhãn** đã định nghĩa ở Pha 01. Phân loại đã kiểm:

- `AdamW(lr=1e-3, wd=0.01)` → quy ước công bố.
- `CosineAnnealingLR` → quy ước; `T_max = epochs` là mặc định của mã.
- `label_smoothing = 0.05` → lựa chọn tác giả (nhẹ hơn Szegedy 0.1); **không** bịa thực nghiệm.
- `clip_grad_norm = 5.0` → mặc định của mã, chưa dò tìm, không tìm thấy tiền lệ.
- `batch = 32` → quy ước.
- `lambda_sub = 0.3`, `lambda_frame = 0.5` → mặc định của mã, chưa dò tìm (Ch4 **đã** nói đúng).
- `test_size = 0.15`, `random_state = 0` → mặc định của mã.
- 3 seed → mặc định của mã; lý do nêu được: ngân sách 10 fold x 3 seed = 30 lần huấn luyện.

## E. Bổ sung hàng còn thiếu

- Ngưỡng NMS/IoU của bộ phát hiện: `build-isafetybench-clip-features.py` gọi `predict` **không
  truyền `iou`** → giá trị là **mặc định Ultralytics**, phải ghi rõ là không chạm tới.

## Todo

- [ ] A — thu hẹp câu dòng 519 + đoạn công bố ba cấu hình
- [ ] B — sửa năm số dòng
- [ ] C — viết đúng cơ chế chọn epoch (không patience, không dừng sớm)
- [ ] D — gắn năm nhãn xuất xứ vào bảng §4.5.1
- [ ] E — thêm hàng NMS/IoU mặc định thư viện
- [ ] Soát §4.2.3 (103–187) và §4.6–4.7 (731–906) — hai vùng audit chưa phủ

## Rủi ro

Sửa câu dòng 519 làm lộ ra một khác biệt mà bản nộp trước không nêu. Đây là **điểm cộng** với
GVHD (đúng tinh thần Y6) nhưng phải viết bình thản, nêu đúng phạm vi ảnh hưởng: không dòng kết
quả công bố nào thay đổi, vì mọi dòng công bố đều từ `run_kfold_relation.py`.

## Tiêu chí hoàn thành

Mỗi siêu tham số ở Ch4 có một nhãn xuất xứ; mọi số dòng trích dẫn khớp với mã hiện tại
(kiểm lại bằng `grep -n`); không câu nào tuyên bố rộng hơn bằng chứng.
