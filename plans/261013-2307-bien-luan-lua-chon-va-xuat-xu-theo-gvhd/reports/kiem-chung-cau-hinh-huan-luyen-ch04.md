# Kiểm Chứng Độc Lập: Cấu Hình Huấn Luyện Mục 4.5

**Ngày:** 2026-10-14. **Người kiểm:** phiên chính (không qua sub-agent).
**Lý do:** báo cáo audit Ch4 nêu nghi vấn ở dòng 519 + ba sai lệch số dòng. Phải xác nhận trước khi sửa.

## Ba script huấn luyện — ba cấu hình KHÁC NHAU (đã xác nhận)

| Script | `--epochs` | `--seeds` | `test_size` | Dùng để làm gì |
|---|---|---|---|---|
| `run_kfold_relation.py` | 60 (`:253`) | 3 (`:252`) | 0.15 (`:189`, `:317`) | **Sinh mọi con số LSO/LCO công bố** |
| `run_relation_transformer.py` | 120 (`:290`) | 5 (`:287`) | 0.18 (`:231`,`:234`,`:320`) | Chia đơn (blind split), sàng lọc kiến trúc |
| `deployment/scripts/train-binary-head-checkpoint.py` | 60 (`:41`) | 1 (`:42`) | 0.18 (`:59`,`:62`) | Xuất checkpoint E3 triển khai |

=> Tuyên bố ở Ch4 dòng 519 ("mọi mô hình làn CLIP dùng **cùng một** cấu hình huấn luyện") **quá rộng**.
Đúng khi thu hẹp về: *mọi dòng kết quả LSO/LCO trong Chương 6 đều do `run_kfold_relation.py` sinh ra
với cùng một cấu hình*. Phải sửa câu, không được bỏ qua.

## Sai lệch số dòng phải sửa trong Ch4

| Ch4 đang ghi | Thực tế | Nội dung |
|---|---|---|
| `run_kfold_relation.py:239` | `:253` | mặc định `--epochs 60` |
| `run_kfold_relation.py:175` và `:300` | `:189` và `:317` | `train_test_split(test_size=0.15)` |
| `run_kfold_relation.py:173-174` | `:187-188` | bỏ fold thoái hoá |
| `run_kfold_relation.py:301-307` | `:191-197` (nhị phân), `:319-328` (bốn lớp) | cộng softmax theo seed |
| `run_relation_transformer.py:177-180` | `:178-180` | chọn epoch theo `val_macro` tốt nhất |

## Số dòng Ch4 đang ghi ĐÚNG (không sửa)

`run_relation_transformer.py` — `:107-108` set_seed, `:117` bs=32, `:118` lambda_sub/lambda_frame,
`:150` clip_grad_norm 5.0, `:156-171` khối cấu hình, `:158-160` docstring n_sub,
`:166` AdamW, `:167` CosineAnnealingLR, `:169` CrossEntropyLoss.

## Một kết luận của audit bị BÁC BỎ

Audit Ch4 đề nghị "công bố `patience=30`". **Không tồn tại `patience` trong toàn bộ mã nguồn**
(`grep -rn patience` trên mọi `.py` → 0 kết quả). Cơ chế thật là:
`train_one` chạy **đủ** số epoch, không dừng sớm; chỉ **giữ lại bản sao trạng thái** ở epoch có
`val_macro` cao nhất (`:170`, `:178-180`). Phải viết đúng là "chọn epoch tốt nhất trên validation",
**không** được viết "dừng sớm" hay nêu một giá trị patience không có trong mã.

**Đính chính phạm vi (cùng ngày).** `patience = 30` **có** tồn tại — nhưng ở cấu hình huấn luyện
**bộ phát hiện** của ultralytics, và Chương 4 mục 4.6 **đã công bố** nó trong bảng đọc từ
`detector_best.pt` / `yolo26_best.pt` (hàng ``batch` / `patience` / `freeze`` = 16 / 30 / 0). Vậy
đề nghị của audit không sai về sự tồn tại của giá trị, chỉ sai về **nơi** nó thuộc về: nó thuộc
huấn luyện bộ phát hiện, **không** thuộc huấn luyện các đầu phân loại. Với các đầu phân loại thì
kết luận trên vẫn đứng: không dừng sớm, không patience.

`CosineAnnealingLR(opt, epochs)` lấy `T_max = epochs`, nên lịch học **gắn cứng vào ngân sách epoch**:
đổi `--epochs` là đổi cả hình dạng lịch học, không chỉ đổi độ dài huấn luyện. Đây là lý do kỹ thuật
để nêu rõ 60 và 120 là hai cấu hình khác nhau chứ không phải "cùng cấu hình, chạy dài hơn".

## Câu hỏi chưa giải quyết

1. Dòng kết quả nào trong Chương 6 (nếu có) được sinh bởi `run_relation_transformer.py` chứ không
   phải `run_kfold_relation.py`? Cần soát Bảng 6.x để câu sửa ở dòng 519 thu hẹp đúng phạm vi.
