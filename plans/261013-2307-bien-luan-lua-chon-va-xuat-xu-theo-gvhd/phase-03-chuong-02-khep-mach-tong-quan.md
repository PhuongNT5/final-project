# Pha 03 — Ch2: Khép Mạch Từ Tổng Quan Sang Lựa Chọn Của Luận Văn

**Tệp đích:** `report/vn/chuong-02-tong-quan.txt` (523 dòng)
**Báo cáo nền:** `reports/audit-ch01-02-05-08-mach-luan.md`
**Ưu tiên:** CAO · **Phụ thuộc:** chạy SAU Pha 01 và 02
**Trạng thái:** chưa bắt đầu

## Nhận xét GVHD được giải quyết

Y1 ("báo cáo chỉ đang liệt kê chứ không phải một bài luận văn nghiên cứu") — gốc rễ nằm ở đây:
Ch2 khảo sát văn liệu nhưng không khép mạch sang lựa chọn của chính luận văn.

## Bằng chứng từ audit: 6 lựa chọn lớn, chỉ 1 được Ch2 biện luận đủ

| # | Lựa chọn lớn | Ch2 có biện luận? |
|---|---|---|
| 1 | CLIP đóng băng thay backbone video | CÓ (yếu — null-result, thiếu tiêu chí quyết định) |
| 2 | Transformer thay ConvLSTM / 3D-CNN | **THIẾU** |
| 3 | Phân rã thực thể (§2.2.4) | **THIẾU** (rõ nhất) |
| 4 | Leave-source-out thay k-fold thường | CÓ (mạnh nhất — dùng làm mẫu) |
| 5 | Balanced accuracy làm đại lượng tiêu đề | **THIẾU** (0 dòng) |
| 6 | Chọn thiết bị Jetson | **THIẾU** |

## Việc phải làm

Dùng **mục 2.4 làm khuôn** (đó là chỗ Ch2 khép mạch tốt nhất, cho lựa chọn #4). Với mỗi lựa chọn
còn THIẾU, thêm một đoạn ngắn ngay sau phần khảo sát tương ứng, theo đúng ba phần:
văn liệu nói gì → khoảng trống nào còn lại → **luận văn chọn gì và vì sao**, kèm trỏ tới mục Ch3/Ch4
nơi lựa chọn đó được triển khai.

1. **Transformer thay ConvLSTM/3D-CNN** — lý do: đầu vào là chuỗi 16 token đã đóng băng, không
   phải chuỗi ảnh thô; tự chú ý trên 16 vị trí rẻ hơn hồi quy tuần tự và không cần tích chập 3D.
2. **Phân rã thực thể** — nối tới §2.2.4; lý do: hành vi không an toàn là **quan hệ giữa các thực
   thể**, không phải thuộc tính của toàn khung.
3. **Balanced accuracy** — nối tới lớp lệch; BA = trung bình recall theo lớp, không thưởng cho mô
   hình đoán theo lớp đa số. Phải viết, vì đây là **đại lượng tiêu đề** của luận văn.
4. **Jetson** — nêu ràng buộc triển khai thật (biên, không mạng, ngân sách điện), không viết như
   thể đã so sánh nhiều thiết bị nếu chưa so sánh.

## Tài liệu mới [44]–[49]

Không gian trích dẫn hiện liên tục [1]–[43]; thêm tiếp **[44]–[49]** vào thư mục gộp ở Ch2:

| # | Nguồn | Dùng ở đâu |
|---|---|---|
| [44] | ActionCLIP (Wang et al., 2021) | tiền lệ CLIP đóng băng cho video (Ch2 #1, Ch3 §3.3) |
| [45] | Cluster-Robust Inference: A Guide to Empirical Practice | bootstrap theo cụm (Ch3 §3.7.4) |
| [46] | When Alpha Disappears (leakage ở backtest tài chính) | tiền lệ gần của "lạm phát giao thức" |
| [47] | Protocol before progress (leakage-aware evaluation, AIS) | như trên |
| [48] | VideoMAE (Tong et al., 2022) — nếu chưa trùng [15] | đối chứng B2 |
| [49] | Latency-aware inference at the edge (USENIX ATC 2025) | tiền lệ gần của "quyết định/giây" |

**Kiểm trước khi thêm:** [48] có thể trùng [15]; nếu trùng thì bỏ, đánh số lại liên tục.

## Cạm bẫy

- **Thiên kiến hậu nghiệm.** Không viết lựa chọn như thể đã biết trước kết quả. Lựa chọn #1 dẫn
  tới một null-result; phải giữ nguyên cách kể đó.
- Không tạo khoảng trống văn liệu không có thật để làm lựa chọn trông mạnh hơn.

## Todo

- [ ] Đoạn khép mạch cho lựa chọn #2 (Transformer)
- [ ] Đoạn khép mạch cho #3 (phân rã thực thể)
- [ ] Đoạn khép mạch cho #5 (balanced accuracy)
- [ ] Đoạn khép mạch cho #6 (Jetson)
- [ ] Gia cố #1 bằng tiêu chí quyết định rõ ràng
- [ ] Thêm [44]–[49], kiểm trùng [48]↔[15]
- [ ] Soát mọi chỗ trích [44]–[49] ở Ch3/Ch4 đã nhất quán

## Tiêu chí hoàn thành

Sáu lựa chọn lớn đều có một đoạn ở Ch2 nối văn liệu → khoảng trống → lựa chọn, mỗi đoạn trỏ tới
đúng mục Ch3/Ch4 triển khai nó.
