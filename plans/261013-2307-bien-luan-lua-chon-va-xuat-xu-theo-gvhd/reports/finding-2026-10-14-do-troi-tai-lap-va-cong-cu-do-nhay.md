# Finding 2026-10-14 — Độ trôi tái lập của hàng tiêu đề B1, và công cụ đo độ nhạy hậu đóng băng

Bối cảnh: GVHD yêu cầu "chạy thêm trên nhiều lớp, tham số, mô hình layer khác nhau" và hỏi "số
liệu tự nghĩ ra, tự đo hay từ nghiên cứu nào". Luận văn đã có thang kiến trúc (Bảng 6.1) nhưng
**chưa có một lần dò siêu tham số huấn luyện nào**. Công cụ dò đã được viết; trước khi dùng nó,
hàng đối chứng phải tái lập được hàng công bố. Nó **không** tái lập tới chữ số thập phân.

## 1. Sự kiện

| Nguồn | luồng CPU | BA (LSO-1100) | top-1 | Ghi chú |
|---|---|---|---|---|
| Tạo tác công bố `enc_isbclip_B1_lso-1100_headline-none_oof.json` (29/09) | không ghi | **84.17%** | 84.09% | hàng tiêu đề 84.2% |
| `run-esta-lso.py --model B1 --seeds 3` hôm nay, mã **không sửa** | 4 (mặc định torch) | **83.25%** | 83.00% | `_REPROCHECK` |
| `run_b1_sensitivity_sweep.py --only B1_published_config` | 4 (mặc định torch) | **83.25%** | 83.00% | hàng đối chứng của bộ dò |
| cùng bộ dò, `OMP_NUM_THREADS=1` | 1 | **84.71%** | 84.64% | chỉ đổi số luồng |

- Hai lần chạy ở 4 luồng **trùng nhau 1100/1100 clip** và trùng từng fold một
  (f1 84.7, f2 87.9, f3 79.4, f4 75.5, f5 81.1, f6 56.6, f7 97.2, f8 96.2, f9 84.0, f10 86.8)
  → bộ dò là bản sao trung thực của công thức công bố, không phải một công thức khác.
- 1 luồng so 4 luồng: **lệch 36/1100 clip**, +1.46 pp BA. Chỉ đổi số luồng CPU.
- Chạy 4 luồng so tạo tác công bố: **lệch 24/1100 clip**, −0.92 pp. Fold nào cũng cùng bộ thành
  viên (f1 n=144, f2–f3 n=107, f4–f10 n=106 ở cả hai), nên **không phải** do đổi cách chia.
- **Con số công bố nằm trong dải trôi**, không nằm ngoài: 83.25% ≤ 84.17% ≤ 84.71%.

BA của cùng một công thức, ba môi trường chạy: **83.25% – 84.17% – 84.71%**, dải **1.46 pp**.

## 2. Nguyên nhân (đã dò, không suy đoán)

Thủ phạm là **bước chọn epoch**, không phải bước huấn luyện. Dò riêng fold 6 — fold yếu nhất —
cùng mã, cùng đặc trưng, cùng ba hạt giống, **chỉ đổi số luồng CPU**:

| `torch.get_num_threads()` | acc fold 6 | epoch được chốt (hạt giống 0/1/2) | `val_macro` tương ứng |
|---|---|---|---|
| 1 | **67.0%** | 4 / 9 / 36 | 98.5852 / 98.2456 / 98.2456 |
| 4 (mặc định torch trên máy này) | **56.6%** | 4 / 10 / 11 | 98.5852 / 98.5852 / 98.5852 |
| 8 | **63.2%** | 4 / 12 / 3 | 98.5852 / 99.1228 / 98.0475 |

Một fold, một công thức, **dải 10.4 pp**. Phép dò khớp **đúng** với lần chạy đầy đủ: ở 4 luồng
cả phép dò và lần chạy đầy đủ đều cho fold 6 = 56.6%; ở 1 luồng cả hai đều cho 67.0%. Tạo tác
công bố cho fold 6 = 62.3%, tức một giá trị thứ tư của cùng công thức.

- `val_macro` đã **bão hoà ~98.0–99.1%** trên tập validation 15% tách từ phần huấn luyện, trong
  khi đại lượng cần dự đoán (BA trên fold giữ ra) chỉ ở 57–67%.
- Giá trị val **trùng điểm**: ở 4 luồng, cả ba hạt giống chốt ở **đúng cùng một số 98.5852** nhưng
  ba epoch khác nhau (4, 10, 11); ở 1 luồng, hai hạt giống trùng ở 98.2456 tại epoch 9 và 36.
- `esta_training.train_one` chọn bằng `if vm > best['val_macro']`, tức "epoch đầu tiên đạt max".
  Một sai số dấu phẩy động cỡ 1e-7 (thứ tự cộng dồn khác nhau khi số luồng CPU khác nhau) đủ để
  đổi epoch được chốt, và đổi epoch thì **đổi toàn bộ dự đoán của fold đó**.
- Vì vậy fold 6 có **bốn** số đo của cùng một công thức: 56.6% (4 luồng), 62.3% (tạo tác 29/09),
  63.2% (8 luồng), 67.0% (1 luồng).

Đã loại trừ: đổi cách chia fold (giống nhau từng clip), đổi tệp đặc trưng
(`feats_isbclip_all.pt` mtime 29/09 21:17, trước log chạy 29/09 21:56, không đổi từ đó), đổi
phiên bản thư viện (torch 2.11.0 / numpy 2.4.3 cài 24–25/03), đổi `in_dim` (B1 dùng 512 ở cả hai
đường chạy).

## 3. Hệ quả cho luận văn

1. **84.2% không phải một con số tái lập được tới chữ số thập phân.** Dải quan sát được của đúng
   một công thức trên đúng một bộ fold, chỉ đổi số luồng CPU, là **83.25–84.71%**, tức khoảng
   ±0.73 pp quanh 83.98%. Phải nói ra; không được trình bày 84.2% như một hằng số. Điểm có lợi:
   con số công bố nằm **giữa** dải, không phải ở biên trên.
2. **Kết luận tiêu đề không đổi dấu.** Sàn tiền đăng ký là 63.3%; biên đi từ +20.9 pp xuống
   +19.95 pp ở đầu dưới của dải. Lạm phát giao thức (97.7% so với LSO) cũng không đổi dấu. Không
   cổng nào đảo phán quyết vì 1.46 pp.
3. **Giữ nguyên số đã đóng băng.** 84.2% là giá trị của tạo tác đã đóng băng 05/10 và vẫn là hàng
   công bố. Mọi hàng mới đi vào mục **độ nhạy hậu đóng băng**, so với **hàng đối chứng 83.25%**
   của chính bộ dò ở **4 luồng** — không so với 84.2%, vì so lẫn hai môi trường chạy là so hai
   thứ khác nhau.
4. **Mọi hàng của bộ dò phải chạy ở cùng số luồng.** Đây là hệ quả bắt buộc của §2: một khác biệt
   1.46 pp sinh ra từ môi trường chạy sẽ lớn hơn phần lớn khác biệt siêu tham số mà bộ dò muốn đo.
   Hai mươi cấu hình còn lại đang chạy với `OMP_NUM_THREADS=4`, đúng môi trường của hàng đối chứng.
5. **Khuyết điểm thiết kế cần nêu, không cần sửa**: chọn checkpoint bằng một đại lượng đã bão hoà
   (98.2%) để dự đoán một đại lượng còn xa bão hoà (83–84%) là chọn gần như ngẫu nhiên giữa các
   epoch trùng điểm. Cách sửa đúng (chọn bằng loss, hoặc trung bình nhiều epoch, hoặc phá trùng
   điểm tất định) **phải không được làm bây giờ**: nó sinh một bộ số thứ hai sau đóng băng.

## 4. Công cụ

`run_b1_sensitivity_sweep.py`: 21 cấu hình, một nhân tố một lần, quanh B1 (`d`, `ff`, `t_layers`,
`heads`, `dropout`, `lr`, `wd`, `label_smoothing`, chặn gradient, số epoch). Chỉ ghi vào
`sweep_results/posthoc_sensitivity/`. Hàng `B1_published_config` là đối chứng. Chi phí đo được:
**4.6–4.9 phút/cấu hình** (10 fold x 3 hạt giống, đặc trưng đã cache) → 20 cấu hình ~1.6 giờ.

Tệp đã có: `B1_published_config_oof.json` (đối chứng chuẩn, 4 luồng, BA 83.25%),
`B1_published_config_threads4_oof.json` (bản sao của nó), `B1_published_config_threads1_oof.json`
(BA 84.71%). Hai tệp phụ chỉ tồn tại để chứng minh §2, không phải hàng kết quả.

## 5. Câu hỏi chưa giải quyết

- Có nên báo cáo dải 83.25–84.71% ngay ở Chương 1 và tóm tắt, hay chỉ ở Chương 4 (mục 4.5.6 sẽ
  viết) và trỏ tới từ Chương 6?
- Hàng `gradclip_5.0` của bộ dò đo đúng khác biệt giữa hai làn (1.0 so với 5.0). Nếu nó cho biên
  lớn hơn dải môi trường 1.46 pp thì phát biểu "một công thức cho mọi kiến trúc" ở Chương 3 cần
  yếu đi nữa — chưa biết trước khi chạy xong.
- Tạo tác công bố 29/09 **không ghi số luồng CPU**, nên không xác định được nó chạy ở môi trường
  nào. Có nên thêm trường đó vào tạo tác mới, dù việc này tạo khác biệt định dạng với tạo tác đã
  đóng băng?
- Tám tạo tác nhịp không ghi nhiệt độ; "tj vào <= 56 °C" là thủ tục của người đo. Có nên hạ câu
  đó xuống thành "không có bản ghi nhiệt độ cho phiên này" ở Chương 5 (hiện archive-only) hay chỉ
  sửa ở Chương 4 (đã sửa)?
