# NP-125-D — phép đo nhị phân theo mục tiêu **phát hiện không an toàn**

| | |
|---|---|
| Ngày | 2026-10-11 |
| Làn | **C duy nhất** (Unsafe-Net, test 125 clip đóng băng: 64 KAT / 61 AT) |
| **Lần chạm test mới** | **0** — đọc lại đúng `Pt` của lần chạm đã tiêu ở G25. Không huấn luyện, không suy luận lại. Ngân sách vẫn còn **1**. |
| Tiền đăng ký | `../tien-dang-ky-np125d-phat-hien-nhi-phan.md` (viết trước khi có số) |
| Số thô | `../ket-qua-np125d-phat-hien.json` |
| Mã | `run_binary_detection_scorecard.py`, `eval_utils/detection_metrics.py`, `eval_utils/cluster_bootstrap.py::metric_cluster_bootstrap` |
| Quan hệ với bảng cổng | **tái diễn đạt, không phán quyết.** Vẫn **tám cổng, hai đạt, sáu trượt**. |

**Kết luận một dòng:** đổi từ BA sang các đại lượng phát hiện **không** cứu được kết quả; nó làm
nguyên nhân thất bại hiện ra rõ hơn — **59% khoảng cách của AUC gộp so với mức ngẫu nhiên là do bối
cảnh, không do nội dung clip**.

## 1. Vì sao phải đo lại, và đo khác ở chỗ nào

BA 74.19% của G25 **đối xứng**: một lần bỏ sót clip Không an toàn tính ngang một lần báo động giả
trên clip An toàn. Với hệ cảnh báo, hai cái khác giá. Lớp dương = **KAT**; mọi số dưới đây theo đó.

Ngưỡng là biến mới, nên **toàn bộ ngưỡng chốt trên val 103 clip** rồi mới đọc test — không ngưỡng
nào chọn trên test, kể cả để minh hoạ (tiền đăng ký §2-3).

## 2. Bốn điểm hoạt động, ngưỡng chốt trên val

| Điểm | τ (từ val) | recall KAT | báo động giả | precision | F1(KAT) | Youden J | TP/FP/FN/TN |
|---|---|---|---|---|---|---|---|
| **OP0** argmax | 0.5000 | 82.81% | 34.43% | 71.62% | 76.81% | 48.39 | 53/21/11/40 |
| **OP1** max F1 trên val | 0.4131 | 92.19% | 55.74% | 63.44% | 75.16% | 36.45 | 59/34/5/27 |
| **OP2** ngân sách FPR ≤ 10% | 0.7276 | 57.81% | **3.28%** | **94.87%** | 71.84% | **54.53** | 37/2/27/59 |
| **OP3** đồng giá báo động | 0.3124 | 95.31% | 72.13% | 58.10% | 72.19% | 23.18 | 61/44/3/17 |

Không phụ thuộc ngưỡng: **PR-AUC(KAT) 0.876**, ROC-AUC 0.848.

CI bootstrap cụm, **đơn vị = nhóm bối cảnh** (8 nhóm phủ test), 5,000 mẫu — đúng như rủi ro số 1 của
kế hoạch dự báo, chúng **rộng tới mức gần như không mang thông tin**:

| Điểm | recall, CI nhóm | báo động giả, CI nhóm |
|---|---|---|
| OP0 | 82.81% [54.2, 100.0] | 34.43% [8.6, 81.8] |
| OP2 | 57.81% [18.6, 85.2] | 3.28% [0.0, 14.3] |

CI mức clip (`WRONG UNIT`, không dùng để kết luận) hẹp hơn nhiều: OP0 recall [71.79, 90.12].

## 3. Luật tiền đăng ký: **không có nội dung**

Chỉ OP3 có hiệu lực so sánh, vì scene-only chỉ có một điểm hoạt động; so recall ở mức báo động giả
khác nhau là so hai thứ khác giá.

> Luật: *có nội dung nếu tại OP3 recall vượt scene-only ≥ 5 pp **VÀ** McNemar trên quyết định KAT p < 0.05.*

Thực đo: biên **+6.25 pp** (95.31% so 89.06%) **đạt**; McNemar b=12, c=22, **p = 0.1214** **không
đạt** ⇒ **không có nội dung**. Và điều này **không phụ thuộc điểm hoạt động** — không điểm nào trong
bốn điểm tách được khỏi scene-only:

| Điểm | McNemar so scene-only (quyết định KAT) |
|---|---|
| OP0 | b=14, c=9, **p = 0.4049** |
| OP1 | b=12, c=14, **p = 0.8450** |
| OP2 | b=28, c=20, **p = 0.3123** |
| OP3 | b=12, c=22, **p = 0.1214** |

Đối chứng dịch sang ngôn ngữ phát hiện (cùng 125 clip):

| Đối chứng | recall KAT | báo động giả | precision | F1 | J |
|---|---|---|---|---|---|
| scene-only, nhãn đa số từ **train** | **89.06%** | 49.18% | 65.52% | 75.50% | 39.88 |
| scene-only **oracle** (chặn trên) | 87.50% | 42.62% | 68.29% | 76.71% | 44.88 |
| hằng số "luôn báo động" | 100% | 100% | 51.20% | 67.72% | 0.00 |

**Scene-only đạt recall 89.06% — cao hơn ConvLSTM ở argmax (82.81%).** Chỉ cần biết clip thuộc nhóm
bối cảnh nào là đã báo động tốt hơn, nếu chỉ tính recall.

## 4. Phát hiện chính: tách phần phân biệt thật khỏi phần đoán theo bối cảnh

ROC-AUC gộp thưởng cho việc xếp một clip KAT của bối cảnh phần lớn KAT lên trên một clip AT của bối
cảnh phần lớn AT — đúng việc mà scene-only làm. Chặn kênh đó bằng cách **chỉ so các cặp trong cùng
một nhóm bối cảnh** (645 cặp, 6 nhóm trộn lẫn):

| Đại lượng | Giá trị |
|---|---|
| ROC-AUC **gộp** | **0.8485** |
| AUC **trong cùng nhóm bối cảnh** | **0.6426** |
| CI bootstrap cụm, đơn vị = **nhóm bối cảnh** | **[0.4923, 0.8443]** — **chứa 0.50** |
| CI mức clip, nhóm cố định (`WRONG_UNIT`) | [0.5106, 0.7680] — không dùng để kết luận |
| Phần khoảng cách trên mức ngẫu nhiên còn sống sau khi điều kiện hoá | (0.6426−0.5)/(0.8485−0.5) = **41%** |

**59% năng lực phân biệt biểu kiến là bối cảnh** (trị số điểm). CI ở đúng đơn vị độc lập — bootstrap
lại **cả nhóm bối cảnh** — là [0.4923, 0.8443] và **chứa 0.50**: dữ liệu **không loại trừ được** khả
năng phần phân biệt trong cùng bối cảnh bằng **không**. Khoảng rộng vì chỉ có **6 nhóm trộn nhãn**.
Khoảng [0.5106, 0.7680] là bootstrap **mức clip** với nhóm cố định — coi các clip cùng nền là bằng
chứng độc lập, nên chỉ in kèm nhãn `WRONG_UNIT`.

Theo từng nhóm, nơi phần lớn số cặp nằm thì phân biệt **bằng hoặc dưới mức ngẫu nhiên**:

| Nhóm | số cặp | AUC trong nhóm | Youden J trong nhóm (OP0) |
|---|---|---|---|
| 4 | 217 | **0.4954** | +18.89 |
| 1 | 132 | **0.4773** | **−13.64** |
| 3 | 270 | 0.8074 | 0.00 |
| 2 | 20 | 1.0000 | +100 |
| 7 | 4 | 1.0000 | +100 |
| 0 | 2 | 1.0000 | 0.00 |

349 trên 645 cặp (**54%**) đến từ hai nhóm mà mô hình xếp hạng ở mức ngẫu nhiên; nhóm 1 còn **dưới**
mức ngẫu nhiên. Phân biệt thật chỉ có ở nhóm 3 và ở ba nhóm quá nhỏ (26 cặp cộng lại).

**Cơ chế, thấy được từng clip.** Tỷ lệ báo động theo nhóm tương quan **r = 0.9266** với tỷ lệ KAT
thật của nhóm. Ở nhóm 0, 3, 5, 6 mô hình báo động **100% số clip** bất kể nhãn — nghĩa là ở nhóm 3
(37 clip) **toàn bộ 10 clip An toàn đều là báo động giả**, và recall 100% ở đó không đến từ việc
nhìn thấy hành vi. Ở nhóm 4 (38 clip, phần lớn An toàn) nó báo động 13.2% và recall rơi xuống
28.57%. Đây là scene-only, do một ConvLSTM thực hiện. G25 chỉ cho thấy điều này bằng McNemar; bản
này cho thấy nó bằng từng clip.

## 5. Cái giá của OP2, đọc kỹ trước khi dùng

OP2 trông hấp dẫn: precision **94.87%**, chỉ **2 báo động giả trên 61 clip An toàn**, J cao nhất
(54.53). Nhưng recall theo **loại** không an toàn cho thấy phần lợi đến từ đâu:

| Loại KAT | OP0 (τ=0.5) | OP2 (τ=0.7276) |
|---|---|---|
| WalkwayViolation | 29/32 = 90.62% | 23/32 = 71.88% |
| UnauthorisedIntrusion | 10/11 = 90.91% | 8/11 = 72.73% |
| OpenedPanel | 6/13 = 46.15% | **0/13 = 0.00%** |
| CarryOverload | 8/8 = 100% | 6/8 = 75.00% |

**OP2 không phát hiện được một clip OpenedPanel nào.** Điểm hoạt động "chính xác cao" ấy mua
precision bằng cách **bỏ trắng một trong bốn loại**. Số gộp — BA, F1, J, precision — che điều đó;
chỉ recall theo loại in ra. Trùng khớp với G21: c2 OpenedPanel là lớp bất định nhất, dịch 30.77 pp
chỉ vì đổi thiết bị.

## 6. Sàn báo động giả do **nhãn** ấn định — kiểm, và nó **không** biện hộ được cho FPR ở argmax

Bốn clip An toàn trong test trùng byte với clip Không an toàn. Mô hình tất định **phải** cho cùng
một điểm trên đầu vào trùng byte, và nó làm đúng thế:

| Clip An toàn (test) | Bản sinh đôi KAT | P(KAT) | Báo động ở OP0? |
|---|---|---|---|
| `4_te9.mp4` | `0_te23.mp4` (**trong test**) | 0.2776 — **trùng từng chữ số** với bản sinh đôi | không |
| `4_te4.mp4` | `0_tr123`, `1_tr55` (train) | 0.4604 | không |
| `5_te13.mp4` | `0_tr136` (train) | 0.4745 | không |
| `4_te7.mp4` | `0_tr138` (train) | 0.4977 | không |

Đọc được ba điều:

1. **Trần cứng hiện ra đúng như dự kiến.** `0_te23`/`4_te9` nhận **cùng** điểm 0.2776; ở OP0 cả hai
   bị gọi là An toàn ⇒ bản KAT là **một trong 11 lần bỏ sót, không thể sửa được bằng ngưỡng nào**.
   Recall 100% và FPR 0% **không thể đồng thời đạt**; recall trần khi không thêm báo động giả là
   **63/64 = 98.44%**.
2. **Ở argmax, 0 trong 21 báo động giả là tạo tác nhãn.** Giả thuyết "FPR cao vì nhãn bẩn" **sai** ở
   điểm hoạt động tiêu đề. Đã kiểm thay vì phát biểu.
3. **Nhưng cơ chế có thật, chỉ chưa vượt ngưỡng.** Ba clip bắc qua train→test đứng ở 0.4604, 0.4745,
   **0.4977** — cái cuối cách ngưỡng **0.0023**. Bản sinh đôi KAT trong train kéo chúng về phía báo
   động, chỉ chưa qua vạch. Ở OP1 và OP3 cả ba **đều** thành báo động giả (sàn FPR do nhãn 4.92%).

## 7. Điều **không** được kết luận từ bản này

- **Không** đảo, nới hay làm mềm G25. G25 trượt theo luật của chính nó; bản này xác nhận bằng một
  họ đại lượng khác, ở bốn điểm hoạt động.
- **Không** gọi OP2 là "kết quả tốt". Nó mù hoàn toàn với OpenedPanel và vẫn không tách khỏi
  scene-only (p = 0.3123).
- **Không** so với 95.81% của [17]: số đó là macro 4 lớp, không phải phát hiện nhị phân.
- **Không** kết luận trên NP-64. 64 clip của [17] toàn KAT ⇒ chỉ recall xác định (**82.81%** ở OP0);
  báo động giả, precision, F1, PR-AUC **không** xác định ở đó và không được in.
- **Không** dùng CI mức clip. Đơn vị độc lập là nhóm bối cảnh; CI nhóm rộng và phải in nguyên trạng.
- Ngưỡng OP2 chốt trên val mà val chỉ có **20 clip An toàn** ⇒ ước lượng FPR trên val rất nhiễu:
  đặt ngân sách 10% nhưng test ra 3.28%. Ngưỡng chuyển sang test **không** chính xác, chỉ cùng hướng.

## 8. Câu hỏi chưa giải quyết

1. AUC trong nhóm 0.6426 có CI đơn vị nhóm [0.4923, 0.8443] — chứa 0.50. Với n = 8 nhóm (6 trộn
   nhãn), có cách nào
   tăng lực mà không phải gán nhãn bối cảnh thật (thay vì proxy KMeans) hay không?
2. Nhóm 3 là nơi duy nhất có phân biệt trong-nhóm đáng kể (AUC 0.8074, 270 cặp) nhưng Youden J
   trong nhóm **bằng 0** ở argmax. Nghĩa là thứ tự xếp hạng tốt mà ngưỡng đặt sai chỗ cho nhóm đó.
   Ngưỡng **theo từng nhóm bối cảnh**, chốt trên val, có hợp lệ trong cùng kế hoạch hay cần tiền
   đăng ký mới?
3. OpenedPanel: 0/13 ở OP2, 6/13 ở OP0, và dịch 30.77 pp vì đổi thiết bị ở G21. Lớp này có vấn đề
   riêng (13 clip, có thể đồng nhất về bối cảnh) hay chỉ là thiếu dữ liệu?
4. ~~BA có còn nên là đại lượng tiêu đề cho tác vụ nhị phân không?~~ **ĐÃ ĐÓNG 11/10/2026** —
   tác giả chốt **GIỮ BA**, không đổi sang recall-ở-ngân-sách-báo-động. Ghi ở mục 8.3.9 và ở khối
   ngoại lệ thứ ba trong `00-trang-thai-bien-soan.txt`.
5. `clip_scene_map_691.json` là KMeans(8) trên nền tĩnh, một **proxy**. Nếu bối cảnh thật mịn hơn 8
   nhóm thì AUC trong nhóm 0.6426 vẫn **ước lượng cao** phần phân biệt thật. Chưa có metadata camera.

## 9. Sửa chữ đã làm trong `report/vn/` (không số mới, không cổng mới)

| Tệp | Sửa | Kiểm |
|---|---|---|
| `chuong-06-danh-gia.txt` | tiêu đề §6.15 thêm NP-125-D; **§6.15.4 mới**; §6.15.4 cũ → §6.15.5, kết luận thứ tư thêm 0.8485 → 0.6426 / 59% | 2340 → 2481 dòng |
| `00-tom-tat.txt` | mệnh đề nhị phân cuối viết lại theo mục tiêu phát hiện (recall 82.81% @ báo động giả 34.43%, precision 71.62%, PR-AUC 0.876, BA 74.19%) | 212 → 224 dòng |
| `chuong-08-ket-luan.txt` | **§8.3.9 mới** — đại lượng tiêu đề của tác vụ nhị phân là hạng mục **mở**, kèm ba ràng buộc (a)(b)(c) nếu lượt sau đổi | 476 → 507 dòng |
| `00-trang-thai-bien-soan.txt` | **ngoại lệ hậu-đóng-băng thứ ba** (0 lần chạm test, ngân sách còn 1, bảng cổng không đổi); bảng tiến độ dựng lại từ đếm máy: 11 tệp = 9,406 dòng / 114,257 từ | 12/12 hàng khớp |
| `chuong-05-trien-khai-bien.txt` | ngắt một dòng 185 ký tự (vi phạm quy ước ≤ 100 có từ trước) | 1338 → 1339 dòng |
| `chuong-06-danh-gia.txt` (vòng 2) | **quyết định vị trí §6.15**: thân chương, không xuống phụ lục — quy ước (5) mới ở §6.1.5, **kết luận thứ năm** ở §6.16, điều **(e)** trong danh sách "không chứng nhận" | 2452 → 2481 dòng |
| `chuong-08-ket-luan.txt` (vòng 2) | §8.3.9 ghi quyết định **giữ BA** làm đại lượng tiêu đề | 504 → 507 dòng |
| ba tệp trên | **sửa đơn vị khoảng tin cậy của AUC trong nhóm**: in [0.4923, 0.8443] (đơn vị nhóm, chứa 0.50) thay cho [0.5106, 0.7680]; khoảng mức clip giữ nhãn `WRONG_UNIT` | §6.15.4, tóm tắt, §8.3.9(c) |

### 9b. Vòng 3 — khai báo đại lượng / đường cơ sở (11/10/2026, không số mới)

Yêu cầu tác giả: "các số liệu kết quả chưa làm rõ metric, baseline cụ thể". Chỉ **sửa chữ**: mọi
trị số đặt vào đều được chép từ một dòng đã có trong `report/vn/` (ch03 Bảng 3.4, ch05 Bảng 5.5,
ch06, tóm tắt). Không phép đo mới, không chạm test (0 lần), ngân sách vẫn **2 / 1 tiêu / 1 còn**.

| Tệp | Sửa | Kiểm |
|---|---|---|
| `chuong-06-danh-gia.txt` | **§6.1.6 mới + Bảng 6.1b**: 11 con số tiêu đề × 4 cột (đại lượng và định nghĩa / giao thức và tập đo / đường cơ sở kèm **trị số** / nguồn `[ĐO]` `[ĐỌC]` `[SỐ HỌC]`); §6.1 "năm → sáu quy ước"; §6.1.4 thêm hai trùng ký hiệu (**+16.9 pp**; tên "đường sàn quy tắc đếm" = 63.3% làn A so với 67.2% làn B, **không trừ chéo**); khối mở đầu gắn đại lượng + đường cơ sở cho cả 5 con số; chú thích Bảng 6.3, 6.4, 6.8 nêu đại lượng cột Δ và vật bị trừ; §6.6.1 đổi tên; §6.13 thêm khối ngưỡng đăng ký trước từng cổng | 2481 → 2613 dòng |
| `00-tom-tat.txt` | tiêu đề quy ước thêm "bốn thứ đi kèm mỗi con số"; biên tiêu đề viết thành **81.2% so với 64.4% macro** ở cả tóm tắt ngắn và mở rộng | 224 → 229 dòng |
| `chuong-08-ket-luan.txt` | §8.2 "+20.9 pp" thêm sàn **63.3%** và lời cấm trừ chéo với 67.2%; tách mẫu số đầu phân loại (**1.7%** tổng giai đoạn mô hình so với **0.3%** khoảng quyết định) ở hai chỗ, thay dải "0.3–1.7%"; giải mã **31.3% khoảng quyết định** (không phải "phần tính toán" — Bảng 5.5 in "—" ở cột đó); §8.3.9 thêm giao thức NP-125-D và đối chứng scene-only **recall 89.06% / báo động giả 49.18%**; 0.50 gắn nhãn mức ngẫu nhiên theo kiến tạo | 507 → 516 dòng |
| `chuong-07-phan-tich-that-bai.txt` | §7.2 ba cổng in cả hai số bị trừ và ngưỡng đăng ký trước (G11 75.2% so với 74.1%, ~3 pp; G14 83.8% so với B1w 84.4%, +3 pp; G18 so với B1w 84.4%, G18a +3.0 / G18b +2.0); "+16.9 pp" nêu macro 4 lớp và 64.4% ở §7.2.3 và §7.3; **Bảng 7.3 thêm 2 hàng** (+16.9 pp; tên "đường sàn quy tắc đếm") → "ba trùng số" thành "năm"; "Quy tắc trình bày" ở §7.8 thêm hai thuộc tính đường cơ sở + nguồn, trỏ §6.1.6 | 996 → 1023 dòng |
| `00-trang-thai-bien-soan.txt` | bảng tiến độ dựng lại từ đếm máy: 11 tệp = **9,579 dòng / 116,892 từ**; ghi chú miễn trừ Ch6 53 → **78 hàng bảng rộng**, thêm Bảng 6.1b | 12/12 hàng khớp |

Kiểm quy ước sau mỗi lô: **(1) 0 dòng > 100 ký tự** ngoài hàng bảng dữ liệu rộng của Ch6 (78 hàng,
101–125 ký tự, thuộc miễn trừ) và 2 dòng đường dẫn của Ch4; **(2) 0 dòng mở đầu bằng token
`^[0-9]\.[0-9]` không phải tiêu đề mục** trên cả 11 tệp nộp. Đếm bằng ký tự Python, không bằng
`awk length()`.

Kiểm tự động sau khi sửa: **148 trị số trong §6.15.4 đối chiếu JSON thô** (132 khớp trực tiếp, 16
còn lại là số mục hoặc trị số dẫn xuất — đã kiểm tay từng cái: val 103 clip, 349/645 = 54% cặp
trong-nhóm ở hai nhóm mức ngẫu nhiên, điểm trùng byte 0.2776/0.4604/0.4745/0.4977, trần 63/64 =
98.44%, Youden J trong nhóm 1 = −13.64 pp); **quy ước đánh dấu: 0 vi phạm trên cả 12 tệp**;
mọi tham chiếu §6.15.1–§6.15.5 phân giải được; **286/286 test qua**.

Câu hỏi 4 ở mục 8 **vẫn mở** — §8.3.9 chỉ *ghi nhận* nó là hạng mục phát triển, không quyết định.

## 10. Sửa theo rà soát mã (cùng chân zero-touch, không số đo mới)

`code-reviewer` nêu một phát hiện HIGH: khoảng tin cậy của AUC trong cùng nhóm bối cảnh được tính
bằng bootstrap **mức clip** trong khi đơn vị độc lập là **nhóm bối cảnh** — đúng lỗi mà cả luận văn
dùng để bác giao thức của công trình gốc. Đã sửa trước khi làm bất cứ việc gì khác:

| Tệp | Sửa |
|---|---|
| `eval_utils/detection_metrics.py` | `within_scene_auc` trả `ci_group_resample` (bootstrap lại cả nhóm) = **[0.4923, 0.8443]**, giữ `ci_clip_resample_WRONG_UNIT` = [0.5106, 0.7680]; F1 **không xác định** (nan) khi không có clip dương thật, F1 = 0 khi có dương mà không gắn cờ gì; ghi rõ quy tắc phá thế hoà về **ngưỡng nhỏ hơn**; `perfect_detection_possible` **tính** từ audit thay vì hằng số; `corrcoef` trả nan khi một cột là hằng | 
| `run_binary_detection_scorecard.py` | thêm `recall_ci_n_valid_draws` / `false_alarm_ci_n_valid_draws`; tiền tố tên clip so bằng tuple thay vì `in '0123'` |
| `eval_utils/tests/test_detection_metrics.py` (mới, 149 dòng, 8 test) | đếm tay, F1 nan/0, lát NP-64 chỉ định nghĩa recall, `pick_thresholds` chỉ đọc val và phá thế hoà, ngân sách OP2, khớp báo động OP3, `duplicate_floor` |
| `eval_utils/tests/test_within_scene_discrimination.py` (mới, 108 dòng, 5 test) | mẫu **thuần bối cảnh**: AUC gộp > 0.85 mà AUC trong nhóm đúng 0.5; nhóm một nhãn bị bỏ; khoảng đơn vị nhóm rộng hơn khoảng mức clip > 3 lần |

Chạy lại bộ đếm **hai lần** từ cùng `Pt` đã lưu, so JSON dàn phẳng: **không trị số đo nào đổi**;
lần thứ hai cho **0 mục khác biệt**. Ngân sách chạm tập kiểm thử vẫn **2 đăng ký / 1 tiêu / 1 còn**.
Kết luận đi theo hướng **bất lợi hơn**: khoảng đúng đơn vị **chứa 0.50**, nên không loại trừ được
khả năng phần phân biệt trong cùng bối cảnh bằng không. Đó là bản đã đưa vào `report/vn/`.

## 11. Vòng 4 — bảng tổng hợp so đường cơ sở (12/10/2026, không số mới)

Phản hồi của tác giả: báo cáo có đủ bảng so sánh *cục bộ* nhưng **không có một bảng tổng hợp** nào
cho phép đọc "kết quả của chúng ta so đường cơ sở nào, biên bao nhiêu, ngưỡng đăng ký trước bao
nhiêu, phán quyết gì" ở một chỗ. Đã bổ sung.

| Tệp | Sửa |
|---|---|
| `report/vn/chuong-06-danh-gia.txt` §6.16 | thêm **Bảng 6.9** — 19 hàng so sánh, ba khối (iSafetyBench LSO-1100; Unsafe-Net làn C; NP-125 hậu đóng băng), 5 cột: phép so sánh (mục) / kết quả / đường cơ sở + trị số + nhãn nguồn / biên và ngưỡng đăng ký trước / phán quyết kèm CI và p |
| `report/vn/00-trang-thai-bien-soan.txt` | Ch6 2613→**2677** dòng, 32,338→**33,221** từ; Tổng 9579→**9643** dòng, 116,892→**117,775** từ; ghi chú miễn trừ hàng bảng rộng 78→**111** hàng, nêu tên Bảng 6.9 |

Chân zero-touch: **mọi ô của Bảng 6.9 được sao từ Bảng 6.1–6.8 và các mục 6.12, 6.15 của chính
chương**. Hai ô duy nhất là phép trừ hai trị số đã công bố cùng giao thức (E0 80.3% = 83.8 − 3.5 của
G15; +4.25 pp = 74.19 − 69.94 của G25). Không phép đo mới, không chạm tập kiểm thử — ngân sách vẫn
**2 đăng ký / 1 tiêu / 1 còn**.

Kiểm quy ước sau khi thêm: 0 dòng > 125 ký tự trong Ch6 (đếm ký tự bằng Python, không bằng
`awk length()`); 0 dòng mới bắt đầu bằng `^[0-9]\.[0-9]` (mọi hàng bảng mở đầu bằng hai dấu cách).

Điều Bảng 6.9 làm lộ rõ, và đó là mục đích của nó: đọc dọc cột phán quyết, **hai** mã ĐẠT trong
toàn chương là G10 và G12 — cả hai là phán quyết về **phương pháp**. Không một biên **kiến trúc**
nào có khoảng tin cậy ở đơn vị cụm đúng loại trừ 0.

Câu hỏi còn mở sau vòng này:

1. Có nên nhân bản Bảng 6.9 (dạng rút gọn, chỉ cột biên và phán quyết) vào §8.1 để chương kết
   luận tự đứng được không, hay để một tham chiếu chéo là đủ.
2. Hai sửa văn bản đã đề xuất **vẫn chờ tác giả cho phép**: hiệu đính Ch1 §1.6 dòng 325–327 và
   Ch2 §2.2 dòng 161–165 về 2,262 hộp / "bị giới hạn vào xương sống đóng băng"; và mục **§6.1.7**
   phát biểu quan hệ ba làn ở một chỗ.

## 12. Vòng 5 — Bảng 8 công trình gốc (tốc độ) + nhóm cổng theo câu hỏi (13/10/2026, không số mới)

Bằng chứng mới do tác giả cung cấp: **Bảng 8** của công trình gốc — ConvLSTM 7.72 s / 0.58 s;
Unsafe-Net 1.87 s / 0.14 s. Ba trị số 7.72 / 0.58 / 7.72 **không có trong `paper/paper.txt`** đã
trích (grep `7.72` = 0 hit); chỉ `1.87` (dòng 37) và `0.14` (dòng 34) có. Hàng ConvLSTM là tư liệu
mới.

Số học trên hai cột họ in:

| Hệ | Thời lượng TB (s) | Thời gian NDHĐ (s) | s xử lý / s video | Nhanh hơn thực |
|---|---|---|---|---|
| ConvLSTM | 7.72 | 0.58 | 0.075130 | 13.31× |
| Unsafe-Net | 1.87 | 0.14 | 0.074866 | 13.36× |

Tỉ lệ thời lượng 4.128 ≈ tỉ lệ thời gian 4.143; thông lượng lệch **0.35%**. ⇒ **toàn bộ "tăng tốc
4.14×" là hệ quả độ dài đầu vào, không phải kiến trúc.** Đây là cặp song sinh của G23: cùng tầng
giảm khung, công trạng **độ chính xác** đo được ±0.13 pp (p = 1.000), công trạng **tốc độ** là
phép chia cho độ dài.

Bốn hệ quả ghi vào §6.15.2:

1. Hai hàng đo trên **hai đầu vào khác nhau** ⇒ không phải so sánh có đối chứng theo §6.1.6; đại
   lượng duy nhất so được qua hai hàng là thông lượng, và nó bằng nhau.
2. Bảng 8 **không tính chi phí tầng YOLO v4** — tầng duy nhất Unsafe-Net thêm vào — tầng phải quét
   **toàn bộ** ~232 khung (7.72 s @ 30 fps, suy từ ví dụ 3.5 s = 105 khung của họ). So sánh đầy đủ:
   0.58 s một tầng ↔ (YOLO v4 trên 232 khung) + 0.14 s hai tầng.
   Cách đọc có lợi cho họ, đã nêu trong báo cáo vì nó đúng: với hai camera IP chạy liên tục, bộ
   phát hiện **dù sao cũng đã trả tiền**, nên 0.14 s là chi phí **biên** mỗi sự kiện — lợi ích
   triển khai thật. Nhưng dưới cách đọc ấy Bảng 8 vẫn không phải "ConvLSTM so Unsafe-Net".
3. Hai chỗ không khớp nội bộ: (a) TB có trọng số từ Bảng 2 của họ = **8.64 s** trên đúng 64 clip
   test mà ConvLSTM được chấm (8.24 s / 125 clip; 8.43 s / 516 clip KAT; 8.29 s / 691 clip), còn
   Bảng 8 ghi 7.72 s — **thấp hơn 10.7%**; (b) Bảng 8 hàm ý giữ **24.2%** khung, ví dụ của họ giữ
   **38.1%** (40/105), và ví dụ ấy còn **tách thành 3 đoạn** ⇒ 0.14 s là **mỗi đoạn**, độ trễ mỗi
   video nguồn = k × 0.14 s với k không được báo ⇒ con số mỗi-video **không suy ra được**.
4. Hai GPU GTX 1080 + i9-9900 ⇒ **tham chiếu tài liệu** (§6.1.6), **không** đường cơ sở, và
   **cấm trừ chéo** với số độ trễ Jetson Orin Nano Super ở Chương 5.

Phát biểu được phép, viết hẹp: *Bảng 8 như đã công bố không chứng minh tầng giảm khung làm hệ nhanh
hơn trên một đơn vị đầu vào.* Luận văn **không** kết luận hệ của họ không thực-thời, và không kết
luận gì về tính trung thực phép đo.

### Thay đổi tệp

| Tệp | Thay đổi |
|---|---|
| `report/vn/chuong-06-danh-gia.txt` §6.15.2 | thêm khối **Bảng 8** cuối mục: bảng 4 hàng (2 cột [ĐỌC] + 2 cột [SỐ HỌC]) + 7 đoạn phân tích, kèm cách đọc có lợi cho họ và điều cấm so với Chương 5 |
| `report/vn/chuong-06-danh-gia.txt` §6.13 | sau dòng "Tổng: tám cổng…" thêm **bảng nhóm cổng theo câu hỏi**: phương pháp (G10, G12) 2/2 ĐẠT — kiến trúc (G11, G14, G15, G16, G18) 0/5 — dung lượng (G13) 0/1, kèm đoạn "kết quả âm đã tái lập" |
| `report/vn/chuong-06-danh-gia.txt` Bảng 6.1b | thêm hàng `0.58 s so 0.14 s (6.15.2)`; mở rộng hàng `2/8 cổng (6.13)` bằng ba câu hỏi 2/2 · 0/5 · 0/1 |
| `report/vn/chuong-06-danh-gia.txt` Bảng 6.9 | thêm hàng `tốc độ, Bảng 8 (6.15.2)` dưới hàng 95.81%; thêm đoạn đóng "hai điều về cách đếm bảng này" |
| `report/vn/00-trang-thai-bien-soan.txt` | Ch6 2677→**2776** dòng, 33,221→**34,656** từ; Tổng 9643→**9742** dòng, 117,775→**119,210** từ; hàng bảng rộng 111→**121** |

Chân zero-touch: hai cột dẫn xuất của bảng Bảng 8 là **phép chia hai trị số họ in**; số khung 232
và 56 là **7.72 × 30** và **1.87 × 30** với 30 fps suy từ chính ví dụ của họ; TB Bảng 2 là **trung
bình có trọng số trên bảng đếm lớp của họ**. Không phép đo mới, không chạm tập kiểm thử — ngân sách
vẫn **2 đăng ký / 1 tiêu / 1 còn**.

Kiểm quy ước sau khi thêm: max 125 ký tự trong Ch6 (không dòng nào > 125); hai dòng mới từng bắt
đầu bằng `4.13` và `8.29` đã được ngắt lại ⇒ **0** dòng vi phạm `^[0-9]\.[0-9]`.

Tác giả **từ chối** đề xuất in nhãn `(ĐƯỜNG SÀN — thấp là tốt)` vào từng ô sàn (vòng 4, mục 3);
ghi chú hai chiều ở đầu Bảng 6.1 giữ nguyên làm cơ chế duy nhất.

Câu hỏi còn mở sau vòng này:

1. (giữ từ vòng 4) Có nên nhân bản Bảng 6.9 rút gọn vào §8.1 hay để tham chiếu chéo.
2. Hiệu đính Ch1 §1.6 dòng 325–327 và Ch2 §2.2 dòng 161–165 về 2,262 hộp **vẫn chờ tác giả cho
   phép**; mục **§6.1.7** quan hệ ba làn đang được giải thích cho tác giả trước khi viết.
3. Có nên nêu Bảng 8 ở Chương 2 (chỗ mô tả 95.81% / 0.14 s của công trình gốc) hay chỉ ở §6.15.2.
   Hiện chỉ ở §6.15.2.
