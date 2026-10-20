# Chiến lược thực nghiệm vòng 3 — thang tháp đặc trưng, đăng ký trước khi chạy

Ngày: 2026-10-15. Căn cứ: 38 dòng hậu kiểm vòng 1–2 (`sweep_results/posthoc_sensitivity/`),
bốn báo cáo nghiên cứu trong `plans/reports/researcher-2026-10-15-*.md`.

## 1. Ba phát hiện buộc phải xuất phát từ đó

1. **Khoảng trống nằm ở biểu diễn, không ở đầu thời gian.** ViT-L/14 + đầu không có mô hình thời
   gian: 88.58%, là dòng duy nhất trong 38 dòng vượt được cả hai điều kiện (+5.33 pp,
   CI [+0.17, +11.65]). Cả bốn đầu thay thế trên B/32 (meanpool 84.89, BiGRU 84.76, TCN 84.67,
   GRU 84.27) đều bằng hoặc hơn transformer đã công bố 83.25 — tức thay đầu không mua được gì.
2. **Xương sống video không phải lối ra.** VideoMAE-base tinh chỉnh Kinetics: 80.62%, kém tháp ảnh
   đông cứng 2.37 pp. Nước đi hiển nhiên "dùng mô hình video" đã bị chính số đo phủ nhận.
3. **Giới hạn thống kê là 151 video nguồn, không phải compute.** Thêm GPU không thu hẹp được
   khoảng tin cậy. Ước lượng công suất: ở 151 nguồn, công suất cho hiệu ứng +5 pp chỉ khoảng 65%;
   để đạt 80% cần cỡ 300–400 nguồn. Vậy vòng 3 **không** đặt mục tiêu làm cho kết quả "có ý nghĩa".

## 2. Quyết định: thang tháp cỡ ViT-B/16, đông cứng

Lý do chọn: đây là việc duy nhất vừa kiểm định được phát hiện (1) vừa có thể **đổi được kết luận**
của luận văn thay vì chỉ trang trí nó. ViT-L/14 không triển khai được, nên 88.58% mãi chỉ là chặn
trên. Lớp B/16 nặng gấp 4 lần B/32 về GFLOPs (17.6 so với 4.4) nhưng vẫn nằm trong khả năng của
Orin Nano Super, nên **nếu** một tháp B/16 đạt 86–88% thì đó là ứng viên đầu bảng thật, không còn
là chặn trên. Điều kiện "nếu" đó phải do phép đo trên thiết bị quyết định, không do suy đoán.

Chín tháp, chốt danh sách trước khi chạy (mẫu số cho hiệu chỉnh BH là 7 tháp không tính 2 control):

| Tháp | Vai trò | Ghi chú |
|---|---|---|
| `clipB32_control` | chốt giải mã | phải ra lại đặc trưng đã công bố |
| `clipL14_control` | chốt thứ hai | phải ra lại ~87.4% của artifact E2 |
| `clipB16` | bậc giữa hiển nhiên | cùng dữ liệu huấn luyện, chỉ khác patch |
| `openclipB16_dc1b`, `dfn2bB16`, `siglip2B16`, `metaclipB16`, `openclipB16_l2b`, `eva02B16` | B/16 dữ liệu tốt hơn | |
| `mobileclip_s1` | nhỏ **hơn** B/32 | nếu thắng là thắng không mất gì về triển khai |

## 3. Luật quyết định, đăng ký trước khi có số

* Mọi dòng huấn luyện ở máy, `OMP_NUM_THREADS=4`, so với control **83.25%**, không bao giờ so 84.2%.
* Hợp thức kép như tiền lệ G15: biên độ ≥ **1.46 pp** (dải trôi đã đo) **và** CI bootstrap theo cụm
  nguồn loại trừ 0.
* Thêm hiệu chỉnh **Benjamini–Hochberg q = 0.05** trên 7 tháp. Vòng 2 so 38 dòng với một control mà
  không hiệu chỉnh; vòng 3 không được lặp lại chuyện đó.
* **Chốt triển khai:** một tháp chỉ được xem là ứng viên đầu bảng sau khi đo trên Jetson đạt
  ≥ 3 quyết định/giây. Chưa đo thì kết quả vào báo cáo như **chặn trên của khoảng trống biểu diễn**.
* Giữ nguyên đại lượng đầu bảng là balanced accuracy (mục 8.3.9). Các phép đo bất đối xứng
  (R@FAR5, AUROC, AP, F2) vào vai trò bổ trợ, không thay thế.

## 4. Dự đoán ghi trước (để kiểm chính mình)

Theo ước lượng công suất ở mục 1.3, dự đoán là **phần lớn tháp B/16 sẽ vượt dải 1.46 pp nhưng CI
vẫn chứa 0**, giống y như `lr_3e-4` (+2.30 [−0.32, +6.03]) và `width_d64` (+1.74 [−0.36, +4.99]) ở
vòng 1. Nếu điều đó xảy ra thì đó không phải thất bại của vòng 3: đó là bằng chứng định lượng cho
luận điểm rằng trần của nghiên cứu này là 151 nguồn, và nó thuộc về chương kết luận.

Các con số "linear probe" trong báo cáo tháp phần lớn là **suy ra từ zero-shot**, không phải đo. Dự
báo "+2–3 pp trên tác vụ này" vì vậy là phỏng đoán. Phát hiện (2) là phản ví dụ sẵn có cho giả định
thứ hạng ImageNet chuyển sang video: Kinetics mạnh mà vẫn kém. Nên vòng 3 **đo, không dự báo**.

## 5. Loại bỏ, kèm lý do

* **Chưng cất tháp:** cần ≥ 1M ảnh (CLIP-KD 3.6M, DIME-FM 40M). Corpus 17.6k khung nhỏ hơn
  100–200 lần. Không khả thi, không phải vấn đề chỉnh tham số.
* **Chưng cất riêng phần đầu:** thông tin không có trong đầu vào của học sinh thì không khôi phục
  được ở đầu ra (bất đẳng thức xử lý dữ liệu). Chặn lý thuyết.
* **Xương sống video:** đã đo, kém 2.37 pp.
* **Gắn thêm bbox:** thang tháp chỉ đọc nhánh toàn khung. Nhãn hộp chỉ cần cho câu hỏi quy trách
  nhiệm "phân rã thực thể vô dụng" so với "bộ nhận diện quá yếu", và cho câu đó thì tập chẩn đoán
  ~100 clip là đủ. Độc lập với vòng 3, ưu tiên thấp hơn.

## 6. Việc cần làm

- [x] ~~Tải corpus lên Kaggle~~ — chạy ở máy thay thế, đo được nên không cần Kaggle cho bước này
- [x] Bước 1 giải mã một lần (10.5 phút); Bước 2 `clipB32_control` đạt chốt trùng khít bit
- [x] Bước 3: 8/8 tháp của họ đã đo ở máy; mỗi file qua `verify-and-install` trước khi dùng
- [x] Đầu chạy ở máy 4 luồng; BH q=0.05 đã áp (`analyze_round3_tower_bh.py`)
- [x] Phép đo bất đối xứng — xem mục 8.4; R@FAR5 không đo được, AUROC/AP đổi thứ hạng tháp
- [x] Đo độ trễ B/16 trên Jetson (việc của thiết bị, không phải của Kaggle) — 19/10/2026,
      sáu lần đo đều có phán quyết cửa sổ sạch. Kết luận: B/16 **vượt** ngưỡng 3 qđ/giây
      (clipB16 4,374 qđ/giây; siglip2B16 4,394 qđ/giây), tức dự đoán P3 bị **bác bỏ**.
      `mobileclip_s1` trượt kiểm tra tương đương FP16 (cos 0,932). Đo tiếp ở FP32 ngày
      20/10/2026 theo chỉ đạo: **cũng trượt** (cos 0,933) — vậy nguyên nhân không phải độ chính
      xác số mà là lỗi chuyển đổi đồ thị của TensorRT 10.3.0, và tháp này không triển khai được ở
      bất kỳ độ chính xác nào trên thiết bị. Thời gian FP32 của nó là 151,8 ms/quyết định, tức
      **chậm hơn B/32 2,83 lần** dù ít hơn 4,08 lần tham số; mốc đối chứng B/32 FP32 cùng phiên
      là 53,6 ms. Nhịp trong vòng lặp (đo lại tối 20/10; lần đầu mất vì bo mạch được tắt theo yêu cầu giữa chuỗi đo, không phải sự cố thiết bị):
      281,495 ms/quyết định = **3,552 qđ/giây**, chỉ vượt ngưỡng 3 qđ/giây 1,18 lần, so với
      4,895 qđ/giây của mốc đối chứng cùng phiên — tức chậm hơn 1,378 lần về nhịp. Mốc đối chứng
      tái lập qua một lần khởi động lại trong vòng 1% (204,293 vs 202,717 ms). Cơ chế "tỷ lệ thang 585×" tôi nêu trước đó đã bị chính phép đo FP32 **bác bỏ** và
      được giữ lại trong runbook như một lời giải thích sai, có ích cho yêu cầu Y5.
      Chi tiết và toàn bộ phần bác bỏ: `deployment/JETSON-RUNBOOK-2026-10-19.md`.
- [ ] Viết mục 4.5.7 (đang hoãn) và bổ sung baseline ngoài vào chương 2/6

## 7. Câu hỏi chưa giải quyết

* iSafetyBench gốc dùng chia ngẫu nhiên hay chia tách nguồn? Báo cáo baseline không xác định được.
  Nếu là ngẫu nhiên thì 43.8% của Ovis2-8B **không** so trực tiếp được với số của luận văn.
* **ĐÃ TRẢ LỜI (20/10) — và câu trả lời bác bỏ chính câu hỏi.** Tôi từng nêu rằng hai con số
  công bố (289.672 ms/quyết định = 3.452 qđ/giây; tiền xử lý = 73.4% phần tính toán) **không tái
  lập được**, và hỏi nên gửi GVHD như đề nghị sửa số hay chỉ kèm ghi chú. Cả hai phương án đều
  không cần: đó là **lỗi quy chiếu của tôi**, không phải lỗi của báo cáo. Bảng 5.2 trong
  `chuong-05-trien-khai-bien.txt` đã trình bày hai con số ấy *chỉ* ở cột **TRƯỚC** (commit
  9c1e219, đo 01/10), còn khẳng định hiện hành của luận văn là cột **SAU** (6319b21):
  **167.291 ms/quyết định = 5.978 qđ/giây**, tiền xử lý 50.867 ms = 46.6%. Phép đo 19/10 trên
  thiết bị **tái lập đúng cột SAU, mọi giai đoạn lệch dưới 2%**: nhịp 169.843 vs 167.291 (+1.5%),
  tiền xử lý 50.707 vs 50.867 (−0.3%), tháp trong vòng 57.441 vs 58.052 (−1.1%), tính toán
  109.922 vs 109.114 (+0.7%), hệ số hoạt động 0.630 vs 0.631, tỷ lệ tiền xử lý 46.1% vs 46.6%.
  Vậy **không sửa số nào, không gửi gì cho GVHD**; thay vào đó đây là bằng chứng Y6 mạnh nhất có
  được cho phần triển khai — cùng một khẳng định, đo lại trên cùng bo mạch sau 16 ngày và nhiều
  lần khởi động lại. Chỗ duy nhất còn nêu 289.672 như kết quả hiện hành là
  `chuong-05-trien-khai-bien-ghim-xung-nhip.txt`, vốn đã là tệp lưu trữ không nộp — và đây là thêm
  một lý do phải giữ nó ngoài bản nộp. Dự đoán P3 sai cũng vì cùng lỗi quy chiếu ấy: nó tính từ
  cột TRƯỚC đã bị thay thế.
* ICC thực tế theo nguồn chưa ước lượng; ước lượng công suất 300–400 nguồn phụ thuộc vào nó.
* **Đã trả lời một phần (16/10):** hai kho LAION (`openclipB16_dc1b`, `openclipB16_l2b`) phát hành
  ở dạng `open_clip`, `config.json` không có khoá `model_type`, nên `AutoModel` từ chối. Đã chuyển
  hai tháp này sang backend `open_clip`. Tức 5 trong 9 tháp cần `open_clip_torch`, không phải 3.
  `dfn2bB16`, `eva02B16`, `mobileclip_s1` vẫn chưa xác nhận tải được.

## 8. Kết quả đã đo, toàn họ tháp (16/10/2026)

Chốt giải mã đạt trước khi đo tháp: cache 1,100 clip tái tạo tensor công bố **trùng khít bit**
(`max|diff| = 0.000e+00`, 0/17600 khung dưới cosine 0.9999). Nên mọi biên độ dưới đây là hiệu ứng
tháp thuần, không lẫn hiệu ứng giải mã — đúng vai của chốt công bằng G14.

### 8.1 Dòng chính: đầu transformer, tức đồ thị B1 đang triển khai

| Tháp | BA | Biên so 83.25% | CI cụm nguồn | p bootstrap | Phán quyết |
|---|---|---|---|---|---|
| `siglip2B16` | 86.39% | +3.14 pp | [−1.53, +8.98] | 0.222 | chỉ vượt dải |
| `metaclipB16` | 86.34% | +3.09 pp | [−0.86, +7.49] | 0.128 | chỉ vượt dải |
| `clipB16` | 85.82% | +2.58 pp | [−2.32, +8.51] | 0.320 | chỉ vượt dải |
| `openclipB16_l2b` | 84.36% | +1.12 pp | [−2.59, +5.18] | 0.566 | trong dải trôi |
| `openclipB16_dc1b` | 84.30% | +1.05 pp | [−2.60, +4.48] | 0.568 | trong dải trôi |
| `mobileclip_s1` | 83.71% | +0.47 pp | [−4.71, +6.49] | 0.929 | trong dải trôi |
| `dfn2bB16` | 83.68% | +0.43 pp | [−4.36, +6.41] | 0.952 | trong dải trôi |
| `eva02B16` | 83.65% | +0.41 pp | [−4.33, +6.99] | 0.934 | trong dải trôi |

Cao nhất cả vòng: `siglip2B16` + đầu meanpool **87.25%** (+4.00 pp, CI [−0.57, +9.99], p = 0.093).

### 8.2 Phán quyết: không tháp nào sống sót luật đăng ký trước

Ngưỡng BH hạng 1 là 0.05/8 = **0.00625**; p nhỏ nhất của cả hai họ là **0.093**. Không tháp nào
gần ngưỡng. **Dự đoán ghi ở mục 4 đúng hoàn toàn**: biên vượt dải, CI chứa 0, hiệu chỉnh xong thì
không còn gì. Script: `analyze_round3_tower_bh.py`, kết quả `posthoc_sensitivity/round3_tower_bh.json`.

Ba điều phải khai báo, đã ghim trong docstring của script:

* **Mẫu số là 8, không phải 7.** Mục 2 viết "7 tháp" nhưng bảng của chính nó liệt kê 10 dòng với 2
  control, tức 8 tháp không-control. Đó là lỗi đếm khi đăng ký. Dùng **8** — lớn hơn, khó vượt hơn.
* **Một dòng mỗi tháp.** Dòng chính là đầu transformer vì đó là đồ thị đang triển khai, và nó
  **không** phải đầu tốt hơn ở mọi tháp (meanpool thắng 4/8), nên luật không chọn theo kết quả.
  Họ meanpool hiệu chỉnh riêng, cũng không ai sống sót.
* **p lấy từ đúng bộ draw của CI** (`default_rng(0)`, 5000 draw, cùng luật loại draw mất lớp), hai
  phía: `p = 2·min(P(d* ≤ 0), P(d* ≥ 0))`, chặn dưới 1/B.

### 8.3 Ba phát hiện ngoài dự kiến

1. **Thứ hạng zero-shot không chuyển sang tác vụ này.** DataComp-1B (+1.05) và DFN-2B (+0.43) có
   zero-shot ImageNet cao nhất nhưng ở đây kém nhất, thua cả `openai/clip-vit-base-patch16` thường
   (+2.58). Dự báo "+2–3 pp nhờ dữ liệu tiền huấn luyện tốt hơn" ở mục 4 — mà chính mục 4 đã nói là
   suy ra từ zero-shot chứ không phải đo — nay bị số liệu phủ định. Cùng chiều phản ví dụ VideoMAE.
   Đây là bằng chứng thứ hai cho một luận điểm: không đại lượng ngoài nào đoán được thứ hạng ở đây.
2. **`mobileclip_s1`, nhỏ hơn B/32, bằng control** (+0.47 pp, p = 0.93). Một tháp rẻ hơn không mất
   gì. Về triển khai đây là kết quả hữu dụng nhất của vòng: trong dải này chi phí tính toán và độ
   chính xác gần như không đổi nhau được.
3. **Trong cùng cỡ B/16, khoảng cách giữa các tháp (83.65–86.39) rộng bằng cả bậc B/32 → B/16.**
   Nên "tăng cỡ tháp" không phải một trục lập kế hoạch được; danh tính tháp quan trọng ngang cỡ.

Thang theo cỡ vẫn đơn điệu ở ba bậc cùng dòng OpenAI — 83.25% (B/32) < 85.82% (B/16) < 87.42%
(L/14) — và `clipB16` là phép thử sạch nhất trong đó: vector 512-d đúng bằng B/32 nên đầu có **cùng
201,090 tham số**, không biến nào khác đổi. So sánh với L/14 không sạch được vậy (`in_dim` → 768).

Đầu meanpool (132,354 tham số, 1.4 phút) bằng hoặc hơn đầu transformer (201,090 tham số, 5.2 phút)
ở 4/8 tháp. Lặp lại kết quả vòng 2 trên tám tháp khác, nên không còn là may của một lần chạy.

### 8.4 Phép đo bất đối xứng: một kết quả âm quan trọng hơn mọi số dương

Chạy lại `analyze_posthoc_asymmetric_measures.py` (không huấn luyện lại gì, chỉ chấm điểm các file
dự đoán đã có) trên 16 dòng tháp. Hai điều rút ra, điều thứ nhất là điều nên vào báo cáo.

**(a) R@FAR5 không đo được trên corpus này.** Điểm ước lượng trông rất hấp dẫn — control 45.48%,
`siglip2B16`+meanpool 74.05%, `clipL14` 75.71% — nhưng CI cụm nguồn của **chính control** là
**[3.74, 80.93]**. Khoảng đó trùm gần hết miền giá trị, nên không một hiệu số R@FAR5 nào ở đây là
phát biểu được. Lý do có thể chỉ đúng tên: phân bố điểm **bão hoà**. Ở ngưỡng 0.5 thì FAR đã là
17.79%, nên muốn ép FAR xuống 5% phải đẩy ngưỡng vào `0.9855` — tức vào giữa khối điểm sát 1.0, nơi
dịch ngưỡng một phần nghìn đổi hàng chục clip. Đây cùng một hiện tượng bão hoà mà mục 4.5.6 đã đo ở
`val_macro`, lần này nhìn từ phía điểm vận hành. Kết luận: corpus 1,100 clip / 151 nguồn **không đủ
sức phân giải một điểm vận hành ở FAR thấp**, và đó là một giới hạn đo lường cần nói thẳng, không
phải một con số để báo cáo.

**(b) Thứ hạng tháp đổi theo phép đo.** Trên hai phép đo không cần ngưỡng:

| Tháp (đầu meanpool) | BA | AUROC | AP |
|---|---|---|---|
| control B/32 | 83.25% | 90.35 [84.08, 95.00] | 79.51 [63.45, 92.72] |
| `openclipB16_l2b` | 85.76% | **93.19** [88.81, 96.75] | **88.98** [79.70, 95.39] |
| `clipB16` | 85.88% | 91.87 [87.36, 95.75] | 87.63 [78.44, 94.22] |
| `siglip2B16` | **87.25%** | 90.72 [82.32, 96.45] | 88.32 [78.02, 95.44] |
| `clipL14` (chặn trên) | 88.58% | 94.62 [89.93, 97.78] | 89.86 [78.25, 97.08] |

`openclipB16_l2b` đứng thứ tư theo BA nhưng **nhất theo cả AUROC và AP** trong lớp B/16; còn
`siglip2B16` nhất theo BA lại gần như ngang control theo AUROC (90.72 so với 90.35). Tức câu "tháp
nào tốt nhất" **không bất biến theo phép đo**. Điều này củng cố mục 8.3.9 theo chiều ngược với
trực giác: giữ BA làm đại lượng đầu bảng là đúng *vì* nó đã đăng ký trước, chứ không vì nó đo đúng
hơn — và bất kỳ ai đổi phép đo sau khi thấy số đều có thể dựng lên một thứ hạng khác.

**Hạn chế của chính mục này:** CI ở đây là CI đơn lẻ từng dòng, **không ghép cặp**, nên không dòng
nào có khoảng tin cậy cho *hiệu số*. Ngưỡng FAR và dòng `BA*` là **trong mẫu**. Không áp BH cho họ
này. Vì vậy mục 8.4 vào báo cáo như mô tả năng lực đo lường, không như kiểm định.
