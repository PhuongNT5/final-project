# Bảng điều khiển chương trình — trạng thái và kết quả

Tệp này là **bảng duy nhất** theo dõi mọi luồng việc của luận văn. Cập nhật tại đây, không mở
bảng mới ở nơi khác. Mốc cập nhật gần nhất: **2026-10-31**. Nhánh: `next-round` (**đã đẩy**, `898f096`).
**NGÀY BẢO VỆ: 31/12/2026** — tác giả chốt 31/10. Còn **61 ngày**. Nhánh chiến lược: **C (rộng)**.

Quy ước trạng thái: `ĐANG CHẠY` · `CHỜ NGƯỜI` (chờ tác giả chốt) · `CHỜ THIẾT BỊ` · `XONG` ·
`ĐÃ LOẠI` (kèm lý do). Mỗi dòng phải có **tệp kết quả** — không có tệp thì không tính là xong.

## 1. Trình chạy song song, vòng 19/10

Bốn luồng độc lập, không luồng nào ghi vào tệp của luồng khác. Tất cả đều **đọc-chỉ** trên kho,
0 phép đo mới, 0 lần chạm tập kiểm thử.

| Mã | Câu hỏi phải trả lời | Tệp kết quả | Trạng thái | Kết quả |
|---|---|---|---|---|
| S1 | Tháp B/16 có đạt ≥ 3 qđ/giây trên Orin Nano Super không, và cần gì để đo | `plans/reports/researcher-2026-10-19-jetson-b16-cadence-feasibility.md` (532 dòng) + `deployment/JETSON-RUNBOOK-2026-10-19.md` | XONG, **đã kiểm chứng, có ba sai số** | phép chiếu theo GFLOPs: clipB16 **3.010**, metaclipB16 **3.060**, siglip2B16 **3.252**, mobileclip_s1 **7.003** qđ/giây (số đã sửa, xem mục 6) — cả bốn **chỉ là chặn trên** |
| S2 | Trần 151 nguồn: iSafetyBench chia ngẫu nhiên hay tách nguồn; có kho nào nâng được số nguồn; ICC tính thế nào | `plans/reports/researcher-2026-10-19-source-ceiling-and-corpora.md` (679 dòng) | XONG | (a) cách chia của iSafetyBench **không nêu trong nguồn công khai** → 43.8 % Ovis2-8B vẫn chưa so trực tiếp được; (b) không kho nào nâng số nguồn đáng kể (Unsafe-Net chỉ +2 nguồn) → **trần 151 nguồn là trần thật**; (c) ICC chạy được hôm nay bằng mô hình hỗn hợp trên dự đoán out-of-fold, `statsmodels 0.15.0` đã có trong `.venv` |
| S3 | Các đường dẫn chưa hợp nhất: cái nào commit, cái nào bỏ; ba sửa Ch5 còn treo không | `plans/reports/audit-2026-10-19-trang-thai-chua-hop-nhat.md` | XONG, **đã kiểm chứng** | **38** đường dẫn (không phải 28 — S1 thêm 5 đường giữa phiên). Hai JSON đã sửa là **thuần thêm**: `asymmetric_measures.json` +1088/−0, `paired_vs_control.json` +289/−1, dòng mất duy nhất là chuỗi siêu dữ liệu `38 rows`→`54 rows` ⇒ **không có vấn đề tái lập**. Ba sửa Ch5 **vẫn treo** (nay ở dòng 552–553 và 1234). Ngân sách chạm tập kiểm thử: 2/1/1 xác nhận qua ba tài liệu độc lập, không có lần chạm nào không ghi sổ |
| S4 | Lỗ hổng bảo vệ: phủ sóng nhận xét GVHD, năm câu hỏi khó nhất, tính mới, trích dẫn thiếu | `plans/reports/researcher-2026-10-19-lo-hong-bao-ve-luan-van.md` (384 dòng) | XONG | Y1–Y7: **4 đủ / 3 một phần** (Y4 tham số mặc định, Y7 **10/22 phép đo chưa có công thức**), 0 mục trống; 4 đóng góp đều thuộc loại **mở rộng/định lượng**, không có loại phát hiện; 15 phương pháp thống kê thiếu trích dẫn (McNemar, Nadeau–Bengio, bootstrap cụm, κ là mức cao) |

Phân bổ cố ý: S1 quyết **có triển khai được hay không**, S2 quyết **có đáng chạy thêm hay không**,
S3 quyết **commit cái gì**, S4 quyết **viết thêm cái gì**. Bốn câu hỏi rời nhau nên chạy song song
được; không luồng nào là đầu vào của luồng nào trong vòng này.

Bốn luồng đã về. **Không trị số nào của luồng nào được chép vào `report/vn/`** — mọi số ở đây là
số của thư mục kế hoạch cho tới khi có phép đo trên thiết bị và tác giả chốt.

## 6. Ba sai số của S1 — đã kiểm chứng tại nguồn, phải sửa trước khi dùng

1. **Bảng chiếu lệch dòng.** S1 in `326.8 ms / 3.06 qđ/giây` cho `clipB16`; `326.8` là trị số
   **đúng của `metaclipB16`**. Phần không-thuộc-tháp đúng là `167.291 − 57.399 = 109.892 ms`
   (lần trung vị `20261003T155843`); S1 dùng `104.5` cho ba dòng đầu và `124.4` cho dòng cuối.
   Số đã sửa: clipB16 **3.010** (sát mép cổng, gần như không còn lề), metaclipB16 3.060,
   siglip2B16 3.252, mobileclip_s1 **7.003** — tức mobileclip bị S1 **hạ thấp** (6.35).
2. **Chiều của hiệu ứng DVFS bị đảo.** S1 viết "tháp lớn hơn có thể sụt nhịp nặng hơn vì GPU rỗi
   lâu hơn". Mục 5.4.5 đo ngược lại: hiệu ứng là **của sự ngắt quãng, không phải của nhiệt** —
   chạy liên tục mất 0.6 %, pipeline có chu kỳ công tác mất **2.09 lần**. Tháp lớn hơn ⇒ chu kỳ
   công tác **cao hơn** ⇒ rỗi **ít hơn** ⇒ sụt nhịp **nhẹ hơn**. Nên phép chiếu nhân nguyên hiệu
   ứng 2.09 lần vào tháp 4 lần GFLOPs là **bi quan**, còn với tháp **nhỏ hơn** thì ngược lại:
   dải thật của mobileclip_s1 là **6.16–7.00 qđ/giây** (+3.1 % đến +17.2 % so với 5.978), biên
   dưới lấy theo dải xung nhịp GPU 3.33 lần. Cũng vậy, S1 gọi 2.48 lần là "sụt nhịp DVFS";
   2.48 là tỷ số vòng-lặp/liên-tiếp, trong đó **2.09** mới là phần đã phân xử.
3. **Hai trích dẫn sai, một trong đó vượt làn.** Khoảng tin cậy của siglip2B16 được dẫn là
   "chương 03, mục 8.1" — Ch3 chỉ có tới §3.9; số đó nằm ở `chien-luoc-thuc-nghiem-vong-3.md` §8.
   Nặng hơn: S1 lấy nhịp ghim xung của **P3** (Bảng 5.4b) để luận rằng tháp B/16 của **B1** sẽ
   đạt. Hai pipeline khác nhau. Thêm nữa, nền 167.291 ms **vốn đã là phép đo dưới DVFS mặc định**
   (`governor: schedutil`, `jetson_clocks_show: null`), nên khung "dưới DVFS thì trượt, ghim thì
   đạt" là sai nền: B1 **chưa bao giờ** được đo khi ghim xung.

Giữ lại từ S1 sau khi trừ ba sai số: bảng GFLOPs/tham số/độ phân giải bốn tháp, năm cơ chế phản
biện phép chiếu GFLOPs, xếp hạng rủi ro xuất engine, và runbook 5 bước — **cổng ≥ 3 qđ/giây** mà
S1 dùng là đúng, đã đăng ký trước ở `chien-luoc-thuc-nghiem-vong-3.md` dòng 44, kèm luật dự phòng
"chưa đo thì vào báo cáo như **chặn trên của khoảng trống biểu diễn**".

## 2. Hàng đợi — việc còn mở

| # | Việc | Chặn bởi | Trạng thái | Ghi chú |
|---|---|---|---|---|
| Q1 | Đo độ trễ tháp B/16 trên Jetson | — | **XONG 19–20/10** | cả ba tháp **đạt** cổng ≥ 3 qđ/giây dưới DVFS mặc định; xem mục 7. Chặn còn lại **không phải phép đo** mà là **bản ghi đối sánh engine** (Q12) |
| Q2 | Ba sửa văn bản Ch5: `53.2 %`→`52.6 %`; `vòng lặp tổ hợp`→`tổng giai đoạn`; `0.49 %`→`0.87 %` | tác giả chốt | CHỜ NGƯỜI | căn cứ: `plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-10-13-kiem-toan-so-lieu-jetson-muc-tieu-nhi-phan.md` mục 5; S3 đang xác minh còn treo hay đã vá ở bf9a627 |
| Q3 | Hợp nhất 38 đường dẫn chưa commit | — | **SẴN SÀNG**, sau hai việc vá | hai JSON đã sửa là thuần thêm, **không chặn**. Hai việc phải làm trước `git add`: (a) nối luật loại trừ cho `deployment/export/r48-tower-*.onnx` theo đúng tiền lệ dòng 132–133 — clipB16 345 MB và siglip2B16 372 MB **vượt hạn 100 MB của GitHub**, kho **không dùng LFS**, luật dòng 35 `deployment/*.onnx` **không khớp thư mục con**; (b) thêm `!sweep_results/posthoc_sensitivity/round3_tower_bh.json` cạnh dòng 150–151 |
| Q11 | `eva02B16` bị ghi là "never loaded" trong khi dữ liệu nằm trên đĩa | — | **XONG — và khuyết điểm nhẹ hơn tôi đã ghi** | Tôi đã ghi dòng này là "EVA02 bị mang vào BH với p = 1 bịa ra". **Sai.** Kiểm lại trên chính bản đã commit `8610c74`: JSON ở đó **đã** chứa `ba 83.65`, `p_bootstrap 0.9336` (sơ cấp) và `ba 84.45`, `p_bootstrap 0.7072` (thứ cấp) — trị số **thật**, không phải chỗ trống. Nhánh dự phòng `os.path.exists` trong `evaluate()` chưa bao giờ chạy vì tệp out-of-fold của EVA02 **có trên đĩa**, nên cả hai lượt BH luôn dùng số thật. Khuyết điểm chỉ là **bốn chuỗi văn bản** lạc hậu (`analyze_round3_tower_bh.py` dòng 16, 48, 57, 167 và trường `note`/`role` trong JSON). Đã sửa tại chỗ: script 174 → 175 dòng; JSON **3 dòng chữ đổi, 0 trường số đổi**. EVA02 xếp **hạng 7** (sơ cấp) và **hạng 6** (thứ cấp), chưa bao giờ là ngưỡng ràng buộc. Kết luận đã công bố — không tháp nào qua ngưỡng BH hạng 1 là 0.00625; p nhỏ nhất 0.093 — **đúng y nguyên**. 0 chạm tập kiểm thử |
| Q4 | Viết mục 4.5.7 | — | CHỜ NGƯỜI | hoãn từ vòng 3; chưa có lý do kỹ thuật nào chặn |
| Q5 | Bổ sung baseline ngoài vào Ch2/Ch6 | — | MỞ | S2 trả lời: cách chia **không xác định được** từ nguồn công khai → viết đúng như vậy (`tham chiếu tài liệu`), **không** so trực tiếp với 43.8 % |
| Q6 | Viết §6.1.7 (quan hệ ba làn, ~25 dòng kèm sơ đồ ASCII) | tác giả chốt | CHỜ NGƯỜI | đã giải thích trong hội thoại 13/10, chưa viết vào chương |
| Q7 | Hiệu đính Ch1 §1.6 (dòng 325–327) và Ch2 §2.2 (dòng 161–165) về 2,262 hộp | tác giả chốt | CHỜ NGƯỜI | treo từ 11/10 |
| Q8 | Ước lượng ICC theo nguồn, tính lại cỡ nguồn cần | — | **XONG — dải tiền đăng ký bị vượt** | ICC đo được **0.429** (B1 vs B2) và **0.442** (B1 vs ESTA), KTC bootstrap theo nguồn [0.256, 0.548] và [0.244, 0.569]; ba bộ ước lượng (REML / ANOVA / logistic-VB) khớp nhau trong ~0.03, hai cặp hệ khớp nhau trong ~0.02 ⇒ **ổn định**. Dải tiền đăng ký của S2 chỉ lên tới "> 0.20 → 400+"; trị số đo **nằm trên mọi ngăn**, và ngay cả **sàn KTC** vẫn thuộc ngăn trên cùng. Cỡ nguồn tính lại với `m = 7.285` đo được: **458–468 nguồn** cho +5 pp (≈ **3.0× số 151 hiện có**, và **xấu hơn** mức "300–400" từng chiếu); 1,300+ cho +3 pp; 3,000+ cho +2 pp. Nguyên nhân cấu trúc: **0/151 nguồn mang cả hai nhãn** — tôi tự xác nhận lại. `report/vn/` **không chứa** phát biểu ICC/công suất/"300–400" nào, nên không số đóng băng nào bị đụng; chỉ một phép chiếu trong `plans/` bị sửa. 0 chạm tập kiểm thử. Báo cáo: `plans/reports/analysis-2026-10-20-icc-theo-nguon-va-so-nguon-can-them.md` |
| Q9 | ch07:149 TSF `+0.2 pp` — BA riêng luồng thực thể không in ở đâu | — | MỞ | treo từ 13/10 |
| Q10 | Làn Unsafe-Net 687 clip 2 lớp (69.6 %) chưa có đường sàn ở `report/vn/` | — | MỞ | treo từ 13/10 |

## 3. Đã xong — kết quả, không chỉ trạng thái

| Mốc | Ngày | Kết quả | Chứng cứ |
|---|---|---|---|
| Đóng băng số liệu | 05/10 | `report/vn/` chỉ còn sửa văn bản | `README.md`, `00-trang-thai-bien-soan.txt` |
| Tái lập giao thức Unsafe-Net, chấm nhị phân | 11/10 | khai báo đủ bốn mục cho mọi trị số kết quả | `a063da5` `eaaccf5` `6889261` `06854b4` |
| Đọc Bảng 8 công trình gốc (phía tốc độ) | 13/10 | thông lượng hai hệ **bằng nhau, lệch 0.35 %**; "nhanh 4.14 lần" chỉ là đầu vào ngắn hơn | `16f75c9`, §6.15.2 |
| Kiểm toán số Jetson của mục tiêu nhị phân | 13/10 | 4 mâu thuẫn → 3 tan, 1 khiếm khuyết thật (tỉ số trộn hai lần chạy) | `finding-2026-10-13-kiem-toan-so-lieu-jetson-muc-tieu-nhi-phan.md` |
| Trả lời 7 nhận xét GVHD (Y1–Y7), 6 pha | 15/10 | 1,373 dòng thêm trên 9 tệp, **không số nào dịch chuyển** | `bf9a627`, `reports/bao-cao-phu-song-y1-y7.md` |
| Quét vòng 1–2: tham số và đầu chưa thử | 15/10 | 38 dòng hậu kiểm; thay đầu **không mua được gì** (meanpool 84.89 ≥ transformer 83.25) | `30291ed` `5644889` |
| Vòng 3: thang chín tháp đặc trưng | 16/10 | chốt giải mã **trùng khít bit**; `siglip2B16` cao nhất 86.39 % (+3.14 pp); **không tháp nào sống sót BH q=0.05** (p nhỏ nhất 0.093 so ngưỡng 0.00625) | `chien-luoc-thuc-nghiem-vong-3.md` §8, `round3_tower_bh.json` |
| R@FAR5 trên corpus này | 16/10 | **không đo được**: CI cụm nguồn của chính control là [3.74, 80.93] — giới hạn đo lường, không phải số để báo cáo | cùng tệp, §8.4 |

Ba kết quả âm có giá trị cao nhất, đã lặp lại độc lập: (a) thứ hạng zero-shot **không** chuyển sang
tác vụ này (DataComp-1B +1.05 và DFN-2B +0.43 kém hơn CLIP thường +2.58); (b) xương sống video
VideoMAE **kém** tháp ảnh đông cứng 2.37 pp; (c) `mobileclip_s1` **nhỏ hơn** B/32 mà bằng control
(+0.47 pp, p = 0.93) — trong dải này độ chính xác và chi phí gần như không đổi nhau được.

## 4. Ràng buộc bất biến — mọi luồng phải tuân

1. `report/vn/` đóng băng: **số chốt 09/10, văn bản chốt 15/10**, kèm **ba ngoại lệ đã khai** tại
   `report/vn/00-trang-thai-bien-soan.txt:14-50` (G20; G21/G25 ngày 11/10 tiêu một lần chạm gộp;
   NP-125-D ngày 11/10 tiêu 0 lần). Số mới nằm ở thư mục `plans/`.
   *(Câu cũ ghi "đóng băng 05/10, chỉ sửa văn bản" — **sai**: phép đo lạm phát giao thức vào
   `report/vn/` ngày 08/10 tại `9895df4`. Nội dung vẫn đúng và đã khai minh bạch; chỉ nhãn sai.
   Sửa theo T1-4, phiên 31/10.)*
2. Ngân sách chạm tập kiểm thử: **2 đăng ký trước, 1 đã dùng (G21+G25, 11/10), 1 còn lại**.
   S3 đã xác minh qua ba tài liệu độc lập: không có lần chạm nào không ghi sổ.
3. Không trộn làn A/B/C. Không so chéo iSafetyBench ↔ Unsafe-Net.
4. Đại lượng đầu bảng là **balanced accuracy** (§8.3.9). Không đổi sau khi thấy số.
5. So với control **83.25 %**, không bao giờ so 84.2 %. Độ chính xác đi kèm số Jetson là **84.2 %**,
   không bao giờ 84.4 %.
6. Bảng tám cổng giữ nguyên: **hai đạt, sáu không đạt**.
7. Hợp thức kép: biên ≥ **1.46 pp** (dải trôi đã đo) **và** CI cụm nguồn loại trừ 0; thêm BH
   q = 0.05 trên **8** tháp (mẫu số 8, không phải 7 như lúc đăng ký viết sai).
8. Không số độ chính xác nào được đọc là độ chính xác của vòng lặp trên thiết bị (chưa đo, §5.9 ý 15).
9. Dòng `report/vn/` ≤ 100 **ký tự**; không dòng nào bắt đầu bằng `^[0-9]\.[0-9]` trừ tiêu đề mục.

## 5. Câu hỏi chưa giải quyết

1. iSafetyBench gốc chia ngẫu nhiên hay tách nguồn? (S2) — chặn Q5.
2. ~~ICC thực tế theo nguồn là bao nhiêu, và cỡ 300–400 nguồn có còn đúng?~~ **ĐÃ TRẢ LỜI 20/10 (Q8):**
   ICC 0.43–0.44, trên ngăn cao nhất của dải; cỡ cần **458–468 nguồn** cho +5 pp, không phải 300–400.
3. Có tháp nào vừa hơn control vừa giữ ≥ 3 qđ/giây? (S1, Q1)
4. Hai tệp JSON hậu kiểm bị **sửa** — giá trị đã công bố có đổi không? (S3) Nếu có thì đó là vấn đề
   tái lập, không phải bổ sung.
5. Có nên commit `kaggle/` khi vòng 3 cuối cùng chạy ở máy, không chạy ở Kaggle? (S3)

## 7. Trình chạy thực nghiệm — Jetson, 19–20/10

Thiết bị: Orin Nano Super 8 GB, JetPack R36.5.0, TensorRT 10.3.0, MAXN_SUPER (mode 2),
**DVFS mặc định** (`schedutil`, GPU 306 MHz trước lần chạy → 1,020 MHz sau). Cổng cửa sổ sạch
`jetson-clean-window-guard.py`: `clean_before = true`, `clean_after = true`,
`dirty_reasons_after = []`, `returncode = 0` cho **mọi** lần chạy.

Phạm vi mọi lần chạy ở đây: **12 clip / 3,002 khung / 109 quyết định**, cửa sổ 48 / bước 24.
Đây **không** phải tập đo của trị số đã công bố (24 clip / 5,559 khung / 201 quyết định), nên
**mọi so sánh phải dùng lần chạy đối chứng cùng phiên**, không trừ vào 167.291 ms.

| Nhánh | Tháp | Tham số nền | Tháp p50 (ms) | Khoảng (ms) | qđ/giây | Chu kỳ | Cổng ≥ 3 | Tệp |
|---|---|---|---|---|---|---|---|---|
| FP16 | B/32 (đối chứng) | 87.8 M | 57.441 | 169.843 | **5.888** | 0.630 | ĐẠT | `r48-composed-cadence-fp16-20261019T234049.json` |
| FP16 | `clipB16` | 149.6 M | 103.683 | 228.629 | **4.374** | 0.702 | **ĐẠT** | `...-r48-tower-clipB16-fp16-20261019T234253.json` |
| FP16 | `siglip2B16` | — | 104.440 | 227.582 | **4.394** | 0.706 | **ĐẠT** | `...-r48-tower-siglip2B16-fp16-20261019T234458.json` |
| FP32 | B/32 (đối chứng) | 87.8 M | 86.823 | 202.717 | 4.933 | 0.679 | ĐẠT | `r48-composed-cadence-fp32-20261020T072902.json` |
| FP32 | B/32 (chạy lại) | 87.8 M | 87.655 | 204.293 | 4.895 | 0.680 | ĐẠT | `r48-composed-cadence-fp32-20261020T193917.json` |
| FP32 | `mobileclip_s1` | ~21.5 M | **161.314** | 281.495 | **3.552** | 0.769 | **ĐẠT** | `...-mobileclip_s1-fp32-20261020T194127.json` |

Bốn điều phải đọc cùng bảng, nếu không sẽ đọc sai:

1. **Đây là phép đo CHI PHÍ, không phải phép đo độ chính xác.** `siglip2B16` chạy với đầu
   `r48-b1-head-cost768-3seed` chưa huấn luyện và chính tệp kết quả ghi
   `p_unsafe_is_meaningful: false`; hai tháp còn lại dùng đầu đã huấn luyện của B/32 nhưng
   `is_published_pair: false`. **Không trị số độ chính xác nào được đọc từ các lần chạy này.**
   Phán quyết độ chính xác của vòng 3 **không đổi**: không tháp nào sống sót BH (m = 8).
2. **Cả ba tháp đều ĐẠT cổng.** Điều này **đảo** kết luận của S1 ("cả bốn tháp có thể trượt dưới
   DVFS") và đảo cả phần tôi chuyển tiếp từ nó.
3. **Phép chiếu theo GFLOPs sai hai chiều ngược nhau.** clipB16: chiếu 222.3 ms, đo **103.683**
   (cao gấp **2.14 lần**). siglip2B16: chiếu 197.6, đo **104.440** (cao gấp **1.89 lần**).
   mobileclip_s1: chiếu 32.9, đo **161.314** (thấp **4.90 lần**). Trên lớp thiết bị này, phép
   chiếu theo GFLOPs **không phải là công cụ lập kế hoạch**.
4. **`mobileclip_s1` chậm hơn đối chứng 1.84 lần** ở cùng mức chính xác số, dù nhỏ hơn ~4 lần về
   tham số. Cộng với vòng 3 (+0.47 pp, p = 0.93 — không phân biệt được với đối chứng), nó **tệ
   hơn ở cả hai trục**. Đây là chỗ đề nghị P1 trước đó của tôi dựa vào, và phép đo đã bác bỏ nó.

Số học trên phép chiếu (**không** phải phép đo, vì không có phép đo liên tiếp cho tháp B/16): hệ
số vòng-lặp/liên-tiếp của đối chứng là 57.441 / 23.386 = **2.456**; với clipB16, lấy nền liên tiếp
ước theo GFLOPs 23.386 × 16.87/4.4 = 89.66 thì hệ số còn **1.156**. Đúng chiều cơ chế mục 5.4.5
(tháp lớn ⇒ chu kỳ công tác cao ⇒ rỗi ít ⇒ sụt nhịp nhẹ), và là chiều **ngược** với điều S1 nói.
Luồng E2 đang kiểm xem dữ liệu có đỡ được một **quy luật** hay chỉ đỡ được một **chiều**.

## 8. Trình chạy song song, vòng 20/10 — đã về đủ bốn, đã kiểm chứng tại nguồn

| Mã | Câu hỏi | Tệp kết quả | Trạng thái sau kiểm chứng |
|---|---|---|---|
| E1 | Vì sao `mobileclip_s1` chậm hơn tháp lớn gấp 4 lần | `plans/reports/researcher-2026-10-20-mobileclip-cham-hon-tren-thiet-bi.md` | **XONG nhưng SAI ĐỀ** — tiền đề tôi giao đã bị runbook §P4 vượt qua trước đó: engine sai ở **cả hai** độ chính xác, nên 161.314 ms không phải chi phí của tháp. Giữ lại được: biểu đồ op đo từ ONNX (215 Conv / 123 nhóm / 80 BN / 72 Identity / 1,398 nút, so với CLIP 1 Conv / 97 MatMul / 0 BN). Loại bỏ: mọi suy luận về "vì sao chậm" như tính chất kiến trúc, và câu "TensorRT 11.1" — thiết bị chạy **10.3.0** |
| E2 | Sụt nhịp khi rỗi có là hàm của chu kỳ công tác | `plans/reports/researcher-2026-10-20-luat-sut-nhip-theo-chu-ky-cong-tac.md` | **XONG, kết luận âm trung thực** — không có hàm nào dựng được. Phần ĐO được: chi phí tháp FP16 **phẳng ~58.2 ms** khi chu kỳ công tác chạy 0.37–0.77, và bước nhảy 26.4 → 58.2 ms (**2.20×**) khi đi từ liên tục sang có chu kỳ. Phần **không** dùng được: trị số sụt 1.02× cho clipB16 — số học trên một phép chiếu GFLOPs tuyến tính, đúng loại phép chiếu mà `mobileclip` vừa bác bỏ. Sáu trích dẫn arXiv chưa kiểm được |
| E3 | Bản ghi đối sánh engine | `plans/reports/audit-2026-10-20-doi-sanh-engine-va-do-chinh-xac-so.md` | **XONG, ĐỌC THIẾU — tôi đã sửa** — báo đúng backend (TensorRT, `benchmark-r48-composed-cadence.py:139-154`), đúng ba phán quyết §P5 ở dạng văn xuôi, đúng việc không có tệp phán quyết. Bỏ sót mục **§P4 dòng 160–207**, nơi có bảng đối sánh **FP32** và kết luận mạnh hơn nhiều (xem Q14) |
| E4 | Kết quả thiết bị được phép nằm ở đâu | `plans/reports/researcher-2026-10-20-cho-ket-qua-thiet-bi-vao-dau.md` | **XONG, một đề nghị phải loại** — "Option C: sửa hạng mục 15 mục 5.9" **sai**. Hạng mục 15 là **độ chính xác** của vòng lặp cửa sổ trượt, không phải nhịp; ch05 dòng 213–226 đã nói hai lần rằng mọi số mục 5.4 là số **chi phí**, và hạng mục 15 ghi rõ phép đo ấy **chạy trên máy chủ, không cần thiết bị**. Một số chi phí không đóng được nó. Giữ: Option A (báo cáo riêng trong `plans/`) |

## 9. Bốn việc mới sinh ra từ vòng 20/10

| # | Việc | Trạng thái | Ghi chú |
|---|---|---|---|
| Q12 | Phán quyết đối sánh engine cho ba tháp vòng 3 | **MỞ — đã rõ là có, nhưng ở dạng văn xuôi** | `JETSON-RUNBOOK-2026-10-19.md` §P5 ghi: `clipB16` cos 0.999973 / rel 4.76e-03 **ĐẠT**; `siglip2B16` cos 0.999918 / rel 4.73e-03 **ĐẠT**; `mobileclip_s1` **TRƯỢT**. Đối sánh máy chủ (torch↔ONNX, batch 16) cả ba cos 1.000000. Thiếu: tệp JSON theo đúng cấu trúc `r48-parity-fp16-20261001T224028.json`. Chưa chặn số của hai tháp ĐẠT nữa — chỉ còn là việc lưu bản ghi |
| Q14 | **TensorRT 10.3.0 chuyển đổi sai đồ thị MobileCLIP-S1** | **MỞ — phát hiện, không phải lỗi của ta** | §P4: sai **bất biến** qua hai độ chính xác và bốn cỡ lô — FP32 cos 0.948259→0.932605 / rel 0.241–0.247; FP16 0.948011→0.932087 / rel 0.241–0.259. ONNX máy chủ đúng (rel 1.22e-05), tổng kiểm điểm ảnh lệch **đúng 0**. Nên: tháp **không triển khai được qua TensorRT trên thiết bị này ở bất kỳ độ chính xác nào**, và **161.314 ms không được trích như chi phí của MobileCLIP**. Chưa định vị op gây lỗi (cần chẻ đồ thị) — runbook nói kết luận không phụ thuộc vào việc đó, đúng |
| Q15 | Một câu trong runbook lỏng hơn phần thân | MỞ, sửa văn, nhỏ | §P5 viết `mobileclip_s1 FAIL and quoted with no latency`, nhưng §P4 **trích** 2.83× **kèm đủ hai điều kiện ràng buộc** (là FP32 nên chỉ cặp được với đối chứng FP32 cùng phiên; engine trượt đối sánh nên là chi phí của một đồ thị không tái tạo được tháp). §P4 đúng — đây không phải mâu thuẫn như tôi ghi lúc đầu, chỉ là câu một dòng ở §P5 nói lỏng hơn phần thân |
| Q16 | Mìn số học trong đồ thị `siglip2B16` | **XONG, phải đi kèm mọi khuyến nghị** | runbook §P4 cuối: đồ thị xuất chứa **3.64e9** trong phân rã GELU-tanh (hạng x³, `layers.9/mlp/activation_fn/Mul_1`), vượt trần 65,504 của FP16 ~5e4 lần. TensorRT 10.3 **hợp nhất** mẫu đó nên engine vẫn đạt đối sánh ⇒ tính chất của **runtime này**, không phải của mô hình. Một runtime khác thực thi phân rã theo đúng nghĩa chữ sẽ vỡ |
| Q13 | `p_unsafe_is_meaningful` | **XONG** | `benchmark-r48-composed-cadence.py:196`: `s['p_unsafe_is_meaningful'] = args.head == HEAD`. Cờ chỉ xét **đầu**, bỏ qua **tháp**. Lượt `mobileclip_s1` có `is_published_pair: false` nhưng cờ `true`, chỉ vì nạp đúng đầu B1 đã công bố — đầu ấy huấn luyện trên đặc trưng B/32 và chưa bao giờ thấy embedding MobileCLIP. Nghĩa đúng: "đầu thật, không phải đầu giả lập". **Không** phải "xác suất hợp lệ". Quy tắc: `is_published_pair: false` ⇒ xác suất không dùng được |

## 10. Mục 1–3 của chiến lược — đã làm, 20/10

| Việc | Trạng thái | Kết quả |
|---|---|---|
| **P0** commit | **XONG** | `8610c74`, **83 đường dẫn**. Tìm thêm **blocker thứ ba** mà tôi chưa thấy ở vòng trước: `.gitignore:55` `deployment/artifacts/*` che toàn bộ, và 133 tệp đang track chỉ nhờ các dòng phủ định bên dưới — nên **11 JSON đo thiết bị** (`is_device_number: true`, 205 KB, không tái tạo được), 11 phán quyết cổng cửa sổ sạch và 4 `.npz` tham chiếu đối sánh đều **vô hình với git**. Ba vá: mở `jetson-measurements/*.json` + `round3-device/` + `r48-parity-fixture/*.npz`; mở `round3_tower_bh.json`; che `demo/out/*.mp4` (giữ hai JSON dòng thời gian). Ba ONNX tháp vòng 3 **không** vào git — hai tệp 329/354 MB vượt hạn 100 MB của GitHub, tệp thứ ba 82.5 MB vượt mức 48 MB kho tự đặt; dựng lại bằng `export-round3-towers-onnx.py` (tự chạy cổng đối sánh và đổi tên đồ thị trượt). Đã soát: không `.env`/khoá/chứng thư. Biên dịch sạch 9 tệp Python + 1 shell; 0 JSON lỗi |
| **P1** Q13, Q16 | **XONG** | xem hai dòng mục 9 ở trên |
| **P1** Q11 BH với EVA02 thật | **XONG — nó sửa tôi** | BH tái lập **y hệt từng byte** trước khi sửa gì. Khuyết điểm là **văn bản, không phải số**: trị số EVA02 thật đã nằm trong cả hai lượt BH ngay từ bản đã công bố. Kết luận không đổi. Xem dòng Q11 mục 2 |
| **P1** Q8 ICC theo nguồn | **XONG — kết quả xấu hơn phép chiếu cũ** | ICC 0.43–0.44 (KTC [0.24, 0.57]) vượt ngăn cao nhất của dải tiền đăng ký; cần **458–468 nguồn** cho +5 pp. Không đụng số đóng băng. Xem dòng Q8 mục 2 |
| **P2** báo cáo kết quả thiết bị | **XONG** | `plans/reports/ket-qua-thiet-bi-2026-10-20-ba-thap-vong-3-tren-jetson.md` (184 dòng) — sáu nhánh, khai báo bốn mục, hai điều kiện ràng buộc của trị số mobileclip, và phần sửa cả bốn luồng E1–E4 |

**ĐÃ ĐẨY 31/10/2026.** `5644889..6a5d01c` rồi `6a5d01c..898f096` lên `origin/next-round`, sau khi
tác giả cho phép. 86 tệp / 20,290 dòng — gồm 11 JSON đo thiết bị không tái tạo được — không còn
nằm trên một đĩa. **Lưu ý còn mở:** `report/vn/` vẫn **không tồn tại trên `origin/main`**; hợp
nhất `next-round` → `main` là việc riêng, chưa làm.

## 11. Hội đồng phản biện & đánh giá khoa học, phiên 31/10 — đã họp, đã tổng hợp

Phiên thứ ba (sau 05/10 và 10/10). 6 phản biện độc lập PB-07…PB-12 + 9 phép tự kiểm chứng của
chủ tịch tại nguồn. Tệp: `plans/261031-hoi-dong-phan-bien-danh-gia-khoa-hoc/`.

**Phán quyết: ĐẠT CÓ ĐIỀU KIỆN** — giữ mức 05/10, nhưng **ràng buộc quyết định đã dịch từ khoa học
sang thủ tục và xuất xứ**. Xác suất bảo vệ 55–65 % → 78–85 % chỉ bằng câu chữ + diễn tập.

Phát biểu trung tâm: **bản nộp sạch hơn vùng xung quanh nó** — 11 tệp `report/vn/` không rò rỉ số
đã rút, 0 vi phạm định dạng, 12/12 hình. Mọi phát hiện nặng nằm **ngoài** bản nộp.

| Mã | Việc sinh ra từ phiên 31/10 | Trạng thái | Ghi chú |
|---|---|---|---|
| **R1** | Sửa `README.md` — dòng 20 rút `3.02 dec/s`, dòng 43–44 dùng lại nó | **CHỜ NGƯỜI** | đã lừa được chủ tịch hội đồng; khuyến nghị xoá mục Key Results, trỏ về `report/vn/` |
| **R2** | **A4** — văn bản GVHD duyệt đổi hướng | **CHỜ NGƯỜI**, treo 3 tuần | hạng mục **duy nhất** cả ba hội đồng nói lật được ĐẠT/KHÔNG ĐẠT; 0 ngày công |
| **R3** | **κ trên nhãn nhị phân iSafetyBench**, n ≥ 200, hai người gán nhãn độc lập | **CHỜ NGƯỜI** | khuyến nghị số một của phiên 10/10, 3 tuần vẫn 0 bước chuẩn bị. **Không tiêu lần chạm test.** Đường tới hạn 3 tuần ⇒ khởi động 01–07/11 |
| **R4** | Lời khai xuất xứ tiền đăng ký (1 trang) | CHỜ NGƯỜI | git không chứng minh được phần "tiền": `risks.md` vào git 01/10 **sau** kết quả G14 ngày 29/09; mẫu lặp ở vòng 3. Vá bằng tự khai, không vá được bằng chạy lại |
| **R5** | Ba vết sửa `report/vn/chuong-05`: `0.49%`→`0.87%`, `vòng lặp tổ hợp`→`tổng giai đoạn`, `53.2%`→`52.6%`; **xoá §5.9(12)** | CHỜ NGƯỜI | §5.9(12) tự khai "hạn chế nghiêm trọng nhất" **không còn tồn tại** — đã track từ `df1dab7` |
| **R6** | Gắn mốc thời gian cho câu "cả hai lần chạm còn nguyên" (`chuong-04:1261,1276-1279`; `chuong-06:2271-2272`) | CHỜ NGƯỜI | bản chất đúng (1 tiêu / 1 còn), câu chữ tự mâu thuẫn với §6.15 |
| **R7** | Đoạn giới hạn **hiệu lực cấu trúc**: 0/151 nguồn mang cả hai nhãn | CHỜ NGƯỜI | lỗ hổng khoa học sâu nhất phiên này tìm được; **không cần đo** |
| **R8** | Viết `§4.5.7` và `§6.1.7` (Q4, Q6) | CHỜ NGƯỜI | hai mục **đang dở**, không phải moot; §6.1.7 nên là mục ba làn + lệnh cấm so làn |
| **R9** | Diễn tập phản biện có ghi âm, kèm **bảng phát ngôn bị cấm** | CHỜ NGƯỜI | hai buổi: ~22–28/11 và ~24/12 |
| **R10** | `requirements.txt` / khoá phiên bản + pin revision HF Hub | CHỜ NGƯỜI | khe hở tái lập thật, lớn hơn vấn đề độ dài tệp |

**Quyết định chiến lược: ngừng mua độ chính xác.** Năm phép đo độc lập cùng chỉ một hướng —
ngưỡng trung thực −0.47 pp, ensemble +0.33 pp, đổi đầu ≤ 0, đổi tháp **0/8 sống sót BH**, thêm
nguồn cần **3.0×** kho hiện có (ICC 0.43–0.44 ⇒ 458–468 nguồn). Hướng "nâng 84.2 %" **ĐÃ ĐÓNG,
có căn cứ số**. Mọi runner nhằm tăng số → dừng hoặc hạ xuống sau bảo vệ.
