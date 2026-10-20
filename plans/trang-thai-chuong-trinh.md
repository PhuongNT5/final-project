# Bảng điều khiển chương trình — trạng thái và kết quả

Tệp này là **bảng duy nhất** theo dõi mọi luồng việc của luận văn. Cập nhật tại đây, không mở
bảng mới ở nơi khác. Mốc cập nhật gần nhất: **2026-10-20**. Nhánh: `next-round`.

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
| Q11 | `eva02B16` bị ghi là "never loaded" trong khi dữ liệu nằm trên đĩa | — | **MỞ, ưu tiên** | `round3_tower_bh.json` ghi `"note": "eva02B16 never loaded; carried as p=1"` và `analyze_round3_tower_bh.py` nhắc lại ở ba chỗ (dòng 16, 57, 167). Nhưng `towerEVA02_control_round2_oof.json` có **ba = 83.65, n = 1100** và `towerEVA02_arch_meanpool_oof.json` có **ba = 84.45, n = 1100**. Tháp **đã được đo**. Kết luận **không đổi** (cả hai trị số dưới dải trôi 1.46 pp, và p = 1 ở hạng cuối chưa bao giờ là ngưỡng ràng buộc của BH), nhưng nhãn "NOT MEASURED" nằm cạnh dữ liệu đã đo là đúng loại mâu thuẫn hội đồng tìm ra. Việc đúng: tính p và lề thật cho EVA02 từ out-of-fold đã có rồi chạy lại BH với đủ 8 tháp — 0 huấn luyện, 0 chạm tập kiểm thử |
| Q4 | Viết mục 4.5.7 | — | CHỜ NGƯỜI | hoãn từ vòng 3; chưa có lý do kỹ thuật nào chặn |
| Q5 | Bổ sung baseline ngoài vào Ch2/Ch6 | — | MỞ | S2 trả lời: cách chia **không xác định được** từ nguồn công khai → viết đúng như vậy (`tham chiếu tài liệu`), **không** so trực tiếp với 43.8 % |
| Q6 | Viết §6.1.7 (quan hệ ba làn, ~25 dòng kèm sơ đồ ASCII) | tác giả chốt | CHỜ NGƯỜI | đã giải thích trong hội thoại 13/10, chưa viết vào chương |
| Q7 | Hiệu đính Ch1 §1.6 (dòng 325–327) và Ch2 §2.2 (dòng 161–165) về 2,262 hộp | tác giả chốt | CHỜ NGƯỜI | treo từ 11/10 |
| Q8 | Ước lượng ICC theo nguồn, tính lại cỡ nguồn cần | — | **SẴN SÀNG CHẠY** | mô hình `accuracy ~ system_pair + (1|source)` trên out-of-fold đã có; 0 huấn luyện, 0 chạm tập kiểm thử; S2 cho dải quy đổi: ICC < 0.10 → ~250 nguồn, 0.15 → ~350, > 0.20 → 400+ |
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

1. `report/vn/` đóng băng 05/10: **chỉ sửa văn bản**. Số mới nằm ở thư mục `plans/`.
2. Ngân sách chạm tập kiểm thử: **2 đăng ký trước, 1 đã dùng, 1 còn lại**. S3 đang xác minh lại sổ.
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
2. ICC thực tế theo nguồn là bao nhiêu, và cỡ 300–400 nguồn có còn đúng? (S2, Q8)
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
| Q15 | Runbook tự mâu thuẫn một chỗ | MỞ, sửa văn | §P5 viết `mobileclip_s1 FAIL and quoted with no latency`, nhưng §P4 **vẫn trích** "2.83x slower, 151.8 ms". Phải chọn một: hoặc không trích, hoặc trích kèm nhãn "chi phí của một đồ thị đã chuyển đổi sai" |
| Q16 | Mìn số học trong đồ thị `siglip2B16` | **MỞ, ưu tiên — đây là tháp ứng viên** | §P4 phần cuối: đồ thị xuất của siglip2 chứa **3.64e9** trong phân rã GELU-tanh (hạng x³). Đối sánh FP16 của nó **đã đạt** (cos 0.999918), nên chưa phải lỗi — nhưng là lý do phải đọc hết mục đó trước khi dựa vào tháp này |
| Q13 | `p_unsafe_is_meaningful: true` | MỞ | hai tháp dùng đầu huấn luyện **trên đặc trưng B/32**; phải đọc mã trước khi trích |
