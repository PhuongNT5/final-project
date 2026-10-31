# KẾ HOẠCH CHIẾN LƯỢC SAU PHIÊN 31/10/2026

Căn cứ: `reports/bien-ban-tong-hop-hoi-dong-2026-10-31.md` (6 phản biện + 9 phép tự kiểm chứng của chủ tịch)
Trạng thái nền: `report/vn/` đóng băng (số 09/10, văn bản 15/10) · nhánh `next-round`, **2 commit chưa đẩy**
Phán quyết nền: **ĐẠT CÓ ĐIỀU KIỆN**, xác suất bảo vệ thành công hiện tại **55–65 %**

---

## 0. QUYẾT ĐỊNH CHIẾN LƯỢC TRUNG TÂM

> **Ngừng mua độ chính xác. Chuyển toàn bộ nguồn lực còn lại sang xuất xứ, thủ tục, diễn tập.**

Không phải lựa chọn phong cách — là **kết luận từ năm phép đo** (biên bản §4): ngưỡng/hiệu chuẩn
−0.47 pp khi chọn trung thực, ensemble +0.33 pp, đổi đầu ≤ 0, đổi tháp **0/8 sống sót BH**, thêm
nguồn cần **3.0× kho hiện có**. Mọi đường rẻ đã cạn, đường còn lại không khả thi trong khung thời gian.

**Hệ quả hành động:** đóng chính thức hướng "nâng 84.2 %" trong `plans/trang-thai-chuong-trinh.md`.
Mọi runner còn mở nhằm tăng số (S2 mở rộng nguồn, các nhánh tháp vòng 4) → **dừng hoặc hạ xuống
hạng mục sau bảo vệ**. Lý do ghi rõ: *đã đo, không phải đã đoán*.

Đổi lại, lợi tức cao nhất nằm ở chỗ **rẻ nhất**: xác suất 55–65 % → **78–85 %** chỉ bằng câu chữ
và diễn tập (0 phép đo mới), → **85–90 %** nếu thêm **một** phép đo κ.

---

## 1. NHÁNH ĐÃ CHỐT: **C — RỘNG**

**Ngày bảo vệ: 31/12/2026** (tác giả chốt 31/10). Còn **61 ngày**. Nhánh đã đẩy (`898f096`).

| Nhánh | Điều kiện | Trạng thái |
|---|---|---|
| A — Gấp | ≤ 10 ngày | không áp dụng |
| B — Vừa | 10–30 ngày | không áp dụng |
| **C — Rộng** | **> 30 ngày** | **ĐANG ÁP DỤNG** — toàn bộ Hạng mục 0, 1, 2, 3 trong phạm vi |

**61 ngày không phải lý do để giãn việc.** Ba hệ quả của việc có nhiều thời gian:

1. **T2-1 (κ) phải khởi động trong tuần đầu, không phải tuần cuối.** Nó phụ thuộc **người thứ hai** — người gán nhãn độc lập. Phụ thuộc người là loại rủi ro không nén được bằng nỗ lực. Bắt đầu muộn = không có κ.
2. **Phải có mốc đóng băng thứ hai: 17/12 — hai tuần trước bảo vệ.** Sau mốc đó **không sửa gì nữa**, chỉ diễn tập. Lý do rút từ chính lịch sử dự án này: mốc đóng băng 05/10 đã bị tràn tới 15/10 vì không có vùng đệm.
3. **Thời gian dư đi vào diễn tập, không đi vào số.** Hai buổi diễn tập cách nhau (một sớm để lộ lỗ hổng còn kịp vá, một sát ngày để luyện phản xạ) đáng hơn mọi phép đo thêm — căn cứ §0.

**Rủi ro lớn nhất của nhánh C: dùng 61 ngày để mở lại hướng tăng độ chính xác.** Xem §0 và §7.3.

## 2. HẠNG MỤC 0 — CHẶN, LÀM TRONG HÔM NAY (~1 giờ)

Ba việc. Không việc nào cần suy nghĩ thêm, cả ba đều là rủi ro mất mát hoặc rủi ro tự gây.

| # | Việc | Chi phí | Vì sao chặn |
|---|---|---|---|
| **0.1** | `git push origin next-round` | 1 phút | `report/vn/` **không tồn tại trên `origin/main`**; 86 tệp/20,290 dòng chỉ nằm trên một đĩa; trong đó **11 JSON đo Jetson không tái tạo được** (cần đúng phần cứng + trạng thái nhiệt 19–20/10). Mất đĩa = mất luận văn **và** mất bằng chứng thiết bị. *Cần một câu cho phép của tác giả.* |
| **0.2** | Sửa `README.md` | 10 phút | `README.md:20` rút `3.02 dec/s`, `README.md:43-44` **dùng lại nó** làm kết quả tiêu đề. Tạo tác này **đã lừa được chủ tịch hội đồng** với kho mở trước mặt (biên bản §2.2). Nó là tệp đầu tiên mọi người đọc. Hai cách: (a) xoá mục Key Results, trỏ về `report/vn/`; (b) thay bằng `3.226 ghim / 2.733 DVFS` kèm nhãn điều kiện. Khuyến nghị **(a)** — README tự khai là SUPERSEDED, đừng để nó phát ngôn số. |
| **0.3** | Ghi ngày bảo vệ + trạng thái A4 vào bảng theo dõi | 5 phút | biến quyết định của toàn bộ kế hoạch |

---

## 3. HẠNG MỤC 1 — ĐÒN BẨY CAO NHẤT, 0 PHÉP ĐO (≈ 1 ngày công, làm trong mọi nhánh)

### T1-1 · A4: văn bản GVHD duyệt đổi hướng — **ưu tiên tuyệt đối**
Hạng mục **duy nhất** mà cả ba hội đồng (05/10, 10/10, 31/10) xác định có thể lật ĐẠT/KHÔNG ĐẠT.
**0 tiến triển trong 3 tuần**, không xuất hiện một lần nào trong bảng theo dõi cập nhật 20/10.
Chi phí: **0 ngày công** — một buổi gặp, một email xác nhận, lưu vào `plans/`.
Nội dung cần duyệt: chuyển từ hướng ban đầu sang hướng "báo cáo kết quả âm + định lượng lạm phát
giao thức". Không có văn bản này, mọi đóng góp khoa học vẫn có thể bị bác trên nền thủ tục.

### T1-2 · Lời khai xuất xứ tiền đăng ký (provenance statement) — 2 giờ
Vấn đề (biên bản §2.1): git **không chứng minh được** phần "tiền" của tiền đăng ký —
`risks.md` vào git 01/10 sau kết quả G14 ngày 29/09; script phán xử vào cùng ngày phán xử;
câu khẳng định "viết trước khi có code" vào 07/10. Lặp lại ở vòng 3 (một commit 20/10 chứa cả
luật và kết quả, trong khi OOF có trên đĩa 14–16/10).

**Không vá được bằng chạy lại. Vá được bằng cách tự nói ra trước.** Viết một trang, đặt trong
`plans/261031-.../` và dẫn chiếu từ Ch3 hoặc Phụ lục:
1. Nêu thẳng: dấu thời gian git của hiến chương cổng **muộn hơn** kết quả nó phán xử; nguyên nhân là tập quán commit theo đợt, không phải chủ ý.
2. Nêu bằng chứng hành vi thay thế: **6/8 cổng công bố TRƯỢT**, gồm giả thuyết trung tâm G14; vòng 3 công bố **0/8 tháp sống sót**; NP-125-D công bố AUC **tụt** 0.8485 → 0.6426 kèm nhãn `WRONG_UNIT` tự đặt. *Không ai lùi ngày hiến chương để tự cho mình trượt.*
3. Nêu đối chứng tích cực: NP-125-D và G21/G25 **có** tiền đăng ký độc lập, commit riêng (`6889261`, 11/10) — chứng tỏ tập quán đã được sửa ngay khi nhận ra.
4. Nếu có bằng chứng ngoài git (email/tin nhắn GVHD, bản nháp có mốc thời gian) → đính kèm; nó **đóng hẳn** vấn đề này.

Nguyên tắc: luận văn này mạnh vì dám công bố kết quả âm. Áp dụng chính nguyên tắc đó cho khuyết
điểm quy trình của chính nó thì khuyết điểm thành bằng chứng về tính nhất quán.

### T1-3 · Ba vết sửa câu chữ trong `report/vn/` (tác giả tự sửa, hội đồng không chạm)

| Vị trí | Hiện tại | Phải thành |
|---|---|---|
| `chuong-05-trien-khai-bien.txt:552-553` | `0.49%` · `vòng lặp tổ hợp` | `0.87%` · `tổng giai đoạn` |
| `chuong-05-trien-khai-bien.txt:1234` | `53.2%` | `52.6%` |
| `chuong-05-trien-khai-bien.txt` §5.9(12) | 6 tạo tác "chưa đồng bộ về kho mã", tự gọi là **"hạn chế nghiêm trọng nhất"** | **xoá/sửa** — đã track từ `df1dab7`; luận văn đang tự khai một hạn chế nghiêm trọng **không còn tồn tại** |
| `chuong-04:1261`, `chuong-04:1276-1279`, `chuong-06:2271-2272` | "đã tiêu **0**" · "cả hai lần chạm **vẫn còn nguyên**" | thêm trạng ngữ mốc: *"tại thời điểm viết mục này, trước ngoại lệ ngày 11/10"* — ngân sách thật là **1 tiêu / 1 còn** (§6.15), bản chất đúng, câu chữ tự mâu thuẫn |

### T1-4 · Sửa câu chữ bất biến đóng băng — 15 phút
`plans/trang-thai-chuong-trinh.md` §4.1 viết "đóng băng 05/10: chỉ sửa văn bản". **Sai như đang
viết**: phép đo mới vào `report/vn/` ngày 08/10 (`9895df4`, lạm phát giao thức) và 11/10.
Viết lại thành: *"số đóng băng 09/10, văn bản 15/10, ba ngoại lệ đã khai tại
`00-trang-thai-bien-soan.txt:14-50`"*. Nội dung đã khai minh bạch và hạch toán đúng — **chỉ nhãn
là sai**, và nhãn sai mời một câu hỏi không cần phải trả lời.

### T1-5 · Đoạn giới hạn hiệu lực cấu trúc — 1 giờ, **không cần đo**
Dữ kiện đã có: **0/151 nguồn mang cả hai nhãn**. Viết thêm một đoạn (Ch6 hoặc Ch7) nói đủ hai chiều:
- Bất lợi: nhãn và danh tính nguồn đồng biến hoàn hảo ⇒ **không thể tách** "nhận ra hành vi" khỏi "nhận ra kênh/phong cách nguồn" bên trong kho này; giới hạn không vá được bằng thêm fold hay thêm mô hình.
- Có lợi: chính vì vậy **leave-source-out là giao thức khó nhất có thể**, và luận văn đã **đo** mức thổi của giao thức dễ hơn (+13.5 pp, CI [+8.0, +20.0], p = 1.39e-41).

Đây là lỗ hổng khoa học sâu nhất phiên này tìm được (biên bản §2.4). **Nói trước khi bị hỏi thì là
chiều sâu; bị hỏi trước khi nói thì là lỗ hổng.**

### T1-6 · Diễn tập phản biện có ghi âm/ghi biên — 2–3 giờ
Theo `reports/phan-bien-12-dien-tap-phan-bien-gay-gat.md`. Bắt buộc gồm:

**Bảng phát ngôn bị cấm** (học thuộc, không ứng biến):
- ❌ "chúng tôi vượt GPT-4o" — **CẤM TUYỆT ĐỐI** (165 lớp zero-shot ≠ nhị phân có giám sát)
- ❌ đặt 84.2 % cạnh 95.81 % hoặc cạnh bảng VLM — **cấm so làn** (§6.1.1)
- ❌ "84.4 %" khi nói kèm số Jetson — luôn là **84.2 %**
- ❌ "hơn baseline" mà không nói **83.25 %** là control
- ❌ `3.02 dec/s`, `331 ms`, `3.91 J`, `95.7 % mAP`, `57/64 calibrated`, `+3.5 pp macro`, "P3 đạt/trượt hợp đồng" — **đã rút**
- ❌ gọi tổng thời gian các giai đoạn là xung nhịp hệ thống

**Câu trả lời chuẩn bị sẵn cho năm câu hỏi chắc chắn bị hỏi:**
1. *"84.2 % có thấp quá không?"* → neo ngoài: Sultani 75.4, RTFM 84.30, MSL 85.62, NTU cross-subject 74–85. 84.2 % **nằm trong dải công bố**, dưới giao thức khó hơn.
2. *"Giả thuyết chính trượt thì luận văn còn gì?"* → đóng góp là **định lượng lạm phát giao thức** (+13.5 pp) và **hệ biên đo thật**; kết quả âm có cổng là kết quả, không phải thất bại.
3. *"Cổng có thật là tiền đăng ký không?"* → trả lời bằng T1-2, **không vòng vo**.
4. *"Sao không dùng mô hình mạnh hơn?"* → đã thử 9 tháp, **0/8 sống sót BH**, p_min 0.093.
5. *"Trần nhãn là bao nhiêu?"* → κ = 0.65 hiện có đo trên **Unsafe-Net 4 lớp + nền, n = 100**, **khác làn**; κ nhị phân iSafetyBench **chưa đo** — nói thẳng là chưa đo (xem T2-1), không suy diễn.

---

## 4. HẠNG MỤC 2 — MỘT PHÉP ĐO DUY NHẤT ĐƯỢC PHÉP (nhánh B và C)

### T2-1 · κ trên chính nhãn nhị phân iSafetyBench — 1–2 ngày
Khuyến nghị **số một** của hội đồng 10/10, sau 3 tuần vẫn **0 bước chuẩn bị**.
Thông số: n ≥ 200 clip, **hai người gán nhãn độc lập**, mù với dự đoán mô hình, lấy mẫu phân tầng
theo nguồn. Tiền đăng ký **trước** khi gán: viết tệp riêng, commit riêng (theo đúng khuôn
NP-125-D đã làm đúng).

**Không tiêu lần chạm test** — đo độ đồng thuận giữa người, không chạm dự đoán mô hình. Khẳng định
điều này trong tiền đăng ký.

Vì sao đây là phép đo duy nhất đáng làm:
- Nó biến lập luận phòng thủ mạnh nhất ("đã sát trần nhãn") từ **phỏng đoán** thành **số đo**.
- Nó **chặn** mọi quyết định nâng cấp xương sống: nếu κ ≈ 0.7 thì 84.2 % **đã sát trần** và việc đóng hướng tăng số (§0) được chứng minh thay vì được suy luận; nếu κ ≈ 0.9 thì còn dư địa thật và đó là hướng sau bảo vệ.
- Kết quả **nào** cũng dùng được. Phép đo không có nhánh thua.

### T2-2 · Viết hai mục còn dở — 0.5 ngày
`§4.5.7` (Ch4 dừng ở 4.5.6) và `§6.1.7` (Ch6 chỉ có tới 6.1.6) — tức **Q4 và Q6 đang dở**, không
phải đã moot (đính chính PB-11, biên bản §5). §6.1.7 nên là mục **ba làn A/B/C và lệnh cấm so làn**
— chính là lá chắn cho câu hỏi nguy hiểm nhất tại buổi bảo vệ.

### T2-3 · 15 trích dẫn thống kê còn thiếu — 2 giờ
Cluster bootstrap, exact McNemar, Benjamini–Hochberg, Nadeau–Bengio, ICC. Kỹ thuật đã dùng đúng;
thiếu nguồn gốc là điểm trừ hình thức dễ mất một cách vô lý.

---

## 5. HẠNG MỤC 3 — SAU BẢO VỆ (nhánh C)

### T3-1 · Công bố: **không có venue gần hạn** — đây là đường dài, không được làm nhiễu việc bảo vệ
Đính chính PB-09 (biên bản §5): **"NeurIPS Negative Results Track hạn 14/11/2026" không tồn tại**.
Negative results là *loại đóng góp* trong main track; hạn NeurIPS 2026 là 06/05/2026, phán quyết
ra 24/09/2026 — **đã đóng**.

| Hướng | Mốc thực tế | Ghi chú |
|---|---|---|
| ICBINB @ ICLR 2027 | chưa có call; bản ICLR 2026 đóng 31/01/2026 | khớp nội dung nhất (kết quả âm + chẩn đoán) |
| Workshop CVPR 2027 | hạn **đề xuất workshop 04/11/2026**, thông báo 22/12/2026, họp 20–21/06/2027 | nộp bài ~03/2027 |
| WACV 2027 | camera-ready 03/11/2026, hội nghị 04–08/01/2027 | **quá muộn** cho vòng này |
| Tạp chí (không hạn) | bất kỳ lúc nào | đường thực tế nhất |

⇒ Công bố là việc của **Q1-2027 trở đi**. Khung bài: *định lượng lạm phát giao thức trên phát hiện
hành vi không an toàn* + *hệ biên đo thật* + *kết quả âm có tiền đăng ký*. Khung này đứng được;
đừng gọi nó là "phát hiện mới" — lạm phát do rò rỉ giao thức là hiện tượng đã biết, **cái mới là
độ lớn đo được và hệ quả xoá khả phân biệt kiến trúc**.

### T3-2 · Khoá phiên bản — 1 giờ
Không có `requirements.txt`/lock ở gốc kho (biên bản §2.7). Đóng băng `transformers`,
`open_clip_torch`, `torch`, TensorRT-python; **pin revision HF Hub** cho trọng số tháp. Hiện tại
`export-round3-towers-onnx.py` dựng lại được engine *tương đương trong sai số cổng*, không phải
engine **đã đo**. Nếu còn khôi phục được diff của `6319b21-dirty` → lưu lại; đó là khe hở duy nhất
trong chuỗi truy xuất vốn đã kín tới từng chữ số.

### T3-3 · Báo lỗi upstream TensorRT (Q14) — 1–2 ngày
Chuỗi lập luận đã **kín** (loại trừ xuất ONNX, nạp dữ liệu, hiệu chuẩn, **và** giả thuyết cạnh
tranh precision-scale do chính tác giả tự đề ra rồi tự bác). TensorRT 10.3.0 chuyển đổi sai đồ thị
MobileCLIP-S1, sai **bất biến** qua 2 độ chính xác × 4 cỡ lô. Cần bisect đồ thị để khoanh node.
Giá trị: đóng góp cộng đồng thật, độc lập với luận văn.

### T3-4 · Đóng hướng mở rộng nguồn — ghi một dòng
ICC 0.43–0.44 ⇒ cần **458–468 nguồn** cho +5 pp = **3.0×** số hiện có (151). S2 kết luận trần 151
nguồn là **trần thật**. Ghi vào bảng theo dõi là **đã đóng, có căn cứ số**, để không ai mở lại.

---

## 6. LỊCH THỰC HIỆN — mốc thật, bảo vệ 31/12

| Tuần | Mốc | Việc | Vì sao ở đây |
|---|---|---|---|
| **Hôm nay** 31/10 | — | ~~0.1 push~~ **XONG** · **0.2 README** · ~~0.3 ghi ngày bảo vệ~~ **XONG** | 0.2 là tạo tác đã lừa được hội đồng, còn mở |
| **T1** 01–07/11 | **gặp GVHD** | **T1-1 A4** · **T2-1 khởi động: tiền đăng ký κ + tìm người gán nhãn thứ hai** · T1-3 ba vết sửa Ch5 · T1-4 nhãn đóng băng *(đã làm trong bảng theo dõi)* | A4 và κ đều **phụ thuộc người khác** ⇒ phải vào tuần đầu. Mọi thứ còn lại tôi/tác giả tự làm được nên hoãn được, hai việc này thì không |
| **T2** 08–14/11 | | T1-2 lời khai xuất xứ · T1-5 đoạn hiệu lực cấu trúc · T2-3 15 trích dẫn thống kê · **T2-1 gán nhãn** | việc viết, không phụ thuộc ai |
| **T3** 15–21/11 | | **T2-1 chốt κ** · T2-2 viết §4.5.7 + §6.1.7 · (nếu hạn CVPR 2027 còn: đề xuất workshop **04/11** — đã quá, bỏ) | κ phải xong trước khi viết §6.1.7 vì nó có thể đổi câu "sát trần nhãn" |
| **T4** 22–28/11 | | **T1-6 diễn tập lần 1** — có ghi âm, có người hỏi thật | sớm **5 tuần** trước bảo vệ ⇒ lỗ hổng phát hiện ra còn kịp vá |
| **T5–T6** 29/11–12/12 | | vá những gì diễn tập lần 1 làm lộ ra · T3-2 khoá phiên bản · (tuỳ sức) T3-3 báo lỗi TensorRT | vùng đệm thật, không phải vùng đệm trên giấy |
| **17/12** | **ĐÓNG BĂNG LẦN HAI** | **ngừng sửa mọi thứ.** Không sửa văn bản, không đo, không chạm kho | lịch sử dự án: mốc 05/10 tràn 10 ngày vì không có mốc này |
| **T7–T9** 18–30/12 | | **chỉ diễn tập.** T1-6 lần 2 (~24/12) · học bảng phát ngôn bị cấm · 5 câu trả lời chuẩn bị | hai tuần cuối chuyển hoàn toàn từ *làm* sang *trình bày* |
| **31/12** | **BẢO VỆ** | | |
| Sau 31/12 | | T3-1 công bố (Q1-2027) · T3-3 upstream · T3-4 đóng hướng mở rộng nguồn | không việc nào trong số này ảnh hưởng kết quả bảo vệ |

Ba việc có đòn bẩy cao nhất, nếu chỉ làm được ba: **T1-1 (A4)** · **T1-6 (diễn tập)** · **T2-1 (κ)**.
Hai trong ba phụ thuộc người khác ⇒ **cả hai nằm ở tuần 01–07/11**.

**Đường tới hạn (critical path):** A4 → (không có) · κ: tìm người gán nhãn → tiền đăng ký → gán
n ≥ 200 → tính κ → viết §6.1.7 → diễn tập lần 1. Chuỗi κ dài **3 tuần** với một mắt phụ thuộc
người; đó là chuỗi duy nhất có thể trượt lịch, nên nó quyết định mốc khởi động.

## 7. ĐIỀU KHÔNG ĐƯỢC LÀM

1. **Không chạm test set.** Còn **1 lần** trong ngân sách 2 (1 đã tiêu cho G21+G25, 11/10). T2-1 không tiêu lần nào.
2. **Không sửa số trong `report/vn/`.** Đóng băng còn hiệu lực; mọi việc ở Hạng mục 1 là câu chữ, nhãn, và đoạn giới hạn.
3. **Không mở lại hướng tăng độ chính xác** trước bảo vệ (căn cứ §0). 61 ngày **không** đổi kết
   luận này: ICC vẫn đòi 3.0× kho dữ liệu, BH vẫn 0/8 sống sót. Thời gian dư không mua được
   dữ liệu không tồn tại.
4. **Không so làn.** 84.2 % (iSafetyBench, nhị phân, có giám sát) không bao giờ đứng cạnh 95.81 % (Unsafe-Net, 4 lớp) hay bảng zero-shot 165 lớp.
5. **Không đẩy nhánh mà không có câu cho phép của tác giả** (0.1 đang chờ).
6. **Không commit thông tin bí mật** (khoá API, dotenv, thông tin đăng nhập).
7. **Không sửa gì sau 17/12.** Mốc đóng băng lần hai là ràng buộc, không phải gợi ý.

---

## 8. CÂU HỎI CHƯA GIẢI QUYẾT

1. ~~**Ngày bảo vệ?**~~ **ĐÃ TRẢ LỜI 31/10: 31/12/2026** ⇒ nhánh **C**. Xem §1 và §6.
2. **A4 bị chặn bởi lịch GVHD, hay bị quên?** Nếu bị chặn, chặn ở đâu.
3. Có **bằng chứng ngoài git** nào xác nhận thời điểm chốt 8 cổng và luật vòng 3 (email/tin nhắn GVHD)? Nếu có → T1-2 rút ngắn còn một đoạn và vấn đề §2.1 **đóng hẳn**.
4. Có **người gán nhãn thứ hai** cho T2-1 không? **Câu hỏi khẩn nhất còn lại** — nó nằm trên đường
   tới hạn dài 3 tuần (§6). Nếu không có người: κ không đo được, lập luận "sát trần nhãn" phải nói
   thẳng là chưa đo (vẫn tốt hơn suy diễn), và tuần 01–07/11 chuyển sang T1-2/T1-5.
5. Tài liệu gốc về cách iSafetyBench thu thập theo nguồn — cần để khép câu hỏi hiệu lực cấu trúc; nguồn công khai **không nêu** cách chia.
6. Có khôi phục được diff của `6319b21-dirty` lúc đo không?
7. Được phép mở `plans/` trước hội đồng, hay chỉ trả lời bằng lời?
8. Đã có buổi diễn tập miệng nào chưa ghi lại, hay T1-6 là lần đầu?
