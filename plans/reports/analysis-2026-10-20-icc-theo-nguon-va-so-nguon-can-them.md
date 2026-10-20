# ICC theo nguồn trên LSO-1100 và cỡ nguồn thật sự cần

**Ngày:** 2026-10-20 · **Loại:** phân tích lại trên dữ liệu đã có · **Chạm tập kiểm thử:** 0
· **Ngân sách chạm tập kiểm thử:** vẫn còn 1/2 · **`report/vn/` không bị chạm.**

## 0. Tại sao báo cáo này tồn tại

Mọi phát biểu "cần 300–400 nguồn để đạt công suất 80%" trong kho này đều dựa trên một ICC
**được giả định**, chưa bao giờ đo. `researcher-2026-10-19-source-ceiling-and-corpora.md:313`
tự nói ra điều đó. Q8 đo nó.

**Phát hiện cần nói trước:** con số 300–400 **không nằm trong luận văn**. Tra cả 11 tệp
`report/vn/`: không có "ICC", không có "công suất thống kê", không có "300–400 nguồn", không có
"65%". Chúng chỉ sống trong báo cáo nền ở `plans/`. Nên kết quả dưới đây **sửa một phép chiếu
trong thư mục kế hoạch**, không đụng vào bất kỳ số đã đóng băng 05/10/2026.

## 1. Khai báo bốn phần

1. **Đại lượng.** ICC(1) theo nguồn trên biến đúng/sai từng clip: tỷ lệ phương sai nằm *giữa*
   các nguồn trên tổng phương sai, `var_source / (var_source + var_resid)`.
2. **Giao thức / tập đo.** Dự đoán out-of-fold đã có dưới leave-source-out, 1,100 clip,
   151 nguồn, làn A (iSafetyBench nhị phân). Ba hệ: `enc_isbclip_B1`, `enc_isbclip_B2`,
   `enc_isbclip_ESTA`, tệp `sweep_results/relation_tokens/*_lso-1100_headline-none_oof.json`.
   Không huấn luyện lại, không suy luận lại, không chạm tập kiểm thử.
3. **Đường cơ sở + trị số.** Dải quy đổi đã tiền đăng ký (S2): ICC < 0.10 → ~250 nguồn;
   0.15 → ~350; **> 0.20 → 400+**. Đây là đối tượng bị kiểm định.
4. **Nguồn.** `[ĐO]` cho ICC và phân bố clip/nguồn; `[SỐ HỌC]` cho cỡ nguồn suy ra.

## 2. Ánh xạ clip → nguồn: dùng trường có thật, không suy đoán

Mỗi hàng OOF mang sẵn `fold_id` — chính là khoá mà bộ chia leave-source-out đã nhóm theo.

| cách xác định nguồn | số nhóm | đúng? |
|---|---|---|
| `fold_id` (trường có thật) | **151** | khớp đặc tả corpus |
| quy tắc thô: cắt `clip_id` trước `_trim_` | 152 | **sai** |

100/1,100 hàng (9.1%) lệch nhau. Nguyên nhân: ~100 clip có thêm đoạn `_clip0` trước `_trim_`
(ví dụ `0QHQ2nmdE78_clip0_trim_10.mp4`), quy tắc thô tách chúng thành nguồn riêng. `fold_id`
gộp đúng. Tôi đã kiểm chứng lại độc lập: `fold_id` → 151, quy tắc thô → 152.

**Phân bố clip/nguồn (n = 151):** min 1 · phân vị 25 là 2 · **trung vị 3.0** ·
**trung bình 7.285** · phân vị 75 là 6.5 · **max 102** · std 11.63. Rất mất cân bằng: một nguồn
chiếm 9.3% toàn corpus.

## 3. Phát hiện cấu trúc: nhãn bị trộn lẫn với định danh nguồn

**0 trong 151 nguồn mang cả hai nhãn** — mỗi nguồn hoặc toàn an toàn hoặc toàn không an toàn.
Độ chính xác theo nguồn vì thế lưỡng cực: 111/151 nguồn đúng 100%, 15/151 đúng 0%.

Đây là động lực chính của ICC cao bên dưới. Và **luận văn đã nói đúng điều này**:
`00-tom-tat.txt:213` — "Trên iSafetyBench, **0 trong 151 nguồn** mang cả hai nhãn, nên nhãn và
định danh nguồn bị trộn lẫn về mặt cấu trúc; hệ quả là khoảng tin cậy theo cụm rất rộng, và luận
văn không đổi đơn vị lấy mẫu để làm chúng hẹp lại."

Nói cách khác: ICC đo được **không phải một phát hiện mới chống lại luận văn**, nó là **lượng
hoá một giới hạn luận văn đã tự nêu bằng lời**. Tôi tự xác nhận lại trên dữ liệu: 0 nguồn trộn.

## 4. Mô hình

`statsmodels.formula.api.mixedlm("correct ~ system", groups=source).fit(reml=True)` — mô hình
hỗn hợp **tuyến tính** trên biến 0/1. Lý do chọn tuyến tính: statsmodels không có GLMM logistic
MLE cổ điển (không có tương đương `lme4::glmer`), và báo cáo nền khuyến nghị ICC tuyến tính đúng
cho trường hợp cụm mất cân bằng nặng (1–102 clip/nguồn). Bản logistic cũng chạy để đối chiếu:
`BinomialBayesMixedGLM.fit_vb()` (xấp xỉ biến phân, không phải MLE chính xác). Cả hai lần đều hội
tụ sạch, **không lần nào chạm biên phương sai bằng 0**.

## 5. ICC đo được

Cùng 151 nguồn / 1,100 clip cho cả hai cặp.

| cặp hệ | var_source | var_resid | **ICC (LMM)** | ICC (ANOVA) | KTC 95% bootstrap | ICC (logistic-VB) |
|---|---|---|---|---|---|---|
| B1 vs B2 | 0.0642 | 0.0855 | **0.4289** | 0.4063 | **[0.2564, 0.5484]** | 0.4267 |
| B1 vs ESTA | 0.0646 | 0.0816 | **0.4418** | 0.4011 | **[0.2436, 0.5688]** | 0.4569 |

KTC từ 2,000 lần lấy lại mẫu **theo nguồn**. Ba bộ ước lượng khác nhau cho cùng một câu trả lời
trong khoảng ~0.01–0.03; hai cặp hệ khác nhau cho cùng một câu trả lời trong khoảng ~0.01–0.02,
KTC chồng nhau gần hết. **ICC ổn định — nó là tính chất của cấu trúc nguồn trong corpus, không
phải tạo tác của việc chọn cặp nào.**

## 6. Cỡ nguồn cần, tính lại từ ICC đã đo

Hệ số thiết kế `DE = 1 + (m−1)·ICC` với `m = 7.285` (đo được, không giả định).
`n_nguồn = n_không_cụm · DE / m`.

| cặp / ICC | +5 pp | +3 pp | +2 pp |
|---|---|---|---|
| B1 vs B2, ICC 0.429 | **458** | 1,333 | 3,062 |
| B1 vs ESTA, ICC 0.442 | **468** | 1,362 | 3,130 |
| B1 vs B2, biên KTC (0.256 / 0.548) | 324 / 551 | 942 / 1,603 | 2,164 / 3,685 |
| B1 vs ESTA, biên KTC (0.244 / 0.569) | 314 / 567 | 913 / 1,650 | 2,098 / 3,791 |

**Kết luận đối chiếu với dải tiền đăng ký.** ICC đo được 0.43–0.44 **nằm trên mọi ngăn**, kể cả
ngăn mở "> 0.20 → 400+". Ngay cả **sàn KTC** (0.24–0.26) vẫn thuộc ngăn trên cùng. Điểm ước
lượng cho +5 pp là **458–468 nguồn**, tức **xấu hơn** mức "400+" từng ngụ ý, và **gấp ~3.0–3.1
lần** 151 nguồn hiện có.

Ở +3 pp và +2 pp, cỡ cần (1,300+ và 3,000+ nguồn) **nằm ngoài mọi thứ dải gốc từng bàn** và
không đạt được bằng cách thêm nguồn ở quy mô thực tế.

## 7. Điều này làm phát biểu chiến lược mạnh hơn, không yếu đi

Phát biểu cũ: *"thêm tính toán thì mua được nhưng không giúp gì; thêm nguồn thì giúp nhưng
không có."* Nửa sau giờ có số: thêm nguồn **giúp ít hơn** mức từng chiếu, và mức "có thể giúp"
bắt đầu từ ~460 nguồn cho hiệu ứng +5 pp — chỉ riêng hiệu ứng lớn nhất trong thang tháp vòng 3
(siglip2B16 + meanpool, +4.00 pp) đã **nhỏ hơn** ngưỡng +5 pp đó.

Ba phát biểu **được phép** nói:
- Trần của vòng 3 là **thống kê**, không phải tính toán. (Đã có bằng chứng thiết bị.)
- ICC theo nguồn đo được 0.43–0.44 (KTC [0.24, 0.57]), **trên** ngăn cao nhất của dải tiền đăng ký.
- Nguyên nhân cấu trúc là nhãn trộn lẫn với nguồn (0/151 nguồn mang hai nhãn) — **luận văn đã nêu**.

Hai phát biểu **không được** nói:
- Không nói "cần 300–400 nguồn" nữa trong tài liệu kế hoạch mới; con số đo được là 458–468.
- Không nói ICC này phủ định kết luận nào trong `report/vn/` — luận văn không chứa phát biểu ICC
  hay công suất nào để phủ định.

## 8. Câu hỏi chưa giải quyết

1. `p1 = 0.80` làm đường cơ sở trong công thức cỡ mẫu là **số tròn chưa dò tìm**, chọn vì gần
   dải 81–84% của các nhánh thật, **không** phải khớp độc lập. Kết quả tỷ lệ theo `p(1−p)`.
2. Công thức dùng là hai tỷ lệ **độc lập**. Công thức ghép đôi / McNemar với độ bất đồng thật sẽ
   hạ cỡ cần xuống một phần — chưa tính; vì vậy các số trên là **chặn trên bảo toàn**.
3. KTC bootstrap dùng bộ ước lượng ANOVA dạng đóng mỗi lần lấy mẫu thay vì khớp lại MixedLM
   2,000 lần; điểm ước lượng ANOVA (0.40–0.41) lệch nhẹ so với REML (0.43–0.44), chưa đối chiếu
   chính thức.
4. Chỉ 2 trong ~10 cặp LSO-1100 có sẵn được kiểm; nhóm mô hình thực thể E0/E1/E3 (backbone khác)
   chưa thử.
5. Việc `y_true` hằng trong mọi nguồn: tạo tác cách dựng corpus của iSafetyBench, hay tính chất
   thật của dữ liệu? Luận văn nêu *hệ quả* nhưng không nêu *nguyên nhân*. Cần tra tài liệu gốc
   của iSafetyBench — chưa làm.
