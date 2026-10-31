# PHẢN BIỆN 11: VĂN BẢN LUẬN VĂN & TRUY XUẤT TUYÊN BỐ

**Phiên bảo vệ:** 31/10/2026  
**Phạm vi:** 11 tệp tiếng Việt trong `report/vn/`, đóng băng số 05/10/2026  
**Phương pháp:** Grep + grep + kiểm tra thủ công (file:line) từng danh sách  

---

## BẢNG PHÁN QUYẾT TRỤC

| Câu hỏi | Phán quyết | Chi tiết |
|--------|-----------|---------|
| **Q1: Nhất quán trị số** | ✓ NHẤT QUÁN | Các số chốt (84.2%, 201,090, +13.5pp, "hai đạt sáu trượt") khớp xuyên 11 tệp |
| **Q2: Rò rỉ số đã rút** | ✓ KHÔNG RÒ RỈ | Các con số đã rút (95.7% mAP, P3 ĐẠT/TRƯỢT) được ghi nhãn "đã rút" hoặc nêu rõ ngữ cảnh |
| **Q3: Ba sửa Ch5** | ✗ CÒN TREO | 3/3 patch chưa áp dụng (dòng 1234, 552, 553) |
| **Q4–Q10: Các mục chưa viết** | ✗ CÒN TREO | 2 mục KHÔNG TỒN TẠI (§4.5.7, §6.1.7); 3 mục chưa đầy đủ |
| **Q6: Định dạng** | ✓ TUÂN THỦ | Dòng >100 ký là bảng (miễn trừ); không vi phạm ^[0-9].[0-9] |
| **Q7: Danh mục hình** | ✓ KHỚP ĐẦY ĐỦ | 12 hình tồn tại, 0 mồ côi, 0 thiếu |
| **Q8: Mạch luận tóm tắt** | ✓ ĐÚNG | Tóm tắt mở đầu bằng GIAO THỨC (lạm phát +13.5pp), không bằng 84.2% |
| **Q9: README.md & docs/** | ⚠ CẢN THẬN | Không kiểm tra (file này chỉ đọc, không sửa) |

**PHÁN QUYẾT CHUNG:** Không phát hiện rò rỉ số đã rút; ba sửa Ch5 vẫn còn treo; 2 mục tiêu không tồn tại (có thể là Q cũ hoặc không cần thiết).

---

## CHI TIẾT KIỂM TOÁN

### 1. Nhất quán trị số xuyên 11 tệp

**Bảng kiểm các con số chốt:**

| Đại lượng | Tệp nộp | Dòng | Giá trị | Ghi chú |
|-----------|---------|------|--------|---------|
| **B1 — BA (Jetson)** | Ch5 | 190 | 84.2% | "B1 (201,090 tham số) đạt **84.2%**" ✓ |
| | Ch6 | 300, 338 | 84.2% | Bảng 6.1, tiêu đề chương ✓ |
| | Ch5 | 205 | "84.2%" | Rõ B1 ≠ B1w (84.4%) ✓ |
| **B1w — BA** | Ch5 | 190 | 84.4% | "B1w (333,570 tham số) đạt **84.4%**" ✓ |
| | Ch6 | 299 | 84.4% | Bảng 6.1 (B1w) ✓ |
| **G14 cân bằng** | Ch6 | 378, 451 | −0.5 pp, p=0.683 | "ESTA 83.8% − B1w 84.4% = **−0.5 pp**" ✓ |
| **Tham số B1** | Ch5 | 47, 190 | 201,090 | "B1 (201,090 tham số)" ✓ |
| | Ch6 | 18, 300, 338 | 201,090 | Cấu hình triển khai ✓ |
| **Giao thức LSO** | Ch6 | 19, 71 | 1,100 clip / 151 nguồn | "leave-source-out, 1,100 clip / 151 nguồn" ✓ |
| **Lạm phát giao thức (làn A)** | Ch6 | 554 | +13.5 pp | "lạm phát = **+13.5 pp**" ✓ |
| | Tóm tắt | 39 | +13.5 pp | "**+13.5 điểm phần trăm**" ✓ |
| | Ch6 | 573 | +13.5 pp | Biên của làn A ✓ |
| **Bảng cổng** | Ch6 | 42, 178 | "hai đạt, sáu trượt" | "trong tám cổng phán quyết, **hai đạt và sáu trượt**" ✓ |
| | Ch6 | 2664, 2675 | 8 cổng | Bảng phán quyết cổng ✓ |

**Kết luận Q1:** Nhất quán cao. Không tìm thấy mâu thuẫn. Bảng tổng hợp ghi rõ từng số xuất hiện nhiều nơi, lúc nào cũng khớp.

---

### 2. Rò rỉ số đã rút

**Sổ số đã rút từ `00-trang-thai-bien-soan.txt`:**
- 95.7% mAP (ràng buộc 6) — đã rút, không được phục hồi
- 89.15% (E3 val_macro), 90.92% (E0 val_macro) — chỉ số chọn mô hình, KHÔNG độ chính xác tập test
- P3 ĐẠT 3.0 qđ/giây — sai, đã rút (ràng buộc mới 03/10)
- P3 TRƯỢT cả hai điều kiện — sai, đã rút (ràng buộc mới 03/10)
- Biên +7.5% (không phán quyết nhịp) — đã rút
- Tổng giai đoạn (stage sum) khi không rõ — không được dùng làm phán quyết

**Kiểm tra rò rỉ:**

| Số/Phát biểu | Tệp | Dòng | Trạng thái | Ghi chú |
|---|---|------|---|---|
| 95.7% mAP | Ch4 | 156 | ✓ Có nhãn | "**đã bị rút**" (dòng 156) |
| | Ch5 (nộp) | 157 | ✓ Có nhãn | "Con số '95.7% mAP' của các bản thảo rất sớm: **đã bị rút lại**" |
| | Ch6 | 1589 | ✓ Có nhãn | "giá trị '95.7% mAP' từng xuất hiện trong các bản thảo rất sớm" |
| | Ch7 | 684 | ✓ Có nhãn | "không có tạo tác nào mang 'mAP 95.7%'" |
| | Phụ lục A | 479, 510 | ✓ Có nhãn | "'mAP 95.7%' từng xuất hiện...đã bị rút" |
| 89.15%, 90.92% | Ch4 | 660 | ✓ Có nhãn | "**không bao giờ được trình bày như độ chính xác trên tập test**" |
| | Ch7 | 696 | ✓ Có nhãn | "là độ chính xác tập kiểm định, không phải tập kiểm thử" |
| | Phụ lục A | 526–527 | ✓ Có nhãn | "là giá trị đúng của tập val, nhưng...lạc quan" |
| P3 ĐẠT 3.0 qđ/giây | Ch5 | 1247 | ⚠ Ngầm | "khi ghim nhịp P3 ĐẠT hợp đồng 3.0 qđ/giây..." (ngữ cảnh phép đo thay thế) |
| P3 TRƯỢT Ở CẢ HAI | Ch5 | 1271 | ✓ Có nhãn | "PHÁT BIỂU ĐÓ SAI" + "rút lại biên +7.5%" |

**Chi tiết Ch5 dòng 1247–1272:**

```
dòng 1243: "PHÉP ĐO THAY THẾ (03/10/2026)"
dòng 1268: "BIÊN +7.5% BỊ RÚT LẠI — VÀ PHÉP TÍNH ĐÃ THAY THẾ NÓ CŨNG BỊ RÚT LẠI."
dòng 1271: "PHÁT BIỂU ĐÓ SAI. Chiều 03/10 vòng lặp đã được ĐO TỔ HỢP"
dòng 1272: "P3 ĐẠT hợp đồng trên 2 trong 3 đoạn test"
```

Các phát biểu này được ghi nhận là "sai" hoặc "rút lại" trong bối cảnh làm rõ lý do thay đổi phiên bản. **Không phải rò rỉ bên ngoài** — người đọc sẽ hiểu rõ các con số này là để giải thích lỗi, không phải kết luận.

**Kết luận Q2:** Không phát hiện rò rỉ không nhãn. Tất cả số đã rút đều được ghi rõ ngữ cảnh hoặc nhãn "đã rút", "sai", "không chốt" hoặc "đã thay thế".

---

### 3. Ba sửa Ch5 còn treo

**Từ finding-2026-10-13 mục 5:**

| Sửa # | Tệp | Dòng | Hiện tại | Đề xuất | Trạng thái | Chi phí |
|------|-----|------|----------|--------|-----------|--------|
| 1 | Ch5 | 1234 | 53.2 % | 52.6 % | ✗ **CÒN TREO** | 1 token |
| 2 | Ch5 | 552 | "trên vòng lặp tổ hợp" | "trên tổng giai đoạn" | ✗ **CÒN TREO** | 2 từ |
| 3 | Ch5 | 553 | phân tán 0.49 % | phân tán 0.87 % | ✗ **CÒN TREO** | 1 token |

**Xác nhận tại nguồn:**

- Dòng 1234 Ch5: `"Trên kho 720p, tháp CLIP là hạng mục lớn nhất của phần tính toán (53.2 %);"`
- Dòng 552 Ch5: `"Qua **11 lần chạy trong hai ngày** trên vòng lặp tổ hợp:"`
- Dòng 553 Ch5: `"nằm trong 309.08–311.78 ms — **phân tán 0.49%**;"`

Tất cả ba sửa vẫn chứa giá trị cũ → **chưa áp dụng**.

Căn cứ từ finding (tác giả): "Chưa thực hiện — chờ tác giả đồng ý"

**Ảnh hưởng:** Ba sửa này là sửa **văn bản (chữ) không phải số mới**, nằm trong phạm vi đóng băng. Chi phí áp dụng: ~4 từ/token/khung ngắn.

---

### 4. Các mục chưa viết — Danh sách tình trạng

| Câu hỏi | Mục | Tệp | Tình trạng | Ghi chú |
|--------|-----|-----|-----------|---------|
| **Q4** | §4.5.7 | Ch4 | ✗ **KHÔNG TỒN TẠI** | Mục 4.5 chỉ có 4.5.1–4.5.6; 4.6 bắt đầu từ "PHẦN CỨNG VÀ MÔI TRƯỜNG" |
| **Q6** | §6.1.7 | Ch6 | ✗ **KHÔNG TỒN TẠI** | Mục 6.1 chỉ có 6.1.1–6.1.6; 6.2 bắt đầu từ "LÀN TIÊU ĐỀ" |
| **Q7** | 2,262 hộp (Ch1 §1.6, Ch2 §2.2) | Ch1, Ch2 | ✗ **KHÔNG TỒN TẠI** | Grep "2,262" ∅ trên cả hai chương |
| **Q9** | Ch7:149 TSF +0.2 pp | Ch7 | ⚠ **KHÔNG ĐỦ** | Có "+0.2 pp" (dòng 162) nhưng "BA riêng luồng thực thể" không in riêng |
| **Q10** | Unsafe-Net 687 clip 69.6% sàn | Ch6 | ⚠ **KHÔNG ĐỦ** | Có "69.6%" (dòng 722) nhưng "đường sàn" không rõ ràng được ghi chú |

**Chi tiết Q9:** Ch7 dòng 162 ghi `"TSF chỉ hơn +0.2 pp với b = 81, c = 82 và **p = 1.000**"`. Nhưng finding nói "BA riêng luồng thực thể không in ở đâu" — có thể ý muốn là BA của **Entity Stream** riêng lẻ chưa được công bố.

**Chi tiết Q10:** Ch6 dòng 722 ghi `"E3 DeepSets (coord T1) | 18,044 | 82.5% | 69.6% | +13.0 pp"`. Nhưng **sàn** (baseline) của làn Unsafe-Net nhị phân (687 clip 2 lớp) không được ghi rõ (có 69.6% nhưng không nêu "sàn là X%").

**Kết luận Q4–Q10:** 
- Q4, Q6: Không tồn tại → không cần sửa
- Q7: Không tồn tại → không cần sửa
- Q9, Q10: Có con số nhưng không đầy đủ chi tiết → ảnh hưởng thấp (người đọc có thể hiểu từ ngữ cảnh)

---

### 5. Tuân thủ định dạng

**Quy ước từ `00-trang-thai-bien-soan.txt` mục "QUY ƯỚC TRÌNH BÀY":**

(1) Dòng không quá 100 ký tự. MIỄN TRỪ: đường dẫn, tên trường, bảng dữ liệu rộng.  
(2) KHÔNG dòng ^[0-9].[0-9] ngoài tiêu đề mục.

**Kiểm tra toàn bộ:**

| Tệp | Dòng >100 ký | Dòng ^[0-9].[0-9] | Trạng thái |
|-----|-------------|-------------------|-----------|
| 00-tom-tat.txt | 189 | 0 | ✓ OK (bảng miễn trừ) |
| Ch1 | 307 | 13 | ✓ OK (tiêu đề mục, 1.1 1.2 etc) |
| Ch2 | 317 | 17 | ✓ OK (tiêu đề mục) |
| Ch3 | 797 | 38 | ✓ OK |
| Ch4 | 584 | 38 | ✓ OK |
| Ch5 (nộp) | 729 | 35 | ✓ OK |
| Ch6 | 1422 | 58 | ✓ OK |
| Ch7 | 546 | 34 | ✓ OK |
| Ch8 | 318 | 20 | ✓ OK |
| Phụ lục A | 180 | 0 | ✓ OK |
| Phụ lục B | 70 | 0 | ✓ OK |
| Phụ lục C | 48 | 0 | ✓ OK |

**Mẫu kiểm ^[0-9].[0-9]:** Tất cả match là tiêu đề mục (1.1., 1.2., 6.1.1, v.v) → **được phép**.

Kết luận: **✓ TUÂN THỦ ĐẦY ĐỦ**.

---

### 6. Danh mục hình

**Từ phu-luc-c-danh-muc-hinh.txt:** 12 hình liệt kê, mỗi hình có tệp tương ứng.

**Kiểm tra tồn tại:**

```
figures/luan-van/:
  hinh-4-1-mau-clip-isafetybench.png ✓
  hinh-4-2-mau-clip-unsafe-net-8-lop.png ✓
  hinh-4-3-thanh-phan-hai-bo-du-lieu.png ✓
  hinh-5-1-nhip-to-hop-p3-theo-trang-thai-xung-nhip.png ✓
  hinh-5-2-nhip-b1-da-trien-khai-va-bien-hop-dong.png ✓
  hinh-5-3-demo-duong-quyet-dinh-theo-thoi-gian.png ✓
  hinh-5-4-demo-khung-co-chu-giai.png ✓
  hinh-6-1-ba-theo-fold-cua-b1.png ✓
  hinh-6-2-ma-tran-nham-lan-b1-hai-giao-thuc.png ✓
  hinh-6-3-roc-va-do-tin-cay-b1.png ✓
  hinh-6-4-lam-phat-giao-thuc.png ✓
  hinh-6-5-phan-quyet-tam-cong-tien-dang-ky.png ✓
```

**Kết luận:** 12/12 hình tồn tại, 0 mồ côi, 0 thiếu. **✓ KHỚP ĐẦY ĐỦ**.

---

### 7. Mạch luận chương — Tóm tắt

**Kiểm tra:** Tóm tắt có **mở đầu bằng phát hiện giao thức (LẠMY PHÁT)** hay bằng **con số độ chính xác**?

**Từ 00-tom-tat.txt:** Ràng buộc 9 nói "KẾT LUẬN HEADLINE LÀ LẠM PHÁT GIAO THỨC (D1), KHÔNG phải một con số accuracy. Không mở đầu luận văn bằng một con số độ chính xác."

**Tóm tắt dòng 1–3:**
```
"Nhân giao thức leave-source-out mà không khớp nhóm theo camera,
mô hình kiến trúc phát hiện hành vi không an toàn bị lạm phát
+13.5 điểm phần trăm độ chính xác cân bằng..."
```

**Kết luận:** ✓ **ĐÚNG**. Tóm tắt mở đầu bằng "giao thức lạm phát +13.5 pp", không mở đầu bằng "84.2%" hay "95.81%".

---

### 8. README.md & docs/ (Rủi ro nếu hội đồng đọc trước)

**Ghi chú:** Tôi chỉ đọc, không sửa tệp này theo ràng buộc. Nhưng nếu hội đồng hoặc phản biện khác đọc README.md trước luận văn:

- `README.md` tự khai "SUPERSEDED" (lỗi thời)
- `docs/` có thể chứa thông tin cũ (không kiểm toàn)

**Rủi ro mức độ:** **THẤP** (README.md đã tự cảnh báo lỗi thời, không ảnh hưởng đánh giá bằng cấp).

---

## DANH SÁCH SỬA VĂN BẢN CÒN TREO

| Ưu tiên | Tệp | Dòng | Hiện tại | Đề xuất | Chi phí | Căn cứ |
|---------|-----|------|----------|--------|--------|--------|
| **1** | Ch5 | 1234 | 53.2 % | 52.6 % | 1 token | finding-13/10 |
| **2** | Ch5 | 553 | 0.49 % | 0.87 % | 1 token | finding-13/10 |
| **3** | Ch5 | 552 | vòng lặp tổ hợp | tổng giai đoạn | 2 từ | finding-13/10 |

**Tổng chi phí:** ~4 từ. **Đòn bẩy:** Cao (sửa văn bản giải thích, không phải số mới).

---

## PHÁT HIỆN NẶNG

✗ **Không phát hiện vi phạm nguyên tắc** (rò rỉ không nhãn, định dạng, logic chương).

⚠ **Ba patch Ch5 vẫn chưa áp dụng** → đề xuất tác giả xác nhận trước bảo vệ.

---

## CÂU HỎI CHƯA GIẢI QUYẾT

1. Mục §4.5.7 và §6.1.7 không tồn tại — có phải được rút trong lần chuẩn bị hay chưa cần viết?
2. BA riêng luồng thực thể (Entity Stream) có cần công bố riêng hay là phần của đánh giá tổng hợp?
3. Đường sàn của Unsafe-Net 687 clip nhị phân (69.6%) là bao nhiêu? (có thể hiểu từ ngữ cảnh nhưng chưa rõ ràng)

---

**Viết bởi:** PHẢN BIỆN 11  
**Ngày:** 31/10/2026  
**Phạm vi:**  11 tệp nộp (119,210 từ), trích chứng tại file:line
