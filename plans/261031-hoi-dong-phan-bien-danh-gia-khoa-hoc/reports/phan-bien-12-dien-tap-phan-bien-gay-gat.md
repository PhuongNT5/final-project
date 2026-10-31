# PHẢN BIỆN 12 — DIỄN TẬP PHẢN BIỆN GAY GẮT

Ngày 31/10/2026 · Vai: người phản biện khó nhất trong phòng · Chỉ đọc, không chạm `report/vn/`, không chạy thực nghiệm.
Quy ước bắt buộc: không đặt 84.2% (iSafetyBench, nhị phân, có giám sát, B1) cạnh 95.81% (Unsafe-Net, 4 lớp) hay cạnh bảng zero-shot VLM 165 lớp. Khi cần, nói "dưới giao thức cùng kiểu". Không lặp 5 câu của S4 (`plans/reports/researcher-2026-10-19-lo-hong-bao-ve-luan-van.md`).

---

## 0. PHÁN QUYẾT TRỤC (mở đầu)

| Trục | Phán quyết | Vì sao (1 dòng) |
|---|---|---|
| Thủ tục / đổi hướng đề cương | **RỦI RO CAO — CHƯA ĐÓNG** | Văn bản GVHD duyệt đổi hướng (A4) ghi "vẫn mở 09/10" ở biên bản 261005; không tìm thấy mục nào trong `trang-thai-chuong-trinh.md` (cập nhật mới nhất 20/10) xác nhận đã có |
| Mục tiêu đăng ký > 95.8% | **KHÔNG ĐẠT theo chữ, có đo thay thế ngoài bản nộp** | B1 đo được 97.7% dưới fold ngẫu nhiên (+13.5 pp so với 84.2% LSO) — phép đo này **không** có trong `report/vn/`, chỉ ở `plans/` |
| Đóng góp khoa học (độ chính xác) | **ĐẠT CÓ ĐIỀU KIỆN** — chỉ đứng nếu tái định vị đúng (phương pháp luận + hệ biên, không phải mô hình thắng) | 2/8 cổng đạt, cả hai thuộc phương pháp luận; G14 trung tâm trượt |
| Tính mới | Không lặp lại ở đây — xem S4 | S4 đã xếp 4 claim vào "re-demonstrate/quantify", không "discovery" |
| Vòng 3 đặc trưng (16/10, sau mốc đóng băng) | **Không tháp nào sống sót BH q=0.05** (p nhỏ nhất 0.093) | Củng cố thêm kết luận âm về kiến trúc, nhưng số này **không** trong bản nộp |

### Xác suất đạt — ba cảnh (xem chi tiết giả định ở mục 4)

| Cảnh | Xác suất đạt (ước lượng) |
|---|---|
| (a) Không làm gì thêm | **55–65%** |
| (b) Chỉ việc chi phí-0 (câu chữ + diễn tập) | **78–85%** |
| (c) Thêm đúng một phép đo (κ trên nhãn nhị phân iSafetyBench, n≥200) | **85–90%** |

---

## 1. MƯỜI BA CÂU HỎI PHẢN BIỆN, XẾP THEO SỨC SÁT THƯƠNG

| # | Câu hỏi | Vì sao sát thương | Câu trả lời bảo vệ được HÔM NAY | Rủi ro |
|---|---|---|---|---|
| 1 | Đề tài đăng ký là "dựa trên YOLO và Transformer" (`PROPOSAL.txt`). Hệ triển khai headline trên Jetson là CLIP ViT-B/32 đóng băng + đầu Transformer — YOLO không nằm trong đường quyết định. Đây có phải đổi đề tài ngầm chưa qua thủ tục khoa? | Đây là rủi ro **thủ tục**, không phải khoa học — hội đồng trước (261005) xác định đây là hạng mục **duy nhất** còn có thể lật phán quyết ĐẠT/KHÔNG ĐẠT, không chỉ ảnh hưởng điểm | **PARTIAL.** Phần câu chữ CÓ: Ch1 §1.4.2 đặt dòng YOLO→CLIP đầu bảng, kèm lý do bằng số (G14 trượt, −0.5 pp, CI [−5.5,+5.0], p=0.683). Phần văn bản GVHD/khoa duyệt đổi hướng: **CHƯA CÓ** — biên bản `bien-ban-tong-hop-hoi-dong.md` mục 8 câu 1 ghi "vẫn mở 09/10"; không thấy cập nhật đóng trong `trang-thai-chuong-trinh.md` | **CAO** |
| 2 | Mục tiêu đăng ký duy nhất có ngưỡng số (> 95.8%) không đạt. Luận văn tự đổi kết luận headline sang "đo lạm phát giao thức" sau khi biết số dưới nhiều giao thức. Đây có là "move the goalposts" kinh điển, hay là một quyết định có chủ ý trước khi biết số? | Nếu hội đồng đọc là hậu nghiệm (biết số xấu rồi mới đổi mục tiêu), toàn bộ uy tín phương pháp luận sụp theo | **PARTIAL.** Logic khoa học CÓ: Ch1 §1.4.2 nói đổi mục tiêu là hệ quả của cổng G14 trượt (cổng đăng ký trước khi số tồn tại, Bảng 3.4). Xác nhận thủ tục độc lập (GVHD) cho đúng quyết định đổi mục tiêu: **CHƯA CÓ** (cùng lỗ hổng A4 ở câu 1) | **CAO** |
| 3 | Sáu trên tám cổng trượt, giả thuyết trung tâm trượt (G14), và vòng 3 (16/10) cho chín tháp đặc trưng mới — không tháp nào sống sót hiệu chỉnh đa so sánh (BH q=0.05, p nhỏ nhất 0.093). Vậy đóng góp mới của em, theo nghĩa "làm được gì mà trước đây chưa có", là gì? | Dồn hai lớp bằng chứng âm (8 cổng + 9 tháp) vào một câu, buộc trả lời "có hay không có gì dương" | **CÓ cho phần trong bản nộp.** Ch1 §1.4.2 và §1.5 định vị lại: đóng góp là (i) phép đo lạm phát giao thức tái lập trên 2 kho độc lập + (ii) hệ triển khai biên đo thật trên thiết bị — không phải cơ chế kiến trúc mới. Vòng 3 (16/10): **CHƯA CÓ trong bản nộp** — là số sau mốc đóng băng, chỉ dùng được như "cùng chiều, ngoài luận văn", không được trích như kết quả của luận văn | **CAO** |
| 4 | κ = 0.65 (Ch1 §1.2.3) đo trên Unsafe-Net, 4 lớp + nền, n=100. Luận văn dùng nó làm "trần nhiễu nhãn" chung, biện minh cho việc không lấy biên nhỏ làm kết quả tiêu đề — kể cả cho headline B1 trên iSafetyBench nhị phân. Trần đó có áp dụng được cho đúng nhãn đang dùng không? | Đây là cơ chế biện hộ quan trọng nhất của toàn luận văn ("chênh lệch nhỏ hơn nhiễu nhãn không giải thích bằng mô hình"); nếu κ sai kho, lập luận sụp cho chính kết quả headline | **CHƯA CÓ.** Hội đồng 10/10 (`bien-ban-hoi-dong-danh-gia-khoa-hoc-va-chien-luoc-vong-sau.md` §3.1) đã tự chỉ ra: "κ=0.65 hiện có được đo trên Unsafe-Net... không phải trên nhãn nhị phân của iSafetyBench". Đây chính là khuyến nghị số 1 của hội đồng đó (đo E1) — **chưa chạy** | **CAO** |
| 5 | Leave-source-out cho số thấp hơn fold ngẫu nhiên. Có phải giao thức khó được chọn làm headline SAU KHI thấy số dưới LSO thấp hơn, để biện minh cho số thấp, hay LSO được chọn làm chuẩn TRƯỚC khi biết số? | Tấn công thiện ý của toàn bộ phương pháp luận — nếu đúng, "lạm phát giao thức" chỉ là một câu chuyện kể lại sau khi biết kết quả | **PARTIAL.** Lập luận phương pháp CÓ: Ch2 §2.4.1 trích Geirhos et al. 2020 [44] + nhánh record-wise/subject-wise CV — LSO là thực hành chống rò rỉ chuẩn, không phải phát minh riêng. Cổng G12 (so LSO với fold ngẫu nhiên) đăng ký trước khi số tồn tại (Bảng 3.4). Mốc thời gian độc lập (bên thứ ba) xác nhận LSO-là-headline được chốt TRƯỚC khi thấy số dưới LSO: **CHƯA CÓ** — cùng lỗ hổng B4 (council 261005): `risks.md` vào git 01/10, sau khi mô hình đã chạy | **TRUNG-CAO** |
| 6 | 0 trong 151 nguồn mang cả hai nhãn (nhãn và nguồn trộn cấu trúc). Vậy điều đó nghĩa là gì cho mọi khoảng tin cậy theo cụm của luận văn — và em có hiểu hệ quả hay chỉ nêu con số? | Kiểm tra hiểu bản chất thống kê, không chỉ thuộc bài; nếu trả lời hời hợt, lộ ra là chưa hiểu mức độ nghiêm trọng của giới hạn chính mình nêu | **CÓ, đã khai trong bản nộp.** Ch1 §1.6(1), Ch6 §6.1.3, Phụ lục B: hệ quả là "mọi khoảng tin cậy theo cụm trong làn tiêu đề đều chứa 0", nên không biên 3–4 pp nào được chứng nhận. Hệ quả định lượng ("cần bao nhiêu nguồn để hẹp lại" — ICC 0.43–0.44, cần 458–468 nguồn): **CHƯA CÓ trong bản nộp**, chỉ ở `plans/reports/analysis-2026-10-20-icc-theo-nguon-va-so-nguon-can-them.md` | **TRUNG-CAO** |
| 7 | Tại sao đóng băng xương sống (CLIP, 87.8M tham số) thay vì tinh chỉnh nó, nếu mục tiêu là đạt độ chính xác cao nhất có thể? | Câu hỏi "hiển nhiên" nhất của một hội đồng không quen lĩnh vực — nếu trả lời lúng túng, có vẻ như một lựa chọn tuỳ ý | **CÓ.** Ch3 §3.1.1 (nguyên tắc 1) + §3.1.2: ràng buộc dữ liệu trước khi là lựa chọn — 1,100/516 clip là quá ít để tinh chỉnh xương sống 87.8M tham số mà không chỉ đo phương sai; đóng băng còn cho phép đối chứng đúng cấu trúc (hai nhánh đọc cùng khung) | **TRUNG** |
| 8 | Vì sao không thử trực tiếp một VLM/mô hình video hiện đại (zero-shot hoặc fine-tune) thay vì CLIP ViT-B/32 cũ? | Hội đồng 2026 kỳ vọng so với SOTA hiện tại; "CLIP B/32" nghe lạc hậu | **PARTIAL.** Trong bản nộp: G16 — VideoMAE-base đã tinh chỉnh trên Kinetics — đã thử, TRƯỢT (biên đúng dấu nhưng CI theo cụm chứa 0, Ch1 §1.4.3 H2). So sánh với VLM zero-shot bị luận văn tự cấm làm baseline (khác giao thức/nhãn, Ch1 §1.5). Vòng 3 (9 tháp đặc trưng hiện đại hơn, 16/10): **CHƯA CÓ trong bản nộp** — chỉ ở `plans/`, kết luận cùng chiều (không tháp nào sống sót BH) | **TRUNG** |
| 9 | iSafetyBench hiện không có đường cơ sở có giám sát nào từ công trình khác để so. Vậy con số 84.2% so với cái gì ngoài chính luận văn tự tạo ra? | Không có đối trọng ngoài — cảm giác "tự chấm bài tự cho điểm" | **CHƯA CÓ.** `bien-ban-hoi-dong-danh-gia-khoa-hoc-va-chien-luoc-vong-sau.md` E4: "hiện iSafetyBench **không có** đường cơ sở có giám sát nào"; việc này được xếp là thí nghiệm giá trị cao nhất của vòng sau, **chưa chạy**. Trong bản nộp chỉ có đường sàn quy tắc đếm (63.3%, tự đo) và tham chiếu tài liệu (không so trực tiếp) | **TRUNG-CAO** |
| 10 | Giá trị thực tiễn công nghiệp của hệ này là gì, khi dải cân bằng theo fold rộng 29.7 pp, ECE = 0.097, và bài toán phát hiện bốn lớp trên Unsafe-Net không tách được khỏi đường cơ sở đoán theo bối cảnh (p = 0.4049)? | Luận văn tự định vị "deployment-led" — nếu giá trị triển khai thực tế yếu, định vị đó mất chỗ dựa | **CÓ, nhưng khiêm tốn.** Ch1 §1.6 giới hạn (4) tự nhận: "giá trị trung bình không dự báo được hiện trường mới", cần hiệu chỉnh tại chỗ theo thứ hạng (Ch8, hạng mục phát triển thứ nhất). Giá trị thật đã giao: chi phí/nhịp/năng lượng đo trên chính thiết bị (Chương 5) — đây là phần thật, tách khỏi độ chính xác | **CAO** |
| 11 | Số đo sau mốc đóng băng (05/10) — protocol inflation +13.5 pp trên B1, ICC 0.43–0.44, vòng 3 BH — có được dùng trong buổi bảo vệ hay không, và dùng thế nào mà không phạm chính nguyên tắc tiền đăng ký mà luận văn đề cao? | Câu hỏi bẫy kinh điển: nếu trả lời "có dùng" mà không phân biệt rõ, phạm chính điều luận văn lên án ở sổ đăng ký số đã rút; nếu trả lời "không biết" thì mất điểm chuẩn bị | **CÓ quy tắc rõ, phải nói đúng khi dùng.** `trang-thai-chuong-trinh.md` mục 4, ràng buộc #1: "report/vn/ đóng băng 05/10: chỉ sửa văn bản. Số mới nằm ở `plans/`." Câu trả lời đúng: các số đó tồn tại, có log, nhưng không phải số của luận văn — nếu hội đồng muốn xem, trình như phụ lục ngoài, có ngày, không lẫn vào bảng kết quả | **TRUNG** |
| 12 | Hợp đồng nhịp ≥ 3 quyết định/giây (hay 1.249/1.427/2.500 ba mức) — có phải một ngưỡng tự đặt cho dễ đạt, không neo vào tiêu chuẩn an toàn lao động nào? | Nếu ngưỡng tự chọn để chắc đạt, "ĐẠT hợp đồng thời gian thực" chỉ là một phép tự thi tự chấm | **CÓ, đã tự khai minh bạch.** Ch5 §5.4.4: ba mức suy từ *f*/24 trên phân bố fps của chính 1,100 clip, **không** từ OSHA/ISO 45001/IEC 61508 hay tiêu chuẩn VN nào. Phát biểu đã sửa thành "giữ kịp tốc độ khung nguồn, biên 2.39–4.79 lần" — không gọi là "chứng nhận an toàn" (đã đóng ở council 09/10, mục B7) | **THẤP-TRUNG** |
| 13 | Nếu con số 84.2% là điểm vận hành thật của hệ đã triển khai — hệ thực sự chạy trên Jetson với vòng lặp cửa sổ trượt — thì con số đó áp dụng được cho chính hệ đang chạy không? | Câu hỏi nối hai trụ (độ chính xác và triển khai) mà luận văn tách riêng — nếu nhập nhằng, mọi con số nhịp/độ chính xác nói cùng một hệ trở thành vô nghĩa | **CÓ, câu trả lời là KHÔNG, và luận văn nói rõ.** Ch5 §5.4.1: quy tắc trích dẫn bắt buộc — không con số độ chính xác nào trong luận văn được đọc là độ chính xác của vòng lặp trên thiết bị; lý do kỹ thuật là khác khoảng thời gian cửa sổ (đánh giá lấy 16 khung trên toàn clip, triển khai lấy 16 khung trong cửa sổ 48/bước 24); chiều thiên lệch **KHÔNG BIẾT** (mục 5.9 hạng mục 15) | **TRUNG** |

---

## 2. CHẾ ĐỘ SỤP CỦA NGƯỜI BẢO VỆ (fail mode — câu lỡ miệng tự phá luận điểm)

| Câu nói CẤM | Vì sao tự sập | Câu thay bằng |
|---|---|---|
| "84.2% của em so với 95.81% của bài báo gốc thì..." | Trộn hai kho (iSafetyBench/Unsafe-Net), hai bài toán, hai giao thức — đúng lỗi mà Ch6 §6.1.1 tồn tại để cấm, với CHÍNH kết quả của mình | "Hai con số thuộc hai kho khác nhau, không so trực tiếp được; dưới giao thức cùng kiểu thì..." |
| "Em vượt GPT-4o / vượt các VLM lớn" | Bảng VLM là zero-shot, 165 lớp; luận văn là nhị phân, có giám sát — so sánh chéo làn ngược hướng nhưng cùng loại lỗi | "Luận văn không so với bảng đó vì khác giao thức và khác số lớp; luận văn định nghĩa một giao thức có giám sát mà bảng đó chưa có số" |
| Trích tổng các giai đoạn đo rời rạc như một con số nhịp hệ thống | Đã bị chính luận văn xác định là sai dấu đổi chiều theo xung nhịp (+7.98% khi ghim, −39.3% khi để tự do) — không phải chặn trên cũng không chặn dưới | "Nhịp hệ thống là số đo vòng lặp tổ hợp (167.291 ms), không phải tổng các giai đoạn — tổng giai đoạn đã được luận văn chứng minh là sai và bị rút" |
| Đọc một trị số chi phí (ms/quyết định, mW) thành trị số độ chính xác, hoặc ngược lại | Mục 5.4.1 đặt quy tắc: mọi số ở Ch5 là chi phí, không phải chất lượng — lẫn hai loại phá vỡ chính ranh giới luận văn tự vạch | "Số này là chi phí (độ trễ/năng lượng); độ chính xác được đo riêng ở Chương 6, trên giao thức khác" |
| Vô tình nhắc lại một số đã bị rút (ví dụ "+3.5 pp macro", "57/64", "95.7% mAP") như số hiện hành | Phục hồi ngầm số đã rút — đúng hành vi mà sổ đăng ký 11 dòng (Phụ lục A.10) tồn tại để ngăn | "Số đó đã bị rút, lý do ghi ở Phụ lục A; số hiện hành là [đúng số trong bản nộp]" |
| Dùng số sau mốc đóng băng (ICC, +13.5 pp trên B1, round-3 BH) như thể là số của luận văn, không gắn ngày/nguồn `plans/` | Phạm chính nguyên tắc tiền đăng ký mà luận văn dùng để tự vệ — một con số đẹp xuất hiện không đúng lúc còn nguy hơn một con số xấu | "Đó là phép đo thăm dò sau mốc đóng băng 05/10, nằm trong `plans/`, không phải số của luận văn — cùng chiều với kết luận đã nộp, không thay thế nó" |
| "Em được 84.2% thay vì 95.8% vì dữ liệu ít" (nói tắt, không nhắc lạm phát giao thức) | Đúng sự kiện nhưng bỏ mất khung chính — nghe như xin lỗi vì thiếu dữ liệu, không như một phát hiện phương pháp luận | "Em có con số chín mươi mấy phần trăm dưới đúng giao thức mà tài liệu dùng, đo được, có log — và chọn không dùng nó, vì phần tăng đó là rò rỉ danh tính nguồn" |

---

## 3. THỨ TỰ TRÌNH BÀY ĐÚNG

**Đề xuất của hội đồng 10/10** (`bien-ban-hoi-dong-danh-gia-khoa-hoc-va-chien-luoc-vong-sau.md` §4): lạm phát giao thức *trước*, accuracy *sau*. Kiểm lại sau khi có ICC + round 3: **vẫn đúng cho phần khoa học, nhưng thiếu một bước đầu bắt buộc.**

| Bước | Nội dung | Vì sao đặt ở đây |
|---|---|---|
| **0 (mới, đứng trước mọi thứ)** | Chủ động khai trạng thái thủ tục: 4/5 thay đổi so đề cương đã có lý do bằng số trong Ch1 §1.4.2; trạng thái văn bản GVHD (CÓ/CHƯA — nói thật) | Đây là hạng mục duy nhất ảnh hưởng ĐẠT/KHÔNG ĐẠT; giấu đến khi bị hỏi mới nói là rủi ro cao nhất của cả buổi. Nói trước, chủ động, khác với thứ tự cũ (không có bước này) |
| 1 | Lạm phát giao thức (đóng góp 1): định nghĩa Δ, hai cổng đạt, đối chứng giả dược | Thiết lập khung đọc TRƯỚC khi bất kỳ số % nào xuất hiện, để câu hỏi về accuracy tự rơi vào khung đã dựng |
| 2 | Accuracy headline 84.2% so sàn 63.3%, đóng khung là **bằng chứng cho phép đo ở bước 1**, không phải tuyên bố cạnh tranh | Không công bố 97.7%/ICC ở đây (ngoài bản nộp); nếu bị hỏi mới nói, đúng khung |
| 3 | Kiến trúc trượt: G14 trung tâm + 4 cổng H2 khác, đối chứng G17/G20 | Đặt sau accuracy để accuracy không bị hiểu lầm là "mô hình dở" — mà là "kiến trúc không thắng, giao thức là vấn đề" |
| 4 | Triển khai biên (đóng góp 2): Jetson, nhịp, năng lượng, ranh giới rõ "đây là chi phí, không là độ chính xác" (mục 5.4.1) | Đặt cuối vì đây là phần không có cổng tiền đăng ký — tách bạch mức độ chặt chẽ khác nhau giữa hai trụ |
| 5 (chỉ khi được hỏi) | Giới hạn sâu: 0/151, κ phạm vi, hợp đồng tự đặt, số sau mốc đóng băng | Không chủ động mở — nhưng phải có câu trả lời sẵn, không ứng biến |

---

## 4. XÁC SUẤT ĐẠT — BA CẢNH, GIẢ ĐỊNH RÕ

**Giả định chung:** hội đồng bảo vệ tương đương mức độ khắt khe của hai hội đồng mô phỏng trước (261005, 261010); "ĐẠT" = tương đương phán quyết "ĐẠT CÓ ĐIỀU KIỆN" trở lên.

| Cảnh | Giả định cụ thể | Xác suất | Vì sao |
|---|---|---|---|
| (a) Không làm gì thêm | A4 (văn bản GVHD) vẫn ở trạng thái không xác nhận; không diễn tập; trả lời ứng biến | **55–65%** | Văn bản `report/vn/` đã rất tự khai và minh bạch (17 lỗ hổng đã đóng 08–09/10) nên nền tốt; rủi ro chính là ứng biến dẫn tới một câu lỡ miệng (mục 2) hoặc bị hỏi đúng Q1/Q4/Q9 (bảng 1) mà không có khung trả lời sẵn |
| (b) Chỉ việc chi phí-0 | A4 được giải quyết bằng một buổi gặp GVHD (council 261010 ghi "0 ngày công, chỉ cần gặp"); diễn tập hỏi-đáp theo đúng thứ tự mục 3; học thuộc bảng cấm ở mục 2 | **78–85%** | Loại bỏ gần hết rủi ro tự sập (mục 2) và rủi ro thủ tục (Q1/Q2); rủi ro còn lại là các câu đòi phép đo chưa có (Q4 κ, Q9 baseline) — vẫn có khung trả lời "CHƯA CÓ" trung thực, chấp nhận được |
| (c) Thêm một phép đo | Như (b), cộng đo κ trên nhãn nhị phân iSafetyBench, n≥200, 2 người gán nhãn độc lập (E1, council 261010 §5) — trình bày đúng là phụ lục ngoài bản nộp, có ngày, không chèn ngầm vào `report/vn/` | **85–90%** | Vá đúng lỗ hổng sát thương nhất còn lại (Q4 — cơ chế biện hộ "trần nhiễu nhãn" cho headline); biến một giả định thành một số đo, mà không phạm nguyên tắc tiền đăng ký (vì trình bày tách bạch, có ngày) |

Rủi ro không giảm ở mọi cảnh: A4 (chữ ký GVHD) nếu không giải quyết được (ví dụ GVHD từ chối hoặc không gặp được trước ngày bảo vệ) thì xác suất đạt ở CẢ BA cảnh giảm mạnh — đây là biến nhị phân ngoài tầm kiểm soát bằng câu chữ.

---

## 5. BA VIỆC ĐÒN BẨY CAO NHẤT (xếp theo phán quyết đổi / ngày công)

| # | Việc | Chi phí | Vì sao đòn bẩy cao nhất |
|---|---|---|---|
| 1 | **Xin chữ ký/văn bản GVHD xác nhận các thay đổi so đề cương (A4)** | ~0 ngày công, một buổi gặp | Hạng mục duy nhất có thể lật ĐẠT/KHÔNG ĐẠT (không chỉ ảnh hưởng điểm) — xác nhận lại ở cả hai hội đồng trước (261005, 261010); chưa thấy đóng ở bản ghi mới nhất |
| 2 | **Diễn tập hỏi-đáp đúng thứ tự (mục 3) + học thuộc bảng cấm (mục 2)** | 2–3 giờ | Chi phí liêm chính = 0 (không cần đo gì mới), nhưng loại gần hết rủi ro tự sập — loại rủi ro có xác suất xảy ra cao nhất trong phòng bảo vệ thực tế (nói nhanh, bị dồn, lỡ miệng) |
| 3 | **Đo κ trên nhãn nhị phân iSafetyBench, n ≥ 200 (E1)** | ~1–2 ngày công người (không cần compute) | Vá đúng cơ chế biện hộ trung tâm ("đã sát trần nhiễu nhãn") hiện đang treo trên một con số đo ở KHO KHÁC — biến giả định thành phép đo, củng cố đồng thời Q1, Q4, Q8 |

---

## 6. CÂU HỎI CHƯA GIẢI QUYẾT

1. Văn bản GVHD/khoa duyệt đổi hướng đề tài (A4) — đã có chưa? Đây là câu hỏi có đòn bẩy cao nhất của toàn bộ báo cáo này.
2. κ trên đúng nhãn nhị phân iSafetyBench — có kế hoạch đo trước ngày bảo vệ không, và còn bao nhiêu ngày?
3. LSO-là-headline được chốt trước hay sau khi thấy số dưới LSO thấp hơn số dưới fold ngẫu nhiên — có mốc thời gian bên thứ ba nào không (cùng lỗ hổng B4, chưa đóng được bằng git nội bộ)?
4. Lịch bảo vệ chính xác là ngày nào — quyết định còn bao nhiêu ngày cho cảnh (b)/(c) ở mục 4.
5. Nếu hội đồng yêu cầu xem trực tiếp số sau mốc đóng băng (ICC, +13.5 pp trên B1, round-3 BH) — người bảo vệ có được phép mở `plans/` trước hội đồng, hay chỉ giữ trong túi riêng để trả lời bằng lời?
