# Phản biện 06 — Tường thuật & Rủi ro (R6)

Ngày: 10/10/2026. Đối tượng: phát biểu của học viên "kết quả 7x%–8x% còn thấp, có vẻ âm", đối chiếu
`bien-ban-tong-hop-hoi-dong.md` (05–09/10), `phan-bien-04-dong-gop-khoa-hoc.md`, `bao-cao-giai-trinh...`,
`PROPOSAL.txt`, Ch7/Ch8/Tóm tắt.

## 1. Phán quyết về tiền đề của học viên

**Tiền đề SAI ở phần diễn giải, ĐÚNG ở phần dữ kiện thô.** 84.2% (LSO) so với 63.3% (sàn tiền đăng
ký) là +20.9 pp trên sàn — không thấp theo khung quy chiếu đúng. Cảm giác "thấp" của học viên chỉ
đúng khi neo vào 95.8% — số của BÀI BÁO GỐC, đo trên giao thức 125-clip không rõ ràng buộc nhóm.
Luận văn tự mình đo được: CHÍNH mô hình triển khai đạt 97.7% nếu chuyển sang fold ngẫu nhiên (cùng
kiểu giao thức như tài liệu dùng) — chỉ +13.5pp so với 84.2%, p=1.39e-41. Vậy khoảng cách tới 95.8%
không phải "mô hình kém" mà là "thước đo khác".

**Lập luận mạnh nhất cho phía "kết quả thấp/âm" (học viên):**
- Mục tiêu đăng ký là >95.8%. Không đạt theo chữ, không thể lập luận vòng quanh việc này.
- 6/8 cổng tiền đăng ký trượt, kể cả giả thuyết trung tâm (G14, p=0.683) — nếu chỉ đếm cổng, trông như thất bại toàn diện.
- Dải cân bằng theo fold 29.7pp (Ch8 §8.1.5) — mô hình không ổn định, phòng thủ "giao thức đúng" không che được phương sai cao.
- Đối chứng G19 (p=0.0476) sát ngưỡng 0.05 — biên an toàn thống kê mỏng.

**Lập luận mạnh nhất cho phía "đang bị đọc sai" (hội đồng/phản biện 4 đã ủng hộ hướng này):**
- Phép đối chứng tham số khớp (cùng model, cùng data, chỉ đổi fold) là thiết kế thực nghiệm sạch nhất có thể để tách nguyên nhân — không phải một "lý do" mà là MỘT ĐO LƯỜNG.
- Tái lập trên 2 kho độc lập (Unsafe-Net +13.0–21.8pp, iSafetyBench +15.4pp) — cùng chiều, cùng bậc độ lớn → không phải nhiễu của 1 kho.
- Đối chứng giả dược G19 cách ly được yếu tố "fold khó hơn về cấu trúc" (+1.6pp) khỏi "rò rỉ danh tính nguồn" (+13.8pp) — loại được phản biện "LSO chỉ khó hơn tự nhiên".
- Luận văn ĐO ĐƯỢC số cao (97.7%) và CHỌN không công bố nó làm tiêu đề — hành vi ngược hẳn với "kết quả thấp nên phải biện hộ".

**Adjudication:** Khi so đúng đơn vị (protocol-matched), 84.2% là một con số TỐT, không thấp. Khi
so sai đơn vị (84.2% vs 95.8% khác giao thức, khác kho — làn A vs làn C), nó trông thấp — và đây
chính là LỖI mà luận văn chứng minh là lỗi hệ thống của literature. Học viên đang vô tình rơi vào
đúng cái bẫy nhận thức mà luận văn của chính mình dựng ra để vạch trần. Mỉa mai nhưng là sự thật
sư phạm quan trọng nhất cần nói với học viên.

Nhưng: điều này KHÔNG xoá việc mục tiêu đăng ký không hoàn thành theo chữ (A2), và KHÔNG xoá việc
6/8 cổng trượt là một con số thật, sẽ bị hỏi trần trụi. Phần 2 xử lý rủi ro đó.

## 2. Năm câu hỏi khó nhất của hội đồng + câu trả lời mạnh nhất

**Phản ứng thực tế nếu hội đồng chỉ nghe "6/8 cổng trượt" + "84.2%" không kèm ngữ cảnh:** sẽ đọc
thành "đồ án thất bại, không đạt accuracy đăng ký, hoang mang không biết đóng góp gì". Framing này
SỤP ngay khi bị hỏi vặn nếu không có câu trả lời số liệu đi kèm. Framing "giao thức rò rỉ, 97.7% đo
được nhưng không công bố" SỐNG SÓT hỏi vặn vì có log, có p-value, có đối chứng — nhưng chỉ sống nếu
trình bày ĐÚNG THỨ TỰ: nói lạm phát giao thức TRƯỚC khi hội đồng tự tính nhẩm 84.2% < 95.8%.

1. **"Đề cương hứa >95.8%, em được 84.2%. Vậy là KHÔNG đạt mục tiêu, đúng không?"**
   → Mạnh nhất: "Đúng, không đạt theo chữ. Nhưng em đo được: chính mô hình này, dưới đúng kiểu
   giao thức mà con số 95.8% dùng (fold ngẫu nhiên bỏ qua nguồn), cho 97.7% — p=1.39e-41. Em có số
   chín mươi mấy phần trăm và chọn không dùng nó, vì giao thức đó bị em chứng minh là phồng.
   84.2% là số em tin, không phải số em không đạt tới được." (Không nói "em vượt 95.8%" — cấm trộn làn A/C.)

2. **"Vậy làm sao biết 84.2% không phải là mô hình kém, mà đúng là do giao thục?"**
   → "Đối chứng giả dược G19: tạo định danh nguồn giả, khớp đúng 3 thuộc tính cấu trúc của nguồn
   thật, chỉ phá xuất xứ. 20 seed, không seed nào chạm số nguồn thật, cách 12.5pp, p hoán vị=0.0476.
   Tách được: +1.6pp do fold khó hơn, +13.8pp do rò rỉ danh tính. Đây là đo, không phải suy luận."

3. **"6/8 cổng trượt — vậy luận văn kết luận gì là DƯƠNG?"** (câu hỏi sống còn nhất — xem bienban §7.10)
   → "Hai cổng đạt đều là cổng phương pháp (G10: kho không giải được bằng quy tắc đếm; G12: lạm
   phát giao thức có ý nghĩa, p=3.58e-45, tái lập 2 kho độc lập). Sáu cổng trượt đều là cổng KIẾN
   TRÚC (phân rã thực thể không thắng toàn khung). Đọc đúng: luận văn bác bỏ giả thuyết kiến trúc
   ban đầu bằng bằng chứng mạnh, và tìm ra một hiện tượng đo lường mạnh hơn giả thuyết ban đầu."
   Rủi ro: nếu hỏi tiếp "vậy có đóng góp KIẾN TRÚC nào không" → phải nói thẳng KHÔNG, không vòng vo.

4. **"Hiện tượng rò rỉ do chia fold đã biết từ y sinh 2012 (Kaufman). Cái mới của em là gì?"**
   → "Không phải phát hiện hiện tượng — phương pháp ĐO. Đối chứng tham số khớp (cùng model/feature/
   train recipe, chỉ đổi fold) + tái lập 2 kho độc lập + đối chứng giả dược tách nguyên nhân + đo
   hệ quả bậc hai (nén phân biệt kiến trúc 5.1→1.4pp, đảo thứ hạng) — domain an toàn lao động công
   nghiệp chưa ai làm phép đo định lượng này." Rủi ro còn mở: PB4 ghi "THIẾU Geirhos 2020" đã đóng
   09/10 — câu trả lời bây giờ an toàn hơn trước.

5. **"Cho xem văn bản GVHD duyệt đổi hướng đề tài từ YOLO–Transformer sang phát hiện phương pháp luận."**
   → Đây là câu KHÔNG có câu trả lời "mạnh" bằng lời — chỉ có câu trả lời bằng tài liệu (A4 còn mở).
   Nếu chưa có: "Thưa hội đồng, em đã gửi báo cáo giải trình 5 thay đổi cho GVHD ngày 10/10, đang
   chờ chữ ký; nội dung lệch được khai tại mục 1.4.2 trước mọi kết quả, không phải phát hiện muộn."
   Đây là câu trả lời THỦ TỤC, không phải khoa học — không có "số" nào cứu được nếu không có chữ ký.

## 3. Hai con đường chiến lược

| Đường | Điều kiện phải đúng | Chi phí ngày (còn ~5–7 ngày tới bảo vệ) | Rủi ro thất bại | Tác động liêm chính |
|---|---|---|---|---|
| **A — Giữ headline phương pháp luận, gia cố** | GVHD ký văn bản giải trình (A4); hội đồng chấp nhận "kết luận chính = phát hiện phương pháp luận" (câu hỏi mở §7.2 của bienban); video demo dự phòng quay xong (C6, gộp luôn B1) | 1–2 ngày (GVHD + 1 buổi quay demo). Không cần đo lại gì — mọi phép đo đã có trong bản nộp | Nếu GVHD/hội đồng KHÔNG chấp nhận định hướng này → không còn "deliverable chính" nào khác để trình, phải vớt P3 (YOLO) tại chỗ, rủi ro bị lỡ | **AN TOÀN NHẤT.** Không chạm sổ đăng ký số rút, không chạm cổng đã đóng băng; chỉ thêm 1 chữ ký thủ tục |
| **B — Chạy đo thêm để nâng accuracy trước bảo vệ** | Phải tìm được cấu hình/giao thức/data mới cho số đẹp hơn mà KHÔNG phá leave-source-out (vì phá LSO = tự mâu thuẫn với toàn bộ luận điểm G12); nếu dùng data/giao thức mới → phải tiền đăng ký LẠI, mà mốc đóng băng đã qua (05/10) | Cao: cần ≥3–5 ngày để chạy sweep, kiểm định, viết lại Ch6 — rủi ro không xong trước bảo vệ giữa 10/2026 | **Thất bại mode rõ: con số sau mốc đóng băng không có giá trị tiền đăng ký** → đúng việc mà Ch6 toàn bộ luận văn tồn tại để phản đối (câu nói chính học viên trích ở §1 biên bản: "mục 6.11.3 của chính luận văn tuyên bố rút số là hành vi mà Chương 6 tồn tại để phản đối") | **PHÁ HOẠI.** Một con số đo SAU 05/10 rồi công bố như headline = tự tay xoá giá trị của toàn bộ sổ đăng ký + tiền đăng ký — phản biện sẽ hỏi "sao số này không có trong mốc đóng băng?" và không có câu trả lời nào cứu được |

**Căng thẳng trung tâm, nói thẳng:** B không chỉ tốn ngày — B PHÁ CHÍNH LUẬN ĐIỂM của luận văn. Nếu
học viên đi tìm một số đẹp hơn SAU 05/10 rồi đưa vào bảo vệ, câu hỏi hiển nhiên là "con số này được
đăng ký trước hay sau khi biết kết quả?" — và câu trả lời sẽ đúng là "sau", đúng kiểu hành vi mà
G19/G12/sổ đăng ký rút số toàn bộ được viết ra để lên án. Điểm liêm chính đã xây (2 ngày 08–09/10
đóng 17 lỗ hổng) sẽ sụp trong 1 câu hỏi. **Road A là đường duy nhất hợp lý còn lại.**

## 4. Thay đổi hiệu quả nhất

**Diễn tập trả lời miệng theo đúng THỨ TỰ trình bày — không phải thêm bất kỳ số liệu/mục nào.**
Mọi số liệu cần đã có trong bản nộp (84.2/97.7/+13.5pp/p=1.39e-41, G19, 3 ma trận). Rủi ro duy nhất
còn lại là NGƯỜI TRÌNH BÀY tự nói lộn thứ tự (ví dụ bị hỏi dồn rồi vô tình nói "em đạt 84.2% thay vì
95.8% vì dữ liệu ít" — câu trả lời ĐÚNG về mặt lỗi diễn đạt nhưng làm sụp toàn bộ framing phương
pháp luận bằng 1 câu). Chi phí: ~2–3 giờ diễn tập hỏi-đáp theo đúng 5 câu ở mục 2. Tác động liêm
chính: 0 (không chạm số). Tỉ lệ lợi/chi phí cao nhất trong toàn bộ danh sách còn lại — cao hơn cả
việc quay video demo (mục 5 #1 dưới đây) vì video chỉ cứu được nếu demo live hỏng, còn việc trình
bày đúng khung cứu được MỌI câu hỏi về accuracy — câu hỏi chắc chắn sẽ bị hỏi.

## 5. Việc tồn đọng xếp theo mức ảnh hưởng phán quyết

1. **A4 — văn bản GVHD duyệt đổi hướng đề tài.** Biên bản 05/10 tự xếp đây là "hạng mục DUY NHẤT
   còn lại có thể ảnh hưởng mức ĐẠT/KHÔNG ĐẠT; mọi việc khác chỉ ảnh hưởng điểm." Báo cáo giải
   trình đã soạn xong (10/10), chỉ còn chờ chữ ký GVHD. Đây là việc có đòn bẩy TUYỆT ĐỐI cao nhất —
   không ký = rủi ro loại luận văn còn mở; ký = rủi ro A1–A4 đóng toàn bộ.
2. **C6/B1 — video demo dự phòng (gộp với đo accuracy cửa sổ trượt).** Biên bản 09/10 tự xếp "tỉ lệ
   lợi/chi phí cao nhất còn lại" vì MỘT buổi chạy `demo/run-unsafe-behavior-demo.py` đóng được CẢ
   HAI: video dự phòng cho demo live, và phép đo accuracy thực của vòng lặp trên thiết bị (B1). Nếu
   không làm: nếu demo live hỏng trước hội đồng, không có phương án hai — rủi ro trình diễn, không
   phải rủi ro khoa học, nhưng ảnh hưởng ấn tượng trực tiếp.
3. **B10 — p95/p99 + soak cho nhịp tổ hợp.** Tác giả tự nhận là thiếu (câu hỏi mở). Với hệ an toàn,
   thiếu đuôi phân phối là một lỗ hổng thật, nhưng đã được HẠ MỨC thành "giới hạn công bố đúng chỗ"
   — không còn là rủi ro sụp luận điểm, chỉ là điểm trừ nhỏ nếu hội đồng kỹ tính hỏi tới.
4. **C4 — INT8/TensorRT cho tháp CLIP.** Đề cương cam kết, chưa đo. Đã công bố minh bạch là "việc
   chưa làm" (không phải "giới hạn được giấu"). Ảnh hưởng điểm, không ảnh hưởng ĐẠT/KHÔNG ĐẠT, vì
   FP16/TensorRT đã đo và đủ để chứng minh khả thi triển khai biên — INT8 chỉ là tối ưu thêm.

**Thứ tự ưu tiên thời gian còn lại:** A4 (chữ ký, 0 ngày công, chỉ cần gặp GVHD) > diễn tập trình
bày (mục 4) > C6+B1 (1 buổi quay) > B10/C4 (chỉ cần giữ nguyên trạng "đã khai minh bạch", không cần
làm thêm).

## 6. Câu hỏi chưa giải đáp

1. GVHD đã ký văn bản giải trình 5 thay đổi chưa (tính đến thời điểm hội đồng họp)? Nếu KHÔNG, R6
   không thể hạ rủi ro A4 xuống mức "đã xử lý" — mọi phân tích ở mục 3–4 giả định thủ tục này xong.
2. GVHD trả lời câu hỏi mở §7.1 của báo cáo giải trình thế nào ("hội đồng có chấp nhận kết luận
   chính là phát hiện phương pháp luận?") — câu trả lời quyết định Road A có khả thi hay phải vớt
   P3 (YOLO) làm deliverable chính tại chỗ.
3. `demo/run-unsafe-behavior-demo.py` đã chạy ra .mp4 thật và đo được accuracy cửa sổ trượt chưa,
   tính đến ngày hội đồng họp? (Biên bản 09/10 ghi "vẫn mở".)
4. Học viên có kế hoạch diễn tập hỏi-đáp (mục 4) trước bảo vệ không, hay đây là lần đầu các câu hỏi
   ở mục 2 được tổng hợp thành một bộ cố định?
