# Phản biện 08 — Trục Phương pháp & Thống kê

Phạm vi: giao thức LSO, 8 cổng tiền đăng ký, quy tắc kiểm định kép, ICC/cỡ nguồn, test-touch,
κ noise-ceiling, công bố kết quả âm. Chỉ đọc kho; không chạy huấn luyện, không chạm test mới.

## 0. Bảng phán quyết

| # | Câu hỏi | Phán quyết | Lý do một dòng |
|---|---|---|---|
| 1 | Giao thức LSO có chặn rò rỉ? | **ĐẠT** | 0/45 cặp fold chung nguồn, 151/151 nguồn giữ đúng 1 lần (`phu-luc-b-cau-hinh-fold.txt` §B.2), tự kiểm lại từ `fold_id` thô khớp |
| 2 | Quy tắc kiểm định kép (margin≥1.46pp + CI + BH q=0.05/m=8): đăng ký trước hay sau khi thấy số? | **CHƯA ĐẠT** | tài liệu "tiền đăng ký" vòng 3 chỉ có DUY NHẤT 1 commit git (20/10), trong khi toàn bộ 8 tháp đã có kết quả thô trên đĩa từ 14–16/10 — không có tạo tác tách rời, có dấu thời gian độc lập, chứng minh luật được khoá TRƯỚC khi nhìn số |
| 3 | BH trên 8 tháp: số học đúng? | **ĐẠT** | thuật toán step-up chuẩn, p min = 0.0928 > 0.05/8=0.00625 đúng, không tháp nào sống sót, không mâu thuẫn CI/p nội tại |
| 4 | Ngân sách chạm test 2/1/1, có giấu lần chạm nào? | **ĐẠT CÓ ĐIỀU KIỆN** | vòng 2/3 tháp chỉ đọc `*_oof.json`, không chạm test; NHƯNG Chương 4 §4.9 (dòng 1261,1276-1279) và Chương 6 §6.14 (dòng 2271-2272) — cùng 11 tệp nộp — còn phát biểu "0/2, cả hai lần chạm còn nguyên" KHÔNG gắn mốc thời gian, mâu thuẫn trực tiếp với Chương 6 §6.15 (dòng 2284,2511) tự thuật lại 1 lần chạm đã tiêu cho G21+G25 ngày 11/10 |
| 5 | ICC 0.43–0.44, 0/151 nguồn hai nhãn — hệ quả nặng hơn gì đã nói? | **ĐẠT CÓ ĐIỀU KIỆN** | luận văn đã tự nêu hệ quả (CI rộng); CHƯA tự nêu rằng cấu trúc này làm phép đo LSO khó phân biệt "học hành vi" với "học nhận diện nguồn/kênh" — đây là câu hỏi giá trị cấu trúc (construct validity), không chỉ câu hỏi công suất thống kê |
| 6 | κ=0.65 "trần nhiễu" có bị suy diễn sang làn A? | **ĐẠT CÓ ĐIỀU KIỆN** | mọi lần dùng số đều giới hạn đúng làn C (`chuong-04:220-222`, `chuong-07:979`); NHƯNG câu rhetorical ở `chuong-04:208` ("κ=0.65 là trần trên của MỌI con số trong luận văn") viết quá rộng, tạo khe hở cho hội đồng đọc sai |
| 7 | Kết quả âm có báo đúng chuẩn? | **ĐẠT** | G11 trượt báo trước G10/E0 (ràng buộc 2, §6.11.1 trước §6.11.2); VideoMAE/zero-shot-ranking/8-tháp-không-sống đều có trong `plans/` với số liệu đầy đủ, không bị cắt |
| 8 | Tự kiểm một con số headline từ OOF thô | **ĐẠT** | BA tính lại từ `enc_isbclip_B1_lso-1100_headline-none_oof.json` = 84.17367% khớp 84.2% công bố |

**Verdict tổng trục: ĐẠT CÓ ĐIỀU KIỆN.** Khung thống kê (bootstrap cụm, McNemar chính xác, BH,
nguyên tắc ưu tiên CI>McNemar, disclosure κ-ceiling, disclosure độ trôi tái lập) được thiết kế và
thực thi đúng kỹ thuật hiếm thấy ở luận văn thạc sĩ. Nhưng có MỘT lỗ hổng tiền đăng ký không thể
kiểm chứng (câu 2) và MỘT mâu thuẫn nội tại chưa sửa trong văn bản đã đóng băng (câu 4) — cả hai
đủ nghiêm trọng để hội đồng phải hỏi trực tiếp tại buổi bảo vệ.

---

## 1. Giao thức leave-source-out — có rò rỉ không

`report/vn/phu-luc-b-cau-hinh-fold.txt` §B.2 (dòng ~trong tệp): 3 kiểm chứng tự chạy —
(a) 0 nguồn chung giữa 45 cặp fold; (b) 151 nguồn phân biệt qua 10 fold test, 0 trùng; (c) 1,100
clip test = 420 unsafe + 680 normal, đúng 42 unsafe/fold.

**Tự kiểm độc lập (không tin lời văn bản):** đọc trực tiếp `fold_id` trong
`sweep_results/relation_tokens/enc_isbclip_B1_lso-1100_headline-none_oof.json` → 151 nhóm, 0 nguồn
mang cả hai nhãn — khớp hoàn toàn với Phụ lục B và với `analysis-2026-10-20-icc-...md` §2-3.

Phát hiện phụ của chính Phụ lục B (§B.5), đáng khen vì tự bất lợi: fold-size là "dấu vân tay giao
thức" — giao thức rò rỉ (StratifiedKFold) luôn cho fold đúng 110 clip, LSO cho fold lệch
(144,107,107,106×7); bảng đối chứng tại §B.5 tự đo lạm phát +15.4pp [10.6,21.5] khi dùng sai giao
thức trên kiến trúc E3. Đây là bằng chứng công cụ, không phải lời khẳng định — **ĐẠT**.

Góc chưa khép: không đọc được chính hàm sinh fold (`ISB.folds`) trong mã nguồn (không tìm thấy
định nghĩa trong các tệp đã rà ở phiên này) — thẩm định hiện dựa trên log xuất ra + tái lập từ
artefact, không dựa trên đọc trực tiếp logic `StratifiedGroupKFold`. Không hạ verdict vì 3 kiểm
chứng độc lập trên dữ liệu thật đã đủ, nhưng ghi vào câu hỏi mở.

---

## 2. Quy tắc kiểm định kép — đăng ký trước hay sau số liệu (câu hỏi nặng nhất)

### 2.1 Luật được dùng
`plans/.../chien-luoc-thuc-nghiem-vong-3.md` §3,§8: một tháp "sống sót" khi (margin > 1.46pp
dải trôi) **VÀ** (CI cụm loại trừ 0) **VÀ** BH q=0.05 trên m=8 reject. `analyze_round3_tower_bh.py`
dòng 94-103 cài đúng thuật toán step-up chuẩn; dòng 131-133 cài đúng phép hội ba điều kiện.

### 2.2 Bằng chứng thời gian — đây là phần tự điều tra, không có trong tài liệu nào đã đọc
```
git log --follow -- chien-luoc-thuc-nghiem-vong-3.md   -> DUY NHẤT 1 commit (8610c74, 2026-10-20 22:35)
git log --diff-filter=A ...                            -> tệp được TẠO (không sửa từ bản cũ) ở commit đó
mtime sweep_results/posthoc_sensitivity/round2/tower*_oof.json  -> 2026-10-14 22:12 .. 2026-10-16 03:38
git log -- analyze_round3_tower_bh.py                   -> commit đầu 8610c74 (2026-10-20 22:35),
                                                            commit sửa 6a5d01c (2026-10-20 22:50,
                                                            "correct the eight-tower BH labels")
```
Nghĩa là: **toàn bộ 8 tháp đã có dự đoán OOF thật trên đĩa 4–6 ngày trước khi "tài liệu đăng ký"
được đưa vào git**, và tài liệu đó — chứa cả luật đăng ký (§1-7) VÀ bảng kết quả (§8) — được viết/
commit trong **một lần**, không có commit riêng nào tách luật ra khỏi số. Không giống NP-125-D
(`00-trang-thai-bien-soan.txt:41-43`: "Tiền đăng ký viết trước khi có số" dẫn một **tệp riêng**,
`tien-dang-ky-np125d-...md`, có xuất xứ tách biệt) — vòng 3 **không có** tạo tác tiền đăng ký độc
lập, có dấu thời gian độc lập, trước khi tháp chạy.

Tự sửa ngay khi viết: ban đầu tôi coi cấu trúc văn bản (§1-7 rồi §8) là bằng chứng thứ tự thời
gian đủ. Sau khi tra git log, phải rút lại — cấu trúc văn bản là một quy ước trình bày, **không**
phải bằng chứng thời gian độc lập, và ở đây hai thứ không trùng nhau.

### 2.3 Vì sao vẫn không coi đây là gian lận, nhưng vẫn phải treo CHƯA ĐẠT
- Giá trị ngưỡng 1.46pp bắt nguồn từ một phát hiện **sớm hơn và tách biệt về git**: commit
  `5644889` (2026-10-15 23:08) chứa `finding-2026-10-14-do-troi-tai-lap-...md` — tự nó không phải
  luật của vòng 3, nhưng là hằng số đo được, khó mà "chọn sau khi thấy số vòng 3" vì nó đo thứ
  khác (độ trôi CPU-thread), không phải margin tháp.
- Denominator 7→8 tự sửa, bản thân sự tự sửa **làm luật khó qua hơn**, không dễ hơn — nếu có động
  cơ gian lận thì hướng sửa phải ngược lại. `analyze_round3_tower_bh.py` dòng 14-19 tự thú nhận lỗi
  đếm và lý do giữ 8. Đây là tín hiệu thiện chí.
- Nhưng thiện chí không thay được bằng chứng thời gian độc lập. Nếu hội đồng hỏi "ai chứng minh
  luật đã khoá trước khi anh/chị nhìn 8 con số" — câu trả lời đúng hiện tại là: **không ai chứng
  minh được bằng git**, chỉ có lời tự thuật trong văn bản.
- Giảm nhẹ quan trọng nhất: **toàn bộ kết quả vòng 3 không vào `report/vn/`** — rà cả 11 tệp nộp
  không ra một dòng nào chứa "vòng 3", "Benjamini", tên các tháp (`siglip2B16`, `metaclipB16`...).
  Vì vậy lỗ hổng này không làm sai một con số nào trong luận văn đã nộp; nó chỉ làm yếu một phần
  bằng chứng hậu thuẫn nằm ở `plans/` mà có thể được dẫn lại miệng khi trả lời phản biện.

**Khuyến nghị cho buổi bảo vệ:** nếu tác giả dẫn kết quả vòng 3 (ví dụ để trả lời "có thử backbone
khác không"), phải nói rõ ngay: đây là khảo sát hậu kiểm (post-hoc), không phải một cổng, và quy
trình đăng ký của nó không có tạo tác thời gian độc lập để kiểm chứng.

---

## 3. Số học BH trên 8 tháp

`sweep_results/posthoc_sensitivity/round3_tower_bh.json`, đối chiếu `analyze_round3_tower_bh.py`:

| Kiểm | Kết quả | Khớp? |
|---|---|---|
| m = 8 (không phải 7) | đúng, `FAMILY` dict dòng 50-59 có 8 khoá | ĐẠT |
| ngưỡng hạng 1 = q·1/m | 0.05/8 = 0.00625 | ĐẠT |
| p nhỏ nhất quan sát | siglip2B16 meanpool, p=0.0928 | > 0.00625, không qua hạng 1 → đúng 0 tháp sống sót |
| p one-sided/two-sided? | **hai phía**: `asl()` dòng 85-91, `2*min(P(d<=0),P(d>=0))` | khai báo rõ trong docstring dòng 26-30 |
| B, seed, cụm theo gì | 5000, seed=0, cụm theo **nguồn** (`groups` từ `fold_id`) | đúng đơn vị độc lập, không cụm theo clip |
| mâu thuẫn CI/p ở siglip2B16? | margin +3.14 đến +4.00pp, CI chứa 0 ở phía dưới ([-0.57,+9.99] theo strategy-doc §8.1), p=0.093 | **không mâu thuẫn** — CI rộng chứa 0 và p>0.05 là cùng một câu chuyện, không phải hai câu chuyện trái nhau |

Không tìm được lỗi số học. **ĐẠT.**

---

## 4. Ngân sách chạm tập test — mâu thuẫn nội tại trong văn bản đã đóng băng

### 4.1 Luồng xác nhận "không chạm test" cho vòng 2/3 tháp
`grep` toàn bộ script vòng 2/3 cho đường dẫn tới tập test đóng băng: **0 kết quả** (đã làm ở phiên
trước, xác nhận lại bằng việc mọi script đọc ở trên chỉ mở `*_oof.json`). Đúng như Chương 4 §4.9
dòng 1273-1274 định nghĩa: "phân tích chỉ đọc trên dự đoán ngoài-fold đã lưu... không tiêu một lần
chạm". Phần này **ĐẠT**, không có chạm giấu.

### 4.2 Nhưng con số 0/2 trong văn bản đã đóng băng — tự mâu thuẫn, bằng chứng trực tiếp

| Vị trí (trong 11 tệp nộp) | Câu | Có gắn mốc thời gian? |
|---|---|---|
| `chuong-04-chi-tiet-thuc-nghiem.txt:1260-1261` | "Số lần đã tiêu **0**" (bảng) | KHÔNG |
| `chuong-04-chi-tiet-thuc-nghiem.txt:1276-1279` | "Hệ quả quan trọng của việc **cả hai lần chạm vẫn còn nguyên**: mọi con số trong Chương 6 là con số ngoài-fold... hai lần chạm tiền đăng ký vẫn khả dụng" | KHÔNG — viết thời hiện tại, không ghi "tại thời điểm viết" |
| `chuong-06-danh-gia.txt:2271-2272` (mục 6.14, G20) | "...một lần chạm tập kiểm tra được tiền đăng ký... **hiện còn nguyên hai lần chưa dùng**" | KHÔNG |
| `chuong-06-danh-gia.txt:2284-2286,2291` (mục 6.15, thêm 11/10) | "Mục này được thêm sau mốc đóng băng theo ngoại lệ ghi nhận ngày 11/10/2026... quyết định của tác giả ngày 11/10/2026" | CÓ |
| `chuong-06-danh-gia.txt:2510-2511` | "...**không tiêu thêm lần chạm test nào**: không huấn luyện lại... chỉ đọc lại `Pt` của **lần chạm đã tiêu ở G25**" | xác nhận 1 ĐÃ tiêu |

Ba dòng đầu (Chương 4 §4.9 và Chương 6 §6.14) nói **0/2 đã tiêu, cả hai còn nguyên**, không gắn
mốc thời gian. Hai dòng cuối (Chương 6 §6.15, cùng tệp, cùng chương, được thêm sau theo đúng lời
tự thuật của chính văn bản) nói **1/2 đã tiêu** cho G21+G25, ngày 11/10. §6.14 đứng NGAY TRƯỚC
§6.15 trong cùng tệp mà vẫn giữ câu "còn nguyên hai lần" — đây không phải một snapshot lịch sử bị
bỏ quên ở xa, mà là hai mục liền kề của CÙNG MỘT CHƯƠNG tự nói hai điều khác nhau.

So sánh với Chương 3 §3.8.3 (đã đọc ở phiên trước): câu tương đương ở đó có gắn mốc
("Trạng thái tại thời điểm viết: 0 trong 2 lần chạm đã tiêu") — Chương 3 làm đúng, Chương 4 và
mục 6.14 không làm theo cùng quy ước đó.

**Đây là lỗi biên tập thật, có thể kiểm chứng bằng số dòng, không phải suy diễn của tôi.** Mức độ:
TRUNG BÌNH — không làm sai một con số kết quả nào (ngân sách thật đúng là 1 đã tiêu/1 còn lại, số
này đúng và khớp `reports/test-touch-log.md`), nhưng làm một phát biểu về **tính toàn vẹn phương
pháp** (Chương 4 dùng câu này để bảo chứng rằng "mọi con số Chương 6 là OOF, chưa ai chạm test")
trở thành phát biểu sai nếu đọc nguyên văn, không gắn mốc, vào ngày bảo vệ 31/10/2026.

**Khuyến nghị:** tác giả cần sửa dòng 1261 và 1276-1279 thành có mốc thời gian ("tại thời điểm
đóng băng 05/10") hoặc cập nhật số thành 1/2, giống cách Chương 6 §6.15 đã làm đúng.

Verdict câu 4: **ĐẠT CÓ ĐIỀU KIỆN** — thực chất (không chạm giấu) đúng, hình thức (văn bản tự
mâu thuẫn) sai.

---

## 5. ICC theo nguồn 0.43-0.44 và 0/151 nguồn hai nhãn — hệ quả nặng nhất của buổi phản biện này

### 5.1 Số đã đo, đã tự kiểm lại
`plans/reports/analysis-2026-10-20-icc-theo-nguon-va-so-nguon-can-them.md`:
- ICC(LMM, REML) B1-B2 = 0.4289 [0.2564,0.5484]; B1-ESTA = 0.4418 [0.2436,0.5688]. 3 bộ ước lượng
  (LMM/ANOVA/Bayes-VB) khác nhau ≤0.03 — ổn định.
- Cỡ nguồn cần cho +5pp: 458-468 (gấp ~3.1x so với 151 hiện có). Vượt MỌI ngăn tiền đăng ký trong
  `researcher-2026-10-19-source-ceiling-and-corpora.md` kể cả ngăn mở ">0.20 → 400+".
- 0/151 nguồn mang cả hai nhãn — tự kiểm lại độc lập từ `fold_id` thô (không chỉ tin lời văn bản):
  khớp.

### 5.2 Luận văn đã nói gì
`00-tom-tat.txt:213` (theo trích dẫn trong tài liệu ICC) và nhiều nơi khác: "0/151 nguồn mang cả
hai nhãn, nên nhãn và định danh nguồn bị trộn lẫn **về mặt cấu trúc**; hệ quả là CI theo cụm rất
rộng". Đây là hệ quả về **độ rộng khoảng tin cậy / công suất thống kê** — luận văn đã tự nhận.

### 5.3 Hệ quả nặng hơn mà luận văn CHƯA tự nói — câu hỏi giá trị cấu trúc (construct validity)
Nếu mỗi nguồn (kênh/video gốc) chỉ mang một nhãn duy nhất, thì dưới leave-source-out, một mô hình
**không bao giờ phải phân biệt hành vi an toàn/không an toàn trong cùng một nguồn** — nó chỉ cần
học "nguồn này thuộc nhóm nào", và nhãn-nguồn là một ánh xạ hầu như tất định (ICC cao chính là
thước đo của việc đó). Hệ quả không chỉ là "CI rộng" (vấn đề ước lượng) mà là một câu hỏi về
**việc 84.2% đo cái gì**: có thể một phần lớn của độ chính xác đến từ nhận diện kênh/phong cách
quay/ngữ cảnh sản xuất của nguồn (ánh sáng, góc camera, biên tập) thay vì nhận diện **hành vi**
diễn ra trong khung hình — điều mà bài toán tuyên bố giải quyết.

Đây không phải một lỗi có thể sửa bằng thêm nguồn hay thêm fold — mục 6, 7, 8 của tài liệu ICC đã
chỉ đúng điều này ("thêm nguồn giúp ít hơn mức từng chiếu"), nhưng tài liệu đó dừng ở câu hỏi công
suất, **không đặt câu hỏi cấu trúc này**. Câu hỏi 5 của tài liệu ICC ("y_true hằng theo nguồn là
tạo tác dựng corpus hay tính chất thật của dữ liệu — CHƯA làm") chính là cửa vào của vấn đề giá trị
cấu trúc, nhưng chưa ai đi qua cửa đó.

**Phải nói công bằng cả hai chiều, vì đây là phát hiện nặng nhất của buổi phản biện:**
- Chiều bất lợi cho luận văn: nếu đúng là nhãn tương đương định danh nguồn, thì G14 (ESTA không
  hơn B1w) có thể đang so hai cách đọc CÙNG MỘT manh mối-nguồn theo hai kiến trúc khác nhau, không
  so hai giả thuyết về hành vi — làm yếu thêm độ tin của việc dùng G14 làm "cổng trung tâm".
- Chiều công bằng cho luận văn: (a) luận văn **đã tự nêu** hệ quả bề mặt của chính cấu trúc này
  (không giấu); (b) luận văn **không dùng** 84.2% để tuyên bố "mô hình học hành vi tổng quát" một
  cách vô điều kiện — Chương 6/7 đã giới hạn claim vào đúng corpus, đúng giao thức, không ngoại
  suy; (c) đây là hạn chế của **bộ dữ liệu iSafetyBench** (cách thu thập theo nguồn/kênh), không
  phải lỗi thiết kế phương pháp của tác giả — và luận văn đã gọi đúng tên "hạn chế dữ liệu, không
  phải hạn chế phương pháp" ở `chuong-07-phan-tich-that-bai.txt` §7.9.2(a).

Verdict: **ĐẠT CÓ ĐIỀU KIỆN** — số liệu đúng, disclosure hệ quả bề mặt đúng, nhưng chưa đặt câu hỏi
giá trị cấu trúc ở mức sâu nhất. Đề nghị hội đồng hỏi trực tiếp câu này tại buổi bảo vệ; không nên
tự kết luận hộ tác giả vì cần tài liệu gốc của iSafetyBench (cách lấy mẫu theo nguồn) mà phiên này
chưa tra được.

---

## 6. κ=0.65 "trần nhiễu" — có lẫn làn không

| Vị trí | Nội dung | Có lẫn làn? |
|---|---|---|
| `chuong-04-chi-tiet-thuc-nghiem.txt:202-222` | κ=0.65 đo trên 100 clip, taxonomy 4 lớp+nền của Unsafe-Net; dòng 220-222 tự nói **"không suy kappa của làn này sang làn kia"** | KHÔNG |
| `chuong-06-danh-gia.txt:1624,1649` | κ=0.775 (mô hình-vs-nhãn, Unsafe-Net) tách rõ khỏi κ=0.65 (giữa người, Unsafe-Net) | KHÔNG |
| `chuong-07-phan-tich-that-bai.txt:979` | "κ=0.65. Một chênh lệch nhỏ hơn mức nhiễu nhãn này không thể giải thích bằng mô hình" — ngay sau §7.9.2(a) về làn C | KHÔNG (giới hạn làn C) |
| `chuong-01-mo-dau.txt:110-111,425` | nêu κ=0.65 ở chương mở đầu, không gắn làn cụ thể trong câu nhưng cùng mạch 100-clip Unsafe-Net | KHÔNG (ngữ cảnh rõ) |
| `chuong-04-chi-tiet-thuc-nghiem.txt:208` | **"kappa = 0.65 là trần trên của MỌI con số trong luận văn"** | RỦI RO — câu quá rộng |

Rà toàn bộ 11 tệp: không tìm thấy một lần nào κ=0.65 bị áp trực tiếp vào một con số của làn A
(iSafetyBench/G14/84.2%). Về **thực chất**, câu hỏi 6 **ĐẠT**.

Nhưng dòng 208 viết câu tổng quát hoá quá tay ("trần của MỌI con số"), rồi dòng 220-222 mới thu
hẹp lại đúng phạm vi. Một hội đồng đọc nhanh, trích câu (1) mà bỏ câu (3), có thể hỏi sai "vậy G14
có nằm trong trần 0.65 không" — và câu trả lời đúng là KHÔNG, vì G14 đo trên nhãn nhị phân
iSafetyBench, không có số κ tương ứng được đo. Mức độ: THẤP-TRUNG BÌNH, lỗi hành văn, không phải
lỗi suy luận. Đề nghị tác giả sửa câu (1) thành phạm vi đúng ngay tại dòng 208 trước khi bảo vệ nếu
còn kịp, hoặc chuẩn bị câu trả lời miệng dẫn thẳng dòng 220-222.

---

## 7. Kết quả âm — có công bố đúng chuẩn không

| Kết quả âm | Vị trí | Đủ chi tiết? |
|---|---|---|
| G11 trượt (encoder không tách khỏi đầu dò toàn khung tuyến tính) | `chuong-06-danh-gia.txt:1199-1221`, margin +1.1pp, CI [-4.8,+6.9] chứa 0, McNemar p=0.625 | ĐẦY ĐỦ, và ràng buộc viết (ghi ở `00-trang-thai-bien-soan.txt:168`) buộc báo G11 TRƯỚC E0 — đã tuân thủ đúng thứ tự tại §6.11.1 trước §6.11.2 |
| G14 trượt (cổng trung tâm) | `chuong-06-danh-gia.txt:371-389`, −0.5pp, CI[−5.5,+5.0], p=0.683 | ĐẦY ĐỦ, gọi đúng tên "phán quyết quyết định hình dạng của luận văn" |
| G15, G16, G18 trượt | Bảng 6.2 dòng 445-456 | ĐẦY ĐỦ, số liệu nguyên vẹn, không làm tròn giấu dấu |
| 8 tháp round-3, không tháp nào sống BH | `round3_tower_bh.json`, strategy-doc §8.2 | ĐẦY ĐỦ nhưng chỉ ở `plans/`, không vào `report/vn/` — xem mục 2.3 |
| Zero-shot ranking không chuyển giao, mobileclip_s1 hoà control, R@FAR5 không đo được | strategy-doc §8.3-8.4 | ĐẦY ĐỦ, cùng vị trí plans/ |
| Độ trôi tái lập 1.46pp từ số luồng CPU | `chuong-04-chi-tiet-thuc-nghiem.txt:867-962` | ĐẦY ĐỦ, có cơ chế gốc (dòng mã cụ thể `esta_training.py:116`), có bảng số, có lý do không sửa |

Không tìm thấy trường hợp nào kết quả âm bị cắt, giấu, hoặc đổi thứ tự trình bày để giảm tác động.
**ĐẠT.** Đây là điểm mạnh thật của luận văn — mức độ trung thực báo cáo thuộc nhóm cao nhất trong
các luận văn thạc sĩ cùng dạng mà người viết từng thẩm định.

---

## 8. Tự kiểm một con số headline

Đọc trực tiếp `sweep_results/relation_tokens/enc_isbclip_B1_lso-1100_headline-none_oof.json`,
tính lại bằng công thức đăng ký (độ nhạy trung bình trên các lớp có mặt):
- lớp 0 (an toàn): n=680, recall=0.8382
- lớp 1 (không an toàn): n=420, recall=0.8452
- BA = mean = **84.17367%** → khớp con số công bố 84.2% (làm tròn 1 chữ số).

Đồng thời tự kiểm lại 151 nguồn / 0 nguồn trộn nhãn trực tiếp từ `fold_id` của cùng tệp — khớp
Phụ lục B. **ĐẠT.**

---

## 9. Điểm mạnh cần ghi nhận

- Kỷ luật "ưu tiên CI cụm trên McNemar khi hai thứ bất đồng" (§3.8.1) — đúng nguyên tắc, vì McNemar
  giả định độc lập theo clip, sai khi 151 nguồn không cân bằng (1-102 clip/nguồn).
- Disclosure tự bất lợi liên tục: độ trôi 1.46pp (§4.5.6), fold-size-as-fingerprint (Phụ lục B§B.5),
  3/8 nhóm KMeans suy biến ở làn C (Phụ lục B§B.4), hai lỗi tính scorecard cũ (§6.12.4) — tất cả
  đều TỰ nêu, không phải bị phát hiện từ ngoài.
- `analyze_posthoc_asymmetric_measures.py` và `analyze_posthoc_paired_vs_control.py` gắn nhãn
  `[trong mẫu]`/`WRONG_UNIT` rõ ràng cho mọi số không đạt chuẩn đơn vị độc lập — kỷ luật tốt hiếm
  gặp.
- Quy tắc năm nhãn xuất xứ ([quy ước]/[mặc định thư viện]/[mặc định của mã]/[dò tìm]/[ràng buộc])
  áp dụng nhất quán, giúp phân biệt số đo với số giả định.

---

## Câu hỏi chưa giải quyết

1. Không đọc được trực tiếp mã nguồn hàm sinh fold (`ISB.folds`/`StratifiedGroupKFold` wrapper) —
   thẩm định giao thức LSO hiện dựa trên log + tái lập từ artefact, chưa dựa trên đọc logic chia
   fold trong mã.
2. Không có tạo tác tiền đăng ký độc lập, có dấu thời gian riêng, cho quy tắc kiểm định kép vòng 3
   (margin+CI+BH) — không thể xác nhận bằng git rằng luật được khoá trước khi 8 tháp có kết quả.
   Cần tác giả cung cấp bằng chứng khác (ví dụ email, log chat, bản nháp có dấu thời gian ngoài
   git) nếu muốn khép câu hỏi này.
3. Câu hỏi giá trị cấu trúc ở mục 5.3 (liệu 84.2% đo hành vi hay đo nhận diện nguồn/kênh) cần tra
   tài liệu gốc mô tả cách iSafetyBench được thu thập theo nguồn — chưa ai trong kho này làm việc
   này (xem `analysis-2026-10-20-icc-...md` câu hỏi mở số 5).
4. Chương 4 §4.9 và Chương 6 §6.14 (câu "0/2, cả hai lần chạm còn nguyên") cần được tác giả sửa lại
   có gắn mốc thời gian hoặc cập nhật số liệu, để không mâu thuẫn với Chương 6 §6.15 trong cùng bộ
   tệp nộp — hiện tại đây là một mâu thuẫn có thể trỏ bằng số dòng, chưa được tác giả xử lý.
5. Chưa đọc trực tiếp nội dung chi tiết của `run_ablation_significance.py`'s tương tác với lane C
   paper-baseline (95.81%) theo chiều ngược — chỉ xác nhận lane-separation đúng, chưa đánh giá sâu
   phương pháp bootstrap+McNemar của riêng kịch bản đó (ngoài phạm vi 8 câu hỏi được giao).
