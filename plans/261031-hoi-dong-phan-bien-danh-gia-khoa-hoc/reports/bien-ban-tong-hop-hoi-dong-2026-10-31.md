# BIÊN BẢN TỔNG HỢP — HỘI ĐỒNG PHẢN BIỆN & ĐÁNH GIÁ KHOA HỌC

Ngày: **31/10/2026** · Nhánh: `next-round` · HEAD cục bộ: `6a5d01c`
Đối tượng: `report/vn/` (11 tệp nộp) + kho bằng chứng `plans/`, `deployment/`, `sweep_results/`
Thành phần: 6 phản biện độc lập (PB-07…PB-12) + chủ tịch tổng hợp kèm **9 phép tự kiểm chứng**
Phiên trước: 05/10 (phản biện luận văn) · 10/10 (đánh giá khoa học)

---

## 1. PHÁN QUYẾT

**ĐẠT CÓ ĐIỀU KIỆN** — giữ nguyên mức của phiên 05/10. Nhưng **ràng buộc quyết định đã dịch
chuyển**: không còn nằm ở khoa học, mà ở **thủ tục và xuất xứ**.

| Trục | PB | Phán quyết | Một câu lý do |
|---|---|---|---|
| Kiểm toán đóng lỗ hổng | 07 | ĐẠT CÓ ĐIỀU KIỆN — **xu hướng xấu** | 18/25 hạng mục đứng vững, nhưng hạng mục rẻ nhất và quyết định nhất (A4) treo 3 tuần, 0 tiến triển |
| Phương pháp & thống kê | 08 | ĐẠT CÓ ĐIỀU KIỆN | khung thống kê đúng kỹ thuật, hiếm ở bậc thạc sĩ; 2 lỗ hổng về **thời điểm đăng ký** và 1 mâu thuẫn nội văn |
| Tính mới & công bố | 09 | ĐẠT CÓ ĐIỀU KIỆN (nếu giữ reframe) | đóng góp đứng được là định lượng lạm phát giao thức + hệ biên đo thật; **không** có venue gần hạn |
| Triển khai biên & tái lập | 10 | ĐẠT ở P1 | truy xuất được tới từng chữ số; 1 mâu thuẫn Ch5 tự hạ điểm mình, 1 khe hở khoá phiên bản |
| Văn bản & truy xuất tuyên bố | 11 | **ĐẠT** — rủi ro thấp | 0 rò rỉ số đã rút, 0 vi phạm định dạng, 12/12 hình, tóm tắt dẫn mạch đúng |
| Diễn tập phản biện | 12 | — | xác suất đạt 55–65 % → **78–85 %** (chỉ câu chữ + diễn tập) → 85–90 % (thêm 1 phép đo) |

### 1.1. Phát biểu trung tâm của phiên này

> **Bản nộp đang ở trạng thái tốt hơn vùng xung quanh nó.** Rà soát toàn bộ 11 tệp `report/vn/`
> không tìm thấy rò rỉ số đã rút, không vi phạm định dạng, không hình mồ côi, tóm tắt mở đầu
> bằng lạm phát giao thức chứ không bằng 84.2 %. **Mọi phát hiện nặng của phiên này nằm NGOÀI
> bản nộp**: ở `README.md`, ở bảng theo dõi, ở dấu thời gian xuất xứ, ở thủ tục hành chính.

Hệ quả chiến lược: **ngừng mua độ chính xác, chuyển toàn bộ nguồn lực còn lại sang xuất xứ,
thủ tục và diễn tập.** Căn cứ ở §4.

---

## 2. PHÁT HIỆN MỚI CỦA PHIÊN — xếp theo sức sát thương

### 2.1. [CHỦ TỊCH + PB-08] Tiền đăng ký tám cổng **không có dấu thời gian độc lập** — NẶNG NHẤT

Tài sản mạnh nhất của luận văn là "cổng tiền đăng ký, báo cáo kết quả âm trung thực". Kho mã
**không chứng minh được** phần "tiền".

| Tạo tác | Nội dung | Vào git | Kết quả nó phán xử |
|---|---|---|---|
| `plans/2026-09-19-.../risks.md` | hiến chương định nghĩa G14–G17 | **01/10** | G14 dated **29/09** |
| `tools/adjudicate-r48-gates.py` | script ngưỡng hằng số | **29/09** | cùng ngày phán quyết |
| `plans/2026-09-19-.../plan.md` | câu khẳng định "cổng viết trước khi có code, không đàm phán lại sau khi thấy kết quả" | **07/10** | 8 ngày sau |
| `plans/2026-09-19-.../number-sheet.md` | sổ số + kết quả | **01/10** | chứa sẵn phán quyết |

**Không phải lỗi vệ sinh kho:** `plans/` đã được track từ **08/07/2026**. Việc track sớm là
khả thi và đã xảy ra với tệp khác.

PB-08 tìm thấy **cùng một mẫu, lặp lại ở vòng 3**: `chien-truoc-thuc-nghiem-vong-3.md` chỉ có
**một commit duy nhất (20/10)** chứa **đồng thời** luật đăng ký (biên ≥ 1.46 pp + CI loại 0 +
BH q=0.05, m=8) **và** bảng kết quả — trong khi dự đoán OOF của cả 8 tháp đã có trên đĩa từ
**14–16/10** (mtime). Đối chiếu: NP-125-D **đã làm đúng** (tệp tiền đăng ký riêng, commit riêng).

⇒ Hai lần ⇒ đây là **mẫu hệ thống về xuất xứ**, không phải một sơ suất.

**Giảm nhẹ, và giảm nhẹ mạnh:** sáu trong tám cổng được công bố là **TRƯỢT**, gồm chính giả
thuyết trung tâm; vòng 3 công bố **0/8 tháp sống sót**; NP-125-D công bố AUC **tụt** 0.8485 →
0.6426 với CI chứa 0.50 và nhãn `WRONG_UNIT` tự đặt. **Không ai làm giả hiến chương để tự cho
mình trượt.** Bằng chứng hành vi rất mạnh; bằng chứng tài liệu thì không có.

**Phán quyết hội đồng:** không phải vấn đề liêm chính. Là **lỗ hổng xuất xứ không thể vá bằng
cách chạy lại**, chỉ vá được bằng **một lời khai đặt đúng chỗ** (§4, T1-5).

### 2.2. [CHỦ TỊCH] `README.md` tự mâu thuẫn với chính lệnh rút số của nó — và đã lừa được hội đồng

| Dòng | Nội dung |
|---|---|
| `README.md:20` | tuyên bố **rút** `331 ms / 3.02 dec/s / 3.91 J` và cấm trích |
| `README.md:43-44` | **dùng** `3.02-3.04 dec/s ... 2.60-2.72` làm kết quả tiêu đề |

Cách ba dòng. Chủ tịch đã **chép chính con số đã rút ấy vào đề bài giao cho PB-10**; PB-10 bắt
lỗi và phân xử lại tại nguồn. Số đúng, hiện hành, theo `chuong-05-...-ghim-xung-nhip.txt:284-287`
(§5.2.2 — chính luận văn rút nó, vì đo dưới trạng thái xung nhịp không khai báo):

| Đại lượng | Ghim xung nhịp | DVFS mặc định |
|---|---|---|
| P3, tải cắt trung vị, vòng lặp tổ hợp | **3.226 qđ/giây (+7.5 %)** | **2.733 qđ/giây (−8.9 %)** |
| P3, bảng ba đoạn (5.4b) | ĐẠT **2/3** | **TRƯỢT 3/3** (1.141 / 1.441 / 1.511) |

Hai nhánh **rời nhau** (max DVFS 1.511 < min ghim 2.493). Con số 3.02 **không thuộc điều kiện
nào**. Luận văn tự ghi: không tuyên bố P3 là hệ thời gian thực; đường đã triển khai là **P1**.

⇒ `README.md` là **tạo tác nguy hiểm nhất trong kho**: nó phục hồi số đã rút, và nó là tệp đầu
tiên mọi người đọc. Nếu nó lừa được chủ tịch với kho mở trước mặt, nó sẽ lừa được uỷ viên hội đồng.

### 2.3. [CHỦ TỊCH + PB-07] Toàn bộ luận văn sống trên **một đĩa, một nhánh**

| Kiểm | Kết quả |
|---|---|
| `report/vn/` trên `origin/main` | **0 tệp** — luận văn không tồn tại trên main |
| `origin/next-round` | dừng ở `5644889` (**15/10**) |
| local `next-round` | `6a5d01c` (**20/10**) |
| Delta chưa đẩy | **86 tệp, 20,290 dòng thêm**, 2 commit |
| Trong đó không tái tạo được | **11 JSON đo Jetson** (205 KB) + 10 phán quyết cổng cửa sổ sạch + 4 `.npz` đối sánh |

Các JSON ấy cần **đúng phần cứng + đúng trạng thái nhiệt của 19–20/10**. Chạy lại script không
dựng lại được chúng. Chi phí sửa: **một lệnh `git push`**. Rủi ro hiện tại: mất toàn bộ bằng
chứng thiết bị + toàn bộ luận văn nếu mất đĩa.

### 2.4. [PB-08] Hiệu lực cấu trúc (construct validity) — phát hiện khoa học sâu nhất, **chưa có trong luận văn**

Dữ kiện đã đo: **0/151 nguồn mang cả hai nhãn.** Mỗi nguồn thuần một nhãn.

Luận văn đã nêu **hệ quả bề mặt** (CI theo cụm rộng, ICC 0.43–0.44, cần 458–468 nguồn cho +5 pp).
Luận văn **chưa đặt câu hỏi sâu hơn**: khi nhãn và danh tính nguồn **đồng biến hoàn hảo**, thì
84.2 % có thể một phần đến từ việc nhận ra **kênh/phong cách sản xuất** của nguồn, không phải
nhận ra **hành vi**. Đây là câu hỏi hiệu lực cấu trúc, không phải câu hỏi công suất thống kê.

Hai chiều, phải nói cả hai:
- **Chiều bất lợi:** không thể tách "hành vi" khỏi "thuộc tính nguồn" **ở bên trong kho này** — giới hạn này không vá được bằng thêm fold hay thêm mô hình.
- **Chiều có lợi:** chính vì đồng biến hoàn hảo mà **leave-source-out là giao thức đúng và khó nhất có thể**; fold ngẫu nhiên bỏ qua nguồn sẽ thổi số lên, và luận văn đã **đo** đúng mức thổi đó (+13.5 pp).

⇒ Viết thành **một đoạn giới hạn**, không cần phép đo mới. Nói trước khi bị hỏi thì nó là chiều
sâu; bị hỏi trước khi nói thì nó là lỗ hổng.

### 2.5. [PB-08] Mâu thuẫn nội văn trong văn bản **đã đóng băng** — ngân sách chạm test

| Vị trí | Phát biểu |
|---|---|
| `chuong-04-chi-tiet-thuc-nghiem.txt:1261` | "Số lần đã tiêu **0**" |
| `chuong-04-...:1276-1279` | "**cả hai lần chạm vẫn còn nguyên**" |
| `chuong-06-danh-gia.txt:2271-2272` | "hiện còn nguyên **hai lần** chưa dùng" |
| `chuong-06-danh-gia.txt:2284-2286`, §6.15 | ngoại lệ 11/10: **đã tiêu một lần gộp** cho G21+G25 |

Thực chất **đúng** (ngân sách thật: 1 tiêu / 1 còn, khớp `test-touch-log.md`, không có lần chạm
giấu). Hình thức **sai**: hai chỗ nói "cả hai còn nguyên" không gắn mốc thời gian. Vá bằng **một
cụm trạng ngữ** ("tại thời điểm viết mục này, trước ngoại lệ 11/10").

### 2.6. [PB-10] Ch5 §5.9(12) **tự hạ điểm mình bằng một câu đã lỗi thời**

§5.4.4/§5.10 nói 6 tạo tác "đã đồng bộ về kho mã"; §5.9 hạng mục (12) nói "chưa đồng bộ" và tự
gọi đó là **"hạn chế nghiêm trọng nhất"**. PB-10 xác minh tại đĩa: **đã track từ `df1dab7`**.
⇒ Luận văn đang tự khai một hạn chế nghiêm trọng **không còn tồn tại**. Xoá/sửa 1 câu.

### 2.7. [PB-10] Khe hở tái lập thật: **không có tệp khoá phiên bản**

Không `requirements.txt`/lock nào ở gốc kho. Không ai chứng minh được phiên bản
`transformers` / `open_clip_torch` / TensorRT-python lúc đo 19–20/10 khớp hiện tại. Ba ONNX tháp
vòng 3 không vào git (vượt hạn 100 MB, kho không dùng LFS); `export-round3-towers-onnx.py` dựng
lại được **engine tương đương trong sai số cổng**, không phải **engine bit-for-bit đã đo**, vì
trọng số tải qua HF Hub **không pin revision**. Nhãn `git_commit: "6319b21-dirty"` trong JSON đo
không lưu diff ⇒ khe hở duy nhất của phép truy xuất.

**Phép truy xuất tái lập của PB-10 (thành công):** `167.291 ms = 5.978 qđ/giây` → JSON
`r48-composed-cadence-fp16-20261003T155843.json` (khớp **từng chữ số**) → `device_state`
(quiet/clocks_pinned khớp mô tả chương) → `git_commit` → commit tồn tại. Chuỗi kín trừ "-dirty".

### 2.8. [PB-07] A4 — văn bản GVHD duyệt đổi hướng: **0 tiến triển sau 3 tuần**

Hạng mục **duy nhất** mà cả hai hội đồng trước xác định có thể lật **ĐẠT/KHÔNG ĐẠT**. Bảng theo
dõi cập nhật 20/10 **không nhắc A4 một lần nào**. Không tệp nào có chữ ký/dấu duyệt. Chi phí:
**0 ngày công, chỉ cần một buổi gặp.**

### 2.9. [PB-07] Khuyến nghị số một của phiên 10/10 — **chưa làm, 0 bước chuẩn bị**

Đo κ trên **chính nhãn nhị phân iSafetyBench**, n ≥ 200, hai người gán nhãn độc lập.
`__pycache__/run_kappa_*.pyc` mtime **03/10** — trước cả khuyến nghị. Không tạo tác người gán
nhãn thứ hai. κ = 0.65 hiện có đo trên **Unsafe-Net 4 lớp + nền, n = 100** — khác làn.
⇒ Lập luận phòng thủ mạnh nhất ("chúng tôi đã sát trần nhãn") vẫn là **phỏng đoán**, và nó
**chặn mọi quyết định nâng cấp xương sống**.

### 2.10. [CHỦ TỊCH] Bất biến "đóng băng 05/10" **sai như đang viết**

`plans/trang-thai-chuong-trinh.md` §4.1: "`report/vn/` đóng băng 05/10: **chỉ sửa văn bản**".
Lịch sử git nói khác:

| Commit | Ngày | Churn trên `report/vn/` | Loại |
|---|---|---|---|
| `df1dab7` | 07/10 | +12,892 (vào git lần đầu) | — |
| `9895df4` | 08/10 | +4,029 / −88 | **có phép đo mới** (lạm phát giao thức 97.7 % / +13.5 pp / p = 1.39e-41) |
| `90b7310` | 09/10 | +3,292 / −330 | đóng lỗ hổng |
| `eaaccf5` | 11/10 | +1,493 / −135 | **ngoại lệ 2 & 3** |
| `06854b4` · `16f75c9` · `bf9a627` | 11–15/10 | +231 · +169 · +1,373 | câu chữ |

Mốc thật: **số = 09/10 (có ngoại lệ tới 11/10), văn bản = 15/10.**
**Quan trọng:** cả ba ngoại lệ **đã được khai minh bạch** trong
`00-trang-thai-bien-soan.txt:14-50`, kèm tiền đăng ký riêng, kèm hạch toán ngân sách chạm test,
kèm câu "bảng cổng vẫn hai đạt sáu trượt". Đây là **lời khai đầy đủ**, không phải vi phạm che giấu.
PB-07 xếp việc 11/10 là "vi phạm thủ tục" — hội đồng **điều chỉnh**: là **ngoại lệ đã khai, hạch
toán đúng**. Rủi ro duy nhất là **câu chữ của bất biến §4.1** mời một câu hỏi không cần thiết.

---

## 3. ĐIỂM MẠNH — hội đồng ghi nhận, có kiểm chứng

1. **Bản nộp sạch.** 0 rò rỉ số đã rút trên 11 tệp (PB-11 grep toàn bộ sổ số đã rút: 95.7 % mAP, P3 ĐẠT/TRƯỢT, 89.15/90.92 val — mọi lần xuất hiện đều có nhãn). 0 vi phạm định dạng. 12/12 hình tồn tại, 0 mồ côi.
2. **Tóm tắt dẫn mạch đúng thứ tự** — mở bằng lạm phát giao thức, không mở bằng 84.2 %.
3. **Số tiêu đề tái lập 1:1.** Chủ tịch + PB-08 độc lập: BA = 84.17 % khớp 84.2 % công bố; 151 nguồn và 0 nguồn trộn nhãn khớp Phụ lục B; số học BH đúng (p_min 0.093 > 0.00625, 0/8 sống sót, không mâu thuẫn CI↔p).
4. **Tám cổng nhất quán** xuyên tóm tắt / `00-trang-thai-bien-soan` / Ch1: "hai đạt, sáu trượt", không trôi.
5. **Kết quả âm công bố đầy đủ, không cắt xén**: G11, G14–G18, 8 tháp vòng 3, thứ hạng zero-shot không chuyển (DataComp-1B +1.05 / DFN-2B +0.43 kém CLIP thường +2.58), VideoMAE kém tháp ảnh 2.37 pp, `mobileclip_s1` tệ hơn ở **cả hai** trục.
6. **Tự sửa chính mình nhiều lần, có ghi sổ**: Q11 (EVA02 — kiểm lại thì khuyết điểm nhẹ hơn tự ghi, BH tái lập **từng byte**), Q13, Q16, ba sai số của S1, bốn luồng E1–E4 bị sửa đề.
7. **Q14 là phát hiện thật.** PB-10 xác minh **chuỗi lập luận KÍN**: loại trừ xuất ONNX (đúng, rel 1.22e-05), nạp dữ liệu (checksum lệch **đúng 0**), hiệu chuẩn/kênh (bit-identical), **và** giả thuyết cạnh tranh "do precision-scale" (tự đề ra, tự bác bằng build FP32 đối chứng). ⇒ TensorRT 10.3.0 chuyển đổi sai đồ thị MobileCLIP-S1, sai **bất biến** qua 2 độ chính xác × 4 cỡ lô. Đủ sức **báo lỗi upstream**.

---

## 4. CĂN CỨ CHO KHUYẾN NGHỊ TRUNG TÂM: **NGỪNG MUA ĐỘ CHÍNH XÁC**

Bốn phép đo độc lập, không phép nào do phiên này sinh ra, cùng chỉ một hướng:

| Đường nâng cấp | Kết quả đo | Nguồn |
|---|---|---|
| Ngưỡng / hiệu chuẩn | oracle +0.12 pp; chọn trung thực **−0.47 pp**; lỗi là tự-tin-sai (3/175 trong băng do dự) | biên bản 10/10 §1 |
| Ensemble cùng họ | +0.33 pp; 87/1100 clip sai ở **cả 5** mô hình | biên bản 10/10 §1 |
| Đổi **đầu** phân loại | meanpool 84.89 ≥ transformer 83.25; B1xl 696K tham số = **82.7 %** (xấu hơn) | vòng 1–2, §6.2.6 |
| Đổi **tháp** (9 tháp, vòng 3) | cao nhất siglip2B16 86.39 % (+3.14 pp) nhưng **0/8 sống sót BH**, p_min 0.093 | `round3_tower_bh.json` |
| Thêm **nguồn** dữ liệu | ICC 0.43–0.44 ⇒ cần **458–468 nguồn** cho +5 pp = **3.0×** số hiện có; S2: trần 151 nguồn là **trần thật** | Q8, S2 |

⇒ Mọi đường rẻ đã cạn; đường duy nhất còn hiệu lực (thêm nguồn) cần **3× kho dữ liệu** và kho đó
**không tồn tại**. **Chính thức đóng hướng "nâng 84.2 %".** Mọi ngày công bỏ thêm vào đó là lãng phí
đã được chứng minh bằng số, không phải bằng phỏng đoán.

---

## 5. ĐIỀU CHỈNH CỦA CHỦ TỊCH ĐỐI VỚI PHẢN BIỆN

Hội đồng không lấy nguyên kết luận của phản biện. Bốn điều chỉnh:

| PB | Điều chỉnh | Căn cứ |
|---|---|---|
| 09 | **Venue sai.** "NeurIPS Negative Results Track, hạn 14/11/2026" **không tồn tại**. Negative results là **contribution type** trong main track; hạn NeurIPS 2026 là **06/05/2026**, phán quyết đã ra **24/09/2026**. Đã đóng. | kiểm tại neurips.cc/Conferences/2026/CallForPapers |
| 09 | **Hạ mức "phát hiện mới"** của lạm phát giao thức. Lạm phát do rò rỉ/giao thức là hiện tượng đã biết từ thập niên 1990. Cái mới là **độ lớn đo được trên tác vụ này** (+13.5 pp, CI [+8.0, +20.0]) và **hệ quả xoá khả phân biệt kiến trúc**. Gọi là **định lượng mạnh**, không gọi là phát hiện — S4 xếp loại "mở rộng/định lượng" là đúng hơn. | tránh overclaim tại buổi bảo vệ |
| 11 | **Đọc sai chiều Q4/Q6.** PB-11 kết luận §4.5.7 và §6.1.7 "không tồn tại → không cần sửa". Chủ tịch kiểm: Ch4 dừng ở **4.5.6**, Ch6 có 6.1.5–**6.1.6**, không có 6.1.7. Hai mục ấy là **việc phải VIẾT (Q4, Q6), đang dở**, không phải việc đã moot. | `chuong-04:867`, `chuong-06:160,187` |
| 07 | **Hạ mức "vi phạm không chạm số" 11/10** xuống **ngoại lệ đã khai, hạch toán đúng** | `00-trang-thai-bien-soan.txt:14-50` — xem §2.10 |

Và **một điều chỉnh đối với chính chủ tịch**: đề bài giao cho PB-10 chứa con số **đã bị luận văn
rút** (`3.02-3.04 / 2.60-2.72`), chép từ `README.md:43-44`. PB-10 bắt đúng. Số đúng ở §2.2.
Một chi tiết PB-10 gộp hơi chặt: `3.226/2.733` là **tải cắt trung vị trên vòng lặp tổ hợp**, còn
`1.141–1.511` là **bảng ba đoạn** — hai đại lượng khác nhau, phải trích kèm nhãn, không thay nhau.

---

## 6. CÂU HỎI CHƯA GIẢI QUYẾT

1. **Ngày bảo vệ là bao giờ?** Không có ở bảng theo dõi, `00-trang-thai-bien-soan.txt`, hay `PROGRESS.md`. Đây là biến quyết định nhánh chiến lược — hội đồng không trả lời thay được.
2. A4 bị chặn bởi lịch GVHD, hay bị quên?
3. Có **bên thứ ba** nào xác nhận được thời điểm chốt 8 cổng và luật vòng 3 không (email, tin nhắn GVHD, bản nháp có dấu thời gian ngoài git)? Nếu có, nó đóng §2.1 ngay.
4. iSafetyBench thu thập theo nguồn như thế nào — cần tài liệu gốc để khép câu hỏi hiệu lực cấu trúc §2.4. (S2: cách chia **không nêu** trong nguồn công khai.)
5. Nội dung "-dirty" lúc đo, và phiên bản thư viện chính xác lúc đo (không có lock file).
6. §5.9(12) là sơ suất tồn đọng hay cố ý giữ?
7. Được phép mở `plans/` trước hội đồng, hay chỉ trả lời bằng lời? (quyết định cách dùng số sau đóng băng)
8. Diễn tập hỏi–đáp đã làm miệng không ghi lại, hay chưa làm?
9. Chuyển đổi sai của TensorRT (Q14) có trung tính về chi phí không — câu hỏi do chính luận văn nêu, cần thiết bị.
