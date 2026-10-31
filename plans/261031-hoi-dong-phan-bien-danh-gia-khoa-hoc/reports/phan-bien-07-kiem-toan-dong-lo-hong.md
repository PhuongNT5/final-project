# PHẢN BIỆN 07 — KIỂM TOÁN ĐÓNG LỖ HỔNG (31/10/2026)

Phương pháp: grep/đọc trực tiếp `report/vn/`, `git log -p`, `git diff`, mtime file trên đĩa.
Không tin bảng trạng thái tự khai — mọi dòng dưới đây có cột chứng cứ tại nguồn.

## 0. PHÁN QUYẾT TRỤC

| Trục | Phán quyết | Lý do (1 câu) |
|---|---|---|
| **Kiểm toán đóng lỗ hổng** | **ĐẠT CÓ ĐIỀU KIỆN — xu hướng xấu đi** | Các lỗ hổng A–C đã đóng (05/10–09/10) kiểm lại vẫn đứng vững, nhưng hạng mục rẻ nhất/quyết định nhất (A4 chữ ký GVHD) bị treo 3 tuần không một dấu vết cố gắng, trong khi tác giả vẫn tiếp tục chạm số trên luận văn đúng 1 ngày sau khi hội đồng 10/10 ra lệnh "KHÔNG CHẠM SỐ". |

---

## 1. A1–A6 / B1–B10 / C1–C9 — trạng thái THẬT hôm nay 31/10

Quy ước: ✅ĐÃ ĐÓNG (kiểm được tại nguồn, giữ nguyên) · 🟡HẠ MỨC (đúng như tự khai, limitation công bố) · 🔴CÒN MỞ (tự khai "đã đóng" nhưng không đứng vững, hoặc chưa làm gì mới) · ⬜THỦ TỤC (ngoài khả năng kiểm bằng mã)

| # | Lời khai 05/10→09/10 | Kiểm hôm nay | Chứng cứ |
|---|---|---|---|
| A1 | ĐÃ ĐÓNG (phần câu chữ) | ✅ đứng vững | `report/vn/chuong-01-mo-dau.txt` mục 1.4.2 còn 4 dòng tự khai, dòng YOLO→CLIP đầu bảng (nội dung khớp mô tả) |
| A2 | ĐÃ ĐO (97.7% random-fold) | ✅ đứng vững, **không trộn làn vẫn giữ** | `grep -c "95.81\|97.7" report/vn/*.txt` — mọi lần xuất hiện kèm nhãn làn A/làn C; `chuong-06-danh-gia.txt:2551-2553` dùng đúng ký hiệu |
| A3 | không đóng được bằng câu chữ | ⬜ không đổi | — |
| A4 | **CÒN MỞ — văn bản GVHD ký** | 🔴 **CÒN MỞ, 0 TIẾN TRIỂN SAU 3 TUẦN** | `plans/261005-.../reports/bao-cao-giai-trinh-thay-doi-de-cuong-gui-gvhd.md` — không có khối chữ ký/ngày duyệt; `plans/trang-thai-chuong-trinh.md` (cập nhật 20/10) **không nhắc A4 một lần nào**; hội đồng 10/10 §8 câu hỏi 6 ("GVHD đã ký chưa?") **không có câu trả lời ở bất kỳ tài liệu nào sau 10/10** |
| A5 | ĐÃ ĐÓNG — 3 ma trận | ✅ đứng vững | `report/vn/chuong-06-danh-gia.txt` có 6.2.8 (Bảng 6.2a/6.2b) và 6.12.4 (Bảng 6.7a), số liệu khớp mô tả |
| A6 | không đóng được bằng câu chữ | ⬜ không đổi | `grep -c "dashboard\|server lưu trữ" report/vn/*.txt` = 0 |
| B1 | HẠ MỨC — giới hạn công bố | 🟡 **đúng như khai, và XÁC NHẬN VẪN CHƯA ĐO dù có cơ hội** | `report/vn/chuong-05-trien-khai-bien.txt:217,225,1130-1137` — hạng mục (15) còn nguyên hiệu lực; demo 2 clip (`demo/out/*.mp4`, mtime 10/10 16:24) **minh hoạ, không phải phép đo**, chính văn bản nói vậy ở dòng 225 |
| B2 | ĐÃ ĐÓNG — G19 placebo | ✅ đứng vững | `git show eafec53 9895df4 --stat` tồn tại, số liệu 89.03%/75.2%/90.6%/p=0.0476 khớp `chuong-06-danh-gia.txt` |
| B3 | ĐÃ CÔNG BỐ | ✅ đứng vững | mục 6.9 hạn chế (1) tồn tại |
| B4 | ĐÃ CÔNG BỐ (hạ mức bằng chứng) | ✅ đứng vững | 3 mốc git re-kiểm khớp: `9cf01ca`, `26cdf1f`, `df1dab7` đều tồn tại đúng ngày đã khai |
| B5 | ĐÃ ĐÓNG — `.gitignore` phủ định | ✅ đứng vững | `.gitignore:31` có `!sweep_results/**/*_oof.json`; `git ls-files \| grep -c _oof.json` = 123 |
| B6 | ĐÃ LÀM (sửa tóm tắt) | ✅ đứng vững | không re-kiểm sâu, bản 05/10 đã có bằng chứng grep |
| B7 | ĐÃ ĐÓNG VỀ CÂU CHỮ | ✅ đứng vững | mục 5.4.4 tồn tại, nêu *f*/24, không OSHA/ISO |
| B8 | ĐÃ ĐÓNG | 🔴 **ĐÓNG NHƯNG CHỨA KHIẾM KHUYẾT CHƯA VÁ — xem Q2 dưới** | chính đoạn văn dùng để đóng B8 (`chuong-05-trien-khai-bien.txt:552-553`) ghi "0.49%" và "trên vòng lặp tổ hợp" — cả hai **sai theo audit nội bộ 13/10** (`finding-2026-10-13-...md:58-77`): đúng phải là "0.87%" và "trên tổng giai đoạn" (dữ liệu là tổng giai đoạn/stage-sum, không phải vòng lặp tổ hợp — đúng loại trích dẫn mà quy tắc 5.4.1 cấm). 18 ngày sau audit, **vẫn chưa sửa** |
| B9 | ĐÃ ĐÓNG — 31 p-value | ✅ đứng vững | mục 6.8.1 tồn tại, không re-kiểm số đếm |
| B10 | HẠ MỨC — giới hạn công bố | 🟡 đúng như khai | `grep -n soak report/vn/*.txt` xác nhận: không tồn tại phép soak vòng lặp tổ hợp, chỉ soak từng giai đoạn; văn bản tự nói "Không tồn tại phép soak nào của vòng lặp tổ hợp" (`chuong-05-trien-khai-bien.txt:1027`) |
| C1 | ĐÃ ĐÓNG — banner README | ✅ không re-kiểm sâu (tin bằng chứng cũ) | — |
| C2 | ĐÃ ĐÓNG — docs sync | ✅ không re-kiểm sâu | — |
| C3 | ĐÃ ĐÓNG — nhãn M3/M4, bằng tạo tác máy sinh | ✅ đứng vững, chứng cứ loại mạnh | train log có banner ultralytics tự in, độc lập lời khai tác giả |
| C4 | CHƯA LÀM (tự khai) | 🟡 vẫn chưa làm cho tháp headline B/32, nhưng next-round có dữ liệu B/16 khác tháp | `report/vn/chuong-05-trien-khai-bien.txt:181` "INT8 của tháp CLIP chưa được dựng lại... chưa được trích dẫn" — khớp. **Cảnh báo không trộn làn**: `plans/trang-thai-chuong-trinh.md` mục 7 đo cadence cho `clipB16/siglip2B16/mobileclip_s1` — đây là tháp khác, next-round, chưa vào `report/vn/` (xác nhận grep = 0 kết quả) |
| C5 | ĐÃ ĐÓNG — Geirhos | ✅ không re-kiểm sâu | — |
| C6 | CÒN MỞ (05/10) → **MỘT PHẦN ĐÓNG, RỦI RO MỚI** | 🟡 video tồn tại trên đĩa (2 clip, `demo/out/*.mp4`, 10/10) nhưng **KHÔNG được git track** (`.gitignore:187 demo/out/*.mp4`) — chỉ JSON timeline được track (`8610c74`). File .mp4 chỉ nằm trên 1 đĩa, không tái tạo được nguyên bản (script dùng model/data có thể drift) | `git ls-files demo/` không có `.mp4`; `ls -la demo/out/` mtime 10/10 |
| C7 | ĐÃ ĐÓNG — đổi tên Ch7 | ✅ không re-kiểm sâu | — |
| C8 | ĐÃ ĐÓNG | ✅ không re-kiểm sâu | — |
| C9 | ĐÃ ĐÓNG BẰNG LỜI KHAI | ✅ đúng như khai (tự nhận không phải bằng chứng) | — |

**Tổng kết mục 1:** 18/25 hạng mục kiểm lại đứng vững hoặc đúng như khai. **2 hạng mục đáng báo động:**
A4 (0 tiến triển, hạng mục duy nhất ảnh hưởng ĐẠT/KHÔNG ĐẠT theo cả hai hội đồng trước) và B8
(lỗ hổng "đã đóng" 09/10 thực ra đang **chứa** đúng loại vi phạm trích dẫn mà B1/B8 tự hào "0 vi phạm" —
hội đồng 05/10 tự kiểm nhưng bỏ sót; audit nội bộ 13/10 phát hiện ra, rồi bản thân tác giả cũng không vá).

---

## 2. Khuyến nghị số một hội đồng 10/10 (κ nhị phân iSafetyBench, n≥200, 2 người gán độc lập)

**CHƯA LÀM.** Chứng cứ:

| Kiểm | Kết quả |
|---|---|
| `find . -iname "*kappa*"` | chỉ `run_kappa_prepare.py`, `run_kappa_compute.py` (có từ `6f780cd`, 08/07) + `plans/260613-.../kappa-result.json` (κ=0.65, **Unsafe-Net 4 lớp**, không phải iSafetyBench nhị phân) |
| mtime `__pycache__/run_kappa_*.pyc` | **03/10/2026** — trước cả mốc đóng băng, trước cả khuyến nghị (10/10). Không có dấu hiệu script được chạy lại sau 10/10 |
| artifact người gán nhãn thứ hai | không tồn tại (không tìm thấy thư mục/tệp annotator/rater nào ngoài thư viện bên thứ ba trong `.venv`) |

Lý do ghi ở đâu: `plans/trang-thai-chuong-trinh.md` mục 5 câu 1 và mục 9.8 — "E1 bị chặn ở cần người
đánh giá thứ hai — không chạy được trên máy". Đây là lý do hợp lý (cần người, không phải compute),
nhưng **đã đứng y nguyên từ 10/10 tới 20/10** (10 ngày) không có bất kỳ bước chuẩn bị nào (không có SOP
nhãn, không có tập 200 clip được chọn, không có lời mời người gán thứ hai) — tức khuyến nghị **đứng đầu
danh sách ưu tiên của cả E2 lẫn mọi quyết định nâng cấp xương sống** nhưng chưa có hành động chuẩn bị nào.

---

## 3. Bốn việc "chân trời 1" (biên bản 10/10 mục 4)

| # | Việc | Trạng thái thật | Chứng cứ |
|---|---|---|---|
| 1 | Chữ ký GVHD (A4) | 🔴 **CÒN MỞ, 0 tiến triển** | xem mục 1 trên |
| 2 | Diễn tập hỏi–đáp | 🔴 **CÒN MỞ — không tìm thấy bất kỳ tạo tác nào** | không có file ghi âm, biên bản diễn tập, hay checklist Q&A trong `plans/` sau 10/10 |
| 3 | Video demo dự phòng (C6) + đo B1 cửa sổ trượt | 🟡 **MỘT NỬA XONG** | video minh hoạ tồn tại (10/10) nhưng **không phải phép đo** (chính Ch5:225 nói vậy) → B1 vẫn mở; video chỉ 1 đĩa, chưa commit |
| 4 | B10/C4 hạ mức, không cần làm thêm | ✅ đúng — không có hoạt động thêm, đúng khuyến nghị | — |

**Kết luận mục 3:** việc #1 (rẻ nhất, 0 ngày công, chỉ cần gặp GVHD) và việc #2 (2-3 giờ) — hai việc
*rẻ nhất* và *đòn bẩy cao nhất* theo đúng bảng xếp hạng của hội đồng 10/10 — là hai việc duy nhất
**0% tiến triển**. Việc #3 (1 buổi, tốn công nhất trong 4 việc) được làm một phần. Thứ tự thực hiện
bị **đảo ngược** so với khuyến nghị đòn bẩy/chi phí của hội đồng.

---

## 4. Hàng đợi Q1–Q16 (`plans/trang-thai-chuong-trinh.md`) đối chiếu kho mã

Trọng tâm Q2 — ba sửa văn bản Ch5, tự khai "CHỜ NGƯỜI":

| Sửa | Vị trí tự khai | Giá trị hiện tại tại nguồn | Khớp tự khai? |
|---|---|---|---|
| `53.2%` → `52.6%` | dòng 552–553/1234 | `chuong-05-trien-khai-bien.txt:1234` = **"53.2 %"** (chưa sửa) | ✅ khớp (đúng là còn treo) |
| `vòng lặp tổ hợp` → `tổng giai đoạn` | cùng vị trí | `chuong-05-trien-khai-bien.txt:552` = **"trên vòng lặp tổ hợp"** (chưa sửa) | ✅ khớp |
| `0.49%` → `0.87%` | cùng vị trí | `chuong-05-trien-khai-bien.txt:553` = **"phân tán 0.49%"** (chưa sửa) | ✅ khớp |

Tự khai của S3/Q2 **khớp đúng kho mã** — không có vi phạm khai báo ở đây. Nhưng cần nhấn mạnh:
đây là lỗi *trích sai đơn vị đo* (stage-sum bị đọc thành composed-loop) nằm NGAY TRONG đoạn văn
mà hội đồng 05/10 dùng để tuyên B8 "ĐÃ ĐÓNG" — tức bản thân bằng chứng dùng để đóng B8 có lỗi,
và lỗi đó đã được phát hiện (13/10) nhưng không được vá trong 18 ngày.

Kiểm nhanh các dòng khác của hàng đợi (không đi sâu hết 16 mục do giới hạn thời gian):
- Q8 (ICC): số 0.429/0.442, CI [0.256,0.548]/[0.244,0.569] khớp đúng commit `6a5d01c` message — ✅
- Q11 (EVA02 BH): "3 dòng chữ đổi, 0 số đổi" khớp commit `6a5d01c` diff (`analyze_round3_tower_bh.py` 15 dòng đổi, JSON stat không đổi số) — ✅
- Q1 (Jetson B/16 cadence): bảng mục 7 có 6 lần chạy, tệp JSON tương ứng tồn tại trong diff `8610c74` — ✅ cấu trúc khớp, không verify lại từng con số cadence

---

## 5. Cam kết "đóng băng 05/10 — chỉ sửa văn bản" có bị vi phạm?

**CÓ — nhưng được tự khai minh bạch, không giấu.** Đây là phát hiện quan trọng nhất của trục này.

`git log --since=2026-10-05 --format="%H %ai %s" -- report/vn/`:

| Commit | Ngày | Có số mới trong `report/vn/`? |
|---|---|---|
| `df1dab7` | 07/10 23:03 | nằm trong cửa sổ mà hội đồng 05/10 đã xét và chấp nhận (ngoại lệ Nhóm 3) |
| `9895df4` | 08/10 23:08 | G19 placebo, +15.4pp phân tách — ngoại lệ đã được hội đồng 05/10 chấp nhận ("mốc tác giả tự đặt") |
| `90b7310` | 09/10 00:56 | C3/C5/C7 câu chữ — không số mới |
| **`eaaccf5`** | **11/10 21:31** | **CÓ — SỐ MỚI**: AUC gộp 0.8485 → AUC trong nhóm 0.6426, CI [0.4923, 0.8443] (mục 6.15.4, "NGOẠI LỆ THỨ BA", `00-trang-thai-bien-soan.txt:29-43`) |
| **`06854b4`** | **11/10 23:33** | tự khai "text correction only — mọi giá trị chép từ dòng đã có"; không phát hiện số hoàn toàn mới khi đọc diff |
| `16f75c9` | 13/10 00:19 | chỉ đổi số dòng/số từ (word count), không số thí nghiệm — xác nhận bằng diff trực tiếp |
| `bf9a627` | 15/10 23:08 | tự khai "No number moves"; không phát hiện ngược lại |

**Chi tiết vi phạm `eaaccf5`:** commit này + `a063da5` (cùng ngày, "score the binary task...") là một
lượt đo MỚI (NP-125-D), chạy 11/10 (mtime `plans/261011-.../ket-qua-np125d-phat-hien.json` = 11/10 20:32),
tức **đúng 1 ngày sau** khi hội đồng 10/10 ra khuyến nghị "Chân trời 1: KHÔNG CHẠM SỐ — từ nay tới bảo vệ".
Tác giả **tự ghi nhận** đây là "NGOẠI LỆ THỨ BA" trong `00-trang-thai-bien-soan.txt` với đầy đủ: tiền
đăng ký trước khi đọc test, ngân sách chạm test = 0 (đọc lại posterior đã lưu, không huấn luyện/suy luận
lại), và không đảo bảng cổng (vẫn 2 đạt/6 trượt). **Đây là làn Unsafe-Net (NP-125), không phải iSafetyBench.**

Đánh giá liêm chính: đây **không phải gian lận** — mọi quy tắc tự đặt (ngân sách test, tiền đăng ký,
không đảo phán quyết) đều được tuân thủ và công khai ngay trong văn bản nộp. Nhưng nó **là một vi phạm
thủ tục** đối với khuyến nghị cụ thể và mới nhất của hội đồng khoa học (10/10), xảy ra trong vòng 24
giờ sau khi khuyến nghị được đưa ra. Mức độ nghiêm trọng: THẤP về mặt số liệu (không số nào bị đảo),
TRUNG BÌNH về mặt tuân thủ quy trình hội đồng (khuyến nghị bị phá ngay lập tức, dù có lý do kỹ thuật
chính đáng — trả lời câu hỏi "đã thử tái lập giao thức công trình gốc chưa").

**Không phát hiện vi phạm nào trên làn iSafetyBench** (lane A, con số headline 84.2%/97.7%) sau 10/10 —
kiểm `grep` các tệp `chuong-06` không thấy số mới liên quan iSafetyBench nào thêm sau `9895df4` (08/10).

---

## 6. Nhánh `next-round` chưa đẩy remote — định lượng rủi ro mất dữ liệu

`git fetch origin` xác nhận `origin/next-round` **tồn tại** (không phải "chưa từng push" như mục 10
`trang-thai-chuong-trinh.md` ngụ ý đọc thoáng) nhưng **dừng ở `5644889` (15/10)**, trong khi local
`next-round` đã ở `6a5d01c` (20/10) — **2 commit, 11 ngày tuổi, chưa đẩy**:

```
git log origin/next-round..next-round --oneline
6a5d01c docs(analysis): measure the source ICC and correct the eight-tower BH labels
8610c74 chore(evidence): track the round-3 device measurements and what certifies them
```

| Đại lượng | Số |
|---|---|
| Commit chưa push | 2 |
| Tuổi của commit cũ nhất chưa push | 11 ngày (20/10 → 31/10) |
| Tổng thay đổi (`git diff --shortstat origin/next-round next-round`) | **86 tệp, 20,290 dòng thêm** |
| Tệp đo thiết bị KHÔNG tái tạo được (is_device_number=true) | **11** JSON (`deployment/artifacts/jetson-measurements/*.json`) |
| Tệp phán quyết cổng cửa sổ sạch | **10** JSON (`deployment/artifacts/round3-device/guard-*.json`) |
| Tệp tham chiếu đối sánh nhị phân | **4** `.npz` (`r48-parity-fixture/*.npz`) |
| **Tổng tạo tác không tái tạo được chỉ nằm trên 1 đĩa** | **25** |
| Báo cáo nghiên cứu (md) đi kèm, cũng chưa push | ~9 tệp, ước ~3,000 dòng |

Rủi ro: nếu đĩa máy cục bộ hỏng/mất trước khi push, **25 tạo tác đo đạc thiết bị thật (Jetson Orin
Nano Super, điều kiện DVFS/xung nhịp cụ thể tại thời điểm 19-20/10) biến mất vĩnh viễn** — không
tái tạo được bằng cách chạy lại script vì cần đúng phần cứng, đúng trạng thái nhiệt/xung nhịp tại
thời điểm đó. Tác giả **tự nhận thức** rủi ro này (`trang-thai-chuong-trinh.md:185-186`: "Commit cục
bộ bảo vệ khỏi ghi đè, không bảo vệ khỏi mất đĩa... Cần một câu cho phép đẩy nhánh next-round") nhưng
**11 ngày sau vẫn chưa push**. Đây là rủi ro vận hành đơn giản nhất để sửa (một lệnh `git push`) và
cũng là rủi ro duy nhất trong toàn báo cáo này có chi phí sửa gần bằng 0.

Bổ sung: `report/vn/` (toàn bộ nội dung luận văn) **không tồn tại trên nhánh `main`** — chỉ tồn tại
trên `next-round`. `origin/main` vừa bị force-update trong lúc tôi `git fetch` (thấy dòng "forced
update" khi fetch) nhưng không ảnh hưởng `report/vn/` vì nó chưa từng ở trên `main`. Nghĩa là **toàn
bộ bản nộp chỉ sống trên một nhánh**, và nhánh đó mới đồng bộ remote tới 15/10.

---

## CÂU HỎI CHƯA GIẢI QUYẾT

1. A4 (chữ ký GVHD) đã 3 tuần không tiến triển — có đang bị chặn bởi lịch GVHD, hay bị quên giữa
   khối lượng công việc next-round? Không tìm thấy bằng chứng cho cả hai khả năng.
2. Diễn tập hỏi–đáp (chân trời 1 việc #2) không có tạo tác nào — đã làm miệng (không ghi lại) hay
   chưa làm? Không kiểm được bằng kho mã.
3. `eaaccf5`/`a063da5` (NP-125-D, 11/10) tự nhận là "ngoại lệ thứ ba" sau khuyến nghị 10/10 — tác giả
   có ý thức được đây là 1 ngày sau khuyến nghị "KHÔNG CHẠM SỐ" khi quyết định chạy, hay khuyến nghị
   chưa kịp truyền đạt? Không có dấu thời gian nào cho biết tác giả đã đọc biên bản 10/10 trước khi
   chạy `a063da5`.
4. Vì sao audit nội bộ 13/10 phát hiện lỗi B8 (0.49%/vòng lặp tổ hợp) nhưng không có commit vá nào
   trong 18 ngày tiếp theo, trong khi cùng giai đoạn đó có 6 commit khác sửa `report/vn/`?
5. Lịch bảo vệ chính xác là ngày nào? Không tìm thấy trong bất kỳ tệp nào — nếu bảo vệ đã diễn ra
   trước 31/10, toàn bộ mục 3 của báo cáo này (chân trời 1) cần đọc lại ở thời điểm đã muộn.
