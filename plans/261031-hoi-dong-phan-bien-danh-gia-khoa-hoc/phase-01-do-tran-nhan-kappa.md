# Phase 01 — Đo trần nhãn iSafetyBench nhị phân (κ) · mã việc **R3**

## Liên kết ngữ cảnh
- Tiền đăng ký: [tien-dang-ky-kappa-nhi-phan-isafetybench.md](tien-dang-ky-kappa-nhi-phan-isafetybench.md)
- Sổ tay mã hoá: `annotation_guide/kappa-isb-binary/so-tay-ma-hoa-va-quy-trinh.md`
- Biên bản hội đồng: [reports/bien-ban-tong-hop-hoi-dong-2026-10-31.md](reports/bien-ban-tong-hop-hoi-dong-2026-10-31.md) §2.9
- Chiến lược: [ke-hoach-chien-luoc-sau-phien-31-10.md](ke-hoach-chien-luoc-sau-phien-31-10.md) §4 (T2-1)

## Tổng quan
**Ưu tiên: cao nhất trong các việc có phép đo.** Trạng thái: **SẴN SÀNG GẮN NHÃN**.
Khuyến nghị số một của hội đồng 10/10, sau 3 tuần vẫn 0 bước chuẩn bị; nay công cụ đã xong.
Nó **chặn** mọi quyết định nâng cấp xương sống và là lá chắn cho câu hỏi *"trần nhãn là bao nhiêu?"*

**Đường tới hạn:** gắn nhãn (3 phiên) → κ → viết phụ lục + §6.1.7 → diễn tập lần 1 (22–28/11).
Phải xong gắn nhãn **trước 14/11**.

## Nhận định then chốt
1. **Tổng điều tra 151 nguồn, 1 clip/nguồn** — không phải lấy mẫu. 76 hazard / 75 normal, cân bằng sẵn. Mạnh hơn n = 200 có cặp (ICC 0.43 ⇒ cỡ hiệu dụng ≈ 140).
2. **0/151 nguồn mang cả hai nhãn** ⇒ clip thứ hai cùng nguồn là đáp án cho sẵn. Một clip/nguồn là **bắt buộc**, không phải tuỳ chọn.
3. **Đường dẫn thật chính là nhãn** (`hazard/` vs `normal/`) ⇒ làm mù là yêu cầu kỹ thuật, không phải hình thức.
4. **Script phân tích commit trước khi có dữ liệu** — vá lỗ hổng xuất xứ §2.1 bằng hành động.
5. **Tiêu 0 lần chạm test.** Dự đoán OOF là kết quả đã công bố.
6. **D1 có thể bất lợi**: nếu κ ≥ 0.85 và người hơn mô hình ≥ 5 pp, lập luận "sát trần nhãn" **phải rút**. Đã chốt trước.

## Tệp liên quan
**Đã tạo:** `tools/sample-kappa-clips.py` · `tools/kappa-annotation-server.py` ·
`tools/analyse-kappa.py` · `annotation_guide/kappa-isb-binary/so-tay-ma-hoa-va-quy-trinh.md` ·
tiền đăng ký · `.gitignore` (loại `key.json` theo tên)
**Sẽ sinh:** `annotation_guide/kappa-isb-binary/annotations.jsonl` (commit theo phiên)
**Sẽ sửa sau khi có kết quả:** `report/vn/00-trang-thai-bien-soan.txt` (ngoại lệ bốn) · phụ lục mới ·
một dòng dẫn chiếu ở `chuong-06-danh-gia.txt`

## Nhiệm vụ

### N0 — trước khi gắn nhãn (≈ 1 giờ, làm một lần)
- [ ] **Đọc danh sách lớp bài báo iSafetyBench** (arXiv 2508.00399). *Mối đe doạ hiệu lực lớn nhất: sổ tay lệch định nghĩa ⇒ κ đo sai lệch định nghĩa, không đo nhiễu nhãn.* Tinh chỉnh sổ tay nếu cần, ghi thành **phụ lục có ngày**.
- [ ] Đọc hết sổ tay mã hoá, **đặc biệt R1–R10 và R4** (thiếu PPE một mình ⇒ bình thường — luật này sẽ bị bản năng chống lại).
- [ ] Quyết câu hỏi §9.4 của tiền đăng ký: **nếu D1 xảy ra, có đủ 60 ngày để viết lại §6 không?** Trả lời trước, không phải sau.
- [ ] (Không bắt buộc, nhưng chặn được thiên lệch của chính tác giả) rủ **người gán nhãn thứ hai** làm 60–80 clip. Nếu có → đăng ký bổ sung trước khi họ bắt đầu.
- [ ] **Hiệu chuẩn:** gắn nhãn thử 10 clip Unsafe-Net ở `data/train/` bằng R1–R10. Không sinh số báo cáo.

### N1 — gắn nhãn (3 phiên × ~60 sự kiện, 2–3 giờ tổng)
- [ ] Phiên 1: `.venv/bin/python tools/kappa-annotation-server.py` → `http://127.0.0.1:8765` → 60 sự kiện → **commit `annotations.jsonl`**
- [ ] Phiên 2 (nghỉ ≥ 30 phút): 60 sự kiện → **commit**
- [ ] Phiên 3: 56 sự kiện còn lại (gồm 25 lần hiện lại) → **commit**
- [ ] Trong mọi phiên: bật `r` **bất cứ khi nào** nhận ra clip. *Bật thiếu làm nghiên cứu sai một cách thầm lặng.*

### N2 — phân tích (15 phút)
- [ ] `.venv/bin/python tools/analyse-kappa.py` — chạy **một lần**, lưu nguyên đầu ra
- [ ] Đọc **κ nội-người trước tiên**: nếu < 0.80 thì theo **D3**, toàn bộ phép đo không diễn giải được, dừng tại đây và báo cáo đúng như vậy
- [ ] Áp **D1–D5** đúng như đã chốt. Không thêm phép phân tích nào không có trong §4 tiền đăng ký

### N3 — đưa vào luận văn (0.5 ngày)
- [ ] Khai **ngoại lệ bốn** ở `00-trang-thai-bien-soan.txt`: dẫn chiếu tiền đăng ký, **tiêu 0 lần chạm test**, **bảng cổng không đổi — hai đạt, sáu trượt, tám cổng**
- [ ] Viết **phụ lục mới có ngày** (ít xâm phạm nhất) + **một dòng dẫn chiếu** ở §6
- [ ] Viết `§6.1.7` (ba làn A/B/C + lệnh cấm so làn) — nay có κ để chống đỡ câu hỏi trần nhãn
- [ ] Cập nhật câu trả lời chuẩn bị số 5 trong bảng diễn tập: thay "chưa đo" bằng số đo

## Tiêu chí hoàn thành
1. `annotations.jsonl` có **176 sự kiện**, commit qua **≥ 3 commit** có dấu thời gian khác nhau.
2. `analyse-kappa.py` chạy sạch, in κ + AC1 + BA người + BA mô hình + CI + McNemar + κ nội-người + các nhóm cắt.
3. Một trong D1–D5 được nêu **tên mã** trong văn bản luận văn.
4. Ngoại lệ bốn đã khai, có câu "bảng cổng không đổi".
5. Không phát ngôn nào trong danh sách cấm ở §5 tiền đăng ký xuất hiện ở bất kỳ đâu.

## Đánh giá rủi ro
| Rủi ro | Mức | Giảm nhẹ |
|---|---|---|
| Sổ tay lệch định nghĩa của iSafetyBench ⇒ κ vô nghĩa | **CAO** | N0 bước 1 (đọc danh sách lớp); R1–R10 chốt trước; đã nêu công khai ở §5 tiền đăng ký |
| Tác giả nhớ nhãn clip (đã sống với kho nhiều tháng) | **CAO** | cờ `r`, cắt nhóm, **D4**; không phủ nhận mà **đo** nó |
| Tác giả có lợi ích trong việc κ thấp (κ thấp ⇒ "sát trần nhãn") | **CAO** | luật diễn giải chốt trước, **gồm D1 bất lợi**; mời người gán nhãn thứ hai; sổ tay là luật cơ học không phải phán xét tổng thể |
| D1 xảy ra, còn 60 ngày phải viết lại §6 | TRUNG BÌNH | trả lời §9.4 **trước** khi gắn nhãn |
| Mỏi khi gắn nhãn ⇒ κ nội-người thấp ⇒ D3 | TRUNG BÌNH | tối đa 60 sự kiện/phiên, nghỉ ≥ 30 phút, dừng lúc nào cũng được |
| Rò rỉ nhãn qua đường dẫn | ĐÃ VÁ | máy chủ chỉ gửi K-id; traceback bị chặn; `key.json` bị gitignore; **K096 loại trước** vì đã rò ở phiên kiểm thử 01/11 |
| n = 151 < 200 mà hội đồng 10/10 nêu | THẤP | 151 là trần thật dưới ràng buộc độc lập; đã đo: mạnh hơn 200 có cặp |

## Bước tiếp theo
Phụ thuộc: **N0 chặn N1**. N2 chặn N3. N3 chặn việc cập nhật bảng diễn tập (T1-6 lần 1, 22–28/11).
Nếu không có người gán nhãn thứ hai: vẫn chạy, nhưng ghi hạn chế vào phụ lục.
