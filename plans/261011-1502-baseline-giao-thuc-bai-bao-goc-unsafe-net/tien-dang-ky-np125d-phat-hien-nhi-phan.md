# Tiền đăng ký NP-125-D — chắt lọc lại phép đo nhị phân theo **mục tiêu phát hiện**

| | |
|---|---|
| Ngày viết | 2026-10-11, **trước** khi tính bất kỳ số nào dưới đây |
| Nền | commit 4598bc5; posterior đã lưu `sweep_results/r2/temporal_convlstm_binary.pt` |
| Làn | **C duy nhất** |
| **Lần chạm test mới** | **0** — không huấn luyện lại, không suy luận lại. Đọc lại *đúng* `Pt` của lần chạm đã tiêu ở G25. |
| Quan hệ với G25 | **tái diễn đạt, không phán quyết.** Không được đảo, nới hay làm mềm phán quyết G25. |

## 1. Vì sao cần bản này

G25 đo **BA** — đại lượng **đối xứng**, coi một lần bỏ sót clip Không an toàn ngang giá một lần
báo động giả trên clip An toàn. Mục tiêu đã nêu lại của đề tài là **phát hiện được cái không an
toàn**. Hai cái không cùng một hàm mất mát: với một hệ cảnh báo, bỏ sót là tai nạn, báo động giả
là chi phí vận hành. BA không in ra cái giá đó. Bản này in.

**Lớp dương = Không an toàn (KAT).** Mọi đại lượng dưới đây định nghĩa theo lớp dương đó.

## 2. Nguy cơ của chính việc này, nêu trước

Đọc lại cùng một `Pt` ở nhiều ngưỡng **là** một dạng dò tìm trên test. Chặn bằng đúng một luật:
**mọi ngưỡng phải được chốt trên val 103 clip, trước khi đọc test ở ngưỡng đó.** Không ngưỡng nào
chọn trên test, kể cả "để minh hoạ".

## 3. Bốn điểm hoạt động, tất cả chốt trên val

| Ký hiệu | Luật chọn ngưỡng (chỉ dùng val) | Vai trò |
|---|---|---|
| **OP0** | τ = 0.5 (argmax) | mặc định, đã báo ở G25 |
| **OP1** | τ cực đại hoá **F1(KAT) trên val** | điểm cân bằng |
| **OP2** | τ nhỏ nhất sao cho **FPR trên val ≤ 0.10** | ngân sách báo động giả vận hành |
| **OP3** | τ sao cho **FPR trên val gần nhất FPR của scene-only trên val** | so sánh **đồng giá báo động** |

Đại lượng chính: **recall(KAT) tại OP2** kèm FPR thực đo. Phụ: F1(KAT) tại OP1; PR-AUC(KAT) không
phụ thuộc ngưỡng. Toàn bộ đường ROC/PR **chỉ mô tả**, không được làm số tiêu đề.

## 4. Luật quyết định — bản tái diễn đạt "có nội dung" khi nào

OP3 là phép so sánh duy nhất có hiệu lực, vì scene-only chỉ có một điểm hoạt động; so recall ở
FPR khác nhau là so hai thứ khác giá.

> **Có nội dung** nếu tại OP3: recall(KAT) vượt recall(KAT) của scene-only **≥ 5 pp** **VÀ**
> McNemar chính xác trên *quyết định KAT* cho **p < 0.05**.
> Ngược lại: kết luận **y nguyên G25** — tác vụ bị bối cảnh chi phối, bản phát hiện cũng vậy.

Không đạt thì không đổi luật. Kết quả bất lợi là kết cục khả dĩ và được chấp nhận.

## 5. Sàn và trần, suy ra từ nhãn chứ không đặt ra

`duplicate_audit_binary.json` có **4** cặp trùng byte dính tới clip An toàn trong test:

- 1 cặp **nằm trọn trong test**: `0_te23.mp4` (KAT) ≡ `4_te9.mp4` (AT);
- 3 cặp **bắc qua train→test**, bản KAT ở train, bản AT ở test: `4_te4`, `5_te13`, `4_te7`.

Hệ quả phải in kèm mọi số FPR: bộ phát hiện tất định cho **cùng** một quyết định trên đầu vào
trùng byte ⇒ **recall 100% và FPR 0% không thể đồng thời đạt được**. Sàn FPR do nhãn ấn định được
báo riêng, và phải kiểm xem bốn clip ấy có thực sự nằm trong số báo động giả hay không.

## 6. Đơn vị thống kê

CI bootstrap cụm 5,000 mẫu, **đơn vị = nhóm bối cảnh** (8 nhóm phủ 125 clip test). Rủi ro số 1 của
kế hoạch đã nêu: n = 8 ⇒ CI rất rộng, **rất có thể chứa 0**. CI mức clip in kèm nhãn `WRONG UNIT`
và không được dùng để kết luận.

## 7. Thứ bản này không được làm

- Không đổi số tiêu đề nào của bản đóng băng, không đổi bảng cổng (vẫn **tám cổng, hai đạt, sáu trượt**).
- Không gộp NP-125 với NP-64. Trên NP-64 (toàn KAT) **chỉ** recall xác định; FPR/precision/F1/PR-AUC **không** xác định, không in.
- Không so với 95.81% của [17]: số đó là macro 4 lớp, không phải phát hiện nhị phân.
