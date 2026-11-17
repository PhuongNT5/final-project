# Phán quyết luật D1–D5 — nghiên cứu κ nhãn nhị phân iSafetyBench

**Ngày:** 17/11/2026 · **Dữ liệu:** commit `0b1252e` · **Khai báo sai lệch:** `8c1de30` (đẩy
remote **trước** khi chạy) · **Kết quả:** `ket-qua-kappa-lan-chay-duy-nhat.txt` · **Lần chạy:** 1/1

## Số đo (n = 150, loại K096 theo tiền-đăng-ký)

| Đại lượng | Giá trị | CI95 |
|---|---|---|
| Cohen κ (người ↔ nhãn gốc) | **0,6933** | [0,5745, 0,8001] |
| Gwet AC1 | 0,6950 | [0,5760, 0,8046] |
| BA của **NGƯỜI** | **84,67 %** | [78,79, 90,12] |
| BA của **MÔ HÌNH** trên đúng 150 clip đó | **85,33 %** | [79,36, 90,70] |
| Hiệu NGƯỜI − MÔ HÌNH | **−0,67 pp** | [−8,51, +7,06] |
| McNemar chính xác | **p = 1,0** | lệch 18 vs 19 |
| κ nội-người (25 clip lặp) | 0,8397 | 2/25 clip tự đổi ý |

Cắt nhóm đã tiền-đăng-ký: κ | tự tin cao (n=114) = **0,8418** · κ | tự tin thấp (n=36) = **0,1486**
· κ | không nhận ra (n=149) = 0,6910 · nhóm "nhận ra" n=1 < 10, **không báo cáo**.

## Phán quyết

| Mã | Điều kiện | Nổ? |
|---|---|---|
| **D1** | κ ≥ 0,85 **và** H−M ≥ +5 pp, CI loại 0 | **KHÔNG** — κ = 0,69 < 0,85; H−M âm |
| **D2** | κ ≤ 0,75 **hoặc** CI của H−M chứa 0 | **NỔ — cả hai nhánh** |
| **D3** | κ nội-người < 0,80 | KHÔNG — 0,8397 ≥ 0,80 |
| **D4** | tỉ lệ "nhận ra" ≥ 50 % | KHÔNG — 0,6 % |

**Kết luận buộc phải viết, theo D2 nguyên văn:** *"84.2 % sát trần nhãn" **được chống đỡ**, phát
biểu kèm số đo. **Không** được nói "ngang bằng con người".*

Lập luận "sát trần nhãn" ở §6 **không phải rút**. Đây là lần đầu nó có số đo thay vì lời khẳng định.

## Gấp gáp hay nhập nhằng thật? — Sửa đổi 2 buộc khai cả hai, nay phân xử được

Chẩn đoán **hậu kiểm** (nhãn rõ, không thuộc tiền-đăng-ký): tỉ lệ gắn đúng **giảm đơn điệu** theo
thời gian xem — 94,4 % (<3 s) · 96,6 % (3–6 s) · 84,7 % (6–12 s) · **67,7 % (≥12 s)**. Nhóm tự tin
thấp có dwell **cao hơn** (11,1 s vs 6,9 s).

Nếu gấp gáp là nguyên nhân, sai phải dồn vào clip xem **nhanh**. Sai dồn vào clip xem **lâu**.
⇒ κ = 0,69 phản ánh **nhập nhằng thật của clip**, không phải tác giả làm vội. Cách giải thích
"8,9 s/sự kiện là quá nhanh" **bị dữ liệu phản bác**, không phải bị bỏ qua.

**Phạm vi của kết luận này, nói hẹp cho đúng:** nó chỉ cho thấy **trong phiên này**, thêm thời gian
**không** làm tăng trùng khớp. Nó **không** chứng minh một chuyên gia an toàn lao động, hay nhiều
người đọc độc lập, sẽ không làm tốt hơn. Nhóm <3 s chỉ n = 18.

## Hai điều cắt ngược, phải khai

1. **Lệch hệ thống, gọi hazard ít hơn gốc 11 clip** (bỏ sót 17, báo động giả 6; nhãn độc lập
   64 hazard / 86 normal so với gốc 75/75). Đây là **mối đe doạ hiệu lực số 1 ở §5 đã thành hiện
   thực một phần**: R4 ("thiếu PPE đơn thuần = bình thường") nghiêm hơn định nghĩa của
   iSafetyBench. ⇒ **một phần** của κ = 0,69 là **lệch định nghĩa**, không phải nhiễu nhãn. BA
   84,67 % thì không bị méo vì nó bình quân recall hai lớp, nhưng **κ thì bị kéo xuống**.
2. **κ nội-người 0,8397 là chặn trên, không phải ước lượng.** Hai lần gắn cách nhau 8–26 phút
   trong cùng một lần ngồi. D3 không nổ, nhưng **cấm** viết "độ tin cậy đã được xác nhận".

## Phát hiện mới dùng được cho §6

Sai số dư **tập trung và tự nhận diện được**: κ = 0,8418 trên 114 clip tự tin cao, κ = 0,1486 trên
36 clip tự tin thấp; riêng mức tự tin 2 chỉ đúng 58,8 % (n=34, gần mức ngẫu nhiên của bài nhị phân).
Nghĩa là phần nhãn không đáng tin **nằm gọn** trong một nhóm nhỏ mà chính người đọc chỉ ra được
trước khi biết đáp án — cơ chế đúng của trần nhãn.

## Phát ngôn vẫn bị cấm

"mô hình ngang con người" · "trần nhãn là X %" như chặn cứng · đặt κ = 0,6933 cạnh **κ = 0,65 cũ**
(Unsafe-Net 4 lớp, **làn khác**) · "κ nội-người xác nhận độ tin cậy" · "hai người gán nhãn độc lập".
Bảng cổng **không đổi: tám cổng, hai đạt, sáu trượt**. Chạm test: **vẫn 0 lần cho nghiên cứu này**.

## Câu hỏi chưa giải quyết

1. Gắn lại 36 clip tự tin thấp, không áp lực thời gian, ≥24 h sau (≈15 phút)? Nếu κ nhóm đó vẫn
   ≈0,15 thì nhập nhằng là thật và §6 rất chắc; nếu tăng mạnh thì phải viết lại mục này. Đây là
   phép đo còn lại có giá trị cao nhất, **không chạm test**.
2. Lệch định nghĩa R4: sửa sổ tay rồi gắn lại, hay khai thành giới hạn? Sửa sổ tay sau khi thấy số
   là **sai quy trình** — chỉ còn cách khai.
3. Pass 3 cho 25 clip lặp (Sửa đổi 2) có làm không, để κ nội-người thành ước lượng thật?
4. §6 viết lại: ai viết, khi nào? Đóng băng lần hai là **17/12** — còn 30 ngày.
