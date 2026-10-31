# Hội đồng phản biện & đánh giá khoa học — 31/10/2026

Phiên thứ ba. Tiếp nối 05/10 (phản biện luận văn) và 10/10 (đánh giá khoa học).

## Kết quả

**ĐẠT CÓ ĐIỀU KIỆN** — giữ mức 05/10. Ràng buộc quyết định **đã dịch từ khoa học sang thủ tục
và xuất xứ**. Xác suất bảo vệ thành công: **55–65 %** hiện tại → **78–85 %** chỉ bằng câu chữ +
diễn tập → **85–90 %** nếu thêm một phép đo κ.

Phát biểu trung tâm: **bản nộp sạch hơn vùng xung quanh nó.** 11 tệp `report/vn/` không rò rỉ số
đã rút, 0 vi phạm định dạng, 12/12 hình. **Mọi phát hiện nặng nằm NGOÀI bản nộp** — ở `README.md`,
bảng theo dõi, dấu thời gian xuất xứ, thủ tục hành chính.

## Tài liệu

| Tệp | Nội dung |
|---|---|
| [reports/bien-ban-tong-hop-hoi-dong-2026-10-31.md](reports/bien-ban-tong-hop-hoi-dong-2026-10-31.md) | **Biên bản tổng hợp** — phán quyết 6 trục, 10 phát hiện mới, 4 điều chỉnh của chủ tịch với phản biện, 9 câu hỏi mở |
| [ke-hoach-chien-luoc-sau-phien-31-10.md](ke-hoach-chien-luoc-sau-phien-31-10.md) | **Kế hoạch chiến lược** — 3 nhánh theo ngày bảo vệ, hạng mục 0/1/2/3, 6 điều không được làm |
| [reports/phan-bien-07-kiem-toan-dong-lo-hong.md](reports/phan-bien-07-kiem-toan-dong-lo-hong.md) | kiểm toán 25 hạng mục của hai phiên trước |
| [reports/phan-bien-08-phuong-phap-thong-ke.md](reports/phan-bien-08-phuong-phap-thong-ke.md) | phương pháp, thống kê, hiệu lực cấu trúc |
| [reports/phan-bien-09-tinh-moi-dinh-vi-cong-bo.md](reports/phan-bien-09-tinh-moi-dinh-vi-cong-bo.md) | tính mới, định vị, công bố *(venue đã bị chủ tịch đính chính)* |
| [reports/phan-bien-10-trien-khai-bien-va-tai-lap.md](reports/phan-bien-10-trien-khai-bien-va-tai-lap.md) | triển khai biên, truy xuất tới từng chữ số, tái lập |
| [reports/phan-bien-11-van-ban-va-truy-xuat-tuyen-bo.md](reports/phan-bien-11-van-ban-va-truy-xuat-tuyen-bo.md) | rà soát văn bản, rò rỉ số đã rút, định dạng |
| [reports/phan-bien-12-dien-tap-phan-bien-gay-gat.md](reports/phan-bien-12-dien-tap-phan-bien-gay-gat.md) | diễn tập hỏi–đáp gay gắt, bảng phát ngôn bị cấm |

## Ba việc chặn — trạng thái 31/10

1. ~~`git push origin next-round`~~ **XONG 31/10** — `898f096`. 86 tệp/20,290 dòng, gồm 11 JSON đo Jetson không tái tạo được, không còn trên một đĩa. *Còn mở: `report/vn/` vẫn không có trên `origin/main`.*
2. **Sửa `README.md`** — **CÒN MỞ**. Dòng 20 rút `3.02 dec/s`, dòng 43–44 dùng lại nó làm kết quả tiêu đề. Tạo tác này đã lừa được chủ tịch hội đồng với kho mở trước mặt. Việc chặn duy nhất còn lại.
3. ~~Ghi ngày bảo vệ~~ **XONG** — **31/12/2026**, còn 61 ngày ⇒ **nhánh C**. A4 vẫn chưa có văn bản.

## Lịch — bảo vệ 31/12/2026 (61 ngày, nhánh C)

Tuần 01–07/11 là tuần quyết định: **A4** và **khởi động κ** đều phụ thuộc người khác nên không
hoãn được. Chuỗi κ dài 3 tuần là đường tới hạn duy nhất. **Đóng băng lần hai 17/12** — sau mốc đó
chỉ diễn tập, không sửa. Lịch đầy đủ ở §6 của kế hoạch chiến lược.

## Ba đòn bẩy cao nhất

**T1-1** A4 văn bản GVHD duyệt đổi hướng (0 ngày công, hạng mục duy nhất lật được ĐẠT/KHÔNG ĐẠT, treo 3 tuần) ·
**T1-6** diễn tập kèm bảng phát ngôn bị cấm (2–3 giờ) ·
**T2-1** κ trên nhãn nhị phân iSafetyBench, n ≥ 200 (1–2 ngày, không tiêu lần chạm test)

## Quyết định chiến lược

**Ngừng mua độ chính xác.** Năm phép đo độc lập cùng chỉ một hướng: ngưỡng trung thực −0.47 pp,
ensemble +0.33 pp, đổi đầu ≤ 0, đổi tháp **0/8 sống sót BH**, thêm nguồn cần **3.0×** kho hiện có.
Đóng hướng, chuyển nguồn lực sang xuất xứ + thủ tục + diễn tập.
