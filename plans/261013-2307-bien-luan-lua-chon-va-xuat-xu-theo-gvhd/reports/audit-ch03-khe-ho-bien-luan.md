# Audit Ch3 — khe hở biện luận (đối chiếu nhận xét GVHD)

Phạm vi: chỉ đọc `report/vn/chuong-03-phuong-phap.txt` (958 dòng) + `report/vn/00-trang-thai-bien-soan.txt`.
Không sửa file luận văn. Không đề xuất phép đo mới. Không bịa lý do — nơi không tìm thấy căn cứ,
ghi "chưa truy được căn cứ".

Quy ước mức biện luận:
- **ĐỦ** = có lý do + có xuất xứ (tự chọn/mặc định/theo công trình) + có nêu đánh đổi hoặc phương
  án bị loại.
- **THIẾU XUẤT XỨ** = có lý do nhưng không nói giá trị từ đâu ra.
- **THIẾU LÝ DO** = chỉ liệt kê giá trị, không nói vì sao.
- **THIẾU ƯU/NHƯỢC** = có lý do + xuất xứ nhưng không nêu phương án thay thế bị loại.

## A. Bảng kiểm từng lựa chọn thiết kế / tham số

### A.1. Đồ thị tính toán & xương sống

| Mục | Dòng | Giá trị | Tình trạng |
|---|---|---|---|
| Chọn CLIP ViT-B/32 (thay xương sống video) | 147-150, 153-157 | đóng băng, 87,849,216 tham số, 512 chiều | **ĐỦ** — lý do (ngữ nghĩa mạnh không cần tinh chỉnh video), xuất xứ ([28]), kiểm chứng bằng cổng G16 (không giả định). Tốt, không nên sửa. |
| Vì sao **ViT-B/32** mà không phải biến thể CLIP khác (B/16, L/14) | 147 | "CLIP ViT-B/32" | **THIẾU LÝ DO** — chỉ nêu tên checkpoint, không so vì sao chọn biến thể này. |
| T = 16 khung lấy mẫu cách đều (toàn clip, B1) | 149-151 | T=16 | **THIẾU LÝ DO** — không có căn cứ vì sao 16 (không phải 8/24/32); P1 lại dùng T=32 (dòng 436) không đối chiếu. |
| Loại RT-DETR khỏi đề cương | 81-90 | — | **ĐỦ** — lý do kỹ thuật rõ, có thử nghiệm sơ bộ đối chứng (không cải thiện, tăng chi phí). Tốt. |
| Khối Transformer thời gian dùng chung: d=128, 4 đầu, FFN=256, 1 lớp (B1/ESTA)/2 lớp (B1w), dropout 0.1, norm_first, GELU | 161-170 | bảng tham số | **THIẾU XUẤT XỨ** — không nói các số d/heads/FFN/dropout tự chọn, mặc định thư viện, hay theo công trình nào. Văn phong liệt kê bảng trần (xem mục C). |
| Mặt nạ `valid` toàn False → quay về chú ý toàn bộ khung | 172-175 | — | **ĐỦ** — có lý do kỹ thuật (tránh NaN, bảo thủ). Tốt, giữ nguyên. |
| Dùng chung đúng một khối thời gian cho 2 nhánh (nguyên tắc 3) | 177-182 | — | **ĐỦ** — lý do + đánh đổi (ràng buộc thiết kế để G14 sạch). Tốt. |

### A.2. Họ mô hình toàn khung (Bảng 3.2)

| Mô hình | Dòng | Giá trị | Tình trạng |
|---|---|---|---|
| B1 — 201,090 tham số | 194-196 | chọn theo "tiêu chí biên nhỏ nhất" | **THIẾU LÝ DO** — "biên tiêu chí" không được định nghĩa cụ thể là gì (biên gì, đo thế nào) trong Ch3. |
| B1w — khối thời gian 2 lớp, 333,570 tham số | 198-202, 218-220 | khớp dung lượng ESTA (352,258) | **THIẾU ƯU/NHƯỢC** — có lý do + xuất xứ (đăng ký 29/09) nhưng không nói vì sao chọn "thêm lớp" để khớp dung lượng chứ không phải tăng d hoặc FFN (phương án thay thế không được nêu/loại). |
| B1xl — d=192, FFN=384, 2 lớp, 696,962 tham số | 204-208, 222-224 | quy tắc "số tham số gần TSF nhất" | **THIẾU ƯU/NHƯỢC** — có quy tắc chọn (theo số tham số, không theo điểm) nhưng không nói vì sao scale d và FFN mà không scale số lớp hay số đầu; không nêu các cấu hình khác cũng đạt gần TSF bị loại ra sao. |
| B2 — VideoMAE-base-finetuned-kinetics, 233,858 tham số | 210-214, 226-236 | chọn đối thủ "mạnh và trung thực" | **ĐỦ** — lý do rõ (loại bản tự giám sát yếu hơn vì thắng đối thủ yếu không chứng minh gì), có xuất xứ (checkpoint HF cụ thể), đánh đổi nêu rõ (chọn theo hướng bất lợi cho luận văn). Tốt, giữ nguyên — ví dụ mẫu của biện luận tốt. |
| Lỗi nhãn backbone trong log (`backbone=CLIP ViT-B/32` in cứng cho cả B2) | 233-236 | — | **ĐỦ** — minh bạch về cái bẫy trích dẫn. Tốt. |

### A.3. Hợp đồng nhịp (cửa sổ/bước nhảy) — đúng mục GVHD nêu tên

| Mục | Dòng | Giá trị | Tình trạng |
|---|---|---|---|
| Cửa sổ quyết định 48 khung | 242-246 | 48 | **THIẾU LÝ DO** — chỉ liệt kê bảng 3 tham số, không nói vì sao 48 (không phải 32/64...). |
| Bước nhảy 24 khung (đường B1) | 244, 260-261 | 24 | **THIẾU LÝ DO** — tương tự; chỉ nói "bộ lấy mẫu triển khai phải trùng bộ lấy mẫu đánh giá" (lý do cho tính NHẤT QUÁN, không phải lý do cho GIÁ TRỊ 24 cụ thể). |
| Bước nhảy 8 khung (đường P3) | 261 | 8 | **THIẾU LÝ DO** — không có căn cứ vì sao 8, chỉ nói khác bước của B1 nên hai hợp đồng khác nhau (hệ quả, không phải nguồn gốc giá trị). |
| Ba mức "hợp đồng" (f/s) lấy từ phân bố tần số khung thật | 255-258 | — | **ĐỦ** — có xuất xứ rõ (đo trên 1,100 clip, Ch4 mục 4.7.4), có đối lập với cách làm sai đã bị rút lại (chọn tần số tiện lợi). Lưu ý: đây CHỈ giải thích gốc của NGƯỠNG hợp đồng, KHÔNG giải thích gốc của bản thân 48/24/8 — hai việc khác nhau, không nên gộp khi đọc. |

### A.4. Chọn thực thể & token hoá (đường thực thể)

| Mục | Dòng | Giá trị | Tình trạng |
|---|---|---|---|
| N_MAX = 8 thực thể/khung | 278-280 | 8 | **THIẾU LÝ DO** — chưa truy được căn cứ vì sao 8. |
| Ngưỡng tin cậy giữ thực thể = 0.25 | 279-280 | 0.25 | **THIẾU LÝ DO** — chưa truy được căn cứ. |
| Dạng A — FEAT=10 chiều (hình học/định danh) | 283-292 | liệt kê 10 trường | **THIẾU LÝ DO** cho việc chọn đúng 10 trường này (dù liệt kê rõ là trường gì — xem mục C về văn phong). |
| APP=64 chiều, nền MobileNetV3-small (ImageNet) | 287-288 | — | **THIẾU XUẤT XỨ** — không nói vì sao chọn MobileNetV3-small (so BA đầu dò khác ở G11 không liệt kê trong Ch3, xem A.8). Không có số [trích dẫn] cho MobileNetV3. |
| Phép chiếu ngẫu nhiên cố định 576→64 (không học được) | 287-288, 294-309 | — | **ĐỦ** — đây là phần biện luận TỐT NHẤT của mục: nêu rõ đây là cổ chai, hệ quả cho cách đọc G11, và lý do R48/G15 tồn tại để kiểm lại ở biểu diễn khác. Giữ nguyên, không sửa. |
| Đệm ngữ cảnh 30% quanh hộp (Dạng B) | 319 | 30% | **THIẾU LÝ DO** — có lý do ĐỊNH TÍNH (không xoá ngữ cảnh) nhưng không có căn cứ cho SỐ 30% cụ thể (không phải 15%/50%). |
| Tổng hợp 3 thành phần (ngoại hình+nhúng lớp+hình học) bằng CỘNG rồi LayerNorm, không nối | 320-322 | — | **ĐỦ** — có lý do kỹ thuật rõ ("hình học giữ nguyên cạnh embedding thay vì bị chiếu mất"). Tốt. |
| `process_clip` dùng chung 1 danh sách khung hợp lệ cho 2 nhánh | 324-330 | — | **ĐỦ** — bảo đảm ở mức mã nguồn, lý do rõ (G14 là đối chứng cấu trúc). Tốt, giữ nguyên. |
| Giới hạn R45 — kênh hình học-theo-vùng = 0 trên iSafetyBench | 311-315 | — | **ĐỦ** — minh bạch giới hạn thiết kế, nhắc lại đúng chỗ dùng. Tốt. |

### A.5. ESTA và thang encoder E0–E3

| Mục | Dòng | Giá trị | Tình trạng |
|---|---|---|---|
| ESTA giai đoạn 1: 1 lớp Transformer trong khung | 338-339 | 1 lớp | **THIẾU LÝ DO** — không nói vì sao 1 lớp ở giai đoạn không gian. |
| ESTA giữ dung lượng gần E0 (352,258 vs 298,118) | 348-349 | — | **ĐỦ** — lý do rõ (loại giải thích "mô hình lớn hơn"). Tốt. |
| Khung rỗng giữ `frame_cls` sống, loại qua mặt nạ thời gian | 345-347 | — | **ĐỦ** — lý do triết lý rõ ("khung rỗng là sự thật, không phải lỗi"), trích dẫn từ mã. Tốt, giữ nguyên. |
| E0: nhúng lớp 32 chiều, bảng vị trí 129 khe, 2 lớp, 4 đầu, FFN 256, dropout 0.3 | 362-366 | bảng tham số | **THIẾU XUẤT XỨ** — các số này KHÁC khối thời gian "dùng chung" ở mục 3.3.2 (1 lớp/dropout 0.1) mà Ch3 không giải thích vì sao thang E0–E3 có bộ siêu tham số riêng, không dùng khối 3.3.2. Không có lý do cho 32 chiều nhúng lớp cụ thể. |
| E0 — bảng vị trí 129 khe là tiên nghiệm SAI, đo bằng E2 | 387-393 | — | **ĐỦ** — ví dụ biện luận mẫu: có giả thuyết, có thí nghiệm cô lập (E2), có kết luận phương pháp. Tốt, giữ nguyên tuyệt đối. |
| E1 (FSTE): d=64 và d=96, bảng vị trí 17 khe | 368-373 | hai độ rộng | **THIẾU LÝ DO** riêng cho đúng 2 giá trị 64/96 (nằm trong khung "thang dung lượng" chung nhưng không nói vì sao đúng 2 điểm này). |
| E2 — đối chứng ngoại khoa (chỉ đổi bảng vị trí) | 375-378 | — | **ĐỦ** — cô lập đúng 1 biến, mẫu biện luận tốt (nguyên tắc 3). Giữ nguyên. |
| E3 — DeepSets, không chú ý, chốt làm encoder tiêu đề (D3) trước khi có bảng kết quả | 380-383, 395-401 | 17,270 / 18,044 tham số | **ĐỦ** — lý do (mô hình nhỏ nhất, chống chọn lọc hậu nghiệm), tường minh hệ quả ("E0 không được đề bạt..."). Mẫu biện luận tốt nhất của cả chương, giữ nguyên tuyệt đối. |
| TSF — embedding luồng học được, 1 lớp mã hoá hợp nhất | 410-416 | 686,726 tham số | **THIẾU LÝ DO** cho số lớp hợp nhất =1 cụ thể. |
| TSF — λ=0.3 cho đầu phụ mỗi nhánh | 421-424 | 0.3 | **THIẾU ƯU/NHƯỢC** — có xuất xứ ("đúng hệ số các đầu quan hệ nhị phân trong dự án đã dùng" = quy ước kế thừa, không phải núm điều chỉnh) nhưng không nêu phương án khác bị loại hay vì sao 0.3 (chứ không phải 0.5 như số hạng khung ở 3.6.1) là hợp lý ở đây. |

### A.6. Đường P1 (giai đoạn trước) — mục liệt kê nặng nhất của chương

| Mục | Dòng | Giá trị | Tình trạng |
|---|---|---|---|
| YOLOv11s đóng băng, lấy đầu ra lớp 22 (C3k2) | 433-434 | lớp 22 | **THIẾU LÝ DO** — không nói vì sao lớp 22. |
| Gộp trung bình thích nghi 4×4, nối 3 thống kê (mean/std/max) → 1,536 chiều | 434-436 | 4×4, 3 thống kê | **THIẾU LÝ DO** cho cả lưới 4×4 và việc chọn đúng 3 thống kê này. |
| Cửa sổ trượt T=32 (khác T=16 của B1, không đối chiếu) | 436 | 32 | **THIẾU LÝ DO** |
| 4 luồng thời gian (thô, đạo hàm 1, đạo hàm 2, thống kê cửa sổ 4 khung), d=256, MLP cổng | 436-439 | — | **THIẾU LÝ DO** cho việc chọn đúng 4 luồng này và d=256. |
| Transformer 4 lớp, 4 đầu, FFN 1,024, dropout 0.3, vị trí học được dài 32 | 439-440 | bảng tham số | **THIẾU XUẤT XỨ** toàn bộ. |
| 5 đầu (direct4, scenario, 4 đầu cặp) hợp nhất bằng xác suất có cổng theo scenario | 440-441 | — | **THIẾU LÝ DO** cho kiến trúc hợp nhất 5-đầu. |
| Phạm vi P1 bị giới hạn tường minh (không triển khai, không headline, 2 kết quả cũ đã rút) | 443-453 | — | **ĐỦ** — minh bạch phạm vi rất tốt; lý do GIỮ LẠI đặc tả (dù đã thay thế) được nêu rõ (443-453). Tốt, giữ nguyên. |

Toàn mục 3.5 (dòng 430-441) là ví dụ rõ nhất khớp nhận xét GVHD #1/#4/#5: liệt kê dày đặc tham số
kiến trúc không kèm lý do lựa chọn nào, dù phần "phạm vi sử dụng" (443-453) được biện luận tốt.

### A.7. Quy trình huấn luyện dùng chung (3.6)

| Mục | Dòng | Giá trị | Tình trạng |
|---|---|---|---|
| AdamW lr=1e-3, weight_decay=0.01 | 465 | — | **THIẾU XUẤT XỨ** — không nói mặc định thư viện, theo công trình, hay tự dò ở đâu. |
| Cosine annealing theo epoch | 466 | — | **THIẾU XUẤT XỨ** |
| `label_smoothing=0.05` | 467, 481 | 0.05 | **THIẾU XUẤT XỨ** |
| Trọng số lớp tỉ lệ nghịch tần suất, tính từ phần huấn luyện fold | 468 | — | **THIẾU ƯU/NHƯỢC** — có xuất xứ (tính từ fold, không rò rỉ), có lý do ngầm (lệch lớp) nhưng không so với phương án cân bằng khác (ví dụ oversampling, focal loss). |
| Chặn gradient chuẩn tối đa 5.0 | 469 | 5.0 | **THIẾU XUẤT XỨ** |
| Kích thước lô 32 | 470 | 32 | **THIẾU XUẤT XỨ** |
| Số epoch 60 | 471 | 60 | **THIẾU XUẤT XỨ** |
| Chọn điểm dừng theo `val_macro` cao nhất | 472, 499-507 | — | **ĐỦ** — lý do rõ, có xử lý rò rỉ fit/val (509-519), tường minh hệ quả. Tốt, giữ nguyên. |
| Hai hệ số mất mát phụ 0.3 và 0.5 | 478-482 | 0.3, 0.5 | **THIẾU LÝ DO** (nhưng CÓ xuất xứ: văn bản tự nhận "giá trị mặc định của vòng huấn luyện và không được dò tìm" — đây là một lời thừa nhận trung thực, không phải che giấu, nhưng chưa phải lý do vì sao các giá trị này hợp lý). |
| Huấn luyện trên CPU (không MPS) | 491-497 | — | **ĐỦ** — lý do kỹ thuật cụ thể (lỗi `addcmul_` trên MPS), đánh đổi nêu rõ (chi phí thấp vì đầu nhỏ). Mẫu biện luận tốt, giữ nguyên. |
| Val split = 15% từ trong phần huấn luyện fold | 505-507 | 15% | **THIẾU LÝ DO** cho đúng số 15% (không phải 10%/20%). |
| 3 hạt giống (0,1,2), tổ hợp trung bình softmax (không lấy hạt giống tốt nhất) | 523-526 | 3 | **ĐỦ** phần "không lấy tốt nhất" (chống chọn lọc) nhưng **THIẾU LÝ DO** riêng cho đúng số 3 hạt giống. |

### A.8. Giao thức đánh giá (3.7) — phần được biện luận tốt nhất, nhưng còn lỗ hổng cụ thể

| Mục | Dòng | Giá trị | Tình trạng |
|---|---|---|---|
| StratifiedGroupKFold vs StratifiedKFold, Δ lạm phát giao thức | 535-634 | — | **ĐỦ** toàn mục — định nghĩa chính xác, tính chất ghép đôi, dấu vết nhận dạng kiểm tra được, ví dụ số liệu kiểm tra ngược (570-572, 576-583). Mẫu biện luận tốt nhất cả chương, giữ nguyên tuyệt đối. |
| Số fold = 10 | 568-572 và nhiều chỗ | 10 | **THIẾU LÝ DO** — không có căn cứ vì sao 10 fold. |
| KMeans(8) cho nhóm bối cảnh Unsafe-Net | 585-590 | K=8 | **THIẾU LÝ DO/XUẤT XỨ** — không nói vì sao K=8 (không phải 4/16...); chỉ nêu HỆ QUẢ của việc chỉ có 8 nhóm (giới hạn thống kê), không nêu CĂN CỨ chọn K=8 khi fit KMeans. |
| BA (độ chính xác cân bằng) là chỉ số chính | 594-597 | — | **ĐỦ** — lý do rõ (lệch lớp, ví dụ số 61.8%). Tốt. Nhưng **không có công thức** — xem mục B. |
| Đường sàn 63.3% (quy tắc đếm 3 đặc trưng) | 599-608 | 63.3% | **ĐỦ** về lý do chọn ngưỡng và quy trình đăng ký trước (mẫu biện luận rất tốt) **nhưng THIẾU XUẤT XỨ cho CƠ CHẾ "khớp"**: văn bản nói 3 đặc trưng "được khớp" nhưng không nói khớp bằng mô hình/quy tắc gì (hồi quy logistic? ngưỡng tay?) — khớp đúng mục GVHD #7 (baseline cần có công thức/cơ chế). |
| 300 clip "thăm dò cân bằng" cho quy tắc đếm | 603 | 300 | **THIẾU LÝ DO** cho đúng số 300. |
| Bootstrap theo cụm, đơn vị = nhóm | 616-621 | — | **ĐỦ** lý do (clip cùng nguồn không độc lập) + xuất xứ ([39]) + hệ quả bất lợi được chấp nhận công khai (629-634). Tốt. Nhưng **không có công thức/thuật toán resampling** — xem mục B. |
| McNemar chính xác, ưu tiên hơn khi trái kết luận với CI cụm | 623-627 | — | **ĐỦ** lý do (McNemar bỏ qua cấu trúc cụm) + xuất xứ ([40]). Không có công thức — xem mục B. |
| "Ba đầu dò" cho G11 (chỉ 1/3 được nêu tên: MobileNetV3-small) | 687-691 | — | **THIẾU XUẤT XỨ** — Ch3 không liệt kê 2 đầu dò còn lại, nên không kiểm tra được "tốt nhất trong ba" là so với cái gì. |

### A.9. Tám cổng (3.8) — ngưỡng biên pp

| Cổng | Dòng | Biên | Tình trạng |
|---|---|---|---|
| G10 | 678-682 | biên dưới ~3pp không coi là kết quả | **THIẾU XUẤT XỨ** — không nói ngưỡng ~3pp từ đâu (tính công suất thống kê? kinh nghiệm dự án trước? mốc tuỳ ý?). |
| G11 | 687-694 | ~3pp | **THIẾU XUẤT XỨ** tương tự |
| G12 | 699-704 | ≥5pp "lớn về lượng" | **THIẾU XUẤT XỨ** — vì sao 5pp (khác ngưỡng 3pp của các cổng khác) không giải thích. |
| G14 | 719-725 | ≥+3pp, p<0.05 | **THIẾU XUẤT XỨ** |
| G15 | 730-733 | ≥+3pp | **THIẾU XUẤT XỨ** |
| G18a/G18b | 757-765 | 3.0pp / 2.0pp | **THIẾU XUẤT XỨ** — hai bậc biên khác nhau, không nói vì sao đúng 3.0 và 2.0. |

Toàn bộ các ngưỡng pp trong Bảng 3.4 dùng nhất quán xuyên các cổng (3pp là "tầm thường", 5pp là
"lớn về lượng") nhưng **không một nơi nào trong Ch3 nói nguồn gốc của chính các số 3/5pp này**
(không trích dẫn công trình, không nêu tính toán công suất thống kê, không nói "theo quy ước dự
án trước"). Đây là khe hở xuyên suốt cả mục 3.8, không phải lỗi riêng một cổng.

## B. Chỗ KHÔNG có công thức toán trong khi GVHD đòi công thức

| Đại lượng | Dòng định nghĩa bằng lời | Thiếu gì |
|---|---|---|
| Balanced accuracy (BA) | 594-595 "trung bình độ nhạy theo từng lớp" | Không viết công thức `BA = (1/C) Σ_c TP_c/(TP_c+FN_c)`. |
| macro-F1 | 596-597 chỉ được NÊU TÊN, không định nghĩa | Không có công thức hay mô tả cách tính. |
| PR-AUC, ROC-AUC | 597 chỉ nêu tên | Không có công thức/định nghĩa đường cong. |
| Δ lạm phát giao thức | 556 | Đây có công thức (Δ = BA(rò rỉ) − BA(có ràng buộc)) — **ĐỦ**, không phải khe hở. |
| Bootstrap theo cụm | 616-621 | Không có công thức/thuật toán resampling (số lần lặp B, cách resample theo nhóm, cách tính CI từ phân vị). Chỉ nêu tên kỹ thuật + trích dẫn [39]. |
| McNemar chính xác | 623-627 | Không có công thức thống kê kiểm định (bảng 2×2, thống kê kiểm định, phân phối). Chỉ nêu tên + trích dẫn [40]. |
| Quy tắc đếm ba đặc trưng (đường sàn 63.3%) | 599-608 | Không nói mô hình/quy tắc "khớp" ba đặc trưng là gì — không có công thức phân loại. |
| Hợp đồng nhịp f/s | 251-252 | Đây CÓ công thức (f/s khung/giây) — **ĐỦ**, không phải khe hở. |

Kết luận mục B: đúng như GVHD nêu, 3.7.3–3.7.4 là nơi tập trung thiếu công thức — mọi đại lượng
suy luận thống kê trong hai mục này chỉ được gọi tên + trích dẫn, không viết ra bằng ký hiệu toán.

## C. Văn phong liệt kê đáng chuyển thành văn biện luận

| Đoạn | Dòng | Vấn đề |
|---|---|---|
| Cấu hình khối Transformer thời gian dùng chung | 161-170 | Bảng 7 dòng tham số trần (d, đầu, FFN, lớp, dropout, chuẩn hoá, vị trí, gộp), không dòng nào có "vì sao". |
| Cấu hình hợp đồng nhịp | 242-246 | Bảng 3 dòng (cửa sổ/bước nhảy/T) trần, lý do chỉ xuất hiện ở đoạn văn sau (247-249) cho tính NHẤT QUÁN, không cho GIÁ TRỊ. |
| Đặc tả đường P1 | 433-441 | Một đoạn văn nhưng thực chất là liệt kê chuỗi ~15 con số/lựa chọn kiến trúc liên tiếp, không câu nào giải thích "vì sao". Đây là đoạn khớp nhất với nhận xét GVHD #1 "chỉ liệt kê". |
| Công thức huấn luyện dùng chung | 465-474 | Bảng 8 dòng siêu tham số trần (lr, lịch học, mất mát, trọng số lớp, chặn gradient, lô, epoch, thiết bị) — chỉ "thiết bị=CPU" có đoạn văn biện luận riêng (491-497); 7 dòng còn lại không có. |
| Dạng A token FEAT=10 chiều | 285-286 | Liệt kê tên 10 trường trong một dòng ký hiệu, không câu văn nào giải thích cơ sở chọn đúng 10 trường này (có giải thích 4 chiều cuối là gì ở 290-292, nhưng không giải thích TẠI SAO cần đúng các trường đó). |
| Bảng 3.3 (thang E0–E3), cột "Tham số" | 360-383 | Cột tham số và mô tả cơ chế khá đầy đủ bằng văn (không hẳn liệt kê trần) — xếp loại nhẹ hơn các mục trên, nêu ở đây để đối chiếu nhưng KHÔNG cần ưu tiên sửa. |

## D. Đã biện luận tốt — KHÔNG nên sửa

1. Bốn nguyên tắc thiết kế, dòng 26-52 — nền tảng lý giải toàn chương.
2. Lý do tách phương pháp khỏi kết quả, dòng 9-20.
3. Loại RT-DETR khỏi đề cương, dòng 81-90.
4. Mặt nạ `valid` toàn False → attention toàn bộ khung, dòng 172-175.
5. Nguyên tắc dùng chung đúng một khối thời gian cho 2 nhánh, dòng 177-182.
6. Chọn xương sống B2 theo hướng bất lợi cho luận văn, dòng 210-214, 226-236.
7. Phép chiếu ngẫu nhiên cố định là cổ chai — hệ quả cách đọc G11/R48, dòng 294-309.
8. Giới hạn R45 (kênh vùng = 0 trên iSafetyBench), dòng 311-315.
9. Bảo đảm cấu trúc mã nguồn cho G14 (`process_clip` dùng chung danh sách khung), dòng 324-330.
10. Khung rỗng giữ `frame_cls` sống — triết lý thiết kế, dòng 345-347.
11. E0 bảng vị trí 129 khe là tiên nghiệm sai, cô lập bằng E2, dòng 387-393, 375-378.
12. E3 chốt làm encoder tiêu đề TRƯỚC khi có bảng kết quả (quyết định D3), dòng 395-401.
13. Huấn luyện trên CPU vì lỗi `addcmul_` trên MPS, dòng 491-497.
14. Chọn điểm dừng theo `val_macro`, không rò rỉ fold giữ ra, dòng 499-519.
15. Tổ hợp 3 hạt giống bằng trung bình softmax, không lấy tốt nhất, dòng 523-526.
16. Toàn mục 3.7.1 (Δ lạm phát giao thức, định nghĩa + 4 tính chất + dấu vết nhận dạng), dòng 535-572.
17. Đường sàn 63.3% đăng ký trước (phần LÝ DO chọn ngưỡng, không phải phần CƠ CHẾ khớp — xem A.8), dòng 599-608.
18. Hệ quả bất lợi bootstrap theo cụm (0/151 nguồn mang cả 2 nhãn) được chấp nhận công khai, dòng 629-634.
19. Bảng 3.4 — khung "giả thuyết / nếu trượt" cho cả 8 cổng, dòng 665-766.
20. Toàn mục 3.9 (phương pháp đo trên thiết bị biên) — 4 điều kiện trích dẫn, 2 trạng thái xung nhịp, lý do cấm tổng giai đoạn — biện luận chặt từ đầu đến cuối, dòng 794-871.

## Câu hỏi chưa giải quyết

1. Vì sao T=16 (B1) và T=32 (P1) khác nhau — có phải do ràng buộc tính toán khác nhau giữa 2 giai đoạn dự án, hay tuỳ ý? Ch3 không nói.
2. Cơ chế "khớp" ba đặc trưng đếm (đường sàn 63.3%, dòng 599-608) là mô hình gì (hồi quy logistic, cây quyết định, ngưỡng tay)? Chưa truy được căn cứ trong Ch3.
3. Hai đầu dò còn lại trong "ba đầu dò" của G11 (dòng 687-691) là gì? Chỉ MobileNetV3-small được nêu tên.
4. Nguồn gốc các ngưỡng biên pp (3pp, 5pp, 2pp, 3.0pp) trong Bảng 3.4 — tự chọn theo kinh nghiệm, hay có tính toán công suất thống kê? Chưa truy được căn cứ.
5. Vì sao thang encoder E0–E3 dùng bộ siêu tham số riêng (2 lớp, dropout 0.3) khác khối Transformer thời gian "dùng chung" ở mục 3.3.2 (1 lớp, dropout 0.1) — có phải vì đầu vào khác (token thực thể 16×8 vs trạng thái khung 16) đủ để biện minh, hay là một sự không nhất quán văn bản chưa được giải thích?
6. Các giá trị số cụ thể 48 (cửa sổ), 24 và 8 (bước nhảy), N_MAX=8, ngưỡng tin cậy 0.25, đệm ngữ cảnh 30%, K=8 (KMeans), val split 15%, 10 fold, 300 clip thăm dò, lr=1e-3/wd=0.01/batch=32/epoch=60/clip=5.0 — tất cả đều "chưa truy được căn cứ" trong Ch3. Cần hỏi tác giả: đây là giá trị mặc định của một thư viện/mã nguồn kế thừa, hay tự chọn qua dò thử không ghi lại, hay theo một công trình cụ thể?
