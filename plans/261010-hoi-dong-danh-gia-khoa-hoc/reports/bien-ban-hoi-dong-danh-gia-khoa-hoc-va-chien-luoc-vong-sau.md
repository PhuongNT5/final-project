# BIÊN BẢN HỘI ĐỒNG ĐÁNH GIÁ KHOA HỌC — VÀ CHIẾN LƯỢC VÒNG SAU

Ngày: 10/10/2026 · Nhánh: `next-round` · Mốc đóng băng số liệu đã qua: 05/10/2026
Thành phần: 4 phản biện khảo sát tài liệu ngoài (R1–R4), 2 phản biện bằng chứng nội bộ (R5–R6), chủ trì tổng hợp + tự kiểm chứng.
Đối tượng: phát biểu của học viên — *"kết quả 7x%–8x% còn thấp và tiêu cực"*.

---

## 1. PHÁN QUYẾT

**Tiền đề của học viên SAI, và sai theo đúng cơ chế mà luận văn tồn tại để vạch ra.**

Cảm giác "thấp" chỉ xuất hiện khi neo 84.2% vào 95.81%. Hai con số đó thuộc hai kho, hai bài toán,
hai giao thức. Chính §6.1.1 của luận văn cấm đặt chúng cạnh nhau — và học viên đang vô tình phạm
đúng điều cấm đó với chính kết quả của mình.

Nhưng hội đồng **không** dừng ở việc bác bỏ tiền đề. Ba phép đo độc lập, chạy lại trên artifact có
sẵn trên đĩa, cho một kết luận mạnh hơn và cụ thể hơn:

> **84.2% không phải một con số thấp cần nâng. Nó là một con số đã cạn dư địa rẻ, bị phương sai
> theo nguồn chi phối, và nằm sát trần nhiễu nhãn ước lượng được.**

| Phép đo | Kết quả | Hệ quả |
|---|---|---|
| Dò ngưỡng oracle (lạc quan, hậu nghiệm) | 84.29% tại t=0.480 — **+0.12 pp** | Ngưỡng 0.5 đã gần tối ưu |
| Ngưỡng chọn "trung thực" (nested, per-fold) | 83.70% — **−0.47 pp** | Chọn ngưỡng đàng hoàng còn làm *xấu đi* |
| Ensemble trung bình 5 mô hình (không chọn tay) | 84.50% — **+0.33 pp** | Cộng gộp an toàn gần như không cho gì |
| Lỗi nằm trong băng do dự 0.4–0.6 | **3/175 = 1.7%** | Lỗi là *tự tin-sai*, hiệu chuẩn không chữa được |
| Clip sai ở **cả 5** mô hình | **87/1100 = 7.9%** | Sàn lỗi không rút được bằng cùng họ đặc trưng |
| 3 fold tệ nhất chiếm | **48.0%** tổng lỗi | Lỗi tập trung ở nguồn khó, không trải đều |

Số tiêu đề cũng đã được **tái lập 1:1** từ tệp OOF 1,100 dòng: BA 84.17%, macro-F1 83.46%, và
**cả 10/10 fold khớp từng chữ số** với dãy in trong §6.2.1. Không có con số gõ tay nào.

Nói gọn: mô hình **không để lại điểm nào trên bàn**. Thứ anh/chị cảm thấy là "thấp" thực ra là
**dải 29.7 pp giữa fold dễ nhất và khó nhất** — một phát biểu về *dữ liệu*, không phải về *mô hình*.

---

## 2. NEO BÊN NGOÀI — CHỦ TRÌ TỰ KIỂM CHỨNG, KHÔNG LẤY NGUYÊN CỦA PHẢN BIỆN

### 2.1. iSafetyBench là benchmark ICCV 2025 Workshop, và nó khó

Nguồn kiểm chứng trực tiếp: arXiv:2508.00399v2 (Abdullah, Rawat, Vyas — VISION'25 @ ICCV 2025).

| Mô hình (zero-shot) | Normal SL acc | Hazard SL acc | ML F1 |
|---|---|---|---|
| Valley-Eagle-7B | 48.8 | 35.9 | 48.7 |
| Ovis2-8B | 47.3 | 40.3 | 53.4 |
| Qwen2.5-VL-7B-Instruct | 46.9 | 33.6 | 49.2 |
| InternVL2.5-8B-MPO | 42.2 | 38.3 | 50.8 |
| GPT-4o | 40.3 | 37.3 | 51.6 |
| VideoLLaMA3-7B | 38.9 | 32.7 | 39.7 |
| VideoChat-Flash-7B | 31.0 | 26.8 | 33.6 |
| Oryx-7B | 25.0 | 21.5 | 30.3 |

**CẤM TUYỆT ĐỐI — ghi ở đây để không ai trong nhóm lỡ miệng:** không được nói "em vượt GPT-4o".
Bảng trên là **165 lớp trắc nghiệm, zero-shot**; luận văn làm **nhị phân, có giám sát**. Đặt 84.2%
cạnh 40.3% là tái phạm đúng lỗi so sánh chéo làn mà toàn bộ Chương 6 tồn tại để lên án. Một câu nói
như vậy trước hội đồng sẽ phá huỷ luận điểm chính nhanh hơn bất kỳ con số thấp nào.

### 2.2. Điều quan trọng hơn bảng số: benchmark gốc KHÔNG định nghĩa phép chia huấn luyện

Bản gốc nêu rõ: đánh giá **zero-shot, không fine-tune, không có train split**. Hệ quả hai chiều:

- **Chiều bất lợi:** không có con số nào của ai để luận văn so sánh. Mọi tuyên bố "SOTA" đều vô nghĩa.
- **Chiều có lợi, và đang bị bỏ phí:** giao thức nhị phân, có giám sát, leave-source-out theo 151
  nguồn **là cấu trúc do chính luận văn dựng ra, và hiện chưa ai có số trên giao thức đó.** Luận
  văn không đang "thua" một bảng xếp hạng — nó đang **định nghĩa** một bảng xếp hạng.

### 2.3. Dải số thật của lĩnh vực ở giao thức khó

| Neo | Giao thức | Số |
|---|---|---|
| Sultani (CVPR'18), UCF-Crime | weakly-supervised | 75.4% AUC |
| RTFM (ICCV'21) | weakly-supervised | 84.30% AUC |
| MSL (AAAI'22) | weakly-supervised | 85.62% AUC |
| UCF-Crime SOTA 2024–25 | weakly-supervised | ~88% AUC |
| XD-Violence SOTA 2024 | weakly-supervised | ~86% AP |
| NTU RGB+D 120 | **cross-subject** | 74–85% |
| **Luận văn, B1** | **leave-source-out** | **84.2% BA** |

Ba neo đầu đã nằm sẵn trong §8.3.6 của chính luận văn. Dải 7x–8x% **chính là dải mà lĩnh vực đạt
được khi đo tử tế**. Các con số 9x% hầu như luôn đi kèm một giao thức dễ hơn — đó là phát biểu
trung tâm của luận văn, và tài liệu ngoài xác nhận nó.

---

## 3. CHẨN ĐOÁN: DƯ ĐỊA NẰM Ở ĐÂU, VÀ KHÔNG NẰM Ở ĐÂU

**Không nằm ở:** ngưỡng (+0.12), hiệu chuẩn (lỗi tự tin-sai), ensemble cùng họ (+0.33), kích thước
đầu phân loại (§6.2.6 đã chứng minh: đầu lớn hơn cho kết quả *xấu hơn* — B1xl 696K tham số = 82.7%).

**Nằm ở đúng ba chỗ, xếp theo bằng chứng:**

1. **Họ đặc trưng khác** — 87 clip sai ở cả 5 mô hình cùng dùng tháp CLIP. Muốn phá sàn đó phải đổi
   *tháp*, không đổi *đầu*. §6.2.6 đã nói trước: lợi ích là một khoản mua xương sống.
2. **Nguồn khó** — 48% lỗi ở 3 fold. Đây là bài toán *dữ liệu*, giải bằng thêm nguồn đa dạng ở vùng
   khó, không giải bằng kiến trúc.
3. **Chất lượng nhãn** — và đây là chỗ hội đồng phải cảnh báo mạnh (xem §4 dưới).

### 3.1. Trần nhiễu nhãn: con số quan trọng nhất của toàn bộ chiến lược, và nó CHƯA ĐƯỢC ĐO

κ = 0.65 hiện có được đo trên **Unsafe-Net, 4 lớp + nền, n = 100** — không phải trên nhãn nhị phân
của iSafetyBench. R5 ép tính hai cách và ra hai kết quả cách nhau 13 điểm (72% nếu chuyển ngây thơ;
85.0% nếu gộp ma trận 5×5 về 2×2 dưới một **giả định ánh xạ chưa ai xác nhận**). Chính độ nhạy đó
là lý do luận văn cấm suy κ giữa hai làn.

**Hệ quả chiến lược, và nó quyết định mọi thứ còn lại:** nếu trần nhãn thật sự ở ~85%, thì toàn bộ
tham vọng nâng 84.2% lên 90% là **không thể về nguyên tắc**, và mọi ngày công bỏ vào đó là lãng phí.
Nếu trần ở 93%, tuyến nâng cấp xương sống đáng giá. **Hiện không ai biết trần ở đâu.**

> **Khuyến nghị số một của hội đồng: đo trần trước khi trả tiền để tiến tới nó.**
> Một phép đo κ trên **chính nhãn nhị phân iSafetyBench**, n ≥ 200, hai người gán nhãn độc lập.
> Rẻ về tính toán, tốn công người, và nó là phép đo duy nhất biến lập luận phòng thủ mạnh nhất của
> luận văn ("chúng tôi đã sát trần nhãn") từ **phỏng đoán** thành **phép đo**.

---

## 4. HAI CHÂN TRỜI — RANH GIỚI LÀ MỐC ĐÓNG BĂNG 05/10

Hội đồng nhấn mạnh: hai chân trời này **không được trộn**, vì trộn chúng phá huỷ tài sản lớn nhất
mà luận văn đang có.

### Chân trời 1 — từ nay tới bảo vệ: KHÔNG CHẠM SỐ

Mọi con số đo **sau** 05/10 rồi đưa vào bảo vệ đều mời đúng một câu hỏi: *"con số này đăng ký trước
hay sau khi biết kết quả?"* — và câu trả lời sẽ là "sau". Đó chính là hành vi mà §6.11.3, sổ đăng ký
số đã rút, và 8 cổng tiền đăng ký được viết ra để lên án. Hai ngày 08–09/10 đóng 17 lỗ hổng liêm
chính có thể sụp trong một câu hỏi.

Việc của chân trời 1, xếp theo đòn bẩy:

| # | Việc | Chi phí | Đòn bẩy |
|---|---|---|---|
| 1 | **Chữ ký GVHD** vào báo cáo giải trình 5 thay đổi (A4) | 0 ngày công, chỉ cần gặp | Hạng mục **duy nhất** còn ảnh hưởng ĐẠT/KHÔNG ĐẠT |
| 2 | **Diễn tập hỏi–đáp đúng thứ tự**: lạm phát giao thức *trước*, accuracy *sau* | 2–3 giờ | Cứu mọi câu hỏi về accuracy; chi phí liêm chính = 0 |
| 3 | **Quay video demo dự phòng** (C6) — gộp luôn phép đo B1 cửa sổ trượt | 1 buổi | Đóng 2 hạng mục bằng 1 buổi |
| 4 | B10 (p95/soak), C4 (INT8) | giữ nguyên | Đã hạ mức thành "khai minh bạch"; không cần làm thêm |

Rủi ro diễn đạt cần tập riêng: một câu lỡ miệng kiểu *"em được 84.2% thay vì 95.8% vì dữ liệu ít"*
đúng về mặt sự kiện nhưng **phá sập toàn bộ khung phương pháp luận trong một câu**. Khung đúng là:
*"em có con số chín mươi mấy phần trăm, đo được, có log — và chọn không dùng nó."*

### Chân trời 2 — nhánh `next-round`: nơi mọi tham vọng nâng chỉ số thuộc về

Đây mới là chỗ trả lời câu hỏi của anh/chị. Chương trình dưới đây **phải được tiền đăng ký trước khi
chạy** — viết ngưỡng và tiêu chí vào git, đẩy lên, *rồi* mới chạy. Lần này làm đúng cách mà §8.3.7
tự khai là lần trước chưa làm được.

---

## 5. CHƯƠNG TRÌNH THÍ NGHIỆM VÒNG SAU — XẾP THEO (BẰNG CHỨNG × GIÁ TRỊ) / CÔNG SỨC

### E1. Đo trần nhiễu nhãn trên chính nhãn nhị phân iSafetyBench — **LÀM TRƯỚC TIÊN**
- **Vì sao đứng đầu:** nó quyết định E2 có đáng làm hay không. Đo trần trước khi trả tiền tiến tới trần.
- **Cách:** n ≥ 200 clip, 2 người gán nhãn độc lập, cùng SOP, tính κ + Po trên nhãn nhị phân.
- **Tiền đăng ký:** nếu Po ≤ 86% → tuyên bố mô hình đã ở trần nhãn, **đóng tuyến accuracy**, chuyển
  toàn bộ công sức sang E3/E4. Nếu Po ≥ 92% → E2 đáng chạy.
- **Chi phí:** công người, không phải compute. `run_kappa_prepare.py` / `run_kappa_compute.py` đã có.

### E2. Thang xương sống dưới ĐÚNG bộ fold đã công bố — tuyến accuracy duy nhất có bằng chứng
- **Căn cứ:** §6.2.6 (lợi ích đến từ tháp, không từ đầu) + 87 clip sai ở cả 5 mô hình cùng họ CLIP.
- **Cách:** giữ nguyên đầu B1 201,090 tham số, **chỉ đổi tháp đóng băng**: SigLIP2-B/16, CLIP
  ViT-L/14, DINOv3, V-JEPA2. Dùng lại đúng `StratifiedGroupKFold(10, seed=0)` theo 151 nguồn ở Phụ lục B.
- **Hạ tầng đã sẵn, chủ trì kiểm trực tiếp 10/10:** video thô nằm sẵn trên máy — `data/isafetybench/`
  **1,100 clip .mp4**, đã chia sẵn `hazard/` (313 MB, 420 clip) và `normal/` (923 MB), 151 nguồn,
  khớp đúng 1,100 dòng OOF. (R5 báo "1,818 clip" — **sai**; và bố cục nhị phân sẵn có là tin tốt hơn.)
  Trọng số **CLIP ViT-L/14 đã nằm trong cache HuggingFace** (1.6 GB) → chạy offline, không cần tải.
  `run-esta-lso.py` đã có cờ `--feats`, và `in_dim` đọc từ chính tệp đặc trưng (dòng 97) → **đổi
  512→768 chiều không cần sửa một dòng nào trong bộ huấn luyện**.
- **Chi phí đã đo trên máy này (MPS, 10/10):** B/32 5.35 ms/ảnh, L/14 91.55 ms/ảnh (chậm 17.1×).
  B1/B1w chỉ đọc nhánh **toàn khung** (`esta_training.py:66-68`) = 17,600 forward → **≈27 phút tính
  toán** cho L/14. Trích cả hai nhánh (thêm 140,800 crop thực thể) mới là ≈4 giờ — **E2 không cần nó**.
  **Đo lại trên lượt chạy thật (10/10, 1,100 clip): 5.75 s/clip → ~100 phút.** Con số 27 phút là
  thuần tính toán; **giải mã video chiếm phần lớn thời gian thực** (CPU một core ~60%, MPS rảnh).
  Hệ quả cho các bậc thang sau: nút cổ chai là giải mã, nên cache khung đã giải mã một lần rồi
  dùng lại cho mọi tháp sẽ cắt gần hết chi phí của SigLIP2/DINOv3 — đáng làm trước bậc thứ hai.
- **Tiền đăng ký:** ≥ +3.0 pp so với 84.2%, CI theo cụm loại trừ 0, trên cùng bộ fold.
- **Phản đối về độ trễ đã được chủ trì bác bỏ bằng số học.** R3 kết luận L/14 "vỡ ngân sách" — sai
  khung: hợp đồng của làn B1 là **nhịp**, không phải trần 100 ms. Thay tháp 58.1 ms bằng tháp ~160 ms:

  | Tháp | Quyết định | dec/s | Biên vs clip trung vị | Biên vs clip 60 fps |
  |---|---|---|---|---|
  | CLIP ViT-B/32 (đo) | 167.3 ms | 5.98 | 4.78× | 2.39× |
  | SigLIP-B/16 (ước) | 164.2 ms | 6.09 | 4.87× | 2.44× |
  | CLIP ViT-L/14 (ước) | 269.2 ms | 3.71 | 2.97× | **1.49×** |
  | DINOv2-L/14 (ước) | 259.2 ms | 3.86 | 3.09× | 1.54× |

  Tháp hạng L **vẫn vượt hợp đồng nhịp ở mọi điều kiện clip**. Độ trễ không chặn tuyến này.
  (Cột "ước" là số học từ độ trễ tháp ước lượng, **không phải phép đo trên thiết bị** — phải đo lại
  trước khi trích dẫn.)

### E3. Chẩn đoán fold khó → mở rộng nguồn có mục tiêu
- **Căn cứ:** 48% lỗi ở 3 fold; dải 29.7 pp. Đây là nguồn thật của cảm giác "thấp".
- **Cách:** đặc tả 3 cụm nguồn khó (bối cảnh, độ phân giải, tỉ lệ lớp, độ dài clip), rồi bổ sung
  nguồn đa dạng *đúng vùng đó*, không bổ sung đều.
- **Giá trị:** tấn công phương sai — thứ thực sự chặn triển khai, theo đúng §8.3.1.

### E4. Đường cơ sở WS-VAD dưới đúng giao thức leave-source-out
- **Căn cứ:** §8.3.6 đã nêu sẵn. Hiện iSafetyBench **không có** đường cơ sở có giám sát nào.
- **Giá trị kép:** (a) cho bài toán nhị phân một đường cơ sở có nghĩa lần đầu tiên; (b) kiểm chứng
  lạm phát giao thức có tái lập ở **họ phương pháp hoàn toàn khác** không — tức mở rộng kết quả chính
  ra ngoài họ kiến trúc đã dùng. Đây là thí nghiệm có giá trị công bố cao nhất.

### E5. Công bố — tuyến khả thi, không phải tham vọng viển vông
- Luận văn đang nắm ba thứ hiếm: một phép đo lạm phát **tái lập trên hai kho**, một **đối chứng giả
  dược** tách hình học fold khỏi danh tính nguồn, và **giao thức có giám sát đầu tiên** trên một
  benchmark ICCVW'25 mới ra.
- Đích hợp lý: workshop ICCV/CVPR hoặc track Datasets & Benchmarks. Điều kiện cần: E1 + E4, và một
  dấu thời gian tiền đăng ký **của bên thứ ba** (OSF) — vá đúng lỗ hổng mà §8.3.7 tự khai.

### E6. Câu hỏi khoa học mở quan trọng nhất (§8.3.5) — quan hệ thực thể–VÙNG
- Kết luận phủ định hiện tại có phạm vi hẹp: nó đúng cho quan hệ **thực thể–thực thể**, trên kho
  **không có đa giác vùng** (giới hạn R45). Chưa ai kiểm chứng thực thể–vùng ở kho đủ nhóm.
- Cần một kho có **đồng thời**: đa giác vùng, ≥ ~50 nhóm độc lập, nhãn mức clip. Chưa tồn tại →
  đây là một đóng góp *dữ liệu* tiềm năng, không chỉ là một thí nghiệm.

---

## 6. ĐỐI CHIẾU `plan-next-strategy.md` CỦA HỌC VIÊN

| Mục trong kế hoạch | Phán quyết hội đồng |
|---|---|
| 1. Tăng cường tập test 64 clip, augmentation, xoay vòng | **Lỗi thời.** Làn 64 clip đã bị thay bằng làn 1,100 clip. Augmentation **không** sửa được rò rỉ nguồn — nó nhân bản chính nguồn đã có. Đừng làm. |
| 2. Thử YOLO26 | **Đã làm rồi, và đã trượt.** Cổng G17: ESTA @ YOLO26s = 81.6% so với 83.8% @ v11s — **−2.2 pp**; 45,512 so với 52,997 phát hiện. YOLO26 là công cụ có thật (Ultralytics, 01/2026, NMS-free, CPU nhanh hơn 43%), nhưng **giả thuyết chưa kiểm chứng** là hạ ngưỡng tin cậy của bộ phát hiện mới có khôi phục chênh lệch không (§8.3.5). Đó mới là thí nghiệm, không phải "đổi sang YOLO26". |
| 3. Dựng lại kiến trúc với YOLO26 + backbone hiệu quả + CometML | Phần **backbone**: đúng hướng — chính là E2, và là tuyến accuracy duy nhất có bằng chứng. Phần **"Transformer hiệu quả hơn"**: hội đồng phản đối — B1xl 696K tham số cho 82.7%, *xấu hơn* B1 201K. Đầu lớn hơn đã được đo là đi lùi. Phần **CometML**: nên làm, hợp chuẩn 2026. |
| 4. Đánh bbox nhiều hơn + tracking ID | **Rủi ro cao, cần đọc kỹ.** Ba cổng độc lập (G11, G14, G18) đã bác bỏ phân rã theo thực thể, và ablation chỉ ra cơ chế: toàn bộ khả năng phân biệt nằm ở vector ngoại hình; hình học hộp một mình mất 11.0 pp. Tracking **không** sửa được điều đó — nó thêm tính bền danh tính, một biến chưa được kiểm chứng. Chỉ đáng làm **trong khuôn khổ E6** (thực thể–vùng), không phải như một bản vá cho G14. |
| 5. Nhiều độ đo hơn, tốc độ trên Jetson, theo metric của paper | Đã có đủ và hơn: BA, macro-F1, PR-AUC, ROC-AUC, ma trận nhầm lẫn ×3, bootstrap theo cụm, McNemar, ECE, dec/s đo trên thiết bị. **Không thiếu độ đo.** Thứ thiếu là p95/soak của vòng lặp tổ hợp (B10). |

---

## 7. NHỮNG GÌ CHỦ TRÌ KHÔNG KIỂM CHỨNG ĐƯỢC — KHÔNG ĐƯỢC TRÍCH DẪN

Theo đúng chuẩn xuất xứ của chính dự án này, các mục sau do phản biện đưa ra nhưng **chủ trì chưa
xác minh**, và **không** được đưa vào luận văn hay bài báo khi chưa kiểm:

- "Roth et al. 2026, 2,047 benchmarks, phân loại 4 lớp rò rỉ" (R2) — chưa xác minh.
- "SafeSceneReason", "PaSBench-Video" (R4) — chưa xác minh tồn tại.
- "Tracking ID cho +8–15% accuracy, CVPR 2025" (R4) — chưa xác minh, và mâu thuẫn với bằng chứng nội bộ.
- "GPT-4o 48.35% trên iSafetyBench" (R3) — **sai**; chủ trì kiểm trực tiếp: 40.3 / 37.3 / 51.6.
- Số trần nhiễu nhãn "85–87%" của R3 — mâu thuẫn nội tại với chính phép tính 1−2ε của nó; dùng số
  của R5 (85.0% Po dưới một giả định ánh xạ được khai rõ) thay thế, và vẫn chỉ ở mức **chỉ dấu**.
- Mọi độ trễ tháp hạng L trên Jetson ở §5/E2 là **số học, chưa đo trên thiết bị**.

---

## 8. CÂU HỎI CHƯA GIẢI ĐÁP

1. Trần nhiễu nhãn trên nhãn nhị phân iSafetyBench — **chưa đo**. Quyết định toàn bộ việc E2 có đáng làm không.
2. ~~`tools/analyse-isb-fold-failures.py` chưa chạy lại; R5 dựng fold map bằng tay.~~ **ĐÃ ĐÓNG
   10/10:** `tools/analyse-backbone-ladder.py` lấy bản đồ fold từ chính `isafetybench_dataset.folds`
   — cùng lời gọi mà driver huấn luyện dùng — và tái lập **đúng 10/10** dãy BA in ở §6.2.1
   (85.8 88.4 74.7 79.1 83.6 67.5 97.2 96.9 81.0 86.2), dải 29.7 pp. Phép đo tay của R5 đúng.
   Phát hiện phụ: dãy đó là **balanced accuracy**, không phải accuracy thuần (acc thuần:
   86.8 86.9 79.4 79.2 81.1 62.3 97.2 96.2 84.0 86.8) — đáng ghi rõ nhãn trong luận văn.
3. Ánh xạ "lớp 0–3 Unsafe-Net = hazard, nền = normal" dùng ở phép gộp κ là **giả định của R5**, chưa ai xác nhận có nghĩa.
4. Cặp ensemble hậu nghiệm tốt nhất (B1w+ESTA = 86.63%, +2.46 pp) là nhiễu lấy mẫu hay thật? Chỉ trả lời được bằng một tập đánh giá mới + chọn cặp **trước** khi nhìn số.
5. ECE 11.4 pp dựa trên bin rất mỏng — nên bootstrap CI hay bỏ hẳn, chỉ giữ "1.7% lỗi trong băng 0.4–0.6"?
6. GVHD đã ký báo cáo giải trình chưa? (A4 — hạng mục duy nhất còn ảnh hưởng ĐẠT/KHÔNG ĐẠT.)
7. Lịch bảo vệ chính xác — quyết định chân trời 1 còn bao nhiêu ngày.
