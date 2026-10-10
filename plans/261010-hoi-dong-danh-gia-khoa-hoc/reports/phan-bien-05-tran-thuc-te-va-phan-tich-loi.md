# Phản biện 05 — Trần thực tế và phân tích lỗi (REVIEWER R5, internal evidence audit)

Phạm vi: chỉ đọc + 2 script phân tích tự viết, chạy trên artifact đã có sẵn trên đĩa. Không sửa file luận văn/report. Script: `/private/tmp/.../scratchpad/analyze_b1.py`, `/private/tmp/.../scratchpad/ensemble.py`.

## 1. Phán quyết

84.2% KHÔNG ở trần tuyệt đối, nhưng dư địa thật — tức dư địa không dính selection bias — là nhỏ, ước tính **<1pp** bằng các thủ thuật mô hình (ngưỡng, ensemble trung bình). Dư địa lớn hơn (ensemble oracle-selection +8pp) chỉ là giới hạn lý thuyết không khả thi khi triển khai (không biết trước phải tin mô hình nào cho từng clip). Trần nhiễu nhãn, suy từ ma trận kappa đã có — dù luận văn tự khai không được chuyển sang làn nhị phân — khi ép buộc tính, lại cho con số ~85% Po, **cao hơn** 84.2% một chút: nghĩa là mô hình đang chạy sát hoặc trên trần nhiễu nhãn ước lượng được, chứ không "thấp" so với trần đó.

Lỗi hiện tại KHÔNG phải lỗi biên giới (borderline/calibration-fixable): chỉ 1.7% số lỗi rơi vào băng prob 0.4–0.6; sai số trung bình |prob−0.5| của lỗi (0.417) gần bằng của dự đoán đúng (0.451) — mô hình sai một cách "tự tin" chứ không phải do ngưỡng. Lỗi tập trung mạnh: 3/10 fold tệ nhất (fold 3,4,6) chiếm 48% tổng lỗi, trong khi fold 7,8 gần như hoàn hảo (97.2%, 96.9%). Vậy cảm giác "thấp" của sinh viên bắt nguồn từ phương sai theo fold/nguồn cao (29.7pp giữa fold tốt nhất–xấu nhất), KHÔNG phải từ một lỗi hệ thống đồng đều có thể vá bằng ngưỡng hay ensemble rẻ.

Dư địa khả tín nhất không nằm ở mô hình/ngưỡng mà ở **dữ liệu/nguồn khó** (xem §3, §6): các fold khó đều là các cụm nguồn ít, lệch nhãn mạnh — hướng cải thiện là thêm nguồn đa dạng ở vùng khó, không phải đổi kiến trúc đầu.

## 2. Kiểm chứng lại số tiêu đề

Script: `analyze_b1.py`, input `sweep_results/relation_tokens/enc_isbclip_B1_lso-1100_headline-none_oof.json` (1100 dòng, n xác nhận đúng), fold map dựng độc lập từ `report/vn/phu-luc-b-cau-hinh-fold.txt` (151 nguồn → 10 fold).

```
Confusion: TP=355 FN=65 FP=110 TN=570 (n=1100, hazard=420, normal=680)
BA = 84.17%  (luận văn in 84.2% — khớp, chỉ khác làm tròn)
Sensitivity=84.52%  Specificity=83.82%  Precision(hazard)=76.34%  macro-F1=83.46% (luận văn 83.5% — khớp)
```

Per-fold BA tái lập, so với dãy in trong `chuong-06-danh-gia.txt:240`:

```
Luận văn :  85.8 88.4 74.7 79.1 83.6 67.5 97.2 96.9 81.0 86.2
Tái lập  :  85.8 88.4 74.7 79.1 83.6 67.5 97.2 96.9 81.0 86.2   -> KHỚP TUYỆT ĐỐI, cả 10/10 fold
```

Xác nhận: `y_pred == (prob_unsafe >= 0.5)` đúng cho toàn bộ 1100 dòng (quy tắc ngưỡng mặc định nhất quán, không có hậu xử lý ẩn). Không phát hiện sai lệch nào giữa số in trong luận văn và số tính lại từ artifact. **Kết luận: con số tiêu đề là thật, tái lập được 1:1 từ OOF JSON trên đĩa, không phải số gõ tay.**

## 3. Phân tích lỗi

**Ngưỡng (operating point), post-hoc trên toàn bộ 1100 dòng:**
- Ngưỡng oracle toàn cục (dò 0.01→0.99, chọn tốt nhất trên CHÍNH tập test — lạc quan, không khả thi thực tế): BA = 84.29% tại t=0.480. So với mặc định 0.5 → BA 84.17%. **Chênh chỉ +0.12pp.**
- Ngưỡng "trung thực" (nested: với mỗi fold test, chọn t tối ưu chỉ từ 9 fold train còn lại): BA = **83.70%**, THẤP HƠN mặc định 0.5 (84.17%). Nghĩa là ngay cả cách chọn ngưỡng "đàng hoàng" cũng không cải thiện — ngưỡng 0.5 đã gần tối ưu, dò ngưỡng không phải nguồn dư địa.

**Hiệu chuẩn (ECE, 10 bin):** ECE = 11.4pp (bin giữa (0.2–0.9) có n rất nhỏ, 5–19 dòng/bin, nên ước lượng nhiễu — không đọc ECE này như một con số chắc). Phân phối prob_unsafe lưỡng cực mạnh: 570/1100 dòng ở bin [0,0.1), 426/1100 ở bin [0.9,1.0] — model phần lớn rất tự tin.

**Lỗi có tự tin không?** n lỗi = 175/1100 (15.9%). Chỉ **3/175 lỗi (1.7%)** có prob_unsafe trong [0.4, 0.6] — băng "do dự". |prob−0.5| trung bình: lỗi = 0.417, đúng = 0.451 — gần bằng nhau. **=> lỗi chủ yếu là lỗi tự tin-sai (confidently wrong), không phải lỗi biên giới quyết định.** Hệ quả trực tiếp: hiệu chuẩn lại xác suất hay chỉnh ngưỡng không chữa được phần lớn lỗi này.

**Tập trung lỗi theo fold** (175 lỗi):
```
fold   1    2    3    4    5    6    7    8    9   10
err%  10.9  8.0 12.6 12.6 11.4 22.9  1.7  2.3  9.7  8.0   (% trên tổng 175 lỗi)
```
3 fold tệ nhất (6,3,4) chiếm **48.0%** tổng lỗi; fold 7+8 chỉ chiếm 4.0%. Lỗi KHÔNG trải đều — tập trung ở vài fold/nguồn khó, đúng như luận văn đã tự nhận (dải 67.5–97.2%, rộng 29.7pp).

## 4. Dư địa cộng gộp/ensemble

**CẢNH BÁO PHƯƠNG PHÁP (bắt buộc đọc trước số bên dưới):** mọi số trong mục này là phân tích hậu nghiệm (post-hoc) trên chính 1100 dòng OOF dùng để tính BA tiêu đề — KHÔNG có tập giữ lại (holdout) độc lập để chọn cặp/ngưỡng ensemble. Số "ensemble tốt nhất" được chọn bằng duyệt toàn bộ 10 cặp/10 bộ ba — đây là multiple-comparison, rủi ro overfit cao. Không được trích như một cải tiến đã xác nhận.

BA từng mô hình đơn (script `ensemble.py`, cùng 1100 clip, đối chiếu y_true khớp giữa các file):
```
B1=84.17%  B1w=84.37%  B1xl=82.71%  ESTA=83.83%  TSF=84.08%
```
Tỉ lệ bất đồng theo cặp (y_pred khác nhau): B1 vs {B1w,B1xl,TSF} chỉ 2.6–3.6% (các biến thể gần B1, cùng họ) nhưng B1 vs ESTA = 15.5%, B1w vs ESTA =13.6%, ESTA vs TSF=14.8% — ESTA là kiến trúc khác biệt nhất.

Trung bình xác suất (prob_unsafe) rồi ngưỡng 0.5:
```
avg(5 mô hình) = 84.50%   (chỉ +0.33pp so với B1 84.17%)
avg(B1+B1w+ESTA) = 84.96% (+0.79pp)
best cặp dò hết 10 cặp: avg(B1w+ESTA) = 86.63% (+2.46pp)   <-- cherry-pick hậu nghiệm, KHÔNG xác nhận được
```
=> Ensemble "an toàn" (trung bình cả 5, không chọn tay) chỉ cho **+0.33pp**. Cặp tốt nhất tìm được bằng duyệt vét hết (+2.46pp) rất có thể là nhiễu lấy mẫu (n=1100, 10 phép so sánh) — cần một tập eval độc lập mới để xác nhận trước khi tin.

Trần lý thuyết không khả thi (oracle per-clip, chọn đúng mô hình nào trong 5 mô hình đúng cho từng clip — không thể biết trước khi suy luận thật):
```
oracle-selection BA = 92.33% (sens 93.33%, spec 91.32%)
```
87/1100 clip (7.9%) SAI ở CẢ 5 mô hình (kể cả kiến trúc khác hẳn ESTA/TSF/VideoMAE B2 không trong tập 5 này nhưng cùng họ CLIP backbone) — đây là sàn lỗi "khó rút" bằng ensemble cùng họ đặc trưng. Gợi ý: muốn vượt sàn này cần đặc trưng/backbone khác hẳn (không chỉ đổi đầu), hoặc dữ liệu mới ở đúng các nguồn khó.

**Kết luận mục 4:** ensemble trung thực mang lại <1pp; số lớn hơn (+2.46pp hay trần 92.3%) đều có caveat nặng (chọn hậu nghiệm / không khả thi triển khai). Không nên đưa +2.46pp vào luận văn như một con số đã kiểm chứng.

## 5. Trần nhiễu nhãn

Nguồn: `report/vn/chuong-04-chi-tiet-thuc-nghiem.txt:147-177` + `plans/260613-thesis-defense-revision/reports/kappa-result.json` — ma trận đồng thuận 2 người gán nhãn, n=100 clip, taxonomy Unsafe-Net 4 lớp + nền (bg). Kappa=0.65, Po (5-way)=72%. **Luận văn tự khai (mục 4.2.4, điểm 3): đây KHÔNG phải phép đo trên nhãn nhị phân của iSafetyBench, không được suy chuyển.**

Script tính tay 2 cách (trong `analyze_b1.py`/console — xem lệnh dưới):

**Cách 1 — chuyển trực tiếp ngây thơ:** giả định kappa/Po 5-lớp áp dụng thẳng cho nhị phân → trần = 72%. Dưới trần này thì 84.2% là **vượt trần**, vô lý — chính vì vậy cách chuyển này sai/không dùng được, như luận văn đã cảnh báo.

**Cách 2 — gộp ma trận 5x5 về 2x2 (giả định: lớp 0-3 = "hazard", bg = "normal"; GIẢ ĐỊNH MAPPING, không có xác nhận từ luận văn rằng ranh giới hazard/normal của Unsafe-Net trùng ranh giới của iSafetyBench):**
```
             rater2=hazard  rater2=normal
rater1=hazard      75            5
rater1=normal      10            10
Po(gộp) = 85/100 = 85.0%     Pe = 0.71     kappa(gộp) = (0.85-0.71)/(1-0.71) = 0.483
```
Dưới cách gộp này, trần nhiễu nhãn ước lượng ≈ **85.0%** Po — CAO hơn 84.2% một chút (+0.8pp), và kappa gộp tụt về 0.48 (so với 0.65 ở 5-way) — do mất cân bằng lớp (80 hazard/20 normal trong mẫu 100).

**Đọc hai cách cùng lúc:** chênh nhau rất lớn (72% vs 85%) chỉ vì cách gộp lớp khác nhau → đây CHÍNH LÀ lý do luận văn cấm suy kappa giữa hai làn: phép gộp nhạy với giả định ranh giới lớp, không ổn định. Nhưng nếu buộc phải cho một ước lượng số (và chỉ nên dùng làm tín hiệu định hướng, không phải số chứng nhận): mô hình 84.2% **đã ở xấp xỉ hoặc vượt nhẹ** trần suy ra từ cách gộp hợp lý nhất (lớp hazard gộp = 0-3, normal = bg), tức dư địa từ phía "nhãn sạch hơn" cũng không lớn — vấn đề của 100 clip kappa test là mẫu nhỏ (n=100) và taxonomy khác hẳn (4+bg) nên số này chỉ mang tính chỉ dấu, không phải bằng chứng định lượng cho làn iSafetyBench.

## 6. Hạ tầng có thể chạy lại

| Thí nghiệm | Script | Dữ liệu sẵn có? | Chi phí ước tính |
|---|---|---|---|
| Train lại 1 đầu (B1/B1w/...) trên token đã trích | `run_feature_sweep.py`, `run_kfold_relation.py` | CÓ — token `.pt` đã cache trong `sweep_results/relation_tokens/` (shard theo model) | Vài phút/đầu/fold, CPU, theo tự khai mục 4.3 của luận văn |
| Trích lại đặc trưng CLIP cho iSafetyBench | `build-isafetybench-clip-features.py` | CÓ — **video thô có sẵn tại `data/isafetybench/` (1.2GB, 1818 .mp4, xác nhận bằng `du -sh` + `find`)**, không cần tải lại | Chục phút, CPU/MPS trên máy M3 hiện tại (theo tự khai luận văn, không đo lại) |
| Trích đặc trưng VideoMAE (B2) | `build-isafetybench-videomae-features.py` | Video có sẵn như trên | Tương tự, nặng hơn do backbone VideoMAE |
| ESTA theo shard (4 phân đoạn) | `run-esta-lso.py`, `tools/merge-esta-shards.py` | Shard `.pt` cache đã có (`shard_isbclip_ESTA_*of4.pt`) | Có thể chạy lại từng shard độc lập, phút-mức, CPU |
| Ensemble/threshold sweep (mục 3,4 ở trên) | tự viết (`analyze_b1.py`, `ensemble.py`) | Chỉ cần OOF JSON đã có | Giây, không cần GPU |
| Phân tích lỗi theo fold/nguồn (công cụ chính thức) | `tools/analyse-isb-fold-failures.py` | Cần `isafetybench_dataset.py` + cache đặc trưng đếm — CHƯA chạy thử trong phản biện này (không cần, đã tự xây lại độc lập và khớp 10/10 fold) | Phút-mức, CPU |
| Kappa gán nhãn lại (mở rộng mẫu > 100 clip, đo trên nhãn nhị phân) | `run_kappa_prepare.py`, `run_kappa_compute.py` | **KHÔNG** — cần 2 người gán nhãn thật, không tái tạo từ cache | Công sức con người, không phải compute |
| Chạy trên Kaggle/Colab cho mô hình nặng hơn (B2 VideoMAE, fine-tune backbone) | `colab/`, `kaggle-detector-retrain/` | Thư mục có sẵn script/notebook; cần tải video lên nền tảng ngoài nếu backbone cần GPU dài hơi | Theo giờ GPU miễn phí Kaggle/Colab, không re-download nếu cache `.pt` đủ dùng |

Phát hiện quan trọng: **video thô iSafetyBench đang nằm sẵn trên máy** (`data/isafetybench/`, không chỉ đặc trưng đã trích) — khác với giả định ban đầu trong đề bài "chỉ có đặc trưng, không có video". Việc này hạ chi phí tái tạo pipeline đầu-cuối xuống mức "chạy lại trên máy M3 hiện có", không cần hạ tầng ngoài.

## 7. Câu hỏi chưa giải đáp

1. `tools/analyse-isb-fold-failures.py` (công cụ CHÍNH THỨC được luận văn trích làm nguồn cho bảng per-fold) chưa được chạy thực tế trong phản biện này — R5 tự xây lại fold map độc lập từ Phụ lục B và khớp 10/10, nhưng không chạy script gốc để xác nhận `isafetybench_dataset.py::folds()` sinh đúng cùng một phép chia `StratifiedGroupKFold(seed=0)` như Phụ lục B in ra bằng tay. Nên chạy 1 lần để chốt.
2. Mapping "lớp 0-3 Unsafe-Net = hazard, bg = normal" dùng ở mục 5 (gộp ma trận kappa) là GIẢ ĐỊNH của R5, không có xác nhận văn bản rằng ranh giới 4 lớp Unsafe-Net ánh xạ 1:1 sang ranh giới hazard/normal của iSafetyBench (2 làn dữ liệu khác nhau, taxonomy khác nhau). Cần hỏi tác giả/GVHD liệu phép gộp này có ý nghĩa gì không, hay chỉ nên bỏ hẳn mục 5 khỏi luận văn như đã làm.
3. Cặp ensemble tốt nhất hậu nghiệm (B1w+ESTA, 86.63%, +2.46pp) có đáng chạy lại trên một tập eval mới (không phải 1100 clip đã dùng để chọn) để kiểm tra có phải nhiễu lấy mẫu không? Nếu hội đồng muốn một con số ensemble "sạch", cần giao thức tiền đăng ký (chọn cặp trước khi nhìn OOF mới).
4. ECE 11.4pp dựa trên bin rất mỏng (nhiều bin giữa <20 điểm) — có cần bootstrap CI cho ECE trước khi trích dẫn, hay nên bỏ số ECE này và chỉ giữ phần "band 0.4-0.6 chiếm 1.7% lỗi" (đo trực tiếp, không cần bin hoá)?
