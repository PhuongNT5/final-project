# BÁO CÁO KẾ HOẠCH THỰC HIỆN TIẾP THEO — Luận văn Thạc sĩ

**Kính gửi:** GVHD — [Thầy/Cô hướng dẫn]
**Học viên:** Nguyễn Thị Phương — MSHV: [……] — Lớp: [……]
**Chương trình:** Thạc sĩ Khoa học Máy tính
**Đề tài:** Phát hiện hành vi không an toàn thời gian thực trong giám sát công nghiệp dựa trên YOLO và Transformer
**Ngày:** 03/07/2026 · *(tiếp nối báo cáo tiến độ 29/06/2026)*

---

## 1. Bối cảnh & quyết định hướng

Sau khi bổ sung đánh giá nghiêm ngặt (k-fold, blind held-out) và chẩn đoán gốc rễ, kết luận đã rõ: **trần độ chính xác bị chặn bởi THIẾU GIÁM SÁT VÙNG/ĐỐI TƯỢNG theo từng khung hình, không phải do mô hình yếu** (VideoMAE-86M cũng đụng cùng trần ≈55/64). Paper đối chứng đạt 95.81% nhờ **2.262 bbox/khung** mã hoá vị trí đối tượng theo lớp hành vi.

**Điều chỉnh mục tiêu MT2 (mới, 03/07):** **KHÔNG đặt cam kết cứng ">95.8%"** như một con số duy nhất phải đạt. Thay vào đó, luận văn (a) **nâng accuracy** bằng tuyến dữ liệu paper-scale (~3.000 bbox) + kiến trúc token mới, và (b) **đánh giá ĐA GÓC NHÌN — nhiều metric, nhiều công thức, nhiều lăng kính** (an toàn, thống kê, robustness, hiệu quả edge, diễn giải) làm **đóng góp đánh giá chính**. Lý do: trần accuracy là data-bound; một con số đơn lẻ trên N=64 dễ bị công kích, còn bộ đánh giá đa góc nhìn cho thấy *kiến trúc thắng ở đâu* (FNR an toàn, accuracy/Watt, tính diễn giải) là đóng góp vững và trung thực hơn.

| # | Ưu tiên | Nhóm mục tiêu | Là đòn bẩy cho |
|---|---|---|---|
| **P1** | Gắn nhãn bbox ~3.000 khung (tập trung, chính xác) | Dữ liệu | **MT2** (nâng accuracy — không chốt 1 số) |
| **P2** | Kiến trúc mới **YOLO26 + Object-Token/Coordinate-Embedding + Transformer** | Kiến trúc | **MT1 + MT3** |
| **P3** | Tăng cường & củng cố **Test set** (đúng phương pháp) | Rigor | Độ tin cậy đánh giá |
| **P4** | **Đánh giá đa góc nhìn**: đúng công thức paper + mở rộng metric + Jetson | Rigor + MT2 mới | Đóng góp đánh giá |

---

## 2. P1 — Gắn nhãn bbox ~3.000 khung *(ưu tiên số 1)*

**Mục tiêu:** cấp cho detector tín hiệu *vị trí đối tượng theo từng khung theo lớp hành vi* — đúng thứ frozen-COCO features không có — để phá cặp nhầm lẫn WalkwayViolation ↔ UnauthorizedIntervention (đang gánh gần như toàn bộ lỗi, FNR Unauth 68%).

### 2.1 Vì sao 3.000 khung mới đủ (bài học từ V10)
- V10 gắn 400 khung → **regressed 54/64**: quá ít, mất COCO prior nhiều hơn tín hiệu thu được.
- Paper dùng **2.262 khung train** (+329 test) → 95.81%. ⇒ **~3.000 khung là quy mô paper+**, đủ để tín hiệu hành vi vượt chi phí mất prior.

### 2.2 Phân bổ khung (tập trung vào lớp nhầm & lớp hiếm)
| Lớp | Số khung mục tiêu | Lý do |
|---|---:|---|
| WalkwayViol (0) | ~600 | lớp nhầm chính |
| UnauthorInt (1) | ~600 | lớp nhầm chính (FNR cao nhất) |
| OpenedPanel (2) | ~350 | |
| CarryOverload (3) | ~350 | hiếm (n=48 clip) → nhiều khung/clip |
| SafeWalkway (4) | ~300 | phân biệt với lớp 0 |
| AuthorizedInterv (5) | ~300 | phân biệt với lớp 1 |
| ClosedPanel (6) | ~250 | |
| SafeCarrying (7) | ~250 | |
| **Tổng** | **~3.000** | |

### 2.3 Quy trình & công cụ (tái dùng asset đã có)
1. **Lấy mẫu khung có mục tiêu:** ưu tiên khung *thấy rõ hành vi*; nhiều khung hơn cho lớp nhầm/hiếm; trải đều 6–8 góc camera.
2. **Auto-draft giảm công:** dùng `best.pt` (mAP50 ~0.89) + **SAM2-assisted** (`auto_annotate.py` đã có) sinh nhãn nháp → người sửa (không vẽ từ đầu).
3. **Công cụ:** CVAT (đã có `annotation_guide/cvat_export_yolo/`), xuất YOLO format.
4. **Chất lượng:** đo **inter-annotator kappa** (đã có `run_kappa_prepare.py` / `run_kappa_compute.py`) trên tập con; hướng dẫn nhãn ở `annotation_guide/LABELING-GUIDE.md`.
5. **(Tuỳ chọn, cộng hưởng)** gắn kèm **nhãn vùng** (walkway/danger polygon đã dựng ở `tools/apply_zone_prior.py`) để bổ trợ tín hiệu vùng.

**Ước lượng công:** ~30–40 giờ gắn nhãn có kỹ năng (nhờ auto-draft), so paper ~190h vẽ tay thuần.

---

## 3. P2 — Thực nghiệm kiến trúc mới YOLO26

**Trạng thái YOLO26 (đã xác minh):** là bản phát hành thật của Ultralytics (09/2025), **DFL-free, NMS-free end-to-end, INT8 ổn định hơn**, hỗ trợ tracking sẵn (ByteTrack/BoT-SORT). Jetson Orin Nano: YOLO26n TRT-INT8 **3.8 ms**. Giấy phép **AGPL-3.0** (luận văn OK; lưu ý nếu thương mại hoá).

### 3.1 Kế hoạch
1. **Train lại detector YOLO26** trên tập 3.000 bbox (P1) — lịch **low-LR, đóng băng backbone 30 epoch đầu** rồi mở 2 block sâu nhất để **giữ COCO prior** (tránh regress như V10).
2. **Dựng lại pipeline:** YOLO26 (backbone/neck) → trích feature layer (**hook P3/P4/P5**; lưu ý index layer YOLO26 ≠ YOLO11, phải in `model.model.model` để verify) → **Transformer head** phân loại hành vi.
3. **Tracking + ID đối tượng (ByteTrack `persist=True`)** → *(hướng nâng cao MT1)* token **object-centric theo track**: RoI-Align đặc trưng mỗi đối tượng + toạ độ chân + nhãn vùng → thay pooling `mean/std/max` (bước phá vị trí hiện nay).
4. **So sánh** YOLO11s vs YOLO26s (accuracy, latency, INT8) — bảng đối chứng.
5. **Ghi log thực nghiệm bằng CometML** (xem P4).

> **Ghi chú trung thực:** bản thân việc đổi YOLO11→YOLO26 **không** tự nâng accuracy; độ chính xác đến từ **train lại trên 3.000 bbox hành vi**. YOLO26 đóng góp: INT8 sạch, latency thấp (MT3), tracking sẵn cho token object-centric (MT1).

### 3.2 Kiến trúc token — Object-Token + Coordinate Embedding *(cơ chế MT1)*

Bóc tách bài toán thành **nhận diện thực thể** + **để Transformer suy luận không gian–thời gian**. Thay `mean/std/max` pooling (bước phá vị trí hiện nay) bằng **token theo từng đối tượng có gắn toạ độ**:

```
Mỗi khung t (T=32):
  YOLO26 detect → N thực thể {Person, Panel, Forklift, ...}
  Mỗi thực thể i → token_i =
       Linear([  RoI-Align(feature_map, box_i)        # đặc trưng HÌNH ẢNH CỤC BỘ của thực thể
                ⊕ [x_c, y_c, w, h] (chuẩn hoá /W,/H)  # Coordinate Embedding
                ⊕ class_embed(loại thực thể)          # Person/Panel/Forklift...
                ⊕ zone_feat(foot_i)                   # vùng: in/out walkway, in danger, distance
                ⊕ temporal_PE(t) ]) → d=384
Chuỗi (T × N) token + padding mask (N thay đổi) → Transformer Encoder (Self-Attention)
  Self-Attention học trọng số tương quan:
     Person + gần Machine_Area + ngoài walkway  ⇒ xác suất cao UnauthorizedIntervention
     Person + trong walkway                     ⇒ WalkwayViolation
→ [CLS] → MLP → lớp hành vi
```

**Vì sao giải quyết được nhầm lẫn:** hai lớp WalkwayViol/UnauthorInt *nhìn giống nhau* nhưng khác **quan hệ không gian giữa người và vùng/máy**. Self-Attention giữa token Person và token Panel/Zone tính đúng quan hệ đó — điều mà đặc trưng toàn ảnh (pooled) không biểu diễn được. Attention lúc này *có việc thật để làm* (trước đây Transformer ≈ linear probe vì token đồng nhất). Tiền lệ: ORViT (CVPR'22), Long-term Feature Banks (CVPR'19), DETR-style coordinate tokens — **chưa công trình nào áp cho Unsafe-Net** ⇒ đóng góp MT1 defensible.

**Tinh chỉnh & lưu ý trung thực (để chắc chắn hoạt động):**
1. **RoI-Align đặc trưng CỤC BỘ per-box**, không dùng 1 global feature — nếu concat toạ độ vào global feature thì vẫn là pooling cũ, mất tính object-centric.
2. **"Zone/Machine_Area" nên là polygon TĨNH per-camera** (đã dựng ở `tools/apply_zone_prior.py`) → mỗi token Person mang `zone_feat`; robust hơn bắt YOLO detect "Zone" mỗi khung (vùng không di chuyển → detect per-frame vừa nhiễu vừa phí). Nếu vẫn muốn gắn nhãn Zone như 1 lớp bbox (đúng cách paper) thì cũng dùng được — quyết định ở P1.
3. **Relative position** (khoảng cách cặp box Person↔Panel) giúp attention hiểu "gần" tốt hơn toạ độ tuyệt đối — nâng cao, tuỳ chọn.
4. **Toàn bộ kiến trúc phụ thuộc chất lượng detector**: đây là lý do P1 (3.000 bbox) là ưu tiên #1. V6-det từng hỏng vì person-detection chỉ 27%; nay COCO đạt 88.7% + train lại YOLO26 → kỳ vọng đủ tốt.
5. **N thực thể thay đổi/khung** → padding + attention mask; giới hạn top-K box theo confidence.

---

## 4. P3 — Tăng cường & củng cố Test set *(đúng phương pháp)*

Có **ba việc khác nhau**, phải tách bạch (nếu không sẽ bị HĐ bắt lỗi nguỵ tạo N):

| Cơ chế | Được phép claim | KHÔNG được |
|---|---|---|
| **(a) k-fold / xoay vòng** (đã có `run_kfold_eval.py`) | tăng N hiệu dụng *để chọn model*, siết CI; dùng **Nadeau–Bengio corrected t-test** | dùng 64-clip test làm fold rồi so với paper |
| **(b) TTA** (augment lúc suy luận, trung bình logit: flip, temporal-crop, multi-scale) | accuracy/robustness khá hơn chút | tăng số mẫu độc lập; **phải áp cho CẢ baseline** |
| **(c) Corruption robustness** (augment test: noise/blur/nén) | **metric riêng** (mCE theo severity) | biến 64→640 rồi báo là generalization |

**Nguyên tắc bất di:** **64-clip test giữ blind, chỉ chạm 1 lần** để so paper; k-fold chạy trên 691 clip để *chọn model*. Đây chính là điều đã cứu luận văn khỏi con "57/64 calibrated" bị thu hồi.

**(Vá điểm yếu 1-dataset — tuỳ chọn):** đánh giá chéo trên **iSafetyBench** (ICCV'25 workshop, public, 1.100 clip) dạng binary safe/unsafe.

---

## 5. P4 — Đánh giá ĐA GÓC NHÌN *(đóng góp MT2 mới)*

> **Không chốt một con số accuracy.** Trình bày kết quả qua **nhiều metric, nhiều công thức, nhiều lăng kính** để cho thấy kiến trúc *thắng ở đâu* và *giới hạn ở đâu* — trung thực và phòng thủ được trước HĐ.

### 5.1 Theo ĐÚNG paper (để so công bằng)
| Metric paper | Giá trị paper | Ta replicate |
|---|---|---|
| Classification accuracy (macro, 4-class) | **95.81%** (62/64) | macro-acc = mean per-class recall |
| No-bbox baseline | 89.06% (57/64) | mốc phải vượt |
| **Action-recognition time / clip** | **0.14 s** | đo `time.perf_counter` /clip |
| Reduced avg video duration | 1.87 s | tương tự (frame-gate) |

### 5.2 Bộ metric đa lăng kính (phân loại)
| Lăng kính | Metric / công thức |
|---|---|
| **Độ chính xác tổng hợp** | top-1, **macro-F1**, balanced-accuracy, per-class precision/recall, **Cohen's κ**, **Matthews MCC**, confusion matrix |
| **An toàn** (quan trọng nhất) | **FNR / miss-rate per-class**; **cost-sensitive** (phạt bỏ sót nặng hơn báo nhầm) |
| **Ngưỡng / operating point** | **ROC-AUC, PR-AUC**, threshold sweep → chọn điểm vận hành theo recall an toàn |
| **Hiệu chuẩn (calibration)** | **ECE** + reliability diagram (độ tin cậy dự đoán) |
| **Thống kê** | bootstrap 5.000 + **Wilson/BCa CI**, **McNemar exact** (2 model / 64 clip), **Nadeau–Bengio** (k-fold) |

### 5.3 Metric Jetson (MT3, mở rộng)
latency **p50/p95/p99** (ms **/frame** và **/clip**), FPS, **multi-stream agg + zero-drop**, model size/params/FLOPs, INT8 compression, **năng lượng: J/inference & FPS/W** (đo bằng `jtop`). Mở rộng `deployment/scripts/benchmark-harness.py`.

### 5.4 Lăng kính hiệu quả & diễn giải *(góc nhìn mới, giá trị cao)*
- **Pareto edge:** **accuracy/Watt, accuracy/param, accuracy/ms** — góc nhìn "hệ thống", nơi mô hình nhỏ chạy biên có lợi thế rõ so với pipeline nặng của paper.
- **Diễn giải (interpretability):** trực quan **attention map** — token đối tượng nào Transformer chú ý → bằng chứng trực tiếp cơ chế *Person ↔ Machine/Zone* hoạt động (rất đắt giá để bảo vệ MT1).
- **Generalization:** cross-dataset **iSafetyBench** (binary safe/unsafe) + k-fold scene-robust.

### 5.5 Hạ tầng log: CometML
Log metrics/params/**confusion-matrix**/artifact tập trung cho cả ~100+ run. ⚠️ đặt `COMET_EVAL_LOG_CONFUSION_MATRIX=true` (mặc định OFF); dùng **offline mode** trên Jetson.

---

## 6. Tiến độ dự kiến

| Tuần | Công việc chính | Sản phẩm |
|---|---|---|
| 1 | Setup CometML (P4.4) + eval harness đúng-paper (P4.1–2) + khởi động gắn nhãn (P1) | log tập trung; bảng metric chuẩn; ~800 khung |
| 2–3 | Hoàn tất **3.000 bbox** (P1) + đo kappa | tập bbox paper-scale + báo cáo chất lượng nhãn |
| 3–4 | Train lại **YOLO26** trên bbox + dựng pipeline + Transformer head (P2.1–2) | detector mới; feature layer; số accuracy sơ bộ |
| 4–5 | Object-centric token + tracking (P2.3) + blind eval + k-fold (P3a) | số MT2 trung thực + CI |
| 5 | TTA + robustness (P3b,c) + metric Jetson đầy đủ (P4.3) | bảng đánh giá hoàn chỉnh |
| 6 | Viết chương 3–5 (kiến trúc + kết quả + triển khai) | bản luận văn cập nhật |

---

## 7. Ánh xạ 3 mục tiêu

| MT | Sau kế hoạch này |
|---|---|
| **MT1** (kiến trúc mới) | YOLO26 + **Object-Token + Coordinate-Embedding + Transformer** (self-attention Person↔Zone/Machine) thay pooling — cơ chế mới, defensible; có trực quan attention chứng minh cơ chế |
| **MT2** (đánh giá đa góc nhìn) | nâng accuracy bằng 3.000 bbox + object-token; **KHÔNG chốt >95.8%** — đóng góp là **bộ đánh giá đa metric/công thức/lăng kính** (an toàn, thống kê, robustness, Pareto edge, diễn giải) + so đúng công thức paper |
| **MT3** (edge real-time) | YOLO26 INT8 (3.8 ms Jetson) + metric năng lượng/đa luồng mở rộng |

---

## 8. Rủi ro & phương án dự phòng

| Rủi ro | Mức | Dự phòng |
|---|---|---|
| 3.000 bbox vẫn regress (mất COCO prior) | Vừa | lịch low-LR + đóng băng backbone 30 epoch; tăng dần (1.000→2.000→3.000) để đo ngưỡng đảo chiều |
| Công gắn nhãn vượt thời gian | Cao | auto-draft `best.pt`+SAM2; ưu tiên 2 lớp nhầm trước (1.200 khung) rồi mở rộng |
| YOLO26 layer index / API khác | Thấp | verify `model.model.model`; test hook trên 1 batch trước |
| N=64 test vẫn nhỏ | Vừa | k-fold (691) để chọn model + CI; giữ 64 blind cho số cuối |
| AGPL-3.0 | Thấp | luận văn OK; ghi rõ ràng buộc nếu chuyển giao |

**Cổng Go/No-Go:** train thử ở **1.000 khung** trước — nếu blind eval **không vượt baseline trung thực 53–54/64** thì rà lại chất lượng nhãn/lịch train **trước khi** gắn tiếp tới 3.000 (tránh lãng phí công như V10). *Do MT2 đã bỏ cam kết cứng >95.8%, tiêu chí thành công là: (a) vượt baseline trung thực + (b) thắng rõ ở bộ đánh giá đa góc nhìn (FNR an toàn, accuracy/Watt, diễn giải) — không phụ thuộc một con số accuracy.*

---

## Câu hỏi chưa được giải đáp
1. **Deadline nộp/bảo vệ** chính xác — để chốt có làm đủ 6 tuần hay rút gọn (bỏ iSafetyBench + object-centric nếu gấp)?
2. Có **người hỗ trợ gắn nhãn** thứ 2 không (để đo kappa + chia tải 3.000 khung)?
3. Được cấp **Springer full-text** để lấy đúng công thức macro-accuracy + confusion matrix của paper (P4.1)?
4. Nhãn bbox theo **8 lớp hành vi** (như paper) hay gộp **person + zone** (nhẹ hơn)? — ảnh hưởng công gắn nhãn và cách detector mã hoá vùng.
5. **CometML** dùng gói academic (miễn phí Pro) hay tự-host offline?
