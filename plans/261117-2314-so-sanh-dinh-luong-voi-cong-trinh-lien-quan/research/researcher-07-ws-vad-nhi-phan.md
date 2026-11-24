# Researcher 07 — WS-VAD: SOTA, tham số, giao thức, khả năng quy đổi

Quy ước: **[V]** = đã đọc nguồn gốc và đối chiếu bảng. **[S]** = chỉ snippet/trang tổng hợp, chưa xác minh. KHÔNG TÌM THẤY = đã tìm, không có.

## (A) SOTA 2018–2026 (AUC = frame-level UCF-Crime; AP = XD-Violence)

| Method, năm | Feature | UCF AUC | XD AP | Nguồn, vị trí | Ghi chú |
|---|---|---|---|---|---|
| Sultani 2018 (1801.04264) | C3D | 75.41 [V] | n/a | Table 3, §5.2 | Bản gốc. Xem (D) |
| RTFM ICCV 2021 (2101.10030) | I3D | 84.30 / 85.66 [V] | 77.81 / 78.27 [V] | UR-DMU T1/T2; VadCLIP T2/T1 | **Hai bài ghi số khác nhau** |
| MSL AAAI 2022 (DOI 10.1609/aaai.v36i2.20028) | I3D / VideoSwin | 85.30 / 85.62 [V] | 78.59 (RGB) [V] | UR-DMU T1/T2 | **arXiv 2104.04527 trong yêu cầu là SAI** — đó là bài vật lý (PrV2Al20) |
| UR-DMU AAAI 2023 (2302.05160) | I3D / RGB | 86.97 [V] | 81.66 [V] | UR-DMU T1/T2 | VadCLIP T2 ghi DMU 86.75 |
| MGFN AAAI 2023 (2211.15098) | I3D / VideoSwin | 86.98 / 86.67 [V] | 79.19 / 80.11 [V] | Table 1/2 | |
| VadCLIP AAAI 2024 (2308.11681) | CLIP ViT-B/16 | 88.02 [V] | 84.51 [V] | Table 2 / Table 1 | **Text ghi 82.08 — mâu thuẫn trong chính bài** |
| Holmes-VAD (2406.12235) | MLLM (Video-LLaVA, instruction-tuned) | 89.51 [V] | 90.67 [V] | Table 1 | Không phải WS-VAD cổ điển |
| LAVAD 2024 (2404.01014) | VLM, training-free | 80.28 [V] | 62.01 [V] | Table 1/2 | |
| VERA CVPR 2025 (2412.01095) | VLM | 86.55 [S] | n/a | — | Chưa mở bài gốc |
| VadCLIP++ (DSP 2025, DOI 10.1016/j.dsp.2025.105560) | CLIP | 88.12 [S] | 85.03 [S] | trang tổng hợp | Chưa mở bài gốc |
| ASK-Hint 2025 (2510.02155) | Qwen2.5-VL, training-free | 89.83 [V] | **90.31 là AUC không phải AP** [V] | Table 1/2 | Frame/video không rõ |
| TrCLIP-VAD (Neural Networks 2026) | CLIP | 88.59 [S] | 86.05 [S] | ScienceDirect | Chưa mở bài gốc |
| DeSC ICME 2026 (2603.19780) | **CLIP ViT-B/16 frozen** | 89.37 [V] | 87.18 [V] | Table I | |
| LAS-VAD CVPR 2026 (2603.00550) | CLIP ViT-B/16 (I3D: 91.05) | **90.86** [V] | 87.92 [V] | Table 2 / Table 1 | Text ghi 90.96 |
| CPL-VAD 2026 (2602.17077) | **CLIP ViT-B/16 frozen** | 88.24 [V] | 88.53 [V] | Table 1 | |
| MemoVAD 2026 (2606.07669) | VideoMAE-S frozen + cloud VLM | 89.45 [V] | n/a | Table 1 | **Có đo Jetson** — xem (C) |
| SES 2026 (2607.10298) | I3D | 88.46 [V] | n/a | Table 1 | |
| Adaptive Multi-Granularity 2026 (2609.05066) | I3D | 87.24 [V] | 83.89 [V] | Table 1/2 | |

Không tìm thấy / không xác minh:
- **Hawk (2405.16886): KHÔNG TÌM THẤY AUC/AP** — bảng chỉ có BLEU và điểm mô tả (Table 1, 3).
- GCTG-CLIP 89.07: KHÔNG TÌM THẤY.
- LATERN (2605.15054): chưa trích được số.

Cao hơn VadCLIP (88.02), đã xác minh: LAS-VAD 90.86 (CLIP), ASK-Hint 89.83 (training-free), Holmes-VAD 89.51 (MLLM), MemoVAD 89.45, DeSC 89.37 (CLIP frozen). Holmes-VAD / ASK-Hint / MemoVAD **không** thuộc dạng "CLIP đóng băng + head nhẹ".

## (B) Số tham số huấn luyện được

**Không bài nào đã đọc nêu tổng số tham số huấn luyện được.**

- VadCLIP [V]: CLIP ViT-B/16 đóng băng ("weights … kept fixed"); phần huấn luyện = LGT-Adapter + FC + C-branch + prompt 20 token. Không có tổng.
- DeSC [V]: CLIP đóng băng; chỉ nêu số lớp (7 TCN + 1 graph transformer; 1 transformer + 3 GCN). Không tổng.
- CPL-VAD [V]: image và text encoder đều đóng băng. Không nêu.
- Holmes-VAD [V]: LanguageBind đóng băng; LoRA r=64; không nêu.
- ASK-Hint [V]: VLM đóng băng, training-free.
- SES [V]: nêu "9.4MB parameters", không nêu số tham số.
- Hawk [V]: BLIP-2 đóng băng, không nêu.
- UR-DMU, MGFN, LAS-VAD, MemoVAD, 2609.05066: không nêu.

→ Luận văn (201,090 tham số) **cùng cỡ thiết kế** với họ CLIP-frozen này, nhưng **không có số nào để đối chiếu**.
→ **Yếu tố gây nhiễu: WS-VAD dùng ViT-B/16, luận văn dùng B/32.**

## (C) Ba câu hỏi phụ

**(i) CI / độ lệch chuẩn qua nhiều seed: KHÔNG TÌM THẤY** trong mọi bảng đã đọc (VadCLIP, UR-DMU, MGFN, LAS-VAD, DeSC, CPL-VAD, MemoVAD, SES, 2609.05066, ASK-Hint). Tất cả là **giá trị đơn**. Gần nhất: 2608.11985 dùng "paired video bootstrap" trên dự đoán có sẵn (abstract [V]; số resample không nêu) — là CI trên mẫu test, không phải phương sai theo seed. RTFM trên ShanghaiTech lấy trung bình 10 lần chia ngẫu nhiên [S].

→ **Luận văn báo cluster-bootstrap CI + 3 seed là chặt hơn toàn bộ bảng SOTA này.**

**(ii) Triển khai biên có số đo thật: đúng một bài** [V]. MemoVAD (2606.07669, Table 2): Jetson AGX Orin, streaming batch=1, precision không nêu — MemoVAD 36.2 FPS / 0.475 s; baseline CLIP-based của chính họ 11.5 FPS / 1.512 s. MemoVAD không dùng CLIP. RTFM trên Jetson 47.56 FPS (TensorRT) chỉ [S]. LAS-VAD không có số độ trễ [V]. VadCLIP / DeSC / CPL-VAD chưa kiểm mục này.

**(iii) Phép chia ràng buộc theo camera/bối cảnh: KHÔNG TÌM THẤY** trong các benchmark WS-VAD chính. Split UCF-Crime đang dùng là split chính thức 1,610 train / 290 test [V, 2609.05066]. Không bài nào định nghĩa split disjoint theo camera. Ba thứ liên quan: (1) 2608.11985 phát hiện scene bias nhưng không đề xuất split; (2) 2007.07843 đề xuất split theo cảnh mới, nhưng là one-class không phải WS [V, abstract]; (3) đánh giá chéo ShanghaiTech↔UCF-Crime trong 2512.06845 [S].

→ **Toàn bộ mảng WS-VAD chưa có giao thức ràng buộc nhóm.**

## (D) Phê bình giao thức — có, và trực tiếp

- **2608.21854** (Song & Lee, "Look Inside Each Video", v1 22/08/2026, v2 26/09/2026) [V]: Micro-AUROC chủ yếu đo **so sánh giữa các video**, không đo định vị trong video. Cặp trong-video chỉ chiếm **0.071–0.388%** tổng cặp anomaly–normal. Video-level classifier đạt **81.40–97.18% Micro-AUROC nhưng Within-AUROC = 50%**. Thay điểm frame bằng trung bình theo video → Within-AUROC = 50% nhưng vẫn giữ median **98.8%** biên Micro-AUROC (120 lần chạy). Dùng 4 benchmark không nêu tên trong trang đã đọc. **Preprint, chưa peer review.**
- **2608.11985** (Abdulaziz & Bondarev, v3 09/2026, ECCV 2026 ET workshop) [V]: AUC pooled **không đủ độ phân giải** ở cỡ test split để hỗ trợ các biên SOTA đã báo cáo. Mọi mô hình đều tách video theo **độ phân giải và color encoding** trên normal footage. Abstract không có số cụ thể; Bảng 2 và §5.2 không đọc được.

**Mâu thuẫn số liệu giữa các bài (đã đối chiếu bảng):**
- Sultani: 75.41 (bản gốc Table 3) vs 84.14 (VadCLIP T2) [V] — không rõ 84.14 là chạy lại với feature khác.
- RTFM: 84.30 (UR-DMU T1) vs 85.66 (VadCLIP T2); XD 77.81 vs 78.27 [V].
- Holmes-VAD: 89.51/90.67 (gốc T1) vs 84.61/84.96 (DeSC T1) [V] — khác setting.
- VadCLIP: text 82.08 vs Table 2 88.02, **trong cùng bản arXiv** [V].
- LAS-VAD: text 90.96 vs Table 2 90.86 [V].
- Metric không nhất quán: ASK-Hint báo AUC trên XD thay vì AP; đơn vị frame/video không rõ.

Nhãn nhiễu: tái chú thích test UCF-Crime tăng tỉ lệ frame bất thường **7.58% → 16.55%** (2408.05191) [S, không đọc được HTML].

**Không tìm thấy reproduction study nào** của VadCLIP hay bài CLIP-based khác.

## (E) Quy đổi AUC frame-level → balanced accuracy clip-level

**KẾT LUẬN: KHÔNG quy đổi được.** Lý do kỹ thuật:
1. **Khác đơn vị.** AUC frame pool qua video → video dài có trọng số lớn; đo chủ yếu so sánh giữa video (xem 2608.21854). BA clip đo một điểm ngưỡng trên clip.
2. **AUC là tích phân toàn đường ROC; BA là một điểm** trên đường ROC tại ngưỡng đã chọn. Không có hàm đơn ánh.
3. **Chỉ có chặn một chiều** (hình học ROC): với J* = max_t(TPR−FPR), ROC nằm dưới y = min(x+J*, 1) → AUC ≤ ½ + J* − J*²/2, suy ra max_t BA ≥ 1 − ½√(2−2·AUC). Ví dụ AUC = 0.88 → max BA ≥ 0.755; chặn trên là 1. **Khoảng quá rộng để so sánh**, và chặn này chỉ áp dụng khi cùng tập điểm cùng đơn vị — không áp dụng được ở đây.
4. **Phân bố lớp.** AUC và BA (ngưỡng cố định) không phụ thuộc prior; PR-AUC/AP phụ thuộc prevalence → XD-Violence AP không so được với PR-AUC clip của luận văn nếu không báo prevalence hai bên.
5. **Tác vụ khác.** UCF-Crime: 13 loại bất thường, video dài, nhãn frame. XD-Violence: 6 loại bạo lực + normal. Luận văn: nhị phân, 1,100 clip. **Cùng tên "AUC" không đồng nghĩa.**

**Để đặt hai số cạnh nhau hợp lệ, cần:** điểm vận hành (ngưỡng chọn trên fold train, TPR/FPR tại ngưỡng, ROC đầy đủ); prevalence từng fold; luật tổng hợp frame→clip; và **chạy cùng protocol**: VadCLIP (code chính thức, cả B/16 và B/32) trên iSafetyBench leave-source-out cùng fold, báo cluster-bootstrap CI theo nguồn + nhiều seed. **Hoặc** chạy pipeline luận văn trên UCF-Crime test (290 video) bằng sliding window, báo frame-level AUC và Within-AUROC theo 2608.21854. Vẫn phải kiểm soát B/16 vs B/32.

**Khuyến nghị: KHÔNG đặt số luận văn vào cùng bảng với WS-VAD.** Nếu cần, chỉ ghi "tham khảo, khác tác vụ và khác đơn vị".

## Câu hỏi chưa giải quyết
1. VERA, VadCLIP++, TrCLIP-VAD, RTFM-Jetson, 2408.05191 chỉ [S] — cần PDF.
2. Tổng tham số huấn luyện được của VadCLIP và LAS-VAD: cần PDF đầy đủ hoặc code.
3. LATERN: chưa trích được số.
4. Số resample và CI trong 2608.11985 / 2608.21854: cần bản đầy đủ.
5. Chi tiết split 2007.07843 và 2512.06845: chưa đọc.
6. arXiv ID của MSL cần xác nhận lại (2104.04527 là bài khác).
7. Sultani 84.14 trong VadCLIP T2: nguồn gốc chưa rõ.
