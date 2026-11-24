# Researcher 03 — hiệu năng mô hình video & triển khai biên (Jetson)

PDF gốc (I3D, SlowFast, X3D, VideoMAE, X-CLIP, ViViT, CLIP, ConvLSTM) không parse được → số lấy từ model zoo chính thức hoặc bài thứ cấp, ghi rõ từng dòng.

## Kết luận nhanh
- **Không có dòng nào đo trực tiếp được** cho mô hình video trên Orin Nano Super. Không tìm thấy throughput TensorRT FP16 hay năng lượng cho I3D, SlowFast, TimeSformer, ViViT, VideoMAE, X-CLIP, EVL, MoViNet, X3D trên Jetson Orin.
- Phép so duy nhất làm được: **tham số và GFLOPs lý thuyết** của CLIP ViT-B/32. Hệ thống luận văn: 16 khung × CLIP ViT-B/32 ≈ **70.6 GMACs ≈ 141 GFLOPs** mỗi quyết định [suy ra].
- **Cảnh báo quy ước "GFLOPs":** nhiều bài đếm GMACs nhưng ghi nhãn GFLOPs. Phải chuẩn hóa trước khi so.
- **Dải 25–60 FPS cho YOLO trên Jetson chưa xác nhận là FPS mô hình đơn.** YOLOv8n TensorRT FP16 trên Orin Nano Super đo **225 FPS** (engine). 58.8 FPS là thông lượng **tổng hợp 4 luồng 4K**, tính cả pipeline.
- **Khoảng trống "decision rate" được xác nhận:** không công trình nào báo decisions/s cho giám sát video cửa sổ trượt trên Jetson.

## Quy ước đơn vị (quan trọng trước khi trích)
- X-CLIP B/32: 8×224 = 39 G, 16×224 = 75 G. CLIP ViT-B/32 ảnh = 4.41 GMACs; 16 khung ≈ 70.6 GMACs → khớp 75 G → **X-CLIP ghi GMACs dưới nhãn GFLOPs** [suy ra].
- EVL: 8×3 views = 444 G → mỗi view ≈ 148 G ≈ 8 × 17.6 GMACs ViT-B/16 → khớp convention MAC [suy ra]. (17.6 từ kiến thức nền [chưa xác minh].)
- open_clip `image_gflops` = 8.82 cho ViT-B-32 = 2 × 4.41 GMACs = **FLOPs thực** [suy ra].
- X3D: zoo ghi 0.60 G (XS); MoViNet Table 9 ghi 23.3 G cho cùng model. Tỷ lệ ~39–40×, convention chưa rõ.

## (A) Bảng mô hình
Verdict: (a) so trực tiếp được; (b) cùng phần cứng khác mô hình; (c) cùng mô hình khác phần cứng; (d) khác đơn vị/khác cả hai → không quy đổi. **(a\*)** = GFLOPs lý thuyết, so được sau chuẩn hóa MAC.

| Mô hình | Params (M) | GFLOPs/clip (views) | Throughput / GPU | Top-1 K400 | Nguồn | Verdict |
|---|---|---|---|---|---|---|
| ConvLSTM (1506.04214) | KHÔNG TÌM THẤY | KHÔNG TÌM THẤY | KHÔNG | KHÔNG (**bài gốc là nowcasting mưa**) | abs 1506.04214; PDF không đọc được | (d) |
| ConvLSTM thay thế: FASTER (1906.04226) T3 | – | – | – | Chỉ LSTM 61.3/66.2/66.3/66.3/64.7 theo 2–32 clip; không FLOPs/params | ar5iv 1906.04226 T3 | (d) |
| I3D (1705.07750) | 25 (Two-Stream, T1) | KHÔNG TÌM THẤY trong bài | KHÔNG | 74.2 two-stream (ImageNet+K400, T3); RGB 71.1 | ar5iv 1705.07750 T1, T3 | (d) |
| I3D-R50 (mmaction2) | 28.0 | 43.5 G/view × 30 views | KHÔNG | 73.47 | mmaction2 i3d README | (d) (kiến trúc khác bài gốc) |
| SlowFast R50 4×16 | 34.5 | 36.3 G/view × 30 | KHÔNG | 75.55 | mmaction2 slowfast README | (a\*) |
| SlowFast R50 8×8 | 34.6 | 66.1 G/view × 30 | KHÔNG | 76.80 | như trên | (a\*) |
| SlowFast R101 8×8 | 62.9 | 126 G/view × 30 | KHÔNG | 78.65 | như trên | (a\*) |
| TimeSformer divST (2102.05095) | 121.4 (T1) | 0.59 T (3 views, T5); mmaction2 196 G × 3 | Chỉ wall-clock: 36 phút / 20K video / 8×V100 ≈ 9.3 video/s [suy ra, giao thức chưa rõ] | 78.0 (T5); mmaction2 77.69 (8×32, 122M) | ar5iv 2102.05095 T1, T5; mmaction2 | (d) |
| TimeSformer joint (mmaction2) | 86.11 (T1: 85.9) | 180 G × 3 | KHÔNG | 76.95 | mmaction2 | (a\*) |
| ViViT joint "Spatio-temporal" (T2) | 88.9 | 455.2 G | 58.9 ms, **TPU-v3** | 80.0 | ar5iv 2103.15691 T2 | (d) |
| ViViT FE (T2) | 115.1 | 284.4 G | 17.4 ms, TPU-v3 | 78.8 | như trên | (d) |
| ViViT FSA / FDP (T2) | 117.3 / 88.9 | 372.3 / 277.1 G | 31.7 / 22.9 ms, TPU-v3 | 77.4 / 76.3 | như trên | (d) |
| ViViT-L/16×2 FE views (T6a) | – | 3.98 T (1×1) / 11.94 T (1×3) | – | 80.6 / 81.7 | như trên | (d) |
| VideoMAE ViT-B (zoo) | 87 | 180 G × 15 views (5 clip × 3 crop) | KHÔNG | 81.3 | mmaction2 videomae README | (a\*) |
| VideoMAE ViT-L | 305 | 597 G × 15 views | KHÔNG | 85.3 | như trên | (a\*) |
| X-CLIP B/32 8×224 | KHÔNG TÌM THẤY trong README | 39 G (views không rõ; log multi-view = 3 crop × 4 clip) | KHÔNG | 80.4 | VideoX X-CLIP README | (a\*) |
| X-CLIP B/32 16×224 | – | **75 G** | KHÔNG | 81.1 | như trên | (a\*) |
| X-CLIP B/16 8× / 16× | – | 145 G / 287 G | KHÔNG | 83.8 / 84.7 | như trên | (a\*) |
| EVL ViT-B/16 frozen CLIP, 8×3 | **+7.41 (1 lớp) / +28.70 (4 lớp)** (T2), chưa gồm CLIP đóng băng | 444 G (T1) vs 454 G (T3): **MÂU THUẪN** | 25.53 views/s, V100-32G, AMP (T3; [cần xác minh]) | 82.9 (8f); 83.6 (16f); 84.2 (32f) | ar5iv 2208.03550 T1–T3 | (d) |
| MoViNet A0…A5, K600 | 3.1 / 4.6 / 4.8 / 5.3 / 4.9 / 15.7 | 2.71 / 6.02 / 10.3 / 56.9 / 105 / 281 (1 view; 50,50,50,120,80,120 frames) | Không throughput. **V100 stream latency (T6): video 183/315/325/1110/1130/2310 ms; per-frame 3.7/6.3/6.5/9.2/14.1/19.2 ms** | 71.5/76.0/77.5/80.8/81.2/82.7 (K600, T2) | arxiv html 2103.11511 T2,3,6,10 | (d) |
| MoViNet K400 (T9) | như trên | 2.71…281; A6 386 | – | 65.8/72.7/75.0/78.2/80.5/80.9; A6 81.5 | như trên | (a\*) |
| X3D XS/S/M/L (zoo) | 3.8/3.8/3.8/6.2 | 0.60/1.96/4.73/18.37 (views không rõ) | KHÔNG | 69.5/73.5/76.2/77.5 (30-view); 68.7/73.1/75.1/76.9 (10-view) | SlowFast MODEL_ZOO.md | (a\*) (xung đột convention) |
| X3D (abs 2004.04730) | – | "4.8× và 5.5× ít multiply-adds và tham số" so công trình trước, cùng độ chính xác | KHÔNG | – | abs 2004.04730 | – |
| **CLIP ViT-B/32 vision (open_clip)** | **87.85** (image); 151.28 toàn CLIP | **4.41 GMACs** (timm [chưa xác minh]) = 8.82 `image_gflops`; **16 khung ≈ 141 G** [suy ra] | KHÔNG | – | open_clip docs/model_profile.csv | **(a)** |
| CLIP paper (2103.00020) | KHÔNG TÌM THẤY | KHÔNG TÌM THẤY | KHÔNG | – | chưa đọc được | – |

## (B) Triển khai biên

### B1. Throughput/latency đã công bố trên Jetson
| Nguồn | Thiết bị | Mô hình | Precision | Kết quả | Verdict |
|---|---|---|---|---|---|
| Ultralytics docs nvidia-jetson, bảng TRT FP16 640 | **Orin Nano Super**, AGX Orin, Orin NX 16GB | YOLO26 (kích thước không xác định) | FP16 TRT | Orin Nano Super **4.57 ms/im**; AGX Orin 2.62; NX 16GB 4.13 (không gồm pre/post) | (b) với hệ đích, nhưng là YOLO26 |
| hokwangchoi/jetson-orin-nano-benchmarks (repo cộng đồng) | **Orin Nano Super, MAXN_SUPER, JetPack 6.2.2** | YOLOv8n 640 | TRT FP16 | **4.43 ms GPU, 225.3 FPS** (trtexec, không gồm H2D/D2H) | (b) |
| Seeed Studio industrial vision benchmark | Orin Nano 8GB, 25W, JetPack 6.2 | YOLOv8n 640 | FP16 | **58.82 FPS tổng hợp 4 luồng 4K**; P95 1108 ms; 11.02 W module input | (b), pipeline đầy đủ, không phải engine |
| cùng nguồn | Orin NX 16GB, MAXN, JP 5.1.3 | YOLOv8n 640 | FP16 | 78.06 FPS; 335.57 ms; 15.76 W | – |
| Jetson AI Lab benchmarks (archive 2024) | không rõ | clip-vit-base-patch32: 196/314/1.60; vit-base-patch32-224: 171/273/1.60 | không rõ | **Đơn vị và thiết bị không có trong text** | (d), không dùng được |
| Forum NVIDIA (snippet) | Orin Nano (4GB/8GB mâu thuẫn), JP 6.2.1 | ViT-S | FP16 | 30.75 ms, 32.1 qps | [chưa xác minh] |
| Forum NVIDIA (05/2024) | Orin Nano 8GB | TarDAL (không phải video) | TRT FP16 | ~60 ms fusion | (d) |
| **Mô hình video nào trên Orin** | – | I3D, SlowFast, TimeSformer, ViViT, VideoMAE, X-CLIP, EVL, MoViNet, X3D | – | **KHÔNG TÌM THẤY** | – |

**Kiểm dải 25–60 FPS (YOLOv5s–YOLO11/YOLOv8n, 640, Jetson):**
- YOLOv8n engine FP16 trên Orin Nano Super: **225 FPS**, vượt xa 60.
- 58.8 FPS là thông lượng **tổng hợp 4 luồng 4K** có decode + postprocess, không phải tốc độ mô hình.
- YOLOv5s và YOLO11 ở 640 trên Orin: không có số đọc được (Ultralytics YOLO11 blog đặt số trong ảnh Fig 4, 5).
- **Kết luận: 25–60 FPS không xác nhận được là FPS mô hình đơn trên Orin Nano Super; chỉ đúng nếu hiểu là FPS pipeline nhiều luồng.** → tham chiếu [21] của chương 2 cần dán nhãn lại.

### B2. Năng lượng mỗi suy luận
| Nguồn | Thiết bị | Đối tượng | Năng lượng | Quy ước | Verdict |
|---|---|---|---|---|---|
| **CoDAT, arXiv 2608.06691, Table V** | Jetson AGX Orin, 50W max | TimeSformer ViT-B, 8×224² | **591.90 mJ/frame** (10.27 ms/F, 54.1 W); ≈4735 mJ/clip | jtop/INA3221 "total power"; E = P̄·T/(N_frames·N_rep); **không trừ nền**; không nêu VDD_IN | (d) |
| cùng nguồn | AGX Orin | MoViNet-A1, 50×224² | 86.84 mJ/F (1.36 ms/F, 37.5 W); ≈4342 mJ/clip | như trên | (d) |
| cùng nguồn | AGX Orin | MoViNet-A3, 50×224² | 155.37 mJ/F (4.04 ms/F, 38.3 W); ≈7768 mJ/clip | như trên | (d) |
| cùng nguồn | AGX Orin | CoDAT-S, 8×256² | 13.79 mJ/F (0.89 ms/F, 15.5 W); ≈110 mJ/clip | như trên | (d) |
| Seeed YOLOv8n | Orin Nano 8GB, 25W | YOLOv8n pipeline 4 luồng | 11.02 W / 58.82 FPS ≈ **187 mJ/frame** | module input power, pipeline đầy đủ | (b) nhưng không per-inference |
| Mô hình video trên Orin Nano Super | – | – | **KHÔNG TÌM THẤY** | – | – |

Quy ước đề xuất để tương thích CoDAT: E = P̄ × T / (N_frames × N_rep); ghi rõ rail (VDD_IN vs INA3221 tổng) và có trừ nền hay không. **CoDAT không trừ nền** → đối chiếu được với số "total-board" của luận văn, không với số "marginal".

### B3. Decision rate
- **Không công trình nào báo decisions/s** cho giám sát video cửa sổ trượt trên Jetson.
- Gần nhất: **ApS (Actions per Second)** trong RW-GCN (arXiv 2201.05739), đo trên Jetson Xavier NX, skeleton, không phải cửa sổ RGB. Chỉ tương tự.
- Seeed (4 luồng) và Ultralytics báo FPS, không báo decision.
- **Khoảng trống được xác nhận ở mức tìm kiếm này.**

## (C) Jetson Orin Nano Super 8GB
Nguồn chính: NVIDIA developer blog "Orin Nano Developer Kit gets a Super boost". **Trang sản phẩm developer.nvidia.com trả 404.**

| Thông số | Super | Orin Nano cũ | Ghi chú |
|---|---|---|---|
| INT8 **sparse** | 67 TOPS | 40 TOPS | Blog spec table |
| INT8 **dense** | 33 TOPS | 20 TOPS | Blog spec table |
| Công suất | 7 / 15 / **25 W** | 7 / 15 W | Thêm mode 25W |
| Giá niêm yết | **$249** | $499 | Reseller khác $408–517 [snippet] |
| GPU clock | 1020 MHz | 635 MHz | Blog |
| CPU clock | 1.7 GHz | 1.5 GHz | Blog |
| Băng thông bộ nhớ | 102 GB/s (text) | 65 (text) / 68 (spec table) | **Blog tự mâu thuẫn** |
| Kiến trúc | cùng phần cứng; 1024 CUDA, 32 Tensor core, 8GB | – | Blog |

- **"Super" là nâng cấp bằng phần mềm** (JetPack 6.1+) trên cùng phần cứng: mở power mode mới, tăng clock GPU/bộ nhớ/CPU. Blog nêu `nvpmodel -m 2` (MAXN).
- **67 TOPS là sparse INT8**, không phải FP16, không dense → **không so với throughput FP16**.
- TFLOPS FP16 chính thức: **KHÔNG TÌM THẤY**. Forum nêu ~17 TFLOPS ở Super mode; nguồn khác nói NVIDIA không công bố FP16 [chưa xác minh].

## Đặt cạnh hệ thống luận văn
- **Params CLIP B/32 (87.85 M): (a)** — khớp 87.8 M của luận văn.
- **GFLOPs/quyết định ≈ 70.6 GMACs ≈ 141 GFLOPs** [suy ra]; chưa có số đo. X-CLIP B/32 (75 G/16 khung) và EVL (444 G/8×3) là tham chiếu gần nhất nhưng khác model và convention.
- Throughput/latency mô hình video: **(d)** — số đo trên V100/TPU, không trên Orin.
- Năng lượng: **(d)** — CoDAT đo trên AGX Orin, không trừ nền.
- YOLO FPS: **(b)** với Orin Nano Super, nhưng khác mô hình/tác vụ.
- Decision rate: **không có tham chiếu nào**.

## Câu hỏi chưa giải quyết
1. ConvLSTM: không đọc được PDF; FLOPs/params trong phân loại video vẫn KHÔNG TÌM THẤY.
2. Nhiều số (I3D, SlowFast, X3D, VideoMAE, X-CLIP, CLIP) từ model zoo/thứ cấp — cần đối chiếu PDF gốc nếu trích trực tiếp.
3. X3D: zoo 0.60 G (XS) vs MoViNet T9 23.3 G, chênh ~40×; convention chưa rõ.
4. EVL: 444 G (T1) vs 454 G (T3); cột 25.53 "views/s" chưa xác minh.
5. ViViT: 88.9 M params xuất hiện cho cả ViViT-B (T2) và ViViT-L/16×2 (MoViNet T9); config cần kiểm.
6. Jetson AI Lab CLIP: thiếu đơn vị và thiết bị → không dùng được.
7. Rail công suất: CoDAT (jtop/INA3221) vs Seeed (module input) khác nhau.
8. Trang sản phẩm NVIDIA 404; băng thông bộ nhớ mâu thuẫn trong blog.
9. Ultralytics YOLO11 trên Orin Nano Super nằm trong ảnh, chưa đọc được.
10. FP16 TFLOPS chính thức của Orin Nano Super chưa xác minh.
