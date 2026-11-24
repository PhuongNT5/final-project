# Researcher 04 — cái giá độ chính xác: FP32 → FP16 → INT8 (ViT, CLIP, video)

Lưu ý phạm vi: câu trích nguyên văn của NVIDIA và ONNX Runtime **chưa lấy được** (công cụ trả bản đã tóm tắt). Mọi chỗ như vậy ghi rõ.

## 1. PTQ4ViT — arXiv 2111.12293, Table 1, Sec 4.2
Nguồn: arxiv.org/html/2111.12293v3. PTQ. W8A8 = weight và activation đều 8-bit. Phạm vi: "We quantize all the weights and inputs for the fully-connect layers" → **chỉ lớp FC**; matmul attention không nêu rõ. Calibration: **32 ảnh** ngẫu nhiên từ train, không nhãn. Drop trong ngoặc (pp).

| Model | FP32 | Base PTQ W8A8 | PTQ4ViT W8A8 | Base PTQ W6A6 | PTQ4ViT W6A6 |
|---|---|---|---|---|---|
| ViT-S/224 | 81.39 | 80.46 (0.91) | 81.00 (0.38) | 70.24 (11.1) | 78.63 (2.75) |
| ViT-B/224 | 84.54 | 83.89 (0.64) | 84.25 (0.29) | 75.66 (8.87) | 81.65 (2.89) |
| DeiT-S/224 | 79.80 | 77.65 (2.14) | 79.47 (0.32) | 72.26 (7.53) | 76.28 (3.51) |
| DeiT-B/224 | 81.80 | 80.94 (0.85) | 81.48 (0.31) | 78.78 (3.01) | 80.25 (1.55) |
| Swin-T/224 | 81.39 | 80.96 (0.42) | 81.24 (0.14) | 78.45 (2.92) | 80.47 (0.91) |
| Swin-S/224 | 83.23 | 82.75 (0.46) | 83.10 (0.12) | 81.74 (1.48) | 82.38 (0.84) |
| Swin-B/224 | 85.27 | 84.79 (0.47) | 85.14 (0.12) | 83.35 (1.91) | 84.01 (1.25) |

Thiếu: các hàng ViT-S/224/32, ViT-B/384, DeiT-B/384 không đọc được trong phiên này.

## 2. FQ-ViT — arXiv 2111.13824, Table 1
Nguồn: arxiv.org/html/2111.13824v4. PTQ, **full-quant**: "quantizing all modules, including Conv, Linear, MatMul, LayerNorm, Softmax, etc." Calibration: "randomly sample **1000** training images from ImageNet or COCO". Cột = (W/A/attention bit).

| Model | FP32 (32/32/32) | FQ-ViT (8/8/8) | FQ-ViT (8/8/4) |
|---|---|---|---|
| DeiT-T | 72.21 | 71.61 | 71.07 |
| DeiT-S | 79.85 | 79.17 | 78.40 |
| DeiT-B | 81.85 | 81.20 | 80.85 |
| ViT-B | 84.53 | 83.31 | 82.68 |
| ViT-L | 85.81 | 85.03 | 84.89 |
| Swin-T | 81.35 | 80.51 | 80.04 |
| Swin-S | 83.20 | 82.71 | 82.47 |
| Swin-B | 83.60 | 82.97 | 82.38 |

Drop 8/8/8 (pp): DeiT-T 0.60; DeiT-S 0.68; DeiT-B 0.65; **ViT-B 1.22**; ViT-L 0.78; Swin-T 0.84; Swin-S 0.49; Swin-B 0.63. Drop 8/8/4: 1.14; 1.45; 0.99; 1.85; 0.92; 1.31; 0.73; 1.22.

**PTQ4ViT (chỉ FC, 32 ảnh) và FQ-ViT (full-quant, 1000 ảnh) KHÔNG so trực tiếp được.**

## 3. X3D-XS — PyTorchVideo model zoo (video, INT8 thật)
Nguồn: pytorchvideo.readthedocs.io/en/latest/model_zoo.html. Trang có **hai bảng khác nhau**:
(a) Bảng Kinetics-400: X3D XS, 4×12 frame, top1 69.12, top5 88.63, 0.91 GFLOPs × 3 × 10 views, 3.79M params.
(b) Bảng "Accelerator Model Zoo" → "Efficient Models for mobile CPU":
- X3D_XS (fp32), EfficientX3d(expansion="XS"): **68.5 | 88.0 | 233 ms | 3.8M**
- X3D_XS (int8), TorchScript sẵn trong checkpoint: **66.9 | 87.2 | 165 ms | 3.8M**

Ghi chú trang: "All top1/top5 accuracies are measured with 10-clip evaluation. Latency is benchmarked on **Samsung S8 phone** with 1s input clip length." Phương pháp INT8 **không nêu**; tên checkpoint có "qnnpack" (gợi ý backend QNNPACK), trang không giải thích. Dataset bảng (b) không ghi, chỉ suy từ tên checkpoint "kinetics".
**Drop: top1 −1.6 pp, top5 −0.8 pp. Tốc độ 233/165 = 1.41×. Mobile CPU, KHÔNG phải Jetson.**

## 4. TensorRT issue #3893 — FP16 có thể sập hoàn toàn
Nguồn: github.com/NVIDIA/TensorRT/issues/3893 (mở 23/05/2024).
- Model: image encoder của mô hình phân đoạn (L2, arXiv 2402.05008), output `image_embeddings` 1×256×64×64. Build trtexec từ ONNX; input 1×3×512×512 (min/opt) → 4×3×512×512 (max); `--fp16`.
- **mIoU: FP16 {all 0.0, large 0.0, medium 0.0, small 0.0}; FP32 {all 79.12385607181146, large 83.05853600575689, medium 81.50597370444349, small 74.8830670481846}.**
- Môi trường: RTX 4090, driver 550.54.14, CUDA 12.4, container TensorRT 24.02.
- Polygraphy: "Passed: 0/1 iterations | Pass Rate: 0.0%"; tolerance **abs=1e-05, rel=1e-05**; "Minimum required tolerance was [abs=0.15575] OR [rel=7.8643e+05]"; abs diff mean 0.015861, max 0.15575; TRT min/max −0.46509/0.55225; ORT −0.51662/0.62824; "all layers failed".
- Dấu hiệu (chưa xác nhận root cause): cảnh báo "FP16 LayerNorm nodes that may overflow", "3 weights … Detected subnormal FP16 values". Auto-reduce thu về subgraph 2 node (Mul → Conv). Debug precision chọn 2576 layer precision cao, **vẫn fail**: "Could not find a configuration that satisfied accuracy requirements."
- Phản hồi NVIDIA: brb-nv (24/05, 28/05) xác nhận khác biệt TRT vs ORT; đồng nghiệp nêu **sai khác kỳ vọng ~1e-3** cho FP16. lix19937 (25/05) nghi overflow FP16. shahdloo (09/09/2024) đề nghị golden I/O. **Trạng thái: còn mở, không có fix.**

## 5. NVIDIA accuracy-considerations (FP16)
Nguồn: docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html. **CHƯA CÓ câu nguyên văn** (chỉ bản tóm tắt của công cụ):
- FP16 "can exhibit reduced precision due to its limited precision and range".
- "Overflow in FP16 results in Inf (infinity) values"; trần **65,504** (exponent 5-bit).
- "Sigmoid or Softmax amplify small numerical differences due to their exponential component".
- "Perform critical, precision-sensitive calculations in FP32".
- **Không có ngưỡng số nào ngoài 65,504.** Không nhắc Polygraphy trong nội dung chính.
- **Ngưỡng cosine similarity chính thức của NVIDIA cho FP16: KHÔNG TÌM THẤY.**

## 6. Polygraphy atol/rtol mặc định
CHƯA CÓ câu nguyên văn. Trang compare_func (TensorRT 10.x) qua snippet: `CompareFunc.simple` "tolerance defaults to 1e-5"; mã nguồn `default_rtol = 1e-5`, `default_atol = 1e-5`; `error_stat` mặc định `elemwise`; `error_quantile` 0.99. Trang compare.py (10.11) không trả giá trị khi fetch. Mức 1e-5 cũng xuất hiện trong issue #3893.
Forum NVIDIA: atol=rtol=**1e-4** làm check pass trong một case; staff coi sai khác mức này là bình thường với FP16. URL: forums.developer.nvidia.com/t/…/263946 (chưa mở đọc).

## 7. arXiv 2509.21173 — độ tin cậy lượng tử hóa CLIP
Nguồn: arxiv.org/html/2509.21173v4.
- Tỉ lệ thất bại: **PTQ 44.1%**, **QAT 25.0%** (Appendix C.2). Simple PTQ 62.0%, Rotation PTQ 5.6%, Rotation+LSQ 0.0% (Appendix E).
- Định nghĩa "thất bại": theo tóm tắt = accuracy drop **>5%**. Chỉ số accuracy cụ thể (zero-shot hay khác): [chưa xác minh].
- Cỡ mẫu: **20,160 lần chạy W8A8** (caption Figure 1), **simulated (fake) quantization**. Calibration: 1000 cặp image-caption từ CC3M/YFCC/SBU; Appendix F.2 chọn N=1000 sau quét 1→10,000.
- 10 kiến trúc: CLIP ViT, CLIP ConvNeXt, SigLIP, ALIGN, CoCa… Biến thể CLIP nêu: **CLIP-ViT-B-32-WIT**, CLIP-ViT-B-32-Laion2B, L-14-Laion2B, DFN-ViT-B-32, ViT-L-14. ViT-B/16 không nêu là CLIP.
- **KHÔNG CÓ bảng FP32 vs W8A8 zero-shot top-1 ImageNet** trong phần đọc được (100k/109,854 ký tự).

NVIDIA ModelOpt CLIP (developer.nvidia.com blog PTQ): **FP8 E4M3, không phải INT8**. W8A8 per-tensor static, AbsMax, calibration 8192 cặp COCO. Số chỉ trong hình (không đọc được). Văn bản: "comparable quality to the CLIP-FP16 model"; "negligible" khi bỏ quantizer patch-embedding.
OpenVINO/NNCF CLIP notebook (docs.openvino.ai 2023.3, 228-clip-zero-shot-quantize): SmoothQuant 8-bit ViT-B/16; số accuracy KHÔNG đọc được (trang bị cắt). [chưa xác minh]

## 8. Tăng tốc Jetson Orin Nano Super (YOLOv8n)
Nguồn: github.com/hokwangchoi/jetson-orin-nano-benchmarks (MAXN_SUPER), latency GPU:
- PyTorch FP32 **20.88 ms** · TensorRT FP32 **8.36 ms** · TensorRT FP16 **4.43 ms** · TensorRT INT8 **3.49 ms**
- Tính: FP32→FP16 = **1.89×**; FP16→INT8 = **1.27×**; PyTorch FP32→INT8 = 5.98×.
- Vị trí chính xác trong repo (tệp/dòng) chưa xác định.

arXiv 2609.13551 (Orin Nano Super, nông nghiệp): TRT FP32 ~2.4× so PyTorch FP32; FP16 ~2.5–2.7× app-level; INT8 gần như không thêm so FP16; **mAP@50:95 giảm 0.002–0.005 khi FP16**; năng lượng giảm 64–66%. [chưa xác minh: chỉ snippet, PDF không đọc được bảng]

## 9. Giảm size engine FP16 vs FP32
- arXiv 2509.18193 (TinyEcoWeedNet): FP16 engine giảm **~35%** trung bình; scratch và context memory ~49% mỗi loại. EcoWeedNet 13.0 → 7.2 MB. [chưa xác minh bảng; từ snippet]
- arXiv 2603.28708: giảm **49%** engine size (halved weight precision + memory planning của TensorRT); memory tối đa giảm 63% so PyTorch FP32. [chưa xác minh bảng; từ snippet]
- Forum NVIDIA: **không đảm bảo** giảm size theo data type ("highly tactic dependent"); có case FP16 engine 87 MB bằng FP32; có user không thấy giảm bộ nhớ.
→ **Dải thực tế: −35% đến −49%. KHÔNG hứa 50%.**

## 10. Weight-only vs activation — và cảnh báo ONNX Runtime
- **Weight-only:** chỉ weight lượng tử hóa, activation giữ float. Size giảm ~4× trên phần weight. Mất độ chính xác thường nhỏ hơn W8A8, nhưng **không có số ViT weight-only có nguồn** trong phiên này [chưa xác minh].
- **W8A8:** số drop ở mục 1 và 2.
- **Cảnh báo `quantize_dynamic` (ORT):** tài liệu onnxruntime.ai/docs/performance/model-optimizations/quantization.html mô tả weight int8 **và activation được lượng tử hóa động khi chạy**. Câu nguyên văn CHƯA LẤY ĐƯỢC (chỉ bản tóm tắt). Vì vậy nhánh INT8 của luận văn **KHÔNG chắc là weight-only thuần**; cần đọc docs đúng phiên bản và **kiểm graph** (node `DynamicQuantizeLinear`, `MatMulInteger`/`ConvInteger`).
- Con số **156 → 40 MB (3.9×)** khớp lượng tử hóa weight INT8 (~4× lý thuyết), **không chứng minh được** activation có lượng tử hóa hay không.

## 11. Trả lời hai câu hỏi hội đồng

**Q1: "Mất 1/64 mẫu khi INT8 là nhiều hay ít?"**
- 56/64 = 87.5%; 55/64 = 85.9%; chênh **1.56 pp**.
- Dải W8A8 đã công bố: **0.12–2.14 pp** (PTQ4ViT, 32 ảnh calib, chỉ FC); **0.49–1.22 pp** (FQ-ViT full-quant, 1000 ảnh). → **1.56 pp ở đầu trên hoặc vượt dải FQ-ViT**, nằm trong dải PTQ4ViT.
- n=64: SE ≈ **4.1 pp**. Nếu chỉ 1 mẫu đổi và không có chiều ngược, **McNemar exact p = 1.0**. → **không có ý nghĩa thống kê**.
- **Cần đếm số flip hai chiều** để kết luận đúng.

**Q2: "FP16 có được coi là không mất gì không?"**
- **Không.** Không nguồn nào coi FP16 lossless. NVIDIA cảnh báo sai số và khuyến nghị giữ op nhạy ở FP32. Issue #3893 cho thấy **sập hoàn toàn** (mIoU 79.12 → 0.0).
- **Căn cứ phòng thủ duy nhất:** tương đương **đo được** trên tập kiểm, dưới **ngưỡng định trước**, kèm max abs diff và tỉ lệ đổi nhãn. → câu "FP16 không mất gì" trong báo cáo phải đổi thành phát biểu tương đương-đo-được.

## 12. KHÔNG TÌM THẤY
- ViT TensorRT FP16 vs FP32 top-1 trên ImageNet (không nghiên cứu nào).
- Ngưỡng cosine similarity chính thức NVIDIA cho FP16.
- CLIP zero-shot top-1 FP32 vs INT8 (2509.21173; OpenVINO notebook).
- I3D, SlowFast, MoViNet: accuracy INT8 hoặc FP16.
- VideoMAE accuracy sau INT8: chỉ có latency NPU 407.573 ms (huggingface.co/qualcomm/Video-MAE-Quantized), không có accuracy.
- Đảm bảo giảm size FP16 = 50%.

## 13. Chưa xác minh (dùng thận trọng)
- BERT-base SST-2: FP16 toàn bộ −0.34 pp; sai số lớn nhất 4.12e-2 lật dự đoán sát ngưỡng. ID arXiv chưa xác định.
- dev.to/tildalice/onnx-int8-vs-fp16-3x-latency-drop-on-jetson-orin-nano: 47 → 15 ms (~3×), **mâu thuẫn** bảng YOLOv8n mục 8.
- Forum NVIDIA: YOLOv8s INT8 ~1.5× so FP16.
- ViT-L 16-bit 76.54 → 75.29 (arXiv 2406.18051): cấu hình 16-bit không rõ là FP16.
- FP16 ~0.2% trung bình, 12 kiến trúc (arXiv 2305.10947): là **training**, không phải TensorRT inference.
- X-CLIP gameplay (arXiv 2505.07721): ONNX INT8 PTQ, size ~4×, accuracy 94.3 → 92.9. Một ứng dụng, không phải benchmark.

## Câu hỏi chưa giải quyết
1. Số flip **hai chiều** của 1/64 mẫu (cần để tính McNemar đúng).
2. `quantize_dynamic` có lượng tử hóa activation trong graph đang dùng không (kiểm node).
3. Ngưỡng FP16 cho dự án (cosine, max abs, tỉ lệ đổi nhãn) phải **tự đặt và biện minh** — không có chuẩn ngoài.
4. Nguyên văn NVIDIA accuracy-considerations, polygraphy compare.py, ORT docs: cần mở trực tiếp.
5. Định nghĩa "thất bại" chính xác trong 2509.21173.
6. Video: cần thêm số INT8/FP16 cho I3D, SlowFast, VideoMAE, MoViNet.
