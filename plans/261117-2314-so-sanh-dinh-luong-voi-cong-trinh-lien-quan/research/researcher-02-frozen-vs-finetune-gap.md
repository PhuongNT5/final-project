# Researcher 02 — hao hụt độ chính xác: đóng băng vs fine-tune toàn bộ

KÝ HIỆU: pp = điểm phần trăm top-1. "Tham số" = số tham số huấn luyện được. Backbone CLIP ViT-B/16 trừ khi ghi khác. [chưa xác minh] = agent tự tính hoặc chưa đối chiếu nguồn gốc.

**ĐÍNH CHÍNH của agent so với bản tóm tắt đầu:** (a) "frozen + temporal module: 0 đến +2.5, không âm đáng kể" SAI cho SSv2 — STA-VPT-Parallel đạt **+12.0 pp**. Đúng: 0 đến +2.7 trên HMDB/UCF/K100/K400; +12 trên SSv2. (b) "mất 0–3 pp" chỉ đúng cho frozen **+ temporal module**; linear head thuần trên K400 vẫn mất 5–6 pp. (c) **ActionCLIP không có nhánh frozen** trong nguồn đọc được — chỉ có full FT.

## A. Video, CLIP ViT-B/16, K400 fully supervised

| Nguồn (bảng) | Chế độ | Tham số | K400 top-1 | Chênh vs full FT |
|---|---|---|---|---|
| ST-Adapter 2206.13559 (T1) | Linear probe, frozen | 0.31M | 76.6 | **−5.1** (vs 81.7) |
| ST-Adapter (T1) | Adapter, frozen | 6.77M | 81.6 | −0.1 |
| ST-Adapter (T1) | ST-Adapter, frozen | 7.20M (8.4% của 86.11M) | 82.0 | +0.3 |
| ST-Adapter (T1) | Full FT SA+TA | 121.57M | 81.7 | ref |
| EVL 2208.03550 (T2, T7) | EVL 1-layer, frozen CLIP | 7.41M | 81.1 (8f×3) | −0.9 vs TimeSformer FT 82.0 |
| EVL (T2, T7) | EVL 4-layer, frozen CLIP | 28.70M (~25% tổng [chưa xác minh]) | 82.9 (8f×3); 84.2 (32f×3) | +0.9 |
| EVL (T7) | EVL 16f frozen | 28.7M | 83.3 | +0.7 vs ActionCLIP 16f FT 82.6 |
| AIM 2302.03024 (T2, T6) | AIM frozen+adapters | 11M / 97M (~11% [chưa xác minh]) | 83.9 (8f×3) | **+1.9** vs TimeSformer CLIP FT 82.0 (121M) |
| AIM (T2) | AIM ViT-L/14 CLIP | 38M / 341M | 87.5 (32f×3) | không có FT cùng xương sống |
| X-CLIP 2208.02816 (T7, ablation B/16 8f) | Frozen visual + frozen text | KHÔNG TÌM THẤY | 79.1 | **−3.3** vs tuned/tuned 82.4 |
| X-CLIP (T7) | Frozen visual + tuned text | KHÔNG TÌM THẤY | 79.3 | −3.1 |
| X-CLIP (T7) | Tuned visual + frozen text | KHÔNG TÌM THẤY | 82.3 | −0.1 |
| **X-CLIP (T1)** | **B/32 vs B/16, 16f, full FT** | — | **81.1 vs 84.7** | **B/16 hơn 3.6 pp — tham chiếu xương sống** |
| ActionCLIP 2109.08472 (T8; EVL T2) | Full FT B/16 | 105.15M | 82.6 (16f); 83.8 (32f) | ref, không có nhánh frozen |
| Vita-CLIP 2304.03307 (T1, T5) | Frozen backbone + prompt | 38.88M [chưa xác minh: có thể gồm nhánh text] | 80.5 (8f×1); 82.9 (16f×4×3) | 8f: −1.8 vs X-CLIP FT 82.3 (protocol khác) |

Ghi chú X-CLIP T7: cột few-shot không xác định dataset; số T7 (82.4) khác T1 (83.8) → không so trực tiếp.

## B. CLIP, full data, attentive probe — VideoGLUE 2307.03166

| Backbone | Benchmark | Probe (T4) | Full FT (T3) | Chênh |
|---|---|---|---|---|
| CLIP | K400 | 75.2 | 81.0 | −5.8 |
| CLIP | SSv2 | 41.0 | 46.6 | −5.6 |
| CLIP | Diving48 | 44.1 | 75.7 | **−31.6** |
| CLIP | Charades mAP | 11.2 | 54.3 | **−43.1** |
| VideoMAE | K400 | 65.1 | 78.7 | −13.6 |
| VideoMAE | SSv2 | 53.9 | 65.5 | −11.6 |

CLIP LoRA (T6): K400 80.2 (−0.8); SSv2 56.0 (**+9.4** vs FT 46.6); D48 77.2 (+1.5). CLIP MLAP (T5): SSv2 50.1 (+3.5 vs FT). → **Full FT không luôn tốt nhất cho CLIP**; kết quả phụ thuộc cách adapt. Tham số không báo cáo.

## C. Linear probe vs fine-tune

C1. CLIP gốc 2103.00020 (Appendix T10, đọc một phần): linear probe ViT-L/14 UCF101 91.5, K700 72.0 (ảnh/frame, không temporal); RN50 UCF101 81.6, K700 57.2. Zero-shot vs RN50 linear: K700 +14.5, UCF101 +7.7 (Sec 3.1.5). Zero-shot thắng RN50 linear 16/27 dataset; linear CLIP thắng Noisy Student EfficientNet-L2 21/27 (Fig 11). **KHÔNG TÌM THẤY**: trung bình 27 dataset linear vs zero-shot vs FT (Fig 10; PDF không giải mã, HTML cắt ở 100k/238k ký tự); linear vs FT từng dataset.

C2. VideoMAE 2203.12602 (T3, SSv2): MoCo v3 300ep FT 54.2 / linear 33.7 (−20.5); VideoMAE 800ep FT 69.6 / linear 38.9 (**−30.7**). Backbone pre-train SSv2, không phải CLIP. K400: chỉ full FT 80.0; linear K400 KHÔNG TÌM THẤY.

## D. Trục dữ liệu nhỏ (quan trọng nhất cho luận văn)

**D1. STA-VPT 2312.10376 (T VIII).** CLIP ViT-B, T=8, Adam lr 1e-4, cosine. Epoch KHÔNG TÌM THẤY. HMDB51 6,766 video / 51 lớp (split train KHÔNG TÌM THẤY; survey 2402.02242 ghi 3,500 [chưa xác minh cùng split]); UCF101 13,320; K100 = 100 lớp ngẫu nhiên 80/20 mỗi lớp; SSv2 subset không nêu.

| Dataset | Head FT (0.07M) | Full FT (85.87M) | Chênh |
|---|---|---|---|
| HMDB51 | 55.49 | 70.72 | −15.2 |
| UCF101 | 86.41 | 94.50 | −8.1 |
| K100 | 77.15 | 83.05 | −5.9 |
| SSv2 | 20.80 | 37.75 | −17.0 |

PEFT cùng bảng:
| Phương pháp | Tham số | HMDB51 | UCF101 | K100 | SSv2 |
|---|---|---|---|---|---|
| VPT-deep | 4.68M | 70.33 (−0.4) | 94.26 (−0.2) | 84.50 (+1.5) | 34.70 (−3.1) |
| STA-VPT-Parallel | 5.35M | 73.20 (+2.5) | 95.48 (+1.0) | 85.70 (+2.7) | 49.70 (**+12.0**) |
| STA-VPT-Cascaded | 5.35M | 72.75 (+2.0) | 95.03 (+0.5) | 85.70 (+2.7) | 46.30 (+8.6) |

Ghi chú: full FT SSv2 37.75 thấp bất thường so bài khác; cần kiểm protocol [chưa xác minh].

**D2. AIM 2302.03024 (Sec 1, Sec 5, Fig 3).** K400 với **1% dữ liệu train**: AIM (frozen + adapter) vs TimeSformer full FT, cùng CLIP ViT-B/16 → **+8.9 pp** (Sec 5) / +9 (Sec 1). Giá trị tuyệt đối Fig 3 KHÔNG TRÍCH ĐƯỢC. 1% K400 ≈ 2.4k clip [chưa xác minh].

**D3. ViFi-CLIP 2212.03640 (T3 few-shot; full FT cả hai encoder):**

| Dataset | K | ViFi-CLIP (full FT) | A5 (prompt + vài layer) | Chênh | CLIP image-FT | Chênh vs image-FT |
|---|---|---|---|---|---|---|
| UCF101 | 2 | 80.7 | 71.4 | **+9.3** | 74.4 | +6.3 |
| HMDB51 | 2 | 57.2 | 39.7 | **+17.5** | 49.6 | +7.6 |
| SSv2 | 2 | 6.2 | KHÔNG TÌM THẤY | — | 4.9 | +1.3 |
| UCF101 | 16 | 92.7 | KHÔNG TÌM THẤY | — | 90.5 | +2.2 |
| HMDB51 | 16 | 66.8 | KHÔNG TÌM THẤY | — | 62.0 | +4.8 |

Vanilla CLIP không train: HMDB 41.9, UCF 63.6, SSv2 2.7 (không đổi theo K). ViFi-CLIP không có linear probe trong bảng nào.

**D4. X-CLIP T7** (B/16, 8f, ablation): few-shot Frozen/Frozen 47.3 vs Tuned/Tuned 54.6 (−7.3); zero-shot 64.2 vs 72.9 (−8.7). Dataset cột few-shot không rõ.
**D5. ST-Adapter Fig 2c:** chênh tăng khi dữ liệu giảm (định tính, không số). Fig 2b: full FT giảm nhanh hơn khi ít bước.
**D6. EVL:** không có thí nghiệm dữ liệu nhỏ. KHÔNG TÌM THẤY.

## E. Mâu thuẫn ViFi-CLIP K=2 (giữ nguyên, nêu điều kiện)
ViFi-CLIP (full FT) thắng A5 (frozen+prompt) 9.3 (UCF) đến 17.5 pp (HMDB) ở K=2. Điều kiện làm nó thắng:
1. Baseline là **A5**, không phải linear probe; A5 vẫn train prompt và vài layer.
2. Full FT **cả hai** encoder; dùng text encoder để phân lớp.
3. **Chỉ ở K=2** (rất ít mẫu); khoảng cách thu hẹp khi K tăng; không có số A5 ở K=16.
4. Protocol few-shot (lấy mẫu, epoch) không đồng nhất với STA-VPT.
Đối chiếu: STA-VPT (full train, HMDB 3.5k, 8f) cho PEFT **thắng** full FT +2.5 pp. Hai kết quả khác protocol → **mâu thuẫn chưa giải quyết**. [chưa xác minh] kết luận nào đúng cho bài toán nhị phân.

## F. Xương sống ảnh đóng băng + đầu thời gian vs video tiền huấn luyện

F1. VideoGLUE 2307.03166 (full data, cùng dataset):
| Chế độ | Benchmark | CLIP (ảnh) | VideoMAE (video) | Chênh |
|---|---|---|---|---|
| Probe (T4) | K400 | 75.2 | 65.1 | **ảnh +10.1** |
| Probe (T4) | SSv2 | 41.0 | 53.9 | video +12.9 |
| Probe (T4) | D48 | 44.1 | 59.5 | video +15.4 |
| Probe (T4) | Charades mAP | 11.2 | 11.3 | ~0 |
| LoRA (T6) | K400 | 80.2 | 73.6 | ảnh +6.6 |
| LoRA (T6) | SSv2 | 56.0 | 61.4 | video +5.4 |

→ Khớp chiều với kết quả luận văn (CLIP+head 84.4% vs VideoMAE frozen 81.5%, +2.9 pp): trên tác vụ **appearance-dominant** như K400, backbone ảnh đóng băng **thắng** backbone video đóng băng. SSv2/D48 (motion-dominant) thì ngược lại.

F2. EVL T3 (IN-21k backbone): frozen EVL 75.4 vs TimeSformer FT IN-21k 76.9–78.0 → EVL thua −1.5 đến −2.6.
F3. 2605.19137 (preprint 2026, chưa peer review), Fig 4 SSv2: frozen DINOv3-L + GMMix 25% dữ liệu = 56.5; frozen RVM 100% = 46.9; VideoMAE-B full = 52.3; VideoMAE-L full = 62.7 (T3). Không cùng tỷ lệ dữ liệu.
KHÔNG TÌM THẤY: frozen ảnh + temporal head vs VideoMAE trên cùng dataset **nhỏ**.

## G. Tham số huấn luyện được (tỷ lệ so backbone ~86M)

| Chế độ | Tham số | Tỷ lệ | Nguồn |
|---|---|---|---|
| Linear head (STA-VPT) | 0.07M | ~0.08% | 2312.10376 T VIII |
| **Luận văn (head Transformer thời gian)** | **0.201M** | **0.23%** = 201,090 / (87,849,216 + 201,090) | repo |
| Linear head (ST-Adapter) | 0.31M | ~0.36% | 2206.13559 T1 |
| VPT-deep | 4.68M | ~5.4% | 2312.10376 |
| STA-VPT-Parallel | 5.35M | ~6.2% | 2312.10376 |
| Adapter (ST-Adapter) | 6.77M | ~7.9% | 2206.13559 T1 |
| ST-Adapter / EVL 1-layer | 7.2–7.4M | ~8% | 2206.13559 T1; 2208.03550 T2 |
| AIM ViT-B | 11M | ~11% | 2302.03024 T2 |
| EVL 4-layer | 28.7M | ~25% tổng [chưa xác minh] | 2208.03550 T2 |
| Vita-CLIP | 38.88M | ~45% [chưa xác minh phạm vi] | 2304.03307 |
| Full FT | 85.87–121.57M | 100% | 2312.10376; 2206.13559 |

Luận văn 0.201M **nhỏ hơn 37× EVL 1-layer** và 27× STA-VPT, nhưng **lớn hơn 2.9× linear head** của STA-VPT. Vị trí: giữa linear-probe và PEFT nhẹ nhất. Chưa có nguồn nào đo xương sống ViT-B/**32**; toàn bộ bảng là B/16.

## H. Bảng tổng hợp: chế độ × chênh pp × điều kiện dữ liệu

| Chế độ huấn luyện | Chênh vs full FT (pp) | Điều kiện dữ liệu | Nguồn |
|---|---|---|---|
| Frozen + temporal module (EVL, AIM, ST-Adapter, STA-VPT) | **0 đến +2.7** trên HMDB/UCF/K100/K400; +12 trên SSv2 | Đủ hoặc nhỏ | 2208.03550; 2302.03024; 2206.13559; 2312.10376 |
| Frozen + temporal module, **1% K400** | **+8.9** (AIM vs TimeSformer FT) | ~2.4k clip [chưa xác minh] | 2302.03024 Sec 5 |
| Linear probe, frozen, K400 | −5.1 (ST-Adapter), −5.8 (VideoGLUE) | Đủ dữ liệu | 2206.13559 T1; 2307.03166 |
| Linear probe, frozen, HMDB/UCF/K100 | **−5.9 đến −15.2** | Nhỏ (3.5k–13k video) | 2312.10376 T VIII |
| Linear probe, SSv2 | −5.6 (VideoGLUE); −17.0 (STA-VPT); −30.7 (VideoMAE backbone) | Đủ/trung bình | 2307.03166; 2312.10376; 2203.12602 |
| Linear probe, Diving48 / Charades | −31.6 / −43.1 | Đủ | 2307.03166 |
| Frozen + prompt/A5 (few-shot K=2) | **−9.3 (UCF), −17.5 (HMDB)** | K=2 | 2212.03640 T3 |
| Frozen (X-CLIP ablation) | −3.3 (full); −7.3 (cột few-shot) | Ablation K400 8f | 2208.02816 T7 |

**Kết luận một dòng:** đầu thời gian nhẹ trên CLIP đóng băng mất **~0–3 pp** so full FT khi dữ liệu nhỏ (và có thể **thắng** +8.9 pp ở 1% K400); chỉ linear head thuần mất 6–15 pp trên HMDB/UCF/K100 và 17–44 pp trên SSv2/motion/Charades. Mâu thuẫn ViFi-CLIP K=2 (full FT thắng 9–18 pp) chưa giải quyết.

**Trả lời hội đồng "không đóng băng thì được thêm bao nhiêu":** nguồn ủng hộ **~0–3 pp** cho kiến trúc giống luận văn (frozen + temporal head), có khả năng âm (tức full FT kém hơn) khi dữ liệu rất nhỏ. Đây là **ước lượng từ văn liệu, chưa đo trên bài toán nhị phân của luận văn**.

## I. KHÔNG TÌM THẤY
- CLIP: trung bình 27 dataset linear vs zero-shot vs FT (Fig 10); linear vs FT từng dataset.
- VideoMAE: linear probe K400.
- EVL: thí nghiệm dữ liệu nhỏ/few-shot.
- X-CLIP: dataset cột few-shot T7; tham số các cấu hình frozen.
- ViFi-CLIP: A5 ở K=4/8/16; linear probe; tham số full FT.
- Số tuyệt đối Fig 3 (AIM 1%) và Fig 2c (ST-Adapter).
- Protocol few-shot ViFi; split/epoch STA-VPT.
- PEFT survey video có bảng SSv2/K400 + tham số: 2402.02242 không có số video; 2409.16434 PDF không giải mã. CHƯA ĐỌC: 2504.14117 (PEFT A2Z); 2602.18977 (Frame2Freq).
- DualPath: chưa tìm.
- Frozen ảnh + temporal head vs VideoMAE trên cùng dataset nhỏ.

## Câu hỏi chưa giải quyết
1. Mâu thuẫn ViFi-CLIP (full FT thắng) vs STA-VPT (PEFT thắng) trên dữ liệu nhỏ: protocol nào áp dụng cho nhị phân?
2. Bài toán nhị phân chưa có nguồn; số đa lớp có thể không áp dụng.
3. Trung bình 27 dataset CLIP linear vs FT: cần bản đầy đủ.
4. Vita-CLIP 38.88M "frozen" có gồm nhánh text không?
5. VideoMAE linear K400: cần full text.
6. Frame2Freq (2602.18977) và PEFT A2Z (2504.14117): chưa đọc số.
7. Toàn bộ bảng là ViT-B/16; luận văn dùng B/32 (X-CLIP T1: B/16 hơn B/32 3.6 pp ở full FT). Chênh lệch frozen-vs-FT trên B/32 chưa có nguồn.

## Nguồn
ST-Adapter 2206.13559 · EVL 2208.03550 · ViFi-CLIP 2212.03640 · X-CLIP 2208.02816 · ActionCLIP 2109.08472 · AIM 2302.03024 · VideoGLUE 2307.03166 · VideoMAE 2203.12602 · CLIP 2103.00020 · STA-VPT 2312.10376 · Vita-CLIP 2304.03307 · preprint 2605.19137 · PEFT survey 2402.02242
