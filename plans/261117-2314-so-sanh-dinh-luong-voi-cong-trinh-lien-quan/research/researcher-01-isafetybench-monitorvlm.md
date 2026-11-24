# Researcher 01 — số liệu đã công bố: iSafetyBench & MonitorVLM

Nguồn đọc: arXiv HTML + WebFetch. **Chưa đối chiếu PDF thô** — cần `pdftotext` xác nhận lần cuối trước khi trích vào bản thảo.

## 1. iSafetyBench (arXiv:2508.00399, VISION'25 @ ICCV 2025)

### Dữ liệu (Abstract; §3; Table 1)
- 1,100 clip, 4–8 s, cắt quanh hành động; nguồn YouTube qua truy vấn do Gemini 2.5 Pro sinh.
- 680 normal / 420 hazard.
- 98 nhãn normal + 67 nhãn hazard, open-vocabulary, multi-label; 1,468 instance normal, 888 instance hazard.
- 18 danh mục cấp cao (8 normal, 10 hazard), Table 1.
- **Số video gốc: KHÔNG TÌM THẤY** — bài không nêu. (Luận văn dùng 151 nguồn → không đối chiếu được ràng buộc nguồn.)
- **Không có train/test split** — toàn bộ đánh giá zero-shot.
- **Không có ràng buộc theo nguồn video.**
- **Không có baseline phi-VLM** (không có supervised classifier nào).

### Tác vụ
- Multiple-choice 16 lựa chọn.
  - Single-correct: 1 đúng + 15 distractor → metric accuracy.
  - Multiple-correct: chọn tất cả đáp án đúng → metric P / R / F1.
- **Không phải phân loại nhị phân** normal/hazard. **Không có balanced accuracy.**
- Chance level single-correct = 1/16 = 6.25% (suy ra, không in trong bài).

### Table 2 — "Performance of VLMs on normal and danger/hazard scenarios" (§4, HTML v2), đơn vị %

| Model | Params (bài nêu) | N-Single Acc | N-Multi P/R/F1 | H-Single Acc | H-Multi P/R/F1 | Avg Acc | Avg F1 |
|---|---|---|---|---|---|---|---|
| Ovis2-8B | 8B | 47.3 | 47.6 / 71.3 / 53.4 | 40.3 | 45.0 / 54.1 / 46.2 | 43.8 | 49.8 |
| InternVL2.5-8B-MPO | 8B | 42.2 | 47.2 / 64.1 / 50.8 | 38.3 | 47.0 / 57.7 / 49.0 | 40.25 | 49.9 |
| Qwen2.5-VL-7B-Instruct | 7B | 46.9 | 44.5 / 62.7 / 49.2 | 33.6 | 40.5 / 47.9 / 41.7 | 40.25 | 45.45 |
| VideoLLaMA3-7B | 7B | 38.9 | 49.6 / 37.4 / 39.7 | 32.7 | 46.2 / 34.1 / 36.5 | 35.8 | 38.1 |
| VideoChat-Flash-7B | 7B | 31.0 | 36.3 / 44.2 / 33.6 | 26.8 | 36.0 / 39.1 / 31.0 | 28.9 | 32.3 |
| Oryx-7B | 7B | 25.0 | 37.8 / 41.7 / 30.3 | 21.5 | 34.7 / 37.2 / 26.6 | 23.25 | 28.45 |
| Valley-Eagle-7B | 7B | 48.8 | 59.1 / 47.7 / 48.7 | 35.9 | 54.5 / 36.8 / 40.9 | 42.35 | 44.8 |
| GPT-4o | không nêu | 40.3 | 50.0 / 59.1 / 51.6 | 37.3 | 49.3 / 45.4 / 45.1 | 38.8 | 48.35 |

- Đúng 8 VLM. Bài nói "parameter counts ranging from 7 to 8 billion" (§4.1); GPT-4o không nêu tham số.
- Avg Acc = trung bình N-Single và H-Single; Avg F1 = trung bình hai F1. Tên cột "Average of Both" không được bài định nghĩa — đây là suy luận từ số học (đã kiểm lại khớp).
- **Số per-category chỉ có trong Figure 10 (hình)** → chưa đọc được số.
- Trang dự án isafetybench.github.io chỉ có `model_scores2.png`; **chưa xác minh** khớp Table 2.

### Phán quyết so sánh — mọi số iSafetyBench Table 2: **(b)**
Cùng bộ dữ liệu (1,100 clip, 680/420) nhưng khác tác vụ (16-option MCQ ≠ nhị phân), khác metric (accuracy/P/R/F1 ≠ balanced accuracy), khác protocol (zero-shot VLM ≠ probe huấn luyện trên CLIP đóng băng). **Chỉ đặt cạnh có nhãn, không trừ chéo.** Khớp với phán quyết sẵn có ở §2.3 chương 2.

Ghi chú: hai đường dẫn dữ liệu không thống nhất — `github.com/iSafetyBench/data` vs `github.com/raiyaan-abdullah/iSafety-Bench`. Cần xác minh.

## 2. MonitorVLM (arXiv:2510.03666, Jiang Wu et al.)

### Dữ liệu (§IV-A)
- Video thô từ **10 mỏ** (mining sites) — khai khoáng, không phải công nghiệp chung.
- Mẫu = **bộ ba khung hình** cách nhau 1 s (§III-A), **không phải clip video**.
- 9,000 mẫu VQA gộp từ Dataset I (3,000 cặp image–question), II (augmented), III (open-vocab detection); số mẫu II/III riêng không nêu.
- Train/test 80/20 = 7,200 / 1,800.
- 40 quy định an toàn mỏ: 17 unsafe worker behavior + 14 unsafe tool/equipment + 9 PPE (Table III).
- Clause filter: 10,000 mẫu image–clause, nhãn 0/1 từ Qwen-VL-Max.
- Metric: Precision / Recall / F1 (không có accuracy). **Tính theo mẫu hay macro/micro qua lớp: KHÔNG TÌM THẤY.**

### Model
Qwen2.5-VL-Instruct 7B / 32B / 72B; LoRA r=16, α=32, lr 1e-4, 3 epoch. Clause filter: ResNet-50 + BERT đóng băng, MLP huấn luyện được.

### Table I — Model Performance Comparison (test, %)

| Model | P | R | F1 |
|---|---|---|---|
| Claude-3.7-Sonnet | 75.45 | 55.47 | 63.94 |
| GPT-4o | 77.92 | 61.22 | 68.57 |
| Gemini-2.5 | 76.38 | 59.16 | 66.68 |
| Qwen2.5-VL-7B | 62.50 | 31.91 | 42.24 |
| Qwen2.5-VL-32B | 69.72 | 53.19 | 60.34 |
| Qwen2.5-VL-72B (baseline, không FT) | 73.33 | 61.11 | 66.66 |
| Qwen2.5-VL-72B (Dataset I) | 78.94 | 65.22 | 71.43 |
| Qwen2.5-VL-72B (I+II) | 89.13 | 79.61 | 84.10 |
| Qwen2.5-VL-72B (I+III) | 84.21 | 68.09 | 75.30 |
| MonitorVLM-7B-basic | 80.25 | 68.42 | 73.86 |
| MonitorVLM-32B-basic | 82.35 | 72.92 | 77.35 |
| MonitorVLM-72B-basic | 89.47 | 82.02 | 85.57 |
| MonitorVLM-72B w/o MB | 89.95 | 82.46 | 86.04 |
| MonitorVLM-72B (đầy đủ) | 93.05 | 89.57 | 91.28 |

### Table II — theo K của clause filter (CF)

| Setting | P | R | Inference time (s) |
|---|---|---|---|
| K=1 | 30.71 | 29.28 | 17.46 |
| K=3 | 75.63 | 70.61 | 17.51 |
| K=5 | 93.05 | 89.57 | 17.59 |
| K=10 | 93.05 | 89.57 | 18.03 |
| w/o CF | 93.05 | 89.57 | 20.35 |

### Headline abstract — ĐÃ KIỂM LẠI: là chênh lệch TƯƠNG ĐỐI, không phải pp
- "+22.01% P, +34.22% R, +28.37% F1" so Qwen2.5-VL-72B baseline: 89.47/73.33 = +22.01% tương đối. **Chênh lệch tuyệt đối = 16.14 / 20.91 / 18.91 pp**, và là so với MonitorVLM-72B-**basic**, không phải bản đầy đủ.
- "+3.45% P, +8.62% R" cho behavior magnifier: tuyệt đối **3.10 / 7.11 pp**.
- "Giảm latency 13.56%": (20.35−17.59)/20.35, giảm tương đối ở K=5 so w/o CF.

### Phần cứng
- Huấn luyện + đo latency trên **8× NVIDIA H100**.
- Có "lightweight web-based interface"; **KHÔNG CÓ** thiết bị biên, RAM, throughput, năng lượng.

### Phán quyết so sánh
- Table I & II: **(c)** — khác bộ dữ liệu, khác miền, khác tác vụ → chỉ tham khảo bậc độ lớn.
- Latency / 8×H100 / 72B: **(d)** — không có đại lượng tương đương.
- Số tham số: **(d)** — bài tính tổng tham số model; luận văn tính tham số huấn luyện được. Khác định nghĩa.

## 3. Công trình khác (2025–2026) báo cáo số trên iSafetyBench
**KHÔNG TÌM THẤY.** 3 truy vấn web; kết quả chỉ trỏ về bài gốc, GitHub, ResearchGate, alphaxiv, sotaverified, emergentmind. Không bài trích dẫn nào có bảng số riêng. Đây là tìm kiếm hạn chế, chưa rà soát đầy đủ Google Scholar / Semantic Scholar.

→ Hệ quả: luận văn là (theo tìm kiếm này) **công trình đầu tiên báo cáo kết quả supervised trên corpus iSafetyBench**, và không có số của ai khác để đặt cạnh trên cùng corpus.

## 4. Tóm tắt đặt cạnh

| Mục | iSafetyBench | MonitorVLM | Phán quyết |
|---|---|---|---|
| Bộ dữ liệu | 1,100 clip, 680/420 — cùng corpus | 9,000 VQA, mỏ khai khoáng, image triplet | (b) / (c) |
| Tác vụ | 16-option MCQ, multi-label | Violation-detection VQA | khác tác vụ / khác miền |
| Metric | Accuracy, P/R/F1 | P, R, F1 | **không bài nào có balanced accuracy** |
| Protocol | Zero-shot | LoRA FT + 80/20 | không cùng protocol |
| Số nguồn video | KHÔNG TÌM THẤY | 10 mỏ | không so được |
| Latency / HW | không có | 8×H100, 17.59 s | (d) |
| Tham số | 7–8B | 7B–72B | (d) |

**Khi viết:** nêu rõ "cùng corpus, khác tác vụ/metric"; đặt dải VLM zero-shot 23.25–43.8% Avg Acc cạnh số của luận văn **có nhãn phân biệt**, tuyệt đối không trừ chéo. Với 680/420, majority-class baseline nhị phân = 50.0% balanced accuracy.

## Nguồn
- https://arxiv.org/abs/2508.00399 · https://arxiv.org/html/2508.00399v2 · https://arxiv.org/pdf/2508.00399 · https://isafetybench.github.io/
- https://arxiv.org/abs/2510.03666 · https://arxiv.org/html/2510.03666v2
- https://github.com/iSafetyBench/data

## Câu hỏi chưa giải quyết
1. Table 2 chưa đối chiếu PDF thô — cần `pdftotext`.
2. Số per-category iSafetyBench chỉ trong Figure 10 (hình) → chưa đọc được.
3. Số video gốc iSafetyBench không có trong bài.
4. MonitorVLM: P/R/F1 theo mẫu hay macro/micro qua 40 lớp — không rõ.
5. MonitorVLM: định nghĩa "basic" (không CF, không BM?) không ghi rõ trong phần đã đọc.
6. Chưa rà soát đầy đủ Scholar/Semantic Scholar cho công trình dùng iSafetyBench.
7. Hai đường dẫn GitHub dữ liệu iSafetyBench không thống nhất.
