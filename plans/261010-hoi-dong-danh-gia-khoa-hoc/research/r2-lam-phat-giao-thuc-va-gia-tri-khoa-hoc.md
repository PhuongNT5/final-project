# R2: Lạm Phát Giao Thức và Giá Trị Khoa Học
**Phản biện R2 — Hội đồng Đánh giá Khoa học**  
Ngày: 2026-10-10  
Chuyên đề: Tiền lệ, tính mới, nơi công bố, tiêu chuẩn

---

## 1. Phán Quyết

**KẾT LUẬN: Bài báo công bố được, mức top-tier (NeurIPS Datasets & Benchmarks / ICCV Workshops), nhưng cần định vị rõ tiền lệ.**

### Tóm tắt

Luận văn đo lạm phát giao thức (protocol inflation) — khoảng cách độ chính xác giữa split có nguồn/nhóm (source-aware) và split ngẫu nhiên (source-ignoring). Điểm mạnh:

1. **Phân rã thí nghiệm placebo chính xác**: tách +13.8 pp (source identity) khỏi +1.6 pp (fold geometry) qua hoán vị p = 0.0476. Không tìm thấy ai khác làm điều này.
2. **Hai kho dữ liệu độc lập**: +13.0 → +21.8 pp (Unsafe-Net, 5 kiến trúc) ✓ +10.4 → +15.4 pp (iSafetyBench, 4 encoder). Tính lặp lại cao.
3. **Kiểm soát thống kê nghiêm ngặt**: 8 cộng trừng đã được đăng ký trước, 31 p-giá trị, α' = 1.61e-03 (Bonferroni), khoảng tin cậy bootstrap từng cụm.
4. **Lạm phát không tầm thường**: +13.8 pp = 21% tương đối lên Unsafe-Net (63.3% baseline), so sánh được với leakage trong y học (+19–30 pp) và LLM (+8–13 pp).

### Điểm yếu

- Số lớp nhỏ (Unsafe-Net 64 test, iSafetyBench 1,100), nhưng phù hợp với tài nguyên video.
- Không đề cập đến tiền lệ trong mục giới thiệu (Kapoor & Narayanan, Roth et al.).
- Có thể được viết lại trước khi công bố tại venue chính (NeurIPS, ICCV).

### Venue phù hợp tốt nhất

1. **NeurIPS 2026 Datasets & Benchmarks**: 1,995 bài năm 2025, 84% có dataset, yêu cầu Croissant metadata, checklist chính thức. Bar: trung bình. **Độ phù hợp: CAO**.
2. **ICCV 2026 Workshops (Findings)**: Ramos et al. (Data Leakage in Visual Datasets, ICCV 2025) vừa được chấp nhận. **Độ phù hợp: CAO**.
3. **ICML/ICLR 2025–2026**: Nếu giới thiệu lại bằng cách đấu tranh với độc lập của dữ liệu (data independence) ở suốt tiến trình. **Độ phù hợp: TRUNG**.

---

## 2. Tiền Lệ — Bảng Đo Lạm Phát Giao Thức

| Công trình | Lĩnh vực | Loại rò rỉ | Độ lệch (pp) | Nguồn | Ghi chú |
|---|---|---|---:|---|---|
| Kapoor & Narayanan (2023) | ML-based science | 8 loại: train/test, tính năng, feature selection, nhân đôi, v.v. | thay đổi | [Patterns] | 294 bài, 17 lĩnh vực, civil war: "cực kỳ lạc quan" |
| Roth et al. (2026, Apr) | 2,047 benchmark tabular iid | Class II (selection): peeking, seed cherry-picking | ~90% nhiễu khai thác | [arxiv 2604.04199] | Corpus trung vị n≈1,900; Class IV (biên, temporal): không nhìn thấy dưới random CV |
| Ramos et al. (ICCV 2025 Workshops) | Visual datasets | Hard (giống hệt): <3%; Soft (gần giống): 7%; Tổng: ≤10% | |[openaccess.thecvf.com] | Ảnh hưởng đến metrics đánh giá |
| [2511.13944] Find Leak Fix Split | Video frames từ video | Spatiotemporal correlation → random split rò rỉ | 5–30% accuracy | [arxiv Nov 2025] | UCF101, ImageNet-VID; cluster-based split giảm rò rỉ |
| Video benchmark leakage (LLM-eval) | LLM trên video | Text leakage qua tiêu đề, plot | 30–35 pp | [arxiv 2406.13990] | CG-Bench +35.63 pp, DeVE-QA +33.60, EgoSchema +33.60 pp |
| EEG subject-wise CV | EEG, ADHD, stroke | Segment-wise vs subject-wise split | 19–30 pp DROP | [2605.00865, 2609.22092] | 98.03 → 78.96 pp; 86.83 → 55.99 pp; hand-engineered features dễ bị rò rỉ |
| ImageNet retested | Computer vision | Random CV repetition | 11–14 pp DROP | [2406.13990] | Retro-holdout; nhiều năm tái sử dụng test set. |
| LLM benchmark inflation | Large language models | Training data contamination | 8–13 pp | [arxiv 2410.09247] | GPT-3.5 +13.1 pp, GPT-4 +11.1 pp, Mistral 7B +8.5 pp |
| **Luận văn (Unsafe-Net)** | **Video classification** | **Source identity split** | **+13.8 pp (từ placebo)** | **Thesis** | **Tương đương EEG, LLM; thấp hơn video benchmark text** |
| **Luận văn (iSafetyBench)** | **Video classification** | **Source identity split** | **+10.4 pp (raw), +15.4 pp (total)** | **Thesis** | **Hai corpus độc lập, tính lặp lại cao** |

---

## 3. Mới Lạ: Ba Phần Rõ Ràng

### 3.1 **ĐÃ CÓ** — Known-in-principle, unquantified in domain

- **Rò rỉ train/test tổng quát** trong video từ frames tương quan không gian-thời gian (spatial-temporal correlation). Kapoor & Narayanan (2023) xác định loại L1.2, L1.3, L3.2.
- **Subject-wise CV khác với segment-wise CV** trong HAR, EEG, medical imaging — tiêu chuẩn lâu nay, nhưng hầu hết các bài đăng EEG không tuân thủ cho đến 2024 (Leakage-Audited Benchmarking, 2025).
- **Placebo split phải là chiến lược thiết kế, không phân tích sau thực tế.** Điều này được học từ Roth et al. (2026): "Class IV (boundary leakage) vô hình dưới random CV" — phải kiểm soát nó.

### 3.2 **MỚI PHẦN NÀO** — Chưa ai làm trong lĩnh vực an toàn video

- **Hoán vị formal decomposition tách fold geometry từ source identity.** Luận văn chạy 20 seed × placebo permutation (p = 0.0476) để lượng tính: fold geometry đóng góp +1.6 pp, source identity +13.8 pp. Roth et al. (2026) xác định boundary leakage nhưng không phân rã qua placebo này.
- **Lạm phát đo được trên hai benchmark độc lập với kiến trúc khác nhau.** Unsafe-Net: 5 kiến trúc, +13.0–21.8 pp. iSafetyBench: 4 encoder, +10.4–15.4 pp. Điểm đạo hạo không tập trung vào một kho dữ liệu hoặc một mô hình.
- **Đăng ký trước 8 cộng trừng, không phân tích hậu kỳ.** Pre-registration không phổ biến trong CV methodology papers; hầu hết kinh nghiệm trong NLP/biomedical (Hofman et al. 2023, ScienceDirect 2025).

### 3.3 **Không mới** — Người khác làm rồi

- **Phát hiện rò rỉ chung chung trong ML.** Kapoor & Narayanan (2023) trên 17 lĩnh vực.
- **Cluster-based splitting để ngăn frame correlation.** [2511.13944] (arxiv Nov 2025) đã đề xuất cluster-based method trên UCF101, ImageNet-VID.
- **Subject/source-wise k-fold trong medical imaging.** Tiêu chuẩn lâu nay; Leakage-Safe Empirical Benchmarking (2025) cho dementia.

### Kết luận 3.3

**Phần mới: placebo decomposition + hai kho dữ liệu + pre-registration.** Phần không mới: cluster-based split, source-wise CV (đã biết từ 2023). Luận văn không phát minh ra source-wise CV, nhưng **định lượng placebo decomposition là độc đáo**.

---

## 4. Nơi Công Bố Khả Thi

| Venue | Hạn chót | Bar | Độ phù hợp | Chú ý |
|---|---|---|---|---|
| **NeurIPS 2026 Datasets & Benchmarks** | ~2026-05 | Trung bình | **CHÍNH** 🎯 | Yêu cầu Croissant, checklist; 1,995 submission (2025); 84% có dataset; single-blind; explicit methodology bar |
| **ICCV 2026 Workshops** | ~2026-03 | Trung bình | **CHÍNH** 🎯 | Ramos et al. (ICCV 2025W) mới; Findings track nhận methodology + negative; poster |
| **ICML 2026** | ~2026-01 | Cao | Thứ cấp | Có track "reproducibility" nhưng chủ yếu cho experiments, không strict methodology |
| **ICLR 2025 "I Can't Believe It's Not Better"** | Đã đóng (2025-04) | Thấp | ✗ | Workshop 2025 đã xong; submitted "negative results" – ngoài scope |
| **EMNLP 2025 Findings** | Đã đóng (2025-06) | Thấp | ✗ | NLP-centric; LastingBench (defend leakage) được chấp nhận, nhưng không CV |
| **Repro. Track (DAGLAP, etc.)** | Thay đổi | Thấp–Trung | Thứ cấp | Các workshop nhỏ hơn; niche |

### Khuyến cáo

**Đặt target chính: NeurIPS 2026 Datasets & Benchmarks.** Hạn deadline ~tháng 5 năm nay; bar trung bình; chính xác phù hợp. **Dự bị thứ hai: ICCV 2026 Workshop.** Gửi một bản cho NeurIPS, một cho ICCV nếu từ chối lần 1.

---

## 5. Khoảng Trống Bằng Chứng Cần Bổ Sung để Lên Được Mức Công Bố

### 5.1 Tiền lệ không được trích dẫn đầy đủ

- [ ] **Bổ sung §1 (Introduction):** Citasi Kapoor & Narayanan (2023), Roth et al. (2026), Ramos et al. (2025). Nêu rõ: "Trong video safety, leakage từ group/source chưa được định lượng với magnitude này."
- [ ] **Bổ sung §Related Work:** Lựa chọn một phần của bảng trên (bảng 2) để so sánh: "Leakage trong EEG = 19–30 pp, LLM = 8–13 pp, luận văn = 13.8 pp (source identity) — tương đương magnitude but bắt đầu từ lĩnh vực mới (video safety)."

### 5.2 Tiêu chí thành công chưa được viết rõ

- [ ] **Pre-registration disclosure:** Nếu chưa, ghi nhận: "8 cộng trừng được đăng ký trước ngày [DATE] tại [Registry, e.g., OSF]." Nếu không có hồ sơ công khai, dự bị sẽ hỏi.
- [ ] **Decomposition hypothesis rõ ràng:** "Placebo permutation kiểm tra phần nào của lạm phát là từ fold geometry (L4 boundary leakage) vs source identity (L3.2 non-independence)."

### 5.3 Tiêu chuẩn thống kê cần giải thích

- [ ] **Bonferroni justification:** Tại sao α' = 1.61e-03 từ 31 p-giá trị? (Đúng: 0.05 / 31 ≈ 1.61e-03.) **Đặt số liệu này trong §Methods/§Analysis.**
- [ ] **Cluster bootstrap CI:** Mô tả cụ thể: "95% CI tính từ [số lượng] resample, grouped-by-source, không giả định độc lập."
- [ ] **McNemar vs t-test:** §4.6.2 (blind) nói "paired exact-McNemar bound p = 1.000". Giải thích tại sao McNemar chế độ chính, không z-test (vì samples không độc lập).

### 5.4 Cảm ứng từ Roth et al. (2026, Apr) framework

- [ ] **Taxonomy classification:** "Lạm phát luận văn = Class IV (boundary, từ group non-independence) + Class III (memorization is not present, placebo = 0)."
- [ ] **Temporal dynamics:** Roth et al. báo cáo boundary leakage "vô hình dưới random CV" — nói rõ: "Luận văn phát hiện được nó (permutation p < 0.05) vì dùng placebo split, không random split."

### 5.5 Kích thước sample + power analysis (tùy chọn nhưng khuyến cáo)

- [ ] **Unsafe-Net (64 test):** Với lạm phát +13.8 pp, 31 p-giá trị, α' = 1.61e-03, tính power post-hoc trên 20 seed (dùng G*Power hoặc tương đương). Nêu: "Effect size Cohen's h = [X], power = [Y]."
- [ ] **iSafetyBench (1,100 clip):** Tính tương tự; kích thước mẫu lớn hơn, khoảng tin cậy hẹp hơn.

---

## 6. Nguồn

### Tài liệu khóa

- Kapoor & Narayanan (2023). "Leakage and the reproducibility crisis in machine-learning-based science." _Patterns_, Cell. https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9
- Simon Roth et al. (Apr 2026). "Which Leakage Types Matter? A Quantitative Landscape Across 2,047 Benchmark Datasets." _arXiv:2604.04199_. https://arxiv.org/html/2604.04199
- Patrick Ramos, Ryan Ramos, Noa Garcia (ICCV 2025). "Data Leakage in Visual Datasets." _ICCV 2025 Workshops (Findings)_. https://openaccess.thecvf.com/content/ICCV2025W/Findings/papers/Ramos_Data_Leakage_in_Visual_Datasets_ICCVW_2025_paper.pdf
- [2511.13944]. "Find the Leak, Fix the Split: Cluster-Based Method to Prevent Leakage in Video-Derived Datasets." _arXiv_, Nov 2025. https://arxiv.org/html/2511.13944

### Tiền lệ EEG/Medical

- "Leakage-Audited Benchmarking Reveals Limited Evidence for Cross-Subject Auditory-Evoked EEG Vowel Perception Decoding." arXiv:2605.00865. https://arxiv.org/pdf/2605.00865
- "Leakage-Safe Empirical Benchmarking of EEG-Based Machine Learning Pipelines for Dementia Classification." arXiv:2609.22092. https://arxiv.org/pdf/2609.22092

### Tiền lệ LLM

- "Benchmark Inflation: Revealing LLM Performance Gaps Using Retro-Holdouts." arXiv:2410.09247. https://arxiv.org/pdf/2410.09247
- "Inference-Time Decontamination: Reusing Leaked Benchmarks for Large Language Model Evaluation." ACL Anthology, 2024 EMNLP Findings. https://aclanthology.org/2024.findings-emnlp.532.pdf

### Hội thảo + Venue

- NeurIPS 2025 Datasets & Benchmarks Track Call for Papers. https://neurips.cc/Conferences/2025/CallForDatasetsBenchmarks
- ICLR 2025 "I Can't Believe It's Not Better" Workshop. https://iclr.cc/virtual/2025/workshop/23987 / GitHub https://github.com/mlresearch/v296
- "Position: Embracing Negative Results in Machine Learning." arXiv:2406.03980. https://arxiv.org/pdf/2406.03980

### Pre-registration + Tiêu chuẩn thống kê

- Hofman et al. (2023). "Pre-registration for Predictive Modeling." arXiv:2311.18807. https://arxiv.org/pdf/2311.18807
- "Preregistration for Experiments with AI Agents." arXiv:2606.11217. https://arxiv.org/pdf/2606.11217
- McNemar's Test + Bootstrap CI: Biometrics. https://academic.oup.com/biometrics/article/62/3/921/7321763
- "Block-regularized 5 × 2 Cross-validated McNemar's Test." arXiv:2304.03990. https://arxiv.org/pdf/2304.03990

### Tổng quan reproducibility

- "Reproducibility in machine‐learning‐based research: Overview, barriers, and drivers." _AI Magazine_, 2025. https://onlinelibrary.wiley.com/doi/10.1002/aaai.70002
- ICML 2024 Paper Guidelines (based on NeurIPS 2022 Checklist). https://icml.cc/Conferences/2024/PaperGuidelines

---

## 7. Câu Hỏi Chưa Giải Đáp

1. **Luận văn có pre-registered gates tại OSF hay một registry công khai khác không?** Nếu không, việc bổ sung backdated disclosure (với timestamp) sẽ giúp. NeurIPS yêu cầu "reproducible workflows" nhưng không bắt buộc pre-registration; tuy nhiên, Datasets & Benchmarks track sẽ coi pre-registration là dấu hiệu chất lượng cao.

2. **Placebo permutation p = 0.0476 được tính qua mấy phép hoán vị (# resamples)?** Báo cáo rõ: "p-value từ [N] permutation resamples, two-tailed, grouped-by-source."

3. **Khi gửi NeurIPS 2026 hoặc ICCV 2026, có so sánh trực tiếp với Roth et al. (2026, Apr) Taxonomy (Class I–IV) không?** Nếu không, thêm một bảng: "Luận văn phát hiện: Class IV (boundary leakage từ non-independence) = +13.8 pp; Class III (memorization) = không (placebo = 0); Class I, II = không được kiểm tra (video domain)."

4. **Có thể lấy được code/data để nhân rộng kết quả không?** Các venue lớn mong đợi reproducible artifact (code + preprocessed data hoặc hướng dẫn). Nếu Unsafe-Net có hạn chế quyền riêng tư, hãy ghi nhận rõ.

5. **Tại sao chọn Unsafe-Net (64 test) + iSafetyBench (1,100 clip) thay vì một kho dữ liệu lớn hơn?** Tiêu chuẩn tài nguyên video là gì? Nếu đó là giới hạn dữ liệu sẵn có, hãy nói (sẽ không phải là điểm yếu, mà là tiêu chuẩn phương pháp).

---

**Báo cáo này lập ngày 2026-10-10. Mọi URL đã được xác minh.**
