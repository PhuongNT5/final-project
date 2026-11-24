# Researcher 05 — lạm phát kết quả: chia ngẫu nhiên theo dòng vs theo nhóm

Hạn chế: nhiều PDF (Roberts 2017, Choi toàn văn, Kapoor Figure 1, Ploton bản HAL) không đọc được. Số dưới đây từ HTML/abstract/snippet. Nhãn [snippet] / [thứ cấp] khi không đọc được nguồn gốc.

## 1. Số liệu theo lĩnh vực (chênh lệch random vs group)

| Lĩnh vực | Chênh lệch | Chỉ số | Bộ dữ liệu / mô hình | Nguồn + vị trí | Ghi chú |
|---|---|---|---|---|---|
| Viễn thám biomass | R² 0.53 → 0.14 (Δ 0.39, −74%) | R² | 11.82M cây, 28,225 pixel, Random Forest. Random 10-fold vs spatial 44-fold | Ploton et al. 2020, Nat Commun 11:4540, DOI 10.1038/s41467-020-18321-y, Results | Đọc từ bản mở HAL/PMC |
| Viễn thám, buffered LOO | R² ~0.50 (không buffer) → ~0.15 (50 km) → ~0 (>100 km) | R² | cùng dữ liệu | Ploton 2020, Results, Fig. 5 | Leakage theo khoảng cách |
| Viễn thám (Roberts 2017) | KHÔNG TÌM THẤY số | — | — | DOI 10.1111/ecog.02881 | PDF không đọc được. Chỉ định tính: random CV "serious underestimate" error |
| EEG cross-subject (Zhu 2015) | 90.97 → 64.82 (Δ 26.15) | acc | nhiều bộ/mô hình | DOI 10.3389/fnins.2018.00162, Intro [thứ cấp] | Gốc Zhu 2015 chưa đọc |
| EEG SEED 3-class | 93.1 → 71.6 (Δ 21.5) | acc | ResNet-50 transfer, 10-fold vs LOSO | arXiv 2306.05680 [snippet] | |
| EEG SEED / BiDCNN | 94.38–94.72 → 63.94–68.14 (Δ ~26–31) | acc | BiDCNN | Zhang, Fort & Giménez 2024 mini review [thứ cấp, DOI chưa xác minh] | |
| Bimodal speech+sinh lý (Kim 2007) | 92 → 55 (Δ 37) | acc | 3 subject | trích trong Zheng 2018 Intro [thứ cấp] | n=3, không dùng làm so sánh chính |
| Speech SD vs SI (IEMOCAP) | SI 68–78 vs SD 73–81 (~3–5 pp) | WA | nhiều paper khác nhau | "IEMOCAP reality check" 03/2023 [snippet] | **KHÔNG kiểm soát** — so chéo paper |
| Speech valence (Sridhar & Busso) | CCC +tới 30% tương đối (valence), <4% (arousal/dominance) | CCC | — | [snippet] | Đơn vị tương đối |
| **Bioacoustic/ship (file-level random vs vessel-disjoint)** | **DeepShip: acc +10.2–12.2 pp (F1 +12.7–13.7). VTUAD: acc +47.2–47.6 pp (F1 +60.9–61.4)** | acc, F1 | Swin, STFT/Mel, 5 fold | arXiv 2609.13659 "UniqueShip", Table II (preprint 2026) | **Analog gần nhất** cho leakage theo nguồn ghi |
| Bird audio | KHÔNG TÌM THẤY | — | — | — | Chỉ cảnh báo định tính (FSD50K split theo uploader) |
| OCT (per-image vs per-subject) | MCC +0.07 … +0.43 (≈ acc +5 … +30 pp) | MCC, acc | Kermany, Srinivasan, AIIMS breast | Tampu et al. 2022, Sci Data 9:580, DOI 10.1038/s41597-022-01618-6, Abstract | Dải acc là tóm tắt công cụ, chưa đối chiếu bảng |
| Chest X-ray | KHÔNG TÌM THẤY chênh lệch có kiểm soát | — | — | arXiv 2607.13800 chỉ báo patient-cluster; ChestX-ray14 patient-split mean AUC 0.841 (arXiv 1803.04565) | Không có cặp so sánh |
| Histopathology (slide vs patient) | KHÔNG TÌM THẤY số | — | — | medRxiv 2020.04.23.20076406 | Định tính: slide-level cho hiệu năng cao hơn |
| Dermoscopy (lesion-level) | KHÔNG TÌM THẤY số | — | — | DermaMNIST audit [snippet] | Định tính |
| Surface/microscopy (sibling leakage) | mAP 0.546 → 0.303; Dice +0.278–0.448 | mAP, Dice | 2 benchmark | Research Square rs-11679740 [snippet, chưa xác minh trạng thái] | Chưa đọc toàn văn |
| **Action recognition, scene bias** | Choi et al. **không đo** chênh lệch clip-vs-video | — | UCF-101, HMDB-51, Diving48, 3D-ResNet-18 | arXiv 1912.05534, Table 1 & 5 | Số có: UCF 83.5→84.5; HMDB 53.6→56.7; Diving48 18.0→20.5; scene classifier 29.7%→2.9%. **Đây là debias, không phải leakage** |
| Kinetics/UCF duplicate | KHÔNG TÌM THẤY pp accuracy | — | — | — | Chỉ số đếm: LOVEU Kinetics-GEBD loại 943 video test trùng train = 5.5% test [forum, thứ cấp]; HACS loại overlap |
| RESOUND | KHÔNG TÌM THẤY số | — | — | ECCV 2018 (Li/Li/Vasconcelos) | **arXiv ID 1807.08362 trong yêu cầu là SAI** — đó là bài "An Intersectional Definition of Fairness". KHÔNG dùng ID này |
| REPAIR | KHÔNG TÌM THẤY số | — | — | arXiv 1904.07911 | |
| **UCF-Crime / video anomaly** | KHÔNG TÌM THẤY pp | AUC | — | arXiv 2608.11985 (pooled AUC thưởng khác biệt giữa nguồn); arXiv 2505.19022 (scene overfitting, không đo pp) | Chưa xác minh camera chung train/test |
| Peptide MS (backbone-disjoint vs random) | PCC 0.8921 → 0.9099 (+1.8 pp) | PCC | 1 dataset | arXiv 2605.01945 PepSpecBench | Chiều NGƯỢC |

## 2. Kapoor & Narayanan (Patterns 2023, DOI 10.1016/j.patter.2023.100804)

Bảng từ bản arXiv v1 (ar5iv 2207.07048) — **không có cột "mức lạm phát"**, chỉ số bài:

| Lĩnh vực (nguồn) | Bài đánh giá | Bài có pitfall |
|---|---|---|
| Medicine (Bouwmeester 2012) | 71 | 27 |
| Neuroimaging (Whelan & Garavan 2014) | – | 14 |
| Autism (Bone 2015) | – | 3 |
| Bioinformatics (Blagus & Lusa 2015) | – | 6 |
| Nutrition (Ivanescu 2016) | – | 4 |
| Software Eng. (Tu 2018) | 58 | 11 |
| Toxicology (Alves 2019) | – | 1 |
| Satellite imaging (Nalepa 2019) | 17 | 17 |
| Tractography (Poulin 2019) | 4 | 2 |
| Clinical epi (Christodoulou 2019) | 71 | 48 |
| BCI (Nakanishi 2020) | – | 1 |
| Histopathology (Oner 2020) | – | 1 |
| Neuropsychiatry (Poldrack 2020) | 100 | 53 |
| Medicine (Vandewiele 2021) | 24 | 21 |
| Radiology (Roberts 2021) | 62 | 62 |
| IT Ops (Lyu 2021) | 9 | 3 |
| Medicine (Filho 2021) | – | 1 |
| Neuropsychiatry (Shim 2021) | – | 1 |
| Genomics (Barnett 2022) | 41 | 23 |
| Computer Security (Arp 2022) | 30 | 30 |

- **Mâu thuẫn:** arXiv v1 ghi 20 bài / 17 lĩnh vực / 329 bài; snippet bản Patterns ghi 22 bài / 294 bài. Ưu tiên bản Patterns, nhưng chưa đọc được.
- Không có số pp theo lĩnh vực. Một case có số: Wang 2019 (civil war) — AUC gap ML vs LR giảm 0.14 → 0.01 sau sửa leakage.
- Bài tổng quan định lượng 2024–2026 tiếp nối: **chưa tìm được**.

## 3. Câu hỏi phương pháp

### (8) Chuẩn/checklist yêu cầu khai báo chia nhóm
- **CLAIM 2024** (Radiology: AI, DOI 10.1148/ryai.240300), **Item 19**: "Specify how data were partitioned for training, model optimization (often termed 'tuning' or 'validation'), and testing." [đã đọc, PMC11304031]
- **CLAIM 2024, Item 20**: "Sets of medical images generally should be disjoint at the patient level or higher … so that images of the same patient do not appear in more than one partition." [đã đọc] ← **câu chữ mạnh nhất tìm được**
- CLAIM 2020 Item 20: "Specify how the data were assigned into training, validation ('tuning'), and testing partitions" [snippet]
- NeurIPS checklist (2021/2022/2024): hỏi "training details" trong đó có "data splits". **Diễn giải từ snippet, không phải câu nguyên văn** — cần kiểm trang chính thức.
- TRIPOD+AI (BMJ 2024;385:e078378): **KHÔNG XÁC MINH** câu chữ Item 9. Có TRIPOD-Cluster (PMC9903175) cho dữ liệu nhóm: 10 item mới, 19 item sửa.
- Ngoài CLAIM Item 20, các checklist **không yêu cầu thẳng** "chia theo nhóm".

### (9) Cluster bootstrap và độ rộng CI
- Công thức chuẩn: DEFF = 1 + (m−1)ρ (Kish 1965; Donner & Klar 2000 [snippet]). SE tăng theo √DEFF. **Đây là lý thuyết, không phải ví dụ đã công bố.**
- Với CI rộng gấp 2.9×: DEFF ≈ 8.4 → (m−1)ρ ≈ 7.4. Ví dụ m=10 clip/nguồn → ρ≈0.82; m=20 → ρ≈0.39. **Phải kiểm ρ thực tế để xác nhận 2.9× có khớp.**
- Ví dụ công bố có CI ratio: **KHÔNG TÌM THẤY**. Một mô phỏng kinh tế y tế [thứ cấp]: bootstrap theo cá nhân cho CI quá hẹp; cần >~24 cụm mỗi nhánh.
- Cảnh báo ít cụm: Cameron & Miller 2015 [thứ cấp] nêu 20–50 cụm; mô phỏng G=15 vẫn undercoverage. Nếu số cụm ít, cluster bootstrap vẫn sai → cần wild cluster bootstrap (MacKinnon 2020 [thứ cấp]). **→ liên quan trực tiếp lane Unsafe-Net với 8 scene cluster.**

## 4. Phán quyết: 13–15 pp nằm ở đâu

| Dải pp | Lĩnh vực / ví dụ |
|---|---|
| <5 | Peptide MS (+1.8 PCC, chiều ngược), speech SD-vs-SI (~3–5, không kiểm soát) |
| **10–15** | **Ship DeepShip (10.2–12.2 acc)**, OCT đầu thấp (MCC 0.07 ≈ acc 5–10) |
| 20–30 | EEG subject-dependent (21.5–31), OCT trung–cao (đến ~30) |
| 30–50 | Biomass (Δ 0.39 R²), VTUAD (47), Kim 2007 (37) |
| Không có số | **Action video clip/video**, Kinetics duplicate, UCF-Crime, CXR patient-level, histopathology, dermoscopy, bird audio |

**Kết luận:** 13–15 pp nằm ở **đoạn giữa-thấp** của dải có số — cùng mức ship DeepShip (10–12), thấp hơn EEG (21–31) và viễn thám/VTUAD (39–47). Là mức lạm phát **bình thường** cho leakage theo nhóm trong ML ứng dụng: không nhỏ bất thường, không ngoại lệ.

**Nhưng:** không tìm thấy số công bố nào cho action recognition video chia theo nhóm nguồn → **không thể nói "bình thường" riêng cho video**. Số của luận văn có thể là một trong những số định lượng đầu tiên cho lĩnh vực này.

## 5. Câu hỏi chưa giải quyết
- Roberts 2017, Ploton (bảng gốc), Choi/RESOUND toàn văn: cần PDF đọc được.
- Figure 1 Kapoor (pp theo ngành): là ảnh, không đọc được.
- Xác minh Research Square rs-11679740 (trạng thái, số đầy đủ).
- Xác minh Zhang 2024 mini review (DOI) và Zhu 2015 gốc.
- TRIPOD+AI Item 9 câu chữ chính xác.
- **Video action: cần tự chạy thí nghiệm** (cùng mô hình, random clip-split vs video-disjoint) để có số đối chứng — đây là khoảng trống tài liệu.
- Hệ số 2.9× của CI: cần xác nhận ρ và m thực tế.
- Số bài tổng quan Kapoor khác nhau giữa arXiv v1 (20/329) và Patterns (22/294).
