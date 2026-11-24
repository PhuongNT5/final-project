# Researcher 06 — bảng hệ thống: giám sát an toàn / hành vi không an toàn bằng CV, 2023–2026

## 0. Kết luận nhanh
- Mendeley `xjmtb22pff` là bộ dữ liệu **đồng hành** của Unsafe-Net (cùng tác giả, bản "Original data" của Data in Brief). Xác nhận.
- **Số "1,901 + 343" KHÔNG TÌM THẤY** trong luận án gốc của Önal (2024). Luận án ghi **2,262 frame train và 329 frame test** được gán bbox (Tablo 3.4, tr.38). → 2,262 đúng; 343 **không có nguồn**.
- Bài MTA (Springer) **không mở được full text** (paywall). Mọi số Unsafe-Net dưới đây lấy từ **luận án** của cùng tác giả (cùng dữ liệu, cùng thí nghiệm), trừ khi ghi khác.
- **Không tìm thấy công trình nào ngoài nhóm tác giả** huấn luyện/đánh giá trên bộ Önal với số liệu. iSafetyBench chỉ trích dẫn (Table 1).
- **Không công trình nào có đủ cả 4:** edge + năng lượng đo thật, CI, phép chia ràng buộc nhóm, code+data công khai.

## 1. Bảng A: dữ liệu, bài toán, kiến trúc, phép chia

| # | Công trình | Dữ liệu (N, loại, công khai?) | Bài toán / kiến trúc | Phép chia | Phán quyết | Ràng buộc nhóm? |
|---|---|---|---|---|---|---|
| 1 | **Unsafe-Net**, Önal & Dandıl, MTA 84(29) 2025, DOI 10.1007/s11042-024-19276-8 | 691 clip, 8 lớp (4 an toàn / 4 không), **2 IP cam, 1 nhà máy** Eskişehir. Công khai Mendeley DOI 10.17632/xjmtb22pff.1, CC BY 4.0. Bbox: **2,262 train / 329 test** (luận án Tbl 3.4) | YOLOv4 phát hiện (8 lớp) + ConvLSTM phân loại video (4 lớp không an toàn) | Train 566 / test 125 clip (DIB Tbl 2). Test phân loại: **64 clip** (32/11/13/8) | **(c)** bộ khác, tác vụ video 4 lớp | **KHÔNG nêu.** Chia theo clip có sẵn, cùng cam / cùng đợt thu → coi là **KHÔNG ràng buộc** |
| 1b | Luận án Önal 2024 (TR), PPE 7 lớp, cùng tác giả | PPE riêng, ban đầu 1,000 ảnh | YOLOv4 phát hiện PPE | Run 5: 888 train / 112 val (Tbl 4.2) | (c) | KHÔNG nêu |
| 2 | Nath et al. 2020, AutoCon, DOI 10.1016/j.autcon.2020.103085 | Pictor-v3: 1,472 ảnh (774 crowdsourced + 698 web). Công khai GitHub ciber-lab/pictor-ppe (license không nêu) [chưa xác minh là bản trong bài] | YOLOv3, 3 lớp worker/hat/vest | Không nêu | (d) ảnh, bộ khác | Không nêu |
| 3 | Sec-YOLO / FEHA, PeerJ CS 2025, cs-3151 (PMC12453777) | 8,799 ảnh, 4 lớp (ngã, ngủ trực, điện thoại, không mũ). Công khai: data Zenodo 10.5281/zenodo.14015767, code Zenodo 10.5281/zenodo.14233625 | Phát hiện per-frame (YOLO + FEHA attention) | **Ngẫu nhiên 7:2:1** | (d) | **KHÔNG** (ngẫu nhiên ảnh) |
| 4 | Mohy et al., ITcon 30 (2025) 1398–1430, DOI 10.36680/j.itcon.2025.057 | 55,594 ảnh, 28 nhóm — **dữ liệu riêng** | One-stage + CoTNet/Triplet attention/ACON/CARAFE, WIoU | Không xác minh | (d) | Không nêu |
| 5 | Springer MTA 2026, s11042-026-21823-4 | Colors Helmet & Vest | YOLOv8 + attention, PPE | [chưa xác minh] | (c) | Không xác minh |
| 6 | ASCE JCEMD, DOI 10.1061/JCEMD4.COENG-14985 | Chưa xác minh; PPE chống ngã | Improved YOLOv8 | [chưa xác minh] | (c)/(d) | Không xác minh |
| 7 | **iSafetyBench**, arXiv 2508.00399 (ICCV-W VISION'25) | **1,100 clip** (680 normal / 420 hazard), YouTube, công khai, CC BY 4.0 | Zero-shot VLM, MCQ đa đáp án | **Không có train/test** (toàn zero-shot) | **(b)** cùng 1,100 clip, khác tác vụ | Không áp dụng |
| 8 | MonitorVLM, arXiv 2510.03666 | 9,000 VQA, 40 quy định, **MỎ** (không phải xây dựng) | VLM 72B + clause filter + behavior magnifier | Không nêu | (d) | Không nêu |
| 9 | Yang et al., AutoCon 2022, DOI 10.1016/j.autcon.2022.104703 | 1,595 clip, 7 nhóm hành vi | STR-Transformer | Không xác minh | (d), **xuất bản 2022: ngoài khoảng** | Không nêu |
| 10 | Shahnavaz et al., ASCE I3CE 2023, arXiv 2308.14843 | Tự thu, không nêu số | Transfer learning (Kinetics-400), MMH | Không nêu | (d) | Không nêu |
| 11 | LS-HAR, arXiv 2410.01962 | VolvoConstAct (visual/skeleton/depth), công khai; eval NTU RGB+D | Language-supervised HAR | Không nêu | (d) | Không nêu |
| 12 | Adil et al., arXiv 2504.09083 | Không truy cập được (404) | GPT-VLM nhận diện mối nguy | — | (d) | — |

## 2. Bảng B: chỉ số, phần cứng, tốc độ, năng lượng, CI

| # | Công trình | Chỉ số (ghi rõ loại) | Phần cứng / tốc độ | Năng lượng | CI |
|---|---|---|---|---|---|
| 1 | Unsafe-Net (luận án Tbl 4.8–4.9; abstract) | **"Overall" 95.81% = TRUNG BÌNH KHÔNG TRỌNG SỐ của 4 độ chính xác lớp (100; 90.91; 92.31; 100)** — tự kiểm = 95.805. Gộp pooled: **62/64 = 96.88%**. ConvLSTM-only macro 85.78%, pooled 57/64 = 89.06%. mAP@0.5 YOLOv4 test = 95.66% (329 frame, Tbl 4.5); val 91.18% (Tbl 4.2) | **2× GTX 1080Ti, i9-9900, 16 GB** (luận án §4, tr.46). **0.14 s "eylem tanıma"** — phần cứng KHÔNG nêu rõ. **1.87 s = độ dài clip trung bình sau lọc, KHÔNG phải latency.** FPS hệ thống: không nêu | **Không** | **Không.** Tự tính Wilson 95% cho 62/64: **[89.3; 99.1]** |
| 1b | Luận án PPE YOLOv4 | mAP@0.5 91.18; P 0.89; R 0.91; F1 0.90; IoU 70.35. Tbl 4.2 cho thấy run 91.18 dùng 888/112 → **nhiều khả năng là VAL, không phải test**; abstract ghi "test" [mâu thuẫn] | 32 FPS, workstation, **không edge** | Không | Không |
| 2 | Nath 2020 | mAP best 72.3% (Approach-2); A1 63.1% — **nguồn thứ cấp**, chưa xác minh trong bài | 11 FPS (A2), 13 FPS (A1), laptop; nguồn thứ cấp | Không | Không |
| 3 | Sec-YOLO | mAP50 92.6; mAP50-95 63.6; P 91.3; R 86.5 (Tbl 5/6). **Tbl 2 ghi P 92.2 — mâu thuẫn nội bộ.** TPR theo lớp (Fig 9): phone 94, sleep 89, helmet 88, fall 86 | RTX 3090; **FPS/latency KHÔNG báo cáo**; không edge | Không | Không |
| 4 | Mohy 2025 | Precision 0.89; mAP95 0.45 (abstract). mAP50 không có | [chưa xác minh] | Không | Không |
| 5 | MTA 2026 PPE | +1.3 / +3 pp mAP50 / mAP50-95 so YOLOv8s. **Số tuyệt đối KHÔNG TÌM THẤY** | Jetson Orin Nano, 30 FPS [snippet, chưa xác minh] | Không | Không |
| 6 | JCEMD | mAP50 92.52% [snippet, chưa xác minh] | Jetson Xavier NX, 9.11 FPS [snippet, chưa xác minh] | Không | Không |
| 7 | iSafetyBench (Tbl 2) | Single-correct Acc: Ovis2-8B avg 43.8; GPT-4o 38.8. Multi-correct F1: Ovis2-8B 49.8; GPT-4o 48.35. **Đây là MCQ, KHÔNG phải nhị phân** | Không báo cáo | Không | Không |
| 8 | MonitorVLM | **CHỈ tương đối** so baseline 72B: +22.01% P, +34.22% R, +28.37% F1; clause filter giảm latency 13.56%. Số tuyệt đối KHÔNG TÌM THẤY trong phần đọc được | Không báo cáo | Không | Không |
| 9 | Yang 2022 | AP 88.7% [snippet, chưa xác minh] | Không xác minh | Không | Không |
| 10–11 | Shahnavaz 2023; LS-HAR | KHÔNG TÌM THẤY số trong abstract | — | — | — |

## 3. Trả lời từng câu hỏi

**(1) Unsafe-Net và bộ dữ liệu**
- Bảng theo lớp: có trong luận án (Tbl 4.8, 4.9). Bài MTA không truy cập được.
- **Định nghĩa "Overall": trung bình KHÔNG trọng số của độ chính xác từng lớp (macro).** Pooled 96.88% là thông tin phụ. → khớp cách đọc trong README.
- Phần cứng: 2× GTX 1080Ti + i9-9900 cho thí nghiệm. 0.14 s không nêu rõ phần cứng. **FPS hệ thống không báo cáo.**
- Phép chia 566/125: **cố định theo clip** (DIB Tbl 2; luận án Tbl 3.3). Không nêu ràng buộc theo cam, phiên, ngày, công nhân. Cùng 2 cam, cùng tòa nhà, cùng đợt thu → **KHÔNG ràng buộc nhóm** (suy luận từ mô tả; không có câu nào nói rõ).
- Bbox: **2,262 (train) / 329 (test)** — luận án Tbl 3.4. **KHÔNG có "1,901 + 343".**
- **Công trình khác dùng bộ Önal/Mendeley: KHÔNG TÌM THẤY bằng số liệu.** iSafetyBench trích dẫn ở Tbl 1 (so sánh, không dùng video). FiftyOne/Voxel51 và Hugging Face là bản mirror, không phải nghiên cứu. Chưa truy cập được danh sách trích dẫn (Scholar/ScienceDirect/Semantic Scholar trả 403) → kết luận chỉ đúng trong phạm vi tìm kiếm.

**(2) PPE công trường 2023–2026 có mAP + edge:** đã có mAP: Sec-YOLO, Mohy, MTA 2026, JCEMD. **Chỉ JCEMD và MTA 2026 có số edge, cả hai đều là snippet chưa xác minh.** Nath 2020 là mốc: 72.3% (nguồn thứ cấp).

**(3) Hành vi không an toàn bằng Transformer/CLIP/VLM 2024–2026:** VLM zero-shot iSafetyBench (Ovis2 43.8 acc — không so được với nhị phân); VLM trên mỏ MonitorVLM (chỉ tương đối); Transformer video xây dựng Yang 2022 (ngoài khoảng); Shahnavaz 2023 không có số. **Không tìm thấy bài CLIP-đóng-băng + head nhỏ trên hành vi an toàn công nghiệp có số liệu.**

**(4) Đối chiếu tiêu chí**
- (a) **Edge + số đo thật: RẤT ÍT** — chỉ 2 snippet (JCEMD Xavier NX 9.11 FPS; MTA 2026 Orin Nano 30 FPS), đều chưa xác minh. **Không công trình nào đo năng lượng.**
- (b) **Khoảng tin cậy: KHÔNG công trình nào báo cáo.** Chỉ tự tính được cho Unsafe-Net: [89.3; 99.1] (Wilson, pooled 62/64).
- (c) **Phép chia ràng buộc nhóm: KHÔNG tìm thấy ở đâu.** Mọi phép chia được nêu là ngẫu nhiên hoặc cố định theo clip/ảnh.
- (d) **Code + data công khai: ÍT.** Sec-YOLO (cả hai, Zenodo, xác minh). iSafetyBench + Unsafe-Net (chỉ dữ liệu; **code Unsafe-Net KHÔNG TÌM THẤY**). Pictor (code + ảnh, license không nêu). LS-HAR (project page).

**(5) Survey / review 2024–2026** (tất cả chỉ snippet; MDPI/ScienceDirect/Tandfonline trả 403)
- Tandfonline 2026, DOI 10.1080/08839514.2026.2739540: manufacturing/construction có nhiều dataset công khai; **thiếu dataset an toàn chuyên biệt**; edge AI là hướng mới.
- MDPI Computers 15(6) 2026, DOI 10.3390/computers15060388: **thiếu dataset có nhãn khớp môi trường công nghiệp địa phương**; detector PPE nhẹ cho edge đang phát triển.
- ScienceDirect 2025, S2666165925002327: **thiếu tổng quát hóa dataset, thiếu chuẩn hóa metric, thiếu kiểm chứng ngoài thực tế**.
- Xác minh được từ DIB §1: "Most occupational health and safety studies are conducted in the construction industry"; nhà máy kín khó thu dữ liệu.
- Xác minh được từ iSafetyBench: VLM yếu nhất ở hành vi nguy hiểm và multi-label.
- **KHÔNG review nào nêu vấn đề chia dữ liệu theo nhóm (leakage) hoặc CI.**

## 4. Phán quyết so với luận văn
- **Không dòng nào là (a)**: không công trình nào trùng cả bộ dữ liệu + tác vụ nhị phân + phép chia leave-source-out.
- **(b):** iSafetyBench (cùng 1,100 clip, khác tác vụ MCQ zero-shot). Số của họ không đo cùng một thứ.
- **(c):** Unsafe-Net, luận án PPE Önal, MTA 2026, JCEMD, Sec-YOLO.
- **(d):** Nath, Mohy, MonitorVLM, Yang, Shahnavaz, LS-HAR, Adil.
- **Phép chia KHÔNG ràng buộc nhóm:** Unsafe-Net (1), luận án PPE (1b), Sec-YOLO (ngẫu nhiên ảnh), Nath, Mohy, MTA 2026, JCEMD. → **các số này không đọc được cạnh con số leave-source-out của luận văn.**

## 5. Câu hỏi chưa giải quyết
1. Bài MTA gốc: bảng kết quả, bbox, định nghĩa "overall", FPS hệ thống — cần full text Springer.
2. **Số ngày thu thập không nhất quán:** 19 ngày (luận án §3), 39 ngày (DIB abstract), "khoảng 40 ngày" (luận án kết luận). Ngày: 5/11–13/12/2022 (DIB) vs 5–23/11 (luận án Tbl 3.3).
3. 0.14 s "eylem tanıma" đo trên phần cứng nào?
4. 91.18 mAP là val (Tbl 4.2) hay test (abstract)? Mâu thuẫn trong luận án.
5. Sec-YOLO: P 92.2 (Tbl 2) vs 91.3 (Tbl 5/6). Mâu thuẫn nội bộ.
6. Nath 2020: 72.3% và FPS chưa xác minh từ bài báo.
7. MTA 2026, JCEMD, Yang 2022: chỉ snippet.
8. Danh sách trích dẫn đầy đủ của bộ Önal/DIB: chưa truy cập được (403).
9. Full text 3 review 2026: chưa truy cập được.
10. Mã Unsafe-Net: không tìm thấy kho mã công khai.

## 6. Nguồn
- Unsafe-Net (MTA, paywall): link.springer.com/article/10.1007/s11042-024-19276-8
- **Luận án Önal 2024 (TR, mở, đã đọc tr.ii–iii, 36–38, 46, 49, 54–56, 62–63, 68–72):** acikkaynak.bilecik.edu.tr/bitstreams/d79903ef-6649-4f1c-be78-b5b92c0cfbfa/download
- DIB dataset paper (đã đọc PDF tr.1–6): acikkaynak.bilecik.edu.tr/bitstreams/dc6b0007-eb9b-4496-9d00-81ecb68fa9b4/download ; DOI 10.1016/j.dib.2024.110791
- Mendeley: data.mendeley.com/datasets/xjmtb22pff/1
- Voxel51 mirror (566/125 xác nhận; DOI 110756 là lỗi đánh máy): docs.voxel51.com/dataset_zoo/datasets_hf/safe_and_unsafe_behaviours.html
- Sec-YOLO: peerj.com/articles/cs-3151 ; PMC12453777
- Nath 2020 repo (thứ cấp): github.com/ciber-lab/pictor-ppe ; DOI 10.1016/j.autcon.2020.103085
- Mohy ITcon 2025: itcon.org/papers/2025_57-ITcon-Mohy.pdf
- MTA 2026 PPE [snippet]: link.springer.com/article/10.1007/s11042-026-21823-4
- JCEMD [snippet]: ascelibrary.com/doi/10.1061/JCEMD4.COENG-14985
- iSafetyBench: arxiv.org/abs/2508.00399 ; html/2508.00399v2 (Tbl 2)
- MonitorVLM: arxiv.org/abs/2510.03666 · LS-HAR: arxiv.org/abs/2410.01962 · Shahnavaz: arxiv.org/abs/2308.14843
- Survey [snippet]: tandfonline.com/doi/full/10.1080/08839514.2026.2739540 ; doi.org/10.3390/computers15060388 ; sciencedirect.com/science/article/pii/S2666165925002327
- Yang 2022 [snippet]: doi.org/10.1016/j.autcon.2022.104703
