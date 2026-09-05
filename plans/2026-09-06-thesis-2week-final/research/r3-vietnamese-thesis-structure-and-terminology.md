# R3 — Vietnamese Thesis Structure, Terminology, Framing, Defence (Writing Playbook)

Sources read: `PROPOSAL.txt`, `plans/2026-09-06-thesis-2week-final/brief.md` (binding, dated 2026-09-06),
`report/*.md` (9 files). Web search confirmed format conventions (Times New Roman 13, 1.5 line spacing,
margins 3.5/3/3.5/2 cm, A4 — Circular 23/2021 & 18/2021 let each university set exact rules; check the
actual faculty template before final formatting) and defence timing (talk ~15-20 min, Q&A ~15 min,
0-3 questions/examiner).

**Critical finding before anything else:** `report/*.md` chapters carry THREE generations of numbers.
`brief.md` (2026-09-06, locked) is the only authoritative number source. Chapter-07 (`multi-entity-and-
scene-robust-evaluation.md`) already matches brief.md's P3 numbers (54/64, 84.4% top-1, 88.4% macro) —
reuse it as the backbone for Ch3/Ch4. Chapters 03/04/05/06 and `thesis-reframe-deployment-led.md` carry
STALE P1/P2 numbers (56/64 89.28%, 53/64 85.4%, Jetson Orin Nano Super old benchmarks) and a more
negative framing ("does not beat even no-bbox baseline") that brief.md's axis-1 table now contradicts
(P3 88.4% macro DOES beat the 85.78% no-bbox baseline). **Every number and framing sentence carried over
from these files must be checked against brief.md §1/§2/§4 before translation — do not translate stale
numbers.**

---

## Part A — Vietnamese CNTT master's thesis: standard structure & format

**Front matter (unnumbered, roman numerals i, ii, iii…):**
Trang bìa chính · trang phụ bìa · Lời cam đoan (1p) · Lời cảm ơn (1p) · Mục lục (2-3p) ·
Danh mục ký hiệu, chữ viết tắt (1-2p) · Danh mục bảng biểu (1-2p) · Danh mục hình vẽ (1-2p) ·
Tóm tắt tiếng Việt (0.5-1p) · Abstract tiếng Anh (0.5-1p, per D1).

**Body (arabic numerals, numbered chương):** MỞ ĐẦU is either (i) a short unnumbered section before
Chương 1 covering lý do chọn đề tài / mục tiêu / đối tượng-phạm vi / phương pháp nghiên cứu / bố cục, or
(ii) folded into Chương 1 §1.1-1.2 as its own subsections. **Recommendation: fold into Chương 1** —
matches brief.md's locked 6-chapter deliverable list (§6: "Ch1 Giới thiệu · Ch2 Tổng quan · Ch3 Phương
pháp · Ch4 Thực nghiệm & Kết quả · Ch5 Triển khai biên · Ch6 Kết luận"), which is itself binding — do
not re-litigate chapter count. This 6-chapter shape is a legitimate, common convention at VN technical
universities (each chapter self-contained, no separate unnumbered mở đầu chapter).

**Citation style:** IEEE numbered `[1]`, `[2]` bracket style is the overwhelming convention for VN CNTT/
kỹ thuật theses (matches PROPOSAL.txt's own `[1] [2] [3]` citations and all 6 `report/*.md` reference
lists already in this format) — **keep IEEE numbered, do not switch to APA.** Reference list ordered by
first-citation-appearance, not alphabetically.

**Phụ lục:** code listing excerpts / repo structure, full per-seed k-fold tables, extra confusion
matrices, Jetson benchmark raw logs, danh mục công bố nếu có (none here) — put bulk data here, keep body
chapters to summary tables + figures only.

**Format (verify against faculty's own quy định trước khi nộp, per Circular 23/2021 — each university
may vary slightly):**
- Font Times New Roman, size 13 (or 13-14, common range), 1.5 line spacing, justified.
- Margins: top 3.5 cm, bottom 3 cm, left 3.5 cm (binding edge), right 2 cm. A4.
- Hình and Bảng numbered per-chapter: `Hình 3.1`, `Bảng 4.2` (chapter.sequence) — matches the EN drafts'
  `Table 3.1`, `Figure 5.1` style already used; direct 1:1 renumber on translation, no restructuring
  needed.
- Chapter/section numbering: `Chương 3`, `3.1`, `3.1.1` — matches EN drafts' `## 3.1`, `### 3.1.1`
  exactly; translation is section-by-section 1:1, not a rewrite of structure.

Typical total length for VN CNTT master thesis body (excl. phụ lục): **70-120 pages**. Given this
project's evidence density (statistics, multi-protocol tables, edge benchmarks), target the upper-middle
of that range (~95-105 pages) rather than padding — brevity favoured over restating tables in prose.

---

## Part B — Definitive Vietnamese chapter outline (page targets + EN source mapping)

PROPOSAL.txt's 3 nội dung nghiên cứu → brief.md's 6-chapter deliverable, reconciled:

| VN Chapter | ND (proposal) | Page target | EN source file(s) | Action required |
|---|---|---|---|---|
| Front matter | — | 8-10p | none (new) | Write fresh: cam đoan, cảm ơn, mục lục, danh mục. Tóm tắt VN + Abstract EN from `abstract.md` **but rewrite abstract's numbers to brief.md §1/§2 axis table** — `abstract.md` itself is a stale intermediate draft (different numbers again: 0.969 mAP, 81.2% scene-only — reconcile against brief before use, don't translate verbatim). |
| **Ch1 Giới thiệu** | ND context (a) | 8-10p | `chapter-01-introduction.md` (292L) | Trim ~40%. Fold in PROPOSAL.txt part (a) wording (lý do chọn đề tài, khả năng ứng dụng, tính thời sự — reuse almost verbatim, it's already VN). §1.4 đóng góp: replace with brief.md §2 tie-framing (Template 1 below), NOT chapter-01's original framing. |
| **Ch2 Tổng quan** | ND1 | 15-18p | `chapter-02-related-work.md` (321L) | Trim ~30%. §2.1-2.3 (YOLO family, Transformer/video models, safety datasets) translate near-1:1. **Expand §2.4** (currently 9 lines on TensorRT/quantization) into a full subsection covering pruning, PTQ/QAT, mixed precision, TensorRT — ND1 explicitly requires this survey and it is currently thin. ND1's 2nd output ("bộ dữ liệu baseline") = restate Unsafe-Net split table here as the surveyed baseline dataset. |
| **Ch3 Phương pháp** | ND2 (arch) | 18-22p | `chapter-03-methodology.md` §3.1 (dataset) **+ chapter-07 §7.2-7.4** (entity ontology, Stage A detector, Stage B relation Transformer) | **Do not translate chapter-03's §3.2-3.5 architecture as-is** — it describes the superseded P1 multi-stream gated-fusion + ensemble/cascade pipeline. The definitive architecture is chapter-07's entity-relation design (YOLO26/11 entity detector → box-token Transformer → MLP+Softmax), which is what brief.md's P3 numbers describe and matches PROPOSAL's diagram (Backbone YOLO → trích xuất đặc trưng không gian → tầng Transformer → phân loại hành vi). Keep chapter-03 §3.2.0 (RT-DETR→Transformer deviation justification) — committee will ask this, see Part E. |
| **Ch4 Thực nghiệm & Kết quả** | ND2 (train/eval) | 22-26p | `chapter-04-experiments-and-results.md` (1164L, protocol + stats methodology only) **+ `chapter-04-section-kfold-cross-validation.md`** (156L, update N to 516) **+ chapter-07 §7.5-7.7** (multi-lens eval, scene-confounding) **+ NEW** (A5 safety/calibration output, not yet drafted anywhere) | **Heaviest compression job.** Drop the V7/V9/V10 experiment-track narrative (§4.2, §4.6-§4.6.2, ~350 lines) — brief.md §2 explicitly excludes chasing 55→56/64 as a dead end; keep only the withdrawn-57/64 methodology note (§4.7.7) and blind-eval discipline (§4.7.11) as the "negative results" required by brief.md §9 rule 2. Reuse §4.7.9 (bootstrap/McNemar) machinery but recompute against brief.md's actual axis-1 numbers (54/64 vs paper 57/64 no-bbox, z=-0.78, p=0.435). Safety FNR/cost-sensitive/ECE/reliability-diagram section (brief §4c-e) has NO EN draft yet — write directly in Vietnamese from A5 script output, do not wait for an English intermediate. |
| **Ch5 Triển khai biên** | ND3 | 16-20p | `chapter-05-edge-deployment.md` (699L) | Structure fully reusable (export → benchmarks → multi-stream → soak → power → dashboard → threat model) but **every number must be re-measured** per D2 (full Jetson access, re-export TensorRT, re-benchmark) — current numbers (8.62ms/121FPS/36-min soak) are the OLD P1 ensemble pipeline, not P3. §5.6.7 "comparison with paper baselines" must drop any literature-only row per brief §9 rule 5. |
| **Ch6 Kết luận** | synthesis | 6-8p | `chapter-06-conclusion.md` (254L) + `thesis-reframe-deployment-led.md` (107L, framing only, do not translate its title suggestions or its "does not beat even no-bbox" framing — superseded) | Rewrite entirely around brief.md §2's 8-axis table. Use Templates 1, 4, 8 below. |
| Tài liệu tham khảo | — | 3-4p | Union of all chapters' reference lists + PROPOSAL.txt [1]-[5] | Dedupe, renumber by first appearance across the whole thesis (not per-chapter). |
| Phụ lục | — | 10-20p | code repo tree, per-seed k-fold table, raw Jetson logs | New; point to GitHub/README per D4 reproducible-repo deliverable. |

**Total body target: ~95-108 pages** (Ch1-6 + refs), phụ lục separate.

---

## Part C — EN → VN terminology glossary

Convention column: **EN** = keep English term as-is (optionally gloss in parentheses on first use) ·
**VN** = translate, use VN term throughout · **BOTH** = VN term in prose, English kept in
parentheses/tables on first use (needed where the VN translation is ambiguous or where the metric name
must match code/figures exactly).

| EN term | VN rendering | Convention | Note |
|---|---|---|---|
| object detection | phát hiện đối tượng | VN | standard, universal |
| backbone | backbone | EN | no stable VN equivalent in ML lit; gloss "(mạng trục chính)" on first use only |
| neck | neck | EN | same as backbone; gloss "(tầng kết nối trung gian)" once |
| feature extraction | trích xuất đặc trưng | VN | standard |
| spatial features | đặc trưng không gian | VN | matches PROPOSAL wording exactly |
| temporal features | đặc trưng chuỗi thời gian / đặc trưng thời gian | VN | PROPOSAL uses "chuỗi thời gian" |
| transformer encoder | tầng mã hoá Transformer | BOTH | "Transformer" always kept capitalized as proper noun, "encoder" translated |
| self-attention | self-attention (cơ chế tự chú ý) | EN | gloss once, then keep English — VN "tự chú ý" reads awkwardly repeated |
| token | token | EN | universal in VN ML writing, no translation used |
| fine-tuning | tinh chỉnh | VN | PROPOSAL's own word |
| frozen backbone | backbone đóng băng / backbone cố định trọng số | VN | "đóng băng" is common informal-technical, "cố định trọng số" more formal — pick one, use consistently |
| inference | suy luận | VN | some papers keep "inference"; VN theses commonly translate |
| latency | độ trễ | VN | standard, matches PROPOSAL |
| throughput | thông lượng | VN | standard, matches PROPOSAL |
| quantization INT8 | lượng tử hoá INT8 | VN | "INT8" kept as-is (precision name) |
| pruning | cắt tỉa (mô hình) | VN | standard |
| edge computing / edge device | điện toán biên / thiết bị biên | VN | matches PROPOSAL exactly |
| ground truth | nhãn thực tế / nhãn chuẩn | VN | pick "nhãn thực tế", use consistently |
| bounding box | hộp giới hạn (bounding box) | BOTH | gloss once; "bbox" abbreviation kept in tables/code refs |
| annotation | gán nhãn / chú thích dữ liệu | VN | "gán nhãn" preferred, matches PROPOSAL's "gán nhãn" |
| confusion matrix | ma trận nhầm lẫn | VN | standard, matches PROPOSAL |
| precision | Precision (độ chính xác dự đoán dương) | BOTH | **keep "Precision" capitalized in English** in metric tables — VN "độ chính xác" is already used for *accuracy*; conflating the two is a real readability risk, flag explicitly to committee if asked |
| recall | Recall (độ bao phủ / độ nhạy) | BOTH | same reasoning; keep "Recall" in tables |
| macro accuracy | macro accuracy (độ chính xác trung bình theo lớp) | BOTH | gloss once, then "macro accuracy" or "độ chính xác macro" — keep consistent with brief.md's own usage |
| mean per-class recall | độ bao phủ trung bình theo lớp | VN | this IS the definition of macro accuracy in brief.md §4a — state the equivalence explicitly once |
| F1-score | chỉ số F1 (F1-score) | BOTH | |
| Cohen's kappa | hệ số Kappa (Cohen's κ) | BOTH | proper-noun metric, keep κ symbol |
| Matthews correlation coefficient | hệ số tương quan Matthews (MCC) | BOTH | keep MCC abbreviation |
| ROC-AUC | ROC-AUC (diện tích dưới đường cong ROC) | EN | gloss once, then keep abbreviation — universal in VN ML lit |
| PR-AUC | PR-AUC (diện tích dưới đường cong Precision-Recall) | EN | same |
| calibration | hiệu chỉnh (xác suất) | VN | |
| expected calibration error | sai số hiệu chỉnh kỳ vọng (ECE) | BOTH | keep ECE abbreviation, used in all tables |
| reliability diagram | biểu đồ độ tin cậy | VN | |
| bootstrap confidence interval | khoảng tin cậy bootstrap | VN | |
| McNemar test | kiểm định McNemar | VN | proper-noun test name kept, "kiểm định" translated |
| k-fold cross-validation | kiểm định chéo k-fold (k-fold cross-validation) | BOTH | gloss once |
| leave-camera-out | leave-camera-out (phương pháp loại-camera-kiểm-tra) | EN | no fixed VN term exists — this protocol is close to novel in this thesis; coin a gloss once, then keep English term for precision (matches brief.md's own usage) |
| false negative rate | tỷ lệ âm tính giả (FNR) / tỷ lệ bỏ sót | VN | "tỷ lệ bỏ sót" is more intuitive for the safety framing (missed unsafe event) — prefer this in safety-section prose, use FNR in tables |
| cost-sensitive learning | học có trọng số chi phí (cost-sensitive) | BOTH | |
| ablation study | nghiên cứu loại bỏ thành phần (ablation) | BOTH | "ablation" commonly kept as a noun label for tables ("Bảng ablation") |
| baseline | baseline / mô hình cơ sở | EN | "baseline" near-universally kept untranslated in VN ML theses; use as-is |
| statistical significance | ý nghĩa thống kê | VN | |
| generalisation | khả năng tổng quát hoá | VN | |
| overfitting | hiện tượng quá khớp (overfitting) | BOTH | gloss once, "overfitting" also frequently kept bare in VN CS writing |
| data leakage | rò rỉ dữ liệu | VN | |

General rule for the writer: **first occurrence of every EN/BOTH term gets a Vietnamese gloss in
parentheses; subsequent occurrences use the shorter form (English or VN, per table) consistently for the
rest of the thesis.** Do not alternate translations of the same term across chapters — build one
`docs/thuat-ngu.md`-style term list while translating and reuse it (existing `docs/` skeleton per lane D).

---

## Part D — 8 ready-to-use Vietnamese paragraph templates (publication-quality)

Numbers below are brief.md §1/§2/§4 canonical values — **update only if A1-A5 produce different final
frozen numbers; do not let translators silently re-insert chapter-04/05/06's stale figures.**

**Template 1 — Đóng góp chính của luận văn (Ch1 §1.4 / Ch6 §6.1 mở đoạn)**

> Trên chỉ số gốc mà công trình của Önal và Dandıl (2024) [17] công bố, hệ thống được đề xuất trong luận
> văn đạt độ chính xác thống kê tương đương (statistical tie) với kết quả đã công bố, đồng thời chứng
> minh rằng ở kích thước tập kiểm tra 64 video như bài báo gốc sử dụng, không một phương pháp nào — dù
> hoàn hảo tuyệt đối — có thể tạo ra khác biệt có ý nghĩa thống kê so với kết quả đã công bố. Đóng góp
> cốt lõi của luận văn không nằm ở việc vượt qua con số 95,81% trên chính giao thức đã bị giới hạn về mặt
> thống kê đó, mà nằm ở năm khía cạnh mà công trình gốc chưa từng đo lường: độ tin cậy thống kê qua kiểm
> định chéo quy mô lớn, khả năng tổng quát hoá qua camera chưa từng quan sát, cấu trúc lỗi theo hướng an
> toàn lao động, mức độ hiệu chỉnh xác suất của mô hình, và hiệu năng triển khai thời gian thực đã được đo
> đạc trên thiết bị biên thực tế.

**Template 2 — So sánh với công trình gốc trên chỉ số của chính bài báo (Ch4)**

> Trên 64 video kiểm tra thuộc 4 lớp hành vi nguy hiểm — đúng tập dữ liệu và đúng giao thức mà Bảng 6-7
> của [17] sử dụng — mô hình đề xuất đạt 54/64 (84,4% top-1, 88,4% macro accuracy), so với 62/64 (96,88%
> top-1, 95,81% macro) của kiến trúc đầy đủ trong công trình gốc và 57/64 (89,06% top-1, 85,78% macro) của
> đường cơ sở không sử dụng hộp giới hạn hành vi (no-bbox baseline) mà chính bài báo gốc cũng báo cáo. So
> với đường cơ sở no-bbox — mức so sánh công bằng hơn vì cùng không sử dụng nhãn hành vi mức khung hình —
> chênh lệch top-1 là −4,7 điểm phần trăm, kiểm định z hai tỷ lệ cho z = −0,78, p = 0,435, tức **không có
> ý nghĩa thống kê**; đồng thời độ chính xác macro của mô hình đề xuất (88,4%) **vượt** đường cơ sở
> no-bbox (85,78%). Kết quả này được diễn giải là tương đương thống kê, không phải là thua kém.

**Template 3 — Giải thích trần thống kê của tập kiểm tra 64 video (Ch4 thảo luận / Ch6 hạn chế)**

> Một phát hiện mang tính phương pháp luận của luận văn là: trên tập kiểm tra 64 video, bài báo gốc chỉ
> sai 2 trên tổng số 64 trường hợp. Do đó, biên độ khác biệt tối đa có thể đạt được giữa bất kỳ phương
> pháp nào và kết quả đã công bố chỉ là 2 video. Ngay cả khi một mô hình giả định đạt độ chính xác tuyệt
> đối 64/64, kiểm định McNemar trong trường hợp tốt nhất cũng chỉ cho p = 0,50 (kiểm định Fisher chính
> xác cho p = 0,496) — nghĩa là **không một phương pháp nào, dù hoàn hảo, có thể tạo ra khác biệt có ý
> nghĩa thống kê so với công trình gốc ở cỡ mẫu này.** Đây là một giới hạn cấu trúc của bộ chỉ số đánh giá
> (benchmark), không phải là hạn chế của phương pháp được đề xuất, và bản thân phát hiện này là một đóng
> góp phương pháp luận độc lập của luận văn.

**Template 4 — Đóng góp về độ tin cậy thống kê (kiểm định chéo k-fold) (Ch4 / Ch6)**

> Để khắc phục hạn chế của một khoảng tin cậy rộng trên tập kiểm tra 64 video, luận văn thực hiện kiểm
> định chéo 5-fold trên toàn bộ 516 video thuộc 4 lớp hành vi nguy hiểm của Unsafe-Net, lặp lại qua nhiều
> seed huấn luyện. Kết quả cho độ chính xác macro trung bình 84,0% với khoảng tin cậy hiệu chỉnh
> Nadeau-Bengio ±3 điểm phần trăm — hẹp hơn khoảng 3,5 lần so với bất kỳ điểm ước lượng đơn lẻ nào trên 64
> video. Đây là minh chứng thống kê duy nhất trong dự án (và duy nhất so với công trình gốc, vốn chỉ báo
> cáo một điểm số đơn trên 64 video, không kèm khoảng tin cậy).

**Template 5 — Đóng góp về khả năng tổng quát hoá qua camera (leave-camera-out) (Ch4 / Ch6)**

> Giao thức leave-camera-out — huấn luyện trên các camera này và kiểm tra trên (các) camera hoàn toàn
> chưa xuất hiện trong tập huấn luyện — cho thấy mô hình đề xuất duy trì 78,6% độ chính xác macro, trong
> khi một đường cơ sở chỉ dựa trên đặc trưng khung cảnh (scene-only, dự đoán hành vi từ danh tính camera)
> sụp đổ còn 16,9%. Khoảng cách +60,3 điểm phần trăm này cho thấy mô hình học được cấu trúc hành vi thực
> sự, không phải ghi nhớ khung cảnh — một câu hỏi mà giao thức cùng-phân-bố (same-distribution) của công
> trình gốc, do sử dụng đúng 8 camera cố định ở cả huấn luyện lẫn kiểm tra, **về nguyên tắc không thể trả
> lời được**.

**Template 6 — So sánh với công trình gốc, đoạn tổng hợp cho Kết luận (Ch6, tránh đọc như "thất bại")**

> Luận văn không đặt mục tiêu vượt qua con số 95,81% mà công trình gốc công bố trên giao thức cùng-phân-bố
> của chính bài báo đó — như đã chứng minh ở mục [X], không phương pháp nào có thể làm được điều này ở cỡ
> mẫu 64 video một cách có ý nghĩa thống kê. Thay vào đó, luận văn cung cấp bằng chứng thực nghiệm trên
> năm khía cạnh mà công trình gốc chưa từng đo lường: độ tin cậy thống kê (khoảng tin cậy hẹp hơn 3,5 lần
> nhờ kiểm định chéo N=516), khả năng tổng quát hoá qua camera chưa quan sát (+60,3 điểm phần trăm so với
> đường cơ sở khung cảnh), cấu trúc lỗi theo hướng an toàn (tỷ lệ bỏ sót theo từng lớp, đánh giá có trọng
> số chi phí), mức độ hiệu chỉnh xác suất (ECE, biểu đồ độ tin cậy) cần thiết trước khi đặt ngưỡng cảnh
> báo trong thực tế, và hiệu năng triển khai thời gian thực đo đạc trực tiếp trên thiết bị biên NVIDIA
> Jetson. Đây là những điều kiện tiên quyết để một hệ thống giám sát an toàn công nghiệp có thể triển khai
> thực tế, và là phạm vi đóng góp chính của luận văn.

**Template 7 — Đóng góp triển khai biên thời gian thực (Ch5 mở đầu / Ch6)**

> Mô hình đề xuất được biên dịch bằng NVIDIA TensorRT ở hai độ chính xác FP16 và INT8, và được đo đạc
> trực tiếp trên thiết bị biên NVIDIA Jetson Orin Nano — không sử dụng bất kỳ số liệu tham chiếu nào từ
> tài liệu công bố. Kết quả đo được bao gồm độ trễ p50/p95/p99 theo từng khung hình và từng video, thông
> lượng khung hình trên giây (FPS), thông lượng tổng hợp khi xử lý đồng thời nhiều luồng video không mất
> khung hình (zero-drop), khả năng chịu tải liên tục qua bài kiểm tra soak-test kéo dài, và hiệu suất năng
> lượng đo bằng joule trên mỗi lần suy luận và FPS trên watt. Đây là các chỉ số triển khai thực tế mà công
> trình gốc — vốn chỉ công bố kết quả huấn luyện phía máy chủ — không cung cấp.

**Template 8 — Hạn chế của luận văn (Ch6 §6.3, trung thực, không tự hạ thấp quá mức)**

> Luận văn còn một số hạn chế cần được nhìn nhận rõ. Thứ nhất, tập kiểm tra chính thức của Unsafe-Net chỉ
> gồm 64 video thuộc 4 lớp hành vi nguy hiểm, nên mọi so sánh trực tiếp trên chỉ số gốc của bài báo mang
> khoảng tin cậy rộng và bị giới hạn về mặt cấu trúc thống kê (xem mục [X]); luận văn giảm thiểu rủi ro
> này bằng kiểm định chéo N=516 nhưng không thể thay thế hoàn toàn một tập kiểm tra độc lập lớn hơn. Thứ
> hai, toàn bộ dữ liệu Unsafe-Net được thu thập từ một nhà máy duy nhất qua 8 camera cố định trong 39
> ngày; mặc dù giao thức leave-camera-out cho thấy tín hiệu hành vi vượt ra ngoài khung cảnh cụ thể, khả
> năng tổng quát hoá sang một nhà máy, bố trí camera, hoặc điều kiện ánh sáng hoàn toàn khác vẫn chưa được
> kiểm chứng thực nghiệm. Thứ ba, các số liệu triển khai biên phản ánh một cấu hình phần cứng cụ thể
> (Jetson Orin Nano); hiệu năng trên các thiết bị biên khác có thể khác biệt. Những hạn chế này xác định
> hướng phát triển tiếp theo của luận văn, trình bày ở mục 6.4.

---

## Part E — Slide outline (defence deck)

Search confirms: talk portion is often **~15-20 min**, followed by ~15 min hội đồng nhận xét + hỏi
(0-3 câu/thành viên). Target **18-20 slides** for a 15-20 min talk (~1 min/slide, front-loaded on
motivation/results, thin on related work).

1. Trang bìa — tên đề tài (VN+EN), học viên, MSHV, GVHD, ngày bảo vệ.
2. Đặt vấn đề — tai nạn công nghiệp thực tế (forklift blind-spot, robot cell, PPE), hạn chế của CCTV thụ
   động.
3. Mục tiêu đề tài — 3 mục tiêu, verbatim từ PROPOSAL.txt.
4. Nội dung nghiên cứu — 3 nội dung → map sang Ch2/Ch3-4/Ch5.
5. Bộ dữ liệu Unsafe-Net — 691 clip, 39 ngày, 8 camera, 8 lớp (4 an toàn/4 nguy hiểm), bảng phân bố lớp.
6. Tổng quan liên quan (1 slide, không quá 1) — YOLO family timeline + Transformer/RT-DETR, tối ưu biên.
7. Kiến trúc đề xuất — sơ đồ Backbone YOLO → trích xuất thực thể → Transformer quan hệ không gian → MLP
   phân loại. Nêu rõ lý do chọn Transformer thuần thay RT-DETR (đề phòng câu hỏi độ lệch so với đề cương).
8. Giao thức đánh giá — split 566/125 cố định theo bài báo, val tách từ train, 1 lần chạm test duy nhất,
   ≥5 seed. (Chứng minh kỷ luật chống rò rỉ dữ liệu — luôn bị hỏi.)
9. Kết quả chính vs bài báo gốc — bảng 54/64 vs 62/64 vs 57/64 no-bbox, kèm z/p-value ngay trên slide.
10. Vì sao không thể "thắng" ở N=64 — trần thống kê McNemar, 1 slide riêng, chuẩn bị sẵn cho Q&A.
11. Kết quả kiểm định chéo N=516 — 84,0% ± 3pp, so sánh độ rộng CI với 64-clip.
12. Kết quả leave-camera-out — 78,6% vs baseline khung cảnh 16,9%, biểu đồ cột.
13. Phân tích an toàn — FNR theo lớp, ma trận chi phí, ma trận nhầm lẫn.
14. Hiệu chỉnh xác suất — ECE, reliability diagram, vì sao cần thiết cho ngưỡng cảnh báo thực tế.
15. Tối ưu & triển khai biên — TensorRT FP16/INT8, Jetson Orin Nano, sơ đồ hệ thống multi-stream.
16. Kết quả triển khai đo được — độ trễ p50/p95/p99, FPS, throughput đa luồng zero-drop, công suất
    (J/inference, FPS/W).
17. Demo trực tiếp / video quay sẵn — hệ thống chạy thực tế, dashboard cảnh báo.
18. Đóng góp của luận văn — 5 điểm gạch đầu dòng (Template 1/6 rút gọn).
19. Hạn chế & hướng phát triển — Template 8 rút gọn + 2-3 hướng mở rộng.
20. Slide cảm ơn — Xin cảm ơn / Hỏi đáp.

Design notes: minimize text-per-slide (search confirms VN convention = "hình ảnh minh hoạ thay vì quá
nhiều chữ"); every accuracy number on-slide must carry its CI or p-value in the same visual line (per
brief §9 rule 1) — this doubles as defence protection.

---

## Part F — Likely committee questions (VN CNTT deep-learning defence) + where the answer lives

| # | Câu hỏi dự kiến | Trả lời dựa trên |
|---|---|---|
| 1 | Tập kiểm tra chỉ 64 video, liệu kết luận có đủ tin cậy không? | Template 3 (trần McNemar) + Template 4 (k-fold N=516, CI ±3pp) |
| 2 | Đóng góp khoa học là gì, hay chỉ là ghép nối kỹ thuật (engineering) các mô hình có sẵn (YOLO + Transformer)? | Ch3 kiến trúc entity-relation (khác biệt supervision so với behaviour-as-object của bài báo gốc) + Ch4 leave-camera-out là đóng góp phương pháp luận, không chỉ engineering |
| 3 | Vì sao không so sánh với [YOLOv8/RT-DETR/phương pháp X khác]? | Ch2 tổng quan giải thích vị trí của lựa chọn kiến trúc; brief §9 rule 5 giới hạn phạm vi so sánh vào baseline đã công bố + baseline tự xây (scene-only, no-bbox) |
| 4 | Vì sao không vượt qua 95,81% của bài báo gốc? | Template 2 + Template 3 — trần thống kê, không phải hạn chế phương pháp |
| 5 | Khả năng ứng dụng thực tế, độ tin cậy khi triển khai ở nhà máy thật? | Ch5 (Jetson đo thật, không dùng số liệu tài liệu) + hạn chế Template 8 (1 nhà máy, 8 camera cố định) |
| 6 | Kiểm soát overfitting / rò rỉ dữ liệu (data leakage) thế nào? | Ch3-4 giao thức: val tách từ train, 1 lần chạm test, ≥5 seed, log test-touch |
| 7 | Vì sao đổi từ RT-DETR (đề cương) sang Transformer thuần? | chapter-03 §3.2.0 — token sequence đã là lưới hạng thấp (T×16×512), cross-attention decoder kiểu DETR không cần thiết |
| 8 | Chi phí phần cứng, khả năng mở rộng số luồng camera đồng thời? | Ch5 multi-stream zero-drop throughput + giá thiết bị Jetson (USD trong bảng) |
| 9 | Mô hình có robust với thay đổi ánh sáng / camera mới không? | Ch4 leave-camera-out là bằng chứng gần nhất; nêu rõ đây KHÔNG phải kiểm chứng đổi ánh sáng, thuộc hạn chế Template 8 |
| 10 | Học được gì từ quá trình làm luận văn (kể cả thất bại)? | Ch4 mục negative-results (57/64 rút lại, MPS bug, box-context âm tính) — chuẩn bị 2-3 câu, đừng né tránh |

**Cách hội đồng VN CNTT thường phản ứng với đóng góp "reproduction + rigor":** hội đồng cấp thạc sĩ ứng
dụng thường coi trọng (1) hệ thống chạy được đầu-cuối (demo sống) hơn là luận điểm thống kê thuần tuý —
nên **demo phải mở màn tốt, không chỉ xuất hiện cuối bài**; (2) đóng góp phương pháp luận/rigor (trần
N=64, leave-camera-out) thường được đánh giá cao nếu trình bày như một *phát hiện* (finding) chứ không
phải một *lời xin lỗi* cho việc "không thắng" — nên dùng Template 1/6, tuyệt đối tránh câu mở đầu kiểu
"luận văn không đạt được mục tiêu ban đầu"; (3) một số thành viên hội đồng có thể hỏi thẳng "vậy đóng góp
mới là gì" nếu không thấy kiến trúc mới — chuẩn bị sẵn câu trả lời phân biệt rõ "phương pháp giám sát mới
(entity-relation thay vì behaviour-as-object)" khỏi "kiến trúc mới" để không bị bắt bẻ là phóng đại.

---

## Unresolved questions

1. Chưa xác định trường/khoa cụ thể → không thể lấy đúng file "quy định trình bày" chính thức (font/lề có
   thể lệch 0.5-1pt so với search kết quả chung). **Cần xin file mẫu chính thức từ phòng sau đại học
   trước khi format chương cuối.**
2. Thời lượng bảo vệ thực tế (15-20-25 phút) và số lượng thành viên hội đồng chưa được owner xác nhận —
   slide outline giả định 18-20 slide / ~20 phút; điều chỉnh số slide nếu thời lượng chính thức khác.
3. `abstract.md` hiện tại mang một bộ số liệu thứ ba (0.969 mAP, scene-only 81.2%, LCO 75.6%/52.5%/16.9%)
   khác cả brief.md lẫn chapter-07 — cần lane C xác nhận bộ số cuối cùng (brief.md §1/§2, giả định là
   nguồn đúng) trước khi viết Tóm tắt/Abstract cuối, không dịch abstract.md nguyên trạng.
4. Ch4 mục an toàn (FNR/cost-sensitive) và hiệu chỉnh (ECE/reliability diagram) chưa có bản nháp tiếng
   Anh nào — phụ thuộc hoàn toàn vào việc lane A hoàn thành A5 đúng hạn (ngày 6, theo lịch §7 brief.md).
   Nếu A5 trễ, Ch4 sẽ thiếu 2 mục bắt buộc theo hợp đồng đánh giá §4c-4e.
