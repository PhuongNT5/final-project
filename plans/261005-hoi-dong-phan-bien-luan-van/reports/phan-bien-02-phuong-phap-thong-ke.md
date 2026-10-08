# PHẢN BIỆN 2 — PHƯƠNG PHÁP LUẬN VÀ THỐNG KÊ

Vai trò: phản biện độc lập, mức khắt khe hội nghị top-tier (CVPR/NeurIPS) + hội đồng VN. Mục tiêu:
công kích kết luận tiêu đề "lạm phát giao thức +15.4pp là nhận nguồn video, không phải nhận hành
vi" và toàn bộ máy thống kê phía sau. Hy sinh ngữ pháp để ngắn gọn.

Quy ước trích: `file.txt:LINE` = số dòng trong file đã đọc. Mọi trích verbatim giữ nguyên tiếng Việt.

---

## PHẦN 1 — TRẢ LỜI QUYẾT LIỆT 10 CÂU HỎI

### Q1. Định nghĩa "nguồn" trên iSafetyBench — metadata thật hay suy luận?

**Trả lời: SUY LUẬN TỪ TÊN TỆP, không phải metadata chính thức, nhưng rủi ro sai lệch chỉ có thể
làm kết luận THẬN TRỌNG hơn, không phải lạc quan hơn.**

Trích verbatim: `chuong-03-phuong-phap.txt:568-569`:
> "Khoá nhóm là **mười một ký tự đầu của tên tệp**, tức định danh video YouTube, trích bằng
> `SRC_RE = r'([A-Za-z0-9_-]{11})_'`. Khoá này cho **151 nhóm**."

Đây là 11 ký tự ID video YouTube (chuẩn YouTube ID có đúng 11 ký tự) — không phải hash, không phải
suy luận mù. Hai rủi ro được tự thú nhận tại `chuong-03-phuong-phap.txt:570-575`: (a) tên tệp không
khớp mẫu → thành nhóm đơn lẻ; (b) khoá này **thô hơn** mức `_trim_` (gộp 2 đoạn cắt cùng video →
152 nhóm nếu tách). Tác giả khẳng định chiều ngược (một nguồn bị xé thành 2 nhóm) "**không bao giờ
xảy ra**" và khoá "**chỉ có thể chặt hơn** mức cần thiết, không bao giờ lỏng hơn" — tức worst-case
chỉ gộp nhầm 2 video khác nhau thành 1 "nguồn", làm **ít nhóm hơn, nhóm to hơn**, khiến ràng buộc
leave-source-out **nghiêm hơn** (khó rò rỉ hơn), không lỏng hơn. `phu-luc-b-cau-hinh-fold.txt`
mục B.2 xác nhận bằng số: 0/45 cặp fold chia sẻ nguồn, 151/151 nguồn được giữ ra đúng 1 lần, bảo
toàn 1,100 clip.

**Lỗ hổng còn lại:** không có đối sánh với metadata YouTube thật (không kiểm tra ID trùng do cắt
ngắn 11 ký tự có đụng độ giữa hai video khác nhau hay không — về lý thuyết ID YouTube đủ 11 ký tự
là duy nhất theo thiết kế của YouTube, nên đụng độ gần như không xảy ra, nhưng luận văn không nói
rõ điều này). Đây là điểm yếu **nhẹ, có thể chữa bằng câu chữ** (thêm 1 câu giải thích vì sao ID
YouTube 11 ký tự không đụng độ), không đủ sức lật kết luận +15.4pp.

### Q2. KMeans(8) của Unsafe-Net — có tuần hoàn (circular) không?

**Trả lời: ĐÃ ĐƯỢC NHẬN DIỆN VÀ SỬA Ở MÃ NGUỒN, có bằng chứng kỹ thuật cụ thể — nhưng còn 1 lỗ hở
chưa kiểm định: độ nhạy theo seed KMeans.**

`tools/build_scene_grouping.py:2-8` (docstring, trích nguyên văn):
> "D1 fix: build a CLASSIFIER-INDEPENDENT scene grouping for leave-scene-out eval. The existing
> `views` are KMeans(8) on the SAME appearance features used for classification
> (tools/apply_zone_prior.py:62) -> **circular**. ... This clusters on each clip's static
> BACKGROUND (temporal-median of its 32 pre-extracted frames, downscaled grayscale) -> a scene
> descriptor **independent of the moving subject / classifier features**."

Đây là bằng chứng mã nguồn: nhóm bối cảnh CUỐI CÙNG dùng `bg_descriptor()` (`build_scene_grouping.py:24-33`)
— trung vị theo thời gian của ảnh thô, chuyển xám, downscale 32×32, **không CNN, không bbox, không
detector** — hoàn toàn khác pipeline đặc trưng mà classifier dùng (token thực thể từ YOLOv11s +
vector ngoại hình cắt theo box). Nhóm KMeans CŨ (`clip_view_map.json`, trường `views`) bị chính tác
giả gọi là "circular" và bị thay thế, không được dùng làm fold chính.

**Lỗ hở còn lại — chưa được kiểm định đầy đủ:** `build_scene_grouping.py:38-40` ghi "KMeans
random_state was hardcoded 0, so the LCO grouping carried no measurable split-choice uncertainty" —
và `reports/test-touch-log.md:189-215` xác nhận: chỉ có **2 phân vùng KMeans khác nhau thực sự**
(cs=0 và cs=1; cs=2 bị phát hiện là suy biến — ARI(cs0,cs2) = 0.9952, chỉ đổi fold của 56 clip toàn
bộ 1 lớp giữa 2 fold có hợp bằng nhau → dự đoán clip-theo-clip không đổi). Biên +16.9pp trên
Unsafe-Net do đó dựa trên **2 cách chia** của 1 thuật toán phân cụm, không có khảo sát độ nhạy rộng
hơn. Đây là lỗ hổng thật nhưng **nhẹ** vì cơ chế circularity chính đã bị chặn bằng thiết kế đặc
trưng khác hẳn.

### Q3. CÂU HỎI TRUNG TÂM — rò rỉ giao thức hay chỉ là fold khó hơn (lệch lớp/dịch chuyển phân phối)?

**Trả lời: KHÔNG tách biệt hoàn toàn bằng 1 kiểm định đối chứng độc lập — nhưng với cấu trúc dữ liệu
của iSafetyBench, hai cách giải thích LÀ MỘT, không phải hai giả thuyết cạnh tranh; trên Unsafe-Net,
luận văn TỰ THÚ NHẬN một phần biên là hiệu ứng dịch chuyển phân phối, chưa định lượng được bao
nhiêu.**

Trên iSafetyBench: `chuong-03-phuong-phap.txt:621-623`:
> "Trên iSafetyBench, **0 trong 151 nguồn mang cả hai nhãn** (76 nguồn chỉ có clip nguy hiểm, 75
> nguồn chỉ có clip bình thường). Nhãn và danh tính nguồn do đó bị trộn lẫn **ở mức cấu trúc dữ
> liệu**."

Vì mỗi nguồn chỉ mang 1 nhãn, BẤT KỲ fold ràng buộc nhóm nào cũng ĐỒNG THỜI là (a) giữ-nguồn-ngoài và
(b) tạo lệch lớp per-fold — hai điều này không thể tách bằng thực nghiệm trên chính bộ dữ liệu này
vì chúng là MỘT hiện tượng, không phải hai nguyên nhân độc lập cộng dồn. Một kiểm định hoán vị nhãn
nguồn (permutation test) sẽ không giúp gì: hoán vị nhãn trên 151 nguồn chỉ có 1 nhãn mỗi nguồn sẽ
phá vỡ đúng cấu trúc cần kiểm định. Đây là giới hạn **cấu trúc của dữ liệu**, không phải lẽ ra phải
chạy thêm 1 thí nghiệm mà tác giả bỏ sót.

Luận văn có loại trừ giả thuyết "ít dữ liệu huấn luyện hơn" bằng lập luận đảo dấu ở
`chuong-06-danh-gia.txt` mục 6.6(a) (fold LCO trung bình có NHIỀU dữ liệu huấn luyện hơn k-fold, nên
nếu khối lượng là nguyên nhân thì dấu phải đảo — nó không đảo).

Trên Unsafe-Net, KHÔNG có cấu trúc confound tuyệt đối (nhóm bối cảnh có nhiều lớp), và tại đây luận
văn **tự thú nhận trực diện** đúng điều Q3 nêu — `chuong-06-danh-gia.txt:970-974`:
> "phần lớn biên gộp đến từ hai nhóm: nhóm 1 (+70.5 pp, n = 139) và nhóm 7 (+44.4 pp, n = 18)... Nhóm
> 1 cũng là nhóm mà đường sàn ngoại hình sụp xuống 10.1% vì lớp đa số của tập huấn luyện (lớp 2)
> không có mặt trong nhóm bị giữ lại — tức **một phần của biên là hiệu ứng dịch chuyển phân phối
> nhãn, không phải năng lực quan hệ**."

**Đây là lỗ hổng thật, được thừa nhận nhưng KHÔNG ĐƯỢC ĐỊNH LƯỢNG**: luận văn không nói bao nhiêu
pp trong +16.9pp (hay dải +13.0–21.8pp 5 kiến trúc) là dịch-chuyển-phân-phối thuần, bao nhiêu là
"nhận bối cảnh thật". **PHẢI CHẠY THÊM THỰC NGHIỆM** (ví dụ: random k-fold được ép có cùng lệch lớp
per-fold như LCO, so BA) để tách hai nguyên nhân trên Unsafe-Net — hiện chưa có.

### Q4. Clustered bootstrap — đơn vị resample, có xử lý đúng tương quan khi so 2 giao thức trên CÙNG dữ liệu?

**Trả lời: ĐÚNG VỀ MẶT KỸ THUẬT, đã kiểm tra mã nguồn.**

`eval_utils/cluster_bootstrap.py:1-31` (hàm `paired_metric_cluster_bootstrap`): đơn vị resample là
**nhóm** (`idx_by_group`, lấy theo `np.unique(groups)`), và ở mỗi lượt lấy mẫu lại, **CÙNG MỘT bộ
chỉ số** được áp lên CẢ `pred_a` và `pred_b`:
```python
pick = rng.integers(0, len(uniq), len(uniq))
idx = np.concatenate([idx_by_group[p] for p in pick])
...
boots.append((metric(y_true[idx], pred_a[idx]) - metric(y_true[idx], pred_b[idx])) * 100.0)
```
Đây chính là bootstrap ghép đôi đúng cách — tương quan giữa 2 giao thức trên cùng clip ĐƯỢC bảo
toàn, không bị đánh giá như 2 mẫu độc lập. Nếu đây là 2 bootstrap độc lập (resample a và b riêng)
thì khoảng tin cậy của hiệu số sẽ RỘNG hơn giả tạo — mã nguồn tránh đúng lỗi này.

**Không có lỗ hổng kỹ thuật ở đây**, miễn là "nhóm = nguồn" là đơn vị độc lập đúng — mà đó lại chính
là giả định của Q2/Q3, không phải của Q4.

### Q5. McNemar p=3.58e-45 — có hợp lệ khi 2 giao thức có fold khác nhau? Con số nhỏ bất thường nghĩa là gì?

**Trả lời: GHÉP ĐÔI VỀ MẶT CLIP LÀ HỢP LỆ (cả 2 giao thức đều out-of-fold trên ĐỦ 1,100 clip), nhưng
ĐỘ LỚN p-value KHÔNG ĐƯỢC PHÉP ĐỌC THEO NGHĨA ĐEN — và đây là điều luận văn ĐÃ TỰ NHẬN, không phải
điều tôi phát hiện ra.**

`eval_utils/paired_tests.py:9-14` (docstring `mcnemar_exact`):
> "Exact two-sided McNemar (sign-binomial on discordant pairs) for two classifiers scored on the
> SAME clips — the only valid significance test for that setup ... two_proportion_z is invalid here
> because it assumes 2 independent samples..."

Cả giao thức rò rỉ và giao thức có ràng buộc nhóm đều là 10-fold CV phủ **toàn bộ** 1,100 clip →
mỗi clip có đúng 1 dự đoán OOF dưới MỖI giao thức → ghép theo ID clip hợp lệ về mặt kỹ thuật
(không phải "2 test set khác nhau không so được" như nghi vấn gốc).

**Lỗ hổng thật**: McNemar coi 1,100 cặp rời-rạc-đồng/khác là Bernoulli độc lập, nhưng clip cùng
nguồn tương quan — số đơn vị độc lập thật là ~151, không phải 1,100. Luận văn **tự nhận** điều này
tại `chuong-03-phuong-phap.txt:615-619`:
> "McNemar bỏ qua cấu trúc cụm, nên trong bối cảnh này nó là kiểm định **quá dễ dãi**... Thứ tự ưu
> tiên khi hai công cụ trái nhau, cố định trước: lấy kết luận của khoảng tin cậy theo cụm."

Và minh chứng thực tế tại G15 (`chuong-06-danh-gia.txt:1221`, Bảng 6.8): McNemar p=1.48e-03 (có ý
nghĩa) nhưng CI theo cụm chứa 0 → **phán quyết TRƯỢT theo CI, không theo McNemar**. Đây là bằng
chứng quy trình thật đã hành xử đúng như đã hứa.

**Nhưng với G12 (kết luận tiêu đề +15.4pp)**, cả McNemar (p=3.58e-45, b=186/c=8) VÀ CI theo cụm
([+10.6,+21.5] loại trừ 0) đều đồng thuận — nên chiều kết luận không sai. Vấn đề còn lại: luận văn
không có 1 câu nào nói rõ "con số e-45 bản thân không có ý nghĩa định lượng thật, nó chỉ là tín
hiệu định hướng vì giả định độc lập bị vi phạm" — câu này chỉ được viết ngầm cho trường hợp 2 công
cụ TRÁI NHAU, không cho trường hợp ĐỒNG THUẬN. **CÓ THỂ CHỮA BẰNG CÂU CHỮ**: thêm 1 câu rằng mọi
p-value McNemar trong luận văn, kể cả khi đồng thuận với CI, bị thổi phồng độ nhỏ do giả định độc
lập bị vi phạm có hệ thống.

### Q6. "0/2 lần chạm test" — vậy 84.2%/90.6%/75.2% từ đâu? Toàn bộ dựa vào CV không test giữ riêng?

**Trả lời: ĐÚNG, VÀ LUẬN VĂN TỰ CÔNG BỐ ĐIỀU NÀY RÕ RÀNG — không phải điều giấu diếm bị phát hiện.**

`chuong-03-phuong-phap.txt:779-783`:
> "Trạng thái tại thời điểm viết: **0 trong 2 lần chạm đã tiêu.** Mọi con số trong Chương 6 là con
> số **out-of-fold** dưới kiểm định chéo, không phải con số trên một tập kiểm thử giữ riêng."

Tất cả: 84.2% (B1 iSafetyBench), 90.6%/75.2% (E3 leak/LSO), 81.2%/64.4% (Unsafe-Net 4 lớp) — đều là
số OOF theo cross-validation, KHÔNG phải số trên test set độc lập chưa từng thấy mô hình nào. Đây là
**rủi ro phương pháp thật** (toàn bộ kết luận của luận văn không có 1 lần xác nhận trên dữ liệu
hoàn toàn chưa đụng tới), nhưng được thiết kế có chủ ý: dành 2 lần chạm cho về sau, tránh dò-và-chọn
trên test (data peeking). Hội đồng sẽ hỏi thẳng câu này — xem Phần 3.

### Q7. Tiền đăng ký 8 cổng — có xác minh được bằng mốc thời gian (git) hay chỉ là lời văn?

**Trả lời: KHÔNG CÓ BẰNG CHỨNG GIT ĐỘC LẬP CHO CHÍNH VĂN BẢN ĐĂNG KÝ CỔNG — đây là lỗ hổng nghiêm
trọng nhất tôi tìm được, và nó lặp lại đúng kiểu lỗi mà chính luận văn đã tự thú ở nơi khác.**

Các cổng được ghi ngày trong văn bản: G10-G13 "đăng ký 20/09/2026" (`chuong-03-phuong-phap.txt:668,677,689,699`),
G14-G18 "đăng ký 29/09/2026" (`chuong-03-phuong-phap.txt:709,720,728,737,747`). Kiểm tra git thực tế:

```
$ git status (đầu phiên) → các file sau có trạng thái "A" (staged, CHƯA TỪNG COMMIT TRƯỚC ĐÓ):
  plans/2026-09-19-thesis-recovery-architecture/phase-12-isafetybench-accuracy-lane.md
  plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-09-20-isafetybench-g10-first-positive-accuracy-result.md
  plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-09-21-protocol-inflation-and-the-capacity-contrast.md
  ... (và ~20 file khác trong cùng thư mục)

$ git log --all -- plans/.../phase-12-isafetybench-accuracy-lane.md   → RỖNG (không một commit nào)
```

Nghĩa là: tài liệu đăng ký cổng G10-G13 (phase-12) **chưa bao giờ nằm trong lịch sử git** tại thời
điểm soát xét — ngày "20/09/2026" ghi trong tên file/nội dung là **tự khai, không có mốc thời gian
độc lập nào xác nhận**. Đối chứng phía mã nguồn: lệnh `--split random` (liên quan G12) chỉ được
thêm vào `run_kfold_relation.py` ở commit `9515ee1`, ngày **2026-09-29 23:22 +0700** — SAU ngày đăng
ký tự khai (20/09) 9 ngày, nên về CHIỀU thì nhất quán — nhưng không chứng minh được VĂN BẢN đăng ký
thật đã tồn tại từ 20/09, vì văn bản đó không hề có trong git ở bất kỳ thời điểm nào trước hôm nay.

Điều đáng chú ý: chính luận văn ĐÃ từng gặp và tự xử lý đúng kiểu lỗi này ở nơi khác —
`reports/test-touch-log.md:6-16`:
> "**This log was created on 2026-09-06. It did not exist before that date.** Every entry dated
> before 2026-09-06 is a **retroactive reconstruction**... and their pre-registration **cannot be
> verified**."

Đây là một "honesty statement" mẫu mực — nhưng nó CHỈ áp dụng cho sổ chạm-test, KHÔNG áp dụng cho
chính 8 cổng G10-G18. Không có "honesty statement" tương đương nào cho các file `phase-12`,
`phase-13`, `finding-2026-09-20-...`, `finding-2026-09-21-...` — những file hiện vẫn chưa hề có mặt
trong lịch sử git. **Đây là lỗ hổng PHẢI CHẠY THÊM THỰC NGHIỆM theo nghĩa ngược — không thể chạy
thêm gì để chứng minh quá khứ; chỉ có thể (a) thêm 1 "honesty statement" đúng kiểu test-touch-log
thừa nhận không xác minh được bằng git, hạ cấp "tiền đăng ký" xuống "tự khai có ngày", hoặc (b) tìm
bằng chứng khác (email, cloud timestamp, backup) ngoài git. Về bản chất đây gần như KHÔNG CHỮA ĐƯỢC
sau khi đã xảy ra — chỉ có thể minh bạch hoá, không thể xác minh lại.**

### Q8. Đa so sánh — có hiệu chỉnh Bonferroni/FDR không?

**Trả lời: KHÔNG. Xác nhận bằng grep toàn bộ `report/vn/*.txt`: 0 lần xuất hiện "Bonferroni", "FDR",
"đa so sánh", "multiple comparison" ở bất kỳ đâu.**

9 mã cổng (G10-G18, G17 là đối chứng không phán quyết) + nhiều so sánh cặp bổ sung ngoài cổng (Bảng
6.4, 6.6: B1w−E0, B1−E0, B1w−B2, B1w−ESTA@26s, B2−E0, ablation pairs...) — tổng cộng ~15-20 kiểm định
trên cùng kho dữ liệu, không hiệu chỉnh gia đình (family-wise).

**Mức độ nghiêm trọng thực tế — THẤP HƠN VỀ NGUY CƠ so với vẻ ngoài**: (a) quy ước phán quyết đòi
ĐỒNG THỜI biên ≥ ngưỡng VÀ CI loại trừ 0 (`chuong-03-phuong-phap.txt:638-640`) — khắt khe hơn p<0.05
đơn thuần; (b) 6/8 cổng phán quyết TRƯỢT dù không hiệu chỉnh — hiệu chỉnh chỉ làm trượt nhiều hơn,
không ít hơn; (c) 2 cổng ĐẠT (G10 p=1.02e-05, G12 p=3.58e-45) đều sống sót dễ dàng qua cả Bonferroni
nghiêm khắc nhất (chia cho 9 ~ ngưỡng 0.0056, cả hai p đều nhỏ hơn nhiều bậc độ lớn). **Kết luận:
đây là lỗ hổng về KỶ LUẬT THỐNG KÊ thật (nên sửa), nhưng không lật được verdict nào đã công bố —
CÓ THỂ CHỮA BẰNG CÂU CHỮ** (thêm 1 đoạn giải trình robustness-to-correction).

### Q9. Đường sàn tiền đăng ký 63.3% — có hợp lý, hay chỉ để hạ thanh so sánh?

**Trả lời: HỢP LÝ VÀ TỰ BẤT LỢI CHO LUẬN VĂN — nhưng có 1 điểm cần hội đồng truy thêm (số liệu
63.3% so với 67.2%/68.2% trong các bảng sau).**

`chuong-03-phuong-phap.txt:591-600`: đường sàn là quy tắc đếm 3 đặc trưng (số người/khung, số vật
thể/khung, tỉ số), khớp **trước khi** encoder nào được huấn luyện, đo trên mẫu thăm dò 300 clip cân
bằng, dưới LSO → 63.3% (so với 67.0% dưới random fold, 50.0% nếu đoán đa số). Việc dùng mức này làm
sàn (thay vì 50%) khiến **mọi biên công bố hẹp lại** — tác giả chọn hướng tự bất lợi, đúng tinh thần
pre-registration.

**Điểm cần hội đồng truy**: Bảng 6.2/6.5 (Ch6) báo "đường sàn quy tắc đếm" = 67.2% (LSO, tính lại
trên toàn bộ 1,100 clip/bộ fold thật), khác với 63.3% (đo trên mẫu thăm dò 300 clip). Đây là 2 số
khác nhau cho cùng khái niệm "đường sàn quy tắc đếm" — luận văn không giải thích rõ quan hệ giữa
"sàn go/no-go ban đầu" (63.3%, dùng 1 lần để quyết định có đáng làm encoder thời gian không) và "sàn
headline tái tính" (67.2%, dùng làm mẫu số cho mọi biên công bố trong Bảng 6.2). Nguy cơ: nếu hội
đồng hiểu nhầm 2 số này là cùng 1 đại lượng đo 2 lần sẽ nghi ngờ reproducibility; thực chất chúng là
2 đại lượng khác nhau có chủ đích (mẫu thăm dò vs. tái tính trên full fold). **CÓ THỂ CHỮA BẰNG CÂU
CHỮ** — thêm 1 câu phân biệt rõ 2 số.

### Q10. 151 nguồn / 8 nhóm bối cảnh — đủ lực thống kê cho CI không? Bậc tự do hiệu dụng?

**Trả lời: VỚI 151 NGUỒN — tạm đủ (CI vẫn tính được, tuy rộng do 0/151 song nhãn). VỚI 8 NHÓM —
KHÔNG ĐỦ, và chính luận văn admit con số headline +16.9pp của Unsafe-Net chỉ có CI ở ĐƠN VỊ SAI.**

`chuong-06-danh-gia.txt:918-923`:
> "**Khoảng tin cậy [+12.5, +21.2] là bootstrap trên 516 clip, KHÔNG phải bootstrap theo cụm trên 8
> nhóm bối cảnh.** Theo quy ước mục 6.1.3, đây là **đơn vị lấy mẫu sai**... Với 8 nhóm, bootstrap
> theo cụm **không có đủ đơn vị để cho một khoảng hữu ích**."

Đây là tự thú trực diện: con số CI duy nhất đi kèm bảng headline của Unsafe-Net (Bảng 6.7, +16.9pp)
là CI SAI ĐƠN VỊ, và KHÔNG CÓ CI đúng đơn vị thay thế — vì 8 cụm không đủ để tính. Cùng logic được
nêu trước ở `chuong-03-phuong-phap.txt:580-582` ("tám nhóm không đủ lực để phân giải các chênh lệch
cỡ vài điểm phần trăm") và lặp lại ở kết luận chương 6.14(c) ("Không chứng nhận bất kỳ con số nào ở
độ phân giải nhóm trên làn Unsafe-Net, vì tám nhóm không đủ đơn vị cho một khoảng tin cậy hữu ích").

**Đây là lỗ hổng thật và nghiêm trọng cho riêng dải số Unsafe-Net (+13.0 đến +21.8pp, 5 kiến trúc) —
PHẢI CHẠY THÊM THỰC NGHIỆM** nếu muốn một CI hợp lệ (ví dụ: thu thập thêm camera/bối cảnh thật, hoặc
dùng một đơn vị phân giải trung gian hợp lý hơn 8). Hiện tại, may mắn là **luận văn không dùng số
Unsafe-Net làm kết luận tiêu đề** (`chuong-06-danh-gia.txt:887-888`: "kết quả tiêu đề của luận văn
không nằm ở làn này") — kết luận tiêu đề dựa trên iSafetyBench (151 nguồn), nơi CI cụm vẫn tính
được (dù rộng). Trên 151 nguồn, bậc tự do hiệu dụng ~150, đủ cho percentile bootstrap, nhưng cấu
trúc 0/151 song-nhãn (Q3) làm CI đó rộng tự nhiên — điều luận văn chủ động không "chữa" bằng cách
đổi đơn vị lấy mẫu (`chuong-03-phuong-phap.txt:624-626`).

---

## PHẦN 2 — 11 LỖ HỔNG XẾP HẠNG THEO KHẢ NĂNG LÀM SỤP KẾT LUẬN HEADLINE

Headline = "+15.4pp là nhận nguồn video, không phải hành vi" (G12, iSafetyBench).

1. **[Q7] Tiền đăng ký 8 cổng không xác minh được bằng git — chỉ là tự khai có ngày.**
   `phase-12-isafetybench-accuracy-lane.md` và các `finding-2026-09-20/21-*.md` (nền tảng văn bản
   của chính G10-G12, tức 2 cổng ĐẠT làm nên headline) **chưa từng nằm trong lịch sử git** tại thời
   điểm soát xét. **KHÔNG CHỮA ĐƯỢC** (không thể xác minh lại quá khứ); chỉ có thể minh bạch hoá
   theo đúng mẫu `test-touch-log.md` đã làm ở nơi khác. Mức độ: không lật được CHIỀU kết luận
   (bằng chứng số liệu/mã nguồn độc lập vẫn ủng hộ +15.4pp thật), nhưng lật được **tính "tiền đăng
   ký"** — nếu hội đồng không chấp nhận tự-khai làm pre-registration, toàn bộ khung "gate đã định
   trước" sụp về "phân tích post-hoc được trình bày lại như pre-registration".

2. **[Q10] CI headline của Unsafe-Net (+16.9pp, [+12.5,+21.2]) là CI SAI ĐƠN VỊ, không có CI đúng
   đơn vị thay thế — tự thú tại `chuong-06-danh-gia.txt:918-923`.** **PHẢI CHẠY THÊM THỰC NGHIỆM**
   để có CI hợp lệ trên 8 cụm (gần như bất khả thi với bộ dữ liệu hiện có → thực chất gần
   KHÔNG CHỮA ĐƯỢC trừ khi thu thêm dữ liệu). Giảm nhẹ: không phải làn tiêu đề của luận văn.

3. **[Q3] Trên Unsafe-Net, luận văn tự thú một phần biên là "hiệu ứng dịch chuyển phân phối nhãn,
   không phải năng lực quan hệ" (`chuong-06-danh-gia.txt:973-974`) nhưng KHÔNG định lượng được bao
   nhiêu.** **PHẢI CHẠY THÊM THỰC NGHIỆM** (control khớp lệch-lớp). Trên iSafetyBench, lỗ hổng này
   không tách được bằng thực nghiệm vì cấu trúc 0/151 song nhãn khiến "rò rỉ" và "khó hơn" là MỘT
   hiện tượng — ở đây xếp hạng thấp hơn vì về bản chất không phải lỗ hổng "chưa làm" mà là giới hạn
   dữ liệu đã được luận văn nêu rõ.

4. **[Q6] Toàn bộ kết luận — kể cả 84.2% triển khai — là số OOF cross-validation, 0/2 lần chạm test
   đã dùng.** Không có 1 con số nào được xác nhận trên dữ liệu test độc lập chưa qua chọn mô hình.
   Về lý, đây không phải "lỗi" (được thiết kế có chủ đích), nhưng về thực chất, **không ai — kể cả
   tác giả — biết liệu kết luận +15.4pp có đứng vững trên test set giữ riêng hay không.** Xếp hạng
   trung bình-cao vì **PHẢI CHẠY THÊM THỰC NGHIỆM** (dùng 1 trong 2 lần chạm) mới trả lời được, và
   về lý thuyết có thể lật ngược nếu test set rơi vào trường hợp xấu.

5. **[Q5] p-value McNemar (kể cả p=3.58e-45) hệ thống bị thổi phồng do vi phạm giả định độc lập
   (coi 1,100 clip tương quan là độc lập), chỉ được cảnh báo khi 2 công cụ TRÁI NHAU, không khi
   ĐỒNG THUẬN.** **CÓ THỂ CHỮA BẰNG CÂU CHỮ** — không lật kết luận vì CI cụm đồng thuận độc lập,
   chỉ là thiếu minh bạch về độ lớn p-value.

6. **[Q9] Hai số "đường sàn quy tắc đếm" (63.3% mẫu thăm dò vs 67.2%/68.2% tái tính full-fold) dùng
   lẫn trong văn bản, không phân biệt rõ.** **CÓ THỂ CHỮA BẰNG CÂU CHỮ.** Rủi ro: hội đồng hiểu nhầm
   là 2 lần đo không khớp nhau của cùng 1 đại lượng → nghi ngờ reproducibility dù không có lỗi thật.

7. **[Q8] 9 kiểm định trên cùng kho dữ liệu, không hiệu chỉnh đa so sánh (Bonferroni/FDR), 0 lần
   nhắc đến trong toàn văn.** **CÓ THỂ CHỮA BẰNG CÂU CHỮ** — cả 2 verdict ĐẠT (G10, G12) sống sót
   dễ dàng qua hiệu chỉnh nghiêm khắc nhất, nên không lật chiều, chỉ là lỗ hổng kỷ luật trình bày.

8. **[Q2] KMeans(8) bối cảnh của Unsafe-Net chỉ khảo sát 2 phân vùng độc lập thực sự (cs=0, cs=1;
   cs=2 suy biến) — không có khảo sát độ nhạy seed rộng.** **PHẢI CHẠY THÊM THỰC NGHIỆM** nếu muốn
   khẳng định kết quả không phụ thuộc lựa chọn seed KMeans cụ thể — nhưng cơ chế circularity chính
   (cùng không gian đặc trưng với classifier) đã được chặn bằng thiết kế, nên mức độ đe doạ thấp.

9. **[Q1] Suy luận "nguồn" từ 11 ký tự tên tệp (ID YouTube) không có xác minh chéo với metadata
   chính thức.** **CÓ THỂ CHỮA BẰNG CÂU CHỮ** (giải thích ID YouTube 11 ký tự không đụng độ theo
   thiết kế) — rủi ro sai lệch, nếu có, chỉ làm ràng buộc NGHIÊM hơn, không nới kết luận, nên gần
   như không đe doạ headline.

10. **[Q4] Cluster bootstrap — kỹ thuật đúng, nhưng phụ thuộc hoàn toàn vào giả định "nguồn/nhóm
    bối cảnh = đơn vị độc lập đúng".** Không phải lỗ hổng độc lập — là hệ quả của Q2/Q3/Q10.
    **KHÔNG CHỮA ĐƯỢC thêm gì về mã nguồn** (mã đã đúng), chỉ còn phụ thuộc đúng-sai của giả định
    đơn vị, đã xếp hạng ở trên.

11. **[Mới, phát hiện phụ] Floor không nhất quán & 2 bản tái lập mAP/metric bị sai rồi sửa lại
    (`chuong-06-danh-gia.txt:1094-1105, 1124-1127`) — cho thấy lịch sử sửa lỗi số liệu dày đặc.**
    Không đe doạ trực tiếp headline (mọi lần sửa đều được công khai, có log), nhưng tích lũy thành
    1 câu hỏi về độ tin cậy quy trình tổng thể. **CÓ THỂ CHỮA BẰNG CÂU CHỮ** — thực ra phần này
    lại là điểm MẠNH nếu trình bày đúng (minh bạch sửa sai), nên xếp cuối, mang tính nhắc nhở hội
    đồng đọc kỹ lịch sử sửa chữa thay vì chỉ đọc số cuối.

---

## PHẦN 3 — CÂU HỎI HỘI ĐỒNG SẼ HỎI + CÂU TRẢ LỜI TỐT NHẤT HIỆN CÓ

**H1. "8 cổng tiền đăng ký này đăng ký ở đâu, ai xác nhận ngày, sao không có trong git?"**
→ Trả lời tốt nhất hiện có: các file kế hoạch (`plans/2026-09-19-.../phase-12...`,
`finding-2026-09-20...`) mang ngày nội bộ nhưng CHƯA commit git tại thời điểm này; tác giả nên công
khai theo đúng mẫu "honesty statement" đã làm ở `test-touch-log.md` — thừa nhận không xác minh được
bằng git, nhưng chỉ ra rằng kết luận số liệu (G12: b=186/c=8, CI cụm loại trừ 0 rộng) độc lập với
việc gate có "tiền đăng ký" hay không — số liệu tự nó vẫn đúng dù khung pre-registration có bị hạ
cấp xuống "tự khai".

**H2. "Toàn bộ Chương 6 là CV, không có test giữ riêng — sao tin được headline?"**
→ Trả lời: đúng, 0/2 lần chạm đã dùng, có chủ đích (tránh peeking); khuyến nghị dùng 1 lần chạm
TRƯỚC bảo vệ để xác nhận riêng con số headline B1 84.2% (hoặc Δ +15.4pp) trên 125 clip test gốc,
hoặc thẳng thắn xác nhận trước hội đồng đây là giới hạn đã biết, để lại cho công việc tương lai.

**H3. "Biên Unsafe-Net +16.9pp có CI nào đúng đơn vị không, hay chỉ là số mô tả?"**
→ Trả lời trung thực nhất theo đúng văn bản: KHÔNG có CI đúng đơn vị (8 cụm không đủ); số liệu
Unsafe-Net toàn bộ nên được đọc như minh hoạ bổ sung, không phải chứng cứ độc lập — kết luận tiêu đề
dựa trên iSafetyBench (151 nguồn, CI cụm hợp lệ).

**H4. "Sao biết +15.4pp không phải vì LSO fold khó hơn (ít dữ liệu/lệch lớp), mà chắc chắn là rò
rỉ nguồn?"**
→ Trả lời: trên iSafetyBench, 0/151 nguồn song nhãn khiến "khó hơn" và "rò rỉ" là cùng 1 hiện tượng
cấu trúc, không tách được bằng 1 thực nghiệm khác — đây chính là cơ chế rò rỉ (mô hình nhớ nguồn →
nhớ nhãn). Luận văn đã loại trừ giả thuyết "ít dữ liệu" bằng lập luận đảo dấu (LCO có nhiều dữ liệu
huấn luyện hơn nhưng không khá hơn). Trên Unsafe-Net, luận văn tự thú một phần biên là dịch-chuyển-
phân-phối, chưa định lượng — đây là điểm yếu thật, nên trả lời thẳng "chưa tách hoàn toàn trên
Unsafe-Net, đã tách được trên iSafetyBench vì cấu trúc dữ liệu".

**H5. "p=3.58e-45 — con số này thật không? Có đáng tin ở độ lớn đó không?"**
→ Trả lời: KHÔNG nên đọc theo nghĩa đen; nó bị thổi phồng vì McNemar coi 1,100 clip tương quan là
độc lập (151 nguồn là đơn vị thật). Nhưng CHIỀU kết luận (loại trừ 0) được CI theo cụm xác nhận độc
lập — nên dấu +15.4pp đáng tin, độ lớn của riêng p-value thì không.

**H6. "Có hiệu chỉnh đa so sánh không? Sao chạy 9 kiểm định mà không Bonferroni?"**
→ Trả lời: không có, đây là thiếu sót trình bày cần sửa; nhưng cả 2 verdict ĐẠT đều vượt ngưỡng
Bonferroni khắt khe nhất (p < 0.05/9 ≈ 0.0056) với biên độ lớn (1.02e-05 và 3.58e-45), nên kết luận
không đổi khi hiệu chỉnh.

**H7. "Đường sàn 63.3% hay 67.2%? Sao 2 số khác nhau cho cùng ý tưởng?"**
→ Trả lời: 63.3% là ngưỡng go/no-go đo trên mẫu thăm dò 300 clip (quyết định có đáng xây encoder
thời gian không); 67.2%/68.2% là đường sàn tái tính trên TOÀN BỘ 1,100 clip/bộ fold thật, dùng làm
mẫu số cho mọi biên công bố trong Bảng 6.2/6.5. Hai đại lượng khác mục đích, không phải đo lại cùng
1 thứ ra 2 kết quả khác nhau.

---

## PHẦN 4 — PHÁN QUYẾT

**ĐẠT CÓ ĐIỀU KIỆN (trục phương pháp luận & thống kê).**

Lý do ĐẠT: máy thống kê (cluster bootstrap ghép đôi, ưu tiên CI-cụm trên McNemar, quy ước phán
quyết 2 điều kiện, công bố đầy đủ cả phán quyết trượt, kỷ luật "mọi số có artifact") thuộc nhóm
cẩn trọng nhất tôi từng thấy ở cấp luận văn thạc sĩ — nhiều lỗ hổng kinh điển (rò rỉ, đa so sánh,
đơn vị bootstrap sai) đã được tác giả TỰ PHÁT HIỆN và ghi lại công khai, kể cả khi bất lợi cho chính
mình (G11/G14/G18 trượt, 6/8 cổng trượt, ECE=0.097, κ=0.65). Headline +15.4pp (iSafetyBench) đứng
vững qua mọi kiểm tra tôi chạy được từ mã nguồn.

Lý do CÓ ĐIỀU KIỆN: (1) nền tảng "tiền đăng ký" của chính 2 cổng tạo ra headline không có mốc thời
gian độc lập xác minh được (Q7, lỗ hổng #1) — đây là vấn đề về KHUNG PHƯƠNG PHÁP (pre-registration),
không phải về SỐ LIỆU, nhưng là trung tâm của tuyên bố "đây là thiết kế phản chứng, không phải lời
hứa"; (2) biên Unsafe-Net hoàn toàn thiếu CI đúng đơn vị; (3) Q3 trên Unsafe-Net chưa được định
lượng. Điều kiện để chuyển ĐẠT: tác giả (a) bổ sung "honesty statement" về tình trạng git của hồ sơ
đăng ký cổng theo đúng mẫu đã làm với test-touch-log.md, (b) ghi rõ Unsafe-Net không có CI nhóm hợp
lệ ở MỌI nơi số liệu đó xuất hiện (không chỉ ở 6.11), (c) thêm 1 đoạn định lượng/giới hạn rõ phần
dịch-chuyển-phân-phối trên Unsafe-Net.

---

## CÂU HỎI CHƯA GIẢI QUYẾT (để cuối theo quy định báo cáo)

1. Chưa đọc `run_duplicate_audit.py`/`run_duplicate_audit_binary.py` đầy đủ (chỉ grep) — chưa xác
   nhận độc lập việc lọc trùng lặp có ảnh hưởng đến cấu trúc nhóm nguồn Q1 không.
2. Chưa có giá trị ARI(KMeans cũ "circular" vs KMeans mới "background") từ lần chạy thực tế của
   `build_scene_grouping.py` (script in ra console, không thấy log số cụ thể được trích trong báo
   cáo) — nên chưa định lượng được mức độ hai grouping thực sự khác nhau bao nhiêu.
3. Chưa xác nhận liệu Bảng 6.1 (Unsafe-Net 5 kiến trúc, +13.0→+21.8pp) có cùng vấn đề "CI sai đơn
   vị" như Bảng 6.7, hay có CI riêng khác — chỉ xác nhận trực tiếp cho Bảng 6.7.
4. Không rõ 2 báo cáo phản biện khác đang chạy song song (Phản biện 3 — truy xuất số liệu; Phản biện
   5 — triển khai biên) có phát hiện trùng hoặc mâu thuẫn với các điểm Q1/Q9 (số liệu) ở trên không
   — nên đối chiếu khi cả 2 hoàn tất.
5. Chưa hỏi tác giả trực tiếp: ngoài git, có email/cloud-timestamp nào cho các file `phase-12`,
   `finding-2026-09-20/21` không? Nếu có, lỗ hổng #1 hạ mức độ nghiêm trọng đáng kể.
