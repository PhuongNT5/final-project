# Chạy vòng 3 — thang tháp đặc trưng đông cứng

Kaggle **chỉ trích đặc trưng**. Phần huấn luyện đầu nhẹ luôn chạy ở máy, ở `OMP_NUM_THREADS=4`, vì
đó là môi trường duy nhất mà một dòng kết quả còn so sánh được với control 83.25% (mục 4.5.6).
Đưa huấn luyện lên GPU sẽ thêm một nguồn trôi số thứ hai chồng lên nguồn mà luận văn đã đo.

Có hai đường để trích đặc trưng. **Đường A chạy ngay tại máy, không cần Kaggle** — đo được:
B/32 mất 16.5 ms/khung ở CPU 4 luồng, tức ~5 phút cho 17,600 khung, nên một tháp B/16 vào khoảng
20 phút, nhanh hơn nữa nếu dùng `mps`. Kaggle (đường B) là tiện lợi khi muốn chạy cả 9 tháp một
lượt hoặc tháp cỡ ViT-L, chứ không phải điều kiện bắt buộc để chạy một tháp.

## Chuẩn bị (đã xong)

```bash
.venv/bin/python kaggle/export-isb-clip-manifest.py     # sinh kaggle/isb-clip-manifest.json
```

---

# Đường A — chạy tại máy

Cache khung nặng ~2.6 GB nên để **ngoài repo** (không có luật .gitignore nào phủ `*.npy`).

```bash
# Bước 1: giải mã một lần cho mọi tháp (~20-40 phút)
.venv/bin/python kaggle/kaggle-decode-frame-cache.py \
  --data data/isafetybench \
  --manifest kaggle/isb-clip-manifest.json \
  --out ~/isb-frame-cache

# Bước 2: chốt giải mã TRƯỚC, rồi mới đến các tháp khác
OMP_NUM_THREADS=4 .venv/bin/python kaggle/kaggle-extract-tower-features.py \
  --cache ~/isb-frame-cache --manifest kaggle/isb-clip-manifest.json \
  --out ~/isb-feats --towers clipB32_control

.venv/bin/python kaggle/verify-and-install-tower-features.py \
  ~/isb-feats/feats_clipB32_control_all.pt --dry-run

# Bước 3: tháp thật (mỗi tháp ~20 phút; cần mạng để tải trọng số lần đầu)
pip install -q open_clip_torch      # cần cho 5 tháp: dc1b, l2b, dfn2bB16, mobileclip_s1, eva02B16
OMP_NUM_THREADS=4 .venv/bin/python kaggle/kaggle-extract-tower-features.py \
  --cache ~/isb-frame-cache --manifest kaggle/isb-clip-manifest.json \
  --out ~/isb-feats --towers clipB16 openclipB16_dc1b dfn2bB16
```

Rồi sang **Bước 4** ở cuối trang.

---

# Đường B — chạy trên Kaggle

## B.1 Tải lên hai dataset

| Dataset | Nội dung | Kích thước |
|---|---|---|
| `isafetybench-1100` | cả hai thư mục `data/isafetybench/hazard/` và `normal/` | 1.24 GB |
| `isb-manifest` | `kaggle/isb-clip-manifest.json` **cùng ba file `kaggle-*.py`** | ~130 KB |

Giữ nguyên hai thư mục `hazard/` và `normal/` — nhãn được suy ra từ chính cấu trúc đó. Để ba file
`.py` vào cùng dataset nhỏ là cách đơn giản nhất, vì khi đó chúng nằm ở
`/kaggle/input/isb-manifest/` và gọi được trực tiếp. Bật Internet cho notebook để tải trọng số tháp.

## B.2 Giải mã một lần (CPU cũng được, ~20-40 phút)

```bash
!python /kaggle/input/isb-manifest/kaggle-decode-frame-cache.py \
  --data /kaggle/input/isafetybench-1100 \
  --manifest /kaggle/input/isb-manifest/isb-clip-manifest.json \
  --out /kaggle/working/cache
```

Sinh `frames.npy` (1100×16×224×224×3 uint8, ~2.6 GB) trong `/kaggle/working`. Giải mã **một lần cho
mọi tháp**: đó vừa là tiết kiệm, vừa là chốt công bằng G14 — không tháp nào đọc pixel khác tháp nào.
Nếu báo thiếu clip thì dừng, vì một cache ngắn không ghép cặp được với control.

## B.3 Control trước, rồi mới đến phần còn lại

```bash
!pip install -q open_clip_torch
!python /kaggle/input/isb-manifest/kaggle-extract-tower-features.py \
  --cache /kaggle/working/cache \
  --manifest /kaggle/input/isb-manifest/isb-clip-manifest.json \
  --out /kaggle/working/feats --towers clipB32_control
```

Tải `feats_clipB32_control_all.pt` về rồi kiểm ở máy:

```bash
.venv/bin/python kaggle/verify-and-install-tower-features.py \
  ~/Downloads/feats_clipB32_control_all.pt --dry-run
```

Phải thấy `VERDICT: the Kaggle decode path reproduces the published features`. Ở máy này, cùng
đường code đó khớp tensor đã công bố ở mức `max|diff| = 1.5e-05`, cosine ≥ 0.9999997. Nếu trên
Kaggle **không** khớp, các tháp vẫn dùng được nhưng mọi biên độ sẽ lẫn hiệu ứng giải mã với hiệu
ứng tháp, và phải viết đúng như vậy trong báo cáo.

## B.4 Thang tháp (mỗi tháp vài phút trên T4)

```bash
!python /kaggle/input/isb-manifest/kaggle-extract-tower-features.py \
  --cache /kaggle/working/cache \
  --manifest /kaggle/input/isb-manifest/isb-clip-manifest.json \
  --out /kaggle/working/feats \
  --towers clipB16 openclipB16_dc1b dfn2bB16 siglip2B16 mobileclip_s1 metaclipB16 \
           openclipB16_l2b eva02B16 clipL14_control
```

Không dùng `--amp`: fp16 nhanh gấp đôi nhưng thêm một nguồn trôi số mà control không có. Giữ
`--bs 64`: đã đo, ở 64 thì bản trích lại B/32 trùng khít bit với tensor công bố, ở 32 lệch 1.5e-05.
Mỗi file
tải về khoảng 36 MB (512-d) hoặc 54 MB (768-d). `clipL14_control` là chốt thứ hai: phải ra lại
khoảng 87.4% như artifact E2 đã đóng băng. Không ra lại được thì lỗi ở cache, không ở tháp.

---

# Bước 4 — ở máy: cài và chạy đầu (chung cho cả hai đường)

```bash
.venv/bin/python kaggle/verify-and-install-tower-features.py ~/isb-feats/feats_clipB16_all.pt
OMP_NUM_THREADS=4 .venv/bin/python run_b1_round2_experiments.py \
  --feats sweep_results/relation_tokens/feats_clipB16_all.pt --tag towerB16_ \
  --only control_round2 arch_meanpool
.venv/bin/python analyze_posthoc_paired_vs_control.py
.venv/bin/python analyze_posthoc_asymmetric_measures.py
```

`arch_meanpool` nằm trong danh sách vì ở vòng 2, đầu **không có mô hình thời gian** là cấu hình duy
nhất vượt được cả hai điều kiện khi đi cùng ViT-L/14 (88.58%).

# Những việc KHÔNG làm ở vòng này

* **Không chưng cất (distillation)** — corpus 17.6k khung nhỏ hơn yêu cầu tối thiểu 100-200 lần.
* **Không chưng cất riêng phần đầu** — thông tin không có trong đầu vào của học sinh thì đầu ra
  không tạo ra được; đây là chặn lý thuyết, không phải vấn đề siêu tham số.
* **Không xương sống video** — đã đo: VideoMAE-Kinetics kém tháp ảnh đông cứng 2.37 pp.
* **Không gắn thêm bbox** — thang tháp chỉ đọc nhánh toàn khung và vay `mask` của bộ nhận diện.
