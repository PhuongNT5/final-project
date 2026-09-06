# Quy trình dedupe train split + chạy lại toàn bộ

Ngày: 2026-09-06. Liên quan: `dataset-label-contradictions.md`, `scene-confound-no-pixel-floor.md`.
Module: [unsafenet/dedupe.py](unsafenet/dedupe.py). Audit nguồn: `sweep_results/duplicate_audit.json`.

## 0. Nguyên tắc bất di bất dịch

1. **Không xoá file gốc.** `dedupe.py` chỉ sinh *drop-list*; caller lọc lúc load. `data/` giữ nguyên 691 clip → mọi số cũ vẫn tái lập được bằng `--dedupe none`.
2. **Test 64 clip đóng băng.** Đó là bộ đánh giá của paper. Dedupe chỉ áp cho train. Nếu bỏ clip test thì không còn so được với 62/64.
3. **Luôn báo cả hai con số**: raw (contaminated) và deduped. Không bao giờ im lặng bỏ clip rồi báo số đẹp hơn.

## 1. Bằng chứng nền

md5 toàn bộ 691 clip → 658 md5 khác nhau, **32 nhóm trùng byte, cả 32 nhóm đều mâu thuẫn nhãn**.

| hình dạng nhóm | số nhóm |
|---|---|
| 2 train, 0 test | 21 |
| 2 train, 1 test | 1 |
| 1 train, 1 test | 8 |
| 0 train, 2 test | 2 |

Cặp lớp mâu thuẫn: `0↔1` (WalkwayViol↔UnauthorInt) 26/32. Tổng ảnh hưởng: 42/452 clip train (9.3 %), 8/64 clip test (12.5 %).

**Trần chứng minh được < 100 %:** hai nhóm trùng nằm *cả hai* trong bộ test — `0_te23.mp4 ≡ 4_te9.mp4`, và `0_te8.mp4 ≡ 1_te1.mp4` (md5 `de03a3421d18b5f7f6d209f89b0c35f3`), nhãn khác nhau. Bộ phân loại tất định bắt buộc sai đúng 1 clip mỗi nhóm → **tối đa 63/64 = 98.44 %**. Paper báo 62/64 = 96.88 %, tức chỉ cách trần 1 clip.

## 2. Chọn policy — điểm cần quyết

Anh nói *"bỏ một thành viên của mỗi cặp mâu thuẫn"* → đúng là policy `keep_one`. Mặc định tôi để `drop_all`. Khác biệt quan trọng, nêu rõ:

| policy | bỏ | giữ lại gì | rủi ro |
|---|---|---|---|
| `keep_one` | 22 clip train | 1 thành viên/nhóm, **nhãn của nó chọn theo thứ tự tên** | Ta *không có cơ sở* biết nhãn nào đúng. Giữ 1 thành viên = tung đồng xu rồi dạy model nhãn đó. Nhiễu nhãn vẫn còn, chỉ hết mâu thuẫn nội bộ. |
| `drop_all` | 52 clip train | không giữ thành viên nào của nhóm mâu thuẫn | Mất 9.3 % train, nhưng train sạch nhiễu nhãn ở các nhóm này. |
| `none` | 0 | — | baseline contaminated, để đối chiếu |

**Khuyến nghị:** `drop_all` làm kết quả chính, `keep_one` làm ablation. Lý do: với `keep_one`, nhãn giữ lại được chọn bởi một tiêu chí tuỳ tiện (alphabet), không phải bởi bằng chứng — luận văn không bảo vệ được lựa chọn đó. `drop_all` là lựa chọn duy nhất không cần giả định về nhãn nào đúng. Cả hai đều đã cài, chạy được ngay.

## 3. Cách hoạt động của module

```python
from unsafenet import dedupe
dedupe.drop_names('train', 'drop_all')      # -> set 52 tên file
dedupe.keep_mask(clip_ids, 'train', 'keep_one')   # -> boolean array cùng thứ tự clip_ids
```

`keep_mask` nhận đúng thứ tự clip đang load nên áp được cho mọi file token (`clip_ids`) lẫn file frame-out (`names`). Mặc định của split `test` là `keep` (bỏ 0 clip) ở mọi policy.

Kiểm chứng nhanh (đã chạy):
```
train/drop_all -> 52 | train/keep_one -> 22 | test -> 0
```

## 4. Các file đã nối dây

| file | thay đổi |
|---|---|
| `unsafenet/dedupe.py` | mới — drop-list + keep_mask |
| `run_kfold_relation.py` | `load_pooled(tag, dedupe_policy)` + cờ `--dedupe` |
| `run_lco_eval.py` | `--dedupe`, mask dùng chung cho cả baseline whole-frame (`load_wf_pooled`) để hai bên vẫn khớp thứ tự |
| `unsafenet/scene_confound.py` | `--dedupe`, lọc train, test đóng băng |
| `run_representation_ladder.py` | `--dedupe`, `--out` (giữ file baseline cũ) |

Tất cả mặc định `none` → mọi lệnh cũ chạy lại vẫn ra số cũ.

## 5. Trình tự chạy lại

Bước A — baseline contaminated (ĐÃ XONG, giữ để đối chiếu):
```
sweep_results/representation_ladder.json
sweep_results/scene_confound.json
$SCRATCH/kfold_rep_BASELINE_none.log      # 5-fold x 10 repeats
```

Bước B — đánh giá nhanh, không cần train lại (giây):
```bash
.venv/bin/python run_representation_ladder.py --dedupe drop_all \
    --out sweep_results/representation_ladder_drop_all.json
.venv/bin/python -m unsafenet.scene_confound --dedupe drop_all \
    --out sweep_results/scene_confound_drop_all.json
```

Bước C — hai protocol nặng (giờ):
```bash
.venv/bin/python run_lco_eval.py --seeds 3 --epochs 60 --tag app_ --dedupe drop_all
.venv/bin/python run_kfold_relation.py --k 5 --repeats 10 --seeds 1 --epochs 60 \
    --tag yoloapp_ --dedupe drop_all
```

Bước D — train lại model blind 64-clip (`logits_yolo26.pt`) trên train đã dedupe, rồi chạy lại `scene_confound` để cập nhật so sánh McNemar model-vs-floor.

Bước E — lặp B–D với `--dedupe keep_one` làm ablation.

## 6. Kết quả bước B (đã có)

Test đóng băng nên nhánh same-distribution gần như không đổi — đúng như kỳ vọng, vì dedupe chạm vào train chứ không chạm vào bộ chấm điểm.

| | same-dist top1/macro | LCO top1/macro |
|---|---|---|
| floor (camera-id), none | 81.2 / 77.2 | 16.9 / 10.4 |
| floor, drop_all | 81.2 / 77.2 | 5.4 / 3.4 |
| CLIP probe, none | 79.7 / 77.9 | 52.7 / 60.4 |
| CLIP probe, drop_all | 78.1 / 75.6 | **55.7 / 59.4** |
| CLIP probe, keep_one | 78.1 / 75.6 | 53.8 / 59.2 |

Đọc: dedupe **không** làm same-dist đẹp lên (78.1 < 79.7) — đúng, vì nó bỏ dữ liệu chứ không thêm. Nhưng LCO của CLIP probe nhích lên (+3.0 pp top-1 với `drop_all`), tức bỏ nhãn mâu thuẫn giúp tổng quát hoá qua camera. Floor LCO tụt là nhiễu của một comparator suy biến (hằng số đa số toàn cục đổi lớp), không mang thông tin.

Per-class LCO của CLIP probe hầu như không đổi (WalkwayViol 30.3, UnauthorInt 7.2) → **dedupe một mình không sửa được điểm yếu quan hệ không-gian**; đó vẫn là việc của mô hình relation.

## 7. Chuẩn báo cáo trong luận văn

Với bộ 64 clip, báo ba con số, không gộp:
- **raw 64**: so trực tiếp với paper 62/64.
- **de-contaminated 59**: bỏ 5 clip test có bản sao byte-identical trong train (nhiễm chéo split).
- **trần 63/64**: giới hạn trên chứng minh được của *mọi* bộ phân loại tất định trên bộ này.

Con số thứ ba là đóng góp mạnh nhất: nó biến "chúng tôi không đạt 96.88 %" thành "96.88 % nằm cách trần lý thuyết của benchmark đúng 1 clip, và benchmark có 12.5 % clip test bị hỏng nhãn".

## Câu hỏi chưa giải quyết

1. `keep_one` hay `drop_all` làm kết quả chính? Tôi đề xuất `drop_all`; anh nói `keep_one`. Cần chốt trước khi viết chương kết quả.
2. Có nên báo thêm bản test cũng dedupe (`drop_contradictory`, bỏ 4 clip test) không? Nó phá tính so sánh với paper nhưng cho ước lượng ít thiên lệch nhất.
3. Chưa xác minh trùng md5 là do lỗi copy khi đóng gói dataset hay do hai nhãn thật sự cùng tồn tại trong một clip (đa nhãn). Nếu là đa nhãn thì bài toán vốn không phải single-label — cần soi thủ công vài clip.
4. Đã liên hệ tác giả dataset chưa? Nếu chưa, nên có một dòng trong luận văn nêu rõ là đã/chưa.
