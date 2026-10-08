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
| `run_relation_transformer.py` | `load(sp, tag, dedupe_policy)` + cờ `--dedupe` — chỉ lọc split `train`, bộ 64 test đóng băng |
| `run_kfold_relation.py` (bổ sung) | tên file OOF nay có hậu tố theo policy, `drop_all` không ghi đè bản `none` nữa |

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

Bước D (**ĐÃ XONG** — xem §6c) — train lại model blind 64-clip trên train đã dedupe rồi chạy lại `scene_confound`. Command chuẩn (dùng đúng provenance đã khôi phục bit-exact của `logits_yolo26.pt`):

```bash
.venv/bin/python run_relation_transformer.py --tag yolo26app_ --seeds 5 --aux_w 0.5 --epochs 120 \
    --dedupe drop_all --save_logits sweep_results/relation_tokens/logits_yolo26_drop_all.pt
.venv/bin/python -m unsafenet.scene_confound --logits logits_yolo26_drop_all.pt --dedupe drop_all \
    --out sweep_results/scene_confound_drop_all_matched.json
```

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

## 6b. Kết quả bước C — hai protocol nặng (ĐÃ XONG)

**Đọc kỹ nguồn số — có hai bộ kết quả `drop_all` trên đĩa và chúng khác nhau.**

- **CHUẨN (dùng cho luận văn):** `sweep_results/relation_tokens/dedupe_drop_all/dedupe_drop_all_ablation.log` (2026-09-06, pre-registered, "reported whatever the result"). Đúng cấu hình headline: token `yolo26app_` cho k-fold, `yolo26L1_` **+ `--scene_map sweep_results/zone_prior/clip_scene_map.json`** cho LCO, 5 seed. Đối chứng `--dedupe none` là hai log canonical cùng cấu hình.
- **Legacy (KHÔNG trích dẫn):** các lần chạy sớm hơn của tôi dùng token `yoloapp_`/`app_` và **thiếu `--scene_map`** — LCO rơi về grouping KMeans-appearance nội bộ, per-class lệch nặng. Giữ lại chỉ để đối chiếu hướng.

### Repeated stratified 5-fold x 10 (50 fold) — canonical

| | n clip | relation top-1 | macro | κ | MCC | scene-only | NB Δ | t | p |
|---|---|---|---|---|---|---|---|---|---|
| `none` | 516 | 80.8 ± 4.0 | 84.2 | 0.741 | 0.744 | 77.1 ± 2.5 | +3.7 pp | 1.93 | 0.0600 **n.s.** |
| `drop_all` | 465 | **86.6 ± 3.1** | **86.6** | **0.802** | **0.803** | 80.6 ± 2.4 | **+6.0 pp** | 3.39 | **0.0014 có ý nghĩa** |

Đây là thay đổi quan trọng nhất của dedupe: biên vs scene-only **vượt ngưỡng ý nghĩa** (p 0.0600 → 0.0014). Trước dedupe không kết luận được; sau dedupe kết luận được. (Legacy `yoloapp_` cho cùng hướng: 80.7→85.9, p 0.0574→0.0071.)

Lưu ý diễn giải: top-1 tăng 80.8 → 86.6 **không phải** "mô hình tốt lên". Bỏ 51 clip có giám sát mâu thuẫn làm bài toán bớt nhiễu nhãn ở cả train lẫn chấm điểm. Con số so được là **biên** vs scene-only, vì baseline chạy trên cùng tập đã lọc.

### Leave-camera-out (8 camera-group) — canonical, có `--scene_map`

| | n clip | relation top-1/macro | scene-appearance | biên macro (honest) |
|---|---|---|---|---|
| `none` | 516 | 77.7 / 81.2 | 53.1 / 64.4 | +16.9 pp [CI +12.5, +21.2] |
| `drop_all` | 465 | 75.9 / 79.3 | 54.6 / 61.6 | **+17.7 pp** [CI +12.9, +22.4] |

Per-class recall LCO (transformer):

| lớp | `none` | `drop_all` | Δ |
|---|---|---|---|
| WalkwayViol | 70.5 | 74.6 | +4.1 |
| UnauthorInt | 65.7 | **72.3** | **+6.6** |
| OpenedPanel | 88.7 | 70.2 | **−18.5** |
| CarryOverload | 100.0 | 100.0 | 0 |

**Đính chính (quan trọng).** Trước đó tôi báo "UnauthorInt tụt 44.4 → 39.8, kỳ vọng ban đầu sai một nửa". **Sai** — đó là số của cấu hình legacy thiếu `--scene_map`. Theo số chuẩn, UnauthorInt **tăng 65.7 → 72.3**, đúng kỳ vọng ban đầu: bỏ 26 nhóm mâu thuẫn 0↔1 làm ranh giới WalkwayViol/UnauthorInt sắc lại, và cả hai lớp đó đều lên.

Cái thực sự tụt là **OpenedPanel (−18.5)**, kéo top-1 LCO xuống 77.7 → 75.9. Đọc trung thực: dedupe **không** cải thiện độ chính xác LCO tuyệt đối; nó (i) làm sắc biên vs baseline và (ii) cân lại hai lớp bị nhiễm nặng nhất, trả giá ở lớp 2. Nguyên nhân khả dĩ là fold mỏng đi sau khi bỏ 51 clip — camera-group 3 chỉ còn n=5 (trước là 11), group 6 còn 152, và OpenedPanel chỉ xuất hiện ở group 0/6. Đây là phương sai LCO, không phải bằng chứng dedupe làm hại lớp 2.

### Ghi chú file

- Cảnh báo cũ vẫn đúng cho **legacy**: `kfold_oof_softmax_yoloapp_.pt` đã bị lần chạy `drop_all` ghi đè (11:32) lên bản contaminated. Muốn có lại bản đó phải chạy lại `--dedupe none`.
- **Đã có hàng rào:** `run_kfold_relation.py:118` nay thêm suffix theo policy, nên canonical an toàn — `kfold_oof_softmax_yolo26app_.pt` (03:22, none) và `kfold_oof_softmax_yolo26app__drop_all.pt` (18:44) tồn tại song song.
- Lỗi cosmetics trong log canonical: dòng tiêu đề ghi "52 of 516 pooled clips dropped", dòng thực thi ngay dưới ghi "dropped 51 of 516" (đúng, 516−51=465). Số 51 là số đúng.

## 6c. Kết quả bước D — bộ 64 clip blind (ĐÃ XONG)

Log: `sweep_results/relation_tokens/dedupe_drop_all/stepD_blind64_drop_all.log`. Train 452 → **401** clip (bỏ 51). Test 64 **không lọc** — đây là bộ đánh giá đóng băng của paper.

| | train | blind top-1 | macro | WalkwayViol | UnauthorInt | OpenedPanel | CarryOverload |
|---|---|---|---|---|---|---|---|
| `none` (headline) | 452 | **54/64 = 84.4 %** | 88.4 | 78.1 | 90.9 | 84.6 | 100.0 |
| `drop_all` | 401 | **54/64 = 84.4 %** | 87.3 | 81.2 | 90.9 | 76.9 | 100.0 |

**Cùng 54/64, nhưng không phải cùng một mô hình.** Hai vector dự đoán khác nhau ở **5/64 clip** (idx 1, 2, 9, 14, 51), discordant 2–2 (mỗi bên đúng 2 clip bên kia sai). Blind seed cũng khác (1 vs 0). Nghĩa là top-1 same-distribution **bất biến** với việc dedupe train, dù mô hình bên trong đã đổi. Hướng per-class lặp lại đúng pattern LCO: WalkwayViol lên (+3.1), OpenedPanel xuống (−7.7).

Bước E của D — `scene_confound` khớp cặp (mô hình dedupe **và** sàn no-pixel cùng tính trên train đã dedupe), `sweep_results/scene_confound_drop_all_matched.json`:

| | pooled | sàn camera-ID (test) | model (test) | McNemar | model đúng khi sàn sai |
|---|---|---|---|---|---|
| `none` | 516 | 52/64 = 81.2 % | 54/64 | p = 0.804 | 9/12 |
| `drop_all` khớp cặp | 465 | 52/64 = 81.2 % | 54/64 | p = 0.804 | 9/12 |

Cramér's V 0.784 → **0.799**; I(view;label) 1.101 → **1.175** bits (59.2 % → **63.1 %** entropy nhãn). Confound **mạnh lên** sau dedupe, vì các clip bị bỏ chính là những clip có cùng pixel mà khác nhãn — tức là những clip *làm yếu* liên hệ camera↔nhãn.

**Kết luận của bước D — đây là điểm cần viết vào luận văn.** Dedupe làm sắc biên vs baseline dưới k-fold (p 0.0600 → 0.0014) và LCO (+16.9 → +17.7 pp), nhưng **không** dịch chuyển bộ 64 clip same-distribution một chút nào (54/64, McNemar vs sàn no-pixel vẫn p = 0.804). Nhiễm nhãn không phải thứ chặn bộ 64 clip lại — **confound camera mới là thứ chặn**. Hai vấn đề độc lập, và bước D là bằng chứng thực nghiệm tách được chúng.

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
