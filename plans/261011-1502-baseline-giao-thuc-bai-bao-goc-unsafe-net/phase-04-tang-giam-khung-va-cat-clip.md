# Pha 04 — Tầng giảm khung và cắt clip đơn-hành-động (cổng G23)

## Context Links

- Cổng: **G23** — biên tiền đăng ký ≥ +5.0 pp macro, CI cụm loại trừ 0
- Chẩn đoán cơ chế: `report/chapter-07-multi-entity-and-scene-robust-evaluation.md:179-185`
- Bài báo, định nghĩa chọn khung: `paper/paper.txt:323`
- Bài báo, **cắt clip**: `paper/paper.txt:325` (105 khung → 40 khung, 3.5 s → 1.33 s, gửi ConvLSTM **thành ba phần**)
- Bài báo, áp cho cả train và test: `paper/paper.txt:327`
- Máy móc đã có: `run_yolo_framegate.py`, `unsafenet/behaviour_clip.py`, `unsafenet/aggregate.py:38-59`,
  `run_relation_transformer.py:82,192-214,325-348`
- Tạo tác: `sweep_results/framegate_conf.pt`, `sweep_results/r2/temporal_aggb{0,1}-gate{25,50,75}.pt`

## Overview

- **Priority:** P1 — đây là thành phần **chưa dựng** chính, và là cơ chế mà §7.6 quy toàn bộ +10 pp của [17].
- **Status:** pending
- Bỏ đi nếu G22 trượt sàn (xem `plan.md`).

## Key Insights

0. **Trọng số bộ phát hiện đúng đã được chốt 11/10, và mặc định cũ đã bị sửa.** `det-gateQ` cần một
   bộ phát hiện hành vi; chỉ có **hai** bộ hợp lệ: `runs/detect/sweep_results/r2/beh_ctx/weights/best.pt`
   (mAP50 **0.806**) và `beh_tight/...` (**0.791**), cả hai imgsz 416, CPU, SGD, `fliplr=0.0`. Hai run
   `behaviour_yolo*` là **lỗi hội tụ MPS** (mAP50 đỉnh 0.015 / 0.002) — không bao giờ dùng.
   `unsafenet/behaviour_clip.py` từng mặc định trỏ đúng vào đường dẫn `behaviour_yolo` ấy (và đường
   dẫn đó còn **không tồn tại**); đã sửa mặc định sang `beh_ctx` kèm chú thích. **0.806 < sàn 0.85
   của G22** ⇒ mọi số clip dựng trên nó là **thăm dò, phải dán nhãn dưới-cổng**.
0b. **Có một lộ trình không tiêu ngân sách test, và phải đi lộ trình đó trước.** G23 hỏi *cơ chế có
   tái dựng được +10 pp hay không* — câu đó trả lời được **hoàn toàn trên val**: so `det-mean`/`det-max`
   với `det-gateQ` trên cùng val slice. Chỉ khi val cho thấy cơ chế có hiệu lực mới đáng bàn tới
   việc tiêu lần chạm test còn lại. Suy luận ngược (chạm test trước) vừa tốn ngân sách vừa không
   trả lời được câu hỏi cơ chế.
1. **Ba bộ máy gating đã tồn tại, không bộ nào là cơ chế của [17].** Đã kiểm toán:
   - `run_yolo_framegate.py` — ngưỡng **tuyệt đối** trên max-conf của 4 lớp unsafe; **không cắt**,
     mà lấy mẫu lại các khung giữ được về đúng T=32 (`np.linspace` trên `keep`).
   - `unsafenet/aggregate.py:38-44` — giữ **top-q tỉ lệ** khung theo `softmax(L8)[..., :4].sum(-1)`;
     tổng hợp trong clip; không cắt. `gate25/50/75` = q ∈ {0.25, 0.50, 0.75}, **tỉ lệ**, không phải ngưỡng.
   - `run_relation_transformer.py:206-212` — ngưỡng **tuyệt đối** τ ∈ {0.0, 0.3…0.9} trên
     `frame_soft.max(-1)`, rồi bình quân có trọng số; không cắt.
   - `unsafenet/behaviour_clip.py:62-68` — `det-gateQ` giữ top-q khung theo bằng chứng unsafe của
     **bộ phát hiện hành vi**; gần [17] nhất về nguồn tín hiệu, nhưng vẫn **không cắt** và nó là một
     **bộ phân loại tự nó** (softmax trên hiệu unsafe − safe), không nuôi ConvLSTM.
2. **Vì vậy thiếu đúng ba thứ**, và chỉ ba thứ:
   (a) **cắt clip thành đoạn đơn-hành-động** — không tồn tại ở đâu;
   (b) **đường ống giảm-khung → ConvLSTM**: mọi gating hiện có chạy **sau** khi mô hình đã thấy toàn
   bộ khung (hậu kiểm trên đặc trưng đã trích), còn [17] giảm khung **trước** khi ConvLSTM nhìn
   (`paper.txt:327`: giảm trên **cả** train và test);
   (c) **luật loại khung theo lớp safe**: `paper.txt:323` loại khung thuộc 4 lớp safe *và* "other
   parts that do not contain other unsafe behaviours" — một luật hai nhánh, không phải một ngưỡng.
3. **Mâu thuẫn mẫu số của [17] chưa giải được.** `paper.txt:325` nói một clip được gửi tới ConvLSTM
   **thành ba phần**, mỗi phần một hành động. Nhưng `paper.txt:334` vẫn báo cáo trên **64 video**
   với cỡ lớp [32, 11, 13, 8] và 62/64. Hai khả năng loại trừ nhau:
   - **(A)** 64 clip test vốn đã đơn-hành-động ⇒ cắt là phép rỗng trên test, chỉ tác động lên train;
   - **(B)** mẫu số 64 là **sau** cắt ⇒ "clip" của [17] không cùng đơn vị với "clip" của ta, và
     57/64 ↔ 62/64 không so được với bất kỳ số nào của ta ở mức clip.
   Phải chốt trước ta giải theo nhánh nào, và nếu cả hai khả thi thì **báo cáo cả hai**.
4. **`paper.txt:334` còn một câu không giải thích được**: *"a total of 32 videos passing through the
   YOLO v4 network"* — 32 trong 64. Không khớp với 62/64. Có thể là lỗi dịch/biên tập, nhưng phải
   ghi ra, vì nếu đọc theo nghĩa chữ thì chỉ nửa tập test đi qua bộ phát hiện.
5. **Cắt clip làm nhãn mức clip thành đa nhãn.** Ví dụ của [17] cắt một clip thành ba đoạn thuộc ba
   lớp **khác nhau** (WalkwayViol, OpenedPanel, UnauthorInt). Nhưng dataset của ta có **một nhãn
   một clip** (thư mục lớp, `unsafenet/config.py::CLASS_DIRS`). Ta **không có nhãn đoạn**. Vì vậy
   cắt chỉ dựng được ở dạng **yếu**: cắt theo tính liên tục của bằng chứng bộ phát hiện, rồi gán
   cho mọi đoạn cùng nhãn clip — điều này **không** tái dựng được ví dụ của [17], và giới hạn đó
   phải là một câu trong báo cáo, không phải một chú thích.
6. **Giảm khung là nơi "đặt sự giám sát", không phải nơi tăng năng lực** (§7.6). [17] chi 2,262 khung
   vào câu trả lời phân loại ở mức khung; ta chi cùng ngân sách vào thực thể và hoãn phân biệt sang
   một head huấn luyện trên 452 nhãn mức clip (~5× ít hơn, thô hơn).
7. **Giảm khung có thể làm hại c1.** Luật của [17] giữ khung có bằng chứng unsafe. c1 UnauthorInt
   khác c5 AuthorInt bằng áo gilê; nếu bộ phát hiện lẫn hai lớp này thì tầng giảm khung sẽ giữ sai
   khung và làm c1 tệ hơn, không tốt hơn. Đây là liên kết trực tiếp tới pha 02 và pha 03.

## Requirements

**Chức năng**
- Một tầng giảm khung **đứng trước** ConvLSTM: vào = clip + trọng số bộ phát hiện (pha 03), ra =
  tensor khung đã giảm, dùng cho **cả** train và test (theo `paper.txt:327`).
- Luật loại khung hai nhánh theo `paper.txt:323`: loại khung có phát hiện lớp safe chiếm trội; loại
  khung không có phát hiện unsafe nào.
- Cắt clip dạng yếu: phân đoạn theo liên tục của bằng chứng unsafe; mỗi đoạn kế thừa nhãn clip; cấu
  hình `min_seg_len`; cờ bật/tắt để đo riêng tác động của *chọn khung* và của *cắt*.
- Giải nhánh (A)/(B) của mâu thuẫn mẫu số bằng một luật tổng hợp tường minh: dự đoán theo đoạn →
  quyết định theo clip (bỏ phiếu / max-conf), để mẫu số **luôn** là 64 và so được với 57/64, 62/64.
- Đo **ba** arm để tách cơ chế: `no-gate`, `gate-only`, `gate+split`. G23 đọc trên
  `gate+split` − `no-gate`.

**Phi chức năng**
- Tái dùng, không viết lại: luật top-q đã có ở `unsafenet/behaviour_clip.py::_score` và điểm
  per-frame ở `frame_scores()`. Chỉ thêm phần **cắt** và phần **nối vào ConvLSTM**.
- `unsafenet/behaviour_clip.py` đang 104 dòng ⇒ phần cắt đặt ở module riêng, giữ < 200 dòng/tệp.
- `frame_scores()` mặc định `device='mps'` — hợp lệ, vì **suy luận tiến** trên xương sống đóng băng
  không chịu lỗi `addcmul_` (`chuong-03-phuong-phap.txt:488`). Chỉ huấn luyện mới cần CUDA.

## Architecture

```
clip (T=32 khung, cache_r2)
    │
    ▼ bộ phát hiện hành vi pha 03  (unsafenet/behaviour_clip.py::frame_scores -> (N,T,8))
per-frame conf 8 lớp
    │
    ▼ luật loại khung hai nhánh (paper.txt:323)        <- MỚI
khung được giữ (số lượng thay đổi theo clip)
    │
    ├─[cờ tắt]──► lấy mẫu lại về T cố định  (như run_yolo_framegate.py đã làm)
    │
    └─[cờ bật]──► cắt theo liên tục bằng chứng  -> các đoạn, mỗi đoạn 1 nhãn clip   <- MỚI
                          │
                          ▼ ConvLSTM (cấu hình thắng G21, pha 02)   <- đường nối MỚI
                   dự đoán theo đoạn
                          │
                          ▼ tổng hợp đoạn -> clip (mẫu số luôn = 64)
                   dự đoán theo clip
```

## Related Code Files

**Sửa**
- `unsafenet/behaviour_clip.py` — phơi `frame_scores()` và `_score()` thành API dùng được từ ngoài;
  thêm luật loại khung hai nhánh bên cạnh `gate` top-q hiện có
- `unsafenet/convlstm.py` — nhận một tensor khung đã giảm thay vì luôn đọc cache đầy đủ (một tham số
  tuỳ chọn, giữ đường mặc định y nguyên để arm A của pha 02 còn tái lập được)
- `run_unsafenet_r2.py` — thêm stage `framereduce`

**Tạo**
- `unsafenet/clip_split.py` (< 150 dòng) — phân đoạn theo liên tục bằng chứng, `min_seg_len`,
  tổng hợp đoạn → clip
- `plans/.../reports/g23-tang-giam-khung-ket-qua.md`
- `plans/.../reports/audit-may-moc-gating-hien-co.md` — biên bản kiểm toán bốn bộ máy ở Key Insight 1

**Không sửa**
- `run_yolo_framegate.py`, `unsafenet/aggregate.py`, `run_relation_transformer.py` — là tạo tác của
  các số đã công bố; kiểm toán nhưng không đổi
- `sweep_results/r2/temporal_aggb*-gate*.pt` — tạo tác lịch sử

## Implementation Steps

1. Xác nhận G22 đạt (hoặc ở vùng 0.85-0.90 kèm nhãn "dưới cổng"). Nếu dưới sàn: **dừng pha này**.
2. Viết `reports/audit-may-moc-gating-hien-co.md`: với mỗi bộ máy, nêu nguồn tín hiệu, ngưỡng tuyệt
   đối hay tỉ lệ, có cắt hay không, chạy trước hay sau mô hình. Kết luận build-vs-reuse tường minh.
3. **Chốt trước** cách giải mâu thuẫn mẫu số (nhánh A hay B, hay cả hai) và ghi vào tiền đăng ký.
4. Chạy `frame_scores()` với trọng số pha 03 → `cache_r2/behaviour_conf_{tag}{split}.npy` cho
   `train` và `test`. Thiết bị: MPS được phép (suy luận tiến).
5. Dựng luật loại khung hai nhánh trong `behaviour_clip.py`; kiểm trên vài clip rằng số khung giữ
   lại giảm theo cùng bậc độ lớn ví dụ của [17] (105 → 40, tức ~38%).
6. Viết `unsafenet/clip_split.py`: phân đoạn theo liên tục, `min_seg_len`, tổng hợp đoạn → clip.
   Unit test: clip toàn-unsafe cho 1 đoạn; clip có khoảng trống giữa cho ≥ 2 đoạn; clip rỗng bằng
   chứng cho fallback là top-k khung (không được trả 0 đoạn).
7. Nối vào ConvLSTM: thêm đường nhận tensor khung đã giảm. Giữ đường cũ nguyên vẹn.
8. Chạy ba arm trên **giao thức val** (83 clip val unsafe) để chọn tham số (`q`, `min_seg_len`,
   luật tổng hợp). **Không** đọc test ở bước này.
9. Phán quyết G23 trên hiệu `gate+split` − `no-gate`, đo ở lần chạm gộp của pha 05 (không ở đây).
   Pha này chỉ **chốt cấu hình** và báo cáo số val.
10. Viết `reports/g23-tang-giam-khung-ket-qua.md`: bảng ba arm trên val, tỉ lệ khung giữ lại, số
    đoạn trung bình mỗi clip, và **giới hạn**: cắt dạng yếu không tái dựng ví dụ đa-lớp của [17]
    vì ta không có nhãn đoạn.

## Todo List

- [ ] Kiểm G22 đạt/sàn trước khi bắt đầu
- [ ] `reports/audit-may-moc-gating-hien-co.md` (4 bộ máy, kết luận build-vs-reuse)
- [ ] Chốt + commit cách giải mâu thuẫn mẫu số (A/B/cả hai)
- [ ] Chạy `frame_scores()` với trọng số pha 03
- [ ] Luật loại khung hai nhánh theo `paper.txt:323`
- [ ] `unsafenet/clip_split.py` + unit test (3 ca: liên tục, có khoảng trống, rỗng)
- [ ] Đường nối khung-đã-giảm → ConvLSTM (giữ đường mặc định nguyên)
- [ ] Stage `framereduce` trong `run_unsafenet_r2.py`
- [ ] Ba arm trên val, chọn `q`/`min_seg_len`/luật tổng hợp
- [ ] `reports/g23-tang-giam-khung-ket-qua.md` + phát biểu giới hạn cắt dạng yếu
- [ ] Kiểm tỉ lệ giảm khung cùng bậc với 105→40 của [17]

## Success Criteria

- Biên bản kiểm toán nêu được, cho từng bộ máy đã có, chính xác nó làm gì và tại sao nó không phải
  cơ chế của [17] — build-vs-reuse có căn cứ, không đoán.
- Tầng giảm khung chạy **trước** ConvLSTM và áp cho **cả** train và test, như `paper.txt:327`.
- Cắt clip tồn tại, có unit test, và mẫu số đầu ra **luôn** là 64 clip.
- Ba arm (`no-gate`, `gate-only`, `gate+split`) phân tách được hai cơ chế: chọn khung và cắt.
- Không đọc nhãn test ở pha này.
- Giới hạn "cắt dạng yếu, không có nhãn đoạn" được viết thành một phát biểu, không phải chú thích.

## Risk Assessment

| Rủi ro | Giảm thiểu |
|---|---|
| **Cắt dạng yếu không phải cơ chế của [17]** ⇒ G23 trượt vì ta dựng sai thứ, không vì cơ chế sai | Ba arm tách được: nếu `gate-only` cho phần lớn lợi ích thì cắt không phải cơ chế chính và §7.6 cần sửa; nếu cả ba ngang nhau thì chẩn đoán §7.6 chưa được chứng thực. Cả hai kết luận đều công bố được. |
| Mẫu số của [17] thực là nhánh (B) ⇒ 57/64 và 62/64 không so được với bất kỳ số nào của ta | Chốt trước và báo cáo cả hai cách giải. Nếu (B) đúng thì đó là **phát hiện về khả năng so sánh của bài báo gốc**, giá trị cao, và nó làm §6.12.5 phải nói mạnh hơn hiện tại. |
| Tầng giảm khung làm **tệ hơn** c1 vì bộ phát hiện lẫn c1/c5 | Đọc per-class sensitivity ở cả ba arm, không chỉ macro. Nếu c1 tệ hơn, nêu như cơ chế, không ẩn trong trung bình. |
| Giảm khung xuống quá ít khung ⇒ ConvLSTM mất tín hiệu thời gian | `min_seg_len` và fallback top-k; kiểm tỉ lệ giữ lại so với ~38% của [17]. |
| Sửa `unsafenet/convlstm.py` làm arm A của pha 02 không tái lập được | Đường mặc định giữ nguyên bit; tham số mới là tuỳ chọn; chạy lại arm A một lần để xác nhận trùng khớp. |
| Quét `q` × `min_seg_len` × luật tổng hợp trên val ⇒ lạm phát do chọn lọc ở mức val | Giới hạn không gian quét, khai báo trước; val là 83 clip nên ghi rõ đây là lựa chọn trên val, không phải số test. |

## Security Considerations

- `cache_r2/behaviour_conf_*.npy` là dẫn xuất confidence, không chứa ảnh ⇒ an toàn, nhưng vẫn
  gitignored theo lệ `cache_r2/`.
- Không đẩy khung ảnh kèm hộp phát hiện ra ngoài repo (ảnh nhà máy thật).

## Next Steps

- Phụ thuộc vào: pha 03 (trọng số bộ phát hiện), pha 02 (cấu hình ConvLSTM thắng G21).
- Chặn: pha 05 (tầng này là arm hợp nhất).
- Theo sau: pha 06 ghi phán quyết G23.

## Unresolved Questions

1. Mẫu số của [17] là nhánh (A) hay (B)? Có đọc được từ bài báo không, hay phải báo cáo cả hai mãi mãi?
2. *"a total of 32 videos passing through the YOLO v4 network"* (`paper.txt:334`) nghĩa là gì? Nếu
   đúng nghĩa chữ, 62/64 được tính thế nào?
3. Nếu `paper.txt:323` đúng là bộ phát hiện **4 lớp** (xem pha 03 câu hỏi #1) thì luật "loại khung
   thuộc lớp safe" không dựng được như viết — vì bộ phát hiện không có lớp safe nào để phát hiện.
   Luật đúng khi đó là gì?
4. Cắt dạng yếu (mọi đoạn kế thừa nhãn clip) có đáng dựng, hay nên bỏ cắt hoàn toàn và báo cáo rằng
   cơ chế của [17] **không dựng lại được** vì thiếu nhãn mức đoạn? Phương án thứ hai trung thực hơn
   và rẻ hơn — nhưng nó bỏ trống đúng thành phần mà §7.6 quy cho +10 pp.
5. Tỉ lệ giữ khung nên chốt trước (ví dụ 38% theo ví dụ của [17]) hay chọn trên val? Chốt trước thì
   trung thực hơn nhưng có thể sai cho dữ liệu của ta.
