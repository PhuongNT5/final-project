# Rà soát chiến lược gán nhãn multi-entity vs. hiện trạng code

Ngày 2026-09-09. Đối chiếu đề xuất ontology multi-entity với `annotation_guide/`, `build_relation_tokens.py`, và token thực tế trên đĩa.

## 1. Kết luận ngắn

Đề xuất **trùng gần như từng điểm** với `annotation_guide/ontology-multi-entity.md` (viết 2026-07, đã thi hành). Không có điểm nào trong đề xuất là hướng mới. Giá trị của lần rà soát này nằm ở chỗ khác: **spec đã đúng, nhưng thi hành hụt ở ba chỗ, và một chỗ làm hỏng nghĩa của một con số headline.**

| Đề xuất | Hiện trạng |
|---|---|
| 8 lớp thực thể, tên `Person_Authorized`…`Forklift_Overload` | Đã có, đúng từng tên — `annotation_guide/dataset/data.yaml` |
| YOLO detect thực thể, Transformer suy hành vi | Đã có — `build_relation_tokens.py` → `run_relation_transformer.py` |
| Bbox ôm sát bàn chân, foot-point quyết định in/out | Đã có — quy tắc 1 của SOP; feature `foot_in_walkway` dùng `Point(cx, y_max)` + point-in-polygon (shapely) |
| Zone vẽ 1 lần/camera, polygon | Đã có — `annotation_guide/zones/view_*.json` |
| Track-id liên tục, nội suy khi che khuất | Đã có trong SOP (CVAT Track mode) |
| Nối vector ảnh với toạ độ chuẩn hoá | Đã có — token 74 chiều = `[class, xc, yc, w, h, conf, iou_walkway, iou_restricted, foot_in_walkway, norm_dist]` + 64 chiều appearance |

## 2. Lỗ hổng A (nghiêm trọng) — CarryOverload không phải kết quả của attention

Kiểm tra trên token thực tế (`tokens_yolo26app_{train,test}.pt`):

| | có token lớp 7 `Forklift_Overload` | |
|---|---|---|
| clip nhãn 0/1/2 | **0 / 404** (train), **0 / 56** (test) | |
| clip nhãn 3 CarryOverload | **48 / 48** (train), **8 / 8** (test) | 100 % |

Luật một dòng — *"có bất kỳ token `Forklift_Overload` nào → CarryOverload"* — đạt **precision 100 %, recall 100 %** trên cả train lẫn test. Không cần Transformer, không cần attention, không cần quan hệ không gian.

Đây là lý do CarryOverload đạt recall 100.0 % ở **mọi** protocol, kể cả LCO. Con số đó không đo năng lực suy luận quan hệ; nó đo việc một lớp detector trùng khít một nhãn clip.

Nguyên nhân gốc: **lớp 6 `Forklift_Safe_Load` chưa bao giờ được detect** (histogram lớp trên toàn bộ token: chỉ có {0, 1, 4, 5, 7}), dù có 170 box được gán nhãn. Nên phân biệt "≤ 2 khối vs ≥ 3 khối" mà ontology đặc tả **không tồn tại lúc suy luận** — thực tế đã suy biến thành "có xe nâng = quá tải".

## 3. Lỗ hổng B — hình học zone là camera-ID trá hình, và nó rò nhãn

- Zone polygon chỉ có cho view 0, 1, 3, 4, 5, 6 (+ `view_na`). **Thiếu view 2 và view 7.**
- 56 clip thuộc hai view đó (10.9 % của 516) có toàn bộ 4 feature hình học **bằng 0**.
- Toàn bộ 56 clip đó mang nhãn 3; và toàn bộ 56 clip nhãn 3 nằm đúng trong hai view đó.

Tức là *"hình học toàn 0"* ⟺ *"CarryOverload"*, đúng hai chiều, trên cả train lẫn test. Đây là đường rò thứ hai dẫn tới cùng con số 100 %, độc lập với lỗ hổng A.

Sâu hơn: lớp 2 `Safe_Walkway` và 3 `Restricted_Zone` có **0 box** trong dataset YOLO. Zone không được *detect*, chúng là polygon tĩnh tra theo view. Nên `iou_walkway`, `iou_restricted`, `norm_dist` là hàm của (bbox người, **ID camera**). Với một dataset mà Cramér's V(view, label) = 0.784, đó chính xác là kênh mà phân tích confound đang cảnh báo. Hình học zone không phải bằng chứng chống confound — một phần nó *là* confound.

## 4. Lỗ hổng C — mất cân bằng lớp người

| | box gán nhãn | token detect được |
|---|---|---|
| `Person_Authorized` (áo xanh) | 365 | 931 |
| `Person_Unauthorized` (áo đỏ-đen) | 4 497 | 18 487 |
| tỷ lệ | 12.3 : 1 | **19.8 : 1** |

Phân biệt áo xanh / áo đỏ-đen là *toàn bộ* cơ sở để tách UnauthorInt khỏi AuthorInt. Ở tỷ lệ này detector gần như luôn đoán `Unauthorized`, và tỷ lệ lệch **tăng** từ nhãn (12.3) sang suy luận (19.8) — dấu hiệu detector đang thu gọn về lớp đa số.

## 5. Đính chính kỹ thuật — IoU là sai công cụ cho quan hệ "nằm trong"

Đề xuất viết: *"Sự kiện vi phạm xảy ra khi IoU giữa bbox người và vùng an toàn tiệm cận 0."* Đo trên 19 418 token người của tập train:

| | n | IoU(người, walkway) trung vị | trung bình | p95 | max |
|---|---|---|---|---|---|
| chân **trong** vạch | 2 352 | **0.0221** | 0.0495 | 0.198 | 0.411 |
| chân **ngoài** vạch | 17 066 | **0.0000** | 0.0064 | 0.037 | 0.408 |

IoU tiệm cận 0 ở **cả hai** trường hợp. Lý do là số học: mẫu số là hợp của hai vùng, mà vùng lối đi lớn hơn người hàng chục lần, nên dù người đứng trọn trong lối đi thì IoU vẫn ≈ diện_tích_người / diện_tích_lối_đi ≈ 0.02. IoU đo *độ trùng khớp hai vùng cùng cỡ*; nó không đo *quan hệ chứa*.

Công cụ đúng cho quan hệ chứa là cái code **đã dùng**: point-in-polygon trên foot-point (`foot_in_walkway`), hoặc IoA = |giao| / |diện tích người| (chuẩn hoá theo người, không theo hợp). Giữ IoU làm feature phụ thì được, nhưng không được phát biểu nó là tiêu chí vi phạm — bảng trên bác bỏ.

Một lỗi nhỏ nữa: bài viết nói *"toạ độ `y_min` (dưới cùng)"*. Trong hệ toạ độ ảnh, gốc ở góc trên-trái nên cạnh dưới là **`y_max`**. SOP và code đều dùng `y_max`, đúng.

## 6. Khuyến nghị, xếp theo tác động

1. **Báo cáo CarryOverload kèm cảnh báo, đừng tính vào bằng chứng cho attention.** Ghi rõ trong luận văn: recall 100 % của lớp này đạt được bởi một luật một dòng, và macro accuracy 4 lớp đang được nâng bởi một lớp đã bão hoà. Đây là việc *viết*, không phải việc *chạy* — chi phí bằng không, và nếu hội đồng tự tìm ra thì rất khó đỡ.
2. **Vẽ polygon cho view 2 và view 7.** Xoá đường rò "hình học toàn 0 ⟺ nhãn 3". Công sức: 2 polygon. Đây là sửa lỗi rẻ nhất trong danh sách.
3. **Chữa lớp 6 `Forklift_Safe_Load`** (170 box, chưa detect được lần nào). Cần gán thêm nhãn cho xe nâng chở ≤ 2 khối. Nếu dữ liệu gốc không có cảnh đó thì phải nói thẳng trong luận văn rằng phân biệt ≤2/≥3 không kiểm chứng được, thay vì đặc tả nó như đang hoạt động.
4. **Cân bằng lại `Person_Authorized`** — hoặc gán thêm, hoặc dùng class-weight/logit-adjust ở tầng detector.
5. **Không đổi ontology.** Ontology hiện tại đúng. Vấn đề nằm ở độ phủ nhãn và hai polygon thiếu, không ở thiết kế lớp.

## 7. Điều cần nói thẳng

Ba lỗ hổng trên **không** giải thích được khoảng cách so với paper. Sửa hết cả ba thì CarryOverload vẫn 100 % (nó vốn đã 100 %), và các lớp còn lại không được lợi trực tiếp. Giá trị của việc sửa là *tính trung thực của con số*, không phải *độ lớn của con số*. Sau khi vá, macro accuracy nhiều khả năng **giảm nhẹ hoặc giữ nguyên**, vì hiện đang được một lớp bão hoà kéo lên.

## Câu hỏi chưa giải quyết

1. Dữ liệu gốc có cảnh xe nâng chở ≤ 2 khối để gán lớp 6 không? Nếu không có, phân biệt ≤2/≥3 phải bị rút khỏi đặc tả.
2. View 2 và 7 thiếu polygon là do sót, hay do hai view đó thực sự không có vạch lối đi / vùng hạn chế? Nếu là lý do thứ hai, feature toàn-0 là *đúng về mặt vật lý* nhưng vẫn rò nhãn, và cần một cờ "view không có zone" riêng thay vì để 0 mang hai nghĩa.
3. Có gán nhãn thêm không, hay đóng băng dataset và chỉ sửa phần diễn giải? Việc này quyết định phần còn lại của lịch trình.
