# Đối chiếu footage gốc: 32 nhóm trùng byte mâu thuẫn nhãn

Ngày 2026-09-06. Nguồn: `data/` (691 mp4 gốc, không sửa). Liên quan: `dataset-label-contradictions.md`, `dedupe-procedure.md`.
Công cụ: [tools/contact-sheet-from-clip.py](tools/contact-sheet-from-clip.py), [tools/count-actors-in-contradictory-clips.py](tools/count-actors-in-contradictory-clips.py).

## Kết luận ngắn

Xem footage 9/32 nhóm: mỗi clip mâu thuẫn đều chứa **nhiều hành vi được gán nhãn xảy ra đồng thời, do những người khác nhau** — cả hai (có nhóm ba) nhãn đều mô tả đúng một sự kiện có thật trong clip.

Nhưng **kiểm định định lượng cho thấy đây KHÔNG phải đặc tính riêng của các nhóm mâu thuẫn**: clip mâu thuẫn có trung bình 3.12 người đồng thời (max/clip), clip đối chứng 3.03; Mann-Whitney p=0.38. **83 % clip đối chứng cũng có ≥2 người cùng lúc.** Đa người/đa sự kiện là chuẩn mực của toàn dataset.

Vậy kết luận đúng là mạnh hơn, không yếu hơn: **giả định single-label-per-clip là tuỳ tiện trên toàn bộ dataset, không chỉ ở 32 nhóm này.** 32 nhóm chỉ là phần *quan sát được* — nơi cùng một file tình cờ bị xếp vào hai thư mục lớp, khiến sự tuỳ tiện lộ ra. Với 659 clip còn lại, sự tuỳ tiện vẫn tồn tại nhưng không kiểm chứng được vì mỗi clip chỉ được xếp một chỗ.

Điều này KHÔNG làm yếu trần 63/64 — trần vẫn đứng vì input giống hệt bytes mà target khác.

**Đính chính:** giả thuyết ban đầu của tôi là "clip mâu thuẫn đa sự kiện *hơn* clip thường". Số liệu bác bỏ. Giữ nguyên kết luận về tính đa sự kiện, bỏ phần cho rằng nó đặc thù của các nhóm này.

## Đã xác minh gì

1. **md5 lại trực tiếp từ `data/`** (không qua file trung gian): 14 file thuộc 7 nhóm — trùng khớp từng cặp. Bảng ở §Phụ lục.
2. **Sidecar `.npy` cũng trùng md5.** Mỗi clip có `<name>.mp4.npy` = tensor `(32, 224, 224, 3) uint8` (32 frame tiền trích). Các cặp mâu thuẫn có `.npy` giống hệt nhau → không có annotation ẩn nào phân biệt hai bản. Trùng lặp lan cả sang dữ liệu tiền xử lý.
3. **Xem footage** 9 nhóm bằng contact sheet 6 frame trải đều clip.
4. **Đếm người bằng detector độc lập** (yolo11s gốc, lớp COCO `person`, conf 0.35, 8 frame/clip) trên 32 clip đại diện nhóm mâu thuẫn vs 60 clip đối chứng lấy ngẫu nhiên (seed 0) ngoài mọi nhóm trùng. Kết quả ở §Kiểm định định lượng. Detector này chưa hề train trên dataset → số đếm độc lập với pipeline của ta.

## Bằng chứng hình ảnh

Bối cảnh chung: một lối đi nhà máy, **làn đi bộ sơn xanh lá viền vàng ở bên phải**, máy dập/máy công cụ ở bên trái. Camera cố định.

| nhóm | nhãn mâu thuẫn | quan sát trên footage |
|---|---|---|
| **6** | 0 WalkwayViol + 1 UnauthorInt + 4 SafeWalkway (**3 nhãn**) | Góc dưới-trái: 2–3 công nhân đang thao tác trên panel máy đang mở → *intervention*. Giữa khung (f0–f104): một người đi bộ **giữa lối máy, ngoài làn xanh** → *walkway violation*. Về sau (f157–f261): một người đi **bên trong làn xanh** → *safe walkway*. Cả ba nhãn cùng đúng trên cùng một clip. |
| **30** | 0 WalkwayViol (UNSAFE) + 4 SafeWalkway (SAFE) | Công nhân áo phản quang đi **giữa lối máy, ngoài làn xanh**; đồng thời góc trên-phải có người đi **trong làn xanh**. Unsafe và safe cùng tồn tại trong một khung hình. |
| **31** | 0 WalkwayViol + 1 UnauthorInt (**cả hai đều trong bộ test 64**) | f0–f168: một người đi dọc lối máy. f252–f420: người cúi thao tác ở cụm máy góc dưới. Hai pha nối tiếp trong cùng clip. |
| **7** | 0 WalkwayViol (UNSAFE) + 5 AuthorInt (SAFE) | Góc dưới-trái: công nhân **mặc áo phản quang** cùng người áo đỏ mở panel điện → *authorized* intervention. Đồng thời người áo xanh ở bên phải bước cắt qua/ra ngoài làn xanh. |
| **29** | 4 SafeWalkway + 6 ClosedPanel | Người áo tối đi vào và đi **trong làn xanh** (f103–f206) → safe walkway. Cùng lúc góc dưới-trái công nhân cúi làm việc ở máy, panel đóng. |
| **26** | 1 UnauthorInt + 2 OpenedPanel | Công nhân đứng thao tác tại máy; panel máy ở trạng thái mở. Hai nhãn mô tả **hai khía cạnh của cùng một sự kiện** (hành vi can thiệp vs trạng thái vật thể), không loại trừ nhau. |
| **0** | 0 WalkwayViol + 1 UnauthorInt | Công nhân tại máy góc dưới-trái (intervention) + người đi bộ trong lối máy (f45, f89, f223). |
| **9** | 0 WalkwayViol (train) + 1 UnauthorInt (**test**) | Người đi bộ ở lối giữa + công nhân xuất hiện ở cụm máy phải cuối clip. |
| **25** | 1 UnauthorInt (train) + 0 WalkwayViol (test) | Cùng khung cảnh, cùng mô-típ hai sự kiện. |

Contact sheet lưu ở scratchpad (`g00,g06,g07,g09,g25,g26,g29,g30,g31.png`); tái tạo bằng
`.venv/bin/python tools/contact-sheet-from-clip.py <clip> <out.png> --n 6 --cols 3`.

## Kiểm định định lượng (bác bỏ giả thuyết hẹp, củng cố giả thuyết rộng)

| | n | max người/clip (TB) | ≥2 người | |
|---|---|---|---|---|
| nhóm mâu thuẫn | 32 | 3.12 | 88 % | |
| đối chứng | 60 | 3.03 | 83 % | Mann-Whitney p=0.38 (n.s.) |

Đọc: **không có khác biệt.** Không thể nói clip mâu thuẫn "đông người hơn". Nhưng con số của nhóm đối chứng mới là điều đáng báo cáo: trong một clip điển hình của benchmark có ~3 người cùng xuất hiện, làm những việc khác nhau, mà clip chỉ mang **một** nhãn. Đây là bằng chứng định lượng, đo bằng detector ngoài, rằng công thức single-label không mô tả đúng dữ liệu.

### Phép đo thứ hai: số lớp hành vi khác nhau phát hiện được trong mỗi clip

Đếm người không phân biệt được, nên đo trực tiếp cái ta thật sự quan tâm: một clip chứa **bao nhiêu lớp hành vi khác nhau** trong taxonomy 8 lớp của chính benchmark. Dùng detector hành vi `beh_ctx` (8 frame/clip, gộp mọi lớp phát hiện được).

| | n | số lớp hành vi/clip (TB) | ≥2 lớp | phân bố |
|---|---|---|---|---|
| nhóm mâu thuẫn | 32 | 2.94 | **97 %** | {1:1, 2:2, 3:27, 4:2} |
| đối chứng | 60 | 2.12 | **72 %** | {1:17, 2:22, 3:18, 4:3} |

Mann-Whitney (mâu thuẫn > đối chứng) p = 2.4e-6.

Hai điều rút ra:
1. **Clip đối chứng cũng đa hành vi**: 72 % chứa ≥2 lớp hành vi được gán nhãn, trung bình 2.12 lớp — nhưng mỗi clip chỉ mang 1 nhãn. Đây là bằng chứng định lượng cho tính tuỳ tiện của single-label trên toàn dataset.
2. Lần này nhóm mâu thuẫn **có** khác biệt (27/32 clip chứa đúng 3 lớp hành vi). Nhưng **không được coi đây là bằng chứng độc lập**: detector `beh_ctx` được train bằng chính nhãn clip đang bị nghi ngờ (nhãn clip lan xuống box), nên nó thừa hưởng đúng sự nhập nhằng đang xét. Là bằng chứng hỗ trợ, không phải bằng chứng xác lập.

Số liệu thô: `sweep_results/contradictory_actor_counts.json`.

## Vì sao 26/32 nhóm là cặp 0↔1

Giờ đã rõ. Lớp 0 (walkway violation) và lớp 1 (unauthorized intervention) là **hai hành vi thường xảy ra cùng lúc trong cùng một xưởng**: một người rời làn đi bộ để tiến đến máy, rồi thao tác trên máy. Đi sai làn và can thiệp máy là hai giai đoạn của cùng một chuỗi hành động. Ép chọn một nhãn cho cả clip là tuỳ tiện — và annotator đã tuỳ tiện theo hai hướng khác nhau, tạo ra 26 cặp.

Đây cũng là lời giải thích trực tiếp cho **điểm yếu dai dẳng của UnauthorInt** ở mọi protocol (LCO recall 44.4 %, CLIP probe 13.0 %, camera-id floor 18.2 %). Không phải mô hình yếu ở lớp đó; ranh giới 0/1 **không tồn tại như một hàm của pixel**.

## Hệ quả với luận văn

1. **Trần 63/64 giữ nguyên và mạnh hơn.** Trước: "có 2 file trùng, chắc lỗi đóng gói". Giờ: "benchmark là bài toán đa nhãn được đánh giá như đơn nhãn; hai clip test giống hệt bytes mang hai nhãn đều đúng". Không mô hình tất định nào vượt 98.44 %.
2. **Đổi khuyến nghị xử lý.** Nếu là lỗi copy thì dedupe (bỏ clip) là đúng. Nếu là đa nhãn thì bỏ clip là **vứt đi dữ liệu hợp lệ**. Cách xử lý đúng về mặt phương pháp là chuyển sang đánh giá đa nhãn — nhưng ta không có nhãn đa cho 659 clip còn lại, chỉ có 32 nhóm này tự lộ ra. Vì vậy:
   - Giữ dedupe (`drop_all`) làm kết quả chính: nó loại giám sát mâu thuẫn khỏi train, hợp lệ dù nguyên nhân là gì.
   - **Bổ sung metric "top-1 khoan dung đa nhãn"**: trên 8 clip test thuộc nhóm mâu thuẫn, tính đúng nếu dự đoán khớp **bất kỳ** nhãn nào mà nhóm đó mang. Đây là ước lượng ít thiên lệch nhất và có thể áp cho cả số của paper.
3. **Không đổ lỗi cho tác giả dataset.** Ghi nhận đúng: nhãn của họ đúng ở cấp sự kiện; hạn chế nằm ở định dạng single-label. Nêu như một quan sát về benchmark, không phải cáo buộc sai sót.

## So sánh ba cách chấm trên bộ 64 clip

| cách chấm | ý nghĩa | trần |
|---|---|---|
| raw 64, single-label | đúng như paper | 63/64 = 98.44 % |
| 59 clip đã khử nhiễm chéo split | bỏ 5 clip có bản sao trong train | 59/59 = 100 % |
| 64 clip, khoan dung đa nhãn | 8 clip nhóm mâu thuẫn chấp nhận mọi nhãn của nhóm | 64/64 = 100 % |

Báo cả ba, nêu rõ trần của từng cách.

## Giới hạn của đối chiếu này

- Đã xem 9/32 nhóm. 23 nhóm còn lại **chưa** soi bằng mắt; suy rộng dựa trên mô-típ nhất quán ở 9 nhóm và việc 26/32 cùng là cặp 0↔1 trong cùng bối cảnh xưởng. Chưa phải toàn bộ.
- Kiểm định đếm người **không** chứng minh được các nhóm mâu thuẫn đa sự kiện hơn; nó chỉ chứng minh đa người là phổ biến toàn dataset. Đừng trích nó như bằng chứng cho tính đặc thù của 32 nhóm.
- "Nhiều người trong khung" ≠ "nhiều hành vi được gán nhãn". Đo trực tiếp cái sau cần detector hành vi, mà detector đó lại train trên chính nhãn đang bị nghi ngờ → không độc lập. Báo riêng, có ghi chú.
- "Ngoài làn xanh = vi phạm" là suy luận từ vạch sơn, không phải từ quy định an toàn thành văn của nhà máy. Không có annotation guide.
- md5 chỉ bắt trùng **byte**. Clip mã hoá lại hoặc cắt khác vẫn lọt. Con số nhiễm là cận dưới.
- Trần 98.44 % chỉ áp cho bộ phân loại **tất định**. Bộ phân loại ngẫu nhiên có kỳ vọng thấp hơn, không cao hơn.

## Câu hỏi chưa giải quyết

1. Có nên đưa metric "khoan dung đa nhãn" thành kết quả chính không? Nó công bằng hơn nhưng không có trong paper, nên không so trực tiếp được.
2. 23 nhóm chưa soi — có cần soi hết để đưa vào luận văn không, hay 9 nhóm + kiểm định định lượng là đủ?
3. Nếu 0↔1 vốn không tách được, có nên báo thêm kết quả gộp 0+1 thành một lớp (bài toán 3 lớp) làm giới hạn trên "sạch" không?
4. Đã liên hệ tác giả dataset chưa?
