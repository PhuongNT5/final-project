# TIỀN ĐĂNG KÝ — G21…G25: BASELINE GIAO THỨC BÀI BÁO GỐC (ref [17])

**Viết và commit TRƯỚC khi chạy bất kỳ lượt huấn luyện nào của chương trình này. Không một con số
mới nào tồn tại khi dòng chữ này được ghi.** Các số đã tồn tại trước đó được liệt kê nguyên văn ở
§2 để không thể bị trình bày lại như kết quả mới.

| | |
|---|---|
| Ngày | 2026-10-11 |
| Nhánh | `next-round` |
| Commit nền | 4598bc5 |
| Làn | **C duy nhất** (Unsafe-Net, 691 clip). Không trộn làn A/B — `chuong-06-danh-gia.txt:82-85` |
| Căn cứ | `plan.md` cùng thư mục; yêu cầu người dùng 11/10 (tái lập ConvLSTM + hợp nhất; thêm phép đo nhị phân) |

---

## 1. XUNG ĐỘT DỮ LIỆU ĐƯỢC CHỐT TRƯỚC: "NHỊ PHÂN TRÊN 64 CLIP TEST" LÀ BẤT KHẢ

Yêu cầu nêu "kết quả nhị phân An toàn/Không an toàn trên 64 clip test". **64 clip test đó toàn bộ
đều là Không an toàn** — chúng là tập con 4 lớp nguy hiểm (`data_index.load_clips`, đếm được:
test = 32/11/13/8 cho lớp 0–3). Trên đúng 64 clip ấy, tác vụ nhị phân suy biến: chỉ đo được
**độ nhạy (recall) lớp Không an toàn**, không có specificity, không có BA, không có PR-AUC.

Giải quyết đã chốt, hai phép đo báo cáo song song, **không được gộp**:

| Ký hiệu | Tập | Thành phần | Đại lượng chính |
|---|---|---|---|
| **NP-125** | 125 clip test đầy đủ (đóng băng) | 64 Không an toàn / 61 An toàn | **BA** = trung bình recall hai lớp |
| **NP-64** | tập con 64 clip của bài báo | 64 Không an toàn / 0 An toàn | **chỉ** recall Không an toàn |

NP-64 **không phải** một phép đo nhị phân; nó là lát cắt của NP-125 để so trực tiếp với các con số
theo lớp của [17]. Mọi bảng phải mang nhãn này.

### Sàn và trần của NP-125, chốt trước

- **Sàn đa số**: 64/125 = **51.20%** top-1 (BA của bộ phân loại hằng = 50.00%).
- **Trần cứng 124/125 = 99.20%**: `0_te23.mp4` và `4_te9.mp4` **giống nhau từng byte** nhưng nhãn
  nhị phân đối nghịch, **cả hai nằm trong test** — mọi bộ phân loại tất định sai đúng một clip.
  Nguồn: `unsafenet/dedupe.py:32-41`, `sweep_results/duplicate_audit_binary.json`.
- Chính sách trùng lặp của **cấu hình tiêu đề**: train `drop_conflict` (566 → **562**: bỏ
  `0_tr123, 0_tr136, 0_tr138, 1_tr55`), test `keep` (**125 clip đóng băng**).
  Phân tích độ nhạy, **không bao giờ là tiêu đề**: test `drop_conflict` → 120 clip (63/57), khi đó
  trần là 100%. Lý do: bỏ clip test phá vỡ tính so sánh được và lấy đi 4 clip của lớp An toàn vốn
  đã nhỏ hơn (`unsafenet/dedupe.py:52-62`).

## 2. SỐ ĐÃ TỒN TẠI TRƯỚC KHI TIỀN ĐĂNG KÝ NÀY ĐƯỢC VIẾT

Ghi nguyên văn để không bị tái trình bày như kết quả của chương trình này:

| Mốc | top-1 | macro (trung bình recall lớp) | Nguồn |
|---|---|---|---|
| ConvLSTM của ta, 4 lớp, `device='mps'` | 50/64 = 78.12% | **75.64%** | `sweep_results/r2/temporal_convlstm.pt` (06/09 01:01, **bị gitignore**, không được trích dẫn ở đâu trong `report/vn/`) |
| — theo lớp | c0 27/32 = 84.38%, **c1 2/11 = 18.18%**, c2 13/13 = 100%, c3 8/8 = 100% | | đo lại 11/10 từ `Pt`/`yt` |
| ConvLSTM của [17] (**họ báo cáo**) | 57/64 = 89.06% | 85.78% | Table 5; `config.PAPER['nobbox']` = `[31/32, 9/11, 10/13, 7/8]` |
| Hợp nhất của [17] (**họ báo cáo**) | 62/64 = 96.88% | 95.81% | Table 6; `config.PAPER['full']` |

Thiếu hụt 10.14 pp macro phân rã: **c1 UnauthorInt góp +15.91 pp**, c2 và c3 của ta **vượt** họ
(100% so với 76.92% và 87.50%). Chân 1 là bài toán **một lớp**.

## 3. G21 — CHÂN 1: TÁI LẬP HAY BÁC BỎ MỐC 57/64

**Can thiệp: đúng một biến** — thiết bị huấn luyện. `unsafenet/convlstm.py:137` mặc định
`device='mps'`; luận văn tự ghi rằng macOS 14 làm hỏng **âm thầm** `addcmul_` của Adam
(`chuong-03-phuong-phap.txt:484-487`). Lượt 06/09 có thể đã đo một optimizer hỏng, không phải một
ConvLSTM. Giữ nguyên tuyệt đối: `px=112, t=16, seeds=5, epochs=18, bs=8, lr=3e-4, n_cls=4`,
`blind_split(val_frac=0.18, seed=0)`, cùng cache `cache_r2/raw_{train,test}_112px_t16.npy`.

**Luật quyết định — chốt trước khi thấy số.** Đọc trên tác vụ 4 lớp, 64 clip, quy tắc quyết định
**ensemble argmax trên 5 seed** (giống hệt tạo tác 06/09 để so sánh cùng loại; **không** dùng
`blind_pred` best-val — sổ chạm test ghi rõ hai quy tắc này cho ra số khác nhau):

| Kết quả | Kết luận đã chốt trước |
|---|---|
| top-1 **≥ 54/64** VÀ biên McNemar khả thi so 57/64 chứa p ≥ 0.05 | **Tái lập được.** Khoảng cách 06/09 là tạo tác thiết bị. `temporal_convlstm.pt` cũ bị rút, không được trích dẫn. |
| top-1 48–53/64 | **Mơ hồ.** Không tuyên bố gì về chân 1 cho tới khi chạy hết H1 (val_frac) ở pha 02. |
| top-1 **≤ 47/64**, hoặc macro trong ±3 pp của 75.64% | **H3 bị bác bỏ:** thiết bị không phải nguyên nhân. Mốc 85.78% mang nhãn "chỉ đọc từ bảng của [17], ta không tái lập được", và §6.12.5 phải sửa chữ ở `:1525`. |

**Dự đoán khả bác bỏ, sắc hơn số tiêu đề.** Nếu nguyên nhân là MPS hỏng thì phải thấy **cả hai**:
1. recall c1 UnauthorInt **≥ 6/11** (hiện 2/11);
2. dải macro giữa 5 seed **không** rộng hơn hiện tại (hiện val macro 78.41–80.18).

Nếu c1 vẫn **≤ 3/11** trên CPU thì nguyên nhân nằm ở dữ liệu/nhãn của riêng lớp đó, không ở thiết
bị — và điều đó được ghi là kết quả dương, không phải thất bại.

## 4. G25 — PHÉP ĐO NHỊ PHÂN An toàn/Không an toàn (MỚI, theo yêu cầu 11/10)

Tác vụ: `y = 1` nếu nhãn nguồn ∈ {0,1,2,3}, `y = 0` nếu ∈ {4,5,6,7}. Huấn luyện trên **toàn bộ**
562 clip train (448/114) — khác tác vụ 4 lớp, nơi clip An toàn chỉ vào nhánh fit. Val cắt từ train
bằng `blind_split` phân tầng theo (nhãn 8 lớp, view) rồi mới nhị phân hoá, để mọi lớp hành vi có mặt
ở hai phía. Test: NP-125 đóng băng. Quy tắc quyết định: ensemble argmax 5 seed.

**Luật quyết định — chốt trước khi thấy số.** Đọc trên **BA**, kèm CI Wilson trên top-1 và
McNemar chính xác so đối chứng scene-only:

| Kết quả | Kết luận đã chốt trước |
|---|---|
| BA **≥ 66.2%** (= sàn đa số 51.2% + 15.0 pp) VÀ McNemar so scene-only p < 0.05 | Tác vụ nhị phân **có tín hiệu thật vượt cả đa số và máy quay**. Đây là con số được báo cáo. |
| BA ≥ 66.2% nhưng **không** phân biệt được với scene-only | Báo cáo là **không phân biệt được với việc đoán theo máy quay** — đúng họ với phát hiện 52/64 = 81.2% của §6.5. |
| BA < 66.2% | Không chứng nhận. Báo cáo nguyên trạng. |

**Điều kiện tiên quyết bắt buộc, chạy cùng lượt, không được bỏ:** đối chứng **scene-only** (chỉ
dùng id view máy quay, `sweep_results/zone_prior/clip_view_map.json`) và **đa-số-theo-view** dưới
đúng NP-125 và đúng chính sách trùng lặp. Không có đối chứng này thì mọi số nhị phân vô nghĩa:
8 máy quay cố định, 3/8 view đơn lớp (`chuong-06-danh-gia.txt`, §6.5).

**NP-125 là giao thức cùng-phân-bố.** Nó **không** đo độ lạm phát giao thức. Phiên bản LCO-691 của
đúng tác vụ nhị phân này là phép đo lạm phát và thuộc pha 05; CI bootstrap theo cụm (đơn vị = nhóm
bối cảnh, n = 8) chỉ áp cho phiên bản LCO, không áp cho NP-125.

## 5. G22 / G23 / G24 — giữ nguyên như `plan.md`

- **G22** bộ phát hiện hành vi: `mAP50 ≥ 0.90` (cổng), sàn `0.85` — đã tiền đăng ký sẵn trong
  `kaggle-detector-retrain/train-behaviour-detector-gpu.py`. Arm trung thành 416 (như [17]) và arm
  800 báo cáo tách biệt.
- **G23** tầng giảm khung + cắt clip: tái dựng được cơ chế +10 pp của [17] hay không.
- **G24** hợp nhất hai giao thức: đối chứng, **không phán quyết** (xem `plan.md`).

## 6. NGÂN SÁCH CHẠM TEST

- Đã tiền đăng ký: **2**. Đã tiêu: **0** (`chuong-04-chi-tiet-thuc-nghiem.txt:949-973`).
- Chương trình này tiêu **đúng 1 lần chạm, gộp**: G21 (4 lớp/64) + G25 (NP-125) + đối chứng
  scene-only được chấm **trong cùng một lượt**, một lần, sau khi mọi lựa chọn đã xong trên val.
- **Không** dừng sớm theo test. `_run_seed` chọn state tốt nhất theo **val macro**, không theo test
  — đã kiểm `unsafenet/convlstm.py:128-132`. Không được thêm đường nhiễm thứ ba
  (`chuong-06-danh-gia.txt:1567-1570`).
- Lượt 06/09 được ghi **hồi tố** vào Section A của `reports/test-touch-log.md` (không tiền đăng ký,
  không thể xác minh), theo đúng tuyên bố trung thực ở đầu sổ.

## 7. ĐIỀU KHÔNG ĐƯỢC LÀM SAU KHI THẤY SỐ

Nới biên, đổi quy tắc quyết định, đổi chính sách trùng lặp, đổi mô hình được cổng đặt tên, chọn
seed, hay đổi `val_frac` rồi báo cáo như thể đã chốt trước (`chuong-06-danh-gia.txt:150-152`).
Mốc đóng băng 05/10 vẫn hiệu lực: `report/vn/` chỉ được **sửa chữ**; mọi số mới sống trong thư mục
kế hoạch này.
