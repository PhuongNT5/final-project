# Kết quả thiết bị: ba tháp vòng 3 trên Jetson Orin Nano Super, 19–20/10/2026

**Loại:** số ĐO mới, nằm ngoài vùng đóng băng 05/10/2026. **Không** sửa `report/vn/`.
**Phạm vi:** đây là phép đo **CHI PHÍ**. Không một con số nào ở đây là số độ chính xác.
**Nhánh:** `next-round` · **git_commit của mọi lượt chạy:** `5644889` · tạo tác: `8610c74`.

## 1. Vì sao có báo cáo này

Vòng 3 đo tám tháp trên máy chủ và không tháp nào qua hiệu chỉnh Benjamini–Hochberg
(ngưỡng hạng 1 = 0.05/8 = 0.00625; p nhỏ nhất 0.093). Câu hỏi còn treo không phải
"tháp nào chính xác hơn" mà **"tháp nào còn chạy được trên bo mạch đích"** — vì nếu các
tháp cho độ chính xác cao hơn đều quá đắt, thì trần của vòng 3 là trần tính toán; nếu
chúng chạy được, trần là trần thống kê. Thực nghiệm này trả lời đúng câu đó.

## 2. Giao thức — khai báo bốn mục

- **(i) Đại lượng:** `steady_state_ms_per_decision.p50` — độ trễ trung vị mỗi quyết định
  trong vòng lặp tổ hợp đã bỏ pha khởi động (97 trong 109 quyết định), và nghịch đảo của
  nó theo quyết định/giây. Kèm theo: `tower_ms.p50` (chi phí riêng tầng tháp trong vòng
  lặp) và `duty_cycle_compute`.
- **(ii) Giao thức/tập đo:** Jetson Orin Nano Super 8GB, JetPack R36.5.0 (`5.15.185-tegra`),
  TensorRT 10.3.0, `MAXN_SUPER` mode 2, **DVFS mặc định — xung nhịp KHÔNG ghim**
  (`clocks_pinned.any = false` trong cả sáu lượt), `schedutil`. Vòng lặp tổ hợp cửa sổ 48 /
  bước 24, 16 khung mỗi quyết định, **không** có bộ phát hiện trong vòng lặp.
  **12 clip / 3,002 khung / 109 quyết định** mỗi lượt. Cổng cửa sổ sạch
  (`jetson-clean-window-guard.py`) đạt **trước và sau** từng lượt; `competing_processes: []`.
- **(iii) Đường cơ sở + trị số:** đối chứng **chạy lại cùng phiên, cùng clip, cùng độ chính
  xác**. Không so với số đã công bố của Chương 5 (24 clip / 201 quyết định — phạm vi khác).
  Cổng triển khai tiền đăng ký: **≥ 3 quyết định/giây**. Hợp đồng nhịp khắt khe nhất của
  kho: **2.500 qđ/giây**; lỏng nhất 1.249.
- **(iv) Nguồn:** `[ĐO]` toàn bộ. Tỷ lệ là `[SỐ HỌC]` trên hai số `[ĐO]`.

## 3. Sáu nhánh đã chạy

| # | Tháp | Độ chính xác | Tháp p50 (ms) | Mỗi quyết định p50 (ms) | qđ/giây | Chu kỳ công tác | Tạo tác |
|---|---|---|---|---|---|---|---|
| 1 | CLIP ViT-B/32 (đối chứng) | FP16 | 57.441 | 169.843 | **5.888** | 0.630 | `...fp16-20261019T234049` |
| 2 | `clipB16` | FP16 | 103.683 | 228.629 | **4.374** | 0.702 | `...clipB16-fp16-20261019T234253` |
| 3 | `siglip2B16` | FP16 | 104.440 | 227.582 | **4.394** | 0.706 | `...siglip2B16-fp16-20261019T234458` |
| 4 | CLIP ViT-B/32 (đối chứng) | FP32 | 86.823 | 202.717 | 4.933 | 0.679 | `...fp32-20261020T072902` |
| 5 | CLIP ViT-B/32 (đối chứng) | FP32 | 87.655 | 204.293 | 4.895 | 0.680 | `...fp32-20261020T193917` |
| 6 | `mobileclip_s1` | FP32 | 161.314 | 281.495 | *3.552* | 0.769 | `...mobileclip_s1-fp32-20261020T194127` |

Nhánh 1–3 là một phiên FP16; nhánh 4–6 là hai phiên FP32. **Không trộn hai độ chính xác.**
Nhánh 6 in nghiêng vì engine của nó không đạt đối sánh — xem mục 6.

Kiểm tra tái lập: đối chứng FP16 phiên này cho 169.843 ms so với 167.291 ms đã công bố —
lệch **+1.53%** trên một tập clip khác hẳn. Vòng lặp tái lập.

## 4. Kết quả chính — hai tháp B/16 vượt cổng triển khai

| | `clipB16` | `siglip2B16` |
|---|---|---|
| qđ/giây | 4.374 | 4.394 |
| so cổng ≥ 3 qđ/giây | **ĐẠT, lề 1.458×** | **ĐẠT, lề 1.465×** |
| so hợp đồng nhịp khắt khe nhất 2.500 | ĐẠT, lề 1.750× | ĐẠT, lề 1.758× |
| so hợp đồng lỏng nhất 1.249 | ĐẠT, lề 3.502× | ĐẠT, lề 3.518× |
| chi phí tháp so đối chứng cùng độ chính xác | 1.805× | 1.818× |
| đối sánh engine FP16 | cos 0.999973 / rel 4.76e-03 **ĐẠT** | cos 0.999918 / rel 4.73e-03 **ĐẠT** |

Hai tháp cho độ chính xác **cao nhất** của vòng 3 (`siglip2B16` 86.39% kiến trúc
transformer, 87.25% với meanpool; `clipB16` 85.82%) **đều triển khai được** trên bo mạch
đích với lề gần 1.5× so cổng tiền đăng ký.

## 5. Hệ quả — trần của vòng 3 là trần THỐNG KÊ, không phải trần TÍNH TOÁN

Phát biểu trung thực nhất mà số liệu này cho phép:

> Tháp cho kết quả cao nhất của vòng 3 **chạy được** trên thiết bị đích (4.394 qđ/giây so
> cổng 3 qđ/giây), nhưng lề độ chính xác của nó **không qua** hiệu chỉnh đa so sánh
> (p = 0.093 so ngưỡng BH hạng 1 = 0.00625). Nên cái chặn kết quả vòng 3 không phải chi
> phí tính toán, mà là **cỡ corpus: 151 nguồn video**.

Ngắn gọn: **thêm tính toán thì mua được nhưng không giúp gì; thêm nguồn thì giúp nhưng
không có.** Đây là lý do việc ưu tiên tiếp theo là ước lượng ICC theo nguồn và số nguồn
cần thêm, không phải tìm tháp rẻ hơn.

**Cập nhật cùng ngày (Q8, đã đo).** ICC theo nguồn **0.43–0.44**, KTC [0.24, 0.57] — trên
ngăn cao nhất của dải tiền đăng ký. Cỡ nguồn cần cho +5 pp là **458–468**, tức **3.0× con
số 151 hiện có** và **xấu hơn** mức 300–400 từng chiếu. Nguyên nhân cấu trúc: **0/151
nguồn mang cả hai nhãn** — một giới hạn luận văn đã tự nêu (`00-tom-tat.txt:213`). Nửa sau
của câu trên vì thế mạnh hơn, không yếu đi: thêm nguồn **giúp ít hơn** mức từng chiếu, và
ngưỡng +5 pp đó còn **lớn hơn** hiệu ứng lớn nhất của vòng 3 (+4.00 pp). Chi tiết:
`analysis-2026-10-20-icc-theo-nguon-va-so-nguon-can-them.md`.

Hai câu **không** được phát biểu từ số này:
1. Không được nói "kiến trúc yếu". Thang B/32 → B/16 → L/14 đơn điệu tăng (83.25 < 85.82
   < 87.42) — tháp lớn hơn **có** tác dụng, chỉ là corpus chưa đủ để chứng minh.
2. Không được nói con số nào ở đây là độ chính xác của vòng lặp trên thiết bị. Hạng mục
   (15) mục 5.9 vẫn nguyên hiệu lực: vòng lặp thiết bị không thấy cùng tập khung với phép
   đánh giá. Phép đo ấy chạy **trên máy chủ**, không cần bo mạch, và chưa được làm.

## 6. `mobileclip_s1`: TensorRT 10.3.0 chuyển đổi sai đồ thị

Giả định ban đầu — tháp nhỏ hơn 4.08× tham số (21,533,632 so 87,849,216) là lựa chọn rẻ —
**sai hai lần độc lập**.

**Thứ nhất, engine sai.** Đối sánh engine-so-ONNX trên khung thật, sai **bất biến** qua
hai độ chính xác và bốn cỡ lô:

| engine | lô 1 | lô 2 | lô 4 | lô 16 |
|---|---|---|---|---|
| FP32, cos / rel | 0.948259 / 0.241 | 0.948259 / 0.241 | 0.941337 / 0.241 | 0.932605 / 0.247 |
| FP16, cos / rel | 0.948011 / 0.241 | 0.947883 / 0.241 | 0.940836 / 0.241 | 0.932087 / 0.259 |

Ngưỡng: FP32 cos ≥ 0.99999 / rel ≤ 1e-4; FP16 cos ≥ 0.999 / rel ≤ 2e-2. Trượt cả hai.
Chuỗi bằng chứng khép lại trên bước chuyển đổi: đồ thị ONNX **đúng** (torch↔ONNX trên máy
chủ rel 1.22e-05, cos 1.000000), đầu vào thiết bị **bit-đồng nhất** (tổng kiểm điểm ảnh
lệch đúng 0), đầu ra engine sai ở **mọi** độ chính xác và mọi cỡ lô. FP32 có độ phân giải
tương đối ~1e-7, nên một cơ chế do độ chính xác phải hiện ra ~0.006%, không phải 24.1%.
⇒ **Tháp không triển khai được qua TensorRT trên thiết bị này ở bất kỳ độ chính xác nào** —
không phải "chỉ không được ở FP16".

**Thứ hai, nó không rẻ.** Ở FP32 cùng phiên cùng cổng: 161.314 ms so 86.823/87.655 ms của
đối chứng = **chậm 1.84–1.86×** trong vòng lặp; đo riêng tầng thì 151.8 so 53.6 ms =
**2.83×**. Đếm tham số chỉ đúng **chiều ngược lại**.

**Hai điều kiện ràng buộc phải đi kèm trị số 161.314 ms ở mọi nơi nó được trích:**
1. Nó là FP32. Mọi đường cơ sở tháp ở Chương 5–6 là FP16. Cặp đối chứng FP32 cùng phiên ở
   nhánh 4–5 là so sánh **duy nhất** mà nó cho phép.
2. Engine trượt đối sánh. Nên đó là chi phí của **một đồ thị không tái tạo được embedding
   của MobileCLIP-S1**. Nó chặn trên chi phí của một engine *đúng* chỉ khi việc chuyển đổi
   sai là trung tính về chi phí — điều chưa biết. Trích được như bằng chứng rằng tháp này
   không phải đường thoát rẻ; **không** trích được như "MobileCLIP-S1 tốn 161.314 ms trên
   Orin Nano".

Chưa định vị op gây lỗi (cần chẻ đồ thị bằng engine cắt ngắn). Kết luận triển khai không
phụ thuộc vào việc đó.

## 7. Phép chiếu GFLOPs sai cả hai chiều

Dự báo trước khi đo (`58.052 × X/4.4` ms, tỷ lệ thuận GFLOPs) so với số đo:

| tháp | dự báo (ms) | đo (ms) | sai số |
|---|---|---|---|
| `clipB16` | 222.3 | 103.683 | **bi quan 2.14×** |
| `siglip2B16` | 197.6 | 104.440 | **bi quan 1.89×** |
| `mobileclip_s1` | 32.9 | 161.314 | **lạc quan 4.90×** |

Cùng một phương pháp sai theo hai chiều đối nghịch. Với hai ViT nó bi quan vì hệ số
sụt-nhịp-khi-rỗi được lấy từ một tháp có chu kỳ công tác thấp hơn, và chu kỳ công tác tăng
theo cỡ tháp (0.630 → 0.702/0.706/0.769), nên tháp lớn nghỉ ít hơn và sụt ít hơn. Với tháp
lai nó lạc quan vì GFLOPs không mô tả được một đồ thị bị runtime phục vụ tồi. **Kết luận
phương pháp: không chiếu chi phí tháp theo GFLOPs trên nền tảng này.**

## 8. Hai điều phải đọc đúng trong tạo tác

- **`p_unsafe_is_meaningful` không có nghĩa "xác suất hợp lệ".** Mã nguồn:
  `s['p_unsafe_is_meaningful'] = args.head == HEAD`
  (`benchmark-r48-composed-cadence.py:196`) — cờ chỉ xét **đầu**, bỏ qua **tháp**. Lượt
  `mobileclip_s1` có `is_published_pair: false` nhưng `p_unsafe_is_meaningful: true`, chỉ vì
  nó nạp đúng đầu B1 đã công bố — trong khi đầu đó được huấn luyện trên đặc trưng B/32 và
  chưa bao giờ thấy embedding của MobileCLIP. Đọc đúng: "đầu thật, không phải đầu giả lập".
  **Không** được đọc là xác suất đúng. Mọi lượt có `is_published_pair: false` thì xác suất
  không dùng được.
- **Mìn số học trong đồ thị `siglip2B16`:** đồ thị xuất chứa **3.64e9** trong phân rã
  GELU-tanh (hạng x³, `layers.9/mlp/activation_fn/Mul_1`), vượt trần 65,504 của FP16 khoảng
  5e4 lần. TensorRT 10.3 **hợp nhất** mẫu đó nên engine vẫn đạt đối sánh — đây là tính chất
  của runtime này, không phải của mô hình. Một runtime khác thực thi phân rã ấy theo đúng
  nghĩa chữ sẽ vỡ. Phải đi kèm mọi khuyến nghị triển khai tháp này.

## 9. Bốn luồng nghiên cứu song song — đã sửa

| Luồng | Giữ được gì | Bỏ gì |
|---|---|---|
| Cơ chế mobileclip | biểu đồ op đo từ ONNX: 1,398 nút, 215 Conv (123 nhóm/depthwise), 80 BN, 72 Identity, so với CLIP 853 nút / 1 Conv / 97 MatMul / 0 BN | toàn bộ phần giải thích "vì sao chậm" như tính chất kiến trúc — câu hỏi đó không còn đúng khi engine sai; và câu "TensorRT 11.1" (thiết bị chạy **10.3.0**) |
| Luật sụt nhịp theo chu kỳ công tác | kết luận âm: **không dựng được hàm nào**. Phần ĐO: chi phí tháp FP16 phẳng ~58.2 ms khi chu kỳ công tác chạy 0.37–0.77, và bước nhảy 26.4 → 58.2 ms (**2.20×**) từ liên tục sang có chu kỳ | trị số sụt 1.02× cho `clipB16` — số học trên một phép chiếu GFLOPs tuyến tính, đúng loại phép chiếu mục 7 vừa bác bỏ. Sáu trích dẫn arXiv chưa kiểm được |
| Bản ghi đối sánh engine | backend đúng là TensorRT (`benchmark-r48-composed-cadence.py:139-154`); ba phán quyết §P5; không có tệp phán quyết JSON cho ba tháp mới | kết luận "mobileclip chạy FP32 vì FP16 trượt" — FP32 **cũng** trượt (bỏ sót §P4) |
| Chỗ được phép viết kết quả | báo cáo riêng trong `plans/` là chỗ duy nhất không rủi ro | đề nghị sửa hạng mục 15 mục 5.9 — hạng mục đó là **độ chính xác**, không phải nhịp; một số chi phí không đóng được nó |

## 10. Việc còn lại

1. Lưu ba phán quyết đối sánh thành JSON theo cấu trúc `r48-parity-fp16-20261001T224028.json`
   — hiện chỉ có ở dạng văn xuôi trong `deployment/JETSON-RUNBOOK-2026-10-19.md` §P5. Cần bo mạch.
2. Hoà hợp một câu: §P5 viết `mobileclip_s1 FAIL and quoted with no latency`, trong khi §P4
   trích 2.83× kèm đủ hai điều kiện ràng buộc. §P4 là bản chính xác; §P5 là bản lỏng.
3. ~~ICC theo nguồn và số nguồn cần thêm~~ — **XONG 20/10**: ICC 0.43–0.44, cần 458–468 nguồn
   cho +5 pp. Xem `analysis-2026-10-20-icc-theo-nguon-va-so-nguon-can-them.md`.
4. ~~Chạy lại BH với trị số EVA02 thật~~ — **XONG 20/10**, và khuyết điểm nhẹ hơn đã ghi: trị số
   thật luôn nằm trong phép tính; chỉ bốn chuỗi văn bản lạc hậu. Kết luận BH không đổi.

## Câu hỏi chưa giải quyết

1. Op nào của MobileCLIP-S1 bị TensorRT 10.3.0 chuyển đổi sai? Chưa chẻ đồ thị. Có đáng làm
   không, khi kết luận triển khai đã đứng mà không cần nó?
2. Việc chuyển đổi sai có trung tính về chi phí? Nếu không, 161.314 ms không chặn trên được
   chi phí của một engine đúng — theo chiều nào thì chưa biết.
3. Chi phí tháp FP16 phẳng theo chu kỳ công tác 0.37–0.77 nhưng nhảy 2.20× từ chu kỳ 1.0: đó
   là ngưỡng hay là bão hoà? Chỉ có ba điểm vận hành rời rạc, không dựng được đường cong.
4. Ba tháp chưa lượt nào ghim xung nhịp. Ghim từng đáng +141.4% trên nhịp tổ hợp của làn P3;
   với tháp lớn hơn (chu kỳ công tác cao hơn, nghỉ ít hơn) lợi ích hẳn nhỏ hơn — chưa đo.
5. `clipB16` và `siglip2B16` ở FP32 chưa đo, nên tỷ lệ FP32/FP16 1.519× chỉ biết cho B/32.
6. 12 clip / 109 quyết định so với 24 clip / 201 quyết định đã công bố: đủ để so nội bộ cùng
   phiên, nhưng chưa lượt nào của ba tháp mới chạy trên đủ tập 24 clip.
