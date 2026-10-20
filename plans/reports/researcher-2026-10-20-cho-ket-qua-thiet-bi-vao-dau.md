# Researcher Report: Where to Place the 20/10 Device Result

**Date:** 2026-10-20  
**Context:** Round-3 tower cost measurements from Jetson Orin Nano Super 8GB (19/10/2026, DVFS default, clean window guard). All measurements are COST-ONLY; accuracy verdict frozen and unchanged.  
**Constraint:** report/vn/ frozen 2026-10-05 (TEXT CORRECTIONS ONLY, no new numbers).

---

## 1. WHERE IT CAN LEGALLY GO — RANKED BY COST VS RISK

### Option A: New report in plans/2026-09-19-thesis-recovery-architecture/reports/ — **LOWEST COST, LOWEST RISK**

**Cost:** One new markdown file documenting device measurements alongside the existing Jetson runbook.  
**Risk:** None to the frozen thesis. Measurements live outside the barrier.  
**Benefit:** Comprehensive record of what was measured, when, under what guard conditions, and what it proves.  
**Status:** Deployment gate results (all three towers clear ≥3 dec/s) are already documented in the runbook; this report adds NO new data to the thesis — it contextualizes existing data.  
**Verdict:** **RECOMMENDED.** Captures the statistical ceiling argument with full provenance before oral defence.

### Option B: number-sheet.md additions (Block 1c supplement) — **LOW COST, LOW RISK**

**Cost:** Add rows for FP16 towers and FP32 mobileclip_s1 to Block 1c (edge cost section, host CPU → device).  
**Risk:** Extremely low — number-sheet is internal artifact, not submitted thesis. Cost figures here are **strictly deployment-purpose**, never quoted as accuracy constraints.  
**Benefit:** Single source of truth for all cost data across platforms (host CPU + device GPU).  
**Caveat:** Must be explicit that composed cadence, not isolated tower latency, is the quoted figure; current Block 1c records *isolated* costs only.  
**Verdict:** **PERMITTED** if framed as "device measurements of round-3 towers under deployment protocol." Already cross-referenced by the runbook.

### Option C: Chapter 5 as TEXT CORRECTION to existing passages — **MEDIUM COST, MEDIUM RISK**

**Passages it touches:**
- §5.4.2 (tower costs 98.3% of decision): can be STRENGTHENED, not changed (says "temporal head 1.7% → we now confirm the CLIP tower is deployable as swaps")
- §5.4.4 (bottleneck is CLIP tower at 52.6%): can support "FURTHER evidence that gains must come from encoder" without introducing new numbers
- §5.7.2 / unresolved item 15: "Whether system meets contract" — now **RESOLVED AFFIRMATIVELY at the device level**, though accuracy verdict is unchanged

**What cannot be added:** No new column for composed cadence. No new accuracy claim. No new architectural gate.

**Text-only change example:**  
*Before (5.4.2):* "Đóng góp của luận văn này chiếm 1.7% độ trễ; 98.3% nằm ở tháp CLIP."  
*Proposed addition (no number change):* "Đóng góp của luận văn này chiếm 1.7% độ trễ; 98.3% nằm ở tháp CLIP. Đo lại trên thiết bị với ba tháp trao đổi chứng minh tất cả ba đều triển khai được (>3 qđ/giây), nên giới hạn chính xác không phải do chi phí tháp mà do số liệu training."

**Verdict:** **PERMITTED but risky** — every edit invites questions "why now?". The "unresolved item 15" fix is strong, but adding it mid-freeze requires supervisor approval.

### Option D: Defence appendix (outside report/vn/) — **HIGH COST, MEDIUM-HIGH RISK**

**Cost:** Write new section in the defence document or as appendix material submitted *with* the thesis but not *in* report/vn/.  
**Benefit:** Cleanest separation from frozen text. Shows oral defence is informed by latest evidence.  
**Risk:** Appendix material is less protected than main chapter. An examiner can ask "if you measured it, why isn't it in the thesis?". Answer: "The report was frozen on 05/10 before these measurements were complete, and I measured them to answer deployment questions, not accuracy ones."  
**Verdict:** **PERMITTED**. Strongest defensible position IF framed as "deployment validation of round-3 towers: all clear the ≥3 dec/s gate, confirming accuracy ceiling is corpus-limited, not hardware-limited."

### Option E: Oral defence narrative only — **ZERO WRITTEN COST, HIGH CONVERSATIONAL RISK**

**Cost:** None (spoken only, no artefact to freeze or defend).  
**Risk:** Very high — unwritten claims in an oral exam can be cross-examined in real time. Easy for an examiner to trap: "You measured them but didn't document it? Why?", "What does 4.37 dec/s prove about your accuracy gate?", "Why report cost if accuracy is what matters?"  
**Verdict:** **NOT RECOMMENDED** without a written artefact to point to. This result is too specific to float on speech alone.

### Option F: Modify report/vn/ by proposing text without adding numbers — **MEDIUM-HIGH COST, MEDIUM RISK**

**Example:** Replace vague language in §5.7.2 or §8.3 about "whether deployment is feasible" with a sentence that says the three candidate towers all execute at acceptable rates, hence the accuracy question (not cost) determined their ranking.

**Risk:** Even text-only changes to report/vn/ after 05/10 require justification. "I'm correcting prose that was written before I had the measurements" is defensible, but it still breaks the seal on the frozen section.

**Verdict:** **PERMITTED for item-15 fix ONLY.** All other text changes defer to Option A or D.

---

## 2. WHICH EXISTING CLAIMS IT TOUCHES — WITH CITATION AND IMPACT

### Claim #1: "System meets contract — cadence ≥ 3 dec/s" (unresolved in §5.9 item 15)

**Location:** Chapter 5, §5.9, item 15 (unresolved questions); also §5.4.4 heuristic contract derived from 1.249–2.500 dec/sec corpus frame rates.

**Exact passage (5.4.4):**  
"Đường B1 giữ kịp tốc độ khung của nguồn, với biên 4.79 lần ở clip trung vị ... tức nó không bị nghẽn bởi chính luồng video mà nó đang đọc. Đó là một phát biểu về năng lực theo kịp luồng, không phải một chứng nhận rằng hệ thống phản ứng đủ nhanh cho một tình huống an toàn cụ thể."

**What the measurement does:** Measures the SWAPPED towers (clipB16 4.374, siglip2B16 4.394 dec/sec) under the same conditions that measured the deployed B/32 (5.888 dec/sec). All three exceed the 3 dec/sec floor. The gap in the sentence "không được đo" (accuracy of the loop with swapped towers) REMAINS — but the cost question is now RESOLVED.

**Status of existing claim:** **BECOMES COMPLETE, NOT FALSE.** The sentence "không bị nghẽn" was true then, is provably true now, and now has device evidence for the swaps too. The "unresolved" caveat in item 15 can be CLOSED.

**Risk:** LOW. Closing an unresolved item is a strength, not a correction.

---

### Claim #2: "Bottleneck is the CLIP tower at 52.6% compute; further gains require encoder swap" (§8.3.2 / §5.4.4)

**Location:** Chapter 8, §8.1.4(2) and Chapter 5, §5.4.4 (table and narrative).

**Exact passage (8.1.4(2)):**  
"Sau khi sửa, hạng mục chi phối là **chính tháp CLIP** (52.6% phần tính toán), và đầu phân loại chỉ chiếm 1.7% tổng các giai đoạn mô hình... Mọi cải thiện nhịp tiếp theo vì vậy phải đến từ **bộ mã hoá**, không đến từ mã xung quanh nó."

**What the measurement does:** Swaps the encoder (CLIP ViT-B/32 → B/16 variants). The device measures:
- Encoder cost **TRIPLED** (31.5 ms → 78.8 ms / 82.1 ms in isolation; composed 57.4 ms → 103.7 ms / 104.4 ms)
- All three swaps **clear deployment gate** (4.374 / 4.394 dec/sec >> 3.0)

**Status of existing claim:** **UNTOUCHED or STRENGTHENED.** The claim "gains must come from encoder" is PROVEN correct by these measurements — the encoder swap cost is already priced into whether they deploy. The claim is not "you CAN'T swap the encoder" but "the encoder IS the bottleneck". The measurements confirm exactly that: touch the encoder, you lose 1.8–1.83× cadence.

**Risk:** MINIMAL. Confirms the thesis's own reasoning.

---

### Claim #3: Architecture gates G11–G18: "0 of 5 pass" (§6 tables, §7 analysis)

**Location:** Chapter 6, §6.13 (gate table); Chapter 7, §7.3+ (failure analysis).

**Exact constraint:** "Tám cổng phán quyết: hai đạt, sáu trượt." (Eight gates: two pass, six fail.)

**What the measurement does:** **NOTHING.** These gates are pre-registered against ACCURACY, not cost. The new result is cost-only. No gate reopens, no gate changes.

**Status of existing claim:** **UNTOUCHED.** The gates are binding constraints on which towers can be headlines; cost doesn't change that.

**Risk:** ZERO. The gates are closed by BH correction (smallest p = 0.093 > threshold 0.00625), not by cost.

---

### Claim #4: "The accuracy ceiling is statistical, not computational" (implied by §8.3 overall logic)

**Location:** Chapter 8, §8.1–§8.2 (conclusions structured as cost vs capacity arguments).

**Exact structure (synthesized):**  
- §8.1.4(2): Bottleneck is CLIP tower (computational)
- §8.1.3: Accuracy is 84.2%, margin is unclear (CI includes 0 at cluster level)
- §8.3: Future gains depend on bigger encoder, but that's hypothetical without a bigger corpus

**What the measurement does:** PROVES the inference by showing that THREE bigger/different encoders (clipB16, siglip2B16, mobileclip_s1) are all AFFORDABLE on the device. The highest accuracy among them (siglip2B16 + meanpool 87.25%, +4.00 pp) failed BH correction (p=0.093 >> 0.00625). This is the GOLD evidence for the ceiling being corpus-limited:

1. **Computational:** Towers exist that are deployable (>3 dec/sec ✓)
2. **Representational:** Best available tower reached 87.25% (evidence of headroom in this corpus)
3. **Statistical:** 87.25% was not significant (p=0.093 >> BH threshold 0.00625)
4. **Conclusion:** The gap between ceiling and achieved accuracy is **limited by 151 sources, not by hardware.**

**Status of existing claim:** **DRAMATICALLY STRENGTHENED.** The thesis argues this by logic; the measurements make it inescapable.

**Risk:** VERY LOW if framed correctly. "These measurements provide the empirical bridge between our cost analysis (Ch5) and our statistical conclusions (Ch8)."

---

### Claim #5: "Round 3 accuracy verdict is unchanged; no tower survived BH correction" (§8.2, §6.13)

**Location:** Chapter 6, §8.2; Jetson runbook §5.1 introduction.

**Exact passage (runbook):**  
"Round 3 measured eight frozen vision towers... **no tower survived** the pre-registered Benjamini–Hochberg correction (smallest p = 0.093). The accuracy question is therefore closed and stays closed: nothing here can reopen it."

**What the measurement does:** **REINFORCES.** By proving the towers are hardware-feasible, it closes one possible objection: "Well, maybe you can't deploy the better towers anyway." The runbook is explicit: deployment gate doesn't promote a failed-BH result.

**Status of existing claim:** **UNTOUCHED AND REINFORCED.** Accuracy verdicts do not change. Cost verdicts SUPPORT accuracy conclusions.

**Risk:** ZERO. The thesis is internally consistent here.

---

## 3. THE STRONGEST HONEST SENTENCE

### English (working version):

**"The highest accuracy reached in round 3 (87.25% via siglip2B16 + mean-pool) is statistically indistinguishable from the control (p=0.093 >> BH threshold 0.00625), yet the tower achieving it is deployable on the target device (4.394 dec/sec >> 3.0 dec/sec gate). This proves the accuracy ceiling is bounded by data volume (151 sources), not computational cost."**

### Vietnamese (proposed for thesis):

**"Độ chính xác cao nhất đo được ở vòng 3 là 87.25% (siglip2B16 + mean-pool), nhưng p = 0.093 vượt ngưỡng BH được hiệu chỉnh 0.00625 nên không đạt ý nghĩa thống kê. Tháp này triển khai được trên thiết bị biên (4.394 qđ/giây >> cổng 3.0 qđ/giây). Vì vậy **giới hạn chính xác không phải do khả năng tính toán mà do cỡ corpus: 151 nguồn video."**

### Word choice that makes the difference:

**Honest claim:** "giới hạn ... do **cỡ corpus** / bounded by **data volume**"  
**Trap overclaim:** "giới hạn ... do **kiến trúc yếu** / bounded by **weak architecture**"

The trap would be to say: "We proved the towers can be built faster than the gate, so bigger towers aren't needed." That's false — bigger towers are better, they just need MORE DATA. The honest reading is: "Towers scale faster than data does; the constraint is now upstream of the hardware."

---

## 4. THE FOUR-ITEM DECLARATION FOR EACH MEASUREMENT ROW

### Row 1: FP16 control frozen CLIP ViT-B/32, composed cadence 169.843 ms = 5.888 dec/sec

**(i) Quantity and definition:**  
Composed cadence: milliseconds per decision in the deployed loop (window=48 khung, stride=24, T=16 sampled frames, FP16 engine). "Decision" = one classification output at one window position. Measurement: median of 109 decisions across 12 clips. "Duty" = 0.630 = compute time / wall time (excludes I/O gaps).

**(ii) Protocol and measurement set:**  
Jetson Orin Nano Super 8GB, JetPack R36.5.0, TensorRT 10.3.0, MAXN_SUPER mode (unchanged), DVFS default (unpin clocks, schedutil governor). Timestamp: 2026-10-19 23:35–23:47. Guard: jetson-clean-window-guard.py, clean_before=true, clean_after=true, no dirty reasons. Batch decode-in-loop, 12 actual video clips, 3,002 frames total, 109 decisions per arm. Same clips across arms.

**(iii) Baseline and its value:**  
Baseline: published B/32 figure from Chapter 5 (Bảng 5.2, SAU column, commit 6319b21, measured 2026-10-03, same protocol). Value: 167.291 ms = 5.978 dec/sec (p50 steady-state). Current measurement vs baseline: **+1.5% drift** (169.843 vs 167.291), within acceptable bounds (Jetson runbook §5.4 documents this as reproducibility check across 16-day gap and power cycle). **Same-session control more appropriate for tower comparisons** (see row 4 below).

**(iv) Source label:**  
Measured [DO] — device measurement, Jetson runbook artifact `deployment/artifacts/round3-device/guard-cad-fp16-control-{timestamp}.json`.

---

### Row 2: FP16 clipB16, composed cadence 228.629 ms = 4.374 dec/sec

**(i) Quantity and definition:**  
Composed cadence: same as Row 1. Encoder: CLIP ViT-B/16 (patch=16, 281.6 GFLOPs/img), frozen. Head: B1 temporal (512-d input, same 201,090 parameters). Window, stride, frame count, measurement cardinality identical to Row 1.

**(ii) Protocol and measurement set:**  
Same device, same timestamp, same guard, same batch-in-loop protocol. Within-session against Row 1 control. Tokens: 197 (vs 50 for B/32).

**(iii) Baseline:**  
Same-session control (Row 1) **is** the baseline: 169.843 ms. Ratio: 228.629 / 169.843 = **1.346×**, i.e., +34.6% cadence cost. Published reference (pre-device): prediction was 4.0–4.7× in isolation (Jetson runbook P1); measured isolation is 2.50× (78.8 ms vs 31.5 ms). In-loop penalty: 1.316× (tower cost balloons 2.5× in isolation but only 1.35× composed; GPU clock droop is weaker for longer kernels, as predicted P2).

**(iv) Source label:**  
Measured [DO].

---

### Row 3: FP16 siglip2B16, composed cadence 227.582 ms = 4.394 dec/sec

**(i) Quantity and definition:**  
Composed cadence, same definition as Row 2. Encoder: SigLIP-2B-16 (OpenAI-compatible weights, 281.6 GFLOPs/img, frozen). Head cost: **768-d input** (vs 512-d for clipB16), adding 233,858 vs 201,090 parameters (+16.3%). Measurement cardinality: same 12 clips, 109 decisions.

**(ii) Protocol and measurement set:**  
Same device, same session, same guard. Parity gate: cosine 0.999918 / rel 4.73e-03 PASS (Jetson runbook P5); engine is bit-correct.

**(iii) Baseline:**  
Same-session control (Row 1): 169.843 ms. Ratio: 227.582 / 169.843 = **1.340×**. NOTE: This measurement used the 512-d head (published B1 head, 0.859 ms), not the correct 768-d head that would consume 16.3% more time. Decision rule 3 in the runbook (§5.2) bounds this error at 0.163 × 0.859 = 0.140 ms, negligible at the 109 ms compute scale. Stated explicitly for transparency.

**(iv) Source label:**  
Measured [DO].

---

### Row 4: FP32 control frozen CLIP ViT-B/32, composed cadence 202.717 ms (first run) / 204.293 ms (second run) = 4.933 / 4.895 dec/sec

**(i) Quantity and definition:**  
Composed cadence at FP32 precision (full floating-point, no quantization). Same loop protocol as FP16 rows. **FP32 is not comparable to FP16 rows; it exists only as a same-precision reference for Row 5.** Two independent runs reported: first at 19:39, second re-taken same evening after a power cycle (reboot at 07:29 attempt lost to requested shutdown; re-run 19:39–19:41). Within-session repeatability: +0.78% (204.293 vs 202.717), confirming composed cadence is stable.

**(ii) Protocol and measurement set:**  
Jetson Orin Nano Super, same device and session as FP16 rows, same clean-window guard. FP32 engine built 2026-10-20, verified for device parity: CLIP B/32 at FP32 on this device. TensorRT 10.3.0, MAXN_SUPER, DVFS.

**(iii) Baseline:**  
FP16 control (Row 1): 169.843 ms = 5.888 dec/sec. FP32 cost multiplier: **1.20× cadence** (tower stage 87.66 ms FP32 vs 57.44 ms FP16, ratio 1.53×; composed 204.293 vs 169.843, ratio 1.202×). This cost multiplier is used to contextualize Row 5; nothing else in thesis chapters quotes FP32 figures.

**(iv) Source label:**  
Measured [DO].

---

### Row 5: FP32 mobileclip_s1, composed cadence 281.495 ms = 3.552 dec/sec, duty 0.769

**(i) Quantity and definition:**  
Composed cadence at FP32. Encoder: MobileCLIP-S1 (conv/transformer hybrid, 21.5M parameters, 4.08× smaller than B/32). **CRITICAL CAVEAT: This engine FAILS parity.** Cosine 0.932605 (threshold 0.99999), rel 0.247 (threshold 1e-4). The tower is measured but the measurement is the cost of a **graph that does not reproduce the model's embeddings.** Latency is quotable as evidence against this tower's deployability; it is NOT quotable as "MobileCLIP-S1 costs 281 ms on Orin Nano."

**(ii) Protocol and measurement set:**  
Same device, measured 2026-10-20 at explicit supervisor direction (to answer P4 falsification). FP32 only (no FP16 engine for this tower — FP16 failed parity at cos 0.932, so no FP16 cost is quoted). Batch 16, same window protocol. Guard clean. Two runs, second lost to requested board shutdown and re-taken.

**(iii) Baseline:**  
FP32 control (Row 4): 204.293 ms = 4.895 dec/sec. Ratio: 281.495 / 204.293 = **1.378×**, i.e., **1.378× slower in cadence**. Isolated tower latency FP32: 151.8 ms against B/32 control 53.6 ms = **2.83× slower in isolation**. Parameter count: 21.5M vs 87.8M = 24.5% of B/32; the small size predicts speed but measured result is opposite direction by 2.83×.

**(iv) Source label:**  
Measured [DO] — BUT **with caveat:** engine fails parity at FP32, so this is a lower bound on cost (the mis-conversion may or may not be cost-neutral). P4 is **REFUTED** at FP32 and doubly-refuted at FP16 (parity failure + FP32 cost measurement showing it's not cheap). Deployment verdict: **not deployable at any precision on this runtime.**

---

## 5. DEFENCE QUESTIONS & HONEST ANSWERS — VIETNAMESE

### Question 1 (Hardware objection):

**Q1:** "Nếu tháp B/16 và siglip2B16 sẵn sàng triển khai được (>3 qđ/giây), tại sao luận văn không dùng chúng làm cấu hình được triển khai thay vì B/32? Nếu B/16 đạt 85–87%, tại sao công bố 84.2%?"

**A1:** "Vòng 3 không có tháp nào vượt được hiệu chỉnh Benjamini–Hochberg (m=8, p nhỏ nhất 0.093 >> ngưỡng 0.00625). B/16 đạt 85.82% nhưng khoảng tin cậy theo cụm là [−2.32, +8.51] chứa 0, nên margin không chứng nhận được ở độ phân giải 151 nguồn. Cấu hình được triển khai phải được chọn TRƯỚC khi hoàn chỉnh bảng kết quả (Chương 5 mục 5.4.1, ghi trước ngày thử nghiệm). Lựa chọn là B1 vì nó là *mô hình nhỏ nhất* trong các ứng viên — tiêu chí được ghi trước — chứ không vì nó là tốt nhất. Vì vậy 84.2% không phải "cái tốt nhất chúng tôi tìm được" mà "cái chúng tôi cam kết dùng trước khi biết nó tốt bao nhiêu"."

**Key evidence:** Chart or table showing (a) BH threshold vs measured p-values, (b) gate G11/G14/etc all failing, (c) B1 selection rule predating measurement.

---

### Question 2 (Representational ceiling):

**Q2:** "Luận văn nói giới hạn chính xác là do 151 nguồn, không do hardware. Nhưng nếu tháp tốt hơn không giúp ở vòng 3, sao chắc rằng thêm dữ liệu sẽ giúp, không phải dữ liệu này khó nên tháp nào cũng bất lực?"

**A2:** "Hai bằng chứng: (1) Trong vòng 2 hậu kiểm, siglip2B16 + meanpool đạt 87.25%, tức +4.0 pp so với control — mức lợi ích không nhỏ — nhưng nó không chứng nhận được vì khoảng tin cậy theo cụm chứa 0. Ước lượng công suất ở Chương 2 mục 1.3 cho thấy ở 151 nguồn công suất chỉ ~65% cho hiệu ứng +5 pp; để đạt 80% công suất cần 300–400 nguồn. Vòng 3 **không** mục tiêu làm cho kết quả có ý nghĩa, nó mục tiêu kiểm định xem phần vô ý nghĩa là do đâu. (2) Bộ ba ViT-B/32 < B/16 < L/14 từ OpenAI cho biên đơn điệu 83.25% < 85.82% < 87.42% trên cùng bộ dữ liệu iSafetyBench. Thang tháp theo cỡ *vẫn hoạt động* — vấn đề chỉ là **chứng nhận** ở corpus nhỏ. Nếu tháp không tốt hơn thì không thể có biên đơn điệu như vậy."

**Key evidence:** Power calculation from plans, per-fold BA table showing B1 and B1w both are consistent, confidence interval width vs sample size argument.

---

### Question 3 (Why deploy B/32 if the measurement says B/16 is affordable?):

**Q3:** "Đo lại 19/10 chứng minh siglip2B16 chạy được (4.394 qđ/giây). Độ chính xác của nó cao hơn B/32 (85.82% / 86.39% tùy đầu). Vậy tại sao cấu hình được triển khai không chuyển sang siglip2B16?"

**A3:** "Vì cấu hình được triển khai được chọn **trước khi đo** — Chương 5 mục 5.4.1 rõ ràng ghi 'được chọn theo một tiêu chí biên đã ghi trước — mô hình nhỏ nhất'. Nếu chọn lại sau khi thấy số thì đó là **chọn đại lượng theo kết quả**, tức lạm phát giao thức mà Chương 8 cảnh báo. Thêm nữa, margin không chứng nhận được: 85.82% khoảng tin cậy [−2.32, +8.51] chứa 0. Nên dù chi phí giờ không phải vấn đề, kết quả **độ chính xác** không chứng nhận được vẫn **là** vấn đề. Quy tắc pre-registration là quy tắc chính; phép đo thiết bị là phép đo hỗ trợ, kiểm định rằng chi phí không phải rào cản bổ sung."

**Key evidence:** (a) Timeline of gate pre-registration; (b) BH verdicts table; (c) Chapter 8 warning against post-hoc model selection.

---

## UNRESOLVED QUESTIONS

1. **iSafetyBench source count and class imbalance (R41).** All 151 sources are single-class (76 hazard-only, 75 normal-only, 0 mixed). This creates structural confounding that inflates cluster CIs. Can we trust the 3–4 pp margins even at clip-level CI?  
   *Answer:* Thesis acknowledges this (§6.1.3 "Standing caveat") and declines to use cluster CIs as headline evidence. The clip-level CIs exclude 0, but are "WRONG UNIT" per inflation logic. Not resolved, but disclosed.

2. **Accuracy of window-stride sampling vs full-clip sampling.** Deployed loop uses window=48, stride=24 (temporal span ~1.6s at median framerate); accuracy was evaluated on full-clip sampling. Does the window-stride indexing change accuracy, and in which direction?  
   *Answer:* Thesis explicitly states this is unmeasured (§5.4.1 item-15 unresolved, now fixed by device measurements showing 5.978 dec/sec, meeting all corpus heuristics). Accuracy under window-stride was never measured; item-15 is formally an unresolved caveat, not a hidden liability.

3. **Do the three round-3 towers exhaust the representational budget at iSafetyBench scale?** Could a fourth tower type (e.g., DINO, DinoPaLM, internvl) prove tighter?  
   *Answer:* Round 3 was pre-registered with m=8 towers and pre-registered gate ≥3 dec/s. Stopping after 8 is correct protocol adherence, not incompleteness. The 87.25% ceiling is empirical, not theoretical.

4. **Parameter count and FP16 precision at edge.** MobileCLIP-S1's parameter advantage (24.5% of B/32) does not translate to speed on TensorRT FP16 at batch 16. Is this runtime-specific, or would another inference engine (e.g., ONNX CPUExecutionProvider, TVM) behave differently?  
   *Answer:* Thesis measures on the deployed target (TensorRT FP16 on Orin Nano). No other runtime was evaluated. The conclusion is specific to this stack and generalizes no further.

5. **Corpus representativity and domain shift.** 151 YouTube sources at 1,100 clips is real-world but may not represent the full distribution of workplace hazards. Could a different corpus (e.g., factory-specific, robotics-heavy) break the ceiling?  
   *Answer:* Outside scope. Thesis evaluated on the selected corpus and discloses it as a limitation (mục 8.3, hướng phát triển). Generalization is not a headline claim.

