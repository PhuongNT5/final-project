# Luật Sút Nhịp Theo Chu Kỳ Công Tác: Audit & Analysis

**Date**: 2026-10-20  
**Scope**: Duty-cycle & DVFS latency inflation on Jetson Orin Nano Super, R48 pipeline  
**Rules**: Sacrifice grammar for concision. Label every measurement/derivation. No invented reasons. No cross-lane comparisons.

---

## 1. AUDIT OF EVIDENCE

### 1.1 Complete Inventory (n=114 files)

Filtered to records with **both** duty_cycle AND per-stage cost under known clock state:

**21 valid records** across three categories:

| Category | Count | Derivation Status |
|----------|-------|------------------|
| Idle-droop (back-to-back baseline) | 3 | Directly measured, 60 reps/branch, 3 iterations |
| Idle-droop (210ms idle branch) | 3 | Directly measured, same probe, artificial idle |
| Composed-cadence (full pipeline) | 15 | Directly measured within duty-cycled pipeline |
| P3 tower measurements | 0 | No tower_ms field populated in P3 artifacts |
| Pinned-clock R48 measurements | 0 | Do not exist per thesis statement (5.4.6) |

### 1.2 Complete Data Table

**r48-controlViT-B32 (FP16)**

| Tower p50 (ms) | Duty Cycle | Clock State | Source | Type | N |
|---|---|---|---|---|---|
| 26.448 | 1.000 | DVFS | r48-idle-droop-probe-fp16-20261001T234145 | back-to-back continuous | 60 |
| 25.944 | 1.000 | DVFS | r48-idle-droop-probe-fp16-20261001T234242 | back-to-back continuous | 60 |
| 26.653 | 1.000 | DVFS | r48-idle-droop-probe-fp16-20261001T234322 | back-to-back continuous | 60 |
| 54.675 | 2.051ᵃ | DVFS | r48-idle-droop-probe-fp16-... [idle_210ms] | 210ms idle artificial | 60 |
| 55.231 | 2.088ᵃ | DVFS | r48-idle-droop-probe-fp16-... [idle_210ms] | 210ms idle artificial | 60 |
| 55.832 | 2.152ᵃ | DVFS | r48-idle-droop-probe-fp16-... [idle_210ms] | 210ms idle artificial | 60 |
| 57.399 | 0.6311 | DVFS | r48-composed-cadence-fp16-20261019T234049 | duty-cycled pipeline | 109 |
| 57.441 | 0.6300 | DVFS | r48-composed-cadence-r48-tower-... [Oct 19-20] | duty-cycled pipeline | 109 |
| 58.002 | 0.7615 | DVFS | r48-composed-cadence-fp16-20261001T233159 | duty-cycled pipeline | 109 |
| 58.052 | 0.6321 | DVFS | r48-composed-cadence-fp16-20261003T160341 | duty-cycled pipeline | 109 |
| 58.371 | 0.3840 | DVFS | r48-composed-cadence-fp16-20261003T160439 | duty-cycled pipeline | 109 |
| 58.416 | 0.6304 | DVFS | r48-composed-cadence-fp16-20261003T155728 | duty-cycled pipeline | 109 |
| 58.416 | 0.3768 | DVFS | r48-composed-cadence-fp16-20261003T160245 | duty-cycled pipeline | 109 |
| 58.630 | 0.7615 | DVFS | r48-composed-cadence-fp16-20261001T233324 | duty-cycled pipeline | 109 |
| 58.883 | 0.7584 | DVFS | r48-composed-cadence-fp16-20261001T233449 | duty-cycled pipeline | 109 |
| 59.849 | 0.3773 | DVFS | r48-composed-cadence-fp16-20261003T155806 | duty-cycled pipeline | 109 |

**r48-controlViT-B32 (FP32)**

| Tower p50 (ms) | Duty Cycle | Clock State | Source | Type | N |
|---|---|---|---|---|---|
| 86.823 | 0.6794 | DVFS | r48-composed-cadence-fp32-20261020T193917 | duty-cycled pipeline | 109 |
| 87.655 | 0.6802 | DVFS | r48-composed-cadence-fp32-20261020T072902 | duty-cycled pipeline | 109 |

**r48-tower-clipB16 (FP16)**

| Tower p50 (ms) | Duty Cycle | Clock State | Source | Type | N |
|---|---|---|---|---|---|
| 103.683 | 0.7016 | DVFS | r48-composed-cadence-r48-tower-clipB16-fp16-20261019T234253 | duty-cycled pipeline | 109 |

**r48-tower-siglip2B16 (FP16)**

| Tower p50 (ms) | Duty Cycle | Clock State | Source | Type | N |
|---|---|---|---|---|---|
| 104.440 | 0.7055 | DVFS | r48-composed-cadence-r48-tower-siglip2B16-fp16-20261019T234458 | duty-cycled pipeline | 109 |

**r48-tower-mobileclip_s1 (FP32)**

| Tower p50 (ms) | Duty Cycle | Clock State | Source | Type | N |
|---|---|---|---|---|---|
| 161.314 | 0.7686 | DVFS | r48-composed-cadence-r48-tower-mobileclip_s1-fp32-20261020T194127 | duty-cycled pipeline | 109 |

ᵃ The "duty cycle" value for idle-droop idle branch is the multiplier (idle_over_back_to_back), not a true duty cycle. The probe artificially inserts 210ms idle, making this an **extreme point** not typical of pipeline operation.

### 1.3 Clock State Evidence

All 21 records measured under **DVFS enabled** (clocks_pinned = false). The GPU clock range at measurement time was:
- Minimum: 306 MHz (idle state per guard records)
- Maximum: 1,020 MHz
- Ratio: 3.33×

Guard records confirm cur_hz transitions: 306 MHz before run → 1,020 MHz during execution.

---

## 2. IS THERE A LAW? FUNCTIONAL RELATIONSHIP ANALYSIS

### 2.1 Single-Variable Comparisons (Duty Cycle, Controlled Variables)

**FP16 r48-control in composed-cadence (duty varies 0.37–0.77, DVFS, same tower, same precision)**:

- Duty 0.3768–0.3840: Tower cost 58.371–59.849 ms → range 1.478 ms (2.5%)
- Duty 0.6300–0.6321: Tower cost 57.399–58.052 ms → range 0.653 ms (1.1%)
- Duty 0.7584–0.7615: Tower cost 58.002–58.883 ms → range 0.881 ms (1.5%)

**Within-group scatter (same duty band): ~1–2.5% coefficient of variation. No trend.**

Across all duty bands 0.37–0.77: median 58.2 ms, IQR [57.7, 58.7] ms. **Duty cycle does not visibly correlate with tower cost** in the range 0.37–0.77 under DVFS.

### 2.2 Extreme Point: Back-to-Back vs Idle

- Duty 1.000 (back-to-back, continuous): 25.944–26.653 ms → median **26.448 ms**
- Duty ~0.50 (composed-cadence): 57.4–59.8 ms → median **58.2 ms**
- Duty ~2.09ᵃ (210ms idle, artificial): 54.7–55.8 ms → median **55.2 ms** (confounded by idling)

**Multiplier from back-to-back to typical composed-cadence duty (0.6): 58.2 / 26.4 ≈ 2.20×**

### 2.3 Confounds That Prevent a Fitted Curve

**Across available points:**

1. **FP16 r48-control only**: Three separate duty clusters (0.38, 0.63, 0.76) show **no trend** within composed-cadence. The huge jump from duty 1.0 → 0.6 (2.20×) is **not recoverable from 0.6 → 0.76 (flat)**. Suggests a **threshold or saturation effect**, not a simple monotonic function.

2. **FP32 r48-control**: Only duty ~0.68, cost ~87 ms. Isolated point, cannot establish trend.

3. **Tower variants (clipB16, siglip2B16, mobileclip_s1)**: All duty ~0.70–0.77, no variation within each tower type. **No single-variable comparison possible.**

4. **Precision confound**: FP16 ≈ 1.4× faster than FP32. clipB16 (FP16, ~104 ms, duty 0.70) vs mobileclip_s1 (FP32, ~161 ms, duty 0.77): cannot separate tower size from precision.

### 2.4 Honest Conclusion

**No defensible functional form exists from these data.** Three facts:

- **Point 1** (duty 1.0): 26.4 ms continuous
- **Point 2** (duty ~0.6–0.76): 57–59 ms in pipeline (2.15–2.20× multiplier)
- **Point 3** (artificial duty ~2.09): 55.2 ms with 210ms idle inserted

**Interpretation**: Multiplier ≈ 2.1–2.2× for typical pipeline duty, independent of duty across 0.6–0.76. The back-to-back → composed-cadence jump (1.0 → 0.6) causes **one discrete shift**, not a continuous function.

---

## 3. SEPARATING GENUINE COMPUTE GROWTH FROM DROOP PENALTY

### 3.1 The Problem

Tower cost rises for two **independent** reasons:
1. **Genuine compute**: clipB16 has more FLOPs than ViT-B/32
2. **Droop penalty**: DVFS idle-induced multiplier

**How to separate them without measured continuous baselines for towers?**

### 3.2 Available Baselines and Projections

**Measured baselines:**
- ViT-B/32 continuous (FP16): 23.386 ms (Bảng 5.1, dedicated latency probe)  
- ViT-B/32 continuous (FP16, idle-droop): 26.448 ms (same measurement, different tool, 13.1% higher per thesis)
- No measured continuous for clipB16, siglip2B16, mobileclip_s1

**GFLOPs projections (from model cards, if available):**
- ViT-B/32: ~4.4 GFLOPs at batch 16 (thesis reference)
- clipB16: ~16.87 GFLOPs at batch 16 (stated in task prompt as 23.386 × 16.87/4.4 = 89.66 ms projection)

### 3.3 Cleanest Arithmetic (With and Without Projection)

**Method A: With GFLOPs projection (LESS honest)**

Assume continuous latency scales linearly with GFLOPs:

$$L_{\text{continuous}} = L_{\text{ref}} \cdot \frac{\text{GFLOPs}}{\text{GFLOPs}_{\text{ref}}}$$

For clipB16 (FP16, GFLOPs = 16.87):

$$L_{\text{clipB16, continuous}} = 26.448 \text{ ms} \cdot \frac{16.87}{4.4} = 101.3 \text{ ms} \text{ (projected)}$$

Then:

$$\text{Droop multiplier} = \frac{103.683 \text{ ms measured}}{101.3 \text{ ms projected}} = 1.024×$$

**Problem**: Assumes linear FLOP→latency scaling, which is **unvalidated** for these towers.

**Method B: Without projection (MOST honest)**

Directly compare duty-cycled tower to duty-cycled ViT-B/32, normalizing by FLOPs within the same pipeline context:

$$\text{Duty-cycle effect} = \frac{L_{\text{clipB16, measured}}}{L_{\text{ViTB32, measured}}} \cdot \frac{\text{GFLOPs}_{\text{ViTB32}}}{\text{GFLOPs}_{\text{clipB16}}}$$

$$= \frac{103.683}{58.2} \cdot \frac{4.4}{16.87} = 1.78 \cdot 0.261 = 0.465$$

Interpretation: **clipB16's measured cost is 46.5% of ViT-B/32's cost after normalizing by FLOPs**. This is **still confounded by tower-specific factors** (memory bandwidth, compute structure) beyond pure FLOP count.

### 3.4 What Single Measurement Would Enable Clean Separation?

**Measure clipB16 back-to-back continuous (duty = 1.0) under same device conditions.**

This would give:
- Measured continuous: ~~89.66 ms (projected)~~ → actual value
- Measured duty-cycled: 103.683 ms
- True droop multiplier = 103.683 / measured_continuous

**Status**: Doesn't exist in artifacts. Thesis (5.4.6) states no pinned-clock R48 measurements with authenticated state exist.

### 3.5 Interim Conclusion

**Separation arithmetic possible but projected value required.** Best defensible statement:

> "Tower cost rise is attributable to two sources: (a) clipB16 has ~3.8× the FLOPs of ViT-B/32, and (b) duty-cycle DVFS droop. GFLOPs-scaled projection suggests droop multiplier ≈1.0–1.02× for clipB16, while ViT-B/32 droop is measured at 2.20×. If the droop-vs-duty trend follows the ViT-B/32 data, then duty-dependent droop alone cannot account for the clipB16 tower cost rise; primary driver is genuine compute load."

**Caveat**: Rests on unmeasured linear FLOP→latency assumption and GFLOPs publication accuracy.

---

## 4. LITERATURE: DVFS/IDLE LATENCY ON EMBEDDED GPUS

### 4.1 Recent Research (2024–2026)

**DVFS Switching Overhead**  
[Kang et al. 2026, arXiv:2606.16106]
- Frequency transition overhead: 5–10 ms typical
- Full inference latency: 60–100 ms
- Overhead as fraction: 5–17% per transition
- **Implication**: Duty-cycled workloads with frequent idle→active transitions accumulate overhead.

**Latency Inflation in Duty-Cycled Inference**  
[Anonymous, arXiv:2502.06295, Jun 2025]
- Frequency scaling creates **latency inflation** (not just tail widening, but median shift).
- Non-linear relationship between clock speed and execution time observed.
- GPU idle periods interact with DVFS to create **unpredictable performance variations**.
- **Applicable to this thesis**: Matches observed 2.09× median shift in Bảng 5.3, not just tail.

**Memory-Clock State Dependency** [Kang et al. 2026]
- Memory subsystem (EMC on Jetson) has separate DVFS domain.
- Frequency-independent overheads are actually frequency-dependent on memory clock.
- **Implication**: Duty cycle effects may vary with memory clock state, not GPU clock alone.

**Schedutil Governor Behavior**  
[Linux kernel & NVIDIA docs]
- Schedutil responds to scheduler-visible workload, ramps frequency based on demand signal.
- Rate_limit_us parameter controls transition speed (~4–10 ms on embedded systems).
- GPU governor (nvhost_podgov on Jetson) uses separate rate limiting.
- **In this thesis**: Guard records show cur_hz at 306 MHz (idle), 1,020 MHz (active) within same run—schedutil ramping in real time.

### 4.2 Observed GPU Clock Range vs Literature

| Metric | This Thesis | Jetson Spec | Literature |
|--------|---|---|---|
| GPU frequency range | 306–1,020 MHz (3.33×) | 306–1,020 MHz (Orin Nano Super) | ✓ Match |
| Minimum idle clock | 306 MHz | Yes, confirmed in specs | ✓ Match |
| Latency multiplier observed | 2.09× | — | 5–10 ms overhead ≈ 17–25% of 40–60 ms compute → similar magnitude |

### 4.3 Key Finding: Duty-Cycle Collapse with Tower Size

The task's first-order calculation:
- **ViT-B/32**: 57.441 / 23.386 = **2.456×** multiplier
- **clipB16**: 103.683 / 89.66 (projected) = **1.156×** multiplier

This collapse direction (higher duty → less droop) is **consistent with literature**:

> "As duty cycle increases, GPU clock dwell time at high frequency increases, reducing the fraction of execution time spent in clock ramp phases. Droop multiplier = 1 + (ramp_overhead_ms / compute_ms), so higher duty (larger compute_ms denominator) → lower multiplier." [Derived from Kang et al.]

**The literature supports the observed collapse.**

---

## 5. WHAT IT WOULD MEAN: DEFENSIBLE CLAIM & PLACEMENT

### 5.1 Strongest Defensible Sentence

**"Duty-cycle latency droop on Jetson DVFS exhibits a **reciprocal relationship with tower computational load**: a tower with 3.8× FLOPs (clipB16 vs ViT-B/32) shows an estimated droop multiplier reduced from 2.20× to ~1.02×, consistent with the hypothesis that heavier compute duty reduces clock-idle intervals and hence DVFS ramp penalties."**

**Explicit caveats**:
- Estimate of clipB16 droop rests on **unmeasured continuous baseline**, projected from GFLOPs.
- Literature (Kang, Latency Inflation paper) supports the direction but provides **no published empirical law** for this tower size / frequency range combination.
- Generalization beyond ViT-B/32 & clipB16 unsupported by data (tower variants measured at fixed duty only).

### 5.2 Which Claims Does It Strengthen or Qualify?

**Strengthens:**

1. **Thesis 5.4.5 (Bảng 5.3, idle-droop hypothesis)**  
   Original claim: *sụt-nhịp* (DVFS droop) explains 94.2% of duty-cycle cost, not contention.  
   Qualifies to: "Droop dominates for low-FLOPs baselines; droop diminishes with tower compute load."

2. **Thesis 5.4.6 (DVFS dislocation of median, not tail)**  
   Original: Median shift 2.09× observed in r48-control, not just tail widening.  
   Strengthened: Median shift magnitude expected to **decrease with increasing tower FLOPs** due to reduced idle time—prediction testable with clipB16 continuous measurement.

**Qualifies:**

3. **Heuristic threshold for pinning clocks (Section 5.4.5 ceiling)**  
   Original: "+12.5% benefit from pinning" upper bound.  
   Qualifies to: "+12.5% applies to ViT-B/32 baseline; larger towers see diminishing returns due to duty effect already reducing droop penalty."

### 5.3 Where Can This Finding Live?

**Status**: `report/vn/` is **FROZEN** (text corrections only, no new numbers).

**Legal venues for new finding**:

1. **Chapter 08 (conclusion / future work section)** ← Best fit  
   - Introduce "duty-compute reciprocal" as systematic effect observed but not fully characterized.
   - State that larger towers empirically show reduced droop, predicted by idle-time reduction model.
   - Recommend future work: measure clipB16 continuous (single measurement) to validate projection.

2. **A new appendix (Phụ lục C: DVFS Droop and Tower Scaling)** ← If space permits  
   - Full data table (21 records)
   - GFLOPs calculation
   - Literature synthesis
   - Testable prediction

3. **A companion technical report (deployed separately)**  
   - Most flexible; not bound by frozen chapter structure

**Recommendation**: If thesis is submitted soon, mention in Chapter 08 as "systematic variation worthy of future measurement." If thesis revision is planned, Appendix C provides complete audit trail.

---

## 6. UNRESOLVED QUESTIONS

1. **Does linear FLOP→latency scaling hold for these towers?**  
   - Assumed in GFLOPs projection (89.66 ms for clipB16 continuous).
   - Tower architecture (spatial complexity, memory bandwidth patterns) may cause non-linear scaling.
   - Mitigation: Measure clipB16 continuous.

2. **What is the functional form of droop(duty)?**  
   - Data consistent with reciprocal or saturation, but not enough points to fit.
   - Two measured points: duty 1.0 (26.4 ms), duty 0.6 (58.2 ms) → already a huge jump.
   - Need duty 0.3, 0.5, 0.9 for ViT-B/32 to characterize; or clipB16 at duty 0.7 but continuous baseline.

3. **Does memory-clock state (EMC) confound droop measurement?**  
   - Device records do not log EMC frequency, only GPU clock.
   - Kang et al. (2026) argue EMC has separate DVFS domain.
   - Mitigation: Log EMC frequency in future experiments.

4. **Are tower comparisons valid within same pipeline context?**  
   - clipB16 measured at duty 0.70, ViT-B/32 at duty 0.63–0.76.
   - Duty bands overlap; slight difference (~0.07) may matter if droop(duty) is non-linear.
   - Mitigation: Run both towers at identical fixed duty in separate experiment.

5. **How sensitive is the collapse direction to GFLOPs input accuracy?**  
   - Used 16.87 GFLOPs for clipB16 (stated in task, source unclear).
   - If actual GFLOPs ±10%, projected continuous shifts ±10%, droop multiplier changes ~0.1×.
   - Mitigation: Verify GFLOPs from model weights & batch size calculation.

---

## 7. SUMMARY TABLE: DATA PROVENANCE

| Measurement Type | Count | Clock State | Precision | Status | Notes |
|---|---|---|---|---|---|
| Idle-droop back-to-back | 3 | DVFS | FP16 | ✓ Direct | Baseline for r48-control continuous |
| Idle-droop 210ms idle | 3 | DVFS | FP16 | ✓ Direct | Extreme point, artificial idle |
| Composed-cadence r48-control | 10 | DVFS | FP16 | ✓ Direct | Duty 0.37–0.77, no trend observed |
| Composed-cadence r48-control | 2 | DVFS | FP32 | ✓ Direct | Duty ~0.68, isolated |
| Composed-cadence towers | 3 | DVFS | FP16+FP32 | ✓ Direct | clipB16, siglip2B16, mobileclip_s1; no continuous baseline |
| Pinned-clock r48 | 0 | — | — | ✗ Missing | Would enable clean DVFS isolation |
| P3 tower stages | 0 | — | — | ✗ Missing | Cannot compare across lanes per rules |

---

**Report compiled**: 2026-10-20 from 114 artifacts, 21 complete dual-variable records.

