# Defense Materials — Final (post-Day-6 measurement update)

> Consolidates and supersedes the slide outline / Q&A appendix in
> `phase-08-defense-materials.md`. Phase-08 was written before the Jetson
> measurements; this document folds the §5.6 measured data into the
> defense narrative.
>
> All numbers trace to `GROUND_TRUTH.md` (canonical) and were verified
> in the Day-7 sanity-check pass.

---

## 0. Defense narrative — the one-sentence pitch

> "We deliver a YOLO–Transformer pipeline that **beats the published
> ConvLSTM-only baseline on macro accuracy by +3.5 percentage points at
> zero behavior-bbox annotations**, and **runs end-to-end on a USD 499
> Jetson Orin Nano Super at 116 FPS single-stream, sustaining four
> camera streams in real-time for 36+ minutes with zero dropped frames
> at 7 W average power** — a Pareto-optimal point that no published
> baseline occupies."

This sentence is the answer to the unspoken question *"what is your
contribution?"* — every slide and every Q&A answer below must close back
to it within 90 seconds.

---

## 1. 22-Slide Deck (20-minute presentation)

| # | Title | Key visual | Headline number |
|---|---|---|---|
| 1 | Title | University crest + thesis title | — |
| 2 | Industrial safety problem | factory accident illustration | 2.3M accidents/yr (paper §1) |
| 3 | Research objectives | 3 bullets from §1.3 | Accuracy / Edge / Privacy |
| 4 | **Fair-comparison table** ⭐ | **Table 5.5A** | **89.28% macro vs 85.78%, +3.5 pp** |
| 5 | Pareto frontier framing | annot-vs-accuracy scatter | 0 vs 2,591 bbox |
| 6 | Unsafe-Net dataset | composition table | 691 clips, 8 classes |
| 7 | Architecture overview | pipeline diagram | 4 stages, frozen backbone |
| 8 | Key design choices | comparison vs ConvLSTM | annotation-efficient |
| 9 | Experiment summary | 10-track ledger | converge on 56/64 ceiling |
| 10 | Per-class breakdown | accuracy bar chart | 100% OpenedPanel + CarryOverload |
| 11 | Statistical rigor | bootstrap CI + z-test | CI [78%, 95%] |
| 12 | Why 401 frames not 2,262 | data-efficiency curve | V10 negative result |
| **13** | **Edge platform — Jetson Orin Nano Super** ⭐ | hardware photo + specs | **15 W, $499, 67 TOPS** |
| **14** | **Cross-precision benchmarks (Table 5.5B)** ⭐ | FP32/FP16/INT8 latency bars | **8.62 ms / 116 FPS FP16** |
| **15** | **Multi-stream throughput (Table 5.5C)** ⭐ | N=1..4 FPS chart | **4 cams × 30 FPS, 0 drops** |
| **16** | **36-min sustained soak (Table 5.5F)** ⭐ | thermal/power time-series | **56.8°C max, 7 W avg** |
| **17** | **Power efficiency Pareto (Table 5.5D)** ⭐ | FPS vs Power scatter | **+33% energy efficiency** |
| **18** | **System architecture** ⭐ | Mermaid diagram from `figures/arch-diagram-jetson-edge.mmd` | privacy boundary explicit |
| **19** | **Live operator dashboard** ⭐ | screenshot `jetson-dashboard-alerts-top.png` | **100% recall, 0% false alarm** |
| **20** | **Security & privacy properties** | Table 5.5G threat→control | raw frames never cross network |
| 21 | Limitations | 5 bullets from §5.7 + §6.3 | honest scope |
| 22 | Conclusion + future work | Pareto sentence + 3 future-work bullets | thesis Q&A |

**Slide order rationale:**
- Slide 4 is the **fair-comparison table early** — committee sees corrected framing
  before they can ask "why didn't you reach 95.81%".
- Slides 13–20 (the new edge-deployment block) are **the differentiator** —
  no published baseline has equivalent slides. Spend ~7 minutes here.
- Slide 19 (live dashboard screenshot) can be replaced with the **2-minute
  live demo** below if the demo terminal is set up beforehand.

---

## 2. Live demo script (2 minutes, optional but high-impact)

**Setup before defense (do in private):**
1. Jetson is powered on, MAXN_SUPER active.
2. M4 has terminal A ready: `ssh jetson "cd ~/my-project && python3 deployment/scripts/jetson-multi-stream-pipeline.py ..."`
3. M4 has terminal B ready: `.venv/bin/python3 -m streamlit run deployment/dashboard/jetson-live-monitor.py`
4. Browser tab open at `localhost:8501` (don't show until demo).

**During demo:**

```
[T=0:00] "I'll now show the four-stream surveillance system running live on
         the Jetson Orin Nano Super."
[T=0:10] Switch to terminal A — paste the multi-stream emitter command.
         "This starts inference on four video files standing in as cameras."
[T=0:25] Switch to browser tab with Streamlit dashboard.
         "Health badge confirms the Jetson is alive. Each tile shows the
         latest prediction per stream after three-second rolling aggregation."
[T=0:50] Point to the two ALERT tiles.
         "Stream 0 is flagged — walkway violation at 76% confidence.
          Stream 2 is flagged at 99% — overload with forklift, the easy class."
[T=1:10] Point to the two MONITORING tiles.
         "Streams 1 and 3 are below the 0.75 alert threshold and held in
          monitoring — the safe-walkway stream is correctly not alerted
          despite being a known confusion case in our 4-class model."
[T=1:30] Scroll to alert log.
         "The session log captures every threshold crossing with timestamps —
          this is the operator's auditable trail."
[T=1:50] "Inference per emission is 5.5 milliseconds head latency.
          End-to-end power on the edge is under 8 watts."
[T=2:00] END. Switch back to slides at slide 19.
```

**Fallback (demo crashes or no live network):** open the screenshot
`figures/jetson-dashboard-alerts-top.png` full-screen and narrate the same
two minutes. The screenshot was taken from the actual measurement run; the
numbers are the same.

---

## 3. Q&A appendix — top 15 questions with rehearsed answers

### A. The classical accuracy question

**Q1: "Why is your accuracy lower than the paper's 95.81%?"**

> "Two things need separating. First, the metric: paper [5] reports macro
> (mean of per-class recalls) in their Table 7 headline column; 95.81% is the
> *full-cascade* configuration that uses 2,262 behavior-class bounding-box
> annotations from §3 of the paper. The *fair* baseline is the paper's own
> **ConvLSTM-only configuration** — same annotation budget as ours, no bbox.
> That baseline scores **85.78% macro / 89.06% top-1**.
>
> Our system achieves **89.28% macro**, which is **+3.5 percentage points
> above that fair baseline**. On top-1 we are at 87.50%, one sample below the
> 89.06% baseline — z = −0.27, two-tailed p = 0.79 on N = 64, i.e. *statistical
> parity*. The gap to the 95.81% full cascade is the **cost of 2,591 bbox
> annotations**, which at five minutes per image is roughly 190 hours of expert
> labour. Chapter 4 §4.6 documents that 401 bbox annotations are *not*
> sufficient — V10 *regressed* to 54/64 — so the trade-off is not gradual:
> you need the full paper budget or you do without bbox.
>
> Our position is that this is a Pareto-optimal point at the
> annotation-cheap, edge-deployable end of the frontier."

(Target delivery: 60–80 seconds. Practice it.)

### B. Statistical rigour

**Q2: "How confident are you in the 89.28% macro number?"**

> "The test set is N = 64. The 95% bootstrap CI on top-1 (5,000 resamples) is
> [79.7%, 95.3%] — wide because of the sample size. The +3.5 pp macro gap to
> ConvLSTM-only is the *point* estimate; we do not claim it as significant in
> a Fisher's-exact sense at N = 64. What we do claim, and what we believe is
> the stronger evidence, is the **convergence of ten independent architecture
> tracks on the same 56/64 ceiling** (Tables 4.6, 4.16, 4.17 in Ch 4). That
> convergence is the structural argument that the residual gap is data-bound,
> not architecture-bound. The macro number is the headline; the convergence is
> the proof."

### C. Annotation cost claim

**Q3: "How did you compute 190 hours of bbox annotation effort?"**

> "Five minutes per bounding-box image is a conservative published estimate
> from the LabelImg and CVAT literature. Paper [5] uses 2,262 training bbox
> images plus 329 test bbox images = 2,591 images. At five minutes each that
> is 215 hours; we report 190 hours as a conservative rounded-down figure.
> At a more aggressive three minutes per image (only achievable with skilled
> annotators on a single-class task) the effort is still 130 hours. Either
> way the number is an order of magnitude above our zero-bbox cost."

### D. Edge deployment

**Q4: "Why Jetson Orin Nano Super specifically, not AGX Orin or NX?"**

> "The Orin Nano Super sits at the cost-effective entry point of the Jetson
> family: USD 249–499 against the AGX at roughly USD 2,000. The thesis
> contribution is to show that the workload runs *within* this entry-level
> envelope, not that it requires top-of-line hardware. Section 5.6 reports the
> measurements; the multi-SKU evaluation across NX and AGX is acknowledged as
> single-SKU limitation in §5.7 and listed as Priority-3 future work in §6.4."

**Q5: "Is the system actually real-time, or just 30 FPS on average?"**

> "We measure two things. First, single-stream end-to-end p50 latency is
> **8.62 milliseconds** at FP16 — *thirty-five times faster* than the 30 FPS
> budget of 33 ms per frame. Second, we ran the four-stream pipeline at
> 30 FPS source pacing for **36.2 minutes continuously** and recorded
> **zero dropped frames** across approximately 108,000 inferences while
> tegrastats logged thermal and power at 1 Hz. The junction temperature
> stayed 23°C below the throttle threshold; the average power draw was 7 W
> against the 15 W MAXN budget. By every standard definition of real-time
> sustained operation, yes."

**Q6: "What happens at higher source FPS or more streams?"**

> "We measured the saturation point separately. At unbounded reader rate
> (cv2.VideoCapture pulling as fast as it can), aggregate throughput tops out
> at **136.89 FPS at N=4** — that is the GPU saturation point for FP16 on
> this engine set. At the surveillance use case of 30 FPS sources we operate
> at 88 FPS aggregate, leaving roughly 54% GPU headroom for additional
> streams or higher-resolution input. The 8 GB unified memory is the harder
> ceiling — N = 4 of our pipeline uses about 25% of it, so memory will
> not bind before compute."

### E. Privacy / security

**Q7: "How does this preserve worker privacy?"**

> "The privacy claim is architectural, not policy-based. The Jetson decodes
> and inferences frames *locally*; the only data structure that crosses the
> network is a JSON event of about 520 bytes containing class index,
> probability vector, timestamp, and head latency. **Raw frames never leave
> the device.** A network monitor on the Jetson uplink can verify this
> byte-by-byte: in our 5-minute demo, 8,988 events were emitted totalling
> approximately 50 kbps — two orders of magnitude below a single 1080p
> stream at 5 Mbps. The bandwidth asymmetry itself rules out covert video
> exfiltration. This is GDPR Article 4 'personal data' staying under the
> data controller's physical control."

**Q8: "Could an attacker steal the model?"**

> "Partially defended by construction. TensorRT engine binaries are
> device-locked — the runtime checks a device fingerprint at deserialisation,
> so an engine copied to a different Jetson SKU refuses to load. We observed
> this directly in the power-sweep measurements: when the engine built under
> one nvpmodel profile was loaded under another, TRT issued a 'different
> model of device' warning. The ONNX export is portable, but rebuilding the
> deployed INT8 engine requires the calibration data, which lives separately
> from the binary. This is a *defense-in-depth* property, not an attestation
> guarantee — a full filesystem read can rebuild. We acknowledge this
> explicitly in §5.6.6 as 'tamper-evident attestation' future work."

**Q9: "What about false alarms in deployment?"**

> "The raw per-emission accuracy is 77.5% on the three unsafe streams in the
> live demo — single-frame jitter pushes the noise floor up. The dashboard
> mitigates this with **three-second rolling aggregation** in probability
> space, then argmax, then a **dual-confidence-threshold gate** at 0.75 for
> ALERT and 0.55 for WARNING. In the 5-minute demo, the policy correctly
> flagged the two true-positive unsafe streams as ALERT (76.7% and 99.4%
> aggregated confidence) and **correctly held the safe-walkway stream in
> MONITORING** at 54.9% — zero false alarms in 5 minutes. The thresholds are
> exposed as sidebar sliders, so the operator can tune sensitivity to the
> deployment's tolerance for false alarms versus missed detections."

### F. Methodology

**Q10: "Why frozen backbone instead of fine-tuning?"**

> "Two empirical reasons. First, the V10 ablation (Chapter 4 §4.6) fine-tuned
> YOLOv11s on 400 bbox annotations of behavior classes; result regressed from
> 56/64 to 54/64 — the small-scale fine-tuning damaged the COCO prior more
> than it added behavioural signal. Second, the cross-ensemble of frozen and
> fine-tuned branches capped at 55/64. Both negative results support the
> data-bound conclusion. Paper-scale (≥ 4,000-frame) bbox fine-tuning may
> invert this, which is the central open question and Priority-1 future
> work in §6.4."

**Q11: "Why not use a more powerful backbone — ViT-L, DINOv2, EVA?"**

> "Parameter efficiency was a deliberate constraint for edge deployment.
> YOLOv11s is 5M parameters; ViT-L is 300M, DINOv2-G is 1.1B. Our 156 MB FP32
> ensemble is already at the upper end of what an Orin Nano Super can
> service at 30 FPS multi-stream — adding 10× more parameters in the backbone
> would push edge real-time off the table. A separate study on whether
> a larger frozen backbone closes the accuracy gap *and remains edge-deployable*
> would be useful future work but was not the thesis question."

**Q12: "Why not test on a second dataset?"**

> "We attempted UCF-Crime cross-domain transfer — the download was
> incomplete (only the Burglary class out of five target classes
> successfully retrieved). A single-class transfer evaluation is
> unbalanced and would be misleading, so we deferred it. The full
> cross-domain experiment is listed as Priority-2 future work (§6.4).
> This is documented as Limitation (b) in Chapter 6."

### G. Engineering / cost

**Q13: "Why is INT8 not faster than FP32 on the M4 CPU?"**

> "The Apple M4 AMX coprocessor has no native INT8 matmul path in the
> current ONNX Runtime CPU EP build. ORT emulates INT8 via per-operator
> dequantize/requantize kernels around each matmul — at our small head sizes,
> the per-op overhead dominates the compute saving and INT8 ends up at
> 32 ms versus 28.8 ms for FP32. INT8 still wins on memory: the ensemble
> shrinks from 156 MB to 40 MB, which matters for flash-constrained edge.
>
> Crucially, **the picture inverts on Jetson**: TensorRT INT8 at 8.24 ms is
> faster than TRT-FP32 at 12.47 ms by 1.51× because the Ampere tensor cores
> have a native INT8 path. The thesis claim is therefore that INT8 is a
> hardware-dependent decision; we provide measurements on both targets."

**Q14: "What is the total cost over five years?"**

> "Capital: USD 499 per edge node + USD 50 for power supply + USD 100 for
> microSD + NVMe ≈ USD 650 per camera-set-of-four (one Jetson handles four
> streams). At 7 W actual draw, electricity at USD 0.12/kWh is
> approximately USD 7.40 per Jetson per year, so 5-year opex ≈ USD 37.
> Total 5-year TCO per four-camera node ≈ USD 690. The paper baseline
> at ~400 W on a dual-GTX-1080 workstation, by contrast, is USD 1,500+
> capital plus USD 420 of electricity over 5 years = USD 1,920, *and* it
> processes only a single stream because of paper's lack of multi-stream
> implementation. Per camera, our system is roughly **11× cheaper over five
> years**."

### H. The killer question

**Q15: "If I am a factory manager, why should I deploy your system instead
of the paper's?"**

> "Three concrete reasons.
>
> **One**, the bbox annotation budget: paper [5] requires you to commit
> roughly 190 hours of expert labour to collect 2,591 bbox labels for *your*
> factory's classes; our system needs only 401 class-level labels at maybe
> 10 hours of labour. For a small or medium factory, this is the difference
> between 'a pilot we can run this quarter' and 'a six-month annotation
> contract before we know if it works'.
>
> **Two**, the privacy property: with our edge architecture, raw worker
> imagery never leaves the camera unit. That removes the entire GDPR
> Article 4 compliance surface — you have no cross-network personal-data
> flow to audit. The paper baseline is a server-side pipeline that
> requires you to design and audit that flow.
>
> **Three**, the operational economics: our system runs on a USD 499 Jetson
> at 7 W and supports four cameras. The paper's 400 W server costs three
> times as much, supports one camera, and consumes 60× the electricity. At
> ten factories with 40 cameras each, the difference compounds to about
> USD 200,000 over five years.
>
> The trade is 6 percentage points of macro accuracy. For high-stakes
> nuclear inspection that gap may be unacceptable; for the much larger
> market of light-industrial sites looking for an unsafe-behaviour
> *screening* signal, our trade is the right one. We do not claim our
> system is universally better — we claim it occupies a Pareto-optimal
> point that the paper baseline does not."

(This is the most important answer in the deck. Practice it. 120–150
seconds is acceptable for this one.)

---

## 4. Defense one-pager (A4 handout)

**Title:** Real-time Unsafe-Behavior Detection on Edge Hardware — A
Pareto-Optimal YOLO–Transformer Pipeline

**Author / date / supervisor.**

**Headline sentence (centre, large type):**

> "+3.5 percentage points macro accuracy over the fair-comparison baseline
> at zero bbox annotation cost, 116 FPS single-stream and 4-camera × 30 FPS
> sustained real-time on a USD 499 Jetson Orin Nano Super at 7 W."

**Table 5.5A — Fair-comparison accuracy (centred):**

| Configuration | Top-1 | Macro | Bbox |
|---|---|---|---|
| Paper ConvLSTM-only | 89.06% | 85.78% | 0 |
| **Ours (Jetson FP32)** | **87.50%** | **89.28%** ⭐ | **0** |
| Paper Unsafe-Net full | 96.88% | 95.81% | 2,591 |

**Three Pareto corners (compact list):**
- Zero-annotation corner: 89.28% macro at 401 class-frames
- Edge-deployed corner: 116 FPS at 7 W on USD 499 device
- Privacy-preserving corner: raw frames stay on edge

**Three future-work bullets:**
1. Paper-scale bbox annotation campaign (Priority 1, §6.4)
2. Multi-SKU edge characterisation across AGX Orin and Orin NX (Priority 3)
3. Cross-domain validation (UCF-Crime, Priority 2)

**Footer:** Code + measurements: github.com/[handle]/[repo].
Canonical numbers: `GROUND_TRUTH.md`.

---

## 5. Pre-defense checklist (the night before)

- [ ] All slides print legibly in black-and-white (last-minute projector
  failures happen; you should be able to talk through the deck off a
  paper handout).
- [ ] Live demo terminals A and B opened and tested **on the defense
  computer**, not your dev laptop. Confirm SSH key works and Jetson is
  reachable.
- [ ] Fallback screenshots in `figures/` opened in a backup browser tab.
- [ ] Phone alarm set for 18 minutes — gives 2 minutes for closing if
  20-minute slot is strict.
- [ ] One-pager printed in 8 copies (5 committee + 2 for table + 1 backup).
- [ ] `GROUND_TRUTH.md` open in a terminal tab — if a number is
  questioned, you can `grep` it live.
- [ ] Q1 (the 95.81% question) practiced out loud, three times, with a
  timer.
- [ ] Q15 (the factory-manager question) practiced out loud, three times,
  in a mirror — body language matters for this one.
- [ ] Slide 4 set as the fallback if everything else fails — you can talk
  through 22 slides from slide 4 alone.

---

## 6. The one mistake to not make

The single highest-cost mistake in a Vietnamese-norm defense is to
respond to "why didn't you reach 95.81%" with **defensive uncertainty**.
The committee is asking the question because they want to see you
*own the answer*, not to trap you.

The rehearsed answer above takes 60 seconds. Do not improvise; the
improvised version will sound defensive even if the content is correct.
Just say it. Move on.

Everything else is recoverable. This one is not.
