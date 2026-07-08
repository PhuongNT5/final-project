# Research: Rigor, Generalization, Contribution Framing, Safety Metrics (cited)

## Q1 — Stronger evaluation on small data (N≈64)
- **Day-grouped stratified 5-fold CV over all 691 clips** → effective N ~10×, CI ~3× narrower. Report mean ± std accuracy + per-class recall.
- **Leave-one-day-out (LODO)** = temporal-robustness check (39 days, 1 factory). Analogous to leave-one-scene-out (arXiv:1908.10899). Prevents temporal leakage (overlapping windows across fold boundary inflate results — arXiv:2310.11950).
- **Significance:** McNemar **exact** test when b+c<25 (single split); **Nadeau–Bengio corrected resampled t-test** for k-fold (naive t-test anti-conservative). Report non-significance honestly, then k-fold gives powered answer.
- Sources: k-fold (MLMastery); McNemar (mlxtend; Dietterich 1998); N–B (2003); 5×2cv BCV McNemar arXiv:2304.03990; HAR leakage arXiv:2310.11950, 1908.10899.

## Q2 — Cross-dataset external validation (public)
| Dataset | Size | Modality | Use |
|---|---|---|---|
| **iSafetyBench** (2508.00399, 2025) | 1,100 clips, 18 hazard cats | **video** | closest; zero-shot/fine-tune transfer |
| **SH17** (2407.04590) | 8,099 img / 17 PPE classes | image | zero-shot PPE-head mAP, manufacturing domain |
| Roboflow Construction-Safety | 2,801 img, 10 classes | image | zero-shot PPE/person |
| Pictor-PPE v3 | 1,500 img, 3 classes | image | fast fine-tune transfer |
| SHEL5K / GDUT-HWD | ~3–5k img | image | helmet benchmarks |
| Multispectral Forklift (Sensors 2022) | 19,420 frames | video RGB+thermal | overlaps "Forklift overload" |

Quickest ROI: iSafetyBench (video) for behavior transfer; SH17/Roboflow for PPE-head zero-shot. Reuse `run_ucf_crime_zeroshot.py` as template.

## Q3 — Defensible CS thesis without beating SOTA
Legitimate contribution axes (systems/measurement track):
- **Pareto framing**: accuracy-vs-Watt and accuracy-vs-latency plots → "Pareto-superior" replaces "weak accuracy" (arXiv:2602.17508).
- **Energy/inference** (FPS/W, mWh/inference): Jetson INT8-TRT 0.13–0.22 mWh vs Pi 1.22–5.87 mWh (5–45×). Primary table, not footnote (Nature SR PMID 41896318).
- **Negative results are publishable** (Beyer 2017 arXiv:1705.04402; negative.vision CVPR'17) — temporal null result reported with test = real finding.
- **Privacy-by-construction**: on-device inference, metadata-only uplink → NIST data-minimization / GDPR Art 5(1)(c). Cloud baselines can't match regardless of accuracy.
- **Reproducible artifacts** (ACM AE badges): release code + weights + split defs.
- **Multi-stream zero-drop** on 7–25 W = non-trivial systems result.

Reframe sentence: "We design, deploy, and rigorously characterize a privacy-by-construction multi-stream edge detector; (1) matches strongest baseline at N× lower energy/latency (Pareto-superior); (2) temporal modeling gives no significant lift (negative result, N–B test); (3) cross-dataset transfer establishes generalization bounds; (4) all artifacts released."

## Q4 — Safety-critical metrics (accuracy is wrong headline)
- **Per-class FNR / miss rate** = primary safety metric (missed hazards).
- **F₂-score** (recall 2× precision), **AUC-PR** (imbalanced), full confusion matrix.
- **Alarm latency** end-to-end per class (buffering + inference + dispatch); rarely reported (arXiv:2402.09082); sub-450 ms emerging standard.
- **Alarm-fatigue / precision floor**: IEC 62682 ⇒ report false alarms per hour per stream; PR curve with deployed operating point.
- Differential class severity weighting (forklift-overload/unauthorized > walkway).

## Highest-ROI rigor moves (impact/effort)
1. Day-grouped 5-fold CV over 691 clips (M) — makes every number defensible.
2. Per-class FNR as primary metric (S).
3. McNemar exact + N–B corrected t-test (S).
4. Pareto plots accuracy-vs-W / vs-ms (S).
5. LODO temporal holdout (M).
6. Zero-shot transfer on SH17 / iSafetyBench (S).
7. Alarm-latency per class (S).
8. Report temporal negative result with test (XS).
9. Release artifacts + license (S).
10. F₂ + AUC-PR curves (S).

## Sources
SH17 2407.04590; iSafetyBench 2508.00399; Pictor-PPE (github ciber-lab); SHEL5K (MDPI Sensors 2022); Roboflow Construction-Safety; negative results 1705.04402; McNemar (mlxtend); N–B corrected t-test (MLMastery); k-fold leakage 2310.11950 / 1908.10899; Pareto AI 2602.17508; edge energy PMID 41896318; detection latency 2402.09082; IEC 62682; F-beta (MLMastery).
