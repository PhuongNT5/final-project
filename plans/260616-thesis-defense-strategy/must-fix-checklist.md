# Must-Fix Checklist — Pre-Submission Corrections

17 verifiable errors a committee member can catch from the thesis's own artifacts. Fix all before printing committee copies. Verified numbers (do NOT change) are at the bottom.

## CRITICAL (headline claims / internal contradictions)
- [ ] **1. N=4/N=2 soak conflation.** Ch6 §6.1 (L21–24), §6.5, Ch1 §1.4, README L69, Ch5 §5.6 intro all say "4 cameras × 30 FPS sustained 36.2 min @ 7.0 W". → The 36.2-min / 7.0 W soak is **N=2** (`soak-jetson-summary.json`, Ch5 Table 5.5F). N=4 zero-drop = separate ~10 s test (`multi-stream-jetson-fp16-realtime.json`). State both separately.
- [ ] **2. ρ=0.21 vs ρ=0.69.** Ch4 §§4.7.8–4.7.9 (L614, L662) cite ρ=0.21 for frozen_d4 vs frozen_cas; Table 4.12 (L460) shows ρ=0.69. → Footnote the distinction (binary error-vector vs probability-vector correlation) or correct to 0.69.
- [ ] **3. 95.81% mislabeled "top-1".** Ch4 L4 "compare against the 95.81% top-1 baseline". → 95.81% is **macro**; top-1 is 96.88% (matches Ch2 L53, all other chapters).
- [ ] **4. Table 5.5B FP16 per-stage.** Backbone 3.95 / Direct4 2.05 / Cascade 5.70 (sum 11.70 ms) > 8.62 ms e2e — impossible. → Use measured 2.76 / 1.41 / 4.44 = 8.61 ms (`benchmarks-jetson-fp16.json`).
- [ ] **5. "33%" energy improvement.** Ch5 §5.6.4 L469. → (45.7−34.2)/45.7 = **25.2%**. (33% = the FPS/W metric, 29.2 vs 21.9.)

## HIGH
- [ ] **6. Live-demo duration contradiction.** Ch5 §5.6.5 has both "772 s / 8,988 events" and "5-minute / 3,826 events" (L516 vs L520, L530); Ch6 L27 echoes the wrong one. → Delete the 5-min paragraph; use **772 s (12.87 min) / 8,988 events** (`live-demo-stream.jsonl`).
- [ ] **7. Bandwidth claims.** Ch5 §5.6.5 (L514, L517) + Ch1 L122 say "~520 bytes/event, ~50 kbps". → Measured **221 bytes/event, 20.6 kbps** (qualitative "far below H.264" still holds).
- [ ] **8. "57/64" dual meaning in Ch6.** L37 = paper no-bbox baseline; L97 = thesis's own withdrawn figure. → Label each: "paper's 57/64 (ConvLSTM-only)" vs "our withdrawn 57/64 (bias-search)".
- [ ] **9. Withdrawn 57/64 macro: 89.1% vs 90.06%.** Ch4 L158/L503 say 89.1%; Table 4.13a L519 says 90.06. → 89.1% = top-1, 90.06% = macro; disambiguate.

## MEDIUM
- [ ] **10. Dead xref §4.2.10.** Ch4 §4.1 L35 → should be **§4.2.3** (Rejected: TTA hflip).
- [ ] **11. Track count 9 vs 10.** Ch1 §1.6 L241 "9 tracks" vs §1.4 L143 + Ch4 L9 "10 tracks". → Use **10** consistently.
- [ ] **12. Compression 3.8× vs 3.9×.** README L39, Ch6 L17, Ch1 L124, PROGRESS L97. → 156.57/40.36 = **3.9×** overall (components 3.8–3.9×).
- [ ] **13. "Typical 53/64" vs "53.3/64".** Ch6 L39 vs L94 / Ch4 Table 4.24 L812. → Use **53.3/64** (seed mean).
- [ ] **14. z-test rounding.** Ch5 L371 "z=−0.26" and L589 "p=0.78" vs canonical **z=−0.27, p=0.79** (Table 4.16b, README, PROGRESS).
- [ ] **15. 400 vs 401 frames.** PROGRESS L69/L113 "401" vs Ch1/Ch4 "400". → Pick one (V10 annotation count).
- [ ] **16. Annotation hours ~190 vs 216.** Ch5 §5.6.7 L596–598: 2,591 imgs × 5 min = 216 h, not 190 h (190 h = train-only 2,262). → Correct the basis.

## LOW
- [ ] **17. Figure 5.6 collision + missing file.** Assigned to both `fps-vs-streams.png` (L241) and `jetson-live-dashboard-demo.png` (L531); the dashboard PNG does not exist on disk. → Renumber dashboard to **Fig 5.7** and regenerate the screenshot. Also: `arch-diagram.mmd` path is `deployment/artifacts/figures/`, not root `figures/` (Ch3 L77); README `run_finetune_e2e.py "(beat baseline)"` is false (E2E = 54/64) — remove.

## Also mandatory pre-submission (structural)
- [ ] Merge per-chapter reference stubs into one unified bibliography; add missing **ref [22] UCF-Crime** (cited Ch4 §4.8.5, unlisted).
- [ ] Produce **Figure 3.2** (multi-stream gating diagram, marked "to be produced" in Ch3 §3.3 L129).

---

## Verified numbers — DO NOT change (artifact-backed)
- 56/64 = 87.5% ensemble top-1; per-class WalkwayViol 27/32, UnauthorInt 8/11, OpenedPanel 13/13, CarryOverload 8/8 — `per-clip-logits-test64.csv`.
- Jetson TRT-FP16 8.62 ms / 116.0 FPS; INT8 8.24 ms / 121.4 FPS; FP32 12.47 ms / 80.2 FPS — `benchmarks-jetson-*.json`. FP16 4-class acc = 55/64.
- M4 CPU-FP32 26.8 / 28.8 ms / 37.3 FPS — `benchmarks.json`.
- Jetson N=4 zero drops, aggFPS 88.06 — `multi-stream-jetson-fp16-realtime.json`.
- Soak N=2: 36.2 min active, 7.03 W mean, 0 throttle, Tj max 56.8 °C, GPU peak 44% — `soak-jetson-summary.json`.
- FP32 ensemble 156.57 MB → INT8 40.36 MB (3.9×) — `benchmarks.md`.
- Live demo: 8,988 events / 772.1 s — `live-demo-stream.jsonl`.
- Blind ≈52/64 ≈81% top-1 (Table 4.24: 51.7±2.0/64); macro ≈84% (83.9%).
- Speedups: 116.0/37.3 = 3.11×; 26.8/12.47 = 2.15×. Power 7.0/15 = 47%.
- Dataset 566 train / 125 test / 691 total — consistent everywhere.
- Paper baselines 96.88% / 95.81% (full-bbox), 89.06% / 85.78% (no-bbox) — consistent.
