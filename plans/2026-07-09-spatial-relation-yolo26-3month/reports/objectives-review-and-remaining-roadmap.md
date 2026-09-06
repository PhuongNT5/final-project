# Objectives Review & Remaining Roadmap

**Date:** 2026-08-16. Full review of all stated goals before MT3 (Jetson). Marks what's done, partial,
and not-yet-done — including the YOLO11 baseline to solidify, and the test-strengthening (k-fold, TTA,
corruption) and statistics still owed.

## 1. Status matrix (all objectives)

| # | Objective | Status | Evidence / gap |
|---|---|---|---|
| A | Multi-entity annotation ~2,262 frames (CVAT, 8 entities) | ✅ done | 2,262 frames, clip-level split, κ QA |
| B | YOLOv11 + Transformer architecture | ✅ done | detector mAP@50 0.969; S1a 54/64 (88.4% macro) |
| C | **YOLO11 baseline — finalise as comparison anchor** | ⚠️ **solidify** | best config S1a (appearance+aux, 7-seed ens) exists; needs a *canonical, frozen, documented* run + saved checkpoint as THE baseline for YOLO26/paper comparison |
| D | YOLO26 architecture experiment | ⏳ MT3 (deferred) | backbone-swap + edge; needs Colab/Jetson |
| **3. Test strengthening** | | | |
| E | K-fold / rotation | ⚠️ **partial** | leave-camera-out done (camera-group CV); **standard stratified random k-fold NOT done** for the relation model |
| F | TTA (flip, temporal-crop, multi-scale) | ❌ not done | needs a TTA wrapper at token-build/inference |
| G | Corruption robustness (noise, blur, compression) | ❌ not done | needs a corrupted-test build (imagecorruptions/albumentations) |
| **4. Evaluation** | | | |
| H | Macro 95.81% replicate / no-bbox 89.06% beat | ✅ measured | 54/64, macro 88.4% > 85.78%; top-1 54<57 same-dist (scene-robust story) |
| I | Action time/clip 0.14 s | ✅ measured | Transformer 0.9 ms/clip; full pipeline ~200 ms/clip |
| J | Reduced avg duration 1.87 s | ✅ measured | kept_frac 0.99 (no reduction — clips already single-action) |
| K | Multi-lens (top-1, macro-F1, bal-acc, per-class P/R, κ, MCC, confusion) | ✅ done | `run_full_eval.py` → full_eval.json + figures |
| L | Safety (per-class FNR, cost-sensitive) | ✅ done | FNR per class; cost(miss×10)=100 |
| M | Operating point (ROC-AUC, PR-AUC, threshold sweep) | ⚠️ partial | ROC-AUC 0.953, PR-AUC 0.874 done; **explicit threshold sweep → safety-recall operating point NOT done** |
| N | Calibration (ECE + reliability) | ✅ done | ECE 0.104 + reliability.png |
| O | Statistics (bootstrap/Wilson/BCa, McNemar, Nadeau–Bengio) | ⚠️ partial | BCa/Wilson/McNemar done; **Nadeau–Bengio NOT done (needs k-fold, item E)** |
| **MT3 (edge)** | | | |
| P | Jetson latency/FPS/multi-stream/size/FLOPs/INT8/energy | ❌ not done | needs YOLO26 export + Jetson device |

## 2. YOLO11 baseline — the comparison anchor (do first)
The current best (**S1a: YOLOv11s detector + appearance+aux Transformer, 7-seed ensemble → 54/64, 88.4 %
macro, LCO 75.6 %**) is the intended baseline for both the paper comparison and the future YOLO26
comparison. Action: run one **canonical, frozen YOLO11 baseline** (fixed seeds, save the ensemble
checkpoint + per-clip logits + full_eval.json), so YOLO26 is a clean drop-in swap measured against it.
This is item C; everything below (E–G, M, O) is then computed *on this baseline* so the numbers are all
consistent and comparable.

## 3. Remaining work — plan (all CPU-doable except D/P)
1. **C — Finalise YOLO11 baseline** (canonical run, saved artifacts). ~30 min.
2. **E — Stratified random k-fold** on the relation model (complements the LCO camera-fold); pool 516
   clips, StratifiedKFold; report mean±std + **N — Nadeau–Bengio** corrected t-test (relation vs a
   linear/scene-only probe). Reuses `eval_utils`. ~1 h.
3. **F — TTA**: horizontal flip + temporal-crop (different frame offsets) + multi-scale token sets,
   average softmax; report Δ on the 64-clip test (expect +1–3 clips or neutral). ~1–2 h.
4. **G — Corruption robustness**: rebuild the token set under gaussian-noise / blur / JPEG-H.264
   compression / brightness at severities {1,3,5}; report accuracy-drop curves (a robustness figure the
   paper lacks). ~2–3 h (re-runs detector on corrupted frames).
5. **M — Threshold sweep**: unsafe-vs-safe operating curve; pick the operating point at recall ≥ 0.95 and
   report its false-alarm cost (safety-first deployment point). ~30 min.
6. **D + P — MT3 YOLO26 + Jetson**: after the above, port detector to YOLO26 (Colab train + TRT-INT8),
   benchmark on Jetson (latency p50/p95/p99, FPS, multi-stream zero-drop, size/params/FLOPs, energy).

## 4. Recommended order
**C → E (+N) → M → F → G → D/P.** C makes every later number comparable; E+N and M close the statistics
and operating-point gaps cheaply; F and G strengthen the test set (your item 3); D/P is the deployment
finale on hardware. Items C, E, F, G, M, O are all **CPU-local** (the relation Transformer is tiny) — no
Colab/MPS-training issues; only D/P needs the GPU/Jetson.

## 5. Notes / honest reminders
- **Scene-robust framing is now primary:** every accuracy number should be reported in BOTH
  same-distribution and leave-camera-out columns (the thesis's core evaluation contribution).
- **Reduced-duration (1.87 s) does not transfer** — our clips are pre-segmented; report honestly rather
  than force a frame-gate that we showed gives no gain.
- **k-fold vs LCO**: both are "rotation" — LCO (camera-group) is the scientifically important one; random
  k-fold is the conventional one the paper-style comparison expects. Report both.

## Unresolved questions
1. For C, is the YOLO11 baseline "best config" fixed as S1a (appearance+aux, no class weights, 7-seed ens)? (recommended.)
2. Corruption suite scope — match a standard (ImageNet-C style 5 types × 3 severities) or a
   surveillance-specific set (H.264 compression, low-light, motion-blur)? (recommend surveillance-specific.)
3. Do we have Jetson access now, or keep D/P as documented future work with the earlier pipeline's edge artifacts as the baseline?
