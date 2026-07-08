# Defense Rebuttal Brief — anticipated questions + answers

Crisp, defensible answers for each committee point. Practice these. Concision > grammar.

---

## Hội đồng — proposal↔product deviation

**Q: RT-DETR was the registered core novelty. Where did it go?**
A: RT-DETR is an end-to-end *detector*, not a temporal-sequence head. Early experiments confirmed the
research question is not "which detector" but "do frozen object-detection features carry safety-rule
context." We froze YOLOv11s as a spatial encoder and put a lightweight Transformer head on top — this
*isolates* the data-bound gap, which RT-DETR (also COCO-pretrained) would not have escaped. The novelty
is re-cast, transparently in §1.x: empirical study + measured edge artifact, not a new mechanism. We do
not claim a RT-DETR contribution and have removed it from the contribution list.

**Q: You promised >95.8%, delivered far less. Is §6.2 post-hoc rationalization?**
A: No — and we now pre-declare it in Ch1. The target was a hypothesis ("a frozen/light-finetune
pipeline can match the bbox-supervised paper"). Ten independent architecture tracks converge on the same
ceiling, and a direct test (V10, 400 bbox frames) *regressed*. The hypothesis is falsified, with a
quantitative cause (COCO features place safe/unsafe pairs at cosine 0.83). A falsified hypothesis with an
isolated mechanism is a valid scientific result. We do not hide that we aimed higher.

**Q: Headline is on 4 classes, proposal was 8.**
A: Stated up front now (Ch1 scope). The 4-unsafe subset isolates the WalkwayViol↔UnauthorInt confusion
pair that drives the gap. 8-class joint (~78%) and binary safe/unsafe are scoped as future work, not
silently dropped.

---

## Phản biện 1 — methodology & statistics (heaviest)

**Q: 56/64 is contaminated by epoch-selection on the test set. What's the clean number?**
A: Correct, and we measured it under a blind held-out protocol (epoch, config, threshold AND seed all
selected on a train-carved validation slice; 64-clip test touched once). The **defensible headline is
the best zero-bbox config — hybrid (frozen ⊕ detection-finetuned) + frame-gate (§4.6.2): blind 53/64
(82.8 % top-1, 85.4 % macro)** — macro parity with the no-bbox baseline (85.78 %) *even blind*. The
frozen single-head pipeline blind is ≈ 52/64 (§4.7.11); 56/64 is the test-selected best-case (deployed
checkpoint). Blind macro parity (85.4 % ≈ 85.78 %) SURVIVES; only the test-selected "+3.5 pp" is
withdrawn. The number is stable across protocols (val_frac 0.20 and 0.15 both land ~52–53), not a
lucky split. Detection-finetuning alone regresses (bbox 79 % macro); the hybrid *combination* recovers
— a small but real finding.

**Q: The headline head is a lucky seed.**
A: Acknowledged in §4.7.8. `direct4_s314` (55/64) is a seed-tail; standard-seed mean is 53.33.
The blind report is mean±seed, not the tail. 56/64 = best-case, flagged as such everywhere.

**Q: Your own stats say the cascade ensemble is not significant. What IS the proven contribution?**
A: Three things, none of which depend on the cascade: (1) the *measured data-bound diagnosis* — frozen
COCO features lack zone semantics, quantified by cosine geometry and two converging negative results
(V10 regress, cross-ensemble cap); (2) the *V10 negative result* itself (small-scale bbox finetune
hurts the COCO prior); (3) the *measured open-source edge deployment* (Jetson, multi-stream, power
Pareto). The pairwise-contrastive cascade is reported honestly as *not* statistically validated on N=64
— we do not claim it as a proven novelty.

**Q: You chose macro because it flatters you (top-1 56 < baseline 57).**
A: We now always show both. Macro = mean per-class recall, the safety-correct metric: missing a rare
dangerous class is the expensive error, and macro penalizes that where top-1 does not. The paper's own
Table 7 Overall column is macro, so it's the directly comparable cell. We state top-1 parity-minus-one
explicitly alongside the +3.5pp macro every time.

---

## Phản biện 2 — architecture & applicability

**Q: Your data rebut "temporal reasoning" — motion stream adds nothing. Is this still a sequence model?**
A: Honest answer, now in §4.7.8: on this dataset the temporal (motion-derivative) stream shows no
measurable lift over frame-level appearance pooling (53.67 vs 53.00 vs 53.33, within seed noise). So
operationally it is closer to a frame-level pooled classifier with an unrewarded temporal path. That is
itself a finding about Unsafe-Net: its discriminative signal is largely per-frame appearance, not
motion dynamics. We renamed the component "multi-stream aggregation head," dropped "temporal reasoning"
from the contributions.

**Q: Ch4 says UCF-Crime fully done (267 clips); Ch6 says download blocked (25 clips). Which is true?**
A: Fixed — Ch6 was stale. Ground truth (`ucf_crime_zeroshot.json`): zero-shot probe completed on **267
clips, 3 classes** (Burglary 100, Fighting 50, RoadAccidents 117). Ch6 now matches Ch4. Remaining
limitation: 3 of 5 UCF anomaly classes (RoadAccidents partial; Vandalism/Stealing not fetched) — that's
the future-work extension, not a failed download.

**Q: You promised live camera surveillance; demo is 4 video files.**
A: Stated as a proof-of-concept (§5.7 lim.4). The contribution is *measured multi-stream concurrency +
real-time sustainability + privacy-by-construction*, demonstrated with file proxies for cameras. A live
USB/RTSP capture path is engineering future work, not a research claim. We will not present it as a
deployed surveillance system.

**Q: "Paper-scale (4000 frames) is the only route" — you never ran 1k/2k.**
A: Softened. The evidence (V10 at 400 frames regresses) *suggests* a larger annotation budget is needed;
we do not claim 4000 is proven-necessary. It's stated as a hypothesis with a concrete test plan, not a
result.

---

## Ủy viên — edge numbers & consistency

**Q: M4 FP32 latency? Speedups?**
A: **26.8 ms p50 / 28.8 ms p95 / 37.3 FPS (100 runs)** — from the re-run `benchmarks.json` (single
source of truth). Speedups: **2.15× FP32** (26.8/12.47) and **3.11× FP16** (26.8/8.62). All report
chapters (Table 5.2, Ch1, Ch6) use 26.8 consistently; the earlier 28.8/34.7 and 26.54 pairs are
superseded.

**Q: Multi-stream N=1 = 17.6 FPS but single-stream = 37.7. Bug?**
A: Harness overhead, not a model regression (now explained in §5.5.2). The async pipeline pays a fixed
per-tick scheduling + 5 ms batch-bucket + drain-on-shutdown cost the synchronous benchmark doesn't; at
N=1 it halves throughput and amortizes as N grows (M4 peaks at N=2). Same effect on Jetson (N=1 24.3 vs
116 sync). The per-stream FPS and zero-drop columns are unaffected.

**Q: Train resolution — 224² (Ch4) or 640×384 (§5.7)?**
A: 224×224 for both train and deploy — no mismatch. §5.7 lim.1 was mis-stated (corrected). The §4.7.10
ablation defends 224² on accuracy *and* compute grounds (224² 55/64 vs 384² 52/64). Real limitation:
224² is a low-res operating point; finer scenes untested.

**Q: Are your security claims research contributions?**
A: Only T1 (edge inference, raw frames never leave device) and T5 (rolling-window dual-threshold,
measured 100% recall / 0% false-alarm) are architectural contributions. T2–T4 (TRT device-locking, SSH)
are standard operational hardening, listed for completeness — now labeled as such. We explicitly state
(§5.6.6) we do NOT provide attestation or model-signing; that's future work.

---

## Unresolved questions (raise proactively if asked)

- Blind headline FINALIZED: best zero-bbox config (hybrid+gate, §4.6.2) = 53/64 (85.4 % macro, macro
  parity with no-bbox baseline even blind); frozen single-head baseline ≈52/64 (§4.7.11); deployed
  best-case 56/64 (89.28 % macro).
- Whether the M4 benchmark should be re-run at 100 runs (caption) vs adopting the 50-run committed
  artifact — decision pending; either is defensible if caption matches data.
- 8-class / binary safe-unsafe results are preliminary (~78%); not defended as a result.
