# Revision Checklist — per committee point, with file:line targets

Legend: ☐ = todo. Each item = (committee point) → (verified ground truth) → (exact edit).

---

## §A — Narrative / scope front-loading (Hội đồng pt.1)

☐ **A1 — RT-DETR deviation.** Proposal names RT-DETR as the temporal-reasoning core
(`PROPOSAL.txt:27,44-46,104-106,138-141`). Never built; thesis uses frozen YOLOv11s + light
Transformer heads. **Edit:** add a **§1.x "Scope evolution from proposal"** subsection in
`report/chapter-01-introduction.md` with a *proposal-vs-delivered table*:
| Proposal commitment | Delivered | Reason | Novelty impact |
|---|---|---|---|
| RT-DETR temporal encoder | Frozen YOLOv11s + multi-stream Transformer head | RT-DETR is a detector, not a sequence head; frozen-feature study isolates the data-bound gap | Novelty re-cast: empirical study + edge artifact, not new mechanism |
| >95.8% top-1 | Falsified-hypothesis finding (blind ~53–54 / best-case 56 /64) | Frozen COCO features lack zone semantics (cosine 0.83) | Negative result is the contribution |
| 8-class safe/unsafe | 4-unsafe headline; 8-class ~78% → future work | 4-class isolates the confusion pair cleanly | Scope narrowed, declared up front |
State plainly: "the core architectural novelty registered in the proposal (RT-DETR temporal
reasoning) was **not** realized; this thesis re-frames its contribution as an empirical/deployment study."

☐ **A2 — Target deviation framing.** Move the ">95.8% not met → falsified hypothesis + Pareto"
framing from `chapter-06-conclusion.md:67-89` *forward* into Ch1 as the stated research question.
Ch1 must say the quantitative target was not met and why this is a valid scientific finding, so the
panel reads §6.2 as pre-declared, not post-hoc rationalization.

☐ **A3 — 8→4 class narrowing.** Add one explicit sentence in Ch1 abstract + scope: headline metrics
are on the 4-class unsafe subset (64 clips); 8-class joint and binary safe/unsafe are future work
(`chapter-06-conclusion.md:162-168`, ~78%).

---

## §B — Mandatory blind experiment (Phản biện 1a) → see `phase-01-blind-eval-experiment.md`

☐ **B1** Run epoch-selection on a TRAIN-carved validation slice (not the 64 test clips); report
blind test number. Expected ~53–54/64 (consistent with Table 4.16 retrains: 53.0–53.67).
Adopt as defensible headline; keep 56/64 labeled "best-case, test-tuned upper bound."

---

## §C — Statistical honesty (Phản biện 1b,1c,1d)

☐ **C1 — Lucky-seed headline.** `direct4_s314` = 55/64 is a seed-distribution tail; standard-seed
retrains avg 53.0–53.67 (`chapter-04:532-538` Table 4.16). After Phase 1, report headline as
**blind mean ± seed**, with 56/64 as best-case. Update README:14-33, Ch1, Ch4 §4.2.2, Ch6 §6.1.

☐ **C2 — Cascade contribution not proven.** Table 4.15 (`chapter-04:611-631`): ensemble−direct4
P=0.63 (CI touches 0); cascade−direct4 P=0.98 *worse*. **Edit:** add an explicit paragraph in §4.7.9
and §6.1 naming what IS proven: "the demonstrated technical contribution is the *direct4 frozen-feature
head + the data-bound diagnosis*, NOT the pairwise-contrastive cascade, which shows no significant
lift on N=64." Stop presenting the cascade as a validated novelty.

☐ **C3 — Macro-vs-top1 defense.** `chapter-05:546-561`, `README:14-20`. Always present top-1 AND
macro together; never lead with macro alone. Add a one-line justification: macro = mean per-class
recall = the safety-relevant metric (a rare dangerous class missed is costly); cite paper Table 7
also uses macro. Acknowledge top-1 56 < baseline 57 explicitly each time macro +3.5pp is claimed.

---

## §D — Architecture / novelty honesty (Phản biện 2a)

☐ **D1 — "Temporal reasoning" overclaim.** Table 4.16 (`chapter-04:532-538`): appearance-only 53.67
≈ motion-only 53.00 ≈ both 53.33 — temporal/motion stream adds nothing measurable. **Edit:** in Ch3
(method) and Ch4 §4.7.8, state honestly: "on this dataset the temporal (motion-derivative) stream
shows no measurable contribution above frame-level appearance pooling; the pipeline is effectively a
frame-level pooled classifier with an optional temporal path that the data does not reward — itself a
finding about the dataset's temporal information content." Avoid marketing the head as "temporal
reasoning" in the contribution list; rename to "multi-stream Transformer aggregation head."

☐ **D2 — Position edge deployment as the strongest contribution** in Ch1 contributions list and §6.5.

---

## §E — Factual contradictions (Ủy viên + Phản biện 2b) — VERIFIED, zero-compute fixes

☐ **E1 — M4 latency single source of truth.** Ground truth `deployment/artifacts/benchmarks.json`:
end_to_end **p50 26.54 ms, p95 28.6 ms, FPS 37.7** (50 runs). Thesis prose uses an inconsistent
**28.8 / 37.3 / 34.7** pair. NOTE: §5.6.2 speedups (2.13× FP32 = 26.54/12.47; Ch6 3.08× = 26.54/8.62)
were computed from 26.54 and are CORRECT — so 26.54 is the value to keep.
  - Fix `chapter-05:153` Table 5.2 CPU-FP32 row → `26.5 ms | 28.6 ms | 37.7`
  - Fix `chapter-05:158` headline prose, `:635` §5.8, `:349` already 26.54 (OK)
  - Fix `chapter-01:97,174`, `chapter-02:156`, `chapter-06:47`, `README:30`
  - Also reconcile run count: Table 5.2 caption says "100 runs"; artifact = 50 runs. Either re-run
    `deployment/scripts/benchmark-harness.py` (cheap, non-training) to regenerate at 100 runs, OR
    change caption to 50 runs. **Recommended: re-run benchmark to settle definitively.**

☐ **E2 — Training resolution.** `run_resolution_ablation.py:8-9,33,186` confirms pipeline = **224²**
(384² was the *rejected* ablation; no 640×384 anywhere). `chapter-05:593` §5.7 lim.1 wrongly says
"Training used 640×384 image crops." **Edit:** rewrite lim.1 → train AND deploy both 224×224 (no
mismatch); the real limitation is that 224² is a low-resolution operating point (small objects may
degrade), per the §4.7.10 ablation that *defends* 224² on accuracy grounds.

☐ **E3 — UCF-Crime contradiction.** Ground truth `sweep_results/ablations/ucf_crime_zeroshot.json`:
**267 clips, 3 classes** (Burglary 100, Fighting 50, RoadAccidents 117), dataset "partial." Ch4 §4.8.5
(`:850-888`) and Ch1 (`:139`) are CORRECT. Ch6 is STALE: `:99-103` says "blocked download, only
Burglary 25 clips, deferred"; `:145-151` P2 says "re-run the failed download." **Edit Ch6:**
  - §6.3(b): replace with — cross-domain zero-shot *was completed* on 267 clips / 3 classes (§4.8.5);
    limitation is it covers 3 of 5 UCF anomaly classes (RoadAccidents partial; Vandalism, Stealing not
    fetched). 
  - §6.4 P2: reframe — *extend* the existing probe to the remaining classes + complete RoadAccidents,
    not "re-run a failed download."

☐ **E4 — Multi-stream N=1 anomaly.** M4 Table 5.4 (`chapter-05:248`) N=1 = 17.6 FPS vs single-stream
34.7/37.7; Jetson Table 5.5C N=1 = 24.28 vs 116. **Edit:** add explanation in §5.5.2 and §5.6.3: the
async multi-stream harness pays a fixed per-tick scheduling + 5 ms batch-assembly-bucket +
drain-on-shutdown cost that the synchronous single-stream benchmark (§5.4) does not; at N=1 this
overhead halves throughput, and it amortizes as N grows (peak N=2 on M4). This is harness overhead,
not a model regression — state it so the panel doesn't distrust the whole multi-stream table.

---

## §F — Overclaim softening (Phản biện 2c,2d + Ủy viên security)

☐ **F1 — V10 extrapolation.** `chapter-06:127-143` P1 "the *only* known route" / "requires
paper-scale." No 1k/2k-frame runs exist. **Edit:** soften "only known route"→"the most direct route
the evidence points to"; "requires"→"evidence suggests may require." Mark the 400→4000 inversion as
hypothesis, not established.

☐ **F2 — Security framing.** Table 5.5G (`chapter-05:520-538`). T1+T5 (edge privacy-by-construction,
metadata-only uplink) = genuine architectural contribution; KEEP. T2 (TRT device-locking), T3/T4 (SSH)
= standard operational properties, NOT research findings. **Edit:** add a sentence above the table:
"T1 and T5 are architectural contributions of this work; T2–T4 are standard operational hardening
listed for completeness, not claimed as novel." §5.6.6 already admits no attestation/model-signing —
keep and cross-ref.

☐ **F3 — Demo = PoC, not deployed system.** `chapter-05:500-511` (4 video files as cameras),
§5.7 lim.4 already admits no live camera. **Edit:** ensure Ch1, Ch6 §6.1 (`:59-62`), and any slide-facing
prose say "multi-stream proof-of-concept with file proxies for cameras" — not "4-camera surveillance
system." Proposal promised live camera streams (`PROPOSAL.txt:99-100,185-186`); state this gap.

---

## §G — Consistency sweep (last)

☐ **G1** After all numbers settle, grep every headline figure across `report/`, `README.md`,
`PROGRESS.md` and confirm each traces to a committed artifact. Targets: blind/56-64 accuracy, macro
89.28, M4 26.54/28.6/37.7, Jetson 8.62/116 & 8.24/121, power 7.0W, soak 36.2min, INT8 40MB.
☐ **G2** Verify all cross-section references (§x.y) resolve and point to the right table.
