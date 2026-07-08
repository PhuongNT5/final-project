# Thesis Improvement Strategy — Unsafe-Behavior Detection (YOLO + Transformer)

## Context

Mock defense committee (see `plans/260616-thesis-defense-strategy/` + prior review) flagged 5 weaknesses:

| # | Weakness | Evidence |
|---|---|---|
| W1 | No new architecture; RT-DETR dropped; temporal modeling = null result | README §3.2.0; ablation `retrain.json` (appear-only 53.67 ≈ both 53.33) |
| W2 | Accuracy below target (>95.8%); only parity with *weakest* baseline; top-1 below it blind | blind 53/64 (85.4% macro) vs paper no-bbox 57/64 (85.78%) vs paper full 62/64 (95.81%) |
| W3 | Statistics too weak — N=64, CI [79.7, 95.3], no significant difference vs baseline | `bootstrap.json`, `significance.json` |
| W4 | Generalization unproven — 1 dataset, 1 factory, 39 days, no cross-dataset | data audit |
| W5 | Scope/title mismatch — delivered an edge system, not the proposed architecture | Chapter 5 vs PROPOSAL.txt |

Goal: turn a thesis that *underdelivered vs its proposal* into one that is *defensible and Pareto-valuable*, with the cheapest possible accuracy win on top. Deep research (2 reports in this folder) + codebase audit confirm most fixes augment **existing assets**, not from-scratch work.

---

## Recommended strategy: HYBRID (1 cheap accuracy win + rigor + reframe)

Do NOT chase 95.8% (data-bound, needs ~2,262 bbox ≈ 190 hrs). Instead:
- **One** high-ROI accuracy move that attacks the root cause (zone confusion) cheaply → **virtual fencing**.
- **All** the low-effort rigor + framing fixes — they convert "weak accuracy" into "rigorous, Pareto-superior, safety-aware edge system."

Why this over the alternatives:
- *Maximize-accuracy* (auto-annotate + VideoMAE + RT-DETR): weeks of work, ceiling ~91–93% macro, still below paper, high time risk.
- *Consolidate-deployment only*: safe but leaves the cheap +6–12% accuracy win on the table and keeps top-1 below baseline.
- *Hybrid*: best ROI — see workstreams.

---

## Workstreams (priority order)

### WS1 — Reframe title + 3 objectives  ·  effort: XS  ·  do first
Realign the thesis to what was delivered (deployment-led, zero-bbox, edge). Keep accuracy honest.
- New framing: "design + rigorous characterization of a privacy-by-construction, multi-stream edge unsafe-behavior detector that matches the no-bbox baseline at N× lower energy/latency."
- MT1 → engineering composition (drop "novel architecture" / "RT-DETR" claims, or justify their removal as a finding).
- MT2 → "match no-bbox baseline at zero bbox cost" + report negative temporal result.
- MT3 → unchanged (edge deployment — already met/exceeded).
- Files: `PROPOSAL.txt`, thesis Ch.1/Ch.6, `README.md` headline.

### WS2 — Rigor: day-grouped k-fold CV + proper significance tests  ·  effort: M  ·  highest defensibility
Single biggest fix for W3. Replace the single N=64 split with **5-fold stratified day-grouped CV over all 691 clips** (effective N ~10×, CI ~3× narrower).
- Report mean ± std accuracy and per-class recall across folds.
- McNemar **exact** test (b+c<25) for fixed-split comparisons; Nadeau–Bengio corrected resampled t-test for k-fold comparisons.
- If non-significant vs baseline, say so — then the k-fold gives the powered answer.
- **Dependency/risk:** need per-clip *day/session* labels to group folds. If filenames lack day metadata → fall back to clip-level stratified k-fold + state the temporal-leakage caveat. **Verify first.**
- Reuse: `run_blind_eval.py` (`stratified_fit_val`, `fit_select_model`, `probs_on`), cached features `sweep_results/pairwise/features/all8_{train,test}.pt`, `run_ablation_significance.py`, `run_ablation_bootstrap.py`.

### WS3 — Safety-critical metrics (replace accuracy as headline)  ·  effort: S
Fixes the "accuracy is the wrong metric" angle and pre-empts a domain committee.
- Per-class **false-negative rate / miss rate** (missed hazards) as primary metric.
- **F₂-score** (recall-weighted), AUC-PR curves, full confusion matrix.
- **Alarm latency** end-to-end per class (instrument the deployed pipeline timing hooks — frames-to-fire + inference + dispatch).
- Reuse: existing confusion matrices, `deployment/artifacts/per-clip-logits-test64.csv`, `deployment/scripts/jetson-multi-stream-pipeline.py` for latency hooks.

### WS4 — Cheap accuracy win: virtual fencing geometric prior  ·  effort: S–M  ·  highest accuracy ROI
Attacks root cause of the ~54/64 ceiling: WalkwayViolation ↔ UnauthorizedIntervention confusion (both = "person in restricted area", needs zone knowledge). Instead of learning zones from 2,262 bbox, draw **one polygon per camera view** (one-time, ~1–2 hrs) and add geometric features.
- Feature vector += [person_in_zone (bool), person_count_in_zone (int)] from person detections ∩ zone polygon.
- Person boxes: reuse fine-tuned YOLO (`runs/detect/.../yolo_finetuned/weights/best.pt`) or YOLO-World (already integrated, `run_yoloworld_push.py`).
- Expected: +6–12% macro on the two confused classes (research estimate).
- **Dependency/risk:** how many distinct camera viewpoints in Unsafe-Net? Few fixed cameras → trivial; every clip a new view → harder. **Verify first.**
- New: a `run_zone_prior.py` that intersects person centroids with per-view polygons, concatenates to stats features, retrains the direct-4 head under blind protocol.

### WS5 — Cross-dataset generalization test  ·  effort: S–M
Fixes W4 with public data. Even one transfer result strengthens the claim.
- **iSafetyBench** (1,100 industrial-safety *video* clips, 2025) — closest match; zero-shot or light fine-tune.
- **SH17** (8k images, 17 PPE classes) or Roboflow Construction-Safety — zero-shot test of the YOLO/PPE detector head; report mAP under "domain transfer".
- Reuse: `run_ucf_crime_zeroshot.py` as a template (zero-shot harness already exists).

### WS6 — (optional, if time) VideoMAE ViT-B temporal replacement  ·  effort: M
Replace the dead derivative streams with a proven data-efficient video model (VideoMAE designed for small data; 566 clips < HMDB-51 scale). Freeze early layers, fine-tune last 3–4 blocks + strong video augmentation (SV-Mix, temporal crop). Expected +3–6% on motion classes; will NOT fix zone confusion. Pursue only after WS1–WS5.

### WS7 — (optional, only if WS4 done) RT-DETRv3 backbone swap  ·  effort: M
Answers the committee's "where did RT-DETR go" question concretely. NMS-free helps crowded scenes (+1–3% AP), but orthogonal to the data-bound gap — marginal unless spatial supervision (WS4) is already in. Frame as "we evaluated the proposed RT-DETR path; here is the measured result."

---

## Weakness → fix map

| Weakness | Primary fix | Secondary |
|---|---|---|
| W1 architecture/temporal null | WS1 reframe + report negative result with test | WS7 RT-DETR evaluated |
| W2 accuracy | WS4 virtual fencing (+6–12% confused classes) | WS6 VideoMAE |
| W3 statistics | WS2 day-grouped k-fold + McNemar/N–B | WS3 CIs per fold |
| W4 generalization | WS5 cross-dataset (iSafetyBench/SH17) | WS2 LODO temporal holdout |
| W5 scope/title | WS1 reframe + Pareto framing (acc vs W / ms) | WS3 safety framing |

## Effort ladder (pick by time budget)
- **< 1 month:** WS1 + WS2 + WS3 (reframe + rigor + safety metrics) — makes thesis defensible with zero new accuracy work.
- **1–3 months:** + WS4 (virtual fencing) + WS5 (cross-dataset). The recommended target.
- **> 3 months:** + WS6 / WS7.

---

## Reusable assets (codebase audit)
- Blind-eval harness: `run_blind_eval.py` (modular utilities).
- Cached features: frozen `sweep_results/pairwise/features/`, bbox `sweep_results/finetune_bbox/features/`, frame-gate `sweep_results/framegate_conf.pt`.
- Annotations: `annotation_guide/` ~1,004 nonempty bbox (more than the "400" the thesis cites) + κ subset.
- Trained detectors: fine-tuned YOLO `best.pt`; YOLO-World pipeline.
- Significance/bootstrap scripts already written.
- Real Jetson artifacts (FP16 8.62 ms/116 FPS, 4-stream 88 FPS zero-drop, 36.2-min soak 7.0 W) — Pareto/efficiency tables ready to build from `deployment/artifacts/*.json`.

## Risks
- WS2/WS4 both depend on metadata that may not exist (day labels; distinct camera views) — verify before committing.
- Virtual fencing assumes person detection is reliable on the dataset (frozen COCO gave only 27% person detection in V6-det; YOLO-World ~99% — use YOLO-World).
- k-fold re-runs invalidate the single-split headline numbers in the current draft — budget time to update all tables/figures.

## Unresolved questions
1. Time remaining until defense? (gates the effort ladder)
2. Still have access to: raw Unsafe-Net videos? the Jetson? a training GPU?
3. Does the data carry per-clip day/session + camera-view metadata (needed for WS2 grouping and WS4 polygons)?
4. Will advisor accept a title/objective change (WS1), or must the original framing stay?
5. Target outcome: "defensible thesis" (WS1–3 enough) or "approach the paper's accuracy" (needs WS4+WS6)?

---

## WS2 progress log (2026-06-28)

**Metadata verification (gates WS2/WS4) — DONE.**
- No day/session/camera metadata anywhere: filenames are sequential `{cls}_tr/te{N}`; mp4s have NO `creation_time` tag (ffprobe); `frame_manifest.csv` has no day column; `dataset-metadata.json` belongs to UCF-Crime, not Unsafe-Net.
- ⇒ **True day-grouped CV / leave-one-day-out is impossible.** WS2 fallback adopted: clip-level stratified k-fold over pooled 691 clips (516 four-class) **+ a scene-cluster GroupKFold proxy** (KMeans on time-averaged features) to substitute for the missing session grouping.
- ⇒ **WS4 impact:** virtual fencing needs per-camera polygons but there are no camera labels → must add a viewpoint-clustering pre-step; original "1–2 hr" estimate is optimistic. Re-scope before starting WS4.

**Implementation — DONE.** `run_kfold_eval.py` (reuses `run_pairwise_contrastive` + `run_blind_eval`): stratified + scene-group k-fold, linear-probe baseline, Nadeau–Bengio corrected t-test, 5k bootstrap CI, per-class recall + **FNR** (feeds WS3). Compiles, smoke + mini-verify pass. Full run (k=5, seeds 42/123/314, 24 clusters, 300 ep) running in background → `reports/kfold-eval.json`.

**FINAL findings (full run k=5/3-seed/24-cluster/300-ep, 302 min — see `reports/ws2-kfold-results.md`):**
- **Rigor win:** CI width 15.6pp (single split [79.7,95.3]) → **~4.4pp** (k-fold [73.1,77.5]).
- **Cross-validated accuracy of core single-head method:** random **75.3±1.2% top-1 / 77.8±1.0% macro**; scene-robust **71.9±1.6% / 71.8±1.0%**. Below the canonical-split headline (82.8–87.5%) because single split + ensembling/selection inflated it; k-fold is the representative generalisation number. Report BOTH.
1. **Transformer head significantly WORSE than a linear probe** on random k-fold (Δ=−4.65pp, N-B p=0.007); tie scene-robust (Δ=+1.7pp, p=0.47). A logistic regression on mean-pooled features ≥ the Transformer → quantifies temporal-null; strong support for reframing/dropping "Transformer" from headline (W1).
2. **Scene-leakage gap = only 3.4pp** (75.3→71.9) — *the 18pp from mini-verify was k=2/8-cluster noise.* Modest scene dependence, not catastrophic memorisation; scene-robust 71.9% is the honest deployable estimate (W4).
3. **Safety FNR:** UnauthorInt is the critical miss — 44.8% (random) / **68.2% (scene-robust)**; WalkwayViol ~30%; Panel/Forklift ~0%. All risk in the 2 zone classes → precise WS4 motivation (W3).
- **Caveat:** k-fold tests the *single* head, not the deployed ensemble+cascade+gate stack — 56/64 deployed number stays valid for the canonical split.

**Decision needed (user):** how to present the lower-but-honest k-fold numbers — recommend **report both** random + scene-robust as a complementary rigorous evaluation alongside the canonical deployed checkpoint, leading with F1 (negative result) + F2 (safety FNR).

**WS2 report delivered:** `report/chapter-04-section-kfold-cross-validation.md` (§4.8, drop-in, EN).

## WS4 progress log (2026-06-29)

**Feasibility — STRONG.** Viewpoint clustering of the 516 clips: silhouette peaks ≈0.45 at k=6 → only ~6–8 distinct camera scenes → hand-drawing zone polygons is practical. **8/12 clusters contain BOTH Walk & Unauth** → the two confused classes share camera views and differ by *person position* — exactly the virtual-fencing case. Representative frames confirm the factory floor has the **walkway boundary painted (green lines)**.

**Automatable proxy — RUNNING.** `run_kfold_eval.py` extended (DRY, `--feature spatial`, `--schemes`): spatial feature = stats(1536) + per-cell energy map(16) + activation centroid x,y + spread → 1555-d, encodes WHERE activity sits. Full run (both schemes, 3 seeds) in background → `reports/kfold-eval-spatial.json`. Compares against the WS2 baseline to test whether location info alone recovers Walk/Unauth.

**Manual zone step — PREPARED (needs user).** `tools/extract_viewpoint_frames.py` → `annotation_guide/zones/view_{0..7}.png` + `viewpoints.json`; instructions in `annotation_guide/zones/README-draw-zones.md`. Minimum useful effort = 2 polygons (views 5,1). Follow-up `tools/apply_zone_prior.py` (person foot-point ∈ polygon → zone features → re-k-fold) to be built AFTER (a) spatial result seen and (b) polygons provided.

**Decision gate:** if `--feature spatial` already recovers Walk/Unauth FNR → manual polygons optional; if not → explicit zones are the stronger lever.

**Spatial-feature result (2026-06-29) — NEGATIVE.** stats→spatial: stratified +0.1pp (75.3→75.4), scene-robust **−3.8pp (71.9→68.1)**, Unauth FNR barely moved (44.8→43.5 / 68.2→66.0). Crude backbone energy/centroid responds to all objects (machines), tracks scene layout → hurts on novel scenes. **Automated cheap proxy fails.** TF-vs-linear gap persists (Transformer still ≤ linear). → Confirms from a 3rd angle the gap is zone-supervision-bound. Remaining levers, in order: (L2) person-detection position features (no manual zones, ~6h compute, untested), (L3) full virtual fence = manual polygons + person foot-point ∈ zone (strongest, needs user to draw ≥2 polygons). **User decision needed: pursue L3 (accuracy win, more work + deadline risk) OR stop and lean into deployment-led framing + the now-triple negative result (low-risk, defensible).**
