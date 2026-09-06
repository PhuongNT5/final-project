# Motion/temporal features for box-token relation-Transformer — research report

Scope: separate WalkwayViol (moving-across-floor) from UnauthorInt (stationary-at-machine) at N~450
clips, LCO-robust. Grounded in actual repo code, not the task-brief's idealized token description.

## 0. Codebase ground-truth (brief re-read, corrects task brief)

- `model_relation_head.py` token layout is **10 dims**: `[class_id, xc,yc,w,h, conf, iou_walkway,
  iou_restricted, foot_in_walkway, norm_dist]`, `n_num=9`. **No appearance-512d, no track-id embedding
  in the committed token** — task brief's "512-d ROI + track-id embed" is either aspirational or a
  different branch. `build_relation_tokens.py` has an optional `--appearance` flag adding a 64-d
  projected MobileNet/YOLO-backbone feature (`APP_DIM=64`), off by default.
- `build_relation_tokens.py` runs its **own** per-frame detector inference (T=16, direct `cv2` seek) and
  does **NOT** track across frames — no track_id, no motion, today.
- `unsafenet/entities.py` (a **separate** Stage-A/B/C pipeline, `unsafenet/streams.py` +
  `unsafenet/temporal.py`) already does greedy-IoU track linking (`link_tracks`, IoU thr 0.25) over
  T=32 uniformly-sampled cached frames, and `streams.py::_motion_vec` already computes a **10-d
  clip-level motion summary** from the longest-lived person track: track count, track persistence
  (frac. of T frames with a live track), net displacement, path length, mean/std/max step, **dwell
  fraction** (frac. steps < 0.004 in normalized xywhn units), bbox-extent x/y. This is unused by
  `RelationTransformer`. This is exactly the reusable asset the task brief points at.
- `RelationTransformer` already has a **per-frame auxiliary head** (`frame_head`, `aux_w` in
  `run_relation_transformer.py`/`run_lco_eval.py`, default 0.5) supervising per-frame behaviour
  logits from masked-mean per-frame token embeddings — precedent for adding a second aux head cheaply.
- `TemporalNet` (the OTHER pipeline, `unsafenet/temporal.py`) already has the exact fusion pattern to
  copy: `clip_proj = LayerNorm→Linear→GELU` on a `(B,d_clip)` vector, added to the pooled embedding
  (`z = z + clip_proj(clip_feat)`). **Steal this pattern verbatim for RelationTransformer.**
- Zone polygons (`annotation_guide/zones/view_*.json`) are pixel-space only, no metric/real-world
  reference points found — true homography calibration is not "free," would need new annotation work.

## 1. Which motion features are discriminative & cross-camera robust

Ranked by (a) discriminative power for moving-vs-stationary, (b) robustness to camera-scale/FOV
differences (LCO), (c) sensitivity to detector/tracker noise:

| Feature | Discrim. power | LCO robustness | Noise sensitivity | Verdict |
|---|---|---|---|---|
| **Dwell fraction** (frac. near-zero-motion steps) | High — near-direct proxy for "standing at machine" | Good IF step-size normalized by scale (see §3); raw-pixel version is scale-biased | Medium (needs a per-frame speed, sensitive to jitter/box-size noise inflating "near-zero" threshold) | Keep, but renormalize threshold (see §3) |
| **Net displacement** (start→end centroid distance) | High — walkway crossing is defined by large net displacement | Medium — pixel-space displacement scales with camera distance/FOV; must scale-normalize | Low (robust to jitter, just 2 endpoints) | Keep + normalize |
| **Path length** (sum of step norms) | Medium-high — correlated with net displacement but also captures back-and-forth machine-tending motion (short path, low net disp.) | Same normalization need as above | Medium (accumulates jitter noise over T steps) | Keep, useful **ratio** = net_disp/path_length (straightness) separates "walking through" (~1.0) from "fidgeting at a panel" (~0) — cheap, camera-scale-invariant since it's a ratio of two length-normalized quantities |
| **Speed mean/std/max** | Medium — max speed differentiates a walkway-crosser (has a fast segment) from a stationary worker (uniformly slow) | Needs same scale-normalization; std is somewhat more scale-invariant differences already but still not free | Medium | Keep mean+max, std lower priority (info overlaps with dwell fraction) |
| **Track persistence / track count** | Low-medium, and **risky**: correlates with per-view detector/tracker quality, not behaviour | **Poor** — this is the exact appearance-cosine-leak failure mode repeated in motion form: a camera with worse detection recall fragments tracks → spurious per-view signal a model can shortcut on under LCO | Flag as a leak risk, do not feed raw; if used, only as a QC/mask signal | Deprioritize / exclude from classifier input |
| **Foot-point trajectory vs bbox-centroid trajectory** | Higher precision than centroid for ground-contact motion (centroid moves with pose/bending, e.g. reaching into a panel looks like motion in centroid but not in foot-point) | Same normalization need | Foot-point noisier at oblique camera angles (foot occluded) | Worth trying if centroid-based dwell underperforms; second-order improvement, not first |
| **Trajectory curvature / turning angle** (loitering-detection lit., zigzag detection) | Low for THIS pair — the paper's classes are motion/no-motion, not "erratic wandering" vs "purposeful walking"; curvature targets a different distinction (loitering vs transit) | N/A | High at T=16 sparse sampling (angle from noisy sparse points is unstable) | Skip — wrong tool for this pair, plus noisy at this T |
| **Optical flow (dense or sparse-Lucas-Kanade patch)** | Not clearly higher than track-delta at this task: OF measures pixel motion in general (camera shake, shadows, forklift moving in background) vs track-delta which is already entity-attributed motion of the SPECIFIC person box | N/A, extra compute+noise | Adds a full-image extra computation for T=16 frames × N clips; no evidence it beats box-centroid deltas for a "translating vs static entity" distinction when a decent detector+box already exists | **Skip.** See §4. |

## 2. Fusion mechanism at N~450

Evidence (arXiv 2304.00379, "Improved Multimodal Fusion for Small Datasets with Auxiliary
Supervision"): high-capacity fusion ops (bilinear, dense cross-attention gates) "are likely to overfit
when applied even to small or low-dimensional datasets" — the paper's own fix is deliberately SIMPLE:
extra supervision / auxiliary losses over fancy fusion. Directly informs the 4 candidate fusion styles:

1. **Extra numeric token dims** (append motion deltas to the existing 9 numeric per-token features,
   consumed by the existing `num_proj: Linear(n_num, d-32)`). Adds ~2×(d-32)=192 params for d=128.
   Near-zero overfitting risk at this param count. Lets self-attention **relate** motion to
   geometry per-token per-frame (exactly what makes this pair separable: "high iou_restricted AND
   near-zero motion" vs "foot outside walkway AND high motion"), which is the mechanism the model
   doc-comment already claims for geometry alone. **This is the highest-leverage fusion point**
   because it lives at the same granularity the attention already operates over — a clip-level
   scalar bolted on afterward cannot be attended against per-frame geometry.
2. **Clip-level motion vector fused via `clip_proj` (copy `TemporalNet` pattern)**: `z = z +
   clip_proj(motion_vec)` before `self.head`. ~10×d + d×d params (d=128) ≈ 17.7k — still <6% of the
   ~298k model, low risk. Cannot be attended against per-token geometry (loses the "WHERE was the
   person motionless" binding) but is trivial to wire (reuses `streams.py` output directly, zero new
   feature-extraction code) — good first experiment / fast sanity check before investing in #1.
3. **Dedicated motion token** (append 1 extra token per frame with a `motion` class-id and the
   per-frame track delta as its numeric features, so it goes through the SAME `cls_embed` +
   `num_proj` + self-attention stack as boxes): elegant, but this dataset has ≤8 real boxes/frame
   (`N_MAX=8`) — 452 clips × T tokens is already a tight attention budget; adding a token/frame is a
   ~12% sequence-length inflation for uncertain gain over option 1. Redundant with option 1 unless
   you want motion decoupled from any specific person box (e.g., when 2 people are present, whose
   motion?). Skip; option 1 already attaches the motion to the person box it belongs to, more
   physically correct.
4. **Learned gating between an appearance stream and a motion stream**: the exact high-capacity
   mechanism arXiv 2304.00379 warns against for small N. Also: this dataset already has one gating
   precedent (the (different) `TransformerClassHead` README pipeline note "learned gating" over
   4 appearance-derivative streams was tested there and — per README's own honest disclosure —
   "temporal streams show no measurable lift on this dataset" (§4.7.8). That is a **direct negative
   result on this exact dataset for stream-gating**, strong prior against reusing that pattern here.
   Skip.
5. **Auxiliary loss** (motion-state regression/classification off the per-frame embeddings, alongside
   the existing `aux_w` behaviour aux head): cheap parameter-wise (reuses `frame_head`-style linear),
   but supervision target must be derived heuristically from `streams.py` motion features (noisy
   pseudo-label) — adds training-signal noise for uncertain gain since option 1 already gives the
   model direct access to the same information as *input*, which is strictly more informative than as
   an auxiliary *target*. Use only as a secondary/ablation add-on after option 1, not a substitute.

**Ranking for N~450: (1) extra per-token numeric dims > (2) clip_proj vector > (5) aux loss on top of
(1) > (3) dedicated token > (4) learned gating (skip, negative prior already exists in-repo).**

## 3. Camera-invariant normalization

- **Do NOT feed raw normalized-pixel (xywhn) displacement across cameras.** Web evidence (MOT/speed-
  estimation lit.): "variations in bounding box size introduce artificial shifts in 2D that greatly
  affect the 3D position in the scene" — i.e. the same real-world walking speed produces a larger
  pixel-space displacement when the person's bbox is bigger (camera closer / more zoomed / more
  frontal). Across 8 fixed-but-different cameras this is a real confound for LCO.
- **Bbox-height (or sqrt(area)) normalization is the cheap, well-precedented fix**: `speed_norm =
  ||Δcentroid_px|| / bbox_height_px` at each step (classic pedestrian-speed-from-monocular-video
  trick — weak-perspective approx: bbox height ∝ 1/depth, so dividing cancels the depth/zoom scale to
  first order). No calibration data needed, reuses boxes already in the token. **Recommended
  default.**
- **True homography (pixel → ground-plane metric)** is the gold standard but this repo's
  `annotation_guide/zones/view_*.json` are pixel-only polygons with no metric reference points found
  — building per-view homography would need NEW calibration annotation (e.g. known walkway width in
  meters + 4 correspondence points per camera). Real payoff (fully removes cross-camera scale
  confound) but nontrivial new work, and with only 8 views the marginal gain over bbox-height
  normalization is unproven at this N. **Deprioritize — stretch goal only if bbox-height norm
  under-delivers on LCO.**
- **Frame-rate/duration normalization**: tokens sample T=16 (relation path) or T=32
  (streams.py path) UNIFORMLY across each clip's full duration regardless of clip length, so the
  real-world Δt between two adjacent sampled tokens varies with clip duration. Convert step
  displacement to "per-second" units using `clip_duration_sec / T` (cheap: `cap.get(CAP_PROP_FPS)` +
  frame count already read in `build_relation_tokens.py::process_clip`). Verify first whether clip
  durations are actually near-constant across this dataset (staged demonstration clips often are) —
  if so this correction is low-priority; if durations vary >20-30%, do it, it's a 3-line fix.
- **Track-quality confound (§1, "track persistence" row)**: normalizing motion does not fix a
  detector/tracker that behaves differently per camera. Sanity-check after adding any motion feature:
  rerun LCO and check per-view whether the NEW feature's distribution correlates with view id more
  than with label — if it does, you've reintroduced the appearance-cosine-leak problem in motion form.

## 4. Cheap optical flow vs detector-track deltas

**Detector-track deltas suffice; skip optical flow.** Reasoning, not just cost:
- This is an **entity-centric** task (know the person box already, high-mAP@50=0.969 detector) — flow
  would recover motion for a generic pixel patch, then you'd still need to attribute it back to the
  person box (i.e., pool flow inside the person's bbox), which is strictly noisier and NOT more
  informative than the box-centroid delta the tracker already gives for free, since:
  - flow inside a person bbox is contaminated by background pixels at box edges, especially at T=16
    sparse temporal sampling where inter-frame appearance changes a lot (flow assumptions — small
    displacement, brightness constancy — degrade badly over large temporal gaps).
  - box-centroid delta between the SAME track-id across frames is already the exact ground-truth
    quantity flow would approximate, obtained for free from a detector you already trust (mAP 0.969).
- Cost: flow (even "lite" — e.g. Farneback or a tiny RAFT variant) adds a full-frame-pair computation
  per token-frame per clip; track-delta is µs, reuses `link_tracks` already written and tested in
  `unsafenet/entities.py`.
- Dense-trajectory-style literature (IDT/MBH, Wang & Schmid) built descriptors ON TOP of flow because
  they had NO detector/box prior — that assumption doesn't hold here; this repo already has a
  strong, purpose-trained detector, which obsoletes the reason flow-based trajectories existed.
- **Action**: extend `build_relation_tokens.py` to call `unsafenet.entities.link_tracks` (already
  written, ~20 lines, generic on `[cls,conf,cx,cy,w,h]` per-frame arrays) on its own T=16×N_MAX
  detections to get track_ids at the SAME grid the tokens are built on (do NOT try to reconcile with
  the separate T=32 `entities.py` cache — different frame grid, unnecessary complexity). Compute
  per-frame-per-track Δcentroid (bbox-height-normalized) as the new numeric feature.

## 5. 2022-2025 literature evidence

- **Loitering/dwell-time detection (production + academic consensus)**: rule-based dwell-time
  thresholds ALONE are the industry-standard way to separate "passage" from "standing," reported as
  effective without any appearance modeling — [Loitering Detection in CCTV](https://www.videoexpertsgroup.com/glossary/loitering-detection),
  [Farsight Security](https://www.farsight.co.uk/blog/loitering-detection-in-cctv-why-it-matters-and-how-to-set-it-up/).
  This is indirect but strong evidence that dwell/motion signal alone carries most of the
  discriminative power for exactly the "moving vs standing" distinction — likely a LARGER lift here
  than typical general-purpose motion-feature ablations report, because the task's two confused
  classes are close to being *defined* by this axis.
- **Núñez et al., "Identifying Loitering Behaviour with Trajectory Analysis," WACV 2024 Workshops**
  ([paper](https://openaccess.thecvf.com/content/WACV2024W/RWS/papers/Nunez_Identifying_Loitering_Behavior_With_Trajectory_Analysis_WACVW_2024_paper.pdf),
  code: github.com/johnnynunez/RS-WACV24_Loitering): trajectory-only features — turning angle θ,
  point-count/direction-changes, sliding-window dwell — used to flag loitering vs transit, no
  appearance. Confirms trajectory/dwell features are a standard, sufficient, camera-agnostic
  signal for exactly this behavioural axis, independent of detector appearance quality. (Their
  curvature feature targets *erratic wandering*, not plain motion/no-motion — matches this report's
  §1 verdict to deprioritize curvature here.)
- **Small-dataset multimodal fusion**: ["Improved Multimodal Fusion for Small Datasets with Auxiliary
  Supervision," arXiv:2304.00379](https://arxiv.org/abs/2304.00379) — high-capacity fusion overfits
  small/low-dim data; favors simple concatenation + auxiliary supervision. Directly informs §2 ranking.
- **Skeleton-based action recognition, velocity/derivative features**: velocity (first-order temporal
  derivative of joint coords) is repeatedly shown to carry "richer motion dynamics ... compared with
  raw skeletal coordinates" across multiple 2022-2024 GCN papers (search summary, multiple arXiv
  hits e.g. bone/velocity/acceleration multi-modality models). Analogous justification here: this
  repo's token currently has only STATIC per-frame box coords, no first-difference — adding the
  temporal delta is the same "derivative features beat raw coordinates" insight, transplanted from
  skeletons to boxes.
- **Pedestrian speed/monocular calibration**: bbox-size artifacts distort naive pixel-displacement
  speed estimates — supports §3's bbox-height normalization requirement (multiple MOT/speed-estimation
  patents and papers found; general pinhole-camera-projection consensus, no single canonical citation
  but consistent across all sources returned).
- **No paper found that studies this EXACT pair (industrial walkway-violation vs unauthorized-machine-
  intervention) or this exact dataset** — expected, this is a niche/small published dataset (Onal &
  Dandil 2025). All evidence above is transferred from adjacent domains (loitering detection, MOT
  speed estimation, skeleton action recognition), not a direct replication. Flag honestly.

## 6. Honest small-N caveats

- N=452 train / 64 test, 4 classes, smallest class UnauthorInt n=11 test / similarly small in train.
  Any new feature dimension is trained on effectively small per-class counts — do not add more than a
  handful of new numeric dims (§2 option 1: 2 dims recommended: bbox-height-normalized speed +
  straightness ratio) in one shot; ablate incrementally (1 dim → 2 dims), not all candidate features
  from §1 at once.
- k-fold macro 84.0% / LCO 78.6% already established as baseline; expect the motion addition to move
  LCO more than k-fold (motion targets the CROSS-camera-robust cue specifically) — if k-fold jumps but
  LCO doesn't, suspect the leak mode flagged in §1 (track-persistence-as-camera-fingerprint) or §3
  (unnormalized pixel-space motion correlating with view).
- Given UnauthorInt LCO recall is only 44.4% on n≈11-14 pooled-view clips, a handful of misclassified
  clips swings the number several points — treat any single-seed LCO delta <5pp as noise; use the
  existing multi-seed ensemble protocol (`run_lco_eval.py --seeds`) before concluding lift is real.
- Realistic expected lift: given how close the task is to a textbook motion/no-motion distinction
  (§5), a genuine, non-leaky implementation of §2-option-1 (2 motion dims added directly to person
  tokens) is plausibly the single highest-expected-value change available in this codebase right now
  — but "plausible high" here means treat any number >15pp LCO lift on UnauthorInt recall as
  suspicious/overfit-until-verified-on-held-out-seeds, not as a target to chase.

## Prioritized recommendation list

1. **[Do first, ~1-2h, lowest risk]** Wire existing `streams.py` clip-level motion vector into
   `RelationTransformer` via a `clip_proj` module copied from `unsafenet/temporal.py::TemporalNet`.
   Zero new feature engineering (reuse `sweep_results` motion tensors already built off
   `unsafenet/entities.py` tracks). Fast sanity check: does motion signal help AT ALL before
   investing in per-token integration. Expected lift: small-moderate on k-fold, uncertain on LCO
   (clip-level, can't bind to WHERE the person was stationary). Risk: low (few added params).
2. **[Do second, ~half-day, highest expected value]** Add track-linking to
   `build_relation_tokens.py` (reuse `unsafenet.entities.link_tracks`, already written) at its own
   T=16×N_MAX grid; compute per-person-token bbox-height-normalized frame-to-frame centroid delta
   (speed) + running dwell indicator; append as 2 extra numeric dims (`n_num` 9→11) consumed by the
   existing `num_proj`. This is the change most likely to fix the WalkwayViol/UnauthorInt confusion
   because it lets self-attention bind motion to geometry PER TOKEN PER FRAME. Risk: medium — verify
   track continuity holds across T=16's frame gaps (clip duration / 16), and run the §1/§3 leak check
   (does the feature correlate with view id more than label under LCO) before trusting any lift.
3. **[Do third, cheap add-on]** Straightness ratio (net_displacement/path_length) as a 3rd added dim
   — camera-scale-invariant by construction (ratio of two lengths), cheap given #2's track deltas are
   already computed, directly separates "walked through" (~1) from "fidgeted at a panel" (~0).
4. **[Optional ablation, after 1-3 land]** Secondary auxiliary loss: a `motion_head` alongside the
   existing `frame_head`/`aux_w` mechanism, regressing/classifying the per-frame motion state derived
   heuristically from the same features as #2. Only pursue if #2 alone underperforms — redundant
   supervision on top of already-available input, not a substitute.
5. **[Explicitly skip]** Optical flow (dense or lite) — §4, no evidence it beats track-deltas given an
   already-strong detector, adds compute+noise. **Skip.**
6. **[Explicitly skip]** Learned gating between motion/appearance streams — negative prior already
   exists in-repo (README §4.7.8, "temporal streams show no measurable lift on this dataset" for a
   gated-stream architecture on the OTHER pipeline). **Skip.**
7. **[Explicitly skip / deprioritize]** Trajectory curvature/turning-angle — targets loitering-vs-
   transit, not motion-vs-stationary; wrong feature for this exact pair, also noisy at T=16. Full
   homography calibration — real payoff but needs new metric-calibration annotation work not
   currently in the repo; revisit only if bbox-height normalization (§3) proves insufficient on LCO.
8. **[Guardrail, apply to every candidate above]** After each addition, rerun `run_lco_eval.py`
   per-view breakdown and check the new feature isn't just re-encoding camera identity (§1/§3 leak
   check) — this dataset already burned this exact failure mode once via appearance cosine similarity;
   motion features are not automatically immune.

## Unresolved questions

- Task brief describes a token with 512-d ROI appearance + track-id embedding; committed code has
  neither (10-dim geometric token, optional 64-d projected appearance off by default). Confirm which
  branch/checkpoint the "84.0% / 78.6%" headline numbers actually came from before building on top —
  if they're from a token version not in this checkout, the exact `n_num`/dims to patch may differ.
- Are clip durations in this dataset roughly constant, or do they vary enough (>20-30%) to need the
  §3 frame-rate normalization fix? Not verified against actual clip metadata in this pass.
- Is per-view detector/track quality already known to differ (e.g. from the entity-detector's
  per-view mAP breakdown)? If some views have materially worse detection, the track-persistence leak
  risk (§1) is not hypothetical and should be checked first, not just guarded after the fact.
