# Zone/region grounding for foot-in-zone features — research + implementable plan

Date 2026-09-02. Grounded in actual code (not hypothetical): `unsafenet/geometry.py`,
`build_relation_tokens.py`, `model_relation_head.py`, `tools/apply_zone_prior.py`,
`tools/check_zone_separation.py`, `annotation_guide/zones/view_*.json`, `annotation_guide/ontology-multi-entity.md`.

## 0. Root-cause diagnosis (confirmed, more precise than the prompt's framing)

Two **separate** pipelines exist and only one is broken:

- `unsafenet/streams.py` (Pipeline **P**, paper-replica path) already calls `geometry.clip_polygons()` →
  per-clip ORB+RANSAC homography + `green_walkway_mask()`. **This path is fine.**
- `build_relation_tokens.py` (Pipeline **O**, the box-token relation-Transformer — the actual thesis
  contribution and the one with `foot_in_walkway≈0.12`/AUC 0.629) has its **own inline `geometry()`**
  fn (line 99) that loads raw `annotation_guide/zones/view_*.json` polygons with **no homography at
  all** — one canonical polygon reused verbatim for every clip in a "view". This is the confirmed bug.
  `tools/apply_zone_prior.py` (source of the AUC=0.629 in `sweep_results/zone_prior/separation_report.json`)
  has the same flaw.
- **Second latent bug**: "view" id is not real camera metadata — `apply_zone_prior.py::build_clip_view_map`
  derives it from **KMeans(k=8) on pooled YOLO features**. If clustering mis-groups even a few clips
  (camera zoom/pan/relocation, similar-looking scenes), those clips get an entirely wrong polygon on
  top of the missing-homography problem. Worth a sanity audit before trusting any AUC delta.
- **Third finding, high leverage, currently unused**: the entity ontology already trains
  `Safe_Walkway` (id 2) and `Restricted_Zone` (id 3) as **detectable classes** in the same 8-class
  entity detector (`sweep_results/entity_detector/detector_best.pt`, bbox format — CVAT polygon
  annotations were exported to YOLO-detect axis-aligned boxes, confirmed via `annotation_guide/dataset/labels/*.txt`
  5-col format). `build_relation_tokens.py`'s per-frame detector call returns these boxes too, and
  since `N_MAX=8` and typical frame occupancy is ~3-7 boxes, **zone tokens are very likely already
  present in `tokens_{train,test}.pt`** — but `geometry()` only computes relational features for
  person classes (0/1) and ignores the zone boxes entirely when they're the detected class. This
  means class-conditioned self-attention *could already* be learning person↔zone proximity implicitly
  from raw box coordinates, uncontaminated by the homography bug — **verify with a coverage check
  before any of P0-P4** (near-zero cost, see plan).

---

## 1. Homography robustness: ORB+RANSAC vs alternatives

- **ORB+RANSAC (current `geometry.py`)**: fast, no GPU, works well on textured static backgrounds.
  Weakness: fails under strong lighting delta and clip-level occlusion of the reference region (a
  forklift/person parked over half the frame at the median-background sampling window). Search
  evidence: pixel-mean-error comparisons show ORB+RANSAC degrades sharply under low light (~69 px
  error) vs LoFTR+RANSAC (~15 px) in a recent benchmark — but that benchmark is generic imagery, not
  near-static industrial floor cameras where ORB is usually adequate because viewpoint delta ≈ 0 (only
  intensity/occlusion varies, not geometry).
- **ECC (`cv2.findTransformECC`)**: better than ORB for *small* residual misalignment between two
  already-similar frames (sub-pixel intensity-gradient alignment), but it needs a good initial guess
  and diverges under large occlusion/illumination change — not a replacement, a good **refinement
  step after ORB** gives the homography.
- **LoFTR / SuperGlue-lite (learned, dense/semi-dense matching)**: meaningfully more robust to
  lighting and partial occlusion because they match on learned features, not corners; but adds a
  torch model + inference cost per clip pair, and for 516 clips × 1 background reg it's a one-time
  offline cost, so compute is not actually a blocker. The real cost is engineering (add a dependency,
  test on Jetson-adjacent stack is irrelevant since this is offline preprocessing, not deployed path).
- **Phase correlation**: only handles pure translation; the current failure mode (polygon totally
  unaligned) suggests more than translation is at play (camera micro-pan/zoom-refocus across shoots),
  so phase correlation alone is insufficient but is a **useful cheap failure detector** (if the best
  phase-correlation translation already gives a bad fit, ORB will too — free early warning).
- **Failure detection + fallback (missing today)**: `geometry.py::homography()` currently silently
  falls back to `np.eye(3)` (identity) whenever ORB has <12 matches or `det` is out of a sane range —
  this is exactly the **silent-failure mode that produced AUC 0.629 in Pipeline O's non-homography
  path**, and it's a risk in `geometry.py` too if inlier COUNT is fine but inlier QUALITY is poor
  (RANSAC can return a "valid" but wrong H with few inliers spread over a small region). Recommend
  logging + a **reprojection-error / inlier-ratio threshold** (e.g., inlier ratio <0.35 or median
  reprojection error >6 px → flag `low_conf_homography=True`, fall back to identity **and** emit that
  flag as a model feature so the transformer can learn to discount zone-geometry evidence on flagged
  frames, rather than silently trusting a bad warp).

## 2. Region priors: color segmentation vs learned segmenter

- HSV green-paint mask (`geometry.py::green_walkway_mask`) is cheap, zero-training, and works as long
  as: (a) paint color/hue is consistent across the 8 cameras, (b) no other green objects (vests —
  note `Person_Authorized` is **defined as green vest** in the ontology, direct hue collision risk
  with the walkway mask — worth confirming vest-green vs paint-green HSV ranges don't overlap given
  identical `(38,45,30)-(90,255,255)` range currently used for the floor mask), (c) lighting is
  roughly diffuse (shadows/glare push saturation/value out of range).
- The mask is explicitly documented as a **second opinion**, not primary signal — right call. Use it
  as (i) a homography-failure cross-check (if homography-warped polygon and independent green mask
  disagree strongly, flag low confidence) and (ii) a fallback foot-in-zone feature when homography
  confidence is low.
- **Learned per-view segmenter beats color prior when**: paint is worn/faded/partially occluded by
  equipment, or when `Restricted_Zone` has no paint marking at all (per ontology it's a
  machine/control-panel area, not necessarily painted — the color prior is walkway-only by design and
  cannot help Restricted_Zone). Given only 8 fixed views, a **per-view learned segmenter is
  low-data-risk** (essentially memorizing 8 backgrounds) and won't generalize under LCO — same
  camera-invariance problem the thesis is trying to solve for. Recommend: do NOT invest in a learned
  segmenter for this thesis; the **existing detector's `Safe_Walkway`/`Restricted_Zone` box classes**
  (already trained, zero extra labeling) are the better learned-prior candidate since they were
  trained across all 8 views jointly (better chance of some cross-view invariance) — but re-verify
  their per-class mAP (unknown / not found in current logs — see unresolved questions).
- Floor-plane vs foot-point projection: foot-point (bbox bottom-center) is already the standard and
  correct choice here (confirmed by both `geometry.py` and `build_relation_tokens.py`) — a full
  floor-plane homography (image→ground-plane metric rectification) is unnecessary complexity (YAGNI):
  the zone polygons are already defined in image pixel space, so point-in-polygon on the image plane
  is sufficient; true metric floor-plane distances would only matter for continuous-distance
  reasoning across cameras with very different focal lengths, which is not needed for a binary
  in/out + signed-distance feature.

## 3. Best geometric features for person-in-region

Camera-invariance ranking (most → least invariant, i.e. best for LCO):
1. **Fraction of frames with foot inside polygon over the clip** (temporal aggregate of a binary) —
   most robust: a single frame's homography glitch is averaged out. Already partly present as
   `streams.py`'s `v13` (fraction off-walkway) but computed on the correctly-registered path; missing
   entirely from `build_relation_tokens.py`.
2. **Signed distance to polygon boundary, normalized by a scale-invariant reference** (frame width, as
   `geometry.py` does, or better: polygon's own bounding-box diagonal, which is more invariant across
   cameras with different framing/zoom than a fixed 960 px width) — camera-invariant if normalization
   is per-polygon rather than per-frame.
3. **Dwell-in-zone duration / longest containment streak** (already computed for *motion*, not zone,
   in `streams.py::_motion_vec` v7 "dwell fraction (near-stationary)") — this is exactly the feature
   the thesis's own docstring says separates the two classes ("WalkwayViol = walking across open
   floor; UnauthorInt = stationary at a machine"). **Combining "stationary" + "near/inside
   Restricted_Zone"** (dwell × zone containment, a product feature or two features fed together) is
   likely higher-yield than fixing `foot_in_walkway` alone, because per the thesis's own motion-vector
   docstring, motion/dwell is claimed as the *primary* discriminator, zone containment secondary.
4. **Containment continuity across frames** (in→out transition count / boundary-crossing count,
   already implemented as `z3_crossings` in `apply_zone_prior.py`, weak signal per the go/no-go probe
   — likely swamped by detector jitter at the boundary; least camera-invariant since jitter scale is
   pixel-space and interacts with each camera's resolution/zoom differently).
5. **Raw single-frame IoU(person_box, zone_box)** (current `iou_walkway`/`iou_restricted` in
   `build_relation_tokens.py`) — least invariant: IoU conflates box aspect ratio, is 0 for anyone not
   overlapping the box, and is scale-sensitive to how tightly zone boxes are drawn per view. Keep it
   as an auxiliary feature but do not rely on it as primary signal; signed distance + foot-in-polygon
   from correctly-registered polygons is the better replacement.

## 4. Encoding zones: tokens vs scalar features vs attention bias

Three real options, given `model_relation_head.py`'s existing architecture:
- **A. Zone-as-token (already structurally possible, likely partially happening today)**: since
  `cls_embed` vocab already includes ids 2/3, and zone boxes plausibly survive the `N_MAX=8` top-conf
  cut, self-attention between a person query token and a zone key token can in principle learn
  "near/inside" from the raw (xc,yc,w,h) coordinates alone — **no hardcoded rule needed**. Pros: fits
  the "let attention learn person×zone" goal exactly, zero new features. Cons: needs relative-position
  reasoning the model isn't explicitly given (current `pos` is a *learned absolute* positional
  embedding over the flattened (T×N) sequence index, not the box coordinates — the model has to learn
  coordinate arithmetic from `num_proj` alone, which linear attention can approximate but noisily with
  this little data, n≈516 clips).
- **B. Zone as derived scalar feature** (current approach: `iou_walkway`, `foot_in_walkway`,
  `norm_dist` appended to the *person* token) — pros: gives the model the geometric relation for free,
  much more sample-efficient than hoping attention discovers it from raw coordinates with <1k clips;
  cons: hardcodes which relation matters (a rule-engine flavor), fragile to the registration bug (this
  is exactly the mechanism that's currently broken).
- **C. Attention bias term** (add a learned or geometric bias to the attention logits as a function of
  person-token↔zone-token relative position/containment, à la relative-position-encoding /
  camera-aware RPE work) — a middle ground: keeps the "let attention learn it" spirit while giving the
  model an explicit geometric prior to bias toward, and does not hardcode the *decision* (in/out
  threshold), only the *relation* the model should weigh. More engineering (custom attention layer)
  than A/B.
- **Recommendation for this thesis (YAGNI + data-scale honesty, n≈516)**: **keep B, fix the input
  (P0/P1/P2 below), and additionally emit the aggregate features from §3** (dwell×containment,
  fraction-in-clip) as extra scalar dims on the CLS-adjacent summary, not per-frame per-box — this
  reuses the existing architecture with zero new modules. **Do not build C** for this dataset size;
  attention-bias/relative-PE modules are proven on datasets with far more supervision (scene-graph /
  detection-scale) and are very likely to overfit or be un-trainable at n=516 clips over 5-fold or LCO
  splits. Do run the **A-diagnostic** (zone-token coverage check, near-zero cost) before deciding
  whether B's fix alone is sufficient or whether A is already silently doing some of the work (which
  would explain part of any lift/no-lift after P0).

## 5. Citations + realistic expected lift

- Perimeter/pedestrian intrusion detection surveys confirm ROI-polygon + foot-point-in-polygon is the
  standard, well-proven design (not novel, but correct): PIDNet, arXiv:2009.00312 (dynamic pedestrian
  intrusion, foot/base-point membership test — matches this thesis's foot-point choice exactly);
  "Perimeter Intrusion Detection by Video Surveillance: A Survey", PMC9104546 (ROI-polygon alarm
  zones, standard practice). These systems, with *correctly registered* zones, typically report
  detection/membership decisions in the high-90s % accuracy range for the geometric test itself
  (it's a solved sub-problem *given correct registration* — the registration is the whole issue here).
- Homography robustness under lighting delta: recent benchmark work (SeFENet arXiv:2412.06352 and
  related LoFTR-vs-classical comparisons) shows learned dense matchers (LoFTR/SuperGlue-family)
  reduce reprojection error by ~4-5× vs ORB/SIFT+RANSAC under low light — directionally supports
  ORB as "usually fine, occasionally badly wrong," i.e. exactly a case for **failure detection**
  rather than wholesale replacement.
- Relative-position / geometry-aware attention: general RPE literature (ICCV'21 "Rethinking RPE for
  ViT"; GRPE for graph transformers) supports option C conceptually but all cited works train on
  orders of magnitude more data than 516 clips — cited to justify *not* pursuing it here, not to
  justify pursuing it.
- **Realistic expected lift for WalkwayViol vs UnauthorInt separation**: current combined AUC = 0.629
  (measured, `separation_report.json`, walkway-only features, unregistered). Fixing registration
  (P0) alone, by analogy to intrusion-detection literature where a correctly-aligned polygon test is
  near-deterministic, should recover **most of the "correct polygon" ceiling** — realistically
  **AUC 0.75-0.85**, not 0.95+, because: person detections still have some localization noise, the
  polygon itself may be imprecisely hand-drawn, and `Restricted_Zone` per the ontology is not
  color-marked (no independent cross-check like the walkway's green mask). Adding dwell/stationary
  features (§3.3, motion × zone) on top is plausibly what pushes it further, per the thesis's own
  hypothesis that motion, not raw position, is the primary discriminator — **do not oversell zone-fix
  alone as sufficient**; expect it to close roughly half the AUC gap to a "clean" ROI-intrusion-style
  system, with the motion/dwell combination doing the rest.

---

## Prioritized implementable plan (reuses `unsafenet/geometry.py`)

| # | Action | Reuses | Expected lift (AUC, Walk vs Unauth) | LCO risk | Compute cost |
|---|---|---|---|---|---|
| **P-1 (do first, ~30 min)** | Diagnostic: check zone-token (`cls_id ∈ {2,3}`) coverage per frame in existing `tokens_{train,test}.pt`, and re-run `check_zone_separation.py` output confidence of `apply_zone_prior.py`'s KMeans view-clustering against any available real camera metadata. | existing tensors, no new code beyond a script | n/a — informs whether P0 alone suffices or A (zone-as-token) is already partially compensating | none (read-only) | trivial (<1 min) |
| **P0** | Replace `build_relation_tokens.py::geometry()`'s raw `zones[view]` lookup with `unsafenet.geometry.clip_polygons(clip, view)` (per-clip ORB+RANSAC-warped polygons); requires each clip to resolve a `clip`-like object / background path compatible with `geometry.homography()` — adapt call signature (`bg_path` expects the `unsafenet` clip index, `build_relation_tokens.py` iterates raw glob paths — needs a thin adapter, not a rewrite). | `unsafenet/geometry.py` directly | **+0.10 to +0.20** (0.629 → ~0.73-0.83), by far the largest single lever — this is fixing the confirmed root cause | **low-medium**: homography needs a reference background *per view*; an unseen LCO camera has no reference at all → must document "held-out camera has no zone-registration reference" as an explicit LCO limitation (falls back to identity, i.e. today's broken behavior, for genuinely new cameras) | low — one-time offline; ORB+RANSAC per clip ≈ 516 clips × <1s, already cached via `_H_CACHE` pattern in `geometry.py` |
| **P1** | Add homography failure detection (inlier ratio / reprojection error threshold) to `geometry.homography()`; emit `low_conf_homography` flag; feed into token features as an extra dim rather than silently trusting a bad warp. | `geometry.py::homography()`, extend | **+0.02 to +0.05** on top of P0 (fewer silently-corrupted frames) | low — flag itself is camera-agnostic | negligible |
| **P2** | Add clip-level aggregate features per §3: fraction-of-frames foot-in-walkway, longest containment streak, and the **dwell × zone** product feature (reuse `streams.py::_motion_vec`'s dwell computation pattern, apply the *same* per-clip-registered polygon from P0). Feed as clip-level scalars concatenated at the CLS head, not per-box tokens. | `streams.py::_motion_vec` pattern + P0's registered polygons | **+0.05 to +0.10** (thesis's own hypothesis: motion/stationarity is the primary discriminator, zone position secondary) | low — ratios/fractions are scale- and camera-invariant by construction | low — pure post-processing over already-detected tracks |
| **P3** | Wire `green_walkway_mask()` as an independent cross-check feature (agreement/disagreement with the homography-warped polygon) rather than only inside `unsafenet/streams.py`; use disagreement as a second low-confidence flag. | `geometry.py::green_walkway_mask()` | **+0.01 to +0.03**, mostly a robustness/robustness-audit gain, not raw separation | **medium**: HSV range is hand-tuned on current cameras' paint hue; an LCO camera with different paint/lighting may silently produce a useless mask — must be treated as optional evidence, never a hard filter | negligible (mask computed once per clip background, already cached) |
| **P4 (larger, optional)** | Investigate using the detector's own `Safe_Walkway`/`Restricted_Zone` **box** detections per-frame as a homography-free complementary geometry source (no registration needed by construction); first verify per-class mAP for ids 2/3 (currently unknown/unverified — see below). If detection is reliable, use zone-box centroid/edges as a coarse per-frame in/out test to sanity-check P0's polygon warp. | existing `entity_detector/detector_best.pt`, no retraining | **+0.02 to +0.05** as a cross-check; **not** a replacement for polygon geometry (axis-aligned boxes are a poor fit for the irregular, diagonal, multi-segment walkway shape seen in `view_1.json`) | low-medium: detector's own generalization to unseen cameras is a separate, already-being-measured LCO question (not new risk introduced by this feature) | low — reuses inference already run for entity tokens, no extra forward pass |
| **Not recommended** | Learned per-view segmenter (§2) or attention-bias/relative-PE module (§4-C) | — | unclear / likely negative at n≈516 | high — per-view segmenter is view-specific overfitting, directly hostile to LCO; attention-bias module needs much more data than available | high engineering cost for uncertain return — skip (YAGNI) |

**Suggested order**: P-1 → P0 → P1 → P2 → (P3, P4 opportunistic/parallel). Re-run
`tools/check_zone_separation.py`-equivalent (pointed at the new registered features) after P0 and
again after P2 to get honest, measured AUC deltas before claiming any number in the thesis — the
current AUC 0.629 must be re-measured, not assumed fixed.

## Unresolved questions
1. Per-class mAP for `Safe_Walkway` (id 2) / `Restricted_Zone` (id 3) in the entity detector is not
   found in any existing log/artifact — needed to judge P4's viability. Run a per-class eval.
2. `apply_zone_prior.py`'s "view" is KMeans-derived from pooled appearance features, not verified
   against actual camera identity/count (task says "8 fixed cameras" — is KMeans(k=8) confirmed to
   recover the true 8 cameras, or just an assumption)? A mislabeled view would corrupt P0 too (wrong
   reference background chosen), so this should be audited before or alongside P0.
3. `build_relation_tokens.py` iterates raw `glob` clip paths with no `clip.key`/`bg_path` concept that
   `unsafenet/geometry.py` expects — the P0 adapter needs a design decision: reuse `unsafenet.data_index`
   clip objects directly (cleanest, requires aligning `build_relation_tokens.py`'s clip discovery with
   `unsafenet`'s), or extend `geometry.py` with a path-based overload. Recommend the former for DRY.
4. No ground truth for "correct" polygon-to-clip alignment exists to directly validate homography
   quality (e.g., manually re-verified clips) — P1's threshold values (inlier ratio, reprojection
   error) will need calibration against a small manually-eyeballed sample, not just picked a priori.

Sources:
- [Restricted Zone Violation Detection with Foot-Based Spatial Intelligence](https://projecthub.arduino.cc/rajanmistry/restricted-zone-violation-detection-with-foot-based-spatial-intelligence-a0ef46)
- [Perimeter Intrusion Detection by Video Surveillance: A Survey (PMC9104546)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9104546/)
- [PIDNet: An Efficient Network for Dynamic Pedestrian Intrusion Detection (arXiv:2009.00312)](https://arxiv.org/pdf/2009.00312)
- [SeFENet: Robust Deep Homography Estimation via Semantic-Driven Feature Enhancement (arXiv:2412.06352)](https://arxiv.org/pdf/2412.06352)
- [Robust Perspective Correction for Real-World Crack Evolution Tracking (arXiv:2506.22437)](https://arxiv.org/pdf/2506.22437)
- [Rethinking and Improving Relative Position Encoding for Vision Transformer (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Rethinking_and_Improving_Relative_Position_Encoding_for_Vision_Transformer_ICCV_2021_paper.pdf)
- [GRPE: Relative Positional Encoding for Graph Transformer (OpenReview)](https://openreview.net/forum?id=GNfAFN_p1d)
