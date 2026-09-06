# Object-centric relation-transformer tricks for a 452-clip, scene-confounded, 4-class problem

Scope: RelationTransformer (`model_relation_head.py`, ~298k params, d=128, 2 layers, 4 heads) fed by
`build_relation_tokens.py` (T=16, N_MAX=8, feat=[class,xc,yc,w,h,conf,iou_walkway,iou_restricted,
foot_in_walkway,norm_dist]+optional 64-d projected appearance). Current: k-fold macro 84.0%, LCO 78.6%,
5.4pp gap = camera memorization. Confusion: WalkwayViol vs UnauthorInt. Verified codebase state below
before recommending — several "obvious" ideas are already implemented; don't re-suggest them.

## What's already in the pipeline (don't re-propose)
- Zone-relative geometric priors ARE injected as deterministic features (iou_walkway, iou_restricted,
  foot_in_walkway, norm_dist) computed from real zone polygons per camera view — this is the "inject
  relation prior without hardcoding rules" pattern from Q2, already done at the feature-engineering level.
- Appearance already frozen (no backbone fine-tuning) + random-projected to 64-d (not full 512-d) —
  Q4's "compress/project appearance" is already done, but via a **fixed random** Gaussian projection
  (not learned), and with 30%-context crops (background leaks into appearance).
- Label smoothing (0.05), dropout (0.3 attn / 0.2 head), aux per-frame head already present.
- A pairwise contrastive 8-class safe/unsafe classifier exists (`run_pairwise_contrastive.py`) but
  operates on whole-clip 512-d features, not per-entity-crop embeddings feeding the token transformer.

## Root-cause read on the WalkwayViol vs UnauthorInt confusion + LCO gap
Both classes are near-boundary geometric events (foot crosses walkway edge vs person enters restricted
zone) that a single-frame IoU/foot-in-polygon feature can only partially disambiguate — real confusion is
plausibly **duration/persistence** (a lingering unauthorized presence vs a transient boundary touch), not
purely instantaneous geometry. Separately, raw `xc,yc` (absolute pixel position) is still in the token,
and with only 8 fixed cameras this is a literal camera-fingerprint channel — it inflates k-fold (camera
seen in both folds) but can't generalize LCO. This is the single most actionable, cheapest lever below.

---

## Prioritized recommendations

### P1 — Strip absolute position, keep only relative/geometric features (cheap, ship first)
Remove `xc,yc` from the numeric feature vector fed to the transformer (keep `w,h` as scale proxy +
already-computed `iou_walkway/iou_restricted/foot_in_walkway/norm_dist`). Absolute coordinates are a
pure camera-identity channel with 8 fixed views; Relation Networks (Hu et al., CVPR'18,
https://arxiv.org/abs/1711.11575) and geometry-aware attention work (GTA, ICLR'24,
https://arxiv.org/html/2310.10375v2) both show relation reasoning improves when position is expressed
*relative* to reference geometry, not absolute pixel frame. **Expected**: k-fold may dip 0–2pp (loses a
"free" camera-conditioned shortcut) but LCO should close some of the 5.4pp gap — this is the direct fix
for the diagnosed leak. **LCO risk: low (this reduces overfitting, doesn't add capacity).** Zero new
params. Ablate as A/B, keep whichever wins on LCO since that's the honest metric.

### P2 — Track-linked dwell/persistence feature (moderate effort, addresses confusion directly)
Task spec mentions track-id in the token but the current 10-feat layout has none — add a per-track
running "time-in-zone-state" scalar (frames-in-restricted / frames-with-foot-outside-walkway, computed
deterministically from the existing tracker, cheap) as an 11th numeric feature, or aggregate at CLS via
the existing frame_head. This targets the *actual* discriminator between a transient crossing and a
sustained violation. Analogous to video-transformer object-persistence encodings in ORViT (CVPR'22,
https://arxiv.org/abs/2110.06915) and DAIR (https://arxiv.org/abs/2404.11903), which fold object-track
continuity into the relation head for compositional/few-shot action recognition. **Expected: +2-4pp
macro on the confused pair** if duration truly is the discriminator (plausible given single-frame IoU
already exists and confusion persists) — this is the highest-expected-value item but needs verifying the
duration hypothesis first (cheap: plot mean dwell time per class on existing features before implementing).
**LCO risk: low** — it's a deterministic geometric aggregate, not a learned embedding, same category as
existing zone features.

### P3 — Domain-adversarial camera-invariance head (targets LCO gap specifically)
Add a small linear "predict camera/view ID" head off the CLS embedding with a gradient-reversal layer
(DANN, Ganin & Lempitsky 2015) — trains the shared representation to be uninformative about camera while
task head still predicts class. Directly attacks "scene-confounded, camera predicts behaviour" — the
paper's own diagnosed failure mode. Cheap (~8-way linear head, few hundred params). **Expected: LCO
+1-3pp**, k-fold roughly flat or -1pp (removing a shortcut). **LCO risk: this is explicitly an
anti-overfitting mechanism — low risk, but tune the GRL lambda schedule (ramp 0→1) or it destabilizes the
tiny 298k model.** Combine with P1 rather than instead of — P1 removes the channel, P3 removes it even if
re-derived from correlated cues (box size, entity mix per camera).

### P4 — Class-preserving cross-camera token MixUp (bespoke augmentation, not generic MixUp)
Standard image MixUp/CutMix (Yun et al., ICCV'19, https://arxiv.org/abs/1905.04899; TokenMix,
https://arxiv.org/abs/2207.08409) mixes inputs+labels of *any* two samples — generic version is
risky here because mixing WalkwayViol-cam3 with UnauthorInt-cam7 tokens blends two label sets that
correspond to different real geometries and produces incoherent supervision at N=452. Instead: mix only
**same-label, different-camera** clips (interpolate numeric geometry features / token subsets, label
unchanged) — a targeted invariance-inducing augmentation, not a generic regularizer. This is the
few-shot/small-N-safe variant. **Expected: LCO +1-2pp**, marginal k-fold change. **LCO risk: low
(label-preserving by construction)**; implementation cost moderate (need clip pairs indexed by
label×view). Do NOT do generic random-pair MixUp on this dataset — 4 classes × 8 confounded cameras
means random mixing has a real chance of producing label-incoherent synthetic examples given N this
small; skip it.

### P5 — Pairwise/edge relation tokens beyond person-vs-zone (moderate effort, moderate risk)
Add person↔person and person↔object (panel, cart) relative-geometry pairs (dx,dy,log(w-ratio),
log(h-ratio), IoU, distance) as extra edge features or via a Relation-Networks-style geometric attention
bias added to QK logits (Hu et al. CVPR'18; VS-GATs, https://arxiv.org/abs/2001.02302; VSGNet,
https://arxiv.org/abs/2003.05541). Useful in principle (HOI literature reports several-point macro gains
from pairwise geometric attention on order-10³–10⁴-sample datasets) but at N=452 the marginal pairs
(person-object, not person-zone) are sparse per clip and add real params/complexity. **Expected: +1-2pp
macro, uncertain** — lower confidence than P1/P2 because HOI papers train on orders of magnitude more
data. **LCO risk: moderate** — new learned pairwise attention can re-discover camera-specific object
co-occurrence (e.g., only camera 5 has a cart) unless pair features are camera-normalized like the
existing zone IoU features. Do only after P1–P3 land and only if confusion persists.

### P6 — Learned (not random) appearance projection, tighter crop, higher label smoothing
Three small, low-risk tweaks bundled because each is <1 line of code and individually marginal:
(a) replace the fixed random 512→64 projection with a small trainable `nn.Linear` + weight decay
(random projections are a JL-embedding, not optimized for the task; a light learned layer is safe at
this N given the backbone stays frozen); (b) drop the 30% context padding on crops when using
`--appearance` (context crops leak background = camera identity, same failure mode as P1, but in the
appearance channel instead of coords); (c) raise label smoothing 0.05→0.1 given 452 train clips / 4
classes (SPT-LSA ViT work on small-dataset ViTs, https://arxiv.org/abs/2112.13492, and general
small-data ViT regularization recipes converge on stronger smoothing + dropout, not larger models).
**Expected: +0.5-1.5pp combined**, mostly stabilizes variance across seeds/folds rather than raw
accuracy. **LCO risk: low.**

### P7 — Token dropout (simulate missed detections)
Randomly zero 10-20% of *valid* token masks per training step (not padding — actual detections). Cheap,
forces non-reliance on any single frame/box, complements existing frame-level corruption robustness
work (`--corrupt` flag already exists). Loosely analogous to AttentionDrop
(https://arxiv.org/abs/2504.12088) and standard structured dropout on set-transformers. **Expected:
+0.5-1pp, mainly robustness/variance not raw macro.** **LCO risk: low**, essentially free regularization.

### P8 (deprioritized) — Supervised contrastive pretext on entity crops to separate the confused pair
Feasible only if crops are camera-normalized (else contrastive learns camera texture, not behaviour —
exactly the leak this whole task is fighting). At N=452 clips × ~2 person-tokens/clip the number of
usable anchor-positive-negative triplets per class is small; contrastive objectives are known to need
more samples per class than plain CE to beat it (see fine-grained contrastive surveys,
https://arxiv.org/abs/2104.04323, https://arxiv.org/abs/2407.14676 — gap between SSL/contrastive and
supervised CE widens as task granularity increases, i.e., worse exactly for a 2-class confusion problem
like this). **Expected lift: uncertain, plausibly 0 to +1pp, real chance of net-negative** if pairs
aren't camera-balanced. **LCO risk: high** unless contrastive pairs are explicitly constructed
same-class/different-camera positive, same-camera/different-class hard-negative — otherwise it actively
teaches the camera shortcut. Do last, small-scale pilot only, don't commit thesis time to it before
P1-P3.

### Explicitly do NOT do (net-negative or wasted effort at this N)
- **Scaling the transformer up** (more layers/heads/dim). 298k params is already correctly sized;
  literature on small-dataset ViTs (SPT-LSA, https://arxiv.org/abs/2112.13492; MSCViT,
  https://arxiv.org/abs/2501.06040) is about adding *inductive bias*, not capacity, when data is scarce.
  More capacity on 452 clips will widen the k-fold/LCO gap, not close it.
- **Fine-tuning the appearance backbone.** Currently frozen — correct call at this N; unfreezing risks
  the backbone memorizing camera texture even faster than the projection head does.
- **Generic random-pair MixUp/CutMix** across all classes/cameras (see P4 rationale).
- **Stochastic depth** — only 2 layers; nothing meaningful to "depth-drop."

---

## Honest ceiling
At 452 train / 64 test / 8 fixed cameras, k-fold and LCO measure different things and neither is a
precise estimator — the plan.md already documents this (val↔test macro r=-0.066, n.s. McNemar on 64
clips). None of P1-P7 will make LCO match k-fold; the 5.4pp gap partly reflects genuine cross-camera
distribution shift that no architecture trick fully removes with 8 cameras. Realistic combined target:
LCO 78.6% → ~82-84% if P1+P2+P3 all land as expected (compounding, not fully additive); k-fold roughly
flat to +1pp. Treat any run reporting >90% LCO on this data as a debugging signal (leak), not a win.

## Sources
- Hu et al., Relation Networks for Object Detection, CVPR 2018 — https://arxiv.org/abs/1711.11575
- Herzig et al., Object-Region Video Transformers (ORViT), CVPR 2022 — https://arxiv.org/abs/2110.06915
- Simultaneous Detection and Interaction Reasoning (DAIR), 2024 — https://arxiv.org/abs/2404.11903
- Liu et al., VS-GATs for HOI Detection — https://arxiv.org/abs/2001.02302
- VSGNet: Spatial Attention Network via Graph Convolutions — https://arxiv.org/abs/2003.05541
- GTA: Geometry-Aware Attention for Multi-View Transformers, ICLR 2024 — https://arxiv.org/html/2310.10375v2
- Lee et al., Vision Transformer for Small-Size Datasets (SPT/LSA) — https://arxiv.org/abs/2112.13492
- MSCViT: Small-size ViT for Tiny Datasets, 2025 — https://arxiv.org/abs/2501.06040
- Yun et al., CutMix, ICCV 2019 — https://arxiv.org/abs/1905.04899
- TokenMix: Image Mixing for ViT Data Augmentation — https://arxiv.org/abs/2207.08409
- AttentionDrop regularization, 2025 — https://arxiv.org/abs/2504.12088
- Context-Aware Aerial Object Detection (inter-object + background relation) — https://arxiv.org/abs/2404.04140
- Object-centric Video Representation for Long-term Action Anticipation, WACV 2024 — https://arxiv.org/abs/2311.00180
- Fine-grained contrastive representation learning survey — https://arxiv.org/abs/2104.04323, https://arxiv.org/abs/2407.14676

## Unresolved questions
1. Is the WalkwayViol/UnauthorInt confusion actually a duration/persistence issue (P2's premise), or
   pure zone-boundary geometric ambiguity (annotation edge cases where a person is legitimately near
   both zones)? Needs a quick confusion-matrix + per-error-clip visual audit before investing in P2.
2. Does `--appearance` (64-d) currently improve or hurt LCO specifically (not just k-fold)? Not
   established in the read code — if it already hurts LCO, P6's crop-tightening is more urgent than P1.
3. Track-id: is there an existing tracker (e.g. ByteTrack via ultralytics) already wired but unused in
   `build_relation_tokens.py`, or does P2 require adding tracking from scratch? Affects P2 effort estimate.
4. Is per-camera train/test split size (8 views over 452+64 clips → ~65 clips/view) large enough for
   LOCO-style domain-adversarial training (P3) to have enough camera classes for a stable GRL signal, or
   does 8-way is too few domains for DANN to help meaningfully?
