# Abstract (draft — deployment-led reframe)

Automated recognition of unsafe worker behaviours from surveillance video is a core problem for
industrial occupational safety. **The headline result of this thesis is a leak-free binary hazard
detector that was actually deployed.** On **iSafetyBench** (1,100 clips, binary hazard/normal;
420 hazard / 680 normal over 151 YouTube source groups, none of which supplies both classes) a
frozen CLIP ViT-B/32 tower feeding a small temporal Transformer head (B1, 201,090 parameters x 3
seeds) scores **84.2 % balanced accuracy under LSO-1100**: ten folds grouped by *source video*, so
no source can supply clips to both sides of a split. The figure is reported against a
**pre-registered 63.3 % floor** — a three-feature detector-count rule fitted on identical folds —
never against 50 %. **The same configuration, on the same 1,100 clips, scores 97.7 % when the split
is allowed to leak** — ten `StratifiedKFold` folds ignoring source identity, everything else held
fixed — so the protocol itself is worth **+13.5 pp** on the deployed graph (clustered bootstrap CI
[+8.0, +20.0], exact McNemar p = 1.39e-41 over 156 discordant pairs). The thesis therefore *measured*
the ninety-something number its own deployed model produces under the literature's protocol, and
chose to publish 84.2 %. That random-split run was **not** pre-registered (gate G12 was registered on
the relation-token lane) and is reported as a post-hoc extension, not as an adjudicated gate.
That same head is exported to TensorRT FP16 and measured on an Orin Nano Super
8 GB (§5.6.9), where the **composed** loop — decode, preprocess, tower, head, nothing omitted —
sustains **5.98 decisions/s** against this corpus's median real-time contract of 1.249 decisions/s
(a 4.79x margin; 4.40 decisions/s and a 3.52x margin on all-1080p sources, against that
same median contract),
and where the detector is not in the class decision at all: YOLOv11s supplies localisation and is
timed separately, never summed into the decision cost. The protocol carries as much of the claim as
the number does: on this corpus a random fold turns source identity into a label proxy and inflates
balanced accuracy by **+15.4 pp** (measured on the E3 encoder, McNemar p = 3.6e-45), while the same
swap moves the three-feature count-rule floor by only +1.0 pp — a capacity contrast that identifies
source memorisation, not a better fit, as the mechanism. A **placebo-grouping control** (20 seeds)
separates that inflation from the competing explanation that grouped folds are merely structurally
harder: fake source ids matched to the real ones on group count, on the per-class multiset of group
sizes and on class purity — provenance destroyed, geometry preserved — score **89.0 % (sd 0.47)**,
with no seed coming within 12.5 pp of the real leave-source-out number (permutation p = 0.0476).
The +15.4 pp therefore decomposes into **+1.6 pp fold geometry** and **+13.8 pp source provenance**.
This control was **not** pre-registered — only its criteria were fixed before it ran — and it
adjudicates no gate.

**A second lane** carries the thesis's evaluation-rigour contribution on the 4-class benchmark the
work started from. The published Unsafe-Net system (Önal & Dandıl, 2024) reports 95.81 %
macro accuracy on a four-class unsafe-behaviour benchmark using a YOLO v4 per-frame behaviour detector
(trained on 2,262 behaviour-bounding-boxes) followed by a ConvLSTM temporal classifier. This thesis
re-implements that pipeline from scratch with two modernisations — a **YOLOv11s (extensible to YOLO26)
spatial encoder** replacing YOLO v4 and a **box-token spatial-relation Transformer** replacing ConvLSTM —
but under a different supervision philosophy: rather than labelling behaviours directly, we annotate an
**eight-class multi-entity ontology** (people by vest colour, walkway/restricted zones, panel and forklift
states) and let the Transformer infer behaviour from the *spatial relations* between detected entities.

On the same 4-class, 64-clip test protocol, the entity detector reaches 0.969 mAP@50 and the behaviour
classifier reaches 54/64 (84.4 % top-1, 88.4 % macro) — matching the no-bbox baseline on macro. That lane's
contribution, however, is an **evaluation-rigor finding: the Unsafe-Net benchmark is
scene-confounded.** With eight fixed cameras and strong camera↔behaviour correlation, a *scene-only*
baseline that predicts behaviour from the camera identity alone scores 81.2 %, and a whole-frame
classifier (the paper's scene-dependent style) scores an identical 81.2 % — i.e. same-distribution
accuracy largely measures *which camera*, not the behaviour. Under a **leave-camera-out (scene-robust)
protocol** a trained, transferable scene-appearance baseline reaches only 64.4 % macro while the
proposed entity-relation model retains **81.2 % macro (77.7 % top-1)** — an **honest margin of +16.9 pp
macro, 95 % bootstrap CI [+12.5, +21.2]** (n_boot = 5 000 over 516 pooled clips — note that this
resamples **clips, not the 8 context groups**, which §6.12(a) of the thesis declares the wrong
sampling unit; the interval is descriptive and is not used as certifying evidence). The per-class
pattern is *consistent with* the margin being carried by the spatially-defined classes (OpenedPanel
88.7 vs 88.0 and CarryOverload 100 vs 100 tie; WalkwayViolation 70.5 vs 17.6 and UnauthorizedIntervention
65.7 vs 51.9), but we deliberately do **not** rest the claim on those four numbers: a stability study
(§7.8) finds LCO per-class recall varies by up to ~20 pp under perturbations that leave the aggregate
margin unchanged. The evidential claim is the aggregate margin; the per-class breakdown is reported as
descriptive support only. We therefore argue that same-distribution
top-1 on this benchmark is substantially scene-inflated, and that a leave-camera-out margin — not
any same-distribution top-1 — is the only defensible way to measure this task. We do not extend that
to a claim that entity decomposition is what a deployed system needs: on the binary lane the
pre-registered entity gate **failed** (G14, −0.5 pp against the capacity-matched baseline, cluster CI
[−5.5, +5.0] including zero, McNemar p = 0.683), and the model actually deployed is accordingly the
whole-frame CLIP tower of the first paragraph, not the entity-relation pipeline. We do **not** claim the paper's own model would fail cross-camera: its
weights and per-clip predictions are unreleased, so it cannot be run under this protocol. The claim is
about the *metric*, not about their system. The work is supported by a full multi-lens evaluation (macro-F1,
balanced accuracy, Cohen's κ = 0.775, MCC = 0.785, per-class miss-rate and cost-sensitive safety
metrics, one-vs-rest ROC-AUC = 0.951, PR-AUC = 0.880, ECE = 0.097, and BCa/Wilson/McNemar statistics)
and an edge-deployment characterisation (YOLO26 + Jetson TensorRT **FP16** — INT8 was never built
for P3, no calibration set exists for it). The P3 verdict is a **composed** measurement — decode,
preprocessing, detector, context crops, appearance encoder, 74-d relation-token assembly and head
inside one wall clock, nothing omitted — and it is **clip-dependent and clock-dependent**: with the
clocks pinned P3 sustains **2.49-3.97 decisions/s (median 3.48) and meets its 3.0 decisions/s
contract on two of three test clips**, while under the default governor it sustains **1.14-1.51
(median 1.44) and meets it on none**. The two arms are disjoint and pinning is worth 2.41x. An
earlier P3 headline built as **a sum of separately-timed stages** is withdrawn: that construct
double-charged a 16-frame window where the contract is defined on an 8-frame stride, *and* omitted
the input preprocessing this same campaign found to be the largest single cost in the deployed B1
graph, so it runs **7.98 % too fast when pinned and 39.3 % too slow under DVFS** — its bias changes
sign with the clock state, and no correction factor could have rescued it. The mechanism is recorded
in the artifacts themselves: under the default governor the second CPU cluster stays parked at its
115 MHz floor, and the composed loop is CPU-bound on precisely the preprocessing the stage sum left
out — so the omission hid the component most sensitive to the clock condition. This is a second
instance, on a different graph, of this thesis's own finding that a stage sum is not a cadence. Every 4-class figure in this paragraph comes
from the **same predictor** — the val-selected blind rule scoring 54/64 — including the probability-based
ones; earlier drafts had to compute those on a seed-averaged ensemble (52/64) because the blind seed's
softmax had not been persisted, and that mismatch is now resolved (§7.5).

**The two lanes are different corpora and different tasks.** 84.2 % balanced accuracy over 1,100
source-grouped clips and 84.4 % top-1 over 64 same-distribution clips are different metrics, under
different protocols, on different data; they are **not** comparable, and this thesis never averages
them, never compares them, and never quotes the larger of the two.

---

## Honest headline (one paragraph)

The deployed claim is the one above: 84.2 % balanced accuracy, leak-free, on the graph that runs on
the device — and that same graph reports **97.7 %** once the split is permitted to leak, a +13.5 pp
protocol gap measured on the deployed configuration itself. That pair of figures is iSafetyBench
binary throughout and is **never** set against any 4-class number in this paragraph. On the 4-class benchmark this thesis does **not** beat the paper's 95.81 % on the
same-distribution top-1 metric, and it is candid that it should not try to: that metric, on this benchmark, is scene-confounded — a model that ignores the
video and guesses the camera scores 81.2 %, and the paper-style whole-frame classifier does no better. What
the thesis **does** deliver is (a) a **leak-free, deployed binary detector** — 84.2 % balanced
accuracy over 1,100 source-grouped clips against a pre-registered 63.3 % floor, exported to FP16 and
measured end-to-end on an Orin Nano Super at 5.98 decisions/s; (b) a modern, reproducible
**YOLOv11/Transformer entity-relation pipeline**
(54/64, 88.4 % macro, above the no-bbox baseline on macro) whose supervision is placed on interpretable
entities rather than behaviours; (c) a **rigorous scene-robustness analysis** showing that under
leave-camera-out evaluation the entity-relation model generalises substantially better than a trained
scene-appearance baseline (81.2 % vs 64.4 % macro, +16.9 pp, clip-unit CI [+12.5, +21.2], not a
group-unit interval and not certifying evidence — §6.12(a)), because it reasons over
person×zone geometry instead of memorising scenes; and (d) the resulting **critique that same-distribution top-1 on this
benchmark rewards scene memorisation**, together with a fairer, cross-camera evaluation protocol for
the task. The contribution is a measured, deployed system reported under a protocol that cannot leak,
plus a correction to how this benchmark should be measured — a more defensible result than a
scene-inflated accuracy number. Where a pre-registered gate failed, including the entity gate the
second lane motivated, the thesis reports the failure rather than the flattering alternative.


---

**Status note (2026-09-06).** The leave-camera-out figures above are now the **canonical 5-seed run**
(`sweep_results/relation_tokens/canonical/lco_seeds5_yolo26L1_CANONICAL.log`, 60 epochs, 8 camera-groups,
516 pooled clips) and supersede earlier 3-seed values (75.6-82.0 %). Two caveats stand:

1. **The camera grouping is a proxy.** Unsafe-Net publishes no camera IDs; the eight "cameras" are a
   KMeans(8) clustering of static-background descriptors. LCO strength is bounded by that proxy.
2. **An earlier version of this abstract claimed "+58.7 pp over a scene-only baseline" under LCO. That
   is withdrawn** - the LCO "scene-only" baseline was a *constant* predictor (training global-majority
   class), not a scene model; its 16.5 % top-1 / 10.1 % macro is arithmetic, not a measurement. The
   figure quoted above is against a logistic regression on whole-frame appearance features fit on
   training cameras only, which actually transfers. See Ch7 §7.7 and brief.md S2.
