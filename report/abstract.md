# Abstract (draft — scene-robust reframe)

Automated recognition of unsafe worker behaviours from surveillance video is a core problem for
industrial occupational safety. The published Unsafe-Net system (Önal & Dandıl, 2024) reports 95.81 %
macro accuracy on a four-class unsafe-behaviour benchmark using a YOLO v4 per-frame behaviour detector
(trained on 2,262 behaviour-bounding-boxes) followed by a ConvLSTM temporal classifier. This thesis
re-implements that pipeline from scratch with two modernisations — a **YOLOv11s (extensible to YOLO26)
spatial encoder** replacing YOLO v4 and a **box-token spatial-relation Transformer** replacing ConvLSTM —
but under a different supervision philosophy: rather than labelling behaviours directly, we annotate an
**eight-class multi-entity ontology** (people by vest colour, walkway/restricted zones, panel and forklift
states) and let the Transformer infer behaviour from the *spatial relations* between detected entities.

On the same 4-class, 64-clip test protocol, the entity detector reaches 0.969 mAP@50 and the behaviour
classifier reaches 54/64 (84.4 % top-1, 88.4 % macro) — matching the no-bbox baseline on macro. The
thesis's central contribution, however, is an **evaluation-rigor finding: the Unsafe-Net benchmark is
scene-confounded.** With eight fixed cameras and strong camera↔behaviour correlation, a *scene-only*
baseline that predicts behaviour from the camera identity alone scores 81.2 %, and a whole-frame
classifier (the paper's scene-dependent style) scores an identical 81.2 % — i.e. same-distribution
accuracy largely measures *which camera*, not the behaviour. Under a **leave-camera-out (scene-robust)
protocol** a trained, transferable scene-appearance baseline reaches only 64.4 % macro while the
proposed entity-relation model retains **81.2 % macro (77.7 % top-1)** — an **honest margin of +16.9 pp
macro, 95 % bootstrap CI [+12.5, +21.2]** (n_boot = 5 000 over 516 pooled clips). The per-class
pattern is *consistent with* the margin being carried by the spatially-defined classes (OpenedPanel
88.7 vs 88.0 and CarryOverload 100 vs 100 tie; WalkwayViolation 70.5 vs 17.6 and UnauthorizedIntervention
65.7 vs 51.9), but we deliberately do **not** rest the claim on those four numbers: a stability study
(§7.8) finds LCO per-class recall varies by up to ~20 pp under perturbations that leave the aggregate
margin unchanged. The evidential claim is the aggregate margin; the per-class breakdown is reported as
descriptive support only. We therefore argue that same-distribution
top-1 on this benchmark is substantially scene-inflated, and that the entity-relation formulation — not
any same-distribution top-1 — is the scientifically sound, cross-camera-deployable approach a real
factory-safety system requires. We do **not** claim the paper's own model would fail cross-camera: its
weights and per-clip predictions are unreleased, so it cannot be run under this protocol. The claim is
about the *metric*, not about their system. The work is supported by a full multi-lens evaluation (macro-F1,
balanced accuracy, Cohen's κ = 0.775, MCC = 0.785, per-class miss-rate and cost-sensitive safety
metrics, one-vs-rest ROC-AUC = 0.951, PR-AUC = 0.880, ECE = 0.097, and BCa/Wilson/McNemar statistics)
and an edge-deployment characterisation (YOLO26 + Jetson TensorRT **FP16** — INT8 was never built for
P3, no calibration set exists for it; the measured verdict is that P3's cadence is
decided by the device's clock state: with the clocks pinned it reaches 3.23 dec/s at median crop
load, clearing the 3.0 dec/s contract by 7.5 %, while under the default dynamic-frequency governor
it reaches only 2.73 dec/s and misses the contract at every crop load; above the median it misses
in both conditions, 2.88-2.76 dec/s pinned). Every figure quoted here comes
from the **same predictor** — the val-selected blind rule scoring 54/64 — including the probability-based
ones; earlier drafts had to compute those on a seed-averaged ensemble (52/64) because the blind seed's
softmax had not been persisted, and that mismatch is now resolved (§7.5).

A **second lane** carries this thesis's leak-free accuracy, and it is the lane that was actually
deployed. On **iSafetyBench** (1,100 clips, binary hazard/normal — a different corpus and a different
task from the 4-class Unsafe-Net protocol above) a frozen CLIP ViT-B/32 tower feeding a small
temporal Transformer head (B1, 201,090 parameters x 3 seeds) scores **84.2 % balanced accuracy under
LSO-1100**: ten folds grouped by *source video*, so no source can supply clips to both sides of a
split. That head is the one exported to TensorRT FP16 and measured on an Orin Nano Super 8 GB
(§5.6.9), where the **composed** loop — decode, preprocess, tower, head, nothing omitted — sustains
**3.45 decisions/s**, and where the detector is not in the class decision at all: YOLO supplies
localisation and is timed separately, never summed into the decision cost. 84.2 % balanced accuracy
over 1,100 source-grouped clips and the 84.4 % top-1 over 64 same-distribution clips are different
metrics, under different protocols, on different data; they are **not** comparable, and this thesis
never averages them or quotes the larger of the two.

---

## Honest headline (one paragraph)

This thesis does **not** beat the paper's 95.81 % on the same-distribution top-1 metric, and it is candid
that it should not try to: that metric, on this benchmark, is scene-confounded — a model that ignores the
video and guesses the camera scores 81.2 %, and the paper-style whole-frame classifier does no better. What
the thesis **does** deliver is (a) a modern, reproducible **YOLOv11/Transformer entity-relation pipeline**
(54/64, 88.4 % macro, above the no-bbox baseline on macro) whose supervision is placed on interpretable
entities rather than behaviours; (b) a **rigorous scene-robustness analysis** showing that under
leave-camera-out evaluation the entity-relation model generalises substantially better than a trained
scene-appearance baseline (81.2 % vs 64.4 % macro, +16.9 pp, CI [+12.5, +21.2]), because it reasons over
person×zone geometry instead of memorising scenes; and (c) the resulting **critique that same-distribution top-1 on this
benchmark rewards scene memorisation**, together with a fairer, cross-camera evaluation protocol for
the task. The contribution is an honest, scene-robust,
deployable method plus a correction to how this benchmark should be measured — a stronger and more
defensible result than a scene-inflated accuracy number.


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
