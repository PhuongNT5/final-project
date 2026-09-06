# S3 Frame-Behaviour Classifier — Research Notes
**Date:** 2026-08-14  
**Context:** Industrial unsafe-behaviour video classification (4 classes; 452 train / 64 test clips; fixed-camera factory).
**Question:** Should we build S3 — a whole-frame per-frame behaviour classifier (YOLO26/YOLO11-cls or ResNet/ConvNeXt), aggregate per clip, replicating the paper's per-frame supervision?

---

## 1. Scene/Background Bias in Action Recognition — Severity & Mitigation

### How severe is it?

Scene/background bias is **one of the most documented failure modes in video action recognition**. The literature is unambiguous:

- **Li et al., ECCV 2018 — RESOUND** ([link](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Yingwei_Li_RESOUND_Towards_Action_ECCV_2018_paper.pdf)): demonstrated that mainstream datasets (Kinetics, Hollywood2, ActivityNet) are dominated by scene/object representation bias, not motion. Models effectively learn "what room is this in?" not "what is the person doing?".
- **Choi et al., NeurIPS 2019 — SDN** ([link](https://proceedings.neurips.cc/paper/2019/hash/ab817c9349cf9c4f6877e1894a1faa00-Abstract.html)): "Why Can't I Dance in the Mall?" quantified scene bias via an adversarial scene-label loss. Models trained on HMDB/UCF learn strong scene-type predictors as a free shortcut. Their Scene Decoupling Network (SDN) uses (1) adversarial scene loss and (2) human-mask confusion loss to disentangle action from scene.
- **Li et al., ICCV 2023 — StillMix / SCUBA** ([link](https://arxiv.org/pdf/2211.12883)): introduced SCUBA and SCUFO synthetic benchmarks. Static bias exists in *both* background and foreground (actor clothing). Their StillMix identifies bias-inducing frames via a 2D reference network and mixes them with other videos during training. Tested 74 trained models on Kinetics-400 — **all exhibited stronger background bias than motion bias**.
- **HAT toolkit analysis** (multiple 2022-2024 works): across 74 Kinetics models, all focus more on scene background than human motion. Background-only Kinetics-400 models (person inpainted out) achieve approximately **50% top-1 accuracy** — substantially above random chance on a 400-class problem (baseline ≈ 0.25%), proving strong scene encoding.
- **"Seeing Beyond the Scene"** (arxiv 2512.17953, 2024): systematic analysis across classification models, CLIP-pretrained models, and Video LLMs — all default to background reasoning. Adding segmented human input reduces background bias by 3.78 pp for classifiers; prompt engineering reduces it 9.85 pp for VLLMs.
- **SOAR (2023)**: ([link](https://arxiv.org/pdf/2309.01265)) open-set action recognition with adversarial scene reconstruction; without debiasing, scene distribution shift decimates open-set performance.

### Detection methods for scene bias

1. **Scene-only baseline:** train a model on background-masked (actor removed) or scene-category-labelled frames; if accuracy >> chance, bias is present.
2. **Leave-Camera-Out (LCO) / Cross-Scene split:** train on all cameras/views except one, test on left-out; if accuracy collapses, scene memorisation is confirmed.
3. **LOSO (Leave-One-Scene-Out):** validated in surveillance literature ("Out the Window" CVPR 2019); shows that networks overfit camera geometry when train/val share the same scene.
4. **HAT synthetic manipulation:** replace background with uniform colour; measure accuracy drop.
5. **Actor-crop ablation:** compare whole-frame vs. tightly-cropped-actor accuracy; if whole-frame >> actor-crop, scene is doing the work.

### Mitigation methods

| Method | Mechanism | Cost |
|--------|-----------|------|
| Actor cropping (tight ROI) | Remove scene entirely; classifier sees only person region | Need person detector |
| Background suppression (inpainting/mask) | Replace scene with noise/mean; ActorCutMix swaps backgrounds across clips | Need segmentation |
| Adversarial scene loss (SDN, SOAR, DEVIAS) | Add auxiliary loss penalising scene-type prediction | Code + hypertuning |
| StillMix / augmentation | Mix bias-inducing frames across classes during training | ~Low overhead |
| Leave-Camera-Out eval | Eval protocol, not training change | Cheap (resplit data) |

### For a fixed-camera industrial dataset

This is the **worst-case scenario for scene bias**. Reasons:
- Each camera view is a unique scene: fixed lighting, fixed background objects, fixed floor markings, fixed spatial layout.
- If certain behaviours are spatially constrained (e.g., "WalkwayViolation" always happens near the walkway visible in camera C1), the classifier only needs to learn "is there a walkway?" — a scene feature, not a behaviour feature.
- The Unsafe-Net dataset has 8 camera locations across 4 behaviour classes — if the mapping camera↔class is even partially deterministic, scene-bias accuracy approaches 100%.

**Conclusion for Q1:** Scene/background bias is a **critical scientific threat** to whole-frame per-frame classifiers, especially on fixed-camera industrial data. A classifier trained this way is not scientifically valid without a scene-robust evaluation protocol (LCO / cross-scene split). Standard accuracy on the same-distribution test set is not a trustworthy metric for behaviour understanding.

---

## 2. Does the Source Paper (Önal & Dandıl, Unsafe-Net) Likely Suffer Scene Bias?

### Assessment

**Yes, very likely — and substantially.**

Arguments:
1. **Fixed-camera factory dataset, few unique scenes.** The Unsafe-Net dataset has a small number of camera positions in a single industrial facility. If class-to-camera-view correlation exists (which is common in safety datasets where e.g., a walk-in-walkway zone is only filmed by specific cameras), per-frame classification has a trivial scene shortcut.
2. **YOLOv4 trained as behaviour-as-object detector (per-frame bboxes, clip label propagated).** The behaviour bbox likely occupies the entire zone: a "WalkwayViolation" bbox encloses the striped walkway floor region, an "OpenedPanel" bbox encloses the electrical panel — both of which are *scene features* as much as *behaviour features*. Even at 95.66% mAP, this detector could be learning "where is the panel?" not "is the person interacting unsafely?".
3. **ConvLSTM trained on YOLO-filtered frames.** Frames selected by the detector carry strong scene context. If the detector memorised view-behaviour correlations, the ConvLSTM receives pre-filtered scene-biased frames.
4. **No cross-scene / LCO evaluation reported.** The paper evaluates on a held-out clip split from the same dataset. No ablation removing the scene context is described, no camera-holdout evaluation, no background-only baseline. This is a gap in their evaluation rigour.
5. **Small dataset (452 train clips, 8-class hierarchy collapsing to 4 behaviours).** With this size, overfitting to scene layout is highly plausible. The jump from 89.06% (ConvLSTM alone) to 95.81% (with YOLO frame-selection) can partly be explained by: the detector excluding ambiguous multi-behaviour frames AND by it selecting frames most visually diagnostic of the *scene* of that behaviour.

### What a fair critique / re-evaluation would look like

1. **Camera-holdout ablation:** train on clips from cameras C1...(k-1), test on camera Ck. If 95.81% drops toward 60-70%, scene memorisation explains the headline number.
2. **Scene-only baseline:** train a classifier on background-masked frames (actor removed via segmentation); measure accuracy.
3. **Actor-crop-only baseline:** tight-crop the detected person from the YOLO bbox, classify from person-only patch; compare to whole-frame.
4. **Dataset bias audit:** compute camera-ID vs class distribution; even partial confounding (e.g., 80% of OpenedPanel clips from camera C2) explains high accuracy without generalisation.

**Honest framing for thesis:** The paper's 95.81% is plausible but likely inflated by scene-view memorisation on a fixed-camera dataset. It is not replicable on a camera it has never seen. Our entity-ontology Transformer approach, while lower accuracy, explicitly encodes behaviour-discriminative spatial relations (foot-in-walkway geometry, etc.) rather than scene appearance — which is a stronger generalisation argument. The gap to 95.81% is partly **supervision-placement** (our diagnosis from gap-analysis.md) and partly **scene-bias advantage** for the paper.

---

## 3. Frame-Level Classification + Temporal Aggregation — How Well Does It Work?

### Evidence

**TSN (Temporal Segment Networks, Wang et al., ECCV 2016 / IEEE TPAMI 2019)**  ([link](https://arxiv.org/abs/1705.02953)): The foundational result. Sample K sparse frames, extract CNN features per frame, average logits. Achieves 94.9% UCF-101, 71.0% HMDB-51 — competitive with video-specific models at the time. The per-frame → average aggregation is a strong baseline.

**SIFAR (Fan et al., ICLR 2022)** ([link](https://openreview.net/forum?id=qhkFX-HLuHV)): Rearranges K frames into a 3×3 or 4×4 "super-image" fed to a standard image classifier. Achieves competitive accuracy on Kinetics-400 and SSv2. Demonstrates that **for appearance-based actions (K-400), per-frame image classifiers are near-SOTA**; for motion-heavy actions (SSv2), they fall behind.

**When per-frame → aggregate is competitive:**
- Actions are scene/appearance-defined (e.g., "playing golf" = golf course scene; or "OpenedPanel" = panel open in frame)
- Temporal dynamics are slow or repetitive (not fast sports)
- Small datasets where video-specific temporal models overfit
- Fixed-camera industrial where a single frame is diagnostic

**When it falls short:**
- Fine-grained motion-based actions (chopping vs stirring — SSv2, Diving48)
- Actions defined by temporal *order* (e.g., "pick up then carry" vs "carry then put down")
- Low inter-frame information redundancy

**For Unsafe-Net specifically:** Given that CarryOverload, OpenedPanel, WalkwayViolation are often detectable from a single frame (overloaded carriage visible, panel open visible, person in striped zone visible), per-frame → majority/mean aggregation is expected to be highly accurate — but for the wrong reason (scene features, not behaviour dynamics). UnauthorizedIntervention may require more temporal context (person *moving toward* machine, not just standing near it).

**Bottom line:** For scene/appearance-dominated datasets, per-frame aggregation is competitive with clip models. The accuracy number will be high. But it will reflect scene memorisation, not generalisation.

---

## 4. YOLO26 / YOLO11 Classification Mode — Viability

### Confirmed: Both have classification variants

- **YOLO11-cls**: yolo11n-cls through yolo11x-cls, pretrained on ImageNet. Top-1 on ImageNet: 75.4% (s) → 78.3% (l). Docs: [Ultralytics classify task](https://docs.ultralytics.com/tasks/classify).
- **YOLO26-cls**: yolo26n-cls through yolo26x-cls, pretrained on ImageNet. Top-1 on ImageNet: 71.4% (n) → 79.9% (x). Released Sept 2025. Available on Replicate, HuggingFace, Ultralytics Platform. Docs: [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26).

### Training on folder-per-class dataset (standard ImageNet layout)

```
dataset/
├── train/
│   ├── WalkwayViolation/  [frames from train clips]
│   ├── UnauthorizedIntervention/
│   ├── OpenedPanel/
│   └── CarryOverload/
└── val/
    ├── WalkwayViolation/  [frames from val clips]
    └── ...
```

```python
from ultralytics import YOLO
model = YOLO("yolo26s-cls.pt")   # or yolo11s-cls.pt
results = model.train(data="path/to/dataset", epochs=50, imgsz=224, task="classify")
```

CLI: `yolo classify train model=yolo26s-cls.pt data=path/to/dataset epochs=50`

### YOLO26-cls vs standard ImageNet CNNs

| Model | ImageNet Top-1 | Size | Inference (ms, CPU) | Notes |
|-------|---------------|------|---------------------|-------|
| YOLO26s-cls | 76.0% | ~10M params | fast | DFL-free, end-to-end, simpler head |
| YOLO11s-cls | 75.4% | ~10M params | fast | Established, well-documented |
| ResNet-50 | ~76.1% | 25M params | moderate | Standard baseline, well-understood |
| EfficientNet-B2 | ~80.3% | 9M params | fast | Strong small-dataset performance |
| ConvNeXt-Tiny | ~82.1% | 28M params | moderate | Best pure-CNN accuracy, stable training |

**Trade-offs for a 4-class frame classifier on this dataset:**

- YOLO26-cls / YOLO11-cls **advantages**: unified Ultralytics ecosystem (same code as detection), COCO-semantic pretrain for YOLO backbone if fine-tuning detection head, DFL-free head is faster, easy CLI. For this use-case (4-class custom), transfer from ImageNet cls weights is what matters — the YOLO architecture's detection-head improvements are irrelevant.
- YOLO26-cls / YOLO11-cls **disadvantages**: ImageNet top-1 of 76% is lower than ConvNeXt-Tiny (82%) or EfficientNet-B2 (80%). The classification head is a simple linear layer; no multi-scale aggregation. For fine-grained behaviour features, a richer pretrained backbone (ConvNeXt, ViT-S) may transfer better.
- **For 4-class, ~452×16=7,232 training frames**: with this few images per class (~1,800 frames/class), the backbone's pretrained representations dominate; a stronger pretrain (ConvNeXt-Tiny from 82% ImageNet) will likely outperform a weaker one. The practical difference may be small (2-3 pp) for such a coarse 4-class problem, but ConvNeXt-Tiny / EfficientNet-B2 are safer bets if maximising accuracy.
- **For thesis expediency**: YOLO11s-cls is already in your codebase; lowest switching cost. Accuracy will be adequate. Use it unless you have time to benchmark alternatives.

**Recommendation**: YOLO11s-cls (lowest friction) or EfficientNet-B0/B2 (strongest small-dataset transfer). YOLO26-cls adds no meaningful advantage over YOLO11-cls for a 4-class classification task; the DFL-free improvements are detection-only.

---

## 5. Leakage Discipline for S3

### What is guaranteed safe

- **Test clip frames never in training**: if clip-level train/test split is already fixed and you sample frames from train clips only, no test-clip contamination. This is the primary leakage risk and you confirm it is controlled.
- **Per-frame label = clip label**: this is *not* leakage — it is a label-smoothing/noisy-label situation. Every frame in a clip carries the clip's behaviour label. This is the paper's exact design; it assumes that most frames within a clip are representative of that behaviour (which may or may not be true — see §1 above).

### What requires care

1. **Val split must be at clip level, not frame level.** If you sample frames from all clips and then randomly 80/20 split *frames*, you risk having frames from the same clip in both train and val (temporal leakage). Frames within a clip are highly correlated. Split by clip, not by frame.
2. **Frame sampling strategy**: if you sample all frames from training clips, you can inflate the effective training set (e.g., 32 frames × 452 clips = 14,464 frames) but must ensure val clips are never sampled into training frames. Stratify by clip ID at the split boundary.
3. **Checkpoint/config selection using val**: the val split must be carved from the train clip split *before* any frame sampling. Using test clips for checkpoint selection is the main historical leakage in this project (§4.7.7 in thesis); S3 must enforce blind test protocol.
4. **Aggregation at test time**: when aggregating per-frame predictions to clip label (majority/mean), this is not leakage — it is the inference procedure. The key is that the aggregation function (majority vs mean) is chosen on the val set, not the test set.
5. **Subtle risk — background shortcut as "leakage substitute"**: even without data leakage, scene memorisation gives inflated accuracy that does not generalise. This is not leakage in the technical sense, but produces the same misleading result.

**Summary**: No subtle leakage in the S3 design *if* the train/val split is done at clip level. The noisy-label aspect (clip label propagated to all frames) is standard practice (TSN, paper's approach) and not a methodological error — it is just a weak supervision assumption.

---

## 6. Recommendation — Is S3 Worth Building?

### Bottom line

**Yes, build S3 — but with honest framing and a scene-robust eval protocol.**

### Why build it

1. **Directly closes the comparison to the paper** on an apples-to-apples supervised training signal (clip label → per-frame supervision → aggregate). This is the strongest "replication" experiment.
2. **Fast to implement**: YOLO11s-cls or EfficientNet-B2, folder-per-class from existing clip-level splits, 50 epochs, standard classification training. Under 1 day of work.
3. **Likely to reach 57-62/64 on same-distribution test** (matching or exceeding paper's no-bbox baseline of 57/64 and approaching full 62/64), since it directly exploits the same scene-correlated signals the paper's YOLOv4 exploited.
4. **Provides strong ablation evidence**: comparing S3 (scene-biased whole-frame) vs our entity-Transformer (scene-agnostic relational) on the standard test *and* on the LCO protocol quantifies exactly how much of the accuracy gap is scene-bias vs genuine behaviour understanding.

### Sound design

**Backbone:** YOLO11s-cls (expediency) or EfficientNet-B2 (stronger ImageNet pretrain for small dataset). YOLO26-cls is marginally better than YOLO11-cls; not worth the migration cost.

**Training:**
- Frame sampling: sample N=8 or 16 frames per clip uniformly (consistent with T=16 used elsewhere in the project)
- Clip-level train/val split: 80/20 stratified, never frame-level split
- Label propagation: clip label → all frames from that clip
- Augmentation: RandAugment or standard ImageNet augmentation; no temporal augmentation
- Epochs: 50, patience 10; no LR tricks needed for 4-class
- Class-weighted loss for CarryOverload (small class: 8 test clips)

**Scene-bias mitigation options (pick at least one for thesis rigour):**
1. **(Recommended, cheapest)** Report both: (a) standard accuracy and (b) Leave-Camera-Out accuracy (train on cameras C1...(k-1), test on Ck). If dataset camera metadata is available — check the Unsafe-Net paper for camera IDs per clip.
2. **(Moderate cost)** Actor-crop baseline: run YOLOv11 person detector, extract tight person crop per frame, train a parallel classifier on crops only. Compare whole-frame vs actor-crop accuracy. If actor-crop << whole-frame, scene is doing the work.
3. **(Dataset audit, free)** Compute camera-ID × class co-occurrence matrix. If correlation coefficient > 0.5, the whole-frame classifier is memorising views.

**Evaluation protocol:**
- Primary metric: clip-level top-1 and macro accuracy on blind test set (majority vote over per-frame predictions)
- Secondary metric: per-frame accuracy (as confirmation that frame-level supervision works)
- **Critical add-on for thesis**: LCO accuracy or actor-crop accuracy; report BOTH in the same table and acknowledge the scene-bias caveat explicitly
- Aggregation: report both majority vote and mean-pooling of softmax; they should be similar for this dataset

### Honest framing for thesis

S3 is **a controlled experiment to quantify the scene-bias advantage**, not a claim of superior behaviour understanding. The correct framing is:

> "S3 replicates the paper's per-frame supervision strategy. On the standard same-distribution test, S3 reaches X/64 (Y%), comparable to the paper's no-bbox baseline (57/64) — confirming that per-frame clip-label supervision is sufficient for *this distribution*. However, under the Leave-Camera-Out protocol, S3 accuracy drops to Z/64, while our entity-Transformer retains W/64 — demonstrating that whole-frame classification exploits scene memorisation rather than generalising behaviour understanding."

If camera metadata is unavailable: substitute actor-crop ablation as the scene-bias quantification.

### If the scene-bias risk is high but you can't do LCO

Still build S3. Report the standard test accuracy. Add the actor-crop ablation (cheap with YOLO person detector). Add a clear caveat: "The Unsafe-Net dataset is fixed-camera with limited view diversity; whole-frame classifiers likely exploit scene-view correlations. The reported accuracy should be interpreted as an upper bound under this distribution, not as view-invariant behaviour recognition."

This is the honest framing. It strengthens the thesis by demonstrating *awareness* of the limitation — which the source paper does not do.

---

## Quick Reference: S3 Implementation Checklist

- [ ] Verify camera-ID × class distribution (bias audit)
- [ ] Sample N=16 frames per clip from train clips, clip-level 80/20 val split
- [ ] Folder structure: `train/{class}/frame_{clip_id}_{frame_idx}.jpg`
- [ ] Train YOLO11s-cls (or EfficientNet-B2) on folder dataset, 50 epochs
- [ ] Checkpoint on val macro accuracy (not test)
- [ ] At test time: majority vote over per-frame predictions per clip
- [ ] Record both majority-vote and mean-softmax accuracy
- [ ] Run actor-crop parallel experiment (YOLO person detector → tight crop → classify)
- [ ] If camera metadata available: LCO eval
- [ ] Report in thesis: standard test + scene-bias ablation in same table

---

## Sources

- [RESOUND (Li et al., ECCV 2018)](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Yingwei_Li_RESOUND_Towards_Action_ECCV_2018_paper.pdf)
- [SDN / Why Can't I Dance (Choi et al., NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/ab817c9349cf9c4f6877e1894a1faa00-Abstract.html)
- [SDN GitHub](https://github.com/vt-vl-lab/SDN)
- [StillMix / SCUBA (Li et al., ICCV 2023)](https://arxiv.org/pdf/2211.12883)
- [Seeing Beyond the Scene (arXiv 2512.17953)](https://arxiv.org/abs/2512.17953)
- [SOAR: Scene-debiasing Open-set AR (2023)](https://arxiv.org/pdf/2309.01265)
- [DEVIAS: Disentangled Video AR](https://link.springer.com/chapter/10.1007/978-3-031-73113-6_25)
- [SIFAR: Image Classifier for AR (Fan et al., ICLR 2022)](https://openreview.net/forum?id=qhkFX-HLuHV)
- [TSN (Wang et al., ECCV 2016 / TPAMI 2019)](https://arxiv.org/abs/1705.02953)
- [Mitigating Representation Bias (arXiv 2209.09393)](https://arxiv.org/pdf/2209.09393)
- [Ultralytics YOLO11 classify task](https://docs.ultralytics.com/tasks/classify)
- [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26)
- [YOLO26-cls on Replicate](https://replicate.com/ultralytics/yolo26-cls)
- [Unsafe-Net paper (Önal & Dandıl, 2025)](https://www.researchgate.net/publication/380344607_Unsafe-Net_YOLO_v4_and_ConvLSTM_based_computer_vision_system_for_real-time_detection_of_unsafe_behaviours_in_workplace)
- [Find the Leak, Fix the Split (video-derived dataset leakage, 2024)](https://arxiv.org/abs/2511.13944)

---

## Unresolved Questions

1. **Camera-ID metadata availability**: does the Unsafe-Net dataset label which clips came from which camera? If yes, LCO eval is possible and should be done. If no, actor-crop ablation is the substitute.
2. **Frame diagnosticity per class**: are WalkwayViolation/OpenedPanel reliably visible in a single frame, or do some clips require temporal context? A manual review of 10 clips per class would inform whether majority-vote is appropriate or attention-weighted aggregation is needed.
3. **Exact YOLO26-cls vs YOLO11-cls accuracy delta on this dataset**: both have similar ImageNet pretrain accuracy (~75-76% top-1 for s-variant); the gap on a 4-class industrial dataset is likely < 2 pp. If time allows, run both and report.
4. **Whether the paper's 2,262 bbox-annotated frames are available or derivable**: if those frames are available (as the gap-analysis.md suggests they exist in `runs/detect/`), S3 can be trained on *those specific frames only* (most diagnostic) rather than all frames from all clips — closer to the paper's exact training set.
