# Chapter 2 — Related Work

> ### Pipeline attribution — read before citing any number in this chapter
>
> This project evolved through **three non-interchangeable pipelines** (full map:
> `docs/system-architecture.md`). Chapters were written at different points in that evolution, so
> their headline accuracies are **not** competing estimates of one system — they describe different
> systems, and mixing them is the single most common error this project has made internally.
>
> | | Pipeline | Best cited | Blind/honest | Protocol |
> |---|---|---|---|---|
> | Ch1, Ch3, Ch4 | **P1** frozen-COCO YOLO -> multi-stream Transformer | 56/64 (87.5 %) | **53/64** (82.8 %) | same-dist, N=64 |
> | Ch5 | **P1** as exported to Jetson | 55/64 (TRT engine, as recorded in `benchmarks-jetson-*.json`) | **53/64** | same-dist, N=64 |
> | Ch7 + Abstract | **P3** entity detector + spatial-relation Transformer | **54/64** (84.4 % **top-1**, 88.4 % macro) | 54/64 | same-dist N=64, **plus k-fold and LCO at N=516** |
> | Ch5 §5.6.9 | **R48** whole-frame CLIP ViT-B/32 -> B1 temporal Transformer — the engine actually on the device | **84.2 % balanced accuracy** (B1, N=1,100) | same number: the protocol cannot leak a source, so there is no separate honest column | **P-D LSO-1100**, folds grouped by source video, N=1,100 |
>
> **The headline is the R48 lane: CLIP + temporal Transformer under LSO-1100.** It is the only
> accuracy in this thesis measured on all 1,100 clips under a protocol that cannot put one source
> video in both the train and the evaluation fold, and it is the model that was exported, deployed
> and measured on the Jetson (§5.6.9). YOLO's role in that lane is **localisation** — the system
> must say *where*, not only *whether* — and localisation latency is timed separately, never summed
> into the class decision. P3's 54/64 is the entity-relation lane's result at N=64 and is what Ch7
> and the abstract report; it keeps its own evidence beyond N=64 (k-fold and LCO at N=516). P1's
> 56/64 is **test-selected** (the ensemble was chosen using the test clips) and must never be
> presented as the result; its honest blind counterpart is 53/64. Chapters 1-6 are retained as the
> experimental record of how the design was arrived at, not as the final claim.
>
> **Caution — "84.4 %" denotes two different quantities here. Never write it bare.** 84.4 %
> *top-1* (P3, blind same-dist split, N=64) is Ch7's figure. 84.4 % *balanced accuracy* (B1w,
> LSO-1100, N=1,100) is the R48 lane's best head — and B1w is **not** the head on the device. The
> deployed engine is B1 (201,090 params x 3 seeds), whose own figure is **84.2 %**; B1 was chosen on
> the pre-registered *edge* criterion, smallest model, not on accuracy, so its own number is the one
> to quote and never max(B1, B1w). §5.6.9 quotes 84.2 % throughout for that reason
> (`plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-10-01-two-different-84-4-percent.md`).
>
> **"84.2 %" likewise denotes two different quantities. Never write it bare either.** 84.2 %
> *balanced accuracy* (B1, LSO-1100, N=1,100, binary) is the deployed figure above. 84.2 % *macro*
> is P3's k-fold mean over 516 ESTA clips on the 4-class task (Ch7 §7.7.1). Different corpus,
> different task, different metric: the two must never be compared, summed, or carried into one
> sentence without their qualifiers.

This chapter surveys three threads that frame the contribution: (2.1) the
YOLO family of real-time detectors, (2.2) Transformer and Transformer-hybrid
architectures for video classification, and (2.3) vision-based industrial
safety-monitoring datasets and systems. Section 2.4 summarizes the
positioning of this thesis.

## 2.1 Real-Time Object Detection: the YOLO Family

Real-time object detection originated with Redmon et al.'s **YOLO** [1],
which framed detection as a single-stage regression over a dense spatial
grid and achieved order-of-magnitude speed-ups over the Fast R-CNN lineage
while retaining competitive mAP. Subsequent versions progressively
integrated ideas from the broader detection literature:

- **YOLOv3** [2] introduced FPN-style multi-scale prediction heads and a
  Darknet-53 backbone, lifting small-object recall.
- **YOLOv4** [3] — the detector used by the Unsafe-Net paper [17] — added
  CSPDarknet, Mish activation, Mosaic augmentation, and CIoU loss,
  balancing accuracy and speed on commodity GPUs.
- **YOLOv5–v8** [4, 5] consolidated the architecture around PyTorch with
  anchor-free heads and improved training recipes.
- **YOLOv11** [6] — the backbone used in this thesis — introduced the C3k2
  block, SPPF spatial pooling, and a streamlined neck that reduces
  parameter count while improving throughput on CPU.
- **YOLO-World** [7] extends YOLOv8 with open-vocabulary text prompting
  via contrastive CLIP-style alignment, enabling zero-shot detection of
  novel categories (e.g., "person", "forklift") without retraining. This
  thesis uses YOLO-World (v2) as one of the backbone variants in Chapter 4.

An alternative real-time-detection thread is **RT-DETR** [8], a
Transformer-based detector that removes NMS and matches YOLO speeds on
T4 GPU. The proposal (Section 3 of PROPOSAL.txt) initially anticipated
RT-DETR as the temporal encoder; the final methodology (Chapter 3)
substitutes a lighter pure-Transformer encoder on top of YOLOv11 spatial
features because the temporal sequence is already a low-rank (T × 16 × 512)
token grid for which a DETR-style cross-attention decoder is unnecessary.

## 2.2 Temporal Modelling for Video Classification

Video behaviour classification adds a temporal axis that static detectors
cannot capture. Three model families are relevant:

**Two-stream / recurrent.** Early work combined CNN appearance features
with optical-flow motion features [9]; later work replaced the flow stream
with recurrent aggregation. **ConvLSTM** [10] — a convolutional variant of
LSTM that preserves spatial structure in the hidden state — is the
temporal backbone of the Unsafe-Net paper [17] and the direct baseline of
this thesis. ConvLSTM achieves **85.78 % macro / 89.06 % top-1 (57/64)**
on the 4-class Unsafe-Net test set (Table 7 macro and Table 5 confusion of
the paper); coupled with YOLO v4 detections the joint system reaches
**95.81 % macro / 96.88 % top-1 (62/64)** (Tables 6/7). The "Overall (%)"
column in [17]'s Table 7 is the macro mean of per-class recalls, *not*
top-1 — a distinction this thesis preserves throughout.

**3D convolution.** I3D [11] and SlowFast [12] inflate 2D convolutions
along the temporal axis and stream multi-rate pathways. These are accurate
on Kinetics-scale datasets but have high FLOPs per clip and have not been
benchmarked on Unsafe-Net.

**Video Transformers.** **TimeSformer** [13] decouples spatial and
temporal attention; **ViViT** [14] enumerates factorized attention
variants; **VideoMAE** [15] applies masked autoencoding pre-training on
Kinetics/Sthv2 and transfers strongly to small downstream datasets. This
thesis evaluates VideoMAE-base as a frozen-backbone baseline in the
broader 10-track sweep summarised in §4.5 and finds the same ~55/64
ceiling as the YOLOv11 + Transformer stack despite VideoMAE's 17×
larger parameter count (86 M vs 5 M) — evidence that the ceiling is
governed by the training signal rather than by representational
capacity.

**YOLO + Transformer hybrids.** Prior work combining detector features
with Transformer temporal encoders is sparse and typically task-specific
(traffic incident detection, action segmentation). No prior work applies
this combination to industrial unsafe-behaviour classification, which is
the empirical-study scope of this thesis.

**Weakly-supervised video classification — a route the thesis does *not*
take.** A parallel literature on **weakly-supervised video anomaly
detection (WS-VAD)** trains video-level classifiers using only
clip-level (not per-frame) labels — the same supervision regime as
this thesis. The canonical baseline is **Sultani et al. (2018)** [22]
on UCF-Crime, which formulates anomaly localisation as a
multiple-instance-learning (MIL) ranking problem over 32 segments per
video and reports 75.4 % AUC. Subsequent work has narrowed the gap
to fully-supervised baselines: **RTFM** (Tian et al., 2021) [23]
introduces robust temporal-feature-magnitude learning and reports
84.30 % AUC on UCF-Crime; **MSL** (Li et al., 2022) [24] adds a
self-training Transformer over multi-sequence inputs and pushes to
85.62 %; **VadCLIP** (Tang et al., 2024) [25] adapts CLIP visual
features and a frozen text encoder to WS-VAD without re-training, a
zero-shot baseline relevant to this thesis's frozen-COCO design.

These methods are *not* used as the primary method in this thesis
because (a) they are formulated as binary normal-vs-anomaly
classifiers, not the 4-way categorical task of Unsafe-Net; (b) the
MIL formulation requires multiple positive video-level labels per
class, whereas Unsafe-Net assigns one label per clip; (c) the
WS-VAD evaluation protocol is AUC over per-frame anomaly scores,
not top-1/macro on clip labels. They are, however, the closest
related family to this thesis's frozen-feature regime and are the
basis of the cross-dataset zero-shot evaluation reported in Ch4
§4.5.4 — applying our frozen features to UCF-Crime tests whether
the zone-pattern collapse documented in Ch4 §4.8 also limits
transfer to a different surveillance domain.

## 2.3 Industrial Safety-Monitoring Benchmarks

Vision-based safety monitoring has historically been hampered by the
absence of public video benchmarks. Available resources include:

- **CHOKE-D** [16] — PPE detection, single-frame, bounding-box labelled.
  Not a video behaviour dataset.
- **MVTec AD / ShanghaiTech Campus** — anomaly detection benchmarks;
  unsupervised and not aligned with the multi-class behaviour-classification
  problem.
- **Unsafe-Net** (Önal & Dandıl, 2024) [17] — the benchmark used in this
  thesis. 691 video clips (1920×1080, 24 FPS) captured across 39 days from
  a single factory, filtered from 4000 hours of raw footage. Each clip
  carries a single behaviour label from 8 classes (4 safe / 4 unsafe).
  Train/test split: 566 train / 125 test. The 4-class unsafe subset used
  for headline evaluation contains 64 test clips: 32 WalkwayViol, 11
  UnauthorInt, 13 OpenedPanel, 8 CarryOverload. Licence: CC BY 4.0.

The Unsafe-Net paper [17] reports per-class YOLO v4 detection AP (Table 4
of the paper — AP 86.30–100% across 8 classes) and the combined YOLO +
ConvLSTM confusion matrix (Table 6 — 62/64 top-1 on the 4-class unsafe
subset). Importantly, the YOLO v4 detector in [17] is trained **with
per-frame bounding-box annotations** that encode behaviour classes as
object categories. These bounding-box annotations are not publicly
released, which constrains the methodology of this thesis to a
**frozen-backbone** regime (COCO or YOLO-World weights) plus
Transformer-head training. This data-availability constraint is the
direct cause of the 6-sample accuracy gap analyzed in Chapter 4.

## 2.4 Edge Deployment of Deep Video Models

Edge deployment combines hardware-aware compilation with model
compression:

- **ONNX Runtime** [18] provides a language-agnostic inference runtime
  with pluggable execution providers (CPU, CUDA, CoreML, TensorRT).
- **TensorRT** [19] compiles ONNX graphs into NVIDIA-optimized kernels
  with graph fusion, FP16/INT8 quantization, and dynamic batching,
  targeting Jetson and datacenter GPUs.
- **Dynamic quantization** (ONNX Runtime's `quantize_dynamic`) applies
  weight-only INT8 quantization without requiring a calibration dataset;
  it typically preserves accuracy within 1 % on Transformer architectures
  while reducing model size 3–4×. Chapter 5 of this thesis applies
  `quantize_dynamic` to the YOLOv11s backbone and all Transformer heads,
  reporting 156 → 40 MB compression (3.8×) with a 1-sample accuracy cost.
- **CoreML Execution Provider** [20] allows ONNX Runtime to dispatch
  supported subgraphs to the Apple Neural Engine on Apple Silicon. In the
  Chapter 5 measurements, the CoreML path is actually slower than the
  pure CPU path (50.6 ms vs 26.8 ms p50) because the Transformer heads
  fall back to CPU and incur cross-device transfer overhead — a common
  pitfall documented in the CoreML EP release notes.

Prior work on YOLO edge deployment (Jetson Nano / Xavier / Orin) reports
single-stream FPS in the 25–60 range for YOLOv5s–YOLOv8n at 640×640 [21];
Chapter 5 projects this thesis's pipeline to Jetson Orin Nano based on
published Orin throughput ratios rather than direct measurement on
hardware, which is deferred to future work.

## 2.5 Positioning of this Thesis

Relative to the three threads:

1. **vs. YOLO family.** This thesis does not propose a new detector; it
   reuses pretrained YOLOv11s / YOLO-World as frozen spatial encoders,
   substituting the per-frame class head with a video-level Transformer
   classifier.

2. **vs. video Transformers.** Unlike VideoMAE or TimeSformer, which
   operate on raw-frame patch tokens, this thesis operates on **YOLO
   spatial-feature tokens** (16 tokens × 512 channels per frame after
   4×4 pooling at layer 22). This compresses the input sequence by ~98 %
   vs VideoMAE while preserving the per-frame spatial grid required for
   object-aware reasoning, and is 17× smaller in parameter count at
   matched accuracy (VideoMAE-base baseline in §4.5).

3. **vs. Unsafe-Net paper.** The paper [17] trains a behaviour-aware
   detector using per-frame bounding-box annotations; this thesis
   operates under the constraint of unavailable bbox annotations and
   therefore relies on off-the-shelf COCO / YOLO-World detectors plus
   video-level training. The gap analysis (Chapter 4) establishes that
   this constraint — not the architecture — is the binding performance
   factor.

4. **vs. edge-deployment literature.** The deployment in Chapter 5 is
   end-to-end reproducible on a commodity laptop CPU (M3 Apple Silicon)
   with no GPU and no cloud dependency. The multi-stream measurements
   (N = 1..4 concurrent streams with a single shared backbone pool)
   operationalize the "multi-luồng" (multi-stream) requirement of
   Objective 3 in a way that is directly transferable to industrial
   multi-camera deployments.

---

## References

Per-chapter numbering; to be merged into a unified bibliography at final
thesis compilation.

[1] J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You Only Look Once:
    Unified, Real-Time Object Detection," *Proc. IEEE Conf. Computer
    Vision and Pattern Recognition (CVPR)*, 2016, pp. 779–788.
    arXiv:1506.02640. DOI: 10.1109/CVPR.2016.91.

[2] J. Redmon and A. Farhadi, "YOLOv3: An Incremental Improvement,"
    arXiv:1804.02767, 2018.

[3] A. Bochkovskiy, C.-Y. Wang, H.-Y. M. Liao, "YOLOv4: Optimal Speed and
    Accuracy of Object Detection," arXiv:2004.10934, 2020.

[4] G. Jocher et al., "YOLOv5 by Ultralytics," 2020. [Online]. Available:
    https://github.com/ultralytics/yolov5. DOI: 10.5281/zenodo.3908559.

[5] G. Jocher, A. Chaurasia, J. Qiu, "Ultralytics YOLOv8," 2023. [Online].
    Available: https://github.com/ultralytics/ultralytics.

[6] Ultralytics, "YOLO11 — Documentation," 2024. [Online]. Available:
    https://docs.ultralytics.com/models/yolo11/ (accessed 2026-04).

[7] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, Y. Shan, "YOLO-World:
    Real-Time Open-Vocabulary Object Detection," *Proc. IEEE/CVF Conf.
    Computer Vision and Pattern Recognition (CVPR)*, 2024, pp. 16901–
    16911. arXiv:2401.17270.

[8] Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, J. Chen,
    "DETRs Beat YOLOs on Real-time Object Detection," *Proc. IEEE/CVF
    Conf. Computer Vision and Pattern Recognition (CVPR)*, 2024, pp.
    16965–16974. arXiv:2304.08069.

[9] K. Simonyan and A. Zisserman, "Two-Stream Convolutional Networks for
    Action Recognition in Videos," *Advances in Neural Information
    Processing Systems (NeurIPS)*, vol. 27, 2014. arXiv:1406.2199.

[10] X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W.-K. Wong, W.-c. Woo,
     "Convolutional LSTM Network: A Machine Learning Approach for
     Precipitation Nowcasting," *Advances in Neural Information
     Processing Systems (NeurIPS)*, vol. 28, 2015. arXiv:1506.04214.

[11] J. Carreira and A. Zisserman, "Quo Vadis, Action Recognition? A New
     Model and the Kinetics Dataset," *Proc. IEEE Conf. Computer Vision
     and Pattern Recognition (CVPR)*, 2017, pp. 4724–4733.
     arXiv:1705.07750. DOI: 10.1109/CVPR.2017.502.

[12] C. Feichtenhofer, H. Fan, J. Malik, K. He, "SlowFast Networks for
     Video Recognition," *Proc. IEEE/CVF Int. Conf. Computer Vision
     (ICCV)*, 2019, pp. 6202–6211. arXiv:1812.03982.
     DOI: 10.1109/ICCV.2019.00630.

[13] G. Bertasius, H. Wang, L. Torresani, "Is Space-Time Attention All You
     Need for Video Understanding?," *Proc. 38th Int. Conf. Machine
     Learning (ICML)*, 2021, pp. 813–824. arXiv:2102.05095.

[14] A. Arnab, M. Dehghani, G. Heigold, C. Sun, M. Lučić, C. Schmid,
     "ViViT: A Video Vision Transformer," *Proc. IEEE/CVF Int. Conf.
     Computer Vision (ICCV)*, 2021, pp. 6816–6826. arXiv:2103.15691.
     DOI: 10.1109/ICCV48922.2021.00676.

[15] Z. Tong, Y. Song, J. Wang, L. Wang, "VideoMAE: Masked Autoencoders
     are Data-Efficient Learners for Self-Supervised Video Pre-Training,"
     *Advances in Neural Information Processing Systems (NeurIPS)*, vol.
     35, 2022, pp. 10078–10093. arXiv:2203.12602.

[16] N. D. Nath, A. H. Behzadan, S. G. Paal, "Deep learning for site
     safety: Real-time detection of personal protective equipment,"
     *Automation in Construction*, vol. 112, 103085, 2020.
     DOI: 10.1016/j.autcon.2020.103085.

[17] O. Önal and E. Dandıl, "Unsafe-Net: YOLO v4 and ConvLSTM based
     computer vision system for real-time detection of unsafe behaviours
     in workplace," *Multimedia Tools and Applications*, vol. 84, no. 29,
     pp. 34967–34993, 2024. DOI: 10.1007/s11042-024-19276-8.
     Dataset: O. Önal and E. Dandıl, "Video Dataset for Safe and Unsafe
     Behaviours," Mendeley Data, 2024, CC BY 4.0.
     DOI: 10.17632/xjmtb22pff.1.
     https://data.mendeley.com/datasets/xjmtb22pff/1

[18] Microsoft, "ONNX Runtime," 2024. [Online]. Available:
     https://onnxruntime.ai/docs/ (accessed 2026-04).

[19] NVIDIA, "TensorRT Developer Guide," version 10.x, 2024. [Online].
     Available: https://docs.nvidia.com/deeplearning/tensorrt/ (accessed
     2026-04).

[20] Microsoft, "CoreML Execution Provider," ONNX Runtime documentation,
     2024. [Online]. Available:
     https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
     (accessed 2026-04).

[21] Ultralytics, "NVIDIA Jetson — YOLO Deployment Guide," 2024.
     [Online]. Available:
     https://docs.ultralytics.com/guides/nvidia-jetson/ (accessed
     2026-04). Published Orin Nano Super (25 W) and AGX Orin (MAXN)
     YOLOv11s benchmarks at 640 × 640 TensorRT FP16.

[22] W. Sultani, C. Chen, M. Shah, "Real-World Anomaly Detection in
     Surveillance Videos," *Proc. IEEE/CVF Conf. Computer Vision and
     Pattern Recognition (CVPR)*, 2018, pp. 6479–6488.
     arXiv:1801.04264. DOI: 10.1109/CVPR.2018.00678. *Introduces
     UCF-Crime (1 900 long surveillance videos, 13 anomaly classes
     + Normal) and the MIL-ranking baseline at 75.4 % AUC.*

[23] Y. Tian, G. Pang, Y. Chen, R. Singh, J. W. Verjans, G.
     Carneiro, "Weakly-supervised Video Anomaly Detection with
     Robust Temporal Feature Magnitude Learning (RTFM),"
     *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2021,
     pp. 4955–4966. arXiv:2101.10030.

[24] S. Li, F. Liu, L. Jiao, "Self-Training Multi-Sequence Learning
     with Transformer for Weakly Supervised Video Anomaly Detection
     (MSL)," *Proc. AAAI Conf. on Artificial Intelligence*, vol. 36,
     no. 2, pp. 1395–1403, 2022. arXiv:2104.04527.

[25] P. Wu, X. Zhou, G. Pang, Y. Sun, J. Liu, P. Wang, Y. Zhang,
     "VadCLIP: Adapting Vision-Language Models for Weakly
     Supervised Video Anomaly Detection," *Proc. AAAI Conf. on
     Artificial Intelligence*, vol. 38, no. 6, pp. 6074–6082, 2024.
     arXiv:2308.11681.
