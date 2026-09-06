# Related Work: Small-N Industrial Unsafe-Behaviour Recognition & Video Backbone Alternatives

## 1. Small-N industrial/worksite unsafe-behaviour recognition table

| Work | N (train/test or total) | Classes | Method | Reported acc | Notes |
|---|---|---|---|---|---|
| Onal & Dandil, Unsafe-Net (our baseline) | 452/64 clips (4-class eval) | 4 unsafe | YOLOv4 + ConvLSTM | 62/64=96.88% top-1, 95.81% macro | see verification doc |
| Yang et al., CMA dataset, "Transformer-based deep learning model... unsafe action identification" (Autom. Constr. 2023) [1] | 1595 clips total (YouTube-sourced) | 7 (5 scored) | dual feature-encoder + fusion transformer | avg 89% (per-class 83-97%) | larger N than us, still small by video-recognition standards; source data is YouTube, not proprietary site footage |
| "Real-Time Detection of Unsafe Worker Behaviors via Adaptive Vision Transformers" (Buildings 2025) [2] | CMA-based, 1595 clips (subset of 2000) | not confirmed exactly | 6 transformer video backbones compared | VideoMAE V2 best at 98.10%, then UniFormer V2, Video Swin, MViT, ViViT, TimeSformer | could not access full text (403); accuracy figure and ranking cross-confirmed via 2 independent search snippets |
| Miner unsafe-action recognition, CBAM-PoseC3D/MFFAR (2024) [3] | not confirmed (fetch failed) | miner-specific unsafe actions | skeleton (PoseC3D) + CBAM attention + RGB fusion | improved over PoseC3D/ST-GCN/2S-AGCN baselines | skeleton-based, industrial, directly relevant family - worth a manual follow-up read |
| STMF-Net, construction worker skeleton actions (2024) [4] | not confirmed | construction worker actions | 3D-skeleton multi-feature network | 79.36% | skeleton-based, construction-specific, lower accuracy than RGB approaches at unconfirmed N |
| "Action Recognition based Industrial Safety Violation Detection" (arXiv 2412.05531) [5] | not extracted (PDF unparseable) | industrial safety violations | not confirmed | not confirmed | flag for manual read if time allows |
| Teacher-student, low-resolution unsafe action recognition, construction (2023) [6] | not confirmed | unsafe actions | teacher-student KD | not confirmed | addresses low-res/far-field, a different axis than small-N |

**Pattern across the field:** every industrial/construction unsafe-behaviour paper we found operates at N in the hundreds-to-low-thousands of clips, i.e. the same regime as Unsafe-Net and us. None reports a confidence interval or significance test (see stats doc). The Yang/CMA line of work (1595 clips, 7-ish classes) is the closest N-comparable competitor to Unsafe-Net and shows modern transformer video backbones (VideoMAE V2 98.10%) beating ConvLSTM-style baselines at a similar or larger N - useful precedent that a stronger backbone helps, but their N is still ~3.5x ours (452 train clips).

## 2. Video backbones vs ConvLSTM - data requirement per backbone (ruthless take, our budget = 452 train clips)

| Backbone | Realistic min. training-set size | Fit for 452 clips | Basis |
|---|---|---|---|
| VideoMAE / VideoMAEv2 (from scratch or SSL pretrain) | "small" in their own papers still means ~3-4k clips minimum for the SSL pretrain stage; **fine-tuning** a Kinetics-400-pretrained checkpoint is far cheaper but still typically wants low thousands of labelled clips per class to avoid catastrophic overfitting on a 400-1000-way head [7][8] | Marginal / risky at 452 total, viable only as a frozen-backbone linear-probe or lightly fine-tuned head, not full fine-tune | [7] arXiv 2203.12602, [8] arXiv 2303.16727 |
| X3D | Designed for efficiency (4.8x fewer FLOPs at equal accuracy vs prior SOTA), still trained on Kinetics-scale data originally; transfer via fine-tuning small head on pretrained backbone is the only realistic path at 452 clips | Viable ONLY as frozen/lightly-fine-tuned pretrained backbone + linear head | [9] X3D paper |
| TimeSformer | Attention-heavy, known to need large pretraining data (ImageNet-21k + big video sets) before it works; not designed for training from scratch on small sets | Not viable to fine-tune fully; frozen-backbone probe only, and even that likely underperforms X3D/ConvLSTM at this N | [10] |
| Video Swin | Similar profile to TimeSformer - Swin locality bias helps somewhat but still large-pretrain-dependent | Not viable full fine-tune; frozen-backbone probe possible | [11] |
| MViTv2 | Trained from scratch on SSv2-scale data in original paper; needs large N unless using released Kinetics/SSv2 checkpoints as frozen feature extractor | Not viable full fine-tune; frozen-backbone probe possible | [12] |
| CLIP-based few-shot (ViFi-CLIP, ActionCLIP, X-CLIP) | **Explicitly designed and benchmarked for K=2,4,6,8-shot regimes** on HMDB-51/UCF-101/SSv2. ViFi-CLIP reports its largest gains precisely in the "extremely limited data" scenarios (+9.3% UCF-101, +4.2% HMDB-51 vs prior few-shot SOTA) [13][14] | **Best fit for 452 clips.** With ~113 clips/class average (452/4), we are well above the K=8-shot regime these methods target - i.e. we have MORE data than their stated sweet spot, not less. Low risk of overfitting because CLIP's visual-language backbone stays frozen or lightly adapted | [13] arXiv 2212.03640 (ViFi-CLIP), [14] emergentmind summary |

**Conclusion on backbones:** ViFi-CLIP-style CLIP few-shot transfer is the only backbone family in this list explicitly validated at our data scale (K-shot << our ~113 clips/class). Everything else (VideoMAE, X3D, TimeSformer, Video Swin, MViTv2) needs either large pretraining data (which we don't generate ourselves, only reuse public checkpoints) or a full fine-tune we cannot safely do at N=452 without frozen-backbone protection.

## 3. Skeleton/pose-based (ST-GCN, PoseC3D) fit for our 4 behaviours

- ST-GCN and PoseC3D are generally more sample-efficient than RGB video backbones because the input (2D/3D joints or pose heatmaps) is already a strong, low-dimensional, motion-focused representation - less for the network to learn from few examples [15][16].
- PoseC3D specifically reported more robust to pose-estimation noise and better cross-dataset generalization than GCN methods [16].
- Industrial precedent exists: CBAM-PoseC3D for miner unsafe actions [3], STMF-Net for construction worker skeleton actions (79.36% acc) [4] - both confirm the family is actively used in this exact problem class.
- **Fit assessment for our 4 classes:** poor-to-moderate. Two of our four unsafe behaviours (Opened Panel Cover, Carrying Overload with Forklift) are defined by **object/scene state** (a panel's physical state, a forklift's load count), not by human pose/motion - skeleton features cannot see the panel or count blocks on a forklift. Only Safe Walkway Violation (person position vs. a zone) and Unauthorized Intervention (person presence/vest colour) are meaningfully human-pose-related, and even those need an object/zone context skeleton data alone doesn't carry. Skeleton-only approaches would need to be fused with an object detector output anyway (as CBAM-MFFAR does, RGB+skeleton fusion) - i.e. pure skeleton is not a viable standalone replacement here, though a fusion variant could be a second-order idea beyond scope.

## Ranked recommendations (see final message for full detail + compute cost)
1. ViFi-CLIP style CLIP few-shot fine-tune (frozen/lightly-adapted backbone + temporal pooling + linear head) - best data-fit at N=452.
2. Frozen-backbone X3D (or similar 3D-CNN pretrained on Kinetics) linear/light-head probe, as a lower-effort fallback if CLIP transfer underperforms - still respects the small-N constraint via frozen features.

## Sources
[1] https://www.sciencedirect.com/science/article/abs/pii/S0926580522005738 (Yang et al., CMA dataset)
[2] https://doi.org/10.3390/buildings15224205 (Real-Time Detection of Unsafe Worker Behaviors via Adaptive Vision Transformers)
[3] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11281114/ (CBAM-PoseC3D/MFFAR miner unsafe action)
[4] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11644321/ (STMF-Net)
[5] https://arxiv.org/pdf/2412.05531 (Action Recognition based Industrial Safety Violation Detection - unparseable, needs manual follow-up)
[6] https://dl.acm.org/doi/10.1016/j.aei.2023.102294 (teacher-student low-res unsafe action)
[7] https://arxiv.org/pdf/2203.12602 (VideoMAE)
[8] https://arxiv.org/pdf/2303.16727 (VideoMAE V2)
[9] https://www.researchgate.net/publication/343457124_X3D_Expanding_Architectures_for_Efficient_Video_Recognition
[10] https://openaccess.thecvf.com/content/CVPR2023/papers (TimeSformer robustness survey context)
[11] https://arxiv.org/pdf/2106.13230 (Video Swin Transformer)
[12] https://mmaction2.readthedocs.io/en/latest/model_zoo/recognition.html (MViT)
[13] https://arxiv.org/pdf/2212.03640 (Fine-tuned CLIP Models are Efficient Video Learners / ViFi-CLIP)
[14] https://www.emergentmind.com/topics/vifi-clip
[15] https://github.com/fmthoker/st-gcn
[16] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11281114/ (PoseC3D robustness claim)

## Unresolved questions
- Could not access full text of [2] (403) or [3]/[4] fetch details (502/blocked) - N and exact protocol for those three not independently confirmed beyond search-snippet level; treat their numbers as indicative, not verified.
- [5] PDF was unparseable (binary/corrupted extraction) - dataset size and method unconfirmed, worth a manual read if this becomes load-bearing for the thesis.
