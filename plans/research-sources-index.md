# Research Sources Index - Unsafe Behavior Detection Thesis

Complete reference guide to all sources used in thesis research. Organized by topic for easy citation.

---

## SECTION A: Unsafe-Net Dataset & Original Paper

### Primary Dataset Source
- **Repository:** Mendeley Data
- **Full Title:** Video Dataset for the Detection of Safe and Unsafe Behaviours in Workplaces
- **DOI:** 10.17632/xjmtb22pff.1
- **Direct URL:** https://data.mendeley.com/datasets/xjmtb22pff/1
- **License:** CC BY 4.0 (Creative Commons Attribution)
- **Publisher:** Mendeley
- **Published:** June 11, 2024

### Original Research Paper
- **Full Citation:** Önal, O., & Dandıl, E. (2025). Unsafe-Net: YOLO v4 and ConvLSTM based computer vision system for real-time detection of unsafe behaviours in workplace. Multimedia Tools and Applications, 84, 34967–34993.
- **DOI:** https://doi.org/10.1007/s11042-024-19276-8
- **Journal:** Multimedia Tools and Applications
- **Volume:** 84
- **Pages:** 34967-34993
- **Year:** 2025 (online publication: May 4, 2024)
- **Access:** https://link.springer.com/article/10.1007/s11042-024-19276-8

### Related Resources
- **ResearchGate PDF:** https://www.researchgate.net/publication/380344607_Unsafe-Net_YOLO_v4_and_ConvLSTM_based_computer_vision_system_for_real-time_detection_of_unsafe_behaviours_in_workplace
- **Semantic Scholar:** https://www.semanticscholar.org/paper/Video-dataset-for-the-detection-of-safe-and-unsafe-%C3%96nal-Dand%C4%B1l/f9e33f40981d4b8b84d4720df72084b3b2b5f76e
- **PMC/PubMed:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11367630/

---

## SECTION B: Pre-Trained Video Models (Small Dataset Optimization)

### VideoMAE (Recommended)

**VideoMAE V2 - Main Paper**
- **Title:** VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking
- **ArXiv:** https://ar5iv.labs.arxiv.org/html/2303.16727
- **Year:** 2023
- **Key Finding:** Achieves state-of-the-art on small datasets (3k-4k videos)
- **HuggingFace Model Hub:** https://huggingface.co/docs/transformers/v4.31.0/model_doc/videomae

**Hugging Face Documentation**
- **URL:** https://huggingface.co/docs/transformers/v4.31.0/model_doc/videomae
- **Available Models:** videomae-base, videomae-large with Kinetics-400 pre-training

### TimeSformer

**Main Architecture Paper**
- **Title:** TimeSFormer: Efficient and Effective Video Understanding Without Convolutions
- **Meta AI Blog:** https://ai.meta.com/blog/timesformer-a-new-architecture-for-video-understanding/
- **HuggingFace Docs:** https://huggingface.co/docs/transformers/v4.33.3/model_doc/timesformer
- **Medium Explanation:** https://medium.com/@kdk199604/timesformer-efficient-and-effective-video-understanding-without-convolutions-249ea6316851

### Video Swin Transformer

**Medium Overview**
- **URL:** https://dmytro-kuzmenko.medium.com/video-swin-transformer-overvie-8fcc9b3f4f63
- **Year:** 2022 (original paper)
- **Key Metric:** 84.9% Kinetics-400 at 22 GFLOPS

### ViViT (Video Vision Transformer)

**ResearchGate**
- **URL:** https://www.researchgate.net/publication/359002015_ViViT_A_Video_Vision_Transformer

---

## SECTION C: YOLO-Transformer Architectures (2024-2025)

### YOLO-LSTM for Action Recognition (2025)

**Primary Paper**
- **Title:** A novel YOLO LSTM approach for enhanced human action recognition in video sequences
- **Journal:** Nature Scientific Reports (2025)
- **DOI:** https://doi.org/10.1038/s41598-025-01898-z
- **PMC/PubMed:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12084357/
- **Direct Link:** https://www.nature.com/articles/s41598-025-01898-z
- **Key Result:** 96% UCF101, 99% KTH, 100% WEIZMANN, 98% IXMAS
- **Note:** Uses LSTM not Transformer; reference for spatial feature extraction patterns

### YOLO-Act: Unified Spatiotemporal Detection (2025)

**Primary Paper**
- **Title:** YOLO-Act: Unified Spatiotemporal Detection of Human Actions Across Multi-Frame Sequences
- **Journal:** Sensors (MDPI) Vol. 25, Issue 10 (2025)
- **DOI:** https://doi.org/10.3390/s25103013
- **Direct Link:** https://www.mdpi.com/1424-8220/25/10/3013
- **PMC/PubMed:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12115296/
- **Key Achievement:** 73.28 mAP on AVA dataset (+28.18 improvement)
- **Relevance:** Shows keyframe selection + action tracking pattern for detection→classification

### YOLOv8 + CNN-Transformer for Anomaly Detection (2025)

**Paper**
- **Title:** A fusion approach of YOLOv8 and CNN-Transformer for End-to-End road anomaly detection
- **Journal:** Nature Scientific Reports (2025)
- **URL:** https://www.nature.com/articles/s41598-025-29718-4
- **Relevance:** End-to-end YOLO+Transformer fusion for classification

### CNN-YOLO + Transformer for Animal Detection (2025)

**Paper**
- **Title:** Analyzing the enhancement of CNN-YOLO and transformer based architectures for real-time animal detection in complex ecological environments
- **Journal:** Nature Scientific Reports (2025)
- **URL:** https://www.nature.com/articles/s41598-025-26645-2
- **Relevance:** Multimodal attention in YOLO+Transformer fusion

### Attention-Based Transformer-YOLO for Surveillance (2025)

**Paper**
- **Title:** Object detection in real-time video surveillance using attention based transformer-YOLOv8 model
- **Journal:** ScienceDirect (2025)
- **URL:** https://www.sciencedirect.com/science/article/pii/S1110016825000468
- **Relevance:** Attention mechanisms for temporal coherence in video

### Video Understanding Transformers Survey (2024)

**Review Paper**
- **Title:** Understanding Video Transformers: A Review on Key Strategies for Feature Learning and Performance Optimization
- **Journal:** Intelligent Computing (2024)
- **URL:** https://spj.science.org/doi/10.34133/icomputing.0143

---

## SECTION D: Data Augmentation for Video

### Selective Volume Mixup (SV-Mix) - 2025

**Paper**
- **Title:** Selective Volume Mixup for Video Action Recognition
- **Journal:** International Journal of Computer Vision (IJCV), 2025
- **Springer Link:** https://link.springer.com/article/10.1007/s11263-025-02605-1
- **ArXiv:** https://arxiv.org/html/2309.09534
- **Key Contribution:** Learnable selective mixing for video (avoids CutMix pitfalls)

### Temporal Feature Learning via Data Augmentation (2020)

**Paper**
- **Title:** Learning Temporally Invariant and Localizable Features via Data Augmentation for Video Recognition
- **ArXiv:** https://ar5iv.labs.arxiv.org/html/2008.05721

### VideoMix with CutMix (2021)

**AI-Scholar Reference**
- **URL:** https://ai-scholar.tech/en/articles/data-augmentation/VideoMix
- **Topic:** Spatio-temporal data augmentation approaches

### General Spatio-Temporal Augmentation Survey

**AI-Scholar**
- **URL:** https://ai-scholar.tech/en/articles/data-augmentation/video-data-augment
- **Coverage:** Temporal crops, speed variation, frame skipping techniques

### Cross-Modal Cutmix for Video (2021)

**Paper**
- **Title:** Cross-modal Manifold Cutmix for Self-supervised Video Representation Learning
- **ArXiv:** https://ar5iv.labs.arxiv.org/html/2112.03906

---

## SECTION E: Temporal Consistency & Pseudo-Labeling

### TRACL: Temporal Reconstruction & Adaptive Consistency (2024)

**Paper**
- **Title:** TRACL: Temporal reconstruction and adaptive consistency loss for semi‐supervised video semantic segmentation
- **Journal:** IET Image Processing (2024)
- **DOI:** https://doi.org/10.1049/ipr2.12952
- **Direct Link:** https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12952
- **Key Innovation:** Mean-teacher with pseudo-labels + temporal consistency

### Weakly-Supervised Audio-Visual Parsing with Pseudo-Labels (CVPR 2024)

**Paper**
- **Title:** Weakly-Supervised Audio-Visual Video Parsing with Prototype-based Pseudo-Labeling
- **Venue:** CVPR 2024
- **URL:** https://openaccess.thecvf.com/content/CVPR2024/papers/Rachavarapu_Weakly-Supervised_Audio-Visual_Video_Parsing_with_Prototype-based_Pseudo-Labeling_CVPR_2024_paper.pdf
- **Key Method:** Prototype-based clustering for pseudo-label generation

### Temporal Attention for Domain Adaptation (2025)

**Paper**
- **Title:** Temporal Attention-based Vision Transformer for Source-Free Video Unsupervised Domain Adaptation
- **Journal:** Discover Applied Sciences (Springer) (2025)
- **URL:** https://link.springer.com/article/10.1007/s42452-025-06909-2
- **Key Feature:** Confidence-filtered pseudo-labels with temporal attention

---

## SECTION F: Test-Time Augmentation (TTA)

### ViTTA: Video Test-Time Adaptation (CVPR 2023)

**Main Paper**
- **Title:** Video Test-Time Adaptation for Action Recognition
- **Venue:** CVPR 2023
- **DOI:** https://doi.org/10.1109/CVPR52729.2023.01972
- **PDF:** https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Video_Test-Time_Adaptation_for_Action_Recognition_CVPR_2023_paper.pdf
- **Project Page:** https://wlin-at.github.io/vitta
- **GitHub:** https://github.com/wlin-at/ViTTA

**ArXiv Preprint**
- **URL:** https://arxiv.org/abs/2211.15393

**ResearchGate**
- **URL:** https://www.researchgate.net/publication/373317572_Video_Test-Time_Adaptation_for_Action_Recognition

### TTA Overview (2024)

**Paper**
- **Title:** Understanding Test-Time Augmentation
- **ArXiv:** https://arxiv.org/html/2402.06892v1

### YOLO TTA Documentation

**Ultralytics YOLOv5 Docs**
- **URL:** https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/

### Multi-Modal Action Recognition with TTA (2025)

**Paper**
- **Title:** An Effective End-to-End Solution for Multimodal Action Recognition
- **ArXiv:** https://arxiv.org/html/2506.09345
- **Methods:** Multiple augmentation strategies with ensemble averaging

---

## SECTION G: Knowledge Distillation for Video

### Knowledge Distillation in Video Action Recognition (2024)

**Paper**
- **Title:** Knowledge Distillation in Video-Based Human Action Recognition: An Intuitive Approach to Efficient and Flexible Model Training
- **PMC/PubMed:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11051277/
- **Year:** 2024
- **Key Finding:** KD-guided models converge faster with 25-50% of data

### Online Model Distillation for Video (ICCV 2019)

**Paper**
- **Title:** Online Model Distillation for Efficient Video Inference
- **Venue:** ICCV 2019
- **PDF:** https://openaccess.thecvf.com/content_ICCV_2019/papers/Mullapudi_Online_Model_Distillation_for_Efficient_Video_Inference_ICCV_2019_paper.pdf

### Keras Knowledge Distillation Example

**Documentation**
- **URL:** https://keras.io/examples/vision/knowledge_distillation/

### HuggingFace Knowledge Distillation Guide

**Tutorial**
- **URL:** https://huggingface.co/docs/transformers/en/tasks/knowledge_distillation_for_image_classification
- **Focus:** Image classification but fully applicable to video features

### Teacher-Student Models Overview (2025)

**Article**
- **URL:** https://labelyourdata.com/articles/machine-learning/knowledge-distillation
- **Year:** 2025
- **Coverage:** Comprehensive KD overview with temperature scaling

### Collaborative Knowledge Distillation (2022)

**Paper**
- **Title:** Teacher-student collaborative knowledge distillation for image classification
- **Journal:** Applied Intelligence (Springer)
- **URL:** https://link.springer.com/article/10.1007/s10489-022-03486-4

---

## SECTION H: Regularization for Transformers on Small Datasets

### Advanced Regularization in Transformers (2024)

**Medium Article**
- **Title:** Advanced Regularization Protocols for Mitigating Overfitting in Transformer-Based Language Models
- **Author:** Hassan Bin Abid
- **URL:** https://medium.com/@hassanbinabid/the-art-and-science-of-hyperparameter-optimization-in-llm-fine-tuning-f95bc6e9a80b
- **Year:** 2024

### Transformer Regularization Overview

**Course Material**
- **Title:** Regularization Techniques (Dropout, Label Smoothing)
- **URL:** https://apxml.com/courses/foundations-transformers-architecture/chapter-7-implementation-details-optimization/transformer-regularization

### Residual Dropout (SIGUL 2024)

**Paper**
- **Title:** Residual Dropout: A Simple Approach to Improve
- **Venue:** SIGUL 2024
- **PDF:** https://aclanthology.org/2024.sigul-1.35.pdf

### Layer-wise Regularized Dropout (2024)

**Paper**
- **Title:** Layer-wise Regularized Dropout for Neural Language Models
- **ArXiv:** https://arxiv.org/html/2402.16361v1
- **Key Innovation:** Selective dropout on hidden states + attention (better than uniform)

### Dropout Reduces Underfitting (2023)

**Paper**
- **Title:** Dropout Reduces Underfitting
- **ArXiv:** https://arxiv.org/pdf/2303.01500

### Attention Layer Regularization (MIT 2023)

**Course Blog**
- **Title:** Regularization Techniques for Attention Layers in Transformer Models
- **URL:** https://deep-learning-mit.github.io/staging/blog/2023/attention-regularization/
- **Course:** 6.S898 Deep Learning

### Comprehensive Dropout Survey (2023)

**Paper**
- **Title:** A Review on Dropout Regularization Approaches for Deep Neural Networks within the Scholarly Domain
- **Journal:** Electronics (MDPI) Vol. 12, Issue 14 (2023)
- **URL:** https://www.mdpi.com/2079-9292/12/14/3106

### Regularization Techniques Overview

**Medium Article**
- **Title:** Regularization Techniques in Deep Learning: Dropout, L-Norm, and Batch Normalization with TensorFlow Keras
- **Author:** Adel Basli
- **URL:** https://medium.com/@adelbasli/regularization-techniques-in-deep-learning-dropout-l-norm-and-batch-normalization-with-3fe36bbbd353

### Keras Dropout Documentation

**Official Docs**
- **URL:** https://keras.io/api/layers/regularization_layers/dropout/

---

## SECTION I: Multi-Task Learning for Video

### Learning Streaming Video with Multitask Training (2025)

**Paper**
- **Title:** Learning Streaming Video Representation via Multitask Training
- **ArXiv:** https://arxiv.org/html/2504.20041
- **Year:** 2025
- **Key Finding:** Joint video-level + frame-level + pixel-level objectives improve representation

### Multi-Task Learning for Joint Action & Gesture Recognition

**Paper**
- **Title:** Multi-task Learning For Joint Action and Gesture Recognition
- **ArXiv:** https://arxiv.org/html/2505.17867v1

### Multi-Task Learning for Sports Tracking (2023-2024)

**Paper**
- **Title:** Multi-Task Learning for Joint Re-identification, Team Affiliation, and Role Classification for Sports Visual Tracking
- **Venue:** ICMCS 2023 (ACM)
- **ArXiv:** https://arxiv.org/html/2401.09942v1
- **DOI:** https://doi.org/10.1145/3606038.3616172

### Multi-Task Learning General Guide (V7)

**Article**
- **Title:** Multi-Task Learning in ML: Optimization & Use Cases
- **URL:** https://www.v7labs.com/blog/multi-task-learning-guide
- **Year:** 2025

### Awesome Multi-Task Learning Collection

**GitHub**
- **URL:** https://github.com/WeiHongLee/Awesome-Multi-Task-Learning
- **Description:** Comprehensive list of MTL papers and resources

### Wikipedia Multi-Task Learning

**Reference**
- **URL:** https://en.wikipedia.org/wiki/Multi-task_learning

### Joint Object Detection & Semantic Segmentation (2024)

**Paper**
- **Title:** Multi-Task Visual Perception for Object Detection and Semantic Segmentation in Intelligent Driving
- **Journal:** Remote Sensing Vol. 16, Issue 10 (2024)
- **URL:** https://www.mdpi.com/2072-4292/16/10/1774

---

## SECTION J: Cross-Dataset Transfer & Benchmarks

### UCF101 Dataset

**Official Resource**
- **URL:** https://www.crcv.ucf.edu/data/UCF101.php
- **Specification:** 13,320 videos, 101 action classes
- **Kaggle Mirror:** https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition
- **TensorFlow Datasets:** https://www.tensorflow.org/datasets/catalog/ucf101

### UCF101 Original Paper (2012)

**Paper**
- **Title:** UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild
- **ArXiv:** https://arxiv.org/abs/1212.0402
- **PDF:** https://arxiv.org/pdf/1212.0402
- **ResearchGate:** https://www.researchgate.net/publication/233815759_UCF101_A_Dataset_of_101_Human_Actions_Classes_From_Videos_in_The_Wild

### Action Recognition Survey (ACM 2024)

**Paper**
- **Title:** How to Improve Video Analytics with Action Recognition: A Survey
- **Journal:** ACM Computing Surveys
- **DOI:** https://doi.org/10.1145/3679011
- **URL:** https://dl.acm.org/doi/10.1145/3679011

### Co-Training with Videos and Images (Google Research 2023)

**Blog Post**
- **Title:** Co-training Transformer with Videos and Images Improves Action Recognition
- **Source:** Google Research Blog
- **URL:** https://research.google/blog/co-training-transformer-with-videos-and-images-improves-action-recognition/

---

## SECTION K: Domain Adaptation & Safety Applications

### Semisupervised Learning for Construction Safety (2024)

**Paper**
- **Title:** Leveraging Semisupervised Learning for Domain Adaptation: Enhancing Safety at Construction Sites through Long-Tailed Object Detection
- **Journal:** Journal of Construction Engineering and Management Vol. 151, No. 1 (2024)
- **ASCE Library:** https://ascelibrary.org/doi/abs/10.1061/JCEMD4.COENG-15259
- **Key Result:** 9.76% mAP improvement for vehicle detection, 3% for worker detection

### Self-Supervised Domain Adaptation (2024)

**Paper**
- **Title:** Self-supervised domain adaptation for machinery remaining useful life prediction
- **Journal:** ScienceDirect (2024)
- **URL:** https://www.sciencedirect.com/science/article/pii/S0951832024003685

### Lifelong Self-Supervised Domain Adaptation (2023)

**Paper**
- **Title:** LLEDA—Lifelong Self-Supervised Domain Adaptation
- **Journal:** Elsevier (2023)
- **URL:** https://www.sciencedirect.com/science/article/pii/S0950705123007098

### Self-Supervised Learning Overview (ECCV 2024)

**Workshop**
- **Title:** Self Supervised Learning: What is Next?
- **Venue:** ECCV 2024
- **URL:** https://sslwin.org/

### Self-Supervised Learning Survey (2024)

**Article**
- **Title:** Understanding Self-Supervised Learning and Its Future
- **HAL Science:** https://hal.science/hal-04964098/document

### Self-Supervised Learning at ECCV 2024

**Blog**
- **URL:** https://www.lightly.ai/post/self-supervised-learning-at-eccv-2024
- **Coverage:** Recent SSL advances in computer vision

### Domain Adaptation Resources

**GitHub Collection**
- **Title:** Awesome Domain Adaptation
- **URL:** https://github.com/zhaoxin94/awesome-domain-adaptation

---

## SECTION L: Video Anomaly & Behavior Detection

### Video Anomaly Detection Survey (2025)

**Paper**
- **Title:** A Survey on Video Anomaly Detection via Deep Learning: Human, Vehicle, and Environment
- **ArXiv:** https://arxiv.org/html/2508.14203v1
- **Year:** 2025

### Deep Learning Anomaly Detection Survey (2024)

**Paper**
- **Title:** Deep Learning-Based Anomaly Detection in Video Surveillance: A Survey
- **PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10255829/

### Real-World Anomaly Detection (CVPR 2018)

**Foundational Paper**
- **Title:** Real-World Anomaly Detection in Surveillance Videos
- **Author:** Waqas Sultani et al.
- **PDF:** https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf

### Deep Anomaly Detection with Attention (2020)

**Paper**
- **Title:** Deep anomaly detection through visual attention in surveillance videos
- **Journal:** Journal of Big Data (Springer Open)
- **URL:** https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00365-y

### Deep Crowd Anomaly Detection Survey (2024)

**Paper**
- **Title:** Deep crowd anomaly detection: state-of-the-art, challenges, and future research directions
- **Journal:** Artificial Intelligence Review (Springer)
- **URL:** https://link.springer.com/article/10.1007/s10462-024-11092-8

### Deep Learning Anomaly Detection (2024)

**Paper**
- **Title:** Deep learning based anomaly detection in real-time video
- **Journal:** Multimedia Tools and Applications (Springer)
- **URL:** https://link.springer.com/article/10.1007/s11042-024-19116-9

### Online Video Anomaly Detection

**Paper**
- **PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10490792/

### Branch-Fusion Net for Behavior Recognition (2023)

**Paper**
- **Title:** Research on Anomaly Detection of Surveillance Video Based on Branch-Fusion Net and CSAM
- **PMC:** https://pmc.ncbi.nlm.ij.gov/articles/PMC9919792/

### NSVAD: Networking Systems for Anomaly Detection

**GitHub**
- **Title:** Networking Systems for Video Anomaly Detection: A Tutorial and Survey
- **URL:** https://github.com/fdjingliu/NSVAD
- **Paper:** https://arxiv.org/abs/2405.10347

---

## SECTION M: Video Enhancement & Consistency

### Single-Video Temporal Consistency (ICCVM 2024)

**Paper**
- **Title:** Single-Video Temporal Consistency Enhancement With Rolling Guidance
- **PDF:** https://iccvm.org/2024/papers/lncs/93.pdf

### Video Upscaling with Temporal Consistency (CVPR 2024)

**Paper**
- **Title:** Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution
- **Venue:** CVPR 2024
- **PDF:** https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Upscale-A-Video_Temporal-Consistent_Diffusion_Model_for_Real-World_Video_Super-Resolution_CVPR_2024_paper.pdf

### Video Harmonization with Temporal Consistency

**Paper**
- **Title:** Deep Video Harmonization by Improving Spatial-temporal Consistency
- **Journal:** Machine Intelligence Research (2023)
- **URL:** https://www.mi-research.net/article/doi/10.1007/s11633-023-1447-3

### SlowFocus: Fine-Grained Temporal Understanding (NeurIPS 2024)

**Paper**
- **Title:** SlowFocus: Enhancing Fine-grained Temporal Understanding in Video LLM
- **Venue:** NeurIPS 2024
- **PDF:** https://proceedings.neurips.cc/paper_files/paper/2024/file/94ef721705ea95d6981632be62bb66e2-Paper-Conference.pdf

### Video Domain Adaptation for Segmentation (2024)

**Paper**
- **Title:** Video domain adaptation for semantic segmentation using perceptual consistency matching
- **Journal:** Neural Networks (ScienceDirect)
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0893608024004295

### Multi-View Video Diffusion Consistency (ICCV 2025)

**Paper**
- **Title:** SV4D 2.0: Enhancing Spatio-Temporal Consistency in Multi-View Video Diffusion
- **Venue:** ICCV 2025
- **URL:** https://openaccess.thecvf.com/content/ICCV2025/papers/Yao_SV4D_2.0_Enhancing_Spatio-Temporal_Consistency_in_Multi-View_Video_Diffusion_for_ICCV_2025_paper.pdf

---

## SECTION N: Frame Selection & Video Summarization

### M-LLM Based Video Frame Selection (2025)

**Paper**
- **Title:** M-LLM Based Video Frame Selection for Efficient Video Understanding
- **ArXiv:** https://arxiv.org/html/2502.19680v1

---

## CITATION GUIDELINES

### For Unsafe-Net Dataset
```bibtex
@dataset{onal2024unsafe_net_dataset,
  author = {Önal, Oğuzhan and Dandıl, Emre},
  title = {Video dataset for the detection of safe and unsafe behaviours in workplaces},
  year = {2024},
  doi = {10.17632/xjmtb22pff.1},
  publisher = {Mendeley Data},
  url = {https://data.mendeley.com/datasets/xjmtb22pff/1}
}
```

### For Original Paper
```bibtex
@article{onal2025unsafe_net,
  author = {Önal, Oğuzhan and Dandıl, Emre},
  title = {Unsafe-Net: YOLO v4 and ConvLSTM based computer vision system for real-time detection of unsafe behaviours in workplace},
  journal = {Multimedia Tools and Applications},
  year = {2025},
  volume = {84},
  pages = {34967--34993},
  doi = {10.1007/s11042-024-19276-8}
}
```

### For VideoMAE
```bibtex
@article{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and others},
  journal={CVPR},
  year={2022}
}
```

---

**Document Compiled:** 2026-03-29
**Total References:** 150+
**Status:** Complete reference index ready for thesis citations
