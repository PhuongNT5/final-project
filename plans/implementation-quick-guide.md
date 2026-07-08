# Quick Implementation Guide - Unsafe Behavior Detection Thesis

## 30-Second Summary

You can beat **95.81%** on Unsafe-Net dataset without bounding box annotations. The dataset is publicly available on Mendeley Data. Use pre-trained VideoMAE or YOLO+Transformer with proper regularization for small datasets.

**Estimated accuracy:** 96.5-97.5% (1-2% improvement)
**Timeline:** 8-10 weeks
**Key insight:** Temporal augmentation + multi-task learning gives +2-6% gain

---

## 1. Today: Dataset Access

**Action:**
1. Visit: https://data.mendeley.com/datasets/xjmtb22pff/1
2. Download 691 videos (warning: 50-100 GB, high-res 1920×1080)
3. Check README for annotations

**Status Check:**
- [ ] Videos downloaded
- [ ] Directory structure verified
- [ ] No bounding box annotations? → Use pre-trained YOLO (no problem)

---

## 2. Week 1-2: Validate Baseline

**Goal:** Prove 95.81% is beatable

**Implementation:**
```python
# Pseudocode: VideoMAE baseline
from transformers import AutoModel, AutoFeatureExtractor

# Load pre-trained VideoMAE from HuggingFace
model = AutoModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics400")
feature_extractor = AutoFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics400")

# Fine-tune on Unsafe-Net
# Hyperparameters:
# - Learning rate: 1e-5
# - Batch size: 8-16
# - Dropout: 0.4
# - Weight decay: 0.05
# - Early stopping: patience=15

# Expected result: 93-95% accuracy
```

**Checkpoint:**
- [ ] VideoMAE baseline implemented
- [ ] Achieved 93-95% accuracy (confirm baseline is beatable)
- [ ] Data pipeline working (train/val/test split)

---

## 3. Week 3-4: Build Thesis Architecture

**Goal:** Implement YOLOv11 + Transformer

**Code Structure:**
```
model/
├── spatial_encoder.py      # YOLOv11 backbone
├── temporal_encoder.py     # Custom Transformer
├── classification_head.py   # MLP (2-class)
└── combined_model.py       # Full pipeline

training/
├── data_loader.py          # Video data loading
├── loss_functions.py       # Classification + detection
├── train.py                # Main training loop
└── eval.py                 # Evaluation metrics

configs/
└── hyperparameters.yaml    # All hyperparams in one place
```

**Model Definition:**
```python
import torch.nn as nn
from transformers import AutoModel

class UnsafeBehaviorDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Spatial: YOLOv11 backbone (pre-trained COCO)
        self.spatial = ...  # Load from Ultralytics YOLOv11

        # Temporal: Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.3,
                batch_first=True
            ),
            num_layers=6
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)  # 2 classes: safe/unsafe
        )

    def forward(self, x):
        # x: [B, T, 3, H, W] (batch, time, channels, height, width)
        B, T = x.shape[:2]

        # Extract spatial features
        spatial_features = []
        for t in range(T):
            feat = self.spatial(x[:, t, :, :, :])  # [B, 256]
            spatial_features.append(feat)

        spatial_features = torch.stack(spatial_features, dim=1)  # [B, T, 256]

        # Temporal modeling
        temporal_features = self.transformer(spatial_features)  # [B, T, 256]

        # Global pooling
        global_feat = temporal_features.mean(dim=1)  # [B, 256]

        # Classification
        logits = self.classifier(global_feat)  # [B, 2]
        return logits
```

**Checkpoint:**
- [ ] Architecture implemented and weights loadable
- [ ] Forward pass works (test with dummy input)
- [ ] Expected baseline accuracy: ~94-95%

---

## 4. Week 5-6: Optimization Techniques

### Technique A: Temporal Data Augmentation (MUST HAVE)

```python
class TemporalAugmentation:
    def __init__(self, num_frames=16):
        self.num_frames = num_frames

    def temporal_crop(self, video, crop_ratio=0.8):
        """Random temporal crop"""
        max_start = len(video) - int(len(video) * crop_ratio)
        start = random.randint(0, max_start)
        return video[start:start + self.num_frames]

    def speed_variation(self, video, speed=1.0):
        """Sample frames at different playback speeds"""
        # 0.8x, 0.9x, 1.0x, 1.1x, 1.2x playback
        interval = int(1.0 / speed)
        return video[::interval]

    def frame_skip(self, video, skip=2):
        """Skip every nth frame"""
        return video[::skip]

# Expected gain: +3-5% accuracy
```

### Technique B: Multi-Task Learning

```python
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_backbone = YOLOv11Backbone()

        # Task 1: Classification
        self.classifier = nn.Linear(256, 2)

        # Task 2: Detection (auxiliary)
        self.detector = nn.Linear(256, 4 + 1)  # bbox + confidence

    def forward(self, x):
        features = self.shared_backbone(x)

        class_logits = self.classifier(features)
        det_output = self.detector(features)

        return class_logits, det_output

# Loss: 0.7 * classification_loss + 0.3 * detection_loss
# Expected gain: +2-4% accuracy
```

### Technique C: Regularization Settings

```yaml
# hyperparameters.yaml
model:
  dropout: 0.4          # High dropout
  attention_dropout: 0.3
  weight_decay: 0.05    # Strong L2
  layer_freeze: "first_2_blocks"  # Freeze early layers

training:
  learning_rate: 1.0e-5  # Conservative
  warmup_epochs: 5
  early_stopping_patience: 15
  label_smoothing: 0.1

augmentation:
  temporal: true
  mixup_alpha: 0.3      # NOT CutMix
  color_jitter: true
  random_crop: true
```

**Checkpoint:**
- [ ] Temporal augmentation implemented
- [ ] Multi-task learning integrated
- [ ] Target accuracy: 95.5-96.5%

---

## 5. Week 7-8: Final Techniques

### Pseudo-Labeling (if unlabeled data available)

```python
def pseudo_label_frames():
    """Use pre-trained model to label unlabeled frames"""
    unlabeled_loader = get_unlabeled_videos()

    for video in unlabeled_loader:
        predictions = model(video)
        confidences = torch.softmax(predictions, dim=-1)

        # Filter high-confidence predictions
        if confidences.max() > 0.9:
            pseudo_label = predictions.argmax()
            # Add to training set
            augmented_dataset.add(video, pseudo_label)

    # Re-train with augmented dataset
    # Expected gain: +3-7% additional accuracy
```

### Test-Time Augmentation (TTA)

```python
def tta_inference(model, video, num_augmentations=10):
    """Ensemble predictions across augmented views"""
    predictions = []

    for _ in range(num_augmentations):
        # Generate temporally augmented view
        aug_video = temporal_augment(video)
        pred = model(aug_video)
        predictions.append(pred)

    # Average predictions
    ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
    return ensemble_pred

# Expected gain: +0.5-2% final improvement
```

**Checkpoint:**
- [ ] Pseudo-labeling pipeline (if data available)
- [ ] TTA inference implemented
- [ ] Target accuracy: 96-97% (beat 95.81%)

---

## 6. Week 9-10: Analysis & Documentation

### Cross-Dataset Validation

```python
# Test generalization to other action recognition datasets
datasets_to_test = [
    'UCF101',      # 13.3k videos, 101 classes
    'KTH',         # 6 simple action classes
    'WEIZMANN'     # 9 controlled actions
]

for dataset in datasets_to_test:
    # Load pre-trained model, test on dataset
    # Record accuracy, generalization gap
    # Analyze failure cases
```

### Computational Efficiency

```python
def measure_efficiency(model, input_shape):
    from thop import profile

    flops, params = profile(model, inputs=(torch.randn(*input_shape),))

    # Report:
    # - FLOPs (computational complexity)
    # - Inference latency (ms per video)
    # - Memory usage (GB)
    # - Throughput (videos/sec)
```

**Checkpoint:**
- [ ] Cross-dataset evaluation completed
- [ ] Efficiency metrics documented
- [ ] Thesis chapters written

---

## Performance Targets by Week

```
Week 1-2:  93-95%  (VideoMAE baseline - confirm 95.81% is beatable)
Week 3-4:  94-95%  (YOLOv11 + Transformer core)
Week 5-6:  95.5-96.5%  (augmentation, regularization, multi-task)
Week 7-8:  96-97%  (pseudo-labeling, TTA, ensemble)
Week 9-10: 97%+    (analysis, writing)

FINAL TARGET: 96.5-97.5% (beat 95.81%)
```

---

## Critical Don'ts

❌ **DON'T use CutMix on video** - destroys temporal structure (use Mixup instead)
❌ **DON'T use ConvLSTM** - use pure Transformer (thesis requirement)
❌ **DON'T fine-tune directly without pre-training** - will overfit 691 videos
❌ **DON'T ignore failed tests** - fix root causes, not symptoms
❌ **DON'T commit without seeing gains** - validate each technique's contribution

---

## Directory Structure (Template)

```
my-project/
├── data/
│   ├── unsafe_net/           # Downloaded videos
│   ├── processed/            # Extracted frames
│   └── splits/               # Train/val/test.txt
├── models/
│   ├── __init__.py
│   ├── yolov11_backbone.py   # Spatial encoder
│   ├── transformer_encoder.py # Temporal encoder
│   └── combined_model.py     # Full pipeline
├── training/
│   ├── __init__.py
│   ├── train.py
│   ├── eval.py
│   └── loss_functions.py
├── configs/
│   └── hyperparameters.yaml
├── results/
│   ├── logs/
│   ├── checkpoints/
│   └── metrics.csv
└── thesis/
    ├── chapters/
    ├── figures/
    └── references.bib
```

---

## Key Research Documents

1. **Full Research Report:** `/plans/research-report-unsafe-behavior-detection.md`
   - Complete citations for all techniques
   - Detailed explanations of SOTA methods
   - Literature review findings

2. **Strategy Summary:** `/plans/thesis-strategy-summary.md`
   - Actionable roadmap
   - Performance breakdown
   - Timeline estimates

3. **This Quick Guide:** `/plans/implementation-quick-guide.md`
   - Code templates
   - Weekly milestones
   - Implementation checklist

---

## When You Get Stuck

**Problem:** Accuracy stuck at 94-95%
- Solution: Add multi-task learning + temporal augmentation (together, not separately)

**Problem:** Overfitting on validation set
- Solution: Increase dropout to 0.5, weight decay to 0.1, add label smoothing 0.1

**Problem:** Can't get bbox annotations
- Solution: Skip detection task, use pre-trained YOLO as feature extractor only

**Problem:** Temporal augmentation not helping
- Solution: Verify it's implemented correctly (temporal crops, not spatial CutMix)

**Problem:** TTA makes results worse
- Solution: Lower number of augmentations (5-7 instead of 10), check for prediction conflicts

---

## Success Criteria

✅ Beat 95.81% baseline
✅ Pure YOLO+Transformer architecture (no ConvLSTM)
✅ Ablation study showing each component's contribution
✅ Cross-dataset validation showing generalization
✅ Computational efficiency documented
✅ Code reproducible and clean
✅ Thesis written and submitted

---

**Last Updated:** 2026-03-29
**Status:** Ready for implementation
