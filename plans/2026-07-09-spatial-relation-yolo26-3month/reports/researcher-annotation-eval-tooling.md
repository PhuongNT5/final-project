# Annotation & Evaluation Tooling — Research Report
**Date:** 2026-07-09  
**Scope:** CVAT workflow, zone supervision, evaluation rigor for thesis 3-month plan  
**Thesis context:** ~3,000 annotated frames, 6-8 fixed camera views, YOLO→Transformer pipeline, 64-clip test set, 4 unsafe classes

---

## 1. CVAT WORKFLOW

### 1.1 Multi-class Bbox + Static-zone Annotation — Fixed Views

**Project vs. Task structure:**  
Create one CVAT **Project** (shared label set) with one **Task per camera view** (or per clip batch). This enforces a consistent label map across all annotators. Label schema:
```
Person_Authorized     → bbox
Person_Unauthorized   → bbox
Forklift_Safe_Load    → bbox
Forklift_Overload     → bbox
Panel_Open            → bbox  (or tag on frame if no spatial extent needed)
Panel_Closed          → bbox
Safe_Walkway          → polygon  (static track)
Restricted_Area       → polygon  (static track)
Machine_Area          → polygon  (static track)
```

**Reusing zone polygons across frames (static zones):**  
CVAT's **Track mode** supports polygon tracks with keyframe interpolation. For a painted walkway that never moves:
1. On frame 0 of the clip, draw the walkway polygon in Track mode → creates a track with `track_id`.
2. Add one keyframe at the clip's last frame — same polygon coordinates (no movement needed).
3. CVAT interpolates linearly between keyframes → the polygon is effectively pinned to every frame.
4. For a fully static zone: set the polygon on frame 0, then navigate to the last frame and mark it as a keyframe with "outside=false". No redrawing required.
5. This means zone polygons are drawn **once per view**, not once per frame. Estimated effort: ~5 min per zone per view, ~30–45 min total for 3 views × 3 zones.

**Per-view task isolation:** Because zone shape differs per viewpoint, keep each camera's footage in its own CVAT task. Copy the same project label schema. Zone tracks persist within a task — they don't need to be redrawn for each clip if the task covers multiple clips from the same camera.

Refs: [CVAT docs — Track mode](https://www.cvat.ai/academy/track-mode) | [Polygon & Polyline in CVAT](https://www.cvat.ai/academy/polygon-and-polyline-annotation) | [Accelerate with interpolation](https://www.cvat.ai/resources/blog/accelerate-and-automate)

---

### 1.2 Interpolation / Tracking for Temporal Consistency

**Track mode mechanics:**
- Switch annotation mode to **Track** before drawing an object.
- CVAT assigns a `track_id` (integer) to the object.
- Mark keyframes at positions where the object changes (entry, occlusion end, exit).
- CVAT linearly interpolates bbox position and size between keyframes.
- Mark an object as `outside=true` during full occlusion — it disappears from interpolation until the next keyframe.

**SAM 2 auto-tracking (CVAT cloud ≥ 2.16):**  
CVAT now integrates Segment Anything Model 2 for automatic video object tracking. For a person walking through frame, one click + SAM 2 propagates the mask/bbox across subsequent frames. Reduces correction load dramatically on moving objects.  
Ref: [CVAT SAM 2 integration](https://www.cvat.ai/resources/changelog/video-annotation-sam-2)

**Exported track_id in YOLO Ultralytics format:**  
`label_id cx cy rw rh track_id` — consistent IDs across frames of a clip, enabling temporal analysis.

---

### 1.3 Throughput & Effort Estimate

**Raw annotation speed benchmarks (literature):**
- Fully manual bbox: ~8–16 s/box (LabelMe-class tools). With CVAT auto-hints/prelabeling: ~3–5 s/box.
- With Track+interpolation (only keyframes drawn): effective rate ~2–4 keyframe boxes + interpolation ≈ **40–80 effective boxes/min** on predictable motion.
- Industry pricing proxy: $0.02–$0.04/box manual → implies ~100–200 boxes/hour at $4/hr labour rate floor; at $20/hr (thesis annotator), ~50–100 boxes/hour.

**Thesis-specific estimate (4–6 boxes/frame, no prelabeling):**

| Mode | Rate | 3,000 frames × 5 avg boxes = 15,000 boxes |
|------|------|-------------------------------------------|
| Manual, no tracking | 50 boxes/hr | 300 h |
| Track+interpolation (keyframe ~1 in 5) | 150 boxes/hr effective | 100 h |
| Track + SAM 2 assisted | 200–300 boxes/hr | 50–75 h |

**Recommendation:** Use Track mode + SAM 2 for persons/forklifts (moving), static polygon track for zones. Realistic annotation time for 3,000 frames: **60–100 person-hours** (1 annotator = 2–3 weeks at 4 h/day; 2 annotators = 1–1.5 weeks).

**Inter-annotator agreement (IAA):**
- Draw a 100-frame quality sample; have two annotators label it independently.
- Compute **bbox IoU-based kappa**: for each frame, match predicted and GT boxes at IoU ≥ 0.5, score class agreement. Cohen's κ > 0.7 = acceptable; > 0.8 = good.
- CVAT has a built-in **Quality Assurance** module (Tasks → Quality → Report) that computes precision/recall/F1 against a "ground truth" job. Set one annotator as GT reviewer.
- For zone polygons, use polygon IoU (Shapely intersection/union); κ > 0.85 expected since zones are drawn once.

```python
# Quick kappa proxy via scikit-learn
from sklearn.metrics import cohen_kappa_score
# class_labels_annotator_A, class_labels_annotator_B  (per-box matched labels)
kappa = cohen_kappa_score(labels_A, labels_B)
```

Ref: [CVAT Quality Assurance blog](https://www.cvat.ai/resources/blog/annotation-quality-assurance) | [IAA guide](https://keymakr.com/blog/measuring-inter-annotator-agreement-building-trustworthy-datasets/)

---

### 1.4 Export Formats

**CVAT → YOLO (Ultralytics):**
1. Task → Actions → Export dataset → format: **Ultralytics YOLO 1.0**
2. Output `.zip` structure:
```
dataset.zip/
├── data.yaml          # path, train, val, names: {0: Person_Authorized, ...}
├── images/train/      # frames (jpg/png)
├── images/val/
├── labels/train/      # one .txt per frame: "label_id cx cy rw rh"
└── labels/val/
```
3. `data.yaml` minimal example:
```yaml
path: /data/safety_dataset
train: images/train
val: images/val
names:
  0: Person_Authorized
  1: Person_Unauthorized
  2: Safe_Walkway
  3: Restricted_Area
  4: Panel_Open
  5: Panel_Closed
  6: Forklift_Safe_Load
  7: Forklift_Overload
```
4. Zones (polygon tracks) export as segmentation polygons in YOLO seg format. If you only need bbox for YOLO training, export zones as rectangular bboxes or handle separately.

**CVAT → COCO:**
- Export format: **COCO 1.0** → `.zip` with `instances_train.json`, `instances_val.json`
- JSON structure: standard COCO `images`, `categories`, `annotations` arrays; polygon zones export as `segmentation` lists; bbox objects as `bbox: [x, y, w, h]`.
- `track_id` preserved as custom attribute in annotations dict.

**COCO → YOLO conversion (Ultralytics helper):**
```python
from ultralytics.data.converter import convert_coco
convert_coco(labels_dir='path/to/coco/annotations/', save_dir='path/to/yolo/')
```
Ref: [CVAT YOLO export docs](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/) | [CVAT COCO docs](https://docs.cvat.ai/docs/dataset_management/formats/format-coco/) | [Ultralytics COCO→YOLO](https://docs.ultralytics.com/guides/coco-to-yolo)

---

### 1.5 Sampling Strategy: 3,000 Frames from 691 Clips

**Goal:** Break WalkwayViolation ↔ UnauthorizedIntervention confusion; cover all viewpoints; include hard cases.

**Step 1 — Per-viewpoint quota:**
6–8 fixed cameras → allocate ~400–500 frames per camera (ensures multi-lens balance).

**Step 2 — Class-balanced sampling:**
```
Target distribution across 3,000 frames (suggested):
  Person_Authorized presence:        ~1,800 frames (60%)
  Person_Unauthorized presence:      ~1,200 frames (40%)  ← over-sample vs. natural rate
  WalkwayViolation co-occurrence:    ~600 frames  (20%)
  UnauthorizedIntervention:          ~600 frames  (20%)
  Panel events (Open/Closed):        ~400 frames  (include both states)
  Forklift (Safe + Overload):        ~400 frames  (balance 50/50)
```

**Step 3 — Hard-case mining to resolve WalkwayViolation↔UnauthorizedIntervention confusion:**
- Run YOLO zero-shot or base detector on all 691 clips.
- Collect frames where the model is uncertain (max softmax < 0.6) or confused (top-2 classes are WalkwayViolation and UnauthorizedIntervention).
- Reserve 300–400 slots in the 3,000-frame budget for these confusion frames.

**Step 4 — Temporal spacing within clips:**
- Sample every ~10th frame from a clip (not every frame — redundancy adds cost, not signal).
- For clips containing unsafe behaviour, sample the unsafe window densely (every 3rd frame); safe buffer at 1 in 15.

**Step 5 — Clip-level stratification:**
- At least 5 clips per unsafe class in the annotation pool (not just the 64-clip test set).
- Ensure ≥1 annotated clip per camera per unsafe class.

---

## 2. ZONE / VIRTUAL-FENCE SUPERVISION

### 2.1 Foot-Point-in-Polygon (Recommended Primary Signal)

**Logic:** For each detected person bbox `(x1,y1,x2,y2)`, compute foot-point = `(cx, y2)` (bottom-center). Check if foot-point falls inside the zone polygon.

```python
from shapely.geometry import Point, Polygon

# Zone polygon coordinates (pixel space, per-view, drawn once)
walkway = Polygon([(x1,y1), (x2,y2), ...])  # from CVAT polygon export

def is_in_walkway(bbox):
    cx = (bbox[0] + bbox[2]) / 2
    fy = bbox[3]  # bottom y = foot point
    return walkway.contains(Point(cx, fy))
```

- `shapely.geometry.Polygon.contains(Point)`: O(log n) after first construction.
- Batch version: `shapely.vectorized` or `geopandas` for frame-level vectorisation.

**Advantages over IoU:** Foot-point avoids false positives from persons standing outside a zone but whose bbox overlaps due to camera angle.

### 2.2 IoU(bbox, zone) — Secondary Signal

```python
from shapely.geometry import box as shapely_box

def bbox_zone_iou(bbox, zone_polygon):
    person_box = shapely_box(bbox[0], bbox[1], bbox[2], bbox[3])
    inter = person_box.intersection(zone_polygon).area
    union = person_box.union(zone_polygon).area
    return inter / union if union > 0 else 0.0
```

Use IoU > 0.1 as a softer "proximity" signal; use foot-point for binary violation flag.

### 2.3 Zone Annotation: Polygon vs. Bbox

**Use polygon (segmentation), not bbox** for painted walkways:
- Walkways are rarely rectangular — painted paths on factory floors are often L-shaped or curved.
- Polygon gives exact boundary → foot-point test is precise.
- CVAT exports polygons in COCO `segmentation` format; read with `pycocotools.mask` or shapely directly.
- Optional: add a thin rectangular bbox around the zone polygon for YOLO detection training (if you want the detector to locate zones at runtime rather than using fixed per-view coords).

**Homography (optional):** For birds-eye-view consistency across zoom/pan, project person foot-points to a ground-plane via a pre-computed homography (4-point calibration per camera using known floor markings). Only needed if camera can pan/tilt; for fixed surveillance cameras, pixel-space polygons are sufficient.

---

## 3. EVALUATION RIGOR

### 3.1 Corruption Robustness

**Library:** `imagecorruptions` (pip install imagecorruptions) — direct port of Hendrycks & Dietterich ImageNet-C.

**Recommended corruption suite for surveillance video:**

| Category | Corruption | Severity levels (1–5) |
|----------|-----------|----------------------|
| Sensor noise | gaussian_noise, shot_noise | 1,3,5 |
| Compression | jpeg_compression | 1,3,5 |
| Codec/stream | h264_abr_corruption (albumentations ImageCompression) | 1,3,5 |
| Blur | motion_blur, defocus_blur | 1,3 |
| Lighting | brightness, contrast | 1,3,5 |
| Weather | fog (if outdoor zones) | 1,3 |

**Total test points:** 6 corruption types × 3 severity levels × 64 clips = 1,152 corrupted test cases.

```python
from imagecorruptions import corrupt
import numpy as np

def apply_frame_corruption(frame_np, corruption_name, severity):
    # frame_np: H×W×3 uint8
    return corrupt(frame_np, corruption_name=corruption_name, severity=severity)

# For JPEG/H.264 compression use albumentations:
import albumentations as A
transform = A.ImageCompression(quality_lower=20, quality_upper=40, p=1.0)
```

**Report metric:** Mean Corruption Error (mCE) = mean over all corruption types of (model_error / AlexNet_baseline_error). For a thesis, just report per-corruption accuracy drop vs. clean baseline.

Refs: [Hendrycks & Dietterich 2019 — ImageNet-C](https://arxiv.org/pdf/1903.12261) | [Video robustness benchmarking](https://arxiv.org/pdf/2110.06513) | [imagecorruptions PyPI](https://github.com/bethgelab/robustness)

---

### 3.2 TTA for Video Classification

**What works on small N (64 clips):**
- Horizontal flip: average logits of original + flipped clip. Expected gain: +0.5–2 pp on balanced datasets.
- Multi-crop (3–5 temporal crops): sample T=32 frames from different starting offsets → average softmax. Reduces temporal variance.
- Multi-scale spatial crops (224, 256, 288): marginal gain if model already trained at 224.

**What does NOT help / may hurt on N=64:**
- Gaussian noise TTA: adds calibration noise; evidence from video benchmarks shows it can degrade temporal consistency.
- Excessive TTA (>10 augments): variance reduction plateaus quickly at n<100 test samples; p95 accuracy gain saturates at ~3–5 augments.

**Evidence:** VideoMix (2020) and temporal augmentation surveys (2022) show 5–12% benefit on UCF-101/HMDB-51 (small datasets). On 64 clips, observed gain in thesis: +1–3 clips in typical run.

**Ultralytics TTA (for YOLO detection stage):**
```python
results = model.predict(source, augment=True)  # flip+scale TTA built-in
```
Ref: [Ultralytics TTA docs](https://docs.ultralytics.com/yolov5/tutorials/test-time-augmentation)

---

### 3.3 Calibration: ECE + Reliability Diagram

```python
pip install netcal
```

```python
import numpy as np
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
from netcal.scaling import TemperatureScaling

# confidences: shape (N, num_classes) softmax outputs
# ground_truth: shape (N,) integer class labels
confidences = np.array(...)  # from model inference on 64 clips
ground_truth = np.array(...)

# Measure ECE (15 bins standard for small N)
ece = ECE(n_bins=15)
print(f"ECE (uncalibrated): {ece.measure(confidences, ground_truth):.4f}")

# Temperature scaling (fit on val set, evaluate on test)
temp = TemperatureScaling()
temp.fit(val_confidences, val_labels)
calibrated = temp.transform(confidences)
print(f"ECE (calibrated): {ece.measure(calibrated, ground_truth):.4f}")

# Reliability diagram
diagram = ReliabilityDiagram(n_bins=15)
diagram.plot(calibrated, ground_truth)
```

**For per-class reliability:** ECE per-class = run above with one-vs-rest binary ground_truth per class.

Refs: [netcal GitHub](https://github.com/EFS-OpenSource/calibration-framework) | [Guo et al. 2017 calibration paper](https://arxiv.org/pdf/1706.04599)

---

### 3.4 Cost-Sensitive / Safety Metrics

**Per-class FNR (miss-rate):** Most critical metric for safety system.
```python
from sklearn.metrics import confusion_matrix
import numpy as np

cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
FNR_per_class = []
for i in range(n_classes):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FNR_per_class.append(FN / (TP + FN + 1e-9))
# Report: FNR for WalkwayViolation, UnauthorizedIntervention, etc.
```

**Cost matrix (suggested for thesis):**
```
                  Pred: Safe   Pred: Unsafe
Actual: Safe         0           1   (false alarm — nuisance)
Actual: Unsafe      10           0   (miss — safety risk, penalize 10×)
```
Weighted accuracy = sum of cost-matrix-weighted confusion entries / total.

**Threshold sweep (per-class):**
```python
from sklearn.metrics import roc_curve, precision_recall_curve, auc

fpr, tpr, thresholds = roc_curve(y_binary, scores)
roc_auc = auc(fpr, tpr)

prec, rec, thresholds_pr = precision_recall_curve(y_binary, scores)
pr_auc = auc(rec, prec)

# Safety operating point: pick threshold where recall (TPR) >= 0.95
safety_threshold = thresholds[np.argmax(tpr >= 0.95)]
```

---

### 3.5 Statistical Testing

**A. 5,000-sample bootstrap CIs (per-class accuracy):**
```python
import numpy as np

def bootstrap_accuracy_ci(y_true, y_pred, n_boot=5000, ci=0.95):
    n = len(y_true)
    boot_accs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        boot_accs.append(np.mean(y_true[idx] == y_pred[idx]))
    lo = np.percentile(boot_accs, (1-ci)/2*100)
    hi = np.percentile(boot_accs, (1+ci)/2*100)
    return lo, hi

# BCa CI: use scipy.stats.bootstrap (method='BCa')
from scipy.stats import bootstrap
result = bootstrap((y_true == y_pred,), np.mean, n_resamples=5000, method='BCa')
```

**B. McNemar exact test (paired model comparison on 64 clips):**
```python
from statsmodels.stats.contingency_tables import mcnemar

# Contingency table: discordant pairs
# [A correct, B wrong] | [A wrong, B correct]
# [A wrong, B wrong]   | [A correct, B correct]
table = [[b, c], [d, a]]  # standard McNemar form: b=A_right_B_wrong, c=A_wrong_B_right
result = mcnemar(table, exact=True)  # exact binomial for small N (64 clips)
print(f"p-value: {result.pvalue:.4f}")
```
Use `exact=True` for n<25 discordant pairs (typical for 64-clip test set).

**C. Nadeau-Bengio corrected t-test (for k-fold results):**
```python
pip install correctipy  # hendersontrent/correctipy
from correctipy import kfold_ttest

# For k-fold cross-validation comparison
result = kfold_ttest(scores_model_A, scores_model_B, k=5, n=len(dataset))
print(f"Corrected p-value: {result['p_value']:.4f}")
```
Or via scikit-learn example:
```python
# sklearn docs: "Statistical comparison of models using grid search"
# https://scikit-learn.org/1.5/auto_examples/model_selection/plot_grid_search_stats.html
from scipy.stats import t

def nadeau_bengio_ttest(diff_scores, k, n_train, n_test):
    n = len(diff_scores)
    mean_diff = np.mean(diff_scores)
    var = np.var(diff_scores, ddof=1)
    corrected_var = var * (1/n + n_test/n_train)
    t_stat = mean_diff / np.sqrt(corrected_var)
    p_val = 2 * t.sf(abs(t_stat), df=n-1)
    return t_stat, p_val
```
Refs: [correctipy GitHub](https://github.com/hendersontrent/correctipy) | [sklearn stats comparison example](https://scikit-learn.org/1.5/auto_examples/model_selection/plot_grid_search_stats.html) | [Nadeau & Bengio 2003](https://www.researchgate.net/publication/227049463_Inference_for_the_Generalization_Error)

**Wilson CI (for per-class proportions):**
```python
from statsmodels.stats.proportion import proportion_confint
lo, hi = proportion_confint(count=correct, nobs=total, alpha=0.05, method='wilson')
```

---

### 3.6 Inference Timing Methodology

**Template matching paper's "0.14 s/clip" and "1.87 s reduced duration" metrics:**

```python
import time
import numpy as np

WARMUP = 10   # clips
MEASURE = 64  # clips (full test set)

# Warmup — stabilize JIT, cache, CPU freq
for clip in test_clips[:WARMUP]:
    _ = model.predict(clip)

# Measurement
latencies = []
for clip in test_clips[:MEASURE]:
    t0 = time.perf_counter()
    _ = model.predict(clip)
    t1 = time.perf_counter()
    latencies.append(t1 - t0)

latencies = np.array(latencies)
print(f"p50: {np.percentile(latencies, 50)*1000:.1f} ms")
print(f"p95: {np.percentile(latencies, 95)*1000:.1f} ms")
print(f"p99: {np.percentile(latencies, 99)*1000:.1f} ms")
print(f"mean: {latencies.mean()*1000:.1f} ms  (= {1/latencies.mean():.1f} FPS)")
```

**For "reduced duration" (1.87 s metric):** Time the YOLO frame-gate (clips skipped) vs. full inference path; report mean ± std of per-clip wall time, then sum across test set to show total pipeline time reduction.

**CPU pinning for reproducibility:**
```bash
taskset -c 0-3 python benchmark.py   # pin to 4 cores, avoid NUMA effects
```

Refs: [NVIDIA TensorRT benchmarking guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html) | [p50/p95/p99 methodology](https://aerospike.com/blog/what-is-p99-latency/)

---

## EFFORT SUMMARY TABLE

| Task | Tool | Estimated Time |
|------|------|----------------|
| Zone polygon annotation (6–8 views × 3 zones) | CVAT polygon track | 2–3 h |
| Bbox annotation, 3,000 frames, Track+SAM2 | CVAT | 60–100 h (1–2 annotators) |
| IAA quality check (100 frames × 2 annotators) | CVAT QA module + sklearn kappa | 4–6 h |
| Export + convert to YOLO/COCO | CVAT export + ultralytics converter | 0.5 h |
| Sampling strategy script | Python (pandas + YOLO inference pass) | 4–8 h |
| Corruption robustness eval (1,152 test cases) | imagecorruptions | 2–4 h coding + GPU run |
| Calibration (ECE + reliability diagram) | netcal | 1–2 h |
| Bootstrap CIs + McNemar + Nadeau-Bengio | scipy/statsmodels/correctipy | 2–3 h |
| Timing benchmark | time.perf_counter pattern | 1 h |
| **Total (excluding annotation)** | | **~15–25 h** |

---

## UNRESOLVED QUESTIONS

1. **CVAT self-hosted vs. cloud:** SAM 2 integration is cloud-CVAT; self-hosted requires manual SAM 2 plugin setup. Confirm which deployment is used.
2. **Zone polygon runtime:** Are zone polygons used only at training time (supervision label) or also at inference time (runtime rule engine)? If runtime, need per-view homography or fixed pixel coordinates bundled with deployment artifact.
3. **WalkwayViolation exact definition:** Is a violation `Person_Unauthorized foot-point outside Safe_Walkway` OR `any person inside Restricted_Area`? Definition affects annotation guidelines and sampling strategy.
4. **691 clips distribution across 4 unsafe classes:** If class counts are severely imbalanced (e.g., <20 Forklift_Overload clips), the 3,000-frame budget may need reallocation — confirm raw clip counts per class before finalizing sampling quotas.
5. **McNemar with n_discordant < 10:** On a 64-clip test set, if two models differ on only 6–8 clips, exact McNemar will return p > 0.05 regardless — this is an underpowered test and should be reported as such.

---

*Sources:*
- [CVAT Track Mode Academy](https://www.cvat.ai/academy/track-mode)
- [CVAT SAM 2 integration](https://www.cvat.ai/resources/changelog/video-annotation-sam-2)
- [CVAT YOLO Ultralytics export](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/)
- [CVAT COCO export](https://docs.cvat.ai/docs/dataset_management/formats/format-coco/)
- [Ultralytics COCO→YOLO converter](https://docs.ultralytics.com/guides/coco-to-yolo)
- [CVAT Quality Assurance blog](https://www.cvat.ai/resources/blog/annotation-quality-assurance)
- [imagecorruptions / Hendrycks & Dietterich ImageNet-C (2019)](https://arxiv.org/pdf/1903.12261)
- [Video robustness benchmarking (2021)](https://arxiv.org/pdf/2110.06513)
- [netcal calibration framework](https://github.com/EFS-OpenSource/calibration-framework)
- [Guo et al. — On Calibration of Modern Neural Networks (2017)](https://arxiv.org/pdf/1706.04599)
- [correctipy — Nadeau-Bengio corrected t-test](https://github.com/hendersontrent/correctipy)
- [scikit-learn stats model comparison](https://scikit-learn.org/1.5/auto_examples/model_selection/plot_grid_search_stats.html)
- [Nadeau & Bengio 2003 original paper](https://www.researchgate.net/publication/227049463_Inference_for_the_Generalization_Error)
- [Shapely 2.0 docs](https://shapely.readthedocs.io/en/2.0.6/reference/shapely.intersection.html)
- [Ultralytics TTA](https://docs.ultralytics.com/yolov5/tutorials/test-time-augmentation)
- [NVIDIA TensorRT benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)
