---
priority: parallel-track
effort: 10 person-days HARD CAP — clock STARTS NOW (2026-05-01); stop by 2026-05-11 regardless of result
depends_on: [phase-03 for framing context]
branches: A=pseudo-bbox improves accuracy, B=no improvement → publishable negative result
---

> **PARALLEL TRACK — runs alongside defense critical path.**
> Not on the 5-week defense critical path (Phases 01–03, 06–08). Work in spare cycles
> only; if defense materials slip, pause this phase. Negative-result Branch B is the
> realistic outcome and is itself thesis-worthy (ties to Phase 03 data-efficiency story).
>
> **Hard cap: 10 person-days from 2026-05-01.** Stop by **2026-05-11** even if mid-experiment.
> Output lands in §4.7 if completed; otherwise stays as Future Work in Ch 7.

# Phase 04 — Weakly-Supervised Pseudo-Bbox

## Context Links

- `PROGRESS.md` gap analysis — "discrimination requires zone boundary knowledge from per-frame bounding box annotations"
- `report/chapter-04-experiments-and-results.md` §4.5 — persistent error clips (9 clips; WalkwayViol ↔ UnauthorInt confusion)
- `sweep_results/pairwise/` — best ensemble checkpoints (direct4-s314, scenario-s7)
- `run_pairwise_contrastive.py` — training script with direct4 head; Grad-CAM hooks can be added here
- Phase 03 data-efficiency curve — contextualizes why pseudo-bbox might help at zero annotation cost

## Overview

The 9 persistent error clips all involve person-in-restricted-area confusion (WalkwayViol vs. UnauthorInt) which requires spatial zone knowledge not present in COCO features. This phase tests whether Grad-CAM saliency maps from the trained Direct4 head can bootstrap pseudo-bounding-boxes at zero manual annotation cost, training a lightweight scene-salience filter that mimics the paper's YOLO v4 frame-selection stage. The hard cap is 10 person-days. If accuracy does not improve, the negative result ties back to Phase 03's data-efficiency curve (pseudo-labels ≠ real labels) and is written up as a publishable finding.

**HARD CAP: Stop at day 10. Do not extend even if promising.**

## Key Insights

- 9 error clips are known; 8 of 9 are WalkwayViol ↔ UnauthorInt confusion (#1,2,29,30 and #37,38 from PROGRESS.md).
- Grad-CAM on a 4-class classifier trained on clip-level labels will produce attention maps for the predicted class, showing which spatial region the model attends to. If the model already partially attends to the zone boundary, pseudo-bboxes can reinforce this.
- The paper's YOLO v4 (§2.2) acts as a frame-selector: it removes frames where no unsafe-behavior-class object is detected, passing only relevant frames to ConvLSTM. A lightweight scene-salience filter mimics this without behavior-class labels.
- Risk: Grad-CAM on a frozen-backbone classifier tends to highlight the backbone's COCO-trained features (persons, objects) rather than zone boundaries. This is likely to produce noisy pseudo-bboxes for the WalkwayViol/UnauthorInt confusion cases precisely because the backbone does not encode zone information.
- Negative result hypothesis (Branch B): "pseudo-labels derived from a model that cannot distinguish zones cannot bootstrap zone-aware detection." This is a publishable hypothesis-test, and directly supports the Phase 03 curve (more real labels = only solution).

## Requirements

**Functional**
- Generate Grad-CAM saliency maps for each training clip using trained direct4-s314 checkpoint.
- Convert saliency maps to pseudo-bounding-boxes: threshold at 50th percentile of activation, bounding-box around the largest contiguous region.
- Train a lightweight scene-salience filter (YOLOv11n or a simpler CNN regressor) on pseudo-bbox annotations.
- Re-run full pipeline with scene-salience-filtered frames as input to backbone.
- Report macro accuracy on 64-clip test set; compare to 89.28 % baseline.

**Non-functional**
- All Grad-CAM computation ≤ 8 h (16 spatial tokens × 32 frames × 452 training clips).
- Scene-salience filter ≤ YOLOv11n size (~3 MB); must run in < 5 ms per frame on M4 CPU.
- Hard stop at 10 person-days; no code refactoring beyond what is needed.

**Acceptance thresholds (Branch A)**
- Macro accuracy ≥ 89.5 % (> 56/64 = marginally better than baseline).
- Saliency filter adds ≤ 10 ms to per-frame latency.

**Acceptance thresholds (Branch B — negative result)**
- Report: (a) quality of pseudo-bboxes (IoU vs. any available ground-truth, or qualitative assessment), (b) accuracy delta vs. baseline, (c) interpretation tied to Phase 03 curve.

## Architecture / Approach

### Step 1 — Grad-CAM on Direct4 Head

```python
# Hook: capture gradient w.r.t. backbone output (T, 16, 512) for target class
from pytorch_grad_cam import GradCAM  # or manual hook
target_layer = model.backbone.layer22  # C3k2 output before pool
cam = GradCAM(model=direct4_head, target_layers=[target_layer])
# For each clip: forward pass on 32 frames, compute CAM per frame
# Output: (T, H, W) saliency maps per clip, H=W=7 (pre-pool spatial grid)
```

### Step 2 — Pseudo-Bbox Generation

- Upsample CAM to 224×224 via bilinear interpolation.
- Threshold at P50 activation.
- Largest connected component → bounding-box coordinates.
- Save as YOLO-format `.txt` files: `class_id cx cy w h` (use Direct4 predicted class as class_id).

### Step 3 — Lightweight Salience Filter

- Train YOLOv11n (fastest YOLO variant) on pseudo-bbox annotations (452 training clips × 32 frames = 14,464 images).
- Expect poor mAP (pseudo-labels are noisy); goal is frame-selection, not accurate detection.
- Threshold detection confidence at 0.3; retain frames with any detection.

### Step 4 — Pipeline Integration

- Insert salience filter before backbone in pipeline: for each frame, run YOLOv11n; if no detection, skip frame or replace with zero token.
- Re-extract backbone features for retained frames; pad skipped frames with mean token.
- Re-train direct4 + cascade heads on new features (5-seed sweep).
- Report accuracy.

### Branch B Write-Up

If accuracy ≤ 89.28 %:
- Quantify: what % of pseudo-bboxes overlap with the actual confusing regions (WalkwayViol / UnauthorInt)?
- Hypothesis confirmation: pseudo-bbox quality for zone-boundary classes is insufficient because the trained classifier has no zone representation.
- Connection to Phase 03: "Phase 3 shows that even 400 real bboxes degrade performance; pseudo-bboxes derived from a model trained on clip labels cannot substitute for zone-annotated ground truth."
- Write 1-page discussion for §4.7 (new section) framing as confirmatory evidence for the data-efficiency hypothesis.

## Related Code Files

| Action | Path |
|--------|------|
| Modify | `run_pairwise_contrastive.py` — add Grad-CAM extraction mode (`--mode gradcam`) |
| Create | `run_pseudo_bbox.py` — orchestrates Steps 1–3 |
| Create | `run_salience_pipeline.py` — Step 4: re-train with salience filter |
| Create | `figures/gradcam-samples.png` — 4×2 grid of CAM overlays for each class (thesis figure) |
| Modify | `report/chapter-04-experiments-and-results.md` — add §4.7 (pseudo-bbox experiment) |

## Implementation Steps

1. **Add Grad-CAM hook** to `run_pairwise_contrastive.py`: use `torch.autograd.grad` or `register_hook` on the backbone output; compute gradient w.r.t. target class logit; average over channels.
2. **Extract CAMs for all 452 training clips**: forward all 32 frames of each clip, compute CAM per frame, save as `.npy` tensors to `pseudo_bbox/cams/`.
3. **Convert CAMs to pseudo-bboxes**: threshold at P50, find largest contiguous region using `scipy.ndimage.label`, save YOLO-format annotations to `pseudo_bbox/labels/`.
4. **Visual sanity check**: generate `figures/gradcam-samples.png` — 4 classes × 2 samples, overlay CAM heatmap on frame. Assess qualitatively whether activations are near the discriminating region.
5. **If CAMs are clearly wrong** (e.g., all activations at image center): stop at day 3, write Branch B; do not proceed with training.
6. **Train YOLOv11n salience filter** on pseudo-bbox annotations: 30 epochs, img-size=224, batch=32. Log mAP (even if low, document it).
7. **Extract new features** with salience-filtered frames: for each clip, run YOLOv11n, retain detected-frame backbone tokens, pad gaps.
8. **Retrain heads** (5-seed sweep): use same protocol as Phase 06 for consistency.
9. **Evaluate on 64-clip test set**: report macro accuracy; per-class breakdown; specifically check WalkwayViol and UnauthorInt accuracy.
10. **Write §4.7**: Branch A (if improved) — describe pseudo-bbox pipeline, accuracy gain, latency cost. Branch B (if not improved) — describe negative result, hypothesis confirmation, connection to data-efficiency curve.

## Todo List

- [ ] 1. Add Grad-CAM hook to run_pairwise_contrastive.py
- [ ] 2. Extract CAMs for all 452 training clips
- [ ] 3. Convert CAMs to pseudo-bbox YOLO annotations
- [ ] 4. Generate gradcam-samples.png for visual sanity check
- [ ] 5. Decision gate: if CAMs clearly wrong, skip to step 10 (Branch B write-up)
- [ ] 6. Train YOLOv11n salience filter on pseudo-bbox annotations
- [ ] 7. Extract new features with salience-filtered frames
- [ ] 8. Retrain heads (5-seed sweep)
- [ ] 9. Evaluate on 64-clip test set; compare to 89.28% baseline
- [ ] 10. Write §4.7 (Branch A: improvement narrative / Branch B: negative result narrative)

## Success Criteria (context-dependent)

**Branch A**: macro accuracy ≥ 89.5 % on 64-clip test set; §4.7 describes the pipeline and gain.

**Branch B**: §4.7 documents the negative result with: (a) CAM quality assessment, (b) accuracy delta, (c) explicit hypothesis statement, (d) connection to Phase 03 data-efficiency curve. The section is framed as a publishable finding, not a failure.

**Both branches**: Grad-CAM visualization figure exists; experiment is reproducible from a single script.

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Grad-CAM hook incompatible with ONNX-exported model | High | Use PyTorch checkpoint (not ONNX) for Grad-CAM; ONNX path is only for deployment |
| pseudo-bbox quality so low that salience filter mAP < 5% | High | Document as Branch B; this is the expected outcome per gap-analysis reasoning |
| YOLOv11n training on 14k noisy images > 10h | Medium | Reduce epochs to 10; use img-size=128; the goal is salience filter, not high-quality detector |
| Day 10 cap reached mid-experiment | Medium | Enforce hard stop; if in Step 7–9, complete evaluation on partial seeds; report what is done |
| Accuracy improves but only marginally (+0.5 pp) | Low | Report accurately; still publishable; combine with Phase 03 curve for a stronger story |

## Security / Ethics / Privacy

- All data is from the Unsafe-Net ethics-approved corpus. No new data collection.
- Pseudo-bbox files derived from model activations: these are derivative research artifacts, not human annotations. Clearly labeled as such in the thesis.

## Dependencies + Next Steps

- Phase 03 output (data-efficiency curve) provides the framing context for Branch B write-up.
- Output consumed by: **Phase 07** (§4.7 new section; Ch6 Discussion references pseudo-bbox attempt).
- Landing chapter: `report/chapter-04-experiments-and-results.md` §4.7 (new).

**Unresolved questions**
- Does the current `run_pairwise_contrastive.py` expose the backbone's layer-22 output as a hookable module attribute, or is it abstracted inside the backbone class? (This determines Grad-CAM hook complexity.)
- Is `pytorch-grad-cam` installed in the project venv? If not, add to requirements.txt.
