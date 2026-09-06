# T2 - Deep research: the paper and related work

**Priority:** P1. **Status:** ready. **Owns:** `research/*`. Read-only + web.

## Overview
Two questions: (i) is our reading of Unsafe-Net right, and (ii) what does the field do that we are not
doing - especially for small-N video action recognition.

## Requirements
- R1 Re-verify the paper's method line by line against `paper/paper.txt`. Confirm or refute:
  - YOLO v4 trained on 2,262 frames, 8 behaviour classes (Table 3), used at nc=4, imgsz 416, batch 65.
  - Frame reduction + object-wise decomposition: one clip split into single-behaviour sub-videos (L325).
  - Whether that decomposition leaks the label into the input (our current reading: yes - the detector
    assigns the class ConvLSTM then "predicts"). This is the single most important claim to get right;
    the thesis's fairness argument rests on it.
  - ConvLSTM split 452/64 (L327), Tables 5-7 numbers.
- R2 Related work sweep, with citations:
  - Small-N industrial/worksite unsafe-behaviour recognition - what N, what protocol, what accuracy.
  - Modern video backbones that beat ConvLSTM at this scale: VideoMAE / VideoMAEv2, X3D, TimeSformer,
    Video Swin, and CLIP/ViFi-CLIP few-shot transfer. Report expected data requirements.
  - Papers reporting on <100 test clips - how do they handle statistical significance? Anyone reporting
    CIs? This supports the T1 (b) claim.
- R3 Evidence on Ultralytics MPS training instability (issue tracker, versions affected) - we should
  cite this properly, it is a reproducibility finding worth a paragraph.

## Implementation steps
1. `paper/paper.txt` full re-read; quote line numbers for every claim.
2. Web/lit search per R2 bullet; prefer 2023-2026 sources.
3. Write `research/paper-method-verification.md` and `research/related-work-and-alternatives.md`.
4. Recommend at most 2 concrete method upgrades that fit 452 training clips, with expected gain and cost.

## Todo
- [ ] Paper method verification w/ line cites
- [ ] Label-leakage question answered yes/no with evidence
- [ ] Related work table (method, N, protocol, accuracy)
- [ ] Small-N statistics practice survey
- [ ] Ultralytics MPS bug references
- [ ] <=2 ranked method recommendations

## Success criteria
T3 can start implementing from R3's recommendations without further reading; T1 can cite R2's statistics
survey to justify the claim it picks.

## Risks
- Recommending a data-hungry backbone (VideoMAE needs far more than 452 clips). Mitigation: every
  recommendation must state its training-set requirement explicitly.
