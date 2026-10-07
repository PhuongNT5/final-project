# Finding: iSafetyBench feasibility as a replacement/second dataset

Date: 2026-09-19. Zero Unsafe-Net test touches. Opened because the user authorised moving to a more
suitable dataset, and because `finding-2026-09-19-walkway-geometry-is-unsalvageable.md` closed the
last accuracy lever on Unsafe-Net.

All facts below were verified by the orchestrator directly against the live sources, **not** taken
from the researcher report (whose MMVIAD recommendation and licence advice are wrong — see the
correction banner on `researcher-2026-09-19-alternative-safety-datasets.md`).

## 1. What it is (verified)

- **iSafetyBench**, arXiv 2508.00399, ICCV 2025 Workshop (VISION'25). Project: isafetybench.github.io
- HF dataset `raiyaanabdullah/isafety-bench`. **Ungated, public, no request form.** 1.29 GB.
- 1,100 real `.mp4` files laid out as `hazard/` (420) and `normal/` (680) directories.
  **The binary Safe/Unsafe objective is native to this dataset** — no relabelling needed.
- Clips cut from publicly available YouTube videos. Licence **CC-BY-NC-SA-4.0**, stated as "for
  research, educational, and non-commercial purposes" — a Master's thesis qualifies directly.
- Paper's framing is a *zero-shot video-language benchmark* (98 routine + 67 hazardous open-vocab
  action tags, MCQ format). **No bounding boxes, no zone polygons, no official train/test split.**
  The HF card exposes a single 1,100-row `train` split with a 2-class `label` field.

Downloaded to `data/isafetybench/` (gitignored, 1.2 GB on disk, 1,100 files, 0 failures).

## 2. Clip properties (measured, 120-clip random sample)

| property | value |
|---|---|
| unreadable clips | **0 / 120** |
| duration | min 1.9 s, median 6.2 s, max 8.0 s |
| resolution | 1280x720 dominant; also 640x360, 854x480, and **portrait 360x640** |
| fps | 30 (60%), 60, 25, 24, 15 |
| T=16 frame sampling feasible | **yes, all clips** (Unsafe-Net pipeline's T) |

Mixed resolution and aspect ratio (portrait clips present) is real-world diversity, and the pipeline
uses normalised `xywhn` coords, so it is handled — but any absolute-pixel geometry step must be
re-derived per clip, not per view as Unsafe-Net does.

## 3. Detector feasibility (measured, COCO-pretrained yolo11s, 80 clips, 8 frames each)

| class | clips with >=1 person | persons/frame | objects/frame |
|---|--:|--:|--:|
| hazard | 90% | 1.32 | 2.51 |
| normal | 98% | 2.08 | 4.48 |

Top COCO classes: person (1089), suitcase (461), car (190), tv (79), truck (64), chair (40).

**The pipeline can run**: a COCO detector finds people in ~94% of clips, so per-entity relation tokens
are constructible. The project's own 8-class Unsafe-Net detector is useless here and must be replaced
by the COCO-pretrained model — a real change, but arguably a more general and more defensible one.

## 4. The confounding property — the whole reason for the switch

Filenames encode the source YouTube video id (`0kog0iuGxOc_trim_34.mp4`), so scene grouping is
recoverable **for free**, with no clustering heuristic (Unsafe-Net needed KMeans(8) on background
descriptors).

| | Unsafe-Net | iSafetyBench |
|---|--:|--:|
| clips | 691 | 1,099 parsed / 1,100 on disk |
| scene groups | **8** (KMeans scene groups) | **150** (distinct YouTube sources) |
| groups per class | `panel_closed`: **1** (fatal) | hazard 76, normal 74 |
| median clips per group | — | 3 (p75=7, p90=19) |
| **sources containing BOTH classes** | — | **0 / 150 (0%)** |

**Read this honestly: iSafetyBench has the SAME pathology.** No source video contributes both a
hazard and a normal clip, so scene appearance remains correlated with the label and a model can still
fit "this scene = hazard".

What changes is that it is **diluted, not cured**: 150 groups instead of 8, and no class is trapped in
a single group. On Unsafe-Net, `panel_closed`'s 0% leave-camera-out recall was *arithmetic* — all 32
clips lived in the held-out group. That failure mode cannot occur at 76 groups per class, and
leave-source-out CV becomes statistically powered rather than degenerate. Concentration is the
residual worry: the top 5 sources hold 24% of all clips, so grouped CV is mandatory and plain random
splits are meaningless.

Registered as **R41**.

## 5. The gate that decides this — R42

Section 3 contains a warning sign: persons/frame 1.32 (hazard) vs 2.08 (normal), objects/frame 2.51
vs 4.48. **A trivial object-count rule may separate the classes**, which is exactly risk R1 — the
finding that hollowed out the Unsafe-Net accuracy claim (carry and panel sub-tasks turned out to be
detector-class lookups).

Pre-registered decision rule, declared **before** the result is known:

> Run the count-rule baseline (persons/frame, objects/frame, mean confidence) under **leave-source-out**
> CV. If a 3-feature rule reaches within ~3 pp balanced accuracy of what the relation head can do,
> iSafetyBench is rejected as a headline dataset and the thesis stays on Unsafe-Net with the negative
> result. No encoder training on this dataset happens before this gate resolves.

This is the S1 discipline applied *before* sinking time, rather than after.

### GATE RESULT — PASSES (provisionally)

300-clip balanced probe (150 hazard / 150 normal, 97 distinct sources), 3 features
(persons/frame, objects/frame, mean detection confidence), logistic regression, class-weighted,
GroupKFold(10) over **source video**, pooled OOF:

| protocol | balanced accuracy |
|---|--:|
| **leave-source-out (grouped), pooled OOF** | **63.3%** (hazard recall 60.0%, normal recall 66.7%) |
| random stratified split, same rule | 67.0% (+3.7 pp) |
| majority-class baseline | 50.0% |

**The trivial detector-count shortcut reaches 63.3%.** That is well above chance but far from
saturating, so it does **not** reject the dataset — unlike Unsafe-Net, where the detector class label
alone gave 100/100 on the carry pair and the panel pair was a pure class lookup. There is real
headroom above the shortcut here.

Two things this also establishes:
- The grouped-vs-random gap is only **+3.7 pp**, so these count features are not heavily
  scene-memorising. On Unsafe-Net the equivalent gap was catastrophic.
- **It sets the bar.** Any encoder trained on iSafetyBench must clearly beat **63.3% BA under
  leave-source-out** or it has learned nothing beyond counting boxes. This number is now the
  pre-registered floor for every accuracy claim on this dataset, and must be printed beside it.

Caveats, stated plainly: n=300 of 1,100 (sampled, not full), 8 frames per clip (not T=16), one probe,
no confidence interval. It is a gate, not a result.

## 6. Recommendation (pending the R42 gate and a user decision)

Reassign the datasets rather than replace one with the other:

- **Accuracy -> iSafetyBench** (1,099 clips, 150 groups, leave-source-out). The only remaining place a
  positive accuracy result is possible.
- **Edge/Jetson -> stays Unsafe-Net.** Latency, energy and thermals are label-independent, so every
  ONNX/TRT artefact and the runbook survive the switch untouched. Zero rework.
- **Unsafe-Net -> the failure-analysis chapter.** Four findings (detector mAP measured for the first
  time; six encoders 18k-300k params giving identical per-sub-class profiles; camera-label
  confounding; geometry unsalvageable) form a stronger chapter than a headline number nobody believes.

Cost of the switch is bounded: the detector swaps to COCO-pretrained, the geometry channel goes to
zero (and geometry is already proven dead for walkway), and token-building is ~1,100 clips of
inference. Unsafe-Net work is already banked, so the downside is capped.

## Unresolved questions

1. **R42 gate result** — the count-rule baseline under leave-source-out. Everything in §6 is
   conditional on it. Appended when the probe lands.
2. Does the thesis keep a per-entity *relation* framing at all if geometry is unavailable on the new
   dataset? Without zone polygons the tokens carry class + box + appearance only, so "relation" would
   mean entity-to-entity, not entity-to-zone. That is a narrative change, not just an experiment.
3. Is the 420/680 class imbalance plus 150-group structure enough for stable leave-source-out CV, or
   does the 24%-in-top-5-sources concentration force grouped-stratified folds?
4. Not checked: whether any iSafetyBench source video also appears in Unsafe-Net (contamination
   across the two datasets). Near-zero prior — Unsafe-Net is bespoke IP-camera footage — but a
   cross-dataset transfer claim should state it was checked.
