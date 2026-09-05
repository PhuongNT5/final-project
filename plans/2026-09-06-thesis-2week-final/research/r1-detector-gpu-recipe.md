# R1 — GPU detector recipe for the behaviour detector (A1)

Verified 2026-09-06. Env: ultralytics 8.4.27 (installed, already ships `cfg/models/26/yolo26*.yaml` —
no upgrade needed), torch 2.11.0.

## 0. CRITICAL — task brief conflates two different detectors. Read this first.

The brief (`brief.md`, `kaggle-detector-retrain/README.md`) describes **one** "the detector" but the
repo has **two**, trained on different label schemes:

| | Behaviour detector | Entity detector |
|---|---|---|
| Script | `train_behaviour_detector.py` | `kaggle-detector-retrain/train-detector-gpu.py`, `colab/kaggle_yolo26_onecell.py` |
| Dataset | `annotation_guide/dataset_behaviour_{tight,ctx}/` | `annotation_guide/dataset/` |
| Classes | WalkwayViol, UnauthorInt, OpenedPanel, CarryOverload, SafeWalkway, AuthorInt, ClosedPanel, SafeCarry (paper Table 3 scheme) | Person_Authorized, Person_Unauthorized, Safe_Walkway, Restricted_Zone, Panel_Closed, Panel_Opened, Forklift_Safe_Load, Forklift_Overload |
| Measured mAP50 | **0.791 (tight) / 0.806 (ctx)** — `plans/2026-09-05-thesis-final-push/reports/t1-objectives-audit.md` §1c | not yet retrained on GPU |
| "3 classes never emitted" claim | **not evidenced** — verified label counts below | **this is where the claim comes from** — `kaggle-detector-retrain/README.md` |

Verified label counts, `dataset_behaviour_tight` (train+val, 5263 boxes total, none zero):
`WalkwayViol 1634, UnauthorInt 1630, ClosedPanel 1181, OpenedPanel 525, SafeWalkway 397, AuthorInt 390,
CarryOverload 195, SafeCarry 170`. So the behaviour detector has **no structurally-dead class** —
worst imbalance is ~9.6:1 (WalkwayViol:SafeCarry). Whether it has zero *recall* (not zero *labels*) at
inference is unknown — no per-class `val()` printout was ever saved for the beh_tight/beh_ctx runs
(`plots=False` in `train_behaviour_detector.py`; `results.csv` has only aggregate mAP50 columns).

The "Safe_Walkway, Restricted_Zone, Forklift_Safe_Load never emitted" fact is real but belongs to the
**entity** detector: `Safe_Walkway`/`Restricted_Zone` have **zero training labels by design** (zones were
never hand-annotated — `kaggle-detector-retrain/README.md` precondition 1) so zero recall there is a
tautology, not a training failure; `Forklift_Safe_Load` has 170 labels and is a genuine recall failure.

**Recommendation:** A1's stated goal ("mAP50 ≥ 0.90, all 8 classes emitted", `tight`/`ctx` naming) is the
**behaviour detector's** vocabulary. Retrain that one for the paper-comparison claim (axis 2). The
existing `kaggle-detector-retrain/` package trains the **wrong dataset** for this goal — it must be
pointed at `dataset_behaviour_{tight,ctx}` (8 behaviour classes), not `annotation_guide/dataset/` (8
entity classes), or a new script must be written. Reusing `train-detector-gpu.py` as-is will produce an
entity detector, not the behaviour detector the thesis compares to the paper. If the entity detector is
*also* wanted (it feeds the P3 relation-Transformer pipeline, axis 4), that is a separate, already-scoped
task — keep the two GPU jobs and their success criteria separate.

---

## 1. Recommended recipe (behaviour detector, dataset_behaviour_tight, 1901 train / 343 val)

```python
from ultralytics import YOLO
from unsafenet import behaviour_dataset as BD

yml = BD.build(ctx=(1.0, 1.0), suffix='_tight')   # or BD.CTX_W/H for the ctx variant

model = YOLO('yolo11m.pt')          # step up from yolo11s; see ranked changes below
model.train(
    data=yml, imgsz=800, epochs=200, batch=16, device=0,
    optimizer='SGD', lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3.0, cos_lr=True, patience=30, close_mosaic=15,
    amp=True, cache=False, workers=8, seed=0, val=True, plots=True,
    # augmentation — see §1a for the fliplr/hsv_h reasoning
    hsv_h=0.0, hsv_s=0.6, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.1, copy_paste=0.0,
    project='sweep_results/r2_gpu', name='beh_tight_gpu', exist_ok=True,
)
val = YOLO('sweep_results/r2_gpu/beh_tight_gpu/weights/best.pt').val(data=yml, imgsz=800, device=0, plots=True)
for i, ap50 in zip(val.box.ap_class_index, val.box.ap50):
    print(f"{val.names[int(i)]:14s} AP50={ap50:.3f} R={val.box.class_result(int(i))[1]:.3f}")
```

No 2-stage freeze→finetune needed here (unlike the entity-detector script) — 1901 images single-stage
with `cos_lr` + `close_mosaic` converges cleanly per Ultralytics' own small-dataset guidance
([hyperparameter-tuning guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/)); 2-stage adds
a knob without a documented benefit at this scale and violates KISS.

### 1a. Ranked list of the 3 changes that matter most

1. **imgsz 416 → 800** (+ matching model capacity). Single biggest lever. 416px on a 858-instance val
   set with boxes as small as 14×50px (person slivers, panel latches) discards resolution the anchor-free
   YOLO11/26 head needs for small-object recall — this is the same reasoning already in this repo's own
   comment (`behaviour_dataset.py`: "a 14x50 px sliver at 416 px ... cannot separate these classes at
   all"). 800 roughly doubles linear resolution at ~4x compute, affordable on a T4 at batch 16.
2. **epochs 25 → ~200 with patience=30, not early-stopped at 10.** `patience=10` in the current script
   stopped training at epoch 15 while `results.csv` shows mAP50 still oscillating in the 0.75–0.81 band
   through epoch 25 — CPU wall-clock (8–10 min/epoch) is almost certainly why patience was set so low,
   not because the model had converged. On GPU (~15–30s/epoch at imgsz 800, batch 16, T4) 200 epochs is
   ~1–2h, cheap.
3. **fliplr 0.0 → 0.5.** The current script's `fliplr=0.0` is not justified by anything in this dataset's
   semantics and is inconsistent with the project's *own* entity-detector script
   (`kaggle-detector-retrain/train-detector-gpu.py` uses `fliplr=0.5`). None of the 8 behaviour classes
   are defined by left/right position (panel open/closed, walkway violation, overload carry are all
   orientation-invariant); disabling it wastes what is effectively free 2x augmentation on a
   1901-image dataset. Keep `flipud=0.0` (gravity-oriented scene, vertical flip is never valid) and
   `degrees=0.0` (same reasoning the current script already applies, correct).
   **`hsv_h=0.0` is correct, keep it** — `AuthorInt` vs `UnauthorInt` (and the entity detector's
   `Person_Authorized` vs `Person_Unauthorized`) are distinguished by **safety-vest hue** (green vs red),
   documented explicitly in this repo's own `train-detector-gpu.py` comment ("vest-colour-safe:
   Authorized=green vs Unauthorized=red is a HUE signal -> hsv_h=0"). Recolouring would flip the label.
   `hsv_s`/`hsv_v` (saturation/brightness) do not touch hue and are safe to enable.

Model size: **yolo11m over yolo11s** is the second-order lever (more capacity for the CarryOverload/
SafeCarry/AuthorInt tail classes at 170–390 labels) but costs ~2x epoch time; only take it if Gate 1 (day
2) risks slipping — yolo11s at imgsz 800/200ep is the safer first attempt given the 2-day window. See
§3 for the YOLO26 option, which is architecturally attractive (STAL = better small/rare-class recall by
name) but newest and least field-tested in this stack.

`mixup=0.1` (nonzero, unlike the current `mixup=0.0`) is a mild addition for the 1901-image regime per
Ultralytics' own small-dataset guidance; drop to 0.0 if it costs a Gate-1 day cushion — it is not one of
the ranked top-3, do not spend debugging time on it.

---

## 2. Class imbalance / dead-class diagnosis

**For the behaviour detector (this task's actual target):** no class has zero training labels (§0). If
GPU retrain still shows near-zero recall on SafeCarry/CarryOverload/AuthorInt (the 170–390-label tail),
likely causes, in order of probability:
1. Under-training at the crippled CPU budget (25 epochs, patience=10 → stopped epoch 15) — the primary
   suspect, fixed by §1's recipe.
2. Box-size / aspect outliers vs the ctx-expanded boxes (person boxes widened 4.0x/1.8x per
   `behaviour_dataset.py` `CTX_W, CTX_H` — an aspect ratio far from COCO's pretrained anchor priors on the
   anchor-free head still shifts stride assignment). Check with `results.box.mean_results()` per class
   and cross-reference box w/h histograms; anchor-free YOLO11/26 assigns objects to feature-map strides
   by size, so a class whose boxes cluster at an unusual scale can still be recall-starved even at high
   res.
3. Confidence threshold at inference — `model.val()` defaults `conf=0.001` (all boxes considered for
   mAP), but any downstream `model.predict(conf=0.25)` (Ultralytics' default predict conf,
   `engine/model.py`) will silently drop a class whose confident detections never clear 0.25. Always
   check **recall at val-time** (not just "any boxes emitted at predict time") before concluding a class
   is dead.

**Verification method (exact API):**
```python
val = model.val(data=yml, imgsz=800, device=0, plots=True)   # plots=True writes confusion_matrix.png, PR curves
for i in val.box.ap_class_index:
    p, r, ap50, ap = val.box.class_result(int(i))
    print(val.names[int(i)], 'P', p, 'R', r, 'AP50', ap50, 'AP', ap)
```
`DetMetrics.class_result(i)` returns `(P, R, AP50, AP50-95)` per class index — this is the API already
used correctly in `kaggle-detector-retrain/train-detector-gpu.py` (`val.box.ap_class_index`/`val.box.ap50`
loop). Reuse that exact pattern for the behaviour detector's val run; it is the only place in the repo
that currently checks per-class recall properly, and it must be run on the new behaviour-detector weights
too since the old ones were never checked this way.

**What Ultralytics offers for imbalance:** no first-class per-class loss-weighting argument exposed via
`train()` kwargs (only classification loss is globally scaled by `cls`, no per-class multiplier in the
CLI/Python API). Practical, KISS-compliant options actually available:
- **Oversample rare-class images** (duplicate image+label pairs for SafeCarry/CarryOverload/AuthorInt
  frames in the train split) — simplest, no code changes to the training loop, directly increases
  gradient signal for the tail classes.
- **`copy_paste`** augmentation (Ultralytics default 0.0, detection support since v8.3+) pastes instances
  from one image onto another; useful for boosting rare-class instance count per batch, but historically
  tuned for segmentation masks — test before relying on it; not one of the ranked top-3.
- **Focal loss swap**: one open Ultralytics issue reports *worse* results after swapping BCE→Focal for
  imbalance on YOLO26 ([ultralytics#24004](https://github.com/ultralytics/ultralytics/issues/24004)) —
  do not do this without controlled A/B evidence; too risky for the timebox.

---

## 3. YOLO26 status (verified against installed package + official docs)

- **Released 2026-01-14** ("Ultralytics officially launches YOLO26" — [Ultralytics blog](https://www.ultralytics.com/blog/ultralytics-yolo26-the-new-standard-for-edge-first-vision-ai)). Installed `ultralytics==8.4.27` already **ships** `ultralytics/cfg/models/26/yolo26{,-seg,-pose,-obb,-p6,-p2,-cls}.yaml` — confirmed by direct filesystem check, no upgrade required to use `YOLO('yolo26s.pt')`.
- **Architecture vs YOLO11** ([docs.ultralytics.com/models/yolo26](https://docs.ultralytics.com/models/yolo26)):
  - **NMS-free end-to-end inference** by default (dual-head: one-to-one for NMS-free ≤300 det, or classic one-to-many with NMS) — [end2end-detection guide](https://docs.ultralytics.com/guides/end2end-detection).
  - **DFL removed** from the head (lighter, hardware-friendlier box regression) — note this also means the recipe's `dfl` loss-gain kwarg is irrelevant for yolo26 (only applies to yolo11/DFL-head models).
  - **ProgLoss** (Progressive Loss Balancing) + **STAL** (Small-Target-Aware Label assignment) — STAL is directly relevant to this dataset's small person/panel-latch boxes.
  - **MuSGD** optimizer (SGD+Muon hybrid) — replaces plain SGD as the "auto" default for yolo26.
- **Documented deltas vs YOLO11** (arXiv 2509.25164, 2510.09653; Ultralytics docs): up to **+7.2 AP pose**, **+2.5 box AP / +3.7 mask AP** segmentation, **+3.4 mAP** oriented-detection on DOTA; **~43% faster CPU ONNX inference** for the nano variant. No detection-only COCO mAP delta vs YOLO11 was found stated as a single clean number in the fetched docs — treat any specific detection-mAP-delta claim as unverified until re-checked against the primary arXiv PDF tables directly (not done here — token budget).
- **Export**: TensorRT export documented and tested at **TensorRT 8.6.1 (A100)**, **10.0.0+ (consumer GPU)**, **10.0.1 (Jetson Orin NX 16GB)** ([TensorRT integration doc](https://docs.ultralytics.com/integrations/tensorrt/)). **TensorRT 11.0 drops DLA support — use 10.x for DLA**, directly relevant since the thesis's declared target is **TensorRT 10.3** — compatible. Jetson guide documents **JetPack 6.1 / 7.2** on Orin Nano Super with YOLO26 benchmarks (FP16 TensorRT ~4.57ms/img, mAP50-95 0.48 for yolo26n on COCO) — [Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/). JetPack 6 (this thesis's D2 hardware) is inside the documented compatibility window.
- **Recommended training defaults** (from [YOLO26 training-recipe doc](https://docs.ultralytics.com/guides/yolo26-training-recipe/)): small-dataset (<1000 img) guidance explicitly says reduce augmentation (`mosaic=0.5, mixup=0.0`), lower LR, `epochs=50, patience=20`, consider `freeze=10`. At 1901 images (a "medium" bucket the doc doesn't name directly) the mid-size defaults (`mosaic~0.99, mixup~0.05-0.4, copy_paste~0.3-0.4, fliplr~0.3, close_mosaic=10`) are closer but were tuned on COCO-scale data — do not blindly copy; the §1 recipe's augmentation choices (grounded in this dataset's own semantics) should override the generic defaults.
- **Verdict for this task**: yolo26s is a legitimate, low-risk swap-in (`YOLO('yolo26s.pt')` in place of `yolo11m.pt` in §1's recipe, drop the `dfl` consideration, keep `optimizer='auto'` to get MuSGD) — worth a *second* run if time permits after the yolo11m baseline lands, not a replacement for it given YOLO26 is 8 months old at most and this specific 1901-image/8-class regime is untested anywhere in the literature. Sequence: yolo11m first (known quantity, matches what `t1-objectives-audit.md` already validated at small scale), yolo26s as a stretch/comparison run if Gate 1 is met early.

---

## 4. Kaggle/Colab logistics

- **Kaggle GPU quota: ~30h/week**, **session cap 9h** (not the 12h assumed in `kaggle-detector-retrain/README.md` — that number should be corrected) — [Kaggle GPU quota discussion](https://www.kaggle.com/general/108481), [efficient-GPU-usage docs](https://www.kaggle.com/docs/efficient-gpu-usage). T4x2 = 2×16GB.
- **Wall-clock estimate for §1's recipe** (yolo11m, imgsz 800, batch 16, 200 epochs, 1901 train images, single T4): the existing `kaggle-detector-retrain/README.md` reports **~3–5h for 60 epochs (20+40) at imgsz 960, batch 16** on this same repo's entity dataset (2262 img, similar scale) → roughly 3–5 min/epoch at imgsz 960. At imgsz 800 (smaller, faster) and yolo11m (bigger, slower) these roughly offset; **expect ~2.5–4h for 200 epochs** — fits one Kaggle session (9h cap) with large margin, well inside the 30h/week quota even with 2-3 retries.
- **Keep-alive / resume**: Kaggle notebooks time out on idle browser tabs but the kernel itself runs headless once "Save & Run All (Commit)" is used (as `colab/kaggle_yolo26_onecell.py` already does) — this is the correct pattern, avoids babysitting. For resume: Ultralytics supports `model.train(resume=True)` from `last.pt` if a run is interrupted; checkpoint every epoch is automatic (`last.pt`/`best.pt` written each epoch by default).
- **Getting weights out**: `/kaggle/working/` persists to the notebook's Output tab after a committed run — download `best.pt` (~19-40MB for s/m) directly, no extra step. This is already the pattern in `retrain-detector.ipynb` cell 5.
- **Dataset upload constraints**: Kaggle Datasets have no hard per-file size limit publicized for typical ML use but private datasets are commonly capped around 20-100GB depending on tier; this project's payload is **~1.2GB** (`prepare-upload.sh` comment) — no issue. Zip upload auto-extracts.
- **Colab** (`colab/kaggle_yolo26_onecell.py` naming is misleading — it is actually written for Kaggle, not Colab, per its own header "Paste into a SINGLE Kaggle cell"): Colab free tier gives a single T4, ~12h session cap but no clean published weekly quota (usage-based throttling, less predictable than Kaggle's stated 30h). **Recommend Kaggle as primary** — quota is explicit and this project's tooling already targets it.

---

## 5. Is mAP50 0.95 (or "95.7%") attainable / comparable? — read directly from `paper/paper.txt`

**The paper never states a numeric "95.7% mAP" for YOLOv4, anywhere in `paper.txt`.** Verified by
exhaustive grep — the string "95.7" does not occur in the paper text. The **only** number this repo's own
docs attribute as "95.7% mAP" (`PROGRESS.md` L64, propagated into `brief.md` L35 citing "Table 3") is
**not supported by the cited table**: Table 3 (paper.txt L206-267) is exclusively an **image-count table**
(2262 train / 329 test images labelled, broken down by 8 classes) — it contains **zero mAP values**.

What the paper **actually** says about YOLOv4 detection accuracy (paper.txt L323, the only sentence in the
whole document that quantifies YOLOv4's own metric):
> "the average mAP value during the iteration period is around 97%"

This is:
- **A qualitative read off a training-curve figure (Fig. 8)**, not a table, not a held-out test-set
  number, no CI, no IoU threshold stated beyond the implicit YOLOv4-era mAP@0.5 convention.
- **Computed on nc=4, not nc=8** — same paragraph, same sentence family: "the number of classes was 4"
  (paper.txt L323). The paper's YOLOv4 was trained to detect only the **4 unsafe** behaviour classes
  (Safe Walkway Violation, Unauthorized Intervention, Opened Panel Cover, Carrying Overload with
  Forklift) — its stated purpose is solely to **select which frames to keep** before ConvLSTM
  classification (paper.txt L323, L325: "by selecting the frames that belong to the unsafe behaviour
  classes ... other parts ... are removed"). The 4 *safe* classes in Table 3 are **labelling-effort
  bookkeeping** (they exist in the ground-truth images) but were **not** part of the YOLOv4 4-class
  training target per the explicit statement.
- **No Table 4 with mAP numbers exists.** Table numbering in the paper: T1=class taxonomy, T2=video
  counts, T3=image-label counts (no mAP), T5=ConvLSTM-only per-class Acc, T6=Unsafe-Net per-class Acc,
  T7=Acc summary (95.81% macro — this is the **ConvLSTM classification** accuracy, not YOLOv4 detection
  mAP; already correctly used elsewhere in this project as the 95.81% clip-classification anchor).

**Conclusion: the "95.7%" figure the project has been comparing against does not exist in the source
paper and should be retracted from PROGRESS.md/brief.md as a detector benchmark.** The only real anchor is
the qualitative "~97% mAP, read off a curve, nc=4, no test-set protocol stated." Our nc=8, held-out-val-set
mAP50 is **not comparable** to that figure on three independent axes: (a) different class count (8 vs 4),
(b) different measurement (proper held-out val() vs eyeballed training curve), (c) different, unstated IoU/
metric protocol. **Recommend the thesis state plainly: no comparable published YOLOv4 detection-mAP number
exists; report our nc=8 mAP50/mAP50-95 as a new, standalone measurement, and drop any "vs 95.7%" framing.**
This does not weaken the thesis — axis 2 ("faithful reproduction... where it holds and where it does not")
already anticipated exactly this kind of protocol mismatch; this is one more, and arguably the cleanest,
instance of it.

**Given that reframing, is mAP50 ≥ 0.90 attainable at nc=8, 1901 train images?** Yes, plausible — reasoning:
YOLO11/26-s/m routinely reach 0.85-0.95 mAP50 on custom 8-class industrial datasets of 1-3k images at
imgsz 640-800 with proper training length (100-300 epochs) when class boxes are visually distinctive
(panels, forklifts, safety vests are high-contrast, low-intra-class-variance objects — easier than COCO's
80 diverse classes). The paper's own qualitative "~97%" (nc=4, presumably easier since fewer, more
separable classes) is broadly consistent with this range being reachable, just not identical or safely
citable as a target number.

---

## Decision-ready summary

**Recommended recipe:** §1, `yolo11m.pt`, imgsz 800, epochs 200, batch 16, SGD, cos_lr, patience 30,
close_mosaic 15, `hsv_h=0` (keep — vest-colour semantics), `fliplr=0.5` (flip from current 0.0 — no
semantic justification for disabling it, inconsistent with this repo's own entity-detector script).

**Ranked top-3 changes:** (1) imgsz 416→800, (2) epochs 25→200 with patience 30 not 10 (CPU-forced
under-training, not convergence), (3) fliplr 0.0→0.5 (unjustified augmentation loss on a small dataset).

**Risk table**

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Wrong dataset retrained (entity vs behaviour, §0) | High if not flagged | Wastes the 2-day gate window | Point training explicitly at `dataset_behaviour_{tight,ctx}`; verify class names in `data.yaml` before launching |
| mAP50 ≥0.90 not reached in 2 days | Medium | Gate 1 fails, brief's own fallback triggers (keep P3 k-fold/LCO as headline) | Brief already has this fallback (§Go/no-go gate 1) — acceptable, not a plan-breaker |
| YOLO26 regression/instability (new model family, thin field track record) | Medium | Wasted GPU-hours if chosen as primary | Run yolo11m first (proven), yolo26s only as a stretch/second run |
| Kaggle session timeout losing progress | Low | Re-run needed | Use "Save & Run All (Commit)" pattern (already in `retrain-detector.ipynb`); `resume=True` from `last.pt` |
| Focal-loss / per-class-weight hack introduces regression | Medium (documented in ultralytics#24004) | Wrong conclusion about which classes are fixed | Do not touch loss weighting; use oversampling + recipe fixes only, verify via `val.box.class_result()` |
| "95.7%" comparison persists uncorrected into the thesis | High if not caught here | Reviewer catches an unverifiable claim, credibility hit | Retract per §5; state paper's actual claim (~97%, nc=4, curve-read) explicitly |

**Expected-mAP estimate:** 0.85–0.93 mAP50 (yolo11m, imgsz 800, 200ep) is a defensible target range given
(a) the CPU run already reached 0.79-0.81 with the crippled 416px/25ep/patience-10 budget while still
rising at the point it early-stopped, (b) imgsz 800 alone typically adds several points on datasets with
small-object content per the resolution/small-box reasoning in §1a, (c) YOLO11/26 on visually-distinct
industrial object classes (panels, forklifts, vests) with 1901 images is inside the range where 0.85-0.95
is routine in comparable custom-dataset reports. 0.90 is a reasonable but not certain gate — treat 0.85 as
the fallback-acceptable floor per the brief's own Gate 1 language ("<0.85 → stop, keep P3 as headline").

---

## Unresolved questions
1. Per-class recall of the *existing* beh_tight/beh_ctx CPU-trained weights was never captured (`plots=False`, no per-class printout saved) — cannot confirm or deny dead classes on the behaviour detector pre-GPU-retrain; only entity detector's dead classes are evidenced. Recommend running `val.box.class_result()` on the existing `runs/detect/sweep_results/r2/beh_tight/weights/best.pt` as a 2-minute sanity check before the GPU retrain, to know the true starting point.
2. Whether A1's "8/8 classes emitted" success criterion (brief.md schedule, Day 2) was meant for the behaviour detector or the entity detector is not stated anywhere explicitly — §0's reading (behaviour detector, since `tight`/`ctx` language matches) is the most consistent interpretation but should be confirmed with the plan owner before the GPU job is launched, since the two require different scripts/datasets.
3. No detection-only (as opposed to seg/pose/OBB) COCO mAP delta table for YOLO26 vs YOLO11 was directly verified against the primary arXiv PDF (2509.25164 / 2510.09653) in this pass — the docs.ultralytics.com summary did not surface one cleanly. If YOLO26 becomes the primary model choice, pull that table directly before citing a specific number in the thesis.
4. `kaggle-detector-retrain/README.md`'s "12h Kaggle GPU session" claim is contradicted by the current documented 9h session cap — the README should be corrected, though this does not change feasibility (recipe fits well inside 9h).
