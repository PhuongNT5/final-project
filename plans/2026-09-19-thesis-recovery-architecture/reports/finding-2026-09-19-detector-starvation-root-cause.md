# Finding: detector-starvation hypothesis — RAISED AND FALSIFIED (same session)

Date: 2026-09-19. **Status: FALSIFIED by direct measurement.** Kept as a record because the
measurement it forced (§3) is a thesis deliverable that did not previously exist.

## 1. The hypothesis (now dead)

Initial read of `sweep_results/entity_detector/stage1_frozen/` (`args.yaml`: `epochs: 3, batch: 4,
device: cpu`; `results.csv`: recall 0.435, mAP50 0.462, still climbing) suggested the entity detector
had never been trained to convergence, starving every downstream relation token.

## 2. Why it is wrong

- `stage1_frozen/` also carries `fraction: 0.1` — it is a **3-epoch, 10%-of-data CPU smoke test**, not
  the production detector. Its weights appear nowhere downstream.
- Provenance settled by md5 + embedded `train_args`: repo-root `best.pt` == `entity_detector/best.pt`
  == `detector_best.pt` (one file, md5 `d8a3cbbdd75f6ee0478745ada56003b7`), trained on **Colab GPU,
  100 epochs, imgsz 800, full data**. `yolo26_best.pt` trained on **Kaggle GPU, 100 epochs, imgsz 800**.
- The headline binary tokens were built with the Kaggle YOLO26 checkpoint:
  `sweep_results/relation_tokens/build_bin8_2026-09-16.log` line 1 →
  `detector=sweep_results/entity_detector/yolo26_best.pt device=mps appearance=True backbone=yolo`.
- The MPS-non-convergence failure is real and IS in the repo (`train_run.log`: local 2-stage MPS run,
  val rows 0.000483/0.00421/0.000248 end of stage 1, then exactly `0 0 0 0` every epoch of stage 2),
  but it is an **abandoned run** superseded the next morning by the Colab checkpoint. It never
  contaminated the tokens. Keep the "do not train detection on MPS" rule — now evidenced.

## 3. The measurement that closes the real gap

The genuine defect was never "an undertrained detector" — it was that **the production detector's mAP
had never been measured or recorded anywhere in this repo** (both 100-epoch runs lived on
Colab/Kaggle filesystems; no `results.csv` survived). "What is your detector's mAP?" was unanswerable
at defence. Now measured — `yolo val`, `annotation_guide/dataset` val split, 343 images / 988
instances, imgsz 640, MPS inference:

| checkpoint | provenance | mAP50 | mAP50-95 | P | R |
|---|---|---:|---:|---:|---:|
| **`yolo26_best.pt`** (built headline tokens) | Kaggle GPU, 100 ep | **0.9658** | **0.8720** | 0.9196 | **0.9448** |
| `best.pt` = `detector_best.pt` (YOLO11s) | Colab GPU, 100 ep | **0.9752** | **0.8883** | 0.9418 | **0.9560** |
| `stage1_frozen/weights/best.pt` | CPU, 3 ep, 10% data | 0.4620 | 0.3536 | 0.9377 | 0.4356 |

Per-class, `yolo26_best.pt`: Person_Authorized P=1.000 R=0.899 mAP50=0.936 · Person_Unauthorized
P=0.860 R=0.904 mAP50=0.932 · Panel_Closed P=0.943 R=0.866 mAP50=0.969 · Panel_Opened P=0.855 R=1.000
mAP50=0.970 · Forklift_Safe_Load P=0.877 R=1.000 mAP50=0.992 · Forklift_Overload P=0.983 R=1.000
mAP50=0.995.

Caveat to state in the thesis: this val split is frame-level out of the **train partition**, so it is
an **in-domain** detector score. It does not establish cross-camera detector quality, and it is not
directly comparable to a clip-level held-out number.

## 4. Consequences

- **The detector lever is dead.** Do not spend Kaggle/Colab hours retraining it. mAP50 0.966 with
  recall 0.945 is not the bottleneck.
- The 11.8:1 Person_Unauthorized:Person_Authorized imbalance did **not** destroy minority-class
  detection (P_Auth recall 0.899). The imbalance hypothesis for the downstream safe-recall collapse is
  **weakened**, though not excluded cross-camera.
- Positive side-effect: the table above is a **new thesis deliverable**. It also supports a fair
  comparison to the baseline paper, which reports only "around 97 %" read off a training curve for a
  nc=4 gate at 416×416 (per `paper/paper.txt`; no test-set mAP published).
- The bottleneck is therefore **downstream of detection** — token representation, the head, or the
  data itself.

## 5. Dataset-side findings that SURVIVE (independent of the falsification)

- `annotation_guide/dataset` declares `nc: 8` but classes **2 (Safe_Walkway) and 3 (Restricted_Zone)
  have ZERO instances** in train and val. Train counts: cls1 3863, cls4 995, cls5 446, cls0 327,
  cls7 167, cls6 146. Val: 634/186/79/38/28/24. 1919 train images / 1902 label files (17 unlabelled).
  Two dead classes waste capacity; `nc: 6` is the correct configuration.
- Person_Unauthorized : Person_Authorized = **11.8:1** train, 16.7:1 val.
- **Zones ARE already wired into token features** — `build_relation_tokens.py:geometry()` consumes
  `annotation_guide/zones/view_*.json` to produce token dims 6-9 (`iou_walkway`, `iou_restricted`,
  `foot_in_walkway`, `norm_dist`). The accurate gap statement is narrower: zones are (a) absent from
  the **detector label space**, (b) computed for person-classes only, and (c) **missing for view_2 and
  view_7, which zeroes the geometry channel on 86 clips**. (c) is a concrete, cheap, unexploited fix.
  Do NOT let the thesis say "zones are unused" — that would be false.
- Appearance: `yolo` backbone → 512-d GAP → **fixed untrained Gaussian projection 512→64**, seed 0;
  crops are box×1.3 (30% context padding — a concrete scene-leak channel). It is a
  Johnson–Lindenstrauss compression of real pretrained features, **not noise** — do not overstate it.
  Untried: learned projection, larger APP_DIM, CLIP/SigLIP on entity crops, zero context padding.

## 6. Method note

This hypothesis cost ~20 minutes to raise and ~15 minutes to kill, before any GPU hour was spent on
it. The cheap-gate-first ordering is what made that possible and should be kept for every remaining
lever.

## Unresolved questions

1. Cross-camera detector quality is unmeasured — the val split is in-domain. Worth an LCO-style
   detector evaluation if any accuracy claim leans on detection quality.
2. Do the 86 zero-geometry clips (view_2/view_7) concentrate in particular LCO clusters or in the
   safe classes? If so, fixing zones could directly lift safe recall.
3. `yolo26_best.pt` (0.9658) vs `best.pt` (0.9752): the YOLO11s checkpoint scores *higher* in-domain,
   yet the YOLO26 one built the headline tokens. Was that choice ever justified by a measurement?
