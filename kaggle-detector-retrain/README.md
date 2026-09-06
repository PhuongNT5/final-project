# Kaggle detector retrain

> **This package trains TWO different detectors. Pick the right one.**
>
> | | **Behaviour** detector | **Entity** detector |
> |---|---|---|
> | Scripts | `build-behaviour-labels.py` + `train-behaviour-detector-gpu.py` | `build-zone-labels.py` + `train-detector-gpu.py` |
> | Classes | WalkwayViol, UnauthorInt, OpenedPanel, CarryOverload, SafeWalkway, AuthorInt, ClosedPanel, SafeCarry (paper Table 3) | Person_Authorized, Person_Unauthorized, Safe_Walkway, Restricted_Zone, Panel_Closed, Panel_Opened, Forklift_Safe_Load, Forklift_Overload |
> | Purpose | **Gate 1 / task A1** — the paper-comparison claim | feeds the P3 relation-Transformer |
> | CPU baseline | mAP50 0.791 (tight) / 0.806 (ctx) | mAP50 0.969, but 3 classes never emitted |
>
> Gate 1's "mAP50 ≥ 0.90, all 8 classes emitted" refers to the **behaviour** detector. Everything below
> the "Two arms" heading describes the **entity** detector and is a separate job — do not conflate them.
> See `plans/2026-09-06-thesis-2week-final/research/r1-detector-gpu-recipe.md` §0.

> **Data-governance note.** The payload contains 2,262 factory frames, which
> `annotation_guide/annotation-sop.md` places under consent forms. **Clearance asserted by the thesis
> author, 2026-09-06** — they hold the rights to this footage and authorised the Kaggle upload. Recorded
> here so the SOP and the decision that overrode it are both on the record.
>
> One thing still worth doing: a Kaggle Dataset is **public by default**. Create it as **Private** unless
> you actively want the footage indexed — public datasets get crawled and mirrored, and un-publishing
> later does not un-mirror them.

## Behaviour detector (Gate 1 / task A1)

```bash
python build-behaviour-labels.py --variant tight     # or --variant ctx
python train-behaviour-detector-gpu.py \
    --data /kaggle/working/annotation_guide/dataset_behaviour_tight/data.yaml
```

Recipe and reasoning: `r1-detector-gpu-recipe.md` §1. Top-3 changes vs the CPU baseline:
imgsz 416→800, epochs 25→200 with patience 30 (the CPU run early-stopped at epoch 15 while still
rising), fliplr 0.0→0.5. `hsv_h` stays **0.0** — AuthorInt vs UnauthorInt is a vest-hue signal.

Labels are built by the repo's own `unsafenet/behaviour_dataset.py` (shipped inside the payload), so the
Kaggle-side labels are bit-identical to the local ones — verified by diff. Dataset: 1901 train / 343 val
frames, **6122 boxes**, no class with zero labels; worst imbalance 9.6:1 (WalkwayViol 1634 : SafeCarry 170).

> **What this mAP does and does not measure.** The train/val split is clip-disjoint but **all 8 camera
> views appear on both sides**, so val mAP measures *in-domain* detection quality. It is not evidence of
> cross-camera generalisation, and it must not be reported as such — that is the same scene confound the
> thesis documents for the classifier. Note also that classes 0/1 (WalkwayViol/UnauthorInt) are person
> boxes labelled by *clip*, so at `--variant tight` they are visually identical and separable only by
> surrounding context; `ctx` widens the box to include that context by construction.

Motivated by the Phase-1
finding: the current CPU-trained detector emits boxes for only **5 of 8** ontology classes — **Safe_Walkway,
Restricted_Zone, Forklift_Safe_Load are never emitted** — which bounds the relation head's ability to
separate WalkwayViol↔UnauthorInt.

## Two arms

| arm | flag | goal | risk |
|---|---|---|---|
| **baseline** | (default) | recover **Forklift_Safe_Load** (170 labels, 0 detected) + higher-res mAP on the 6 movable entities | none — pure detector-quality fix |
| **zones** | `--add_zones` | make the detector **emit zone boxes** (classes 2,3) so person↔zone relations become real, camera-invariant features | experimental — see leak caveat |

## Preconditions found (why this is non-trivial)

1. **Zones were never labeled** — classes 2/3 have 0 annotations by design; polygons live in `zones/view_*.json`
   and were injected at inference. `build-zone-labels.py` converts polygons → axis-aligned YOLO boxes.
2. **Image→view mapping is partial** — only 1550/2262 detector images map to a camera view via
   `clip_view_map.json`; views 2 & 7 have no zone file. So the zone arm trains on **~1350 images** (views
   0,1,3,4,5,6). Unmappable / no-zone-file images are EXCLUDED (never false-negatives for zones).
3. **Leak risk (zone arm).** A zone is a static per-camera region. If the detector memorizes "walkway is at
   pixel-region X for camera K" it will NOT transfer to a held-out camera → the same leave-camera-out (LCO)
   collapse lever L2 showed. It only helps if walkways share a *visual* cue (paint/markings) the model
   generalizes. **The downstream LCO eval decides** — do not trust the in-distribution val mAP alone.
4. **Loose boxes.** AABB of a diagonal walkway polygon can span most of the frame (e.g. w=0.28, h=1.0),
   making "foot inside walkway box" weakly discriminative. Inherent to axis-aligned boxes.

## Files
- `build-behaviour-labels.py` — builds the 8-class **behaviour** dataset on Kaggle by calling the repo's own
  `unsafenet/behaviour_dataset.py` (shipped under `scripts/unsafenet/`), so there is no second copy of the
  conversion logic to drift.
- `train-behaviour-detector-gpu.py` — the Gate-1 behaviour detector; prints per-class P/R/AP50, checks the
  pre-registered gate, and writes `<name>_result.json`.
- `build-zone-labels.py` — polygon → zone-box labels; builds a writable zone-augmented dataset (non-destructive).
- `train-detector-gpu.py` — 2-stage COCO transfer (freeze→finetune), GPU-tuned (AMP, imgsz 960), vest-colour-safe aug (`hsv_h=0`); `--add_zones` selects the zone arm; prints per-class AP50 at the end.
- `retrain-detector.ipynb` — the Kaggle notebook (both arms + smoke test + weight collection).
- `prepare-upload.sh` — assembles the Kaggle-Dataset payload locally.

## Run it

### 1. Local — assemble the upload payload (once, ~1.2 GB)
```bash
bash kaggle-detector-retrain/prepare-upload.sh
# -> kaggle-detector-retrain/upload/unsafenet-detector/  (dataset/ zones/ clip_view_map.json yolo11s.pt scripts/)
```

### 2. Kaggle — create the dataset + notebook
1. kaggle.com → **Datasets → New Dataset** → upload `upload/unsafenet-detector.zip` (Kaggle auto-extracts). Title `unsafenet-detector` is nice-to-have but **not required** — cell 0 auto-detects the input by content.
2. **Notebooks → New Notebook** → upload `retrain-detector.ipynb` (or paste cells).
3. **Add Input** → your dataset. It mounts under `/kaggle/input/<slug>`; cell 0 finds it by locating the dir that contains `dataset/` + `scripts/`, so the exact slug does not matter.
4. **Settings → Accelerator → GPU T4**. Internet can stay **off** (weights are bundled).
5. Run cells top-to-bottom. The smoke cell is a ~2-min check; run the full arms after it passes.

> **If cell 0 asserts "Dataset not found":** the input was not attached. Click **Add Input** (right panel) and pick your dataset. `inputs mounted:` in the cell output lists what Kaggle actually sees — `NONE` means nothing is attached yet.

Timing (T4, imgsz 960, batch 16): ~3–5 h per arm for 20+40 epochs. Kaggle's session cap is **9 h** and the
GPU quota **~30 h/week** ([efficient-GPU-usage docs](https://www.kaggle.com/docs/efficient-gpu-usage)) — an
earlier version of this README said 12 h, which was wrong. To stay inside it, lower `--epochs2`, drop
`--imgsz` to 800, or run one arm per session. On CUDA OOM, `--batch 8`. The behaviour-detector job
(yolo11s, imgsz 800, 200 epochs) is ~2.5–4 h and fits one session with margin.

### 3. Handoff back to the repo (judge the new detector)
Download `/kaggle/working/weights/detector_{baseline,zones}_best.pt`, then in the repo:
1. Point the token extractor at the new weights and re-extract relation tokens on the 516 behaviour clips.
2. Rebuild the L1 token set (`tools/build_l1_tokens.py`) from the new detections.
3. Re-run `run_lco_eval.py --scene_map …` (primary) + `run_kfold_relation.py` and compare vs the current
   L1 base. The canonical LCO baseline is **77.7% top-1 / 81.2% macro** (5-seed,
   `sweep_results/relation_tokens/canonical/lco_seeds5_yolo26L1_CANONICAL.log`). An earlier version of this
   README used **82.0%**, which was a superseded 3-seed figure — do not use it as the acceptance threshold.
   Keep the new detector **iff LCO macro rises with no leak** (k-fold↑/LCO↓ = reject, per the campaign rule).
4. Pass `--dedupe drop_all` to both evals. Defaults are `none`, which leaves the 32 byte-identical
   label-contradictory clip groups in the pool (see the data-integrity note in `dedupe-procedure.md`).

## Unresolved questions
1. Zone arm: does detected-zone geometry beat L1 on **LCO** (not just k-fold)? Unknown until step 3 — this is
   the whole hypothesis test.
2. Baseline arm changes the detector for ALL classes; re-extracting tokens shifts every relation-head number
   → the L1 82.0% baseline must be recomputed on the new detections before any lever comparison is fair.
3. Forklift_Safe recovery: does it improve CarryOverload (already 100% LCO recall) or only precision? Likely
   marginal for the headline; still worth fixing for detector completeness.
