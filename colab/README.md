# Colab GPU Training — Multi-Entity YOLOv11s Detector

Train Stage A (entity detector) of the paper-faithful pipeline on a **free Colab GPU**.
Local MPS training is broken on macOS 14 (verified); CUDA on Colab avoids it entirely and is far faster.

## What's in this package
| File | Purpose |
|---|---|
| `../unsafe-net-detector-dataset.zip` | The dataset to upload (1.2 GB: images + labels, 2262 frames) |
| `train_detector_colab.ipynb` | Ready-to-run Colab notebook (8 steps) |
| this README | instructions |

## Steps (≈ 1 hour total, ~30–60 min of it is GPU training)

1. **Upload the dataset to Google Drive.**
   Put `unsafe-net-detector-dataset.zip` in the **root of My Drive** (drag-drop at drive.google.com). One-time.

2. **Open the notebook in Colab.**
   Go to https://colab.research.google.com → *File → Upload notebook* → pick `train_detector_colab.ipynb`.
   (Or drop the .ipynb in Drive and open with Colab.)

3. **Enable the GPU.**
   *Runtime → Change runtime type → Hardware accelerator = GPU (T4 is free)* → Save.

4. **Run cells top to bottom** (▶ each, or *Runtime → Run all*):
   - Step 1 `nvidia-smi` — confirms a GPU is attached.
   - Step 2 installs Ultralytics.
   - Step 3 mounts Drive, copies + unzips the dataset to `/content/ds` (prints frame counts — should be train 1919 / val 343).
   - Step 4 writes `data.yaml` (nc=8).
   - Step 5 **Stage 1** — freeze backbone, 30 epochs.
   - Step 6 **Stage 2** — unfreeze, fine-tune, 100 epochs (early-stops via patience=30).
   - Step 7 prints **per-class mAP** (watch Person_Authorized vs Unauthorized).
   - Step 8 saves `best.pt` + `runs.zip` (curves, confusion matrix) to `My Drive/unsafe-net-detector-out/`.

5. **Download `best.pt`** from `My Drive/unsafe-net-detector-out/` → drop it into the repo
   (e.g. `sweep_results/entity_detector/best.pt`) for the next step.

## Training recipe (already set in the notebook)
- **Model:** yolo11s.pt (COCO-pretrained, auto-downloads)
- **Optimizer:** SGD lr0=0.01 (stage 1) → 0.001 (stage 2), momentum 0.937 — the stable YOLO default
- **imgsz 800** — needed for small/distant people (5% of boxes are tiny)
- **hsv_h = 0** — the key thesis strategy: never recolour vests (green=Authorized vs red-black=Unauthorized)
- **2-stage freeze→fine-tune**, mosaic 0.5, patience 30, seed 0 (reproducible)
- Zones (classes 2,3) have 0 detection instances by design → handled as per-view constants from `annotation_guide/zones/`

## What "good" looks like
box_loss and cls_loss **decreasing** each epoch; mAP50 **climbing**. On the local CPU sanity run this config hit
mAP50 = 0.46 in just 3 epochs on 10% data, so expect strong numbers on the full run. Especially check
**Person_Authorized recall** — it has only 365 samples; if it lags, we'll add samples or switch to a single
`Person` class + vest-colour feature (see the Phase-02 review).

## Troubleshooting
- **CUDA out of memory** (small T4): set `batch=8` in Step 5's `COMMON`.
- **"Upload ... zip first" assertion**: the zip isn't in My Drive root, or Drive didn't finish syncing.
- **Runtime disconnects**: free Colab has idle limits; keep the tab active. Results are saved to Drive at the end,
  and each stage saves checkpoints under `/content/runs/*/weights/` you can resume from.
- **No GPU / very slow**: you're on CPU runtime — redo Step 3 (Change runtime type → GPU).

## After training
Download `best.pt` → next we run `build_relation_tokens` **locally** (inference on MPS is fine — only *training*
is affected) to combine YOLO detections + zone geometry (foot-in-polygon, IoU) into tokens for the Transformer (Stage B).
