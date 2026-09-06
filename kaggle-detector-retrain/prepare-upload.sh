#!/usr/bin/env bash
# Build the Kaggle-Dataset payload as a single .zip, read directly from the originals (NO 1.2GB staging copy).
# Kaggle auto-extracts an uploaded .zip, so its contents land directly under /kaggle/input/<dataset-slug>/.
# Archive layout (matches what the notebook/scripts expect at /kaggle/input/unsafenet-detector/):
#   dataset/{data.yaml,images/{train,val},labels/{train,val}}
#   zones/view_*.json
#   clip_view_map.json
#   frame_manifest.csv                      # behaviour labels are derived from this
#   yolo11s.pt
#   scripts/{build-zone-labels.py,train-detector-gpu.py,
#            build-behaviour-labels.py,train-behaviour-detector-gpu.py}
#   scripts/unsafenet/{__init__.py,config.py,behaviour_dataset.py}   # single source of truth for
#                                                                   # the entity->behaviour conversion
#
# Run once from the repo root:  bash kaggle-detector-retrain/prepare-upload.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTDIR="$ROOT/kaggle-detector-retrain/upload"
ZIP="$OUTDIR/unsafenet-detector.zip"
TMP="$OUTDIR/.stage"          # holds only the small/modified files (few MB), not the images
mkdir -p "$OUTDIR"; rm -f "$ZIP"; rm -rf "$TMP"

echo "[1/4] big dirs (images+labels under dataset/, zones/) — streamed straight into the zip ..."
# -r recurse, -q quiet, -x exclude ultralytics label caches. cd so archive paths start at dataset/ , zones/
( cd "$ROOT/annotation_guide" && zip -r -q "$ZIP" dataset/images dataset/labels zones -x '*.cache' )

echo "[2/4] staging small + modified files ..."
mkdir -p "$TMP/dataset" "$TMP/scripts"
# data.yaml with the Kaggle-resolved path (input is read-only at /kaggle/input/unsafenet-detector)
sed 's|^path:.*|path: /kaggle/input/unsafenet-detector/dataset|' \
    "$ROOT/annotation_guide/dataset/data.yaml" > "$TMP/dataset/data.yaml"
cp "$ROOT/sweep_results/zone_prior/clip_view_map.json" "$TMP/clip_view_map.json"
cp -L "$ROOT/yolo11s.pt" "$TMP/yolo11s.pt" 2>/dev/null \
    || echo "  (yolo11s.pt missing — turn Kaggle Internet ON so ultralytics can auto-download it)"
cp "$ROOT/annotation_guide/frame_manifest.csv" "$TMP/frame_manifest.csv"
cp "$ROOT/kaggle-detector-retrain/build-zone-labels.py" "$TMP/scripts/"
cp "$ROOT/kaggle-detector-retrain/train-detector-gpu.py" "$TMP/scripts/"
cp "$ROOT/kaggle-detector-retrain/build-behaviour-labels.py" "$TMP/scripts/"
cp "$ROOT/kaggle-detector-retrain/train-behaviour-detector-gpu.py" "$TMP/scripts/"
# ship the package itself so the Kaggle-side behaviour labels come from the SAME code as local
mkdir -p "$TMP/scripts/unsafenet"
for f in __init__.py config.py behaviour_dataset.py; do
    cp "$ROOT/unsafenet/$f" "$TMP/scripts/unsafenet/$f"
done

echo "[3/4] adding small files to the zip ..."
( cd "$TMP" && zip -r -q "$ZIP" dataset/data.yaml clip_view_map.json frame_manifest.csv scripts \
    $( [ -f yolo11s.pt ] && echo yolo11s.pt ) )
rm -rf "$TMP"

echo "[4/4] done."
du -h "$ZIP" 2>/dev/null || true
echo "[archive] $ZIP"
echo ""
echo "Upload it as a Kaggle Dataset (name/slug EXACTLY 'unsafenet-detector'):"
echo "  UI : Datasets -> New Dataset -> upload $ZIP  (Kaggle auto-extracts)"
echo "  CLI: kaggle datasets create -p \"$OUTDIR\"   # after editing dataset-metadata.json"
echo "Then open retrain-detector.ipynb, add the dataset as input, enable GPU T4, run."
