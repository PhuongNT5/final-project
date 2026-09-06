#!/usr/bin/env bash
# Turnkey detector-arm evaluation: extract relation tokens from a detector's weights, strip abs-coords (L1),
# then judge under LCO (headline) + k-fold (secondary) against the champion. Used to de-confound the
# zones-detector rejection: run this on the YOLO11s BASELINE (no zones) weights and compare its LCO macro to
# the zones arm (79.7) and the YOLO26 champion (82.0) — separating the architecture effect from the zone-token
# effect. Also re-runnable on the zones weights to regenerate that arm.
#
# Usage: tools/eval-detector-arm.sh <detector_best.pt> <out_tag> [device]
#   tools/eval-detector-arm.sh sweep_results/retrain-detector/detector_baseline_best.pt baseline_ cpu
# Produces: tokens_<tag>app_{train,test}.pt, tokens_<tag>L1_{train,test}.pt, and prints LCO + k-fold.
set -euo pipefail
cd "$(dirname "$0")/.."

DET="${1:?detector weights path required}"
TAG="${2:?out_tag required (e.g. baseline_)}"
DEV="${3:-cpu}"
PY=.venv/bin/python
SCENE=sweep_results/zone_prior/clip_scene_map.json
[ -f "$DET" ] || { echo "ERROR: detector weights not found: $DET" >&2; exit 1; }

echo "== [1/4] extract appearance tokens from $DET (tag ${TAG}app_) =="
$PY build_relation_tokens.py --detector "$DET" --appearance --appearance_backbone yolo \
    --out_tag "${TAG}app_" --device "$DEV"

echo "== [2/4] strip abs-coords -> ${TAG}L1_ =="
$PY tools/build_l1_tokens.py --src "${TAG}app_" --dst "${TAG}L1_"

echo "== [3/4] LCO (headline) — compare to champion 82.0 / zones 79.7 =="
$PY run_lco_eval.py --tag "${TAG}L1_" --scene_map "$SCENE" --seeds 3 --epochs 60

echo "== [4/4] k-fold (secondary) — compare to champion 84.7 / zones 83.6 =="
$PY run_kfold_relation.py --tag "${TAG}L1_" --k 5 --seeds 3 --epochs 60

echo "== DONE. De-confound rule: baseline LCO ~= zones (79.7) => the -2.3 is ARCHITECTURE (YOLO11s<YOLO26),"
echo "   zone tokens neutral. baseline LCO ~= 82 => zone tokens HURT. In between => mixed. =="
