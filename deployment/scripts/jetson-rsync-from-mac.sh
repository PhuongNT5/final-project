#!/usr/bin/env bash
# Run ON the MacBook (M3) to push code, ONNX models, and test features to the Jetson.
# Excludes large caches that aren't needed on the edge.
#
# Usage (on M3):
#   JETSON_HOST=phuong@192.168.1.42 bash deployment/scripts/jetson-rsync-from-mac.sh
#   LANE=r48 JETSON_HOST=jetson bash deployment/scripts/jetson-rsync-from-mac.sh
#
# LANE=r48 drops the April P1 graph families (backbone/direct4/scenario/pair*, ~1.5 GB of
# .onnx + .onnx.data) from the transfer. The R48 lane does not use them and the CLIP tower
# alone is 335 MB, so over wifi this is the difference between a ~0.5 GB and a ~2 GB push.
# Use LANE=all (default) when a P1 engine actually has to be rebuilt on the device.
#
# Re-run any time you change scripts. Engines (deployment/export/jetson/*.engine) are
# device-specific and never copied back — they're rebuilt on the Jetson.
set -euo pipefail

: "${JETSON_HOST:?Set JETSON_HOST=user@jetson-ip (e.g. phuong@192.168.1.42)}"
: "${REMOTE_DIR:=~/my-project}"
: "${LANE:=all}"

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

LANE_EXCLUDES=()
if [[ "$LANE" == "r48" ]]; then
  for fam in backbone backbone-ft direct4-s314 scenario-s7 pair0-s7 pair1-s7 pair2-s7 pair3-s7; do
    LANE_EXCLUDES+=(--exclude="deployment/export/${fam}-*.onnx"
                    --exclude="deployment/export/${fam}-*.onnx.data")
  done
elif [[ "$LANE" != "all" ]]; then
  echo "LANE must be 'all' or 'r48' (got '$LANE')" >&2
  exit 2
fi

# Stamp the host commit so the device, which has no .git, can record what it measured.
mkdir -p deployment/artifacts
git rev-parse --short HEAD > deployment/artifacts/host-commit.txt 2>/dev/null || \
  echo unknown > deployment/artifacts/host-commit.txt
echo "[rsync] host commit $(cat deployment/artifacts/host-commit.txt)"

echo "[rsync] $PROJECT_DIR  →  $JETSON_HOST:$REMOTE_DIR   (LANE=$LANE)"
echo "[rsync] (excluding features/, runs/, data/train, .git, __pycache__, sweep_results/v6_*, *.pt that aren't deployment artefacts)"

rsync -avhz --progress \
  --exclude='features/' \
  --exclude='runs/' \
  --exclude='data/train/' \
  --exclude='data/val/' \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='.venv/' \
  --exclude='*.pyc' \
  --exclude='sweep_results/v6_*/' \
  --exclude='sweep_results/finetune*/checkpoints/' \
  --exclude='sweep_results/yoloworld*/checkpoints/' \
  --exclude='*.mp4.tmp' \
  --exclude='*.log' \
  "${LANE_EXCLUDES[@]}" \
  --include='deployment/export/*.onnx' \
  --include='deployment/export/*.onnx.data' \
  --include='deployment/artifacts/features/all8_test_onnx.pt' \
  ./ "$JETSON_HOST:$REMOTE_DIR/"

echo ""
echo "[rsync] done."
echo ""
if [[ "$LANE" == "r48" ]]; then
  echo "On the Jetson, run (see deployment/JETSON-RUNBOOK-2026-10-01.md):"
  echo "  python3 $REMOTE_DIR/deployment/scripts/jetson-build-trt-engine.py --mode fp16 \\"
  echo "      --model r48-clip-vit-b32-vision --batch_profile 1:16:16"
  echo "  python3 $REMOTE_DIR/deployment/scripts/jetson-build-trt-engine.py --mode fp16 \\"
  echo "      --model r48-b1-clip-temporal-3seed"
  echo "  python3 $REMOTE_DIR/deployment/scripts/benchmark-r48-clip-pipeline.py \\"
  echo "      --backend tensorrt --mode fp16 --detector --power --json"
else
  echo "On the Jetson, run:"
  echo "  bash $REMOTE_DIR/deployment/scripts/jetson-bootstrap.sh"
  echo "  python3 $REMOTE_DIR/deployment/scripts/jetson-build-trt-engine.py --mode fp16"
  echo "  python3 $REMOTE_DIR/deployment/scripts/jetson-benchmark.py --mode fp16 --runs 100"
fi
