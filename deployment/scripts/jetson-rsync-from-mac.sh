#!/usr/bin/env bash
# Run ON the MacBook (M4) to push code, ONNX models, and test features to the Jetson.
# Excludes large caches that aren't needed on the edge.
#
# Usage (on M4):
#   JETSON_HOST=phuong@192.168.1.42 bash deployment/scripts/jetson-rsync-from-mac.sh
#
# Re-run any time you change scripts. Engines (deployment/export/jetson/*.engine) are
# device-specific and never copied back — they're rebuilt on the Jetson.
set -euo pipefail

: "${JETSON_HOST:?Set JETSON_HOST=user@jetson-ip (e.g. phuong@192.168.1.42)}"
: "${REMOTE_DIR:=~/my-project}"

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

echo "[rsync] $PROJECT_DIR  →  $JETSON_HOST:$REMOTE_DIR"
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
  --include='deployment/export/*.onnx' \
  --include='deployment/export/*.onnx.data' \
  --include='deployment/artifacts/features/all8_test_onnx.pt' \
  ./ "$JETSON_HOST:$REMOTE_DIR/"

echo ""
echo "[rsync] done."
echo ""
echo "On the Jetson, run:"
echo "  bash $REMOTE_DIR/deployment/scripts/jetson-bootstrap.sh"
echo "  python3 $REMOTE_DIR/deployment/scripts/jetson-build-trt-engine.py --mode fp16"
echo "  python3 $REMOTE_DIR/deployment/scripts/jetson-benchmark.py --mode fp16 --runs 100"
