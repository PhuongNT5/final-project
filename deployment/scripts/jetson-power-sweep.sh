#!/usr/bin/env bash
# jetson-power-sweep.sh — Run FP16 benchmark at each available nvpmodel profile
# and capture per-mode tegrastats so we can compute the FPS-vs-Power Pareto for §5.6.
#
# Runs ON the Jetson. Idempotent — re-running overwrites the per-mode JSON/log.
#
# Output:
#   ~/my-project/deployment/artifacts/benchmarks-jetson-fp16-pwr{0,1,2}.json
#   ~/tegrastats-pwr{0,1,2}.log
#
# Usage (on Jetson):
#   bash ~/my-project/deployment/scripts/jetson-power-sweep.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/my-project}"
ART_DIR="$PROJECT_DIR/deployment/artifacts"
cd "$PROJECT_DIR"

MODES=(0 1 2)

echo "=== Available nvpmodel profiles ==="
sudo nvpmodel -p | head -15
echo ""

for MODE in "${MODES[@]}"; do
  echo ""
  echo "================================================================"
  echo "=== nvpmodel=$MODE  (start at $(date +%H:%M:%S)) ==="
  echo "================================================================"

  sudo nvpmodel -m "$MODE"
  sudo jetson_clocks
  sleep 3
  echo "Active profile: $(sudo nvpmodel -q | tail -1)"

  # Start tegrastats logging at 500ms granularity
  TEGRA_LOG="$HOME/tegrastats-pwr${MODE}.log"
  : > "$TEGRA_LOG"
  sudo tegrastats --interval 500 > "$TEGRA_LOG" 2>&1 &
  TEGRA_PID=$!
  sleep 1

  # Run FP16 benchmark
  python3 deployment/scripts/jetson-benchmark.py --mode fp16 --runs 100 --warmup 10 \
    || { sudo kill "$TEGRA_PID" 2>/dev/null; echo "[ERR] bench failed at mode=$MODE"; continue; }

  # Stop tegrastats
  sudo kill "$TEGRA_PID" 2>/dev/null || true
  sleep 1

  # Rename per-mode JSON
  if [[ -f "$ART_DIR/benchmarks-jetson-fp16.json" ]]; then
    mv "$ART_DIR/benchmarks-jetson-fp16.json" "$ART_DIR/benchmarks-jetson-fp16-pwr${MODE}.json"
    echo "[SAVED] benchmarks-jetson-fp16-pwr${MODE}.json"
  fi

  # Brief thermal cooldown between modes
  echo "Cooling down 10s before next mode ..."
  sleep 10
done

# Always restore MAXN at end
echo ""
echo "=== restoring MAXN (mode 0) ==="
sudo nvpmodel -m 0
sudo jetson_clocks
echo "Final profile: $(sudo nvpmodel -q | tail -1)"

echo ""
echo "=== Per-mode JSONs ==="
ls -lh "$ART_DIR"/benchmarks-jetson-fp16-pwr*.json 2>/dev/null

echo ""
echo "=== Per-mode tegrastats logs ==="
ls -lh "$HOME"/tegrastats-pwr*.log 2>/dev/null

echo ""
echo "Sweep complete at $(date +%H:%M:%S)."
