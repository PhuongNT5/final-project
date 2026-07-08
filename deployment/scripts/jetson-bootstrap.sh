#!/usr/bin/env bash
# Day 0 bootstrap — run ON the Jetson Orin Nano after first SSH login.
# Installs: jetson-stats, PyTorch (JetPack 6 wheel), ONNX Runtime GPU+TRT, pycuda, project deps.
# Sets nvpmodel to MAXN, locks clocks, verifies TensorRT + CUDA + EP availability.
#
# Usage (on Jetson):
#   bash deployment/scripts/jetson-bootstrap.sh 2>&1 | tee ~/jetson-bootstrap.log
#
# Idempotent — safe to re-run.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/my-project}"
JETPACK_RELEASE_FILE="/etc/nv_tegra_release"
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------- 0. Pre-flight checks ----------
LOG "=== Day 0 Jetson Bootstrap ==="
if [[ ! -f "$JETPACK_RELEASE_FILE" ]]; then
  echo "ERROR: not a Jetson (no $JETPACK_RELEASE_FILE). Run this on the Orin Nano, not the M4." >&2
  exit 1
fi
LOG "JetPack release: $(head -1 $JETPACK_RELEASE_FILE)"
LOG "Architecture: $(uname -m)"
LOG "Python: $(python3 --version)"

# ---------- 1. Power profile: MAXN + locked clocks ----------
LOG "--- 1. Setting MAXN (full 15 W envelope) ---"
sudo nvpmodel -m 0
sudo jetson_clocks
sudo nvpmodel -q  # show active mode
LOG "Active power profile: $(sudo nvpmodel -q | tail -1)"

# ---------- 2. System packages ----------
# Note: libcudnn9-cuda-12 already ships with nvidia-jetpack 6.2 — no need to install libcudnn*.
LOG "--- 2. System packages ---"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  python3-pip python3-dev python3-venv \
  libopenblas-dev libomp-dev \
  libjpeg-dev zlib1g-dev libpython3-dev \
  libavcodec-dev libavformat-dev libswscale-dev \
  curl rsync htop nano build-essential
python3 -m pip install --upgrade pip wheel setuptools

# ---------- 3. jetson-stats (jtop) for monitoring ----------
LOG "--- 3. jetson-stats ---"
if ! command -v jtop &>/dev/null; then
  sudo -H pip3 install -U jetson-stats
  sudo systemctl restart jtop.service || true
fi
LOG "jtop installed: $(jtop --version 2>&1 | head -1 || echo 'pending service restart')"

# ---------- 4. PyTorch (JetPack 6.2 / CUDA 12.6) ----------
# Multi-source download — NVIDIA developer CDN first (works from VN), then Box mirror, then jetson-ai-lab.
LOG "--- 4. PyTorch (JetPack 6.2 ARM64 wheel) ---"
if ! python3 -c "import torch" 2>/dev/null; then
  pip3 install "numpy<2"
  TORCH_WHL="torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
  TORCH_TMP="/tmp/$TORCH_WHL"

  if [[ ! -f "$TORCH_TMP" ]]; then
    LOG "Trying NVIDIA developer CDN ..."
    wget -q --show-progress -O "$TORCH_TMP" \
      "https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/$TORCH_WHL" \
      || { LOG "developer CDN failed — trying NVIDIA Box mirror ..."; \
           wget -q --show-progress -O "$TORCH_TMP" \
             "https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl"; }

    if [[ ! -s "$TORCH_TMP" ]] || [[ $(stat -c%s "$TORCH_TMP" 2>/dev/null || echo 0) -lt 1000000 ]]; then
      rm -f "$TORCH_TMP"
      LOG "Both wheel mirrors failed — trying jetson-ai-lab pip index ..."
      pip3 install --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126 torch torchvision \
        || { echo "ERROR: All PyTorch sources failed. Possible DNS/network issue. See https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048" >&2; exit 1; }
    else
      LOG "Wheel downloaded: $(du -h $TORCH_TMP | cut -f1)"
      pip3 install "$TORCH_TMP"
    fi
  else
    LOG "Wheel cached at $TORCH_TMP — installing"
    pip3 install "$TORCH_TMP"
  fi
fi
python3 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ---------- 5. ONNX Runtime GPU + TensorRT EP (JetPack 6.2 build) ----------
LOG "--- 5. ONNX Runtime (TensorRT EP) for JetPack 6.2 ---"
if ! python3 -c "import onnxruntime" 2>/dev/null; then
  LOG "Installing onnxruntime-gpu from pypi.jetson-ai-lab.dev/jp6/cu126 ..."
  pip3 install --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126 onnxruntime-gpu \
    || { LOG "jetson-ai-lab index failed — falling back to Jetson-Zoo wheel"; \
         ORT_WHL="onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl"; \
         wget -q -O "/tmp/$ORT_WHL" \
           "https://nvidia.box.com/shared/static/zostg6agm00fb6t5uisw51qi6kpcuwzd.whl" \
           && pip3 install "/tmp/$ORT_WHL" \
           || pip3 install onnxruntime-gpu; }
fi
python3 -c "
import onnxruntime as ort
ps = ort.get_available_providers()
print(f'ONNX Runtime {ort.__version__}')
print(f'Available providers: {ps}')
assert 'TensorrtExecutionProvider' in ps, 'TensorRT EP missing — see https://elinux.org/Jetson_Zoo#ONNX_Runtime'
"

# ---------- 6. pycuda (for tensorrt python bindings) ----------
LOG "--- 6. pycuda ---"
if ! python3 -c "import pycuda" 2>/dev/null; then
  pip3 install pycuda
fi
python3 -c "import pycuda.autoinit, pycuda.driver as cuda; print(f'CUDA driver: {cuda.get_version()}')"

# ---------- 7. TensorRT (ships with JetPack — verify only) ----------
LOG "--- 7. TensorRT ---"
python3 -c "import tensorrt as trt; print(f'TensorRT {trt.__version__}')"

# ---------- 8. Project deps ----------
LOG "--- 8. Project Python deps ---"
pip3 install --no-cache-dir \
  numpy pandas matplotlib seaborn \
  opencv-python-headless \
  scikit-learn

# ---------- 9. Project layout sanity ----------
LOG "--- 9. Project layout ---"
if [[ -d "$PROJECT_DIR" ]]; then
  LOG "Found project at $PROJECT_DIR"
  cd "$PROJECT_DIR"
  ls -la deployment/scripts/jetson-*.py 2>/dev/null || LOG "Jetson scripts not yet rsynced from M4"
  ls -la deployment/export/*-fp32.onnx 2>/dev/null | head -5 || LOG "ONNX models not yet rsynced from M4"
else
  LOG "Project not present yet. Run jetson-rsync-from-mac.sh on the M4 to push it."
fi

# ---------- 10. Summary ----------
LOG ""
LOG "=== Bootstrap complete ==="
LOG "Power mode:    $(sudo nvpmodel -q | tail -1)"
LOG "Python:        $(python3 --version)"
LOG "PyTorch:       $(python3 -c 'import torch;print(torch.__version__)')"
LOG "TensorRT:      $(python3 -c 'import tensorrt as t;print(t.__version__)')"
LOG "ORT:           $(python3 -c 'import onnxruntime as o;print(o.__version__)')"
LOG "Disk free:     $(df -h / | tail -1 | awk '{print $4}')"
LOG ""
LOG "Next: run \`jtop\` in another terminal to watch power/temp, then"
LOG "      cd $PROJECT_DIR && python3 deployment/scripts/jetson-build-trt-engine.py --mode fp16"
