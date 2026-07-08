# Jetson Orin Nano Quickstart — Days 0–17

End-to-end execution path from "unboxing" to "Chapter 5 §5.6 measured numbers".
Tied to [`plans/260430-2336-thesis-completion/phase-01-jetson-edge-measurement.md`](../plans/260430-2336-thesis-completion/phase-01-jetson-edge-measurement.md).

Hardware confirmed available: MacBook + Orin Nano Dev Kit + USB-C PD 65 W + 128 GB microSD + Ethernet.

## Day 0 — Flash + first boot (~2 h)

On the **MacBook**:

1. Download JetPack 6.x SD image: <https://developer.nvidia.com/embedded/jetpack>
2. Flash with Balena Etcher (drag → select SD → flash) **or** `dd`:
   ```bash
   diskutil list && diskutil unmountDisk /dev/diskN
   sudo dd if=jetson-orin-nano-sd-card-image.img of=/dev/rdiskN bs=4m status=progress
   ```
3. Insert SD into Jetson, attach HDMI + USB keyboard, connect Ethernet, plug USB-C power.
4. First-boot wizard: locale, user (`phuong`), password, network.
5. On the Jetson HDMI session: `ip addr show` → note the IP (e.g. `192.168.1.42`).

Back on the **MacBook**:

```bash
ssh-copy-id phuong@192.168.1.42
ssh phuong@192.168.1.42 "echo 'SSH OK' && uname -a && df -h /"
```

Once SSH works, HDMI + keyboard can be unplugged. Everything from here is over the network.

## Day 0.5 — Push project from MacBook (~5 min)

```bash
cd /Users/nguyenthiphuong/Learning/Proposal/project/my-project
JETSON_HOST=phuong@192.168.1.42 bash deployment/scripts/jetson-rsync-from-mac.sh
```

This pushes code + ONNX exports + test features (~500 MB), excluding `features/`, `runs/`,
`data/train/`, and other caches you don't need on the edge.

## Day 1 — Bootstrap Jetson software (~45 min)

On the **Jetson** (via SSH):

```bash
bash ~/my-project/deployment/scripts/jetson-bootstrap.sh 2>&1 | tee ~/jetson-bootstrap.log
```

This installs: jetson-stats (jtop), PyTorch (JetPack 6 wheel), ONNX Runtime GPU + TRT EP,
pycuda, project Python deps. Sets MAXN power profile + locked clocks. Verifies TensorRT
and all execution providers.

At the end you should see:
```
Power mode:    MAXN
PyTorch:       2.3.0
TensorRT:      10.x
ORT:           1.18.0
```

If `TensorrtExecutionProvider missing` → wrong ORT wheel. See
<https://elinux.org/Jetson_Zoo#ONNX_Runtime> for the right URL for your JetPack revision.

## Day 2 — Build TensorRT engines + first benchmark (~1 h)

On the **Jetson**:

```bash
cd ~/my-project

# Build FP16 engines (default Orin Nano deployment target)
python3 deployment/scripts/jetson-build-trt-engine.py --mode fp16
# Builds ~7 engines; expect ~2–3 min total. Engines saved to deployment/export/jetson/

# First benchmark — 100 warm runs, batch=1, 32-frame window
python3 deployment/scripts/jetson-benchmark.py --mode fp16 --runs 100 --warmup 10
# Output: deployment/artifacts/benchmarks-jetson-fp16.json
```

In a second SSH session, run `jtop` to watch power + GPU + thermal in real time.

Sanity-check expected ranges:
- backbone p50: 6–10 ms
- direct4 + cascade: <1 ms each
- e2e p50: 10–15 ms / 65–100 FPS
- accuracy: 56/64 (matches M4 ensemble)

## Day 3 — INT8 calibration + comparison (~2 h)

```bash
# Prep calibration batches (uses train features)
python3 deployment/scripts/jetson-prep-int8-calib.py

# Build INT8 engines with entropy calibrator
python3 deployment/scripts/jetson-build-trt-engine.py --mode int8 \
  --calib-dir deployment/artifacts/calib-jetson

# Benchmark INT8
python3 deployment/scripts/jetson-benchmark.py --mode int8 --runs 100

# Also benchmark FP32 for the cross-precision row in Ch 5 Table 5.5
python3 deployment/scripts/jetson-build-trt-engine.py --mode fp32
python3 deployment/scripts/jetson-benchmark.py --mode fp32 --runs 100
```

Three result JSONs → roll into a single `benchmarks-jetson.md` table for Ch 5.

## Day 4 — Multi-stream (N=1..4) + soak (~3 h)

```bash
# Background power log
nohup tegrastats --interval 1000 --logfile ~/tegrastats-multistream.log &
TEGRA_PID=$!

# Multi-stream pipeline
python3 deployment/scripts/multi-stream-pipeline.py \
  --providers tensorrt,cuda \
  --streams 1 2 3 4 \
  --duration 120 \
  --out deployment/artifacts/multi-stream-jetson.json

# 30-min soak at N=2 (matches M4 protocol)
python3 deployment/scripts/multi-stream-pipeline.py \
  --providers tensorrt,cuda \
  --streams 2 --duration 1800 \
  --out deployment/artifacts/soak-30min-jetson.json

kill $TEGRA_PID
```

Acceptance: zero dropped frames; thermal stable < 80 °C.

## Day 5 — Power profiling at three nvpmodel settings (~2 h)

```bash
for MODE in 0 1 2; do          # 0 = MAXN, 1 = 15W, 2 = 7W
  sudo nvpmodel -m $MODE
  sudo jetson_clocks
  echo "=== nvpmodel=$MODE ==="
  nohup tegrastats --interval 500 --logfile ~/tegrastats-mode$MODE.log &
  TEGRA=$!
  python3 deployment/scripts/jetson-benchmark.py --mode fp16 --runs 100
  kill $TEGRA
  cp deployment/artifacts/benchmarks-jetson-fp16.json \
     deployment/artifacts/benchmarks-jetson-fp16-pwr$MODE.json
done
sudo nvpmodel -m 0  # restore MAXN
```

Three rows for the Ch 5 §5.6 power-vs-throughput Pareto plot.

## Day 6–7 — Pull results back + write §5.6 (~1 d)

On the **MacBook**:

```bash
rsync -avz phuong@192.168.1.42:~/my-project/deployment/artifacts/benchmarks-jetson*.json \
  deployment/artifacts/
rsync -avz phuong@192.168.1.42:~/my-project/deployment/artifacts/multi-stream-jetson.json \
  deployment/artifacts/
rsync -avz phuong@192.168.1.42:~/my-project/deployment/artifacts/soak-30min-jetson.json \
  deployment/artifacts/
rsync -avz phuong@192.168.1.42:~/tegrastats-*.log deployment/artifacts/
```

Update [`report/chapter-05-edge-deployment.md`](../report/chapter-05-edge-deployment.md) §5.6:
replace projection text with measured table; update GROUND_TRUTH.md §1 with Jetson rows.

## Common pitfalls

| Symptom | Fix |
|---|---|
| `ImportError: libcudnn.so.8` | PyTorch wheel ≠ JetPack version. Recheck `cat /etc/nv_tegra_release` → wheel URL. |
| `TensorrtExecutionProvider` absent | Wrong ORT build. Use Jetson Zoo wheel for your JetPack revision. |
| TRT build hangs > 5 min on backbone | Workspace too small. Edit `WORKSPACE_GB = 4` → `8` in `jetson-build-trt-engine.py`. |
| First inference takes 3 min | Normal — TRT JIT. Engine then cached. Warm up before timing. |
| Random hangs / OOM at N=4 streams | 8 GB shared CPU+GPU memory. Drop to N=3 or input 192². |
| `tegrastats` shows 0 W | INA3221 read needs sudo or systemd-tmpfiles regen. Run with `sudo tegrastats`. |
| Fan never spins | `sudo /usr/sbin/jetson_clocks --fan` to force max. |
| Engine file too big to commit | `.engine` files are device-specific binaries — leave in `deployment/export/jetson/`, gitignored. |

## Backup plan

If anything blocks for > 0.5 days, see Phase 01 plan §"Branch B / Branch C" for cloud Jetson
(NVIDIA LaunchPad) or x86 + GPU fallback. Don't sink more than half a day before pivoting.

## Daily rsync-back to MacBook (data-safety habit)

After each evening's experiments, on the MacBook:
```bash
rsync -avz phuong@192.168.1.42:~/my-project/deployment/artifacts/ \
  deployment/artifacts/ \
  --include='benchmarks-jetson*' --include='*-jetson.json' \
  --include='multi-stream-jetson*' --include='soak-30min-jetson*' \
  --include='tegrastats-*.log'
```

SD-card failure mid-experiment costs only the latest day, not the week.
