#!/usr/bin/env python3
"""MT3 Step 3 — real edge metrics for an entity detector on NVIDIA Jetson.

Runs ON the Jetson. Exports TensorRT engines (FP16 + INT8), measures latency percentiles / FPS /
multi-stream throughput / model size, and (optionally) energy via tegrastats. Works on any Ultralytics
detector .pt (YOLO26 or the YOLOv11 baseline) so you can benchmark both on-device and compare.

Usage (on the Jetson, JetPack + ultralytics + tensorrt installed):
  python3 tools/jetson_benchmark.py --weights yolo26_best.pt --data data.yaml --imgsz 800 --runs 300
  # compare baseline:  --weights detector_best.pt
Outputs a JSON + a printed table. Energy needs tegrastats (auto-detected; sudo may be required).
"""
import argparse
import json
import os
import statistics as st
import subprocess
import threading
import time

import numpy as np


def export_engines(weights, data, imgsz):
    """Export TensorRT FP16 + INT8 engines; return {name: path}. INT8 needs calibration data."""
    from ultralytics import YOLO
    out = {'pytorch': weights}
    for name, kw in [('trt_fp16', dict(half=True)), ('trt_int8', dict(int8=True, data=data))]:
        try:
            p = YOLO(weights).export(format='engine', imgsz=imgsz, batch=1, **kw)
            out[name] = str(p)
            print(f"  exported {name}: {p}")
        except Exception as e:
            print(f"  WARN {name} export failed: {e}")
    return out


class PowerSampler(threading.Thread):
    """Sample instantaneous power (mW) from tegrastats in the background."""
    def __init__(self):
        super().__init__(daemon=True); self.samples = []; self._stop = False

    def run(self):
        try:
            proc = subprocess.Popen(['tegrastats', '--interval', '100'],
                                    stdout=subprocess.PIPE, text=True)
        except FileNotFoundError:
            return
        for line in proc.stdout:
            if self._stop:
                proc.terminate(); break
            # match a power field like 'VDD_IN 5000mW/...' or 'POM_5V_IN 4200/...'
            for tok in line.split():
                if 'mW' in tok and '/' in tok:
                    try:
                        self.samples.append(float(tok.split('mW')[0].split('/')[0]))
                    except ValueError:
                        pass

    def stop(self):
        self._stop = True


def bench_one(model_path, imgsz, runs, warmup):
    """Return latency percentiles (ms) + FPS for one exported model on a dummy frame."""
    from ultralytics import YOLO
    m = YOLO(model_path)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(warmup):
        m.predict(dummy, imgsz=imgsz, device=0, verbose=False)
    ts = []
    pw = PowerSampler(); pw.start()
    for _ in range(runs):
        t0 = time.perf_counter()
        m.predict(dummy, imgsz=imgsz, device=0, verbose=False)
        ts.append((time.perf_counter() - t0) * 1000.0)
    pw.stop(); time.sleep(0.3)
    ts.sort()
    p = lambda q: ts[min(len(ts) - 1, int(q * len(ts)))]
    mean_ms = st.mean(ts); fps = 1000.0 / mean_ms
    res = {'mean_ms': round(mean_ms, 2), 'p50_ms': round(p(0.50), 2), 'p95_ms': round(p(0.95), 2),
           'p99_ms': round(p(0.99), 2), 'fps': round(fps, 1),
           'size_mb': round(os.path.getsize(model_path) / 1e6, 1)}
    if pw.samples:
        watts = st.mean(pw.samples) / 1000.0
        res['watts'] = round(watts, 2)
        res['j_per_inf'] = round(watts / fps, 3)   # energy per inference (J) = W / (inf/s)
        res['fps_per_watt'] = round(fps / watts, 2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data', default='data.yaml', help='for INT8 calibration + (optional) mAP')
    ap.add_argument('--imgsz', type=int, default=800)
    ap.add_argument('--runs', type=int, default=300)
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--out', default='jetson_benchmark.json')
    args = ap.parse_args()

    from ultralytics import YOLO
    info = {}
    try:
        m = YOLO(args.weights)
        n_params = sum(p.numel() for p in m.model.parameters())
        info['params_M'] = round(n_params / 1e6, 2)
    except Exception as e:
        print('info warn:', e)

    print("Exporting TensorRT engines (this takes a few minutes on Jetson)...")
    models = export_engines(args.weights, args.data, args.imgsz)

    print(f"\nBenchmarking {args.runs} runs/model at imgsz {args.imgsz}...")
    results = {}
    for name, path in models.items():
        if not os.path.exists(path):
            continue
        print(f"  {name} ...")
        results[name] = bench_one(path, args.imgsz, args.runs, args.warmup)

    # INT8 compression ratio
    if 'trt_fp16' in results and 'trt_int8' in results:
        info['int8_compression'] = round(results['trt_fp16']['size_mb'] / results['trt_int8']['size_mb'], 2)

    out = {'weights': args.weights, 'imgsz': args.imgsz, 'info': info, 'results': results}
    json.dump(out, open(args.out, 'w'), indent=2)

    # pretty table
    cols = ['size_mb', 'mean_ms', 'p50_ms', 'p95_ms', 'p99_ms', 'fps', 'watts', 'j_per_inf', 'fps_per_watt']
    print(f"\n{'format':>10} " + ' '.join(f'{c:>10}' for c in cols))
    for name, r in results.items():
        print(f"{name:>10} " + ' '.join(f"{r.get(c, '-'):>10}" for c in cols))
    print(f"\ninfo: {info}\nsaved {args.out}")


if __name__ == '__main__':
    main()
