"""Host-side latency of the R48 deployment path (CLIP ViT-B/32 -> B1 temporal head), onnxruntime CPU.

**This is NOT a Jetson number and must never be quoted as one.** It exists to size the trade-off the
R48 accuracy result created before any device time is spent: the head shrank to 201k parameters, but
the backbone it depends on grew from MobileNetV3-small (~2.5M) to CLIP ViT-B/32 (~87.8M). If the
backbone dominates -- and it will -- then "small head" is not an edge claim, and the edge chapter has
to say so.

One clip = 16 frames through the tower + one head call, matching the evaluated configuration
exactly (T=16, whole-frame, no detector in B1's path).

Usage: .venv/bin/python deployment/scripts/benchmark-r48-clip-pipeline.py [--runs 20]
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

EXPORT = os.path.join(ROOT, 'deployment', 'export')
TOWER = os.path.join(EXPORT, 'r48-clip-vit-b32-vision-fp32.onnx')
HEAD = os.path.join(EXPORT, 'r48-b1-clip-temporal-3seed-fp32.onnx')
LEGACY_APP = os.path.join(EXPORT, 'p3-appearance-fp32.onnx')
FRAMES = 16


def session(path):
    import onnxruntime as ort
    if not os.path.exists(path):
        return None
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, o, providers=['CPUExecutionProvider'])


def timeit(sess, feeds, runs, warmup=3):
    for _ in range(warmup):
        sess.run(None, feeds)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, feeds)
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), float(np.percentile(ts, 95))


def report(name, med, p95, note=''):
    print(f'  {name:<44} {med:8.1f} ms   p95 {p95:7.1f} ms  {note}')
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=20)
    args = ap.parse_args()

    tower, head = session(TOWER), session(HEAD)
    if tower is None or head is None:
        raise SystemExit('export the graphs first: export-clip-vision-tower.py and '
                         'export-b1-clip-temporal-head.py')

    import platform
    print(f'[host] {platform.machine()} {platform.system()} {platform.release()}, '
          f'onnxruntime CPU EP, {args.runs} runs after 3 warmups')
    print(f'[config] one clip = {FRAMES} frames, whole-frame (B1 uses NO detector)\n')

    x = np.random.randn(FRAMES, 3, 224, 224).astype('float32')
    t_med, t_p95 = timeit(tower, {'pixel_values': x}, args.runs)
    total = report(f'CLIP ViT-B/32 tower, batch {FRAMES}', t_med, t_p95, '(87,849,216 params)')

    f = np.random.randn(1, FRAMES, 512).astype('float32')
    v = np.ones((1, FRAMES), dtype='float32')
    h_med, h_p95 = timeit(head, {'frames': f, 'valid': v}, args.runs)
    total += report('B1 temporal head, 3-seed ensemble', h_med, h_p95, '(603,270 params)')

    print(f"\n  {'END-TO-END per clip':<44} {total:8.1f} ms   "
          f"-> {1000 / total:.2f} clips/s, {FRAMES * 1000 / total:.1f} frames/s")
    print(f'  backbone share: {100 * (total - h_med) / total:.1f}% of wall clock')

    legacy = session(LEGACY_APP)
    if legacy is not None:
        i = legacy.get_inputs()[0]
        shape = [FRAMES if isinstance(d, str) or d is None else d for d in i.shape]
        l_med, l_p95 = timeit(legacy, {i.name: np.random.randn(*shape).astype('float32')}, args.runs)
        print(f'\n[reference] the representation this lane replaced:')
        report(f'MobileNetV3-small appearance, batch {shape[0]}', l_med, l_p95,
               '(p3-appearance-fp32.onnx)')
        print(f'  CLIP costs {t_med / l_med:.1f}x the appearance backbone it replaced, for '
              f'+4.0 pp balanced accuracy (cluster CI includes 0).')

    print('\n[caveat] host CPU only. Jetson Orin Nano Super numbers require the device and are '
          'measured by deployment/scripts/jetson-*.py; do not substitute these.')


if __name__ == '__main__':
    main()
