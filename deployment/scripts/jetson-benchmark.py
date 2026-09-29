#!/usr/bin/env python3
"""F11 — Jetson Orin TRT benchmark harness (mirrors benchmark-harness.py on M4).

Loads pre-built TRT engines from deployment/export/jetson/ and reports
end-to-end p50/p95 latency, FPS, peak memory, and 4-class accuracy on the
held-out test features. Designed for JetPack 6 (CUDA 12.x, TRT 10.x).

Outputs: deployment/artifacts/benchmarks-jetson.json + .md

Run on the Jetson, NOT on M4.

Usage:
    python3 deployment/scripts/jetson-benchmark.py --mode fp16 --runs 100
"""
import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE_DIR = os.path.join(ROOT, 'deployment', 'export', 'jetson')
FEAT_PATH = os.path.join(ROOT, 'deployment', 'artifacts', 'features', 'all8_test_onnx.pt')
ART_DIR = os.path.join(ROOT, 'deployment', 'artifacts')

MODELS = ['backbone', 'direct4-s314', 'scenario-s7',
          'pair0-s7', 'pair1-s7', 'pair2-s7', 'pair3-s7']

# TRTRunner binds all engine inputs/outputs by NAME (needed for the P3 relation head,
# which has 2 inputs: tokens + mask — see trt-runner-utils.py docstring for the bug this
# fixes). Loaded via importlib (hyphenated filename); shared with
# jetson-multi-stream-pipeline.py so the fix lives in one place.
_tru_spec = importlib.util.spec_from_file_location(
    'trt_runner_utils', os.path.join(ROOT, 'deployment', 'scripts', 'trt-runner-utils.py'))
_tru = importlib.util.module_from_spec(_tru_spec)
_tru_spec.loader.exec_module(_tru)
TRTRunner = _tru.TRTRunner


def import_trt():
    try:
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401  (initialises CUDA context)
        import pycuda.driver as cuda
        return trt, cuda
    except ImportError as e:
        sys.exit(f'ERROR: {e}. Need tensorrt + pycuda (JetPack 6 default).')


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def stats_pool_np(feats):  # (1,32,16,512) -> (1,32,1536)
    return np.concatenate([feats.mean(2), feats.std(2), feats.max(2)], axis=-1)


def benchmark(trt, cuda, mode, features, labels, warmup, runs):
    runners = {}
    for m in MODELS:
        path = os.path.join(ENGINE_DIR, f'{m}-{mode}.engine')
        if not os.path.exists(path):
            sys.exit(f'engine missing: {path}; run jetson-build-trt-engine.py --mode {mode}')
        runners[m] = TRTRunner(trt, cuda, path)

    sample_feat = features[0:1]
    sample_stats = stats_pool_np(sample_feat).astype(np.float32)
    dummy_img = np.random.randn(1, 3, 224, 224).astype(np.float32)

    def run_bb(): return runners['backbone'].run(dummy_img)
    def run_d4(): return runners['direct4-s314'].run(sample_stats)
    def run_cas():
        sc = softmax(runners['scenario-s7'].run(sample_stats))
        # Pair models output [P(safe), P(unsafe)]. Cascade weight is P(unsafe) =
        # column 1, matching M4 bench-utils.py accuracy_4class which uses [0, 1].
        pair = np.concatenate(
            [softmax(runners[f'pair{i}-s7'].run(sample_stats))[:, 1:2] for i in range(4)],
            axis=1)
        cas = sc * pair
        return cas / (cas.sum(1, keepdims=True) + 1e-8)

    print(f'  warmup ({warmup})...', end=' ', flush=True)
    for _ in range(warmup): run_bb(); run_d4(); run_cas()
    print('done')

    def timed(fn):
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter(); fn()
            ts.append((time.perf_counter() - t0) * 1000)
        return np.array(ts)

    t_bb = timed(run_bb); t_d4 = timed(run_d4); t_cas = timed(run_cas)
    t_e2e = t_bb + t_d4 + t_cas

    # Accuracy on full test set (uses real features, not dummy).
    # Pair models output [P(safe), P(unsafe)] — cascade uses P(unsafe) (col 1)
    # to match M4 bench-utils.py reference implementation.
    correct = 0
    for i in range(len(features)):
        s = stats_pool_np(features[i:i + 1]).astype(np.float32)
        d4 = softmax(runners['direct4-s314'].run(s))
        sc = softmax(runners['scenario-s7'].run(s))
        pair = np.concatenate(
            [softmax(runners[f'pair{i_}-s7'].run(s))[:, 1:2] for i_ in range(4)],
            axis=1)
        cas = sc * pair
        cas = cas / (cas.sum(1, keepdims=True) + 1e-8)
        ens = (d4 + cas) / 2
        # Match M4 reference: renormalize ensemble before argmax (doesn't change
        # argmax for non-negative inputs, but kept for parity).
        ens = ens / (ens.sum(1, keepdims=True) + 1e-8)
        if int(ens.argmax(1)[0]) == int(labels[i]):
            correct += 1

    def pct(arr):
        return {p: round(float(np.percentile(arr, p)), 2) for p in (50, 95, 99)}

    return {
        'mode': mode, 'warmup': warmup, 'runs': runs,
        'backbone_ms': pct(t_bb),
        'direct4_ms': pct(t_d4),
        'cascade_ms': pct(t_cas),
        'end_to_end_ms': pct(t_e2e),
        'fps': round(1000.0 / float(np.percentile(t_e2e, 50)), 1),
        'accuracy_4class_correct': correct,
        'accuracy_4class_total': len(features),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fp32', 'fp16', 'int8'], default='fp16')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--runs', type=int, default=100)
    args = ap.parse_args()

    trt, cuda = import_trt()
    d = torch.load(FEAT_PATH, map_location='cpu', weights_only=False)
    features = d['features'].numpy().astype(np.float32)
    labels = d['labels'].numpy()
    # Filter 4-class subset
    mask = labels < 4
    features, labels = features[mask], labels[mask]
    print(f'TRT {trt.__version__} | {platform.node()} | mode={args.mode} | N={len(labels)}')

    res = benchmark(trt, cuda, args.mode, features, labels, args.warmup, args.runs)
    summary = {
        'machine': platform.node(),
        'platform': platform.platform(),
        'tensorrt_version': trt.__version__,
        'timestamp': datetime.now().isoformat(),
        'config': res,
    }
    os.makedirs(ART_DIR, exist_ok=True)
    out = os.path.join(ART_DIR, f'benchmarks-jetson-{args.mode}.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[SAVED] {out}')
    print(f"e2e p50={res['end_to_end_ms'][50]} ms | "
          f"FPS={res['fps']} | acc={res['accuracy_4class_correct']}/"
          f"{res['accuracy_4class_total']}")


if __name__ == '__main__':
    main()
