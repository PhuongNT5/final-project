"""
benchmark-harness.py — Measure latency/FPS/memory across ORT EPs.

Configs: CPU-FP32, CPU-INT8, CoreML-FP32, CoreML-INT8
Stages: backbone / stats-agg / direct4-head / cascade / end-to-end
Outputs: benchmarks.json, benchmarks.md, accuracy-report.txt

Usage:
    .venv/bin/python3 deployment/scripts/benchmark-harness.py [--warmup N] [--runs N]
"""
import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import psutil
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "deployment", "scripts"))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "bench_utils", os.path.join(ROOT, "deployment", "scripts", "bench-utils.py"))
bench_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_utils)

EXPORT_DIR = os.path.join(ROOT, "deployment", "export")
FEAT_PATH = os.path.join(ROOT, "deployment", "artifacts", "features", "all8_test_onnx.pt")
ART_DIR = os.path.join(ROOT, "deployment", "artifacts")

CONFIGS = [("cpu", "fp32"), ("cpu", "int8"), ("coreml", "fp32"), ("coreml", "int8")]
EP_MAP = {
    "cpu": ["CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}


def ensure_features():
    if not os.path.exists(FEAT_PATH):
        print("Features missing — running extract-features-onnx.py...")
        script = os.path.join(ROOT, "deployment", "scripts", "extract-features-onnx.py")
        subprocess.run([sys.executable, script], check=True)
    d = torch.load(FEAT_PATH, map_location="cpu")
    return d["features"].numpy().astype(np.float32), d["labels"].numpy()


def load_sessions(precision, providers):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    def sess(name): return ort.InferenceSession(
        os.path.join(EXPORT_DIR, f"{name}-{precision}.onnx"), so, providers=providers)
    bb = sess("backbone")
    d4 = sess("direct4-s314")
    sc = sess("scenario-s7")
    pairs = [sess(f"pair{i}-s7") for i in range(4)]
    return bb, d4, sc, pairs, bb.get_providers()


def benchmark_config(provider, precision, features, labels, warmup, runs):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if provider == "coreml" and "CoreMLExecutionProvider" not in avail:
        return None, "CoreML EP not available"

    providers = EP_MAP[provider]
    bb_sess, d4_sess, sc_sess, pair_sess, actual_eps = load_sessions(precision, providers)
    bb_inp = bb_sess.get_inputs()[0].name
    d4_inp = d4_sess.get_inputs()[0].name
    sc_inp = sc_sess.get_inputs()[0].name
    pair_inps = [s.get_inputs()[0].name for s in pair_sess]

    sample_feat = features[0:1]  # (1,32,16,512)
    sample_stats = bench_utils.run_stats(sample_feat)  # (1,32,1536)
    dummy_img = np.random.randn(1, 3, 224, 224).astype(np.float32)
    dummy_bgr = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)

    import cv2
    from PIL import Image
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def run_preprocess():
        rgb = cv2.cvtColor(dummy_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        normalized = (arr - _MEAN) / _STD
        return normalized.transpose(2, 0, 1)[np.newaxis]

    def run_bb(): return bb_sess.run(None, {bb_inp: dummy_img})
    def run_stats_fn(): return bench_utils.run_stats(sample_feat)
    def run_d4(): return d4_sess.run(None, {d4_inp: sample_stats})
    def run_cascade():
        sc_p = bench_utils.softmax(sc_sess.run(None, {sc_inp: sample_stats})[0])
        pair_arr = np.concatenate([
            bench_utils.softmax(pair_sess[i].run(None, {pair_inps[i]: sample_stats})[0])[:, 0:1]
            for i in range(4)], axis=1)
        cas = sc_p * pair_arr
        return cas / (cas.sum(axis=1, keepdims=True) + 1e-8)

    print(f"  Warmup ({warmup})...", end=" ", flush=True)
    for _ in range(warmup):
        run_preprocess(); run_bb(); run_d4(); run_cascade()
    print("done")

    def timed(fn, n):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000)
        return np.array(ts)

    rss_before = bench_utils.peak_rss_mb()
    t_pre = timed(run_preprocess, runs)
    t_bb = timed(run_bb, runs)
    t_st = timed(run_stats_fn, runs)
    t_d4 = timed(run_d4, runs)
    t_cas = timed(run_cascade, runs)
    t_e2e = t_bb + t_st + t_cas
    rss_delta = bench_utils.peak_rss_mb() - rss_before

    acc = bench_utils.accuracy_4class(features, labels, bb_sess, d4_sess, sc_sess,
                                      pair_sess, bb_inp, d4_inp, sc_inp, pair_inps)
    return {
        "provider": provider, "precision": precision, "actual_eps": actual_eps,
        "warmup": warmup, "runs": runs,
        "stages": {
            "preprocess": bench_utils.percentiles(t_pre),
            "backbone": bench_utils.percentiles(t_bb),
            "stats_agg": bench_utils.percentiles(t_st),
            "direct4_head": bench_utils.percentiles(t_d4),
            "cascade_heads": bench_utils.percentiles(t_cas),
            "end_to_end": bench_utils.percentiles(t_e2e),
        },
        "fps": round(1000.0 / float(np.percentile(t_e2e, 50)), 1),
        "peak_rss_delta_mb": round(rss_delta, 1),
        "accuracy_4class": acc,
    }, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(ART_DIR, exist_ok=True)
    features, labels = ensure_features()
    model_sizes = bench_utils.model_sizes_mb(EXPORT_DIR)

    results, skipped = [], []
    for provider, precision in CONFIGS:
        tag = f"{provider.upper()}-{precision.upper()}"
        print(f"\n{'='*50}\nBenchmarking: {tag}")
        result, err = benchmark_config(provider, precision, features, labels, args.warmup, args.runs)
        if err:
            print(f"  [SKIP] {err}")
            skipped.append({"config": tag, "reason": err})
        else:
            results.append(result)
            e = result["stages"]["end_to_end"]
            print(f"  e2e p50={e['p50']:.1f}ms fps={result['fps']}  "
                  f"ensemble={result['accuracy_4class']['ensemble']}")
        gc.collect()

    summary = {
        "machine": platform.node(), "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
        "configs": results, "skipped": skipped,
        "model_sizes_mb": model_sizes,
    }

    json_path = os.path.join(ART_DIR, "benchmarks.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVED] {json_path}")

    md_path = os.path.join(ART_DIR, "benchmarks.md")
    with open(md_path, "w") as f:
        f.write(bench_utils.write_benchmarks_md(results, model_sizes))
    print(f"[SAVED] {md_path}")

    acc_path = os.path.join(ART_DIR, "accuracy-report.txt")
    with open(acc_path, "w") as f:
        f.write(bench_utils.write_accuracy_report(results))
    print(f"[SAVED] {acc_path}")


if __name__ == "__main__":
    main()
