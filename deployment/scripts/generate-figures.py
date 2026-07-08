"""
generate-figures.py — Render thesis Chapter 5 figures from JSON benchmark data.

Outputs (PNG @ 150 dpi):
  - latency-by-stage.png    : stacked bars per config (CPU-FP32/CPU-INT8/CoreML-FP32/CoreML-INT8)
  - fps-vs-streams.png      : multi-stream throughput scaling + ideal linear reference
  - accuracy-by-config.png  : ensemble accuracy across configs (baseline reference line)
  - model-size-comparison.png : FP32 vs INT8 model sizes (bytes)

Usage:
    .venv/bin/python3 deployment/scripts/generate-figures.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BM_PATH = os.path.join(ROOT, "deployment", "artifacts", "benchmarks.json")
MS_PATH = os.path.join(ROOT, "deployment", "artifacts", "multi-stream-results.json")
FIG_DIR = os.path.join(ROOT, "deployment", "artifacts", "figures")
EXPORT_DIR = os.path.join(ROOT, "deployment", "export")

STAGE_KEYS = ["backbone", "stats_agg", "direct4_head", "cascade_heads"]
STAGE_LABELS = ["Backbone", "StatsAgg", "Direct4 Head", "Cascade Heads"]


def _cfg_label(c):
    return f"{c['provider'].upper()}-{c['precision'].upper()}"


def plot_latency_by_stage(bm) -> str:
    configs = bm["configs"]
    labels = [_cfg_label(c) for c in configs]
    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = plt.get_cmap("tab10").colors
    for i, (key, lbl) in enumerate(zip(STAGE_KEYS, STAGE_LABELS)):
        vals = np.array([c["stages"][key]["p50"] for c in configs])
        ax.bar(x, vals, bottom=bottoms, label=lbl, color=colors[i])
        bottoms += vals
    e2e = [c["stages"]["end_to_end"]["p50"] for c in configs]
    for xi, v in zip(x, e2e):
        ax.text(xi, v + 1.5, f"{v:.1f}ms", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Latency p50 (ms)")
    ax.set_title("End-to-End Latency by Stage (batch=1, 100 runs, Apple M4)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "latency-by-stage.png")
    plt.savefig(out, dpi=150); plt.close()
    return out


def plot_fps_vs_streams(ms) -> str:
    configs = sorted(ms["configs"], key=lambda c: c["n_streams"])
    n = [c["n_streams"] for c in configs]
    fps = [c["aggregate_fps"] for c in configs]
    ideal = [n[i] * fps[0] for i in range(len(n))]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(n, fps, "bo-", lw=2, ms=8, label="Measured aggregate FPS")
    ax.plot(n, ideal, "k--", lw=1.3, alpha=0.6, label="Ideal linear scaling")
    for xi, v in zip(n, fps):
        ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlabel("Number of Concurrent Streams (N)")
    ax.set_ylabel("Aggregate FPS")
    ax.set_title("Multi-Stream Throughput Scaling on Apple M4 CPU")
    ax.set_xticks(n)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fps-vs-streams.png")
    plt.savefig(out, dpi=150); plt.close()
    return out


def plot_accuracy_by_config(bm) -> str:
    configs = bm["configs"]
    labels = [_cfg_label(c) for c in configs]
    acc = [int(c["accuracy_4class"]["ensemble"].split("/")[0]) for c in configs]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, acc, color=["#2b7bba" if a == 56 else "#a0a0a0" for a in acc])
    ax.axhline(56, color="g", ls="--", lw=1.2, label="Training baseline 56/64")
    for b, v in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v}/64", ha="center", fontsize=10)
    ax.set_ylabel("Ensemble top-1 accuracy (count out of 64)")
    ax.set_ylim(40, 60)
    ax.set_title("Accuracy Preservation Across Execution Providers")
    ax.legend(loc="lower right")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "accuracy-by-config.png")
    plt.savefig(out, dpi=150); plt.close()
    return out


def plot_model_sizes() -> str:
    files = sorted(f for f in os.listdir(EXPORT_DIR) if f.endswith(".onnx"))
    sizes = {f: os.path.getsize(os.path.join(EXPORT_DIR, f)) / 1e6 for f in files}
    groups = {}
    for f, mb in sizes.items():
        base = f.rsplit("-", 1)[0]
        prec = "int8" if "int8" in f else "fp32"
        groups.setdefault(base, {})[prec] = mb
    labels = sorted(groups.keys())
    fp = [groups[k].get("fp32", 0) for k in labels]
    q = [groups[k].get("int8", 0) for k in labels]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, fp, w, label="FP32", color="#3b82f6")
    ax.bar(x + w / 2, q, w, label="INT8", color="#f59e0b")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Size (MB)")
    ax.set_title("ONNX Model Sizes — FP32 vs INT8")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "model-size-comparison.png")
    plt.savefig(out, dpi=150); plt.close()
    return out


def main() -> int:
    for path in (BM_PATH, MS_PATH):
        if not os.path.exists(path):
            print(f"[ERROR] Missing input: {path}", file=sys.stderr)
            return 1
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(BM_PATH) as f:
        bm = json.load(f)
    with open(MS_PATH) as f:
        ms = json.load(f)
    outputs = [
        plot_latency_by_stage(bm),
        plot_fps_vs_streams(ms),
        plot_accuracy_by_config(bm),
        plot_model_sizes(),
    ]
    for p in outputs:
        print(f"  wrote {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
