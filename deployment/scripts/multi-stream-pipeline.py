"""
multi-stream-pipeline.py — Concurrent N-stream ONNX inference on M3 CPU.

Design:
- Single shared backbone ORT session; frames from N streams batched to (N,3,224,224).
- Per-stream sliding window (T=32) of (16,512) tokens; emit prediction every K=8 frames.
- Prediction = ensemble (direct4 + cascade) / 2, L1-normalised (matches run-inference).
- Frame-drop policy: if queue >= QUEUE_MAX, oldest drop counted per stream.
- Metrics: aggregate FPS, per-stream latency p50/p95/p99, drop counts, backbone batch util.

Usage:
    .venv/bin/python3 deployment/scripts/multi-stream-pipeline.py \
        --streams 4 --source v1.mp4 v2.mp4 v3.mp4 v4.mp4 --max-frames 300
"""
import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
from typing import Dict, List

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
EXPORT_DIR = os.path.join(ROOT, "deployment", "export")
OUT_PATH = os.path.join(ROOT, "deployment", "artifacts", "multi-stream-results.json")

def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "deployment", "scripts", fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pu = _load("pipeline_utils", "pipeline-utils.py")
# Shared StreamState/aggregate_fps/video_reader/softmax/stats_agg/percentiles — was
# near-verbatim duplicated with jetson-multi-stream-pipeline.py (code-review 2026-09-16 §6).
_spu = _load("stream_pipeline_utils", "stream-pipeline-utils.py")
StreamState, aggregate_fps, video_reader = _spu.StreamState, _spu.aggregate_fps, _spu.video_reader
softmax, stats_agg, percentiles = _spu.softmax, _spu.stats_agg, _spu.percentiles
QUEUE_MAX, T_WINDOW, EMIT_EVERY = _spu.QUEUE_MAX, _spu.T_WINDOW, _spu.EMIT_EVERY


async def inference_loop(queue: asyncio.Queue, backbone, heads: Dict[str, object],
                         states: Dict[int, StreamState], batch_util: List[float],
                         stop: asyncio.Event) -> None:
    bb_in = backbone.get_inputs()[0].name
    head_in = heads["direct4"].get_inputs()[0].name
    n_streams = len(states)
    # Continue draining until stop is set AND the queue is empty — prevents silent
    # frame loss at shutdown when readers have put() frames but not yet yielded.
    while not (stop.is_set() and queue.empty()):
        try:
            first = await asyncio.wait_for(queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        items = [first]
        # 5ms sync window for stragglers — trades tiny latency for larger batch
        bucket_deadline = time.perf_counter() + 0.005
        while len(items) < n_streams and time.perf_counter() < bucket_deadline:
            try:
                items.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)
        batch = np.stack([f for _, f in items]).astype(np.float32)
        batch_util.append(len(items) / n_streams)
        feats = backbone.run(None, {bb_in: batch})[0]  # (B,16,512)
        for i, (sid, _) in enumerate(items):
            s = states[sid]
            s.processed += 1
            s.buf.append(feats[i])
            if len(s.buf) == T_WINDOW and s.frame_count % EMIT_EVERY == 0:
                stats = stats_agg(np.stack(s.buf))[None]  # (1,32,1536)
                t0 = time.perf_counter()
                d4 = softmax(heads["direct4"].run(None, {head_in: stats})[0])
                sc = softmax(heads["scenario"].run(None, {head_in: stats})[0])
                pair_cols = [
                    softmax(heads[k].run(None, {head_in: stats})[0])[:, 1:2]
                    for k in pu.PAIR_KEYS
                ]
                cas = sc * np.concatenate(pair_cols, axis=1)
                cas = cas / (cas.sum(1, keepdims=True) + 1e-8)
                ens = (d4 + cas) / 2.0
                ens = ens / (ens.sum(1, keepdims=True) + 1e-8)
                s.head_latencies.append((time.perf_counter() - t0) * 1000)
                s.predictions.append(int(ens.argmax(1)[0]))


def _load_sessions(precision: str):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    backbone = ort.InferenceSession(
        os.path.join(EXPORT_DIR, f"backbone-{precision}.onnx"), so, providers=providers)
    heads = {
        k: ort.InferenceSession(
            os.path.join(EXPORT_DIR, f"{fname}-{precision}.onnx"), so, providers=providers)
        for k, fname in pu.HEAD_FILES.items()
    }
    return backbone, heads


async def run_config(n: int, sources: List[str], max_frames: int, fps: float,
                     precision: str) -> Dict:
    backbone, heads = _load_sessions(precision)
    states = {i: StreamState(i) for i in range(n)}
    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX * n)
    batch_util: List[float] = []
    stop = asyncio.Event()
    t0 = time.perf_counter()
    readers = [asyncio.create_task(video_reader(sources[i % len(sources)], queue,
                                                 states[i], max_frames, fps, stop, pu.preprocess))
               for i in range(n)]
    inf = asyncio.create_task(inference_loop(queue, backbone, heads, states, batch_util, stop))
    await asyncio.gather(*readers)
    stop.set()
    # inference_loop exits by itself once `stop` is set AND queue is empty,
    # so the final items put by readers are processed before timing stops.
    await inf
    elapsed = time.perf_counter() - t0
    agg = aggregate_fps(states, elapsed)
    return {
        "n_streams": n, "precision": precision, "elapsed_sec": agg["elapsed_sec"],
        "aggregate_fps": agg["aggregate_fps"], "read_fps": agg["read_fps"], "drop_rate": agg["drop_rate"],
        "backbone_batch_utilisation": round(float(np.mean(batch_util)) if batch_util else 0.0, 3),
        "streams": [{
            "id": s.sid, "frames": s.frame_count, "processed": s.processed,
            "dropped": s.dropped,
            "head_latency_ms": percentiles(s.head_latencies),
            "n_predictions": len(s.predictions),
        } for s in states.values()],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--source", nargs="+", required=True)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "int8"])
    args = ap.parse_args()

    results = {"configs": []}
    for n in args.streams:
        print(f"\n=== N={n} streams, precision={args.precision} ===", flush=True)
        r = asyncio.run(run_config(n, args.source, args.max_frames, args.fps, args.precision))
        print(f"  aggFPS(processed)={r['aggregate_fps']}  readFPS={r['read_fps']}"
              f"  batchUtil={r['backbone_batch_utilisation']}"
              f"  drops={sum(s['dropped'] for s in r['streams'])}  drop_rate={r['drop_rate']}")
        results["configs"].append(r)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {OUT_PATH}")


if __name__ == "__main__":
    main()
