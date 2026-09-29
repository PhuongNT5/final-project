"""jetson-multi-stream-pipeline.py — Concurrent N-stream TRT inference on Jetson Orin.
Mirrors multi-stream-pipeline.py but swaps ORT sessions for TensorRT engines
(deployment/export/jetson/*-{mode}.engine); engines are batch=1, so per-stream backbone
calls run sequentially through the shared TRT runner and GPU stream scheduling provides the
parallelism instead of CPU-side batching. One TRTRunner per model (backbone + 6 heads),
shared across streams; T=32 sliding window, emit every K=8; frame-drop if queue>=QUEUE_MAX.
Output: deployment/artifacts/multi-stream-jetson-{mode}.json

Usage on Jetson: python3 deployment/scripts/jetson-multi-stream-pipeline.py --streams 1 2 3 4
    --source data/test/0_*/0_te1.mp4 --max-frames 240 --mode fp16
"""
import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, TextIO

# pycuda.autoinit must precede tensorrt for correct CUDA context handling.
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE_DIR = os.path.join(ROOT, "deployment", "export", "jetson")
OUT_DIR = os.path.join(ROOT, "deployment", "artifacts")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "deployment", "scripts", fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Shared TRTRunner (binds engine I/O tensors by name) + StreamState/aggregate_fps/video_reader/
# softmax/stats_agg/percentiles (see their docstrings); preprocess() stays local per-platform.
_tru = _load("trt_runner_utils", "trt-runner-utils.py")
_spu = _load("stream_pipeline_utils", "stream-pipeline-utils.py")
StreamState, aggregate_fps, video_reader = _spu.StreamState, _spu.aggregate_fps, _spu.video_reader
softmax, stats_agg, percentiles = _spu.softmax, _spu.stats_agg, _spu.percentiles
QUEUE_MAX, T_WINDOW, EMIT_EVERY = _spu.QUEUE_MAX, _spu.T_WINDOW, _spu.EMIT_EVERY

# ImageNet normalisation
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return (arr - _MEAN) / _STD


def _make_trt_runner(trt, engine_path: str):
    """Binds module-level `cuda` (from pycuda.autoinit) to the shared TRTRunner."""
    return _tru.TRTRunner(trt, cuda, engine_path)


async def inference_loop(queue: asyncio.Queue, runners: Dict[str, "_tru.TRTRunner"],
                         states: Dict[int, StreamState], stop: asyncio.Event,
                         emit_fh: Optional[TextIO] = None,
                         class_labels: Optional[List[str]] = None) -> None:
    while not (stop.is_set() and queue.empty()):
        try:
            sid, frame = await asyncio.wait_for(queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        # Sequential backbone — no batching since engines are batch=1.
        feats = runners["backbone"].run(frame[None].astype(np.float32))  # (1,16,512)
        s = states[sid]
        s.processed += 1
        s.buf.append(feats[0])
        if len(s.buf) == T_WINDOW and s.frame_count % EMIT_EVERY == 0:
            stats = stats_agg(np.stack(s.buf))[None].astype(np.float32)  # (1,32,1536)
            t0 = time.perf_counter()
            d4 = softmax(runners["direct4-s314"].run(stats))
            sc = softmax(runners["scenario-s7"].run(stats))
            pair = np.concatenate(
                [softmax(runners[f"pair{i}-s7"].run(stats))[:, 1:2] for i in range(4)],
                axis=1)
            cas = sc * pair
            cas = cas / (cas.sum(1, keepdims=True) + 1e-8)
            ens = (d4 + cas) / 2.0
            ens = ens / (ens.sum(1, keepdims=True) + 1e-8)
            head_ms = (time.perf_counter() - t0) * 1000
            cls = int(ens.argmax(1)[0])
            s.head_latencies.append(head_ms)
            s.predictions.append(cls)
            if emit_fh is not None:
                probs = ens[0].tolist()
                event = {
                    "ts": time.time(),
                    "iso": datetime.now().isoformat(timespec="milliseconds"),
                    "stream_id": sid,
                    "frame_n": s.frame_count,
                    "class": cls,
                    "label": class_labels[cls] if class_labels and cls < len(class_labels) else f"class_{cls}",
                    "probs": [round(p, 4) for p in probs],
                    "confidence": round(float(probs[cls]), 4),
                    "head_ms": round(head_ms, 2),
                }
                emit_fh.write(json.dumps(event) + "\n")
                emit_fh.flush()


def _load_runners(mode: str) -> Dict[str, "_tru.TRTRunner"]:
    import tensorrt as trt
    names = ["backbone", "direct4-s314", "scenario-s7",
             "pair0-s7", "pair1-s7", "pair2-s7", "pair3-s7"]
    runners = {}
    for n in names:
        path = os.path.join(ENGINE_DIR, f"{n}-{mode}.engine")
        if not os.path.exists(path):
            sys.exit(f"engine missing: {path}; build with --mode {mode}")
        runners[n] = _make_trt_runner(trt, path)
    return runners


async def run_config(n: int, sources: List[str], max_frames: int, fps: float,
                     mode: str, emit_jsonl: Optional[str] = None,
                     class_labels: Optional[List[str]] = None) -> Dict:
    runners = _load_runners(mode)
    states = {i: StreamState(i) for i in range(n)}
    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX * n)
    stop = asyncio.Event()
    emit_fh = open(emit_jsonl, "a", buffering=1) if emit_jsonl else None
    try:
        t0 = time.perf_counter()
        readers = [asyncio.create_task(video_reader(sources[i % len(sources)], queue,
                                                    states[i], max_frames, fps, stop, preprocess))
                   for i in range(n)]
        inf = asyncio.create_task(inference_loop(queue, runners, states, stop, emit_fh, class_labels))
        await asyncio.gather(*readers)
        stop.set()
        await inf
        elapsed = time.perf_counter() - t0
    finally:
        if emit_fh is not None:
            emit_fh.close()
    agg = aggregate_fps(states, elapsed)
    return {
        "n_streams": n, "mode": mode, "elapsed_sec": agg["elapsed_sec"],
        "aggregate_fps": agg["aggregate_fps"], "read_fps": agg["read_fps"], "drop_rate": agg["drop_rate"],
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
    ap.add_argument("--mode", default="fp16", choices=["fp32", "fp16", "int8"])
    ap.add_argument("--emit-jsonl", default=None,
                    help="Append JSON-Lines prediction events to this file (live stream "
                         "for downstream dashboards). Each emission = one line.")
    ap.add_argument("--class-labels", default=None,
                    help="Comma-separated JSONL `label` values in class-index order (no "
                         "default -> \"class_N\"; not a hardcoded list since the head is "
                         "moving 4-class->binary+aux). E.g. --class-labels Safe,Unsafe")
    args = ap.parse_args()
    class_labels = [s.strip() for s in args.class_labels.split(",")] if args.class_labels else None

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.emit_jsonl:
        # Truncate at start of a fresh run so the dashboard sees a clean stream.
        open(args.emit_jsonl, "w").close()
        print(f"[INFO] emitting prediction events to {args.emit_jsonl}")

    results = {"configs": []}
    for n in args.streams:
        print(f"\n=== N={n} streams, mode={args.mode} ===", flush=True)
        r = asyncio.run(run_config(n, args.source, args.max_frames, args.fps,
                                   args.mode, args.emit_jsonl, class_labels))
        drops = sum(s["dropped"] for s in r["streams"])
        print(f"  aggFPS(processed)={r['aggregate_fps']}  readFPS={r['read_fps']}  "
              f"drops={drops}  drop_rate={r['drop_rate']}  elapsed={r['elapsed_sec']}s")
        results["configs"].append(r)

    out = os.path.join(OUT_DIR, f"multi-stream-jetson-{args.mode}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
