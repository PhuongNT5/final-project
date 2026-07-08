"""
jetson-multi-stream-pipeline.py — Concurrent N-stream TRT inference on Jetson Orin.

Mirrors the M4 multi-stream-pipeline.py but swaps ORT sessions for TensorRT
engines (deployment/export/jetson/*-{mode}.engine). Engines are batch=1 only,
so per-stream backbone calls run sequentially through the shared TRT runner;
GPU stream scheduling provides the parallelism instead of CPU-side batching.

Design:
- One TRTRunner per model (backbone + 6 heads), shared across streams.
- Per-stream sliding window (T=32) of (16,512) tokens; emit prediction every K=8.
- Prediction = ensemble (direct4 + cascade) / 2, L1-normalised.
- Frame-drop policy: if queue >= QUEUE_MAX, oldest drop counted per stream.

Output: deployment/artifacts/multi-stream-jetson-{mode}.json

Usage on Jetson:
    python3 deployment/scripts/jetson-multi-stream-pipeline.py \
        --streams 1 2 3 4 --source data/test/0_*/0_te1.mp4 \
        --max-frames 240 --mode fp16
"""
import argparse
import asyncio
import collections
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, TextIO

CLASS_LABELS = ["walkway_violation", "unauthorized_intervention",
                "opened_panel_cover", "overload_with_forklift"]

# pycuda.autoinit must precede tensorrt for correct CUDA context handling.
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE_DIR = os.path.join(ROOT, "deployment", "export", "jetson")
OUT_DIR = os.path.join(ROOT, "deployment", "artifacts")

QUEUE_MAX = 4
T_WINDOW = 32
EMIT_EVERY = 8

# ImageNet normalisation
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return (arr - _MEAN) / _STD


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def stats_agg(buf: np.ndarray) -> np.ndarray:
    """(T, S, C) -> (T, 1536) via concat(mean, std, max along S)."""
    return np.concatenate([buf.mean(1), buf.std(1), buf.max(1)], axis=-1)


def percentiles(xs):
    if not xs:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    a = np.asarray(xs, dtype=np.float64)
    return {p: round(float(np.percentile(a, p)), 2) for p in (50, 95, 99)}


class TRTRunner:
    """Single-engine inference wrapper. Pre-allocates host+device buffers and
    reuses them across calls. Not thread-safe — call from a single coroutine."""

    def __init__(self, trt, engine_path: str):
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = np.empty(shape, dtype=dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(dev))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.host_in, self.dev_in = host, dev
                self.in_shape = shape
            else:
                self.host_out, self.dev_out = host, dev

    def run(self, x: np.ndarray) -> np.ndarray:
        np.copyto(self.host_in, x.reshape(self.in_shape))
        cuda.memcpy_htod_async(self.dev_in, self.host_in, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_out, self.dev_out, self.stream)
        self.stream.synchronize()
        return self.host_out.copy()


class StreamState:
    def __init__(self, sid: int):
        self.sid = sid
        self.buf = collections.deque(maxlen=T_WINDOW)
        self.frame_count = 0
        self.dropped = 0
        self.head_latencies: List[float] = []
        self.predictions: List[int] = []


async def video_reader(source: str, queue: asyncio.Queue, state: StreamState,
                       max_frames: int, fps: float, stop: asyncio.Event) -> None:
    loop = asyncio.get_event_loop()
    cap = cv2.VideoCapture(source)
    period = 1.0 / max(fps, 1.0)
    try:
        while not stop.is_set() and state.frame_count < max_frames:
            ok, frame = await loop.run_in_executor(None, cap.read)
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if queue.qsize() >= QUEUE_MAX:
                state.dropped += 1
            else:
                await queue.put((state.sid, preprocess(frame)))
            state.frame_count += 1
            await asyncio.sleep(period)
    finally:
        cap.release()


async def inference_loop(queue: asyncio.Queue, runners: Dict[str, TRTRunner],
                         states: Dict[int, StreamState], stop: asyncio.Event,
                         emit_fh: Optional[TextIO] = None) -> None:
    while not (stop.is_set() and queue.empty()):
        try:
            sid, frame = await asyncio.wait_for(queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        # Sequential backbone — no batching since engines are batch=1.
        feats = runners["backbone"].run(frame[None].astype(np.float32))  # (1,16,512)
        s = states[sid]
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
                    "label": CLASS_LABELS[cls] if cls < len(CLASS_LABELS) else f"class_{cls}",
                    "probs": [round(p, 4) for p in probs],
                    "confidence": round(float(probs[cls]), 4),
                    "head_ms": round(head_ms, 2),
                }
                emit_fh.write(json.dumps(event) + "\n")
                emit_fh.flush()


def _load_runners(mode: str) -> Dict[str, TRTRunner]:
    import tensorrt as trt
    names = ["backbone", "direct4-s314", "scenario-s7",
             "pair0-s7", "pair1-s7", "pair2-s7", "pair3-s7"]
    runners = {}
    for n in names:
        path = os.path.join(ENGINE_DIR, f"{n}-{mode}.engine")
        if not os.path.exists(path):
            sys.exit(f"engine missing: {path}; build with --mode {mode}")
        runners[n] = TRTRunner(trt, path)
    return runners


async def run_config(n: int, sources: List[str], max_frames: int, fps: float,
                     mode: str, emit_jsonl: Optional[str] = None) -> Dict:
    runners = _load_runners(mode)
    states = {i: StreamState(i) for i in range(n)}
    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX * n)
    stop = asyncio.Event()
    emit_fh = open(emit_jsonl, "a", buffering=1) if emit_jsonl else None
    try:
        t0 = time.perf_counter()
        readers = [asyncio.create_task(video_reader(sources[i % len(sources)], queue,
                                                    states[i], max_frames, fps, stop))
                   for i in range(n)]
        inf = asyncio.create_task(inference_loop(queue, runners, states, stop, emit_fh))
        await asyncio.gather(*readers)
        stop.set()
        await inf
        elapsed = time.perf_counter() - t0
    finally:
        if emit_fh is not None:
            emit_fh.close()
    total_frames = sum(s.frame_count for s in states.values())
    return {
        "n_streams": n, "mode": mode, "elapsed_sec": round(elapsed, 3),
        "aggregate_fps": round(total_frames / elapsed, 2),
        "streams": [{
            "id": s.sid, "frames": s.frame_count, "dropped": s.dropped,
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
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.emit_jsonl:
        # Truncate at start of a fresh run so the dashboard sees a clean stream.
        open(args.emit_jsonl, "w").close()
        print(f"[INFO] emitting prediction events to {args.emit_jsonl}")

    results = {"configs": []}
    for n in args.streams:
        print(f"\n=== N={n} streams, mode={args.mode} ===", flush=True)
        r = asyncio.run(run_config(n, args.source, args.max_frames, args.fps,
                                   args.mode, args.emit_jsonl))
        drops = sum(s["dropped"] for s in r["streams"])
        print(f"  aggFPS={r['aggregate_fps']}  drops={drops}  "
              f"elapsed={r['elapsed_sec']}s")
        results["configs"].append(r)

    out = os.path.join(OUT_DIR, f"multi-stream-jetson-{args.mode}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
