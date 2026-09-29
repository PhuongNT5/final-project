"""stream-pipeline-utils.py — Shared per-stream state, throughput accounting, and video-frame
producer coroutine for the multi-stream pipelines: multi-stream-pipeline.py (Mac, ONNX Runtime)
and jetson-multi-stream-pipeline.py (Jetson, TensorRT).

Both files carried a near-verbatim-duplicated copy of StreamState / aggregate_fps / video_reader
(flagged in plans/2026-09-06-thesis-2week-final/reports/code-reviewer-2026-09-16-jetson-session-
prep-review.md §6) — a fix landing in only one copy already happened once (the aggregate_fps
drop-counting bug, fixed identically-but-separately in both files 2026-09-16). This module is now
the single source of truth for those three; each caller still owns its own `preprocess()` (the
one genuine per-platform difference — PIL-bilinear+pipeline-utils on Mac vs cv2-only on Jetson),
passed into video_reader() as `preprocess_fn` rather than duplicated.

Loaded via importlib (hyphenated filename), same pattern already used for trt-runner-utils.py —
this module has no tensorrt/pycuda/onnxruntime dependency, so it imports cleanly on Mac or
Jetson alike.
"""
import asyncio
import collections
from typing import Callable, Dict, List

import cv2
import numpy as np

QUEUE_MAX = 4
T_WINDOW = 32
EMIT_EVERY = 8


class StreamState:
    def __init__(self, sid: int):
        self.sid = sid
        self.buf = collections.deque(maxlen=T_WINDOW)
        self.frame_count = 0     # every frame read (incl. dropped)
        self.processed = 0       # frames that actually reached backbone inference
        self.dropped = 0
        self.head_latencies: List[float] = []
        self.predictions: List[int] = []


async def video_reader(source: str, queue: "asyncio.Queue", state: StreamState,
                       max_frames: int, fps: float, stop: "asyncio.Event",
                       preprocess_fn: Callable) -> None:
    """Reads frames from `source` at `fps`, applying `preprocess_fn` (caller-supplied) before
    enqueueing. Drops the frame (counted in state.dropped, not enqueued) if the shared queue is
    already at QUEUE_MAX — never blocks the reader loop on a full queue."""
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
                await queue.put((state.sid, preprocess_fn(frame)))
            state.frame_count += 1
            await asyncio.sleep(period)
    finally:
        cap.release()


def softmax(x, axis: int = -1):
    """Same formula in both callers (was duplicated as multi-stream-pipeline.py's local
    softmax and jetson-multi-stream-pipeline.py's local softmax — identical modulo the
    always-default `axis` param)."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def stats_agg(buf):
    """(T, S, C) -> (T, 3C) via concat(mean, std, max) over the spatial dim S. Identical in
    both callers before this extraction."""
    return np.concatenate([buf.mean(1), buf.std(1), buf.max(1)], axis=-1)


def percentiles(xs) -> Dict[str, float]:
    """Latency percentile summary for per-stream head_latency_ms reporting. Identical in both
    callers before this extraction (mac left values unrounded; standardised on rounded here
    since neither file's tests assert exact float precision)."""
    if not xs:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    a = np.asarray(xs, dtype=np.float64)
    return {"p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2)}


def aggregate_fps(states: Dict[int, StreamState], elapsed: float) -> Dict:
    """Throughput/drop metrics, from PROCESSED frames only (not frame_count, which includes
    drops) — see tests/test_stream_fps_accounting.py for the regression this fixes."""
    total_frames = sum(s.frame_count for s in states.values())
    total_processed = sum(s.processed for s in states.values())
    total_dropped = sum(s.dropped for s in states.values())
    return {
        "elapsed_sec": round(elapsed, 3),
        "total_frames": total_frames, "total_processed": total_processed,
        "total_dropped": total_dropped,
        "aggregate_fps": round(total_processed / elapsed, 2),
        "read_fps": round(total_frames / elapsed, 2),
        "drop_rate": round(total_dropped / total_frames, 4) if total_frames else 0.0,
    }
