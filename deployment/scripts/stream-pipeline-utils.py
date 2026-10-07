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
import os
from typing import Callable, Dict, List

import cv2
import numpy as np

QUEUE_MAX = 4
T_WINDOW = 32
EMIT_EVERY = 8

_HERE = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'deployment', 'artifacts')


def _sibling(mod_name, fname):
    """Lazy: only the artifact helpers below need these, and nothing else here should pay for
    subprocess-capable imports just to decode frames."""
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location(mod_name, os.path.join(_HERE, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _measure_utils():
    return _sibling('jetson_measure_utils', 'jetson-measure-utils.py')


def _clock_utils():
    return _sibling('jetson_clock_state_utils', 'jetson-clock-state-utils.py')


class StreamState:
    def __init__(self, sid: int, t_window: int = T_WINDOW):
        """t_window: sliding-window length. P1 aggregates T=32 backbone feature maps; the P3
        decision path keeps T=16 per-frame detection results, so the window is a parameter
        rather than the module constant it used to be."""
        self.sid = sid
        self.buf = collections.deque(maxlen=t_window)
        self.frame_count = 0     # every frame read (incl. dropped)
        self.processed = 0       # frames that actually reached backbone inference
        self.dropped = 0
        self.head_latencies: List[float] = []
        self.frame_latencies: List[float] = []   # whole per-frame pipeline, not just the head
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


def open_sweep_artifact(name, payload, out=None) -> tuple:
    """(payload, path) for a concurrency sweep, provenance attached before the first config runs.

    A sweep writes several numbers from one device state, so the state is captured here and again
    by close_sweep_artifact() — a point-in-time check cannot see a clock pin that lands mid-sweep,
    which is how the 2026-10-01 artifacts were contaminated. Unlabelled latency is what withdrew
    the 09-29 device set; neither streaming harness may write a number without this.
    """
    from datetime import datetime
    U = _measure_utils()
    stamp = datetime.now()
    payload = {'git_commit': U.git_commit(), 'timestamp': stamp.isoformat(),
               'device_state': U.device_state(False), **payload}
    _clock_utils().warn_if_unreliable(payload['device_state'])
    path = out or os.path.join(ART_DIR, f'{name}-{stamp.strftime("%Y%m%dT%H%M%S")}.json')
    return payload, path


def close_sweep_artifact(payload) -> bool:
    """Re-read the device state and record whether it held. Returns False if the clock pin or the
    contention changed, in which case the configs are not comparable with each other."""
    after = _measure_utils().device_state(False)
    before = payload.get('device_state', {})
    stable = (before.get('clocks_pinned') == after.get('clocks_pinned')
              and before.get('quiet') == after.get('quiet'))
    payload['device_state_after'] = after
    payload['clock_state_stable'] = stable
    return stable


def flush_json(path, payload):
    """Write after every config, not once at the end: a sweep that aborts on its last N must not
    throw away the earlier ones — device time is the scarce resource."""
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)


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
