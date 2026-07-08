"""
app.py — Streamlit dashboard for YOLO-Transformer unsafe-behavior detection.

Views:
- Single Video : upload one file → sliding-window inference → class + confidences
- Multi-Stream : up to 4 uploads → 2×2 tile grid, synchronous polling
Sidebar        : model mode (direct4 / cascade / ensemble), precision (fp32 / int8)

Usage:
    .venv/bin/python3 -m streamlit run deployment/dashboard/app.py
"""
import collections
import importlib.util
import os
import sys
import tempfile
import time
from typing import List

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
EXPORT_DIR = os.path.join(ROOT, "deployment", "export")

_spec = importlib.util.spec_from_file_location(
    "pipeline_utils", os.path.join(ROOT, "deployment", "scripts", "pipeline-utils.py"))
pu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pu)

CLASSES = ["Walkway Violation", "Unauthorized Intervention",
           "Opened Panel Cover", "Overload w/ Forklift"]
T_WINDOW = 32
EMIT_EVERY = 8


@st.cache_resource(show_spinner="Loading ONNX models…")
def load_sessions(precision: str):
    so = ort.SessionOptions(); so.log_severity_level = 3
    ep = ["CPUExecutionProvider"]
    return {
        "backbone": ort.InferenceSession(
            os.path.join(EXPORT_DIR, f"backbone-{precision}.onnx"), so, providers=ep),
        **{k: ort.InferenceSession(os.path.join(EXPORT_DIR, f"{fn}-{precision}.onnx"), so, providers=ep)
           for k, fn in pu.HEAD_FILES.items()},
    }


def predict(buf_32x16x512: np.ndarray, heads, mode: str) -> np.ndarray:
    """Returns (4,) probabilities according to mode."""
    stats = pu.stats_agg(buf_32x16x512)[None]  # (1,32,1536)
    head_in = heads["direct4"].get_inputs()[0].name
    if mode == "direct4":
        return pu.softmax(heads["direct4"].run(None, {head_in: stats})[0])[0]
    sc = pu.softmax(heads["scenario"].run(None, {head_in: stats})[0])
    pair_cols = [pu.softmax(heads[k].run(None, {head_in: stats})[0])[:, 1:2]
                 for k in pu.PAIR_KEYS]
    cas = sc * np.concatenate(pair_cols, axis=1)
    cas = cas / (cas.sum(1, keepdims=True) + 1e-8)
    if mode == "cascade":
        return cas[0]
    d4 = pu.softmax(heads["direct4"].run(None, {head_in: stats})[0])
    ens = (d4 + cas) / 2.0
    ens = ens / (ens.sum(1, keepdims=True) + 1e-8)
    return ens[0]


def render_prediction(probs: np.ndarray, latency_ms: float, slot) -> None:
    top = int(np.argmax(probs))
    slot.markdown(f"### Prediction: **{CLASSES[top]}**  ·  {probs[top] * 100:.1f}%")
    slot.bar_chart({CLASSES[i]: float(probs[i]) for i in range(4)})
    slot.caption(f"Head latency: {latency_ms:.1f} ms")


def _tmp_video(uploaded) -> str:
    ext = os.path.splitext(uploaded.name)[1] or ".mp4"
    fd, path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.read())
    return path


def single_video(sessions: dict, mode: str) -> None:
    up = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    if not up:
        return
    path = _tmp_video(up)
    try:
        cap = cv2.VideoCapture(path)
        bb_in = sessions["backbone"].get_inputs()[0].name
        buf: collections.deque = collections.deque(maxlen=T_WINDOW)
        frame_slot = st.empty(); pred_slot = st.empty(); status_slot = st.empty()
        n = 0; emitted = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tokens = sessions["backbone"].run(None, {bb_in: pu.preprocess(frame)[None]})[0][0]
            buf.append(tokens)
            n += 1
            if n % 4 == 0:  # display every 4th frame (~7fps browser update)
                frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB",
                                 use_container_width=True)
            if len(buf) == T_WINDOW and n % EMIT_EVERY == 0:
                t0 = time.perf_counter()
                probs = predict(np.stack(buf), sessions, mode)
                render_prediction(probs, (time.perf_counter() - t0) * 1000, pred_slot)
                emitted += 1
                status_slot.caption(f"Frame {n}  ·  {emitted} predictions emitted")
        cap.release()
    finally:
        os.unlink(path)


def multi_stream(sessions: dict, mode: str) -> None:
    n = st.slider("Number of streams", 1, 4, 2)
    uploads = [st.file_uploader(f"Stream {i}", type=["mp4", "avi", "mov"],
                                 key=f"ms_{i}") for i in range(n)]
    if not all(uploads):
        st.info("Upload all streams to start.")
        return
    paths: List[str] = []
    caps: List[cv2.VideoCapture] = []
    try:
        # Create temp files + capture handles inside try so partial failures still clean up
        for u in uploads:
            paths.append(_tmp_video(u))
            caps.append(cv2.VideoCapture(paths[-1]))
        bb_in = sessions["backbone"].get_inputs()[0].name
        buffers = [collections.deque(maxlen=T_WINDOW) for _ in range(n)]
        frame_counts = [0] * n
        cols = st.columns(2) if n > 1 else st.columns(1)
        frame_slots = [cols[i % len(cols)].empty() for i in range(n)]
        pred_slots = [cols[i % len(cols)].empty() for i in range(n)]
        active = [True] * n
        while any(active):
            batch_frames, batch_ids = [], []
            for i in range(n):
                if not active[i]:
                    continue
                ok, frame = caps[i].read()
                if not ok:
                    active[i] = False
                    continue
                batch_frames.append(pu.preprocess(frame))
                batch_ids.append(i)
                frame_counts[i] += 1
                if frame_counts[i] % 4 == 0:
                    frame_slots[i].image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                         channels="RGB", use_container_width=True)
            if not batch_frames:
                break
            batch = np.stack(batch_frames).astype(np.float32)
            tokens_batch = sessions["backbone"].run(None, {bb_in: batch})[0]
            for j, i in enumerate(batch_ids):
                buffers[i].append(tokens_batch[j])
                if len(buffers[i]) == T_WINDOW and frame_counts[i] % EMIT_EVERY == 0:
                    t0 = time.perf_counter()
                    probs = predict(np.stack(buffers[i]), sessions, mode)
                    top = int(np.argmax(probs))
                    pred_slots[i].markdown(
                        f"**Stream {i}**: {CLASSES[top]} ({probs[top] * 100:.0f}%) · "
                        f"{(time.perf_counter() - t0) * 1000:.1f} ms")
    finally:
        for c in caps:
            c.release()
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass


def main() -> None:
    st.set_page_config(page_title="Unsafe Behavior Detection — M4 Edge Demo", layout="wide")
    st.title("Real-Time Unsafe Behavior Detection")
    st.caption("YOLOv11s backbone + Transformer heads · ONNX Runtime on Apple M4 CPU")

    precision = st.sidebar.selectbox("Precision", ["fp32", "int8"], index=0)
    mode = st.sidebar.selectbox("Model mode", ["ensemble", "direct4", "cascade"], index=0)
    view = st.sidebar.radio("View", ["Single Video", "Multi-Stream"])
    st.sidebar.markdown("---")
    st.sidebar.metric("Baseline accuracy", "56/64 (87.5%)")
    st.sidebar.metric("e2e p50 @ CPU-FP32", "26.9 ms")
    st.sidebar.metric("Max FPS (N=1)", "37")

    sessions = load_sessions(precision)
    if view == "Single Video":
        single_video(sessions, mode)
    else:
        multi_stream(sessions, mode)


if __name__ == "__main__":
    main()
