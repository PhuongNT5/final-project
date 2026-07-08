# Phase 04 — Streamlit Dashboard

**Status:** done (2026-04-19)
**Priority:** P2
**Estimated time:** 8h (actual ~1.5h)

---

## Delivered

- `deployment/dashboard/app.py` (193 lines, single file — KISS over component split).
- Sidebar selectors: precision (fp32/int8), mode (ensemble/direct4/cascade), view switch.
- Single-Video view: file upload → frame loop → sliding window emit every 8 frames → bar
  chart of 4-class probabilities + predicted label + latency.
- Multi-Stream view: up to 4 uploads, 2×2 grid, batched backbone call per frame across
  active streams, per-tile prediction caption.
- Sidebar metrics card: 56/64, 26.9 ms p50, 37 FPS.
- Boots via `.venv/bin/python3 -m streamlit run deployment/dashboard/app.py`, HTTP 200.
- Predict path verified end-to-end on dummy input for all 3 modes.

### Scope Deviation
Error gallery view deferred to Phase 05 (depends on artefact generation).
**Dependency:** Phase 03 (multi-stream-pipeline.py)

---

## Context Links

- Entry point: `deployment/dashboard/app.py`
- Components: `deployment/dashboard/components/`
- Pipeline: `deployment/scripts/multi-stream-pipeline.py` (Phase 03)
- Models: `deployment/export/*.onnx` (Phase 01)
- Error clips reference: `PROGRESS.md` §9 persistent-error clips

---

## Overview

A local Streamlit app for thesis defense demos. Three views: (a) single-video upload with
live class overlay and confidence bars, (b) 4-tile multi-stream grid using the Phase 03 pipeline,
(c) model selector between direct4, cascade, and ensemble modes. ORT sessions created once at
startup in `st.session_state`. No authentication, no cloud deployment — `streamlit run` only.
`app.py` stays ≤ 200 lines; UI logic extracted to three helper components.

---

## Key Insights

- Streamlit re-runs the entire script on each widget interaction — all ORT sessions MUST be
  cached in `@st.cache_resource` (Streamlit 1.18+) or `st.session_state`. Never create a
  session inside a callback or per-frame loop.
- Video streaming in Streamlit: use `st.image()` inside a `while` loop with `st.empty()`
  placeholder; update placeholder at ~10fps (sufficient for demo; 30fps would saturate the
  browser websocket).
- Multi-stream grid: use `st.columns(2)` × 2 rows → 4 placeholders. Update each column's
  placeholder independently. Each column owns one `StreamState` from the Phase 03 pipeline.
- The Phase 03 pipeline is asyncio-based; Streamlit runs in a sync context. Bridge via
  `asyncio.run()` in a background thread using `concurrent.futures.ThreadPoolExecutor`.
  Alternatively, simplify: for the dashboard, use a synchronous polling loop that calls the
  backbone + head directly — KISS over purity.
- `st.sidebar` for model selector, FPS readout, latency display.
- Optional error gallery: load 9 pre-computed clip results from a JSON file — no live inference
  needed, avoids embedding 9 video files in the demo.
- Keep component files < 200 lines each (dev rules).

---

## Requirements

### Functional
- View A: upload video → extract frames → preprocess → ONNX inference → display top-1 class
  label + 4-class confidence bar chart (st.bar_chart or st.progress) per frame block
- View B: multi-stream grid (2×2) using up to 4 uploaded video files, live FPS counter
- View C: sidebar model selector — direct4 / cascade / ensemble; switching reloads ORT sessions
- Latency counter (ms, rolling 10-frame average) + FPS readout in sidebar
- Optional View D: error gallery — 9 clips with ground-truth, predicted class, confidence

### Non-functional
- `streamlit run deployment/dashboard/app.py` — single command startup
- ORT sessions created once via `@st.cache_resource`
- `app.py` ≤ 200 lines; each component ≤ 200 lines
- No external auth, no cloud, no Docker required
- Tested locally on M4 macOS

---

## Related Code Files

**Create:**
- `deployment/dashboard/app.py` — main Streamlit entry point
- `deployment/dashboard/components/prediction-card.py` — renders class label + confidence bars
- `deployment/dashboard/components/stream-tile.py` — renders single stream frame + overlay
- `deployment/dashboard/components/model-selector.py` — sidebar model picker + session init

**Read (do not modify):**
- `deployment/scripts/multi-stream-pipeline.py` — borrow `stats_agg`, `preprocess` helpers
- `deployment/scripts/pipeline-utils.py` — shared utils (Phase 03)
- `deployment/export/*.onnx` — loaded at startup
- `PROGRESS.md` — error clip filenames for optional View D

---

## Implementation Steps

### Step 1 — Scaffold `app.py` (1h)

```python
# deployment/dashboard/app.py
import streamlit as st
import onnxruntime as ort
from components.model_selector import load_sessions, render_sidebar
from components.stream_tile import render_stream_tile
from components.prediction_card import render_prediction_card

st.set_page_config(page_title="Unsafe Behavior Detection", layout="wide")

# --- Session init (once per process) ---
@st.cache_resource
def get_sessions(model_mode: str):
    return load_sessions(model_mode)  # returns {backbone, head, cascade_heads}

model_mode = st.sidebar.selectbox("Model", ["direct4", "cascade", "ensemble"])
sessions = get_sessions(model_mode)

# --- View routing ---
view = st.sidebar.radio("View", ["Single Video", "Multi-Stream", "Error Gallery"])
if view == "Single Video":
    render_single_video(sessions)
elif view == "Multi-Stream":
    render_multistream(sessions)
else:
    render_error_gallery()
```

### Step 2 — Write `model-selector.py` (1h)

```python
# deployment/dashboard/components/model-selector.py
import onnxruntime as ort, os

ONNX_DIR = 'deployment/export'

def load_sessions(mode: str) -> dict:
    """Load ORT sessions for selected mode. Called once via @st.cache_resource."""
    backbone = ort.InferenceSession(f'{ONNX_DIR}/backbone_fp32.onnx',
                                     providers=['CPUExecutionProvider'])
    if mode == 'direct4':
        head = ort.InferenceSession(f'{ONNX_DIR}/direct4_s314_fp32.onnx', ...)
        return {'backbone': backbone, 'head': head, 'mode': 'direct4'}
    elif mode == 'cascade':
        # load scenario_s7 + pair{0..3}
        ...
    elif mode == 'ensemble':
        # load direct4 + cascade; average logits
        ...

def render_sidebar(sessions):
    st.sidebar.metric("Model", sessions['mode'])
    # latency + FPS placeholders updated by caller
```

### Step 3 — Write `prediction-card.py` (1h)

```python
# deployment/dashboard/components/prediction-card.py
import streamlit as st, numpy as np

CLASSES = ['Walkway', 'Intervention', 'Panel', 'Forklift']

def render_prediction_card(logits, latency_ms):
    """Render top-1 class label + confidence bar chart."""
    probs = softmax(logits)
    top1 = CLASSES[np.argmax(probs)]
    st.markdown(f"### Predicted: **{top1}**")
    st.bar_chart({c: float(p) for c, p in zip(CLASSES, probs)})
    st.caption(f"Latency: {latency_ms:.1f} ms")

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()
```

### Step 4 — Write `stream-tile.py` (1h)

`render_stream_tile(col, stream_id, frame, label, fps)` — copy frame, `cv2.putText` overlay
with stream ID + label + FPS, call `col.image(annotated, channels='BGR', use_container_width=True)`.

### Step 5 — Implement Single Video view (1.5h)

`st.file_uploader` → write bytes to `tempfile.mkstemp()` → `cv2.VideoCapture`. Main loop:
- Per frame: `run_backbone(frame, sessions['backbone'])` → push to `deque(maxlen=32)`
- Every 8 frames when buffer full: `stats_agg(buf)` → head session → `render_prediction_card(logits, latency)`
- Update `st.empty()` frame placeholder at 10fps (skip frames to avoid websocket overload)
- Delete temp file in `finally` block.

### Step 6 — Implement Multi-stream view (1.5h)

```python
def render_multistream(sessions):
    n = st.slider("Streams", 1, 4, 2)
    files = [st.file_uploader(f"Stream {i}", type=['mp4','avi']) for i in range(n)]
    if not all(files): st.info("Upload all streams"); return
    cols = st.columns(min(n, 2))
    # Synchronous polling loop — simpler than asyncio in Streamlit context (KISS)
    # Run each stream's frame one at a time in round-robin; update placeholders
```

### Step 7 — Optional error gallery (30 min)

Load `deployment/artifacts/error-gallery.json` (generated in Phase 05).
Each entry: `{clip, gt, pred, conf}`. Render with `st.write` — one row per clip.
Skip gracefully if JSON absent: `if not os.path.exists(path): st.info("No error gallery yet")`.

---

## Todo List

- [ ] Scaffold `app.py` with view routing and `@st.cache_resource` sessions
- [ ] Write `model-selector.py` — load sessions for direct4/cascade/ensemble
- [ ] Write `prediction-card.py` — class label + confidence bars
- [ ] Write `stream-tile.py` — frame display with overlay text
- [ ] Implement Single Video view — sliding window inference loop
- [ ] Implement Multi-stream view — N tiles, round-robin polling loop
- [ ] Implement Optional error gallery view (if `error-gallery.json` exists)
- [ ] Confirm `app.py` ≤ 200 lines; extract helpers if needed
- [ ] Smoke-test single video with a test clip; confirm prediction displayed
- [ ] Smoke-test multi-stream with N=2; confirm both tiles update

---

## Success Criteria

1. `streamlit run deployment/dashboard/app.py` launches without error on M4
2. Single video view: prediction + confidence bars appear within 2 seconds of upload
3. Multi-stream 2-tile view updates at ≥ 5fps per tile (sufficient for demo)
4. Model selector: switching between direct4/cascade reloads ORT sessions correctly
5. Latency counter visible in sidebar
6. All files ≤ 200 lines; components importable without circular deps

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Streamlit re-runs break ORT session state | High | High | Use `@st.cache_resource` — tested pattern; must not use `st.session_state` for sessions |
| Asyncio bridge with Streamlit sync context | Medium | Medium | Avoid asyncio in dashboard — use synchronous polling loop (KISS) |
| Video upload → cv2 temp file path on macOS | Low | Low | Write to `tempfile.mkstemp()`; delete after processing |
| Frame rate too low for demo (< 3fps) | Medium | Medium | Reduce display to every 4th frame; inference every 8th — threshold for visibility |
| Component import path issues (flat vs package) | Low | Low | Use `sys.path.insert(0, 'deployment/dashboard')` in app.py |

---

## Next Phase

Phase 05 (Thesis Artifacts) depends on benchmark numbers from Phase 02 and multi-stream numbers
from Phase 03. Dashboard is independent of Phase 05 — both can proceed once Phase 03 is done.
