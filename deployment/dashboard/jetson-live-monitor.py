"""
jetson-live-monitor.py — Streamlit dashboard that consumes the Jetson edge node's
live prediction stream, applies rolling-window aggregation, and raises ALERT
badges when unsafe behavior is detected with high confidence.

Architecture (Camera → Edge → Server → Dashboard):
    [Video files (camera proxy)]
       └→ Jetson Orin Nano Super: jetson-multi-stream-pipeline.py --emit-jsonl <path>
            └→ writes one JSON event per inference emission to ~/jetson-stream.jsonl
                 └→ this M4 dashboard pulls the file via rsync over SSH every poll-interval
                      └→ rolling 3-second aggregation → alert/warning/monitoring tiers

Privacy property (by construction):
- Raw video frames NEVER leave the Jetson; only prediction metadata
  (class id, confidences, timestamp, head latency) crosses the network.
- Transport is SSH key-auth + AES (rsync default), so transit is encrypted.

Decision policy (real-time alerting):
- Aggregate predictions over the most recent ROLLING_WINDOW_SEC seconds per stream
  (averaging probability vectors, then argmax) — robust to single-frame jitter.
- ALERT badge when aggregated confidence ≥ ALERT_THRESHOLD (default 0.75).
- WARNING badge when aggregated confidence in [WARN_THRESHOLD, ALERT_THRESHOLD).
- Otherwise MONITORING (low confidence — uncertain or likely safe).

Usage (on M4):
    .venv/bin/python3 -m streamlit run deployment/dashboard/jetson-live-monitor.py
"""
import collections
import json
import os
import subprocess
import time
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ----- Configuration ----------------------------------------------------------

DEFAULT_JETSON_HOST = "jetson"
DEFAULT_REMOTE_PATH = "~/jetson-stream.jsonl"
LOCAL_CACHE = "/tmp/jetson-stream.jsonl"
POLL_INTERVAL_SEC = 1.0
MAX_EVENTS_DISPLAYED = 2000

# Decision policy thresholds
ROLLING_WINDOW_SEC = 3.0      # aggregate predictions over last 3 sec per stream
ALERT_THRESHOLD = 0.75        # aggregated confidence ≥ this → ALERT
WARN_THRESHOLD = 0.55         # aggregated confidence ≥ this → WARNING

CLASSES = ["Walkway Violation", "Unauthorized Intervention",
           "Opened Panel Cover", "Overload w/ Forklift"]
CLASS_COLORS = ["#e74c3c", "#e67e22", "#f39c12", "#3498db"]

# Alert log state — persisted across reruns
if "alert_log" not in st.session_state:
    st.session_state.alert_log = collections.deque(maxlen=50)
if "last_alert_per_stream" not in st.session_state:
    st.session_state.last_alert_per_stream = {}


# ----- Data plumbing ----------------------------------------------------------

def pull_stream(jetson_host: str, remote_path: str) -> bool:
    """rsync the Jetson JSONL to local cache. Returns True on success."""
    try:
        r = subprocess.run(
            ["rsync", "-az", "--timeout=5",
             f"{jetson_host}:{remote_path}", LOCAL_CACHE],
            capture_output=True, timeout=10)
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def read_events(max_n: int = MAX_EVENTS_DISPLAYED) -> List[Dict]:
    if not os.path.exists(LOCAL_CACHE):
        return []
    try:
        with open(LOCAL_CACHE, "r") as f:
            lines = f.readlines()[-max_n:]
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ----- Aggregation + decision policy -----------------------------------------

def aggregate_stream(events: List[Dict], sid: int,
                     window_sec: float = ROLLING_WINDOW_SEC) -> Optional[Dict]:
    """Return aggregated prediction for stream `sid` over the last window_sec.
    Averages probability vectors, then argmax. Returns None if no events."""
    now = time.time()
    window = [e for e in events
              if e.get("stream_id") == sid and now - e["ts"] <= window_sec]
    if not window:
        return None
    probs = np.mean([e["probs"] for e in window], axis=0)
    probs = probs / probs.sum()  # renormalise after averaging
    cls = int(probs.argmax())
    return {
        "stream_id": sid,
        "n_emissions": len(window),
        "window_sec": window_sec,
        "class": cls,
        "label": CLASSES[cls],
        "confidence": float(probs[cls]),
        "probs": probs.tolist(),
        "latest_ts": max(e["ts"] for e in window),
        "latest_frame": max(e["frame_n"] for e in window),
        "head_ms_p50": float(np.median([e["head_ms"] for e in window])),
        "head_ms_p99": float(np.percentile([e["head_ms"] for e in window], 99))
        if len(window) > 1 else window[0]["head_ms"],
    }


def alert_state(agg: Dict) -> str:
    """Map aggregated confidence to alert tier."""
    conf = agg["confidence"]
    if conf >= ALERT_THRESHOLD:
        return "ALERT"
    elif conf >= WARN_THRESHOLD:
        return "WARNING"
    else:
        return "MONITORING"


# ----- Render helpers ---------------------------------------------------------

STATE_STYLES = {
    "ALERT":      {"emoji": "🚨", "color": "#c0392b", "bg": "#fdecea"},
    "WARNING":    {"emoji": "⚠️", "color": "#d35400", "bg": "#fef5e7"},
    "MONITORING": {"emoji": "🟢", "color": "#27ae60", "bg": "#eafaf1"},
}


def render_alert_banner(aggregates: List[Dict]) -> None:
    """Top-of-page banner — fires when any stream is in ALERT state."""
    alerts = [a for a in aggregates if alert_state(a) == "ALERT"]
    if not alerts:
        st.success("🟢 All streams within normal monitoring range — no high-confidence "
                   "unsafe behavior detected.")
        return
    # Sort by confidence descending
    alerts.sort(key=lambda a: a["confidence"], reverse=True)
    banner_lines = ["🚨 **UNSAFE BEHAVIOR DETECTED**"]
    for a in alerts:
        banner_lines.append(
            f"&nbsp;&nbsp;• **Stream {a['stream_id']}** — {a['label']} "
            f"({a['confidence'] * 100:.1f}% confident, "
            f"{a['n_emissions']} samples in last {a['window_sec']:.0f}s)"
        )
    st.error("  \n".join(banner_lines))


def log_alerts(aggregates: List[Dict]) -> None:
    """Append new alerts to session log when stream transitions into ALERT state."""
    for a in aggregates:
        state = alert_state(a)
        sid = a["stream_id"]
        prev = st.session_state.last_alert_per_stream.get(sid)
        if state == "ALERT" and (prev != "ALERT"):
            st.session_state.alert_log.appendleft({
                "ts": a["latest_ts"],
                "iso": time.strftime("%H:%M:%S", time.localtime(a["latest_ts"])),
                "stream_id": sid,
                "label": a["label"],
                "confidence": a["confidence"],
            })
        st.session_state.last_alert_per_stream[sid] = state


def render_stream_tile(col, agg: Optional[Dict], sid: int) -> None:
    if agg is None:
        col.info(f"Stream {sid} — waiting for first window…")
        return
    state = alert_state(agg)
    style = STATE_STYLES[state]
    cls = agg["class"]
    conf = agg["confidence"] * 100
    label = agg["label"]
    age = time.time() - agg["latest_ts"]
    col.markdown(
        f"<div style='padding:14px;border-radius:8px;"
        f"background:{style['bg']};border-left:5px solid {style['color']}'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<b style='font-size:1.05em'>Stream {sid}</b>"
        f"<span style='color:{style['color']};font-weight:bold'>"
        f"{style['emoji']} {state}</span>"
        f"</div>"
        f"<div style='margin-top:6px;font-size:1.25em;color:{CLASS_COLORS[cls]}'>"
        f"<b>{label}</b>"
        f"</div>"
        f"<div style='color:#555'>"
        f"Aggregated confidence: <b>{conf:.1f}%</b> "
        f"({agg['n_emissions']} samples over {agg['window_sec']:.0f}s)"
        f"</div>"
        f"<div style='color:#777;font-size:0.85em;margin-top:4px'>"
        f"frame {agg['latest_frame']} · {age:.1f}s ago · "
        f"head {agg['head_ms_p50']:.1f}ms (p99 {agg['head_ms_p99']:.1f}ms)"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_aggregated_grid(aggregates: List[Dict]) -> None:
    stream_ids = sorted(a["stream_id"] for a in aggregates)
    if not stream_ids:
        st.info("No streams active yet…")
        return
    by_sid = {a["stream_id"]: a for a in aggregates}
    cols = st.columns(min(len(stream_ids), 2)) if len(stream_ids) > 1 else st.columns(1)
    for idx, sid in enumerate(stream_ids):
        render_stream_tile(cols[idx % len(cols)], by_sid.get(sid), sid)


def render_confidence_chart(events: List[Dict], window_sec: float = 60.0) -> None:
    if not events:
        return
    now = time.time()
    recent = [e for e in events if now - e["ts"] < window_sec]
    if not recent:
        return
    df = pd.DataFrame([{
        "t_rel_sec": e["ts"] - now,
        "stream": f"S{e['stream_id']}",
        "confidence": e["confidence"],
    } for e in recent])
    pivot = df.pivot_table(index="t_rel_sec", columns="stream",
                           values="confidence", aggfunc="mean")
    st.line_chart(pivot, height=200)


def render_alert_log() -> None:
    log = list(st.session_state.alert_log)
    if not log:
        st.caption("No alerts triggered yet in this session.")
        return
    df = pd.DataFrame([{
        "Time": a["iso"],
        "Stream": f"S{a['stream_id']}",
        "Detected behaviour": a["label"],
        "Confidence": f"{a['confidence'] * 100:.1f}%",
    } for a in log])
    st.dataframe(df, hide_index=True, use_container_width=True)


def render_summary_metrics(events: List[Dict], aggregates: List[Dict]) -> None:
    if not events:
        return
    now = time.time()
    last_30s = [e for e in events if now - e["ts"] < 30.0]
    n_streams = len(set(e["stream_id"] for e in events))
    alert_count = sum(1 for a in aggregates if alert_state(a) == "ALERT")
    warning_count = sum(1 for a in aggregates if alert_state(a) == "WARNING")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total events", f"{len(events)}")
    c2.metric("Events / 30 s", f"{len(last_30s)}")
    c3.metric("Active streams", f"{n_streams}")
    c4.metric("Streams in ALERT", f"{alert_count}",
              delta=None if alert_count == 0 else "Action needed")
    c5.metric("Streams in WARNING", f"{warning_count}")


# ----- Main page --------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Jetson Edge — Live Monitor",
                       layout="wide", initial_sidebar_state="expanded")
    st.title("Unsafe Behavior Detection — Jetson Edge Live Monitor")
    st.caption(
        "Camera → Jetson Orin Nano Super (TensorRT FP16 inference) → SSH transport → "
        "this M4 dashboard. Raw frames stay on edge; only predictions traverse the network."
    )

    # Sidebar — connection + policy config
    st.sidebar.header("Edge connection")
    jetson_host = st.sidebar.text_input("Jetson SSH host", DEFAULT_JETSON_HOST)
    remote_path = st.sidebar.text_input("Remote JSONL", DEFAULT_REMOTE_PATH)
    poll_sec = st.sidebar.slider("Poll interval (sec)", 0.5, 5.0,
                                  POLL_INTERVAL_SEC, 0.5)
    st.sidebar.markdown("---")
    st.sidebar.header("Decision policy")
    window_sec = st.sidebar.slider("Rolling window (sec)", 1.0, 10.0,
                                    ROLLING_WINDOW_SEC, 0.5)
    alert_thr = st.sidebar.slider("ALERT confidence threshold", 0.50, 0.95,
                                   ALERT_THRESHOLD, 0.05)
    warn_thr = st.sidebar.slider("WARNING confidence threshold", 0.30, 0.80,
                                  WARN_THRESHOLD, 0.05)
    # Mutate module globals from slider (small hack — keeps render simple)
    globals()["ALERT_THRESHOLD"] = alert_thr
    globals()["WARN_THRESHOLD"] = min(warn_thr, alert_thr - 0.05)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Edge node")
    st.sidebar.metric("Edge target", "Jetson Orin Nano Super")
    st.sidebar.metric("Inference engine", "TensorRT 10.3 FP16")
    st.sidebar.metric("nvpmodel", "MAXN_SUPER")
    st.sidebar.metric("Measured FPS (N=1)", "116–153")

    # Poll loop
    placeholder = st.empty()
    while True:
        ok = pull_stream(jetson_host, remote_path)
        events = read_events()

        # Compute per-stream rolling aggregates
        stream_ids = sorted(set(e["stream_id"] for e in events)) if events else []
        aggregates = [a for a in (aggregate_stream(events, sid, window_sec)
                                  for sid in stream_ids) if a]
        log_alerts(aggregates)

        with placeholder.container():
            health_col, count_col = st.columns([1, 4])
            if ok and events and (time.time() - events[-1]["ts"]) < 5.0:
                health_col.success("● Jetson live")
            elif ok:
                health_col.warning("● Jetson reachable, no recent events")
            else:
                health_col.error("● Jetson unreachable")
            count_col.write(f"Polling `{jetson_host}:{remote_path}` every "
                            f"{poll_sec}s · rolling {window_sec:.0f}s window · "
                            f"alert ≥ {alert_thr:.0%} · warn ≥ {warn_thr:.0%}")

            if aggregates:
                # Top alert banner — first thing operator sees
                render_alert_banner(aggregates)

                st.subheader("Per-stream status (rolling-window aggregation)")
                render_aggregated_grid(aggregates)

                st.subheader(f"Per-emission confidence (last 60 s)")
                render_confidence_chart(events, 60)

                st.subheader("Stream stats")
                render_summary_metrics(events, aggregates)

                st.subheader("Alert log (session)")
                render_alert_log()
            else:
                st.warning(
                    "No events yet. Start the emitter on Jetson:\n\n"
                    "```bash\n"
                    "ssh jetson 'cd ~/my-project && python3 "
                    "deployment/scripts/jetson-multi-stream-pipeline.py "
                    "--streams 4 --source data/test/0_*/0_te1.mp4 "
                    "data/test/1_*/1_te1.mp4 data/test/3_*/3_te1.mp4 "
                    "data/test/4_*/4_te1.mp4 "
                    "--max-frames 18000 --fps 30 --mode fp16 "
                    "--emit-jsonl ~/jetson-stream.jsonl'\n"
                    "```"
                )

        time.sleep(poll_sec)


if __name__ == "__main__":
    main()
