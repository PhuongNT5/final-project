---
priority: must-have
effort: 2 person-days
depends_on: []
branches: A=real RTSP cameras, B=RTSP-over-file simulation (VLC/ffmpeg), C=file-only pipeline
---

# Phase 02 — Multi-Stream Live System

## Context Links

- `deployment/scripts/multi-stream-pipeline.py` — existing pipeline; promote to live demo
- `deployment/dashboard/app.py` — Streamlit dashboard
- `deployment/artifacts/multi-stream-results.json` — existing N=1..4 results (CPU-FP32)
- `deployment/README.md` §Phase 03 — multi-stream results table
- `report/chapter-05-edge-deployment.md` §5.5 — current multi-stream text; needs live-system upgrade
- `plans/260417-1913-track-b-edge-deployment/phase-03-multi-stream-pipeline.md` — prior spec

## Overview

**Already measured (GROUND_TRUTH.md §4):** N=1..4 FP32 multi-stream results exist in
`deployment/artifacts/multi-stream-results.json` — 240 frames per stream, 0 dropped frames
at every N, aggregate FPS: 17.6 / 29.1 / 26.2 / 20.8. Head p99 latency: 86 / 53 / 61 / 82 ms.
Alert latency percentiles from the JSON can be extracted directly. These numbers are already
referenced in Chapter 5 Table 5.4.

**Remaining work (2 days only):**
(a) Re-run multi-stream with INT8 and CoreML branches to add two columns to Chapter 5 Table 5.4.
(b) 30-min soak run with continuous logging (not done yet).
(c) Dashboard polish for defense demo.
(d) RTSP camera ingest path verified (or confirmed working with file simulation).

Drop the alert P50/P95/P99 instrumentation work from the original plan — head-latency percentiles
are already in `multi-stream-results.json` and can serve this purpose.

## Key Insights

- FP32 N=1..4 results already measured (see multi-stream-results.json). N=2 peaks at 29.1 FPS, 0 drops. N=4 drops to 20.8 FPS (backbone saturation) — legitimate finding.
- Head p99 latency actually *improves* from 86 ms (N=1) to 82 ms (N=4) because backbone batching offsets head contention. This is worth a sentence in Ch 5.
- INT8 and CoreML branches not yet measured for multi-stream — main remaining experiment.
- 30-min soak not yet done — required for the "demonstrated system" claim.
- Dashboard (`deployment/dashboard/app.py`) already exists; polish for demo-readiness is the final task.
- Drop alert-latency instrumentation from original plan scope — p99 head-latency in the JSON already covers this adequately for thesis purposes.
- Privacy: test videos are ethics-approved. Live RTSP from real cameras requires consent forms.

## Requirements

**Functional**
- Accept RTSP stream URLs or file paths as input (toggle via `--source rtsp://...` or `--source path/to/file.mp4`).
- Branch B: simulate live RTSP by serving files via VLC (`vlc --rtsp-host`) or ffmpeg (`ffmpeg -re -i file.mp4 -f rtsp rtsp://localhost:8554/stream0`); no real cameras required.
- Log alert latency: time from first frame of a sliding window predicted as "unsafe" to the moment the dashboard receives the alert.
- Output per-stream: frame count, drop count, inference count, alert count.
- Sustain 30-minute soak run at N=2, logging every 60 s.
- Produce `deployment/artifacts/live-system-results.json` with P50/P95/P99 alert latency.

**Non-functional**
- Pipeline thread model: one thread per stream + shared backbone queue (already implemented; verify under load).
- Dashboard must remain responsive (non-blocking) during soak; use websocket or polling interval ≤ 1 s.
- Log file rotation to prevent unbounded disk usage during soak.

**Acceptance thresholds**
- Alert P50 ≤ window_duration + 1 inference cycle = (32/30 + 0.033) s ≈ 1.1 s for single stream.
- 0 frame drops in 30-min N=2 soak (matching existing 240-frame result).
- Dashboard screenshot captured and saved to `deployment/artifacts/figures/dashboard-screenshot.png`.

## Architecture / Approach

### RTSP Source Setup (Branch B — file simulation)

```bash
# Install mediamtx (lightweight RTSP server, no VLC needed)
# brew install mediamtx  OR  pip install av + custom server
# OR use ffmpeg re-mux (simplest):
ffmpeg -re -stream_loop -1 -i data/test/0_safe_walkway_violation/0_te1.mp4 \
  -f rtsp -rtsp_transport tcp rtsp://localhost:8554/stream0 &
```

Pipeline receives `rtsp://localhost:8554/streamN` via `cv2.VideoCapture(url)` (already supported by OpenCV RTSP).

### Alert Latency Instrumentation

Add to `multi-stream-pipeline.py`:
```python
alert_ts = {}  # stream_id -> timestamp of first unsafe frame in window
dispatch_ts = {}  # stream_id -> timestamp alert sent to dashboard
alert_latency_ms = dispatch_ts[sid] - alert_ts[sid]
```
Collect into `alert_latencies` list; compute P50/P95/P99 at end.

### Soak Test Logging

30-minute run: log `{timestamp, stream_id, fps, drops, alerts}` every 60 s to `soak-log.jsonl`. At end, compute: total drops, alert rate, fps variance.

### Dashboard Recording

Use `scrcpy` or system screen-record (macOS `screencapture -V`) to capture a 2-min demo session. Export a representative PNG for thesis figure. Dashboard already has live alert feed; add a "Alert Latency" panel showing last 100 P50.

## Related Code Files

| Action | Path |
|--------|------|
| Modify | `deployment/scripts/multi-stream-pipeline.py` — add RTSP input, alert latency logging |
| Modify | `deployment/dashboard/app.py` — add Alert Latency panel, improve polling |
| Create | `deployment/scripts/rtsp-source-setup.sh` — ffmpeg RTSP server launch helper |
| Create | `deployment/scripts/soak-test.py` — 30-min automated soak |
| Create | `deployment/artifacts/live-system-results.json` |
| Create | `deployment/artifacts/figures/dashboard-screenshot.png` |
| Modify | `report/chapter-05-edge-deployment.md` §5.5 — add RTSP, alert latency table, soak result |

## Implementation Steps (Day-by-Day, 2 days)

**Day 1 — INT8 / CoreML multi-stream runs + soak**
1. Verify FP32 pipeline still reproduces: `python deployment/scripts/multi-stream-pipeline.py --streams 2 --max-frames 240`. Confirm 29.1 FPS.
2. Re-run with INT8 branch for N=1,2,3,4 (240 frames each). Log to `multi-stream-results-int8.json`.
3. Re-run with CoreML-FP32 branch for N=1,2,3,4. Log to `multi-stream-results-coreml.json`.
4. Run 30-min soak at N=2 FP32: `python deployment/scripts/soak-test.py --streams 2 --duration 1800`. Log drop count, FPS variance every 60 s to `soak-log.jsonl`.

**Day 2 — Dashboard polish + chapter update**
5. Set up RTSP simulation (Branch B): write `rtsp-source-setup.sh` (N ffmpeg `-re -stream_loop -1` processes). Verify `cv2.VideoCapture(rtsp://...)` works.
6. Polish dashboard (`deployment/dashboard/app.py`): remove debug output, ensure clean UI for defense demo. Confirm 4-stream panel visible.
7. Capture dashboard screenshot: save to `deployment/artifacts/figures/dashboard-screenshot.png`.
8. Update `report/chapter-05-edge-deployment.md` §5.5: add INT8/CoreML multi-stream columns to Table 5.4, soak result (drop count), dashboard screenshot reference. Add sentence on p99 head-latency behavior (improves N=1→4 due to backbone batching).

## Todo List

- [ ] Day 1: Verify FP32 pipeline reproduces 29.1 FPS at N=2
- [ ] Day 1: Run INT8 multi-stream N=1..4 → multi-stream-results-int8.json
- [ ] Day 1: Run CoreML-FP32 multi-stream N=1..4 → multi-stream-results-coreml.json
- [ ] Day 1: Run 30-min N=2 FP32 soak → soak-log.jsonl (0 drops criterion)
- [ ] Day 2: Write rtsp-source-setup.sh; verify RTSP ingest works
- [ ] Day 2: Polish dashboard (clean UI, no debug output)
- [ ] Day 2: Capture dashboard-screenshot.png
- [ ] Day 2: Update §5.5 with INT8/CoreML multi-stream columns + soak result

## Success Criteria

- Pipeline accepts RTSP URLs or file paths interchangeably.
- 30-minute soak at N=2: 0 dropped frames (matching 240-frame result).
- Alert latency P50 reported and ≤ 2 s (1 inference window + 1 dispatch cycle).
- `live-system-results.json` exists with N=1..4 FPS, drop count, alert latency P50/P95/P99.
- Dashboard screenshot saved and referenced in §5.5.
- §5.5 describes the system as "demonstrated" not "designed," with the measured numbers inline.

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| ffmpeg RTSP server setup fails | Low | Fall back to file-based pipeline with synthetic timestamps; clearly label as "file simulation" in thesis |
| cv2.VideoCapture RTSP timing drift | Medium | Use `time.monotonic()` for latency measurement, not video PTS; no real-time dependency for correctness |
| Dashboard blocking on inference queue | Low | Use `asyncio` or background thread for dashboard; already partially implemented |
| Alert latency > 2 s | Low | Reduce sliding-window stride K from 8 to 4 frames (halves alert lag at 2× compute cost); acceptable for N≤2 |
| 30-min soak crashes (memory leak) | Low | Run with `tracemalloc` for first 5 min; if leak present, fix via explicit `del` of frame buffers after inference |

## Security / Ethics / Privacy

- **Camera feeds**: live RTSP in Branch A from real cameras requires written consent from any persons in frame per GDPR/Vietnamese data law. For thesis purposes, use recorded Unsafe-Net clips (already ethics-cleared per §2.1 of paper) streamed via ffmpeg.
- **False-positive harm**: an alert for a safe behavior (false alarm) in a real deployment could trigger unnecessary evacuations. §5.5 must include a false-alarm rate note: current ensemble achieves 56/64 on unsafe subset; safe-class accuracy not evaluated here and must be stated as a limitation.
- **Data security**: do not embed actual factory camera URLs in any committed script. Use placeholder `rtsp://localhost:8554/streamN`.
- **Alert logging**: soak logs contain timestamps and class predictions. In a real deployment these constitute surveillance records; the thesis must note retention/access-control requirements.

## Dependencies + Next Steps

- Output consumed by: **Phase 07** (thesis §5.5 upgrade); **Phase 08** (2-min live demo script uses this system).
- Phase 01 results (Jetson) can be inserted into this system post-hoc; Phase 02 is independent.
- Landing chapter: `report/chapter-05-edge-deployment.md` §5.5.

**Unresolved questions**
- Is a real RTSP camera available for the defense demo? (If yes, consent form needed.)
- Is the dashboard currently publicly accessible, or only localhost? Thesis committee may expect a live demo URL.
