# R6 — Operational safety-alarm & streaming evaluation protocol

Scope: gap = paper evaluates ONLY 64 pre-cut clips (top-1/macro) + 2 timing numbers. Never evaluated
as a continuous-stream alarm system. §4a-4h of `brief.md` (binding contract) also does NOT cover this —
it's clip-classification + static Jetson single-inference latency only. This report specs the missing
axis: streaming/alarm/reliability/privacy eval, additive to (not overlapping) the existing contract and
`eval_utils/` (clip-level `safety_metrics.py`, `calibration.py`, `bootstrap_ci.py` — all static, no time
axis). Nothing here modifies those files.

**Repo-fact that shapes everything below: there is NO continuous/untrimmed footage anywhere in this
project.** `data/{train,test}/` = pre-cut clips only, 7–15 s @ ~24.8 fps, one action each (paper's own
dataset design). `deployment/artifacts/live-demo-stream.jsonl` (8988 lines) is 4 pre-cut clips looped
per stream, not a real continuous camera feed with intermixed safe/unsafe periods. Real event-level
streaming eval therefore needs a **synthetic stream** built by concatenation (§B) — this is the central
finding of this report.

---

## A. Priority-ranked metric list

Cost = wall-clock engineering hours, 1 person, assumes infra in §C exists. Ranked by (leverage ÷ cost).

| # | Item | Formula/method | Data needed | Cost (h) | Priority |
|---|---|---|---|---|---|
| 1 | **Synthetic streaming harness** (concat clips + GT event log) | §B | existing 64 unsafe + safe(4-7) clips | 4–6 | **P0 — unlocks 2,3,4** |
| 2 | **Alarm metrics**: FAR/hr, TTD p50/p95, miss-rate, alarm precision, MTBFA, DET curve | §RQ2 below | stream from #1 + emission log | 3–4 | **P0** |
| 3 | **Glass-to-alarm latency decomposition** (stacked stage budget) | §RQ5 | timer instrumentation in `jetson-multi-stream-pipeline.py` | 3–4 | **P0 — paper has zero comparable figure** |
| 4 | **Privacy egress proof** (tcpdump test) | §RQ7 | live Jetson + dashboard run | 2 | **P1 — directly serves proposal's "bảo mật dữ liệu" motivation** |
| 5 | **Paper's 2 numbers, reproduced + extended** (0.14 s, 1.87 s) | §RQ4 | frame-gate script + benchmark harness | 2–3 | **P1 — already a §4a commitment; this is the streaming extension** |
| 6 | Temporal flicker + latency-vs-stability tradeoff curve | §RQ3 | #1 + sweep `ROLLING_WINDOW_SEC` | 2–3 | **P2** |
| 7 | Comparative table (paper vs thesis) | §D | synthesis, no new run | 1–2 | **P2 — cheap, high narrative value** |
| 8 | Soak extension ≥2 h + mem/throttle stats | §RQ6 | Jetson idle time (Day 7 buffer) | 1–2 h setup + 2 h wall-clock | **P2 — only if Jetson free** |
| 9 | Event-level tIoU / mAP-over-tIoU (ActivityNet/THUMOS protocol) | §RQ1 | #1, ≥30 synthetic events | 2 (analysis, reuses #1) | **P3 — nice-to-have, wide CI at n≈30-60 events, say so** |

Total P0+P1 ≈ 14–19 h (~2 days). Fits only if Lane B's Day 7 buffer or Lane F slack absorbs it — it is
**not currently in the brief's task list**; flag to owner before committing days.

---

## RQ1 — Event-level vs clip-level: what's feasible

Formal toolkit (standard, cite for the methods chapter):
- **Temporal IoU**: tIoU(a,b) = |a∩b| / |a∪b| (interval overlap ratio).
- **Segment P/R/F1 @ tIoU θ**: greedy/Hungarian-match predicted segments to GT by descending
  confidence; match valid iff tIoU ≥ θ and GT unmatched. TP=matched, FP=unmatched pred, FN=unmatched
  GT. P=TP/(TP+FP), R=TP/(TP+FN), F1=2PR/(P+R). Report at θ∈{0.3,0.5,0.7} (THUMOS-style).
- **mAP over tIoU** (ActivityNet/THUMOS): AP per class from confidence-ranked, tIoU-matched
  detections; mAP = mean AP across θ grid (ActivityNet: {0.5,0.75,0.95}; THUMOS: {0.3:0.1:0.7}).
- **Online action detection (OAD) point-level metrics** (De Geest et al. 2016; Gao et al. 2017 "RED"):
  per-frame classification treated as instance → per-frame mAP / per-frame F1, no segment boundary
  matching. This is what a continuously-running clip-classifier effectively degenerates to when you
  don't have real segment boundaries.

**Which applies here**: with clip-level-only labels (current state) — NONE of tIoU/segment-F1/mAP-tIoU
are computable; no start/end timestamps exist for events inside a continuous stream because no
continuous stream exists. Only OAD-style point-level metrics are computable directly (treat each
sliding-window emission as a point prediction, compare to the clip's single label) — this is
equivalent to what §4b already does, just re-labeled as "point-level."

**Minimal annotation to unlock real segment metrics**: build a **synthetic stream** (§B) by
concatenating existing labeled clips. Concatenation timestamps ARE exact GT start/end — **zero
incremental annotation labor**, cost is purely engineering (~4-6 h, counted in item 1). This is
categorically different from "annotate 20 clips for onset/offset" (also considered, ~2 min/clip × 20 =
40 min, but low value: clips are already tightly trimmed around a single action per the paper's own
dataset design, so sub-clip onset precision has nothing to buy against real-world value; **do not do
this** — the concat approach dominates it in both cost and realism).

**Verdict**: worth doing (cost ≈ 4-6 h unlocks tIoU/segment-F1/mAP-tIoU/alarm-latency all at once), but
report mAP-over-tIoU with wide CI caveat — a synthetic stream stitched from 64 clips yields at most
~50-80 discrete events, n too small for a tight interval (bootstrap over events, not over emissions).

---

## B. Synthetic streaming harness — the unlock

Script (new, e.g. `deployment/scripts/build-synthetic-stream.py` — NOT owned by this report, propose to
whichever agent owns `deployment/scripts/`):
1. Sample an ordered sequence of clips from `data/test/{0..7}_*` (mix unsafe 0-3 and their safe
   counterparts 4-7), shuffle with fixed seed.
2. Insert a **≥2 s safe filler buffer** between clips (repeat last frame or use a genuinely safe
   clip) — required because `T_WINDOW=32` frames @ ~24.8 fps ≈ 1.3 s sliding window; without a buffer
   the window straddles two different labels and neither ground truth nor emission is well-defined at
   the boundary (an artifact of concatenation, not of real deployment — real events have organic
   lead-in/lead-out).
3. Concatenate frames (opencv writer or feed frame-by-frame to the existing pipeline in order),
   log GT event list: `[{event_id, class, t_start_sec, t_end_sec, source_clip}]` computed purely from
   cumulative clip durations — no manual labeling.
4. Run the existing inference pipeline (`jetson-multi-stream-pipeline.py` logic or `app.py`'s
   `predict()`) over the concatenated stream, emit predictions exactly as `live-demo-stream.jsonl`
   already does, extended per §C schema.
5. Repeat with several random orderings/seeds (≥5) to get event-count large enough for non-degenerate
   bootstrap CIs on alarm metrics.

Target stream length: 20-40 clips ≈ 5-10 min per synthetic stream × 5 seeds = enough events (~100-150
total) for §RQ2 distributions to not be single-digit-n.

**Explicit external-validity caveat** (state in thesis): hard-cut concatenation ≠ real camera feed.
No camera motion/lighting drift across the boundary, no organic gradual onset. Good enough to compare
*relative* effect of window size / debounce policy (§RQ3) and to produce the DET curve shape, NOT a
substitute for field validation. Say this explicitly, don't let the figure imply more than it proves.

---

## RQ2 — Alarm-system metrics (the ones a safety engineer asks for)

Let a "true event" = GT unsafe interval [t_s, t_e] (class ∈ {0,1,2,3} = the 4 unsafe behaviours). Let
an "alarm" = a stream transition into ALERT state (as `jetson-live-monitor.py` already implements:
rolling-window mean-prob → argmax → threshold tiers, `ALERT_THRESHOLD=0.75`/`WARN_THRESHOLD=0.55`).

- **False Alarm Rate (FAR)** = `n_false_alarms / stream_hours`. Per-camera-per-shift variant: compute
  per `stream_id`, normalize by that stream's active hours in the shift window (e.g. 8 h).
  `n_false_alarms` = alarms with no overlapping GT unsafe interval of the *alarmed class*
  (class-specific matching — an ALERT for the wrong class while the right class is truly occurring is
  BOTH a miss on the true class AND a false alarm on the alarmed class; count both).
- **Time-To-Detection (TTD) / alarm latency** = `t_alarm - t_event_start` for each GT event that
  received a matching alarm. Report distribution: p50, p95 (not mean — heavy right tail from
  window/debounce delay is the whole point of §RQ3). Negative TTD (alarm before GT onset) should not
  occur by construction in the synthetic stream; if it does, it's a boundary-buffer bug — flag as QA
  check, don't silently clip.
- **Missed-event rate** = `n_events_with_no_alarm / n_events_total`.
- **Alarm precision** at the deployed operating point = `TP_alarms / (TP_alarms + FP_alarms)`.
- **Mean Time Between False Alarms (MTBFA)** = `stream_hours / n_false_alarms` (reciprocal of FAR;
  report both — engineers ask for MTBFA in operator-facing docs, FAR/hr in tables).
- **Debounce / N-of-M voting / hysteresis**: `jetson-live-monitor.py` already has 2-tier hysteresis
  (ALERT/WARNING/MONITORING by threshold) but **no refractory timer** — a stream flapping ALERT→
  WARNING→ALERT repeatedly for the same ongoing event currently logs only a state-transition
  (`log_alerts()` only appends on `prev != "ALERT"`, so it's already debounced per-continuous-ALERT-run,
  but there is no minimum-dwell requirement before firing). Recommend adding: fire ALERT only after N
  of the last M rolling-window emissions agree (N-of-M voting), e.g. N=3,M=5, as an explicit
  measurable parameter, and report flap count before/after (§RQ3).
- **DET-style curve for streaming**: x = FAR (log scale, per hour), y = miss-rate (%). Sweep the
  alert-confidence threshold (and separately, window size W) to trace the curve — classic
  Detection Error Tradeoff plot (Martin et al. 1997, NIST). This replaces a single-operating-point
  claim with the full tradeoff surface — directly answers brief §4d's "operating point selected by
  safety recall" but in event-space instead of clip-space.

**Cite + explicit pitfall — point-adjusted F1 (PA-F1)**: in time-series/streaming anomaly detection,
a common but **broken** protocol relabels an ENTIRE GT anomalous segment as correctly detected if ANY
single point inside it was flagged. Xu et al. 2018 popularized it; Kim et al. 2022 ("Towards a Rigorous
Evaluation of Time-Series Anomaly Detection", AAAI) show a **random-score detector scores >0.9 PA-F1**
on standard benchmarks — the metric is gameable by definition. **Do not use PA-F1 as a headline number
here.** Use tIoU-matched segment P/R/F1 (§RQ1) instead, which requires genuine temporal overlap, not
"touched one point." If PA-F1 is reported at all (e.g. for comparability with anomaly-detection
literature), always report it ALONGSIDE the un-adjusted point-level F1 and the tIoU-matched F1 in the
same table, never alone.

Also cite: Lavin & Ahmad 2015 (Numenta Anomaly Benchmark, NAB) — application-weighted scoring that
rewards early detection inside a window and penalizes FPs — a reasonable alternative scoring
philosophy for the TTD trade-off, but note the counter-critique: Wu & Keogh 2021 ("Current Time Series
Anomaly Detection Benchmarks Are Flawed", TKDE) show several widely-used streaming-AD benchmarks
(including ones NAB-adjacent) contain trivial/mislabeled anomalies that inflate reported numbers across
the field — a general caution to sanity-check the synthetic stream's "difficulty" isn't trivially easy
(e.g. don't let filler buffers make transitions absurdly obvious).

---

## RQ3 — Temporal consistency & the stability/latency tradeoff (the paper has nothing like this)

- **Flicker rate** = fraction of consecutive same-stream emission pairs where argmax class changes
  WITHOUT crossing a GT segment boundary between them. `flicker_rate = n_within_segment_switches /
  n_within_segment_consecutive_pairs`. Temporal Consistency Score = `1 - flicker_rate`.
- **Window-size sweep**: `jetson-live-monitor.py`'s `ROLLING_WINDOW_SEC` slider already spans 1-10 s —
  reuse directly. Sweep W ∈ {0.5, 1, 2, 3, 5, 8} s (also sweep N-of-M vote params if added). For each
  W measure: flicker_rate (↓ as W↑), TTD p50/p95 (↑ as W↑), FAR/hr (↓ as W↑ up to saturation),
  miss-rate (may ↑ at very large W if event is shorter than W).
- **The thesis figure**: 2D Pareto plot, x = alarm latency (TTD p50, ms/s), y = FAR (per hour, log) or
  miss-rate (%), points/curve parametrized by W. This is explicitly the figure the brief's
  hard-rules (§9.1: "no comparison without an interval") wants — plot with bootstrap CI bands per W
  (resample events, not emissions, to respect within-event correlation).
- Report flap count (ALERT↔WARNING transitions per event) pre/post adding N-of-M debounce, as a
  secondary stability number.

---

## RQ4 — Reproducing + extending the paper's 2 numbers

### 0.14 s/clip action-recognition time
Paper's scope is ambiguous (full pipeline vs classifier-only, hardware unstated). Measure BOTH,
declare scope explicitly per number:
- **(a) Classifier-only** (post-tokenization → class probs): most comparable framing. Already have the
  pattern in `deployment/scripts/benchmark-harness.py` → `benchmarks.md` (e.g. cascade head 17.5 ms
  p50 on M4 CPU-FP32). Just re-run on Jetson TRT and report against paper's 0.14 s (=140 ms) — note our
  numbers are already 10-100x faster; frame this as "not comparable, ours strictly dominates" not as
  a contest.
- **(b) Full per-clip pipeline** (file open → decode all frames → preprocess → backbone → classifier →
  probs): literal reading of "time to process one clip." Time with `time.perf_counter()`,
  t0=`cv2.VideoCapture(path)` open, t1=after final class-probs returned.
- **Timer placement rules**: exclude model/session load (one-time cost, not inference); exclude file
  I/O to fetch video off network (assume local decode); INCLUDE decode+preprocess in variant (b),
  EXCLUDE in variant (a).
- **Warmup**: discard first 10-20 iterations (TensorRT engine + CUDA context warmup; CoreML ANE
  warmup already implicit in repo's 100-run harness pattern — confirm and reuse).
- **Repetitions**: ≥30 distinct clips × ≥5 repeats = ≥150 samples minimum (more than the paper's
  presumed n=64-or-fewer, unstated). Report mean, sd, p50, p95, min, max — **not a bare mean** (the
  paper gives a single number with no distribution; that's the low bar we should visibly clear).

### 1.87 s reduced average duration
Paper's frame-reduction gate discards non-action frames; reduced clips average 1.87 s. Our
`run_yolo_framegate.py` currently gates **in feature space and always resamples back to T=32** — it
does NOT currently emit a variable-length "seconds" number comparable to the paper's claim. **Needed
instrumentation change** (measurement-only, does not touch gate logic): before the resample step in
`gate_indices()`, record `len(keep)` and `(max(keep)-min(keep))/native_fps` per clip pre-resample;
run across all 64 unsafe test clips; report distribution (mean, p50, p95) of reduced duration in
seconds, not a bare mean, exactly as above.

### Superior operational metrics that replace these two numbers
- **End-to-end glass-to-alarm latency** (§RQ5) — the paper never measures anything past
  "classifier finishes"; nothing about capture, alert delivery, or a running system.
- **Sustained throughput under multi-camera load** — already have `multi-stream-jetson-fp16-realtime.json`
  pattern (aggregate FPS + per-stream p50/p95/p99 head latency by N=1..4, 0 drops). Extend duration
  (current runs are 240 frames ≈ 10 s — too short for a "sustained" claim) to ≥10 min continuous.
- **Streaming latency under load** = per-stream p50/p95/p99 head latency AND end-to-end glass-to-alarm
  latency, both conditioned on N concurrent streams — shows degradation curve the paper's single-clip
  timing cannot show at all (paper never ran more than one clip at a time).

---

## RQ5 — End-to-end latency decomposition

Stage boundaries (map to actual code):
`capture` (frame acquisition, `cap.read()` in `app.py`/`jetson-multi-stream-pipeline.py`) →
`decode` (video demux, folded into capture for file-based sources; separate stage only if RTSP/camera
adds explicit decode) → `preprocess` (`pu.preprocess(frame)`) → `detector/tokenize` (YOLO backbone
`sessions["backbone"].run(...)`, produces per-frame token) → `transformer head` (`predict()`:
stats-agg + direct4/cascade/ensemble) → `post-process` (argmax, threshold, alert-state decision in
`alert_state()`) → `alert delivery` (write JSONL line → **rsync poll, currently `POLL_INTERVAL_SEC=1.0`
in `jetson-live-monitor.py`** — this is very likely the single largest term in the whole budget, 0.5-3 s
depending on jitter, dwarfing every model-inference stage measured so far; flag prominently).

**Instrumentation**: insert `time.perf_counter()` timestamps at each boundary; write per-emission
`stage_ms` dict into the JSONL record (schema §C). Aggregate stacked bar chart: median contribution per
stage, summed = total glass-to-alarm latency, with p95 error bars per stage.

**Real-time criterion (rigorous)**: model the pipeline as a **G/G/1 queue**, arrival rate
λ = source frame rate (24 fps ⇒ 41.7 ms/frame), service rate μ = sustained processing rate. Stability
requires **ρ = λ/μ < 1** strictly; if ρ ≥ 1 queue length grows unboundedly (Lindley recursion:
`W_{n+1} = max(0, W_n + S_n - A_n)` diverges in expectation). Near ρ→1, mean wait blows up
(Kingman/Pollaczek-Khinchine heavy-traffic approximation) even if nominally "keeping up" — so report
margin `1-ρ`, not just "μ > λ" as a pass/fail. **Zero-drop proof method**: run sustained load for
duration T ≫ 1/(μ-λ) (settling-time scale), count `frames_in`, `frames_processed`, `frames_dropped`
continuously (not the current fixed-240-frame snapshot in `multi-stream-results.json` — extend to
≥10 min unbounded run per §RQ4's throughput item); zero-drop over that duration at the target N is the
empirical proof, current 240-frame/~10 s runs are too short to license a "real-time" claim.

---

## RQ6 — Reliability / availability under sustained operation

Existing template: `deployment/artifacts/soak-jetson-summary.json` (36.2 min active / 42.5 min wall,
0 throttle events, tj_c max 56.8°C vs 80°C threshold, power mean 7.0 W of 15 W budget). Good schema,
reuse verbatim, extend:
- **Uptime** = active_duration / wall_duration (current 36.2/42.5 = 85.2% — investigate the ~6.3 min
  gap before quoting this; likely tegrastats sampling gaps, not real downtime — don't let an artifact
  masquerade as an availability number).
- **Drop rate** = dropped/total frames over the FULL soak (not the 240-frame multi-stream snapshot).
- **Thermal throttling events** = count of samples where `tj_c ≥ throttle_threshold_c` (80°C) —
  already computed, 0 in the 36-min pilot.
- **Memory stability**: `soak-timeseries.csv` currently has no RAM column
  (`t_sec,gpu_util_pct,gpu_freq_mhz,cpu_c,gpu_c,tj_c,power_w`) — **instrumentation change**: add
  RAM-used/swap columns from `tegrastats` (it reports `RAM x/yMB` natively, just not parsed into this
  CSV yet). Test for leak via linear regression slope of RAM-used vs t_sec; significant positive slope
  over a multi-hour soak = leak signal.
- **Recovery from stream loss**: kill the camera/file source mid-run, measure reconnection time and
  whether the pipeline crashes or self-heals; report as a discrete pass/fail + reconnection-latency
  number, not a continuous metric.
- **Minimum credible soak duration**: current 36.2 min is a pilot, not a claim-worthy soak. Thermal
  steady-state is typically reached within the first 20-30 min (already visible: tj_c plateaus early
  in the existing log) so 36 min barely captures ~10-15 min of steady-state. Recommend **≥2 h**
  continuous as the minimum credible thesis claim (Day 7 buffer slot in the brief's schedule is the
  only place this fits) — gives >1.5 h at steady state. An 8 h (full shift) soak would be ideal but is
  schedule-incompatible; if only 2 h is achieved, **explicitly do not extrapolate to "24/7 capable"**
  (see pitfall §E) — state it as "stable for the tested duration," full stop.

---

## RQ7 — Privacy as a measurable property, not a claim

Claim in `jetson-live-monitor.py` docstring: "Raw video frames NEVER leave the Jetson; only prediction
metadata... crosses the network." This is currently an assertion in a comment, not a test result.

**The test that proves it** (script, ~2 h to build + run once during a live demo session):
1. Run `tcpdump -i <iface> -w capture.pcap` on the Jetson (or on the network segment between Jetson and
   dashboard host) for a fixed window (e.g. 5 min) during live multi-stream inference.
2. Filter to the Jetson↔dashboard connection (rsync/SSH port 22 traffic).
3. **Byte-volume check**: sum bytes transferred in the window; compare against the theoretical
   raw-video-equivalent for the same duration at 1920×1080@24fps (~1.5 GB/s uncompressed, or
   H.264-compressed few hundred KB/s even compressed) vs actual JSONL egress (from
   `live-demo-stream.jsonl` sample, ~200 bytes/event × emission rate ≈ few KB/s). Report as an
   **N-times reduction factor** — this is the quantitative "metadata-only" evidence.
4. **Payload content check**: grep the pcap payload (post-decrypt is not possible/needed — see next
   point) for image magic bytes (`FFD8` JPEG, `89504E47` PNG) — should find NONE, confirming no frame
   data transits in cleartext form; note SSH encrypts the payload anyway (AES), so this check is
   belt-and-suspenders on top of the schema check (step 5) which is the real proof.
5. **Schema check** (the actual proof — already verifiable by inspection, done in this report):
   `live-demo-stream.jsonl` fields = `ts, iso, stream_id, frame_n, class, label, probs, confidence,
   head_ms` — all scalars, zero image/pixel data. Confirm the production emitter
   (`jetson-multi-stream-pipeline.py`) never adds an image/frame field — grep source for
   `cv2.imencode`/`imwrite`/base64 of frame buffers; absence = proof.
6. **On-device retention**: confirm no frame-to-disk writes on the Jetson beyond the transient
   in-memory ring buffer (`T_WINDOW=32` frames ≈ 1.3 s, held in a `collections.deque`, discarded on
   slide) — verify via `lsof`/`inotifywait` on the Jetson during a run: no growing file except the
   JSONL stream log.
7. **Transport encryption**: rsync over SSH ⇒ AES; verify cipher negotiation via `ssh -vv` log (already
   true by construction, cheap to document as a screenshot/log excerpt).

**Frameworks to cite** (design-lens, not compliance claim): GDPR Art. 5(1)(c) data minimisation +
Art. 25 privacy-by-design; Cavoukian's 7 Foundational Principles of Privacy by Design. Frame as "this
edge architecture instantiates data-minimisation by construction; here is the measured evidence,"
not "this system is GDPR-compliant" (no legal claim).

---

## C. Instrumentation changes + streaming-metrics JSON schema

**Changes needed in `deployment/`** (for whichever agent owns that dir — this report specifies, does
not implement):
1. `jetson-multi-stream-pipeline.py`: add per-stage `time.perf_counter()` timestamps; extend JSONL
   emission with `stage_ms` (RQ5) and `event_id`/`gt_class` when running against a synthetic stream
   (RQ1/RQ2 — null/absent field on real deployment).
2. New `deployment/scripts/build-synthetic-stream.py` (§B): concat clips, emit GT event log
   `synthetic-stream-gt.json`.
3. `jetson-live-monitor.py`: add optional N-of-M vote debounce (currently 2-tier hysteresis only, no
   dwell requirement — §RQ3); log alarm transitions with enough fields to compute TTD (already close:
   `alert_log` has `ts`, needs `event_id` linkage when run against synthetic GT).
4. Tegrastats parser (wherever `soak-timeseries.csv` is generated): add RAM/swap columns (§RQ6).

**Extended JSONL schema** (superset of current `live-demo-stream.jsonl` fields, backward compatible —
old fields unchanged, new fields additive/optional):

```json
{
  "ts": 1781282733.732,
  "iso": "2026-06-12T23:45:33.732",
  "stream_id": 1,
  "frame_n": 32,
  "class": 0,
  "label": "walkway_violation",
  "probs": [0.72, 0.27, 0.00, 0.01],
  "confidence": 0.7208,
  "head_ms": 34.31,

  "stage_ms": {
    "capture": 2.1, "decode": 0.0, "preprocess": 1.4,
    "detector": 4.8, "transformer": 17.5, "postprocess": 0.2,
    "alert_delivery": 850.0
  },
  "alarm_state": "ALERT",
  "alarm_transition": true,

  "event_id": "synstream-003-ev07",
  "gt_class": 0,
  "gt_t_start": 142.3,
  "gt_t_end": 149.8,

  "source": "synthetic"
}
```

`event_id`/`gt_*`/`source` are present only for synthetic-stream runs; absent (or null) on real
deployment logs where no ground truth exists. `alarm_transition=true` marks the emission where
`alarm_state` changed from the previous emission for that `stream_id` (drives TTD/FAR computation
directly off the log without re-deriving state machine externally).

---

## D. Comparative table design (paper vs thesis) — likely the single most important figure

Row-groups = axis; two-column diff (Paper / This thesis). Sparse "not reported" cells on the Paper
side are the point — don't pad them with guesses.

| Axis | Paper (Önal & Dandıl 2025) | This thesis |
|---|---|---|
| **Eval unit** | 64 pre-cut clips | 64 clips (parity) **+ synthetic continuous stream** |
| **Accuracy metric** | top-1 (62/64) | top-1 + macro-F1 + balanced acc + κ + MCC + confusion matrix |
| **Statistics** | single point, no CI | bootstrap 5000 + Wilson/BCa, McNemar exact, Nadeau-Bengio k-fold |
| **N** | 64 | 64 (parity) + 516 (k-fold/LCO) |
| **Generalisation** | same-distribution only | leave-camera-out (78.6% macro vs 16.9% scene-baseline) |
| **Safety/alarm behaviour** | none | FAR/hr, TTD p50/p95, miss-rate, alarm precision, MTBFA, DET curve |
| **Temporal stability** | none (clips are pre-cut, no continuity) | flicker rate, latency-vs-stability Pareto curve |
| **Operational timing** | 1 number (0.14 s/clip), no distribution, scope unstated | distribution (mean/sd/p50/p95), scope declared, + full stage decomposition |
| **Frame reduction** | 1 number (1.87 s avg), no distribution | distribution across 64 clips, instrumented |
| **Throughput** | none | sustained FPS by N concurrent streams, zero-drop over ≥10 min |
| **Real-time criterion** | asserted informally | queueing-theory margin (1-ρ), zero-drop proof |
| **Reliability/soak** | none | tegrastats soak, throttle events, thermal/mem stability |
| **Privacy** | not addressed | quantified egress reduction + schema/on-device-retention proof |
| **Energy** | none | J/inference, FPS/W (see §4g, owned elsewhere) |

---

## E. Pitfalls (explicit warnings)

1. **Point-adjusted F1** (§RQ2) — inflates scores arbitrarily, a random detector clears 0.9. Never
   headline it; always pair with tIoU-matched or raw point-level F1.
2. **Unbounded soak extrapolation** — 36.2 min (or even 2 h) measured stability ≠ "24/7 production
   ready." State the tested duration as the claim boundary, explicitly.
3. **Cherry-picked operating point** — reporting FAR/miss-rate at one threshold without the DET curve
   hides how fragile the choice is. Always show the curve, then name the chosen point on it.
4. **Synthetic-stream realism ceiling** — hard-cut concatenation has no organic onset/lighting/motion
   drift; good for relative comparisons (window size, debounce), not a field-validation substitute.
   Say so in the same paragraph as the numbers, not buried in a footnote.
5. **Small-n event counts** — a stitched stream from 64 clips yields ~50-150 discrete events at most;
   mAP-over-tIoU / DET curves at this n have wide CIs. Bootstrap over events (not emissions, which are
   correlated within an event) and show the interval.
6. **Availability-artifact confusion** — soak "active" vs "wall" duration gap (36.2/42.5 in the
   existing summary) may be a logging gap, not downtime. Investigate before quoting an uptime %.
7. **Confusing classifier-only latency with pipeline latency** when comparing to the paper's 0.14 s —
   state scope every time a latency number appears near that comparison.

---

## Unresolved questions

1. Does `jetson-multi-stream-pipeline.py` currently support ingesting a pre-built concatenated video /
   frame sequence with injected GT timestamps, or does §B's harness need to build that ingestion path
   from scratch? (Not confirmed from reading — file exists but full contents not reviewed line-by-line
   here; whoever implements should check.)
2. Is tcpdump/packet-capture tooling available/permitted on the Jetson and on whatever network segment
   sits between it and the dashboard host (corporate network restrictions, sudo access)? RQ7's test
   assumes it is.
3. Is there Jetson idle time in the schedule (Day 7 buffer is shared with B2 power/soak work already) to
   fit a ≥2 h soak without displacing committed §4g work?
4. Does the owner want this streaming/alarm axis added to the binding §4 contract at all, given it's not
   currently listed in `brief.md` §4 or the Day-by-day schedule — recommend a go/no-go decision before
   spending the ~14-19 h P0+P1 budget.
