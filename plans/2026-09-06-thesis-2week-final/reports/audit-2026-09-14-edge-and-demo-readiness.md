# Audit — edge deployment + live demo readiness, 2026-09-14

Scope: read-only audit of `deployment/`, brief.md D2/D4/Gate3, prior review
(`review-2026-09-13-edge-deployment.md`), `laneb-p3-edge-cost-and-export.md`. Local checks only
(py_compile all deployment scripts, onnx.checker.check_model on P3 exports incl. new imgsz
variants) — no Jetson SSH, no network. Today = Day 9/14 of the brief's schedule; Gate 3 is
end of Day 10 (tomorrow, Sep 15).

## 1. Inventory of `deployment/`

### Scripts (`deployment/scripts/*.py`, `*.sh`) — all 24 .py files + 2 dashboard files pass
`py_compile` clean (checked this session).

| Script | Exists | Runnable (syntax) | Tested | Pipeline | Notes |
|---|---|---|---|---|---|
| `export-backbone.py`, `export-transformer-heads.py`, `transformer-head.py` | yes | yes | yes (Phase 01 parity) | P1 | ONNX export, run |
| `quantize-int8.py` | yes | yes | yes | P1 | ORT dynamic INT8 |
| `verify-onnx-parity.py` | yes | yes | yes | P1 | parity-report.txt exists |
| `extract-features-onnx.py` | yes | yes | yes | P1 | Phase 02 |
| `benchmark-harness.py` | yes | yes | yes | P1 | `benchmarks.json/md` exist |
| `multi-stream-pipeline.py` | yes | yes | yes (M4 CPU) | P1 | has H4/H5 bugs below |
| `jetson-bootstrap.sh`, `jetson-rsync-from-mac.sh`, `jetson-power-sweep.sh` | yes | shell, not checked | run on device 2026-06-03 per JETSON-DEPLOYMENT.md | P1 | |
| `jetson-build-trt-engine.py` | yes | yes | run on device | P1 only | no P3 engine build path |
| `jetson-benchmark.py` | yes | yes | run on device (P1 only) | P1 hardcoded (`MODELS=['backbone','direct4-s314','scenario-s7','pair0-3-s7']`, line 31-32) | H3 bug; cannot run P3 relation head (2 inputs) |
| `jetson-multi-stream-pipeline.py` | yes | yes | run on device (P1 only) | P1 hardcoded, same MODELS-shaped pipeline | H3/H4/H5 bugs; supports `--emit-jsonl` (feeds dashboard) |
| `jetson-prep-int8-calib.py` | yes | yes | run (produced calib-jetson/*.npy) | P1 | real train frames, no leakage (confirmed by prior review) |
| `export-p3-graphs.py` | yes | yes | yes, ran 2026-09-06 + 09 | P3 | exports detector/appearance/head + `--cost_imgsz` probes |
| `benchmark-p3-onnx.py` | yes | yes | yes, ran 2026-09-09 (Mac ORT-CPU) | P3 | offline only, no Jetson variant exists |
| `bench-utils.py`, `pipeline-utils.py`, `run-inference.py`, `colab-tensorrt-template.py`, `generate-figures.py` | yes | yes | — | mixed | supporting utils |

**No `jetson-benchmark-p3.py` or equivalent P3-on-device harness exists.** Every P3 number is
Mac ORT-CPU + a calibrated transfer-factor projection — nothing P3-specific has ever run on the
Jetson (confirmed: `jetson-benchmark.py`/`jetson-multi-stream-pipeline.py` MODELS lists are
P1-only; `trt-compat-report.md` scanned 18 files, all P1 — **0 P3 graphs**, so P3's TRT-10
parseability is still unverified, contra a checked-in claim that this was "pre-checked").

### Exports (`deployment/export/*.onnx`)

- P1: `backbone{,-ft}-fp32/int8`, `direct4-s314`, `scenario-s7`, `pair{0-3}-s7`, `eight-s7`
  (fp32+int8 each) — all present, dated Apr 19.
- P3: `p3-detector-fp32.onnx` (38.2 MB, imgsz800), `p3-appearance-fp32.onnx` (4.2 MB),
  `p3-relation-head-fp32.onnx` (1.4 MB), plus **new since the 2026-09-13 review**:
  `p3-detector-fp32-imgsz{640,512,416}.onnx` (cost-probe detectors, dated Sep 9 22:50).
- **Verified this session (onnx.checker.check_model, all pass):**
  - `p3-detector-fp32{,-640,-512,-416}.onnx`: opset 18, input `images [1,3,{800,640,512,416},{same}]`, output `output0 [1,300,6]` (NMS-free).
  - `p3-appearance-fp32.onnx`: opset 18, input `crops [boxes,3,224,224]`, output `[boxes,64]`.
  - `p3-relation-head-fp32.onnx`: opset 18, input **`tokens [batch,16,8,74]` + `mask [batch,16,8]` — confirms 2 inputs**, output `logits [1,4]`.
- No `.engine` files anywhere in `deployment/export/` — expected, since engines are
  device-specific and gitignored; none has ever been built for P3 on Jetson (no P3 build path in
  `jetson-build-trt-engine.py` at all).

### Demo/server/dashboard code

- `deployment/dashboard/app.py` — Streamlit, standalone video-upload demo (not networked to
  Jetson). M5 from the prior review stands: no auth, renders raw frames, fine only if
  `localhost`-bound.
- `deployment/dashboard/jetson-live-monitor.py` — the actual "video → Jetson → server →
  dashboard" architecture per its own docstring: Jetson runs
  `jetson-multi-stream-pipeline.py --emit-jsonl ~/jetson-stream.jsonl`, this M4 Streamlit app
  rsync-pulls that file over SSH every `POLL_INTERVAL_SEC=1.0` s, does rolling 3 s aggregation,
  renders ALERT/WARNING/MONITORING tiles. Privacy-by-construction (metadata only over the wire)
  is a real, well-documented property. **Runnable in isolation (syntax OK); untested as an
  actual live Jetson→Mac loop in this session (no device access).** No separate "server"
  process — the dashboard itself pulls via rsync, i.e. server+dashboard are the same script.
  **P3 is not wired to this at all** — the emitter it depends on (`jetson-multi-stream-pipeline.py`)
  only runs the P1 MODELS list.
- `deployment/artifacts/live-demo-stream.jsonl` — 8988 lines, timestamps `2026-06-12` (3 months
  stale relative to today). This is an **event-log sample**, not a video fallback recording —
  useful for testing the dashboard's parsing/rendering offline, but it is not what Gate 3 means
  by "fallback recording" (Gate 3 wants a recorded *demo run*, i.e. a video or a fresh
  self-contained replay artifact, for the case live fails at defence). **No `.mp4`/`.mov`/`.avi`
  fallback recording exists anywhere in the repo** (checked `find . -iname "*.mp4"` etc. under
  `deployment/` — zero hits).
- `deployment/README.md` — documents the P1 export→benchmark→demo flow (`streamlit run
  deployment/dashboard/app.py`), does not mention `jetson-live-monitor.py` at all — README is
  stale relative to the actual demo architecture.

## 2. Reported bugs — status (all confirmed still present, unfixed)

| Bug | File:line (current) | Status | Minimal fix |
|---|---|---|---|
| **TRTRunner single-input** | `deployment/scripts/jetson-benchmark.py:63` (`self.host_in, self.dev_in, self.in_name, self.in_shape = host, dev, name, shape` inside `if mode == INPUT` in a loop over all IO tensors); `deployment/scripts/jetson-multi-stream-pipeline.py:96` (`self.host_in, self.dev_in = host, dev`, same pattern) | **present**. Confirmed via onnx.checker this session that `p3-relation-head-fp32.onnx` genuinely has 2 inputs (`tokens`, `mask`) — so pointing either harness at it today silently keeps only the last-enumerated input's buffer and leaves the other's device memory unpopulated garbage (per prior review H3, worse than "can't bind"). | Replace the single `host_in/dev_in` pair with a `dict[name] -> (host, dev, shape)`; iterate that dict in `run()`. ~20-30 min in each file. Add an assert (`num_io_tensors <= 2`) as a stopgap so a future accidental multi-input run fails loudly instead of silently. |
| **aggregate_fps counts dropped frames** | `deployment/scripts/multi-stream-pipeline.py:151` (`total_frames = sum(s.frame_count ...)`) feeding `:154`; `deployment/scripts/jetson-multi-stream-pipeline.py:219` feeding `:222` | **present**, unchanged from prior review. `frame_count` increments on every frame *read* including ones dropped because the queue was full; `aggregate_fps` therefore mixes reader I/O rate with real inference throughput. Immaterial for the zero-drop headline configs, material for any saturation-point figure (e.g. Ch5's "136.89 FPS at N=4, 328/960 dropped" — that FPS number counts the 328 discarded frames). | Add a `processed` counter (incremented only where `buf` is appended to / inference actually runs) alongside `frame_count`; compute `aggregate_fps` from `processed`, keep `frame_count`/`dropped` for the read-rate report. ~15 min per file. |
| **benchmark-p3-onnx.py excludes non-graph stages; uses mean crop count** | `deployment/scripts/benchmark-p3-onnx.py:44` (`MEAN_CROPS = 58`), `run_bench` now at `:96-118` (was 100-113, file grew) | **present**, unchanged. `run_bench` times only `sess.run()` on synthetic already-shaped tensors — NMS, box→crop extraction/geometry (`iou_walkway`, `iou_restricted`, `foot_in_walkway`, `norm_dist`), and frame decode/resize are all outside the timed region, so the reported 0.55 dec/s (Mac) / 1.13-1.71 dec/s (Jetson-projected) are **optimistic lower bounds**, not the true number. `--crops` defaults to `MEAN_CROPS=58` (line 221); no `--crops 106` (p95) or `128` (cap) run exists in `benchmarks-p3-onnx-cpu.json` (checked this session — only `crops: 58` appears in the sweep). | Cheap partial fix (no code change, just a flag): re-run `--crops 106` and `--crops 128` for a worst-case row (appearance is only 4.4% of cost today, so this alone won't flip the FAIL verdict). Real fix (medium effort): build an end-to-end script that also times decode→NMS→crop-extraction→token-build, not just the 3 graphs — not done, correctly flagged as a stated limitation by the prior review, still not fixed. |

No new bugs found beyond the three; the prior review's C1/C2/H1/H2/H5/M1-M5/L1-L2 findings were
spot-checked for currency (file line numbers, `TRTRunner` code, `MODELS` lists, `trt-compat-report.md`
contents) and are all still accurate — nothing has regressed or been silently fixed since 2026-09-13.

## 3. P3 real-time options — ranked, with speedup estimate and risk

Baseline: measured Mac ORT-CPU 1825 ms/decision = 0.55 dec/s; detector is 95.6% of cost
(1745/1825 ms); Jetson-projected via calibrated P1 transfer factor 1.13-1.71 dec/s vs 3.0
required. **All numbers below are cost-only projections except where marked measured; every one
needs a fresh 5-seed blind-protocol accuracy re-check per brief S19/S20 discipline before any
number can be quoted in the thesis** — a single point estimate is explicitly banned by brief §4h.5.

| Rank | Option | Mechanism | Speedup (measured cost curve, `laneb...md` §6.1) | Projected Jetson fp16 dec/s | Accuracy re-check required | Risk |
|---|---|---|---|---|---|---|
| **1** | **imgsz 800→512, T=16 unchanged** | Detector cost scales ~quadratically with imgsz; T unchanged means the relation-head graph shape `(1,16,8,74)` doesn't change, so **no retrain of the head architecture, no re-export of the head graph** — only detector re-export + token rebuild + head *fine-tune/retrain* on new tokens | measured cost-probe: 1897→824 ms/decision (Mac) | **3.77** (int8 3.95) — clears 3.0 | Full: rebuild `tokens_yolo26app_*` at imgsz512, retrain relation head, 5-seed blind re-run on 64 test clips + ideally k-fold/LCO sanity. Accuracy at 512² is **completely unmeasured** — paper itself used 416² so precedent exists that lower res doesn't necessarily collapse accuracy, but this is P3's own untested claim, not inherited. | **Medium.** Cheapest credible route per the existing report's own analysis (imgsz doesn't touch head shape, only token values) — but "cheap to try" ≠ "cheap to validate honestly under the 5-seed/blind protocol", and could eat 1-2 days if it needs iteration. |
| **2** | **Detector TRT FP16 on Orin (vs Mac ORT-CPU)** | Pure hardware/precision speedup, same graph, same imgsz | measured P1 backbone gets 3.11x (fp16); if that ratio transfers to P3's detector, 1745→~561 ms just from this alone | rolls into the 1.13-1.71 dec/s figures already reported — **this IS the transfer factor being applied**, not an additional lever on top | None *if* output parity is verified (argmax match on 64 test clips through the TRT engine vs the ONNX reference) — this is a deployment-fidelity check, not a new accuracy claim, since the model itself is unchanged | **Medium-high**, not low: this is the option the prior review's option A covers, and it requires actually building the TRT engine on-device (TRT-10 parse of an 800² YOLO26 NMS-free graph is **unverified** — `trt-compat-report.md` never scanned any P3 graph, contra what a casual reading of the deployment doc might suggest) plus fixing the TRTRunner H3 bug (relation head has 2 inputs) before a full-pipeline number can exist. Compute-only, does not by itself clear 3.0 (1.64-1.71 dec/s per the already-measured/projected numbers) — **must combine with another lever** (e.g. #1) to clear the bar. |
| **3** | **Detector every-k-frames, reuse boxes across sliding windows (T=16, stride 8)** | `EMIT_EVERY=8 == T/2` already means consecutive decisions overlap 8 of 16 frames; caching the overlapping half's detections instead of re-running the detector on them roughly halves fresh-detector calls per decision | ~2x on the detector's 95.6% share → best case ~1.9x overall, i.e. ~1.0 dec/s Mac-equivalent, ~2-3x Jetson-projected (still likely short of 3.0 alone) | not independently measured; prior report's own estimate says this "alone doesn't clear ~2x-3x miss without also reducing imgsz" | Needs a qualitative accuracy sanity check on stale-detection risk: reused (t-8) boxes may miss entities that appeared mid-window, specifically risking the two classes already flagged fragile (WalkwayViol, UnauthorInt — brief S19/S20). No tokens/retrain needed if boxes are just cached, but the caching logic itself doesn't exist in the current P3 export/token pipeline — net-new code. | **Medium.** Intellectually the most "free lunch"-like (real overlap exists structurally), but nothing implements it yet; combine with #1 for headroom rather than relying on it alone. |
| **4** | **Batch frames through the detector** | Run the 16-frame window as one batched detector call instead of 16 sequential calls | Unquantified in any report read this session — depends on whether the exported detector graph supports batch>1 (P1's export notably does **not**: `deployment/README.md:85-87` "Dynamo export bakes batch=1... use batch=1 inference") | unknown — plausibly modest on Orin Nano's limited memory bandwidth/8GB shared pool, and the P1 precedent (batch=1 baked in) suggests the P3 detector export may need the same re-export work before batching is even possible | If purely a batching change with no shape/architecture change, no accuracy re-check needed — output should be numerically identical to per-frame inference (worth a parity check, not a re-run) | **Medium-high**: unverified whether current export supports it at all; needs an export-time check before any speedup number can be estimated, so it's a research spike, not a known win. |
| **5** | **Drop ROI appearance (A2)** | Remove the appearance branch entirely | **Near zero** — A2 is 4.4% of total cost (79.6/1825 ms) per both the FLOP accounting and the wall-clock measurement; removing it cannot plausibly clear a ~2x gap | negligible | Head retrain required (token dim changes from 74→10, i.e. back to the non-headline `feat[10]` variant per `build_relation_tokens.py`'s own stale docstring) — and this variant is explicitly **not** the accuracy headline (feat=74, tag `yolo26app_`, is; feat=10/72 are the ablation tags) | **Low payoff for real effort** — don't do this for speed; it was already shown to help the wrong metric (the two reports agree A2 was never the cost driver). |

**Smallest change that plausibly reaches ≥3 dec/s: Option 1 (imgsz 800→512, T=16 fixed) alone**,
per the cost curve already measured in `laneb-p3-edge-cost-and-export.md` §6.1 (projected 3.77
dec/s fp16). It is the only single lever in the table that clears the bar on its own without also
needing a second lever, and it is the one flagged by the existing report as "the cheapest credible
route" for exactly this reason (imgsz doesn't touch the head's graph shape). Its cost is not
implementation effort but **honest accuracy validation** — the 5-seed blind protocol this project
holds itself to is not fast, and there is no shortcut available with the ~1 day realistically left
before Gate 3 (tomorrow) or even before Ch5 needs to be finalized (Day 12-13). Realistically this
is future work, not a Day-9/10 fix — matches the prior review's own conclusion (option B: "only
worth it if it can be measured... risk of consuming most of the 6 days").

## 4. Demo readiness vs Gate 3

Gate 3 (brief §8, end of Day 10 / tomorrow, Sep 15): **"Demo runs end-to-end unattended for 10
min, or the fallback recording is cut and the live demo becomes a recorded demo."** Schedule says
today (Day 9) = "B3 dashboard + live run", tomorrow (Day 10) = "B3 rehearsal + fallback recording".

**Which pipeline should the demo run: P1, unambiguously.** P3 fails its own real-time requirement
by ~2x even on optimistic Jetson projections (§3), has no on-device harness, no TRT engine ever
built for it, and the `TRTRunner`/`MODELS`-list infrastructure that feeds the live
Jetson→dashboard loop is P1-only end to end. Running P3 live at defence is not a live-readiness
gap that can close in one day — it is a different, larger project. This matches the prior
review's own recommendation (option D: present P1 as deployed, P3 as the accuracy study with
deployment infeasible at its current operating point) — nothing found this session changes that
call.

**What exists for "video → Jetson → server → dashboard" (P1):**
- Jetson-side emitter: `jetson-multi-stream-pipeline.py --emit-jsonl` — code exists, syntax
  clean, was run on-device as of 2026-06-03 per JETSON-DEPLOYMENT.md's status box (that date is
  for the benchmark artifacts, not necessarily this exact emit-jsonl flag path).
- Transport: rsync-over-SSH pull, `jetson-live-monitor.py:pull_stream()` — code exists, not
  exercised in this audit (no device access).
- Dashboard: `jetson-live-monitor.py` — full aggregation/alert/render logic present, syntax
  clean, visually complete (ALERT/WARNING/MONITORING tiers, alert log, confidence chart).

**What's missing/unverified for Gate 3, in priority order:**
1. **No fallback recording exists.** `find` across `deployment/` for `.mp4/.mov/.avi` = zero
   hits. `live-demo-stream.jsonl` (8988 lines, dated 2026-06-12) is a stale JSONL event sample,
   not a recorded run — useful for offline dashboard testing, not a defence-day fallback. Per
   Gate 3's own wording, if the live path isn't solid by tomorrow the fallback **recording**
   is the deliverable, and none currently exists.
2. **No end-to-end rehearsal evidence found in the repo.** No artifact/log dated near today
   (Sep 13-14) showing the emit→rsync→dashboard loop was run together; all Jetson artifacts are
   dated Jun 3 or Sep 6-9 (offline benchmarks). "B3 demo pipeline wiring" (scheduled Day 8, Sep
   13) and "B3 dashboard + live run" (scheduled Day 9, today) show no dated artifact confirming
   they landed — this audit cannot confirm or deny completion without Jetson access; flag as
   open.
3. **README is stale**: `deployment/README.md` documents only the standalone `app.py` Streamlit
   demo (file upload, no Jetson), not `jetson-live-monitor.py` (the actual live architecture) —
   a defence-day operator following the README would set up the wrong demo.
4. **`app.py` has no auth** (prior review M5, unresolved) — low risk if `localhost`-only at
   defence, needs one sentence of scoping either in Ch5 or as a runbook note; not a blocker if
   nobody exposes it beyond localhost.
5. **10-minute unattended-run requirement is unverified.** Nothing in the repo (soak logs,
   artifacts) demonstrates a 10-min *dashboard-visible* run specifically — the 30-min soak
   (`soak-jetson-summary.json`) tests raw inference stability, not the full emit→rsync→render
   loop's stability (SSH reconnects, rsync partial-file races, Streamlit long-poll behavior are
   all unexercised by the soak test).

## 5. Ordered Jetson-hands-on steps (user must be notified before each)

None of these were run in this audit (no device access, per rules). Ordered by dependency; each
is a distinct point to check in with the user before executing, since they explicitly want
notice before any on-device step. Commands are the ones already documented in
`JETSON_QUICKSTART.md` / `JETSON-DEPLOYMENT.md` — reproduced here, not re-derived.

| # | Step | Commands | Expected duration | Why it's needed for Gate 3 |
|---|---|---|---|---|
| 1 | **Verify Jetson is reachable and matches the documented state** | `ssh <jetson-host> "uname -a && cat /etc/nv_tegra_release && python3 -c 'import tensorrt; print(tensorrt.__version__)'"` | ~2 min | Sanity check before touching anything — confirms the 2026-06-03 measured state (TensorRT 10.3.0, L4T 36.x) hasn't drifted. |
| 2 | **Confirm P1 engines/artifacts still present and current code is synced** | `ssh <jetson-host> "ls ~/my-project/deployment/export/jetson/*.engine"`; if stale/missing, re-sync: `JETSON_HOST=<host> bash deployment/scripts/jetson-rsync-from-mac.sh` | ~5 min (sync ~500 MB if needed) | The live demo depends on these engines existing and matching the current repo state (post Sep 6-13 changes to scripts). |
| 3 | **Dry-run the emitter alone, short duration, watch for errors** | `ssh <jetson-host> "cd ~/my-project && python3 deployment/scripts/jetson-multi-stream-pipeline.py --streams 1 --source data/test/0_*/0_te1.mp4 --max-frames 300 --fps 30 --mode fp16 --emit-jsonl ~/jetson-stream-test.jsonl"` | ~15-30 s run + a few min setup | First real check that the emit-jsonl path (scheduled but unconfirmed as landed, §4 item 2) actually works before wiring the dashboard to it. |
| 4 | **Pull the test JSONL back and confirm the dashboard renders it** | From Mac: `rsync -az <jetson-host>:~/jetson-stream-test.jsonl /tmp/jetson-stream.jsonl` then `.venv/bin/python3 -m streamlit run deployment/dashboard/jetson-live-monitor.py` and point it at the same host | ~10 min | Validates transport + render before attempting a full multi-stream live session. |
| 5 | **Run the full multi-stream live session end-to-end for ≥10 min unattended** | On Jetson: `python3 deployment/scripts/jetson-multi-stream-pipeline.py --streams 4 --source data/test/0_*/0_te1.mp4 data/test/1_*/1_te1.mp4 data/test/2_*/2_te1.mp4 data/test/3_*/3_te1.mp4 --max-frames 18000 --fps 30 --mode fp16 --emit-jsonl ~/jetson-stream.jsonl` (loop the source clips for 10+ min); on Mac, `jetson-live-monitor.py` running against it | 10-15 min | This **is** the Gate 3 acceptance test ("runs end-to-end unattended for 10 min"). Must be watched for SSH drops, rsync partial-read races, dropped-frame counts. |
| 6 | **If step 5 fails or is flaky: record a fallback video** | Screen-record (e.g. `ffmpeg -f avfoundation ...` on Mac, or phone) a successful run of steps 3-5, or at minimum a clean single-stream pass; save under `deployment/artifacts/demo-fallback/` | 15-30 min incl. re-takes | Currently **zero** fallback recordings exist (§4 item 1) — this is the single highest-priority missing Gate-3 artifact. Should happen regardless of whether step 5 succeeds, per Gate 3's own wording ("or the fallback recording is cut"). |
| 7 | **(Optional, lower priority, do only if steps 1-6 leave spare time before Day 10 ends) Attempt P3 on-device measurement** | Build TRT engine for `p3-detector-fp32.onnx` per `JETSON-DEPLOYMENT.md`'s build-engine pattern (needs `jetson-build-trt-engine.py` extended for P3 — not currently supported per §1); fix `TRTRunner` (§2 bug 1) before attempting the relation head | 1.5-2+ days realistically (per prior review option A) — **not a Day-9/10 task**, flag explicitly if the user wants to attempt it, since it will not finish before Gate 3 | Would replace the projected P3 numbers with a real measurement, but is out of scope for Gate 3 (which only requires the *deployed* pipeline, i.e. P1, to demo). Do not let this compete for time against steps 1-6. |

## Unresolved questions

1. Did "B3 demo pipeline wiring" (Day 8, Sep 13) and "B3 dashboard + live run" (Day 9, today)
   actually execute on the Jetson? No dated artifact in the repo confirms or denies this —
   needs the user to check Jetson-side state (step 1-2 above) before assuming either is done.
2. Is `live-demo-stream.jsonl` (2026-06-12) leftover from an earlier dry run, or was it
   deliberately kept as a dashboard-testing fixture? If the latter, it should be documented as
   such (currently undocumented) rather than looking like a forgotten stale file.
3. Does the Gate 3 "fallback recording" need to be a literal screen-capture video, or would a
   replayable JSONL + a scripted "replay" mode in `jetson-live-monitor.py` (reading from a file
   instead of live rsync) satisfy the intent more cheaply? Not currently implemented either way.
4. Is there Jetson access today/tomorrow (Day 9-10) specifically, distinct from brief.md's
   general "Jetson: full physical access" (D2)? This audit found no evidence one way or the
   other and was explicitly barred from SSH.
5. Same as the 2026-09-13 review's open Q1: does `trt-compat-report.md` need to be re-run
   against the P3 graphs before any Jetson engine-build attempt (step 7), given it currently
   covers 0 of the 4 P3 ONNX files (confirmed this session)?
