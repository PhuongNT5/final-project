# Code review — edge deployment (MT3), 2026-09-13

Scope: deployment/scripts/*, deployment/dashboard/*, deployment/README.md, deployment/JETSON-DEPLOYMENT.md,
deployment/artifacts/benchmarks.md, report/chapter-05-edge-deployment.md, tools/jetson_benchmark.py,
plans/2026-09-06-thesis-2week-final/reports/laneb-p3-edge-cost-and-export.md. No files edited.

## Critical

### C1. Chapter 5 body violates its own attribution rule — presents P1's test-selected 56/64 as "the result" in 5+ places
`report/chapter-05-edge-deployment.md:1-19` sets a hard rule: *"P1's 56/64 is test-selected ... and must
never be presented as the result; its honest blind counterpart is 53/64."* Yet the body breaks this rule
repeatedly: `:36` (§5.1 intro, no blind caveat), `:203-209` (§5.4.2 table + prose, "matches the PyTorch
training baseline exactly"), `:377-386` (§5.6.2 Jetson table, "56/64 top-1, 89.28% macro"), `:700-702`
(§5.8 conclusion, restated as headline finding with zero hedge). Only §5.6.7 (`:321-323`) does it right. A
committee reader who reads the box at the top and then the conclusion will see the rule stated and then
broken — worse than never stating the rule. Fix: append "(test-selected; blind 53/64, see box above)" at
each of the 4 sites, or better, replace the standalone 56/64 with "56/64 test-selected / 53/64 blind" as
the reported pair everywhere in Ch5. ~30 min, no re-measurement needed.

### C2. Ch5's entire polished Jetson deployment story (TRT, multi-stream zero-drop, 36-min soak, power
Pareto, live dashboard, threat model) is P1 — a pipeline explicitly demoted elsewhere in the same document
— while P3, the pipeline the abstract/Ch7 actually leads with, is *not* deployed and, when benchmarked,
misses the real-time budget by ~2x (`plans/2026-09-06-thesis-2week-final/reports/laneb-p3-edge-cost-and-export.md:161-165`,
brief.md S20 measured 2026-09-09). §5.6 reads as "MT3 solved" but MT3 was set for the accuracy headline
pipeline, not P1. `deployment/JETSON-DEPLOYMENT.md:13-16` discloses this honestly in its own status box,
but that caveat is two files away from where a reader/examiner encounters the Ch5 narrative. This is the
single biggest coherence risk in the thesis for the remaining ~6 days. See §4 below for ranked options.

## High

### H1. Detector benchmark times raw ONNX graph only — excludes NMS, crop extraction, zone-feature geometry, decode/resize, H2D copy
`deployment/scripts/benchmark-p3-onnx.py:100-113` (`run_bench`) times three `sess.run()` calls on
synthetic tensors already in the exact target shape. The detector graph is exported NMS-free
(`p3-detector-fp32.onnx` → `(1,300,6)`, confirmed `export-p3-graphs.py:74` uses ultralytics default
ONNX export, no `nms=True`), so post-detection NMS, box→crop extraction (`build_relation_tokens.py`
geometry: `iou_walkway`, `iou_restricted`, `foot_in_walkway`, `norm_dist` per box), and the actual
frame decode/resize to 800² and 224² are all *outside* the timed region. `run_sweep` (`:160-166`)
explicitly documents having found and fixed one such mistiming bug (RNG generation inside the timed
lambda) — good practice — but the fix only removed one source of contamination, it didn't add the
missing pipeline stages. Net effect: the already-failing 0.55 dec/s (Mac) / 1.13-1.71 dec/s (Jetson
projected) figures are *optimistic* lower bounds on true per-decision cost, not the real number. Any
future on-device P3 run should time the full decode→NMS→crop→token→head pipeline, not the three graphs
in isolation. Medium effort to fix (need a real end-to-end script); flag as a stated limitation if not
fixed before deadline — cheap (one sentence) and cheaper than the risk of an examiner asking "does this
include NMS?".

### H2. `MEAN_CROPS=58` (mean, not P95/worst-case) used for the only P3 latency number reported
`deployment/scripts/benchmark-p3-onnx.py:44` — `MEAN_CROPS = 58` with p95 106 noted in the same comment,
cap 128. `run_bench` (`:96-118`) benchmarks appearance/head at the mean crop count. For a real-time safety
system the operating requirement is a worst-case per-decision latency bound (the clip must be scored
before the next window closes regardless of how crowded the scene is), not an average. Appearance is only
4.4% of P3's per-decision cost today (`laneb-p3-edge-cost-and-export.md:148`) so this doesn't change the
FAIL verdict, but if imgsz/T are reduced per §4 below and appearance/detector cost become comparable, using
the mean instead of p95 will understate tail latency. Cheap fix: also report `--crops 106` (p95) and
`--crops 128` (cap) as a one-line addition to the sweep output.

### H3. `TRTRunner` in both `jetson-benchmark.py` and `jetson-multi-stream-pipeline.py` silently
corrupts multi-input models — not just "can't bind", corrupts silently if naively pointed at 2-input model
`deployment/scripts/jetson-benchmark.py:53-65`, `deployment/scripts/jetson-multi-stream-pipeline.py:88-99`.
The constructor loop does `if mode == INPUT: self.host_in, self.dev_in, ... = host, dev, ...` — for a model
with 2 inputs (the P3 relation head: `tokens`, `mask`) this assigns twice and the *second* input silently
overwrites the reference to the first. `set_tensor_address` is called correctly for both tensors during the
loop, so the engine's internal binding table is fine, but `run()` only ever writes/copies into the single
`self.host_in`/`self.dev_in` pair it still holds a reference to (whichever input was enumerated last) — the
other input's device buffer is allocated but never populated, and stays at whatever `cuda.mem_alloc`
happened to zero/leave it as. This is worse than the documented "can't bind" framing (`laneb...md:128`,
`JETSON-DEPLOYMENT.md:198`) suggests: if someone extends `MODELS` to include the relation head without
also extending `TRTRunner` to a dict-of-buffers, it will build, run, and silently return garbage instead of
erroring. Document this explicitly (not just "structural limit") and gate: `TRTRunner.__init__` should
`assert self.engine.num_io_tensors <= 2` (1 in, 1 out) or raise until multi-input support is added, so a
future accidental invocation fails loudly instead of producing silent wrong numbers.

### H4. `aggregate_fps` counts dropped frames as throughput in the multi-stream harness
`deployment/scripts/multi-stream-pipeline.py:151-154` and `jetson-multi-stream-pipeline.py:219-222`: both
compute `total_frames = sum(s.frame_count for s in states.values())` where `frame_count` increments in
`video_reader` on *every* frame read (`:59-68` resp. `:126-135`), including ones immediately dropped at
`:64-65`/`:131-132` because the queue was full — those frames never reach the backbone, never produce a
prediction. `aggregate_fps = total_frames / elapsed` therefore mixes reader I/O rate with inference
throughput. For the headline "zero-drop" configs (Table 5.4, Table 5.5B — drops=0) this is immaterial. It
is *not* immaterial for the number chapter-05 uses to identify "the GPU saturation point": *"the aggregate
caps at 136.89 FPS at N=4 with 328 of 960 frames dropped"* (`chapter-05-edge-deployment.md:437-439`) — that
136.89 FPS figure counts 328 frames that were read from disk and discarded, never inferenced. The true
inference throughput at saturation is `(960-328)/elapsed`, a materially smaller and more honest number for
identifying "the GPU saturation point." Recommend recomputing that one figure from `processed` frames
(frames that reached `inference_loop`) before the deadline — cheap, artifacts already exist as JSON.

### H5. Emit-cadence gate uses raw read-count, not processed-frame count — misaligns window emission under drops
`multi-stream-pipeline.py:101` / `jetson-multi-stream-pipeline.py:153`: `if len(s.buf) == T_WINDOW and
s.frame_count % EMIT_EVERY == 0`. `frame_count` is incremented in the reader task on every frame *read*
(dropped or not); `buf` is only appended to in the inference task on frames that were actually processed.
Under zero drops the two counters stay in lockstep and the headline zero-drop numbers are unaffected. Under
any drop (e.g. the unbounded-throughput sweep, §5.6.3 second paragraph) a dropped frame advances
`frame_count` without advancing `buf`, so the `% EMIT_EVERY == 0` test can fire on a frame whose buffer
content is stale relative to what "every 8th frame" was meant to mean, or skip firing for a whole cycle.
Doesn't crash, doesn't affect the zero-drop headline table, but the "GPU saturation point" sweep (H4) is
exactly the regime where this also applies — worth a one-line footnote in Ch5 or a fix (gate on a
separate processed-frame counter) if time allows. ~20 min fix, low priority given it only touches a
secondary (non-headline) sweep.

## Medium

### M1. `_p1_transfer()` Mac→Jetson ratio calibrated on a memory/overhead-bound MLP pipeline (P1), applied
to a compute-bound 800² conv detector (P3) — direction plausible but unquantified
`deployment/scripts/benchmark-p3-onnx.py:121-141`, `laneb-p3-edge-cost-and-export.md:168-175`. The
transfer factor (2.15x/3.11x/3.25x fp32/fp16/int8) is measured, not assumed — genuinely better practice
than most "assume 3x on embedded" claims in the literature — and the code/docs already flag the direction
of the risk ("P1 is overhead-bound at 0.81 TFLOP/s ... a GPU should favour P3's 800-px conv, so the factor
is if anything conservative"). That's a plausible argument but it is not verified: TensorRT engine-build
tactic selection, kernel fusion, and DLA offload behave differently for a YOLO-family conv backbone at
800² than for P1's small stats-pooling + tiny-Transformer heads, and Ampere tensor-core utilisation curves
are nonlinear in FLOP density — the ratio could just as easily be *worse* for P3 if the export hits
unfused/fallback FP32 layers TRT can't quantize cleanly (`jetson-build-trt-engine.py:138-139` already has
an FP16-fallback path for exactly this reason on P1's heads). Given the "FAIL by ~2x" headline rests on
this factor, the thesis should present it as a *bounded projection*, which it already does in
`laneb-...md:161-167` ("Cite as a projection, never as a Jetson result") — good — but Ch5 doesn't currently
say anything about it at all (C2). If a real Jetson P3 run isn't feasible in 6 days, keep the "projection,
not measurement" framing prominent wherever the FAIL number is cited in the thesis body, not just in the
planning report.

### M2. Two independent, differently-behaved INT8 paths in the same repo, easy to conflate
`deployment/scripts/quantize-int8.py` does ONNX Runtime **dynamic** quantization (`quantize_dynamic`,
weight-only INT8, activations computed in FP32 at runtime) for the Mac/CPU path. `jetson-build-trt-engine.py`
does TensorRT **static** calibrated INT8 (`IInt8EntropyCalibrator2`, `jetson-prep-int8-calib.py`, real
train-split frames/features) for the Jetson path. These are different algorithms with different accuracy
and latency profiles; `deployment/artifacts/accuracy-report.txt` / Ch5 §5.4.2 "INT8 drops 1 sample (55/64)"
is the *dynamic* Mac number, while §5.6.2's "INT8 with 256 real-frame calibration matches FP16 accuracy"
is the *static* Jetson number. Both are correctly labelled in Ch5 as far as I can see (`:400-408`), so this
is not currently a citation bug — flagging as a maintenance/clarity risk: a future edit that moves a number
between sections without re-checking which INT8 it came from would silently introduce an error. Consider a
one-line glossary note in `deployment/README.md` ("ORT dynamic INT8 ≠ TRT calibrated INT8, see X/Y").

### M3. No committed script reproduces `power-sweep-summary.json` / `soak-jetson-summary.json` from raw tegrastats logs
`grep` across `deployment/`, `tools/`, `report/` finds `power-sweep-summary.json` and
`soak-jetson-summary.json` referenced as sources in Ch5 (`:481-482`, `:452-453`) but no committed parser
that turns `tegrastats-pwr{0,1,2}.log` / `tegrastats-soak30min.log` into those summary JSONs.
`jetson-power-sweep.sh` only captures raw tegrastats text and renames the benchmark JSON per profile
(`:37-55`); it does not compute the averages/peaks Table 5.5D/5.5F cite (mJ/inference, FPS/W, GPU
utilisation peak, throttle-event count). Either the summary JSONs were hand-computed once (not
reproducible from the current pipeline — the docs-standards.md "no simulated/manual results" rule cuts
against this) or a parsing script exists but wasn't committed/rsynced back from the Jetson. Given the raw
`.log` files *are* committed (`deployment/artifacts/tegrastats-*.log`), the fix is cheap: write
`deployment/scripts/parse-tegrastats.py` that regenerates the two summary JSONs from the logs already in
the repo, so the numbers are reproducible from a clean clone as `deployment/README.md`'s "seven commands"
claim (Ch5 §5.8) implies they should be.

### M4. `tools/jetson_benchmark.py` PowerSampler regex is fragile and duplicates `deployment/scripts/jetson-benchmark.py`'s purpose
`tools/jetson_benchmark.py:52-58`: matches any whitespace-split token containing both `'mW'` and `'/'`,
then does `tok.split('mW')[0].split('/')[0]`. tegrastats field format differs across JetPack/L4T versions
and Jetson SKUs (e.g. `VDD_IN 6000mW/6000mW` vs fields that report peak/avg in a different order, or SoC
rail names that changed between L4T 35→36). No unit test or sample-log fixture asserts this parses
correctly on the actual JetPack 6 tegrastats format used for this project; if the format doesn't match, it
fails **silently** (`except ValueError: pass`, `:56-58` — a malformed token is just skipped, `samples` stays
short or empty, and `res['watts']` is simply absent with no warning). Also: this is a second,
overlapping "jetson benchmark" script (raw single-detector .engine export+bench) alongside
`deployment/scripts/jetson-benchmark.py` (full P1 pipeline bench) and `jetson-power-sweep.sh` (which calls
the *latter*, not this one) — three scripts with "benchmark" in the name and overlapping-but-different
scope is a discoverability/DRY smell (`development-rules.md` DRY principle). Not urgent to fix given
deadline; note for future cleanup.

### M5. `app.py` Streamlit demo has no auth and displays raw uploaded video frames of workers in-browser
`deployment/dashboard/app.py:105-106,152-153`: `frame_slot.image(...)` renders decoded worker video frames
directly to any browser hitting the Streamlit port; `st.file_uploader` accepts arbitrary uploads with only
extension-based client-side filtering (`type=["mp4","avi","mov"]`, not content-validated). No
authentication, no session isolation beyond Streamlit defaults. Contrast with
`jetson-live-monitor.py:13-16`, which explicitly documents "raw video frames NEVER leave the Jetson" as a
privacy property with evidence (§5.6.6 T1) — `app.py` is the opposite architecture (frames uploaded to and
rendered by the dashboard host) and is not covered by that same claim anywhere in Ch5's threat model table
(`:568-575`). If `app.py` is only ever run `localhost`-bound for live demo purposes this is low risk; if it
is ever exposed on the factory LAN (even unintentionally, e.g. `streamlit run --server.address 0.0.0.0`)
it becomes a direct worker-privacy exposure with zero access control. Recommend one sentence in Ch5
Limitations or the security section scoping `app.py` to "demo only, localhost, not part of the T1-T5
threat model" so the threat model isn't read as covering it.

## Low

### L1. `jetson-benchmark.py --warmup 10` default may be short for TRT/GPU clock ramp-up
`deployment/scripts/jetson-benchmark.py:161`. `jetson-power-sweep.sh:44` explicitly passes `--warmup 10`
too. 10 iterations is on the low side for letting DVFS clocks settle after `jetson_clocks`/`nvpmodel`
switches, though `jetson_clocks` is called before the sweep (`:32`) which locks clocks and mitigates most
of the concern. Timing itself is sound: `TRTRunner.run()` calls `stream.synchronize()` before returning
(`:72`, `:106` in the multi-stream variant), so each `time.perf_counter()` delta genuinely waits for the
GPU to finish — no premature-return/async-timing bug here, which is worth calling out as a **positive**
(this is a common mistake this codebase avoided).

### L2. `benchmark-p3-onnx.py` warmup default (3) is fine for CPU EP but would be too low if ever pointed
at a CUDA/TRT EP later; not a bug today (Mac CPU only), flag only if `--ep` grows a GPU option.

## Positive observations

- INT8 calibration data for the Jetson TRT path is explicitly drawn from `all8_train.pt` / `data/train`
  (`jetson-prep-int8-calib.py:20-27,74-75`), never from the 64 test/blind clips — no calibration leakage
  found. The script even explicitly bumped N from 32→256 after observing an accuracy drop, with the
  rationale left in a comment (`:28`) — good engineering hygiene.
- `benchmark-p3-onnx.py` refuses to report latency unless the parity gate passes (`:235-240`) — a real
  guard against exactly the kind of "graph doesn't match the model it claims to represent" bug that would
  otherwise be easy to ship silently.
- Two prior measurement bugs (thread-pool oversubscription from duplicate ORT sessions; RNG generation
  timed inside the benchmarked lambda) were found and fixed with the fix documented in the docstring
  (`benchmark-p3-onnx.py:150-166`) rather than silently corrected — good for reproducibility/audit trail.
- `jetson-live-monitor.py` privacy-by-construction design (metadata-only over the wire, SSH transport) is
  a genuine, well-evidenced architectural contribution and matches what Ch5 §5.6.6 claims.
- Ch5's pipeline-attribution box (`:1-19`) is exactly the right instinct — the problem is body-text
  compliance with its own rule (C1), not the rule itself.
- `JETSON-DEPLOYMENT.md`'s status box is honest and up to date about what's measured vs projected for P3;
  better disclosure than most of what ends up in the thesis body.

## 3. Is the Mac→Jetson projection method sound for P3?

Sound as a *documented, bounded projection* — not sound as a *feasibility verdict*. It is calibrated
(measured on P1 on both machines, not assumed), the direction-of-bias argument is stated and plausible, and
the code refuses to call it more than a projection (`benchmark-p3-onnx.py:128`: "Never present a transferred
number as measured"). But P1 and P3 stress the hardware differently enough (small-matmul/overhead-bound MLP
heads vs a large 800² conv backbone) that a single scalar ratio derived from one architecture applied to a
structurally different one carries real uncertainty in *both* directions — it is not simply "conservative"
as claimed, that's an untested assumption (M1). Given the ~2x miss margin (0.55 Mac dec/s vs 3.0 needed,
even optimistic Jetson projections only reach 1.71), the qualitative FAIL conclusion is probably robust to
this uncertainty — a ratio would need to be off by >2x in the favourable direction to flip the verdict,
which is a lot even granting GPUs favour large convs. Treat the FAIL as a safe qualitative conclusion, the
exact "1.64 dec/s" number as illustrative only.

## 4. Closing the P1-deployed / P3-headline gap — options, effort, recommendation (6 days to 2026-09-19)

| # | Option | Effort | Risk | Outcome |
|---|---|---|---|---|
| A | Fix `TRTRunner` for 2-input models (dict of host/dev buffers keyed by tensor name), run P3 detector+appearance+head engines on the actual Jetson, get one real measured P3 number | 1.5-2 days (engine build + INT8 calib rerun + on-device script) + hardware access | Medium — TRT 800² detector build/calibration on Orin Nano could be slow or hit workspace-memory limits; opset 18 parsing already validated in `trt-compat-report.md`? verify it covers p3 graphs, not just P1's | Best outcome: a real number replaces a projection. Recommended if Jetson access is available in the window. |
| B | Reduce cost: imgsz 800→512 at T=16 (cost curve already shows 3.77 Mac dec/s fp16-projected, `laneb...md:184`), rebuild tokens + retrain relation head, re-measure accuracy | 1-2 days (token rebuild + retrain is fast per brief's existing pipeline; re-run 5-seed blind protocol to get an honest accuracy number, not just point-estimate) | Medium — accuracy at 512² is *unmeasured*; could drop below the 84.4% P3 headline enough to undercut the "accuracy headline" framing, requiring a second iteration | Only worth it if it can be measured, not just claimed; risk of consuming most of the 6 days if accuracy needs multiple iso-cost variants tried |
| C | Detector every-N-frames + reuse tokens across sliding windows (T=16, stride 8 → 8 of 16 detections already computed for the next window) | 0.5 day design + 1 day implementation/validation | Low-medium — halves detector cost roughly 2x (8 fresh + 8 reused frames per decision instead of 16 fresh), which alone doesn't clear ~2x-3x miss without also reducing imgsz; token staleness (reused detections from t-8 may miss entities that appeared mid-window) needs at least a qualitative accuracy sanity check on the 64 test clips before claiming it's free | Directly answers the user's amortisation question — yes, cost amortises, since `EMIT_EVERY=8 == T/2` already means consecutive decisions overlap 8 of 16 frames; this is close to free-lunch cost reduction but needs the actual overlap logic implemented (nothing in the current P3 export/token pipeline caches detections across windows) and a check that stale half-window detections don't silently degrade recall on WalkwayViol/UnauthorInt (the two classes already shown to be seed/cluster-fragile in brief S19-S20) |
| D | Present P1 explicitly as "the deployed system" and P3 explicitly as "the accuracy study, deployment infeasible at its current operating point, cost curve characterised" — no further engineering, just tighten the prose (fix C1, expand C2's one-line caveat into a proper paragraph/subsection in Ch5 with the cost-curve table from Lane B pulled in) | 0.5-1 day (writing only) | Low | Safest option given 6 days; turns "gap" into an explicit, defensible thesis finding ("P3 buys its accuracy at an edge cost P1 does not pay" — already the recommended framing in `laneb...md:204-206`) rather than an unaddressed inconsistency |

**Recommendation.** Do D unconditionally regardless of what else is attempted — it's cheap, low-risk, and
fixes C1/C2 which are real internal-consistency defects today. Then, if Jetson hardware access and time
allow, attempt A (real measurement beats any projection for the thesis's own credibility) but timebox it to
2 days and fall back to D-only if it slips. C is the most intellectually interesting option (the
`EMIT_EVERY=8, T=16` sliding window genuinely does overlap 50% frame-to-frame, so token reuse is a real,
not hypothetical, amortisation opportunity) but implementing + validating it safely against the two
fragile classes is real engineering risk for 6 days; treat it as future work rather than something to land
before 2026-09-19. B is the lowest expected value of the four given accuracy is fully unmeasured and the
5-seed blind protocol this project holds itself to (brief S19/S20) is not a quick re-run.

## Unresolved questions

1. Does `trt-compat-report.md` (M4 pre-check) cover the P3 graphs (`p3-detector-fp32.onnx` etc.), or only
   the P1 backbone/heads? If only P1, opset-18 TRT-10 parseability of the P3 detector is unverified before
   any Jetson attempt (option A above).
2. Is there Jetson hardware access in the remaining 6 days, and for how long? This gates whether option A
   is realistic at all.
3. Where do `power-sweep-summary.json` / `soak-jetson-summary.json` actually come from (M3)? If a parser
   script exists but wasn't committed, it should be added for reproducibility before the deadline; if the
   numbers were computed by hand from the logs, that should be disclosed since `docs/code-standards.md`
   and `development-rules.md` both prohibit non-reproduced/manual results presented as pipeline output.
4. Does `app.py` (M5) ever get run other than `localhost` for the live thesis demo? If it's demo-only and
   local, M5 is paperwork (one sentence in Limitations); if it's meant to be shown to a committee over a
   shared network, it needs at minimum a basic auth gate before that happens.
