# Finding — measurement-code audit and cleanup (2026-09-29)

Scope: whole-repo review + cleanup requested after the Jetson latency session.
Written in English to match the sibling `finding-2026-09-*.md` files; the thesis
text itself stays Vietnamese per the binding decision.

## 1. Headline

Four measurement defects found in `deployment/scripts/jetson-measure-power.py`.
**None of them contaminated a number**: at the time of the audit no energy,
memory, or thermal-soak artifact existed in
`deployment/artifacts/jetson-measurements/` (14 files, all latency/parity/
model-size/streams). All four are fixed before the first energy run, not after.

This is the same defect class as the two caught on 2026-09-29 earlier
(detector parity via `argmax` on a `(300,6)` box table; decode measured as
seek): a harness that returns a plausible number for a quantity it is not
actually measuring.

## 2. Defects and fixes

**D1 — the clock and the power sampler started before engine deserialisation.**
`sampler.start()` and `t_end = time.time() + duration_sec` both ran *before*
`TRTRunner(...)` was constructed. Engine deserialisation on Orin Nano takes
seconds; that dead time (n=0, low draw) sat inside both the energy integral and
the denominator. Effect: `fps` and `avg_power_w` deflated, `joules_per_inference`
inflated, by the deserialise fraction of the window (plausibly 5-17% at
`--energy-sec 60`, worse at 30).
Fix: `_build()` deserialises + runs `ENERGY_WARMUP=20` warm-up inferences before
any clock starts, and the result divides by **measured** elapsed
(`duration_sec_measured`), not the requested `--energy-sec`.

**D2 — no idle baseline; the project spec requires one.**
`research/researcher-03-demo-and-edge-deployment.md` §5-6 defines
`J/inference = (P_avg_W - P_idle_W) x t` and
`fps_per_watt = fps / (P_avg_W - P_idle_W)`. The implementation used raw
`P_avg` with no idle subtraction, so the code contradicted the project's own
spec. `power_now_mw` is parsed from `VDD_IN` — total module input, including
CPU/RAM/IO idle draw, not a GPU rail.
Fix: `--idle-sec` (default 10) measures `P_idle` in the same run, with the
engine already resident and no work submitted. Both framings are reported and
explicitly named: `*_total_board` (what the box draws) and `*_delta_idle` (the
spec's marginal cost). The ambiguous unlabelled `joules_per_inference` /
`fps_per_watt` keys are gone, so neither can be quoted without a framing.
A baseline at or above the load draw is **not** clamped to a positive number —
it emits `delta_idle_skipped_reason` and nulls the delta figures.

**D3 — `run_memory` loaded no engine at all.**
It took `engine_dir` and `mode`, used neither, and reported `cuda.mem_get_info()`
from a process that had deserialised nothing. On Orin Nano memory is unified, so
`device_used_mb` was whole-system occupancy (OS included) tagged `{"mode":"fp16"}`
— it reads as a model footprint and is not one.
Fix: loads every engine discovered for `mode`, holds them resident, and reports
`engines_resident_delta_mb` (free-before minus free-after) plus `engines_loaded`.
`device_used_mb` is kept but carries `device_used_note` stating it is system-wide.

**D4 — thermal-soak baseline was the cold first window.**
`possible_throttling` compared the last window's p50 against **window 0**, which
includes clock ramp. A slow window 0 makes drift look negative and hides real
throttling.
Fix: baseline is the fastest window (`baseline_p50_ms`,
`baseline_window_index`); `SOAK_WARMUP=20` warms the engine before window 0; both
`latency_drift_pct_vs_fastest` (used by the flag) and
`latency_drift_pct_vs_first` are reported.

Unit note: `n_inferences` counts invocations of ONE engine. A P3 decision is 16
detector passes + 1 batched appearance pass + 1 head pass, so per-decision energy
must be composed from per-stage runs the way per-decision latency is. The script
and the runbook both say this explicitly now; no per-decision energy figure is
synthesised from a single stage.

## 3. Test suite

`pytest` was not installed in any interpreter on this machine — the project rule
"run tests before push" was unenforceable. Installed into `.venv`; the suite then
ran clean:

| suite | result |
|---|---|
| `tests/` (deployment, TRT, streams, gates) | 73 passed |
| `eval_utils/tests/` (statistics, calibration, bootstrap, conformal) | 77 passed |
| total | **150 passed, 0 failed** |

`tests/test_jetson_measure_dry_run_guards.py` **caught the D2 rename** on the
first run (`KeyError: 'joules_per_inference'`). The test was updated to the new
contract, not weakened: both energy guards now iterate
`POW._ENERGY_KEYS` — the implementation's own tuple — so a future schema rename
cannot outrun the guard again, and an explicit assertion forbids the ambiguous
unlabelled key from returning.

That 150-test figure is quotable in the defence as evidence the evaluation
statistics are covered by tests, which is otherwise hard to demonstrate.

## 4. Files changed

- `deployment/scripts/jetson-measure-power.py` — rewritten (126 -> 216 lines).
  Over the 200-line guideline; the excess is measurement-semantics docstrings,
  which is the point of the file. Backup of the pre-audit version kept in the
  session scratchpad.
- `deployment/scripts/jetson-full-measurement.py` — `--idle-sec` flag, threaded
  through to `run_energy`.
- `deployment/JETSON-RUNBOOK-2026-09-19.md` — §7/§8/§9 acceptance criteria
  rewritten for the new key names, the two energy framings, the per-invocation
  caveat, and the footprint-vs-system-memory distinction.
- `tests/test_jetson_measure_dry_run_guards.py` — guards bound to `_ENERGY_KEYS`.

## 5. Cleanup executed

Removed: 315 `__pycache__` dirs, 4 `.DS_Store`, 6 empty `runs/detect/val*` dirs,
and the root `best.pt`. **65 MB freed.**

Root `best.pt` (md5 `d8a3cbbdd75f6ee0478745ada56003b7`) was byte-identical to
`sweep_results/entity_detector/detector_best.pt` — the path
`build_relation_tokens.py:32` actually uses — and to
`sweep_results/entity_detector/best.pt`. Two identical copies remain.
Restore if ever needed:
`cp sweep_results/entity_detector/detector_best.pt best.pt`

Not touched, deliberately: 20 tracked `.py` files over the 200-line rule
(largest `run_finetune_bbox.py` 781, `run_yoloworld_push.py` 579,
`run_pairwise_contrastive.py` 558). These produced thesis numbers; refactoring
them before the 2026-10-05 freeze buys nothing and risks everything.

Device-side cleanup (`/tmp/build-p3.sh`, `/tmp/build-p3-fp32.sh`,
`~/build-*.log`) remains deferred while the user's parallel process runs.

## 6. Consequence for the run plan

The energy and soak runs in runbook §7/§8 must be executed with the **fixed**
script. The fixed script must be rsynced to the Jetson before those steps —
the device currently holds the pre-audit version. No energy artifact produced by
the old script would be valid; none exists.

## Unresolved

1. Which energy framing does the thesis quote — `*_total_board` or
   `*_delta_idle`? Both are measured; the text must name one and define it.
   Recommend total-board for the deployment claim (it is what the box draws) with
   delta-idle reported alongside as the marginal cost.
2. `--energy-sec 30` in the runbook vs the script's default 60. With warm-up plus
   a 10 s idle baseline now preceding the window, 30 s of load is thin for a
   stable VDD_IN mean at 500 ms sampling (60 samples). Recommend 60 s.
3. Per-decision energy needs per-stage runs for detector, appearance, and head —
   §7 currently says "repeat swapping `backbone`" without naming the P3 stages.
4. R32b still open: every 2026-09-29 measurement ran at MAXN_SUPER, not the
   `nvpmodel -m 0` 15 W envelope R32 specifies.
