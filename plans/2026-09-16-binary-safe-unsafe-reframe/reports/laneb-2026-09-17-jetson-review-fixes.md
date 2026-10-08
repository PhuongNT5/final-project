# Lane B — Jetson Session Prep: Code-Review Fixes (2026-09-17)

Closes all 6 findings in `plans/2026-09-06-thesis-2week-final/reports/code-reviewer-2026-09-16-jetson-session-prep-review.md`
before the fixed, non-repeatable 2026-09-19 morning device session. File
ownership this session: `deployment/` + `tests/test_jetson_*.py` only. No commit made (per instruction).

## Finding -> Fix

### 1. `run_memory` dry-run dead key (Medium)
File: `deployment/scripts/jetson-measure-power.py`, `run_memory()`.
Was: dry-run branch set `device_free_total_mb` — a key never produced on the
real (success) path, so skipped JSON had the real 3 keys entirely absent.
Fix: dry-run branch now sets `device_free_mb`/`device_total_mb`/`device_used_mb`
= `None`, matching the exact success-path schema. Verified live:
`deployment/artifacts/jetson-measurements/memory-fp16-20260917T223311.json`
shows all 3 real keys null, `device_free_total_mb` absent.

### 2. Runbook JSON path + full audit (Medium)
File: `deployment/JETSON-RUNBOOK-2026-09-19.md`.
Root bug: §8 (thermal soak triage) said cross-check
`windows[i].tegrastats.samples[*].temp_c`. Actual schema
(`run_thermal_soak()` in `jetson-measure-power.py`): `result['tegrastats']`
is TOP-LEVEL (`= sampler.stop()` output, not nested per-window),
`result['windows']` = list of `{t_offset_sec, latency_ms}` only — no
`tegrastats` key inside a window. `parse_tegrastats_line()` in
`jetson-measure-utils.py` produces `temps_c` (plural dict keyed by sensor
name, e.g. `{cpu: .., gpu: ..}`), never scalar `temp_c`. Fixed the runbook
line to `tegrastats.samples[*].temps_c` (top-level) with the dict-vs-scalar
and nesting corrections spelled out inline.

Also found + fixed while auditing:
- §5 `--parity` command passed `--mode fp16`, but `run_parity()` has no
  `mode` param at all — it auto-discovers and compares EVERY precision per
  stage against the fp32 reference in one pass. The flag was a silent no-op.
  Removed it from the command, added an explanatory note.
- §9 memory "Good:" line said `device used_mb/total_mb` (informal) — made
  exact: `device_used_mb`, `device_total_mb`, `device_free_mb`.
- §0 pre-flight file-transfer list was missing the new
  `stream-pipeline-utils.py` (Task 4's new shared module —
  `jetson-multi-stream-pipeline.py` now imports it at load time; the
  `rsync -avz deployment/scripts/` command already transfers the whole dir
  so this wasn't a functional gap, but the explicit list was stale). Added.
- §6 `--streams` command updated to show the new `--class-labels` flag
  (Task 3) with an explanation of the fallback (`class_N`) behavior.

**Full audit checklist (every command/flag/path checked against code):**
| Runbook ref | Item | Checked against | Result |
|---|---|---|---|
| §0 | `jetson-full-measurement.py --dry-run --all` | live execution, this session | matches, ran clean |
| §1/§2 | `jetson-build-trt-engine.py --model NAME [--onnx PATH] --mode` | code read (pycuda blocks `--help` on Mac — see below) | matches |
| §2 | fallback `trtexec --onnx= --saveEngine= --fp16` | trtexec is a standard TensorRT CLI, not project code — not executable-verified on Mac (no device) | syntax standard, left as-is |
| §3 | `--model-size` | `--help` live run | matches |
| §4 | `--latency --mode --warmup --runs` | `--help` live run | matches |
| §4 | `end_to_end_naive_sum_ms_p50` caveat wording | `run_latency()` code read | matches (correctly labeled a naive proxy) |
| §5 | `--parity --rule --threshold` | `run_parity()` code read + `--help` | `--mode` was a no-op, fixed (see above) |
| §6 | `--streams --sources --max-frames --fps --class-labels` | `--help` live run + `run_streams()`/`jetson-multi-stream-pipeline.py` code read | matches after Task 3's `--class-labels` addition |
| §6 | unquoted glob explanation | `argparse nargs='*'` behavior, shell expansion | correct |
| §7 | `--energy STAGE --mode --energy-sec` | `--help` live run | matches |
| §8 | `--thermal-soak STAGE --mode --soak-min --soak-window-sec` | `--help` live run | matches |
| §8 | JSON path for temps | `run_thermal_soak()` + `parse_tegrastats_line()` code read | WAS WRONG, fixed (root finding) |
| §9 | `--memory --mode` | `--help` live run | matches |
| §9 | output keys | `run_memory()` code read (post-fix) | now exact |
| §10 | `--power-state` | `--help` live run | matches |
| §12-B | calib filename `<model>-batch-{N}.npy` | `make_calibrator()` code read | matches |
| §12-D | discovery regex `^(.*)-(fp32\|fp16\|int8)\.engine$` | `discover_engines()` code read | matches exactly |
| §12-A | pycuda/tensorrt import order | code comment in `jetson-build-trt-engine.py` | matches |

**Execution constraint documented in the runbook and here**: `jetson-build-trt-engine.py`
and `jetson-multi-stream-pipeline.py` both have unconditional top-level
`import pycuda.autoinit`, which raises `ModuleNotFoundError` on the Mac (no
pycuda installed) — confirmed by direct execution attempt. Their CLI syntax
was verified by code reading only, not `--help` execution. Every other
script (`jetson-full-measurement.py`, `jetson-measure-latency.py`,
`jetson-measure-power.py`, `jetson-measure-utils.py`) lazy-imports
TRT/pycuda inside functions, so `--help` and `--dry-run` both run clean on
Mac and were exercised live.

### 3. `CLASS_LABELS` hardcode removed (Low)
Files: `deployment/scripts/jetson-multi-stream-pipeline.py`,
`deployment/scripts/jetson-full-measurement.py`.
Removed the hardcoded 4-class `CLASS_LABELS` list entirely. Added optional
`--class-labels` CLI arg (comma-separated, no default) threaded through:
`main()` -> `run_config()` -> `inference_loop()` in the pipeline file, and
`main()` -> `run_streams()` -> `run_config()` (kwarg) in the orchestrator.
No default value chosen deliberately (YAGNI) — omitting it falls back to
`class_N` labels in the JSONL `label` field, which still works and doesn't
re-hardcode a class-count assumption while the binary head isn't landed yet.

### 4. DRY dedup of `StreamState`/`aggregate_fps`/`video_reader` (Low)
New file: `deployment/scripts/stream-pipeline-utils.py` (104 lines) —
importlib-loaded (hyphenated filename), same pattern as `trt-runner-utils.py`.
Holds `StreamState`, `video_reader` (takes `preprocess_fn` as a param instead
of hardcoding per-platform preprocess), `aggregate_fps`, plus `softmax`/
`stats_agg`/`percentiles` (extracted beyond what the review literally named,
necessary to hit the <200-line budget — see below).
- `jetson-multi-stream-pipeline.py`: 267 -> 199 lines.
- `multi-stream-pipeline.py`: 207 -> 164 lines.
Both files keep their own `preprocess()` (the one genuine per-platform
difference: PIL-bilinear via `pipeline-utils.py` on Mac vs cv2-only on
Jetson), passed into `video_reader()` as `preprocess_fn`.
Judgment call: standardized `percentiles()` on the jetson file's rounded/
string-key format in the shared module — grepped `tests/*.py` first and
confirmed neither original local `percentiles()` was under test (only
`jetson-measure-utils.py`'s separate `percentiles()` is), so this was safe.

### 5. Dry-run regression test (Medium)
New file: `tests/test_jetson_measure_dry_run_guards.py` (126 lines, 8 tests).
Covers every CLI-facing hardware-dependent function named in the review:
`run_latency`, `run_parity`, `run_energy` (both no-engine and
engine-present-dry-run paths), `run_thermal_soak`, `run_memory` (explicit
regression assertion that `device_free_total_mb` is absent), `run_streams`
(both `dry_run=True` and `dry_run=False`-with-no-`--sources` paths — the
latter also regresses against attempting a pycuda-dependent import on a
non-Jetson host). All 8 pass.

### 6. Runbook header + 18/9 checklist + phase-04 placeholder
File: `deployment/JETSON-RUNBOOK-2026-09-19.md`.
- Header now says "2026-09-19 morning" + explicit "fixed, non-repeatable...
  no second session" framing.
- New "18/9 Mac dry-run checklist" section: full `--dry-run --all` command +
  separate dry-run command for `--streams`/`--energy`/`--thermal-soak`
  (verified live, both work), expected output filename pattern (corrected
  from an initial draft mistake — real pattern is `{name}-{timestamp}.json`,
  no `DRYRUN` marker in the filename itself), what to inspect (every
  hardware field null + non-empty `skipped_reason`), points at the new
  regression test, points at full test suite run, checksum-vs-MANIFEST
  check, `data/test` transfer-size check.
- New "18/9 prerequisite — binary relation head export (phase-04,
  placeholder)" section, explicit checkbox list of exact phase-04 outputs
  needed (`p3-relation-head-bin-fp32.onnx`, `MANIFEST.md` w/ checksums +
  graph hash + seed, G4 parity confirmation, phase-04's own rehearsal step)
  with a documented fallback (old 4-class graph + `--rule argmax`) if any
  box is unchecked by session morning.
- Binary head status note added near the top explaining
  `p3-relation-head-fp32.onnx` is currently still the OLD 4-class graph.

### 7. Full test run + fresh dry-run (actual output, not fabricated)
```
tests/test_jetson_measure_dry_run_guards.py  8/8 PASS
tests/test_jetson_measure_utils.py           6/6 PASS
tests/test_jetson_multi_stream_import.py     2/2 PASS
tests/test_trt_runner_utils.py               3/3 PASS
tests/test_stream_fps_accounting.py          2/2 PASS
tests/test_benchmark_p3_onnx.py              4/4 PASS
```
25/25 tests pass, zero failures, zero skips.

Fresh `jetson-full-measurement.py --dry-run --all` (this session, 2026-09-17
22:33:11 local): 4 files saved (`model-size`, `latency-fp16`, `memory-fp16`,
`parity`) — `--all` does not cover `--streams`/`--energy`/`--thermal-soak`
(need explicit args), separately dry-run verified working (see §6 above,
also verified live: 3 more files saved). Spot-checked `memory-fp16-*.json`
and `latency-fp16-*.json` — every hardware number is `null` with a non-empty
`skipped_reason`, zero fabricated numbers, `device_state.platform` correctly
shows macOS (proves dry-run plumbing is genuinely exercised, not mocked).

## Files Modified
- `deployment/scripts/jetson-measure-power.py` — `run_memory()` dead-key fix (Task 1)
- `deployment/scripts/jetson-multi-stream-pipeline.py` — CLASS_LABELS removal + DRY dedup, 267->199 lines (Tasks 3, 4)
- `deployment/scripts/multi-stream-pipeline.py` — DRY dedup, 207->164 lines (Task 4)
- `deployment/scripts/jetson-full-measurement.py` — `--class-labels` threading (Task 3)
- `deployment/JETSON-RUNBOOK-2026-09-19.md` — full audit fixes + 18/9 sections (Tasks 2, 6)

## Files Created
- `deployment/scripts/stream-pipeline-utils.py` (104 lines, Task 4)
- `tests/test_jetson_measure_dry_run_guards.py` (126 lines, 8 tests, Task 5)

## Unresolved questions
None outstanding — both unresolved questions from the source review (dedup
timing, memory-fix ownership) are resolved by this session landing both in
scope before 9/19.
