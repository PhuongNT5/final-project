# Code Review — Jetson Fixes Verification (2026-09-17)

Scope: files listed in task (stream-pipeline-utils.py, jetson-multi-stream-pipeline.py,
multi-stream-pipeline.py, jetson-measure-{power,latency,utils}.py, jetson-full-measurement.py,
trt-runner-utils.py, benchmark-p3-*.py, JETSON-RUNBOOK-2026-09-19.md, tests/test_jetson_*.py,
test_stream_fps_accounting.py, test_benchmark_p3_onnx.py, test_trt_runner_utils.py). Read-only —
no edits made. Verified against laneb-2026-09-17-jetson-review-fixes.md's claims, not just
trusting the report.

## Verification performed
- Read every file in scope in full; diffed against the prior review's findings one by one.
- Ran all 6 test files (25 tests) directly via `.venv/bin/python3 tests/test_*.py` — 25/25 pass,
  matches the implementer's report exactly.
- `py_compile` on all `deployment/scripts/*.py` — clean.
- Live-ran every runbook command with `--dry-run` (streams/parity/energy/thermal-soak/memory) and
  confirmed output matches documented schema/paths; cleaned up generated artifacts after.
- Tested `multi-stream-pipeline.py`'s importlib load from `/tmp` (non-repo CWD) to confirm
  `stream-pipeline-utils.py` resolves via `__file__`-derived `ROOT`, not CWD.
- Inspected actual ONNX graphs (`onnx.load`) to check dynamic-shape assumptions in
  `trt-runner-utils.py` against both P1 (already measured on-device 2026-06-03) and the new,
  never-run-on-device P3 relation head.
- `grep` for leftover `CLASS_LABELS` hardcodes, gitignore conflicts for the new files.

## 1-6: Prior findings — all 6 genuinely fixed

1. **`run_memory` dead key** — fixed. Dry-run branch now sets
   `device_free_mb`/`device_total_mb`/`device_used_mb` = `None` (jetson-measure-power.py:119),
   matching the real success-path schema exactly. Regression-tested
   (`test_run_memory_dry_run_nulls_the_real_schema_not_a_dead_key`), and I independently confirmed
   with a live dry-run JSON.
2. **Runbook `tegrastats` path** — fixed. §8 now correctly says `tegrastats.samples[*].temps_c`
   is TOP-LEVEL (not nested per-window) and plural/dict (not scalar `temp_c`) — matches
   `run_thermal_soak`'s actual `result['tegrastats'] = sampler.stop()` and
   `parse_tegrastats_line()`'s `temps_c` dict exactly. Also caught and fixed a real bug while
   auditing: §5's `--mode fp16` flag on `--parity` was a silent no-op (`run_parity()` has no
   `mode` param) — now removed with an explanatory note. Verified by reading `run_parity()`.
3. **`CLASS_LABELS` hardcode** — fully removed from both `jetson-multi-stream-pipeline.py` and
   `jetson-full-measurement.py` (grep confirms zero remaining references); replaced with an
   optional `--class-labels` flag threaded `main() -> run_config() -> inference_loop()` and
   `main() -> run_streams() -> run_config()`. Fallback to `class_N` when omitted, verified in code
   and via `--help`.
4. **DRY dedup** — `stream-pipeline-utils.py` (104 lines) now holds `StreamState`, `video_reader`
   (takes `preprocess_fn` as a param — genuinely eliminates the per-platform duplication),
   `aggregate_fps`, `softmax`, `stats_agg`, `percentiles`. Both pipeline files import it via the
   same `_load()`-importlib pattern as `trt-runner-utils.py`. Confirmed byte-for-byte that both
   `jetson-multi-stream-pipeline.py` (199 lines) and `multi-stream-pipeline.py` (164 lines) are
   now under the 200-line guideline, matching the claimed 267->199 / 207->164.
5. **Dry-run regression tests** — `tests/test_jetson_measure_dry_run_guards.py` (8 tests) covers
   `run_latency`, `run_parity`, `run_energy` (both no-engine and engine-present-dry-run), `run_
   thermal_soak`, `run_memory`, and `run_streams` (both `dry_run=True` and the
   `dry_run=False`-no-`--sources` edge case flagged in the prior review). All 8 pass; I re-ran
   independently.
6. **Runbook 18/9 sections** — present and accurate: dry-run checklist, phase-04 binary-head
   placeholder with explicit fallback, pre-flight file-transfer list now includes
   `stream-pipeline-utils.py`.

## Regression check: DRY refactor / importlib from any CWD (Priority 2)

No regression. `ROOT` in both pipeline files is computed from `os.path.abspath(__file__)`
(triple-`dirname`), not CWD — confirmed by loading `multi-stream-pipeline.py` via importlib from
`/tmp` and getting a correct absolute `ROOT`. `stream-pipeline-utils.py` has no
tensorrt/pycuda/onnxruntime import at module level, so it loads cleanly standalone. `aggregate_fps`
semantics are unchanged: both callers still compute `total_processed / elapsed` (not
`frame_count`), verified against `test_stream_fps_accounting.py`'s real (not reimplemented)
`aggregate_fps()` import — passes, 15.2 fps (not the old buggy 20.0) still asserted correctly.

## Priority 3: silent/fabricated numbers on device — no new risk found

- All `run_*` orchestration functions null every hardware field before any real computation on
  `dry_run`/missing-engine paths — traced by hand, confirmed by the new regression tests.
- `trt-runner-utils.py` uses `set_tensor_address` + `execute_async_v3(stream.handle)`, which is the
  correct/required TRT 10.x API (device confirmed TensorRT 10.3.0 per JETSON-DEPLOYMENT.md and the
  runbook header — no TRT 8.6 code path exists in this codebase, so no version-branching risk).
- One thing worth flagging (not a regression, but genuinely new exposure): `TRTRunner.__init__`
  calls `context.get_tensor_shape(name)` without ever calling `context.set_input_shape(name, ...)`.
  I confirmed via `onnx.load()` that **both** P1 inputs (`image`, `features` — dynamic on `batch`)
  **and** the new P3 relation head's two inputs (`tokens`, `mask` — both dynamic on `batch`) use
  TensorRT optimization profiles with min=opt=max=1. This exact pattern already ran successfully
  on-device for P1 on 2026-06-03 (real measured numbers exist in JETSON-DEPLOYMENT.md), so it's
  very likely fine for the P3 relation head too — same shape-binding mechanism, same batch=1
  profile shape. But P3's 2-input engine has never actually been built/run on the Jetson before
  (per the runbook's own "Read this first" callout), so this is an *unverified-for-P3-specifically*
  assumption riding on P1 precedent. If it fails, failure mode is a loud crash at `TRTRunner`
  construction (`np.empty()` on a `-1` shape raises immediately) — not a silent/fabricated number,
  so it's a should-fix / watch-item, not a blocker for the "no fabrication" contract.
- `run_streams`'s lazy import of `jetson-multi-stream-pipeline.py` (which top-level `import
  pycuda.autoinit`) only triggers when `not dry_run and source_globs` — confirmed via the new
  `test_run_streams_no_sources_has_skipped_reason_even_without_dry_run` test, which explicitly
  guards the edge case the prior review flagged (no-`--sources`, no-`--dry-run`, non-Jetson host
  would otherwise crash on import). Fixed.

## Priority 4: runbook commands vs actual CLI — all verified live

Ran every `--dry-run` variant of the runbook's commands directly (`--streams --sources
--class-labels`, `--parity --rule argmax`, `--energy backbone --energy-sec 30`, `--thermal-soak
backbone --soak-min 30 --soak-window-sec 60`, `--memory --mode fp16`) — all produced valid JSON at
the documented `deployment/artifacts/jetson-measurements/{name}-{timestamp}.json` path with no
argparse errors. `--help` output matches every flag referenced in the runbook exactly (`--model
--onnx` on `jetson-build-trt-engine.py`, `--rule/--threshold` on `--parity`, `--class-labels` on
`--streams`). `nargs='*'` on both `--streams` and `--sources` does not misparse when chained in one
command (`--streams 1 2 4 8 ... --sources a.mp4 b.mp4 ...` — argparse correctly stops each list at
the next `--flag`, verified live). Confirmed new files (`stream-pipeline-utils.py`, runbook,
dry-run-guards test) are not accidentally `.gitignore`d despite `.gitignore` being modified on this
branch.

## Findings ranked by severity

**Should-fix (before 19/9, cheap insurance, not observed to fail today):**
- `TRTRunner.__init__` never calls `context.set_input_shape()` for dynamic-batch inputs — relies on
  P1's June precedent to imply it'll also work for P3's 2-input relation head, which has never run
  on-device. Recommend either (a) a 2-minute defensive addition — call
  `self.context.set_input_shape(name, shape)` for each input right after computing `shape` in the
  constructor (harmless no-op if TRT already defaults to it, cheap insurance if it doesn't), or (b)
  at minimum, note in the runbook's Step 2 "Good:" line to explicitly watch for a shape-related
  exception on the *first* P3 relation-head engine build/run, since that's the one truly new
  code path this session touches on real hardware.

**Nit:**
- `deployment/scripts/export-p3-graphs.py` (221 lines, new/untracked this session) is 10% over the
  200-line guideline. Not in the 6-finding scope and not part of Lane B's stated file ownership,
  so flagging only for whoever owns it next.

## Everything else
All 6 prior findings are genuinely closed, not just claimed — I independently re-derived each fix
by reading the actual diffs/code, not by trusting the implementer's report. 25/25 tests pass
(re-run independently, not just re-quoted). No fabrication paths found. No security issues (no
`eval`/`shell=True`/pickle in touched files). The DRY refactor (`stream-pipeline-utils.py`) is a
genuine, tested extraction with correct CWD-independent importlib loading — the exact regression
class asked about (module resolution breaking under a different working directory) does not occur.

## Unresolved questions
1. Should `TRTRunner` add the defensive `set_input_shape()` call before 9/19, given it's a ~10-line
   change that only touches the one truly novel (never-run) code path (P3 2-input engine), or is
   the P1 June precedent considered sufficient confidence to skip it given time pressure?
2. `export-p3-graphs.py`'s 221 lines — in scope for a follow-up split, or accepted as-is since it's
   outside Lane B's file ownership?
