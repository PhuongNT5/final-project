"""Regression test for code-review gap #5 (code-reviewer-2026-09-16-jetson-session-prep-review.md
section 5): every hardware-dependent CLI orchestration function in jetson-measure-latency.py /
jetson-measure-power.py / jetson-full-measurement.py must return `skipped_reason` + explicit
`None`/empty fields on a non-Jetson host, never a fabricated number. Before this file only the
shared *helpers* (discover_engines, percentiles, decide, TegrastatsSampler, ...) had tests
(test_jetson_measure_utils.py) — the CLI-facing run_* functions that actually assemble the
skipped/null-field results had none.

Also regresses the run_memory dead-key fix from the same review (§ "Real bug found — schema
inconsistency in run_memory"): the dry-run branch previously set a never-produced
`device_free_total_mb` key instead of nulling the real 3-key schema.

Usage: .venv/bin/python tests/test_jetson_measure_dry_run_guards.py
"""
import importlib.util
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, 'deployment', 'scripts', fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LAT = _load('jetson_measure_latency', 'jetson-measure-latency.py')
POW = _load('jetson_measure_power', 'jetson-measure-power.py')
FULL = _load('jetson_full_measurement', 'jetson-full-measurement.py')


def _assert_skipped(result, numeric_keys):
    assert result.get('skipped_reason'), f'expected a non-empty skipped_reason, got {result}'
    for k in numeric_keys:
        assert result.get(k) is None, f'{k} must be None on skip, got {result.get(k)!r}'


def test_run_latency_dry_run_has_skipped_reason_and_no_numbers():
    with tempfile.TemporaryDirectory() as d:
        r = LAT.run_latency(d, 'fp16', warmup=1, runs=1, dry_run=True)
        _assert_skipped(r, ['end_to_end_naive_sum_ms_p50', 'throughput_decisions_per_sec_naive'])
        assert r['per_stage_ms'] == {}


def test_run_parity_dry_run_has_skipped_reason():
    with tempfile.TemporaryDirectory() as d:
        r = LAT.run_parity(d, warmup=1, runs=1, rule='argmax', threshold=0.5, dry_run=True)
        assert r.get('skipped_reason')
        assert r['stages'] == {}


def test_run_energy_no_engine_has_skipped_reason_and_no_numbers():
    with tempfile.TemporaryDirectory() as d:
        r = POW.run_energy(d, 'fp16', 'backbone', duration_sec=1, dry_run=True)
        assert r['skipped_reason'].startswith('engine not found')
        # Driven off the module's own key tuple so a schema rename cannot silently outrun
        # this guard (it did once: joules_per_inference -> *_total_board / *_delta_idle).
        for k in POW._ENERGY_KEYS:
            assert k not in r, f'{k} should not be set when the engine is missing, got {r}'


def test_run_energy_dry_run_with_engine_present_still_no_numbers():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, 'backbone-fp16.engine'), 'w').close()
        r = POW.run_energy(d, 'fp16', 'backbone', duration_sec=1, dry_run=True)
        assert r.get('skipped_reason', '').startswith('dry-run')
        for k in POW._ENERGY_KEYS:
            assert k in r, f'{k} missing from dry-run energy result, got {r}'
            assert r[k] is None, f'{k} must be null on dry-run, got {r[k]}'
        # Both energy framings must be present-and-null, not just the total-board one:
        # a dry run may never imply the idle-subtracted figure was measurable.
        assert r['joules_per_inference_total_board'] is None
        assert r['joules_per_inference_delta_idle'] is None
        assert 'joules_per_inference' not in r, 'ambiguous unlabelled key must not come back'


def test_run_thermal_soak_dry_run_has_skipped_reason_and_no_windows():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, 'backbone-fp16.engine'), 'w').close()
        r = POW.run_thermal_soak(d, 'fp16', 'backbone', duration_min=0.01, window_sec=1, dry_run=True)
        assert r.get('skipped_reason', '').startswith('dry-run')
        assert r['windows'] == []
        assert 'possible_throttling' not in r


def test_run_memory_dry_run_nulls_the_real_schema_not_a_dead_key():
    """Regression for the run_memory dry-run dead-key fix — must set device_free_mb/
    device_total_mb/device_used_mb (the real success-path keys, per JETSON-RUNBOOK-2026-09-19.md
    Step 9), not the never-produced device_free_total_mb."""
    r = POW.run_memory('/nonexistent', 'fp16', dry_run=True)
    assert r.get('skipped_reason', '').startswith('dry-run')
    assert r.get('device_free_mb') is None
    assert r.get('device_total_mb') is None
    assert r.get('device_used_mb') is None
    assert 'device_free_total_mb' not in r
    # Engine-footprint keys added 2026-09-29 (run_memory previously loaded no engine at all).
    assert r.get('engines_resident_delta_mb') is None
    assert r.get('engines_loaded') is None
    assert isinstance(r['host_peak_rss_mb'], float)  # real host-only measurement, ok non-null


def test_run_streams_dry_run_has_skipped_reason_no_configs():
    r = FULL.run_streams('/nonexistent', 'fp16', [1, 2], ['a.mp4'], 240, 30.0, dry_run=True)
    assert r.get('skipped_reason', '').startswith('dry-run')
    assert 'configs' not in r


def test_run_streams_no_sources_has_skipped_reason_even_without_dry_run():
    """Non-Jetson host with --streams but no --sources must not attempt to import the
    pycuda-dependent jetson-multi-stream-pipeline.py module (which would crash at import time
    on a machine without pycuda, per the edge case noted in the 2026-09-16 code review)."""
    r = FULL.run_streams('/nonexistent', 'fp16', [1], None, 240, 30.0, dry_run=False)
    assert r.get('skipped_reason') == 'no --sources given'


if __name__ == '__main__':
    import traceback
    failures = 0
    mod = sys.modules[__name__]
    for name in sorted(dir(mod)):
        if not name.startswith('test_'):
            continue
        fn = getattr(mod, name)
        try:
            fn()
            print(f'  PASS  {name}')
        except Exception:
            failures += 1
            print(f'  FAIL  {name}')
            traceback.print_exc()
    print('ALL PASS' if failures == 0 else f'{failures} FAILURE(S)')
    sys.exit(1 if failures else 0)
