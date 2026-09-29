"""Sanity tests for the Task 2 Jetson measurement suite's non-hardware logic: engine discovery
by filename (must stay class-count-agnostic), percentiles, the generic decision rule, and the
dry-run tegrastats sampler (must record ticks, never fabricate telemetry numbers).

Usage: .venv/bin/python tests/test_jetson_measure_utils.py
"""
import importlib.util
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, 'deployment', 'scripts', fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load('jetson_measure_utils', 'jetson-measure-utils.py')
LAT = _load('jetson_measure_latency', 'jetson-measure-latency.py')


def test_discover_engines_is_class_count_agnostic():
    """Must work whether stages are the old 4-class heads or the new binary+aux heads —
    discovery is purely filename-pattern based, no hardcoded stage list."""
    with tempfile.TemporaryDirectory() as d:
        for fn in ('backbone-fp32.engine', 'backbone-fp16.engine', 'binary-head-fp16.engine',
                  'binary-head-int8.engine', 'aux-cause-head-fp16.engine', 'notes.txt'):
            open(os.path.join(d, fn), 'w').close()
        out = LAT.discover_engines(d)
        assert set(out['backbone']) == {'fp32', 'fp16'}, out
        assert set(out['binary-head']) == {'fp16', 'int8'}, out
        assert set(out['aux-cause-head']) == {'fp16'}, out
        assert 'notes' not in out


def test_percentiles_empty_and_nonempty():
    empty = U.percentiles([])
    assert empty['n'] == 0 and empty['p50'] is None
    xs = list(range(1, 101))
    p = U.percentiles(xs)
    assert p['p50'] == 50.5, p
    assert p['n'] == 100


def test_decide_argmax_and_threshold():
    probs4 = np.array([[0.1, 0.7, 0.1, 0.1], [0.9, 0.03, 0.03, 0.04]])
    assert list(U.decide(probs4, 'argmax')) == [1, 0]
    probs_bin = np.array([[0.2, 0.8], [0.6, 0.4]])
    assert list(U.decide(probs_bin, 'threshold', 0.5)) == [1, 0]


def test_model_size_report_real_bytes():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'x-fp16.engine')
        with open(p, 'wb') as f:
            f.write(b'0' * (2 * 1024 * 1024))
        out = U.model_size_report(d)
        assert out['x-fp16.engine'] == 2.0, out


def test_tegrastats_dry_run_never_fabricates_numbers():
    """No tegrastats binary on Mac -> sampler falls back to dry_run ticks with no power/temp
    fields, not invented numbers."""
    s = U.TegrastatsSampler(interval_ms=50)
    s.start()
    import time
    time.sleep(0.2)
    out = s.stop()
    assert out['dry_run'] is True
    assert out['n_samples'] >= 1
    for sample in out['samples']:
        assert 'power_now_mw' not in sample
        assert sample.get('dry_run') is True


def test_parse_tegrastats_line_real_format():
    line = ('RAM 3123/7620MB (lfb 12x4MB) SWAP 0/3810MB CPU [10%@1500,5%@1500] '
            'GR3D_FREQ 42% [email protected] VDD_IN 4321mW/4000mW')
    out = U.parse_tegrastats_line(line)
    assert out['ram_used_mb'] == 3123 and out['ram_total_mb'] == 7620
    assert out['power_now_mw'] == 4321 and out['power_avg_mw'] == 4000
    assert out['gpu_util_pct'] == 42


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
