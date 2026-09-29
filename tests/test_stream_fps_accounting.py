"""Regression test for Task 1b — aggregate_fps() in multi-stream-pipeline.py and
jetson-multi-stream-pipeline.py.

Old bug: aggregate_fps was computed as total_frame_count / elapsed, where frame_count includes
frames DROPPED because the queue was full — a dropped frame was never run through the backbone,
so this inflated reported throughput at saturation (e.g. "136.89 FPS at N=4, 328/960 dropped"
counted 328 never-inferenced frames as processed).

This test calls the ACTUAL aggregate_fps() function from both production files (via importlib,
not a reimplementation) with fabricated StreamState objects, so it fails if the bug regresses.

Usage: .venv/bin/python tests/test_stream_fps_accounting.py
"""
import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub_pycuda_if_needed():
    try:
        import pycuda.autoinit  # noqa: F401
    except ImportError:
        pycuda_mod = types.ModuleType('pycuda')
        autoinit_mod = types.ModuleType('pycuda.autoinit')
        driver_mod = types.ModuleType('pycuda.driver')
        driver_mod.Stream = type('Stream', (), {})
        pycuda_mod.autoinit = autoinit_mod
        pycuda_mod.driver = driver_mod
        sys.modules.setdefault('pycuda', pycuda_mod)
        sys.modules.setdefault('pycuda.autoinit', autoinit_mod)
        sys.modules.setdefault('pycuda.driver', driver_mod)


def _check_aggregate(mod, StreamStateCls):
    """Shared assertions for both files' aggregate_fps + StreamState pair."""
    s0 = StreamStateCls(sid=0)
    s0.frame_count, s0.processed, s0.dropped = 100, 82, 18
    s1 = StreamStateCls(sid=1)
    s1.frame_count, s1.processed, s1.dropped = 100, 70, 30
    states = {0: s0, 1: s1}
    elapsed = 10.0

    out = mod.aggregate_fps(states, elapsed)
    # processed-only throughput: (82+70)/10 = 15.2 — NOT (100+100)/10 = 20.0 (the old bug).
    assert out['aggregate_fps'] == 15.2, out
    assert out['read_fps'] == 20.0, out
    assert out['total_dropped'] == 48 and out['total_processed'] == 152 and out['total_frames'] == 200
    assert abs(out['drop_rate'] - 48 / 200) < 1e-9, out

    # Zero-frame edge case must not divide by zero.
    z0 = StreamStateCls(sid=0)
    out_zero = mod.aggregate_fps({0: z0}, elapsed=5.0)
    assert out_zero['drop_rate'] == 0.0
    assert out_zero['aggregate_fps'] == 0.0


def test_multi_stream_pipeline_aggregate_fps():
    mod = _import(os.path.join(ROOT, 'deployment', 'scripts', 'multi-stream-pipeline.py'),
                 'multi_stream_pipeline')
    _check_aggregate(mod, mod.StreamState)


def test_jetson_multi_stream_pipeline_aggregate_fps():
    _stub_pycuda_if_needed()
    mod = _import(os.path.join(ROOT, 'deployment', 'scripts', 'jetson-multi-stream-pipeline.py'),
                 'jetson_multi_stream_pipeline')
    _check_aggregate(mod, mod.StreamState)


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
