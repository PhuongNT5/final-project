"""Syntactic-exercise test for jetson-multi-stream-pipeline.py (Task 1a/1b).

This file has an unconditional top-level `import pycuda.autoinit` / `import pycuda.driver as
cuda`, so it cannot be imported on a Mac dev machine (no CUDA) without stubbing pycuda first —
unlike jetson-benchmark.py, which defers its tensorrt/pycuda import into import_trt(). Per the
task instructions, this ensures the file "at least imports and is syntactically exercised" with
a fake/stub runner, since the TRTRunner/aggregate_fps fixes inside it cannot be exercised
end-to-end without real Jetson hardware.

Usage: .venv/bin/python tests/test_jetson_multi_stream_import.py
"""
import sys
import types

try:
    import pycuda.autoinit  # noqa: F401
    _REAL_PYCUDA = True
except ImportError:
    _REAL_PYCUDA = False
    pycuda_mod = types.ModuleType('pycuda')
    autoinit_mod = types.ModuleType('pycuda.autoinit')
    driver_mod = types.ModuleType('pycuda.driver')

    class _FakeStream:
        pass

    driver_mod.Stream = _FakeStream
    pycuda_mod.autoinit = autoinit_mod
    pycuda_mod.driver = driver_mod
    sys.modules['pycuda'] = pycuda_mod
    sys.modules['pycuda.autoinit'] = autoinit_mod
    sys.modules['pycuda.driver'] = driver_mod

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_target():
    spec = importlib.util.spec_from_file_location(
        'jetson_multi_stream_pipeline',
        os.path.join(ROOT, 'deployment', 'scripts', 'jetson-multi-stream-pipeline.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_imports_without_error():
    mod = _import_target()
    assert hasattr(mod, 'StreamState')
    assert hasattr(mod, 'run_config')
    assert hasattr(mod, 'aggregate_fps')
    assert hasattr(mod, '_make_trt_runner')
    assert hasattr(mod, 'main')


def test_stream_state_has_processed_field():
    """Task 1b: StreamState must distinguish processed (reached backbone) from frame_count
    (incl. dropped)."""
    mod = _import_target()
    s = mod.StreamState(sid=0)
    assert hasattr(s, 'processed') and hasattr(s, 'frame_count') and hasattr(s, 'dropped')
    assert s.processed == 0 and s.frame_count == 0 and s.dropped == 0


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
            print(f'  PASS  {name}' + ('' if _REAL_PYCUDA else '  (stubbed pycuda)'))
        except Exception:
            failures += 1
            print(f'  FAIL  {name}')
            traceback.print_exc()
    print('ALL PASS' if failures == 0 else f'{failures} FAILURE(S)')
    sys.exit(1 if failures else 0)
