"""Regression test for Task 1a — deployment/scripts/trt-runner-utils.py.

Old bug (see trt-runner-utils.py docstring): the constructor bound every engine I/O tensor's
device address correctly, but `self.host_in, self.dev_in = host, dev` was reassigned
unconditionally for every INPUT tensor, so for a 2-input engine (relation head: tokens + mask)
run() only ever wrote into whichever input was enumerated last — the other input's device
buffer stayed allocated but never populated (silent wrong-answer, not a crash).

This test builds a fake `trt`/`cuda` pair (no real TensorRT/CUDA needed — works on Mac) with a
2-input, 1-output fake engine whose "kernel" computes output = [sum(tokens), sum(mask)]. If
TRTRunner failed to populate one of the two inputs, that input's contribution would be 0 (or a
crash from an unset numpy array), which this test would catch.

Usage: .venv/bin/python tests/test_trt_runner_utils.py
"""
import importlib.util
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location(
    'trt_runner_utils', os.path.join(ROOT, 'deployment', 'scripts', 'trt-runner-utils.py'))
tru = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tru)
TRTRunner = tru.TRTRunner


class FakeDeviceMem:
    _registry = {}

    def __init__(self, nbytes):
        self.nbytes = nbytes
        self.data = None
        FakeDeviceMem._registry[id(self)] = self

    def __int__(self):
        return id(self)


class FakeStream:
    handle = 0

    def synchronize(self):
        pass


class FakeCuda:
    Stream = FakeStream

    @staticmethod
    def mem_alloc(nbytes):
        return FakeDeviceMem(nbytes)

    @staticmethod
    def memcpy_htod_async(dev, host, stream):
        dev.data = np.array(host, copy=True)

    @staticmethod
    def memcpy_dtoh_async(host, dev, stream):
        if dev.data is None:
            raise AssertionError('reading from a device buffer that was never written — '
                                  'this is exactly the silent-overwrite bug being regressed')
        np.copyto(host, dev.data)


class FakeContext:
    def __init__(self, engine):
        self.engine = engine
        self.addr = {}

    def get_tensor_shape(self, name):
        return self.engine.tensors[name]['shape']

    def set_tensor_address(self, name, addr):
        self.addr[name] = addr

    def execute_async_v3(self, stream_handle):
        tok = FakeDeviceMem._registry[self.addr['tokens']].data
        msk = FakeDeviceMem._registry[self.addr['mask']].data
        if tok is None or msk is None:
            raise AssertionError('execute called with an unpopulated input buffer')
        out_dev = FakeDeviceMem._registry[self.addr['logits']]
        out_dev.data = np.array([[float(tok.sum()), float(msk.sum())]], dtype=np.float32)


class FakeEngine:
    def __init__(self):
        self.tensors = {
            'tokens': {'shape': (1, 2, 2), 'dtype': np.float32, 'mode': 'INPUT'},
            'mask': {'shape': (1, 2), 'dtype': np.float32, 'mode': 'INPUT'},
            'logits': {'shape': (1, 2), 'dtype': np.float32, 'mode': 'OUTPUT'},
        }
        self.names = list(self.tensors)

    @property
    def num_io_tensors(self):
        return len(self.tensors)

    def get_tensor_name(self, i):
        return self.names[i]

    def get_tensor_dtype(self, name):
        return self.tensors[name]['dtype']

    def get_tensor_mode(self, name):
        return self.tensors[name]['mode']

    def create_execution_context(self):
        return FakeContext(self)


class FakeLogger:
    WARNING = 0

    def __init__(self, level=None):
        pass


class FakeRuntime:
    def __init__(self, logger):
        pass

    def deserialize_cuda_engine(self, data):
        return FakeEngine()


class FakeTensorIOMode:
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'


class FakeTrt:
    Runtime = FakeRuntime
    Logger = FakeLogger
    TensorIOMode = FakeTensorIOMode

    @staticmethod
    def nptype(dtype):
        return dtype


def _build_runner():
    FakeDeviceMem._registry.clear()
    with tempfile.NamedTemporaryFile(suffix='.engine', delete=False) as f:
        f.write(b'not-a-real-engine-fake-runtime-ignores-this')
        path = f.name
    try:
        runner = TRTRunner(FakeTrt(), FakeCuda(), path)
    finally:
        os.unlink(path)
    return runner


def test_two_input_engine_binds_both_by_kwarg():
    """The bug this regresses: silent overwrite meant only ONE of {tokens, mask} was ever
    copied to its device buffer."""
    runner = _build_runner()
    tok = np.full((1, 2, 2), 3.0, dtype=np.float32)   # sum = 12
    msk = np.full((1, 2), 5.0, dtype=np.float32)      # sum = 10
    out = runner.run(tokens=tok, mask=msk)
    assert out.shape == (1, 2), out.shape
    assert abs(out[0, 0] - 12.0) < 1e-5, f'tokens input not bound correctly: {out}'
    assert abs(out[0, 1] - 10.0) < 1e-5, f'mask input not bound correctly: {out}'


def test_positional_args_bind_in_engine_order():
    """Back-compat: single/positional call sites (e.g. old jetson-benchmark.py usage)."""
    runner = _build_runner()
    tok = np.full((1, 2, 2), 1.0, dtype=np.float32)   # sum = 4
    msk = np.full((1, 2), 2.0, dtype=np.float32)      # sum = 4
    out = runner.run(tok, msk)
    assert abs(out[0, 0] - 4.0) < 1e-5
    assert abs(out[0, 1] - 4.0) < 1e-5


def test_missing_kwarg_raises():
    runner = _build_runner()
    tok = np.zeros((1, 2, 2), dtype=np.float32)
    try:
        runner.run(tokens=tok)
        raise AssertionError('expected ValueError for missing "mask" input')
    except ValueError as e:
        assert 'mask' in str(e)


class FakeDynamicContext(FakeContext):
    """Mimics TRT: dynamic (-1) batch dims stay unresolved until set_input_shape() is called,
    and output shapes resolve only once every input shape is set."""

    def __init__(self, engine):
        super().__init__(engine)
        self.set_shapes = {}

    def set_input_shape(self, name, shape):
        self.set_shapes[name] = tuple(shape)

    def get_tensor_shape(self, name):
        t = self.engine.tensors[name]
        if t['mode'] == 'INPUT':
            return self.set_shapes.get(name, (-1,) + t['shape'][1:])
        inputs = [n for n, v in self.engine.tensors.items() if v['mode'] == 'INPUT']
        return t['shape'] if all(n in self.set_shapes for n in inputs) else (-1,) + t['shape'][1:]


class FakeDynamicEngine(FakeEngine):
    def get_tensor_profile_shape(self, name, profile):
        s = self.tensors[name]['shape']
        return [s, s, s]                                   # min=opt=max (batch=1)

    def create_execution_context(self):
        return FakeDynamicContext(self)


def test_dynamic_batch_inputs_resolved_via_profile_before_output_alloc():
    """P3 relation head exports tokens/mask with a dynamic batch dim; without set_input_shape
    the constructor would np.empty() a -1 shape and crash on device."""
    FakeDeviceMem._registry.clear()
    orig = FakeRuntime.deserialize_cuda_engine
    FakeRuntime.deserialize_cuda_engine = lambda self, data: FakeDynamicEngine()
    try:
        runner = _build_runner()
    finally:
        FakeRuntime.deserialize_cuda_engine = orig
    assert runner.context.set_shapes == {'tokens': (1, 2, 2), 'mask': (1, 2)}
    assert runner.input_order == ['tokens', 'mask']
    out = runner.run(tokens=np.ones((1, 2, 2), np.float32), mask=np.ones((1, 2), np.float32))
    assert abs(out[0, 0] - 4.0) < 1e-5 and abs(out[0, 1] - 2.0) < 1e-5


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
