"""trt-runner-utils.py — Shared TensorRT engine I/O binding for the Jetson harnesses.

Fixes a defect present in the old jetson-benchmark.py / jetson-multi-stream-pipeline.py
TRTRunner: the constructor looped over every engine I/O tensor and called
set_tensor_address() correctly for each one (so the engine's own binding table was fine),
but then did `self.host_in, self.dev_in = host, dev` unconditionally for every INPUT
tensor — for a 2-input engine (the P3 relation head takes `tokens` + `mask`) the second
input assignment silently overwrote the reference to the first, so run() only ever wrote
into whichever input was enumerated last. The other input's device buffer stayed
allocated-but-never-populated. Not a crash — a silent wrong-answer risk.

This version keeps a dict of {name: {host, dev, shape}} per input/output and binds every
call argument by NAME, so it works for any number of inputs/outputs.

Both jetson-*.py callers load this via importlib (hyphenated filename, same pattern
already used for pipeline-utils.py) since they have no tensorrt/pycuda dependency at
import time here — this module only takes `trt`/`cuda` as constructor arguments, so it
imports cleanly on a machine with neither installed (e.g. for unit tests on Mac).
"""
import numpy as np


class TRTRunner:
    """Wraps one TensorRT engine. Pre-allocates host+device buffers per tensor, keyed by
    name, and reuses them across calls. Not thread-safe — call from a single thread/coroutine."""

    def __init__(self, trt, cuda, engine_path, batch=None):
        """batch: concrete size for a dynamic leading dim. Default None = the profile's opt
        shape, which is what every fixed-batch engine wants. p3-appearance is built with a
        real range (1:54:128) because one P3 decision embeds the clip's whole crop set in a
        single batched pass, so its per-decision cost must be measured at the real crop-count
        percentiles, not at batch 1."""
        self.cuda = cuda
        self.batch = batch
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs, self.outputs = {}, {}
        self.input_order, self.output_order = [], []
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        is_input = {n: self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in names}
        # Dynamic dims (-1, e.g. batch) must be resolved on the context BEFORE output shapes are
        # queried; use the profile-0 opt shape the engine was built with (min=opt=max=1 here).
        for name in names:
            if is_input[name] and -1 in tuple(self.context.get_tensor_shape(name)):
                lo, opt, hi = [tuple(x) for x in self.engine.get_tensor_profile_shape(name, 0)]
                want = opt
                if batch is not None:
                    want = (batch,) + opt[1:]
                    if not lo[0] <= batch <= hi[0]:
                        raise ValueError(
                            f'batch {batch} outside engine profile for {name}: '
                            f'min {lo[0]} max {hi[0]} (rebuild with --batch_profile)')
                self.context.set_input_shape(name, want)
        # Inputs first so output shapes are fully resolved when their buffers are sized.
        for name in sorted(names, key=lambda n: not is_input[n]):
            shape = tuple(self.context.get_tensor_shape(name))
            if -1 in shape:
                raise RuntimeError(f'unresolved dynamic shape for tensor {name}: {shape}')
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = np.empty(shape, dtype=dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(dev))
            entry = {'host': host, 'dev': dev, 'shape': shape}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = entry
                self.input_order.append(name)
            else:
                self.outputs[name] = entry
                self.output_order.append(name)

    def run(self, *args, **kwargs):
        """Positional args bind to inputs in engine-declared order (back-compat with the
        old single-input call sites, e.g. `runner.run(x)`); kwargs bind by tensor name —
        required for multi-input engines, e.g. `runner.run(tokens=tok, mask=msk)`.

        Returns the single output array directly if the engine has one output (matches the
        old single-output call sites), else a {name: array} dict.
        """
        if kwargs:
            values = kwargs
        elif len(args) == len(self.input_order):
            values = dict(zip(self.input_order, args))
        else:
            raise ValueError(
                f'expected {len(self.input_order)} input(s) {self.input_order}, '
                f'got {len(args)} positional arg(s)')
        missing = [n for n in self.input_order if n not in values]
        if missing:
            raise ValueError(f'missing input(s): {missing}')

        for name in self.input_order:
            entry = self.inputs[name]
            np.copyto(entry['host'], np.asarray(values[name]).reshape(entry['shape']))
            self.cuda.memcpy_htod_async(entry['dev'], entry['host'], self.stream)
        self.context.execute_async_v3(self.stream.handle)
        for name in self.output_order:
            entry = self.outputs[name]
            self.cuda.memcpy_dtoh_async(entry['host'], entry['dev'], self.stream)
        self.stream.synchronize()

        outs = {name: self.outputs[name]['host'].copy() for name in self.output_order}
        return outs[self.output_order[0]] if len(outs) == 1 else outs
