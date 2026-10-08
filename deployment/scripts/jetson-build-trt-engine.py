#!/usr/bin/env python3
"""F11 — Build TensorRT engines from FP32 ONNX on Jetson Orin (JetPack 6).

Inputs : deployment/export/*-fp32.onnx (TRT-compatible per trt-compat-report.md).
Outputs: deployment/export/jetson/*.engine + build log.

Build modes:
  --mode fp32 : pure FP32 engine (baseline parity check)
  --mode fp16 : FP16 engine (default; standard Jetson Orin deployment target)
  --mode int8 : INT8 engine (requires --calib-dir from jetson-prep-int8-calib.py)

Run on the Jetson, NOT on M3. Requires `tensorrt` (10.x bundled with JetPack 6),
`pycuda`, and CUDA visible. Engines are device-specific; do not copy across
Jetson SKUs.

Usage (on Jetson):
    python3 deployment/scripts/jetson-build-trt-engine.py --mode fp16
    python3 deployment/scripts/jetson-build-trt-engine.py --mode int8 \
        --calib-dir deployment/artifacts/calib-jetson
"""
import argparse
import os
import sys
import time

# IMPORTANT: pycuda.autoinit MUST be imported before tensorrt so that the primary
# CUDA context is established for both the Builder and any INT8 Calibrator.
# Mixing pycuda + tensorrt with reversed import order causes
# "CUDA context changed between createInferBuilder and buildSerializedNetwork"
# and tactic-skip failures during INT8 calibration.
import pycuda.autoinit  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPORT_DIR = os.path.join(ROOT, 'deployment', 'export')
ENGINE_DIR = os.path.join(ROOT, 'deployment', 'export', 'jetson')

MODELS = ['backbone', 'direct4-s314', 'scenario-s7',
          'pair0-s7', 'pair1-s7', 'pair2-s7', 'pair3-s7']

WORKSPACE_GB = 4
DEFAULT_BATCH = 1


def import_trt():
    try:
        import tensorrt as trt
        return trt
    except ImportError:
        sys.exit('ERROR: tensorrt not installed. JetPack 6 ships TRT 10.x; '
                 'verify with "python3 -c \'import tensorrt; print(tensorrt.__version__)\'".')


def make_calibrator(trt, calib_dir, model_name, batch_size=DEFAULT_BATCH):
    """Build a trt.IInt8EntropyCalibrator2-derived calibrator. TRT 10 requires the
    Python calibrator to subclass one of the trt.IInt8* base classes; defining the
    class inside the factory keeps trt import local to the build step."""
    import numpy as np
    import pycuda.driver as cuda
    from os.path import join

    class _Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            trt.IInt8EntropyCalibrator2.__init__(self)
            files = sorted(f for f in os.listdir(calib_dir)
                           if f.startswith(f'{model_name}-batch-') and f.endswith('.npy'))
            self.batches = [np.load(join(calib_dir, f)) for f in files]
            self.idx = 0
            self.cache_path = join(calib_dir, f'{model_name}.calib-cache')
            self.batch_size = batch_size
            self._device_buffer = None

        def get_batch_size(self): return self.batch_size

        def get_batch(self, names):
            if self.idx >= len(self.batches):
                return None
            arr = self.batches[self.idx].astype('float32', copy=False)
            if self._device_buffer is None:
                self._device_buffer = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(self._device_buffer, arr)
            self.idx += 1
            return [int(self._device_buffer)]

        def read_calibration_cache(self):
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'rb') as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache_path, 'wb') as f:
                f.write(cache)

    return _Calib()


def build_engine(trt, onnx_path, mode, calib_dir, model_name, batch_profile=None):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print('  parser:', parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB << 30)

    # TRT 10: any dynamic input shape requires an optimization profile.
    # Default is min=opt=max=DEFAULT_BATCH, which is correct for every P1 model and for the
    # P3 detector/head (one frame, one clip per pass). It is NOT correct for p3-appearance:
    # a P3 decision embeds the clip's whole crop set (p50=54, cap=128) in ONE batched pass,
    # which is how benchmark-p3-onnx.py times it. Pinning that input to 1 forces a 54-iteration
    # loop at measurement time and overstates P3's per-decision cost against its own ONNX
    # baseline. --batch_profile MIN:OPT:MAX builds the real shape range for such graphs.
    lo, opt, hi = batch_profile or (DEFAULT_BATCH, DEFAULT_BATCH, DEFAULT_BATCH)
    profile = builder.create_optimization_profile()
    has_dynamic = False
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape = list(tensor.shape)
        if any(d < 0 for d in shape):
            has_dynamic = True
            mk = lambda b: tuple(b if d < 0 else d for d in shape)
            profile.set_shape(tensor.name, mk(lo), mk(opt), mk(hi))
            print(f'  profile: {tensor.name} {shape} -> min{mk(lo)} opt{mk(opt)} max{mk(hi)}')
    if has_dynamic:
        config.add_optimization_profile(profile)

    if mode == 'fp16':
        if not builder.platform_has_fast_fp16:
            print('  WARN: platform_has_fast_fp16=False; engine will be slow')
        config.set_flag(trt.BuilderFlag.FP16)
    elif mode == 'int8':
        if not builder.platform_has_fast_int8:
            print('  WARN: platform_has_fast_int8=False; INT8 may not be supported')
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = make_calibrator(trt, calib_dir, model_name)
        if has_dynamic:
            config.set_calibration_profile(profile)
        # Keep FP16 as fallback for layers that don't quantise cleanly
        config.set_flag(trt.BuilderFlag.FP16)

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return None
    print(f'  build time: {time.time() - t0:.1f}s')
    return serialized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fp32', 'fp16', 'int8'], default='fp16')
    ap.add_argument('--calib-dir', default=os.path.join(ROOT, 'deployment',
                                                        'artifacts', 'calib-jetson'))
    ap.add_argument('--model', help='build a single named model instead of the '
                     'full P1 MODELS list (e.g. p3-detector); used for the output '
                     'filename <model>-<mode>.engine')
    ap.add_argument('--onnx', help='ONNX path for --model (default: '
                     'deployment/export/<model>-fp32.onnx)')
    ap.add_argument('--batch_profile', help='MIN:OPT:MAX for dynamic batch dims, e.g. 1:54:128 '
                     'for p3-appearance (p50/cap crop counts). Default pins them to '
                     f'{DEFAULT_BATCH}, which is right for per-frame/per-clip graphs only.')
    args = ap.parse_args()
    bp = tuple(int(x) for x in args.batch_profile.split(':')) if args.batch_profile else None
    if bp is not None and (len(bp) != 3 or not bp[0] <= bp[1] <= bp[2] or bp[0] < 1):
        sys.exit(f'--batch_profile must be MIN:OPT:MAX with 1 <= MIN <= OPT <= MAX (got {args.batch_profile})')

    trt = import_trt()
    print(f'TensorRT {trt.__version__} | mode={args.mode}')
    os.makedirs(ENGINE_DIR, exist_ok=True)

    targets = [(args.model, args.onnx or os.path.join(EXPORT_DIR, f'{args.model}-fp32.onnx'))] \
        if args.model else [(name, os.path.join(EXPORT_DIR, f'{name}-fp32.onnx')) for name in MODELS]

    for name, onnx in targets:
        if not os.path.exists(onnx):
            print(f'[SKIP] {name}: ONNX not found at {onnx}')
            continue
        print(f'\n[BUILD] {name} ({args.mode})')
        eng = build_engine(trt, onnx, args.mode, args.calib_dir, name, bp)
        if eng is None:
            print(f'  FAILED to build {name}')
            continue
        out = os.path.join(ENGINE_DIR, f'{name}-{args.mode}.engine')
        with open(out, 'wb') as f:
            f.write(eng)
        print(f'  [SAVED] {out} ({os.path.getsize(out) / 1024 / 1024:.2f} MB)')


if __name__ == '__main__':
    main()
