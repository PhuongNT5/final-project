"""Latency of the R48 deployment path (CLIP ViT-B/32 -> B1 temporal head), host CPU or Jetson TRT.

The composition lives here ONCE and is timed through either backend, so the Jetson number and the
host number are the same arithmetic over the same graphs and are directly comparable. The backend
is never inferred: `--backend onnxruntime` is a host number and must never be quoted as a device
one; `--backend tensorrt` requires the engines built by jetson-build-trt-engine.py.

Why this benchmark exists: the R48 accuracy result shrank the head to 201k parameters but grew the
backbone it depends on from MobileNetV3-small (~2.5M) to CLIP ViT-B/32 (~87.8M). If the backbone
dominates, "small head" is not an edge claim and the edge chapter has to say so.

One decision = 16 frames through the tower + one head call, matching the evaluated configuration
exactly (T=16, whole-frame, B1 uses NO detector). --detector adds the YOLO localisation cost, which
is a separate obligation of the system (say WHERE the hazard is), not part of the B1 class decision.

Usage:
    .venv/bin/python deployment/scripts/benchmark-r48-clip-pipeline.py [--runs 20]
    python3 deployment/scripts/benchmark-r48-clip-pipeline.py --backend tensorrt --mode fp16 \
        --detector --json
"""
import argparse
import os
import statistics
import time
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
EXPORT = os.path.join(ROOT, 'deployment', 'export')
ENGINE_DIR = os.path.join(EXPORT, 'jetson')
FRAMES = 16
# (stem, batch, params, note) -- stem resolves to <stem>-fp32.onnx or <stem>-<mode>.engine.
TOWER = ('r48-clip-vit-b32-vision', FRAMES, 87_849_216, 'frozen, shared by every R48 model')
HEAD = ('r48-b1-clip-temporal-3seed', 1, 603_270, '3-seed ensemble, the headline classifier')
DETECTOR = ('p3-detector', 1, None, 'YOLOv11s localisation, not in the B1 class decision')
LEGACY = ('p3-appearance', FRAMES, 2_542_856, 'the representation this lane replaced')


def _load(name, fname):
    spec = spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ort_runner(path, batch):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, o, providers=['CPUExecutionProvider'])
    shapes = {i.name: [d if isinstance(d, int) and d > 0 else batch for d in i.shape]
              for i in sess.get_inputs()}
    return (lambda feeds: sess.run(None, feeds)), shapes


def _trt_runner(path, batch):
    import pycuda.autoinit  # noqa: F401  (must precede tensorrt; see jetson-build-trt-engine.py)
    import pycuda.driver as cuda
    import tensorrt as trt
    runner = _load('trt_runner_utils', 'trt-runner-utils.py').TRTRunner(
        trt, cuda, path, batch=batch)
    shapes = {n: list(v['shape']) for n, v in runner.inputs.items()}
    return (lambda feeds: runner.run(**feeds)), shapes


def resolve(stem, backend, mode, engine_dir):
    if backend == 'tensorrt':
        return os.path.join(engine_dir, f'{stem}-{mode}.engine')
    return os.path.join(EXPORT, f'{stem}-fp32.onnx')


def synth(shapes):
    """Random activations, except validity flags: a clip with all 16 frames present is the
    configuration that was evaluated, and a random mask would time a different graph path."""
    out = {}
    for name, shape in shapes.items():
        low = name.lower()
        out[name] = (np.ones(shape, dtype='float32') if 'valid' in low or 'mask' in low
                     else np.random.randn(*shape).astype('float32'))
    return out


def time_stage(stem, batch, backend, mode, engine_dir, runs, warmup):
    path = resolve(stem, backend, mode, engine_dir)
    if not os.path.exists(path):
        return None, path
    call, shapes = (_trt_runner if backend == 'tensorrt' else _ort_runner)(path, batch)
    feeds = synth(shapes)
    for _ in range(warmup):
        call(feeds)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        call(feeds)
        ts.append((time.perf_counter() - t) * 1000)
    return {'path': os.path.relpath(path, ROOT), 'batch': batch, 'runs': runs,
            'median_ms': round(statistics.median(ts), 3),
            'p95_ms': round(float(np.percentile(ts, 95)), 3),
            'p99_ms': round(float(np.percentile(ts, 99)), 3),
            'min_ms': round(min(ts), 3), 'max_ms': round(max(ts), 3)}, path


def report(label, r, note=''):
    print(f'  {label:<44} {r["median_ms"]:8.1f} ms   p95 {r["p95_ms"]:7.1f} ms  {note}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend', default='onnxruntime', choices=['onnxruntime', 'tensorrt'])
    ap.add_argument('--mode', default='fp16', choices=['fp32', 'fp16', 'int8'],
                    help='engine precision suffix; --backend tensorrt only')
    ap.add_argument('--engine-dir', default=ENGINE_DIR)
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--detector', action='store_true',
                    help='also time YOLO localisation (reported separately, never folded into '
                         'the B1 decision cost)')
    ap.add_argument('--power', action='store_true',
                    help='sample tegrastats across the run; null on a host without it')
    ap.add_argument('--json', action='store_true',
                    help='write deployment/artifacts/jetson-measurements/r48-pipeline-*.json '
                         'with device state and git commit embedded')
    args = ap.parse_args()
    host = args.backend == 'onnxruntime'

    U = _load('jetson_measure_utils', 'jetson-measure-utils.py')
    sampler = None
    if args.power:
        sampler = U.TegrastatsSampler()
        sampler.start()

    print(f'[backend] {args.backend}'
          + ('' if host else f' {args.mode} engines from {os.path.relpath(args.engine_dir, ROOT)}')
          + f', {args.runs} runs after {args.warmup} warmups')
    print(f'[config] one decision = {FRAMES} frames, whole-frame (B1 uses NO detector)\n')

    stages = {}
    for key, (stem, batch, params, note) in (('tower', TOWER), ('head', HEAD)):
        r, path = time_stage(stem, batch, args.backend, args.mode, args.engine_dir,
                             args.runs, args.warmup)
        if r is None:
            raise SystemExit(f'missing {path}\n'
                             + ('export the graphs first: export-clip-vision-tower.py and '
                                'export-b1-clip-temporal-head.py' if host else
                                'build the engines first: jetson-build-trt-engine.py '
                                f'--mode {args.mode} --model {stem} --batch_profile '
                                f'1:{batch}:{batch}'))
        r['params'] = params
        stages[key] = r
        report(f'{stem}, batch {batch}', r, f'({params:,} params)')

    decision = stages['tower']['median_ms'] + stages['head']['median_ms']
    backbone_share = 100 * stages['tower']['median_ms'] / decision
    print(f"\n  {'PER DECISION (tower + head)':<44} {decision:8.1f} ms   "
          f"-> {1000 / decision:.2f} decisions/s, {FRAMES * 1000 / decision:.1f} frames/s")
    print(f'  backbone share: {backbone_share:.1f}% of wall clock')

    extras = {}
    if args.detector:
        r, path = time_stage(*DETECTOR[:2], args.backend, args.mode, args.engine_dir,
                             args.runs, args.warmup)
        if r is None:
            print(f'\n[detector] SKIPPED, missing {os.path.relpath(path, ROOT)}')
        else:
            extras['detector'] = r
            print(f'\n[localisation] a separate obligation, one frame per pass:')
            report('YOLOv11s detector', r, DETECTOR[3])

    r, _ = time_stage(*LEGACY[:2], args.backend, args.mode, args.engine_dir, args.runs,
                      args.warmup)
    if r is not None:
        extras['legacy_appearance'] = r
        print(f'\n[reference] {LEGACY[3]}:')
        report(f'MobileNetV3-small appearance, batch {FRAMES}', r, f'({LEGACY[2]:,} params)')
        print(f'  CLIP costs {stages["tower"]["median_ms"] / r["median_ms"]:.1f}x the appearance '
              f'backbone it replaced, for +4.0 pp balanced accuracy (cluster CI includes 0).')

    payload = {'backend': args.backend, 'mode': None if host else args.mode, 'frames': FRAMES,
               'stages': stages, 'extras': extras,
               'per_decision_ms': round(decision, 3),
               'decisions_per_sec': round(1000 / decision, 3),
               'backbone_share_pct': round(backbone_share, 2),
               'is_device_number': not host}
    if sampler is not None:
        payload['tegrastats'] = sampler.stop()
    if args.json:
        print(f'\n[SAVED] {U.write_json_result("r48-pipeline", payload, dry_run=host)}')
    if host:
        print('\n[caveat] host CPU only. Jetson Orin Nano Super numbers require the device and '
              'are produced by this same script with --backend tensorrt; do not substitute these.')


if __name__ == '__main__':
    main()
