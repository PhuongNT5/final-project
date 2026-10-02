"""Measured end-to-end cadence of the B1 decision path, decode included, on device.

Block 3b-note 4 records a stage-SUM duty budget for this path. A sum is not a cadence: it cannot see
the preprocessing between decode and the tower, the host-to-device copies, or decode and inference
contending for one memory bus. This measures the composed loop instead -- sequential decode into a
ring buffer, one decision every `--stride` frames sampled by the same linspace rule the evaluation
uses -- and reports the sum beside it so the gap is visible rather than hidden.

Two cadences are reported and they answer different questions. `steady_state` is the interval
between consecutive decisions, which is what a running deployment sees. `measured_ms_per_decision`
is wall clock over decisions emitted, which also charges the `--window` frames decoded before the
first decision can be taken -- on 6 s iSafetyBench clips that fill is a large share, so the two
differ, and quoting the second as the deployed cadence would overstate the cost.

Single-threaded on purpose: a deployment would overlap decode with inference, so throughput here is
a LOWER bound and duty an UPPER bound. The detector is not in this loop -- localisation is timed
separately (Block 3b) and the class decision never consumes boxes.

Usage (on the Jetson, in a quiet window):
  python3 deployment/scripts/benchmark-r48-composed-cadence.py \
      --engine-dir deployment/export/jetson-r48 --clips 12 --json
"""
import argparse
import collections
import os
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
ENGINE_DIR = os.path.join(ROOT, 'deployment', 'export', 'jetson')
FIXTURE = os.path.join(ROOT, 'deployment', 'artifacts', 'r48-parity-fixture', 'fixture.npz')
DATA_ROOT = os.path.join(ROOT, 'data', 'isafetybench')
FRAMES = 16
TOWER, HEAD = 'r48-clip-vit-b32-vision', 'r48-b1-clip-temporal-3seed'


def _load(name, fname):
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_clip(path, tower, head, mean, std, window, stride, prep, sample):
    """One clip, decoded start to finish. Returns per-decision timings and the wall clock."""
    cap = cv2.VideoCapture(path)
    buf, since, out = collections.deque(maxlen=window), 0, []
    dec, n_read = [], 0
    valid = np.ones((1, FRAMES), dtype='float32')
    t_wall = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        ok, fr = cap.read()
        if not ok:
            break
        dec.append((time.perf_counter() - t0) * 1000)
        buf.append(fr)
        n_read += 1
        since += 1
        if len(buf) < window or since < stride:
            continue
        since = 0
        t0 = time.perf_counter()
        x = prep(sample(buf), mean, std).astype('float32', copy=False)
        t_prep = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        emb = tower.run(pixel_values=x)
        t_tower = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        p = head.run(frames=emb[None].astype('float32'), valid=valid)[0]
        t_head = (time.perf_counter() - t0) * 1000
        out.append({'prep_ms': t_prep, 'tower_ms': t_tower, 'head_ms': t_head,
                    'compute_ms': t_prep + t_tower + t_head, 'p_unsafe': float(p[1]),
                    'emit_sec': time.perf_counter() - t_wall})
    cap.release()
    return {'clip': os.path.basename(path), 'frames_decoded': n_read,
            'wall_sec': time.perf_counter() - t_wall, 'decode_ms': dec, 'decisions': out}


def probe_idle(tower, x, gap_ms, runs, warmup, pct):
    """Why this exists: the tower costs ~2,5x more inside the loop than measured in isolation, and
    two causes predict that -- GPU clock droop over the idle gap between decisions, or contention
    with decode/preprocess. Arm B reproduces only the IDLE: identical call, same unpinned clocks,
    nothing else running. Contention is excluded by construction, so a rise in arm B isolates droop.
    The adjudication rule is pre-registered in
    reports/finding-2026-10-01-stage-sum-is-not-a-cadence.md and is not renegotiated here."""
    out = {}
    for arm, gap in (('back_to_back', 0.0), (f'idle_{gap_ms:.0f}ms', gap_ms / 1000.0)):
        for _ in range(warmup):
            tower.run(pixel_values=x)
        ts = []
        for _ in range(runs):
            if gap:
                time.sleep(gap)
            t0 = time.perf_counter()
            tower.run(pixel_values=x)
            ts.append((time.perf_counter() - t0) * 1000)
        out[arm] = pct(ts)
        print(f"  {arm:16s} tower p50 {out[arm]['p50']:7.3f} ms  p95 {out[arm]['p95']:7.3f} ms  "
              f"n={out[arm]['n']}")
    a, b = out['back_to_back']['p50'], out[f'idle_{gap_ms:.0f}ms']['p50']
    out['idle_over_back_to_back'] = round(b / a, 3)
    out['idle_gap_ms'] = gap_ms
    out['in_loop_tower_p50_ms'] = 58.630      # 3-repeat median, r48-composed-cadence 2026-10-01
    out['is_device_number'] = True
    print(f"  idle / back-to-back = {b / a:.3f}x   (in-loop p50 was 58.630 ms)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine-dir', default=ENGINE_DIR)
    ap.add_argument('--mode', default='fp16', choices=['fp32', 'fp16'])
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--clips', type=int, default=12)
    ap.add_argument('--window', type=int, default=48, help='demo default; one decision per window')
    ap.add_argument('--stride', type=int, default=24, help='demo default; frames between decisions')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--probe-idle', action='store_true',
                    help='discriminator only: tower back-to-back vs with an injected idle gap')
    ap.add_argument('--idle-gap-ms', type=float, default=210.0,
                    help='measured in-loop non-tower time per decision')
    ap.add_argument('--probe-runs', type=int, default=60)
    args = ap.parse_args()

    import pycuda.autoinit  # noqa: F401  (must precede tensorrt)
    import pycuda.driver as cuda
    import tensorrt as trt
    TRU = _load('trt_runner_utils', 'trt-runner-utils.py')
    U = _load('jetson_measure_utils', 'jetson-measure-utils.py')
    PAR = _load('verify_r48_device_parity', 'verify-r48-device-parity.py')
    AGG = _load('benchmark_r48_cadence_utils', 'benchmark-r48-cadence-utils.py')

    f = np.load(FIXTURE, allow_pickle=True)
    mean, std = f['mean'], f['std']
    tower = TRU.TRTRunner(trt, cuda, os.path.join(args.engine_dir, f'{TOWER}-{args.mode}.engine'),
                          batch=FRAMES)
    head = TRU.TRTRunner(trt, cuda, os.path.join(args.engine_dir, f'{HEAD}-{args.mode}.engine'),
                         batch=1)
    paths = AGG.pick_clips(args.data_root, args.clips, args.seed)
    print(f'[composed] {len(paths)} clips, window {args.window} stride {args.stride}, '
          f'{args.mode} TRT engines from {args.engine_dir}')
    print('[config] one decision = 16 frames, whole-frame CLIP -> B1; NO detector in this loop')

    if args.probe_idle:
        cap = cv2.VideoCapture(paths[0])
        frames = []
        while len(frames) < FRAMES:
            ok, fr = cap.read()
            if not ok:
                raise SystemExit(f'{paths[0]} has fewer than {FRAMES} frames')
            frames.append(fr)
        cap.release()
        x = PAR.preprocess(frames, mean, std).astype('float32', copy=False)
        print(f'[probe] tower only, idle gap {args.idle_gap_ms:.0f} ms, '
              f'{args.probe_runs} runs/arm, clocks as found')
        r = probe_idle(tower, x, args.idle_gap_ms, args.probe_runs, 10, U.percentiles)
        if args.json:
            print(f"\n[SAVED] {U.write_json_result(f'r48-idle-droop-probe-{args.mode}', r, False)}")
        return

    clips = []
    for i, p in enumerate(paths):
        c = run_clip(p, tower, head, mean, std, args.window, args.stride,
                     PAR.preprocess, AGG.sample_window)
        clips.append(c)
        print(f"  [{i + 1:2d}/{len(paths)}] {c['clip'][:38]:38s} "
              f"{c['frames_decoded']:4d} frames  {len(c['decisions']):3d} decisions  "
              f"{c['wall_sec']:6.2f} s", flush=True)

    s = AGG.summarise(clips, args.window, args.stride, U.percentiles)
    AGG.report(s)
    if args.json:
        s['per_clip'] = [{k: v for k, v in c.items() if k != 'decode_ms'} for c in clips]
        print(f"\n[SAVED] {U.write_json_result(f'r48-composed-cadence-{args.mode}', s, False)}")


if __name__ == '__main__':
    main()
