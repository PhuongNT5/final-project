"""Profile the preprocessing stage of the R48 decision path, sub-step by sub-step, on real frames.

Why this exists: the composed cadence measurement
(reports/finding-2026-10-01-stage-sum-is-not-a-cadence.md) found preprocessing is 55,5 % of the
decision interval -- 160,8 ms for 16 frames, ~10 ms/frame of CPU resize+normalise -- and that no
budget in the thesis ever contained it. That finding records the split INSIDE preprocessing as
unmeasured and explicitly forbids claiming a win from it. This script measures the split, and tests
candidate replacements against a rule written before the first run.

ACCEPTANCE RULE (pre-registered; not renegotiated after seeing timings):
  A variant replaces the current preprocessing ONLY IF both hold.
    (1) BIT-IDENTICAL output on every frame tested -- exact equality of the whole tensor, not a
        tolerance. The deployed 84,2 % was measured through the current pixels; a variant that moves
        any pixel is a different system and its latency is not comparable.
    (2) p50 at least 10 % below the current p50, over --repeats independent repeats.
  Faster but not bit-identical => REJECTED, and reported as rejected with its diff.

Runs on host or device; needs only cv2 + numpy, no CUDA, no engines.

Usage:
  .venv/bin/python deployment/scripts/profile-r48-preprocess.py --data-root data/test
  python3 deployment/scripts/profile-r48-preprocess.py --data-root data/isafetybench --json
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, ROOT)
FIXTURE = os.path.join(ROOT, 'deployment', 'artifacts', 'r48-parity-fixture', 'fixture.npz')
FRAMES = 16
# A host run of this script is useful (the bit-exactness gate is cheap to check twice) but its
# timings are NOT device timings. Decided by the platform, not by a flag someone can forget.
IS_DEVICE = os.path.exists('/etc/nv_tegra_release')


def _load(name, fname):
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode(path, n):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def time_variant(fn, frames, mean, std, runs, warmup, pct):
    for _ in range(warmup):
        fn(frames, mean, std)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(frames, mean, std)
        ts.append((time.perf_counter() - t0) * 1000)
    return pct(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.path.join(ROOT, 'data', 'test'))
    ap.add_argument('--clip', default=None, help='default: first .mp4 found under --data-root')
    ap.add_argument('--runs', type=int, default=40)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    U = _load('jetson_measure_utils', 'jetson-measure-utils.py')
    V = _load('preprocess_variants_utils', 'preprocess-variants-utils.py')
    AGG = _load('benchmark_r48_cadence_utils', 'benchmark-r48-cadence-utils.py')
    pct = U.percentiles
    f = np.load(FIXTURE, allow_pickle=True)
    mean, std = f['mean'], f['std']

    path = args.clip or AGG.pick_clips(args.data_root, 1, 0)[0]
    frames = decode(path, FRAMES)
    if len(frames) < FRAMES:
        raise SystemExit(f'{path} has only {len(frames)} frames')
    h, w = frames[0].shape[:2]
    print(f'[profile] {os.path.basename(path)}  {w}x{h}  {FRAMES} frames  '
          f'{args.runs} runs x {args.repeats} repeats  cv2 threads {cv2.getNumThreads()}')
    print('[where]  ' + ('JETSON -- device numbers'
                         if IS_DEVICE else 'HOST -- timings are NOT device numbers'))

    ref = V.current(frames, mean, std)
    res = {'clip': os.path.basename(path), 'width': w, 'height': h, 'frames': FRAMES,
           'cv2_threads': cv2.getNumThreads(), 'cv2_version': cv2.__version__,
           'runs': args.runs, 'repeats': args.repeats, 'variants': {},
           'is_device_number': IS_DEVICE}
    for name, fn in V.VARIANTS.items():
        got = fn(frames, mean, std)
        same = bool(got.shape == ref.shape and np.array_equal(got, ref))
        md = 0.0 if same else float(np.abs(got.astype('float64') - ref.astype('float64')).max())
        reps = [time_variant(fn, frames, mean, std, args.runs, args.warmup, pct)
                for _ in range(args.repeats)]
        p50 = float(np.median([r['p50'] for r in reps]))
        res['variants'][name] = {'bit_identical': same, 'max_abs_diff': md,
                                 'p50_ms_per_16_frames': round(p50, 3),
                                 'p50_ms_per_frame': round(p50 / FRAMES, 3), 'repeats': reps,
                                 'diff_expected': name in V.EXPECT_DIFF}
    base = res['variants']['current']['p50_ms_per_16_frames']
    for name, v in res['variants'].items():
        v['speedup_vs_current'] = round(base / v['p50_ms_per_16_frames'], 3)
        gate = v['bit_identical'] and v['p50_ms_per_16_frames'] <= 0.9 * base
        v['adoptable'] = bool(gate and name != 'current')
        flag = 'BIT-IDENTICAL' if v['bit_identical'] else f"DIFFERS max {v['max_abs_diff']:.3e}"
        verdict = 'ADOPTABLE' if v['adoptable'] else ('baseline' if name == 'current' else 'no')
        print(f"  {name:20s} {v['p50_ms_per_16_frames']:8.3f} ms  "
              f"({v['p50_ms_per_frame']:6.3f} ms/frame)  {v['speedup_vs_current']:5.2f}x  "
              f"{flag:28s} {verdict}")

    ok = {k: v for k, v in res['variants'].items() if v['adoptable']}
    best = min(ok, key=lambda k: ok[k]['p50_ms_per_16_frames']) if ok else None
    res['recommended'] = best
    print(f"  -> adopt: {best or 'none -- no variant clears both legs of the rule'}")

    res['stage_split_ms'] = V.stage_split(frames, mean, std, max(8, args.runs // 2), pct)
    print('  --- inside `current`, per 16 frames ---')
    for k, v in res['stage_split_ms'].items():
        print(f"  {k:28s} p50 {v['p50']:8.3f} ms")
    if args.json:
        print(f"\n[SAVED] {U.write_json_result('r48-preprocess-profile', res, False)}")


if __name__ == '__main__':
    main()
