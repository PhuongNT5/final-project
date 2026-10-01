"""Pure aggregation for the composed B1 cadence measurement -- no CUDA, no device required.

Split out of benchmark-r48-composed-cadence.py so the arithmetic that decides what gets printed as
a cadence is testable on the host. The script that owns the device loop stays thin.
"""
import glob
import os
import random

import numpy as np

FRAMES = 16          # one decision = 16 frames; the evaluation's sampling rule, not a tuning knob


def pick_clips(data_root, n, seed):
    """Sample clips from both label directories, so the decode cost is not measured on one
    resolution or one source's encoder settings."""
    paths = sorted(glob.glob(os.path.join(data_root, '*', '*.mp4')))
    if not paths:
        raise SystemExit(f'no .mp4 under {data_root}')
    random.Random(seed).shuffle(paths)
    return paths[:n]


def sample_window(buf):
    """FRAMES frames spread across the ring buffer -- the rule demo/run-unsafe-behavior-demo.py
    sample() and the evaluation both use. Indexing the deque, not re-seeking the file: a seek is
    5.8 s per window on this board and belongs to the offline path only."""
    return [buf[int(i)] for i in np.linspace(0, len(buf) - 1, FRAMES)]


def summarise(clips, window, stride, pct):
    """Aggregate across clips into the two cadences and the sum they are compared against.

    `steady_state` is the interval between consecutive decisions within a clip -- the deployed
    cadence, and the figure to compare against the stage sum. `measured_ms_per_decision` is wall
    clock over decisions emitted, so it also charges the `window` frames decoded before the first
    decision; on short clips that fill dominates, which is why both are reported."""
    dec = [d for c in clips for d in c['decode_ms']]
    comp = [d['compute_ms'] for c in clips for d in c['decisions']]
    n = len(comp)
    wall = sum(c['wall_sec'] for c in clips)
    frames = sum(c['frames_decoded'] for c in clips)
    busy = sum(comp) / 1000.0
    gaps = [1000.0 * (d['emit_sec'] - prev['emit_sec'])
            for c in clips for prev, d in zip(c['decisions'], c['decisions'][1:])]
    d50, c50 = pct(dec)['p50'], pct(comp)['p50']
    # A clip shorter than one window emits nothing; report None rather than inventing a cadence.
    sum_pred = None if d50 is None or c50 is None else d50 * stride + c50
    mps = round(1000.0 * wall / n, 3) if n else None
    g = pct(gaps)
    ss50 = g['p50']
    return {
        'n_clips': len(clips), 'n_decisions': n, 'frames_decoded': frames,
        'window': window, 'stride': stride,
        'wall_sec_total': round(wall, 3),
        'decode_ms_per_frame': pct(dec),
        'prep_ms': pct([d['prep_ms'] for c in clips for d in c['decisions']]),
        'tower_ms': pct([d['tower_ms'] for c in clips for d in c['decisions']]),
        'head_ms': pct([d['head_ms'] for c in clips for d in c['decisions']]),
        'compute_ms_per_decision': pct(comp),
        'steady_state_ms_per_decision': g,
        'measured_ms_per_decision': mps,
        'measured_decisions_per_sec': round(n / wall, 3) if n and wall else None,
        'duty_cycle_compute': round(busy / wall, 4) if n and wall else None,
        'stage_sum_ms_per_decision': None if sum_pred is None else round(sum_pred, 3),
        'measured_over_stage_sum': round(mps / sum_pred, 3) if mps and sum_pred else None,
        # the comparison that actually tests the sum: both sides are per-decision steady figures
        'steady_state_over_stage_sum': round(ss50 / sum_pred, 3) if ss50 and sum_pred else None,
        'overlapped': False, 'detector_in_loop': False, 'is_device_number': True,
    }


def report(s):
    print(f"\n  clips {s['n_clips']}   decisions {s['n_decisions']}   "
          f"frames decoded {s['frames_decoded']}   window {s['window']} stride {s['stride']}")
    if not s['n_decisions']:
        print('  no decision was emitted -- every clip was shorter than one window; nothing to '
              'quote as a cadence')
        return
    print(f"  decode            p50 {s['decode_ms_per_frame']['p50']:7.3f} ms/frame")
    print(f"  preprocess        p50 {s['prep_ms']['p50']:7.3f} ms   "
          f"tower p50 {s['tower_ms']['p50']:7.3f} ms   head p50 {s['head_ms']['p50']:6.3f} ms")
    print(f"  compute/decision  p50 {s['compute_ms_per_decision']['p50']:7.3f} ms   "
          f"p95 {s['compute_ms_per_decision']['p95']:7.3f} ms")
    ss = s['steady_state_ms_per_decision']
    if ss['p50'] is not None:
        print(f"  STEADY-STATE cadence  {ss['p50']:7.3f} ms/decision  p95 {ss['p95']:7.3f}  -> "
              f"{1000 / ss['p50']:.3f} decisions/s   (n={ss['n']} intervals)")
    print(f"  incl. window fill     {s['measured_ms_per_decision']:7.3f} ms/decision  -> "
          f"{s['measured_decisions_per_sec']:.3f} decisions/s   "
          f"duty {100 * s['duty_cycle_compute']:.1f}%")
    r = s['steady_state_over_stage_sum']
    ratio = 'no interval' if r is None else f'steady-state / sum = {r:.3f}'
    print(f"  stage sum would say   {s['stage_sum_ms_per_decision']:7.3f} ms  "
          f"({ratio}, incl-fill / sum = {s['measured_over_stage_sum']:.3f})")
    print('  single-threaded: no decode/inference overlap, so this is a throughput LOWER bound')
