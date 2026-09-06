"""Paper-anchor timing helpers (brief S4.4a).

Paper marks: action-recognition time 0.14 s/clip; average reduced clip duration
1.87 s after the frame-reduction gate. These helpers measure wall-clock time with
`time.perf_counter()` (monotonic, sub-microsecond resolution) — they do not
simulate or hardcode a result; the caller supplies the real per-clip callable /
real frame counts from an actual run.
"""
from __future__ import annotations
import time
import numpy as np

PAPER_ACTION_TIME_S = 0.14
PAPER_REDUCED_DURATION_S = 1.87


def _summary(values, paper_mark=None):
    v = np.asarray(values, dtype=float)
    out = {
        'n': int(len(v)),
        'mean_s': float(v.mean()) if len(v) else float('nan'),
        'median_s': float(np.median(v)) if len(v) else float('nan'),
        'p95_s': float(np.percentile(v, 95)) if len(v) else float('nan'),
        'std_s': float(v.std()) if len(v) else float('nan'),
    }
    if paper_mark is not None:
        out['paper_mark_s'] = paper_mark
        out['delta_vs_paper_s'] = out['mean_s'] - paper_mark
    return out


def time_per_clip(fn, clips):
    """Call fn(clip) once per item in `clips`, timing each call with
    time.perf_counter(). Returns per-clip durations (s) + summary stats compared
    against the paper's 0.14 s/clip action-recognition mark. `fn` must be the real
    inference/decision callable — no synthetic timings."""
    durations = []
    for clip in clips:
        t0 = time.perf_counter()
        fn(clip)
        durations.append(time.perf_counter() - t0)
    out = _summary(durations, paper_mark=PAPER_ACTION_TIME_S)
    out['per_clip_s'] = [float(d) for d in durations]
    return out


def frame_gate_reduced_duration(frame_counts_before, frame_counts_after, fps):
    """Average clip duration before/after a frame-reduction gate, from real per-clip
    frame counts (e.g. len(all_frames) vs len(kept_frames)) and the clip's fps.
    Compared against the paper's 1.87 s reduced-duration mark."""
    before = np.asarray(frame_counts_before, dtype=float) / float(fps)
    after = np.asarray(frame_counts_after, dtype=float) / float(fps)
    if len(before) != len(after):
        raise ValueError('frame_counts_before/after must have the same length (per-clip pairs)')
    reduction_frac = 1.0 - (after.sum() / before.sum()) if before.sum() else 0.0
    return {
        'before': _summary(before),
        'after': _summary(after, paper_mark=PAPER_REDUCED_DURATION_S),
        'mean_reduction_fraction': float(reduction_frac),
    }
