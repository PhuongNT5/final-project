"""4a timing anchors: real wall-clock measurement via time.perf_counter, checked
against deterministic sleep durations (not a fake/mocked evaluation number — this
tests the TIMING UTILITY's correctness, the actual model timing is done by callers)."""
import time
import numpy as np

from eval_utils.timing import time_per_clip, frame_gate_reduced_duration, PAPER_ACTION_TIME_S


def test_time_per_clip_measures_real_elapsed_time():
    delay = 0.02
    n = 5
    out = time_per_clip(lambda c: time.sleep(delay), list(range(n)))
    assert out['n'] == n
    # perf_counter is monotonic and precise; allow generous scheduler slack
    assert delay * 0.5 < out['mean_s'] < delay * 5, out['mean_s']
    assert out['paper_mark_s'] == PAPER_ACTION_TIME_S
    assert len(out['per_clip_s']) == n


def test_time_per_clip_zero_cost_function():
    out = time_per_clip(lambda c: None, list(range(3)))
    assert out['n'] == 3
    assert out['mean_s'] >= 0.0


def test_frame_gate_reduced_duration_hand_computed():
    # 3 clips, fps=25: before=[50,75,100] frames -> [2.0,3.0,4.0]s; after=[25,25,25] -> 1.0s each
    before = [50, 75, 100]
    after = [25, 25, 25]
    out = frame_gate_reduced_duration(before, after, fps=25)
    assert abs(out['before']['mean_s'] - 3.0) < 1e-9
    assert abs(out['after']['mean_s'] - 1.0) < 1e-9
    assert out['after']['paper_mark_s'] == 1.87
    # reduction fraction = 1 - sum(after)/sum(before) = 1 - 75/225 = 2/3
    assert abs(out['mean_reduction_fraction'] - 2 / 3) < 1e-9


def test_frame_gate_length_mismatch_raises():
    try:
        frame_gate_reduced_duration([1, 2, 3], [1, 2], fps=25)
        assert False, 'expected ValueError'
    except ValueError:
        pass


if __name__ == '__main__':
    import sys
    from eval_utils.tests.run_all import run_module
    sys.exit(run_module(sys.modules[__name__]))
