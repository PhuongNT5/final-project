"""The composed-cadence arithmetic, checked off-device.

The point of the composed measurement is that it must NOT agree with a stage sum by construction.
These tests pin the three things that would silently turn it back into a sum: the cadence must be
wall clock over decisions emitted (so buffer-fill decode is charged), the duty cycle must be
compute over the same wall clock, and a clip that never fills its window must contribute zero
decisions instead of dividing by zero.
"""
import os
from importlib.util import module_from_spec, spec_from_file_location

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'deployment', 'scripts')


def _load(name, fname):
    spec = spec_from_file_location(name, os.path.join(SCRIPTS, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CAD = _load('benchmark_r48_cadence_utils', 'benchmark-r48-cadence-utils.py')
PCT = _load('jetson_measure_utils', 'jetson-measure-utils.py').percentiles


def _clip(n_decisions, compute_ms, decode_ms, n_frames, wall_sec, gap_sec=0.15, t0=0.5):
    """One clip's record. emit_sec is spaced by gap_sec so the steady-state interval is known
    exactly, and starts at t0 so the window-fill delay before the first decision is represented."""
    return {'clip': 'x.mp4', 'frames_decoded': n_frames, 'wall_sec': wall_sec,
            'decode_ms': [decode_ms] * n_frames,
            'decisions': [{'prep_ms': 1.0, 'tower_ms': compute_ms - 1.4, 'head_ms': 0.4,
                           'compute_ms': compute_ms, 'p_unsafe': 0.5,
                           'emit_sec': t0 + i * gap_sec}
                          for i in range(n_decisions)]}


def test_sample_window_takes_sixteen_frames_spanning_the_buffer():
    buf = list(range(48))
    got = CAD.sample_window(buf)
    assert len(got) == 16
    assert got[0] == 0 and got[-1] == 47      # the window's ends are both represented
    assert got == sorted(got)                  # temporal order preserved for the temporal head


def test_sample_window_handles_a_buffer_exactly_sixteen_long():
    assert CAD.sample_window(list(range(16))) == list(range(16))


def test_cadence_is_wall_clock_over_decisions_not_the_stage_sum():
    # 2 clips, 10 decisions each, 5 s wall each -> 10 s / 20 decisions = 500 ms per decision.
    clips = [_clip(10, 25.0, 5.0, 300, 5.0), _clip(10, 25.0, 5.0, 300, 5.0)]
    s = CAD.summarise(clips, 48, 24, PCT)
    assert s['n_decisions'] == 20
    assert s['measured_ms_per_decision'] == 500.0
    assert s['measured_decisions_per_sec'] == 2.0


def test_duty_cycle_is_compute_over_the_same_wall_clock():
    clips = [_clip(10, 25.0, 5.0, 300, 5.0)]
    s = CAD.summarise(clips, 48, 24, PCT)
    assert s['duty_cycle_compute'] == round((10 * 25.0 / 1000.0) / 5.0, 4)


def test_stage_sum_is_reported_and_differs_from_the_measurement():
    clips = [_clip(10, 25.0, 5.0, 300, 5.0)]
    s = CAD.summarise(clips, 48, 24, PCT)
    assert s['stage_sum_ms_per_decision'] == 5.0 * 24 + 25.0        # 145.0
    # the whole reason the script exists: the sum under-reports, and the ratio says by how much
    assert s['measured_over_stage_sum'] == round(500.0 / 145.0, 3)
    assert s['measured_over_stage_sum'] > 1.0


def test_a_clip_that_never_fills_its_window_contributes_no_decisions():
    s = CAD.summarise([_clip(0, 25.0, 5.0, 12, 0.4)], 48, 24, PCT)
    assert s['n_decisions'] == 0
    assert s['measured_ms_per_decision'] is None       # not a ZeroDivisionError, not 0.0
    assert s['measured_decisions_per_sec'] is None     # not 0.0: nothing emitted is not a rate
    assert s['duty_cycle_compute'] is None            # nor is it a measured duty of zero
    assert s['stage_sum_ms_per_decision'] is None     # no compute samples -> no sum to quote
    assert s['measured_over_stage_sum'] is None
    assert s['frames_decoded'] == 12


def test_the_artifact_declares_what_it_did_not_measure():
    s = CAD.summarise([_clip(10, 25.0, 5.0, 300, 5.0)], 48, 24, PCT)
    assert s['overlapped'] is False          # no decode/inference overlap -> throughput lower bound
    assert s['detector_in_loop'] is False    # localisation is timed separately, never summed in
    assert s['is_device_number'] is True
    assert s['window'] == 48 and s['stride'] == 24


def test_pick_clips_is_deterministic_and_spans_label_directories(tmp_path):
    for label in ('hazard', 'normal'):
        d = tmp_path / label
        os.makedirs(str(d))
        for i in range(6):
            (d / f'{label}{i}.mp4').write_bytes(b'')
    a = CAD.pick_clips(str(tmp_path), 8, seed=0)
    b = CAD.pick_clips(str(tmp_path), 8, seed=0)
    assert a == b and len(a) == 8
    assert len({os.path.basename(os.path.dirname(p)) for p in a}) == 2


def test_pick_clips_refuses_an_empty_corpus(tmp_path):
    try:
        CAD.pick_clips(str(tmp_path), 4, seed=0)
    except SystemExit as e:
        assert 'no .mp4' in str(e)
    else:
        raise AssertionError('an empty corpus must abort, not measure zero clips')


def test_report_does_not_crash_when_nothing_was_measured(capsys):
    """The empty case must print a refusal, not raise a format error on None."""
    CAD.report(CAD.summarise([_clip(0, 25.0, 5.0, 12, 0.4)], 48, 24, PCT))
    assert 'nothing to quote as a cadence' in capsys.readouterr().out


def test_steady_state_is_the_interval_between_consecutive_decisions():
    """150 ms apart by construction -- and NOT the 500 ms that wall/n reports for the same clips."""
    clips = [_clip(10, 25.0, 5.0, 300, 5.0, gap_sec=0.15), _clip(10, 25.0, 5.0, 300, 5.0,
                                                                 gap_sec=0.15)]
    s = CAD.summarise(clips, 48, 24, PCT)
    ss = s['steady_state_ms_per_decision']
    assert ss['p50'] == 150.0
    assert ss['n'] == 18                      # 9 intervals per clip, never across the clip boundary
    assert s['measured_ms_per_decision'] == 500.0


def test_the_window_fill_gap_is_not_counted_as_a_cadence_interval():
    """t0 is the fill delay. Charging it as an interval would inflate the steady-state figure."""
    s = CAD.summarise([_clip(3, 25.0, 5.0, 151, 1.0, gap_sec=0.15, t0=9.9)], 48, 24, PCT)
    assert s['steady_state_ms_per_decision']['p50'] == 150.0
    assert s['steady_state_ms_per_decision']['n'] == 2


def test_a_single_decision_yields_no_interval():
    s = CAD.summarise([_clip(1, 25.0, 5.0, 60, 0.4)], 48, 24, PCT)
    assert s['steady_state_ms_per_decision']['p50'] is None
    assert s['n_decisions'] == 1


def test_the_sum_is_compared_against_the_steady_state_not_the_fill_inflated_figure():
    """Both sides of this ratio are per-decision steady figures, so it is the one that actually
    tests whether a stage sum is a usable proxy for the composed cadence."""
    s = CAD.summarise([_clip(10, 25.0, 5.0, 300, 5.0, gap_sec=0.15)], 48, 24, PCT)
    assert s['stage_sum_ms_per_decision'] == 145.0
    assert s['steady_state_over_stage_sum'] == round(150.0 / 145.0, 3)
    assert s['measured_over_stage_sum'] == round(500.0 / 145.0, 3)   # kept, but not the comparison


def test_report_prints_both_ratios_and_survives_a_missing_interval(capsys):
    CAD.report(CAD.summarise([_clip(1, 25.0, 5.0, 60, 0.4)], 48, 24, PCT))
    out = capsys.readouterr().out
    assert 'no interval' in out and 'incl-fill / sum' in out


# --- the idle-droop discriminator -------------------------------------------------------------
# probe_idle exists to decide WHY the tower costs ~2,5x more in the loop than back to back. It is
# a device loop, but the arms, the ordering and the reported ratio are what the verdict rests on,
# and those are checkable with a stub engine. Rule in
# reports/finding-2026-10-01-stage-sum-is-not-a-cadence.md.
CAD_SCRIPT = _load('benchmark_r48_composed_cadence', 'benchmark-r48-composed-cadence.py')


class _StubTower:
    """Reports a fixed cost per call, and a different one when the caller slept first -- the
    signature the real droop produces."""

    def __init__(self, busy_ms=1.0, after_sleep_ms=3.0):
        self.busy, self.after_sleep, self.calls, self.slept = busy_ms, after_sleep_ms, 0, False

    def run(self, **kw):
        import time as _t
        self.calls += 1
        _t.sleep((self.after_sleep if self.slept else self.busy) / 1000.0)
        return None


def _probe(busy_ms=1.0, after_sleep_ms=4.0, gap_ms=2.0, runs=5):
    """Drives probe_idle with a stub whose cost depends on whether the gap was taken. The stub
    learns that from the gap itself: probe_idle sleeps before the call in arm B only."""
    tower = _StubTower(busy_ms, after_sleep_ms)
    real_sleep = CAD_SCRIPT.time.sleep

    def spy(sec):
        tower.slept = True
        real_sleep(sec)

    CAD_SCRIPT.time.sleep = spy
    try:
        return CAD_SCRIPT.probe_idle(tower, None, gap_ms, runs, 2, PCT), tower
    finally:
        CAD_SCRIPT.time.sleep = real_sleep


def test_probe_reports_both_arms_and_their_ratio():
    out, _ = _probe()
    assert 'back_to_back' in out and 'idle_2ms' in out
    assert out['idle_over_back_to_back'] == round(
        out['idle_2ms']['p50'] / out['back_to_back']['p50'], 3)


def test_probe_runs_the_arms_in_order_so_the_idle_arm_is_never_the_warm_one():
    """Arm A first, arm B second. Reversed, arm A would inherit arm B's idle state and the ratio
    would be measured against an already-drooped baseline."""
    out, _ = _probe()
    assert list(out)[:2] == ['back_to_back', 'idle_2ms']


def test_probe_warms_up_before_each_arm_and_times_only_the_measured_runs():
    runs, warmup = 5, 2
    out, tower = _probe(runs=runs)
    assert tower.calls == 2 * (runs + warmup)
    assert out['back_to_back']['n'] == runs and out['idle_2ms']['n'] == runs


def test_probe_excludes_the_gap_from_the_timed_span():
    """The gap must be slept BEFORE t0, not inside it: otherwise the probe would 'find' droop of
    exactly the gap size on any engine at all."""
    out, _ = _probe(busy_ms=1.0, after_sleep_ms=1.0, gap_ms=50.0, runs=4)
    assert out['idle_50ms']['p50'] < 25.0


def test_probe_records_the_in_loop_figure_it_is_being_compared_against():
    out, _ = _probe()
    assert out['in_loop_tower_p50_ms'] == 58.630
    assert out['idle_gap_ms'] == 2.0
    assert out['is_device_number'] is True
