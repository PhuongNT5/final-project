"""jetson-measure-power.py — energy (J/inference, FPS/W), 30-min thermal soak, and host/device
memory, all driven off tegrastats + whatever engines jetson-measure-latency.discover_engines()
finds. Real subprocess/psutil telemetry only — --dry-run skips hardware-only numbers instead of
fabricating them (see jetson-measure-utils.TegrastatsSampler docstring).

Energy is reported TWO ways because they answer different questions and the thesis must say
which it quotes: `*_total_board` is raw VDD_IN (what the deployed box draws) and `*_delta_idle`
subtracts the idle baseline per the project spec (plans/2026-09-19-thesis-recovery-architecture/
research/researcher-03-demo-and-edge-deployment.md §5-6: `J/inf = (P_avg_W - P_idle_W) x t`).
Neither is recoverable from the other after the fact, so both are measured in the same run.
"""
import os
import resource
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

ENERGY_WARMUP = 20    # inferences discarded before the measured window opens
ENERGY_IDLE_SEC = 10  # idle baseline: engine resident, no work submitted
SOAK_WARMUP = 20
THROTTLE_RATIO = 1.15


def _load(name, fname):
    spec = spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load('jetson_measure_utils', 'jetson-measure-utils.py')
LAT = _load('jetson_measure_latency', 'jetson-measure-latency.py')

_ENERGY_KEYS = ('n_inferences', 'fps', 'avg_power_w', 'idle_power_w', 'delta_power_w',
                'joules_per_inference_total_board', 'joules_per_inference_delta_idle',
                'fps_per_watt_total_board', 'fps_per_watt_delta_idle')


def _avg_power_mw(tegra_samples):
    vals = [s['power_now_mw'] for s in tegra_samples if 'power_now_mw' in s]
    return float(np.mean(vals)) if vals else None


def _idle_power_w(idle_sec, interval_ms=500):
    """Board draw with the engine resident but no work submitted — the baseline the spec's
    marginal (idle-subtracted) energy figure is defined against."""
    s = U.TegrastatsSampler(interval_ms=interval_ms)
    s.start()
    time.sleep(idle_sec)
    t = s.stop()
    mw = _avg_power_mw(t['samples'])
    return (mw / 1000.0 if mw is not None else None), t['n_samples']


def _build(trt, cuda, engine_path, warmup):
    """Deserialise + warm up. Callers must do this BEFORE starting any clock or sampler: engine
    deserialisation on Orin Nano takes seconds, and counting it inside the measured window
    deflates fps/avg_power and inflates J/inference by that dead time."""
    runner = LAT.TRU.TRTRunner(trt, cuda, engine_path)
    fixed_in = LAT._fixed_inputs(runner)
    for _ in range(warmup):
        runner.run(**fixed_in)
    return runner, fixed_in


def run_energy(engine_dir, mode, stage, duration_sec, dry_run, idle_sec=ENERGY_IDLE_SEC):
    """Runs one engine in a tight loop for `duration_sec` while sampling tegrastats power.

    UNITS: `n_inferences` counts invocations of THIS ONE engine. A P3 decision is not one
    invocation — it is DECISION_T_FRAMES detector passes + 1 batched appearance pass + 1 head
    pass (see jetson-measure-latency.run_decision_latency). Per-decision energy must be composed
    from per-stage runs exactly the way per-decision latency is; never quote a single-stage
    joules_per_inference as joules-per-decision.
    """
    result = {'mode': mode, 'stage': stage, 'duration_sec_requested': duration_sec,
              'what': 'per-invocation energy for ONE engine; NOT per-decision'}
    stages = LAT.discover_engines(engine_dir)
    if stage not in stages or mode not in stages.get(stage, {}):
        result['skipped_reason'] = f'engine not found: {stage}-{mode}.engine in {engine_dir}'
        return result
    trt, cuda = (None, None) if dry_run else LAT._import_trt()
    if trt is None:
        result['skipped_reason'] = (
            'dry-run: tensorrt/pycuda not available' if dry_run else 'tensorrt import failed')
        result.update({k: None for k in _ENERGY_KEYS})
        s = U.TegrastatsSampler(interval_ms=500)   # still exercise the sampler start/stop
        s.start()                                  # plumbing (documented --dry-run contract)
        time.sleep(min(duration_sec, 2))
        t = s.stop()
        result['tegrastats'] = {'dry_run': t['dry_run'], 'n_samples': t['n_samples']}
        return result
    runner, fixed_in = _build(trt, cuda, stages[stage][mode], ENERGY_WARMUP)
    idle_w, idle_n = _idle_power_w(idle_sec)
    sampler = U.TegrastatsSampler(interval_ms=500)
    sampler.start()
    t0 = time.perf_counter()
    n = 0
    t_end = time.time() + duration_sec
    while time.time() < t_end:
        runner.run(**fixed_in)
        n += 1
    elapsed = time.perf_counter() - t0
    tegra = sampler.stop()
    avg_mw = _avg_power_mw(tegra['samples'])
    result.update({k: None for k in _ENERGY_KEYS})
    result.update({'duration_sec_measured': round(elapsed, 3), 'n_inferences': n,
                   'warmup_inferences': ENERGY_WARMUP, 'idle_baseline_sec': idle_sec,
                   'idle_n_samples': idle_n,
                   'idle_power_w': round(idle_w, 3) if idle_w is not None else None,
                   'fps': round(n / elapsed, 3) if n and elapsed > 0 else None})
    if avg_mw is None or not n or elapsed <= 0:
        result.setdefault('skipped_reason', 'no tegrastats power samples in the load window')
    else:
        avg_w, fps = avg_mw / 1000.0, n / elapsed
        result['avg_power_w'] = round(avg_w, 3)
        result['joules_per_inference_total_board'] = round(avg_w * elapsed / n, 5)
        result['fps_per_watt_total_board'] = round(fps / avg_w, 4)
        delta_w = None if idle_w is None else avg_w - idle_w
        result['delta_power_w'] = None if delta_w is None else round(delta_w, 3)
        if delta_w is not None and delta_w > 0:
            result['joules_per_inference_delta_idle'] = round(delta_w * elapsed / n, 5)
            result['fps_per_watt_delta_idle'] = round(fps / delta_w, 4)
        else:
            # Not clamped to a positive number: an idle baseline at or above the load draw means
            # the baseline is invalid (contention, or the loop never reached the GPU).
            result['delta_idle_skipped_reason'] = (
                'no idle baseline (tegrastats gave no power samples)' if idle_w is None else
                f'load draw {avg_w:.3f}W not above idle {idle_w:.3f}W — baseline invalid')
    result['tegrastats'] = {'dry_run': tegra['dry_run'], 'n_samples': tegra['n_samples'],
                            'power_rail': 'VDD_IN (total module input, not a GPU-only rail)'}
    return result


def run_thermal_soak(engine_dir, mode, stage, duration_min, window_sec, dry_run):
    """Runs `stage` continuously for duration_min minutes; every window_sec, records a latency
    percentile window + concurrent tegrastats sample. possible_throttling is a heuristic (last
    window p50 > THROTTLE_RATIO x the FASTEST window p50) — real throttling confirmation needs
    actual GPU clock (not just utilisation%), which tegrastats doesn't expose as raw Hz on all
    JetPack versions. Baseline is the fastest window, not the first: the engine is warmed before
    window 0, but clocks still ramp, and a slow first window would hide real drift.
    """
    result = {'mode': mode, 'stage': stage, 'duration_min': duration_min, 'windows': []}
    stages = LAT.discover_engines(engine_dir)
    if stage not in stages or mode not in stages.get(stage, {}):
        result['skipped_reason'] = f'engine not found: {stage}-{mode}.engine'
        return result
    trt, cuda = (None, None) if dry_run else LAT._import_trt()
    if trt is None:
        result['skipped_reason'] = (
            'dry-run: tensorrt/pycuda not available' if dry_run else 'tensorrt import failed')
        return result
    runner, fixed_in = _build(trt, cuda, stages[stage][mode], SOAK_WARMUP)
    sampler = U.TegrastatsSampler(interval_ms=1000)
    sampler.start()
    t_start = time.time()
    t_end = t_start + duration_min * 60
    while time.time() < t_end:
        w_end = min(time.time() + window_sec, t_end)
        ts = []
        while time.time() < w_end:
            t0 = time.perf_counter()
            runner.run(**fixed_in)
            ts.append((time.perf_counter() - t0) * 1000)
        result['windows'].append({'t_offset_sec': round(time.time() - t_start, 1),
                                  'latency_ms': U.percentiles(ts)})
    result['tegrastats'] = sampler.stop()
    result['warmup_inferences'] = SOAK_WARMUP
    p50s = [w['latency_ms']['p50'] for w in result['windows'] if w['latency_ms']['p50']]
    if len(p50s) >= 2:
        best = min(p50s)
        result['baseline_p50_ms'] = best
        result['baseline_window_index'] = p50s.index(best)
        result['possible_throttling'] = bool(p50s[-1] > best * THROTTLE_RATIO)
        result['latency_drift_pct_vs_fastest'] = round((p50s[-1] / best - 1) * 100, 2)
        result['latency_drift_pct_vs_first'] = round((p50s[-1] / p50s[0] - 1) * 100, 2)
    return result


def run_memory(engine_dir, mode, dry_run):
    """Device memory attributable to the engine set: free-before minus free-after loading every
    discovered engine for `mode`. `device_used_mb` alone is NOT a model footprint — Orin Nano
    memory is unified and shared with the OS, so it counts the whole system."""
    result = {'mode': mode}
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['host_peak_rss_mb'] = round(ru / (1024.0 if sys.platform != 'darwin' else 1024.0 * 1024.0), 3)
    trt, cuda = (None, None) if dry_run else LAT._import_trt()
    if trt is None:
        # Real schema (success path below) is device_free_mb/device_total_mb/device_used_mb —
        # null every one of those on skip, not a dead placeholder key (code-review fix,
        # 2026-09-16 report: dry-run branch previously set the never-produced
        # `device_free_total_mb`, leaving the real 3 keys entirely absent from skipped JSON).
        result['device_free_mb'] = result['device_total_mb'] = result['device_used_mb'] = None
        result['engines_resident_delta_mb'] = result['engines_loaded'] = None
        result['skipped_reason'] = 'dry-run: no CUDA device on this machine' if dry_run else 'tensorrt import failed'
        return result
    free0, total = cuda.mem_get_info()
    stages = LAT.discover_engines(engine_dir)
    runners, loaded = [], []
    for st in sorted(stages):
        if mode in stages[st]:
            runners.append(LAT.TRU.TRTRunner(trt, cuda, stages[st][mode]))  # held resident
            loaded.append(st)
    free1, _ = cuda.mem_get_info()
    mb = 1024 * 1024
    result['device_total_mb'] = round(total / mb, 2)
    result['device_free_mb'] = round(free1 / mb, 2)
    result['device_used_mb'] = round((total - free1) / mb, 2)
    result['device_used_note'] = 'system-wide (unified memory shared with OS), not a model footprint'
    result['engines_loaded'] = loaded
    result['n_engines_resident'] = len(runners)
    result['engines_resident_delta_mb'] = round((free0 - free1) / mb, 2)
    return result
