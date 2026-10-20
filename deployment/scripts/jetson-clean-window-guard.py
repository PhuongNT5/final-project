"""Run a measurement only inside a verified clean window, and refuse to report one taken outside it.

On 2026-10-01 two sessions shared this board and the CLIP tower measured 41.8 ms and 23.5 ms -- a
1.79x spread -- with byte-identical provenance in both JSONs. The lesson recorded then was that a
shared device needs a GUARD, not a pre-flight check: a check passes, the measurement starts, a
second session wakes up, and the number is already contaminated by the time anyone looks again.

So this wraps the measurement. It waits for the window, settles, re-reads the state, runs the
command, and re-reads the state once more. If the window was dirty at the START it never runs the
command. If it turned dirty by the END, the command's output is declared unquotable and the exit
code is non-zero -- the artifact may exist on disk, but this script's verdict is what licenses
quoting it.

Clocks must be UNPINNED (default DVFS). That is not a preference: the published 23.386 ms tower and
289.672 ms/decision cadence were taken under DVFS, and a pinned run is about 12.5% faster, so a
pinned measurement cannot be compared against them.

Usage (on the Jetson):
  python3 deployment/scripts/jetson-clean-window-guard.py --settle 90 -- \
      python3 deployment/scripts/benchmark-r48-clip-pipeline.py --backend tensorrt --mode fp16 \
          --tower r48-tower-clipB16 --json
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GPU_LOAD = '/sys/devices/platform/gpu.0/load'   # per-mille, 0..1000
GPU_BUSY_PERMILLE = 20                          # 2% of one sampling interval
# Any OTHER process touching this repo's measurement scripts, or trtexec, is contention.
# This is deliberately a substring of the PATH rather than a list of script names:
# jetson-clock-state-utils.competing_gpu_processes() matches a whitelist frozen on 2026-10-01,
# which cannot see an engine BUILD or any script written since -- it called a window "clean"
# while jetson-build-trt-engine.py was running. That function is left alone because published
# provenance fields record its output; the stronger test belongs here, where quoting is licensed.
PROC_MARKERS = (os.path.join('deployment', 'scripts'), 'trtexec')


def other_repo_processes():
    """-> other processes running this repo's device scripts, excluding our own process group."""
    try:
        own_group = os.getpgrp()
    except OSError:
        own_group = None
    found = []
    for d in glob.glob('/proc/[0-9]*'):
        pid = int(os.path.basename(d))
        if pid == os.getpid():
            continue
        try:
            if own_group is not None and os.getpgid(pid) == own_group:
                continue            # the guard itself, its shell, and the command it wraps
            with open(os.path.join(d, 'cmdline'), 'rb') as f:
                cmd = f.read().replace(b'\x00', b' ').decode('utf-8', 'replace').strip()
        except (OSError, ProcessLookupError):
            continue
        if cmd and any(m in cmd for m in PROC_MARKERS):
            found.append({'pid': pid, 'cmdline': cmd[:200]})
    return found


def gpu_busy_permille(samples=6, interval=0.5):
    """Peak GPU utilisation over a short window -- a name-independent contention signal.

    A single sample is not enough: an engine build drives the GPU in short bursts between long
    CPU-side tactic search, so an instantaneous read of 0 proves nothing. The peak does.
    """
    peak, read = 0, 0
    for i in range(samples):
        try:
            with open(GPU_LOAD) as f:
                peak, read = max(peak, int(f.read().strip())), read + 1
        except (OSError, ValueError):
            pass
        if i + 1 < samples:
            time.sleep(interval)
    return (peak if read else None), read


def read_state(CS):
    """The state block, with the harness's own warnings suppressed -- the verdict is printed here."""
    import io
    cpu, gpu = CS.cpu_clocks(), CS.gpu_clocks()
    competing = CS.competing_gpu_processes()
    others = other_repo_processes()
    peak, nread = gpu_busy_permille()
    return {'cpu_clocks': cpu, 'gpu_clocks': gpu, 'clocks_pinned': CS.clocks_pinned(cpu, gpu),
            'load_average': CS.load_average(), 'competing_processes': competing,
            'quiet': len(competing) == 0, 'other_repo_processes': others,
            'gpu_peak_permille': peak, 'gpu_load_samples': nread}, io.StringIO()


def verdict(state):
    """-> (clean, reasons). 'clean' means a number taken here is comparable to the published ones."""
    reasons = []
    if not state['quiet']:
        pids = ', '.join(str(p['pid']) for p in state['competing_processes'])
        reasons.append(f'another measurement process is running (pid {pids})')
    if (state['clocks_pinned'] or {}).get('any'):
        reasons.append('clocks are pinned (min == max); not the DVFS state the baselines used')
    others = state.get('other_repo_processes') or []
    if others:
        pids = ', '.join(str(p['pid']) for p in others)
        reasons.append(f'another process is using this repo\'s device scripts (pid {pids})')
    peak = state.get('gpu_peak_permille')
    if peak is None:
        reasons.append(f'{GPU_LOAD} is unreadable, so GPU contention cannot be ruled out')
    elif peak > GPU_BUSY_PERMILLE:
        reasons.append(f'GPU was {peak / 10:.1f}% busy while idle (> '
                       f'{GPU_BUSY_PERMILLE / 10:.1f}%); something else is on the device')
    la = state.get('load_average') or [0]
    if la[0] > 2.0:
        reasons.append(f'1-minute load average {la[0]:.2f} > 2.0')
    return not reasons, reasons


def wait_for_window(CS, timeout, poll):
    t0 = time.time()
    while True:
        state, _ = read_state(CS)
        ok, reasons = verdict(state)
        if ok:
            return state
        if time.time() - t0 > timeout:
            print(f'[guard] still dirty after {timeout}s: {"; ".join(reasons)}', file=sys.stderr)
            return None
        print(f'[guard] waiting ({int(time.time() - t0)}s): {"; ".join(reasons)}', flush=True)
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settle', type=float, default=90.0,
                    help='idle seconds between the window opening and the measurement starting')
    ap.add_argument('--timeout', type=float, default=900.0)
    ap.add_argument('--poll', type=float, default=10.0)
    ap.add_argument('--state-out', default=None, help='write the before/after state blocks here')
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == '--' else args.cmd
    if not cmd:
        raise SystemExit('nothing to run; put the command after --')

    CS = _load('clock_state', 'jetson-clock-state-utils.py')
    before = wait_for_window(CS, args.timeout, args.poll)
    if before is None:
        raise SystemExit('[guard] REFUSED to measure: the window never became clean')
    print(f'[guard] window clean; settling {args.settle:.0f}s before measuring', flush=True)
    time.sleep(args.settle)
    before, _ = read_state(CS)
    ok, reasons = verdict(before)
    if not ok:
        raise SystemExit(f'[guard] REFUSED to measure: window turned dirty during settle '
                         f'({"; ".join(reasons)})')

    print(f'[guard] running: {" ".join(cmd)}\n', flush=True)
    rc = subprocess.call(cmd)
    after, _ = read_state(CS)
    ok_after, reasons_after = verdict(after)
    if args.state_out:
        with open(args.state_out, 'w') as f:
            json.dump({'before': before, 'after': after, 'cmd': cmd, 'returncode': rc,
                       'clean_before': True, 'clean_after': ok_after,
                       'dirty_reasons_after': reasons_after}, f, indent=2)
        print(f'\n[guard] state -> {args.state_out}')
    if not ok_after:
        print(f'[guard] VERDICT: UNQUOTABLE -- the window turned dirty during the run '
              f'({"; ".join(reasons_after)}). Re-measure; do not report this output.',
              file=sys.stderr)
        raise SystemExit(3)
    print('[guard] VERDICT: clean before and after; this output may be quoted.')
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
