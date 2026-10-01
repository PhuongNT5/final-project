"""jetson-clock-state-utils.py — read the device's real clock and contention state from sysfs.

Why this exists: on 2026-10-01 two measurement sessions shared one Orin Nano. One run was taken
while a concurrent 60 s energy sweep saturated the GPU; another was taken while a concurrent
session had pinned every clock with `jetson_clocks`. The CLIP tower measured 41.8 ms and 23.5 ms
respectively -- a 1.8x spread -- and *both* JSONs recorded byte-identical provenance, because
`nvpmodel -q` still reads MAXN_SUPER under a clock pin and `jetson_clocks --show` needs root and
so was captured as null. A latency number whose device state cannot be reconstructed is not a
measurement, so the state is read here from sysfs, which needs no privileges.

Everything degrades to None off-device (no sysfs nodes), never to a fabricated value.
"""
import glob
import os
import sys

CPU_POLICY_GLOB = '/sys/devices/system/cpu/cpufreq/policy*'
GPU_DEVFREQ_GLOB = '/sys/devices/platform/bus@0/*.gpu/devfreq/*'
PROC_GLOB = '/proc/[0-9]*'


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _read_str(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def cpu_clocks(policy_glob=CPU_POLICY_GLOB):
    """-> per-policy {min,cur,max} kHz. Empty off-device."""
    out = {}
    for d in sorted(glob.glob(policy_glob)):
        out[os.path.basename(d)] = {
            'min_khz': _read_int(os.path.join(d, 'scaling_min_freq')),
            'cur_khz': _read_int(os.path.join(d, 'scaling_cur_freq')),
            'max_khz': _read_int(os.path.join(d, 'scaling_max_freq')),
            'governor': _read_str(os.path.join(d, 'scaling_governor')),
        }
    return out


def gpu_clocks(devfreq_glob=GPU_DEVFREQ_GLOB):
    """-> per-devfreq-node {min,cur,max} Hz. Empty off-device."""
    out = {}
    for d in sorted(glob.glob(devfreq_glob)):
        out[os.path.basename(d)] = {
            'min_hz': _read_int(os.path.join(d, 'min_freq')),
            'cur_hz': _read_int(os.path.join(d, 'cur_freq')),
            'max_hz': _read_int(os.path.join(d, 'max_freq')),
            'governor': _read_str(os.path.join(d, 'governor')),
        }
    return out


def _all_pinned(nodes, lo, hi):
    """True only when every node reports min == max, i.e. DVFS has no range left to scale in.

    Returns None when there is nothing to judge, so 'not measured' is never reported as 'not
    pinned' -- that conflation is what made the 22:25 run look like the 22:18 run.
    """
    vals = [(n[lo], n[hi]) for n in nodes.values() if n[lo] is not None and n[hi] is not None]
    if not vals:
        return None
    return all(a == b for a, b in vals)


def clocks_pinned(cpu, gpu):
    """-> {'cpu':bool|None, 'gpu':bool|None, 'any':bool} — 'any' is the flag to filter runs on.

    `jetson_clocks` pins by raising every min_freq to max_freq, so min == max is the signature,
    and it is visible without root and without trusting a wrapper binary to exist.
    """
    c, g = _all_pinned(cpu, 'min_khz', 'max_khz'), _all_pinned(gpu, 'min_hz', 'max_hz')
    return {'cpu': c, 'gpu': g, 'any': bool(c) or bool(g)}


def load_average():
    try:
        return list(os.getloadavg())
    except OSError:
        return None


def competing_gpu_processes(own_pid=None, proc_glob=PROC_GLOB):
    """-> list of other processes that look like they are holding the GPU for measurement.

    Deliberately name-based over /proc rather than nvidia-smi: Orin's integrated GPU exposes no
    per-process accounting, so there is no authoritative list. A name match is a warning signal,
    not a guarantee -- the point is that a contended run is flagged at all instead of being
    averaged into the headline.
    """
    own_pid = os.getpid() if own_pid is None else own_pid
    markers = ('jetson-full-measurement', 'jetson-measure-latency', 'jetson-measure-power',
               'benchmark-r48-clip-pipeline', 'benchmark-p3', 'verify-r48-device-parity',
               'trtexec', 'jetson-benchmark')
    found = []
    for d in glob.glob(proc_glob):
        pid = int(os.path.basename(d))
        if pid in (own_pid, os.getppid()):
            continue
        cmd = _read_str(os.path.join(d, 'cmdline'))
        if not cmd:
            continue
        cmd = cmd.replace('\x00', ' ').strip()
        if any(m in cmd for m in markers):
            found.append({'pid': pid, 'cmdline': cmd[:200]})
    return found


def warn_if_unreliable(state, stream=sys.stderr):
    """Say out loud, at measurement time, that this run cannot be quoted. Silence here is what
    let a contended run and a clock-pinned run be filed as if they were the same measurement."""
    if not state.get('quiet'):
        pids = ', '.join(str(p['pid']) for p in state.get('competing_processes') or [])
        print(f'[WARN] another measurement process is running (pid {pids}); this run is CONTENDED '
              f'and must not be quoted as a device latency', file=stream)
    if (state.get('clocks_pinned') or {}).get('any'):
        print('[WARN] clocks are PINNED (min == max); not the default DVFS state, not comparable '
              'with unpinned runs', file=stream)


def clock_and_contention_state(policy_glob=CPU_POLICY_GLOB, devfreq_glob=GPU_DEVFREQ_GLOB,
                               proc_glob=PROC_GLOB):
    """The block folded into every measurement JSON's device_state."""
    cpu, gpu = cpu_clocks(policy_glob), gpu_clocks(devfreq_glob)
    competing = competing_gpu_processes(proc_glob=proc_glob)
    state = {
        'cpu_clocks': cpu,
        'gpu_clocks': gpu,
        'clocks_pinned': clocks_pinned(cpu, gpu),
        'load_average': load_average(),
        'competing_processes': competing,
        'quiet': len(competing) == 0,
    }
    warn_if_unreliable(state)
    return state
