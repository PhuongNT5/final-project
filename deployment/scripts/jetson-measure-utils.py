"""jetson-measure-utils.py — shared helpers for the Jetson measurement suite
(jetson-full-measurement.py + jetson-measure-latency.py + jetson-measure-power.py).

Device-state capture (nvpmodel/jetson_clocks/tegrastats) is real subprocess I/O, not
simulated — on a machine without those binaries (e.g. this Mac), calls fail gracefully and the
result is recorded as null/dry_run, never as a fabricated number.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART_DIR = os.path.join(ROOT, 'deployment', 'artifacts', 'jetson-measurements')


def percentiles(xs):
    if not xs:
        return {'p50': None, 'p95': None, 'p99': None, 'n': 0}
    import numpy as np
    a = np.asarray(xs, dtype=np.float64)
    return {'p50': round(float(np.percentile(a, 50)), 3),
            'p95': round(float(np.percentile(a, 95)), 3),
            'p99': round(float(np.percentile(a, 99)), 3), 'n': len(xs)}


# The rsync to the device excludes .git, so `git rev-parse` there returns nothing and every
# device JSON from the 2026-09-29 session recorded 'unknown' -- a measurement that cannot be tied
# to the code that produced it. jetson-rsync-from-mac.sh now stamps the host commit into this file
# as part of the transfer, and it is the fallback below.
HOST_COMMIT_FILE = os.path.join(ROOT, 'deployment', 'artifacts', 'host-commit.txt')


def git_commit():
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    try:
        with open(HOST_COMMIT_FILE) as f:
            return f.read().strip() or 'unknown'
    except OSError:
        return 'unknown'


def run_cmd(cmd, timeout=5):
    """Returns stripped stdout, or None if the binary is missing / errors / times out — this is
    how --dry-run works on non-Jetson machines without special-casing every call site."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def device_state(dry_run):
    return {
        'dry_run': dry_run,
        'nvpmodel_q': run_cmd(['nvpmodel', '-q']),
        'jetson_clocks_show': run_cmd(['jetson_clocks', '--show']),
        'platform': __import__('platform').platform(),
        'python_version': sys.version.split()[0],
    }


_TEGRA_RAM = re.compile(r'RAM (\d+)/(\d+)MB')
_TEGRA_TEMP = re.compile(r'(\w+)@([\-\d.]+)C')
_TEGRA_POWER = re.compile(r'VDD_IN (\d+)mW/(\d+)mW')
_TEGRA_GR3D = re.compile(r'GR3D_FREQ (\d+)%')


def parse_tegrastats_line(line):
    """Best-effort parse of one `tegrastats` line. Orin Nano format varies by JetPack version —
    unmatched fields are left None rather than guessed, so a format change is visible as nulls
    in the JSON instead of silently-wrong numbers."""
    out = {'raw': line}
    m = _TEGRA_RAM.search(line)
    if m:
        out['ram_used_mb'], out['ram_total_mb'] = int(m.group(1)), int(m.group(2))
    m = _TEGRA_POWER.search(line)
    if m:
        out['power_now_mw'], out['power_avg_mw'] = int(m.group(1)), int(m.group(2))
    m = _TEGRA_GR3D.search(line)
    if m:
        out['gpu_util_pct'] = int(m.group(1))
    temps = {k: float(v) for k, v in _TEMP_MATCHES(line)}
    if temps:
        out['temps_c'] = temps
    return out


def _TEMP_MATCHES(line):
    return [(m.group(1), m.group(2)) for m in _TEGRA_TEMP.finditer(line)]


class TegrastatsSampler:
    """Background `tegrastats --interval <ms>` reader. On a machine without tegrastats (Mac),
    start() records dry_run wall-clock ticks with no fabricated telemetry — samples() then
    returns an explicitly dry_run-flagged list, never invented power/thermal numbers."""

    def __init__(self, interval_ms=1000):
        self.interval_ms = interval_ms
        self._proc = None
        self._thread = None
        self._samples = []
        self._stop = threading.Event()
        self._dry_run = False

    def start(self):
        try:
            self._proc = subprocess.Popen(
                ['tegrastats', '--interval', str(self.interval_ms)],
                stdout=subprocess.PIPE, text=True)
        except (FileNotFoundError, OSError):
            self._proc = None
            self._dry_run = True
            self._thread = threading.Thread(target=self._dry_loop, daemon=True)
            self._thread.start()
            return
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            parsed = parse_tegrastats_line(line.strip())
            parsed['ts'] = time.time()
            self._samples.append(parsed)

    def _dry_loop(self):
        while not self._stop.is_set():
            self._samples.append({'ts': time.time(), 'dry_run': True})
            time.sleep(self.interval_ms / 1000.0)

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return {'dry_run': self._dry_run, 'n_samples': len(self._samples),
                'samples': self._samples}


def write_json_result(name, payload, dry_run):
    os.makedirs(ART_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')
    payload = dict(payload)
    payload.setdefault('git_commit', git_commit())
    payload.setdefault('timestamp', datetime.now().isoformat())
    payload.setdefault('device_state', device_state(dry_run))
    path = os.path.join(ART_DIR, f'{name}-{ts}.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def model_size_report(engine_dir):
    sizes = {}
    if os.path.isdir(engine_dir):
        for fn in sorted(os.listdir(engine_dir)):
            if fn.endswith(('.engine', '.onnx')):
                p = os.path.join(engine_dir, fn)
                sizes[fn] = round(os.path.getsize(p) / (1024 * 1024), 3)
    return sizes


def decide(probs, rule='argmax', threshold=0.5):
    """Generic decision rule — 'argmax' (multi-class) or 'threshold' (binary, prob of the
    positive/unsafe class >= threshold). Parameterised so this suite doesn't assume a fixed
    output class count (the relation head is moving from 4-class to binary+aux heads)."""
    import numpy as np
    p = np.asarray(probs)
    if rule == 'threshold':
        pos = p[..., -1] if p.shape[-1] > 1 else p[..., 0]
        return (pos >= threshold).astype(int)
    return p.argmax(-1)
