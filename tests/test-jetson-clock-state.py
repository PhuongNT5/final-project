"""The provenance fields that distinguish a quotable device latency from an unquotable one.

Context: on 2026-10-01 the CLIP tower measured 41.8 ms under a concurrent energy sweep and 23.5 ms
under another session's `jetson_clocks` pin, and both JSONs carried identical device_state. These
tests cover the three distinctions that were missing: pinned vs free clocks, contended vs quiet,
and 'not measured' vs 'not pinned'.

Real sysfs- and /proc-shaped files on disk are used, not patched readers, so the parsing exercised
here is the same code that runs on the device.
"""
import os
from importlib.util import module_from_spec, spec_from_file_location

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(stem):
    path = os.path.join(ROOT, 'deployment', 'scripts', f'{stem}.py')
    spec = spec_from_file_location(stem.replace('-', '_'), path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load('jetson-clock-state-utils')


def _write(d, **files):
    os.makedirs(d, exist_ok=True)
    for name, val in files.items():
        with open(os.path.join(d, name), 'w') as f:
            f.write(f'{val}\n')


def _cpu_tree(tmp_path, lo, hi):
    root = tmp_path / 'cpufreq'
    _write(str(root / 'policy0'), scaling_min_freq=lo, scaling_cur_freq=hi,
           scaling_max_freq=hi, scaling_governor='schedutil')
    return str(root / 'policy*')


def _gpu_tree(tmp_path, lo, hi):
    root = tmp_path / 'devfreq'
    _write(str(root / '17000000.gpu'), min_freq=lo, cur_freq=hi, max_freq=hi,
           governor='nvhost_podgov')
    return str(root / '*.gpu')


def test_pinned_clocks_are_detected(tmp_path):
    """min == max is the jetson_clocks signature; this is the 22:25 state."""
    cpu = C.cpu_clocks(_cpu_tree(tmp_path, 1728000, 1728000))
    gpu = C.gpu_clocks(_gpu_tree(tmp_path, 1020000000, 1020000000))
    assert C.clocks_pinned(cpu, gpu) == {'cpu': True, 'gpu': True, 'any': True}


def test_free_clocks_are_not_reported_as_pinned(tmp_path):
    """The default DVFS state, which is what the 09-29 numbers were taken in."""
    cpu = C.cpu_clocks(_cpu_tree(tmp_path, 115200, 1728000))
    gpu = C.gpu_clocks(_gpu_tree(tmp_path, 306000000, 1020000000))
    assert C.clocks_pinned(cpu, gpu) == {'cpu': False, 'gpu': False, 'any': False}


def test_unmeasured_clocks_are_none_not_false(tmp_path):
    """Off-device there is nothing to judge. Reporting 'not pinned' there would be a claim about a
    device state that was never read -- exactly the conflation that hid today's defect."""
    pinned = C.clocks_pinned({}, {})
    assert pinned['cpu'] is None and pinned['gpu'] is None
    assert pinned['any'] is False


def test_partially_pinned_device_is_flagged(tmp_path):
    """GPU pinned, CPU free still invalidates a GPU latency comparison."""
    cpu = C.cpu_clocks(_cpu_tree(tmp_path, 115200, 1728000))
    gpu = C.gpu_clocks(_gpu_tree(tmp_path, 1020000000, 1020000000))
    assert C.clocks_pinned(cpu, gpu)['any'] is True


def _proc_tree(tmp_path, pids):
    root = tmp_path / 'proc'
    for pid, cmd in pids.items():
        d = root / str(pid)
        os.makedirs(str(d), exist_ok=True)
        with open(str(d / 'cmdline'), 'wb') as f:
            f.write('\x00'.join(cmd).encode())
    return str(root / '[0-9]*')


def test_competing_measurement_process_is_found(tmp_path):
    g = _proc_tree(tmp_path, {4242: ['python3', 'deployment/scripts/jetson-full-measurement.py',
                                     '--energy', 'p3-appearance']})
    found = C.competing_gpu_processes(own_pid=1, proc_glob=g)
    assert [p['pid'] for p in found] == [4242]
    assert 'jetson-full-measurement' in found[0]['cmdline']


def test_unrelated_processes_are_not_flagged(tmp_path):
    g = _proc_tree(tmp_path, {4243: ['/usr/bin/gnome-shell'], 4244: ['sshd:', 'phuongnguyen']})
    assert C.competing_gpu_processes(own_pid=1, proc_glob=g) == []


def test_own_process_is_never_counted_as_competition(tmp_path):
    """Otherwise every run flags itself and the signal is worthless."""
    g = _proc_tree(tmp_path, {999: ['python3', 'benchmark-r48-clip-pipeline.py', '--detector']})
    assert C.competing_gpu_processes(own_pid=999, proc_glob=g) == []


def test_contended_run_warns_on_stderr():
    import io
    buf = io.StringIO()
    C.warn_if_unreliable({'quiet': False, 'competing_processes': [{'pid': 4242, 'cmdline': 'x'}],
                          'clocks_pinned': {'any': False}}, stream=buf)
    assert 'CONTENDED' in buf.getvalue() and '4242' in buf.getvalue()


def test_pinned_run_warns_on_stderr():
    import io
    buf = io.StringIO()
    C.warn_if_unreliable({'quiet': True, 'competing_processes': [],
                          'clocks_pinned': {'any': True}}, stream=buf)
    assert 'PINNED' in buf.getvalue()


def test_quiet_unpinned_run_is_silent():
    import io
    buf = io.StringIO()
    C.warn_if_unreliable({'quiet': True, 'competing_processes': [],
                          'clocks_pinned': {'any': False}}, stream=buf)
    assert buf.getvalue() == ''


def test_device_state_carries_the_new_provenance_keys():
    """The six measurement scripts all write through device_state; the keys must be there or the
    JSONs stay indistinguishable."""
    U = _load('jetson-measure-utils')
    st = U.device_state(dry_run=True)
    for k in ('cpu_clocks', 'gpu_clocks', 'clocks_pinned', 'load_average',
              'competing_processes', 'quiet'):
        assert k in st, k


def test_state_block_survives_a_machine_without_sysfs(tmp_path):
    """On the Mac the globs match nothing. That must be empty dicts, not an exception."""
    st = C.clock_and_contention_state(policy_glob=str(tmp_path / 'nope*'),
                                     devfreq_glob=str(tmp_path / 'nope*'),
                                     proc_glob=str(tmp_path / 'nope*'))
    assert st['cpu_clocks'] == {} and st['gpu_clocks'] == {}
    assert st['quiet'] is True and st['clocks_pinned']['any'] is False
