#!/usr/bin/env python3
"""jetson-full-measurement.py — single entry point for the 2026-09-19 Jetson measurement
session. See deployment/JETSON-RUNBOOK-2026-09-19.md for the ordered command list.

Every subcommand writes ONE JSON file to deployment/artifacts/jetson-measurements/ with
device_state (nvpmodel/jetson_clocks/platform), git commit, and timestamp embedded
automatically (jetson-measure-utils.write_json_result). Engines are discovered from
deployment/export/jetson/*.engine by filename, not a hardcoded stage/class-count list, so this
suite works unmodified once the binary+aux relation head is exported.

--dry-run runs the full plumbing (arg parsing, engine discovery, JSON writing, tegrastats
sampler start/stop) on a machine without TensorRT/CUDA/tegrastats (e.g. this Mac) — any number
that needs real hardware is recorded as null with a skipped_reason, never fabricated.

Usage:
    .venv/bin/python deployment/scripts/jetson-full-measurement.py --dry-run --all
    python3 deployment/scripts/jetson-full-measurement.py --mode fp16 --latency --streams 1 2 4 8
    python3 deployment/scripts/jetson-full-measurement.py --mode fp16 --thermal-soak 30
"""
import argparse
import os
from importlib.util import module_from_spec, spec_from_file_location

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))


def _load(name, fname):
    spec = spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load('jetson_measure_utils', 'jetson-measure-utils.py')
LAT = _load('jetson_measure_latency', 'jetson-measure-latency.py')
POW = _load('jetson_measure_power', 'jetson-measure-power.py')

ENGINE_DIR = os.path.join(ROOT, 'deployment', 'export', 'jetson')


def run_streams(engine_dir, mode, ns, source_globs, max_frames, fps, dry_run, class_labels=None):
    result = {'mode': mode, 'streams_requested': ns}
    if dry_run or not source_globs:
        result['skipped_reason'] = (
            'dry-run: multi-stream saturation needs a real Jetson + source videos' if dry_run
            else 'no --sources given')
        return result
    import asyncio
    jmsp = _load('jetson_multi_stream_pipeline', 'jetson-multi-stream-pipeline.py')
    configs = []
    for n in ns:
        configs.append(asyncio.run(
            jmsp.run_config(n, source_globs, max_frames, fps, mode, class_labels=class_labels)))
    result['configs'] = configs
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true', help='exercise plumbing on non-Jetson hardware')
    ap.add_argument('--engine-dir', default=ENGINE_DIR)
    ap.add_argument('--mode', default='fp16', choices=['fp32', 'fp16', 'int8'])
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--runs', type=int, default=100)
    ap.add_argument('--rule', default='argmax', choices=['argmax', 'threshold'],
                    help='decision rule for --parity; use threshold for the binary head')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--all', action='store_true', help='latency + model-size + memory + parity')
    ap.add_argument('--latency', action='store_true')
    ap.add_argument('--decision', action='store_true',
                     help='per-DECISION latency on the real P3 composition (T frames of '
                          'detection + postprocess + crops + batched appearance + head). '
                          'Use this, not --latency, for the cadence claim: the latter\'s '
                          'throughput_decisions_per_sec_naive counts one pass per stage.')
    ap.add_argument('--clips-dir', default=None,
                     help='real clip tree for --decision to time frame decode against '
                          '(e.g. data/test). Omit to report the compute path only.')
    ap.add_argument('--n-clips', type=int, default=20,
                     help='how many real clips --clips-dir should time (one pass each)')
    ap.add_argument('--head-stage', default=None,
                     help='which discovered head engine --decision should time '
                          '(default: first alphabetically)')
    ap.add_argument('--model-size', action='store_true')
    ap.add_argument('--memory', action='store_true')
    ap.add_argument('--parity', action='store_true')
    ap.add_argument('--power-state', action='store_true', help='dump nvpmodel/jetson_clocks only')
    ap.add_argument('--energy', metavar='STAGE_NAME',
                    help='e.g. --energy backbone: J/inference + FPS/W over --energy-sec seconds')
    ap.add_argument('--energy-sec', type=int, default=60)
    ap.add_argument('--idle-sec', type=int, default=10,
                     help='idle baseline window for --energy; the spec defines '
                          'J/inference against (P_avg - P_idle), so 0 disables the '
                          'delta-idle figure and leaves only the total-board one')
    ap.add_argument('--thermal-soak', metavar='STAGE_NAME',
                    help='e.g. --thermal-soak backbone: continuous run for --soak-min minutes')
    ap.add_argument('--soak-min', type=float, default=30.0)
    ap.add_argument('--soak-window-sec', type=int, default=60)
    ap.add_argument('--streams', type=int, nargs='*', default=None,
                    help='multi-stream saturation, e.g. --streams 1 2 4 8')
    ap.add_argument('--sources', nargs='*', default=None, help='video files for --streams')
    ap.add_argument('--max-frames', type=int, default=240)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--class-labels', default=None,
                    help='comma-separated JSONL/decision labels passed through to --streams '
                         '(e.g. --class-labels Safe,Unsafe for the binary head); no default')
    args = ap.parse_args()
    class_labels = [s.strip() for s in args.class_labels.split(',')] if args.class_labels else None

    written = []
    if args.all:
        args.latency = args.model_size = args.memory = args.parity = True

    if args.power_state:
        written.append(U.write_json_result('power-state', {}, args.dry_run))
    if args.model_size:
        sizes = U.model_size_report(args.engine_dir)
        written.append(U.write_json_result('model-size', {'engine_dir': args.engine_dir, 'sizes_mb': sizes}, args.dry_run))
    if args.latency:
        r = LAT.run_latency(args.engine_dir, args.mode, args.warmup, args.runs, args.dry_run)
        written.append(U.write_json_result(f'latency-{args.mode}', r, args.dry_run))
    if args.decision:
        r = LAT.run_decision_latency(args.engine_dir, args.mode, args.warmup, args.runs,
                                     args.dry_run, args.head_stage, args.clips_dir, args.n_clips)
        written.append(U.write_json_result(f'decision-latency-{args.mode}', r, args.dry_run))
    if args.memory:
        r = POW.run_memory(args.engine_dir, args.mode, args.dry_run)
        written.append(U.write_json_result(f'memory-{args.mode}', r, args.dry_run))
    if args.parity:
        r = LAT.run_parity(args.engine_dir, args.warmup, args.runs, args.rule, args.threshold, args.dry_run)
        written.append(U.write_json_result('parity', r, args.dry_run))
    if args.energy:
        r = POW.run_energy(args.engine_dir, args.mode, args.energy, args.energy_sec,
                           args.dry_run, args.idle_sec)
        written.append(U.write_json_result(f'energy-{args.energy}-{args.mode}', r, args.dry_run))
    if args.thermal_soak:
        r = POW.run_thermal_soak(args.engine_dir, args.mode, args.thermal_soak, args.soak_min,
                                 args.soak_window_sec, args.dry_run)
        written.append(U.write_json_result(f'thermal-soak-{args.thermal_soak}-{args.mode}', r, args.dry_run))
    if args.streams is not None:
        r = run_streams(args.engine_dir, args.mode, args.streams, args.sources, args.max_frames,
                        args.fps, args.dry_run, class_labels)
        written.append(U.write_json_result(f'streams-{args.mode}', r, args.dry_run))

    if not written:
        ap.error('nothing to do — pass --all or one of --latency/--decision/--streams/--energy/--thermal-soak/'
                 '--memory/--model-size/--parity/--power-state')
    for p in written:
        print(f'[SAVED] {p}')


if __name__ == '__main__':
    main()
