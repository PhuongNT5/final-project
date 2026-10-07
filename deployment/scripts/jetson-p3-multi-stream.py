#!/usr/bin/env python3
"""jetson-p3-multi-stream.py — how many 24 fps camera streams can one Orin sustain on the P3
decision path?

jetson-multi-stream-pipeline.py answers that for P1 (one backbone pass per frame + 6 tiny heads).
P3 is a different pipeline — detector -> select + zone geometry -> context crops -> appearance ->
relation head — so its concurrency behaviour cannot be inferred from P1's, and
jetson-measure-latency.py --decision is a SINGLE-stream sum of separately-timed stage percentiles
over synthetic detections. This file runs the real composition, concurrently, on real video.

Reported: achieved decisions/s per stream against the 3.0 dec/s contract (24 fps at EMIT_EVERY=8),
frame drop rate, per-frame pipeline latency, head latency.
NOT reported: accuracy. The 74-d token assembly (jetson-p3-decision-utils.py) mirrors the token
builder's layout, but its numerical parity with the training-time pipeline is unverified on-device
— the limit already stated in jetson-p3-demo-video.py — so decision COUNTS are reported and
predicted labels are not.

Usage on Jetson:
    python3 deployment/scripts/jetson-p3-multi-stream.py --streams 1 2 3 \
        --source data/test/0_*/0_te1.mp4 --max-frames 240 --fps 24 --mode fp16 \
        --engine-dir deployment/export/jetson-p3 --detector-stage p3-detector-imgsz640 \
        --head-stage p3-relation-head-bin-e3
"""
import argparse
import asyncio
import functools
import os
import time
from importlib.util import module_from_spec, spec_from_file_location
from typing import Dict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))


def _load(name, fname):
    spec = spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SPU = _load('stream_pipeline_utils', 'stream-pipeline-utils.py')
P3 = _load('jetson_p3_decision_utils', 'jetson-p3-decision-utils.py')

CADENCE = 3.0                                   # decisions/s contract: 24 fps / EMIT_EVERY=8


async def inference_loop(queue, det, apps, head, states, zones_by_sid, stop, cfg):
    """Single GPU consumer over the shared queue — engines are batch-1 per frame, so streams
    interleave through one CUDA context exactly as in the P1 harness.

    Any failure sets `stop` before propagating, not just the non-finite one below: readers
    otherwise keep decoding to max_frames with nothing consuming the queue.
    """
    t_name, m_name = cfg['token_input'], cfg['mask_input']
    try:
        while not (stop.is_set() and queue.empty()):
            try:
                sid, packed = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            s = states[sid]
            t0 = time.perf_counter()
            rows, mask = P3.frame_tokens(det, apps, zones_by_sid[sid], packed, cfg['conf'],
                                         cfg['n_max'], cfg['crop_px'])
            s.buf.append((rows, mask))
            s.processed += 1
            s.frame_latencies.append((time.perf_counter() - t0) * 1000)
            # Emit per EMIT_EVERY frames actually PROCESSED (the P1 harness keys off frames READ).
            # A dropped frame must cost a decision here, otherwise an overloaded run reports a
            # healthy cadence assembled from a staler window.
            if len(s.buf) == cfg['t_window'] and s.processed % SPU.EMIT_EVERY == 0:
                tok = np.stack([r for r, _ in s.buf])[None]
                msk = np.stack([m for _, m in s.buf])[None]
                t1 = time.perf_counter()
                logits = P3._out(head.run(**P3.feed(head, {t_name: tok, m_name: msk})))
                s.head_latencies.append((time.perf_counter() - t1) * 1000)
                if not np.all(np.isfinite(logits)):
                    # A plain exception, not SystemExit: SystemExit aborts the event loop from
                    # inside the callback, so `await inf` never retrieves it and asyncio reports a
                    # stray task exception at shutdown instead of a failure.
                    raise RuntimeError('head returned non-finite logits — an fp16 overflow, not a '
                                       'throughput result; re-run the head stage in fp32')
                # The frame index each decision fired on, not a label: len() is the decision count
                # and the spacing stays checkable, while nothing can be misread as a prediction.
                s.predictions.append(s.processed)
    except BaseException:
        stop.set()
        raise


async def run_config(n, sources, max_frames, fps, det, apps, head, cfg) -> Dict:
    states = {i: SPU.StreamState(i, t_window=cfg['t_window']) for i in range(n)}
    srcs = [sources[i % len(sources)] for i in range(n)]
    zv = {i: P3.zones_for_clip(srcs[i]) for i in range(n)}
    zones_by_sid = {i: zv[i][0] for i in range(n)}
    queue: asyncio.Queue = asyncio.Queue(maxsize=SPU.QUEUE_MAX * n)
    stop = asyncio.Event()
    pre = functools.partial(P3.preprocess, size=cfg['imgsz'])
    t0 = time.perf_counter()
    readers = [asyncio.create_task(SPU.video_reader(srcs[i], queue, states[i], max_frames,
                                                    fps, stop, pre)) for i in range(n)]
    inf = asyncio.create_task(inference_loop(queue, det, apps, head, states, zones_by_sid,
                                             stop, cfg))
    # return_exceptions so a reader failure cannot skip the stop/await shutdown: the engines are
    # built once and reused across every --streams value, and a cancellation landing mid-head.run()
    # leaves the next config measuring on a context of unknown state.
    rc = await asyncio.gather(*readers, return_exceptions=True)
    stop.set()
    await inf
    for e in rc:
        if isinstance(e, BaseException):
            raise e
    elapsed = time.perf_counter() - t0
    agg = SPU.aggregate_fps(states, elapsed)
    per = [{'id': s.sid, 'source': os.path.basename(srcs[s.sid]),
            'view': zv[s.sid][1], 'has_zones': zones_by_sid[s.sid] is not None,
            'frames': s.frame_count, 'processed': s.processed, 'dropped': s.dropped,
            'n_decisions': len(s.predictions),
            'decisions_per_sec': round(len(s.predictions) / elapsed, 3),
            'frame_latency_ms': SPU.percentiles(s.frame_latencies),
            'head_latency_ms': SPU.percentiles(s.head_latencies)} for s in states.values()]
    worst = min((p['decisions_per_sec'] for p in per), default=0.0)
    # Both conditions are needed: decisions/s alone can look healthy while the reader is throwing
    # frames away, which means the stream is not actually being watched at 24 fps.
    return {'n_streams': n, 'target_fps': fps, 'max_frames': max_frames, **cfg,
            'elapsed_sec': agg['elapsed_sec'], 'aggregate_fps': agg['aggregate_fps'],
            'read_fps': agg['read_fps'], 'drop_rate': agg['drop_rate'],
            'worst_stream_decisions_per_sec': worst,
            'meets_cadence_all_streams': bool(worst >= CADENCE and agg['drop_rate'] == 0.0),
            'streams': per}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--streams', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--source', nargs='+', required=True)
    ap.add_argument('--max-frames', type=int, default=240)
    ap.add_argument('--fps', type=float, default=24.0, help='source rate; 24 is the contract')
    ap.add_argument('--mode', default='fp16', choices=['fp32', 'fp16', 'int8'])
    ap.add_argument('--engine-dir', default=os.path.join(ROOT, 'deployment', 'export', 'jetson-p3'))
    ap.add_argument('--detector-stage', default='p3-detector',
                    help='e.g. p3-detector-imgsz640 for the 640 px arm')
    ap.add_argument('--head-stage', default='p3-relation-head-bin-e3')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    # pycuda lives in main(), not at module level (the P1 harness imports it at the top): keeping
    # it here is what lets the async/accounting logic above be exercised with stub runners on a
    # machine without CUDA. autoinit must still precede the tensorrt import inside _import_trt().
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    trt, err = P3.LAT._import_trt()
    if trt is None:
        raise SystemExit(f'tensorrt/pycuda unavailable: {err}')
    det, apps, head, cfg = P3.build_runners(trt, cuda, args.engine_dir, args.mode,
                                            args.detector_stage, args.head_stage, P3.BRT.N_MAX)
    print(f"[CFG] detector {cfg['imgsz']}px  crops {cfg['crop_px']}px  T={cfg['t_window']} "
          f"N_MAX={cfg['n_max']}  app batches {cfg['app_buckets']}  emit_every={SPU.EMIT_EVERY}",
          flush=True)

    # Provenance is not optional on this device: a clock pin raises every scaling_min_freq without
    # touching the power model, so an artifact can be 1.8x faster than its twin with byte-identical
    # nvpmodel output. That is what withdrew the 09-29 latency set.
    results, out = SPU.open_sweep_artifact(
        f'multi-stream-p3-jetson-{args.mode}',
        {'mode': args.mode, 'cadence_contract_decisions_per_sec': CADENCE, 'configs': []}, args.out)

    # Discarded warmup. TRT context setup, the CUDA caching allocator and the DVFS ramp otherwise
    # all land on whichever N runs first, biasing the N-scaling trend this sweep exists to measure.
    print('[WARMUP] N=1, 24 frames, discarded', flush=True)
    asyncio.run(run_config(1, args.source, 24, args.fps, det, apps, head, cfg))

    for n in args.streams:
        print(f'\n=== N={n} streams @ {args.fps} fps, {args.mode} ===', flush=True)
        r = asyncio.run(run_config(n, args.source, args.max_frames, args.fps, det, apps, head, cfg))
        slow = min(r['streams'], key=lambda s: s['decisions_per_sec'])
        print(f"  worst stream {r['worst_stream_decisions_per_sec']} dec/s (need {CADENCE})  "
              f"drop_rate={r['drop_rate']}  its frame p50={slow['frame_latency_ms']['p50']}ms"
              f"  -> {'MEETS' if r['meets_cadence_all_streams'] else 'FAILS'}", flush=True)
        results['configs'].append(r)
        SPU.flush_json(out, results)

    if not SPU.close_sweep_artifact(results):
        print('[WARN] clock/contention state CHANGED during the sweep — the configs are not '
              'comparable with each other', flush=True)
    SPU.flush_json(out, results)
    print(f'\n[SAVED] {out}')


if __name__ == '__main__':
    main()
