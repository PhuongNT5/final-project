"""jetson-measure-latency.py — per-engine latency/throughput + FP16/INT8 vs FP32 numerical
parity, for whatever engines exist in deployment/export/jetson/. Engines are discovered by
filename (`{stage}-{precision}.engine`) instead of a hardcoded stage list/class count, since the
relation head is moving from 4-class to binary+aux (see design-2026-09-16-binary-encoder-
redesign.md) and this suite must not assume a fixed output width.

--dry-run works on Mac: engines are still discovered from disk (real glob, not fabricated), but
timing/parity is skipped with an explicit `skipped_reason` instead of inventing numbers, since
tensorrt/pycuda aren't installed here.
"""
import glob
import os
import re
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_u_spec = spec_from_file_location('jetson_measure_utils', os.path.join(_HERE, 'jetson-measure-utils.py'))
U = module_from_spec(_u_spec)
_u_spec.loader.exec_module(U)
_tru_spec = spec_from_file_location('trt_runner_utils', os.path.join(_HERE, 'trt-runner-utils.py'))
TRU = module_from_spec(_tru_spec)
_tru_spec.loader.exec_module(TRU)

_NAME_RE = re.compile(r'^(.*)-(fp32|fp16|int8)\.engine$')


def discover_engines(engine_dir):
    """{stage_name: {precision: path}} from every *.engine file present."""
    out = {}
    for p in sorted(glob.glob(os.path.join(engine_dir, '*.engine'))):
        m = _NAME_RE.match(os.path.basename(p))
        if not m:
            continue
        stage, prec = m.groups()
        out.setdefault(stage, {})[prec] = p
    return out


def _import_trt():
    try:
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        return trt, cuda
    except ImportError as e:
        return None, str(e)


def _fixed_inputs(runner, seed=0):
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(entry['shape']).astype(entry['host'].dtype)
            for name, entry in runner.inputs.items()}


def measure_stage_latency(trt, cuda, path, warmup, runs):
    runner = TRU.TRTRunner(trt, cuda, path)
    inputs = _fixed_inputs(runner)
    for _ in range(warmup):
        runner.run(**inputs)
    ts = []
    out = None
    for _ in range(runs):
        import time
        t0 = time.perf_counter()
        out = runner.run(**inputs)
        ts.append((time.perf_counter() - t0) * 1000)
    return U.percentiles(ts), out


def run_latency(engine_dir, mode, warmup, runs, dry_run):
    stages = discover_engines(engine_dir)
    result = {'mode': mode, 'engine_dir': engine_dir, 'stages_found': sorted(stages),
              'warmup': warmup, 'runs': runs}
    if dry_run:
        trt, cuda = None, None
    else:
        trt, cuda = _import_trt()
    if trt is None:
        result['skipped_reason'] = (
            'dry-run: tensorrt/pycuda not available on this machine' if dry_run
            else f'tensorrt/pycuda import failed: {cuda}')
        result['per_stage_ms'] = {}
        result['end_to_end_naive_sum_ms_p50'] = None
        return result
    per_stage = {}
    for stage, precisions in stages.items():
        if mode not in precisions:
            continue
        pct, _ = measure_stage_latency(trt, cuda, precisions[mode], warmup, runs)
        per_stage[stage] = pct
    result['per_stage_ms'] = per_stage
    # naive_sum: sum of per-stage p50s in discovery order. NOT necessarily the real decision
    # path (that depends on the pipeline's stage order, which is architecture-specific and
    # currently being redesigned) — callers needing the true end-to-end number should pass
    # --stages to define the real order; this is a coarse sanity figure only.
    p50s = [v['p50'] for v in per_stage.values() if v['p50'] is not None]
    result['end_to_end_naive_sum_ms_p50'] = round(sum(p50s), 3) if p50s else None
    result['throughput_decisions_per_sec_naive'] = (
        round(1000.0 / sum(p50s), 3) if p50s else None)
    return result


DECISION_T_FRAMES, DECISION_IMGSZ, DECISION_CROP = 16, 800, 224
DECISION_EMIT_EVERY = 8   # cadence contract: one decision per 8 NEW frames (24 fps -> 3.0 dec/s)


def _bench_utils():
    """benchmark-p3-utils.py holds the REAL postprocess/crop/crop-percentile code that the
    Mac ONNX baseline (benchmark-p3-onnx.py) is timed with. Reusing it here — rather than
    re-implementing it — is what makes the Jetson number comparable to that baseline."""
    spec = spec_from_file_location('benchmark_p3_utils',
                                   os.path.join(_HERE, 'benchmark-p3-utils.py'))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _timed(fn, warmup, runs):
    import time
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return U.percentiles(ts)


def run_decision_latency(engine_dir, mode, warmup, runs, dry_run, head_stage=None,
                         clips_dir=None, n_clips=20):
    """Per-DECISION latency on the real composition, not a sum of one pass per stage.

    benchmark-p3-onnx.py defines a P3 decision as: T frames of detection + postprocess
    (confidence-select + Shapely zone geometry) + crop extraction + the clip's whole crop set
    through appearance in ONE batched pass + one head pass. `end_to_end_naive_sum_ms_p50` from
    run_latency() counts a SINGLE detector pass and a SINGLE batch-1 appearance pass, so its
    `throughput_decisions_per_sec_naive` overstates the decision rate by more than an order of
    magnitude and must never be reported as one.

    Same formula and the same crop-count percentiles as the Mac ONNX baseline, so the two are
    directly comparable. The cadence contract is 24 fps at EMIT_EVERY=8 = 3.0 decisions/s.
    """
    stages = discover_engines(engine_dir)
    B = _bench_utils()
    crop_scenarios = B.crop_count_percentiles()
    heads = [k for k in stages if 'head' in k]
    head_stage = head_stage or (sorted(heads)[0] if heads else None)
    result = {'mode': mode, 'engine_dir': engine_dir, 'warmup': warmup, 'runs': runs,
              'frames_per_decision': DECISION_T_FRAMES, 'imgsz': DECISION_IMGSZ,
              'crop_size': DECISION_CROP, 'head_stage': head_stage,
              'crop_scenarios': crop_scenarios, 'stages_found': sorted(stages),
              'stages_included': ['detector', 'postprocess(select+geometry)',
                                  'crop_extract(cv2)', 'appearance(batched)', 'head'],
              'formula': ('T_FRAMES*detector_p50 + postprocess_p50 + crop_extract_p50 '
                          '+ appearance_p50 + head_p50'),
              'baseline_ref': 'deployment/artifacts/benchmarks-p3-onnx-cpu.json'}
    trt, cuda = (None, None) if dry_run else _import_trt()
    if trt is None:
        result['skipped_reason'] = ('dry-run: tensorrt/pycuda not available on this machine'
                                    if dry_run else f'tensorrt/pycuda import failed: {cuda}')
        return result
    missing = [k for k in ('p3-detector', 'p3-appearance') if mode not in stages.get(k, {})]
    if missing or head_stage is None:
        result['skipped_reason'] = f'missing {mode} engine(s): {missing or ["<head>"]}'
        return result

    det = TRU.TRTRunner(trt, cuda, stages['p3-detector'][mode])
    head = TRU.TRTRunner(trt, cuda, stages[head_stage][mode])
    det_in, head_in = _fixed_inputs(det), _fixed_inputs(head)
    result['detector_ms_per_frame'] = _timed(lambda: det.run(**det_in), warmup, runs)

    zones = next(iter(B.BRT.load_zones().values()), None)
    raw_frames = B.synthetic_raw_frames(DECISION_T_FRAMES)
    postproc = lambda: [B.postprocess_frame(r, zones) for r in raw_frames]
    result['postprocess_ms_per_decision'] = _timed(postproc, warmup, runs)
    result['head_ms_per_decision'] = _timed(lambda: head.run(**head_in), warmup, runs)

    cap_boxes = [b for fb in postproc() for b in fb] or \
        [{'cx': 0.5, 'cy': 0.5, 'w': 0.2, 'h': 0.4}] * crop_scenarios.get('cap', 128)
    frame = (np.random.default_rng(0).random((B.CLIP_H, B.CLIP_W, 3)) * 255).astype(np.uint8)

    t_det, t_post, t_head = (result['detector_ms_per_frame'], result['postprocess_ms_per_decision'],
                             result['head_ms_per_decision'])
    scenarios = {}
    for name, n_crops in crop_scenarios.items():
        if name in ('source', 'n_clips') or not isinstance(n_crops, int):
            continue
        pool = (cap_boxes * (n_crops // len(cap_boxes) + 1))[:n_crops]
        t_crop = _timed(lambda p=pool: B.extract_crops(frame, p, DECISION_CROP), warmup, runs)
        try:
            app = TRU.TRTRunner(trt, cuda, stages['p3-appearance'][mode], batch=n_crops)
        except ValueError as e:
            scenarios[name] = {'n_crops': n_crops, 'crop_extract_ms': t_crop, 'error': str(e)}
            continue
        app_in = _fixed_inputs(app)
        t_app = _timed(lambda: app.run(**app_in), warmup, runs)
        per_dec = (DECISION_T_FRAMES * t_det['p50'] + t_post['p50'] + t_crop['p50']
                   + t_app['p50'] + t_head['p50'])
        scenarios[name] = {'n_crops': n_crops, 'crop_extract_ms': t_crop, 'appearance_ms': t_app,
                           'ms_per_decision_p50': round(per_dec, 2),
                           'decisions_per_sec_p50': round(1000.0 / per_dec, 2),
                           'meets_3_per_sec_cadence': bool(1000.0 / per_dec >= 3.0)}
        del app
    if clips_dir:
        result['decode'] = _measure_decode(clips_dir, n_clips, DECISION_T_FRAMES, result)
        # Only the STREAM term belongs in the cadence verdict: at EMIT_EVERY the pipeline
        # consumes 8 new sequential frames per decision. The seek-sampled term is the cost of
        # the OFFLINE token-extraction path (random access into archived files) and is reported
        # separately, never added to the deployment cadence.
        dec = (result['decode'].get('stream_frames_per_decision') or {}).get('p50')
        if dec is not None:
            for s_ in scenarios.values():
                if 'ms_per_decision_p50' not in s_:
                    continue
                tot = s_['ms_per_decision_p50'] + dec
                s_['ms_per_decision_p50_with_stream_decode'] = round(tot, 2)
                s_['decisions_per_sec_p50_with_stream_decode'] = round(1000.0 / tot, 2)
                s_['meets_3_per_sec_cadence_with_stream_decode'] = bool(1000.0 / tot >= 3.0)
    result['by_scenario'] = scenarios
    return result


def _measure_decode(clips_dir, n_clips, t_frames, result):
    """Frame-decode cost, split into the two DIFFERENT costs that the one word 'decode' hides.

    (a) seek-sampled: what build_relation_tokens.process_clip pays offline — T frames spread
        across an archived file, each reached by CAP_PROP_POS_FRAMES. A random seek into H.264
        forces re-decoding from the preceding keyframe, so this is dominated by seek, not by
        decode, and it is a property of the offline token-extraction path only.
    (b) stream: what a deployed camera pipeline pays — EMIT_EVERY new CONSECUTIVE frames per
        decision, no seeking, the rest of the T-frame window already in the rolling buffer.

    Only (b) may enter the cadence verdict. Each clip is timed ONCE and a different clip is used
    every iteration, so neither figure is a warm-page-cache repeat of one file. Both are still
    file decode rather than live sensor capture, which remains a stated limit.
    """
    import glob as _glob
    import random
    import time
    import cv2
    clips = sorted(_glob.glob(os.path.join(clips_dir, '**', '*.mp4'), recursive=True))
    out = {'source': {'clips_dir': clips_dir, 'clips_available': len(clips),
                      'clips_timed': min(n_clips, len(clips)),
                      'emit_every': DECISION_EMIT_EVERY, 't_frames': t_frames,
                      'caveat': 'file decode, not live sensor capture; not cache-warmed'}}
    if not clips:
        out['error'] = 'no clips found'
        return out
    random.Random(0).shuffle(clips)
    picked = clips[:n_clips]

    seek_ts, stream_ts, per_frame_ts = [], [], []
    for path in picked:
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or t_frames
        for i in [int(x) for x in np.linspace(0, max(0, total - 1), t_frames)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(i, max(0, total - 2)))
            cap.read()
        cap.release()
        seek_ts.append((time.perf_counter() - t0) * 1000)

        # Stream path: one open + sequential reads, the seek excluded (a live feed never seeks).
        cap = cv2.VideoCapture(path)
        for _ in range(DECISION_EMIT_EVERY):   # prime: skip past the container header frames
            cap.read()
        t0 = time.perf_counter()
        n_read = 0
        for _ in range(DECISION_EMIT_EVERY):
            ok, _f = cap.read()
            if not ok:
                break
            n_read += 1
        dt = (time.perf_counter() - t0) * 1000
        cap.release()
        if n_read:
            stream_ts.append(dt * DECISION_EMIT_EVERY / n_read)
            per_frame_ts.append(dt / n_read)

    out['seek_sampled_t_frames'] = U.percentiles(seek_ts)
    out['seek_sampled_t_frames']['what'] = (
        f'{t_frames} frames reached by CAP_PROP_POS_FRAMES; OFFLINE extraction cost, '
        'NOT part of the deployment cadence')
    out['stream_frames_per_decision'] = U.percentiles(stream_ts)
    out['stream_frames_per_decision']['what'] = (
        f'{DECISION_EMIT_EVERY} consecutive frames, no seek; the per-decision decode increment '
        'a live pipeline actually pays')
    out['stream_ms_per_frame'] = U.percentiles(per_frame_ts)
    return out


def run_parity(engine_dir, warmup, runs, rule, threshold, dry_run):
    stages = discover_engines(engine_dir)
    result = {'engine_dir': engine_dir, 'rule': rule, 'threshold': threshold,
              'input_source': 'random-fixed-seed(0)', 'stages': {}}
    if dry_run:
        trt, cuda = None, None
    else:
        trt, cuda = _import_trt()
    if trt is None:
        result['skipped_reason'] = (
            'dry-run: tensorrt/pycuda not available on this machine' if dry_run
            else f'tensorrt/pycuda import failed: {cuda}')
        return result
    for stage, precisions in stages.items():
        ref_prec = 'fp32' if 'fp32' in precisions else next(iter(precisions), None)
        if ref_prec is None:
            continue
        ref_runner = TRU.TRTRunner(trt, cuda, precisions[ref_prec])
        fixed_in = _fixed_inputs(ref_runner, seed=0)
        ref_out = ref_runner.run(**fixed_in)
        ref_dec = U.decide(ref_out, rule, threshold)
        stage_res = {'reference_precision': ref_prec}
        for prec, path in precisions.items():
            if prec == ref_prec:
                continue
            runner = TRU.TRTRunner(trt, cuda, path)
            out = runner.run(**fixed_in)
            diff = float(np.abs(np.asarray(out) - np.asarray(ref_out)).max())
            dec = U.decide(out, rule, threshold)
            agree = float(np.mean(dec == ref_dec))
            stage_res[prec] = {'max_abs_diff': diff, 'decision_agreement_rate': agree}
        result['stages'][stage] = stage_res
    return result
