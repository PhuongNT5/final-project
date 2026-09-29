"""
benchmark-p3-onnx.py — P3 (YOLO26 + RelationTransformer) ONNX harness. Mirrors benchmark-harness.py.

Does two things the P1 harness structurally cannot (see
plans/2026-09-06-thesis-2week-final/reports/laneb-p3-edge-cost-and-export.md):

  1. **Parity gate.** Runs the exported relation head over the 64 test clips and checks its softmax
     against the `blind_softmax` stored in logits_yolo26.pt. No latency number is worth reporting
     for a graph that does not reproduce the accuracy it is supposed to represent, so --bench
     refuses to run unless parity passes (override with --skip_parity, which is recorded in the JSON).
  2. **Per-decision timing.** P1's benchmark sums a per-FRAME backbone term with a per-CLIP head
     term, which is only meaningful under a stated cadence. Everything here is reported per
     DECISION: T frames of detection + the clip's crops of appearance + one head pass.

Run this on an IDLE machine — concurrent training makes the timings meaningless.

Usage:
    .venv/bin/python deployment/scripts/benchmark-p3-onnx.py --parity
    .venv/bin/python deployment/scripts/benchmark-p3-onnx.py --bench --runs 20
"""
import argparse
import json
import os
import platform
import sys
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import build_relation_tokens as BRT  # noqa: E402  (read-only: CONF/N_MAX/geometry() reuse)

# benchmark-p3-utils.py / benchmark-p3-sweep.py hold helpers split out to keep each file
# under ~200 lines (hyphenated filenames -> importlib, same pattern as pipeline-utils.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_u_spec = spec_from_file_location('benchmark_p3_utils', os.path.join(_HERE, 'benchmark-p3-utils.py'))
U = module_from_spec(_u_spec)
_u_spec.loader.exec_module(U)
_s_spec = spec_from_file_location('benchmark_p3_sweep', os.path.join(_HERE, 'benchmark-p3-sweep.py'))
S = module_from_spec(_s_spec)
_s_spec.loader.exec_module(S)

EXPORT_DIR = os.path.join(ROOT, 'deployment', 'export')
ART_DIR = os.path.join(ROOT, 'deployment', 'artifacts')
TOKENS = os.path.join(ROOT, 'sweep_results', 'relation_tokens', 'tokens_yolo26app_test.pt')
STORED = os.path.join(ROOT, 'sweep_results', 'relation_tokens', 'logits_yolo26.pt')

GRAPHS = {'detector': 'p3-detector-fp32.onnx',
          'appearance': 'p3-appearance-fp32.onnx',
          'head': 'p3-relation-head-fp32.onnx'}
T_FRAMES, IMGSZ, CROP = 16, 800, 224
CLIP_W, CLIP_H = U.CLIP_W, U.CLIP_H
EP = {'cpu': ['CPUExecutionProvider'], 'coreml': ['CoreMLExecutionProvider', 'CPUExecutionProvider']}


def run_parity(sess):
    """Exported head vs the softmax the thesis actually reports."""
    d = torch.load(TOKENS, weights_only=False)
    stored = torch.load(STORED, weights_only=False)
    if 'blind_softmax' not in stored:
        sys.exit(f'{STORED} has no blind_softmax — it predates the --save_logits patch (brief S4).')
    tok, mask, y = d['tokens'].float(), d['mask'].float(), np.asarray(d['labels'])
    s = sess['head']
    names = [i.name for i in s.get_inputs()]
    logits = np.concatenate([
        s.run(None, {names[0]: tok[i:i + 1].numpy(), names[1]: mask[i:i + 1].numpy()})[0]
        for i in range(len(y))])
    sm = torch.softmax(torch.from_numpy(logits), 1)
    want = stored['blind_softmax'].float()
    diff = (sm - want).abs().max().item()
    agree = int((sm.argmax(1) == want.argmax(1)).sum())
    acc = int((sm.argmax(1).numpy() == y).sum())
    print(f'  clips {len(y)} | max abs softmax diff {diff:.3e} | argmax agreement {agree}/{len(y)}'
          f' | onnx top-1 {acc}/{len(y)}')
    ok = agree == len(y) and diff < 1e-3
    print(f'  parity: {"PASS" if ok else "*** FAIL ***"}')
    return {'max_abs_diff': diff, 'argmax_agreement': agree, 'n': len(y),
            'onnx_top1_correct': acc, 'pass': ok}


def run_bench(sess, warmup, runs, crop_scenarios=None):
    """Per-decision cost over the FULL decision path: detector x T_FRAMES + postprocess
    (confidence-select + real Shapely geometry) + crop extraction (real cv2 crop/resize) +
    appearance (batched) + head. Dummy detector/appearance/head *inputs* are still synthetic
    (this measures the graphs+glue code, not real footage) — but every stage of the pipeline
    that used to be silently excluded (NMS-equivalent box selection, crop extraction,
    zone-geometry) is now inside the timed region, and it is run once per REAL crop-count
    percentile (p50/p95/p99/cap, from the actual stored mask tensors) instead of only the
    mean, so the reported number is no longer an optimistic lower bound on tail latency.
    """
    if crop_scenarios is None:
        crop_scenarios = U.crop_count_percentiles()
    det, app, head = sess['detector'], sess['appearance'], sess['head']
    dn, an, hn = det.get_inputs()[0].name, app.get_inputs()[0].name, [i.name for i in head.get_inputs()]
    frame_img = np.random.randn(1, 3, IMGSZ, IMGSZ).astype(np.float32)
    tok = np.random.randn(1, T_FRAMES, 8, 74).astype(np.float32)
    msk = np.ones((1, T_FRAMES, 8), dtype=np.float32)
    zones = next(iter(BRT.load_zones().values()), None)
    synth_frame = (np.random.rand(CLIP_H, CLIP_W, 3) * 255).astype(np.uint8)

    # Fixed synthetic raw detector output, generated ONCE outside every timed lambda — this file
    # already documented (run_sweep) that generating random inputs inside a timed lambda
    # contaminates the measurement (RNG cost, not graph/glue cost); same discipline applies here
    # to NMS-equivalent selection and crop extraction.
    raw_frames = U.synthetic_raw_frames(T_FRAMES)

    def run_postprocess_all_frames():
        return [U.postprocess_frame(raw, zones) for raw in raw_frames]

    t_det = U.timed(lambda: det.run(None, {dn: frame_img}), warmup, runs)
    t_post = U.timed(run_postprocess_all_frames, warmup, runs)

    cap_boxes = [b for frame_boxes in run_postprocess_all_frames() for b in frame_boxes]
    if not cap_boxes:  # degenerate synthetic draw — fall back to a flat pool
        cap_boxes = [{'cx': 0.5, 'cy': 0.5, 'w': 0.2, 'h': 0.4}] * crop_scenarios.get('cap', 128)

    t_head = U.timed(lambda: head.run(None, {hn[0]: tok, hn[1]: msk}), warmup, runs)

    print(f"  detector     p50 {t_det['50']:8.2f} ms/frame  x{T_FRAMES} = {T_FRAMES * t_det['50']:8.2f}")
    print(f"  postprocess  p50 {t_post['50']:8.2f} ms/decision (select + geometry, {T_FRAMES} frames)")
    print(f"  head         p50 {t_head['50']:8.2f} ms/decision")
    scenarios = {}
    for name, n_crops in crop_scenarios.items():
        if name in ('source', 'n_clips') or not isinstance(n_crops, int):
            continue
        pool = (cap_boxes * (n_crops // len(cap_boxes) + 1))[:n_crops]
        t_crop = U.timed(lambda p=pool: U.extract_crops(synth_frame, p, CROP), warmup, runs)
        batch = np.random.randn(n_crops, 3, CROP, CROP).astype(np.float32)
        t_app = U.timed(lambda b=batch: app.run(None, {an: b}), warmup, runs)
        per_dec = (T_FRAMES * t_det['50'] + t_post['50'] + t_crop['50'] + t_app['50'] + t_head['50'])
        print(f"  [{name:>4}={n_crops:>3} crops] crop-extract p50 {t_crop['50']:6.2f}  "
              f"appearance p50 {t_app['50']:6.2f}  => {per_dec:8.1f} ms/decision "
              f"= {1000 / per_dec:.2f} decisions/s")
        scenarios[name] = {
            'n_crops': n_crops, 'crop_extract_ms': t_crop, 'appearance_ms': t_app,
            'ms_per_decision_p50': round(per_dec, 2),
            'decisions_per_sec_p50': round(1000 / per_dec, 2),
        }
    print('     (24 fps with EMIT_EVERY=8 needs 3.0 decisions/s; previously only the MEAN '
          'crop count was reported — see p95/p99/cap rows above for tail cost)')
    return {'detector_ms_per_frame': t_det, 'postprocess_ms_per_decision': t_post,
            'head_ms_per_decision': t_head, 'frames_per_decision': T_FRAMES,
            'crop_scenarios': crop_scenarios, 'by_scenario': scenarios,
            'stages_included': ['detector', 'postprocess(select+geometry)', 'crop_extract(cv2)',
                                 'appearance', 'head']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ep', choices=list(EP), default='cpu')
    ap.add_argument('--parity', action='store_true')
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--skip_parity', action='store_true', help='bench without the parity gate')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--crops', type=int, default=58,
                    help='crop count for --sweep only; --bench now uses real p50/p95/p99/cap '
                         'from crop_count_percentiles() instead of one fixed count')
    ap.add_argument('--sweep', action='store_true',
                    help='detector cost curve x frames-per-decision (needs export-p3-graphs.py '
                         '--cost_imgsz first). COST ONLY — see run_sweep docstring.')
    ap.add_argument('--frames', type=int, nargs='*', default=[16, 8, 4],
                    help='frames-per-decision options to cross with the imgsz curve')
    args = ap.parse_args()
    if not (args.parity or args.bench or args.sweep):
        args.parity = args.bench = True

    sess = U.sessions(args.ep, EXPORT_DIR, GRAPHS, EP)
    out = {'machine': platform.node(), 'platform': platform.platform(), 'ep': args.ep,
           'timestamp': datetime.now().isoformat()}

    if args.parity or (args.bench and not args.skip_parity):
        print('Parity — exported relation head vs stored blind_softmax:')
        out['parity'] = run_parity(sess)
        if args.bench and not args.skip_parity and not out['parity']['pass']:
            sys.exit('parity FAILED — refusing to report latency for a graph that does not '
                     'reproduce the headline predictor. Override with --skip_parity.')
    if args.bench:
        print(f'\nPer-decision latency ({args.ep}, warmup {args.warmup}, runs {args.runs}):')
        out['bench'] = run_bench(sess, args.warmup, args.runs)
        out['bench']['parity_skipped'] = bool(args.skip_parity)

    if args.sweep:
        out['sweep'] = S.run_sweep(sess, args.ep, args.warmup, args.runs, args.crops, args.frames,
                                   EXPORT_DIR, GRAPHS, EP, T_FRAMES, CROP, IMGSZ)

    os.makedirs(ART_DIR, exist_ok=True)
    p = os.path.join(ART_DIR, f'benchmarks-p3-onnx-{args.ep}.json')
    json.dump(out, open(p, 'w'), indent=2)
    print(f'\n[SAVED] {p}')


if __name__ == '__main__':
    main()
