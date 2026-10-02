"""End-to-end FP16 parity for the R48 deployment path, on real iSafetyBench footage.

The edge chapter wants to quote the FP16 Jetson latency next to the 84.4% balanced accuracy. That
is only legitimate if the FP16 engines make the SAME decisions as the FP32 graphs the accuracy was
measured with. This script checks that on real clips rather than on random tensors, because the
drift that matters is drift on the data distribution.

Two modes, one preprocessing function, so the device cannot silently preprocess differently:

  --make-fixture  (host)   pick N clips stratified by label, decode them, preprocess, run the FP32
                           ONNX tower + head, and write the reference probabilities plus a checksum
                           of the preprocessed pixels. Copies the clips next to the fixture so the
                           rsync carries them.
  (default)       (device) re-decode the same clips, re-preprocess, run the FP16 TRT engines, and
                           compare. The pixel checksum is compared FIRST: a decode/preprocess
                           mismatch is reported as such and never charged to quantisation.

This measures NUMERICAL agreement between the two runtimes, not accuracy -- the deployed 3-seed
checkpoint saw these clips in training, so its probabilities here are not an accuracy claim.

Usage:
    .venv/bin/python deployment/scripts/verify-r48-device-parity.py --make-fixture --n 24
    python3 deployment/scripts/verify-r48-device-parity.py --mode fp16        # on the Jetson
"""
import argparse
import os
import shutil
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, ROOT)

EXPORT = os.path.join(ROOT, 'deployment', 'export')
ENGINE_DIR = os.path.join(EXPORT, 'jetson')
FIXTURE_DIR = os.path.join(ROOT, 'deployment', 'artifacts', 'r48-parity-fixture')
FIXTURE = os.path.join(FIXTURE_DIR, 'fixture.npz')
TOWER, HEAD = 'r48-clip-vit-b32-vision', 'r48-b1-clip-temporal-3seed'
FRAMES = 16
# Agreement is the gate; max_abs_diff is reported for the thesis table but does not fail the run.
DECISION_GATE = 1.0


def _load(name, fname):
    spec = spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def preprocess(frames, mean, std):
    """BGR uint8 frames -> CLIP input (T,3,224,224). Numerically identical to
    build-isafetybench-clip-features.to_batch: RGB, resize 224, /255, then the processor's own
    mean/std. The mean/std come from the fixture, not from a constant retyped here, so host and
    device cannot diverge.

    The channel swap is applied to the 224x224 RESULT, not to the full-resolution frame. Both
    orders give bit-identical pixels -- cv2.resize weights within a channel and never across
    channels -- but `f[:, :, ::-1]` is a negative-stride view, so OpenCV materialises a contiguous
    copy of the whole frame first: 90,9 of 101,0 ms of this function at 1080p
    (profile-r48-preprocess.py). Bit-equality is not assumed here; it is gated by that script and
    by tests/test-r48-preprocess-variants.py."""
    ims = [cv2.resize(f, (224, 224))[:, :, ::-1].astype('float32') / 255.0 for f in frames]
    x = np.stack(ims).transpose(0, 3, 1, 2)
    return (x - mean) / std


def pixel_checksum(x):
    """Cheap fingerprint of the preprocessed pixels. Separates 'the device decoded different frames'
    from 'FP16 drifted', which otherwise look identical in the output.

    The last term is frame-index-weighted on purpose: sum/mean/std are all order-invariant, so
    without it a decode that returned the right 16 frames in the WRONG ORDER would pass the guard
    while feeding the temporal head a different sequence."""
    a = np.asarray(x, dtype=np.float64)
    w = np.arange(1, a.shape[0] + 1, dtype=np.float64)
    return [float(a.sum()), float(a.mean()), float(a.std()),
            float((w * a.reshape(a.shape[0], -1).sum(1)).sum())]


def pixels_match(got, want, rtol=1e-6):
    """Tolerant on purpose. A different OpenCV build resizing to within a few ULPs is not a finding;
    decoding different FRAMES moves these statistics by percent, which this still catches.

    The length check is not pedantry: `zip` would silently compare only the shared prefix, so a
    fixture written before the frame-index-weighted term was added would quietly lose the
    frame-order guard and still report PARITY PASS."""
    got, want = [float(v) for v in got], [float(v) for v in want]
    if len(got) != len(want):
        raise ValueError(f'checksum length {len(got)} != fixture {len(want)} -- '
                         'the fixture predates the current checksum; rebuild it with --make-fixture')
    return all(abs(g - w) <= rtol * max(1.0, abs(w)) for g, w in zip(got, want))


def decode_clip(path):
    import frame_sampling as FS
    frames, _, _ = FS.decode(path, FRAMES)
    present = [f is not None for f in frames]
    filled = [f if f is not None else np.zeros_like(next(g for g in frames if g is not None))
              for f in frames]
    return filled, np.asarray(present, dtype='float32')


def make_fixture(n, seed):
    import build_isafetybench_tokens as ISB
    import onnxruntime as ort
    from transformers import CLIPImageProcessor
    proc = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')
    mean = np.asarray(proc.image_mean, dtype='float32').reshape(1, 3, 1, 1)
    std = np.asarray(proc.image_std, dtype='float32').reshape(1, 3, 1, 1)

    items = ISB.clips()
    rng = np.random.RandomState(seed)
    chosen = []
    for label in (0, 1):
        pool = [i for i, (_, lab, _) in enumerate(items) if lab == label]
        chosen += list(rng.choice(pool, size=min(n // 2, len(pool)), replace=False))
    chosen.sort()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    tower = ort.InferenceSession(os.path.join(EXPORT, f'{TOWER}-fp32.onnx'), so,
                                 providers=['CPUExecutionProvider'])
    head = ort.InferenceSession(os.path.join(EXPORT, f'{HEAD}-fp32.onnx'), so,
                                providers=['CPUExecutionProvider'])

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    names, labels, probs, sums, valids = [], [], [], [], []
    for k, i in enumerate(chosen):
        path, label, _ = items[i]
        frames, valid = decode_clip(path)
        x = preprocess(frames, mean, std)
        emb = tower.run(None, {'pixel_values': x.astype('float32')})[0]
        p = head.run(None, {'frames': emb[None].astype('float32'),
                            'valid': valid[None].astype('float32')})[0][0]
        shutil.copy2(path, os.path.join(FIXTURE_DIR, os.path.basename(path)))
        names.append(os.path.basename(path)); labels.append(label)
        probs.append(p); sums.append(pixel_checksum(x)); valids.append(valid)
        print(f'  [{k+1}/{len(chosen)}] {os.path.basename(path):<40} y={label} '
              f'p_unsafe={p[1]:.6f}', flush=True)

    np.savez(FIXTURE, clip_names=np.array(names), labels=np.array(labels),
             ref_prob=np.array(probs, dtype='float32'), pixel_checksum=np.array(sums),
             valid=np.array(valids, dtype='float32'), mean=mean, std=std,
             reference='onnxruntime CPU FP32, deployment/export/r48-*-fp32.onnx')
    print(f'\n[fixture] {len(names)} clips -> {os.path.relpath(FIXTURE, ROOT)} '
          f'({sum(os.path.getsize(os.path.join(FIXTURE_DIR, f)) for f in os.listdir(FIXTURE_DIR)) / 1e6:.1f} MB '
          f'incl. clips)')


def check(mode, engine_dir, json_out):
    import pycuda.autoinit  # noqa: F401  (must precede tensorrt)
    import pycuda.driver as cuda
    import tensorrt as trt
    TRU = _load('trt_runner_utils', 'trt-runner-utils.py')
    U = _load('jetson_measure_utils', 'jetson-measure-utils.py')

    f = np.load(FIXTURE, allow_pickle=True)
    mean, std = f['mean'], f['std']
    tower = TRU.TRTRunner(trt, cuda, os.path.join(engine_dir, f'{TOWER}-{mode}.engine'),
                          batch=FRAMES)
    head = TRU.TRTRunner(trt, cuda, os.path.join(engine_dir, f'{HEAD}-{mode}.engine'), batch=1)

    rows, lat = [], []
    for i, name in enumerate(f['clip_names']):
        path = os.path.join(FIXTURE_DIR, str(name))
        t0 = time.perf_counter()
        frames, valid = decode_clip(path)
        x = preprocess(frames, mean, std)
        t_dec = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        emb = tower.run(pixel_values=x.astype('float32'))
        p = head.run(frames=emb[None].astype('float32'),
                     valid=valid[None].astype('float32'))[0]
        t_inf = (time.perf_counter() - t0) * 1000
        ref = f['ref_prob'][i]
        rows.append({'clip': str(name), 'label': int(f['labels'][i]),
                     'ref_p_unsafe': float(ref[1]), 'dev_p_unsafe': float(p[1]),
                     'max_abs_diff': float(np.abs(p - ref).max()),
                     'ref_decision': int(np.argmax(ref)), 'dev_decision': int(np.argmax(p)),
                     'pixels_match': pixels_match(pixel_checksum(x), f['pixel_checksum'][i]),
                     'decode_ms': round(t_dec, 2), 'inference_ms': round(t_inf, 2)})
        lat.append(t_dec + t_inf)

    bad_pixels = [r['clip'] for r in rows if not r['pixels_match']]
    agree = float(np.mean([r['ref_decision'] == r['dev_decision'] for r in rows]))
    worst = max(r['max_abs_diff'] for r in rows)
    print(f'\n  clips                     {len(rows)}')
    print(f'  preprocessing mismatches  {len(bad_pixels)}'
          + ('  <-- decode differs; parity below is NOT a quantisation result'
             if bad_pixels else ''))
    print(f'  decision agreement        {100 * agree:.1f}%   (gate {100 * DECISION_GATE:.0f}%)')
    print(f'  max |p_fp16 - p_fp32|     {worst:.2e}')
    print(f'  end-to-end incl. decode   median {np.median(lat):.1f} ms  '
          f'p95 {np.percentile(lat, 95):.1f} ms')

    payload = {'mode': mode, 'n_clips': len(rows), 'decision_agreement_rate': agree,
               'max_abs_prob_diff': worst, 'preprocessing_mismatches': bad_pixels,
               'reference': str(f['reference']), 'decision_gate': DECISION_GATE,
               'end_to_end_incl_decode_ms': U.percentiles(lat), 'rows': rows,
               'is_device_number': True}
    if json_out:
        print(f'\n[SAVED] {U.write_json_result(f"r48-parity-{mode}", payload, dry_run=False)}')
    if bad_pixels or agree < DECISION_GATE:
        sys.exit(f'PARITY FAIL: agreement {100 * agree:.1f}%, '
                 f'{len(bad_pixels)} preprocessing mismatch(es)')
    print('\nPARITY PASS')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--make-fixture', action='store_true', help='host side; see module docstring')
    ap.add_argument('--n', type=int, default=24, help='clips in the fixture, split evenly by label')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mode', default='fp16', choices=['fp32', 'fp16', 'int8'])
    ap.add_argument('--engine-dir', default=ENGINE_DIR)
    ap.add_argument('--no-json', action='store_true')
    args = ap.parse_args()
    if args.make_fixture:
        make_fixture(args.n, args.seed)
    else:
        if not os.path.exists(FIXTURE):
            raise SystemExit(f'no fixture at {FIXTURE} -- run --make-fixture on the host first')
        check(args.mode, args.engine_dir, not args.no_json)


if __name__ == '__main__':
    main()
