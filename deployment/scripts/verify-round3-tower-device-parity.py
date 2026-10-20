"""FP16-vs-FP32 parity for a swapped round-3 vision tower, on real footage.

verify-r48-device-parity.py cannot do this job: its fixture stores the reference PROBABILITIES of
the published B/32 tower plus the trained head. A round-3 tower has a different embedding and, for
SigLIP 2, no trained head at all, so there is no reference probability to compare against.

What is comparable is the tower's own output. This script checks that the FP16 TensorRT engine
reproduces the FP32 ONNX embedding on the same real frames, which is exactly the claim a quoted
FP16 latency depends on: that the engine being timed computes what the accuracy was measured with.

Host mode writes the reference; device mode checks it. The preprocessed PIXELS are checksummed
first and separately, so "the device decoded different frames" can never be charged to FP16 drift.

Usage:
  .venv/bin/python deployment/scripts/verify-round3-tower-device-parity.py --make-fixture \
      --tower r48-tower-clipB16 --normalizer clipB16
  python3 deployment/scripts/verify-round3-tower-device-parity.py --tower r48-tower-clipB16   # device
"""
import argparse
import os
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
EXPORT = os.path.join(ROOT, 'deployment', 'export')
ENGINE_DIR = os.path.join(EXPORT, 'jetson')
FIXTURE_DIR = os.path.join(ROOT, 'deployment', 'artifacts', 'r48-parity-fixture')
FRAMES = 16
# FP16 has ~3 decimal digits, so an FP32-vs-FP16 gate is a COSINE gate, not a max-abs one; the
# head consumes this vector's direction. The published B/32 path was accepted at this floor.
# One tolerance per precision. An fp32 engine checked against the fp16 tolerance would pass
# without being tested at all: fp16's budget is ~200x looser, so it certifies nothing about a
# graph that is supposed to be reproducing the reference exactly.
TOL = {'fp16': (0.999, 2e-2), 'fp32': (0.99999, 1e-4)}


def _load(name, fname):
    spec = spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalization(normalizer):
    """mean/std for one tower, from its own preprocessing config -- weights are NOT downloaded.

    Host mode only. The values are then stored IN the fixture and read back from it on the device,
    so the two machines cannot normalise differently; nothing is retyped as a constant here.
    """
    EX = _load('tower_table', os.path.join(ROOT, 'kaggle', 'kaggle-extract-tower-features.py'))
    backend, model_id = EX.TOWERS[normalizer]
    if backend == 'hf':
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(model_id)
        return np.asarray(proc.image_mean, 'float32'), np.asarray(proc.image_std, 'float32')
    import json
    from huggingface_hub import hf_hub_download
    repo = model_id.split('hf-hub:', 1)[1]
    cfg = json.load(open(hf_hub_download(repo, 'open_clip_config.json')))['preprocess_cfg']
    return np.asarray(cfg['mean'], 'float32'), np.asarray(cfg['std'], 'float32')


def frames_and_pixels(mean, std, prep):
    """The same 16 real frames and the same preprocessing on host and device."""
    import cv2
    clips = sorted(f for f in os.listdir(FIXTURE_DIR) if f.endswith('.mp4'))
    cap = cv2.VideoCapture(os.path.join(FIXTURE_DIR, clips[0]))
    fr = []
    while len(fr) < FRAMES:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    cap.release()
    assert len(fr) == FRAMES, f'{clips[0]} gave {len(fr)} of {FRAMES} frames'
    x = prep(fr, mean.reshape(1, 3, 1, 1), std.reshape(1, 3, 1, 1))
    return x.astype('float32', copy=False), clips[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tower', required=True, help='engine/ONNX stem, e.g. r48-tower-clipB16')
    ap.add_argument('--normalizer', help='TOWERS key for mean/std; host mode only')
    ap.add_argument('--make-fixture', action='store_true')
    ap.add_argument('--mode', default='fp16')
    args = ap.parse_args()
    PAR = _load('device_parity', 'verify-r48-device-parity.py')
    ref_path = os.path.join(FIXTURE_DIR, f'{args.tower}-reference.npz')

    if args.make_fixture:
        assert args.normalizer, '--normalizer is required when writing the fixture'
        mean, std = normalization(args.normalizer)
        x, clip = frames_and_pixels(mean, std, PAR.preprocess)
        import onnxruntime as ort
        sess = ort.InferenceSession(os.path.join(EXPORT, f'{args.tower}-fp32.onnx'),
                                    providers=['CPUExecutionProvider'])
        emb = sess.run(None, {'pixel_values': x})[0]
        np.savez(ref_path, emb=emb, pixel_checksum=PAR.pixel_checksum(x), clip=clip,
                 normalizer=args.normalizer, mean=mean, std=std)
        print(f'[fixture] {clip}: {emb.shape} embeddings, |emb| mean '
              f'{np.linalg.norm(emb, axis=-1).mean():.4g} -> '
              f'{os.path.relpath(ref_path, ROOT)}')
        return

    f = np.load(ref_path, allow_pickle=True)
    x, clip = frames_and_pixels(f['mean'], f['std'], PAR.preprocess)
    assert clip == str(f['clip']), f'fixture is for {f["clip"]}, this machine read {clip}'
    got_px = PAR.pixel_checksum(x)
    dpx = float(np.abs(np.asarray(got_px) - f['pixel_checksum']).max())
    print(f'[pixels] max |checksum diff| = {dpx:.3g} '
          f'-> {"identical preprocessing" if dpx < 1e-5 else "DIFFERENT PIXELS"}')
    if dpx >= 1e-5:
        raise SystemExit('the device preprocessed different pixels; fix that before judging FP16')

    import pycuda.autoinit  # noqa: F401  (must precede tensorrt)
    import pycuda.driver as cuda
    import tensorrt as trt
    TRU = _load('trt_runner_utils', 'trt-runner-utils.py')
    eng = os.path.join(ENGINE_DIR, f'{args.tower}-{args.mode}.engine')
    got = TRU.TRTRunner(trt, cuda, eng, batch=FRAMES).run(pixel_values=x)
    ref = f['emb']
    nr = np.linalg.norm(ref, axis=-1, keepdims=True)
    ng = np.linalg.norm(got, axis=-1, keepdims=True)
    cos = float(((ref / nr) * (got / ng)).sum(-1).min())
    rel = float(np.abs(ref - got).max() / np.abs(ref).max())
    TOL_COS, TOL_REL = TOL[args.mode]
    ok = cos >= TOL_COS and rel <= TOL_REL
    print(f'[parity] {args.tower} {args.mode} engine vs fp32 ONNX on {clip}: '
          f'min row cosine {cos:.6f}  rel {rel:.3g}  '
          f'-> {"PASS" if ok else "FAIL"} (cos >= {TOL_COS}, rel <= {TOL_REL})')
    if not ok:
        raise SystemExit('FP16 engine does not reproduce the FP32 tower; its latency is not '
                         'the latency of the measured tower')


if __name__ == '__main__':
    main()
