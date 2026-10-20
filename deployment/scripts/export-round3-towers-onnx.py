"""Export the round-3 frozen vision towers to ONNX so the device can price them.

Why a second exporter instead of editing export-clip-vision-tower.py: that script produced the
published B/32 graph whose 23.386 ms is a reported number, and it is hardcoded to that one
checkpoint on purpose. This script leaves it alone and reads the SAME tower table the feature
extraction used (kaggle/kaggle-extract-tower-features.py), so a tower measured here is by
construction the tower whose accuracy was measured there, not a re-specified lookalike.

Resolution is fixed at 224x224 and is not an option: the round-3 features were extracted from a
224x224 frame cache, so a tower exported at any other resolution could not be paired with its
balanced accuracy.

How parity is judged, and why not the way the B/32 exporter judges it
--------------------------------------------------------------------
The B/32 exporter gates on `max|torch - onnxruntime| <= 1e-4` over a `torch.randn` input. Measured
2026-10-19: that gate does not transfer. `mobileclip_s1` emits embeddings with norm ~8,200 and
elements up to ~3,800, so 1e-4 absolute is ~2.6e-8 relative -- below fp32 resolution, a gate no
correct graph can pass. Worse, on `randn` input PyTorch eager disagrees with ITSELF across batch
sizes by 2.6% relative on that tower, so the reference was not a fixed target either.

So this gates on (a) RELATIVE max difference, (b) a per-row cosine floor -- the head consumes the
direction of this vector -- and (c) REAL preprocessed frames from the device parity fixture through
the same `preprocess` the device runs, because the drift that matters is drift on the data
distribution. Each tower is normalised with ITS OWN mean/std, taken from the extractor's loader.
The eager batch-16-vs-one-at-a-time self-consistency is measured and printed beside the parity
numbers: where it dominates, the instability belongs to the tower, not to the export.

Usage:
  .venv/bin/python deployment/scripts/export-round3-towers-onnx.py \
      --towers clipB16 siglip2B16 mobileclip_s1 [--force]
"""
import argparse
import importlib.util
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
EXPORT_DIR = os.path.join(ROOT, 'deployment', 'export')
EXTRACTOR = os.path.join(ROOT, 'kaggle', 'kaggle-extract-tower-features.py')
FIXTURE_DIR = os.path.join(ROOT, 'deployment', 'artifacts', 'r48-parity-fixture')
RES, OPSET, DEPLOYED_BATCH = 224, 18, 16
TOL_REL = 1e-4        # relative to max|reference|, so it means the same thing on every tower
TOL_COS = 0.9999      # the floor the decode control used; the head reads this vector's direction


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Tower(nn.Module):
    """One input, one output: the projected embedding the head was trained on. `fwd` is the
    extractor's own closure, so this is the same object the feature files store."""

    def __init__(self, fwd, module):
        super().__init__()
        self.fwd = fwd
        self.m = module          # registered so .parameters() reaches the real weights

    def forward(self, pixel_values):
        return self.fwd(pixel_values)


def build(name, ex):
    backend, model_id = ex.TOWERS[name]
    fwd, mean, std = (ex.load_hf if backend == 'hf' else ex.load_open_clip)(model_id, 'cpu')
    module = next(c.cell_contents for c in fwd.__closure__
                  if isinstance(c.cell_contents, nn.Module))
    tower = Tower(fwd, module).eval()
    with torch.no_grad():
        dim = int(tower(torch.zeros(1, 3, RES, RES)).shape[-1])
    return tower, dim, backend, model_id, mean.numpy(), std.numpy()


def real_frames(mean, std, n, prep):
    """n real preprocessed frames from the device parity fixture -- in-distribution input."""
    clips = sorted(f for f in os.listdir(FIXTURE_DIR) if f.endswith('.mp4'))
    assert clips, f'no fixture clips in {FIXTURE_DIR}'
    cap = cv2.VideoCapture(os.path.join(FIXTURE_DIR, clips[0]))
    frames = []
    while len(frames) < n:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    assert len(frames) == n, f'{clips[0]} gave {len(frames)} of {n} frames'
    return prep(frames, mean, std).astype('float32', copy=False), clips[0]


def compare(ref, got):
    scale = float(np.abs(ref).max())
    nr = np.linalg.norm(ref, axis=-1, keepdims=True)
    ng = np.linalg.norm(got, axis=-1, keepdims=True)
    return {'abs': float(np.abs(ref - got).max()),
            'rel': float(np.abs(ref - got).max() / scale),
            'cos': float(((ref / nr) * (got / ng)).sum(-1).min()),
            'ref_max_abs': scale}


def export_one(name, ex, prep, force, batch):
    out = os.path.join(EXPORT_DIR, f'r48-tower-{name}-fp32.onnx')
    if os.path.exists(out) and not force:
        print(f'[skip] {os.path.relpath(out, ROOT)} exists; --force to overwrite')
        return True
    tower, dim, backend, model_id, mean, std = build(name, ex)
    n = sum(p.numel() for p in tower.parameters())
    print(f'[export] {name} | {backend} | {model_id} | {n:,} params in the loaded module '
          f'| {RES}x{RES} -> {dim}-d', flush=True)

    x16, clip = real_frames(mean, std, DEPLOYED_BATCH, prep)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    torch.onnx.export(tower, (torch.from_numpy(x16[:batch]).clone(),), out,
                      input_names=['pixel_values'], output_names=['image_embeds'],
                      opset_version=OPSET, dynamo=False,
                      dynamic_axes={'pixel_values': {0: 'batch'}, 'image_embeds': {0: 'batch'}})

    # Parameters in the EXPORTED graph, not in the loaded module: the loader count includes the
    # text tower, which is never exported and never runs on the device. Measured 2026-10-19 this
    # is the figure that matters -- clipB16 exports 86,192,640 params against the deployed B/32's
    # 87,849,216, i.e. 1.9% FEWER parameters for 4x the arithmetic, so a parameter count is not a
    # cost and must not be quoted as one.
    import onnx
    g = onnx.load(out, load_external_data=False).graph
    print(f'[export] {name}: {sum(int(np.prod(i.dims)) for i in g.initializer):,} params in the '
          f'exported graph')

    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    t = torch.from_numpy(x16)
    with torch.no_grad():
        e16 = tower(t).numpy()
        e1 = np.concatenate([tower(t[i:i + 1]).numpy() for i in range(DEPLOYED_BATCH)])
    g16 = sess.run(None, {'pixel_values': x16})[0]
    g1 = sess.run(None, {'pixel_values': x16[:1]})[0]

    par = compare(e16, g16)                      # the gate: deployed batch, real pixels
    dyn = compare(e1[:1], g1)                    # does the batch axis survive tracing
    self_ = compare(e16, e1)                     # the tower's own batch instability, disclosed
    ok = par['rel'] <= TOL_REL and par['cos'] >= TOL_COS
    print(f"[parity] {name} on real frames from {clip}, max|ref| = {par['ref_max_abs']:.4g}")
    print(f"[parity]   onnx vs torch, batch {DEPLOYED_BATCH} (deployed): rel {par['rel']:.3g}  "
          f"abs {par['abs']:.3g}  min row cos {par['cos']:.6f}  "
          f"-> {'PASS' if ok else 'FAIL'} (rel <= {TOL_REL}, cos >= {TOL_COS})")
    print(f"[parity]   onnx vs torch, batch 1: rel {dyn['rel']:.3g}  cos {dyn['cos']:.6f}  "
          f"-> dynamic batch {'YES' if dyn['rel'] <= TOL_REL else 'NO'}")
    print(f"[parity]   torch vs itself, batch {DEPLOYED_BATCH} vs one-at-a-time: "
          f"rel {self_['rel']:.3g}  cos {self_['cos']:.6f}  (the tower's own numerics, not ours)")
    print(f'[export] -> {os.path.relpath(out, ROOT)}  ({os.path.getsize(out) / 1e6:.1f} MB)',
          flush=True)
    if not ok:
        os.rename(out, out + '.PARITY_FAILED')
        print(f'[export] renamed {os.path.basename(out)}.PARITY_FAILED so no engine can be built')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--towers', nargs='+', required=True)
    ap.add_argument('--batch', type=int, default=1,
                    help='example batch to trace at; 1 matches the published B/32 export and '
                         'leaves the batch axis genuinely exercised by the checks')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    ex = load_module('tower_table', EXTRACTOR)
    par = load_module('device_parity', os.path.join(_HERE, 'verify-r48-device-parity.py'))
    bad = [n for n in args.towers
           if not export_one(n, ex, par.preprocess, args.force, args.batch)]
    if bad:
        raise SystemExit(f'parity failed, no engine may be built for: {", ".join(bad)}')
    print('[export] all requested towers passed parity')


if __name__ == '__main__':
    main()
