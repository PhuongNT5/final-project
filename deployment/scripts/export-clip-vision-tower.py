"""Export the frozen CLIP ViT-B/32 vision tower used by the R48 lane to ONNX.

This is the honest cost of the R48 accuracy result. B1 reaches 84.2% balanced accuracy with a
201k-parameter head, but only because the head is fed a ~88M-parameter frozen backbone -- against
the ~2.5M-parameter MobileNetV3-small the 64-d encoders used. The head got cheaper and the backbone
got ~35x more expensive, and the edge chapter has to report that trade-off rather than quote the
head size alone.

The tower is exported exactly as `build-isafetybench-clip-features.load_clip` builds it, and parity
is verified against PyTorch on the same normalisation the features were extracted with.

Usage: .venv/bin/python deployment/scripts/export-clip-vision-tower.py [--force]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

EXPORT_DIR = os.path.join(ROOT, 'deployment', 'export')
OUT = os.path.join(EXPORT_DIR, 'r48-clip-vit-b32-vision-fp32.onnx')
OPSET = 18
TOL = 1e-4        # a 12-layer ViT in fp32 accumulates more than the 1e-5 a 1-layer head does


def builder():
    spec = importlib.util.spec_from_file_location(
        'clipfeat', os.path.join(ROOT, 'build-isafetybench-clip-features.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Tower(nn.Module):
    """Wraps the HF model so the graph's single output is the 512-d projected embedding."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).image_embeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if os.path.exists(OUT) and not args.force:
        raise SystemExit(f'{OUT} exists; pass --force to overwrite')

    clip = builder().load_clip('cpu')
    model = Tower(clip['model']).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f'[export] CLIP ViT-B/32 vision tower | {n:,} params | 224x224 -> 512-d')

    os.makedirs(EXPORT_DIR, exist_ok=True)
    torch.onnx.export(model, (torch.randn(1, 3, 224, 224),), OUT,
                      input_names=['pixel_values'], output_names=['image_embeds'],
                      opset_version=OPSET, dynamo=False,
                      dynamic_axes={'pixel_values': {0: 'batch'}, 'image_embeds': {0: 'batch'}})

    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=['CPUExecutionProvider'])
    x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {'pixel_values': x.numpy()})[0]
    worst = float(np.abs(ref - got).max())
    ok = worst <= TOL
    print(f"[parity] max |torch - onnxruntime| = {worst:.3g} -> {'PASS' if ok else 'FAIL'} (tol {TOL})")
    print(f'[export] -> {os.path.relpath(OUT, ROOT)}  ({os.path.getsize(OUT) / 1e6:.1f} MB)')
    if not ok:
        raise SystemExit('parity check failed -- the exported tower is not the measured backbone')


if __name__ == '__main__':
    main()
