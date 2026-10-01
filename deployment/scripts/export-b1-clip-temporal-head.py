"""Export the R48 deployment head (B1: whole-frame CLIP -> temporal transformer) to ONNX.

Why B1 and not ESTA: G14 was adjudicated on 2026-09-29 and FAILED (ESTA -0.5 pp vs B1w, McNemar
p=0.683). The entity decomposition is not in the deployed accuracy path. B1 is chosen over the
statistically indistinguishable B1w (84.2% vs 84.4%) on the pre-registered *edge* criterion --
smallest model -- not on accuracy; its own OOF number is the one to quote, never max(B1, B1w).

The exported graph is the 3-seed ENSEMBLE, because the reported OOF numbers are 3-seed ensemble
numbers: deploying a single seed would ship something whose accuracy was never measured. The seeds
are stacked into one graph so the edge measurement covers what actually runs.

Parity is verified against PyTorch rather than asserted; the script exits non-zero if it drifts.

Usage: .venv/bin/python deployment/scripts/export-b1-clip-temporal-head.py [--force]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import esta_training as T                                   # noqa: E402
from model_entity_spatiotemporal import count_params        # noqa: E402

EXPORT_DIR = os.path.join(ROOT, 'deployment', 'export')
CKPT = os.path.join(ROOT, 'sweep_results/relation_tokens/deploy/b1-deploy-3seed.pt')
OUT = os.path.join(EXPORT_DIR, 'r48-b1-clip-temporal-3seed-fp32.onnx')
OPSET = 18            # matches export-p3-graphs.py: torch 2.11's opset-17 downgrade fails
FRAMES, TOL = 16, 1e-5


class SeedEnsemble(nn.Module):
    """Mean softmax over the 3 training seeds -- the same ensemble the OOF numbers were computed with."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, frames, valid):
        p = torch.stack([torch.softmax(m(frames, valid), 1) for m in self.models])
        return p.mean(0)


def load_ensemble():
    ck = torch.load(CKPT, weights_only=False, map_location='cpu')
    if ck['model'] != 'B1':
        raise SystemExit(f"checkpoint is {ck['model']}, not B1 -- refusing to export it as the B1 graph")
    ms = []
    for st in ck['states']:
        m = T.build('B1', ck['in_dim'])
        m.load_state_dict(st)
        m.eval()
        ms.append(m)
    return SeedEnsemble(ms).eval(), ck


def check_parity(model, path):
    """PyTorch vs onnxruntime on inputs that include the degenerate all-invalid clip."""
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for valid in (torch.ones(2, FRAMES), torch.zeros(2, FRAMES),
                  (torch.rand(2, FRAMES, generator=g) > 0.5).float()):
        frames = torch.randn(2, FRAMES, model.models[0].proj[0].in_features, generator=g)
        with torch.no_grad():
            ref = model(frames, valid).numpy()
        got = sess.run(None, {'frames': frames.numpy(), 'valid': valid.numpy()})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='overwrite an existing graph')
    args = ap.parse_args()
    if os.path.exists(OUT) and not args.force:
        raise SystemExit(f'{OUT} exists; pass --force to overwrite')

    model, ck = load_ensemble()
    d = ck['in_dim']
    print(f"[export] B1 3-seed ensemble | in_dim={d} | {count_params(model):,} params "
          f"({count_params(model.models[0]):,} per seed) | val macro/seed: "
          f"{', '.join(f'{v:.1f}' for v in ck['val_macro'])}")

    os.makedirs(EXPORT_DIR, exist_ok=True)
    torch.onnx.export(
        model, (torch.randn(1, FRAMES, d), torch.ones(1, FRAMES)), OUT,
        input_names=['frames', 'valid'], output_names=['prob'], opset_version=OPSET,
        dynamo=False,
        dynamic_axes={'frames': {0: 'batch'}, 'valid': {0: 'batch'}, 'prob': {0: 'batch'}})

    worst = check_parity(model, OUT)
    ok = worst <= TOL
    print(f"[parity] max |torch - onnxruntime| = {worst:.3g} over 3 validity patterns "
          f"(all-valid, all-invalid, random) -> {'PASS' if ok else 'FAIL'} (tol {TOL})")
    print(f"[export] -> {os.path.relpath(OUT, ROOT)}  ({os.path.getsize(OUT) / 1e6:.2f} MB)")
    if not ok:
        raise SystemExit('parity check failed -- the exported graph is not the measured model')


if __name__ == '__main__':
    main()
