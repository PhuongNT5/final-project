"""Export a B1 head graph at a NON-512 input dimension, for cost measurement only.

Why this exists. Round 3's best row is `siglip2B16`, whose embedding is 768-d. The published head
engine takes 512-d frames and its input shape simply rejects a 768-d tower, so the composed-cadence
loop cannot run at all without a matching head graph. No 768-d head CHECKPOINT exists -- the round-3
sweep saved out-of-fold predictions, not weights -- so this graph carries randomly initialised
weights.

That is sound for the only question asked of it. Latency on a fixed-shape TensorRT engine is a
function of the graph's shapes and operators, not of the values in its weights. It is NOT sound for
anything else: this graph's output is noise, so the harness is told the probability is meaningless
and no accuracy, threshold or decision may be read from a run that uses it.

The guard against self-deception is the parameter count. The round-3 sweep logged 233,858
parameters for the 768-d transformer head and 201,090 for the 512-d one. If the graph built here
does not match the count for its dimension, it is not the head whose accuracy was measured and the
export refuses to write.

Usage: .venv/bin/python deployment/scripts/export-b1-head-cost-only.py --in_dim 768 [--force]
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

EXPORT_DIR = os.path.join(ROOT, 'deployment', 'export')
OPSET = 18
FRAMES, TOL, SEEDS = 16, 1e-5, 3
# From the round-3 sweep log; the key is the head input dimension.
EXPECTED_PARAMS = {512: 201_090, 768: 233_858}


class SeedEnsemble(nn.Module):
    """Same structure as the deployed 3-seed graph, so the measured cost is the deployed cost."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, frames, valid):
        p = torch.stack([torch.softmax(m(frames, valid), 1) for m in self.models])
        return p.mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dim', type=int, required=True)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    out = os.path.join(EXPORT_DIR, f'r48-b1-head-cost{args.in_dim}-3seed-fp32.onnx')
    if os.path.exists(out) and not args.force:
        raise SystemExit(f'{out} exists; pass --force to overwrite')

    ms = []
    for s in range(SEEDS):
        torch.manual_seed(s)
        m = T.build('B1', args.in_dim)
        m.eval()
        ms.append(m)
    one = sum(p.numel() for p in ms[0].parameters())
    want = EXPECTED_PARAMS.get(args.in_dim)
    print(f'[head] B1 at in_dim={args.in_dim}: {one:,} params per seed x {SEEDS} seeds')
    if want is not None and one != want:
        raise SystemExit(f'{one:,} params but the sweep logged {want:,} for in_dim={args.in_dim}; '
                         'this is not the head whose accuracy was measured')
    if want is None:
        print(f'[head] no sweep-logged count for in_dim={args.in_dim}; nothing to check against')

    model = SeedEnsemble(ms).eval()
    frames = torch.randn(1, FRAMES, args.in_dim)
    valid = torch.ones(1, FRAMES)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    torch.onnx.export(model, (frames, valid), out,
                      input_names=['frames', 'valid'], output_names=['probs'],
                      opset_version=OPSET, dynamo=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    with torch.no_grad():
        ref = model(frames, valid).numpy()
    got = sess.run(None, {'frames': frames.numpy(), 'valid': valid.numpy()})[0]
    worst = float(np.abs(ref - got).max())
    print(f"[parity] max |torch - onnxruntime| = {worst:.3g} -> "
          f"{'PASS' if worst <= TOL else 'FAIL'} (tol {TOL})")
    print(f'[export] -> {os.path.relpath(out, ROOT)}  ({os.path.getsize(out) / 1e6:.1f} MB)')
    print('[export] COST ONLY: untrained weights. Timings from this graph are valid; its output '
          'probability is noise and must never be reported as a prediction.')
    if worst > TOL:
        raise SystemExit('parity check failed')


if __name__ == '__main__':
    main()
