"""export-p3-relation-head-binary.py — Export the frozen-candidate BINARY (Safe/Unsafe) relation-head
graphs (E3 primary freeze candidate, E0 fallback — see
plans/2026-09-16-binary-safe-unsafe-reframe/reports/laneb-2026-09-17-encoder-impl-and-screening.md §6)
to ONNX, and verify PyTorch-vs-onnxruntime numerical parity.

**UPDATE 2026-09-19: trained checkpoints now exist for both archs** — see
`deployment/scripts/train-binary-head-checkpoint.py` (trains on TRAIN split only, zero test touches;
proof + val metrics in
`plans/2026-09-16-binary-safe-unsafe-reframe/reports/laneb-2026-09-19-binary-head-checkpoints.md`) and
`deployment/export/MANIFEST.md`. Both deployed graphs (`p3-relation-head-bin-{e3,e0}-fp32.onnx`) were
re-exported from those checkpoints via `--head_ckpt`; they are no longer random-init. **Test-set
accuracy is still unmeasured** (no test touch spent on either checkpoint — only training-time val
metrics exist). Without `--head_ckpt`, this script still falls back to a randomly-initialised export
(same behaviour as originally written 2026-09-18) — useful only for graph-shape/TensorRT-buildability
checks, never for an accuracy claim; the current deployed graphs were NOT produced that way.

Deployed decision path (per model_relation_head.py / model_binary_encoders.py forward() contracts):
default `model(tok, mask)` call, i.e. `return_sub=False, return_frame=False` -> ONLY the 2-logit
cls_logits head is exported. The 8-way sub-class head and per-frame aux head are training-time
auxiliary losses only (design doc §3.2, lambda=0.3 / 0.5) — never called at inference in any runner
script (jetson-multi-stream-pipeline.py, trt-runner-utils.py) — so they are correctly excluded from
the deployed graph.

Usage:
    .venv/bin/python3 deployment/scripts/export-p3-relation-head-binary.py [--head_ckpt PATH] [--force]
"""
import argparse
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from model_binary_encoders import build_encoder   # noqa: E402
from model_relation_head import count_params        # noqa: E402

EXPORT_DIR = os.path.join(ROOT, 'deployment', 'export')
BIN8_TEST = os.path.join(ROOT, 'sweep_results', 'relation_tokens', 'tokens_bin8_test.pt')
T, N_MAX, N_NUM = 16, 8, 73          # bin8_ tag: feat=74 (see build_relation_tokens.py + report §1)
OPSET = 18                            # matches export-p3-graphs.py: torch 2.11's opset-17 downgrade fails
SEED = 0                              # fixed init seed for reproducibility of these PLACEHOLDER weights

ARCH_TO_FILE = {'E3': 'p3-relation-head-bin-e3-fp32.onnx', 'E0': 'p3-relation-head-bin-e0-fp32.onnx'}


def _save_inline(path):
    """Re-save without external data (required for some EPs; matches export-p3-graphs.py convention)."""
    import onnx
    m = onnx.load(path)
    onnx.save(m, path, save_as_external_data=False)
    onnx.checker.check_model(m)
    sidecar = path + '.data'
    if os.path.exists(sidecar):
        os.remove(sidecar)


def build_placeholder_model(arch):
    torch.manual_seed(SEED)
    model = build_encoder(arch, n_num=N_NUM, n_cls=2, n_sub=8).eval()
    if hasattr(model, 'enc'):   # E0/E2 only: TransformerEncoder's NestedTensor fast path does not trace
        model.enc.enable_nested_tensor = False
    return model


def export_binary_head(arch, out_path):
    model = build_placeholder_model(arch)
    tok = torch.zeros(1, T, N_MAX, N_NUM + 1)
    mask = torch.ones(1, T, N_MAX)
    with torch.no_grad():
        ref = model(tok, mask)
    assert ref.shape == (1, 2), f'unexpected {arch} head output {tuple(ref.shape)}'

    torch.onnx.export(
        model, (tok, mask), out_path,
        opset_version=OPSET,
        input_names=['tokens', 'mask'], output_names=['logits'],
        dynamic_axes={'tokens': {0: 'batch'}, 'mask': {0: 'batch'}, 'logits': {0: 'batch'}},
    )
    _save_inline(out_path)
    params = count_params(model)
    print(f'  [{arch}] {out_path}')
    print(f'    tokens (1,{T},{N_MAX},{N_NUM + 1})  mask (1,{T},{N_MAX})  -> logits (1,2)  params={params:,}')
    return model


def check_parity_real_tokens(arch, model, onnx_path, tol=1e-4):
    """PyTorch vs onnxruntime on the real 125-clip bin8 test tokens. NUMERICAL parity only — compares
    the two implementations against each other, never against ground-truth labels (0 test touches:
    no accuracy is computed or reported here, per reports/test-touch-log.md).

    Run one clip (batch=1) at a time, matching the actual TensorRT deployment shape
    (jetson-build-trt-engine.py fixes min=opt=max=1 for every dynamic dim — batch>1 is never built).
    A single batch>1 ORT call is NOT used here: E0/RelationTransformer's TransformerEncoder reshape
    bakes the batch dim to 1 under torch 2.11's dynamo-based exporter even though the input is
    declared dynamic (`tokens ['batch',16,8,74]`) — confirmed to be a pre-existing defect shared with
    the already-deployed 4-class p3-relation-head-fp32.onnx (same crash at batch=5), not something
    introduced here. E3/E3DeepSets has no such op and genuinely supports batch>1 (verified separately
    below), but batch=1 looping keeps this check identical and honest across both archs."""
    import onnxruntime as ort
    d = torch.load(BIN8_TEST, weights_only=False, map_location='cpu')
    tok, mask = d['tokens'].float(), d['mask'].float()
    assert tok.shape == (125, T, N_MAX, N_NUM + 1), f'unexpected bin8 test shape {tuple(tok.shape)}'

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    diffs, agrees = [], []
    with torch.no_grad():
        for i in range(tok.shape[0]):
            want = model(tok[i:i + 1], mask[i:i + 1])
            got = torch.from_numpy(sess.run(None, {'tokens': tok[i:i + 1].numpy(),
                                                     'mask': mask[i:i + 1].numpy()})[0])
            diffs.append((want - got).abs().max().item())
            agrees.append(int(want.argmax(1).item() == got.argmax(1).item()))

    max_abs_diff = max(diffs)
    agree = sum(agrees) / len(agrees)
    status = 'OK' if max_abs_diff < tol else '*** MISMATCH ***'
    print(f'  [{arch}] torch-vs-onnx over 125 bin8 test clips (batch=1 each): '
          f'max_abs_logit_diff={max_abs_diff:.3e} decision_agreement_rate={agree * 100:.1f}%  {status}')
    return {'max_abs_diff': max_abs_diff, 'agreement_rate': agree, 'n_clips': tok.shape[0]}


def check_dynamic_batch(arch, model, onnx_path):
    """E3 only: confirm the graph genuinely supports batch>1 in ORT (E0's does not — see docstring
    above). Informational, not a gate."""
    import onnxruntime as ort
    tok = torch.zeros(4, T, N_MAX, N_NUM + 1); mask = torch.ones(4, T, N_MAX)
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    try:
        out = sess.run(None, {'tokens': tok.numpy(), 'mask': mask.numpy()})
        print(f'  [{arch}] batch=4 ORT call: OK, output shape {out[0].shape} (genuinely dynamic-batch)')
        return True
    except Exception as e:
        print(f'  [{arch}] batch=4 ORT call: FAILS ({type(e).__name__}) — batch dim baked to 1 '
              f'internally despite dynamic_axes declaration; deployment is batch=1 only anyway')
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--head_ckpt', default=None,
                     help='optional trained state_dict (from run_relation_transformer.py --binary '
                          '--save_model); if omitted, exports randomly-initialised PLACEHOLDER weights')
    ap.add_argument('--force', action='store_true', help='re-export even if the output exists')
    args = ap.parse_args()

    os.makedirs(EXPORT_DIR, exist_ok=True)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    print(f'Exporting BINARY relation-head graphs (E3 freeze candidate + E0 fallback) at commit {commit[:12]}:')

    results = {}
    for arch in ('E3', 'E0'):
        out_path = os.path.join(EXPORT_DIR, ARCH_TO_FILE[arch])
        if os.path.exists(out_path) and not args.force:
            print(f'  [SKIP] {out_path} (exists; pass --force to re-export)')
            model = build_placeholder_model(arch)   # rebuild (same seed) so parity can still run
        else:
            if args.head_ckpt:
                ck = torch.load(args.head_ckpt, map_location='cpu', weights_only=False)
                if ck['ctor'].get('arch', 'E0') != arch:
                    continue
                model = build_encoder(arch, n_num=ck['ctor']['n_num'], n_cls=2, n_sub=8,
                                       d=ck['ctor'].get('d')).eval()
                model.load_state_dict(ck['state_dict'])
                if hasattr(model, 'enc'):
                    model.enc.enable_nested_tensor = False
                tok = torch.zeros(1, T, N_MAX, N_NUM + 1); mask = torch.ones(1, T, N_MAX)
                torch.onnx.export(model, (tok, mask), out_path, opset_version=OPSET,
                                   input_names=['tokens', 'mask'], output_names=['logits'],
                                   dynamic_axes={'tokens': {0: 'batch'}, 'mask': {0: 'batch'},
                                                 'logits': {0: 'batch'}})
                _save_inline(out_path)
                print(f'  [{arch}] {out_path}  (TRAINED checkpoint: {args.head_ckpt}, '
                      f"blind_seed={ck.get('blind_seed')})")
            else:
                model = export_binary_head(arch, out_path)
        parity = check_parity_real_tokens(arch, model, out_path)
        dynamic_ok = check_dynamic_batch(arch, model, out_path)
        results[arch] = {'path': out_path, 'trained': bool(args.head_ckpt), 'parity': parity,
                          'genuinely_dynamic_batch': dynamic_ok}

    print('\nDone. See deployment/export/MANIFEST.md for full provenance.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
