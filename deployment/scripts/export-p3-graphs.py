"""
export-p3-graphs.py — Export the P3 (YOLO26 entity + RelationTransformer) edge graphs to ONNX.

Chapter 5 currently benchmarks P1 (frozen-COCO backbone + pairwise-cascade heads).  P3 is the
pipeline that carries the headline accuracy, and it has a *different* and heavier edge cost:

    stage A1  YOLO26 entity detector           16 frames  @ 800x800   (exported by ultralytics)
    stage A2  MobileNetV3-Small appearance     <=128 crops @ 224x224  <-- exported here
    stage B   RelationTransformer              1 clip, (16,8,74) tokens  <-- exported here

Stage A2 is the part with no P1 analogue: `tokens_yolo26app_*.pt` carry feat=74 =
[class, xc, yc, w, h, conf, iou_walkway, iou_restricted, foot_in_walkway, norm_dist] + app_0..63,
so every box in every sampled frame goes through a MobileNet forward pass.  T=16 frames x
N_MAX=8 boxes = up to 128 crops per clip, on top of detection.  Benchmarking P3 without stage A2
would understate the true edge cost.

The 576->64 random projection (numpy default_rng(0), /sqrt(64)) is folded into the exported graph
as a constant Linear, so the ONNX output is directly the app_0..63 the Transformer expects.

Usage:
    python deployment/scripts/export-p3-graphs.py [--head_ckpt PATH] [--force]
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from model_relation_head import RelationTransformer          # noqa: E402
import build_relation_tokens as BRT                          # noqa: E402

EXPORT_DIR = os.path.join(ROOT, 'deployment', 'export')
DEFAULT_HEAD = os.path.join(ROOT, 'sweep_results', 'relation_tokens', 'canonical',
                            'p3_blind_relation_head.pt')
DEFAULT_DET = os.path.join(ROOT, 'sweep_results', 'entity_detector', 'yolo26_best.pt')
T, N_MAX = BRT.T, BRT.N_MAX
# torch 2.11's exporter emits opset 18 for these graphs and its downgrade-to-17 pass fails
# (onnx version_converter: 'No initializer or constant input to node found'), so 17 is not
# reachable here.  TensorRT >= 8.6 parses opset 18; the P1 graphs remain at 17.
OPSET = 18


class AppearanceEmbedder(nn.Module):
    """MobileNetV3-Small features -> avgpool -> flatten -> fixed random projection to APP_DIM.

    Wraps exactly what build_relation_tokens.get_appearance() builds, so the exported graph and the
    graph that produced the training tokens are the same computation (the projection is seeded).
    """

    def __init__(self):
        super().__init__()
        app = BRT.get_appearance('cpu', backbone='mobilenet')
        self.net = app['feat']
        self.proj = nn.Linear(app['proj'].shape[0], app['proj'].shape[1], bias=False)
        self.proj.weight.data.copy_(app['proj'].t())

    def forward(self, crops):                 # (K,3,224,224) ImageNet-normalised
        return self.proj(self.net(crops))     # (K, APP_DIM)


def export_detector(pt_path, out_path, imgsz=None):
    """Stage A1 via ultralytics' own exporter, pinned to the IMGSZ the tokens were built at.

    Exporting at any other imgsz would benchmark a detector that never produced a token, so the
    default stays at BRT.IMGSZ.  `imgsz` is the opt-in escape hatch used ONLY by --cost_imgsz to
    measure the detector cost curve; the graphs it writes are cost probes, never accuracy claims.
    """
    from ultralytics import YOLO
    sz = imgsz or BRT.IMGSZ
    res = YOLO(pt_path).export(format='onnx', imgsz=sz, opset=OPSET, simplify=False)
    os.replace(str(res), out_path)
    print(f'  detector      : {out_path}')
    if imgsz is None:
        print(f'    imgsz={sz} (== build_relation_tokens.IMGSZ), {T} frames per decision')
    else:
        print(f'    imgsz={sz} COST PROBE ONLY — no tokens were ever built at this size, so this '
              f'graph carries no accuracy meaning')


def _save_inline(path):
    """Re-save without external data — required for the CoreML EP (ORT issue #18518).

    Also removes the sidecar the first save wrote: once the weights are inline the .onnx.data is
    orphaned, and shipping it to the Jetson would suggest the graph still depends on it.
    """
    import onnx
    m = onnx.load(path)
    onnx.save(m, path, save_as_external_data=False)
    onnx.checker.check_model(m)
    sidecar = path + '.data'
    if os.path.exists(sidecar):
        os.remove(sidecar)


def export_relation_head(ckpt_path, out_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    n_num = ck['ctor']['n_num']
    model = RelationTransformer(n_num=n_num, t=T, n_max=N_MAX)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    # TransformerEncoder's NestedTensor fast path does not trace; force the dense path.
    model.enc.enable_nested_tensor = False

    tok = torch.zeros(1, T, N_MAX, n_num + 1)
    mask = torch.ones(1, T, N_MAX)
    with torch.no_grad():
        ref = model(tok, mask)
    assert ref.shape == (1, 4), f'unexpected head output {tuple(ref.shape)}'

    torch.onnx.export(
        model, (tok, mask), out_path,
        opset_version=OPSET,
        input_names=['tokens', 'mask'], output_names=['logits'],
        dynamic_axes={'tokens': {0: 'batch'}, 'mask': {0: 'batch'}, 'logits': {0: 'batch'}},
    )
    _save_inline(out_path)
    print(f'  relation head : {out_path}')
    print(f'    tokens (1,{T},{N_MAX},{n_num + 1})  mask (1,{T},{N_MAX})  -> logits (1,4)')
    print(f"    blind_seed={ck.get('blind_seed')} blind_top1={ck.get('blind_top1')}")
    return model, (tok, mask), ref


def export_appearance(out_path):
    model = AppearanceEmbedder().eval()
    crops = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        ref = model(crops)
    assert ref.shape == (1, BRT.APP_DIM), f'unexpected appearance output {tuple(ref.shape)}'

    torch.onnx.export(
        model, crops, out_path,
        opset_version=OPSET,
        input_names=['crops'], output_names=['appearance'],
        dynamic_axes={'crops': {0: 'boxes'}, 'appearance': {0: 'boxes'}},
    )
    _save_inline(out_path)
    print(f'  appearance    : {out_path}')
    print(f'    crops (K,3,224,224) -> appearance (K,{BRT.APP_DIM});'
          f' K <= {T * N_MAX} per clip')
    return model, (crops,), ref


def check_parity(name, model, inputs, ref, onnx_path, tol=1e-4):
    """Torch vs onnxruntime on the same input. A silently-wrong export is worse than no export."""
    try:
        import onnxruntime as ort
    except ImportError:
        print(f'  [{name}] parity SKIPPED — onnxruntime not installed')
        return
    torch.manual_seed(0)
    rnd = tuple(torch.randn_like(x) for x in inputs)
    if name == 'relation-head':
        rnd = (rnd[0], (torch.rand(1, T, N_MAX) > 0.3).float())   # mask must stay 0/1
    with torch.no_grad():
        want = model(*rnd)
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    feed = {i.name: r.numpy() for i, r in zip(sess.get_inputs(), rnd)}
    got = torch.from_numpy(sess.run(None, feed)[0])
    d = (want - got).abs().max().item()
    print(f'  [{name}] torch-vs-onnx max abs diff = {d:.3e}  {"OK" if d < tol else "*** MISMATCH ***"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--head_ckpt', default=DEFAULT_HEAD,
                    help='checkpoint from run_relation_transformer.py --save_model')
    ap.add_argument('--detector', default=DEFAULT_DET, help='stage A1 YOLO26 weights')
    ap.add_argument('--force', action='store_true', help='re-export even if the output exists')
    ap.add_argument('--cost_imgsz', type=int, nargs='*', default=None,
                    help='ALSO export the detector at these extra imgsz, as p3-detector-fp32-imgszNNN.onnx. '
                         'Cost-curve probes for benchmark-p3-onnx.py --sweep. The accuracy of the '
                         'pipeline at these sizes is UNMEASURED — the tokens, and therefore the 54/64, '
                         'exist only at imgsz=%d.' % BRT.IMGSZ)
    args = ap.parse_args()

    os.makedirs(EXPORT_DIR, exist_ok=True)
    det_out = os.path.join(EXPORT_DIR, 'p3-detector-fp32.onnx')
    app_out = os.path.join(EXPORT_DIR, 'p3-appearance-fp32.onnx')
    head_out = os.path.join(EXPORT_DIR, 'p3-relation-head-fp32.onnx')

    print('Exporting P3 edge graphs (A1 detector, A2 appearance, B relation head):')
    if os.path.exists(det_out) and not args.force:
        print(f'  [SKIP] {det_out}')
    elif not os.path.exists(args.detector):
        print(f'  [MISSING] {args.detector}', file=sys.stderr)
    else:
        export_detector(args.detector, det_out)

    for sz in (args.cost_imgsz or []):
        out = os.path.join(EXPORT_DIR, f'p3-detector-fp32-imgsz{sz}.onnx')
        if os.path.exists(out) and not args.force:
            print(f'  [SKIP] {out}')
        elif not os.path.exists(args.detector):
            print(f'  [MISSING] {args.detector}', file=sys.stderr)
        else:
            export_detector(args.detector, out, imgsz=sz)

    if os.path.exists(app_out) and not args.force:
        print(f'  [SKIP] {app_out}')
    else:
        check_parity('appearance', *export_appearance(app_out), app_out)

    if not os.path.exists(args.head_ckpt):
        print(f'  [MISSING] {args.head_ckpt}\n'
              f'    produce it with: run_relation_transformer.py --tag yolo26app_ --seeds 5 '
              f'--aux_w 0.5 --epochs 120 --save_model {args.head_ckpt}', file=sys.stderr)
        return 1
    if os.path.exists(head_out) and not args.force:
        print(f'  [SKIP] {head_out}')
    else:
        check_parity('relation-head', *export_relation_head(args.head_ckpt, head_out), head_out)
    print('\nDone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
