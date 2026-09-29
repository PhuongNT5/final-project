"""benchmark-p3-sweep.py — detector cost-curve x frames-per-decision sweep, split out of
benchmark-p3-onnx.py to keep each file under the ~200-line guideline. COST ONLY (see
run_sweep docstring) — imported by benchmark-p3-onnx.py's --sweep flag.
"""
import json
import os

import numpy as np

from importlib.util import spec_from_file_location, module_from_spec

_HERE = os.path.dirname(os.path.abspath(__file__))
_u_spec = spec_from_file_location('benchmark_p3_utils', os.path.join(_HERE, 'benchmark-p3-utils.py'))
_u = module_from_spec(_u_spec)
_u_spec.loader.exec_module(_u)
timed = _u.timed
ROOT = _u.ROOT

ART_DIR = os.path.join(ROOT, 'deployment', 'artifacts')


def p1_transfer():
    """Mac-CPU -> Jetson speedup factors MEASURED on P1, which exists on both machines.

    P1's end-to-end p50 is 26.80 ms on this Mac under ORT-CPU and 12.47/8.62/8.24 ms on the Orin
    Nano under TensorRT fp32/fp16/int8.  Those ratios are the only empirical Mac->Jetson bridge this
    project owns.  They are a PROXY: P1 is a 224-px backbone plus tiny heads, P3 is dominated by an
    800-px detector, and a GPU generally utilises the larger convolution better — so the factors
    below are, if anything, conservative for P3.  Never present a transferred number as measured.
    """
    out = {}
    try:
        b = json.load(open(os.path.join(ART_DIR, 'benchmarks.json')))
        cpu = [c for c in b['configs'] if c['provider'] == 'cpu' and c['precision'] == 'fp32'][0]
        mac = cpu['stages']['end_to_end']['p50']
        for m in ('fp32', 'fp16', 'int8'):
            f = os.path.join(ART_DIR, f'benchmarks-jetson-{m}.json')
            if os.path.exists(f):
                out[m] = mac / json.load(open(f))['config']['end_to_end_ms']['50']
    except Exception as e:
        print(f'  [transfer] unavailable ({e})')
    return out


def run_sweep(sess, ep, warmup, runs, crops, frames_opts, export_dir, graphs, ep_providers,
             t_frames, crop_dim, imgsz):
    """Detector cost curve x frames-per-decision — what operating point could reach 3.0 dec/s.

    COST ONLY.  Only imgsz=800 with 16 frames has an accuracy number attached (54/64); every other
    cell would need tokens rebuilt and the head retrained before it could be claimed.

    Re-uses the caller's appearance/head sessions and holds exactly ONE detector session alive at a
    time.  An earlier version built its own duplicates: every extra ORT session spawns a thread pool
    sized to the core count, and the resulting oversubscription inflated appearance from 82 to 182 ms
    and detector@800 from 108 to 133 ms in the same process.  Duplicate sessions measure the
    scheduler, not the graph.
    """
    import gc
    import onnxruntime as ort
    app, head = sess['appearance'], sess['head']
    an, hn = app.get_inputs()[0].name, [i.name for i in head.get_inputs()]
    # Inputs are built ONCE, outside every timed lambda — as run_bench does. Generating them inside
    # the lambda times np.random.randn as well as the graph: it cost 58*3*224*224 gaussians per
    # appearance call (82 ms -> 180 ms) and scaled with sz^2 on the detector, which would have made
    # the whole cost curve steeper than the model actually is.
    b_app = np.random.randn(crops, 3, crop_dim, crop_dim).astype(np.float32)
    b_tok = np.random.randn(1, t_frames, 8, 74).astype(np.float32)
    b_msk = np.ones((1, t_frames, 8), dtype=np.float32)
    t_app = timed(lambda: app.run(None, {an: b_app}), warmup, runs)['50']
    t_head = timed(lambda: head.run(None, {hn[0]: b_tok, hn[1]: b_msk}), warmup, runs)['50']

    cands = [(imgsz, graphs['detector'])]
    for f in sorted(os.listdir(export_dir)):
        if f.startswith('p3-detector-fp32-imgsz') and f.endswith('.onnx'):
            cands.append((int(f.split('imgsz')[1].split('.')[0]), f))
    cands.sort(key=lambda c: -c[0])

    tf = p1_transfer()
    print(f'\nOperating-point sweep ({ep}, {crops} crops, appearance {t_app:.2f} ms, head {t_head:.2f} ms)')
    print('  COST ONLY — accuracy exists ONLY at imgsz=800 x 16 frames (54/64). Other rows are '
          'unmeasured for accuracy.')
    hdr = f"  {'imgsz':>6} {'T':>3} {'det/frame':>10} {'ms/decision':>12} {'Mac dec/s':>10}"
    hdr += ''.join(f'{"Jetson~"+m:>13}' for m in tf)
    print(hdr)
    rows = []
    for sz, fn in cands:
        # the canonical 800 graph is already open in sess — opening a second copy would double its
        # thread pool and mis-time the one row that has an accuracy number attached.
        det = sess['detector'] if fn == graphs['detector'] else \
            ort.InferenceSession(os.path.join(export_dir, fn), providers=ep_providers[ep])
        dn = det.get_inputs()[0].name
        b_det = np.random.randn(1, 3, sz, sz).astype(np.float32)
        t_det = timed(lambda: det.run(None, {dn: b_det}), warmup, runs)['50']
        if fn != graphs['detector']:
            del det                  # release its thread pool before the next session is built
            gc.collect()
        for T in frames_opts:
            ms = T * t_det + t_app + t_head
            line = f'  {sz:>6} {T:>3} {t_det:>10.2f} {ms:>12.1f} {1000/ms:>10.2f}'
            proj = {}
            for m, f in tf.items():
                d = 1000 / (ms / f)
                proj[m] = round(d, 2)
                line += f'{d:>9.2f} {"OK " if d >= 3.0 else "no "}'
            print(line)
            rows.append({'imgsz': sz, 'frames_per_decision': T, 'det_ms_per_frame': round(t_det, 2),
                         'ms_per_decision': round(ms, 1), 'mac_decisions_per_sec': round(1000/ms, 2),
                         'jetson_projected_decisions_per_sec': proj,
                         'accuracy_measured': (sz == imgsz and T == t_frames)})
    print('  OK/no = meets the 3.0 decisions/s the 24 fps + EMIT_EVERY=8 cadence needs.')
    return {'transfer_factors': {k: round(v, 3) for k, v in tf.items()},
            'appearance_ms': t_app, 'head_ms': t_head, 'crops': crops, 'rows': rows}
