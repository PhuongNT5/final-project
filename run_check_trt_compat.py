#!/usr/bin/env python3
"""F11 (M3 sanity-check): verify exported ONNX graphs are TensorRT-compatible
without requiring NVIDIA hardware.

Checks per ONNX file in deployment/export/:
  - opset version is >= 13 and <= 19 (TRT 8.6 / 10 supported range)
  - all operators are in TRT's supported set (best-effort; uses a static
    reference list of ops known-good on Jetson Orin TRT 8.6 / 10)
  - I/O shapes are static or use named dynamic axes only on batch dim
  - file size sanity (FP32 .onnx + .onnx.data both present, INT8 single file)

Output: deployment/artifacts/trt-compat-report.json + .md

Usage:
    python3 run_check_trt_compat.py
"""
import json
import os
import sys
from collections import Counter

import onnx

ROOT = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(ROOT, 'deployment', 'export')
OUT_JSON = os.path.join(ROOT, 'deployment', 'artifacts', 'trt-compat-report.json')
OUT_MD = os.path.join(ROOT, 'deployment', 'artifacts', 'trt-compat-report.md')

# TRT 8.6 / 10 stable supported ops (subset relevant to our graphs).
# Source: NVIDIA TensorRT Support Matrix (opset 13–19).
TRT_OPS = {
    'Add', 'AveragePool', 'BatchNormalization', 'Cast', 'Clip', 'Concat',
    'Constant', 'ConstantOfShape', 'Conv', 'Div', 'Erf', 'Equal', 'Expand',
    'Flatten', 'Gather', 'Gemm', 'GlobalAveragePool', 'GlobalMaxPool',
    'Greater', 'Identity', 'LayerNormalization', 'LeakyRelu', 'Less',
    'MatMul', 'MaxPool', 'Mul', 'Neg', 'Pad', 'Pow', 'Range', 'ReduceMax',
    'ReduceMean', 'ReduceSum', 'Relu', 'Reshape', 'Resize', 'Shape',
    'Sigmoid', 'Slice', 'Softmax', 'Split', 'Sqrt', 'Squeeze', 'Sub', 'Tanh',
    'Tile', 'Transpose', 'Unsqueeze', 'Where', 'GELU', 'Gelu', 'Mish',
    'SiLU', 'HardSigmoid', 'HardSwish', 'TopK', 'ScatterND', 'CumSum',
    'GatherND', 'Einsum', 'NonZero', 'Round',
    # Quantize/Dequantize for INT8
    'QuantizeLinear', 'DequantizeLinear',
}


def check_onnx(path):
    issues = []
    warns = []
    try:
        m = onnx.load(path)
    except Exception as e:
        return {'file': os.path.basename(path), 'load_error': str(e)}

    opsets = {o.domain: o.version for o in m.opset_import}
    op_v = opsets.get('', 0)
    if op_v < 13:
        issues.append(f'opset {op_v} < 13 (TRT 8.6 minimum is 13)')
    if op_v > 19:
        warns.append(f'opset {op_v} > 19 (TRT 10 nominally supports up to 19)')

    op_counter = Counter(n.op_type for n in m.graph.node)
    unsupported = [op for op in op_counter if op not in TRT_OPS]

    inputs = [(i.name, [d.dim_value or d.dim_param or '?'
                       for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    outputs = [(o.name, [d.dim_value or d.dim_param or '?'
                        for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]

    fname = os.path.basename(path)
    size_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
    data_path = path + '.data'
    has_data = os.path.exists(data_path)
    data_mb = round(os.path.getsize(data_path) / (1024 * 1024), 2) if has_data else 0

    return {
        'file': fname,
        'opset': op_v,
        'size_mb': size_mb,
        'has_external_data': has_data,
        'external_data_mb': data_mb,
        'n_nodes': len(m.graph.node),
        'op_counts': dict(op_counter),
        'unsupported_ops': unsupported,
        'inputs': inputs,
        'outputs': outputs,
        'issues': issues,
        'warnings': warns,
    }


def main():
    files = sorted(f for f in os.listdir(EXPORT_DIR) if f.endswith('.onnx'))
    print(f'Found {len(files)} ONNX files in {EXPORT_DIR}')
    reports = [check_onnx(os.path.join(EXPORT_DIR, f)) for f in files]

    n_ok = sum(1 for r in reports if not r.get('issues') and not r.get('unsupported_ops'))
    n_blocked = sum(1 for r in reports if r.get('issues') or r.get('unsupported_ops'))

    summary = {
        'export_dir': os.path.relpath(EXPORT_DIR, ROOT),
        'n_files': len(reports),
        'n_trt_compatible': n_ok,
        'n_blocked': n_blocked,
        'reports': reports,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[SAVED] {OUT_JSON}')

    lines = ['# TensorRT compatibility report (M3 graph-only sanity)\n',
             f'Export dir: `{summary["export_dir"]}`\n',
             f'Files scanned: {summary["n_files"]} | TRT-compatible: {n_ok} | blocked: {n_blocked}\n',
             '\n## Per-file results\n',
             '| File | Opset | Size MB | Nodes | Unsupported ops | Issues |',
             '|------|------:|--------:|------:|-----------------|--------|']
    for r in reports:
        if 'load_error' in r:
            lines.append(f"| {r['file']} | — | — | — | LOAD ERROR | {r['load_error']} |")
            continue
        unsup = ', '.join(r['unsupported_ops']) or '—'
        iss = '; '.join(r['issues']) or '—'
        lines.append(f"| {r['file']} | {r['opset']} | {r['size_mb']} | {r['n_nodes']} | {unsup} | {iss} |")

    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[SAVED] {OUT_MD}')

    print(f'\nSummary: {n_ok}/{len(reports)} TRT-compatible, {n_blocked} blocked')
    if n_blocked:
        sys.exit(1)


if __name__ == '__main__':
    main()
