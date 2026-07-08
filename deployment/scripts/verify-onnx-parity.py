"""
verify-onnx-parity.py — Confirm ONNX FP32 matches PyTorch; measure INT8 drift.

For each exported head, runs 16 random inputs through PyTorch, ORT-FP32, ORT-INT8
and reports max absolute logit differences. Success criterion: FP32 diff < 1e-3
(rounding only); INT8 diff noted for thesis discussion.

Backbone parity is also checked on a single 224x224 random image.

Accuracy-on-test-set validation is deferred to Phase 02 (benchmark-harness.py)
where the full inference pipeline runs with extracted video features.

Usage:
    python deployment/scripts/verify-onnx-parity.py
"""
import importlib.util
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

SCRIPTS_DIR = os.path.join(ROOT, "deployment", "scripts")
EXPORT_DIR = os.path.join(ROOT, "deployment", "export")
PAIRWISE_DIR = os.path.join(ROOT, "sweep_results", "pairwise")
REPORT_PATH = os.path.join(ROOT, "deployment", "artifacts", "parity-report.txt")

# Import TransformerHead + wrapper via kebab-case filename
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

TH = _load("th", os.path.join(SCRIPTS_DIR, "transformer-head.py")).TransformerHead
BW = _load("bb", os.path.join(SCRIPTS_DIR, "export-backbone.py")).YOLOv11BackboneTokenizer

HEADS = {
    "direct4-s314": ("direct4_s314.pt", 4, 384, 8, 6, 0.3),
    "scenario-s7":  ("scenario_s7.pt", 4, 384, 8, 6, 0.3),
    "pair0-s7":     ("pair0_Walkway_s7.pt", 2, 192, 6, 3, 0.3),
    "pair1-s7":     ("pair1_Intervention_s7.pt", 2, 192, 6, 3, 0.3),
    "pair2-s7":     ("pair2_Panel_s7.pt", 2, 192, 6, 3, 0.3),
    "pair3-s7":     ("pair3_Forklift_s7.pt", 2, 192, 6, 3, 0.3),
    "eight-s7":     ("eight_s7.pt", 8, 384, 8, 6, 0.3),
}
STATS_DIM = 1536
T_FIXED = 32
N_SAMPLES = 16


def max_diff(a, b):
    return float(np.max(np.abs(a - b)))


def run_ort(path, input_name, x_np):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, {input_name: x_np})[0]


def verify_head(name, pt_file, nc, dim, nhead, layers, drop):
    pt_path = os.path.join(PAIRWISE_DIR, pt_file)
    fp32 = os.path.join(EXPORT_DIR, f"{name}-fp32.onnx")
    int8 = os.path.join(EXPORT_DIR, f"{name}-int8.onnx")

    model = TH(STATS_DIM, nc, dim, nhead, layers, drop)
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    # Heads exported with batch=1 fixed in dynamo trace — loop per-sample to avoid
    # dynamic-axes reshape issues (production inference is batch=1 anyway).
    sess_fp32 = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])

    torch.manual_seed(0)
    x = torch.randn(N_SAMPLES, T_FIXED, STATS_DIM)
    with torch.no_grad():
        y_pt = model(x).numpy()
    y_fp32 = np.stack([sess_fp32.run(None, {"features": x[i:i+1].numpy()})[0][0] for i in range(N_SAMPLES)])
    y_int8 = np.stack([sess_int8.run(None, {"features": x[i:i+1].numpy()})[0][0] for i in range(N_SAMPLES)])

    return {
        "head": name,
        "shape": tuple(y_pt.shape),
        "pt_vs_fp32": max_diff(y_pt, y_fp32),
        "pt_vs_int8": max_diff(y_pt, y_int8),
        "fp32_vs_int8": max_diff(y_fp32, y_int8),
    }


def verify_backbone():
    yolo = YOLO(os.path.join(ROOT, "yolo11s.pt"))
    wrapper = BW(yolo.model)
    wrapper.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y_pt = wrapper(x).numpy()
    y_fp32 = run_ort(os.path.join(EXPORT_DIR, "backbone-fp32.onnx"), "image", x.numpy())
    y_int8 = run_ort(os.path.join(EXPORT_DIR, "backbone-int8.onnx"), "image", x.numpy())

    return {
        "head": "backbone (yolo11s)",
        "shape": tuple(y_pt.shape),
        "pt_vs_fp32": max_diff(y_pt, y_fp32),
        "pt_vs_int8": max_diff(y_pt, y_int8),
        "fp32_vs_int8": max_diff(y_fp32, y_int8),
    }


def main():
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    rows = [verify_backbone()]
    for name, spec in HEADS.items():
        print(f"  {name} ...", flush=True)
        rows.append(verify_head(name, *spec))

    hdr = f"{'model':<22} {'shape':<18} {'pt~fp32':>10} {'pt~int8':>10} {'fp32~int8':>11}"
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r['head']:<22} {str(r['shape']):<18} "
            f"{r['pt_vs_fp32']:>10.2e} {r['pt_vs_int8']:>10.2e} {r['fp32_vs_int8']:>11.2e}"
        )

    fp32_ok = all(r["pt_vs_fp32"] < 1e-3 for r in rows)
    lines.append("")
    lines.append(f"FP32 parity (<1e-3): {'PASS' if fp32_ok else 'FAIL'}")

    out = "\n".join(lines)
    print("\n" + out)
    with open(REPORT_PATH, "w") as f:
        f.write(out + "\n")
    print(f"\nReport: {REPORT_PATH}")

    if not fp32_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
