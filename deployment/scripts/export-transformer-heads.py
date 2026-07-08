"""
export-transformer-heads.py — Bulk export all 7 TransformerHead models to ONNX (FP32).

Input shape:  (1, 32, 1536) — T=32 fixed, stats_dim=512*3
Output shape: (1, nc)       — nc varies per head

Usage:
    python deployment/scripts/export-transformer-heads.py [--force]
"""
import argparse
import sys
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import importlib.util

SCRIPTS_DIR = os.path.join(ROOT, "deployment", "scripts")

# Load TransformerHead from kebab-case filename via importlib
_spec = importlib.util.spec_from_file_location(
    "transformer_head",
    os.path.join(SCRIPTS_DIR, "transformer-head.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TransformerHead = _mod.TransformerHead

EXPORT_DIR = os.path.join(ROOT, "deployment", "export")
PAIRWISE_DIR = os.path.join(ROOT, "sweep_results", "pairwise")

# Head registry: name -> (pt_filename, num_classes, dim, nhead, layers, drop)
HEADS = {
    "direct4-s314": ("direct4_s314.pt",           4, 384, 8, 6, 0.3),
    "scenario-s7":  ("scenario_s7.pt",             4, 384, 8, 6, 0.3),
    "pair0-s7":     ("pair0_Walkway_s7.pt",         2, 192, 6, 3, 0.3),
    "pair1-s7":     ("pair1_Intervention_s7.pt",    2, 192, 6, 3, 0.3),
    "pair2-s7":     ("pair2_Panel_s7.pt",           2, 192, 6, 3, 0.3),
    "pair3-s7":     ("pair3_Forklift_s7.pt",        2, 192, 6, 3, 0.3),
    "eight-s7":     ("eight_s7.pt",                 8, 384, 8, 6, 0.3),
}

STATS_DIM = 1536   # 512 * 3 (mean + std + max over 16 spatial tokens)
T_FIXED = 32


def export_head(name: str, pt_path: str, nc: int, dim: int, nhead: int,
                layers: int, drop: float, out_path: str, opset: int = 17) -> None:
    import onnx

    if not os.path.exists(pt_path):
        print(f"  [MISSING] {pt_path}", file=sys.stderr)
        return

    print(f"  Loading {name} from {os.path.basename(pt_path)} ...")
    model = TransformerHead(STATS_DIM, nc, dim, nhead, layers, drop)
    state = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(1, T_FIXED, STATS_DIM)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, nc), f"Unexpected output shape: {out.shape}"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.export(
        model, dummy, out_path,
        opset_version=opset,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
    )

    # Re-save inline — required for CoreML EP (ORT issue #18518)
    m = onnx.load(out_path)
    onnx.save(m, out_path, save_as_external_data=False)
    onnx.checker.check_model(m)
    print(f"    Exported: {out_path}  shape={out.shape}")


def main():
    parser = argparse.ArgumentParser(description="Export TransformerHead models to ONNX")
    parser.add_argument("--force", action="store_true", help="Re-export even if output exists")
    args = parser.parse_args()

    for name, (pt_file, nc, dim, nhead, layers, drop) in HEADS.items():
        pt_path = os.path.join(PAIRWISE_DIR, pt_file)
        out_path = os.path.join(EXPORT_DIR, f"{name}-fp32.onnx")

        if os.path.exists(out_path) and not args.force:
            print(f"[SKIP] Already exists: {out_path}")
            continue

        export_head(name, pt_path, nc, dim, nhead, layers, drop, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
