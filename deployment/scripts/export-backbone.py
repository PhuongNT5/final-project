"""
export-backbone.py — Export YOLOv11s backbone to ONNX (FP32).

Uses YOLOv11BackboneTokenizer wrapper (researcher report §3).
Input:  (1, 3, 224, 224) normalized RGB frame
Output: (1, 16, 512) spatial tokens

Usage:
    python deployment/scripts/export-backbone.py [--force]
"""
import argparse
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

EXPORT_DIR = os.path.join(ROOT, "deployment", "export")


class YOLOv11BackboneTokenizer(nn.Module):
    """Layer-22 output → (B, 16, 512) spatial tokens. From researcher report §3."""

    def __init__(self, detection_model, pool: int = 4):
        super().__init__()
        self.layers = detection_model.model   # nn.ModuleList
        self.save = detection_model.save      # [4,6,10,13,16,19,22]
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: list = []
        for i, layer in enumerate(self.layers[:23]):
            f = layer.f if hasattr(layer, "f") else -1
            if f != -1:
                x = (
                    [y[j] if j >= 0 else x for j in f]
                    if isinstance(f, list)
                    else (y[f] if f >= 0 else x)
                )
            x = layer(x)
            y.append(x if i in self.save else None)
        # Training pipeline uses bilinear interpolate to (pool,pool) then pool (identity).
        # Must match exactly or downstream heads drop accuracy (run_pairwise_contrastive.py L162-168).
        x = F.interpolate(x, size=(self.pool, self.pool), mode="bilinear", align_corners=False)
        return x.flatten(2).transpose(1, 2)                    # (B, 16, 512)


def export_backbone(pt_path: str, out_path: str, opset: int = 17) -> None:
    import onnx
    from ultralytics import YOLO

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Loading: {pt_path}")
    dm = YOLO(pt_path).model
    dm.eval()
    wrapper = YOLOv11BackboneTokenizer(dm)
    wrapper.eval()

    dummy = torch.zeros(1, 3, 224, 224)
    # Smoke-check shape
    with torch.no_grad():
        out = wrapper(dummy)
    assert out.shape == (1, 16, 512), f"Unexpected output shape: {out.shape}"
    print(f"  Forward check passed: {out.shape}")

    torch.onnx.export(
        wrapper, dummy, out_path,
        opset_version=opset,
        input_names=["image"],
        output_names=["tokens"],
        dynamic_axes={"image": {0: "batch"}, "tokens": {0: "batch"}},
    )

    # Re-save inline (no external-data split) — required for CoreML EP (ORT issue #18518)
    model = onnx.load(out_path)
    onnx.save(model, out_path, save_as_external_data=False)
    onnx.checker.check_model(model)
    print(f"  Exported + verified: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv11s backbone to ONNX")
    parser.add_argument("--force", action="store_true", help="Re-export even if output exists")
    args = parser.parse_args()

    exports = [
        (
            os.path.join(ROOT, "yolo11s.pt"),
            os.path.join(EXPORT_DIR, "backbone-fp32.onnx"),
        ),
        (
            os.path.join(ROOT, "runs/detect/sweep_results/finetune_bbox/yolo_finetuned/weights/best.pt"),
            os.path.join(EXPORT_DIR, "backbone-ft-fp32.onnx"),
        ),
    ]

    for pt_path, out_path in exports:
        if not os.path.exists(pt_path):
            print(f"[SKIP] Model not found: {pt_path}", file=sys.stderr)
            continue
        if os.path.exists(out_path) and not args.force:
            print(f"[SKIP] Already exists (use --force to re-export): {out_path}")
            continue
        export_backbone(pt_path, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
