"""
quantize-int8.py — Apply ONNX Runtime dynamic INT8 quantization to all FP32 models.

Usage:
    python deployment/scripts/quantize-int8.py [--force]
"""
import argparse
import glob
import os
import sys
import tempfile

from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPORT_DIR = os.path.join(ROOT, "deployment", "export")


def _strip_value_info(in_path: str, out_path: str) -> None:
    # Dynamo-exported transformer graphs carry stale value_info entries that conflict
    # with onnx.shape_inference (dim mismatch errors). Clearing them lets the quantizer
    # re-infer from scratch.
    import onnx

    m = onnx.load(in_path)
    del m.graph.value_info[:]
    onnx.save(m, out_path)


def quantize_one(fp32_path: str, int8_path: str) -> None:
    print(f"  {os.path.basename(fp32_path)} -> {os.path.basename(int8_path)}")
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        stripped = tmp.name
    try:
        _strip_value_info(fp32_path, stripped)
        quantize_dynamic(
            model_input=stripped,
            model_output=int8_path,
            weight_type=QuantType.QInt8,
        )
    finally:
        if os.path.exists(stripped):
            os.remove(stripped)
    fp32_mb = os.path.getsize(fp32_path) / 1e6
    int8_mb = os.path.getsize(int8_path) / 1e6
    print(f"    {fp32_mb:.2f} MB -> {int8_mb:.2f} MB ({int8_mb / fp32_mb * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Dynamic INT8 quantize all FP32 ONNX models")
    parser.add_argument("--force", action="store_true", help="Re-quantize even if output exists")
    args = parser.parse_args()

    fp32_files = sorted(glob.glob(os.path.join(EXPORT_DIR, "*-fp32.onnx")))
    if not fp32_files:
        print(f"[ERROR] No FP32 models in {EXPORT_DIR}", file=sys.stderr)
        sys.exit(1)

    for fp32_path in fp32_files:
        int8_path = fp32_path.replace("-fp32.onnx", "-int8.onnx")
        if os.path.exists(int8_path) and not args.force:
            print(f"[SKIP] {int8_path}")
            continue
        quantize_one(fp32_path, int8_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
