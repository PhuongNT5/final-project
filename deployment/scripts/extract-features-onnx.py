"""
extract-features-onnx.py — Extract video features via ONNX backbone.

Walk data/test/, sample 32 frames per video, run FP32 backbone ONNX (CPU EP).
Output: deployment/artifacts/features/all8_test_onnx.pt
  features: (N, 32, 16, 512), labels: (N,)

Usage:
    .venv/bin/python3 deployment/scripts/extract-features-onnx.py [--force]
"""
import argparse
import gc
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data", "test")
BACKBONE_ONNX = os.path.join(ROOT, "deployment", "export", "backbone-fp32.onnx")
OUT_PATH = os.path.join(ROOT, "deployment", "artifacts", "features", "all8_test_onnx.pt")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
T = 32
IMG_SIZE = 224


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR frame → (1, 3, 224, 224) float32. Matches training: PIL BILINEAR + ImageNet norm."""
    # Training uses torchvision transforms.Resize((224,224)) on PIL — PIL default is BILINEAR.
    # cv2.resize produces subtly different pixels and drops ~9/64 top-1 on this dataset.
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    resized_pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(resized_pil, dtype=np.float32) / 255.0
    normalized = (arr - MEAN) / STD
    return normalized.transpose(2, 0, 1)[np.newaxis]  # (1,3,224,224)


def extract_video(vpath: str, sess) -> np.ndarray:
    """Extract (32, 16, 512) tokens from one video via ONNX backbone."""
    cap = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.zeros((T, 16, 512), dtype=np.float32)

    indices = np.linspace(0, total - 1, T, dtype=int)
    tokens_list = []
    input_name = sess.get_inputs()[0].name

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            tokens_list.append(np.zeros((16, 512), dtype=np.float32))
            continue
        inp = preprocess_frame(frame)
        out = sess.run(None, {input_name: inp})[0]  # (1,16,512)
        tokens_list.append(out[0])  # (16,512)

    cap.release()

    # Pad short videos by repeating last frame
    while len(tokens_list) < T:
        tokens_list.append(tokens_list[-1] if tokens_list else np.zeros((16, 512), dtype=np.float32))

    return np.stack(tokens_list[:T])  # (32,16,512)


def collect_samples():
    """Walk data/test/ → list of (video_path, label)."""
    samples = []
    for folder in sorted(os.listdir(DATA_DIR)):
        full = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(full):
            continue
        label = int(folder.split("_")[0])
        for vf in sorted(os.listdir(full)):
            if vf.lower().endswith((".mp4", ".avi", ".mov")):
                samples.append((os.path.join(full, vf), label))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-extract even if cached")
    args = parser.parse_args()

    if os.path.exists(OUT_PATH) and not args.force:
        data = torch.load(OUT_PATH, map_location="cpu")
        print(f"[CACHED] {OUT_PATH}")
        print(f"  features: {data['features'].shape}, labels: {data['labels'].shape}")
        return

    assert os.path.exists(BACKBONE_ONNX), f"Backbone ONNX not found: {BACKBONE_ONNX}"

    import onnxruntime as ort
    sess = ort.InferenceSession(BACKBONE_ONNX, providers=["CPUExecutionProvider"])
    print(f"Backbone loaded: {BACKBONE_ONNX}")
    print(f"  Providers: {sess.get_providers()}")

    samples = collect_samples()
    print(f"\nFound {len(samples)} videos in {DATA_DIR}")

    all_feats, all_labels = [], []
    for vi, (vpath, label) in enumerate(samples):
        tokens = extract_video(vpath, sess)  # (32,16,512)
        all_feats.append(tokens)
        all_labels.append(label)
        if (vi + 1) % 20 == 0 or vi + 1 == len(samples):
            print(f"  {vi+1}/{len(samples)}", flush=True)

    features = torch.tensor(np.stack(all_feats))   # (N,32,16,512)
    labels = torch.tensor(all_labels, dtype=torch.long)

    class_counts = [int((labels == c).sum()) for c in range(8)]
    print(f"\nfeatures: {features.shape}, labels: {labels.shape}")
    print(f"class counts (0-7): {class_counts}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save({"features": features, "labels": labels}, OUT_PATH)
    print(f"\nSaved → {OUT_PATH}")

    del sess
    gc.collect()


if __name__ == "__main__":
    main()
