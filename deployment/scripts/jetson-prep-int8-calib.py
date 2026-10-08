#!/usr/bin/env python3
"""F11 — Prepare INT8 calibration batches for Jetson TRT build.

Streams 32 calibration samples per head from the cached training-set features
into per-model .npy batch files. The Jetson INT8 calibrator
(jetson-build-trt-engine.py) reads these without needing PyTorch on-device.

Run on M3 (or any machine with the .pt cache); copy the output dir to the
Jetson before running jetson-build-trt-engine.py --mode int8.

Output: deployment/artifacts/calib-jetson/<model>-batch-{0..31}.npy
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEAT_TRAIN = os.path.join(ROOT, 'sweep_results', 'pairwise', 'features', 'all8_train.pt')
OUT_DIR = os.path.join(ROOT, 'deployment', 'artifacts', 'calib-jetson')
# Real-frame search paths for backbone calibration (in priority order).
TRAIN_IMG_CANDIDATES = [
    os.path.join(ROOT, 'data', 'train'),
    os.path.join(ROOT, 'annotation_guide', 'train'),
    os.path.join(ROOT, 'annotation_guide'),
]
N = 256  # was 32 — too few samples caused 2/64 acc drop on INT8
T = 32
RAW_HW = (224, 224)

# Backbone takes (1, 3, 224, 224) image-shaped input; head/cascade take
# stats tensor (1, T, 1536). Calibration batches must match each model's
# input layout exactly.
HEAD_MODELS = ['direct4-s314', 'scenario-s7',
               'pair0-s7', 'pair1-s7', 'pair2-s7', 'pair3-s7']


def stats_pool(features):  # features: (N, T, 16, 512) → (N, T, 1536)
    return torch.cat([features.mean(2), features.std(2), features.max(2).values], -1)


def load_real_backbone_frames(n, rng):
    """Load `n` real training frames for backbone INT8 calibration.

    Priority:
      1. Direct .jpg/.png files in candidate dirs
      2. Extract frames from .mp4 videos in data/train/ (preferred — matches
         deployment input distribution exactly).
    Returns np.ndarray (n, 3, 224, 224) ImageNet-normalised, or None if no
    source available."""
    import glob
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    # Source 1: pre-extracted image files
    for root_dir in TRAIN_IMG_CANDIDATES:
        if not os.path.isdir(root_dir):
            continue
        paths = sorted(glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True) +
                       glob.glob(os.path.join(root_dir, '**', '*.png'), recursive=True))
        if len(paths) >= n:
            from PIL import Image
            print(f'  source: image files from {root_dir} (n={len(paths)}, sampling {n})')
            sel_paths = [paths[i] for i in rng.choice(len(paths), n, replace=False)]
            imgs = np.zeros((n, 3, RAW_HW[0], RAW_HW[1]), dtype=np.float32)
            for i, p in enumerate(sel_paths):
                im = Image.open(p).convert('RGB').resize(RAW_HW, Image.BILINEAR)
                arr = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
                imgs[i] = (arr - mean) / std
            return imgs

    # Source 2: extract frames from training videos
    train_dir = os.path.join(ROOT, 'data', 'train')
    if os.path.isdir(train_dir):
        try:
            import cv2
        except ImportError:
            print('  WARN: opencv-python not installed; cannot extract video frames. '
                  'Run "pip install opencv-python" then retry.')
            return None
        videos = sorted(glob.glob(os.path.join(train_dir, '*', '*.mp4')))
        if not videos:
            return None
        print(f'  source: extracting frames from {len(videos)} train videos (sampling {n})')
        imgs = np.zeros((n, 3, RAW_HW[0], RAW_HW[1]), dtype=np.float32)
        # Sample with replacement if N > #videos; pick random frame per video.
        vid_idx = rng.choice(len(videos), n, replace=(n > len(videos)))
        ok = 0
        for i, vi in enumerate(vid_idx):
            cap = cv2.VideoCapture(videos[vi])
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n_frames < 1:
                cap.release(); continue
            fpos = int(rng.randint(0, max(1, n_frames - 1)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fpos)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue
            # BGR (cv2) → RGB, resize, normalise
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, RAW_HW, interpolation=cv2.INTER_LINEAR)
            arr = frame.astype(np.float32).transpose(2, 0, 1) / 255.0
            imgs[i] = (arr - mean) / std
            ok += 1
        print(f'  extracted {ok}/{n} frames successfully')
        if ok < n // 2:
            return None  # too many failed reads → unreliable
        return imgs
    return None


def main():
    if not os.path.exists(FEAT_TRAIN):
        sys.exit(f'ERROR: training features not found at {FEAT_TRAIN}; '
                 f'run tools/preextract_yolo_features.py --subset train first.')
    os.makedirs(OUT_DIR, exist_ok=True)
    # Clear stale cache from previous N=32 run.
    for stale in os.listdir(OUT_DIR):
        if stale.endswith('.calib-cache') or stale.endswith('.npy'):
            os.remove(os.path.join(OUT_DIR, stale))

    print(f'Loading {FEAT_TRAIN}')
    d = torch.load(FEAT_TRAIN, map_location='cpu', weights_only=False)
    feats = d['features']
    print(f'features shape={tuple(feats.shape)}  (sampling N={N} for calibration)')

    rng = np.random.RandomState(7)
    n_feat = min(N, len(feats))
    idx = rng.choice(len(feats), n_feat, replace=False)
    sel = feats[idx]
    stats = stats_pool(sel).numpy().astype('float32')
    print(f'calibration stats: shape={stats.shape}')

    for m in HEAD_MODELS:
        for i in range(n_feat):
            np.save(os.path.join(OUT_DIR, f'{m}-batch-{i}.npy'), stats[i:i + 1])
    print(f'[SAVED] {n_feat} stats batches × {len(HEAD_MODELS)} head models in {OUT_DIR}')

    # Backbone calibration: prefer real training frames over synthetic noise to
    # capture the true activation distribution at conv layers — much better INT8
    # tactic selection. Falls back to synthetic only if no frames available.
    print(f'\nBackbone calibration:')
    img_calib = load_real_backbone_frames(n_feat, rng)
    if img_calib is None:
        print(f'  WARN: no real frames found in {TRAIN_IMG_CANDIDATES}; '
              f'falling back to synthetic noise')
        img_calib = rng.randn(n_feat, 3, RAW_HW[0], RAW_HW[1]).astype('float32')
        source = 'synthetic noise'
    else:
        source = 'real ImageNet-normalised frames'
    for i in range(n_feat):
        np.save(os.path.join(OUT_DIR, f'backbone-batch-{i}.npy'), img_calib[i:i + 1])
    print(f'[SAVED] {n_feat} backbone calibration tensors ({source})')

    print(f'\nNext step: rsync {os.path.relpath(OUT_DIR, ROOT)} to the Jetson, '
          f'then run jetson-build-trt-engine.py --mode int8.')


if __name__ == '__main__':
    main()
