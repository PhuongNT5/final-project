"""
Pre-extract video frames to .npy cache files for fast training data loading.
Saves 32 frames at 256x256 per video alongside the video file as <video>.npy.
Run once before training:
    python tools/preextract_frames.py
"""
import cv2
import numpy as np
import os
import time

T_CACHE = 32     # number of frames to cache per video
CACHE_SIZE = 224  # spatial resolution to save (~3.1GB total, fits in available disk)


def preextract_video(path):
    cache_path = path + '.npy'
    if os.path.exists(cache_path):
        return False  # already cached

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        blank = np.zeros((T_CACHE, CACHE_SIZE, CACHE_SIZE, 3), dtype=np.uint8)
        np.save(cache_path, blank)
        return True

    indices = np.linspace(0, total - 1, T_CACHE, dtype=int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               (CACHE_SIZE, CACHE_SIZE),
                               interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        elif frames:
            frames.append(frames[-1])
        else:
            frames.append(np.zeros((CACHE_SIZE, CACHE_SIZE, 3), dtype=np.uint8))
    cap.release()

    while len(frames) < T_CACHE:
        frames.append(frames[-1])

    raw = np.array(frames[:T_CACHE], dtype=np.uint8)
    np.save(cache_path, raw)
    return True


def main():
    data_root = 'data'
    video_exts = ('.mp4', '.avi', '.mov')

    # Collect all video paths
    paths = []
    for subset in ['train', 'test']:
        subset_dir = os.path.join(data_root, subset)
        if not os.path.isdir(subset_dir):
            continue
        for folder in sorted(os.listdir(subset_dir)):
            folder_path = os.path.join(subset_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for vf in sorted(os.listdir(folder_path)):
                if vf.lower().endswith(video_exts):
                    paths.append(os.path.join(folder_path, vf))

    total = len(paths)
    print(f"Found {total} videos. Extracting frames ({T_CACHE} frames @ {CACHE_SIZE}px)...")

    done = 0
    skipped = 0
    t0 = time.time()
    for i, p in enumerate(paths):
        extracted = preextract_video(p)
        if extracted:
            done += 1
        else:
            skipped += 1
        if (i + 1) % 50 == 0 or i + 1 == total:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  {i+1}/{total}  new={done}  skipped={skipped}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"Done. {done} extracted, {skipped} skipped. Total: {elapsed:.0f}s")
    cache_mb = done * T_CACHE * CACHE_SIZE * CACHE_SIZE * 3 / 1e6
    print(f"Approx cache size: {cache_mb:.0f} MB")


if __name__ == '__main__':
    main()
