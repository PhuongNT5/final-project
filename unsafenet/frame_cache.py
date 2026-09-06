"""Decode every clip once into T uniformly-spaced JPEG frames + a median background image.

Sequential decode (no seeking) keeps this fast; the cache is reused by every later stage
(detector inference, frame classifier, temporal features), so decoding never repeats.
Layout:  cache_r2/frames/<subset>/<clip>/f00.jpg .. f31.jpg  and  bg.jpg
"""
from __future__ import annotations
import os
import sys

import cv2
import numpy as np

from . import config as C
from .data_index import load_clips, Clip


def clip_dir(c: Clip) -> str:
    return os.path.join(C.FRAME_DIR, c.subset, c.name.replace('.mp4', ''))


def frame_paths(c: Clip, t: int = C.T) -> list[str]:
    d = clip_dir(c)
    return [os.path.join(d, f'f{i:02d}.jpg') for i in range(t)]


def bg_path(c: Clip) -> str:
    return os.path.join(clip_dir(c), 'bg.jpg')


def is_cached(c: Clip) -> bool:
    return os.path.exists(bg_path(c)) and all(os.path.exists(p) for p in frame_paths(c))


def _read_uniform(path: str, t: int):
    """Sequential read; return t frames uniformly spaced over the decoded stream."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(fr, (C.CACHE_W, C.CACHE_H), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, t).round().astype(int)
    return [frames[i] for i in idx], frames


def build(subsets=('train', 'test'), t: int = C.T, force: bool = False) -> None:
    clips = load_clips(subsets)
    todo = [c for c in clips if force or not is_cached(c)]
    print(f'[frame_cache] {len(todo)}/{len(clips)} clips to decode', flush=True)
    for k, c in enumerate(todo):
        d = clip_dir(c)
        os.makedirs(d, exist_ok=True)
        got = _read_uniform(c.path, t)
        if not got:
            print(f'  !! unreadable {c.path}', file=sys.stderr)
            continue
        sel, allf = got
        for i, fr in enumerate(sel):
            cv2.imwrite(os.path.join(d, f'f{i:02d}.jpg'), fr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        # background = temporal median over <=48 evenly spaced frames (moving people vanish)
        sub = [allf[i] for i in np.linspace(0, len(allf) - 1, min(48, len(allf))).round().astype(int)]
        bg = np.median(np.stack(sub), axis=0).astype(np.uint8)
        cv2.imwrite(os.path.join(d, 'bg.jpg'), bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if (k + 1) % 50 == 0:
            print(f'  [{k+1}/{len(todo)}] {c.key}', flush=True)
    print('[frame_cache] done', flush=True)


if __name__ == '__main__':
    build(force='--force' in sys.argv)
