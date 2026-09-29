#!/usr/bin/env python3
"""Check whether any iSafetyBench clip shares footage with Unsafe-Net (feasibility report Q4).

A cross-dataset transfer or dataset-reassignment claim has to state that this was checked, not that
it is unlikely. Method: 64-bit dHash of 3 evenly-spaced frames per clip, then exact-hash intersection
plus the minimum Hamming distance across every cross-corpus frame pair. dHash is robust to
re-encoding and scaling, which is the realistic contamination route (same YouTube clip re-cut), and
its false-positive rate at distance <=6 on 1,791 clips is low enough to inspect by hand.

Usage: .venv/bin/python tools/check-cross-dataset-contamination.py
"""
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = 3
NEAR = 6          # Hamming distance treated as "inspect this pair by hand"


def dhash_frames(path, n=FRAMES):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out = []
    for i in np.linspace(0, max(0, total - 1), n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (9, 8))
        bits = (g[:, 1:] > g[:, :-1]).flatten()
        out.append(np.packbits(bits).view(np.uint64)[0])
    cap.release()
    return out


def corpus(patterns, name):
    paths = sorted(p for pat in patterns for p in glob.glob(os.path.join(ROOT, pat)))
    hashes, owner = [], []
    for k, p in enumerate(paths):
        for h in dhash_frames(p):
            hashes.append(h); owner.append(os.path.relpath(p, ROOT))
        if (k + 1) % 200 == 0:
            print(f"  [{name}] {k+1}/{len(paths)}", flush=True)
    print(f"[{name}] {len(paths)} clips -> {len(hashes)} frame hashes")
    return np.array(hashes, dtype=np.uint64), np.array(owner)


def main():
    a, ao = corpus(['data/train/*/*.mp4', 'data/test/*/*.mp4'], 'unsafenet')
    b, bo = corpus(['data/isafetybench/hazard/*.mp4', 'data/isafetybench/normal/*.mp4'], 'isafetybench')
    if not len(a) or not len(b):
        print('ABORT: one corpus produced no hashes'); return 1

    shared = set(a.tolist()) & set(b.tolist())
    print(f"\nexact frame-hash collisions: {len(shared)}")
    for h in list(shared)[:20]:
        print(f"  {h}: unsafenet={sorted(set(ao[a == h]))[:3]} isafetybench={sorted(set(bo[b == h]))[:3]}")

    # min Hamming over the full cross product, chunked so the (n_a x n_b) popcount never materialises
    lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    best, best_pair = 64, None
    for i in range(0, len(a), 512):
        x = a[i:i + 512, None] ^ b[None, :]
        d = lut[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1)
        j = int(d.argmin()); r, c = divmod(j, d.shape[1])
        if d[r, c] < best:
            best, best_pair = int(d[r, c]), (ao[i + r], bo[c])
    print(f"\nminimum cross-corpus Hamming distance: {best}/64  pair={best_pair}")
    print("VERDICT: " + ("NO contamination detected (no exact collision, min distance > "
                         f"{NEAR})" if not shared and best > NEAR else
                         "INSPECT — a collision or near-duplicate exists, resolve before any "
                         "transfer claim"))
    return 0


if __name__ == '__main__':
    sys.exit(main())
