#!/usr/bin/env python3
"""STAGE 1 on Kaggle: decode the 1,100 clips ONCE into a uint8 frame cache.

Why decode is a separate stage
------------------------------
The GPU work is trivial -- 1,100 clips x 16 frames = 17,600 images, which even ViT-L/14 finishes
in minutes. Video decode is the expensive part (cv2 seeks frame by frame), and re-paying it per
tower would dominate the session clock and, worse, would let two towers read two different sets of
pixels. Decoding once and caching is therefore also the fairness control that section 6.1 G14
rests on: `the baseline saw different frames` stops being an objection when every tower provably
reads the same array.

What is held byte-identical to the local builder
------------------------------------------------
  * frame indices: np.linspace(0, total-1, 16), seek clamped to total-2 -- frame_sampling.decode
  * geometry: cv2.resize(frame, (224,224)) with default INTER_LINEAR, no aspect-ratio preserving
  * channel order: BGR -> RGB AFTER the resize, as build-isafetybench-clip-features.to_batch does
  * clip order: asserted against kaggle/isb-clip-manifest.json, including its order digest
Normalisation is NOT applied here -- it is tower-specific and belongs to stage 2.

Writes (next to --out): frames.npy (1100,16,224,224,3) uint8 ~2.6 GB, valid.npy (1100,16) uint8.

Usage on Kaggle (GPU not required for this stage):
  python kaggle-decode-frame-cache.py --data /kaggle/input/isafetybench-1100 \
      --manifest /kaggle/input/isb-manifest/isb-clip-manifest.json --out /kaggle/working/cache
"""
import argparse
import hashlib
import json
import os
import time

import cv2
import numpy as np

T = 16
RES = 224


def decode_clip(path, t=T):
    """frame_sampling.decode, with the resize and channel swap folded in. -> (t,224,224,3) uint8."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or t
    buf = np.zeros((t, RES, RES, 3), dtype=np.uint8)
    ok_mask = np.zeros(t, dtype=np.uint8)
    for k, i in enumerate([int(x) for x in np.linspace(0, max(0, total - 1), t)]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i, max(0, total - 2)))
        ok, fr = cap.read()
        if ok and fr is not None:
            buf[k] = cv2.resize(fr, (RES, RES))[:, :, ::-1]
            ok_mask[k] = 1
    cap.release()
    return buf, ok_mask


def resolve(data_dir, clip_id, label):
    """The uploaded dataset must keep the hazard/ and normal/ split the labels were derived from."""
    sub = 'hazard' if label == 1 else 'normal'
    p = os.path.join(data_dir, sub, clip_id)
    if os.path.exists(p):
        return p
    flat = os.path.join(data_dir, clip_id)        # tolerate a flattened upload
    return flat if os.path.exists(flat) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='dir holding hazard/ and normal/')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', default='/kaggle/working/cache')
    ap.add_argument('--limit', type=int, default=None, help='smoke test on the first N clips')
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    cids, labels = man['clip_ids'], man['labels']
    dig = hashlib.sha256('\n'.join(cids).encode()).hexdigest()
    assert dig == man['order_sha256'], 'manifest is internally inconsistent; re-export it'
    assert man['T'] == T and man['resize'] == RES, (man['T'], man['resize'])
    n = args.limit or man['n']

    missing = [c for c, l in zip(cids[:n], labels[:n]) if resolve(args.data, c, l) is None]
    if missing:
        raise SystemExit(f'[decode] {len(missing)} clips not found under {args.data}. '
                         f'First few: {missing[:5]}. Upload is incomplete -- stop here, because a '
                         f'short cache cannot be paired with the published control.')

    os.makedirs(args.out, exist_ok=True)
    frames = np.lib.format.open_memmap(f'{args.out}/frames.npy', mode='w+',
                                       dtype=np.uint8, shape=(n, T, RES, RES, 3))
    valid = np.zeros((n, T), dtype=np.uint8)
    t0 = time.time()
    for k in range(n):
        frames[k], valid[k] = decode_clip(resolve(args.data, cids[k], labels[k]))
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            print(f'  [decode] {k+1}/{n}  {el/(k+1):.2f}s/clip  '
                  f'eta {(n-k-1)*el/(k+1)/60:.1f} min', flush=True)
    frames.flush()
    np.save(f'{args.out}/valid.npy', valid)
    json.dump({'n': n, 'T': T, 'resize': RES, 'order_sha256': dig,
               'dead_frames': int((valid == 0).sum()),
               'clips_fully_dead': int((valid.sum(1) == 0).sum()),
               'opencv': cv2.__version__},
              open(f'{args.out}/cache-meta.json', 'w'), indent=1)
    print(f'[decode] {n} clips in {(time.time()-t0)/60:.1f} min  '
          f'dead frames={int((valid == 0).sum())}  '
          f'fully dead clips={int((valid.sum(1) == 0).sum())}  opencv={cv2.__version__}')
    print(f'[decode] cache at {args.out} -- stage 2 reads it; do NOT re-decode per tower')


if __name__ == '__main__':
    main()
