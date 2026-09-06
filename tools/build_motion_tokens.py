#!/usr/bin/env python3
"""Phase 2 · Motion — append track-linked motion features to relation tokens.

WalkwayViol (a person CROSSING a walkway) vs UnauthorInt (a person STANDING in a restricted zone) is
the core LCO confusion; both are dominated by Person_Unauthorized so static geometry alone is weak.
Motion separates them: a crosser has high net displacement + a straight path; an intruder loiters.

This post-processes an EXISTING token set that still carries (xc,yc) — i.e. the appearance base BEFORE
L1 strips coords. For each clip it greedily links person boxes (class 0/1) across the T frames by nearest
centroid, then appends 2 camera-robust, track-level dims (broadcast to every token of the track):
  mot_speed    = mean per-frame centroid displacement / bbox-height   (body-heights per frame; scale-free)
  mot_straight = net displacement / path length                        (1=straight line, 0=wandering)
Non-person / unlinked tokens get 0. Run build_l1_tokens.py afterwards to strip (xc,yc) but keep motion.

Usage: build_motion_tokens.py --src yolo26app_ --dst yolo26mot_
Then:  build_l1_tokens.py --src yolo26mot_ --dst yolo26motL1_    (drops xc,yc; keeps the 2 motion dims)
"""
import argparse, os
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = os.path.join(ROOT, 'sweep_results', 'relation_tokens')
PERSON = (0, 1)
GATE = 0.12          # max normalized centroid jump to link a box to a track between adjacent frames
SPEED_CAP = 5.0      # clip runaway speeds (detector id-swaps) to keep the feature bounded


def link_tracks(frames):
    """frames: list (len T) of lists of (slot, cx, cy, h). Greedy nearest-centroid chaining.
    Returns tracks: list of list of (t, slot, cx, cy, h)."""
    tracks, active = [], []          # active: indices into tracks with a live last position
    for t, boxes in enumerate(frames):
        used = set()
        # extend existing tracks by nearest unused box within GATE
        for ti in active[:]:
            lt = tracks[ti][-1]
            best, bd = None, GATE
            for j, (slot, cx, cy, h) in enumerate(boxes):
                if j in used:
                    continue
                d = ((cx - lt[2]) ** 2 + (cy - lt[3]) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, j
            if best is None:
                active.remove(ti)                     # track ended
            else:
                slot, cx, cy, h = boxes[best]
                tracks[ti].append((t, slot, cx, cy, h)); used.add(best)
        # unused boxes start new tracks
        for j, (slot, cx, cy, h) in enumerate(boxes):
            if j not in used:
                tracks.append([(t, slot, cx, cy, h)]); active.append(len(tracks) - 1)
    return tracks


def track_motion(tr):
    """(speed, straightness) for one track; 0/0 if too short to define motion."""
    if len(tr) < 2:
        return 0.0, 0.0
    pts = np.array([(cx, cy) for _, _, cx, cy, _ in tr], dtype=np.float32)
    hs = np.array([h for _, _, _, _, h in tr], dtype=np.float32)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)      # per-frame displacement
    hbar = float(np.clip(np.mean(hs), 1e-3, None))            # scale reference = mean body height
    speed = float(np.clip(np.mean(steps) / hbar, 0.0, SPEED_CAP))
    path = float(np.sum(steps))
    net = float(np.linalg.norm(pts[-1] - pts[0]))
    straight = float(net / path) if path > 1e-6 else 0.0
    return speed, straight


def convert(src, dst, split):
    d = torch.load(os.path.join(TOK, f'tokens_{src}{split}.pt'), weights_only=False)
    tok = d['tokens']; mask = d['mask']                       # (N,T,B,F), (N,T,B)
    N, T, B, F = tok.shape
    lay = d.get('feat_layout')
    ix = {n: i for i, n in enumerate(lay)} if lay else {'class': 0, 'xc': 1, 'yc': 2, 'h': 4}
    out = torch.zeros(N, T, B, F + 2, dtype=tok.dtype)
    out[..., :F] = tok
    linked = 0
    for n in range(N):
        frames = []
        for t in range(T):
            boxes = []
            for b in range(B):
                if mask[n, t, b] > 0 and int(round(float(tok[n, t, b, ix['class']]))) in PERSON:
                    boxes.append((b, float(tok[n, t, b, ix['xc']]), float(tok[n, t, b, ix['yc']]),
                                  float(tok[n, t, b, ix['h']])))
            frames.append(boxes)
        for tr in link_tracks(frames):
            spd, strt = track_motion(tr)
            if spd == 0.0 and strt == 0.0:
                continue
            linked += 1
            for (t, slot, _, _, _) in tr:
                out[n, t, slot, F] = spd
                out[n, t, slot, F + 1] = strt
    d['tokens'] = out
    if lay:
        d['feat_layout'] = lay + ['mot_speed', 'mot_straight']
    torch.save(d, os.path.join(TOK, f'tokens_{dst}{split}.pt'))
    print(f"  {split}: {(N, T, B, F)} -> {tuple(out.shape)} | tracks with motion={linked} "
          f"[saved] tokens_{dst}{split}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='yolo26app_', help='base token tag (must still contain xc,yc)')
    ap.add_argument('--dst', default='yolo26mot_')
    args = ap.parse_args()
    for sp in ('train', 'test'):
        convert(args.src, args.dst, sp)


if __name__ == '__main__':
    main()
