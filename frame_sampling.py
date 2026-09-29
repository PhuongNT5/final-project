"""The one frame sampler every iSafetyBench feature builder uses.

Extracted so the CLIP entity/whole-frame arms and the VideoMAE opponent provably decode the SAME
frame indices from the SAME files. If each builder kept its own copy, "the baseline saw different
frames" would be an untestable objection; here it is a single function.

Indices are `np.linspace(0, total-1, T)`, exactly what build_relation_tokens.process_clip takes, so
the clip set and ordering stay identical to every run already on the number sheet.
"""
import cv2
import numpy as np


def decode(path, t):
    """-> (frames, width, height). `frames` has length t; an unreadable frame is None."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or t
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    frames = []
    for i in [int(x) for x in np.linspace(0, max(0, total - 1), t)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i, max(0, total - 2)))
        ok, fr = cap.read()
        frames.append(fr if ok else None)
    cap.release()
    return frames, w, h
