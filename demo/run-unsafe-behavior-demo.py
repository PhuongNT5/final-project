"""Unsafe-behaviour demo: a per-window verdict from B1, and YOLOv11s boxes saying WHERE.

Slides a window over the video, fills the same 16-frame tensor the evaluation uses, and runs the
deployed 3-seed B1 ensemble over whole-frame CLIP features. Writes an annotated .mp4 and a .json
timeline.

The verdict and the boxes come from two separate passes and are reported separately. The detector is
not an input to the class decision -- `--no_localisation` turns it off and the probabilities are
unchanged, which is the point: the 84.2% LSO-1100 balanced accuracy was obtained without it.

The decision threshold is 0.5 and is NOT tuned -- no threshold was ever selected on this corpus, and
picking one for a demo would make the demo disagree with the reported numbers. The probability is
shown alongside the verdict so the margin is visible rather than hidden behind a label.

Usage:
  .venv/bin/python demo/run-unsafe-behavior-demo.py --video path/to/clip.mp4 \
      --ckpt sweep_results/relation_tokens/deploy/b1-deploy-3seed.pt --out demo/out
"""
import argparse
import collections
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import build_relation_tokens as B          # noqa: E402

THRESHOLD = 0.5
DECISION_PATH = ('whole-frame CLIP ViT-B/32 (frozen) -> B1 temporal head, 3-seed ensemble; '
                 'the detector is localisation only and is not an input')
RED, GREEN, WHITE, BLACK = (60, 60, 230), (90, 200, 120), (255, 255, 255), (0, 0, 0)
BAR_BOTTOM = 46   # the verdict bar occupies the top-left corner; box labels are kept clear of it


def pipeline_cls():
    spec = importlib.util.spec_from_file_location(
        'b1_window_pipeline', os.path.join(ROOT, 'demo', 'b1-window-pipeline.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.B1WindowPipeline


def windows(n_frames, win, stride):
    """Start indices of each window; the tail is kept as a short final window, never dropped."""
    starts = list(range(0, max(1, n_frames - win + 1), stride))
    if starts and starts[-1] + win < n_frames:
        starts.append(max(0, n_frames - win))
    return starts or [0]


def sample(cap, start, win, n_frames):
    """T frames spread evenly across one window — the sampling rule the evaluation uses."""
    idx = [int(x) for x in np.linspace(start, min(start + win - 1, n_frames - 1), B.T)]
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        frames.append(fr if ok else None)
    return frames, idx


def localisation_summary(boxes_per_frame):
    """What was localised in this window: how many boxes, and the classes they carried.

    Reported beside the verdict rather than folded into it, so a reader can see the detector
    disagreeing with the classifier instead of that disagreement being averaged away.
    """
    counts = collections.Counter(b[4] for frame in boxes_per_frame for b in frame)
    return {'detections': int(sum(counts.values())),
            'classes': dict(counts.most_common(5))}


def draw(frame, boxes, prob, unsafe):
    h, w = frame.shape[:2]
    for x1, y1, x2, y2, name, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), RED if unsafe else GREEN, 2)
        cv2.putText(frame, f'{name} {conf:.2f}', (x1, max(BAR_BOTTOM + 14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    bar_w = int(0.4 * w)
    cv2.rectangle(frame, (12, 12), (12 + bar_w, BAR_BOTTOM), BLACK, -1)
    cv2.rectangle(frame, (12, 12), (12 + int(bar_w * prob), BAR_BOTTOM), RED if unsafe else GREEN, -1)
    cv2.putText(frame, f"{'UNSAFE' if unsafe else 'normal'}  p={prob:.2f}", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--ckpt', default='sweep_results/relation_tokens/deploy/b1-deploy-3seed.pt')
    ap.add_argument('--out', default='demo/out')
    ap.add_argument('--window', type=int, default=48, help='frames per decision window')
    ap.add_argument('--stride', type=int, default=24)
    ap.add_argument('--detector', default='yolo11s.pt')
    ap.add_argument('--no_localisation', action='store_true',
                    help='skip the detector entirely; the verdicts must come out identical')
    ap.add_argument('--no_video', action='store_true', help='timeline only, skip rendering')
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f'no checkpoint at {args.ckpt} -- run '
                         f'train-esta-deployment-checkpoint.py --model B1 first')
    pipe = pipeline_cls()(args.ckpt, detector=None if args.no_localisation else args.detector)
    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    starts = windows(n, args.window, args.stride)
    print(f'[demo] {os.path.basename(args.video)} {W}x{H} {n} frames @{fps:.1f}fps | '
          f'{len(starts)} windows of {args.window} (stride {args.stride}) | model={pipe.name}')
    print(f'[decision] {DECISION_PATH}')
    print(f"[localisation] {'OFF' if pipe.det is None else f'YOLOv11s @{pipe.imgsz}'}")

    timeline, per_frame = [], {}
    for k, s in enumerate(starts):
        frames, idx = sample(cap, s, args.window, n)
        pack = pipe.features(frames)
        prob, seeds = pipe.unsafe_prob(pack)
        boxes = pipe.localise(frames, W, H)
        loc = localisation_summary(boxes)
        timeline.append({'window': k, 'start_frame': s, 'start_sec': round(s / fps, 2),
                         'prob_unsafe': round(prob, 4), 'per_seed': [round(v, 4) for v in seeds],
                         'verdict': 'unsafe' if prob >= THRESHOLD else 'normal',
                         'frames_decoded': int(pack['valid'].sum()),
                         'localisation': loc})
        for t, fi in enumerate(idx):
            per_frame[fi] = (boxes[t], prob)
        print(f"  window {k:3d} @{s/fps:6.1f}s  p_unsafe={prob:.3f}  "
              f"{'UNSAFE' if prob >= THRESHOLD else 'normal'}  boxes={loc['detections']}",
              flush=True)

    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]
    jp = os.path.join(args.out, f'{base}-timeline.json')
    json.dump({'video': args.video, 'model': pipe.name, 'ckpt': args.ckpt,
               'decision_path': DECISION_PATH,
               'localisation': None if pipe.det is None else {'detector': args.detector,
                                                              'imgsz': pipe.imgsz,
                                                              'conf': pipe.conf},
               'threshold': THRESHOLD, 'threshold_note': 'fixed at 0.5, never tuned',
               'window': args.window, 'stride': args.stride, 'fps': fps,
               'windows': timeline}, open(jp, 'w'), indent=1)
    print(f'[demo] timeline -> {jp}')

    if args.no_video:
        return
    vp = os.path.join(args.out, f'{base}-annotated.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    last = ([], 0.0)
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        last = per_frame.get(i, last)
        vw.write(draw(fr, last[0], last[1], last[1] >= THRESHOLD))
    vw.release(); cap.release()
    print(f'[demo] annotated video -> {vp}')


if __name__ == '__main__':
    main()
