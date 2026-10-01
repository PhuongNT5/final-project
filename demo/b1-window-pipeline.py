"""Sliding-window inference for the deployed R48 path: video in, per-window unsafe probability out.

Two obligations of the system, kept apart on purpose:

  CLASSIFY  frozen CLIP ViT-B/32 over the WHOLE frame -> B1 temporal head, 3-seed ensemble. The
            detector is NOT an input. This is the path that produced the LSO-1100 balanced accuracy
            and the path timed on the Jetson, so the demo's verdict is the measured system's verdict.
  LOCALISE  one YOLOv11s pass per frame, drawn on the video and counted in the timeline. It never
            reaches the classifier: gates G14/G17/G18 found the entity arm added no accuracy, so
            routing it back into the decision would make the demo disagree with the reported number.

The evaluation samples 16 frames spread across a whole short clip. A demo runs on a longer video, so
the same 16-frame tensor is filled from a sliding window instead. The CLIP tower, its preprocessing
and T are imported from the feature builder rather than re-implemented, so the demo cannot silently
drift from the thing that was measured.
"""
import importlib.util
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import build_relation_tokens as B   # noqa: E402  (T, IMGSZ, CONF, N_MAX -- shared with the builder)
import esta_training as T           # noqa: E402

# Checkpoints this pipeline can serve. An entity-arm checkpoint (ESTA/TSF) needs crops and geometry
# that are deliberately absent here, so it is refused rather than fed zeros.
WHOLE_FRAME_MODELS = ('B1', 'B1w')


def require_whole_frame(name, ckpt_path):
    """Refuse an entity-arm checkpoint instead of feeding it zeros for the crops it expects.

    Checked before the 350 MB CLIP tower loads, and split out as a function so the refusal itself is
    testable without standing up the tower.
    """
    if name in WHOLE_FRAME_MODELS:
        return name
    raise SystemExit(
        f"{os.path.relpath(ckpt_path, ROOT)} holds a '{name}' checkpoint, which reads the entity "
        f"arm. This demo serves the deployed whole-frame path {WHOLE_FRAME_MODELS}; train one with "
        f"train-esta-deployment-checkpoint.py --model B1")


def _builder():
    spec = importlib.util.spec_from_file_location(
        'clipfeat', os.path.join(ROOT, 'build-isafetybench-clip-features.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class B1WindowPipeline:
    """Whole-frame CLIP + the deployed B1 ensemble for the verdict; YOLOv11s for the boxes.

    `detector=None` disables localisation entirely. The verdict is then produced by exactly the same
    arithmetic -- that the class decision survives the detector being switched off is the property
    this split exists to make visible.
    """

    def __init__(self, ckpt_path, detector='yolo11s.pt', device=None, conf=None, imgsz=None):
        ck = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        self.name = require_whole_frame(ck['model'], ckpt_path)
        self.models = []
        for st in ck['states']:
            m = T.build(self.name, ck.get('in_dim'))
            m.load_state_dict(st); m.eval()
            self.models.append(m)
        self.val_macro = ck.get('val_macro')
        self.bld = _builder()
        self.imgsz = B.IMGSZ if imgsz is None else imgsz
        self.conf = B.CONF if conf is None else conf
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.clip = self.bld.load_clip(self.device)
        self.det = None
        if detector:
            from ultralytics import YOLO
            self.det = YOLO(detector)

    def features(self, frames):
        """-> pack {'wholeframe' (1,T,512), 'valid' (1,T)}: the classifier's only inputs.

        One CLIP pass over the frames that decoded. A frame that failed to decode stays zero AND is
        marked invalid, so the temporal head masks it; letting a zero vector count as content would
        feed the head a frame the video never had.
        """
        wf = np.zeros((B.T, self.bld.CLIP_DIM), dtype=np.float32)
        valid = np.zeros(B.T, dtype=np.float32)
        present = [(t, fr) for t, fr in enumerate(frames[:B.T]) if fr is not None]
        if present:
            for (t, _), e in zip(present, self.bld.embed([fr for _, fr in present], self.clip)):
                wf[t], valid[t] = e, 1.0
        return {'wholeframe': torch.from_numpy(wf)[None], 'valid': torch.from_numpy(valid)[None]}

    def localise(self, frames, W, H):
        """-> per frame, a list of (x1, y1, x2, y2, class name, confidence) in pixels.

        Capped at B.N_MAX boxes per frame by descending confidence: the same per-frame detector
        budget the edge measurement timed, so the drawn cost and the measured cost agree.
        """
        out = [[] for _ in range(B.T)]
        present = [(t, fr) for t, fr in enumerate(frames[:B.T]) if fr is not None]
        if self.det is None or not present:
            return out
        res = self.det.predict([fr for _, fr in present], imgsz=self.imgsz, conf=self.conf,
                               verbose=False, device=self.device)
        for (t, _), r in zip(present, res):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy().astype(int)
            cls = b.cls.cpu().numpy().astype(int)
            cf = b.conf.cpu().numpy()
            for j in np.argsort(-cf)[:B.N_MAX]:
                x1, y1, x2, y2 = xyxy[j]
                out[t].append((max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2)),
                               str(r.names.get(int(cls[j]), cls[j])), float(cf[j])))
        return out

    @torch.no_grad()
    def unsafe_prob(self, pack):
        """Mean softmax over the seed ensemble -- the same ensemble the reported OOF numbers used.

        The two arguments, in this order, are exactly what `esta_training.parts_for('B1', ...)` hands
        the model during training and evaluation; tests/test-demo-b1-pipeline.py pins that, so the
        demo cannot be fed its inputs in a different order than the measured path was.
        """
        p = torch.stack([torch.softmax(m(pack['wholeframe'], pack['valid']), 1)[0, 1]
                         for m in self.models])
        return float(p.mean()), [float(v) for v in p]
