"""Sliding-window ESTA inference: video in, per-window unsafe probability out.

The evaluation pipeline samples 16 frames spread across a whole short clip. A demo runs on a longer
video, so the same 16-frame tensor is instead filled from a sliding window. Everything downstream of
the window — detector, crop rule, CLIP tower, token layout — is imported from the feature builder
rather than re-implemented, so the demo cannot silently drift from the thing that was measured.
"""
import importlib.util
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import build_relation_tokens as B          # noqa: E402
import build_isafetybench_tokens as ISBT   # noqa: E402
import esta_training as T                  # noqa: E402


def _builder():
    spec = importlib.util.spec_from_file_location(
        'clipfeat', os.path.join(ROOT, 'build-isafetybench-clip-features.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class EstaWindowPipeline:
    """Detector + CLIP + the deployed ESTA ensemble, applied to one window of frames at a time."""

    def __init__(self, ckpt_path, detector='yolo11s.pt', device=None, conf=0.25, imgsz=640):
        from ultralytics import YOLO
        self.bld = _builder()
        B.IMGSZ, B.CONF, B.CORRUPT = imgsz, conf, None
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.det = YOLO(detector)
        self.clip = self.bld.load_clip(self.device)
        self.lut = ISBT.coco_to_group()
        ck = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        self.name = ck['model']
        self.models = []
        for st in ck['states']:
            m = T.build(self.name, ck.get('in_dim'))
            m.load_state_dict(st); m.eval()
            self.models.append(m)
        self.val_macro = ck.get('val_macro')

    def features(self, frames, W, H):
        """-> (entity (1,T,N,512), geom (1,T,N,6), mask (1,T,N), boxes per frame in pixels)."""
        ent = np.zeros((B.T, B.N_MAX, self.bld.CLIP_DIM), dtype=np.float32)
        geo = np.zeros((B.T, B.N_MAX, self.bld.GEOM), dtype=np.float32)
        msk = np.zeros((B.T, B.N_MAX), dtype=np.float32)
        drawn = [[] for _ in range(B.T)]
        for t, fr in enumerate(frames[:B.T]):
            if fr is None:
                continue
            r = self.det.predict(fr, imgsz=B.IMGSZ, conf=B.CONF, verbose=False,
                                 device=self.device)[0]
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xywhn = b.xywhn.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            conf = b.conf.cpu().numpy()
            order = np.argsort(-conf)[:B.N_MAX]
            crops = []
            for j, oi in enumerate(order):
                cx, cy, w, h = xywhn[oi]
                crops.append(self.bld.crop_box(fr, cx, cy, w, h, W, H))
                x1, y1, x2, y2 = self.bld.box_px(cx, cy, w, h, W, H) or (0, 0, W, H)
                geo[t, j] = [self.lut[int(np.clip(cls[oi], 0, 79))], cx, cy, w, h, float(conf[oi])]
                msk[t, j] = 1.0
                drawn[t].append((x1, y1, x2, y2, ISBT.GROUP_NAMES[self.lut[int(np.clip(cls[oi], 0, 79))]],
                                 float(conf[oi])))
            emb = self.bld.embed(crops, self.clip)
            ent[t, :len(emb)] = emb
        pack = {'entity': torch.from_numpy(ent)[None], 'geom': torch.from_numpy(geo)[None],
                'mask': torch.from_numpy(msk)[None]}
        return pack, drawn

    @torch.no_grad()
    def unsafe_prob(self, pack):
        """Mean softmax over the seed ensemble — the same ensemble the reported OOF numbers used."""
        parts = T.parts_for(self.name, pack, np.array([0]))
        p = torch.stack([torch.softmax(m(*parts), 1)[0, 1] for m in self.models])
        return float(p.mean()), [float(v) for v in p]
