"""Stub TRT runners and scene fixtures for the P3 streaming tests.

Separated from test_p3_stream_decision.py so the assertions and the fixtures they run against stay
independently readable. Hyphenated and loaded via importlib, the same pattern as the deployment
scripts — pytest must not collect this file.

Stub mirrors TRTRunner's public surface (.inputs/.input_order/.run) and nothing else: the point is
that a shape or format mistake fails on a dev machine rather than on the Jetson.
"""
import numpy as np

IMGSZ, CROP, T, NMAX, NB = 640, 224, 16, 8, 5
CFG = {'imgsz': IMGSZ, 'crop_px': CROP, 't_window': T, 'n_max': NMAX, 'conf': 0.25,
       'app_buckets': [1, 2, 4, 8], 'token_input': 'tokens', 'mask_input': 'mask',
       'detector_stage': 'stub-detector', 'head_stage': 'stub-head'}


class Stub:
    """TRTRunner's public surface only, recording every call's input shapes."""

    def __init__(self, inputs, out_fn):
        self.inputs = {k: {'shape': v, 'host': np.zeros(1, dtype=np.float32)}
                       for k, v in inputs.items()}
        self.input_order = list(inputs)
        self.out_fn = out_fn
        self.calls = []

    def run(self, **kw):
        self.calls.append({k: v.shape for k, v in kw.items()})
        return self.out_fn(**kw)


def det_out(**kw):
    """NB boxes above CONF in letterboxed pixel space, rest below. Classes 0,1,0,1,2 — the last
    must receive zero geometry: geometry() runs for classes 0/1 only."""
    raw = np.zeros((1, 300, 6), dtype=np.float32)
    raw[0, :, 4] = 0.01
    for i in range(NB):
        x1, y1 = 100 + 60 * i, 150 + 40 * i
        raw[0, i] = [x1, y1, x1 + 55, y1 + 120, 0.9 - 0.05 * i, [0, 1, 0, 1, 2][i]]
    return raw


def runners(det_out=det_out, app_out=None, head_out=None):
    d = Stub({'images': (1, 3, IMGSZ, IMGSZ)}, det_out)
    app_out = app_out or (lambda b, **kw: np.zeros((b, 64), np.float32))
    apps = {b: Stub({'crops': (b, 3, CROP, CROP)}, lambda b=b, **kw: app_out(b, **kw))
            for b in (1, 2, 4, 8)}
    head = Stub({'tokens': (1, T, NMAX, 74), 'mask': (1, T, NMAX)},
                head_out or (lambda **kw: np.array([[0.3, 0.7]], np.float32)))
    return d, apps, head


def scene(P3):
    """(shapely zones for one view, preprocessed 1920x1080 frame) — what frame_tokens takes."""
    frame = (np.random.default_rng(0).random((1080, 1920, 3)) * 255).astype(np.uint8)
    return next(iter(P3.BRT.load_zones().values())), P3.preprocess(frame, IMGSZ)
