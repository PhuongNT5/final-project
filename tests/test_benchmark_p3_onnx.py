"""Regression test for Task 1c — deployment/scripts/benchmark-p3-utils.py /
benchmark-p3-onnx.py.

Old bug: benchmark-p3-onnx.py excluded NMS-equivalent box selection, crop extraction, and zone
geometry from the timed region, and reported only the MEAN crop count (58) even though its own
comment already recorded p95=106 next to it — understating tail latency.

This test checks, against real code (importlib, not reimplementation):
  1. crop_count_percentiles() returns REAL percentiles from the stored mask tensors (matches the
     audit's cited p50=54 / p95=106 / p99~119 / cap=128, or the documented fallback).
  2. postprocess_frame() correctly filters by confidence, caps at N_MAX, and produces
     normalised [0,1] box coords — using the SAME units (CLIP_W x CLIP_H) it expects.
  3. extract_crops() returns crop_size x crop_size images for each input box.

Usage: .venv/bin/python tests/test_benchmark_p3_onnx.py
"""
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location(
    'benchmark_p3_utils', os.path.join(ROOT, 'deployment', 'scripts', 'benchmark-p3-utils.py'))
U = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(U)

import build_relation_tokens as BRT  # noqa: E402  (read-only)


def test_crop_count_percentiles_real_or_documented_fallback():
    out = U.crop_count_percentiles()
    for k in ('p50', 'p95', 'p99', 'cap'):
        assert k in out and isinstance(out[k], int), out
    assert out['p50'] <= out['p95'] <= out['p99'] <= out['cap'], out
    if out['source'] == 'measured':
        # matches the audit's cited numbers (sweep_results/relation_tokens/*.pt, 516 clips)
        assert out['p50'] == 54, out
        assert out['p95'] == 106, out
        assert 115 <= out['p99'] <= 122, out
        assert out['cap'] == 128, out
    else:
        assert out == U.FALLBACK_CROP_PCTS, out


def test_postprocess_frame_filters_by_confidence_and_caps_at_n_max():
    n = 300
    rng = np.random.default_rng(0)
    x1y1 = rng.uniform(0, U.CLIP_W * 0.5, size=(n, 2)) * [1, U.CLIP_H / U.CLIP_W]
    wh = rng.uniform(10, 100, size=(n, 2))
    x2y2 = np.clip(x1y1 + wh, [0, 0], [U.CLIP_W, U.CLIP_H])
    conf = np.zeros(n, dtype=np.float32)
    conf[:5] = 0.9   # only 5 boxes clear BRT.CONF
    cls = np.zeros(n, dtype=np.float32)
    raw = np.concatenate([x1y1, x2y2, conf[:, None], cls[:, None]], axis=1).astype(np.float32)

    boxes = U.postprocess_frame(raw, zones=None)
    assert len(boxes) == min(5, BRT.N_MAX), (len(boxes), BRT.N_MAX)
    for b in boxes:
        assert 0.0 <= b['cx'] <= 1.0 and 0.0 <= b['cy'] <= 1.0, b   # unit-consistent normalisation
        assert 0.0 <= b['w'] <= 1.0 and 0.0 <= b['h'] <= 1.0, b

    # below-threshold-only input -> no boxes
    raw_empty = raw.copy()
    raw_empty[:, 4] = 0.0
    assert U.postprocess_frame(raw_empty, zones=None) == []


def test_extract_crops_returns_correct_shape():
    frame = (np.random.rand(U.CLIP_H, U.CLIP_W, 3) * 255).astype(np.uint8)
    boxes = [{'cx': 0.5, 'cy': 0.5, 'w': 0.2, 'h': 0.3},
             {'cx': 0.1, 'cy': 0.1, 'w': 0.05, 'h': 0.05}]
    crops = U.extract_crops(frame, boxes, crop_size=224)
    assert len(crops) == 2
    for c in crops:
        assert c.shape == (224, 224, 3), c.shape


def test_synthetic_raw_frames_units_match_postprocess_frame():
    """Regression for the units-mismatch bug: synthetic boxes must be generated in CLIP_W x
    CLIP_H space, not IMGSZ space, or normalised cx/cy/w/h would fall outside [0,1] en masse."""
    frames = U.synthetic_raw_frames(t_frames=4, seed=1)
    assert len(frames) == 4
    out_of_range = 0
    total = 0
    for raw in frames:
        boxes = U.postprocess_frame(raw, zones=None)
        for b in boxes:
            total += 1
            if not (0.0 <= b['cx'] <= 1.0 and 0.0 <= b['cy'] <= 1.0):
                out_of_range += 1
    assert total > 0
    assert out_of_range == 0, f'{out_of_range}/{total} boxes out of [0,1] — units mismatch'


if __name__ == '__main__':
    import traceback
    failures = 0
    mod = sys.modules[__name__]
    for name in sorted(dir(mod)):
        if not name.startswith('test_'):
            continue
        fn = getattr(mod, name)
        try:
            fn()
            print(f'  PASS  {name}')
        except Exception:
            failures += 1
            print(f'  FAIL  {name}')
            traceback.print_exc()
    print('ALL PASS' if failures == 0 else f'{failures} FAILURE(S)')
    sys.exit(1 if failures else 0)
