"""The 30%-context crop must stay byte-identical to the rule the measured features were built with.

`crop_box` was split into `box_px` (geometry) + `crop_box` (slice) so the demo can draw exactly the
rectangle it embeds. That refactor touches the file every reported feature tensor came from, so it
is only safe if the pixels are unchanged. These tests compare against `build_relation_tokens`'s own
padding arithmetic, re-derived here independently, over a grid that includes boxes clipped at every
edge and boxes that degenerate to zero area.

Run: .venv/bin/python tests/test-crop-geometry-matches-token-builder.py
"""
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location(
    'clipfeat', os.path.join(ROOT, 'build-isafetybench-clip-features.py'))
CF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CF)

W, H = 96, 64
FRAME = np.arange(H * W * 3, dtype=np.uint8).reshape(H, W, 3)


def reference(frame, cx, cy, w, h, W, H):
    """The pre-refactor single-expression form, transcribed verbatim from git history."""
    x1 = max(0, int((cx - w / 2 * 1.3) * W)); y1 = max(0, int((cy - h / 2 * 1.3) * H))
    x2 = min(W, int((cx + w / 2 * 1.3) * W)); y2 = min(H, int((cy + h / 2 * 1.3) * H))
    return frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame


def boxes():
    rng = np.random.default_rng(0)
    grid = [(cx, cy, w, h)
            for cx in (0.0, 0.02, 0.5, 0.98, 1.0)
            for cy in (0.0, 0.02, 0.5, 0.98, 1.0)
            for w in (0.0, 0.001, 0.2, 1.0)
            for h in (0.0, 0.001, 0.2, 1.0)]
    return grid + [tuple(v) for v in rng.random((300, 4))]


def test_crop_pixels_unchanged():
    for cx, cy, w, h in boxes():
        got = CF.crop_box(FRAME, cx, cy, w, h, W, H)
        want = reference(FRAME, cx, cy, w, h, W, H)
        assert got.shape == want.shape, f'shape drift at {(cx, cy, w, h)}: {got.shape} vs {want.shape}'
        assert np.array_equal(got, want), f'pixel drift at {(cx, cy, w, h)}'
    print(f'  ok   crop_box is pixel-identical to the pre-refactor rule on {len(boxes())} boxes')


def test_box_px_slices_to_the_same_crop():
    degenerate = 0
    for cx, cy, w, h in boxes():
        b = CF.box_px(cx, cy, w, h, W, H)
        if b is None:
            degenerate += 1
            assert np.array_equal(CF.crop_box(FRAME, cx, cy, w, h, W, H), FRAME), \
                'a degenerate box must fall back to the whole frame'
            continue
        x1, y1, x2, y2 = b
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H, f'box out of frame: {b}'
        assert np.array_equal(FRAME[y1:y2, x1:x2], CF.crop_box(FRAME, cx, cy, w, h, W, H)), \
            f'box_px does not slice to the embedded crop at {(cx, cy, w, h)}'
    assert degenerate > 0, 'the grid must exercise the degenerate path'
    print(f'  ok   box_px slices to the embedded crop ({degenerate} degenerate boxes exercised)')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f'{len(tests)}/{len(tests)} pass')
