"""The R48 device-parity guard must separate 'decoded different frames' from 'FP16 drifted'.

If the preprocessing checksum were exact, a different OpenCV build would fail parity for a reason
that is not quantisation; if it were loose, a genuinely different frame set would pass and FP16
drift would be measured against the wrong reference. Both failure modes are pinned here.
"""
import os
import sys
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load():
    path = os.path.join(ROOT, 'deployment', 'scripts', 'verify-r48-device-parity.py')
    spec = spec_from_file_location('verify_r48_device_parity', path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load()
MEAN = np.array([0.481, 0.457, 0.408], dtype='float32').reshape(1, 3, 1, 1)
STD = np.array([0.268, 0.261, 0.275], dtype='float32').reshape(1, 3, 1, 1)


def _frames(seed, n=4):
    rng = np.random.RandomState(seed)
    return [rng.randint(0, 256, (120, 160, 3), dtype=np.uint8) for _ in range(n)]


def test_preprocess_shape_and_normalisation():
    x = V.preprocess(_frames(0), MEAN, STD)
    assert x.shape == (4, 3, 224, 224)
    assert x.dtype == np.float32 or x.dtype == np.float64
    # channel-wise: undoing the normalisation must land back inside [0, 1]
    raw = x * STD + MEAN
    assert raw.min() >= -1e-5 and raw.max() <= 1 + 1e-5


def test_preprocess_is_deterministic():
    a = V.preprocess(_frames(0), MEAN, STD)
    b = V.preprocess(_frames(0), MEAN, STD)
    assert V.pixel_checksum(a) == V.pixel_checksum(b)


def test_checksum_tolerates_ulp_noise():
    """A few ULPs of resize difference between OpenCV builds is not a finding."""
    c = V.pixel_checksum(V.preprocess(_frames(1), MEAN, STD))
    jittered = [v * (1 + 1e-9) for v in c]
    assert V.pixels_match(jittered, c)


def test_checksum_rejects_different_frames():
    a = V.pixel_checksum(V.preprocess(_frames(1), MEAN, STD))
    b = V.pixel_checksum(V.preprocess(_frames(2), MEAN, STD))
    assert not V.pixels_match(a, b)


def test_checksum_rejects_reordered_frames():
    """The temporal head is fed a sequence, so the right frames in the wrong order is a real defect.
    sum/mean/std are all order-invariant, which is exactly why the checksum carries a
    frame-index-weighted term -- this test is what pins that term in place."""
    fr = _frames(3)
    a = V.pixel_checksum(V.preprocess(fr, MEAN, STD))
    b = V.pixel_checksum(V.preprocess(list(reversed(fr)), MEAN, STD))
    assert a[:3] == pytest.approx(b[:3])        # the order-invariant terms cannot see the shuffle
    assert not V.pixels_match(a, b)             # the weighted term can


def test_decision_gate_is_total_agreement():
    """Anything below 100% means an FP16 latency cannot be printed beside the FP32 accuracy."""
    assert V.DECISION_GATE == 1.0


def test_stale_fixture_checksum_is_refused_not_truncated():
    """A fixture written before the order-sensitive term was added must raise, not silently pass on
    its shared prefix -- that would report PARITY PASS with the frame-order guard switched off."""
    full = V.pixel_checksum(V.preprocess(_frames(4), MEAN, STD))
    with pytest.raises(ValueError, match='rebuild it'):
        V.pixels_match(full, full[:3])
