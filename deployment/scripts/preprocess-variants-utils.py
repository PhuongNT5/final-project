"""The preprocessing variants the R48 decision path is profiled over, and the sub-step split.

Separated from profile-r48-preprocess.py so the candidate formulas -- the part that must stay
bit-identical to what the deployed accuracy was measured through -- can be imported by
tests/test-r48-preprocess-variants.py without the CLI, the clip decode, or the timing harness.

`current` is a verbatim copy of the PRE-CHANGE verify-r48-device-parity.preprocess. It is the
baseline every speedup is divided by and the reference every candidate is compared against, so it
must never be refactored into a call to the production function.
"""
import time

import cv2
import numpy as np

SIZE = (224, 224)


def current(frames, mean, std):
    """The formula as verify-r48-device-parity.preprocess stood BEFORE the channel-swap fix. It is
    kept written out here, not imported, because it is the independent side of the comparison: if
    it called the production function the bit-exactness gate would compare that function to
    itself."""
    ims = [cv2.resize(f[:, :, ::-1], SIZE).astype('float32') / 255.0 for f in frames]
    x = np.stack(ims).transpose(0, 3, 1, 2)
    return (x - mean) / std


def flip_after_resize(frames, mean, std):
    """Same arithmetic, but the BGR->RGB channel swap happens on the 224x224 result instead of on
    the full-resolution frame. cv2.resize weights pixels within a channel and never across
    channels, so a channel permutation commutes with it exactly -- the bit-exactness gate is what
    actually establishes that here. The point: f[:, :, ::-1] is a negative-stride view, so OpenCV
    must materialise a contiguous copy of the WHOLE frame (6,2 MB at 1080p) before resizing."""
    ims = [cv2.resize(f, SIZE)[:, :, ::-1].astype('float32') / 255.0 for f in frames]
    x = np.stack(ims).transpose(0, 3, 1, 2)
    return (x - mean) / std


def nhwc_normalise(frames, mean, std):
    """flip_after_resize, plus two more changes that are element-wise and therefore cannot move a
    pixel. (a) The uint8->float32 conversion and the /255 are done once on the stacked
    (16,224,224,3) block instead of 16 times on separate arrays. (b) The mean/std normalise runs on
    that contiguous NHWC block and the NCHW transpose happens last -- the current order normalises
    through a transposed VIEW, so every element is gathered with a stride; that is why the normalise
    costs 7,2 ms for 2,4 M floats on the host."""
    ims = np.stack([cv2.resize(f, SIZE)[:, :, ::-1] for f in frames])
    x = (ims.astype('float32') / 255.0 - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def area_interp(frames, mean, std):
    """NOT a candidate: INTER_AREA is the correct filter for a >4x downscale and will fail the
    bit-exactness gate by construction. Timed only so the chapter can say what the current
    INTER_LINEAR choice costs relative to the alternative it did not take."""
    ims = [cv2.resize(f, SIZE, interpolation=cv2.INTER_AREA)[:, :, ::-1].astype('float32') / 255.0
           for f in frames]
    x = np.stack(ims).transpose(0, 3, 1, 2)
    return (x - mean) / std


VARIANTS = {'current': current, 'flip_after_resize': flip_after_resize,
            'nhwc_normalise': nhwc_normalise, 'area_interp': area_interp}
EXPECT_DIFF = {'area_interp'}      # declared before the run, not after


def stage_split(frames, mean, std, runs, pct):
    """Where the time goes inside `current`. Each sub-step is timed on its own inputs, so the
    numbers are additive by construction and can be checked against the whole-call figure."""
    out, flipped, rgb224 = {}, None, None

    def t(name, fn):
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            r = fn()
            ts.append((time.perf_counter() - t0) * 1000)
        out[name] = pct(ts)
        return r

    flipped = t('a_channel_flip_full_res', lambda: [np.ascontiguousarray(f[:, :, ::-1])
                                                    for f in frames])
    t('b_resize_from_flipped', lambda: [cv2.resize(f, SIZE) for f in flipped])
    rgb224 = t('c_resize_then_flip', lambda: [cv2.resize(f, SIZE)[:, :, ::-1] for f in frames])
    small = t('d_astype_div255_per_frame',
              lambda: [np.ascontiguousarray(f).astype('float32') / 255.0 for f in rgb224])
    st = t('e_stack_transpose', lambda: np.stack(small).transpose(0, 3, 1, 2))
    t('f_normalise', lambda: (st - mean) / std)
    t('g_redundant_astype_float32', lambda: ((st - mean) / std).astype('float32'))
    return out
