"""The preprocessing fix must not move a single pixel.

verify-r48-device-parity.preprocess was changed to swap colour channels AFTER the resize instead of
before it, because the before-version forces OpenCV to copy the whole full-resolution frame to
escape a negative-stride view (90,9 of 101,0 ms at 1080p). That is a latency fix, and it is only
legitimate if the pixels are bit-identical: the deployed 84,2 % balanced accuracy was measured
through the old pixels, and the 3-seed head was trained on features built by the same formula.

So this file hard-codes the PRE-CHANGE formula as the reference and demands exact equality -- no
tolerance. If a future OpenCV build ever makes the two orders disagree, this fails instead of the
thesis quietly reporting a latency win on a different system.
"""
import os
import sys
from importlib.util import module_from_spec, spec_from_file_location

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'deployment', 'scripts')
SHAPES = [(240, 320), (1080, 1920), (137, 251), (224, 224), (720, 1280)]


def _load(name, fname, where=SCRIPTS):
    spec = spec_from_file_location(name, os.path.join(where, fname))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PAR = _load('verify_r48_device_parity', 'verify-r48-device-parity.py')
PROF = _load('profile_r48_preprocess', 'profile-r48-preprocess.py')
VAR = _load('preprocess_variants_utils', 'preprocess-variants-utils.py')


def reference_preprocess(frames, mean, std):
    """The formula as it stood before the fix, written out here on purpose. Do not refactor this to
    call the production function -- it is the independent side of the comparison."""
    ims = [cv2.resize(f[:, :, ::-1], (224, 224)).astype('float32') / 255.0 for f in frames]
    x = np.stack(ims).transpose(0, 3, 1, 2)
    return (x - mean) / std


def _fixture(seed, shape, n=4):
    rng = np.random.default_rng(seed)
    frames = [rng.integers(0, 256, (shape[0], shape[1], 3), dtype=np.uint8) for _ in range(n)]
    mean = rng.random((1, 3, 1, 1), dtype=np.float32)
    std = (0.1 + rng.random((1, 3, 1, 1), dtype=np.float32)).astype('float32')
    return frames, mean, std


@pytest.mark.parametrize('shape', SHAPES)
def test_preprocess_is_bit_identical_to_the_pre_change_formula(shape):
    frames, mean, std = _fixture(7, shape)
    got, want = PAR.preprocess(frames, mean, std), reference_preprocess(frames, mean, std)
    assert got.shape == want.shape
    assert np.array_equal(got, want), (f'pixels moved at {shape}: '
                                       f'max |d| = {np.abs(got - want).max()}')


def test_preprocess_keeps_float32_so_the_caller_copy_can_be_elided():
    """benchmark-r48-composed-cadence now calls .astype('float32', copy=False). That only avoids a
    copy if this function already returns float32 -- if it ever returned float64 the elision would
    silently become a conversion again."""
    frames, mean, std = _fixture(11, (240, 320))
    x = PAR.preprocess(frames, mean, std)
    assert x.dtype == np.float32
    assert x.astype('float32', copy=False) is x


def test_profiler_baseline_reproduces_the_pre_change_formula():
    """`VAR.current` is the baseline every speedup is divided by. If it drifts from the pre-change
    formula, every ratio in the chapter is against the wrong denominator."""
    frames, mean, std = _fixture(13, (240, 320))
    assert np.array_equal(VAR.current(frames, mean, std), reference_preprocess(frames, mean, std))


@pytest.mark.parametrize('name', ['flip_after_resize', 'nhwc_normalise'])
def test_adoptable_variants_are_bit_identical_not_merely_close(name):
    frames, mean, std = _fixture(17, (1080, 1920), n=3)
    got = VAR.VARIANTS[name](frames, mean, std)
    assert np.array_equal(got, VAR.current(frames, mean, std))


def test_the_rejected_variant_is_declared_as_differing_before_the_run():
    """area_interp exists to price the filter the system did not choose. It must stay OUT of the
    adoptable set by declaration, so a run cannot adopt it by accident."""
    assert 'area_interp' in VAR.EXPECT_DIFF
    frames, mean, std = _fixture(19, (240, 320), n=2)
    assert not np.array_equal(VAR.VARIANTS['area_interp'](frames, mean, std),
                              VAR.current(frames, mean, std))


def test_host_runs_are_not_labelled_device_numbers():
    """A host profile is useful for the bit-exactness gate but its timings are not device timings;
    the flag is derived from the platform so it cannot be set by a forgotten argument."""
    assert PROF.IS_DEVICE == os.path.exists('/etc/nv_tegra_release')
