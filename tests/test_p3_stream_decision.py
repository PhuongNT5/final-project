"""Tests for the P3 streaming decision path: jetson-p3-decision-utils.py (per-frame token
assembly) and jetson-p3-multi-stream.py (concurrent cadence harness).

Runners are stubbed to TRTRunner's public surface (.inputs/.input_order/.run), so everything but
the TRT calls runs on a dev machine — the harness runs unattended on the Jetson, where a shape or
format mistake costs a device session.

One bug this already caught: jetson-p3-demo-video.load_zones_for_clip returns RAW coordinate lists
(it only draws them with cv2.fillPoly) while geometry() needs Shapely polygons, raising
AttributeError on .area mid-run. zones_for_clip() now takes polygons from BRT.load_zones().

Three tests need real clips under data/test, which is gitignored — they SKIP, never pass silently.
Usage: .venv/bin/python tests/test_p3_stream_decision.py   (or via pytest)
"""
import asyncio
import glob
import importlib.util
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _clips(n=1):
    """Real clips, or skip — never a bare `return`, which prints PASS for a test that asserted
    nothing. SkipTest is honoured by pytest and counted separately by __main__ below."""
    found = sorted(glob.glob(os.path.join(ROOT, 'data/test/*/*.mp4')))
    if len(found) < n:
        raise unittest.SkipTest(f'need {n} clip(s) under data/test, found {len(found)}')
    return found


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P3 = _load('jetson_p3_decision_utils', 'deployment/scripts/jetson-p3-decision-utils.py')
MS = _load('jetson_p3_multi_stream', 'deployment/scripts/jetson-p3-multi-stream.py')
STUBS = _load('p3_stream_stubs', 'tests/p3-stream-stubs.py')
IMGSZ, CROP, T, NMAX, NB, CFG = STUBS.IMGSZ, STUBS.CROP, STUBS.T, STUBS.NMAX, STUBS.NB, STUBS.CFG
_stubs, _scene = STUBS.runners, lambda: STUBS.scene(P3)


def test_frame_tokens_layout_matches_token_builder():
    """Row = [class, xc, yc, w, h, conf] + 4 geometry + APP_DIM appearance, geometry columns
    numerically equal to build_relation_tokens.geometry() on the same box."""
    det, apps, _ = _stubs(
        app_out=lambda b, **k: np.arange(b * 64, dtype=np.float32).reshape(b, 64) + 1)
    zones, packed = _scene()
    assert packed[0].shape == (1, 3, IMGSZ, IMGSZ)
    rows, mask = P3.frame_tokens(det, apps, zones, packed, P3.BRT.CONF, NMAX, CROP)

    F = P3.BRT.FEAT
    assert rows.shape == (NMAX, F + P3.BRT.APP_DIM) and mask.shape == (NMAX,)
    assert mask.sum() == NB and rows[NB:].sum() == 0, 'unmasked slots must stay zero'
    assert list(rows[:NB, 0]) == [0, 1, 0, 1, 2]
    assert np.allclose(rows[:NB, 5], [0.9, 0.85, 0.8, 0.75, 0.7], atol=1e-6)
    # letterbox round-trip: a 55x120 px box at IMGSZ maps back to the 1920x1080 frame
    s = min(IMGSZ / 1920, IMGSZ / 1080)
    assert np.allclose(rows[:NB, 3], 55 / s / 1920, atol=1e-6)
    assert np.allclose(rows[:NB, 4], 120 / s / 1080, atol=1e-6)
    geo = rows[:NB, 6:F]
    assert np.abs(geo).sum() > 0, 'all-zero geometry means the zone polygons never applied'
    assert np.abs(geo[4]).sum() == 0, 'class 2 must get no geometry'
    ref = [P3.BRT.geometry(*rows[i, 1:5], 1920, 1080, zones) for i in range(4)]
    assert np.allclose(geo[:4], np.array(ref, dtype=np.float32), atol=1e-6)
    # only the first NB of the padded appearance batch is kept, in order
    assert np.allclose(rows[:NB, F:], np.arange(8 * 64, dtype=np.float32).reshape(8, 64)[:NB] + 1)


def test_appearance_batch_is_bucketed_not_per_count():
    """One runner per bucket, padded up: 5 crops -> batch-8 once, 1 -> batch-1, 0 -> no call."""
    zones, packed = _scene()
    det, apps, _ = _stubs()
    P3.frame_tokens(det, apps, zones, packed, P3.BRT.CONF, NMAX, CROP)
    assert [b for b, a in apps.items() if a.calls] == [8]
    assert apps[8].calls[0]['crops'] == (8, 3, CROP, CROP)

    def _one(**kw):
        raw = np.zeros((1, 300, 6), np.float32)
        raw[0, 0] = [200, 300, 260, 420, 0.9, 0]
        return raw
    det1, apps1, _ = _stubs(det_out=_one)
    P3.frame_tokens(det1, apps1, zones, packed, P3.BRT.CONF, NMAX, CROP)
    assert [b for b, a in apps1.items() if a.calls] == [1]

    det0, apps0, _ = _stubs(det_out=lambda **kw: np.zeros((1, 300, 6), np.float32))
    rows, mask = P3.frame_tokens(det0, apps0, zones, packed, P3.BRT.CONF, NMAX, CROP)
    assert rows.sum() == 0 and mask.sum() == 0
    assert not any(a.calls for a in apps0.values()), 'empty frame must not touch appearance'


def test_zones_for_clip_returns_shapely_not_raw_points():
    clips = _clips()
    zones, view = P3.zones_for_clip(clips[0])
    assert view is not None, 'the 691-clip view map must cover a test clip'
    if zones is None:
        # views 2 and 7 have no polygon file; zero geometry is correct, but then this clip cannot
        # exercise the polygon path at all — a skip, not a pass
        raise unittest.SkipTest(f'view {view} has no zone polygons')
    for kind in ('walkway', 'restricted'):
        for pg in zones[kind]:
            assert hasattr(pg, 'area') and hasattr(pg, 'centroid'), f'{kind} is raw points'
    # the same view returns the SAME object, so geometry()'s per-view precompute is built once
    assert P3.zones_for_clip(clips[0])[0] is zones
    assert P3.frame_tokens(*_stubs()[:2], zones, _scene()[1],
                           P3.BRT.CONF, NMAX, CROP)[1].sum() == NB


def test_run_config_emits_on_processed_frames_and_accounts_drops():
    clips = _clips(2)[:2]
    det, apps, head = _stubs()
    r = asyncio.run(MS.run_config(2, clips, 40, 24.0, det, apps, head, CFG))

    assert r['n_streams'] == 2 and r['t_window'] == T and r['imgsz'] == IMGSZ
    assert r['detector_stage'] == 'stub-detector'
    tot_p = sum(s['processed'] for s in r['streams'])
    assert tot_p + sum(s['dropped'] for s in r['streams']) == sum(s['frames'] for s in r['streams'])
    assert len(det.calls) == tot_p, 'the detector must run exactly once per processed frame'
    for s in r['streams']:
        # a decision fires only once the T-window is full, then every EMIT_EVERY PROCESSED frames
        expect = [d for d in range(T, s['processed'] + 1) if d % MS.SPU.EMIT_EVERY == 0]
        assert s['n_decisions'] == len(expect), (s['processed'], s['n_decisions'], expect)
        assert s['frame_latency_ms']['p50'] > 0 and s['decisions_per_sec'] >= 0
    assert isinstance(r['meets_cadence_all_streams'], bool)
    assert r['worst_stream_decisions_per_sec'] == min(s['decisions_per_sec'] for s in r['streams'])


def test_nonfinite_head_output_aborts_the_run():
    """An fp16 overflow must fail the run, not be reported as a throughput result — and it must
    stop the readers instead of letting them stream to the end."""
    clips = _clips()
    det, apps, head = _stubs(head_out=lambda **kw: np.array([[np.nan, 1.0]], np.float32))
    try:
        asyncio.run(MS.run_config(1, clips[:1], 400, 60.0, det, apps, head, CFG))
        raise AssertionError('non-finite logits did not raise')
    except RuntimeError as e:
        assert 'non-finite' in str(e)
    assert det.calls and len(det.calls) < 60, f'readers not stopped: {len(det.calls)} frames'


if __name__ == '__main__':
    import traceback
    tally = {'PASS': 0, 'SKIP': 0, 'FAIL': 0}
    for name in sorted(n for n in dir() if n.startswith('test_')):
        try:
            globals()[name]()
            tally['PASS'] += 1
            print(f'  PASS  {name}')
        except unittest.SkipTest as e:
            tally['SKIP'] += 1
            print(f'  SKIP  {name}: {e}')
        except Exception:
            tally['FAIL'] += 1
            print(f'  FAIL  {name}')
            traceback.print_exc()
    # Skips get their own column: "5 PASS" and "2 PASS / 3 SKIP" must never read alike.
    print(' / '.join(f'{v} {k}' for k, v in tally.items()))
    sys.exit(1 if tally['FAIL'] else 0)
