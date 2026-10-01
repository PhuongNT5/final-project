"""The demo must run the SAME decision path that was evaluated and timed, not a lookalike.

Framing (confirmed 2026-10-01): B1 classifies from whole-frame CLIP features, YOLOv11s localises.
Four ways that can silently break, each pinned below:

  1. The demo calls the model directly as `m(wholeframe, valid)` rather than through
     `esta_training.parts_for`. If that argument order ever stops matching what training and LSO-1100
     evaluation used, the demo would report confident nonsense and look fine doing it.
  2. A frame that fails to decode must be marked invalid, not merely left as a zero vector. A zero
     vector that counts as content feeds the temporal head a frame the video never had.
  3. The detector must not touch the class decision. `detector=None` has to be a complete path, so
     the demo can demonstrate the verdict surviving localisation being switched off.
  4. An entity-arm checkpoint (ESTA/TSF) must be refused, not served with zeros for the crops and
     geometry it expects.

Run: .venv/bin/python -m pytest tests/test-demo-b1-pipeline.py -q
"""
import importlib.util
import os
import sys
import warnings

import numpy as np
import pytest
import torch

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import esta_training as ET  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


PIPE = _load('b1_window_pipeline', 'demo/b1-window-pipeline.py')
RUN = _load('run_unsafe_behavior_demo', 'demo/run-unsafe-behavior-demo.py')
T, D = PIPE.B.T, 512


class StubTower:
    """Stands in for the CLIP tower only -- the tower is not what these tests are about.

    Returns a distinct, frame-identifying embedding so a misplaced frame is visible in the output
    rather than hidden by every row being the same.
    """
    CLIP_DIM = D

    def embed(self, frames, clip):
        return np.stack([np.full(D, float(f), dtype=np.float32) for f in frames])


def pipe(detector=None, n_models=3):
    """A pipeline with real B1 models and real feature plumbing, bypassing the CLIP download."""
    p = object.__new__(PIPE.B1WindowPipeline)
    p.name, p.bld, p.clip, p.det = 'B1', StubTower(), None, detector
    p.imgsz, p.conf, p.device = PIPE.B.IMGSZ, PIPE.B.CONF, 'cpu'
    p.models = []
    for s in range(n_models):
        torch.manual_seed(s)
        m = ET.build('B1', D); m.eval()
        p.models.append(m)
    return p


def test_demo_argument_order_matches_the_evaluated_path():
    wf = torch.randn(2, T, D)
    parts = ET.parts_for('B1', {'wholeframe': wf}, np.array([0, 1]))
    assert len(parts) == 2, 'B1 takes exactly (frames, validity); the demo passes two arguments'
    assert torch.equal(parts[0], wf), 'frames come first'
    assert parts[1].shape == (2, T), 'validity is one flag per frame, second'


def test_undecoded_frames_are_masked_not_zero_padded():
    frames = [1.0, None, 3.0] + [None] * (T - 3)
    pack = pipe().features(frames)
    valid = pack['valid'][0].numpy()
    assert valid[0] == 1.0 and valid[2] == 1.0
    assert valid[1] == 0.0 and valid[3:].sum() == 0.0
    wf = pack['wholeframe'][0].numpy()
    assert wf[0, 0] == 1.0 and wf[2, 0] == 3.0, 'embeddings land on their own frame index'
    assert np.all(wf[1] == 0.0), 'an undecoded frame carries no content'


def test_features_shapes_are_the_evaluated_tensor():
    pack = pipe().features([float(i) for i in range(T)])
    assert pack['wholeframe'].shape == (1, T, D)
    assert pack['valid'].shape == (1, T)
    assert pack['valid'].sum() == T


def test_localisation_off_is_a_complete_path():
    p = pipe(detector=None)
    boxes = p.localise([np.zeros((8, 8, 3), np.uint8)] * T, 8, 8)
    assert len(boxes) == T and all(b == [] for b in boxes)
    prob, seeds = p.unsafe_prob(p.features([float(i) for i in range(T)]))
    assert 0.0 <= prob <= 1.0 and len(seeds) == 3


def test_verdict_is_the_mean_over_seeds():
    p = pipe()
    pack = p.features([float(i) for i in range(T)])
    prob, seeds = p.unsafe_prob(pack)
    with torch.no_grad():
        want = [float(torch.softmax(m(pack['wholeframe'], pack['valid']), 1)[0, 1])
                for m in p.models]
    assert seeds == pytest.approx(want, abs=1e-6)
    assert prob == pytest.approx(sum(want) / len(want), abs=1e-6)


def test_entity_arm_checkpoint_is_refused():
    for name in ('ESTA', 'TSF'):
        with pytest.raises(SystemExit, match='entity'):
            PIPE.require_whole_frame(name, 'sweep_results/relation_tokens/deploy/x.pt')
    assert PIPE.require_whole_frame('B1', 'x.pt') == 'B1'
    assert PIPE.require_whole_frame('B1w', 'x.pt') == 'B1w'


def test_threshold_is_untuned_half():
    assert RUN.THRESHOLD == 0.5, 'no threshold was ever selected on this corpus'


def test_window_tail_is_never_dropped():
    starts = RUN.windows(100, 48, 24)
    assert starts[0] == 0
    assert starts[-1] + 48 >= 100, 'the final frames must fall inside some window'
    assert RUN.windows(10, 48, 24) == [0], 'a video shorter than one window still yields a window'


def test_localisation_summary_counts_without_entering_the_decision():
    boxes = [[(0, 0, 1, 1, 'person', 0.9), (0, 0, 1, 1, 'truck', 0.5)], [(0, 0, 1, 1, 'person', 0.8)]]
    loc = RUN.localisation_summary(boxes)
    assert loc['detections'] == 3
    assert loc['classes']['person'] == 2 and loc['classes']['truck'] == 1
