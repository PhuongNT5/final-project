"""Guards on the ESTA lane (R48 / G14-G16).

What matters here is not that the models run — the smoke test showed that — but that the things
which would silently invalidate the gate cannot happen quietly:
  1. the CLIP features are permuted onto the relation-encoder's clip order, and any id or label
     disagreement aborts instead of scoring a differently-ordered corpus;
  2. B1 sees exactly ESTA's frame validity, so the two differ only by the entity axis;
  3. a frame with no detections does not produce NaNs in either model.

Run: .venv/bin/python tests/test-esta-alignment-and-guards.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util

import esta_training as T
import isafetybench_dataset as ISB
from model_entity_spatiotemporal import CLIP_DIM


def _driver():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run-esta-lso.py')
    spec = importlib.util.spec_from_file_location('run_esta_lso', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _feats_file(clip_ids, labels, t=16, n=8):
    """A feature file in the real on-disk shape, carrying the real corpus's ids and labels."""
    k = len(clip_ids)
    d = {'entity': torch.zeros(k, t, n, CLIP_DIM), 'geom': torch.zeros(k, t, n, 6),
         'mask': torch.zeros(k, t, n), 'wholeframe': torch.zeros(k, t, CLIP_DIM),
         'labels': torch.as_tensor(labels).long(), 'clip_ids': np.array(clip_ids)}
    fh = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
    torch.save(d, fh.name)
    return fh.name


def test_alignment_permutes_onto_the_encoder_clip_order():
    drv = _driver()
    _, _, y, _, ids = ISB.load('isb_')
    order = np.random.default_rng(0).permutation(len(ids))      # a deliberately shuffled build
    path = _feats_file([ids[i] for i in order], [int(y[i]) for i in order])
    feats, y2, groups, ids2 = drv.load_aligned(path)
    assert list(ids2) == list(ids), 'clip order not restored to the encoder order'
    assert torch.equal(y2, y), 'labels not restored'
    assert feats['entity'].shape[0] == len(ids)
    os.unlink(path)
    print('  ok: shuffled feature file is permuted back onto the relation-encoder clip order')


def test_alignment_refuses_wrong_labels_and_missing_clips():
    drv = _driver()
    _, _, y, _, ids = ISB.load('isb_')
    bad = [int(v) for v in y]; bad[7] = 1 - bad[7]
    path = _feats_file(list(ids), bad)
    try:
        drv.load_aligned(path); raise AssertionError('accepted a label mismatch')
    except SystemExit as e:
        assert 'label mismatch' in str(e)
    os.unlink(path)

    path = _feats_file(list(ids[:-5]), [int(v) for v in y[:-5]])
    try:
        drv.load_aligned(path); raise AssertionError('accepted a truncated corpus')
    except SystemExit as e:
        assert 'missing' in str(e)
    os.unlink(path)
    print('  ok: a flipped label and a truncated corpus both abort instead of scoring')


def test_b1_inherits_esta_frame_validity():
    m = torch.zeros(3, 16, 8)
    m[0, :4, :2] = 1.0; m[1, 7, 0] = 1.0                 # clip 2 has no detections anywhere
    feats = {'entity': torch.zeros(3, 16, 8, CLIP_DIM), 'geom': torch.zeros(3, 16, 8, 6),
             'mask': m, 'wholeframe': torch.zeros(3, 16, CLIP_DIM)}
    _, valid = T.parts_for('B1', feats, np.arange(3))
    assert torch.equal(valid, (m.sum(-1) > 0).float()), 'B1 validity diverges from ESTA mask'
    assert valid[2].sum() == 0, 'expected an all-empty clip in this fixture'
    print('  ok: B1 frame validity is ESTA mask.sum(-1) > 0, including the empty clip')


def test_no_nan_on_empty_frames():
    idx = np.arange(3)
    m = torch.zeros(3, 16, 8); m[0, :4, :2] = 1.0
    feats = {'entity': torch.randn(3, 16, 8, CLIP_DIM), 'geom': torch.rand(3, 16, 8, 6),
             'mask': m, 'wholeframe': torch.randn(3, 16, CLIP_DIM)}
    for name in ('ESTA', 'B1', 'B1w'):
        out = T.build(name)(*T.parts_for(name, feats, idx))
        assert out.shape == (3, 2) and torch.isfinite(out).all(), f'{name} produced non-finite logits'
    print('  ok: ESTA/B1/B1w stay finite on a clip with zero detections in every frame')


if __name__ == '__main__':
    for fn in (test_alignment_permutes_onto_the_encoder_clip_order,
               test_alignment_refuses_wrong_labels_and_missing_clips,
               test_b1_inherits_esta_frame_validity,
               test_no_nan_on_empty_frames):
        print(f'{fn.__name__}:'); fn()
    print('\nall esta-lane guards pass')
