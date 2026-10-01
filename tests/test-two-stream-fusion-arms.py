"""TwoStreamFusion must reuse the G14 arms exactly, not a copy that can drift from them.

Two properties are worth a test here:

  1. The `encode`/`head` split I introduced into EntitySpatioTemporal and WholeFrameTemporal is
     behaviour-preserving. Those two modules produced every Block 1b number; if the refactor moved
     them even slightly, the G14 verdict and the G18 baseline stop being comparable.
  2. TSF's auxiliary heads are the *same function* as the standalone arms. The whole G18 claim is
     "entity stream + whole-frame stream, fused". If the aux path diverged from the arm it is named
     after, a PASS would not mean what the gate says it means.

Run: .venv/bin/python tests/test-two-stream-fusion-arms.py
"""
import os
import sys
import warnings

import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_entity_spatiotemporal import (EntitySpatioTemporal, TwoStreamFusion,
                                         WholeFrameTemporal, count_params)

B, T, N = 3, 16, 8


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, T, N, 512, generator=g),
            torch.rand(B, T, N, 6, generator=g),
            (torch.rand(B, T, N, generator=g) > 0.3).float(),
            torch.randn(B, T, 512, generator=g),
            (torch.rand(B, T, generator=g) > 0.2).float())


def test_encode_head_split_is_identity():
    """forward() must still be exactly head(encode()) for both refactored arms."""
    app, geom, mask, fr, valid = batch()
    torch.manual_seed(0); e = EntitySpatioTemporal().eval()
    torch.manual_seed(0); w = WholeFrameTemporal().eval()
    with torch.no_grad():
        assert torch.equal(e(app, geom, mask), e.head(e.encode(app, geom, mask)))
        assert torch.equal(w(fr, valid), w.head(w.encode(fr, valid)))
    print('  PASS encode/head split is bit-exact on both arms')


def test_aux_heads_are_the_standalone_arms():
    """TSF's aux logits must equal what the arm would emit on its own."""
    app, geom, mask, fr, valid = batch(1)
    torch.manual_seed(3); m = TwoStreamFusion().eval()
    with torch.no_grad():
        fused, aux_e, aux_f = m(app, geom, mask, fr, valid)
        assert torch.equal(aux_e, m.entity(app, geom, mask)), 'entity aux != entity arm'
        assert torch.equal(aux_f, m.frame(fr, valid)), 'frame aux != frame arm'
    assert fused.shape == (B, 2) and aux_e.shape == (B, 2) and aux_f.shape == (B, 2)
    print('  PASS aux heads are the standalone arms')


def test_both_streams_reach_the_fused_output():
    """Perturbing either stream alone must move the fused logits.

    This is the collapse guard: if the fusion block learned to ignore an arm the gradient check
    below would still pass, but a *dead wire* -- an arm never connected to the output -- would not,
    and that is a bug the accuracy number could not distinguish from a negative result.
    """
    app, geom, mask, fr, valid = batch(2)
    torch.manual_seed(4); m = TwoStreamFusion().eval()
    with torch.no_grad():
        base = m(app, geom, mask, fr, valid)[0]
        ent = m(app * 0, geom, mask, fr, valid)[0]
        frm = m(app, geom, mask, fr * 0, valid)[0]
    assert not torch.allclose(base, ent), 'entity stream does not reach the fused output'
    assert not torch.allclose(base, frm), 'whole-frame stream does not reach the fused output'
    print('  PASS both streams reach the fused output')


def test_gradients_flow_to_both_arms():
    app, geom, mask, fr, valid = batch(5)
    torch.manual_seed(6); m = TwoStreamFusion()
    m(app, geom, mask, fr, valid)[0].sum().backward()
    ge = m.entity.tok.app.weight.grad
    gf = m.frame.proj[0].weight.grad
    assert ge is not None and ge.abs().sum() > 0, 'no gradient into the entity arm'
    assert gf is not None and gf.abs().sum() > 0, 'no gradient into the whole-frame arm'
    print('  PASS gradients reach both arms from the fused head alone')


def test_degenerate_clips_stay_finite():
    """Empty detections and dead frames are facts about this corpus, not bugs -- never NaN."""
    app, geom, mask, fr, valid = batch(7)
    torch.manual_seed(8); m = TwoStreamFusion().eval()
    with torch.no_grad():
        for mk, v in ((mask * 0, valid), (mask, valid * 0), (mask * 0, valid * 0)):
            out = torch.cat(m(app, geom, mk, fr, v), 1)
            assert torch.isfinite(out).all(), 'non-finite logits on a degenerate clip'
    print('  PASS degenerate clips stay finite')


def test_arm_param_counts_unchanged():
    """The G14 arms must still be the sizes Block 1b reports."""
    assert count_params(EntitySpatioTemporal()) == 352258
    assert count_params(WholeFrameTemporal()) == 201090
    assert count_params(WholeFrameTemporal(t_layers=2)) == 333570
    print(f'  PASS arm param counts unchanged (TSF itself = {count_params(TwoStreamFusion())})')


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for f in fns:
        f()
    print(f'\n{len(fns)}/{len(fns)} pass')
