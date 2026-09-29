"""Unit tests for the binary Safe/Unsafe encoder family (model_binary_encoders.py,
model_relation_head.py) — design-2026-09-16-binary-encoder-redesign.md §2.

Covers: forward-shape contract for E0/E1/E2/E3 under every (return_sub, return_frame,
return_repr) combination, all-padding frame/clip edge cases (no NaN), and permutation
invariance of E1's intra-frame stage (design's core structural claim — no positional
embedding on entity slots).

No pytest in the project venv (see tests/test_dedupe_binary.py precedent) — plain assert
functions named test_*, runnable standalone via `if __name__ == '__main__'` below.

Usage: .venv/bin/python tests/test_binary_encoder_forward.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_binary_encoders import build_encoder, ARCHES

N_NUM = 73     # feat=74 (yolo26app_ / bin8_ appearance layout)
B, T, N = 3, 16, 8


def _rand_batch(all_pad_clip=False, all_pad_frame=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    tok = torch.randn(B, T, N, N_NUM + 1, generator=g)
    tok[..., 0] = torch.randint(0, 8, (B, T, N), generator=g).float()   # class id in [0, n_entity)
    mask = (torch.rand(B, T, N, generator=g) > 0.3).float()
    mask[:, :, 0] = 1.0                                                  # >=1 real box per frame normally
    if all_pad_frame:
        mask[0, 0] = 0.0                                                 # clip 0, frame 0 fully padded
    if all_pad_clip:
        mask[1] = 0.0                                                    # clip 1 fully padded (every frame)
    return tok, mask


def test_all_archs_build_with_expected_param_order_of_magnitude():
    for arch in ARCHES:
        m = build_encoder(arch, n_num=N_NUM, d=64 if arch == 'E1' else None)
        n_params = sum(p.numel() for p in m.parameters())
        assert 1000 < n_params < 400_000, (arch, n_params)


def test_forward_default_shape_all_archs():
    tok, mask = _rand_batch()
    for arch in ARCHES:
        m = build_encoder(arch, n_num=N_NUM, d=64 if arch == 'E1' else None)
        m.eval()
        out = m(tok, mask)
        assert isinstance(out, torch.Tensor), (arch, type(out))
        assert out.shape == (B, 2), (arch, out.shape)
        assert torch.isfinite(out).all(), arch


def test_forward_return_sub_shape_all_archs():
    tok, mask = _rand_batch()
    for arch in ARCHES:
        m = build_encoder(arch, n_num=N_NUM, n_sub=8, d=64 if arch == 'E1' else None)
        m.eval()
        cls_logits, sub_logits = m(tok, mask, return_sub=True)
        assert cls_logits.shape == (B, 2), (arch, cls_logits.shape)
        assert sub_logits.shape == (B, 8), (arch, sub_logits.shape)
        assert torch.isfinite(sub_logits).all(), arch


def test_forward_return_frame_shape_all_archs():
    tok, mask = _rand_batch()
    for arch in ARCHES:
        m = build_encoder(arch, n_num=N_NUM, d=64 if arch == 'E1' else None)
        m.eval()
        cls_logits, frame_logits, frame_valid = m(tok, mask, return_frame=True)
        assert cls_logits.shape == (B, 2), arch
        assert frame_logits.shape == (B, T, 2), (arch, frame_logits.shape)
        assert frame_valid.shape == (B, T), (arch, frame_valid.shape)
        assert frame_valid.dtype == torch.bool, arch
        assert torch.isfinite(frame_logits).all(), arch


def test_forward_all_flags_combined_all_archs():
    """The (return_sub=True, return_frame=True, return_repr=True) combo exercised by
    run_epoch(binary=True) in run_relation_transformer.py."""
    tok, mask = _rand_batch()
    for arch in ARCHES:
        m = build_encoder(arch, n_num=N_NUM, n_sub=8, d=64 if arch == 'E1' else None)
        m.eval()
        out = m(tok, mask, return_frame=True, return_sub=True, return_repr=True)
        assert len(out) == 5, (arch, len(out))
        cls_logits, sub_logits, frame_logits, frame_valid, cls_repr = out
        assert cls_logits.shape == (B, 2) and sub_logits.shape == (B, 8)
        assert frame_logits.shape == (B, T, 2) and frame_valid.shape == (B, T)
        assert cls_repr.dim() == 2 and cls_repr.shape[0] == B, (arch, cls_repr.shape)


def test_all_padding_frame_does_not_nan_all_archs():
    """design §2 risk: an intra-frame stage with a fully-masked key row can NaN on attention
    softmax. clip 0 / frame 0 is fully padded in this batch."""
    tok, mask = _rand_batch(all_pad_frame=True)
    for arch in ARCHES:
        m = build_encoder(arch, n_num=N_NUM, n_sub=8, d=64 if arch == 'E1' else None)
        m.eval()
        out = m(tok, mask, return_frame=True, return_sub=True)
        for t in out:
            assert torch.isfinite(t).all(), (arch, 'NaN/Inf with all-padded frame')


def test_all_padding_clip_does_not_nan_all_archs():
    """clip 1 has every frame fully padded (a degenerate/corrupt clip) — the CLS token / temporal
    stage must still produce finite logits (E0/E2 always attend CLS; E1/E3 use empty_frame /
    masked-mean-of-nothing -> 0)."""
    tok, mask = _rand_batch(all_pad_clip=True)
    for arch in ARCHES:
        m = build_encoder(arch, n_num=N_NUM, n_sub=8, d=64 if arch == 'E1' else None)
        m.eval()
        out = m(tok, mask, return_frame=True, return_sub=True)
        for t in out:
            assert torch.isfinite(t).all(), (arch, 'NaN/Inf with all-padded clip')


def test_e1_intra_frame_permutation_invariance():
    """Core structural claim of E1 (design §2): the intra-frame stage has NO positional
    embedding on entity slots, so permuting the N_MAX box slots within every frame must not
    change the model's output (up to float tolerance)."""
    torch.manual_seed(0)
    m = build_encoder('E1', n_num=N_NUM, n_sub=8, d=64)
    m.eval()
    tok, mask = _rand_batch()
    perm = torch.randperm(N)
    tok_p = tok[:, :, perm, :]
    mask_p = mask[:, :, perm]
    with torch.no_grad():
        out_a = m(tok, mask, return_sub=True, return_frame=True)
        out_b = m(tok_p, mask_p, return_sub=True, return_frame=True)
    cls_a, sub_a, fr_a, fv_a = out_a
    cls_b, sub_b, fr_b, fv_b = out_b
    assert torch.allclose(cls_a, cls_b, atol=1e-4), (cls_a - cls_b).abs().max()
    assert torch.allclose(sub_a, sub_b, atol=1e-4), (sub_a - sub_b).abs().max()
    assert torch.allclose(fr_a, fr_b, atol=1e-4), (fr_a - fr_b).abs().max()
    assert torch.equal(fv_a, fv_b)


def test_e0_backward_compat_default_call_matches_pre_binary_contract():
    """model_relation_head.RelationTransformer with n_sub=None (the untouched legacy default)
    must still return a bare tensor on a plain call, and a 3-tuple with return_frame=True —
    the exact shapes every existing 4-class call site relies on."""
    from model_relation_head import RelationTransformer
    tok, mask = _rand_batch()
    m = RelationTransformer(n_num=N_NUM)   # n_cls=4 default, n_sub=None default
    m.eval()
    out = m(tok, mask)
    assert isinstance(out, torch.Tensor) and out.shape == (B, 4)
    out2 = m(tok, mask, return_frame=True)
    assert len(out2) == 3
    cls_logits, frame_logits, frame_valid = out2
    assert cls_logits.shape == (B, 4) and frame_logits.shape == (B, T, 4) and frame_valid.shape == (B, T)
    out3 = m(tok, mask, return_repr=True)
    assert len(out3) == 2 and out3[1].shape == (B, 128)   # default d=128


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
            print(f"  PASS  {name}")
        except Exception:
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
    sys.exit(1 if failures else 0)
