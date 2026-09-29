"""Channel ablations for the relation-token tensors (n,T,N,74).

Why a module and not three inline `X[..., i] = 0` lines in run_kfold_relation.py: every ablation has
to be applied to BOTH the encoder input and the R42 count-rule floor (otherwise the encoder is denied
a cue the shortcut still gets, and the reported margin is meaningless), and both call sites sit in
different functions. One place to define them, one place to test them.

Channel indices come from the token file's own `feat_layout`, never hardcoded, because
build_relation_tokens.py and build_isafetybench_tokens.py could reorder FEAT independently.

Ablations are applied to the RAW tokens, before run_relation_transformer.apply_coord_variant, since
T2 deletes the xc/yc columns and shifts every index after them.
"""
import numpy as np
import torch

FEAT_LAYOUT = ['class', 'xc', 'yc', 'w', 'h', 'conf', 'iou_walkway', 'iou_restricted',
               'foot_in_walkway', 'norm_dist']
N_FEAT = len(FEAT_LAYOUT)

BOX_COLS = ('xc', 'yc', 'w', 'h')
MODES = ('none', 'no_conf', 'no_appearance', 'blind_class', 'boxes_only', 'no_boxes', 'appearance_only')


def describe(mode):
    return {
        'none': 'full tokens (default recipe)',
        'no_conf': 'detection-confidence channel zeroed (isolates the mean_conf image-quality cue)',
        'no_appearance': 'the 64 appearance columns zeroed (geometry + class ids only)',
        'blind_class': 'entity class id held constant on every valid slot (identity-blind)',
        'boxes_only': 'conf + appearance + class ALL neutralised -- only box xc/yc/w/h over time '
                      'survives (the three single ablations were each within noise, so this is the '
                      'one that can actually falsify the geometry reading)',
        'no_boxes': 'box xc/yc/w/h zeroed, everything else intact -- the converse of boxes_only',
        'appearance_only': 'every one of the 10 FEAT columns zeroed -- only the 64 appearance dims '
                           'survive; the run that decides whether this is a relation model or an '
                           'appearance classifier over detected crops',
    }[mode]


def _check_layout(feat_layout):
    """feat_layout may be absent (older token files) -> fall back to the module constant. When it IS
    present the first N_FEAT entries must match exactly; a silent reorder would make every ablation
    zero the wrong column and the ablation study would be fiction."""
    if feat_layout is None:
        return
    head = [str(f) for f in list(feat_layout)[:N_FEAT]]
    if head != FEAT_LAYOUT:
        raise ValueError(f'token feat_layout {head} != expected {FEAT_LAYOUT}; '
                         f'ablation column indices are no longer valid')


def apply(X, M, mode, feat_layout=None):
    """-> a COPY of X with `mode`'s channels neutralised on valid slots. X untouched.

    Every ablation writes the constant 0, which padding slots already hold, so the
    padding-is-all-zero invariant is preserved and no "this slot is padding" signal leaks into the
    ablated channel. M is unused here but kept in the signature so callers cannot forget that the
    mask must travel with the ablated tensor."""
    if mode in (None, 'none'):
        return X
    if mode not in MODES:
        raise ValueError(f'unknown ablation {mode!r}; expected one of {MODES}')
    _check_layout(feat_layout)
    out = X.clone() if torch.is_tensor(X) else np.array(X, copy=True)
    if mode == 'boxes_only':
        for m in ('no_conf', 'no_appearance', 'blind_class'):
            out = apply(out, M, m, feat_layout)
        return out
    if mode == 'appearance_only':
        out[..., :N_FEAT] = out[..., :N_FEAT] * 0
        return out
    if mode == 'no_boxes':
        for name in BOX_COLS:
            col = FEAT_LAYOUT.index(name)
            out[..., col] = out[..., col] * 0
        return out
    if mode == 'no_appearance':
        out[..., N_FEAT:] = out[..., N_FEAT:] * 0
    else:
        # 'conf' -> the mean_conf image-quality cue; 'class' -> the entity identity channel. Both are
        # neutralised to the constant 0, which is also what padding slots already hold, so the
        # padding-is-all-zero invariant survives without a masked write.
        col = FEAT_LAYOUT.index('conf' if mode == 'no_conf' else 'class')
        out[..., col] = out[..., col] * 0
    return out


def changed_fraction(X, Xa, M):
    """Guard for the callers: fraction of valid slots whose feature vector actually moved. An
    ablation that changes nothing (empty corpus, already-zero channel) must be reported, not treated
    as a null result."""
    d = (X - Xa).abs().sum(-1) if torch.is_tensor(X) else np.abs(X - Xa).sum(-1)
    valid = (M > 0)
    n = float(valid.sum())
    if n == 0:
        return 0.0
    return float(((d > 1e-8) & valid).sum()) / n
