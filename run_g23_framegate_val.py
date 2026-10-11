#!/usr/bin/env python3
"""G23 on val only: does [17]'s frame-reduction step reconstruct their +10 pp?  No test touch.

[17] attribute +10.03 pp macro (85.78 -> 95.81) to fusing the behaviour detector into the ConvLSTM
path, with a frame-reduction step ("keep only frames carrying unsafe evidence") as the named
mechanism.  That claim decomposes into two separable questions, and only the second needs the test
set:
  (a) MECHANISM - does frame reduction add anything over plain aggregation of the same detector
      posteriors?  Answerable entirely on the 83-clip val slice shared by every 4-class member
      (`det-gateQ` vs `det-mean`/`det-max`/`det-pair`), so it costs nothing from the 1 remaining
      pre-registered test touch.
  (b) MAGNITUDE - does the fused system reach 95.81% on the 64 test clips?  Needs the test set.
This script answers (a) and reports the val-selected fusion weights for (b) WITHOUT scoring test.

The detectors behind every `det*` member are BELOW gate G22's 0.85 mAP50 floor (beh_ctx 0.806,
beh_tight 0.791), so every number here is exploratory by construction - see `gate_status` stamped
into each artifact by `unsafenet.behaviour_clip.detector_gate_status`.
"""
from __future__ import annotations
import json
import os

import numpy as np
import torch

from unsafenet import config as C, fusion as F, metrics_stats as S

PLAN = os.path.join(C.ROOT, 'plans', '261011-1502-baseline-giao-thuc-bai-bao-goc-unsafe-net')
GATE_MODES = ('gate25', 'gate50', 'gate75')
PLAIN_MODES = ('mean', 'max', 'pair')
DETECTORS = {'ctx': 'detctx', 'ctx_sep06': 'detC', 'tight_sep06': 'detT'}


def load(name):
    d = torch.load(os.path.join(C.OUT_DIR, f'temporal_{name}.pt'), weights_only=False)
    d['_name'] = name
    return d


def val_row(d):
    pv = np.asarray(d['Pv']).argmax(1)
    return {'val_macro': d['val_macro'], 'val_top1': d['val_top1'],
            'map50': d.get('detector_map50_peak'), 'gate_status': d.get('gate_status'), '_pred': pv}


def mechanism(prefix):
    """Best frame-reduction member vs best plain-aggregation member of the SAME detector."""
    rows, yv = {}, None
    for mode in PLAIN_MODES + GATE_MODES:
        try:
            d = load(f'{prefix}-{mode}')
        except FileNotFoundError:
            continue
        yv = np.asarray(d['yv'])
        rows[mode] = val_row(d)
    if not rows:
        return None
    pick = lambda ms: max((m for m in ms if m in rows), key=lambda m: rows[m]['val_macro'])
    g, p = pick(GATE_MODES), pick(PLAIN_MODES)
    delta = (rows[g]['val_macro'] - rows[p]['val_macro']) * 100
    return {
        'members': {m: {k: v for k, v in r.items() if k != '_pred'} for m, r in rows.items()},
        'best_gate': g, 'best_plain': p,
        'delta_val_macro_pp_gate_minus_plain': delta,
        'mcnemar_val_gate_vs_plain': S.mcnemar_exact(yv, rows[g]['_pred'], rows[p]['_pred']),
        'paper_attributes_pp': 100 * (C.PAPER['full']['macro'] - C.PAPER['nobbox']['macro']),
    }


def fusion_val(pattern, rounds=12):
    """Caruana forward selection on val only.  Saves test posteriors; does NOT score them."""
    res = F.run(pattern=pattern, rounds=rounds, save=f'fusion_g23_{pattern.strip("*")}' or 'fusion_g23')
    return {'pattern': pattern, 'picked_val_selected': res['picked'],
            'val_macro': res['val_macro'], 'val_top1': res['val_top1'],
            'n_members_offered': len(res['members']),
            'test_scored': False,
            'note': 'test posteriors saved, deliberately NOT scored: that needs the 1 remaining '
                    'pre-registered test touch and a decision rule written before it is spent.'}


def fusion_members(names, rounds=12, save='fusion_g23_paper'):
    """Caruana on an EXPLICIT member list - [17] fuse the ConvLSTM path WITH the detector path, a
    union that no single glob expresses (`fusion.load_members` takes one pattern)."""
    ms = [load(n) for n in names]
    counts, vm, vt = F.bagged_select(ms, rounds)
    res = {'picked': {m['name']: round(float(c), 3) for m, c in zip(ms, counts) if c > 1e-6},
           'val_macro': vm, 'val_top1': vt,
           'Pv': F.apply_weights(ms, counts, 'Pv'), 'yv': ms[0]['yv'],
           'Pt': F.apply_weights(ms, counts, 'Pt'), 'yt': ms[0]['yt'],
           'test_names': ms[0]['test_names'], 'members': names}
    torch.save(res, os.path.join(C.OUT_DIR, f'{save}.pt'))
    print(f'[{save}] picked (val-selected):', res['picked'])
    print(f'[{save}] val macro={vm*100:.1f} top1={vt*100:.1f}')
    return {'members_offered': names, 'picked_val_selected': res['picked'],
            'val_macro': vm, 'val_top1': vt, 'test_scored': False}


def main():
    out = {'gate': 'G23 (mechanism only, val slice, no test touch)',
           'preregistration': 'tien-dang-ky-g21-g25.md',
           'detector_gate_floor_G22': 0.85,
           'baseline_raw_frame_convlstm_val_macro': {
               n: load(n)['val_macro'] for n in ('convlstm', 'convlstm_cpu')}}
    out['mechanism'] = {k: m for k, v in DETECTORS.items() if (m := mechanism(v))}
    # Paper-faithful fusion set: raw-frame ConvLSTM + the behaviour-detector members, nothing else.
    # `*` would sweep in 40+ appearance/temporal members that [17] never had.
    out['fusion_val_only'] = {p: fusion_val(p) for p in ('detctx-*', 'detT-*')}
    # The actual [17] architecture: raw-frame ConvLSTM path UNION behaviour-detector path.
    for tag, pre in (('convlstm_cpu+detT', 'detT'), ('convlstm_cpu+detctx', 'detctx')):
        names = ['convlstm_cpu'] + [f'{pre}-{m}' for m in PLAIN_MODES + GATE_MODES]
        out['fusion_val_only'][tag] = fusion_members(names, save=f'fusion_g23_{tag}')
    f = os.path.join(PLAN, 'ket-qua-g23-val.json')
    json.dump(out, open(f, 'w'), indent=2)
    print(f'\n-> {os.path.relpath(f, C.ROOT)}')
    for k, m in out['mechanism'].items():
        print(f"[{k:12s}] best gate {m['best_gate']} {m['members'][m['best_gate']]['val_macro']*100:.1f}"
              f"  vs best plain {m['best_plain']} {m['members'][m['best_plain']]['val_macro']*100:.1f}"
              f"  -> {m['delta_val_macro_pp_gate_minus_plain']:+.2f} pp"
              f"  (McNemar p={m['mcnemar_val_gate_vs_plain']['p']:.4f},"
              f" paper attributes {m['paper_attributes_pp']:+.2f} pp)")


if __name__ == '__main__':
    main()
