#!/usr/bin/env python3
"""LOCAL gate between a Kaggle-built tower file and the sweep. Nothing reaches the sweep unchecked.

Three jobs, in this order
-------------------------
1. ORDER AND LABELS. The clip ids must be the published 1,100 carrying the published labels. A file
   that fails here is unusable: every post-hoc margin is paired per clip against the 83.25% control
   and `analyze_posthoc_paired_vs_control.py` would refuse it anyway, later and less clearly.

2. THE DECODE CONTROL. For `clipB32_control` -- the Kaggle re-extraction of the published backbone
   -- the whole-frame tensor is compared against feats_isbclip_all.pt element by element. This is
   the only evidence that decoding on another machine, with another OpenCV build, did not shift the
   input side. Agreement to float noise means every other tower file is a clean tower swap on
   identical pixels. Disagreement does not invalidate the files, but it does mean a tower margin
   mixes a tower effect with a decode effect, and must be reported that way.

3. BORROW mask AND geom FROM THE PUBLISHED FILE. This is not cosmetic. The head computes its
   temporal mask as `valid = (mask.sum(-1) > 0)`, and that mask is the YOLO detector's output, not
   decode validity: a frame where the detector found nothing is a padded frame. A tower file
   without `mask` would silently fall back to all-ones, giving the new tower 16 live frames where
   the control had fewer -- a second variable inside a one-variable experiment. The E2 ladder's
   --mask_from flag exists for exactly this reason and is reproduced here.

Usage: .venv/bin/python kaggle/verify-and-install-tower-features.py ~/Downloads/feats_clipB16_all.pt
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUB = os.path.join(ROOT, 'sweep_results/relation_tokens/feats_isbclip_all.pt')
DEST = os.path.join(ROOT, 'sweep_results/relation_tokens')


def check_order(d, pub):
    cids, pcids = [str(c) for c in d['clip_ids']], [str(c) for c in pub['clip_ids']]
    if cids != pcids:
        same = sum(a == b for a, b in zip(cids, pcids))
        sys.exit(f'[FAIL] clip order differs from the published file '
                 f'({same}/{len(pcids)} positions agree, {len(cids)} rows present). '
                 f'Re-run stage 1 against kaggle/isb-clip-manifest.json.')
    if not torch.equal(torch.as_tensor(d['labels']).long(),
                       torch.as_tensor(pub['labels']).long()):
        sys.exit('[FAIL] labels differ from the published file.')
    print(f'[ok] order and labels match the published 1,100 clips')


def control_report(wf, pub_wf):
    """Element-wise agreement of the Kaggle B/32 re-extraction with the published tensor."""
    diff = (wf - pub_wf).abs()
    denom = pub_wf.norm(dim=-1) * wf.norm(dim=-1)
    cos = torch.where(denom > 0, (wf * pub_wf).sum(-1) / denom.clamp(min=1e-12),
                      torch.ones_like(denom))
    print(f'[control] max|diff|={diff.max():.3e}  mean|diff|={diff.mean():.3e}')
    print(f'[control] per-frame cosine: min={cos.min():.6f} mean={cos.mean():.6f}')
    frames_off = int((cos < 0.9999).sum())
    print(f'[control] frames below cosine 0.9999: {frames_off}/{cos.numel()}')
    if diff.max() < 1e-3 and frames_off == 0:
        print('[control] VERDICT: the Kaggle decode path reproduces the published features. '
              'Tower margins from this cache are clean tower effects.')
    else:
        print('[control] VERDICT: the Kaggle decode path does NOT reproduce the published '
              'features. Tower margins from this cache mix a tower effect with a decode effect '
              'and must be reported as such. Do not present them as pure tower swaps.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', help='the feats_<tower>_all.pt downloaded from Kaggle')
    ap.add_argument('--tag', default=None, help='install as feats_<tag>_all.pt (default: as named)')
    ap.add_argument('--dry-run', action='store_true', help='verify only, install nothing')
    args = ap.parse_args()

    d = torch.load(args.src, weights_only=False)
    pub = torch.load(PUB, weights_only=False)
    wf = torch.as_tensor(d['wholeframe']).float()
    print(f'[src] {os.path.basename(args.src)} backbone={d.get("backbone")} '
          f'shape={tuple(wf.shape)} built_on={d.get("meta", {}).get("built_on")} '
          f'amp={d.get("meta", {}).get("amp_fp16")}')
    check_order(d, pub)
    assert wf.shape[:2] == pub['wholeframe'].shape[:2], (wf.shape, pub['wholeframe'].shape)

    if wf.shape[-1] == pub['wholeframe'].shape[-1] and 'patch32' in str(d.get('backbone', '')):
        control_report(wf, pub['wholeframe'].float())
    if d.get('meta', {}).get('amp_fp16'):
        print('[warn] built with fp16 autocast: this file carries a numeric drift source the '
              'published control does not have. Prefer re-extracting without --amp.')

    tag = args.tag or os.path.basename(args.src).replace('feats_', '').replace('_all.pt', '')
    out = f'{DEST}/feats_{tag}_all.pt'
    if args.dry_run:
        print(f'[dry-run] would install {out}')
        return
    if os.path.exists(out):
        sys.exit(f'[FAIL] {out} exists. Published artifacts are never overwritten; pick a new --tag.')
    # mask/geom borrowed so frame validity is byte-identical to the control (see module docstring).
    torch.save({'wholeframe': wf, 'mask': pub['mask'], 'geom': pub['geom'],
                'labels': torch.as_tensor(d['labels']).long(),
                'clip_ids': [str(c) for c in d['clip_ids']],
                'views': [str(v) for v in d['views']],
                'backbone': d.get('backbone'), 'dataset': 'isafetybench',
                'wholeframe_only': True,
                'meta': dict(d.get('meta', {}), mask_from=os.path.relpath(PUB, ROOT))}, out)
    print(f'[installed] {os.path.relpath(out, ROOT)}  (mask/geom borrowed from the B/32 file)')
    # --tag is concatenated raw onto the row name, so it carries its own trailing underscore.
    print(f'[next] OMP_NUM_THREADS=4 .venv/bin/python run_b1_round2_experiments.py '
          f'--feats {os.path.relpath(out, ROOT)} --tag {tag}_ '
          f'--only control_round2 arch_meanpool arch_bigru')


if __name__ == '__main__':
    main()
