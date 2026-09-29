#!/usr/bin/env python3
"""Whole-frame (NO detector, NO crops) features for iSafetyBench — the falsification baseline.

The entity-token ablations showed no single channel is necessary, which raises the obvious committee
question: does the detector + relation head buy anything at all over classifying the raw frames? This
builder produces the input for that comparison, using the SAME backbone, the SAME 16-frame sampling
and the SAME 64-d random projection as the crop appearance features, so the only difference between
the two pipelines is "whole frame" vs "detected entities".

Reuses `build_wholeframe_features.get_model/process` unchanged (composition, no forked copy) and
`build_isafetybench_tokens.clips()` for the clip list + source-id grouping, so the baseline is scored
on exactly the same 1,100 clips and the same leave-source-out folds as the encoder.

Usage: .venv/bin/python build_isafetybench_wholeframe.py
"""
import argparse
import os

import numpy as np
import torch

import build_isafetybench_tokens as ISB
import build_wholeframe_features as W

OUT = W.OUT


def build(tag='isb_', limit=None, device=None):
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    feat, tf, proj = W.get_model(device)
    items = ISB.clips()
    if limit:
        items = items[:limit]
    feats, labels, cids, srcs = [], [], [], []
    for k, (path, label, src) in enumerate(items):
        feats.append(W.process(path, feat, tf, proj, device))
        labels.append(label); cids.append(os.path.basename(path)); srcs.append(src)
        if (k + 1) % 100 == 0:
            print(f"  [wf-isb] {k+1}/{len(items)}", flush=True)
    F = torch.tensor(np.stack(feats))
    path = os.path.join(OUT, f'tokens_wf_{tag}all.pt')
    torch.save({'feats': F, 'labels': torch.tensor(labels), 'views': srcs, 'clip_ids': cids,
                'backbone': 'mobilenet_v3_small/IMAGENET1K_V1 -> fixed 64d random projection',
                'dataset': 'isafetybench'}, path)
    print(f"[wf-isb] {len(items)} clips -> {path} | shape {tuple(F.shape)} | "
          f"hazard={int(sum(labels))} | sources={len(set(srcs))}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--out_tag', default='isb_')
    args = ap.parse_args()
    build(args.out_tag, args.limit, args.device)


if __name__ == '__main__':
    main()
