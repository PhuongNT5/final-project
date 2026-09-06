#!/usr/bin/env python3
"""Phase 1 · L1 — strip absolute (xc, yc) from relation tokens (camera-fingerprint removal).

Absolute box centre re-encodes camera identity (a person recurs in the same screen region per camera),
inflating same-dist/random-kfold and rotting leave-scene-out. This drops columns 1,2 (xc,yc) from an
EXISTING token set — no re-detection/appearance needed — keeping [class, w, h, conf, iou_walkway,
iou_restricted, foot_in_walkway, norm_dist, +appearance]. Preserves mask/labels/clip_ids/views so the
repaired-scene LCO and k-fold run unchanged.

Usage: build_l1_tokens.py --src yolo26app_ --dst yolo26L1_
"""
import argparse, os, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = os.path.join(ROOT, 'sweep_results', 'relation_tokens')
DROP = [1, 2]  # xc, yc (absolute centre)


def convert(src, dst, split):
    d = torch.load(os.path.join(TOK, f'tokens_{src}{split}.pt'), weights_only=False)
    tok = d['tokens']                                   # (N,T,box,feat)
    keep = [i for i in range(tok.shape[-1]) if i not in DROP]
    d['tokens'] = tok[..., keep].contiguous()
    if 'feat_layout' in d:
        d['feat_layout'] = [d['feat_layout'][i] for i in keep]
    out = os.path.join(TOK, f'tokens_{dst}{split}.pt')
    torch.save(d, out)
    print(f"  {split}: {tuple(tok.shape)} -> {tuple(d['tokens'].shape)}  [saved] {out}")
    if 'feat_layout' in d:
        print(f"    layout: {d['feat_layout'][:10]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='yolo26app_')
    ap.add_argument('--dst', default='yolo26L1_')
    args = ap.parse_args()
    for sp in ('train', 'test'):
        convert(args.src, args.dst, sp)


if __name__ == '__main__':
    main()
