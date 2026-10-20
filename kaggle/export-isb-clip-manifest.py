#!/usr/bin/env python3
"""Freeze the published clip order into a manifest the Kaggle run must reproduce exactly.

Why this file exists
--------------------
Every post-hoc row is PAIRED with the 83.25% four-thread control on the same clips in the same
order: `analyze_posthoc_paired_vs_control.py` refuses any row whose clip order or labels differ.
A feature file built on another machine is therefore only usable if its row k is the same clip as
row k of `feats_isbclip_all.pt`. Kaggle re-enumerates the corpus from the filesystem, and a
different glob order, a different locale collation or one missing upload would silently produce a
file that still loads and still trains -- and whose every margin would be meaningless.

So the order is exported here, from the published artifact itself, and asserted there.

Writes kaggle/isb-clip-manifest.json: clip_ids, labels, views (source video id), in published order.

Usage: .venv/bin/python kaggle/export-isb-clip-manifest.py
"""
import hashlib
import json
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUB = os.path.join(ROOT, 'sweep_results/relation_tokens/feats_isbclip_all.pt')
OUT = os.path.join(ROOT, 'kaggle/isb-clip-manifest.json')


def main():
    d = torch.load(PUB, weights_only=False)
    cids = [str(c) for c in d['clip_ids']]
    labels = [int(v) for v in d['labels']]
    views = [str(v) for v in d['views']]
    assert len(cids) == len(labels) == len(views) == 1100, (len(cids), len(labels), len(views))

    man = {
        'n': len(cids),
        'T': int(d['wholeframe'].shape[1]),
        'resize': 224,
        'source_artifact': os.path.relpath(PUB, ROOT),
        'published_backbone': str(d.get('backbone')),
        'published_dim': int(d['wholeframe'].shape[-1]),
        # The Kaggle side asserts against this digest, so a reordered or incomplete upload fails
        # loudly at minute one instead of producing an unusable 54 MB download an hour later.
        'order_sha256': hashlib.sha256('\n'.join(cids).encode()).hexdigest(),
        'clip_ids': cids, 'labels': labels, 'views': views,
    }
    json.dump(man, open(OUT, 'w'), indent=1)
    print(f'[manifest] {man["n"]} clips, T={man["T"]}, order_sha256={man["order_sha256"][:16]}...')
    print(f'[manifest] published backbone={man["published_backbone"]} dim={man["published_dim"]}')
    print(f'[manifest] wrote {os.path.relpath(OUT, ROOT)}')
    print(f'[manifest] hazard={sum(labels)} normal={len(labels) - sum(labels)} '
          f'sources={len(set(views))}')


if __name__ == '__main__':
    main()
