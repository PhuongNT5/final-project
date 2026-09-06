#!/usr/bin/env python3
"""Build the paper's 8-class *behaviour* YOLO dataset on Kaggle, from the shipped entity labels.

The Kaggle input mount is read-only and flattens `annotation_guide/` to the archive root, so this
script stitches a small symlink tree that matches the layout `unsafenet.behaviour_dataset` expects
and points it at a writable output dir. The conversion itself is NOT reimplemented here — it calls
the repo's own `behaviour_dataset.build()`, so the Kaggle labels are bit-identical to the local ones.

Usage (Kaggle):
  python build-behaviour-labels.py --variant tight     # -> /kaggle/working/.../dataset_behaviour_tight
  python build-behaviour-labels.py --variant ctx
"""
import argparse
import os
import sys

VARIANTS = {'tight': (1.0, 1.0), 'ctx': None}     # ctx resolved from behaviour_dataset.CTX_W/H


def link(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='/kaggle/input/unsafenet-detector',
                    help='the mounted Kaggle dataset (holds dataset/ and frame_manifest.csv)')
    ap.add_argument('--work', default='/kaggle/working', help='writable output root')
    ap.add_argument('--variant', default='tight', choices=sorted(VARIANTS))
    a = ap.parse_args()

    # the shipped `scripts/` dir holds the unsafenet package
    sys.path.insert(0, os.path.join(a.data_root, 'scripts'))
    from unsafenet import behaviour_dataset as BD

    src_root = os.path.join(a.work, '_src')
    os.makedirs(os.path.join(src_root, 'annotation_guide'), exist_ok=True)
    link(os.path.join(a.data_root, 'dataset'), os.path.join(src_root, 'annotation_guide', 'dataset'))
    link(os.path.join(a.data_root, 'frame_manifest.csv'),
         os.path.join(src_root, 'annotation_guide', 'frame_manifest.csv'))

    ctx = VARIANTS[a.variant] or (BD.CTX_W, BD.CTX_H)
    yml = BD.build(ctx=ctx, suffix='_' + a.variant, src_root=src_root, dst_root=a.work)
    print(f'[build-behaviour-labels] variant={a.variant} ctx={ctx}\n[data.yaml] {yml}')
    return yml


if __name__ == '__main__':
    main()
