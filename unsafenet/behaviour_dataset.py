"""Convert the existing entity annotations into the paper's *behaviour* annotations.

Onal & Dandil train YOLO v4 on 2,262 frames whose boxes carry the eight **behaviour** labels
(Table 3).  This project annotated the very same 2,262 frames with **entity** labels instead, and
`frame_manifest.csv` records each frame's source clip and therefore its behaviour class.  So the
paper's label set is recoverable with no new annotation:

* panel and forklift boxes already *are* behaviours (opened/closed cover, overload/safe load);
* person boxes take the behaviour of the clip they came from, which is the only thing that
  distinguishes a walkway violation from an unauthorized intervention;
* person boxes in panel/forklift clips carry no known behaviour and are dropped.
"""
from __future__ import annotations
import csv
import os
import shutil

from . import config as C

SRC = 'annotation_guide/dataset'
DST = 'annotation_guide/dataset_behaviour'
MANIFEST = 'annotation_guide/frame_manifest.csv'

PERSON_CLIP_CLASSES = {0, 1, 4, 5}      # clips whose person box has a known behaviour
# A person box is a 14x50 px sliver at 416 px: what makes it a *walkway violation* rather than an
# *unauthorized intervention* is the floor and machinery AROUND it, not the pixels inside it.  A
# detector classifying from box-local features therefore cannot separate these classes at all (first
# attempt: mAP50 = 0.014).  Person boxes are widened into behaviour *regions* that contain the cue.
CTX_W, CTX_H = 4.0, 1.8
ENTITY_TO_BEHAVIOUR = {4: 6, 5: 2, 6: 7, 7: 3}   # panel/forklift entity id -> behaviour id


def clip_labels(root: str | None = None) -> dict[str, int]:
    rows = list(csv.DictReader(open(os.path.join(root or C.ROOT, MANIFEST))))
    return {os.path.splitext(r['clip'])[0]: int(r['source_class_id']) for r in rows}


def _expand(x, y, w, h, fw, fh):
    w, h = min(w * fw, 1.0), min(h * fh, 1.0)
    x = min(max(x, w / 2), 1 - w / 2)
    y = min(max(y, h / 2), 1 - h / 2)
    return x, y, w, h


def _convert(lines: list[str], clip_cls: int, ctx=(CTX_W, CTX_H)) -> list[str]:
    out = []
    for ln in lines:
        p = ln.split()
        if len(p) != 5:
            continue
        eid = int(p[0])
        box = tuple(map(float, p[1:]))
        if eid in ENTITY_TO_BEHAVIOUR:
            bid = ENTITY_TO_BEHAVIOUR[eid]      # panels/forklifts already are behaviours
        elif eid in C.PERSON_IDS and clip_cls in PERSON_CLIP_CLASSES:
            bid = clip_cls                      # the clip's behaviour defines the person's behaviour
            box = _expand(*box, *ctx)           # widen to include the cue that defines it
        else:
            continue                            # zone boxes, or a person with unknown behaviour
        out.append(' '.join([str(bid)] + [f'{v:.6f}' for v in box]))
    return out


def build(ctx=(CTX_W, CTX_H), suffix='', src_root=None, dst_root=None) -> str:
    """src_root/dst_root let a read-only mount (Kaggle /kaggle/input) feed a writable output dir."""
    labs = clip_labels(src_root)
    src = os.path.join(src_root or C.ROOT, SRC)
    dst = os.path.join(dst_root or src_root or C.ROOT, DST + suffix)
    stats = {k: 0 for k in range(8)}
    kept = dropped = 0
    for sp in ('train', 'val'):
        os.makedirs(os.path.join(dst, 'images', sp), exist_ok=True)
        os.makedirs(os.path.join(dst, 'labels', sp), exist_ok=True)
        for img in sorted(os.listdir(os.path.join(src, 'images', sp))):
            stem = os.path.splitext(img)[0]
            clip = stem.rsplit('_', 1)[0]
            if clip not in labs:
                dropped += 1
                continue
            lp = os.path.join(src, 'labels', sp, stem + '.txt')
            lines = open(lp).read().splitlines() if os.path.exists(lp) else []
            new = _convert(lines, labs[clip], ctx)
            if not new:
                dropped += 1
                continue
            for ln in new:
                stats[int(ln.split()[0])] += 1
            dl = os.path.join(dst, 'images', sp, img)
            if not os.path.exists(dl):
                os.symlink(os.path.join(src, 'images', sp, img), dl)
            open(os.path.join(dst, 'labels', sp, stem + '.txt'), 'w').write('\n'.join(new) + '\n')
            kept += 1
    yml = os.path.join(dst, 'data.yaml')
    names = '\n'.join(f'  {i}: {n}' for i, n in enumerate(C.CLASS_NAMES8))
    open(yml, 'w').write(
        f'# Behaviour-as-object dataset - the paper\'s YOLO v4 label scheme (Table 3),\n'
        f'# derived from the existing entity annotations + frame_manifest.csv clip labels.\n'
        f'path: {dst}\ntrain: images/train\nval: images/val\nnc: 8\nnames:\n{names}\n')
    print(f'[behaviour_dataset] frames kept={kept} dropped={dropped}')
    for i, n in enumerate(C.CLASS_NAMES8):
        print(f'  {i} {n:14s} {stats[i]:5d} boxes')
    return yml


if __name__ == '__main__':
    build()
