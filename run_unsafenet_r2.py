#!/usr/bin/env python3
"""Driver for the Unsafe-Net R2 pipeline.

    python run_unsafenet_r2.py cache            # decode clips -> frame cache
    python run_unsafenet_r2.py detect           # entity detections + tracks
    python run_unsafenet_r2.py streams          # zone geometry + motion evidence
    python run_unsafenet_r2.py frame  --tag b0  # per-frame behaviour recogniser (Stage B)
    python run_unsafenet_r2.py agg    --tag b0  # training-free aggregations
    python run_unsafenet_r2.py temporal --tag b0 --kind lstm
    python run_unsafenet_r2.py fuse             # val-selected ensemble
    python run_unsafenet_r2.py eval             # blind test report vs the paper
    python run_unsafenet_r2.py all    --tag b0  # everything, in order
"""
import argparse
import sys

from unsafenet import (aggregate, entities, evaluate, frame_cache, fusion, streams,
                       train_frame, train_temporal)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('stage', choices=['cache', 'detect', 'streams', 'frame', 'agg', 'temporal',
                                     'fuse', 'eval', 'all'])
    a.add_argument('--tag', default='b0'); a.add_argument('--backbone', default='resnet34')
    a.add_argument('--kind', default='lstm'); a.add_argument('--use', default='emb,logit,geom')
    a.add_argument('--seeds', type=int, default=5); a.add_argument('--epochs', type=int, default=14)
    a.add_argument('--device', default='mps'); a.add_argument('--name', default=None)
    a.add_argument('--pattern', default='*'); a.add_argument('--path', default='fusion')
    v = a.parse_args()

    if v.stage in ('cache', 'all'):
        frame_cache.build()
    if v.stage in ('detect', 'all'):
        entities.build(device=v.device)
    if v.stage in ('streams', 'all'):
        streams.build()
    if v.stage in ('frame', 'all'):
        train_frame.run(backbone=v.backbone, epochs=v.epochs, device=v.device, tag=v.tag)
    if v.stage in ('agg', 'all'):
        aggregate.run(tag=v.tag)
    if v.stage in ('temporal', 'all'):
        train_temporal.run(tag=v.tag, kind=v.kind, use=tuple(v.use.split(',')),
                           seeds=v.seeds, device=v.device, name=v.name)
    if v.stage in ('fuse', 'all'):
        fusion.run(v.pattern)
    if v.stage in ('eval', 'all'):
        evaluate.run(v.path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
