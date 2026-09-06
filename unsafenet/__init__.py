"""Unsafe-Net R2 - paper-faithful rewrite of the unsafe-behaviour recognition pipeline.

Stage A  entities.py       YOLOv11s entity detector + IoU tracks      (paper: YOLO v4 detector)
Stage B  frame_model.py    per-frame behaviour recogniser             (paper: per-frame supervision)
         geometry.py       clip-registered zone polygons + colour prior
         streams.py        entity / zone / motion evidence per frame
Stage C  temporal.py       BiLSTM | Transformer | mean over frames    (paper: ConvLSTM)
         aggregate.py      training-free frame-reduction aggregations (paper: YOLO frame gate)
Stage D  fusion.py         val-selected ensemble
         evaluate.py       blind scoring vs the published numbers
"""
__all__ = ['config', 'data_index', 'frame_cache', 'frame_model', 'train_frame', 'entities',
           'geometry', 'streams', 'temporal', 'train_temporal', 'aggregate', 'fusion', 'evaluate']
