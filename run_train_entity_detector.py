#!/usr/bin/env python3
"""Phase 02 — train the multi-entity YOLOv11s detector (Stage A of the paper-faithful pipeline).

Two-stage transfer from COCO: Stage-1 freezes the backbone (head adapts fast), Stage-2 unfreezes
and fine-tunes end-to-end at a low LR. Reads annotation_guide/dataset/data.yaml (nc=8, train != val).

Zones (classes 2,3) have 0 detection instances by design — polygons live in annotation_guide/zones/
and are injected as per-view constants at inference. The detector learns the 6 movable entities.

Usage (repo root):
  .venv/bin/python run_train_entity_detector.py --smoke     # 1-epoch tiny run to validate pipeline
  .venv/bin/python run_train_entity_detector.py             # full two-stage train
"""
import argparse
import os
import yaml
import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'annotation_guide', 'dataset', 'data.yaml')
OUTDIR = os.path.join(ROOT, 'sweep_results', 'entity_detector')


def resolve_data_yaml():
    """Write a copy of data.yaml with an absolute `path` so Ultralytics resolves it correctly."""
    d = yaml.safe_load(open(DATA))
    d['path'] = os.path.join(ROOT, 'annotation_guide', 'dataset')
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, 'data_resolved.yaml')
    yaml.safe_dump(d, open(out, 'w'), sort_keys=False)
    return out


def pick_device():
    if torch.cuda.is_available():
        return 0
    # MPS is intentionally NOT used for TRAINING: on macOS < 15 the addcmul_/addcdiv_ ops
    # silently fail → loss diverges at epoch 2 (see reports/researcher-mps-training-fix.md,
    # verified: MPS explodes, CPU trains cleanly). Use --device mps only for inference/export.
    return 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='yolo11s.pt', help='COCO-pretrained weights to start from')
    ap.add_argument('--imgsz', type=int, default=800, help='train res; 800 balances small-object recall vs 16GB RAM')
    ap.add_argument('--batch', type=int, default=4, help='keep MPS mem < ~8GB on a 16GB M3 (800/b4 ~4.7GB, 960/b8 ~11GB=risky)')
    ap.add_argument('--epochs1', type=int, default=20, help='Stage-1 frozen-backbone epochs')
    ap.add_argument('--epochs2', type=int, default=30, help='Stage-2 fine-tune epochs (0 to skip)')
    ap.add_argument('--freeze', type=int, default=10, help='layers to freeze in Stage-1')
    ap.add_argument('--fraction', type=float, default=1.0, help='fraction of train set (for quick tests)')
    ap.add_argument('--smoke', action='store_true', help='tiny 1-epoch run to validate the pipeline')
    ap.add_argument('--device', default=None, help="override device (e.g. 'cpu', 'mps', '0')")
    args = ap.parse_args()

    dev = args.device if args.device is not None else pick_device()
    data = resolve_data_yaml()
    if args.smoke:
        args.epochs1, args.epochs2, args.imgsz, args.fraction, args.batch = 1, 0, 640, 0.05, 4

    common = dict(data=data, imgsz=args.imgsz, batch=args.batch, device=dev,
                  project=OUTDIR, exist_ok=True, seed=0,
                  fraction=args.fraction, patience=15, plots=True, verbose=True,
                  # SGD lr0=0.01 = ultralytics battle-tested default. optimizer=auto picked AdamW
                  # (lr~8e-4) which blew up box_loss at epoch 2 (post-warmup); forced AdamW lr=1e-3
                  # gave flat loss. SGD is the stable path for YOLO fine-tuning.
                  optimizer='SGD', momentum=0.937, weight_decay=0.0005,   # lr0 set per stage below
                  amp=False,                       # CRITICAL on Apple MPS: fp16 AMP overflows after
                                                   # epoch 1 → loss explodes/diverges. fp32 is stable.
                  workers=4, cache=False,          # 16GB M3: limit dataloader RAM, no image cache
                  # --- vest-color-safe augmentation ---------------------------------------
                  # Authorized=green vs Unauthorized=red-black is a HUE signal → keep hsv_h=0
                  # so vests are never recoloured; still vary saturation/brightness for lighting.
                  hsv_h=0.0, hsv_s=0.5, hsv_v=0.4,
                  degrees=0.0, flipud=0.0, fliplr=0.5, translate=0.1, scale=0.5,
                  mosaic=0.5, close_mosaic=10, mixup=0.0)   # mosaic=0.5 (smoke-proven; 1.0 destabilised warmup)

    print(f"[strategy] imgsz={args.imgsz} batch={args.batch} device={dev} | "
          f"hsv_h=0 (protect vest colour) | 2-stage transfer | deterministic seed=0")
    print(f"[Stage 1] freeze={args.freeze} epochs={args.epochs1} lr0=1e-3 (adapt head on frozen backbone)")
    m = YOLO(os.path.join(ROOT, args.model))
    m.train(name='stage1_frozen', freeze=args.freeze, epochs=args.epochs1,
            lr0=0.01, **common)
    best = os.path.join(OUTDIR, 'stage1_frozen', 'weights', 'best.pt')

    if args.epochs2 > 0:
        print(f"\n[Stage 2] unfreeze epochs={args.epochs2} lr0=1e-4 (from {os.path.relpath(best, ROOT)})")
        m2 = YOLO(best)
        m2.train(name='stage2_finetune', freeze=0, epochs=args.epochs2, lr0=0.001, **common)
        best = os.path.join(OUTDIR, 'stage2_finetune', 'weights', 'best.pt')

    print(f"\n[done] best detector → {os.path.relpath(best, ROOT)}")
    print("Next: validate per-class mAP (esp. Person_Authorized/Unauthorized), then build relation tokens.")


if __name__ == '__main__':
    main()
