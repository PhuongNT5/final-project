#!/usr/bin/env python3
"""GPU retrain of the multi-entity YOLO detector (Kaggle) — 2-stage COCO transfer.

Mirrors the repo's `run_train_entity_detector.py` strategy (2-stage freeze->finetune, SGD, vest-colour-safe
aug hsv_h=0) but tuned for a CUDA GPU: AMP on, larger batch, image cache, imgsz 960. Two arms:

  baseline (default) : 6 movable entities. Primary goal — recover Forklift_Safe_Load (170 labels but the
                       CPU-trained detector emits 0 boxes) and lift per-class mAP at higher res.
  --add_zones        : train on the zone-augmented dataset (classes 2,3 present) built by
                       build-zone-labels.py. EXPERIMENTAL — static per-camera zones may not transfer to a
                       held-out camera (the LCO leak L2 showed); the downstream LCO eval is the judge.

Usage (Kaggle, baseline):
  python train-detector-gpu.py --data_root /kaggle/input/<ds>
Usage (Kaggle, zone arm — after running build-zone-labels.py):
  python train-detector-gpu.py --zones_yaml /kaggle/working/dataset_zones/data.yaml --add_zones
"""
import argparse, os
import torch
from ultralytics import YOLO


NAMES = ['Person_Authorized', 'Person_Unauthorized', 'Safe_Walkway', 'Restricted_Zone',
         'Panel_Closed', 'Panel_Opened', 'Forklift_Safe_Load', 'Forklift_Overload']


def pick_device():
    return 0 if torch.cuda.is_available() else 'cpu'


def resolve_baseline_yaml(data_root, out_dir='/kaggle/working'):
    """Write a data.yaml with an ABSOLUTE path derived from the real mount.

    The shipped dataset/data.yaml has a baked-in `path:` that only matches one mount slug; the actual
    Kaggle mount (e.g. /kaggle/input/datasets/<user>/<slug>) differs, so we regenerate it from data_root.
    """
    ds = os.path.abspath(os.path.join(data_root, 'dataset'))
    assert os.path.isdir(os.path.join(ds, 'images', 'val')), f'val images missing under {ds}/images/val'
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'data_baseline.yaml')
    with open(out, 'w') as f:
        f.write(f"path: {ds}\ntrain: images/train\nval: images/val\nnc: {len(NAMES)}\nnames:\n"
                + ''.join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='/kaggle/input/unsafenet-detector',
                    help='root holding dataset/ (used to locate dataset/data.yaml for the baseline arm)')
    ap.add_argument('--zones_yaml', default='/kaggle/working/dataset_zones/data.yaml',
                    help='data.yaml produced by build-zone-labels.py (used with --add_zones)')
    ap.add_argument('--add_zones', action='store_true', help='train the experimental zone-detection arm')
    ap.add_argument('--model', default='yolo11s.pt')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--batch', type=int, default=16, help='16 fits a 16GB T4 at 960; drop to 8 on OOM')
    ap.add_argument('--epochs1', type=int, default=20, help='Stage-1 frozen-backbone epochs')
    ap.add_argument('--epochs2', type=int, default=40, help='Stage-2 fine-tune epochs (0 to skip)')
    ap.add_argument('--freeze', type=int, default=10)
    ap.add_argument('--project', default='/kaggle/working/runs')
    ap.add_argument('--smoke', action='store_true', help='tiny 1-epoch validation run')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    dev = args.device if args.device is not None else pick_device()
    data = args.zones_yaml if args.add_zones else resolve_baseline_yaml(args.data_root, args.project)
    arm = 'ZONES (experimental)' if args.add_zones else 'baseline (6 movable entities)'
    if args.smoke:
        args.epochs1, args.epochs2, args.imgsz, args.batch = 1, 0, 640, 8

    common = dict(
        data=data, imgsz=args.imgsz, batch=args.batch, device=dev,
        project=args.project, exist_ok=True, seed=0, patience=15, plots=True, verbose=True,
        optimizer='SGD', momentum=0.937, weight_decay=0.0005,
        amp=True,                          # GPU: AMP is stable & ~2x faster (the repo's amp=False was MPS-only)
        workers=4, cache=False,            # RAM-safe on Kaggle (2262 full-res frames ~14GB would OOM);
                                           # set cache='disk' for faster epochs if disk quota allows
        # vest-colour-safe: Authorized=green vs Unauthorized=red is a HUE signal -> hsv_h=0 (never recolour)
        hsv_h=0.0, hsv_s=0.5, hsv_v=0.4,
        degrees=0.0, flipud=0.0, fliplr=0.5, translate=0.1, scale=0.5,
        mosaic=1.0, close_mosaic=10, mixup=0.0)

    print(f"[arm] {arm}\n[data] {data}\n[strategy] imgsz={args.imgsz} batch={args.batch} device={dev} "
          f"| AMP on | hsv_h=0 | 2-stage transfer | seed=0")
    print(f"[Stage 1] freeze={args.freeze} epochs={args.epochs1} lr0=0.01")
    m = YOLO(args.model)
    m.train(name='stage1_frozen', freeze=args.freeze, epochs=args.epochs1, lr0=0.01, **common)
    best = os.path.join(args.project, 'stage1_frozen', 'weights', 'best.pt')

    if args.epochs2 > 0:
        print(f"\n[Stage 2] unfreeze epochs={args.epochs2} lr0=0.001 (from {best})")
        m2 = YOLO(best)
        m2.train(name='stage2_finetune', freeze=0, epochs=args.epochs2, lr0=0.001, **common)
        best = os.path.join(args.project, 'stage2_finetune', 'weights', 'best.pt')

    print(f"\n[validate] per-class metrics on val split ...")
    val = YOLO(best).val(data=data, imgsz=args.imgsz, device=dev, project=args.project,
                         name='val_final', exist_ok=True, plots=True)
    names = val.names
    print("  per-class AP50 / AP50-95:")
    for i, ap50 in zip(val.box.ap_class_index, val.box.ap50):
        print(f"    {names[int(i)]:20s} AP50={ap50:.3f}  AP50-95={val.box.maps[int(i)]:.3f}")
    print(f"  overall  mAP50={val.box.map50:.3f}  mAP50-95={val.box.map:.3f}")
    print(f"\n[done] best detector -> {best}")
    print("Handoff: download best.pt to the repo, then re-extract relation tokens on the 516 behaviour")
    print("clips and re-run run_lco_eval.py / run_kfold_relation.py to judge the new detector.")


if __name__ == '__main__':
    main()
