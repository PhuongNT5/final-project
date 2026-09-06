#!/usr/bin/env python3
"""GPU retrain of the paper's 8-class *behaviour* detector (Gate 1 / task A1), on Kaggle.

This is NOT the entity detector. `train-detector-gpu.py` in this same folder trains the 8 *entity*
classes (Person_Authorized, Panel_Closed, ...) that feed the P3 relation-Transformer. This script
trains the 8 *behaviour* classes of the paper's Table 3 scheme (WalkwayViol, UnauthorInt, ...),
which is the vocabulary Gate 1's "mAP50 >= 0.90, all 8 classes emitted" criterion refers to.
Run `build-behaviour-labels.py` first to materialise the dataset.

Recipe follows plans/2026-09-06-thesis-2week-final/research/r1-detector-gpu-recipe.md §1:
the CPU baseline (mAP50 0.791 tight / 0.806 ctx) was trained at imgsz 416 for 25 epochs with
patience=10, which early-stopped at epoch 15 while mAP50 was still rising. The three changes that
matter, in order: imgsz 416->800, epochs 25->200 with patience 30, fliplr 0.0->0.5.

hsv_h stays 0.0 on purpose: AuthorInt vs UnauthorInt is a safety-vest HUE signal (green vs red).
Recolouring flips the label. hsv_s/hsv_v do not touch hue and stay enabled.

Usage (Kaggle):
  python build-behaviour-labels.py --variant tight
  python train-behaviour-detector-gpu.py --data /kaggle/working/annotation_guide/dataset_behaviour_tight/data.yaml
"""
import argparse, json, os
import torch
from ultralytics import YOLO

# Pre-registered Gate 1 criteria (brief.md). Fixed BEFORE the run; do not edit after seeing output.
GATE_MAP50, FLOOR_MAP50, GATE_MIN_RECALL = 0.90, 0.85, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/kaggle/working/annotation_guide/dataset_behaviour_tight/data.yaml')
    ap.add_argument('--model', default='yolo11s.pt',
                    help="yolo11s = safer first attempt; yolo11m = more capacity for the 170-390-label "
                         "tail classes at ~2x epoch time; yolo26s = stretch run, see recipe §3")
    ap.add_argument('--imgsz', type=int, default=800)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=16, help='16 fits a 16GB T4 at 800; drop to 8 on OOM')
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--project', default='/kaggle/working/runs_behaviour')
    ap.add_argument('--name', default=None, help='defaults to the dataset variant name')
    ap.add_argument('--device', default=None)
    ap.add_argument('--smoke', action='store_true', help='1-epoch shape/plumbing check')
    a = ap.parse_args()

    dev = a.device if a.device is not None else (0 if torch.cuda.is_available() else 'cpu')
    variant = os.path.basename(os.path.dirname(a.data)).replace('dataset_behaviour_', '') or 'tight'
    name = a.name or f'beh_{variant}_gpu'
    if a.smoke:
        a.epochs, a.imgsz, a.batch, a.patience = 1, 640, 8, 5

    assert os.path.exists(a.data), f'data.yaml missing: {a.data} — run build-behaviour-labels.py first'
    print(f'[behaviour detector] variant={variant} model={a.model} imgsz={a.imgsz} '
          f'epochs={a.epochs} batch={a.batch} device={dev}')
    print(f'[data] {a.data}')
    print('[note] the val split shares all 8 camera views with train (scene-confounded), so this '
          'mAP measures in-domain detection, NOT cross-camera generalisation. See README.')

    YOLO(a.model).train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=dev,
        optimizer='SGD', lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3.0, cos_lr=True, patience=a.patience, close_mosaic=15,
        amp=True, cache=False, workers=8, seed=0, val=True, plots=True,
        hsv_h=0.0, hsv_s=0.6, hsv_v=0.4,                      # hsv_h=0: vest hue is the label
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.1, copy_paste=0.0,
        project=a.project, name=name, exist_ok=True)

    best = os.path.join(a.project, name, 'weights', 'best.pt')
    print(f'\n[validate] per-class metrics from {best}')
    val = YOLO(best).val(data=a.data, imgsz=a.imgsz, device=dev,
                         project=a.project, name=f'{name}_val', exist_ok=True, plots=True)

    per_class, dead = {}, []
    for i in val.box.ap_class_index:
        i = int(i)
        p, r, ap50, ap = val.box.class_result(i)
        per_class[val.names[i]] = {'P': float(p), 'R': float(r), 'AP50': float(ap50), 'AP50_95': float(ap)}
        print(f'  {val.names[i]:14s} P={p:.3f} R={r:.3f} AP50={ap50:.3f} AP50-95={ap:.3f}')
        if r <= GATE_MIN_RECALL:
            dead.append(val.names[i])
    missing = [n for n in val.names.values() if n not in per_class]   # class absent from val entirely
    dead += missing

    res = {'variant': variant, 'model': a.model, 'imgsz': a.imgsz, 'epochs': a.epochs,
           'mAP50': float(val.box.map50), 'mAP50_95': float(val.box.map),
           'per_class': per_class, 'dead_classes': dead,
           'gate_map50_pass': bool(val.box.map50 >= GATE_MAP50),
           'gate_floor_pass': bool(val.box.map50 >= FLOOR_MAP50),
           'gate_all_classes_pass': not dead, 'weights': best}
    out = os.path.join(a.project, f'{name}_result.json')
    json.dump(res, open(out, 'w'), indent=2)

    print(f'\n  overall mAP50={val.box.map50:.4f}  mAP50-95={val.box.map:.4f}')
    print(f'\n[GATE 1] pre-registered: mAP50>={GATE_MAP50} PASS | >={FLOOR_MAP50} floor | all 8 classes recall>0')
    print(f'  mAP50 >= {GATE_MAP50} : {"PASS" if res["gate_map50_pass"] else "FAIL"}')
    print(f'  mAP50 >= {FLOOR_MAP50} : {"PASS" if res["gate_floor_pass"] else "FAIL"}')
    print(f'  all 8 classes emitted : {"PASS" if res["gate_all_classes_pass"] else "FAIL " + str(dead)}')
    print(f'\n[result json] {out}\n[weights] {best}')


if __name__ == '__main__':
    main()
