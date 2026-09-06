"""Train the paper's YOLO v4 stage: a per-frame *behaviour* detector (nc=8).

NOTE: device defaults to cpu. Ultralytics detection training on the MPS backend in this
stack (torch 2.11.0 / ultralytics 8.4.27) does not converge (mAP50 ~2e-4 on data where the
same recipe on cpu reaches 0.94 after one epoch); MPS *inference* is unaffected.
"""
import argparse
from ultralytics import YOLO
from unsafenet import behaviour_dataset as BD

a = argparse.ArgumentParser()
a.add_argument('--model', default='yolo11s.pt'); a.add_argument('--imgsz', type=int, default=416)
a.add_argument('--epochs', type=int, default=25); a.add_argument('--batch', type=int, default=8)
a.add_argument('--device', default='cpu'); a.add_argument('--name', default='behaviour_yolo')
a.add_argument('--boxes', default='tight', choices=['tight', 'ctx'])
v = a.parse_args()
ctx = (1.0, 1.0) if v.boxes == 'tight' else (BD.CTX_W, BD.CTX_H)
yml = BD.build(ctx=ctx, suffix='_' + v.boxes)
YOLO(v.model).train(data=yml, imgsz=v.imgsz, epochs=v.epochs, batch=v.batch, device=v.device,
                    amp=False, optimizer='SGD', lr0=0.01,
                    project='sweep_results/r2', name=v.name, exist_ok=True, patience=10,
                    hsv_h=0.0, fliplr=0.0, degrees=0.0, translate=0.05, scale=0.3, mosaic=0.5,
                    seed=0, val=True, plots=False, workers=4)
