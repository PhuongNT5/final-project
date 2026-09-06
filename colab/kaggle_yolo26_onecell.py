# === ONE-CELL: YOLO26 detector — train + save + validate + benchmark (Kaggle) ===
# Paste into a SINGLE Kaggle cell. Delete all other cells. Then Save Version -> Save & Run All (Commit).
# Requirements: Session -> GPU T4 x2, Internet ON, dataset 'unsafe-net-detector-dataset' attached.
import glob, os, shutil, subprocess
subprocess.run(['pip', '-q', 'install', '-U', 'ultralytics'], check=True)
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

# 1. dataset (Kaggle auto-extracts the uploaded zip; copy to writable working dir)
hits = glob.glob('/kaggle/input/**/images/train', recursive=True)
assert hits, f'No images/train under /kaggle/input = {os.listdir("/kaggle/input")}'
src = os.path.dirname(os.path.dirname(hits[0]))
root = '/kaggle/working/ds'
if not os.path.exists(root):
    shutil.copytree(src, root)
print('dataset:', root, '| train', len(glob.glob(root + '/images/train/*.jpg')),
      '| val', len(glob.glob(root + '/images/val/*.jpg')))

# 2. data.yaml
DY = '/kaggle/working/data.yaml'
open(DY, 'w').write(
    f"path: {root}\ntrain: images/train\nval: images/val\nnc: 8\n"
    "names: [Person_Authorized,Person_Unauthorized,Safe_Walkway,Restricted_Zone,"
    "Panel_Closed,Panel_Opened,Forklift_Safe_Load,Forklift_Overload]\n")

# 3. train — SAME recipe as the YOLOv11s baseline (fair swap), 2-stage freeze->fine-tune
RUNS = '/kaggle/working/runs26'
AUG = dict(hsv_h=0.0, hsv_s=0.5, hsv_v=0.4, degrees=0.0, flipud=0.0, fliplr=0.5,
           translate=0.1, scale=0.5, mosaic=0.5, close_mosaic=10, mixup=0.0)
COMMON = dict(data=DY, imgsz=800, batch=16, device=0, optimizer='SGD', momentum=0.937,
              weight_decay=0.0005, seed=0, patience=30, plots=False, project=RUNS, exist_ok=True, **AUG)
YOLO('yolo26s.pt').train(name='s1', freeze=10, epochs=30, lr0=0.01, **COMMON)
YOLO(RUNS + '/s1/weights/best.pt').train(name='s2', freeze=0, epochs=100, lr0=0.001, **COMMON)
BEST = RUNS + '/s2/weights/best.pt'
shutil.copy(BEST, '/kaggle/working/yolo26_best.pt')
print('SAVED -> /kaggle/working/yolo26_best.pt', round(os.path.getsize('/kaggle/working/yolo26_best.pt') / 1e6, 1), 'MB')

# 4. validate + edge benchmark
best = YOLO(BEST); best.info()
mm = best.val(data=DY, imgsz=800, device=0)
print('YOLO26 mAP50=', round(float(mm.box.map50), 3), 'mAP50-95=', round(float(mm.box.map), 3),
      '(YOLO11 was 0.969 / 0.894)')
df = benchmark(model=BEST, data=DY, imgsz=800, half=True, int8=True)
print(df); df.to_csv('/kaggle/working/yolo26_benchmark.csv')
print('DONE. Download /kaggle/working/yolo26_best.pt and yolo26_benchmark.csv from the Output tab.')
