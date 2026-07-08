#!/usr/bin/env python3
"""
Auto-annotate 400 frames using YOLO-World open-vocabulary detection.

For each frame, detects relevant objects and assigns the class label
from the folder name. Saves annotations in YOLO format (.txt files).

Strategy per class:
  - Person-centric classes (0,1,4,5): detect "person/worker" → bbox
  - Panel classes (2,6): detect "electrical panel/cabinet" → bbox
  - Forklift classes (3,7): detect "forklift" + "load/box/pallet" → bbox

The user should REVIEW these in CVAT and correct any errors.
"""
import os, sys, cv2, argparse, numpy as np

def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument('--overwrite', action='store_true',
                    help='re-draft ALL unsafe frames fresh (including the 400 V10 labels) for a '
                         'uniform starting point — use when re-annotating from scratch')
    args = ap.parse_args()

    FRAMES_DIR = 'annotation_guide/frames_to_annotate'
    ANNOT_DIR = 'annotation_guide/annotations'
    os.makedirs(ANNOT_DIR, exist_ok=True)

    # Class definitions matching YOLO training labels
    # These are the 8 behavior classes we want to detect
    CLASS_NAMES = [
        'walkway_violation',        # 0
        'unauthorized_intervention', # 1
        'opened_panel',             # 2
        'carrying_overload',        # 3
        'safe_walkway',             # 4
        'authorized_intervention',  # 5
        'closed_panel',             # 6
        'safe_carrying',            # 7
    ]

    # Detection strategy per class
    # For each class, what YOLO-World text prompt to use and what to annotate
    CLASS_DETECT_CONFIG = {
        0: {'prompts': ['person', 'worker'], 'strategy': 'largest_person',
            'desc': 'Person in restricted walkway'},
        1: {'prompts': ['person', 'worker'], 'strategy': 'person_near_equipment',
            'desc': 'Person touching equipment'},
        2: {'prompts': ['electrical panel', 'cabinet', 'open door', 'switchboard'],
            'strategy': 'largest_object', 'desc': 'Open electrical panel'},
        3: {'prompts': ['forklift', 'vehicle', 'load', 'pallet', 'box'],
            'strategy': 'largest_vehicle', 'desc': 'Forklift with oversize load'},
        4: {'prompts': ['person', 'worker'], 'strategy': 'largest_person',
            'desc': 'Person in safe walkway'},
        5: {'prompts': ['person', 'worker'], 'strategy': 'person_near_equipment',
            'desc': 'Person authorized at equipment'},
        6: {'prompts': ['electrical panel', 'cabinet', 'closed door', 'switchboard'],
            'strategy': 'largest_object', 'desc': 'Closed electrical panel'},
        7: {'prompts': ['forklift', 'vehicle', 'load', 'pallet'],
            'strategy': 'largest_vehicle', 'desc': 'Forklift with safe load'},
    }

    # PAPER-FAITHFUL 4-class frame-filter: only unsafe classes 0-3 get boxes; safe classes 4-7 are
    # BACKGROUND (empty labels). Existing V10 labels (annotations/class_{c}/) are PRESERVED for unsafe
    # classes (skip-if-exists) and EMPTIED for safe classes. No garbage fallback box.
    UNSAFE = {0, 1, 2, 3}
    print("Loading YOLO-World...")
    model = YOLO('yolov8s-worldv2.pt')

    total_annotated = 0
    total_no_detection = 0
    total_empty = 0

    for cls_id in range(8):
        cls_dir = os.path.join(FRAMES_DIR, f'class_{cls_id}')
        ann_dir = os.path.join(ANNOT_DIR, f'class_{cls_id}')
        os.makedirs(ann_dir, exist_ok=True)

        if not os.path.isdir(cls_dir):
            print(f"  Class {cls_id}: directory not found, skipping")
            continue

        config = CLASS_DETECT_CONFIG[cls_id]
        frames = sorted([f for f in os.listdir(cls_dir) if f.endswith('.jpg')])
        is_unsafe = cls_id in UNSAFE
        print(f"\nClass {cls_id} ({CLASS_NAMES[cls_id]}): {len(frames)} frames "
              f"[{'UNSAFE → draft boxes' if is_unsafe else 'SAFE → background (empty)'}]")

        # Safe classes: write empty label for every frame (background); overwrite stale V10 safe labels.
        if not is_unsafe:
            n_empty = 0
            for fname in frames:
                open(os.path.join(ann_dir, fname.replace('.jpg', '.txt')), 'w').close()
                n_empty += 1
            total_empty += n_empty
            print(f"  Emptied (background): {n_empty}")
            continue

        print(f"  Strategy: {config['strategy']}, prompts: {config['prompts']}")
        model.set_classes(config['prompts'])
        annotated = 0
        no_det = 0
        skipped = 0

        for fi, fname in enumerate(frames):
            txt_path = os.path.join(ann_dir, fname.replace('.jpg', '.txt'))
            if os.path.exists(txt_path) and not args.overwrite:
                skipped += 1   # preserve existing label (use --overwrite to re-draft fresh)
                continue
            img_path = os.path.join(cls_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]

            results = model.predict(img, conf=0.05, verbose=False, imgsz=640)
            boxes = results[0].boxes
            selected_boxes = []

            if len(boxes) == 0:
                # No detection → leave EMPTY (no garbage box); user adds the box in review.
                no_det += 1
            else:
                xyxyn = boxes.xyxyn.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                if config['strategy'] == 'largest_person':
                    # Pick the person with largest bbox area
                    areas = (xyxyn[:, 2] - xyxyn[:, 0]) * (xyxyn[:, 3] - xyxyn[:, 1])
                    best_idx = areas.argmax()
                    x1, y1, x2, y2 = xyxyn[best_idx]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    w, h = x2 - x1, y2 - y1
                    selected_boxes.append((cls_id, cx, cy, w, h))

                elif config['strategy'] == 'person_near_equipment':
                    # Pick person closest to center (likely the one interacting)
                    centers = np.stack([(xyxyn[:, 0] + xyxyn[:, 2]) / 2,
                                        (xyxyn[:, 1] + xyxyn[:, 3]) / 2], axis=1)
                    dists = np.sqrt((centers[:, 0] - 0.5)**2 + (centers[:, 1] - 0.5)**2)
                    best_idx = dists.argmin()
                    x1, y1, x2, y2 = xyxyn[best_idx]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    w, h = x2 - x1, y2 - y1
                    # Expand bbox slightly to include nearby equipment
                    w = min(1.0, w * 1.3)
                    h = min(1.0, h * 1.3)
                    selected_boxes.append((cls_id, cx, cy, w, h))

                elif config['strategy'] == 'largest_object':
                    # Pick largest detection (panel/cabinet)
                    areas = (xyxyn[:, 2] - xyxyn[:, 0]) * (xyxyn[:, 3] - xyxyn[:, 1])
                    best_idx = areas.argmax()
                    x1, y1, x2, y2 = xyxyn[best_idx]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    w, h = x2 - x1, y2 - y1
                    selected_boxes.append((cls_id, cx, cy, w, h))

                elif config['strategy'] == 'largest_vehicle':
                    # Pick largest detection (forklift + load)
                    areas = (xyxyn[:, 2] - xyxyn[:, 0]) * (xyxyn[:, 3] - xyxyn[:, 1])
                    best_idx = areas.argmax()
                    x1, y1, x2, y2 = xyxyn[best_idx]
                    # If multiple detections, merge into one big bbox
                    if len(boxes) > 1:
                        # Union of top-2 largest
                        top2 = areas.argsort()[-2:]
                        x1 = xyxyn[top2, 0].min()
                        y1 = xyxyn[top2, 1].min()
                        x2 = xyxyn[top2, 2].max()
                        y2 = xyxyn[top2, 3].max()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    w, h = x2 - x1, y2 - y1
                    selected_boxes.append((cls_id, cx, cy, w, h))

            # Write YOLO format annotation
            txt_fname = fname.replace('.jpg', '.txt')
            txt_path = os.path.join(ann_dir, txt_fname)
            with open(txt_path, 'w') as f:
                for (c, cx, cy, w, h) in selected_boxes:
                    # Clamp to valid range
                    cx = max(0.001, min(0.999, cx))
                    cy = max(0.001, min(0.999, cy))
                    w = max(0.01, min(0.999, w))
                    h = max(0.01, min(0.999, h))
                    f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            annotated += 1

        total_annotated += annotated
        total_no_detection += no_det
        print(f"  Drafted: {annotated} | empty (needs manual box): {no_det} | "
              f"skipped existing V10: {skipped}")

    # 4-class names (paper-faithful): only the unsafe behaviour classes are detector classes.
    UNSAFE_NAMES = CLASS_NAMES[:4]
    names_path = os.path.join(ANNOT_DIR, 'classes.txt')
    with open(names_path, 'w') as f:
        for name in UNSAFE_NAMES:
            f.write(name + '\n')

    # Write dataset YAML for YOLOv11 training (4 unsafe classes; safe frames = background negatives)
    yaml_path = os.path.join(ANNOT_DIR, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: ../annotation_guide\n")
        f.write(f"train: frames_to_annotate\n")
        f.write(f"val: frames_to_annotate\n")
        f.write(f"nc: 4\n")
        f.write(f"names: {UNSAFE_NAMES}\n")

    print(f"\n{'='*60}")
    print(f"AUTO-ANNOTATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total frames annotated: {total_annotated}")
    print(f"  Frames with no detection (fallback bbox): {total_no_detection}")
    print(f"  Annotations saved to: {ANNOT_DIR}/")
    print(f"  Class names: {names_path}")
    print(f"  Dataset YAML: {yaml_path}")
    print(f"\n  NEXT STEPS:")
    print(f"  1. Review annotations in CVAT (import images + YOLO txt)")
    print(f"  2. Correct any wrong/missing boxes")
    print(f"  3. Tell me when done → I retrain YOLOv11")


if __name__ == '__main__':
    main()
