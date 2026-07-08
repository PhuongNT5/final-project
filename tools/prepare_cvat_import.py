#!/usr/bin/env python3
"""
Prepare CVAT import packages: one zip per class with images + YOLO annotations.

CVAT import format (YOLO 1.1):
  obj.names       — class names
  obj.data        — dataset config
  train.txt       — list of image paths
  obj_train_data/ — images + label .txt files

Usage:
  python tools/prepare_cvat_import.py

Output:
  annotation_guide/cvat_import/class_0.zip
  annotation_guide/cvat_import/class_1.zip
  ...
"""
import os, shutil, zipfile

ANNOT_DIR = 'annotation_guide'
FRAMES_DIR = os.path.join(ANNOT_DIR, 'frames_to_annotate')
LABELS_DIR = os.path.join(ANNOT_DIR, 'annotations')
OUTPUT_DIR = os.path.join(ANNOT_DIR, 'cvat_import')

CLASS_NAMES = [
    'walkway_violation',
    'unauthorized_intervention',
    'opened_panel',
    'carrying_overload',
    'safe_walkway',
    'authorized_intervention',
    'closed_panel',
    'safe_carrying',
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for cls_id in range(8):
        frame_dir = os.path.join(FRAMES_DIR, f'class_{cls_id}')
        label_dir = os.path.join(LABELS_DIR, f'class_{cls_id}')

        if not os.path.isdir(frame_dir):
            print(f"  class_{cls_id}: frames not found, skipping")
            continue

        zip_path = os.path.join(OUTPUT_DIR, f'class_{cls_id}.zip')
        if os.path.exists(zip_path):
            print(f"  class_{cls_id}: zip exists, skipping")
            continue

        frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        print(f"  class_{cls_id} ({CLASS_NAMES[cls_id]}): {len(frames)} frames")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # obj.names — all 8 class names (CVAT needs full list)
            names_content = '\n'.join(CLASS_NAMES) + '\n'
            zf.writestr('obj.names', names_content)

            # obj.data
            data_content = f"classes = 8\ntrain = train.txt\nnames = obj.names\n"
            zf.writestr('obj.data', data_content)

            # train.txt — list of image filenames
            train_lines = [f'obj_train_data/{f}' for f in frames]
            zf.writestr('train.txt', '\n'.join(train_lines) + '\n')

            # obj_train_data/ — images + labels
            for fname in frames:
                # Image
                img_path = os.path.join(frame_dir, fname)
                zf.write(img_path, f'obj_train_data/{fname}')

                # Label (YOLO .txt)
                txt_name = fname.replace('.jpg', '.txt')
                txt_path = os.path.join(label_dir, txt_name)
                if os.path.exists(txt_path):
                    zf.write(txt_path, f'obj_train_data/{txt_name}')
                else:
                    # Empty label — no annotations yet
                    zf.writestr(f'obj_train_data/{txt_name}', '')

        print(f"    → {zip_path}")

    print(f"\nDone! Import these zips into CVAT:")
    print(f"  CVAT → Create Task → Upload: select .zip")
    print(f"  Format: YOLO 1.1")


if __name__ == '__main__':
    main()
