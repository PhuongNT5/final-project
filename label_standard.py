import cv2
import os
import sys
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# --- CONFIGURATION (Customize these if needed) ---
INPUT_ROOT = "data"             # Your current folder with 'train' and 'test'
OUTPUT_ROOT = "UnsafeNet_Ready" # Where the processed data will go
IMG_SIZE = (640, 640)           # Standard size for YOLO
SAMPLING_RATE = 5               # Save 1 frame every 5 frames (to reduce duplicates)
CONF_THRESHOLD = 0.4            # Minimum confidence to accept a detection

# --- STEP 1: LOAD MODEL SAFELY ---
def load_model():
    """
    Tries to load YOLO11n. If not found or download fails, 
    falls back to YOLOv8n which is more likely to exist.
    """
    print("--- Model Initialization ---")
    try:
        # Try loading the latest SOTA model (YOLO11 Nano)
        print("Attempting to load YOLO11n...")
        model = YOLO("yolo11n.pt") 
        print("✅ Success: YOLO11n loaded.")
        return model
    except Exception as e:
        print(f"⚠️ Warning: Could not load YOLO11n ({e}).")
        print("⬇️ Falling back to YOLOv8n (Standard)...")
        try:
            model = YOLO("yolov8n.pt")
            print("✅ Success: YOLOv8n loaded.")
            return model
        except Exception as e2:
            print(f"❌ Critical Error: Could not load any YOLO model. Check internet connection.")
            sys.exit(1)

# --- STEP 2: PROCESSING FUNCTION ---
def process_dataset(helper_model):
    # Check if input exists
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ Error: Input folder '{INPUT_ROOT}' not found!")
        print("Please make sure your 'data' folder is in the same directory as this script.")
        return

    # Subsets to process (train and test)
    subsets = ['train', 'test']

    for subset in subsets:
        subset_path = os.path.join(INPUT_ROOT, subset)
        if not os.path.exists(subset_path):
            print(f"Skipping '{subset}' (not found in data folder).")
            continue

        print(f"\n🚀 Processing subset: {subset.upper()}")
        
        # Define output paths
        img_out_dir = os.path.join(OUTPUT_ROOT, "images", subset)
        lbl_out_dir = os.path.join(OUTPUT_ROOT, "labels", subset)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)

        # Get class folders (e.g., '6_closed_panel_cover')
        # We sort them to ensure consistent processing order
        class_folders = sorted([f for f in Path(subset_path).iterdir() if f.is_dir()])
        
        if not class_folders:
            print(f"⚠️ No class folders found in {subset_path}. Check structure!")
            continue

        for class_folder in class_folders:
            folder_name = class_folder.name
            
            # --- PARSE CLASS ID ---
            # Extract the number from "6_closed_panel_cover"
            try:
                class_id = int(folder_name.split('_')[0])
            except ValueError:
                print(f"⚠️ Skipping folder '{folder_name}': Name must start with a number (e.g., '0_safe').")
                continue

            # Find video files
            video_files = list(class_folder.glob("*.mp4")) + \
                          list(class_folder.glob("*.avi")) + \
                          list(class_folder.glob("*.mov"))
            
            print(f"   📂 Class {class_id} ({folder_name}): Found {len(video_files)} videos.")

            for video_path in tqdm(video_files, desc=f"     Processing {folder_name}", leave=False):
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    continue

                video_stem = video_path.stem  # Filename without extension
                frame_idx = 0
                saved_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # --- SAMPLING & DETECTION ---
                    if frame_idx % SAMPLING_RATE == 0:
                        # 1. Resize Frame
                        frame_resized = cv2.resize(frame, IMG_SIZE)

                        # 2. Run Inference (Detect 'Person' - Class 0 in COCO)
                        # verbose=False keeps the console clean
                        results = helper_model(frame_resized, classes=[0], verbose=False, conf=CONF_THRESHOLD)
                        
                        # 3. If a person is found, save the data
                        if len(results[0].boxes) > 0:
                            # Generate unique filename
                            # Format: ClassID_VideoName_FrameIdx
                            unique_name = f"c{class_id}_{video_stem}_f{frame_idx:06d}"
                            
                            # A. Save Image
                            img_save_path = os.path.join(img_out_dir, unique_name + ".jpg")
                            cv2.imwrite(img_save_path, frame_resized)
                            
                            # B. Save Label
                            lbl_save_path = os.path.join(lbl_out_dir, unique_name + ".txt")
                            with open(lbl_save_path, "w") as f:
                                for box in results[0].boxes:
                                    # Get normalized xywh
                                    x, y, w, h = box.xywhn[0].tolist()
                                    
                                    # IMPORTANT: We save the folder's Class ID (e.g., 6)
                                    # Not the detected Class ID (0 - Person)
                                    f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                            
                            saved_count += 1
                    
                    frame_idx += 1
                cap.release()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load the helper model
    yolo_model = load_model()
    
    # 2. Run the processing
    process_dataset(yolo_model)
    
    print("\n" + "="*40)
    print("✅ DATASET PREPARATION COMPLETE!")
    print(f"   Outputs saved to: {os.path.abspath(OUTPUT_ROOT)}")
    print("   Next Step: Create your 'data.yaml' and start training.")
    print("="*40)