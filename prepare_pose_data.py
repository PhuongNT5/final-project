import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# --- CONFIGURATION ---
DATA_ROOT = "data"  # Folder containing train/val/test
SAVE_DIR = "Pose_Data_Processed" # Where to save the new .npy files
SEQUENCE_LENGTH = 30
IMG_SIZE = 640

# Load YOLO-Pose Model (It will auto-download 'yolov8n-pose.pt')
print("⏳ Loading YOLO-Pose model...")
model = YOLO('yolo11n-pose.pt')

# Define Classes (Must match your folder names)
CLASSES = [
    "0_safe_walkway_violation", 
    "1_unauthorized_intervention", 
    "2_opened_panel_cover",
    "3_carrying_overload_with_forklift", 
    "4_safe_walkway", 
    "5_authorized_intervention",
    "6_closed_panel_cover", "7_safe_carrying"
]

def extract_pose_sequence(video_path):
    """
    Reads a video and extracts a sequence of Keypoints (Skeletons).
    Returns: Array of shape (30, 34) -> 30 frames, 17 points * 2 (x,y)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret: break
        
        # Run YOLO Pose
        results = model(frame, verbose=False)
        
        # --- FIXED LOGIC ---
        # Check if keypoints exist AND at least one person is detected
        if (results[0].keypoints is not None and 
            results[0].keypoints.xyn.shape[0] > 0): 
            
            # Get the person with the highest confidence (index 0)
            # Shape is (17, 2) -> 17 Keypoints (x, y) normalized
            keypoints = results[0].keypoints.xyn[0].cpu().numpy() 
            flat_keypoints = keypoints.flatten() # Flatten to vector of size 34
        else:
            # No person detected? Fill with zeros
            flat_keypoints = np.zeros(34)
            
        frames.append(flat_keypoints)
    
    cap.release()
    
    # Padding (If video is shorter than 30 frames, add zeros)
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(np.zeros(34))
        
    # Ensure exactly 30 frames
    return np.array(frames[:SEQUENCE_LENGTH])

def process_dataset(split_name):
    """Process train/val/test folders"""
    print(f"\n🚀 Processing {split_name} data...")
    X_data = []
    y_data = []
    
    split_dir = os.path.join(DATA_ROOT, split_name)
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir): continue
        
        video_files = [f for f in os.listdir(class_dir) if f.endswith(('.mp4', '.avi'))]
        print(f"   Category: {class_name} ({len(video_files)} videos)")
        
        for vid in video_files:
            video_path = os.path.join(class_dir, vid)
            try:
                sequence = extract_pose_sequence(video_path)
                X_data.append(sequence)
                y_data.append(class_idx)
            except Exception as e:
                print(f"⚠️ Error processing video {vid}: {e}")
                continue
            
    return np.array(X_data), np.array(y_data)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    # 1. Process Train
    X_train, y_train = process_dataset("train")
    np.save(f"{SAVE_DIR}/X_train_pose.npy", X_train)
    np.save(f"{SAVE_DIR}/y_train.npy", y_train)
    
    # 2. Process Val
    X_val, y_val = process_dataset("val")
    np.save(f"{SAVE_DIR}/X_val_pose.npy", X_val)
    np.save(f"{SAVE_DIR}/y_val.npy", y_val)

    # 3. Process Test
    X_test, y_test = process_dataset("test")
    np.save(f"{SAVE_DIR}/X_test_pose.npy", X_test)
    np.save(f"{SAVE_DIR}/y_test.npy", y_test)
    
    print(f"\n✅ DONE! Pose data saved in '{SAVE_DIR}'")
    print(f"Sample Shape: {X_train.shape} (Batch, 30 frames, 34 features)")