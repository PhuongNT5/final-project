import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

# --- CONFIGURATION ---
# 1. Path to your trained YOLO model (The "Spatial" feature extractor)
MODEL_PATH = "UnsafeNet_Training/safe_run/weights/best.pt" 

# 2. Path to your raw video dataset
VIDEO_DATASET_PATH = "data" 

# 3. Path where extracted features will be saved
FEATURES_OUTPUT_PATH = "UnsafeNet_Features" 

# --- PARAMETERS ---
# How many objects to track per frame?
# If YOLO detects 10 objects, we take top 5. If 2, we pad with zeros.
MAX_OBJECTS = 5 

# Feature vector size per frame: 
# 5 objects * 6 attributes (x1, y1, x2, y2, confidence, class_id) = 30 dimensions
FEATURE_DIM = MAX_OBJECTS * 6 

def extract_features_from_video(model, video_path):
    """
    Runs YOLO on a video and returns a sequence of feature vectors.
    Returns: numpy array of shape (Total_Frames, FEATURE_DIM)
    """
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video
            
        # Optional: Resize for faster processing (YOLO does this internally, but good for consistency)
        # frame = cv2.resize(frame, (640, 640))
        
        # 1. Run YOLO Inference
        # We use a low confidence threshold (0.2) to let the Transformer decide what is important
        results = model(frame, verbose=False, conf=0.2)
        
        frame_vector = []
        
        # 2. Process Detections
        if len(results[0].boxes) > 0:
            # Get box data: [x1, y1, x2, y2, conf, cls]
            boxes = results[0].boxes.data.cpu().numpy()
            
            # Sort by confidence (descending) to keep the most important objects first
            # boxes[:, 4] is the confidence column
            boxes = boxes[boxes[:, 4].argsort()[::-1]] 
            
            # Get image dimensions for normalization
            h_img, w_img = frame.shape[:2]
            
            # Loop through the top N objects
            for i in range(min(len(boxes), MAX_OBJECTS)):
                box = boxes[i]
                
                # Normalize coordinates to 0-1 range (Crucial for Neural Networks!)
                norm_box = [
                    box[0] / w_img,  # x1 normalized
                    box[1] / h_img,  # y1 normalized
                    box[2] / w_img,  # x2 normalized
                    box[3] / h_img,  # y2 normalized
                    box[4],          # Confidence score
                    box[5]           # Class ID
                ]
                frame_vector.extend(norm_box)
                
        # 3. Padding
        # If fewer objects than MAX_OBJECTS, fill the rest with zeros
        while len(frame_vector) < FEATURE_DIM:
            frame_vector.append(0.0)
            
        # 4. Truncating (Just in case)
        frames_data.append(frame_vector[:FEATURE_DIM])

    cap.release()
    return np.array(frames_data)

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists(FEATURES_OUTPUT_PATH):
        os.makedirs(FEATURES_OUTPUT_PATH)
        
    print(f"🚀 Loading YOLO model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Iterate through 'train' and 'test' folders
    for split in ['train', 'test']:
        split_path = os.path.join(VIDEO_DATASET_PATH, split)
        if not os.path.exists(split_path): 
            print(f"⚠️ Warning: Folder not found: {split_path}")
            continue
        
        print(f"\n📂 Processing dataset split: {split.upper()}")
        
        # Iterate through each class folder (e.g., 'safe_walkway_violation')
        for class_name in os.listdir(split_path):
            class_dir = os.path.join(split_path, class_name)
            if not os.path.isdir(class_dir): continue
            
            # Create corresponding output folder structure
            save_dir = os.path.join(FEATURES_OUTPUT_PATH, split, class_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # Find all video files
            video_files = [f for f in os.listdir(class_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
            
            # Process each video with a progress bar
            for vid_name in tqdm(video_files, desc=f"   Extracting {class_name}"):
                vid_path = os.path.join(class_dir, vid_name)
                
                # Output filename: video.mp4 -> video.npy
                save_filename = os.path.splitext(vid_name)[0] + ".npy"
                save_path = os.path.join(save_dir, save_filename)
                
                # Skip if already processed
                if os.path.exists(save_path): continue
                
                # Extract and Save
                features = extract_features_from_video(model, vid_path)
                
                if len(features) > 0:
                    np.save(save_path, features)
                else:
                    print(f"   ⚠️ Warning: No frames extracted from {vid_name}")

    print("\n✅ Feature Extraction Complete!")
    print(f"📊 Data saved to: {os.path.abspath(FEATURES_OUTPUT_PATH)}")

if __name__ == "__main__":
    main()