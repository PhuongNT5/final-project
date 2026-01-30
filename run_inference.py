import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
from transformer_model import UnsafeNetTransformer
import sys

# --- CONFIGURATION ---
VIDEO_PATH = "data/test/0_safe_walkway_violation/0_te18.mp4" 
MODEL_PATH = "Transformer_Pose_Training/best_pose_transformer.pth"
YOLO_MODEL = "yolo11n-pose.pt"

SAFE_ZONE_POLYGON = np.array([[1700, 1073], [1790, 1070], [1513, 355], [1498, 374]], np.int32)

# --- SETTINGS ---
CONFIDENCE_THRESHOLD = 0.60 
HISTORY_LENGTH = 10 

ACTION_GROUPS = {
    0: "WALKING", 4: "WALKING",
    7: "CARRYING", 3: "FORKLIFT",
    1: "INTERVENTION", 2: "PANEL_OPEN", 5: "INTERVENTION", 6: "PANEL_CLOSED"
}

def load_models():
    print("⏳ Loading Models...")
    yolo = YOLO(YOLO_MODEL)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    transformer = UnsafeNetTransformer(input_dim=34, num_classes=8, d_model=128, nhead=4, num_layers=3).to(device)
    transformer.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    transformer.eval()
    return yolo, transformer, device

def is_in_safe_zone(keypoints, width, height):
    # Check feet position (Ankles: 15, 16)
    ankles = keypoints[15:17]
    if len(ankles) == 0: return False
    
    x = int(np.mean(ankles[:, 0]) * width)
    y = int(np.mean(ankles[:, 1]) * height)
    
    # Check if inside polygon (Result >= 0 means inside or on edge)
    return cv2.pointPolygonTest(SAFE_ZONE_POLYGON, (x, y), False) >= 0

def run():
    yolo, transformer, device = load_models()
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened(): sys.exit("❌ Error: Video not found.")
    
    # Auto-detect resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Video Resolution: {frame_width}x{frame_height}")

    pose_buffers = {}
    prediction_history = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. DRAW SAFE ZONE OVERLAY (Transparent Green)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [SAFE_ZONE_POLYGON], (0, 255, 0)) # Fill Green
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)   # Blend 30% opacity
        
        # 2. RUN YOLO
        results = yolo.track(frame, persist=True, verbose=False)
        
        if results[0].boxes.id is not None and results[0].keypoints is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints_list = results[0].keypoints.xyn.cpu().numpy()
            
            for track_id, kp in zip(track_ids, keypoints_list):
                flat_kp = kp.flatten()
                
                # Manage Buffers
                if track_id not in pose_buffers:
                    pose_buffers[track_id] = []
                    prediction_history[track_id] = deque(maxlen=HISTORY_LENGTH)
                
                pose_buffers[track_id].append(flat_kp)
                if len(pose_buffers[track_id]) > 30: pose_buffers[track_id].pop(0)
                
                # Predict
                current_label = "Analyzing..."
                if len(pose_buffers[track_id]) == 30:
                    input_tensor = torch.FloatTensor([pose_buffers[track_id]]).to(device)
                    with torch.no_grad():
                        outputs = transformer(input_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        max_prob, pred_idx = torch.max(probs, 1)
                        
                        if max_prob.item() > CONFIDENCE_THRESHOLD:
                            action = ACTION_GROUPS.get(pred_idx.item(), "UNKNOWN")
                            prediction_history[track_id].append(action)

                if prediction_history[track_id]:
                    current_label = Counter(prediction_history[track_id]).most_common(1)[0][0]

                # Check Logic
                in_zone = is_in_safe_zone(kp, frame_width, frame_height)
                
                status_text = f"ID {track_id}: {current_label}"
                text_color = (255, 255, 255)
                bg_color = (0, 200, 0) # Green (Safe)

                if in_zone:
                    status_text = f"SAFE: {current_label}"
                    bg_color = (0, 200, 0) # Green

                # 2. If OUTSIDE the zone -> VIOLATION (Unless it's a Forklift)
                else:
                    if current_label == "FORKLIFT":
                         status_text = "VEHICLE: ALLOWED"
                         bg_color = (255, 200, 0) # Yellow/Orange
                    else:
                         # Flag ANY pedestrian outside as a violation
                         status_text = "VIOLATION: WRONG LANE!"
                         bg_color = (0, 0, 255) # Red

                # Draw Label
                head_x = int(kp[0, 0] * frame_width)
                head_y = max(30, int(kp[0, 1] * frame_height))
                
                (w, h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (head_x, head_y - 30), (head_x + w + 10, head_y), bg_color, -1)
                cv2.putText(frame, status_text, (head_x + 5, head_y - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

                # Draw Skeleton
                for i in range(17):
                    x_pt = int(kp[i, 0] * frame_width)
                    y_pt = int(kp[i, 1] * frame_height)
                    if x_pt > 0 and y_pt > 0:
                        cv2.circle(frame, (x_pt, y_pt), 3, (255, 0, 0), -1)

        cv2.imshow("UnsafeNet V5 (Visualizer)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()