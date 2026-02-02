import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
from transformer_model import UnsafeNetTransformer
import sys
import os
import csv
import time

# --- CONFIGURATION ---
VIDEO_PATH = "data/test/0_safe_walkway_violation/0_te3.mp4" 
OUTPUT_VIDEO_PATH = "output/result_final.mp4"
OUTPUT_LOG_PATH = "output/safety_log.csv"
MODEL_PATH = "Transformer_Pose_Training/best_pose_transformer.pth"
YOLO_MODEL = "yolo11n-pose.pt"

# 1. SAFE ZONE (From your previous steps)
SAFE_ZONE_POLYGON = np.array([
    [1691, 1072], [1799, 1071], [1511, 346], [1488, 345]
], np.int32)

# 2. DANGER ZONE (PASTE YOUR NEW COORDINATES HERE)
# Draw this zone on the "Road" gap.
DANGER_ZONE_POLYGON = np.array([[1160, 1071], [1703, 1070], [1491, 343], [1417, 340]], np.int32)

WORK_ZONE_LIMIT_X = 1450 

def load_models():
    print("⏳ Loading Models...")
    yolo = YOLO(YOLO_MODEL)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    transformer = UnsafeNetTransformer(input_dim=34, num_classes=8, d_model=128, nhead=4, num_layers=3).to(device)
    transformer.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    transformer.eval()
    return yolo, transformer, device

def get_smoothed_action(history_queue):
    if not history_queue: return "UNKNOWN"
    counts = Counter(history_queue)
    return counts.most_common(1)[0][0]

def check_zone_status(keypoints, width, height, safe_poly, danger_poly):
    """
    Logic mới chặt chẽ hơn:
    1. Kiểm tra Safe/Danger với độ dung sai (Tolerance).
    2. Nếu không thuộc 2 vùng trên -> Kiểm tra tọa độ X để xem có phải Work Zone không.
    3. Nếu không phải Work Zone -> Mặc định là DANGER (để bắt nhạy hơn).
    """
    ankles = keypoints[15:17]
    if len(ankles) == 0: return "UNKNOWN", (0,0)
    
    foot_x = int(np.mean(ankles[:, 0]) * width)
    foot_y = int(np.mean(ankles[:, 1]) * height)
    
    # Dung sai (Tolerance): +10 pixel (cho phép lệch ra ngoài một chút vẫn tính)
    dist_safe = cv2.pointPolygonTest(safe_poly, (foot_x, foot_y), True)
    dist_danger = cv2.pointPolygonTest(danger_poly, (foot_x, foot_y), True)

    # 1. Ưu tiên Safe Zone (Nếu chân chạm hoặc rất gần Safe Zone)
    if dist_safe >= -15: 
        return "SAFE", (foot_x, foot_y)

    # 2. Kiểm tra Danger Zone
    if dist_danger >= -15:
        return "DANGER", (foot_x, foot_y)
        
    # 3. Logic "Authorized": Chỉ khi thực sự nằm bên trái đường
    if foot_x < WORK_ZONE_LIMIT_X:
        return "WORK_AREA", (foot_x, foot_y)
    
    # 4. Nếu lơ lửng ở giữa hoặc lỗi -> Mặc định cảnh báo nguy hiểm
    return "DANGER", (foot_x, foot_y)

def run():
    yolo, transformer, device = load_models()
    cap = cv2.VideoCapture(VIDEO_PATH)
    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = int(cap.get(5))
    
    if not os.path.exists("output"): os.makedirs("output")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    # Logging
    log_file = open(OUTPUT_LOG_PATH, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["Frame", "Timestamp", "ID", "Action", "Zone", "Decision"])
    
    pose_buffers = {}
    prediction_history = {}

    print("🚀 Running Fix V6...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # --- VISUALIZATION ---
        overlay = frame.copy()
        # Vẽ Safe Zone (Xanh)
        cv2.fillPoly(overlay, [SAFE_ZONE_POLYGON], (0, 255, 0))
        # Vẽ Danger Zone (Đỏ)
        cv2.fillPoly(overlay, [DANGER_ZONE_POLYGON], (0, 0, 255))
        # Vẽ ranh giới Authorized (Cam) - Dạng đường kẻ dọc
        cv2.line(overlay, (WORK_ZONE_LIMIT_X, 0), (WORK_ZONE_LIMIT_X, h), (255, 165, 0), 2)
        
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [SAFE_ZONE_POLYGON], True, (0, 255, 0), 2)
        cv2.polylines(frame, [DANGER_ZONE_POLYGON], True, (0, 0, 255), 2)

        results = yolo.track(frame, persist=True, verbose=False)
        
        if results[0].boxes.id is not None and results[0].keypoints is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints_xyn = results[0].keypoints.xyn.cpu().numpy()
            
            for track_id, kp_norm in zip(track_ids, keypoints_xyn):
                flat_kp = kp_norm.flatten()
                
                # --- ACTION RECOGNITION ---
                if track_id not in pose_buffers:
                    pose_buffers[track_id] = []
                    prediction_history[track_id] = deque(maxlen=15)
                
                pose_buffers[track_id].append(flat_kp)
                if len(pose_buffers[track_id]) > 30: pose_buffers[track_id].pop(0)
                
                if len(pose_buffers[track_id]) == 30:
                    input_tensor = torch.FloatTensor([pose_buffers[track_id]]).to(device)
                    with torch.no_grad():
                        outputs = transformer(input_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        _, pred_idx = torch.max(probs, 1)
                        classes = ["VIOLATION_WALK", "INTERVENTION", "OPEN_PANEL", "FORKLIFT", 
                                   "SAFE_WALK", "INTERVENTION", "CLOSED_PANEL", "CARRYING"]
                        prediction_history[track_id].append(classes[pred_idx.item()])

                smoothed_action = get_smoothed_action(prediction_history[track_id])
                
                # --- SPATIAL CHECK ---
                zone_status, foot_pt = check_zone_status(kp_norm, w, h, SAFE_ZONE_POLYGON, DANGER_ZONE_POLYGON)

                # --- DECISION LOGIC ---
                final_status = "UNKNOWN"
                box_color = (128, 128, 128)

                if "FORKLIFT" in smoothed_action:
                    final_status = "VEHICLE"
                    box_color = (0, 255, 255) # Vàng
                elif zone_status == "DANGER":
                    final_status = "VIOLATION"
                    box_color = (0, 0, 255) # Đỏ
                elif zone_status == "SAFE":
                    final_status = "SAFE"
                    box_color = (0, 255, 0) # Xanh
                elif zone_status == "WORK_AREA":
                    final_status = "AUTHORIZED"
                    box_color = (255, 165, 0) # Cam
                else:
                    final_status = "WARNING" # Trường hợp lọt khe
                    box_color = (0, 0, 255)

                # --- DRAW & LOG ---
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                writer.writerow([int(cap.get(cv2.CAP_PROP_POS_FRAMES)), round(timestamp,2), track_id, smoothed_action, zone_status, final_status])

                # Vẽ điểm chốt (QUAN TRỌNG: Giúp bạn thấy máy đang bắt điểm nào)
                cv2.circle(frame, foot_pt, 8, (255, 255, 255), -1) # Chấm trắng viền
                cv2.circle(frame, foot_pt, 5, box_color, -1)       # Chấm màu theo trạng thái

                head_x, head_y = int(kp_norm[0, 0] * w), max(30, int(kp_norm[0, 1] * h))
                cv2.putText(frame, final_status, (head_x, head_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        out.write(frame)
        cv2.imshow("V6 Fix", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    out.release()
    log_file.close()
    cv2.destroyAllWindows()
    print("✅ Done.")

if __name__ == "__main__":
    run()