import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformer import UnsafeNetTransformer
from collections import deque, Counter
import os

# --- CONFIGURATION ---
YOLO_PATH = "UnsafeNet_Training/safe_run/weights/best.pt"
TRANSFORMER_PATH = "Transformer_Training/best_transformer.pth"
VIDEO_SOURCE = "data/test/0_safe_walkway_violation/0_te9.mp4" # Or 0 for webcam

# TUNE THESE PARAMETERS
YOLO_CONF_THRESHOLD = 0.4
SMOOTHING_WINDOW = 15 

# *** CRITICAL: PASTE YOUR COORDINATES FROM STEP 1 HERE ***
SAFE_ZONE_POLY = np.array([
[1700, 1073],
[1794, 1073],
[1510, 346],
[1495, 358],
[1746, 926],
[1840, 933],
[1809, 873],
[1718, 859],
[1651, 678],
[1705, 685],
[1686, 651],
[1637, 644],
[1579, 498],
[1571, 485],
[1599, 487],
[1605, 502],
], np.int32)

CLASSES = [
    "0_safe_walkway_violation", 
    "1_unauthorized_intervention", 
    "2_opened_panel_cover",
    "3_carrying_overload_with_forklift", 
    "4_safe_walkway", 
    "5_authorized_intervention",
    "6_closed_panel_cover", 
    "7_safe_carrying"
]

class HybridSystem:
    def __init__(self):
        # Device Check
        if torch.backends.mps.is_available(): self.device = torch.device("mps")
        elif torch.cuda.is_available(): self.device = torch.device("cuda")
        else: self.device = torch.device("cpu")
            
        print(f"🔥 System running on: {self.device}")
        
        # Load Models
        self.yolo = YOLO(YOLO_PATH)
        self.transformer = UnsafeNetTransformer(input_dim=30, num_classes=len(CLASSES)).to(self.device)
        self.transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=self.device))
        self.transformer.eval()
        
        # Buffers
        self.sequence_buffer = deque(maxlen=30) 
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)

    def extract_features(self, results, frame_shape):
        """Convert YOLO box to Normalized Feature Vector"""
        frame_vector = []
        h, w = frame_shape[:2]
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.data.cpu().numpy()
            boxes = boxes[boxes[:, 4].argsort()[::-1]] # Sort by confidence
            
            for i in range(min(len(boxes), 5)): 
                box = boxes[i]
                norm_box = [box[0]/w, box[1]/h, box[2]/w, box[3]/h, box[4], box[5]]
                frame_vector.extend(norm_box)
        
        while len(frame_vector) < 30: 
            frame_vector.append(0.0)
            
        return frame_vector[:30]

    def get_smoothed_prediction(self, new_prediction):
        """Majority Voting to prevent label flickering"""
        self.prediction_history.append(new_prediction)
        return Counter(self.prediction_history).most_common(1)[0][0]

    def check_spatial_logic(self, box, ai_class):
        """
        HYBRID LOGIC:
        If the AI detects a person, we check if their feet are inside the Polygon.
        Inside = Safe (Green). Outside = Violation (Red).
        This overrides the Transformer's guess.
        """
        if "walkway" in ai_class:
            x1, y1, x2, y2 = box
            # Calculate foot position (bottom center)
            foot_point = (int((x1+x2)/2), int(y2))
            
            # Check if point is inside polygon (>= 0 means inside or on edge)
            is_inside = cv2.pointPolygonTest(SAFE_ZONE_POLY, foot_point, False) >= 0
            
            if is_inside:
                return "4_safe_walkway"         # FORCE SAFE
            else:
                return "0_safe_walkway_violation" # FORCE VIOLATION
        
        return ai_class # Trust AI for other classes (Forklift, Panels, etc.)

    def run(self):
        if VIDEO_SOURCE != 0 and not os.path.exists(VIDEO_SOURCE):
            print(f"❌ Error: Video file not found: {VIDEO_SOURCE}")
            return

        cap = cv2.VideoCapture(VIDEO_SOURCE)
        width = int(cap.get(3))
        height = int(cap.get(4))
        out = cv2.VideoWriter('final_system_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        
        print("🚀 System Started. Press 'q' to stop.")
        current_action = "Initializing..."
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 1. YOLO Detection
            results = self.yolo(frame, verbose=False, conf=YOLO_CONF_THRESHOLD)
            
            # 2. Transformer Analysis
            features = self.extract_features(results, frame.shape)
            self.sequence_buffer.append(features)
            
            if len(self.sequence_buffer) == 30:
                input_seq = torch.FloatTensor([list(self.sequence_buffer)]).to(self.device)
                with torch.no_grad():
                    output = self.transformer(input_seq)
                    pred_idx = torch.argmax(output, dim=1).item()
                    raw_action = CLASSES[pred_idx]
                    
                    # Apply Smoothing
                    current_action = self.get_smoothed_prediction(raw_action)
            
            # 3. Visualization
            # Draw Safe Zone (Yellow Polygon)
            cv2.polylines(frame, [SAFE_ZONE_POLY], True, (0, 255, 255), 2)
            cv2.putText(frame, "SAFE ZONE", (SAFE_ZONE_POLY[0][0], SAFE_ZONE_POLY[0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # --- APPLY HYBRID LOGIC ---
                    final_decision = self.check_spatial_logic((x1, y1, x2, y2), current_action)
                    
                    # Color Coding
                    if "violation" in final_decision or "unauthorized" in final_decision:
                        color = (0, 0, 255) # Red
                    elif "safe" in final_decision:
                        color = (0, 255, 0) # Green
                    else:
                        color = (255, 165, 0) # Orange
                    
                    # Draw Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Fix for IndexError (The crash you saw earlier)
                    label = final_decision.split('_', 1)[1] if '_' in final_decision else final_decision
                    
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out.write(frame)
            cv2.imshow("Hybrid System", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("✅ Finished! Output saved to 'final_system_output.mp4'")

if __name__ == "__main__":
    HybridSystem().run()