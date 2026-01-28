import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformer import UnsafeNetTransformer
from collections import deque, Counter
import os

# --- TUNING PARAMETERS ---
YOLO_CONF_THRESHOLD = 0.5  # Increase if seeing "ghost" boxes
SMOOTHING_WINDOW = 15      # Higher = More stable labels, but slower reaction (10-30 is good)
SAFE_ZONE_POLY = np.array([[300, 400], [900, 400], [900, 900], [300, 900]], np.int32) # <--- ADJUST THIS!

# Paths
YOLO_PATH = "UnsafeNet_Training/safe_run/weights/best.pt"
TRANSFORMER_PATH = "Transformer_Training/best_transformer.pth"
VIDEO_SOURCE = "data/test/4_safe_walkway/4_te5.mp4" # Or 0 for webcam

CLASSES = [
    "0_safe_walkway_violation", "1_unauthorized_intervention", "2_opened_panel_cover",
    "3_carrying_overload_with_forklift", "4_safe_walkway", "5_authorized_intervention",
    "6_closed_panel_cover", "7_safe_carrying"
]

class HybridSystem:
    def __init__(self):
        device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device_type)
        print(f"🔥 System running on: {self.device}")
        
        self.yolo = YOLO(YOLO_PATH)
        self.transformer = UnsafeNetTransformer(input_dim=30, num_classes=len(CLASSES)).to(self.device)
        self.transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=self.device))
        self.transformer.eval()
        
        # Buffer for input features (Frames)
        self.sequence_buffer = deque(maxlen=30) 
        
        # Buffer for output predictions (Voting mechanism)
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)

    def extract_features(self, results, frame_shape):
        """Extracts normalized features for the Transformer"""
        frame_vector = []
        h, w = frame_shape[:2]
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.data.cpu().numpy()
            boxes = boxes[boxes[:, 4].argsort()[::-1]]
            for i in range(min(len(boxes), 5)): 
                box = boxes[i]
                frame_vector.extend([box[0]/w, box[1]/h, box[2]/w, box[3]/h, box[4], box[5]])
        
        while len(frame_vector) < 30: frame_vector.append(0.0)
        return frame_vector[:30]

    def get_smoothed_prediction(self, new_prediction):
        """
        Takes the new prediction, adds it to history, and returns the MOST COMMON prediction.
        This removes flickering.
        """
        self.prediction_history.append(new_prediction)
        # Count occurrences in history (Majority Vote)
        counts = Counter(self.prediction_history)
        most_common = counts.most_common(1)[0][0]
        return most_common

    def check_spatial_logic(self, box, ai_class):
        """Forces Walkway rules based on Geometry"""
        if "walkway" in ai_class:
            x1, y1, x2, y2 = box
            foot_point = (int((x1+x2)/2), int(y2))
            is_inside = cv2.pointPolygonTest(SAFE_ZONE_POLY, foot_point, False) >= 0
            return "4_safe_walkway" if is_inside else "0_safe_walkway_violation"
        return ai_class

    def run(self):
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        width = int(cap.get(3))
        height = int(cap.get(4))
        out = cv2.VideoWriter('optimized_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        
        print("🚀 Processing Video...")
        current_action = "Initializing..."

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 1. YOLO
            results = self.yolo(frame, verbose=False, conf=YOLO_CONF_THRESHOLD)
            
            # 2. Transformer
            features = self.extract_features(results, frame.shape)
            self.sequence_buffer.append(features)
            
            if len(self.sequence_buffer) == 30:
                input_seq = torch.FloatTensor([list(self.sequence_buffer)]).to(self.device)
                with torch.no_grad():
                    output = self.transformer(input_seq)
                    pred_idx = torch.argmax(output, dim=1).item()
                    raw_action = CLASSES[pred_idx]
                    
                    # --- APPLY SMOOTHING ---
                    current_action = self.get_smoothed_prediction(raw_action)
            
            # 3. Visualization
            cv2.polylines(frame, [SAFE_ZONE_POLY], True, (255, 255, 0), 2) # Draw Zone
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # --- APPLY SPATIAL LOGIC ---
                    final_decision = self.check_spatial_logic((x1, y1, x2, y2), current_action)
                    
                    # Coloring
                    color = (0, 255, 0) if "safe" in final_decision and "violation" not in final_decision else (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = final_decision.split('_', 1)[1] if '_' in final_decision else final_decision
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            out.write(frame)
            cv2.imshow("Optimized System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("✅ Finished. Check 'optimized_output.mp4'")

if __name__ == "__main__":
    HybridSystem().run()