import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformer import UnsafeNetTransformer
from collections import deque
import os

# --- CONFIGURATION ---
# 1. Paths to your models
YOLO_PATH = "UnsafeNet_Training/safe_run/weights/best.pt"
TRANSFORMER_PATH = "Transformer_Training/best_transformer.pth"

# 2. Input Video (Set to 0 for Webcam, or provide a file path)
VIDEO_SOURCE = "data/test/1_unauthorized_intervention/1_te1.mp4" 

# 3. Class Names (MUST match the training order exactly)
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

# 4. SAFE ZONE COORDINATES (Spatial Logic)
# IMPORTANT: You must update these points to match YOUR video's floor markings.
# Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
# Example: A rectangle in the middle of a 1920x1080 frame
SAFE_ZONE_POLY = np.array([[200, 400], [800, 400], [800, 900], [200, 900]], np.int32)

class HybridSystem:
    def __init__(self):
        # Detect device (Mac M1/M2 -> mps, NVIDIA -> cuda, else -> cpu)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"🔥 System running on: {self.device}")
        
        # Load YOLO
        print(f"   Loading YOLO from: {YOLO_PATH}")
        self.yolo = YOLO(YOLO_PATH)
        
        # Load Transformer
        print(f"   Loading Transformer from: {TRANSFORMER_PATH}")
        self.transformer = UnsafeNetTransformer(input_dim=30, num_classes=len(CLASSES)).to(self.device)
        self.transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=self.device))
        self.transformer.eval() # Set to evaluation mode
        
        # Buffer to store the last 30 frames of features
        self.sequence_buffer = deque(maxlen=30) 
        self.feature_dim = 30 # 5 objects * 6 attributes

    def extract_features(self, results, frame_shape):
        """
        Converts YOLO detections into a feature vector [x, y, w, h, conf, class]
        Matches the format used in 'extract_features.py'
        """
        frame_vector = []
        h, w = frame_shape[:2]
        
        if len(results[0].boxes) > 0:
            # Get boxes and sort by confidence (high to low)
            boxes = results[0].boxes.data.cpu().numpy()
            boxes = boxes[boxes[:, 4].argsort()[::-1]] 
            
            # Take top 5 objects
            for i in range(min(len(boxes), 5)): 
                box = boxes[i]
                # Normalize coordinates (0-1)
                norm_box = [
                    box[0]/w, box[1]/h, box[2]/w, box[3]/h, # x1, y1, x2, y2
                    box[4],   # Confidence
                    box[5]    # Class ID
                ]
                frame_vector.extend(norm_box)
        
        # Padding (Fill with zeros if fewer than 5 objects)
        while len(frame_vector) < self.feature_dim:
            frame_vector.append(0.0)
            
        return frame_vector[:self.feature_dim]

    def check_spatial_logic(self, box, ai_predicted_class):
        """
        HYBRID LOGIC: Combines AI prediction with Geometric Rules.
        If the AI is confused about 'walkway', we use the polygon to decide.
        """
        # We only apply this logic to Walkway classes because AI often confuses them
        if "walkway" not in ai_predicted_class:
            return ai_predicted_class 
            
        x1, y1, x2, y2 = box
        
        # calculate the "foot" point (bottom center of the person)
        foot_point = (int((x1+x2)/2), int(y2))
        
        # Check if foot is inside the Safe Zone Polygon
        # pointPolygonTest returns +1 (inside), -1 (outside), 0 (on edge)
        is_inside = cv2.pointPolygonTest(SAFE_ZONE_POLY, foot_point, False) >= 0
        
        if is_inside:
            return "4_safe_walkway" # Force Correction: Safe
        else:
            return "0_safe_walkway_violation" # Force Correction: Violation

    def run(self, video_path):
        if not os.path.exists(video_path) and video_path != 0:
            print(f"❌ Error: Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        
        # Setup Video Writer to save output
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter('final_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        
        print("🚀 Starting Inference... Press 'q' to stop.")
        
        current_action = "Initializing..."
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Run YOLO (Spatial Detection)
            results = self.yolo(frame, verbose=False, conf=0.4)
            
            # 2. Extract Features & Update Temporal Buffer
            features = self.extract_features(results, frame.shape)
            self.sequence_buffer.append(features)
            
            # 3. Run Transformer (Sequence Analysis)
            # Only predict if we have enough history (30 frames)
            if len(self.sequence_buffer) == 30:
                # Convert buffer to tensor: Shape (1, 30, 30)
                input_seq = torch.FloatTensor([list(self.sequence_buffer)]).to(self.device)
                
                with torch.no_grad():
                    output = self.transformer(input_seq)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    pred_idx = torch.argmax(probs).item()
                    conf = probs[0][pred_idx].item()
                    
                    # Update global action if confidence is decent
                    if conf > 0.4: 
                        current_action = CLASSES[pred_idx]
            
            # 4. Visualization
            # Draw Safe Zone
            cv2.polylines(frame, [SAFE_ZONE_POLY], True, (0, 255, 255), 2)
            cv2.putText(frame, "SAFE ZONE", (SAFE_ZONE_POLY[0][0], SAFE_ZONE_POLY[0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Draw Detections
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # --- APPLY HYBRID LOGIC ---
                    final_decision = self.check_spatial_logic((x1, y1, x2, y2), current_action)
                    
                    # Color Logic: Red for Danger, Green for Safe
                    if "violation" in final_decision or "unauthorized" in final_decision:
                        color = (0, 0, 255) # Red
                    elif "safe" in final_decision:
                        color = (0, 255, 0) # Green
                    else:
                        color = (255, 165, 0) # Orange (Warning/Other)
                    
                    # Draw Box and Label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = final_decision.split('_', 1)[1] if '_' in final_decision else final_decision
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # System Status Overlay
            cv2.rectangle(frame, (0, 0), (400, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Action: {current_action}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show and Save
            cv2.imshow("UnsafeNet Hybrid System", frame)
            out.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("✅ Finished! Output saved to 'final_output.mp4'")

if __name__ == "__main__":
    system = HybridSystem()
    system.run(VIDEO_SOURCE)