import cv2
import numpy as np

# --- CONFIG ---
VIDEO_PATH = "data/test/0_safe_walkway_violation/0_te18.mp4" 

SAFE_ZONE_POLYGON = np.array([[1691, 1072], [1799, 1071], [1511, 346], [1488, 345]], np.int32)


cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if ret:
    # Draw the zone filled
    overlay = frame.copy()
    cv2.fillPoly(overlay, [SAFE_ZONE_POLYGON], (0, 255, 0))
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Draw the border thick
    cv2.polylines(frame, [SAFE_ZONE_POLYGON], True, (0, 255, 0), 3)
    
    cv2.imwrite("debug_zone_view.jpg", frame)
    print("✅ Saved 'debug_zone_view.jpg'. Open this image to check if the zone covers the YELLOW lines.")