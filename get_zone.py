import cv2
import numpy as np

# --- CONFIG ---
VIDEO_PATH = "data/test/0_safe_walkway_violation/0_te18.mp4" 

points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.line(img, tuple(points[-2]), tuple(points[-1]), (0, 0, 255), 2)
        cv2.imshow('Define Danger Zone', img)

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if ret:
    img = frame.copy()
    print("\n-----------------------------------------------------------")
    print("INSTRUCTIONS:")
    print("1. Click 4 points to define the DANGER ROAD (The gap between machines and green walkway).")
    print("2. Make sure to cover the area where the pedestrian walks.")
    print("3. Press any key to finish and see coordinates.")
    print("-----------------------------------------------------------\n")

    cv2.imshow('Define Danger Zone', img)
    cv2.setMouseCallback('Define Danger Zone', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n✅ COPY THIS ARRAY INTO run_inference_final.py:\n")
    print(f"DANGER_ZONE_POLYGON = np.array({points}, np.int32)")
else:
    print("❌ Error: Could not read video.")