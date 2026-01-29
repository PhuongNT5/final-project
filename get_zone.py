import cv2
import numpy as np

# --- CONFIGURATION ---
# Replace with your actual video path
VIDEO_PATH = "data/test/0_safe_walkway_violation/0_te2.mp4" 

points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[{x}, {y}],")
        points.append([x, y])
        
        # Draw the point on the image
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Get Coordinates (Press any key to exit)', img)

# Read the first frame
cap = cv2.VideoCapture(VIDEO_PATH)
ret, img = cap.read()
cap.release()

if ret:
    # Resize is optional, but keep it 1.0 to get accurate pixel coords
    img = cv2.resize(img, (0,0), fx=1.0, fy=1.0) 
    
    cv2.imshow('Get Coordinates (Press any key to exit)', img)
    cv2.setMouseCallback('Get Coordinates (Press any key to exit)', click_event)
    
    print("\n👉 INSTRUCTIONS: Click the 4 corners of the Safe Walkway.")
    print("The output format will appear below:\n")
    print("np.array([")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("], np.int32)")
else:
    print(f"❌ Error: Could not read video at {VIDEO_PATH}")