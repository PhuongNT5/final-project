import cv2
import numpy as np

# --- CONFIG ---
VIDEO_PATH = "data/test/0_safe_walkway_violation/0_te18.mp4" 

points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Point selected: [{x}, {y}]")
        
        # Draw a red circle where you clicked
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        
        # Connect to the previous point with a line
        if len(points) > 1:
            cv2.line(img, tuple(points[-2]), tuple(points[-1]), (0, 255, 255), 2)
            
        cv2.imshow('Define Safe Zone', img)

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Error: Could not read video. Check the path!")
else:
    img = frame.copy()
    print("\n-----------------------------------------------------------")
    print("INSTRUCTIONS:")
    print("1. Click points along the border of the GREEN WALKWAY.")
    print("2. Start Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left.")
    print("3. Press 'q' when finished to generate your code.")
    print("-----------------------------------------------------------\n")

    cv2.imshow('Define Safe Zone', img)
    cv2.setMouseCallback('Define Safe Zone', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Format the output for easy copy-pasting
    print("\n✅ COPY THIS CODE BLOCK BELOW INTO YOUR MAIN SCRIPT:\n")
    print(f"SAFE_ZONE_POLYGON = np.array({points}, np.int32)\n")