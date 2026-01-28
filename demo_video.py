import cv2
import os
import time
from ultralytics import YOLO

# --- CONFIGURATION SECTION ---
# 1. Path to your best trained model
MODEL_PATH = "UnsafeNet_Training/safe_run/weights/best.pt" 

# 2. Path to the input video file (YOU MUST CHANGE THIS)
INPUT_VIDEO = "data/test/6_closed_panel_cover/6_te1.mp4" 

# 3. Path where the result video will be saved
OUTPUT_VIDEO = "demo_results/demo_result.mp4"

# 4. Confidence threshold (0.0 to 1.0)
# Detections below this confidence will be ignored.
CONF_THRESHOLD = 0.4 

def run_demo():
    # --- STEP 1: VALIDATION ---
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at: {MODEL_PATH}")
        print("Please check the path to your .pt file.")
        return

    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ Error: Input video not found at: {INPUT_VIDEO}")
        print("Please update the 'INPUT_VIDEO' variable in the script.")
        return

    # --- STEP 2: INITIALIZATION ---
    print(f"🔄 Loading model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return

    print(f"🎥 Opening video: {INPUT_VIDEO}")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize Video Writer
    # 'mp4v' is a standard codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"🚀 Processing started! Total frames: {total_frames}")
    print("   (Press 'Ctrl+C' in terminal to stop early if needed)")

    frame_count = 0
    start_time = time.time()

    # --- STEP 3: PROCESSING LOOP ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video
        
        frame_count += 1

        # Run YOLO Inference
        # verbose=False keeps the terminal clean
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)

        # Draw Bounding Boxes
        # .plot() automatically draws boxes, labels, and confidence scores
        annotated_frame = results[0].plot()

        # Add an overlay info text
        info_text = f"UnsafeNet Demo | Frame: {frame_count}/{total_frames}"
        cv2.putText(annotated_frame, info_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save frame to output video
        out.write(annotated_frame)

        # Print progress every 50 frames
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")

    # --- STEP 4: CLEANUP ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    duration = end_time - start_time
    avg_fps = frame_count / duration if duration > 0 else 0

    print("\n" + "="*40)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print(f"📁 Output saved to: {os.path.abspath(OUTPUT_VIDEO)}")
    print(f"⏱️  Time taken: {duration:.2f} seconds")
    print(f"⚡ Average Processing Speed: {avg_fps:.1f} FPS")
    print("="*40)

if __name__ == "__main__":
    run_demo()