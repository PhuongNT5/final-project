from ultralytics import YOLO
import os

# --- CONFIGURATION ---
# Path to your best trained model
MODEL_PATH = "UnsafeNet_Training/safe_run/weights/best.pt" 

# Path to the video you want to test
# Replace 'video_test_01.mp4' with an actual video file name from your test folder
VIDEO_PATH = "data/test/6_closed_panel_cover/6_te1.mp4" 

def run_inference():
    # 1. Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Please check the path to your 'best.pt' file.")
        return

    # 2. Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video not found at {VIDEO_PATH}")
        print("Please update VIDEO_PATH in the code to point to a valid video file.")
        return

    print(f"🎥 Running inference on: {VIDEO_PATH}...")
    
    # 3. Load the trained model
    model = YOLO(MODEL_PATH)
    
    # 4. Run prediction and save the result
    # conf=0.4 means only show detections with >40% confidence
    results = model.predict(
        source=VIDEO_PATH,
        save=True,             # Save the output video
        conf=0.4,              # Confidence threshold
        project="UnsafeNet_Inference", # Output folder
        name="test_result",    # Sub-folder name
        exist_ok=True          # Overwrite if exists
    )
    
    print("\n✅ Inference Complete!")
    print(f"   Output video saved in: {os.path.join('UnsafeNet_Inference', 'test_result')}")

if __name__ == "__main__":
    run_inference()