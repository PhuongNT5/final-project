from ultralytics import YOLO
import os

# --- CONFIGURATION ---
MODEL_NAME = "yolo11n.pt"   # Use Nano (fastest) or 'yolo11s.pt' (more accurate)
DATA_CONFIG = "data.yaml"   # Path to your data config
EPOCHS = 50                 # 50-100 is usually sufficient for this dataset size
IMG_SIZE = 640              # Standard input size
BATCH_SIZE = 16             # Reduce to 8 or 4 if you get "Out of Memory" errors
PROJECT_NAME = "UnsafeNet_Training" # Folder where results will be saved

def start_training():
    # 1. Load Pre-trained Model
    print(f"🔄 Loading model {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    # 2. Start Training
    print("🚀 Starting training process...")
    try:
        results = model.train(
            data=DATA_CONFIG,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name="experiment_1",  # Name of this specific run
            patience=10,          # Stop early if no improvement after 10 epochs
            save=True,            # Save checkpoints
            device='mps',             # Use '0' for GPU or 'cpu' for CPU
            verbose=True,
            plots=True            # Automatically generate training graphs
        )
        print("\n✅ Training Complete!")
        print(f"📊 Results saved to: {os.path.join(PROJECT_NAME, 'experiment_1')}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Tip: If you run out of memory, try reducing BATCH_SIZE to 8.")

if __name__ == "__main__":
    start_training() 