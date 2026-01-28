from ultralytics import YOLO
import torch
import os

# --- CONFIGURATION FOR STABILITY ---
MODEL_NAME = "yolo11n.pt"      # Using Nano model (lightweight)
DATA_CONFIG = "data.yaml"      # Path to your data config
EPOCHS = 50                    # Total training cycles
IMG_SIZE = 640                 # Input image size
BATCH_SIZE = 8                 # Reduced to 8 to prevent RAM overflow (Use 4 if it still crashes)
PROJECT_NAME = "UnsafeNet_Training"

def start_training():
    # 1. Setup Device (Apple Silicon GPU or CPU)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"⚙️  Running on device: {device}")

    # 2. Load the Model
    print(f"🔄 Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    # 3. Start "Safe Mode" Training
    print("🚀 Starting training ...")
    
    try:
        model.train(
            data=DATA_CONFIG,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name="safe_run",    # Name of the folder for this run
            
            # --- CRASH PREVENTION SETTINGS ---
            device=device,      # Force use of MPS (Metal Performance Shaders)
            workers=1,          # Use only 1 CPU worker to load data (Saves significant RAM)
            exist_ok=True,      # Overwrite existing folder (Prevents creating duplicate folders)
            
            # --- DISK SPACE SAVING ---
            save=True,          # Save the model
            save_period=-1,     # IMPORTANT: Do NOT save a checkpoint every epoch (Saves GBs of disk space)
            
            # --- MEMORY SPIKE PREVENTION ---
            val=False,          # IMPORTANT: Disable validation during training to avoid NMS memory spikes
            plots=True          # Still generate loss graphs
        )
        
        print("\n✅ Training Complete!")
        
        # 4. Run Validation (Only once at the end)
        print("📊 Running final validation on the best model...")
        metrics = model.val()
        print(f"Final mAP50-95: {metrics.box.map}")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Tip: If the error is 'RuntimeError', your dataset might be corrupted. Run the data cleaner script.")
        print("Tip: If the error is 'zsh: killed', try reducing BATCH_SIZE to 4.")

if __name__ == "__main__":
    start_training()