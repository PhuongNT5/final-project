from ultralytics import YOLO
import torch
import os

# --- CONFIGURATION FOR 2ND RUN ---
# Option: Change to "yolo11s.pt" (Small) if you want higher accuracy 
# and trust your Mac's RAM. Otherwise, keep "yolo11n.pt" (Nano).
MODEL_NAME = "yolo11n.pt"      
DATA_CONFIG = "data.yaml"      
EPOCHS = 100                   # Increased to 100
IMG_SIZE = 640                 
BATCH_SIZE = 16              
PROJECT_NAME = "UnsafeNet_Training"

def start_training():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"⚙️  Running 2nd Training on: {device}")

    # Load fresh model
    print(f"🔄 Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    print("🚀 Starting IMPROVED training (100 Epochs + Augmentation)...")
    
    try:
        model.train(
            data=DATA_CONFIG,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name="experiment_2", # New folder name
            
            # --- STRATEGY 2: DATA AUGMENTATION (New!) ---
            # These settings make the problem "harder" so the model learns "better"
            fliplr=0.5,     # 50% chance to flip image horizontally (Left <-> Right)
            scale=0.5,      # Random zoom between 50% and 150%
            mosaic=1.0,     # Mosaic augmentation (combining 4 images) - Crucial for mAP!
            mixup=0.1,      # Mix 2 images together (10% chance)
            
            # --- TRAINING CONTROLS ---
            patience=20,    # Stop early if no improvement for 20 epochs
            optimizer='auto',
            
            # --- SAFETY SETTINGS (Keep these to prevent crash) ---
            device=device,
            workers=1,
            exist_ok=True,
            save=True,
            save_period=-1, # Only save best.pt and last.pt (Save Disk Space)
            val=True,      # Disable validation during training (Save RAM)
            plots=True
        )
        
        print("\n✅ Training Complete!")
        
        # Validate at the very end
        print("📊 Running final validation...")
        metrics = model.val()
        print(f"Final mAP50-95: {metrics.box.map}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Tip: If 'zsh: killed' occurs again, change BATCH_SIZE to 4.")

if __name__ == "__main__":
    start_training()