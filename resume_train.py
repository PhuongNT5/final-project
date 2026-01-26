from ultralytics import YOLO
import os

# --- CONFIGURATION ---
LAST_WEIGHT = "UnsafeNet_Training/experiment_1/weights/last.pt" 

def resume_process():
    # 1. Check if the checkpoint exists
    if not os.path.exists(LAST_WEIGHT):
        print(f"❌ Error: Could not find file at {LAST_WEIGHT}")
        print("Please check the folder name inside 'UnsafeNet_Training' and update the path in the code.")
        return

    print(f"🔄 Resuming training from: {LAST_WEIGHT}")
    
    model = YOLO(LAST_WEIGHT)

    model.train(
        resume=True,
        batch=4,          
        workers=1,        
        imgsz=640,
        device='mps',     #
        
        val=False,        # VITAL: Disable validation to avoid the NMS memory spike
        save_period=-1,   # Do not save frequent checkpoints (Saves Disk Space)
        plots=True        # Keep plotting graphs
    )

if __name__ == "__main__":
    resume_process()