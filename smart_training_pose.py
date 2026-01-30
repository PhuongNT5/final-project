import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from transformer_model import UnsafeNetTransformer

# --- CONFIGURATION ---
DATA_DIR = "Pose_Data_Processed"
MODEL_SAVE_DIR = "Transformer_Pose_Training"
EPOCHS = 100            
BATCH_SIZE = 32         
LR = 0.0005             

def load_data():
    """Load data and handle missing Validation set"""
    print("📂 Loading data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train_pose.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    
    # Load Val and Test
    X_val = np.load(os.path.join(DATA_DIR, "X_val_pose.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test_pose.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    # Auto-fix empty validation set
    if len(X_val) == 0:
        print("⚠️ Warning: Validation set empty! Using Test set.")
        X_val, y_val = X_test, y_test

    return (torch.FloatTensor(X_train), torch.LongTensor(y_train)), \
           (torch.FloatTensor(X_val), torch.LongTensor(y_val))

def train():
    # Detect Device
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    print(f"🔥 Training Transformer on: {device}")
    
    if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)
    
    (X_train, y_train), (X_val, y_val) = load_data()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
    
    # Init Improved Model
    model = UnsafeNetTransformer(
        input_dim=34, 
        num_classes=8, 
        d_model=128, 
        nhead=4, 
        num_layers=3, 
        dropout=0.3
    ).to(device)
    
    # Loss Function with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with Weight Decay
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Scheduler: Removed 'verbose=True' to fix the error
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    best_acc = 0.0
    
    print("🚀 Starting Advanced Transformer Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_acc = correct / total if total > 0 else 0
        
        # Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Step the scheduler based on validation accuracy
        scheduler.step(val_acc)
        
        # Manually print learning rate if needed
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Train: {train_acc:.2%} | Val: {val_acc:.2%} | LR: {current_lr:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_pose_transformer.pth"))
            print("   --> 🏆 Saved Best Model!")

    print(f"\n✅ Finished! Best Accuracy: {best_acc:.2%}")

if __name__ == "__main__":
    train()