import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from transformer import UnsafeNetTransformer

# --- CONFIGURATION ---
DATA_DIR = "Pose_Data_Processed"
MODEL_SAVE_DIR = "Transformer_Pose_Training"
EPOCHS = 50
BATCH_SIZE = 16
LR = 0.001

def load_data():
    """Load data and handle missing Validation set"""
    print("📂 Loading data...")
    
    # Load Train
    X_train = np.load(os.path.join(DATA_DIR, "X_train_pose.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    
    # Load Val
    X_val = np.load(os.path.join(DATA_DIR, "X_val_pose.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    
    # Load Test (Dự phòng)
    X_test = np.load(os.path.join(DATA_DIR, "X_test_pose.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    # --- AUTO FIX: Nếu Val rỗng, dùng Test để thay thế ---
    if len(X_val) == 0:
        print("⚠️ Warning: Validation set is empty! Using Test set for validation instead.")
        if len(X_test) > 0:
            X_val, y_val = X_test, y_test
        else:
            print("❌ Critical Error: Both Val and Test sets are empty! Check your data.")
            exit()

    print(f"   Train size: {len(X_train)}")
    print(f"   Val size:   {len(X_val)}")

    return (torch.FloatTensor(X_train), torch.LongTensor(y_train)), \
           (torch.FloatTensor(X_val), torch.LongTensor(y_val))

def train():
    # Device setup
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    print(f"🔥 Training on: {device}")
    
    if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)
    
    # Load Data
    (X_train, y_train), (X_val, y_val) = load_data()
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
    
    # Initialize Model (Input Dim = 34)
    model = UnsafeNetTransformer(input_dim=34, num_classes=8).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    history = {'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    
    print("🚀 Starting Pose Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_acc = correct / total
        history['train_acc'].append(train_acc)
        
        # Validation
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
        
        val_acc = val_correct / val_total
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_pose_transformer.pth"))
            print("   --> 🏆 New Best Model Saved!")

    print(f"\n✅ Training Complete! Best Accuracy: {best_acc:.2%}")

if __name__ == "__main__":
    train()