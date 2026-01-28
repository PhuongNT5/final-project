import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# --- IMPORT YOUR ARCHITECTURE ---
# Make sure transformer_model.py is in the same folder
from transformer import UnsafeNetTransformer

# --- CONFIGURATION ---
FEATURES_DIR = "UnsafeNet_Features"    # Folder containing .npy files
OUTPUT_DIR = "Transformer_Training"    # Folder to save results
SEQ_LEN = 30                           # Fixed sequence length (e.g., 1 second)
FEATURE_DIM = 30                       # 5 objects * 6 attributes
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001

# Define your classes explicitly (Must match folder names)
CLASSES = [
    "0_safe_walkway_violation",
    "1_unauthorized_intervention",
    "2_opened_panel_cover",
    "3_carrying_overload_with_forklift",
    "4_safe_walkway",
    "5_authorized_intervention",
    "6_closed_panel_cover",
    "7_safe_carrying"
]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}

# --- DATASET CLASS ---
class UnsafeNetDataset(Dataset):
    def __init__(self, root_dir, split="train", seq_len=30):
        self.root_dir = os.path.join(root_dir, split)
        self.seq_len = seq_len
        self.samples = []
        
        # Scan folder to load file paths and labels
        if not os.path.exists(self.root_dir):
            print(f"⚠️ Warning: {split} folder not found in {root_dir}")
            return

        for cls_name in os.listdir(self.root_dir):
            cls_folder = os.path.join(self.root_dir, cls_name)
            if os.path.isdir(cls_folder) and cls_name in CLASS_TO_IDX:
                label = CLASS_TO_IDX[cls_name]
                for file_name in os.listdir(cls_folder):
                    if file_name.endswith('.npy'):
                        self.samples.append((os.path.join(cls_folder, file_name), label))
        
        print(f"✅ Loaded {len(self.samples)} samples for {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # Load features: Shape (Frames, 30)
        features = np.load(path)
        
        # --- PREPROCESSING (PADDING/TRUNCATING) ---
        # We need fixed sequence length for batching
        current_len = features.shape[0]
        
        if current_len >= self.seq_len:
            # If too long, take the middle part (often most relevant) or just the beginning
            # Here we take the first SEQ_LEN frames
            features = features[:self.seq_len, :]
        else:
            # If too short, pad with zeros
            padding = np.zeros((self.seq_len - current_len, FEATURE_DIM))
            features = np.vstack((features, padding))
            
        # Convert to Tensor
        return torch.FloatTensor(features), torch.tensor(label, dtype=torch.long)

# --- VISUALIZATION FUNCTION ---
def plot_training_history(history, save_dir):
    # 1. Loss & Accuracy Curves
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='orange')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='green')
    plt.plot(history['val_acc'], label='Val Acc', color='red')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    print("📊 Saved training curves graph.")

def plot_confusion_matrix(model, val_loader, device, save_dir):
    """
    Generates a Confusion Matrix and a Text Report.
    Fixes the 'ValueError' by explicitly defining expected labels.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # 1. Get Predictions
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # --- THE FIX IS HERE ---
    # We explicitly pass 'labels=range(len(CLASSES))' to tell sklearn 
    # that we expect 8 classes, even if some are missing in the test set.
    
    # 2. Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(CLASSES)))
    
    # Plot Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground Truth (Actual)')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 3. Generate Classification Report
    try:
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=CLASSES, 
            labels=range(len(CLASSES)), # <--- CRITICAL FIX
            zero_division=0             # <--- Prevents error if a class has 0 samples
        )
        
        # Save report to text file
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
            
        print("📊 Saved Confusion Matrix and Classification Report.")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not generate text report: {e}")

# --- MAIN TRAINING LOOP ---
def train():
    # Setup Directories
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔥 Training on: {device}")

    # Load Data
    train_dataset = UnsafeNetDataset(FEATURES_DIR, split="train", seq_len=SEQ_LEN)
    val_dataset = UnsafeNetDataset(FEATURES_DIR, split="test", seq_len=SEQ_LEN)
    
    if len(train_dataset) == 0:
        print("❌ Error: No training data found. Did you run extract_features.py?")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Init Model
    model = UnsafeNetTransformer(
        input_dim=FEATURE_DIM, 
        num_classes=len(CLASSES),
        d_model=64,
        nhead=4,
        num_layers=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # History tracking
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0

    print(f"🚀 Starting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100 * val_correct / val_total
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        print(f"   Done. Train Loss: {epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%")

        # Save Best Model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_transformer.pth'))

    # Save Last Model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'last_transformer.pth'))
    
    # --- VISUALIZE & FINISH ---
    print("\n📈 Generating visualizations...")
    plot_training_history(history, OUTPUT_DIR)
    plot_confusion_matrix(model, val_loader, device, OUTPUT_DIR)
    
    print("\n" + "="*40)
    print(f"✅ Training Complete!")
    print(f"🏆 Best Validation Accuracy: {best_acc:.2f}%")
    print(f"📂 Results saved in: {os.path.abspath(OUTPUT_DIR)}")
    print("="*40)

if __name__ == "__main__":
    train()