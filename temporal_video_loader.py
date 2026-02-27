import cv2
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class UnsafeNetTemporalDataset(Dataset):
    def __init__(self, root_dir, subset='train', sequence_length=16, transform=None):
        """
        Args:
            root_dir (string): Path to the 'data' folder.
            subset (string): 'train' or 'test'.
            sequence_length (int): T (number of frames per sequence).
            transform (callable): Image augmentations/normalization.
        """
        self.subset_path = os.path.join(root_dir, subset)
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []
        self.subset = subset
        self._failed_reads = 0
        self._total_reads = 0
        
        # 1. Identify all class folders (e.g., '0_safe_walkway_violation')
        if not os.path.exists(self.subset_path):
            raise FileNotFoundError(f"Directory not found: {self.subset_path}")
            
        self.class_folders = sorted([f for f in os.listdir(self.subset_path) 
                                   if os.path.isdir(os.path.join(self.subset_path, f))])
        
        # 2. Build the list of (video_path, label)
        for folder in self.class_folders:
            # Extract class ID from folder name (e.g., '0' from '0_...')
            label = int(folder.split('_')[0])
            class_dir = os.path.join(self.subset_path, folder)
            
            for video_file in sorted(os.listdir(class_dir)):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(class_dir, video_file)
                    self.samples.append((video_path, label))
        
        print(f"Loaded {len(self.samples)} videos for {subset} subset.")

    def _get_frames(self, video_path):
        self._total_reads += 1
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._failed_reads += 1
            return torch.zeros((self.sequence_length, 3, 224, 224))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            self._failed_reads += 1
            return torch.zeros((self.sequence_length, 3, 224, 224))

        if self.subset == 'train':
            # random contiguous sampling if long enough, else pad
            if total_frames >= self.sequence_length:
                start = random.randint(0, total_frames - self.sequence_length)
                indices = list(range(start, start + self.sequence_length))
            else:
                indices = list(range(total_frames))
        else:
            # deterministic uniform sampling
            indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int).tolist()

        frames = []
        for idx in indices:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
            except Exception:
                break
        cap.release()

        if len(frames) == 0:
            self._failed_reads += 1

        # pad with last frame if needed
        while len(frames) < self.sequence_length:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
        return torch.stack(frames)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        sequence = self._get_frames(video_path)
        return sequence, label

# --- Execution Block ---

if __name__ == '__main__':
    training_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # Suggest 224 for faster training, 640 for final accuracy
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_path = 'data' 
    train_set = UnsafeNetTemporalDataset(dataset_path, subset='train', transform=training_transforms)
    # Using num_workers=2 for speed, now safe inside this block
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

    for batch_sequences, batch_labels in train_loader:
        print(f"Verified Shape: {batch_sequences.shape}")
        break