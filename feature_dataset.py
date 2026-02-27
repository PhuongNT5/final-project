import os
import torch
from torch.utils.data import Dataset


class PreExtractedFeatureDataset(Dataset):
    def __init__(self, feature_pt_path: str):
        if not os.path.isfile(feature_pt_path):
            raise FileNotFoundError(feature_pt_path)
        payload = torch.load(feature_pt_path, map_location='cpu')
        self.payload = payload
        self.features = payload['features']
        self.labels = payload['labels']
        self.layout = payload.get('layout', 'vector')

        # Supported:
        # - (N,T,C)            => layout 'vector'
        # - (N,T,S,C)          => layout 'tokens'
        if self.features.dim() not in (3, 4):
            raise ValueError(f"Unsupported feature tensor shape: {tuple(self.features.shape)}")

    def __len__(self):
        return int(self.labels.numel())

    def __getitem__(self, idx):
        x = self.features[idx]
        y = int(self.labels[idx])
        return x, y
