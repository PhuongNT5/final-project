import torch
import torch.nn as nn
import math

class UnsafeNetTransformer(nn.Module):
    def __init__(self, input_dim=34, num_classes=8, d_model=128, nhead=4, num_layers=3, dropout=0.3):
        super(UnsafeNetTransformer, self).__init__()
        
        self.d_model = d_model
        
        # 1. Input Projection: Increased d_model to 128 for richer feature extraction
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # 2. Transformer Encoder: Increased layers to 3
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 3. Classifier Head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, src):
        # src shape: [Batch, Sequence_Length, Features] -> [Batch, 30, 34]
        
        # Embed and add position info
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Pass through Transformer
        output = self.transformer_encoder(src) # Shape: [Batch, 30, 128]
        
        # --- CRITICAL UPGRADE: Global Average Pooling ---
        # Instead of taking just the last frame (which might be noisy),
        # we take the AVERAGE of all 30 frames.
        output = output.mean(dim=1) # Shape: [Batch, 128]
        
        output = self.fc(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)