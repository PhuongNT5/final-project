import torch
import torch.nn as nn
from typing import Optional


class TemporalTransformerHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int = 8,
        model_dim: Optional[int] = None,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_dim = int(in_dim)

        # Compress very wide feature vectors (e.g., pooled grids) to a transformer-friendly width.
        # If model_dim is not provided, pick a safe default.
        if model_dim is None:
            model_dim = 512 if self.in_dim >= 1024 else self.in_dim
        self.model_dim = int(model_dim)

        self.input_proj = (
            nn.Identity()
            if self.in_dim == self.model_dim
            else nn.Sequential(
                nn.Linear(self.in_dim, self.model_dim),
                nn.LayerNorm(self.model_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        )

        self.proj = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(self.model_dim)

        # Ensure nhead divides model_dim.
        if self.model_dim % int(nhead) != 0:
            # pick the largest head count <= nhead that divides model_dim
            for h in range(int(nhead), 0, -1):
                if self.model_dim % h == 0:
                    nhead = h
                    break
        self.nhead = int(nhead)

        enc = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.nhead,
            dim_feedforward=self.model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(self.model_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B,T,C)
        x = self.input_proj(x)
        x = self.proj(x)
        x = self.norm(x)
        x = self.transformer(x)
        x, _ = x.max(dim=1)
        return self.fc(x)
