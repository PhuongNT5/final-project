import torch
import torch.nn as nn


class SpatioTemporalTransformerHead(nn.Module):
    """Transformer head for tokenized pre-extracted features.

    Input:
      x: (B, T, S, C_in)
        - T: frames per clip
        - S: spatial tokens per frame (e.g., pool*pool)
        - C_in: per-token feature dim (e.g., 768)

    The model flattens tokens to (B, T*S, D), adds factorized learned positional
    embeddings (time + spatial), applies a TransformerEncoder, pools over tokens,
    and classifies.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 8,
        model_dim: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.5,
        max_t: int = 32,
        max_s: int = 64,
        use_cls_token: bool = True,
        pool: str = 'cls',
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.model_dim = int(model_dim)
        self.max_t = int(max_t)
        self.max_s = int(max_s)
        self.use_cls_token = bool(use_cls_token)
        self.pool = str(pool)

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

        # Factorized learned positional encodings
        self.time_pos = nn.Embedding(self.max_t, self.model_dim)
        self.spatial_pos = nn.Embedding(self.max_s, self.model_dim)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))
            self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        else:
            self.cls_token = None
            self.cls_pos = None

        # Ensure nhead divides model_dim
        if self.model_dim % int(nhead) != 0:
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
        self.transformer = nn.TransformerEncoder(enc, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.model_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.model_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected (B,T,S,C) input, got {tuple(x.shape)}")
        b, t, s, _ = x.shape
        if t > self.max_t:
            raise ValueError(f"T={t} exceeds max_t={self.max_t}. Increase --max_t.")
        if s > self.max_s:
            raise ValueError(f"S={s} exceeds max_s={self.max_s}. Increase --max_s.")

        x = self.input_proj(x)  # (B,T,S,D)

        # Build factorized (T,S,D) positional tensor
        t_idx = torch.arange(t, device=x.device)
        s_idx = torch.arange(s, device=x.device)
        pos = self.time_pos(t_idx)[:, None, :] + self.spatial_pos(s_idx)[None, :, :]  # (T,S,D)
        pos = pos.reshape(1, t * s, self.model_dim)

        x = x.reshape(b, t * s, self.model_dim) + pos

        if self.use_cls_token:
            cls = self.cls_token.expand(b, -1, -1)
            cls_pos = self.cls_pos.expand(b, -1, -1)
            x = torch.cat([cls + cls_pos, x], dim=1)  # (B, 1+TS, D)

        x = self.norm(x)
        x = self.transformer(x)

        if self.use_cls_token and self.pool == 'cls':
            x = x[:, 0]
        else:
            x, _ = x.max(dim=1)  # pool over tokens
        return self.fc(x)


class FactorizedSTHead(nn.Module):
    """Factorized Spatial-then-Temporal Transformer head.

    Instead of flattening T*S tokens (quadratic attention cost), this head:
      1. Spatial stage: per-frame attention over S spatial tokens → per-frame summary (B*T, D)
      2. Temporal stage: attention over T frame summaries → video-level representation (B, D)
      3. Classifier: linear head → (B, num_classes)

    This scales linearly with T and S instead of quadratically with T*S,
    making it feasible for T=32+ on CPU with small datasets.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 8,
        model_dim: int = 512,
        nhead: int = 8,
        spatial_layers: int = 2,
        temporal_layers: int = 2,
        dropout: float = 0.5,
        max_t: int = 64,
        max_s: int = 64,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.model_dim = int(model_dim)
        self.max_t = int(max_t)
        self.max_s = int(max_s)

        # Input projection
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

        # Ensure nhead divides model_dim
        nh = int(nhead)
        if self.model_dim % nh != 0:
            for h in range(nh, 0, -1):
                if self.model_dim % h == 0:
                    nh = h
                    break
        self.nhead = nh

        # Spatial stage: per-frame attention over S tokens
        self.spatial_pos = nn.Embedding(self.max_s, self.model_dim)
        self.spatial_cls = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        spatial_enc = nn.TransformerEncoderLayer(
            d_model=self.model_dim, nhead=self.nhead,
            dim_feedforward=self.model_dim * 4, dropout=dropout, batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_enc, num_layers=int(spatial_layers))
        self.spatial_norm = nn.LayerNorm(self.model_dim)

        # Temporal stage: attention over T frame summaries
        self.temporal_pos = nn.Embedding(self.max_t, self.model_dim)
        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        temporal_enc = nn.TransformerEncoderLayer(
            d_model=self.model_dim, nhead=self.nhead,
            dim_feedforward=self.model_dim * 4, dropout=dropout, batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_enc, num_layers=int(temporal_layers))
        self.temporal_norm = nn.LayerNorm(self.model_dim)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(self.model_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, S, C)
        if x.dim() != 4:
            raise ValueError(f"Expected (B,T,S,C), got {tuple(x.shape)}")
        b, t, s, _ = x.shape

        x = self.input_proj(x)  # (B, T, S, D)

        # --- Spatial stage: process each frame independently ---
        x = x.reshape(b * t, s, self.model_dim)  # (BT, S, D)
        s_pos = self.spatial_pos(torch.arange(s, device=x.device)).unsqueeze(0)  # (1, S, D)
        x = x + s_pos
        s_cls = self.spatial_cls.expand(b * t, -1, -1)
        x = torch.cat([s_cls, x], dim=1)  # (BT, 1+S, D)
        x = self.spatial_norm(x)
        x = self.spatial_transformer(x)
        x = x[:, 0]  # CLS token output: (BT, D)

        # --- Temporal stage: attend over frame summaries ---
        x = x.reshape(b, t, self.model_dim)  # (B, T, D)
        t_pos = self.temporal_pos(torch.arange(t, device=x.device)).unsqueeze(0)  # (1, T, D)
        x = x + t_pos
        t_cls = self.temporal_cls.expand(b, -1, -1)
        x = torch.cat([t_cls, x], dim=1)  # (B, 1+T, D)
        x = self.temporal_norm(x)
        x = self.temporal_transformer(x)
        x = x[:, 0]  # CLS output: (B, D)

        return self.fc(x)
