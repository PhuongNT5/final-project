import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLO_Temporal_Transformer(nn.Module):
    def __init__(
        self,
        num_classes=8,
        nhead=8,
        num_layers=4,
        dropout=0.5,
        yolo_weights: str = 'yolo11s.pt',
        feature_layer_idx: int = -3,
        embed_dim_mult: int = 2,
    ):
        super(YOLO_Temporal_Transformer, self).__init__()
        yolo = YOLO(yolo_weights)
        self.yolo_model = yolo.model

        self.feature_layer_idx = feature_layer_idx
        self.embed_dim_mult = embed_dim_mult
        self.num_classes = num_classes

        # Lazy init (built after first forward when feat_channels is known)
        self.embed_dim = None
        self.projection = None
        self.norm = None
        self.transformer = None
        self.fc = None

        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        self.adapter = nn.AdaptiveAvgPool2d((1, 1))

    def _extract_features(self, x):
        features = None
        layers = self.yolo_model.model
        idx = self.feature_layer_idx if self.feature_layer_idx >= 0 else len(layers) + self.feature_layer_idx

        def hook(module, inp, out):
            nonlocal features
            features = out

        handle = layers[idx].register_forward_hook(hook)
        _ = self.yolo_model(x)
        handle.remove()
        return features

    def _lazy_build(self, feat_channels: int):
        if self.projection is not None:
            return

        self.embed_dim = int(feat_channels * self.embed_dim_mult)
        self.projection = nn.Sequential(
            nn.Linear(feat_channels, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU()
        )

        self.norm = nn.LayerNorm(self.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.nhead,
            dim_feedforward=self.embed_dim * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.num_classes)
        )

        print(f" [Model Build] spatial channels {feat_channels}, embed_dim set to {self.embed_dim}")

    def get_backbone_parameters(self):
        params = []
        layers = self.yolo_model.model
        idx = self.feature_layer_idx if self.feature_layer_idx >= 0 else len(layers) + self.feature_layer_idx
        for i, m in enumerate(layers):
            if i > idx:
                break
            params += list(m.parameters())
        return params

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        feat = self._extract_features(x)
        if feat is None:
            raise RuntimeError("Failed to extract YOLO features (hook returned None). Check feature_layer_idx.")

        feat = self.adapter(feat).flatten(1)
        self._lazy_build(int(feat.shape[1]))

        feat = self.projection(feat)
        feat = feat.view(b, t, -1)
        feat = self.norm(feat)
        feat = self.transformer(feat)
        out, _ = feat.max(dim=1)
        return self.fc(out)