"""
Hierarchical two-stage classification head for Unsafe-Net behaviour detection.

PROPOSED (thesis contribution):
  HierarchicalSTHead — hybrid YOLOv11 (spatial) + Transformer (temporal)
    Uses pre-extracted YOLOv11 spatial features as input tokens, then applies
    a shared Transformer encoder with factorised spatiotemporal positional
    encoding, followed by two task-specific heads:
      Stage 1 — Scenario classifier (4 classes: walkway, intervention, panel, forklift)
      Stage 2 — Safety classifier (binary: unsafe / safe)
    The final 8-class prediction is the combination of both stages.

BASELINE (paper reference, NOT re-implemented):
  Unsafe-Net (Kurnaz et al.) — YOLOv4 + ConvLSTM → 95.81% on 4 unsafe classes.
  We compare our proposed YOLOv11 + Transformer results against their reported
  numbers directly.  No ConvLSTM is trained in this project.

Mapping (8-class ↔ scenario + safety):
  Scenario 0 (walkway):      class 0 = unsafe,  class 4 = safe
  Scenario 1 (intervention): class 1 = unsafe,  class 5 = safe
  Scenario 2 (panel):        class 2 = unsafe,  class 6 = safe
  Scenario 3 (forklift):     class 3 = unsafe,  class 7 = safe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Label mapping helpers ------------------------------------------------

CLASS_TO_SCENARIO = {0: 0, 4: 0, 1: 1, 5: 1, 2: 2, 6: 2, 3: 3, 7: 3}
CLASS_TO_SAFE     = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
SCENARIO_SAFE_TO_CLASS = {
    (0, 0): 0, (0, 1): 4,
    (1, 0): 1, (1, 1): 5,
    (2, 0): 2, (2, 1): 6,
    (3, 0): 3, (3, 1): 7,
}

CLASS_NAMES_8 = [
    'safe_walkway_violation', 'unauthorized_intervention',
    'opened_panel_cover', 'carrying_overload',
    'safe_walkway', 'authorized_intervention',
    'closed_panel_cover', 'safe_carrying',
]
SCENARIO_NAMES = ['walkway', 'intervention', 'panel', 'forklift']
SAFETY_NAMES = ['unsafe', 'safe']

# Vectorised lookup tensors (created once, moved to device as needed)
_SCENARIO_MAP = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
_SAFETY_MAP   = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)


def labels_8_to_scenario(labels_8: torch.Tensor) -> torch.Tensor:
    """Convert 8-class labels → 4-scenario labels (0-3)."""
    return _SCENARIO_MAP.to(labels_8.device)[labels_8]


def labels_8_to_safety(labels_8: torch.Tensor) -> torch.Tensor:
    """Convert 8-class labels → binary safety labels (0=unsafe, 1=safe)."""
    return _SAFETY_MAP.to(labels_8.device)[labels_8]


def combine_predictions(scenario_pred: torch.Tensor,
                        safety_pred: torch.Tensor) -> torch.Tensor:
    """Combine scenario (0-3) and safety (0=unsafe,1=safe) → 8-class label.

    unsafe class = scenario_pred  (0,1,2,3)
    safe   class = scenario_pred + 4  (4,5,6,7)
    """
    return scenario_pred + safety_pred * 4


# ---- Proposed architecture ------------------------------------------------

class HierarchicalSTHead(nn.Module):
    """Hierarchical Spatiotemporal Transformer Head  (YOLOv11 + Transformer).

    This is the **proposed thesis architecture**.  It receives pre-extracted
    YOLOv11s spatial feature tokens (B, T, S, C) and:

      1. Projects tokens to ``model_dim`` via a linear bottleneck.
      2. Adds factorised learned positional encodings (time + space).
      3. Prepends a learnable [CLS] token.
      4. Runs a standard Transformer encoder (joint spatiotemporal attention).
      5. Reads the [CLS] output as the video-level representation.
      6. Feeds it into **two parallel heads**:
         - Scenario head → 4-class (walkway / intervention / panel / forklift)
         - Safety head   → 2-class (unsafe / safe)
      7. At inference the two predictions are combined into an 8-class label.

    Training requires a **custom loop** that back-propagates two losses:
        loss = α · CE_scenario  +  (1-α) · CE_safety
    See ``run_experiment.py`` for the integration.

    Why hierarchical?
      • Scenario classification separates visually distinct environments
        (forklift area vs. walkway vs. panel vs. intervention zone).
      • Safety classification is a much simpler binary decision once the
        scenario context is established.
      • Decomposition reduces the effective number of classes each head must
        distinguish, improving learning on the small Unsafe-Net dataset (566 train).
    """

    def __init__(
        self,
        in_dim: int,
        model_dim: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.5,
        max_t: int = 32,
        max_s: int = 64,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.model_dim = int(model_dim)
        self.max_t = int(max_t)
        self.max_s = int(max_s)

        # ---- Input projection (bottleneck) ----
        self.input_proj = (
            nn.Identity()
            if self.in_dim == self.model_dim
            else nn.Sequential(
                nn.Linear(self.in_dim, self.model_dim),
                nn.LayerNorm(self.model_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        )

        # ---- Factorised positional embeddings ----
        self.time_pos    = nn.Embedding(self.max_t, self.model_dim)
        self.spatial_pos = nn.Embedding(self.max_s, self.model_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        self.cls_pos     = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

        # ---- Transformer encoder ----
        nh = int(nhead)
        if self.model_dim % nh != 0:
            for h in range(nh, 0, -1):
                if self.model_dim % h == 0:
                    nh = h
                    break
        self.nhead = nh

        self.pre_norm = nn.LayerNorm(self.model_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.nhead,
            dim_feedforward=self.model_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,          # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=int(num_layers),
        )
        self.post_norm = nn.LayerNorm(self.model_dim)

        # ---- Scenario head: 4 classes ----
        self.scenario_head = nn.Sequential(
            nn.Linear(self.model_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4),
        )

        # ---- Safety head: 2 classes (unsafe / safe) ----
        self.safety_head = nn.Sequential(
            nn.Linear(self.model_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder: (B, T, S, C) → (B, model_dim)  via [CLS] token."""
        if x.dim() != 4:
            raise ValueError(f"Expected (B,T,S,C), got {tuple(x.shape)}")
        b, t, s, _ = x.shape

        x = self.input_proj(x)                         # (B, T, S, D)

        # Factorised positional encoding: pos[t,s] = time_pos[t] + spatial_pos[s]
        t_idx = torch.arange(t, device=x.device)
        s_idx = torch.arange(s, device=x.device)
        pos = (self.time_pos(t_idx)[:, None, :]         # (T, 1, D)
               + self.spatial_pos(s_idx)[None, :, :])    # (1, S, D)
        pos = pos.reshape(1, t * s, self.model_dim)      # (1, T*S, D)

        x = x.reshape(b, t * s, self.model_dim) + pos    # (B, T*S, D)

        # Prepend [CLS] token
        cls = self.cls_token.expand(b, -1, -1) + self.cls_pos.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)                   # (B, 1+T*S, D)

        x = self.pre_norm(x)
        x = self.transformer(x)
        cls_out = self.post_norm(x[:, 0])                 # (B, D)
        return cls_out

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """Forward pass.

        Returns
        -------
        scenario_logits : (B, 4)  — raw logits for scenario classification
        safety_logits   : (B, 2)  — raw logits for safety classification
        combined_pred   : (B,)    — argmax 8-class prediction (no gradient)
        """
        feat = self._encode(x)                            # (B, D)
        scenario_logits = self.scenario_head(feat)         # (B, 4)
        safety_logits   = self.safety_head(feat)           # (B, 2)

        # Discrete 8-class prediction for evaluation (detached from graph)
        with torch.no_grad():
            scenario_pred = scenario_logits.argmax(dim=1)
            safety_pred   = safety_logits.argmax(dim=1)
            combined_pred = combine_predictions(scenario_pred, safety_pred)

        return scenario_logits, safety_logits, combined_pred

    def predict_8class(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: returns discrete 8-class predictions."""
        _, _, combined = self.forward(x)
        return combined
