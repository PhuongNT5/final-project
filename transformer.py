import torch
import torch.nn as nn

class UnsafeNetTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        """
        Args:
            input_dim: Size of input vector (e.g., 30 for 5 objects * 6 coords)
            num_classes: Number of actions to detect (e.g., 8)
            d_model: Internal dimension of Transformer (Hidden size)
            nhead: Number of attention heads
            num_layers: Number of Transformer Encoder layers
        """
        super(UnsafeNetTransformer, self).__init__()
        
        # 1. Embedding Layer
        # Projects the raw YOLO features (30-dim) to the Transformer dimension (64-dim)
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        # Helps the model understand the order of frames (t1 comes before t2).
        # We use a learnable parameter here for simplicity.
        # Max sequence length is set to 1000 frames (adjustable).
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model))
        
        # 3. Transformer Encoder
        # The core component that learns temporal relationships.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout, 
            batch_first=True # Expected input: (Batch, Seq, Feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classification Head
        # Converts the Transformer output into class probabilities
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32), # Intermediate layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes) # Final output layer
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch_Size, Sequence_Length, Input_Dim)
        """
        # Step 1: Embedding
        x = self.embedding(x) # -> (Batch, Seq, d_model)
        
        # Step 2: Add Positional Encoding
        # We slice pos_encoder to match the current sequence length
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Step 3: Pass through Transformer
        x = self.transformer_encoder(x) # -> (Batch, Seq, d_model)
        
        # Step 4: Pooling (Aggregation)
        # We need to condense the sequence of information into a single prediction.
        # We use Mean Pooling (Average of all frames)
        x = torch.mean(x, dim=1) # -> (Batch, d_model)
        
        # Step 5: Classification
        x = self.fc(x) # -> (Batch, num_classes)
        
        return x

# --- SANITY CHECK ---
# This block runs only if you execute this file directly.
# It tests if the model structure is correct.
if __name__ == "__main__":
    print("🧪 Testing Transformer Architecture...")
    
    # Configuration
    INPUT_DIM = 30  # 5 objects * 6 features
    NUM_CLASSES = 8
    BATCH_SIZE = 2
    SEQ_LEN = 30    # Looking at 30 frames (approx 1 second)
    
    # Initialize Model
    model = UnsafeNetTransformer(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    
    # Create Dummy Data (Random Noise)
    dummy_input = torch.rand(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    
    # Forward Pass
    try:
        output = model(dummy_input)
        print("\n✅ Test Passed!")
        print(f"Input Shape:  {dummy_input.shape}  (Batch, Time, Feats)")
        print(f"Output Shape: {output.shape}       (Batch, Classes)")
        print("-" * 30)
        print("The model is ready for training!")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")