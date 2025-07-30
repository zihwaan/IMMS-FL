import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(attention_output)
        
        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

class AnomalyTransformer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_len: int = 100,
                 dropout: float = 0.1,
                 num_classes: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # Reconstruction head (for pretraining)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, input_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_attention=False, return_reconstruction=False):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        x = self.dropout(x)
        
        # Pass through encoder layers
        attention_weights = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x)
            if return_attention:
                attention_weights.append(attn_weights)
        
        # For anomaly detection: use last time step
        last_hidden = x[:, -1, :]  # (batch_size, d_model)
        detection_output = self.detection_head(last_hidden)
        
        outputs = {'logits': detection_output}
        
        if return_reconstruction:
            # Reconstruction for the entire sequence
            reconstruction_output = self.reconstruction_head(x)  # (batch_size, seq_len, input_dim)
            outputs['reconstruction'] = reconstruction_output
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
            
        return outputs
    
    def get_encoder_params(self):
        """Get encoder parameters (for freezing)"""
        encoder_params = []
        encoder_params.extend(list(self.input_projection.parameters()))
        encoder_params.extend(list(self.pos_encoding.parameters()))
        for layer in self.encoder_layers:
            encoder_params.extend(list(layer.parameters()))
        return encoder_params
    
    def get_detection_head_params(self):
        """Get detection head parameters (for fine-tuning when encoder is frozen)"""
        return list(self.detection_head.parameters())
    
    def freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.get_encoder_params():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.get_encoder_params():
            param.requires_grad = True

class AnomalyLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, outputs, targets, inputs=None):
        logits = outputs['logits']
        
        # Classification loss
        classification_loss = self.ce_loss(logits, targets.squeeze())
        
        total_loss = classification_loss
        
        # Reconstruction loss (if available)
        if 'reconstruction' in outputs and inputs is not None:
            reconstruction_loss = self.mse_loss(outputs['reconstruction'], inputs)
            total_loss = self.alpha * classification_loss + (1 - self.alpha) * reconstruction_loss
        
        return total_loss

def create_model(input_dim: int, 
                model_config: dict = None,
                frozen_encoder: bool = False) -> AnomalyTransformer:
    """Create Anomaly Transformer model with specified configuration"""
    
    default_config = {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 1024,
        'max_seq_len': 100,
        'dropout': 0.1,
        'num_classes': 2
    }
    
    if model_config:
        default_config.update(model_config)
    
    model = AnomalyTransformer(input_dim=input_dim, **default_config)
    
    if frozen_encoder:
        model.freeze_encoder()
    
    return model

def pretrain_encoder(model: AnomalyTransformer, 
                    dataloader,
                    device: torch.device,
                    epochs: int = 10,
                    lr: float = 1e-4) -> AnomalyTransformer:
    """Pretrain encoder on normal data using reconstruction task"""
    
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch_x, return_reconstruction=True)
            reconstruction = outputs['reconstruction']
            
            loss = criterion(reconstruction, batch_x)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Pretrain Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model

if __name__ == "__main__":
    # Test the model
    batch_size = 32
    seq_len = 60
    input_dim = 123  # WADI features
    
    # Create model
    model = create_model(input_dim)
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    outputs = model(x, return_attention=True, return_reconstruction=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Reconstruction shape: {outputs['reconstruction'].shape}")
    print(f"Number of attention layers: {len(outputs['attention_weights'])}")
    
    # Test frozen encoder
    frozen_model = create_model(input_dim, frozen_encoder=True)
    print(f"Frozen encoder parameters: {sum(p.numel() for p in frozen_model.get_encoder_params() if not p.requires_grad)}")
    print(f"Trainable parameters: {sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)}")