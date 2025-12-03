"""
Transformer Block using Multi-Head Latent Attention
Combines MLA with Feed-Forward Network, Layer Norms, and Residual Connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    # allow import when package is installed or top-level name exists
    from mla import MultiHeadLatentAttention, create_causal_mask
except ModuleNotFoundError:
    # fallback when running main.py from project root
    from .mla import MultiHeadLatentAttention, create_causal_mask


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Architecture: Linear -> GELU -> Dropout -> Linear -> Dropout
    Typically expands dimension by 4x in the middle layer
    
    Args:
        d_model: Model dimension (e.g., 128)
        d_ff: Hidden dimension (e.g., 512 = 4 * 128)
        dropout: Dropout probability
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # First linear layer: expand dimension
        # d_model -> d_ff (e.g., 128 -> 512)
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Second linear layer: project back
        # d_ff -> d_model (e.g., 512 -> 128)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass of feed-forward network
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # First linear + activation
        # Shape: (batch, seq_len, d_model) -> (batch, seq_len, d_ff)
        x = self.linear1(x)
        x = F.gelu(x)  # GELU activation (smoother than ReLU)
        x = self.dropout(x)
        
        # Second linear
        # Shape: (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer Block with Multi-Head Latent Attention
    
    Architecture (Pre-LN variant - more stable training):
        x -> LayerNorm -> MLA -> Add(x) -> LayerNorm -> FFN -> Add -> output
        
    This is called "Pre-Layer Normalization" because we apply LayerNorm
    BEFORE the sub-layers (attention and FFN) rather than after.
    
    Args:
        d_model: Model dimension (e.g., 128)
        num_heads: Number of attention heads (e.g., 4)
        d_latent: Latent dimension for MLA compression (e.g., 16)
        d_ff: Feed-forward hidden dimension (e.g., 512)
        dropout: Dropout probability
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_latent: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # ================================================================
        # ATTENTION SUB-LAYER
        # ================================================================
        # Layer normalization before attention
        self.ln1 = nn.LayerNorm(d_model)
        
        # Multi-Head Latent Attention
        self.attention = MultiHeadLatentAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_latent=d_latent,
            dropout=dropout
        )
        
        # ================================================================
        # FEED-FORWARD SUB-LAYER
        # ================================================================
        # Layer normalization before feed-forward
        self.ln2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, mask=None, use_cache=False, cache=None):
        """
        Forward pass through transformer block
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            use_cache: Whether to use KV caching
            cache: Previous cache from attention layer
        
        Returns:
            output: Output tensor (batch, seq_len, d_model)
            new_cache: Updated cache (if use_cache=True)
        """
        # ================================================================
        # ATTENTION SUB-LAYER with Pre-LN and Residual Connection
        # ================================================================
        # Pre-Layer Normalization
        x_norm = self.ln1(x)
        
        # Multi-Head Latent Attention
        attn_output, new_cache = self.attention(
            x_norm, 
            mask=mask, 
            use_cache=use_cache, 
            cache=cache
        )
        
        # Residual connection (skip connection)
        # Add original input to attention output
        x = x + attn_output
        
        # ================================================================
        # FEED-FORWARD SUB-LAYER with Pre-LN and Residual Connection
        # ================================================================
        # Pre-Layer Normalization
        x_norm = self.ln2(x)
        
        # Feed-forward network
        ffn_output = self.ffn(x_norm)
        
        # Residual connection
        x = x + ffn_output
        
        return x, new_cache
    
    def __repr__(self):
        return (f"TransformerBlock(\n"
                f"  d_model={self.d_model},\n"
                f"  attention={self.attention},\n"
                f"  ffn={self.ffn}\n"
                f")")


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

def test_feedforward():
    """Test the feed-forward network"""
    print("=" * 70)
    print("TEST 1: FEED-FORWARD NETWORK")
    print("=" * 70)
    
    d_model = 128
    d_ff = 512  # 4x expansion
    batch_size = 2
    seq_len = 10
    
    # Create FFN
    ffn = FeedForward(d_model, d_ff, dropout=0.1)
    print(f"\nFeed-Forward Network:")
    print(f"  Input dim:  {d_model}")
    print(f"  Hidden dim: {d_ff}")
    print(f"  Output dim: {d_model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in ffn.parameters())
    print(f"\n  Total parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape:  {x.shape}")
    
    ffn.eval()
    output = ffn(x)
    print(f"Output shape: {output.shape}")
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len, d_model)
    print("\n✓ Feed-forward test passed!")


def test_transformer_block():
    """Test the complete transformer block"""
    print("\n" + "=" * 70)
    print("TEST 2: TRANSFORMER BLOCK")
    print("=" * 70)
    
    # Configuration
    config = {
        'd_model': 128,
        'num_heads': 4,
        'd_latent': 16,
        'd_ff': 512,
        'dropout': 0.1
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create transformer block
    block = TransformerBlock(**config)
    block.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in block.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Breakdown by component
    print("\nParameter breakdown:")
    attn_params = sum(p.numel() for p in block.attention.parameters())
    ffn_params = sum(p.numel() for p in block.ffn.parameters())
    ln_params = sum(p.numel() for p in block.ln1.parameters()) + sum(p.numel() for p in block.ln2.parameters())
    
    print(f"  Attention: {attn_params:,} ({attn_params/total_params*100:.1f}%)")
    print(f"  FFN:       {ffn_params:,} ({ffn_params/total_params*100:.1f}%)")
    print(f"  LayerNorm: {ln_params:,} ({ln_params/total_params*100:.1f}%)")
    
    # Test forward pass
    print("\n" + "-" * 70)
    print("Forward Pass Test")
    print("-" * 70)
    
    batch_size = 2
    seq_len = 256  # Our actual context length!
    d_model = 128
    
    x = torch.randn(batch_size, seq_len, d_model)
    mask = create_causal_mask(seq_len)
    
    print(f"\nInput shape:  {x.shape}")
    print(f"Mask shape:   {mask.shape}")
    
    # Forward pass without cache
    output, cache = block(x, mask=mask, use_cache=False)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Cache:        {cache}")
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len, d_model)
    print("\n✓ Output shape correct!")
    
    # Test with cache
    print("\n" + "-" * 70)
    print("Cache Test")
    print("-" * 70)
    
    # Process first 5 tokens
    x_init = x[:, :5, :]
    mask_init = create_causal_mask(5)
    output_init, cache = block(x_init, mask=mask_init, use_cache=True)
    
    print(f"Initial tokens:     {x_init.shape}")
    print(f"Initial cache size: {cache['c_kv'].shape}")
    
    # Process next token
    x_new = x[:, 5:6, :]
    mask_new = torch.ones(1, 1, 1, 6)  # Can attend to all previous + self
    output_new, cache = block(x_new, mask=mask_new, use_cache=True, cache=cache)
    
    print(f"New token:          {x_new.shape}")
    print(f"Updated cache size: {cache['c_kv'].shape}")
    
    assert cache['c_kv'].shape[1] == 6  # Should have 6 tokens cached
    print("\n✓ Cache test passed!")


def test_residual_connections():
    """Test that residual connections work properly"""
    print("\n" + "=" * 70)
    print("TEST 3: RESIDUAL CONNECTIONS")
    print("=" * 70)
    
    config = {
        'd_model': 128,
        'num_heads': 4,
        'd_latent': 16,
        'd_ff': 512,
        'dropout': 0.0  # Disable dropout for this test
    }
    
    block = TransformerBlock(**config)
    block.eval()
    
    # Create input
    batch_size = 1
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config['d_model'])
    
    # Get output
    output, _ = block(x, use_cache=False)
    
    # Output should be different from input (not just pass-through)
    diff = torch.abs(output - x).mean().item()
    print(f"\nMean difference between input and output: {diff:.4f}")
    assert diff > 0.01, "Output should be modified by the layers!"
    print("✓ Residual connections working (output differs from input)")
    
    # But output should still be somewhat related to input (residual helps)
    correlation = torch.corrcoef(torch.stack([
        x.flatten(),
        output.flatten()
    ]))[0, 1].item()
    print(f"Correlation between input and output: {correlation:.4f}")
    assert correlation > 0.3, "Output should still correlate with input due to residuals!"
    print("✓ Residual connections preserve information from input")


def test_layer_norm():
    """Test layer normalization behavior"""
    print("\n" + "=" * 70)
    print("TEST 4: LAYER NORMALIZATION")
    print("=" * 70)
    
    d_model = 128
    ln = nn.LayerNorm(d_model)
    
    # Create input with different scales
    x = torch.randn(2, 10, d_model) * 100  # Large scale
    
    print(f"\nInput statistics:")
    print(f"  Mean: {x.mean().item():.2f}")
    print(f"  Std:  {x.std().item():.2f}")
    
    # Apply layer norm
    x_norm = ln(x)
    
    print(f"\nAfter LayerNorm:")
    print(f"  Mean: {x_norm.mean().item():.6f}")
    print(f"  Std:  {x_norm.std().item():.6f}")
    
    # LayerNorm should normalize to mean≈0, std≈1
    assert abs(x_norm.mean().item()) < 0.01, "Mean should be close to 0"
    assert abs(x_norm.std().item() - 1.0) < 0.1, "Std should be close to 1"
    
    print("\n✓ Layer normalization working correctly!")


if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("TRANSFORMER BLOCK TESTING SUITE")
    print("🚀" * 35 + "\n")
    
    # Run all tests
    test_feedforward()
    test_transformer_block()
    test_residual_connections()
    test_layer_norm()
    
    print("\n" + "=" * 70)
    print("✅ ALL TRANSFORMER BLOCK TESTS PASSED!")
    print("=" * 70)
    print("\n📝 Next step: Build the complete language model!")
    print("   (Stack multiple transformer blocks + embeddings + output head)")