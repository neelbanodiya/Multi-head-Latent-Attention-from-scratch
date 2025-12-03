"""Input: X (batch, seq_len, d_model)
    ↓
    ├─→ Query Path (standard)
    │   └─→ Linear(d_model → num_heads * d_head) → Split into heads
    │
    └─→ Key-Value Path (compressed)
        ├─→ Down-projection: Linear(d_model → d_latent)  ← COMPRESSION
        │   └─→ This gets cached! (C_kv)
        │
        └─→ Up-projection (separate for each head):
            ├─→ K: Linear(d_latent → d_head) for each head
            └─→ V: Linear(d_latent → d_head) for each head
    
Attention Computation:
    Score = (Q @ K^T) / sqrt(d_head)
    Attention_weights = softmax(Score)
    Output = Attention_weights @ V
    
Final: Concatenate all heads → Linear(num_heads * d_head → d_model)"""


"""INPUT: x (batch=2, seq_len=10, d_model=128)
│
├─── QUERY PATH (Standard) ───────────────────────────┐
│    1. Linear: x @ W_q → (2, 10, 128)                │
│    2. Reshape: → (2, 10, 4, 32)                     │
│    3. Transpose: → (2, 4, 10, 32)                   │
│    Result: Q (batch, heads, seq, d_head)            │
│                                                      │
└─── KEY-VALUE PATH (MLA Innovation) ─────────────────┤
     1. DOWN-PROJECT (COMPRESSION!):                  │
        x @ W_down_kv → C_kv (2, 10, 16)             │
        └─ 128 dims → 16 dims (8x compression!)      │
                                                      │
     2. CACHE CHECK:                                  │
        if cache exists: concat with past C_kv       │
        └─ This is what gets cached! Only 16 dims!   │
                                                      │
     3. UP-PROJECT (Per-Head):                        │
        For each head h:                              │
          K_h = C_kv @ W_up_k[h] → (2, 10, 32)       │
          V_h = C_kv @ W_up_v[h] → (2, 10, 32)       │
        Stack: K, V → (2, 4, 10, 32)                 │
                                                      │
ATTENTION COMPUTATION ──────────────────────────────┐ │
│    4. Scores = (Q @ K^T) / √32                    │ │
│       Shape: (2, 4, 10, 10)                       │ │
│                                                    │ │
│    5. Apply Causal Mask (if provided)             │ │
│       scores[i,j] = -inf where j > i              │ │
│                                                    │ │
│    6. Softmax(scores, dim=-1)                     │ │
│       attention_weights: (2, 4, 10, 10)           │ │
│       └─ Each row sums to 1 (probability)         │ │
│                                                    │ │
│    7. Dropout(attention_weights)                  │ │
│                                                    │ │
│    8. output = attention_weights @ V              │ │
│       Shape: (2, 4, 10, 32)                       │ │
│                                                    ◄─┘
COMBINE HEADS ────────────────────────────────────────┐
     9. Transpose: (2, 4, 10, 32) → (2, 10, 4, 32)   │
    10. Reshape: → (2, 10, 128)                      │
    11. Output proj: @ W_o → (2, 10, 128)            │
                                                      │
OUTPUT: (batch=2, seq_len=10, d_model=128) ──────────┘
CACHE: {'c_kv': (2, 10, 16)} ← Only this is stored!"""


"""
Multi-Head Latent Attention (MLA) Module
Step 1: Class structure and linear layer initialization
"""

import torch
import torch.nn as nn
import math


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA)
    
    Key innovation: Compresses Key and Value representations into a shared
    low-dimensional latent space, reducing memory usage for KV cache.
    
    Architecture:
        - Query: Standard projection (d_model → num_heads * d_head)
        - Key/Value: Compressed path
            1. Down-project: d_model → d_latent (COMPRESSION)
            2. Up-project: d_latent → d_head (per head)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_latent: int,
        dropout: float = 0.1
    ):
        """
        Initialize Multi-Head Latent Attention
        
        Args:
            d_model: Model dimension (e.g., 128)
            num_heads: Number of attention heads (e.g., 4)
            d_latent: Latent dimension for compression (e.g., 16)
            dropout: Dropout probability (e.g., 0.1)
        
        Example:
            mla = MultiHeadLatentAttention(d_model=128, num_heads=4, d_latent=16)
        """
        super().__init__()
        
        # Store configuration
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent
        self.dropout_prob = dropout
        
        # Calculate dimension per head
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_head = d_model // num_heads
        
        # ================================================================
        # QUERY PROJECTION (Standard Multi-Head Attention approach)
        # ================================================================
        # Single large projection that we'll later split into heads
        # Input: (batch, seq_len, d_model)
        # Output: (batch, seq_len, num_heads * d_head)
        self.W_q = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        
        # ================================================================
        # KEY-VALUE COMPRESSION PATH (MLA Innovation!)
        # ================================================================
        
        # Step 1: Down-projection (COMPRESSION)
        # This is the key memory-saving innovation
        # Compresses d_model → d_latent (shared for both K and V)
        # Input: (batch, seq_len, d_model)
        # Output: (batch, seq_len, d_latent)
        self.W_down_kv = nn.Linear(d_model, d_latent, bias=False)
        
        # Step 2: Up-projection for Keys (separate for each head)
        # Each head gets its own projection from latent space
        # Input per head: (batch, seq_len, d_latent)
        # Output per head: (batch, seq_len, d_head)
        self.W_up_k = nn.ModuleList([
            nn.Linear(d_latent, self.d_head, bias=False)
            for _ in range(num_heads)
        ])
        
        # Step 3: Up-projection for Values (separate for each head)
        # Similar to Keys, each head gets its own projection
        # Input per head: (batch, seq_len, d_latent)
        # Output per head: (batch, seq_len, d_head)
        self.W_up_v = nn.ModuleList([
            nn.Linear(d_latent, self.d_head, bias=False)
            for _ in range(num_heads)
        ])
        
        # ================================================================
        # OUTPUT PROJECTION
        # ================================================================
        # Combines all heads back to model dimension
        # Input: (batch, seq_len, num_heads * d_head)
        # Output: (batch, seq_len, d_model)
        self.W_o = nn.Linear(num_heads * self.d_head, d_model, bias=False)
        
        # ================================================================
        # DROPOUT
        # ================================================================
        # Applied to attention weights for regularization
        self.dropout = nn.Dropout(dropout)
        
        # ================================================================
        # INITIALIZE WEIGHTS
        # ================================================================
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights using Xavier/Glorot uniform initialization
        This helps with gradient flow during training
        """
        # Initialize query projection
        nn.init.xavier_uniform_(self.W_q.weight)
        
        # Initialize down-projection (compression layer)
        nn.init.xavier_uniform_(self.W_down_kv.weight)
        
        # Initialize up-projections for keys
        for layer in self.W_up_k:
            nn.init.xavier_uniform_(layer.weight)
        
        # Initialize up-projections for values
        for layer in self.W_up_v:
            nn.init.xavier_uniform_(layer.weight)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.W_o.weight)
    
    def forward(self, x, mask=None, use_cache=False, cache=None):
        """
        Forward pass of Multi-Head Latent Attention
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask (batch_size, seq_len, seq_len)
            use_cache: Whether to use KV caching (for inference)
            cache: Previous cache dict with 'c_kv' key
        
        Returns:
            output: Attention output (batch_size, seq_len, d_model)
            new_cache: Updated cache dict (if use_cache=True)
        """
        batch_size, seq_len, d_model = x.shape
        
        # ====================================================================
        # STEP 1: QUERY PROJECTION (Standard Multi-Head Attention)
        # ====================================================================
        # Project input to queries
        # Shape: (batch, seq_len, d_model) -> (batch, seq_len, num_heads * d_head)
        Q = self.W_q(x)
        
        # Reshape to separate heads
        # Shape: (batch, seq_len, num_heads * d_head) -> (batch, seq_len, num_heads, d_head)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head)
        
        # Transpose to get: (batch, num_heads, seq_len, d_head)
        # This puts heads dimension before sequence for easier batch matrix operations
        Q = Q.transpose(1, 2)
        
        # ====================================================================
        # STEP 2: KEY-VALUE COMPRESSION (MLA Innovation!)
        # ====================================================================
        # Down-project to latent space (COMPRESSION HAPPENS HERE!)
        # Shape: (batch, seq_len, d_model) -> (batch, seq_len, d_latent)
        C_kv = self.W_down_kv(x)
        
        # If using cache, concatenate with previous cached states
        if use_cache and cache is not None and 'c_kv' in cache:
            # Concatenate along sequence dimension
            # Previous: (batch, past_seq_len, d_latent)
            # Current:  (batch, curr_seq_len, d_latent)
            # Result:   (batch, past_seq_len + curr_seq_len, d_latent)
            C_kv = torch.cat([cache['c_kv'], C_kv], dim=1)
        
        # Get the actual sequence length (may be longer if using cache)
        kv_seq_len = C_kv.shape[1]
        
        # ====================================================================
        # STEP 3: UP-PROJECT TO KEYS AND VALUES (Per-Head)
        # ====================================================================
        # For each head, project from latent space to head-specific K and V
        # This is where we reconstruct K and V from the compressed representation
        
        K_heads = []
        V_heads = []
        
        for h in range(self.num_heads):
            # Project latent to Key for this head
            # Shape: (batch, kv_seq_len, d_latent) -> (batch, kv_seq_len, d_head)
            K_h = self.W_up_k[h](C_kv)
            K_heads.append(K_h)
            
            # Project latent to Value for this head
            # Shape: (batch, kv_seq_len, d_latent) -> (batch, kv_seq_len, d_head)
            V_h = self.W_up_v[h](C_kv)
            V_heads.append(V_h)
        
        # Stack all heads
        # Shape: list of (batch, kv_seq_len, d_head) -> (batch, num_heads, kv_seq_len, d_head)
        K = torch.stack(K_heads, dim=1)
        V = torch.stack(V_heads, dim=1)
        
        # ====================================================================
        # STEP 4: COMPUTE ATTENTION SCORES (Q @ K^T)
        # ====================================================================
        # Compute dot product attention scores
        # Q: (batch, num_heads, seq_len, d_head)
        # K: (batch, num_heads, kv_seq_len, d_head)
        # K^T: (batch, num_heads, d_head, kv_seq_len)
        # Result: (batch, num_heads, seq_len, kv_seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Scale by sqrt(d_head) to prevent softmax saturation
        scores = scores / math.sqrt(self.d_head)
        
        # ====================================================================
        # STEP 5: APPLY MASK (if provided)
        # ====================================================================
        if mask is not None:
            # Mask should be (batch, 1, seq_len, kv_seq_len) or broadcastable
            # Fill masked positions with -inf so softmax outputs 0
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # ====================================================================
        # STEP 6: SOFTMAX - Convert scores to attention weights
        # ====================================================================
        # Apply softmax along the last dimension (over key positions)
        # Shape: (batch, num_heads, seq_len, kv_seq_len)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply dropout for regularization
        attention_weights = self.dropout(attention_weights)
        
        # ====================================================================
        # STEP 7: WEIGHTED SUM with VALUES
        # ====================================================================
        # Multiply attention weights with values
        # attention_weights: (batch, num_heads, seq_len, kv_seq_len)
        # V: (batch, num_heads, kv_seq_len, d_head)
        # Result: (batch, num_heads, seq_len, d_head)
        output = torch.matmul(attention_weights, V)
        
        # ====================================================================
        # STEP 8: CONCATENATE HEADS
        # ====================================================================
        # Transpose back: (batch, num_heads, seq_len, d_head) -> (batch, seq_len, num_heads, d_head)
        output = output.transpose(1, 2)
        
        # Reshape to concatenate heads: (batch, seq_len, num_heads, d_head) -> (batch, seq_len, num_heads * d_head)
        output = output.contiguous().view(batch_size, seq_len, self.num_heads * self.d_head)
        
        # ====================================================================
        # STEP 9: OUTPUT PROJECTION
        # ====================================================================
        # Final linear transformation
        # Shape: (batch, seq_len, num_heads * d_head) -> (batch, seq_len, d_model)
        output = self.W_o(output)
        
        # ====================================================================
        # STEP 10: PREPARE CACHE (if requested)
        # ====================================================================
        new_cache = None
        if use_cache:
            # Store the compressed latent representation
            # This is much smaller than storing full K and V!
            new_cache = {'c_kv': C_kv}
        
        return output, new_cache

    def get_memory_stats(self, seq_len: int) -> dict:
        """
        Calculate memory usage for KV cache
        
        Args:
            seq_len: Sequence length
        
        Returns:
            Dictionary with memory statistics
        """
        # Standard attention cache size (per token)
        # Stores K and V separately for each head
        standard_kv_cache = self.num_heads * self.d_head * 2  # 2 for K and V
        
        # MLA cache size (per token)
        # Only stores compressed latent representation
        mla_cache = self.d_latent
        
        # Calculate total for sequence
        standard_total = standard_kv_cache * seq_len
        mla_total = mla_cache * seq_len
        
        # Compression ratio
        compression_ratio = standard_kv_cache / mla_cache
        memory_savings = (1 - mla_cache / standard_kv_cache) * 100
        
        return {
            'standard_cache_per_token': standard_kv_cache,
            'mla_cache_per_token': mla_cache,
            'standard_total_seq': standard_total,
            'mla_total_seq': mla_total,
            'compression_ratio': f"{compression_ratio:.2f}x",
            'memory_savings': f"{memory_savings:.1f}%"
        }
    
        def __repr__(self):
        
            return (f"MultiHeadLatentAttention(\n"
                f"  d_model={self.d_model},\n"
                f"  num_heads={self.num_heads},\n"
                f"  d_head={self.d_head},\n"
                f"  d_latent={self.d_latent},\n"
                f"  dropout={self.dropout_prob}\n"
                f")")

def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal (lower triangular) mask for autoregressive attention
    
    Position i can only attend to positions <= i (no future information)
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on (CPU or CUDA)
    
    Returns:
        Causal mask of shape (1, 1, seq_len, seq_len)
        - 1 means "can attend"
        - 0 means "cannot attend" (will be masked with -inf)
    
    Example:
        For seq_len=4:
        [[1, 0, 0, 0],   # Position 0 can only see itself
        [1, 1, 0, 0],   # Position 1 can see 0 and 1
        [1, 1, 1, 0],   # Position 2 can see 0, 1, and 2
        [1, 1, 1, 1]]   # Position 3 can see all
    """
    # Create lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    
    # Add batch and head dimensions: (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
    # This allows broadcasting across batch and heads
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    return mask


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

def test_mla_initialization():
    """Test that MLA initializes correctly with proper dimensions"""
    
    print("=" * 70)
    print("TESTING MLA INITIALIZATION")
    print("=" * 70)
    
    # Configuration (matching our project specs)
    config = {
        'd_model': 128,
        'num_heads': 4,
        'd_latent': 16,
        'dropout': 0.1
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create MLA module
    print("\nCreating MLA module...")
    mla = MultiHeadLatentAttention(**config)
    print("✓ MLA module created successfully!")
    
    
    # Test each linear layer exists and has correct shape
    print("\n" + "=" * 70)
    print("VERIFYING LAYER DIMENSIONS")
    print("=" * 70)
    
    # Query projection
    print(f"\n1. Query Projection (W_q):")
    print(f"   Expected: ({config['d_model']}, {config['num_heads'] * config['d_model'] // config['num_heads']})")
    print(f"   Actual:   {tuple(mla.W_q.weight.shape)}")
    assert mla.W_q.weight.shape == (config['num_heads'] * config['d_model'] // config['num_heads'], config['d_model'])
    print("   ✓ Correct!")
    
    # Down-projection
    print(f"\n2. Down-projection (W_down_kv) - COMPRESSION LAYER:")
    print(f"   Expected: ({config['d_latent']}, {config['d_model']})")
    print(f"   Actual:   {tuple(mla.W_down_kv.weight.shape)}")
    assert mla.W_down_kv.weight.shape == (config['d_latent'], config['d_model'])
    print("   ✓ Correct!")
    
    # Up-projection for Keys
    print(f"\n3. Up-projection for Keys (W_up_k) - {config['num_heads']} heads:")
    for i, layer in enumerate(mla.W_up_k):
        expected_shape = (config['d_model'] // config['num_heads'], config['d_latent'])
        actual_shape = tuple(layer.weight.shape)
        print(f"   Head {i}: Expected {expected_shape}, Actual {actual_shape}")
        assert layer.weight.shape == expected_shape
    print("   ✓ All correct!")
    
    # Up-projection for Values
    print(f"\n4. Up-projection for Values (W_up_v) - {config['num_heads']} heads:")
    for i, layer in enumerate(mla.W_up_v):
        expected_shape = (config['d_model'] // config['num_heads'], config['d_latent'])
        actual_shape = tuple(layer.weight.shape)
        print(f"   Head {i}: Expected {expected_shape}, Actual {actual_shape}")
        assert layer.weight.shape == expected_shape
    print("   ✓ All correct!")
    
    # Output projection
    print(f"\n5. Output Projection (W_o):")
    print(f"   Expected: ({config['d_model']}, {config['num_heads'] * config['d_model'] // config['num_heads']})")
    print(f"   Actual:   {tuple(mla.W_o.weight.shape)}")
    assert mla.W_o.weight.shape == (config['d_model'], config['num_heads'] * config['d_model'] // config['num_heads'])
    print("   ✓ Correct!")
    
    # Memory statistics
    print("\n" + "=" * 70)
    print("MEMORY SAVINGS ANALYSIS")
    print("=" * 70)
    
    seq_len = 256
    stats = mla.get_memory_stats(seq_len)
    
    print(f"\nFor sequence length = {seq_len}:")
    print(f"\n  Standard Attention KV Cache:")
    print(f"    Per token:  {stats['standard_cache_per_token']} values")
    print(f"    Total:      {stats['standard_total_seq']:,} values")
    
    print(f"\n  MLA KV Cache:")
    print(f"    Per token:  {stats['mla_cache_per_token']} values")
    print(f"    Total:      {stats['mla_total_seq']:,} values")
    
    print(f"\n  Compression: {stats['compression_ratio']}")
    print(f"  Memory Savings: {stats['memory_savings']}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    
    return mla



def test_forward_pass():
    """Test the forward pass with sample input"""
    
    print("\nTest 1: Basic Forward Pass")
    print("-" * 70)
    
    # Create MLA module
    mla = MultiHeadLatentAttention(d_model=128, num_heads=4, d_latent=16, dropout=0.1)
    mla.eval()  # Set to eval mode (disables dropout)
    
    # Create sample input
    batch_size = 2
    seq_len = 10
    d_model = 128
    
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    
    # Forward pass without mask
    output, cache = mla(x, use_cache=False)
    
    print(f"Output shape: {output.shape}")
    print(f"Cache: {cache}")
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch!"
    print("✓ Output shape correct!")
    
    # Forward pass with causal mask
    print("\nTest 2: Forward Pass with Causal Mask")
    print("-" * 70)
    
    mask = create_causal_mask(seq_len)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask (first 5x5):\n{mask[0, 0, :5, :5]}")
    
    output_masked, _ = mla(x, mask=mask, use_cache=False)
    print(f"Output shape with mask: {output_masked.shape}")
    assert output_masked.shape == (batch_size, seq_len, d_model)
    print("✓ Masked output shape correct!")
    
    # Outputs should be different (mask affects attention)
    assert not torch.allclose(output, output_masked), "Mask should change output!"
    print("✓ Mask affects output as expected!")
    
    print("\n✅ Forward pass tests passed!")


def test_kv_cache():
    """Test KV caching mechanism for efficient inference"""
    
    print("\nTest 3: KV Cache Functionality")
    print("-" * 70)
    
    mla = MultiHeadLatentAttention(d_model=128, num_heads=4, d_latent=16, dropout=0.0)
    mla.eval()
    
    batch_size = 1
    d_model = 128
    
    # Method 1: Process full sequence at once
    full_seq_len = 8
    x_full = torch.randn(batch_size, full_seq_len, d_model)
    mask_full = create_causal_mask(full_seq_len)
    
    output_full, _ = mla(x_full, mask=mask_full, use_cache=False)
    print(f"Full sequence output shape: {output_full.shape}")
    
    # Method 2: Process incrementally with cache (simulating inference)
    cache = None
    outputs_incremental = []
    
    for i in range(full_seq_len):
        # Get next token
        x_new = x_full[:, i:i+1, :]  # (batch, 1, d_model)
        
        # Create mask for this position
        if cache is not None:
            past_len = cache['c_kv'].shape[1]
            # Current token can attend to all past tokens and itself
            mask = torch.ones(1, 1, 1, past_len + 1)
        else:
            mask = torch.ones(1, 1, 1, 1)
        
        # Forward pass with cache
        output_new, cache = mla(x_new, mask=mask, use_cache=True, cache=cache)
        outputs_incremental.append(output_new)
        
        print(f"  Step {i+1}: Processed token {i}, cache size: {cache['c_kv'].shape[1]}")
    
    # Concatenate incremental outputs
    output_incremental = torch.cat(outputs_incremental, dim=1)
    print(f"\nIncremental output shape: {output_incremental.shape}")
    
    # Compare outputs (should be very close, small numerical differences okay)
    max_diff = torch.max(torch.abs(output_full - output_incremental)).item()
    print(f"\nMax difference between full and incremental: {max_diff:.6f}")
    
    # Check if outputs are close (tolerance for floating point differences)
    assert torch.allclose(output_full, output_incremental, atol=1e-5), "Cache outputs don't match!"
    print("✓ Cached inference matches full sequence processing!")
    
    # Verify cache size
    final_cache_size = cache['c_kv'].shape
    print(f"\nFinal cache shape: {final_cache_size}")
    print(f"Cache stores: {final_cache_size[1]} tokens × {final_cache_size[2]} latent dims")
    
    standard_cache = full_seq_len * (128)  # Would store full K,V
    mla_cache = full_seq_len * 16  # Only stores compressed
    print(f"\nMemory comparison:")
    print(f"  Standard cache: {standard_cache} values")
    print(f"  MLA cache: {mla_cache} values")
    print(f"  Savings: {(1 - mla_cache/standard_cache)*100:.1f}%")
    
    print("\n✅ KV cache tests passed!")


def test_causality():
    """Test that causality is properly enforced (no information leakage from future)"""
    
    print("\nTest 4: Causality Verification")
    print("-" * 70)
    
    mla = MultiHeadLatentAttention(d_model=128, num_heads=4, d_latent=16, dropout=0.0)
    mla.eval()
    
    batch_size = 1
    seq_len = 10
    d_model = 128
    
    # Create input sequence
    x = torch.randn(batch_size, seq_len, d_model)
    mask = create_causal_mask(seq_len)
    
    # Get output
    output1, _ = mla(x, mask=mask, use_cache=False)
    
    # Modify a future token (position 7)
    x_modified = x.clone()
    x_modified[:, 7, :] = torch.randn(batch_size, d_model) * 10  # Large change
    
    # Get output with modified future token
    output2, _ = mla(x_modified, mask=mask, use_cache=False)
    
    # Check positions before modification (0-6)
    for pos in range(7):
        diff = torch.max(torch.abs(output1[:, pos, :] - output2[:, pos, :])).item()
        print(f"  Position {pos}: max diff = {diff:.6f}")
        assert diff < 1e-5, f"Position {pos} was affected by future token!"
    
    print("✓ Positions 0-6 unaffected by change at position 7 (causality maintained)")
    
    # Check positions after modification (7-9) - these SHOULD change
    for pos in range(7, seq_len):
        diff = torch.max(torch.abs(output1[:, pos, :] - output2[:, pos, :])).item()
        print(f"  Position {pos}: max diff = {diff:.6f}")
        assert diff > 1e-3, f"Position {pos} should be affected by change at position 7!"
    
    print("✓ Positions 7-9 correctly affected by change at position 7")
    
    print("\n✅ Causality tests passed!")


if __name__ == "__main__":
    # Run initialization tests
    print("PART 1: INITIALIZATION TESTS")
    print("=" * 70)
    mla = test_mla_initialization()
    
    # Run forward pass tests
    print("\n\n")
    print("PART 2: FORWARD PASS TESTS")
    print("=" * 70)
    test_forward_pass()
    
    print("\n\n")
    print("PART 3: CACHE TESTS")
    print("=" * 70)
    test_kv_cache()
    
    print("\n\n")
    print("PART 4: CAUSALITY TESTS")
    print("=" * 70)
    test_causality()
    
    print("\n\n🎉 ALL TESTS PASSED!")
    print("✅ MLA module is ready for training!")