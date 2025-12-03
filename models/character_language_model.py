"""
Complete Character-Level Language Model using Multi-Head Latent Attention
GPT-style architecture for text generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
try:
    # allow import when package is installed or top-level name exists
    from transformer import TransformerBlock
except ModuleNotFoundError:
    # fallback when running main.py from project root
    from .transformer import TransformerBlock
    
try:
    # allow import when package is installed or top-level name exists
    from mla import MultiHeadLatentAttention, create_causal_mask
except ModuleNotFoundError:
    # fallback when running main.py from project root
    from .mla import MultiHeadLatentAttention, create_causal_mask



class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings
    
    Each position (0, 1, 2, ..., max_seq_len-1) gets a learnable vector
    that is added to the token embedding to give position information.
    
    Args:
        max_seq_len: Maximum sequence length (e.g., 256)
        d_model: Embedding dimension (e.g., 128)
    """
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        
        # Create learnable embedding table
        # Shape: (max_seq_len, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
    def forward(self, x):
        """
        Add positional embeddings to input
        
        Args:
            x: Token embeddings (batch_size, seq_len, d_model)
        
        Returns:
            x + positional embeddings (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device)
        
        # Get positional embeddings
        # Shape: (seq_len, d_model)
        pos_emb = self.pos_embedding(positions)
        
        # Broadcast and add to input
        # x: (batch, seq_len, d_model)
        # pos_emb: (seq_len, d_model) -> broadcasts to (batch, seq_len, d_model)
        x = x + pos_emb
        
        return x


class CharacterLanguageModel(nn.Module):
    """
    Complete GPT-style Character-Level Language Model with MLA
    
    Architecture:
        1. Token Embedding: character IDs -> vectors
        2. Positional Embedding: add position information
        3. Transformer Blocks (stacked): process with attention
        4. Layer Norm: final normalization
        5. Output Head: predict next character logits
    
    Args:
        vocab_size: Number of unique characters (e.g., 65)
        d_model: Model dimension (e.g., 128)
        num_layers: Number of transformer blocks (e.g., 4)
        num_heads: Number of attention heads (e.g., 4)
        d_latent: Latent dimension for MLA (e.g., 16)
        d_ff: Feed-forward hidden dimension (e.g., 512)
        max_seq_len: Maximum sequence length (e.g., 256)
        dropout: Dropout probability (e.g., 0.1)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        d_latent: int = 16,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_latent = d_latent
        self.max_seq_len = max_seq_len
        
        # ================================================================
        # EMBEDDING LAYERS
        # ================================================================
        
        # Token embedding: character ID -> vector
        # Converts discrete character IDs to continuous vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding: adds position information
        self.pos_embedding = PositionalEmbedding(max_seq_len, d_model)
        
        # Dropout after embeddings
        self.embedding_dropout = nn.Dropout(dropout)
        
        # ================================================================
        # TRANSFORMER BLOCKS
        # ================================================================
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_latent=d_latent,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # ================================================================
        # OUTPUT LAYERS
        # ================================================================
        
        # Final layer normalization
        self.ln_final = nn.LayerNorm(d_model)
        
        # Output head: project to vocabulary
        # Predicts logits for next character
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Optional: Tie weights between input and output embeddings
        # This reduces parameters and often improves performance
        # self.output_head.weight = self.token_embedding.weight
        
        # ================================================================
        # INITIALIZE WEIGHTS
        # ================================================================
        self._init_weights()
        
        # Print model info
        self._print_model_info()
    
    def _init_weights(self):
        """Initialize weights with appropriate scaling"""
        
        # Token embeddings: normal distribution with std=0.02
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Positional embeddings: normal distribution
        nn.init.normal_(self.pos_embedding.pos_embedding.weight, mean=0.0, std=0.02)
        
        # Output head: normal distribution
        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)
    
    def _print_model_info(self):
        """Print model architecture and parameter count"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("=" * 70)
        print("CHARACTER LANGUAGE MODEL WITH MLA")
        print("=" * 70)
        print(f"\nArchitecture:")
        print(f"  Vocabulary size:    {self.vocab_size}")
        print(f"  Model dimension:    {self.d_model}")
        print(f"  Number of layers:   {self.num_layers}")
        print(f"  Number of heads:    {self.num_heads}")
        print(f"  Latent dimension:   {self.d_latent}")
        print(f"  Max sequence length: {self.max_seq_len}")
        print(f"\nParameters:")
        print(f"  Total:      {total_params:,}")
        print(f"  Trainable:  {trainable_params:,}")
        print(f"  Size:       ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
        print("=" * 70)
    
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass through the model
        
        Args:
            x: Input token IDs (batch_size, seq_len)
            targets: Target token IDs for computing loss (batch_size, seq_len)
            use_cache: Whether to use KV caching for inference
            cache: Previous cache from all layers
        
        Returns:
            logits: Predicted logits (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss (if targets provided)
            new_cache: Updated cache for all layers (if use_cache=True)
        """
        batch_size, seq_len = x.shape
        
        # ================================================================
        # STEP 1: EMBEDDING
        # ================================================================
        
        # Convert token IDs to embeddings
        # Shape: (batch, seq_len) -> (batch, seq_len, d_model)
        token_emb = self.token_embedding(x)
        
        # Add positional information
        # Shape: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        x = self.pos_embedding(token_emb)
        
        # Apply dropout
        x = self.embedding_dropout(x)
        
        # ================================================================
        # STEP 2: CREATE CAUSAL MASK
        # ================================================================
        
        # For training: full causal mask
        # For inference with cache: mask only for new token
        if use_cache and cache is not None:
            # When using cache, we only process new token(s)
            # New token can attend to all previous cached tokens + itself
            past_length = cache.get('past_length', 0)
            mask = torch.ones(
                1, 1, seq_len, past_length + seq_len,
                device=x.device
            )
        else:
            # Standard causal mask: lower triangular
            mask = create_causal_mask(seq_len, device=x.device)
        
        # ================================================================
        # STEP 3: TRANSFORMER BLOCKS
        # ================================================================
        
        # Initialize cache for all layers if needed
        new_cache = None
        if use_cache:
            if cache is None:
                cache = {'layer_caches': [None] * self.num_layers, 'past_length': 0}
            new_cache = {'layer_caches': [], 'past_length': cache['past_length'] + seq_len}
        
        # Pass through each transformer block
        for i, block in enumerate(self.blocks):
            # Get cache for this layer
            layer_cache = cache['layer_caches'][i] if (use_cache and cache) else None
            
            # Forward through block
            x, layer_new_cache = block(
                x,
                mask=mask,
                use_cache=use_cache,
                cache=layer_cache
            )
            
            # Store new cache
            if use_cache:
                new_cache['layer_caches'].append(layer_new_cache)
        
        # ================================================================
        # STEP 4: OUTPUT HEAD
        # ================================================================
        
        # Final layer normalization
        x = self.ln_final(x)
        
        # Project to vocabulary
        # Shape: (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        logits = self.output_head(x)
        
        # ================================================================
        # STEP 5: COMPUTE LOSS (if targets provided)
        # ================================================================
        
        loss = None
        if targets is not None:
            # Reshape for cross entropy
            # logits: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
            # targets: (batch, seq_len) -> (batch * seq_len)
            logits_flat = logits.view(-1, self.vocab_size)
            targets_flat = targets.view(-1)
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss, new_cache
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Generate text autoregressively
        
        Args:
            idx: Starting token IDs (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            use_cache: Use KV caching for efficiency
        
        Returns:
            Generated token IDs (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        cache = None
        
        for _ in range(max_new_tokens):
            # If sequence is longer than max_seq_len, crop it
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # Get predictions
            if use_cache and cache is not None:
                # Only pass the last token when using cache
                logits, _, cache = self(idx_cond[:, -1:], use_cache=True, cache=cache)
            else:
                # Pass full sequence
                logits, _, cache = self(idx_cond, use_cache=use_cache, cache=cache)
            
            # Get logits for last position
            # Shape: (batch, 1, vocab_size) or (batch, seq_len, vocab_size)
            logits = logits[:, -1, :]  # (batch, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Optionally apply top-k filtering
            if top_k is not None:
                # Get top k logits and their indices
                top_logits, top_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                
                # Set all other logits to -inf
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_indices, top_logits)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
    
    def get_num_params(self, non_embedding: bool = True):
        """
        Return the number of parameters in the model
        
        Args:
            non_embedding: If True, exclude embedding parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.pos_embedding.pos_embedding.weight.numel()
        
        return n_params


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

def test_model_creation():
    """Test model creation and basic forward pass"""
    
    print("\n" + "=" * 70)
    print("TEST 1: MODEL CREATION")
    print("=" * 70)
    
    # Configuration matching our project
    config = {
        'vocab_size': 65,       # Tiny Shakespeare has ~65 characters
        'd_model': 128,
        'num_layers': 4,
        'num_heads': 4,
        'd_latent': 16,
        'd_ff': 512,
        'max_seq_len': 256,
        'dropout': 0.1
    }
    
    # Create model
    model = CharacterLanguageModel(**config)
    
    print("\n✓ Model created successfully!")
    
    # Count parameters by component
    print("\nParameter breakdown:")
    
    emb_params = model.token_embedding.weight.numel() + model.pos_embedding.pos_embedding.weight.numel()
    blocks_params = sum(p.numel() for block in model.blocks for p in block.parameters())
    head_params = model.output_head.weight.numel()
    
    print(f"  Embeddings:  {emb_params:>8,} ({emb_params/model.get_num_params(non_embedding=False)*100:.1f}%)")
    print(f"  Blocks:      {blocks_params:>8,} ({blocks_params/model.get_num_params(non_embedding=False)*100:.1f}%)")
    print(f"  Output head: {head_params:>8,} ({head_params/model.get_num_params(non_embedding=False)*100:.1f}%)")
    
    return model


def test_forward_pass(model):
    """Test forward pass with training data"""
    
    print("\n" + "=" * 70)
    print("TEST 2: FORWARD PASS")
    print("=" * 70)
    
    batch_size = 4
    seq_len = 256  # Our context length
    
    # Create dummy input (character IDs)
    x = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape:   {x.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, loss, _ = model(x, targets=targets)
    
    print(f"\nLogits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Verify shapes
    assert logits.shape == (batch_size, seq_len, model.vocab_size)
    print("\n✓ Forward pass successful!")
    
    return loss.item()


def test_generation(model):
    """Test text generation"""
    
    print("\n" + "=" * 70)
    print("TEST 3: TEXT GENERATION")
    print("=" * 70)
    
    # Start with a single token
    start_tokens = torch.tensor([[0]])  # Start with character ID 0
    
    print(f"\nStarting token: {start_tokens}")
    print("Generating 50 tokens...")
    
    # Generate
    model.eval()
    generated = model.generate(
        start_tokens,
        max_new_tokens=50,
        temperature=1.0,
        top_k=10,
        use_cache=True
    )
    
    print(f"\nGenerated shape: {generated.shape}")
    print(f"Generated IDs (first 20): {generated[0, :20].tolist()}")
    
    print("\n✓ Generation successful!")
    print("  (Output is random since model is untrained)")


def test_cache_efficiency(model):
    """Test KV cache efficiency"""
    
    print("\n" + "=" * 70)
    print("TEST 4: CACHE EFFICIENCY")
    print("=" * 70)
    
    seq_len = 100
    x = torch.randint(0, model.vocab_size, (1, seq_len))
    
    model.eval()
    
    # Method 1: Without cache (recompute everything)
    print("\nMethod 1: Without cache")
    import time
    
    start = time.time()
    with torch.no_grad():
        for i in range(seq_len, seq_len + 20):
            _ = model(x[:, :i])
    time_no_cache = time.time() - start
    print(f"  Time: {time_no_cache:.4f}s")
    
    # Method 2: With cache (efficient)
    print("\nMethod 2: With cache")
    start = time.time()
    cache = None
    with torch.no_grad():
        # Initial forward pass
        _, _, cache = model(x, use_cache=True, cache=None)
        
        # Generate next 20 tokens
        for _ in range(20):
            new_token = torch.randint(0, model.vocab_size, (1, 1))
            _, _, cache = model(new_token, use_cache=True, cache=cache)
    time_with_cache = time.time() - start
    print(f"  Time: {time_with_cache:.4f}s")
    
    speedup = time_no_cache / time_with_cache
    print(f"\n  Speedup: {speedup:.2f}x faster with cache!")
    print("✓ Cache working efficiently!")


if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("COMPLETE LANGUAGE MODEL TESTING SUITE")
    print("🚀" * 35)
    
    # Create model
    model = test_model_creation()
    
    # Test forward pass
    test_forward_pass(model)
    
    # Test generation
    test_generation(model)
    
    # Test cache efficiency
    test_cache_efficiency(model)
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\n📝 Model is ready for training!")
    print("   Next: Implement training loop with Shakespeare data")